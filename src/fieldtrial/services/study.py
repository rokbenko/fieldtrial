"""Study lifecycle: create, validate, plan, lock, amend, unblind, and report status."""

import difflib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

import fieldtrial
from fieldtrial.design import (
    LoadedStudy,
    Slot,
    StudySpec,
    StudyValidationError,
    blind_codes,
    build_schedule,
    conditions,
    design_hash,
    load_study,
)
from fieldtrial.design.blinding import code_pool
from fieldtrial.services._context import ServiceError, StudyContext, open_study, resolve_folder
from fieldtrial.services.events import append_event, list_events
from fieldtrial.stats import mde
from fieldtrial.store import models as m
from fieldtrial.store.db import DB_FILENAME, open_database, session_factory
from fieldtrial.store.models import utcnow
from fieldtrial.templates import template_text

SYSTEM_ACTOR = "fieldtrial"


# --- init / validate / plan -----------------------------------------------------------------


def init_study(folder: str | Path, *, template: str = "basic", name: str | None = None) -> Path:
    """Create a study folder: ``study.yaml`` from a template, plus ``media/`` and ``reports/``."""
    root = resolve_folder(folder)
    study_file = root / "study.yaml"
    if study_file.exists():
        raise ServiceError(f"{study_file} already exists")
    study_name = name or root.name
    text = template_text(template, study_name)
    root.mkdir(parents=True, exist_ok=True)
    (root / "media").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    study_file.write_text(text, encoding="utf-8")
    load_study(study_file)  # templates must always validate
    return study_file


def validate_study(folder: str | Path) -> LoadedStudy:
    """Validate ``study.yaml``. Raises ``StudyValidationError`` listing every problem."""
    return load_study(resolve_folder(folder) / "study.yaml")


@dataclass(frozen=True, slots=True)
class PlanSummary:
    """What a locked study would run, and what it could detect."""

    spec: StudySpec
    warnings: tuple[str, ...]
    design_hash: str
    n_conditions: int
    trials_per_arm: dict[str, int]
    schedule: tuple[Slot, ...]
    baseline: float
    detectable_increase: float | None
    detectable_decrease: float | None


def plan_study(folder: str | Path, *, baseline: float = 0.5, power: float = 0.80) -> PlanSummary:
    """Validate the study and preview its schedule and minimum detectable effects.

    The MDE uses the independent-samples pooled-z formula at the primary alpha with the
    trials planned per arm. For paired (complete-block) designs this is conservative: pairing
    usually detects smaller differences.
    """
    loaded = validate_study(folder)
    spec = loaded.spec
    schedule = tuple(build_schedule(spec))
    per_arm = Counter(s.arm for s in schedule)
    alpha = spec.analysis.primary.alpha
    increase = decrease = None
    if spec.design.type == "randomized_block" and spec.analysis.primary.comparison is not None:
        comparison = spec.analysis.primary.comparison
        n_control, n_treatment = per_arm[comparison.control], per_arm[comparison.treatment]
        up = mde(baseline, n_control, n_treatment, alpha=alpha, power=power)
        down = mde(baseline, n_control, n_treatment, alpha=alpha, power=power, direction="decrease")
        increase, decrease = up.effect, down.effect
    return PlanSummary(
        spec=spec,
        warnings=loaded.warnings,
        design_hash=design_hash(spec),
        n_conditions=spec.conditions.count(),
        trials_per_arm=dict(per_arm),
        schedule=schedule,
        baseline=baseline,
        detectable_increase=increase,
        detectable_decrease=decrease,
    )


# --- lock / amend / unblind ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LockResult:
    """Outcome of locking a study."""

    study_id: str
    design_hash: str
    slots: int
    blind_codes: dict[str, str]


def _insert_design(
    db: Session, study_id: str, spec: StudySpec
) -> tuple[dict[str, str], dict[str, str]]:
    """Insert arms and conditions; return id maps by arm key and condition key."""
    codes = blind_codes([a.id for a in spec.arms], spec.design.seed)
    arm_ids: dict[str, str] = {}
    for arm in spec.arms:
        row = m.Arm(
            study_id=study_id,
            key=arm.id,
            blind_code=codes[arm.id],
            label=arm.label,
            runner=arm.runner,
            policy_ref=dict(arm.policy),
            serving=dict(arm.serving),
        )
        db.add(row)
        db.flush()
        arm_ids[arm.id] = row.id
    condition_ids: dict[str, str] = {}
    for cond in conditions(spec):
        crow = m.Condition(
            study_id=study_id, key=cond.key, factors=dict(cond.factors), position=cond.index
        )
        db.add(crow)
        db.flush()
        condition_ids[cond.key] = crow.id
    return arm_ids, condition_ids


def lock_study(folder: str | Path, *, actor: str = SYSTEM_ACTOR) -> LockResult:
    """Freeze the design: create the database, store the design, its hash and the schedule."""
    root = resolve_folder(folder)
    loaded = validate_study(root)
    spec = loaded.spec
    db_path = root / DB_FILENAME
    if db_path.exists():
        raise ServiceError(f"{root} is already locked; use `fieldtrial amend` to change it")
    engine = open_database(db_path, create=True)
    try:
        sessions = session_factory(engine)
        digest = design_hash(spec)
        schedule = build_schedule(spec)
        with sessions.begin() as db:
            now = utcnow()
            study = m.Study(
                name=spec.name,
                title=spec.title,
                design_yaml=loaded.text,
                design_hash=digest,
                status="locked",
                locked_at=now,
                fieldtrial_version=fieldtrial.__version__,
            )
            db.add(study)
            db.flush()
            arm_ids, condition_ids = _insert_design(db, study.id, spec)
            for slot in schedule:
                db.add(
                    m.ScheduleSlot(
                        study_id=study.id,
                        seq=slot.seq,
                        block=slot.block,
                        replicate=slot.replicate,
                        position=slot.position,
                        condition_id=condition_ids[slot.condition],
                        arm_id=arm_ids[slot.arm],
                        origin="planned",
                        status="pending",
                    )
                )
            append_event(
                db,
                study.id,
                "design_locked",
                actor,
                {"design_hash": digest, "slots": len(schedule), "seed": spec.design.seed},
            )
            codes = {a.key: a.blind_code for a in db.scalars(select(m.Arm))}
            study_id = study.id
    except Exception:
        engine.dispose()
        db_path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
        raise
    engine.dispose()
    return LockResult(study_id=study_id, design_hash=digest, slots=len(schedule), blind_codes=codes)


@dataclass(frozen=True, slots=True)
class AmendResult:
    """Outcome of an amendment."""

    old_hash: str
    new_hash: str
    added_slots: int
    voided_slots: int


def amend_study(folder: str | Path, reason: str, *, actor: str = SYSTEM_ACTOR) -> AmendResult:
    """Apply edits made to ``study.yaml`` after locking, as a logged amendment.

    Completed and running slots are kept. Pending slots that the new design no longer
    contains are voided, and slots the new design adds are appended to the run order. Arms
    and conditions already used keep their identities; new ones are added. The amendment,
    its reason and a diff of the YAML are recorded in the event log and in every report.
    """
    if not reason.strip():
        raise ServiceError("an amendment needs a reason")
    root = resolve_folder(folder)
    loaded = validate_study(root)
    new_spec = loaded.spec
    with open_study(root) as ctx, ctx.db() as db, db.begin():
        study = db.get_one(m.Study, ctx.study_id)
        old_hash = study.design_hash
        new_hash = design_hash(new_spec)
        if new_hash == old_hash and study.design_yaml == loaded.text:
            raise ServiceError("study.yaml has not changed since it was locked")
        if new_spec.design.seed != ctx.spec.design.seed:
            raise ServiceError("design.seed cannot be amended; it defines the randomization")

        arms = {a.key: a for a in db.scalars(select(m.Arm).where(m.Arm.study_id == study.id))}
        conds = {
            c.key: c
            for c in db.scalars(select(m.Condition).where(m.Condition.study_id == study.id))
        }
        used_codes = {a.blind_code for a in arms.values()}
        for arm in new_spec.arms:
            if arm.id in arms:
                row = arms[arm.id]
                row.label, row.runner = arm.label, arm.runner
                row.policy_ref, row.serving = dict(arm.policy), dict(arm.serving)
            else:
                code = next(c for c in code_pool(new_spec.design.seed) if c not in used_codes)
                used_codes.add(code)
                row = m.Arm(
                    study_id=study.id,
                    key=arm.id,
                    blind_code=code,
                    label=arm.label,
                    runner=arm.runner,
                    policy_ref=dict(arm.policy),
                    serving=dict(arm.serving),
                )
                db.add(row)
                db.flush()
                arms[arm.id] = row
        for cond in conditions(new_spec):
            if cond.key not in conds:
                crow = m.Condition(
                    study_id=study.id, key=cond.key, factors=dict(cond.factors), position=cond.index
                )
                db.add(crow)
                db.flush()
                conds[cond.key] = crow

        arm_key = {a.id: k for k, a in arms.items()}
        cond_key = {c.id: k for k, c in conds.items()}
        slots = list(
            db.scalars(
                select(m.ScheduleSlot)
                .where(m.ScheduleSlot.study_id == study.id)
                .order_by(m.ScheduleSlot.seq)
            )
        )
        wanted = Counter((s.condition, s.arm, s.replicate) for s in build_schedule(new_spec))
        kept: Counter[tuple[str, str, int]] = Counter()
        voided = 0
        for slot in slots:
            key = (cond_key[slot.condition_id], arm_key[slot.arm_id], slot.replicate)
            if slot.status == "void":
                continue
            if kept[key] < wanted[key]:
                kept[key] += 1
            elif slot.status == "pending":
                slot.status = "void"
                voided += 1
            else:
                kept[key] += 1  # completed work is never discarded
        next_seq = max((s.seq for s in slots), default=0)
        next_block = max((s.block for s in slots), default=0)
        added = 0
        new_blocks: dict[tuple[str, int], int] = {}
        for planned in build_schedule(new_spec):
            key = (planned.condition, planned.arm, planned.replicate)
            if kept[key] >= wanted[key]:
                continue
            kept[key] += 1
            block_key = (planned.condition, planned.replicate)
            if block_key not in new_blocks:
                next_block += 1
                new_blocks[block_key] = next_block
            next_seq += 1
            added += 1
            db.add(
                m.ScheduleSlot(
                    study_id=study.id,
                    seq=next_seq,
                    block=new_blocks[block_key],
                    replicate=planned.replicate,
                    position=planned.position,
                    condition_id=conds[planned.condition].id,
                    arm_id=arms[planned.arm].id,
                    origin="planned",
                    status="pending",
                )
            )
        diff = "".join(
            difflib.unified_diff(
                study.design_yaml.splitlines(keepends=True),
                loaded.text.splitlines(keepends=True),
                "study.yaml (locked)",
                "study.yaml (amended)",
            )
        )
        study.design_yaml = loaded.text
        study.design_hash = new_hash
        study.title = new_spec.title
        append_event(
            db,
            study.id,
            "amendment",
            actor,
            {
                "reason": reason,
                "old_hash": old_hash,
                "new_hash": new_hash,
                "added_slots": added,
                "voided_slots": voided,
                "diff": diff,
            },
        )
    return AmendResult(old_hash, new_hash, added, voided)


def unblind_study(folder: str | Path, *, actor: str = SYSTEM_ACTOR) -> dict[str, str]:
    """Reveal arm identities. Logged; the report flags unblinding before the study completed."""
    with open_study(folder) as ctx, ctx.db() as db, db.begin():
        study = db.get_one(m.Study, ctx.study_id)
        pending = db.scalar(
            select(func.count())
            .select_from(m.ScheduleSlot)
            .where(m.ScheduleSlot.study_id == study.id, m.ScheduleSlot.status == "pending")
        )
        already = study.unblinded_at is not None
        if not already:
            study.unblinded_at = utcnow()
        append_event(
            db,
            study.id,
            "unblinded",
            actor,
            {"pending_slots": int(pending or 0), "repeat": already},
        )
        return {
            a.blind_code: a.key for a in db.scalars(select(m.Arm).where(m.Arm.study_id == study.id))
        }


# --- status ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatusReport:
    """Progress of a study. Per-arm outcomes are included only once it is safe to show them."""

    name: str
    status: str
    design_hash: str
    blinded: bool
    planned: int
    done: int
    pending: int
    void: int
    invalid_trials: int
    sessions: int
    amendments: int
    locked_at: datetime | None
    unblinded_at: datetime | None
    per_arm: dict[str, tuple[int, int]] | None  # arm -> (successes, completed)


def study_status(folder: str | Path) -> StatusReport:
    """Progress counts. While the study is blinded, no per-arm results are returned."""
    with open_study(folder) as ctx, ctx.db() as db:
        study = db.get_one(m.Study, ctx.study_id)
        slot_counts = Counter(
            db.scalars(select(m.ScheduleSlot.status).where(m.ScheduleSlot.study_id == study.id))
        )
        invalid = db.scalar(
            select(func.count())
            .select_from(m.Trial)
            .where(m.Trial.study_id == study.id, m.Trial.status == "invalid")
        )
        sessions = db.scalar(
            select(func.count()).select_from(m.Session).where(m.Session.study_id == study.id)
        )
        amendments = len(list_events(db, study.id, "amendment"))
        blinded = ctx.spec.design.blinding == "operator" and study.unblinded_at is None
        per_arm = None
        if not blinded:
            rows = db.execute(
                select(m.Arm.key, m.Trial.success)
                .join(m.ScheduleSlot, m.Trial.slot_id == m.ScheduleSlot.id)
                .join(m.Arm, m.ScheduleSlot.arm_id == m.Arm.id)
                .where(m.Trial.study_id == study.id, m.Trial.status == "completed")
            ).all()
            per_arm = {a.id: (0, 0) for a in ctx.spec.arms}
            for key, success in rows:
                k, n = per_arm.get(key, (0, 0))
                per_arm[key] = (k + int(bool(success)), n + 1)
        return StatusReport(
            name=study.name,
            status=study.status,
            design_hash=study.design_hash,
            blinded=blinded,
            planned=sum(slot_counts.values()),
            done=slot_counts.get("done", 0),
            pending=slot_counts.get("pending", 0),
            void=slot_counts.get("void", 0),
            invalid_trials=int(invalid or 0),
            sessions=int(sessions or 0),
            amendments=amendments,
            locked_at=study.locked_at,
            unblinded_at=study.unblinded_at,
            per_arm=per_arm,
        )


def is_blinded(ctx: StudyContext) -> bool:
    """Whether arm identities and per-arm results must stay hidden right now."""
    if ctx.spec.design.blinding != "operator":
        return False
    with ctx.db() as db:
        return db.get_one(m.Study, ctx.study_id).unblinded_at is None


@dataclass(frozen=True, slots=True)
class StudyEntry:
    """A study folder found under a served directory."""

    slug: str  # folder name, used in URLs
    folder: Path
    name: str | None
    title: str | None
    status: str  # draft, invalid, locked, running, closed
    locked: bool


def _entry(folder: Path) -> StudyEntry:
    if (folder / DB_FILENAME).exists():
        report = study_status(folder)
        with open_study(folder) as ctx:
            title = ctx.spec.title
        return StudyEntry(folder.name, folder, report.name, title, report.status, True)
    try:
        spec = validate_study(folder).spec
    except (StudyValidationError, OSError):
        return StudyEntry(folder.name, folder, None, None, "invalid", False)
    return StudyEntry(folder.name, folder, spec.name, spec.title, "draft", False)


def list_studies(root: str | Path) -> list[StudyEntry]:
    """The study at ``root``, or the studies in its subfolders (sorted by folder name)."""
    base = resolve_folder(root)
    if (base / "study.yaml").exists():
        return [_entry(base)]
    if not base.is_dir():
        raise ServiceError(f"{base} is not a folder")
    return [
        _entry(child)
        for child in sorted(base.iterdir())
        if child.is_dir() and (child / "study.yaml").exists()
    ]
