"""Link trials to the episodes of a LeRobot dataset (docs/PLAN.md section 13).

LeRobot episodes record no wall-clock time, so trials are matched to episodes in run
order: the first unlinked trial to ``first_episode``, the next to the next episode, and so
on. A mapping file (``trial,episode_index``, where ``trial`` is a trial id or its trial
number in the console) gives explicit pairs instead. A plan is shown before anything is
written; linking records one ``dataset_link`` event.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fieldtrial.io.lerobot import Dataset, DatasetError, read_dataset
from fieldtrial.services._context import ServiceError, StudyContext
from fieldtrial.services.events import append_event, list_events
from fieldtrial.services.trial import collect_records


@dataclass(frozen=True, slots=True)
class LinkRow:
    """One trial and the episode it will be linked to. Arms appear by blind code."""

    trial_id: str
    seq: int
    blind_code: str
    status: str
    episode_index: int
    length: int
    intervention_frames: int | None


@dataclass(frozen=True, slots=True)
class LinkPlan:
    """What linking would write, plus what is left over on either side."""

    dataset: str
    root: Path
    codebase_version: str
    rows: tuple[LinkRow, ...]
    unmatched_trials: int
    unmatched_episodes: int


def _linked(ctx: StudyContext) -> dict[str, dict[str, Any]]:
    """Trial id -> its latest link."""
    out: dict[str, dict[str, Any]] = {}
    with ctx.db() as db:
        for event in list_events(db, ctx.study_id, "dataset_link"):
            for link in event.payload.get("links", []):
                out[str(link["trial_id"])] = {**link, "dataset": event.payload.get("dataset")}
    return out


def _read_mapping(path: Path) -> list[tuple[str, int]]:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or not {"trial", "episode_index"} <= set(reader.fieldnames):
                raise ServiceError("the mapping file needs the columns trial,episode_index")
            return [(row["trial"].strip(), int(row["episode_index"])) for row in reader]
    except (OSError, ValueError) as exc:
        raise ServiceError(f"cannot read the mapping file: {exc}") from exc


def plan_links(
    ctx: StudyContext,
    dataset: str | Path,
    *,
    first_episode: int = 0,
    mapping: Path | None = None,
    relink: bool = False,
) -> LinkPlan:
    """Match trials to episodes without writing anything."""
    try:
        data: Dataset = read_dataset(dataset)
    except DatasetError as exc:
        raise ServiceError(str(exc)) from exc
    episodes = {e.index: e for e in data.episodes}
    records, _ = collect_records(ctx)
    trials = sorted(records, key=lambda r: (r.started_at, r.seq))
    done = {} if relink else _linked(ctx)
    used_episodes = {
        int(link["episode_index"]) for link in done.values() if link.get("dataset") == str(dataset)
    }
    pending = [r for r in trials if r.trial_id not in done]

    pairs: list[tuple[Any, int]] = []
    if mapping is not None:
        by_key = {r.trial_id: r for r in trials} | {str(r.seq): r for r in trials}
        for key, episode in _read_mapping(mapping):
            if key not in by_key:
                raise ServiceError(f"the mapping names an unknown trial {key!r}")
            pairs.append((by_key[key], episode))
    else:
        free = [i for i in sorted(episodes) if i >= first_episode and i not in used_episodes]
        pairs = list(zip(pending, free, strict=False))
    rows = []
    for record, index in pairs:
        if index not in episodes:
            raise ServiceError(f"the dataset has no episode {index}")
        ep = episodes[index]
        rows.append(
            LinkRow(
                trial_id=record.trial_id,
                seq=record.seq,
                blind_code=record.blind_code,
                status=record.status,
                episode_index=ep.index,
                length=ep.length,
                intervention_frames=ep.intervention_frames,
            )
        )
    linked_trials = {r.trial_id for r in rows}
    linked_eps = {r.episode_index for r in rows} | used_episodes
    return LinkPlan(
        dataset=str(dataset),
        root=data.root,
        codebase_version=data.codebase_version,
        rows=tuple(rows),
        unmatched_trials=sum(1 for r in pending if r.trial_id not in linked_trials),
        unmatched_episodes=sum(1 for i in episodes if i >= first_episode and i not in linked_eps),
    )


def link_episodes(ctx: StudyContext, plan: LinkPlan, *, actor: str = "fieldtrial") -> int:
    """Record a plan's links (one ``dataset_link`` event). Returns how many were linked."""
    if not plan.rows:
        raise ServiceError("nothing to link")
    with ctx.db() as db, db.begin():
        append_event(
            db,
            ctx.study_id,
            "dataset_link",
            actor,
            {
                "dataset": plan.dataset,
                "root": str(plan.root),
                "codebase_version": plan.codebase_version,
                "links": [
                    {
                        "trial_id": r.trial_id,
                        "episode_index": r.episode_index,
                        "length": r.length,
                        "intervention_frames": r.intervention_frames,
                    }
                    for r in plan.rows
                ],
            },
        )
    return len(plan.rows)


def record_episode_link(
    ctx: StudyContext,
    trial_id: str,
    *,
    root: str,
    codebase_version: str,
    episode_index: int,
    length: int,
    actor: str = "runner",
) -> None:
    """Link a trial to the episode a runner recorded for it (one ``dataset_link`` event)."""
    with ctx.db() as db, db.begin():
        append_event(
            db,
            ctx.study_id,
            "dataset_link",
            actor,
            {
                "dataset": root,
                "root": root,
                "codebase_version": codebase_version,
                "source": "runner",
                "links": [
                    {
                        "trial_id": trial_id,
                        "episode_index": episode_index,
                        "length": length,
                        "intervention_frames": None,
                    }
                ],
            },
        )
