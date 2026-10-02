"""Reward-model scores, blind human review and proxy data (docs/guides/reward-models.md).

- ``score_episodes`` runs a scorer over a LeRobot dataset and records one ``episode_score``
  event. Episodes linked to trials take the trial's blind code; the others take the blind
  code given for the whole dataset (for example extra rollouts of one arm), if any.
- ``draw_review_sample`` draws a random subset of the scored, unlinked episodes with the
  study seed and records it (``review_sample``) before anyone reviews, so the sample cannot
  depend on scores or labels.
- Review is blind first: ``record_review`` stores the reviewer's label (``review_label``)
  before the suggestion is shown; ``revise_review`` may change it afterwards with a reason
  (``review_revision``). Agreement uses the first labels; estimates use the final ones.
- ``import_proxy`` records scores from elsewhere (for example simulation, as in SureSim),
  with labels for a random subset (``proxy_import``).
"""

import csv
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from fieldtrial.io.lerobot import DatasetError, VideoRef, episode_video, read_dataset
from fieldtrial.rewards import EpisodeScorer, ScorerError
from fieldtrial.services._context import ServiceError, StudyContext
from fieldtrial.services.events import append_event, list_events
from fieldtrial.services.study import SYSTEM_ACTOR
from fieldtrial.services.trial import collect_records
from fieldtrial.stats._rng import STREAM_REVIEW, StableRng
from fieldtrial.store import models as m

DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """What a scoring run recorded."""

    dataset: str
    model: str
    episodes: int
    linked: int
    suggested_successes: int


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """One sampled episode to review. Nothing about its arm or score."""

    root: str
    episode_index: int
    position: int  # 1-based position in the review queue
    total: int
    video: VideoRef | None


@dataclass(frozen=True, slots=True)
class ReviewState:
    """A reviewed episode: the blind first label, the final label and the suggestion."""

    root: str
    episode_index: int
    first: bool
    final: bool
    suggested: bool | None
    score: float | None
    revised: bool


def _require_locked(ctx: StudyContext) -> None:
    with ctx.db() as db:
        if db.get_one(m.Study, ctx.study_id).status == "draft":
            raise ServiceError("lock the study first")


def _threshold(ctx: StudyContext, threshold: float | None) -> float:
    if threshold is not None:
        if not 0 < threshold < 1:
            raise ServiceError("threshold must be between 0 and 1")
        return threshold
    proxy = ctx.spec.analysis.proxy
    return proxy.threshold if proxy is not None else DEFAULT_THRESHOLD


def _events(ctx: StudyContext, kind: str) -> list[dict[str, Any]]:
    with ctx.db() as db:
        return [dict(e.payload) for e in list_events(db, ctx.study_id, kind)]


def _codes(ctx: StudyContext) -> set[str]:
    with ctx.db() as db:
        return set(db.scalars(select(m.Arm.blind_code).where(m.Arm.study_id == ctx.study_id)))


def _links(ctx: StudyContext, root: Path) -> dict[int, str]:
    """Episode index -> trial id, for links to this dataset."""
    out: dict[int, str] = {}
    for event in _events(ctx, "dataset_link"):
        if Path(str(event.get("root", ""))) != root:
            continue
        for link in event.get("links", []):
            out[int(link["episode_index"])] = str(link["trial_id"])
    return out


def score_episodes(
    ctx: StudyContext,
    dataset: str | Path,
    scorer: EpisodeScorer,
    *,
    blind_code: str | None = None,
    threshold: float | None = None,
    progress: Callable[[int, int], None] | None = None,
    actor: str = SYSTEM_ACTOR,
) -> ScoreSummary:
    """Score every episode of a dataset and record the scores (one event)."""
    _require_locked(ctx)
    try:
        data = read_dataset(dataset)
    except DatasetError as exc:
        raise ServiceError(str(exc)) from exc
    if blind_code is not None and blind_code not in _codes(ctx):
        raise ServiceError(f"{blind_code!r} is not a blind code of this study")
    cut = _threshold(ctx, threshold)
    links = _links(ctx, data.root)
    records, _ = collect_records(ctx)
    trial_codes = {r.trial_id: r.blind_code for r in records}
    rows: list[dict[str, Any]] = []
    for i, episode in enumerate(data.episodes):
        try:
            value = float(scorer.score(data.root, episode.index))
        except ScorerError as exc:
            raise ServiceError(f"episode {episode.index}: {exc}") from exc
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ServiceError(f"episode {episode.index}: score {value} is not in [0, 1]")
        trial = links.get(episode.index)
        rows.append(
            {
                "episode_index": episode.index,
                "score": value,
                "suggested": value >= cut,
                "trial_id": trial,
                "blind_code": trial_codes.get(trial) if trial else blind_code,
            }
        )
        if progress is not None:
            progress(i + 1, len(data.episodes))
    if not rows:
        raise ServiceError("the dataset has no episodes")
    with ctx.db() as db, db.begin():
        append_event(
            db,
            ctx.study_id,
            "episode_score",
            actor,
            {
                "dataset": str(dataset),
                "root": str(data.root),
                "model": scorer.name,
                "threshold": cut,
                "scores": rows,
            },
        )
    return ScoreSummary(
        dataset=str(dataset),
        model=scorer.name,
        episodes=len(rows),
        linked=sum(1 for r in rows if r["trial_id"]),
        suggested_successes=sum(1 for r in rows if r["suggested"]),
    )


def latest_scores(ctx: StudyContext) -> dict[tuple[str, int], dict[str, Any]]:
    """(dataset root, episode) -> its latest score row (with ``model`` and ``threshold``)."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for event in _events(ctx, "episode_score"):
        for row in event.get("scores", []):
            out[str(event["root"]), int(row["episode_index"])] = {
                **row,
                "model": event.get("model"),
                "threshold": event.get("threshold"),
            }
    return out


def _sampled(ctx: StudyContext) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for event in _events(ctx, "review_sample"):
        out.extend((str(event["root"]), int(i)) for i in event.get("episodes", []))
    return out


def draw_review_sample(
    ctx: StudyContext, dataset: str | Path, n: int, *, actor: str = SYSTEM_ACTOR
) -> list[int]:
    """Draw ``n`` scored, unlinked episodes of a dataset for blind review, and record them."""
    _require_locked(ctx)
    if n < 1:
        raise ServiceError("n must be at least 1")
    try:
        root = read_dataset(dataset).root
    except DatasetError as exc:
        raise ServiceError(str(exc)) from exc
    scores = latest_scores(ctx)
    already = set(_sampled(ctx))
    candidates = sorted(
        idx
        for (r, idx), row in scores.items()
        if r == str(root) and not row.get("trial_id") and (r, idx) not in already
    )
    if not candidates:
        raise ServiceError("no scored episodes left to review in this dataset; score it first")
    draws = len(_events(ctx, "review_sample"))
    rng = StableRng(ctx.spec.design.seed, STREAM_REVIEW * 1000 + draws)
    order = list(candidates)
    rng.shuffle(order)
    chosen = sorted(order[: min(n, len(order))])
    with ctx.db() as db, db.begin():
        append_event(
            db,
            ctx.study_id,
            "review_sample",
            actor,
            {
                "dataset": str(dataset),
                "root": str(root),
                "episodes": chosen,
                "candidates": len(candidates),
                "draw": draws + 1,
            },
        )
    return chosen


def _labels(ctx: StudyContext) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for event in _events(ctx, "review_label"):
        key = (str(event["root"]), int(event["episode_index"]))
        out.setdefault(key, {"first": bool(event["success"]), "final": bool(event["success"])})
    for event in _events(ctx, "review_revision"):
        key = (str(event["root"]), int(event["episode_index"]))
        if key in out:
            out[key]["final"] = bool(event["success"])
            out[key]["revised"] = True
    return out


def next_review(ctx: StudyContext) -> ReviewItem | None:
    """The next sampled episode without a label, or None when the queue is empty."""
    queue = _sampled(ctx)
    labels = _labels(ctx)
    for position, (root, idx) in enumerate(queue, start=1):
        if (root, idx) in labels:
            continue
        try:
            video: VideoRef | None = episode_video(root, idx)
        except DatasetError:
            video = None
        return ReviewItem(
            root=root, episode_index=idx, position=position, total=len(queue), video=video
        )
    return None


def review_progress(ctx: StudyContext) -> tuple[int, int]:
    """(reviewed, sampled) episodes."""
    queue = _sampled(ctx)
    labels = _labels(ctx)
    return sum(1 for key in queue if key in labels), len(queue)


def review_state(ctx: StudyContext, root: str, episode_index: int) -> ReviewState:
    """A reviewed episode with its suggestion (only after the first label exists)."""
    label = _labels(ctx).get((root, episode_index))
    if label is None:
        raise ServiceError("this episode has not been labelled yet")
    score = latest_scores(ctx).get((root, episode_index))
    return ReviewState(
        root=root,
        episode_index=episode_index,
        first=label["first"],
        final=label["final"],
        suggested=None if score is None else bool(score["suggested"]),
        score=None if score is None else float(score["score"]),
        revised=bool(label.get("revised")),
    )


def record_review(
    ctx: StudyContext, root: str, episode_index: int, success: bool, *, reviewer: str
) -> ReviewState:
    """Record the reviewer's blind first label of a sampled episode."""
    if (root, episode_index) not in set(_sampled(ctx)):
        raise ServiceError("this episode is not in the review sample")
    if (root, episode_index) in _labels(ctx):
        raise ServiceError("this episode already has a first label; revise it instead")
    with ctx.db() as db, db.begin():
        append_event(
            db,
            ctx.study_id,
            "review_label",
            reviewer,
            {"root": root, "episode_index": episode_index, "success": success},
        )
    return review_state(ctx, root, episode_index)


def revise_review(
    ctx: StudyContext,
    root: str,
    episode_index: int,
    success: bool,
    *,
    reason: str,
    reviewer: str,
) -> ReviewState:
    """Change a reviewed label after seeing the suggestion (logged with a reason)."""
    if not reason.strip():
        raise ServiceError("give a reason for changing the label")
    review_state(ctx, root, episode_index)  # raises without a first label
    with ctx.db() as db, db.begin():
        append_event(
            db,
            ctx.study_id,
            "review_revision",
            reviewer,
            {
                "root": root,
                "episode_index": episode_index,
                "success": success,
                "reason": reason.strip(),
            },
        )
    return review_state(ctx, root, episode_index)


def _flag(text: str) -> bool | None:
    value = text.strip().lower()
    if value in ("", "na", "none"):
        return None
    if value in ("1", "true", "yes", "success"):
        return True
    if value in ("0", "false", "no", "failure"):
        return False
    raise ServiceError(f"cannot read the label {text!r} (use 1/0, true/false or empty)")


def import_proxy(ctx: StudyContext, path: Path, *, source: str, actor: str = SYSTEM_ACTOR) -> int:
    """Record proxy scores from a CSV with columns ``blind_code,score[,label]``.

    Rows with a label are the labeled set and must be a random subset of the rows of their
    arm; the others are proxy-only. Returns the number of rows.
    """
    _require_locked(ctx)
    if not source.strip():
        raise ServiceError("give the proxy source a name")
    codes = _codes(ctx)
    rows: list[dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or ())
            if not {"blind_code", "score"} <= fields:
                raise ServiceError("the proxy file needs the columns blind_code,score[,label]")
            for line, row in enumerate(reader, start=2):
                code = row["blind_code"].strip()
                if code not in codes:
                    raise ServiceError(f"line {line}: {code!r} is not a blind code of this study")
                score = float(row["score"])
                if not (math.isfinite(score) and 0.0 <= score <= 1.0):
                    raise ServiceError(f"line {line}: score {score} is not in [0, 1]")
                rows.append(
                    {"blind_code": code, "score": score, "label": _flag(row.get("label") or "")}
                )
    except (OSError, ValueError) as exc:
        raise ServiceError(f"cannot read the proxy file: {exc}") from exc
    if not rows:
        raise ServiceError("the proxy file has no rows")
    with ctx.db() as db, db.begin():
        append_event(
            db, ctx.study_id, "proxy_import", actor, {"source": source.strip(), "rows": rows}
        )
    return len(rows)


def proxy_items(ctx: StudyContext) -> tuple[dict[str, Any], ...]:
    """Every scored item for the analysis: source, kind, blind code, score and labels.

    ``kind`` is ``trial`` (a linked trial; its live label is blind to the model),
    ``review`` (a sampled episode with a blind first label), ``unlabeled`` or ``import``.
    """
    records, _ = collect_records(ctx)
    trials = {r.trial_id: r for r in records}
    labels = _labels(ctx)
    items: list[dict[str, Any]] = []
    for (root, idx), row in sorted(latest_scores(ctx).items()):
        base = {
            "source": root,
            "episode_index": idx,
            "blind_code": row.get("blind_code"),
            "score": float(row["score"]),
            "suggested": bool(row["suggested"]),
            "model": row.get("model"),
            "threshold": row.get("threshold"),
        }
        trial = trials.get(str(row.get("trial_id"))) if row.get("trial_id") else None
        if trial is not None:
            if trial.status == "completed" and trial.success is not None:
                items.append(
                    {**base, "kind": "trial", "first": trial.success, "final": trial.success}
                )
        elif (root, idx) in labels:
            label = labels[root, idx]
            items.append(
                {**base, "kind": "review", "first": label["first"], "final": label["final"]}
            )
        else:
            items.append({**base, "kind": "unlabeled", "first": None, "final": None})
    for event in _events(ctx, "proxy_import"):
        for i, row in enumerate(event.get("rows", [])):
            label = row.get("label")
            items.append(
                {
                    "source": f"import:{event['source']}",
                    "episode_index": i,
                    "blind_code": row["blind_code"],
                    "score": float(row["score"]),
                    "suggested": None,
                    "model": event["source"],
                    "kind": "import",
                    "first": label,
                    "final": label,
                }
            )
    return tuple(items)
