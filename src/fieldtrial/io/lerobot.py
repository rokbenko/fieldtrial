"""Read LeRobot v3.0 datasets: episodes and interventions (``lerobot`` extra: pyarrow).

The layout follows LeRobot 0.6.1 (``lerobot/datasets/utils.py`` and
``dataset_metadata.py``): ``meta/info.json`` holds ``codebase_version`` and ``fps``;
``meta/episodes/chunk-*/file-*.parquet`` has one row per episode with ``episode_index``,
``length`` and ``tasks``; ``data/chunk-*/file-*.parquet`` has one row per frame with
``episode_index`` and, for DAgger rollouts, an ``intervention`` flag. A dataset given by
repo id is looked up where LeRobot caches it: ``$HF_LEROBOT_HOME/<repo_id>``, by default
``~/.cache/huggingface/lerobot/<repo_id>``. Nothing is downloaded.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    """A dataset that cannot be read."""


@dataclass(frozen=True, slots=True)
class Episode:
    """One recorded episode. ``intervention_frames`` is None without intervention flags."""

    index: int
    length: int
    tasks: tuple[str, ...]
    intervention_frames: int | None = None


@dataclass(frozen=True, slots=True)
class Dataset:
    """A LeRobot dataset's episodes."""

    root: Path
    codebase_version: str
    fps: float
    episodes: tuple[Episode, ...]
    has_interventions: bool


def _pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetError(
            "reading LeRobot datasets needs the lerobot extra: pip install 'fieldtrial[lerobot]'"
        ) from exc
    return pq


def lerobot_home() -> Path:
    """Where LeRobot keeps datasets by repo id (``HF_LEROBOT_HOME``)."""
    if os.environ.get("HF_LEROBOT_HOME"):
        return Path(os.environ["HF_LEROBOT_HOME"]).expanduser()
    hf_home = os.environ.get("HF_HOME") or "~/.cache/huggingface"
    return Path(hf_home).expanduser() / "lerobot"


def dataset_root(dataset: str | Path) -> Path:
    """A dataset folder, or the cached folder of a repo id such as ``user/name``."""
    path = Path(dataset).expanduser()
    if (path / "meta" / "info.json").exists():
        return path
    cached = lerobot_home() / str(dataset)
    if (cached / "meta" / "info.json").exists():
        return cached
    raise DatasetError(
        f"no LeRobot dataset at {path} or {cached} (expected meta/info.json); "
        "download it with LeRobot first"
    )


def _flag_count(values: list[Any]) -> int:
    count = 0
    for v in values:
        if isinstance(v, list | tuple):
            count += int(any(bool(x) for x in v))
        else:
            count += int(bool(v))
    return count


def read_dataset(dataset: str | Path) -> Dataset:
    """Read a v3.0 dataset's episodes (and intervention counts, if it has the flag)."""
    pq = _pyarrow()
    root = dataset_root(dataset)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise DatasetError(
            f"{root} is a LeRobot {version or 'unknown'} dataset; only v3.0 is supported "
            "(convert it with LeRobot's dataset conversion script)"
        )
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise DatasetError(f"{root} has no episode metadata (meta/episodes/*/*.parquet)")
    rows: dict[int, tuple[int, tuple[str, ...]]] = {}
    for f in files:
        table = pq.read_table(f, columns=["episode_index", "length", "tasks"]).to_pydict()
        for idx, length, tasks in zip(
            table["episode_index"], table["length"], table["tasks"], strict=True
        ):
            rows[int(idx)] = (int(length), tuple(str(t) for t in (tasks or [])))

    has_flag = "intervention" in info.get("features", {})
    flags: dict[int, int] = {}
    if has_flag:
        for f in sorted((root / "data").glob("*/*.parquet")):
            table = pq.read_table(f, columns=["episode_index", "intervention"]).to_pydict()
            by_episode: dict[int, list[Any]] = {}
            for idx, flag in zip(table["episode_index"], table["intervention"], strict=True):
                by_episode.setdefault(int(idx), []).append(flag)
            for idx, values in by_episode.items():
                flags[idx] = flags.get(idx, 0) + _flag_count(values)
    episodes = tuple(
        Episode(
            index=idx,
            length=length,
            tasks=tasks,
            intervention_frames=flags.get(idx, 0) if has_flag else None,
        )
        for idx, (length, tasks) in sorted(rows.items())
    )
    return Dataset(
        root=root,
        codebase_version=version,
        fps=float(info.get("fps", 0) or 0),
        episodes=episodes,
        has_interventions=has_flag,
    )
