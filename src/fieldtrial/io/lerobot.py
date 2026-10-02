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


@dataclass(frozen=True, slots=True)
class VideoRef:
    """Where one episode's video is: a file shared by several episodes, and the time span."""

    camera: str
    path: Path
    start_s: float
    end_s: float


def video_cameras(dataset: str | Path) -> list[str]:
    """The dataset's video features (camera keys), in the order ``info.json`` lists them."""
    root = dataset_root(dataset)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    return [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]


def episode_video(dataset: str | Path, episode_index: int, camera: str | None = None) -> VideoRef:
    """The video file and time span of one episode (the first camera unless one is named).

    In LeRobot v3.0, episodes share video files; ``meta/episodes`` holds each episode's
    ``videos/<camera>/chunk_index``, ``file_index``, ``from_timestamp`` and
    ``to_timestamp``, and ``info.json`` the ``video_path`` template.
    """
    pq = _pyarrow()
    root = dataset_root(dataset)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    cameras = video_cameras(root)
    if not cameras:
        raise DatasetError(f"{root} has no video features")
    key = camera or cameras[0]
    if key not in cameras:
        raise DatasetError(f"{root} has no camera {key!r}; it has {', '.join(cameras)}")
    template = str(info.get("video_path") or "")
    if not template:
        raise DatasetError(f"{root} has no video_path in meta/info.json")
    columns = [
        "episode_index",
        f"videos/{key}/chunk_index",
        f"videos/{key}/file_index",
        f"videos/{key}/from_timestamp",
        f"videos/{key}/to_timestamp",
    ]
    for f in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        table = pq.read_table(f, columns=columns).to_pydict()
        for i, idx in enumerate(table["episode_index"]):
            if int(idx) == episode_index:
                rel = template.format(
                    video_key=key,
                    chunk_index=int(table[columns[1]][i]),
                    file_index=int(table[columns[2]][i]),
                )
                return VideoRef(
                    camera=key,
                    path=root / rel,
                    start_s=float(table[columns[3]][i]),
                    end_s=float(table[columns[4]][i]),
                )
    raise DatasetError(f"{root} has no episode {episode_index}")
