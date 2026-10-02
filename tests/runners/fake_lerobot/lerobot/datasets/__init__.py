"""A fake LeRobot v3.0 dataset: metadata only (``meta/info.json`` and episode parquet)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lerobot import record
from lerobot.utils.feature_utils import hw_to_dataset_features


def create_initial_features(action: dict | None = None, observation: dict | None = None) -> dict:
    return {"action": action or {}, "observation": observation or {}}


def aggregate_pipeline_dataset_features(
    pipeline: Any, initial_features: dict, use_videos: bool = True
) -> dict:
    out: dict[str, Any] = {}
    for prefix, hw in initial_features.items():
        out.update(hw_to_dataset_features(hw, prefix))
    return out


class LeRobotDataset:
    def __init__(self, repo_id: str, root: Path, **kwargs: Any) -> None:
        self.repo_id = repo_id
        self.root = Path(root)
        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.features = info["features"]
        self.fps = info["fps"]
        self._episodes: list[dict[str, Any]] = json.loads(
            (self.root / "meta" / "fake_episodes.json").read_text()
        )
        self.buffer: list[dict[str, Any]] = []
        self.finalized = False
        self.meta = SimpleNamespace(stats={"action": {"mean": [0.0]}} if self._episodes else None)
        self.kwargs = kwargs

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        **kwargs: Any,
    ) -> "LeRobotDataset":
        root = Path(root)  # type: ignore[arg-type]
        if root.exists():
            raise FileExistsError(root)
        (root / "meta").mkdir(parents=True)
        info = {
            "codebase_version": "v3.0",
            "fps": fps,
            "robot_type": robot_type,
            "features": {k: {**v, "shape": list(v["shape"])} for k, v in features.items()},
            "total_episodes": 0,
        }
        (root / "meta" / "info.json").write_text(json.dumps(info))
        (root / "meta" / "fake_episodes.json").write_text("[]")
        record("dataset_create", repo_id, str(root), use_videos, kwargs)
        return cls(repo_id, root, **kwargs)

    @classmethod
    def resume(
        cls, repo_id: str, root: str | Path | None = None, **kwargs: Any
    ) -> "LeRobotDataset":
        record("dataset_resume", repo_id, str(root))
        return cls(repo_id, Path(root), **kwargs)  # type: ignore[arg-type]

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def add_frame(self, frame: dict) -> None:
        if self.finalized:
            raise RuntimeError("the dataset is finalized")
        missing = {*self.features, "task"} - set(frame)
        if missing:
            raise ValueError(f"frame misses {sorted(missing)}")
        self.buffer.append(frame)

    def save_episode(
        self, episode_data: dict | None = None, parallel_encoding: bool = True
    ) -> None:
        if not self.buffer:
            raise ValueError("the episode buffer is empty")
        self._episodes.append(
            {
                "episode_index": len(self._episodes),
                "length": len(self.buffer),
                "tasks": sorted({f["task"] for f in self.buffer}),
            }
        )
        self.buffer = []

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        self.buffer = []

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        (self.root / "meta" / "fake_episodes.json").write_text(json.dumps(self._episodes))
        info = json.loads((self.root / "meta" / "info.json").read_text())
        info["total_episodes"] = len(self._episodes)
        (self.root / "meta" / "info.json").write_text(json.dumps(info))
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:  # read_dataset is then skipped by the tests
            return
        if self._episodes:
            folder = self.root / "meta" / "episodes" / "chunk-000"
            folder.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [e["episode_index"] for e in self._episodes],
                        "length": [e["length"] for e in self._episodes],
                        "tasks": [e["tasks"] for e in self._episodes],
                    }
                ),
                folder / "file-000.parquet",
            )
