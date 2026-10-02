"""LeRobot v3.0 datasets for tests, with video metadata (the video files are placeholders)."""

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

CAMERA = "observation.images.top"


def make_video_dataset(root: Path, lengths: list[int], fps: int = 10) -> Path:
    """Episodes share one video file; each episode's span follows the previous one."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": fps,
                "features": {
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    CAMERA: {"dtype": "video", "shape": [3, 48, 64]},
                },
                "total_episodes": len(lengths),
                "video_path": (
                    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
                ),
            }
        )
    )
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(len(lengths))),
                "tasks": [["put the cup on the plate"]] * len(lengths),
                "length": lengths,
                f"videos/{CAMERA}/chunk_index": [0] * len(lengths),
                f"videos/{CAMERA}/file_index": [0] * len(lengths),
                f"videos/{CAMERA}/from_timestamp": [s / fps for s in starts],
                f"videos/{CAMERA}/to_timestamp": [
                    (s + n) / fps for s, n in zip(starts, lengths, strict=True)
                ],
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    video = root / "videos" / CAMERA / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # a placeholder; tests never decode it
    return root
