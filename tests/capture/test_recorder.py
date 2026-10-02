"""Per-trial camera recording, with a fake frame source and real encoding (capture extra)."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
pytest.importorskip("cv2")

from fieldtrial.capture.recorder import (  # noqa: E402
    CameraSource,
    CaptureError,
    Frame,
    TrialRecorder,
)


class FakeCamera:
    """Odd-sized RGB frames with a moving bar; counts reads; can fail or go dark."""

    def __init__(self, width: int = 101, height: int = 75, dark: bool = False) -> None:
        self.width, self.height, self.dark = width, height, dark
        self.reads = 0
        self.closed = False
        self.lock = threading.Lock()

    def read(self) -> Frame | None:
        with self.lock:
            self.reads += 1
            n = self.reads
        if self.dark:
            return None
        frame = np.full((self.height, self.width, 3), 30, np.uint8)
        x = n % self.width
        frame[:, x : x + 5] = (240, 200, 10)
        return frame

    def close(self) -> None:
        self.closed = True


def _decode(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]


def test_records_a_clip_at_the_frame_rate(tmp_path: Path) -> None:
    camera = FakeCamera()
    recorder = TrialRecorder(lambda: camera, fps=20)
    target = tmp_path / "clip.mp4"
    recorder.start(target)
    assert recorder.recording
    time.sleep(0.6)
    clip = recorder.stop()
    assert clip == target
    frames = _decode(target)
    assert 6 <= len(frames) <= 16  # about 0.6 s at 20 fps
    assert frames[0].shape == (74, 100, 3)  # cropped to even sizes for yuv420p
    assert not recorder.recording
    recorder.close()
    assert camera.closed


def test_no_frames_means_no_clip(tmp_path: Path) -> None:
    recorder = TrialRecorder(lambda: FakeCamera(dark=True), fps=20)
    recorder.start(tmp_path / "clip.mp4")
    time.sleep(0.2)
    assert recorder.stop() is None
    assert not (tmp_path / "clip.mp4").exists()
    with pytest.raises(CaptureError, match="no frame"):
        recorder.grab(attempts=2)


def test_grab_and_double_start(tmp_path: Path) -> None:
    recorder = TrialRecorder(lambda: FakeCamera(), fps=10)
    assert recorder.grab().shape == (75, 101, 3)
    recorder.start(tmp_path / "a.mp4")
    with pytest.raises(CaptureError, match="already recording"):
        recorder.start(tmp_path / "b.mp4")
    recorder.close()
    assert recorder.stop() is None


def test_opencv_camera_reads_a_video_file(tmp_path: Path) -> None:
    # An OpenCV "camera" can be a file or stream URL; this exercises the real cv2 path.
    source_clip = tmp_path / "source.mp4"
    recorder = TrialRecorder(lambda: FakeCamera(width=64, height=48), fps=20)
    recorder.start(source_clip)
    time.sleep(0.3)
    recorder.stop()
    camera = CameraSource(str(source_clip))
    frame = camera.read()
    assert frame is not None
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    camera.close()
    with pytest.raises(CaptureError, match="cannot open camera"):
        CameraSource(str(tmp_path / "missing.mp4"))


def test_cli_rig_check_from_the_camera(tmp_path: Path) -> None:
    from tests.services.conftest import write_small_study
    from typer.testing import CliRunner

    from fieldtrial.cli.main import app as cli
    from fieldtrial.services.study import lock_study

    clip = tmp_path / "camera.mp4"
    recorder = TrialRecorder(lambda: FakeCamera(width=320, height=240), fps=20)
    recorder.start(clip)
    time.sleep(0.3)
    recorder.stop()
    folder = write_small_study(tmp_path / "study")
    path = folder / "study.yaml"
    path.write_text(path.read_text() + f"capture:\n  camera: {clip}\n")
    lock_study(folder)
    runner = CliRunner()
    out = runner.invoke(cli, ["rig-check", str(folder), "--camera", "--set-reference"])
    assert out.exit_code == 0, out.output
    assert (folder / "rig" / "reference.png").exists()
    out = runner.invoke(cli, ["rig-check", str(folder), "--camera"])
    assert out.exit_code == 0, out.output
    assert "ok" in out.output or "flagged" in out.output
    out = runner.invoke(cli, ["rig-check", str(folder)])
    assert out.exit_code == 2
