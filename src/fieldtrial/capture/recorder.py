"""Record an evaluation camera for each trial (``capture`` extra: OpenCV and PyAV).

The camera is opened once and read in a background thread while a trial runs. Frames are
encoded to H.264 MP4 at the configured frame rate (MPEG-4 Part 2 when the H.264 encoder
is missing). The finished clip is attached to the trial under ``media/<trial>/``.

The frame source is an interface, so tests and other cameras can provide frames without
OpenCV; :class:`CameraSource` reads an OpenCV device index or stream URL.
"""

import contextlib
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


class CaptureError(RuntimeError):
    """The camera or the encoder failed."""


class FrameSource(Protocol):
    """Gives RGB frames (H×W×3, uint8); ``None`` when no frame is available."""

    def read(self) -> Frame | None:
        """The next frame."""

    def close(self) -> None:
        """Release the device."""


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise CaptureError(
            "camera capture needs the capture extra: pip install 'fieldtrial[capture]'"
        ) from exc
    return cv2


def _av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise CaptureError(
            "camera capture needs the capture extra: pip install 'fieldtrial[capture]'"
        ) from exc
    return av


class CameraSource:
    """An OpenCV camera: a device index (0 is the first camera) or a stream or file URL."""

    def __init__(
        self,
        camera: int | str,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> None:
        cv2 = _cv2()
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(camera)
        if not self._cap.isOpened():
            raise CaptureError(f"cannot open camera {camera!r}")
        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, width),
            (cv2.CAP_PROP_FRAME_HEIGHT, height),
            (cv2.CAP_PROP_FPS, fps),
        ):
            if value:
                self._cap.set(prop, float(value))

    def read(self) -> Frame | None:
        """The next frame as RGB, or None."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return np.asarray(self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB), dtype=np.uint8)

    def close(self) -> None:
        """Release the camera."""
        self._cap.release()


def _encoder(av: Any) -> str:
    return "libx264" if "libx264" in av.codecs_available else "mpeg4"


class TrialRecorder:
    """Records one clip per trial from a frame source opened on first use."""

    def __init__(self, open_source: Callable[[], FrameSource], *, fps: float = 15.0) -> None:
        self._open_source = open_source
        self.fps = fps
        self._source: FrameSource | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._path: Path | None = None
        self._error: BaseException | None = None
        self.frames = 0

    def _ensure_source(self) -> FrameSource:
        with self._lock:
            if self._source is None:
                self._source = self._open_source()
            return self._source

    def grab(self, attempts: int = 10) -> Frame:
        """One frame now (for a rig check). Waits briefly for a camera that warms up."""
        source = self._ensure_source()
        for _ in range(attempts):
            with self._lock:
                frame = source.read()
            if frame is not None:
                return frame
            time.sleep(0.05)
        raise CaptureError("the camera gave no frame")

    @property
    def recording(self) -> bool:
        """Whether a clip is being recorded."""
        return self._thread is not None

    def start(self, path: Path) -> None:
        """Start recording into ``path`` (an .mp4 file)."""
        if self._thread is not None:
            raise CaptureError("already recording")
        source = self._ensure_source()
        av = _av()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path, self._error, self.frames = path, None, 0
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._record, args=(av, source, path), name="trial-recorder", daemon=True
        )
        self._thread.start()

    def _record(self, av: Any, source: FrameSource, path: Path) -> None:
        container = None
        try:
            container = av.open(str(path), "w")
            stream = None
            period = 1.0 / self.fps
            began = time.monotonic()
            while not self._stop.is_set():
                with self._lock:
                    frame = source.read()
                if frame is None:
                    time.sleep(period / 2)
                    continue
                h, w = frame.shape[0] // 2 * 2, frame.shape[1] // 2 * 2  # yuv420p needs even
                if stream is None:
                    stream = container.add_stream(_encoder(av), rate=round(self.fps))
                    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
                video = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(frame[:h, :w, :3]), format="rgb24"
                )
                for packet in stream.encode(video):
                    container.mux(packet)
                self.frames += 1
                # Pace to the frame rate; a slower camera simply gives fewer frames.
                wait = began + self.frames * period - time.monotonic()
                if wait > 0:
                    self._stop.wait(wait)
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
        except BaseException as exc:
            self._error = exc
        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()

    def stop(self) -> Path | None:
        """Stop recording; returns the clip, or None when no frame was recorded."""
        thread, path = self._thread, self._path
        if thread is None:
            return None
        self._stop.set()
        thread.join(timeout=30)
        self._thread = None
        if self._error is not None:
            raise CaptureError(f"recording failed: {self._error}") from self._error
        if path is None or self.frames == 0 or not path.exists():
            if path is not None:
                path.unlink(missing_ok=True)
            return None
        return path

    def close(self) -> None:
        """Stop any recording and release the camera."""
        with contextlib.suppress(CaptureError):
            self.stop()
        with self._lock:
            if self._source is not None:
                self._source.close()
                self._source = None
