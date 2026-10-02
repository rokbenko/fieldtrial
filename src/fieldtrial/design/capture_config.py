"""Camera and rig-check settings in ``study.yaml`` (the ``capture:`` section).

These are about the setup, not the design: they are left out of the design hash, so
moving the camera to another port does not count as a change to the study.
"""

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DriftConfig(_Strict):
    """When a rig check against the reference photo is flagged.

    ``every_trials`` repeats the check from the camera every so many trials (camera only).
    """

    max_shift_px: float = Field(default=8.0, gt=0)
    max_brightness_change: float = Field(default=0.15, gt=0, lt=1)
    min_similarity: float = Field(default=0.75, gt=0, lt=1)
    every_trials: int | None = Field(default=None, ge=1)


class CaptureConfig(_Strict):
    """An evaluation camera (``capture`` extra) and rig drift checks.

    ``camera`` is an OpenCV device index (0 for the first camera) or a stream URL. With
    ``record``, every trial is recorded from Start to Stop into ``media/<trial>/``.
    """

    camera: int | str | None = None
    fps: float = Field(default=15.0, gt=0, le=120)
    width: int | None = Field(default=None, ge=16)
    height: int | None = Field(default=None, ge=16)
    record: bool = True
    drift: DriftConfig = Field(default_factory=DriftConfig)
