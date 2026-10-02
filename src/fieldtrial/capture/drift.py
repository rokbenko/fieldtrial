"""Rig drift: compare a photo of the rig with its reference photo (numpy only).

A rig drifts when the camera is bumped, the light changes or the scene layout moves. Three
numbers describe the difference between the reference image and a current one, both
reduced to grayscale at about 256 pixels wide:

- **shift**: the translation that best aligns the two images, found by phase correlation
  (Kuglin and Hines, 1975), reported in pixels of the reference image
- **brightness change**: the relative change in mean brightness
- **similarity**: the structural similarity index (SSIM; Wang et al., 2004) of the aligned
  images, with uniform 7 × 7 windows, from 1 (identical structure) downwards

A check is flagged when any number crosses its threshold. A flag is a reason to look at
the rig, not proof that it changed.

References
----------
Kuglin, C. D. and Hines, D. C. (1975). The phase correlation image alignment method.
*Proc. IEEE Conference on Cybernetics and Society*, 163–165.

Wang, Z., Bovik, A. C., Sheikh, H. R. and Simoncelli, E. P. (2004). Image quality
assessment: from error visibility to structural similarity. *IEEE Transactions on Image
Processing* 13, 600–612.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

WORK_WIDTH = 256
WINDOW = 7
_DATA_RANGE = 255.0
_C1 = (0.01 * _DATA_RANGE) ** 2
_C2 = (0.03 * _DATA_RANGE) ** 2

Gray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DriftThresholds:
    """When a rig check is flagged."""

    max_shift_px: float = 8.0
    max_brightness_change: float = 0.15
    min_similarity: float = 0.75


@dataclass(frozen=True, slots=True)
class DriftResult:
    """The difference between a rig photo and the reference.

    ``dx`` and ``dy`` are in pixels of the reference image: the current image's content
    sits ``dx`` pixels to the right and ``dy`` pixels lower. ``brightness_change`` is
    relative (+0.2 means 20% brighter).
    """

    dx: float
    dy: float
    shift_px: float
    brightness_change: float
    similarity: float
    flagged: bool
    reasons: tuple[str, ...]


def to_gray(image: NDArray[np.generic]) -> Gray:
    """An image (H×W, H×W×3 or H×W×4, any numeric dtype on 0–255) as float grayscale."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[2] not in (3, 4):
            raise ValueError(f"expected 3 or 4 colour channels, got {arr.shape[2]}")
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114])  # ITU-R BT.601 luma
    elif arr.ndim != 2:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {arr.shape}")
    if min(arr.shape) < 2 * WINDOW:
        raise ValueError(f"image is too small ({arr.shape[1]}×{arr.shape[0]} pixels)")
    return arr


def downscale(gray: Gray, width: int = WORK_WIDTH) -> tuple[Gray, float]:
    """Area-average downscale to about ``width`` pixels; returns the image and the factor."""
    factor = max(1, gray.shape[1] // width)
    if factor == 1:
        return gray, 1.0
    h = gray.shape[0] // factor * factor
    w = gray.shape[1] // factor * factor
    small = gray[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return small, float(factor)


def _resize_nearest(gray: Gray, shape: tuple[int, int]) -> Gray:
    rows = (np.arange(shape[0]) * gray.shape[0] / shape[0]).astype(int)
    cols = (np.arange(shape[1]) * gray.shape[1] / shape[1]).astype(int)
    return gray[np.ix_(rows, cols)]


def phase_shift(reference: Gray, current: Gray) -> tuple[int, int]:
    r"""Integer ``(dy, dx)`` by which ``current`` is shifted relative to ``reference``.

    The normalized cross-power spectrum :math:`R = F_c \overline{F_r} / |F_c \overline{F_r}|`
    has an inverse transform that peaks at the shift. A Hann window reduces edge effects.
    """
    if reference.shape != current.shape:
        raise ValueError("images must have the same shape")
    window = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1]))
    fr = np.fft.fft2((reference - reference.mean()) * window)
    fc = np.fft.fft2((current - current.mean()) * window)
    cross = fc * np.conj(fr)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.ifft2(cross).real
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    return (int(dy) - h if dy > h // 2 else int(dy), int(dx) - w if dx > w // 2 else int(dx))


def _box_mean(x: Gray, k: int) -> Gray:
    """Mean over every k × k window (valid positions only), via summed-area tables."""
    s = np.pad(x, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    total = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    mean: Gray = total / (k * k)
    return mean


def ssim(a: Gray, b: Gray, window: int = WINDOW) -> float:
    r"""Mean structural similarity of two equally sized grayscale images (0–255 scale).

    For each ``window`` × ``window`` patch, with means :math:`\mu`, variances
    :math:`\sigma^2` and covariance :math:`\sigma_{ab}` (population moments),

    .. math::

        \mathrm{SSIM} = \frac{(2\mu_a\mu_b + C_1)(2\sigma_{ab} + C_2)}
                             {(\mu_a^2 + \mu_b^2 + C_1)(\sigma_a^2 + \sigma_b^2 + C_2)},

    with :math:`C_1 = (0.01 \cdot 255)^2` and :math:`C_2 = (0.03 \cdot 255)^2`, averaged
    over all patches (Wang et al., 2004, with uniform instead of Gaussian windows).
    """
    if a.shape != b.shape:
        raise ValueError("images must have the same shape")
    if min(a.shape) < window:
        raise ValueError(f"images must be at least {window} pixels in each direction")
    mu_a, mu_b = _box_mean(a, window), _box_mean(b, window)
    var_a = _box_mean(a * a, window) - mu_a**2
    var_b = _box_mean(b * b, window) - mu_b**2
    cov = _box_mean(a * b, window) - mu_a * mu_b
    num = (2 * mu_a * mu_b + _C1) * (2 * cov + _C2)
    den = (mu_a**2 + mu_b**2 + _C1) * (var_a + var_b + _C2)
    return float(np.mean(num / den))


def compare(
    reference: NDArray[np.generic],
    current: NDArray[np.generic],
    thresholds: DriftThresholds | None = None,
) -> DriftResult:
    """Compare a current rig image with the reference image."""
    limits = thresholds or DriftThresholds()
    ref, factor = downscale(to_gray(reference))
    cur_full = to_gray(current)
    # Bring the current image to the reference's working size (cameras may differ slightly).
    cur, _ = downscale(cur_full, width=max(1, cur_full.shape[1] // max(1, int(factor))))
    if cur.shape != ref.shape:
        cur = _resize_nearest(cur, (ref.shape[0], ref.shape[1]))

    dy, dx = phase_shift(ref, cur)
    h, w = ref.shape
    # Overlap of the two images after undoing the shift.
    r = ref[max(0, -dy) : h + min(0, -dy), max(0, -dx) : w + min(0, -dx)]
    c = cur[max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)]
    similarity = ssim(r, c) if min(r.shape) >= WINDOW else 0.0

    ref_mean = float(ref.mean())
    brightness = (float(cur.mean()) - ref_mean) / max(ref_mean, 1.0)
    shift = math.hypot(dx, dy) * factor

    reasons = []
    if shift > limits.max_shift_px:
        reasons.append(f"camera or scene moved by {shift:.0f} px")
    if abs(brightness) > limits.max_brightness_change:
        reasons.append(f"brightness changed by {brightness * 100:+.0f}%")
    if similarity < limits.min_similarity:
        reasons.append(f"scene looks different (similarity {similarity:.2f})")
    return DriftResult(
        dx=dx * factor,
        dy=dy * factor,
        shift_px=shift,
        brightness_change=brightness,
        similarity=similarity,
        flagged=bool(reasons),
        reasons=tuple(reasons),
    )
