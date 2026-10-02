"""Rig drift checks on synthetic images.

SSIM is checked against a direct per-window implementation (loops over every 7 × 7 patch,
the definition in Wang et al., 2004). Shifts, brightness changes and occlusions are applied
to a seeded texture, so the expected answers are known exactly.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fieldtrial.capture.drift import (
    DriftThresholds,
    compare,
    downscale,
    phase_shift,
    ssim,
    to_gray,
)


def texture(seed: int = 0, shape: tuple[int, int] = (480, 640)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = rng.random((-(-shape[0] // 8), -(-shape[1] // 8)))
    img = np.kron(blocks, np.ones((8, 8)))[: shape[0], : shape[1]] * 200 + 20
    return np.clip(img + rng.normal(0, 3, shape), 0, 255)


def ssim_reference(a: np.ndarray, b: np.ndarray, k: int = 7) -> float:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    values = []
    for i in range(a.shape[0] - k + 1):
        for j in range(a.shape[1] - k + 1):
            x, y = a[i : i + k, j : j + k], b[i : i + k, j : j + k]
            mx, my = x.mean(), y.mean()
            vx, vy = x.var(), y.var()
            cov = ((x - mx) * (y - my)).mean()
            values.append(
                ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
            )
    return float(np.mean(values))


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 10_000), noise=st.floats(0, 60))
def test_ssim_matches_the_per_window_definition(seed: int, noise: float) -> None:
    rng = np.random.default_rng(seed)
    a = rng.random((24, 30)) * 255
    b = np.clip(a + rng.normal(0, noise, a.shape), 0, 255)
    assert ssim(a, b) == pytest.approx(ssim_reference(a, b), rel=1e-9, abs=1e-12)
    assert ssim(a, b) == pytest.approx(ssim(b, a), rel=1e-12)  # symmetric
    assert ssim(a, a) == pytest.approx(1.0)


@pytest.mark.parametrize(("dy", "dx"), [(0, 0), (6, -10), (-12, 4), (20, 30)])
def test_phase_correlation_finds_the_shift(dy: int, dx: int) -> None:
    ref = texture()[:256, :256]
    cur = np.roll(ref, (dy, dx), axis=(0, 1))
    assert phase_shift(ref, cur) == (dy, dx)


def test_identical_images_are_not_flagged() -> None:
    img = texture()
    res = compare(img, img)
    assert res.shift_px == 0
    assert res.brightness_change == 0
    assert res.similarity == pytest.approx(1.0)
    assert not res.flagged
    assert res.reasons == ()


def test_shift_is_reported_in_reference_pixels() -> None:
    img = texture()
    res = compare(img, np.roll(img, (10, -16), axis=(0, 1)))
    assert (res.dy, res.dx) == (10.0, -16.0)
    assert res.shift_px == pytest.approx(np.hypot(10, 16))
    assert res.similarity > 0.95  # same scene once aligned
    assert res.flagged
    assert res.reasons == ("camera or scene moved by 19 px",)
    small = compare(img, np.roll(img, (2, 2), axis=(0, 1)))
    assert not small.flagged


def test_brightness_and_occlusion() -> None:
    img = texture()
    brighter = compare(img, np.clip(img * 1.3, 0, 255))
    assert brighter.brightness_change > 0.2
    assert any("brightness" in r for r in brighter.reasons)
    occluded = img.copy()
    occluded[100:400, 150:500] = 0
    res = compare(img, occluded)
    assert res.similarity < 0.75
    assert any("scene looks different" in r for r in res.reasons)


def test_color_and_size_differences() -> None:
    gray = texture()
    rgb = np.repeat(gray[..., None], 3, axis=2)
    assert compare(gray, rgb).similarity == pytest.approx(1.0)
    # A camera at a slightly different resolution still compares.
    bigger = np.kron(gray, np.ones((2, 2)))
    res = compare(gray, bigger)
    assert res.similarity > 0.9
    assert res.shift_px <= 4


def test_thresholds_and_invalid_images() -> None:
    img = texture()
    shifted = np.roll(img, (10, 0), axis=(0, 1))
    assert not compare(img, shifted, DriftThresholds(max_shift_px=20)).flagged
    with pytest.raises(ValueError, match="too small"):
        to_gray(np.zeros((5, 5)))
    with pytest.raises(ValueError, match="colour channels"):
        to_gray(np.zeros((40, 40, 2)))
    small, factor = downscale(texture(shape=(100, 120)))
    assert factor == 1.0
    assert small.shape == (100, 120)
