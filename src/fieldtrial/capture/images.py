"""Reading and writing rig photos with Pillow (a dependency of matplotlib)."""

import io
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MAX_PIXELS = 40_000_000  # refuse decompression bombs; a 40 MP photo is plenty


def load_image(source: str | Path | bytes) -> NDArray[np.uint8]:
    """An image file (PNG, JPEG, ...) or its bytes as an RGB array."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    data = io.BytesIO(source) if isinstance(source, bytes) else Path(source)
    try:
        with Image.open(data) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"cannot read the image: {exc}") from exc


def save_png(image: NDArray[np.generic], path: str | Path) -> Path:
    """Save an RGB or grayscale array as PNG."""
    from PIL import Image

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(target, format="PNG")
    return target
