"""Randomness that is identical on every platform and numpy version.

NumPy guarantees the raw bit stream of ``PCG64`` seeded from a ``SeedSequence``, but not the
output of ``Generator`` methods such as ``shuffle`` across versions. Schedules must be
reproducible from the study seed forever, so shuffling and sampling here use only the raw
64-bit outputs.
"""

from collections.abc import MutableSequence
from typing import TypeVar

import numpy as np
from numpy.random import PCG64, SeedSequence

T = TypeVar("T")

# Independent streams derived from the study seed, one per purpose.
STREAM_SCHEDULE = 1
STREAM_BLINDING = 2
STREAM_SIMULATION = 3
STREAM_BOOTSTRAP = 4

_TWO_64 = 2**64


class StableRng:
    """Uniform integers, floats and shuffles from ``PCG64(SeedSequence([seed, stream]))``."""

    def __init__(self, seed: int, stream: int) -> None:
        if seed < 0 or stream < 0:
            raise ValueError("seed and stream must be non-negative")
        self._bits = PCG64(SeedSequence([seed, stream]))

    def raw(self) -> int:
        """Next raw 64-bit output."""
        return int(self._bits.random_raw())

    def below(self, n: int) -> int:
        """Uniform integer in ``[0, n)``, by rejection sampling (no modulo bias)."""
        if n <= 0:
            raise ValueError("n must be positive")
        limit = (_TWO_64 // n) * n
        while True:
            draw = self.raw()
            if draw < limit:
                return draw % n

    def uniform(self) -> float:
        """Uniform float in ``[0, 1)`` with 53 random bits."""
        return (self.raw() >> 11) * (1.0 / (1 << 53))

    def shuffle(self, items: MutableSequence[T]) -> None:
        """Shuffle in place (Fisher–Yates)."""
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def integers(self, n: int, size: int) -> np.ndarray:
        """``size`` uniform integers in ``[0, n)``, vectorized rejection sampling."""
        if n <= 0 or size < 0:
            raise ValueError("n must be positive and size non-negative")
        if n == 1:
            return np.zeros(size, dtype=np.int64)
        remainder = _TWO_64 % n
        out = np.empty(0, dtype=np.uint64)
        while out.size < size:
            draws = np.asarray(self._bits.random_raw(size - out.size + 8), dtype=np.uint64)
            if remainder:  # reject the top `remainder` values so every residue is equally likely
                draws = draws[draws < np.uint64(_TWO_64 - remainder)]
            out = np.concatenate([out, draws])
        result: np.ndarray = (out[:size] % np.uint64(n)).astype(np.int64)
        return result
