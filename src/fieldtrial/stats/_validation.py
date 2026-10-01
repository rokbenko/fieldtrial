"""Argument checks shared by the statistics modules."""

import math
import numbers

from fieldtrial.stats._types import ALTERNATIVES, Alternative


def check_count(name: str, value: object) -> int:
    """Return ``value`` as an int, or raise if it is not a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return int(value)


def check_counts(k: object, n: object, *, k_name: str = "k", n_name: str = "n") -> tuple[int, int]:
    """Validate a successes/trials pair: ``0 <= k <= n`` and ``n >= 1``."""
    k_int = check_count(k_name, k)
    n_int = check_count(n_name, n)
    if n_int == 0:
        raise ValueError(f"{n_name} must be at least 1, got 0")
    if k_int > n_int:
        raise ValueError(f"{k_name} ({k_int}) cannot exceed {n_name} ({n_int})")
    return k_int, n_int


def check_open_unit(name: str, value: float) -> float:
    """Return ``value`` as a float, or raise unless ``0 < value < 1``."""
    v = float(value)
    if not (math.isfinite(v) and 0.0 < v < 1.0):
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return v


def check_closed_unit(name: str, value: float) -> float:
    """Return ``value`` as a float, or raise unless ``0 <= value <= 1``."""
    v = float(value)
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
    return v


def check_alternative(alternative: str) -> Alternative:
    """Return ``alternative`` if it is one of the supported alternatives."""
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")
    return alternative


def check_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    """Return ``value`` if it is in ``choices``."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value
