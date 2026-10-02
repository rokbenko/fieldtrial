"""Fake LeRobot 0.6.1 for fieldtrial's tests (see README.md)."""

from typing import Any

__version__ = "0.6.1"

CALLS: list[tuple[Any, ...]] = []


def record(*call: Any) -> None:
    CALLS.append(call)
