"""Processors: identity pipelines."""

from typing import Any


class _Identity:
    steps: tuple[Any, ...] = ()

    def __call__(self, value: Any) -> Any:
        if isinstance(value, tuple):  # (action, observation) -> action
            return value[0]
        return value


def make_default_processors() -> tuple[Any, Any, Any]:
    return _Identity(), _Identity(), _Identity()


def rename_stats(stats: Any, rename_map: dict[str, str]) -> dict[str, Any]:
    return dict(stats or {})
