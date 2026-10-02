"""Fake draccus: ``decode`` picks a registered subclass by its ``type``."""

from typing import Any


def decode(cls: Any, data: dict[str, Any]) -> Any:
    data = dict(data)
    choice = data.pop("type", None)
    registry = getattr(cls, "_choices", None)
    if registry is not None:
        if choice not in registry:
            raise ValueError(f"unknown type {choice!r}; known: {sorted(registry)}")
        cls = registry[choice]
    return cls(**data)


class ChoiceRegistry:
    """Subclasses register under a name and are chosen by ``type``."""

    _choices: dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "_choices" not in cls.__dict__ and not any(
            "_choices" in b.__dict__ for b in cls.__mro__[1:-1]
        ):
            cls._choices = {}

    @classmethod
    def register_subclass(cls, name: str) -> Any:
        def wrap(sub: Any) -> Any:
            cls._choices[name] = sub
            sub._choice_name = name
            return sub

        return wrap

    @property
    def type(self) -> str:
        return str(self._choice_name)
