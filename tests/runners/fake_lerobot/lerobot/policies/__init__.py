"""A fake policy: action = joint state + ``offset`` (from the checkpoint's config)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lerobot import record
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep


class _Param:
    def numel(self) -> int:
        return 1024 * 1024

    def element_size(self) -> int:
        return 4


class FakePolicy:
    def __init__(self, config: Any, offset: float) -> None:
        self.config = config
        self.offset = offset
        self.device = "cpu"
        self.rtc_processor: Any = None
        self.resets = 0

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        *,
        config: Any,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "FakePolicy":
        if config.fail_load:
            raise RuntimeError(f"cannot load weights from {pretrained_name_or_path}")
        data = json.loads((Path(pretrained_name_or_path) / "config.json").read_text())
        record("load", str(pretrained_name_or_path), revision)
        return cls(config, float(data.get("offset", 0.0)))

    def to(self, device: str) -> "FakePolicy":
        self.device = device
        return self

    def eval(self) -> "FakePolicy":
        return self

    def parameters(self) -> list[_Param]:
        return [_Param()]

    def supports_rtc(self) -> bool:
        return bool(self.config.supports_rtc)

    def init_rtc_processor(self) -> None:
        self.rtc_processor = self.config.rtc_config
        record("init_rtc", self.config.rtc_config)

    def reset(self) -> None:
        self.resets += 1


def get_policy_class(name: str) -> type[FakePolicy]:
    if name != "fake":
        raise ValueError(f"unknown policy type {name!r}")
    return FakePolicy


def make_pre_post_processors(
    policy_cfg: Any,
    pretrained_path: str | None = None,
    pretrained_revision: str | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    record(
        "processors",
        pretrained_path,
        kwargs.get("dataset_stats"),
        kwargs.get("preprocessor_overrides"),
    )
    steps = (RelativeActionsProcessorStep(),) if policy_cfg.relative_actions else ()
    return SimpleNamespace(steps=steps), SimpleNamespace(steps=())
