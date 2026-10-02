"""Policy configurations read from a checkpoint folder's ``config.json``."""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from lerobot import record


class FeatureType(StrEnum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ACTION = "ACTION"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]


@dataclass
class PreTrainedConfig:
    type: str = "fake"
    device: str | None = None
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    action_feature_names: list[str] | None = None
    use_peft: bool = False
    offset: float = 0.0
    supports_rtc: bool = False
    relative_actions: bool = False
    fail_load: bool = False
    engine_fails: bool = False
    rtc_config: Any = None
    pretrained_path: str | None = None

    @classmethod
    def from_pretrained(
        cls, pretrained_name_or_path: str, *, revision: str | None = None, **kwargs: Any
    ) -> "PreTrainedConfig":
        path = Path(pretrained_name_or_path) / "config.json"
        if not path.exists():
            raise FileNotFoundError(f"config.json not found in {pretrained_name_or_path}")
        data = json.loads(path.read_text())
        record("config", str(pretrained_name_or_path), revision)
        features = {
            k: PolicyFeature(FeatureType(v["type"]), tuple(v["shape"]))
            for k, v in data.pop("input_features", {}).items()
        }
        return cls(input_features=features, **data)
