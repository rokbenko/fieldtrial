from types import SimpleNamespace
from typing import Any


def is_torch_device_available(device: str) -> bool:
    return device == "cpu"


def auto_select_torch_device() -> Any:
    return SimpleNamespace(type="cpu")
