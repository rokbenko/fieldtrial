"""Dataset features: ``action`` and ``observation.state`` vectors plus camera images."""

from typing import Any


def hw_to_dataset_features(hw: dict[str, Any], prefix: str) -> dict[str, Any]:
    joints = [k for k, v in hw.items() if v is float]
    out: dict[str, Any] = {}
    if joints:
        key = "action" if prefix == "action" else f"{prefix}.state"
        out[key] = {"dtype": "float32", "shape": (len(joints),), "names": joints}
    for k, v in hw.items():
        if isinstance(v, tuple):
            out[f"{prefix}.images.{k}"] = {"dtype": "video", "shape": v, "names": None}
    return out


def combine_feature_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in dicts:
        out.update(d)
    return out


def build_dataset_frame(features: dict[str, Any], values: dict[str, Any], prefix: str) -> dict:
    frame: dict[str, Any] = {}
    for key, ft in features.items():
        if not key.startswith(prefix):
            continue
        if ft["dtype"] == "video":
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]
        else:
            frame[key] = [values[name] for name in ft["names"]]
    return frame
