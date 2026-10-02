from typing import Any

from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame


def send_next_action(
    obs_processed: dict, obs_raw: dict, ctx: Any, interpolator: Any
) -> dict | None:
    engine = ctx.policy.inference
    keys = ctx.data.ordered_action_keys
    if interpolator.needs_new_action():
        frame = build_dataset_frame(ctx.data.dataset_features, obs_processed, prefix=OBS_STR)
        action = engine.get_action(frame)
        if action is not None:
            interpolator.add(action)
    interp = interpolator.get()
    if interp is None:
        return None
    action_dict = {k: interp[i] for i, k in enumerate(keys)}
    processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
    ctx.hardware.robot_wrapper.send_action(processed)
    return action_dict
