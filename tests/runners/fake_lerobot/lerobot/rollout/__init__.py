"""Rollout: contexts, inference engines and the strategy base class."""

import threading
from dataclasses import dataclass, field
from typing import Any

from draccus import ChoiceRegistry

from lerobot import record


@dataclass
class RuntimeContext:
    cfg: Any
    shutdown_event: threading.Event


@dataclass
class HardwareContext:
    robot_wrapper: Any
    teleop: Any
    initial_position: dict | None = None


@dataclass
class PolicyContext:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    inference: Any


@dataclass
class ProcessorContext:
    teleop_action_processor: Any
    robot_action_processor: Any
    robot_observation_processor: Any


@dataclass
class DatasetContext:
    dataset: Any
    dataset_features: dict = field(default_factory=dict)
    hw_features: dict = field(default_factory=dict)
    ordered_action_keys: list[str] = field(default_factory=list)


@dataclass
class RolloutContext:
    runtime: RuntimeContext
    hardware: HardwareContext
    policy: PolicyContext
    processors: ProcessorContext
    data: DatasetContext


class InferenceEngineConfig(ChoiceRegistry):
    pass


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    pass


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    rtc: dict = field(default_factory=lambda: {"execution_horizon": 10})
    queue_threshold: int = 30


class _Engine:
    """Action = observed joint positions + the policy's offset."""

    def __init__(self, kind: str, policy: Any, keys: list[str], task: str) -> None:
        self.kind, self.policy, self.keys, self.task = kind, policy, keys, task
        self.started = self.paused = False
        self.calls = 0
        self.failed = False

    def start(self) -> None:
        self.started = True
        record("engine_start", self.kind)

    def stop(self) -> None:
        self.started = False
        record("engine_stop", self.kind)

    def reset(self) -> None:
        self.policy.reset()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def notify_observation(self, obs: dict) -> None:
        pass

    @property
    def ready(self) -> bool:
        return True

    def get_action(self, obs_frame: dict | None) -> list[float] | None:
        assert self.started
        assert not self.paused
        self.calls += 1
        if self.policy.config.engine_fails:
            self.failed = True
        state = (obs_frame or {}).get("observation.state", [0.0] * len(self.keys))
        return [float(x) + self.policy.offset for x in state]


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    robot_wrapper: Any,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: threading.Event | None = None,
) -> _Engine:
    record("engine", config.type, task, fps, device, use_torch_compile)
    return _Engine(config.type, policy, ordered_action_keys, task)


class ActionInterpolator:
    def __init__(self, multiplier: int = 1) -> None:
        self.multiplier = multiplier
        self._current: list[float] | None = None
        self._left = 0

    def get_control_interval(self, fps: float) -> float:
        return 1.0 / (fps * self.multiplier)

    def reset(self) -> None:
        self._current, self._left = None, 0

    def needs_new_action(self) -> bool:
        return self._left == 0

    def add(self, action: list[float]) -> None:
        self._current, self._left = action, self.multiplier

    def get(self) -> list[float] | None:
        if self._current is None:
            return None
        self._left = max(self._left - 1, 0)
        return self._current


class RolloutStrategy:
    def __init__(self, config: Any) -> None:
        self.config = config
        self._engine: Any = None
        self._interpolator: Any = None
        self._cached_obs_processed: dict | None = None

    def _init_engine(self, ctx: RolloutContext) -> None:
        self._interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        self._engine = ctx.policy.inference
        self._engine.reset()
        self._engine.start()
        self._cached_obs_processed = None

    def _process_observation_and_notify(self, processors: ProcessorContext, obs_raw: dict) -> dict:
        if self._cached_obs_processed is None or self._interpolator.needs_new_action():
            processed = processors.robot_observation_processor(obs_raw)
            self._engine.notify_observation(processed)
            self._cached_obs_processed = processed
        return self._cached_obs_processed

    def _teardown_hardware(
        self, hw: HardwareContext, return_to_initial_position: bool = True
    ) -> None:
        if self._engine is not None:
            self._engine.stop()
        robot = hw.robot_wrapper.inner
        if robot.is_connected:
            if return_to_initial_position and hw.initial_position:
                self._return_to_initial_position(hw)
            robot.disconnect()

    @staticmethod
    def _return_to_initial_position(
        hw: HardwareContext, duration_s: float = 3.0, fps: int = 50
    ) -> None:
        record("return_to_initial", duration_s)
        hw.robot_wrapper.send_action(dict(hw.initial_position or {}))
