"""Robots: a configurable fake arm (``type: fake_arm``) with one camera."""

from dataclasses import dataclass
from typing import Any

from draccus import ChoiceRegistry

from lerobot import record


class RobotConfig(ChoiceRegistry):
    pass


@RobotConfig.register_subclass("fake_arm")
@dataclass
class FakeArmConfig(RobotConfig):
    id: str | None = None
    port: str = "/dev/null"
    joints: int = 2
    camera: str = "top"
    fail_after: int | None = None  # get_observation raises after this many calls
    fail_connect: bool = False


class Robot:
    def __init__(self, config: Any) -> None:
        self.config = config


class FakeArm(Robot):
    name = "fake_arm"
    robot_type = "fake_arm"

    def __init__(self, config: FakeArmConfig) -> None:
        super().__init__(config)
        self.pos = {f"j{i}.pos": 0.0 for i in range(config.joints)}
        self.connected = False
        self.observations = 0
        self.actions: list[dict[str, float]] = []

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**dict.fromkeys(self.pos, float), self.config.camera: (4, 6, 3)}

    @property
    def action_features(self) -> dict[str, Any]:
        return dict.fromkeys(self.pos, float)

    @property
    def cameras(self) -> dict[str, Any]:
        return {self.config.camera: object()}

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, calibrate: bool = True) -> None:
        if self.config.fail_connect:
            raise ConnectionError(f"no robot on {self.config.port}")
        self.connected = True
        record("connect", self.config.port)

    def get_observation(self) -> dict[str, Any]:
        self.observations += 1
        if self.config.fail_after is not None and self.observations > self.config.fail_after:
            raise OSError("motor bus timeout")
        return {**self.pos, self.config.camera: [[0] * 6] * 4}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.actions.append(dict(action))
        self.pos.update({k: float(v) for k, v in action.items() if k in self.pos})
        return action

    def disconnect(self) -> None:
        self.connected = False
        record("disconnect")


ROBOTS: list[FakeArm] = []


def make_robot_from_config(config: RobotConfig) -> Robot:
    robot = FakeArm(config)  # type: ignore[arg-type]
    ROBOTS.append(robot)
    return robot
