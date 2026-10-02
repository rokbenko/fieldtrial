import threading
from typing import Any


class ThreadSafeRobot:
    def __init__(self, robot: Any) -> None:
        self._robot = robot
        self._lock = threading.Lock()

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return self._robot.get_observation()

    def send_action(self, action: dict[str, Any]) -> Any:
        with self._lock:
            return self._robot.send_action(action)

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def inner(self) -> Any:
        return self._robot
