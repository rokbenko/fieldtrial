"""Runner settings in ``study.yaml`` and the command-template renderer (docs/PLAN.md §13).

The renderer lives here, not in :mod:`fieldtrial.runners`, so that ``fieldtrial validate``
and ``fieldtrial lock`` can reject a template with an unknown placeholder before any trial
runs.

Template syntax
---------------
A template is split into arguments like a shell would (``shlex``), but it is never run
through a shell. Placeholders are filled in each argument afterwards, so values with
spaces stay one argument. ``{{`` and ``}}`` are literal braces.

- ``{policy.KEY}`` or ``{policy[KEY]}``: a value from the arm's ``policy`` (``KEY`` may
  contain dots, as in ``{serving[inference.queue_threshold]}``)
- ``{serving.KEY}`` or ``{serving[KEY]}``: a value from the arm's ``serving``
- ``{factors.NAME}``: a factor of the trial's condition
- ``{blind_code}``, ``{trial_id}``, ``{seq}``, ``{condition}``, ``{instruction}``,
  ``{timeout_s}``, ``{study}``

The arm id is not available: the command line is visible to anyone at the robot, so it
gets the blind code only.
"""

import re
import shlex
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")
_SCALARS = ("blind_code", "trial_id", "seq", "condition", "instruction", "timeout_s", "study")
_MAPS = ("policy", "serving", "factors")


class TemplateError(ValueError):
    """A command template that cannot be rendered."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CommandRunnerConfig(_Strict):
    """The ``command`` runner: run a command per trial (or keep one per arm).

    ``launch: per_trial`` starts the command when a trial starts and stops it when the
    trial stops. ``launch: per_arm`` keeps the command running while consecutive trials
    use the same arm and tells it about each trial on its standard input (one line
    ``start {json}`` per trial start, ``stop`` per trial stop). Switching arms closes its
    standard input, which means "shut down", and signals it if it is still running after
    ``grace_s``.
    """

    template: str = Field(min_length=1)
    launch: Literal["per_trial", "per_arm"] = "per_trial"
    stop_signal: Literal["SIGINT", "SIGTERM"] = "SIGINT"
    grace_s: float = Field(default=10.0, gt=0, le=600)
    success_exit_code: int | None = None
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("template")
    @classmethod
    def _parses(cls, template: str) -> str:
        split_template(template)
        return template


class RouterConfig(_Strict):
    """The ``openpi_router`` runner: a websocket proxy in front of one policy server per arm.

    The robot's openpi client connects to ``listen``. Each arm's ``policy.url`` is its
    upstream server. ``metadata: identical`` refuses to serve when the arms' servers send
    different metadata (the client could tell them apart); ``first`` forwards the first
    arm's metadata and records the difference.
    """

    listen: str = Field(default="127.0.0.1:8000", pattern=r"^[^\s:]+:\d{1,5}$")
    metadata: Literal["identical", "first"] = "identical"


class LeRobotDatasetConfig(_Strict):
    """Where the ``lerobot`` runner records its episodes (a LeRobot v3.0 dataset).

    ``root`` is relative to the study folder; by default ``lerobot/<name>``, where ``name``
    is the part of ``repo_id`` after the slash. The dataset is reused across sessions.
    """

    repo_id: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
    root: str | None = None
    video: bool = True
    streaming_encoding: bool = False


class LeRobotRunnerConfig(_Strict):
    """The ``lerobot`` runner: LeRobot policies run inside fieldtrial (Python 3.12+).

    The robot is connected once and every trial is recorded as one episode. ``robot`` is
    LeRobot's robot configuration (as for ``lerobot-rollout --robot.*``), with its
    ``type``. ``keep_loaded`` is how many policies stay in memory (``all`` keeps every
    arm's policy loaded after its first trial). ``reset_to_initial_position`` moves the
    robot back to the pose it had when connected after each trial; ``return_to_initial_position``
    does so before disconnecting.

    ``robot``, ``dataset``, ``device`` and ``keep_loaded`` describe the setup and are left
    out of the design hash, so the robot's port can change after locking.
    """

    robot: dict[str, Any]
    fps: int = Field(default=30, gt=0, le=1000)
    device: str | None = None
    dataset: LeRobotDatasetConfig = Field(default_factory=LeRobotDatasetConfig)
    keep_loaded: int | Literal["all"] = 1
    rename_map: dict[str, str] = Field(default_factory=dict)
    reset_to_initial_position: bool = True
    reset_s: float = Field(default=2.0, gt=0, le=60)
    return_to_initial_position: bool = True

    @field_validator("robot")
    @classmethod
    def _robot_type(cls, robot: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(robot.get("type"), str) or not robot["type"]:
            raise ValueError("runners.lerobot.robot needs a type, for example so101_follower")
        return robot

    @field_validator("keep_loaded")
    @classmethod
    def _keep(cls, keep: int | str) -> int | str:
        if isinstance(keep, int) and keep < 1:
            raise ValueError("keep_loaded must be at least 1, or all")
        return keep


class RunnersConfig(_Strict):
    """Settings for the runners that need them."""

    command: CommandRunnerConfig | None = None
    openpi_router: RouterConfig | None = None
    lerobot: LeRobotRunnerConfig | None = None


def split_template(template: str) -> list[str]:
    """Split a template into arguments, without filling placeholders."""
    try:
        parts = shlex.split(template)
    except ValueError as exc:
        raise TemplateError(f"the command template cannot be split into arguments: {exc}") from exc
    if not parts:
        raise TemplateError("the command template is empty")
    return parts


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise TemplateError("a placeholder refers to an empty value")
    if isinstance(value, dict | list):
        raise TemplateError("a placeholder refers to a nested value; name a single key")
    return str(value)


def _lookup(field: str, values: Mapping[str, Any]) -> str:
    field = field.strip()
    if field in _SCALARS:
        return _format(values[field])
    match = re.fullmatch(r"(policy|serving|factors)(?:\.(.+)|\[(.+)\])", field)
    if match is None:
        raise TemplateError(
            f"unknown placeholder {{{field}}}; use {{policy.KEY}}, {{serving[KEY]}}, "
            f"{{factors.NAME}} or one of {', '.join('{' + s + '}' for s in _SCALARS)}"
        )
    name, key = match.group(1), match.group(2) or match.group(3)
    mapping = values[name]
    if key not in mapping:
        known = ", ".join(sorted(map(str, mapping))) or "none"
        raise TemplateError(f"{{{field}}}: {name} has no key {key!r} (keys: {known})")
    return _format(mapping[key])


def render_command(template: str, values: Mapping[str, Any]) -> list[str]:
    """The command's arguments, with every placeholder filled.

    ``values`` holds the scalars named above and the ``policy``, ``serving`` and
    ``factors`` mappings.
    """
    missing = [k for k in (*_SCALARS, *_MAPS) if k not in values]
    if missing:
        raise TemplateError(f"missing template values: {missing}")

    def fill(part: str) -> str:
        def sub(m: re.Match[str]) -> str:
            token = m.group(0)
            if token == "{{":
                return "{"
            if token == "}}":
                return "}"
            return _lookup(m.group(1), values)

        out = _PLACEHOLDER.sub(sub, part)
        if "{" in _PLACEHOLDER.sub("", part) or "}" in _PLACEHOLDER.sub("", part):
            raise TemplateError(f"unbalanced brace in {part!r}; write {{{{ or }}}} for a literal")
        return out

    return [fill(part) for part in split_template(template)]


def sample_values(
    *, policy: Mapping[str, Any], serving: Mapping[str, Any], factors: Mapping[str, Any]
) -> dict[str, Any]:
    """Placeholder values for checking a template before any trial exists."""
    return {
        "policy": dict(policy),
        "serving": dict(serving),
        "factors": dict(factors),
        "blind_code": "XX",
        "trial_id": "00000000-0000-0000-0000-000000000000",
        "seq": 1,
        "condition": "sample",
        "instruction": "",
        "timeout_s": "",
        "study": "study",
    }


def lerobot_arm_error(policy: Mapping[str, Any], serving: Mapping[str, Any]) -> str | None:
    """What is wrong with an arm of the ``lerobot`` runner, or None.

    ``policy.path`` is a checkpoint folder or Hub repo id (``policy.revision`` optional).
    ``serving.inference`` is LeRobot's inference configuration (``type: sync`` or ``rtc``
    with its options) and ``serving.interpolation_multiplier`` a whole number from 1.
    """
    path = policy.get("path")
    if not isinstance(path, str) or not path.strip():
        return "needs policy.path: a checkpoint folder or Hub repo id"
    revision = policy.get("revision")
    if revision is not None and not isinstance(revision, str):
        return "policy.revision must be text"
    inference = serving.get("inference")
    if inference is not None and (
        not isinstance(inference, dict) or inference.get("type", "sync") not in ("sync", "rtc")
    ):
        return "serving.inference needs type: sync or rtc"
    multiplier = serving.get("interpolation_multiplier", 1)
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 1:
        return "serving.interpolation_multiplier must be a whole number from 1"
    return None
