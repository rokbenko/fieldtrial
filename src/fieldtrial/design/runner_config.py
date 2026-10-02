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


class RunnersConfig(_Strict):
    """Settings for the runners that need them."""

    command: CommandRunnerConfig | None = None
    openpi_router: RouterConfig | None = None


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
