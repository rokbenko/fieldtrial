"""Load and validate ``study.yaml`` with errors that point at the YAML path and line."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from fieldtrial.design.models import MAX_CONDITIONS_WARNING, StudySpec


@dataclass(frozen=True, slots=True)
class Issue:
    """One validation problem: where it is and what is wrong."""

    path: str
    message: str
    line: int | None = None

    def __str__(self) -> str:
        where = self.path or "(top level)"
        if self.line is not None:
            where = f"line {self.line}, {where}"
        return f"{where}: {self.message}"


class StudyValidationError(ValueError):
    """``study.yaml`` is not a valid study. ``issues`` lists every problem found."""

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        super().__init__("\n".join(str(i) for i in issues))


@dataclass(frozen=True, slots=True)
class LoadedStudy:
    """A validated study plus the source text and any warnings."""

    spec: StudySpec
    text: str
    warnings: tuple[str, ...]


def _format_path(loc: tuple[int | str, ...]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out


def _line_of(root: yaml.Node | None, loc: tuple[int | str, ...]) -> int | None:
    """Line (1-based) of the deepest YAML node that matches ``loc``."""
    node = root
    line = None if node is None else node.start_mark.line + 1
    for part in loc:
        child = None
        if isinstance(node, yaml.MappingNode) and isinstance(part, str):
            for key, value in node.value:
                if isinstance(key, yaml.ScalarNode) and key.value == part:
                    child = value
                    line = key.start_mark.line + 1
                    break
        elif (
            isinstance(node, yaml.SequenceNode)
            and isinstance(part, int)
            and 0 <= part < len(node.value)
        ):
            child = node.value[part]
            line = child.start_mark.line + 1
        if child is None:
            break
        node = child
    return line


def _clean_message(error: ErrorDetails) -> str:
    message = str(error["msg"])
    for prefix in ("Value error, ", "Assertion failed, "):
        if message.startswith(prefix):
            message = message[len(prefix) :]
    if error["type"] == "extra_forbidden":
        message = "unknown key (check the spelling; see docs/PLAN.md section 10)"
    return message


def parse_study(text: str) -> LoadedStudy:
    """Validate the YAML text of a study.

    Raises
    ------
    StudyValidationError
        With one :class:`Issue` per problem, each pointing at the YAML path and line.
    """
    try:
        data = yaml.safe_load(text)
        root = yaml.compose(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = None if mark is None else mark.line + 1
        raise StudyValidationError([Issue("", f"invalid YAML: {exc}", line)]) from exc
    if not isinstance(data, dict):
        raise StudyValidationError([Issue("", "study.yaml must be a mapping of keys to values")])
    try:
        spec = StudySpec.model_validate(data)
    except ValidationError as exc:
        issues = []
        for error in exc.errors():
            loc = tuple(p for p in error["loc"] if not (isinstance(p, str) and "[" in p))
            issues.append(Issue(_format_path(loc), _clean_message(error), _line_of(root, loc)))
        raise StudyValidationError(issues) from exc

    warnings = []
    n_conditions = spec.conditions.count()
    if n_conditions > MAX_CONDITIONS_WARNING:
        warnings.append(
            f"conditions.factors expand to {n_conditions} conditions "
            f"(more than {MAX_CONDITIONS_WARNING}); is that intended?"
        )
    return LoadedStudy(spec=spec, text=text, warnings=tuple(warnings))


def load_study(path: str | Path) -> LoadedStudy:
    """Read and validate ``study.yaml`` (or the ``study.yaml`` inside a study folder)."""
    p = Path(path)
    if p.is_dir():
        p = p / "study.yaml"
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise StudyValidationError([Issue("", f"{p} not found")]) from exc
    return parse_study(text)
