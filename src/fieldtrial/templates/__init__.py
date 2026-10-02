"""Starter ``study.yaml`` templates for ``fieldtrial init``."""

from importlib.resources import files

TEMPLATES = ("basic", "checkpoint-ladder", "serving-sweep")
NAME_PLACEHOLDER = "__NAME__"


def template_text(template: str, name: str) -> str:
    """The template's YAML with the study name filled in."""
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {TEMPLATES}, got {template!r}")
    text = files(__package__).joinpath(f"{template}.yaml").read_text(encoding="utf-8")
    return text.replace(NAME_PLACEHOLDER, name)
