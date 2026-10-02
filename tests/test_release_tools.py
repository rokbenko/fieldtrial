"""Release helpers: changelog sections for GitHub release notes."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "changelog", ROOT / "scripts/changelog_section.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEXT = """# Changelog

## [Unreleased]

## [0.2.0] - 2027-01-01

### Added

- Things.

[0.2.0]: https://example.invalid/compare

## [0.1.0a1] - 2026-10-01

- First.
"""


def test_section_extraction() -> None:
    section = _load().section
    assert section(TEXT, "0.2.0") == "### Added\n\n- Things."
    assert section(TEXT, "0.1.0a1") == "- First."
    assert section(TEXT, "Unreleased") == ""
    assert section(TEXT, "9.9.9") is None
    assert section(TEXT, "0.1.0") is None  # no prefix matches


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    main = _load().main
    assert main(["x", "v0.1.0a1"]) == 0
    assert "calculator commands" in capsys.readouterr().out
    assert main(["x", "0.0.0"]) == 1
    assert main(["x"]) == 2
