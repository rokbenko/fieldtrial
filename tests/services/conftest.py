"""Fixtures for service tests: small study folders built from the bundled templates."""

from pathlib import Path

import pytest

from fieldtrial.services.study import init_study, lock_study


def write_small_study(folder: Path, *, slots: int = 4, replicates: int = 1) -> Path:
    """The basic template, shrunk to a few conditions so tests run fast."""
    init_study(folder, template="basic", name="small")
    path = folder / "study.yaml"
    text = path.read_text(encoding="utf-8")
    text = text.replace("range: [1, 40]", f"range: [1, {slots}]")
    text = text.replace("replicates: 1", f"replicates: {replicates}")
    path.write_text(text, encoding="utf-8")
    return folder


@pytest.fixture
def small_study(tmp_path: Path) -> Path:
    """An unlocked two-arm study with 4 conditions (8 slots)."""
    return write_small_study(tmp_path / "study")


@pytest.fixture
def locked_study(small_study: Path) -> Path:
    """The small study, locked."""
    lock_study(small_study)
    return small_study
