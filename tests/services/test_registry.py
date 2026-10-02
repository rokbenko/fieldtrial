"""The study registry used by the server: lookup, caching and reload after amendments."""

from pathlib import Path

import pytest

from fieldtrial.services import ServiceError
from fieldtrial.services.registry import StudyRegistry, UnknownStudyError
from fieldtrial.services.study import amend_study


def test_single_study_folder(locked_study: Path) -> None:
    registry = StudyRegistry(locked_study / "study.yaml")
    assert registry.single
    assert [e.slug for e in registry.entries()] == [locked_study.name]
    assert registry.get(locked_study.name).spec.name == "small"
    assert registry.get(locked_study.name) is registry.get(locked_study.name)
    with pytest.raises(UnknownStudyError):
        registry.get("other")
    registry.close()


def test_reloads_design_after_amendment(locked_study: Path) -> None:
    registry = StudyRegistry(locked_study.parent)
    before = registry.get(locked_study.name)
    assert before.spec.conditions.count() == 4
    path = locked_study / "study.yaml"
    path.write_text(path.read_text().replace("range: [1, 4]", "range: [1, 5]"))
    amend_study(locked_study, "one more slot")
    after = registry.get(locked_study.name)
    assert after.spec.conditions.count() == 5
    assert after.engine is before.engine
    for bad in ("", ".", "..", "a/b"):
        with pytest.raises(UnknownStudyError):
            registry.folder(bad)
    registry.close()


def test_root_must_be_a_folder(tmp_path: Path) -> None:
    with pytest.raises(ServiceError, match="not a folder"):
        StudyRegistry(tmp_path / "missing")
