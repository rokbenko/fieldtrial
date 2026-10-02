"""Fixtures for the console and API: a folder with locked studies and a test client."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.services.conftest import write_small_study

from fieldtrial.services.study import lock_study
from fieldtrial.web.app import create_app

API = {"X-Fieldtrial-Client": "test"}


@pytest.fixture
def studies(tmp_path: Path) -> Path:
    """A folder with a locked blinded study ``alpha``, an open one ``open`` and a draft."""
    root = tmp_path / "studies"
    lock_study(write_small_study(root / "alpha"))
    open_study = write_small_study(root / "open")
    path = open_study / "study.yaml"
    path.write_text(path.read_text().replace("blinding: operator", "blinding: none"))
    lock_study(open_study)
    write_small_study(root / "draft")
    return root


@pytest.fixture
def client(studies: Path) -> Iterator[TestClient]:
    """A client for the local (127.0.0.1) server."""
    with TestClient(create_app(studies), base_url="http://127.0.0.1") as c:
        yield c
