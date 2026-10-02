"""The Results JSON schema published in the docs must match the model."""

import json
from pathlib import Path

from fieldtrial.analysis.results import SCHEMA_VERSION, Results

SCHEMA = Path(__file__).parents[2] / "docs" / "reference" / "results.schema.json"


def test_published_schema_is_current() -> None:
    published = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert published == Results.model_json_schema(), (
        "docs/reference/results.schema.json is stale; regenerate it with "
        '`uv run python -c "import json; from fieldtrial.analysis.results import Results; '
        'print(json.dumps(Results.model_json_schema(), indent=2))" '
        "> docs/reference/results.schema.json`"
    )
    assert published["properties"]["schema_version"]["default"] == SCHEMA_VERSION
