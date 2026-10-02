"""Write the REST API's OpenAPI document to docs/reference/openapi.json."""

import json
import tempfile
from pathlib import Path

from fieldtrial.web.app import create_app

OUT = Path(__file__).parents[1] / "docs" / "reference" / "openapi.json"

with tempfile.TemporaryDirectory() as folder:
    spec = create_app(folder).openapi()
OUT.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
