"""JSON Lines export: one trial per line."""

import json
from collections.abc import Sequence
from pathlib import Path

from fieldtrial.analysis.records import TrialRecord
from fieldtrial.io.csv import export_rows


def write_trials_jsonl(records: Sequence[TrialRecord], path: str | Path, *, blinded: bool) -> int:
    """Write trials as JSON Lines. While blinded, ``arm`` holds blind codes. Returns rows."""
    rows = export_rows(records, blinded=blinded)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    return len(rows)
