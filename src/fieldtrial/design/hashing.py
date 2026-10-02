"""The design hash: a fingerprint of everything that matters for the analysis."""

import hashlib
import json
from typing import Any

from fieldtrial.design.models import StudySpec

# Cosmetic fields (docs/PLAN.md section 10): editing them does not change the hash.
_COSMETIC: dict[str, Any] = {
    "title": True,
    "task": {"description": True, "instruction": True},
    "arms": {"__all__": {"label": True}},
    "rubric": {"stages": {"__all__": {"label": True}}},
}


def normalized_design(spec: StudySpec) -> dict[str, Any]:
    """The study without cosmetic fields, with every default filled in."""
    return spec.model_dump(mode="json", exclude=_COSMETIC)


def design_hash(spec: StudySpec) -> str:
    """SHA-256 of the normalized design, serialized as JSON with sorted keys.

    Cosmetic edits (``title``, ``task.description``, ``task.instruction``, arm and stage
    ``label``) leave the hash unchanged; any other change produces a new hash. Writing a
    default value explicitly does not change the hash either.
    """
    canonical = json.dumps(normalized_design(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
