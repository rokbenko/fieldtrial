"""Study design: the ``study.yaml`` schema, validation, schedules, blinding and hashing.

May import :mod:`fieldtrial.stats` (for power and MDE) but no other fieldtrial package.
"""

from fieldtrial.design.blinding import blind_codes
from fieldtrial.design.hashing import design_hash, normalized_design
from fieldtrial.design.loader import (
    Issue,
    LoadedStudy,
    StudyValidationError,
    load_study,
    parse_study,
)
from fieldtrial.design.models import StudySpec
from fieldtrial.design.schedule import Condition, Slot, build_schedule, conditions

__all__ = [
    "Condition",
    "Issue",
    "LoadedStudy",
    "Slot",
    "StudySpec",
    "StudyValidationError",
    "blind_codes",
    "build_schedule",
    "conditions",
    "design_hash",
    "load_study",
    "normalized_design",
    "parse_study",
]
