"""Analyzing a study: load its records from the database and run the analysis engine."""

from pathlib import Path

from sqlalchemy import select

from fieldtrial.analysis.engine import analyze
from fieldtrial.analysis.results import Results
from fieldtrial.services._context import ServiceError, open_study
from fieldtrial.services.trial import collect_records
from fieldtrial.store import models as m


def analyze_study(folder: str | Path) -> Results:
    """Analyze a locked study.

    While the study is blinded (``blinding: operator`` and not yet unblinded), per-arm
    results stay hidden: this raises ``ServiceError`` and asks for ``fieldtrial unblind``.
    """
    with open_study(folder) as ctx:
        with ctx.db() as db:
            study = db.get_one(m.Study, ctx.study_id)
            if ctx.spec.design.blinding == "operator" and study.unblinded_at is None:
                raise ServiceError(
                    "the study is blinded; per-arm results stay hidden until you run "
                    f"`fieldtrial unblind {ctx.folder}` (unblinding is logged)"
                )
            codes = {
                a.key: a.blind_code
                for a in db.scalars(select(m.Arm).where(m.Arm.study_id == ctx.study_id))
            }
        records, info = collect_records(ctx)
        return analyze(ctx.spec, records, info, blind_codes=codes)
