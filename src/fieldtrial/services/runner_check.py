"""Check a study's runner before a session: commands exist, policy servers answer.

Results name arms by blind code only, so checking does not unblind the operator.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from fieldtrial.design.runner_config import TemplateError, render_command, sample_values
from fieldtrial.runners.base import RunnerError
from fieldtrial.services._context import StudyContext, open_study
from fieldtrial.store import models as m


@dataclass(frozen=True, slots=True)
class RunnerCheck:
    """One check: ``subject`` is a blind code or ``router``."""

    subject: str
    ok: bool
    message: str


def _codes(ctx: StudyContext) -> dict[str, str]:
    with ctx.db() as db:
        return {
            a.key: a.blind_code
            for a in db.scalars(select(m.Arm).where(m.Arm.study_id == ctx.study_id))
        }


def check_runners(folder: str | Path, *, api_key: str | None = None) -> list[RunnerCheck]:
    """Check the study's runner. Manual and simulated studies need no checks."""
    with open_study(folder) as ctx:
        spec = ctx.spec
        codes = _codes(ctx)
        kinds = {a.runner for a in spec.arms}
        checks: list[RunnerCheck] = []
        if "command" in kinds:
            assert spec.runners is not None
            assert spec.runners.command is not None
            config = spec.runners.command
            cwd = Path(ctx.folder) / config.cwd if config.cwd else Path(ctx.folder)
            factors = spec.conditions.expand()[0]
            for arm in spec.arms:
                code = codes.get(arm.id, "?")
                try:
                    argv = render_command(
                        config.template,
                        sample_values(policy=arm.policy, serving=arm.serving, factors=factors),
                    )
                except TemplateError as exc:
                    checks.append(RunnerCheck(code, False, str(exc)))
                    continue
                exe = argv[0]
                found = shutil.which(exe) or (str(cwd / exe) if (cwd / exe).exists() else None)
                if found is None:
                    checks.append(
                        RunnerCheck(code, False, f"{exe!r} not found on PATH or in {cwd}")
                    )
                else:
                    checks.append(
                        RunnerCheck(code, True, f"runs {exe} ({len(argv) - 1} arguments)")
                    )
        if "openpi_router" in kinds:
            from fieldtrial.runners.openpi_router import probe

            frames: dict[str, bytes] = {}
            for arm in spec.arms:
                code = codes.get(arm.id, "?")
                try:
                    frames[code] = probe(str(arm.policy["url"]), api_key=api_key)
                except RunnerError as exc:
                    checks.append(RunnerCheck(code, False, str(exc)))
                else:
                    checks.append(RunnerCheck(code, True, "policy server answered"))
            if len(frames) == len(spec.arms) > 1:
                same = len(set(frames.values())) == 1
                checks.append(
                    RunnerCheck(
                        "router",
                        same
                        or (
                            spec.runners is not None
                            and spec.runners.openpi_router is not None
                            and spec.runners.openpi_router.metadata == "first"
                        ),
                        "the arms' metadata are identical"
                        if same
                        else "the arms' metadata differ: the robot could tell them apart",
                    )
                )
        return checks
