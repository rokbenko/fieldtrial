"""Study commands: design, lock, fill, analyze.

Exit codes: 0 on success, 1 when the study's state does not allow the request (for example
analyzing a blinded study), 2 when the input is invalid (``study.yaml``, a CSV, an option).
"""

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, NoReturn, ParamSpec, TypeVar

import typer
from rich.console import Console

from fieldtrial.analysis import wording
from fieldtrial.analysis.wording import fmt_pp, fmt_rate
from fieldtrial.cli.calc import JsonOption, _jsonable, _print_json
from fieldtrial.design import StudyValidationError
from fieldtrial.io.csv import CsvImportError, parse_mapping
from fieldtrial.report import render_html, render_markdown
from fieldtrial.services import ServiceError
from fieldtrial.services.analysis import analyze_study
from fieldtrial.services.interim import run_interim
from fieldtrial.services.runner_check import check_runners
from fieldtrial.services.simulate import simulate_study
from fieldtrial.services.study import (
    amend_study,
    init_study,
    lock_study,
    plan_study,
    study_status,
    unblind_study,
    validate_study,
)
from fieldtrial.services.transfer import export_trials, import_trials
from fieldtrial.templates import TEMPLATES

_console = Console(highlight=False, soft_wrap=True)
P = ParamSpec("P")
R = TypeVar("R")

FolderArg = Annotated[
    Path, typer.Argument(help="Study folder (or its study.yaml).", show_default=False)
]


def _exit(code: int, message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=code)


def _errors(func: Callable[P, R]) -> Callable[P, R]:
    """Map service and validation errors to exit codes 1 and 2."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except StudyValidationError as exc:
            for issue in exc.issues:
                typer.echo(f"  {issue}", err=True)
            _exit(2, f"study.yaml has {len(exc.issues)} problem(s)")
        except CsvImportError as exc:
            for problem in exc.problems:
                typer.echo(f"  {problem}", err=True)
            _exit(2, "cannot import the CSV")
        except (ServiceError, FileNotFoundError, FileExistsError) as exc:
            _exit(1, str(exc))
        except ValueError as exc:
            _exit(2, str(exc))

    return wrapper


def _parse_rates(text: str) -> dict[str, float]:
    rates: dict[str, float] = {}
    for part in text.split(","):
        arm, sep, value = part.partition("=")
        try:
            rate = float(value)
        except ValueError:
            rate = -1.0
        if not sep or not arm.strip() or not 0.0 <= rate <= 1.0:
            raise typer.BadParameter(
                f"expected arm=rate with rate in [0, 1], got {part!r}", param_hint="--rates"
            )
        rates[arm.strip()] = rate
    return rates


@_errors
def init(
    folder: FolderArg,
    template: Annotated[
        str, typer.Option("--template", help=f"One of {', '.join(TEMPLATES)}.")
    ] = "basic",
    name: Annotated[
        str | None, typer.Option("--name", help="Study name (default: folder name).")
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Create a study folder with a study.yaml to edit."""
    path = init_study(folder, template=template, name=name)
    if as_json:
        _print_json({"study_yaml": str(path)})
        return
    _console.print(f"Created {path}")
    _console.print(f"Edit it, then run: fieldtrial validate {folder} && fieldtrial plan {folder}")


@_errors
def validate(folder: FolderArg, as_json: JsonOption = False) -> None:
    """Check study.yaml and report every problem with its line."""
    loaded = validate_study(folder)
    if as_json:
        _print_json({"valid": True, "name": loaded.spec.name, "warnings": list(loaded.warnings)})
        return
    _console.print(f"[green]study.yaml is valid[/green] ({loaded.spec.name})")
    for warning in loaded.warnings:
        _console.print(f"[yellow]warning:[/yellow] {warning}")


@_errors
def plan(
    folder: FolderArg,
    baseline: Annotated[
        float, typer.Option("--baseline", help="Expected control success rate, for the MDE.")
    ] = 0.5,
    as_json: JsonOption = False,
) -> None:
    """Validate, preview the schedule, and show what the planned trials can detect."""
    summary = plan_study(folder, baseline=baseline)
    if as_json:
        _print_json(
            {
                "design_hash": summary.design_hash,
                "conditions": summary.n_conditions,
                "trials_per_arm": summary.trials_per_arm,
                "trials": len(summary.schedule),
                "baseline": summary.baseline,
                "detectable_increase": summary.detectable_increase,
                "detectable_decrease": summary.detectable_decrease,
                "warnings": list(summary.warnings),
                "schedule": [
                    {"seq": s.seq, "block": s.block, "condition": s.condition, "arm": s.arm}
                    for s in summary.schedule
                ],
            }
        )
        return
    spec = summary.spec
    _console.print(f"[bold]{spec.title or spec.name}[/bold]  design {summary.design_hash[:12]}")
    _console.print(
        f"{summary.n_conditions} conditions × {spec.conditions.replicates} replicate(s); "
        f"{len(summary.schedule)} trials: "
        + ", ".join(f"{a} {n}" for a, n in summary.trials_per_arm.items())
    )
    if summary.detectable_increase is not None or summary.detectable_decrease is not None:
        up = summary.detectable_increase
        down = summary.detectable_decrease
        _console.print(
            f"From a {fmt_rate(summary.baseline)} baseline, 80% power at α = "
            f"{spec.analysis.primary.alpha:g} for changes of at least "
            f"{fmt_pp(up) if up is not None else 'n/a'} / "
            f"{fmt_pp(down) if down is not None else 'n/a'} (independent-samples formula; "
            "paired designs usually do better)."
        )
    _console.print("First trials:")
    for s in summary.schedule[:10]:
        _console.print(f"  {s.seq:>4}  block {s.block:>3}  {s.condition:<24} {s.arm}")
    if len(summary.schedule) > 10:
        _console.print(f"  ... {len(summary.schedule) - 10} more")
    for warning in summary.warnings:
        _console.print(f"[yellow]warning:[/yellow] {warning}")


@_errors
def lock(folder: FolderArg, as_json: JsonOption = False) -> None:
    """Freeze the design: store it, its hash and the randomized schedule."""
    result = lock_study(folder)
    if as_json:
        _print_json(result)
        return
    _console.print(f"Locked: design {result.design_hash[:12]}, {result.slots} trials scheduled")


@_errors
def amend(
    folder: FolderArg,
    reason: Annotated[str, typer.Option("--reason", help="Why the design changed (logged).")],
    as_json: JsonOption = False,
) -> None:
    """Apply edits made to study.yaml after locking, as a logged amendment."""
    result = amend_study(folder, reason)
    if as_json:
        _print_json(result)
        return
    _console.print(
        f"Amended: design {result.old_hash[:12]} → {result.new_hash[:12]}; "
        f"{result.added_slots} trials added, {result.voided_slots} removed. "
        "Reports list this amendment."
    )


@_errors
def status(folder: FolderArg, as_json: JsonOption = False) -> None:
    """Progress. While blinded, no per-arm results are shown."""
    report = study_status(folder)
    if as_json:
        payload = _jsonable(report)
        for key in ("locked_at", "unblinded_at"):
            payload[key] = None if payload[key] is None else payload[key].isoformat()
        _print_json(payload)
        return
    _console.print(f"[bold]{report.name}[/bold]  {report.status}  design {report.design_hash[:12]}")
    _console.print(
        f"{report.done}/{report.planned - report.void} trials done, {report.pending} pending, "
        f"{report.invalid_trials} invalid, {report.sessions} sessions, "
        f"{report.amendments} amendments"
    )
    if report.blinded:
        _console.print("Blinded: per-arm results are hidden until `fieldtrial unblind`.")
    elif report.per_arm:
        for arm, (k, n) in report.per_arm.items():
            _console.print(f"  {arm:<16} {k}/{n}" + (f" = {fmt_rate(k / n)}" if n else ""))


@_errors
def unblind(
    folder: FolderArg,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")] = False,
    as_json: JsonOption = False,
) -> None:
    """Reveal which blind code is which arm. Logged; reports flag early unblinding."""
    if not yes and not as_json:
        typer.confirm("Unblinding is logged and cannot be undone. Continue?", abort=True)
    codes = unblind_study(folder)
    if as_json:
        _print_json(codes)
        return
    for code, arm in sorted(codes.items()):
        _console.print(f"  {code}  {arm}")


@_errors
def simulate(
    folder: FolderArg,
    rates: Annotated[
        str, typer.Option("--rates", help="True success rate per arm: baseline=0.76,q50=0.90.")
    ],
    seed: Annotated[int, typer.Option("--seed", help="Simulation seed.")] = 1,
    invalid_rate: Annotated[
        float, typer.Option("--invalid-rate", help="Share of trials voided as robot faults.")
    ] = 0.0,
    max_trials: Annotated[
        int | None, typer.Option("--max-trials", help="Stop after this many trials.")
    ] = None,
    interim_looks: Annotated[
        bool,
        typer.Option(
            "--interim/--no-interim",
            help="Run planned interim looks as they come due (group-sequential studies).",
        ),
    ] = True,
    as_json: JsonOption = False,
) -> None:
    """Fill a locked study with simulated trials (sim runner + auto-operator)."""
    result = simulate_study(
        folder,
        _parse_rates(rates),
        seed=seed,
        invalid_rate=invalid_rate,
        max_trials=max_trials,
        interim=interim_looks,
    )
    if as_json:
        _print_json(result)
        return
    _console.print(
        f"Simulated {result.completed} completed and {result.invalid} invalid trials "
        f"in {result.sessions} sessions"
    )
    if result.interim_looks:
        stop = f"; stopped at look {result.stopped_at}" if result.stopped_at else ""
        _console.print(f"Ran {result.interim_looks} interim look(s){stop}")


@_errors
def import_cmd(
    folder: FolderArg,
    csv_path: Annotated[Path, typer.Argument(help="CSV file with one trial per row.")],
    mapping: Annotated[
        str | None,
        typer.Option("--map", help="Rename columns: field=column,... (e.g. arm=policy)."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Import trials from a CSV. All rows are checked first; nothing is written on error."""
    result = import_trials(folder, csv_path, mapping=parse_mapping(mapping))
    if as_json:
        _print_json(result)
        return
    _console.print(
        f"Imported {result.completed} completed and {result.invalid} invalid trials "
        f"in {result.sessions} sessions"
    )


@_errors
def export(
    folder: FolderArg,
    fmt: Annotated[str, typer.Option("--format", help="csv or jsonl.")] = "csv",
    out: Annotated[Path | None, typer.Option("--out", help="Output file.")] = None,
    as_json: JsonOption = False,
) -> None:
    """Export all trials. While blinded, arms appear as blind codes."""
    if fmt not in ("csv", "jsonl"):
        raise typer.BadParameter("must be csv or jsonl", param_hint="--format")
    target = out or Path(folder) / f"trials.{fmt}"
    count = export_trials(folder, target, fmt=fmt)  # type: ignore[arg-type]
    if as_json:
        _print_json({"rows": count, "path": str(target)})
        return
    _console.print(f"Wrote {count} trials to {target}")


@_errors
def interim(folder: FolderArg, as_json: JsonOption = False) -> None:
    """Run the next planned interim look of a group-sequential study.

    Prints only "continue" or "stop", so the study stays blinded. A stop voids the
    remaining trials; unblind and report as usual.
    """
    result = run_interim(folder)
    if as_json:
        _print_json(result)
        return
    _console.print(wording.interim(result.decision, result.look, result.planned_looks))
    if result.voided_slots:
        _console.print(f"{result.voided_slots} pending trials were cancelled.")


@_errors
def check_runners_cmd(
    folder: FolderArg,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="OPENPI_API_KEY", help="Api-Key for the policy servers."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Check the study's runner: commands exist, policy servers answer with the same metadata.

    Arms are named by blind code only, so checking does not unblind anyone.
    """
    checks = check_runners(folder, api_key=api_key)
    if as_json:
        _print_json(checks)
    elif not checks:
        _console.print("This study's arms use the manual or sim runner; nothing to check.")
    for c in checks:
        if not as_json:
            mark = "[green]ok[/green]" if c.ok else "[red]failed[/red]"
            _console.print(f"  {c.subject:8} {mark}  {c.message}")
    if any(not c.ok for c in checks):
        raise typer.Exit(1)


@_errors
def analyze(folder: FolderArg, as_json: JsonOption = False) -> None:
    """Run the pre-registered analysis and print the summary."""
    results = analyze_study(folder)
    if as_json:
        typer.echo(results.model_dump_json(indent=2))
        return
    for sentence in results.summary:
        _console.print(f"• {sentence}")
    for deviation in results.deviations:
        _console.print(f"[yellow]deviation:[/yellow] {deviation.message}")


@_errors
def report(
    folder: FolderArg,
    fmt: Annotated[str, typer.Option("--format", help="html, md or json.")] = "html",
    out: Annotated[Path | None, typer.Option("--out", help="Output file.")] = None,
) -> None:
    """Write the report to the study's reports/ folder (or --out)."""
    if fmt not in ("html", "md", "json"):
        raise typer.BadParameter("must be html, md or json", param_hint="--format")
    results = analyze_study(folder)
    root = Path(folder).parent if Path(folder).name == "study.yaml" else Path(folder)
    names = {"html": "report.html", "md": "report.md", "json": "results.json"}
    target = out or root / "reports" / names[fmt]
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "html":
        target.write_text(render_html(results), encoding="utf-8")
    elif fmt == "md":
        target.write_text(render_markdown(results), encoding="utf-8")
    else:
        target.write_text(
            json.dumps(_jsonable(results.model_dump(mode="json")), indent=2) + "\n",
            encoding="utf-8",
        )
    _console.print(f"Wrote {target}")


def _register_all(app: typer.Typer, commands: dict[str, Callable[..., Any]]) -> None:
    for name, command in commands.items():
        app.command(name)(command)


def register(app: typer.Typer) -> None:
    """Add the study commands to the main app."""
    _register_all(
        app,
        {
            "init": init,
            "validate": validate,
            "plan": plan,
            "lock": lock,
            "amend": amend,
            "status": status,
            "unblind": unblind,
            "simulate": simulate,
            "import": import_cmd,
            "export": export,
            "interim": interim,
            "check-runners": check_runners_cmd,
            "analyze": analyze,
            "report": report,
        },
    )
