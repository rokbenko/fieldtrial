"""Calculator commands: work on plain counts, no study needed.

Every command prints a short human-readable summary, or JSON with ``--json``. Exit codes:
0 on success, 1 on unexpected errors, 2 on invalid input.
"""

import dataclasses
import json
import math
import re
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console

from fieldtrial.stats import (
    ADJUST_METHODS,
    INDEPENDENT_TESTS,
    INTERVAL_METHODS,
    POWER_METHODS,
    adjust_pvalues,
    compare_independent,
    compare_paired,
    mde,
    proportion_ci,
    sample_size,
)
from fieldtrial.stats._types import ALTERNATIVES, Alternative

_COUNT = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
_console = Console(highlight=False, soft_wrap=True)

JsonOption = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")]
LevelOption = Annotated[float, typer.Option("--level", help="Confidence level.")]
AlphaOption = Annotated[float, typer.Option("--alpha", help="Significance level.")]
AlternativeOption = Annotated[
    str,
    typer.Option(
        "--alternative", help=f"One of {', '.join(ALTERNATIVES)}; 'greater' means arm 1 higher."
    ),
]


def parse_count(text: str) -> tuple[int, int]:
    """Parse ``"36/40"`` into ``(36, 40)``."""
    match = _COUNT.match(text)
    if not match:
        raise typer.BadParameter(f"expected successes/trials such as 36/40, got {text!r}")
    k, n = int(match.group(1)), int(match.group(2))
    if n == 0 or k > n:
        raise typer.BadParameter(f"need 0 <= successes <= trials and trials >= 1, got {text!r}")
    return k, n


def _alternative(text: str) -> Alternative:
    if text not in ALTERNATIVES:
        raise typer.BadParameter(
            f"must be one of {', '.join(ALTERNATIVES)}, got {text!r}", param_hint="--alternative"
        )
    return text


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None  # JSON has no NaN or infinity
    return value


def _print_json(payload: Any) -> None:
    typer.echo(json.dumps(_jsonable(payload), indent=2, allow_nan=False))


def fmt_p(p: float) -> str:
    """Format a p-value with four decimals, or ``< 0.0001``."""
    if math.isnan(p):
        return "n/a"
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def fmt_pp(x: float) -> str:
    """Format a difference in proportions as signed percentage points."""
    value = round(x * 100, 1)
    if value == 0:
        return "0.0 pp"
    return f"{'+' if value > 0 else '−'}{abs(value):.1f} pp"


def _pct(level: float) -> str:
    return f"{level * 100:g}%"


def ci(
    count: Annotated[str, typer.Argument(help="Successes/trials, for example 36/40.")],
    method: Annotated[
        str, typer.Option("--method", help=f"One of {', '.join(INTERVAL_METHODS)}.")
    ] = "wilson",
    level: LevelOption = 0.95,
    as_json: JsonOption = False,
) -> None:
    """Confidence interval for one success rate."""
    k, n = parse_count(count)
    try:
        est = proportion_ci(k, n, level=level, method=method)
    except ValueError as exc:
        _fail(str(exc))
    if as_json:
        _print_json(est)
        return
    iv = est.interval
    name = method.replace("-", "–").title()  # clopper-pearson -> Clopper–Pearson
    _console.print(f"[bold]{k}/{n} = {est.estimate:.3f}[/bold]")
    _console.print(f"{name} {_pct(iv.level)} CI [{iv.low:.4f}, {iv.high:.4f}]")


def compare(
    arm1: Annotated[str, typer.Argument(help="Arm 1 successes/trials, for example 74/80.")],
    arm2: Annotated[str, typer.Argument(help="Arm 2 successes/trials, for example 91/120.")],
    test: Annotated[
        str, typer.Option("--test", help=f"Primary test: {', '.join(INDEPENDENT_TESTS)}.")
    ] = "boschloo",
    alternative: AlternativeOption = "two-sided",
    level: LevelOption = 0.95,
    alpha: AlphaOption = 0.05,
    as_json: JsonOption = False,
) -> None:
    """Compare two independent arms: difference, Newcombe CI and exact test."""
    k1, n1 = parse_count(arm1)
    k2, n2 = parse_count(arm2)
    try:
        res = compare_independent(
            k1, n1, k2, n2, test=test, alternative=_alternative(alternative), level=level
        )
    except ValueError as exc:
        _fail(str(exc))
    if as_json:
        _print_json(res)
        return
    a, b, iv = res.arm1, res.arm2, res.interval
    _console.print(
        f"Arm 1: {k1}/{n1} = {a.estimate:.1%} "
        f"({_pct(level)} CI {a.interval.low:.1%}–{a.interval.high:.1%})"
    )
    _console.print(
        f"Arm 2: {k2}/{n2} = {b.estimate:.1%} "
        f"({_pct(level)} CI {b.interval.low:.1%}–{b.interval.high:.1%})"
    )
    _console.print(
        f"[bold]Difference {fmt_pp(res.difference)}[/bold], "
        f"Newcombe {_pct(level)} CI [{fmt_pp(iv.low)}, {fmt_pp(iv.high)}]"
    )
    verdict = "rejects" if res.primary.rejects(alpha) else "does not reject"
    _console.print(
        f"{res.primary.test.title()} p = {fmt_p(res.primary.pvalue)} ({alternative}); "
        f"{verdict} H0 at α = {alpha:g}"
    )
    for other in res.secondary:
        _console.print(f"{other.test.title()} p = {fmt_p(other.pvalue)}")


def paired(
    b: Annotated[int, typer.Option("--b", help="Pairs where only arm 1 succeeded.")],
    c: Annotated[int, typer.Option("--c", help="Pairs where only arm 2 succeeded.")],
    n: Annotated[
        int | None, typer.Option("--n", help="Total pairs; adds the Tango CI when given.")
    ] = None,
    alternative: AlternativeOption = "two-sided",
    level: LevelOption = 0.95,
    as_json: JsonOption = False,
) -> None:
    """Paired comparison: exact McNemar test, plus the Tango CI when --n is given."""
    try:
        res = compare_paired(b, c, n, alternative=_alternative(alternative), level=level)
    except ValueError as exc:
        _fail(str(exc))
    if as_json:
        _print_json(res)
        return
    _console.print(f"Discordant pairs: b = {b}, c = {c}")
    if res.difference is not None and res.interval is not None:
        _console.print(
            f"[bold]Difference {fmt_pp(res.difference)}[/bold], Tango {_pct(level)} CI "
            f"[{fmt_pp(res.interval.low)}, {fmt_pp(res.interval.high)}]"
        )
    _console.print(f"Exact McNemar p = {fmt_p(res.test.pvalue)} ({alternative})")


def power_cmd(
    p1: Annotated[float, typer.Option("--p1", help="Success rate of arm 1 (e.g. baseline).")],
    p2: Annotated[float, typer.Option("--p2", help="Success rate of arm 2.")],
    alpha: AlphaOption = 0.05,
    power: Annotated[float, typer.Option("--power", help="Desired power.")] = 0.80,
    method: Annotated[
        str, typer.Option("--method", help=f"One of {', '.join(POWER_METHODS)}.")
    ] = "pooled-z",
    ratio: Annotated[float, typer.Option("--ratio", help="Allocation ratio n2/n1.")] = 1.0,
    alternative: AlternativeOption = "two-sided",
    as_json: JsonOption = False,
) -> None:
    """Trials per arm needed to detect a difference between two success rates."""
    try:
        res = sample_size(
            p1,
            p2,
            alpha=alpha,
            power=power,
            method=method,
            ratio=ratio,
            alternative=_alternative(alternative),
        )
    except ValueError as exc:
        _fail(str(exc))
    if as_json:
        _print_json(res)
        return
    if ratio == 1.0:
        _console.print(f"[bold]{res.n1} per arm[/bold] ({method})")
    else:
        _console.print(f"[bold]n1 = {res.n1}, n2 = {res.n2}[/bold] ({method}, ratio {ratio:g})")
    _console.print(
        f"Detects {p1:g} → {p2:g} ({fmt_pp(p2 - p1)}) with {power:.0%} power, "
        f"α = {alpha:g} ({alternative}); exact n1 = {res.n1_exact:.2f}"
    )


def mde_cmd(
    p1: Annotated[float, typer.Option("--p1", help="Baseline success rate.")],
    n: Annotated[int, typer.Option("--n", help="Trials in the baseline arm.")],
    n2: Annotated[
        int | None, typer.Option("--n2", help="Trials in the other arm (default: same as --n).")
    ] = None,
    alpha: AlphaOption = 0.05,
    power: Annotated[float, typer.Option("--power", help="Desired power.")] = 0.80,
    method: Annotated[
        str, typer.Option("--method", help=f"One of {', '.join(POWER_METHODS)}.")
    ] = "pooled-z",
    as_json: JsonOption = False,
) -> None:
    """Minimum detectable effect from a baseline rate with a given number of trials."""
    try:
        up = mde(p1, n, n2, alpha=alpha, power=power, direction="increase", method=method)
        down = mde(p1, n, n2, alpha=alpha, power=power, direction="decrease", method=method)
    except ValueError as exc:
        _fail(str(exc))
    if as_json:
        _print_json({"increase": up, "decrease": down})
        return

    def show(res: Any) -> str:
        if res.target is None:
            return "none"
        return f"{fmt_pp(res.effect)} (to {res.target:.4f})"

    _console.print(f"[bold]{show(up)} / {show(down)}[/bold]")
    _console.print(
        f"Baseline {p1:g}, n1 = {up.n1}, n2 = {up.n2}: {power:.0%} power, "
        f"α = {alpha:g} two-sided ({method})"
    )


def adjust(
    pvalues: Annotated[list[float], typer.Argument(help="Raw p-values.")],
    method: Annotated[
        str, typer.Option("--method", help=f"One of {', '.join(ADJUST_METHODS)}.")
    ] = "holm",
    alpha: AlphaOption = 0.05,
    as_json: JsonOption = False,
) -> None:
    """Adjust p-values for multiple comparisons."""
    try:
        res = adjust_pvalues(pvalues, method=method, alpha=alpha)
    except ValueError as exc:
        _fail(str(exc))
    if as_json:
        _print_json(res)
        return
    _console.print(f"{method} adjustment, α = {alpha:g}")
    for i, (raw, adj, rej) in enumerate(zip(res.raw, res.adjusted, res.reject, strict=True), 1):
        _console.print(f"{i:>3}  raw {raw:.5f}  adjusted {adj:.5f}  {'reject' if rej else '-'}")


def register(app: typer.Typer) -> None:
    """Add the calculator commands to the main app."""
    app.command("ci")(ci)
    app.command("compare")(compare)
    app.command("paired")(paired)
    app.command("power")(power_cmd)
    app.command("mde")(mde_cmd)
    app.command("adjust")(adjust)
