"""Report charts as inline SVG (docs/PLAN.md section 15).

matplotlib's object API is used directly (no ``pyplot`` global state), so charts can be drawn
from several threads. Colors follow the Okabe–Ito palette, which stays distinguishable
with the common forms of color blindness. Each function returns an ``<svg>`` element
ready to inline, or ``None`` when there is nothing to draw. SVG ids are fixed and no
timestamp is written, so the same results always give the same file.
"""

import io
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from fieldtrial.analysis.results import Results

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Okabe, M. and Ito, K. (2008). Color Universal Design. https://jfly.uni-koeln.de/color/
OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442")
INK = "#1d2430"
GRID = "#d5dae1"
WIDTH = 7.0  # inches; SVGs scale to the page width
_DOCTYPE = re.compile(r"<\?xml[^>]*\?>\s*|<!DOCTYPE[^>]*>\s*", re.DOTALL)
_METADATA = re.compile(r"\s*<metadata>.*?</metadata>", re.DOTALL)


def arm_colors(arms: Sequence[str]) -> dict[str, str]:
    """A fixed color per arm, in design order."""
    return {arm: OKABE_ITO[i % len(OKABE_ITO)] for i, arm in enumerate(arms)}


def _figure(height: float) -> "Figure":
    from matplotlib.figure import Figure

    fig = Figure(figsize=(WIDTH, height), layout="constrained")
    fig.patch.set_alpha(0)
    return fig


def _style(ax: "Axes") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.set_facecolor("none")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _svg(fig: "Figure", title: str) -> str:
    import matplotlib

    buffer = io.StringIO()
    with matplotlib.rc_context(
        {"svg.fonttype": "none", "svg.hashsalt": "fieldtrial", "font.family": "sans-serif"}
    ):
        fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": None})
    svg = _METADATA.sub("", _DOCTYPE.sub("", buffer.getvalue())).strip()
    # Accessible name for screen readers; the figure's own title is drawn in the image.
    return svg.replace("<svg ", f'<svg role="img" aria-label="{_escape(title)}" ', 1)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _pct(ax: "Axes", axis: str = "x") -> None:
    from matplotlib.ticker import PercentFormatter

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(PercentFormatter(1.0, decimals=0))


# --- charts --------------------------------------------------------------------------------------


def success_rates(results: Results) -> str | None:
    """Success rate with its confidence interval, one row per arm."""
    arms = [a for a in results.arms if a.rate is not None and a.ci is not None]
    if not arms:
        return None
    colors = arm_colors(results.study.arms)
    fig = _figure(0.55 * len(arms) + 0.9)
    ax = fig.add_subplot()
    _style(ax)
    for y, arm in enumerate(reversed(arms)):
        assert arm.rate is not None
        assert arm.ci is not None
        ax.errorbar(
            arm.rate,
            y,
            xerr=[[arm.rate - arm.ci.low], [arm.ci.high - arm.rate]],
            fmt="o",
            color=colors[arm.arm],
            ecolor=colors[arm.arm],
            elinewidth=2.5,
            capsize=4,
            markersize=7,
        )
        ax.annotate(
            f"{arm.successes}/{arm.completed}",
            (arm.ci.high, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(range(len(arms)), [a.arm for a in reversed(arms)])
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.set_xlim(0, 1.08)
    _pct(ax)
    ax.set_xlabel(f"Success rate with {round(arms[0].ci.level * 100)}% CI")  # type: ignore[union-attr]
    ax.grid(axis="y", visible=False)
    return _svg(fig, "Success rate per arm with confidence intervals")


def stage_stacks(results: Results) -> str | None:
    """Stacked bars of the furthest stage reached, per arm."""
    stages = [s for s in results.stages if s.n]
    if not stages:
        return None
    names: list[str | None] = [None, *results.study.stages]
    fig = _figure(0.55 * len(stages) + 1.4)
    ax = fig.add_subplot()
    _style(ax)
    shades = ["#e3e6ea", *_ramp(len(results.study.stages))]
    for y, summary in enumerate(reversed(stages)):
        left = 0.0
        for i, name in enumerate(names):
            share = next((d.share for d in summary.distribution if d.stage == name), 0.0)
            ax.barh(y, share, left=left, color=shades[i], edgecolor="white", height=0.6)
            left += share
    ax.set_yticks(range(len(stages)), [s.arm for s in reversed(stages)])
    ax.set_xlim(0, 1)
    _pct(ax)
    ax.set_xlabel("Share of trials by furthest stage reached")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    labels = ["none", *results.study.stages]
    ax.legend(
        [Patch(color=c) for c in shades],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncols=min(len(labels), 5),
        frameon=False,
        fontsize=8,
    )
    return _svg(fig, "Furthest stage reached per arm")


def _ramp(n: int) -> list[str]:
    # Light to dark blue: later stages darker, the success stage darkest.
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    cmap = LinearSegmentedColormap.from_list("stages", ["#bcd7ee", "#0b3c6b"])
    return [to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def stage_funnel(results: Results) -> str | None:
    """Share of trials that reached each stage, per arm."""
    stages = [s for s in results.stages if s.n]
    if not stages:
        return None
    colors = arm_colors(results.study.arms)
    fig = _figure(2.8)
    ax = fig.add_subplot()
    _style(ax)
    x = list(range(len(results.study.stages)))
    for summary in stages:
        ax.plot(
            x,
            [step.cumulative for step in summary.funnel],
            marker="o",
            color=colors[summary.arm],
            linewidth=2,
            label=summary.arm,
        )
    ax.set_xticks(x, results.study.stages)
    ax.set_ylim(0, 1.03)
    _pct(ax, "y")
    ax.set_ylabel("Reached the stage")
    ax.axvline(
        results.study.stages.index(results.study.success_stage),
        color=INK,
        linestyle=":",
        linewidth=1,
    )
    ax.legend(frameon=False, fontsize=9)
    return _svg(fig, "Stage funnel: share of trials reaching each stage")


def time_curves(results: Results) -> str | None:
    """Cumulative share of trials that had succeeded by each time."""
    curves = [t for t in results.timing if t.curve_times]
    if not curves:
        return None
    colors = arm_colors(results.study.arms)
    fig = _figure(2.8)
    ax = fig.add_subplot()
    _style(ax)
    for timing in curves:
        times = [0.0, *timing.curve_times]
        fraction = [0.0, *timing.curve_fraction]
        ax.step(
            times, fraction, where="post", color=colors[timing.arm], linewidth=2, label=timing.arm
        )
    ax.set_ylim(0, 1.03)
    ax.set_xlim(left=0)
    _pct(ax, "y")
    ax.set_xlabel("Seconds since the start of the trial")
    ax.set_ylabel("Succeeded by then")
    ax.legend(frameon=False, fontsize=9)
    return _svg(fig, "Cumulative success over time per arm")


def forest(results: Results) -> str | None:
    """Differences against the control with confidence intervals (primary and sensitivity)."""
    primary = results.primary
    kind = {"group_sequential": "repeated CI", "anytime": "confidence sequence"}.get(
        primary.method, "paired"
    )
    rows: list[tuple[str, float, float, float, str]] = [
        (f"{w.treatment} vs {w.control} ({kind})", w.difference, w.ci.low, w.ci.high, INK)
        for w in primary.pairwise
        if w.difference is not None and w.ci is not None
    ]
    if primary.method == "crossover" and primary.estimate is not None and primary.ci is not None:
        rows.append(
            (
                f"{primary.treatment} vs {primary.control} (crossover)",
                primary.estimate,
                primary.ci.low,
                primary.ci.high,
                INK,
            )
        )
    rows += [
        (
            f"{c.treatment} vs {c.control} (independent)",
            c.difference,
            c.ci.low,
            c.ci.high,
            "#6b7280",
        )
        for c in results.sensitivity
    ]
    if not rows:
        return None
    fig = _figure(0.5 * len(rows) + 1.0)
    ax = fig.add_subplot()
    _style(ax)
    for y, (_label, diff, low, high, color) in enumerate(reversed(rows)):
        ax.errorbar(
            diff,
            y,
            xerr=[[diff - low], [high - diff]],
            fmt="s",
            color=color,
            capsize=4,
            elinewidth=2,
        )
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    limit = max(0.05, *(abs(v) for r in rows for v in r[2:4])) * 1.1
    ax.set_xlim(-limit, limit)
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _pos: "0 pp" if round(v * 100) == 0 else f"{v * 100:+.0f} pp")
    )
    ax.set_xlabel("Difference in success rate (95% CI)")
    ax.grid(axis="y", visible=False)
    return _svg(fig, "Forest plot of differences against the control")


def condition_heatmap(results: Results) -> str | None:
    """Success rate per condition and arm."""
    rows = [r for r in results.conditions if any(c.completed for c in r.cells)]
    if not rows:
        return None
    import numpy as np

    arms = results.study.arms
    data = np.full((len(rows), len(arms)), np.nan)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row.cells):
            if cell.completed:
                data[i, j] = cell.successes / cell.completed
    height = min(0.22 * len(rows) + 1.2, 14.0)
    fig = _figure(height)
    ax = fig.add_subplot()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "rate", ["#D55E00", "#f7f7f7", "#0072B2"]
    ).with_extremes(bad="#e3e6ea")
    image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(arms)), arms)
    ax.xaxis.tick_top()
    step = max(1, len(rows) // 40)
    ax.set_yticks(range(0, len(rows), step), [rows[i].condition for i in range(0, len(rows), step)])
    ax.tick_params(colors=INK, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=0.6, pad=0.02)
    bar.ax.tick_params(labelsize=8, colors=INK)
    from matplotlib.ticker import PercentFormatter

    bar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    return _svg(fig, "Success rate per condition and arm")


def session_trend(results: Results) -> str | None:
    """Success rate per session, in the order the sessions started."""
    sessions = [s for s in results.sessions if any(c.completed for c in s.cells)]
    if len(sessions) < 2:
        return None
    colors = arm_colors(results.study.arms)
    fig = _figure(2.6)
    ax = fig.add_subplot()
    _style(ax)
    x = list(range(1, len(sessions) + 1))
    for j, arm in enumerate(results.study.arms):
        points: list[tuple[int, float, int]] = []
        for i, session in enumerate(sessions, 1):
            cell = session.cells[j]
            if cell.completed:
                points.append((i, cell.successes / cell.completed, cell.completed))
        if points:
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                color=colors[arm],
                linewidth=1.8,
                label=arm,
            )
    ax.set_xticks(x, [f"{s.started_at:%m-%d %H:%M}\n{s.operator}" for s in sessions], fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    _pct(ax, "y")
    ax.set_ylabel("Success rate in the session")
    ax.legend(frameon=False, fontsize=9)
    return _svg(fig, "Success rate per session over time")


def ladder_rates(results: Results) -> str | None:
    """Success rate with its interval at each checkpoint of a ladder, in step order."""
    ladder = results.ladder
    if ladder is None:
        return None
    by_arm = {a.arm: a for a in results.arms}
    points = [
        (i, by_arm[arm])
        for i, arm in enumerate(ladder.arms)
        if by_arm[arm].rate is not None and by_arm[arm].ci is not None
    ]
    if len(points) < 2:
        return None
    fig = _figure(2.8)
    ax = fig.add_subplot()
    _style(ax)
    xs = [i for i, _ in points]
    rates = [float(a.rate or 0.0) for _, a in points]
    ax.plot(xs, rates, color=OKABE_ITO[0], linewidth=1.5, zorder=1)
    for i, arm in points:
        assert arm.rate is not None
        assert arm.ci is not None
        ax.errorbar(
            i,
            arm.rate,
            yerr=[[arm.rate - arm.ci.low], [arm.ci.high - arm.rate]],
            fmt="o",
            color=OKABE_ITO[0],
            elinewidth=2.5,
            capsize=4,
            markersize=7,
            zorder=2,
        )
    if ladder.plateau_arm is not None:
        start = ladder.arms.index(ladder.plateau_arm)
        ax.axvspan(start - 0.3, len(ladder.arms) - 0.7, color=GRID, alpha=0.5, zorder=0)
    labels = [f"{arm}\n{step:g}" for arm, step in zip(ladder.arms, ladder.steps, strict=True)]
    ax.set_xticks(range(len(ladder.arms)), labels, fontsize=8)
    ax.set_xlim(-0.5, len(ladder.arms) - 0.5)
    ax.set_ylim(-0.03, 1.03)
    _pct(ax, "y")
    ax.set_ylabel("Success rate (95% CI)")
    ax.grid(axis="x", visible=False)
    return _svg(fig, "Success rate at each checkpoint in training-step order")


def crossover_rounds(results: Results) -> str | None:
    """Success rate of every crossover round, colored by arm."""
    crossover = results.crossover
    if crossover is None:
        return None
    rounds = [r for r in crossover.rounds if r.rate is not None]
    if not rounds:
        return None
    colors = arm_colors(results.study.arms)
    fig = _figure(2.6)
    ax = fig.add_subplot()
    _style(ax)
    for arm in (crossover.treatment, crossover.control):
        pts = [r for r in rounds if r.arm == arm]
        ax.plot(
            [r.round for r in pts],
            [float(r.rate or 0.0) for r in pts],
            marker="o",
            linestyle="none",
            color=colors[arm],
            markersize=7,
            label=arm,
        )
    for cycle in range(1, crossover.cycles + 1):
        pair = [r for r in rounds if r.cycle == cycle]
        if len(pair) == 2:
            ax.plot(
                [p.round for p in pair],
                [float(p.rate or 0.0) for p in pair],
                color=GRID,
                zorder=0,
            )
    ax.set_xticks(range(1, 2 * crossover.cycles + 1))
    ax.set_ylim(-0.03, 1.03)
    _pct(ax, "y")
    ax.set_xlabel("Round (lines join the two rounds of a cycle)")
    ax.set_ylabel("Success rate in the round")
    ax.legend(frameon=False, fontsize=9)
    return _svg(fig, "Success rate per crossover round")


def _pp_axis(ax: "Axes", axis: str = "y") -> None:
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _pos: "0 pp" if round(v * 100) == 0 else f"{v * 100:+.0f} pp")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def confidence_sequence(results: Results) -> str | None:
    """The anytime-valid confidence sequence for the paired difference, block by block."""
    anytime = results.anytime
    if anytime is None or not anytime.sequence:
        return None
    blocks = [row.blocks for row in anytime.sequence]
    fig = _figure(2.6)
    ax = fig.add_subplot()
    _style(ax)
    ax.fill_between(
        blocks,
        [row.low for row in anytime.sequence],
        [row.high for row in anytime.sequence],
        step="post",
        color=GRID,
        label="Confidence sequence",
    )
    ax.axhline(0, color=INK, linewidth=1)
    if anytime.stopped_at:
        ax.axvline(anytime.stopped_at, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_ylim(-1.02, 1.02)
    _pp_axis(ax)
    ax.set_xlabel("Complete blocks")
    ax.set_ylabel("Difference in success rate")
    return _svg(fig, "Anytime-valid confidence sequence by block")


def selection_pairs(results: Results) -> str | None:
    """Confidence sequences for every pair of arms at the end of best-arm selection."""
    selection = results.selection
    if selection is None or not selection.pairs:
        return None
    rows = [p for p in selection.pairs if p.blocks]
    if not rows:
        return None
    fig = _figure(0.45 * len(rows) + 1.0)
    ax = fig.add_subplot()
    _style(ax)
    for y, pair in enumerate(reversed(rows)):
        mid = pair.estimate if pair.estimate is not None else (pair.low + pair.high) / 2
        ax.errorbar(
            mid,
            y,
            xerr=[[mid - pair.low], [pair.high - mid]],
            fmt="s",
            color=INK,
            capsize=4,
            elinewidth=2,
        )
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(range(len(rows)), [f"{p.first} − {p.second}" for p in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-1.05, 1.05)
    _pp_axis(ax, "x")
    ax.set_xlabel("Difference in success rate (confidence sequence)")
    ax.grid(axis="y", visible=False)
    return _svg(fig, "Pairwise confidence sequences of best-arm selection")


CHARTS: tuple[tuple[str, str, Any], ...] = (
    ("rates", "Success rate per arm", success_rates),
    ("forest", "Differences against the control", forest),
    ("confseq", "Anytime-valid confidence sequence", confidence_sequence),
    ("pairs", "Best-arm selection pairs", selection_pairs),
    ("ladder", "Success by training step", ladder_rates),
    ("rounds", "Success per crossover round", crossover_rounds),
    ("stages", "Furthest stage reached", stage_stacks),
    ("funnel", "Stage funnel", stage_funnel),
    ("timing", "Time to success", time_curves),
    ("conditions", "Success per condition", condition_heatmap),
    ("sessions", "Success per session", session_trend),
)


def all_charts(results: Results) -> dict[str, str]:
    """Every chart that has data, by key."""
    out = {}
    for key, _title, draw in CHARTS:
        svg = draw(results)
        if svg is not None:
            out[key] = svg
    return out
