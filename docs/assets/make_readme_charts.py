"""Draw the two data charts in README.md with fieldtrial's own statistics.

    uv run python docs/assets/make_readme_charts.py

- ``rollouts.png``: how wide a 95% interval is for an observed 80% success rate, and how
  many rollouts per arm a difference needs (Wilson intervals; pooled-z sample sizes).
- ``dream-machines.png``: every comparison in the re-analysis of Dream Machines' published
  pi0.5 results (examples/dream-machines-pi05), resolved or not after Holm's adjustment.
"""

import sys
from pathlib import Path

from matplotlib.figure import Figure

from fieldtrial.stats import proportion_ci, sample_size

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "examples" / "dream-machines-pi05"))

import reanalysis  # noqa: E402

BLUE, ORANGE, GREY, INK, GRID = "#0072B2", "#E69F00", "#9aa3ad", "#1d2430", "#d5dae1"


def style(ax) -> None:  # type: ignore[no-untyped-def]
    """The report charts' look: no top or right spine, a light grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def rollouts() -> None:
    """Interval width by rollouts, and rollouts needed per improvement."""
    fig = Figure(figsize=(10, 3.6), layout="constrained", facecolor="white")
    left, right = fig.subplots(1, 2)
    ns = list(range(10, 201, 2))
    lows, highs = [], []
    for n in ns:
        ci = proportion_ci(round(0.8 * n), n).interval
        lows.append(ci.low * 100)
        highs.append(ci.high * 100)
    left.fill_between(ns, lows, highs, color=BLUE, alpha=0.18, linewidth=0)
    left.plot(ns, [80] * len(ns), color=BLUE, linewidth=1.5)
    for n in (20, 40, 100):
        ci = proportion_ci(round(0.8 * n), n).interval
        left.plot([n, n], [ci.low * 100, ci.high * 100], color=BLUE, linewidth=2)
        left.annotate(
            f"{round(0.8 * n)}/{n}: {ci.low:.0%}–{ci.high:.0%}",
            (n, ci.low * 100),
            xytext=(4, -14),
            textcoords="offset points",
            fontsize=9,
            color=INK,
        )
    left.set_ylim(40, 100)
    left.set_xlim(0, 200)
    left.set_xlabel("Rollouts", color=INK)
    left.set_ylabel("Success rate (%)", color=INK)
    left.set_title("An observed 80%: what the true rate could be (95%)", fontsize=11, color=INK)
    style(left)

    gains = [5, 10, 15, 20]
    for base, color in ((0.6, ORANGE), (0.75, BLUE)):
        needed = [sample_size(base, base + g / 100).n1 for g in gains]
        right.plot(gains, needed, marker="o", color=color, label=f"from {base:.0%}")
        for g, n in zip(gains, needed, strict=True):
            right.annotate(f"{n}", (g, n), xytext=(5, 4), textcoords="offset points", fontsize=8)
    right.set_yscale("log")
    right.set_xticks(gains, [f"+{g} pp" for g in gains])
    right.set_xlabel("True improvement", color=INK)
    right.set_ylabel("Rollouts per arm (80% power)", color=INK)
    right.set_title("Rollouts each arm needs to show it", fontsize=11, color=INK)
    right.legend(frameon=False, fontsize=9)
    style(right)
    fig.savefig(HERE / "rollouts.png", dpi=160)


def dream_machines() -> None:
    """Every comparison of the Dream Machines re-analysis as a forest plot."""
    experiments = reanalysis.load(
        HERE.parent.parent / "examples" / "dream-machines-pi05" / "published_counts.csv"
    )
    rows = []
    for experiment, arms in experiments.items():
        if experiment not in reanalysis.REFERENCE:
            continue
        found = reanalysis.compare(
            experiment, reanalysis.pairs(arms, reanalysis.REFERENCE[experiment])
        )
        title = reanalysis.TITLES.get(experiment, experiment)
        rows += [(title, c) for c in found]
    best = experiments["serving_best_on_21h_plus_1h"][0]
    default = next(a for a in experiments["data_quality"] if a.name == "21h_plus_1h_hq")
    rows += [("Best serving vs default", c) for c in reanalysis.compare("cross", [(best, default)])]

    fig = Figure(figsize=(9, 0.27 * len(rows) + 1.4), layout="constrained", facecolor="white")
    ax = fig.subplots()
    for i, (_, c) in enumerate(rows):
        y = len(rows) - i
        color = BLUE if c.resolved else GREY
        ax.plot([c.low * 100, c.high * 100], [y, y], color=color, linewidth=2.2)
        ax.plot(
            c.difference * 100,
            y,
            marker="o",
            markersize=6,
            color=color,
            markerfacecolor=color if c.resolved else "white",
        )
    ax.axvline(0, color=INK, linewidth=1)
    labels = [f"{c.arm.name} vs {c.reference.name}" for _, c in rows]
    labels = [
        f"{label} ({title.split(', ')[-1]})" if labels.count(label) > 1 else label
        for label, (title, _) in zip(labels, rows, strict=True)
    ]
    ax.set_yticks([len(rows) - i for i in range(len(rows))], labels, fontsize=8)
    ax.set_xlabel("Difference in success rate, percentage points (95% CI)", color=INK)
    resolved = sum(c.resolved for _, c in rows)
    fig.suptitle(
        f"{resolved} of {len(rows)} published comparisons resolve (filled); "
        "the rest could be noise (hollow)",
        fontsize=11,
        color=INK,
    )
    style(ax)
    fig.savefig(HERE / "dream-machines.png", dpi=160)


if __name__ == "__main__":
    rollouts()
    dream_machines()
    print("wrote rollouts.png and dream-machines.png")
