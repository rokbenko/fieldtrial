"""Re-analyze the success counts Dream Machines published for fine-tuning pi0.5.

Source: Dream Machines, "Fine-tuning pi0.5" (September 2026),
https://dream-machines.eu/blog/pi05-fine-tuning. The counts in published_counts.csv are
transcribed from that post (docs/PLAN.md, Appendix A).

Each experiment's arms are compared with its reference arm (or with each other) as
independent samples: Boschloo's exact test, a Newcombe interval for the difference, and
Holm's adjustment within each experiment. For every comparison that stays unresolved, the
report gives the number of trials per arm that would detect the observed difference with
80% power.

Run it from the repository root:

    uv run python examples/dream-machines-pi05/reanalysis.py

It writes REPORT.md next to this file (or to the path given as the first argument).
"""

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fieldtrial.analysis.wording import fmt_p, fmt_p_eq, fmt_pp, fmt_range, fmt_rate, fmt_signed
from fieldtrial.stats import adjust_pvalues, compare_independent, proportion_ci, sample_size

HERE = Path(__file__).parent
SOURCE = "https://dream-machines.eu/blog/pi05-fine-tuning"
ALPHA = 0.05

# How each experiment is compared: against a reference arm, or all pairs.
REFERENCE = {
    "lora_vs_full": "full_finetune",
    "learning_rate": "lr_5e-5_baseline",
    "batch_size": "bs_256_baseline",
    "action_space_and_augmentation": "baseline",
    "operators_10h": None,
    "data_quantity": None,
    "scene_diversity_4h": "lightbox_only",
    "data_quality": "21h_baseline",
    "interventions_standard_positions": "1h_hq_only",
    "interventions_diverse_positions": "1h_hq_only",
    "serving_sweep_21h": "baseline_R30_B16",
    "initialization": "pi05_pretrained",
}
TITLES = {
    "lora_vs_full": "LoRA vs full fine-tuning",
    "learning_rate": "Learning rate",
    "batch_size": "Batch size",
    "action_space_and_augmentation": "Action space and augmentation",
    "operators_10h": "Operators (10 h of data each)",
    "data_quantity": "Data quantity",
    "scene_diversity_4h": "Scene diversity (4 h)",
    "data_quality": "Data quality",
    "interventions_standard_positions": "Interventions, standard positions",
    "interventions_diverse_positions": "Interventions, diverse positions",
    "serving_sweep_21h": "Serving settings (21 h model)",
    "serving_best_on_21h_plus_1h": "Best serving setting on the 21 h + 1 h model",
    "initialization": "Initialization",
}


@dataclass(frozen=True)
class Arm:
    """One published row."""

    experiment: str
    name: str
    k: int
    n: int


@dataclass(frozen=True)
class Comparison:
    """One arm against another, as independent samples."""

    experiment: str
    arm: Arm
    reference: Arm
    difference: float
    low: float
    high: float
    pvalue: float
    adjusted: float
    resolved: bool
    note: str = ""


def load(path: Path) -> dict[str, list[Arm]]:
    """The published counts, grouped by experiment, in file order."""
    experiments: dict[str, list[Arm]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arm = Arm(row["experiment"], row["arm"], int(row["successes"]), int(row["trials"]))
            experiments[arm.experiment].append(arm)
    return dict(experiments)


def pairs(arms: list[Arm], reference: str | None) -> list[tuple[Arm, Arm]]:
    """(arm, reference) pairs: each arm against the reference, or every pair in order."""
    if reference is None:
        return [(b, a) for i, a in enumerate(arms) for b in arms[i + 1 :]]
    ref = next(a for a in arms if a.name == reference)
    return [(a, ref) for a in arms if a is not ref]


def compare(experiment: str, todo: list[tuple[Arm, Arm]], note: str = "") -> list[Comparison]:
    """Boschloo tests and Newcombe intervals, Holm-adjusted within the experiment."""
    results = [compare_independent(a.k, a.n, r.k, r.n) for a, r in todo]
    raw = [res.primary.pvalue for res in results]
    adjusted = adjust_pvalues(raw, method="holm", alpha=ALPHA)
    return [
        Comparison(
            experiment=experiment,
            arm=a,
            reference=r,
            difference=res.difference,
            low=res.interval.low,
            high=res.interval.high,
            pvalue=p,
            adjusted=adj,
            resolved=rej,
            note=note,
        )
        for (a, r), res, p, adj, rej in zip(
            todo, results, raw, adjusted.adjusted, adjusted.reject, strict=True
        )
    ]


def trials_needed(c: Comparison) -> str:
    """Trials per arm for 80% power if the observed rates were the true ones."""
    p1, p2 = c.reference.k / c.reference.n, c.arm.k / c.arm.n
    if p1 == p2:
        return "no observed difference"
    try:
        n = sample_size(min(max(p1, 0.005), 0.995), min(max(p2, 0.005), 0.995)).n1
    except ValueError:
        return "–"
    return f"{n} per arm"


def rate(arm: Arm) -> str:
    """``91/120 = 75.8% (67.4–82.6%)``."""
    iv = proportion_ci(arm.k, arm.n).interval
    return f"{arm.k}/{arm.n} = {fmt_rate(arm.k / arm.n)} ({fmt_range(iv.low, iv.high)})"


def verdict(c: Comparison) -> str:
    """Resolved comparisons state the direction; the others only that they are open."""
    if not c.resolved:
        return "not resolved"
    return "resolved: higher" if c.difference > 0 else "resolved: lower"


def table(comparisons: list[Comparison], *, holm: bool) -> list[str]:
    """A Markdown table of comparisons."""
    head = ["Arm", "vs", "Arm rate (95% CI)", "Reference rate", "Difference (95% CI)", "Boschloo p"]
    if holm:
        head.append("Holm p")
    head.append("Verdict")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in comparisons:
        cells = [
            f"`{c.arm.name}`",
            f"`{c.reference.name}`",
            rate(c.arm),
            f"{c.reference.k}/{c.reference.n} = {fmt_rate(c.reference.k / c.reference.n)}",
            f"{fmt_pp(c.difference)} ({fmt_signed(c.low)} to {fmt_signed(c.high)})",
            fmt_p(c.pvalue),
        ]
        if holm:
            cells.append(fmt_p(c.adjusted))
        cells.append(f"**{verdict(c)}**" if c.resolved else verdict(c))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def report(experiments: dict[str, list[Arm]]) -> str:
    """The full Markdown report."""
    all_comparisons: list[Comparison] = []
    sections: list[str] = []
    for experiment, arms in experiments.items():
        title = TITLES.get(experiment, experiment)
        if experiment not in REFERENCE:
            # A single arm: its interval only.
            lines = [f"### {title}", ""]
            lines += [f"- `{a.name}`: {rate(a)}" for a in arms]
            sections += [*lines, ""]
            continue
        comparisons = compare(experiment, pairs(arms, REFERENCE[experiment]))
        all_comparisons += comparisons
        ref = REFERENCE[experiment]
        how = f"against `{ref}`" if ref else "pairwise"
        holm = len(comparisons) > 1
        sections += [
            f"### {title}",
            "",
            f"Compared {how}" + (", Holm-adjusted." if holm else "."),
            "",
        ]
        sections += [*table(comparisons, holm=holm), ""]

    # The post's best serving setting on the 21 h + 1 h model, against default serving.
    best = experiments["serving_best_on_21h_plus_1h"][0]
    default = next(a for a in experiments["data_quality"] if a.name == "21h_plus_1h_hq")
    cross = compare(
        "cross",
        [(best, default)],
        note="across experiments",
    )
    all_comparisons += cross

    resolved = [c for c in all_comparisons if c.resolved]
    open_ = [c for c in all_comparisons if not c.resolved]
    sweep = [c for c in all_comparisons if c.experiment == "serving_sweep_21h"]

    out = [
        "# Re-analysis of Dream Machines' pi0.5 fine-tuning results",
        "",
        "Source: Dream Machines, *Fine-tuning pi0.5* (September 2026),",
        f"<{SOURCE}>. All counts are theirs; this re-analysis was generated by",
        "`examples/dream-machines-pi05/reanalysis.py` in",
        "[fieldtrial](https://github.com/rokbenko/fieldtrial).",
        "",
        "> **Caveat.** These were separately run evaluations, not one randomized study. Arms",
        "> were evaluated at different times, so session-to-session drift (lighting, gripper",
        "> wear, operator) is uncontrolled and can masquerade as an effect. The 91/120 baseline",
        "> is the same set of runs reused across several experiments, so those comparisons are",
        "> not independent of each other. Treat every result below as an upper bound on what",
        "> the data can show.",
        "",
        "## How to read this",
        "",
        "- Every comparison treats the two arms as independent samples: Boschloo's exact test",
        "  (two-sided, α = 0.05) and a Newcombe 95% interval for the difference in success rate.",
        "- Within an experiment with several comparisons, p-values are Holm-adjusted; a",
        "  comparison is **resolved** when its adjusted p-value is at most 0.05.",
        "- Rates show a Wilson 95% interval. Not resolved means the data cannot tell the arms",
        "  apart; it does not mean they are equal.",
        "",
        "## Summary",
        "",
        f"- {len(all_comparisons)} comparisons; **{len(resolved)} resolved**, {len(open_)} not.",
        f"- Serving sweep: {sum(c.resolved for c in sweep)} of {len(sweep)} settings differ from "
        "the default after Holm's adjustment.",
        "",
        "Resolved comparisons:",
        "",
    ]
    out += [
        f"- `{c.arm.name}` vs `{c.reference.name}` "
        f"({TITLES.get(c.experiment, 'across experiments')}): "
        f"{fmt_pp(c.difference)} ({fmt_signed(c.low)} to {fmt_signed(c.high)}; "
        f"{'Holm ' if c.adjusted != c.pvalue else ''}{fmt_p_eq(c.adjusted)})"
        for c in resolved
    ]
    out += ["", "## Experiments", "", *sections]
    out += [
        "### Best serving setting vs default serving, 21 h + 1 h model",
        "",
        "Across experiments: `R50_B20` on the 21 h + 1 h model against the same model with",
        "default serving (from the data-quality experiment).",
        "",
        *table(cross, holm=False),
        "",
        "## Trials needed to resolve the open questions",
        "",
        "If the observed rates were the true ones, this many trials per arm would detect the",
        "difference with 80% power (two-sided α = 0.05, pooled-z, equal allocation). A paired",
        "design, with both arms run on the same starting conditions in randomized blocks,",
        "usually needs fewer. Small observed differences need very large studies; treat those",
        'as "probably no practically relevant difference" rather than as a target.',
        "",
        "| Comparison | Observed difference | Trials needed |",
        "|---|---|---|",
    ]
    out += [
        f"| `{c.arm.name}` vs `{c.reference.name}` | {fmt_pp(c.difference)} | {trials_needed(c)} |"
        for c in open_
    ]
    out += [
        "",
        "## Reproduce",
        "",
        "```console",
        "uv run python examples/dream-machines-pi05/reanalysis.py",
        "```",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str]) -> None:
    """Write the report."""
    target = Path(argv[1]) if len(argv) > 1 else HERE / "REPORT.md"
    target.write_text(report(load(HERE / "published_counts.csv")), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main(sys.argv)
