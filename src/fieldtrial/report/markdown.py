"""The Markdown report (``report.md``), for GitHub, pull requests and lab notebooks.

Every sentence about results comes from :mod:`fieldtrial.analysis.wording`; this module
only lays out tables and headings.
"""

from collections.abc import Iterable, Sequence

from fieldtrial.analysis.results import CI, Results
from fieldtrial.analysis.wording import fmt_p, fmt_pp, fmt_range, fmt_rate, fmt_signed

_DASH = "–"
_LABELS = {
    "mcnemar_tango": "Exact McNemar test, Tango score interval",
    "cochran_q": "Cochran's Q, pairwise exact McNemar",
    "cmh": "Cochran–Mantel–Haenszel test",
    "threshold": "Exact binomial test against a threshold",
    "descriptive": "Descriptive only",
}


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def _p(p: float | None) -> str:
    return _DASH if p is None else fmt_p(p)


def _rate_ci(ci: CI | None) -> str:
    return _DASH if ci is None else fmt_range(ci.low, ci.high)


def _diff_ci(ci: CI | None) -> str:
    return _DASH if ci is None else f"{fmt_signed(ci.low)} to {fmt_signed(ci.high)} pp"


def _section(title: str, body: list[str]) -> list[str]:
    return [f"## {title}", "", *body, ""]


def render_markdown(results: Results) -> str:
    """Render a complete Markdown report."""
    r = results
    s = r.study
    lines = [f"# {s.title or s.name}", ""]
    lines += [
        f"Study `{s.name}` · design `{s.design_hash[:12]}` · {s.design_type.replace('_', ' ')} · "
        f"{s.n_conditions} conditions × {s.replicates} replicate"
        f"{'s' if s.replicates != 1 else ''} · seed {s.seed} · "
        f"generated {r.provenance.generated_at:%Y-%m-%d %H:%M} UTC with fieldtrial "
        f"{r.provenance.fieldtrial_version}",
        "",
    ]
    lines += _section("Summary", [f"- {sentence}" for sentence in r.summary] or ["- " + _DASH])
    lines += _section(
        "Deviations from the plan",
        [f"- **{d.kind.replace('_', ' ')}**: {d.message}" for d in r.deviations] or ["None."],
    )

    level = "95%"
    lines += _section(
        "Success rate per arm",
        _table(
            ["Arm", "Label", "Code", "Successes", "Completed", "Rate", f"{level} CI", "Invalid"],
            (
                [
                    f"**{a.arm}**" if a.arm == s.treatment else a.arm,
                    a.label or "",
                    a.blind_code,
                    a.successes,
                    a.completed,
                    _DASH if a.rate is None else fmt_rate(a.rate),
                    _rate_ci(a.ci),
                    a.invalid,
                ]
                for a in r.arms
            ),
        ),
    )

    p = r.primary
    if p.estimate is None:
        estimate = _DASH
    elif p.estimate_kind == "difference":
        estimate = fmt_pp(p.estimate)
    elif p.estimate_kind == "odds_ratio":
        estimate = f"odds ratio {p.estimate:.2f}"
    else:
        estimate = fmt_rate(p.estimate)
    if p.ci is None:
        interval = _DASH
    elif p.estimate_kind == "difference":
        interval = _diff_ci(p.ci)
    elif p.estimate_kind == "odds_ratio":
        interval = f"{p.ci.low:.2f} to {p.ci.high:.2f}"
    else:
        interval = _rate_ci(p.ci)
    body = [
        p.description,
        "",
        *_table(
            ["Method", "Used", "Estimate", f"{level} CI", "p", "α", "Alternative", "H0"],
            [
                [
                    _LABELS[p.method],
                    p.n_used,
                    estimate,
                    interval,
                    _p(p.pvalue),
                    f"{p.alpha:g}",
                    p.alternative,
                    "rejected" if p.rejected else "not rejected",
                ]
            ],
        ),
    ]
    if len(p.pairwise) > 1:
        body += [
            "",
            *_table(
                ["Arm", "vs", "Pairs", "b", "c", "Difference", f"{level} CI", "p", "Adjusted p"],
                (
                    [
                        w.treatment,
                        w.control,
                        w.n_pairs,
                        w.b,
                        w.c,
                        _DASH if w.difference is None else fmt_pp(w.difference),
                        _diff_ci(w.ci),
                        fmt_p(w.pvalue),
                        fmt_p(w.adjusted_pvalue),
                    ]
                    for w in p.pairwise
                ),
            ),
        ]
    elif p.pairwise:
        w = p.pairwise[0]
        body += [
            "",
            f"Discordant blocks: {w.b} where only {w.treatment} succeeded, {w.c} where only "
            f"{w.control} did, out of {w.n_pairs}.",
        ]
    lines += _section("Primary analysis", body)

    if r.sensitivity:
        lines += _section(
            "Sensitivity analysis: arms as independent samples",
            _table(
                [
                    "Comparison",
                    "Counts",
                    "Difference",
                    f"Newcombe {level} CI",
                    "Boschloo p",
                    "Fisher p",
                ],
                (
                    [
                        f"{c.treatment} vs {c.control}",
                        f"{c.k1}/{c.n1} vs {c.k2}/{c.n2}",
                        fmt_pp(c.difference),
                        _diff_ci(c.ci),
                        fmt_p(c.boschloo_p),
                        fmt_p(c.fisher_p),
                    ]
                    for c in r.sensitivity
                ),
            ),
        )

    with_stages = [st for st in r.stages if st.n]
    if with_stages:
        names: list[str | None] = [None, *s.stages]
        rows = []
        for name in names:
            row: list[object] = [name or "(none)"]
            for st in with_stages:
                share = next(d for d in st.distribution if d.stage == name)
                row.append(f"{share.count} ({fmt_rate(share.share)})")
            rows.append(row)
        body = [
            f"Furthest stage reached. Success means reaching `{s.success_stage}`.",
            "",
            *_table(["Stage", *(st.arm for st in with_stages)], rows),
            "",
            "Funnel: of the trials that reached the previous stage, how many reached this one.",
            "",
        ]
        funnel_rows = []
        for i, name in enumerate(s.stages):
            row = [name]
            for st in with_stages:
                step = st.funnel[i]
                conv = _DASH if step.conversion is None else fmt_rate(step.conversion)
                row.append(f"{step.reached}/{step.entered} ({conv})")
            funnel_rows.append(row)
        body += _table(["Stage", *(st.arm for st in with_stages)], funnel_rows)
        if r.stage_comparisons:
            body += [
                "",
                *_table(
                    ["Brunner–Munzel on stage", "p", "Pre-registered"],
                    (
                        [
                            f"{c.treatment} vs {c.control}",
                            _p(c.pvalue),
                            "yes" if c.preregistered else "no",
                        ]
                        for c in r.stage_comparisons
                    ),
                ),
            ]
        lines += _section("Progress stages", body)

    lines += _section(
        "Time to success",
        _table(
            ["Arm", "Successes", "Median (s)", f"{level} CI (bootstrap)"],
            (
                [
                    t.arm,
                    t.n_successes,
                    _DASH if t.median_s is None else f"{t.median_s:.1f}",
                    _DASH if t.ci is None else f"{t.ci.low:.1f}{_DASH}{t.ci.high:.1f}",
                ]
                for t in r.timing
            ),
        ),
    )

    arms = s.arms
    lines += _section(
        "Outcomes per condition",
        [
            "<details><summary>Successes / completed trials per condition and arm</summary>",
            "",
            *_table(
                ["Condition", *arms],
                (
                    [row.condition, *(f"{c.successes}/{c.completed}" for c in row.cells)]
                    for row in r.conditions
                ),
            ),
            "",
            "</details>",
        ],
    )

    checks = _table(
        ["Check", "Arm", "Groups", "Method", "p", "Flag"],
        (
            [
                f"across {d.grouping}s",
                d.arm,
                d.groups,
                d.method,
                _p(d.pvalue),
                "flagged" if d.flagged else "",
            ]
            for d in r.drift
        ),
    )
    inv = r.invalid
    invalid_counts = ", ".join(f"{a} {inv.per_arm[a]}/{inv.attempts[a]}" for a in inv.per_arm)
    lines += _section(
        "Sessions and validity checks",
        [
            *_table(
                ["Started (UTC)", "Operator", "Rig", *arms],
                (
                    [
                        f"{row.started_at:%Y-%m-%d %H:%M}",
                        row.operator,
                        row.rig,
                        *(f"{c.successes}/{c.completed}" for c in row.cells),
                    ]
                    for row in r.sessions
                ),
            ),
            "",
            *(checks if r.drift else ["Drift checks need at least two sessions."]),
            "",
            f"Invalid trials (invalid/attempts): {invalid_counts}"
            + (f"; {inv.method} p = {_p(inv.pvalue)}" if inv.method else "")
            + (" (flagged)." if inv.flagged else "."),
        ],
    )

    pv = r.provenance
    lines += _section(
        "Provenance",
        [
            f"- Design hash: `{pv.design_hash}` (seed {pv.seed})",
            f"- Locked with fieldtrial {pv.locked_with_version}; analyzed with "
            f"{pv.fieldtrial_version} (Python {pv.python}, numpy {pv.numpy}, scipy {pv.scipy})",
            "- Arm fingerprints: "
            + ", ".join(f"{a} `{h}`" for a, h in pv.policy_fingerprints.items()),
            f"- {pv.sessions} sessions; operators: {', '.join(pv.operators) or _DASH}; "
            f"rigs: {', '.join(pv.rigs) or _DASH}",
            f"- Results schema version {r.schema_version}",
        ],
    )
    return "\n".join(lines).rstrip() + "\n"
