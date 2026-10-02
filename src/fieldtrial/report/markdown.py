"""The Markdown report (``report.md``), for GitHub, pull requests and lab notebooks.

Every sentence about results comes from :mod:`fieldtrial.analysis.wording`; this module
only lays out tables and headings.
"""

from collections.abc import Iterable, Sequence

from fieldtrial.analysis.results import CI, Results
from fieldtrial.analysis.wording import (
    SPENDING_LABELS,
    fmt_p,
    fmt_p_eq,
    fmt_pp,
    fmt_range,
    fmt_rate,
    fmt_signed,
)

_DASH = "–"
METHOD_LABELS = {
    "mcnemar_tango": "Exact McNemar test, Tango score interval",
    "cochran_q": "Cochran's Q, pairwise exact McNemar",
    "cmh": "Cochran–Mantel–Haenszel test",
    "threshold": "Exact binomial test against a threshold",
    "descriptive": "Descriptive only",
    "crossover": "Crossover rounds, randomization test",
    "ladder": "Mantel test of association with training step",
    "group_sequential": "Group-sequential McNemar, repeated CI",
    "anytime": "Anytime-valid betting test, confidence sequence",
    "selection": "Best-arm selection by successive elimination",
}


def thin(n: int, most: int = 12) -> list[int]:
    """Up to ``most`` evenly spread 1-based positions out of ``n``, always the last."""
    if n <= most:
        return list(range(1, n + 1))
    step = (n - 1) / (most - 1)
    return sorted({1 + round(i * step) for i in range(most)})


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


def _bound(low: float, high: float) -> str:
    return f"{fmt_signed(low)} to {fmt_signed(high)} pp"


def _section(title: str, body: list[str]) -> list[str]:
    return [f"## {title}", "", *body, ""]


def _design_sections(r: Results) -> list[str]:
    """Sections for sequential designs, best-arm selection, crossover rounds and ladders."""
    lines: list[str] = []
    if r.sequential is not None:
        q = r.sequential
        at = ", ".join(f"{t:.0%}" for t in q.planned_fractions)
        spending = SPENDING_LABELS.get(q.spending, q.spending)
        body = [
            f"{q.planned_looks} planned looks at {at} of {q.planned_blocks} blocks, "
            f"{spending} error spending."
            + (f" Stopped at look {q.stopped_at}." if q.stopped_at else ""),
            "",
            *_table(
                ["Look", "Kind", "Information", "Blocks", "Z", "Boundary", "Crossed", "Recorded"],
                (
                    [
                        k.look,
                        k.kind,
                        f"{k.fraction:.0%}",
                        k.blocks,
                        _DASH if k.z is None else f"{k.z:.2f}",
                        f"{k.boundary:.3f}",
                        "yes" if k.crossed else "no",
                        k.recorded_decision or _DASH,
                    ]
                    for k in q.looks
                ),
            ),
        ]
        lines += _section("Group-sequential looks", body)
    if r.anytime is not None:
        av = r.anytime
        status = ""
        if av.stopped_at:
            status = f" Stopped after block {av.stopped_at}."
        elif av.rejected_at:
            status = f" The data reject the null from block {av.rejected_at}."
        thinned = [av.sequence[i - 1] for i in thin(len(av.sequence))]
        body = [
            f"Checked after every complete block from block {av.min_blocks}; {av.blocks} of "
            f"{av.planned_blocks} planned blocks complete.{status} Largest capital against the "
            f"null: {av.capital:.3g}.",
            "",
            *_table(
                ["Blocks", "Confidence sequence for the difference"],
                ([row.blocks, _bound(row.low, row.high)] for row in thinned),
            ),
        ]
        lines += _section("Anytime-valid test", body)
    if r.selection is not None:
        sel = r.selection
        body = [
            f"Successive elimination with δ = {sel.delta:g}, checked after every complete block "
            f"from block {sel.min_blocks}; {sel.blocks} of {sel.planned_blocks} planned blocks "
            f"complete. Surviving: {', '.join(sel.survivors)}."
            + (" The study stopped when one arm remained." if sel.stopped else ""),
            "",
        ]
        if sel.eliminations:
            body += [
                *_table(
                    ["Dropped arm", "After block", "Beaten by"],
                    ([e.arm, e.block, e.by] for e in sel.eliminations),
                ),
                "",
            ]
        body += _table(
            ["Pair", "Shared blocks", "Difference", "Confidence sequence"],
            (
                [
                    f"{q.first} − {q.second}",
                    q.blocks,
                    _DASH if q.estimate is None else fmt_pp(q.estimate),
                    _bound(q.low, q.high),
                ]
                for q in sel.pairs
            ),
        )
        lines += _section("Best-arm selection", body)
    if r.agreement:
        body = [
            "Reward-model suggestions against blind human labels: linked trials (labelled live) "
            "and reviewed episodes (their first label, given before the suggestion was shown). "
            "Descriptive only.",
            "",
            *_table(
                ["Model", "Episodes", "Trials", "Reviewed", "Agreement", "Cohen's κ", "95% CI"],
                (
                    [
                        g.model,
                        g.n,
                        g.trials,
                        g.reviews,
                        fmt_rate(g.agreement),
                        _DASH if g.kappa is None else f"{g.kappa:.2f}",
                        _DASH if g.ci is None else f"{g.ci.low:.2f} to {g.ci.high:.2f}",
                    ]
                    for g in r.agreement
                ),
            ),
        ]
        lines += _section("Reward-model agreement", body)
    if r.proxy:
        body = [
            "Pre-registered secondary analysis: prediction-powered (PPI++) estimates from "
            "reward-model scores and a random sample of human labels. The primary analysis "
            "uses the scheduled trials only.",
            "",
            *_table(
                [
                    "Source",
                    "Arm",
                    "Labeled",
                    "Scored only",
                    "λ",
                    "Estimate",
                    "95% CI",
                    "Labels alone",
                ],
                (
                    [
                        q.source,
                        q.arm,
                        q.labeled,
                        q.unlabeled,
                        f"{q.lam:.2f}",
                        fmt_rate(q.estimate),
                        _rate_ci(q.ci),
                        _rate_ci(q.classical_ci),
                    ]
                    for q in r.proxy
                ),
            ),
        ]
        if r.proxy_differences:
            body += [
                "",
                *_table(
                    ["Source", "Difference", "Estimate", "95% CI"],
                    (
                        [
                            d.source,
                            f"{d.treatment} − {d.control}",
                            fmt_pp(d.estimate),
                            _bound(d.ci.low, d.ci.high),
                        ]
                        for d in r.proxy_differences
                    ),
                ),
            ]
        lines += _section("Proxy-assisted estimates", body)
    if r.crossover is not None:
        c = r.crossover
        effect = _DASH if c.period_effect is None else fmt_pp(c.period_effect)
        body = [
            f"{c.cycles} cycles ({c.ab} with {c.treatment} first, {c.ba} with {c.control} "
            f"first); {c.cycles_used} complete. Period effect (second round minus first): "
            f"{effect}.",
            "",
            *_table(
                ["Round", "Cycle", "Period", "Arm", "Successes", "Completed", "Rate"],
                (
                    [
                        w.round,
                        w.cycle,
                        w.period,
                        w.arm,
                        w.successes,
                        w.completed,
                        _DASH if w.rate is None else fmt_rate(w.rate),
                    ]
                    for w in c.rounds
                ),
            ),
        ]
        lines += _section("Crossover rounds", body)
    if r.episodes:
        body = [
            "Trials linked to LeRobot dataset episodes. Descriptive only: no test is run on it.",
            "",
            *_table(
                [
                    "Arm",
                    "Linked trials",
                    "Frames",
                    "Trials with interventions",
                    "Intervention frames",
                ],
                (
                    [
                        e.arm,
                        e.linked_trials,
                        e.frames,
                        _DASH if e.trials_with_intervention is None else e.trials_with_intervention,
                        _DASH if e.intervention_frames is None else e.intervention_frames,
                    ]
                    for e in r.episodes
                ),
            ),
        ]
        lines += _section("Dataset episodes", body)
    if r.rig_checks:
        body = [
            "Photos of the rig compared with its reference photo. A flag is a reason to look "
            "at the rig, not proof that it changed.",
            "",
            *_table(
                ["Checked (UTC)", "Source", "Shift (px)", "Brightness", "Similarity", "Flag"],
                (
                    [
                        f"{c.checked_at:%Y-%m-%d %H:%M}",
                        c.source,
                        f"{c.shift_px:.0f}",
                        f"{c.brightness_change * 100:+.0f}%",
                        f"{c.similarity:.2f}",
                        "flagged" if c.flagged else "",
                    ]
                    for c in r.rig_checks
                ),
            ),
        ]
        lines += _section("Rig checks", body)
    if r.runner:
        # A study has one runner: the lerobot runner reports loads, the others requests.
        served = [u for u in r.runner if u.policy_loads is None]
        in_process = [u for u in r.runner if u.policy_loads is not None]
        body = ["What the runner measured, per arm. Descriptive only: no test is run on it."]
        if served:
            body += [
                "",
                *_table(
                    [
                        "Arm",
                        "Trials",
                        "Requests",
                        "Errors",
                        "Median latency (ms)",
                        "Max p95 latency (ms)",
                        "Abnormal exits",
                    ],
                    (
                        [
                            u.arm,
                            u.trials,
                            _DASH if u.requests is None else u.requests,
                            _DASH if u.errors is None else u.errors,
                            _DASH if u.latency_ms_median is None else f"{u.latency_ms_median:.1f}",
                            _DASH if u.latency_ms_p95 is None else f"{u.latency_ms_p95:.1f}",
                            _DASH if u.abnormal_exits is None else u.abnormal_exits,
                        ]
                        for u in served
                    ),
                ),
            ]
        if in_process:
            body += [
                "",
                *_table(
                    [
                        "Arm",
                        "Trials",
                        "Policy loads",
                        "Median load (s)",
                        "Median recording rate (Hz)",
                        "Overruns",
                        "Control-loop errors",
                    ],
                    (
                        [
                            u.arm,
                            u.trials,
                            u.policy_loads,
                            _DASH if u.load_s_median is None else f"{u.load_s_median:.1f}",
                            _DASH if u.record_hz_median is None else f"{u.record_hz_median:.1f}",
                            u.overruns,
                            u.loop_errors,
                        ]
                        for u in in_process
                    ),
                ),
            ]
        lines += _section("Runner", body)
    if r.ladder is not None:
        lad = r.ladder
        a = lad.association
        body = [
            f"Checkpoints in order: {', '.join(lad.arms)} (steps "
            f"{', '.join(f'{x:g}' for x in lad.steps)}). Mantel test of a linear association "
            f"between step and success: Z = "
            f"{_DASH if a.statistic is None else f'{a.statistic:.2f}'}, "
            f"{_DASH if a.pvalue is None else fmt_p_eq(a.pvalue)} "
            f"({'primary' if a.primary else 'pre-registered secondary'} analysis).",
        ]
        if lad.margin is not None and lad.plateau:
            final = lad.arms[-1]
            body += [
                "",
                f"Plateau: each checkpoint against {final}, non-inferiority margin "
                f"{lad.margin * 100:.1f} pp ({lad.plateau_method}). Plateau from: "
                f"{lad.plateau_arm or _DASH}.",
                "",
                *_table(
                    ["Checkpoint", f"{final} minus it", "Interval", "Tested", "Within margin"],
                    (
                        [
                            row.arm,
                            fmt_pp(row.difference),
                            _diff_ci(row.ci),
                            "yes" if row.tested else "no",
                            "yes" if row.noninferior else "no",
                        ]
                        for row in lad.plateau
                    ),
                ),
            ]
        lines += _section("Checkpoint ladder", body)
    return lines


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
                    METHOD_LABELS[p.method],
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
    lines += _design_sections(r)

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
