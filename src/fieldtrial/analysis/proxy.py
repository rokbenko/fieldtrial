"""Reward-model agreement and proxy-assisted estimates (docs/stats/agreement.md, ppi.md).

Agreement is descriptive and is shown whenever scored episodes have human labels: linked
trials (labelled live, before the model saw them) and reviewed episodes (their first,
blind label). Proxy-assisted (PPI++) estimates are a pre-registered secondary analysis,
computed only with ``analysis.proxy``. For each source (a scored dataset or an imported
file) and arm, the labeled episodes are the reviewed sample (final labels) or the imported
rows with a label; the unlabeled ones are the other scored episodes of that source and arm.
Linked trials are not part of these estimates: they are the scheduled trials, analyzed by
the primary analysis, and not a random sample of the scored episodes.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from fieldtrial.analysis.common import LEVEL
from fieldtrial.analysis.results import CI, AgreementSummary, ProxyDifference, ProxyRow
from fieldtrial.design import StudySpec
from fieldtrial.stats import Interval, PPIResult, cohens_kappa, ppi_difference, ppi_mean


def _ci(interval: Interval) -> CI:
    return CI(low=interval.low, high=interval.high, level=interval.level, method=interval.method)


def agreement(items: Sequence[Mapping[str, Any]]) -> list[AgreementSummary]:
    """Agreement per reward model, on items with a human first label and a suggestion."""
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        if item["kind"] in ("trial", "review") and item.get("suggested") is not None:
            by_model.setdefault(str(item.get("model") or "unknown"), []).append(item)
    out = []
    for model, rows in sorted(by_model.items()):
        table = [[0, 0], [0, 0]]
        for row in rows:
            table[int(bool(row["first"]))][int(bool(row["suggested"]))] += 1
        res = cohens_kappa(table, level=LEVEL)
        kappa = None if res.kappa != res.kappa else res.kappa  # NaN when p_e = 1
        thresholds = {row.get("threshold") for row in rows} - {None}
        out.append(
            AgreementSummary(
                model=model,
                threshold=thresholds.pop() if len(thresholds) == 1 else None,
                n=res.n,
                trials=sum(1 for r in rows if r["kind"] == "trial"),
                reviews=sum(1 for r in rows if r["kind"] == "review"),
                table=table,
                agreement=res.agreement,
                kappa=kappa,
                ci=None if kappa is None else _ci(res.interval),
            )
        )
    return out


def proxy_estimates(
    spec: StudySpec, items: Sequence[Mapping[str, Any]], arm_of: Mapping[str, str]
) -> tuple[list[ProxyRow], list[ProxyDifference], list[str]]:
    """PPI++ estimates per source and arm, the primary comparison's difference, and notes."""
    if spec.analysis.proxy is None:
        return [], [], []
    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    for item in items:
        arm = arm_of.get(str(item.get("blind_code")))
        if arm is None or item["kind"] == "trial":
            continue
        group = groups.setdefault((str(item["source"]), arm), {"y": [], "f": [], "fu": []})
        if item["final"] is None:
            group["fu"].append(float(item["score"]))
        else:
            group["y"].append(float(bool(item["final"])))
            group["f"].append(float(item["score"]))
    rows: list[ProxyRow] = []
    fits: dict[tuple[str, str], PPIResult] = {}
    notes: list[str] = []
    for (source, arm), g in sorted(groups.items()):
        if len(g["y"]) < 2 or not g["fu"]:
            notes.append(
                f"No proxy-assisted estimate for {arm} from {source}: it needs at least 2 "
                f"labeled and 1 unlabeled scored episodes ({len(g['y'])} and {len(g['fu'])})."
            )
            continue
        fit = ppi_mean(g["y"], g["f"], g["fu"], level=LEVEL)
        fits[source, arm] = fit
        rows.append(
            ProxyRow(
                source=source,
                arm=arm,
                labeled=fit.labeled,
                unlabeled=fit.unlabeled,
                lam=fit.lam,
                estimate=fit.estimate,
                ci=_ci(fit.interval),
                classical_ci=_ci(fit.classical),
            )
        )
    diffs: list[ProxyDifference] = []
    comparison = spec.analysis.primary.comparison
    if comparison is not None:
        for source in sorted({s for s, _ in fits}):
            t, c = fits.get((source, comparison.treatment)), fits.get((source, comparison.control))
            if t is None or c is None:
                continue
            d = ppi_difference(t, c, level=LEVEL)
            diffs.append(
                ProxyDifference(
                    source=source,
                    treatment=comparison.treatment,
                    control=comparison.control,
                    estimate=d.estimate,
                    ci=_ci(d.interval),
                )
            )
    return rows, diffs, notes
