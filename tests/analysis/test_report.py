"""Markdown report layout for every primary method."""

from tests.analysis.test_engine import NOW, Recorder, info, spec

from fieldtrial.analysis.engine import analyze
from fieldtrial.report import render_markdown


def _three_arms() -> str:
    rec = Recorder()
    for i in range(1, 13):
        rec.add("baseline", i, i % 3 == 0)
        rec.add("q40", i, i % 2 == 0)
        rec.add("q50", i, True)
    return render_markdown(
        analyze(spec(("baseline", "q40", "q50"), slots=12), rec.records, info(), now=NOW)
    )


def test_three_arm_report_has_pairwise_table() -> None:
    text = _three_arms()
    assert "Cochran's Q, pairwise exact McNemar" in text
    assert "| Arm | vs | Pairs | b | c | Difference | 95% CI | p | Adjusted p |" in text
    assert "| q40 | baseline | 12 |" in text
    assert text.count("Sensitivity analysis treating trials as independent") == 2


def test_cmh_report() -> None:
    rec = Recorder()
    for cond in range(1, 4):
        for rep in (1, 2):
            block = cond * 10 + rep
            rec.add("q50", block, True, condition=f"slot={cond}", replicate=rep)
            rec.add("baseline", block, rep == 1, condition=f"slot={cond}", replicate=rep)
    text = render_markdown(analyze(spec(slots=3, replicates=2), rec.records, info(), now=NOW))
    assert "Cochran–Mantel–Haenszel test" in text
    assert "3 conditions × 2 replicates" in text


def test_single_arm_reports() -> None:
    rec = Recorder()
    for i in range(1, 21):
        rec.add("a", i, i > 1)
    text = render_markdown(
        analyze(
            spec(("a",), slots=20, design="single_arm", threshold=0.7), rec.records, info(), now=NOW
        )
    )
    assert "Exact binomial test against a threshold" in text
    assert "| 95.0% |" in text
    assert "Sensitivity analysis" not in text
    assert "Drift checks need at least two sessions." in text
    text = render_markdown(
        analyze(spec(("a",), slots=20, design="single_arm"), rec.records, info(), now=NOW)
    )
    assert "Descriptive only" in text


def test_empty_report() -> None:
    text = render_markdown(analyze(spec(), [], info(planned_slots=20, pending_slots=20), now=NOW))
    assert "Not enough data for the paired comparison yet." in text
    assert "**incomplete**" in text
    assert "Progress stages" not in text
    assert "| 0 | 0 | – | – | 0 |" in text


def test_report_escapes_pipes() -> None:
    rec = Recorder()
    rec.add("baseline", 1, True, condition="a|b")
    rec.add("q50", 1, True, condition="a|b")
    text = render_markdown(analyze(spec(), rec.records, info(), now=NOW))
    assert "a\\|b" in text
