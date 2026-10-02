"""The HTML report: self-contained, deterministic, escaped, and complete."""

import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest
from tests.analysis.test_engine import NOW, Recorder, info, spec

from fieldtrial.analysis.engine import analyze
from fieldtrial.report import render_html
from fieldtrial.report.charts import OKABE_ITO, all_charts, arm_colors
from fieldtrial.services.analysis import analyze_study
from fieldtrial.services.simulate import prepare_demo, sim_rates, simulate_study
from fieldtrial.services.study import lock_study, unblind_study
from fieldtrial.services.transfer import import_trials

GOLDEN = Path(__file__).parents[1] / "fixtures" / "golden"
SVG_NAMESPACES = ("http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink")


class _Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


def assert_self_contained(html: str) -> None:
    """No scripts, no external stylesheets, fonts, images or links that load anything."""
    stripped = html
    for namespace in SVG_NAMESPACES:
        stripped = stripped.replace(namespace, "")
    assert not re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", stripped, re.IGNORECASE)
    assert "@import" not in html
    # Only references inside the document: url(#clip-id) in the SVG charts.
    assert not re.search(r"url\((?!#)", html)
    parser = _Tags()
    parser.feed(html)
    for tag, attrs in parser.tags:
        assert tag not in ("script", "link", "iframe", "object", "embed", "img", "base"), tag
        for name, value in attrs.items():
            assert not name.startswith("on"), name
            if name in ("src", "href", "xlink:href", "action") and value:
                # In-document anchors, or the heatmap's pixels embedded as a PNG data URI.
                assert value.startswith(("#", "data:image/png;base64,")), (tag, name, value[:40])


@pytest.fixture(scope="module")
def demo_html(tmp_path_factory: pytest.TempPathFactory) -> str:
    folder = prepare_demo(tmp_path_factory.mktemp("demo") / "demo")
    simulate_study(folder, sim_rates(folder), seed=2, invalid_rate=0.03, trials_per_session=20)
    unblind_study(folder)
    return render_html(analyze_study(folder))


@pytest.fixture(scope="module")
def golden_html(tmp_path_factory: pytest.TempPathFactory) -> str:
    folder = tmp_path_factory.mktemp("golden")
    shutil.copy(GOLDEN / "study.yaml", folder / "study.yaml")
    lock_study(folder)
    import_trials(folder, GOLDEN / "trials.csv")
    return render_html(analyze_study(folder))


def test_demo_report_is_self_contained(demo_html: str) -> None:
    assert_self_contained(demo_html)
    assert demo_html.count("<svg") == 7
    assert demo_html.startswith("<!doctype html>")
    assert "<!DOCTYPE svg" not in demo_html


def test_golden_report(golden_html: str) -> None:
    assert_self_contained(golden_html)
    for text in (
        "+16.7 pp (95% CI +6.2 to +26.0; Boschloo p = 0.0020)",
        "<td>74/80 vs 91/120</td>",
        "<td>+6.2 to +26.0 pp</td>",
        "Discordant blocks: 17 where only q50 succeeded, 3 where only baseline did",
        "40 blocks without a completed trial",
        '<td class="num">92.5%</td>',
    ):
        assert text in golden_html, text


def test_report_is_deterministic(tmp_path: Path) -> None:
    rec = Recorder()
    for i in range(1, 11):
        rec.add("baseline", i, i % 3 != 0, session="s1" if i < 6 else "s2")
        rec.add("q50", i, i != 4, session="s1" if i < 6 else "s2")
    results = analyze(spec(), rec.records, info(), now=NOW)
    first, second = render_html(results), render_html(results)
    assert first == second
    assert "<dc:date>" not in first  # matplotlib's SVG timestamp is left out


def test_report_escapes_text() -> None:
    rec = Recorder()
    rec.add("baseline", 1, True, condition="<script>alert(1)</script>")
    rec.add("q50", 1, False, condition="<script>alert(1)</script>")
    html = render_html(analyze(spec(), rec.records, info(), now=NOW))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert_self_contained(html)


def test_empty_study_has_no_charts() -> None:
    results = analyze(spec(), [], info(planned_slots=20, pending_slots=20), now=NOW)
    assert all_charts(results) == {}
    html = render_html(results)
    assert "<svg" not in html
    assert "Not enough data for the paired comparison yet." in html
    assert "Drift checks need at least two sessions." in html
    assert_self_contained(html)


def test_three_arm_and_cmh_reports() -> None:
    rec = Recorder()
    for i in range(1, 13):
        rec.add("baseline", i, i % 3 == 0)
        rec.add("q40", i, i % 2 == 0)
        rec.add("q50", i, True)
    html = render_html(
        analyze(spec(("baseline", "q40", "q50"), slots=12), rec.records, info(), now=NOW)
    )
    assert "Adjusted p" in html
    assert html.count("(paired)") == 2
    rec = Recorder()
    for cond in range(1, 4):
        for rep in (1, 2):
            rec.add("q50", cond * 10 + rep, True, condition=f"slot={cond}", replicate=rep)
            rec.add("baseline", cond * 10 + rep, rep == 1, condition=f"slot={cond}", replicate=rep)
    html = render_html(analyze(spec(slots=3, replicates=2), rec.records, info(), now=NOW))
    assert "odds ratio" in html
    assert_self_contained(html)


def test_single_arm_report() -> None:
    rec = Recorder()
    for i in range(1, 21):
        rec.add("a", i, i > 2)
    html = render_html(
        analyze(
            spec(("a",), slots=20, design="single_arm", threshold=0.7), rec.records, info(), now=NOW
        )
    )
    assert "Exact binomial test against a threshold" in html
    assert "Sensitivity analysis" not in html
    assert_self_contained(html)


def test_render_without_charts() -> None:
    rec = Recorder()
    rec.add("baseline", 1, True)
    rec.add("q50", 1, True)
    assert "<svg" not in render_html(analyze(spec(), rec.records, info(), now=NOW), charts=False)


def test_arm_colors_cycle() -> None:
    colors = arm_colors([f"a{i}" for i in range(9)])
    assert colors["a0"] == OKABE_ITO[0]
    assert colors["a7"] == OKABE_ITO[0]
    assert len(set(colors.values())) == len(OKABE_ITO)
