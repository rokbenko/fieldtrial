"""The Dream Machines re-analysis example: reproducible and self-contained."""

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "dream-machines-pi05"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reanalysis", EXAMPLE / "reanalysis.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> str:
    target = tmp_path_factory.mktemp("example") / "REPORT.md"
    _load().main(["reanalysis.py", str(target)])
    return target.read_text(encoding="utf-8")


def test_committed_report_is_current(report: str) -> None:
    committed = (EXAMPLE / "REPORT.md").read_text(encoding="utf-8")
    assert committed == report, (
        "examples/dream-machines-pi05/REPORT.md is stale; rerun "
        "`uv run python examples/dream-machines-pi05/reanalysis.py`"
    )


def test_counts_match_the_plan_appendix() -> None:
    plan = (Path(__file__).parents[1] / "docs" / "PLAN.md").read_text(encoding="utf-8")
    appendix = plan.split("## Appendix A", 1)[1].split("```csv", 1)[1].split("```", 1)[0]
    csv = (EXAMPLE / "published_counts.csv").read_text(encoding="utf-8")
    assert csv.strip() == appendix.strip()


def test_report_content(report: str) -> None:
    # The section 15 golden numbers, as they appear in the serving sweep.
    assert (
        "| `R50_B20` | `baseline_R30_B16` | 74/80 = 92.5% (84.6–96.5%) | 91/120 = 75.8% | "
        "+16.7 pp (+6.2 to +26.0) | 0.0020 |" in report
    )
    assert "+14.2 pp (−0.5 to +24.5) | 0.056 |" in report
    assert "30 comparisons; **13 resolved**, 17 not." in report
    assert "Serving sweep: 3 of 8 settings differ" in report
    assert "not one randomized study" in report
    assert "https://dream-machines.eu/blog/pi05-fine-tuning" in report
    for banned in ("better", "worse", "trend", "almost significant"):
        assert banned not in report.lower()


def test_report_is_self_contained(report: str) -> None:
    assert "![" not in report  # no images to fetch
    assert "<img" not in report
    links = re.findall(r"https?://[^\s)>]+", report)
    assert set(links) == {
        "https://dream-machines.eu/blog/pi05-fine-tuning",
        "https://github.com/rokbenko/fieldtrial",
    }
