"""Study commands end to end: design, lock, fill (simulate or import), analyze, report."""

import json
import re
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fieldtrial.cli.main import app
from fieldtrial.stats import mcnemar_exact

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
GOLDEN = Path(__file__).parent / "fixtures" / "golden"


def run(*args: object) -> tuple[int, str]:
    result = runner.invoke(app, [str(a) for a in args])
    return result.exit_code, _ANSI.sub("", result.output)


@pytest.fixture
def golden(tmp_path: Path) -> Path:
    folder = tmp_path / "golden"
    folder.mkdir()
    shutil.copy(GOLDEN / "study.yaml", folder / "study.yaml")
    assert run("lock", folder)[0] == 0
    code, out = run("import", folder, GOLDEN / "trials.csv")
    assert code == 0, out
    assert "Imported 200 completed and 0 invalid trials in 2 sessions" in out
    return folder


def test_golden_report(golden: Path) -> None:
    code, out = run("report", golden)
    assert code == 0, out
    report = (golden / "reports" / "report.md").read_text(encoding="utf-8")
    # The section 15 golden numbers, from the independent-samples analysis.
    assert "| q50 vs baseline | 74/80 vs 91/120 | +16.7 pp | +6.2 to +26.0 pp | 0.0020 |" in report
    assert "+16.7 pp (95% CI +6.2 to +26.0; Boschloo p = 0.0020)" in report
    assert "| 74 | 80 | 92.5% | 84.6–96.5% |" in report
    assert "| 91 | 120 | 75.8% |" in report
    # The primary analysis pairs the 80 complete blocks: 17 vs 3 discordant.
    assert "Discordant blocks: 17 where only q50 succeeded, 3 where only baseline did" in report
    assert "40 blocks without a completed trial for every arm were left out" in report
    assert "40 of 240 planned trials have not been run" in report
    assert "out of the scheduled order" not in report


def test_golden_results_json(golden: Path) -> None:
    code, out = run("analyze", golden, "--json")
    assert code == 0
    results = json.loads(out)
    primary = results["primary"]
    assert primary["method"] == "mcnemar_tango"
    assert primary["n_used"] == 80
    assert primary["pvalue"] == pytest.approx(mcnemar_exact(17, 3).pvalue)
    assert primary["rejected"] is True
    sens = results["sensitivity"][0]
    assert sens["boschloo_p"] == pytest.approx(0.00196, abs=5e-5)
    assert sens["difference"] == pytest.approx(74 / 80 - 91 / 120)
    assert run("report", golden, "--format", "json")[0] == 0
    saved = json.loads((golden / "reports" / "results.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1


def test_simulation_recovers_rates(tmp_path: Path) -> None:
    folder = tmp_path / "sim"
    assert run("init", folder)[0] == 0
    assert run("validate", folder)[0] == 0
    code, out = run("plan", folder, "--baseline", "0.76")
    assert code == 0
    assert "+21.1 pp / −30.1 pp" in out
    assert run("lock", folder)[0] == 0
    code, out = run("simulate", folder, "--rates", "baseline=0.76,q50=0.90", "--seed", "1")
    assert code == 0, out
    code, out = run("status", folder)
    assert "Blinded" in out
    code, out = run("analyze", folder)
    assert code == 1
    assert "blinded" in out
    code, out = run("unblind", folder, "--yes")
    assert code == 0
    code, out = run("analyze", folder, "--json")
    assert code == 0
    results = json.loads(out)
    truth = {"baseline": 0.76, "q50": 0.90}
    for arm in results["arms"]:
        assert arm["completed"] == 40
        assert arm["ci"]["low"] <= truth[arm["arm"]] <= arm["ci"]["high"]
    assert results["deviations"] == []
    code, out = run("status", folder, "--json")
    assert json.loads(out)["per_arm"]["q50"][1] == 40


def test_amend_export_and_errors(tmp_path: Path) -> None:
    folder = tmp_path / "s"
    run("init", folder)
    yaml_path = folder / "study.yaml"
    assert run("analyze", folder)[0] == 1  # not locked
    yaml_path.write_text(yaml_path.read_text().replace("alpha: 0.05", "alpha: 0.5"))
    code, out = run("validate", folder)
    assert code == 2
    assert "analysis.primary.alpha" in out
    yaml_path.write_text(yaml_path.read_text().replace("alpha: 0.5", "alpha: 0.05"))
    assert run("lock", folder, "--json")[0] == 0
    assert run("lock", folder)[0] == 1
    yaml_path.write_text(yaml_path.read_text().replace("range: [1, 40]", "range: [1, 42]"))
    code, out = run("amend", folder, "--reason", "two more slots")
    assert code == 0
    assert "4 trials added" in out
    assert run("simulate", folder, "--rates", "baseline=2")[0] == 2
    assert run("simulate", folder, "--rates", "baseline=0.5")[0] == 1
    code, out = run(
        "simulate", folder, "--rates", "baseline=0.5,q50=0.6", "--max-trials", "10", "--json"
    )
    assert json.loads(out)["completed"] + json.loads(out)["invalid"] == 10
    code, out = run("export", folder, "--format", "jsonl")
    assert code == 0
    rows = [json.loads(x) for x in (folder / "trials.jsonl").read_text().splitlines()]
    assert len(rows) >= 10
    assert {r["arm"] for r in rows}.isdisjoint({"baseline", "q50"})
    assert run("export", folder, "--format", "xml")[0] == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("x,y\n1,2\n")
    code, out = run("import", folder, bad)
    assert code == 2
    assert "missing column" in out
    assert run("import", folder, bad, "--map", "colour=x")[0] == 2
    run("unblind", folder, "--yes")
    code, out = run("analyze", folder)
    assert code == 0
    assert "deviation:" in out
    assert "two more slots" in out
