"""Calculator commands. The acceptance table is docs/PLAN.md section 16."""

import json
import re

import pytest
from typer.testing import CliRunner

from fieldtrial.cli.calc import fmt_p, fmt_pp
from fieldtrial.cli.main import app

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def run(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, _ANSI.sub("", result.output)


# --- acceptance (section 16) ----------------------------------------------------------


def test_acceptance_ci() -> None:
    code, out = run("ci", "36/40")
    assert code == 0
    assert "0.900" in out
    assert "Wilson 95% CI [0.7695, 0.9604]" in out


def test_acceptance_compare_fisher() -> None:
    code, out = run("compare", "50/80", "13/40", "--test", "fisher")
    assert code == 0
    assert "+30.0 pp" in out
    assert "[+11.0 pp, +45.8 pp]" in out
    assert "Fisher p = 0.0034" in out


def test_acceptance_power() -> None:
    code, out = run("power", "--p1", "0.76", "--p2", "0.90")
    assert code == 0
    assert "112 per arm (pooled-z)" in out


def test_acceptance_mde() -> None:
    code, out = run("mde", "--p1", "0.76", "--n", "40")
    assert code == 0
    assert "+21.1 pp" in out
    assert "−30.1 pp" in out


# --- other output ----------------------------------------------------------------------


def test_compare_default_is_boschloo_and_reports_decision() -> None:
    code, out = run("compare", "74/80", "91/120")
    assert code == 0
    assert "Boschloo p = 0.0020" in out
    assert "rejects H0 at α = 0.05" in out
    assert "Fisher p = 0.0022" in out
    _, out = run("compare", "36/40", "91/120")
    assert "does not reject H0" in out


def test_ci_method_names() -> None:
    _, out = run("ci", "36/40", "--method", "clopper-pearson", "--level", "0.9")
    assert "Clopper–Pearson 90% CI" in out


def test_paired_with_and_without_n() -> None:
    code, out = run("paired", "--b", "10", "--c", "2", "--n", "40")
    assert code == 0
    assert "Tango 95% CI [+3.5 pp, +36.6 pp]" in out
    assert "Exact McNemar p = 0.0386" in out
    _, out = run("paired", "--b", "10", "--c", "2")
    assert "Tango" not in out


def test_power_with_ratio_and_mde_with_unequal_arms() -> None:
    _, out = run("power", "--p1", "0.76", "--p2", "0.90", "--ratio", "3")
    assert "n1 = " in out
    assert "n2 = " in out
    _, out = run("mde", "--p1", "0.5", "--n", "3")
    assert "none / none" in out
    code, _ = run("mde", "--p1", "0.76", "--n", "40", "--n2", "120")
    assert code == 0


def test_adjust_matches_spec_example() -> None:
    code, out = run("adjust", "0.00893", "0.00302", "0.01624", "--method", "holm")
    assert code == 0
    assert out.count("reject") == 3


# --- JSON ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ("ci", "36/40"),
        ("compare", "74/80", "91/120"),
        ("paired", "--b", "10", "--c", "2", "--n", "40"),
        ("power", "--p1", "0.76", "--p2", "0.9"),
        ("mde", "--p1", "0.76", "--n", "40"),
        ("adjust", "0.01", "0.04"),
    ],
)
def test_every_command_supports_json(args: tuple[str, ...]) -> None:
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 0
    json.loads(result.output)


def test_json_content_and_nan_becomes_null() -> None:
    data = json.loads(runner.invoke(app, ["compare", "40/40", "30/30", "--json"]).output)
    assert data["odds_ratio"]["estimate"] is None  # NaN in Python
    data = json.loads(runner.invoke(app, ["power", "--p1", "0.76", "--p2", "0.9", "--json"]).output)
    assert data["n1"] == 112
    assert data["n1_exact"] == pytest.approx(111.8216, abs=1e-4)


# --- errors exit with code 2 ------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ("ci", "41/40"),
        ("ci", "abc"),
        ("ci", "3/10", "--method", "wald"),
        ("compare", "3/10", "4/10", "--alternative", "bigger"),
        ("compare", "3/10", "4/10", "--test", "chi2"),
        ("power", "--p1", "0.5", "--p2", "0.5"),
        ("mde", "--p1", "1.5", "--n", "10"),
        ("paired", "--b", "5", "--c", "6", "--n", "10"),
        ("adjust", "1.5"),
    ],
)
def test_invalid_input_exits_with_2(args: tuple[str, ...]) -> None:
    code, _ = run(*args)
    assert code == 2


def test_formatters() -> None:
    assert fmt_p(0.00337) == "0.0034"
    assert fmt_p(0.00001) == "< 0.0001"
    assert fmt_p(float("nan")) == "n/a"
    assert fmt_pp(-0.301) == "−30.1 pp"
    assert fmt_pp(0.0001) == "0.0 pp"
