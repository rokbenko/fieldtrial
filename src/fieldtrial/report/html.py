"""The self-contained HTML report (``report.html``).

One file with inline CSS and inline SVG charts: no scripts, no fonts, no images or styles
from other hosts, so it opens offline and can be attached to an email or a pull request.
Every sentence about results comes from :mod:`fieldtrial.analysis.wording`.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

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
from fieldtrial.report.charts import CHARTS, all_charts
from fieldtrial.report.markdown import METHOD_LABELS

TEMPLATES = Path(__file__).parent / "templates"
DASH = "–"


def _p(p: float | None) -> str:
    return DASH if p is None else fmt_p(p)


def _p_eq(p: float | None) -> str:
    return f"p = {DASH}" if p is None else fmt_p_eq(p)


def _rate(x: float | None) -> str:
    return DASH if x is None else fmt_rate(x)


def _rate_ci(ci: CI | None) -> str:
    return DASH if ci is None else fmt_range(ci.low, ci.high)


def _diff_ci(ci: CI | None) -> str:
    return DASH if ci is None else f"{fmt_signed(ci.low)} to {fmt_signed(ci.high)} pp"


def _pp(x: float | None) -> str:
    return DASH if x is None else fmt_pp(x)


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        {"p": _p, "p_eq": _p_eq, "rate": _rate, "rate_ci": _rate_ci, "diff_ci": _diff_ci, "pp": _pp}
    )
    return env


def render_html(results: Results, *, charts: bool = True) -> str:
    """Render the full report as one self-contained HTML document."""
    svgs = {k: Markup(v) for k, v in all_charts(results).items()} if charts else {}
    template = _environment().get_template("report.html")
    return template.render(
        r=results,
        s=results.study,
        charts=svgs,
        chart_titles={key: title for key, title, _ in CHARTS},
        method_label=METHOD_LABELS[results.primary.method],
        spending_labels=SPENDING_LABELS,
        css=(TEMPLATES / "report.css").read_text(encoding="utf-8"),
    )
