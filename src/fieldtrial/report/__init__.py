"""Reports rendered from a ``Results`` model. No database access, no statistics."""

from fieldtrial.report.html import render_html
from fieldtrial.report.markdown import render_markdown

__all__ = ["render_html", "render_markdown"]
