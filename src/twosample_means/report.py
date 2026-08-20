"""Backward-compatible aliases for the package reporting API.

Use :mod:`twosample_means.reporting` for new code. This module remains
importable because early package versions exposed ``twosample_means.report``
as a module name.
"""

from .reporting import (
    RunReport,
    TestResult,
    render_experiment_html,
    render_experiment_json,
    render_experiment_markdown,
    render_json,
    render_markdown,
    write_experiment_report,
    write_report,
)

__all__ = [
    "RunReport",
    "TestResult",
    "render_experiment_html",
    "render_experiment_json",
    "render_experiment_markdown",
    "render_json",
    "render_markdown",
    "write_experiment_report",
    "write_report",
]
