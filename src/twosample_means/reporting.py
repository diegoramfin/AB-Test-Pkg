"""Reporting: TestResult schema, Markdown report, JSON sidecar.

This module defines the unified ``TestResult`` schema that all tests
return, plus functions to render a Markdown report and JSON sidecar.
The report is the primary output of the procedure — it presents all
results without making any accept/reject decision.

Academic rationale
------------------
A unified result schema ensures that every test in the battery
reports the same metadata (citation, assumptions, statistic, p-value,
CI), enabling transparent comparison. The Markdown + JSON dual output
serves both human readers (Markdown) and machine-readable archival
(JSON), following reproducible research best practices (Peng, 2011).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from twosample_means.ab_testing.results import ExperimentResult
from twosample_means.schemas import (
    EXPERIMENT_RESULT_SCHEMA_VERSION,
    validate_experiment_json,
)


@dataclass(frozen=True)
class TestResult:
    """Unified result schema for all tests in the battery.

    Attributes
    ----------
    method_name:
        Name of the test or effect size.
    category:
        "parametric", "nonparametric", "bayesian",
        "effect_size", or "diagnostic".
    citation:
        Academic reference string.
    statistic:
        Test statistic (may be None for some methods).
    p_value:
        P-value (may be None for Bayesian/effect-size methods).
    ci_lower:
        Lower bound of the CI/HDI.
    ci_upper:
        Upper bound of the CI/HDI.
    ci_level:
        Confidence level used.
    extra:
        Dictionary of method-specific fields (e.g., BF10, ROPE
        proportion, degrees of freedom).
    assumption_notes:
        Human-readable notes on the assumptions of this test.
    status:
        ``"ok"`` for an estimable result, or a structured status such as
        ``"skipped"``, ``"failed"``, or ``"not_estimable"``.
    """

    __test__ = False  # not a pytest test class

    method_name: str
    category: str
    citation: str
    statistic: float | None
    p_value: float | None
    ci_lower: float | None
    ci_upper: float | None
    ci_level: float | None
    extra: dict[str, Any] = field(default_factory=dict)
    assumption_notes: str = ""
    status: str = "ok"


@dataclass(frozen=True)
class RunReport:
    """A complete run report containing all test results.

    Attributes
    ----------
    data_hash:
        SHA-256 hash of the input data.
    source_description:
        Description of the data source.
    config:
        Run configuration as a dictionary.
    results:
        List of all TestResult objects.
    """

    data_hash: str
    source_description: str
    config: dict[str, Any]
    results: list[TestResult]


def render_markdown(report: RunReport) -> str:
    """Render a RunReport as a Markdown document.

    The report presents all results in a structured format without
    making any accept/reject decision. The analyst interprets the
    output.

    Parameters
    ----------
    report:
        The complete run report.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines: list[str] = []
    lines.append("# Two-Sample Mean Difference Analysis Report")
    lines.append("")
    lines.append("## Data Provenance")
    lines.append("")
    lines.append(f"- **Source**: {report.source_description}")
    lines.append(f"- **SHA-256 hash**: `{report.data_hash}`")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for key, value in report.config.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    categories = [
        ("diagnostic", "Assumption Diagnostics"),
        ("parametric", "Parametric Tests"),
        ("nonparametric", "Non-Parametric Tests"),
        ("bayesian", "Bayesian Tests"),
        ("effect_size", "Effect Sizes"),
    ]
    for cat, title in categories:
        cat_results = [r for r in report.results if r.category == cat]
        if not cat_results:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for result in cat_results:
            lines.extend(_render_result_section(result))
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*This report presents all results without "
        "making any accept/reject decision. The analyst "
        "interprets the output.*"
    )
    return "\n".join(lines)


def render_json(report: RunReport) -> str:
    """Render a RunReport as a JSON string.

    Parameters
    ----------
    report:
        The complete run report.

    Returns
    -------
    str
        JSON-formatted report.
    """
    data = {
        "data_hash": report.data_hash,
        "source_description": report.source_description,
        "config": report.config,
        "results": [_result_to_dict(r) for r in report.results],
    }
    return json.dumps(_json_safe(data), indent=2, allow_nan=False, default=str)


def write_report(
    report: RunReport, output_dir: str | Path
) -> tuple[Path, Path]:
    """Write the Markdown and JSON reports to disk.

    Parameters
    ----------
    report:
        The complete run report.
    output_dir:
        Directory to write the reports to.

    Returns
    -------
    tuple[Path, Path]
        Paths to the Markdown and JSON files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "report.md"
    json_path = out / "report.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(render_json(report), encoding="utf-8")
    return md_path, json_path


def _render_result_section(result: TestResult) -> list[str]:
    """Render a single TestResult as Markdown lines.

    Parameters
    ----------
    result:
        The test result to render.

    Returns
    -------
    list[str]
        Markdown lines for this result.
    """
    lines: list[str] = []
    lines.append(f"### {result.method_name}")
    lines.append("")
    lines.append(f"> {result.citation}")
    if result.status != "ok":
        lines.append("")
        lines.append(f"- **Status**: {result.status}")
    lines.append("")
    if result.statistic is not None:
        lines.append(f"- **Statistic**: {result.statistic:.6f}")
    if result.p_value is not None:
        lines.append(f"- **p-value**: {result.p_value:.6f}")
    if result.ci_lower is not None and result.ci_upper is not None:
        level_pct = (
            result.ci_level * 100 if result.ci_level is not None else 95.0
        )
        lines.append(
            f"- **{level_pct:.0f}% CI**: "
            f"[{result.ci_lower:.6f}, "
            f"{result.ci_upper:.6f}]"
        )
    for key, value in result.extra.items():
        if isinstance(value, float):
            lines.append(f"- **{key}**: {value:.6f}")
        else:
            lines.append(f"- **{key}**: {value}")
    if result.assumption_notes:
        lines.append(f"- **Assumptions**: {result.assumption_notes}")
    return lines


def _json_safe(value: Any) -> Any:
    """Replace non-finite numbers with JSON ``null`` recursively."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def render_experiment_markdown(result: ExperimentResult) -> str:
    """Render an experiment result with assignment diagnostics and metrics."""
    lines = [
        "# Experiment Analysis Report",
        "",
        "## Experiment",
        "",
        f"- **Experiment ID**: {result.experiment_id}",
        f"- **Status**: {result.status}",
        f"- **SHA-256 hash**: `{result.data_hash}`",
        f"- **Source rows**: {result.source_rows}",
        f"- **Analysis rows**: {result.analysis_rows}",
        f"- **Excluded rows**: {result.excluded_rows}",
        f"- **Multiplicity**: {result.config.get('multiplicity', 'none')}",
        f"- **Multiplicity scope**: "
        f"{result.config.get('multiplicity_scope', 'family')}",
        "",
        "## Assignment Diagnostics",
        "",
    ]
    diagnostics = result.assignment_diagnostics
    lines.extend(
        [
            f"- **Status**: {diagnostics.status}",
            f"- **Assignment counts**: {diagnostics.assignment_counts}",
            f"- **Missing assignments**: {diagnostics.missing_assignment}",
            f"- **Unknown assignments**: {diagnostics.unknown_assignment}",
            f"- **Missing unit IDs**: {diagnostics.missing_unit}",
            f"- **Duplicate units**: {diagnostics.duplicate_units}",
            f"- **Multi-arm units**: {diagnostics.multi_arm_units}",
            "- **Sample-ratio mismatch evaluated**: "
            f"{diagnostics.sample_ratio_mismatch_evaluated}",
            "- **Expected allocation**: "
            f"{_format_report_value(diagnostics.expected_allocation)}",
            "- **Sample-ratio mismatch p-value**: "
            f"{diagnostics.sample_ratio_mismatch_p_value}",
        ]
    )
    if diagnostics.covariate_balance:
        lines.extend(
            [
                "",
                "### Covariate balance (standardized mean difference)",
                "",
                "| Covariate | Arm | Control mean | Arm mean | "
                "Pooled SD | SMD |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for entry in diagnostics.covariate_balance:
            smd_cell = f"{entry.smd:.4f}" if entry.smd is not None else "n/a"
            flag = " ⚠" if entry.exceeds_threshold else ""
            lines.append(
                f"| {entry.covariate} | {entry.arm} | "
                f"{_format_report_value(entry.control_mean)} | "
                f"{_format_report_value(entry.arm_mean)} | "
                f"{_format_report_value(entry.pooled_sd)} | "
                f"{smd_cell}{flag} |"
            )
        lines.extend(
            [
                "",
                "*Threshold: |SMD| > 0.1 flags imbalance.*",
            ]
        )
    if diagnostics.stratum_srm:
        lines.extend(
            [
                "",
                "### Sample-ratio mismatch by stratum",
                "",
                "| Stratum | n | SRM p-value |",
                "|---|---:|---:|",
            ]
        )
        for stratum_entry in diagnostics.stratum_srm:
            p_cell = (
                f"{stratum_entry.p_value:.4g}"
                if stratum_entry.p_value is not None
                else "n/a"
            )
            lines.append(
                f"| {stratum_entry.stratum} | {stratum_entry.n} | {p_cell} |"
            )
        lines.append("")
    for warning in diagnostics.warnings:
        lines.append(f"- **Warning**: {warning}")
    lines.extend(["", "## Metrics", ""])
    for metric in result.metrics:
        lines.extend(_render_experiment_metric(metric))
        lines.append("")
    return "\n".join(lines)


def render_experiment_json(result: ExperimentResult) -> str:
    """Render an experiment result as a versioned JSON document."""
    data = asdict(result)
    data["schema_version"] = EXPERIMENT_RESULT_SCHEMA_VERSION
    data["status"] = result.status
    return json.dumps(_json_safe(data), indent=2, allow_nan=False, default=str)


def write_experiment_report(
    result: ExperimentResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write Markdown, HTML, and JSON files for an experiment result.

    The rendered JSON is validated against the bundled schema before it is
    written, so a future report change cannot silently drift from the
    declared contract. The HTML report is a self-contained artifact with no
    external assets.
    """
    rendered = render_experiment_json(result)
    validate_experiment_json(rendered)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "report.md"
    json_path = out / "report.json"
    html_path = out / "report.html"
    md_path.write_text(render_experiment_markdown(result), encoding="utf-8")
    json_path.write_text(rendered, encoding="utf-8")
    html_path.write_text(render_experiment_html(result), encoding="utf-8")
    return md_path, json_path


def render_experiment_html(result: ExperimentResult) -> str:
    """Render an experiment result as a self-contained HTML document."""
    import html as html_module

    escape = html_module.escape
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Experiment {escape(result.experiment_id)}</title>",
        "<style>",
        _HTML_STYLE,
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<h1>Experiment Analysis Report</h1>",
        '<section class="card">',
        "<h2>Experiment</h2>",
        '<table class="meta">',
    ]
    meta_rows = [
        ("Experiment ID", result.experiment_id),
        ("Status", result.status),
        ("Schema version", EXPERIMENT_RESULT_SCHEMA_VERSION),
        ("Source rows", result.source_rows),
        ("Analysis rows", result.analysis_rows),
        ("Excluded rows", result.excluded_rows),
        ("Multiplicity", result.config.get("multiplicity", "none")),
        (
            "Multiplicity scope",
            result.config.get("multiplicity_scope", "family"),
        ),
        ("Data hash", f"<code>{escape(result.data_hash)}</code>"),
    ]
    for label, value in meta_rows:
        lines.extend(
            [
                "<tr>",
                f"<th>{escape(str(label))}</th>",
                f"<td>{_html_cell(value)}</td>",
                "</tr>",
            ]
        )
    lines.extend(["</table>", "</section>"])

    diagnostics = result.assignment_diagnostics
    lines.extend(
        [
            '<section class="card">',
            "<h2>Assignment Diagnostics</h2>",
            '<table class="meta">',
            "<tr><th>Status</th>",
            f"<td>{escape(diagnostics.status)}</td></tr>",
            "<tr><th>Assignment counts</th>",
            f"<td>{escape(str(diagnostics.assignment_counts))}</td></tr>",
            "<tr><th>Missing assignments</th>",
            f"<td>{diagnostics.missing_assignment}</td></tr>",
            "<tr><th>Unknown assignments</th>",
            f"<td>{diagnostics.unknown_assignment}</td></tr>",
            "<tr><th>Missing unit IDs</th>",
            f"<td>{diagnostics.missing_unit}</td></tr>",
            "<tr><th>Duplicate units</th>",
            f"<td>{diagnostics.duplicate_units}</td></tr>",
            "<tr><th>Multi-arm units</th>",
            f"<td>{diagnostics.multi_arm_units}</td></tr>",
            "<tr><th>SRM evaluated</th>",
            f"<td>{diagnostics.sample_ratio_mismatch_evaluated}</td></tr>",
            "<tr><th>Expected allocation</th>",
            f"<td>{escape(str(diagnostics.expected_allocation))}</td></tr>",
            "<tr><th>SRM p-value</th>",
            f"<td>{diagnostics.sample_ratio_mismatch_p_value}</td></tr>",
            "</table>",
        ]
    )
    if diagnostics.covariate_balance:
        lines.extend(
            [
                "<h3>Covariate balance (SMD)</h3>",
                '<table class="meta">',
                "<tr><th>Covariate</th><th>Arm</th><th>Control "
                "mean</th><th>Arm mean</th><th>Pooled SD</th>"
                "<th>SMD</th></tr>",
            ]
        )
        for entry in diagnostics.covariate_balance:
            smd_cell = f"{entry.smd:.4f}" if entry.smd is not None else "n/a"
            if entry.exceeds_threshold:
                smd_cell = f"{smd_cell} ⚠"
            lines.extend(
                [
                    "<tr>",
                    f"<td>{escape(entry.covariate)}</td>",
                    f"<td>{escape(entry.arm)}</td>",
                    f"<td>{escape(str(entry.control_mean))}</td>",
                    f"<td>{escape(str(entry.arm_mean))}</td>",
                    f"<td>{escape(str(entry.pooled_sd))}</td>",
                    f"<td>{escape(smd_cell)}</td>",
                    "</tr>",
                ]
            )
        lines.append("</table>")
    if diagnostics.stratum_srm:
        lines.extend(
            [
                "<h3>Sample-ratio mismatch by stratum</h3>",
                '<table class="meta">',
                "<tr><th>Stratum</th><th>n</th><th>SRM p-value</th></tr>",
            ]
        )
        for stratum_entry in diagnostics.stratum_srm:
            p_cell = (
                f"{stratum_entry.p_value:.4g}"
                if stratum_entry.p_value is not None
                else "n/a"
            )
            lines.extend(
                [
                    "<tr>",
                    f"<td>{escape(str(stratum_entry.stratum))}</td>",
                    f"<td>{stratum_entry.n}</td>",
                    f"<td>{escape(p_cell)}</td>",
                    "</tr>",
                ]
            )
        lines.append("</table>")
    if diagnostics.warnings:
        lines.append('<ul class="warnings">')
        for warning in diagnostics.warnings:
            lines.append(f"<li>{escape(warning)}</li>")
        lines.append("</ul>")
    lines.append("</section>")

    lines.append('<section class="card">')
    lines.append("<h2>Metrics</h2>")
    for metric in result.metrics:
        lines.extend(_render_experiment_metric_html(metric))
    lines.append("</section>")
    lines.extend(
        [
            "<footer>Schema "
            f"<code>{escape(EXPERIMENT_RESULT_SCHEMA_VERSION)}</code>; "
            "estimates are reported without accept/reject "
            "decisions.</footer>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(lines) + "\n"


def _html_cell(value: Any) -> str:
    """Render one table cell, allowing prebuilt HTML fragments."""
    import html as html_module

    if isinstance(value, str) and value.startswith("<"):
        return value
    return html_module.escape(str(value))


_HTML_STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, sans-serif; margin: 0;
  padding: 2rem; background: #f7f7f8; color: #1c1c1e; }
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.6rem; }
h2 { font-size: 1.15rem; margin-top: 1.5rem; }
.card { background: #fff; border: 1px solid #e3e3e6; border-radius: 8px;
  padding: 1rem 1.25rem; margin-bottom: 1rem; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 0.35rem 0.5rem;
  border-bottom: 1px solid #eee; font-size: 0.9rem; vertical-align: top; }
th { width: 14rem; color: #555; font-weight: 600; }
.warnings li { color: #8a4b00; }
footer { margin-top: 2rem; color: #777; font-size: 0.8rem; }
@media (prefers-color-scheme: dark) {
  body { background: #1b1b1d; color: #e8e8ea; }
  .card { background: #26262a; border-color: #3a3a40; }
  th, td { border-color: #33333a; }
  th { color: #b0b0b8; }
  footer { color: #8e8e96; }
}
"""


def _render_experiment_metric_html(metric: Any) -> list[str]:
    """Render one metric as an HTML mini-table."""
    import html as html_module

    escape = html_module.escape
    rows: list[tuple[str, str]] = [
        ("Status", str(metric.status)),
        ("Role", str(metric.role)),
        ("Family", str(metric.family)),
        ("Method", str(metric.method)),
        (
            "Comparison",
            f"{metric.treatment_label} - {metric.control_label}",
        ),
    ]
    contrast = getattr(metric, "contrast_name", None)
    if contrast is not None:
        rows.append(("Contrast", str(contrast)))
    if getattr(metric, "cluster_robust", False):
        rows.append(("Inference", "cluster-robust (G-2 df)"))
    rows.extend(_arm_summary_rows("Control", metric.control))
    rows.extend(_arm_summary_rows("Treatment", metric.treatment))
    for label, value in (
        ("Absolute effect (adjusted)", metric.absolute_effect),
        (
            "Absolute effect (unadjusted)",
            getattr(metric, "unadjusted_absolute_effect", None),
        ),
        ("Risk ratio", getattr(metric, "risk_ratio", None)),
        ("Variance reduction", getattr(metric, "variance_reduction", None)),
        ("CUPED theta", getattr(metric, "theta", None)),
        (
            "Covariate leakage guard",
            getattr(metric, "covariate_leakage_guard", None),
        ),
        ("Clusters", getattr(metric, "clusters", None)),
        (
            "Naive standard error",
            getattr(metric, "naive_standard_error", None),
        ),
        ("Standard error", getattr(metric, "standard_error", None)),
        ("p-value", metric.p_value),
        ("Adjusted p-value", metric.adjusted_p_value),
        ("Practical effect", metric.practical_effect),
        ("Practically significant", metric.practically_significant),
    ):
        if value is not None:
            rows.append((label, str(value)))
    if metric.ci_lower is not None and metric.ci_upper is not None:
        rows.append(
            (
                f"{metric.ci_level * 100.0:.0f}% nominal CI",
                f"[{metric.ci_lower:.6f}, {metric.ci_upper:.6f}]",
            )
        )
    if (
        metric.simultaneous_ci_lower is not None
        and metric.simultaneous_ci_upper is not None
        and metric.simultaneous_ci_level is not None
    ):
        rows.append(
            (
                f"{metric.simultaneous_ci_level * 100.0:.1f}% simultaneous CI "
                f"({metric.simultaneous_ci_method or 'corrected'})",
                f"[{metric.simultaneous_ci_lower:.6f}, "
                f"{metric.simultaneous_ci_upper:.6f}]",
            )
        )
    lines = [f"<h3>{escape(metric.metric_name)}</h3>", '<table class="meta">']
    for label, value in rows:
        lines.extend(
            [
                "<tr>",
                f"<th>{escape(label)}</th>",
                f"<td>{escape(value)}</td>",
                "</tr>",
            ]
        )
    for warning in metric.warnings:
        lines.append(
            "<tr><th>Warning</th>"
            f'<td class="warnings">{escape(warning)}</td></tr>'
        )
    lines.append("</table>")
    return lines


def _arm_summary_rows(label: str, summary: Any) -> list[tuple[str, str]]:
    """Return HTML-ready arm summary rows for any summary type."""
    rows = [
        (f"{label} n", str(summary.n)),
        (f"{label} missing", str(summary.missing)),
    ]
    if hasattr(summary, "rate"):
        rows.append((f"{label} rate", str(summary.rate)))
        rows.append((f"{label} successes", str(summary.successes)))
    elif hasattr(summary, "ratio"):
        rows.append((f"{label} ratio", str(summary.ratio)))
        rows.append((f"{label} numerator mean", str(summary.numerator_mean)))
        rows.append(
            (f"{label} denominator mean", str(summary.denominator_mean))
        )
    else:
        rows.append((f"{label} mean", str(summary.mean)))
        rows.append((f"{label} sd", str(summary.standard_deviation)))
        unadjusted = getattr(summary, "unadjusted_mean", None)
        if unadjusted is not None:
            rows.append((f"{label} unadjusted mean", str(unadjusted)))
    return rows


def _render_experiment_metric(metric: Any) -> list[str]:
    """Render one binary or continuous metric result."""
    lines = [
        f"### {metric.metric_name}",
        "",
        f"- **Status**: {metric.status}",
        f"- **Role**: {metric.role}",
        f"- **Family**: {metric.family}",
        f"- **Method**: {metric.method}",
        f"- **Comparison**: {metric.treatment_label} - {metric.control_label}",
    ]
    if getattr(metric, "contrast_name", None) is not None:
        lines.append(f"- **Contrast**: {metric.contrast_name}")
    if getattr(metric, "cluster_robust", False):
        lines.append("- **Inference**: cluster-robust (t with G-2 df)")
    lines.extend(_render_arm_summary("Control", metric.control))
    lines.extend(_render_arm_summary("Treatment", metric.treatment))
    for label, value in (
        ("Absolute effect (adjusted)", metric.absolute_effect),
        (
            "Absolute effect (unadjusted)",
            getattr(metric, "unadjusted_absolute_effect", None),
        ),
        ("Variance reduction", getattr(metric, "variance_reduction", None)),
        ("CUPED theta", getattr(metric, "theta", None)),
        (
            "Covariate leakage guard",
            getattr(metric, "covariate_leakage_guard", None),
        ),
        ("Clusters", getattr(metric, "clusters", None)),
        (
            "Naive standard error",
            getattr(metric, "naive_standard_error", None),
        ),
        ("Relative lift", metric.relative_lift),
        ("Standard error", getattr(metric, "standard_error", None)),
        ("p-value", metric.p_value),
        ("Adjusted p-value", metric.adjusted_p_value),
        ("Practical effect", metric.practical_effect),
        ("Practically significant", metric.practically_significant),
    ):
        if value is not None:
            lines.append(f"- **{label}**: {_format_report_value(value)}")
    if metric.ci_lower is not None and metric.ci_upper is not None:
        level = metric.ci_level * 100.0
        lines.append(
            f"- **{level:.0f}% nominal CI**: "
            f"[{metric.ci_lower:.6f}, {metric.ci_upper:.6f}]"
        )
    if (
        metric.simultaneous_ci_lower is not None
        and metric.simultaneous_ci_upper is not None
        and metric.simultaneous_ci_level is not None
    ):
        simultaneous_level = metric.simultaneous_ci_level * 100.0
        method = metric.simultaneous_ci_method or "corrected"
        lines.append(
            f"- **{simultaneous_level:.1f}% simultaneous CI "
            f"({method})**: "
            f"[{metric.simultaneous_ci_lower:.6f}, "
            f"{metric.simultaneous_ci_upper:.6f}]"
        )
    for warning in metric.warnings:
        lines.append(f"- **Warning**: {warning}")
    return lines


def _render_arm_summary(label: str, summary: Any) -> list[str]:
    """Render common and metric-specific arm summary fields."""
    values = [
        f"n={summary.n}",
        f"missing={summary.missing}",
    ]
    if hasattr(summary, "rate"):
        values.append(f"rate={summary.rate}")
        values.append(f"successes={summary.successes}")
    elif hasattr(summary, "ratio"):
        values.append(f"ratio={summary.ratio}")
        values.append(f"numerator_mean={summary.numerator_mean}")
        values.append(f"denominator_mean={summary.denominator_mean}")
    else:
        values.append(f"mean={summary.mean}")
        values.append(f"sd={summary.standard_deviation}")
        unadjusted_mean = getattr(summary, "unadjusted_mean", None)
        if unadjusted_mean is not None:
            values.append(f"unadjusted_mean={unadjusted_mean}")
    return [f"- **{label}**: {', '.join(values)}"]


def _format_report_value(value: Any) -> str:
    """Format report scalars consistently without assuming a float type."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _result_to_dict(result: TestResult) -> dict[str, Any]:
    """Convert a TestResult to a JSON-serializable dict.

    Parameters
    ----------
    result:
        The test result.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary.
    """
    d = asdict(result)
    return d
