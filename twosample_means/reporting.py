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
from pathlib import Path
from typing import Any


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
    return json.dumps(data, indent=2, default=str)


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
