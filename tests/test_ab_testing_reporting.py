"""Tests for experiment orchestration and generated reports."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
)
from twosample_means.reporting import (
    render_experiment_json,
    render_experiment_markdown,
    write_experiment_report,
)


def experiment_config() -> ExperimentConfig:
    """Build an experiment plan with binary and continuous metrics."""
    return ExperimentConfig(
        experiment_id="checkout-copy",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        expected_allocation={"control": 0.5, "treatment": 0.5},
        metrics=(
            MetricSpec(
                name="conversion_rate",
                column="converted",
                kind="binary",
                role="primary",
                practical_effect=0.01,
            ),
            MetricSpec(
                name="revenue",
                column="revenue",
                kind="continuous",
                role="secondary",
            ),
        ),
    )


def experiment_data() -> pd.DataFrame:
    """Build balanced user-level experiment data."""
    return pd.DataFrame(
        {
            "user_id": range(8),
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "converted": [0, 0, 1, 1, 0, 1, 1, 1],
            "revenue": [10.0, 11.0, 12.0, 13.0, 11.0, 13.0, 14.0, 15.0],
        }
    )


def test_analyze_experiment_preserves_assignment_diagnostics() -> None:
    """ExperimentResult contains diagnostics and all configured metrics."""
    result = analyze_experiment(experiment_data(), experiment_config())

    assert result.experiment_id == "checkout-copy"
    assert result.assignment_diagnostics.assignment_counts == {
        "control": 4,
        "treatment": 4,
    }
    assert result.assignment_diagnostics.sample_ratio_mismatch_p_value == 1.0
    assert result.source_rows == 8
    assert result.analysis_rows == 8
    assert result.excluded_rows == 0
    assert [metric.metric_name for metric in result.metrics] == [
        "conversion_rate",
        "revenue",
    ]
    assert all(
        metric.adjusted_p_value is not None for metric in result.metrics
    )
    assert all(
        metric.simultaneous_ci_lower is not None
        and metric.simultaneous_ci_upper is not None
        and metric.simultaneous_ci_level is not None
        for metric in result.metrics
    )
    assert result.status == "ok"


def test_global_scope_pools_metric_families_in_result() -> None:
    """Global scope changes corrections and is preserved in JSON config."""
    base = experiment_config()
    separated = replace(
        base,
        metrics=(
            replace(base.metrics[0], family="conversion"),
            replace(base.metrics[1], family="engagement"),
        ),
    )
    family_result = analyze_experiment(experiment_data(), separated)
    global_result = analyze_experiment(
        experiment_data(),
        replace(separated, multiplicity_scope="global"),
    )

    assert global_result.config["multiplicity_scope"] == "global"
    assert "Multiplicity scope**: global" in render_experiment_markdown(
        global_result
    )
    family_p_values = [
        metric.adjusted_p_value for metric in family_result.metrics
    ]
    global_p_values = [
        metric.adjusted_p_value for metric in global_result.metrics
    ]
    assert all(value is not None for value in family_p_values)
    assert all(value is not None for value in global_p_values)
    assert global_p_values != family_p_values


def test_fdr_report_labels_conservative_simultaneous_intervals() -> None:
    """FDR reports make the family-wise interval fallback explicit."""
    result = analyze_experiment(
        experiment_data(),
        replace(experiment_config(), multiplicity="fdr_bh"),
    )

    assert all(
        metric.simultaneous_ci_method == "bonferroni_for_fdr"
        for metric in result.metrics
    )
    assert "bonferroni_for_fdr" in render_experiment_markdown(result)


def test_markdown_includes_assignment_section() -> None:
    """Markdown shows assignment integrity and SRM fields."""
    result = analyze_experiment(experiment_data(), experiment_config())

    markdown = render_experiment_markdown(result)

    assert "# Experiment Analysis Report" in markdown
    assert "## Assignment Diagnostics" in markdown
    assert "Sample-ratio mismatch p-value" in markdown
    assert "Duplicate units" in markdown
    assert "## Metrics" in markdown
    assert "simultaneous CI" in markdown
    assert "holm_step_down" in markdown
    assert "conversion_rate" in markdown
    assert "revenue" in markdown


def test_json_includes_versioned_assignment_diagnostics() -> None:
    """JSON contains a schema version and machine-readable diagnostics."""
    result = analyze_experiment(experiment_data(), experiment_config())

    parsed = json.loads(render_experiment_json(result))

    assert parsed["schema_version"] == "experiment-result-v1"
    assert parsed["status"] == "ok"
    assert parsed["assignment_diagnostics"]["duplicate_units"] == 0
    assert parsed["assignment_diagnostics"]["assignment_counts"] == {
        "control": 4,
        "treatment": 4,
    }
    assert len(parsed["metrics"]) == 2
    assert all(
        metric["simultaneous_ci_lower"] is not None
        and metric["simultaneous_ci_upper"] is not None
        for metric in parsed["metrics"]
    )


def test_write_experiment_report_creates_both_formats(
    tmp_path: Path,
) -> None:
    """Experiment reports write the standard report filenames."""
    result = analyze_experiment(experiment_data(), experiment_config())

    markdown_path, json_path = write_experiment_report(result, tmp_path)

    assert markdown_path.name == "report.md"
    assert json_path.name == "report.json"
    assert markdown_path.exists()
    assert json_path.exists()
    json.loads(json_path.read_text(encoding="utf-8"))


def test_write_experiment_report_includes_self_contained_html(
    tmp_path: Path,
) -> None:
    """The HTML report is standalone and contains the analysis output."""
    result = analyze_experiment(experiment_data(), experiment_config())

    write_experiment_report(result, tmp_path)
    html_text = (tmp_path / "report.html").read_text(encoding="utf-8")

    assert html_text.startswith("<!DOCTYPE html>")
    assert "Experiment Analysis Report" in html_text
    assert result.experiment_id in html_text
    assert "conversion_rate" in html_text
    assert "Assignment Diagnostics" in html_text
    assert "<style>" in html_text
    assert html_text.count("<html") == 1
    assert "</html>" in html_text
