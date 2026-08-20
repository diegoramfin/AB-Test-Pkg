"""Tests for CUPED variance reduction in experiment analyses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twosample_means.__main__ import main
from twosample_means.ab_testing import (
    CupedMetricResult,
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
    normalize_experiment_data,
)
from twosample_means.data_io import DataValidationError


def perfect_cuped_data() -> pd.DataFrame:
    """Covariate perfectly predicts outcomes with a constant arm effect.

    Control: X = 1..4, Y = 2X.  Treatment: X = 1..4, Y = 2X + 1.
    The pooled covariate slope is exactly 2 and adjusted outcomes are
    constant within each arm.
    """
    return pd.DataFrame(
        {
            "user_id": range(8),
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "pre_spend": [1.0, 2.0, 3.0, 4.0] * 2,
            "revenue": [2.0, 4.0, 6.0, 8.0, 3.0, 5.0, 7.0, 9.0],
        }
    )


def cuped_config() -> ExperimentConfig:
    """A continuous metric configured with a pre-experiment covariate."""
    return ExperimentConfig(
        experiment_id="cuped",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        metrics=(
            MetricSpec(
                "revenue",
                "revenue",
                "continuous",
                role="primary",
                covariate="pre_spend",
            ),
        ),
    )


def test_cuped_perfect_adjustment_recovers_constant_effect() -> None:
    """CUPED preserves the effect and removes all outcome variance."""
    result = analyze_experiment(perfect_cuped_data(), cuped_config())

    metric = result.metrics[0]
    assert isinstance(metric, CupedMetricResult)
    assert metric.method == "cuped_welch"
    assert metric.absolute_effect == pytest.approx(1.0)
    assert metric.unadjusted_absolute_effect == pytest.approx(1.0)
    assert metric.theta == pytest.approx(2.0)
    assert metric.correlation == pytest.approx(1.0)
    assert metric.variance_reduction == pytest.approx(1.0)
    assert metric.standard_error == 0.0
    assert metric.ci_lower == pytest.approx(1.0)
    assert metric.ci_upper == pytest.approx(1.0)
    assert metric.p_value == 0.0


def test_cuped_reduces_variance_on_noisy_data() -> None:
    """Variance reduction is positive and below one for noisy covariates."""
    rng = np.random.default_rng(7)
    n = 200
    covariate = rng.normal(0.0, 1.0, size=n)
    outcome = 2.0 * covariate + rng.normal(0.0, 1.0, size=n)
    data = pd.DataFrame(
        {
            "user_id": range(n),
            "variant": ["control"] * (n // 2) + ["treatment"] * (n // 2),
            "pre_score": covariate,
            "outcome": outcome,
        }
    )
    config = ExperimentConfig(
        experiment_id="noisy",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        metrics=(
            MetricSpec(
                "outcome",
                "outcome",
                "continuous",
                role="primary",
                covariate="pre_score",
            ),
        ),
    )
    metric = analyze_experiment(data, config).metrics[0]
    assert isinstance(metric, CupedMetricResult)
    assert metric.status == "ok"
    assert metric.variance_reduction is not None
    assert 0.0 < metric.variance_reduction < 1.0
    assert metric.correlation == pytest.approx(2.0 / np.sqrt(5.0), abs=0.05)


def test_cuped_excludes_units_with_missing_covariate() -> None:
    """Missing covariate rows are excluded and counted in the result."""
    data = perfect_cuped_data()
    data.loc[0, "pre_spend"] = np.nan
    normalized = normalize_experiment_data(data, cuped_config())

    assert normalized.missing_covariates == {"revenue": 1}
    metric = analyze_experiment(data, cuped_config()).metrics[0]
    assert isinstance(metric, CupedMetricResult)
    assert metric.control.missing == 1
    assert any("excluded" in warning for warning in metric.warnings)


def test_cuped_error_policy_rejects_missing_covariate() -> None:
    """The strict missing policy applies to covariate columns too."""
    config = ExperimentConfig(
        experiment_id="cuped-strict",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        metrics=(
            MetricSpec(
                "revenue",
                "revenue",
                "continuous",
                role="primary",
                covariate="pre_spend",
                missing="error",
            ),
        ),
    )
    data = perfect_cuped_data()
    data.loc[0, "pre_spend"] = np.nan

    with pytest.raises(DataValidationError, match="covariate"):
        normalize_experiment_data(data, config)


def test_covariate_is_rejected_for_binary_metrics() -> None:
    """CUPED applies only to continuous and count metrics."""
    with pytest.raises(ValueError, match="continuous and count"):
        MetricSpec(
            "conversion",
            "converted",
            "binary",
            covariate="pre_score",
        )


def test_covariate_must_differ_from_metric_column() -> None:
    """The covariate cannot reuse the outcome column."""
    with pytest.raises(ValueError, match="differ"):
        MetricSpec(
            "revenue",
            "revenue",
            "continuous",
            covariate="revenue",
        )


def test_cuped_metric_joins_multiplicity_correction() -> None:
    """CUPED results are corrected like any other metric."""
    data = perfect_cuped_data()
    config = ExperimentConfig(
        experiment_id="cuped-holm",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        metrics=(
            MetricSpec(
                "revenue",
                "revenue",
                "continuous",
                role="primary",
                covariate="pre_spend",
                family="engagement",
            ),
        ),
        multiplicity="holm",
    )

    result = analyze_experiment(data, config)
    metric = result.metrics[0]

    assert isinstance(metric, CupedMetricResult)
    assert metric.adjusted_p_value is not None
    assert metric.simultaneous_ci_lower == pytest.approx(1.0)
    assert metric.simultaneous_ci_upper == pytest.approx(1.0)
    assert metric.simultaneous_ci_method == "holm_step_down"


def test_cuped_cli_flag_end_to_end(tmp_path: Path) -> None:
    """The --covariate flag produces a valid CUPED report."""
    data_path = tmp_path / "cuped.csv"
    perfect_cuped_data().to_csv(data_path, index=False)
    output = tmp_path / "cuped-report"

    exit_code = main(
        [
            "experiment",
            str(data_path),
            "--unit-col",
            "user_id",
            "--assignment-col",
            "variant",
            "--control",
            "control",
            "--treatment",
            "treatment",
            "--metric",
            "revenue=revenue:continuous:primary",
            "--covariate",
            "revenue=pre_spend",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["metrics"][0]["method"] == "cuped_welch"
    assert report["metrics"][0]["variance_reduction"] == 1.0
    assert report["metrics"][0]["theta"] == pytest.approx(2.0)
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "Variance reduction" in markdown
