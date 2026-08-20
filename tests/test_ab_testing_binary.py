"""Tests for binary experiment metric estimation."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from twosample_means.ab_testing import (
    BinaryMetricResult,
    ExperimentConfig,
    MetricSpec,
    NormalizedExperimentData,
    estimate_binary_metric,
    normalize_experiment_data,
)


def config_for_binary(**kwargs: object) -> ExperimentConfig:
    """Build a valid binary experiment plan."""
    defaults: dict[str, object] = {
        "experiment_id": "checkout-copy",
        "unit_id": "user_id",
        "assignment": "variant",
        "control": "control",
        "treatments": ("treatment",),
        "metrics": (
            MetricSpec(
                name="conversion_rate",
                column="converted",
                kind="binary",
                role="primary",
                practical_effect=0.10,
            ),
        ),
    }
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)  # type: ignore[arg-type]


def normalized_binary_data(
    control: list[float], treatment: list[float]
) -> tuple[NormalizedExperimentData, ExperimentConfig, MetricSpec]:
    """Build normalized data and its configured metric."""
    frame = pd.DataFrame(
        {
            "user_id": range(len(control) + len(treatment)),
            "variant": ["control"] * len(control)
            + ["treatment"] * len(treatment),
            "converted": control + treatment,
        }
    )
    config = config_for_binary()
    normalized = normalize_experiment_data(frame, config)
    return normalized, config, config.metrics[0]


def test_binary_estimate_reports_treatment_minus_control() -> None:
    """Rates, difference, p-value, and CI use the documented direction."""
    normalized, config, metric = normalized_binary_data(
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    )

    result = estimate_binary_metric(normalized, config, metric)

    assert isinstance(result, BinaryMetricResult)
    assert result.status == "ok"
    assert result.control.rate == 0.5
    assert result.treatment.rate == 0.75
    assert result.absolute_effect == 0.25
    assert result.relative_lift == pytest.approx(0.5)
    assert result.risk_ratio == pytest.approx(1.5)
    assert result.ci_lower is not None
    assert result.ci_upper is not None
    assert result.p_value is not None
    assert result.ci_lower < result.absolute_effect < result.ci_upper
    assert result.ci_lower >= -1.0
    assert result.ci_upper <= 1.0
    assert 0.0 <= result.p_value <= 1.0


def test_binary_p_value_matches_pooled_score_formula() -> None:
    """The reported p-value is the two-sided pooled score test."""
    normalized, config, metric = normalized_binary_data(
        [0.0] * 8,
        [1.0] * 8,
    )
    result = estimate_binary_metric(normalized, config, metric)
    pooled = 8 / 16
    standard_error = np.sqrt(pooled * (1.0 - pooled) * (2.0 / 8.0))
    expected = 2.0 * stats.norm.sf(1.0 / standard_error)

    assert result.p_value == pytest.approx(expected)


def test_missing_values_are_reported_per_arm() -> None:
    """Excluded missing values reduce the observed denominator only."""
    normalized, config, metric = normalized_binary_data(
        [1.0, np.nan, 0.0],
        [1.0, 1.0, np.nan],
    )
    result = estimate_binary_metric(normalized, config, metric)

    assert result.control.n == 2
    assert result.control.missing == 1
    assert result.treatment.n == 2
    assert result.treatment.missing == 1
    assert result.control.rate == 0.5
    assert result.treatment.rate == 1.0


def test_zero_control_rate_leaves_relative_effects_undefined() -> None:
    """Relative lift and risk ratio are null when the control rate is zero."""
    normalized, config, metric = normalized_binary_data(
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
    )
    result = estimate_binary_metric(normalized, config, metric)

    assert result.absolute_effect == 0.75
    assert result.relative_lift is None
    assert result.risk_ratio is None
    assert any("undefined" in warning for warning in result.warnings)


def test_all_missing_arm_is_not_estimable() -> None:
    """An arm with no observed outcomes produces a structured result."""
    normalized, config, metric = normalized_binary_data(
        [np.nan, np.nan],
        [1.0, 0.0],
    )
    result = estimate_binary_metric(normalized, config, metric)

    assert result.status == "not_estimable"
    assert result.absolute_effect is None
    assert result.ci_lower is None
    assert result.p_value is None
    assert result.control.n == 0
    assert result.control.missing == 2


def test_multiple_treatments_require_explicit_comparison() -> None:
    """An ambiguous multi-arm request fails instead of choosing an arm."""
    frame = pd.DataFrame(
        {
            "user_id": range(6),
            "variant": ["control", "control", "a", "a", "b", "b"],
            "converted": [0, 1, 0, 1, 1, 1],
        }
    )
    config = config_for_binary(treatments=("a", "b"))
    normalized = normalize_experiment_data(frame, config)

    with pytest.raises(ValueError, match="multiple treatment arms"):
        estimate_binary_metric(normalized, config, config.metrics[0])

    result = estimate_binary_metric(
        normalized, config, config.metrics[0], treatment="b"
    )
    assert result.treatment_label == "b"


def test_continuous_metric_is_rejected() -> None:
    """The binary estimator cannot silently reinterpret continuous data."""
    metric = MetricSpec(
        name="revenue",
        column="revenue",
        kind="continuous",
        role="primary",
    )
    config = config_for_binary(metrics=(metric,))
    frame = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [1.0, 2.0, 3.0, 4.0],
        }
    )
    normalized = normalize_experiment_data(frame, config)

    with pytest.raises(ValueError, match="kind='binary'"):
        estimate_binary_metric(normalized, config, metric)
