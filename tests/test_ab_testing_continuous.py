"""Tests for continuous metric estimation via Welch inference."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from twosample_means.ab_testing import (
    ContinuousMetricResult,
    ExperimentConfig,
    MetricSpec,
    NormalizedExperimentData,
    estimate_continuous_metric,
    normalize_experiment_data,
)


def config_for_continuous(**kwargs: object) -> ExperimentConfig:
    """Build a valid continuous experiment plan."""
    defaults: dict[str, object] = {
        "experiment_id": "checkout-copy",
        "unit_id": "user_id",
        "assignment": "variant",
        "control": "control",
        "treatments": ("treatment",),
        "metrics": (
            MetricSpec(
                name="revenue",
                column="revenue",
                kind="continuous",
                role="primary",
                practical_effect=0.5,
            ),
        ),
    }
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)  # type: ignore[arg-type]


def normalized_continuous_data(
    control: list[float], treatment: list[float]
) -> tuple[NormalizedExperimentData, ExperimentConfig, MetricSpec]:
    """Build normalized data and its configured metric."""
    frame = pd.DataFrame(
        {
            "user_id": range(len(control) + len(treatment)),
            "variant": ["control"] * len(control)
            + ["treatment"] * len(treatment),
            "revenue": control + treatment,
        }
    )
    config = config_for_continuous()
    normalized = normalize_experiment_data(frame, config)
    return normalized, config, config.metrics[0]


def test_continuous_estimate_reports_welch_effect() -> None:
    """The adapter reports treatment minus control using Welch inference."""
    normalized, config, metric = normalized_continuous_data(
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
    )

    result = estimate_continuous_metric(normalized, config, metric)

    assert isinstance(result, ContinuousMetricResult)
    assert result.status == "ok"
    assert result.control.mean == 2.5
    assert result.treatment.mean == 3.5
    assert result.absolute_effect == 1.0
    assert result.relative_lift == pytest.approx(0.4)
    assert result.standard_error == pytest.approx(np.sqrt(5.0 / 6.0))
    assert result.ci_lower is not None
    assert result.ci_upper is not None
    assert result.ci_lower < 1.0 < result.ci_upper
    assert result.p_value is not None
    assert result.p_value == pytest.approx(
        stats.ttest_ind(
            [2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0],
            equal_var=False,
        ).pvalue
    )


def test_continuous_ci_uses_experiment_alpha() -> None:
    """The adapter translates alpha into the configured CI level."""
    normalized, config, metric = normalized_continuous_data(
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
    )
    config = config_for_continuous(alpha=0.01)
    normalized = normalize_experiment_data(normalized.frame, config)

    result = estimate_continuous_metric(normalized, config, metric)

    assert result.ci_level == 0.99
    assert result.ci_lower is not None
    assert result.ci_upper is not None


def test_missing_values_reduce_each_observed_denominator() -> None:
    """Missing continuous outcomes are counted and excluded from Welch."""
    normalized, config, metric = normalized_continuous_data(
        [1.0, np.nan, 3.0],
        [2.0, 4.0, np.nan],
    )
    result = estimate_continuous_metric(normalized, config, metric)

    assert result.control.n == 2
    assert result.control.missing == 1
    assert result.treatment.n == 2
    assert result.treatment.missing == 1
    assert result.status == "ok"


def test_zero_control_mean_leaves_relative_lift_undefined() -> None:
    """Relative lift is not reported when the control mean is zero."""
    normalized, config, metric = normalized_continuous_data(
        [-1.0, 1.0, -2.0, 2.0],
        [1.0, 3.0, 2.0, 4.0],
    )
    result = estimate_continuous_metric(normalized, config, metric)

    assert result.control.mean == 0.0
    assert result.absolute_effect == 2.5
    assert result.relative_lift is None
    assert any("undefined" in warning for warning in result.warnings)


def test_constant_arms_are_not_estimable() -> None:
    """Welch uncertainty is structured as unavailable for two constants."""
    normalized, config, metric = normalized_continuous_data(
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    )
    result = estimate_continuous_metric(normalized, config, metric)

    assert result.status == "not_estimable"
    assert result.absolute_effect is None
    assert result.ci_lower is None
    assert result.p_value is None
    assert result.warnings


def test_one_observed_value_in_an_arm_is_not_estimable() -> None:
    """Welch inference requires two observed outcomes in each arm."""
    normalized, config, metric = normalized_continuous_data(
        [1.0, np.nan, np.nan],
        [2.0, 3.0, 4.0],
    )
    result = estimate_continuous_metric(normalized, config, metric)

    assert result.status == "not_estimable"
    assert result.control.n == 1
    assert result.control.standard_deviation is None
    assert result.p_value is None


def test_multiple_treatments_require_explicit_comparison() -> None:
    """An ambiguous multi-arm request is rejected."""
    frame = pd.DataFrame(
        {
            "user_id": range(6),
            "variant": ["control", "control", "a", "a", "b", "b"],
            "revenue": [1.0, 2.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    config = config_for_continuous(treatments=("a", "b"))
    normalized = normalize_experiment_data(frame, config)

    with pytest.raises(ValueError, match="multiple treatment arms"):
        estimate_continuous_metric(normalized, config, config.metrics[0])
