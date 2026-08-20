"""Tests for extended experiment metrics and planning APIs."""

import json
from importlib.resources import files

import pandas as pd
import pytest

from twosample_means.ab_testing import (
    ContrastSpec,
    ExperimentConfig,
    MetricSpec,
    PowerSpec,
    RatioMetricResult,
    SequentialPlan,
    alpha_spending_boundaries,
    analyze_experiment,
    estimate_mde,
    evaluate_sequential,
    normalize_experiment_data,
    simulate_power,
)
from twosample_means.data_io import DataValidationError


def test_count_metric_uses_unit_level_welch() -> None:
    """Count metrics report a mean-count effect without event-row inflation."""
    data = pd.DataFrame(
        {
            "user_id": range(8),
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "events": [1, 2, 3, 4, 2, 3, 4, 5],
        }
    )
    config = ExperimentConfig(
        experiment_id="count",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        metrics=(MetricSpec("events", "events", "count", role="primary"),),
    )

    result = analyze_experiment(data, config)

    assert result.metrics[0].method == "welch_t_count"
    assert result.metrics[0].absolute_effect == 1.0


def test_ratio_metric_uses_user_level_delta_method() -> None:
    """Ratio metrics use numerator/denominator means and valid uncertainty."""
    data = pd.DataFrame(
        {
            "user_id": range(8),
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "revenue": [
                10.0,
                20.0,
                31.0,
                40.0,
                20.0,
                40.0,
                61.0,
                80.0,
            ],
            "orders": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    metric = MetricSpec(
        "revenue_per_order",
        "revenue_per_order",
        "ratio",
        role="primary",
        numerator="revenue",
        denominator="orders",
    )
    config = ExperimentConfig(
        experiment_id="ratio",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        metrics=(metric,),
    )

    result = analyze_experiment(data, config)

    assert isinstance(result.metrics[0], RatioMetricResult)
    assert result.metrics[0].method == "delta_method_ratio"
    assert result.metrics[0].status == "ok"
    assert result.metrics[0].control.ratio == pytest.approx(10.1)
    assert result.metrics[0].treatment.ratio == pytest.approx(20.1)
    assert result.metrics[0].absolute_effect == pytest.approx(10.0)


def test_ratio_metric_rejects_nonpositive_denominator() -> None:
    """Ratio denominators must be positive at the normalization boundary."""
    data = pd.DataFrame(
        {
            "user_id": [1, 2],
            "variant": ["control", "treatment"],
            "numerator": [1.0, 2.0],
            "denominator": [0.0, 1.0],
        }
    )
    metric = MetricSpec(
        "ratio",
        "ratio",
        "ratio",
        role="primary",
        numerator="numerator",
        denominator="denominator",
    )
    config = ExperimentConfig(
        experiment_id="ratio",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        metrics=(metric,),
    )

    with pytest.raises(DataValidationError, match="positive"):
        normalize_experiment_data(data, config)


def test_planned_contrasts_support_multi_arm_comparisons() -> None:
    """Explicit contrasts produce named, multiplicity-corrected results."""
    data = pd.DataFrame(
        {
            "user_id": range(6),
            "variant": ["control", "control", "a", "a", "b", "b"],
            "converted": [0, 0, 1, 1, 0, 1],
        }
    )
    config = ExperimentConfig(
        experiment_id="multi-arm",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("a", "b"),
        contrasts=(
            ContrastSpec("a_vs_control", "a", family="primary"),
            ContrastSpec("b_vs_control", "b", family="primary"),
        ),
        metrics=(
            MetricSpec("conversion", "converted", "binary", role="primary"),
        ),
    )

    result = analyze_experiment(data, config)

    assert [metric.metric_name for metric in result.metrics] == [
        "conversion:a_vs_control",
        "conversion:b_vs_control",
    ]
    assert [metric.contrast_name for metric in result.metrics] == [
        "a_vs_control",
        "b_vs_control",
    ]
    assert all(
        metric.adjusted_p_value is not None for metric in result.metrics
    )


def test_planned_arbitrary_contrast_reorients_control() -> None:
    """ContrastSpec can compare two treatment arms directly."""
    data = pd.DataFrame(
        {
            "user_id": range(6),
            "variant": ["control", "control", "a", "a", "b", "b"],
            "converted": [0, 0, 1, 1, 0, 1],
        }
    )
    config = ExperimentConfig(
        experiment_id="arbitrary-contrast",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("a", "b"),
        contrasts=(ContrastSpec("b_vs_a", "b", control="a"),),
        metrics=(
            MetricSpec("conversion", "converted", "binary", role="primary"),
        ),
    )

    result = analyze_experiment(data, config)

    assert result.metrics[0].control_label == "a"
    assert result.metrics[0].treatment_label == "b"


def test_power_simulation_is_seeded_and_mde_is_positive() -> None:
    """Power results are reproducible and support simulated MDE search."""
    spec = PowerSpec(
        kind="continuous",
        control=0.0,
        effect=0.8,
        sample_size_control=30,
        sample_size_treatment=30,
        replications=100,
    )

    first = simulate_power(spec)
    second = simulate_power(spec)
    mde = estimate_mde(spec, target_power=0.60, max_effect=2.0, iterations=8)

    assert first == second
    assert 0.0 <= first.power <= 1.0
    assert mde > 0.0


def test_alpha_spending_plan_evaluates_predeclared_looks() -> None:
    """Sequential boundaries spend alpha and stop at the first crossing."""
    plan = SequentialPlan((0.5, 0.75, 1.0), method="obrien_fleming")
    boundaries = alpha_spending_boundaries(plan)
    continued = evaluate_sequential(plan, [0.0, 0.0, 0.0])
    stopped = evaluate_sequential(
        plan,
        [boundaries[0].z_boundary + 1.0, 0.0, 0.0],
    )

    assert boundaries[-1].cumulative_alpha == pytest.approx(0.05)
    assert continued.status == "continue"
    assert stopped.status == "stop"
    assert stopped.crossed_look == 1


def test_versioned_schema_is_bundled() -> None:
    """The experiment result schema is available from the installed package."""
    schema_path = files("twosample_means.schemas").joinpath(
        "experiment-result-v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"]["const"] == (
        "experiment-result-v1"
    )
