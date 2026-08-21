"""Tests for correlation-aware group-sequential boundary calibration."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    SequentialPlan,
    alpha_spending_boundaries,
    analyze_experiment,
    evaluate_sequential,
)
from twosample_means.ab_testing.results import MetricResult
from twosample_means.ab_testing.sequential import (
    SequentialBoundary,
    always_valid_confidence_sequence,
    difference_confidence_sequence,
    marginal_alpha_spending_boundaries,
    sequential_power,
)


def _canonical_crossing_rate(
    plan: SequentialPlan,
    z_boundaries: tuple[float, ...],
    *,
    replications: int = 200_000,
    seed: int = 3,
) -> float:
    """Monte Carlo family-wise crossing rate under the canonical model."""
    rng = np.random.default_rng(seed)
    fractions = np.asarray(plan.information_fractions)
    increments = np.sqrt(np.diff(np.concatenate(([0.0], fractions))))
    draws = rng.normal(size=(replications, len(fractions)))
    information_statistics = np.cumsum(draws * increments, axis=1)
    z_statistics = information_statistics / np.sqrt(fractions)
    boundaries = np.asarray(z_boundaries)
    if plan.two_sided:
        crossed = np.abs(z_statistics) >= boundaries
    else:
        crossed = z_statistics >= boundaries
    return float(crossed.any(axis=1).mean())


def test_calibrated_boundaries_control_familywise_error() -> None:
    """Family-wise crossing under the joint distribution equals alpha."""
    plan = SequentialPlan((0.5, 0.75, 1.0), method="obrien_fleming")
    boundaries = alpha_spending_boundaries(plan)

    rate = _canonical_crossing_rate(
        plan,
        tuple(boundary.z_boundary for boundary in boundaries),
    )

    assert abs(rate - plan.alpha) < 0.006


def test_calibrated_boundaries_match_marginal_at_first_look() -> None:
    """The first boundary has no earlier looks to account for."""
    plan = SequentialPlan((0.5, 0.75, 1.0), method="pocock")

    calibrated = alpha_spending_boundaries(plan)
    marginal = marginal_alpha_spending_boundaries(plan)

    assert calibrated[0].z_boundary == pytest.approx(
        marginal[0].z_boundary, abs=1e-9
    )


def test_calibration_differs_from_marginal_quantiles() -> None:
    """Later calibrated boundaries are not marginal quantiles."""
    plan = SequentialPlan((0.5, 0.75, 1.0), method="pocock", two_sided=True)

    calibrated = alpha_spending_boundaries(plan)
    marginal = marginal_alpha_spending_boundaries(plan)

    diff = [
        calibrated[index].z_boundary - marginal[index].z_boundary
        for index in range(len(calibrated))
    ]
    assert max(abs(value) for value in diff) > 0.1
    assert calibrated[-1].cumulative_alpha == pytest.approx(plan.alpha)


def test_calibrated_evaluation_stops_at_predeclared_look() -> None:
    """evaluate_sequential uses the calibrated boundary schedule."""
    plan = SequentialPlan((0.5, 0.75, 1.0), method="obrien_fleming")
    boundaries = alpha_spending_boundaries(plan)

    stopped = evaluate_sequential(
        plan,
        [boundaries[0].z_boundary + 1.0, 0.0, 0.0],
    )
    continued = evaluate_sequential(plan, [0.0, 0.0, 0.0])

    assert stopped.status == "stop"
    assert stopped.crossed_look == 1
    assert continued.status == "continue"


def _z_from_metric(metric: MetricResult) -> float:
    """Two-sided z from the estimator's p-value, signed by effect direction."""
    assert metric.p_value is not None and 0.0 < metric.p_value <= 1.0
    assert metric.absolute_effect is not None
    magnitude = float(stats.norm.ppf(1.0 - metric.p_value / 2.0))
    return magnitude if metric.absolute_effect >= 0.0 else -magnitude


def _pipeline_crossing_rate(
    plan: SequentialPlan,
    boundaries: tuple[SequentialBoundary, ...],
    *,
    replications: int,
    seed: int,
) -> tuple[float, list[int]]:
    """End-to-end family-wise error under the null.

    Generates balanced two-arm data with identical conversion rates, runs
    the real ``analyze_experiment`` pipeline on the accumulated data at
    each planned look, converts the primary metric's p-value into a signed
    z-statistic, and counts replicates that cross any calibrated boundary.
    This exercises data generation, estimation, p-value conversion, and
    boundary comparison together, rather than the canonical increments
    model directly.
    """
    total = 4_000
    conversion_rate = 0.05
    look_sizes = [
        round(total * fraction) for fraction in plan.information_fractions
    ]
    config = ExperimentConfig(
        experiment_id="fwer-simulation",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        expected_allocation={"control": 0.5, "treatment": 0.5},
        metrics=(
            MetricSpec(
                "conversion_rate",
                "converted",
                "binary",
                role="primary",
                family="conversion",
            ),
        ),
        multiplicity="none",
    )
    rng = np.random.default_rng(seed)
    crossed_looks = [0] * len(plan.information_fractions)
    for _ in range(replications):
        arm = np.arange(total) % 2
        converted = (rng.random(total) < conversion_rate).astype(int)
        frame = pd.DataFrame(
            {
                "user_id": range(total),
                "variant": np.where(arm == 1, "treatment", "control"),
                "converted": converted,
            }
        )
        z_statistics = [
            _z_from_metric(
                analyze_experiment(frame.iloc[:size], config).metrics[0]
            )
            for size in look_sizes
        ]
        for look, (z, boundary) in enumerate(
            zip(z_statistics, boundaries, strict=True),
            start=1,
        ):
            if abs(z) >= boundary.z_boundary:
                crossed_looks[look - 1] += 1
                break
    return float(sum(crossed_looks) / replications), crossed_looks


def test_pipeline_controls_familywise_error_under_null() -> None:
    """The estimator pipeline reproduces the calibrated FWER under the null."""
    plan = SequentialPlan((0.5, 1.0), alpha=0.05, method="obrien_fleming")
    boundaries = alpha_spending_boundaries(plan)

    crossing_rate, crossed_looks = _pipeline_crossing_rate(
        plan,
        boundaries,
        replications=300,
        seed=20260820,
    )

    assert abs(crossing_rate - plan.alpha) < 0.04, (
        f"pipeline crossing rate {crossing_rate:.3f} deviates from "
        f"alpha={plan.alpha}; crossings per look: {crossed_looks}"
    )


def test_confidence_sequence_covers_at_all_times() -> None:
    """The always-valid interval covers the mean at every look time."""
    rng = np.random.default_rng(7)
    mu, sigma, size, replications = 3.0, 2.0, 200, 4_000
    per_time_miss = np.zeros(size)
    for _ in range(replications):
        values = rng.normal(mu, sigma, size)
        sequence = always_valid_confidence_sequence(
            values,
            alpha=0.05,
            variance_proxy=sigma**2,
        )
        for interval in sequence.intervals:
            if interval.lower > mu or interval.upper < mu:
                per_time_miss[interval.n - 1] += 1
    worst = float(per_time_miss.max() / replications)
    assert worst <= 0.05, f"worst per-time miss rate {worst:.3f} exceeds alpha"


def test_confidence_sequence_is_wider_than_fixed_sample_interval() -> None:
    """Time-uniformity costs width relative to the fixed-sample interval."""
    rng = np.random.default_rng(21)
    values = rng.normal(1.0, 3.0, 500)
    sequence = always_valid_confidence_sequence(
        values,
        alpha=0.05,
        variance_proxy=9.0,
    )
    final = sequence.intervals[-1]
    fixed_half_width = stats.norm.ppf(0.975) * 3.0 / np.sqrt(500)
    assert final.upper - final.lower > 2.0 * fixed_half_width
    assert final.point_estimate == pytest.approx(float(np.mean(values)))


def test_difference_confidence_sequence_covers_known_difference() -> None:
    """The two-arm difference sequence holds at all look times."""
    rng = np.random.default_rng(11)
    delta, size, replications = 0.7, 150, 3_000
    per_time_miss = np.zeros(size)
    for _ in range(replications):
        control = rng.normal(5.0, 1.5, size)
        treatment = rng.normal(5.0 + delta, 1.5, size)
        sequence = difference_confidence_sequence(
            control,
            treatment,
            alpha=0.05,
            variance_proxy=1.5**2,
        )
        for interval in sequence.intervals:
            if interval.lower > delta or interval.upper < delta:
                per_time_miss[interval.n - 1] += 1
    worst = float(per_time_miss.max() / replications)
    assert worst <= 0.05, f"worst per-time miss rate {worst:.3f} exceeds alpha"


def test_difference_confidence_sequence_requires_parallel_streams() -> None:
    """Mismatched or empty streams are rejected."""
    with pytest.raises(ValueError, match="equal length"):
        difference_confidence_sequence(
            np.array([1.0, 2.0]),
            np.array([1.0]),
        )
    with pytest.raises(ValueError, match="non-empty"):
        difference_confidence_sequence(np.array([]), np.array([]))


def test_sequential_power_under_null_matches_familywise_error() -> None:
    """Power at zero drift equals the calibrated FWER."""
    plan = SequentialPlan((0.5, 1.0), method="obrien_fleming")
    result = sequential_power(plan, 0.0, replications=200_000, seed=1)
    assert result.power == pytest.approx(plan.alpha, abs=0.004)
    assert result.average_sample_information == pytest.approx(1.0, abs=0.01)


def test_sequential_power_increases_with_drift_and_reports_asn() -> None:
    """Power rises with drift; average sample information falls."""
    plan = SequentialPlan((0.5, 1.0), method="obrien_fleming")
    weak = sequential_power(plan, 1.0, replications=100_000, seed=2)
    strong = sequential_power(plan, 3.5, replications=100_000, seed=2)
    assert strong.power > weak.power
    assert strong.average_sample_information < weak.average_sample_information
    assert sum(strong.stopping_probabilities) == pytest.approx(1.0)
    assert strong.power < 1.0


def test_sequential_power_rejects_invalid_inputs() -> None:
    """Non-finite drift and bad replication counts are rejected."""
    plan = SequentialPlan((0.5, 1.0), method="pocock")
    with pytest.raises(ValueError, match="drift"):
        sequential_power(plan, float("nan"))
    with pytest.raises(ValueError, match="replications"):
        sequential_power(plan, 1.0, replications=1)
