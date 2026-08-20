"""Tests for correlation-aware group-sequential boundary calibration."""

import numpy as np
import pytest

from twosample_means.ab_testing.sequential import (
    SequentialPlan,
    alpha_spending_boundaries,
    evaluate_sequential,
    marginal_alpha_spending_boundaries,
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
