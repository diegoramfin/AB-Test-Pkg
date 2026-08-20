"""Tests for experiment metric multiplicity correction."""

from dataclasses import replace

import pytest
from scipy import stats

from twosample_means.ab_testing import (
    adjust_p_values,
    apply_multiplicity,
    simultaneous_ci_levels,
)
from twosample_means.ab_testing.continuous import (
    ContinuousMetricResult,
    ContinuousSummary,
)


def make_result(
    name: str,
    family: str,
    p_value: float | None,
) -> ContinuousMetricResult:
    """Build a compact metric result for correction tests."""
    summary = ContinuousSummary(
        label="arm",
        n=10,
        missing=0,
        mean=1.0,
        standard_deviation=1.0,
        standard_error=0.3,
    )
    return ContinuousMetricResult(
        metric_name=name,
        role="secondary",
        family=family,
        control_label="control",
        treatment_label="treatment",
        method="welch_t",
        status="ok" if p_value is not None else "not_estimable",
        control=summary,
        treatment=summary,
        absolute_effect=1.0 if p_value is not None else None,
        relative_lift=1.0 if p_value is not None else None,
        standard_error=0.3 if p_value is not None else None,
        degrees_of_freedom=18.0 if p_value is not None else None,
        ci_lower=(
            1.0 - stats.t.ppf(0.975, 18.0) * 0.3
            if p_value is not None
            else None
        ),
        ci_upper=(
            1.0 + stats.t.ppf(0.975, 18.0) * 0.3
            if p_value is not None
            else None
        ),
        ci_level=0.95,
        p_value=p_value,
        adjusted_p_value=None,
        practical_effect=None,
        practically_significant=None,
    )


def test_holm_adjustment_preserves_positions_and_none() -> None:
    """Holm step-down correction is applied only to estimable p-values."""
    adjusted = adjust_p_values([0.02, None, 0.01, 0.5], "holm")

    assert adjusted == pytest.approx((0.04, None, 0.03, 0.5))


def test_bh_adjustment_is_monotone() -> None:
    """Benjamini-Hochberg returns monotone adjusted values in input order."""
    adjusted = adjust_p_values([0.5, 0.01, 0.02, None], "fdr_bh")

    assert adjusted == pytest.approx((0.5, 0.03, 0.03, None))


def test_simultaneous_levels_preserve_positions_and_none() -> None:
    """Holm levels follow raw-p ranks while excluding non-estimable metrics."""
    levels = simultaneous_ci_levels([0.02, None, 0.01, 0.5], "holm")

    assert levels == pytest.approx((0.975, None, 0.9833333333, 0.95))


def test_invalid_alpha_is_rejected_for_simultaneous_levels() -> None:
    """Confidence-level adjustment requires a valid family alpha."""
    with pytest.raises(ValueError, match="alpha"):
        simultaneous_ci_levels([0.1], "holm", alpha=1.0)


def test_none_returns_raw_p_values() -> None:
    """The explicit no-correction mode preserves raw values."""
    assert adjust_p_values([0.01, None, 0.8], "none") == (0.01, None, 0.8)


def test_invalid_p_values_and_methods_are_rejected() -> None:
    """Correction inputs must be valid probabilities and known methods."""
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        adjust_p_values([1.1], "holm")
    with pytest.raises(ValueError, match="unsupported"):
        adjust_p_values([0.1], "bonferroni")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported"):
        apply_multiplicity([], "bonferroni")  # type: ignore[arg-type]


def test_adjustment_is_scoped_to_metric_families() -> None:
    """Metrics in separate families do not affect each other's corrections."""
    results = (
        make_result("primary-a", "primary", 0.01),
        make_result("secondary-a", "secondary", 0.02),
        make_result("secondary-b", "secondary", 0.03),
        make_result("not-estimable", "secondary", None),
    )

    adjusted = apply_multiplicity(results, "holm")

    assert adjusted[0].adjusted_p_value == pytest.approx(0.01)
    assert adjusted[1].adjusted_p_value == pytest.approx(0.04)
    assert adjusted[2].adjusted_p_value == pytest.approx(0.04)
    assert adjusted[3].adjusted_p_value is None
    assert adjusted[3].status == "not_estimable"


def test_global_scope_pools_all_metric_families() -> None:
    """Global scope applies one correction across every declared family."""
    results = (
        make_result("primary", "primary", 0.01),
        make_result("secondary-a", "secondary", 0.02),
        make_result("secondary-b", "secondary", 0.03),
    )

    adjusted = apply_multiplicity(results, "holm", scope="global")

    assert [result.adjusted_p_value for result in adjusted] == pytest.approx(
        [0.03, 0.04, 0.04]
    )
    assert adjusted[0].simultaneous_ci_level == pytest.approx(1.0 - 0.05 / 3.0)
    assert adjusted[1].simultaneous_ci_level == pytest.approx(0.975)
    assert adjusted[2].simultaneous_ci_level == pytest.approx(0.95)


def test_invalid_multiplicity_scope_is_rejected() -> None:
    """Only family and global correction scopes are supported."""
    with pytest.raises(ValueError, match="unsupported multiplicity scope"):
        apply_multiplicity([], "holm", scope="per_role")  # type: ignore[arg-type]


def test_holm_adds_step_down_simultaneous_intervals() -> None:
    """Holm intervals use per-rank confidence levels and widen as needed."""
    results = (
        make_result("small-p", "family", 0.01),
        make_result("large-p", "family", 0.20),
    )

    adjusted = apply_multiplicity(results, "holm", alpha=0.05)

    assert adjusted[0].simultaneous_ci_level == pytest.approx(0.975)
    assert adjusted[1].simultaneous_ci_level == pytest.approx(0.95)
    assert adjusted[0].simultaneous_ci_method == "holm_step_down"
    assert adjusted[0].simultaneous_ci_lower is not None
    assert adjusted[0].simultaneous_ci_upper is not None
    assert results[0].ci_lower is not None
    assert results[0].ci_upper is not None
    assert adjusted[0].simultaneous_ci_lower < results[0].ci_lower
    assert adjusted[0].simultaneous_ci_upper > results[0].ci_upper
    assert adjusted[1].simultaneous_ci_lower == pytest.approx(
        results[1].ci_lower
    )
    assert adjusted[1].simultaneous_ci_upper == pytest.approx(
        results[1].ci_upper
    )


def test_fdr_adds_conservative_familywise_intervals() -> None:
    """BH p-values receive explicitly labeled Bonferroni intervals."""
    results = (
        make_result("first", "family", 0.01),
        make_result("second", "family", 0.02),
    )

    adjusted = apply_multiplicity(results, "fdr_bh", alpha=0.05)

    assert adjusted[0].simultaneous_ci_level == pytest.approx(0.975)
    assert adjusted[1].simultaneous_ci_level == pytest.approx(0.975)
    assert all(
        result.simultaneous_ci_method == "bonferroni_for_fdr"
        for result in adjusted
    )
    for result in adjusted:
        assert result.simultaneous_ci_lower is not None
        assert result.simultaneous_ci_upper is not None
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.simultaneous_ci_lower < result.ci_lower
        assert result.simultaneous_ci_upper > result.ci_upper


def test_no_correction_copies_nominal_intervals() -> None:
    """No multiplicity correction retains the nominal pointwise interval."""
    result = make_result("metric", "family", 0.2)

    adjusted = apply_multiplicity((result,), "none")[0]

    assert adjusted.simultaneous_ci_level == pytest.approx(result.ci_level)
    assert adjusted.simultaneous_ci_lower == pytest.approx(result.ci_lower)
    assert adjusted.simultaneous_ci_upper == pytest.approx(result.ci_upper)
    assert adjusted.simultaneous_ci_method == "unadjusted"


def test_adjustment_does_not_mutate_frozen_results() -> None:
    """Correction returns new result objects rather than mutating inputs."""
    original = make_result("metric", "family", 0.02)
    adjusted = apply_multiplicity((original,), "holm")[0]

    assert original.adjusted_p_value is None
    assert adjusted is not original
    assert adjusted.adjusted_p_value == pytest.approx(0.02)
    assert adjusted.simultaneous_ci_method == "holm_step_down"
    assert (
        replace(
            original,
            adjusted_p_value=0.02,
            simultaneous_ci_lower=adjusted.simultaneous_ci_lower,
            simultaneous_ci_upper=adjusted.simultaneous_ci_upper,
            simultaneous_ci_level=adjusted.simultaneous_ci_level,
            simultaneous_ci_method=adjusted.simultaneous_ci_method,
        )
        == adjusted
    )
