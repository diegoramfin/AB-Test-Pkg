"""Simulation validation for family-scoped Benjamini-Hochberg correction."""

import numpy as np

from twosample_means.ab_testing import adjust_p_values

SEED = 20260819
REPLICATES = 300
FAMILY_COUNT = 16
METRICS_PER_FAMILY = 40
NULL_METRICS = 30
ALTERNATIVE_METRICS = METRICS_PER_FAMILY - NULL_METRICS
DISCOVERY_Q = 0.10
FDR_FLOOR = 0.13
SPARSE_ALTERNATIVES = 4
DENSE_ALTERNATIVES = 20
POWER_FLOOR = 0.40


def _family_fdr(
    p_values: list[float | None],
    null_flags: list[bool | None],
) -> float:
    """Calculate the realized false-discovery proportion for one family."""
    adjusted = adjust_p_values(p_values, "fdr_bh")
    discoveries = [
        adjusted_value is not None and adjusted_value <= DISCOVERY_Q
        for adjusted_value in adjusted
    ]
    false_discoveries = sum(
        discovery and is_null is True
        for discovery, is_null in zip(discoveries, null_flags, strict=True)
    )
    total_discoveries = sum(discoveries)
    return false_discoveries / total_discoveries if total_discoveries else 0.0


def test_bh_controls_fdr_across_many_metric_families() -> None:
    """BH controls mean realized FDR within independent mixed families."""
    rng = np.random.default_rng(SEED)
    family_fdrs: list[float] = []

    for _ in range(REPLICATES):
        for _ in range(FAMILY_COUNT):
            null_p_values = rng.uniform(0.0, 1.0, size=NULL_METRICS)
            alternative_p_values = rng.beta(
                0.25,
                1.0,
                size=ALTERNATIVE_METRICS,
            )
            p_values = [
                *null_p_values.tolist(),
                *alternative_p_values.tolist(),
                None,
            ]
            null_flags: list[bool | None] = [
                *([True] * NULL_METRICS),
                *([False] * ALTERNATIVE_METRICS),
                None,
            ]
            family_fdrs.append(_family_fdr(p_values, null_flags))

    assert float(np.mean(family_fdrs)) <= FDR_FLOOR


def _empirical_power(
    rng: np.random.Generator,
    alternative_count: int,
) -> float:
    """Estimate BH power for a fixed alternative density."""
    rejected_alternatives = 0
    total_alternatives = 0
    null_count = METRICS_PER_FAMILY - alternative_count
    for _ in range(REPLICATES):
        for _ in range(FAMILY_COUNT):
            null_p_values = rng.uniform(0.0, 1.0, size=null_count)
            alternative_p_values = rng.beta(
                0.10,
                1.0,
                size=alternative_count,
            )
            p_values = [
                *null_p_values.tolist(),
                *alternative_p_values.tolist(),
            ]
            adjusted = adjust_p_values(p_values, "fdr_bh")
            rejected_alternatives += sum(
                value is not None and value <= DISCOVERY_Q
                for value in adjusted[null_count:]
            )
            total_alternatives += alternative_count
    return rejected_alternatives / total_alternatives


def test_bh_has_empirical_power_in_sparse_and_dense_scenarios() -> None:
    """BH detects alternatives under both sparse and dense signals."""
    rng = np.random.default_rng(SEED + 2)

    sparse_power = _empirical_power(rng, SPARSE_ALTERNATIVES)
    dense_power = _empirical_power(rng, DENSE_ALTERNATIVES)

    assert sparse_power >= POWER_FLOOR
    assert dense_power >= POWER_FLOOR
    assert dense_power >= sparse_power - 0.05


def test_bh_all_null_discovery_rate_stays_at_q_across_families() -> None:
    """Under a complete null, family-level discoveries stay near q."""
    rng = np.random.default_rng(SEED + 1)
    family_discovery_rates: list[float] = []

    for _ in range(REPLICATES):
        for _ in range(FAMILY_COUNT):
            p_values = rng.uniform(0.0, 1.0, size=METRICS_PER_FAMILY)
            adjusted = adjust_p_values(p_values.tolist(), "fdr_bh")
            family_discovery_rates.append(
                float(
                    any(
                        value is not None and value <= DISCOVERY_Q
                        for value in adjusted
                    )
                )
            )

    assert float(np.mean(family_discovery_rates)) <= FDR_FLOOR
