"""Simulation validation for family-scoped Holm correction."""

import numpy as np

from twosample_means.ab_testing import adjust_p_values

SEED = 20260819
REPLICATES = 300
FAMILY_COUNT = 16
METRICS_PER_FAMILY = 40
NULL_METRICS = 30
ALTERNATIVE_METRICS = METRICS_PER_FAMILY - NULL_METRICS
ALPHA = 0.05
FWER_CEILING = 0.075


def _family_has_false_rejection(
    p_values: list[float],
    null_flags: list[bool],
) -> bool:
    """Return whether Holm rejects at least one true null in a family."""
    adjusted = adjust_p_values(p_values, "holm")
    return any(
        adjusted_value is not None and adjusted_value <= ALPHA and is_null
        for adjusted_value, is_null in zip(adjusted, null_flags, strict=True)
    )


def test_holm_controls_fwer_under_complete_null_across_families() -> None:
    """Holm family-wise error stays near alpha when all metrics are null."""
    rng = np.random.default_rng(SEED)
    family_error_rates: list[float] = []

    for _ in range(REPLICATES):
        for _ in range(FAMILY_COUNT):
            p_values = rng.uniform(0.0, 1.0, size=METRICS_PER_FAMILY).tolist()
            family_error_rates.append(
                float(
                    _family_has_false_rejection(
                        p_values,
                        [True] * METRICS_PER_FAMILY,
                    )
                )
            )

    assert float(np.mean(family_error_rates)) <= FWER_CEILING


def test_holm_controls_strong_fwer_with_true_alternatives() -> None:
    """Holm controls false rejection of nulls alongside strong alternatives."""
    rng = np.random.default_rng(SEED + 1)
    family_error_rates: list[float] = []

    for _ in range(REPLICATES):
        for _ in range(FAMILY_COUNT):
            null_p_values = rng.uniform(0.0, 1.0, size=NULL_METRICS)
            alternative_p_values = rng.beta(
                0.10,
                1.0,
                size=ALTERNATIVE_METRICS,
            )
            family_error_rates.append(
                float(
                    _family_has_false_rejection(
                        [
                            *null_p_values.tolist(),
                            *alternative_p_values.tolist(),
                        ],
                        [
                            *([True] * NULL_METRICS),
                            *([False] * ALTERNATIVE_METRICS),
                        ],
                    )
                )
            )

    assert float(np.mean(family_error_rates)) <= FWER_CEILING
