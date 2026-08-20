"""Simulation-based coverage checks for experiment metric intervals."""

import numpy as np
import pandas as pd

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    estimate_binary_metric,
    estimate_continuous_metric,
    normalize_experiment_data,
)

SEED = 20260819
REPLICATES = 200
COVERAGE_FLOOR = 0.85


def binary_config() -> ExperimentConfig:
    """Build a binary metric experiment plan."""
    return ExperimentConfig(
        experiment_id="binary-coverage",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        metrics=(
            MetricSpec(
                name="conversion_rate",
                column="converted",
                kind="binary",
                role="primary",
            ),
        ),
    )


def continuous_config() -> ExperimentConfig:
    """Build a continuous metric experiment plan."""
    return ExperimentConfig(
        experiment_id="continuous-coverage",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        metrics=(
            MetricSpec(
                name="outcome",
                column="outcome",
                kind="continuous",
                role="primary",
            ),
        ),
    )


def test_binary_intervals_cover_null_and_non_null_rate_differences() -> None:
    """Newcombe-Wilson intervals cover binary population differences."""
    rng = np.random.default_rng(SEED)
    scenarios = ((0.10, 0.10), (0.10, 0.20))
    coverages: list[float] = []

    for control_rate, treatment_rate in scenarios:
        covered = 0
        true_effect = treatment_rate - control_rate
        for _ in range(REPLICATES):
            control = rng.binomial(1, control_rate, size=100).astype(float)
            treatment = rng.binomial(1, treatment_rate, size=100).astype(float)
            frame = pd.DataFrame(
                {
                    "user_id": range(200),
                    "variant": ["control"] * 100 + ["treatment"] * 100,
                    "converted": np.concatenate([control, treatment]),
                }
            )
            config = binary_config()
            normalized = normalize_experiment_data(frame, config)
            result = estimate_binary_metric(
                normalized, config, config.metrics[0]
            )
            assert result.ci_lower is not None
            assert result.ci_upper is not None
            covered += result.ci_lower <= true_effect <= result.ci_upper
        coverages.append(covered / REPLICATES)

    assert min(coverages) >= COVERAGE_FLOOR


def test_continuous_welch_intervals_cover_unequal_variance_effect() -> None:
    """Welch intervals cover a known mean difference with unequal variances."""
    rng = np.random.default_rng(SEED)
    covered = 0
    true_effect = 0.5
    config = continuous_config()

    for _ in range(REPLICATES):
        control = rng.normal(0.0, 1.0, size=60)
        treatment = rng.normal(0.5, 2.0, size=40)
        frame = pd.DataFrame(
            {
                "user_id": range(100),
                "variant": ["control"] * 60 + ["treatment"] * 40,
                "outcome": np.concatenate([control, treatment]),
            }
        )
        normalized = normalize_experiment_data(frame, config)
        result = estimate_continuous_metric(
            normalized, config, config.metrics[0]
        )
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        covered += result.ci_lower <= true_effect <= result.ci_upper

    assert covered / REPLICATES >= COVERAGE_FLOOR
