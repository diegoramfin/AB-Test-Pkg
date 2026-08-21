"""Simulation-based coverage checks for experiment metric intervals."""

import numpy as np
import pandas as pd

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    estimate_binary_metric,
    estimate_clustered_metric,
    estimate_continuous_metric,
    normalize_experiment_data,
)

SEED = 20260819
REPLICATES = 200
COVERAGE_FLOOR = 0.85
CLUSTER_REPLICATES = 300


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


def _clustered_continuous_config() -> ExperimentConfig:
    """Build a cluster-robust continuous metric experiment plan."""
    return ExperimentConfig(
        experiment_id="clustered-continuous-coverage",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        cluster="cluster_id",
        metrics=(
            MetricSpec(
                name="outcome",
                column="outcome",
                kind="continuous",
                role="primary",
            ),
        ),
    )


def _clustered_continuous_frame(
    rng: np.random.Generator,
    delta: float,
    *,
    clusters_per_arm: int = 12,
    units_per_cluster: int = 10,
) -> pd.DataFrame:
    """Generate clustered outcomes with a known mean difference.

    A shared within-cluster shift (sd 3.0 vs unit noise sd 1.0) creates
    strong intra-cluster correlation; adding ``delta`` to every treatment
    outcome makes the true treatment-minus-control mean difference exactly
    ``delta``.
    """
    rows: list[dict[str, object]] = []
    for cluster in range(clusters_per_arm * 2):
        shift = rng.normal(0.0, 3.0)
        arm = "control" if cluster % 2 == 0 else "treatment"
        for unit in range(units_per_cluster):
            rows.append(
                {
                    "user_id": f"{cluster}-{unit}",
                    "variant": arm,
                    "cluster_id": f"g{cluster}",
                    "outcome": (
                        shift
                        + rng.normal(0.0, 1.0)
                        + (delta if arm == "treatment" else 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_clustered_continuous_intervals_cover_known_mean_differences() -> None:
    """Cluster-robust continuous intervals hold nominal coverage.

    Strong within-cluster correlation inflates the outcome variance. The
    cluster-robust interval must cover the known mean difference near the
    nominal rate while the naive Welch interval systematically undercovers.
    """
    rng = np.random.default_rng(SEED)
    config = _clustered_continuous_config()

    for delta in (0.0, 0.5):
        covered = 0
        naive_covered = 0
        for _ in range(CLUSTER_REPLICATES):
            frame = _clustered_continuous_frame(rng, delta)
            normalized = normalize_experiment_data(frame, config)
            result = estimate_clustered_metric(
                normalized, config, config.metrics[0]
            )
            assert result.ci_lower is not None
            assert result.ci_upper is not None
            assert result.naive_standard_error is not None
            assert result.absolute_effect is not None
            covered += result.ci_lower <= delta <= result.ci_upper
            z = 1.959964
            naive_lower = (
                result.absolute_effect - z * result.naive_standard_error
            )
            naive_upper = (
                result.absolute_effect + z * result.naive_standard_error
            )
            naive_covered += naive_lower <= delta <= naive_upper

        robust_coverage = covered / CLUSTER_REPLICATES
        naive_coverage = naive_covered / CLUSTER_REPLICATES
        assert robust_coverage >= COVERAGE_FLOOR
        assert naive_coverage < 0.80


def _clustered_ratio_config() -> ExperimentConfig:
    """Build a cluster-robust ratio metric experiment plan."""
    return ExperimentConfig(
        experiment_id="clustered-ratio-coverage",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        cluster="cluster_id",
        metrics=(
            MetricSpec(
                name="arpu",
                column="revenue",
                kind="ratio",
                role="primary",
                numerator="revenue",
                denominator="orders",
            ),
        ),
    )


def _clustered_ratio_frame(
    rng: np.random.Generator,
    delta: float,
    *,
    clusters_per_arm: int = 20,
    units_per_cluster: int = 10,
) -> pd.DataFrame:
    """Generate a clustered numerator/denominator frame with a known ratio.

    ``den = 10 + shift + eps`` and ``num = 40 + 7 * shift + eps`` share the
    cluster shift, so the arm ratio is ``4`` and the influence values carry
    within-cluster correlation. Scaling treatment numerators by ``1 + delta``
    makes the true treatment-minus-control ratio effect ``4 * delta``.
    """
    rows: list[dict[str, object]] = []
    for cluster in range(clusters_per_arm * 2):
        shift = rng.normal(0.0, 1.5)
        arm = "control" if cluster % 2 == 0 else "treatment"
        for unit in range(units_per_cluster):
            denominator = 10.0 + shift + rng.normal(0.0, 1.0)
            numerator = 40.0 + 7.0 * shift + rng.normal(0.0, 1.0)
            if arm == "treatment":
                numerator *= 1.0 + delta
            rows.append(
                {
                    "user_id": f"{cluster}-{unit}",
                    "variant": arm,
                    "cluster_id": f"g{cluster}",
                    "revenue": numerator,
                    "orders": denominator,
                }
            )
    return pd.DataFrame(rows)


def test_clustered_ratio_intervals_cover_known_ratio_differences() -> None:
    """Cluster-robust ratio intervals hold nominal coverage; naive ones fail.

    Within-cluster correlation inflates the ratio influence values. The
    cluster-robust delta-method interval must cover the known population
    ratio difference near the nominal rate, while the naive user-level
    interval systematically undercovers — demonstrating the correction is
    doing its job rather than just being wider.
    """
    rng = np.random.default_rng(SEED)
    config = _clustered_ratio_config()

    for delta in (0.0, 0.10):
        true_effect = 4.0 * delta
        covered = 0
        naive_covered = 0
        for _ in range(CLUSTER_REPLICATES):
            frame = _clustered_ratio_frame(rng, delta)
            normalized = normalize_experiment_data(frame, config)
            result = estimate_clustered_metric(
                normalized, config, config.metrics[0]
            )
            assert result.ci_lower is not None
            assert result.ci_upper is not None
            assert result.naive_standard_error is not None
            assert result.absolute_effect is not None
            covered += result.ci_lower <= true_effect <= result.ci_upper
            z = 1.959964
            naive_lower = (
                result.absolute_effect - z * result.naive_standard_error
            )
            naive_upper = (
                result.absolute_effect + z * result.naive_standard_error
            )
            naive_covered += naive_lower <= true_effect <= naive_upper

        robust_coverage = covered / CLUSTER_REPLICATES
        naive_coverage = naive_covered / CLUSTER_REPLICATES
        assert robust_coverage >= COVERAGE_FLOOR
        assert naive_coverage < 0.80


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
