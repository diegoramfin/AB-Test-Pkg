"""Simulation-based coverage checks for effect-size intervals."""

import numpy as np
from scipy import stats

from twosample_means.config import RunConfig
from twosample_means.effect_size import (
    cliff_delta,
    cohens_d,
    hedges_g,
    hodges_lehmann,
    rank_biserial,
)

SEED = 20260819
REPLICATES = 160
COVERAGE_FLOOR = 0.85


def test_dominance_effect_intervals_have_nominal_coverage() -> None:
    """Cliff and rank-biserial intervals cover their population effect."""
    rng = np.random.default_rng(SEED)
    config = RunConfig(ci_level=0.95)
    true_delta = float(2.0 * stats.norm.cdf(0.5 / np.sqrt(2.0)) - 1.0)
    covered_cliff = 0
    covered_rank = 0

    for _ in range(REPLICATES):
        sample_a = rng.normal(0.5, 1.0, size=30)
        sample_b = rng.normal(0.0, 1.0, size=30)
        cliff = cliff_delta(sample_a, sample_b, config)
        rank = rank_biserial(sample_a, sample_b, config)
        covered_cliff += cliff.ci_lower <= true_delta <= cliff.ci_upper
        covered_rank += rank.ci_lower <= true_delta <= rank.ci_upper

    assert covered_cliff / REPLICATES >= COVERAGE_FLOOR
    assert covered_rank / REPLICATES >= COVERAGE_FLOOR


def test_standardized_mean_effect_intervals_have_nominal_coverage() -> None:
    """Cohen's d and Hedges' g intervals cover a known normal effect."""
    rng = np.random.default_rng(SEED)
    config = RunConfig(ci_level=0.95)
    covered_cohen = 0
    covered_hedges = 0

    for _ in range(REPLICATES):
        sample_a = rng.normal(0.5, 1.0, size=40)
        sample_b = rng.normal(0.0, 1.0, size=40)
        cohen = cohens_d(sample_a, sample_b, config)
        hedges = hedges_g(sample_a, sample_b, config)
        covered_cohen += cohen.ci_lower <= 0.5 <= cohen.ci_upper
        covered_hedges += hedges.ci_lower <= 0.5 <= hedges.ci_upper

    assert covered_cohen / REPLICATES >= COVERAGE_FLOOR
    assert covered_hedges / REPLICATES >= COVERAGE_FLOOR


def test_hodges_lehmann_intervals_have_nominal_coverage() -> None:
    """The two-sample location-shift interval covers a known shift."""
    rng = np.random.default_rng(SEED)
    config = RunConfig(ci_level=0.95)
    covered = 0

    for _ in range(REPLICATES):
        sample_a = rng.normal(0.5, 1.0, size=20)
        sample_b = rng.normal(0.0, 1.0, size=20)
        result = hodges_lehmann(sample_a, sample_b, config)
        covered += result.ci_lower <= 0.5 <= result.ci_upper

    assert covered / REPLICATES >= COVERAGE_FLOOR
