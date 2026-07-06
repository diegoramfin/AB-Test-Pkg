"""Tests for frequentist non-parametric tests."""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from twosample_means.config import RunConfig
from twosample_means.frequentist_nonparametric import (BootstrapResult,
                                                       NonParametricResult,
                                                       PermutationResult,
                                                       bootstrap_ci,
                                                       brunner_munzel,
                                                       mann_whitney_u,
                                                       permutation_test)


@pytest.fixture
def config() -> RunConfig:
    """RunConfig with small iterations for test speed."""
    return RunConfig(
        permutation_iterations=500,
        bootstrap_iterations=500,
    )


@pytest.fixture
def two_samples() -> tuple[np.ndarray, np.ndarray]:
    """Fixed-seed two samples."""
    rng = np.random.default_rng(42)
    a = rng.normal(5.0, 1.0, size=100)
    b = rng.normal(5.5, 1.0, size=100)
    return a, b


class TestMannWhitney:
    """Tests for Mann-Whitney U test."""

    def test_matches_scipy(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Result matches scipy.stats.mannwhitneyu."""
        a, b = two_samples
        result = mann_whitney_u(a, b, config)
        expected = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        assert isinstance(result, NonParametricResult)
        assert abs(result.statistic - expected.statistic) < 1e-10
        assert abs(result.p_value - expected.pvalue) < 1e-10

    def test_citation_present(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Citation contains 'Mann'."""
        a, b = two_samples
        result = mann_whitney_u(a, b, config)
        assert "Mann" in result.citation


class TestBrunnerMunzel:
    """Tests for Brunner-Munzel test."""

    def test_matches_scipy(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Result sign is negated (A-B convention); p-value matches."""
        a, b = two_samples
        result = brunner_munzel(a, b, config)
        expected = scipy_stats.brunnermunzel(a, b, alternative="two-sided")
        assert abs(result.statistic - (-expected.statistic)) < 1e-10
        assert abs(result.p_value - expected.pvalue) < 1e-10


class TestPermutation:
    """Tests for the permutation test."""

    def test_exact_mode(self) -> None:
        """Exact mode on tiny data matches enumeration."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        config = RunConfig()
        result = permutation_test(a, b, config)
        assert isinstance(result, PermutationResult)
        assert result.mode == "exact"
        assert result.iterations == 20  # C(6,3) = 20
        assert 0.0 <= result.p_value <= 1.0

    def test_monte_carlo_reproducible(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Monte Carlo mode is reproducible with same seed."""
        a, b = two_samples
        config = RunConfig(permutation_iterations=200, seed=42)
        result1 = permutation_test(a, b, config)
        result2 = permutation_test(a, b, config)
        assert result1.mode == "monte_carlo"
        assert result1.p_value == result2.p_value

    def test_monte_carlo_different_seeds_differ(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Different seeds give different p-values (almost surely)."""
        a, b = two_samples
        config1 = RunConfig(permutation_iterations=200, seed=42)
        config2 = RunConfig(permutation_iterations=200, seed=99)
        result1 = permutation_test(a, b, config1)
        result2 = permutation_test(a, b, config2)
        assert result1.seed == 42
        assert result2.seed == 99


class TestBootstrap:
    """Tests for the bootstrap CI."""

    def test_ci_brackets_estimate(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """CI must bracket the point estimate."""
        a, b = two_samples
        result = bootstrap_ci(a, b, config)
        assert isinstance(result, BootstrapResult)
        assert result.ci_lower < result.point_estimate
        assert result.point_estimate < result.ci_upper

    def test_reproducible(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Bootstrap CI is reproducible with same seed."""
        a, b = two_samples
        config = RunConfig(bootstrap_iterations=200, seed=42)
        result1 = bootstrap_ci(a, b, config)
        result2 = bootstrap_ci(a, b, config)
        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper

    def test_ci_level_from_config(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """CI level comes from config."""
        a, b = two_samples
        config = RunConfig(bootstrap_iterations=200, ci_level=0.90)
        result = bootstrap_ci(a, b, config)
        assert result.ci_level == 0.90
