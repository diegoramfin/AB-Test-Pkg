"""Tests for frequentist parametric tests."""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from twosample_means.config import RunConfig
from twosample_means.frequentist_parametric import (
    MissingVarianceError,
    ParametricResult,
    students_t,
    welch_t,
    z_test,
)


@pytest.fixture
def config() -> RunConfig:
    """Default RunConfig."""
    return RunConfig()


@pytest.fixture
def two_samples() -> tuple[np.ndarray, np.ndarray]:
    """Fixed-seed two samples from normal distributions."""
    rng = np.random.default_rng(42)
    a = rng.normal(5.0, 1.0, size=100)
    b = rng.normal(5.5, 1.0, size=100)
    return a, b


class TestStudentsT:
    """Tests for Student's t-test."""

    def test_matches_scipy(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Result matches scipy.stats.ttest_ind(equal_var=True)."""
        a, b = two_samples
        result = students_t(a, b, config)
        expected = scipy_stats.ttest_ind(a, b, equal_var=True)
        assert isinstance(result, ParametricResult)
        assert abs(result.statistic - expected.statistic) < 1e-10
        assert abs(result.p_value - expected.pvalue) < 1e-10
        assert result.degrees_of_freedom == pytest.approx(expected.df)

    def test_ci_brackets_mean_diff(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """CI must bracket the mean difference."""
        a, b = two_samples
        result = students_t(a, b, config)
        assert result.ci_lower < result.mean_difference
        assert result.mean_difference < result.ci_upper

    def test_citation_present(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Citation must be present and contain 'Student'."""
        a, b = two_samples
        result = students_t(a, b, config)
        assert "Student" in result.citation
        assert "1908" in result.citation


class TestWelchT:
    """Tests for Welch's t-test."""

    def test_matches_scipy(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Result matches scipy.stats.ttest_ind(equal_var=False)."""
        a, b = two_samples
        result = welch_t(a, b, config)
        expected = scipy_stats.ttest_ind(a, b, equal_var=False)
        assert abs(result.statistic - expected.statistic) < 1e-10
        assert abs(result.p_value - expected.pvalue) < 1e-10

    def test_ci_brackets_mean_diff(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """CI must bracket the mean difference."""
        a, b = two_samples
        result = welch_t(a, b, config)
        assert result.ci_lower < result.mean_difference
        assert result.mean_difference < result.ci_upper

    def test_no_decision(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """No accept/reject decision in the result."""
        a, b = two_samples
        result = welch_t(a, b, config)
        assert "reject" not in result.assumption_notes.lower()


class TestZTest:
    """Tests for the z-test."""

    def test_with_known_variance(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """z-test with known variance returns valid result."""
        a, b = two_samples
        config = RunConfig(
            population_variance_a=1.0,
            population_variance_b=1.0,
        )
        result = z_test(a, b, config)
        assert result.method_name == "z-test"
        assert np.isfinite(result.statistic)
        assert 0.0 <= result.p_value <= 1.0
        assert result.degrees_of_freedom is None

    def test_missing_variance_raises(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Missing variance raises MissingVarianceError."""
        a, b = two_samples
        with pytest.raises(MissingVarianceError):
            z_test(a, b, config)

    def test_ci_brackets_mean_diff(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """CI brackets the mean difference."""
        a, b = two_samples
        config = RunConfig(
            population_variance_a=1.0,
            population_variance_b=1.0,
        )
        result = z_test(a, b, config)
        assert result.ci_lower < result.mean_difference
        assert result.mean_difference < result.ci_upper
