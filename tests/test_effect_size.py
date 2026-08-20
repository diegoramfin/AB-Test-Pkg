"""Tests for effect-size computations."""

import numpy as np
import pytest

from twosample_means.config import RunConfig
from twosample_means.effect_size import (
    EffectSizeResult,
    ResourceLimitError,
    cliff_delta,
    cohens_d,
    hedges_g,
    hodges_lehmann,
    rank_biserial,
)


@pytest.fixture
def config() -> RunConfig:
    """Default RunConfig."""
    return RunConfig()


@pytest.fixture
def two_samples() -> tuple[np.ndarray, np.ndarray]:
    """Fixed-seed two samples with a real difference."""
    rng = np.random.default_rng(42)
    a = rng.normal(5.0, 1.0, size=50)
    b = rng.normal(5.5, 1.0, size=50)
    return a, b


class TestCohensD:
    """Tests for Cohen's d."""

    def test_returns_valid_result(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Cohen's d returns a valid result."""
        a, b = two_samples
        result = cohens_d(a, b, config)
        assert isinstance(result, EffectSizeResult)
        assert np.isfinite(result.point_estimate)
        assert result.ci_lower < result.ci_upper
        assert "Cohen" in result.citation

    def test_zero_for_identical_samples(self, config: RunConfig) -> None:
        """Cohen's d is near zero for identical samples."""
        rng = np.random.default_rng(42)
        a = rng.normal(5.0, 1.0, size=100)
        result = cohens_d(a, a, config)
        assert abs(result.point_estimate) < 0.1


class TestHedgesG:
    """Tests for Hedges' g."""

    def test_returns_valid_result(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Hedges' g returns a valid result."""
        a, b = two_samples
        result = hedges_g(a, b, config)
        assert np.isfinite(result.point_estimate)
        assert result.ci_lower < result.ci_upper
        assert "Hedges" in result.citation


class TestCliffDelta:
    """Tests for Cliff's delta."""

    def test_returns_valid_result(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Cliff's delta returns a valid result."""
        a, b = two_samples
        result = cliff_delta(a, b, config)
        assert -1.0 <= result.point_estimate <= 1.0
        assert result.ci_lower < result.ci_upper
        assert "Cliff" in result.citation


class TestRankBiserial:
    """Tests for rank-biserial correlation."""

    def test_returns_valid_result(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Rank-biserial returns a valid result."""
        a, b = two_samples
        result = rank_biserial(a, b, config)
        assert -1.0 <= result.point_estimate <= 1.0
        assert result.ci_lower < result.ci_upper
        assert "Kerby" in result.citation


class TestHodgesLehmann:
    """Tests for the Hodges-Lehmann estimator."""

    def test_returns_valid_result(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """HL estimator returns a valid result."""
        a, b = two_samples
        result = hodges_lehmann(a, b, config)
        assert np.isfinite(result.point_estimate)
        assert result.ci_lower < result.ci_upper
        assert "Hodges" in result.citation

    def test_ci_is_bounded_and_ordered(self) -> None:
        """The rank-based CI stays within the pairwise differences."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = hodges_lehmann(a, b, RunConfig())
        assert -5.0 <= result.ci_lower < result.ci_upper <= -1.0

    def test_pairwise_budget_is_enforced(self) -> None:
        """Large exact HL computations fail before outer allocation."""
        config = RunConfig(max_pairwise_comparisons=8)
        with pytest.raises(ResourceLimitError, match="pairwise budget"):
            hodges_lehmann(
                np.arange(3, dtype=float),
                np.arange(3, dtype=float),
                config,
            )


class TestEffectSizeEdgeCases:
    """Edge cases for bounded effect-size CIs."""

    def test_cliff_delta_complete_separation(self, config: RunConfig) -> None:
        """Cliff's delta at complete separation has a non-point CI."""
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([1.0, 2.0, 3.0])
        result = cliff_delta(a, b, config)
        assert result.point_estimate == 1.0
        assert -1.0 <= result.ci_lower < result.ci_upper <= 1.0
        assert result.ci_lower < 1.0

    def test_rank_biserial_complete_separation(
        self, config: RunConfig
    ) -> None:
        """Rank-biserial at complete separation has a non-point CI."""
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([1.0, 2.0, 3.0])
        result = rank_biserial(a, b, config)
        assert result.point_estimate == 1.0
        assert -1.0 <= result.ci_lower < result.ci_upper <= 1.0
        assert result.ci_lower < 1.0

    def test_cliff_delta_with_ties(self, config: RunConfig) -> None:
        """Cliff's delta stays inside [-1, 1] with ties."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])
        result = cliff_delta(a, b, config)
        assert -1.0 <= result.point_estimate <= 1.0
        assert -1.0 <= result.ci_lower < result.ci_upper <= 1.0

    def test_rank_biserial_with_ties(self, config: RunConfig) -> None:
        """Rank-biserial stays inside [-1, 1] with ties."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])
        result = rank_biserial(a, b, config)
        assert -1.0 <= result.point_estimate <= 1.0
        assert -1.0 <= result.ci_lower < result.ci_upper <= 1.0
