"""Tests for effect-size computations."""

import numpy as np
import pytest

from twosample_means.config import RunConfig
from twosample_means.effect_size import (
    EffectSizeResult,
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
