"""Tests for Bayesian tests."""

import numpy as np
import pytest

from twosample_means.bayesian import (
    BayesFactorResult,
    BESTResult,
    bayes_factor_jzs,
    best,
)
from twosample_means.config import RunConfig


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


class TestBayesFactor:
    """Tests for JZS Bayes factor."""

    def test_returns_valid_result(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Bayes factor returns a valid result."""
        a, b = two_samples
        result = bayes_factor_jzs(a, b, config)
        assert isinstance(result, BayesFactorResult)
        assert result.bf10 > 0
        assert result.bf01 == pytest.approx(1.0 / result.bf10)
        assert result.prior_width == config.bayes_factor_prior_width

    def test_citation_present(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Citation contains 'Rouder'."""
        a, b = two_samples
        result = bayes_factor_jzs(a, b, config)
        assert "Rouder" in result.citation


class TestBEST:
    """Tests for the BEST Bayesian t-test."""

    def test_returns_valid_result(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """BEST returns a valid result with MCMC diagnostics."""
        a, b = two_samples
        config = RunConfig(mcmc_draws=200, mcmc_chains=2, seed=42)
        result = best(a, b, config)
        assert isinstance(result, BESTResult)
        assert np.isfinite(result.posterior_mean_diff)
        assert result.hdi_lower < result.hdi_upper
        assert 0.0 <= result.rope_proportion <= 1.0
        assert result.r_hat < 1.1  # convergence
        assert result.ess > 100  # adequate ESS

    def test_no_decision(
        self,
        two_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """No accept/reject decision in the result."""
        a, b = two_samples
        config = RunConfig(mcmc_draws=200, mcmc_chains=2, seed=42)
        result = best(a, b, config)
        assert "reject" not in result.assumption_notes.lower()
