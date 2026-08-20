"""Tests for assumption diagnostics."""

import numpy as np
import pytest

from twosample_means.assumptions import (
    DiagnosticResult,
    OutlierResult,
    anderson_darling,
    bartlett,
    brown_forsythe,
    dagostino_k2,
    flag_outliers,
    levene,
    shapiro_wilk,
)
from twosample_means.config import RunConfig


@pytest.fixture
def config() -> RunConfig:
    """Default RunConfig for tests."""
    return RunConfig()


@pytest.fixture
def normal_data() -> np.ndarray:
    """Fixed-seed normal data."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 1.0, size=200)


@pytest.fixture
def skewed_data() -> np.ndarray:
    """Fixed-seed lognormal (skewed) data."""
    rng = np.random.default_rng(42)
    return rng.lognormal(0.0, 1.0, size=200)


@pytest.fixture
def equal_var_data() -> tuple[np.ndarray, np.ndarray]:
    """Two samples with equal variance."""
    rng = np.random.default_rng(42)
    a = rng.normal(0.0, 1.0, size=200)
    b = rng.normal(1.0, 1.0, size=200)
    return a, b


@pytest.fixture
def unequal_var_data() -> tuple[np.ndarray, np.ndarray]:
    """Two samples with unequal variance."""
    rng = np.random.default_rng(42)
    a = rng.normal(0.0, 1.0, size=200)
    b = rng.normal(1.0, 3.0, size=200)
    return a, b


class TestNormality:
    """Tests for normality diagnostics."""

    def test_shapiro_normal_data(
        self, normal_data: np.ndarray, config: RunConfig
    ) -> None:
        """Shapiro-Wilk on normal data returns valid result."""
        result = shapiro_wilk(normal_data, config)
        assert isinstance(result, DiagnosticResult)
        assert result.method_name == "Shapiro-Wilk"
        assert np.isfinite(result.statistic)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0
        assert "Shapiro" in result.citation

    def test_shapiro_skewed_data(
        self, skewed_data: np.ndarray, config: RunConfig
    ) -> None:
        """Shapiro-Wilk on skewed data returns valid result."""
        result = shapiro_wilk(skewed_data, config)
        assert np.isfinite(result.statistic)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0

    def test_anderson_darling_normal(
        self, normal_data: np.ndarray, config: RunConfig
    ) -> None:
        """Anderson-Darling on normal data returns valid result."""
        result = anderson_darling(normal_data, config)
        assert result.method_name == "Anderson-Darling"
        assert np.isfinite(result.statistic)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0
        assert "Anderson" in result.citation

    def test_dagostino_normal(
        self, normal_data: np.ndarray, config: RunConfig
    ) -> None:
        """D'Agostino K² on normal data returns valid result."""
        result = dagostino_k2(normal_data, config)
        assert result.method_name == "D'Agostino K²"
        assert np.isfinite(result.statistic)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0
        assert "D'Agostino" in result.citation

    def test_dagostino_skewed(
        self, skewed_data: np.ndarray, config: RunConfig
    ) -> None:
        """D'Agostino K² on skewed data returns valid result."""
        result = dagostino_k2(skewed_data, config)
        assert np.isfinite(result.statistic)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0

    def test_no_decision_in_results(
        self, normal_data: np.ndarray, config: RunConfig
    ) -> None:
        """Results must not contain accept/reject decisions."""
        for func in (shapiro_wilk, anderson_darling, dagostino_k2):
            result = func(normal_data, config)
            assert "reject" not in result.details.lower()
            assert "significant" not in result.details.lower()


class TestVarianceHomogeneity:
    """Tests for variance homogeneity tests."""

    def test_levene_equal_var(
        self,
        equal_var_data: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Levene on equal-variance data returns valid result."""
        a, b = equal_var_data
        result = levene(a, b, config)
        assert result.method_name == "Levene"
        assert np.isfinite(result.statistic)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0
        assert "Levene" in result.citation

    def test_levene_unequal_var(
        self,
        unequal_var_data: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Levene on unequal-variance data returns valid result."""
        a, b = unequal_var_data
        result = levene(a, b, config)
        assert np.isfinite(result.statistic)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0

    def test_bartlett_equal_var(
        self,
        equal_var_data: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Bartlett on equal-variance data returns valid result."""
        a, b = equal_var_data
        result = bartlett(a, b, config)
        assert result.method_name == "Bartlett"
        assert np.isfinite(result.statistic)
        assert "Bartlett" in result.citation

    def test_brown_forsythe_unequal_var(
        self,
        unequal_var_data: tuple[np.ndarray, np.ndarray],
        config: RunConfig,
    ) -> None:
        """Brown-Forsythe on unequal-variance data."""
        a, b = unequal_var_data
        result = brown_forsythe(a, b, config)
        assert result.method_name == "Brown-Forsythe"
        assert np.isfinite(result.statistic)
        assert "Brown" in result.citation


class TestOutliers:
    """Tests for outlier flagging."""

    def test_iqr_finds_outliers(self, config: RunConfig) -> None:
        """IQR method flags injected outliers."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0])
        result = flag_outliers(data, config)
        assert isinstance(result, OutlierResult)
        assert result.count >= 1
        assert 7 in result.indices

    def test_does_not_mutate_data(self, config: RunConfig) -> None:
        """Outlier flagging must not modify the input data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0])
        original = data.copy()
        flag_outliers(data, config)
        np.testing.assert_array_equal(data, original)

    def test_zscore_method(self) -> None:
        """Z-score method flags extreme values."""
        config = RunConfig(outlier_method="zscore", outlier_threshold=2.0)
        data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0])
        result = flag_outliers(data, config)
        assert result.count >= 1
        assert 7 in result.indices

    def test_no_outliers_in_clean_data(self, config: RunConfig) -> None:
        """No outliers flagged in clean normal data."""
        rng = np.random.default_rng(42)
        data = rng.normal(0.0, 1.0, size=100)
        result = flag_outliers(data, config)
        assert result.count < 10
