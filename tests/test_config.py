"""Tests for config dataclasses and citations registry."""

import pytest

from twosample_means.citations import CITATIONS, format_citation, get_citation
from twosample_means.config import InputSpec, RunConfig


class TestRunConfig:
    """Tests for RunConfig validation and defaults."""

    def test_defaults_are_valid(self) -> None:
        """Default RunConfig constructs without error."""
        config = RunConfig()
        assert config.alpha == 0.05
        assert config.ci_level == 0.95
        assert config.hdi_mass == 0.95
        assert config.mcmc_draws == 2000
        assert config.mcmc_chains == 4
        assert config.seed == 42

    def test_is_frozen(self) -> None:
        """RunConfig must be immutable."""
        config = RunConfig()
        with pytest.raises(AttributeError):
            config.alpha = 0.01  # type: ignore[misc]

    def test_invalid_alpha_rejected(self) -> None:
        """Alpha outside (0, 1) must raise."""
        with pytest.raises(ValueError, match="alpha"):
            RunConfig(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            RunConfig(alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            RunConfig(alpha=-0.1)

    def test_invalid_ci_level_rejected(self) -> None:
        """CI level outside (0, 1) must raise."""
        with pytest.raises(ValueError, match="ci_level"):
            RunConfig(ci_level=0.0)

    def test_invalid_rope_width_rejected(self) -> None:
        """Non-positive rope width must raise."""
        with pytest.raises(ValueError, match="rope_width"):
            RunConfig(rope_width=0.0)
        with pytest.raises(ValueError, match="rope_width"):
            RunConfig(rope_width=-0.01)

    def test_invalid_mcmc_draws_rejected(self) -> None:
        """Too few MCMC draws must raise."""
        with pytest.raises(ValueError, match="mcmc_draws"):
            RunConfig(mcmc_draws=10)

    def test_invalid_mcmc_chains_rejected(self) -> None:
        """Fewer than 2 chains must raise."""
        with pytest.raises(ValueError, match="mcmc_chains"):
            RunConfig(mcmc_chains=1)

    def test_invalid_outlier_method_rejected(self) -> None:
        """Unknown outlier method must raise."""
        with pytest.raises(ValueError, match="outlier_method"):
            RunConfig(outlier_method="invalid")

    def test_custom_values_accepted(self) -> None:
        """Valid custom values must be accepted."""
        config = RunConfig(
            alpha=0.01,
            ci_level=0.99,
            mcmc_draws=4000,
            seed=123,
        )
        assert config.alpha == 0.01
        assert config.ci_level == 0.99
        assert config.mcmc_draws == 4000
        assert config.seed == 123


class TestInputSpec:
    """Tests for InputSpec construction."""

    def test_from_arrays(self) -> None:
        """InputSpec accepts in-memory arrays."""
        spec = InputSpec(
            sample_a=[1.0, 2.0, 3.0],
            sample_b=[4.0, 5.0, 6.0],
        )
        assert list(spec.sample_a) == [1.0, 2.0, 3.0]
        assert list(spec.sample_b) == [4.0, 5.0, 6.0]

    def test_from_paths(self) -> None:
        """InputSpec accepts file paths."""
        spec = InputSpec(
            sample_a="data_a.csv",
            sample_b="data_b.csv",
        )
        assert spec.sample_a == "data_a.csv"
        assert spec.column_a is None

    def test_with_columns(self) -> None:
        """InputSpec accepts column names."""
        spec = InputSpec(
            sample_a="data.csv",
            sample_b="data.csv",
            column_a="control",
            column_b="treatment",
        )
        assert spec.column_a == "control"
        assert spec.column_b == "treatment"

    def test_is_frozen(self) -> None:
        """InputSpec must be immutable."""
        spec = InputSpec(
            sample_a=[1.0],
            sample_b=[2.0],
        )
        with pytest.raises(AttributeError):
            spec.column_a = "x"  # type: ignore[misc]


class TestCitationsRegistry:
    """Tests for the citations registry."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "shapiro",
            "anderson_darling",
            "dagostino_k2",
            "levene",
            "bartlett",
            "brown_forsythe",
            "students_t",
            "welch_t",
            "z_test",
            "mann_whitney",
            "brunner_munzel",
            "permutation",
            "bootstrap_ci",
            "best",
            "bayes_factor_jzs",
            "cohen_d",
            "hedges_g",
            "cliff_delta",
            "rank_biserial",
            "hodges_lehmann",
        ],
    )
    def test_citation_exists(self, method_name: str) -> None:
        """Every battery method must have a citation."""
        cite = get_citation(method_name)
        assert cite["authors"]
        assert cite["year"] > 1900
        assert cite["title"]
        assert cite["source"]

    def test_missing_citation_raises(self) -> None:
        """Unknown method must raise KeyError."""
        with pytest.raises(KeyError, match="no_citation"):
            get_citation("no_citation")

    def test_format_citation(self) -> None:
        """Formatted citation contains all fields."""
        formatted = format_citation("students_t")
        assert "Student" in formatted
        assert "1908" in formatted
        assert "Biometrika" in formatted

    def test_all_methods_covered(self) -> None:
        """Registry must have at least 20 methods."""
        assert len(CITATIONS) >= 20
