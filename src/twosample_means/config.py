"""Configuration dataclasses for the two-sample testing procedure.

This module defines ``RunConfig`` and ``InputSpec``, the two configuration
objects that parameterise every method in the battery. No tunable
parameter is hardcoded inside a method module — all thresholds, iteration
counts, and seeds flow from a ``RunConfig`` instance.

Design rationale
----------------
Centralising configuration in frozen dataclasses enforces the
anti-p-hacking discipline: the analyst declares all parameters up front
in one place, making it auditable and impossible to silently tweak a
threshold inside a method to chase a desired result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Literal

import numpy.typing as npt

SampleSource = str | Path | npt.ArrayLike | None
MissingValuePolicy = Literal["error", "exclude"]


@dataclass(frozen=True)
class RunConfig:
    """All tunable parameters for a single battery run.

    Every field has a documented default rooted in standard statistical
    practice. The analyst may override any field at construction time,
    but the values are frozen thereafter — preventing mid-analysis
    parameter manipulation (an anti-p-hacking safeguard).

    Attributes
    ----------
    alpha:
        Significance level used for assumption checks and confidence
        interval complement. Default 0.05 per Fisher (1925) and the
        near-universal convention in experimental research.
    ci_level:
        Confidence level for frequentist intervals (e.g., 0.95 for a
        95% CI). Must satisfy ``0 < ci_level < 1``.
    hdi_mass:
        Highest-density interval mass for Bayesian posteriors
        (e.g., 0.95 for a 95% HDI). Following Kruschke (2013), the HDI
        is preferred over equal-tailed intervals for posterior summaries.
    rope_width:
        Half-width of the Region of Practical Equivalence (ROPE) around
        zero for the mean difference. Following Kruschke (2013), the
        ROPE defines a range of values considered practically equivalent
        to no effect. When ``rope_scale`` is ``"auto"``, this value is
        multiplied by the pooled SD of the data to produce a
        scale-appropriate ROPE. When ``rope_scale`` is ``"fixed"``, this
        value is used directly (in the raw units of the data).
    rope_scale:
        ``"auto"`` (default) scales ``rope_width`` by the pooled SD,
        following Kruschke's (2013) recommendation of 0.1 * SD as a
        default ROPE. ``"fixed"`` uses ``rope_width`` as-is in raw data
        units. The analyst should use ``"auto"`` unless they have domain
        knowledge specifying a raw-unit ROPE.
    mcmc_draws:
        Number of posterior draws per chain for PyMC sampling. Default
        2000 follows Kruschke (2013) and PyMC best practice for stable
        R-hat and adequate ESS.
    mcmc_chains:
        Number of independent MCMC chains. Default 4 follows the
        Gelman et al. (2013) recommendation for reliable convergence
        diagnostics.
    permutation_iterations:
        Number of Monte Carlo permutations when exact enumeration is
        infeasible. Default 9999 follows the Phipson & Smyth (2010)
        recommendation for stable p-value estimates.
    bootstrap_iterations:
        Number of bootstrap resamples for the bootstrap CI. Default
        9999 follows Efron & Tibshirani (1993) for stable CI bounds.
    seed:
        Random seed for all stochastic methods (permutation, bootstrap,
        MCMC). Fixed default ensures reproducibility — a core
        auditability requirement.
    missing_values:
        How legacy numeric samples handle missing values. ``"error"``
        preserves the strict default; ``"exclude"`` removes NaN values
        before analysis while continuing to reject infinite values.
    population_variance_a:
        Known population variance for sample A, required only by the
        z-test. ``None`` means unknown; the z-test will raise an
        explicit error if invoked without this.
    population_variance_b:
        Known population variance for sample B, required only by the
        z-test. If ``None`` while ``population_variance_a`` is set,
        the z-test assumes equal known variances.
    bayes_factor_prior_width:
        Prior width (``r`` scale) for the JZS Bayes factor, following
        Rouder et al. (2009). Default ``sqrt(2)/2`` is the
        "medium" default recommended by Rouder et al.
    outlier_method:
        Method for outlier flagging: ``"iqr"`` (Tukey 1977) or
        ``"zscore"``.
    outlier_threshold:
        Threshold for the chosen outlier method: for IQR, the
        multiplier (default 1.5 per Tukey 1977); for z-score, the
        absolute z cutoff (default 3.0).
    include_bayesian:
        Whether the runner should execute Bayesian methods. Defaults to
        ``True`` for library compatibility; scalable clients can disable it.
    include_resampling:
        Whether permutation and bootstrap methods should run.
    max_bayesian_observations:
        Maximum combined sample size allowed for Bayesian methods.
    max_diagnostic_observations:
        Maximum sample size for Monte Carlo Anderson-Darling diagnostics.
    max_pairwise_comparisons:
        Maximum exact pairwise differences allowed for Hodges-Lehmann.
    max_resampling_operations:
        Maximum approximate observation-iterations for resampling methods.

    """

    alpha: float = 0.05
    ci_level: float = 0.95
    hdi_mass: float = 0.95
    rope_width: float = 0.1
    rope_scale: str = "auto"
    mcmc_draws: int = 2000
    mcmc_chains: int = 4
    permutation_iterations: int = 9999
    bootstrap_iterations: int = 9999
    seed: int = 42
    missing_values: MissingValuePolicy = "error"
    population_variance_a: float | None = None
    population_variance_b: float | None = None
    bayes_factor_prior_width: float = 0.707
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    include_bayesian: bool = True
    include_resampling: bool = True
    max_bayesian_observations: int = 10_000
    max_diagnostic_observations: int = 10_000
    max_pairwise_comparisons: int = 5_000_000
    max_resampling_operations: int = 10_000_000

    def __post_init__(self) -> None:
        """Validate parameter ranges at construction time."""
        for name, value in (
            ("alpha", self.alpha),
            ("ci_level", self.ci_level),
            ("hdi_mass", self.hdi_mass),
        ):
            if not isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if not isfinite(self.rope_width) or self.rope_width <= 0.0:
            raise ValueError(
                f"rope_width must be positive, got {self.rope_width}"
            )
        if self.rope_scale not in ("auto", "fixed"):
            raise ValueError(
                f"rope_scale must be 'auto' or 'fixed', "
                f"got '{self.rope_scale}'"
            )
        if self.outlier_method not in ("iqr", "zscore"):
            raise ValueError(
                f"outlier_method must be 'iqr' or "
                f"'zscore', got {self.outlier_method}"
            )
        if not isfinite(self.outlier_threshold) or (
            self.outlier_threshold <= 0.0
        ):
            raise ValueError(
                "outlier_threshold must be positive and finite, "
                f"got {self.outlier_threshold}"
            )
        for name, variance in (
            ("population_variance_a", self.population_variance_a),
            ("population_variance_b", self.population_variance_b),
        ):
            if variance is not None and (
                not isfinite(variance) or variance <= 0.0
            ):
                raise ValueError(
                    f"{name} must be positive and finite, got {variance}"
                )
        if not isfinite(self.bayes_factor_prior_width) or (
            self.bayes_factor_prior_width <= 0.0
        ):
            raise ValueError(
                "bayes_factor_prior_width must be positive and finite, "
                f"got {self.bayes_factor_prior_width}"
            )
        for name, value in (
            ("mcmc_draws", self.mcmc_draws),
            ("mcmc_chains", self.mcmc_chains),
            ("permutation_iterations", self.permutation_iterations),
            ("bootstrap_iterations", self.bootstrap_iterations),
            ("max_bayesian_observations", self.max_bayesian_observations),
            ("max_diagnostic_observations", self.max_diagnostic_observations),
            ("max_pairwise_comparisons", self.max_pairwise_comparisons),
            ("max_resampling_operations", self.max_resampling_operations),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer, got {value}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError(f"seed must be an integer, got {self.seed}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.missing_values not in ("error", "exclude"):
            raise ValueError(
                "missing_values must be 'error' or 'exclude', "
                f"got {self.missing_values!r}"
            )
        if self.mcmc_draws < 100:
            raise ValueError(
                f"mcmc_draws must be >= 100, got {self.mcmc_draws}"
            )
        if self.mcmc_chains < 2:
            raise ValueError(
                f"mcmc_chains must be >= 2, got {self.mcmc_chains}"
            )
        if self.permutation_iterations < 100:
            raise ValueError(
                "permutation_iterations must be >= 100, "
                f"got {self.permutation_iterations}"
            )
        if self.bootstrap_iterations < 100:
            raise ValueError(
                "bootstrap_iterations must be >= 100, "
                f"got {self.bootstrap_iterations}"
            )
        if not isinstance(self.include_bayesian, bool):
            raise ValueError("include_bayesian must be a bool")
        if not isinstance(self.include_resampling, bool):
            raise ValueError("include_resampling must be a bool")


@dataclass(frozen=True)
class InputSpec:
    """Specification of the two input samples.

    Accepts either file paths (CSV or parquet) or in-memory array-likes
    for each sample. When file paths are used, ``column_a`` and
    ``column_b`` specify which columns to read (defaults to the first
    data column).

    The spec is frozen so the data source cannot be swapped after
    construction — an auditability safeguard.

    Attributes
    ----------
    sample_a:
        Path to a CSV/parquet file, or an in-memory sequence of numbers
        for sample A (the "control" or first group).
    sample_b:
        Path to a CSV/parquet file, or an in-memory sequence of numbers
        for sample B (the "test" or second group).
    column_a:
        Column name to read from ``sample_a`` when it is a file path.
        If ``None``, the first numeric column is used.
    column_b:
        Column name to read from ``sample_b`` when it is a file path.
        If ``None``, the first numeric column is used.
    missing_values:
        Default NaN policy for ``data_io.load``. ``"error"`` is strict;
        ``"exclude"`` removes NaNs before analysis.

    """

    sample_a: SampleSource
    sample_b: SampleSource
    column_a: str | None = None
    column_b: str | None = None
    missing_values: MissingValuePolicy = "error"

    def __post_init__(self) -> None:
        """Validate input specification at construction time."""
        if self.sample_a is None:
            raise ValueError("sample_a must not be None")
        if self.sample_b is None:
            raise ValueError("sample_b must not be None")
        if self.column_a is not None and not isinstance(self.column_a, str):
            raise ValueError(
                f"column_a must be a str or None, "
                f"got {type(self.column_a).__name__}"
            )
        if self.column_b is not None and not isinstance(self.column_b, str):
            raise ValueError(
                f"column_b must be a str or None, "
                f"got {type(self.column_b).__name__}"
            )
        if self.missing_values not in ("error", "exclude"):
            raise ValueError(
                "missing_values must be 'error' or 'exclude', "
                f"got {self.missing_values!r}"
            )
