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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    population_variance_a: float | None = None
    population_variance_b: float | None = None
    bayes_factor_prior_width: float = 0.707
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5

    def __post_init__(self) -> None:
        """Validate parameter ranges at construction time."""
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError(
                f"ci_level must be in (0, 1), got {self.ci_level}"
            )
        if not 0.0 < self.hdi_mass < 1.0:
            raise ValueError(
                f"hdi_mass must be in (0, 1), got {self.hdi_mass}"
            )
        if self.rope_width <= 0.0:
            raise ValueError(
                f"rope_width must be positive, got {self.rope_width}"
            )
        if self.rope_scale not in ("auto", "fixed"):
            raise ValueError(
                f"rope_scale must be 'auto' or 'fixed', "
                f"got '{self.rope_scale}'"
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
        if self.outlier_method not in ("iqr", "zscore"):
            raise ValueError(
                f"outlier_method must be 'iqr' or "
                f"'zscore', got {self.outlier_method}"
            )
        if self.outlier_threshold <= 0.0:
            raise ValueError(
                "outlier_threshold must be positive, "
                f"got {self.outlier_threshold}"
            )


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
    """

    sample_a: str | Path | Sequence[Any]
    sample_b: str | Path | Sequence[Any]
    column_a: str | None = None
    column_b: str | None = None
