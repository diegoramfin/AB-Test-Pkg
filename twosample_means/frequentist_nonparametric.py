"""Frequentist non-parametric tests: Mann-Whitney, Brunner-Munzel,
permutation, bootstrap CI.

Each function is a thin wrapper over scipy.stats or a well-established
resampling procedure. Results include the test statistic, p-value,
citation, and assumption notes — but NEVER an accept/reject decision.

Academic rationale
------------------
- Mann-Whitney U (Mann & Whitney, 1947): distribution-free test for
  stochastic dominance. Does not assume normality.
- Brunner-Munzel (Brunner & Munzel, 2000): non-parametric Behrens-
  Fisher problem. Robust when both normality and equal variances
  are violated.
- Permutation test (Fisher, 1935): exact test that exchanges labels.
  The gold standard for non-parametric inference.
- Bootstrap CI (Efron & Tibshirani, 1993): non-parametric CI via
  resampling. Does not assume a specific distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from twosample_means.citations import Citation, get_citation
from twosample_means.config import RunConfig


@dataclass(frozen=True)
class NonParametricResult:
    """Result of a frequentist non-parametric test.

    Attributes
    ----------
    method_name:
        Name of the test.
    citation:
        Academic reference string.
    statistic:
        Test statistic.
    p_value:
        P-value.
    assumption_notes:
        Human-readable notes on assumptions.
    """

    method_name: str
    citation: str
    statistic: float
    p_value: float
    assumption_notes: str


@dataclass(frozen=True)
class PermutationResult:
    """Result of a permutation test.

    Attributes
    ----------
    method_name:
        Name of the test.
    citation:
        Academic reference string.
    statistic:
        Observed test statistic (mean difference).
    p_value:
        Permutation p-value.
    mode:
        "exact" or "monte_carlo".
    iterations:
        Number of permutations evaluated.
    seed:
        Random seed used (for Monte Carlo).
    assumption_notes:
        Human-readable notes on assumptions.
    """

    method_name: str
    citation: str
    statistic: float
    p_value: float
    mode: str
    iterations: int
    seed: int | None
    assumption_notes: str


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a bootstrap CI.

    Attributes
    ----------
    method_name:
        Name of the procedure.
    citation:
        Academic reference string.
    point_estimate:
        Observed mean difference.
    ci_lower:
        Lower CI bound.
    ci_upper:
        Upper CI bound.
    iterations:
        Number of bootstrap resamples.
    seed:
        Random seed used.
    ci_level:
        Confidence level.
    """

    method_name: str
    citation: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    iterations: int
    seed: int
    ci_level: float


def mann_whitney_u(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> NonParametricResult:
    """Perform the Mann-Whitney U test.

    Tests H₀: the distributions of A and B are equal (specifically,
    that a random observation from A is equally likely to be greater
    or less than one from B). Thin wrapper over
    ``scipy.stats.mannwhitneyu``.

    Citation: Mann & Whitney (1947).

    Assumptions: independence. Does not assume normality. Assumes
    ordinal-level measurement.

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration.

    Returns
    -------
    NonParametricResult
        Test result.
    """
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    cite = get_citation("mann_whitney")
    return NonParametricResult(
        method_name="Mann-Whitney U",
        citation=_fmt(cite),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        assumption_notes=("Assumes independence. Does not assume normality."),
    )


def brunner_munzel(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> NonParametricResult:
    """Perform the Brunner-Munzel test.

    Tests H₀: P(X > Y) = P(Y > X) (the relative effect is 0.5).
    Thin wrapper over ``scipy.stats.brunnermunzel``.

    Citation: Brunner & Munzel (2000).

    Assumptions: independence. Does not assume normality or equal
    variances. The non-parametric analogue of the Behrens-Fisher
    problem.

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration.

    Returns
    -------
    NonParametricResult
        Test result.
    """
    result = stats.brunnermunzel(a, b, alternative="two-sided")
    cite = get_citation("brunner_munzel")
    return NonParametricResult(
        method_name="Brunner-Munzel",
        citation=_fmt(cite),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        assumption_notes=(
            "Assumes independence. Does not assume " "normality or equal variances."
        ),
    )


def permutation_test(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> PermutationResult:
    """Perform a permutation test for the mean difference.

    Tests H₀: the labels A and B are exchangeable. Uses exact
    enumeration for small samples (total n <= 20) and Monte Carlo
    resampling for larger samples.

    Citation: Fisher (1935).

    Assumptions: independence, exchangeability under the null.

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration (uses ``permutation_iterations`` and
        ``seed`` for Monte Carlo mode).

    Returns
    -------
    PermutationResult
        Test result with mode, iterations, and seed.
    """
    cite = get_citation("permutation")
    observed_diff = float(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    n_total = len(combined)
    if n_total <= 20:
        return _permutation_exact(combined, n_a, observed_diff, cite)
    return _permutation_montecarlo(
        combined,
        n_a,
        observed_diff,
        config.permutation_iterations,
        config.seed,
        cite,
    )


def bootstrap_ci(a: np.ndarray, b: np.ndarray, config: RunConfig) -> BootstrapResult:
    """Compute a bootstrap CI for the mean difference.

    Resamples each group with replacement and computes the percentile
    CI for the mean difference (A - B).

    Citation: Efron & Tibshirani (1993).

    Assumptions: independence within and between groups. The bootstrap
    approximates the sampling distribution of the mean difference
    without assuming a specific parametric form.

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration (uses ``bootstrap_iterations``,
        ``ci_level``, and ``seed``).

    Returns
    -------
    BootstrapResult
        The point estimate, CI bounds, iterations, and seed.
    """
    cite = get_citation("bootstrap_ci")
    rng = np.random.default_rng(config.seed)
    n_a = len(a)
    n_b = len(b)
    diffs = np.empty(config.bootstrap_iterations)
    for i in range(config.bootstrap_iterations):
        boot_a = rng.choice(a, size=n_a, replace=True)
        boot_b = rng.choice(b, size=n_b, replace=True)
        diffs[i] = np.mean(boot_a) - np.mean(boot_b)
    alpha = 1.0 - config.ci_level
    ci_lo = float(np.percentile(diffs, 100 * alpha / 2))
    ci_hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return BootstrapResult(
        method_name="Bootstrap CI",
        citation=_fmt(cite),
        point_estimate=float(np.mean(a) - np.mean(b)),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        iterations=config.bootstrap_iterations,
        seed=config.seed,
        ci_level=config.ci_level,
    )


def _permutation_exact(
    combined: np.ndarray,
    n_a: int,
    observed_diff: float,
    cite: Citation,
) -> PermutationResult:
    """Exact permutation test via exhaustive enumeration.

    Parameters
    ----------
    combined:
        Pooled observations.
    n_a:
        Size of sample A.
    observed_diff:
        Observed mean difference.
    cite:
        Citation dict.

    Returns
    -------
    PermutationResult
        Exact test result.
    """
    from itertools import combinations

    n_total = len(combined)
    indices = range(n_total)
    count_extreme = 0
    total_perms = 0
    for combo in combinations(indices, n_a):
        mask = np.zeros(n_total, dtype=bool)
        mask[list(combo)] = True
        perm_diff = np.mean(combined[mask]) - np.mean(combined[~mask])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
        total_perms += 1
    p_value = (count_extreme + 1) / (total_perms + 1)
    return PermutationResult(
        method_name="Permutation test (exact)",
        citation=_fmt(cite),
        statistic=observed_diff,
        p_value=float(p_value),
        mode="exact",
        iterations=total_perms,
        seed=None,
        assumption_notes=("Assumes independence and exchangeability under the null."),
    )


def _permutation_montecarlo(
    combined: np.ndarray,
    n_a: int,
    observed_diff: float,
    iterations: int,
    seed: int,
    cite: Citation,
) -> PermutationResult:
    """Monte Carlo permutation test.

    Parameters
    ----------
    combined:
        Pooled observations.
    n_a:
        Size of sample A.
    observed_diff:
        Observed mean difference.
    iterations:
        Number of Monte Carlo permutations.
    seed:
        Random seed.
    cite:
        Citation dict.

    Returns
    -------
    PermutationResult
        Monte Carlo test result.
    """
    rng = np.random.default_rng(seed)
    n_total = len(combined)
    count_extreme = 0
    for _ in range(iterations):
        perm = rng.permutation(n_total)
        perm_a = combined[perm[:n_a]]
        perm_b = combined[perm[n_a:]]
        perm_diff = np.mean(perm_a) - np.mean(perm_b)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    p_value = (count_extreme + 1) / (iterations + 1)
    return PermutationResult(
        method_name="Permutation test (Monte Carlo)",
        citation=_fmt(cite),
        statistic=observed_diff,
        p_value=float(p_value),
        mode="monte_carlo",
        iterations=iterations,
        seed=seed,
        assumption_notes=("Assumes independence and exchangeability under the null."),
    )


def _fmt(cite: Citation) -> str:
    """Format a citation as a readable string."""
    return (
        f"{cite['authors']} ({cite['year']}). " + f"{cite['title']}. {cite['source']}."
    )
