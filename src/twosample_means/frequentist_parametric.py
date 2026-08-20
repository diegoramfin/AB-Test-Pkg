"""Frequentist parametric tests: Student's t, Welch's t, z-test.

Each function is a thin wrapper over scipy.stats (or an analytical
formula for the z-test) that returns the test statistic, p-value,
degrees of freedom (where applicable), confidence interval, citation,
and assumption notes — but NEVER applies an accept/reject decision.

Academic rationale
------------------
- Student's t-test (Student, 1908): assumes equal population variances
  and normality. Appropriate when both assumptions hold.
- Welch's t-test (Welch, 1947): does not assume equal variances. More
  robust when sample sizes or variances differ. Generally recommended
  over Student's t (Ruxton, 2006).
- z-test (Neyman & Pearson, 1933): requires known population variance.
  Rarely applicable in practice but included for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from twosample_means.citations import Citation, get_citation
from twosample_means.config import RunConfig


class MissingVarianceError(ValueError):
    """Raised when the z-test is called without known variance."""


@dataclass(frozen=True)
class ParametricResult:
    """Result of a frequentist parametric test.

    Attributes
    ----------
    method_name:
        Name of the test.
    citation:
        Academic reference string.
    statistic:
        Test statistic (t or z).
    p_value:
        Two-sided p-value.
    degrees_of_freedom:
        Degrees of freedom (None for z-test).
    ci_lower:
        Lower bound of the confidence interval for the mean
        difference.
    ci_upper:
        Upper bound of the confidence interval.
    mean_difference:
        Observed difference in means (A - B).
    assumption_notes:
        Human-readable notes on the assumptions of this test.
    """

    method_name: str
    citation: str
    statistic: float
    p_value: float
    degrees_of_freedom: float | None
    ci_lower: float
    ci_upper: float
    mean_difference: float
    assumption_notes: str


def students_t(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> ParametricResult:
    """Perform Student's t-test (equal variance assumption).

    Tests H₀: μ_A = μ_B assuming both samples come from normal
    distributions with equal variance. Thin wrapper over
    ``scipy.stats.ttest_ind(equal_var=True)``.

    Citation: Student (1908).

    Assumptions: independence, normality, equal variances. If
    variances are unequal, Welch's t-test is preferred.

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration (uses ``ci_level`` for the CI).

    Returns
    -------
    ParametricResult
        Test result with statistic, p-value, dof, CI, and notes.
    """
    result = stats.ttest_ind(a, b, equal_var=True)
    cite = get_citation("students_t")
    mean_diff = float(np.mean(a) - np.mean(b))
    ci_lo, ci_hi = _t_ci(
        mean_diff,
        result.statistic,
        result.df,
        np.std(a, ddof=1),
        np.std(b, ddof=1),
        len(a),
        len(b),
        config.ci_level,
        equal_var=True,
    )
    return ParametricResult(
        method_name="Student's t-test",
        citation=_fmt(cite),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        degrees_of_freedom=float(result.df),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        mean_difference=mean_diff,
        assumption_notes=(
            "Assumes independence, normality, and equal variances."
        ),
    )


def welch_t(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> ParametricResult:
    """Perform Welch's t-test (unequal variance).

    Tests H₀: μ_A = μ_B without assuming equal variances. Thin
    wrapper over ``scipy.stats.ttest_ind(equal_var=False)``.

    Citation: Welch (1947).

    Assumptions: independence, normality. Does NOT assume equal
    variances. Generally recommended over Student's t-test
    (Ruxton, 2006).

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration (uses ``ci_level`` for the CI).

    Returns
    -------
    ParametricResult
        Test result with statistic, p-value, dof, CI, and notes.
    """
    result = stats.ttest_ind(a, b, equal_var=False)
    cite = get_citation("welch_t")
    mean_diff = float(np.mean(a) - np.mean(b))
    ci_lo, ci_hi = _t_ci(
        mean_diff,
        result.statistic,
        result.df,
        np.std(a, ddof=1),
        np.std(b, ddof=1),
        len(a),
        len(b),
        config.ci_level,
        equal_var=False,
    )
    return ParametricResult(
        method_name="Welch's t-test",
        citation=_fmt(cite),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        degrees_of_freedom=float(result.df),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        mean_difference=mean_diff,
        assumption_notes=(
            "Assumes independence and normality. "
            "Does not assume equal variances."
        ),
    )


def z_test(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> ParametricResult:
    """Perform a z-test for known population variance.

    Tests H₀: μ_A = μ_B when the population variances are known.
    Requires ``config.population_variance_a`` and optionally
    ``config.population_variance_b``. If either is missing, raises
    ``MissingVarianceError`` — no silent default.

    Citation: Neyman & Pearson (1933).

    Assumptions: independence, known population variances. The
    z-test is only valid when variances are known a priori, not
    estimated from the data.

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration (must include
        ``population_variance_a`` and optionally
        ``population_variance_b``).

    Returns
    -------
    ParametricResult
        Test result with statistic, p-value, CI, and notes.

    Raises
    ------
    MissingVarianceError
        If population variance(s) are not supplied.
    """
    if config.population_variance_a is None:
        raise MissingVarianceError(
            "z_test requires config.population_variance_a "
            "to be set. The z-test is only valid when the "
            "population variance is known a priori."
        )
    var_a = config.population_variance_a
    var_b = (
        config.population_variance_b
        if config.population_variance_b is not None
        else var_a
    )
    cite = get_citation("z_test")
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    n_a = len(a)
    n_b = len(b)
    se = np.sqrt(var_a / n_a + var_b / n_b)
    z_stat = (mean_a - mean_b) / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    z_crit = stats.norm.ppf(1 - (1 - config.ci_level) / 2)
    ci_lo = (mean_a - mean_b) - z_crit * se
    ci_hi = (mean_a - mean_b) + z_crit * se
    return ParametricResult(
        method_name="z-test",
        citation=_fmt(cite),
        statistic=float(z_stat),
        p_value=float(p_value),
        degrees_of_freedom=None,
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        mean_difference=mean_a - mean_b,
        assumption_notes=(
            "Assumes independence and known population variances."
        ),
    )


def _t_ci(
    mean_diff: float,
    t_stat: float,
    df: float,
    sd_a: float,
    sd_b: float,
    n_a: int,
    n_b: int,
    ci_level: float,
    equal_var: bool,
) -> tuple[float, float]:
    """Compute the CI for the mean difference.

    Parameters
    ----------
    mean_diff:
        Observed mean difference (A - B).
    t_stat:
        The t statistic.
    df:
        Degrees of freedom.
    sd_a, sd_b:
        Sample standard deviations.
    n_a, n_b:
        Sample sizes.
    ci_level:
        Confidence level (e.g., 0.95).
    equal_var:
        Whether equal variance was assumed.

    Returns
    -------
    tuple[float, float]
        Lower and upper CI bounds.
    """
    if equal_var:
        pooled_var = ((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / (
            n_a + n_b - 2
        )
        se = np.sqrt(pooled_var * (1.0 / n_a + 1.0 / n_b))
    else:
        se = np.sqrt(sd_a**2 / n_a + sd_b**2 / n_b)
    t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df)
    return (
        float(mean_diff - t_crit * se),
        float(mean_diff + t_crit * se),
    )


def _fmt(cite: Citation) -> str:
    """Format a citation as a readable string."""
    return (
        f"{cite['authors']} ({cite['year']}). "
        + f"{cite['title']}. {cite['source']}."
    )
