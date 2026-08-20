"""Assumption diagnostics: normality, variance homogeneity, outliers.

This module provides thin wrappers over scipy.stats for checking the
assumptions underlying the parametric tests in the battery. Each
function returns a diagnostic result with the test statistic, p-value,
academic citation, and an assumption outcome ("met" or "not_met") based
on the configured alpha — but NEVER applies an accept/reject decision
about the main hypothesis.

Academic rationale
------------------
Checking assumptions before selecting a test is a foundational
principle of statistical analysis (see Wasserman, 2004, "All of
Statistics"; Kruschke, 2013). The normality and variance homogeneity
checks inform the analyst which parametric test is appropriate, but
the procedure reports all tests regardless — the analyst interprets
which to weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from twosample_means.citations import Citation, get_citation
from twosample_means.config import RunConfig


@dataclass(frozen=True)
class DiagnosticResult:
    """Result of a single assumption diagnostic.

    Attributes
    ----------
    method_name:
        Name of the diagnostic method.
    citation:
        Academic reference string.
    statistic:
        Test statistic value.
    p_value:
        P-value of the test (may be ``None`` if not applicable).
    assumption_outcome:
        "met" or "not_met" based on whether p_value > alpha.
        This is descriptive, NOT a decision about the main hypothesis.
    details:
        Additional human-readable details.
    """

    method_name: str
    citation: str
    statistic: float
    p_value: float | None
    assumption_outcome: Literal["met", "not_met"]
    details: str = ""


@dataclass(frozen=True)
class OutlierResult:
    """Result of outlier flagging.

    Attributes
    ----------
    method_name:
        Name of the outlier method used.
    citation:
        Academic reference (Tukey 1977 for IQR).
    count:
        Number of flagged outliers.
    indices:
        Indices of flagged outlier points.
    threshold_used:
        The threshold value applied.
    """

    method_name: str
    citation: str
    count: int
    indices: tuple[int, ...]
    threshold_used: float


def shapiro_wilk(x: np.ndarray, config: RunConfig) -> DiagnosticResult:
    """Perform the Shapiro-Wilk test for normality.

    Tests the null hypothesis that the sample was drawn from a normal
    distribution. Thin wrapper over ``scipy.stats.shapiro``.

    Citation: Shapiro & Wilk (1965).

    Assumptions: The test is sensitive to sample size — with large n,
    even trivial deviations from normality will be flagged. The analyst
    should consider effect size alongside the p-value.

    Parameters
    ----------
    x:
        1-D array of observations.
    config:
        Run configuration (uses ``alpha`` for the outcome threshold).

    Returns
    -------
    DiagnosticResult
        The test result with statistic (W), p-value, and outcome.
    """
    result = stats.shapiro(x)
    cite = get_citation("shapiro")
    outcome = _assumption_outcome(result.pvalue, config.alpha)
    return DiagnosticResult(
        method_name="Shapiro-Wilk",
        citation=_format_cite(cite),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        assumption_outcome=outcome,
    )


def anderson_darling(x: np.ndarray, config: RunConfig) -> DiagnosticResult:
    """Perform the Anderson-Darling test for normality.

    Tests the null hypothesis that the sample was drawn from a normal
    distribution. Thin wrapper over ``scipy.stats.anderson``.

    Citation: Anderson & Darling (1952).

    Assumptions: The Anderson-Darling test gives more weight to the
    tails than the Kolmogorov-Smirnov test, making it more sensitive
    to tail deviations.

    Parameters
    ----------
    x:
        1-D array of observations.
    config:
        Run configuration (uses ``alpha`` to find the closest
        significance level for comparison).

    Returns
    -------
    DiagnosticResult
        The test result with statistic (A²), p-value approximated
        from the critical values, and outcome.
    """
    result = stats.anderson(
        x,
        dist="norm",
        method=stats.MonteCarloMethod(
            n_resamples=9999, rng=np.random.default_rng(config.seed)
        ),
    )
    cite = get_citation("anderson_darling")
    p_value = float(result.pvalue)
    outcome = _assumption_outcome(p_value, config.alpha)
    return DiagnosticResult(
        method_name="Anderson-Darling",
        citation=_format_cite(cite),
        statistic=float(result.statistic),
        p_value=p_value,
        assumption_outcome=outcome,
    )


def dagostino_k2(x: np.ndarray, config: RunConfig) -> DiagnosticResult:
    """Perform D'Agostino's K² test for normality.

    Combines skew and kurtosis tests into an omnibus test of normality.
    Thin wrapper over ``scipy.stats.normaltest``.

    Citation: D'Agostino & Pearson (1973).

    Assumptions: Requires at least 8 observations. The test is based
    on transformations of sample skewness and kurtosis.

    Parameters
    ----------
    x:
        1-D array of observations.
    config:
        Run configuration (uses ``alpha`` for the outcome threshold).

    Returns
    -------
    DiagnosticResult
        The test result with statistic (K²), p-value, and outcome.
    """
    result = stats.normaltest(x)
    cite = get_citation("dagostino_k2")
    outcome = _assumption_outcome(result.pvalue, config.alpha)
    return DiagnosticResult(
        method_name="D'Agostino K²",
        citation=_format_cite(cite),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        assumption_outcome=outcome,
    )


def levene(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> DiagnosticResult:
    """Perform Levene's test for equality of variances.

    Tests the null hypothesis that all samples come from populations
    with equal variances. Thin wrapper over ``scipy.stats.levene``.

    Citation: Levene (1960).

    Assumptions: Levene's classic test uses the mean as center,
    which is more powerful but less robust to non-normality than
    the Brown-Forsythe variant (which uses the median).

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration (uses ``alpha`` for the outcome threshold).

    Returns
    -------
    DiagnosticResult
        The test result with statistic (W), p-value, and outcome.
    """
    result = stats.levene(a, b, center="mean")
    cite = get_citation("levene")
    outcome = _assumption_outcome(result.pvalue, config.alpha)
    return DiagnosticResult(
        method_name="Levene",
        citation=_format_cite(cite),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        assumption_outcome=outcome,
    )


def bartlett(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> DiagnosticResult:
    """Perform Bartlett's test for equality of variances.

    Tests the null hypothesis that all samples come from populations
    with equal variances. Thin wrapper over ``scipy.stats.bartlett``.

    Citation: Bartlett (1937).

    Assumptions: Bartlett's test is sensitive to non-normality. If
    normality is in doubt, Levene's or Brown-Forsythe is preferred.

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration (uses ``alpha`` for the outcome threshold).

    Returns
    -------
    DiagnosticResult
        The test result with statistic, p-value, and outcome.
    """
    result = stats.bartlett(a, b)
    cite = get_citation("bartlett")
    outcome = _assumption_outcome(result.pvalue, config.alpha)
    return DiagnosticResult(
        method_name="Bartlett",
        citation=_format_cite(cite),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        assumption_outcome=outcome,
    )


def brown_forsythe(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> DiagnosticResult:
    """Perform the Brown-Forsythe test for equality of variances.

    A variant of Levene's test using the median, which is more robust
    to non-normality. Thin wrapper over ``scipy.stats.levene`` with
    ``center='median'`` (same as Levene here, but the Brown-Forsythe
    label distinguishes the historical reference).

    Citation: Brown & Forsythe (1974).

    Assumptions: More robust than Bartlett's test when distributions
    are non-normal or have unequal skewness.

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration (uses ``alpha`` for the outcome threshold).

    Returns
    -------
    DiagnosticResult
        The test result with statistic (F), p-value, and outcome.
    """
    result = stats.levene(a, b, center="median")
    cite = get_citation("brown_forsythe")
    outcome = _assumption_outcome(result.pvalue, config.alpha)
    return DiagnosticResult(
        method_name="Brown-Forsythe",
        citation=_format_cite(cite),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        assumption_outcome=outcome,
    )


def flag_outliers(x: np.ndarray, config: RunConfig) -> OutlierResult:
    """Flag potential outliers without removing them.

    Supports two methods:
    - ``"iqr"``: Tukey's fence (values beyond Q1 - 1.5*IQR or
      Q3 + 1.5*IQR). Citation: Tukey (1977).
    - ``"zscore"``: Values with absolute z-score exceeding the
      threshold (default 3.0).

    The data is NOT modified — this function only reports which
    indices are flagged.

    Parameters
    ----------
    x:
        1-D array of observations.
    config:
        Run configuration (uses ``outlier_method`` and
        ``outlier_threshold``).

    Returns
    -------
    OutlierResult
        The count and indices of flagged outliers.
    """
    if config.outlier_method == "iqr":
        indices = _iqr_outliers(x, config.outlier_threshold)
        citation = (
            "Tukey, J. W. (1977). Exploratory "
            + "Data Analysis. Addison-Wesley."
        )
        method_name = "IQR"
    else:
        indices = _zscore_outliers(x, config.outlier_threshold)
        citation = (
            "Standard z-score rule (threshold = "
            + f"{config.outlier_threshold})"
        )
        method_name = "Z-score"
    return OutlierResult(
        method_name=method_name,
        citation=citation,
        count=len(indices),
        indices=indices,
        threshold_used=config.outlier_threshold,
    )


def _iqr_outliers(x: np.ndarray, multiplier: float) -> tuple[int, ...]:
    """Find outlier indices using the IQR method.

    Parameters
    ----------
    x:
        1-D array of observations.
    multiplier:
        IQR multiplier (typically 1.5).

    Returns
    -------
    tuple[int, ...]
        Indices of flagged outliers.
    """
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (x < lower) | (x > upper)
    return tuple(np.where(mask)[0].tolist())


def _zscore_outliers(x: np.ndarray, threshold: float) -> tuple[int, ...]:
    """Find outlier indices using the z-score method.

    Parameters
    ----------
    x:
        1-D array of observations.
    threshold:
        Absolute z-score threshold (typically 3.0).

    Returns
    -------
    tuple[int, ...]
        Indices of flagged outliers.
    """
    z_scores = np.abs((x - np.mean(x)) / np.std(x))
    mask = z_scores > threshold
    return tuple(np.where(mask)[0].tolist())


def _assumption_outcome(
    p_value: float, alpha: float
) -> Literal["met", "not_met"]:
    """Determine if an assumption is met based on p-value.

    If p_value > alpha, we fail to reject the null (assumption met).
    If p_value <= alpha, we reject the null (assumption not met).
    This is descriptive only — not a decision about the main hypothesis.

    Parameters
    ----------
    p_value:
        The p-value from the assumption test.
    alpha:
        The significance level threshold.

    Returns
    -------
    str
        "met" or "not_met".
    """
    if p_value > alpha:
        return "met"
    return "not_met"


def _format_cite(cite: Citation) -> str:
    """Format a citation dict as a readable string.

    Parameters
    ----------
    cite:
        Citation dictionary with authors, year, title, source.

    Returns
    -------
    str
        Formatted citation.
    """
    return (
        f"{cite['authors']} ({cite['year']}). "
        + f"{cite['title']}. {cite['source']}."
    )
