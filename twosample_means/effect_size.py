"""Effect sizes: Cohen's d, Hedges' g, Cliff's delta,
rank-biserial, Hodges-Lehmann.

Each function is a thin wrapper over pingouin or scipy.stats, returning
the point estimate, confidence interval, and citation — but NEVER an
accept/reject decision.

Academic rationale
------------------
- Cohen's d (Cohen, 1988): standardized mean difference using the
  pooled standard deviation.
- Hedges' g (Hedges, 1981): Cohen's d with a small-sample bias
  correction factor J.
- Cliff's delta (Cliff, 1993): non-parametric measure of
  stochastic dominance, in [-1, 1].
- Rank-biserial correlation (Kerby, 2014): the simple difference
  formula for the rank-biserial correlation.
- Hodges-Lehmann estimator (Hodges & Lehmann, 1963): a robust
  estimator of location shift based on Walsh averages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from twosample_means.citations import Citation, get_citation
from twosample_means.config import RunConfig


@dataclass(frozen=True)
class EffectSizeResult:
    """Result of an effect-size computation.

    Attributes
    ----------
    method_name:
        Name of the effect size.
    citation:
        Academic reference string.
    point_estimate:
        The effect-size estimate.
    ci_lower:
        Lower bound of the confidence interval.
    ci_upper:
        Upper bound of the confidence interval.
    ci_level:
        Confidence level used.
    """

    method_name: str
    citation: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float


def cohens_d(a: np.ndarray, b: np.ndarray, config: RunConfig) -> EffectSizeResult:
    """Compute Cohen's d (pooled-SD standardized mean difference).

    Citation: Cohen (1988).

    Assumptions: Assumes approximately equal variances for the pooled
    SD to be meaningful. For unequal variances, consider Hedges' g
    with Welch's SE or a non-parametric effect size.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration (uses ``ci_level``).

    Returns
    -------
    EffectSizeResult
        Cohen's d with CI.
    """
    import pingouin as pg

    cite = get_citation("cohen_d")
    d = float(pg.compute_effsize(a, b, eftype="cohen"))
    ci = pg.compute_esci(
        stat=d,
        nx=len(a),
        ny=len(b),
        eftype="cohen",
        confidence=config.ci_level,
        decimals=10,
    )
    return EffectSizeResult(
        method_name="Cohen's d",
        citation=_fmt(cite),
        point_estimate=d,
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        ci_level=config.ci_level,
    )


def hedges_g(a: np.ndarray, b: np.ndarray, config: RunConfig) -> EffectSizeResult:
    """Compute Hedges' g (bias-corrected Cohen's d).

    Applies the small-sample correction factor
    J = 1 - 3/(4*df - 1) to Cohen's d.

    Citation: Hedges (1981).

    Assumptions: Same as Cohen's d, with a small-sample correction
    that reduces upward bias in the estimate.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration.

    Returns
    -------
    EffectSizeResult
        Hedges' g with CI.
    """
    import pingouin as pg

    cite = get_citation("hedges_g")
    g = float(pg.compute_effsize(a, b, eftype="hedges"))
    ci = pg.compute_esci(
        stat=g,
        nx=len(a),
        ny=len(b),
        eftype="cohen",
        confidence=config.ci_level,
        decimals=10,
    )
    return EffectSizeResult(
        method_name="Hedges' g",
        citation=_fmt(cite),
        point_estimate=g,
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        ci_level=config.ci_level,
    )


def cliff_delta(a: np.ndarray, b: np.ndarray, config: RunConfig) -> EffectSizeResult:
    """Compute Cliff's delta (non-parametric stochastic dominance).

    Citation: Cliff (1993).

    Assumptions: No distributional assumptions. The delta ranges
    from -1 to 1, where 0 means no difference and ±1 means complete
    stochastic dominance.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration.

    Returns
    -------
    EffectSizeResult
        Cliff's delta with asymptotic CI.
    """
    cite = get_citation("cliff_delta")
    n_a = len(a)
    n_b = len(b)
    delta = _cliff_delta(a, b)
    se = _cliff_delta_se(delta, n_a, n_b)
    z_crit = stats.norm.ppf(1 - (1 - config.ci_level) / 2)
    ci_lo = float(delta - z_crit * se)
    ci_hi = float(delta + z_crit * se)
    return EffectSizeResult(
        method_name="Cliff's delta",
        citation=_fmt(cite),
        point_estimate=float(delta),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=config.ci_level,
    )


def rank_biserial(a: np.ndarray, b: np.ndarray, config: RunConfig) -> EffectSizeResult:
    """Compute the rank-biserial correlation.

    Citation: Kerby (2014).

    Assumptions: No distributional assumptions. The rank-biserial
    correlation is the simple difference formula applied to the
    ranks of the two samples. The CI uses the asymptotic variance
    from Cliff (1993), which accounts for the observed effect size
    (same formula used for Cliff's delta), rather than the null
    variance which would produce an misleadingly narrow CI.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration.

    Returns
    -------
    EffectSizeResult
        Rank-biserial correlation with approximate CI.
    """
    cite = get_citation("rank_biserial")
    n_a = len(a)
    n_b = len(b)
    u_stat, _p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = (2.0 * u_stat) / (n_a * n_b) - 1.0
    se = np.sqrt((n_a + n_b - 1) * (1.0 - r**2) / (n_a * n_b))
    z_crit = stats.norm.ppf(1 - (1 - config.ci_level) / 2)
    ci_lo = float(r - z_crit * se)
    ci_hi = float(r + z_crit * se)
    return EffectSizeResult(
        method_name="Rank-biserial",
        citation=_fmt(cite),
        point_estimate=float(r),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=config.ci_level,
    )


def hodges_lehmann(a: np.ndarray, b: np.ndarray, config: RunConfig) -> EffectSizeResult:
    """Compute the Hodges-Lehmann estimator of location shift.

    Citation: Hodges & Lehmann (1963).

    Assumptions: No distributional assumptions. The HL estimator is
    the median of all pairwise differences (Walsh averages), robust
    to outliers.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration.

    Returns
    -------
    EffectSizeResult
        HL estimate with CI.
    """
    cite = get_citation("hodges_lehmann")
    estimate = _hodges_lehmann_estimate(a, b)
    ci_lo, ci_hi = _hodges_lehmann_ci(a, b, config.ci_level)
    return EffectSizeResult(
        method_name="Hodges-Lehmann",
        citation=_fmt(cite),
        point_estimate=estimate,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=config.ci_level,
    )


def _cliff_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cliff's delta manually.

    Parameters
    ----------
    a, b:
        Sample arrays.

    Returns
    -------
    float
        Cliff's delta in [-1, 1].
    """
    n_a = len(a)
    n_b = len(b)
    count_greater = 0
    count_less = 0
    for x in a:
        for y in b:
            if x > y:
                count_greater += 1
            elif x < y:
                count_less += 1
    return (count_greater - count_less) / (n_a * n_b)


def _hodges_lehmann_estimate(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Hodges-Lehmann location-shift estimate.

    The HL estimator is the median of all pairwise differences
    (a_i - b_j), also known as the Walsh averages.

    Parameters
    ----------
    a, b:
        Sample arrays.

    Returns
    -------
    float
        The HL estimate.
    """
    diffs = np.subtract.outer(a, b).flatten()
    return float(np.median(diffs))


def _hodges_lehmann_ci(
    a: np.ndarray, b: np.ndarray, ci_level: float
) -> tuple[float, float]:
    """Compute the HL confidence interval via Walsh averages.

    Uses the asymptotic approximation based on the Wilcoxon
    rank-sum distribution.

    Parameters
    ----------
    a, b:
        Sample arrays.
    ci_level:
        Confidence level (e.g., 0.95).

    Returns
    -------
    tuple[float, float]
        Lower and upper CI bounds.
    """
    n_a = len(a)
    n_b = len(b)
    diffs = np.subtract.outer(a, b).flatten()
    diffs_sorted = np.sort(diffs)
    alpha = 1.0 - ci_level
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(n_a * n_b * (n_a + n_b + 1) / 12.0)
    rank_lower = int(np.floor(n_a * n_b / 2 - z_crit * se))
    rank_upper = int(np.ceil(n_a * n_b / 2 + z_crit * se))
    rank_lower = max(0, rank_lower)
    rank_upper = min(len(diffs_sorted) - 1, rank_upper)
    return (
        float(diffs_sorted[rank_lower]),
        float(diffs_sorted[rank_upper]),
    )


def _cliff_delta_se(delta: float, n_a: int, n_b: int) -> float:
    """Compute the asymptotic SE for Cliff's delta.

    Uses the formula from Cliff (1993, eq. 2.24).

    Parameters
    ----------
    delta:
        Cliff's delta estimate.
    n_a, n_b:
        Sample sizes.

    Returns
    -------
    float
        Standard error.
    """
    var_d = (n_a + n_b - 1) * (1 - delta**2) / (n_a * n_b)
    return float(np.sqrt(var_d))


def _fmt(cite: Citation) -> str:
    """Format a citation as a readable string."""
    return (
        f"{cite['authors']} ({cite['year']}). " + f"{cite['title']}. {cite['source']}."
    )
