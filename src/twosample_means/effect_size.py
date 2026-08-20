"""Effect-size measures for two samples.

Supported measures are Cohen's d, Hedges' g, Cliff's delta,
rank-biserial, and Hodges-Lehmann.

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
  two-sample estimator of location shift based on pairwise differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pingouin as pg
from scipy import stats

from twosample_means.citations import Citation, get_citation
from twosample_means.config import RunConfig


class ResourceLimitError(ValueError):
    """Raised when an exact pairwise computation exceeds its budget."""


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


def cohens_d(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    config: RunConfig,
) -> EffectSizeResult:
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
    cite = get_citation("cohen_d")
    _require_pooled_variation(a, b)
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


def hedges_g(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    config: RunConfig,
) -> EffectSizeResult:
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
    cite = get_citation("hedges_g")
    _require_pooled_variation(a, b)
    g = float(pg.compute_effsize(a, b, eftype="hedges"))
    ci = pg.compute_esci(
        stat=g,
        nx=len(a),
        ny=len(b),
        eftype="hedges",
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


def cliff_delta(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    config: RunConfig,
) -> EffectSizeResult:
    """Compute Cliff's delta (non-parametric stochastic dominance).

    Citation: Cliff (1993).

    Assumptions: No distributional assumptions. The delta ranges
    from -1 to 1, where 0 means no difference and ±1 means complete
    stochastic dominance. The CI uses a dependence-aware
    influence-function variance with a conservative finite-sample floor.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration.

    Returns
    -------
    EffectSizeResult
        Cliff's delta with a dependence-aware CI.

    """
    cite = get_citation("cliff_delta")
    delta = _cliff_delta(a, b)
    ci_lo, ci_hi = _dominance_ci(a, b, config.ci_level)
    return EffectSizeResult(
        method_name="Cliff's delta",
        citation=_fmt(cite),
        point_estimate=float(delta),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=config.ci_level,
    )


def rank_biserial(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    config: RunConfig,
) -> EffectSizeResult:
    """Compute the rank-biserial correlation.

    Citation: Kerby (2014).

    Assumptions: No distributional assumptions. The rank-biserial
    correlation is the simple difference formula applied to the
    ranks of the two samples. The CI uses a dependence-aware
    influence-function variance with a conservative finite-sample floor.

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
    ci_lo, ci_hi = _dominance_ci(a, b, config.ci_level)
    return EffectSizeResult(
        method_name="Rank-biserial",
        citation=_fmt(cite),
        point_estimate=float(r),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=config.ci_level,
    )


def hodges_lehmann(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    config: RunConfig,
) -> EffectSizeResult:
    """Compute the Hodges-Lehmann estimator of location shift.

    Citation: Hodges & Lehmann (1963).

    Assumptions: No distributional assumptions. The two-sample HL
    estimator is the median of all pairwise differences (a_i - b_j),
    robust to outliers.

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
    n_pairs = len(a) * len(b)
    if n_pairs > config.max_pairwise_comparisons:
        raise ResourceLimitError(
            "Hodges-Lehmann computation exceeds the configured pairwise "
            f"budget: {n_pairs} > {config.max_pairwise_comparisons}"
        )
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


def _require_pooled_variation(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> None:
    """Reject standardized mean effects with no pooled variation."""
    n_a = len(a)
    n_b = len(b)
    pooled_variance = (
        (n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)
    ) / (n_a + n_b - 2)
    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        raise ValueError(
            "standardized mean effect size is not estimable when pooled "
            "variance is zero"
        )


def _cliff_delta(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> float:
    """Compute Cliff's delta vectorised.

    Parameters
    ----------
    a, b:
        Sample arrays.

    Returns
    -------
    float
        Cliff's delta in [-1, 1].

    """
    n_b = len(b)
    sorted_b = np.sort(b)
    less_counts = np.searchsorted(sorted_b, a, side="left")
    less_or_equal_counts = np.searchsorted(sorted_b, a, side="right")
    greater_counts = n_b - less_or_equal_counts
    count_greater = int(np.sum(less_counts, dtype=np.int64))
    count_less = int(np.sum(greater_counts, dtype=np.int64))
    return (count_greater - count_less) / (len(a) * n_b)


def _dominance_ci(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    ci_level: float,
) -> tuple[float, float]:
    """Compute a dependence-aware CI for a dominance effect.

    The pairwise comparisons behind Cliff's delta and rank-biserial
    correlation are not independent Bernoulli observations. This estimate
    uses the two-sample AUC/DeLong influence-function variance instead of
    treating ``n_a * n_b`` comparisons as independent. A conservative
    finite-sample variance floor keeps intervals non-degenerate at complete
    separation.
    """
    n_a = len(a)
    n_b = len(b)
    sorted_a = np.sort(a)
    sorted_b = np.sort(b)

    b_less = np.searchsorted(sorted_b, a, side="left")
    b_equal_end = np.searchsorted(sorted_b, a, side="right")
    row_scores = (b_less + 0.5 * (b_equal_end - b_less)) / n_b

    a_less = np.searchsorted(sorted_a, b, side="left")
    a_equal_end = np.searchsorted(sorted_a, b, side="right")
    col_scores = ((n_a - a_equal_end) + 0.5 * (a_equal_end - a_less)) / n_a

    p_hat = float(np.mean(row_scores))
    variance = float(
        np.var(row_scores, ddof=1) / n_a + np.var(col_scores, ddof=1) / n_b
    )
    effective_n = n_a * n_b / (n_a + n_b)
    smoothed_p = (p_hat * effective_n + 0.5) / (effective_n + 1.0)
    variance_floor = smoothed_p * (1.0 - smoothed_p) / effective_n
    standard_error = 2.0 * np.sqrt(max(variance, variance_floor))
    z = float(stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0))
    delta = 2.0 * p_hat - 1.0
    return (
        max(-1.0, delta - z * standard_error),
        min(1.0, delta + z * standard_error),
    )


def _hodges_lehmann_estimate(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> float:
    """Compute the Hodges-Lehmann location-shift estimate.

    The two-sample HL estimator is the median of all pairwise
    differences (a_i - b_j).

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
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    ci_level: float,
) -> tuple[float, float]:
    """Compute the HL confidence interval via pairwise differences.

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
    half_width = z_crit * se
    rank_lower = int(np.floor(n_a * n_b / 2 - half_width))
    rank_upper = int(np.ceil(n_a * n_b / 2 + half_width))
    rank_lower = max(0, rank_lower)
    rank_upper = min(len(diffs_sorted) - 1, rank_upper)
    return (
        float(diffs_sorted[rank_lower]),
        float(diffs_sorted[rank_upper]),
    )


def _fmt(cite: Citation) -> str:
    """Format a citation as a readable string."""
    return (
        f"{cite['authors']} ({cite['year']}). "
        f"{cite['title']}. {cite['source']}."
    )
