"""Sequential-look planning with alpha spending and calibrated boundaries.

The canonical group-sequential model assumes the look statistics follow the
joint distribution of cumulative standardized increments
``Z_k = sum_{l<=k} sqrt(t_l - t_{l-1}) * X_l`` with ``X_l ~ N(0, 1)``.
Boundaries are calibrated by recursive numerical quadrature of the joint
null distribution (Armitage, McPherson & Rowe, 1969; Jennison and Turnbull,
"Group Sequential Methods with Applications to Clinical Trials", 2000,
Ch. 19), so the family-wise error rate equals the spent alpha rather than
the marginal quantile approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.optimize import brentq

# Numerical grid for recursive boundary quadrature.
_GRID_STEP = 0.005
_GRID_MAX = 12.0

SpendingMethod = Literal["obrien_fleming", "pocock"]


@dataclass(frozen=True)
class SequentialPlan:
    """Predeclared sequential look schedule and spending method."""

    information_fractions: tuple[float, ...]
    alpha: float = 0.05
    method: SpendingMethod = "obrien_fleming"
    two_sided: bool = True

    def __post_init__(self) -> None:
        """Validate information fractions and alpha-spending settings."""
        object.__setattr__(
            self,
            "information_fractions",
            tuple(self.information_fractions),
        )
        if not self.information_fractions:
            raise ValueError("information_fractions must not be empty")
        if any(
            not 0.0 < fraction <= 1.0
            for fraction in self.information_fractions
        ):
            raise ValueError("information fractions must be in (0, 1]")
        if any(
            right <= left
            for left, right in zip(
                self.information_fractions,
                self.information_fractions[1:],
                strict=False,
            )
        ):
            raise ValueError(
                "information fractions must be strictly increasing"
            )
        if self.information_fractions[-1] != 1.0:
            raise ValueError("the final information fraction must equal 1")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.method not in ("obrien_fleming", "pocock"):
            raise ValueError(f"unsupported spending method: {self.method!r}")
        if not isinstance(self.two_sided, bool):
            raise ValueError("two_sided must be a bool")


@dataclass(frozen=True)
class SequentialBoundary:
    """One alpha-spending boundary."""

    look: int
    information_fraction: float
    cumulative_alpha: float
    incremental_alpha: float
    z_boundary: float


@dataclass(frozen=True)
class SequentialResult:
    """Evaluation of observed standardized statistics at planned looks."""

    plan: SequentialPlan
    boundaries: tuple[SequentialBoundary, ...]
    z_statistics: tuple[float, ...]
    crossed_look: int | None
    status: Literal["continue", "stop"]


def marginal_alpha_spending_boundaries(
    plan: SequentialPlan,
) -> tuple[SequentialBoundary, ...]:
    """Calculate alpha-spending quantiles ignoring look-to-look correlation.

    This is the fast marginal approximation: each z boundary is the normal
    quantile of that look's incremental alpha, independent of the other
    looks. Calibrated boundaries are preferred; this function remains
    available for reference and comparison.
    """
    previous = 0.0
    boundaries: list[SequentialBoundary] = []
    for look, fraction in enumerate(plan.information_fractions, start=1):
        cumulative = _cumulative_alpha(plan, fraction)
        incremental = cumulative - previous
        tail = incremental / (2.0 if plan.two_sided else 1.0)
        z_boundary = float(stats.norm.ppf(1.0 - tail))
        boundaries.append(
            SequentialBoundary(
                look=look,
                information_fraction=fraction,
                cumulative_alpha=cumulative,
                incremental_alpha=incremental,
                z_boundary=z_boundary,
            )
        )
        previous = cumulative
    return tuple(boundaries)


def group_sequential_boundaries(
    plan: SequentialPlan,
) -> tuple[SequentialBoundary, ...]:
    """Calibrate correlation-aware boundaries for a predeclared look plan.

    Each look boundary is solved from the recursive joint null distribution
    of the canonical group-sequential statistics, so the total family-wise
    error across all planned looks equals ``plan.alpha``. The cumulative
    alpha at each look follows the selected spending function.

    The recursion tracks the density of the information-scale partial sum
    ``T_k`` (variance ``t_k``) restricted to the continuation region of
    earlier looks. Each look increments the statistic by ``W_l`` with
    variance ``t_l - t_{l-1}``. At each look the boundary ``b`` is found by
    solving for the spent alpha increment on a uniform grid.
    """
    grid = np.arange(
        -_GRID_MAX,
        _GRID_MAX + _GRID_STEP,
        _GRID_STEP,
        dtype=float,
    )
    step = float(grid[1] - grid[0])

    # Entering look 1, the statistic T_0 is a point mass at zero with
    # density 1/step on the uniform grid.
    density = np.zeros_like(grid)
    density[np.argmin(np.abs(grid))] = 1.0 / step
    previous = 0.0
    previous_fraction = 0.0
    boundaries: list[SequentialBoundary] = []
    for look, fraction in enumerate(plan.information_fractions, start=1):
        cumulative = _cumulative_alpha(plan, fraction)
        incremental = cumulative - previous
        increment_sigma = np.sqrt(fraction - previous_fraction)
        boundary_s = _solve_boundary(
            density,
            incremental,
            increment_sigma,
            plan.two_sided,
        )
        z_boundary = boundary_s / np.sqrt(fraction)
        boundaries.append(
            SequentialBoundary(
                look=look,
                information_fraction=fraction,
                cumulative_alpha=cumulative,
                incremental_alpha=incremental,
                z_boundary=float(z_boundary),
            )
        )
        if look < len(plan.information_fractions):
            kernel = _normal_kernel(grid, increment_sigma)
            density = _restrict_density(
                _convolve(density, kernel, step),
                boundary_s,
                plan.two_sided,
            )
        previous = cumulative
        previous_fraction = fraction
    return tuple(boundaries)


def alpha_spending_boundaries(
    plan: SequentialPlan,
) -> tuple[SequentialBoundary, ...]:
    """Return correlation-aware alpha-spending boundaries.

    ``group_sequential_boundaries`` is the calibrated implementation;
    this alias keeps the established public name while correcting the
    marginal-quantile approximation it previously used.
    """
    return group_sequential_boundaries(plan)


def evaluate_sequential(
    plan: SequentialPlan,
    z_statistics: tuple[float, ...] | list[float],
) -> SequentialResult:
    """Return the first planned look whose statistic crosses its boundary."""
    statistics = tuple(float(value) for value in z_statistics)
    if len(statistics) != len(plan.information_fractions):
        raise ValueError("one z statistic is required for every planned look")
    if not np.isfinite(statistics).all():
        raise ValueError("z_statistics must contain only finite values")
    boundaries = alpha_spending_boundaries(plan)
    crossed_look: int | None = None
    for boundary, statistic in zip(boundaries, statistics, strict=True):
        if abs(statistic) >= boundary.z_boundary:
            crossed_look = boundary.look
            break
    return SequentialResult(
        plan=plan,
        boundaries=boundaries,
        z_statistics=statistics,
        crossed_look=crossed_look,
        status="stop" if crossed_look is not None else "continue",
    )


@dataclass(frozen=True)
class ConfidenceSequenceInterval:
    """One always-valid interval in a confidence sequence."""

    n: int
    point_estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ConfidenceSequence:
    """A time-uniform (always-valid) confidence sequence.

    Every interval is valid simultaneously: the probability that any
    interval ever misses the true mean is at most ``alpha``, so repeated
    peeking at intermediate times does not inflate the error rate. The
    widths follow the normal-mixture confidence sequence of Howard et al.
    (2021) and are wider than the fixed-sample interval, especially early.
    """

    alpha: float
    variance_proxies: tuple[float, ...]
    intervals: tuple[ConfidenceSequenceInterval, ...]


@dataclass(frozen=True)
class SequentialPowerResult:
    """Monte Carlo power and average sample information for a look plan."""

    plan: SequentialPlan
    drift: float
    power: float
    average_sample_information: float
    stopping_probabilities: tuple[float, ...]
    replications: int
    seed: int


def always_valid_confidence_sequence(
    values: npt.NDArray[np.float64],
    *,
    alpha: float = 0.05,
    variance_proxy: float | None = None,
    lambda_prior: float | None = None,
) -> ConfidenceSequence:
    """Build a time-uniform confidence sequence for a running mean.

    The sequence is valid for every sample size simultaneously: with
    probability at least ``1 - alpha`` every interval contains the true
    mean, regardless of when (or how often) the data are examined. This is
    the normal-mixture bound of Howard, Ramdas, McAuliffe & Sekhon (2021,
    "Time-uniform Chernoff bounds via nonnegative supermartingales").

    ``variance_proxy`` is the sub-Gaussian variance proxy; when omitted it
    is estimated from the full sample variance, which is acceptable in
    practice for iid data but strictly requires a pre-specified proxy (for
    bounded outcomes, ``0.25`` for a proportion is a safe choice).
    ``lambda_prior`` is the mixture scale and defaults to the variance
    proxy, matching the standard recommendation.

    Parameters
    ----------
    values:
        Observations in the order they arrive; each prefix is one look.
    alpha:
        Maximum probability that any interval misses the true mean.
    variance_proxy:
        Known sub-Gaussian variance proxy (estimated from the data if
        omitted).
    lambda_prior:
        Mixture prior scale (defaults to the variance proxy).
    """
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or len(observations) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not np.isfinite(observations).all():
        raise ValueError("values must contain only finite numbers")
    proxy = (
        variance_proxy
        if variance_proxy is not None
        else float(np.var(observations, ddof=1))
    )
    if not np.isfinite(proxy) or proxy <= 0.0:
        raise ValueError("variance_proxy must be positive and finite")
    if lambda_prior is None:
        lambda_prior = proxy
    if not np.isfinite(lambda_prior) or lambda_prior <= 0.0:
        raise ValueError("lambda_prior must be positive and finite")

    cumulative_sums = np.cumsum(observations)
    intervals: list[ConfidenceSequenceInterval] = []
    for n in range(1, len(observations) + 1):
        information = n * proxy
        width = float(
            np.sqrt(
                2.0
                * (information + lambda_prior)
                * np.log(
                    (1.0 / alpha)
                    * np.sqrt((information + lambda_prior) / lambda_prior)
                )
            )
            / n
        )
        estimate = float(cumulative_sums[n - 1] / n)
        intervals.append(
            ConfidenceSequenceInterval(
                n=n,
                point_estimate=estimate,
                lower=estimate - width,
                upper=estimate + width,
            )
        )
    return ConfidenceSequence(
        alpha=alpha,
        variance_proxies=(proxy,),
        intervals=tuple(intervals),
    )


def difference_confidence_sequence(
    control_values: npt.NDArray[np.float64],
    treatment_values: npt.NDArray[np.float64],
    *,
    alpha: float = 0.05,
    variance_proxy: float | None = None,
) -> ConfidenceSequence:
    """Build an always-valid confidence sequence for the mean difference.

    Each arm gets its own time-uniform sequence at error level ``alpha/2``;
    the difference interval combines the two arm widths. The union-bound
    combination keeps the whole sequence valid at level ``alpha`` for all
    look times. Both streams must be observed in parallel (same length),
    as in an accumulating A/B experiment.

    ``variance_proxy`` applies to both arms; when omitted each arm's
    variance is estimated from its own full sample.
    """
    control = np.asarray(control_values, dtype=float)
    treatment = np.asarray(treatment_values, dtype=float)
    if control.ndim != 1 or treatment.ndim != 1:
        raise ValueError(
            "control and treatment values must be one-dimensional"
        )
    if len(control) == 0 or len(control) != len(treatment):
        raise ValueError(
            "control and treatment streams must be non-empty and equal length"
        )
    if not np.isfinite(control).all() or not np.isfinite(treatment).all():
        raise ValueError("values must contain only finite numbers")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    if variance_proxy is not None:
        control_proxy = variance_proxy
        treatment_proxy = variance_proxy
    else:
        control_proxy = float(np.var(control, ddof=1))
        treatment_proxy = float(np.var(treatment, ddof=1))
    if not np.isfinite(control_proxy) or control_proxy <= 0.0:
        raise ValueError("control variance_proxy must be positive and finite")
    if not np.isfinite(treatment_proxy) or treatment_proxy <= 0.0:
        raise ValueError(
            "treatment variance_proxy must be positive and finite"
        )

    control_sequence = always_valid_confidence_sequence(
        control,
        alpha=alpha / 2.0,
        variance_proxy=control_proxy,
    )
    treatment_sequence = always_valid_confidence_sequence(
        treatment,
        alpha=alpha / 2.0,
        variance_proxy=treatment_proxy,
    )
    intervals = tuple(
        ConfidenceSequenceInterval(
            n=control_interval.n,
            point_estimate=(
                treatment_interval.point_estimate
                - control_interval.point_estimate
            ),
            lower=(treatment_interval.lower - control_interval.upper),
            upper=(treatment_interval.upper - control_interval.lower),
        )
        for control_interval, treatment_interval in zip(
            control_sequence.intervals,
            treatment_sequence.intervals,
            strict=True,
        )
    )
    return ConfidenceSequence(
        alpha=alpha,
        variance_proxies=(control_proxy, treatment_proxy),
        intervals=intervals,
    )


def sequential_power(
    plan: SequentialPlan,
    drift: float,
    *,
    replications: int = 200_000,
    seed: int = 42,
) -> SequentialPowerResult:
    """Estimate power and average sample information by Monte Carlo.

    Under the canonical group-sequential model the standardized statistic
    at look ``k`` is ``Z_k ~ N(drift * sqrt(t_k), 1)``, where ``drift`` is
    the standardized effect ``theta = delta * sqrt(I)`` (the true mean
    difference times the square root of the full-sample information).
    Increments carry variance ``t_k - t_{k-1}``, matching the calibrated
    boundary recursion, so the reported power is consistent with the
    family-wise error of the boundary schedule.

    ``average_sample_information`` is the expected information fraction at
    the stopping look; every run stops by the final look, so it is the
    information-weighted sum of the first-crossing probabilities.
    """
    if not np.isfinite(drift):
        raise ValueError("drift must be finite")
    if isinstance(replications, bool) or not isinstance(replications, int):
        raise ValueError("replications must be an integer")
    if replications < 2:
        raise ValueError("replications must be at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    fractions = np.asarray(plan.information_fractions, dtype=float)
    boundaries = np.asarray(
        [boundary.z_boundary for boundary in alpha_spending_boundaries(plan)],
        dtype=float,
    )
    increments_standard_deviation = np.sqrt(
        np.diff(np.concatenate(([0.0], fractions)))
    )
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=(replications, len(fractions)))
    increments = draws * increments_standard_deviation + drift * np.diff(
        np.concatenate(([0.0], fractions))
    )
    partial_sums = np.cumsum(increments, axis=1)
    z_statistics = partial_sums / np.sqrt(fractions)
    if plan.two_sided:
        crossed = np.abs(z_statistics) >= boundaries
    else:
        crossed = z_statistics >= boundaries

    first_crossing = np.zeros((replications, len(fractions)), dtype=bool)
    remaining = np.ones(replications, dtype=bool)
    for look in range(len(fractions)):
        first_crossing[:, look] = remaining & crossed[:, look]
        remaining &= ~crossed[:, look]
    # Power is the rejection probability: crossing any boundary. Every run
    # that never crosses still stops at the final look with a decision, so
    # the final-look mass is added for the average sample information but
    # not to the rejection probability.
    crossing_probabilities = first_crossing.mean(axis=0)
    stopping_probabilities = crossing_probabilities.copy()
    stopping_probabilities[-1] += remaining.mean()
    power = float(crossing_probabilities.sum())
    average_sample_information = float(
        np.dot(fractions, stopping_probabilities)
    )
    return SequentialPowerResult(
        plan=plan,
        drift=drift,
        power=power,
        average_sample_information=average_sample_information,
        stopping_probabilities=tuple(
            float(value) for value in stopping_probabilities
        ),
        replications=replications,
        seed=seed,
    )


def _normal_kernel(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Return a normal kernel with standard deviation ``sigma``."""
    kernel = stats.norm.pdf(grid / sigma) / sigma
    return np.asarray(kernel, dtype=float)


def _convolve(
    density: np.ndarray,
    kernel: np.ndarray,
    step: float,
) -> np.ndarray:
    """Convolve the continuation density with one independent increment.

    ``density`` and ``kernel`` live on the same uniform grid; ``'same'``
    keeps the result on that grid and the factor ``step`` makes the discrete
    sum approximate the continuous convolution integral.
    """
    return np.convolve(density, kernel, mode="same") * step


def _restrict_density(
    density: np.ndarray,
    boundary_s: float,
    two_sided: bool,
) -> np.ndarray:
    """Zero the density beyond the continuation region on the S_k scale."""
    restricted = density.copy()
    positions = np.arange(len(restricted)) * _GRID_STEP - _GRID_MAX
    if two_sided:
        restricted[np.abs(positions) > boundary_s] = 0.0
    else:
        restricted[positions > boundary_s] = 0.0
    return restricted


def _solve_boundary(
    density: np.ndarray,
    incremental_alpha: float,
    increment_sigma: float,
    two_sided: bool,
) -> float:
    """Solve the boundary for one look on the T_k scale."""
    target = incremental_alpha

    def crossing(boundary_s: float) -> float:
        return _crossing_probability(
            density,
            boundary_s,
            increment_sigma,
            two_sided,
        )

    lower = 0.05
    upper = 12.0
    if crossing(lower) < target:
        return float(lower)
    if crossing(upper) > target:
        return float(upper)
    return float(brentq(lambda b: crossing(b) - target, lower, upper))


def _crossing_probability(
    density: np.ndarray,
    boundary_s: float,
    increment_sigma: float,
    two_sided: bool,
) -> float:
    """Integrate the crossing probability for a candidate boundary.

    For each partial-sum value ``u`` of ``T_{k-1}``, the next increment
    ``W_k`` (standard deviation ``increment_sigma``) must push the statistic
    past ``+/-boundary_s`` on the ``T_k`` scale. The contribution is
    integrated over the continuation density of ``T_{k-1}``.
    """
    positions = np.arange(len(density), dtype=float) * _GRID_STEP - _GRID_MAX
    upper_tail = 1.0 - stats.norm.cdf(
        (boundary_s - positions) / increment_sigma
    )
    integral = float(np.trapezoid(density * upper_tail, dx=_GRID_STEP))
    if two_sided:
        lower_tail = stats.norm.cdf(
            (-boundary_s - positions) / increment_sigma
        )
        integral += float(np.trapezoid(density * lower_tail, dx=_GRID_STEP))
    return integral


def _cumulative_alpha(
    plan: SequentialPlan,
    information_fraction: float,
) -> float:
    """Return cumulative alpha spent at one information fraction."""
    if plan.method == "obrien_fleming":
        base_alpha = plan.alpha / 2.0 if plan.two_sided else plan.alpha
        z_alpha = stats.norm.ppf(1.0 - base_alpha)
        multiplier = 2.0 if plan.two_sided else 1.0
        return float(
            multiplier
            * (1.0 - stats.norm.cdf(z_alpha / np.sqrt(information_fraction)))
        )
    return float(
        plan.alpha
        * np.log(1.0 + (np.e - 1.0) * information_fraction)
        / np.log(np.e)
    )
