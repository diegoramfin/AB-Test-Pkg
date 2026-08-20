"""Fixed-horizon sequential-look planning with alpha spending."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

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


def alpha_spending_boundaries(
    plan: SequentialPlan,
) -> tuple[SequentialBoundary, ...]:
    """Calculate cumulative and incremental alpha-spending boundaries.

    The reported z boundary is the marginal normal quantile associated with
    cumulative spending. Correlation-aware group-sequential calibration can be
    added later; callers should preserve the predeclared look schedule.
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
