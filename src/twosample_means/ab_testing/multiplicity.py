"""Multiple-comparison corrections for experiment metric families."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import isfinite

import scipy.stats as stats

from .binary import BinaryMetricResult, _newcombe_difference_ci
from .config import Multiplicity, MultiplicityScope
from .continuous import ContinuousMetricResult
from .ratio import RatioMetricResult
from .results import MetricResult


def adjust_p_values(
    p_values: Sequence[float | None],
    method: Multiplicity,
) -> tuple[float | None, ...]:
    """Adjust p-values with no correction, Holm, or Benjamini-Hochberg FDR.

    ``None`` values represent non-estimable metrics and are excluded from the
    correction family while retaining their original positions in the result.
    """
    _validate_method(method)
    indexed = _indexed_p_values(p_values)
    if method == "none":
        return tuple(p_values)
    if not indexed:
        return tuple(None for _ in p_values)

    ordered = sorted(indexed, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted_by_index: dict[int, float] = {}
    if method == "holm":
        running_max = 0.0
        for rank, (index, p_value) in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * p_value)
            running_max = max(running_max, adjusted)
            adjusted_by_index[index] = running_max
    else:
        running_min = 1.0
        for rank in range(count - 1, -1, -1):
            index, p_value = ordered[rank]
            adjusted = min(1.0, count * p_value / (rank + 1))
            running_min = min(running_min, adjusted)
            adjusted_by_index[index] = running_min
    return tuple(
        None if p_value is None else adjusted_by_index[index]
        for index, p_value in enumerate(p_values)
    )


def simultaneous_ci_levels(
    p_values: Sequence[float | None],
    method: Multiplicity,
    alpha: float = 0.05,
) -> tuple[float | None, ...]:
    """Return per-metric confidence levels for a corrected interval family.

    Holm intervals invert the Holm step-down tests, so the smallest raw
    p-value receives the most stringent level. ``fdr_bh`` controls the false
    discovery rate rather than family-wise coverage and therefore has no
    exact simultaneous interval construction from BH alone. For a safe,
    family-wise interval contract, this function uses a conservative
    Bonferroni interval for the BH family and labels it accordingly in metric
    results.

    ``None`` p-values remain ``None`` and do not consume family alpha.
    """
    _validate_method(method)
    _validate_alpha(alpha)
    indexed = _indexed_p_values(p_values)
    if not indexed:
        return tuple(None for _ in p_values)
    count = len(indexed)
    levels_by_index: dict[int, float] = {}
    if method == "holm":
        ordered = sorted(indexed, key=lambda item: (item[1], item[0]))
        for rank, (index, _p_value) in enumerate(ordered):
            levels_by_index[index] = 1.0 - alpha / (count - rank)
    elif method == "fdr_bh":
        level = 1.0 - alpha / count
        levels_by_index = {index: level for index, _p_value in indexed}
    else:
        level = 1.0 - alpha
        levels_by_index = {index: level for index, _p_value in indexed}
    return tuple(
        None if p_value is None else levels_by_index[index]
        for index, p_value in enumerate(p_values)
    )


def apply_multiplicity(
    results: Sequence[MetricResult],
    method: Multiplicity,
    alpha: float | None = None,
    scope: MultiplicityScope = "family",
) -> tuple[MetricResult, ...]:
    """Adjust p-values and add corrected intervals within each metric family.

    The original ``ci_lower``/``ci_upper`` fields remain nominal, pointwise
    intervals. Corrected intervals are exposed separately as
    ``simultaneous_ci_lower``/``simultaneous_ci_upper``. If ``alpha`` is not
    supplied, it is inferred from the first result's ``ci_level`` (or defaults
    to 0.05 for an empty result collection). With ``scope="global"``, all
    estimable metrics share one correction family regardless of their labels.
    """
    _validate_method(method)
    _validate_scope(scope)
    if alpha is None:
        alpha = _infer_alpha(results)
    _validate_alpha(alpha)

    family_indices: dict[str, list[int]] = {}
    for index, result in enumerate(results):
        if result.p_value is not None:
            family_indices.setdefault(result.family, []).append(index)
    correction_groups = (
        [[index for indices in family_indices.values() for index in indices]]
        if scope == "global"
        else list(family_indices.values())
    )

    adjusted: dict[int, float | None] = {}
    levels: dict[int, float | None] = {}
    for indices in correction_groups:
        family_p_values = [results[index].p_value for index in indices]
        family_adjusted = adjust_p_values(family_p_values, method)
        family_levels = simultaneous_ci_levels(
            family_p_values,
            method,
            alpha,
        )
        adjusted.update(zip(indices, family_adjusted, strict=True))
        levels.update(zip(indices, family_levels, strict=True))

    return tuple(
        _replace_with_adjusted_values(
            result,
            adjusted.get(index),
            levels.get(index),
            method,
        )
        for index, result in enumerate(results)
    )


def _replace_with_adjusted_values(
    result: MetricResult,
    adjusted_p_value: float | None,
    simultaneous_level: float | None,
    method: Multiplicity,
) -> MetricResult:
    """Add corrected p-value and interval fields to one metric result."""
    if result.p_value is None or simultaneous_level is None:
        return replace(
            result,
            adjusted_p_value=None,
            simultaneous_ci_lower=None,
            simultaneous_ci_upper=None,
            simultaneous_ci_level=None,
            simultaneous_ci_method=None,
        )
    interval = (
        (result.ci_lower, result.ci_upper)
        if method == "none"
        else _interval_at_level(result, simultaneous_level)
    )
    if method == "holm":
        interval_method = "holm_step_down"
    elif method == "fdr_bh":
        interval_method = "bonferroni_for_fdr"
    else:
        interval_method = "unadjusted"
    if interval is None:
        lower, upper = None, None
    else:
        lower, upper = interval
    return replace(
        result,
        adjusted_p_value=adjusted_p_value,
        simultaneous_ci_lower=lower,
        simultaneous_ci_upper=upper,
        simultaneous_ci_level=simultaneous_level,
        simultaneous_ci_method=interval_method,
    )


def _interval_at_level(
    result: MetricResult,
    level: float,
) -> tuple[float, float] | None:
    """Recompute one metric's interval at a corrected confidence level."""
    if result.status != "ok":
        return None
    if result.ci_lower is None or result.ci_upper is None:
        return None
    if isinstance(result, BinaryMetricResult):
        return _newcombe_difference_ci(
            result.control.successes,
            result.control.n,
            result.treatment.successes,
            result.treatment.n,
            level,
        )
    if isinstance(result, ContinuousMetricResult):
        if (
            result.absolute_effect is None
            or result.standard_error is None
            or result.degrees_of_freedom is None
        ):
            return None
        critical_value = stats.t.ppf(
            (1.0 + level) / 2.0,
            result.degrees_of_freedom,
        )
        margin = critical_value * result.standard_error
        return (
            result.absolute_effect - margin,
            result.absolute_effect + margin,
        )
    if isinstance(result, RatioMetricResult):
        if result.absolute_effect is None or result.standard_error is None:
            return None
        critical_value = stats.norm.ppf((1.0 + level) / 2.0)
        margin = critical_value * result.standard_error
        return (
            result.absolute_effect - margin,
            result.absolute_effect + margin,
        )
    return None


def _indexed_p_values(
    p_values: Sequence[float | None],
) -> list[tuple[int, float]]:
    """Validate and index estimable p-values."""
    indexed = [
        (index, p_value)
        for index, p_value in enumerate(p_values)
        if p_value is not None
    ]
    for index, p_value in indexed:
        if not isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(
                f"p_values[{index}] must be finite and in [0, 1], "
                f"got {p_value}"
            )
    return indexed


def _infer_alpha(results: Sequence[MetricResult]) -> float:
    """Infer the shared alpha from metric confidence levels."""
    for result in results:
        if result.ci_level is not None:
            return 1.0 - result.ci_level
    return 0.05


def _validate_method(method: Multiplicity) -> None:
    """Reject unsupported multiplicity methods."""
    if method not in ("none", "holm", "fdr_bh"):
        raise ValueError(f"unsupported multiplicity method: {method!r}")


def _validate_scope(scope: MultiplicityScope) -> None:
    """Reject unsupported correction scopes."""
    if scope not in ("family", "global"):
        raise ValueError(f"unsupported multiplicity scope: {scope!r}")


def _validate_alpha(alpha: float) -> None:
    """Validate a family-wise error rate."""
    if not isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be finite and in (0, 1), got {alpha}")
