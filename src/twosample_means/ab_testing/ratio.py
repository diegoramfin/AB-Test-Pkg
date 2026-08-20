"""Ratio metric estimation with user-level delta-method inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from ._shared import (
    practical_significance,
    resolve_treatment,
    validate_metric_inputs,
)
from .config import ExperimentConfig, MetricSpec
from .data import NormalizedExperimentData

RatioStatus = Literal["ok", "not_estimable"]


@dataclass(frozen=True)
class RatioSummary:
    """Numerator, denominator, and ratio summary for one arm."""

    label: str
    n: int
    missing: int
    numerator_mean: float | None
    denominator_mean: float | None
    ratio: float | None
    standard_error: float | None


@dataclass(frozen=True)
class RatioMetricResult:
    """Treatment-control difference in ratios using the delta method."""

    metric_name: str
    role: str
    family: str
    control_label: str
    treatment_label: str
    method: str
    status: RatioStatus
    control: RatioSummary
    treatment: RatioSummary
    absolute_effect: float | None
    relative_lift: float | None
    standard_error: float | None
    degrees_of_freedom: float | None
    ci_lower: float | None
    ci_upper: float | None
    ci_level: float
    p_value: float | None
    adjusted_p_value: float | None
    practical_effect: float | None
    practically_significant: bool | None
    warnings: tuple[str, ...] = ()
    simultaneous_ci_lower: float | None = None
    simultaneous_ci_upper: float | None = None
    simultaneous_ci_level: float | None = None
    simultaneous_ci_method: str | None = None
    contrast_name: str | None = None


def estimate_ratio_metric(
    data: NormalizedExperimentData,
    config: ExperimentConfig,
    metric: MetricSpec,
    treatment: str | None = None,
) -> RatioMetricResult:
    """Estimate a treatment-control ratio-of-means difference.

    The arm-level estimand is ``mean(numerator) / mean(denominator)``. The
    standard error uses the user-level influence value
    ``(numerator - ratio * denominator) / mean(denominator)``. Denominators
    must be positive and are validated during normalization.
    """
    validate_metric_inputs(data, config, metric, "ratio")
    if metric.numerator is None or metric.denominator is None:
        raise ValueError("ratio metrics require numerator and denominator")
    treatment_label = resolve_treatment(config, treatment)
    frame = data.frame
    control = _summarize_arm(
        frame[frame[data.assignment] == config.control],
        config.control,
        metric.numerator,
        metric.denominator,
    )
    treatment_summary = _summarize_arm(
        frame[frame[data.assignment] == treatment_label],
        treatment_label,
        metric.numerator,
        metric.denominator,
    )
    ci_level = 1.0 - config.alpha
    if control.ratio is None or treatment_summary.ratio is None:
        missing_arm = "control" if control.ratio is None else "treatment"
        return _not_estimable(
            metric,
            config,
            treatment_label,
            control,
            treatment_summary,
            ci_level,
            f"No observed {missing_arm} ratio outcomes.",
        )
    if control.n < 2 or treatment_summary.n < 2:
        return _not_estimable(
            metric,
            config,
            treatment_label,
            control,
            treatment_summary,
            ci_level,
            "Delta-method inference requires at least 2 observed outcomes "
            "per arm.",
        )
    if (
        control.standard_error is None
        or treatment_summary.standard_error is None
    ):
        return _not_estimable(
            metric,
            config,
            treatment_label,
            control,
            treatment_summary,
            ci_level,
            "Ratio uncertainty is not estimable for the observed data.",
        )

    effect = treatment_summary.ratio - control.ratio
    standard_error = float(
        np.sqrt(
            control.standard_error**2 + treatment_summary.standard_error**2
        )
    )
    if not np.isfinite(standard_error) or standard_error <= 0.0:
        return _not_estimable(
            metric,
            config,
            treatment_label,
            control,
            treatment_summary,
            ci_level,
            "Ratio uncertainty is not estimable for the observed data.",
        )
    z_statistic = effect / standard_error
    p_value = float(2.0 * stats.norm.sf(abs(z_statistic)))
    critical_value = float(stats.norm.ppf((1.0 + ci_level) / 2.0))
    margin = critical_value * standard_error
    ci_lower, ci_upper = effect - margin, effect + margin
    relative_lift = effect / control.ratio if control.ratio != 0.0 else None
    warnings: list[str] = []
    if control.ratio == 0.0:
        warnings.append("Relative lift is undefined at zero control ratio.")
    return RatioMetricResult(
        metric_name=metric.name,
        role=metric.role,
        family=metric.family,
        control_label=config.control,
        treatment_label=treatment_label,
        method="delta_method_ratio",
        status="ok",
        control=control,
        treatment=treatment_summary,
        absolute_effect=effect,
        relative_lift=relative_lift,
        standard_error=standard_error,
        degrees_of_freedom=None,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_level=ci_level,
        p_value=p_value,
        adjusted_p_value=None,
        practical_effect=metric.practical_effect,
        practically_significant=practical_significance(
            float(ci_lower), float(ci_upper), metric.practical_effect
        ),
        warnings=tuple(warnings),
    )


def _summarize_arm(
    frame: pd.DataFrame,
    label: str,
    numerator_column: str,
    denominator_column: str,
) -> RatioSummary:
    """Summarize one arm and calculate its delta-method standard error."""
    numerator = frame[numerator_column].to_numpy(dtype=float)
    denominator = frame[denominator_column].to_numpy(dtype=float)
    missing_mask = np.isnan(numerator) | np.isnan(denominator)
    numerator = numerator[~missing_mask]
    denominator = denominator[~missing_mask]
    n = len(numerator)
    missing = int(np.sum(missing_mask))
    if n == 0:
        return RatioSummary(label, 0, missing, None, None, None, None)
    numerator_mean = float(np.mean(numerator))
    denominator_mean = float(np.mean(denominator))
    if denominator_mean <= 0.0:
        return RatioSummary(
            label,
            n,
            missing,
            numerator_mean,
            denominator_mean,
            None,
            None,
        )
    ratio = numerator_mean / denominator_mean
    standard_error = None
    if n >= 2:
        influence = (numerator - ratio * denominator) / denominator_mean
        standard_error = float(np.std(influence, ddof=1) / np.sqrt(n))
    return RatioSummary(
        label,
        n,
        missing,
        numerator_mean,
        denominator_mean,
        float(ratio),
        standard_error,
    )


def _not_estimable(
    metric: MetricSpec,
    config: ExperimentConfig,
    treatment_label: str,
    control: RatioSummary,
    treatment: RatioSummary,
    ci_level: float,
    warning: str,
) -> RatioMetricResult:
    """Build a structured unavailable ratio result."""
    return RatioMetricResult(
        metric_name=metric.name,
        role=metric.role,
        family=metric.family,
        control_label=config.control,
        treatment_label=treatment_label,
        method="delta_method_ratio",
        status="not_estimable",
        control=control,
        treatment=treatment,
        absolute_effect=None,
        relative_lift=None,
        standard_error=None,
        degrees_of_freedom=None,
        ci_lower=None,
        ci_upper=None,
        ci_level=ci_level,
        p_value=None,
        adjusted_p_value=None,
        practical_effect=metric.practical_effect,
        practically_significant=None,
        warnings=(warning,),
    )
