"""CUPED variance reduction for continuous and count metric outcomes.

CUPED (Controlled-experiment Using Pre-Experiment Data; Deng, Xu, Kohavi &
Xu, 2013) adjusts outcomes with a pre-experiment covariate observed before
treatment. The adjusted outcome is ``Y - theta * (X - mean(X))`` where
``theta`` is the pooled covariate slope ``Cov(X, Y) / Var(X)``.

The adjusted treatment effect estimate is unbiased whenever the covariate is
measured before treatment, and its variance is reduced by approximately the
squared correlation between covariate and outcome. The caller declares the
``covariate`` column on ``MetricSpec``; this package validates that the
column exists and is numeric but cannot verify that the covariate predates
treatment.
"""

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

CupedStatus = Literal["ok", "not_estimable"]


@dataclass(frozen=True)
class CupedSummary:
    """Adjusted and unadjusted outcome summary for one experiment arm."""

    label: str
    n: int
    missing: int
    mean: float | None
    unadjusted_mean: float | None
    standard_deviation: float | None
    standard_error: float | None


@dataclass(frozen=True)
class CupedMetricResult:
    """CUPED-adjusted treatment-control result for one metric.

    Effects are oriented as treatment minus control on adjusted outcomes.
    The confidence interval and p-value use Welch inference on the adjusted
    outcomes. ``variance_reduction`` reports ``1 - Var(adjusted)/Var(raw)``
    on the pooled sample.
    """

    metric_name: str
    role: str
    family: str
    control_label: str
    treatment_label: str
    method: str
    status: CupedStatus
    control: CupedSummary
    treatment: CupedSummary
    absolute_effect: float | None
    unadjusted_absolute_effect: float | None
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
    theta: float | None
    correlation: float | None
    variance_reduction: float | None
    warnings: tuple[str, ...] = ()
    simultaneous_ci_lower: float | None = None
    simultaneous_ci_upper: float | None = None
    simultaneous_ci_level: float | None = None
    simultaneous_ci_method: str | None = None
    contrast_name: str | None = None


def estimate_cuped_metric(
    data: NormalizedExperimentData,
    config: ExperimentConfig,
    metric: MetricSpec,
    treatment: str | None = None,
) -> CupedMetricResult:
    """Estimate a CUPED-adjusted treatment-control metric effect.

    Units with a missing covariate or missing outcome are excluded from the
    estimate and reported in the arm summaries. The adjusted effect uses the
    pooled covariate slope ``theta``, which for a pre-treatment covariate
    preserves the unadjusted effect while reducing variance.
    """
    validate_metric_inputs(data, config, metric, metric.kind)
    if metric.covariate is None:
        raise ValueError("CUPED estimation requires a declared covariate")
    treatment_label = resolve_treatment(config, treatment)
    frame = data.frame
    ci_level = 1.0 - config.alpha

    pooled_y = frame[metric.column].to_numpy(dtype=float)
    pooled_x = frame[metric.covariate].to_numpy(dtype=float)
    complete = ~(np.isnan(pooled_y) | np.isnan(pooled_x))
    x = pooled_x[complete]
    y = pooled_y[complete]
    if len(x) < 2:
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            "Fewer than 2 complete covariate-outcome pairs.",
        )
    if float(np.var(x, ddof=1)) == 0.0:
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            "CUPED is not estimable when the covariate is constant.",
        )

    theta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    if not np.isfinite(theta):
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            "CUPED slope is not finite for the observed data.",
        )

    control_rows = frame[frame[data.assignment] == config.control]
    treatment_rows = frame[frame[data.assignment] == treatment_label]
    control = _summarize_arm(
        control_rows[metric.column],
        control_rows[metric.covariate],
        config.control,
        theta,
    )
    treatment_summary = _summarize_arm(
        treatment_rows[metric.column],
        treatment_rows[metric.covariate],
        treatment_label,
        theta,
    )
    if control.mean is None or treatment_summary.mean is None:
        missing_arm = "control" if control.mean is None else "treatment"
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            f"No observed {missing_arm} adjusted outcomes.",
        )
    if control.n < 2 or treatment_summary.n < 2:
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            "Welch inference requires at least 2 observed adjusted outcomes "
            "per arm.",
        )

    if control.standard_deviation is None or (
        treatment_summary.standard_deviation is None
    ):
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            "Adjusted uncertainty is not estimable for the observed data.",
        )
    assert control.standard_deviation is not None
    assert treatment_summary.standard_deviation is not None
    assert control.unadjusted_mean is not None
    assert treatment_summary.unadjusted_mean is not None
    standard_error = float(
        np.sqrt(
            control.standard_deviation**2 / control.n
            + treatment_summary.standard_deviation**2 / treatment_summary.n
        )
    )
    effect = float(treatment_summary.mean - control.mean)
    unadjusted_effect = float(
        treatment_summary.unadjusted_mean - control.unadjusted_mean
    )
    # CUPED statistics are within-arm: mixing arm means into the covariate
    # slope inflates the apparent correlation and understates variance
    # reduction. Center each arm at its own covariate mean, matching the
    # standard errors used for inference.
    correlation, variance_reduction = _within_arm_cuped_stats(
        frame,
        data.assignment,
        metric,
        config.control,
        treatment_label,
        theta,
    )
    if standard_error == 0.0:
        p_value = 0.0 if effect != 0.0 else 1.0
        degrees_of_freedom = float(control.n + treatment_summary.n - 2)
        ci_lower = ci_upper = effect
    else:
        degrees_of_freedom = _welch_degrees_of_freedom(
            control.standard_deviation,
            control.n,
            treatment_summary.standard_deviation,
            treatment_summary.n,
        )
        t_statistic = effect / standard_error
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom))
        critical_value = float(
            stats.t.ppf((1.0 + ci_level) / 2.0, degrees_of_freedom)
        )
        margin = critical_value * standard_error
        ci_lower, ci_upper = effect - margin, effect + margin

    relative_lift = (
        unadjusted_effect / control.unadjusted_mean
        if control.unadjusted_mean != 0.0
        else None
    )
    warnings: list[str] = []
    if control.unadjusted_mean == 0.0:
        warnings.append("Relative lift is undefined at zero control mean.")
    excluded = control.missing + treatment_summary.missing
    if excluded:
        warnings.append(
            f"{excluded} row(s) excluded because covariate or outcome "
            "was missing for the CUPED estimate."
        )

    return CupedMetricResult(
        metric_name=metric.name,
        role=metric.role,
        family=metric.family,
        control_label=config.control,
        treatment_label=treatment_label,
        method="cuped_welch",
        status="ok",
        control=control,
        treatment=treatment_summary,
        absolute_effect=effect,
        unadjusted_absolute_effect=unadjusted_effect,
        relative_lift=relative_lift,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_level=ci_level,
        p_value=float(p_value),
        adjusted_p_value=None,
        practical_effect=metric.practical_effect,
        practically_significant=practical_significance(
            float(ci_lower),
            float(ci_upper),
            metric.practical_effect,
        ),
        theta=theta,
        correlation=correlation,
        variance_reduction=variance_reduction,
        warnings=tuple(warnings),
    )


def _summarize_arm(
    outcome: pd.Series,
    covariate: pd.Series,
    label: str,
    theta: float,
) -> CupedSummary:
    """Compute adjusted and unadjusted summaries for one arm."""
    y = outcome.to_numpy(dtype=float)
    x = covariate.to_numpy(dtype=float)
    missing_mask = np.isnan(y) | np.isnan(x)
    missing = int(np.sum(missing_mask))
    y = y[~missing_mask]
    x = x[~missing_mask]
    n = len(y)
    if n == 0:
        return CupedSummary(label, 0, missing, None, None, None, None)
    adjusted = y - theta * (x - float(np.mean(x)))
    if n < 2:
        return CupedSummary(
            label,
            n,
            missing,
            float(np.mean(adjusted)),
            float(np.mean(y)),
            None,
            None,
        )
    standard_deviation = float(np.std(adjusted, ddof=1))
    return CupedSummary(
        label=label,
        n=n,
        missing=missing,
        mean=float(np.mean(adjusted)),
        unadjusted_mean=float(np.mean(y)),
        standard_deviation=standard_deviation,
        standard_error=standard_deviation / np.sqrt(n),
    )


def _within_arm_cuped_stats(
    frame: pd.DataFrame,
    assignment_column: str,
    metric: MetricSpec,
    control: str,
    treatment: str,
    theta: float,
) -> tuple[float | None, float | None]:
    """Return within-arm covariate correlation and variance reduction.

    Both statistics pool within-arm sums of squares, so the arm allocation
    cannot masquerade as predictive power. ``variance_reduction`` is
    ``1 - Var(adjusted)/Var(raw)`` computed from the pooled within-arm
    variances.
    """
    if metric.covariate is None:
        return None, None
    raw_squares = 0.0
    adjusted_squares = 0.0
    within_degrees = 0
    covariance = 0.0
    raw_variance_x = 0.0
    raw_variance_y = 0.0
    for arm in (control, treatment):
        rows = frame[frame[assignment_column] == arm]
        y = rows[metric.column].to_numpy(dtype=float)
        sigma = rows[metric.covariate].to_numpy(dtype=float)
        complete = ~(np.isnan(y) | np.isnan(sigma))
        y = y[complete]
        sigma = sigma[complete]
        n = len(y)
        if n < 2:
            continue
        mean_y = float(np.mean(y))
        mean_x = float(np.mean(sigma))
        adjusted = y - theta * (sigma - mean_x)
        raw_squares += float(np.sum((y - mean_y) ** 2))
        adjusted_squares += float(np.sum((adjusted - np.mean(adjusted)) ** 2))
        covariance += float(np.sum((sigma - mean_x) * (y - mean_y)))
        raw_variance_x += float(np.sum((sigma - mean_x) ** 2))
        raw_variance_y += float(np.sum((y - mean_y) ** 2))
        within_degrees += n - 1
    if within_degrees == 0 or raw_squares == 0.0:
        return None, None
    correlation = None
    if raw_variance_x > 0.0 and raw_variance_y > 0.0:
        correlation = covariance / np.sqrt(raw_variance_x * raw_variance_y)
    variance_reduction = 1.0 - adjusted_squares / raw_squares
    return correlation, float(variance_reduction)


def _welch_degrees_of_freedom(
    sd_a: float,
    n_a: int,
    sd_b: float,
    n_b: int,
) -> float:
    """Compute Welch-Satterthwaite degrees of freedom."""
    se_a = sd_a**2 / n_a
    se_b = sd_b**2 / n_b
    numerator = (se_a + se_b) ** 2
    denominator = se_a**2 / (n_a - 1) + se_b**2 / (n_b - 1)
    if denominator == 0.0:
        return float(n_a + n_b - 2)
    return float(numerator / denominator)


def _not_estimable(
    metric: MetricSpec,
    config: ExperimentConfig,
    treatment_label: str,
    ci_level: float,
    warning: str,
) -> CupedMetricResult:
    """Build a structured unavailable CUPED result."""
    return CupedMetricResult(
        metric_name=metric.name,
        role=metric.role,
        family=metric.family,
        control_label=config.control,
        treatment_label=treatment_label,
        method="cuped_welch",
        status="not_estimable",
        control=CupedSummary(config.control, 0, 0, None, None, None, None),
        treatment=CupedSummary(treatment_label, 0, 0, None, None, None, None),
        absolute_effect=None,
        unadjusted_absolute_effect=None,
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
        theta=None,
        correlation=None,
        variance_reduction=None,
        warnings=(warning,),
    )
