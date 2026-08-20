"""Continuous metric estimation for randomized two-arm experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from twosample_means.frequentist_parametric import welch_t

from ._shared import (
    practical_significance,
    resolve_treatment,
    validate_metric_inputs,
    welch_run_config,
)
from .config import ExperimentConfig, MetricSpec
from .data import NormalizedExperimentData

ContinuousStatus = Literal["ok", "not_estimable"]


@dataclass(frozen=True)
class ContinuousSummary:
    """Observed continuous-outcome summary for one experiment arm."""

    label: str
    n: int
    missing: int
    mean: float | None
    standard_deviation: float | None
    standard_error: float | None


@dataclass(frozen=True)
class ContinuousMetricResult:
    """Treatment-control Welch result for one continuous metric.

    Effects are oriented as treatment minus control. The interval, p-value,
    and degrees of freedom come from the project's existing Welch estimator.
    """

    metric_name: str
    role: str
    family: str
    control_label: str
    treatment_label: str
    method: str
    status: ContinuousStatus
    control: ContinuousSummary
    treatment: ContinuousSummary
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
    cluster_robust: bool = False
    clusters: int | None = None
    naive_standard_error: float | None = None


def estimate_continuous_metric(
    data: NormalizedExperimentData,
    config: ExperimentConfig,
    metric: MetricSpec,
    treatment: str | None = None,
) -> ContinuousMetricResult:
    """Estimate a treatment-control difference for a continuous metric."""
    return _estimate_continuous_metric(
        data,
        config,
        metric,
        treatment,
        expected_kind="continuous",
        method_name="welch_t",
    )


def _estimate_continuous_metric(
    data: NormalizedExperimentData,
    config: ExperimentConfig,
    metric: MetricSpec,
    treatment: str | None,
    *,
    expected_kind: str,
    method_name: str,
) -> ContinuousMetricResult:
    """Estimate a continuous-like metric with configurable metadata.

    Parameters
    ----------
    data:
        Normalized user-level experiment data.
    config:
        Experiment analysis plan used to identify the arms and alpha.
    metric:
        A continuous metric declared in ``config``.
    treatment:
        Treatment label to compare with control. It may be omitted only when
        the configuration declares exactly one treatment arm.

    Returns
    -------
    ContinuousMetricResult
        Arm summaries, Welch mean difference, confidence interval, p-value,
        relative lift, and practical-effect metadata.
    """
    validate_metric_inputs(data, config, metric, expected_kind)
    treatment_label = resolve_treatment(config, treatment)
    frame = data.frame
    control_rows = frame[frame[data.assignment] == config.control]
    treatment_rows = frame[frame[data.assignment] == treatment_label]
    control, control_values = _summarize_arm(
        control_rows[metric.column], config.control
    )
    treatment_summary, treatment_values = _summarize_arm(
        treatment_rows[metric.column], treatment_label
    )
    ci_level = 1.0 - config.alpha

    if control.mean is None or treatment_summary.mean is None:
        missing_arm = "control" if control.mean is None else "treatment"
        return _not_estimable_result(
            metric,
            config,
            treatment_label,
            control,
            treatment_summary,
            ci_level,
            f"No observed {missing_arm} outcomes.",
            method_name,
        )
    if control.n < 2 or treatment_summary.n < 2:
        return _not_estimable_result(
            metric,
            config,
            treatment_label,
            control,
            treatment_summary,
            ci_level,
            "Welch inference requires at least 2 observed outcomes per arm.",
            method_name,
        )
    if (
        control.standard_deviation == 0.0
        and treatment_summary.standard_deviation == 0.0
    ):
        return _not_estimable_result(
            metric,
            config,
            treatment_label,
            control,
            treatment_summary,
            ci_level,
            "Welch uncertainty is not estimable when both arms are constant.",
            method_name,
        )

    welch_result = welch_t(
        treatment_values,
        control_values,
        welch_run_config(config),
    )
    absolute_effect = welch_result.mean_difference
    relative_lift = (
        absolute_effect / control.mean if control.mean != 0.0 else None
    )
    practically_significant = practical_significance(
        welch_result.ci_lower,
        welch_result.ci_upper,
        metric.practical_effect,
    )
    warnings: list[str] = []
    if control.mean == 0.0:
        warnings.append("Relative lift is undefined at zero control mean.")

    return ContinuousMetricResult(
        metric_name=metric.name,
        role=metric.role,
        family=metric.family,
        control_label=config.control,
        treatment_label=treatment_label,
        method=method_name,
        status="ok",
        control=control,
        treatment=treatment_summary,
        absolute_effect=absolute_effect,
        relative_lift=relative_lift,
        standard_error=_welch_standard_error(treatment_values, control_values),
        degrees_of_freedom=welch_result.degrees_of_freedom,
        ci_lower=welch_result.ci_lower,
        ci_upper=welch_result.ci_upper,
        ci_level=ci_level,
        p_value=welch_result.p_value,
        adjusted_p_value=None,
        practical_effect=metric.practical_effect,
        practically_significant=practically_significant,
        warnings=tuple(warnings),
    )


def _summarize_arm(
    values: pd.Series,
    label: str,
) -> tuple[ContinuousSummary, np.ndarray]:
    """Summarize a continuous arm and return its observed values."""
    missing_mask = values.isna()
    missing = int(missing_mask.sum())
    observed = values.loc[~missing_mask].to_numpy(dtype=float)
    n = len(observed)
    mean = float(np.mean(observed)) if n else None
    standard_deviation = float(np.std(observed, ddof=1)) if n >= 2 else None
    standard_error = (
        standard_deviation / np.sqrt(n)
        if standard_deviation is not None
        else None
    )
    return (
        ContinuousSummary(
            label=label,
            n=n,
            missing=missing,
            mean=mean,
            standard_deviation=standard_deviation,
            standard_error=standard_error,
        ),
        observed,
    )


def _welch_standard_error(
    treatment: np.ndarray,
    control: np.ndarray,
) -> float:
    """Compute the standard error corresponding to Welch's t statistic."""
    return float(
        np.sqrt(
            np.var(treatment, ddof=1) / len(treatment)
            + np.var(control, ddof=1) / len(control)
        )
    )


def _not_estimable_result(
    metric: MetricSpec,
    config: ExperimentConfig,
    treatment_label: str,
    control: ContinuousSummary,
    treatment: ContinuousSummary,
    ci_level: float,
    warning: str,
    method_name: str = "welch_t",
) -> ContinuousMetricResult:
    """Build a structured result when Welch inference cannot run."""
    return ContinuousMetricResult(
        metric_name=metric.name,
        role=metric.role,
        family=metric.family,
        control_label=config.control,
        treatment_label=treatment_label,
        method=method_name,
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
