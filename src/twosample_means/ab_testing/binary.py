"""Binary metric estimation for randomized two-arm experiments."""

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

BinaryStatus = Literal["ok", "not_estimable"]


@dataclass(frozen=True)
class BinarySummary:
    """Observed binary outcome summary for one experiment arm."""

    label: str
    n: int
    successes: int
    missing: int
    rate: float | None


@dataclass(frozen=True)
class BinaryMetricResult:
    """Treatment-control result for one binary metric.

    Effects are oriented as treatment minus control. The confidence interval
    is a Newcombe interval constructed from independent Wilson score intervals;
    the p-value is the pooled two-proportion score test.
    """

    metric_name: str
    role: str
    family: str
    control_label: str
    treatment_label: str
    method: str
    status: BinaryStatus
    control: BinarySummary
    treatment: BinarySummary
    absolute_effect: float | None
    relative_lift: float | None
    risk_ratio: float | None
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


def estimate_binary_metric(
    data: NormalizedExperimentData,
    config: ExperimentConfig,
    metric: MetricSpec,
    treatment: str | None = None,
) -> BinaryMetricResult:
    """Estimate a treatment-control difference for a binary metric.

    Parameters
    ----------
    data:
        Normalized user-level experiment data.
    config:
        Experiment analysis plan used to identify the arms and alpha.
    metric:
        A binary metric declared in ``config``.
    treatment:
        Treatment label to compare with control. It may be omitted only when
        the configuration declares exactly one treatment arm.

    Returns
    -------
    BinaryMetricResult
        Rates, absolute difference, relative lift, risk ratio, Newcombe-Wilson
        interval, and pooled score-test p-value.

    Raises
    ------
    ValueError
        If the metric is not declared as binary in the plan, the normalized
        data does not match the plan, or a multi-treatment plan omits the arm.
    """
    validate_metric_inputs(data, config, metric, "binary")
    treatment_label = resolve_treatment(config, treatment)
    frame = data.frame
    control_rows = frame[frame[data.assignment] == config.control]
    treatment_rows = frame[frame[data.assignment] == treatment_label]
    control = _summarize_arm(control_rows[metric.column], config.control)
    treatment_summary = _summarize_arm(
        treatment_rows[metric.column], treatment_label
    )
    ci_level = 1.0 - config.alpha

    if control.rate is None or treatment_summary.rate is None:
        missing_arm = "control" if control.rate is None else "treatment"
        return BinaryMetricResult(
            metric_name=metric.name,
            role=metric.role,
            family=metric.family,
            control_label=config.control,
            treatment_label=treatment_label,
            method="score_test_newcombe_wilson",
            status="not_estimable",
            control=control,
            treatment=treatment_summary,
            absolute_effect=None,
            relative_lift=None,
            risk_ratio=None,
            ci_lower=None,
            ci_upper=None,
            ci_level=ci_level,
            p_value=None,
            adjusted_p_value=None,
            practical_effect=metric.practical_effect,
            practically_significant=None,
            warnings=(f"No observed {missing_arm} outcomes.",),
        )

    absolute_effect = treatment_summary.rate - control.rate
    relative_lift = (
        absolute_effect / control.rate if control.rate > 0.0 else None
    )
    risk_ratio = (
        treatment_summary.rate / control.rate if control.rate > 0.0 else None
    )
    ci_lower, ci_upper = _newcombe_difference_ci(
        control.successes,
        control.n,
        treatment_summary.successes,
        treatment_summary.n,
        ci_level,
    )
    p_value = _pooled_score_test(
        control.successes,
        control.n,
        treatment_summary.successes,
        treatment_summary.n,
    )
    practically_significant = practical_significance(
        ci_lower,
        ci_upper,
        metric.practical_effect,
    )
    warnings: list[str] = []
    if control.rate == 0.0:
        warnings.append(
            "Relative lift and risk ratio are undefined at zero control rate."
        )

    return BinaryMetricResult(
        metric_name=metric.name,
        role=metric.role,
        family=metric.family,
        control_label=config.control,
        treatment_label=treatment_label,
        method="score_test_newcombe_wilson",
        status="ok",
        control=control,
        treatment=treatment_summary,
        absolute_effect=absolute_effect,
        relative_lift=relative_lift,
        risk_ratio=risk_ratio,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        p_value=p_value,
        adjusted_p_value=None,
        practical_effect=metric.practical_effect,
        practically_significant=practically_significant,
        warnings=tuple(warnings),
    )


def _summarize_arm(values: pd.Series, label: str) -> BinarySummary:
    """Summarize canonical 0/1 values while preserving missing counts."""
    missing_mask = values.isna()
    missing = int(missing_mask.sum())
    observed = values.loc[~missing_mask]
    n = int(len(observed))
    successes = int(np.sum(np.asarray(observed, dtype=float)))
    rate = successes / n if n else None
    return BinarySummary(
        label=label,
        n=n,
        successes=successes,
        missing=missing,
        rate=rate,
    )


def _pooled_score_test(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
) -> float:
    """Return the two-sided pooled score-test p-value."""
    pooled_rate = (control_successes + treatment_successes) / (
        control_n + treatment_n
    )
    standard_error = np.sqrt(
        pooled_rate
        * (1.0 - pooled_rate)
        * (1.0 / control_n + 1.0 / treatment_n)
    )
    if standard_error == 0.0:
        return 1.0
    observed_difference = treatment_successes / treatment_n - (
        control_successes / control_n
    )
    z_statistic = observed_difference / standard_error
    return float(2.0 * stats.norm.sf(abs(z_statistic)))


def _newcombe_difference_ci(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    ci_level: float,
) -> tuple[float, float]:
    """Return Newcombe's interval from independent Wilson intervals."""
    control_rate = control_successes / control_n
    treatment_rate = treatment_successes / treatment_n
    control_lower, control_upper = _wilson_interval(
        control_successes, control_n, ci_level
    )
    treatment_lower, treatment_upper = _wilson_interval(
        treatment_successes, treatment_n, ci_level
    )
    difference = treatment_rate - control_rate
    lower = difference - np.sqrt(
        (treatment_rate - treatment_lower) ** 2
        + (control_upper - control_rate) ** 2
    )
    upper = difference + np.sqrt(
        (treatment_upper - treatment_rate) ** 2
        + (control_rate - control_lower) ** 2
    )
    return max(-1.0, float(lower)), min(1.0, float(upper))


def _wilson_interval(
    successes: int,
    n: int,
    ci_level: float,
) -> tuple[float, float]:
    """Return a Wilson score interval for one proportion."""
    rate = successes / n
    z = float(stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0))
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = (rate + z_squared / (2.0 * n)) / denominator
    margin = (
        z
        * np.sqrt(rate * (1.0 - rate) / n + z_squared / (4.0 * n * n))
        / denominator
    )
    return max(0.0, float(center - margin)), min(1.0, float(center + margin))
