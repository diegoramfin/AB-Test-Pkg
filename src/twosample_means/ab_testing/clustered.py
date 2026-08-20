"""Cluster-robust inference for continuous and count metric outcomes.

Cluster-robust standard errors (Liang and Zeger, 1986) relax the
independence assumption by allowing arbitrary within-cluster correlation.
For the two-arm saturated model ``Y ~ treatment``, the effect estimate is
the arm mean difference and its variance is the cluster-robust sandwich
``(X'X)^{-1} M (X'X)^{-1}`` with ``M = sum_g u_g u_g'`` where ``u_g``
contains the cluster sums of residuals for the intercept and treatment
columns.

A small-sample correction ``(G/(G-1)) * ((N-1)/(N-2))`` is applied and
inference uses a t distribution with ``G - 2`` degrees of freedom, where
``G`` is the number of clusters. When ``MetricSpec.covariate`` is also
configured, the sandwich runs on CUPED-adjusted outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._shared import (
    practical_significance,
    resolve_treatment,
    validate_metric_inputs,
)
from .config import ExperimentConfig, MetricSpec
from .continuous import ContinuousMetricResult, ContinuousSummary
from .data import NormalizedExperimentData


def estimate_clustered_metric(
    data: NormalizedExperimentData,
    config: ExperimentConfig,
    metric: MetricSpec,
    treatment: str | None = None,
) -> ContinuousMetricResult:
    """Estimate a cluster-robust treatment-control metric effect.

    The effect estimate matches the plain arm mean difference; only the
    uncertainty accounts for within-cluster correlation. Continuous and
    count metrics are supported, optionally combined with CUPED adjustment
    when ``metric.covariate`` is declared.
    """
    validate_metric_inputs(data, config, metric, metric.kind)
    if config.cluster is None:
        raise ValueError("cluster-robust estimation requires a cluster column")
    if metric.kind not in ("continuous", "count"):
        raise ValueError(
            "cluster-robust inference is only supported for continuous and "
            "count metrics"
        )
    treatment_label = resolve_treatment(config, treatment)
    frame = data.frame
    ci_level = 1.0 - config.alpha

    if metric.covariate is not None:
        try:
            adjusted = _cuped_adjusted_outcomes(frame, metric)
        except ValueError as error:
            return _not_estimable(
                metric,
                config,
                treatment_label,
                ci_level,
                str(error),
            )
    else:
        adjusted = frame[metric.column].to_numpy(dtype=float)

    control_mask = (frame[data.assignment] == config.control).to_numpy()
    treatment_mask = (frame[data.assignment] == treatment_label).to_numpy()
    cluster_ids = frame[config.cluster].astype(str).to_numpy()

    valid = (control_mask | treatment_mask) & ~np.isnan(adjusted)
    y = adjusted[valid]
    treat = treatment_mask[valid].astype(float)
    clusters = cluster_ids[valid]
    n = len(y)
    if n == 0:
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            "No observed outcomes for cluster-robust estimation.",
        )

    control_values = y[treat == 0.0]
    treatment_values = y[treat == 1.0]
    control = _summarize_cluster_arm(control_values, config.control)
    treatment_summary = _summarize_cluster_arm(
        treatment_values, treatment_label
    )
    if control.mean is None or treatment_summary.mean is None:
        missing_arm = "control" if control.mean is None else "treatment"
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            f"No observed {missing_arm} outcomes.",
        )
    if n < 2:
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            "Cluster-robust inference requires at least 2 rows.",
        )

    unique_clusters = np.unique(clusters)
    cluster_count = len(unique_clusters)
    if cluster_count < 3:
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            "Cluster-robust inference requires at least 3 clusters.",
        )

    effect = float(np.mean(treatment_values) - np.mean(control_values))
    residuals = y - np.where(
        treat == 1.0, np.mean(treatment_values), np.mean(control_values)
    )
    meat = _sandwich_meat(residuals, treat, clusters)
    xpx_inverse = _design_inverse(n, len(treatment_values))
    sandwich = xpx_inverse @ meat @ xpx_inverse
    correction = (cluster_count / (cluster_count - 1.0)) * (
        (n - 1.0) / (n - 2.0)
    )
    robust_variance = float(sandwich[1, 1]) * correction
    if not np.isfinite(robust_variance) or robust_variance <= 0.0:
        return _not_estimable(
            metric,
            config,
            treatment_label,
            ci_level,
            "Cluster-robust variance is not estimable for the observed data.",
        )
    standard_error = float(np.sqrt(robust_variance))
    naive_standard_error = _naive_standard_error(
        control_values,
        treatment_values,
    )
    degrees_of_freedom = float(cluster_count - 2)
    t_statistic = effect / standard_error
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom))
    critical_value = float(
        stats.t.ppf((1.0 + ci_level) / 2.0, degrees_of_freedom)
    )
    margin = critical_value * standard_error
    ci_lower = float(effect - margin)
    ci_upper = float(effect + margin)

    relative_lift = effect / control.mean if control.mean != 0.0 else None
    warnings: list[str] = []
    if control.mean == 0.0:
        warnings.append("Relative lift is undefined at zero control mean.")
    spanning = _clusters_spanning_arms(frame, config)
    if spanning:
        warnings.append(
            f"{spanning} cluster(s) span both arms; cluster-robust inference "
            "assumes clusters are nested within arms."
        )

    return ContinuousMetricResult(
        metric_name=metric.name,
        role=metric.role,
        family=metric.family,
        control_label=config.control,
        treatment_label=treatment_label,
        method="cluster_robust",
        status="ok",
        control=control,
        treatment=treatment_summary,
        absolute_effect=effect,
        relative_lift=relative_lift,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        p_value=p_value,
        adjusted_p_value=None,
        practical_effect=metric.practical_effect,
        practically_significant=practical_significance(
            ci_lower,
            ci_upper,
            metric.practical_effect,
        ),
        warnings=tuple(warnings),
        cluster_robust=True,
        clusters=cluster_count,
        naive_standard_error=naive_standard_error,
    )


def _summarize_cluster_arm(
    values: np.ndarray,
    label: str,
) -> ContinuousSummary:
    """Summarize one arm for a cluster-robust result."""
    n = len(values)
    if n == 0:
        return ContinuousSummary(label, 0, 0, None, None, None)
    standard_deviation = float(np.std(values, ddof=1)) if n >= 2 else None
    standard_error = (
        float(np.std(values, ddof=1) / np.sqrt(n)) if n >= 2 else None
    )
    return ContinuousSummary(
        label=label,
        n=n,
        missing=0,
        mean=float(np.mean(values)),
        standard_deviation=standard_deviation,
        standard_error=standard_error,
    )


def _sandwich_meat(
    residuals: np.ndarray,
    treat: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    """Return the cluster-sum ``M`` matrix of the robust sandwich."""
    frame = pd.DataFrame(
        {
            "residual": residuals,
            "unit_contribution": residuals * treat,
        },
        index=clusters,
    )
    grouped = frame.groupby(level=0, sort=False)
    u1 = grouped["residual"].sum().to_numpy()
    u2 = grouped["unit_contribution"].sum().to_numpy()
    meat = np.zeros((2, 2), dtype=float)
    meat[0, 0] = float(np.sum(u1**2))
    meat[0, 1] = meat[1, 0] = float(np.sum(u1 * u2))
    meat[1, 1] = float(np.sum(u2**2))
    return meat


def _design_inverse(total_n: int, treatment_n: int) -> np.ndarray:
    """Return ``(X'X)^{-1}`` for the saturated two-arm design."""
    control_n = total_n - treatment_n
    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("both arms need observed outcomes")
    return np.linalg.inv(
        np.array(
            [
                [float(total_n), float(treatment_n)],
                [float(treatment_n), float(treatment_n)],
            ]
        )
    )


def _naive_standard_error(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
) -> float:
    """Return the independence-assumption Welch standard error."""
    control_variance = float(np.var(control_values, ddof=1))
    treatment_variance = float(np.var(treatment_values, ddof=1))
    return float(
        np.sqrt(
            control_variance / len(control_values)
            + treatment_variance / len(treatment_values)
        )
    )


def _cuped_adjusted_outcomes(
    frame: pd.DataFrame,
    metric: MetricSpec,
) -> np.ndarray:
    """Calculate CUPED-adjusted outcomes for cluster-robust inference."""
    if metric.covariate is None:
        raise ValueError("CUPED adjustment requires a declared covariate")
    y = frame[metric.column].to_numpy(dtype=float)
    x = frame[metric.covariate].to_numpy(dtype=float)
    complete = ~(np.isnan(y) | np.isnan(x))
    if np.sum(complete) < 2:
        raise ValueError(
            "CUPED adjustment requires at least 2 complete covariate pairs"
        )
    if float(np.var(x[complete], ddof=1)) == 0.0:
        raise ValueError(
            "CUPED adjustment is not estimable for a constant covariate"
        )
    theta = float(
        np.cov(x[complete], y[complete], ddof=1)[0, 1]
        / np.var(x[complete], ddof=1)
    )
    adjusted = y - theta * (x - float(np.mean(x)))
    return np.asarray(adjusted, dtype=float)


def _clusters_spanning_arms(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> int:
    """Count clusters containing units from both arms."""
    if config.cluster is None:
        return 0
    table = pd.DataFrame(
        {
            "cluster": frame[config.cluster].astype(str),
            "arm": frame[config.assignment],
        }
    )
    arm_count = table.groupby("cluster", sort=False)["arm"].nunique()
    return int((arm_count > 1).sum())


def _not_estimable(
    metric: MetricSpec,
    config: ExperimentConfig,
    treatment_label: str,
    ci_level: float,
    warning: str,
) -> ContinuousMetricResult:
    """Build a structured unavailable cluster-robust result."""
    return ContinuousMetricResult(
        metric_name=metric.name,
        role=metric.role,
        family=metric.family,
        control_label=config.control,
        treatment_label=treatment_label,
        method="cluster_robust",
        status="not_estimable",
        control=ContinuousSummary(config.control, 0, 0, None, None, None),
        treatment=ContinuousSummary(treatment_label, 0, 0, None, None, None),
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
        cluster_robust=True,
        clusters=None,
        naive_standard_error=None,
    )
