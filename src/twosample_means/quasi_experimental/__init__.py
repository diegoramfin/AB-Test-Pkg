"""Quasi-experimental designs: difference in differences.

This namespace is deliberately separate from ``twosample_means.ab_testing``:
randomized A/B inference and quasi-experimental panel methods rest on
different identifying assumptions, and mixing them invites causal
overreach. DiD estimates rely on the parallel-trends assumption plus no
anticipation and no interference; nothing here verifies those assumptions
from data alone, so results are report-only descriptive estimates.

Implemented designs
-------------------

- Canonical two-group, two-period (or multi-period) difference-in-differences
  interaction model with unit and period fixed effects.
- Cluster-robust standard errors (CR1 sandwich, ``G-2`` degrees of freedom)
  by a declared cluster column or by unit.
- Time-varying covariate adjustment in the within-unit regression.
- Event-study coefficients with an omitted reference period (the last pre
  period) and a joint parallel-trends placebo test on the pre-period
  coefficients, when at least two pre periods exist.
- Treatment-timing validation: unit-specific or staggered post indicators
  are rejected because the canonical design requires a single global onset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd
from scipy import stats

# Identifying assumptions reported for every DiD result.
_ASSUMPTION_NOTES = (
    "Parallel trends: in the absence of treatment, treated and control arms "
    "must follow parallel counterfactual trends. The pre-period event-study "
    "coefficients are a placebo check of this assumption, not proof of it.",
    "No anticipation: treatment effects begin only in the first post period; "
    "units must not react to treatment before it is observed.",
    "No interference: one unit's treatment must not affect another unit's "
    "outcome (SUTVA).",
    "Constant effect: the canonical design reports an average effect over "
    "the post window; heterogeneous or time-varying effects require "
    "staggered-adoption estimators beyond this implementation.",
    "Unit fixed effects absorb time-invariant unit differences; period "
    "fixed effects absorb common time shocks. Time-invariant covariates "
    "are absorbed by the unit effects.",
)


@dataclass(frozen=True)
class GroupTimeMean:
    """Raw outcome mean for one treated-by-period cell."""

    treated: bool
    post: bool
    mean: float | None
    n: int


@dataclass(frozen=True)
class EventStudyCoefficient:
    """One event-study coefficient relative to the omitted reference period."""

    period: str
    relative_time: int
    coefficient: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    reference: bool = False


@dataclass(frozen=True)
class EventStudyResult:
    """Event-study coefficients and the pre-trend placebo test."""

    reference_period: str
    coefficients: tuple[EventStudyCoefficient, ...]
    pre_trend_p_value: float | None
    ci_level: float


@dataclass(frozen=True)
class DidResult:
    """Complete difference-in-differences estimation result."""

    method: str
    effect: float
    standard_error: float
    naive_standard_error: float
    degrees_of_freedom: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    p_value: float
    clusters: int
    units: int
    periods: int
    pre_periods: int
    post_periods: int
    covariates: tuple[str, ...]
    group_time_means: tuple[GroupTimeMean, ...]
    event_study: EventStudyResult | None
    assumption_notes: tuple[str, ...]
    warnings: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class DifferenceInDifferences:
    """Declarative specification for a canonical DiD analysis.

    Parameters mirror the panel columns: ``outcome`` is the numeric response,
    ``unit`` identifies panel members, ``time`` orders the periods, ``treated``
    marks units in the treatment group (constant within unit), and ``post``
    marks the post-onset periods (a pure time indicator, identical across
    units). ``cluster`` defaults to the unit column; pass a higher-level
    cluster (for example store or region) when errors are correlated within it.
    ``covariates`` are time-varying columns included in the within-unit
    regression; time-invariant covariates are absorbed by the unit effects.
    """

    outcome: str
    unit: str
    time: str
    treated: str
    post: str
    cluster: str | None = None
    covariates: tuple[str, ...] = ()
    alpha: float = 0.05

    def __post_init__(self) -> None:
        """Validate the declaration."""
        for field_name, value in (
            ("outcome", self.outcome),
            ("unit", self.unit),
            ("time", self.time),
            ("treated", self.treated),
            ("post", self.post),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.cluster is not None and (
            not isinstance(self.cluster, str) or not self.cluster.strip()
        ):
            raise ValueError("cluster must be a non-empty string or None")
        labels = {
            self.outcome,
            self.unit,
            self.time,
            self.treated,
            self.post,
            *((self.cluster,) if self.cluster else ()),
            *self.covariates,
        }
        if len(labels) != (
            5 + (1 if self.cluster else 0) + len(self.covariates)
        ):
            raise ValueError("column names must be unique")
        if any(
            not isinstance(covariate, str) or not covariate.strip()
            for covariate in self.covariates
        ):
            raise ValueError("covariates must be non-empty strings")
        if not isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

    def fit(self, data: pd.DataFrame) -> DidResult:
        """Estimate the canonical DiD model on a panel frame."""
        return _fit_difference_in_differences(self, data)


def _fit_difference_in_differences(
    spec: DifferenceInDifferences,
    data: pd.DataFrame,
) -> DidResult:
    """Run panel validation, the within-unit regression, and diagnostics."""
    _validate_columns(data, spec)
    periods = _ordered_periods(data[spec.time])
    period_index = _period_index(data[spec.time], periods)
    treated = _binary_column(data[spec.treated], spec.treated)
    post = _binary_column(data[spec.post], spec.post)
    outcome = pd.to_numeric(data[spec.outcome], errors="coerce").to_numpy(
        dtype=float
    )
    if np.isnan(outcome).any():
        raise ValueError(
            f"outcome column {spec.outcome!r} contains missing or "
            "non-numeric values"
        )
    covariate_columns = tuple(
        data[column].to_numpy(dtype=float) for column in spec.covariates
    )
    unit_ids = data[spec.unit].to_numpy()
    _validate_panel_structure(
        spec,
        data,
        periods,
        period_index,
        treated,
        post,
        unit_ids,
    )
    cluster_ids = (
        data[spec.cluster].to_numpy() if spec.cluster is not None else unit_ids
    )
    unique_clusters = np.unique(cluster_ids)
    if len(unique_clusters) < 3:
        raise ValueError(
            "cluster-robust inference requires at least 3 clusters, "
            f"got {len(unique_clusters)}"
        )

    warnings: list[str] = []
    post_by_period = {
        period: bool(np.all(post[period_index == index] == 1))
        for index, period in enumerate(periods)
    }
    pre_periods = tuple(
        period for period in periods if not post_by_period[period]
    )
    post_periods = tuple(
        period for period in periods if post_by_period[period]
    )
    if not pre_periods or not post_periods:
        raise ValueError(
            "the post indicator must mark at least one pre and one post period"
        )

    design, y_within, coefficient_map = _within_transformed_design(
        outcome,
        treated,
        post,
        unit_ids,
        covariate_columns,
    )
    beta, *_ = np.linalg.lstsq(design, y_within, rcond=None)
    residuals = y_within - design @ beta
    effect_index = coefficient_map["treated:post"]
    effect = float(beta[effect_index])
    standard_error, naive_standard_error, degrees_of_freedom = (
        _cluster_robust_standard_errors(
            design,
            residuals,
            cluster_ids,
            effect_index,
        )
    )
    if standard_error == 0.0:
        raise ValueError("DiD standard error is zero for the observed data")
    ci_level = 1.0 - spec.alpha
    critical_value = float(
        stats.t.ppf((1.0 + ci_level) / 2.0, degrees_of_freedom)
    )
    margin = critical_value * standard_error
    p_value = float(
        2.0 * stats.t.sf(abs(effect) / standard_error, degrees_of_freedom)
    )

    event_study = _event_study(
        spec,
        data,
        periods,
        period_index,
        treated,
        post,
        outcome,
        unit_ids,
        cluster_ids,
        ci_level,
    )
    group_time_means = _group_time_means(
        outcome,
        treated,
        post,
    )
    if len(pre_periods) == 1 and len(periods) == 2:
        warnings.append(
            "single pre period: parallel-trends placebo is not testable."
        )
    if spec.covariates:
        warnings.append(
            "covariates are time-varying and within-unit centered; "
            "time-invariant covariates are absorbed by unit fixed effects."
        )
    return DidResult(
        method="difference_in_differences",
        effect=effect,
        standard_error=standard_error,
        naive_standard_error=naive_standard_error,
        degrees_of_freedom=degrees_of_freedom,
        ci_lower=float(effect - margin),
        ci_upper=float(effect + margin),
        ci_level=ci_level,
        p_value=p_value,
        clusters=len(unique_clusters),
        units=len(np.unique(unit_ids)),
        periods=len(periods),
        pre_periods=len(pre_periods),
        post_periods=len(post_periods),
        covariates=spec.covariates,
        group_time_means=group_time_means,
        event_study=event_study,
        assumption_notes=_ASSUMPTION_NOTES,
        warnings=tuple(warnings),
        status="ok" if not warnings else "warning",
    )


def _validate_columns(
    data: pd.DataFrame,
    spec: DifferenceInDifferences,
) -> None:
    """Raise when declared columns are missing from the frame."""
    declared = [
        spec.outcome,
        spec.unit,
        spec.time,
        spec.treated,
        spec.post,
        *((spec.cluster,) if spec.cluster else ()),
        *spec.covariates,
    ]
    missing = [column for column in declared if column not in data.columns]
    if missing:
        raise ValueError(f"missing panel columns: {', '.join(missing)}")


def _ordered_periods(series: pd.Series) -> tuple[str, ...]:
    """Return unique period labels in chronological order."""
    unique = list(series.unique())
    try:
        parsed = pd.to_datetime(pd.Series(unique), errors="raise")
        order = np.argsort(parsed.to_numpy())
    except (ValueError, TypeError):
        order = np.argsort([str(value) for value in unique])
    return tuple(str(unique[index]) for index in order)


def _period_index(series: pd.Series, periods: tuple[str, ...]) -> np.ndarray:
    """Map each row to the integer rank of its period."""
    lookup = {period: index for index, period in enumerate(periods)}
    return np.asarray([lookup[str(value)] for value in series], dtype=int)


def _binary_column(series: pd.Series, name: str) -> np.ndarray:
    """Return a 0/1 indicator for a treated or post column."""
    values = series.to_numpy()
    unique = set(values)
    if unique <= {0, 1}:
        return np.asarray(
            [1 if bool(value) else 0 for value in values], dtype=float
        )
    raise ValueError(
        f"column {name!r} must be binary (0/1 or True/False), "
        f"found values {sorted(unique, key=str)[:6]}"
    )


def _validate_panel_structure(
    spec: DifferenceInDifferences,
    data: pd.DataFrame,
    periods: tuple[str, ...],
    period_index: np.ndarray,
    treated: np.ndarray,
    post: np.ndarray,
    unit_ids: np.ndarray,
) -> None:
    """Validate the canonical two-group, common-onset panel structure."""
    if len(np.unique(treated)) < 2:
        raise ValueError("the treated column must contain both groups")
    # Treatment status must be constant within a unit.
    frame = pd.DataFrame({"unit": unit_ids, "treated": treated, "post": post})
    per_unit = frame.groupby("unit", sort=False)["treated"].nunique()
    if int(per_unit.max()) != 1:
        raise ValueError(
            "treatment status changes within a unit; the canonical DiD "
            "design requires time-constant treatment assignment"
        )
    # post must be a pure time indicator: constant within each period, so
    # there is one global treatment onset and no staggered adoption.
    period_frame = pd.DataFrame({"period": period_index, "post": post})
    inconsistent = period_frame.groupby("period", sort=False)["post"].nunique()
    if int(inconsistent.max()) != 1:
        raise ValueError(
            "post varies within a period; unit-specific or staggered "
            "treatment timing is not supported by the canonical design"
        )
    # Every unit must be observed in at least one pre and one post period.
    unit_post_values = frame.groupby("unit", sort=False)["post"].agg(
        lambda values: set(int(value) for value in values)
    )
    for unit, observed in unit_post_values.items():
        if observed != {0, 1}:
            raise ValueError(
                f"unit {unit!r} is missing pre or post observations; the "
                "canonical DiD design requires every unit in both periods"
            )


def _within_transformed_design(
    outcome: np.ndarray,
    treated: np.ndarray,
    post: np.ndarray,
    unit_ids: np.ndarray,
    covariate_columns: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build the within-unit demeaned design for the 2x2 interaction.

    Demeaning each column at the unit level absorbs the unit fixed effects
    (and therefore the time-invariant treated main effect), leaving the
    ``post`` time effect and the ``treated:post`` interaction of interest.
    """

    def demean(values: np.ndarray) -> np.ndarray:
        unit_mean = np.asarray(
            pd.Series(values)
            .groupby(pd.Series(unit_ids), sort=False)
            .transform("mean"),
            dtype=float,
        )
        adjusted: np.ndarray = values - unit_mean
        return adjusted

    treated_post = treated * post
    columns: list[np.ndarray] = [demean(post), demean(treated_post)]
    names = ["post", "treated:post"]
    for covariate in covariate_columns:
        columns.append(demean(covariate))
        names.append("covariate")
    design = np.column_stack(columns)
    coefficient_map = {name: index for index, name in enumerate(names)}
    return design, demean(outcome), coefficient_map


def _group_time_means(
    outcome: np.ndarray,
    treated: np.ndarray,
    post: np.ndarray,
) -> tuple[GroupTimeMean, ...]:
    """Return raw outcome means for each treated-by-post cell."""
    means: list[GroupTimeMean] = []
    for treated_value in (0, 1):
        for post_value in (0, 1):
            mask = (treated == treated_value) & (post == post_value)
            values = outcome[mask]
            means.append(
                GroupTimeMean(
                    treated=bool(treated_value),
                    post=bool(post_value),
                    mean=float(np.mean(values)) if len(values) else None,
                    n=int(len(values)),
                )
            )
    return tuple(means)


def _cluster_robust_standard_errors(
    design: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
    effect_index: int,
) -> tuple[float, float, float]:
    """Return (robust SE, naive SE, degrees of freedom) for one coefficient."""
    n, coefficient_count = design.shape
    xtx_inverse = np.linalg.inv(design.T @ design)
    naive_variance = float(
        np.sum(residuals**2)
        / (n - coefficient_count)
        * xtx_inverse[effect_index, effect_index]
    )
    unique_clusters = np.unique(cluster_ids)
    cluster_count = len(unique_clusters)
    score_sums = np.zeros((cluster_count, coefficient_count))
    for index, cluster in enumerate(unique_clusters):
        rows = cluster_ids == cluster
        score_sums[index] = design[rows].T @ residuals[rows]
    sandwich = score_sums.T @ score_sums
    correction = (cluster_count / (cluster_count - 1)) * (
        (n - 1) / (n - coefficient_count)
    )
    robust_variance = (xtx_inverse @ sandwich @ xtx_inverse)[
        effect_index, effect_index
    ] * correction
    return (
        float(np.sqrt(max(robust_variance, 0.0))),
        float(np.sqrt(max(naive_variance, 0.0))),
        float(cluster_count - 2),
    )


def _event_study(
    spec: DifferenceInDifferences,
    data: pd.DataFrame,
    periods: tuple[str, ...],
    period_index: np.ndarray,
    treated: np.ndarray,
    post: np.ndarray,
    outcome: np.ndarray,
    unit_ids: np.ndarray,
    cluster_ids: np.ndarray,
    ci_level: float,
) -> EventStudyResult | None:
    """Estimate event-study coefficients with an omitted reference period."""
    pre_indices = [
        index
        for index, period in enumerate(periods)
        if not bool(np.all(post[period_index == index] == 1))
    ]
    if len(pre_indices) < 2:
        return None
    reference = pre_indices[-1]

    def demean(values: np.ndarray) -> np.ndarray:
        unit_mean = np.asarray(
            pd.Series(values)
            .groupby(pd.Series(unit_ids), sort=False)
            .transform("mean"),
            dtype=float,
        )
        adjusted: np.ndarray = values - unit_mean
        return adjusted

    treated_dummy = (treated == 1).astype(float)
    period_columns: list[np.ndarray] = []
    interaction_columns: list[np.ndarray] = []
    coefficient_periods: list[int] = []
    for index in range(len(periods)):
        if index == reference:
            continue
        period_dummy = (period_index == index).astype(float)
        period_columns.append(demean(period_dummy))
        interaction = demean(treated_dummy * period_dummy)
        interaction_columns.append(interaction)
        coefficient_periods.append(index)
    design = np.column_stack([*period_columns, *interaction_columns])
    y_within = demean(outcome)
    beta, *_ = np.linalg.lstsq(design, y_within, rcond=None)
    residuals = y_within - design @ beta
    n, coefficient_count = design.shape
    xtx_inverse = np.linalg.inv(design.T @ design)
    unique_clusters = np.unique(cluster_ids)
    cluster_count = len(unique_clusters)
    score_sums = np.zeros((cluster_count, coefficient_count))
    for index, cluster in enumerate(unique_clusters):
        rows = cluster_ids == cluster
        score_sums[index] = design[rows].T @ residuals[rows]
    sandwich = score_sums.T @ score_sums
    correction = (cluster_count / (cluster_count - 1)) * (
        (n - 1) / (n - coefficient_count)
    )
    covariance = xtx_inverse @ sandwich @ xtx_inverse * correction
    degrees_of_freedom = float(cluster_count - 2)
    critical_value = float(
        stats.t.ppf((1.0 + ci_level) / 2.0, degrees_of_freedom)
    )

    coefficients: list[EventStudyCoefficient] = []
    for offset, index in enumerate(coefficient_periods):
        coefficient = float(beta[len(period_columns) + offset])
        standard_error = float(
            np.sqrt(
                max(
                    covariance[
                        len(period_columns) + offset,
                        len(period_columns) + offset,
                    ],
                    0.0,
                )
            )
        )
        p_value = float(
            2.0
            * stats.t.sf(abs(coefficient) / standard_error, degrees_of_freedom)
        )
        margin = critical_value * standard_error
        coefficients.append(
            EventStudyCoefficient(
                period=periods[index],
                relative_time=index - reference,
                coefficient=coefficient,
                standard_error=standard_error,
                ci_lower=coefficient - margin,
                ci_upper=coefficient + margin,
                p_value=p_value,
            )
        )
    pre_coefficient_offsets = [
        offset
        for offset, index in enumerate(coefficient_periods)
        if index < reference
    ]
    pre_trend_p_value: float | None = None
    if pre_coefficient_offsets:
        pre_offsets = np.asarray(pre_coefficient_offsets, dtype=int)
        pre_beta = np.asarray(
            [coefficients[offset].coefficient for offset in pre_offsets],
            dtype=float,
        )
        block = covariance[
            np.ix_(
                len(period_columns) + pre_offsets,
                len(period_columns) + pre_offsets,
            )
        ]
        try:
            inverse = np.linalg.inv(block)
            statistic = float(pre_beta @ inverse @ pre_beta)
        except np.linalg.LinAlgError:
            statistic = float("nan")
        if np.isfinite(statistic):
            pre_trend_p_value = float(
                stats.chi2.sf(statistic, len(pre_offsets))
            )
    coefficients.append(
        EventStudyCoefficient(
            period=periods[reference],
            relative_time=0,
            coefficient=0.0,
            standard_error=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            p_value=float("nan"),
            reference=True,
        )
    )
    return EventStudyResult(
        reference_period=periods[reference],
        coefficients=tuple(coefficients),
        pre_trend_p_value=pre_trend_p_value,
        ci_level=ci_level,
    )


def render_did_markdown(result: DidResult) -> str:
    """Render a DiD result as a self-contained Markdown report."""
    lines = [
        "# Difference-in-Differences Estimate",
        "",
        f"- **Method**: {result.method}",
        f"- **Effect (treated x post)**: {result.effect:.6f}",
        f"- **Cluster-robust SE**: {result.standard_error:.6f} "
        f"(naive {result.naive_standard_error:.6f})",
        f"- **Degrees of freedom**: {result.degrees_of_freedom:.1f}",
        f"- **{result.ci_level * 100.0:.0f}% CI**: "
        f"[{result.ci_lower:.6f}, {result.ci_upper:.6f}]",
        f"- **p-value**: {result.p_value:.6g}",
        f"- **Clusters**: {result.clusters}",
        f"- **Units**: {result.units} across {result.periods} periods "
        f"({result.pre_periods} pre, {result.post_periods} post)",
        "",
        "## Raw group-time means",
        "",
        "| Group | Period | Mean | n |",
        "|---|---|---|---|",
    ]
    for mean in result.group_time_means:
        group = "treated" if mean.treated else "control"
        period = "post" if mean.post else "pre"
        value = f"{mean.mean:.6f}" if mean.mean is not None else "-"
        lines.append(f"| {group} | {period} | {value} | {mean.n} |")
    if result.event_study is not None:
        lines.extend(
            [
                "",
                "## Event study (reference period "
                f"{result.event_study.reference_period!r})",
                "",
                "| Period | Relative time | Coefficient | SE | p-value |",
                "|---|---|---|---|---|",
            ]
        )
        for coefficient in result.event_study.coefficients:
            if coefficient.reference:
                lines.append(
                    f"| {coefficient.period} | 0 | 0 (reference) | - | - |"
                )
                continue
            lines.append(
                f"| {coefficient.period} | {coefficient.relative_time:+d} | "
                f"{coefficient.coefficient:.6f} | "
                f"{coefficient.standard_error:.6f} | "
                f"{coefficient.p_value:.4f} |"
            )
        pre_trend = result.event_study.pre_trend_p_value
        if pre_trend is not None:
            lines.extend(
                [
                    "",
                    f"**Parallel-trends placebo p-value**: {pre_trend:.4f}",
                ]
            )
        else:
            lines.append("")
            lines.append("Pre-trend placebo not estimable for this panel.")
    lines.extend(
        [
            "",
            "## Identifying assumptions",
            "",
        ]
    )
    for note in result.assumption_notes:
        lines.append(f"- {note}")
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in result.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"
