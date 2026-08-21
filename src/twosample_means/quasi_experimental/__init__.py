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
- Staggered-adoption difference-in-differences in the style of Callaway &
  Sant'Anna (2021): group-time average treatment effects ATT(g, t) for
  every adoption cohort and period, not-yet-treated comparison units,
  optional anticipation windows, and group/calendar/event-time/overall
  aggregation with a joint cluster-robust covariance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

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


# ---------------------------------------------------------------------------
# Staggered-adoption difference-in-differences (Callaway & Sant'Anna, 2021)
# ---------------------------------------------------------------------------

_CS_ASSUMPTION_NOTES = (
    "Parallel trends conditional on group and time: in the absence of "
    "treatment, each adoption cohort would follow the same trend as the "
    "not-yet-treated comparison units.",
    "Staggered adoption with absorbing treatment: once a cohort is treated "
    "it stays treated; switching back to control is not allowed.",
    "No anticipation beyond the declared window: outcomes may respond only "
    "from the first treated period minus the anticipation parameter.",
    "Comparison group: at each period, the not-yet-treated units (later "
    "cohorts plus never-treated units).",
    "Balanced panel: every unit is observed in every period exactly once; "
    "unbalanced panels are not supported by this estimator.",
    "No interference: one unit's treatment must not affect another unit's "
    "outcome (SUTVA).",
)


@dataclass(frozen=True)
class _AttCell:
    """One estimable (cohort, period) cell with its estimation sample."""

    group: int
    period: int
    base: int
    treated: list[int]
    comparison: list[int]


@dataclass(frozen=True)
class GroupTimeAtt:
    """The ATT for one adoption cohort in one calendar period."""

    group: str
    period: str
    relative_time: int
    att: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_comparison: int


@dataclass(frozen=True)
class GroupAtt:
    """Group-level ATT averaged over the cohort's post periods."""

    group: str
    att: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    units: int
    post_periods: int


@dataclass(frozen=True)
class CalendarAtt:
    """Calendar-time ATT averaged over cohorts treated by that period."""

    period: str
    att: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    p_value: float


@dataclass(frozen=True)
class EventTimeAtt:
    """Event-time ATT for one distance from treatment onset."""

    relative_time: int
    att: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    p_value: float


@dataclass(frozen=True)
class OverallAtt:
    """Size-weighted ATT over all post group-time cells."""

    att: float
    standard_error: float
    naive_standard_error: float
    ci_lower: float
    ci_upper: float
    p_value: float


@dataclass(frozen=True)
class PlaceboTest:
    """Joint Wald test that clean pre-treatment cells are zero."""

    statistic: float
    degrees_of_freedom: int
    p_value: float
    cells: int


@dataclass(frozen=True)
class StaggeredDidResult:
    """Complete staggered-adoption difference-in-differences result."""

    method: str
    anticipation: int
    comparison: str
    units: int
    clusters: int
    periods: int
    ci_level: float
    group_labels: tuple[str, ...]
    period_labels: tuple[str, ...]
    group_time_atts: tuple[GroupTimeAtt, ...]
    group_atts: tuple[GroupAtt, ...]
    calendar_atts: tuple[CalendarAtt, ...]
    event_time_atts: tuple[EventTimeAtt, ...]
    overall_att: OverallAtt
    placebo: PlaceboTest | None
    assumption_notes: tuple[str, ...]
    warnings: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class CallawaySantAnna:
    """Declarative specification for staggered-adoption DiD.

    ``group`` names the column holding each unit's first-treated period
    (a label present in the ``time`` column). Never-treated units must hold
    the sentinel value ``"never"`` (or NaN/None). ``anticipation`` is the
    number of periods before treatment onset in which outcomes may already
    respond; those periods are treated as contaminated and excluded from the
    clean pre-treatment placebo test, and the effective treatment start of
    each cohort becomes ``group - anticipation``. ``cluster`` defaults to
    the unit column; pass a higher-level cluster when errors are correlated
    within it. The panel must be balanced: every unit observed in every
    period exactly once.
    """

    outcome: str
    unit: str
    time: str
    group: str
    anticipation: int = 0
    cluster: str | None = None
    alpha: float = 0.05

    def __post_init__(self) -> None:
        """Validate the declaration."""
        for field_name, value in (
            ("outcome", self.outcome),
            ("unit", self.unit),
            ("time", self.time),
            ("group", self.group),
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
            self.group,
            *((self.cluster,) if self.cluster else ()),
        }
        if len(labels) != 4 + (1 if self.cluster else 0):
            raise ValueError("column names must be unique")
        if isinstance(self.anticipation, bool) or not isinstance(
            self.anticipation, int
        ):
            raise ValueError(
                f"anticipation must be a non-negative integer, "
                f"got {self.anticipation!r}"
            )
        if self.anticipation < 0:
            raise ValueError(
                f"anticipation must be non-negative, got {self.anticipation}"
            )
        if not isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

    def fit(self, data: pd.DataFrame) -> StaggeredDidResult:
        """Estimate group-time ATTs and their aggregations."""
        return _fit_staggered_did(self, data)


def _fit_staggered_did(
    spec: CallawaySantAnna,
    data: pd.DataFrame,
) -> StaggeredDidResult:
    """Run panel validation and the joint group-time ATT estimation."""
    declared = [
        spec.outcome,
        spec.unit,
        spec.time,
        spec.group,
        *((spec.cluster,) if spec.cluster else ()),
    ]
    missing = [column for column in declared if column not in data.columns]
    if missing:
        raise ValueError(f"missing panel columns: {', '.join(missing)}")
    periods = _ordered_periods(data[spec.time])
    period_index = _period_index(data[spec.time], periods)
    outcome = pd.to_numeric(data[spec.outcome], errors="coerce").to_numpy(
        dtype=float
    )
    if np.isnan(outcome).any():
        raise ValueError(
            f"outcome column {spec.outcome!r} contains missing or "
            "non-numeric values"
        )
    unit_ids = data[spec.unit].to_numpy()
    _validate_balanced_panel(data, spec, periods)
    group_index = _parse_groups(data[spec.group], periods)
    cluster_ids = (
        data[spec.cluster].to_numpy() if spec.cluster is not None else unit_ids
    )
    _validate_unit_constants(unit_ids, group_index, cluster_ids)
    unique_clusters = np.unique(cluster_ids)
    if len(unique_clusters) < 3:
        raise ValueError(
            "cluster-robust inference requires at least 3 clusters, "
            f"got {len(unique_clusters)}"
        )

    unit_values, unit_inverse = np.unique(unit_ids, return_inverse=True)
    unit_count = len(unit_values)
    period_count = len(periods)
    anticipation = spec.anticipation
    unit_group = np.full(unit_count, -1, dtype=int)
    for unit in range(unit_count):
        unit_group[unit] = group_index[unit_inverse == unit][0]
    unit_cluster = np.full(unit_count, -1, dtype=int)
    cluster_lookup = {
        cluster: index for index, cluster in enumerate(unique_clusters)
    }
    for unit in range(unit_count):
        unit_cluster[unit] = cluster_lookup[
            cluster_ids[unit_inverse == unit][0]
        ]
    group_sizes = {
        int(group): int((unit_group == group).sum())
        for group in np.unique(unit_group)
    }

    outcome_matrix = np.full((unit_count, period_count), np.nan)
    for row in range(len(data)):
        outcome_matrix[unit_inverse[row], period_index[row]] = outcome[row]

    treated_groups = sorted(
        int(group) for group in np.unique(unit_group) if group >= 0
    )
    if not treated_groups:
        raise ValueError(
            "no treated cohorts found; the group column must mark each "
            "unit's first-treated period or 'never'"
        )
    usable_groups = [
        group for group in treated_groups if group - anticipation - 1 >= 0
    ]
    dropped_groups = [
        group for group in treated_groups if group - anticipation - 1 < 0
    ]
    warnings: list[str] = []
    for group in dropped_groups:
        warnings.append(
            f"cohort treated in period {periods[group]!r} has no clean "
            "pre-treatment base period after the anticipation window and "
            "was dropped; its effects are not identified."
        )
    if not usable_groups:
        raise ValueError(
            "no cohort has a clean pre-treatment base period; treat later "
            "or reduce the anticipation window"
        )

    cells: list[_AttCell] = []
    for group in usable_groups:
        base = group - anticipation - 1
        treated_units = [
            unit for unit in range(unit_count) if unit_group[unit] == group
        ]
        treated_set = set(treated_units)
        for period in range(period_count):
            if period == base:
                # The base period is the reference: the difference is zero
                # by construction and carries no identifying information.
                continue
            # Comparison units must be not-yet-treated at the current period
            # AND at the base period: a cohort whose effective treatment
            # starts inside (base, period] would contaminate its long
            # difference through the base-period outcome.
            effective_after = max(period, base) + anticipation
            comparison_units = [
                unit
                for unit in range(unit_count)
                if unit not in treated_set
                and (
                    unit_group[unit] == -1
                    or unit_group[unit] > effective_after
                )
            ]
            if len(treated_units) < 2 or len(comparison_units) < 2:
                continue
            cells.append(
                _AttCell(
                    group=group,
                    period=period,
                    base=base,
                    treated=treated_units,
                    comparison=comparison_units,
                )
            )
    if not cells:
        raise ValueError(
            "no estimable (cohort, period) cells; every cohort lacks either "
            "treated units or a not-yet-treated comparison group"
        )

    cell_count = len(cells)
    cluster_count = len(unique_clusters)
    # Each cell is a regression of the long difference (Y_t - Y_b) on a
    # treated-unit indicator with an intercept, so the ATT is the slope:
    # mean(treated) - mean(comparison). Stacking the cells gives a
    # block-diagonal design; the cluster-robust sandwich over the blocks
    # yields the joint covariance across all ATT(g, t) cells.
    beta = np.zeros(cell_count)
    counts = np.zeros(cell_count, dtype=int)
    treated_counts = np.zeros(cell_count, dtype=int)
    scores = np.zeros((cluster_count, cell_count, 2))
    residuals_squared_sum = 0.0
    for cell_index, cell in enumerate(cells):
        sample_units = [*cell.treated, *cell.comparison]
        y_diff = (
            outcome_matrix[sample_units, cell.period]
            - outcome_matrix[sample_units, cell.base]
        )
        treated_dummy = np.asarray(
            [1.0] * len(cell.treated) + [0.0] * len(cell.comparison),
            dtype=float,
        )
        design = np.column_stack([np.ones(len(sample_units)), treated_dummy])
        coefficient, *_ = np.linalg.lstsq(design, y_diff, rcond=None)
        beta[cell_index] = float(coefficient[1])
        counts[cell_index] = len(sample_units)
        treated_counts[cell_index] = len(cell.treated)
        residual = y_diff - design @ coefficient
        residuals_squared_sum += float(np.sum(residual**2))
        for offset, unit in enumerate(sample_units):
            scores[unit_cluster[unit], cell_index, 0] += residual[offset]
            scores[unit_cluster[unit], cell_index, 1] += (
                treated_dummy[offset] * residual[offset]
            )

    stacked_rows = int(counts.sum())
    xtx_inverse = np.zeros((cell_count, 2, 2))
    for cell_index in range(cell_count):
        block = np.asarray(
            [
                [counts[cell_index], treated_counts[cell_index]],
                [treated_counts[cell_index], treated_counts[cell_index]],
            ],
            dtype=float,
        )
        xtx_inverse[cell_index] = np.linalg.inv(block)
    correction = (cluster_count / (cluster_count - 1)) * (
        (stacked_rows - 1) / (stacked_rows - 2 * cell_count)
    )
    # Joint covariance of the ATT slopes: each cell contributes its
    # intercept row dropped via a_c = [0, 1] XtX_inv_c.
    covariance = np.zeros((cell_count, cell_count))
    for row_cell in range(cell_count):
        a_row = xtx_inverse[row_cell][1]
        for column_cell in range(cell_count):
            block_sum = np.zeros((2, 2))
            for cluster in range(cluster_count):
                block_sum += np.outer(
                    scores[cluster, row_cell], scores[cluster, column_cell]
                )
            covariance[row_cell, column_cell] = (
                a_row @ block_sum @ xtx_inverse[column_cell][1]
            )
    covariance *= correction
    sigma_squared = residuals_squared_sum / (stacked_rows - 2 * cell_count)
    naive_variance = sigma_squared * np.asarray(
        [xtx_inverse[cell_index][1, 1] for cell_index in range(cell_count)],
        dtype=float,
    )
    degrees_of_freedom = float(cluster_count - 2)
    ci_level = 1.0 - spec.alpha

    group_time_atts = tuple(
        GroupTimeAtt(
            group=periods[cell.group],
            period=periods[cell.period],
            relative_time=cell.period - cell.group,
            att=float(beta[index]),
            standard_error=float(np.sqrt(max(covariance[index, index], 0.0))),
            ci_lower=float(
                beta[index]
                - _t_critical(ci_level, degrees_of_freedom)
                * np.sqrt(max(covariance[index, index], 0.0))
            ),
            ci_upper=float(
                beta[index]
                + _t_critical(ci_level, degrees_of_freedom)
                * np.sqrt(max(covariance[index, index], 0.0))
            ),
            p_value=float(
                2.0
                * stats.t.sf(
                    abs(beta[index])
                    / np.sqrt(max(covariance[index, index], 0.0)),
                    degrees_of_freedom,
                )
            ),
            n_treated=len(cell.treated),
            n_comparison=len(cell.comparison),
        )
        for index, cell in enumerate(cells)
    )

    requested_cells = sum(
        period_count - 1 for _ in usable_groups
    )  # one reference (base) period per cohort is not estimable
    skipped_cells = requested_cells - cell_count
    if skipped_cells > 0:
        warnings.append(
            f"{skipped_cells} (cohort, period) cell(s) were skipped because "
            "no usable not-yet-treated comparison group exists."
        )
    if not any(cell.period >= cell.group for cell in cells):
        warnings.append(
            "no post-treatment cells were estimable; every reported ATT is "
            "a pre-treatment placebo."
        )

    group_atts = _group_aggregations(
        cells,
        beta,
        covariance,
        degrees_of_freedom,
        ci_level,
        group_sizes,
        periods,
    )
    calendar_atts = _calendar_aggregations(
        cells,
        beta,
        covariance,
        degrees_of_freedom,
        ci_level,
        group_sizes,
        periods,
    )
    event_time_atts = _event_time_aggregations(
        cells, beta, covariance, degrees_of_freedom, ci_level, group_sizes
    )
    overall = _overall_aggregation(
        cells,
        beta,
        covariance,
        naive_variance,
        degrees_of_freedom,
        ci_level,
        group_sizes,
    )
    placebo = _placebo_test(cells, beta, covariance, anticipation)
    if placebo is not None and placebo.p_value <= spec.alpha:
        warnings.append(
            "parallel-trends placebo test fails: clean pre-treatment cells "
            f"are jointly nonzero (p={placebo.p_value:.6g})."
        )
    status = "warning" if warnings else "ok"
    return StaggeredDidResult(
        method="callaway_santanna",
        anticipation=anticipation,
        comparison="not-yet-treated",
        units=unit_count,
        clusters=cluster_count,
        periods=period_count,
        ci_level=ci_level,
        group_labels=tuple(periods[group] for group in usable_groups),
        period_labels=periods,
        group_time_atts=group_time_atts,
        group_atts=group_atts,
        calendar_atts=calendar_atts,
        event_time_atts=event_time_atts,
        overall_att=overall,
        placebo=placebo,
        assumption_notes=_CS_ASSUMPTION_NOTES,
        warnings=tuple(warnings),
        status=status,
    )


def _validate_balanced_panel(
    data: pd.DataFrame,
    spec: CallawaySantAnna,
    periods: tuple[str, ...],
) -> None:
    """Raise unless every unit appears in every period exactly once."""
    counts = data.groupby([spec.unit, spec.time], sort=False).size()
    if int((counts != 1).sum()):
        raise ValueError(
            "balanced panel required: each unit must appear exactly once "
            "in every period"
        )
    per_unit = counts.groupby(level=0, sort=False).size()
    if int((per_unit != len(periods)).sum()):
        raise ValueError(
            "balanced panel required: every unit must be observed in every "
            "period"
        )


def _parse_groups(series: pd.Series, periods: tuple[str, ...]) -> np.ndarray:
    """Map group values to first-treated period indices (-1 for never)."""
    period_lookup = {label: index for index, label in enumerate(periods)}
    parsed: list[int] = []
    for value in series:
        if pd.isna(value) or str(value) == "never":
            parsed.append(-1)
            continue
        label = str(value)
        if label not in period_lookup:
            raise ValueError(
                f"group value {label!r} is not a period label or 'never'"
            )
        parsed.append(period_lookup[label])
    return np.asarray(parsed, dtype=int)


def _validate_unit_constants(
    unit_ids: np.ndarray,
    group_index: np.ndarray,
    cluster_ids: np.ndarray,
) -> None:
    """Raise when group or cluster status varies within a unit."""
    frame = pd.DataFrame(
        {"unit": unit_ids, "group": group_index, "cluster": cluster_ids}
    )
    for column in ("group", "cluster"):
        per_unit = frame.groupby("unit", sort=False)[column].nunique()
        if int(per_unit.max()) > 1:
            raise ValueError(
                f"{column} status changes within a unit; staggered DiD "
                "requires time-constant cohort and cluster assignment"
            )


def _t_critical(ci_level: float, degrees_of_freedom: float) -> float:
    """Two-sided t critical value for a confidence level."""
    return float(stats.t.ppf((1.0 + ci_level) / 2.0, degrees_of_freedom))


def _cell_se(covariance: np.ndarray, index: int) -> float:
    """Non-negative standard error for one cell."""
    return float(np.sqrt(max(covariance[index, index], 0.0)))


def _aggregation_weights(
    cells: list[_AttCell],
    group_sizes: dict[int, int],
    period_count: int,
) -> dict[str, Any]:
    """Return linear-combination weight vectors for each aggregation."""
    cell_count = len(cells)
    group_weights: dict[int, np.ndarray] = {
        group: np.zeros(cell_count) for group in group_sizes if group >= 0
    }
    calendar_weights = [np.zeros(cell_count) for _ in range(period_count)]
    event_weights: dict[int, np.ndarray] = {}
    overall = np.zeros(cell_count)
    for index, cell in enumerate(cells):
        group = cell.group
        period = cell.period
        relative = period - group
        size = float(group_sizes[group])
        if period >= group:
            group_weights[group][index] = 1.0
            if period < len(calendar_weights):
                calendar_weights[period][index] = size
            overall[index] = size
        event_weights.setdefault(relative, np.zeros(cell_count))[index] = size
    for group in group_weights:
        total = float(group_weights[group].sum())
        if total > 0.0:
            group_weights[group] /= total
    for period in range(period_count):
        total = float(calendar_weights[period].sum())
        if total > 0.0:
            calendar_weights[period] /= total
    for relative in event_weights:
        total = float(event_weights[relative].sum())
        if total > 0.0:
            event_weights[relative] /= total
    total = float(overall.sum())
    if total > 0.0:
        overall /= total
    return {
        "group": group_weights,
        "calendar": calendar_weights,
        "event": event_weights,
        "overall": overall,
    }


def _linear_combination(
    weights: np.ndarray,
    beta: np.ndarray,
    covariance: np.ndarray,
    degrees_of_freedom: float,
    ci_level: float,
) -> tuple[float, float, float, float, float]:
    """Return (att, se, ci_lower, ci_upper, p_value)."""
    att = float(weights @ beta)
    variance = float(weights @ covariance @ weights)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    critical = _t_critical(ci_level, degrees_of_freedom)
    margin = critical * standard_error
    p_value = float(
        2.0 * stats.t.sf(abs(att) / standard_error, degrees_of_freedom)
    )
    return att, standard_error, att - margin, att + margin, p_value


def _group_aggregations(
    cells: list[_AttCell],
    beta: np.ndarray,
    covariance: np.ndarray,
    degrees_of_freedom: float,
    ci_level: float,
    group_sizes: dict[int, int],
    periods: tuple[str, ...],
) -> tuple[GroupAtt, ...]:
    """Simple average over each cohort's post-treatment cells."""
    weights = _aggregation_weights(cells, group_sizes, len(periods))["group"]
    results: list[GroupAtt] = []
    for group, vector in weights.items():
        if float(vector.sum()) == 0.0:
            continue
        att, se, lower, upper, p_value = _linear_combination(
            vector, beta, covariance, degrees_of_freedom, ci_level
        )
        post_count = int((vector > 0).sum())
        results.append(
            GroupAtt(
                group=periods[group],
                att=att,
                standard_error=se,
                ci_lower=lower,
                ci_upper=upper,
                p_value=p_value,
                units=group_sizes[group],
                post_periods=post_count,
            )
        )
    return tuple(results)


def _calendar_aggregations(
    cells: list[_AttCell],
    beta: np.ndarray,
    covariance: np.ndarray,
    degrees_of_freedom: float,
    ci_level: float,
    group_sizes: dict[int, int],
    periods: tuple[str, ...],
) -> tuple[CalendarAtt, ...]:
    """Size-weighted ATT per calendar period over cohorts treated by then."""
    weights = _aggregation_weights(cells, group_sizes, len(periods))[
        "calendar"
    ]
    results: list[CalendarAtt] = []
    for period, vector in enumerate(weights):
        if float(vector.sum()) == 0.0:
            continue
        att, se, lower, upper, p_value = _linear_combination(
            vector, beta, covariance, degrees_of_freedom, ci_level
        )
        results.append(
            CalendarAtt(
                period=periods[period],
                att=att,
                standard_error=se,
                ci_lower=lower,
                ci_upper=upper,
                p_value=p_value,
            )
        )
    return tuple(results)


def _event_time_aggregations(
    cells: list[_AttCell],
    beta: np.ndarray,
    covariance: np.ndarray,
    degrees_of_freedom: float,
    ci_level: float,
    group_sizes: dict[int, int],
) -> tuple[EventTimeAtt, ...]:
    """Size-weighted ATT by distance from treatment onset."""
    weights = _aggregation_weights(cells, group_sizes, 0)["event"]
    results: list[EventTimeAtt] = []
    for relative, vector in sorted(weights.items()):
        if float(vector.sum()) == 0.0:
            continue
        att, se, lower, upper, p_value = _linear_combination(
            vector, beta, covariance, degrees_of_freedom, ci_level
        )
        results.append(
            EventTimeAtt(
                relative_time=relative,
                att=att,
                standard_error=se,
                ci_lower=lower,
                ci_upper=upper,
                p_value=p_value,
            )
        )
    return tuple(results)


def _overall_aggregation(
    cells: list[_AttCell],
    beta: np.ndarray,
    covariance: np.ndarray,
    naive_variance: np.ndarray,
    degrees_of_freedom: float,
    ci_level: float,
    group_sizes: dict[int, int],
) -> OverallAtt:
    """Size-weighted ATT over all post cells with robust and naive SEs."""
    weights = _aggregation_weights(cells, group_sizes, 0)["overall"]
    att, se, lower, upper, p_value = _linear_combination(
        weights, beta, covariance, degrees_of_freedom, ci_level
    )
    # naive_variance is the per-cell (diagonal) OLS variance; the aggregate
    # naive variance is the size-weighted sum of the squared weights.
    naive_variance_total = float(np.sum(weights**2 * naive_variance))
    naive_se = float(np.sqrt(max(naive_variance_total, 0.0)))
    return OverallAtt(
        att=att,
        standard_error=se,
        naive_standard_error=naive_se,
        ci_lower=lower,
        ci_upper=upper,
        p_value=p_value,
    )


def _placebo_test(
    cells: list[_AttCell],
    beta: np.ndarray,
    covariance: np.ndarray,
    anticipation: int,
) -> PlaceboTest | None:
    """Joint Wald test that clean pre-treatment cells are zero."""
    pre_indices = [
        index
        for index, cell in enumerate(cells)
        if cell.period < cell.group - anticipation
    ]
    if not pre_indices:
        return None
    pre_beta = np.asarray([beta[index] for index in pre_indices], dtype=float)
    block = covariance[np.ix_(pre_indices, pre_indices)]
    try:
        inverse = np.linalg.inv(block)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(block)
    statistic = float(pre_beta @ inverse @ pre_beta)
    if not np.isfinite(statistic):
        return None
    return PlaceboTest(
        statistic=statistic,
        degrees_of_freedom=len(pre_indices),
        p_value=float(stats.chi2.sf(statistic, len(pre_indices))),
        cells=len(pre_indices),
    )


def render_staggered_did_markdown(result: StaggeredDidResult) -> str:
    """Render a staggered-adoption DiD result as Markdown."""
    lines = [
        "# Staggered-Adoption Difference-in-Differences "
        "(Callaway & Sant'Anna)",
        "",
        f"- **Method**: {result.method}",
        f"- **Anticipation**: {result.anticipation} period(s)",
        f"- **Comparison group**: {result.comparison}",
        f"- **Units**: {result.units} across {result.periods} periods; "
        f"**clusters**: {result.clusters}",
        f"- **{result.ci_level * 100.0:.0f}% level across group-time ATTs",
        "",
        "## Group-time ATT(g, t)",
        "",
        "| Group | Period | Relative time | ATT | SE | CI | p-value |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in result.group_time_atts:
        lines.append(
            f"| {cell.group} | {cell.period} | {cell.relative_time:+d} | "
            f"{cell.att:.4f} | {cell.standard_error:.4f} | "
            f"[{cell.ci_lower:.4f}, {cell.ci_upper:.4f}] | "
            f"{cell.p_value:.4f} |"
        )
    lines.extend(
        [
            "",
            "## ATT by adoption group",
            "",
            "| Group | ATT | SE | CI | p-value | units | post |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for group in result.group_atts:
        lines.append(
            f"| {group.group} | {group.att:.4f} | {group.standard_error:.4f} "
            f"| [{group.ci_lower:.4f}, {group.ci_upper:.4f}] | "
            f"{group.p_value:.4f} | {group.units} | {group.post_periods} |"
        )
    lines.extend(
        [
            "",
            "## Calendar-time ATT",
            "",
            "| Period | ATT | SE | CI | p-value |",
            "|---|---|---|---|---|",
        ]
    )
    for period in result.calendar_atts:
        lines.append(
            f"| {period.period} | {period.att:.4f} | "
            f"{period.standard_error:.4f} | "
            f"[{period.ci_lower:.4f}, {period.ci_upper:.4f}] | "
            f"{period.p_value:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Event-time ATT (relative to treatment)",
            "",
            "| Time | ATT | SE | CI | p-value |",
            "|---|---|---|---|---|",
        ]
    )
    for event in result.event_time_atts:
        lines.append(
            f"| {event.relative_time:+d} | {event.att:.4f} | "
            f"{event.standard_error:.4f} | "
            f"[{event.ci_lower:.4f}, {event.ci_upper:.4f}] | "
            f"{event.p_value:.4f} |"
        )
    overall = result.overall_att
    lines.extend(
        [
            "",
            "## Overall ATT",
            "",
            f"- **ATT**: {overall.att:.4f} "
            f"(SE {overall.standard_error:.4f}, "
            f"naive {overall.naive_standard_error:.4f})",
            f"- **{result.ci_level * 100.0:.0f}% CI**: "
            f"[{overall.ci_lower:.4f}, {overall.ci_upper:.4f}]",
            f"- **p-value**: {overall.p_value:.4f}",
        ]
    )
    if result.placebo is not None:
        lines.extend(
            [
                "",
                "## Parallel-trends placebo (pre-treatment cells)",
                "",
                f"- Joint chi-square {result.placebo.statistic:.4f} on "
                f"{result.placebo.degrees_of_freedom} df "
                f"({result.placebo.cells} cells); "
                f"p = {result.placebo.p_value:.4f}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Parallel-trends placebo",
                "",
                "- Not estimable: no clean pre-treatment cells.",
            ]
        )
    lines.extend(["", "## Identifying assumptions", ""])
    for note in result.assumption_notes:
        lines.append(f"- {note}")
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in result.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"
