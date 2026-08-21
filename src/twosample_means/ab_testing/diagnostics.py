"""Assignment and randomization diagnostics for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from twosample_means.data_io import DataValidationError

from .config import ExperimentConfig

AssignmentStatus = Literal["ok", "warning"]

# Standardized-mean-difference threshold used to flag pre-treatment
# covariate imbalance (Austin, 2011 recommends |SMD| < 0.1 for
# well-balanced arms).
BALANCE_SMD_THRESHOLD = 0.1


@dataclass(frozen=True)
class CovariateBalance:
    """Standardized mean difference for one covariate in one arm.

    The SMD is computed at the randomization-unit level against the
    control arm: ``(mean_arm - mean_control) / pooled_sd`` with the
    pooled standard deviation estimated from the two arms. ``smd`` is
    ``None`` when the covariate is constant within an arm, non-numeric,
    or has too few usable units.
    """

    covariate: str
    arm: str
    smd: float | None
    control_mean: float | None
    arm_mean: float | None
    pooled_sd: float | None
    exceeds_threshold: bool
    warning: str | None = None


@dataclass(frozen=True)
class StratumSrm:
    """Sample-ratio mismatch test within one stratum."""

    stratum: str | None
    n: int
    p_value: float | None
    observed: dict[str, int]
    expected: dict[str, float]


@dataclass(frozen=True)
class AssignmentDiagnostics:
    """Quality diagnostics for an experiment assignment column."""

    status: AssignmentStatus
    assignment_counts: dict[str, int]
    missing_assignment: int
    unknown_assignment: int
    missing_unit: int
    duplicate_units: int
    multi_arm_units: int
    sample_ratio_mismatch_p_value: float | None
    sample_ratio_mismatch_evaluated: bool
    expected_allocation: dict[str, float] | None
    covariate_balance: tuple[CovariateBalance, ...] = ()
    stratum_srm: tuple[StratumSrm, ...] = ()
    warnings: tuple[str, ...] = ()


def diagnose_assignment(
    data: pd.DataFrame,
    config: ExperimentConfig,
) -> AssignmentDiagnostics:
    """Diagnose assignment integrity and optional sample-ratio mismatch.

    The function intentionally accepts raw data rather than
    ``NormalizedExperimentData``. This allows callers to report duplicate,
    missing, and unknown assignment problems before strict normalization
    rejects them.

    Sample-ratio mismatch uses a multinomial chi-square goodness-of-fit test
    against ``config.expected_allocation``. Missing and unknown assignments
    are excluded from the tested total and reported separately. When
    ``config.strata`` is declared, the mismatch test also runs within each
    stratum because a balanced marginal allocation can hide offsetting
    per-stratum imbalances.

    Pre-treatment covariate balance is reported as a standardized mean
    difference (SMD) for every declared metric covariate and every
    ``config.balance_columns`` entry against each treatment arm, using the
    |SMD| < 0.1 heuristic to flag imbalance. ``balance_columns`` declares
    columns that are checked for balance without being used for variance
    reduction.
    """
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("experiment data must be a pandas DataFrame")
    required = {config.unit_id, config.assignment}
    if config.strata is not None:
        required.add(config.strata)
    for metric in config.metrics:
        if metric.covariate is not None:
            required.add(metric.covariate)
    required.update(config.balance_columns)
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise DataValidationError(
            f"experiment data is missing required columns: {missing_columns}"
        )

    unit_series = data[config.unit_id]
    assignment_series = data[config.assignment]
    valid_labels = set(config.arms)
    missing_assignment_mask = assignment_series.isna()
    unknown_assignment_mask = (
        ~missing_assignment_mask & ~assignment_series.isin(valid_labels)
    )
    assignment_counts = {
        label: int((assignment_series == label).sum()) for label in config.arms
    }
    missing_unit = int(unit_series.isna().sum())
    duplicate_units = _duplicate_unit_count(unit_series)
    multi_arm_units = _multi_arm_unit_count(unit_series, assignment_series)

    warnings: list[str] = []
    if config.unit_type == "aggregate":
        warnings.append(
            "Data is declared aggregate-level; assignment diagnostics and "
            "standard errors describe aggregate rows, not individual users."
        )
    elif config.unit_type == "unknown":
        warnings.append(
            "Unit type is unknown. Confirm that each row represents the "
            "declared randomization unit before interpreting causal effects."
        )
    if missing_unit:
        warnings.append(f"{missing_unit} row(s) have missing unit IDs.")
    if duplicate_units:
        warnings.append(f"{duplicate_units} unit(s) appear in multiple rows.")
    if multi_arm_units:
        warnings.append(
            f"{multi_arm_units} unit(s) appear in multiple assignment arms."
        )
    missing_assignment = int(missing_assignment_mask.sum())
    if missing_assignment:
        warnings.append(
            f"{missing_assignment} row(s) have missing assignments."
        )
    unknown_assignment = int(unknown_assignment_mask.sum())
    if unknown_assignment:
        warnings.append(
            f"{unknown_assignment} row(s) have unknown assignments."
        )
    covariate_balance = _covariate_balance(data, config)
    for balance_entry in covariate_balance:
        if balance_entry.warning is not None:
            warnings.append(f"Covariate balance: {balance_entry.warning}")
        elif balance_entry.exceeds_threshold:
            warnings.append(
                f"Covariate imbalance: {balance_entry.covariate} vs arm "
                f"'{balance_entry.arm}' has "
                f"|SMD|={abs(balance_entry.smd or 0.0):.4f} > "
                f"{BALANCE_SMD_THRESHOLD}."
            )

    stratum_srm = _stratum_srm(data, config)
    for stratum_entry in stratum_srm:
        if (
            stratum_entry.p_value is not None
            and stratum_entry.p_value <= config.alpha
        ):
            warnings.append(
                "Sample-ratio mismatch within stratum "
                f"'{stratum_entry.stratum}': "
                f"p={stratum_entry.p_value:.6g} <= "
                f"alpha={config.alpha:.6g}."
            )

    srm_assignments = _unit_level_assignments(
        unit_series, assignment_series, valid_labels
    )
    srm_p_value = _sample_ratio_mismatch(srm_assignments, config)
    if srm_p_value is not None and srm_p_value <= config.alpha:
        warnings.append(
            "Sample-ratio mismatch detected: "
            f"p={srm_p_value:.6g} <= alpha={config.alpha:.6g}."
        )
    if config.expected_allocation is None:
        warnings.append(
            "Sample-ratio mismatch was not evaluated because expected "
            "allocation is not configured."
        )
    if (
        config.strata is not None
        and config.expected_allocation is not None
        and not stratum_srm
    ):
        warnings.append(
            "No usable units to evaluate sample-ratio mismatch within strata."
        )

    return AssignmentDiagnostics(
        status="warning" if warnings and _has_data_warning(warnings) else "ok",
        assignment_counts=assignment_counts,
        missing_assignment=missing_assignment,
        unknown_assignment=unknown_assignment,
        missing_unit=missing_unit,
        duplicate_units=duplicate_units,
        multi_arm_units=multi_arm_units,
        sample_ratio_mismatch_p_value=srm_p_value,
        sample_ratio_mismatch_evaluated=config.expected_allocation is not None,
        expected_allocation=config.expected_allocation,
        covariate_balance=covariate_balance,
        stratum_srm=stratum_srm,
        warnings=tuple(warnings),
    )


def _duplicate_unit_count(unit_series: pd.Series) -> int:
    """Count distinct non-missing units represented by multiple rows."""
    non_missing = unit_series.dropna()
    duplicated = non_missing[non_missing.duplicated(keep=False)]
    return int(duplicated.nunique())


def _multi_arm_unit_count(
    unit_series: pd.Series,
    assignment_series: pd.Series,
) -> int:
    """Count units assigned to more than one distinct arm."""
    frame = pd.DataFrame(
        {"unit": unit_series, "assignment": assignment_series}
    )
    frame = frame.dropna(subset=["unit", "assignment"])
    arm_counts = frame.groupby("unit", sort=False)["assignment"].nunique()
    return int((arm_counts > 1).sum())


def _unit_level_assignments(
    unit_series: pd.Series,
    assignment_series: pd.Series,
    valid_labels: set[str],
) -> pd.Series:
    """Return one unambiguous assignment per usable randomization unit."""
    frame = pd.DataFrame(
        {"unit": unit_series, "assignment": assignment_series}
    )
    frame = frame.dropna(subset=["unit", "assignment"])
    frame = frame[frame["assignment"].isin(valid_labels)]
    arm_counts = frame.groupby("unit", sort=False)["assignment"].nunique()
    ambiguous_units = set(arm_counts[arm_counts > 1].index)
    frame = frame[~frame["unit"].isin(ambiguous_units)]
    return frame.drop_duplicates("unit")["assignment"]


def _unit_level_frame(
    data: pd.DataFrame,
    config: ExperimentConfig,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return one row per unambiguous unit with assignment and extras.

    Rows with missing units or assignments are dropped, undeclared labels
    are excluded, units that appear under more than one arm are removed as
    ambiguous, and the first row per unit is kept. Extra column values
    (strata, covariates) are carried from the first surviving row of each
    unit.
    """
    valid_labels = set(config.arms)
    frame = pd.DataFrame(
        {
            "unit": data[config.unit_id],
            "assignment": data[config.assignment],
        }
    )
    for column in extra_columns:
        frame[column] = data[column]
    frame = frame.dropna(subset=["unit", "assignment"])
    frame = frame[frame["assignment"].isin(valid_labels)]
    arm_counts = frame.groupby("unit", sort=False)["assignment"].nunique()
    ambiguous_units = set(arm_counts[arm_counts > 1].index)
    frame = frame[~frame["unit"].isin(ambiguous_units)]
    return frame.drop_duplicates("unit", keep="first")


def _covariate_balance(
    data: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[CovariateBalance, ...]:
    """Compute unit-level SMD balance for every declared covariate.

    The checked set is the union of metric covariate columns and
    ``config.balance_columns``. Metric covariates are listed first so the
    report's balance rows stay stable when balance-only columns are added.
    """
    covariates: list[str] = []
    for metric in config.metrics:
        if metric.covariate is not None and metric.covariate not in covariates:
            covariates.append(metric.covariate)
    for column in config.balance_columns:
        if column not in covariates:
            covariates.append(column)
    if not covariates:
        return ()

    frame = _unit_level_frame(data, config, extra_columns=tuple(covariates))
    entries: list[CovariateBalance] = []
    for covariate in covariates:
        values = frame[covariate]
        if not pd.api.types.is_numeric_dtype(values):
            entries.append(
                CovariateBalance(
                    covariate=covariate,
                    arm=config.treatments[0],
                    smd=None,
                    control_mean=None,
                    arm_mean=None,
                    pooled_sd=None,
                    exceeds_threshold=False,
                    warning=(f"covariate column {covariate!r} is not numeric"),
                )
            )
            continue
        for arm in config.treatments:
            control_values = (
                values[frame["assignment"] == config.control]
                .dropna()
                .to_numpy(dtype=float)
            )
            arm_values = (
                values[frame["assignment"] == arm]
                .dropna()
                .to_numpy(dtype=float)
            )
            if len(control_values) < 2 or len(arm_values) < 2:
                entries.append(
                    CovariateBalance(
                        covariate=covariate,
                        arm=arm,
                        smd=None,
                        control_mean=None,
                        arm_mean=None,
                        pooled_sd=None,
                        exceeds_threshold=False,
                        warning=(
                            f"too few units to estimate SMD for "
                            f"covariate {covariate!r} in arm {arm!r}"
                        ),
                    )
                )
                continue
            n_control = len(control_values)
            n_arm = len(arm_values)
            control_mean = float(control_values.mean())
            arm_mean = float(arm_values.mean())
            control_var = float(control_values.var(ddof=1))
            arm_var = float(arm_values.var(ddof=1))
            pooled_sd = sqrt(
                ((n_control - 1) * control_var + (n_arm - 1) * arm_var)
                / (n_control + n_arm - 2)
            )
            if pooled_sd == 0.0:
                entries.append(
                    CovariateBalance(
                        covariate=covariate,
                        arm=arm,
                        smd=None,
                        control_mean=control_mean,
                        arm_mean=arm_mean,
                        pooled_sd=0.0,
                        exceeds_threshold=False,
                        warning=(
                            f"covariate {covariate!r} is constant within "
                            f"arm {arm!r}"
                        ),
                    )
                )
                continue
            smd = (arm_mean - control_mean) / pooled_sd
            entries.append(
                CovariateBalance(
                    covariate=covariate,
                    arm=arm,
                    smd=smd,
                    control_mean=control_mean,
                    arm_mean=arm_mean,
                    pooled_sd=pooled_sd,
                    exceeds_threshold=abs(smd) > BALANCE_SMD_THRESHOLD,
                )
            )
    return tuple(entries)


def _stratum_srm(
    data: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[StratumSrm, ...]:
    """Run the sample-ratio mismatch test within each declared stratum."""
    if config.strata is None:
        return ()
    frame = _unit_level_frame(data, config, extra_columns=(config.strata,))
    results: list[StratumSrm] = []
    strata_values = frame[config.strata].dropna()
    for stratum in sorted(strata_values.unique()):
        stratum_assignments = frame.loc[
            frame[config.strata] == stratum, "assignment"
        ]
        observed = {
            label: int((stratum_assignments == label).sum())
            for label in config.arms
        }
        expected = (
            {label: config.expected_allocation[label] for label in config.arms}
            if config.expected_allocation is not None
            else {}
        )
        p_value = _sample_ratio_mismatch(stratum_assignments, config)
        results.append(
            StratumSrm(
                stratum=stratum,
                n=int(len(stratum_assignments)),
                p_value=p_value,
                observed=observed,
                expected=expected,
            )
        )
    return tuple(results)


def _sample_ratio_mismatch(
    valid_assignments: pd.Series,
    config: ExperimentConfig,
) -> float | None:
    """Calculate the multinomial sample-ratio mismatch p-value."""
    if config.expected_allocation is None or valid_assignments.empty:
        return None
    observed = np.asarray(
        [int((valid_assignments == label).sum()) for label in config.arms],
        dtype=float,
    )
    total = float(observed.sum())
    expected = np.asarray(
        [config.expected_allocation[label] for label in config.arms],
        dtype=float,
    )
    expected = expected / expected.sum() * total
    if np.any(expected <= 0.0):
        return None
    result = stats.chisquare(observed, f_exp=expected)
    return float(result.pvalue)


def _has_data_warning(warnings: list[str]) -> bool:
    """Return whether warnings indicate a data-quality issue."""
    return any(
        not warning.startswith("Sample-ratio mismatch was not evaluated")
        for warning in warnings
    )
