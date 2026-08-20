"""Assignment and randomization diagnostics for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from twosample_means.data_io import DataValidationError

from .config import ExperimentConfig

AssignmentStatus = Literal["ok", "warning"]


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
    are excluded from the tested total and reported separately.
    """
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("experiment data must be a pandas DataFrame")
    required = {config.unit_id, config.assignment}
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
