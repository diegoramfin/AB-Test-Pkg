"""Normalization and validation for experiment-level input data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.data_io import DataValidationError

from .config import ExperimentConfig, MetricSpec


def load_separate_experiment_csvs(
    control_path: str | Path,
    treatment_path: str | Path,
    config: ExperimentConfig,
    *,
    delimiter: str = ",",
) -> pd.DataFrame:
    """Load separate control and treatment CSVs into the experiment contract.

    Each file must contain the configured unit and metric input columns. The
    assignment column is synthesized from the file role, so separate-arm
    files do not need to carry an assignment label. If they do, existing
    labels must agree with the declared arm rather than being silently
    overwritten.
    """
    if len(config.treatments) != 1:
        raise DataValidationError(
            "separate control/treatment CSV ingestion requires exactly one "
            "treatment arm"
        )
    control = _read_arm_csv(control_path, config, config.control, delimiter)
    treatment = _read_arm_csv(
        treatment_path,
        config,
        config.treatments[0],
        delimiter,
    )
    return pd.concat((control, treatment), ignore_index=True, sort=False)


def _read_arm_csv(
    path: str | Path,
    config: ExperimentConfig,
    arm_label: str,
    delimiter: str,
) -> pd.DataFrame:
    """Read one separate-arm CSV and attach its declared assignment label."""
    frame = pd.read_csv(path, sep=delimiter)
    required = {config.unit_id}
    for metric in config.metrics:
        required.update(_metric_input_columns(metric))
    if config.time_column is not None:
        required.add(config.time_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(
            f"{path}: missing required columns: {missing}"
        )
    if config.assignment in frame.columns:
        observed = frame[config.assignment].dropna()
        if not observed.eq(arm_label).all():
            raise DataValidationError(
                f"{path}: existing assignment values do not match "
                f"declared arm {arm_label!r}"
            )
    frame[config.assignment] = arm_label
    return frame


@dataclass(frozen=True)
class NormalizedExperimentData:
    """Validated experiment data consumed by metric estimators.

    The frame retains one row per analysis unit and keeps missing metric
    values as ``NaN`` so each metric can apply its declared missing policy
    without changing the denominator of other metrics.
    """

    frame: pd.DataFrame
    unit_id: str
    assignment: str
    metric_names: tuple[str, ...]
    data_hash: str
    source_rows: int
    analysis_rows: int
    excluded_rows: int
    missing_outcomes: dict[str, int]
    missing_covariates: dict[str, int]

    @property
    def arm_counts(self) -> dict[str, int]:
        """Return assignment counts in the normalized frame."""
        counts = self.frame[self.assignment].value_counts(sort=False)
        return {str(label): int(count) for label, count in counts.items()}


def normalize_experiment_data(
    data: pd.DataFrame,
    config: ExperimentConfig,
) -> NormalizedExperimentData:
    """Validate and normalize a raw experiment DataFrame.

    Time-window filtering is applied before unit and outcome validation. This
    permits duplicate event rows outside the declared analysis window while
    still requiring one row per unit inside the analysis frame.
    """
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("experiment data must be a pandas DataFrame")
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise DataValidationError(
            f"experiment data contains duplicate columns: {duplicates}"
        )

    required = {config.unit_id, config.assignment}
    for metric in config.metrics:
        required.update(_metric_input_columns(metric))
    if config.time_column is not None:
        required.add(config.time_column)
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise DataValidationError(
            f"experiment data is missing required columns: {missing_columns}"
        )

    source_rows = len(data)
    frame = data.copy(deep=True)
    if config.time_column is not None and (
        config.analysis_start is not None or config.analysis_end is not None
    ):
        frame = _apply_time_window(frame, config)
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise DataValidationError("analysis window contains no rows")

    _validate_units_and_assignment(frame, config)
    missing_outcomes: dict[str, int] = {}
    missing_covariates: dict[str, int] = {}
    for metric in config.metrics:
        outcome_missing, covariate_missing = _normalize_metric(frame, metric)
        missing_outcomes[metric.name] = outcome_missing
        missing_covariates[metric.name] = covariate_missing

    return NormalizedExperimentData(
        frame=frame,
        unit_id=config.unit_id,
        assignment=config.assignment,
        metric_names=tuple(metric.name for metric in config.metrics),
        data_hash=_hash_frame(frame),
        source_rows=source_rows,
        analysis_rows=len(frame),
        excluded_rows=source_rows - len(frame),
        missing_outcomes=missing_outcomes,
        missing_covariates=missing_covariates,
    )


def _apply_time_window(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Apply an inclusive UTC-normalized analysis window."""
    if config.time_column is None:
        raise DataValidationError(
            "time_column is required to apply an analysis window"
        )
    timestamps = pd.to_datetime(
        frame[config.time_column], errors="coerce", utc=True
    )
    if timestamps.isna().any():
        invalid = int(timestamps.isna().sum())
        raise DataValidationError(
            f"time column '{config.time_column}' contains "
            f"{invalid} invalid value(s)"
        )
    try:
        start = _utc_timestamp(config.analysis_start)
        end = _utc_timestamp(config.analysis_end)
    except (TypeError, ValueError) as error:
        raise DataValidationError(
            "analysis_start and analysis_end must be valid timestamps"
        ) from error
    if start is not None and end is not None and start > end:
        raise DataValidationError(
            "analysis_start must not be after analysis_end"
        )
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= timestamps >= start
    if end is not None:
        mask &= timestamps <= end
    return frame.loc[mask].copy()


def _utc_timestamp(value: str | None) -> pd.Timestamp | None:
    """Convert an optional timestamp to UTC."""
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _validate_units_and_assignment(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """Validate the randomization unit and assignment labels."""
    unit_series = frame[config.unit_id]
    if unit_series.isna().any():
        raise DataValidationError(
            f"unit column '{config.unit_id}' contains missing values"
        )
    duplicate_units = unit_series.duplicated(keep=False)
    if duplicate_units.any():
        count = int(duplicate_units.sum())
        raise DataValidationError(
            f"unit column '{config.unit_id}' contains {count} duplicate rows"
        )

    assignment_series = frame[config.assignment]
    if assignment_series.isna().any():
        raise DataValidationError(
            f"assignment column '{config.assignment}' contains missing values"
        )
    valid_labels = set(config.arms)
    unknown = ~assignment_series.isin(valid_labels)
    if unknown.any():
        labels = assignment_series.loc[unknown].drop_duplicates().tolist()
        raise DataValidationError(
            f"assignment column '{config.assignment}' contains "
            "unknown labels: "
            f"{labels}; expected {sorted(valid_labels)}"
        )


def _normalize_metric(
    frame: pd.DataFrame,
    metric: MetricSpec,
) -> tuple[int, int]:
    """Validate and canonicalize one metric's input columns in place."""
    covariate_missing = _normalize_covariate(frame, metric)
    if metric.kind == "ratio":
        assert metric.numerator is not None
        assert metric.denominator is not None
        numerator = _numeric_series(frame, metric.numerator, metric.name)
        denominator = _numeric_series(frame, metric.denominator, metric.name)
        missing_mask = numerator.isna() | denominator.isna()
        missing = int(missing_mask.sum())
        if missing and metric.missing == "error":
            raise DataValidationError(
                f"metric '{metric.name}' contains {missing} missing value(s)"
            )
        invalid_denominator = denominator.notna() & (denominator <= 0.0)
        if invalid_denominator.any():
            raise DataValidationError(
                f"ratio metric '{metric.name}' requires positive "
                f"denominator values in '{metric.denominator}'"
            )
        frame[metric.numerator] = numerator.astype(float)
        frame[metric.denominator] = denominator.astype(float)
        return missing, covariate_missing

    series = _numeric_series(frame, metric.column, metric.name)
    missing = int(series.isna().sum())
    if missing and metric.missing == "error":
        raise DataValidationError(
            f"metric '{metric.name}' contains {missing} missing value(s)"
        )
    if metric.kind == "binary":
        invalid_binary = series.notna() & ~series.isin((0, 1))
        if invalid_binary.any():
            values = series.loc[invalid_binary].drop_duplicates().tolist()
            raise DataValidationError(
                f"binary metric '{metric.name}' contains values outside 0/1: "
                f"{values}"
            )
        success = int(metric.success_value)
        converted_values = series.to_numpy(dtype=float)
        canonical_values = np.where(
            np.isnan(converted_values),
            np.nan,
            (converted_values == success).astype(float),
        )
        frame[metric.column] = pd.Series(
            canonical_values,
            index=frame.index,
            dtype="float64",
        )
    else:
        frame[metric.column] = series.astype(float)
    return missing, covariate_missing


def _normalize_covariate(
    frame: pd.DataFrame,
    metric: MetricSpec,
) -> int:
    """Validate and canonicalize an optional CUPED covariate column."""
    if metric.covariate is None:
        return 0
    series = _numeric_series(frame, metric.covariate, metric.name)
    missing = int(series.isna().sum())
    if missing and metric.missing == "error":
        raise DataValidationError(
            f"metric '{metric.name}' covariate '{metric.covariate}' "
            f"contains {missing} missing value(s)"
        )
    frame[metric.covariate] = series.astype(float)
    return missing


def _metric_input_columns(metric: MetricSpec) -> tuple[str, ...]:
    """Return raw columns required to estimate one metric."""
    if metric.kind == "ratio":
        assert metric.numerator is not None
        assert metric.denominator is not None
        return metric.numerator, metric.denominator
    columns = [metric.column]
    if metric.covariate is not None:
        columns.append(metric.covariate)
    return tuple(columns)


def _numeric_series(
    frame: pd.DataFrame,
    column: str,
    metric_name: str,
) -> pd.Series:
    """Coerce one metric input column and reject invalid numeric values."""
    original = frame[column]
    converted = pd.to_numeric(original, errors="coerce")
    invalid_conversion = original.notna() & converted.isna()
    if invalid_conversion.any():
        count = int(invalid_conversion.sum())
        raise DataValidationError(
            f"metric '{metric_name}' contains {count} non-numeric value(s)"
        )
    finite_values = converted.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite_values).all():
        raise DataValidationError(
            f"metric '{metric_name}' contains non-finite value(s)"
        )
    return converted.astype(float)


def _hash_frame(frame: pd.DataFrame) -> str:
    """Create a deterministic hash of normalized values and schema."""
    hasher = hashlib.sha256()
    schema = "\n".join(
        f"{column}:{frame[column].dtype}" for column in frame.columns
    )
    hasher.update(schema.encode("utf-8"))
    payload = frame.to_json(
        orient="split",
        date_format="iso",
        date_unit="ns",
        default_handler=str,
    )
    hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()
