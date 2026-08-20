"""Data I/O: load CSV/parquet/in-memory arrays, validate, hash.

This module provides ``load()``, the single entry point for converting
an ``InputSpec`` into validated numeric arrays with a deterministic
provenance hash. All downstream methods receive the output of
``load()`` — they never touch files or raw input directly.

Academic rationale
------------------
Data provenance is a foundational requirement for auditable statistical
analysis. Recording a deterministic hash of the raw input data ensures
that any subsequent analysis can be traced back to the exact data it
was run on. This follows the reproducibility principles articulated in
Peng (2011, "Reproducible Research in Computational Science") and the
FAIR data principles (Wilkinson et al., 2016).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from twosample_means.config import (
    InputSpec,
    MissingValuePolicy,
    SampleSource,
)

FloatArray = npt.NDArray[np.float64]


class DataValidationError(ValueError):
    """Raised when input data fails validation."""


@dataclass(frozen=True)
class LoadedData:
    """Validated loaded data with provenance hash.

    Attributes
    ----------
    sample_a:
        Validated numeric array for sample A.
    sample_b:
        Validated numeric array for sample B.
    data_hash:
        SHA-256 hash of the raw input bytes, for provenance.
    source_description:
        Human-readable description of the data source.

    """

    sample_a: FloatArray
    sample_b: FloatArray
    data_hash: str
    source_description: str


def validate_array(
    source: object,
    label: str,
    missing_values: MissingValuePolicy = "error",
) -> FloatArray:
    """Coerce and validate one sample for public analysis APIs.

    Raises ``DataValidationError`` rather than leaking NumPy conversion or
    shape errors so callers get one stable input-validation boundary.
    """
    try:
        array = np.asarray(source, dtype=float)
    except (TypeError, ValueError) as error:
        raise DataValidationError(
            f"{label}: data must be a one-dimensional numeric array."
        ) from error
    array = _validate_array(array, label, missing_values)
    return np.ascontiguousarray(array, dtype=np.float64)


def validate_samples(
    sample_a: object,
    sample_b: object,
    missing_values: MissingValuePolicy = "error",
) -> tuple[FloatArray, FloatArray]:
    """Validate and normalize both samples for a direct ``run`` call."""
    return (
        validate_array(sample_a, "sample_a", missing_values),
        validate_array(sample_b, "sample_b", missing_values),
    )


def load(
    spec: InputSpec,
    missing_values: MissingValuePolicy | None = None,
) -> LoadedData:
    """Load and validate two samples from an ``InputSpec``.

    Accepts either file paths (CSV or parquet) or in-memory
    array-likes. Validates that the data is non-empty, numeric, finite,
    and meets a minimum sample size. Computes a deterministic SHA-256
    hash of the raw input for provenance.

    Parameters
    ----------
    spec:
        The input specification containing the two samples.

    Returns
    -------
    LoadedData
        The validated arrays and provenance hash.

    Raises
    ------
    DataValidationError
        If the data is empty, non-numeric, contains non-finite
        values, or has fewer than 2 observations per sample.
    FileNotFoundError
        If a file path does not exist.

    """
    policy = spec.missing_values if missing_values is None else missing_values
    sample_a, desc_a = _load_single(
        spec.sample_a, spec.column_a, "sample_a", policy
    )
    sample_b, desc_b = _load_single(
        spec.sample_b, spec.column_b, "sample_b", policy
    )
    combined_hash = _compute_hash(spec)
    description = f"A: {desc_a} | B: {desc_b}"
    return LoadedData(
        sample_a=sample_a,
        sample_b=sample_b,
        data_hash=combined_hash,
        source_description=description,
    )


def _load_single(
    source: SampleSource,
    column: str | None,
    label: str,
    missing_values: MissingValuePolicy,
) -> tuple[FloatArray, str]:
    """Load a single sample from a file or in-memory array.

    Parameters
    ----------
    source:
        File path or in-memory sequence.
    column:
        Column name for file-based input. If ``None``, the first
        numeric column is used.
    label:
        Human-readable label for error messages.

    Returns
    -------
    tuple[FloatArray, str]
        The validated numeric array and a source description.

    """
    if source is None:
        raise DataValidationError(f"{label}: source must not be None.")
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"{label}: file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            array, desc = _load_csv(path, column, label)
        elif suffix in (".parquet", ".pq"):
            array, desc = _load_parquet(path, column, label)
        else:
            raise DataValidationError(
                f"{label}: unsupported file format "
                f"'{suffix}'. Use .csv or .parquet."
            )
    else:
        array = validate_array(source, label, missing_values)
        desc = f"in-memory array ({len(array)} values)"
        return array, desc
    array = _validate_array(array, label, missing_values)
    return np.ascontiguousarray(array, dtype=np.float64), desc


def _load_csv(
    path: Path,
    column: str | None,
    label: str,
) -> tuple[FloatArray, str]:
    """Load a numeric column from a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file.
    column:
        Column name. If ``None``, the first numeric column is used.
    label:
        Human-readable label for error messages.

    Returns
    -------
    tuple[FloatArray, str]
        The numeric array and a source description.

    """
    df = pd.read_csv(path)
    if column is not None:
        if column not in df.columns:
            raise DataValidationError(
                f"{label}: column '{column}' not found in "
                f"{path}. Available: {list(df.columns)}"
            )
        try:
            values = df[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise DataValidationError(
                f"{label}: column '{column}' is not numeric in {path}."
            ) from error
        desc = f"CSV {path.name} column '{column}'"
    else:
        values = _first_numeric_column(df, label, path.name)
        desc = f"CSV {path.name} (first numeric column)"
    return np.asarray(values, dtype=float), desc


def _load_parquet(
    path: Path,
    column: str | None,
    label: str,
) -> tuple[FloatArray, str]:
    """Load a numeric column from a parquet file.

    Parameters
    ----------
    path:
        Path to the parquet file.
    column:
        Column name. If ``None``, the first numeric column is used.
    label:
        Human-readable label for error messages.

    Returns
    -------
    tuple[FloatArray, str]
        The numeric array and a source description.

    """
    df = pd.read_parquet(path)
    if column is not None:
        if column not in df.columns:
            raise DataValidationError(
                f"{label}: column '{column}' not found in "
                f"{path}. Available: {list(df.columns)}"
            )
        try:
            values = df[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise DataValidationError(
                f"{label}: column '{column}' is not numeric in {path}."
            ) from error
        desc = f"parquet {path.name} column '{column}'"
    else:
        values = _first_numeric_column(df, label, path.name)
        desc = f"parquet {path.name} (first numeric column)"
    return np.asarray(values, dtype=float), desc


def _first_numeric_column(
    df: pd.DataFrame,
    label: str,
    filename: str,
) -> FloatArray:
    """Extract the first numeric column from a DataFrame.

    Parameters
    ----------
    df:
        The DataFrame to search.
    label:
        Human-readable label for error messages.
    filename:
        Name of the source file for error messages.

    Returns
    -------
    FloatArray
        The first numeric column as a float array.

    Raises
    ------
    DataValidationError
        If no numeric column is found.

    """
    for col in df.columns:
        if df[col].dtype.kind in "iuf":
            return np.asarray(df[col].to_numpy(dtype=float))
    raise DataValidationError(
        f"{label}: no numeric column found in {filename}. "
        f"Columns: {list(df.columns)}"
    )


def _validate_array(
    array: FloatArray,
    label: str,
    missing_values: MissingValuePolicy = "error",
) -> FloatArray:
    """Validate that an array meets minimum quality criteria.

    Parameters
    ----------
    array:
        The array to validate.
    label:
        Human-readable label for error messages.

    Raises
    ------
    DataValidationError
        If the array is empty, non-numeric, contains non-finite
        values, or has fewer than 2 observations.

    """
    if array.size == 0:
        raise DataValidationError(f"{label}: data is empty.")
    if array.ndim != 1:
        raise DataValidationError(
            f"{label}: expected 1-D array, got {array.ndim}-D."
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise DataValidationError(f"{label}: data is not numeric.")
    if missing_values not in ("error", "exclude"):
        raise ValueError(
            "missing_values must be 'error' or 'exclude', "
            f"got {missing_values!r}"
        )
    if missing_values == "exclude" and np.isnan(array).any():
        array = array[~np.isnan(array)]
    if not np.all(np.isfinite(array)):
        non_finite = np.sum(~np.isfinite(array))
        raise DataValidationError(
            f"{label}: data contains {non_finite} "
            "non-finite value(s) (NaN or Inf)."
        )
    if array.size < 2:
        raise DataValidationError(
            f"{label}: need at least 2 observations after missing-value "
            f"handling, got {array.size}."
        )
    return array


def _compute_hash(spec: InputSpec) -> str:
    """Compute a deterministic SHA-256 hash of the input data.

    For file inputs, the file bytes are hashed. For in-memory inputs,
    the array bytes are hashed. The hash is prefixed with the source
    type for traceability.

    Parameters
    ----------
    spec:
        The input specification.

    Returns
    -------
    str
        A hex digest of the SHA-256 hash.

    """
    hasher = hashlib.sha256()
    for source, column in (
        (spec.sample_a, spec.column_a),
        (spec.sample_b, spec.column_b),
    ):
        if isinstance(source, str | Path):
            path = Path(source)
            if path.exists():
                hasher.update(path.read_bytes())
                hasher.update(b"|column=")
                hasher.update((column or "<auto>").encode("utf-8"))
        else:
            array = np.ascontiguousarray(np.asarray(source, dtype=float))
            hasher.update(array.tobytes())
            hasher.update(b"|shape=")
            hasher.update(str(array.shape).encode())
            hasher.update(b"|dtype=float64")
    return hasher.hexdigest()
