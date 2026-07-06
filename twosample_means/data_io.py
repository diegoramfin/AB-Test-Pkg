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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twosample_means.config import InputSpec


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

    sample_a: np.ndarray
    sample_b: np.ndarray
    data_hash: str
    source_description: str


def load(spec: InputSpec) -> LoadedData:
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
    sample_a, desc_a = _load_single(spec.sample_a, spec.column_a, "sample_a")
    sample_b, desc_b = _load_single(spec.sample_b, spec.column_b, "sample_b")
    combined_hash = _compute_hash(spec)
    description = f"A: {desc_a} | B: {desc_b}"
    return LoadedData(
        sample_a=sample_a,
        sample_b=sample_b,
        data_hash=combined_hash,
        source_description=description,
    )


def _load_single(
    source: str | Path | Sequence[Any],
    column: str | None,
    label: str,
) -> tuple[np.ndarray, str]:
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
    tuple[np.ndarray, str]
        The validated numeric array and a source description.
    """
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
        array = np.asarray(source, dtype=float)
        desc = f"in-memory array ({len(array)} values)"
    _validate_array(array, label)
    return array, desc


def _load_csv(
    path: Path,
    column: str | None,
    label: str,
) -> tuple[np.ndarray, str]:
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
    tuple[np.ndarray, str]
        The numeric array and a source description.
    """
    df = pd.read_csv(path)
    if column is not None:
        if column not in df.columns:
            raise DataValidationError(
                f"{label}: column '{column}' not found in "
                f"{path}. Available: {list(df.columns)}"
            )
        values = df[column].to_numpy(dtype=float)
        desc = f"CSV {path.name} column '{column}'"
    else:
        values = _first_numeric_column(df, label, path.name)
        desc = f"CSV {path.name} (first numeric column)"
    return np.asarray(values, dtype=float), desc


def _load_parquet(
    path: Path,
    column: str | None,
    label: str,
) -> tuple[np.ndarray, str]:
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
    tuple[np.ndarray, str]
        The numeric array and a source description.
    """
    df = pd.read_parquet(path)
    if column is not None:
        if column not in df.columns:
            raise DataValidationError(
                f"{label}: column '{column}' not found in "
                f"{path}. Available: {list(df.columns)}"
            )
        values = df[column].to_numpy(dtype=float)
        desc = f"parquet {path.name} column '{column}'"
    else:
        values = _first_numeric_column(df, label, path.name)
        desc = f"parquet {path.name} (first numeric column)"
    return np.asarray(values, dtype=float), desc


def _first_numeric_column(
    df: pd.DataFrame,
    label: str,
    filename: str,
) -> np.ndarray:
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
    np.ndarray
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


def _validate_array(array: np.ndarray, label: str) -> None:
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
        raise DataValidationError(f"{label}: expected 1-D array, got {array.ndim}-D.")
    if not np.issubdtype(array.dtype, np.floating):
        raise DataValidationError(f"{label}: data is not numeric.")
    if not np.all(np.isfinite(array)):
        non_finite = np.sum(~np.isfinite(array))
        raise DataValidationError(
            f"{label}: data contains {non_finite} " "non-finite value(s) (NaN or Inf)."
        )
    if array.size < 2:
        raise DataValidationError(
            f"{label}: need at least 2 observations, got {array.size}."
        )


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
