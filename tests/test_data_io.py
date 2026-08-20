"""Tests for data_io: loading, validation, and hashing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twosample_means.config import InputSpec
from twosample_means.data_io import (
    DataValidationError,
    LoadedData,
    load,
    validate_array,
)


class TestLoadFromArrays:
    """Tests for in-memory array input."""

    def test_loads_valid_arrays(self) -> None:
        """Valid in-memory arrays load successfully."""
        spec = InputSpec(
            sample_a=[1.0, 2.0, 3.0, 4.0, 5.0],
            sample_b=[2.0, 3.0, 4.0, 5.0, 6.0],
        )
        data = load(spec)
        assert isinstance(data, LoadedData)
        assert len(data.sample_a) == 5
        assert len(data.sample_b) == 5
        assert data.data_hash

    def test_loads_numpy_arrays(self) -> None:
        """Numpy arrays are accepted."""
        spec = InputSpec(
            sample_a=np.array([1.0, 2.0, 3.0]),
            sample_b=np.array([4.0, 5.0, 6.0]),
        )
        data = load(spec)
        assert len(data.sample_a) == 3

    def test_hash_deterministic(self) -> None:
        """Same data yields the same hash."""
        spec = InputSpec(
            sample_a=[1.0, 2.0, 3.0],
            sample_b=[4.0, 5.0, 6.0],
        )
        data1 = load(spec)
        data2 = load(spec)
        assert data1.data_hash == data2.data_hash

    def test_hash_changes_with_data(self) -> None:
        """Different data yields a different hash."""
        spec1 = InputSpec(
            sample_a=[1.0, 2.0, 3.0],
            sample_b=[4.0, 5.0, 6.0],
        )
        spec2 = InputSpec(
            sample_a=[1.0, 2.0, 3.1],
            sample_b=[4.0, 5.0, 6.0],
        )
        assert load(spec1).data_hash != load(spec2).data_hash

    def test_hash_includes_column_name(self, tmp_path: Path) -> None:
        """Same file, different columns must yield different hashes."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {"control": [1.0, 2.0, 3.0], "treatment": [4.0, 5.0, 6.0]}
        ).to_csv(csv_path, index=False)
        spec_ctrl = InputSpec(
            sample_a=csv_path,
            sample_b=csv_path,
            column_a="control",
            column_b="control",
        )
        spec_treat = InputSpec(
            sample_a=csv_path,
            sample_b=csv_path,
            column_a="treatment",
            column_b="treatment",
        )
        assert load(spec_ctrl).data_hash != load(spec_treat).data_hash


class TestLoadFromCSV:
    """Tests for CSV file input."""

    def test_loads_csv_with_column(self, tmp_path: Path) -> None:
        """CSV loads with a specified column."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {"group": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]}
        ).to_csv(csv_path, index=False)
        spec = InputSpec(
            sample_a=csv_path,
            sample_b=csv_path,
            column_a="value",
            column_b="value",
        )
        data = load(spec)
        assert len(data.sample_a) == 3

    def test_loads_csv_first_numeric(self, tmp_path: Path) -> None:
        """CSV loads the first numeric column by default."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "label": ["x", "y", "z"],
                "value": [10.0, 20.0, 30.0],
            }
        ).to_csv(csv_path, index=False)
        spec = InputSpec(
            sample_a=csv_path,
            sample_b=csv_path,
        )
        data = load(spec)
        assert data.sample_a[0] == 10.0

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        """Missing column name raises DataValidationError."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"value": [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)
        spec = InputSpec(
            sample_a=csv_path,
            sample_b=[1.0, 2.0, 3.0],
            column_a="nonexistent",
        )
        with pytest.raises(DataValidationError, match="nonexistent"):
            load(spec)

    def test_missing_file_raises(self) -> None:
        """Non-existent file raises FileNotFoundError."""
        spec = InputSpec(
            sample_a="nonexistent_file.csv",
            sample_b=[1.0, 2.0, 3.0],
        )
        with pytest.raises(FileNotFoundError):
            load(spec)


class TestLoadFromParquet:
    """Tests for parquet file input."""

    def test_loads_parquet(self, tmp_path: Path) -> None:
        """Parquet files load successfully."""
        pq_path = tmp_path / "data.parquet"
        pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}).to_parquet(pq_path)
        spec = InputSpec(
            sample_a=pq_path,
            sample_b=pq_path,
            column_a="value",
            column_b="value",
        )
        data = load(spec)
        assert len(data.sample_a) == 4


class TestValidation:
    """Tests for input validation."""

    def test_empty_array_raises(self) -> None:
        """Empty array raises DataValidationError."""
        spec = InputSpec(
            sample_a=[],
            sample_b=[1.0, 2.0, 3.0],
        )
        with pytest.raises(DataValidationError, match="empty"):
            load(spec)

    def test_single_element_raises(self) -> None:
        """Single-element array raises (need >= 2)."""
        spec = InputSpec(
            sample_a=[1.0],
            sample_b=[1.0, 2.0, 3.0],
        )
        with pytest.raises(DataValidationError, match="2"):
            load(spec)

    def test_non_finite_raises(self) -> None:
        """NaN values raise DataValidationError."""
        spec = InputSpec(
            sample_a=[1.0, float("nan"), 3.0],
            sample_b=[1.0, 2.0, 3.0],
        )
        with pytest.raises(DataValidationError, match="non-finite"):
            load(spec)

    def test_inf_raises(self) -> None:
        """Inf values raise DataValidationError."""
        spec = InputSpec(
            sample_a=[1.0, float("inf"), 3.0],
            sample_b=[1.0, 2.0, 3.0],
        )
        with pytest.raises(DataValidationError, match="non-finite"):
            load(spec)

    def test_non_numeric_array_raises_data_validation_error(self) -> None:
        """Non-numeric arrays use the package's stable error type."""
        with pytest.raises(DataValidationError, match="one-dimensional"):
            validate_array(["control", "treatment"], "sample_a")

    def test_multidimensional_array_raises_data_validation_error(self) -> None:
        """Multidimensional arrays are rejected at the validation boundary."""
        with pytest.raises(DataValidationError, match="1-D"):
            validate_array([[1.0, 2.0], [3.0, 4.0]], "sample_a")

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        """Unsupported file format raises DataValidationError."""
        bad_path = tmp_path / "data.txt"
        bad_path.write_text("1.0 2.0 3.0")
        spec = InputSpec(
            sample_a=bad_path,
            sample_b=[1.0, 2.0, 3.0],
        )
        with pytest.raises(DataValidationError, match="unsupported"):
            load(spec)


class TestProvenance:
    """Tests for provenance tracking."""

    def test_source_description_present(self) -> None:
        """LoadedData includes a source description."""
        spec = InputSpec(
            sample_a=[1.0, 2.0, 3.0],
            sample_b=[4.0, 5.0, 6.0],
        )
        data = load(spec)
        assert "in-memory" in data.source_description

    def test_hash_is_hex(self) -> None:
        """Hash is a valid hex string."""
        spec = InputSpec(
            sample_a=[1.0, 2.0, 3.0],
            sample_b=[4.0, 5.0, 6.0],
        )
        data = load(spec)
        assert len(data.data_hash) == 64
        int(data.data_hash, 16)  # valid hex
