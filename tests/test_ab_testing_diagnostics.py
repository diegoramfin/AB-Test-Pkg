"""Tests for assignment and sample-ratio diagnostics."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    diagnose_assignment,
)
from twosample_means.data_io import DataValidationError


def make_config(**kwargs: object) -> ExperimentConfig:
    """Build a minimal experiment configuration for diagnostics."""
    defaults: dict[str, object] = {
        "experiment_id": "assignment-check",
        "unit_id": "user_id",
        "assignment": "variant",
        "control": "control",
        "treatments": ("treatment",),
        "metrics": (
            MetricSpec(
                name="conversion_rate",
                column="converted",
                kind="binary",
                role="primary",
            ),
        ),
    }
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)  # type: ignore[arg-type]


def test_balanced_assignment_has_no_srm_signal() -> None:
    """Balanced arms match a configured 50/50 allocation."""
    data = pd.DataFrame(
        {
            "user_id": range(100),
            "variant": ["control"] * 50 + ["treatment"] * 50,
        }
    )
    config = make_config(
        expected_allocation={"control": 0.5, "treatment": 0.5}
    )

    result = diagnose_assignment(data, config)

    assert result.status == "ok"
    assert result.assignment_counts == {"control": 50, "treatment": 50}
    assert result.sample_ratio_mismatch_p_value == pytest.approx(1.0)
    assert result.duplicate_units == 0
    assert result.multi_arm_units == 0
    assert result.warnings == ()


def test_srm_uses_multinomial_chi_square() -> None:
    """SRM matches the multinomial goodness-of-fit test."""
    data = pd.DataFrame(
        {
            "user_id": range(100),
            "variant": ["control"] * 90 + ["treatment"] * 10,
        }
    )
    config = make_config(
        expected_allocation={"control": 0.5, "treatment": 0.5}
    )

    result = diagnose_assignment(data, config)
    expected = stats.chisquare([90, 10], f_exp=[50, 50]).pvalue

    assert result.status == "warning"
    assert result.sample_ratio_mismatch_p_value is not None
    assert result.sample_ratio_mismatch_p_value == pytest.approx(expected)
    assert result.sample_ratio_mismatch_p_value < 0.05


def test_duplicate_and_multi_arm_units_are_reported() -> None:
    """Unit duplication and cross-arm assignment are counted separately."""
    data = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "variant": [
                "control",
                "control",
                "control",
                "treatment",
                "treatment",
            ],
        }
    )
    config = make_config(
        expected_allocation={"control": 0.5, "treatment": 0.5}
    )

    result = diagnose_assignment(data, config)

    assert result.status == "warning"
    assert result.duplicate_units == 2
    assert result.multi_arm_units == 1
    assert result.sample_ratio_mismatch_p_value == pytest.approx(1.0)
    assert any("multiple rows" in warning for warning in result.warnings)
    assert any(
        "multiple assignment arms" in warning for warning in result.warnings
    )


def test_missing_and_unknown_assignment_values_are_reported() -> None:
    """Missing and undeclared labels do not disappear from diagnostics."""
    data = pd.DataFrame(
        {
            "user_id": [1, 2, 3, np.nan],
            "variant": ["control", None, "unknown", "treatment"],
        }
    )
    config = make_config()

    result = diagnose_assignment(data, config)

    assert result.status == "warning"
    assert result.missing_unit == 1
    assert result.missing_assignment == 1
    assert result.unknown_assignment == 1
    assert result.assignment_counts == {"control": 1, "treatment": 1}


def test_srm_is_optional_without_expected_allocation() -> None:
    """SRM remains optional without a planned ratio."""
    data = pd.DataFrame(
        {
            "user_id": range(3),
            "variant": ["control", "control", "treatment"],
        }
    )

    result = diagnose_assignment(data, make_config())

    assert result.status == "ok"
    assert result.sample_ratio_mismatch_p_value is None
    assert any("not evaluated" in warning for warning in result.warnings)


def test_missing_assignment_columns_raise_stable_error() -> None:
    """Diagnostics identify missing unit/assignment columns clearly."""
    data = pd.DataFrame({"user_id": [1, 2, 3]})

    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnose_assignment(data, make_config())
