"""Tests for assignment and sample-ratio diagnostics."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from twosample_means.ab_testing import (
    BALANCE_SMD_THRESHOLD,
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


def _balance_frame() -> pd.DataFrame:
    """A frame with a balanced and an imbalanced pre-treatment covariate."""
    rng = np.random.default_rng(7)
    # ``pre_spend`` is assigned by alternating the same values, so the arms
    # share an identical distribution and the SMD is exactly zero.
    pre_spend = rng.normal(50.0, 8.0, 60)
    frame = pd.DataFrame(
        {
            "user_id": range(120),
            "variant": ["control"] * 60 + ["treatment"] * 60,
            "pre_spend": np.concatenate([pre_spend, pre_spend]),
        }
    )
    frame["tenure"] = rng.normal(0.0, 1.0, 120)
    frame.loc[frame["variant"] == "treatment", "tenure"] += 0.35
    return frame


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


def test_covariate_balance_matches_hand_computed_smd() -> None:
    """SMD equals (mean_arm - mean_control) / pooled_sd at unit level."""
    data = _balance_frame()
    config = make_config(
        metrics=(
            MetricSpec(
                name="revenue",
                column="pre_spend",
                kind="continuous",
                role="primary",
                covariate="tenure",
            ),
        )
    )

    result = diagnose_assignment(data, config)

    assert len(result.covariate_balance) == 1
    entry = result.covariate_balance[0]
    control = data.loc[data["variant"] == "control", "tenure"].to_numpy()
    arm = data.loc[data["variant"] == "treatment", "tenure"].to_numpy()
    pooled = np.sqrt(
        (
            (len(control) - 1) * control.var(ddof=1)
            + (len(arm) - 1) * arm.var(ddof=1)
        )
        / (len(control) + len(arm) - 2)
    )
    expected_smd = (arm.mean() - control.mean()) / pooled
    assert entry.smd == pytest.approx(expected_smd)
    assert entry.control_mean == pytest.approx(control.mean())
    assert entry.arm_mean == pytest.approx(arm.mean())
    assert entry.exceeds_threshold is True
    assert any("Covariate imbalance" in w for w in result.warnings)


def test_covariate_balance_without_declared_covariates_is_empty() -> None:
    """No declared covariates means no balance rows and no warnings."""
    data = _balance_frame()
    result = diagnose_assignment(data, make_config())

    assert result.covariate_balance == ()
    assert not any("balance" in w.lower() for w in result.warnings)


def test_covariate_balance_flags_constant_covariate() -> None:
    """A constant covariate cannot produce an SMD and warns instead."""
    data = _balance_frame()
    data["constant_col"] = 1.0
    config = make_config(
        metrics=(
            MetricSpec(
                name="revenue",
                column="pre_spend",
                kind="continuous",
                role="primary",
                covariate="constant_col",
            ),
        )
    )

    result = diagnose_assignment(data, config)

    assert len(result.covariate_balance) == 1
    entry = result.covariate_balance[0]
    assert entry.smd is None
    assert entry.pooled_sd == 0.0
    assert "constant" in (entry.warning or "")
    assert any("Covariate balance" in w for w in result.warnings)


def test_covariate_balance_reports_balanced_covariate_cleanly() -> None:
    """Small SMDs stay under the threshold without warnings."""
    data = _balance_frame()
    data["revenue"] = data["pre_spend"] + data["tenure"]
    config = make_config(
        metrics=(
            MetricSpec(
                name="revenue",
                column="revenue",
                kind="continuous",
                role="primary",
                covariate="pre_spend",
            ),
        )
    )

    result = diagnose_assignment(data, config)

    assert len(result.covariate_balance) == 1
    entry = result.covariate_balance[0]
    assert entry.exceeds_threshold is False
    assert abs(entry.smd or 0.0) <= BALANCE_SMD_THRESHOLD
    assert not any("Covariate imbalance" in w for w in result.warnings)


def test_stratum_srm_detects_offsetting_strata_imbalance() -> None:
    """Balanced margins can hide per-stratum mismatch."""
    rows = []
    for region, control_count, treatment_count in (
        ("A", 70, 30),
        ("B", 30, 70),
    ):
        for _ in range(control_count):
            rows.append((region, "control"))
        for _ in range(treatment_count):
            rows.append((region, "treatment"))
    data = pd.DataFrame(rows, columns=["region", "variant"])
    data["user_id"] = range(len(data))
    config = make_config(
        strata="region",
        expected_allocation={"control": 0.5, "treatment": 0.5},
    )

    result = diagnose_assignment(data, config)

    # Marginal allocation is exactly 50/50, so the marginal test passes.
    assert result.sample_ratio_mismatch_p_value == pytest.approx(1.0)
    assert result.status == "warning"
    assert len(result.stratum_srm) == 2
    assert all(entry.n == 100 for entry in result.stratum_srm)
    assert all(entry.p_value is not None for entry in result.stratum_srm)
    assert all((entry.p_value or 0.0) < 0.05 for entry in result.stratum_srm)
    assert any("within stratum" in w for w in result.warnings)


def test_stratum_srm_without_expected_allocation_is_unavailable() -> None:
    """Per-stratum tests respect the same optionality as the marginal test."""
    rows = []
    for region, count in (("A", 50), ("B", 50)):
        for _ in range(count):
            rows.append((region, "control"))
        for _ in range(count):
            rows.append((region, "treatment"))
    data = pd.DataFrame(rows, columns=["region", "variant"])
    data["user_id"] = range(len(data))
    config = make_config(strata="region")

    result = diagnose_assignment(data, config)

    assert len(result.stratum_srm) == 2
    assert all(entry.p_value is None for entry in result.stratum_srm)
    assert any("not evaluated" in w for w in result.warnings)


def test_stratum_srm_requires_declared_strata_column() -> None:
    """A declared strata column must exist in the data."""
    data = _balance_frame()
    config = make_config(strata="region")

    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnose_assignment(data, config)


def test_balance_columns_produce_smd_entries_without_covariates() -> None:
    """Balance-only columns are checked even without metric adjustment."""
    data = _balance_frame()
    config = make_config(balance_columns=("tenure", "pre_spend"))

    result = diagnose_assignment(data, config)

    covariates = [entry.covariate for entry in result.covariate_balance]
    assert covariates == ["tenure", "pre_spend"]
    tenure_entries = [
        entry
        for entry in result.covariate_balance
        if entry.covariate == "tenure"
    ]
    assert len(tenure_entries) == 1
    # tenure has a 0.35-shift vs control, which exceeds the 0.1 threshold.
    assert tenure_entries[0].exceeds_threshold is True
    assert tenure_entries[0].smd is not None
    pre_spend_entries = [
        entry
        for entry in result.covariate_balance
        if entry.covariate == "pre_spend"
    ]
    assert len(pre_spend_entries) == 1
    # pre_spend is identical across arms, so its SMD is exactly zero.
    assert pre_spend_entries[0].smd == pytest.approx(0.0)
    assert pre_spend_entries[0].exceeds_threshold is False
    assert any("Covariate imbalance" in w for w in result.warnings)


def test_balance_columns_combine_with_metric_covariates() -> None:
    """Metric covariates and balance-only columns produce distinct rows."""
    data = _balance_frame()
    data["revenue"] = data["pre_spend"] + data["tenure"]
    config = make_config(
        metrics=(
            MetricSpec(
                name="revenue",
                column="revenue",
                kind="continuous",
                role="primary",
                covariate="tenure",
            ),
        ),
        balance_columns=("pre_spend",),
    )

    result = diagnose_assignment(data, config)

    covariates = [entry.covariate for entry in result.covariate_balance]
    # The metric covariate is listed first; the balance-only column is added
    # after it without duplication.
    assert covariates == ["tenure", "pre_spend"]
    assert len(result.covariate_balance) == 2


def test_missing_balance_column_raises_stable_error() -> None:
    """A declared balance column must exist in the data."""
    data = _balance_frame()
    config = make_config(balance_columns=("missing_col",))

    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnose_assignment(data, config)
