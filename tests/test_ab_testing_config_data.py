"""Tests for the experiment-level configuration and data contract."""

import numpy as np
import pandas as pd
import pytest

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    NormalizedExperimentData,
    analyze_experiment,
    normalize_experiment_data,
)
from twosample_means.data_io import DataValidationError


def make_config(**kwargs: object) -> ExperimentConfig:
    """Build a valid two-arm experiment configuration for tests."""
    defaults: dict[str, object] = {
        "experiment_id": "checkout-copy",
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
                practical_effect=0.002,
            ),
            MetricSpec(
                name="revenue",
                column="revenue",
                kind="continuous",
                role="secondary",
            ),
        ),
    }
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)  # type: ignore[arg-type]


def make_data() -> pd.DataFrame:
    """Build a small user-level experiment frame."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "variant": ["control", "control", "treatment", "treatment"],
            "converted": [0, 1, 1, np.nan],
            "revenue": [10.0, 20.0, 30.0, np.nan],
        }
    )


def test_config_normalizes_collections_and_exposes_arms() -> None:
    """List inputs become immutable tuples and arms preserve direction."""
    config = make_config(treatments=["treatment"])

    assert config.treatments == ("treatment",)
    assert config.arms == ("control", "treatment")
    assert config.metrics[0].role == "primary"
    assert config.multiplicity_scope == "family"


def test_config_accepts_global_multiplicity_scope() -> None:
    """Global scope is an explicit opt-in over family-scoped correction."""
    config = make_config(multiplicity_scope="global")

    assert config.multiplicity_scope == "global"


def test_config_rejects_unknown_multiplicity_scope() -> None:
    """Correction scope values are validated at the frozen config boundary."""
    with pytest.raises(ValueError, match="multiplicity_scope"):
        make_config(multiplicity_scope="per_role")


def test_config_rejects_non_string_time_column() -> None:
    """Time-column validation should fail cleanly for malformed inputs."""
    with pytest.raises(ValueError, match="time_column"):
        make_config(time_column=42)


def test_orchestrator_rejects_unsupported_multi_arm_analysis() -> None:
    """The two-arm orchestrator fails clearly before estimator dispatch."""
    config = make_config(treatments=("treatment", "other"))

    with pytest.raises(ValueError, match="exactly one treatment arm"):
        analyze_experiment(make_data(), config)


def test_continuous_metric_rejects_binary_success_value() -> None:
    """Success values are meaningful only for binary metrics."""
    with pytest.raises(ValueError, match="only configurable"):
        MetricSpec("value", "value", "continuous", success_value=False)


def test_config_requires_one_primary_metric() -> None:
    """The analysis plan cannot omit or duplicate the primary metric."""
    metrics = (
        MetricSpec("a", "a", "continuous"),
        MetricSpec("b", "b", "continuous"),
    )
    with pytest.raises(ValueError, match="exactly one primary"):
        make_config(metrics=metrics)


def test_config_validates_expected_allocation() -> None:
    """Expected allocation must name all arms and sum to one."""
    config = make_config(
        expected_allocation={"control": 0.5, "treatment": 0.5}
    )
    assert config.expected_allocation == {"control": 0.5, "treatment": 0.5}
    with pytest.raises(ValueError, match="sum to 1"):
        make_config(expected_allocation={"control": 0.4, "treatment": 0.4})


def test_config_validates_strata_column() -> None:
    """Strata must be a non-empty string distinct from other columns."""
    config = make_config(strata="region")
    assert config.strata == "region"

    with pytest.raises(ValueError, match="non-empty string"):
        make_config(strata="")
    with pytest.raises(ValueError, match="differ from unit and assignment"):
        make_config(strata="user_id")
    with pytest.raises(ValueError, match="cluster column"):
        make_config(strata="cluster", cluster="cluster")
    with pytest.raises(ValueError, match="metric columns"):
        make_config(strata="converted")


def test_config_normalizes_balance_columns() -> None:
    """List inputs become immutable tuples of column names."""
    config = make_config(balance_columns=["tenure", "region"])

    assert config.balance_columns == ("tenure", "region")


def test_config_rejects_duplicate_balance_columns() -> None:
    """A column cannot be declared twice in the balance check."""
    with pytest.raises(ValueError, match="duplicate columns"):
        make_config(balance_columns=("tenure", "tenure"))


def test_config_rejects_empty_balance_column_name() -> None:
    """Balance column names must be non-empty strings."""
    with pytest.raises(ValueError, match="non-empty strings"):
        make_config(balance_columns=("tenure", ""))


def test_config_rejects_balance_column_overlapping_metric_covariate() -> None:
    """A metric covariate is already checked; balance cannot duplicate it."""
    metrics = (
        MetricSpec(
            name="revenue",
            column="revenue",
            kind="continuous",
            role="primary",
            covariate="tenure",
        ),
    )

    with pytest.raises(ValueError, match="metric covariate columns"):
        make_config(metrics=metrics, balance_columns=("tenure",))


def test_config_rejects_balance_column_overlapping_reserved_columns() -> None:
    """Balance columns must differ from assignment, cluster, and metrics."""
    with pytest.raises(ValueError, match="differ from unit, assignment"):
        make_config(balance_columns=("user_id",))
    with pytest.raises(ValueError, match="differ from unit, assignment"):
        make_config(balance_columns=("converted",))
    with pytest.raises(ValueError, match="differ from unit, assignment"):
        make_config(
            balance_columns=("cluster",),
            cluster="cluster",
            strata="region",
        )


def test_normalize_returns_user_level_contract() -> None:
    """Normalization preserves rows while canonicalizing metric values."""
    result = normalize_experiment_data(make_data(), make_config())

    assert isinstance(result, NormalizedExperimentData)
    assert result.source_rows == 4
    assert result.analysis_rows == 4
    assert result.excluded_rows == 0
    assert result.metric_names == ("conversion_rate", "revenue")
    assert result.missing_outcomes == {"conversion_rate": 1, "revenue": 1}
    assert result.arm_counts == {"control": 2, "treatment": 2}
    assert result.frame["converted"].tolist()[:3] == [0.0, 1.0, 1.0]
    assert pd.isna(result.frame.loc[3, "converted"])
    assert len(result.data_hash) == 64


def test_normalization_hash_is_deterministic() -> None:
    """Equivalent frames produce the same provenance hash."""
    config = make_config()
    first = normalize_experiment_data(make_data(), config)
    second = normalize_experiment_data(make_data(), config)

    assert first.data_hash == second.data_hash


def test_time_window_is_inclusive_and_counts_exclusions() -> None:
    """The configured inclusive window is part of normalization provenance."""
    data = make_data().assign(
        observed_at=pd.date_range("2026-01-01", periods=4, freq="D")
    )
    config = make_config(
        time_column="observed_at",
        analysis_start="2026-01-02",
        analysis_end="2026-01-03",
    )
    result = normalize_experiment_data(data, config)

    assert result.source_rows == 4
    assert result.analysis_rows == 2
    assert result.excluded_rows == 2
    assert result.frame["user_id"].tolist() == [2, 3]


def test_duplicate_units_are_rejected() -> None:
    """Repeated event rows cannot silently be analyzed as independent users."""
    data = make_data()
    data.loc[1, "user_id"] = data.loc[0, "user_id"]

    with pytest.raises(DataValidationError, match="duplicate"):
        normalize_experiment_data(data, make_config())


def test_unknown_assignment_is_rejected() -> None:
    """Assignment labels outside the declared arms fail early."""
    data = make_data()
    data.loc[0, "variant"] = "unknown"

    with pytest.raises(DataValidationError, match="unknown labels"):
        normalize_experiment_data(data, make_config())


def test_invalid_binary_values_are_rejected() -> None:
    """Binary metrics accept only 0/1 values after coercion."""
    data = make_data()
    data.loc[0, "converted"] = 2

    with pytest.raises(DataValidationError, match="outside 0/1"):
        normalize_experiment_data(data, make_config())


def test_non_numeric_continuous_values_are_rejected() -> None:
    """Continuous metrics reject non-numeric non-missing values."""
    data = make_data()
    data["revenue"] = data["revenue"].astype(object)
    data.loc[0, "revenue"] = "not-a-number"

    with pytest.raises(DataValidationError, match="non-numeric"):
        normalize_experiment_data(data, make_config())


def test_missing_error_policy_is_enforced() -> None:
    """A metric can require complete observations instead of excluding them."""
    config = make_config(
        metrics=(
            MetricSpec(
                name="conversion_rate",
                column="converted",
                kind="binary",
                role="primary",
                missing="error",
            ),
        )
    )

    with pytest.raises(DataValidationError, match="missing"):
        normalize_experiment_data(make_data(), config)
