"""Regression tests for SDK ingestion and CLI extensions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twosample_means import kaggle
from twosample_means.__main__ import main
from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    diagnose_assignment,
    load_separate_experiment_csvs,
    normalize_experiment_data,
)
from twosample_means.config import InputSpec, RunConfig
from twosample_means.data_io import DataValidationError, load


def _experiment_config(**kwargs: object) -> ExperimentConfig:
    """Build a small valid experiment config for extension tests."""
    values: dict[str, object] = {
        "experiment_id": "extension-test",
        "unit_id": "user_id",
        "assignment": "variant",
        "control": "control",
        "treatments": ("treatment",),
        "metrics": (
            MetricSpec("orders", "orders", "count", role="primary"),
            MetricSpec(
                "revenue_per_order",
                "revenue_per_order",
                "ratio",
                numerator="revenue",
                denominator="orders",
            ),
        ),
    }
    values.update(kwargs)
    return ExperimentConfig(**values)  # type: ignore[arg-type]


def _arm_frame(offset: int, assignment: str | None = None) -> pd.DataFrame:
    """Create a separate-arm frame without an assignment column."""
    frame = pd.DataFrame(
        {
            "user_id": range(offset, offset + 4),
            "orders": [1.0, 2.0, 3.0, 4.0],
            "revenue": [10.0, 20.0, 31.0, 40.0],
        }
    )
    if assignment is not None:
        frame["variant"] = assignment
    return frame


def test_aggregate_unit_type_is_reported_as_a_warning() -> None:
    """Aggregate rows are never silently presented as user-level evidence."""
    config = _experiment_config(unit_type="aggregate")
    data = _arm_frame(0).assign(
        variant=["control", "control", "treatment", "treatment"]
    )

    diagnostics = diagnose_assignment(data, config)

    assert diagnostics.status == "warning"
    assert any(
        "aggregate-level" in warning for warning in diagnostics.warnings
    )


def test_separate_arm_csvs_synthesize_assignment_labels(
    tmp_path: Path,
) -> None:
    """Separate control/treatment files become the normalized contract."""
    control_path = tmp_path / "control.csv"
    treatment_path = tmp_path / "treatment.csv"
    _arm_frame(0).to_csv(control_path, index=False)
    _arm_frame(10).to_csv(treatment_path, index=False)
    config = _experiment_config()

    combined = load_separate_experiment_csvs(
        control_path,
        treatment_path,
        config,
    )
    normalized = normalize_experiment_data(combined, config)

    assert normalized.arm_counts == {"control": 4, "treatment": 4}
    assert normalized.frame["variant"].tolist() == [
        "control",
        "control",
        "control",
        "control",
        "treatment",
        "treatment",
        "treatment",
        "treatment",
    ]


def test_separate_arm_csv_rejects_conflicting_assignment(
    tmp_path: Path,
) -> None:
    """Existing assignment labels cannot contradict the file role."""
    control_path = tmp_path / "control.csv"
    treatment_path = tmp_path / "treatment.csv"
    _arm_frame(0, assignment="treatment").to_csv(control_path, index=False)
    _arm_frame(10).to_csv(treatment_path, index=False)

    with pytest.raises(DataValidationError, match="do not match"):
        load_separate_experiment_csvs(
            control_path,
            treatment_path,
            _experiment_config(),
        )


def test_legacy_nan_handling_can_exclude_missing_values() -> None:
    """The legacy loader can exclude NaNs while retaining strict default."""
    spec = InputSpec(
        sample_a=[1.0, np.nan, 3.0],
        sample_b=[2.0, 4.0, 6.0],
    )

    with pytest.raises(DataValidationError, match="non-finite"):
        load(spec)
    loaded = load(spec, missing_values="exclude")
    configured = load(
        InputSpec(
            sample_a=[1.0, np.nan, 3.0],
            sample_b=[2.0, 4.0, 6.0],
            missing_values="exclude",
        )
    )

    assert loaded.sample_a.tolist() == [1.0, 3.0]
    assert configured.sample_a.tolist() == [1.0, 3.0]


def test_legacy_cli_exposes_nan_exclusion(tmp_path: Path) -> None:
    """The documented CLI switch runs successfully on NaN-containing CSVs."""
    csv_path = tmp_path / "legacy.csv"
    pd.DataFrame(
        {
            "group": [
                "control",
                "control",
                "control",
                "treatment",
                "treatment",
                "treatment",
            ],
            "value": [1.0, np.nan, 2.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(csv_path, index=False)

    exit_code = main(
        [
            "analyze",
            str(csv_path),
            "--group-col",
            "group",
            "--value-col",
            "value",
            "--group-a",
            "control",
            "--group-b",
            "treatment",
            "--missing-values",
            "exclude",
            "--output",
            str(tmp_path / "legacy-report"),
        ]
    )

    assert exit_code == 0


def test_legacy_nan_policy_flows_through_run_config() -> None:
    """Direct callers can configure NaN exclusion without pre-cleaning."""
    from twosample_means.runner import run

    report = run(
        (
            np.array([1.0, np.nan, 3.0]),
            np.array([2.0, 4.0, 6.0]),
        ),
        RunConfig(
            missing_values="exclude",
            include_bayesian=False,
            include_resampling=False,
        ),
    )

    assert report.config["missing_values"] == "exclude"


def test_experiment_cli_supports_separate_count_and_ratio_files(
    tmp_path: Path,
) -> None:
    """The CLI exposes both new metric kinds through separate-arm input."""
    control_path = tmp_path / "control.csv"
    treatment_path = tmp_path / "treatment.csv"
    _arm_frame(0).to_csv(control_path, index=False)
    _arm_frame(10).assign(
        orders=[2.0, 3.0, 4.0, 5.0],
        revenue=[20.0, 40.0, 61.0, 80.0],
    ).to_csv(treatment_path, index=False)
    output = tmp_path / "report"

    exit_code = main(
        [
            "experiment",
            "--csv-a",
            str(control_path),
            "--csv-b",
            str(treatment_path),
            "--unit-col",
            "user_id",
            "--assignment-col",
            "variant",
            "--control",
            "control",
            "--treatment",
            "treatment",
            "--metric",
            "orders=orders:count:primary",
            "--metric",
            "revenue_per_order=revenue/orders:ratio:secondary",
            "--unit-type",
            "user",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert [metric["method"] for metric in report["metrics"]] == [
        "welch_t_count",
        "delta_method_ratio",
    ]
    assert report["config"]["unit_type"] == "user"


def test_experiment_cli_accepts_expected_allocation(tmp_path: Path) -> None:
    """The SRM CLI flag reaches diagnostics and renders a valid report."""
    data_path = tmp_path / "experiment.csv"
    pd.DataFrame(
        {
            "user_id": range(8),
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "converted": [0, 0, 1, 1, 0, 1, 1, 1],
        }
    ).to_csv(data_path, index=False)
    output = tmp_path / "srm-report"

    exit_code = main(
        [
            "experiment",
            str(data_path),
            "--unit-col",
            "user_id",
            "--assignment-col",
            "variant",
            "--control",
            "control",
            "--treatment",
            "treatment",
            "--metric",
            "conversion_rate=converted:binary:primary",
            "--expected-allocation",
            "control=0.5,treatment=0.5",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    diagnostics = report["assignment_diagnostics"]
    assert diagnostics["sample_ratio_mismatch_evaluated"] is True
    assert diagnostics["expected_allocation"] == {
        "control": 0.5,
        "treatment": 0.5,
    }
    assert diagnostics["sample_ratio_mismatch_p_value"] is not None


def test_render_experiment_json_validates_against_bundled_schema(
    tmp_path: Path,
) -> None:
    """Rendered experiment JSON is validated before it is written."""
    from twosample_means import reporting
    from twosample_means.ab_testing import analyze_experiment
    from twosample_means.schemas import (
        SchemaValidationError,
        validate_experiment_json,
    )

    config = _experiment_config(
        metrics=(MetricSpec("orders", "orders", "count", role="primary"),)
    )
    data = pd.concat(
        [_arm_frame(0), _arm_frame(4)],
        ignore_index=True,
    ).assign(
        variant=["control"] * 4 + ["treatment"] * 4,
    )

    document = reporting.render_experiment_json(
        analyze_experiment(data, config)
    )
    parsed = validate_experiment_json(document)

    assert parsed["schema_version"] == "experiment-result-v1"
    missing = parsed.copy()
    del missing["experiment_id"]
    with pytest.raises(SchemaValidationError, match="missing required"):
        validate_experiment_json(json.dumps(missing))
    with pytest.raises(SchemaValidationError, match="not valid JSON"):
        validate_experiment_json("{")


def test_kaggle_registry_exposes_manifests_and_multiple_datasets(
    tmp_path: Path,
) -> None:
    """Registered datasets carry provenance metadata and are discoverable."""
    assert len(kaggle.DATASETS) >= 2
    manifest = kaggle.get_dataset_manifest("marketing-campaign-ab")
    assert manifest.aggregation_level == "campaign-day"
    assert manifest.source.startswith("https://")
    assert manifest.license
    assert manifest.expected_unit_semantics

    for filename in manifest.expected_files:
        (tmp_path / filename).write_text("value\n1\n", encoding="utf-8")
    kaggle.fetch_dataset("marketing-campaign-ab", tmp_path)
    written = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert written["aggregation_level"] == "campaign-day"
    assert written["expected_files"] == list(manifest.expected_files)
