"""Smoke tests for the bundled example scripts.

Examples are real, runnable workflows. These tests execute their ``run``
entry points against temporary directories so broken examples fail CI.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# Summary-style examples write a text report instead of an experiment
# report; the value is (filename, expected substring). Everything else
# writes report.md/json/html.
SUMMARY_FILES = {
    "04_planning_power_sequential": ("planning.txt", "calibrated"),
    "09_sequential_analysis": ("sequential.txt", "calibrated"),
    "10_difference_in_differences": (
        "did_report.md",
        "difference-in-differences",
    ),
    "11_kaggle_manifest_adapter": ("kaggle_adapter.txt", "manifest"),
    "15_staggered_adoption": ("staggered_did_report.md", "callaway"),
}


@pytest.mark.parametrize(
    "module_name",
    [
        "01_binary_conversion",
        "02_continuous_cuped",
        "03_count_ratio",
        "04_planning_power_sequential",
        "05_clustered_ratio",
        "06_holm_multiplicity",
        "07_multi_arm_contrasts",
        "08_separate_csvs",
        "09_sequential_analysis",
        "10_difference_in_differences",
        "11_kaggle_manifest_adapter",
        "12_stratified_balance",
        "13_clustered_stratified_balance",
        "14_clustered_stratified_balance_cli",
        "15_staggered_adoption",
    ],
)
def test_example_runs_to_completion(
    module_name: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each example runs and writes its expected output."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        module = importlib.import_module(module_name)
        output = module.run(tmp_path)
    finally:
        sys.path.remove(str(EXAMPLES))

    assert output.exists()
    if module_name in SUMMARY_FILES:
        summary_name, expected = SUMMARY_FILES[module_name]
        summary = output / summary_name
        assert summary.exists()
        assert expected in summary.read_text(encoding="utf-8").lower()
    else:
        assert (output / "report.md").exists()
        assert (output / "report.json").exists()
        assert (output / "report.html").exists()
        # Every experiment example declares an expected allocation, so the
        # sample-ratio mismatch test must be evaluated, not skipped.
        diagnostics = json.loads(
            (output / "report.json").read_text(encoding="utf-8")
        )["assignment_diagnostics"]
        assert diagnostics["sample_ratio_mismatch_evaluated"] is True
        assert diagnostics["sample_ratio_mismatch_p_value"] is not None
        assert "not evaluated" not in " ".join(diagnostics["warnings"])
    if module_name == "06_holm_multiplicity":
        # The point of this example is an active correction: the primary
        # metric's Holm-adjusted p-value must exceed its raw value.
        metrics = json.loads(
            (output / "report.json").read_text(encoding="utf-8")
        )["metrics"]
        primary = metrics[0]
        assert primary["adjusted_p_value"] > primary["p_value"]
    if module_name == "12_stratified_balance":
        # The point of this example is the offsetting per-stratum
        # imbalance: marginal SRM passes while every stratum fails.
        diagnostics = json.loads(
            (output / "report.json").read_text(encoding="utf-8")
        )["assignment_diagnostics"]
        assert diagnostics["sample_ratio_mismatch_p_value"] == pytest.approx(
            1.0
        )
        assert len(diagnostics["stratum_srm"]) == 2
        assert all(
            entry["p_value"] is not None and entry["p_value"] < 0.05
            for entry in diagnostics["stratum_srm"]
        )
        assert len(diagnostics["covariate_balance"]) == 2
        assert any(
            entry["exceeds_threshold"]
            for entry in diagnostics["covariate_balance"]
        )
    if module_name == "13_clustered_stratified_balance":
        # The point of this example is cluster + strata + balance together:
        # SRM passes at both levels, the balance-only column is flagged, and
        # the cluster-robust SE is wider than the naive user-level SE.
        report = json.loads(
            (output / "report.json").read_text(encoding="utf-8")
        )
        diagnostics = report["assignment_diagnostics"]
        assert len(diagnostics["stratum_srm"]) == 2
        assert all(
            entry["p_value"] is not None and entry["p_value"] > 0.05
            for entry in diagnostics["stratum_srm"]
        )
        balance = {
            entry["covariate"]: entry
            for entry in diagnostics["covariate_balance"]
        }
        assert set(balance) == {"pre_spend", "device_score"}
        assert balance["pre_spend"]["smd"] == pytest.approx(0.0, abs=1e-9)
        assert balance["device_score"]["exceeds_threshold"] is True
        primary = report["metrics"][0]
        assert primary["cluster_robust"] is True
        assert primary["standard_error"] > primary["naive_standard_error"]
    if module_name == "14_clustered_stratified_balance_cli":
        # The point of this example is that the CLI flags survive the full
        # round trip: the config recorded in the report must show cluster,
        # strata, and balance columns, and the analysis must reflect them.
        report = json.loads(
            (output / "report.json").read_text(encoding="utf-8")
        )
        config = report["config"]
        assert config["cluster"] == "store_id"
        assert config["strata"] == "region"
        assert config["balance_columns"] == ["device_score"]
        diagnostics = report["assignment_diagnostics"]
        assert len(diagnostics["stratum_srm"]) == 2
        balance = {
            entry["covariate"]: entry
            for entry in diagnostics["covariate_balance"]
        }
        assert balance["pre_spend"]["smd"] == pytest.approx(0.0, abs=1e-9)
        assert balance["device_score"]["exceeds_threshold"] is True
        primary = report["metrics"][0]
        assert primary["cluster_robust"] is True
        assert primary["standard_error"] > primary["naive_standard_error"]
    assert capsys.readouterr().out != ""
