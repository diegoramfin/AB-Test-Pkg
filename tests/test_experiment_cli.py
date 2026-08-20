"""Tests for the experiment-level CLI workflow."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from twosample_means import __main__ as cli
from twosample_means.ab_testing import ExperimentConfig


def experiment_arguments(csv_path: Path) -> list[str]:
    """Build representative experiment CLI arguments."""
    return [
        "experiment",
        str(csv_path),
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
        "--metric",
        "revenue=revenue:continuous:secondary",
        "--metric-family",
        "conversion_rate=conversion",
        "--metric-family",
        "revenue=engagement",
        "--multiplicity",
        "fdr_bh",
        "--multiplicity-scope",
        "global",
        "--output",
        str(csv_path.parent / "artifacts"),
    ]


def test_parser_exposes_multiplicity_and_metric_families(
    tmp_path: Path,
) -> None:
    """Experiment CLI flags become the typed configuration fields."""
    parser = cli._build_parser()
    args = parser.parse_args(experiment_arguments(tmp_path / "experiment.csv"))

    config = cli._build_experiment_config(args)

    assert config.multiplicity == "fdr_bh"
    assert config.multiplicity_scope == "global"
    assert [(metric.name, metric.family) for metric in config.metrics] == [
        ("conversion_rate", "conversion"),
        ("revenue", "engagement"),
    ]
    assert [metric.kind for metric in config.metrics] == [
        "binary",
        "continuous",
    ]


def test_analyze_experiment_alias_is_available(tmp_path: Path) -> None:
    """The explicit alias is accepted for discoverability."""
    parser = cli._build_parser()
    arguments = experiment_arguments(tmp_path / "experiment.csv")
    arguments[0] = "analyze-experiment"

    args = parser.parse_args(arguments)

    assert args.command == "analyze-experiment"


def test_unknown_metric_family_is_rejected(tmp_path: Path) -> None:
    """Family mappings cannot silently target undeclared metrics."""
    parser = cli._build_parser()
    arguments = experiment_arguments(tmp_path / "experiment.csv")
    family_index = arguments.index("conversion_rate=conversion")
    arguments[family_index] = "unknown=conversion"
    args = parser.parse_args(arguments)

    with pytest.raises(ValueError, match="undeclared"):
        cli._build_experiment_config(args)


def test_experiment_command_runs_end_to_end(tmp_path: Path) -> None:
    """The CLI produces both report formats for a real experiment run."""
    csv_path = tmp_path / "experiment.csv"
    pd.DataFrame(
        {
            "user_id": range(8),
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "converted": [0, 0, 1, 1, 0, 1, 1, 1],
            "revenue": [10.0, 11.0, 12.0, 13.0, 11.0, 13.0, 14.0, 15.0],
        }
    ).to_csv(csv_path, index=False)

    exit_code = cli.main(experiment_arguments(csv_path))
    report_dir = csv_path.parent / "artifacts"

    assert exit_code == 0
    assert (report_dir / "report.md").exists()
    parsed = json.loads((report_dir / "report.json").read_text())
    assert parsed["config"]["multiplicity"] == "fdr_bh"
    assert parsed["config"]["multiplicity_scope"] == "global"
    assert all(
        metric["simultaneous_ci_method"] == "bonferroni_for_fdr"
        for metric in parsed["metrics"]
    )


def test_experiment_command_dispatches_config_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command loads CSV data and passes configured families downstream."""
    csv_path = tmp_path / "experiment.csv"
    pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "variant": ["control", "control", "treatment", "treatment"],
            "converted": [0, 1, 1, 1],
            "revenue": [10.0, 11.0, 12.0, 13.0],
        }
    ).to_csv(csv_path, index=False)
    captured: dict[str, object] = {}

    def fake_analyze(data: pd.DataFrame, config: object) -> SimpleNamespace:
        captured["data"] = data
        captured["config"] = config
        return SimpleNamespace(
            experiment_id="experiment",
            status="ok",
            analysis_rows=len(data),
        )

    monkeypatch.setattr(cli, "analyze_experiment", fake_analyze)
    monkeypatch.setattr(
        cli,
        "write_experiment_report",
        lambda result, output: (
            Path(output) / "report.md",
            Path(output) / "report.json",
        ),
    )

    exit_code = cli.main(experiment_arguments(csv_path))

    assert exit_code == 0
    config = captured["config"]
    assert isinstance(config, ExperimentConfig)
    assert config.multiplicity == "fdr_bh"
    assert config.multiplicity_scope == "global"
    assert [metric.family for metric in config.metrics] == [
        "conversion",
        "engagement",
    ]
    assert "Experiment experiment: status=ok" in capsys.readouterr().out
