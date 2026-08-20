"""Tests for terminal Kaggle retrieval and explicit output paths."""

from pathlib import Path

import pytest

from twosample_means import kaggle
from twosample_means.__main__ import _build_parser, _build_spec, main


def test_fetch_reuses_complete_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete cache avoids invoking the Kaggle CLI."""
    for filename in kaggle.MARKETING_CAMPAIGN.expected_files:
        (tmp_path / filename).write_text("metric\n1\n", encoding="utf-8")

    monkeypatch.setattr(
        "twosample_means.kaggle.subprocess.run",
        lambda *args, **kwargs: pytest.fail("Kaggle CLI should not run"),
    )

    files = kaggle.fetch_dataset("marketing-campaign-ab", tmp_path)

    assert files == (
        tmp_path / "control_group.csv",
        tmp_path / "test_group.csv",
    )


def test_fetch_requires_kaggle_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clear error is raised when the optional CLI is unavailable."""
    monkeypatch.setattr(
        "twosample_means.kaggle.shutil.which",
        lambda _: None,
    )

    with pytest.raises(kaggle.KaggleFetchError, match="Kaggle CLI"):
        kaggle.fetch_dataset("marketing-campaign-ab", tmp_path)


def test_fetch_command_reports_cached_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The fetch command delegates to the registered dataset adapter."""
    files = (tmp_path / "control_group.csv", tmp_path / "test_group.csv")
    monkeypatch.setattr(
        "twosample_means.__main__.fetch_dataset",
        lambda dataset, output: files,
    )

    exit_code = main(
        [
            "fetch",
            "marketing-campaign-ab",
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert "Dataset cache" in capsys.readouterr().out


def test_analysis_defaults_to_scalable_methods() -> None:
    """The CLI avoids expensive methods unless explicitly requested."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "analyze",
            "data.csv",
            "--group-col",
            "group",
            "--value-col",
            "value",
            "--group-a",
            "control",
            "--group-b",
            "treatment",
            "--output",
            "artifacts/test",
        ]
    )
    assert args.full_battery is False
    assert args.include_bayesian is False
    assert args.include_resampling is False


def test_analysis_output_is_required() -> None:
    """Every terminal analysis must declare an artifact directory."""
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "analyze",
                "data.csv",
                "--group-col",
                "group",
                "--value-col",
                "value",
                "--group-a",
                "control",
                "--group-b",
                "treatment",
            ]
        )


def test_legacy_cli_rejects_ambiguous_input_sources() -> None:
    """The legacy command does not silently prefer one input mode."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "analyze",
            "data.csv",
            "--csv-a",
            "control.csv",
            "--csv-b",
            "treatment.csv",
            "--output",
            "artifacts/test",
        ]
    )

    with pytest.raises(SystemExit):
        _build_spec(args)


def test_legacy_cli_reports_invalid_input_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """User-facing validation errors return code 2 and concise output."""
    exit_code = main(
        [
            "analyze",
            str(tmp_path / "missing.csv"),
            "--alpha",
            "2",
            "--output",
            str(tmp_path / "report"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("Error: alpha must be in (0, 1)")
    assert "Traceback" not in captured.err
