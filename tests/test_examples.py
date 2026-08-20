"""Smoke tests for the bundled example scripts.

Examples are real, runnable workflows. These tests execute their ``run``
entry points against temporary directories so broken examples fail CI.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(
    "module_name",
    [
        "01_binary_conversion",
        "02_continuous_cuped",
        "03_count_ratio",
        "04_planning_power_sequential",
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
    if module_name == "04_planning_power_sequential":
        summary = output / "planning.txt"
        assert summary.exists()
        assert "calibrated" in summary.read_text(encoding="utf-8").lower()
    else:
        assert (output / "report.md").exists()
        assert (output / "report.json").exists()
        assert (output / "report.html").exists()
    assert capsys.readouterr().out != ""
