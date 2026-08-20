"""Smoke tests for the installed wheel and console script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pandas as pd
import pytest


class WheelRuntime:
    """Paths and environment for a wheel installed in a temporary venv."""

    def __init__(
        self,
        python: Path,
        executable: Path,
        environment: dict[str, str],
        working_directory: Path,
    ) -> None:
        self.python = python
        self.executable = executable
        self.environment = environment
        self.working_directory = working_directory

    def run(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Run the installed console script outside the repository root."""
        return subprocess.run(
            [str(self.executable), *arguments],
            cwd=self.working_directory,
            env=self.environment,
            text=True,
            capture_output=True,
            check=False,
        )


@pytest.fixture(scope="session")
def wheel_runtime(tmp_path_factory: pytest.TempPathFactory) -> WheelRuntime:
    """Build and install the wheel and all dependencies in a clean venv.

    Set ``TWOSAMPLE_MEANS_WHEEL_OFFLINE=1`` to force dependency resolution
    from uv's local cache instead of contacting package indexes.
    """
    repository = Path(__file__).resolve().parents[1]
    workspace = tmp_path_factory.mktemp("wheel-cli")
    wheel_directory = workspace / "wheel"
    wheel_directory.mkdir()
    build = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(wheel_directory),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    wheels = tuple(wheel_directory.glob("*.whl"))
    assert len(wheels) == 1

    offline = os.environ.get("TWOSAMPLE_MEANS_WHEEL_OFFLINE") == "1"
    requirements = workspace / "requirements.txt"
    export_command = [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-hashes",
        "--output-file",
        str(requirements),
    ]
    if offline:
        export_command.append("--offline")
    export = subprocess.run(
        export_command,
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert export.returncode == 0, export.stderr

    environment_directory = workspace / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(environment_directory)
    python = _venv_path(environment_directory, "python")
    install_command = ["uv", "pip", "install"]
    if offline:
        install_command.append("--offline")
    install_command.extend(
        [
            "--python",
            str(python),
            "--requirement",
            str(requirements),
            str(wheels[0]),
        ]
    )
    install = subprocess.run(
        install_command,
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("VIRTUAL_ENV", None)
    import_check = subprocess.run(
        [
            str(python),
            "-c",
            "import twosample_means; print(twosample_means.__file__)",
        ],
        cwd=workspace,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert import_check.returncode == 0, import_check.stderr
    assert str(environment_directory) in import_check.stdout
    return WheelRuntime(
        python=python,
        executable=_venv_path(environment_directory, "twosample-means"),
        environment=environment,
        working_directory=workspace,
    )


def test_built_wheel_exposes_legacy_cli(
    wheel_runtime: WheelRuntime,
    tmp_path: Path,
) -> None:
    """The installed wheel runs the legacy two-sample command."""
    control_path = tmp_path / "control.csv"
    treatment_path = tmp_path / "treatment.csv"
    pd.DataFrame({"outcome": [1.0, 2.0, 3.0, 4.0]}).to_csv(
        control_path,
        index=False,
    )
    pd.DataFrame({"outcome": [2.0, 3.0, 4.0, 5.0]}).to_csv(
        treatment_path,
        index=False,
    )
    output = tmp_path / "legacy-report"

    result = wheel_runtime.run(
        [
            "analyze",
            "--csv-a",
            str(control_path),
            "--col-a",
            "outcome",
            "--csv-b",
            str(treatment_path),
            "--col-b",
            "outcome",
            "--output",
            str(output),
        ]
    )

    assert result.returncode == 0, result.stderr
    assert (output / "report.md").exists()
    assert (output / "report.json").exists()


def test_built_wheel_runs_experiment_cli(
    wheel_runtime: WheelRuntime,
    tmp_path: Path,
) -> None:
    """The installed wheel runs a binary experiment and writes JSON output."""
    data_path = tmp_path / "experiment.csv"
    pd.DataFrame(
        {
            "user_id": range(8),
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "converted": [0, 0, 1, 1, 0, 1, 1, 1],
        }
    ).to_csv(data_path, index=False)
    output = tmp_path / "experiment-report"

    result = wheel_runtime.run(
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
            "--multiplicity-scope",
            "global",
            "--output",
            str(output),
        ]
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "experiment-result-v1"
    assert report["config"]["multiplicity_scope"] == "global"
    assert report["metrics"][0]["metric_name"] == "conversion_rate"


def _venv_path(environment: Path, name: str) -> Path:
    """Return a platform-appropriate executable path in a virtualenv."""
    if sys.platform == "win32":
        suffix = ".exe" if name != "python" else ".exe"
        return environment / "Scripts" / f"{name}{suffix}"
    return environment / "bin" / name
