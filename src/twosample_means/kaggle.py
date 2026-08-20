"""Optional Kaggle dataset retrieval for terminal workflows.

This module invokes the user-installed Kaggle CLI. Authentication remains in
Kaggle's normal user-level configuration and is never read, written, or logged
by this package.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KaggleDataset:
    """A Kaggle dataset supported by the terminal fetch workflow.

    Attributes
    ----------
    dataset_id:
        Kaggle owner/slug identifier.
    expected_files:
        Files expected after the Kaggle archive is extracted.
    """

    dataset_id: str
    expected_files: tuple[str, ...]


MARKETING_CAMPAIGN = KaggleDataset(
    dataset_id="amirmotefaker/ab-testing-dataset",
    expected_files=("control_group.csv", "test_group.csv"),
)

DATASETS: dict[str, KaggleDataset] = {
    "marketing-campaign-ab": MARKETING_CAMPAIGN,
}


class KaggleFetchError(RuntimeError):
    """Raised when the optional Kaggle download cannot complete."""


def fetch_dataset(dataset_name: str, output_dir: Path) -> tuple[Path, ...]:
    """Download a supported Kaggle dataset into a local cache directory.

    The user must configure Kaggle authentication independently with the
    Kaggle CLI. Existing complete downloads are reused without a network call.

    Parameters
    ----------
    dataset_name:
        Registered dataset name accepted by the CLI.
    output_dir:
        Directory where Kaggle extracts the dataset files.

    Returns
    -------
    tuple[Path, ...]
        Paths to the expected downloaded files.

    Raises
    ------
    KaggleFetchError
        If the dataset is unknown, the Kaggle CLI is unavailable, or its
        download command fails.
    """
    dataset = DATASETS.get(dataset_name)
    if dataset is None:
        supported = ", ".join(sorted(DATASETS))
        message = f"Unknown dataset '{dataset_name}'."
        raise KaggleFetchError(f"{message} Supported datasets: {supported}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = tuple(output_dir / name for name in dataset.expected_files)
    if all(path.is_file() for path in files):
        return files

    executable = shutil.which("kaggle")
    if executable is None:
        raise KaggleFetchError(
            "Kaggle CLI is unavailable. Install the optional dependency with "
            "'uv sync --extra kaggle' and configure Kaggle authentication."
        )

    command = [
        executable,
        "datasets",
        "download",
        dataset.dataset_id,
        "--path",
        str(output_dir),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise KaggleFetchError(
            "Kaggle CLI is unavailable. Install the optional dependency with "
            "'uv sync --extra kaggle' and configure Kaggle authentication."
        ) from error
    except subprocess.CalledProcessError as error:
        raise KaggleFetchError(
            "Kaggle download failed. Configure Kaggle authentication and "
            "confirm access to the requested dataset."
        ) from error

    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise KaggleFetchError(
            "Kaggle download completed but expected files are missing: "
            f"{', '.join(missing)}."
        )
    return files
