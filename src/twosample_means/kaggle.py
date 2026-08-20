"""Optional Kaggle dataset retrieval and dataset manifests.

This module invokes the user-installed Kaggle CLI. Authentication remains in
Kaggle's normal user-level configuration and is never read, written, or logged
by this package. Registry metadata is descriptive: users must verify the
source license and unit semantics before using a dataset for causal analysis.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetManifest:
    """Provenance and unit-semantics metadata for a registered dataset."""

    name: str
    source: str
    license: str
    aggregation_level: str
    expected_unit_semantics: str
    expected_files: tuple[str, ...]
    unit_column: str | None = None
    assignment_column: str | None = None
    quality: str = "verified"
    quality_notes: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible manifest metadata."""
        return asdict(self)


@dataclass(frozen=True)
class KaggleDataset:
    """A Kaggle dataset supported by the terminal fetch workflow."""

    dataset_id: str
    expected_files: tuple[str, ...]
    manifest: DatasetManifest


MARKETING_CAMPAIGN = KaggleDataset(
    dataset_id="amirmotefaker/ab-testing-dataset",
    expected_files=("control_group.csv", "test_group.csv"),
    manifest=DatasetManifest(
        name="marketing-campaign-ab",
        source="https://www.kaggle.com/datasets/amirmotefaker/ab-testing-dataset",
        license="Kaggle dataset page; verify before redistribution",
        aggregation_level="campaign-day",
        expected_unit_semantics=(
            "One row is a daily aggregate for a campaign arm, not an "
            "individual user."
        ),
        expected_files=("control_group.csv", "test_group.csv"),
        unit_column="Date",
        assignment_column="file role (control_group/test_group)",
    ),
)

LANDING_PAGE_AB = KaggleDataset(
    dataset_id="zhangluyuan/ab-testing",
    expected_files=("ab_data.csv",),
    manifest=DatasetManifest(
        name="landing-page-ab",
        source="https://www.kaggle.com/datasets/zhangluyuan/ab-testing",
        license="Kaggle dataset page; verify before redistribution",
        aggregation_level="user-row",
        expected_unit_semantics=(
            "One row per user with assignment and conversion fields; original "
            "row counts imply an unusually high and implausible baseline "
            "conversion rate, and duplicate-user contamination has been "
            "reported publicly for this dataset."
        ),
        expected_files=("ab_data.csv",),
        unit_column="user_id",
        assignment_column="group",
        quality="teaching-sample",
        quality_notes=(
            "Known to contain contaminated/duplicated users; use only as a "
            "pipeline teaching example, never as evidence about landing-page "
            "effects."
        ),
    ),
)

DATASETS: dict[str, KaggleDataset] = {
    dataset.manifest.name: dataset
    for dataset in (MARKETING_CAMPAIGN, LANDING_PAGE_AB)
}


class KaggleFetchError(RuntimeError):
    """Raised when the optional Kaggle download cannot complete."""


def get_dataset_manifest(dataset_name: str) -> DatasetManifest:
    """Return the registered manifest for a dataset name."""
    dataset = DATASETS.get(dataset_name)
    if dataset is None:
        supported = ", ".join(sorted(DATASETS))
        raise KaggleFetchError(
            f"Unknown dataset '{dataset_name}'. Supported datasets: "
            f"{supported}."
        )
    return dataset.manifest


def fetch_dataset(dataset_name: str, output_dir: Path) -> tuple[Path, ...]:
    """Download a supported Kaggle dataset into a local cache directory.

    The user must configure Kaggle authentication independently with the
    Kaggle CLI. Existing complete downloads are reused without a network call.
    A ``manifest.json`` sidecar is written alongside downloaded files so source
    and unit assumptions travel with the cache.
    """
    dataset = DATASETS.get(dataset_name)
    if dataset is None:
        supported = ", ".join(sorted(DATASETS))
        message = f"Unknown dataset '{dataset_name}'."
        raise KaggleFetchError(f"{message} Supported datasets: {supported}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = tuple(output_dir / name for name in dataset.expected_files)
    if not all(path.is_file() for path in files):
        executable = shutil.which("kaggle")
        if executable is None:
            raise KaggleFetchError(
                "Kaggle CLI is unavailable. Install the optional dependency "
                "with 'uv sync --extra kaggle' and configure Kaggle "
                "authentication."
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
                "Kaggle CLI is unavailable. Install the optional dependency "
                "with 'uv sync --extra kaggle' and configure Kaggle "
                "authentication."
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
    _write_manifest(output_dir, dataset.manifest)
    return files


def _write_manifest(output_dir: Path, manifest: DatasetManifest) -> None:
    """Write a deterministic manifest sidecar for a cached dataset."""
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
