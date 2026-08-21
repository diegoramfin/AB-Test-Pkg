"""Example 11: the Kaggle dataset manifest adapter workflow.

Shows how a fetched Kaggle cache is consumed: the manifest records the
source, license, aggregation level, expected unit semantics, and quality
flag; the analysis reads those declarations before touching the data.
The ``landing-page-ab`` registry entry is flagged ``teaching-sample`` and
prints a warning on fetch, so this example demonstrates the full flow
with a local stand-in cache (the bundled demo CSV) instead of requiring
network access or Kaggle credentials.

Run directly:

    uv run python examples/11_kaggle_manifest_adapter.py \\
        artifacts/examples/kaggle-adapter
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
    diagnose_assignment,
)
from twosample_means.kaggle import get_dataset_manifest

_DEMO_SOURCE = (
    Path(__file__).resolve().parents[1] / "data" / "demos" / "marketing_AB.csv"
)


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    cache = target / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    manifest = get_dataset_manifest("landing-page-ab")
    lines = [
        f"Manifest: {manifest.name}",
        f"  source: {manifest.source}",
        f"  license: {manifest.license}",
        f"  aggregation level: {manifest.aggregation_level}",
        f"  expected unit semantics: {manifest.expected_unit_semantics}",
        f"  expected files: {', '.join(manifest.expected_files)}",
        f"  unit column: {manifest.unit_column}",
        f"  assignment column: {manifest.assignment_column}",
        f"  quality flag: {manifest.quality}",
        f"  quality notes: {manifest.quality_notes}",
    ]
    if manifest.quality != "verified":
        print(
            f"Note: {manifest.name} is flagged '{manifest.quality}'. "
            f"{manifest.quality_notes}"
        )

    # Stand in for a fetched cache: copy the bundled user-row demo data and
    # write the manifest sidecar exactly as the fetch workflow does.
    shutil.copyfile(_DEMO_SOURCE, cache / "ab_data.csv")
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines.append(f"  cache prepared at: {cache}")

    # Consume the cache guided by the manifest's declared contract: the
    # stand-in demo file is renamed onto the manifest's expected columns.
    frame = pd.read_csv(cache / "ab_data.csv").rename(
        columns={"user id": "user_id", "test group": "group"}
    )
    config = ExperimentConfig(
        experiment_id="kaggle-manifest-adapter",
        unit_id=manifest.unit_column or "user_id",
        assignment=manifest.assignment_column or "group",
        control="psa",
        treatments=("ad",),
        metrics=(
            MetricSpec(
                "conversion_rate",
                "converted",
                "binary",
                role="primary",
            ),
        ),
        multiplicity="none",
        unit_type="user",
    )
    diagnostics = diagnose_assignment(frame, config)
    result = analyze_experiment(frame, config)
    metric = result.metrics[0]
    lines.extend(
        [
            "",
            f"Analysis: {result.analysis_rows} user rows, "
            f"hash {result.data_hash[:12]}...",
            f"  SRM evaluated: {diagnostics.sample_ratio_mismatch_evaluated}",
            f"  conversion lift: {metric.absolute_effect:.6f} "
            f"(p={metric.p_value:.4g})",
        ]
    )
    summary = target / "kaggle_adapter.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/kaggle-adapter")
    )
    print(f"Adapter summary written to {run(target) / 'kaggle_adapter.txt'}")
