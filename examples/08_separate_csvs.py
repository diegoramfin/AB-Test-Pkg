"""Example 08: separate control and treatment CSV ingestion.

Writes a control file and a treatment file (the shape you get from an
export where each arm lives in its own CSV, e.g. a Kaggle cache), then
loads them with ``load_separate_experiment_csvs``. The assignment column
is synthesized from the file role, so the source files do not need to
carry a variant label.

Run directly:

    uv run python examples/08_separate_csvs.py artifacts/examples/separate-csvs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
    load_separate_experiment_csvs,
)
from twosample_means.reporting import write_experiment_report


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    target = Path(output)
    data_dir = target / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(808)
    n = 10_000
    control = pd.DataFrame(
        {
            "user_id": range(0, n // 2),
            "revenue": rng.normal(25.0, 8.0, size=n // 2),
        }
    )
    treatment = pd.DataFrame(
        {
            "user_id": range(n // 2, n),
            "revenue": rng.normal(26.5, 8.0, size=n // 2),
        }
    )
    control_path = data_dir / "control.csv"
    treatment_path = data_dir / "treatment.csv"
    control.to_csv(control_path, index=False)
    treatment.to_csv(treatment_path, index=False)

    config = ExperimentConfig(
        experiment_id="08-separate-csvs",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        expected_allocation={"control": 0.5, "treatment": 0.5},
        metrics=(
            MetricSpec(
                "revenue",
                "revenue",
                "continuous",
                role="primary",
                family="revenue",
            ),
        ),
        multiplicity="holm",
    )
    frame = load_separate_experiment_csvs(
        control_path,
        treatment_path,
        config,
    )
    result = analyze_experiment(frame, config)
    metric = result.metrics[0]
    write_experiment_report(result, target)
    print(
        f"Revenue effect {metric.absolute_effect:.4f} "
        f"(p={metric.p_value:.4g}); loaded {len(frame)} rows from "
        f"{control_path.name} and {treatment_path.name}"
    )
    return target


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/separate-csvs")
    )
    print(f"Reports written to {run(target)}")
