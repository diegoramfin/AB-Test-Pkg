"""Example 01: binary conversion-rate experiment.

Generates a synthetic two-arm conversion experiment, estimates the
treatment-minus-control conversion lift, and writes Markdown, HTML, and
JSON reports.

Run directly:

    uv run python examples/01_binary_conversion.py \
        artifacts/examples/conversion
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.ab_testing import (
    BinaryMetricResult,
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
)
from twosample_means.reporting import write_experiment_report


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    rng = np.random.default_rng(101)
    n = 20_000
    control_rate = 0.030
    treatment_rate = 0.034
    frame = pd.DataFrame(
        {
            "user_id": range(n),
            "variant": np.where(np.arange(n) % 2 == 0, "control", "treatment"),
            "converted": np.where(
                np.arange(n) % 2 == 0,
                rng.random(n) < control_rate,
                rng.random(n) < treatment_rate,
            ).astype(int),
        }
    )
    config = ExperimentConfig(
        experiment_id="01-binary-conversion",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        expected_allocation={"control": 0.5, "treatment": 0.5},
        metrics=(
            MetricSpec(
                "conversion_rate",
                "converted",
                "binary",
                role="primary",
                practical_effect=0.001,
                family="conversion",
            ),
        ),
        multiplicity="holm",
    )
    result = analyze_experiment(frame, config)
    metric = result.metrics[0]
    assert isinstance(metric, BinaryMetricResult)
    assert metric.control.rate is not None
    assert metric.treatment.rate is not None
    write_experiment_report(result, output)
    print(
        f"Control rate {metric.control.rate:.4%}; treatment rate "
        f"{metric.treatment.rate:.4%}; lift {metric.absolute_effect:.4%}; "
        f"95% CI [{metric.ci_lower:.4%}, {metric.ci_upper:.4%}]; "
        f"p={metric.p_value:.4g}"
    )
    return Path(output)


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/conversion")
    )
    print(f"Reports written to {run(target)}")
