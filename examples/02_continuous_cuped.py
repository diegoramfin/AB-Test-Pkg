"""Example 02: continuous revenue metric with CUPED variance reduction.

Generates a synthetic experiment where a pre-period covariate predicts
post-treatment revenue, then estimates the CUPED-adjusted treatment effect
and writes Markdown, HTML, and JSON reports.

Run directly:

    uv run python examples/02_continuous_cuped.py artifacts/examples/cuped
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.ab_testing import (
    CupedMetricResult,
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
)
from twosample_means.reporting import write_experiment_report


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    rng = np.random.default_rng(202)
    n = 12_000
    pre_spend = rng.gamma(2.0, 5.0, size=n)
    treatment_effect = 1.2
    revenue = (
        2.0 * pre_spend
        + rng.normal(0.0, 6.0, size=n)
        + np.where(np.arange(n) % 2 == 0, 0.0, treatment_effect)
    )
    frame = pd.DataFrame(
        {
            "user_id": range(n),
            "variant": np.where(np.arange(n) % 2 == 0, "control", "treatment"),
            "pre_spend": pre_spend,
            "revenue": revenue,
        }
    )
    config = ExperimentConfig(
        experiment_id="02-continuous-cuped",
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
                covariate="pre_spend",
                family="revenue",
            ),
        ),
        multiplicity="holm",
    )
    result = analyze_experiment(frame, config)
    metric = result.metrics[0]
    assert isinstance(metric, CupedMetricResult)
    write_experiment_report(result, output)
    print(
        f"Adjusted effect {metric.absolute_effect:.4f} (unadjusted "
        f"{metric.unadjusted_absolute_effect:.4f}); correlation "
        f"{metric.correlation:.3f}; variance reduction "
        f"{metric.variance_reduction:.1%}; p={metric.p_value:.4g}"
    )
    return Path(output)


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/cuped")
    )
    print(f"Reports written to {run(target)}")
