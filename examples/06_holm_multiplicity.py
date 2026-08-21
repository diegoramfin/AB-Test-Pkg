"""Example 06: family-scoped Holm correction across same-family metrics.

Generates a checkout experiment with two monetization metrics: revenue per
user (continuous, primary) and purchase rate (binary, secondary). The
primary metric alone is significant at ``alpha = 0.05``, but Holm correction
across the monetization family doubles its p-value above the bar and widens
its interval until it includes zero. This is the point of family-wise
control: declaring two metrics in one family means you cannot claim either
effect at the unadjusted level.

Run directly:

    uv run python examples/06_holm_multiplicity.py \
        artifacts/examples/holm-multiplicity
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.ab_testing import (
    ContinuousMetricResult,
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
)
from twosample_means.reporting import write_experiment_report


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    rng = np.random.default_rng(606)
    n = 12_000
    arm = np.arange(n) % 2  # 0 = control, 1 = treatment
    revenue = 20.0 + 0.24 * arm + rng.normal(0.0, 5.0, size=n)
    control_rate = 0.20
    treatment_rate = 0.20 + 0.001
    converted = np.where(
        arm == 1,
        rng.random(n) < treatment_rate,
        rng.random(n) < control_rate,
    ).astype(int)
    frame = pd.DataFrame(
        {
            "user_id": range(n),
            "variant": np.where(arm == 1, "treatment", "control"),
            "revenue": revenue,
            "converted": converted,
        }
    )
    config = ExperimentConfig(
        experiment_id="06-holm-multiplicity",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        expected_allocation={"control": 0.5, "treatment": 0.5},
        metrics=(
            MetricSpec(
                "revenue_per_user",
                "revenue",
                "continuous",
                role="primary",
                family="monetization",
            ),
            MetricSpec(
                "purchase_rate",
                "converted",
                "binary",
                role="secondary",
                family="monetization",
            ),
        ),
        multiplicity="holm",
    )
    result = analyze_experiment(frame, config)
    primary = result.metrics[0]
    secondary = result.metrics[1]
    assert isinstance(primary, ContinuousMetricResult)
    assert primary.adjusted_p_value is not None
    assert primary.ci_lower is not None
    assert primary.ci_upper is not None
    assert primary.simultaneous_ci_lower is not None
    assert primary.simultaneous_ci_upper is not None
    assert primary.simultaneous_ci_level is not None
    write_experiment_report(result, output)
    print(
        f"Revenue per user: raw p={primary.p_value:.4f} -> Holm "
        f"{primary.adjusted_p_value:.4f}; 95% CI [{primary.ci_lower:.3f}, "
        f"{primary.ci_upper:.3f}] -> "
        f"{primary.simultaneous_ci_level * 100:.1f}% simultaneous "
        f"[{primary.simultaneous_ci_lower:.3f}, "
        f"{primary.simultaneous_ci_upper:.3f}]"
    )
    print(
        f"Purchase rate: raw p={secondary.p_value:.4f} -> Holm "
        f"{secondary.adjusted_p_value:.4f}"
    )
    return Path(output)


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/holm-multiplicity")
    )
    print(f"Reports written to {run(target)}")
