"""Example 07: multi-arm experiment with predeclared planned contrasts.

Generates a three-arm conversion experiment (control, a discount variant,
and a loyalty variant) and declares explicit ``ContrastSpec`` comparisons:
each variant against control plus a direct discount-vs-loyalty contrast.
Holm correction runs across all three contrasts in the ``conversion``
family, so the discount arm's nominal significance flips after correction
while the loyalty arm's strong effect survives.

Run directly:

    uv run python examples/07_multi_arm_contrasts.py \
        artifacts/examples/multi-arm-contrasts
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.ab_testing import (
    ContrastSpec,
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
)
from twosample_means.reporting import write_experiment_report


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    rng = np.random.default_rng(707)
    n = 9_000
    arm = np.arange(n) % 3  # 0 = control, 1 = variant_a, 2 = variant_b
    rates = {0: 0.050, 1: 0.060, 2: 0.068}
    labels = {0: "control", 1: "variant_a", 2: "variant_b"}
    converted = np.where(
        rng.random(n) < np.array([rates[int(a)] for a in arm]), 1, 0
    ).astype(int)
    frame = pd.DataFrame(
        {
            "user_id": range(n),
            "variant": [labels[int(a)] for a in arm],
            "converted": converted,
        }
    )
    config = ExperimentConfig(
        experiment_id="07-multi-arm-contrasts",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("variant_a", "variant_b"),
        expected_allocation={
            "control": 1.0 / 3.0,
            "variant_a": 1.0 / 3.0,
            "variant_b": 1.0 / 3.0,
        },
        metrics=(
            MetricSpec(
                "conversion_rate",
                "converted",
                "binary",
                role="primary",
                family="conversion",
            ),
        ),
        contrasts=(
            ContrastSpec("discount_vs_control", "variant_a"),
            ContrastSpec("loyalty_vs_control", "variant_b"),
            ContrastSpec(
                "discount_vs_loyalty",
                "variant_a",
                control="variant_b",
            ),
        ),
        multiplicity="holm",
    )
    result = analyze_experiment(frame, config)
    write_experiment_report(result, output)
    for metric in result.metrics:
        print(
            f"{metric.contrast_name:22s} effect {metric.absolute_effect:+.5f} "
            f"raw p={metric.p_value:.4f} -> Holm {metric.adjusted_p_value:.4f}"
        )
    return Path(output)


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/multi-arm-contrasts")
    )
    print(f"Reports written to {run(target)}")
