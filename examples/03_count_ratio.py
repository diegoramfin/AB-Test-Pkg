"""Example 03: count and ratio metrics in one experiment.

Generates a synthetic experiment with a count metric (orders per user) and
a ratio metric (revenue per order), and writes Markdown, HTML, and JSON
reports for both.

Run directly:

    uv run python examples/03_count_ratio.py artifacts/examples/count-ratio
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    RatioMetricResult,
    analyze_experiment,
)
from twosample_means.reporting import write_experiment_report


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    rng = np.random.default_rng(303)
    n = 8_000
    orders = (
        np.where(
            np.arange(n) % 2 == 0,
            rng.poisson(0.8, size=n),
            rng.poisson(1.1, size=n),
        ).astype(float)
        + 1.0
    )
    revenue = orders * rng.lognormal(mean=2.5, sigma=0.5, size=n)
    frame = pd.DataFrame(
        {
            "user_id": range(n),
            "variant": np.where(np.arange(n) % 2 == 0, "control", "treatment"),
            "orders": orders,
            "revenue": revenue,
        }
    )
    config = ExperimentConfig(
        experiment_id="03-count-ratio",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        metrics=(
            MetricSpec(
                "orders",
                "orders",
                "count",
                role="primary",
                family="engagement",
            ),
            MetricSpec(
                "revenue_per_order",
                "revenue_per_order",
                "ratio",
                role="secondary",
                numerator="revenue",
                denominator="orders",
                family="monetization",
            ),
        ),
        multiplicity="holm",
    )
    result = analyze_experiment(frame, config)
    orders_metric = result.metrics[0]
    ratio_metric = result.metrics[1]
    assert isinstance(ratio_metric, RatioMetricResult)
    write_experiment_report(result, output)
    print(
        f"Orders effect {orders_metric.absolute_effect:.4f} "
        f"(p={orders_metric.p_value:.4g}); revenue per order effect "
        f"{ratio_metric.absolute_effect:.4f} (p={ratio_metric.p_value:.4g})"
    )
    return Path(output)


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/count-ratio")
    )
    print(f"Reports written to {run(target)}")
