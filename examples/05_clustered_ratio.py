"""Example 05: store-clustered revenue-per-order ratio metric.

Generates a synthetic experiment randomized at the store level (each store
is entirely control or treatment) where within-store price variation
inflates the revenue-per-order influence values. Users are the analysis
unit, so the cluster-robust standard error correctly widens the interval
compared with the user-level (naive) delta method, and writes Markdown,
HTML, and JSON reports.

Run directly:

    uv run python examples/05_clustered_ratio.py \
        artifacts/examples/clustered-ratio
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
    rng = np.random.default_rng(505)
    stores = 120
    users_per_store = 15
    base_price = 20.0
    # Store-level price spread of ~1.5 USD keeps the between-arm store noise
    # (sd ~1.5 * sqrt(2 / 60)) well below the injected 5% lift, while the
    # shared within-store price still inflates the ratio influence values.
    store_price_sd = 1.5
    treatment_price_lift = 0.05

    rows: list[dict[str, object]] = []
    for store in range(stores):
        arm = "control" if store % 2 == 0 else "treatment"
        price_level = base_price + store_price_sd * rng.normal(0.0, 1.0)
        if arm == "treatment":
            price_level *= 1.0 + treatment_price_lift
        for unit in range(users_per_store):
            rate = 1.0 if arm == "control" else 1.15
            orders = float(rng.poisson(rate)) + 1.0
            price = price_level + rng.normal(0.0, 1.0)
            rows.append(
                {
                    "user_id": f"{store}-{unit}",
                    "variant": arm,
                    "store_id": f"store-{store}",
                    "orders": orders,
                    "revenue": orders * price,
                }
            )
    frame = pd.DataFrame(rows)

    config = ExperimentConfig(
        experiment_id="05-clustered-ratio",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        expected_allocation={"control": 0.5, "treatment": 0.5},
        cluster="store_id",
        metrics=(
            MetricSpec(
                "revenue_per_order",
                "revenue_per_order",
                "ratio",
                role="primary",
                numerator="revenue",
                denominator="orders",
                practical_effect=0.5,
                family="monetization",
            ),
        ),
        multiplicity="holm",
    )
    result = analyze_experiment(frame, config)
    metric = result.metrics[0]
    assert isinstance(metric, RatioMetricResult)
    assert metric.cluster_robust is True
    write_experiment_report(result, output)
    print(
        f"Revenue per order effect {metric.absolute_effect:.4f} "
        f"(p={metric.p_value:.4g}); cluster-robust SE "
        f"{metric.standard_error:.4f} vs naive "
        f"{metric.naive_standard_error:.4f} "
        f"({metric.clusters} clusters, df {metric.degrees_of_freedom:.0f})"
    )
    return Path(output)


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/clustered-ratio")
    )
    print(f"Reports written to {run(target)}")
