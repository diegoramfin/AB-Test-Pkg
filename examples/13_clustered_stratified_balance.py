"""Example 13: clustered, stratified experiment with balance-only columns.

Randomizes at the store level (each store is entirely control or
treatment) inside two regional strata. Because assignment follows the
store, cluster-robust standard errors are wider than the naive
user-level intervals, and the report shows the within-stratum SRM and a
covariate balance table that mixes a balanced adjustment covariate
(``pre_spend``, also used for CUPED) with a balance-only column
(``device_score``) that is imbalanced and would otherwise never be
checked.

Run directly:

    uv run python examples/13_clustered_stratified_balance.py \
        artifacts/examples/clustered-stratified-balance
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
    rng = np.random.default_rng(1301)
    stores_per_region = 60
    users_per_store = 15

    rows: list[dict[str, object]] = []
    for region in ("north", "south"):
        # Within each region the store assignment is exactly 50/50, so both
        # the marginal and the per-stratum sample-ratio checks pass.
        for store in range(stores_per_region):
            arm = "control" if store % 2 == 0 else "treatment"
            # Store-level shocks make within-store outcomes correlated, so
            # the cluster-robust SE must exceed the naive user-level SE.
            store_shock = rng.normal(0.0, 4.0)
            store_order_shock = rng.normal(0.0, 0.3)
            for unit in range(users_per_store):
                rows.append(
                    {
                        "user_id": f"{region}-{store}-{unit}",
                        "store_id": f"{region}-{store}",
                        "region": region,
                        "variant": arm,
                        # Filled below; kept in construction order so the
                        # balanced covariate can be assigned by concatenation.
                        "pre_spend": np.nan,
                        "device_score": np.nan,
                        "spend": np.nan,
                        "orders": np.nan,
                        "_store_shock": store_shock,
                        "_store_order_shock": store_order_shock,
                    }
                )
    frame = pd.DataFrame(rows)
    n = len(frame)
    treatment_mask = frame["variant"] == "treatment"

    # Balanced covariate: the same half-sample is assigned to each arm, so
    # the SMD is exactly zero by construction. It also predicts the outcome,
    # so CUPED reduces variance while cluster-robust SEs stay honest.
    pre_spend = rng.normal(40.0, 6.0, n // 2)
    frame.loc[~treatment_mask, "pre_spend"] = pre_spend
    frame.loc[treatment_mask, "pre_spend"] = pre_spend

    # Imbalanced balance-only column: treatment users are shifted by half an
    # SD. It is not used for adjustment, so balance_columns is the only way
    # it appears in the report.
    frame["device_score"] = rng.normal(0.0, 1.0, n)
    frame.loc[frame["variant"] == "treatment", "device_score"] += 0.5

    store_shocks = frame.pop("_store_shock").to_numpy()
    store_order_shocks = frame.pop("_store_order_shock").to_numpy()
    frame["spend"] = (
        25.0
        + 1.5 * treatment_mask.to_numpy(dtype=float)
        + store_shocks
        + 0.8 * (frame["pre_spend"].to_numpy() - 40.0)
        + rng.normal(0.0, 6.0, n)
    )
    frame["orders"] = rng.poisson(
        2.0 + 0.25 * treatment_mask.to_numpy(dtype=float) + store_order_shocks
    ).astype(float)

    config = ExperimentConfig(
        experiment_id="13-clustered-stratified-balance",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        expected_allocation={"control": 0.5, "treatment": 0.5},
        cluster="store_id",
        strata="region",
        balance_columns=("device_score",),
        metrics=(
            MetricSpec(
                "spend_per_user",
                "spend",
                "continuous",
                role="primary",
                practical_effect=0.5,
                family="monetization",
                covariate="pre_spend",
            ),
            MetricSpec(
                "orders_per_user",
                "orders",
                "count",
                role="secondary",
                family="engagement",
            ),
        ),
        multiplicity="holm",
    )
    result = analyze_experiment(frame, config)
    write_experiment_report(result, output)

    diagnostics = result.assignment_diagnostics
    print(
        f"Marginal SRM p={diagnostics.sample_ratio_mismatch_p_value:.4g}; "
        "per-stratum SRM: "
        + ", ".join(
            f"{entry.stratum} p={entry.p_value:.4g}"
            for entry in diagnostics.stratum_srm
        )
    )
    for entry in diagnostics.covariate_balance:
        smd = "n/a" if entry.smd is None else f"{entry.smd:.4f}"
        flagged = " (flagged)" if entry.exceeds_threshold else ""
        source = (
            "covariate" if entry.covariate == "pre_spend" else "balance-only"
        )
        print(f"  SMD {entry.covariate} [{source}]: {smd}{flagged}")

    primary = result.metrics[0]
    assert isinstance(primary, ContinuousMetricResult)
    assert primary.cluster_robust is True
    print(
        f"Spend effect {primary.absolute_effect:.4f} "
        f"(p={primary.p_value:.4g}); cluster-robust SE "
        f"{primary.standard_error:.4f} vs naive "
        f"{primary.naive_standard_error:.4f} "
        f"({primary.clusters} clusters, df {primary.degrees_of_freedom:.0f})"
    )
    return Path(output)


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/clustered-stratified-balance")
    )
    print(f"Reports written to {run(target)}")
