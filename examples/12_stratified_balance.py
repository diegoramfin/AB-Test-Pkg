"""Example 12: stratified randomization and pre-treatment covariate balance.

Generates a two-region experiment whose marginal allocation is exactly
50/50 but whose per-region allocation is off in opposite directions, so
the within-stratum sample-ratio mismatch check catches what the marginal
check misses. Also reports standardized mean differences for two
pre-treatment covariates, flagging the one that is imbalanced.

Run directly:

    uv run python examples/12_stratified_balance.py \
        artifacts/examples/stratified-balance
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
)
from twosample_means.reporting import write_experiment_report


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    rng = np.random.default_rng(1201)
    region_rows = []
    # Region 'north' skews control-heavy; region 'south' skews
    # treatment-heavy. Together they cancel to an exact 50/50 marginal
    # allocation, which a marginal SRM test would call fine.
    for region, control_count, treatment_count in (
        ("north", 70, 30),
        ("south", 30, 70),
    ):
        for _ in range(control_count):
            region_rows.append((region, "control"))
        for _ in range(treatment_count):
            region_rows.append((region, "treatment"))
    frame = pd.DataFrame(region_rows, columns=["region", "variant"])

    n = len(frame)
    frame["user_id"] = range(n)
    # Balanced covariate: the same values alternate across arms, so the
    # SMD is zero by construction.
    pre_spend = rng.normal(40.0, 6.0, n // 2)
    frame["pre_spend"] = np.concatenate([pre_spend, pre_spend])
    # Imbalanced covariate: treatment units are shifted by half an SD.
    frame["tenure_days"] = rng.normal(0.0, 1.0, n)
    frame.loc[frame["variant"] == "treatment", "tenure_days"] += 0.5
    # Outcome: small conversion lift in treatment.
    control_rate = 0.035
    treatment_rate = 0.040
    frame["converted"] = np.where(
        frame["variant"] == "treatment",
        rng.random(n) < treatment_rate,
        rng.random(n) < control_rate,
    ).astype(int)
    frame["revenue"] = frame["pre_spend"] + rng.normal(0.0, 4.0, n)

    config = ExperimentConfig(
        experiment_id="12-stratified-balance",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        strata="region",
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
            MetricSpec(
                "revenue_per_user",
                "revenue",
                "continuous",
                role="secondary",
                family="monetization",
                covariate="pre_spend",
            ),
            MetricSpec(
                "revenue_per_user_adj",
                "revenue",
                "continuous",
                role="secondary",
                family="monetization",
                covariate="tenure_days",
            ),
        ),
        multiplicity="holm",
    )
    result = analyze_experiment(frame, config)
    write_experiment_report(result, output)

    diagnostics = result.assignment_diagnostics
    print(
        f"Marginal SRM p={diagnostics.sample_ratio_mismatch_p_value:.4g}; "
        f"per-stratum SRM: "
        + ", ".join(
            f"{entry.stratum} p={entry.p_value:.4g}"
            for entry in diagnostics.stratum_srm
        )
    )
    for entry in diagnostics.covariate_balance:
        smd = "n/a" if entry.smd is None else f"{entry.smd:.4f}"
        flagged = " (flagged)" if entry.exceeds_threshold else ""
        print(f"  SMD {entry.covariate}: {smd}{flagged}")
    return Path(output)


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/stratified-balance")
    )
    print(f"Reports written to {run(target)}")
