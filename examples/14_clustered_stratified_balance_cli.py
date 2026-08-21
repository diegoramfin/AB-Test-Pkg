"""Example 14: cluster, strata, and balance columns through the CLI.

The same design as example 13, but driven entirely through the
``twosample-means experiment`` terminal command instead of the Python
API: a store-level randomized experiment inside two regional strata is
written to a CSV, and the CLI is invoked with ``--cluster``,
``--strata``, and ``--balance-columns`` together. The report proves the
flags reach the analysis: cluster-robust SEs are wider than the naive
user-level intervals, per-stratum SRM is evaluated, and the balance
table mixes a CUPED covariate with a flagged balance-only column.

Run directly:

    uv run python examples/14_clustered_stratified_balance_cli.py \
        artifacts/examples/clustered-stratified-balance-cli
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _generate_frame() -> pd.DataFrame:
    """Build the store-level stratified experiment frame."""
    rng = np.random.default_rng(1401)
    stores_per_region = 60
    users_per_store = 12

    rows: list[dict[str, object]] = []
    for region in ("north", "south"):
        for store in range(stores_per_region):
            arm = "control" if store % 2 == 0 else "treatment"
            for unit in range(users_per_store):
                rows.append(
                    {
                        "user_id": f"{region}-{store}-{unit}",
                        "store_id": f"{region}-{store}",
                        "region": region,
                        "variant": arm,
                    }
                )
    frame = pd.DataFrame(rows)
    n = len(frame)
    treatment_mask = frame["variant"] == "treatment"

    # Balanced covariate: the same half-sample is assigned to each arm, so
    # the SMD is exactly zero by construction. It also predicts spend, so
    # CUPED reduces variance while cluster-robust SEs stay honest.
    pre_spend = rng.normal(40.0, 6.0, n // 2)
    frame.loc[~treatment_mask, "pre_spend"] = pre_spend
    frame.loc[treatment_mask, "pre_spend"] = pre_spend

    # Imbalanced balance-only column: treatment users are shifted by half an
    # SD. It is not used for adjustment, so --balance-columns is the only way
    # it appears in the report.
    frame["device_score"] = rng.normal(0.0, 1.0, n)
    frame.loc[treatment_mask, "device_score"] += 0.5

    # Store-level shocks make within-store outcomes correlated, so the
    # cluster-robust SE must exceed the naive user-level SE.
    store_shocks = np.repeat(
        rng.normal(0.0, 4.0, stores_per_region * 2), users_per_store
    )
    store_order_shocks = np.repeat(
        rng.normal(0.0, 0.3, stores_per_region * 2), users_per_store
    )
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
    return frame


def run(output: str | Path) -> Path:
    """Generate a CSV, run the CLI end-to-end, and return the report dir."""
    out = Path(output)
    with tempfile.TemporaryDirectory(prefix="example-14-") as tmp:
        csv_path = Path(tmp) / "experiment.csv"
        _generate_frame().to_csv(csv_path, index=False)

        command = [
            sys.executable,
            "-m",
            "twosample_means",
            "experiment",
            str(csv_path),
            "--unit-col",
            "user_id",
            "--assignment-col",
            "variant",
            "--control",
            "control",
            "--treatment",
            "treatment",
            "--metric",
            "spend_per_user=spend:continuous:primary",
            "--metric",
            "orders_per_user=orders:count:secondary",
            "--covariate",
            "spend_per_user=pre_spend",
            "--cluster",
            "store_id",
            "--strata",
            "region",
            "--balance-columns",
            "device_score",
            "--expected-allocation",
            "control=0.5,treatment=0.5",
            "--output",
            str(out),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "twosample-means experiment failed with exit code "
                f"{completed.returncode}:\n{completed.stderr}"
            )

    # Print the CLI output so running the example shows the same terminal
    # transcript the command produces.
    print(completed.stdout)
    return out


if __name__ == "__main__":
    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/clustered-stratified-balance-cli")
    )
    print(f"Reports written to {run(target)}")
