"""Example 09: evaluating a running experiment against calibrated boundaries.

Predeclares a two-look group-sequential plan (interim look at 50% of the
planned sample, final look at 100%), then evaluates the experiment as it
would run in practice: analyze the accumulated data at each look, convert
the primary metric's p-value to a z-statistic, and compare against the
correlation-aware alpha-spending boundaries. The interim look continues;
the final look crosses and stops.

Run directly:

    uv run python examples/09_sequential_analysis.py \
        artifacts/examples/sequential-analysis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    SequentialPlan,
    alpha_spending_boundaries,
    analyze_experiment,
    evaluate_sequential,
)


def _z_statistic(
    p_value: float | None,
    effect: float | None,
) -> float:
    """Two-sided z from a p-value, signed by the observed effect direction."""
    assert p_value is not None and 0.0 < p_value <= 1.0
    assert effect is not None
    magnitude = float(stats.norm.ppf(1.0 - p_value / 2.0))
    return magnitude if effect >= 0.0 else -magnitude


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    plan = SequentialPlan((0.5, 1.0), alpha=0.05, method="obrien_fleming")
    boundaries = alpha_spending_boundaries(plan)

    rng = np.random.default_rng(902)
    total = 40_000
    arm = np.arange(total) % 2
    converted = (rng.random(total) < np.where(arm == 1, 0.056, 0.050)).astype(
        int
    )
    frame = pd.DataFrame(
        {
            "user_id": range(total),
            "variant": np.where(arm == 1, "treatment", "control"),
            "converted": converted,
        }
    )
    config = ExperimentConfig(
        experiment_id="09-sequential-analysis",
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
                family="conversion",
            ),
        ),
        multiplicity="none",
    )

    look_sizes = (total // 2, total)
    z_statistics = []
    for look, size in enumerate(look_sizes, start=1):
        result = analyze_experiment(frame.iloc[:size], config)
        metric = result.metrics[0]
        z = _z_statistic(metric.p_value, metric.absolute_effect)
        z_statistics.append(z)
        print(
            f"Look {look} @ info {plan.information_fractions[look - 1]:.1f}: "
            f"z={z:.3f} vs boundary {boundaries[look - 1].z_boundary:.3f}"
        )
    decided = evaluate_sequential(plan, z_statistics)
    print(
        f"Decision: {decided.status}"
        + (f" at look {decided.crossed_look}" if decided.crossed_look else "")
    )

    lines = [
        "Calibrated group-sequential evaluation (O'Brien-Fleming spending):",
    ]
    for boundary in boundaries:
        lines.append(
            f"  look {boundary.look} @ info "
            f"{boundary.information_fraction:.2f}: "
            f"z = {boundary.z_boundary:.3f} "
            f"(cumulative alpha {boundary.cumulative_alpha:.4f})"
        )
    for look, (size, z) in enumerate(
        zip(look_sizes, z_statistics, strict=True)
    ):
        lines.append(f"  observed look {look + 1} (n={size}): z = {z:.3f}")
    lines.append(f"Decision: status={decided.status}")
    if decided.crossed_look is not None:
        lines.append(f"  crossed boundary at look {decided.crossed_look}")
    summary = target / "sequential.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/sequential-analysis")
    )
    print(f"Sequential summary written to {run(target) / 'sequential.txt'}")
