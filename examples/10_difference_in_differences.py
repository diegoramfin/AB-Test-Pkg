"""Example 10: difference in differences with event-study diagnostics.

Builds a synthetic store panel (two pre, two post periods) with
region-level treatment assignment and region-by-period shocks, then runs
the canonical DiD interaction model with cluster-robust standard errors.
The event study recovers null pre-period coefficients and a post-period
treatment effect, and the parallel-trends placebo p-value confirms the
pre-period trends are parallel. The identifying assumptions are listed
explicitly in the written report.

Run directly:

    uv run python examples/10_difference_in_differences.py \\
        artifacts/examples/did
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.quasi_experimental import (
    DifferenceInDifferences,
    render_did_markdown,
)


def _synthetic_panel(seed: int = 902) -> pd.DataFrame:
    """Region-randomized store panel with a known +4.0 revenue effect."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    regions = 24
    stores_per_region = 10
    for region in range(regions):
        treated = region % 2 == 0
        for store in range(stores_per_region):
            unit_fe = rng.normal(0.0, 3.0)
            for period, time_effect in enumerate((0.0, 0.6, 1.2, 1.8)):
                post = 1 if period >= 2 else 0
                region_time_shock = rng.normal(0.0, 1.5)
                revenue = (
                    unit_fe
                    + time_effect
                    + region_time_shock
                    + (4.0 if treated and post else 0.0)
                    + rng.normal(0.0, 0.4)
                )
                rows.append(
                    {
                        "store_id": region * stores_per_region + store,
                        "region": f"r{region}",
                        "period": period,
                        "treated": int(treated),
                        "post": post,
                        "revenue": revenue,
                    }
                )
    return pd.DataFrame(rows)


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)

    frame = _synthetic_panel()
    model = DifferenceInDifferences(
        outcome="revenue",
        unit="store_id",
        time="period",
        treated="treated",
        post="post",
        cluster="region",
    )
    result = model.fit(frame)

    event_study = result.event_study
    pre_trend = (
        event_study.pre_trend_p_value if event_study is not None else None
    )
    print(
        f"DiD revenue effect {result.effect:.3f} (true 4.0); "
        f"cluster-robust SE {result.standard_error:.3f} vs naive "
        f"{result.naive_standard_error:.3f}; p={result.p_value:.4g}"
    )
    print(
        f"Parallel-trends placebo p-value: {pre_trend:.4f}"
        if pre_trend is not None
        else "not estimable"
    )

    report = target / "did_report.md"
    report.write_text(render_did_markdown(result), encoding="utf-8")
    return target


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/did")
    )
    print(f"DiD report written to {run(target) / 'did_report.md'}")
