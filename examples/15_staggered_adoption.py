"""Example 15: staggered-adoption DiD with an anticipation window.

Builds a synthetic panel where two regions roll out a treatment in
different periods (periods 2 and 3 of 0..5) while a third region is
never treated, and where outcomes already respond one period before
treatment is observed. The Callaway & Sant'Anna estimator reports the
group-time ATT(g, t) matrix with not-yet-treated comparison units, the
anticipation effect at relative time -1, group/calendar/event-time/
overall aggregations, and a parallel-trends placebo test on the clean
pre-treatment cells.

Run directly:

    uv run python examples/15_staggered_adoption.py \\
        artifacts/examples/staggered-adoption
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.quasi_experimental import (
    CallawaySantAnna,
    render_staggered_did_markdown,
)


def _synthetic_panel(seed: int = 1501) -> pd.DataFrame:
    """Region-rollout panel with +2.0 effect and one-period anticipation."""
    rng = np.random.default_rng(seed)
    periods = 6
    regions = 27
    stores_per_region = 8
    # Regions 0-8 never treated; 9-17 first treated at period 2; 18-26 at
    # period 3. Each region is a cluster, so rollouts are region-wide.
    rollout = {
        region: "never" if region < 9 else (2 if region < 18 else 3)
        for region in range(regions)
    }
    time_effects = np.asarray([0.4 * period for period in range(periods)])
    rows: list[dict[str, object]] = []
    for region, group in rollout.items():
        region_time_shock = rng.normal(0.0, 1.2, periods)
        for store in range(stores_per_region):
            store_fe = rng.normal(0.0, 2.5)
            for period in range(periods):
                treated = int(group != "never" and period >= int(group))
                anticipated = int(
                    group != "never" and period == int(group) - 1
                )
                revenue = (
                    store_fe
                    + time_effects[period]
                    + region_time_shock[period]
                    + 2.0 * treated
                    + 2.0 * anticipated
                    + rng.normal(0.0, 0.5)
                )
                rows.append(
                    {
                        "store_id": f"r{region}-s{store}",
                        "region": f"r{region}",
                        "period": period,
                        "cohort": group,
                        "revenue": revenue,
                    }
                )
    return pd.DataFrame(rows)


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)

    frame = _synthetic_panel()
    model = CallawaySantAnna(
        outcome="revenue",
        unit="store_id",
        time="period",
        group="cohort",
        anticipation=1,
        cluster="region",
    )
    result = model.fit(frame)

    event_by_time = {
        event.relative_time: event for event in result.event_time_atts
    }
    overall = result.overall_att
    print(
        f"Staggered ATT {overall.att:.3f} (true 2.0); "
        f"cluster-robust SE {overall.standard_error:.3f} vs naive "
        f"{overall.naive_standard_error:.3f}; p={overall.p_value:.4g}"
    )
    print(
        "Event-time ATT: "
        + ", ".join(
            f"{relative:+d}={event.att:.2f}"
            for relative, event in sorted(event_by_time.items())
        )
    )
    placebo = result.placebo
    print(
        f"Parallel-trends placebo p-value: {placebo.p_value:.4f}"
        if placebo is not None
        else "not estimable"
    )

    report = target / "staggered_did_report.md"
    report.write_text(render_staggered_did_markdown(result), encoding="utf-8")
    return target


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/staggered-adoption")
    )
    report_path = run(target) / "staggered_did_report.md"
    print(f"Staggered DiD report written to {report_path}")
