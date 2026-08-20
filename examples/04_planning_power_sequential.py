"""Example 04: power, MDE, and calibrated sequential looks.

Demonstrates the planning APIs: simulation-based power and minimum
detectable effect for a binary metric, plus correlation-aware
group-sequential boundaries for a predeclared look schedule. Writes a
short planning summary to the output directory.

Run directly:

    uv run python examples/04_planning_power_sequential.py \
        artifacts/examples/planning
"""

from __future__ import annotations

from pathlib import Path

from twosample_means.ab_testing import (
    PowerSpec,
    SequentialPlan,
    alpha_spending_boundaries,
    estimate_mde,
    evaluate_sequential,
    simulate_power,
)


def run(output: str | Path) -> Path:
    """Run the example and return the output directory."""
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)

    power_spec = PowerSpec(
        kind="binary",
        control=0.03,
        effect=0.004,
        sample_size_control=20_000,
        sample_size_treatment=20_000,
        replications=300,
    )
    power = simulate_power(power_spec)
    mde = estimate_mde(power_spec, target_power=0.8, max_effect=0.01)

    plan = SequentialPlan(
        (0.5, 0.75, 1.0),
        alpha=0.05,
        method="obrien_fleming",
        two_sided=True,
    )
    boundaries = alpha_spending_boundaries(plan)
    decided = evaluate_sequential(
        plan, [boundaries[0].z_boundary + 0.1, 0.0, 0.0]
    )

    lines = [
        f"Power at effect 0.4pp: {power.power:.3f}",
        f"Minimum detectable effect at 80% power: {mde:.5f}",
        "",
        "Calibrated group-sequential boundaries (O'Brien-Fleming spending):",
    ]
    for boundary in boundaries:
        lines.append(
            f"  look {boundary.look} @ info "
            f"{boundary.information_fraction:.2f}: "
            f"z = {boundary.z_boundary:.3f} "
            f"(cumulative alpha {boundary.cumulative_alpha:.4f})"
        )
    lines.append(f"Boundary crossing at first look: status={decided.status}")
    summary = target / "planning.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return target


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/examples/planning")
    )
    print(f"Planning summary written to {run(target) / 'planning.txt'}")
