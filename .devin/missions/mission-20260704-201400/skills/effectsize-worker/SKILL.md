---
name: effectsize-worker
description: Implements effect-size estimators — Cohen's d, Hedges' g, Cliff's delta, rank-biserial, Hodges-Lehmann — with CIs and citations.
allowed-tools:
  - read
  - edit
  - write
  - grep
  - glob
  - exec
permissions:
  allow:
    - Exec(uv run pytest)
    - Exec(uv run mypy *)
    - Exec(uv run ruff *)
    - Write(twosample_means/effect_size.py)
    - Write(tests/test_effect_size.py)
---

# Effect Size Worker Procedure

## Your Task
You are implementing feature: $ARGUMENTS

## Setup
1. Read `<missionDir>/mission.md`, `<missionDir>/AGENTS.md`, and
   `<missionDir>/architecture.md`.
2. Read your feature's `fulfills` assertions from
   `<missionDir>/validation-contract.md`.
3. Read `twosample_means/config.py` and `twosample_means/citations.py`.
4. Run `uv run pytest -q` to confirm the starting state.

## Work Procedure
1. Implement `twosample_means/effect_size.py`:
   - `cohens_d(a, b, config)` — pooled-SD standardized mean difference.
   - `hedges_g(a, b, config)` — Cohen's d with the small-sample bias
     correction factor J = 1 - 3/(4*df - 1).
   - `cliff_delta(a, b, config)` — probability that a random draw from
     a exceeds one from b, minus its complement; in [-1, 1].
   - `rank_biserial(a, b, config)` — rank-biserial correlation.
   - `hodges_lehmann(a, b, config)` — location-shift estimator with CI
     via `scipy.stats.hodgeslehmann` (or manual Walsh averages).
   - Each returns point estimate, CI at `config.ci_level`, citation.
2. Prefer `pingouin.compute_effsize` for Cohen's d / Hedges' g / Cliff's
   delta where it provides the reference implementation; otherwise use
   scipy. Do NOT implement bespoke formulas when a proven library
   function exists.
3. Every public function: docstring with academic citation + assumptions.
4. Write `tests/test_effect_size.py`: compare to `pingouin` and
   `scipy.stats.hodgeslehmann` to tolerance; assert Hedges' g differs
   from Cohen's d by the correction; Cliff's delta in [-1,1]; CIs
   bracket point estimates.

## Verification (REQUIRED before finishing)
1. `uv run pytest -q tests/test_effect_size.py` passes.
2. `uv run mypy twosample_means` passes.
3. `uv run ruff check twosample_means tests` passes.
4. Verify each `fulfills` assertion's pass criteria is met.

## Completion
```bash
python3 ~/.config/devin/scripts/mission_utils.py update-feature <missionDir> <featureId> --status completed
python3 ~/.config/devin/scripts/mission_utils.py append-progress <missionDir> worker_completed --feature-id <featureId> --success-state success
```
If blocked, set status to `blocked` and write a handoff to
`<missionDir>/handoffs/`.
