---
name: diagnostics-worker
description: Implements assumption diagnostics — normality tests, variance homogeneity tests, and outlier flagging — for the two-sample procedure.
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
    - Write(twosample_means/assumptions.py)
    - Write(tests/test_assumptions.py)
---

# Diagnostics Worker Procedure

## Your Task
You are implementing feature: $ARGUMENTS

## Setup
1. Read `<missionDir>/mission.md`, `<missionDir>/AGENTS.md`, and
   `<missionDir>/architecture.md`.
2. Read your feature's `fulfills` assertions from
   `<missionDir>/validation-contract.md`.
3. Read `twosample_means/config.py` and `twosample_means/citations.py`
   to use `RunConfig` and the citations registry.
4. Run `uv run pytest -q` to confirm the starting state.

## Work Procedure
1. Implement `twosample_means/assumptions.py` with single-responsibility
   functions, each a thin wrapper over `scipy.stats`:
   - `shapiro_wilk(x, ...)`, `anderson_darling(x, ...)`,
     `dagostino_k2(x, ...)` for normality of each sample.
   - `levene(a, b, ...)`, `bartlett(a, b, ...)`,
     `brown_forsythe(a, b, ...)` for variance homogeneity.
   - `flag_outliers(x, method="iqr"|"zscore", threshold=...)` returning
     indices of flagged points WITHOUT modifying the data.
2. Each function returns a result dataclass with: method name, statistic,
   p-value (where applicable), citation (from the registry), and an
   assumption outcome string ("met"/"not_met" based on config alpha) —
   but NO accept/reject decision about the main hypothesis.
3. Every public function has a docstring citing its academic source and
   stating assumptions. No magic values; thresholds come from `RunConfig`.
4. Write `tests/test_assumptions.py` testing on fixed-seed synthetic data:
   normal vs skewed (lognormal) for normality; equal vs unequal variance
   for homogeneity; injected outliers for flagging. Assert finite results,
   p-values in [0,1], and that outlier flagging does not mutate input.

## Verification (REQUIRED before finishing)
1. `uv run pytest -q tests/test_assumptions.py` passes.
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
