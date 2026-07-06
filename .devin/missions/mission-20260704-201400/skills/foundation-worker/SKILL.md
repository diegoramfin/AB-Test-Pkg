---
name: foundation-worker
description: Builds project scaffold, config dataclasses, citations registry, and data I/O for the two-sample mean-difference testing procedure.
allowed-tools:
  - read
  - edit
  - write
  - grep
  - glob
  - exec
permissions:
  allow:
    - Exec(uv *)
    - Write(twosample_means/**)
    - Write(tests/**)
    - Write(pyproject.toml)
    - Write(notebooks/sample_data/**)
---

# Foundation Worker Procedure

## Your Task
You are implementing feature: $ARGUMENTS

## Setup
1. Read `<missionDir>/mission.md` for mission context.
2. Read `<missionDir>/AGENTS.md` for mission boundaries and rules.
3. Read `<missionDir>/architecture.md` for the package layout.
4. Read your feature's `fulfills` assertions from
   `<missionDir>/validation-contract.md`.
5. Run `uv run pytest -q` to confirm the starting state (may be no tests yet).

## Work Procedure
1. Use UV only for dependency management. Pin all versions explicitly in
   `pyproject.toml` (no `latest`, `*`, or unbounded `>=`). Core deps:
   numpy, scipy, statsmodels, pandas, pyarrow, pymc, arviz, pingouin,
   matplotlib, jupyter. Dev deps: pytest, pytest-cov, ruff, mypy, nbmake.
2. Create the `twosample_means/` package with every module listed in
   `architecture.md`. Modules not yet implemented must be valid stubs
   with a module docstring and a `"""Not yet implemented."""` placeholder
   so `import twosample_means` works.
3. For config: implement `RunConfig` (alpha, ci_level, hdi_mass,
   rope_width, mcmc_draws, mcmc_chains, permutation_iterations,
   bootstrap_iterations, seed) and `InputSpec` (sample_a/sample_b as
   paths or array-likes, plus optional column names) as frozen dataclasses
   with documented defaults. No magic values elsewhere.
4. For citations: implement a `CITATIONS` dict mapping every battery
   method name to a complete academic reference (authors, year, title,
   journal/source). Cover: shapiro, anderson_darling, dagostino_k2,
   levene, bartlett, brown_forsythe, students_t, welch_t, z_test,
   mann_whitney, brunner_munzel, permutation, bootstrap_ci, best,
   bayes_factor_jzs, cohen_d, hedges_g, cliff_delta, rank_biserial,
   hodges_lehmann.
5. For data_io: implement `load(spec: InputSpec) -> LoadedData` that
   reads CSV (pandas) and parquet (pyarrow/pandas), accepts in-memory
   array-likes, validates (non-empty, numeric, finite, min sample size
   >= 2), raises typed errors (e.g., `DataValidationError`) with clear
   messages, and computes a deterministic SHA-256 hash of the raw bytes
   for provenance. Return a dataclass with the two arrays and the hash.
6. Write unit tests in `tests/` covering each implemented piece against
   fixed-seed synthetic data.

## Verification (REQUIRED before finishing)
1. `uv sync` succeeds.
2. `uv run pytest -q` passes.
3. `uv run mypy twosample_means` passes with no errors.
4. `uv run ruff check twosample_means tests` passes.
5. Verify each `fulfills` assertion's pass criteria is met.

## Completion
When done, update your feature status:
```bash
python3 ~/.config/devin/scripts/mission_utils.py update-feature <missionDir> <featureId> --status completed
```
Append to the progress log:
```bash
python3 ~/.config/devin/scripts/mission_utils.py append-progress <missionDir> worker_completed --feature-id <featureId> --success-state success
```
If blocked, set status to `blocked` and write a handoff item to
`<missionDir>/handoffs/`.
