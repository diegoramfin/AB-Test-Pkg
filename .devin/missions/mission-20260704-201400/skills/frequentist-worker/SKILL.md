---
name: frequentist-worker
description: Implements frequentist parametric and non-parametric two-sample tests (t-tests, z-test, Mann-Whitney, Brunner-Munzel, permutation, bootstrap CI).
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
    - Write(twosample_means/frequentist_parametric.py)
    - Write(twosample_means/frequentist_nonparametric.py)
    - Write(tests/test_frequentist_parametric.py)
    - Write(tests/test_frequentist_nonparametric.py)
---

# Frequentist Worker Procedure

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
1. Implement `twosample_means/frequentist_parametric.py`:
   - `students_t(a, b, config)` — thin wrapper over
     `scipy.stats.ttest_ind(equal_var=True)`.
   - `welch_t(a, b, config)` — `scipy.stats.ttest_ind(equal_var=False)`.
   - `z_test(a, b, config)` — analytical z-test using population
     variance(s) supplied via config (`config.population_variance_a/b`
     or a single known variance). If variance not supplied, raise
     `MissingVarianceError` (no silent default).
   - Each returns statistic, p-value, dof (where applicable), CI at
     `config.ci_level`, citation, assumption notes. NO decision.
2. Implement `twosample_means/frequentist_nonparametric.py`:
   - `mann_whitney_u(a, b, config)` — `scipy.stats.mannwhitneyu`.
   - `brunner_munzel(a, b, config)` — `scipy.stats.brunnermunzel`.
   - `permutation_test(a, b, config)` — exact mode for small n
     (exhaustive enumeration), Monte Carlo for large n using
     `config.permutation_iterations` and `config.seed`. Return
     statistic, p-value, mode, iterations, seed, citation.
   - `bootstrap_ci(a, b, config)` — bootstrap CI for mean difference
     using `config.bootstrap_iterations`, `config.ci_level`,
     `config.seed`. Return point estimate, CI bounds, iterations, seed.
3. Every public function: docstring with academic citation + assumptions.
   All stochastic methods accept/use `config.seed` and are reproducible.
4. Write tests comparing to scipy reference implementations on
   fixed-seed synthetic data; test exact permutation against manual
   enumeration on a tiny example; test reproducibility of Monte Carlo
   and bootstrap given the same seed.

## Verification (REQUIRED before finishing)
1. `uv run pytest -q tests/test_frequentist_parametric.py tests/test_frequentist_nonparametric.py` passes.
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
