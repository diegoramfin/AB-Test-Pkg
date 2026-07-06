---
name: bayesian-worker
description: Implements Bayesian two-sample tests — BEST (Kruschke) via PyMC/ArviZ and the JZS Bayes factor via pingouin — with HDI/ROPE reporting and no decision logic.
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
    - Write(twosample_means/bayesian.py)
    - Write(tests/test_bayesian.py)
---

# Bayesian Worker Procedure

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
1. Implement `twosample_means/bayesian.py`:
   - `best(a, b, config)` — the BEST model (Kruschke 2013) via PyMC:
     Student-t likelihoods per group with shared or separate priors,
     sampling `config.mcmc_draws` draws across `config.mcmc_chains`
     chains with `config.seed`. Use ArviZ to compute the posterior of
     the mean difference, the HDI at `config.hdi_mass`, the ROPE
     proportion over `config.rope_width`, and MCMC diagnostics
     (`az.rhat`, `az.ess`). Return a result with posterior mean
     difference, HDI bounds, ROPE proportion, R-hat, ESS, draws, chains,
     seed, citation. NO decision string.
   - `bayes_factor_jzs(a, b, config)` — JZS Bayes factor via
     `pingouin.bayesfactor_ttest`, returning BF10, BF01, the prior
     width used (from config), and citation (Rouder et al. 2009).
2. Every public function: docstring with academic citation + assumptions.
   Reproducible given `config.seed`.
3. Write `tests/test_bayesian.py`: on bundled/fixed-seed sample data,
   assert posterior mean difference is finite and within data range,
   R-hat < 1.01, ESS > 400, ROPE proportion in [0,1], reproducibility
   given seed, BF10 finite and positive and matching pingouin to
   tolerance. Keep MCMC draws modest in tests (e.g., 1000 draws,
   2 chains) for speed but still meeting ESS threshold.

## Verification (REQUIRED before finishing)
1. `uv run pytest -q tests/test_bayesian.py` passes.
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
