# Exhaustive auditable two-sample mean-difference hypothesis testing procedure

## Plan Overview

### Scope

A **notebook-driven statistical procedure** for testing the difference in means
between **two independent samples**. The procedure runs an **exhaustive battery**
of hypothesis tests spanning three paradigms — frequentist parametric,
frequentist non-parametric, and Bayesian — together with assumption
diagnostics and effect-size estimation. Every method is backed by an
**academic framework** with an explicit citation; no heuristic or
unjustified shortcut is permitted.

The deliverable is:
1. A Python helper package (`twosample_means/`) implementing each method as a
   small, documented, single-responsibility function.
2. A **procedure template notebook** (`notebooks/two_sample_mean_test.ipynb`)
   that data scientists run per experiment: configure inputs -> load data ->
   run diagnostics -> run the full battery -> emit an auditable report.

### Approach

- **Report-only, no auto-decision.** The procedure computes and reports all
  evidence (test statistics, p-values, Bayes factors, posteriors, HDIs, effect
  sizes, confidence intervals, assumption-check outcomes) but **never** applies
  an accept/reject decision. This is the strongest anti-p-hacking stance: the
  analyst interprets pre-registered evidence rather than chasing a gated goal.
- **Both file and in-memory input.** The runner accepts CSV/parquet paths or
  in-memory array-likes, validates them, and records a data hash for
  provenance.
- **Standard auditable report.** Per-test results with citations and
  assumption outcomes, plus a summary table, emitted as Markdown (with a JSON
  sidecar for machine consumption).
- **UV for dependency management** (project rule). Pinned versions.

### Strategy

Build foundational layers first (scaffold, config, data I/O, citations
registry), then assumption diagnostics, then the three test paradigms in
parallel-friendly milestones, then effect sizes, then the reporting/runner
layer, and finally the notebook that ties everything together. Each milestone
is independently testable; milestone validators auto-inject at each gate.

### Out of scope

- Paired/repeated-measures tests (independent samples only).
- One-sample vs reference tests.
- Power analysis / a-priori sample-size planning (testing battery only).
- Automatic decision rules / gating logic.
- GUI / web interface.

## Expected Functionality (Milestones)

### M1 — Foundation
- `feat-foundation-001`: UV project scaffold, package skeleton, `pyproject.toml`
- `feat-foundation-002`: Config dataclasses + academic citations registry
- `feat-foundation-003`: Data I/O (file + in-memory), validation, hashing

### M2 — Assumption Diagnostics
- `feat-assumptions-001`: Normality tests (Shapiro-Wilk, Anderson-Darling, D'Agostino K^2)
- `feat-assumptions-002`: Variance homogeneity (Levene, Bartlett, Brown-Forsythe) + outlier flagging

### M3 — Frequentist Parametric
- `feat-parametric-001`: Student's t, Welch's t, z-test (known variance)

### M4 — Frequentist Non-Parametric
- `feat-nonparametric-001`: Mann-Whitney U, Brunner-Munzel
- `feat-nonparametric-002`: Permutation test (exact + Monte Carlo) + bootstrap CI

### M5 — Bayesian
- `feat-bayesian-001`: BEST (Kruschke) via PyMC + HDI/ROPE reporting
- `feat-bayesian-002`: Bayes factor (JZS) via pingouin

### M6 — Effect Sizes
- `feat-effectsize-001`: Cohen's d, Hedges' g, Cliff's delta, rank-biserial, Hodges-Lehmann

### M7 — Reporting & Runner
- `feat-report-001`: Result schema + Markdown report writer + summary table
- `feat-runner-001`: Runner orchestrator (runs full battery, collects results)

### M8 — Notebook Procedure
- `feat-notebook-001`: Procedure template notebook

## Environment Setup

- **Runtime**: Python >=3.11
- **Dependency manager**: UV (project rule). `uv sync` to install.
- **Core deps**: numpy, scipy, statsmodels, pandas, pyarrow (parquet),
  pymc, arviz, pingouin, matplotlib (notebook plots), jupyter.
- **Dev deps**: pytest, pytest-cov, ruff, mypy, nbmake (notebook test).
- **Setup**: `bash init.sh` runs `uv sync` and a baseline pytest.

## Infrastructure

- **No network services.** Pure library + notebook. No ports bound.
- **No external services called at runtime.** All computation is local.
- **Off-limits**: no writes outside the project root; no `~/.config`,
  no system Python installs, no global pip installs (UV only).

## Testing Strategy

- **Unit tests** per module in `tests/` (pytest). Each method tested against
  scipy/reference implementations on fixed-seed synthetic data with known
  properties (e.g., normal vs skewed, equal vs unequal variance).
- **Property/sanity tests**: results are finite, CIs bracket the point
  estimate, p-values in [0,1], HDIs in data range.
- **Notebook test**: `nbmake` executes the template notebook end-to-end on
  bundled sample data and asserts a report file is produced.
- **Typecheck**: mypy strict on the package. **Lint**: ruff.
- **Milestone validators** (auto-injected) run test/typecheck/lint and review.

## Non-Functional Requirements

- **Auditable**: every result carries method name, citation, assumptions
  checked + outcomes, statistic, p-value/BF/posterior, effect size, CI/HDI,
  random seed, software versions, data hash.
- **Reproducible**: all stochastic methods accept a `seed`; default seed fixed.
- **Documented**: every public function has a docstring citing its academic
  source and stating assumptions. No magic values; config-driven.
- **No abstractions over rigor**: thin wrappers over proven implementations
  (scipy, statsmodels, PyMC, pingouin). No bespoke statistical algorithms.
- **PEP 8**, 79-char lines, snake_case, type hints, single-responsibility
  functions (project rules).
- **Performance**: Bayesian MCMC is the only heavy step; sampling defaults
  are conservative (e.g., 2000 draws, 2 chains) and configurable.
