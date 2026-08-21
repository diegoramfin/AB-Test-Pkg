# Changelog

All notable changes to `twosample-means` are documented here.

## [Unreleased]

### Added

- Explicit experiment `unit_type` semantics with aggregate/unknown warnings.
- Separate control/treatment CSV ingestion for experiment analyses.
- CLI count and ratio metric declarations.
- Configurable legacy NaN exclusion with strict error defaults.
- Kaggle dataset manifests and an additional landing-page A/B registry entry.
- CLI `--expected-allocation` flag that enables the sample-ratio mismatch test.
- Strict validation of rendered experiment JSON against the bundled schema.
- Dataset quality flags with fetch-time warnings for teaching-only datasets.
- CUPED variance reduction for continuous and count metrics with CLI
  `--covariate` support.
- Cluster-robust standard errors for continuous and count metrics with CLI
  `--cluster` support.
- Cluster-robust ratio metrics via the delta method: arm ratio influence
  values with cluster-sandwich variances combined in quadrature and `G-2`
  degrees of freedom.
- Store-clustered revenue-per-order example (`examples/05_clustered_ratio.py`)
  demonstrating cluster-robust ratio inference.
- Seeded coverage simulations for cluster-robust intervals (continuous and
  ratio): near-nominal coverage on clustered data while the naive user-level
  interval undercovers.
- PEP 561 `py.typed` marker shipped in the wheel and sdist so consumers get
  static type information from the installed package.
- `ExperimentConfig.balance_columns` (CLI `--balance-columns`, repeatable):
  pre-treatment columns checked for unit-level standardized-mean-difference
  balance without being used for variance reduction. Validated against
  reserved columns and metric covariates, and rendered in the report's
  covariate balance table.
- `examples/13_clustered_stratified_balance.py` combining store-level
  cluster-robust inference, regional strata, and balance-only columns in
  one experiment report.
- End-to-end FWER simulation for calibrated group-sequential boundaries: a
  seeded Monte Carlo test that runs the real experiment pipeline (binary
  estimator, p-value to signed z conversion, boundary comparison) under the
  null and verifies the empirical early-stopping rate matches the spent
  alpha.
- `twosample-means --version` flag reporting the installed package version.
- ANCOVA covariate adjustment on `MetricSpec`: `covariate_method="ancova"`
  fits `Y ~ treatment + X` with heteroskedasticity-robust standard errors;
  `"interaction"` adds a treatment-by-covariate term for arm-specific
  slopes. Every covariate-adjusted report now carries an explicit
  covariate leakage guard stating that temporal ordering is
  caller-declared.
- Always-valid (time-uniform) confidence sequences: the normal-mixture
  bound of Howard et al. (2021) for a running mean and a two-arm
  difference, valid at every look time so repeated peeking does not
  inflate the error rate.
- `sequential_power()`: Monte Carlo power and average sample information
  under the calibrated alpha-spending boundaries, consistent with the
  canonical group-sequential model.
- `twosample_means.quasi_experimental` namespace: canonical
  difference-in-differences with unit and period fixed effects,
  cluster-robust standard errors, panel and treatment-timing validation,
  time-varying covariate adjustment, event-study coefficients with an
  omitted reference period, and a parallel-trends placebo test.
- Examples `10_difference_in_differences.py` (region-clustered store panel
  with event-study diagnostics) and `11_kaggle_manifest_adapter.py`
  (manifest-driven cache consumption with quality-flag handling).
- Documentation site via mkdocs (`docs/`, `mkdocs.yml`) with a `docs`
  optional extra; built in CI with `mkdocs build --strict`.
- Stratified randomization support: `ExperimentConfig.strata` (CLI
  `--strata`) runs the sample-ratio mismatch test within each stratum, so
  offsetting per-stratum imbalances that a balanced marginal allocation
  hides are reported and flagged.
- Pre-treatment covariate balance diagnostics: every declared metric
  covariate gets a unit-level standardized mean difference against each
  treatment arm, with |SMD| > 0.1 flagged in the report, JSON, and schema.
- `examples/12_stratified_balance.py` demonstrating both new diagnostics.
- CI security workflow: gitleaks secret scan on push/PR and a dependency
  review gate on PRs.
- Coverage badge (`coverage.svg`) committed to the repo and verified
  current by a CI drift check on the Python 3.12 matrix leg.

### Fixed

- Corrected the normal-mixture confidence-sequence width (the bound is
  divided by the sample size, not the information squared) and the
  sequential-power definition (power is the boundary-crossing probability;
  the final-look mass counts toward average sample information only).
- Example demonstrating an active family-scoped Holm correction
  (`examples/06_holm_multiplicity.py`): a primary metric that is nominally
  significant but flips after correction.
- Examples covering multi-arm planned contrasts, separate-arm CSV
  ingestion, and live sequential-look evaluation
  (`examples/07_multi_arm_contrasts.py`, `08_separate_csvs.py`,
  `09_sequential_analysis.py`).

- Arbitrary contrasts mixed with implicit control contrasts no longer
  crash: rebuilding the comparison config for a swapped control drops the
  contrast list, which previously re-resolved `control=None` entries
  against the new control into self-comparisons.
- Correlation-aware group-sequential boundary calibration replacing
  marginal normal quantiles.
- Self-contained HTML experiment reports.
- Runnable `examples/` workflows with pytest smoke coverage.
- ``CONTRIBUTING.md`` and ``SECURITY.md`` release-hygiene documentation.

## [0.2.0] - 2026-08-19

### Added

- `src/` package layout with clean wheel-boundary imports.
- Count metrics analyzed as unit-level means with Welch inference.
- User-level ratio metrics with delta-method uncertainty.
- Seeded simulation-based power and MDE planning APIs.
- Alpha-spending sequential-look plans using O'Brien–Fleming or Pocock spending.
- Explicit `ContrastSpec` declarations for planned multi-arm contrasts.
- Versioned `experiment-result-v1` JSON Schema bundled in the wheel.
- Release metadata, citation information, and package URLs.

### Changed

- Removed the unused direct `statsmodels` dependency.
- Experiment reports now use the bundled schema-version constant.

## [0.1.0] - Initial release

- Auditable two-sample continuous-outcome battery.
- Binary and continuous experiment-level metric adapters.
- Assignment diagnostics and family/global multiplicity correction.
