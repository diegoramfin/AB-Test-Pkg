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
