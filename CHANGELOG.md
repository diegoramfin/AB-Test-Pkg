# Changelog

All notable changes to `twosample-means` are documented here.

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
