# twosample-means

An auditable terminal-first analysis engine for independent two-sample
continuous outcomes. It reports frequentist, non-parametric, Bayesian, and
effect-size estimates without making accept/reject decisions.

This is not a general A/B experimentation suite: it does not yet provide
power planning, sequential monitoring, or binary/count/ratio metrics beyond the
experiment-level API described below. Treat the legacy battery as sensitivity
analysis, not independent confirmatory evidence.

## Quick start: conversion-rate analysis

The bundled marketing example has a binary `converted` outcome. Analyze that
outcome as the primary metric instead of using `total ads`, which is an
exposure metric:

```bash
uv sync --all-extras

uv run twosample-means experiment data/demos/marketing_AB.csv \
  --unit-col "user id" --assignment-col "test group" \
  --control psa --treatment ad \
  --metric conversion_rate=converted:binary:primary \
  --metric-family conversion_rate=conversion \
  --multiplicity holm --multiplicity-scope global \
  --output artifacts/marketing-conversion
```

`--output` is required for every run. Each run directory receives `report.md`
and `report.json`; it is safe to retain for audit and comparison. Generated
artifacts are ignored by Git.

For the bundled data, the report estimates a control conversion rate of
1.785%, an ad-arm rate of 2.555%, and an absolute treatment-minus-control lift
of 0.769 percentage points (43.1% relative lift). The 95% interval is
[0.587, 0.936] percentage points. These are reported estimates, not an
accept/reject decision. Because this quick-start command does not declare the
intended allocation, its assignment report records that sample-ratio mismatch
was not evaluated; configure `expected_allocation` through the Python API when
that design information is available.

## Kaggle workflow

Kaggle retrieval is optional and uses the normal Kaggle CLI authentication
outside this repository. It never reads, writes, or reports credentials.

```bash
uv sync --extra kaggle
uv run twosample-means fetch marketing-campaign-ab \
  --output data/cache/marketing-campaign-ab

uv run twosample-means analyze \
  --csv-a data/cache/marketing-campaign-ab/control_group.csv \
  --col-a "Spend [USD]" \
  --csv-b data/cache/marketing-campaign-ab/test_group.csv \
  --col-b "Spend [USD]" --delimiter ";" \
  --output artifacts/marketing-spend
```

The initial fetch workflow supports Kaggle dataset
`amirmotefaker/ab-testing-dataset`. It reuses a complete local cache. Configure
Kaggle separately with its official CLI before fetching.

## CLI usage

```text
uv run twosample-means analyze [OPTIONS] [CSV_PATH]
uv run twosample-means experiment CSV_PATH [OPTIONS]
uv run twosample-means fetch marketing-campaign-ab --output CACHE_DIRECTORY
```

The legacy `analyze` command runs the low-level two-sample battery. For
experiment-level binary/continuous metrics, use `experiment` (also available
as `analyze-experiment`) with one row per randomization unit:

```bash
uv run twosample-means experiment data/checkout.csv \
  --unit-col user_id --assignment-col variant \
  --control control --treatment treatment \
  --metric conversion_rate=converted:binary:primary \
  --metric revenue=revenue:continuous:secondary \
  --metric-family conversion_rate=conversion \
  --metric-family revenue=engagement \
  --multiplicity holm --multiplicity-scope global \
  --output artifacts/checkout
```

Repeat `--metric` for each declared metric using
`NAME=COLUMN:KIND[:ROLE]`. `KIND` is `binary`, `continuous`, or `count`;
`ROLE` is `primary`, `secondary`, or `guardrail`. Ratio metrics require the
Python API because they declare numerator and denominator columns. Repeat
`--metric-family NAME=FAMILY`
to isolate correction families. `--multiplicity` accepts `none`, `holm`, or
`fdr_bh`; the selected method controls adjusted p-values and reported
simultaneous intervals. `--multiplicity-scope family` (the default) corrects
within each `--metric-family`; `--multiplicity-scope global` pools every
estimable metric into one correction family.

For two separate CSVs, pass `--csv-a`, `--col-a`, `--csv-b`, and `--col-b`.
For a single CSV with group labels, pass its path with `--group-col`,
`--value-col`, `--group-a`, and `--group-b`.

The CLI defaults to a scalable analytical run and skips Bayesian and
resampling methods. Use `--include-bayesian`, `--include-resampling`, or
`--full-battery` to enable the additional methods. Resource limits and skipped
methods are recorded in the report.

## Using the library

```python
from twosample_means.config import InputSpec, RunConfig
from twosample_means.data_io import load
from twosample_means.reporting import write_report
from twosample_means.runner import run

spec = InputSpec(sample_a=[1.0, 2.0, 3.0], sample_b=[2.0, 3.0, 4.0])
report = run(load(spec), RunConfig())
write_report(report, "artifacts/example")
```

## Experiment-level data contract

The `ab_testing` namespace adds an experiment plan and normalized user-level
input contract without changing the legacy two-sample API:

```python
from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
    diagnose_assignment,
    normalize_experiment_data,
)
from twosample_means.reporting import write_experiment_report

config = ExperimentConfig(
    experiment_id="checkout-copy",
    unit_id="user_id",
    assignment="variant",
    control="control",
    treatments=("treatment",),
    metrics=(
        MetricSpec(
            name="conversion_rate",
            column="converted",
            kind="binary",
            role="primary",
            practical_effect=0.002,
        ),
    ),
)
assignment_diagnostics = diagnose_assignment(dataframe, config)
normalized = normalize_experiment_data(dataframe, config)
experiment_result = analyze_experiment(dataframe, config)
write_experiment_report(experiment_result, "artifacts/checkout-copy")
```

Assignment diagnostics report unit duplication, cross-arm assignment, missing
or unknown labels, arm counts, and an optional sample-ratio mismatch p-value
when `expected_allocation` is configured. Normalization validates one row per
unit, assignment labels, metric types,
missingness, optional analysis windows, and computes a deterministic data
fingerprint. Missing metric values remain explicit so each estimator can
apply its declared policy.

Metric results are corrected independently within each declared `family`
by default. Set `multiplicity_scope="global"` on `ExperimentConfig` to pool
all estimable metrics into one correction family.
`multiplicity="holm"` adds Holm step-down simultaneous confidence intervals
and adjusted p-values. `multiplicity="fdr_bh"` adjusts p-values with
Benjamini-Hochberg FDR and adds conservative Bonferroni family-wise intervals;
BH itself controls FDR and does not define exact simultaneous intervals. The
nominal `ci_lower`/`ci_upper` fields remain available alongside
`simultaneous_ci_lower`/`simultaneous_ci_upper`.

### Extended metric and design APIs

- `kind="count"` estimates unit-level mean counts with Welch inference. It
  does not treat repeated event rows as independent Poisson observations.
- `kind="ratio"` requires `numerator="..."` and `denominator="..."` columns
  and estimates a ratio of user-level means with delta-method uncertainty.
  Denominators must be positive.
- `ContrastSpec` declares named treatment-vs-control or arbitrary arm
  contrasts for multi-arm experiments. Unspecified multi-arm analyses are
  rejected rather than silently generating comparisons.
- `PowerSpec`, `simulate_power()`, and `estimate_mde()` provide seeded,
  simulation-based planning for binary, continuous, count, and ratio metrics.
- `SequentialPlan`, `alpha_spending_boundaries()`, and
  `evaluate_sequential()` provide predeclared O'Brien–Fleming or Pocock
  alpha-spending looks. Boundaries use marginal normal quantiles and should be
  treated as a documented initial sequential contract.

Experiment JSON reports use the bundled `experiment-result-v1` JSON Schema.
The schema is versioned independently from the Python package release.

## What the battery includes

| Category | Methods |
|---|---|
| Diagnostics | Shapiro-Wilk, Anderson-Darling, D'Agostino K², Levene, Bartlett, Brown-Forsythe, IQR/z-score outlier flagging |
| Parametric | Student's t, Welch's t, z-test with known variance |
| Non-parametric | Mann-Whitney U, Brunner-Munzel, permutation test, bootstrap CI |
| Bayesian | BEST via PyMC with HDI/ROPE, JZS Bayes factor via pingouin |
| Effect sizes | Cohen's d, Hedges' g, Cliff's delta, rank-biserial, Hodges-Lehmann |

Every method includes an academic citation in the result.

## Verification

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/twosample_means tests
uv run pytest --cov --cov-report=term-missing
uv build

# Build/install the wheel and run CLI smoke tests with locked dependencies.
uv run pytest tests/test_wheel_cli.py
# Use uv's local cache only when network access is unavailable.
TWOSAMPLE_MEANS_WHEEL_OFFLINE=1 uv run pytest tests/test_wheel_cli.py
```

## Project layout

```text
src/twosample_means/  # library and terminal commands
data/demos/       # packaged, versioned demo data
tests/            # automated test suite
```

## Design principles

- No accept/reject decisions; analysts interpret the evidence.
- Every method has an academic citation.
- SHA-256 data hashing and configuration logging provide provenance.
- Outliers are flagged but never removed.
- The full battery is descriptive sensitivity analysis, not independent
  confirmatory evidence.
