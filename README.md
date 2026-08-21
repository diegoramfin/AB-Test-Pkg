# twosample-means

![coverage](coverage.svg)

An auditable terminal-first analysis engine for independent two-sample
outcomes and randomized experiments. It reports binary, continuous, count, and
ratio estimates plus frequentist, non-parametric, Bayesian, and effect-size
results without making accept/reject decisions.

This is not a general A/B experimentation suite: it does not provide
assignment generation, automatic causal validation, or guaranteed user-level
randomization. Treat the legacy battery as sensitivity analysis, not
independent confirmatory evidence. Difference-in-differences lives in a
separate `quasi_experimental` namespace because it rests on different
identifying assumptions.

## Installation

Requires **Python 3.11 or newer**.

```bash
pip install twosample-means
# or, with uv
uv add twosample-means
```

Optional extras:

```bash
pip install "twosample-means[kaggle]"   # Kaggle dataset retrieval
pip install "twosample-means[docs]"      # documentation site tooling
```

Check the installed version with `twosample-means --version`. The package
ships a PEP 561 `py.typed` marker, so type checkers get inline annotations
from the installed distribution.

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

The registry currently supports:

| Name | Dataset | Expected rows |
|---|---|---|
| `marketing-campaign-ab` | `amirmotefaker/ab-testing-dataset` | campaign-day aggregates |
| `landing-page-ab` | `zhangluyuan/ab-testing` | expected user rows; flagged `teaching-sample` |

Each cache includes `manifest.json` with source URL, license guidance,
aggregation level, expected unit semantics, quality flag, and expected
columns. Datasets flagged `teaching-sample` print a warning during fetch and
must not be used as evidence about product effects. The manifest is
descriptive and does not certify randomization or causal validity. Configure
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

When the intended assignment shares are known, declare them so the run
actually evaluates sample-ratio mismatch instead of recording that it was
not evaluated:

```bash
uv run twosample-means experiment data/checkout.csv ... \
  --expected-allocation control=0.5,treatment=0.5
```

CUPED variance reduction is available from the CLI for continuous and count
metrics with a pre-experiment column:

```bash
uv run twosample-means experiment data/checkout.csv \
  ... --metric revenue=revenue:continuous:primary \
  --covariate revenue=pre_period_spend
```

Cluster-robust standard errors are available through `--cluster COLUMN` for
continuous, count, and ratio metrics:

```bash
uv run twosample-means experiment data/checkout.csv \
  ... --metric arpu=revenue/orders:ratio:primary \
  --cluster store_id
```

Declared randomization strata enable within-stratum sample-ratio mismatch
checks, which catch offsetting imbalances a marginal test cannot see:

```bash
uv run twosample-means experiment data/checkout.csv \
  ... --strata region \
  --expected-allocation control=0.5,treatment=0.5
```

Covariates that are not used for adjustment can still be checked for
pre-treatment balance with `--balance-columns` (repeatable):

```bash
uv run twosample-means experiment data/checkout.csv \
  ... --balance-columns tenure --balance-columns device_score
```

Every generated `report.json` is validated against the bundled
`experiment-result-v1` JSON Schema before it is written, so rendered reports
cannot silently drift from the declared contract.

For a single experiment CSV, pass its path with `--unit-col`,
`--assignment-col`, `--control`, and `--treatment`. Separate control and
treatment CSVs are also supported; the CLI synthesizes the assignment column
from the file role:

```bash
uv run twosample-means experiment \
  --csv-a data/control.csv --csv-b data/treatment.csv \
  --unit-col user_id --assignment-col variant \
  --control control --treatment treatment \
  --metric orders=orders:count:primary \
  --metric revenue_per_order=revenue/orders:ratio:secondary \
  --unit-type user --output artifacts/separate-arms
```

Count metrics estimate unit-level mean counts with Welch inference. Ratio
metrics use `NUMERATOR/DENOMINATOR` columns and a user-level delta-method
ratio of means; denominators must be positive. Use `--unit-type aggregate` for
campaign/day or other pre-aggregated rows. The report will show a warning and
will not imply user-level causal validity. `--unit-type unknown` records an
explicit warning when the row semantics are not known.

For the legacy `analyze` command, missing values fail by default. Pass
`--missing-values exclude` to remove NaNs independently from each arm before
running the battery; infinite values remain invalid. For a single legacy CSV
with group labels, pass `--group-col`, `--value-col`, `--group-a`, and
`--group-b`.

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
or unknown labels, arm counts, and an optional sample-ratio mismatch p-value.
They also record the declared `unit_type`; aggregate and unknown unit types
produce explicit warnings because row-level standard errors do not establish
individual-user causal validity.
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
- `MetricSpec(covariate="pre_period_column")` enables variance reduction
  for continuous and count metrics. `covariate_method` selects the
  estimator: `"cuped"` (default; Deng et al., 2013) residualizes outcomes
  with the pooled slope and uses Welch inference; `"ancova"` fits
  `Y ~ treatment + X` with heteroskedasticity-robust standard errors;
  `"interaction"` adds a `treatment * X` term for arm-specific slopes
  with the effect at the mean covariate. Every covariate-adjusted report
  carries an explicit covariate leakage guard stating that temporal
  ordering is caller-declared and not verifiable from the data.
- `ExperimentConfig(cluster="cluster_column")` enables cluster-robust
  (Liang & Zeger, 1986) standard errors for continuous, count, and ratio
  metrics. The CR1 sandwich with a small-sample correction and `G-2`
  degrees of freedom replaces Welch inference; clusters spanning both arms
  produce an explicit warning. CUPED adjustment and clustering compose.
  Ratio metrics use the same influence-function delta method as the
  user-level estimator, with a cluster sandwich over the arm influence
  values combined in quadrature.
- `ContrastSpec` declares named treatment-vs-control or arbitrary arm
  contrasts for multi-arm experiments. Unspecified multi-arm analyses are
  rejected rather than silently generating comparisons.
- `PowerSpec`, `simulate_power()`, and `estimate_mde()` provide seeded,
  simulation-based planning for binary, continuous, count, and ratio metrics.
- `SequentialPlan`, `alpha_spending_boundaries()`, and
  `evaluate_sequential()` provide predeclared O'Brien–Fleming or Pocock
  alpha-spending looks. Boundaries are calibrated by recursive numerical
  quadrature over the canonical group-sequential joint distribution, so the
  family-wise error rate across all looks equals the declared alpha.
  `sequential_power()` reports power and average sample information under
  those boundaries. `marginal_alpha_spending_boundaries()` remains
  available for reference.
- `always_valid_confidence_sequence()` and
  `difference_confidence_sequence()` build time-uniform (always-valid)
  confidence sequences (Howard et al., 2021): every interval is valid
  simultaneously, so repeated peeking does not inflate the error rate.
- `write_experiment_report()` writes `report.html` alongside the Markdown
  and JSON files: a self-contained document with inline styling and no
  external assets.
- Assignment diagnostics include a pre-treatment covariate balance table:
  every declared metric covariate is compared against the control arm with a
  standardized mean difference (SMD) at the randomization-unit level, and
  |SMD| > 0.1 flags imbalance. `ExperimentConfig.balance_columns` (CLI
  `--balance-columns`, repeatable) adds columns to the balance check without
  using them for variance reduction. When `ExperimentConfig.strata` is set,
  the sample-ratio mismatch test also runs within each stratum and the
  report lists per-stratum p-values.
- The `twosample_means.quasi_experimental` namespace provides
  `DifferenceInDifferences` for the canonical two-group panel design with
  cluster-robust standard errors, panel and treatment-timing validation,
  event-study coefficients, and a parallel-trends placebo test.
  `render_did_markdown()` writes a report that lists the identifying
  assumptions explicitly.

## Examples

`examples/` contains runnable, tested workflows (each also a pytest smoke
case). Run one with:

```bash
uv run python examples/01_binary_conversion.py artifacts/examples/conversion
```

- `01_binary_conversion.py` — two-arm conversion-rate experiment with
  expected-allocation SRM and Holm correction.
- `02_continuous_cuped.py` — continuous revenue metric with CUPED variance
  reduction.
- `03_count_ratio.py` — count and ratio metrics in one experiment.
- `04_planning_power_sequential.py` — simulation power/MDE and calibrated
  sequential looks.
- `05_clustered_ratio.py` — store-clustered revenue-per-order ratio with
  cluster-robust delta-method inference.
- `06_holm_multiplicity.py` — two same-family monetization metrics where
  Holm correction flips the primary's nominal significance.
- `07_multi_arm_contrasts.py` — three-arm experiment with predeclared
  planned contrasts, including a direct variant-vs-variant comparison.
- `08_separate_csvs.py` — separate control/treatment CSV ingestion with
  synthesized assignment labels.
- `09_sequential_analysis.py` — evaluating a running experiment at planned
  looks against calibrated group-sequential boundaries.
- `10_difference_in_differences.py` — region-clustered store panel with an
  event study and a parallel-trends placebo test.
- `11_kaggle_manifest_adapter.py` — consuming a Kaggle cache through its
  manifest contract, including the teaching-sample quality warning.
- `12_stratified_balance.py` — two-region experiment whose marginal
  allocation passes SRM while both strata fail, plus a covariate balance
  table that flags the imbalanced covariate.
- `13_clustered_stratified_balance.py` — store-level randomization inside
  regional strata: cluster-robust SEs wider than naive, per-stratum SRM,
  and a balance table mixing a CUPED covariate with a balance-only column.
- `14_clustered_stratified_balance_cli.py` — the same design driven
  entirely through `twosample-means experiment` with `--cluster`,
  `--strata`, and `--balance-columns` on a generated CSV, proving the
  flags work end-to-end from the terminal.

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
uv run mypy src/twosample_means tests examples
uv run pytest --cov --cov-report=term-missing
uv build

# Build/install the wheel and run CLI smoke tests with locked dependencies.
uv run pytest tests/test_wheel_cli.py
# Use uv's local cache only when network access is unavailable.
TWOSAMPLE_MEANS_WHEEL_OFFLINE=1 uv run pytest tests/test_wheel_cli.py

# Documentation site (mkdocs).
uv run --extra docs mkdocs build --strict
```

## Project layout

```text
src/twosample_means/  # library and terminal commands
data/demos/       # packaged, versioned demo data
examples/         # runnable, smoke-tested example workflows
docs/             # documentation site (mkdocs)
tests/            # automated test suite
CONTRIBUTING.md   # contribution and quality-gate guidance
SECURITY.md       # vulnerability reporting policy
```

## Design principles

- No accept/reject decisions; analysts interpret the evidence.
- Every method has an academic citation.
- SHA-256 data hashing and configuration logging provide provenance.
- Outliers are flagged but never removed.
- The full battery is descriptive sensitivity analysis, not independent
  confirmatory evidence.
