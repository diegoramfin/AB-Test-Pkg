# A/B Testing Package Review and Roadmap

Review date: July 24, 2026

## Executive assessment

The repository is a strong prototype for an auditable **independent
two-sample continuous-outcome analysis**, but it is not yet a general A/B
testing or statistical analysis suite.

The current implementation has several real strengths:

- Clear separation between configuration, data loading, statistical
  methods, orchestration, and reporting.
- Frozen configuration objects and deterministic random seeds.
- Input validation and SHA-256 data provenance.
- Frequentist, non-parametric, Bayesian, and effect-size outputs.
- Citations attached to methods.
- A meaningful automated test suite: 116 tests pass.
- No hardcoded API keys, Kaggle credentials, passwords, or common secret
  patterns were found in the current tracked source or notebook.

The main product risk is positioning. Calling the current package an
"exhaustive A/B testing suite" would overstate its scope. It presently
analyzes one continuous metric from two independent groups and runs a
large battery of overlapping tests. A publishable A/B suite needs an
experiment specification, explicit estimands, metric types, assignment
diagnostics, power analysis, multiplicity control, variance reduction,
and decision-oriented reporting.

**Recommendation:** preserve the existing package as the continuous
two-sample engine, then build a higher-level `ab_testing` API around it.
Do not rewrite the working statistical functions before the experiment
and metric abstractions are designed.

## Validation results

| Check | Result | Notes |
|---|---:|---|
| Unit/integration tests | Pass | 116 passed in 145.52 seconds |
| Secret scan | Pass | No current hardcoded credential patterns found |
| Ruff | Fail | 39 lint/format errors, including notebook cells |
| Mypy | Fail | 17 errors across tests and input typing |
| Notebook execution | Fail | `ModuleNotFoundError: twosample_means` in the notebook kernel |
| Package build | Blocked | Build frontend/backend is not installed in the existing environment |
| Virtual environment | Broken | Console-script shebangs point to an old `Mission Test` project path |
| Coverage claim | Unverified | Existing `.coverage` also points to the old project path |
| CI | Missing | No GitHub Actions workflow |
| Release files | Missing | No license, changelog, contributing guide, citation file, or security policy |

The README says there are 112 passing tests and 96% coverage, while the
current suite contains 116 passing tests and the saved coverage data is
not usable after the project move. The README verification section should
only contain numbers produced by CI.

## Highest-priority findings

### 1. The public API models samples, not experiments

`InputSpec` accepts only sample A and sample B. `RunConfig` contains test
parameters, but the package has no first-class representation of:

- Experiment ID and hypothesis.
- Treatment and control labels.
- Randomization unit and analysis unit.
- Exposure eligibility and analysis window.
- Primary, secondary, and guardrail metrics.
- Metric type and denominator semantics.
- Minimum detectable effect or practical significance threshold.
- Pre-experiment covariates.
- Planned contrasts and multiple-testing family.

This makes the code reusable as a statistics battery but not yet safe as
an experimentation workflow.

### 2. Running every test does not solve researcher degrees of freedom

The README presents "all tests run" as an anti-p-hacking safeguard. A
battery of correlated tests can instead create interpretation ambiguity
and inflate false-positive risk if users select favorable results after
seeing the output.

The suite should require one primary estimator/test per metric, selected
from the metric and design specification before analysis. Sensitivity
analyses can still be shown, but must be labeled separately and must not
be treated as independent confirmations.

Recommended default for an independent continuous metric:

1. Welch difference in means with confidence interval.
2. Absolute and relative lift.
3. A standardized effect size.
4. Optional robust/permutation sensitivity analysis.
5. A declared practical-significance threshold.

Normality and variance tests should be descriptive diagnostics, not a
mechanical decision tree for choosing the main test.

### 3. The example analyzes ad exposure, not the conversion outcome

The bundled marketing dataset contains a binary `converted` field, but
the README and notebook analyze `total ads`. That may answer whether the
groups saw different ad volumes, but it does not demonstrate the central
business A/B question: conversion lift caused by ad exposure.

The primary example should analyze conversion rate using a binary metric
estimator. `total ads` can remain as:

- A randomization/implementation diagnostic.
- A secondary exposure metric.
- A covariate only when its causal timing and interpretation are valid.

The dataset is also highly imbalanced: 564,577 `ad` rows versus 23,524
`psa` rows. The demo should explicitly report allocation, run a sample
ratio mismatch check against the intended allocation, and explain whether
the imbalance is expected.

### 4. Subsampling changes the estimand and discards information

The example randomly reduces both groups to 500 observations because the
Bayesian model is expensive. This throws away almost all data and forces
equal sample sizes despite the original allocation imbalance.

The fast demo should run full-data analytical methods first. Expensive
Bayesian analyses should be optional, use a separately documented sample,
or use a scalable approximation. Reports must make clear whether results
come from full data or a subsample.

### 5. Reproducibility is currently broken after a directory move

The `.venv` executable wrappers and `.coverage` file contain absolute
paths to another project. Recreating the environment should be a required
setup step rather than relying on a copied virtual environment.

Recommended recovery:

```bash
rm -rf .venv .coverage
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run mypy src/twosample_means tests
uv run pytest --nbmake notebooks/two_sample_procedure.ipynb
uv build
```

### 6. The repository is not release-ready

Before publishing, resolve the existing staged `.devin` deletions and
decide whether all generated reports and the 588,101-row Kaggle CSV should
remain tracked. The current `.gitignore` lists those files, but already
tracked files are not removed by ignore rules.

The repository also needs a license and CI. Without an explicit license,
other people cannot confidently reuse the code.

## Target product scope

### Core promise

Given experiment data and a predeclared experiment specification, the
package should validate assignment, estimate treatment effects for one or
more metrics, control inferential error across the planned family, and
produce an auditable report with effect sizes, uncertainty, practical
significance, diagnostics, and provenance.

### Supported designs by maturity

**Version 0.2: randomized two-arm experiments**

- Independent users assigned to control or treatment.
- Continuous, binary, count, and ratio metrics.
- Difference in means/proportions and regression equivalents.
- Power, minimum detectable effect, confidence intervals, and multiplicity.
- Sample ratio mismatch and basic data-quality diagnostics.

**Version 0.3: variance reduction and richer experiments**

- CUPED/ANCOVA with pre-period covariates.
- Stratified and blocked randomization.
- Cluster-robust standard errors.
- Multi-arm experiments and planned contrasts.
- Robust and bootstrap sensitivity estimators.

**Version 0.4: observational and quasi-experimental methods**

- Difference in differences.
- Event-study diagnostics.
- Covariate balance reporting.
- Optional matching/weighting interfaces.

Keep observational methods visibly separated from randomized experiment
results; their identifying assumptions are different.

## Proposed API design

Use a small set of typed, serializable specifications rather than exposing
every method as the primary user experience.

```python
from ab_testing import (
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
)

config = ExperimentConfig(
    experiment_id="marketing-demo",
    unit_id="user id",
    assignment="test group",
    control="psa",
    treatments=["ad"],
    primary_metrics=[
        MetricSpec(
            name="conversion_rate",
            column="converted",
            kind="binary",
            practical_effect=0.002,
        )
    ],
    secondary_metrics=[
        MetricSpec(
            name="ads_seen",
            column="total ads",
            kind="continuous",
        )
    ],
    alpha=0.05,
    multiplicity="holm",
    seed=42,
)

result = analyze_experiment(dataframe, config)
result.write("artifacts/marketing-demo")
```

### Key result objects

- `ExperimentResult`: provenance, sample sizes, diagnostics, metric
  results, warnings, software version, and configuration snapshot.
- `MetricResult`: estimand, control/treatment summaries, absolute effect,
  relative effect, standard error, interval, p-value when applicable,
  adjusted p-value, practical-significance result, and method metadata.
- `DiagnosticResult`: severity, code, message, supporting values, and
  recommended action.
- `AnalysisPlan`: frozen or serialized declaration of primary metrics,
  contrasts, alpha allocation, exclusions, and estimator choices.

The existing `twosample_means` functions can power the continuous metric
adapter while the higher-level API decides which outputs are primary and
which are sensitivity analyses.

## Statistical modules to add

### Assignment and data quality

- Sample ratio mismatch using a chi-square or exact multinomial check.
- Duplicate randomization-unit detection.
- Missing assignment and missing outcome reporting.
- Users appearing in multiple variants.
- Pre-treatment covariate balance with standardized mean differences.
- Exposure and eligibility checks.
- Configurable missing-data policy with counts preserved in the report.

### Metric estimators

- Continuous: Welch difference in means and regression equivalent.
- Binary: difference in proportions, risk ratio, odds ratio, and intervals.
- Count: robust mean comparison initially; negative-binomial regression
  when overdispersion matters.
- Ratio: delta-method estimator with user-level numerator and denominator;
  never average row-level ratios by default.
- Quantile: bootstrap interval for median or selected quantiles.
- Clustered/repeated observations: cluster-robust standard errors at the
  randomization unit.

### Multiple testing

- Require metric family labels: primary, secondary, and guardrail.
- Holm as a defensible default for a small confirmatory family.
- Benjamini-Hochberg as an explicit exploratory option.
- Planned contrasts for multi-arm experiments.
- Report raw and adjusted values without hiding effect estimates or
  confidence intervals.

### Variance reduction

- CUPED using pre-period values measured before treatment.
- ANCOVA/regression adjustment with treatment interactions considered.
- Cross-fitting or train/evaluation separation for learned adjustments.
- Report variance reduction and verify no post-treatment variable is used.

### Power and design analysis

- Sample size from baseline rate/variance, allocation, alpha, power, and
  minimum detectable effect.
- Achieved precision and confidence-interval width after the experiment.
- Simulation-based power for ratio, clustered, and adjusted estimators.
- Avoid presenting post-hoc observed power as evidence quality.

### Sequential analysis

Add only after fixed-horizon inference is stable:

- Alpha-spending or group-sequential boundaries.
- Always-valid confidence sequences or e-values as a separate mode.
- Explicit look schedule and stopping rule in the analysis plan.
- No repeated unadjusted peeking.

## Difference-in-differences design

Add difference in differences under a separate `quasi_experimental`
namespace so users do not confuse it with randomized A/B inference.

Suggested interface:

```python
from ab_testing.quasi_experimental import DifferenceInDifferences

model = DifferenceInDifferences(
    outcome="revenue",
    unit="store_id",
    time="date",
    treated="treated_store",
    post="post_launch",
    cluster="store_id",
)
result = model.fit(dataframe)
```

Minimum implementation:

- Canonical two-group, two-period interaction model.
- Cluster-robust standard errors.
- Panel validation and missing-period diagnostics.
- Treatment timing validation.
- Covariate-adjusted specification.
- Effect plot and raw group-time means.
- Explicit identifying-assumption section in the report.

Next layer:

- Multiple pre-period parallel-trends visualization and placebo tests.
- Event-study coefficients with an omitted reference period.
- Staggered-adoption estimators designed for heterogeneous treatment
  effects; do not rely only on naive two-way fixed effects.
- Sensitivity and anticipation-window configuration.

Tests should use synthetic panels with known treatment effects, no-effect
cases, violated parallel trends, staggered timing, and clustered errors.

## Data ingestion and Kaggle workflow

### Clean ingestion boundary

Introduce a normalized dataset contract:

```python
class DataSource(Protocol):
    def load(self) -> pandas.DataFrame: ...
    def fingerprint(self) -> DataFingerprint: ...
```

Implement these adapters:

- `FrameSource`: in-memory dataframe.
- `FileSource`: CSV, parquet, and optionally Arrow.
- `SyntheticSource`: deterministic scenario generator.
- `KaggleSource`: optional download/cache adapter, isolated from analysis.

The analysis package should never require Kaggle credentials. Kaggle
support belongs in an optional dependency group and should read credentials
from the user's normal Kaggle configuration or environment variables.
Never write credentials into notebooks, configuration committed to Git, or
generated reports.

### Dataset manifest

Use a small YAML or TOML manifest per example:

```yaml
id: marketing_ab
provider: kaggle
dataset: faviovaz/marketing-ab-testing
file: marketing_AB.csv
sha256: <expected-file-hash>
license: <dataset-license>
unit_id: user id
assignment: test group
control: psa
treatments: [ad]
```

The loader should:

1. Use a local cached file when its hash matches.
2. Download only when the optional Kaggle client is installed and the
   user has configured credentials outside the repository.
3. Fail with a clear manual-download instruction when offline.
4. Record provider, dataset slug, file hash, retrieval date, and license.

### Offline demo pipeline

The default CI and quick-start demo must not require network access.

Provide two datasets:

- A small synthetic user-level experiment generated from a fixed seed.
- A tiny, legally redistributable fixture derived from or separate from a
  public dataset, with license and provenance documented.

Pipeline:

```bash
uv run ab-test demo --output artifacts/demo
uv run ab-test analyze examples/configs/marketing.toml \
  --data data/raw/marketing_AB.csv \
  --output artifacts/marketing
```

Each run should generate:

- `report.html` for humans.
- `report.json` for machines.
- `config.json` containing the frozen plan.
- `data_fingerprint.json` containing provenance, not raw data.
- Static plots that do not require a notebook to inspect.

## Example notebook plan

Notebooks should demonstrate workflows, while tested Python modules contain
the real logic.

1. `01_binary_conversion_ab.ipynb`
   - Synthetic and marketing conversion examples.
   - Allocation check, conversion lift, interval, power/precision, and
     practical significance.
2. `02_continuous_metric_ab.ipynb`
   - Existing two-sample engine with Welch as primary and robustness
     checks labeled as sensitivity analyses.
3. `03_ratio_metric.ipynb`
   - Revenue per user or clicks per session using numerator/denominator
     semantics and delta-method inference.
4. `04_cuped_variance_reduction.ipynb`
   - Simulated pre-period covariate, before/after standard error, and
     leakage guard.
5. `05_multiple_metrics.ipynb`
   - Primary/secondary/guardrail metrics with Holm and BH comparisons.
6. `06_difference_in_differences.ipynb`
   - Synthetic panel with parallel-trends and event-study diagnostics.
7. `07_kaggle_dataset_adapter.ipynb`
   - Optional local-cache workflow; no credentials or network requirement
     for the committed notebook.

Every notebook should have a lightweight parameter set for CI and should
write outputs to an ignored `artifacts/` directory.

## Recommended project layout

```text
ab-testing-suite/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CITATION.cff
├── src/
│   └── ab_testing/
│       ├── __init__.py
│       ├── api.py
│       ├── config.py
│       ├── results.py
│       ├── diagnostics/
│       │   ├── assignment.py
│       │   └── data_quality.py
│       ├── metrics/
│       │   ├── base.py
│       │   ├── binary.py
│       │   ├── continuous.py
│       │   ├── count.py
│       │   └── ratio.py
│       ├── inference/
│       │   ├── multiplicity.py
│       │   ├── power.py
│       │   ├── robust.py
│       │   └── sequential.py
│       ├── variance_reduction/
│       │   └── cuped.py
│       ├── quasi_experimental/
│       │   └── did.py
│       ├── data/
│       │   ├── sources.py
│       │   ├── manifests.py
│       │   └── synthetic.py
│       └── reporting/
│           ├── html.py
│           ├── json.py
│           └── plots.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── statistical/
│   └── fixtures/
├── examples/
│   ├── configs/
│   └── scripts/
├── notebooks/
├── data/
│   ├── manifests/
│   └── README.md
├── docs/
└── .github/workflows/ci.yml
```

The `src/` layout prevents accidental imports from the repository root and
would have exposed the notebook/package-install problem earlier.

## Testing strategy

### Unit tests

- Configuration validation and serialization.
- Metric type validation.
- Closed-form estimators against hand-calculated examples.
- Missing values, zero denominators, constant arrays, ties, and tiny
  samples.
- Deterministic bootstrap/permutation behavior.

### Statistical validation tests

- Null simulations with empirical type-I error near the configured level.
- Power simulations increasing with sample size and effect magnitude.
- Confidence-interval coverage simulations.
- CUPED unbiasedness and variance reduction when assumptions hold.
- Multiple-testing procedures against trusted library results.
- DiD recovery under parallel trends and bias when trends are violated.

Use tolerances and seeded simulations selected to keep CI reliable and
fast. Put slow Monte Carlo validation in a scheduled workflow rather than
every pull request.

### Integration and golden tests

- Full synthetic experiment to HTML/JSON outputs.
- CLI execution from an installed wheel, not the repository root.
- Notebook execution in a clean environment.
- JSON schema stability and report snapshot tests.
- Build and install the wheel in a fresh temporary environment.

## CI checks

Run on Python 3.11, 3.12, and 3.13 where dependencies support them:

1. `uv sync --all-extras --frozen`
2. `ruff format --check .`
3. `ruff check .`
4. `mypy src tests`
5. `pytest --cov --cov-report=term-missing`
6. Fast notebook execution.
7. `uv build`
8. Install and smoke-test the built wheel.
9. Secret scanning with `gitleaks` or an equivalent scanner.
10. Dependency review and a scheduled vulnerability scan.

Use a minimum coverage threshold only after regenerating a trustworthy
baseline. Coverage should support statistical validation, not replace it.

## Prioritized roadmap

### Quick wins: publication blockers

- [ ] Resolve the staged `.devin` deletions before making new commits.
- [ ] Recreate `.venv` and regenerate coverage from the current path.
- [ ] Fix 39 Ruff errors and 17 Mypy errors.
- [ ] Make the notebook run in a clean installed environment.
- [ ] Add MIT or another chosen license.
- [ ] Add GitHub Actions for lint, typing, tests, notebook, and build.
- [ ] Update README test/coverage claims from CI results.
- [ ] Decide whether to untrack generated reports and the large Kaggle CSV.
- [ ] Add complete project metadata to `pyproject.toml`.
- [ ] Add a small synthetic offline demo.
- [ ] Rename the public positioning to "two-sample continuous analysis"
  until experiment-level features land.

### Medium term: credible A/B testing version

- [ ] Add `ExperimentConfig`, `MetricSpec`, and stable result schemas.
- [ ] Implement binary, continuous, count, and ratio metric adapters.
- [ ] Add sample ratio mismatch and assignment integrity diagnostics.
- [ ] Add absolute/relative lift and practical-significance thresholds.
- [ ] Add sample-size, MDE, and precision calculations.
- [ ] Add Holm and Benjamini-Hochberg correction.
- [ ] Make Welch the continuous default; label the battery as sensitivity
  analysis.
- [ ] Add CUPED/ANCOVA with pre-treatment validation.
- [ ] Generate polished HTML and JSON reports.
- [ ] Create the first five example notebooks and offline CLI demo.
- [ ] Add simulation-based statistical correctness tests.

### Nice to have: broader statistical suite

- [ ] Multi-arm experiments and planned contrasts.
- [ ] Cluster-randomized and repeated-measure support.
- [ ] Sequential testing with explicit stopping rules.
- [ ] Difference in differences and event studies.
- [ ] Optional Kaggle adapter and dataset manifest registry.
- [ ] Documentation site and gallery of reproducible case studies.
- [ ] Plugin interface for custom estimators and metrics.
- [ ] Versioned JSON schema and backward-compatibility policy.

## GitHub publishing checklist

### Repository hygiene

- [ ] Working tree contains only intentional changes.
- [ ] No credentials in current files or Git history.
- [ ] Generated outputs and local environments are ignored.
- [ ] Large datasets are licensed, documented, and either excluded or
  stored using an appropriate large-file/data-release mechanism.
- [ ] Notebook outputs contain no usernames, local paths, tokens, or
  private data.

### Documentation

- [ ] README states exactly which experiment designs and metrics are
  supported.
- [ ] Quick start runs fully offline.
- [ ] One end-to-end example shows input, configuration, output, and
  interpretation.
- [ ] Statistical assumptions and limitations are prominent.
- [ ] Kaggle download instructions never ask users to paste secrets into a
  notebook.
- [ ] API reference and contribution instructions exist.

### Packaging and release

- [ ] License selected and committed.
- [ ] Authors, classifiers, keywords, URLs, and license metadata added.
- [ ] Package builds as sdist and wheel.
- [ ] Built wheel installs and imports in a clean environment.
- [ ] Semantic versioning policy documented.
- [ ] Changelog started.
- [ ] `CITATION.cff` added for academic/portfolio use.
- [ ] Initial tag created only after CI is green.

### Quality gates

- [ ] Ruff formatting and lint pass.
- [ ] Mypy passes.
- [ ] Tests pass on the supported Python matrix.
- [ ] Statistical simulation checks pass within documented tolerances.
- [ ] Notebook smoke tests pass.
- [ ] Secret and dependency scans pass.
- [ ] Coverage is regenerated and the README badge matches CI.

## Suggested first release story

For a strong portfolio launch, avoid claiming to solve every statistical
problem. A credible initial message is:

> I built an auditable Python toolkit for two-arm experiments. It validates
> assignment, supports binary and continuous metrics, reports treatment
> effects with uncertainty and practical significance, controls planned
> multiple comparisons, and produces reproducible HTML/JSON reports. The
> repository includes synthetic and real-data examples plus statistical
> simulation tests.

That story becomes defensible after the quick wins and the core medium-term
features are complete. Difference in differences and sequential testing
can then be presented as later extensions rather than unfinished promises.
