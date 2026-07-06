# Validation Contract

Exhaustive behavioral assertions for the two-sample mean-difference testing
procedure. Each assertion has a unique ID, behavior, pass criteria, and fail
criteria. Every assertion is claimed by exactly one feature's `fulfills`.

## Surface: Foundation

### Area: Project scaffold

- `ASSERT-FOUND-001`: The project is UV-managed with a `pyproject.toml`
  pinning all runtime and dev dependencies.
  - **Pass**: `pyproject.toml` exists; `uv sync` succeeds; versions are
    pinned (no floating `latest`/`*`/unbounded `>=`).
  - **Fail**: `uv sync` fails, or any dependency uses a floating range.

- `ASSERT-FOUND-002`: A `twosample_means/` package exists with the modules
  declared in `architecture.md` and an `__init__.py`.
  - **Pass**: all listed modules importable; `import twosample_means` works.
  - **Fail**: any listed module missing or import errors.

### Area: Configuration

- `ASSERT-FOUND-003`: A `RunConfig` dataclass holds all configurable
  parameters (alpha, CI level, HDI mass, ROPE width, MCMC draws/chains,
  permutation iterations, bootstrap iterations, seed) with documented
  defaults and no hardcoded magic values elsewhere.
  - **Pass**: `RunConfig` fields cover all tunables; no module reads a
    magic literal for these parameters.
  - **Fail**: a tunable parameter is hardcoded in a method module.

- `ASSERT-FOUND-004`: An `InputSpec` dataclass accepts either file paths
  (CSV/parquet) or in-memory array-likes for the two samples.
  - **Pass**: `InputSpec` constructed from paths and from arrays both load.
  - **Fail**: only one input mode supported.

### Area: Citations registry

- `ASSERT-FOUND-005`: A citations registry maps every statistical method
  name to an academic reference (author, year, title/journal).
  - **Pass**: every method used by the battery has a registry entry with a
    complete citation.
  - **Fail**: any method lacks a citation or has an incomplete one.

### Area: Data I/O

- `ASSERT-FOUND-006`: `data_io.load()` reads CSV and parquet files into
  validated numeric arrays and accepts in-memory array-likes.
  - **Pass**: CSV, parquet, and array inputs all yield equal-length-checkable
    numeric arrays.
  - **Fail**: any input mode fails or returns non-numeric data.

- `ASSERT-FOUND-007`: `data_io.load()` validates inputs: non-empty, numeric,
  finite, minimum sample size enforced, and raises explicit errors otherwise.
  - **Pass**: empty/non-numeric/non-finite/too-small inputs raise typed
    errors with messages.
  - **Fail**: invalid input passes silently or raises a generic exception.

- `ASSERT-FOUND-008`: `data_io.load()` computes a deterministic hash of the
  loaded data for provenance.
  - **Pass**: same data yields same hash; altered data yields a different
    hash; hash is recorded in results.
  - **Fail**: hash is non-deterministic or not recorded.

## Surface: Assumption Diagnostics

### Area: Normality

- `ASSERT-DIAG-001`: Normality is assessed with Shapiro-Wilk, Anderson-
  Darling, and D'Agostino K^2, each returning statistic, p-value, citation,
  and an assumption outcome (met/not-met) without applying a decision.
  - **Pass**: all three tests return finite statistics and p-values in
    [0,1] on normal and skewed synthetic data; citations present.
  - **Fail**: any test missing, returns non-finite values, or applies a
    decision rule.

### Area: Variance homogeneity

- `ASSERT-DIAG-002`: Variance homogeneity is assessed with Levene, Bartlett,
  and Brown-Forsythe, each returning statistic, p-value, citation, and
  assumption outcome.
  - **Pass**: all three return finite results on equal- and unequal-variance
    synthetic data; citations present.
  - **Fail**: any test missing or returns non-finite values.

### Area: Outliers

- `ASSERT-DIAG-003`: Outlier flagging reports count and indices of flagged
  points (e.g., via IQR or z-score rule) without removing data.
  - **Pass**: flagged outlier indices are reported; original data unchanged.
  - **Fail**: data is modified or outliers not reported.

## Surface: Frequentist Parametric

### Area: Student's and Welch's t

- `ASSERT-PARAM-001`: Student's t-test (equal variance) and Welch's t-test
  (unequal variance) return statistic, p-value, degrees of freedom, CI,
  citation, and assumption notes; no decision applied.
  - **Pass**: results match `scipy.stats.ttest_ind` (equal_var True/False)
    to tolerance on synthetic data; CIs bracket the mean difference.
  - **Fail**: results diverge from scipy or a decision rule is applied.

### Area: z-test

- `ASSERT-PARAM-002`: A z-test for known variance returns statistic, p-value,
  CI, and citation, requiring the population variance(s) to be supplied via
  config (no silent default).
  - **Pass**: result matches the analytical z-test on synthetic data with
    known variance; missing variance raises an explicit error.
  - **Fail**: variance defaults silently or result is incorrect.

## Surface: Frequentist Non-Parametric

### Area: Rank-based tests

- `ASSERT-NONPARAM-001`: Mann-Whitney U and Brunner-Munzel tests return
  statistic, p-value, citation, and assumption notes; no decision applied.
  - **Pass**: results match `scipy.stats.mannwhitneyu` and
    `scipy.stats.brunnermunzel` to tolerance on synthetic data.
  - **Fail**: results diverge from scipy or a decision rule is applied.

### Area: Permutation test

- `ASSERT-NONPARAM-002`: A permutation test supports exact (small n) and
  Monte Carlo (large n) modes, returns statistic, p-value, iteration count,
  seed, and citation.
  - **Pass**: exact mode p-value matches exhaustive enumeration on a tiny
    example; Monte Carlo mode is reproducible given the same seed.
  - **Fail**: exact mode incorrect or Monte Carlo not reproducible.

### Area: Bootstrap CI

- `ASSERT-NONPARAM-003`: A bootstrap CI for the mean difference returns the
  point estimate, CI bounds, iteration count, seed, and citation.
  - **Pass**: CI brackets the point estimate; reproducible given seed; CI
    level comes from config.
  - **Fail**: CI does not bracket estimate or not reproducible.

## Surface: Bayesian

### Area: BEST (Kruschke)

- `ASSERT-BAYES-001`: The BEST Bayesian t-test (Kruschke 2013) via PyMC
  returns posterior mean difference, HDI, ROPE outcome, MCMC diagnostics
  (R-hat, ESS), draws/chains, seed, and citation.
  - **Pass**: posterior mean difference is finite and within data range;
    R-hat < 1.01 and ESS > 400 on bundled sample data; reproducible given
    seed.
  - **Fail**: non-finite posterior, R-hat >= 1.01, or not reproducible.

### Area: Bayes factor

- `ASSERT-BAYES-002`: A JZS Bayes factor is computed via pingouin and
  returns BF10 (and BF01), citation (Rouder et al. 2009), and the prior
  width used (from config).
  - **Pass**: BF10 is finite and positive on synthetic data; matches
    `pingouin.bayesfactor_ttest` to tolerance.
  - **Fail**: BF non-finite, non-positive, or diverges from pingouin.

### Area: ROPE / HDI reporting

- `ASSERT-BAYES-003`: ROPE and HDI are reported as descriptive evidence
  (proportion of posterior inside ROPE, HDI bounds) with NO accept/reject
  decision.
  - **Pass**: ROPE proportion in [0,1]; HDI bounds within data range; no
    decision string like "reject" produced.
  - **Fail**: a decision is applied or ROPE proportion out of range.

## Surface: Effect Sizes

### Area: Standardized effect sizes

- `ASSERT-EFFECT-001`: Cohen's d and Hedges' g are computed with bias
  correction for g, returning point estimate, CI, and citation.
  - **Pass**: values match `pingouin.compute_effsize` to tolerance; Hedges'
    g differs from Cohen's d by the small-sample correction.
  - **Fail**: values diverge or g lacks bias correction.

### Area: Non-parametric effect sizes

- `ASSERT-EFFECT-002`: Cliff's delta and rank-biserial correlation are
  computed, returning point estimate, CI, and citation.
  - **Pass**: values match `pingouin` reference to tolerance; delta in
    [-1, 1].
  - **Fail**: values diverge or delta out of [-1, 1].

### Area: Hodges-Lehmann

- `ASSERT-EFFECT-003`: The Hodges-Lehmann estimator of location shift is
  computed with a CI, returning estimate, CI, and citation.
  - **Pass**: estimate matches `scipy.stats.hodgeslehmann` to tolerance; CI
    brackets the estimate.
  - **Fail**: estimate diverges or CI does not bracket it.

## Surface: Reporting & Runner

### Area: Result schema

- `ASSERT-REPORT-001`: Every test returns a `TestResult` with fields: method
  name, paradigm, citation, statistic, p-value or BF or posterior summary,
  effect size, CI/HDI, assumptions checked + outcomes, seed, software
  versions, data hash.
  - **Pass**: all battery results conform to the schema; no field missing.
  - **Fail**: any result missing a required field.

### Area: Markdown report

- `ASSERT-REPORT-002`: The report writer produces a Markdown file with a
  summary table (one row per test) and per-test sections including
  citations and assumption outcomes, plus a JSON sidecar.
  - **Pass**: Markdown file parses; summary table has one row per battery
    method; JSON sidecar round-trips the results.
  - **Fail**: table incomplete, citations missing, or JSON invalid.

### Area: Runner

- `ASSERT-RUNNER-001`: The runner orchestrates the full battery
  (assumptions + all tests + effect sizes) from a single `run_battery()`
  call given an `InputSpec` and `RunConfig`, returning all results and
  writing the report.
  - **Pass**: one call produces all results and a report file on bundled
    sample data; no method silently skipped.
  - **Fail**: any battery method skipped or report not written.

## Surface: Notebook Procedure

### Area: Template notebook

- `ASSERT-NOTEBOOK-001`: The template notebook runs end-to-end on bundled
  sample data via `nbmake`, producing a report file without errors.
  - **Pass**: `uv run pytest --nbmake notebooks/two_sample_mean_test.ipynb`
    exits 0 and a report file exists afterward.
  - **Fail**: notebook errors or no report produced.

- `ASSERT-NOTEBOOK-002`: The notebook documents each step (configure, load,
  diagnostics, battery, report) with markdown cells citing the academic
  rationale and stating the report-only, no-decision discipline.
  - **Pass**: markdown cells present for each step; anti-p-hacking stance
    stated.
  - **Fail**: steps undocumented or decision discipline not stated.

## Cross-area: Anti-p-hacking & rigor

- `ASSERT-RIGOR-001`: No function in the package applies an accept/reject
  decision or a "significant/not-significant" label.
  - **Pass**: grep for "reject", "significant", "accept" in
    `twosample_means/` returns no decision-producing code (only
    descriptive mentions in docstrings/reports allowed).
  - **Fail**: any function produces a decision label.

- `ASSERT-RIGOR-002`: Every public function in `twosample_means/` has a
  docstring with an academic citation and an assumptions statement.
  - **Pass**: all public functions documented with citation + assumptions.
  - **Fail**: any public function lacks citation or assumptions.

- `ASSERT-RIGOR-003`: All stochastic methods accept a `seed` and are
  reproducible.
  - **Pass**: same seed -> same output for permutation, bootstrap, and
    Bayesian methods.
  - **Fail**: any stochastic method non-reproducible or lacks a seed param.
