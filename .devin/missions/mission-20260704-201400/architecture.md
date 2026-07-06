# Architecture

## Package layout

```
twosample_means/              # helper package (the "procedure" library)
  __init__.py
  config.py                   # dataclasses: InputSpec, RunConfig
  citations.py                # academic citation registry (method -> reference)
  data_io.py                  # load CSV/parquet/arrays, validate, hash
  assumptions.py              # normality + variance homogeneity + outliers
  frequentist_parametric.py   # Student t, Welch t, z-test
  frequentist_nonparametric.py# Mann-Whitney, Brunner-Munzel, permutation, bootstrap
  bayesian.py                 # BEST (PyMC), Bayes factor (pingouin), HDI/ROPE
  effect_size.py              # Cohen d, Hedges g, Cliff delta, rank-biserial, HL
  report.py                   # result schema + Markdown report + summary table
  runner.py                   # orchestrates the full battery
notebooks/
  two_sample_mean_test.ipynb  # procedure template notebook
  sample_data/                # bundled CSV for notebook smoke test
tests/
  test_data_io.py
  test_assumptions.py
  test_frequentist_parametric.py
  test_frequentist_nonparametric.py
  test_bayesian.py
  test_effect_size.py
  test_report.py
  test_runner.py
pyproject.toml               # UV-managed, pinned deps
```

## Dependency fit

- **scipy.stats**: parametric + non-parametric test statistics, exact
  distributions. The canonical reference implementation.
- **statsmodels**: z-test with known variance, robust variance helpers.
- **pingouin**: effect sizes (Cohen d, Hedges g, Cliff delta, rank-biserial)
  and the JZS Bayes factor (`pingouin.bayesfactor_ttest`). Well-cited,
  wraps Rouder et al. (2009).
- **PyMC + ArviZ**: BEST (Kruschke 2013) Bayesian t-test via MCMC; ArviZ
  for HDI/ROPE. The standard Bayesian stack.
- **pandas / pyarrow**: CSV + parquet I/O.
- **nbmake**: executes the template notebook in CI as a smoke test.

## Data flow

`InputSpec` (paths or arrays) -> `data_io.load()` validates and hashes ->
`runner.run_battery()` calls each module -> each returns a `TestResult`
(method, citation, statistic, p/BF/posterior, effect size, CI/HDI,
assumptions, seed) -> `report.assemble()` builds summary table + per-test
sections -> `report.write_markdown()` + JSON sidecar.
