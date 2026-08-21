# Experiment API

The `twosample_means.ab_testing` namespace provides a typed experiment
boundary: declare the analysis plan, analyze a normalized frame, and write
a validated report.

```python
from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
)
from twosample_means.reporting import write_experiment_report

config = ExperimentConfig(
    experiment_id="checkout-copy",
    unit_id="user_id",
    assignment="variant",
    control="control",
    treatments=("treatment",),
    expected_allocation={"control": 0.5, "treatment": 0.5},
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
result = analyze_experiment(dataframe, config)
write_experiment_report(result, "artifacts/checkout-copy")
```

## Metric kinds

| Kind | Column declaration | Inference |
|---|---|---|
| `binary` | `success_value` outcome | pooled score test, Newcombe-style intervals |
| `continuous` | numeric outcome | Welch t |
| `count` | unit-level counts | Welch t on means |
| `ratio` | `numerator`/`denominator` | user-level delta method |

## Multiplicity

Correction is applied within each declared `family` by default. Set
`multiplicity_scope="global"` to pool all estimable metrics into one
family. `holm` adds Holm step-down adjusted p-values and simultaneous
confidence intervals; `fdr_bh` applies Benjamini-Hochberg adjusted
p-values with conservative Bonferroni family-wise intervals.

## Variance reduction

`MetricSpec(covariate="pre_period_column")` enables CUPED for continuous
and count metrics. `covariate_method` selects the estimator:

- `"cuped"` (default): residualizes outcomes with the pooled slope,
  Welch inference on adjusted values.
- `"ancova"`: regression `Y ~ treatment + X` with
  heteroskedasticity-robust standard errors.
- `"interaction"`: adds a `treatment * X` interaction, allowing
  arm-specific slopes, with the effect evaluated at the mean covariate.

Every covariate-adjusted report carries an explicit **covariate leakage
guard** stating that temporal ordering is caller-declared and not
verifiable from the data.

## Cluster-robust inference

`ExperimentConfig(cluster="cluster_column")` enables the CR1 sandwich
with a small-sample correction and `G-2` degrees of freedom for
continuous, count, and ratio metrics. CUPED adjustment and clustering
compose; clusters spanning both arms produce an explicit warning.

## Diagnostics

Assignment diagnostics report unit duplication, cross-arm assignment,
missing or unknown labels, arm counts, and an optional sample-ratio
mismatch p-value when `expected_allocation` is configured. Normalization
requires one row per unit, validates labels and metric types, and computes
a deterministic data fingerprint.

### Stratified sample-ratio mismatch

`ExperimentConfig(strata="region")` runs the same multinomial SRM test
within each declared stratum. A balanced marginal allocation can hide
offsetting per-stratum imbalances, so the report lists per-stratum
p-values and warns when any stratum fails.

### Pre-treatment covariate balance

Every metric covariate declared via `MetricSpec(covariate=...)` is also
checked for balance before estimation, as are any columns listed in
`ExperimentConfig(balance_columns=...)` — the balance-only list for
pre-treatment covariates you want audited without using them for variance
reduction. For each checked column the report computes a standardized
mean difference (SMD) per arm at the randomization-unit level,
`(mean_arm - mean_control) / pooled_sd`, and flags |SMD| > 0.1. Columns
that are non-numeric, constant within an arm, or have too few units are
listed with an explanatory warning instead of an SMD.
