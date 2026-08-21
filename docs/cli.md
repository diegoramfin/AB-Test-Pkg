# Command line

```text
uv run twosample-means --version
uv run twosample-means experiment CSV_PATH [OPTIONS]
uv run twosample-means analyze [OPTIONS] [CSV_PATH]
uv run twosample-means fetch DATASET_NAME --output CACHE_DIRECTORY
```

## The `experiment` command

Analyze declared metrics from a user-level or aggregate CSV.

```bash
uv run twosample-means experiment data/checkout.csv \
  --unit-col user_id --assignment-col variant \
  --control control --treatment treatment \
  --metric conversion_rate=converted:binary:primary \
  --metric revenue=revenue:continuous:secondary \
  --metric-family conversion_rate=conversion \
  --multiplicity holm --multiplicity-scope global \
  --output artifacts/checkout
```

- Repeat `--metric NAME=COLUMN:KIND[:ROLE]` per metric. `KIND` is
  `binary`, `continuous`, `count`, or `ratio` (ratio syntax:
  `NAME=NUMERATOR/DENOMINATOR:ratio`). `ROLE` defaults to `secondary`.
- `--multiplicity` accepts `none`, `holm`, or `fdr_bh`;
  `--multiplicity-scope` corrects within `family` (default) or `global`.
- `--expected-allocation control=0.5,treatment=0.5` enables the
  sample-ratio mismatch test.
- `--strata COLUMN` enables within-stratum sample-ratio mismatch checks.
- `--covariate NAME=COLUMN` enables CUPED variance reduction for
  continuous and count metrics (and adds the covariate to the balance
  table).
- Repeat `--balance-columns COLUMN` to audit additional pre-treatment
  columns for balance (SMD) without using them for adjustment.
- `--cluster COLUMN` enables cluster-robust standard errors for
  continuous, count, and ratio metrics.
- `--unit-type user|aggregate|unknown` records row semantics; aggregate
  and unknown types produce explicit warnings.
- Separate control/treatment CSVs: `--csv-a`/`--csv-b` (the assignment
  column is synthesized from the file role).

## The legacy `analyze` command

Runs the low-level two-sample battery (diagnostics, parametric,
non-parametric, effect sizes). Missing values fail by default; pass
`--missing-values exclude` to remove NaNs per arm. Add
`--full-battery` (or `--include-bayesian` / `--include-resampling`) to
enable the heavier Bayesian and resampling methods.

## The `fetch` command

Downloads a registered Kaggle dataset into a local cache and writes a
`manifest.json` sidecar with source, license, aggregation level, unit
semantics, and quality flag. Configure Kaggle authentication separately
with the official Kaggle CLI.
