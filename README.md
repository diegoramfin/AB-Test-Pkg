# twosample-means

An exhaustive, auditable hypothesis-testing procedure for independent
two-sample mean differences. Runs the full battery — frequentist
(parametric + non-parametric), Bayesian, and effect sizes — and
generates a report **without making any accept/reject decision**.

## Quick start

```bash
# 1. Install dependencies (creates .venv automatically)
uv sync --all-extras

# 2. Run the CLI on a CSV (single file with a group column)
uv run twosample-means notebooks/sample_data/marketing_AB.csv \
    --group-col "test group" \
    --value-col "total ads" \
    --group-a ad --group-b psa \
    --subsample 500 \
    --output output/

# 3. Or run the procedure notebook headlessly
uv run pytest --nbmake notebooks/two_sample_procedure.ipynb

# 4. Or open the notebook interactively
uv run jupyter notebook notebooks/two_sample_procedure.ipynb
```

> **Always prefix commands with `uv run`.** This ensures the local
> `twosample_means` package is importable from the project venv.

## CLI usage

```
uv run twosample-means [OPTIONS] CSV_PATH
```

**Single CSV** (both groups in one file, identified by a column):

```bash
uv run twosample-means data.csv \
    --group-col "test group" \
    --value-col "total ads" \
    --group-a ad --group-b psa \
    --output output/
```

**Two separate CSVs** (one per group):

```bash
uv run twosample-means \
    --csv-a group_a.csv --col-a value \
    --csv-b group_b.csv --col-b value \
    --output output/
```

**Subsample a large file** (MCMC on 588K rows is impractical):

```bash
uv run twosample-means data.csv \
    --group-col "test group" --value-col "total ads" \
    --group-a ad --group-b psa \
    --subsample 500 --seed 42 \
    --output output/
```

**Tune the statistical parameters:**

```bash
uv run twosample-means data.csv \
    --group-col "test group" --value-col "total ads" \
    --group-a ad --group-b psa \
    --alpha 0.01 \
    --ci-level 0.99 \
    --mcmc-draws 4000 --mcmc-chains 4 \
    --permutation-iterations 20000 \
    --bootstrap-iterations 20000 \
    --output output/
```

**Print the full Markdown report to stdout** (in addition to writing
files):

```bash
uv run twosample-means data.csv ... --print-summary
```

Run `uv run twosample-means --help` for all options.

## Using the library directly

```python
import numpy as np
from twosample_means.config import RunConfig, InputSpec
from twosample_means.data_io import load
from twosample_means.runner import run
from twosample_means.reporting import write_report

# Load data (CSV, parquet, or in-memory arrays)
spec = InputSpec(
    sample_a=[1.0, 2.0, 3.0, 4.0, 5.0],
    sample_b=[2.0, 3.0, 4.0, 5.0, 6.0],
)
data = load(spec)

# Run the full battery
config = RunConfig()
report = run(data, config)

# Write Markdown + JSON reports
md_path, json_path = write_report(report, "output/")
```

## What the battery includes

| Category | Methods |
|---|---|
| **Diagnostics** | Shapiro-Wilk, Anderson-Darling, D'Agostino K², Levene, Bartlett, Brown-Forsythe, IQR/z-score outlier flagging |
| **Parametric** | Student's t, Welch's t, z-test (requires known variance) |
| **Non-parametric** | Mann-Whitney U, Brunner-Munzel, permutation test (exact/Monte Carlo), bootstrap CI |
| **Bayesian** | BEST (Kruschke 2013) via PyMC + HDI/ROPE, JZS Bayes factor via pingouin |
| **Effect sizes** | Cohen's d, Hedges' g, Cliff's delta, rank-biserial, Hodges-Lehmann |

Every method includes its academic citation in the result.

## Verification

```bash
uv run ruff check twosample_means tests   # lint
uv run mypy twosample_means                # type check
uv run pytest                              # tests (112 passing, 96% coverage)
uv run pytest --nbmake notebooks/two_sample_procedure.ipynb  # notebook
```

## Project layout

```
twosample_means/          # the library
  config.py               # RunConfig, InputSpec
  citations.py            # academic citations registry
  data_io.py              # load + validate + hash
  assumptions.py          # normality, variance, outliers
  frequentist_parametric.py
  frequentist_nonparametric.py
  bayesian.py             # BEST + Bayes factor
  effect_size.py          # d, g, delta, rank-biserial, HL
  reporting.py            # TestResult schema + Markdown + JSON
  runner.py               # orchestrator (runs the full battery)
notebooks/
  two_sample_procedure.ipynb  # template notebook with RIGOR assertions
  sample_data/                # bundled marketing A/B dataset
tests/                    # 112 tests
```

## Design principles

- **No accept/reject decisions** — the procedure reports evidence; the
  analyst interprets it.
- **Every method has an academic citation** — traceable to original
  literature.
- **Full provenance** — SHA-256 data hashing + configuration logging.
- **Outliers flagged but never removed** — data integrity preserved.
- **All tests run** — avoids the "garden of forking paths" problem
  (Gelman & Loken, 2014).
