# twosample-means

An auditable terminal-first analysis engine for independent two-sample
outcomes and randomized experiments. It reports binary, continuous, count,
and ratio estimates plus frequentist, non-parametric, Bayesian, and
effect-size results without making accept/reject decisions.

The experiment-level API validates assignment, applies multiplicity
correction, supports variance reduction (CUPED and ANCOVA), cluster-robust
standard errors, power and sequential planning, and produces
Markdown/HTML/JSON reports that validate against a versioned schema.

!!! warning "Scope"

    This is not a general A/B experimentation suite: it does not provide
    assignment generation or automatic causal validation. Treat the legacy
    battery as sensitivity analysis, not independent confirmatory evidence.
    Difference-in-differences lives in a separate `quasi_experimental`
    namespace because it rests on different identifying assumptions.

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

## Quick start

```bash
uv run twosample-means experiment data/demos/marketing_AB.csv \
  --unit-col "user id" --assignment-col "test group" \
  --control psa --treatment ad \
  --metric conversion_rate=converted:binary:primary \
  --metric-family conversion_rate=conversion \
  --multiplicity holm --multiplicity-scope global \
  --output artifacts/marketing-conversion
```

Every run writes `report.md`, `report.json`, and `report.html` into the
required `--output` directory. The JSON is validated against the bundled
`experiment-result-v1` schema before it is written.

## Design principles

- No accept/reject decisions; analysts interpret the evidence.
- Every method has an academic citation.
- SHA-256 data hashing and configuration logging provide provenance.
- Outliers are flagged but never removed.
- The full battery is descriptive sensitivity analysis, not independent
  confirmatory evidence.
