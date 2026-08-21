# Statistical validation

The test suite includes Monte Carlo simulations that validate the
statistical behavior of the estimators, not just their plumbing.

## Coverage simulations

- **Effect-size interval coverage** (`tests/test_effect_size.py`,
  `tests/test_ab_testing_coverage.py`): seeded simulations confirm the
  confidence intervals cover their known population values near the
  nominal rate.
- **Cluster-robust coverage** (continuous and ratio): with within-cluster
  correlation, the robust interval holds near-nominal coverage where the
  naive user-level interval badly undercovers (roughly 0.96 vs 0.58 on
  the bundled scenarios).
- **CUPED variance reduction**: adjusted outcomes match the theoretical
  `1 - rho^2` reduction on generated data.

## Error-rate simulations

- **Holm FWER** (`tests/test_ab_testing_holm.py`): family-wise error
  stays at or below the ceiling under complete-null and strong-alternative
  scenarios.
- **FDR (BH)** (`tests/test_ab_testing_multiplicity.py`): false discovery
  rate is controlled across many metric families.
- **Sequential boundaries** (`tests/test_ab_testing_sequential_calibration.py`):
  the calibrated boundaries reproduce the spent alpha (family-wise
  crossing rate ≈ 0.05) under the canonical model, through the real
  estimator pipeline end to end, and the always-valid confidence sequence
  covers the true mean at every look time.

## Power simulations

- Empirical power tracks the analytic curves for binary and continuous
  metrics (`tests/test_ab_testing_power.py`).
- `sequential_power` matches the calibrated FWER at zero drift and rises
  with the standardized effect while average sample information falls.

## Running the simulations

```bash
uv sync --all-extras
uv run pytest --cov --cov-report=term-missing
uv run ruff check . && uv run ruff format --check .
uv run mypy src/twosample_means tests examples
uv build
uv run pytest tests/test_wheel_cli.py
```
