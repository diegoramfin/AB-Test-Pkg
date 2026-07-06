# User Testing

Testing surface findings for the two-sample mean-difference mission.

## Primary testing surface
- **Programmatic API**: `twosample_means.runner.run_battery(input_spec,
  run_config)` returns all results and writes a Markdown + JSON report.
- **Notebook surface**: `notebooks/two_sample_mean_test.ipynb` runs the
  procedure end-to-end on bundled sample data.

## Validation prerequisites
- `uv sync` must succeed before any test runs.
- Bayesian tests (BEST) require PyMC sampling; keep test draws modest
  but sufficient for R-hat < 1.01 and ESS > 400.
- Notebook test requires `nbmake` and the bundled sample CSV.

## Commands (from services.yaml)
- `uv run pytest -q` — full unit suite.
- `uv run mypy twosample_means` — typecheck.
- `uv run ruff check twosample_means tests` — lint.
- `uv run pytest --nbmake notebooks/two_sample_mean_test.ipynb` —
  notebook smoke test.

## Anti-p-hacking validation
- The notebook worker verifies that no function produces an
  accept/reject decision label (RIGOR assertions).
