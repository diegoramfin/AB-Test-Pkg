---
name: reporting-worker
description: Implements the TestResult schema, the Markdown report writer with summary table and JSON sidecar, and the runner orchestrator that executes the full battery.
allowed-tools:
  - read
  - edit
  - write
  - grep
  - glob
  - exec
permissions:
  allow:
    - Exec(uv run pytest)
    - Exec(uv run mypy *)
    - Exec(uv run ruff *)
    - Write(twosample_means/report.py)
    - Write(twosample_means/runner.py)
    - Write(twosample_means/__init__.py)
    - Write(tests/test_report.py)
    - Write(tests/test_runner.py)
    - Write(notebooks/sample_data/**)
---

# Reporting Worker Procedure

## Your Task
You are implementing feature: $ARGUMENTS

## Setup
1. Read `<missionDir>/mission.md`, `<missionDir>/AGENTS.md`, and
   `<missionDir>/architecture.md`.
2. Read your feature's `fulfills` assertions from
   `<missionDir>/validation-contract.md`.
3. Read all `twosample_means/` method modules to understand the result
   shapes they return.
4. Run `uv run pytest -q` to confirm the starting state.

## Work Procedure
1. Implement `twosample_means/report.py`:
   - A `TestResult` dataclass with fields: method_name, paradigm
     (frequentist-parametric/frequentist-nonparametric/bayesian/
     effect-size/diagnostic), citation, statistic, p_value (or
     bayes_factor or posterior_summary), effect_size, ci_or_hdi,
     assumptions (list of {check, outcome}), seed, software_versions
     (dict), data_hash.
   - `assemble(results) -> Report` building a summary table (one row
     per test) and per-test sections.
   - `write_markdown(report, path)` writing a Markdown file with the
     summary table and per-test sections including citations and
     assumption outcomes, plus `write_json(report, path)` writing a
     JSON sidecar that round-trips the results.
2. Implement `twosample_means/runner.py`:
   - `run_battery(input_spec, run_config) -> BatteryResult` that loads
     data via `data_io.load`, runs assumptions + all frequentist +
     non-parametric + Bayesian + effect-size methods, collects
     `TestResult` objects, and writes the Markdown + JSON report.
   - No method is silently skipped; if a method errors, record the
     error in its result rather than dropping it.
3. Bundle a small sample CSV in `notebooks/sample_data/` for the runner
   and notebook smoke tests (two columns: group_a, group_b).
4. Every public function: docstring. No decision logic anywhere.
5. Write `tests/test_report.py` (schema conformance, Markdown parses,
   JSON round-trips) and `tests/test_runner.py` (one call produces all
   results and a report file on bundled sample data; no method skipped).

## Verification (REQUIRED before finishing)
1. `uv run pytest -q tests/test_report.py tests/test_runner.py` passes.
2. `uv run mypy twosample_means` passes.
3. `uv run ruff check twosample_means tests` passes.
4. Verify each `fulfills` assertion's pass criteria is met.

## Completion
```bash
python3 ~/.config/devin/scripts/mission_utils.py update-feature <missionDir> <featureId> --status completed
python3 ~/.config/devin/scripts/mission_utils.py append-progress <missionDir> worker_completed --feature-id <featureId> --success-state success
```
If blocked, set status to `blocked` and write a handoff to
`<missionDir>/handoffs/`.
