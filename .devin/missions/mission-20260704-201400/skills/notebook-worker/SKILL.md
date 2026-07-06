---
name: notebook-worker
description: Builds the procedure template notebook that runs the full battery end-to-end on bundled sample data, with documented steps and the anti-p-hacking discipline, and verifies it via nbmake.
allowed-tools:
  - read
  - edit
  - write
  - grep
  - glob
  - exec
permissions:
  allow:
    - Exec(uv run pytest --nbmake *)
    - Exec(uv run pytest -q)
    - Exec(uv run mypy *)
    - Exec(uv run ruff *)
    - Write(notebooks/two_sample_mean_test.ipynb)
    - Write(notebooks/sample_data/**)
    - Write(tests/test_notebook.py)
---

# Notebook Worker Procedure

## Your Task
You are implementing feature: $ARGUMENTS

## Setup
1. Read `<missionDir>/mission.md`, `<missionDir>/AGENTS.md`, and
   `<missionDir>/architecture.md`.
2. Read your feature's `fulfills` assertions from
   `<missionDir>/validation-contract.md` — this feature also owns the
    cross-cutting RIGOR assertions.
3. Read `twosample_means/runner.py`, `twosample_means/config.py`, and
   `twosample_means/report.py` to use the public API.
4. Run `uv run pytest -q` to confirm the starting state.

## Work Procedure
1. Create `notebooks/two_sample_mean_test.ipynb` as the procedure
   template with this structure (markdown + code cells):
   - **Title + rationale**: explain the procedure, the three paradigms,
     and the report-only, no-decision discipline (anti-p-hacking).
   - **Step 1 — Configure**: a code cell constructing `InputSpec`
     (pointing at bundled sample CSV) and `RunConfig` (with documented
     defaults). Markdown cell citing the academic rationale for each
     config parameter.
   - **Step 2 — Load data**: call `data_io.load`; display shapes and
     data hash. Markdown cell explaining provenance.
   - **Step 3 — Assumption diagnostics**: run assumptions; display
     outcomes. Markdown cell citing each diagnostic.
   - **Step 4 — Run the full battery**: call `runner.run_battery`;
     display the summary table. Markdown cell citing each test family.
   - **Step 5 — Report**: write the Markdown + JSON report; show the
     report path. Markdown cell restating that the procedure reports
     evidence and applies NO accept/reject decision.
2. Ensure the notebook runs top-to-bottom without errors on the bundled
   sample data and produces a report file.
3. Add `tests/test_notebook.py` (or rely on the `notebook_test` command
   in `services.yaml`) that runs `nbmake` on the notebook and asserts a
   report file exists afterward.
4. Verify the RIGOR assertions across the whole package:
   - `ASSERT-RIGOR-001`: grep `twosample_means/` for decision-producing
     code ("reject"/"significant"/"accept" as a produced label) — none
     allowed (descriptive mentions in docstrings/reports are fine).
   - `ASSERT-RIGOR-002`: every public function has a docstring with a
     citation and assumptions statement.
   - `ASSERT-RIGOR-003`: all stochastic methods accept a seed and are
     reproducible (permutation, bootstrap, BEST).

## Verification (REQUIRED before finishing)
1. `uv run pytest --nbmake notebooks/two_sample_mean_test.ipynb` exits 0.
2. `uv run pytest -q` passes (full suite).
3. `uv run mypy twosample_means` passes.
4. `uv run ruff check twosample_means tests` passes.
5. Verify each `fulfills` assertion's pass criteria is met, including
   the RIGOR assertions.

## Completion
```bash
python3 ~/.config/devin/scripts/mission_utils.py update-feature <missionDir> <featureId> --status completed
python3 ~/.config/devin/scripts/mission_utils.py append-progress <missionDir> worker_completed --feature-id <featureId> --success-state success
```
If blocked, set status to `blocked` and write a handoff to
`<missionDir>/handoffs/`.
