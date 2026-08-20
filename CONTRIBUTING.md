# Contributing

Thanks for considering a contribution to `twosample-means`.

## Setup

```bash
uv sync --all-extras
```

Python 3.11+ is required. The package uses `uv` for environment and lockfile
management; do not edit `uv.lock` by hand.

## Development loop

```bash
uv run ruff format .          # format
uv run ruff check .           # lint
uv run mypy src/twosample_means tests   # type check
uv run pytest                 # tests
uv build                      # sdist and wheel
```

Every change must keep all four gates green. Statistical simulations are part
of the test suite; rerun the full suite before opening a PR rather than only
the changed test file.

## What belongs in the package

- New estimators must declare their assumptions in module and function
  docstrings and attach an academic citation in `citations.py`.
- Inference methods must not make accept/reject decisions; they report
  estimates and uncertainty. Practical-significance thresholds are reporting
  metadata, never decision rules.
- Data contracts reject ambiguous input (duplicate units, unknown
  assignments, non-finite values) instead of silently correcting it.
- New report fields must be reflected in the bundled JSON Schema and covered
  by the schema validation path; a rendered report that drifts from the
  schema is a bug.
- Any new public API needs an export in the relevant `__init__.py`, a
  documented CLI (when applicable), README mention, and focused tests.

## Testing expectations

- Deterministic, seeded tests preferred; keep statistical simulations small
  enough that CI stays fast.
- Edge cases to cover for every estimator: missing values, constant samples,
  tiny samples, ties, and non-estimable configurations returning structured
  `not_estimable` results.
- CLI changes need at least one end-to-end test through `main()` plus parser
  unit tests.

## Releases

Changes are listed in `CHANGELOG.md` under `[Unreleased]` and promoted into a
versioned section on release. Version bumps follow semantic versioning.