# Environment

Verified environment state for the two-sample mean-difference mission.

## Runtime
- **OS**: macOS (Darwin 24.6.0).
- **Python**: >=3.11 required. UV manages the interpreter via `uv sync`.
- **Dependency manager**: UV (project rule). No global pip.

## Dependencies (pinned in pyproject.toml)
- Runtime: numpy, scipy, statsmodels, pandas, pyarrow, pymc, arviz,
  pingouin, matplotlib, jupyter.
- Dev: pytest, pytest-cov, ruff, mypy, nbmake.

## Setup
- `bash .devin/missions/mission-20260704-201400/init.sh` runs `uv sync`
  and a baseline `uv run pytest -q`.

## Boundaries
- No network services, no ports bound.
- No runtime network calls. PyPI only at `uv sync` time.
- No writes outside the project root.
