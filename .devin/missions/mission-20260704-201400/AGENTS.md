# Mission Guidance

## Mission Boundaries

- **Port ranges**: none. This mission binds no network ports. Pure library +
  notebook. Workers must not start servers.
- **Off-limits resources**:
  - No writes outside the project root
    (`/Users/diegoramirez/CascadeProjects/Mission Test`).
  - No `~/.config`, `~/.cache`, or other home-directory mutations.
  - No system Python installs; no global `pip install`. Use UV only
    (`uv add`, `uv sync`).
  - No network calls at runtime. All computation is local.
  - Do not modify `.devin/missions/` system-managed files (`state.json`,
    `features.json` status fields, `validation-state.json`) directly — use
    the mission utility script.
- **External services**: none allowed at runtime. PyPI is accessed only via
  UV during `uv sync` (setup time).

## Worker Rules

- Stay in scope — only work on your assigned feature. Do not implement
  methods belonging to another feature.
- Never violate mission boundaries above.
- Read `mission.md` and this file before starting work.
- Every statistical method MUST be a thin wrapper over a proven library
  implementation (scipy, statsmodels, PyMC, pingouin). Do NOT implement
  bespoke statistical algorithms.
- Every public function MUST have a docstring citing its academic source
  and stating its assumptions. No method without a citation ships.
- Report-only discipline: NEVER add accept/reject decision logic. Compute
  and return evidence only.
- No magic values. All thresholds/parameters come from config.
- All stochastic methods accept a `seed`; default seed is fixed and
  documented.
- Follow project rules: PEP 8, 79-char lines, snake_case, type hints,
  single-responsibility functions, no global state, no hardcoded config.
- Run `uv run pytest`, `uv run mypy`, `uv run ruff check` before finishing.
- Report discovered issues in `handoffs/`, do not fix unrelated bugs.

## Known Pre-Existing Issues

_None yet. Greenfield project._
