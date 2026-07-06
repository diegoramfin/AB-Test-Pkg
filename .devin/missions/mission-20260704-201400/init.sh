#!/usr/bin/env bash
# Mission environment setup script
# Run this before starting workers
set -euo pipefail

echo "[mission] Running init.sh..."
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "[mission] ERROR: uv is not installed. Install from https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [ ! -f pyproject.toml ]; then
    echo "[mission] No pyproject.toml yet — project scaffold is created by"
    echo "[mission] feat-foundation-001. Skipping uv sync; re-run after scaffold."
else
    echo "[mission] Running uv sync..."
    uv sync

    echo "[mission] Running baseline tests (may be empty initially)..."
    uv run pytest -q || echo "[mission] No tests / baseline failures recorded."
fi

echo "[mission] init.sh complete."
