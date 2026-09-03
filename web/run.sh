#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")"
export FORGE_BIN_DIR="${FORGE_BIN_DIR:-$ROOT/build-native}"
export FORGE_EXAMPLES="${FORGE_EXAMPLES:-$ROOT/examples}"

if [[ ! -x "$FORGE_BIN_DIR/forge_solve" ]]; then
  echo "error: $FORGE_BIN_DIR/forge_solve not found — build FORGE first" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python app.py
