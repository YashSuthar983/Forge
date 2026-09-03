#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")"
export SOR_BIN_DIR="${SOR_BIN_DIR:-$ROOT/build-native}"
export SOR_EXAMPLES="${SOR_EXAMPLES:-$ROOT/examples}"

if [[ ! -x "$SOR_BIN_DIR/sor_solve" ]]; then
  echo "error: $SOR_BIN_DIR/sor_solve not found — build SOR first" >&2
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
