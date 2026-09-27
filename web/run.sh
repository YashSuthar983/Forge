#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")"

BIN_DIR="${FORGE_BIN_DIR:-${SOR_BIN_DIR:-}}"
if [[ -z "$BIN_DIR" ]]; then
  for cand in "$ROOT/build" "$ROOT/build-native"; do
    if [[ -x "$cand/sor_solve" ]]; then
      BIN_DIR="$cand"
      break
    fi
  done
fi
export FORGE_BIN_DIR="$BIN_DIR"

if [[ -z "$BIN_DIR" || ! -x "$BIN_DIR/sor_solve" ]]; then
  echo "error: sor_solve not found - build FORGE first (cmake -S . -B build && cmake --build build -j)" >&2
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
