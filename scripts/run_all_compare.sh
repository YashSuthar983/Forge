#!/usr/bin/env bash
# Run Netlib benchmarks against every solver available on this machine.
#
# Solvers in run_compare.py:
#   SOR-simplex, SOR-pdhg, SOR-hpr, SOR-hpr-full, SOR-hpr-vulkan,
#   highs, cbc, scipy-ipm, scipy-simplex  (+ gurobi if gurobipy installed)
#
# Solvers in run_compare_solver_accl.py (49/93 Netlib loadable by accl):
#   SOR-simplex, SOR-pdhg, SOR-hpr, accl-simplex, accl-pdhg, highs
#
# Usage:
#   ./scripts/run_all_compare.sh [--time-limit 20] [--suite netlib]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TIME_LIMIT=20
SUITE=netlib
while [[ $# -gt 0 ]]; do
  case "$1" in
    --time-limit) TIME_LIMIT="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

VENV="$ROOT/benchmarks/.venv-baseline"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating baseline venv..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q highspy scipy pulp
fi

if [[ ! -x "$ROOT/build/sor_solve" ]]; then
  echo "error: build/sor_solve missing — run: cmake --build build" >&2
  exit 2
fi

SOLVERS="sor-simplex,sor-pdhg,sor-hpr,sor-hpr-full,sor-hpr-vulkan,highs,cbc,scipy-ipm,scipy-simplex"
if "$VENV/bin/python" -c "import gurobipy" 2>/dev/null; then
  SOLVERS="$SOLVERS,gurobi"
  echo "gurobipy found — including gurobi"
else
  echo "gurobipy not installed — skipping gurobi"
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/benchmarks/results"
mkdir -p "$OUT"

echo "=== run_compare: $SOLVERS ==="
python3 scripts/run_compare.py \
  --suite "$SUITE" \
  --time-limit "$TIME_LIMIT" \
  --solvers "$SOLVERS" \
  2>&1 | tee "$OUT/run_compare_${STAMP}.log"

if [[ -d "${SOLVER_ACCL_ROOT:-/tmp/solver_accl}" ]]; then
  echo "=== run_compare_solver_accl ==="
  python3 scripts/run_compare_solver_accl.py \
    --suite "$SUITE" \
    --time-limit "$TIME_LIMIT" \
    2>&1 | tee "$OUT/run_compare_accl_${STAMP}.log"
else
  echo "SOLVER_ACCL_ROOT not set — skipping accl-simplex / accl-pdhg"
fi

echo "Done. Results in $OUT/compare-*.md"
