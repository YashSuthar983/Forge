#!/usr/bin/env bash
# Full LP/MILP/QP performance vs HiGHS only (no CBC / SciPy / Gurobi).
#
# Usage:
#   ./scripts/run_full_perf_highs.sh
#   ./scripts/run_full_perf_highs.sh --time-limit 30 --industrial-limit 120
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TIME_LIMIT=30
INDUSTRIAL_LIMIT=120
while [[ $# -gt 0 ]]; do
  case "$1" in
    --time-limit) TIME_LIMIT="$2"; shift 2 ;;
    --industrial-limit) INDUSTRIAL_LIMIT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

VENV="$ROOT/benchmarks/.venv-baseline"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating baseline venv..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q highspy
fi
if [[ ! -x "$ROOT/build/sor_solve" || ! -x "$ROOT/build/sor_gen" ]]; then
  echo "error: build sor_solve and sor_gen first" >&2
  exit 2
fi
if [[ ! -d "$ROOT/benchmarks/netlib/mps" ]]; then
  echo "error: benchmarks/netlib/mps missing" >&2
  exit 2
fi
if [[ ! -d "$ROOT/benchmarks/miplib-easy/mps" ]]; then
  echo "Fetching MIPLIB-easy subset..."
  python3 scripts/fetch_miplib_easy.py
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/benchmarks/results"
mkdir -p "$OUT"
MASTER="$OUT/FULL_PERF_HIGHS_${STAMP}.log"
INDEX="$OUT/FULL_PERF_HIGHS_${STAMP}.md"

export PYTHONUNBUFFERED=1

{
  echo "# FULL PERFORMANCE BENCHMARK (HiGHS only)"
  echo "stamp=$STAMP host=$(hostname) time_limit=${TIME_LIMIT}s industrial_limit=${INDUSTRIAL_LIMIT}s"
  echo "started=$(date -Is)"
  echo
} | tee "$MASTER"

SOLVERS="sor-simplex,sor-pdhg,sor-hpr,highs"
echo "=== [1/3] Netlib LP (93) solvers=$SOLVERS ===" | tee -a "$MASTER"
python3 scripts/run_compare.py \
  --suite netlib \
  --time-limit "$TIME_LIMIT" \
  --solvers "$SOLVERS" \
  2>&1 | tee -a "$MASTER"
NETLIB_MD="$(ls -t "$OUT"/compare-netlib-*.md | head -1)"

echo "=== [2/3] Industrial ladder S→HUGE (HiGHS only) ===" | tee -a "$MASTER"
python3 scripts/run_industrial_perf.py \
  --time-limit "$INDUSTRIAL_LIMIT" \
  --max-nodes 10000 \
  --sizes S,M,L,XL,XXL,HUGE \
  --baselines-only highs \
  2>&1 | tee -a "$MASTER"
IND_MD="$(ls -t "$OUT"/industrial-perf-*.md | head -1)"

echo "=== [3/3] MIPLIB-easy + industrial demos (HiGHS only) ===" | tee -a "$MASTER"
python3 scripts/run_new_features_compare.py \
  --suite all \
  --time-limit "$TIME_LIMIT" \
  --max-nodes 10000 \
  --baselines-only highs \
  2>&1 | tee -a "$MASTER"
NEW_MD="$(ls -t "$OUT"/compare-new-all-*.md | head -1)"

{
  echo
  echo "=== DONE $(date -Is) ==="
  echo "netlib:     $NETLIB_MD"
  echo "industrial: $IND_MD"
  echo "new/mip:    $NEW_MD"
} | tee -a "$MASTER"

cat > "$INDEX" <<EOF
# Full performance benchmark — HiGHS only ($STAMP)

- Host: \`$(hostname)\`
- Netlib time limit: ${TIME_LIMIT}s · Industrial limit: ${INDUSTRIAL_LIMIT}s
- Baseline: **HiGHS only** (external process — never linked)

## Reports

1. **Netlib LP (93)** — [\`$(basename "$NETLIB_MD")\`]($(basename "$NETLIB_MD"))
2. **Industrial ladder S→HUGE** — [\`$(basename "$IND_MD")\`]($(basename "$IND_MD"))
3. **MIPLIB-easy + demos** — [\`$(basename "$NEW_MD")\`]($(basename "$NEW_MD"))

## Master log

[\`$(basename "$MASTER")\`]($(basename "$MASTER"))
EOF

echo "Index: $INDEX"
echo "Master log: $MASTER"
