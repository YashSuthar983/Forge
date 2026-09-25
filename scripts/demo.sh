#!/usr/bin/env bash
# SOR SIH26119 - CLI film track.
# Usage:
#   ./scripts/demo.sh              # full track
#   ./scripts/demo.sh blend        # one beat
#   ./scripts/demo.sh check
#   ./scripts/demo.sh qp
#   ./scripts/demo.sh milp
#   ./scripts/demo.sh hpr-cpu
#   ./scripts/demo.sh hpr-vulkan
#   ./scripts/demo.sh cleanroom
#   ./scripts/demo.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for cand in "$ROOT/build" "$ROOT/build-native"; do
  if [[ -x "$cand/sor_solve" ]]; then BIN="$cand"; break; fi
done
if [[ -z "${BIN:-}" ]]; then
  echo "error: build sor_solve first (cmake -S . -B build && cmake --build build -j)" >&2
  exit 1
fi

EX="$ROOT/examples"
OUT="$ROOT/demo_out"
mkdir -p "$OUT"

banner() { printf '\n======== %s ========\n' "$1"; }

run_blend() {
  banner "1. Blend LP  (simplex → Optimal + ProvedOptimalFP)"
  echo "INPUT:  $EX/crude_blending/blend_s42.mps"
  echo "OPTS:   --engine simplex --method auto --solution-out $OUT/blend.sol"
  echo "OUTPUT: status / proof_level / objective / timing → $OUT/blend.sol"
  "$BIN/sor_solve" "$EX/crude_blending/blend_s42.mps" \
    --engine simplex --method auto \
    --solution-out "$OUT/blend.sol" | tee "$OUT/blend.log"
}

run_check() {
  banner "2. Independent check  (sor_check on written .sol)"
  local model="${1:-$EX/crude_blending/blend_s42.mps}"
  local sol="${2:-$OUT/blend.sol}"
  if [[ ! -f "$sol" ]]; then
    echo "no solution yet - running blend first"
    run_blend
  fi
  echo "INPUT:  model=$model  sol=$sol"
  echo "OUTPUT: VERIFIED / FAIL residuals"
  "$BIN/sor_check" "$model" "$sol" | tee "$OUT/check.log"
}

run_qp() {
  banner "3. Dispatch QP  (convex QP → Optimal + ProvedKKT)"
  echo "INPUT:  $EX/dispatch/dispatch_s42.qps"
  echo "OPTS:   --engine qp --solution-out $OUT/dispatch.sol"
  "$BIN/sor_solve" "$EX/dispatch/dispatch_s42.qps" \
    --engine qp \
    --solution-out "$OUT/dispatch.sol" | tee "$OUT/dispatch.log"
}

run_milp() {
  banner "4. Schedule MILP  (B&C, time-limit 30s)"
  echo "INPUT:  $EX/scheduling/schedule_s42.mps"
  echo "OPTS:   --engine milp --time-limit 30 --verbose --solution-out $OUT/schedule.sol"
  echo "NOTE:   say Feasible honestly if not Optimal"
  "$BIN/sor_solve" "$EX/scheduling/schedule_s42.mps" \
    --engine milp --time-limit 30 --verbose \
    --solution-out "$OUT/schedule.sol" | tee "$OUT/schedule.log"
}

run_hpr_cpu() {
  banner "5a. HPR first-order (CPU)  - FO ≠ proved Optimal"
  echo "INPUT:  $EX/sparse500.mps"
  echo "OPTS:   --engine hpr --backend cpu --max-iter 50000 --time-limit 15"
  "$BIN/sor_solve" "$EX/sparse500.mps" \
    --engine hpr --backend cpu \
    --max-iter 50000 --time-limit 15 \
    --solution-out "$OUT/hpr_cpu.sol" | tee "$OUT/hpr_cpu.log"
}

run_hpr_vulkan() {
  banner "5b. HPR first-order (Vulkan GPU)  - transfer-inclusive timing"
  echo "INPUT:  $EX/sparse500.mps"
  echo "OPTS:   --engine hpr --backend vulkan --max-iter 50000 --time-limit 30"
  echo "LOOK:   host->device / device->host lines"
  "$BIN/sor_solve" "$EX/sparse500.mps" \
    --engine hpr --backend vulkan \
    --max-iter 50000 --time-limit 30 \
    --solution-out "$OUT/hpr_vulkan.sol" | tee "$OUT/hpr_vulkan.log"
}

run_cleanroom() {
  banner "6. Clean-room  (no foreign solver in the binary)"
  echo "COMMAND: ldd $BIN/sor_solve"
  if command -v ldd >/dev/null; then
    ldd "$BIN/sor_solve" | tee "$OUT/ldd.txt"
    if ldd "$BIN/sor_solve" | grep -Ei 'highs|scip|cbc|gurobi|cplex|cuopt' ; then
      echo "FAIL: unexpected solver library linked" >&2
      exit 1
    fi
    echo "OK: no HiGHS/SCIP/CBC/Gurobi/CPLEX/cuOpt in ldd"
  else
    echo "(ldd not available on this OS)"
  fi
}

run_gen() {
  banner "0. Generate industrial presets (optional)"
  "$BIN/sor_gen" all --seed 42 --outdir "$EX" | tee "$OUT/gen.log"
}

usage() {
  cat <<EOF
SOR demo CLI - film track

  $0 all          full video track
  $0 blend        Blend LP + write .sol
  $0 check        sor_check on blend.sol
  $0 qp           Dispatch QP
  $0 milp         Schedule MILP
  $0 hpr-cpu      HPR on CPU
  $0 hpr-vulkan   HPR on Vulkan (GPU)
  $0 cleanroom    ldd clean-room proof
  $0 gen          regenerate examples with sor_gen

Binaries: $BIN
Outputs:  $OUT
EOF
}

cmd="${1:-all}"
case "$cmd" in
  -h|--help|help) usage ;;
  gen)            run_gen ;;
  blend)          run_blend ;;
  check)          run_check ;;
  qp)             run_qp ;;
  milp)           run_milp ;;
  hpr-cpu)        run_hpr_cpu ;;
  hpr-vulkan)     run_hpr_vulkan ;;
  cleanroom)      run_cleanroom ;;
  all)
    run_blend
    run_check
    run_qp
    run_milp
    run_hpr_cpu
    # Vulkan may be slow/unavailable - still film if present
    if "$BIN/sor_solve" "$EX/testlp.mps" --engine hpr --backend vulkan --max-iter 10 --time-limit 5 >/dev/null 2>&1; then
      run_hpr_vulkan
    else
      echo "(skip hpr-vulkan - backend not available)"
    fi
    run_cleanroom
    banner "DONE - logs + .sol files in $OUT"
    ls -la "$OUT"
    ;;
  *)
    echo "unknown beat: $cmd" >&2
    usage
    exit 2
    ;;
esac
