#!/usr/bin/env bash
# Smoke test for the Julia GPU sidecar: start it, ping it, shut it down.
#
# Verifies the length-prefixed JSON framing end to end without needing the C++
# build. Exits 77 (SKIP) when julia is unavailable.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PROJECT="$ROOT/tools/julia_gpu"
JULIA="${SOR_JULIA:-julia}"

if ! command -v "$JULIA" >/dev/null 2>&1; then
  echo "SKIP: julia not found (set SOR_JULIA to its path)"
  exit 77
fi
if [[ ! -r "$PROJECT/server.jl" ]]; then
  echo "FAIL: $PROJECT/server.jl not readable"
  exit 1
fi

TMPDIR_RUN="$(mktemp -d)"
IN="$TMPDIR_RUN/in"
OUT="$TMPDIR_RUN/out"
ERR="$TMPDIR_RUN/err"
mkfifo "$IN" "$OUT"
cleanup() {
  [[ -n "${PID:-}" ]] && kill "$PID" 2>/dev/null
  rm -rf "$TMPDIR_RUN"
}
trap cleanup EXIT

"$JULIA" --project="$PROJECT" --startup-file=no "$PROJECT/server.jl" \
         --device "${SOR_JULIA_DEVICE:-auto}" <"$IN" >"$OUT" 2>"$ERR" &
PID=$!

# Hold the request pipe open for the whole session.
exec 3>"$IN"
exec 4<"$OUT"

# Julia JIT warmup is seconds, not milliseconds.
echo "waiting for sidecar to become ready..."
for _ in $(seq 1 300); do
  if grep -q SORGPU_READY "$ERR" 2>/dev/null; then break; fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "FAIL: sidecar exited during startup"
    cat "$ERR"
    exit 1
  fi
  sleep 1
done
if ! grep -q SORGPU_READY "$ERR" 2>/dev/null; then
  echo "FAIL: timed out waiting for SORGPU_READY"
  cat "$ERR"
  exit 1
fi
grep SORGPU_READY "$ERR"

send() {  # send <json>
  local body="$1"
  printf '%s\n' "${#body}" >&3
  printf '%s' "$body" >&3
}

recv() {  # prints the response body
  local n
  IFS= read -r n <&4 || { echo "FAIL: no response header" >&2; exit 1; }
  local body
  IFS= read -r -N "$n" body <&4 || { echo "FAIL: short response" >&2; exit 1; }
  printf '%s' "$body"
}

echo "--- ping ---"
send '{"op":"ping","id":1}'
RESP="$(recv)"
echo "$RESP"
if [[ "$RESP" != *'"pong":true'* && "$RESP" != *'"pong": true'* ]]; then
  echo "FAIL: no pong in response"
  exit 1
fi

echo "--- spmv on a 2x2 identity ---"
send '{"op":"upload_pattern","id":2,"n_rows":2,"n_cols":2,"row_ptr":[0,1,2],"col_idx":[0,1]}'
PAT="$(recv)"; echo "$PAT"
send '{"op":"upload_vals","id":3,"pattern_id":1,"vals":[2.0,3.0]}'
VAL="$(recv)"; echo "$VAL"
send '{"op":"spmv","id":4,"pattern_id":1,"vals_id":1,"x":[1.0,1.0]}'
SPMV="$(recv)"; echo "$SPMV"
# diag(2,3) * [1,1] = [2,3]
if [[ "$SPMV" != *"2.0"* || "$SPMV" != *"3.0"* ]]; then
  echo "FAIL: unexpected spmv result"
  exit 1
fi

echo "--- shutdown ---"
send '{"op":"shutdown","id":5}'
recv; echo

wait "$PID" 2>/dev/null
echo "PASS smoke_julia_gpu"
