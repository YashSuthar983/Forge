#!/bin/bash
# Build the HPR diagnostic probes against an existing build/ tree.
#   usage: tools/fo_diag/build.sh [BUILD_DIR]   (default: build)
set -e
cd "$(dirname "$0")/../.."
B=${1:-build}
INC=""
for d in core sparse model io backend engines la presolve certify search; do
  INC="$INC -I src/$d/include"
done
for p in fo_probe fo_probe2 dualbnd; do
  g++ -O2 -std=c++20 $INC "tools/fo_diag/$p.cpp" -o "$B/$p" \
    -L "$B" -lforge_engines -lforge_presolve -lforge_io -lforge_model -lforge_backend \
    -lforge_la_cpu -lforge_sparse -lforge_certify -lforge_core -lvulkan
  echo "built $B/$p"
done
