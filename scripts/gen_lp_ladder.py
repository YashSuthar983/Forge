#!/usr/bin/env python3
"""Generate the sparse-LP size ladder for the CPU/GPU crossover study.

Netlib's largest instance is maros-r7 at 144,848 nnz. Bandwidth-bound SpMV needs
roughly 1e6+ nnz before a discrete GPU beats a cache-resident CPU loop, so the
whole Netlib suite sits below the crossover and cannot demonstrate GPU benefit
either way. This ladder spans the crossover so it can be located rather than
assumed -- including the sizes where the GPU LOSES, which is the honest half of
the result.

Device memory check: CSR + CSC in fp64 costs ~24 bytes/nnz (8 val + 4 idx, twice),
so 5e6 nnz is ~120 MB of the RX 5500M's 4 GB. The binding constraint is MPS text
size on disk, not VRAM.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmarks" / "synthetic" / "mps"

# (name, rows, cols, nnz_per_row) -> target nnz
LADDER = [
    ("lp_1e5",  20_000,  20_000,   5),   # ~1.0e5  below crossover, Netlib scale
    ("lp_5e5",  50_000,  50_000,  10),   # ~5.0e5
    ("lp_1e6", 100_000, 100_000,  10),   # ~1.0e6  crossover expected near here
    ("lp_2e6", 100_000, 100_000,  20),   # ~2.0e6
    ("lp_5e6", 100_000, 100_000,  50),   # ~5.0e6  well above crossover
]

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gen = ROOT / "scripts" / "gen_sparse_lp.py"
    for name, rows, cols, per_row in LADDER:
        dst = OUT / f"{name}.mps"
        if dst.exists():
            print(f"  {name:<8} exists, skipping ({dst.stat().st_size/1048576:.1f} MB)")
            continue
        cmd = [sys.executable, str(gen), "--rows", str(rows), "--cols", str(cols),
               "--nnz-per-row", str(per_row), "--seed", "42", "-o", str(dst)]
        print(f"  {name:<8} {rows}x{cols} @ {per_row}/row -> target {rows*per_row:,} nnz",
              flush=True)
        subprocess.run(cmd, check=True)
        print(f"           wrote {dst.stat().st_size/1048576:.1f} MB", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
