# SIH26119 — copy-paste + verified comparisons

**One-time setup on a fresh clone.** The baseline venv holds the *external*
solvers (HiGHS/SciPy) used only as separate processes for comparison. It is
deliberately not in the repository — it contains `libhighs.so`, and shipping a
competitor's binary would contradict `docs/clean_room_policy.md`. Recreate it:

```bash
python3 -m venv benchmarks/.venv-baseline
benchmarks/.venv-baseline/bin/pip install -r benchmarks/requirements-baseline.txt
```

Only the `PY=` comparison beats need it; every `$SOLVE` / `$CHECK` beat runs
against SOR alone and works immediately after `cmake --build`.

```bash
cd /home/yash/Desktop/Sih/sor
mkdir -p demo_out
SOLVE=./build/sor_solve; [ -x ./build-native/sor_solve ] && SOLVE=./build-native/sor_solve
CHECK=./build/sor_check; [ -x ./build-native/sor_check ] && CHECK=./build-native/sor_check
PY=benchmarks/.venv-baseline/bin/python
```

---

## Help

```bash
$SOLVE
$CHECK
```

---

## Demo toys — verified live (SOR vs HiGHS)

Host check: `yash-Bravo-15-B5DD` · HiGHS **1.15.1** external · all ✅ obj agree

| Kind | File | SOR | SOR obj | SOR time | HiGHS | HiGHS obj | agree |
|------|------|-----|--------:|---------:|-------|----------:|-------|
| **LP** | `blend_s42.mps` | Optimal · ProvedOptimalFP | 1708676.13 | ~0.06 ms | Optimal | 1708676.13 | ✅ |
| **QP** | `dispatch_s42.qps` | Optimal · ProvedKKT | 4224.15 | ~0.005 ms | Optimal | 4224.15 | ✅ |
| **MILP** | `schedule_s42.mps` | Optimal · ProvedGlobalEpsilon | 535.63 | ~0.16 ms | Optimal | 535.63 | ✅ |
| **check** | blend.sol | `VERIFIED` | — | — | — | — | — |

### LP

```bash
$SOLVE examples/crude_blending/blend_s42.mps \
  --engine simplex --method auto --backend cpu \
  --solution-out demo_out/blend.sol

$PY scripts/run_highs_baseline.py examples/crude_blending/blend_s42.mps --time-limit 30
```

```bash
$SOLVE examples/crude_blending/blend_s42.mps \
  --engine simplex --method dual --backend cpu \
  --solution-out demo_out/blend_dual.sol
```

```bash
$SOLVE examples/crude_blending/blend_s42.mps \
  --engine simplex --method primal --backend cpu \
  --solution-out demo_out/blend_primal.sol
```

### Independent check

```bash
$CHECK examples/crude_blending/blend_s42.mps demo_out/blend.sol
```

### QP

```bash
$SOLVE examples/dispatch/dispatch_s42.qps \
  --engine qp --backend cpu \
  --solution-out demo_out/dispatch.sol

$PY scripts/run_highs_qp_baseline.py examples/dispatch/dispatch_s42.qps --time-limit 30
```

### MILP

```bash
$SOLVE examples/scheduling/schedule_s42.mps \
  --engine milp --method auto --backend cpu \
  --time-limit 30 --verbose \
  --solution-out demo_out/schedule.sol

$PY scripts/run_highs_baseline.py examples/scheduling/schedule_s42.mps --time-limit 30
```

### First-order (HPR / PDHG) — expect Feasible / Interrupted, not Optimal

```bash
$SOLVE examples/sparse500.mps \
  --engine hpr --backend cpu \
  --max-iter 50000 --time-limit 30 \
  --solution-out demo_out/hpr_cpu.sol
```

```bash
$SOLVE examples/sparse500.mps \
  --engine hpr --backend vulkan \
  --max-iter 50000 --time-limit 30 \
  --solution-out demo_out/hpr_vulkan.sol
```

```bash
$SOLVE examples/sparse500.mps \
  --engine pdhg --backend cpu \
  --max-iter 50000 --time-limit 30 \
  --solution-out demo_out/pdhg.sol
```

### All three side-by-side (one paste)

```bash
cd /home/yash/Desktop/Sih/sor
SOLVE=./build/sor_solve; [ -x ./build-native/sor_solve ] && SOLVE=./build-native/sor_solve
PY=benchmarks/.venv-baseline/bin/python

echo '===== LP · SOR ====='
$SOLVE examples/crude_blending/blend_s42.mps --engine simplex --method auto --backend cpu
echo '===== LP · HiGHS ====='
$PY scripts/run_highs_baseline.py examples/crude_blending/blend_s42.mps --time-limit 30

echo '===== QP · SOR ====='
$SOLVE examples/dispatch/dispatch_s42.qps --engine qp --backend cpu
echo '===== QP · HiGHS ====='
$PY scripts/run_highs_qp_baseline.py examples/dispatch/dispatch_s42.qps --time-limit 30

echo '===== MILP · SOR ====='
$SOLVE examples/scheduling/schedule_s42.mps --engine milp --method auto --backend cpu --time-limit 30
echo '===== MILP · HiGHS ====='
$PY scripts/run_highs_baseline.py examples/scheduling/schedule_s42.mps --time-limit 30
```

---

## Clean-room

```bash
ldd $SOLVE | head
```

---

## Netlib LP · vs HiGHS · `compare-netlib-20260905-162536`

Host `yash-Bravo-15-B5DD` · 93 instances · 30 s / instance

| Solver | Solved | Obj match | SGM time | vs HiGHS |
|--------|-------:|----------:|---------:|---------:|
| **SOR-simplex** | **92/93** | **92/92** | 0.1915 s | **~2.2×** slower |
| HiGHS | 93/93 | 93/93 | 0.0870 s | — |
| SOR-hpr | 17/93 | 17/93 | 0.5342 s | FO only |
| SOR-pdhg | 8/93 | 8/93 | 0.3556 s | FO only |

Unsolved: **`dfl001`** (Interrupted). Do **not** claim 93/93.

```bash
# smoke
python3 scripts/run_compare.py --suite netlib --limit 10 --time-limit 20

# full
python3 scripts/run_compare.py --suite netlib --time-limit 30

# SOR-only
python3 scripts/run_netlib.py --engine simplex --time-limit 60
```

---

## LP benchmark — crude blend ladder · `industrial-perf-20260905-163821`

Host `yash-Bravo-15-B5DD` · seed 42 · HiGHS external · 120 s limit

### Timing

| Tier | size | SOR | HiGHS | speedup | obj agree |
|------|------|----:|------:|--------:|-----------|
| S | 18×20 | **1.56 ms** | 0.73 ms | 0.47× | ✅ |
| M | 62×90 | **3.03 ms** | 2.32 ms | 0.77× | ✅ |
| L | 162×280 | **13.2 ms** | 20.0 ms | **1.51×** | ✅ |
| XL | 302×650 | **50.7 ms** | 74.3 ms | **1.47×** | ✅ |
| XXL | 602×1500 | **266 ms** | 693 ms | **2.61×** | ✅ |
| HUGE | 1002×3000 | **1.14 s** | 4.32 s | **3.80×** | ✅ |

All SOR statuses: **Optimal**.

### Objectives

| Tier | SOR obj | HiGHS obj |
|------|--------:|----------:|
| S | 21218603.575 | 21218603.575 |
| M | 406858886.13 | 406858886.13 |
| L | 3811632020.6 | 3811632020.6 |
| XL | 17653904010.0 | 17653904010.0 |
| XXL | 83686115383.0 | 83686115383.0 |
| HUGE | 281735104090.0 | 281735104088.0 |

**Pitch:** SOR matches HiGHS on all blend rungs; faster from L→HUGE.

```bash
python3 scripts/run_industrial_perf.py --sizes S --kinds blend_lp --time-limit 30
# full ladder (long):
# python3 scripts/run_industrial_perf.py --kinds blend_lp --time-limit 120
```

---

## MILP benchmark — schedule ladder · `industrial-perf-20260905-163821`

Host `yash-Bravo-15-B5DD` · seed 42 · HiGHS external · 120 s · max nodes 10000

### Timing

| Tier | size | SOR | HiGHS | speedup | obj agree |
|------|------|----:|------:|--------:|-----------|
| S | 168×288 | **3.52 ms** | 5.60 ms | **1.59×** | ✅ |
| M | 624×1152 | **17.4 ms** | 17.2 ms | ~1× | ✅ |
| L | 2016×3840 | 136 ms | **44.3 ms** | 0.33× | ✅ |
| XL | 5208×10080 | 900 ms | **133 ms** | 0.15× | ✅ |
| XXL | 13776×26880 | 4.60 s | **408 ms** | 0.09× | ✅ |
| HUGE | 34272×67200 | 24.9 s | **1.31 s** | 0.05× | ✅ |

All SOR statuses: **Optimal** · all objectives agree.

### Objectives

| Tier | SOR obj | HiGHS obj |
|------|--------:|----------:|
| S | 2791.4649 | 2791.4649 |
| M | 9197.1363 | 9197.1363 |
| L | 27850.2342 | 27850.2342 |
| XL | 63789.6816 | 63789.6816 |
| XXL | 180354.3728 | 180354.3728 |
| HUGE | 447369.0216 | 447369.0216 |

**Pitch:** Correct on every rung; HiGHS still faster on L→HUGE — say the gap honestly.

```bash
python3 scripts/run_industrial_perf.py --sizes S --kinds schedule_milp --time-limit 30
```

---

## QP benchmark — economic dispatch ladder · `industrial-perf-20260905-163821`

Host `yash-Bravo-15-B5DD` · seed 42 · HiGHS-QP external · 120 s limit

### Timing

| Tier | gens | SOR | HiGHS-QP | speedup | obj agree |
|------|-----:|----:|---------:|--------:|-----------|
| S | 40 | **1.63 ms** | 1.26 ms | 0.77× | ✅ |
| M | 200 | **2.08 ms** | 9.50 ms | **4.58×** | ✅ |
| L | 1000 | **4.31 ms** | 714 ms | **166×** | ✅ |
| XL | 4000 | **12.3 ms** | 69.3 s | **~5600×** | ✅ |
| XXL | 8000 | **23.5 ms** | timeout (120 s) | — | ❌ |
| HUGE | 10000 | **29.1 ms** | timeout (120 s) | — | ❌ |

All SOR statuses: **Optimal** (diagonal / one-row KKT path).

### Objectives

| Tier | SOR obj | HiGHS obj |
|------|--------:|----------:|
| S | 46819.048 | 46819.048 |
| M | 232310.13 | 232310.13 |
| L | 1150294.1 | 1150294.1 |
| XL | 4545674.4 | 4545674.4 |
| XXL | 9095410.7 | 9326779.1 (incomplete) |
| HUGE | 11417149.6 | 12043434.0 (incomplete) |

### Demo QP
`compare-new-all-20260905-164417`: Optimal **4224.15** in ~**1.9 ms**, matches HiGHS ✅

**Pitch:** S→XL correct and much faster at scale; XXL/HUGE say Optimal + fast, but **do not claim HiGHS agreement** (timeout).

```bash
python3 scripts/run_industrial_perf.py --sizes S --kinds dispatch_qp --time-limit 30
```

---

## One-shot industrial S (LP + MILP + QP vs HiGHS)

```bash
python3 scripts/run_industrial_perf.py --sizes S --time-limit 30
```

---

## Web UI

```bash
./web/run.sh
# http://127.0.0.1:8765
```

---

## Packed demo

```bash
./scripts/demo.sh all
```

---

## All CLI pastes (one block)

```bash
cd /home/yash/Desktop/Sih/sor
mkdir -p demo_out
SOLVE=./build/sor_solve; [ -x ./build-native/sor_solve ] && SOLVE=./build-native/sor_solve
CHECK=./build/sor_check; [ -x ./build-native/sor_check ] && CHECK=./build-native/sor_check
PY=benchmarks/.venv-baseline/bin/python

$SOLVE
$CHECK

$SOLVE examples/crude_blending/blend_s42.mps \
  --engine simplex --method auto --backend cpu \
  --solution-out demo_out/blend.sol
$CHECK examples/crude_blending/blend_s42.mps demo_out/blend.sol
$PY scripts/run_highs_baseline.py examples/crude_blending/blend_s42.mps --time-limit 30

$SOLVE examples/dispatch/dispatch_s42.qps \
  --engine qp --backend cpu \
  --solution-out demo_out/dispatch.sol
$PY scripts/run_highs_qp_baseline.py examples/dispatch/dispatch_s42.qps --time-limit 30

$SOLVE examples/scheduling/schedule_s42.mps \
  --engine milp --method auto --backend cpu \
  --time-limit 30 --verbose \
  --solution-out demo_out/schedule.sol
$PY scripts/run_highs_baseline.py examples/scheduling/schedule_s42.mps --time-limit 30

$SOLVE examples/sparse500.mps \
  --engine hpr --backend cpu \
  --max-iter 50000 --time-limit 30 \
  --solution-out demo_out/hpr_cpu.sol

ldd $SOLVE | head

python3 scripts/run_compare.py --suite netlib --limit 10 --time-limit 20
python3 scripts/run_industrial_perf.py --sizes S --time-limit 30
```
