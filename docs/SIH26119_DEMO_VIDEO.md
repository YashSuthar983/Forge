# SIH26119 — Demo video (CLI + web)

**Build for this demo:** `sor/build` with **cpu + vulkan**.  
Other backends (if any) stay off-camera — do not discuss them in the video.

**Product line:** CLI is the interface; web is a thin console over the same binaries.

---

## Quick start

```bash
cd /home/yash/Desktop/Sih/sor

# CLI film track (writes demo_out/*.sol + *.log)
chmod +x scripts/demo.sh
./scripts/demo.sh all

# Web console
./web/run.sh
# → http://127.0.0.1:8765
```

| Surface | How to launch | What judges see |
|---------|---------------|-----------------|
| **CLI** | `./scripts/demo.sh all` | Terminal I/O, proofs, `sor_check`, `ldd` |
| **Python API** | `python3 scripts/sor_repl.py` | `solve()` / `check()` |
| **Web** | `./web/run.sh` | Same engines via model list + options |

---

## Inputs / options / outputs

### Engines & backends (this demo)

| | Values |
|--|--------|
| `--engine` | `simplex` · `milp` · `qp` · `hpr` · `pdhg` |
| `--backend` | `cpu` · `vulkan` |
| `--method` | `auto` · `primal` · `dual` (simplex / milp) |
| Other | `--time-limit` · `--max-iter` · `--tol` · `--verbose` · `--solution-out` |

### Presets

| ID | File | Engine | Backend | Expect on camera |
|----|------|--------|---------|------------------|
| **blend** | `examples/crude_blending/blend_s42.mps` | simplex | cpu | `Optimal` · `ProvedOptimalFP` |
| **dispatch** | `examples/dispatch/dispatch_s42.qps` | qp | cpu | `Optimal` · `ProvedKKT` |
| **schedule** | `examples/scheduling/schedule_s42.mps` | milp | cpu | `Optimal` (or Feasible — say it honestly) |
| **hpr_cpu** | `examples/sparse500.mps` | hpr | cpu | Feasible / limit — **not** proved Optimal |
| **hpr_vulkan** | `examples/sparse500.mps` | hpr | vulkan | Feasible + **host↔device** timing lines |

### Output artifacts (`demo_out/`)

| File | From |
|------|------|
| `blend.sol` / `blend.log` | CLI blend |
| `check.log` | `sor_check` |
| `dispatch.sol` / `.log` | QP |
| `schedule.sol` / `.log` | MILP |
| `hpr_cpu.sol` / `hpr_vulkan.sol` | FO |
| `web_*.sol` | Web solves |
| `ldd.txt` | Clean-room |

**Stdout fields to zoom on:** `status` · `proof_level` · `objective` · `timing (ms)` · for HPR Vulkan also `host->device` / `device->host`.

---

## Video script (~3:30)

| Time | Shot | Action / say |
|------|------|----------------|
| 0:00–0:20 | Title | SIH26119 · SOR · Point Blank · sovereign LP/MILP/QP |
| 0:20–0:40 | Problem | Closed foreign solvers; we need an inspectable engine |
| 0:40–1:00 | Architecture slide | Simplex proofs · B&C · QP · Vulkan HPR · `finalize_result` + `sor_check` |
| 1:00–1:35 | **CLI blend** | `./scripts/demo.sh blend` → Optimal · ProvedOptimalFP |
| 1:35–1:55 | **CLI check** | `./scripts/demo.sh check` → **VERIFIED** |
| 1:55–2:20 | **CLI qp + milp** | `./scripts/demo.sh qp` then `milp` |
| 2:20–2:50 | **Web** | Open `http://127.0.0.1:8765` · preset Blend · Solve · Verify |
| 2:50–3:15 | **Vulkan HPR** | Web or CLI `hpr-vulkan` · show transfer lines · “FO ≈ Feasible, simplex proves Optimal” |
| 3:15–3:30 | Clean-room + close | `./scripts/demo.sh cleanroom` · GitHub · “CLI is the product” |

**If short:** drop web; keep CLI blend → check → qp → milp → cleanroom.

---

## CLI cheat sheet (film-ready)

```bash
# 1 Blend LP
./build/sor_solve examples/crude_blending/blend_s42.mps \
  --engine simplex --method auto \
  --solution-out demo_out/blend.sol

# 2 Independent verify
./build/sor_check examples/crude_blending/blend_s42.mps demo_out/blend.sol

# 3 Dispatch QP
./build/sor_solve examples/dispatch/dispatch_s42.qps \
  --engine qp --solution-out demo_out/dispatch.sol

# 4 Schedule MILP
./build/sor_solve examples/scheduling/schedule_s42.mps \
  --engine milp --time-limit 30 --verbose \
  --solution-out demo_out/schedule.sol

# 5 Vulkan HPR (GPU FO)
./build/sor_solve examples/sparse500.mps \
  --engine hpr --backend vulkan --max-iter 50000 --time-limit 30 \
  --solution-out demo_out/hpr_vulkan.sol

# 6 Clean-room
ldd ./build/sor_solve | head
```

Or one shot: `./scripts/demo.sh all`

---

## Web UI — what to click

1. Preset **Blend LP** → **Solve** → read Status / Proof / Objective  
2. **Verify with sor_check** → VERIFIED  
3. Preset **Dispatch QP** → Solve  
4. Preset **Schedule MILP** → Solve (time limit 30)  
5. Preset **HPR (Vulkan GPU)** → Solve → scroll to transfer timing  
6. Optional: paste MPS text → Solve  

Options panel mirrors CLI: engine, backend (`cpu`/`vulkan`), method, time limit, max iter, tol, verbose.

---

## Narration rules

| Do | Do not |
|----|--------|
| Say Feasible vs Optimal correctly | Call HPR “proved Optimal” |
| Say web calls the same CLI binaries | Sell a polished modelling GUI |
| Show Vulkan transfer-inclusive time | Claim GPU always faster |
| Show `ldd` clean of foreign solvers | Compare live to CPLEX/Gurobi as “we beat them” |

---

## PS checklist (video must hit)

| Demand | Beat |
|--------|------|
| LP + MILP + QP | blend · schedule · dispatch |
| From scratch | `ldd` / spoken |
| CLI | `demo.sh` |
| Checker | `sor_check` VERIFIED |
| Industrial | `sor_gen` / blend·schedule·dispatch presets |
| GPU where measurable | HPR + vulkan + transfer lines |
| Not polished GUI | one line: “CLI is enough; web is demo only” |
