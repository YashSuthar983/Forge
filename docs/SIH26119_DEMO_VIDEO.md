# SIH26119 — Presentable demo (CLI + thin web) for the video

**PS rule (do not fight this):**  
> “A robust optimization engine with a basic application programming interface (API) or command-line interface is sufficient; **a polished graphical user interface is not required**.”  
> Emphasis is on the **solver core**, not modelling tools.

**What that means for us**

| Do | Do not |
|---|---|
| Film **CLI** as the primary product interface | Claim we built a modelling language / Aspen alternative |
| Add a **thin web console** that shells out to `sor_solve` / `sor_check` / `sor_gen` | Make the web UI the story (“look at our dashboard”) |
| Show engine: LP / MILP / QP, proofs, HiGHS compare, industrial cases | Fake Optimal, hide Feasible as Optimal, claim GPU always wins |
| Say “demo UI over our sovereign engine” | Say “polished GUI product” |

Web is **optional packaging for the video**. CLI alone already satisfies the PS.

---

## 1. What you can film **today** (already built)

Binaries: `build-native/sor_solve`, `sor_check`, `sor_gen`  
Examples: `examples/crude_blending/blend_s42.mps`, `examples/scheduling/schedule_s42.mps`, `examples/dispatch/dispatch_s42.qps`

| Beat | Command | What the viewer sees |
|---|---|---|
| Generate industrial case | `./sor_gen all --seed 42 --outdir examples/` | Refinery-shaped public instances |
| Crude blending **LP** | `./sor_solve examples/crude_blending/blend_s42.mps --engine simplex --solution-out /tmp/blend.sol` | `Optimal` + proof level + objective |
| Independent check | `./sor_check examples/crude_blending/blend_s42.mps /tmp/blend.sol` | PASS residuals (sovereignty / honesty) |
| Power dispatch **QP** | `./sor_solve examples/dispatch/dispatch_s42.qps --engine qp` | QP path |
| Schedule **MILP** | `./sor_solve examples/scheduling/schedule_s42.mps --engine milp --time-limit 30 --verbose` | Nodes / incumbent / Feasible or Optimal |
| GPU FO (if Vulkan works) | `./sor_solve examples/sparse500.mps --engine hpr --backend vulkan` | Backend + transfer table |
| Netlib credibility | One small Netlib MPS + flash the Netlib summary table | vs HiGHS, honest SGM |
| Clean-room | `ldd build-native/sor_solve` | No HiGHS/SCIP/CBC linked |

That is already a **complete PS-aligned video** without any web.

---

## 2. Recommended presentable surface: **CLI + thin web console**

### Product story (one sentence for the video)

> SOR is a from-scratch LP / MILP / QP engine. The CLI is the product interface; this web page is a demo console that runs the same binaries.

### Thin web — what to build (1–2 days max)

A single-page app that **does not** reimplement the solver:

```text
Browser
  → small FastAPI / Flask / Node server
      → subprocess: sor_gen | sor_solve | sor_check
  ← JSON: status, proof, objective, time, log tail
```

**Screens (keep to 3):**

1. **Solve** — pick preset (Blend LP / Schedule MILP / Dispatch QP / Upload MPS) → Run → status / proof / objective / wall time  
2. **Verify** — after solve, one-click `sor_check` → pass/fail residuals  
3. **Compare** (optional) — show precomputed HiGHS row for the same preset (external process; never claim in-process)

**Do not build:** modelling canvas, drag-drop constraints, full MIPLIB browser, fancy charts of 50 metrics. That fights the PS.

**Stack suggestion (fastest):**

- Backend: Python FastAPI, `asyncio.create_subprocess_exec` on `sor_solve`  
- Frontend: one HTML + vanilla JS or minimal React  
- Auth: none (localhost demo)  
- Timeout: kill subprocess at 60s so the video never hangs  

---

## 3. Video script (3–4 minutes)

| Time | Shot | Narration / action |
|---|---|---|
| 0:00–0:20 | Title card | SIH26119 · SOR · sovereign LP/MILP/QP engine · MRPL / Smart Automation |
| 0:20–0:45 | Problem | India depends on CPLEX / Gurobi / Xpress; we need an inspectable engine, not another GUI |
| 0:45–1:15 | Architecture one slide | From-scratch · simplex + B&C + QP · GPU where measured · `finalize_result` + `sor_check` |
| 1:15–1:45 | **CLI live** | Blend LP → Optimal; `sor_check` PASS |
| 1:45–2:15 | **CLI live** | Dispatch QP → Optimal; Schedule MILP → incumbent / gap (say Feasible honestly if not proved) |
| 2:15–2:45 | **Web console** (optional) | Same Blend run from browser — “demo UI calling the same engine” |
| 2:45–3:15 | Benchmarks | Netlib 92/93 vs HiGHS table; industrial blend size ladder flash |
| 3:15–3:35 | GPU / clean-room | Vulkan HPR transfer line **or** `ldd` showing no solver libs |
| 3:35–4:00 | Close | Extensible foundation; MIQP/NLP roadmap; GitHub + CLI |

**Cut if short on time:** drop web; keep CLI + table + `sor_check`. That still fully matches the PS.

---

## 4. Preset pack for demos (always use these)

| Preset | File | Engine | Expect |
|---|---|---|---|
| A — Blend | `examples/crude_blending/blend_s42.mps` | `simplex` | Optimal, fast |
| B — Dispatch | `examples/dispatch/dispatch_s42.qps` | `qp` | Optimal, fast |
| C — Schedule | `examples/scheduling/schedule_s42.mps` | `milp` | Feasible or Optimal; use `--time-limit 30` |
| D — Netlib small | e.g. `afiro` / `adlittle` | `simplex` | Optimal, matches HiGHS |
| E — Sparse FO | `examples/sparse500.mps` | `hpr` + `cpu` or `vulkan` | Feasible / residuals; show timing |

Avoid filming huge schedule MILP that times out ugly unless you narrate “incumbent under time limit.”

---

## 5. PS checklist — what the video must prove

| PS demand | How we show it in the video |
|---|---|
| LP + MILP + QP | Presets A, C, B |
| From scratch | `ldd` / spoken “no HiGHS linked” |
| Sparse / robust engine | Simplex on blend + Netlib; mention Harris / LU |
| CLI or API | Live `sor_solve` |
| Benchmarks vs ≥1 solver | Netlib / MIPLIB table vs HiGHS |
| Industrial scope | `sor_gen` blend / schedule / dispatch |
| GPU where measurable | One Vulkan (or honest “CPU FO today; GPU seam ready”) |
| Not polished GUI | Say once: “CLI is enough per PS; web is demo only” |
| Modular later MIQP/NLP | One roadmap line — do not demo unfinished NLP as production |

---

## 6. Build order for “presentable” this week

### Must (film without these and you look incomplete)

1. Freeze 5 preset commands in a `demo.sh` that always works  
2. Re-run Netlib + one MIPLIB-easy table the morning of recording  
3. Practice narration for Feasible vs Optimal  
4. Record CLI-only cut as the safe master  

### Should (makes web + video nicer)

5. Thin FastAPI + one HTML page (upload MPS optional)  
6. Show `sor_check` as a button after every solve  
7. Side panel: status · proof · objective · ms · engine · backend  

### Skip for SIH video

- Full modelling UI  
- Login / cloud multi-tenant  
- Live Mittelmann leaderboard  
- Claiming NLP/MINLP as solved  

---

## 7. Suggested `demo.sh` (CLI film track)

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build-native"
EX="${ROOT}/examples"

echo "=== 1. Blend LP ==="
"$BIN/sor_solve" "$EX/crude_blending/blend_s42.mps" --engine simplex \
  --solution-out /tmp/sor_blend.sol
"$BIN/sor_check" "$EX/crude_blending/blend_s42.mps" /tmp/sor_blend.sol

echo "=== 2. Dispatch QP ==="
"$BIN/sor_solve" "$EX/dispatch/dispatch_s42.qps" --engine qp

echo "=== 3. Schedule MILP ==="
"$BIN/sor_solve" "$EX/scheduling/schedule_s42.mps" --engine milp --time-limit 30

echo "=== 4. First-order (CPU) ==="
"$BIN/sor_solve" "$EX/sparse500.mps" --engine hpr --backend cpu --max-iter 50000

# Optional GPU:
# "$BIN/sor_solve" "$EX/sparse500.mps" --engine hpr --backend vulkan --max-iter 50000

echo "=== 5. Clean-room ==="
ldd "$BIN/sor_solve" | head
```

---

## 8. One-line verdict

**PS wants the engine.** Film **CLI + checker + industrial presets + HiGHS table**. Add a **thin web console** only as a second camera angle that calls the same binaries — never as the main claim.
