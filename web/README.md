# FORGE Demo Console (web)

A browser console for the FORGE engine. Pick a model, press **Solve**, and get the answer, the solver's own proof level, and an independent re-check.

It is a thin FastAPI layer over the same `sor_solve` and `sor_check` binaries as the CLI. Every solve is a subprocess call, and the console shows the exact command it ran, so anything you see in the browser can be reproduced in a terminal. It is a demo surface, not a modelling environment.

## Run

```bash
# from the repo root
cmake --build build -j"$(nproc)"   # builds sor_solve, sor_check

./web/run.sh
# or: cd web && source .venv/bin/activate && python app.py
```

Open **http://127.0.0.1:8765**. The first run creates `web/.venv` and installs the requirements.

## How it works

The screen has three parts: a sidebar to choose a model, a header to configure and run it, and below that the result next to the model source.

### 1. Choose a model (sidebar)

The sidebar has three tabs.

- **Solution (.sol).** After every solve the result opens on the solver's own solution file: status, proof and objective, plus every variable (`x`) and dual (`y`) value named from the model, with search, a non-zero filter, Clear, Copy, Download and the raw file. Values are read from the file, nothing is precomputed.
- **Models.** Every `.mps`, `.qps` and `.qplib` file (optionally `.gz`) under `examples/` and `benchmarks/`, grouped by folder. Each entry shows its type badge (**LP**, **MILP**, **QP**, **MIQP**) and its size (rows × columns, nonzeros, file size), read from the file itself. Generated examples show their seed and generator parameters instead. The search box filters by name, type or folder.
- **Write.** Type a model as plain equations (`Maximize`/`Minimize`, `Subject To`, `Bounds`, `Binary`/`General`, `End`). Start from a template: Tiny LP, Diet, Knapsack or Mini crude blend. The equations are converted to MPS as you type, and the **MPS preview** tab shows the result.
**Open file** (top right) uploads your own model from disk.

### 2. Configure and run (header)

The header shows the model's name, type, path and size, plus notes such as "synthetic instance" or "large model".

- **Engine.** Defaults to *Match model*, which picks the engine for the model's type: `simplex` for LP, `milp` for integer models, `qp` for quadratic models, `auto` for QPLIB files. This matters because the plain CLI default solves only the LP relaxation of a MILP and ignores a QP's quadratic term. Any engine the binary offers can be chosen instead.
- **Hardware.** The backends the binary supports. Each one is tested when the server starts, and any that don't work in this build are greyed out (see [GPU backends](#gpu-backends)).
- **Time limit.** Seconds, 30 by default.
- **Options.** Simplex method, thread count, whether to run the independent checker automatically, verbose solver log, and why any hardware backend is unavailable.
- **Solve**, or **⌘/Ctrl + Enter**.

The source editor on the right is editable. If you change the MPS, the header shows an *edited* badge and your edited version is what gets solved. Models larger than 400 KB open read-only, and Solve reads the full file from disk.

### 3. Read the answer (result panel)

The top of the result panel is a summary:

- **Status** (Optimal, Feasible, Infeasible, Unbounded, Time limit, …), colour-coded, with the **proof level** from the solver's certification step (`ProvedOptimalFP`, `ProvedGlobalEpsilon`, `ProvedKKT`, …) and the wall-clock time.
- **Objective value** and the solver's own one-line explanation.
- **Four key metrics** from the solver report, such as dual bound, gap, primal violation and dual residual.
- **Verification line**, e.g. `VERIFIED · 5/5 checks passed`.

Below the summary are four tabs:

- **Checks.** Each `sor_check` test with its residual and tolerance.
- **Report.** Every field the solver printed.
- **Command.** The exact `sor_solve` call, with a Copy button.
- **Log.** The full solver and checker output.

## Independent check

`sor_check` is a separate binary that re-reads the original model file and recomputes the row and bound residuals, the objective, and the dual/gap values from the solver's solution file. It shares no search code with the solver.

It runs automatically for **LP** models. For **MILP** its dual and gap checks don't apply yet, so a correct answer can be rejected. It doesn't read the quadratic part of **QP** models. For those classes the console says so and offers **Run sor_check anyway** rather than showing a misleading REJECTED. When the checker gains support, add the class to `FORGE_CHECK_CLASSES`.

## Reference results

Measured through the console on an Apple M4 (CPU build), 60 s limit. Each model used its matched engine, and times are wall-clock including process start.

| Model | Type | Rows × Cols | Nonzeros | Engine | Status | Proof level | Objective | Time | Check |
|---|---|---|---:|---|---|---|---:|---:|---|
| `examples/crude_blending/blend_s42` | LP | 8 × 7 | 26 | simplex | Optimal | ProvedOptimalFP | 1,708,676.1279 | 0.01 s | VERIFIED |
| `examples/scheduling/schedule_s42` | MILP | 18 × 24 | 46 | milp | Optimal | ProvedGlobalEpsilon | 535.6339 | 0.01 s | LP only |
| `examples/dispatch/dispatch_s42` | QP | 1 × 4 | 4 | qp | Optimal | ProvedKKT | 4,224.1497 | < 0.01 s | LP only |
| `examples/sparse500` | LP | 500 × 500 | 2,987 | simplex | Optimal | ProvedOptimalFP | −3,752.0357 | 0.01 s | VERIFIED |
| `open-refinery-lp/hydroskimming` | LP | 25 × 26 | 77 | simplex | Optimal | ProvedOptimalFP | −7,275,000 | < 0.01 s | VERIFIED |
| `open-refinery-lp/conversion` | LP | 31 × 31 | 99 | simplex | Optimal | ProvedOptimalFP | −7,275,000 | < 0.01 s | VERIFIED |
| `open-refinery-lp/complex` | LP | 38 × 37 | 128 | simplex | Optimal | ProvedOptimalFP | −7,275,000 | < 0.01 s | VERIFIED |
| `industrial-ladder/blend_lp_xl_s42` | LP | 302 × 650 | 76,300 | simplex | Optimal | ProvedOptimalFP | 17,653,904,010 | 0.06 s | VERIFIED |
| `industrial-ladder/blend_lp_huge_s42` | LP | 1,002 × 3,000 | 1,256,000 | simplex | Optimal | ProvedOptimalFP | 281,735,104,090 | 2.91 s | VERIFIED |
| `industrial-ladder/schedule_milp_xl_s42` | MILP | 5,208 × 10,080 | 20,130 | milp | Optimal | ProvedGlobalEpsilon | 63,789.6816 | 1.45 s | LP only |
| `industrial-ladder/dispatch_qp_huge_s42` | QP | 1 × 10,000 | 10,000 | qp | Optimal | ProvedKKT | 11,417,149.644 | 0.02 s | LP only |

The `examples/` and `industrial-ladder/` models are synthetic, seeded `sor_gen` instances, not MRPL data. The `open-refinery-lp/` models are public (see `benchmarks/industry/README.md`).

## Demo path

1. **Models → Examples → blend_s42**, then Solve: a refinery crude-blending LP, `Optimal`, `ProvedOptimalFP`, VERIFIED in about 10 ms.
2. **Models → Industrial ladder → blend_lp_huge_s42**: 1,002 × 3,000 with 1.26M nonzeros, proved and VERIFIED in about 3 s.
3. **schedule_s42** and **dispatch_s42**: MILP and QP proved optimal by `milp` and `qp`. The Checks tab explains the checker's LP-only scope.
4. **Command** tab: copy the command and run it in a terminal to show it's the same binary.
5. **Write** your own model and solve it.

## GPU backends

The Hardware menu lists the backends in `sor_solve --help` and tests each one when the server starts. Unavailable backends are greyed out, and **Options** explains why:

- **Vulkan** needs the binary built with `SOR_ENABLE_VULKAN=ON`, which requires the Vulkan SDK at build time (on Linux: `libvulkan-dev` and `glslang-tools`). It also needs a GPU that supports fp64 in shaders (`shaderFloat64`): NVIDIA and AMD do, Apple GPUs don't, so it can't run on macOS even through MoltenVK. On a machine that meets both, Vulkan appears automatically. Restart `run.sh` after rebuilding.
- **CUDA** is a stub in the engine today and reports Unsupported.
- Only the first-order engines (`hpr`, `hprqp`, `binquad`) use the GPU. Simplex and MILP always run on the CPU.

## Configuration

Environment variables, `FORGE_*` (the older `SOR_*` names still work):

| Variable | Default | Meaning |
|---|---|---|
| `FORGE_BIN_DIR` | `../build`, then `../build-native` | Directory with `sor_solve`, `sor_check` |
| `FORGE_MODEL_DIRS` | `../examples:../benchmarks` | Folders scanned for models (`:`-separated) |
| `FORGE_MODEL_EXTS` | `.mps,.qps,.qplib` | File types listed |
| `FORGE_TEMPLATE_DIR` | `templates/` | Equation templates for the Write tab (`# title:` / `# about:` headers) |
| `FORGE_CHECK_CLASSES` | `LP` | Model types where `sor_check` runs automatically |
| `FORGE_WEB_TIME_LIMIT` / `FORGE_WEB_MAX_TIME_LIMIT` | `30` / `300` | Default and maximum solve time (s) |
| `FORGE_WEB_TIMEOUT` | `90` | Subprocess timeout when no time limit is set |
| `FORGE_WEB_EDITOR_MAX_BYTES` | `400000` | Larger models open read-only and are solved from disk |
| `FORGE_WEB_UPLOAD_MAX_BYTES` | `40000000` | Upload size cap |
| `FORGE_WEB_HOST` / `FORGE_WEB_PORT` | `127.0.0.1` / `8765` | Bind address |

## Files

| Path | Purpose |
|---|---|
| `app.py` | FastAPI server: model discovery, solve / check / generate endpoints |
| `lp_text.py` | Plain-equation → MPS converter for the Write tab |
| `templates/*.lp` | Write-tab starting models |
| `static/` | `index.html`, `app.css`, `app.js`, logo and favicon |
| `run.sh` | Finds the binaries, sets up `.venv`, starts the server |
