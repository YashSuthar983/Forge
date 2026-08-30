# SOR — Clean-room reference policy

**Status:** team rule · binding for all numeric-core work  
**See also:** `master_spec.md` §7 · `architecture.md` §13 · `dependency_ledger.md` §4

**This file is the single source of truth for the forbidden-dependency list (§"Forbidden dependencies" below).** Other documents reference this list rather than restating it.

---

## Purpose

The SIH problem statement requires a solver **built from mathematical foundations**, not built upon an existing open-source **solver library**. MIT/Apache licenses do **not** override that rule — they only govern copyright if you copy code.

This document defines how the team may use HiGHS, SCIP, and similar projects **without embedding them** and **without porting their implementations**.

---

## Forbidden dependencies — the canonical list

Never in the solve path, never linked, never vendored, never translated:

**LP/MIP solvers** — COIN-OR CBC / CLP · HiGHS (including `pdlp_gpu`, which is cuPDLP-C) · GLPK · SCIP / SoPlex / PaPILO · Google OR-Tools / PDLP · NVIDIA cuOpt · CPLEX / Gurobi / Xpress / MOSEK / COPT · SciPy `linprog` · JuMP / MathOptInterface solver backends

**NLP/global solvers** — Ipopt / Bonmin / Couenne / SHOT / BARON

**Numeric kernels** — any third-party sparse LU, sparse Cholesky, or LP/MIP/NLP kernel

**Research first-order LP codes** — cuPDLP / cuPDLP-C / **cuPDLPx** · **HPR-LP / HPR-LP-C** · **PSLP** (the presolver from Cederberg & Boyd, arXiv:2604.23951). These are the codes SOR's first-order engine is modelled on, so they are the *highest-risk* sources to open. Their **papers** are the required input; their repositories are forbidden. See `paper_bibliography.md` Phase 1.

### Explicitly allowed

Language standard library · CUDA toolkit and compiler · **Vulkan / SPIR-V and the shader toolchain** (a device specification, not a solver) · HIP / ROCm / SYCL toolchains · dense BLAS/LAPACK for dense blocks only (ledger it) · published papers and textbooks · public benchmark **instances** (MIPLIB, Netlib, QPLIB, MINLPLib, pooling libraries) · HiGHS/SCIP/CBC as an **external process** for differential testing and baselines — never linked, never inside a certificate.

---


## Core rule

```text
Open-source solver repos: REFERENCE ONLY
  — never in the solve path
  — never in git as vendored code
  — never in a certificate as "our result"
```

**Default workflow:** **oracle, not source browsing** — run HiGHS/SCIP as a **separate binary** on the same MPS and compare numbers.

---

## Allowed

| Practice | Notes |
|---|---|
| HiGHS / SCIP installed **separately** (`highs model.mps`, etc.) | Baseline and differential testing only |
| Read their README, docs, and **papers they cite** | Preferred input for design |
| Compare objective, residuals, status, timing on the **same MPS** | Primary debug loop |
| When SOR fails: *"HiGHS says 1.84e6 — fix our bug"* | Oracle-driven debugging |
| Public benchmark **instances** (Netlib, MIPLIB, QPLIB) | Data only — not code |
| CUDA toolkit, dense BLAS/LAPACK for dense blocks | Ledger in dependency policy; not solvers |
| Implement algorithms from **published papers and textbooks** | Required clean-room path |

---

## Allowed with caution (one person only)

Use only when stuck after paper + oracle comparison.

1. Open upstream source to answer: **"Does this presolve rule / trick appear in the literature?"**
2. If **yes** → close the repo tab → implement from the **paper**.
3. If **no** → **do not** implement that rule from their code.

**Never** side-by-side reimplementation with their `.cpp` open.

---

## Forbidden

| Action | Why |
|---|---|
| Copy-paste any file or snippet into `sor/` | Derivative work + SIH disqualification |
| `third_party/highs`, git submodule to SCIP/CBC | Vendored solver |
| Side-by-side reimplementation while reading their `.cpp` | Porting |
| "Same structure, our variable names" | Still derivative |
| `#include` anything from HiGHS / SCIP / cuOpt / PaPILO | Embedded solver |
| Link `-lhighs`, `-lscip`, etc. | Embedded solver |
| Return HiGHS/SCIP output in **SOR certificates** | Misrepresentation |
| Ship anything that fails `check_forbidden_deps.sh` or `ldd` | CI gate |

**Porting is still derivative.** Reading `HEkkDual.cpp` (or any upstream file) and re-typing the logic in C++ — in any language — is not clean-room.

---

## Oracle vs source reference (risk)

| Method | Risk of accidental copy | Use |
|---|---|---|
| HiGHS **binary** on same MPS | **Low** — numbers only | **Default** |
| HiGHS **GitHub open in IDE** | **High** — especially under deadline | Avoid; log if unavoidable |
| **Papers / textbooks only** | **Lowest** | **Preferred** for implementation |

---

## Team guardrails

| Guardrail | Enforced by |
|---|---|
| `scripts/check_forbidden_deps.sh` in CI | `ldd`, `nm`, CMake link graph |
| **No** `third_party/highs` (or any solver tree) | Code review + repo scan |
| Numeric-core PRs require **2nd reviewer** | Git workflow |
| Commit messages cite **papers**, not line numbers | e.g. `Harris ratio test (Harris 1973)` not `from HiGHS line 420` |
| HiGHS baseline labeled `"kind": "external_process"` in bench JSON | Never inside certificate |

---

## What to tell judges (idea PDF / demo)

> We implemented from published algorithms. HiGHS was used only as an **external benchmark process** — never linked, copied, or translated. Dependencies are ledgered. CUDA accelerates our kernels; no third-party solver library is in the solve path.

Do **not** say: *"We referenced HiGHS source heavily."*

---

## MIT license reminder

HiGHS (MIT), SCIP (Apache), etc. are **legally** copyable with attribution in normal software. **SIH forbids building upon them as the engine** regardless of license. Legal ≠ compliant with the problem statement.

---

## One-line summary

> **Clone the pipeline, not the repo. Papers write the code; HiGHS only checks the answer.**
