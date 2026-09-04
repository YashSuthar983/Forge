# SOR — Clean-room reference policy

**Status:** team rule · binding for all numeric-core work  
**See also:** `architecture.md` §8 · `dependency_ledger.md` §4 · `paper_bibliography.md`

**This file is the single source of truth for the forbidden-dependency list (§"Forbidden dependencies" below).** Other documents reference this list rather than restating it.

---

## Purpose

The SIH problem statement requires a solver **built from mathematical foundations**, not built upon an existing open-source **solver library**. MIT/Apache licenses do **not** override that rule — they only govern copyright if you copy code.

This document defines how the team may use HiGHS, SCIP, and similar projects **without embedding them** and **without porting their implementations**.

**Reading upstream source for understanding is allowed.** Copying, linking, or vendoring it is not.

---

## Forbidden dependencies — the canonical list

Never in the solve path, never linked, never vendored, never translated:

**LP/MIP solvers** — COIN-OR CBC / CLP · HiGHS (including `pdlp_gpu`, which is cuPDLP-C) · GLPK · SCIP / SoPlex / PaPILO · Google OR-Tools / PDLP · NVIDIA cuOpt · CPLEX / Gurobi / Xpress / MOSEK / COPT · SciPy `linprog` · JuMP / MathOptInterface solver backends

**NLP/global solvers** — Ipopt / Bonmin / Couenne / SHOT / BARON

**Numeric kernels** — any third-party sparse LU, sparse Cholesky, or LP/MIP/NLP kernel

**Research first-order LP codes** — cuPDLP / cuPDLP-C / **cuPDLPx** · **HPR-LP / HPR-LP-C** · **PSLP** (the presolver from Cederberg & Boyd, arXiv:2604.23951). These are high-risk for accidental porting because SOR's first-order engine targets the same algorithms. **Reading their source to understand architecture is allowed** under the rules in §"Reading source for understanding" below; **copying or translating it into SOR is forbidden.** Their **papers** remain the primary implementation spec. See `paper_bibliography.md` Phase 1.

### Explicitly allowed

Language standard library · CUDA toolkit and compiler · **Vulkan / SPIR-V and the shader toolchain** (a device specification, not a solver) · HIP / ROCm / SYCL toolchains · dense BLAS/LAPACK for dense blocks only (ledger it) · published papers and textbooks · public benchmark **instances** (MIPLIB, Netlib, QPLIB, MINLPLib, pooling libraries) · HiGHS/SCIP/CBC as an **external process** for differential testing and baselines — never linked, never inside a certificate · **reading upstream solver source for understanding** (see below) — never copied into the tree.

---

## Core rule

```text
Open-source solver repos: READ for understanding, NEVER ship
  — never in the solve path
  — never in git as vendored code
  — never copy-pasted or line-translated into sor/
  — never in a certificate as "our result"
```

**Default workflow:** **papers first, source second, oracle always**

1. Read the **paper** for the algorithm.
2. Optionally read **upstream source** to understand control flow, data structures, and why benchmarks differ (log it — see below).
3. **Close the source tab** and implement independently in SOR.
4. Run HiGHS/SCIP as a **separate binary** on the same MPS and compare numbers.

---

## Reading source for understanding

**Allowed.** Team members may browse HiGHS, SCIP, CBC, cuPDLPx, HPR-LP, PSLP, and similar repositories to:

- understand how a published algorithm is wired in practice (e.g. BFRT, hypersparse FTRAN, dual phase 1);
- map benchmark gaps to concrete subsystems ("where does fit2d spend its iterations?");
- answer design questions when the paper is ambiguous;
- debug SOR vs oracle discrepancies ("HiGHS takes 129 iterations here — what ratio test is active?").

### Rules while reading source

| Rule | Detail |
|---|---|
| **Log significant lookups** | Record in `reference_log.md`: date, person, repo/file, question, outcome, paper used for implementation |
| **Papers remain the implementation spec** | Source informs *what* to build; the paper (or textbook) is what you implement from |
| **No side-by-side coding** | Do **not** keep upstream `.cpp` / `.h` open in one pane while writing the matching SOR file in the other |
| **No transcription** | Do not copy identifiers, function order, comment blocks, or control-flow skeletons from upstream |
| **Re-read from paper before coding** | After closing the source tab, write the algorithm from the paper's description, not from memory of their code |

Brief, unlogged glances (e.g. confirming a public enum name in a header) are fine. Anything that influences a design decision or a PR should be logged.

---

## Allowed

| Practice | Notes |
|---|---|
| HiGHS / SCIP installed **separately** (`highs model.mps`, etc.) | Baseline and differential testing only |
| Read their README, docs, **papers they cite**, and **source code** | Source for understanding; papers for implementation |
| Compare objective, residuals, status, timing on the **same MPS** | Primary debug loop |
| When SOR fails: *"HiGHS says 1.84e6 — fix our bug"* | Oracle-driven debugging |
| Public benchmark **instances** (Netlib, MIPLIB, QPLIB) | Data only — not code |
| CUDA toolkit, dense BLAS/LAPACK for dense blocks | Ledger in dependency policy; not solvers |
| Implement algorithms from **published papers and textbooks** | Required path; source reading supplements, does not replace |

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

**Porting is still derivative.** Reading `HEkkDual.cpp` (or any upstream file) to understand BFRT, then re-typing their function structure line-for-line — in any language — is not acceptable even under this relaxed policy.

---

## Oracle vs source reference (risk)

| Method | Risk of accidental copy | Use |
|---|---|---|
| HiGHS **binary** on same MPS | **Low** — numbers only | **Always** — verification |
| **Papers / textbooks** | **Lowest** | **Primary** implementation spec |
| HiGHS **source, read then close** | **Medium** — manageable with logging + no side-by-side | **Allowed** for understanding |
| HiGHS **source open while coding** | **High** | **Forbidden** |

---

## Team guardrails

| Guardrail | Enforced by |
|---|---|
| `scripts/check_forbidden_deps.sh` in CI | `ldd`, `nm`, CMake link graph |
| **No** `third_party/highs` (or any solver tree) | Code review + repo scan |
| Numeric-core PRs require **2nd reviewer** | Git workflow |
| Commit messages cite **papers**, not upstream line numbers | e.g. `BFRT (Koberstein 2008; Huangfu & Hall 2018)` not `from HiGHS line 420` |
| Significant source lookups logged in **`reference_log.md`** | Code review |
| HiGHS baseline labeled `"kind": "external_process"` in bench JSON | Never inside certificate |

---

## What to tell judges (idea PDF / demo)

> We implemented from published algorithms. HiGHS was used as an **external benchmark process** — never linked, copied, or translated. Where papers were ambiguous, we consulted open-source implementations **for understanding only** and implemented independently. Dependencies are ledgered. CUDA accelerates our kernels; no third-party solver library is in the solve path.

---

## MIT license reminder

HiGHS (MIT), SCIP (Apache), etc. are **legally** copyable with attribution in normal software. **SIH forbids building upon them as the engine** regardless of license. Legal ≠ compliant with the problem statement.

---

## One-line summary

> **Clone the pipeline, not the repo. Papers write the code; source explains the gaps; HiGHS checks the answer.**
