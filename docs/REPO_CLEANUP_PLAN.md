# Repository cleanup plan

**Date:** 2026-09-05 · Audited against the working tree, not from memory.
Every number below came from `git ls-files`, `git status`, and `du` on this checkout.

---

## P0 — Clean-room violation: HiGHS is committed to the repository

**This is the only item that is not cosmetic.** It undermines the single claim the
project is built around.

```
$ git ls-files benchmarks/.venv-baseline | wc -l
2214
$ git ls-files | grep libhighs
benchmarks/.venv-baseline/lib/python3.12/site-packages/highspy/libhighs.so.1
benchmarks/.venv-baseline/lib/python3.12/site-packages/highspy/libhighs.so.1.15.1   (5.1 MB)
$ git log --oneline -1 -- benchmarks/.venv-baseline
6ef8896 Initial commit
```

`docs/clean_room_policy.md` states the rule plainly:

> Open-source solver repos: READ for understanding, NEVER ship
>   — never in git as vendored code

`.gitignore` already contains `benchmarks/.venv-baseline/` **with a comment saying exactly
why** — "never commit (HiGHS/CBC/SciPy live here, and committing them would also muddy the
clean-room story)". But a `.gitignore` rule has no effect on files that are *already tracked*,
and these were committed in the very first commit. The rule has been silently doing nothing.

Scale: 2 214 of the 2 442 tracked files under `benchmarks/` are this venv — i.e. **the
majority of tracked files in the repository are a vendored copy of a competitor solver**,
including its compiled `.so`.

### Fix, in two stages

**Stage 1 — stop tracking (safe, reversible, do immediately):**

```bash
git rm -r --cached benchmarks/.venv-baseline
git commit -m "Untrack the HiGHS/SciPy baseline venv (clean-room policy)"
```

`--cached` leaves the directory on disk, so benchmarking keeps working. The existing
`.gitignore` rule then starts doing its job.

**Stage 2 — purge from history (needs your decision):** Stage 1 removes it from the *current*
tree, but `libhighs.so` remains reachable in commit `6ef8896` forever. Anyone cloning and
running `git log -p` or `git ls-tree 6ef8896` still finds vendored HiGHS. For a submission
whose differentiator is "built from mathematical foundations, not on an existing solver", that
is worth removing properly:

```bash
# git-filter-repo is the maintained tool (pip install git-filter-repo)
git filter-repo --path benchmarks/.venv-baseline --invert-paths
```

There are only 7 commits, so the rewrite is quick and low-risk. **It rewrites every commit
hash**, so coordinate if anyone else has a clone. Recommended: do it, and re-verify with
`git log --all --numstat | grep -c libhighs` returning 0.

**Also worth checking** before submission: `benchmarks/.venv-baseline/` contains SciPy and
possibly CBC wheels too. Same treatment; the venv should be reproducible from a
`requirements.txt`, which is the right artifact to commit instead.

---

## P1 — `.gitignore` is incomplete; ~188 MB of generated data is untracked but unignored

Untracked and currently showing up in every `git status`:

| path | size | what it is |
|---|---:|---|
| `benchmarks/industry/` | 59 M | generated benchmark instances |
| `web/.venv/` | 59 M | Python venv for the web console |
| `benchmarks/industrial-ladder/` | 54 M | generated ladder instances (seed 42, regenerable) |
| `build-native/` | 16 M | second build tree (`SOR_NATIVE_ARCH=ON`) |
| `demo_out/` | 108 K | `.sol` / `.log` files written by the demo script |
| `scripts/__pycache__/`, `web/__pycache__/`, `python/__pycache__/` | 164 K | bytecode |
| `tools/julia_gpu/bench_julia_gpu_overhead` | 60 K | compiled benchmark binary |
| `tools/julia_gpu/bench_overhead_results.jsonl` | 4 K | generated results |
| `Highs.log` | 8 K | **HiGHS writes this into the repo root on every `highspy` run** |

`Highs.log` deserves a specific mention: it regenerates itself constantly and is a
competitor's log file sitting at the top level of your repo. It should be ignored *and*
ideally suppressed at the source (`h.setOptionValue("log_file", "")` in the baseline scripts).

**Proposed replacement `.gitignore`** — see §5 below. It is additive; nothing currently
tracked stops being tracked except via the explicit P0 step.

---

## P2 — The `benchmarks/results/` allowlist is stale

The ignore-with-exceptions pattern works correctly (300 files on disk, 11 tracked), but the
allowlist pins runs from `20260904-092218` / `-152608` while the current evidence is from
2026-09-05. Published claims should point at files that are actually in the repo.

**Action:** decide which runs are the citable evidence, force-add exactly those, and drop the
stale exceptions. Candidates for the current state:

- `compare-netlib-20260905-140353.{md,jsonl}` — latest Netlib vs HiGHS
- `miplib-easy-20260905-141554.{md,jsonl}` — latest MIPLIB-easy status set

Keep the count small and deliberate: these are evidence, not logs.

---

## P3 — Documentation sprawl: 6 overlapping performance documents

`docs/` holds 20 files (~300 KB). Six of them (~113 KB) are overlapping performance analysis
written over two days, and they contradict each other by design — later ones supersede earlier
ones:

| file | size | status |
|---|---:|---|
| `PERFORMANCE_PLAN_20260904.md` | 22 K | **superseded** (its ordering is explicitly wrong; says so at the top) |
| `PERFORMANCE_PLAN_20260905.md` | 34 K | superseded by the report + findings; contains two appendices of self-correction |
| `PERFORMANCE_REPORT_20260905.md` | 15 K | measurements — still current |
| `ALGORITHM_FRONTIER_20260905.md` | 15 K | literature review — still current |
| `FINDINGS_20260905.md` | 18 K | consolidation of all of the above |
| `MIP_PERFORMANCE_RESEARCH_20260904.md` | 10 K | MIP track — separate topic, keep |

A reviewer opening `docs/` cannot tell which of these is authoritative.

**Recommended:** keep three living documents and archive the rest.

```
docs/performance_audit.md        <- the running record of what shipped (already serves this)
docs/FINDINGS_20260905.md        <- current state + open levers (rename: PERFORMANCE.md)
docs/ALGORITHM_FRONTIER.md       <- literature + what is ruled out (drop the date)
docs/archive/                    <- PERFORMANCE_PLAN_2026090{4,5}.md, PERFORMANCE_REPORT_20260905.md
```

Dated filenames are the root cause: they invite accumulation. Prefer stable names with dated
*sections* inside, which is what `performance_audit.md` and `reference_log.md` already do well.

---

## P4 — Commit hygiene: one large mixed changeset is staged

`git status` shows ~29 paths staged as a single blob, mixing at least four unrelated efforts:
the seeded-solve performance work (`lu.*`, `dual_simplex.cpp`), a demo/REPL surface
(`python/`, `scripts/demo.sh`, `scripts/sor_repl.py`), web console changes (`web/*`), and
documentation.

**Recommended split:**

1. `perf: seeded FTRAN/BTRAN + per-slot bound cache` — `sor_la_cpu/*`, `sor_engines/*`,
   `docs/performance_audit.md`, `docs/reference_log.md`
2. `fix: remove duplicate declarations that broke the build` — the `lu.hpp` dedup (worth its
   own commit; see P5)
3. `demo: Python API, REPL, and demo script` — `python/`, `scripts/demo.sh`,
   `scripts/sor_repl.py`, `docs/SIH26119_DEMO_SCRIPT.md`
4. `web: …` — `web/*`
5. `docs: performance findings and frontier review` — the remaining docs
6. `chore: untrack baseline venv` — P0 stage 1

---

## P5 — Build-gate defect (process, not files)

Recorded here because it is the reason the P0 problem and a build breakage both went unnoticed:

**`ctest` passes over a failed build.** When compilation fails, CMake leaves the previous test
binaries on disk and CTest happily runs them. During this session the tree did not compile for
roughly an hour while `ctest` reported 22/22.

**Fix:** make the gate fail on build failure.

```bash
cmake --build build -j"$(nproc)" || exit 1
ctest --test-dir build --output-on-failure
```

Also worth adding: a CI/pre-commit check that `git ls-files | grep -qiE 'libhighs|site-packages'`
returns nothing — that single line would have caught P0 at the initial commit.

---

## 5. Proposed `.gitignore`

```gitignore
# ---- build trees -----------------------------------------------------------
build/
build-*/
*.o
*.a
*.so
CMakeCache.txt
CMakeFiles/

# ---- Python ----------------------------------------------------------------
__pycache__/
*.py[cod]
.venv/
venv/
# External-solver baselines (HiGHS/CBC/SciPy). NEVER commit: see
# docs/clean_room_policy.md. Reproduce with:
#   python3 -m venv benchmarks/.venv-baseline
#   benchmarks/.venv-baseline/bin/pip install -r benchmarks/requirements-baseline.txt
benchmarks/.venv-baseline/
web/.venv/

# ---- competitor log files written into the tree ----------------------------
# highspy drops this in the CWD on every run.
Highs.log

# ---- generated demo / run outputs ------------------------------------------
demo_out/
*.sol
nohup.out

# ---- generated benchmark instances -----------------------------------------
# MIPLIB 2017 collection (~7.3 GB). Re-download:
#   curl -fL -o /tmp/collection.zip https://miplib.zib.de/downloads/collection.zip
#   unzip -o -d benchmarks/miplib2017/mps /tmp/collection.zip
benchmarks/miplib2017/mps/
benchmarks/miplib2017/*.zip
# Synthetic scaling-study LPs (up to ~150 MB each): python3 scripts/gen_lp_ladder.py
benchmarks/synthetic/mps/
# Industrial ladder + industry sets (generated, seed 42): scripts/gen_*.py
benchmarks/industrial-ladder/
benchmarks/industry/

# ---- generated tool binaries / results -------------------------------------
tools/julia_gpu/bench_julia_gpu_overhead
tools/julia_gpu/*.jsonl

# ---- benchmark results (ignore all; force-add the citable ones) ------------
# git add -f benchmarks/results/<run>.md benchmarks/results/<run>.jsonl
benchmarks/results/*
!benchmarks/results/.gitkeep
```

Note `*.sol` is deliberately broad — solution files are always reproducible output. If any
fixture solution needs to be committed, force-add it.

---

## Execution order

| # | item | risk | reversible |
|---|---|---|---|
| 1 | P0 stage 1 — `git rm -r --cached benchmarks/.venv-baseline` | none | yes |
| 2 | P1 — replace `.gitignore` | none | yes |
| 3 | P4 — split the staged changeset into logical commits | none | yes |
| 4 | P2 — refresh the results allowlist | none | yes |
| 5 | P3 — consolidate docs into `docs/archive/` | none | yes |
| 6 | P5 — fix the build gate + add the `git ls-files` guard | none | yes |
| 7 | **P0 stage 2 — `git filter-repo` history purge** | **rewrites all hashes** | **no** |

Items 1–6 are safe and can be done now. Item 7 is the one that needs an explicit decision,
because it rewrites history — but for this project's central claim it is the one that
actually closes the issue.

Add `benchmarks/requirements-baseline.txt` (pinning `highspy`, `scipy`, `numpy`) as part of
item 1, so untracking the venv does not lose the ability to reproduce it.
