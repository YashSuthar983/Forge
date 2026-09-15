// SOR — Kernel Pump: kernel-search decomposition of Feasibility Pump for
// binary MILPs (Assunção, Urrutia & Santos, MPC 2026,
// DOI:10.1007/s12532-026-00333-2). Clean-room from paper abstract / algorithm
// description; no SCIP/HiGHS/Gurobi/CBC or KP GitHub source was read.
//
// LAYER L5. Primal heuristic ONLY: produces candidate points. Never tightens
// dual bounds or writes proof certificates. Strict wall-clock and LP-iteration
// caps are mandatory.
//
// Algorithm sketch (paper §3):
//   1. Rank binary columns by LP fractionality (then reduced-cost sign when
//      available) into a kernel + ordered buckets.
//   2. Optionally refine the kernel so LP feasibility is attainable with
//      excluded binaries fixed to 0.
//   3. Run a bounded FP-style L1 projection on kernel ∪ first k buckets,
//      fixing remaining binaries to 0, until a feasible integer point appears
//      or the time budget is exhausted.
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct KernelPumpOptions {
    bool enabled = true;
    // Hard wall-clock for the whole KP call (mandatory).
    double time_limit_s = 2.0;
    // Fractionality ranges κ (paper default 10); buckets are up to 3(κ+1).
    int kappa = 10;
    // Max FP projection rounds across all buckets (shared counter).
    int max_pumps_total = 80;
    // Cap per bucket sub-FP (except the last, which may spend remaining time).
    int max_pumps_per_bucket = 25;
    std::uint64_t max_lp_iterations = 4000;
    f64 int_tol = 1e-6;
    f64 feas_tol = 1e-7;
    // When true, solve the paper's kernel-refinement LP (min sum of
    // out-of-kernel binaries subject to Ax in P). Skip if time is tiny.
    bool refine_kernel = true;
    // Soft objective pull toward improving the original objective after a
    // feasible point exists (caller usually passes no incumbent → off).
    f64 objective_cutoff = core::kPosInf;
};

struct KernelPumpDiagnostics {
    std::uint64_t buckets = 0;
    std::uint64_t kernel_size = 0;
    std::uint64_t pumps = 0;
    std::uint64_t lp_solves = 0;
    std::uint64_t lp_iterations = 0;
    bool found = false;
    double ms = 0.0;
};

// Partition of binary columns: kernel first, then ordered buckets. Empty when
// the model has no binaries (caller should fall back to plain FP / FeasJump).
struct KernelBuckets {
    std::vector<Index> kernel;
    std::vector<std::vector<Index>> buckets;
};

// Rank binaries by distance-to-integrality, then reduced-cost sign when
// `reduced_cost` is sized n. `x_lp` must be sized n. Returns false if no
// binaries exist.
bool build_kernel_buckets(const model::LpProblem& mip,
                          const std::vector<f64>& x_lp,
                          const std::vector<f64>* reduced_cost,
                          int kappa,
                          KernelBuckets& out);

// Apply the KP restriction: binaries outside `active` are fixed to 0 (clipped
// to their box). Continuous and general-integer columns keep their domains.
model::LpProblem apply_kernel_restriction(const model::LpProblem& mip,
                                          const std::vector<Index>& active);

// Bounded FP-style projection on a (possibly kernel-restricted) model.
// Returns true only when an integer-feasible point of `mip` is found.
bool feasibility_pump_restricted(const model::LpProblem& mip,
                                 const std::vector<f64>& x_start,
                                 const KernelPumpOptions& opts,
                                 double time_limit_s,
                                 int max_pumps,
                                 std::vector<f64>& x_out,
                                 KernelPumpDiagnostics& diag);

// Full Kernel Pump. Heuristic only — caller must re-validate against the
// original model before accepting an incumbent.
bool kernel_pump(const model::LpProblem& mip,
                 const std::vector<f64>& x_lp,
                 const std::vector<f64>* reduced_cost,
                 const KernelPumpOptions& opts,
                 std::vector<f64>& x_out,
                 KernelPumpDiagnostics& diag);

}  // namespace sor::search
