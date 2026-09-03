// SOR — root-node cutting-plane separation and cut-pool management
// (Branch-and-Cut items 1, 9, and 11 of docs/SIH26119_PS_ALIGNMENT.md §5).
//
// LAYER L5 (search), sibling of bab.hpp. Given a proved-optimal LP relaxation
// and its basis, separate_gomory_mi() derives valid inequalities from the
// simplex tableau (Wolsey; Achterberg thesis 2007 Ch. 8.2-8.3) that cut off
// the current fractional point without excluding any integer-feasible point
// of the original MILP. apply_cuts() appends accepted cuts as new rows,
// mirroring the existing add_binary_cover_cuts() CSR-rebuild pattern in
// bab.cpp.
//
// Scope: root-level cutting loop only (bab.cpp calls this before the B&B tree
// starts). The pool ranks, deduplicates, diversifies, and ages root candidates.
// Per-node/local cuts need basis-extension-on-row-add machinery that does not
// exist yet; see the plan note in bab.cpp's cutting loop.
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct CutOptions {
    int max_rounds = 20;
    f64 min_progress_rel = 1e-4;   // stop the round loop if bound gain falls below this
    f64 dynamism_max = 1e6;        // reject a cut whose max|coef|/min|coef| exceeds this
    f64 violation_min = 1e-4;      // reject a cut that doesn't cut off the current point by this much
    f64 frac_min = 1e-4;           // skip tableau rows whose fractional part is too close to 0/1
    int max_cuts_per_round = 200;
    int max_candidates_per_round = 500;
    std::size_t pool_max_size = 5000;
    int pool_max_age = 5;
    f64 pool_parallelism_max = 0.995;
    f64 pool_duplicate_tol = 1e-9;
    f64 pool_efficacy_min = 1e-6;
};

struct CutDiagnostics {
    int rounds = 0;
    std::uint64_t candidates_considered = 0;
    std::uint64_t gmi_cuts_added = 0;
    std::uint64_t rejected_dynamism = 0;
    std::uint64_t rejected_violation = 0;
    std::uint64_t rejected_free_nonbasic = 0;
    std::uint64_t pool_inserted = 0;
    std::uint64_t pool_duplicates = 0;
    std::uint64_t pool_dominated = 0;
    std::uint64_t pool_rejected_parallel = 0;
    std::uint64_t pool_aged_out = 0;
    std::uint64_t pool_evicted = 0;
    std::uint64_t pool_selected = 0;
    f64 root_bound_before = core::kNaN;
    f64 root_bound_after = core::kNaN;
};

// A single valid inequality in two-sided row form: row_lo <= sum(vals[k] * x[cols[k]]) <= row_hi.
struct CutRow {
    std::vector<Index> cols;
    std::vector<f64> vals;
    f64 row_lo = -model::kInf;
    f64 row_hi = model::kInf;
    std::string name;
};

// A bounded global pool for generated, not-yet-active cuts. Cuts are stored in
// a scale-invariant <= representation so positive rescalings are recognized as
// duplicates. Selection ranks normalized violation (efficacy), rejects nearly
// parallel rows within a batch, and ages unused candidates. Once selected, a
// cut remains fingerprinted as active so a later separator round cannot append
// the same LP row again.
class CutPool {
public:
    explicit CutPool(const CutOptions& opts) : opts_(opts) {}

    void start_round(CutDiagnostics& diag);
    void add(const std::vector<CutRow>& candidates, CutDiagnostics& diag);
    std::vector<CutRow> select_violated(const std::vector<f64>& x,
                                        CutDiagnostics& diag);
    std::size_t size() const { return entries_.size(); }
    std::size_t active_size() const;

private:
    struct Entry {
        CutRow cut;
        std::vector<f64> unit_vals;
        f64 rhs_unit = 0.0;
        f64 last_efficacy = 0.0;
        int age = 0;
        bool active = false;
    };

    CutOptions opts_;
    std::vector<Entry> entries_;
};

// One separation pass over a proved-optimal relaxation of `lp` at point `x`
// with basis `basis` (as returned by engines::solve_simplex/solve_dual_simplex
// with an out_basis pointer). Returns valid inequalities that cut off `x`,
// already filtered by `opts`'s numerical thresholds. Rows whose basic integer
// variable is within `opts.frac_min` of an integer, or that touch a free
// (AtZeroFree) nonbasic, are skipped rather than cut.
std::vector<CutRow> separate_gomory_mi(const model::LpProblem& lp,
                                       const std::vector<f64>& x,
                                       const engines::SimplexBasis& basis,
                                       const CutOptions& opts,
                                       CutDiagnostics& diag);

// Rebuild `lp` with `cuts` appended as new rows. Mirrors
// bab.cpp's add_binary_cover_cuts CSR-rebuild pattern.
model::LpProblem apply_cuts(const model::LpProblem& lp,
                            const std::vector<CutRow>& cuts);

}  // namespace sor::search
