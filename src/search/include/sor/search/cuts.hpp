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
// Scope: root cutting loop plus optional tree/local separation (see
// sor/search/tree_cuts.hpp). Local cuts are tagged and expire with their
// subtree; they must never leak across siblings. Classical policy keeps
// root-only separation as an ablation.
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct CutOptions {
    int max_rounds = 20;
    f64 min_progress_rel = 1e-4;   // stop the round loop if bound gain falls below this
    // Reject a cut whose max|coef|/min|coef| exceeds this. 1e6 (the old
    // default) is far too permissive: on rgn.mps (MIPLIB-easy) a GMI cut
    // with dynamism in [1e3, 1e6) made the ROOT node's post-cut LP return
    // proposed_status=Optimal with a NaN-contaminated solution vector and a
    // dual residual of ~28 -- relaxation_proved() correctly refused to trust
    // it (no wrong answer reached the B&C proof), but the node then gave up
    // immediately ("node LP unproved") instead of continuing, so the whole
    // instance silently went from solvable (4013 nodes, Optimal) to an
    // instant failure. Verified empirically: 1e2 fixes rgn.mps; 1e3 and
    // above all reproduce the failure identically.
    f64 dynamism_max = 1e2;
    f64 violation_min = 1e-4;      // reject a cut that doesn't cut off the current point by this much
    f64 frac_min = 1e-4;           // skip tableau rows whose fractional part is too close to 0/1
    int max_cuts_per_round = 200;
    int max_candidates_per_round = 500;
    std::size_t pool_max_size = 5000;
    int pool_max_age = 5;
    f64 pool_parallelism_max = 0.995;
    f64 pool_duplicate_tol = 1e-9;
    f64 pool_efficacy_min = 1e-6;
    // Composite selection score (Achterberg, Constraint Integer Programming
    // 2007, and SCIP's default scoring; the base the context-aware scheme of
    // Turner et al. 2023 builds on):
    //
    //   score = efficacy + w_obj * objective-parallelism + w_int * integer-support
    //
    // Efficacy alone -- the only thing this pool used to rank on -- says how far
    // a cut moves the current point, not whether that movement is in a
    // direction that can improve the BOUND. A cut orthogonal to the objective
    // can be maximally violated and still leave the relaxation value untouched.
    // That was tolerable when Gomory was the only family; with covers and MIR
    // also competing it stopped being so: adding MIR took gt2's post-cut root
    // bound DOWN from 20725 to 20382, purely because higher-violation cuts
    // displaced more useful ones.
    // All four score weights default to ZERO, which makes the composite
    // collapse to the efficacy ranking this pool has always used. That is a
    // measurement, not a lack of nerve: with the extra terms on and the new cut
    // families off, the wider set's median gap still moved from 7.11% to 9.41%.
    // The terms are principled and cheap, so they stay available, but they do
    // not go in as defaults on evidence that says they cost more than they buy
    // here.
    f64 pool_weight_objective_parallelism = 0.0;
    f64 pool_weight_integer_support = 0.0;
    // --- Context-aware cut selection (Turner, Berthold & Besancon,
    // arXiv:2307.07322, 2023; 5% solve time and 8% nodes over SCIP default on
    // MIPLIB 2017). Three of the paper's findings are used here, and they are
    // the three its parameter tuning reached a consensus on rather than the
    // ones that varied between runs:
    //
    //   * dense cuts should be FILTERED at a 40-50% density threshold;
    //   * parallel cuts should NOT be filtered but LIGHTLY PENALISED, with a
    //     cut dropped only once its penalised score falls below zero;
    //   * the per-round limit should be on NONZEROS ADDED, as a multiple of the
    //     number of columns, rather than on the count of cuts.
    //
    // Plus the paper's sparsity score, which rewards sparse cuts on a linear
    // ramp that reaches zero at `endsps` density (dense cuts slow every
    // subsequent node LP), and the complement of its lock score, which promotes
    // cuts on variables that few other rows already constrain.
    // MEASURED OFF by default (a value > 1 can never trigger). The density
    // filter and the nonzero budget are the two Turner components that cost
    // proofs here: on miplib-easy the full configuration proves 10 of 20
    // against 12 with them disabled, and raising the absolute floor from 30 to
    // 200 nonzeros recovered the time but not the proofs. Their evidence on
    // MIPLIB 2017 stands; this benchmark suite is the regime the paper's
    // training set excluded, so they are kept as options with the measurement
    // rather than shipped as defaults. See MILP_CUTS_20260907.md 8.
    f64 pool_max_density = 2.0;          // reject a cut denser than this
    // ...but never below this many nonzeros, and the floor has to be GENEROUS.
    // Turner et al. tuned the 40% threshold on a training set that explicitly
    // excluded instances SCIP solves in under five seconds or fifty nodes, and
    // the paper says it does not expect the result to generalise outside that.
    // Measured here on miplib-easy, whose models run 30-300 columns: at a floor
    // of 30 the filter cost TWO PROOFS and 66 s (9 proved / 421 s against
    // 11 / 355 s with the filter off), because a useful Gomory cut on a
    // 160-column model routinely spans 99 of them -- 62% density, which is not
    // a large cut in any absolute sense. At 200 the filter engages only on
    // models where 40% is genuinely a lot of nonzeros.
    std::size_t pool_min_dense_nnz = 200;
    // Hard filtering is the DEFAULT here, against the paper's recommendation,
    // on measurement: swapping it for the penalty cost a proof on miplib-easy.
    // Set false to use the penalty scheme instead.
    bool pool_parallel_hard_filter = true;
    f64 pool_parallelism_penalty = 0.5;  // subtracted, scaled by the cosine
    f64 pool_parallelism_penalty_min = 0.3;  // cosine below this is ignored
    f64 pool_nnz_budget_factor = 0.0;    // nonzeros per round <= this * n_cols (0 = off)
    std::size_t pool_min_nnz_budget = 200;   // ...with the same kind of floor
    f64 pool_weight_sparsity = 0.0;
    f64 pool_sparsity_end_density = 0.4;
    f64 pool_weight_low_locks = 0.0;
    // Reject a cut whose direction is this close to a row the model already
    // has. DISABLED BY DEFAULT (>1 can never trigger, since |cos| <= 1), and
    // the default is a measured result rather than caution.
    //
    // The idea was that a cut nearly parallel to an existing row is nearly
    // linearly dependent on it and so buys conditioning trouble for no new
    // direction. That reasoning is wrong for Gomory cuts specifically: a GMI
    // cut is DERIVED from a tableau row, which is itself a combination of model
    // rows, so being highly parallel to the model is its normal condition, not
    // a sign of redundancy. Filtering at 0.999 dropped 27 of rgn's 84 cuts and
    // took its root bound from 64.96 to 58.35, weakened lseu (977.6 -> 966.9)
    // and n5-3 (5054 -> 4860), and cost gt2 its proof outright -- 12 of 20
    // proved on miplib-easy became 11. It also did not fix the problem it was
    // written for (see MILP_PRIMAL_20260907.md 3c).
    f64 parallel_to_row_max = 2.0;
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
    // Under the penalty scheme these mean different things: `penalized` counts
    // cuts whose score was reduced for being parallel to a selected cut, and
    // `rejected` counts the subset that this pushed below zero and out of
    // consideration entirely.
    std::uint64_t pool_penalized_parallel = 0;
    std::uint64_t pool_rejected_parallel = 0;
    std::uint64_t pool_rejected_dense = 0;
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
    // Supplies what the composite score needs: the objective direction cuts are
    // measured against, and which columns are integral. Without it the pool
    // falls back to ranking on efficacy alone.
    void set_scoring_context(const model::LpProblem& lp);
    // Optional external score for sequence selection (HGTSM under Latest).
    // When set, select_violated ranks by scorer(cut, x) instead of the efficacy
    // composite. Violation / density / parallelism filters are unchanged.
    // Pass an empty function to revert to efficacy scoring.
    using ExternalScoreFn =
        std::function<f64(const CutRow& cut, const std::vector<f64>& x)>;
    void set_external_scorer(ExternalScoreFn fn);
    // Batch / graph scorer (HGTSM paper path). When set, takes precedence over
    // the per-cut ExternalScoreFn. Receives eligible cuts in pool order and
    // must write one score per cut.
    using ExternalBatchScoreFn = std::function<void(
        const std::vector<CutRow>& cuts, const std::vector<f64>& x,
        std::vector<f64>& scores_out)>;
    void set_external_batch_scorer(ExternalBatchScoreFn fn);
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
        f64 last_score = 0.0;
        int age = 0;
        bool active = false;
    };

    CutOptions opts_;
    std::vector<Entry> entries_;
    std::vector<f64> obj_unit_;      // c / ||c||, empty when unavailable
    std::vector<bool> is_integer_;
    std::vector<f64> locks_;         // down-locks + up-locks per column
    f64 max_locks_ = 0.0;
    Index n_cols_ = 0;
    const model::LpProblem* lp_ctx_ = nullptr;
    ExternalScoreFn external_score_;
    ExternalBatchScoreFn external_batch_score_;
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
                            const std::vector<CutRow>& cuts,
                            const CutOptions& opts);

}  // namespace sor::search
