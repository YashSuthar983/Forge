// SOR - root-node cutting-plane separation and cut-pool management.
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
#include "sor/core/route_debug.hpp"

namespace sor::search {

using core::f64;
using core::Index;

struct CutOptions {
    // Treat a row-activity variable as integral only if every nonzero term
    // is an exactly integral coefficient of a declared integer column.
    // Experimental until validity and full-set proof ablations pass.
    bool integer_slack_gmi = true;
    bool integer_activity_basic_gmi = false;
    int max_rounds = 20;
    f64 min_progress_rel = 1e-4;   // a round gaining less than this is "stalled"
    // Consecutive stalled rounds tolerated before the loop gives up.
    //
    // Stopping on the FIRST stalled round is myopic, and measurably so. On
    // gt2 the default (GMI only) runs 13 rounds for 66 cuts and a root bound
    // of 20991; enabling MIR made one round gain a lot and the next gain
    // little, so the loop broke at round 4 with 28 GMI cuts and a WORSE bound
    // of 20089. Adding valid cuts cannot lower an LP bound -- the loss came
    // entirely from quitting nine rounds early.
    //
    // Cut loops are not monotone in per-round gain: a round that adds little
    // often exposes structure the next round exploits. Patience is what lets
    // the loop cross that dip.
    int min_progress_patience = 2;   // plan 3E: stop after two low-yield rounds
    // Retract the trailing run of rounds that bought no bound, instead of
    // only stopping once it is long enough.
    //
    // min_progress_rel / min_progress_patience above are a STOPPING rule: they
    // decide when to stop adding rounds. They never remove a round that was
    // already appended and gained nothing, so on blend2 thirty MIR cuts rode
    // into every one of 9,382 node LPs for a root bound identical to the one
    // the loop had without them (6.9214204824 -> 7.0439990704 either way).
    //
    // What this adds is the retraction the gate lacked, and only for rounds
    // the loop's own measure says were free: when the loop exits, the rounds
    // added SINCE THE LAST REALISED BOUND IMPROVEMENT are removed. The bound
    // is by construction unchanged by removing them -- it is the bound those
    // rounds failed to move -- so patience keeps its job (the loop still
    // crosses a dip to reach a later gain, and a dip round that leads to a
    // gain is kept), and only cuts that provably bought nothing are dropped.
    bool rollback_stalled_rounds = false;
    // Drop root cuts that carry no dual price at the final root LP optimum.
    //
    // Round-level rollback can only remove a round that bought no bound. It
    // says nothing about a round that DID buy bound with two of its thirty
    // cuts, which is the shape the per-round trace actually shows: on blend2
    // round 0 adds 28 MIR cuts and gains 1.55%, and the run without MIR
    // reaches the identical bound -- so the cuts are redundant, not from a
    // stalled round, and no round-level rule can see that.
    //
    // A row whose optimal multiplier is zero is not holding the bound up:
    // (x*, y with that entry deleted) is primal-feasible, dual-feasible and
    // complementary for the model without the row, so the LP value is
    // unchanged by construction. Requiring slack as well is belt-and-braces
    // against a degenerate zero multiplier at a tight row.
    //
    // Cannot produce a wrong answer in either direction: this only REMOVES
    // rows, so the result is a relaxation of a relaxation and still contains
    // every integer-feasible point. The risk it carries is a weaker bound in
    // the tree (a cut slack at the root can be tight deeper down), which is a
    // performance question, not a soundness one.
    // Carry the cut loop's basis from one round to the next.
    //
    // The loop has a warm-continuation path (bab.cpp: extend the previous
    // round's basis by one logical per new row and re-solve with the dual).
    // Before this flag existed it could not run: the guard it tests,
    // have_cut_loop_basis, was initialised false and assigned nowhere, so
    // every cut round paid a COLD solve_simplex. The same guard also gates
    // seeding the root node with the loop's final basis, so the root started
    // cold too. Measured cost of that: p0201's cut loop 241 ms -> 77 ms.
    //
    // OFF BY DEFAULT, and not part of any recommended combination until it
    // has had its own 3-rep sweep. Two reasons beyond the usual. It changes
    // pivot sequences everywhere it fires, so it must be measured alone; and
    // the basis-EXTENSION block it enables has by construction never executed
    // in production, so switching this on runs untested code for the first
    // time.
    bool warm_start_rounds = false;
    // Per-separator MARGINAL contribution gate.
    //
    // The realised-gain gate and the rollback built on it both measure a
    // ROUND. A round is the wrong unit when several families share it: on
    // blend2 round 0 gains 1.55% and the run with MIR disabled reaches the
    // identical bound to eleven significant digits, so MIR's marginal
    // contribution is exactly zero while its round's gain is not. No
    // round-level rule can see that, and neither can purging -- purging pays
    // the separation, pays the rows for every round before it fires, and then
    // pays the purge. A marginal test pays one probe and then stops
    // generating the cuts at all.
    //
    // At `marginal_gate_round`, re-solve the root once with all selected cuts
    // and once per optional family with that family's cuts held out. A family
    // whose held-out bound matches the full bound is contributing nothing;
    // drop its cuts from that round and stop running it for the rest of the
    // loop. GMI is never gated: it is the base separator the others are
    // measured against.
    //
    // Costs one LP re-solve per optional family present, all at the root.
    // marginal_gate_probe_ms / _probe_solves report that so it can be
    // weighed against the rows it removes.
    // MEASURED UNSAFE 2026-09-22 -- DO NOT ENABLE. Keeping the code because
    // it is the evidence for a latent defect elsewhere, not because it works.
    //
    // On blend2 the gate correctly prices MIR's marginal contribution at
    // exactly 0 and drops 31 of 32 root cuts. That leaves the root model
    // nearly cut-free, and the tree separator then derives MIR cuts at nodes
    // still sitting at root bounds. Those cuts PASS both promotion gates
    // (node_at_root_bounds, used_local_bound) and are stamped globally valid
    // while excluding blend2's true optimum: 23 of them with GCS on, 72 with
    // --no-gcs. Controls: 0 for default, cover, mir, cover+mir, cover+mir at
    // 1 and 2 rounds, and 0 across the other 13 reference models. The gate is
    // the trigger.
    //
    // No wrong answer was observed (the run stayed Feasible at 7.69 against a
    // true optimum of 7.598985), but a globally-promoted cut that excludes the
    // optimum is the precursor to a false Optimal, which this codebase has
    // shipped before. The underlying hole is documented in bab.cpp's own node
    // promotion comment: node_lp is global_lp PLUS the node's local cut rows,
    // so a cut derived from a local row touches no tightened bound and the
    // bound-provenance test cannot see it. Fixing that needs source-row
    // provenance, which no separator reports yet.
    bool marginal_gate = false;
    int marginal_gate_round = 0;
    f64 marginal_gate_min_rel = 1e-6;
    bool purge_nonbinding_cuts = false;
    f64 purge_dual_tol = 1e-9;
    f64 purge_slack_tol = 1e-7;
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
    std::uint64_t integral_activity_rows = 0;
    std::uint64_t integer_activity_candidates = 0;
    std::uint64_t integer_activity_terms = 0;
    std::uint64_t fractional_integer_bound_terms = 0;
    std::uint64_t gmi_cuts_added = 0;
    std::uint64_t gmi_missing_basis = 0;
    std::uint64_t gmi_invalid_factor = 0;
    std::uint64_t gmi_empty_rows = 0;
    std::uint64_t rejected_dynamism = 0;
    std::uint64_t rejected_violation = 0;
    std::uint64_t rejected_free_nonbasic = 0;
    // A5: a term whose GMI coefficient rounded to ~0 but whose bound was
    // nonzero (finite or infinite) -- the coeff*bound contribution that
    // dropping it silently omits from the RHS is not itself negligible in
    // that case, so the candidate cut is refused instead of risking an
    // unsound (too-tight) inequality.
    std::uint64_t rejected_dropped_term_bound = 0;
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

    // Did this cut's derivation rely on a bound that branching tightened?
    //
    // Separators like MIR and lifted covers substitute variables onto a BOUND
    // (x = bound +- s). At a node those bounds are branch-tightened, so the
    // resulting cut holds only inside that subtree and must not enter the
    // global pool. Global validity was previously inferred from the cut's
    // NAME, which produced false Optimals and a false Infeasible.
    //
    // Checking the cut's SUPPORT against root bounds is NOT sufficient: a
    // separator substitutes out variables that branching has FIXED, so those
    // columns never appear in the support while still having shaped the rhs.
    // Only the separator itself knows which bounds it consumed, so it reports
    // that here.
    //
    // DEFAULT true = "assume local", so a separator that has not been taught
    // to track provenance stays conservative and its node cuts stay local.
    bool used_local_bound = true;
};

// A bounded global pool for generated, not-yet-active cuts. Cuts are stored in
// a scale-invariant <= representation so positive rescalings are recognized as
// duplicates. Selection ranks normalized violation (efficacy), rejects nearly
// parallel rows within a batch, and ages unused candidates. Once selected, a
// cut remains fingerprinted as active so a later separator round cannot append
// the same LP row again.
class CutPool {
public:
    explicit CutPool(const CutOptions& opts) : opts_(opts) { SOR_FN();}

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
    std::size_t size() const { SOR_FN(); return entries_.size(); }
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

// Everything one apply_cuts_inplace() call changed about an LpProblem, in the
// form needed to put it back. Recorded so a root cut ROUND can be retracted
// when the re-solve shows it bought no bound.
//
// Two kinds of change have to be captured, not one. A round APPENDS rows, and
// it also TIGHTENS pre-existing rows: a cut whose linear form the model
// already has is folded into that row's bounds rather than added as a second,
// linearly dependent row. Undoing only the appends would silently keep the
// second kind.
struct CutUndo {
    Index rows_before = 0;          // lp.n_rows() before the round
    bool names_were_empty = false;  // lp.row_names was empty before the round
    // Pre-existing rows the round tightened, with the bounds they had before.
    // Recorded at a row's FIRST tightening in the round, so replaying these
    // restores the pre-round state even when several cuts hit the same row.
    std::vector<Index> tightened_rows;
    std::vector<f64> tightened_lo;
    std::vector<f64> tightened_hi;

    bool empty() const {
        SOR_FN();
        return tightened_rows.empty() && rows_added_ == 0;
    }
    Index rows_added() const { SOR_FN(); return rows_added_; }

    // Set by apply_cuts_inplace.
    Index rows_added_ = 0;
};

// Append accepted cuts as new rows (or tighten an existing row of the same
// shape). Mutates `lp` in place: CSR grows by push_back, so a single learned
// nogood costs O(row nnz) rather than a full-matrix rebuild.
void apply_cuts_inplace(model::LpProblem& lp,
                        const std::vector<CutRow>& cuts,
                        const CutOptions& opts,
                        CutUndo* undo = nullptr);

// Copy-then-inplace convenience for callers that hold a const LP.
model::LpProblem apply_cuts(const model::LpProblem& lp,
                            const std::vector<CutRow>& cuts,
                            const CutOptions& opts);

// Undo exactly what the apply_cuts_inplace() call that filled `undo` did to
// `lp`. `lp` must not have been structurally changed since (this is a strict
// LIFO: retract the most recent round first).
//
// Restoring the tightened bounds is not optional. A cut whose linear form the
// model already carries is folded into THAT row's bounds instead of being
// appended (see apply_cuts_inplace), so a round can strengthen the model
// without adding a single row; truncating the appended rows alone would leave
// that strengthening in place and the "retracted" round still active.
void retract_cuts_inplace(model::LpProblem& lp, const CutUndo& undo);

}  // namespace sor::search
