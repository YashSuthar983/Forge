// Primal simplex: known optima, statuses that must not be over-claimed, and
// the properties an optimal basis has to satisfy.
//
// The point of most of these cases is not "does it find the number" but "does it
// refuse to claim Optimal when it should not". A solver that is right on afiro
// and wrong about infeasibility is worse than useless.
#include "sor/certify/finalize.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/engines/farkas.hpp"
#include "sor/engines/lp.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/io/mps.hpp"
#include "sor/presolve/presolve.hpp"
#include "sor/sparse/csr.hpp"

#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "sor/core/route_debug.hpp"

using sor::core::f64;
using sor::core::Index;
using sor::core::ProofLevel;
using sor::core::Status;
using sor::engines::NonbasicStatus;
using sor::engines::SimplexBasis;
using sor::engines::SimplexDiagnostics;
using sor::engines::SimplexOptions;

namespace {

struct Run {
    sor::core::SolveResult r;
    SimplexDiagnostics diag;
    SimplexBasis basis;
    sor::model::LpProblem problem;
};

class ScopedEnvironment {
public:
    ScopedEnvironment(const char* name, const char* value) : name_(name) {
        SOR_FN();
        if (const char* old = std::getenv(name)) {
            had_old_ = true;
            old_ = old;
        }
        ::setenv(name, value, 1);
    }

    ~ScopedEnvironment() {
        SOR_FN();
        if (had_old_) ::setenv(name_.c_str(), old_.c_str(), 1);
        else ::unsetenv(name_.c_str());
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

private:
    std::string name_;
    std::string old_;
    bool had_old_ = false;
};

Run solve_text(const std::string& mps, SimplexOptions opts = {}) {
    SOR_FN();
    Run out;
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    out.problem = sor::io::read_mps(in, rep);
    auto raw = sor::engines::solve_simplex(out.problem, opts, out.diag, &out.basis);
    auto ev = sor::engines::simplex_evidence(out.diag, opts);
    ev = sor::certify::check_lp_result(out.problem, raw, ev);
    out.r = sor::certify::finalize_result(std::move(raw), ev);
    return out;
}

Run solve_problem(const sor::model::LpProblem& problem,
                  SimplexOptions opts = {}) {
    SOR_FN();
    Run out;
    out.problem = problem;
    auto raw = sor::engines::solve_simplex(
        out.problem, opts, out.diag, &out.basis);
    auto ev = sor::engines::simplex_evidence(out.diag, opts);
    ev = sor::certify::check_lp_result(out.problem, raw, ev);
    out.r = sor::certify::finalize_result(std::move(raw), ev);
    return out;
}

// A crossed bound is infeasible by the data alone, and no row-multiplier
// Farkas ray can show it when the column sits in no row or the row's own
// sides cross. The dual used to stop with "no primal-infeasible basic
// variable" and NoSolutionFound; finalize_result accepts the empty domain
// as the proof, re-derived from the model.
void test_empty_domain_is_infeasible() {
    SOR_FN();
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0}, {0}, {1.0});
    lp.c = {1.0, 1.0};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {10.0};
    lp.col_lo = {0.0, 3.0};
    lp.col_hi = {5.0, 2.0};  // column 1 is empty and in no row
    lp.col_names = {"x", "z"};
    for (const bool presolve : {true, false}) {
        SimplexOptions opts;
        opts.presolve = presolve;
        const auto run = solve_problem(lp, opts);
        CHECK(run.r.status == Status::Infeasible);
        CHECK(run.r.downgrade_reason.empty());
        CHECK(run.r.termination_reason ==
              "column 'z' has lower bound 3 above upper bound 2");
    }

    sor::model::LpProblem rows = lp;
    rows.col_lo = {0.0, 0.0};
    rows.col_hi = {5.0, 5.0};
    rows.row_lo = {4.0};
    rows.row_hi = {1.0};  // the row's sides cross
    const auto run = solve_problem(rows);
    CHECK(run.r.status == Status::Infeasible);
    const auto empty = rows.find_empty_domain();
    CHECK(empty.is_row && empty.index == 0);

    // Every strategy, including the first-order ones that cannot represent
    // an empty interval, reports it the same way.
    for (const auto strategy : {sor::core::LpStrategy::Auto,
                                sor::core::LpStrategy::Hpr,
                                sor::core::LpStrategy::Barrier}) {
        sor::core::LpOptions o;
        o.strategy = strategy;
        sor::core::LpDiagnostics d;
        sor::core::ProofEvidence ev;
        auto raw = sor::engines::solve_lp(lp, o, d, &ev);
        const auto r = sor::certify::finalize_result(
            sor::certify::check_lp_candidate(lp, std::move(raw), ev));
        CHECK(r.status == Status::Infeasible);
    }

    // The evidence comes from the model, never from the engine's word: the
    // same bare claim (no ray) is accepted for the crossed model and refused
    // once the crossing is gone.
    sor::core::RawResult claim;
    claim.proposed_status = Status::Infeasible;
    const auto accepted = sor::certify::finalize_result(
        claim, sor::certify::check_lp_result(lp, claim, sor::core::ProofEvidence{}));
    CHECK(accepted.status == Status::Infeasible);
    sor::model::LpProblem feasible = lp;
    feasible.col_lo[1] = 0.0;
    const auto refused = sor::certify::finalize_result(
        claim, sor::certify::check_lp_result(feasible, claim, sor::core::ProofEvidence{}));
    CHECK(refused.status == Status::NoSolutionFound);
}

// The time limit is a wall for the whole call: once presolve has used it,
// neither scaling nor any simplex stage runs.
void test_time_limit_covers_presolve_and_preparation() {
    SOR_FN();
    for (const bool presolve : {true, false}) {
        SimplexOptions opts;
        opts.presolve = presolve;
        opts.time_limit_s = 1e-9;
        const auto run = solve_text(sor::test::kTestLpMps, opts);
        CHECK(run.r.status == Status::Interrupted);
        CHECK(run.diag.iterations == 0);
        CHECK(run.r.termination_reason.find(
                  presolve ? "reached in presolve" : "reached before the solve") !=
              std::string::npos);
    }
}

// A maximization with an objective offset, through every place that turns
// the minimization form's numbers back into the model's sense. The copy used
// to keep the offset un-negated and each consumer compensated by hand.
void test_maximization_offset_is_reported_in_model_sense() {
    SOR_FN();
    // max 3x + 2y + 10 s.t. x + y <= 4, x <= 3; optimum x=3, y=1: 21.
    sor::model::LpProblem lp;
    lp.maximize = true;
    lp.obj_offset = 10.0;
    lp.A = sor::sparse::from_triplets(2, 2, {0, 0, 1}, {0, 1, 0}, {1.0, 1.0, 1.0});
    lp.c = {3.0, 2.0};
    lp.row_lo = {-sor::model::kInf, -sor::model::kInf};
    lp.row_hi = {4.0, 3.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {sor::model::kInf, sor::model::kInf};

    const auto pmin = sor::model::minimization_form(lp);
    CHECK(!pmin.maximize && pmin.obj_offset == -10.0 && pmin.c[0] == -3.0);
    CHECK(pmin.objective({3.0, 1.0}) == -lp.objective({3.0, 1.0}));

    for (const auto method : {sor::engines::SimplexMethod::Dual,
                              sor::engines::SimplexMethod::Primal}) {
        SimplexOptions opts;
        opts.method = method;
        opts.exact_proof = true;
        const auto run = solve_problem(lp, opts);
        CHECK(run.r.status == Status::Optimal);
        CHECK_NEAR(run.r.objective, 21.0, 1e-9);
        CHECK_NEAR(run.diag.primal_objective, 21.0, 1e-9);
        CHECK(std::isfinite(run.r.dual_bound));
        CHECK_NEAR(run.r.dual_bound, 21.0, 1e-9);
    }
    for (const auto strategy : {sor::core::LpStrategy::Hpr,
                                sor::core::LpStrategy::Barrier}) {
        sor::core::LpOptions o;
        o.strategy = strategy;
        sor::core::LpDiagnostics d;
        sor::core::ProofEvidence ev;
        auto raw = sor::engines::solve_lp(lp, o, d, &ev);
        const auto r = sor::certify::finalize_result(
            sor::certify::check_lp_candidate(lp, std::move(raw), ev));
        CHECK_NEAR(r.objective, 21.0, 1e-5);
    }
}

void test_fixture_lp() {
    SOR_FN();
    const auto run = solve_text(sor::test::kTestLpMps);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(run.r.downgrade_reason.empty());
    CHECK_NEAR(run.r.objective, sor::test::kTestLpOptimum, 1e-9);
    CHECK(run.r.x.size() == 2);
    CHECK_NEAR(run.r.x[0], sor::test::kTestLpX1, 1e-9);
    CHECK_NEAR(run.r.x[1], sor::test::kTestLpX2, 1e-9);
    CHECK(run.r.max_primal_violation < 1e-9);
    CHECK(run.r.max_dual_violation < 1e-9);

    // Strong duality is not assumed anywhere in the engine -- the dual value is
    // built from the reduced costs and compared. If the basis were wrong this
    // would be the check that fails.
    CHECK(run.diag.dual_bound_finite);
    CHECK_NEAR(run.diag.dual_objective, sor::test::kTestLpOptimum, 1e-9);
    CHECK(run.diag.gap_rel < 1e-9);
    CHECK(run.diag.preprocessing_builds == 1);
    CHECK(run.diag.stages >= 1);
    CHECK(run.diag.primal_stages + run.diag.dual_stages == run.diag.stages);
    CHECK(run.diag.cold_stages + run.diag.basis_restarts == run.diag.stages);
}

void test_auto_commits_to_one_engine_without_a_discarded_probe() {
    SOR_FN();
    // Auto must reach a proved result in ONE stage, on ONE shared
    // preprocessing build. Two regressions are fenced off here: the old
    // dispatcher that ran and then discarded a 256-iteration probe of the
    // other engine, and the per-stage preprocessing that rebuilt scaling and
    // CSC for every candidate.
    //
    // This LP is deliberately wide and sparse -- the shape the retired
    // primal-first classifier keyed on. The route no longer depends on shape,
    // so the assertion is on the dual engine and, more importantly, on the
    // stage counts, which is what the test was ever really about.
    sor::model::LpProblem lp;
    lp.name = "WIDE_PRIMAL_FIRST";
    lp.A = sor::sparse::from_triplets(1, 8, {0}, {0}, {1.0});
    lp.c = {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {1.0};
    lp.col_lo.assign(8, 0.0);
    lp.col_hi.assign(8, 1.0);
    lp.is_integer.assign(8, false);

    SimplexOptions opts;
    opts.presolve = false;  // preserve the deliberately wide shape
    SimplexDiagnostics diag;
    const auto raw = sor::engines::solve_simplex(lp, opts, diag);
    CHECK(raw.proposed_status == Status::Optimal);
    CHECK_NEAR(raw.objective, -1.0, 1e-9);
    CHECK(diag.stages == 1);
    CHECK(diag.primal_stages == 0);
    CHECK(diag.dual_stages == 1);
    CHECK(diag.cold_stages == 1);
    CHECK(diag.basis_restarts == 0);
    CHECK(diag.preprocessing_builds == 1);
    // Auto reports the structural summary it would route on, so the LP Auto
    // layer above can read it without a second pass over the model.
    CHECK(diag.route_features_valid);
    CHECK(diag.route_features.rows == 1);
    CHECK(diag.route_features.cols == 8);
}

void test_route_features_describe_the_model() {
    SOR_FN();
    using sor::engines::detail::route_features;

    // Two rows, four columns, and one nonzero per position we care about:
    //
    //   min  x0 - x2          [ 1  2  0  0 ] x  = 3        (equality)
    //        subject to       [ 0  0  4 100] x <= 5        (upper only)
    //
    //   x0 in [0,1] boxed, x1 in [2,2] fixed, x2 free, x3 in [0,inf).
    sor::model::LpProblem lp;
    lp.name = "FEATURES";
    lp.A = sor::sparse::from_triplets(2, 4, {0, 0, 1, 1}, {0, 1, 2, 3},
                                      {1.0, 2.0, 4.0, 100.0});
    lp.c = {1.0, 0.0, -1.0, 0.0};
    lp.row_lo = {3.0, -sor::model::kInf};
    lp.row_hi = {3.0, 5.0};
    lp.col_lo = {0.0, 2.0, -sor::model::kInf, 0.0};
    lp.col_hi = {1.0, 2.0, sor::model::kInf, sor::model::kInf};
    lp.is_integer.assign(4, false);

    const auto f = route_features(lp, 1e-7);
    CHECK(f.rows == 2);
    CHECK(f.cols == 4);
    CHECK(f.nnz == 4);
    CHECK_NEAR(f.density, 0.5, 1e-12);
    CHECK_NEAR(f.aspect, 2.0, 1e-12);
    CHECK_NEAR(f.row_degree, 2.0, 1e-12);
    CHECK_NEAR(f.col_degree, 1.0, 1e-12);

    // x2 alone is free, and it carries a cost.
    CHECK(f.free_cols == 1);
    CHECK(f.objective_free_cols == 1);
    CHECK_NEAR(f.free_fraction, 0.25, 1e-12);
    // x0 and x1 have both bounds finite; x1 alone is fixed.
    CHECK_NEAR(f.boxed_fraction, 0.5, 1e-12);
    CHECK_NEAR(f.fixed_fraction, 0.25, 1e-12);
    // x0 and x2 carry a cost.
    CHECK_NEAR(f.objective_fraction, 0.5, 1e-12);
    // Row 0 is an equality; row 1 has only an upper bound, so it is neither
    // ranged nor free.
    CHECK_NEAR(f.equality_fraction, 0.5, 1e-12);
    CHECK_NEAR(f.ranged_fraction, 0.0, 1e-12);
    CHECK_NEAR(f.free_row_fraction, 0.0, 1e-12);
    // Every column has degree 1.
    CHECK_NEAR(f.singleton_fraction, 1.0, 1e-12);
    // log10(100 / 1).
    CHECK_NEAR(f.coefficient_spread, 2.0, 1e-12);

    // The cold parking point is x = (0, 2, ., 0) -- x2 is free and parks at 0.
    // Row 0 reads 1*0 + 2*2 = 4 != 3, so the logical point is infeasible.
    CHECK(!f.logical_point_feasible);

    // Relaxing that equality to cover the parking activity flips the flag, and
    // nothing else about the model changes.
    lp.row_lo[0] = 4.0;
    lp.row_hi[0] = 4.0;
    const auto g = route_features(lp, 1e-7);
    CHECK(g.logical_point_feasible);
    CHECK_NEAR(g.equality_fraction, 0.5, 1e-12);

    // An empty model must not divide by zero.
    sor::model::LpProblem empty;
    empty.A = sor::sparse::from_triplets(0, 0, {}, {}, {});
    const auto z = route_features(empty, 1e-7);
    CHECK(z.rows == 0);
    CHECK(z.cols == 0);
    CHECK_NEAR(z.density, 0.0, 1e-12);
}

void test_dual_periodic_resync_is_not_tied_to_verbose() {
    SOR_FN();
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Dual;
    opts.presolve = false;
    opts.dual_resync_interval = 1;
    opts.verbose = false;
    const auto run = solve_text(sor::test::kTestLpMps, opts);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.diag.iterations > 0);
    CHECK(run.diag.dual_resyncs > 0);
}

void test_pruned_basic_pivotal_entries_match_full_path() {
    SOR_FN();
    // Basic columns cannot enter the dual ratio test and their reduced costs
    // are not maintained. The optimized pivotal-row support omits them unless
    // a Devex reference framework needs them. Compare against the retained
    // legacy support to pin the complete pivot trajectory, not just the final
    // objective.
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Dual;
    opts.presolve = false;
    Run full;
    {
        ScopedEnvironment keep("SOR_DUAL_KEEP_BASIC_PIVOTAL", "1");
        full = solve_text(sor::test::kTestLpMps, opts);
    }
    Run accumulated;
    {
        ScopedEnvironment keep_values(
            "SOR_DUAL_ACCUMULATE_BASIC_PIVOTAL", "1");
        accumulated = solve_text(sor::test::kTestLpMps, opts);
    }
    Run filtered;
    {
        ScopedEnvironment filter("SOR_DUAL_FILTER_ACTIVE_PIVOTAL", "1");
        filtered = solve_text(sor::test::kTestLpMps, opts);
    }
    const auto pruned = solve_text(sor::test::kTestLpMps, opts);
    CHECK(full.r.status == Status::Optimal);
    CHECK(pruned.r.status == Status::Optimal);
    CHECK(full.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(pruned.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(full.diag.iterations > 0);
    CHECK(full.diag.dual_pivotal_entries_full > 0);
    CHECK(full.diag.dual_pivotal_entries_kept ==
          full.diag.dual_pivotal_entries_full);
    // The active-only path reads a nonbasic-partitioned copy of A, so it no
    // longer VISITS the entries it would discard -- touched and kept coincide
    // on it. The saving is therefore expressed against the retained path,
    // which is the stronger statement: strictly fewer entries touched, for the
    // same trajectory asserted below.
    CHECK(pruned.diag.dual_pivotal_entries_full <
          full.diag.dual_pivotal_entries_full);
    CHECK(pruned.diag.dual_pivotal_entries_kept <=
          pruned.diag.dual_pivotal_entries_full);
    CHECK(accumulated.diag.dual_pivotal_entries_kept ==
          pruned.diag.dual_pivotal_entries_kept);
    CHECK(accumulated.r.status == pruned.r.status);
    CHECK(accumulated.r.proof == pruned.r.proof);
    CHECK(accumulated.diag.iterations == pruned.diag.iterations);
    CHECK(accumulated.r.objective == pruned.r.objective);
    CHECK(accumulated.r.x == pruned.r.x);
    CHECK(accumulated.r.y == pruned.r.y);
    CHECK(accumulated.basis.basic == pruned.basis.basic);
    CHECK(accumulated.basis.status == pruned.basis.status);
    CHECK(filtered.r.status == pruned.r.status);
    CHECK(filtered.r.proof == pruned.r.proof);
    CHECK(filtered.diag.iterations == pruned.diag.iterations);
    CHECK(filtered.r.objective == pruned.r.objective);
    CHECK(filtered.r.x == pruned.r.x);
    CHECK(filtered.r.y == pruned.r.y);
    CHECK(filtered.basis.basic == pruned.basis.basic);
    CHECK(filtered.basis.status == pruned.basis.status);
    CHECK(pruned.diag.iterations == full.diag.iterations);
    CHECK(pruned.diag.phase1_iterations == full.diag.phase1_iterations);
    CHECK(pruned.diag.phase2_iterations == full.diag.phase2_iterations);
    CHECK(pruned.r.objective == full.r.objective);
    CHECK(pruned.r.x == full.r.x);
    CHECK(pruned.r.y == full.r.y);
    CHECK(pruned.basis.basic == full.basis.basic);
    CHECK(pruned.basis.status == full.basis.status);
}

void test_pruned_fixed_pivotal_entries_match_retained_path() {
    SOR_FN();
    // The first equality is deliberately the first leaving row. Once X0
    // replaces its fixed logical, the second BTRAN reaches both rows, so that
    // now-nonbasic fixed logical is present in the mathematical pivotal row.
    // It can never enter and its reduced cost has no sign condition.
    sor::model::LpProblem lp;
    lp.name = "DUAL_FIXED_PIVOTAL";
    lp.A = sor::sparse::from_triplets(
        2, 2, {0, 1, 1}, {0, 0, 1}, {1.0, 1.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {10.0, 2.0};
    lp.row_hi = lp.row_lo;
    lp.col_lo = {0.0, -sor::core::kPosInf};
    lp.col_hi = {sor::core::kPosInf, sor::core::kPosInf};

    SimplexOptions opts;
    // Exercise both pivotal-row sweeps; the crash would discharge both rows.
    opts.dual_crash = false;
    opts.method = sor::engines::SimplexMethod::Dual;
    opts.presolve = false;
    opts.ruiz_iterations = 0;
    Run retained;
    {
        ScopedEnvironment keep_fixed("SOR_DUAL_KEEP_FIXED_PIVOTAL", "1");
        retained = solve_problem(lp, opts);
    }
    const auto pruned = solve_problem(lp, opts);
    CHECK(retained.r.status == Status::Optimal);
    CHECK(pruned.r.status == Status::Optimal);
    CHECK(retained.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(pruned.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(retained.diag.iterations >= 2);
    CHECK(pruned.diag.iterations == retained.diag.iterations);
    CHECK(retained.diag.dual_pivotal_entries_kept >
          pruned.diag.dual_pivotal_entries_kept);
    CHECK(pruned.r.objective == retained.r.objective);
    CHECK(pruned.r.x == retained.r.x);
    CHECK(pruned.r.y == retained.r.y);
    CHECK(pruned.basis.basic == retained.basis.basic);
    CHECK(pruned.basis.status == retained.basis.status);
}

void test_dual_cost_perturbation_cleans_before_optimality() {
    SOR_FN();
    SimplexOptions exact_opts;
    exact_opts.method = sor::engines::SimplexMethod::Dual;
    exact_opts.presolve = false;
    const auto exact = solve_text(sor::test::kTestLpMps, exact_opts);

    SimplexOptions perturbed_opts = exact_opts;
    perturbed_opts.dual_cost_perturbation_multiplier = 1.0;
    const auto perturbed = solve_text(sor::test::kTestLpMps, perturbed_opts);

    CHECK(exact.r.status == Status::Optimal);
    CHECK(perturbed.r.status == Status::Optimal);
    CHECK(perturbed.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK_NEAR(perturbed.r.objective, exact.r.objective, 1e-9);
    CHECK(perturbed.diag.perturbed_costs > 0);
    CHECK(perturbed.diag.perturbation_cleanups == 1);
    CHECK(perturbed.diag.primal_residual <= 1e-9);
    CHECK(perturbed.diag.dual_residual <= 1e-9);
}

// The hand-off from the dual to the primal clean-up must not damage the point
// it hands over.
//
// When the true costs are restored at an optimal exit, boxed nonbasics can
// come out parked on the dual-infeasible side. Flipping them is free for the
// DUAL -- pi and every reduced cost are untouched -- but each flip moves a
// nonbasic across its whole range and drags xB with it, so on the
// primal-feasible basis of an optimal exit a flip pass manufactures primal
// infeasibility. If a non-flippable (one-sided or free) column is dual
// infeasible as well, the dual cannot continue and the primal engine takes
// over; handing it the FLIPPED basis makes it rebuild primal feasibility in
// phase 1 from a point the dual had already driven to optimality.
//
// This LP puts both kinds of column in that state at once, deterministically:
//
//   min  x0 + x1 + 100*xbig + 1.00001*x3
//   s.t. x0 + x1 + xbig + x3 = 5
//        x0 in [0, 1]   boxed     x1 in [0, inf)  lower-bounded
//        xbig fixed 0             x3 in [4.5, 5]  boxed, and basic below
//
// At the supplied basis (x3 basic, everything else at its lower bound) x3 = 5
// is primal feasible, so the dual exits at once. y = c3, which makes the true
// reduced cost of BOTH x0 and x1 equal to -1e-5: x0 is a boxed column parked
// on the wrong side, x1 is the non-flippable one that forces the hand-off.
// The deterministic perturbation (xbig only exists to set its scale) is large
// enough to hold both reduced costs positive while it is installed, so the
// dual never sees either infeasibility until the costs come back.
//
// Flipping x0 to its upper bound puts x3 at 4.0, half a unit below its lower
// bound. That is exactly the damage under test.
void test_dual_cleanup_hands_over_a_primal_feasible_basis() {
    SOR_FN();
    const std::string mps = R"(NAME          WSA2CLEAN
ROWS
 N  COST
 E  R1
COLUMNS
    X0        COST      1.0        R1        1.0
    X1        COST      1.0        R1        1.0
    XBIG      COST      100.0      R1        1.0
    X3        COST      1.00001    R1        1.0
RHS
    RHS       R1        5.0
BOUNDS
 UP BND       X0        1.0
 FX BND       XBIG      0.0
 LO BND       X3        4.5
 UP BND       X3        5.0
ENDATA
)";
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    auto problem = sor::io::read_mps(in, rep);

    for (const bool boxed_only : {false, true}) {
        problem.col_hi[1] = boxed_only ? 1.0 : sor::model::kInf;
        SimplexOptions opts;
        opts.method = sor::engines::SimplexMethod::Dual;
        opts.presolve = false;
        // Unscaled, so the reduced costs above are the ones the engine sees and
        // the per-column dual tolerance is exactly dual_feas_tol.
        opts.ruiz_iterations = 0;
        opts.dual_cost_perturbation_multiplier = 1.0;

        SimplexBasis start;
        start.n_struct = 4;
        start.basic = {3};
        start.status = {NonbasicStatus::AtLower, NonbasicStatus::AtLower,
                        NonbasicStatus::AtLower, NonbasicStatus::Basic,
                        NonbasicStatus::AtLower};

        SimplexDiagnostics diag;
        SimplexBasis final_basis;
        auto raw = sor::engines::solve_dual_simplex(problem, opts, diag,
                                                    &final_basis, &start);
        const auto result = sor::certify::finalize_result(
            std::move(raw), sor::engines::simplex_evidence(diag, opts));

        CHECK(diag.warm_starts == 1);
        CHECK(result.status == Status::Optimal);
        CHECK(result.proof == ProofLevel::ProvedOptimalFP);
        CHECK_NEAR(result.objective, 5.000045, 1e-9);

        // The scenario is the point of the test: if the perturbation ever stops
        // hiding the two infeasibilities, the dual solves this outright and the
        // assertions below would pass vacuously.
        CHECK(diag.perturbed_costs > 0);
        CHECK(diag.primal_cleanups == 1);
        CHECK_NEAR(diag.cleanup_dual_infeasibility, 2e-5, 1e-7);

        // The property under test. Before the fix this was 0.5 -- the flip pass
        // had pushed basic x3 from 5.0 to 4.0, under its lower bound of 4.5.
        CHECK(diag.cleanup_primal_infeasibility == 0.0);
        // ...and its consequence: the clean-up starts in phase 2 and stays there.
        // Neither engine runs a phase-1 pivot on this model.
        CHECK(diag.phase1_iterations == 0);
        CHECK(diag.primal_cleanup_iterations <= 4);
    }
}

// Partitioned vs full row PRICE, over many random bases.
//
// The pivotal row is now assembled from a nonbasic-partitioned copy of A whose
// permutation is maintained incrementally across pivots. An incremental
// permutation is exactly the kind of thing that is right for a hundred pivots
// and wrong on the hundred-and-first, and a wrong pivotal row does not
// necessarily produce a wrong ANSWER -- it produces a different, still-optimal
// trajectory. So compare trajectories, not objectives:
// SOR_DUAL_FILTER_ACTIVE_PIVOTAL forces the unpartitioned scan with the same
// keep filter, which is the reference implementation for this.
void test_partitioned_price_matches_full_scan_on_random_bases() {
    SOR_FN();
    ScopedEnvironment verify_heap("SOR_DUAL_VERIFY_CHUZR_HEAP", "1");
    std::mt19937 rng(20260909u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int compared = 0;

    for (int trial = 0; trial < 60; ++trial) {
        const Index rows = 4 + static_cast<Index>(rng() % 14);
        const Index cols = rows + static_cast<Index>(rng() % 18);
        std::vector<Index> ri, ci;
        std::vector<f64> vv;
        for (Index i = 0; i < rows; ++i)
            for (Index j = 0; j < cols; ++j)
                if (rng() % 100 < 35) {
                    ri.push_back(i); ci.push_back(j);
                    vv.push_back(std::round((4.0 * unit(rng) - 2.0) * 4.0) / 4.0);
                }
        if (ri.empty()) continue;

        sor::model::LpProblem p;
        p.A = sor::sparse::from_triplets(rows, cols, ri, ci, vv);
        p.c.resize(static_cast<std::size_t>(cols));
        p.col_lo.assign(static_cast<std::size_t>(cols), 0.0);
        p.col_hi.assign(static_cast<std::size_t>(cols), 0.0);
        for (Index j = 0; j < cols; ++j) {
            const auto u = static_cast<std::size_t>(j);
            p.c[u] = std::round((2.0 * unit(rng) - 1.0) * 8.0) / 4.0;
            // Mix bound classes: the partition only holds columns that are
            // nonbasic AND not permanently fixed, so fixed and boxed columns
            // are the interesting ones.
            switch (rng() % 4) {
                case 0: p.col_hi[u] = sor::model::kInf; break;      // one-sided
                case 1: p.col_hi[u] = 1.0 + 3.0 * unit(rng); break; // boxed
                case 2: p.col_hi[u] = 0.0; break;                   // fixed
                default: p.col_hi[u] = 2.0; break;
            }
        }
        p.row_lo.assign(static_cast<std::size_t>(rows), 0.0);
        p.row_hi.assign(static_cast<std::size_t>(rows), 0.0);
        for (Index i = 0; i < rows; ++i) {
            const auto u = static_cast<std::size_t>(i);
            p.row_lo[u] = -1.0 - 4.0 * unit(rng);
            p.row_hi[u] = (rng() % 3 == 0) ? p.row_lo[u] : 1.0 + 4.0 * unit(rng);
        }

        SimplexOptions opts;
        opts.method = sor::engines::SimplexMethod::Dual;
        opts.presolve = false;

        SimplexDiagnostics part_diag, full_diag;
        SimplexBasis part_basis, full_basis;
        auto part_raw = sor::engines::solve_simplex(p, opts, part_diag, &part_basis);
        SimplexDiagnostics fd;
        SimplexBasis fb;
        {
            ScopedEnvironment filter("SOR_DUAL_FILTER_ACTIVE_PIVOTAL", "1");
            full_diag = fd;
            auto full_raw = sor::engines::solve_simplex(p, opts, full_diag, &full_basis);
            CHECK(part_raw.proposed_status == full_raw.proposed_status);
            if (part_raw.proposed_status == Status::Optimal) {
                CHECK(part_raw.objective == full_raw.objective);   // bitwise
                CHECK(part_raw.x == full_raw.x);
                CHECK(part_raw.y == full_raw.y);
            }
        }
        CHECK(part_diag.iterations == full_diag.iterations);
        CHECK(part_basis.basic == full_basis.basic);
        CHECK(part_basis.status == full_basis.status);
        // The partitioned path must touch no more than the filtered scan, and
        // on anything non-trivial strictly less.
        CHECK(part_diag.dual_pivotal_entries_kept ==
              full_diag.dual_pivotal_entries_kept);
        CHECK(part_diag.dual_pivotal_entries_full <=
              full_diag.dual_pivotal_entries_full);
        ++compared;
    }
    CHECK(compared > 40);
}

// The numerical-trouble trigger, and the cost-shift bound that shares its
// response. Both exist because an eta-count or nnz trigger cannot see
// accumulated error in the factorization -- only a disagreement between two
// computations of the same number can.
void test_numerical_trouble_trigger_and_shift_bound() {
    SOR_FN();
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Dual;
    opts.presolve = false;

    // A well-conditioned model must not trip either guard. If it did, the
    // trigger would be refactorizing on healthy arithmetic and the shift bound
    // would be refusing legitimate corrections.
    {
        const auto run = solve_text(sor::test::kTestLpMps, opts);
        CHECK(run.r.status == Status::Optimal);
        CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
        CHECK(run.diag.numerical_trouble_refactors == 0);
        CHECK(run.diag.refused_cost_shifts == 0);
    }

    // ...and the trigger must be reachable at all: disabling it is the only
    // difference, so a run with it off has to agree on the answer.
    {
        SimplexOptions off = opts;
        off.numerical_trouble_tol = 0.0;
        const auto a = solve_text(sor::test::kTestLpMps, opts);
        const auto b = solve_text(sor::test::kTestLpMps, off);
        CHECK(a.r.objective == b.r.objective);
        CHECK(a.diag.iterations == b.diag.iterations);
    }

    // A model that does shift: the bound scales with dual_feas_tol, so
    // tightening the tolerance by six orders of magnitude must refuse at least
    // as many shifts as the default does, and must still prove.
    {
        SimplexOptions shifty = opts;
        shifty.dual_cost_perturbation_multiplier = 1.0;
        const auto loose = solve_text(sor::test::kTestLpMps, shifty);
        SimplexOptions tight = shifty;
        tight.dual_feas_tol = 1e-12;
        const auto strict = solve_text(sor::test::kTestLpMps, tight);
        CHECK(loose.r.status == Status::Optimal);
        CHECK(strict.r.status == Status::Optimal);
        CHECK(strict.diag.refused_cost_shifts >=
              loose.diag.refused_cost_shifts);
    }
}

void test_auto_candidate_order_uses_feasibility_before_gap() {
    SOR_FN();
    SimplexOptions opts;
    sor::core::RawResult early, later;
    early.proposed_status = Status::Interrupted;
    later.proposed_status = Status::Interrupted;
    early.objective = 1.0;
    later.objective = 100.0;

    // A dual-feasible early probe may have a numerically zero objective gap
    // while still being badly primal infeasible. The gap is not an optimality
    // metric in that state; the later basis with the smaller actual violation
    // must win. This directly guards Auto's multi-stage selection invariant.
    SimplexDiagnostics early_diag, later_diag;
    early_diag.primal_residual = 100.0;
    early_diag.dual_residual = 0.0;
    early_diag.dual_bound_finite = true;
    early_diag.gap_rel = 0.0;
    later_diag.primal_residual = 1.0;
    later_diag.dual_residual = 0.0;
    later_diag.dual_bound_finite = true;
    later_diag.gap_rel = 10.0;
    CHECK(sor::engines::detail::prefer_simplex_candidate(
        later, later_diag, early, early_diag, opts, false));
    CHECK(!sor::engines::detail::prefer_simplex_candidate(
        early, early_diag, later, later_diag, opts, false));

    // Once both points are primal and dual feasible, gap becomes legitimate
    // and is used before objective quality.
    early_diag.primal_residual = 0.0;
    later_diag.primal_residual = 0.0;
    early_diag.gap_rel = 1e-8;
    later_diag.gap_rel = 1e-10;
    CHECK(sor::engines::detail::prefer_simplex_candidate(
        later, later_diag, early, early_diag, opts, false));

    // A usable primal-feasible point dominates an infeasible point regardless
    // of the latter's status label or apparently better objective.
    early.proposed_status = Status::Optimal;
    early_diag.primal_residual = 1e-3;
    early_diag.gap_rel = 0.0;
    later.proposed_status = Status::Interrupted;
    later_diag.primal_residual = 0.0;
    later_diag.dual_residual = 1.0;
    later_diag.dual_bound_finite = false;
    CHECK(sor::engines::detail::prefer_simplex_candidate(
        later, later_diag, early, early_diag, opts, false));
}

// Every row must have exactly one basic variable and the basis must be a
// permutation of distinct variables. A duplicated basic variable is the classic
// symptom of a mishandled leaving-variable update.
void test_basis_wellformed() {
    SOR_FN();
    const auto run = solve_text(sor::test::kTestLpMps);
    const Index m = run.problem.n_rows();
    CHECK(static_cast<Index>(run.basis.basic.size()) == m);

    std::vector<int> seen(run.basis.status.size(), 0);
    for (const Index v : run.basis.basic) {
        CHECK(v >= 0 && v < static_cast<Index>(run.basis.status.size()));
        CHECK(run.basis.status[static_cast<std::size_t>(v)] == NonbasicStatus::Basic);
        ++seen[static_cast<std::size_t>(v)];
    }
    for (const int k : seen) CHECK(k <= 1);

    std::size_t n_basic = 0;
    for (const auto s : run.basis.status)
        if (s == NonbasicStatus::Basic) ++n_basic;
    CHECK(n_basic == static_cast<std::size_t>(m));
}

void test_features_mps_agrees_with_model() {
    SOR_FN();
    const auto run = solve_text(sor::test::kFeaturesMps);
    // RANGES / MI / FR / FX and an objective constant all have to survive the
    // augmented-form conversion. The assertion is self-consistency: whatever the
    // engine returns, the model's own objective evaluation must agree with it.
    CHECK(run.r.status == Status::Optimal || run.r.status == Status::Infeasible ||
          run.r.status == Status::Unbounded);
    if (run.r.status == Status::Optimal) {
        CHECK_NEAR(run.problem.objective(run.r.x), run.r.objective, 1e-9);
        CHECK(run.problem.max_row_violation(run.r.x) < 1e-9);
        CHECK(run.problem.max_bound_violation(run.r.x) < 1e-9);
    }
}

// x1 + x2 <= 1 with x1, x2 >= 2 has no feasible point. Phase 1 must terminate
// with positive infeasibility and the result must NOT be labelled Optimal.
void test_infeasible() {
    SOR_FN();
    const std::string mps = R"(NAME          INFEAS
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0        R1        1.0
    X2        COST      1.0        R1        1.0
RHS
    RHS       R1        1.0
BOUNDS
 LO BND       X1        2.0
 LO BND       X2        2.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Infeasible);
    CHECK(run.r.proof < ProofLevel::ProvedOptimalFP);
}

void test_primal_phase1_exports_checked_farkas_certificate() {
    SOR_FN();
    const std::string mps = R"(NAME          PINF
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      0.0        R1        1.0
    X2        COST      0.0        R1        1.0
RHS
    RHS       R1        1.0
BOUNDS
 LO BND       X1        2.0
 LO BND       X2        2.0
ENDATA
)";
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Primal;
    opts.presolve = false;
    const auto run = solve_text(mps, opts);
    CHECK(run.r.status == Status::Infeasible);
    CHECK(run.r.ray_certified);
    CHECK(run.r.dual_farkas_ray.certified);
    CHECK(run.r.ray.size() == static_cast<std::size_t>(run.problem.n_rows()));
    const f64 v = sor::engines::farkas_violation(run.problem, run.r.ray);
    ::sor::test::report(std::isfinite(v) && v <= opts.primal_feas_tol,
                        "primal phase-I independent farkas re-check",
                        __FILE__, __LINE__,
                        "violation=" + std::to_string(v));
}

// Same infeasible LP as test_infeasible(), forced through the dual engine
// (where Farkas capture lives) and checked two ways: finalize_result()'s own
// gate (ray_certified) and an INDEPENDENTLY re-run farkas_violation() call in
// this test, so a bug shared between the capture code and the verification
// code inside finalize_result can't hide.
void test_infeasible_farkas_certificate() {
    SOR_FN();
    const std::string mps = R"(NAME          INFEAS
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0        R1        1.0
    X2        COST      1.0        R1        1.0
RHS
    RHS       R1        1.0
BOUNDS
 LO BND       X1        2.0
 LO BND       X2        2.0
ENDATA
)";
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Dual;
    const auto run = solve_text(mps, opts);
    CHECK(run.r.status == Status::Infeasible);
    CHECK(run.r.ray_certified);
    CHECK(run.r.ray.size() == static_cast<std::size_t>(run.problem.n_rows()));

    const f64 v = sor::engines::farkas_violation(run.problem, run.r.ray);
    ::sor::test::report(std::isfinite(v) && v <= 1e-7, "independent farkas re-check",
                        __FILE__, __LINE__, "violation=" + std::to_string(v));
}

// A no-entering-column Farkas conclusion must be retried after restoring the
// original costs: perturbation can change which bound a boxed nonbasic occupies,
// and the ray sign conditions depend on those statuses even though feasibility
// itself does not depend on the objective.
void test_dual_cost_perturbation_cleans_before_farkas_proof() {
    SOR_FN();
    const std::string mps = R"(NAME          PERTINF
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0        R1        1.0
    X2        COST      1.0        R1        1.0
RHS
    RHS       R1        1.0
BOUNDS
 LO BND       X1        2.0
 LO BND       X2        2.0
ENDATA
)";
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Dual;
    opts.presolve = false;
    opts.dual_cost_perturbation_multiplier = 1.0;
    const auto run = solve_text(mps, opts);
    CHECK(run.r.status == Status::Infeasible);
    CHECK(run.r.ray_certified);
    CHECK(run.diag.perturbed_costs > 0);
    CHECK(run.diag.perturbation_cleanups == 1);
    const f64 v = sor::engines::farkas_violation(run.problem, run.r.ray);
    CHECK(std::isfinite(v) && v <= 1e-7);
}

// A feasible LP must never carry a certified ray -- finalize_result()'s gate
// is keyed on Status::Infeasible, not on whatever an engine happens to leave
// in raw.ray.
void test_feasible_lp_has_no_ray() {
    SOR_FN();
    const auto run = solve_text(sor::test::kTestLpMps);
    CHECK(run.r.status == Status::Optimal);
    CHECK(!run.r.ray_certified);
    CHECK(run.r.ray.empty());
}

// min -x with x >= 0 and no constraint on growth.
void test_unbounded() {
    SOR_FN();
    const std::string mps = R"(NAME          UNBND
ROWS
 N  COST
 G  R1
COLUMNS
    X1        COST      -1.0       R1        1.0
RHS
    RHS       R1        0.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Unbounded);
    CHECK(run.r.proof < ProofLevel::ProvedOptimalFP);

    // Force the dual phase-1 terminal branch: the negative-cost, lower-only
    // column has no dual-feasible bound at the logical basis. Once the
    // subproblem proves dual infeasibility, restoring the real bounds reveals
    // a primal-feasible point and therefore unboundedness.
    SimplexOptions dual_opts;
    dual_opts.method = sor::engines::SimplexMethod::Dual;
    dual_opts.presolve = false;
    dual_opts.dual_cost_perturbation_multiplier = 1.0;
    const auto dual = solve_text(mps, dual_opts);
    CHECK(dual.r.status == Status::Unbounded);
    CHECK(dual.r.proof < ProofLevel::ProvedOptimalFP);
    CHECK(dual.diag.phase_restarts > 0);
    // Phase 2, not 1: under the Choose policy's exact-DSE drift rebuild the
    // run restores the true bounds and finishes the unboundedness proof in
    // phase 2 instead of terminating inside phase 1. Same outcome, one more
    // handover -- the status and proof-level checks above are the contract.
    CHECK(dual.diag.final_phase == 2);
    CHECK(dual.diag.perturbed_costs > 0);
    CHECK(dual.diag.perturbation_cleanups == 1);
}

// A phase-1 run that produces an infeasibility candidate is deliberately
// cross-checked by primal in the public forced-dual dispatcher. This covers
// the dual-failed fallback without relying on Auto's routing policy.
void test_dual_phase1_infeasible_falls_back_to_primal() {
    SOR_FN();
    const std::string mps = R"(NAME          DP1UNDET
ROWS
 N  COST
 G  RLO
 L  RHI
COLUMNS
    X1        COST      -1.0       RLO       1.0
    X1        RHI       1.0
RHS
    RHS       RLO       1.0        RHI       0.0
ENDATA
)";
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    const auto problem = sor::io::read_mps(in, rep);

    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Dual;
    opts.presolve = false;
    SimplexDiagnostics direct_diag;
    const auto direct = sor::engines::solve_dual_simplex(
        problem, opts, direct_diag, nullptr, nullptr);
    CHECK(direct.proposed_status == Status::Infeasible);
    CHECK(direct_diag.phase1_iterations > 0);
    CHECK(!direct.ray.empty());

    const auto dispatched = solve_text(mps, opts);
    CHECK(dispatched.r.status == Status::Infeasible);
    CHECK(dispatched.r.proof < ProofLevel::ProvedOptimalFP);
    CHECK(dispatched.diag.dual_stages == 1);
    CHECK(dispatched.diag.primal_stages == 1);
    CHECK(dispatched.diag.stages == 2);
    // Cross-checking the infeasibility candidate must spend the remainder of
    // the caller's allowance, rather than giving each engine a fresh cap.
    for (const auto method : {sor::engines::SimplexMethod::Dual,
                              sor::engines::SimplexMethod::Auto}) {
        opts.method = method;
        for (std::uint64_t cap = 1; cap <= dispatched.diag.iterations + 1; ++cap) {
            opts.max_iterations = cap;
            const auto limited = solve_text(mps, opts);
            CHECK(limited.diag.iterations <= cap);
            CHECK(limited.r.iterations == limited.diag.iterations);
            CHECK(limited.r.status != Status::Optimal);
        }
    }
}

// An equality row plus a range row, where the optimum is forced to an interior
// point of one variable's box. Exercises a basic structural variable rather than
// a vertex made only of bounds.
void test_equality_and_range() {
    SOR_FN();
    //  min  x1 + 2*x2
    //  s.t. x1 + x2 == 3
    //       1 <= x1 <= 2
    //       0 <= x2
    //  Cheapest is to push x1 to its upper bound 2, so x2 = 1, objective 4.
    const std::string mps = R"(NAME          EQ
ROWS
 N  COST
 E  R1
COLUMNS
    X1        COST      1.0        R1        1.0
    X2        COST      2.0        R1        1.0
RHS
    RHS       R1        3.0
BOUNDS
 LO BND       X1        1.0
 UP BND       X1        2.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, 4.0, 1e-9);
    CHECK_NEAR(run.r.x[0], 2.0, 1e-9);
    CHECK_NEAR(run.r.x[1], 1.0, 1e-9);
}

// A free variable must be able to go negative. min x s.t. x >= -5, x free.
void test_free_variable() {
    SOR_FN();
    const std::string mps = R"(NAME          FREEV
ROWS
 N  COST
 G  R1
COLUMNS
    X1        COST      1.0        R1        1.0
RHS
    RHS       R1        -5.0
BOUNDS
 FR BND       X1
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, -5.0, 1e-9);
    CHECK_NEAR(run.r.x[0], -5.0, 1e-9);
}

// ---- dual phase 1 (Koberstein-Suhl subproblem) ------------------------------

// A logical starting basis that is DUAL INFEASIBLE, which is what forces dual
// phase 1 to run at all: x1 is one-sided [0, inf) with a negative cost, so at
// pi = 0 its reduced cost has the wrong sign and no bound flip can fix it.
//
// The old phase 1 priced such a column directly with a primal-style ratio test
// and could reach "unblocked improving column at a primal-infeasible basis",
// which is not a certificate of anything -- 30 of the 93 Netlib models ended
// there and were silently finished by the primal engine. The subproblem method
// gives every column two finite artificial bounds, so that state is
// unreachable and `--method dual` must solve this by itself.
void test_dual_phase1_recovers_dual_infeasible_start() {
    SOR_FN();
    const std::string mps = R"(NAME          DP1
ROWS
 N  COST
 L  R1
 L  R2
COLUMNS
    X1        COST      -3.0       R1        2.0
    X1        R2        1.0
    X2        COST      -2.0       R1        1.0
    X2        R2        3.0
RHS
    RHS       R1        12.0       R2        15.0
ENDATA
)";
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Dual;
    const auto run = solve_text(mps, opts);
    CHECK(run.r.status == Status::Optimal);
    // 2x1 + x2 <= 12, x1 + 3x2 <= 15 meet at (21/5, 18/5), where
    // -3(21/5) - 2(18/5) = -99/5 = -19.8.
    CHECK_NEAR(run.r.objective, -19.8, 1e-7);
    CHECK_NEAR(run.r.x[0], 4.2, 1e-7);
    CHECK_NEAR(run.r.x[1], 3.6, 1e-7);
    CHECK(run.diag.phase1_iterations > 0);   // phase 1 really ran
    CHECK(run.diag.final_phase == 2);        // and handed over to phase 2
}

// The dual's leaving variable is parked by inferring which bound it was
// violating. When it violates NEITHER -- a degenerate step, or a row chosen
// within tolerance -- the fallback branch used to be able to name a bound that
// does not exist, setting value[] to +-infinity. Nothing catches that
// downstream: xB inherits the infinity, the objective becomes NaN, and BOTH
// infeasibility sums read zero because every NaN comparison is false, so the
// engine spins at a "feasible" point it can never leave. Seen on 80bau3b as
// 47 000+ phase-2 iterations against `prim-infeas 0.0  obj -nan`.
//
// This model mixes a free column, a one-sided column and an equality row so
// that leaving variables with an infinite bound on one side actually occur.
void test_dual_leaving_variable_never_parks_on_an_infinite_bound() {
    SOR_FN();
    const std::string mps = R"(NAME          DPINF
ROWS
 N  COST
 E  R1
 G  R2
 L  R3
COLUMNS
    XF        COST      1.0        R1        1.0
    XF        R2        1.0
    XP        COST      -2.0       R1        1.0
    XP        R3        1.0
    XQ        COST      -1.0       R2        1.0
    XQ        R3        2.0
RHS
    RHS       R1        4.0        R2        1.0
    RHS       R3        6.0
BOUNDS
 FR BND       XF
ENDATA
)";
    for (const auto method : {sor::engines::SimplexMethod::Dual,
                              sor::engines::SimplexMethod::Auto}) {
        SimplexOptions opts;
        opts.method = method;
        const auto run = solve_text(mps, opts);
        CHECK(run.r.status == Status::Optimal);
        CHECK(std::isfinite(run.r.objective));
        for (const f64 v : run.r.x) CHECK(std::isfinite(v));
        // The real guard: a NaN/infinite value[] shows up as a violated
        // constraint that the engine's own tolerance test could not see.
        CHECK(run.diag.primal_residual <= 1e-6);
        if (method == sor::engines::SimplexMethod::Dual)
            CHECK(run.diag.phase1_iterations > 0);
    }
}

// maximize must produce the same point as the equivalent minimize, with the
// objective reported in the original sense.
void test_maximize() {
    SOR_FN();
    const std::string mps = R"(NAME          MAXP
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0        R1        1.0
RHS
    RHS       R1        7.0
ENDATA
)";
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    auto problem = sor::io::read_mps(in, rep);
    problem.maximize = true;

    SimplexDiagnostics diag;
    SimplexOptions opts;
    auto raw = sor::engines::solve_simplex(problem, opts, diag, nullptr);
    const auto r = sor::certify::finalize_result(
        std::move(raw), sor::engines::simplex_evidence(diag, opts));

    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, 7.0, 1e-9);
    CHECK_NEAR(r.x[0], 7.0, 1e-9);
    CHECK_NEAR(problem.objective(r.x), 7.0, 1e-9);
}

// A pivot can make the all-logical starting point primal feasible and switch
// the primal engine from its local infeasibility objective to the real one.
// Those two objectives deliberately want opposite things here:
//
//   phase 1: increase X1 until the row activity reaches its lower bound;
//   phase 2: minimize +X1, so stop at that lower bound.
//
// The phase-1 pivotal-row update leaves the row logical with a stale negative
// reduced cost.  If the normal pivot-driven phase transition fails to
// invalidate/rebuild reduced costs, phase 2 performs two unnecessary pivots
// (to X1's upper bound and back) before a later refactorization repairs the
// state.  Thus the zero phase-2-pivot assertion is a path-level regression
// check for the rebuild, while the proof checks that the final answer remains
// independently certifiable.
void test_primal_phase1_pivot_rebuilds_phase2_reduced_costs() {
    SOR_FN();
    const std::string mps = R"(NAME          P1P2RC
ROWS
 N  COST
 G  R1
COLUMNS
    X1        COST       1.0       R1         1.0
RHS
    RHS       R1         1.0
BOUNDS
 UP BND       X1         2.0
ENDATA
)";

    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Primal;
    opts.presolve = false;
    const auto run = solve_text(mps, opts);

    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK_NEAR(run.r.objective, 1.0, 1e-9);
    CHECK_NEAR(run.r.x[0], 1.0, 1e-9);
    CHECK(run.diag.primal_residual <= 1e-9);
    CHECK(run.diag.dual_residual <= 1e-9);
    CHECK(run.diag.phase1_iterations == 1);
    CHECK(run.diag.final_phase == 2);
    CHECK(run.diag.phase2_iterations == 0);
    CHECK(run.diag.dual_rebuilds >= 2);
}

// A time limit of zero-ish must interrupt rather than run to completion, and the
// interrupted result must never carry an Optimal claim.
void test_time_limit_is_honoured() {
    SOR_FN();
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Primal;
    opts.max_iterations = 1;          // force a limit hit on a non-trivial LP
    const auto run = solve_text(sor::test::kTestLpMps, opts);
    CHECK(run.r.status == Status::Interrupted);
    CHECK(run.r.proof < ProofLevel::ProvedOptimalFP);
    CHECK(run.diag.iterations == 1);
    CHECK(run.diag.stages == 1);
    CHECK(!run.r.termination_reason.empty());
}

// A degenerate LP with many ties. The requirement is termination with a correct
// objective (Harris tie-break plus EXPAND on zero-length steps).
void test_degenerate() {
    SOR_FN();
    //  min -x1 - x2 - x3
    //  s.t. three identical rows  x1 + x2 + x3 <= 1
    //       0 <= xi <= 1
    const std::string mps = R"(NAME          DEGEN
ROWS
 N  COST
 L  R1
 L  R2
 L  R3
COLUMNS
    X1        COST      -1.0       R1        1.0
    X1        R2        1.0        R3        1.0
    X2        COST      -1.0       R1        1.0
    X2        R2        1.0        R3        1.0
    X3        COST      -1.0       R1        1.0
    X3        R2        1.0        R3        1.0
RHS
    RHS       R1        1.0        R2        1.0
    RHS       R3        1.0
BOUNDS
 UP BND       X1        1.0
 UP BND       X2        1.0
 UP BND       X3        1.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, -1.0, 1e-9);
    CHECK(run.r.max_primal_violation < 1e-9);
}

// Badly scaled coefficients: Ruiz equilibration plus threshold pivoting have to
// keep this accurate enough that the claim gate still accepts it.
void test_ill_conditioned() {
    SOR_FN();
    //  min  -x1 - x2
    //  s.t. 1e6*x1 +   1e-6*x2 <= 1e6
    //          x1  +      x2   <= 1.5
    //       xi >= 0
    const std::string mps = R"(NAME          ILLC
ROWS
 N  COST
 L  R1
 L  R2
COLUMNS
    X1        COST      -1.0       R1        1000000.0
    X1        R2        1.0
    X2        COST      -1.0       R1        0.000001
    X2        R2        1.0
RHS
    RHS       R1        1000000.0  R2        1.5
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, -1.5, 1e-7);
    CHECK(run.r.max_primal_violation < 1e-7);
}

// No rows at all: the LP is a box, and the answer is each variable at its
// cheapest bound. Exercises the m == 0 path through the factorization.
void test_no_rows() {
    SOR_FN();
    const std::string mps = R"(NAME          BOXONLY
ROWS
 N  COST
COLUMNS
    X1        COST      1.0
    X2        COST      -1.0
RHS
BOUNDS
 UP BND       X1        5.0
 UP BND       X2        5.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, -5.0, 1e-9);
    CHECK_NEAR(run.r.x[0], 0.0, 1e-9);
    CHECK_NEAR(run.r.x[1], 5.0, 1e-9);
}

// Presolve can tighten a singleton inequality into a variable bound and remove
// the now-redundant row.  The reduced dual is not automatically a valid dual
// for the original variable bounds (x=1 is interior to x>=0 here), so the
// solver must retry without presolve before claiming a proof.
void test_presolve_certificate_fallback() {
    SOR_FN();
    const std::string mps = R"(NAME          PRECERT
ROWS
 N  COST
 G  R1
COLUMNS
    X1        COST      1.0        R1        1.0
RHS
    RHS       R1        1.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK_NEAR(run.r.objective, 1.0, 1e-9);
    CHECK(run.r.max_primal_violation <= 1e-9);
    CHECK(run.r.max_dual_violation <= 1e-9);
}

void test_presolve_singleton_column_elimination_lifts_proof_and_basis() {
    SOR_FN();
    // x occurs only in an equality and is free, so its bounds cannot induce a
    // hidden constraint when x is substituted out:
    //   x + 2y = 5, 0 <= y <= 2, min 3x+y
    // becomes min 15-5y. The lifted optimum is y=2, x=1, with equality
    // multiplier 3 and x basic in the restored row.
    const std::string mps = R"(NAME          SCELIM
ROWS
 N  COST
 E  R1
COLUMNS
    X         COST      3.0        R1        1.0
    Y         COST      1.0        R1        2.0
RHS
    RHS       R1        5.0
BOUNDS
 FR BND       X
 UP BND       Y         2.0
ENDATA
)";
    const auto reduced = solve_text(mps);
    CHECK(reduced.r.status == Status::Optimal);
    CHECK(reduced.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK_NEAR(reduced.r.objective, 5.0, 1e-9);
    CHECK_NEAR(reduced.r.x[0], 1.0, 1e-9);
    CHECK_NEAR(reduced.r.x[1], 2.0, 1e-9);
    CHECK(reduced.r.y.size() == 1);
    CHECK_NEAR(reduced.r.y[0], 3.0, 1e-9);
    CHECK(reduced.diag.presolve_rows_removed == 1);
    CHECK(reduced.diag.presolve_cols_removed == 2);
    CHECK(reduced.diag.presolve_singleton_columns_removed == 1);
    CHECK(reduced.basis.basic.size() == 1);
    CHECK(reduced.basis.basic[0] == 0);
    CHECK(reduced.basis.status[0] == NonbasicStatus::Basic);

    SimplexOptions no_presolve;
    no_presolve.presolve = false;
    const auto original = solve_text(mps, no_presolve);
    CHECK(original.r.status == Status::Optimal);
    CHECK_NEAR(original.r.objective, reduced.r.objective, 1e-9);
    CHECK_NEAR(original.r.x[0], reduced.r.x[0], 1e-9);
    CHECK_NEAR(original.r.x[1], reduced.r.x[1], 1e-9);

    // Objective substitution and empty-column bound choice must also respect
    // maximization sense. The same expression 15-5y is now maximized at y=0.
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    auto maximization = sor::io::read_mps(in, rep);
    maximization.maximize = true;
    SimplexDiagnostics max_diag;
    SimplexBasis max_basis;
    SimplexOptions max_opts;
    auto max_raw = sor::engines::solve_simplex(
        maximization, max_opts, max_diag, &max_basis);
    const auto max_result = sor::certify::finalize_result(
        std::move(max_raw), sor::engines::simplex_evidence(max_diag, max_opts));
    CHECK(max_result.status == Status::Optimal);
    CHECK(max_result.proof == ProofLevel::ProvedOptimalFP);
    CHECK_NEAR(max_result.objective, 15.0, 1e-9);
    CHECK_NEAR(max_result.x[0], 5.0, 1e-9);
    CHECK_NEAR(max_result.x[1], 0.0, 1e-9);
    CHECK(max_diag.presolve_singleton_columns_removed == 1);
}

void test_presolve_chained_singleton_columns_recover_duals_and_basis() {
    SOR_FN();
    // Both free columns are equality singletons, but they share bounded y:
    //   x + y = 3
    //       y + z = 2
    //   0 <= y <= 2, min x + 2z.
    // Forward elimination records (R0,x) then (R1,z), after which empty y is
    // fixed high. Reverse dual recovery must therefore set y_R1=2 first and
    // y_R0=1 second. Replaying these in forward order would leave a nonzero
    // reduced cost for an eliminated basic column.
    sor::model::LpProblem lp;
    lp.name = "CHAINED_SINGLETON_COLUMNS";
    lp.A = sor::sparse::from_triplets(
        2, 3, {0, 0, 1, 1}, {0, 1, 1, 2},
        {1.0, 1.0, 1.0, 1.0});
    lp.c = {1.0, 0.0, 2.0};
    lp.row_lo = lp.row_hi = {3.0, 2.0};
    lp.col_lo = {-sor::model::kInf, 0.0, -sor::model::kInf};
    lp.col_hi = { sor::model::kInf, 2.0,  sor::model::kInf};

    const auto run = solve_problem(lp);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(run.r.downgrade_reason.empty());
    CHECK(run.r.x.size() == 3);
    CHECK_NEAR(run.r.x[0], 1.0, 1e-9);
    CHECK_NEAR(run.r.x[1], 2.0, 1e-9);
    CHECK_NEAR(run.r.x[2], 0.0, 1e-9);
    CHECK_NEAR(run.r.objective, 1.0, 1e-9);
    CHECK(run.r.y.size() == 2);
    CHECK_NEAR(run.r.y[0], 1.0, 1e-9);
    CHECK_NEAR(run.r.y[1], 2.0, 1e-9);
    CHECK(run.r.max_primal_violation <= 1e-9);
    CHECK(run.r.max_dual_violation <= 1e-9);
    CHECK(run.diag.dual_bound_finite);
    CHECK_NEAR(run.diag.dual_objective, 1.0, 1e-9);
    CHECK(run.diag.gap_rel <= 1e-9);
    CHECK(run.diag.presolve_rows_removed == 2);
    CHECK(run.diag.presolve_cols_removed == 3);
    CHECK(run.diag.presolve_singleton_columns_removed == 2);
    CHECK(run.diag.presolve_retries == 0);
    CHECK(run.diag.stages == 1);

    // Lifted basis: x owns R0 and z owns R1. The survivor y became an empty
    // objective column and is nonbasic at its upper bound.
    CHECK(run.basis.n_struct == 3);
    CHECK(run.basis.basic.size() == 2);
    CHECK(run.basis.status.size() == 5);
    CHECK(run.basis.basic[0] == 0);
    CHECK(run.basis.basic[1] == 2);
    CHECK(run.basis.status[0] == NonbasicStatus::Basic);
    CHECK(run.basis.status[1] == NonbasicStatus::AtUpper);
    CHECK(run.basis.status[2] == NonbasicStatus::Basic);
    std::vector<int> seen(run.basis.status.size(), 0);
    for (const Index j : run.basis.basic) {
        CHECK(j >= 0 && j < static_cast<Index>(seen.size()));
        ++seen[static_cast<std::size_t>(j)];
    }
    for (const int count : seen) CHECK(count <= 1);
}

void test_presolve_retry_counter_survives_rejected_retry() {
    SOR_FN();
    // At ordinary tolerances this chained singleton model is proved directly.
    // With a deliberately sub-ulp gap tolerance, conservative dual reporting
    // cannot prove the gap, so the safety retry is attempted. Diagnostics must
    // include both stages' work, regardless of which candidate is retained.
    constexpr f64 a = 3.1;
    sor::model::LpProblem lp;
    lp.name = "REJECTED_PRESOLVE_RETRY";
    lp.A = sor::sparse::from_triplets(
        2, 3, {0, 0, 1, 1}, {0, 1, 1, 2},
        {a, 1.0, 1.0, a});
    lp.c = {1.0, 0.0, 2.0};
    lp.row_lo = lp.row_hi = {3.0, 2.0};
    lp.col_lo = {-sor::model::kInf, 0.0, -sor::model::kInf};
    lp.col_hi = { sor::model::kInf, 2.0,  sor::model::kInf};

    SimplexOptions opts;
    opts.gap_tol = 1e-18;
    const auto retained = solve_problem(lp, opts);
    CHECK(retained.diag.presolve_retries == 1);
    CHECK(retained.diag.gap_rel > opts.gap_tol);
    CHECK(retained.r.status == Status::Feasible);
    CHECK(retained.r.proof == ProofLevel::FeasibleWithGap);

    SimplexOptions original_opts = opts;
    original_opts.presolve = false;
    const auto rejected = solve_problem(lp, original_opts);
    CHECK(rejected.diag.iterations > 0);
    CHECK(retained.diag.iterations >= rejected.diag.iterations);
    CHECK(retained.diag.stages >= 2);
    CHECK(retained.diag.total_ms >= rejected.diag.total_ms);
    CHECK(rejected.diag.gap_rel > opts.gap_tol);
    CHECK_NEAR(retained.r.objective, rejected.r.objective, 1e-12);
}

void test_presolve_forcing_rows_lift_primal_dual_and_proof() {
    SOR_FN();
    // Maximum-activity forcing: x-y >= 3 with x in [0,2], y in [-1,4]
    // has only x=2,y=-1. For min 2x+y, the recovered lower-row multiplier
    // is +2; the upper x has zero reduced cost and lower y has positive
    // reduced cost.
    {
        sor::model::LpProblem lp;
        lp.name = "FORCE_MAX";
        lp.A = sor::sparse::from_triplets(
            1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
        lp.c = {2.0, 1.0};
        lp.obj_offset = 4.0;
        lp.row_lo = {3.0};
        lp.row_hi = {sor::model::kInf};
        lp.col_lo = {0.0, -1.0};
        lp.col_hi = {2.0, 4.0};

        const auto run = solve_problem(lp);
        CHECK(run.r.status == Status::Optimal);
        CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
        CHECK_NEAR(run.r.x[0], 2.0, 1e-9);
        CHECK_NEAR(run.r.x[1], -1.0, 1e-9);
        CHECK_NEAR(run.r.objective, 7.0, 1e-9);
        CHECK(run.r.y.size() == 1);
        CHECK_NEAR(run.r.y[0], 2.0, 1e-9);
        CHECK(run.r.y[0] >= 0.0);  // multiplier sign at a lower-active row
        CHECK(run.r.max_primal_violation <= 1e-9);
        CHECK(run.r.max_dual_violation <= 1e-9);
        CHECK(run.diag.dual_bound_finite);
        CHECK_NEAR(run.diag.dual_objective, 7.0, 1e-9);
        CHECK(run.diag.gap_rel <= 1e-9);
        CHECK(run.diag.presolve_forcing_rows_removed == 1);
        CHECK(run.diag.presolve_forcing_columns_fixed == 2);
        CHECK(run.diag.presolve_retries == 0);
        CHECK(run.diag.stages == 1);
    }

    // Minimum-activity forcing is symmetric: x-y <= -3 forces x=0,y=3.
    // The recovered upper-row multiplier is -2 in canonical minimization.
    {
        sor::model::LpProblem lp;
        lp.name = "FORCE_MIN";
        lp.A = sor::sparse::from_triplets(
            1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
        lp.c = {0.0, 2.0};
        lp.obj_offset = -1.0;
        lp.row_lo = {-sor::model::kInf};
        lp.row_hi = {-3.0};
        lp.col_lo = {0.0, 0.0};
        lp.col_hi = {2.0, 3.0};

        const auto run = solve_problem(lp);
        CHECK(run.r.status == Status::Optimal);
        CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
        CHECK_NEAR(run.r.x[0], 0.0, 1e-9);
        CHECK_NEAR(run.r.x[1], 3.0, 1e-9);
        CHECK_NEAR(run.r.objective, 5.0, 1e-9);
        CHECK(run.r.y.size() == 1);
        CHECK_NEAR(run.r.y[0], -2.0, 1e-9);
        CHECK(run.r.y[0] <= 0.0);  // multiplier sign at an upper-active row
        CHECK(run.r.max_primal_violation <= 1e-9);
        CHECK(run.r.max_dual_violation <= 1e-9);
        CHECK(run.diag.dual_bound_finite);
        CHECK_NEAR(run.diag.dual_objective, 5.0, 1e-9);
        CHECK(run.diag.gap_rel <= 1e-9);
        CHECK(run.diag.presolve_forcing_rows_removed == 1);
        CHECK(run.diag.presolve_forcing_columns_fixed == 2);
        CHECK(run.diag.presolve_retries == 0);
        CHECK(run.diag.stages == 1);
    }
}

void test_presolve_cross_kind_recovery_and_lifted_basis_statuses() {
    SOR_FN();
    // R0 forces upper-bound x=2,y=3. R1 then becomes empty, R2 fixes the
    // interior z=4 through a singleton equality, and R3 tightens w<=3 before
    // negative cost fixes it at that implied upper bound. Reverse dual recovery
    // must produce [+1,0,+1,-1]. The lifted basis is the vertex those duals
    // price: R0's multiplier is nonzero, so R0 is tight (nonbasic at its
    // lower side) and x, whose reduced cost fixes it (d_x = 0), is basic at
    // its forced upper bound; interior equality-fixed z is basic in R2; R3's
    // multiplier is nonzero, so R3 is tight at its upper side and w, at a
    // bound only R3 implies, is basic. Basic {x, s1, z, w} gives exactly
    // y = [+1,0,+1,-1].
    sor::model::LpProblem lp;
    lp.name = "FORCE_CASCADE";
    lp.A = sor::sparse::from_triplets(
        4, 4,
        {0, 0, 1, 1, 2, 2, 3, 3},
        {0, 1, 0, 1, 0, 2, 2, 3},
        {1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0});
    lp.c = {2.0, 0.0, 0.0, -1.0};
    lp.obj_offset = 5.0;
    lp.row_lo = {5.0, -1.0, 6.0, -sor::model::kInf};
    lp.row_hi = {sor::model::kInf, -1.0, 6.0, 7.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {2.0, 3.0, 10.0, 10.0};

    const auto run = solve_problem(lp);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(run.r.x.size() == 4);
    CHECK_NEAR(run.r.x[0], 2.0, 1e-9);
    CHECK_NEAR(run.r.x[1], 3.0, 1e-9);
    CHECK_NEAR(run.r.x[2], 4.0, 1e-9);
    CHECK_NEAR(run.r.x[3], 3.0, 1e-9);
    CHECK_NEAR(run.r.objective, 6.0, 1e-9);
    CHECK(run.r.y.size() == 4);
    CHECK_NEAR(run.r.y[0], 1.0, 1e-9);
    CHECK_NEAR(run.r.y[1], 0.0, 1e-9);
    CHECK_NEAR(run.r.y[2], 1.0, 1e-9);
    CHECK_NEAR(run.r.y[3], -1.0, 1e-9);
    CHECK(run.r.max_primal_violation <= 1e-9);
    CHECK(run.r.max_dual_violation <= 1e-9);
    CHECK(run.diag.dual_bound_finite);
    CHECK_NEAR(run.diag.dual_objective, 6.0, 1e-9);
    CHECK(run.diag.gap_rel <= 1e-9);
    CHECK(run.diag.presolve_forcing_rows_removed == 1);
    CHECK(run.diag.presolve_forcing_columns_fixed == 2);
    CHECK(run.diag.presolve_retries == 0);
    CHECK(run.diag.stages == 1);

    CHECK(run.basis.n_struct == 4);
    CHECK(run.basis.basic.size() == 4);
    CHECK(run.basis.status.size() == 8);
    CHECK(run.basis.basic[0] == 0);
    CHECK(run.basis.status[0] == NonbasicStatus::Basic);
    CHECK(run.basis.status[1] == NonbasicStatus::AtUpper);
    CHECK(run.basis.basic[2] == 2);
    CHECK(run.basis.status[2] == NonbasicStatus::Basic);
    CHECK(run.basis.status[3] == NonbasicStatus::Basic);
    CHECK(run.basis.status[4] == NonbasicStatus::AtLower);
    CHECK(run.basis.status[6] == NonbasicStatus::AtLower);
    CHECK(run.basis.status[7] == NonbasicStatus::AtUpper);
}

void test_presolve_equality_aggregation_chain_lifts_proof_and_basis() {
    SOR_FN();
    // Substitution chain:
    //   x + y = 3, x + z = 4  ->  z-y=1,
    // followed by z elimination.  The reduced problem is
    //   min 6+y+2w : y+w>=2, 0<=y<=2, 0<=w<=10,
    // whose unique optimum is y=2,w=0.  This exercises reverse primal and
    // dual journal replay, then reuses the lifted basis on the original model
    // so a structurally invalid aggregation lift cannot hide behind residuals.
    sor::model::LpProblem lp;
    lp.name = "EQUALITY_AGGREGATION_CHAIN";
    lp.A = sor::sparse::from_triplets(
        3, 4,
        {0, 0, 1, 1, 2, 2}, {0, 2, 0, 1, 1, 3},
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    lp.c = {1.0, 2.0, 0.0, 2.0};
    lp.obj_offset = 1.0;
    lp.row_lo = {3.0, 4.0, 3.0};
    lp.row_hi = {3.0, 4.0, sor::model::kInf};
    lp.col_lo = {-sor::model::kInf, -sor::model::kInf, 0.0, 0.0};
    lp.col_hi = {sor::model::kInf, sor::model::kInf, 2.0, 10.0};

    // Every equation here has an orientation whose bounds are exactly
    // implied, so live presolve leaves it to the kernel aggregation: with
    // live presolve off or on, the journal, the answer and the basis are
    // the same.
    for (const bool live : {false, true}) {
        ScopedEnvironment no_retry("SOR_PRESOLVE_NO_RETRY", "1");
        SimplexOptions opts;
        opts.method = sor::engines::SimplexMethod::Dual;
        opts.presolve = true;
        opts.presolve_live_reductions = live;
        const auto run = solve_problem(lp, opts);

        CHECK(run.r.status == Status::Optimal);
        CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
        CHECK(run.r.x.size() == 4);
        CHECK_NEAR(run.r.x[0], 1.0, 1e-9);
        CHECK_NEAR(run.r.x[1], 3.0, 1e-9);
        CHECK_NEAR(run.r.x[2], 2.0, 1e-9);
        CHECK_NEAR(run.r.x[3], 0.0, 1e-9);
        CHECK_NEAR(run.r.objective, 8.0, 1e-9);
        CHECK(run.r.y.size() == 3);
        CHECK_NEAR(run.r.y[0], 0.0, 1e-9);
        CHECK_NEAR(run.r.y[1], 1.0, 1e-9);
        CHECK_NEAR(run.r.y[2], 1.0, 1e-9);
        CHECK(run.r.max_primal_violation <= 1e-9);
        CHECK(run.r.max_dual_violation <= 1e-9);
        CHECK(run.diag.dual_bound_finite);
        CHECK_NEAR(run.diag.dual_objective, 8.0, 1e-9);
        CHECK(run.diag.gap_rel <= 1e-9);
        CHECK(run.diag.presolve_equality_aggregations == 2);
        CHECK(run.diag.presolve_retries == 0);
        CHECK(run.diag.stages == 1);

        CHECK(run.basis.n_struct == 4);
        CHECK(run.basis.basic.size() == 3);
        CHECK(run.basis.status.size() == 7);
        CHECK(run.basis.basic[0] == 0);
        CHECK(run.basis.basic[1] == 1);
        CHECK(run.basis.status[0] == NonbasicStatus::Basic);
        CHECK(run.basis.status[1] == NonbasicStatus::Basic);
        CHECK(run.basis.basic[0] != run.basis.basic[1]);
        CHECK(run.basis.basic[0] != run.basis.basic[2]);
        CHECK(run.basis.basic[1] != run.basis.basic[2]);

        // The original dual engine accepts only dimensionally and structurally
        // sane warm starts, and immediately factorizes their basis.  Re-solving
        // from the lifted basis validates that the two substituted columns really
        // form valid pivots for their restored equality rows.
        SimplexOptions warm_opts = opts;
        warm_opts.presolve = false;
        SimplexDiagnostics warm_diag;
        SimplexBasis warm_basis;
        auto warm_raw = sor::engines::solve_dual_simplex(
            lp, warm_opts, warm_diag, &warm_basis, &run.basis);
        const auto warm_ev = sor::engines::simplex_evidence(warm_diag, warm_opts);
        const auto warm_result =
            sor::certify::finalize_result(std::move(warm_raw), warm_ev);
        CHECK(warm_diag.warm_starts == 1);
        CHECK(warm_result.status == Status::Optimal);
        CHECK(warm_result.proof == ProofLevel::ProvedOptimalFP);
        CHECK_NEAR(warm_result.objective, 8.0, 1e-9);
        CHECK(warm_result.max_primal_violation <= 1e-9);
        CHECK(warm_result.max_dual_violation <= 1e-9);
    }
}

void test_singletons_before_aggregation_replay_stored_duals() {
    SOR_FN();
    // The first two equality rows eliminate singleton columns a and b.  Both
    // rows also contain s, so their substitutions change s's working cost
    // from 7 to 7-2-2*3=-1.  The later aggregation eliminates bounded but
    // implied-free s through s+x=5.  At the optimum s=3 is strictly inside
    // [0,10], so its stationarity equation exposes recovery-order errors.
    //
    // The historical bug recomputed the singleton multipliers from original A
    // while replaying this mixed journal.  The forward values 2 and 3 must be
    // replayed instead; otherwise the shared s column retains a dual residual.
    sor::model::LpProblem lp;
    lp.name = "SINGLETONS_BEFORE_AGGREGATION";
    lp.A = sor::sparse::from_triplets(
        4, 5,
        {0, 0, 1, 1, 2, 2, 3, 3},
        {0, 2, 1, 2, 2, 3, 2, 4},
        {1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0});
    lp.c = {2.0, 3.0, 7.0, 2.0, 3.0};
    lp.row_lo = {4.0, 7.0, 5.0, 4.0};
    lp.row_hi = {4.0, 7.0, 5.0, sor::model::kInf};
    lp.col_lo = {-sor::model::kInf, -sor::model::kInf, 0.0, 2.0, 0.0};
    lp.col_hi = {sor::model::kInf, sor::model::kInf, 10.0, 4.0, 10.0};

    const auto pmap = sor::presolve::presolve_lp(lp);
    CHECK(pmap.stats.singleton_columns_removed == 2);
    CHECK(pmap.stats.equality_aggregations == 1);
    CHECK(pmap.singleton_columns.size() == 2);
    CHECK(pmap.equality_aggregations.size() == 1);
    CHECK(pmap.recovery_steps.size() == 3);
    CHECK(pmap.recovery_steps[0].kind ==
          sor::presolve::DualRecoveryKind::SingletonColumnElimination);
    CHECK(pmap.recovery_steps[0].record == 0);
    CHECK(pmap.recovery_steps[1].kind ==
          sor::presolve::DualRecoveryKind::SingletonColumnElimination);
    CHECK(pmap.recovery_steps[1].record == 1);
    CHECK(pmap.recovery_steps[2].kind ==
          sor::presolve::DualRecoveryKind::EqualityAggregation);
    CHECK(pmap.recovery_steps[2].record == 0);
    CHECK(pmap.singleton_columns[0].other_cols == std::vector<Index>({2}));
    CHECK(pmap.singleton_columns[1].other_cols == std::vector<Index>({2}));
    CHECK_NEAR(pmap.singleton_columns[0].dual_value, 2.0, 1e-15);
    CHECK_NEAR(pmap.singleton_columns[1].dual_value, 3.0, 1e-15);
    CHECK(pmap.equality_aggregations[0].col == 2);
    CHECK_NEAR(pmap.equality_aggregations[0].dual_value, -1.0, 1e-15);

    // Every equation here has an orientation whose bounds are exactly
    // implied, so live presolve leaves it to the kernel aggregation: with
    // live presolve off or on, the journal, the answer and the basis are
    // the same.
    for (const bool live : {false, true}) {
        ScopedEnvironment no_retry("SOR_PRESOLVE_NO_RETRY", "1");
        SimplexOptions opts;
        opts.method = sor::engines::SimplexMethod::Dual;
        opts.presolve = true;
        opts.presolve_live_reductions = live;
        const auto run = solve_problem(lp, opts);
        CHECK(run.r.status == Status::Optimal);
        CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
        CHECK(run.r.x.size() == 5);
        CHECK_NEAR(run.r.x[0], 1.0, 1e-9);
        CHECK_NEAR(run.r.x[1], 1.0, 1e-9);
        CHECK_NEAR(run.r.x[2], 3.0, 1e-9);
        CHECK_NEAR(run.r.x[3], 2.0, 1e-9);
        CHECK_NEAR(run.r.x[4], 1.0, 1e-9);
        CHECK_NEAR(run.r.objective, 33.0, 1e-9);
        CHECK(run.r.y.size() == 4);
        CHECK_NEAR(run.r.y[0], 2.0, 1e-9);
        CHECK_NEAR(run.r.y[1], 3.0, 1e-9);
        CHECK_NEAR(run.r.y[2], -4.0, 1e-9);
        CHECK_NEAR(run.r.y[3], 3.0, 1e-9);
        CHECK(run.r.max_primal_violation <= 1e-9);
        CHECK(run.r.max_dual_violation <= 1e-9);
        CHECK(run.diag.dual_bound_finite);
        CHECK_NEAR(run.diag.dual_objective, 33.0, 1e-9);
        CHECK(run.diag.gap_rel <= 1e-9);
        CHECK(run.diag.presolve_singleton_columns_removed == 2);
        CHECK(run.diag.presolve_equality_aggregations == 1);
        CHECK(run.diag.presolve_retries == 0);
    }
}

void test_singleton_then_equality_fix_replays_stage_state() {
    SOR_FN();
    // Eliminating singleton a from a-2x=0 changes x's working cost from 1 to
    // 7.  Only then does the singleton equality x=3 fix x at an interior point
    // of its original [0,10] bounds.  Row 2 is still live at that instant and
    // contributes -y2 to x stationarity.  Recovery therefore needs the stored
    // stage equation 7 - (-1)y2 - y1 = 0, not original c_x in isolation.
    sor::model::LpProblem lp;
    lp.name = "SINGLETON_THEN_EQUALITY_FIX";
    lp.A = sor::sparse::from_triplets(
        3, 4,
        {0, 0, 1, 2, 2, 2}, {0, 1, 1, 1, 2, 3},
        {1.0, -2.0, 1.0, -1.0, 1.0, 1.0});
    lp.c = {3.0, 1.0, 1.0, 2.0};
    lp.row_lo = {0.0, 3.0, 0.0};
    lp.row_hi = {0.0, 3.0, sor::model::kInf};
    lp.col_lo = {-sor::model::kInf, 0.0, 0.0, 0.0};
    lp.col_hi = {sor::model::kInf, 10.0, 10.0, 10.0};

    const auto pmap = sor::presolve::presolve_lp(lp);
    CHECK(pmap.stats.singleton_columns_removed == 1);
    CHECK(pmap.recovery_steps.size() == 2);
    CHECK(pmap.recovery_steps[0].kind ==
          sor::presolve::DualRecoveryKind::SingletonColumnElimination);
    CHECK(pmap.recovery_steps[0].record == 0);
    CHECK(pmap.recovery_steps[1].kind ==
          sor::presolve::DualRecoveryKind::EqualitySingletonFix);
    const auto& fix = pmap.recovery_steps[1];
    CHECK(fix.row == 1);
    CHECK(fix.col == 1);
    CHECK_NEAR(fix.coeff, 1.0, 1e-15);
    CHECK_NEAR(fix.stage_cost, 7.0, 1e-15);
    CHECK(fix.other_rows == std::vector<Index>({2}));
    CHECK(fix.other_row_coefficients == std::vector<f64>({-1.0}));

    ScopedEnvironment no_retry("SOR_PRESOLVE_NO_RETRY", "1");
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Dual;
    opts.presolve = true;
    const auto run = solve_problem(lp, opts);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(run.r.x.size() == 4);
    CHECK_NEAR(run.r.x[0], 6.0, 1e-9);
    CHECK_NEAR(run.r.x[1], 3.0, 1e-9);
    CHECK_NEAR(run.r.x[2], 3.0, 1e-9);
    CHECK_NEAR(run.r.x[3], 0.0, 1e-9);
    CHECK_NEAR(run.r.objective, 24.0, 1e-9);
    CHECK(run.r.y.size() == 3);
    CHECK_NEAR(run.r.y[0], 3.0, 1e-9);
    CHECK_NEAR(run.r.y[1], 8.0, 1e-9);
    CHECK_NEAR(run.r.y[2], 1.0, 1e-9);
    CHECK(run.r.max_primal_violation <= 1e-9);
    CHECK(run.r.max_dual_violation <= 1e-9);
    CHECK(run.diag.dual_bound_finite);
    CHECK_NEAR(run.diag.dual_objective, 24.0, 1e-9);
    CHECK(run.diag.gap_rel <= 1e-9);
    CHECK(run.diag.presolve_singleton_columns_removed == 1);
    CHECK(run.diag.presolve_retries == 0);
}

void test_singleton_then_negative_bound_tightening_replays_stage_state() {
    SOR_FN();
    // The same first elimination changes x's working cost to 7, but now the
    // negative singleton inequality -x<=-3 tightens x's lower bound from 0 to
    // 3.  The optimum x=3 is interior to its ORIGINAL bounds, so the restored
    // row multiplier must be exactly -8 after including live row 2's -x term.
    // This simultaneously covers the negative-coefficient/sign branch.
    sor::model::LpProblem lp;
    lp.name = "SINGLETON_THEN_NEGATIVE_BOUND_TIGHTENING";
    lp.A = sor::sparse::from_triplets(
        3, 4,
        {0, 0, 1, 2, 2, 2}, {0, 1, 1, 1, 2, 3},
        {1.0, -2.0, -1.0, -1.0, 1.0, 1.0});
    lp.c = {3.0, 1.0, 1.0, 2.0};
    lp.row_lo = {0.0, -sor::model::kInf, 0.0};
    lp.row_hi = {0.0, -3.0, sor::model::kInf};
    lp.col_lo = {-sor::model::kInf, 0.0, 0.0, 0.0};
    lp.col_hi = {sor::model::kInf, 10.0, 10.0, 10.0};

    const auto pmap = sor::presolve::presolve_lp(lp);
    CHECK(pmap.stats.singleton_columns_removed == 1);
    CHECK(pmap.stats.bounds_tightened == 1);
    CHECK(pmap.recovery_steps.size() == 2);
    CHECK(pmap.recovery_steps[0].kind ==
          sor::presolve::DualRecoveryKind::SingletonColumnElimination);
    CHECK(pmap.recovery_steps[1].kind ==
          sor::presolve::DualRecoveryKind::BoundTightening);
    const auto& tightening = pmap.recovery_steps[1];
    CHECK(tightening.row == 1);
    CHECK(tightening.col == 1);
    CHECK_NEAR(tightening.coeff, -1.0, 1e-15);
    CHECK_NEAR(tightening.old_lo, 0.0, 1e-15);
    CHECK_NEAR(tightening.old_hi, 10.0, 1e-15);
    CHECK_NEAR(tightening.new_lo, 3.0, 1e-15);
    CHECK_NEAR(tightening.new_hi, 10.0, 1e-15);
    CHECK_NEAR(tightening.stage_cost, 7.0, 1e-15);
    CHECK(tightening.other_rows == std::vector<Index>({2}));
    CHECK(tightening.other_row_coefficients == std::vector<f64>({-1.0}));

    ScopedEnvironment no_retry("SOR_PRESOLVE_NO_RETRY", "1");
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Dual;
    opts.presolve = true;
    const auto run = solve_problem(lp, opts);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(run.r.x.size() == 4);
    CHECK_NEAR(run.r.x[0], 6.0, 1e-9);
    CHECK_NEAR(run.r.x[1], 3.0, 1e-9);
    CHECK_NEAR(run.r.x[2], 3.0, 1e-9);
    CHECK_NEAR(run.r.x[3], 0.0, 1e-9);
    CHECK_NEAR(run.r.objective, 24.0, 1e-9);
    CHECK(run.r.y.size() == 3);
    CHECK_NEAR(run.r.y[0], 3.0, 1e-9);
    CHECK_NEAR(run.r.y[1], -8.0, 1e-9);
    CHECK_NEAR(run.r.y[2], 1.0, 1e-9);
    CHECK(run.r.max_primal_violation <= 1e-9);
    CHECK(run.r.max_dual_violation <= 1e-9);
    CHECK(run.diag.dual_bound_finite);
    CHECK_NEAR(run.diag.dual_objective, 24.0, 1e-9);
    CHECK(run.diag.gap_rel <= 1e-9);
    CHECK(run.diag.presolve_singleton_columns_removed == 1);
    CHECK(run.diag.presolve_retries == 0);
}

void test_primal_crash_builds_a_feasible_triangular_basis() {
    SOR_FN();
    // Standalone CLI policy is applied by the app; internal callers (notably
    // MILP node LPs) retain the conservative library default.
    CHECK(!SimplexOptions{}.primal_crash);
    // x first repairs the violated lower row. Its cross-entry then violates
    // the upper row, where y can be selected because y is exactly zero in the
    // already claimed row. The selected structural block is [[1,0],[1,1]]:
    // triangular, nonsingular, and primal feasible without phase 1.
    sor::model::LpProblem lp;
    lp.name = "PRIMAL_CRASH_TRIANGULAR";
    lp.A = sor::sparse::from_triplets(
        2, 2, {0, 1, 1}, {0, 0, 1}, {1.0, 1.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {3.0, -sor::model::kInf};
    lp.row_hi = {sor::model::kInf, 1.0};
    lp.col_lo = {-sor::model::kInf, -sor::model::kInf};
    lp.col_hi = {sor::model::kInf, sor::model::kInf};

    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Primal;
    opts.presolve = false;
    opts.ruiz_iterations = 0;
    opts.primal_crash = true;
    const auto run = solve_problem(lp, opts);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(run.diag.primal_crash_columns == 2);
    CHECK_NEAR(run.diag.primal_crash_infeasibility_before, 3.0, 1e-12);
    CHECK_NEAR(run.diag.primal_crash_infeasibility_after, 0.0, 1e-12);
    CHECK(run.diag.phase1_iterations == 0);
    CHECK(run.diag.basis_repairs == 0);
    CHECK(run.basis.basic.size() == 2);
    CHECK(run.basis.basic[0] == 0);
    CHECK(run.basis.basic[1] == 1);
    CHECK(run.r.max_primal_violation <= 1e-9);
    CHECK(run.r.max_dual_violation <= 1e-9);
}

void test_primal_crash_rejects_a_net_harmful_pivot() {
    SOR_FN();
    // Making row 0 feasible through x would create 1000 units of violation in
    // row 1. The ordinary simplex can later use free y to repair it, but the
    // crash must not call a 10 -> 1000 move progress.
    sor::model::LpProblem lp;
    lp.name = "PRIMAL_CRASH_NET_HARM";
    lp.A = sor::sparse::from_triplets(
        2, 2, {0, 1, 1}, {0, 0, 1}, {1.0, 100.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {10.0, 0.0};
    lp.row_hi = {10.0, 0.0};
    lp.col_lo = {-sor::model::kInf, -sor::model::kInf};
    lp.col_hi = {sor::model::kInf, sor::model::kInf};

    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Primal;
    opts.presolve = false;
    opts.ruiz_iterations = 0;
    opts.primal_crash = true;
    const auto run = solve_problem(lp, opts);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(run.diag.primal_crash_columns == 0);
    CHECK_NEAR(run.diag.primal_crash_infeasibility_before, 10.0, 1e-12);
    CHECK_NEAR(run.diag.primal_crash_infeasibility_after, 10.0, 1e-12);
    CHECK(run.r.max_primal_violation <= 1e-9);
}

void test_primal_crash_rejects_columns_that_destroy_triangularity() {
    SOR_FN();
    // Row 0 is deliberately the larger violation, so x is installed there.
    // The remaining candidate y could repair row 1, but it has a nonzero in
    // the claimed row and would turn the structural block into an unverified
    // matching. The crash leaves that work to phase 1 instead.
    sor::model::LpProblem lp;
    lp.name = "PRIMAL_CRASH_CLAIMED_ROW";
    lp.A = sor::sparse::from_triplets(
        2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {1.0, 1.0, 1.0, 2.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {10.0, 3.0};
    lp.row_hi = {10.0, 3.0};
    lp.col_lo = {-sor::model::kInf, -sor::model::kInf};
    lp.col_hi = {sor::model::kInf, sor::model::kInf};

    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Primal;
    opts.presolve = false;
    opts.ruiz_iterations = 0;
    opts.primal_crash = true;
    const auto run = solve_problem(lp, opts);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(run.diag.primal_crash_columns == 1);
    CHECK_NEAR(run.diag.primal_crash_infeasibility_before, 13.0, 1e-12);
    CHECK(run.diag.primal_crash_infeasibility_after <
          run.diag.primal_crash_infeasibility_before);
    CHECK(run.diag.primal_crash_infeasibility_after > 0.0);
    CHECK(run.diag.basis_repairs == 0);
}

void test_primal_crash_rejects_an_unstable_or_out_of_bounds_pivot() {
    SOR_FN();
    // With scaling disabled, x's 1e-6 pivot is tiny relative to the row's
    // fixed unit coefficient. It would also need x=1e6. The crash rejects the
    // unstable diagonal; the proof-preserving ordinary iteration remains free
    // to solve the model. Tightening x's upper bound exercises the independent
    // out-of-bounds rejection on a second run.
    sor::model::LpProblem lp;
    lp.name = "PRIMAL_CRASH_UNSTABLE";
    lp.A = sor::sparse::from_triplets(
        1, 2, {0, 0}, {0, 1}, {1e-6, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {1.0};
    lp.row_hi = {1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {2e6, 0.0};

    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Primal;
    opts.presolve = false;
    opts.ruiz_iterations = 0;
    opts.primal_crash = true;
    const auto stable = solve_problem(lp, opts);
    CHECK(stable.r.status == Status::Optimal);
    CHECK(stable.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(stable.diag.primal_crash_columns == 0);
    CHECK_NEAR(stable.diag.primal_crash_infeasibility_before, 1.0, 1e-12);
    CHECK_NEAR(stable.diag.primal_crash_infeasibility_after, 1.0, 1e-12);

    lp.name = "PRIMAL_CRASH_OUT_OF_BOUNDS";
    lp.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {1.0});
    lp.c = {0.0};
    lp.col_lo = {0.0};
    lp.col_hi = {0.5};
    const auto bounded = solve_problem(lp, opts);
    CHECK(bounded.r.status == Status::Infeasible);
    CHECK(bounded.diag.primal_crash_columns == 0);
    CHECK_NEAR(bounded.diag.primal_crash_infeasibility_before, 1.0, 1e-12);
    CHECK_NEAR(bounded.diag.primal_crash_infeasibility_after, 1.0, 1e-12);
}

void test_simplex_consumes_terminal_presolve_outcome() {
    SOR_FN();
    sor::model::LpProblem lp;
    lp.name = "PRESOLVE_EMPTY_UNBOUNDED";
    lp.A = sor::sparse::from_triplets(1, 1, {}, {}, {});
    lp.c = {-1.0};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {sor::model::kInf};
    lp.col_lo = {0.0};
    lp.col_hi = {sor::model::kInf};

    SimplexOptions opts;
    opts.presolve = true;
    SimplexDiagnostics diag;
    SimplexBasis basis;
    const auto raw = sor::engines::solve_simplex(lp, opts, diag, &basis);
    CHECK(raw.proposed_status == Status::Unbounded);
    CHECK(!raw.primal_ray.direction.empty());
    const auto ev = sor::certify::check_lp_result(lp, raw, sor::engines::simplex_evidence(diag, opts));
    const auto result = sor::certify::finalize_result(raw, ev);
    CHECK(result.status == Status::Unbounded);
    CHECK(result.primal_ray.certified);
    CHECK(diag.status == Status::Unbounded);
}

}  // namespace

void test_residual_monitor_tracks_pivoted_dual_vector() {
    // Independent unit rows have exact pivots and no equation drift. Checking
    // every pivot must not refactor merely because the basic costs changed.
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(5, 5, {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4}, {1.0, 1.0, 1.0, 1.0, 1.0});
    lp.c = {-1.0, -2.0, -3.0, -4.0, -5.0};
    lp.col_lo.assign(5, 0.0);
    lp.col_hi.assign(5, sor::model::kInf);
    lp.row_lo.assign(5, -sor::model::kInf);
    lp.row_hi.assign(5, 1.0);
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Primal;
    opts.presolve = false;
    opts.primal_crash = false;
    opts.ruiz_iterations = 0;
    opts.residual_check_interval = 1;
    const auto run = solve_problem(lp, opts);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, -15.0, 1e-12);
    CHECK(run.diag.phase2_iterations >= 5);
    CHECK(run.diag.residual_refactors == 0);
}

void test_dual_bfrt_retains_selected_leaving_bound() {
    // The first boxed variable flips to 1. The remaining travel to the lower
    // row bound is below the primal tolerance, but the row's upper bound is
    // still 2. Parking on that opposite side corrupts the row equation.
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {1.0, 2.0};
    lp.col_lo = {0.0, 0.0}; lp.col_hi = {1.0, sor::model::kInf};
    lp.row_lo = {1.0 + 5e-7}; lp.row_hi = {2.0};
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Dual;
    opts.presolve = false; opts.ruiz_iterations = 0;
    opts.primal_feas_tol = 1e-6; opts.use_expand = false;
    opts.residual_check_interval = 1;
    const auto run = solve_problem(lp, opts);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, 1.0 + 1e-6, 1e-12);
    CHECK(run.diag.bound_flips >= 1);
    CHECK(run.diag.residual_refactors == 0);
    CHECK_NEAR(run.r.x[0], 1.0, 1e-12);
    CHECK_NEAR(run.r.x[1], 5e-7, 1e-12);
}

void test_unrepresentable_scaling_preserves_original_model() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {100.0});
    lp.c = {1.0}; lp.col_lo = {0.0}; lp.col_hi = {1e308};
    lp.row_lo = lp.row_hi = {1.0};
    for (bool maximize : {false, true}) {
        lp.maximize = maximize; lp.c[0] = maximize ? -1.0 : 1.0; lp.obj_offset = 5.0;
        for (const auto method : {sor::engines::SimplexMethod::Primal, sor::engines::SimplexMethod::Dual}) {
            SimplexOptions opts;
            opts.method = method; opts.presolve = false;
            opts.time_limit_s = 1.0;
            SimplexDiagnostics diag;
            auto raw = sor::engines::solve_simplex(lp, opts, diag);
            const auto checked = sor::certify::check_lp_result(lp, raw,
                sor::engines::simplex_evidence(diag, opts));
            const auto result = sor::certify::finalize_result(std::move(raw), checked);
            CHECK(result.status == Status::Optimal);
            CHECK_NEAR(result.x[0], 0.01, 1e-12);
            CHECK_NEAR(result.objective, maximize ? 4.99 : 5.01, 1e-12);
            CHECK(lp.col_hi[0] == 1e308);
            CHECK(lp.A.vals[0] == 100.0);
        }
    }
}

int main() {
    test_dual_bfrt_retains_selected_leaving_bound();
    test_residual_monitor_tracks_pivoted_dual_vector();
    test_unrepresentable_scaling_preserves_original_model();
    SOR_FN();
    test_empty_domain_is_infeasible();
    test_time_limit_covers_presolve_and_preparation();
    test_maximization_offset_is_reported_in_model_sense();
    test_fixture_lp();
    test_auto_commits_to_one_engine_without_a_discarded_probe();
    test_route_features_describe_the_model();
    test_dual_periodic_resync_is_not_tied_to_verbose();
    test_pruned_basic_pivotal_entries_match_full_path();
    test_pruned_fixed_pivotal_entries_match_retained_path();
    test_dual_cost_perturbation_cleans_before_optimality();
    test_dual_cleanup_hands_over_a_primal_feasible_basis();
    test_partitioned_price_matches_full_scan_on_random_bases();
    test_numerical_trouble_trigger_and_shift_bound();
    test_auto_candidate_order_uses_feasibility_before_gap();
    test_basis_wellformed();
    test_features_mps_agrees_with_model();
    test_infeasible();
    test_primal_phase1_exports_checked_farkas_certificate();
    test_infeasible_farkas_certificate();
    test_dual_cost_perturbation_cleans_before_farkas_proof();
    test_feasible_lp_has_no_ray();
    test_unbounded();
    test_dual_phase1_infeasible_falls_back_to_primal();
    test_equality_and_range();
    test_free_variable();
    test_dual_phase1_recovers_dual_infeasible_start();
    test_dual_leaving_variable_never_parks_on_an_infinite_bound();
    test_maximize();
    test_primal_phase1_pivot_rebuilds_phase2_reduced_costs();
    test_time_limit_is_honoured();
    test_degenerate();
    test_ill_conditioned();
    test_no_rows();
    test_presolve_certificate_fallback();
    test_presolve_singleton_column_elimination_lifts_proof_and_basis();
    test_presolve_chained_singleton_columns_recover_duals_and_basis();
    test_presolve_retry_counter_survives_rejected_retry();
    test_presolve_forcing_rows_lift_primal_dual_and_proof();
    test_presolve_cross_kind_recovery_and_lifted_basis_statuses();
    test_presolve_equality_aggregation_chain_lifts_proof_and_basis();
    test_singletons_before_aggregation_replay_stored_duals();
    test_singleton_then_equality_fix_replays_stage_state();
    test_singleton_then_negative_bound_tightening_replays_stage_state();
    test_primal_crash_builds_a_feasible_triangular_basis();
    test_primal_crash_rejects_a_net_harmful_pivot();
    test_primal_crash_rejects_columns_that_destroy_triangularity();
    test_primal_crash_rejects_an_unstable_or_out_of_bounds_pivot();
    test_simplex_consumes_terminal_presolve_outcome();
    return sor::test::finish("test_simplex");
}
