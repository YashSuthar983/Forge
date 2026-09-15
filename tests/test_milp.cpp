// Minimal MILP branch-and-bound: a tiny binary knapsack that must find the
// proven incumbent, and an infeasible integer program that must say so.
#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

using sor::core::Status;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

// max 5 x1 + 3 x2 + 2 x3
// s.t. 4 x1 + 2 x2 + x3 <= 5
//      x binary
// Optimum: x1=1, x2=0, x3=1, obj=7
const char* kKnapsack = R"(NAME          KNAP
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      -5             CAP       4
    X2        COST      -3             CAP       2
    X3        COST      -2             CAP       1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       5
BOUNDS
 UI BND       X1        1
 UI BND       X2        1
 UI BND       X3        1
ENDATA
)";

// Same integer knapsack with a fractional upper bound. For integer columns,
// CAP <= 5.2 is exactly equivalent to CAP <= 5 and must be tightened without
// changing the optimum.
const char* kFractionalKnapsack = R"(NAME          KNAPFRAC
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      -5             CAP       4
    X2        COST      -3             CAP       2
    X3        COST      -2             CAP       1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       5.2
BOUNDS
 UI BND       X1        1
 UI BND       X2        1
 UI BND       X3        1
ENDATA
)";

// x1 + x2 <= 1, x1 >= 1, x2 >= 1, binary — infeasible
const char* kInfeasMip = R"(NAME          INFEASMIP
ROWS
 N  COST
 L  R1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      1              R1        1
    X2        COST      1              R1        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        1
BOUNDS
 LO BND       X1        1
 UP BND       X1        1
 LO BND       X2        1
 UP BND       X2        1
ENDATA
)";

// The continuous slack S is integral in every feasible solution because the
// equality S + Y = 1 has an integer RHS and binary Y. The search may infer
// that domain, but the original model remains the certificate reference.
const char* kImpliedInteger = R"(NAME          IMPLIEDINT
ROWS
 N  COST
 E  LINK
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    Y         LINK       1
    MARK0001  'MARKER'                 'INTEND'
    S         COST       1             LINK       1
RHS
    RHS       LINK       1
BOUNDS
 BV BND       Y
 LO BND       S         0
ENDATA
)";

void test_knapsack() {
    auto lp = read_text(kKnapsack);
    CHECK(lp.n_integer() == 3);
    BabOptions opts;
    opts.max_nodes = 1000;
    opts.verbose = false;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Feasible || r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -7.0, 1e-6);  // minimize -value → -7
    CHECK(diag.nodes > 0);
}

void test_infeasible_mip() {
    auto lp = read_text(kInfeasMip);
    BabOptions opts;
    opts.max_nodes = 100;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Infeasible);
}

void test_integer_row_rounding() {
    auto lp = read_text(kFractionalKnapsack);
    BabOptions opts;
    opts.max_nodes = 1000;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(diag.integer_row_roundings >= 1);
    CHECK(r.status == Status::Optimal || r.status == Status::Feasible);
    CHECK_NEAR(r.objective, -7.0, 1e-6);
}

void test_implied_integer_slack() {
    auto lp = read_text(kImpliedInteger);
    BabOptions opts;
    opts.max_nodes = 100;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal || r.status == Status::Feasible);
    CHECK_NEAR(r.objective, 0.0, 1e-6);
}

void test_hybrid_node_selection_matches_best_bound() {
    // Node selection ordering must never change the proven answer -- it can
    // only change how many nodes it takes to get there. Cross-check hybrid
    // (best-bound + bounded plunging, item 16) against pure best-bound on
    // every existing fixture.
    for (const char* mps : {kKnapsack, kFractionalKnapsack, kImpliedInteger}) {
        auto lp = read_text(mps);

        BabOptions hybrid;
        hybrid.max_nodes = 1000;
        hybrid.hybrid_node_selection = true;
        BabDiagnostics diag_hybrid;
        auto raw_hybrid = sor::search::solve_milp(lp, hybrid, diag_hybrid);
        const auto r_hybrid = sor::certify::finalize_result(
            std::move(raw_hybrid), sor::search::milp_evidence(diag_hybrid, hybrid));

        BabOptions best_bound;
        best_bound.max_nodes = 1000;
        best_bound.hybrid_node_selection = false;
        BabDiagnostics diag_bb;
        auto raw_bb = sor::search::solve_milp(lp, best_bound, diag_bb);
        const auto r_bb = sor::certify::finalize_result(
            std::move(raw_bb), sor::search::milp_evidence(diag_bb, best_bound));

        CHECK(r_hybrid.status == r_bb.status);
        CHECK_NEAR(r_hybrid.objective, r_bb.objective, 1e-6);
        CHECK(diag_bb.plunge_nodes == 0);  // best-bound must never use the plunge path
    }
}

void test_hybrid_node_selection_infeasible_still_detected() {
    // Exercises the tree_exhausted/dual-bound-drain plumbing that had to
    // account for the plunge stack as well as the open queue -- if that
    // drain were incomplete, this could wrongly report Interrupted instead
    // of a proved Infeasible.
    auto lp = read_text(kInfeasMip);
    BabOptions opts;
    opts.max_nodes = 100;
    opts.hybrid_node_selection = true;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Infeasible);
}

// A search abandoned mid-node must not certify its incumbent as optimal.
//
// The global dual bound is seeded with the incumbent -- sound, because the
// EXPLORED part of the tree holds nothing better -- and then lowered by
// draining every open and plunged node. A node that was popped and then
// abandoned is in neither container, so without folding its bound in by hand
// the seed survives, the gap comes out 0, and an interrupted run reports
// ProvedGlobalEpsilon.
//
// Capping the node LP at one iteration reproduces that deterministically: the
// root LP comes back Interrupted with a point that is not primal feasible, the
// loop breaks with the root abandoned, and both containers are empty. Measured
// on `pg` at a 60 s limit before the fix: incumbent 7250 certified optimal
// against a published optimum of -8674.34.
std::string make_equality_knapsack(int n, int rhs) {
    std::string m = "NAME          EQKNAP\nROWS\n N  COST\n E  R1\nCOLUMNS\n";
    m += "    MARK0000  'MARKER'                 'INTORG'\n";
    for (int j = 0; j < n; ++j) {
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "    X%-9d COST      %d             R1        1\n",
                      j, (j % 7) + 1);
        m += buf;
    }
    m += "    MARK0001  'MARKER'                 'INTEND'\nRHS\n";
    char buf[80];
    std::snprintf(buf, sizeof buf, "    RHS       R1        %d\n", rhs);
    m += buf;
    m += "BOUNDS\n";
    for (int j = 0; j < n; ++j) {
        char b2[80];
        std::snprintf(b2, sizeof b2, " BV BND       X%d\n", j);
        m += b2;
    }
    m += "ENDATA\n";
    return m;
}

// A search abandoned mid-node must not certify its incumbent as optimal.
//
// The global dual bound is seeded with the incumbent -- sound, because the
// EXPLORED part of the tree holds nothing better -- and then lowered by
// draining every open and plunged node. A node that was POPPED and then
// abandoned is in neither container, so unless its bound is folded in by hand
// the seed survives every min, the gap comes out 0, and gap_proved turns an
// interrupted run into ProvedGlobalEpsilon.
//
// Capping the node LP at one iteration reproduces it deterministically: the
// root LP returns Interrupted with a point that is not primal feasible (x = 0
// violates the equality), the loop breaks with the root abandoned, and both
// containers are empty. Before the fix this reported the heuristic incumbent
// 121 as proved optimal; the true optimum, which the same model reaches once
// the LP has 4 iterations to work with, is 66.
//
// Found on `pg` from benchmarks/miplib-small at a 60 s limit: incumbent 7250
// certified ProvedGlobalEpsilon against a published optimum of -8674.34 and
// this solver's own feasible point at -8192.85.
void test_abandoned_node_is_not_a_proof() {
    const std::string mps = make_equality_knapsack(60, 30);
    BabOptions opts;
    // Isolate the abandoned-node proof regression from Latest defaults
    // (DynSep / Balans / mip-presolve / symmetry) that can finish or
    // re-label the root before the 1-iteration LP interrupt fires.
    opts.policy = sor::search::MilpPolicy::Classical;
    opts.max_nodes = 1000;
    opts.lp.max_iterations = 1;
    opts.lp.presolve = false;
    opts.cuts_enabled = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.probing = false;
    opts.feasibility_jump = true;  // still need an incumbent seed
    opts.sub_mip_lns = false;
    opts.balans.enabled = false;
    opts.kernel_pump.enabled = false;
    opts.mrens.enabled = false;
    opts.dynsep.enabled = false;
    auto lp = read_text(mps);
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);

    CHECK(diag.termination_reason == "node LP interrupted");
    CHECK(diag.nodes == 1);                 // the root was popped, then dropped
    CHECK(std::isfinite(diag.incumbent));   // ...with a heuristic incumbent
    CHECK(!diag.globally_proved);
    CHECK(r.status != Status::Optimal);
    CHECK(r.proof != sor::core::ProofLevel::ProvedGlobalEpsilon);

    // And the incumbent it did not prove is genuinely not the optimum: the
    // same model, with an LP budget that lets the search finish, does better.
    BabOptions full;
    full.policy = sor::search::MilpPolicy::Classical;
    full.max_nodes = 1000;
    full.lp.presolve = false;
    full.cuts_enabled = false;
    full.mip_presolve = false;
    full.symmetry = false;
    full.balans.enabled = false;
    full.kernel_pump.enabled = false;
    full.mrens.enabled = false;
    full.dynsep.enabled = false;
    BabDiagnostics fdiag;
    auto fraw = sor::search::solve_milp(read_text(mps), full, fdiag);
    const auto fr = sor::certify::finalize_result(
        std::move(fraw), sor::search::milp_evidence(fdiag, full));
    CHECK(fdiag.globally_proved);
    CHECK(fr.objective < diag.incumbent - 1e-6);
}

// The other half: a legitimate early exit must still be a proof. The gap
// tolerance exists precisely so a search can stop before exhausting the tree,
// so the fix above must not be "drop the incumbent seed unless exhausted".
void test_gap_closed_before_exhaustion_still_proves() {
    auto lp = read_text(kKnapsack);
    BabOptions opts;
    opts.max_nodes = 1000;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal);
    CHECK(diag.globally_proved);
    CHECK_NEAR(r.objective, -7.0, 1e-6);
}

}  // namespace

int main() {
    test_knapsack();
    test_infeasible_mip();
    test_integer_row_rounding();
    test_implied_integer_slack();
    test_hybrid_node_selection_matches_best_bound();
    test_hybrid_node_selection_infeasible_still_detected();
    test_abandoned_node_is_not_a_proof();
    test_gap_closed_before_exhaustion_still_proves();
    return sor::test::finish("test_milp");
}
