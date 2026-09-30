// Minimal MILP branch-and-bound: a tiny binary knapsack that must find the
// proven incumbent, and an infeasible integer program that must say so.
#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

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

// The cut loop proves this root LP and adds no rows. Search should consume
// that exact result instead of solving the same LP a second time.
const char* kRootReuse = R"(NAME          ROOTREUSE
ROWS
 N  COST
 G  NEED
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X         COST      1              NEED       1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       NEED      1
BOUNDS
 BV BND       X
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

// x1 + x2 <= 1, x1 >= 1, x2 >= 1, binary - infeasible
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

// x + 5e-10*y = 1; x binary, y integer in [0,2e9]; min x.
// The true optimum is x=0, y=2e9. Historically probing discarded that side
// and zero-half emitted an invalid x>=1 cut, producing false Optimal 1.
const char* kSmallCoefficientProof = R"(NAME          SMALLCOEF
ROWS
 N  OBJ
 E  EQ1
COLUMNS
    X         OBJ       1
    X         EQ1       1
    Y         EQ1       5e-10
RHS
    RHS1      EQ1       1
BOUNDS
 BV BND1      X
 LI BND1      Y         0
 UI BND1      Y         2000000000
ENDATA
)";

void test_small_coefficient_optimum_is_not_cut_off() {
    auto lp = read_text(kSmallCoefficientProof);
    CHECK(lp.n_integer() == 2);
    BabOptions opts;
    opts.max_nodes = 100;
    opts.time_limit_s = 5.0;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, 0.0, 1e-9);
    CHECK(r.x.size() == 2);
    if (r.x.size() == 2) {
        CHECK_NEAR(r.x[0], 0.0, 1e-9);
        CHECK_NEAR(r.x[1], 2e9, 1e-3);
        CHECK(lp.max_row_violation(r.x) <= opts.primal_feas_tol);
    }
}

void test_knapsack() {
    auto lp = read_text(kKnapsack);
    CHECK(lp.n_integer() == 3);
    BabOptions opts;
    opts.max_nodes = 1000;
    opts.verbose = false;
    // This test is about the branch-and-bound tree; the root primal pass
    // would settle such a tiny model before any node.
    opts.root_primal_early = false;
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

// Symmetry folding merges the identical columns X1 and X3 into X1 := X1+X3
// and fixes X3 = 0. Probing ran first and recorded X0 <= 1 + X3 (valid on the
// original columns); applied after folding it reads X0 <= 1 and cut off the
// optimum X = (2,1,0,1,1), objective -9. The solver claimed Optimal -6.
const char* kFoldedImpliedBound = R"(NAME          FOLDIB
ROWS
 N  OBJ
 L  R1
 L  R2
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X0        OBJ       -4             R1        -1
    X0        R2        3
    X1        OBJ       1              R1        -1
    X1        R2        -4
    X2        OBJ       5              R1        -1
    X2        R2        4
    X3        OBJ       1              R1        -1
    X3        R2        -4
    X4        OBJ       -3             R1        -1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        -1             R2        1
BOUNDS
 UP BND       X0        2
 UP BND       X1        1
 UP BND       X2        1
 UP BND       X3        1
 UP BND       X4        1
ENDATA
)";

void test_folding_invalidates_probed_implied_bounds() {
    auto lp = read_text(kFoldedImpliedBound);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // reach probing + folding
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -9.0, 1e-9);
}

void test_integer_row_rounding() {
    auto lp = read_text(kFractionalKnapsack);
    BabOptions opts;
    // Structural presolve rounds these sides itself (Achterberg Alg. 10.1
    // step 1b); keep the model unreduced to exercise the B&B's own pass.
    opts.structural_presolve.enabled = false;
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
    // The root is popped, set aside, retried once (a second pop, same 1-pivot
    // budget, same failure) and only then folded into the bound.
    CHECK(diag.nodes >= 1 && diag.nodes <= 2);
    CHECK(diag.node_lp_deferred == 1 && diag.abandoned_unproved_nodes == 1);
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

void test_objective_lattice_requires_exact_costs() {
    auto lp = read_text(kKnapsack);
    BabOptions opts;
    opts.max_nodes = 1000;

    BabDiagnostics exact_diag;
    sor::search::solve_milp(lp, opts, exact_diag);
    CHECK_NEAR(exact_diag.objective_granularity, 1.0, 1e-12);

    // This difference is inside the former 1e-6 integrality tolerance, but
    // the objective is no longer confined to integer values.
    lp.c[0] = -5.00000005;
    BabDiagnostics near_diag;
    sor::search::solve_milp(lp, opts, near_diag);
    CHECK_NEAR(near_diag.objective_granularity, 0.0, 1e-12);

    // A tiny nonzero cost is still a nonzero cost, irrespective of tolerance.
    lp.c[0] = -5.0;
    lp.c[1] = 1e-8;
    BabDiagnostics tiny_diag;
    sor::search::solve_milp(lp, opts, tiny_diag);
    CHECK_NEAR(tiny_diag.objective_granularity, 0.0, 1e-12);
}

void test_node_cutoff_values() {
    struct Case { double inc, g, offset, gap, expected; };
    const Case cases[] = {
        {5.0, 1.0, 0.0, 1e-4, 4.9995},
        {5.0, 1.0, 0.0, 0.5, 4.000005},
        {5.0000001, 1.0, 0.0, 0.5, 4.0 + 5.0000001e-6},
        {5.3, 1.0, 0.0, 0.5, 5.0 + 5.3e-6},
        {100.0, 0.0, 0.0, 1e-4, 99.99},
        {5.5, 1.0, 0.5, 0.5, 4.5 + 5.5e-6},
        {0.0, 0.0, 0.0, 1e-4, -1e-6},
    };
    for (const auto& t : cases) {
        const double c = sor::search::node_cutoff(t.inc, t.g, t.offset, t.gap, 1e-6);
        CHECK(std::fabs(c - t.expected) <= 1e-12);
        CHECK(c <= t.inc);
        if (t.g > 0.0) {
            double b = -std::numeric_limits<double>::infinity();
            for (int j = -10; j <= 10; ++j) {
                const double point = t.offset + j * t.g;
                if (point < t.inc - 1e-6 * std::max(1.0, std::fabs(t.inc)))
                    b = std::max(b, point);
            }
            CHECK(std::isfinite(b));
            CHECK(b < c);
        }
    }
}

sor::model::LpProblem gap_knapsack_model(unsigned seed, int n) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> value(1000, 9999), weight(10, 99);
    sor::model::LpProblem lp;
    lp.maximize = true;
    lp.c.resize(n);
    lp.col_lo.assign(n, 0.0);
    lp.col_hi.assign(n, 1.0);
    lp.is_integer.assign(n, true);
    for (int j = 0; j < n; ++j) lp.c[j] = value(rng) / 1000.0;
    std::vector<sor::core::Index> rows, cols;
    std::vector<double> vals;
    lp.row_lo.assign(5, -sor::model::kInf);
    lp.row_hi.assign(5, 0.0);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < n; ++j) {
            const double w = weight(rng);
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(w);
            lp.row_hi[i] += 0.5 * w;
        }
    }
    lp.A = sor::sparse::from_triplets(5, n, rows, cols, vals);
    return lp;
}

double gap_knapsack_optimum(const sor::model::LpProblem& lp) {
    std::vector<std::array<double, 5>> weight(lp.n_cols());
    for (int i = 0; i < 5; ++i) {
        for (auto k = lp.A.pattern.row_ptr()[i]; k < lp.A.pattern.row_ptr()[i + 1]; ++k)
            weight[lp.A.pattern.col_idx()[k]][i] = lp.A.vals[k];
    }
    std::array<double, 5> used{};
    double best = 0.0;
    std::function<void(int, double)> dfs = [&](int j, double obj) {
        if (j == lp.n_cols()) {
            best = std::max(best, obj);
            return;
        }
        bool fits = true;
        for (int i = 0; i < 5; ++i)
            if (used[i] + weight[j][i] > lp.row_hi[i]) fits = false;
        if (fits) {
            for (int i = 0; i < 5; ++i) used[i] += weight[j][i];
            dfs(j + 1, obj + lp.c[j]);
            for (int i = 0; i < 5; ++i) used[i] -= weight[j][i];
        }
        dfs(j + 1, obj);
    };
    dfs(0, 0.0);
    return best;
}

BabOptions gap_knapsack_options(double gap, int threads) {
    BabOptions opts;
    opts.cuts_enabled = false;
    opts.mip_presolve = false;
    opts.probing = false;
    opts.symmetry = false;
    opts.integer_row_rounding = false;
    opts.structural_presolve.enabled = false;
    opts.sub_mip_lns = false;
    opts.balans.enabled = false;
    opts.kernel_pump.enabled = false;
    opts.mrens.enabled = false;
    opts.feasibility_jump = true;
    opts.rounding_heuristic = true;
    opts.abs_gap_tol = 1e-6;
    opts.gap_tol = gap;
    opts.para_bab.threads = threads;
    if (threads > 1) {
        opts.para_bab_cost.adaptive = false;
        opts.plunge_max_depth = 0;
    }
    return opts;
}

// Calibrated over sizes {16,18,20} and gaps {0.005,0.01,0.02,0.05}.
// The smallest qualifying pair is 16 items and gap 0.005 (30 pruning seeds).
void test_gap_pruning_reports_honest_dual_bound(int threads) {
    constexpr int n = 16;
    constexpr double gap = 0.005;
    std::uint64_t prunes = 0, worker_prunes = 0, expansions = 0;
    int deep_seeds = 0, pruning_seeds = 0;
    const auto begin = std::chrono::steady_clock::now();
    for (unsigned seed = 1; seed <= 30; ++seed) {
        const auto lp = gap_knapsack_model(seed, n);
        const double optimum = gap_knapsack_optimum(lp);
        auto opts = gap_knapsack_options(gap, threads);
        BabDiagnostics diag;
        auto raw = sor::search::solve_milp(lp, opts, diag);
        const auto result = sor::certify::finalize_result(std::move(raw),
                            sor::search::milp_evidence(diag, opts));
        CHECK(result.status == Status::Optimal || result.status == Status::Feasible);
        CHECK(result.x.size() == lp.c.size());
        if (result.x.size() == lp.c.size()) {
            CHECK(lp.max_row_violation(result.x) <= opts.primal_feas_tol);
            CHECK(lp.max_bound_violation(result.x) <= opts.primal_feas_tol);
            for (double x : result.x) CHECK(std::fabs(x - std::round(x)) <= opts.int_tol);
        }
        CHECK(diag.objective_granularity == 0.0);
        CHECK(result.objective >= optimum - gap * std::fabs(optimum) - 1e-6);
        CHECK(diag.dual_bound >= optimum - 1e-6);
        if (diag.gap_prunes > 0) {
            CHECK(diag.gap_pruned_integral_lp <= diag.gap_prunes);
            CHECK(std::isfinite(diag.gap_pruned_floor));
            CHECK(diag.dual_bound >= diag.gap_pruned_floor - 1e-9);
        }
        prunes += diag.gap_prunes;
        worker_prunes += diag.para_gap_prunes;
        expansions += diag.para_bab.parallel_expansions;
        deep_seeds += diag.nodes > 10;
        pruning_seeds += diag.gap_prunes > 0;
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    std::printf("R3v2 threads=%d gap_prunes=%llu worker_gap_prunes=%llu expansions=%llu pruning_seeds=%d deep_seeds=%d seconds=%.3f\n",
                threads, static_cast<unsigned long long>(prunes),
                static_cast<unsigned long long>(worker_prunes),
                static_cast<unsigned long long>(expansions), pruning_seeds, deep_seeds, elapsed);
    CHECK(prunes > 0);
    if (threads == 1) { CHECK(deep_seeds >= 5); CHECK(pruning_seeds >= 10); }
    else { CHECK(expansions > 0); CHECK(worker_prunes > 0); }
}

void test_integral_lp_points_reach_incumbent_validation_before_gap_pruning() {
    sor::io::MpsReadReport report;
    const auto lp = sor::io::read_mps_file(
        std::string(SOR_SOURCE_DIR) + "/benchmarks/miplib-easy/mps/rgn.mps", report);
    BabOptions opts;
    opts.time_limit_s = 20.0;
    opts.para_bab.threads = 1;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(raw.proposed_status == Status::Optimal);
    CHECK_NEAR(raw.objective, 82.19999924, 1e-6);
    CHECK(diag.gap_pruned_integral_lp == 0);
    CHECK(sor::search::milp_point_max_violation(lp, raw.x, opts.int_tol) <=
          opts.primal_feas_tol);
}

void test_enigma_stops_when_the_global_gap_closes() {
    sor::io::MpsReadReport report;
    const auto lp = sor::io::read_mps_file(
        std::string(SOR_SOURCE_DIR) + "/benchmarks/miplib-easy/mps/enigma.mps",
        report);
    BabOptions opts;
    opts.time_limit_s = 20.0;
    opts.para_bab.threads = 1;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(raw.proposed_status == Status::Optimal);
    CHECK(diag.globally_proved);
    CHECK_NEAR(raw.objective, 0.0, 1e-7);
    // enigma has objective 0: every pseudocost and strong-branching gain is
    // zero, so reliability branching (the default since 2026-09-25) ranks
    // candidates only by the section 5.9 inference/cutoff/conflict terms and
    // needs ~3.5k nodes where the old fractionality-degree score needed 225.
    // The property under test is the gap-closed stop, not the node count;
    // the cap only guards against a runaway search.
    CHECK(diag.nodes < 20000);
    CHECK(diag.termination_reason == "global gap closed");
}

void test_root_neighborhood_obeys_milp_deadline() {
    // n5-3 reaches the root rounding heuristic. Its first local-neighborhood
    // call formerly consumed the full three-second default after a one-second
    // solve limit, yielding more than four seconds total on this fixture.
    sor::io::MpsReadReport report;
    const auto lp = sor::io::read_mps_file(
        std::string(SOR_SOURCE_DIR) + "/benchmarks/miplib-easy/mps/n5-3.mps",
        report);
    BabOptions opts;
    opts.time_limit_s = 1.0;
    opts.para_bab.threads = 1;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.total_ms < 3000.0);
}

// The report's "heuristics: X ms total" line and the budget ceiling both read
// heuristic_spent_ms(). It must be the sum of the disjoint heuristic timers;
// a stub returning 0 hid 40 s of in-tree rounding on drayage-25-23.
void test_heuristic_spent_ms_sums_heuristic_timers() {
    sor::io::MpsReadReport report;
    const auto lp = sor::io::read_mps_file(
        std::string(SOR_SOURCE_DIR) + "/benchmarks/miplib-easy/mps/n5-3.mps",
        report);
    BabOptions opts;
    // Node-bounded, not time-bounded: under a loaded CTest a 1 s limit
    // can end before the first heuristic round.
    opts.time_limit_s = 30.0;
    opts.max_nodes = 20;
    opts.para_bab.threads = 1;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    const double parts = diag.heuristic_ms + diag.feasjump_ms +
                         diag.fixprop_ms + diag.sub_mip_ms;
    CHECK(diag.heuristic_ms > 0.0);
    CHECK(parts > 0.0);
    CHECK_NEAR(sor::search::heuristic_spent_ms(diag), parts, 1e-9);
}

// Each node-rounding round tries at most four direct roundings (objective,
// nearest, ceil, and the lower-corner retry). A fallback that repeated the
// first three with identical arguments could never succeed and doubled the
// cost of every failed round (40 s of 60 s on drayage-25-23).
void test_node_rounding_does_not_repeat_identical_attempts() {
    sor::io::MpsReadReport report;
    const auto lp = sor::io::read_mps_file(
        std::string(SOR_SOURCE_DIR) + "/benchmarks/miplib-easy/mps/n5-3.mps",
        report);
    BabOptions opts;
    // Node-bounded, not time-bounded: under a loaded CTest a 1 s limit
    // can end before the first heuristic round.
    opts.time_limit_s = 30.0;
    opts.max_nodes = 20;
    opts.para_bab.threads = 1;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.node_rounding_rounds > 0);
    CHECK(diag.rounding_calls <= 4 * diag.node_rounding_rounds);
}

// Direct node rounding backs off after rounds that find no improving point,
// so a model whose LP points keep rounding to infeasible or worse points does
// not pay for four repair searches at every node (it ran at 400 of 400 n5-3
// nodes before the backoff).
void test_node_rounding_backs_off_after_misses() {
    sor::io::MpsReadReport report;
    const auto lp = sor::io::read_mps_file(
        std::string(SOR_SOURCE_DIR) + "/benchmarks/miplib-easy/mps/n5-3.mps",
        report);
    BabOptions opts;
    opts.time_limit_s = 60.0;
    opts.max_nodes = 200;
    opts.para_bab.threads = 1;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.nodes >= 100);
    CHECK(diag.node_rounding_rounds > 0);
    CHECK(2 * diag.node_rounding_rounds < diag.nodes);
}

// max x s.t. 3x <= 2.9999985, x integer in [0, 5]. The LP optimum
// x = 0.9999995 is within int_tol (1e-6) of 1, so the search accepts it as
// an integer point -- but x = 1 violates the row by 1.5e-6 > 1e-7 and the
// true optimum is x = 0. Whatever the search does, a reported Optimal must be
// integral and feasible at primal_feas_tol on the original model.
void test_reported_optimal_is_integral_at_feasibility_tolerance() {
    sor::model::LpProblem lp;
    lp.name = "near-integral";
    lp.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {3.0});
    lp.c = {1.0};
    lp.maximize = true;
    lp.col_lo = {0.0};
    lp.col_hi = {5.0};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {2.9999985};
    lp.is_integer = {true};
    BabOptions opts;
    opts.max_nodes = 100;
    opts.para_bab.threads = 1;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto r = sor::certify::finalize_result(
        std::move(raw), sor::search::milp_evidence(diag, opts));
    if (r.status == Status::Optimal) {
        CHECK(r.x.size() == 1);
        CHECK(std::fabs(r.x[0] - std::round(r.x[0])) <= 1e-7);
        CHECK(lp.max_row_violation(r.x) <= 1e-7);
        CHECK_NEAR(r.objective, 0.0, 1e-9);
    }
}

void test_ambiguous_node_status_requires_bounded_root() {
    CHECK(!sor::search::node_lp_status_proves_infeasible(
        Status::InfeasibleOrUnbounded, false));
    CHECK(sor::search::node_lp_status_proves_infeasible(
        Status::InfeasibleOrUnbounded, true));
    CHECK(sor::search::node_lp_status_proves_infeasible(
        Status::Infeasible, false));
    CHECK(!sor::search::node_lp_status_proves_infeasible(
        Status::Interrupted, true));
}

void test_node_infeasibility_needs_a_checked_ray() {
    auto lp = read_text(kKnapsack);
    sor::core::RawResult raw;
    raw.proposed_status = Status::Infeasible;
    CHECK(!sor::search::node_lp_infeasibility_proved(lp, raw, 1e-7, true));

    // The row requires at least 8, while bounded binary columns can supply
    // at most 4+2+1=7. A negative row multiplier separates those boxes.
    lp.row_lo[0] = 8.0;
    lp.row_hi[0] = sor::model::kInf;
    raw.ray = {-1.0};
    CHECK(sor::search::node_lp_infeasibility_proved(lp, raw, 1e-7, true));
    CHECK(sor::search::node_lp_infeasibility_proved(lp, raw, 1e-7, false));

    // A claimed ray is tied to this node's bounds. Widening one column makes
    // the LP feasible and must invalidate the same certificate.
    lp.col_hi[0] = 2.0;
    CHECK(!sor::search::node_lp_infeasibility_proved(lp, raw, 1e-7, true));
    raw.ray.clear();
    raw.proposed_status = Status::InfeasibleOrUnbounded;
    CHECK(!sor::search::node_lp_infeasibility_proved(lp, raw, 1e-7, true));
}

void test_proved_cut_loop_root_is_reused_only_at_identical_bounds(bool trace = false) {
    auto lp = read_text(kRootReuse);
    BabOptions opts;
    opts.time_limit_s = 5.0;
    opts.verbose = trace;
    opts.trace_branching = trace ? 4 : 0;
    opts.max_nodes = 10;
    opts.cuts_enabled = true;
    opts.cut.max_rounds = 1;
    opts.auto_cuts = false;
    opts.mip_presolve = false;
    opts.probing = false;
    opts.symmetry = false;
    opts.domain_propagation = false;
    opts.conflict_propagation = false;
    opts.feasibility_jump = false;
    opts.fixprop = false;
    opts.sub_mip_lns = false;
    opts.rounding_heuristic = false;
    opts.lp_rounding_repair = false;
    opts.integer_dive = false;
    opts.integer_neighborhood = false;
    opts.lp.presolve = false;
    opts.root_primal_early = false;   // the root LP hand-off is what is measured
    BabDiagnostics d;
    const auto result = sor::search::solve_milp(lp, opts, d);
    CHECK(result.proposed_status == Status::Optimal);
    CHECK(d.root_lp_reuses == 1);
    CHECK(d.root_lp_session_handoffs == 1);
    CHECK(d.lp_solves == 1);

    // Row propagation tightens X from [0,1] to [1,1]. The old root LP
    // certificate belongs to a different bound box and cannot be replayed.
    opts.domain_propagation = true;
    BabDiagnostics changed;
    const auto with_propagation =
        sor::search::solve_milp(lp, opts, changed);
    CHECK(with_propagation.proposed_status == Status::Optimal);
    CHECK(changed.root_lp_reuses == 0);
    CHECK(changed.root_lp_session_handoffs == 1);
    CHECK(changed.lp_session_solves == 1);
    CHECK(changed.lp_session_builds_prepared == 0);

    // A connected odd-cycle relaxation stays fractional after valid root
    // row/cover preprocessing. Exhaustive binary enumeration gives optimum -1.
    sor::model::LpProblem fractional;
    fractional.A = sor::sparse::from_triplets(3, 3, {0, 0, 1, 1, 2, 2},
        {0, 1, 1, 2, 0, 2}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    fractional.c = {-1.0, -1.0, -1.0};
    fractional.col_lo = {0.0, 0.0, 0.0};
    fractional.col_hi = {1.0, 1.0, 1.0};
    fractional.row_lo.assign(3, -sor::model::kInf);
    fractional.row_hi = {1.0, 1.0, 1.0};
    fractional.is_integer = {true, true, true};
    opts.domain_propagation = false;
    opts.root_primal_final = false;
    opts.para_bab.threads = 1;
    opts.conflict_cut.enabled = false;
    opts.conflict_cut.nogood_cuts = false;
    opts.reduced_cost_strengthening = false;
    BabDiagnostics probes;
    const auto branched = sor::search::solve_milp(fractional, opts, probes);
    CHECK(branched.proposed_status == Status::Optimal);
    CHECK_NEAR(branched.objective, -1.0, 1e-9);
    CHECK(fractional.max_row_violation(branched.x) <= 1e-7);
    CHECK(fractional.max_bound_violation(branched.x) <= 1e-7);
    CHECK(probes.root_lp_reuses == 1);
    CHECK(probes.root_lp_session_handoffs == 1);
    CHECK(probes.strong_branch_solves > 0);
    CHECK(probes.strong_branch_factor_reuses > 0);
    CHECK(probes.strong_branch_prep_ms == 0.0);
    CHECK(probes.lp_session_builds_prepared == 0);
}

}  // namespace

void test_incumbent_point_validation() {
    sor::model::LpProblem p;
    p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    p.c = {1.0, 2.0}; p.col_lo = {0.0, 0.0}; p.col_hi = {1.0, 2.0};
    p.row_lo = {0.0}; p.row_hi = {2.0}; p.is_integer = {true, false};
    CHECK(sor::search::milp_point_max_violation(p, {1.0, 0.5}, 1e-6) == 0.0);
    CHECK(std::isinf(sor::search::milp_point_max_violation(p, {0.5, 0.5}, 1e-6)));
    CHECK(sor::search::milp_point_max_violation(p, {1.0, 1.5}, 1e-6) > 1e-7);
    CHECK(sor::search::milp_point_max_violation(p, {2.0, 0.0}, 1e-6) > 1e-7);
    CHECK(sor::search::milp_point_max_violation(p, {1.0, -0.25}, 1e-6) > 1e-7);
    CHECK(std::isinf(sor::search::milp_point_max_violation(p, {1.0}, 1e-6)));
    CHECK(std::isinf(sor::search::milp_point_max_violation(p,
        {1.0, std::numeric_limits<double>::quiet_NaN()}, 1e-6)));
}

void test_folded_incumbents_stay_in_original_space() {
    for (double rhs : {4.0, 5.0}) {
        sor::model::LpProblem lp;
        lp.A = sor::sparse::from_triplets(1, 5, {0, 0, 0, 0, 0}, {0, 1, 2, 3, 4}, {2, 2, 2, 2, 2});
        lp.c.assign(5, -1.0); lp.col_lo.assign(5, 0.0); lp.col_hi.assign(5, 1.0);
        lp.is_integer.assign(5, true); lp.row_lo = {-sor::model::kInf}; lp.row_hi = {rhs};
        sor::search::BabOptions o;
        o.time_limit_s = 5; o.para_bab.threads = 1;
        o.structural_presolve.enabled = false; o.cuts_enabled = false;
        o.mip_presolve = false; o.probing = false; o.integer_row_rounding = false;
        o.feasibility_jump = false; o.fixprop = false;
        o.sub_mip_lns = false; o.balans.enabled = false; o.kernel_pump.enabled = false;
        o.mrens.enabled = false; o.integer_dive = false;
        o.sym.orbital_fixing = false; o.sym.reflection = false;
        sor::search::BabDiagnostics d;
        auto raw = sor::search::solve_milp(lp, o, d);
        CHECK(d.symmetry_diag.folding_applied);
        CHECK(raw.x.size() == 5);
        CHECK(lp.max_row_violation(raw.x) <= o.primal_feas_tol);
        CHECK(lp.max_bound_violation(raw.x) <= o.primal_feas_tol);
        CHECK_NEAR(lp.objective(raw.x), raw.objective, 1e-7);
        double optimum = 0.0;
        for (unsigned mask = 0; mask < 32; ++mask) {
            std::vector<double> point(5, 0.0);
            for (unsigned j = 0; j < 5; ++j) point[j] = (mask >> j) & 1u;
            if (lp.max_row_violation(point) <= 1e-9)
                optimum = std::min(optimum, lp.objective(point));
        }
        CHECK_NEAR(raw.objective, optimum, 1e-7);
        const auto result = sor::certify::finalize_result(raw, sor::search::milp_evidence(d, o));
        CHECK(result.status == sor::core::Status::Optimal);
        std::printf("FOLD_TEST rhs=%g nodes=%llu hits=%llu obj=%g point_obj=%g\n", rhs, (unsigned long long)d.nodes, (unsigned long long)d.heuristic_hits, raw.objective, lp.objective(raw.x));
    }
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--root-handoff") {
        test_proved_cut_loop_root_is_reused_only_at_identical_bounds(true);
        return sor::test::finish("test_milp_root_handoff");
    }
    CHECK(!BabOptions{}.batch_lp_strong_branch);
    test_incumbent_point_validation();
    test_folded_incumbents_stay_in_original_space();
    test_small_coefficient_optimum_is_not_cut_off();
    test_knapsack();
    test_infeasible_mip();
    test_folding_invalidates_probed_implied_bounds();
    test_integer_row_rounding();
    test_implied_integer_slack();
    test_hybrid_node_selection_matches_best_bound();
    test_hybrid_node_selection_infeasible_still_detected();
    test_abandoned_node_is_not_a_proof();
    test_gap_closed_before_exhaustion_still_proves();
    test_objective_lattice_requires_exact_costs();
    test_node_cutoff_values();
    test_gap_pruning_reports_honest_dual_bound(1);
    test_gap_pruning_reports_honest_dual_bound(2);
    test_enigma_stops_when_the_global_gap_closes();
    test_integral_lp_points_reach_incumbent_validation_before_gap_pruning();
    test_root_neighborhood_obeys_milp_deadline();
    test_heuristic_spent_ms_sums_heuristic_timers();
    test_node_rounding_does_not_repeat_identical_attempts();
    test_node_rounding_backs_off_after_misses();
    test_ambiguous_node_status_requires_bounded_root();
    test_reported_optimal_is_integral_at_feasibility_tolerance();
    test_node_infeasibility_needs_a_checked_ray();
    test_proved_cut_loop_root_is_reused_only_at_identical_bounds();
    return sor::test::finish("test_milp");
}
