// Gomory Mixed-Integer cut separator (sor_search/src/cuts.cpp): a hand-built
// fixture whose root LP relaxation is fractional but whose first GMI cut is
// exact (proves integrality with zero branching), plus a cuts-on/cuts-off
// parity check on the existing MILP fixtures -- cuts must never change the
// proven answer, only the path to it.
#include "sor/certify/finalize.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/cuts.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include "sor/core/route_debug.hpp"

using sor::core::Status;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

// min -x  s.t.  2x <= 3, x integer, x >= 0.
// LP optimum: x = 1.5. A single GMI cut on the (only) tableau row derives
// exactly x <= 1, which is both valid (excludes no integer-feasible point)
// and proves the MILP optimum (x = 1) without any branching.
// The PL bound is required: an MPS marker integer with no BOUNDS record is
// binary, which would make the LP optimum integral and leave nothing to cut.
const char* kGmiFixture = R"(NAME          GMICUT
ROWS
 N  COST
 L  R1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X         COST      -1             R1        2
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        3
BOUNDS
 PL BND       X
ENDATA
)";

// max 5 x1 + 3 x2 + 2 x3  s.t. 4 x1 + 2 x2 + x3 <= 5, x binary. Same fixture
// as test_milp.cpp's knapsack, reused here for the cuts-on/off parity check.
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

void test_gmi_separates_and_is_valid() {
    auto lp = read_text(kGmiFixture);
    CHECK(lp.n_integer() == 1);

    sor::engines::SimplexOptions lp_opts;
    lp_opts.presolve = false;  // basis must be in lp's own bound space
    sor::engines::SimplexDiagnostics sd;
    sor::engines::SimplexBasis basis;
    const auto raw = sor::engines::solve_simplex(lp, lp_opts, sd, &basis);
    CHECK(raw.proposed_status == Status::Optimal);
    CHECK_NEAR(raw.x[0], 1.5, 1e-6);

    sor::search::CutOptions copts;
    sor::search::CutDiagnostics cdiag;
    const auto cuts = sor::search::separate_gomory_mi(lp, raw.x, basis, copts, cdiag);
    CHECK(cuts.size() == 1);
    CHECK(cdiag.gmi_cuts_added == 1);
    if (cuts.empty()) return;

    const auto& cut = cuts[0];
    // Valid: every integer point satisfying the original constraint (x = 0
    // or x = 1, since 2x <= 3) must also satisfy the cut.
    for (double xi : {0.0, 1.0}) {
        double activity = 0.0;
        for (std::size_t k = 0; k < cut.cols.size(); ++k)
            if (cut.cols[k] == 0) activity += cut.vals[k] * xi;
        CHECK(activity >= cut.row_lo - 1e-9);
    }
    // Cuts off the LP point x = 1.5.
    double activity_lp = 0.0;
    for (std::size_t k = 0; k < cut.cols.size(); ++k)
        if (cut.cols[k] == 0) activity_lp += cut.vals[k] * raw.x[0];
    CHECK(activity_lp < cut.row_lo - 1e-9);

    // Applying it and re-solving proves the integer optimum directly.
    const auto cut_lp = sor::search::apply_cuts(lp, cuts, sor::search::CutOptions{});
    sor::engines::SimplexDiagnostics sd2;
    const auto raw2 = sor::engines::solve_simplex(cut_lp, lp_opts, sd2, nullptr);
    CHECK(raw2.proposed_status == Status::Optimal);
    CHECK_NEAR(raw2.x[0], 1.0, 1e-6);
    CHECK_NEAR(raw2.objective, -1.0, 1e-6);
}

// A row activity A_i x is an integer variable when every nonzero A_ij and
// x_j is integral. Exercise that path on deterministic small integer models,
// and check every emitted cut against *all* feasible integer assignments in
// the finite box. This tests validity, not just whether the cut separates the
// current fractional LP point.
// Unequal arc capacities yield a valid GMI with coefficient range 150.
// c-MIR can recover the unit-coefficient cover without relaxing that limit.
void test_gmi_dynamism_recovery() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(3, 4, {0, 0, 1, 1, 2, 2},
        {2, 3, 2, 0, 3, 1}, {1.0, 1.0, 1.0, -20.0, 1.0, -3000.0});
    lp.row_lo = {3.0, -sor::model::kInf, -sor::model::kInf};
    lp.row_hi = {sor::model::kInf, 0.0, 0.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 20.0, 3000.0};
    lp.c = {1.0, 300.0, 0.0, 0.0};
    lp.is_integer = {true, true, false, false};
    sor::engines::SimplexOptions so;
    so.presolve = false;
    sor::engines::SimplexDiagnostics sd;
    sor::engines::SimplexBasis basis;
    const auto raw = sor::engines::solve_simplex(lp, so, sd, &basis);
    CHECK(raw.proposed_status == Status::Optimal);
    sor::search::CutOptions opts;
    // Isolate the recovery path: the small-term relaxation and the tableau
    // c-MIR would each resolve this row before recovery is reached.
    opts.gmi_cmir_recovery = true;
    opts.relax_small_terms = false;
    opts.tableau_cmir = false;
    sor::search::CutDiagnostics diag;
    const auto cuts = sor::search::separate_gomory_mi(lp, raw.x, basis, opts, diag);
    CHECK(!cuts.empty());
    CHECK(diag.cmir_recovered > 0);
    opts.gmi_cmir_recovery = false;
    sor::search::CutDiagnostics plain;
    CHECK(sor::search::separate_gomory_mi(lp, raw.x, basis, opts, plain).empty());
    CHECK(plain.cmir_attempted == 0);
    CHECK(plain.rejected_dynamism > 0);
    for (const auto& cut : cuts) {
        double activity = 0.0, largest = 0.0, smallest = 1e300;
        for (std::size_t k = 0; k < cut.cols.size(); ++k) {
            activity += cut.vals[k] * raw.x[std::size_t(cut.cols[k])];
            largest = std::max(largest, std::fabs(cut.vals[k]));
            smallest = std::min(smallest, std::fabs(cut.vals[k]));
        }
        CHECK(activity < cut.row_lo - opts.violation_min);
        CHECK(largest / smallest <= opts.dynamism_max);
        // Minimize cut activity for every binary assignment. Continuous
        // flows remain free, so this covers the whole feasible polyhedron.
        for (int a = 0; a <= 1; ++a) for (int b = 0; b <= 1; ++b) {
            auto sub = lp;
            sub.col_lo[0] = sub.col_hi[0] = double(a);
            sub.col_lo[1] = sub.col_hi[1] = double(b);
            sub.c.assign(4, 0.0);
            for (std::size_t k = 0; k < cut.cols.size(); ++k)
                sub.c[std::size_t(cut.cols[k])] = cut.vals[k];
            sor::engines::SimplexDiagnostics check;
            const auto optimum = sor::engines::solve_simplex(sub, so, check);
            if (optimum.proposed_status == Status::Infeasible) continue;
            CHECK(optimum.proposed_status == Status::Optimal);
            CHECK(optimum.objective >= cut.row_lo - 1e-7);
        }
    }
}

void test_integer_activity_gmi_validity() {
    std::uint32_t state = 0x73493b19u;
    const auto draw = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state;
    };
    std::uint64_t eligible = 0, terms = 0;
    for (int trial = 0; trial < 128; ++trial) {
        sor::model::LpProblem lp;
        std::vector<sor::core::Index> row, col;
        std::vector<double> val;
        for (sor::core::Index i = 0; i < 3; ++i)
            for (sor::core::Index j = 0; j < 3; ++j) {
                const double a = static_cast<double>(draw() % 5u);
                if (a == 0.0) continue;
                row.push_back(i);
                col.push_back(j);
                val.push_back(a);
            }
        lp.A = sor::sparse::from_triplets(3, 3, row, col, val);
        lp.c = {-static_cast<double>(1u + draw() % 4u),
                -static_cast<double>(1u + draw() % 4u),
                -static_cast<double>(1u + draw() % 4u)};
        lp.row_lo = {-sor::model::kInf, -sor::model::kInf,
                     -sor::model::kInf};
        lp.row_hi = {static_cast<double>(3u + draw() % 9u),
                     static_cast<double>(3u + draw() % 9u),
                     static_cast<double>(3u + draw() % 9u)};
        lp.col_lo = {0.0, 0.0, 0.0};
        lp.col_hi = {3.0, 3.0, 3.0};
        lp.is_integer = {true, true, true};
        sor::engines::SimplexOptions lp_opts;
        lp_opts.presolve = false;
        sor::engines::SimplexDiagnostics sd;
        sor::engines::SimplexBasis basis;
        const auto raw = sor::engines::solve_simplex(lp, lp_opts, sd, &basis);
        if (raw.proposed_status != Status::Optimal) continue;
        sor::search::CutOptions copts;
        copts.integer_slack_gmi = true;
        copts.integer_activity_basic_gmi = true;
        copts.rank_gmi_candidates = (trial % 2 == 0);
        copts.gmi_max_tableau_trials = (trial % 2 == 0) ? 1 : 0;
        sor::search::CutDiagnostics cd;
        const auto cuts = sor::search::separate_gomory_mi(
            lp, raw.x, basis, copts, cd);
        if (copts.gmi_max_tableau_trials > 0)
            CHECK(cd.candidates_considered <= 1);
        eligible += cd.integral_activity_rows;
        terms += cd.integer_activity_terms;
        for (int x0 = 0; x0 <= 3; ++x0)
            for (int x1 = 0; x1 <= 3; ++x1)
                for (int x2 = 0; x2 <= 3; ++x2) {
                    const double x[3] = {double(x0), double(x1), double(x2)};
                    bool feasible = true;
                    const auto& rp = lp.A.pattern.row_ptr();
                    const auto& ci = lp.A.pattern.col_idx();
                    for (sor::core::Index i = 0; i < 3; ++i) {
                        double activity = 0.0;
                        for (sor::core::Offset k = rp[std::size_t(i)];
                             k < rp[std::size_t(i) + 1]; ++k)
                            activity += lp.A.vals[std::size_t(k)] *
                                        x[std::size_t(ci[std::size_t(k)])];
                        if (activity > lp.row_hi[std::size_t(i)])
                            feasible = false;
                    }
                    if (!feasible) continue;
                    for (const auto& cut : cuts) {
                        double activity = 0.0;
                        for (std::size_t k = 0; k < cut.cols.size(); ++k)
                            activity += cut.vals[k] *
                                        x[std::size_t(cut.cols[k])];
                        CHECK(activity >= cut.row_lo - 1e-7);
                    }
                }
    }
    CHECK(eligible > 0);
    CHECK(terms > 0);
}

// The integer-activity argument does not depend on nonnegative matrix
// coefficients or a nonnegative integer box. Enumerate signed models too:
// complementing a variable at its upper bound and substituting a signed row
// activity exercise different tableau coefficients from the test above.
void test_integer_activity_gmi_signed_box_validity() {
    std::uint32_t state = 0xa31c9e72u;
    const auto draw = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state;
    };
    std::uint64_t activity_terms = 0;
    for (int trial = 0; trial < 128; ++trial) {
        sor::model::LpProblem lp;
        std::vector<sor::core::Index> row, col;
        std::vector<double> val;
        for (sor::core::Index i = 0; i < 3; ++i)
            for (sor::core::Index j = 0; j < 3; ++j) {
                const double a = static_cast<double>(
                    static_cast<int>(draw() % 7u) - 3);
                if (a == 0.0) continue;
                row.push_back(i);
                col.push_back(j);
                val.push_back(a);
            }
        lp.A = sor::sparse::from_triplets(3, 3, row, col, val);
        lp.c = {static_cast<double>(static_cast<int>(draw() % 5u) - 2),
                static_cast<double>(static_cast<int>(draw() % 5u) - 2),
                static_cast<double>(static_cast<int>(draw() % 5u) - 2)};
        lp.row_lo = {-5.0, -5.0, -5.0};
        lp.row_hi = {5.0, 5.0, 5.0};
        lp.col_lo = {-2.0, -2.0, -2.0};
        lp.col_hi = {2.0, 2.0, 2.0};
        lp.is_integer = {true, true, true};
        sor::engines::SimplexOptions lp_opts;
        lp_opts.presolve = false;
        sor::engines::SimplexDiagnostics sd;
        sor::engines::SimplexBasis basis;
        const auto raw = sor::engines::solve_simplex(lp, lp_opts, sd, &basis);
        if (raw.proposed_status != Status::Optimal) continue;
        sor::search::CutOptions copts;
        copts.integer_slack_gmi = true;
        sor::search::CutDiagnostics cd;
        const auto cuts = sor::search::separate_gomory_mi(
            lp, raw.x, basis, copts, cd);
        activity_terms += cd.integer_activity_terms;
        for (int x0 = -2; x0 <= 2; ++x0)
            for (int x1 = -2; x1 <= 2; ++x1)
                for (int x2 = -2; x2 <= 2; ++x2) {
                    const double x[3] = {double(x0), double(x1), double(x2)};
                    bool feasible = true;
                    const auto& rp = lp.A.pattern.row_ptr();
                    const auto& ci = lp.A.pattern.col_idx();
                    for (sor::core::Index i = 0; i < 3; ++i) {
                        double activity = 0.0;
                        for (sor::core::Offset k = rp[std::size_t(i)];
                             k < rp[std::size_t(i) + 1]; ++k)
                            activity += lp.A.vals[std::size_t(k)] *
                                        x[std::size_t(ci[std::size_t(k)])];
                        if (activity < lp.row_lo[std::size_t(i)] ||
                            activity > lp.row_hi[std::size_t(i)])
                            feasible = false;
                    }
                    if (!feasible) continue;
                    for (const auto& cut : cuts) {
                        double activity = 0.0;
                        for (std::size_t k = 0; k < cut.cols.size(); ++k)
                            activity += cut.vals[k] *
                                        x[std::size_t(cut.cols[k])];
                        CHECK(activity >= cut.row_lo - 1e-7);
                    }
                }
    }
    CHECK(activity_terms > 0);
}

void test_integer_activity_requires_exact_integer_terms() {
    for (int case_id = 0; case_id < 2; ++case_id) {
        sor::model::LpProblem lp;
        lp.A = sor::sparse::from_triplets(
            1, 2, {0, 0}, {0, 1},
            {1.0, case_id == 0 ? 1.0 : 0.5});
        lp.c = {-1.0, -1.0};
        lp.row_lo = {-sor::model::kInf};
        lp.row_hi = {3.0};
        lp.col_lo = {0.0, 0.0};
        lp.col_hi = {3.0, 3.0};
        lp.is_integer = {true, case_id != 0};
        sor::engines::SimplexOptions lp_opts;
        lp_opts.presolve = false;
        sor::engines::SimplexDiagnostics sd;
        sor::engines::SimplexBasis basis;
        const auto raw = sor::engines::solve_simplex(lp, lp_opts, sd, &basis);
        CHECK(raw.proposed_status == Status::Optimal);
        if (raw.proposed_status != Status::Optimal) continue;
        sor::search::CutOptions copts;
        copts.integer_slack_gmi = true;
        sor::search::CutDiagnostics cd;
        sor::search::separate_gomory_mi(lp, raw.x, basis, copts, cd);
        CHECK(cd.integral_activity_rows == 0);
        CHECK(cd.integer_activity_terms == 0);
    }
}

// x is integer but the LP can sit at its fractional lower bound 1/2. The
// shifted nonbasic value x-1/2 is NOT an integer. The old separator treated
// it as one and derived a GMI inequality that cut off the feasible integer
// point (x,y)=(1,0). Row-activity strengthening must not be required for
// this guard: it applies to the default GMI separator as well.
void test_gmi_fractional_integer_bound_keeps_feasible_point() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1},
                                      {5.0, 2.0});
    lp.c = {0.0, -1.0};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {5.0};
    lp.col_lo = {0.5, 0.0};
    lp.col_hi = {2.0, 3.0};
    lp.is_integer = {true, true};
    sor::engines::SimplexOptions lp_opts;
    lp_opts.presolve = false;
    sor::engines::SimplexDiagnostics sd;
    sor::engines::SimplexBasis basis;
    const auto raw = sor::engines::solve_simplex(lp, lp_opts, sd, &basis);
    CHECK(raw.proposed_status == Status::Optimal);
    if (raw.proposed_status != Status::Optimal) return;
    CHECK_NEAR(raw.x[0], 0.5, 1e-8);
    CHECK_NEAR(raw.x[1], 1.25, 1e-8);
    for (bool integer_activity : {false, true}) {
        sor::search::CutOptions copts;
        copts.integer_slack_gmi = integer_activity;
        sor::search::CutDiagnostics cd;
        const auto cuts = sor::search::separate_gomory_mi(
            lp, raw.x, basis, copts, cd);
        CHECK(!cuts.empty());
        for (const auto& cut : cuts) {
            double activity = 0.0;
            for (std::size_t k = 0; k < cut.cols.size(); ++k)
                activity += cut.vals[k] * (cut.cols[k] == 0 ? 1.0 : 0.0);
            CHECK(activity >= cut.row_lo - 1e-9);
        }
    }
}

void test_gmi_fixture_via_solve_milp() {
    auto lp = read_text(kGmiFixture);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.max_nodes = 1000;
    opts.cuts_enabled = true;
    // Isolate the cut loop from newer root reductions that already prove
    // this one-row fixture before GMI runs (integral-row rounding / MIP
    // presolve / symmetry).
    opts.integer_row_rounding = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal || r.status == Status::Feasible);
    CHECK_NEAR(r.objective, -1.0, 1e-6);
    CHECK(diag.gmi_cuts_added >= 1);
    CHECK(diag.cut_rounds >= 1);
    CHECK(diag.cut_pool_inserted >= 1);
    CHECK(diag.cut_pool_inserted >= diag.gmi_cuts_added);
}

// Cuts must never change the proven answer on an existing fixture -- only
// the number of nodes needed to reach it.
void test_cuts_do_not_change_knapsack_answer() {
    auto lp = read_text(kKnapsack);

    BabOptions with_cuts;
    with_cuts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    with_cuts.max_nodes = 1000;
    with_cuts.cuts_enabled = true;
    BabDiagnostics diag_with;
    auto raw_with = sor::search::solve_milp(lp, with_cuts, diag_with);
    const auto r_with = sor::certify::finalize_result(
        std::move(raw_with), sor::search::milp_evidence(diag_with, with_cuts));

    BabOptions without_cuts;
    without_cuts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    without_cuts.max_nodes = 1000;
    without_cuts.cuts_enabled = false;
    BabDiagnostics diag_without;
    auto raw_without = sor::search::solve_milp(lp, without_cuts, diag_without);
    const auto r_without = sor::certify::finalize_result(
        std::move(raw_without),
        sor::search::milp_evidence(diag_without, without_cuts));

    CHECK(r_with.status == r_without.status);
    CHECK_NEAR(r_with.objective, r_without.objective, 1e-6);
    CHECK_NEAR(r_with.objective, -7.0, 1e-6);
}

sor::search::CutRow lower_cut(std::vector<sor::core::Index> cols,
                              std::vector<double> vals, double rhs,
                              const char* name) {
    sor::search::CutRow cut;
    cut.cols = std::move(cols);
    cut.vals = std::move(vals);
    cut.row_lo = rhs;
    cut.row_hi = sor::model::kInf;
    cut.name = name;
    return cut;
}

// The penalty scheme is still implemented and still selected by
// pool_parallel_hard_filter=false; it is simply not the default. Covering it
// separately keeps the non-default branch from rotting unnoticed.
void test_cut_pool_parallel_penalty_branch() {
    sor::search::CutOptions opts;
    opts.max_cuts_per_round = 2;
    opts.pool_max_age = 1;
    opts.pool_parallelism_max = 0.99;
    opts.pool_efficacy_min = 1e-9;
    opts.pool_parallel_hard_filter = false;   // the non-default branch
    sor::search::CutDiagnostics diag;
    sor::search::CutPool pool(opts);

    // The pair that is actually near-parallel is the pure x0 row against the
    // {x0, x1} row; the x1 row is orthogonal to both and is what a diverse
    // batch keeps. A pool holding only the last two has nothing near-parallel
    // in it at all -- their cosine is 0.01, under pool_parallelism_penalty_min.
    pool.add({lower_cut({0}, {1.0}, 1.5, "x0_row")}, diag);
    pool.add({lower_cut({0, 1}, {1.0, 0.01}, 1.6, "near_parallel"),
              lower_cut({1}, {1.0}, 1.0, "orthogonal")}, diag);
    const auto selected = pool.select_violated({0.0, 0.0}, diag);
    CHECK(selected.size() == 2);
    CHECK(selected[0].name == "near_parallel");
    CHECK(selected[1].name == "orthogonal");
    // Scored down rather than removed: the same diverse batch comes out, by a
    // different mechanism, and nothing was rejected outright.
    CHECK(diag.pool_penalized_parallel >= 1);
    CHECK(diag.pool_rejected_parallel == 0);
}

void test_cut_pool_management() {
    sor::search::CutOptions opts;
    opts.max_cuts_per_round = 2;
    opts.pool_max_age = 1;
    opts.pool_parallelism_max = 0.99;
    opts.pool_efficacy_min = 1e-9;
    sor::search::CutDiagnostics diag;
    sor::search::CutPool pool(opts);

    // Positive rescaling is the same inequality, and a weaker parallel row is
    // dominated. A stronger row replaces an inactive weaker pool entry.
    pool.add({lower_cut({0}, {1.0}, 1.0, "base")}, diag);
    pool.add({lower_cut({0}, {2.0}, 2.0, "scaled_duplicate")}, diag);
    pool.add({lower_cut({0}, {1.0}, 0.5, "weaker")}, diag);
    pool.add({lower_cut({0}, {1.0}, 1.5, "stronger")}, diag);
    CHECK(pool.size() == 1);
    CHECK(diag.pool_duplicates == 1);
    CHECK(diag.pool_dominated == 2);

    // Efficacy ranking takes the most violated near-parallel x0 row, filters
    // the other, and retains the orthogonal x1 row for a diverse batch.
    pool.add({lower_cut({0, 1}, {1.0, 0.01}, 1.6, "near_parallel"),
              lower_cut({1}, {1.0}, 1.0, "orthogonal")}, diag);
    const auto selected = pool.select_violated({0.0, 0.0}, diag);
    CHECK(selected.size() == 2);
    CHECK(selected[0].name == "near_parallel");
    CHECK(selected[1].name == "orthogonal");
    // The DEFAULT mechanism is the hard filter, not the penalty. cuts.hpp
    // records why: the penalty is what Turner et al. (2023) 2.2.2 recommend,
    // and swapping to it cost a proof on miplib-easy, so `pool_parallel_hard_
    // filter` stays true and the near-parallel row is REJECTED outright rather
    // than scored down. Assert the mechanism the tree actually ships.
    CHECK(diag.pool_rejected_parallel == 1);
    CHECK(diag.pool_penalized_parallel == 0);
    CHECK(pool.active_size() == 2);

    // The unselected parallel row expires, while active fingerprints survive
    // aging so a later separator round cannot append duplicates.
    pool.start_round(diag);
    pool.start_round(diag);
    CHECK(diag.pool_aged_out == 1);
    CHECK(pool.size() == 2);
}

}  // namespace

// apply_cuts_inplace + retract_cuts_inplace must be an exact round trip, and
// crucially must undo BOTH kinds of edit a round makes: the appended rows and
// the bounds it folded into a pre-existing row of the same shape.
void test_cut_retraction_round_trip() {
    using sor::search::CutRow;
    using sor::search::CutUndo;

    // Two previously appended CUT rows, above an empty original-row prefix:
    // x + y <= 10 and x - y <= 4, both continuous-free boxes.
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1},
                                      {1.0, 1.0, 1.0, -1.0});
    lp.c = {-1.0, -1.0};
    lp.row_lo = {-sor::model::kInf, -sor::model::kInf};
    lp.row_hi = {10.0, 4.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {20.0, 20.0};
    lp.is_integer = {true, true};

    const sor::model::LpProblem before = lp;
    sor::search::CutOptions opts;
    opts.original_rows = 0;

    // One cut on a NEW shape (appends a row) and one that is a positive
    // rescaling of row 0 with a tighter rhs (folds into row 0's bounds).
    std::vector<CutRow> cuts(2);
    cuts[0].cols = {0};
    cuts[0].vals = {1.0};
    cuts[0].row_lo = -sor::model::kInf;
    cuts[0].row_hi = 6.0;
    cuts[0].name = "NEW_0";
    cuts[1].cols = {0, 1};
    cuts[1].vals = {2.0, 2.0};
    cuts[1].row_lo = -sor::model::kInf;
    cuts[1].row_hi = 14.0;          // == x + y <= 7, tighter than 10
    cuts[1].name = "MERGE_0";

    CutUndo undo;
    sor::search::apply_cuts_inplace(lp, cuts, opts, &undo);

    CHECK(lp.n_rows() == 3);                 // exactly one row appended
    CHECK(undo.rows_added() == 1);
    CHECK(undo.rows_before == 2);
    CHECK(undo.tightened_rows.size() == 1);  // and one row tightened in place
    CHECK(undo.tightened_rows[0] == 0);
    CHECK_NEAR(lp.row_hi[0], 7.0, 1e-12);    // the merge really happened
    lp.validate();

    sor::search::retract_cuts_inplace(lp, undo);

    // Every observable is back exactly where it started. If the retraction
    // only truncated the appended row, row_hi[0] would still read 7.0 and the
    // "retracted" round would still be constraining the model.
    CHECK(lp.n_rows() == before.n_rows());
    CHECK(lp.nnz() == before.nnz());
    CHECK(lp.A.pattern.row_ptr() == before.A.pattern.row_ptr());
    CHECK(lp.A.pattern.col_idx() == before.A.pattern.col_idx());
    CHECK(lp.A.vals == before.A.vals);
    CHECK(lp.row_lo == before.row_lo);
    CHECK(lp.row_hi == before.row_hi);
    CHECK(lp.row_names.empty());             // names were empty; still are
    lp.validate();

    // A round that merges into the SAME row twice must restore the pre-round
    // bound, not the intermediate one.
    std::vector<CutRow> twice(2);
    twice[0] = cuts[1];                      // x + y <= 7
    twice[1] = cuts[1];
    twice[1].row_hi = 10.0;                  // == x + y <= 5, tighter still
    twice[1].vals = {2.0, 2.0};
    twice[1].name = "MERGE_1";
    CutUndo undo2;
    sor::search::apply_cuts_inplace(lp, twice, opts, &undo2);
    CHECK(undo2.tightened_rows.size() == 1);
    CHECK_NEAR(undo2.tightened_hi[0], 10.0, 1e-12);   // the PRE-round bound
    sor::search::retract_cuts_inplace(lp, undo2);
    CHECK_NEAR(lp.row_hi[0], 10.0, 1e-12);
    lp.validate();
}

// The PURGE kernel: cut rows are a contiguous suffix, so dropping a scattered
// subset of them is truncate-to-the-prefix then re-append the survivors. This
// checks the survivors come back byte-identical and in order -- if they did
// not, purging would silently rewrite constraints rather than remove them.
void test_purge_kernel_preserves_survivors() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 3, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {1.0, 1.0, 1.0};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {10.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {5.0, 5.0, 5.0};

    const sor::core::Index m_orig = lp.n_rows();
    lp.A.append_row({0}, {1.0});         lp.row_lo.push_back(-sor::model::kInf); lp.row_hi.push_back(1.0);
    lp.A.append_row({1}, {2.0});         lp.row_lo.push_back(-sor::model::kInf); lp.row_hi.push_back(2.0);
    lp.A.append_row({2}, {3.0});         lp.row_lo.push_back(-sor::model::kInf); lp.row_hi.push_back(3.0);
    lp.A.append_row({0, 2}, {4.0, 5.0}); lp.row_lo.push_back(-sor::model::kInf); lp.row_hi.push_back(4.0);
    CHECK(lp.n_rows() == 5);
    lp.validate();

    // Keep cut rows 2 and 4 -- a SCATTERED subset, which is the case a plain
    // truncation cannot express.
    struct Kept { std::vector<sor::core::Index> cols;
                  std::vector<sor::core::f64> vals;
                  sor::core::f64 lo, hi; };
    std::vector<Kept> keep;
    {
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        const auto& av = lp.A.vals;
        for (sor::core::Index i : {sor::core::Index(2), sor::core::Index(4)}) {
            Kept k;
            k.lo = lp.row_lo[std::size_t(i)];
            k.hi = lp.row_hi[std::size_t(i)];
            for (auto q = rp[std::size_t(i)]; q < rp[std::size_t(i) + 1]; ++q) {
                k.cols.push_back(ci[std::size_t(q)]);
                k.vals.push_back(av[std::size_t(q)]);
            }
            keep.push_back(std::move(k));
        }
    }

    lp.A.truncate_rows(m_orig);
    lp.row_lo.resize(std::size_t(m_orig));
    lp.row_hi.resize(std::size_t(m_orig));
    for (auto& k : keep) {
        lp.A.append_row(k.cols, k.vals);
        lp.row_lo.push_back(k.lo);
        lp.row_hi.push_back(k.hi);
    }

    CHECK(lp.n_rows() == 3);
    lp.validate();
    CHECK_NEAR(lp.row_hi[0], 10.0, 1e-15);
    CHECK_NEAR(lp.row_hi[1], 2.0, 1e-15);
    CHECK_NEAR(lp.row_hi[2], 4.0, 1e-15);
    const auto& rp2 = lp.A.pattern.row_ptr();
    const auto& ci2 = lp.A.pattern.col_idx();
    const auto& av2 = lp.A.vals;
    CHECK(rp2[2] - rp2[1] == 1);
    CHECK(ci2[std::size_t(rp2[1])] == 1);
    CHECK_NEAR(av2[std::size_t(rp2[1])], 2.0, 1e-15);
    CHECK(rp2[3] - rp2[2] == 2);
    CHECK_NEAR(av2[std::size_t(rp2[2])], 4.0, 1e-15);
    CHECK_NEAR(av2[std::size_t(rp2[2]) + 1], 5.0, 1e-15);

    // Basis invalidation: a basis sized for the pre-purge model no longer
    // matches, which is why the purge drops it instead of extending it.
    CHECK(lp.n_rows() != 5);
}

void test_restart_reduced_cost_fixes_integer_not_continuous() {
    // LP optimum pins the continuous y at its upper bound and leaves an
    // integer fractional. The integer optimum moves y off that bound. A
    // second integer, z, sits at 0 with a large reduced cost and is a
    // legitimate fix. Removing the caller's integer gate would pin y.
    sor::model::LpProblem lp;
    lp.name = "rc_restart";
    // d >= 0.5 forces one more integer branch, so an incumbent is found
    // before the tree is empty. The next node then hits the restart check.
    lp.A = sor::sparse::from_triplets(2, 5,
                                      {0, 0, 0, 1},
                                      {0, 1, 2, 4},
                                      {1.0, 1.0, 1.0, 1.0});
    lp.row_lo = {1.2, 0.5};
    lp.row_hi = {sor::model::kInf, sor::model::kInf};
    lp.c = {0.5, 0.5, 0.15, 10.0, 0.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0, 1.0, 1.0};
    lp.is_integer = {true, true, false, true, true};
    lp.col_names = {"x", "x2", "y", "z", "d"};

    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = sor::search::MilpPolicy::Latest;
    opts.tree_restart = true;
    opts.reduced_cost_strengthening = false;  // isolate the restart's RC fallback
    opts.tree_restart_node_gap = 0;
    opts.tree_restart_max = 1;
    // Keep the tree: node cut re-solves would otherwise close this tiny model.
    opts.tree_cut.resolve_with_local = false;
    // The model has two independent parts; keep it one search.
    opts.component_solve = false;
    opts.max_nodes = 40;
    opts.time_limit_s = 10.0;
    opts.fixprop = false;
    opts.cuts_enabled = false;
    opts.probing = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.rounding_heuristic = false;
    opts.lp_rounding_repair = false;
    opts.integer_dive = false;
    opts.integer_neighborhood = false;
    opts.conflict_propagation = false;
    opts.kernel_pump.enabled = false;
    opts.balans.enabled = false;
    opts.mrens.enabled = false;
    opts.btbs.enabled = false;
    opts.cl_tlns.enabled = false;

    BabDiagnostics diag;
    const auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.tree_restarts >= 1);
    CHECK(diag.restart_rc_col_lo.size() == 5);
    CHECK(diag.restart_rc_col_hi.size() == 5);
    CHECK(diag.restart_rc_integer_fixed >= 1);
    if (diag.restart_rc_col_lo.size() == 5 && diag.restart_rc_col_hi.size() == 5) {
        CHECK_NEAR(diag.restart_rc_col_lo[3], 0.0, 1e-8);
        CHECK_NEAR(diag.restart_rc_col_hi[3], 0.0, 1e-8);
        CHECK_NEAR(diag.restart_rc_col_lo[2], 0.0, 1e-8);
        CHECK_NEAR(diag.restart_rc_col_hi[2], 1.0, 1e-8);
    }
    CHECK(raw.x.size() == 5);
    if (raw.x.size() == 5) {
        CHECK(raw.x[2] < 0.9);
        CHECK(raw.x[2] > 0.05);
    }
}

void test_bound_snapshot_names_the_invocation() {
    sor::model::LpProblem lp;
    lp.name = "snap";
    lp.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {1.0});
    lp.row_lo = {1.0};
    lp.row_hi = {1.0};
    lp.c = {1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {1.0};
    lp.is_integer = {true};
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = sor::search::MilpPolicy::Classical;
    opts.max_nodes = 5;
    opts.cuts_enabled = false;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    const char* path = "/tmp/sor-snap-review.txt";
    ::setenv("SOR_DUMP_MIP_BOUNDS", path, 1);
    std::remove((std::string(path) + ".invocations.jsonl").c_str());
    BabDiagnostics d1;
    BabDiagnostics d2;
    sor::search::solve_milp(lp, opts, d1);
    sor::search::solve_milp(lp, opts, d2);
    ::unsetenv("SOR_DUMP_MIP_BOUNDS");
    CHECK(d1.invocation_id != d2.invocation_id);
    std::ifstream in(std::string(path) + ".invocations.jsonl");
    std::string line;
    std::string snap1;
    std::string snap2;
    std::string run;
    while (std::getline(in, line)) {
        if (line.find("\"phase\":\"received_box\"") == std::string::npos) continue;
        auto field = [&](const char* key) {
            const std::string needle = std::string("\"") + key + "\":\"";
            const auto at = line.find(needle);
            CHECK(at != std::string::npos);
            const auto b = at + needle.size();
            const auto e = line.find('"', b);
            return line.substr(b, e - b);
        };
        const std::string inv_key = "\"invocation\":";
        const auto inv_at = line.find(inv_key);
        CHECK(inv_at != std::string::npos);
        const std::string inv = line.substr(inv_at + inv_key.size(),
                                            line.find(',', inv_at) - (inv_at + inv_key.size()));
        const std::string file = field("snapshot");
        const std::string sha = field("snapshot_sha256");
        const std::string content = field("content_fp");
        const std::string cols = field("col_order_fp");
        CHECK(line.find("\"column_map\":\"identity\"") != std::string::npos);
        CHECK(!content.empty());
        CHECK(!cols.empty());
        if (run.empty()) run = field("run");
        else CHECK(field("run") == run);
        CHECK(file.find(".run" + run + ".inv" + inv + ".received_box") !=
              std::string::npos);
        std::ifstream body(file);
        std::stringstream buf;
        buf << body.rdbuf();
        FILE* pipe = popen(("sha256sum " + file).c_str(), "r");
        CHECK(pipe != nullptr);
        char hex[80] = {};
        CHECK(std::fgets(hex, sizeof hex, pipe) != nullptr);
        pclose(pipe);
        CHECK(std::string(hex).rfind(sha, 0) == 0);
        CHECK(buf.str().rfind("1\n", 0) == 0);
        if (inv == std::to_string(d1.invocation_id)) snap1 = file;
        if (inv == std::to_string(d2.invocation_id)) snap2 = file;
    }
    CHECK(!snap1.empty());
    CHECK(!snap2.empty());
    CHECK(snap1 != snap2);
}

void test_integer_reduced_cost_fix_unit_step() {
    using sor::search::integer_reduced_cost_fix;
    const double tol = 1e-8;
    double lo = 0.0;
    double hi = 5.0;
    CHECK(integer_reduced_cost_fix(lo, hi, 0.0, 3.0, 3.0, tol));
    CHECK_NEAR(lo, 0.0, tol);
    CHECK_NEAR(hi, 0.0, tol);

    lo = 0.0;
    hi = 5.0;
    CHECK(!integer_reduced_cost_fix(lo, hi, 0.0, 1.5, 3.0, tol));
    CHECK_NEAR(hi, 5.0, tol);

    lo = 0.5;
    hi = 5.0;
    CHECK(!integer_reduced_cost_fix(lo, hi, 0.5, 3.0, 3.0, tol));
    CHECK_NEAR(lo, 0.5, tol);
    CHECK_NEAR(hi, 5.0, tol);

    lo = 0.0;
    hi = 5.0;
    CHECK(integer_reduced_cost_fix(lo, hi, 5.0, -3.0, 3.0, tol));
    CHECK_NEAR(lo, 5.0, tol);
    CHECK_NEAR(hi, 5.0, tol);
    // Every exclusion must come from the certified Lagrangian: the result has
    // to retain every point as good as the incumbent.
    for (bool maximize : {false, true}) {
        sor::model::LpProblem p;
        p.A = sor::sparse::from_triplets(0, 2, {}, {}, {});
        p.c = maximize ? std::vector<double>{-10., -1.} : std::vector<double>{10., 1.};
        p.col_lo = {0., 0.}; p.col_hi = {3., 3.}; p.is_integer = {1, 1}; p.maximize = maximize;
        auto lows = p.col_lo, highs = p.col_hi;
        const auto result = sor::search::certified_reduced_cost_bounds(
            p, {}, maximize ? -3. : 3., lows, highs, tol);
        CHECK(result.checked == 2);
        CHECK(result.fixed == 1);
        for (int a = 0; a <= 3; ++a) for (int b = 0; b <= 3; ++b) {
            const auto obj = p.objective({double(a), double(b)});
            if ((maximize ? -obj : obj) <= 3.) {
                CHECK(a >= lows[0] && a <= highs[0]);
                CHECK(b >= lows[1] && b <= highs[1]);
            }
        }
        p.c = maximize ? std::vector<double>{1., 0.} : std::vector<double>{-1., 0.};
        p.col_hi = {100., 1.}; p.is_integer = {1, 0};
        lows = p.col_lo; highs = p.col_hi;
        const auto upper = sor::search::certified_reduced_cost_bounds(
            p, {}, maximize ? 90. : -90., lows, highs, tol);
        CHECK(upper.tightened == 1);
        CHECK(lows[0] == 90. && highs[0] == 100.);
        CHECK(lows[1] == 0. && highs[1] == 1.);
    }
}

// A tiny reduced cost against a large incumbent gap must not turn an INFINITE
// bound into an astronomical finite one (gap / d = 2e13 here): such a bound
// tightens nothing, and it makes later Lagrangian dual objectives charge
// roundoff-sized reduced costs 2e13 each, which failed the primal-dual gap
// test on every nu25-pr12 node LP. A finite old bound is still tightened.
void test_rc_tightening_skips_infinite_to_astronomical() {
    sor::model::LpProblem p;
    p.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {1.0});
    p.c = {1e-6};                       // reduced cost 1e-6 with y = 0
    p.col_lo = {0.0};
    p.col_hi = {sor::model::kInf};
    p.row_lo = {-sor::model::kInf};
    p.row_hi = {1e30};                  // never binding
    p.is_integer = {1};
    p.validate();
    std::vector<double> lows = p.col_lo, highs = p.col_hi;
    // incumbent 2e7 above the LP value 0: gap / d = 2e13
    const auto r = sor::search::certified_reduced_cost_bounds(
        p, {0.0}, 2e7, lows, highs, 1e-7);
    CHECK(highs[0] == sor::model::kInf);
    CHECK(r.tightened == 0);
    // Same reduced cost, finite old bound: tightened as before.
    highs = {1e15};
    const auto r2 = sor::search::certified_reduced_cost_bounds(
        p, {0.0}, 2e7, lows, highs, 1e-7);
    CHECK(highs[0] < 1e15);
    CHECK(r2.tightened == 1);
}

void test_gmi_structural_drop_relaxes_rhs() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {2.0, 1e-12});
    lp.c = {-1.0, 0.0};
    lp.col_lo = {0.0, -1e6}; lp.col_hi = {2.0, 1e6};
    lp.is_integer = {true, false};
    lp.row_lo = {-sor::model::kInf}; lp.row_hi = {3.0};
    sor::engines::SimplexBasis basis;
    basis.n_struct = 2; basis.basic = {0};
    using NB = sor::engines::NonbasicStatus;
    basis.status = {NB::Basic, NB::AtLower, NB::AtUpper};
    const std::vector<double> x{(3.0 + 1e-6) / 2.0, -1e6};
    CHECK(lp.max_row_violation(x) <= 1e-12);
    sor::search::CutOptions o;
    sor::search::CutDiagnostics d;
    const auto cuts = sor::search::separate_gomory_mi(lp, x, basis, o, d);
    CHECK(cuts.size() == 1);
    for (int xi = 0; xi <= 2; ++xi)
        for (double y : {-1e6, 0.0, 1e6}) {
            const std::vector<double> point{static_cast<double>(xi), y};
            if (lp.max_row_violation(point) > 1e-12) continue;
            for (const auto& cut : cuts) {
                double activity = 0.0;
                for (std::size_t k = 0; k < cut.cols.size(); ++k)
                    activity += cut.vals[k] * point[static_cast<std::size_t>(cut.cols[k])];
                CHECK(activity >= cut.row_lo - 1e-9);
            }
        }
    // The required relaxing bound is infinite: do not emit this candidate.
    lp.col_lo[1] = -sor::model::kInf;
    sor::search::CutDiagnostics unbounded;
    CHECK(sor::search::separate_gomory_mi(lp, x, basis, o, unbounded).empty());
}

void test_cut_merge_preserves_original_rows() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1e-12});
    lp.c = {0.0, 0.0}; lp.is_integer = {true, false};
    lp.col_lo = {0.0, -1e6}; lp.col_hi = {1.0, 1e6};
    lp.row_lo = {-sor::model::kInf}; lp.row_hi = {1.0 + 2e-6};
    const auto original = lp;
    sor::search::CutRow cut;
    cut.cols = {0, 1}; cut.vals = {1.0, 1e-12 * (1.0 - 1e-10)};
    cut.row_lo = -sor::model::kInf; cut.row_hi = 1.0 + 1e-6;
    sor::search::CutOptions o;
    sor::search::apply_cuts_inplace(lp, {cut}, o);
    CHECK(lp.n_rows() == 2);
    CHECK(lp.row_hi[0] == original.row_hi[0]);
    CHECK(lp.row_lo[0] == original.row_lo[0]);
    for (int x : {0, 1})
        for (double y : {-1e6, 0.0, 1e6}) {
            const std::vector<double> point{static_cast<double>(x), y};
            CHECK(original.max_row_violation(point) <= 1e-12);
            CHECK(lp.max_row_violation(point) <= 1e-12);
        }
}

void test_cut_merge_requires_exact_normalized_coefficients() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1e-12});
    lp.c = {0.0, 0.0}; lp.is_integer = {true, false};
    lp.col_lo = {0.0, -1e6}; lp.col_hi = {1.0, 1e6};
    lp.row_lo = {-sor::model::kInf}; lp.row_hi = {1.0 + 2e-6};
    sor::search::CutOptions o; o.original_rows = 0; // Existing row is a cut.
    sor::search::CutRow cut;
    cut.cols = {0, 1}; cut.vals = {1.0, 1e-12 * (1.0 - 1e-10)};
    cut.row_lo = -sor::model::kInf; cut.row_hi = 1.0 + 1e-6;
    auto different = lp;
    sor::search::apply_cuts_inplace(different, {cut}, o);
    CHECK(different.n_rows() == 2);
    CHECK(different.row_hi[0] == lp.row_hi[0]);
    cut.vals = {2.0, 2e-12}; cut.row_hi = 2.0 + 2e-6;
    sor::search::apply_cuts_inplace(lp, {cut}, o);
    CHECK(lp.n_rows() == 1);
    CHECK(lp.row_hi[0] == cut.row_hi / 2.0);
}

int main() {
    test_gmi_dynamism_recovery();
    test_cut_merge_preserves_original_rows();
    test_cut_merge_requires_exact_normalized_coefficients();
    test_rc_tightening_skips_infinite_to_astronomical();
    test_gmi_structural_drop_relaxes_rhs();
    test_cut_retraction_round_trip();
    test_purge_kernel_preserves_survivors();
    test_gmi_separates_and_is_valid();
    test_integer_activity_gmi_validity();
    test_integer_activity_gmi_signed_box_validity();
    test_integer_activity_requires_exact_integer_terms();
    test_gmi_fractional_integer_bound_keeps_feasible_point();
    test_gmi_fixture_via_solve_milp();
    test_cuts_do_not_change_knapsack_answer();
    test_cut_pool_management();
    test_cut_pool_parallel_penalty_branch();
    test_integer_reduced_cost_fix_unit_step();
    test_restart_reduced_cost_fixes_integer_not_continuous();
    test_bound_snapshot_names_the_invocation();
    return sor::test::finish("test_cuts");
}
