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

// min -x  s.t.  2x <= 3, x integer, x >= 0.
// LP optimum: x = 1.5. A single GMI cut on the (only) tableau row derives
// exactly x <= 1, which is both valid (excludes no integer-feasible point)
// and proves the MILP optimum (x = 1) without any branching.
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

void test_gmi_fixture_via_solve_milp() {
    auto lp = read_text(kGmiFixture);
    BabOptions opts;
    opts.max_nodes = 1000;
    opts.cuts_enabled = true;
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
    with_cuts.max_nodes = 1000;
    with_cuts.cuts_enabled = true;
    BabDiagnostics diag_with;
    auto raw_with = sor::search::solve_milp(lp, with_cuts, diag_with);
    const auto r_with = sor::certify::finalize_result(
        std::move(raw_with), sor::search::milp_evidence(diag_with, with_cuts));

    BabOptions without_cuts;
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

int main() {
    test_gmi_separates_and_is_valid();
    test_gmi_fixture_via_solve_milp();
    test_cuts_do_not_change_knapsack_answer();
    test_cut_pool_management();
    test_cut_pool_parallel_penalty_branch();
    return sor::test::finish("test_cuts");
}
