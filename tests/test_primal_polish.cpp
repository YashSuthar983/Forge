#include "sor/search/primal_polish.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <limits>
#include <vector>

using sor::core::Index;
using sor::model::LpProblem;
using sor::search::polish_relaxation_primal;
using sor::sparse::from_triplets;

LpProblem model(bool duplicate_row = false) {
    LpProblem lp;
    lp.name = "free_singleton_polish";
    const Index rows = duplicate_row ? 2 : 1;
    std::vector<Index> ri{0, 0}, ci{0, 1};
    std::vector<double> av{1.0, -1e7};
    if (duplicate_row) {
        ri.push_back(1);
        ci.push_back(0);
        av.push_back(1.0);
    }
    lp.A = from_triplets(rows, 2, ri, ci, av);
    lp.c = {1.0, -1e7};
    lp.row_lo.assign(rows, 0.0);
    lp.row_hi.assign(rows, 0.0);
    if (duplicate_row)
        lp.row_lo[1] = lp.row_hi[1] = 1e7 + 1e-6;
    lp.col_lo = {-std::numeric_limits<double>::infinity(), 1.0};
    lp.col_hi = {std::numeric_limits<double>::infinity(), 1.0};
    lp.is_integer = {false, true};
    return lp;
}

void test_free_singleton_repair_requires_full_certificate() {
    auto lp = model();
    sor::core::RawResult raw;
    raw.proposed_status = sor::core::Status::Optimal;
    raw.x = {1e7 + 1e-6, 1.0};
    raw.y = {1.0};
    sor::engines::SimplexOptions opts;
    opts.primal_feas_tol = 1e-7;
    opts.dual_feas_tol = 1e-7;
    opts.gap_tol = 1e-7;
    sor::engines::SimplexDiagnostics diag;
    diag.primal_residual = lp.max_row_violation(raw.x);
    // The row multiplier is charged in full while an equality is outside
    // primal tolerance; repairing that row must be allowed to clear both
    // apparent primal and dual infeasibility before independent rechecking.
    diag.dual_residual = 1.0;
    diag.dual_bound_finite = true;
    CHECK(diag.primal_residual > opts.primal_feas_tol);
    Index corrected = 0;
    CHECK(polish_relaxation_primal(lp, opts, raw, diag, &corrected));
    CHECK(corrected == 1);
    CHECK(lp.max_row_violation(raw.x) <= opts.primal_feas_tol);
    CHECK(diag.dual_residual <= opts.dual_feas_tol);
    CHECK(diag.gap_rel <= opts.gap_tol);

    // An integer-declared column is continuous in the node relaxation.
    // Incumbent integrality is checked later against the original MILP.
    lp.is_integer[0] = true;
    raw.x = {1e7 + 1e-6, 1.0};
    raw.y = {1.0};
    diag.primal_residual = lp.max_row_violation(raw.x);
    diag.dual_residual = 1.0;
    CHECK(polish_relaxation_primal(lp, opts, raw, diag));
    CHECK(lp.max_row_violation(raw.x) <= opts.primal_feas_tol);

    raw.x = {1e7 + 1e-6, 1.0};
    diag.primal_residual = lp.max_row_violation(raw.x);
    raw.y.clear();
    CHECK(!polish_relaxation_primal(lp, opts, raw, diag));
}

void test_no_repair_without_an_admissible_column() {
    auto lp = model();
    sor::core::RawResult raw;
    raw.proposed_status = sor::core::Status::Optimal;
    raw.x = {1e7 + 1e-6, 1.0};
    raw.y = {1.0};
    sor::engines::SimplexOptions opts;
    sor::engines::SimplexDiagnostics diag;
    diag.primal_residual = lp.max_row_violation(raw.x);
    diag.dual_residual = 0.0;
    diag.dual_bound_finite = true;
    lp.col_lo[0] = raw.x[0];
    CHECK(!polish_relaxation_primal(lp, opts, raw, diag));
    CHECK(raw.x[0] == 1e7 + 1e-6);

    lp = model(true);
    raw.y = {1.0, 0.0};
    diag.primal_residual = lp.max_row_violation(raw.x);
    CHECK(!polish_relaxation_primal(lp, opts, raw, diag));
    CHECK(raw.x[0] == 1e7 + 1e-6);
}

int main() {
    test_free_singleton_repair_requires_full_certificate();
    test_no_repair_without_an_admissible_column();
    return sor::test::finish("test_primal_polish");
}
