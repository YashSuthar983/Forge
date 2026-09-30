// Duplicate-column merging in the structural presolve: columns with identical
// coefficients, cost and integrality become one column carrying their sum,
// and postsolve splits it back. Unit checks pin the reduction and the split;
// the oracle stress plants duplicates in random MILPs and requires the solved
// (and postsolved) answer to equal exhaustive enumeration of the ORIGINAL.
#include "sor/certify/finalize.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/milp_presolve.hpp"
#include "sor/sparse/csr.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace sor::search;
using sor::core::Status;
using sor::model::LpProblem;
using sor::test::oracle::make_random_milp;
using sor::test::oracle::solve_oracle;

namespace {

// Adds `copies` duplicates of column `j` (same rows/values/cost/bounds/type).
LpProblem with_duplicates(const LpProblem& base, sor::core::Index j, int copies) {
    LpProblem p = base;
    const auto& rp = base.A.pattern.row_ptr();
    const auto& ci = base.A.pattern.col_idx();
    std::vector<sor::core::Index> rows, cols;
    std::vector<double> vals;
    for (sor::core::Index i = 0; i < base.n_rows(); ++i)
        for (auto k = rp[i]; k < rp[i + 1]; ++k) {
            rows.push_back(i); cols.push_back(ci[k]); vals.push_back(base.A.vals[k]);
            if (ci[k] == j)
                for (int q = 0; q < copies; ++q) {
                    rows.push_back(i);
                    cols.push_back(base.n_cols() + q);
                    vals.push_back(base.A.vals[k]);
                }
        }
    for (int q = 0; q < copies; ++q) {
        p.c.push_back(base.c[j]);
        p.col_lo.push_back(base.col_lo[j]);
        p.col_hi.push_back(base.col_hi[j]);
        p.is_integer.push_back(base.is_integer[j]);
        if (!base.col_names.empty()) p.col_names.push_back("dup" + std::to_string(q));
    }
    p.A = sor::sparse::from_triplets(base.n_rows(), base.n_cols() + copies, rows, cols, vals);
    return p;
}

void test_merge_and_postsolve() {
    // min x0 + 2 x1 + 2 x2 + 2 x3   s.t. x0 + x1 + x2 + x3 >= 3.5,
    // x1..x3 identical binaries.
    LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 4, {0, 0, 0, 0}, {0, 1, 2, 3}, {1, 1, 1, 1});
    lp.row_lo = {3.5}; lp.row_hi = {1e30};
    lp.c = {1, 2, 2, 2};
    lp.col_lo = {0, 0, 0, 0}; lp.col_hi = {5, 1, 1, 1};
    lp.is_integer = {false, true, true, true};
    MilpPresolveOptions o;
    o.enabled = true;
    MilpPresolveStats st;
    const auto pre = run_structural_presolve(lp, o, st);
    CHECK(!pre.infeasible);
    CHECK(st.merged_cols == 2 && st.merge_groups == 1);
    CHECK(pre.reduced.n_cols() == 2);
    // The merged column is an integer in [0, 3] with cost 2.
    sor::core::Index merged = -1;
    for (sor::core::Index j = 0; j < pre.reduced.n_cols(); ++j)
        if (pre.reduced.is_integer[j]) merged = j;
    CHECK(merged >= 0);
    if (merged >= 0) {
        CHECK(pre.reduced.col_lo[merged] == 0.0 && pre.reduced.col_hi[merged] == 3.0);
        CHECK(pre.reduced.c[merged] == 2.0);
    }
    // Postsolve splits a value of 2 over the members in order.
    std::vector<double> xr(static_cast<std::size_t>(pre.reduced.n_cols()), 0.0);
    xr[static_cast<std::size_t>(pre.reduced_col[0])] = 1.5;
    xr[static_cast<std::size_t>(pre.reduced_col[1])] = 2.0;
    const auto x = postsolve_point(pre, xr);
    CHECK(x.size() == 4);
    CHECK(x[0] == 1.5 && x[1] == 1.0 && x[2] == 1.0 && x[3] == 0.0);
    CHECK(lp.max_row_violation(x) <= 1e-9);
    // A different cost or type keeps columns apart.
    auto lp2 = lp;
    lp2.c[3] = 2.5;
    MilpPresolveStats st2;
    (void)run_structural_presolve(lp2, o, st2);
    CHECK(st2.merged_cols == 1);
}

BabOptions options(bool merge) {
    BabOptions o;
    o.para_bab.threads = 1;
    o.structural_presolve.enabled = true;
    o.structural_presolve.merge_duplicate_columns = merge;
    o.mip_presolve = o.probing = o.symmetry = false;
    o.time_limit_s = 20.0;
    o.max_nodes = 20000;
    o.gap_tol = o.abs_gap_tol = 1e-9;
    return o;
}

void stress() {
    std::uint64_t merged_models = 0, optimal = 0, models = 0;
    for (std::uint32_t seed = 1; seed <= 300; ++seed) {
        auto base = make_random_milp(seed);
        const auto j = static_cast<sor::core::Index>(seed % static_cast<std::uint32_t>(base.n_cols()));
        const int copies = 1 + static_cast<int>(seed % 3);
        auto lp = with_duplicates(base, j, copies);
        const auto oracle = solve_oracle(lp);
        ++models;
        for (const bool merge : {true, false}) {
            auto o = options(merge);
            BabDiagnostics d;
            auto raw = sor::search::solve_milp(lp, o, d);
            const auto res = sor::certify::finalize_result(
                std::move(raw), sor::search::milp_evidence(d, o));
            if (merge && d.structural_presolve.merged_cols > 0) ++merged_models;
            if (!merge) CHECK(d.structural_presolve.merged_cols == 0);
            if (res.status == Status::Optimal) {
                if (merge) ++optimal;
                CHECK(oracle.feasible);
                if (oracle.feasible) {
                    const bool same = std::fabs(res.objective - oracle.objective) <=
                                      1e-6 * (1.0 + std::fabs(oracle.objective));
                    if (!same)
                        std::cout << "WRONG seed=" << seed << " merge=" << merge << " got "
                                  << res.objective << " oracle " << oracle.objective << '\n';
                    CHECK(same);
                    // The reported point is a feasible point of the ORIGINAL.
                    CHECK(res.x.size() == static_cast<std::size_t>(lp.n_cols()));
                    CHECK(lp.max_row_violation(res.x) <= 1e-6);
                }
            } else if (res.status == Status::Infeasible) {
                CHECK(!oracle.feasible);
            }
        }
    }
    std::cout << "COLUMN_MERGE models=" << models << " merged=" << merged_models
              << " optimal=" << optimal << '\n';
    CHECK(merged_models > 40);
    CHECK(optimal > 50);
}

}  // namespace

int main() {
    test_merge_and_postsolve();
    stress();
    return sor::test::finish("test_column_merge");
}
