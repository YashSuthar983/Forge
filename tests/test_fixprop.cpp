#include "forge/search/fixprop.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <vector>

using forge::core::Index;
using forge::model::LpProblem;
using forge::search::FixPropDiagnostics;
using forge::search::FixPropOptions;
using forge::search::fix_and_propagate;
using forge::sparse::from_triplets;

// A spent propagation budget must end the entire heuristic. Continuing with
// another dive repeats the same root propagation and can overrun the MILP
// deadline before the search tree starts.
void test_work_budget_ends_portfolio() {
    constexpr Index rows = 64;
    constexpr Index cols = 3;
    std::vector<Index> ri, ci;
    std::vector<double> av;
    for (Index i = 0; i < rows; ++i)
        for (Index j = 0; j < cols; ++j) {
            ri.push_back(i);
            ci.push_back(j);
            av.push_back(1.0);
        }
    LpProblem lp;
    lp.name = "fixprop_budget";
    lp.A = from_triplets(rows, cols, ri, ci, av);
    lp.c.assign(cols, 0.0);
    lp.row_lo.assign(rows, 0.0);
    lp.row_hi.assign(rows, 3.0);
    lp.col_lo.assign(cols, 0.0);
    lp.col_hi.assign(cols, 1.0);
    lp.is_integer.assign(cols, true);

    FixPropOptions o;
    o.time_limit_s = 0.0;
    o.max_dives = 3;
    o.work_limit_nnz_multiple = 1;
    std::vector<double> x;
    FixPropDiagnostics d;
    CHECK(!fix_and_propagate(lp, lp.col_lo, lp.col_hi, nullptr, o, x, d));
    CHECK(d.dives == 1);
    CHECK(d.propagations == 1);
    CHECK(x.empty());
}

int main() {
    test_work_budget_ends_portfolio();
    return forge::test::finish("test_fixprop");
}
