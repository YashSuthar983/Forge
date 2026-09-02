#include "sor/presolve/presolve.hpp"
#include "test_helpers.hpp"

using sor::core::Index;
using sor::core::f64;

int main() {
    using sor::model::LpProblem;
    using sor::presolve::presolve_lp;
    using sor::presolve::postsolve;

    // Fixed-column substitution shifts the row bounds before the equality
    // singleton pins the remaining variable.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        p.c = {0.0, 0.0};
        p.row_lo = p.row_hi = {3.0};
        p.col_lo = {2.0, 0.0};
        p.col_hi = {2.0, 10.0};
        const auto map = presolve_lp(p);
        const auto x = postsolve(map, {});
        CHECK(x.size() == 2);
        CHECK_NEAR(x[0], 2.0, 1e-15);
        CHECK_NEAR(x[1], 1.0, 1e-15);
    }

    // A structurally empty bounded column can be removed at its minimizing
    // bound without changing the lifted primal point.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(0, 1, {}, {}, {});
        p.c = {-1.0};
        p.row_lo = {};
        p.row_hi = {};
        p.col_lo = {0.0};
        p.col_hi = {4.0};
        const auto map = presolve_lp(p);
        const auto x = postsolve(map, {});
        CHECK(x.size() == 1);
        CHECK_NEAR(x[0], 4.0, 1e-15);
    }

    return sor::test::finish("test_presolve");
}
