#include "sor/presolve/presolve.hpp"
#include "test_helpers.hpp"
#include <vector>

using sor::core::Index;
using sor::core::f64;

int main() {
    using sor::model::LpProblem;
    using sor::presolve::presolve_lp;
    using sor::presolve::postsolve;

    // ---- implied slack: a zero-cost singleton column IS the row's slack ----
    //
    //   min  x0                    (x1 costs nothing and occurs only in row 0)
    //   s.t. x0 + x1 = 5           x1 >= 0, unbounded above
    //        x0 in [0, 10]
    //
    // x1 is the surplus of x0 <= 5 written out by hand. The rule drops it and
    // turns the equality back into that inequality; x1 comes back from the row
    // at postsolve. Opt-in: see the note on presolve_lp.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        p.c = {1.0, 0.0};
        p.row_lo = p.row_hi = {5.0};
        p.col_lo = {0.0, 0.0};
        p.col_hi = {10.0, sor::model::kInf};

        // Off by default, so nothing happens and the model is untouched.
        const auto plain = presolve_lp(p);
        CHECK(plain.problem.n_cols() == 2);
        CHECK(plain.singleton_columns.empty());

        const auto map = presolve_lp(p, /*implied_slack=*/true);
        CHECK(map.singleton_columns.size() == 1);
        // The ROW SURVIVES. That is what distinguishes this from the
        // implied-free variant, and postsolve depends on the distinction: a
        // surviving row keeps the dual the solve produced for it.
        CHECK(!map.singleton_columns[0].row_removed);
        CHECK(map.problem.n_rows() == 1);
        CHECK(map.problem.n_cols() == 1);
        CHECK(map.orig_to_new[1] < 0);          // x1 is gone
        // a*x1 over [0, inf) is [0, inf), so the row keeps its upper end at
        // 5 - 0 and loses its lower end entirely: x0 <= 5.
        CHECK(map.problem.row_hi[0] == 5.0);
        CHECK(map.problem.row_lo[0] <= -sor::model::kInf);
        CHECK(map.problem.c[0] == 1.0);          // objective untouched
        CHECK(map.problem.obj_offset == p.obj_offset);

        // Recovery solves the ORIGINAL equality for x1, exactly.
        for (const f64 v : {0.0, 2.5, 5.0}) {
            const auto x = postsolve(map, {v});
            CHECK(x.size() == 2);
            CHECK_NEAR(x[0], v, 1e-15);
            CHECK_NEAR(x[1], 5.0 - v, 1e-15);
            CHECK(x[1] >= -1e-15);               // inside x1's own bounds
        }
    }

    // A BOXED zero-cost singleton is taken too, and then the row becomes a
    // two-sided range rather than an inequality. Bounds are chosen so neither
    // column is implied free -- otherwise the implied-free rule takes one of
    // them first and this path is never reached.
    //
    //   min  x0                    x0 in [0, 3], x1 in [0, 4] zero-cost
    //   s.t. x0 + x1 = 5
    //
    // a*x1 over [0, 4] is [0, 4], so the row becomes 5-4 <= x0 <= 5-0.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        p.c = {1.0, 0.0};
        p.row_lo = p.row_hi = {5.0};
        p.col_lo = {0.0, 0.0};
        p.col_hi = {3.0, 4.0};
        const auto map = presolve_lp(p, /*implied_slack=*/true);
        CHECK(map.singleton_columns.size() == 1);
        CHECK(!map.singleton_columns[0].row_removed);
        CHECK(map.problem.n_rows() == 1);
        CHECK(map.problem.n_cols() == 1);
        CHECK_NEAR(map.problem.row_lo[0], 1.0, 1e-15);
        CHECK_NEAR(map.problem.row_hi[0], 5.0, 1e-15);
        for (const f64 v : {1.0, 2.0, 3.0}) {
            const auto x = postsolve(map, {v});
            CHECK(x.size() == 2);
            CHECK_NEAR(x[0], v, 1e-15);
            CHECK_NEAR(x[1], 5.0 - v, 1e-15);
            CHECK(x[1] >= -1e-15 && x[1] <= 4.0 + 1e-15);
        }
    }

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

    // A free singleton column in an equality is algebraically removable. The
    // objective substitution must choose the same point and reverse postsolve
    // must reconstruct the eliminated variable, not its fixed-value sentinel.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            1, 2, {0, 0}, {0, 1}, {1.0, 2.0});
        p.c = {3.0, 1.0};
        p.row_lo = p.row_hi = {5.0};
        p.col_lo = {-sor::model::kInf, 0.0};
        p.col_hi = { sor::model::kInf, 2.0};
        const auto map = presolve_lp(p);
        CHECK(map.singleton_columns.size() == 1);
        CHECK(map.stats.singleton_columns_removed == 1);
        CHECK(map.problem.n_rows() == 0);
        CHECK(map.problem.n_cols() == 0);
        CHECK_NEAR(map.problem.obj_offset, 5.0, 1e-15);
        const auto x = postsolve(map, {});
        CHECK_NEAR(x[0], 1.0, 1e-15);
        CHECK_NEAR(x[1], 2.0, 1e-15);
        CHECK_NEAR(p.objective(x), 5.0, 1e-15);
        CHECK_NEAR(map.singleton_columns[0].dual_value, 3.0, 1e-15);
    }

    // Chained eliminations have to unwind backwards: x depends on y, while y
    // remains until z is eliminated from the second row and then optimized as
    // an empty column.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            2, 3, {0, 0, 1, 1}, {0, 1, 1, 2},
            {1.0, 1.0, 1.0, 1.0});
        p.c = {1.0, 0.0, 2.0};
        p.row_lo = p.row_hi = {3.0, 2.0};
        p.col_lo = {-sor::model::kInf, 0.0, -sor::model::kInf};
        p.col_hi = { sor::model::kInf, 2.0,  sor::model::kInf};
        const auto map = presolve_lp(p);
        CHECK(map.singleton_columns.size() == 2);
        CHECK(map.stats.singleton_columns_removed == 2);
        CHECK(map.recovery_steps.size() == 2);
        CHECK(map.recovery_steps[0].kind ==
              sor::presolve::DualRecoveryKind::SingletonColumnElimination);
        CHECK(map.recovery_steps[0].row == 0);
        CHECK(map.recovery_steps[0].col == 0);
        CHECK_NEAR(map.recovery_steps[0].coeff, 1.0, 1e-15);
        CHECK(map.recovery_steps[1].kind ==
              sor::presolve::DualRecoveryKind::SingletonColumnElimination);
        CHECK(map.recovery_steps[1].row == 1);
        CHECK(map.recovery_steps[1].col == 2);
        CHECK_NEAR(map.recovery_steps[1].coeff, 1.0, 1e-15);
        const auto x = postsolve(map, {});
        CHECK_NEAR(x[0] + x[1], 3.0, 1e-15);
        CHECK_NEAR(x[1] + x[2], 2.0, 1e-15);
        CHECK_NEAR(x[0], 1.0, 1e-15);
        CHECK_NEAR(x[1], 2.0, 1e-15);
        CHECK_NEAR(x[2], 0.0, 1e-15);
        CHECK_NEAR(p.objective(x), 1.0, 1e-15);
    }

    // If eliminating either singleton would turn its bounds into a tighter
    // bound on the survivor, this proof-complete subset must decline the
    // transformation. Silently dropping the equality here enlarges the LP.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        p.c = {0.0, 0.0};
        p.row_lo = p.row_hi = {1.5};
        p.col_lo = {0.0, 0.0};
        p.col_hi = {1.0, 1.0};
        const auto map = presolve_lp(p);
        CHECK(map.singleton_columns.empty());
        CHECK(map.problem.n_rows() == 1);
        CHECK(map.problem.n_cols() == 2);
    }

    // An unbounded implied interval must never manufacture an infinite
    // comparison tolerance. Initially x is the only singleton candidate and
    // y's unbounded upper range makes x's lower bound load-bearing. After the
    // singleton inequality tightens y and disappears, y's own lower bound is
    // also load-bearing, so neither side may be eliminated on the next pass.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            2, 2, {0, 0, 1}, {0, 1, 1}, {1.0, 1.0, 1.0});
        p.c = {0.0, 0.0};
        p.row_lo = {0.0, -sor::model::kInf};
        p.row_hi = {0.0, 10.0};
        p.col_lo = {0.0, -0.5};
        p.col_hi = {1.0,  sor::model::kInf};
        const auto map = presolve_lp(p);
        CHECK(map.singleton_columns.empty());
        CHECK(map.orig_to_new[0] >= 0);
        CHECK(map.row_orig_to_new[0] >= 0);
    }

    // A lower row bound equal to the maximum activity forces every live
    // column to the bound that maximizes its signed contribution. Besides the
    // mixed-sign upper/lower choice, keep a second row alive to prove that
    // fixed contributions are shifted out of its bounds.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            2, 4,
            {0, 0, 1, 1, 1, 1}, {0, 1, 0, 1, 2, 3},
            {2.0, -3.0, 1.0, 1.0, 1.0, 1.0});
        p.c = {5.0, -2.0, 1.0, 4.0};
        p.obj_offset = 3.0;
        p.row_lo = {7.0, 4.0};
        p.row_hi = {sor::model::kInf, 8.0};
        p.col_lo = {0.0, -1.0, 0.0, 0.0};
        p.col_hi = {2.0, 3.0, 10.0, 10.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.forcing_rows_removed == 1);
        CHECK(map.stats.forcing_columns_fixed == 2);
        CHECK(map.orig_to_new[0] == -1);
        CHECK(map.orig_to_new[1] == -1);
        CHECK_NEAR(map.fixed_value[0], 2.0, 1e-15);
        CHECK_NEAR(map.fixed_value[1], -1.0, 1e-15);
        CHECK(map.problem.n_rows() == 1);
        CHECK(map.problem.n_cols() == 2);
        CHECK_NEAR(map.problem.obj_offset, 15.0, 1e-15);
        CHECK_NEAR(map.problem.row_lo[0], 3.0, 1e-15);
        CHECK_NEAR(map.problem.row_hi[0], 7.0, 1e-15);
        const auto x = postsolve(map, {1.0, 2.0});
        CHECK_NEAR(x[0], 2.0, 1e-15);
        CHECK_NEAR(x[1], -1.0, 1e-15);
        CHECK_NEAR(x[2], 1.0, 1e-15);
        CHECK_NEAR(x[3], 2.0, 1e-15);
        CHECK(map.recovery_steps.size() == 1);
        CHECK(map.recovery_steps[0].kind ==
              sor::presolve::DualRecoveryKind::ForcingRow);
        CHECK(map.recovery_steps[0].at_max);
    }

    // Symmetric forcing case: an upper row bound equal to minimum activity
    // fixes the positive column low and the negative column high. The other
    // row again makes the fixed-activity shift observable in reduced space.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            2, 4,
            {0, 0, 1, 1, 1, 1}, {0, 1, 0, 1, 2, 3},
            {2.0, -3.0, 1.0, 1.0, 1.0, 1.0});
        p.c = {3.0, 2.0, 1.0, 1.0};
        p.obj_offset = -4.0;
        p.row_lo = {-sor::model::kInf, 5.0};
        p.row_hi = {-19.0, 12.0};
        p.col_lo = {-2.0, -1.0, 0.0, 0.0};
        p.col_hi = {4.0, 5.0, 10.0, 10.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.forcing_rows_removed == 1);
        CHECK(map.stats.forcing_columns_fixed == 2);
        CHECK_NEAR(map.fixed_value[0], -2.0, 1e-15);
        CHECK_NEAR(map.fixed_value[1], 5.0, 1e-15);
        CHECK_NEAR(map.problem.obj_offset, 0.0, 1e-15);
        CHECK(map.problem.n_rows() == 1);
        CHECK(map.problem.n_cols() == 2);
        CHECK_NEAR(map.problem.row_lo[0], 2.0, 1e-15);
        CHECK_NEAR(map.problem.row_hi[0], 9.0, 1e-15);
        CHECK(map.recovery_steps.size() == 1);
        CHECK(!map.recovery_steps[0].at_max);
    }

    // Equality rows are forcing too when their right-hand side is an exact
    // extremum. This guards the genuinely multi-variable path (three live
    // entries), including the negative coefficient's reversed bound choice.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 2.0, -1.0});
        p.c = {1.0, 1.0, 1.0};
        p.row_lo = p.row_hi = {9.0};
        p.col_lo = {0.0, 1.0, -2.0};
        p.col_hi = {1.0, 3.0, 2.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.forcing_rows_removed == 1);
        CHECK(map.stats.forcing_columns_fixed == 3);
        CHECK(map.problem.n_rows() == 0);
        CHECK(map.problem.n_cols() == 0);
        const auto x = postsolve(map, {});
        CHECK_NEAR(x[0], 1.0, 1e-15);
        CHECK_NEAR(x[1], 3.0, 1e-15);
        CHECK_NEAR(x[2], -2.0, 1e-15);
    }

    // Forcing is deliberately exact, not tolerance based. One representable
    // number below the maximum still leaves feasible alternatives and must not
    // remove either the row or its columns.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            1, 2, {0, 0}, {0, 1}, {1.0, 2.0});
        p.c = {0.0, 0.0};
        p.row_lo = {std::nextafter(3.0, 0.0)};
        p.row_hi = {sor::model::kInf};
        p.col_lo = {0.0, 0.0};
        p.col_hi = {1.0, 1.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.forcing_rows_removed == 0);
        CHECK(map.stats.forcing_columns_fixed == 0);
        CHECK(map.problem.n_rows() == 1);
        CHECK(map.problem.n_cols() == 2);
    }

    // An unbounded activity extremum carries no forcing proof, regardless of
    // how large the finite row bound is.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        p.c = {0.0, 0.0};
        p.row_lo = {5.0};
        p.row_hi = {sor::model::kInf};
        p.col_lo = {0.0, 0.0};
        p.col_hi = {sor::model::kInf, 1.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.forcing_rows_removed == 0);
        CHECK(map.stats.forcing_columns_fixed == 0);
        CHECK(map.problem.n_rows() == 1);
        CHECK(map.problem.n_cols() == 2);
    }

    // Cross-kind cascade in one presolve fixed point:
    //   R0 forces x=2,y=3;
    //   R1 then becomes an empty satisfied row;
    //   R2 becomes a singleton equality and pins interior z=4;
    //   R3 tightens w<=3, becomes redundant, and leaves w empty.
    // The chronological recovery journal is the load-bearing part: dual
    // postsolve must replay it in reverse order.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            4, 4,
            {0, 0, 1, 1, 2, 2, 3, 3},
            {0, 1, 0, 1, 0, 2, 2, 3},
            {1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0});
        p.c = {2.0, 0.0, 0.0, -1.0};
        p.obj_offset = 5.0;
        p.row_lo = {5.0, -1.0, 6.0, -sor::model::kInf};
        p.row_hi = {sor::model::kInf, -1.0, 6.0, 7.0};
        p.col_lo = {0.0, 0.0, 0.0, 0.0};
        p.col_hi = {2.0, 3.0, 10.0, 10.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.forcing_rows_removed == 1);
        CHECK(map.stats.forcing_columns_fixed == 2);
        CHECK(map.stats.bounds_tightened == 1);
        CHECK(map.stats.rows_removed == 4);
        CHECK(map.stats.cols_removed == 4);
        CHECK(map.problem.n_rows() == 0);
        CHECK(map.problem.n_cols() == 0);
        CHECK(map.recovery_steps.size() == 3);
        CHECK(map.recovery_steps[0].kind ==
              sor::presolve::DualRecoveryKind::ForcingRow);
        CHECK(map.recovery_steps[1].kind ==
              sor::presolve::DualRecoveryKind::EqualitySingletonFix);
        CHECK(map.recovery_steps[2].kind ==
              sor::presolve::DualRecoveryKind::BoundTightening);
        const auto x = postsolve(map, {});
        CHECK_NEAR(x[0], 2.0, 1e-15);
        CHECK_NEAR(x[1], 3.0, 1e-15);
        CHECK_NEAR(x[2], 4.0, 1e-15);
        CHECK_NEAR(x[3], 3.0, 1e-15);
        CHECK_NEAR(p.objective(x), 6.0, 1e-15);
    }

    // Fixed contributions remain ordinary objective constants for a
    // maximization model; sense conversion belongs to the solver, not to the
    // algebraic presolve substitution.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        p.c = {2.0, -4.0};
        p.obj_offset = 7.0;
        p.maximize = true;
        p.row_lo = {3.0};
        p.row_hi = {sor::model::kInf};
        p.col_lo = {0.0, 0.0};
        p.col_hi = {1.0, 2.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.forcing_rows_removed == 1);
        CHECK(map.stats.forcing_columns_fixed == 2);
        CHECK_NEAR(map.problem.obj_offset, 1.0, 1e-15);
        const auto x = postsolve(map, {});
        CHECK_NEAR(x[0], 1.0, 1e-15);
        CHECK_NEAR(x[1], 2.0, 1e-15);
        CHECK_NEAR(p.objective(x), 1.0, 1e-15);
    }

    // Equality aggregation with a genuinely free pivot.  The pivot appears in
    // two other rows, with multipliers of opposite sign; checking the entire
    // reduced matrix catches sign errors in both coefficient and bound shifts.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            3, 3,
            {0, 0, 0, 1, 1, 2, 2},
            {0, 1, 2, 0, 1, 0, 2},
            {-2.0, 1.0, 1.0, 4.0, 3.0, -6.0, 1.0});
        p.c = {2.0, 5.0, -1.0};
        p.obj_offset = 1.0;
        p.row_lo = {6.0, 10.0, -5.0};
        p.row_hi = {6.0, 20.0, 7.0};
        p.col_lo = {-sor::model::kInf, 0.0, 0.0};
        p.col_hi = {sor::model::kInf, 10.0, 10.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.equality_aggregations == 1);
        CHECK(map.problem.n_rows() == 2);
        CHECK(map.problem.n_cols() == 2);
        CHECK_NEAR(map.problem.obj_offset, -5.0, 1e-15);
        CHECK_NEAR(map.problem.c[0], 6.0, 1e-15);
        CHECK_NEAR(map.problem.c[1], 0.0, 1e-15);
        CHECK(map.problem.A.pattern.row_ptr() ==
              std::vector<sor::core::Offset>({0, 2, 4}));
        CHECK(map.problem.A.pattern.col_idx() == std::vector<Index>({0, 1, 0, 1}));
        CHECK_NEAR(map.problem.A.vals[0], 5.0, 1e-15);
        CHECK_NEAR(map.problem.A.vals[1], 2.0, 1e-15);
        CHECK_NEAR(map.problem.A.vals[2], -3.0, 1e-15);
        CHECK_NEAR(map.problem.A.vals[3], -2.0, 1e-15);
        CHECK_NEAR(map.problem.row_lo[0], 22.0, 1e-15);
        CHECK_NEAR(map.problem.row_hi[0], 32.0, 1e-15);
        CHECK_NEAR(map.problem.row_lo[1], -23.0, 1e-15);
        CHECK_NEAR(map.problem.row_hi[1], -11.0, 1e-15);

        CHECK(map.equality_aggregations.size() == 1);
        const auto& rec = map.equality_aggregations[0];
        CHECK(rec.row == 0);
        CHECK(rec.col == 0);
        CHECK_NEAR(rec.coeff, -2.0, 1e-15);
        CHECK_NEAR(rec.rhs, 6.0, 1e-15);
        CHECK(rec.other_cols == std::vector<Index>({1, 2}));
        CHECK(rec.affected_rows == std::vector<Index>({1, 2}));
        CHECK_NEAR(rec.row_multipliers[0], -2.0, 1e-15);
        CHECK_NEAR(rec.row_multipliers[1], 3.0, 1e-15);
        CHECK_NEAR(rec.dual_value, -1.0, 1e-15);

        const auto x = postsolve(map, {2.0, 4.0});
        CHECK_NEAR(x[0], 0.0, 1e-15);
        CHECK_NEAR(x[1], 2.0, 1e-15);
        CHECK_NEAR(x[2], 4.0, 1e-15);
        CHECK_NEAR(p.objective(x), map.problem.objective({2.0, 4.0}), 1e-15);
    }

    // A bounded pivot is eliminable only when its complete implied interval is
    // inside its declared bounds.  Here x=3-2y lies in [-1,3] subset [-5,5].
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            3, 3,
            {0, 0, 1, 1, 2, 2}, {0, 1, 0, 2, 1, 2},
            {1.0, 2.0, 1.0, 1.0, 1.0, 1.0});
        p.c = {1.0, 0.0, 0.0};
        p.row_lo = {3.0, -sor::model::kInf, -sor::model::kInf};
        p.row_hi = {3.0, 10.0, 11.0};
        p.col_lo = {-5.0, 0.0, 0.0};
        p.col_hi = {5.0, 2.0, 10.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.equality_aggregations == 1);
        CHECK(map.equality_aggregations.size() == 1);
        CHECK(map.equality_aggregations[0].col == 0);
        CHECK(map.problem.n_rows() == 2);
        CHECK(map.problem.n_cols() == 2);
        CHECK_NEAR(map.problem.row_hi[0], 7.0, 1e-15);
        const auto x = postsolve(map, {1.0, 2.0});
        CHECK_NEAR(x[0], 1.0, 1e-15);
    }

    // Neither bounded variable in x+y=1 is implied-free over [0,2]^2, so
    // aggregation (which still requires an implied-free pivot) must decline.
    // Bound-transfer singleton substitution may still remove one column and
    // keep a ranged image of its bounds — that is a different, sound rule.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            3, 3,
            {0, 0, 1, 1, 2, 2}, {0, 1, 0, 2, 1, 2},
            {1.0, 1.0, 1.0, 1.0, 1.0, -1.0});
        p.c = {0.0, 0.0, 0.0};
        p.row_lo = {1.0, -sor::model::kInf, -sor::model::kInf};
        p.row_hi = {1.0, 2.0, 1.0};
        p.col_lo = {0.0, 0.0, 0.0};
        p.col_hi = {2.0, 2.0, 2.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.equality_aggregations == 0);
        CHECK(map.problem.n_rows() == 3);
        CHECK(map.problem.n_cols() == 3);
    }

    // A proportional affected equality cancels exactly.  This exercises both
    // erase-on-zero arithmetic and the subsequent empty-row/empty-column pass.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {1.0, 1.0, 2.0, 2.0});
        p.c = {1.0, 3.0};
        p.row_lo = p.row_hi = {0.0, 0.0};
        p.col_lo = {-sor::model::kInf, 0.0};
        p.col_hi = {sor::model::kInf, 3.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.equality_aggregations == 1);
        CHECK(map.stats.aggregation_fill == 0);
        CHECK(map.problem.n_rows() == 0);
        CHECK(map.problem.n_cols() == 0);
        CHECK(map.equality_aggregations[0].affected_rows ==
              std::vector<Index>({1}));
        CHECK_NEAR(map.equality_aggregations[0].row_multipliers[0], 2.0, 1e-15);
        const auto x = postsolve(map, {});
        CHECK_NEAR(x[0], 0.0, 1e-15);
        CHECK_NEAR(x[1], 0.0, 1e-15);
    }

    // A nominally free pivot below the row- and column-scaled stability floor
    // must not be used.  The other equality coefficient is bounded and not
    // implied-free, so there is no safe alternate pivot.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            3, 3,
            {0, 0, 1, 1, 2, 2}, {0, 1, 0, 2, 1, 2},
            {1e-12, 1.0, 1.0, 1.0, 1.0, 1.0});
        p.c = {0.0, 0.0, 0.0};
        p.row_lo = {1.0, -sor::model::kInf, -sor::model::kInf};
        p.row_hi = {1.0, 2.0, 3.0};
        p.col_lo = {-sor::model::kInf, 0.0, 0.0};
        p.col_hi = {sor::model::kInf, 2.0, 2.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.equality_aggregations == 0);
        CHECK(map.problem.n_rows() == 3);
        CHECK(map.problem.n_cols() == 3);
    }

    // Width guard: a 33-entry equality is deliberately outside the reversible
    // aggregation kernel even though all of its variables are otherwise safe.
    {
        constexpr Index n = 33;
        std::vector<Index> rows, cols;
        std::vector<f64> vals;
        for (Index j = 0; j < n; ++j) {
            rows.push_back(0); cols.push_back(j); vals.push_back(1.0);
            rows.push_back(1); cols.push_back(j); vals.push_back(1.0);
        }
        LpProblem p;
        p.A = sor::sparse::from_triplets(2, n, rows, cols, vals);
        p.c.assign(n, 0.0);
        p.row_lo = {0.0, -1.0};
        p.row_hi = {0.0, 1.0};
        p.col_lo.assign(n, -sor::model::kInf);
        p.col_hi.assign(n, sor::model::kInf);

        const auto map = presolve_lp(p);
        CHECK(map.stats.equality_aggregations == 0);
        CHECK(map.problem.n_rows() == 2);
        CHECK(map.problem.n_cols() == n);
    }

    // Fill guard: a legal-width (32) equality would add 600 structural entries
    // across twenty affected rows, beyond both the per-step and global budget.
    {
        constexpr Index ny = 31;
        constexpr Index naffected = 20;
        constexpr Index n = 1 + ny + naffected;
        std::vector<Index> rows, cols;
        std::vector<f64> vals;
        for (Index j = 0; j <= ny; ++j) {
            rows.push_back(0); cols.push_back(j); vals.push_back(1.0);
        }
        for (Index h = 0; h < naffected; ++h) {
            rows.push_back(1 + h); cols.push_back(0); vals.push_back(1.0);
            rows.push_back(1 + h); cols.push_back(1 + ny + h); vals.push_back(1.0);
        }
        LpProblem p;
        p.A = sor::sparse::from_triplets(1 + naffected, n, rows, cols, vals);
        p.c.assign(n, 0.0);
        p.row_lo.assign(1 + naffected, 0.0);
        p.row_hi.assign(1 + naffected, 1.0);
        p.row_lo[0] = p.row_hi[0] = 0.0;
        p.col_lo.assign(n, 0.0);
        p.col_hi.assign(n, 1.0);
        p.col_lo[0] = -sor::model::kInf;
        p.col_hi[0] = sor::model::kInf;

        const auto map = presolve_lp(p);
        CHECK(map.stats.equality_aggregations == 0);
        CHECK(map.stats.aggregation_fill == 0);
        CHECK(map.problem.n_rows() == 1 + naffected);
        CHECK(map.problem.n_cols() == n);
    }

    // Two equalities form a dependency chain.  x is removed first; its rewrite
    // creates z-y=1, then z is removed.  Reverse postsolve must recover z before
    // x or the reconstructed point is wrong.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            3, 4,
            {0, 0, 1, 1, 2, 2}, {0, 2, 0, 1, 1, 3},
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
        p.c = {0.0, 0.0, 0.0, 0.0};
        p.row_lo = {3.0, 4.0, 2.0};
        p.row_hi = {3.0, 4.0, 8.0};
        p.col_lo = {-sor::model::kInf, -sor::model::kInf, 0.0, 0.0};
        p.col_hi = {sor::model::kInf, sor::model::kInf, 2.0, 10.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.equality_aggregations == 2);
        CHECK(map.equality_aggregations.size() == 2);
        CHECK(map.equality_aggregations[0].row == 0);
        CHECK(map.equality_aggregations[0].col == 0);
        CHECK(map.equality_aggregations[1].row == 1);
        CHECK(map.equality_aggregations[1].col == 1);
        CHECK(map.problem.n_rows() == 1);
        CHECK(map.problem.n_cols() == 2);
        CHECK_NEAR(map.problem.row_lo[0], 1.0, 1e-15);
        CHECK_NEAR(map.problem.row_hi[0], 7.0, 1e-15);
        const auto x = postsolve(map, {1.0, 2.0});
        CHECK_NEAR(x[0], 2.0, 1e-15);
        CHECK_NEAR(x[1], 2.0, 1e-15);
        CHECK_NEAR(x[2], 1.0, 1e-15);
        CHECK_NEAR(x[3], 2.0, 1e-15);
    }

    // Objective substitution is algebraic for maximization too: x=3-y turns
    // 5+2x+y into 11-y, without an accidental objective-sense sign flip.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(
            2, 3, {0, 0, 1, 1}, {0, 1, 0, 2}, {1.0, 1.0, 1.0, 1.0});
        p.c = {2.0, 1.0, 0.0};
        p.obj_offset = 5.0;
        p.maximize = true;
        p.row_lo = {3.0, -sor::model::kInf};
        p.row_hi = {3.0, 5.0};
        p.col_lo = {-sor::model::kInf, 0.0, 0.0};
        p.col_hi = {sor::model::kInf, 2.0, 10.0};

        const auto map = presolve_lp(p);
        CHECK(map.stats.equality_aggregations == 1);
        CHECK(map.problem.maximize);
        CHECK_NEAR(map.problem.obj_offset, 11.0, 1e-15);
        CHECK_NEAR(map.problem.c[0], -1.0, 1e-15);
        const auto x = postsolve(map, {1.0, 2.0});
        CHECK_NEAR(x[0], 2.0, 1e-15);
        CHECK_NEAR(p.objective(x), 10.0, 1e-15);
        CHECK_NEAR(map.problem.objective({1.0, 2.0}), 10.0, 1e-15);
    }

    return sor::test::finish("test_presolve");
}
