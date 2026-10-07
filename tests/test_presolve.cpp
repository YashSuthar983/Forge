#include "sor/presolve/presolve.hpp"
#include "test_helpers.hpp"

#include <chrono>
#include <cstdint>
#include <string>
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
    // keep a ranged image of its bounds - that is a different, sound rule.
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

    // ---- PresolveOutcome: the interface the LP Auto layer consumes ----
    //
    // A boolean "did it reduce" cannot express the outcomes that change what
    // the caller does next. These are the ones that do.
    {
        using sor::presolve::PresolveOptions;
        using sor::presolve::PresolveStatus;

        // Ordinary reduction. Stats must describe the work honestly: shapes on
        // both sides, and a pass count that actually ran.
        sor::model::LpProblem p;
        p.A = sor::sparse::from_triplets(2, 3, {0, 0, 1, 1}, {0, 1, 1, 2},
                                         {1.0, 1.0, 1.0, 1.0});
        p.c = {1.0, 1.0, 1.0};
        p.row_lo = {1.0, -sor::model::kInf};
        p.row_hi = {1.0, 4.0};
        p.col_lo = {0.0, 0.0, 0.0};
        p.col_hi = {sor::model::kInf, sor::model::kInf, sor::model::kInf};

        const auto ok = sor::presolve::presolve(p);
        CHECK(ok.status == PresolveStatus::Reduced);
        CHECK(ok.reduced());
        CHECK(ok.stats().original_rows == 2);
        CHECK(ok.stats().original_cols == 3);
        CHECK(ok.stats().original_nnz == 4);
        CHECK(ok.stats().passes >= 1);
        CHECK(ok.stats().reduced_rows <= 2);
        CHECK(ok.stats().reduced_cols <= 3);
        CHECK(ok.stats().elapsed_ms >= 0.0);

        // Disabled presolve must still return a usable, postsolvable identity
        // map, so a caller needs no second code path for it.
        PresolveOptions off;
        off.enabled = false;
        const auto passthrough = sor::presolve::presolve(p, off);
        CHECK(passthrough.status == PresolveStatus::Reduced);
        CHECK(passthrough.map.problem.n_rows() == 2);
        CHECK(passthrough.map.problem.n_cols() == 3);
        CHECK(passthrough.stats().reduced_nnz == 4);
        CHECK(passthrough.stats().passes == 0);
        const auto lifted = postsolve(passthrough.map, {1.0, 2.0, 3.0});
        CHECK(lifted.size() == 3);
        CHECK_NEAR(lifted[0], 1.0, 1e-15);
        CHECK_NEAR(lifted[2], 3.0, 1e-15);

        // A column whose bounds cross is infeasible, and presolve must SAY so
        // rather than hand an engine a problem to rediscover it on. The witness
        // is in ORIGINAL index space.
        sor::model::LpProblem bad;
        bad.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        bad.c = {1.0, 1.0};
        bad.row_lo = {-sor::model::kInf};
        bad.row_hi = {10.0};
        bad.col_lo = {5.0, 0.0};
        bad.col_hi = {1.0, 1.0};          // x0 in [5, 1] -- empty
        const auto infeas = sor::presolve::presolve(bad);
        CHECK(infeas.status == PresolveStatus::Infeasible);
        CHECK(!infeas.reduced());
        CHECK(infeas.witness_col == 0);
        CHECK(!infeas.reason.empty());

        // A nonzero-cost empty column with an improving infinite bound is a
        // presolve proof of unboundedness, not a column to leave behind.
        sor::model::LpProblem unbounded;
        unbounded.A = sor::sparse::from_triplets(1, 1, {}, {}, {});
        unbounded.c = {-1.0};
        unbounded.row_lo = {-sor::model::kInf};
        unbounded.row_hi = {sor::model::kInf};
        unbounded.col_lo = {0.0};
        unbounded.col_hi = {sor::model::kInf};
        const auto unb = sor::presolve::presolve(unbounded);
        CHECK(unb.status == PresolveStatus::Unbounded);
        CHECK(unb.witness_col == 0);
        CHECK(!unb.reason.empty());

        // Fixed-column substitution can make a row empty and infeasible.
        sor::model::LpProblem empty_bad;
        empty_bad.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {1.0});
        empty_bad.c = {0.0};
        empty_bad.row_lo = {1.0};
        empty_bad.row_hi = {1.0};
        empty_bad.col_lo = {0.0};
        empty_bad.col_hi = {0.0};
        const auto empty_infeas = sor::presolve::presolve(empty_bad);
        CHECK(empty_infeas.status == PresolveStatus::Infeasible);
        CHECK(empty_infeas.witness_row == 0);

        // The compatibility wrapper still works and still discards the status.
        const auto legacy = presolve_lp(p);
        CHECK(legacy.problem.n_cols() <= 3);

        CHECK(std::string(sor::presolve::to_string(PresolveStatus::Infeasible))
              == "Infeasible");
        CHECK(std::string(sor::presolve::to_string(PresolveStatus::Reduced))
              == "Reduced");
    }

    // ---- recover_solution: primal, dual, and basis must all validate ----
    {
        using sor::presolve::PresolveRecoveryOptions;
        using sor::presolve::PresolveReducedSolve;
        using sor::presolve::recover_solution;

        LpProblem p;
        p.A = sor::sparse::from_triplets(
            1, 2, {0, 0}, {0, 1}, {1.0, 2.0});
        p.c = {3.0, 1.0};
        p.row_lo = p.row_hi = {5.0};
        p.col_lo = {-sor::model::kInf, 0.0};
        p.col_hi = { sor::model::kInf, 2.0};
        const auto map = presolve_lp(p);
        CHECK(map.stats.singleton_columns_removed == 1);

        PresolveReducedSolve rs;
        rs.x = {};
        rs.y = {};
        PresolveRecoveryOptions ropts;
        ropts.gap_tol = 1e-9;
        const auto ok = recover_solution(p, map, rs, ropts);
        CHECK(ok.validated);
        CHECK(ok.raw.x.size() == 2);
        CHECK_NEAR(ok.raw.x[0], 1.0, 1e-9);
        CHECK_NEAR(ok.raw.x[1], 2.0, 1e-9);
        CHECK(ok.evidence.max_primal_violation <= 1e-9);
        CHECK(ok.basis.basic.size() == 1);
        CHECK(ok.basis.basic[0] == 0);

        auto tampered = ok.raw.x;
        tampered[0] += 1.0;
        CHECK(p.max_row_violation(tampered) > ropts.primal_feas_tol);
    }

    // ---- lifted basis after singleton-row bound tightening ----
    //
    //   min  -x0 + x1
    //   s.t. 2 x0      <= 4        (singleton: becomes x0 <= 2, row removed)
    //          x0 + x1 >= 3
    //        x0 >= 0 (no upper bound), x1 in [0, 10]
    //
    // The reduced optimum has x0 nonbasic at the tightened bound 2. The
    // original model has no such bound, so the lifted basis must make x0
    // basic and the singleton row nonbasic at its upper side. Leaving the
    // row's logical basic names a basis whose point is x0 = +inf.
    {
        using sor::presolve::PostsolveNonbasicStatus;
        using sor::presolve::PresolveRecoveryOptions;
        using sor::presolve::PresolveReducedSolve;
        using sor::presolve::recover_solution;

        LpProblem p;
        p.A = sor::sparse::from_triplets(
            2, 2, {0, 1, 1}, {0, 0, 1}, {2.0, 1.0, 1.0});
        p.c = {-1.0, 1.0};
        p.row_lo = {-sor::model::kInf, 3.0};
        p.row_hi = {4.0, sor::model::kInf};
        p.col_lo = {0.0, 0.0};
        p.col_hi = {sor::model::kInf, 10.0};
        const auto map = presolve_lp(p);
        CHECK(map.stats.bounds_tightened == 1);
        CHECK(map.problem.n_rows() == 1);
        CHECK(map.problem.n_cols() == 2);
        if (map.problem.n_rows() == 1 && map.problem.n_cols() == 2) {
            CHECK(map.problem.col_hi[0] == 2.0);
            PresolveReducedSolve rs;
            rs.x = {2.0, 1.0};
            rs.y = {1.0};
            rs.has_basis = true;
            rs.basis.n_struct = 2;
            rs.basis.basic = {1};
            rs.basis.status = {PostsolveNonbasicStatus::AtUpper,
                               PostsolveNonbasicStatus::Basic,
                               PostsolveNonbasicStatus::AtLower};
            PresolveRecoveryOptions ropts;
            ropts.gap_tol = 1e-9;
            const auto lifted = recover_solution(p, map, rs, ropts);
            CHECK(lifted.validated);
            CHECK_NEAR(lifted.raw.x[0], 2.0, 1e-12);
            CHECK_NEAR(lifted.raw.x[1], 1.0, 1e-12);
            CHECK(lifted.basis.basic.size() == 2);
            CHECK(lifted.basis.basic[0] == 0);
            CHECK(lifted.basis.basic[1] == 1);
            CHECK(lifted.basis.status[0] == PostsolveNonbasicStatus::Basic);
            CHECK(lifted.basis.status[1] == PostsolveNonbasicStatus::Basic);
            CHECK(lifted.basis.status[2] == PostsolveNonbasicStatus::AtUpper);
            CHECK(lifted.basis.status[3] == PostsolveNonbasicStatus::AtLower);
        }
    }

    // ---- forcing-row multiplier after an earlier cost substitution ----
    //
    //   min  s
    //   s.t. s + 2 j     = 3       (E: s is a free singleton column)
    //            j + k  <= 0       (F: forcing once k >= 0 is known)
    //                k  >= 0       (S: singleton row)
    //        s free, j >= 0, k in [-1, 5]
    //
    // Pass 1 eliminates s through E, which moves cost -2 onto j, and tightens
    // k >= 0 from S. Pass 2 then finds F forcing and fixes j = k = 0. F's
    // multiplier must make j dual feasible for the cost j carries at that
    // stage (-2, so y_F <= -2); measured against j's original cost 0 it comes
    // out as 0 and j is left with reduced cost -2 at its lower bound.
    {
        using sor::presolve::PresolveRecoveryOptions;
        using sor::presolve::PresolveReducedSolve;
        using sor::presolve::recover_solution;

        LpProblem p;
        p.A = sor::sparse::from_triplets(
            3, 3, {0, 0, 1, 1, 2}, {0, 1, 1, 2, 2}, {1.0, 2.0, 1.0, 1.0, 1.0});
        p.c = {1.0, 0.0, 0.0};
        p.row_lo = {3.0, -sor::model::kInf, 0.0};
        p.row_hi = {3.0, 0.0, sor::model::kInf};
        p.col_lo = {-sor::model::kInf, 0.0, -1.0};
        p.col_hi = { sor::model::kInf, sor::model::kInf, 5.0};
        const auto map = presolve_lp(p);
        CHECK(map.stats.singleton_columns_removed == 1);
        CHECK(map.stats.forcing_rows_removed == 1);
        CHECK(map.problem.n_cols() == 0);
        if (map.problem.n_cols() == 0 && map.problem.n_rows() == 0) {
            PresolveReducedSolve rs;
            PresolveRecoveryOptions ropts;
            ropts.gap_tol = 1e-9;
            const auto lifted = recover_solution(p, map, rs, ropts);
            CHECK_NEAR(lifted.raw.x[0], 3.0, 1e-12);
            CHECK_NEAR(lifted.raw.x[1], 0.0, 1e-12);
            CHECK_NEAR(lifted.raw.x[2], 0.0, 1e-12);
            CHECK(lifted.evidence.max_dual_violation <= 1e-9);
            CHECK(lifted.raw.y.size() == 3);
            CHECK(lifted.raw.y[1] <= -2.0 + 1e-9);
            CHECK(lifted.validated);
        }
    }

    // ---- Presolve v2 (live CSR/CSC queue driver) ----
    {
        using sor::presolve::PresolveOptions;
        using sor::presolve::PresolveReducedSolve;
        using sor::presolve::PresolveStatus;
        using sor::presolve::recover_solution;

        PresolveOptions v2;
        v2.live_reductions = true;

        // Dual fixing: positive cost, finite lower, no down-lock on an empty column.
        {
            LpProblem p;
            p.A = sor::sparse::from_triplets(0, 1, {}, {}, {});
            p.c = {4.0};
            p.col_lo = {2.0};
            p.col_hi = {9.0};
            const auto out = sor::presolve::presolve(p, v2);
            CHECK(out.status == PresolveStatus::Solved);
            CHECK(out.map.problem.n_cols() == 0);
            const auto x = postsolve(out.map, {});
            CHECK_NEAR(x[0], 2.0, 1e-15);
        }

        // A zero-cost semi-bounded slack next to a boxed column:
        //   min x0  s.t.  x0 + x1 = 5,  x0 in [0, 10],  x1 in [0, inf).
        // The doubleton eliminates x0 and transfers its box onto the slack
        // (x1 in [0, 5], cost -1 after substitution). The lift of the reduced
        // optimum x1 = 5 must be the original optimum x0 = 0 with exact duals
        // and the basis {x1}: x1 sits at a transferred bound, so x0 is the
        // nonbasic column. (These models used to assert that no reduction
        // happens at all; the transfer is now recovered.)
        {
            using sor::presolve::PostsolveNonbasicStatus;
            using sor::presolve::PresolveRecoveryOptions;
            LpProblem p;
            p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
            p.c = {1.0, 0.0};
            p.row_lo = p.row_hi = {5.0};
            p.col_lo = {0.0, 0.0};
            p.col_hi = {10.0, sor::model::kInf};
            const auto plain = sor::presolve::presolve(p);
            const auto out = sor::presolve::presolve(p, v2);
            CHECK(out.stats().passes >= plain.stats().passes);
            CHECK(out.stats().doubleton_substitutions == 1);
            CHECK(out.map.problem.n_rows() == 0);
            // x1 may also be dual-fixed at 5 once its row is gone.
            CHECK(out.map.problem.n_cols() <= 1);
            if (out.map.problem.n_rows() == 0 && out.map.problem.n_cols() <= 1) {
                PresolveReducedSolve rs;
                rs.has_basis = true;
                if (out.map.problem.n_cols() == 1) {
                    CHECK(out.map.problem.col_lo[0] == 0.0);
                    CHECK(out.map.problem.col_hi[0] == 5.0);
                    rs.x = {5.0};
                    rs.basis.n_struct = 1;
                    rs.basis.status = {PostsolveNonbasicStatus::AtUpper};
                }
                PresolveRecoveryOptions ropts;
                ropts.gap_tol = 1e-9;
                const auto rec = recover_solution(p, out.map, rs, ropts);
                CHECK(rec.validated);
                CHECK_NEAR(rec.raw.x[0], 0.0, 1e-15);
                CHECK_NEAR(rec.raw.x[1], 5.0, 1e-15);
                CHECK(rec.evidence.max_dual_violation <= 1e-15);
                CHECK(rec.basis.basic.size() == 1 && rec.basis.basic[0] == 1);
                CHECK(rec.basis.status[0] == PostsolveNonbasicStatus::AtLower);
                CHECK(rec.basis.status[1] == PostsolveNonbasicStatus::Basic);
            }
        }

        // Seeded small LP: lift a reduced candidate and validate primal+dual.
        {
            LpProblem p;
            p.A = sor::sparse::from_triplets(
                2, 3, {0, 0, 1, 1}, {0, 1, 0, 2}, {1.0, 1.0, 2.0, 1.0});
            p.c = {1.0, 2.0, 0.0};
            p.row_lo = {0.0, 1.0};
            p.row_hi = {2.0, 3.0};
            p.col_lo = {0.0, 0.0, 0.0};
            p.col_hi = {2.0, 2.0, 2.0};
            const auto out = sor::presolve::presolve(p, v2);
            CHECK(out.reduced());
            PresolveReducedSolve rs;
            rs.x = {1.0, 1.0, 1.0};
            const auto ok = recover_solution(p, out.map, rs);
            CHECK(ok.evidence.max_primal_violation <= 1e-9);
        }

        // Implied bounds tighten finite columns when explicitly enabled.
        // Zero costs keep the other live rules out: with c > 0 nothing locks
        // the columns downward and dual fixing removes both; with c < 0 the
        // row is cost-tight (x0 alone would pass 3), becomes an equation and
        // the doubleton substitution removes x0.
        {
            PresolveOptions ib = v2;
            ib.implied_bounds = true;
            LpProblem p;
            p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
            p.c = {0.0, 0.0};
            p.row_lo = {-sor::model::kInf};
            p.row_hi = {3.0};
            p.col_lo = {0.0, 0.0};
            p.col_hi = {10.0, 10.0};
            const auto out = sor::presolve::presolve(p, ib);
            CHECK(out.reduced());
            CHECK(out.map.problem.n_cols() == 2);
            CHECK(out.stats().bounds_tightened >= 1);
            CHECK(out.map.problem.col_hi[0] <= 3.0 + 1e-12);
            CHECK(out.map.problem.col_hi[1] <= 3.0 + 1e-12);
        }

        // Dominated column: looser parallel column removed (opt-in). A >=
        // row locks both columns downward, so dual fixing (c > 0) leaves
        // them, and the row is not cost-tight (4 - 10 - 0 < 0).
        {
            PresolveOptions dom = v2;
            dom.dominated_columns = true;
            LpProblem p;
            p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
            p.c = {1.0, 1.0};
            p.row_lo = {4.0};
            p.row_hi = {sor::model::kInf};
            p.col_lo = {0.0, 0.0};
            p.col_hi = {sor::model::kInf, 10.0};
            const auto out = sor::presolve::presolve(p, dom);
            CHECK(out.stats().dominated_columns_removed >= 1);
            bool saw_dom = false;
            for (const auto& step : out.map.recovery_steps)
                if (step.kind == sor::presolve::DualRecoveryKind::DominatedColumn)
                    saw_dom = true;
            CHECK(saw_dom);
            PresolveReducedSolve rs;
            rs.x.assign(static_cast<std::size_t>(out.map.problem.n_cols()), 4.0);
            rs.y.assign(static_cast<std::size_t>(out.map.problem.n_rows()), 1.0);
            const auto rec = recover_solution(p, out.map, rs);
            CHECK(rec.evidence.max_primal_violation <= 1e-9);
        }
        // The same model as above with the rule off: nothing is removed.
        // (x0 must be the unbounded receiver, or the rule could not fire even
        // when enabled and this would test nothing.)
        {
            PresolveOptions dom = v2;
            dom.dominated_columns = false;
            LpProblem p;
            p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
            p.c = {1.0, 1.0};
            p.row_lo = {4.0};
            p.row_hi = {sor::model::kInf};
            p.col_lo = {0.0, 0.0};
            p.col_hi = {sor::model::kInf, 10.0};
            const auto out = sor::presolve::presolve(p, dom);
            CHECK(out.stats().dominated_columns_removed == 0);
            CHECK(out.map.problem.n_cols() == 2);
        }

        // Parallel rows: tighter duplicate survives; journal + primal lift.
        // Negative costs: the <= rows lock the columns upward (with c > 0
        // nothing locks them downward and dual fixing empties both rows).
        {
            PresolveOptions par = v2;
            par.parallel_rows = true;
            LpProblem p;
            p.A = sor::sparse::from_triplets(
                2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {1.0, 1.0, 1.0, 1.0});
            p.c = {-1.0, -1.0};
            p.row_lo = {-sor::model::kInf, -sor::model::kInf};
            p.row_hi = {5.0, 3.0};
            p.col_lo = {0.0, 0.0};
            p.col_hi = {10.0, 10.0};
            const auto out = sor::presolve::presolve(p, par);
            CHECK(out.stats().duplicate_rows_merged >= 1);
            CHECK(out.map.problem.n_rows() == 1);
            bool saw_row = false;
            for (const auto& step : out.map.recovery_steps)
                if (step.kind == sor::presolve::DualRecoveryKind::ParallelRowMerge)
                    saw_row = true;
            CHECK(saw_row);
            PresolveReducedSolve rs;
            rs.x = {3.0, 0.0};
            rs.y = {1.0};
            const auto rec = recover_solution(p, out.map, rs);
            CHECK(rec.evidence.max_primal_violation <= 1e-9);
            CHECK_NEAR(rec.raw.x[0] + rec.raw.x[1], 3.0, 1e-9);
        }

        // Parallel rows: the multiplier and the tight logical belong to the
        // row that supplied the active merged side.
        //
        //   min -x0 - x1   s.t.  R0:  x0 + x1 <= 4   (kept)
        //                        R1: s(x0 + x1) in R1's sides (merged)
        //   x in [0, 10]
        //
        // With R1 = 2x0 + 2x1 <= 6 (or -2x0 - 2x1 >= -6) the merged row is
        // x0 + x1 <= 3 and its reduced multiplier is -1. R0 is then slack at
        // the optimum, so the lift must give y_R0 = 0 and y_R1 = -1/s, with
        // R1's logical nonbasic and R0's basic. Keeping y' on R0 (the old
        // lift) leaves a multiplier of 1 on a slack row. When R0's own bound
        // is the tighter one, nothing moves.
        {
            using sor::presolve::PostsolveNonbasicStatus;
            using sor::presolve::PresolveRecoveryOptions;
            struct Case { f64 a1; f64 lo1; f64 hi1; f64 merged_hi; f64 y0; f64 y1;
                          PostsolveNonbasicStatus s0, s1; };
            const f64 inf = sor::model::kInf;
            const Case cases[] = {
                {2.0, -inf, 6.0, 3.0, 0.0, -0.5,
                 PostsolveNonbasicStatus::Basic, PostsolveNonbasicStatus::AtUpper},
                {-2.0, -6.0, inf, 3.0, 0.0, 0.5,
                 PostsolveNonbasicStatus::Basic, PostsolveNonbasicStatus::AtLower},
                {2.0, -inf, 10.0, 4.0, -1.0, 0.0,
                 PostsolveNonbasicStatus::AtUpper, PostsolveNonbasicStatus::Basic},
            };
            for (const auto& cs : cases) {
                PresolveOptions par = v2;
                par.parallel_rows = true;
                LpProblem p;
                p.A = sor::sparse::from_triplets(
                    2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {1.0, 1.0, cs.a1, cs.a1});
                p.c = {-1.0, -1.0};
                p.row_lo = {-inf, cs.lo1};
                p.row_hi = {4.0, cs.hi1};
                p.col_lo = {0.0, 0.0};
                p.col_hi = {10.0, 10.0};
                const auto out = sor::presolve::presolve(p, par);
                CHECK(out.stats().duplicate_rows_merged == 1);
                CHECK(out.map.problem.n_rows() == 1);
                CHECK(out.map.problem.n_cols() == 2);
                if (out.map.problem.n_rows() != 1 || out.map.problem.n_cols() != 2) continue;
                CHECK(out.map.problem.row_hi[0] == cs.merged_hi);
                PresolveReducedSolve rs;
                rs.x = {cs.merged_hi, 0.0};
                rs.y = {-1.0};
                rs.has_basis = true;
                rs.basis.n_struct = 2;
                rs.basis.basic = {0};
                rs.basis.status = {PostsolveNonbasicStatus::Basic,
                                   PostsolveNonbasicStatus::AtLower,
                                   PostsolveNonbasicStatus::AtUpper};
                PresolveRecoveryOptions ropts;
                ropts.gap_tol = 1e-9;
                const auto rec = recover_solution(p, out.map, rs, ropts);
                CHECK(rec.validated);
                CHECK(rec.evidence.max_dual_violation <= 1e-12);
                CHECK(rec.raw.y.size() == 2);
                if (rec.raw.y.size() == 2) {
                    CHECK_NEAR(rec.raw.y[0], cs.y0, 1e-12);
                    CHECK_NEAR(rec.raw.y[1], cs.y1, 1e-12);
                }
                CHECK(rec.basis.status.size() == 4);
                if (rec.basis.status.size() == 4) {
                    CHECK(rec.basis.status[0] == PostsolveNonbasicStatus::Basic);
                    CHECK(rec.basis.status[2] == cs.s0);
                    CHECK(rec.basis.status[3] == cs.s1);
                    CHECK(rec.basis.basic[0] == 0);
                }
            }
        }

        // Parallel columns: proportional costs required; full solve + lift.
        // Two rows, so neither column is a singleton (a lone inequality is
        // cost-tight here and the doubleton would take it), and negative
        // costs against R0's upward lock keep dual fixing out.
        {
            PresolveOptions par = v2;
            par.parallel_columns = true;
            LpProblem p;
            p.A = sor::sparse::from_triplets(
                2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {1.0, 2.0, 1.0, 2.0});
            p.c = {-1.0, -2.0};
            p.row_lo = {-sor::model::kInf, 1.0};
            p.row_hi = {10.0, sor::model::kInf};
            p.col_lo = {0.0, 0.0};
            p.col_hi = {5.0, 10.0};
            const auto out = sor::presolve::presolve(p, par);
            CHECK(out.stats().duplicate_columns_merged >= 1);
            bool saw_col = false;
            for (const auto& step : out.map.recovery_steps)
                if (step.kind == sor::presolve::DualRecoveryKind::ParallelColumnMerge)
                    saw_col = true;
            CHECK(saw_col);
            // The merged column z = x0 + 2 x1 in [0, 25]; z = 5 meets both rows.
            PresolveReducedSolve rs;
            rs.x.assign(static_cast<std::size_t>(out.map.problem.n_cols()), 5.0);
            rs.y = {};
            const auto rec = recover_solution(p, out.map, rs);
            CHECK(rec.evidence.max_primal_violation <= 1e-9);
        }

        // F4: aggregate costs and nonzero lower bounds must preserve the
        // original optimum (the audit counterexample has optimum 1).
        {
            PresolveOptions par = v2;
            par.parallel_columns = true;
            LpProblem p;
            p.A = sor::sparse::from_triplets(1, 2, {0,0}, {0,1}, {1.0,1.0});
            p.c = {1.0,1.0}; p.col_lo = {0.0,1.0}; p.col_hi = {2.0,2.0};
            p.row_lo = {1.0}; p.row_hi = {sor::model::kInf};
            const auto out = sor::presolve::presolve(p, par);
            std::vector<double> reduced = out.map.problem.col_lo;
            const auto lifted = postsolve(out.map, reduced);
            CHECK(p.max_bound_violation(lifted) <= 1e-12);
            CHECK(p.max_row_violation(lifted) <= 1e-12);
            CHECK_NEAR(p.objective(lifted), 1.0, 1e-12);
            CHECK_NEAR(out.map.problem.objective(reduced), 1.0, 1e-12);
        }
        // A receiver with finite capacity cannot absorb a dominated column.
        {
            PresolveOptions dom = v2; dom.dominated_columns = true;
            LpProblem p;
            p.A = sor::sparse::from_triplets(1,2,{0,0},{0,1},{1.0,1.0});
            // x1 <= 3, not 2: with x1 <= 2 the row is cost-tight (x0 = 2 - x1
            // >= 0 always) and the doubleton removes x0 before this rule.
            // The pair has the same support, ratio 1 and c0 * 1 <= c1, so
            // the only thing refusing it is x0's finite capacity.
            p.c = {1.0,2.0}; p.col_lo = {0.0,0.0}; p.col_hi = {1.0,3.0};
            p.row_lo = {2.0}; p.row_hi = {sor::model::kInf};
            const auto out = sor::presolve::presolve(p, dom);
            CHECK(out.stats().dominated_columns_removed == 0);
        }
        // Negative proportional rows reverse lower/upper endpoints.
        {
            PresolveOptions par = v2; par.parallel_rows = true;
            LpProblem p;
            p.A = sor::sparse::from_triplets(2,2,{0,0,1,1},{0,1,0,1},{1.0,1.0,-1.0,-1.0});
            p.c = {1.0,1.0}; p.col_lo = {0.0,0.0}; p.col_hi = {10.0,10.0};
            p.row_lo = {1.0,-3.0}; p.row_hi = {5.0,-2.0};
            const auto out = sor::presolve::presolve(p, par);
            CHECK(out.map.problem.n_rows() == 1);
            for (const auto& x : {std::vector<double>{2.0,0.0}, std::vector<double>{3.0,0.0}})
                CHECK(out.map.problem.max_row_violation(x) <= 1e-12);
            CHECK(out.map.problem.max_row_violation({1.0,0.0}) >= 1.0);
        }

        // Doubleton equality: free x appears in two rows so the singleton
        // column rule cannot take it. x0 is free, so its bounds are exactly
        // implied: live presolve leaves the equation to the kernel
        // aggregation (the live doubleton only transfers bounds), which must
        // eliminate x0 and lift it. The remaining column may then dual-fix
        // once its row becomes redundant.
        {
            LpProblem p;
            p.A = sor::sparse::from_triplets(
                2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {1.0, 1.0, 1.0, 2.0});
            p.c = {0.0, 1.0};
            p.row_lo = {1.0, -sor::model::kInf};
            p.row_hi = {1.0, 3.0};
            p.col_lo = {-sor::model::kInf, 0.0};
            p.col_hi = {sor::model::kInf, 1.0};
            const auto out = sor::presolve::presolve(p, v2);
            CHECK(out.stats().doubleton_substitutions == 0);
            CHECK(out.stats().equality_aggregations >= 1);
            CHECK(out.map.problem.n_cols() <= 1);
            // Any reduced point lifts onto the equation x0 + x1 = 1.
            if (out.map.problem.n_cols() == 1) {
                const auto x = postsolve(out.map, {0.25});
                CHECK_NEAR(x[0] + x[1], 1.0, 1e-12);
            }
            // The reduced optimum: what is left is x1 in [0, 1] with cost
            // 1 - 0 * 1 = 1 and the slack row x1 <= 2 (the aggregation runs
            // after the live pass that would have removed it), so x1 = 0, its
            // lower bound. `validated` needs an optimal input.
            std::vector<f64> x_red(out.map.problem.col_lo.begin(),
                                   out.map.problem.col_lo.end());
            const auto x = postsolve(out.map, x_red);
            CHECK_NEAR(x[0] + x[1], 1.0, 1e-12);
            PresolveReducedSolve rs;
            rs.x = x_red;
            rs.y.assign(static_cast<std::size_t>(out.map.problem.n_rows()), 0.0);
            const auto recovered = recover_solution(p, out.map, rs);
            CHECK(recovered.validated);
            CHECK_NEAR(recovered.raw.x[0] + recovered.raw.x[1], 1.0, 1e-9);
        }

        // Negative: a semi-bounded column is never the one a doubleton
        // eliminates (the implied-slack case); here x0 is the elimination
        // candidate (equal |a|, first column) and must stay.
        {
            LpProblem p;
            p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
            p.c = {0.0, 1.0};
            p.row_lo = p.row_hi = {5.0};
            p.col_lo = {0.0, 0.0};
            p.col_hi = {sor::model::kInf, 10.0};
            const auto out = sor::presolve::presolve(p, v2);
            CHECK(out.stats().doubleton_substitutions == 0);
            CHECK(out.map.problem.n_cols() == 2);
        }
    }

    // Cancelling fixed terms leave an exact unit contribution. Neither the
    // singleton substitution nor redundant-row check may lose it.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(1, 4, {0,0,0,0}, {0,1,2,3}, {1e10,1,-1e10,1});
        p.c = {0,0,0,1};
        p.col_lo = {1e10,1,1e10,-sor::model::kInf};
        p.col_hi = {1e10,1,1e10,sor::model::kInf};
        p.row_lo = p.row_hi = {0};
        const auto reduced = presolve_lp(p);
        CHECK(reduced.problem.n_cols() == 0);
        const auto x = postsolve(reduced, {});
        CHECK(x.size() == 4);
        CHECK(x[3] == -1);
        CHECK(p.max_row_violation(x) == 0);
        CHECK(p.objective(x) == -1);
    }
    // A deadline that has passed abandons the reductions and returns the
    // identity map, valid and postsolvable, rather than finishing presolve:
    // the caller has no time left to solve the reduced model.
    {
        LpProblem p;
        p.A = sor::sparse::from_triplets(2, 3, {0,0,1}, {0,1,2}, {1,1,1});
        p.c = {1,1,1};
        p.col_lo = {0,2,0};
        p.col_hi = {5,2,5};  // column 1 is fixed: normally removed
        p.row_lo = {3,1};
        p.row_hi = {9,1};    // row 1 is a singleton equality
        sor::presolve::PresolveOptions opts;
        const auto normal = sor::presolve::presolve(p, opts);
        CHECK(normal.map.problem.n_cols() < 3);
        CHECK(!normal.map.stats.stopped_at_deadline);

        opts.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
        const auto late = sor::presolve::presolve(p, opts);
        CHECK(late.status == sor::presolve::PresolveStatus::Reduced);
        CHECK(late.map.stats.stopped_at_deadline);
        CHECK(late.map.problem.n_rows() == 2 && late.map.problem.n_cols() == 3);
        CHECK(late.map.recovery_steps.empty());
        const auto x = postsolve(late.map, {1.0, 2.0, 1.0});
        CHECK(x.size() == 3 && x[0] == 1.0 && x[1] == 2.0 && x[2] == 1.0);
    }
    return sor::test::finish("test_presolve");
}
