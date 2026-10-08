// Basis carried from a parent LP to a child with a subset of its rows
// (search/basis_map.hpp): the mapped basis has the right shape and, used as a
// warm start, reaches the same optimum as a cold solve.
#include "forge/search/basis_map.hpp"
#include "forge/engines/dual_simplex.hpp"

#include "forge/engines/simplex.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <iostream>
#include <random>

using forge::core::f64;
using forge::core::Index;
using forge::model::kInf;
using forge::model::LpProblem;

namespace {

LpProblem random_lp(std::mt19937& rng, Index n, Index m) {
    LpProblem lp;
    lp.name = "bm";
    std::vector<Index> r, c;
    std::vector<f64> v;
    for (Index i = 0; i < m; ++i)
        for (Index j = 0; j < n; ++j)
            if (rng() % 3 == 0) {
                r.push_back(i); c.push_back(j);
                v.push_back(static_cast<f64>(1 + rng() % 5));
            }
    lp.A = forge::sparse::from_triplets(m, n, r, c, v);
    lp.c.resize(static_cast<std::size_t>(n));
    for (auto& x : lp.c) x = -static_cast<f64>(1 + rng() % 7);   // maximise-ish, bounded by rows
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 4.0);
    lp.row_lo.assign(static_cast<std::size_t>(m), -kInf);
    lp.row_hi.resize(static_cast<std::size_t>(m));
    for (auto& x : lp.row_hi) x = static_cast<f64>(3 + rng() % 12);
    lp.is_integer.assign(static_cast<std::size_t>(n), false);
    return lp;
}

LpProblem take_rows(const LpProblem& lp, const std::vector<Index>& rows) {
    LpProblem out = lp;
    std::vector<Index> r, c;
    std::vector<f64> v;
    out.row_lo.clear(); out.row_hi.clear();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (std::size_t k = 0; k < rows.size(); ++k) {
        const auto i = static_cast<std::size_t>(rows[k]);
        for (auto q = rp[i]; q < rp[i + 1]; ++q) {
            r.push_back(static_cast<Index>(k));
            c.push_back(ci[static_cast<std::size_t>(q)]);
            v.push_back(lp.A.vals[static_cast<std::size_t>(q)]);
        }
        out.row_lo.push_back(lp.row_lo[i]);
        out.row_hi.push_back(lp.row_hi[i]);
    }
    out.A = forge::sparse::from_triplets(static_cast<Index>(rows.size()), lp.n_cols(), r, c, v);
    return out;
}

// Deleting a binding row leaves a surplus basic column, and which one may
// leave is fixed by the matrix: rows r0: x0 + x1, r1: x1, r2: x2 with basis
// {x0, x1, x2}. Without r2, x2's column is empty, so x2 must leave; demoting
// x0 (nearest its bound at x = (0, 3, 3)) would leave {x1, x2} singular.
void test_fit_drops_the_column_the_rows_force() {
    using forge::engines::NonbasicStatus;
    LpProblem lp;
    lp.name = "fit";
    lp.A = forge::sparse::from_triplets(2, 3, {0, 0, 1}, {0, 1, 1}, {1.0, 1.0, 1.0});
    lp.c = {1.0, 1.0, 1.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {4.0, 4.0, 4.0};
    lp.row_lo = {-kInf, -kInf};
    lp.row_hi = {5.0, 3.0};
    lp.is_integer.assign(3, false);
    // Stored over the three old rows minus r2: status for 3 + 2 columns, three basics.
    forge::engines::SimplexBasis b;
    b.n_struct = 3;
    b.basic = {0, 1, 2};
    b.status = {NonbasicStatus::Basic, NonbasicStatus::Basic, NonbasicStatus::Basic,
                NonbasicStatus::AtUpper, NonbasicStatus::AtUpper};
    forge::search::BasisFitStats fs;
    CHECK(forge::search::fit_stored_basis(lp, 2, 0, b, lp.col_lo, lp.col_hi, &fs));
    CHECK(fs.dropped == 1 && fs.repaired == 0 && fs.appended == 0);
    CHECK(b.basic.size() == 2);
    CHECK(b.status[2] != NonbasicStatus::Basic);
    CHECK(b.status[0] == NonbasicStatus::Basic && b.status[1] == NonbasicStatus::Basic);

    // The same basis one global row later (appended since) and with one local
    // row after the globals: the new global row's logical enters basic, the
    // local logical moves past it, and x2 still leaves.
    LpProblem lp2 = lp;
    lp2.A = forge::sparse::from_triplets(4, 3, {0, 0, 1, 2, 3}, {0, 1, 1, 0, 2},
                                       {1.0, 1.0, 1.0, 1.0, 1.0});
    lp2.row_lo = {-kInf, -kInf, -kInf, -kInf};
    lp2.row_hi = {5.0, 3.0, 4.0, 2.0};
    forge::engines::SimplexBasis b2;
    b2.n_struct = 3;
    b2.basic = {0, 1, 2, 3 + 2};   // local row's logical basic, at basis row 2
    b2.status = {NonbasicStatus::Basic, NonbasicStatus::Basic, NonbasicStatus::Basic,
                 NonbasicStatus::AtUpper, NonbasicStatus::AtUpper, NonbasicStatus::Basic};
    CHECK(forge::search::fit_stored_basis(lp2, 3, 1, b2, lp2.col_lo, lp2.col_hi, &fs));
    CHECK(fs.appended == 1 && fs.dropped == 1 && fs.repaired == 0);
    CHECK(b2.basic.size() == 4);
    CHECK(b2.status[3 + 2] == NonbasicStatus::Basic);   // new global row 2
    CHECK(b2.status[3 + 3] == NonbasicStatus::Basic);   // local row, now lp row 3
    std::vector<Index> singular, vacant;
    forge::search::basis_rank_selection(lp2, 4, b2.basic, singular, vacant);
    CHECK(singular.empty());
}

}  // namespace

int main() {
    test_fit_drops_the_column_the_rows_force();
    std::mt19937 rng(5);
    int mapped = 0, agreed = 0, warm_fewer = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const Index n = 6 + static_cast<Index>(rng() % 8);
        const Index m = 6 + static_cast<Index>(rng() % 10);
        auto lp = random_lp(rng, n, m);
        forge::engines::SimplexOptions so;
        so.presolve = false;
        forge::engines::SimplexDiagnostics sd;
        forge::engines::SimplexBasis basis;
        const auto pr = forge::engines::solve_simplex(lp, so, sd, &basis);
        if (pr.proposed_status != forge::core::Status::Optimal) continue;

        // Child: first `base` rows plus a random subset of the rest; tighter box.
        const Index base = static_cast<Index>(rng() % static_cast<unsigned>(m));
        std::vector<Index> keep, extra;
        for (Index i = 0; i < base; ++i) keep.push_back(i);
        for (Index i = base; i < m; ++i)
            if (rng() % 2) { keep.push_back(i); extra.push_back(i); }
        auto child = take_rows(lp, keep);
        for (Index j = 0; j < n; ++j)
            if (rng() % 4 == 0) child.col_hi[static_cast<std::size_t>(j)] = 2.0;

        const auto mb = forge::search::map_basis_to_child(basis, n, m, base, 0, extra, pr.x, child);
        CHECK(mb.has_value());
        if (!mb) continue;
        CHECK(static_cast<Index>(mb->basic.size()) == child.n_rows());
        CHECK(mb->status.size() == static_cast<std::size_t>(n + child.n_rows()));
        int basics = 0;
        for (const auto s : mb->status) basics += s == forge::engines::NonbasicStatus::Basic;
        CHECK(basics == child.n_rows());
        // Exact selection: the mapped basis factors over the child's rows.
        std::vector<Index> singular, vacant;
        forge::search::basis_rank_selection(child, child.n_rows(), mb->basic, singular, vacant);
        CHECK(singular.empty());
        ++mapped;

        forge::engines::SimplexDiagnostics cold_d, warm_d;
        forge::engines::SimplexBasis cb, wb;
        const auto cold = forge::engines::solve_simplex(child, so, cold_d, &cb);
        const auto warm = forge::engines::solve_dual_simplex(child, so, warm_d, &wb, &*mb);
        if (cold.proposed_status == forge::core::Status::Optimal &&
            warm.proposed_status == forge::core::Status::Optimal) {
            CHECK(std::fabs(cold.objective - warm.objective) <=
                  1e-6 * (1.0 + std::fabs(cold.objective)));
            ++agreed;
            if (warm_d.iterations < cold_d.iterations) ++warm_fewer;
        }
    }
    std::cout << "BASIS_MAP mapped=" << mapped << " agreed=" << agreed
              << " warm_fewer_iterations=" << warm_fewer << '\n';
    CHECK(mapped > 150);
    CHECK(agreed > 120);
    return forge::test::finish("test_basis_map");
}
