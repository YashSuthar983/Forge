// Basis carried from a parent LP to a child with a subset of its rows
// (search/basis_map.hpp): the mapped basis has the right shape and, used as a
// warm start, reaches the same optimum as a cold solve.
#include "sor/search/basis_map.hpp"
#include "sor/engines/dual_simplex.hpp"

#include "sor/engines/simplex.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <iostream>
#include <random>

using sor::core::f64;
using sor::core::Index;
using sor::model::kInf;
using sor::model::LpProblem;

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
    lp.A = sor::sparse::from_triplets(m, n, r, c, v);
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
    out.A = sor::sparse::from_triplets(static_cast<Index>(rows.size()), lp.n_cols(), r, c, v);
    return out;
}

}  // namespace

int main() {
    std::mt19937 rng(5);
    int mapped = 0, agreed = 0, warm_fewer = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const Index n = 6 + static_cast<Index>(rng() % 8);
        const Index m = 6 + static_cast<Index>(rng() % 10);
        auto lp = random_lp(rng, n, m);
        sor::engines::SimplexOptions so;
        so.presolve = false;
        sor::engines::SimplexDiagnostics sd;
        sor::engines::SimplexBasis basis;
        const auto pr = sor::engines::solve_simplex(lp, so, sd, &basis);
        if (pr.proposed_status != sor::core::Status::Optimal) continue;

        // Child: first `base` rows plus a random subset of the rest; tighter box.
        const Index base = static_cast<Index>(rng() % static_cast<unsigned>(m));
        std::vector<Index> keep, extra;
        for (Index i = 0; i < base; ++i) keep.push_back(i);
        for (Index i = base; i < m; ++i)
            if (rng() % 2) { keep.push_back(i); extra.push_back(i); }
        auto child = take_rows(lp, keep);
        for (Index j = 0; j < n; ++j)
            if (rng() % 4 == 0) child.col_hi[static_cast<std::size_t>(j)] = 2.0;

        const auto mb = sor::search::map_basis_to_child(basis, n, m, base, 0, extra, pr.x,
                                                       lp.col_lo, lp.col_hi);
        CHECK(mb.has_value());
        if (!mb) continue;
        CHECK(static_cast<Index>(mb->basic.size()) == child.n_rows());
        CHECK(mb->status.size() == static_cast<std::size_t>(n + child.n_rows()));
        int basics = 0;
        for (const auto s : mb->status) basics += s == sor::engines::NonbasicStatus::Basic;
        CHECK(basics == child.n_rows());
        ++mapped;

        sor::engines::SimplexDiagnostics cold_d, warm_d;
        sor::engines::SimplexBasis cb, wb;
        const auto cold = sor::engines::solve_simplex(child, so, cold_d, &cb);
        const auto warm = sor::engines::solve_dual_simplex(child, so, warm_d, &wb, &*mb);
        if (cold.proposed_status == sor::core::Status::Optimal &&
            warm.proposed_status == sor::core::Status::Optimal) {
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
    return sor::test::finish("test_basis_map");
}
