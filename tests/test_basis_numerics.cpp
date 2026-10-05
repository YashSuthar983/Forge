#include "sor/la/basis_numerics.hpp"
#include "test_helpers.hpp"
#include "sor/core/parallel.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <limits>
#include <vector>

int main() {
    using namespace sor::la;
    const std::vector<Offset> p{0, 2, 4};
    const std::vector<Index> r{0, 1, 0, 1};
    const std::vector<f64> a{4, 1, 2, 3};
    BasisFactor factor;
    CHECK(factor.factorize(2, p, r, a, {}));
    CHECK(factor.stats().condition_estimate >= 1);
    std::vector<f64> x{1.0001, 1.9998};
    const auto refined = refine_basis_solution(factor, 2, p, r, a, {8, 7}, x);
    CHECK(refined.corrections > 0);
    CHECK(std::fabs(x[0] - 1) < 1e-14);
    CHECK(std::fabs(x[1] - 2) < 1e-14);
    x = {1.0001, 1.9998};
    const auto transpose = refine_basis_solution(factor, 2, p, r, a, {6, 8}, x, true);
    CHECK(transpose.corrections > 0);
    CHECK(std::fabs(x[0] - 1) < 1e-14);
    CHECK(std::fabs(x[1] - 2) < 1e-14);
    sor::core::ThreadPool pool(3);
    std::vector<std::vector<f64>> batch{{8,7},{4,1},{2,3},{-1,4},{10,-4}};
    auto serial = batch;
    for (auto& rhs : serial) factor.ftran(rhs);
    std::vector<SpikeCapture> spikes;
    factor.solve_batch(batch, &pool, false, &spikes);
    CHECK(batch == serial); CHECK(spikes.size() == batch.size());
    for (const auto& spike : spikes) CHECK(spike.valid);
    serial = batch;
    for (auto& rhs : serial) factor.btran(rhs);
    factor.solve_batch(batch, &pool, true);
    CHECK(batch == serial);
    const std::vector<Offset> diagonal_p{0, 1, 2};
    const std::vector<Index> diagonal_r{0, 1};
    CHECK(factor.factorize(2, diagonal_p, diagonal_r, {1, 1e-10}, {}));
    CHECK(factor.stats().condition_estimate >= 1e9);
    CHECK(factor.stats().effective_threshold == 0.5);
    LuOptions fixed; fixed.adaptive_threshold = false;
    CHECK(factor.factorize(2, diagonal_p, diagonal_r, {1, 1e-10}, fixed));
    CHECK(factor.stats().effective_threshold == fixed.markowitz_threshold);
    x = {1, std::numeric_limits<f64>::infinity()};
    CHECK(!refine_basis_solution(factor, 2, diagonal_p, diagonal_r, {1, 1e-10}, {1, 1}, x).finite);
    // Exercise the blocked nucleus after sparse peeling, with independently
    // computed right-hand sides in both orientations.
    constexpr Index n = 40;
    std::vector<Offset> cp{0}; std::vector<Index> ri;
    std::vector<f64> dense, expected(n), normal(n,0), transposed(n,0);
    for (Index i = 0; i < n; ++i) expected[i] = (i%7)-3;
    for (Index j = 0; j < n; ++j) {
        for (Index i = 0; i < n; ++i) {
            const f64 value = i == j ? 100.0 : ((i*11+j*7)%13-6)/16.0;
            ri.push_back(i); dense.push_back(value);
            normal[i] += value*expected[j]; transposed[j] += value*expected[i];
        }
        cp.push_back(static_cast<Offset>(dense.size()));
    }
    CHECK(factor.factorize(n,cp,ri,dense,{}));
    CHECK(factor.stats().blocked_nucleus_pivots == n);
    factor.ftran(normal); factor.btran(transposed);
    for (Index i = 0; i < n; ++i) {
        CHECK(std::fabs(normal[i]-expected[i]) < 1e-12);
        CHECK(std::fabs(transposed[i]-expected[i]) < 1e-12);
    }
    // Two dense components exceed the global size cap, while each fits it.
    // Permute the rows independently of columns so that the implementation
    // must find bipartite components rather than diagonal index ranges.
    constexpr Index total = 80;
    cp = {0}; ri.clear(); dense.clear();
    expected.resize(total); normal.assign(total, 0); transposed.assign(total, 0);
    for (Index i = 0; i < total; ++i) expected[i] = (i%7)-3;
    for (Index j = 0; j < total; ++j) {
        for (Index local = 0; local < 40; ++local) {
            const Index logical_row = (j/40)*40 + local;
            const Index i = (logical_row*17)%total;
            const f64 value = logical_row == j ? 100.0 : ((logical_row*11+j*7)%13-6)/16.0;
            ri.push_back(i); dense.push_back(value);
            normal[i] += value*expected[j]; transposed[j] += value*expected[i];
        }
        cp.push_back(static_cast<Offset>(dense.size()));
    }
    LuOptions blocks; blocks.dense_nucleus_limit = 48;
    CHECK(factor.factorize(total, cp, ri, dense, blocks));
    CHECK(factor.stats().blocked_nucleus_blocks == 2);
    CHECK(factor.stats().blocked_nucleus_pivots == total);
    factor.ftran(normal); factor.btran(transposed);
    for (Index i = 0; i < total; ++i) {
        CHECK(std::fabs(normal[i]-expected[i]) < 1e-12);
        CHECK(std::fabs(transposed[i]-expected[i]) < 1e-12);
    }
    // A small sparse component must survive alongside the blocked component
    // and finish through ordinary Markowitz elimination.
    constexpr Index mixed = 43;
    cp = {0}; ri.clear(); dense.clear();
    expected.resize(mixed); normal.assign(mixed, 0); transposed.assign(mixed, 0);
    for (Index i = 0; i < mixed; ++i) expected[i] = (i%7)-3;
    for (Index j = 0; j < mixed; ++j) {
        const auto append = [&](Index logical_row, f64 value) {
            const Index i = (logical_row*17)%mixed;
            ri.push_back(i); dense.push_back(value);
            normal[i] += value*expected[j]; transposed[j] += value*expected[i];
        };
        if (j < 40) {
            for (Index i = 0; i < 40; ++i)
                append(i, i == j ? 100.0 : ((i*11+j*7)%13-6)/16.0);
        } else {
            append(j, 2.0);
            append(40+(j-40+1)%3, 1.0);
        }
        cp.push_back(static_cast<Offset>(dense.size()));
    }
    blocks.dense_nucleus_limit = 40;
    CHECK(factor.factorize(mixed, cp, ri, dense, blocks));
    CHECK(factor.stats().blocked_nucleus_blocks == 1);
    CHECK(factor.stats().blocked_nucleus_pivots == 40);
    CHECK(factor.stats().nucleus_pivots > 40);
    CHECK(factor.stats().nucleus_pivots + factor.stats().triangular_pivots == mixed);
    factor.ftran(normal); factor.btran(transposed);
    for (Index i = 0; i < mixed; ++i) {
        CHECK(std::fabs(normal[i]-expected[i]) < 1e-12);
        CHECK(std::fabs(transposed[i]-expected[i]) < 1e-12);
    }
    // A connected sparse nucleus with catastrophic fill: Markowitz starts it,
    // the Schur complement fills in, and the dense tail finishes it with
    // partial row pivoting. Off-diagonal values dominate the diagonal, so the
    // tail must pivot rows for stability. Checked against a pure Markowitz
    // factorization, through FTRAN/BTRAN, and across a Forrest-Tomlin update.
    {
        constexpr Index fronts = 220;
        std::uint64_t seed = 0x9e3779b97f4a7c15ull;
        const auto next = [&]() { seed = seed*6364136223846793005ull + 1442695040888963407ull;
                                  return static_cast<Index>(seed >> 33); };
        std::vector<std::vector<std::pair<Index, f64>>> columns(fronts);
        for (Index j = 0; j < fronts; ++j) {
            columns[j].push_back({j, 0.25});
            for (int t = 0; t < 3; ++t) {
                const Index i = next() % fronts;
                if (i == j) continue;
                bool seen = false;
                for (auto& [row, value] : columns[j]) if (row == i) seen = true;
                if (!seen) columns[j].push_back({i, static_cast<f64>(next() % 17) / 2.0 - 4.0 + 0.125});
            }
            std::sort(columns[j].begin(), columns[j].end());
        }
        const auto assemble = [&](std::vector<Offset>& ptr, std::vector<Index>& rows, std::vector<f64>& values) {
            ptr = {0}; rows.clear(); values.clear();
            for (Index j = 0; j < fronts; ++j) {
                for (auto [i, v] : columns[j]) { rows.push_back(i); values.push_back(v); }
                ptr.push_back(static_cast<Offset>(rows.size()));
            }
        };
        const auto check_solves = [&](const BasisFactor& f, f64 tolerance) {
            std::vector<f64> want(fronts), b(fronts, 0.0), c(fronts, 0.0);
            for (Index i = 0; i < fronts; ++i) want[i] = static_cast<f64>((i * 7) % 11) - 5.0;
            for (Index j = 0; j < fronts; ++j)
                for (auto [i, v] : columns[j]) { b[i] += v * want[j]; c[j] += v * want[i]; }
            f.ftran(b); f.btran(c);
            f64 error = 0.0;
            for (Index i = 0; i < fronts; ++i)
                error = std::max({error, std::fabs(b[i] - want[i]), std::fabs(c[i] - want[i])});
            return error < tolerance;
        };
        std::vector<Offset> sp; std::vector<Index> sr; std::vector<f64> sv;
        assemble(sp, sr, sv);
        LuOptions tail; tail.adaptive_threshold = false;
        BasisFactor blocked;
        CHECK(blocked.factorize(fronts, sp, sr, sv, tail));
        CHECK(blocked.stats().dense_tail_blocks == 1);
        CHECK(blocked.stats().blocked_nucleus_blocks == 1);
        CHECK(blocked.stats().blocked_nucleus_pivots >= 32);
        CHECK(blocked.stats().nucleus_pivots > blocked.stats().blocked_nucleus_pivots);
        CHECK(blocked.stats().triangular_pivots + blocked.stats().nucleus_pivots == fronts);
        CHECK(check_solves(blocked, 1e-8));
        LuOptions sparse = tail; sparse.blocked_nucleus = false;
        BasisFactor reference;
        CHECK(reference.factorize(fronts, sp, sr, sv, sparse));
        CHECK(reference.stats().dense_tail_blocks == 0);
        CHECK(check_solves(reference, 1e-8));
        // Replace slot 17 and update both representations of the new basis.
        std::vector<f64> entering(fronts, 0.0);
        for (Index i = 0; i < fronts; i += 9) entering[i] = static_cast<f64>(i % 5) - 2.0 + 0.5;
        std::vector<f64> alpha = entering;
        blocked.ftran(alpha);
        CHECK(blocked.update_ft(17, alpha, tail));
        columns[17].clear();
        for (Index i = 0; i < fronts; ++i) if (entering[i] != 0.0) columns[17].push_back({i, entering[i]});
        CHECK(check_solves(blocked, 1e-8));
        // A sparse cyclic band never fills enough to leave Markowitz.
        std::vector<Offset> bp{0}; std::vector<Index> br; std::vector<f64> bv;
        for (Index j = 0; j < fronts; ++j) {
            std::vector<std::pair<Index, f64>> col{{(j + fronts - 1) % fronts, -1.0}, {j, 3.0}, {(j + 1) % fronts, -1.0}};
            std::sort(col.begin(), col.end());
            for (auto [i, v] : col) { br.push_back(i); bv.push_back(v); }
            bp.push_back(static_cast<Offset>(br.size()));
        }
        BasisFactor band;
        CHECK(band.factorize(fronts, bp, br, bv, tail));
        CHECK(band.stats().dense_tail_blocks == 0);
        CHECK(band.stats().blocked_nucleus_blocks == 0);
    }
    return sor::test::finish("test_basis_numerics");
}
