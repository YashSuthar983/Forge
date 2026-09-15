// Mixed-Integer Rounding cuts (sor_search/src/mir.cpp).
//
// A previous MIR implementation in this codebase was reverted because it was
// unsound for CONTINUOUS variables and produced false `Infeasible` answers on
// real instances. Enumerating integer points is therefore not a sufficient
// test: it is precisely the continuous directions that were wrong, and a cut
// can be satisfied at every integer point while still slicing through the
// polyhedron between them.
//
// So validity is checked the only way that actually covers those directions:
// for each feasible integer assignment, the cut's left-hand side is MAXIMISED
// over the continuous variables by solving an LP against the model's own rows.
// If that maximum ever exceeds the cut's right-hand side, the cut is invalid.
#include "sor/engines/simplex.hpp"
#include "sor/search/mir.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <functional>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::search::CutRow;
using sor::search::MirDiagnostics;
using sor::search::MirOptions;
using sor::search::separate_mir;
using sor::sparse::from_triplets;

namespace {

// max { cut'x : model rows hold, integer columns fixed to `fix` }.
// Returns false when that region is empty (the assignment is infeasible, so it
// says nothing about the cut).
bool max_cut_over_continuous(const LpProblem& lp, const CutRow& cut,
                             const std::vector<f64>& fix, f64& best) {
    LpProblem sub = lp;
    sub.maximize = true;
    sub.obj_offset = 0.0;
    sub.c.assign(static_cast<std::size_t>(lp.n_cols()), 0.0);
    for (std::size_t k = 0; k < cut.cols.size(); ++k)
        sub.c[static_cast<std::size_t>(cut.cols[k])] += cut.vals[k];
    for (Index j = 0; j < lp.n_cols(); ++j) {
        const auto u = static_cast<std::size_t>(j);
        if (lp.is_integer.empty() || !lp.is_integer[u]) continue;
        sub.col_lo[u] = sub.col_hi[u] = fix[u];
    }
    sor::engines::SimplexOptions o;
    o.time_limit_s = 5.0;
    o.presolve = false;
    sor::engines::SimplexDiagnostics d;
    const auto r = sor::engines::solve_simplex(sub, o, d);
    if (r.proposed_status != sor::core::Status::Optimal) return false;
    best = r.objective;
    return true;
}

// Every integer assignment on a small grid.
void for_each_integer_point(const LpProblem& lp,
                            const std::vector<int>& span,
                            const std::function<void(const std::vector<f64>&)>& fn) {
    const Index n = lp.n_cols();
    std::vector<f64> p(static_cast<std::size_t>(n), 0.0);
    std::uint64_t total = 1;
    for (Index j = 0; j < n; ++j) total *= static_cast<std::uint64_t>(span[static_cast<std::size_t>(j)]);
    for (std::uint64_t code = 0; code < total; ++code) {
        std::uint64_t rest = code;
        for (Index j = 0; j < n; ++j) {
            const auto u = static_cast<std::size_t>(j);
            const auto s = static_cast<std::uint64_t>(span[u]);
            p[u] = lp.col_lo[u] + static_cast<f64>(rest % s);
            rest /= s;
        }
        fn(p);
    }
}

LpProblem random_mixed_lp(std::mt19937& rng, Index n_int, Index n_cont, Index m) {
    const Index n = n_int + n_cont;
    std::uniform_int_distribution<int> coef(-5, 6);
    std::uniform_real_distribution<double> pick(0.0, 1.0);
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    LpProblem lp;
    lp.name = "mir";
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 0.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), false);
    for (Index j = 0; j < n_int; ++j) {
        lp.is_integer[static_cast<std::size_t>(j)] = true;
        lp.col_hi[static_cast<std::size_t>(j)] = pick(rng) < 0.6 ? 1.0 : 2.0;
    }
    for (Index j = n_int; j < n; ++j)
        lp.col_hi[static_cast<std::size_t>(j)] = 3.0;

    for (Index i = 0; i < m; ++i) {
        f64 lo_act = 0.0, hi_act = 0.0;
        int cnt = 0;
        for (Index j = 0; j < n; ++j) {
            if (pick(rng) > 0.7) continue;
            int a = coef(rng);
            if (a == 0) a = 2;
            rows.push_back(i); cols.push_back(j); vals.push_back(f64(a));
            const f64 hi = lp.col_hi[static_cast<std::size_t>(j)];
            (a > 0 ? hi_act : lo_act) += a * hi;
            ++cnt;
        }
        if (cnt == 0) {
            rows.push_back(i); cols.push_back(0); vals.push_back(1.0);
            hi_act += lp.col_hi[0];
        }
        // A fractional rhs is what gives MIR something to round.
        std::uniform_real_distribution<double> r(lo_act, hi_act);
        const f64 rhs = std::round(r(rng) * 2.0) / 2.0 + 0.25;
        if (pick(rng) < 0.75) { lp.row_lo.push_back(-kInf); lp.row_hi.push_back(rhs); }
        else { lp.row_lo.push_back(rhs); lp.row_hi.push_back(kInf); }
    }
    lp.A = from_triplets(m, n, rows, cols, vals);
    lp.c.assign(static_cast<std::size_t>(n), 1.0);
    return lp;
}

// ---------------------------------------------------------------- units ----

// The worked example from mir.hpp: x integer >= 0, y >= 0, x - y <= 0.5.
// f = 0.5, so the MIR cut is x - 2y <= 0, which is valid and cuts off
// (x, y) = (0.75, 0.25).
void test_textbook_mixed_example() {
    LpProblem lp;
    lp.name = "textbook";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
    lp.c = {-1.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {0.5};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {4.0, 4.0};
    lp.is_integer = {true, false};

    const std::vector<f64> x = {0.75, 0.25};
    MirOptions o;
    MirDiagnostics d;
    const auto cuts = separate_mir(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(!cuts.empty());
    bool found = false;
    for (const auto& c : cuts) {
        f64 lhs = 0.0;
        for (std::size_t k = 0; k < c.cols.size(); ++k)
            lhs += c.vals[k] * x[static_cast<std::size_t>(c.cols[k])];
        if (lhs > c.row_hi + 1e-7) found = true;
    }
    CHECK(found);
}

// Scaling is not cosmetic: 0.5x <= 0.7 gives nothing at delta = 1 and gives
// x <= 1 at delta = 2. The c-MIR scaling sweep must find it.
void test_scaling_finds_the_cut() {
    LpProblem lp;
    lp.name = "scale";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {0.5, 0.5});
    lp.c = {-1.0, -1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {0.7};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {3.0, 3.0};
    lp.is_integer = {true, true};

    const std::vector<f64> x = {1.4, 0.0};
    MirOptions o;
    MirDiagnostics d;
    const auto cuts = separate_mir(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(!cuts.empty());
    // every integer point of the row must satisfy every returned cut
    for (const auto& c : cuts)
        for (int a = 0; a <= 3; ++a)
            for (int b = 0; b <= 3; ++b) {
                if (0.5 * a + 0.5 * b > 0.7 + 1e-9) continue;
                f64 lhs = 0.0;
                const std::vector<f64> p = {f64(a), f64(b)};
                for (std::size_t k = 0; k < c.cols.size(); ++k)
                    lhs += c.vals[k] * p[static_cast<std::size_t>(c.cols[k])];
                CHECK(lhs <= c.row_hi + 1e-7);
            }
}

// A variable on a NONZERO lower bound is the case that breaks a naive MIR:
// the formula assumes non-negativity, so the substitution must shift it.
void test_nonzero_lower_bound_is_substituted() {
    LpProblem lp;
    lp.name = "shifted";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {-1.0, -1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {7.5};
    lp.col_lo = {2.0, 3.0};      // neither variable starts at zero
    lp.col_hi = {6.0, 6.0};
    lp.is_integer = {true, true};

    const std::vector<f64> x = {2.5, 5.0};
    MirOptions o;
    MirDiagnostics d;
    const auto cuts = separate_mir(lp, x, lp.col_lo, lp.col_hi, o, d);
    for (const auto& c : cuts)
        for (int a = 2; a <= 6; ++a)
            for (int b = 3; b <= 6; ++b) {
                if (a + b > 7.5 + 1e-9) continue;
                const std::vector<f64> p = {f64(a), f64(b)};
                f64 lhs = 0.0;
                for (std::size_t k = 0; k < c.cols.size(); ++k)
                    lhs += c.vals[k] * p[static_cast<std::size_t>(c.cols[k])];
                CHECK(lhs <= c.row_hi + 1e-7);
            }
}

// ----------------------------------------------------------- randomized ----

// THE test. Random mixed models; for every feasible integer assignment the cut
// is MAXIMISED over the continuous variables by LP. This is what covers the
// directions the reverted implementation got wrong.
void test_random_mixed_cuts_are_valid_by_lp() {
    std::mt19937 rng(2718u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int total_cuts = 0, checked_points = 0, models = 0;
    for (int trial = 0; trial < 120; ++trial) {
        const Index n_int = 3 + static_cast<Index>(trial % 2);
        const Index n_cont = 1 + static_cast<Index>(trial % 2);
        const Index m = 2 + static_cast<Index>(trial % 2);
        const LpProblem lp = random_mixed_lp(rng, n_int, n_cont, m);
        const Index n = lp.n_cols();

        std::vector<int> span(static_cast<std::size_t>(n), 1);
        for (Index j = 0; j < n_int; ++j)
            span[static_cast<std::size_t>(j)] =
                static_cast<int>(lp.col_hi[static_cast<std::size_t>(j)]) + 1;

        std::vector<f64> x(static_cast<std::size_t>(n));
        for (Index j = 0; j < n; ++j)
            x[static_cast<std::size_t>(j)] =
                lp.col_lo[static_cast<std::size_t>(j)] +
                unit(rng) * (lp.col_hi[static_cast<std::size_t>(j)] -
                             lp.col_lo[static_cast<std::size_t>(j)]);

        MirOptions o;
        MirDiagnostics d;
        const auto cuts = separate_mir(lp, x, lp.col_lo, lp.col_hi, o, d);
        if (cuts.empty()) continue;
        ++models;
        total_cuts += static_cast<int>(cuts.size());

        for (const auto& c : cuts) {
            for_each_integer_point(lp, span, [&](const std::vector<f64>& p) {
                f64 best = 0.0;
                if (!max_cut_over_continuous(lp, c, p, best)) return;
                ++checked_points;
                CHECK(best <= c.row_hi + 1e-6);
            });
        }
    }
    CHECK(models > 20);
    CHECK(total_cuts > 40);
    CHECK(checked_points > 500);
}

}  // namespace

int main() {
    test_textbook_mixed_example();
    test_scaling_finds_the_cut();
    test_nonzero_lower_bound_is_substituted();
    test_random_mixed_cuts_are_valid_by_lp();
    return sor::test::finish("test_mir");
}
