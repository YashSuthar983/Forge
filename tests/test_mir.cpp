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
#include <iostream>
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

// An integer variable shifted by a fractional active bound is not an
// integer MIR term. The old implementation treated x-0.5 as integral and
// emitted 5x+2y <= 4.5, excluding the feasible integer point (1,0).
void test_fractional_integer_bound_uses_continuous_term() {
    LpProblem lp;
    lp.name = "fractional-integer-bound";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {5.0, 2.0});
    lp.c = {0.0, -1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {5.0};
    lp.col_lo = {0.5, 0.0};
    lp.col_hi = {2.0, 3.0};
    lp.is_integer = {true, true};
    MirOptions o;
    MirDiagnostics d;
    const auto cuts = separate_mir(
        lp, {0.5, 1.25}, lp.col_lo, lp.col_hi, o, d);
    CHECK(!cuts.empty());
    for (const auto& cut : cuts) {
        f64 lhs = 0.0;
        for (std::size_t k = 0; k < cut.cols.size(); ++k)
            lhs += cut.vals[k] * (cut.cols[k] == 0 ? 1.0 : 0.0);
        if (lhs > cut.row_hi + 1e-9) {
            std::cerr << "invalid fractional-bound MIR: lhs=" << lhs
                      << " rhs=" << cut.row_hi;
            for (std::size_t k = 0; k < cut.cols.size(); ++k)
                std::cerr << " (" << cut.cols[k] << "," << cut.vals[k] << ")";
            std::cerr << '\n';
        }
        CHECK(lhs <= cut.row_hi + 1e-9);
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

// Fixed-charge structure: continuous flows x_k with x_k <= u_k * y_k (y
// binary or small integer), coupled by random demand/capacity rows. These are
// exactly the rows the variable-bound substitution rewrites, so every cut is
// checked by maximising it over the continuous directions at every integer
// assignment, as above.
LpProblem random_fixed_charge_lp(std::mt19937& rng, Index k, Index m_extra) {
    const Index n = 2 * k;   // y_0..y_{k-1}, then x_0..x_{k-1}
    std::uniform_int_distribution<int> cap(1, 6), coef(-3, 4);
    std::uniform_real_distribution<double> pick(0.0, 1.0);
    LpProblem lp;
    lp.name = "mir-vub";
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 0.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), false);
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    Index r = 0;
    for (Index q = 0; q < k; ++q) {
        const auto y = static_cast<std::size_t>(q), x = static_cast<std::size_t>(k + q);
        lp.is_integer[y] = true;
        lp.col_hi[y] = pick(rng) < 0.7 ? 1.0 : 2.0;
        lp.col_hi[x] = 20.0;
        const f64 u = cap(rng) + (pick(rng) < 0.5 ? 0.5 : 0.0);
        // x - u y <= 0, written in either orientation.
        const f64 sgn = pick(rng) < 0.5 ? 1.0 : -1.0;
        rows.push_back(r); cols.push_back(static_cast<Index>(x)); vals.push_back(sgn);
        rows.push_back(r); cols.push_back(static_cast<Index>(y)); vals.push_back(-sgn * u);
        if (sgn > 0.0) { lp.row_lo.push_back(-kInf); lp.row_hi.push_back(0.0); }
        else           { lp.row_lo.push_back(0.0);   lp.row_hi.push_back(kInf); }
        ++r;
    }
    for (Index e = 0; e < m_extra; ++e, ++r) {
        f64 hi_act = 0.0;
        for (Index j = 0; j < n; ++j) {
            if (pick(rng) > 0.6) continue;
            int a = coef(rng);
            if (a == 0) a = 1;
            rows.push_back(r); cols.push_back(j); vals.push_back(f64(a));
            if (a > 0) hi_act += a * lp.col_hi[static_cast<std::size_t>(j)];
        }
        const f64 rhs = std::floor(pick(rng) * hi_act * 0.5) + 0.5;
        if (pick(rng) < 0.5) { lp.row_lo.push_back(-kInf); lp.row_hi.push_back(rhs); }
        else { lp.row_lo.push_back(rhs); lp.row_hi.push_back(kInf); }
    }
    lp.A = from_triplets(r, n, rows, cols, vals);
    lp.c.assign(static_cast<std::size_t>(n), 1.0);
    return lp;
}

void test_variable_bound_cuts_are_valid_by_lp() {
    std::mt19937 rng(31337u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int total_cuts = 0, vb_cuts = 0, checked = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const Index k = 3 + static_cast<Index>(trial % 2);
        const LpProblem lp = random_fixed_charge_lp(rng, k, 2);
        const Index n = lp.n_cols();
        std::vector<int> span(static_cast<std::size_t>(n), 1);
        for (Index j = 0; j < k; ++j)
            span[static_cast<std::size_t>(j)] =
                static_cast<int>(lp.col_hi[static_cast<std::size_t>(j)]) + 1;
        // LP-like point: fractional y, flows at or near their variable bound.
        std::vector<f64> x(static_cast<std::size_t>(n));
        for (Index j = 0; j < k; ++j)
            x[static_cast<std::size_t>(j)] =
                unit(rng) * lp.col_hi[static_cast<std::size_t>(j)];
        for (Index j = k; j < n; ++j) x[static_cast<std::size_t>(j)] = unit(rng) * 5.0;
        MirOptions o;
        MirDiagnostics d;
        const auto cuts = separate_mir(lp, x, lp.col_lo, lp.col_hi, o, d);
        vb_cuts += d.variable_bound_substitutions > 0 ? 1 : 0;
        total_cuts += static_cast<int>(cuts.size());
        for (const auto& c : cuts)
            for_each_integer_point(lp, span, [&](const std::vector<f64>& p) {
                f64 best = 0.0;
                if (!max_cut_over_continuous(lp, c, p, best)) return;
                ++checked;
                CHECK(best <= c.row_hi + 1e-6);
            });
    }
    CHECK(vb_cuts > 20);
    CHECK(total_cuts > 40);
    CHECK(checked > 500);
}

// Single-node fixed-charge flow: x1 + x2 >= 3, x_k <= 2 y_k, y binary. The LP
// point y = (0.75, 0.75), x = (1.5, 1.5) satisfies every row. With simple
// bounds only, both flows have positive coefficients in the <= orientation
// (-x1 - x2 <= -3) ... and the continuous terms carry no integer information.
// Substituting x_k = 2 y_k - s_k gives -2y1 - 2y2 + s1 + s2 <= -3, whose MIR
// cut y1 + y2 >= 2 is violated at the point and valid (both arcs must open).
void test_single_node_flow_needs_variable_bounds() {
    LpProblem lp;
    lp.name = "flow";
    // cols: y1, y2, x1, x2
    lp.A = from_triplets(3, 4, {0, 0, 1, 1, 2, 2}, {2, 3, 2, 0, 3, 1},
                         {1.0, 1.0, 1.0, -2.0, 1.0, -2.0});
    lp.row_lo = {3.0, -kInf, -kInf};
    lp.row_hi = {kInf, 0.0, 0.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 10.0, 10.0};
    lp.is_integer = {true, true, false, false};
    lp.c = {1.0, 1.0, 0.0, 0.0};
    const std::vector<f64> x = {0.75, 0.75, 1.5, 1.5};

    MirOptions plain;
    plain.variable_bounds = false;
    MirDiagnostics d0;
    const auto without = separate_mir(lp, x, lp.col_lo, lp.col_hi, plain, d0);

    MirOptions o;
    MirDiagnostics d;
    const auto cuts = separate_mir(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(d.variable_bound_rows >= 2);
    CHECK(d.variable_bound_substitutions >= 2);
    CHECK(cuts.size() > without.size());
    bool violated = false;
    for (const auto& c : cuts) {
        f64 act = 0.0;
        for (std::size_t q = 0; q < c.cols.size(); ++q)
            act += c.vals[q] * x[static_cast<std::size_t>(c.cols[q])];
        violated |= act > c.row_hi + 1e-6;
        for_each_integer_point(lp, {2, 2, 1, 1}, [&](const std::vector<f64>& p) {
            f64 best = 0.0;
            if (!max_cut_over_continuous(lp, c, p, best)) return;
            CHECK(best <= c.row_hi + 1e-6);
        });
    }
    CHECK(violated);
}

}  // namespace

int main() {
    test_variable_bound_cuts_are_valid_by_lp();
    test_single_node_flow_needs_variable_bounds();
    test_textbook_mixed_example();
    test_scaling_finds_the_cut();
    test_nonzero_lower_bound_is_substituted();
    test_fractional_integer_bound_uses_continuous_term();
    test_random_mixed_cuts_are_valid_by_lp();
    return sor::test::finish("test_mir");
}
