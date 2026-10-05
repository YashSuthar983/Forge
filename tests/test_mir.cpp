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
#include "sor/search/conflict.hpp"
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

// Demand node with two in-arcs: x1 + x2 >= 2, x_k <= 10 y_k. At the LP point
// arc 1 carries the demand with y1 = 0.2 and arc 2 is unused (x2 = y2 = 0).
// Arc 2's VUB distance ties its simple lower-bound distance. It must still be
// substituted: left on its simple bound, x2 keeps a negative coefficient and
// the cut becomes y1 + y2 + x2/2 >= 1, which the LP satisfies by routing flow
// through arc 2 with y2 = 0.2. The pure cut-set y1 + y2 >= 1 has no flow term.
void test_unused_arc_gives_pure_cutset() {
    LpProblem lp;
    lp.name = "cutset";
    // cols: y1, y2, x1, x2
    lp.A = from_triplets(3, 4, {0, 0, 1, 1, 2, 2}, {2, 3, 2, 0, 3, 1},
                         {1.0, 1.0, 1.0, -10.0, 1.0, -10.0});
    lp.row_lo = {2.0, -kInf, -kInf};
    lp.row_hi = {kInf, 0.0, 0.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 10.0, 10.0};
    lp.is_integer = {true, true, false, false};
    lp.c = {1.0, 1.0, 0.0, 0.0};
    const std::vector<f64> x = {0.2, 0.0, 2.0, 0.0};
    MirOptions o;
    MirDiagnostics d;
    const auto cuts = separate_mir(lp, x, lp.col_lo, lp.col_hi, o, d);
    bool pure = false;
    for (const auto& c : cuts) {
        bool flow_free = true;
        f64 act = 0.0;
        for (std::size_t q = 0; q < c.cols.size(); ++q) {
            if (c.cols[q] >= 2) flow_free = false;
            act += c.vals[q] * x[static_cast<std::size_t>(c.cols[q])];
        }
        // -y1 - y2 <= -1 (any positive scaling), violated by 0.8.
        if (flow_free && act > c.row_hi + 0.5) pure = true;
        for_each_integer_point(lp, {2, 2, 1, 1}, [&](const std::vector<f64>& p) {
            f64 best = 0.0;
            if (!max_cut_over_continuous(lp, c, p, best)) return;
            CHECK(best <= c.row_hi + 1e-6);
        });
    }
    CHECK(pure);
}

// Variable LOWER bounds and nonzero-intercept variable bounds, both read off
// two-term rows: x_k >= b + l*y_k and x_k <= b' + u*y_k with general-integer
// y, tied together by random mixing rows. Every cut is checked by maximising
// it over the continuous columns at every integer assignment.
void test_variable_lower_bound_cuts_are_valid_by_lp() {
    std::mt19937 rng(4242u);
    std::uniform_int_distribution<int> small(1, 4);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int total_cuts = 0, checked = 0, lower_rows = 0;
    for (int trial = 0; trial < 150; ++trial) {
        const Index k = 2 + static_cast<Index>(trial % 2);   // pairs (y, x)
        const Index n = 2 * k;
        std::vector<Index> rows, cols;
        std::vector<f64> vals;
        LpProblem lp;
        lp.name = "vlb";
        lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
        lp.col_hi.assign(static_cast<std::size_t>(n), 0.0);
        lp.is_integer.assign(static_cast<std::size_t>(n), false);
        Index m = 0;
        for (Index j = 0; j < k; ++j) {
            const auto y = static_cast<std::size_t>(j);
            const auto xc = static_cast<std::size_t>(k + j);
            lp.is_integer[y] = true;
            lp.col_hi[y] = trial % 3 == 0 ? 2.0 : 1.0;
            lp.col_hi[xc] = 12.0;
            // x - l*y >= b   (VLB, intercept b)
            const f64 l = small(rng), b = small(rng) - 2.0;
            rows.insert(rows.end(), {m, m});
            cols.insert(cols.end(), {k + j, j});
            vals.insert(vals.end(), {1.0, -l});
            lp.row_lo.push_back(b);
            lp.row_hi.push_back(kInf);
            ++m;
            // x - u*y <= b'  (VUB, intercept b')
            const f64 u = l + small(rng), bu = small(rng);
            rows.insert(rows.end(), {m, m});
            cols.insert(cols.end(), {k + j, j});
            vals.insert(vals.end(), {1.0, -u});
            lp.row_lo.push_back(-kInf);
            lp.row_hi.push_back(bu);
            ++m;
        }
        // Mixing rows over the flows (and one integer), fractional sides.
        for (int r = 0; r < 2; ++r) {
            for (Index j = 0; j < k; ++j) {
                rows.push_back(m);
                cols.push_back(k + j);
                vals.push_back(unit(rng) < 0.5 ? 1.0 : -1.0);
            }
            rows.push_back(m);
            cols.push_back(static_cast<Index>(r % k));
            vals.push_back(r == 0 ? 1.0 : -2.0);
            const f64 side = std::round(unit(rng) * 8.0) + 0.5 - 4.0;
            if (r == 0) { lp.row_lo.push_back(side); lp.row_hi.push_back(kInf); }
            else { lp.row_lo.push_back(-kInf); lp.row_hi.push_back(side); }
            ++m;
        }
        lp.A = from_triplets(m, n, rows, cols, vals);
        lp.c.assign(static_cast<std::size_t>(n), 1.0);
        std::vector<f64> x(static_cast<std::size_t>(n));
        for (Index j = 0; j < n; ++j) {
            const auto u = static_cast<std::size_t>(j);
            x[u] = unit(rng) * lp.col_hi[u];
        }
        // Put some flows exactly on a variable bound so ties are exercised.
        // Row 2j reads x - l*y >= b: stored as vals {1, -l}, row_lo = b.
        for (Index j = 0; j < k; ++j) {
            if (unit(rng) >= 0.5) continue;
            const auto y = static_cast<std::size_t>(j);
            const f64 l = -lp.A.vals[static_cast<std::size_t>(4 * j + 1)];
            const f64 b = lp.row_lo[static_cast<std::size_t>(2 * j)];
            x[static_cast<std::size_t>(k + j)] = std::max(0.0, b + l * x[y]);
        }
        MirOptions o;
        o.aggregate = true;
        MirDiagnostics d;
        const auto cuts = separate_mir(lp, x, lp.col_lo, lp.col_hi, o, d);
        lower_rows += static_cast<int>(d.variable_bound_rows);
        total_cuts += static_cast<int>(cuts.size());
        std::vector<int> span(static_cast<std::size_t>(n), 1);
        for (Index j = 0; j < k; ++j)
            span[static_cast<std::size_t>(j)] =
                static_cast<int>(lp.col_hi[static_cast<std::size_t>(j)]) + 1;
        for (const auto& c : cuts)
            for_each_integer_point(lp, span, [&](const std::vector<f64>& p) {
                f64 best = 0.0;
                if (!max_cut_over_continuous(lp, c, p, best)) return;
                ++checked;
                CHECK(best <= c.row_hi + 1e-6);
            });
    }
    CHECK(lower_rows > 300);
    CHECK(total_cuts > 30);
    CHECK(checked > 200);
}

void test_probe_bound_affine_mir_is_valid() {
    // The useful bound x <= 1 + 2y follows through z, so it is invisible
    // to the explicit two-column-row VUB scanner.
    LpProblem lp;
    lp.A = from_triplets(3, 4,
        {0,0,1,1,2,2}, {2,0,3,2,1,3}, {1.,-2.,1.,-1.,1.,1.});
    lp.row_lo = {-kInf,-kInf,-kInf};
    lp.row_hi = {1.,0.,2.5};
    lp.col_lo = {0.,0.,0.,0.};
    lp.col_hi = {1.,3.,3.,3.};
    lp.is_integer = {true,true,false,false};
    lp.c = {0.,0.,0.,0.};
    std::vector<f64> lo=lp.col_lo, hi=lp.col_hi;
    sor::search::ConflictGraph graph;
    sor::search::ProbingOptions probe;
    probe.row_cliques=false;
    const auto gd=sor::search::build_conflict_graph(lp,lo,hi,graph,probe);
    CHECK(!gd.infeasible);
    bool found=false;
    for(const auto& ib:graph.implied_bounds())
        found|=ib.upper && ib.bin==0 && ib.col==3 &&
               std::fabs(ib.b0-1.)<1e-9 && std::fabs(ib.b1-2.5)<1e-9;
    CHECK(found);
    const std::vector<f64> x={0.5,0.75,2.,1.75};
    MirOptions opts;
    opts.probe_bounds=true;
    MirDiagnostics diag;
    const auto cuts=separate_mir(lp,x,lp.col_lo,lp.col_hi,opts,diag,
                                 nullptr,nullptr,&graph);
    CHECK(diag.probe_bound_candidates>0);
    CHECK(diag.probe_bound_substitutions>0);
    CHECK(!cuts.empty());
    for(const auto& cut:cuts)
        for_each_integer_point(lp,{2,4,1,1},[&](const std::vector<f64>& fixed){
            f64 best=0.;
            if(!max_cut_over_continuous(lp,cut,fixed,best))return;
            CHECK(best<=cut.row_hi+1e-6);
        });
}

void test_shared_cmir_uses_variable_bounds() {
    LpProblem lp;
    lp.A = from_triplets(3, 4, {0, 0, 1, 1, 2, 2}, {2, 3, 2, 0, 3, 1},
                         {1.0, 1.0, 1.0, -2.0, 1.0, -2.0});
    lp.row_lo = {3.0, -kInf, -kInf};
    lp.row_hi = {kInf, 0.0, 0.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 10.0, 10.0};
    lp.is_integer = {true, true, false, false};
    lp.c = {1.0, 1.0, 0.0, 0.0};
    const std::vector<f64> x{0.75, 0.75, 1.5, 1.5};
    std::vector<Index> cols;
    std::vector<f64> vals;
    f64 rhs = 0.0;
    MirOptions plain;
    plain.variable_bounds = false;
    MirDiagnostics without;
    CHECK(!sor::search::apply_cmir_geq(lp, {2, 3}, {1.0, 1.0}, 3.0,
        x, lp.col_lo, lp.col_hi, plain, true, cols, vals, rhs, without));
    MirOptions opts;
    MirDiagnostics diag;
    const bool found = sor::search::apply_cmir_geq(lp, {2, 3}, {1.0, 1.0},
        3.0, x, lp.col_lo, lp.col_hi, opts, true, cols, vals, rhs, diag);
    CHECK(found);
    CHECK(diag.variable_bound_rows >= 2);
    if (!found) return;
    f64 activity = 0.0;
    for (std::size_t k = 0; k < cols.size(); ++k)
        activity += vals[k] * x[static_cast<std::size_t>(cols[k])];
    CHECK(activity < rhs - 1e-6);
    CutRow cut;
    cut.cols = cols;
    for (const auto v : vals) cut.vals.push_back(-v);
    cut.row_hi = -rhs;
    for_each_integer_point(lp, {2, 2, 1, 1}, [&](const std::vector<f64>& p) {
        f64 best = 0.0;
        if (max_cut_over_continuous(lp, cut, p, best))
            CHECK(best <= cut.row_hi + 1e-6);
    });
}

}  // namespace

void test_mir_base_drop_relaxes_rhs() {
    LpProblem lp;
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1e-12});
    lp.c = {-1.0, 0.0}; lp.is_integer = {true, false};
    lp.col_lo = {0.0, -1e6}; lp.col_hi = {2.0, 1e6};
    lp.row_lo = {-kInf}; lp.row_hi = {1.0 - 5e-7};
    const std::vector<double> x{lp.row_hi[0], 0.0};
    CHECK(lp.max_row_violation(x) <= 1e-12);
    MirOptions o; o.variable_bounds = false;
    o.min_fractionality = 1e-9; o.violation_min = 1e-8;
    MirDiagnostics d;
    const auto cuts = separate_mir(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(d.bases_built > 0);
    for (int xi = 0; xi <= 2; ++xi)
        for (double y : {-1e6, 0.0, 1e6}) {
            const std::vector<double> point{static_cast<double>(xi), y};
            if (lp.max_row_violation(point) > 1e-12) continue;
            for (const auto& cut : cuts) {
                double activity = 0.0;
                for (std::size_t k = 0; k < cut.cols.size(); ++k)
                    activity += cut.vals[k] * point[static_cast<std::size_t>(cut.cols[k])];
                CHECK(activity <= cut.row_hi + 1e-9);
            }
        }
    lp.col_lo[1] = -kInf;
    MirDiagnostics unbounded;
    CHECK(separate_mir(lp, x, lp.col_lo, lp.col_hi, o, unbounded).empty());
    lp.col_lo[1] = -1e6;
    lp.row_hi[0] = 1.5;
    auto node_lo = lp.col_lo;
    node_lo[1] = -5e5;
    MirDiagnostics local;
    const auto local_cuts = separate_mir(lp, {1.5, 0.0}, node_lo, lp.col_hi,
        o, local, &lp.col_lo, &lp.col_hi);
    CHECK(!local_cuts.empty());
    for (const auto& cut : local_cuts) CHECK(cut.used_local_bound);
}

// Fixed-charge knapsacks with INTEGRAL right-hand sides: after variable-bound
// substitution many bases have no fractional part for c-MIR to round, which is
// where the lifted mixed-binary cover must supply the cut. Every cut is still
// checked by maximising it over the continuous directions at every binary
// assignment.
void test_lifted_cover_on_integral_bases_is_valid_by_lp() {
    std::mt19937 rng(9091u);
    std::uniform_int_distribution<int> cap(2, 9);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uint64_t covers = 0;
    int checked = 0;
    for (int trial = 0; trial < 150; ++trial) {
        const Index k = 3 + static_cast<Index>(trial % 3);
        const Index n = 2 * k;  // y_0..y_{k-1}, then x_0..x_{k-1}
        LpProblem lp;
        lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
        lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
        lp.is_integer.assign(static_cast<std::size_t>(n), false);
        std::vector<Index> rows, cols;
        std::vector<f64> vals;
        f64 total = 0.0;
        std::vector<f64> u(static_cast<std::size_t>(k));
        for (Index q = 0; q < k; ++q) {
            const auto y = static_cast<std::size_t>(q);
            lp.is_integer[y] = true;
            u[y] = cap(rng);
            total += u[y];
            lp.col_hi[static_cast<std::size_t>(k + q)] = 10.0;
            rows.push_back(q); cols.push_back(k + q); vals.push_back(1.0);
            rows.push_back(q); cols.push_back(q); vals.push_back(-u[y]);
            lp.row_lo.push_back(-kInf); lp.row_hi.push_back(0.0);
        }
        // Capacity row sum x <= B and a pure-binary knapsack on the y's.
        const f64 B = std::floor(total * (0.3 + 0.4 * unit(rng)));
        for (Index q = 0; q < k; ++q) {
            rows.push_back(k); cols.push_back(k + q); vals.push_back(1.0);
            rows.push_back(k + 1); cols.push_back(q); vals.push_back(u[static_cast<std::size_t>(q)]);
        }
        lp.row_lo.push_back(-kInf); lp.row_hi.push_back(B);
        lp.row_lo.push_back(-kInf); lp.row_hi.push_back(B);
        lp.A = from_triplets(k + 2, n, rows, cols, vals);
        lp.c.assign(static_cast<std::size_t>(n), -1.0);

        // A point of the LP relaxation: fractional y, flows at their VUB,
        // scaled into both knapsacks.
        std::vector<f64> x(static_cast<std::size_t>(n));
        f64 load = 0.0;
        for (Index q = 0; q < k; ++q) {
            x[static_cast<std::size_t>(q)] = 0.3 + 0.7 * unit(rng);
            load += u[static_cast<std::size_t>(q)] * x[static_cast<std::size_t>(q)];
        }
        const f64 scale = load > B ? B / load : 1.0;
        for (Index q = 0; q < k; ++q) {
            const auto y = static_cast<std::size_t>(q);
            x[y] *= scale;
            x[static_cast<std::size_t>(k + q)] = u[y] * x[y];
        }

        MirOptions o;
        o.lifted_cover = true;
        MirDiagnostics d;
        const auto cuts = separate_mir(lp, x, lp.col_lo, lp.col_hi, o, d);
        covers += d.lifted_cover_cuts;
        std::vector<int> span(static_cast<std::size_t>(n), 1);
        for (Index q = 0; q < k; ++q) span[static_cast<std::size_t>(q)] = 2;
        for (const auto& c : cuts) {
            for_each_integer_point(lp, span, [&](const std::vector<f64>& p) {
                f64 best = 0.0;
                if (!max_cut_over_continuous(lp, c, p, best)) return;
                ++checked;
                CHECK(best <= c.row_hi + 1e-6);
            });
        }
    }
    CHECK(covers > 20);
    CHECK(checked > 500);
}

int main() {
    test_shared_cmir_uses_variable_bounds();
    test_mir_base_drop_relaxes_rhs();
    test_variable_bound_cuts_are_valid_by_lp();
    test_single_node_flow_needs_variable_bounds();
    test_unused_arc_gives_pure_cutset();
    test_variable_lower_bound_cuts_are_valid_by_lp();
    test_textbook_mixed_example();
    test_scaling_finds_the_cut();
    test_nonzero_lower_bound_is_substituted();
    test_fractional_integer_bound_uses_continuous_term();
    test_random_mixed_cuts_are_valid_by_lp();
    test_probe_bound_affine_mir_is_valid();
    test_lifted_cover_on_integral_bases_is_valid_by_lp();
    return sor::test::finish("test_mir");
}
