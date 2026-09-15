// Conflict graph, probing, and clique cuts (sor_search/src/conflict.cpp).
//
// The load-bearing tests here are the randomized ones. Every reduction in
// conflict.cpp claims the same thing -- "no integer-feasible point of the MILP
// is excluded" -- and for small binary models that claim is decidable by brute
// force: enumerate all 2^n assignments, keep the feasible ones, and check them
// against the probed bounds, every extracted clique, every separated cut, and
// every conflict propagation. A separator that is merely plausible passes unit
// tests on hand-built examples; this catches the case the hand-built example
// did not think of.
#include "sor/search/conflict.hpp"
#include "sor/search/propagate.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::search::Clique;
using sor::search::ConflictGraph;
using sor::search::lit_of;
using sor::search::ProbingOptions;
using sor::sparse::from_triplets;

namespace {

// Every integer assignment inside lp's box that satisfies all rows, by brute
// force. Columns may have any small integer domain, not only {0,1}, so the
// mixed-domain generator below can exercise the implied-bound path (a variable
// bound is only interesting when the bounded column has room to move).
std::vector<std::vector<f64>> all_feasible(const LpProblem& lp) {
    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    std::vector<std::vector<f64>> out;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<int> span(static_cast<std::size_t>(n));
    std::uint64_t total = 1;
    for (Index j = 0; j < n; ++j) {
        const auto u = static_cast<std::size_t>(j);
        span[u] = static_cast<int>(lp.col_hi[u] - lp.col_lo[u]) + 1;
        total *= static_cast<std::uint64_t>(span[u]);
    }
    for (std::uint64_t code = 0; code < total; ++code) {
        std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
        std::uint64_t rest = code;
        for (Index j = 0; j < n; ++j) {
            const auto u = static_cast<std::size_t>(j);
            x[u] = lp.col_lo[u] +
                   static_cast<f64>(rest % static_cast<std::uint64_t>(span[u]));
            rest /= static_cast<std::uint64_t>(span[u]);
        }
        bool ok = true;
        for (Index i = 0; i < m && ok; ++i) {
            f64 act = 0.0;
            for (auto k = rp[static_cast<std::size_t>(i)];
                 k < rp[static_cast<std::size_t>(i) + 1]; ++k)
                act += av[static_cast<std::size_t>(k)] *
                       x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            if (act < lp.row_lo[static_cast<std::size_t>(i)] - 1e-9 ||
                act > lp.row_hi[static_cast<std::size_t>(i)] + 1e-9)
                ok = false;
        }
        if (ok) out.push_back(std::move(x));
    }
    return out;
}

// `n_general` of the columns get domain [0,2] instead of [0,1]; the rest stay
// binary. A model with only binaries can never produce an interesting variable
// bound, because the bounded column has nowhere to sit between 0 and 1.
LpProblem random_binary_lp(std::mt19937& rng, Index n, Index m,
                           Index n_general = 0) {
    std::uniform_int_distribution<int> coef(-4, 6);
    std::uniform_real_distribution<double> pick(0.0, 1.0);
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    LpProblem lp;
    lp.name = "rnd";
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
    for (Index j = n - n_general; j < n; ++j)
        lp.col_hi[static_cast<std::size_t>(j)] = 2.0;
    for (Index i = 0; i < m; ++i) {
        f64 pos = 0.0, neg = 0.0;
        bool any = false;
        for (Index j = 0; j < n; ++j) {
            if (pick(rng) > 0.55) continue;
            int a = coef(rng);
            if (a == 0) a = 1;
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(static_cast<f64>(a));
            const f64 hi = lp.col_hi[static_cast<std::size_t>(j)];
            (a > 0 ? pos : neg) += a * hi;
            any = true;
        }
        if (!any) {  // keep every row nonempty so from_triplets stays honest
            rows.push_back(i);
            cols.push_back(0);
            vals.push_back(1.0);
            pos += 1.0;
        }
        // An rhs strictly inside the row's activity range makes the row bite
        // without making the model trivially infeasible.
        std::uniform_real_distribution<double> r(neg, pos);
        const f64 rhs = std::floor(r(rng));
        if (pick(rng) < 0.75) {
            lp.row_lo.push_back(-kInf);
            lp.row_hi.push_back(rhs);
        } else {
            lp.row_lo.push_back(rhs);
            lp.row_hi.push_back(kInf);
        }
    }
    lp.A = from_triplets(m, n, rows, cols, vals);
    lp.c.assign(static_cast<std::size_t>(n), 1.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), true);
    return lp;
}

// ---------------------------------------------------------------- units ----

// x1 + x2 + x3 <= 1 over binaries: the row IS a clique, and probing any of the
// three to 1 must imply the other two are 0.
void test_set_packing_row_gives_clique() {
    LpProblem lp;
    lp.name = "packing";
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    lp.c = {-1.0, -1.0, -1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {1.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    const auto d = sor::search::build_conflict_graph(lp, lo, hi, cg, {});
    CHECK(!d.infeasible);
    CHECK(cg.n_edges() == 3);  // (1,2), (1,3), (2,3) as "both = 1" conflicts
    CHECK(cg.conflicts(lit_of(0, 1), lit_of(1, 1)));
    CHECK(cg.conflicts(lit_of(0, 1), lit_of(2, 1)));
    CHECK(cg.conflicts(lit_of(1, 1), lit_of(2, 1)));
    CHECK(!cg.conflicts(lit_of(0, 0), lit_of(1, 0)));  // all-zero is feasible

    bool found = false;
    for (const auto& c : cg.cliques())
        if (c.lits.size() == 3) found = true;
    CHECK(found);

    // The LP point (0.5, 0.5, 0.5) satisfies the row but violates the clique.
    const std::vector<f64> x = {0.5, 0.5, 0.5};
    const auto cuts = sor::search::separate_clique_cuts(cg, x, {});
    CHECK(!cuts.empty());
    CHECK_NEAR(cuts[0].row_hi, 1.0, 1e-12);
}

// A precedence pair x1 <= x2 (i.e. x1 - x2 <= 0) is a conflict between
// "x1 = 1" and "x2 = 0" -- the implication structure that row propagation
// alone never turns into a reusable fact.
void test_precedence_row_gives_implication() {
    LpProblem lp;
    lp.name = "prec";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
    lp.c = {1.0, 1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    const auto d = sor::search::build_conflict_graph(lp, lo, hi, cg, {});
    CHECK(!d.infeasible);
    CHECK(cg.conflicts(lit_of(0, 1), lit_of(1, 0)));
    CHECK(!cg.conflicts(lit_of(0, 0), lit_of(1, 1)));  // x1=0, x2=1 is feasible

    // Fixing x1 = 1 must force x2 = 1 through the graph, with no LP solve.
    std::vector<f64> nlo = {1.0, 0.0}, nhi = {1.0, 1.0};
    std::uint64_t tightened = 0;
    CHECK(sor::search::propagate_conflicts(cg, nlo, nhi, tightened));
    CHECK(tightened >= 1);
    CHECK_NEAR(nlo[1], 1.0, 1e-12);
    CHECK_NEAR(nhi[1], 1.0, 1e-12);
}

// Both sides of a probe dying means the model has no integer point at all.
void test_probing_detects_infeasibility() {
    // x1 + x2 >= 3 with two binaries: max activity is 2.
    LpProblem lp;
    lp.name = "infeas";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {1.0, 1.0};
    lp.row_lo = {3.0};
    lp.row_hi = {kInf};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    const auto d = sor::search::build_conflict_graph(lp, lo, hi, cg, {});
    CHECK(d.infeasible);
}

// One dead side fixes the column globally: x1 + x2 >= 2 forces both to 1.
void test_probing_fixes_column() {
    LpProblem lp;
    lp.name = "fix";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {1.0, 1.0};
    lp.row_lo = {2.0};
    lp.row_hi = {kInf};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    const auto d = sor::search::build_conflict_graph(lp, lo, hi, cg, {});
    CHECK(!d.infeasible);
    CHECK(d.probe_fixings >= 1);
    CHECK_NEAR(lo[0], 1.0, 1e-12);
    CHECK_NEAR(lo[1], 1.0, 1e-12);
}

// The hull tightening applies to continuous columns too: y is in [0,10], and
// row y <= 3*x1 with x1 binary means y <= 3 whichever way x1 goes.
void test_probing_hull_tightens_continuous() {
    LpProblem lp;
    lp.name = "hull";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -3.0});  // y - 3*x1 <= 0
    lp.c = {1.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {10.0, 1.0};
    lp.is_integer = {false, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    const auto d = sor::search::build_conflict_graph(lp, lo, hi, cg, {});
    CHECK(!d.infeasible);
    CHECK(d.probe_tightenings >= 1);
    CHECK_NEAR(hi[0], 3.0, 1e-9);
}

// y <= 3*x1 with y continuous in [0,10]: the hull only proves y <= 3, but the
// implied bound keeps the dependence and gives y - 3*x1 <= 0, which is the
// original row back and is strictly stronger wherever x1 is fractional.
void test_probing_records_implied_bound() {
    LpProblem lp;
    lp.name = "vub";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -3.0});
    lp.c = {1.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {10.0, 1.0};
    lp.is_integer = {false, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    const auto d = sor::search::build_conflict_graph(lp, lo, hi, cg, {});
    CHECK(!d.infeasible);
    CHECK(d.implied_bounds >= 1);

    bool found = false;
    for (const auto& ib : cg.implied_bounds())
        if (ib.upper && ib.col == 0 && ib.bin == 1) {
            CHECK_NEAR(ib.b0, 0.0, 1e-9);  // x1 = 0 forces y <= 0
            CHECK_NEAR(ib.b1, 3.0, 1e-9);  // x1 = 1 allows y <= 3
            found = true;
        }
    CHECK(found);

    // (y, x1) = (1.5, 0.5) satisfies y <= 3 but violates y - 3*x1 <= 0? No --
    // it is exactly tight. (2.0, 0.5) violates it and must be cut.
    const std::vector<f64> x = {2.0, 0.5};
    const auto cuts = sor::search::separate_implied_bound_cuts(cg, x, {});
    CHECK(!cuts.empty());
}

// ----------------------------------------------------------- randomized ----

// The soundness contract, checked by enumeration on 400 random models: after
// build_conflict_graph, every brute-force feasible point must still lie inside
// the tightened box and satisfy every clique in the table. When the graph
// reports infeasible, there must genuinely be no feasible point.
void test_random_models_preserve_all_feasible_points() {
    std::mt19937 rng(20260907u);
    int with_cliques = 0, with_edges = 0, tightened_models = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const Index n = 4 + static_cast<Index>(trial % 7);   // 4..10 binaries
        const Index m = 2 + static_cast<Index>(trial % 5);
        const LpProblem lp = random_binary_lp(rng, n, m);
        const auto feasible = all_feasible(lp);

        std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
        ConflictGraph cg;
        const auto d = sor::search::build_conflict_graph(lp, lo, hi, cg, {});

        if (d.infeasible) {
            CHECK(feasible.empty());
            continue;
        }
        if (!cg.cliques().empty()) ++with_cliques;
        if (cg.n_edges() > 0) ++with_edges;
        if (d.probe_tightenings + d.probe_fixings > 0) ++tightened_models;

        for (const auto& x : feasible) {
            for (Index j = 0; j < n; ++j) {
                const auto u = static_cast<std::size_t>(j);
                CHECK(x[u] >= lo[u] - 1e-9);
                CHECK(x[u] <= hi[u] + 1e-9);
            }
            // Every clique: at most one literal true.
            for (const auto& c : cg.cliques()) {
                f64 act = 0.0;
                for (const Index l : c.lits) act += sor::search::lit_value(l, x);
                CHECK(act <= 1.0 + 1e-9);
            }
            // Every edge: the two literals are never both true.
            for (Index j = 0; j < n; ++j)
                for (int v = 0; v < 2; ++v) {
                    const Index l = lit_of(j, v);
                    if (sor::search::lit_value(l, x) < 0.5) continue;
                    for (const Index nb : cg.neighbors(l))
                        CHECK(sor::search::lit_value(nb, x) < 0.5);
                }
        }
    }
    // Guard against a vacuous pass: the generator must actually be producing
    // models where the machinery has something to say.
    CHECK(with_cliques > 50);
    CHECK(with_edges > 50);
    CHECK(tightened_models > 20);
}

// Separated cuts must be valid inequalities, not merely violated ones: every
// feasible point of the model satisfies every cut the separator returns, at
// arbitrary fractional query points.
void test_random_separated_cuts_are_valid() {
    std::mt19937 rng(777u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int cut_count = 0, vub_count = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const Index n = 5 + static_cast<Index>(trial % 6);
        const Index m = 3 + static_cast<Index>(trial % 4);
        // Half the trials carry two general-integer columns so the
        // implied-bound separator has something to find.
        const Index gen = (trial % 2) ? 2 : 0;
        const LpProblem lp = random_binary_lp(rng, n, m, gen);
        const auto feasible = all_feasible(lp);
        if (feasible.empty()) continue;

        std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
        ConflictGraph cg;
        const auto d = sor::search::build_conflict_graph(lp, lo, hi, cg, {});
        if (d.infeasible) continue;

        for (int q = 0; q < 4; ++q) {
            std::vector<f64> x(static_cast<std::size_t>(n));
            for (Index j = 0; j < n; ++j) {
                const auto u = static_cast<std::size_t>(j);
                x[u] = lp.col_lo[u] + unit(rng) * (lp.col_hi[u] - lp.col_lo[u]);
            }
            auto cuts = sor::search::separate_clique_cuts(cg, x, {});
            cut_count += static_cast<int>(cuts.size());
            const auto vubs = sor::search::separate_implied_bound_cuts(cg, x, {});
            vub_count += static_cast<int>(vubs.size());
            cuts.insert(cuts.end(), vubs.begin(), vubs.end());
            for (const auto& cut : cuts) {
                // Valid: satisfied by every integer-feasible point.
                for (const auto& xf : feasible) {
                    f64 act = 0.0;
                    for (std::size_t k = 0; k < cut.cols.size(); ++k)
                        act += cut.vals[k] *
                               xf[static_cast<std::size_t>(cut.cols[k])];
                    CHECK(act <= cut.row_hi + 1e-9);
                    CHECK(act >= cut.row_lo - 1e-9);
                }
                // Useful: it actually cuts off the query point, on whichever
                // side of the row is finite.
                f64 act = 0.0;
                for (std::size_t k = 0; k < cut.cols.size(); ++k)
                    act += cut.vals[k] * x[static_cast<std::size_t>(cut.cols[k])];
                if (std::isfinite(cut.row_hi)) CHECK(act > cut.row_hi + 1e-6);
                else CHECK(act < cut.row_lo - 1e-6);
            }
        }
    }
    CHECK(cut_count > 100);
    CHECK(vub_count > 20);
}

// Conflict propagation must never prune a feasible point: for every random
// model and every sub-box that fixes a few binaries, if a feasible point lies
// in that box it must survive propagation.
void test_random_conflict_propagation_keeps_feasible_points() {
    std::mt19937 rng(31337u);
    int pruned = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const Index n = 5 + static_cast<Index>(trial % 6);
        const Index m = 3 + static_cast<Index>(trial % 4);
        const LpProblem lp = random_binary_lp(rng, n, m);
        const auto feasible = all_feasible(lp);

        std::vector<f64> rlo = lp.col_lo, rhi = lp.col_hi;
        ConflictGraph cg;
        const auto d = sor::search::build_conflict_graph(lp, rlo, rhi, cg, {});
        if (d.infeasible) continue;

        std::uniform_int_distribution<int> pick_col(0, static_cast<int>(n) - 1);
        for (int t = 0; t < 6; ++t) {
            std::vector<f64> lo = rlo, hi = rhi;
            for (int f = 0; f < 2; ++f) {
                const auto j = static_cast<std::size_t>(pick_col(rng));
                const f64 v = (rng() & 1u) ? 1.0 : 0.0;
                if (v < lo[j] - 0.5 || v > hi[j] + 0.5) continue;
                lo[j] = hi[j] = v;
            }
            std::vector<f64> plo = lo, phi = hi;
            std::uint64_t tightened = 0;
            const bool ok =
                sor::search::propagate_conflicts(cg, plo, phi, tightened);
            if (!ok) ++pruned;

            for (const auto& x : feasible) {
                bool in_box = true;
                for (Index j = 0; j < n; ++j) {
                    const auto u = static_cast<std::size_t>(j);
                    if (x[u] < lo[u] - 1e-9 || x[u] > hi[u] + 1e-9) in_box = false;
                }
                if (!in_box) continue;
                CHECK(ok);  // a box containing a feasible point is never pruned
                for (Index j = 0; j < n; ++j) {
                    const auto u = static_cast<std::size_t>(j);
                    CHECK(x[u] >= plo[u] - 1e-9);
                    CHECK(x[u] <= phi[u] + 1e-9);
                }
            }
        }
    }
    CHECK(pruned > 0);  // non-vacuous: propagation does prune some boxes
}

}  // namespace

int main() {
    test_set_packing_row_gives_clique();
    test_precedence_row_gives_implication();
    test_probing_detects_infeasibility();
    test_probing_fixes_column();
    test_probing_hull_tightens_continuous();
    test_probing_records_implied_bound();
    test_random_models_preserve_all_feasible_points();
    test_random_separated_cuts_are_valid();
    test_random_conflict_propagation_keeps_feasible_points();
    return sor::test::finish("test_conflict");
}
