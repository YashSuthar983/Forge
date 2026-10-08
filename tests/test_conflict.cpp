// Conflict graph, probing, and clique cuts (forge_search/src/conflict.cpp).
//
// The load-bearing tests here are the randomized ones. Every reduction in
// conflict.cpp claims the same thing -- "no integer-feasible point of the MILP
// is excluded" -- and for small binary models that claim is decidable by brute
// force: enumerate all 2^n assignments, keep the feasible ones, and check them
// against the probed bounds, every extracted clique, every separated cut, and
// every conflict propagation. A separator that is merely plausible passes unit
// tests on hand-built examples; this catches the case the hand-built example
// did not think of.
#include "forge/search/conflict.hpp"
#include "forge/model/dyadic.hpp"
#include "forge/search/clique_rows.hpp"
#include "forge/search/propagate.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

using forge::core::f64;
using forge::core::Index;
using forge::model::kInf;
using forge::model::LpProblem;
using forge::search::Clique;
using forge::search::ConflictGraph;
using forge::search::lit_of;
using forge::search::ProbingOptions;
using forge::sparse::from_triplets;

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
    const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, {});
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
    const auto cuts = forge::search::separate_clique_cuts(cg, x, {});
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
    const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, {});
    CHECK(!d.infeasible);
    CHECK(cg.conflicts(lit_of(0, 1), lit_of(1, 0)));
    CHECK(!cg.conflicts(lit_of(0, 0), lit_of(1, 1)));  // x1=0, x2=1 is feasible

    // Fixing x1 = 1 must force x2 = 1 through the graph, with no LP solve.
    std::vector<f64> nlo = {1.0, 0.0}, nhi = {1.0, 1.0};
    std::uint64_t tightened = 0;
    CHECK(forge::search::propagate_conflicts(cg, nlo, nhi, tightened));
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
    const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, {});
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
    const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, {});
    CHECK(!d.infeasible);
    // The pre-probing propagation fixpoint already forces both columns; a
    // dead probe side would fix them otherwise. Either way they are fixed.
    CHECK(d.probe_fixings + d.probe_tightenings >= 1);
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
    const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, {});
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
    const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, {});
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
    const auto cuts = forge::search::separate_implied_bound_cuts(cg, x, {});
    CHECK(!cuts.empty());
}

void test_dual_fix_probe_does_not_create_global_implied_bound() {
    // With x=0, every y in [0,2] is feasible, although minimizing y lets
    // dual fixing choose y=0. The objective-based choice cannot become the
    // globally valid implication y <= 0 when x=0.
    LpProblem lp;
    lp.name = "dual_probe_scope";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {-1.0, 1.0});
    lp.c = {0.0, 1.0};
    lp.row_lo = {0.0};
    lp.row_hi = {kInf};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 2.0};
    lp.is_integer = {true, false};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    ProbingOptions opts;
    opts.dual_fix_in_probing = true;
    ConflictGraph cg;
    const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, opts);
    CHECK(!d.infeasible);
    const std::vector<f64> feasible_witness = {0.0, 2.0};
    for (const auto& ib : cg.implied_bounds()) {
        const f64 rhs = ib.b0 + (ib.b1 - ib.b0) * feasible_witness[ib.bin];
        if (ib.upper)
            CHECK(feasible_witness[ib.col] <= rhs + 1e-9);
        else
            CHECK(feasible_witness[ib.col] >= rhs - 1e-9);
    }
}

void test_dual_fix_probe_does_not_create_conflict_edge() {
    // y >= x and min y: x=0, y=1 is feasible but dual fixing picks y=0
    // on the x=0 probe. That objective choice cannot imply x=0 -> y=0.
    LpProblem lp;
    lp.name = "dual_probe_binary_scope";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {-1.0, 1.0});
    lp.c = {0.0, 1.0};
    lp.row_lo = {0.0};
    lp.row_hi = {kInf};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    ProbingOptions opts;
    opts.dual_fix_in_probing = true;
    ConflictGraph cg;
    const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, opts);
    CHECK(!d.infeasible);
    CHECK(!cg.conflicts(lit_of(0, 0), lit_of(1, 1)));
    std::vector<f64> node_lo = {0.0, 0.0}, node_hi = {0.0, 1.0};
    std::uint64_t tightened = 0;
    CHECK(forge::search::propagate_conflicts(
        cg, node_lo, node_hi, tightened));
    CHECK_NEAR(node_hi[1], 1.0, 1e-12);
}

void test_dual_fix_probe_random_scope_and_optimum() {
    std::mt19937 rng(20260924u);
    std::uniform_int_distribution<int> cost(-3, 3);
    int nonempty = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const Index n = 4 + static_cast<Index>(trial % 3);
        const Index m = 2 + static_cast<Index>(trial % 3);
        LpProblem lp = random_binary_lp(rng, n, m, trial % 2);
        for (f64& cj : lp.c) cj = static_cast<f64>(cost(rng));
        lp.maximize = trial % 2 == 0;
        const auto feasible = all_feasible(lp);
        if (feasible.empty()) continue;
        ++nonempty;

        std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
        ProbingOptions opts;
        opts.dual_fix_in_probing = true;
        ConflictGraph cg;
        const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, opts);
        CHECK(!d.infeasible);
        if (d.infeasible) continue;

        f64 best = lp.maximize ? -kInf : kInf;
        for (const auto& point : feasible) {
            const f64 obj = lp.objective(point);
            best = lp.maximize ? std::max(best, obj) : std::min(best, obj);
        }
        bool retained_optimum = false;
        for (const auto& point : feasible) {
            bool in_box = true;
            for (Index j = 0; j < n; ++j)
                if (point[j] < lo[j] - 1e-9 || point[j] > hi[j] + 1e-9)
                    in_box = false;
            if (!in_box) continue;
            if (std::fabs(lp.objective(point) - best) <= 1e-9)
                retained_optimum = true;

            for (const auto& ib : cg.implied_bounds()) {
                const f64 rhs = ib.b0 + (ib.b1 - ib.b0) * point[ib.bin];
                if (ib.upper) CHECK(point[ib.col] <= rhs + 1e-9);
                else CHECK(point[ib.col] >= rhs - 1e-9);
            }
            for (const auto& clique : cg.cliques()) {
                int active = 0;
                for (Index lit : clique.lits)
                    active += point[forge::search::lit_var(lit)] ==
                              forge::search::lit_val(lit);
                CHECK(active <= 1);
            }
            for (Index j = 0; j < n; ++j)
                for (Index k = j + 1; k < n; ++k)
                    if (cg.is_binary(j) && cg.is_binary(k))
                        CHECK(!cg.conflicts(
                            lit_of(j, static_cast<int>(point[j])),
                            lit_of(k, static_cast<int>(point[k]))));
        }
        CHECK(retained_optimum);
    }
    CHECK(nonempty >= 100);
}

void test_probe_preserves_wide_integer_side() {
    // x=0, y=2e9 is feasible and minimizes x. A small row coefficient is
    // still decisive over y's wide domain; probing must retain this side.
    LpProblem lp;
    lp.name = "probe_wide_integer";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 5e-10});
    lp.c = {1.0, 0.0};
    lp.row_lo = {1.0};
    lp.row_hi = {1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 2e9};
    lp.is_integer = {true, true};
    std::vector<f64> probe_lo = {0.0, 0.0}, probe_hi = {0.0, 2e9};
    const auto pr = forge::search::propagate_bounds(
        lp, probe_lo, probe_hi, 1e-7, 5);
    CHECK(pr.feasible);
    CHECK(probe_lo[1] <= 2e9);
    CHECK(probe_hi[1] >= 2e9);
    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    ProbingOptions opts;
    opts.dual_fix_in_probing = false;
    ConflictGraph cg;
    const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, opts);
    CHECK(!d.infeasible);
    CHECK(lo[0] <= 0.0);
    CHECK(hi[0] >= 1.0);
    CHECK(lo[1] <= 2e9);
    CHECK(hi[1] >= 2e9);
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
        const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, {});

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
                for (const Index l : c.lits) act += forge::search::lit_value(l, x);
                CHECK(act <= 1.0 + 1e-9);
            }
            // Every edge: the two literals are never both true.
            for (Index j = 0; j < n; ++j)
                for (int v = 0; v < 2; ++v) {
                    const Index l = lit_of(j, v);
                    if (forge::search::lit_value(l, x) < 0.5) continue;
                    for (const Index nb : cg.neighbors(l))
                        CHECK(forge::search::lit_value(nb, x) < 0.5);
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
        const auto d = forge::search::build_conflict_graph(lp, lo, hi, cg, {});
        if (d.infeasible) continue;

        for (int q = 0; q < 4; ++q) {
            std::vector<f64> x(static_cast<std::size_t>(n));
            for (Index j = 0; j < n; ++j) {
                const auto u = static_cast<std::size_t>(j);
                x[u] = lp.col_lo[u] + unit(rng) * (lp.col_hi[u] - lp.col_lo[u]);
            }
            auto cuts = forge::search::separate_clique_cuts(cg, x, {});
            cut_count += static_cast<int>(cuts.size());
            const auto vubs = forge::search::separate_implied_bound_cuts(cg, x, {});
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
        const auto d = forge::search::build_conflict_graph(lp, rlo, rhi, cg, {});
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
                forge::search::propagate_conflicts(cg, plo, phi, tightened);
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

// Probing must not stop at the first few thousand binaries: on drayage-25-23
// the first 4000 probes fixed nothing and the rest fixed 1162 columns. Here a
// column that only probing can fix (z = 0 forces p = q = 1, which a third row
// forbids) sits after 3000 filler binaries.
void test_probing_budget_reaches_late_binaries() {
    const int fillers = 3000;
    const Index n = fillers + 3;   // ..., p, q, z
    const Index p = fillers, q = fillers + 1, z = fillers + 2;
    std::vector<Index> rows, cols;
    std::vector<double> vals;
    Index m = 0;
    for (int i = 0; i + 2 < fillers; i += 3) {   // packing rows over the fillers
        for (int k = 0; k < 3; ++k) { rows.push_back(m); cols.push_back(i + k); vals.push_back(1.0); }
        ++m;
    }
    const Index r_first = m;
    for (const auto& e : std::vector<std::pair<Index, Index>>{{z, p}, {z, q}}) {
        rows.push_back(m); cols.push_back(e.first); vals.push_back(1.0);
        rows.push_back(m); cols.push_back(e.second); vals.push_back(1.0);
        ++m;
    }
    rows.push_back(m); cols.push_back(p); vals.push_back(1.0);
    rows.push_back(m); cols.push_back(q); vals.push_back(1.0);
    ++m;
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(m, n, rows, cols, vals);
    lp.row_lo.assign(static_cast<std::size_t>(m), -forge::model::kInf);
    lp.row_hi.assign(static_cast<std::size_t>(m), 1.0);
    for (Index i = r_first; i < r_first + 2; ++i) {   // z + p >= 1, z + q >= 1
        lp.row_lo[static_cast<std::size_t>(i)] = 1.0;
        lp.row_hi[static_cast<std::size_t>(i)] = forge::model::kInf;
    }
    lp.c.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), true);
    auto lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph g;
    ProbingOptions po;   // defaults
    const auto d = build_conflict_graph(lp, lo, hi, g, po);
    CHECK(!d.infeasible);
    CHECK(!d.probing_truncated);
    CHECK(d.probes >= 2 * static_cast<std::uint64_t>(fillers));   // past the old 4000-probe cap
    CHECK(d.probe_fixings + d.probe_tightenings >= 1);
    CHECK(lo[static_cast<std::size_t>(z)] == 1.0);
}

}  // namespace

// add_edge keeps each literal's list as a sorted base, sorted runs merged
// like a binary counter, and a short tail, so a duplicate check is a few
// binary searches instead of a scan of the whole list. It must answer
// exactly as a set would: return value, conflicts(), edge count,
// forget_columns and the final sorted lists. The last trials have a few
// literals of degree in the thousands (runs up to 2048 entries) and
// sort_adjacency mid-stream (a non-empty base under new runs).
void test_edge_set_matches_reference() {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 46; ++trial) {
        const bool hubs = trial >= 40;
        const Index n = hubs ? 1500 + static_cast<Index>(rng() % 1000)
                             : 8 + static_cast<Index>(rng() % 70);
        forge::search::ConflictGraph g;
        g.reset(n);
        std::set<std::pair<Index, Index>> ref;
        const auto key = [](Index a, Index b) { return std::make_pair(std::min(a, b), std::max(a, b)); };
        const int ops = hubs ? 8000 + static_cast<int>(rng() % 2000)
                             : 2000 + static_cast<int>(rng() % 4000);
        for (int op = 0; op < ops; ++op) {
            const Index l1 = hubs ? static_cast<Index>(rng() % 4)
                                  : static_cast<Index>(rng() % (2 * n));
            const Index l2 = static_cast<Index>(rng() % (2 * n));
            const bool valid = l1 != l2 && l1 / 2 != l2 / 2;
            const bool fresh = valid && ref.find(key(l1, l2)) == ref.end();
            CHECK(g.add_edge(l1, l2) == fresh);
            if (fresh) ref.insert(key(l1, l2));
            const Index q1 = static_cast<Index>(rng() % (hubs ? 4 : 2 * n));
            const Index q2 = static_cast<Index>(rng() % (2 * n));
            const bool qvalid = q1 != q2 && q1 / 2 != q2 / 2;
            CHECK(g.conflicts(q1, q2) == (qvalid && ref.count(key(q1, q2)) > 0));
            // In the hub trials both re-sort the hubs' lists, so they come
            // once, early, and the runs after them grow past 1024.
            if (hubs ? op == 400 : op % 1499 == 700) g.sort_adjacency();
            if (hubs ? op == 800 : op % 997 == 0) {
                std::vector<char> drop(static_cast<std::size_t>(n), 0);
                const Index j = static_cast<Index>(hubs ? 2 + rng() % (n - 2) : rng() % n);
                drop[static_cast<std::size_t>(j)] = 1;
                g.forget_columns(drop);
                for (auto it = ref.begin(); it != ref.end();)
                    it = (it->first / 2 == j || it->second / 2 == j) ? ref.erase(it) : std::next(it);
            }
        }
        CHECK(g.n_edges() == ref.size());
        g.sort_adjacency();
        CHECK(g.n_edges() == ref.size());
        for (Index l = 0; l < 2 * n; ++l) {
            std::vector<Index> want;
            for (const auto& [a, b] : ref) {
                if (a == l) want.push_back(b);
                if (b == l) want.push_back(a);
            }
            std::sort(want.begin(), want.end());
            CHECK(g.neighbors(l) == want);
        }
    }
}

// Dual fixing inside a probe side counts only the locks that still bind
// there (audit section 6). z (cost +1) has one down-lock, from
// 5x + 5w + z >= 5. With x = 1 that row cannot bind; with x = 0, x + w >= 1
// forces w = 1 and again it cannot bind. So in each side z drops to its
// lower bound, 0, and the hull fixes z = 0 -- where the global locks
// (one down-lock) leave z in [0, 9].
void test_probing_dual_fix_uses_side_redundancy() {
    forge::model::LpProblem lp;
    // columns x, w, z
    lp.c = {0.0, 0.0, 1.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 10.0};
    lp.is_integer = {true, true, false};
    lp.row_lo = {5.0, 1.0, -kInf};
    lp.row_hi = {kInf, kInf, 9.0};
    lp.A = forge::sparse::from_triplets(3, 3, {0, 0, 0, 1, 1, 2, 2}, {0, 1, 2, 0, 1, 0, 2},
                                      {5.0, 5.0, 1.0, 1.0, 1.0, 3.0, 1.0});
    lp.validate();
    auto lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    ProbingOptions po;
    po.dual_fix_in_probing = true;
    forge::search::build_conflict_graph(lp, lo, hi, cg, po);
    CHECK(hi[2] == 0.0);
}

// Row cliques rest on an exact activity floor (audit section 2): with
// M = 2^67, continuous p = q = r = 1 and binary x, y, the row
// M p - q - M r + x + y <= 1 has floor -1 and slack 2, so x = y = 1 is
// feasible. Its f64 floor in column order is 0, which gave x + y <= 1.
void test_row_cliques_use_exact_floor() {
    const double M = 0x1p67;
    forge::model::LpProblem lp;
    lp.c = {0.0, 0.0, 0.0, 0.0, 0.0};
    lp.col_lo = {1.0, 1.0, 1.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0, 1.0, 1.0};
    lp.is_integer = {false, false, false, true, true};
    lp.row_lo = {-forge::model::kInf};
    lp.row_hi = {1.0};
    lp.A = forge::sparse::from_triplets(1, 5, {0, 0, 0, 0, 0}, {0, 1, 2, 3, 4},
                                      {M, -1.0, -M, 1.0, 1.0});
    lp.validate();
    auto lo = lp.col_lo, hi = lp.col_hi;
    forge::search::ConflictGraph cg;
    ProbingOptions po;
    po.enabled = false;    // the row's own cliques only
    forge::search::build_conflict_graph(lp, lo, hi, cg, po);
    CHECK(!cg.conflicts(lit_of(3, 1), lit_of(4, 1)));
    // With the side at 0.5 the pair really conflicts (slack 1.5 < 2).
    lp.row_hi = {0.5};
    lo = lp.col_lo; hi = lp.col_hi;
    forge::search::ConflictGraph cg2;
    forge::search::build_conflict_graph(lp, lo, hi, cg2, po);
    CHECK(cg2.conflicts(lit_of(3, 1), lit_of(4, 1)));
}

// Odd-cycle cuts: a 5-cycle of conflicts among x0..x4 at x = 1/2 gives
// x0 + ... + x4 <= 2. On random conflict graphs (with some negative literals)
// every cut must hold at every 0/1 point that respects all conflicts.
void test_odd_cycle_cuts() {
    const auto make_graph = [](Index n) {
        LpProblem lp;
        lp.c.assign(static_cast<std::size_t>(n), 0.0);
        lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
        lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
        lp.is_integer.assign(static_cast<std::size_t>(n), true);
        lp.A = forge::sparse::from_triplets(0, n, {}, {}, {});
        ConflictGraph g;
        g.reset(n);
        g.mark_binaries(lp, lp.col_lo, lp.col_hi);
        return g;
    };
    {
        ConflictGraph g = make_graph(5);
        for (Index j = 0; j < 5; ++j) g.add_edge(lit_of(j, 1), lit_of((j + 1) % 5, 1));
        g.sort_adjacency();
        const std::vector<f64> x(5, 0.5);
        const auto cuts = forge::search::separate_odd_cycle_cuts(g, x, 10, 10, 1e-4);
        CHECK(!cuts.empty());
        if (!cuts.empty()) {
            CHECK(cuts[0].cols.size() == 5);
            CHECK(cuts[0].row_hi == 2.0);
            for (const f64 v : cuts[0].vals) CHECK(v == 1.0);
        }
    }
    {
        // K5: every odd cycle has a chord, down to triangles (cliques): no
        // odd-cycle row (sum <= 2 would sit beside the clique's sum <= 1).
        ConflictGraph g = make_graph(5);
        for (Index a = 0; a < 5; ++a)
            for (Index b = a + 1; b < 5; ++b) g.add_edge(lit_of(a, 1), lit_of(b, 1));
        g.sort_adjacency();
        const std::vector<f64> x(5, 0.45);   // a 5-cycle weighs 0.5: violated but for its chords
        CHECK(forge::search::separate_odd_cycle_cuts(g, x, 10, 10, 1e-4).empty());
    }
    {
        // A 5-cycle of explicit edges inside a long stored clique (whose
        // pairs are no explicit edges): chords through the clique, no row.
        const Index n = 300;
        ConflictGraph g = make_graph(n);
        Clique c;
        for (Index j = 0; j < n; ++j) c.lits.push_back(lit_of(j, 1));
        g.add_clique(c);
        for (Index j = 0; j < 5; ++j) g.add_edge(lit_of(j, 1), lit_of((j + 1) % 5, 1));
        g.sort_adjacency();
        std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
        for (Index j = 0; j < 5; ++j) x[static_cast<std::size_t>(j)] = 0.45;
        CHECK(forge::search::separate_odd_cycle_cuts(g, x, 10, 10, 1e-4).empty());
    }
    std::mt19937 rng(424242);
    int found = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const Index n = 6 + static_cast<Index>(rng() % 5);
        ConflictGraph g = make_graph(n);
        std::vector<std::pair<Index, Index>> edges;
        for (int e = 0; e < 3 * n; ++e) {
            const Index a = lit_of(static_cast<Index>(rng() % n), static_cast<int>(rng() % 4 != 0));
            const Index b = lit_of(static_cast<Index>(rng() % n), static_cast<int>(rng() % 4 != 0));
            if ((a >> 1) == (b >> 1)) continue;
            if (g.add_edge(a, b)) edges.emplace_back(a, b);
        }
        g.sort_adjacency();
        std::vector<f64> x(static_cast<std::size_t>(n));
        for (auto& v : x) v = 0.05 + 0.9 * static_cast<f64>(rng() % 1000) / 1000.0;
        const auto cuts = forge::search::separate_odd_cycle_cuts(g, x, 20, 64, 1e-4);
        found += static_cast<int>(cuts.size());
        for (std::uint32_t mask = 0; mask < (1u << n); ++mask) {
            const auto val = [&](Index l) {
                const int bit = static_cast<int>((mask >> (l >> 1)) & 1u);
                return (l & 1) ? bit : 1 - bit;
            };
            bool ok = true;
            for (const auto& [a, b] : edges) ok = ok && !(val(a) && val(b));
            if (!ok) continue;
            for (const auto& c : cuts) {
                f64 act = 0.0;
                for (std::size_t q = 0; q < c.cols.size(); ++q)
                    act += c.vals[q] * static_cast<f64>((mask >> c.cols[q]) & 1u);
                CHECK(act <= c.row_hi + 1e-12);
            }
        }
    }
    CHECK(found > 10);
}

// Clique-aware row propagation (clique_rows.hpp) never removes a feasible
// point. Random models: binaries x0..x5 and an integer z in [0,3]; two
// set-packing rows make two cliques; two random rows with mixed signs and
// sides. On random sub-boxes, every integer point of the box that satisfies
// all rows must survive propagation, and "infeasible" is reported only when
// none exists.
void test_clique_rows_are_sound() {
    std::mt19937 rng(777);
    int deductions = 0, infeasible_seen = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const Index nb = 6, n = 7;   // x0..x5 binary, z = column 6 in [0,3]
        std::vector<Index> ri, cj;
        std::vector<f64> v;
        LpProblem lp;
        lp.c.assign(n, 0.0);
        lp.col_lo.assign(n, 0.0);
        lp.col_hi.assign(n, 1.0);
        lp.col_hi[6] = 3.0;
        lp.is_integer.assign(n, true);
        Index row = 0;
        // Cliques {x0,x1,x2} and {x3,x4} as packing rows (x3 complemented).
        for (Index j : {0, 1, 2}) { ri.push_back(row); cj.push_back(j); v.push_back(1.0); }
        lp.row_lo.push_back(-kInf); lp.row_hi.push_back(1.0); ++row;
        ri.push_back(row); cj.push_back(3); v.push_back(-1.0);
        ri.push_back(row); cj.push_back(4); v.push_back(1.0);
        lp.row_lo.push_back(-kInf); lp.row_hi.push_back(0.0); ++row;   // x4 <= x3: (x3=0) & (x4=1) conflict
        for (int r = 0; r < 2; ++r, ++row) {
            for (Index j = 0; j < n; ++j) {
                if (rng() % 3 == 0) continue;
                f64 a = static_cast<f64>(1 + rng() % 5);
                if (rng() % 3 == 0) a = -a;
                ri.push_back(row); cj.push_back(j); v.push_back(a);
            }
            const f64 b = static_cast<f64>(static_cast<int>(rng() % 9) - 2);
            if (rng() % 2) { lp.row_lo.push_back(b); lp.row_hi.push_back(kInf); }
            else { lp.row_lo.push_back(-kInf); lp.row_hi.push_back(b + 4.0); }
        }
        lp.A = forge::sparse::from_triplets(row, n, ri, cj, v);
        ConflictGraph g;
        g.reset(n);
        g.mark_binaries(lp, lp.col_lo, lp.col_hi);
        g.add_edge(lit_of(0, 1), lit_of(1, 1));
        g.add_edge(lit_of(0, 1), lit_of(2, 1));
        g.add_edge(lit_of(1, 1), lit_of(2, 1));
        g.add_edge(lit_of(3, 0), lit_of(4, 1));
        g.sort_adjacency();
        forge::search::CliqueRowOptions co;
        const auto idx = forge::search::build_clique_row_index(lp, g, co);
        // A random sub-box.
        std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
        for (Index j = 0; j < n; ++j) {
            if (rng() % 4 == 0) { const f64 f = static_cast<f64>(rng() % 2); if (j < nb) lo[j] = hi[j] = f; }
        }
        if (rng() % 2) lo[6] = 1.0;
        const auto feasible_points = [&](const std::vector<f64>& L, const std::vector<f64>& H) {
            std::vector<std::vector<f64>> pts;
            for (std::uint32_t mask = 0; mask < (1u << nb); ++mask)
                for (int z = 0; z <= 3; ++z) {
                    std::vector<f64> x(n);
                    for (Index j = 0; j < nb; ++j) x[j] = static_cast<f64>((mask >> j) & 1u);
                    x[6] = z;
                    bool in = true;
                    for (Index j = 0; j < n; ++j) in = in && x[j] >= L[j] && x[j] <= H[j];
                    if (in && lp.max_row_violation(x) == 0.0) pts.push_back(x);
                }
            return pts;
        };
        const auto before = feasible_points(lo, hi);
        forge::search::PropTrail trail;
        forge::search::ReasonStore store;
        std::vector<Index> moved;
        const auto pr = forge::search::propagate_clique_rows(idx, lp, lo, hi, nullptr, lp.col_lo,
                                                           lp.col_hi, &trail, &store, 1, 1e-9, &moved);
        deductions += static_cast<int>(pr.tightened);
        if (!pr.feasible) {
            ++infeasible_seen;
            CHECK(before.empty());
            continue;
        }
        for (const auto& x : before)
            for (Index j = 0; j < n; ++j) CHECK(x[j] >= lo[j] - 1e-12 && x[j] <= hi[j] + 1e-12);
    }
    CHECK(deductions > 20);
}

// An implied bound t <= b0 (b = 0), t <= b1 (b = 1) with b0 = -2^54, b1 = 1:
// the rounded slope s = 2^54 made  t - s b <= b0  exclude (b, t) = (1, 1).
// The row must admit both endpoints exactly, and still cut a point violating
// the implication.
void test_implied_bound_cut_keeps_its_endpoints() {
    forge::search::ConflictGraph cg;
    LpProblem lp;
    lp.name = "vub_endpoints";
    lp.A = from_triplets(0, 2, {}, {}, {});
    lp.c = {0.0, 0.0};
    lp.col_lo = {-std::ldexp(1.0, 54), 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {false, true};
    cg.reset(2);
    cg.mark_binaries(lp, lp.col_lo, lp.col_hi);
    const f64 b0 = -std::ldexp(1.0, 54), b1 = 1.0;
    cg.add_implied_bound({1, 0, b0, b1, true});
    const std::vector<f64> x = {1.0, 0.5};   // t = 1 at b = 0.5: t <= b0 + 0.5 s violated
    const auto cuts = forge::search::separate_implied_bound_cuts(cg, x, {});
    CHECK(cuts.size() == 1);
    if (cuts.size() != 1) return;
    const auto& c = cuts[0];
    CHECK(c.exact);
    const auto activity = [&](f64 t, f64 b) {
        forge::model::DyadicSum a;
        for (std::size_t k = 0; k < c.cols.size(); ++k)
            a.add_product(c.vals[k], c.cols[k] == 0 ? t : b);
        return a;
    };
    CHECK(activity(1.0, 1.0).compare(c.row_hi) <= 0);   // b = 1 allows t = 1
    CHECK(activity(b0, 0.0).compare(c.row_hi) <= 0);    // b = 0 allows t = b0
    CHECK(activity(1.0, 0.5).compare(c.row_hi) > 0);    // the fractional point is cut
}

int main() {
    test_implied_bound_cut_keeps_its_endpoints();
    test_clique_rows_are_sound();
    test_odd_cycle_cuts();
    test_probing_dual_fix_uses_side_redundancy();
    test_row_cliques_use_exact_floor();
    test_edge_set_matches_reference();
    test_probing_budget_reaches_late_binaries();
    test_set_packing_row_gives_clique();
    test_precedence_row_gives_implication();
    test_probing_detects_infeasibility();
    test_probing_fixes_column();
    test_probing_hull_tightens_continuous();
    test_probing_records_implied_bound();
    test_dual_fix_probe_does_not_create_global_implied_bound();
    test_dual_fix_probe_does_not_create_conflict_edge();
    test_dual_fix_probe_random_scope_and_optimum();
    test_probe_preserves_wide_integer_side();
    test_random_models_preserve_all_feasible_points();
    test_random_separated_cuts_are_valid();
    test_random_conflict_propagation_keeps_feasible_points();
    return forge::test::finish("test_conflict");
}
