// Probing state carried between stages (conflict.hpp: ProbingState,
// ProbingCarry, ConflictGraph::remapped). Decidable by brute force on small
// binary models: enumerate every feasible point and check the facts.
//   * resume: probing under a small budget then resuming covers every binary
//     exactly once (total probes equal one uninterrupted pass) and leaves only
//     valid facts;
//   * remap: eliminate columns at values taken from a feasible point of the
//     original model; every remapped edge / clique / implied bound must hold on
//     ALL feasible points of the reduced model;
//   * carry validity: a wider box or a different matrix is refused.
#include "sor/search/conflict.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::search::ConflictGraph;
using sor::search::lit_of;
using sor::search::lit_val;
using sor::search::lit_var;
using sor::search::ProbingCarry;
using sor::search::ProbingOptions;
using sor::search::ProbingState;
using sor::sparse::from_triplets;

namespace {

LpProblem random_binary_model(std::mt19937& rng, Index n, Index m) {
    LpProblem lp;
    lp.name = "carry";
    std::vector<Index> r, c;
    std::vector<f64> v;
    lp.row_lo.assign(static_cast<std::size_t>(m), -kInf);
    lp.row_hi.assign(static_cast<std::size_t>(m), 0.0);
    for (Index i = 0; i < m; ++i) {
        f64 pos = 0.0;
        for (Index j = 0; j < n; ++j) {
            if (rng() % 3 != 0) continue;
            const f64 a = static_cast<f64>(static_cast<int>(rng() % 7) - 3);
            if (a == 0.0) continue;
            r.push_back(i); c.push_back(j); v.push_back(a);
            if (a > 0) pos += a;
        }
        lp.row_hi[static_cast<std::size_t>(i)] = std::floor(pos * 0.5) + static_cast<f64>(rng() % 2);
    }
    lp.A = from_triplets(m, n, r, c, v);
    lp.c.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), true);
    return lp;
}

std::vector<std::vector<f64>> feasible_points(const LpProblem& lp) {
    const Index n = lp.n_cols();
    std::vector<std::vector<f64>> out;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (std::uint64_t code = 0; code < (1ull << n); ++code) {
        std::vector<f64> x(static_cast<std::size_t>(n));
        for (Index j = 0; j < n; ++j) x[static_cast<std::size_t>(j)] = (code >> j) & 1u;
        bool ok = true;
        for (Index i = 0; i < lp.n_rows() && ok; ++i) {
            f64 act = 0.0;
            for (auto k = rp[static_cast<std::size_t>(i)]; k < rp[static_cast<std::size_t>(i) + 1]; ++k)
                act += lp.A.vals[static_cast<std::size_t>(k)] * x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            if (act < lp.row_lo[static_cast<std::size_t>(i)] - 1e-9 ||
                act > lp.row_hi[static_cast<std::size_t>(i)] + 1e-9) ok = false;
        }
        for (Index j = 0; j < n && ok; ++j)
            if (x[static_cast<std::size_t>(j)] < lp.col_lo[static_cast<std::size_t>(j)] - 1e-9 ||
                x[static_cast<std::size_t>(j)] > lp.col_hi[static_cast<std::size_t>(j)] + 1e-9) ok = false;
        if (ok) out.push_back(std::move(x));
    }
    return out;
}

bool lit_true(Index l, const std::vector<f64>& x) {
    const f64 xv = x[static_cast<std::size_t>(lit_var(l))];
    return lit_val(l) == 1 ? xv > 0.5 : xv < 0.5;
}

// Every fact of `g` holds on every point of `pts`.
bool facts_valid(const ConflictGraph& g, const std::vector<std::vector<f64>>& pts) {
    for (const auto& x : pts) {
        for (Index l = 0; l < 2 * g.n_cols(); ++l)
            for (const Index b : g.neighbors(l))
                if (lit_true(l, x) && lit_true(b, x)) return false;
        for (const auto& cl : g.cliques()) {
            int t = 0;
            for (const Index l : cl.lits) t += lit_true(l, x);
            if (t > 1) return false;
        }
        for (const auto& ib : g.implied_bounds()) {
            const int bv = x[static_cast<std::size_t>(ib.bin)] > 0.5 ? 1 : 0;
            const f64 bound = bv ? ib.b1 : ib.b0;
            const f64 xv = x[static_cast<std::size_t>(ib.col)];
            if (ib.upper ? xv > bound + 1e-9 : xv < bound - 1e-9) return false;
        }
    }
    return true;
}

void test_resume_covers_each_binary_once() {
    std::mt19937 rng(7);
    int checked = 0, resumed_after_stop = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const Index n = 5 + static_cast<Index>(rng() % 5);
        auto lp = random_binary_model(rng, n, 2 + static_cast<Index>(rng() % 4));
        const auto pts = feasible_points(lp);
        ProbingOptions po;
        po.give_up_unproductive = false;
        po.probe_time_limit_s = 0.0;

        // Reference: one uninterrupted pass.
        auto lo = lp.col_lo, hi = lp.col_hi;
        ConflictGraph full;
        ProbingState fs;
        const auto fd = sor::search::build_conflict_graph(lp, lo, hi, full, po, &fs);
        if (fd.infeasible) { CHECK(pts.empty()); continue; }
        CHECK(!pts.empty());

        // Interrupted after k binaries, then resumed on the (possibly tightened) box.
        auto lo2 = lp.col_lo, hi2 = lp.col_hi;
        ConflictGraph g;
        ProbingState st;
        ProbingOptions cut = po;
        cut.max_binaries_probed = 1 + static_cast<Index>(rng() % static_cast<unsigned>(n));
        const auto d1 = sor::search::build_conflict_graph(lp, lo2, hi2, g, cut, &st);
        CHECK(!d1.infeasible);
        if (!st.complete) ++resumed_after_stop;
        const auto d2 = sor::search::build_conflict_graph(lp, lo2, hi2, g, po, &st);
        CHECK(!d2.infeasible);
        CHECK(st.complete);
        // Together they probed no literal twice: at most one pass's worth.
        CHECK(d1.probes + d2.probes <= fd.probes);
        // A finished state makes a further pass probe nothing.
        const auto d3 = sor::search::build_conflict_graph(lp, lo2, hi2, g, po, &st);
        CHECK(d3.probes == 0);
        // Everything learned is valid, and the box keeps every feasible point.
        CHECK(facts_valid(g, pts));
        for (const auto& x : pts)
            for (Index j = 0; j < n; ++j) {
                CHECK(x[static_cast<std::size_t>(j)] >= lo2[static_cast<std::size_t>(j)] - 1e-9);
                CHECK(x[static_cast<std::size_t>(j)] <= hi2[static_cast<std::size_t>(j)] + 1e-9);
            }
        ++checked;
    }
    std::cout << "PROBING_RESUME checked=" << checked << " stopped_early=" << resumed_after_stop << '\n';
    CHECK(checked > 200);
    CHECK(resumed_after_stop > 50);
}

void test_remap_valid_on_reduced_model() {
    std::mt19937 rng(11);
    int checked = 0, facts = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const Index n = 6 + static_cast<Index>(rng() % 4);
        auto lp = random_binary_model(rng, n, 2 + static_cast<Index>(rng() % 4));
        const auto pts = feasible_points(lp);
        if (pts.empty()) continue;
        ProbingOptions po;
        po.give_up_unproductive = false;
        po.probe_time_limit_s = 0.0;
        auto lo = lp.col_lo, hi = lp.col_hi;
        ConflictGraph g;
        ProbingState st;
        if (sor::search::build_conflict_graph(lp, lo, hi, g, po, &st).infeasible) continue;

        // Eliminate a random subset at the values of one feasible point.
        const auto& ref = pts[rng() % pts.size()];
        std::vector<Index> new_of_old(static_cast<std::size_t>(n), -1);
        std::vector<Index> kept;
        for (Index j = 0; j < n; ++j)
            if (rng() % 3 != 0) { new_of_old[static_cast<std::size_t>(j)] = static_cast<Index>(kept.size()); kept.push_back(j); }
        if (kept.size() < 2 || kept.size() == static_cast<std::size_t>(n)) continue;
        const Index nk = static_cast<Index>(kept.size());
        LpProblem red;
        red.name = "reduced";
        std::vector<Index> r, c;
        std::vector<f64> v;
        red.row_lo = lp.row_lo;
        red.row_hi = lp.row_hi;
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        for (Index i = 0; i < lp.n_rows(); ++i)
            for (auto k = rp[static_cast<std::size_t>(i)]; k < rp[static_cast<std::size_t>(i) + 1]; ++k) {
                const Index j = ci[static_cast<std::size_t>(k)];
                const f64 a = lp.A.vals[static_cast<std::size_t>(k)];
                const Index f = new_of_old[static_cast<std::size_t>(j)];
                if (f >= 0) { r.push_back(i); c.push_back(f); v.push_back(a); }
                else {
                    const f64 shift = a * ref[static_cast<std::size_t>(j)];
                    red.row_lo[static_cast<std::size_t>(i)] -= shift;
                    red.row_hi[static_cast<std::size_t>(i)] -= shift;
                }
            }
        red.A = from_triplets(lp.n_rows(), nk, r, c, v);
        red.c.assign(static_cast<std::size_t>(nk), 0.0);
        red.col_lo.assign(static_cast<std::size_t>(nk), 0.0);
        red.col_hi.assign(static_cast<std::size_t>(nk), 1.0);
        red.is_integer.assign(static_cast<std::size_t>(nk), true);
        const auto rpts = feasible_points(red);
        CHECK(!rpts.empty());   // ref survives

        const ConflictGraph rg = g.remapped(new_of_old, nk);
        const ProbingState rs = st.remapped(new_of_old, nk);
        CHECK(rs.n_cols == nk);
        for (Index a = 0; a < n; ++a) {
            const Index f = new_of_old[static_cast<std::size_t>(a)];
            if (f >= 0) CHECK(rs.probed[static_cast<std::size_t>(f)] == st.probed[static_cast<std::size_t>(a)]);
        }
        CHECK(facts_valid(rg, rpts));
        facts += static_cast<int>(rg.n_edges() + rg.cliques().size() + rg.implied_bounds().size());
        ++checked;
    }
    std::cout << "PROBING_REMAP checked=" << checked << " facts=" << facts << '\n';
    CHECK(checked > 150);
    CHECK(facts > 300);
}

void test_carry_validity() {
    std::mt19937 rng(3);
    auto lp = random_binary_model(rng, 6, 3);
    ProbingCarry carry;
    ProbingOptions po;
    auto lo = lp.col_lo, hi = lp.col_hi;
    (void)sor::search::build_conflict_graph(lp, lo, hi, carry.graph, po, &carry.state);
    carry.lo = lo;
    carry.hi = hi;
    carry.fingerprint = sor::search::matrix_fingerprint(lp);
    auto same = lp;
    same.col_lo = lo;
    same.col_hi = hi;
    CHECK(carry.usable_for(same));
    auto narrower = same;
    for (Index j = 0; j < narrower.n_cols(); ++j)
        if (narrower.col_hi[static_cast<std::size_t>(j)] > narrower.col_lo[static_cast<std::size_t>(j)])
            { narrower.col_hi[static_cast<std::size_t>(j)] = narrower.col_lo[static_cast<std::size_t>(j)]; break; }
    CHECK(carry.usable_for(narrower));
    auto wider = same;
    wider.col_hi[0] += 1.0;
    CHECK(!carry.usable_for(wider));
    auto other = same;
    if (!other.A.vals.empty()) other.A.vals[0] += 1.0;
    CHECK(!carry.usable_for(other));
    ProbingCarry empty;
    CHECK(!empty.usable_for(same));
}

}  // namespace

int main() {
    test_resume_covers_each_binary_once();
    test_remap_valid_on_reduced_model();
    test_carry_validity();
    return sor::test::finish("test_probing_carry");
}
