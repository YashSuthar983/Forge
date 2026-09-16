// Flow-cover cut validity tests - VUB projection + SI lifting.
#include "sor/search/flowcover.hpp"
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
using sor::search::CutRow;
using sor::search::FlowCoverDiagnostics;
using sor::search::FlowCoverOptions;
using sor::search::separate_flow_covers;
using sor::sparse::from_triplets;

namespace {

bool model_feasible(const LpProblem& lp, const std::vector<f64>& x) {
    for (Index j = 0; j < lp.n_cols(); ++j) {
        const auto u = static_cast<std::size_t>(j);
        if (x[u] < lp.col_lo[u] - 1e-9 || x[u] > lp.col_hi[u] + 1e-9)
            return false;
        if (!lp.is_integer.empty() && lp.is_integer[u] &&
            std::fabs(x[u] - std::round(x[u])) > 1e-9)
            return false;
    }
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        f64 a = 0.0;
        for (auto k = rp[static_cast<std::size_t>(i)];
             k < rp[static_cast<std::size_t>(i) + 1]; ++k)
            a += av[static_cast<std::size_t>(k)] *
                 x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
        if (a < lp.row_lo[static_cast<std::size_t>(i)] - 1e-8 ||
            a > lp.row_hi[static_cast<std::size_t>(i)] + 1e-8)
            return false;
    }
    return true;
}

f64 cut_lhs(const CutRow& c, const std::vector<f64>& x) {
    f64 s = 0.0;
    for (std::size_t k = 0; k < c.cols.size(); ++k)
        s += c.vals[k] * x[static_cast<std::size_t>(c.cols[k])];
    return s;
}

// Continuous LP-max check: for fixed binaries, max cut LHS over y in VUB/flow
// box should stay ≤ RHS (approx validity for the continuous hull).
bool cut_ok_lp_max_binaries(const LpProblem& lp, const CutRow& c,
                            f64 tol = 1e-5) {
    // Identify binary cols.
    std::vector<Index> bins;
    for (Index j = 0; j < lp.n_cols(); ++j)
        if (!lp.is_integer.empty() && lp.is_integer[static_cast<std::size_t>(j)] &&
            std::fabs(lp.col_lo[static_cast<std::size_t>(j)]) <= 1e-9 &&
            std::fabs(lp.col_hi[static_cast<std::size_t>(j)] - 1.0) <= 1e-9)
            bins.push_back(j);
    if (bins.size() > 12) return true;  // skip huge
    const std::uint64_t total = 1ull << bins.size();
    for (std::uint64_t code = 0; code < total; ++code) {
        std::vector<f64> p(static_cast<std::size_t>(lp.n_cols()), 0.0);
        for (std::size_t b = 0; b < bins.size(); ++b)
            p[static_cast<std::size_t>(bins[b])] =
                f64((code >> b) & 1ull);
        // Set continuous cols to upper if their linked binary is 1, else 0 -
        // using column bounds (VUB-feasible extreme).
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (!lp.is_integer.empty() &&
                lp.is_integer[static_cast<std::size_t>(j)])
                continue;
            // Heuristic extreme: 0 or hi (caller also checks model_feasible).
            p[static_cast<std::size_t>(j)] = 0.0;
        }
        // Try a few y extremes per binary assignment by scanning cut cols.
        std::vector<Index> cont_in_cut;
        for (Index j : c.cols)
            if (lp.is_integer.empty() ||
                !lp.is_integer[static_cast<std::size_t>(j)])
                cont_in_cut.push_back(j);
        const int qn = static_cast<int>(
            std::min<std::size_t>(cont_in_cut.size(), 3));
        const int qtot = 1 << std::max(qn, 0);
        for (int q = 0; q < qtot; ++q) {
            for (int t = 0; t < qn; ++t) {
                const Index j = cont_in_cut[static_cast<std::size_t>(t)];
                p[static_cast<std::size_t>(j)] =
                    ((q >> t) & 1)
                        ? lp.col_hi[static_cast<std::size_t>(j)]
                        : lp.col_lo[static_cast<std::size_t>(j)];
            }
            if (!model_feasible(lp, p)) continue;
            CHECK(cut_lhs(c, p) <= c.row_hi + tol);
        }
    }
    return true;
}

void test_two_arc_flow_cover() {
    // cols: y1, y2, x1, x2
    LpProblem lp;
    lp.name = "fc2";
    lp.A = from_triplets(
        3, 4,
        {0, 0, 1, 1, 2, 2},
        {0, 1, 0, 2, 1, 3},
        {1.0, 1.0, 1.0, -5.0, 1.0, -5.0});
    lp.c = {-1.0, -1.0, 0.0, 0.0};
    lp.row_lo = {-kInf, -kInf, -kInf};
    lp.row_hi = {6.0, 0.0, 0.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {5.0, 5.0, 1.0, 1.0};
    lp.is_integer = {false, false, true, true};

    const std::vector<f64> x = {3.0, 3.0, 0.6, 0.6};
    FlowCoverOptions o;
    FlowCoverDiagnostics d;
    const auto cuts = separate_flow_covers(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(d.structures_built >= 1);
    CHECK(!cuts.empty());

    for (const auto& c : cuts) {
        CHECK(cut_lhs(c, x) > c.row_hi + 1e-7);
        for (int mask = 0; mask < 4; ++mask) {
            const f64 x1 = f64(mask & 1);
            const f64 x2 = f64((mask >> 1) & 1);
            for (int q = 0; q < 3; ++q) {
                const f64 y1 = (q == 0 ? 0.0 : (q == 1 ? 2.5 : 5.0)) * x1;
                const f64 y2 = (q == 0 ? 0.0 : (q == 1 ? 2.5 : 5.0)) * x2;
                if (y1 + y2 > 6.0 + 1e-9) continue;
                std::vector<f64> p = {y1, y2, x1, x2};
                if (!model_feasible(lp, p)) continue;
                CHECK(cut_lhs(c, p) <= c.row_hi + 1e-6);
            }
        }
        cut_ok_lp_max_binaries(lp, c);
    }
}

// Three arcs: SI lifting should be able to bring the third arc into the cut.
void test_si_lift_three_arcs() {
    // y1+y2+y3 <= 10; yj <= 6 xj.
    // Cover {1,2} with u=6+6=12, λ=2; arc 3 is outside → SI candidate.
    LpProblem lp;
    lp.name = "fc3si";
    lp.A = from_triplets(
        4, 6,
        {0, 0, 0, 1, 1, 2, 2, 3, 3},
        {0, 1, 2, 0, 3, 1, 4, 2, 5},
        {1, 1, 1, 1, -6, 1, -6, 1, -6});
    lp.c.assign(6, -1.0);
    lp.row_lo.assign(4, -kInf);
    lp.row_hi = {10.0, 0.0, 0.0, 0.0};
    lp.col_lo.assign(6, 0.0);
    lp.col_hi = {6, 6, 6, 1, 1, 1};
    lp.is_integer = {false, false, false, true, true, true};

    // Fractional: all x=0.7, y=4.2 → capacity tight-ish.
    const std::vector<f64> x = {4.2, 4.2, 4.2, 0.7, 0.7, 0.7};
    FlowCoverOptions o;
    o.sequence_independent_lift = true;
    FlowCoverDiagnostics d;
    const auto cuts = separate_flow_covers(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(d.covers_found >= 1);
    // SI may or may not fire depending on cover choice; validity is mandatory.
    for (const auto& c : cuts) {
        CHECK(cut_lhs(c, x) > c.row_hi + 1e-9);
        for (int mask = 0; mask < 8; ++mask) {
            const f64 x1 = f64(mask & 1);
            const f64 x2 = f64((mask >> 1) & 1);
            const f64 x3 = f64((mask >> 2) & 1);
            for (f64 f : {0.0, 0.5, 1.0}) {
                const f64 y1 = 6.0 * f * x1;
                const f64 y2 = 6.0 * f * x2;
                const f64 y3 = 6.0 * f * x3;
                if (y1 + y2 + y3 > 10.0 + 1e-9) continue;
                std::vector<f64> p = {y1, y2, y3, x1, x2, x3};
                if (!model_feasible(lp, p)) continue;
                CHECK(cut_lhs(c, p) <= c.row_hi + 1e-5);
            }
        }
    }
}

// Projected VUB: y - 5x + z <= 0 with z continuous in [0,0] effectively, or
// z >= 0 moved at max → still recover y-5x <= 0 when z<=0 forced.
void test_projected_vub() {
    // row0: y1+y2 <= 6
    // row1: y1 - 5 x1 + z <= 0  with z in [0, 0] (fixed) → projects to VUB
    // row2: y2 - 5 x2 <= 0
    LpProblem lp;
    lp.name = "fc_proj";
    lp.A = from_triplets(
        3, 5,
        {0, 0, 1, 1, 1, 2, 2},
        {0, 1, 0, 2, 4, 1, 3},
        {1, 1, 1, -5, 1, 1, -5});
    lp.c.assign(5, 0.0);
    lp.row_lo = {-kInf, -kInf, -kInf};
    lp.row_hi = {6.0, 0.0, 0.0};
    lp.col_lo = {0, 0, 0, 0, 0};
    lp.col_hi = {5, 5, 1, 1, 0};  // z fixed at 0
    lp.is_integer = {false, false, true, true, false};

    const std::vector<f64> x = {3.0, 3.0, 0.6, 0.6, 0.0};
    FlowCoverOptions o;
    o.project_vubs = true;
    FlowCoverDiagnostics d;
    const auto cuts = separate_flow_covers(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(d.vubs_found >= 1);
    CHECK(d.vubs_projected >= 1 || !cuts.empty());
    for (const auto& c : cuts) {
        CHECK(cut_lhs(c, x) > c.row_hi + 1e-7);
        cut_ok_lp_max_binaries(lp, c);
    }
}

void test_no_vub_means_no_cut() {
    LpProblem lp;
    lp.name = "novub";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {1.0, 1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {3.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {5.0, 5.0};
    lp.is_integer = {false, false};
    const std::vector<f64> x = {2.0, 2.0};
    FlowCoverOptions o;
    FlowCoverDiagnostics d;
    const auto cuts = separate_flow_covers(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(cuts.empty());
}

}  // namespace

int main() {
    test_two_arc_flow_cover();
    test_si_lift_three_arcs();
    test_projected_vub();
    test_no_vub_means_no_cut();
    return sor::test::finish("test_flowcover");
}
