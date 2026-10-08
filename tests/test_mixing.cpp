// Mixing inequalities over a shared conditional activity (search/mixing.hpp).
//   * the example: z >= 1 - y1, z >= 2 (1 - y2), z >= 0; at (z, y1, y2) =
//     (1, 0, 0.5) the chain row z + y1 + y2 >= 2 is found;
//   * validity, by enumeration: every row separated from random fractional
//     points of random families (both literal orientations, inexact
//     thresholds) holds exactly at every integer-feasible point;
//   * optimum: random fixed-charge models solve to the same value with the
//     separator on and off.
#include "sor/certify/finalize.hpp"
#include "sor/model/dyadic.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/mixing.hpp"
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
using namespace sor::search;

namespace {

std::size_t z(Index i) { return static_cast<std::size_t>(i); }

// Column 0 is z >= 0 (continuous), columns 1..k binaries. Row i:
//   kind 0: a z + b y_i >= r  with 0 < r <= b   (z >= (r/a)(1 - y_i))
//   kind 1: a z - b y_i >= 0                    (z >= (b/a) y_i)
LpProblem family_model(const std::vector<f64>& a, const std::vector<f64>& b,
                       const std::vector<f64>& r, const std::vector<int>& kind) {
    const Index k = static_cast<Index>(a.size());
    LpProblem lp;
    lp.name = "mixing";
    std::vector<Index> ri, cj;
    std::vector<f64> v;
    for (Index i = 0; i < k; ++i) {
        ri.push_back(i); cj.push_back(0); v.push_back(a[z(i)]);
        ri.push_back(i); cj.push_back(i + 1); v.push_back(kind[z(i)] == 0 ? b[z(i)] : -b[z(i)]);
        lp.row_lo.push_back(kind[z(i)] == 0 ? r[z(i)] : 0.0);
        lp.row_hi.push_back(kInf);
    }
    lp.A = sor::sparse::from_triplets(k, k + 1, ri, cj, v);
    lp.c.assign(z(k + 1), 0.0);
    lp.col_lo.assign(z(k + 1), 0.0);
    lp.col_hi.assign(z(k + 1), 1.0);
    lp.col_hi[0] = 1e6;
    lp.is_integer.assign(z(k + 1), true);
    lp.is_integer[0] = false;
    return lp;
}

void test_example_chain() {
    const auto lp = family_model({1.0, 1.0}, {1.0, 2.0}, {1.0, 2.0}, {0, 0});
    const auto idx = build_mixing_index(lp);
    CHECK(idx.families.size() == 1);
    const auto cuts = separate_mixing_cuts(idx, {1.0, 0.0, 0.5}, 10, 1e-6);
    CHECK(cuts.size() == 1);
    if (cuts.size() != 1) return;
    const auto& c = cuts[0];
    CHECK(c.exact);
    CHECK(c.row_lo == 2.0);
    f64 coef[3] = {0, 0, 0};
    for (std::size_t q = 0; q < c.cols.size(); ++q) coef[z(c.cols[q])] = c.vals[q];
    CHECK(coef[0] == 1.0 && coef[1] == 1.0 && coef[2] == 1.0);
}

void test_rows_hold_at_every_integer_point() {
    std::mt19937 rng(17);
    int cuts_checked = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const int k = 2 + static_cast<int>(rng() % 5);
        std::vector<f64> a(z(k)), b(z(k)), r(z(k));
        std::vector<int> kind(z(k));
        for (int i = 0; i < k; ++i) {
            a[z(i)] = 0.5 + static_cast<f64>(rng() % 7) / 3.0;   // inexact thresholds
            b[z(i)] = 1.0 + static_cast<f64>(rng() % 9) / 7.0;
            kind[z(i)] = static_cast<int>(rng() % 2);
            r[z(i)] = b[z(i)] * (0.2 + 0.8 * static_cast<f64>(rng() % 5) / 4.0);
        }
        const auto lp = family_model(a, b, r, kind);
        const auto idx = build_mixing_index(lp);
        if (idx.empty()) continue;
        for (int p = 0; p < 20; ++p) {
            std::vector<f64> x(z(k + 1));
            for (int i = 1; i <= k; ++i) x[z(i)] = static_cast<f64>(rng() % 101) / 100.0;
            x[0] = static_cast<f64>(rng() % 101) / 100.0;
            for (const auto& c : separate_mixing_cuts(idx, x, 10, 1e-9)) {
                ++cuts_checked;
                // Every binary assignment with its smallest feasible z.
                for (std::uint32_t mask = 0; mask < (1u << k); ++mask) {
                    sor::model::DyadicSum zmin;   // max over rows of the exact threshold
                    f64 zv = 0.0;
                    for (int i = 0; i < k; ++i) {
                        const f64 y = (mask >> i) & 1u ? 1.0 : 0.0;
                        // a z >= r - b y (kind 0) or a z >= b y (kind 1): z >= rhs / a.
                        sor::model::DyadicSum rhs;
                        if (kind[z(i)] == 0) { rhs.add(r[z(i)]); rhs.add_product(-b[z(i)], y); }
                        else rhs.add_product(b[z(i)], y);
                        const f64 need = rhs.quotient_up(a[z(i)]);
                        zv = std::max(zv, need);
                    }
                    (void)zmin;
                    std::vector<f64> pt(z(k + 1));
                    pt[0] = zv;
                    for (int i = 0; i < k; ++i) pt[z(i + 1)] = (mask >> i) & 1u ? 1.0 : 0.0;
                    sor::model::DyadicSum act;
                    for (std::size_t q = 0; q < c.cols.size(); ++q) act.add_product(c.vals[q], pt[z(c.cols[q])]);
                    // zv may sit an ulp above the exact minimum; the row must
                    // hold there (and so at every larger z).
                    CHECK(act.compare(c.row_lo) >= 0);
                }
            }
        }
    }
    std::cout << "MIXING_VALIDITY cuts=" << cuts_checked << '\n';
    CHECK(cuts_checked > 200);
}

// Fixed charge: open facility i (y_i, cost f_i) or pay through the shared
// slack z; demand rows z >= h_i (1 - y_i); minimize f'y + z.
void test_optimum_survives() {
    std::mt19937 rng(23);
    int compared = 0, with_cuts = 0;
    for (int trial = 0; trial < 60; ++trial) {
        const int k = 4 + static_cast<int>(rng() % 5);
        std::vector<f64> a(z(k), 1.0), b(z(k)), r(z(k));
        std::vector<int> kind(z(k), 0);
        for (int i = 0; i < k; ++i) {
            b[z(i)] = 1.0 + static_cast<f64>(rng() % 20);
            r[z(i)] = b[z(i)];
        }
        auto lp = family_model(a, b, r, kind);
        lp.c[0] = 1.0;
        for (int i = 1; i <= k; ++i) lp.c[z(i)] = 0.5 + static_cast<f64>(rng() % 15);
        f64 obj[2];
        for (int on = 0; on < 2; ++on) {
            BabOptions o;
            o.para_bab.threads = 1;
            o.gap_tol = o.abs_gap_tol = 1e-9;
            o.mixing_cuts = on == 1;
            o.structural_presolve.enabled = false;
            BabDiagnostics d;
            auto raw = solve_milp(lp, o, d);
            const auto res = sor::certify::finalize_result(std::move(raw), milp_evidence(d, o));
            CHECK(res.status == sor::core::Status::Optimal);
            obj[on] = res.objective;
            if (on == 1 && d.mixing_cut_candidates > 0) ++with_cuts;
        }
        CHECK(std::fabs(obj[0] - obj[1]) <= 1e-7);
        ++compared;
    }
    std::cout << "MIXING_OPTIMUM compared=" << compared << " with_cuts=" << with_cuts << '\n';
    CHECK(compared == 60);
}

}  // namespace

int main() {
    test_example_chain();
    test_rows_hold_at_every_integer_point();
    test_optimum_survives();
    return sor::test::finish("test_mixing");
}
