// Cut provenance: a cut a separator reports as global (CutRow::used_local_bound
// false) must hold over the ROOT box, although it was derived under a node
// box. The tree promotes such cuts into the global LP, so a wrong report is a
// cut that removes feasible points outside the subtree that made it: a false
// Optimal or a false Infeasible.
//
// Brute force, as for the separators' own validity tests: small pure-integer
// models, every integer point of the root box enumerated, node boxes made by
// tightening root bounds the way branching does, and each separator run at the
// node LP's optimum with the root box handed in.
#include "forge/engines/simplex.hpp"
#include "forge/search/covers.hpp"
#include "forge/search/cuts.hpp"
#include "forge/search/mir.hpp"
#include "forge/search/zerohalf.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using forge::core::f64;
using forge::core::Index;
using forge::model::kInf;
using forge::model::LpProblem;
using forge::search::CutRow;

namespace {

std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

bool rows_hold(const LpProblem& lp, const std::vector<f64>& x) {
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        f64 a = 0.0;
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            a += av[sz(k)] * x[sz(ci[sz(k)])];
        if (a < lp.row_lo[sz(i)] - 1e-9 || a > lp.row_hi[sz(i)] + 1e-9) return false;
    }
    return true;
}

bool cut_holds(const CutRow& c, const std::vector<f64>& x) {
    f64 a = 0.0;
    for (std::size_t k = 0; k < c.cols.size(); ++k) a += c.vals[k] * x[sz(c.cols[k])];
    const f64 tol = 1e-6 * (1.0 + std::fabs(a));
    return a >= c.row_lo - tol && a <= c.row_hi + tol;
}

// Every integer point of [lo, hi] satisfying the rows of `lp`.
std::vector<std::vector<f64>> integer_points(const LpProblem& lp, const std::vector<f64>& lo,
                                             const std::vector<f64>& hi) {
    std::vector<std::vector<f64>> pts;
    const Index n = lp.n_cols();
    std::vector<f64> x(lo.begin(), lo.end());
    for (;;) {
        if (rows_hold(lp, x)) pts.push_back(x);
        Index j = 0;
        while (j < n && x[sz(j)] + 1.0 > hi[sz(j)]) { x[sz(j)] = lo[sz(j)]; ++j; }
        if (j == n) break;
        x[sz(j)] += 1.0;
    }
    return pts;
}

struct Tally {
    int global = 0;
    int local = 0;
};

// Every cut reported global holds on every root point; every cut holds on
// every node point (the derivation itself, under the box it was run in).
void judge(const char* family, const std::vector<CutRow>& cuts,
           const std::vector<std::vector<f64>>& root_pts,
           const std::vector<f64>& node_lo, const std::vector<f64>& node_hi, Tally& t,
           const std::string& where) {
    for (const CutRow& c : cuts) {
        for (const auto& p : root_pts) {
            bool in_node = true;
            for (std::size_t j = 0; j < p.size(); ++j)
                if (p[j] < node_lo[j] || p[j] > node_hi[j]) in_node = false;
            if (in_node && !cut_holds(c, p)) {
                forge::test::report(false, "cut holds on its node box", __FILE__, __LINE__,
                                  std::string(family) + " " + where);
                break;
            }
            if (!c.used_local_bound && !cut_holds(c, p)) {
                forge::test::report(false, "cut reported global holds on the root box",
                                  __FILE__, __LINE__, std::string(family) + " " + where);
                break;
            }
        }
        if (c.used_local_bound) ++t.local;
        else ++t.global;
    }
}

void test_random_models() {
    std::mt19937 rng(20261008);
    std::uniform_int_distribution<int> coef(-4, 4);
    Tally gmi, mir, cov, zh;
    int solved = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const Index n = 3 + static_cast<Index>(rng() % 3);
        const Index m = 1 + static_cast<Index>(rng() % 3);
        std::vector<f64> root_lo(sz(n)), root_hi(sz(n));
        for (Index j = 0; j < n; ++j) {
            // Some columns start below zero: zero-half's premise x >= 0 then
            // holds at a node only.
            root_lo[sz(j)] = (rng() % 4 == 0) ? -1.0 : 0.0;
            // Half binaries, so that covers have items to work with.
            root_hi[sz(j)] = (rng() % 2) ? 1.0 : static_cast<f64>(1 + rng() % 3);
        }
        std::vector<Index> ri, cj;
        std::vector<f64> v;
        LpProblem lp;
        for (Index i = 0; i < m; ++i) {
            f64 big = 0.0;
            for (Index j = 0; j < n; ++j) {
                const int a = coef(rng);
                if (a == 0) continue;
                ri.push_back(i);
                cj.push_back(j);
                v.push_back(a);
                big += std::fabs(a) * root_hi[sz(j)];
            }
            lp.row_lo.push_back(-kInf);
            lp.row_hi.push_back(std::floor(big * (0.3 + 0.4 * (rng() % 100) / 100.0)));
        }
        lp.A = forge::sparse::from_triplets(m, n, ri, cj, v);
        lp.is_integer.assign(sz(n), true);
        lp.c.resize(sz(n));
        for (Index j = 0; j < n; ++j) lp.c[sz(j)] = -static_cast<f64>(1 + rng() % 5);

        // Branching: tighten one or two columns.
        std::vector<f64> lo = root_lo, hi = root_hi;
        const int moves = 1 + static_cast<int>(rng() % 2);
        for (int k = 0; k < moves; ++k) {
            const Index j = static_cast<Index>(rng() % static_cast<unsigned>(n));
            if (hi[sz(j)] <= lo[sz(j)]) continue;
            if (rng() % 2) lo[sz(j)] += 1.0;
            else hi[sz(j)] -= 1.0;
        }
        lp.col_lo = lo;
        lp.col_hi = hi;

        forge::engines::SimplexOptions so;
        so.presolve = false;
        forge::engines::SimplexDiagnostics sd;
        forge::engines::SimplexBasis basis;
        const auto raw = forge::engines::solve_simplex(lp, so, sd, &basis);
        if (raw.proposed_status != forge::core::Status::Optimal ||
            static_cast<Index>(raw.x.size()) != n)
            continue;
        ++solved;
        LpProblem root_lp = lp;
        root_lp.col_lo = root_lo;
        root_lp.col_hi = root_hi;
        const auto pts = integer_points(root_lp, root_lo, root_hi);
        const std::string where = "trial " + std::to_string(trial);

        for (const bool tableau : {false, true}) {
            forge::search::CutOptions co;
            co.relax_small_terms = true;
            co.tableau_cmir = tableau;
            co.dynamism_max = 1e6;
            co.violation_min = 1e-6;
            forge::search::CutDiagnostics cd;
            judge("GMI", forge::search::separate_gomory_mi(lp, raw.x, basis, co, cd, &root_lo,
                                                         &root_hi),
                  pts, lo, hi, gmi, where);
        }
        for (const bool cover : {false, true}) {
            forge::search::MirOptions mo;
            mo.relax_small_terms = true;
            mo.violation_min = 1e-6;
            mo.lifted_cover = cover;
            mo.aggregate = true;
            forge::search::MirDiagnostics md;
            judge("MIR", forge::search::separate_mir(lp, raw.x, lo, hi, mo, md, &root_lo,
                                                   &root_hi),
                  pts, lo, hi, mir, where);
        }
        forge::search::CoverOptions cvo;
        forge::search::CoverDiagnostics cvd;
        judge("COV", forge::search::separate_lifted_covers(lp, raw.x, lo, hi, cvo, cvd, &root_lo,
                                                         &root_hi),
              pts, lo, hi, cov, where);
        forge::search::ZeroHalfOptions zo;
        forge::search::ZeroHalfDiagnostics zd;
        judge("ZH", forge::search::separate_zerohalf(lp, raw.x, lo, hi, zo, zd, &root_lo, &root_hi),
              pts, lo, hi, zh, where);
    }
    std::printf("  solved %d node LPs; global/local cuts: GMI %d/%d MIR %d/%d COV %d/%d ZH %d/%d\n",
                solved, gmi.global, gmi.local, mir.global, mir.local, cov.global, cov.local,
                zh.global, zh.local);
    // Not vacuous: each family reported both kinds somewhere.
    CHECK(gmi.global > 0 && gmi.local > 0);
    CHECK(mir.global > 0 && mir.local > 0);
    CHECK(cov.global > 0 && cov.local > 0);
    CHECK(zh.global > 0 && zh.local > 0);
}

// The MIR lifted cover reads a column as 0/1 from its node box. z has root box
// [0, 5] and node box [0, 1]; it is substituted at its lower bound 0, which the
// root box implies, but the cover also uses z <= 1, which it does not. The
// cover  x0 + z <= 1  excludes the root point (0, 2).
void test_mir_lifted_cover_width() {
    LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {3.0, 2.0});
    lp.row_lo = {-kInf};
    lp.row_hi = {4.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.c = {-1.0, -1.0};
    lp.is_integer = {true, true};
    const std::vector<f64> root_lo = {0.0, 0.0}, root_hi = {1.0, 5.0};
    const std::vector<f64> x = {1.0, 0.4};
    forge::search::MirOptions mo;
    mo.violation_min = 1e-6;
    mo.lifted_cover = true;   // opt-in in the solver
    forge::search::MirDiagnostics md;
    const auto cuts = forge::search::separate_mir(lp, x, lp.col_lo, lp.col_hi, mo, md,
                                                &root_lo, &root_hi);
    CHECK(!cuts.empty());
    LpProblem root_lp = lp;
    root_lp.col_lo = root_lo;
    root_lp.col_hi = root_hi;
    const auto pts = integer_points(root_lp, root_lo, root_hi);
    Tally t;
    judge("MIR", cuts, pts, lp.col_lo, lp.col_hi, t, "lifted cover width");
}

// A cover whose knapsack moved a node-fixed column into its capacity is local;
// the same row under the root box is global.
void test_cover_fixed_column() {
    LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, 4, {0, 0, 0, 0}, {0, 1, 2, 3}, {3.0, 3.0, 3.0, 2.0});
    lp.row_lo = {-kInf};
    lp.row_hi = {7.0};
    lp.col_lo = {0.0, 0.0, 0.0, 1.0};   // y fixed at 1 by branching
    lp.col_hi = {1.0, 1.0, 1.0, 1.0};
    lp.c = {-1.0, -1.0, -1.0, 0.0};
    lp.is_integer = {true, true, true, true};
    const std::vector<f64> root_lo = {0.0, 0.0, 0.0, 0.0}, root_hi = {1.0, 1.0, 1.0, 1.0};
    // x0 + x1 <= 1 is violated here, and excludes the root point (1, 1, 0, 0).
    const std::vector<f64> x = {0.8, 0.8, 0.0, 1.0};
    forge::search::CoverOptions o;
    forge::search::CoverDiagnostics d;
    const auto node = forge::search::separate_lifted_covers(lp, x, lp.col_lo, lp.col_hi, o, d,
                                                          &root_lo, &root_hi);
    CHECK(!node.empty());
    for (const CutRow& c : node) CHECK(c.used_local_bound);
    // Under the root box nothing local was consumed.
    forge::search::CoverDiagnostics d2;
    const std::vector<f64> xr = {0.9, 0.9, 0.9, 0.0};
    const auto root = forge::search::separate_lifted_covers(lp, xr, root_lo, root_hi, o, d2,
                                                          &root_lo, &root_hi);
    CHECK(!root.empty());
    for (const CutRow& c : root) CHECK(!c.used_local_bound);
}

// Zero-half halves 3 x0 + 2 x1 <= 3 into x0 + x1 <= 1, rounding 3/2 x0 down
// to x0, which needs x0 >= 0. The node box has it; the root box x0 in [-1, 1]
// does not, and (-1, 3) is a root point the cut excludes.
void test_zerohalf_nonnegativity() {
    LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {3.0, 2.0});
    lp.row_lo = {-kInf};
    lp.row_hi = {3.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 3.0};
    lp.c = {-1.0, -1.0};
    lp.is_integer = {true, true};
    const std::vector<f64> root_lo = {-1.0, 0.0}, root_hi = {1.0, 3.0};
    const std::vector<f64> x = {0.5, 0.75};
    forge::search::ZeroHalfOptions o;
    forge::search::ZeroHalfDiagnostics d;
    const auto cuts = forge::search::separate_zerohalf(lp, x, lp.col_lo, lp.col_hi, o, d,
                                                     &root_lo, &root_hi);
    CHECK(!cuts.empty());
    LpProblem root_lp = lp;
    root_lp.col_lo = root_lo;
    root_lp.col_hi = root_hi;
    Tally t;
    judge("ZH", cuts, integer_points(root_lp, root_lo, root_hi), lp.col_lo, lp.col_hi, t,
          "nonnegativity");
    CHECK(t.local > 0);
}

}  // namespace

int main() {
    test_zerohalf_nonnegativity();
    test_mir_lifted_cover_width();
    test_cover_fixed_column();
    test_random_models();
    return forge::test::finish("test_cut_provenance");
}
