// Global nonconvex QP (GLB-1/GLB-5/GLB-2): hand-checked optima, a brute-force
// cross-check on random box QPs, and the soundness of every bound helper.
#include "sor/search/global_qp.hpp"
#include "sor/sparse/csr.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using sor::engines::QpProblem;
using sor::search::GlobalQpOptions;
using sor::search::GlobalQpResult;
using sor::search::GlobalRelaxation;
using sor::search::solve_global_qp;
using Index = sor::core::Index;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// min 0.5 x'Qx + c'x over rows and box.  Q given dense, row-major.
QpProblem make(int n, const std::vector<double>& Qd, const std::vector<double>& c,
               const std::vector<double>& lo, const std::vector<double>& hi,
               const std::vector<std::vector<double>>& A = {},
               const std::vector<double>& rlo = {}, const std::vector<double>& rhi = {}) {
    QpProblem p;
    auto& L = p.linear;
    L.name = "t";
    L.c = c;
    L.col_lo = lo;
    L.col_hi = hi;
    L.is_integer.assign(static_cast<std::size_t>(n), false);
    std::vector<Index> r, cc;
    std::vector<double> v;
    for (std::size_t i = 0; i < A.size(); ++i)
        for (int j = 0; j < n; ++j)
            if (A[i][static_cast<std::size_t>(j)] != 0.0) {
                r.push_back(static_cast<Index>(i)); cc.push_back(j);
                v.push_back(A[i][static_cast<std::size_t>(j)]);
            }
    L.A = sor::sparse::from_triplets(static_cast<Index>(A.size()), n, r, cc, v);
    L.row_lo = rlo;
    L.row_hi = rhi;
    std::vector<Index> qr, qc;
    std::vector<double> qv;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const double q = Qd[static_cast<std::size_t>(i * n + j)];
            if (q != 0.0) { qr.push_back(i); qc.push_back(j); qv.push_back(q); }
        }
    p.q_matrix = sor::sparse::from_triplets(n, n, qr, qc, qv);
    return p;
}

GlobalQpOptions quiet(GlobalRelaxation rel = GlobalRelaxation::Auto) {
    GlobalQpOptions o;
    o.time_limit_s = 20.0;
    o.relaxation = rel;
    return o;
}

// Proved, and within tolerance of the hand-checked optimum, under every
// relaxation choice: a bound error in any one of them would show up as a
// wrong proved value here.
void expect_global(const QpProblem& p, double fstar, const char* what) {
    for (auto rel : {GlobalRelaxation::McCormick, GlobalRelaxation::Shift, GlobalRelaxation::Both}) {
        const GlobalQpResult r = solve_global_qp(p, quiet(rel));
        const bool ok = r.proved && r.have_incumbent &&
                        std::fabs(r.incumbent - fstar) <= 1e-6 * (1.0 + std::fabs(fstar)) &&
                        r.bound <= fstar + 1e-9 * (1.0 + std::fabs(fstar));
        CHECK(ok);
        if (!ok)
            std::fprintf(stderr, "  %s rel=%d: proved=%d inc=%.12g bound=%.12g want %.12g (%s)\n",
                         what, static_cast<int>(rel), r.proved, r.incumbent, r.bound, fstar,
                         r.reason.c_str());
        // The certified bound may never exceed the true optimum.
        if (r.bound_valid) CHECK(r.bound <= fstar + 1e-9 * (1.0 + std::fabs(fstar)));
        // A PSD cut is an inequality on the moment matrix of a REAL point, so
        // the root bound it produces is still a bound: it may not pass the
        // true optimum.  This is the one way a wrong cut would show up.
        if (r.psd_cuts > 0) {
            const bool psd_ok = r.root_bound_psd <= fstar + 1e-7 * (1.0 + std::fabs(fstar)) &&
                                r.root_bound_psd >= r.root_bound_lp - 1e-7 * (1.0 + std::fabs(fstar));
            CHECK(psd_ok);
            if (!psd_ok)
                std::fprintf(stderr, "  %s: psd root bound %.12g outside [lp %.12g, f* %.12g]\n",
                             what, r.root_bound_psd, r.root_bound_lp, fstar);
        }
    }
}

// Exact global minimum of a box QP by enumerating every KKT face: each
// variable at its lower bound, upper bound, or free; free variables solve
// Q_FF x_F = -(c_F + Q_FB x_B).  A global minimiser of a box QP is a
// stationary point of the face it lies in, and when Q_FF is singular a
// minimiser also exists on a smaller face, so nonsingular faces suffice.
double box_qp_brute(int n, const std::vector<double>& Q, const std::vector<double>& c,
                    const std::vector<double>& lo, const std::vector<double>& hi) {
    double best = kInf;
    int total = 1;
    for (int i = 0; i < n; ++i) total *= 3;
    for (int code = 0; code < total; ++code) {
        std::vector<int> st(static_cast<std::size_t>(n));
        int cc = code;
        for (int i = 0; i < n; ++i) { st[static_cast<std::size_t>(i)] = cc % 3; cc /= 3; }
        std::vector<double> x(static_cast<std::size_t>(n), 0.0);
        std::vector<int> F;
        for (int i = 0; i < n; ++i) {
            const auto s = static_cast<std::size_t>(i);
            if (st[s] == 0) x[s] = lo[s];
            else if (st[s] == 1) x[s] = hi[s];
            else F.push_back(i);
        }
        const int k = static_cast<int>(F.size());
        if (k > 0) {
            std::vector<double> M(static_cast<std::size_t>(k * (k + 1)));
            for (int a = 0; a < k; ++a) {
                double rhs = -c[static_cast<std::size_t>(F[static_cast<std::size_t>(a)])];
                for (int j = 0; j < n; ++j)
                    if (st[static_cast<std::size_t>(j)] != 2)
                        rhs -= Q[static_cast<std::size_t>(F[static_cast<std::size_t>(a)] * n + j)] *
                               x[static_cast<std::size_t>(j)];
                for (int b = 0; b < k; ++b)
                    M[static_cast<std::size_t>(a * (k + 1) + b)] =
                        Q[static_cast<std::size_t>(F[static_cast<std::size_t>(a)] * n +
                                                   F[static_cast<std::size_t>(b)])];
                M[static_cast<std::size_t>(a * (k + 1) + k)] = rhs;
            }
            bool sing = false;
            for (int col = 0; col < k && !sing; ++col) {
                int piv = col;
                for (int r = col + 1; r < k; ++r)
                    if (std::fabs(M[static_cast<std::size_t>(r * (k + 1) + col)]) >
                        std::fabs(M[static_cast<std::size_t>(piv * (k + 1) + col)]))
                        piv = r;
                if (std::fabs(M[static_cast<std::size_t>(piv * (k + 1) + col)]) < 1e-12) { sing = true; break; }
                for (int b = 0; b <= k; ++b)
                    std::swap(M[static_cast<std::size_t>(col * (k + 1) + b)],
                              M[static_cast<std::size_t>(piv * (k + 1) + b)]);
                for (int r = 0; r < k; ++r) {
                    if (r == col) continue;
                    const double f = M[static_cast<std::size_t>(r * (k + 1) + col)] /
                                     M[static_cast<std::size_t>(col * (k + 1) + col)];
                    for (int b = 0; b <= k; ++b)
                        M[static_cast<std::size_t>(r * (k + 1) + b)] -= f * M[static_cast<std::size_t>(col * (k + 1) + b)];
                }
            }
            if (sing) continue;
            bool inside = true;
            for (int a = 0; a < k; ++a) {
                const auto j = static_cast<std::size_t>(F[static_cast<std::size_t>(a)]);
                x[j] = M[static_cast<std::size_t>(a * (k + 1) + k)] / M[static_cast<std::size_t>(a * (k + 1) + a)];
                if (x[j] < lo[j] - 1e-12 || x[j] > hi[j] + 1e-12) inside = false;
            }
            if (!inside) continue;
        }
        double f = 0.0;
        for (int i = 0; i < n; ++i) {
            f += c[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
            for (int j = 0; j < n; ++j)
                f += 0.5 * Q[static_cast<std::size_t>(i * n + j)] * x[static_cast<std::size_t>(i)] *
                     x[static_cast<std::size_t>(j)];
        }
        best = std::min(best, f);
    }
    return best;
}

std::uint64_t lcg_state = 12345;
double urand() {   // deterministic, in [0, 1)
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(lcg_state >> 11) / 9007199254740992.0;
}

}  // namespace

int main() {
    // 1. min -x^2 on [0,1]: -1 at x = 1 (the textbook concave case).
    expect_global(make(1, {-2.0}, {0.0}, {0.0}, {1.0}), -1.0, "min -x^2");

    // 2. min -(x-0.3)^2 on [0,1] = -x^2 + 0.6x - 0.09 -> -0.49 at x = 1.
    {
        auto p = make(1, {-2.0}, {0.6}, {0.0}, {1.0});
        p.linear.obj_offset = -0.09;
        expect_global(p, -0.49, "min -(x-0.3)^2");
    }

    // 3. min x1 x2 on [-1,1]^2: -1 at (1,-1) and (-1,1).
    expect_global(make(2, {0, 1, 1, 0}, {0, 0}, {-1, -1}, {1, 1}), -1.0, "min x1 x2");

    // 4. min -x^2 - y^2 s.t. x + y = 1, x, y in [0,1]: -1 at a vertex.
    expect_global(make(2, {-2, 0, 0, -2}, {0, 0}, {0, 0}, {1, 1}, {{1, 1}}, {1}, {1}), -1.0,
                  "min -|x|^2 on a segment");

    // 5. Concave box QP on [0,1]^3:
    //    f = -x1^2-x2^2-x3^2 + x1x2 + x2x3 (Hessian negative definite), so
    //    the minimum is at a vertex: (1,0,1) gives -2, every other vertex
    //    gives more.
    expect_global(make(3, {-2, 1, 0, 1, -2, 1, 0, 1, -2}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}),
                  -2.0, "concave 3-var box QP");

    // 6. Interior coordinate: f = (x1 - 0.3)^2 - x2^2 on [0,1]^2 -> -1 at
    //    (0.3, 1).  The minimiser is NOT a vertex, so branching alone never
    //    lands on it; the bound has to close on an interior point.
    {
        auto p = make(2, {2, 0, 0, -2}, {-0.6, 0}, {0, 0}, {1, 1});
        p.linear.obj_offset = 0.09;
        expect_global(p, -1.0, "interior coordinate");
    }

    // 7. Motzkin-Straus: for a graph with clique number w,
    //    max_{x in simplex} x'Ax = 1 - 1/w.  Triangle {0,1,2} plus a
    //    pendant vertex 3 joined to 0: w = 3, so min -x'Ax = -2/3.
    {
        std::vector<double> Q(16, 0.0);
        const int edges[][2] = {{0, 1}, {0, 2}, {1, 2}, {0, 3}};
        for (const auto& e : edges) {   // 0.5 x'Qx = -x'Ax  ->  Q = -2A
            Q[static_cast<std::size_t>(e[0] * 4 + e[1])] = -2.0;
            Q[static_cast<std::size_t>(e[1] * 4 + e[0])] = -2.0;
        }
        expect_global(make(4, Q, {0, 0, 0, 0}, {0, 0, 0, 0}, {kInf, kInf, kInf, kInf},
                           {{1, 1, 1, 1}}, {1}, {1}),
                      -2.0 / 3.0, "Motzkin-Straus triangle");
    }

    // 8. Random indefinite box QPs, n = 4..6, against the face enumeration.
    for (int trial = 0; trial < 12; ++trial) {
        const int n = 4 + trial % 3;
        std::vector<double> Q(static_cast<std::size_t>(n * n)), c(static_cast<std::size_t>(n)),
            lo(static_cast<std::size_t>(n)), hi(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                const double v = std::round(20.0 * (urand() - 0.5));
                Q[static_cast<std::size_t>(i * n + j)] = v;
                Q[static_cast<std::size_t>(j * n + i)] = v;
            }
            c[static_cast<std::size_t>(i)] = std::round(20.0 * (urand() - 0.5));
            lo[static_cast<std::size_t>(i)] = -1.0 + std::round(4.0 * urand()) * 0.25;
            hi[static_cast<std::size_t>(i)] = lo[static_cast<std::size_t>(i)] + 0.5 + std::round(4.0 * urand()) * 0.5;
        }
        const double fstar = box_qp_brute(n, Q, c, lo, hi);
        expect_global(make(n, Q, c, lo, hi), fstar, "random box QP");
    }

    // 9. No proof without closing the tree: one node on a problem whose root
    //    relaxation is not tight must not report proved.
    {
        auto p = make(3, {-2, 1, 0, 1, -2, 1, 0, 1, -2}, {0.3, -0.2, 0.1}, {-1, -1, -1}, {1, 1, 1});
        auto o = quiet(GlobalRelaxation::McCormick);
        o.max_nodes = 1;
        o.obbt_max_vars = 0;
        o.local_search = false;
        const auto r = solve_global_qp(p, o);
        if (r.proved) CHECK(r.gap_rel <= o.gap_tol);   // only if the root happened to close
        CHECK(r.bound_valid);
    }

    // 10. An unbounded product variable is refused, not guessed.
    {
        const auto r = solve_global_qp(make(1, {-2.0}, {0.0}, {0.0}, {kInf}), quiet());
        CHECK(!r.supported);
        CHECK(!r.proved);
    }

    // 11. Safe LP bound: min x + y s.t. x + y >= 1 on [0,5]^2 has optimum 1.
    //     Any y gives a valid bound; the exact dual (1) gives 1, a perturbed
    //     dual a smaller but still valid one.
    {
        sor::model::LpProblem lp;
        lp.c = {1.0, 1.0};
        lp.col_lo = {0.0, 0.0};
        lp.col_hi = {5.0, 5.0};
        lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        lp.row_lo = {1.0};
        lp.row_hi = {kInf};
        for (double y : {1.0, -1.0, 0.9, 1.3, 0.0, 1e3}) {
            double b = 0.0;
            const bool ok = sor::search::safe_lp_bound(lp, {y}, b);
            CHECK(ok);
            CHECK(b <= 1.0 + 1e-12);
            if (y == 1.0 || y == -1.0) CHECK_NEAR(b, 1.0, 1e-12);
        }
        // Farkas: x + y >= 11 on [0,5]^2 is empty.
        lp.row_lo = {11.0};
        CHECK(sor::search::farkas_proves_empty(lp, {1.0}));
        lp.row_lo = {10.0};   // feasible at (5,5): no ray may claim otherwise
        CHECK(!sor::search::farkas_proves_empty(lp, {1.0}));
        CHECK(!sor::search::farkas_proves_empty(lp, {-1.0}));
    }

    // 12. FBBT: x + y = 1 with x, y >= 0 gives x, y <= 1 (never below).
    {
        sor::model::LpProblem lp;
        lp.c = {0.0, 0.0};
        lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        lp.row_lo = {1.0};
        lp.row_hi = {1.0};
        lp.col_lo = {0.0, 0.0};
        lp.col_hi = {kInf, kInf};
        std::vector<double> lo = lp.col_lo, hi = lp.col_hi;
        CHECK(sor::search::fbbt_linear(lp, lo, hi));
        CHECK(hi[0] >= 1.0 && hi[0] <= 1.0 + 1e-12);
        CHECK(hi[1] >= 1.0 && hi[1] <= 1.0 + 1e-12);
        lo = {0.6, 0.6};
        hi = {1.0, 1.0};
        CHECK(!sor::search::fbbt_linear(lp, lo, hi));   // 0.6 + 0.6 > 1: empty
    }

    // ---- symmetric_eigen: residual, orthonormality, ascending order.
    {
        // Known spectrum: diag(4,1,1) - the 3x3 all-ones-off-diagonal matrix
        // A = [[2,-1,0],[-1,2,-1],[0,-1,2]] has eigenvalues 2-sqrt2, 2, 2+sqrt2.
        const int n = 3;
        const std::vector<double> A = {2, -1, 0, -1, 2, -1, 0, -1, 2};
        std::vector<double> vals, vecs;
        sor::search::symmetric_eigen(n, A, vals, vecs);
        const double r2 = std::sqrt(2.0);
        CHECK_NEAR(vals[0], 2.0 - r2, 1e-12);
        CHECK_NEAR(vals[1], 2.0, 1e-12);
        CHECK_NEAR(vals[2], 2.0 + r2, 1e-12);
        for (int k = 0; k < n; ++k) {
            double nrm = 0.0;
            for (int i = 0; i < n; ++i) {
                const double vi = vecs[static_cast<std::size_t>(i * n + k)];
                nrm += vi * vi;
            }
            CHECK_NEAR(nrm, 1.0, 1e-12);
            for (int i = 0; i < n; ++i) {   // (A v)_i = lambda v_i
                double av = 0.0;
                for (int j = 0; j < n; ++j)
                    av += A[static_cast<std::size_t>(i * n + j)] *
                          vecs[static_cast<std::size_t>(j * n + k)];
                CHECK_NEAR(av, vals[static_cast<std::size_t>(k)] *
                                   vecs[static_cast<std::size_t>(i * n + k)], 1e-11);
            }
            for (int l = k + 1; l < n; ++l) {   // columns are orthogonal
                double dot = 0.0;
                for (int i = 0; i < n; ++i)
                    dot += vecs[static_cast<std::size_t>(i * n + k)] *
                           vecs[static_cast<std::size_t>(i * n + l)];
                CHECK_NEAR(dot, 0.0, 1e-12);
            }
        }
        // A rank-1 moment matrix [1 x'; x xx'] is PSD: no eigenvalue may come
        // out meaningfully negative, or the cut loop would separate a cut
        // against a genuinely feasible point.
        const std::vector<double> xs = {0.3, -0.7};
        std::vector<double> M = {1.0,            xs[0],               xs[1],
                                 xs[0], xs[0] * xs[0],       xs[0] * xs[1],
                                 xs[1], xs[0] * xs[1],       xs[1] * xs[1]};
        sor::search::symmetric_eigen(3, M, vals, vecs);
        CHECK(vals[0] >= -1e-14);
    }

    // ---- PSD cuts: on a nonconvex box QP with a known optimum they must
    // tighten the root LP bound and never pass the optimum.  Bound validity
    // with and without them is checked against the brute-force value.
    {
        const int n = 5;
        std::mt19937 rng(20260922u);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        int improved = 0, tried = 0;
        for (int rep = 0; rep < 6; ++rep) {
            std::vector<double> Q(static_cast<std::size_t>(n * n)), c(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                c[static_cast<std::size_t>(i)] = u(rng);
                for (int j = i; j < n; ++j) {
                    const double q = u(rng);
                    Q[static_cast<std::size_t>(i * n + j)] = q;
                    Q[static_cast<std::size_t>(j * n + i)] = q;
                }
            }
            const std::vector<double> lo(static_cast<std::size_t>(n), 0.0),
                                      hi(static_cast<std::size_t>(n), 1.0);
            const double fstar = box_qp_brute(n, Q, c, lo, hi);
            const QpProblem prob = make(n, Q, c, lo, hi);
            GlobalQpOptions on = quiet(GlobalRelaxation::McCormick);
            GlobalQpOptions off = on;
            off.psd_cuts = false;
            // Root bounds only: no tree, so the comparison is of relaxations.
            on.max_nodes = 1;
            off.max_nodes = 1;
            const GlobalQpResult ron = solve_global_qp(prob, on);
            const GlobalQpResult roff = solve_global_qp(prob, off);
            const double tolf = 1e-7 * (1.0 + std::fabs(fstar));
            CHECK(roff.root_bound_lp <= fstar + tolf);
            CHECK(ron.root_bound_lp <= fstar + tolf);
            if (ron.psd_cuts > 0) {
                ++tried;
                CHECK(ron.root_bound_psd <= fstar + tolf);
                CHECK(ron.root_bound_psd >= ron.root_bound_lp - tolf);
                if (ron.root_bound_psd > ron.root_bound_lp + 1e-6 * (1.0 + std::fabs(fstar)))
                    ++improved;
            }
        }
        // The cuts are not guaranteed to bite on every instance, but on a
        // batch of dense indefinite box QPs at least one must: if none does,
        // the separator is not finding the negative eigenvalues.
        CHECK(tried > 0);
        CHECK(improved > 0);
    }

    return sor::test::finish("test_global_qp");
}
