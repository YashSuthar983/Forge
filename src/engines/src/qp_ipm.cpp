// FORGE — primal-dual interior-point method for convex QP.
//
// LAYER L3.  The exact path for convex QP: Newton on the perturbed KKT
// conditions converges in tens of iterations almost regardless of scaling,
// where a first-order method on QPLIB's badly scaled instances did not
// converge at all.  Method: Mehrotra's predictor-corrector (Mehrotra 1992)
// for the QP form in Nocedal & Wright ch. 16 / Vanderbei ch. 18; implemented
// from those texts, no solver source consulted.
//
// FORM.  min 1/2 x'Qx + c'x  s.t.  A x - w = 0,  l <= x <= u,  rl <= w <= ru.
// Row activities w are variables, so ranged, one-sided and equality rows
// are all "a bounded variable".  Every finite bound gets a slack and a
// multiplier: s_l = z - l, s_u = u - z, lam, nu >= 0.  Lagrangian sign:
//   Q x + c - A'y - lam_x + nu_x = 0,     y - lam_w + nu_w = 0.
// Newton, eliminating the bound multipliers and then dw, gives the reduced
// system, QUASI-DEFINITE in any order (Vanderbei 1995):
//   [ -(Q + Tx + rho I)   A'        ] [dx]   [ r_dx + h_x                  ]
//   [   A                 Tw^-1 + d ] [dy] = [ -r_p - Tw^-1 (r_dw + h_w)   ]
// with T = lam/s_l + nu/s_u per component and h the complementarity terms.
// Equality rows have w fixed (Tw^-1 = 0, regularised by d); free columns have
// T = 0 (regularised by rho).  rho and d act as proximal terms (Friedlander &
// Orban 2012), so the regularised system is the one solved.
//
// LP (Q = 0, no quadratic rows).  Eliminating dx from the augmented system
// leaves the normal equation of order m,
//   M = A D A^T + G,   D_j = 1 / (T_j + rho),
//   G_i = delta on equalities and 1/T_i + delta on inequalities.
// The sparsity pattern of the sparse-column product is analysed once; each
// Newton iteration only reassembles its values and refactorises.  The (up to
// 256) longest columns, those with at least 64 nonzeros, are a low-rank
// Sherman-Morrison-Woodbury update on top of the factor of the rest (fit2p).
// The augmented LDL' stays as the fallback -- chosen up front when M is the
// costlier factor, taken over mid-run if the normal equations break down --
// and is the only path for Q != 0.
//
// CENTRALITY.  A pair (s, lam) can drift to s ~ 0 with lam ~ 1/s while its
// product stays near mu: "converged" complementarity, hopeless conditioning.
// Its T = lam/s then reaches 1e17, which FREEZES that variable -- Newton can
// no longer move it, and for a row slack the primal residual is left for the
// dual regularisation to absorb as a huge dy (measured on QPLIB_8845: row
// 511 stuck at rp = 1.2 with T = 1e17 and dy = 1e8, for 100 iterations).
// Gondzio's multiple centrality correctors (Gondzio 1996) are the fix: after
// the Mehrotra corrector, retarget the products that sit outside
// [beta_min*mu, beta_max*mu] and re-solve with the SAME factorization,
// keeping the new direction only if it buys a longer step.
//
// PRESOLVE, the minimum the form needs: fixed columns are substituted out
// (branch-and-bound fixes variables, so this is common) and free rows are
// dropped.  Their duals are reconstructed afterwards.
//
// THE CLAIM.  The IPM's own residuals are in scaled, presolved units.  The
// solution is mapped back and re-checked by qpc::kkt_original on the
// ORIGINAL data -- the same independent check PDHCG-II's claim rests on --
// and only that check can return Optimal.
#include "forge/engines/qp.hpp"
#include "forge/core/env_switches.hpp"
#include "forge/engines/qcqp.hpp"
#include "forge/la/ldlt.hpp"
#include "forge/sparse/csr.hpp"

#include "forge/core/parallel.hpp"
#include "ipm_core.hpp"
#include "qp_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace forge::engines {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
using qpc::sz;
constexpr f64 kInf = std::numeric_limits<f64>::infinity();

using ipmc::Core;

// Full symmetric Q x from the upper-triangle column lists.
void q_mul(const Core& k, const std::vector<f64>& x, std::vector<f64>& out) {
    out.assign(sz(k.n), 0.0);
    for (core::Index j = 0; j < k.n; ++j)
        for (const auto& [i, v] : k.qcol[sz(j)]) {
            out[sz(i)] += v * x[sz(j)];
            if (i != j) out[sz(j)] += v * x[sz(i)];
        }
}

struct Result {
    std::vector<f64> x, y;   // core space; y in the Lagrangian sign above
    int iterations = 0;
    bool converged = false;
    f64 worst_solve_residual = 0.0;
    std::string reason;
};

// Normal equations for the LP barrier (Q = 0).  Eliminating dx from the
// regularised augmented system leaves the order-m SPD matrix
//   M = A D A^T + G,  D_j = 1 / (T_j + rho),  G_i = delta (+ 1/T_i),
// built here from the SPARSE columns only; the few dense columns are a
// low-rank update applied with the Sherman-Morrison-Woodbury identity on top
// of one Cholesky-type (LDL', positive pivots) factorisation of the sparse
// part.  The sparsity pattern is analysed once; each Newton iteration only
// reassembles M's values (column by column, in parallel) and refactorises.
struct NormalEq {
    struct Col {
        std::vector<core::Index> rows;   // ascending, duplicates merged
        std::vector<f64> vals;
    };

    core::Index n = 0, m = 0;
    std::vector<Col> cols;                   // A by column
    std::vector<core::Offset> rrp;           // A by row (same merged entries)
    std::vector<core::Index> rcj;
    std::vector<f64> rcv;
    std::vector<core::Index> dense_of;       // columns taken out into the Schur update
    std::vector<std::uint8_t> is_dense;
    int dense_candidates = 0;                // columns long enough to be considered
    bool schur = false;
    bool ok = true;                          // false: M too large, stay augmented

    la::SymCsc M;
    std::vector<core::Offset> diag_slot;
    std::vector<std::int8_t> sign;
    la::Ldlt fac;
    int reg_pivots = 0;

    std::vector<f64> dscale;                 // 1/(T_j + rho), all columns
    std::vector<f64> sroot;                  // sqrt(dscale) of the dense columns
    std::vector<f64> Y;                      // m * p, column-major, M^{-1} A_dense
    std::vector<f64> schur_l;                // p * p row-major lower Cholesky of I + D^1/2 A'M^-1A D^1/2
    std::vector<f64> bwork, zwork, wwork, colwork;
    std::vector<std::vector<f64>> wscratch;  // per-worker dense accumulator (assemble)

    void build(const ipmc::Core& k) {
        n = k.n;
        m = k.m;
        cols.assign(sz(n), {});
        dscale.assign(sz(n), 0.0);
        const auto& rp = k.A.pattern.row_ptr();
        const auto& ci = k.A.pattern.col_idx();
        for (core::Index i = 0; i < m; ++i)
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
                cols[sz(ci[sz(t)])].rows.push_back(i), cols[sz(ci[sz(t)])].vals.push_back(k.A.vals[sz(t)]);
        for (core::Index j = 0; j < n; ++j) {
            auto& c = cols[sz(j)];
            const std::size_t knt = c.rows.size();
            std::vector<std::size_t> ord(knt);
            for (std::size_t e = 0; e < knt; ++e) ord[e] = e;
            std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) {
                return c.rows[a] < c.rows[b];
            });
            Col merged;
            for (std::size_t t = 0; t < knt; ++t) {
                const core::Index r = c.rows[ord[t]];
                const f64 v = c.vals[ord[t]];
                if (!merged.rows.empty() && merged.rows.back() == r) merged.vals.back() += v;
                else { merged.rows.push_back(r); merged.vals.push_back(v); }
            }
            c = std::move(merged);
        }
        // The same merged entries by row (columns ascending within a row).
        rrp.assign(sz(m) + 1, 0);
        for (core::Index j = 0; j < n; ++j)
            for (core::Index r : cols[sz(j)].rows) ++rrp[sz(r) + 1];
        for (core::Index i = 0; i < m; ++i) rrp[sz(i) + 1] += rrp[sz(i)];
        rcj.assign(sz(rrp[sz(m)]), 0);
        rcv.assign(sz(rrp[sz(m)]), 0.0);
        {
            std::vector<core::Offset> cur(rrp.begin(), rrp.end() - 1);
            for (core::Index j = 0; j < n; ++j) {
                const auto& c = cols[sz(j)];
                for (std::size_t e = 0; e < c.rows.size(); ++e) {
                    const auto at = static_cast<std::size_t>(cur[sz(c.rows[e])]++);
                    rcj[at] = j;
                    rcv[at] = c.vals[e];
                }
            }
        }

        // A column this long fills a dense patch of ADA^T.  The longest of
        // them may be cheaper as a Schur update than inside the sparse
        // factor, but each costs one solve with M per factorisation and m
        // doubles of storage, so whether and how many is decided by trying:
        // the pattern is analysed for a few split sizes and the cheapest per
        // Newton iteration wins (pilot87's 178 columns of 64+ nonzeros: not
        // split, 3x faster; fit2p's few thousand-nonzero columns: split).
        constexpr core::Index kDenseNnz = 64;
        constexpr std::size_t kMaxSchur = 256;
        constexpr std::size_t kMaxSchurStore = 30000000;   // m * p doubles
        std::vector<core::Index> cand;
        for (core::Index j = 0; j < n; ++j)
            if (static_cast<core::Index>(cols[sz(j)].rows.size()) >= kDenseNnz) cand.push_back(j);
        dense_candidates = static_cast<int>(cand.size());
        std::stable_sort(cand.begin(), cand.end(), [&](core::Index a, core::Index b) {
            return cols[sz(a)].rows.size() > cols[sz(b)].rows.size();
        });
        // Columns over a quarter of the rows would make M dense: always split
        // (analysing that M only to reject it costs more than the solve).
        std::vector<std::size_t> trials;
        if (m > 0) {
            const std::size_t cap = std::min({cand.size(), kMaxSchur,
                                              static_cast<std::size_t>(std::max<core::Index>(m / 2, 1)),
                                              std::max<std::size_t>(kMaxSchurStore / sz(m), 1)});
            std::size_t p0 = 0;
            while (p0 < cap && static_cast<core::Index>(cols[sz(cand[p0])].rows.size()) >= m / 4) ++p0;
            trials.push_back(p0);
            for (std::size_t t = std::max<std::size_t>(p0, 1) * 4; t < cap; t *= 4) trials.push_back(t);
            if (cap > p0) trials.push_back(cap);
        }

        is_dense.assign(sz(n), 0);
        if (m == 0) return;
        f64 best_cost = kInf;
        ok = false;
        for (const std::size_t p : trials) {
            std::vector<std::uint8_t> dense(sz(n), 0);
            for (std::size_t t = 0; t < p; ++t) dense[sz(cand[t])] = 1;
            la::SymCsc Mt;
            std::vector<core::Offset> dslot;
            if (!pattern(dense, Mt, dslot)) continue;
            la::Ldlt ft;
            ft.analyze(Mt);
            // Factor multiply-adds, plus per split column one solve with M
            // (two passes over L, memory-bound: weighted x4) and the dense
            // work of applying the update ~100 times per iteration.
            const f64 cost = ft.flops() + static_cast<f64>(p) * (8.0 * static_cast<f64>(ft.nnz_l()) +
                                                              100.0 * static_cast<f64>(m));
            if (cost < best_cost) {
                best_cost = cost;
                ok = true;
                is_dense = std::move(dense);
                M = std::move(Mt);
                diag_slot = std::move(dslot);
                fac = std::move(ft);
                dense_of.assign(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(p));
                std::sort(dense_of.begin(), dense_of.end());
                schur = p > 0;
            }
        }
        if (!ok) return;
        M.vals.assign(M.row_idx.size(), 0.0);
        sign.assign(sz(m), 1);
    }

    // Pattern of the sparse part for a given dense set: column q of the upper
    // triangle holds every r <= q sharing a sparse column with row q, plus
    // the diagonal.  False when it would be too large to hold.
    bool pattern(const std::vector<std::uint8_t>& dense, la::SymCsc& Mo,
                 std::vector<core::Offset>& dslot) const {
        constexpr std::size_t kMaxNnzM = 150000000;
        Mo.n = m;
        Mo.col_ptr.assign(sz(m) + 1, 0);
        dslot.assign(sz(m), 0);
        std::vector<core::Index> mark(sz(m), -1), tmp;
        for (core::Index q = 0; q < m; ++q) {
            tmp.clear();
            for (auto t = rrp[sz(q)]; t < rrp[sz(q) + 1]; ++t) {
                const core::Index j = rcj[sz(t)];
                if (dense[sz(j)]) continue;
                for (core::Index r : cols[sz(j)].rows) {
                    if (r > q) break;
                    if (mark[sz(r)] != q) { mark[sz(r)] = q; tmp.push_back(r); }
                }
            }
            if (mark[sz(q)] != q) tmp.push_back(q);
            std::sort(tmp.begin(), tmp.end());
            for (core::Index r : tmp) Mo.row_idx.push_back(r);
            Mo.col_ptr[sz(q) + 1] = static_cast<core::Offset>(Mo.row_idx.size());
            dslot[sz(q)] = Mo.col_ptr[sz(q) + 1] - 1;   // row == q is last
            if (Mo.row_idx.size() > kMaxNnzM) return false;
        }
        return true;
    }

    // T is length n+m (x then w).  rho, dreg are the proximal terms already
    // scaled by any regularisation boost.
    void assemble(const std::vector<f64>& tdiag, f64 rho, f64 dreg,
                  const std::vector<std::uint8_t>& fixed_w) {
        for (core::Index j = 0; j < n; ++j) dscale[sz(j)] = 1.0 / (tdiag[sz(j)] + rho);
        if (m == 0) return;
        // Column q of M reads only A and D, so columns are independent tasks;
        // every entry is summed in the same order whatever the schedule.
        auto column = [&](core::Index q, std::vector<f64>& w) {
            for (auto t = rrp[sz(q)]; t < rrp[sz(q) + 1]; ++t) {
                const core::Index j = rcj[sz(t)];
                if (is_dense[sz(j)]) continue;
                const f64 coef = rcv[sz(t)] * dscale[sz(j)];
                const auto& c = cols[sz(j)];
                for (std::size_t e = 0; e < c.rows.size(); ++e) {
                    if (c.rows[e] > q) break;
                    w[sz(c.rows[e])] += coef * c.vals[e];
                }
            }
            for (auto p = M.col_ptr[sz(q)]; p < M.col_ptr[sz(q) + 1]; ++p) {
                const auto r = sz(M.row_idx[sz(p)]);
                M.vals[sz(p)] = w[r];
                w[r] = 0.0;
            }
            f64 g = dreg;
            const f64 tw = tdiag[sz(n + q)];
            if (!fixed_w[sz(q)] && tw > 0.0) g += 1.0 / tw;
            M.vals[sz(diag_slot[sz(q)])] += g;
        };
        core::ThreadPool& pool = core::global_pool();
        constexpr core::Offset kChunk = 64;
        const core::Offset nchunks = (m + kChunk - 1) / kChunk;
        const int nw = std::max(1, pool.size());
        if (wscratch.size() != sz(nw)) wscratch.assign(sz(nw), {});
        for (auto& w : wscratch) if (w.size() != sz(m)) w.assign(sz(m), 0.0);
        auto body = [&](core::Offset chunk, int worker) {
            auto& w = wscratch[sz(worker)];
            const core::Index lo = static_cast<core::Index>(chunk * kChunk);
            const core::Index hi = static_cast<core::Index>(std::min<core::Offset>(m, (chunk + 1) * kChunk));
            for (core::Index q = lo; q < hi; ++q) column(q, w);
        };
        if (nw == 1 || nchunks <= 1 || m < 512) {
            for (core::Offset c = 0; c < nchunks; ++c) body(c, 0);
        } else {
            pool.parallel_for(nchunks, body);
        }
    }

    bool factorize() {
        reg_pivots = 0;
        if (m == 0) return true;
        if (!fac.factorize(M, sign, 1e-12)) return false;
        reg_pivots = static_cast<int>(fac.regularized_pivots());
        if (schur && !prepare_schur()) return false;
        return true;
    }

    // K_reg [dx; dy] = rhs, in place.  rhs is length n+m.
    void apply(std::vector<f64>& rhs) {
        if (m == 0) {
            for (core::Index j = 0; j < n; ++j) rhs[sz(j)] = -dscale[sz(j)] * rhs[sz(j)];
            return;
        }
        bwork.assign(sz(m), 0.0);
        for (core::Index i = 0; i < m; ++i) bwork[sz(i)] = rhs[sz(n + i)];
        for (core::Index j = 0; j < n; ++j) {
            const f64 t = dscale[sz(j)] * rhs[sz(j)];
            const auto& c = cols[sz(j)];
            for (std::size_t e = 0; e < c.rows.size(); ++e)
                bwork[sz(c.rows[e])] += c.vals[e] * t;
        }
        solve_m(bwork);
        for (core::Index j = 0; j < n; ++j) {
            f64 atd = 0.0;
            const auto& c = cols[sz(j)];
            for (std::size_t e = 0; e < c.rows.size(); ++e) atd += c.vals[e] * bwork[sz(c.rows[e])];
            rhs[sz(j)] = dscale[sz(j)] * (atd - rhs[sz(j)]);
        }
        for (core::Index i = 0; i < m; ++i) rhs[sz(n + i)] = bwork[sz(i)];
    }

private:
    static core::Index static_core_index(std::size_t n) { return static_cast<core::Index>(n); }

    // Full matrix S = M + A_d D_d A_d^T.  With s = sqrt(D_d) and Y = M^-1 A_d,
    //   S^-1 = M^-1 - Y s C^-1 s Y^T,   C = I + s A_d^T M^-1 A_d s,
    // C symmetric with eigenvalues >= 1, so its Cholesky needs no pivoting
    // however large D_d is (a free column carries D = 1/rho).
    bool prepare_schur() {
        const core::Index p = static_core_index(dense_of.size());
        Y.assign(sz(m) * sz(p), 0.0);
        schur_l.assign(sz(p) * sz(p), 0.0);
        sroot.assign(sz(p), 0.0);
        for (core::Index k = 0; k < p; ++k) {
            sroot[sz(k)] = std::sqrt(dscale[sz(dense_of[sz(k)])]);
            colwork.assign(sz(m), 0.0);
            const auto& c = cols[sz(dense_of[sz(k)])];
            for (std::size_t e = 0; e < c.rows.size(); ++e) colwork[sz(c.rows[e])] = c.vals[e];
            fac.solve(colwork);
            for (core::Index r = 0; r < m; ++r) Y[sz(r) + sz(k) * sz(m)] = colwork[sz(r)];
        }
        for (core::Index i = 0; i < p; ++i) {
            const auto& ci = cols[sz(dense_of[sz(i)])];
            for (core::Index k = 0; k < p; ++k) {
                f64 s = 0.0;
                const f64* yk = Y.data() + sz(k) * sz(m);
                for (std::size_t e = 0; e < ci.rows.size(); ++e)
                    s += ci.vals[e] * yk[sz(ci.rows[e])];
                schur_l[sz(i) * sz(p) + sz(k)] = s * sroot[sz(i)] * sroot[sz(k)];
            }
        }
        for (core::Index i = 0; i < p; ++i)
            for (core::Index k = i + 1; k < p; ++k) {
                const f64 s = 0.5 * (schur_l[sz(i) * sz(p) + sz(k)] + schur_l[sz(k) * sz(p) + sz(i)]);
                schur_l[sz(i) * sz(p) + sz(k)] = schur_l[sz(k) * sz(p) + sz(i)] = s;
            }
        for (core::Index i = 0; i < p; ++i) schur_l[sz(i) * sz(p) + sz(i)] += 1.0;
        for (core::Index j = 0; j < p; ++j) {
            f64 s = schur_l[sz(j) * sz(p) + sz(j)];
            for (core::Index k = 0; k < j; ++k) {
                const f64 l = schur_l[sz(j) * sz(p) + sz(k)];
                s -= l * l;
            }
            if (!std::isfinite(s)) return false;
            if (!(s > 1e-12)) { s = 1e-12; ++reg_pivots; }
            const f64 ljj = std::sqrt(s);
            schur_l[sz(j) * sz(p) + sz(j)] = ljj;
            for (core::Index i = j + 1; i < p; ++i) {
                f64 t = schur_l[sz(i) * sz(p) + sz(j)];
                for (core::Index k = 0; k < j; ++k)
                    t -= schur_l[sz(i) * sz(p) + sz(k)] * schur_l[sz(j) * sz(p) + sz(k)];
                schur_l[sz(i) * sz(p) + sz(j)] = t / ljj;
            }
        }
        return true;
    }

    void solve_schur(std::vector<f64>& w) const {
        const core::Index p = static_core_index(dense_of.size());
        for (core::Index j = 0; j < p; ++j) {
            f64 s = w[sz(j)];
            for (core::Index k = 0; k < j; ++k) s -= schur_l[sz(j) * sz(p) + sz(k)] * w[sz(k)];
            w[sz(j)] = s / schur_l[sz(j) * sz(p) + sz(j)];
        }
        for (core::Index j = p; j-- > 0; ) {
            f64 s = w[sz(j)];
            for (core::Index k = j + 1; k < p; ++k) s -= schur_l[sz(k) * sz(p) + sz(j)] * w[sz(k)];
            w[sz(j)] = s / schur_l[sz(j) * sz(p) + sz(j)];
        }
    }

    // Overwrites b with S^{-1} b, S = M_sparse + A_d D_d A_d^T.
    void solve_m(std::vector<f64>& b) {
        if (!schur) { fac.solve(b); return; }
        const core::Index p = static_core_index(dense_of.size());
        zwork = b;
        fac.solve(zwork);
        wwork.assign(sz(p), 0.0);
        for (core::Index k = 0; k < p; ++k) {
            const auto& c = cols[sz(dense_of[sz(k)])];
            f64 s = 0.0;
            for (std::size_t e = 0; e < c.rows.size(); ++e) s += c.vals[e] * zwork[sz(c.rows[e])];
            wwork[sz(k)] = s * sroot[sz(k)];
        }
        solve_schur(wwork);
        b = zwork;
        for (core::Index k = 0; k < p; ++k) {
            const f64 wk = wwork[sz(k)] * sroot[sz(k)];
            const f64* yk = Y.data() + sz(k) * sz(m);
            for (core::Index r = 0; r < m; ++r) b[sz(r)] -= yk[sz(r)] * wk;
        }
    }
};

// ---- the iteration --------------------------------------------------------
// orig_rel (optional, LP only): the ORIGINAL model's relative primal and dual
// residuals at the point `accept` was last called on.
Result run_ipm(const Core& k, const QpOptions& opts, Clock::time_point t0,
               const std::function<bool(const std::vector<f64>&, const std::vector<f64>&)>& accept,
               const std::function<void(f64&, f64&)>& orig_rel = nullptr) {
    const core::Index n = k.n, m = k.m, N = n + m;
    Result res;
    bool has_q = false;
    int stall_count = 0;
    f64 stall_rp = kInf, stall_rd = kInf, mu_start = -1.0;
    // LP: the iterate with the smallest original-model relative merit, which is
    // what comes back when the run ends stalled or out of budget.
    f64 best_merit = kInf;
    std::vector<f64> best_z, best_y;
    for (const auto& col : k.qcol)
        for (const auto& e : col) has_q = has_q || e.second != 0.0;
    // A quadratic row makes the stationarity depend on x through J(x)'y and
    // the Hessian on y, so the equal-step argument below applies to it too.
    has_q = has_q || !k.quad.empty();
    // Normal equations are the LP path.  A quadratic objective or a quadratic
    // row keeps the quasi-definite augmented factor.
    const bool lp = !has_q;
    bool use_ne = lp;
    NormalEq ne;
    if (lp) {
        ne.build(k);
        use_ne = ne.ok;
    }
    // Primal-dual regularisation as a PROXIMAL term (Friedlander & Orban,
    // Math. Prog. Comp. 4, 2012): the regularised system is the exact Newton
    // system of a proximal-point subproblem, so it is solved as is -- the
    // refinement below only removes round-off.  An earlier version refined
    // against the UNregularised matrix; on near-flat directions (QPLIB_9002:
    // Q entries down to 1e-11, no linear term) that re-solved a nearly
    // singular system and took a first step of ~1e9.
    // Fixed proximal regularisation.  Letting it follow mu down (rho = mu/100)
    // was tried, on the theory that a fixed 1e-8 term floors the residuals it
    // perturbs; measured, it was a net loss -- QPLIB_8906 went from matching
    // its published objective to 100x out -- so it stays fixed.
    // rho (primal) and the row term d (dual) share one value.  An LP tapers it
    // with mu (lp_prox below); prox_cur is the value in force.
    const f64 rho0 = 1e-8;
    f64 prox_cur = rho0;
    f64 reg_boost = 1.0;
    constexpr f64 kUlp = 2.220446049250313e-16;

    // Bound structure: component c in [0, n+m): x then w.
    auto bound_lo = [&](core::Index c) {
        return c < n ? k.l[sz(c)] : k.rl[sz(c - n)];
    };
    auto bound_hi = [&](core::Index c) {
        return c < n ? k.u[sz(c)] : k.ru[sz(c - n)];
    };
    std::vector<std::uint8_t> has_l(sz(N)), has_u(sz(N)), fixed_w(sz(m));
    for (core::Index c = 0; c < N; ++c) {
        has_l[sz(c)] = std::isfinite(bound_lo(c));
        has_u[sz(c)] = std::isfinite(bound_hi(c));
    }
    for (core::Index i = 0; i < m; ++i) {
        fixed_w[sz(i)] = k.rl[sz(i)] == k.ru[sz(i)];
        if (fixed_w[sz(i)]) { has_l[sz(n + i)] = 0; has_u[sz(n + i)] = 0; }
    }

    std::vector<f64> z(sz(N), 0.0), lam(sz(N), 0.0), nu(sz(N), 0.0), y(sz(m), 0.0);

    // ---- quadratic rows: supports and scratch ------------------------------
    // Per quadratic row t: its support (sorted columns), each Hessian entry's
    // local (row, col) in that support, Q_t x on the support (qix), and
    // 1/2 x'Q_t x (qval) at the current x.  eta_t is the row's weight in
    // the Lagrangian Hessian (sign: see quad_eval).
    const std::size_t nq = k.quad.size();
    std::vector<std::vector<core::Index>> qsup(nq), qlr(nq), qlc(nq);
    std::vector<std::vector<f64>> qix(nq);
    std::vector<f64> qval(nq, 0.0), eta(nq, 0.0);
    for (std::size_t t = 0; t < nq; ++t) {
        const auto& q = k.quad[t];
        auto& S = qsup[t];
        for (std::size_t e = 0; e < q.v.size(); ++e) { S.push_back(q.r[e]); S.push_back(q.c[e]); }
        std::sort(S.begin(), S.end());
        S.erase(std::unique(S.begin(), S.end()), S.end());
        for (std::size_t e = 0; e < q.v.size(); ++e) {
            qlr[t].push_back(static_cast<core::Index>(std::lower_bound(S.begin(), S.end(), q.r[e]) - S.begin()));
            qlc[t].push_back(static_cast<core::Index>(std::lower_bound(S.begin(), S.end(), q.c[e]) - S.begin()));
        }
        qix[t].assign(S.size(), 0.0);
    }
    // Adds each quadratic row's part to a row activity and a J'y at x: the
    // row value 1/2 x'Q_t x to ax, (Q_t x) y_row to aty -- the IPM's
    // Lagrangian sign (stationarity grad f - J'y - lam + nu = 0) -- and
    // refreshes qix/qval/eta for the next KKT fill.
    //
    // The Hessian weight: the Lagrangian Hessian is Q0 - sum_t y_t Q_t.  A
    // canonical row is upper-bounded only, so at dual feasibility
    // y_t = -nu_t <= 0 and -y_t Q_t is PSD.  In the infeasible iteration y_t
    // can transiently carry the wrong sign; the weight is then nu_t instead
    // (see below), which keeps the (1,1) block negative definite on the
    // row's support -- the system QUASI-DEFINITE, which the static-pivot
    // LDL' relies on (Vanderbei 1995) -- and the weight equals the exact
    // Hessian's whenever the sign is right, which it is near the solution.
    auto quad_eval = [&](const std::vector<f64>& xv, const std::vector<f64>& yv,
                         std::vector<f64>& ax_out, std::vector<f64>* aty_out) {
        for (std::size_t t = 0; t < nq; ++t) {
            const auto& q = k.quad[t];
            auto& w = qix[t];
            std::fill(w.begin(), w.end(), 0.0);
            f64 val = 0.0;
            for (std::size_t e = 0; e < q.v.size(); ++e) {
                const f64 xr = xv[sz(q.r[e])], xc = xv[sz(q.c[e])];
                if (q.r[e] == q.c[e]) {
                    w[sz(qlr[t][e])] += q.v[e] * xr;
                    val += 0.5 * q.v[e] * xr * xr;
                } else {
                    w[sz(qlr[t][e])] += q.v[e] * xc;
                    w[sz(qlc[t][e])] += q.v[e] * xr;
                    val += q.v[e] * xr * xc;
                }
            }
            qval[t] = val;
            ax_out[sz(q.row)] += val;
            const f64 yr = yv[sz(q.row)];
            // -y when it has the convex sign, else the slack's own
            // multiplier nu (> 0 in the interior; = -y at dual feasibility),
            // so the row never loses its curvature to a transient sign.
            eta[t] = -yr > 0.0 ? -yr : nu[sz(n + q.row)];
            if (aty_out)
                for (std::size_t s = 0; s < qsup[t].size(); ++s)
                    (*aty_out)[sz(qsup[t][s])] += w[s] * yr;
        }
    };

    // ---- KKT pattern (upper triangle): x block (Q upper + diag), then A' ----
    // With quadratic rows the x block also holds every Q_t's upper triangle
    // (the Lagrangian Hessian) and column n+i holds row i's Jacobian support
    // supp(a_i) u supp(Q_i).  The extra entries are APPENDED after the QP
    // pattern of each column, so a QP's pattern and values are exactly what
    // they were before quadratic rows existed.
    la::SymCsc K;
    std::vector<core::Offset> diag_slot(sz(N));
    std::vector<std::vector<core::Offset>> qpos(nq), jpos(nq);   // K slots: Q_t entries / J_t support
    std::vector<std::int8_t> sign;
    la::Ldlt fac;
    bool aug_built = false;
    auto build_aug = [&]() {
    if (aug_built) return;
    aug_built = true;
    K.n = N;
    K.col_ptr.assign(sz(N) + 1, 0);
    {
        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> quad_by_col(sz(n));
        for (std::size_t t = 0; t < nq; ++t) {
            qpos[t].assign(k.quad[t].v.size(), 0);
            for (std::size_t e = 0; e < k.quad[t].v.size(); ++e)
                quad_by_col[sz(k.quad[t].c[e])].push_back({t, e});
        }
        std::vector<core::Offset> slot_of(sz(N), -1);   // row -> slot, current column only
        // Column j (x): Q upper entries (i <= j) + quad extras + diagonal.
        for (core::Index j = 0; j < n; ++j) {
            bool diag = false;
            const auto start = static_cast<core::Offset>(K.row_idx.size());
            for (const auto& [i, v] : k.qcol[sz(j)]) {
                (void)v;
                K.row_idx.push_back(i);
                slot_of[sz(i)] = static_cast<core::Offset>(K.row_idx.size()) - 1;
                if (i == j) { diag = true; diag_slot[sz(j)] = static_cast<core::Offset>(K.row_idx.size()) - 1; }
            }
            if (!quad_by_col[sz(j)].empty()) {
                std::vector<core::Index> extra;
                for (const auto& [t, e] : quad_by_col[sz(j)])
                    if (slot_of[sz(k.quad[t].r[e])] < 0) extra.push_back(k.quad[t].r[e]);
                std::sort(extra.begin(), extra.end());
                extra.erase(std::unique(extra.begin(), extra.end()), extra.end());
                for (core::Index i : extra) {
                    K.row_idx.push_back(i);
                    slot_of[sz(i)] = static_cast<core::Offset>(K.row_idx.size()) - 1;
                    if (i == j) { diag = true; diag_slot[sz(j)] = static_cast<core::Offset>(K.row_idx.size()) - 1; }
                }
                for (const auto& [t, e] : quad_by_col[sz(j)]) qpos[t][e] = slot_of[sz(k.quad[t].r[e])];
            }
            if (!diag) { K.row_idx.push_back(j); diag_slot[sz(j)] = static_cast<core::Offset>(K.row_idx.size()) - 1; }
            for (auto t = start; t < static_cast<core::Offset>(K.row_idx.size()); ++t) slot_of[sz(K.row_idx[sz(t)])] = -1;
            K.col_ptr[sz(j) + 1] = static_cast<core::Offset>(K.row_idx.size());
        }
        // Column n+i (y): A row i entries (x indices) + quad support extras + diagonal.
        std::vector<std::size_t> quad_of_row(sz(m), nq);
        for (std::size_t t = 0; t < nq; ++t) quad_of_row[sz(k.quad[t].row)] = t;
        const auto& rp = k.A.pattern.row_ptr();
        const auto& ci = k.A.pattern.col_idx();
        for (core::Index i = 0; i < m; ++i) {
            const auto start = static_cast<core::Offset>(K.row_idx.size());
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
                K.row_idx.push_back(ci[sz(t)]);
                slot_of[sz(ci[sz(t)])] = static_cast<core::Offset>(K.row_idx.size()) - 1;
            }
            if (const std::size_t t = quad_of_row[sz(i)]; t < nq) {
                for (core::Index j : qsup[t])
                    if (slot_of[sz(j)] < 0) {
                        K.row_idx.push_back(j);
                        slot_of[sz(j)] = static_cast<core::Offset>(K.row_idx.size()) - 1;
                    }
                for (core::Index j : qsup[t]) jpos[t].push_back(slot_of[sz(j)]);
            }
            for (auto t = start; t < static_cast<core::Offset>(K.row_idx.size()); ++t) slot_of[sz(K.row_idx[sz(t)])] = -1;
            K.row_idx.push_back(n + i);
            diag_slot[sz(n + i)] = static_cast<core::Offset>(K.row_idx.size()) - 1;
            K.col_ptr[sz(n + i) + 1] = static_cast<core::Offset>(K.row_idx.size());
        }
        K.vals.assign(K.row_idx.size(), 0.0);
    }
    sign.assign(sz(N), 1);
    for (core::Index j = 0; j < n; ++j) sign[sz(j)] = -1;
    fac.analyze(K);
    if (opts.verbose)
        std::fprintf(stderr, "ipm kkt n=%lld nnzL=%lld supernodes=%lld flops=%.3e\n",
                     static_cast<long long>(K.n), static_cast<long long>(fac.nnz_l()),
                     static_cast<long long>(fac.supernodes()), fac.flops());
    };
    if (!use_ne) {
        build_aug();
    } else if (ne.fac.flops() > 2e9) {
        // M's factor is big enough for the choice to matter: keep the
        // augmented system when it is the cheaper one (dense ADA^T).
        const f64 ne_cost = ne.fac.flops() + 4.0 * static_cast<f64>(ne.dense_of.size()) *
                                                 static_cast<f64>(ne.fac.nnz_l());
        build_aug();
        use_ne = ne_cost <= 2.0 * fac.flops();
    }
    if (use_ne && opts.verbose) {
        std::fprintf(stderr,
                     "ipm normal-eq m=%lld nnzM=%lld nnzL=%lld dense_cols=%d schur=%d flops=%.3e\n",
                     static_cast<long long>(ne.m),
                     static_cast<long long>(ne.m == 0 ? 0 : ne.M.row_idx.size()),
                     static_cast<long long>(ne.m == 0 ? 0 : ne.fac.nnz_l()),
                     ne.dense_candidates, static_cast<int>(ne.dense_of.size()),
                     ne.m == 0 ? 0.0 : ne.fac.flops());
    }

    // Fill K's values for the current T.  Two diagonals are kept: the
    // REGULARISED one that is factorized, and the UNREGULARISED one that the
    // refinement targets -- the factor is then a preconditioner for the true
    // Newton system, which is how a regularised IPM recovers the accuracy the
    // proximal term costs it (Saunders; Friedlander & Orban 2012).
    std::vector<f64> tdiag(sz(N)), kdiag_reg(sz(N)), kdiag_true(sz(N));
    auto fill_K = [&]() {
        core::Offset pos = 0;
        // Q = 0 (LP barrier): qcol is empty; skip Hessian value writes and
        // only clear the diagonal / structural slots the pattern reserved.
        for (core::Index j = 0; j < n; ++j) {
            if (has_q) {
                for (const auto& [i, v] : k.qcol[sz(j)])
                    K.vals[sz(pos++)] = -v;
            }
            // explicit diagonal slot, and any quad-only Hessian slots
            while (pos < K.col_ptr[sz(j) + 1]) K.vals[sz(pos++)] = 0.0;
        }
        const auto& rp = k.A.pattern.row_ptr();
        for (core::Index i = 0; i < m; ++i) {
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) K.vals[sz(pos++)] = k.A.vals[sz(t)];
            while (pos < K.col_ptr[sz(n + i) + 1]) K.vals[sz(pos++)] = 0.0;
        }
        // Quadratic rows: -eta_t Q_t into the Hessian block, Q_t x into the
        // Jacobian row (row t's J = a_t + Q_t x at the current x).
        for (std::size_t t = 0; t < nq; ++t) {
            if (eta[t] != 0.0)
                for (std::size_t e = 0; e < k.quad[t].v.size(); ++e)
                    K.vals[sz(qpos[t][e])] -= eta[t] * k.quad[t].v[e];
            for (std::size_t s2 = 0; s2 < jpos[t].size(); ++s2) K.vals[sz(jpos[t][s2])] += qix[t][s2];
        }
        for (core::Index j = 0; j < n; ++j) {
            const f64 base = K.vals[sz(diag_slot[sz(j)])];
            kdiag_true[sz(j)] = base - tdiag[sz(j)];
            kdiag_reg[sz(j)] = kdiag_true[sz(j)] - prox_cur * reg_boost;
            K.vals[sz(diag_slot[sz(j)])] = kdiag_reg[sz(j)];
        }
        for (core::Index i = 0; i < m; ++i) {
            const core::Index c = n + i;
            kdiag_true[sz(c)] = fixed_w[sz(i)] ? 0.0 : 1.0 / tdiag[sz(c)];
            kdiag_reg[sz(c)] = kdiag_true[sz(c)] + prox_cur * reg_boost;
            K.vals[sz(diag_slot[sz(c)])] = kdiag_reg[sz(c)];
        }
    };
    // K v with the UNREGULARISED diagonal: the system refinement targets.
    // Parallelized with fixed-size column chunks and thread-local accumulators.
    auto K_true_mul = [&](const std::vector<f64>& v, std::vector<f64>& out) {
        out.assign(sz(N), 0.0);
        core::ThreadPool& pool = core::global_pool();
        const int nthreads = pool.size();
        const core::Offset chunk_size = 16384;  // Fixed chunk independent of thread count
        const core::Offset nchunks = (N + chunk_size - 1) / chunk_size;

        // Thread-local output buffers
        std::vector<std::vector<f64>> thread_out(nthreads, std::vector<f64>(sz(N), 0.0));

        // Process chunks of columns in parallel
        auto body = [&](core::Offset chunk, int worker) {
            const core::Index col_start = chunk * chunk_size;
            const core::Index col_end = col_start + chunk_size < N ? col_start + chunk_size : N;
            auto& out_local = thread_out[sz(worker)];

            for (core::Index j = col_start; j < col_end; ++j)
                for (auto t = K.col_ptr[sz(j)]; t < K.col_ptr[sz(j) + 1]; ++t) {
                    const core::Index i = K.row_idx[sz(t)];
                    f64 val = K.vals[sz(t)];
                    if (i == j) val = kdiag_true[sz(j)];
                    out_local[sz(i)] += val * v[sz(j)];
                    if (i != j) out_local[sz(j)] += val * v[sz(i)];
                }
        };

        if (nchunks <= 1 || pool.size() == 1) {
            // Serial path: process all chunks serially with fixed order
            for (core::Offset chunk = 0; chunk < nchunks; ++chunk) body(chunk, 0);
        } else {
            pool.parallel_for(nchunks, body);
        }

        // Merge thread-local buffers into output in deterministic order
        for (int w = 0; w < nthreads; ++w)
            for (core::Index i = 0; i < N; ++i)
                out[sz(i)] += thread_out[sz(w)][sz(i)];
    };
    f64 fac_ms = 0.0, solve_ms = 0.0;   // reported under --verbose
    f64 worst_solve_rel = 0.0;          // max ||K_true x - b|| / (1+||b||) this iteration
    f64 worst_solve_run = 0.0;          // the same over the whole run (reported)
    // Solve K_true x = rhs with the REGULARISED factor as a preconditioner
    // (FGMRES; qp_krylov.cpp has why plain refinement was not enough).
    // A fixed 1e-12.  Tying it to mu (clamp(1e-4 mu, 1e-12, 1e-6), inexact
    // Newton) was tried: 5-25% faster on the instances it already proved,
    // but 10034 went from stalling at iteration 101 with primal 5.4e-5 to
    // the 300-iteration limit at 6.6e-3.  Accuracy before speed.
    const f64 solve_tol = 1e-12;
    // Unregularised augmented product for Q = 0, without materialising K.
    auto K_true_mul_lp = [&](const std::vector<f64>& v, std::vector<f64>& out) {
        out.assign(sz(N), 0.0);
        for (core::Index j = 0; j < n; ++j) out[sz(j)] = -tdiag[sz(j)] * v[sz(j)];
        for (core::Index i = 0; i < m; ++i) {
            const f64 tw = tdiag[sz(n + i)];
            const f64 g = (!fixed_w[sz(i)] && tw > 0.0) ? 1.0 / tw : 0.0;
            out[sz(n + i)] = g * v[sz(n + i)];
        }
        for (core::Index j = 0; j < n; ++j) {
            const f64 dx = v[sz(j)];
            const auto& col = ne.cols[sz(j)];
            for (std::size_t e = 0; e < col.rows.size(); ++e) {
                const core::Index i = col.rows[e];
                const f64 a = col.vals[e];
                out[sz(n + i)] += a * dx;
                out[sz(j)] += a * v[sz(n + i)];
            }
        }
    };
    auto solve_refined = [&](std::vector<f64>& rhs, const char* phase = nullptr) {
        const auto ts0 = Clock::now();
        const f64 rel = use_ne
            ? qpc::fgmres_solve([&](std::vector<f64>& b) { ne.apply(b); }, K_true_mul_lp, rhs,
                                20, solve_tol, phase)
            : qpc::fgmres_solve(fac, K_true_mul, rhs, 20, solve_tol, phase);
        worst_solve_rel = std::max(worst_solve_rel, rel);
        worst_solve_run = std::max(worst_solve_run, rel);
        solve_ms += ms_since(ts0);
    };
    // LP proximal terms taper from 1e-8 toward 1e-10 as mu falls through
    // [1e-4, 1e-8], so the regularised Newton system approaches the true one
    // near termination.  QP keeps the fixed 1e-8 (following mu was a net loss
    // on QPLIB_8906).
    auto lp_prox = [](f64 mu_now) {
        const f64 prox_hi_v = 1e-8;
        const f64 prox_lo_v = 1e-10;
        if (!(mu_now > 0.0) || mu_now >= 1e-4) return prox_hi_v;
        if (mu_now <= 1e-8) return prox_lo_v;
        const f64 u = (std::log10(mu_now) + 8.0) / 4.0;
        return std::pow(10.0, std::log10(prox_lo_v) +
                                  u * (std::log10(prox_hi_v) - std::log10(prox_lo_v)));
    };
    // Factor the regularised Newton matrix for the current T: M (normal
    // equations) or the augmented K (whose values fill_K wrote with prox_cur).
    auto factor_newton = [&](f64 prox) {
        if (!use_ne) return fac.factorize(K, sign, 1e-12);
        ne.assemble(tdiag, prox * reg_boost, prox * reg_boost, fixed_w);
        return ne.factorize();
    };
    // The normal equations broke down (non-finite factor, or the Krylov
    // refinement could not reach a usable residual): continue on the
    // augmented quasi-definite LDL', one way.
    constexpr f64 kNeSolveFail = 1e-6;
    auto fall_back_to_aug = [&](const char* why) {
        build_aug();
        use_ne = false;
        worst_solve_rel = 0.0;
        reg_boost = 1.0;
        if (opts.verbose)
            std::fprintf(stderr, "ipm normal equations abandoned (%s); augmented LDL' from here\n", why);
    };


    // ---- starting point (Mehrotra 1992, adapted to bounded variables) ----
    //
    // The primal-dual pair comes from ONE solve of the KKT system with unit
    // bound weights (T = I), which is the least-squares point that balances
    // the objective against the constraints -- not a guess.  It is then
    // pushed strictly inside the box, and the bound multipliers are seeded
    // from the reduced costs there so the starting dual residual is small.
    //
    // The classical Mehrotra shift moves each slack independently; a bounded
    // variable has TWO slacks tied to one z, so the shift here is on the
    // multipliers only, with z placed by the box geometry.  Offsets are
    // relative to each bound's magnitude: a fixed offset of 1 sits on the
    // boundary when the bound is 1e6.
    auto interior = [&](core::Index c, f64 v) {
        const f64 a = bound_lo(c), b = bound_hi(c);
        const f64 ta = std::isfinite(a) ? std::max(1.0, 1e-2 * std::fabs(a)) : 0.0;
        const f64 tb = std::isfinite(b) ? std::max(1.0, 1e-2 * std::fabs(b)) : 0.0;
        if (std::isfinite(a) && std::isfinite(b)) {
            const f64 w = b - a;
            if (w <= 2.0 * std::min(ta, tb)) return a + 0.5 * w;
            return std::clamp(v, a + std::min(ta, 0.25 * w), b - std::min(tb, 0.25 * w));
        }
        if (std::isfinite(a)) return std::max(v, a + ta);
        if (std::isfinite(b)) return std::min(v, b - tb);
        return v;
    };
    {
        std::vector<f64> ls(sz(N), 0.0);
        std::fill(tdiag.begin(), tdiag.end(), 1.0);
        if (!use_ne) fill_K();
        bool start_ok = factor_newton(rho0);
        if (!start_ok && use_ne) {
            fall_back_to_aug("non-finite factor at the starting point");
            fill_K();
            start_ok = factor_newton(rho0);
        }
        if (start_ok) {
            // Stationarity residual at (0, 0) is c; the rows contribute their
            // fixed right-hand side.
            auto start_solve = [&]() {
                for (core::Index j = 0; j < n; ++j) ls[sz(j)] = k.c[sz(j)];
                for (core::Index i = 0; i < m; ++i)
                    ls[sz(n + i)] = fixed_w[sz(i)] ? k.rl[sz(i)] : 0.0;
                solve_refined(ls, "startup");
            };
            start_solve();
            if (use_ne && !(worst_solve_rel <= kNeSolveFail)) {
                fall_back_to_aug("starting-point solve did not converge");
                fill_K();
                if (factor_newton(rho0)) start_solve();
            }
            bool ok = true;
            for (core::Index c = 0; c < N && ok; ++c) ok = std::isfinite(ls[sz(c)]);
            if (!ok) std::fill(ls.begin(), ls.end(), 0.0);
        }
        worst_solve_rel = 0.0;
        for (core::Index j = 0; j < n; ++j) z[sz(j)] = interior(j, ls[sz(j)]);
        for (core::Index i = 0; i < m; ++i) y[sz(i)] = ls[sz(n + i)];

        std::vector<f64> ax(sz(m), 0.0), aty(sz(n), 0.0), q0;
        const auto& rp0 = k.A.pattern.row_ptr();
        const auto& ci0 = k.A.pattern.col_idx();
        for (core::Index i = 0; i < m; ++i)
            for (auto t = rp0[sz(i)]; t < rp0[sz(i) + 1]; ++t) {
                ax[sz(i)] += k.A.vals[sz(t)] * z[sz(ci0[sz(t)])];
                aty[sz(ci0[sz(t)])] += k.A.vals[sz(t)] * y[sz(i)];
            }
        if (nq > 0) {
            // Quadratic rows' activity at the starting x, so w starts at the
            // row's value pushed inside, as for a linear row.
            const std::vector<f64> xs(z.begin(), z.begin() + n);
            quad_eval(xs, y, ax, &aty);
        }
        for (core::Index i = 0; i < m; ++i)
            z[sz(n + i)] = fixed_w[sz(i)] ? k.rl[sz(i)] : interior(n + i, ax[sz(i)]);

        std::vector<f64> x0(z.begin(), z.begin() + n);
        q_mul(k, x0, q0);
        f64 rmax = 0.0;
        for (core::Index j = 0; j < n; ++j)
            rmax = std::max(rmax, std::fabs(q0[sz(j)] + k.c[sz(j)] - aty[sz(j)]));
        // Keep mu0 commensurate with starting infeasibility. A hard floor of
        // 1.0 inflated early mu on well-scaled LPs (extra Newton steps).
        const f64 floor_ = (nq == 0) ? std::max(1e-2, rmax)
                                    : std::max(1.0, rmax);
        for (core::Index j = 0; j < n; ++j) {
            const f64 r = q0[sz(j)] + k.c[sz(j)] - aty[sz(j)];   // wants lam - nu = r
            if (has_l[sz(j)]) lam[sz(j)] = std::max(floor_, has_u[sz(j)] ? std::max(r, 0.0) : r);
            if (has_u[sz(j)]) nu[sz(j)] = std::max(floor_, has_l[sz(j)] ? std::max(-r, 0.0) : -r);
        }
        for (core::Index i = 0; i < m; ++i) {
            const core::Index c = n + i;
            const f64 r = -y[sz(i)];                             // wants lam - nu = -y
            if (has_l[sz(c)]) lam[sz(c)] = std::max(floor_, has_u[sz(c)] ? std::max(r, 0.0) : r);
            if (has_u[sz(c)]) nu[sz(c)] = std::max(floor_, has_l[sz(c)] ? std::max(-r, 0.0) : -r);
        }
        // Centre: raise every product to at least the average.
        f64 mu0 = 0.0;
        int np = 0;
        for (core::Index c = 0; c < N; ++c) {
            if (has_l[sz(c)]) { mu0 += (z[sz(c)] - bound_lo(c)) * lam[sz(c)]; ++np; }
            if (has_u[sz(c)]) { mu0 += (bound_hi(c) - z[sz(c)]) * nu[sz(c)]; ++np; }
        }
        mu0 = np > 0 ? mu0 / np : 1.0;
        for (core::Index c = 0; c < N; ++c) {
            if (has_l[sz(c)]) lam[sz(c)] = std::max(lam[sz(c)], mu0 / (z[sz(c)] - bound_lo(c)));
            if (has_u[sz(c)]) nu[sz(c)] = std::max(nu[sz(c)], mu0 / (bound_hi(c) - z[sz(c)]));
        }
        // A quadratic row's multiplier from the least-squares solve above
        // carries no curvature information (that solve saw J = A at x = 0),
        // and a wrong sign would start the iteration with no Hessian term:
        // on a linear objective over free columns the first Newton step is
        // then limited only by the 1e-8 proximal term (measured: |rp| 1e31
        // after one step on min -x-y s.t. (x-1)^2 + y^2 <= 1).  Start it
        // dual feasible for its slack instead: y = -nu (rdw = 0), eta = nu.
        for (std::size_t t = 0; t < nq; ++t) {
            const core::Index i = k.quad[t].row;
            y[sz(i)] = lam[sz(n + i)] - nu[sz(n + i)];
        }
    }

    std::vector<f64> qx, rdx(sz(n)), rdw(sz(m)), rp(sz(m));
    std::vector<f64> dz(sz(N)), dlam(sz(N)), dnu(sz(N)), dy(sz(m)), rhs(sz(N));
    // Second-order term of the quadratic rows in the corrector (zero for a
    // linear row, and zero throughout for a QP).
    std::vector<f64> soc(sz(m), 0.0);
    std::vector<f64> sl(sz(N)), su(sz(N)), tl(sz(N)), tu(sz(N));
    // The caller's cap, but never more than 300: an interior point that has
    // not converged in 300 Newton steps is not going to, and polishing is
    // the better use of the time.
    const int max_it = static_cast<int>(std::min<std::uint64_t>(300, std::max<std::uint64_t>(1, opts.max_iterations)));
    for (int it = 0; it < max_it; ++it) {
        const auto it_t0 = Clock::now();
        const f64 fac_at_it = fac_ms, sol_at_it = solve_ms;
        if (opts.time_limit_s > 0.0 && ms_since(t0) > opts.time_limit_s * 1000.0) {
            res.reason = "time limit";
            break;
        }
        if (core::cancel_requested(opts.cancel)) {
            res.reason = "cancelled (concurrent race lost)";
            break;
        }
        // Counted once it runs: a step stopped before any work is none.
        res.iterations = it + 1;
        // Residuals.
        std::vector<f64> x(z.begin(), z.begin() + n);
        q_mul(k, x, qx);
        std::vector<f64> aty(sz(n), 0.0), ax(sz(m), 0.0);
        {
            const auto& rpp = k.A.pattern.row_ptr();
            const auto& ci = k.A.pattern.col_idx();
            for (core::Index i = 0; i < m; ++i)
                for (auto t = rpp[sz(i)]; t < rpp[sz(i) + 1]; ++t) {
                    ax[sz(i)] += k.A.vals[sz(t)] * x[sz(ci[sz(t)])];
                    aty[sz(ci[sz(t)])] += k.A.vals[sz(t)] * y[sz(i)];
                }
        }
        if (nq > 0) {
            quad_eval(x, y, ax, &aty);   // g(x), J(x)'y, and the next K's Q_t x, eta
            // Slack reset (Byrd, Hribar & Nocedal, SIAM J. Optim. 9, 1999,
            // s.3): where a quadratic row's true value is BELOW its slack
            // variable, move the slack down to it.  w stays strictly inside
            // (g < w < ru), the row's residual becomes 0 instead of being
            // left for later Newton steps, and ru - w only grows.
            // Resetting in both directions (w = g whenever g < ru) was tried:
            // 2784 and 2456 went from proved to stalled (stationarity 4e-7,
            // 5e-8), 2468 unchanged -- shrinking a slack under its multiplier
            // wrecks centrality.  One direction only.
            for (std::size_t t = 0; t < nq; ++t) {
                const core::Index i = k.quad[t].row;
                if (ax[sz(i)] < z[sz(n + i)]) z[sz(n + i)] = ax[sz(i)];
            }
        }
        for (core::Index j = 0; j < n; ++j)
            rdx[sz(j)] = qx[sz(j)] + k.c[sz(j)] - aty[sz(j)] - lam[sz(j)] + nu[sz(j)];
        for (core::Index i = 0; i < m; ++i) {
            rdw[sz(i)] = fixed_w[sz(i)] ? 0.0 : y[sz(i)] - lam[sz(n + i)] + nu[sz(n + i)];
            rp[sz(i)] = ax[sz(i)] - z[sz(n + i)];
        }
        f64 mu = 0.0;
        int pairs = 0;
        for (core::Index c = 0; c < N; ++c) {
            // z - lo is computed, not stored: once the slack falls below the
            // rounding unit of the bound it can come out as exactly 0 and
            // make lam/s infinite.  A slack is never allowed below one ulp of
            // its bound (10038 hit this at mu ~ 1e-22).
            sl[sz(c)] = has_l[sz(c)] ? std::max(z[sz(c)] - bound_lo(c), kUlp * std::max(1.0, std::abs(bound_lo(c)))) : 0.0;
            su[sz(c)] = has_u[sz(c)] ? std::max(bound_hi(c) - z[sz(c)], kUlp * std::max(1.0, std::abs(bound_hi(c)))) : 0.0;
            if (has_l[sz(c)]) { mu += sl[sz(c)] * lam[sz(c)]; ++pairs; }
            if (has_u[sz(c)]) { mu += su[sz(c)] * nu[sz(c)]; ++pairs; }
        }
        mu = pairs > 0 ? mu / pairs : 0.0;

        // The claim is the caller's: stop as soon as the original-units
        // check passes.  Checked every iteration -- it is O(nnz).
        if (accept(x, y)) {
            res.converged = true;
            res.reason = "interior point: original-units KKT satisfied";
            break;
        }

        // LP barrier: stop and stall on the ORIGINAL model's RELATIVE
        // residuals.  The claim above stays absolute and only advisory here
        // (crossover and the checker decide what the point proves), but the
        // scaled residuals this loop minimises can sit at their floor while
        // the caller's units are converged (d2q06c: |rd| 5e-7 scaled, 3e-9
        // relative) or the reverse, and a mu floor measured against mu_start
        // says nothing about either.  The complementarity term is the usual
        // relative gap, pairs*mu / (1 + |c'x|).
        f64 orig_p = kInf, orig_d = kInf, orig_c = kInf;
        const bool lp_rel = lp && static_cast<bool>(orig_rel);
        if (lp_rel) {
            orig_rel(orig_p, orig_d);
            f64 pobj = 0.0;
            for (core::Index j = 0; j < n; ++j) pobj += k.c[sz(j)] * x[sz(j)];
            orig_c = mu * static_cast<f64>(pairs) / (1.0 + std::fabs(pobj));
            const f64 merit = std::max({orig_p, orig_d, orig_c});
            if (std::isfinite(merit) && merit < best_merit) {
                best_merit = merit;
                best_z = z;
                best_y = y;
            }
            // A margin under the tolerances: crossover starts from this point.
            constexpr f64 kRelMargin = 0.1;
            if (orig_p <= kRelMargin * opts.feas_tol && orig_d <= kRelMargin * opts.stationarity_tol &&
                orig_c <= opts.gap_tol) {
                res.converged = true;
                res.reason = "interior point: original-model relative KKT satisfied";
                break;
            }
        }

        // Stalled: complementarity at its floor and neither residual improving
        // by 10% (against the best so far, not the last: iterate-to-iterate
        // noise otherwise resets the count) over 5 iterations.  Nothing
        // further comes out of Newton steps here (8500 repeated one iterate
        // from step 35 to its time limit at 98), and polishing is what can
        // still certify, so stop.  An LP reads both off the original model's
        // relative residuals; a QP off the scaled ones and a mu floor measured
        // from where mu started ("at the floor" relative to mu_start: an
        // absolute 1e-14 never fired on 8567, mu0 2.6e5).
        {
            f64 rpn = 0.0, rdn = 0.0;
            for (f64 v : rp) rpn = std::max(rpn, std::fabs(v));
            for (f64 v : rdx) rdn = std::max(rdn, std::fabs(v));
            for (f64 v : rdw) rdn = std::max(rdn, std::fabs(v));
            const f64 sp = lp_rel ? orig_p : rpn, sd = lp_rel ? orig_d : rdn;
            const bool better = sp < 0.9 * stall_rp || sd < 0.9 * stall_rd;
            stall_rp = std::min(stall_rp, sp);
            stall_rd = std::min(stall_rd, sd);
            if (mu_start < 0.0) mu_start = mu;
            const bool at_floor = lp_rel ? orig_c <= opts.gap_tol
                                         : mu <= (nq > 0 ? 1e-20 : 1e-14) * std::max(1.0, mu_start);
            if (at_floor && !better) {
                if (++stall_count >= 5) {
                    res.reason = lp_rel ? "stalled: original-model relative residuals stopped improving"
                                        : "stalled at the mu floor";
                    break;
                }
            } else {
                stall_count = 0;
            }
        }

        for (core::Index c = 0; c < N; ++c)
            tdiag[sz(c)] = (has_l[sz(c)] ? lam[sz(c)] / sl[sz(c)] : 0.0) +
                           (has_u[sz(c)] ? nu[sz(c)] / su[sz(c)] : 0.0);
        // The proximal term in force: an LP tapers it with mu (B3), a QP keeps 1e-8.
        const f64 prox = lp ? lp_prox(mu) : rho0;
        prox_cur = prox;
        if (!use_ne) fill_K();
        // A non-finite pivot near the optimum comes from complementarity
        // ratios lam/s spanning ~1e20: more regularisation keeps the pivots
        // bounded, and refinement against the true system restores accuracy.
        // Same budget as the non-finite-direction guard below.
        const auto tf0 = Clock::now();
        const bool fac_ok = factor_newton(prox);
        fac_ms += ms_since(tf0);
        // Developer hook: FORGE_DUMP_KKT=<path> writes the first in-loop system
        // (upper CSC + pivot signs; the normal-equation matrix M for an LP on
        // that path) for apps/forge_ldlt_bench, so the factorization can be
        // timed and checked in isolation on a real instance's system.  Read
        // once; costs nothing when unset.
        if (it == 0)
            if (!core::env_switches().dump_kkt.empty())
                la::write_sym_csc(core::env_switches().dump_kkt.c_str(),
                                  use_ne ? ne.M : K, use_ne ? ne.sign : sign);
        if (!fac_ok) {
            if (use_ne && reg_boost >= 1e4) {
                fall_back_to_aug("non-finite pivot");
                continue;
            }
            if (reg_boost >= 1e4) {
                res.reason = "factorization produced a non-finite pivot even at 1e4x regularisation";
                break;
            }
            reg_boost *= 100.0;
            continue;
        }

        // Solve for a direction given complementarity targets tl, tu.
        auto direction = [&](const char* phase = nullptr) {
            for (core::Index j = 0; j < n; ++j) {
                const f64 h = (has_l[sz(j)] ? -tl[sz(j)] / sl[sz(j)] + lam[sz(j)] : 0.0) +
                              (has_u[sz(j)] ? tu[sz(j)] / su[sz(j)] - nu[sz(j)] : 0.0);
                rhs[sz(j)] = rdx[sz(j)] + h;
            }
            std::vector<f64> hw(sz(m), 0.0);
            for (core::Index i = 0; i < m; ++i) {
                const core::Index c = n + i;
                if (fixed_w[sz(i)]) { rhs[sz(c)] = -(rp[sz(i)] + soc[sz(i)]); continue; }
                hw[sz(i)] = (has_l[sz(c)] ? -tl[sz(c)] / sl[sz(c)] + lam[sz(c)] : 0.0) +
                            (has_u[sz(c)] ? tu[sz(c)] / su[sz(c)] - nu[sz(c)] : 0.0);
                rhs[sz(c)] = -(rp[sz(i)] + soc[sz(i)]) - (rdw[sz(i)] + hw[sz(i)]) / tdiag[sz(c)];
            }
            solve_refined(rhs, phase);
            for (core::Index j = 0; j < n; ++j) dz[sz(j)] = rhs[sz(j)];
            for (core::Index i = 0; i < m; ++i) {
                dy[sz(i)] = rhs[sz(n + i)];
                const core::Index c = n + i;
                dz[sz(c)] = fixed_w[sz(i)] ? 0.0
                                           : (-rdw[sz(i)] - hw[sz(i)] - dy[sz(i)]) / tdiag[sz(c)];
            }
            for (core::Index c = 0; c < N; ++c) {
                dlam[sz(c)] = has_l[sz(c)]
                    ? (tl[sz(c)] - sl[sz(c)] * lam[sz(c)] - lam[sz(c)] * dz[sz(c)]) / sl[sz(c)] : 0.0;
                dnu[sz(c)] = has_u[sz(c)]
                    ? (tu[sz(c)] - su[sz(c)] * nu[sz(c)] + nu[sz(c)] * dz[sz(c)]) / su[sz(c)] : 0.0;
            }
        };
        auto max_steps = [&](f64& ap, f64& ad) {
            ap = 1.0; ad = 1.0;
            for (core::Index c = 0; c < N; ++c) {
                if (has_l[sz(c)] && dz[sz(c)] < 0.0) ap = std::min(ap, -sl[sz(c)] / dz[sz(c)]);
                if (has_u[sz(c)] && dz[sz(c)] > 0.0) ap = std::min(ap, su[sz(c)] / dz[sz(c)]);
                if (has_l[sz(c)] && dlam[sz(c)] < 0.0) ad = std::min(ad, -lam[sz(c)] / dlam[sz(c)]);
                if (has_u[sz(c)] && dnu[sz(c)] < 0.0) ad = std::min(ad, -nu[sz(c)] / dnu[sz(c)]);
            }
            // One step length for both sides when Q != 0.  The stationarity
            // residual contains Qx, so after a step it is
            //     rd+ = (1 - ad) rd + (ap - ad) Q dx,
            // and with separate lengths the second term keeps re-injecting
            // dual infeasibility: on QPLIB_10034 (ap 0.995, ad 0.05) |rd| sat
            // at 0.2 for 250 iterations while mu fell to 1e-7.  Separate
            // lengths are an LP device; with Q = 0 the term vanishes and
            // they are kept (Wright, Primal-Dual Interior-Point Methods,
            // 1997, ch. 11 makes the same distinction).
            if (has_q) ap = ad = std::min(ap, ad);
        };

        // Predictor (affine scaling).
        std::fill(soc.begin(), soc.end(), 0.0);   // the predictor is purely linear
        std::fill(tl.begin(), tl.end(), 0.0);
        std::fill(tu.begin(), tu.end(), 0.0);
        direction("predictor");
        if (use_ne && !(worst_solve_rel <= kNeSolveFail)) {
            // More proximal regularisation first (as for a non-finite pivot);
            // the augmented factor once that budget is spent.
            if (reg_boost >= 1e4) {
                char why[64];
                std::snprintf(why, sizeof why, "Newton solve residual %.1e", worst_solve_rel);
                fall_back_to_aug(why);
            } else {
                if (opts.verbose)
                    std::fprintf(stderr, "ipm normal-equation solve residual %.1e: regularisation x100\n",
                                 worst_solve_rel);
                reg_boost *= 100.0;
                worst_solve_rel = 0.0;
            }
            continue;
        }
        f64 ap = 1.0, ad = 1.0;
        max_steps(ap, ad);
        f64 mu_aff = 0.0;
        for (core::Index c = 0; c < N; ++c) {
            if (has_l[sz(c)]) mu_aff += (sl[sz(c)] + ap * dz[sz(c)]) * (lam[sz(c)] + ad * dlam[sz(c)]);
            if (has_u[sz(c)]) mu_aff += (su[sz(c)] - ap * dz[sz(c)]) * (nu[sz(c)] + ad * dnu[sz(c)]);
        }
        mu_aff = pairs > 0 ? mu_aff / pairs : 0.0;
        const f64 sigma = mu > 0.0 ? std::pow(std::clamp(mu_aff / mu, 0.0, 1.0), 3.0) : 0.0;

        // Corrector: centring plus Mehrotra's second-order term.  For a
        // quadratic row the constraint itself is second order too:
        //   g(x + dx) = g(x) + J dx + 1/2 dx'Q dx   exactly,
        // so the corrector's linearised row carries the predictor's
        // 1/2 dx_aff'Q dx_aff, the constraint analogue of Mehrotra's
        // dz*dlam term (the second-order correction of Mehrotra 1992 applied
        // to the constraint instead of the complementarity).
        std::fill(soc.begin(), soc.end(), 0.0);
        for (std::size_t t = 0; t < nq; ++t) {
            const auto& q = k.quad[t];
            f64 v2 = 0.0;
            for (std::size_t e = 0; e < q.v.size(); ++e)
                v2 += (q.r[e] == q.c[e] ? 0.5 : 1.0) * q.v[e] * dz[sz(q.r[e])] * dz[sz(q.c[e])];
            soc[sz(q.row)] = v2;
        }
        for (core::Index c = 0; c < N; ++c) {
            tl[sz(c)] = has_l[sz(c)] ? sigma * mu - dz[sz(c)] * dlam[sz(c)] : 0.0;
            tu[sz(c)] = has_u[sz(c)] ? sigma * mu + dz[sz(c)] * dnu[sz(c)] : 0.0;
        }
        direction("corrector");
        max_steps(ap, ad);

        // Gondzio centrality correctors: re-target the products that sit
        // outside the band, re-solve with the SAME factorization, and keep
        // the result only when it buys a longer step.  This is what stops a
        // pair reaching s ~ 0 with lam ~ 1/s and freezing its variable.
        {
            constexpr f64 bmin = 0.1, bmax = 10.0;
            std::vector<f64> bz(dz), blam(dlam), bnu(dnu), bdy(dy);
            f64 bap = ap, bad = ad;
            const int gondzio_limit = std::max(0, opts.gondzio_correctors);
            for (int corr = 0; corr < gondzio_limit && mu > 0.0; ++corr) {
                const f64 ta = std::min(1.0, bap + 0.2), td = std::min(1.0, bad + 0.2);
                bool any = false;
                for (core::Index c = 0; c < N; ++c) {
                    if (has_l[sz(c)]) {
                        const f64 v = (sl[sz(c)] + ta * bz[sz(c)]) *
                                      (lam[sz(c)] + td * blam[sz(c)]);
                        const f64 want = std::clamp(v, bmin * mu, bmax * mu);
                        tl[sz(c)] = sigma * mu - bz[sz(c)] * blam[sz(c)] + (want - v);
                        any = any || want != v;
                    }
                    if (has_u[sz(c)]) {
                        const f64 v = (su[sz(c)] - ta * bz[sz(c)]) *
                                      (nu[sz(c)] + td * bnu[sz(c)]);
                        const f64 want = std::clamp(v, bmin * mu, bmax * mu);
                        tu[sz(c)] = sigma * mu + bz[sz(c)] * bnu[sz(c)] + (want - v);
                        any = any || want != v;
                    }
                }
                if (!any) break;
                direction("gondzio");
                f64 nap = 1.0, nad = 1.0;
                max_steps(nap, nad);
                if (nap < bap + 0.05 && nad < bad + 0.05) break;
                bz = dz; blam = dlam; bnu = dnu; bdy = dy;
                bap = nap; bad = nad;
            }
            dz = bz; dlam = blam; dnu = bnu; dy = bdy;
            ap = bap; ad = bad;
        }

        // A direction with a non-finite entry is never stepped along: raise
        // the regularisation and redo the iteration.
        {
            bool finite = true;
            for (core::Index c = 0; c < N && finite; ++c)
                finite = std::isfinite(dz[sz(c)]) && std::isfinite(dlam[sz(c)]) &&
                         std::isfinite(dnu[sz(c)]);
            for (core::Index i = 0; i < m && finite; ++i) finite = std::isfinite(dy[sz(i)]);
            if (!finite) {
                if (reg_boost >= 1e4) {
                    res.reason = "direction not finite even at 1e4x regularisation";
                    break;
                }
                reg_boost *= 100.0;
                continue;
            }
        }
        // Fraction-to-boundary grows toward 0.9995 as mu falls (Mehrotra /
        // Andersen–Andersen). Frozen 0.995 left late iterations short.
        const f64 eta_ftb = (nq == 0)
            ? std::min(0.9995, std::max(0.99, 1.0 - mu))
            : 0.995;
        ap = std::min(1.0, eta_ftb * ap);
        ad = std::min(1.0, eta_ftb * ad);
        // Quadratic rows: step to the boundary of the NONLINEAR constraint.
        // Along the step a row's value is exactly quadratic,
        //   g(x + a dx) = g(x) + a J dx + a^2/2 dx'Q dx,
        // while the linearised system only saw the first two terms.  Where
        // an inactive row is all that bounds a direction, the Newton step in
        // that direction is long (its curvature eta Q -> 0 with the
        // multiplier) and the true row value overshoots by a^2/2 dx'Q dx:
        // measured on QPLIB_2784, |rp| 9e-4 -> 2e2 in one step, after which
        // the iteration crawled at step 0.01 into the mu floor.  So a row
        // that is feasible now, g(x) < ru, keeps at least 0.5% of its slack
        // ru - g(x): the fraction-to-boundary rule applied to the true
        // constraint -- what a conic IPM's step-to-boundary of the cone does
        // for a second-order-cone reformulation of the same row.  The
        // largest such step solves 1/2 q2 a^2 + q1 a = tau s0, taken in the
        // cancellation-free form 2c / (b + sqrt(b^2 + 4ac)).
        if (nq > 0) {
            const auto& rpa = k.A.pattern.row_ptr();
            const auto& cia = k.A.pattern.col_idx();
            f64 amax = ap;
            for (std::size_t t = 0; t < nq; ++t) {
                const core::Index i = k.quad[t].row;
                const f64 s0 = k.ru[sz(i)] - ax[sz(i)];
                if (!(s0 > 0.0)) continue;   // infeasible now: the residual path decides
                f64 q1 = 0.0, q2 = 0.0;
                for (auto e = rpa[sz(i)]; e < rpa[sz(i) + 1]; ++e) q1 += k.A.vals[sz(e)] * dz[sz(cia[sz(e)])];
                for (std::size_t s2 = 0; s2 < qsup[t].size(); ++s2) q1 += qix[t][s2] * dz[sz(qsup[t][s2])];
                const auto& q = k.quad[t];
                for (std::size_t e = 0; e < q.v.size(); ++e)
                    q2 += (q.r[e] == q.c[e] ? 1.0 : 2.0) * q.v[e] * dz[sz(q.r[e])] * dz[sz(q.c[e])];
                q2 = std::max(q2, 0.0);   // PSD: negative only by rounding
                const f64 cc = 0.995 * s0;
                const f64 den = q1 + std::sqrt(q1 * q1 + 2.0 * q2 * cc);
                if (den > 0.0) amax = std::min(amax, 2.0 * cc / den);
            }
            ap = ad = std::min(ap, amax);
        }
        for (core::Index c = 0; c < N; ++c) {
            if (!(c >= n && fixed_w[sz(c - n)])) z[sz(c)] += ap * dz[sz(c)];
            lam[sz(c)] += ad * dlam[sz(c)];
            nu[sz(c)] += ad * dnu[sz(c)];
        }
        for (core::Index i = 0; i < m; ++i) y[sz(i)] += ad * dy[sz(i)];
        if (opts.verbose)
        {
            f64 rpn = 0.0, rdn = 0.0;
            core::Index rpi = -1;
            for (core::Index i = 0; i < m; ++i)
                if (std::fabs(rp[sz(i)]) > rpn) { rpn = std::fabs(rp[sz(i)]); rpi = i; }
            for (f64 v : rdx) rdn = std::max(rdn, std::fabs(v));
            for (f64 v : rdw) rdn = std::max(rdn, std::fabs(v));
            if (nq > 0 && rpi >= 0) std::fprintf(stderr, "ipm   worst rp row %lld (core)\n", static_cast<long long>(rpi));
            const int nreg = use_ne ? ne.reg_pivots : static_cast<int>(fac.regularized_pivots());
            std::fprintf(stderr, "ipm it=%d mu=%.3e |rp|=%.3e |rd|=%.3e sigma=%.2f ap=%.3f ad=%.3f reg=%d solve_res=%.1e iter %.0fms (factor %.0f, solves %.0f) total %.0fms\n",
                         it + 1, mu, rpn, rdn, sigma, ap, ad, nreg,
                         worst_solve_rel, ms_since(it_t0), fac_ms - fac_at_it, solve_ms - sol_at_it,
                         ms_since(t0));
            if (lp_rel)
                std::fprintf(stderr, "ipm   original model: primal_rel=%.2e dual_rel=%.2e compl_rel=%.2e\n",
                             orig_p, orig_d, orig_c);
        }
        worst_solve_rel = 0.0;
        if (it + 1 == max_it) res.reason = "iteration limit";
    }
    // An LP run that ended without meeting a test (stalled, iteration or time
    // limit) returns its best point, not the last one.
    if (!res.converged && !best_z.empty()) {
        z = best_z;
        y = best_y;
    }
    res.x.assign(z.begin(), z.begin() + n);
    res.y = y;
    res.worst_solve_residual = worst_solve_run;
    return res;
}

}  // namespace

core::RawResult solve_qp_ipm(const QpProblem& p, const QpOptions& opts, QpDiagnostics& diag) {
    const auto t0 = Clock::now();
    diag = QpDiagnostics{};
    core::RawResult raw;
    raw.engine = "qp_ipm";
    raw.backend = "cpu";
    const auto& lp = p.linear;
    auto fail = [&](core::Status s, const std::string& why) {
        raw.proposed_status = s;
        raw.termination_reason = why;
        diag.termination_reason = why;
        diag.total_ms = ms_since(t0);
        return raw;
    };
    if (lp.maximize) return fail(core::Status::Unsupported, "convex QP supports minimization only");
    try { lp.validate(); } catch (const std::exception& e) {
        return fail(core::Status::Unsupported, std::string("invalid QP: ") + e.what());
    }
    std::string why;
    f64 slack = 0.0;
    if (!certify_qp_convex(p, opts, why, slack)) return fail(core::Status::Unsupported, why);
    diag.convexity_certified = true;

    // Scale (Ruiz), exactly as the first-order path does.
    const auto sc = opts.scale ? qpc::ruiz(p, opts.ruiz_iterations)
                               : qpc::Scaling{std::vector<f64>(qpc::sz(lp.n_cols()), 1.0),
                                              std::vector<f64>(qpc::sz(lp.n_rows()), 1.0)};
    const QpProblem ps = qpc::apply_scaling(p, sc);
    const auto sb = qpc::scale_bounds({lp.col_lo, lp.col_hi, lp.row_lo, lp.row_hi}, sc);
    const core::Index n0 = lp.n_cols(), m0 = lp.n_rows();

    // Presolve: substitute fixed columns, drop free rows.
    Core k;
    std::vector<f64> xfix(sz(n0), 0.0);
    std::vector<std::uint8_t> is_fixed(sz(n0), 0);
    for (core::Index j = 0; j < n0; ++j)
        if (sb.col_lo[sz(j)] == sb.col_hi[sz(j)]) { is_fixed[sz(j)] = 1; xfix[sz(j)] = sb.col_lo[sz(j)]; }
    std::vector<core::Index> new_col(sz(n0), -1);
    for (core::Index j = 0; j < n0; ++j)
        if (!is_fixed[sz(j)]) { new_col[sz(j)] = k.n++; k.col_of.push_back(j); }
    // Q x_fixed contributions and constant.
    std::vector<f64> qfx(sz(n0), 0.0);
    if (qpc::has_sparse_q(ps)) {
        const auto& qp_ = ps.q_matrix.pattern.row_ptr();
        const auto& qi = ps.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n0; ++i)
            for (auto t = qp_[sz(i)]; t < qp_[sz(i) + 1]; ++t)
                if (is_fixed[sz(qi[sz(t)])]) qfx[sz(i)] += ps.q_matrix.vals[sz(t)] * xfix[sz(qi[sz(t)])];
    } else {
        for (core::Index j = 0; j < n0; ++j) qfx[sz(j)] = ps.q_diag[sz(j)] * xfix[sz(j)];
    }
    k.offset = ps.linear.obj_offset;
    for (core::Index j = 0; j < n0; ++j)
        if (is_fixed[sz(j)])
            k.offset += ps.linear.c[sz(j)] * xfix[sz(j)] + 0.5 * xfix[sz(j)] * qfx[sz(j)];
    k.qcol.assign(sz(k.n), {});
    k.c.resize(sz(k.n));
    for (core::Index j = 0; j < n0; ++j) {
        if (is_fixed[sz(j)]) continue;
        const core::Index jj = new_col[sz(j)];
        k.c[sz(jj)] = ps.linear.c[sz(j)] + qfx[sz(j)];
        k.l.push_back(sb.col_lo[sz(j)]);
        k.u.push_back(sb.col_hi[sz(j)]);
    }
    if (qpc::has_sparse_q(ps)) {
        const auto& qp_ = ps.q_matrix.pattern.row_ptr();
        const auto& qi = ps.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n0; ++i) {
            if (is_fixed[sz(i)]) continue;
            for (auto t = qp_[sz(i)]; t < qp_[sz(i) + 1]; ++t) {
                const core::Index j = qi[sz(t)];
                if (is_fixed[sz(j)]) continue;
                const core::Index a = new_col[sz(i)], b = new_col[sz(j)];
                if (a <= b) k.qcol[sz(b)].push_back({a, ps.q_matrix.vals[sz(t)]});
            }
        }
    } else {
        for (core::Index j = 0; j < n0; ++j)
            if (!is_fixed[sz(j)] && ps.q_diag[sz(j)] != 0.0)
                k.qcol[sz(new_col[sz(j)])].push_back({new_col[sz(j)], ps.q_diag[sz(j)]});
    }
    // Rows: shift by fixed columns; drop free rows.
    std::vector<f64> row_shift(sz(m0), 0.0);
    {
        const auto& rp = ps.linear.A.pattern.row_ptr();
        const auto& ci = ps.linear.A.pattern.col_idx();
        std::vector<core::Index> tr, tc;
        std::vector<f64> tv;
        for (core::Index i = 0; i < m0; ++i) {
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
                if (is_fixed[sz(ci[sz(t)])]) row_shift[sz(i)] += ps.linear.A.vals[sz(t)] * xfix[sz(ci[sz(t)])];
            if (!std::isfinite(sb.row_lo[sz(i)]) && !std::isfinite(sb.row_hi[sz(i)])) continue;
            const core::Index ii = k.m++;
            k.row_of.push_back(i);
            k.rl.push_back(sb.row_lo[sz(i)] - row_shift[sz(i)]);
            k.ru.push_back(sb.row_hi[sz(i)] - row_shift[sz(i)]);
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
                if (!is_fixed[sz(ci[sz(t)])]) {
                    tr.push_back(ii); tc.push_back(new_col[sz(ci[sz(t)])]); tv.push_back(ps.linear.A.vals[sz(t)]);
                }
        }
        k.A = sparse::from_triplets(k.m, k.n, tr, tc, tv);
    }

    // Objective scaling: normalise so the largest objective coefficient is
    // O(1).  Primal unchanged; duals come back divided by omega.
    f64 omega = 1.0;
    {
        f64 mx = 0.0;
        for (f64 v : k.c) mx = std::max(mx, std::fabs(v));
        for (const auto& col : k.qcol) for (const auto& e : col) mx = std::max(mx, std::fabs(e.second));
        if (mx > 0.0 && std::isfinite(mx)) omega = 1.0 / mx;
        for (f64& v : k.c) v *= omega;
        for (auto& col : k.qcol) for (auto& e : col) e.second *= omega;
    }

    // Map a core iterate to ORIGINAL units, in PDHCG's dual sign (y_p = -y).
    const backend::LaneBounds ob{lp.col_lo, lp.col_hi, lp.row_lo, lp.row_hi};
    auto to_original = [&](const std::vector<f64>& xc, const std::vector<f64>& yc,
                           std::vector<f64>& xo, std::vector<f64>& yo) {
        xo.assign(sz(n0), 0.0);
        yo.assign(sz(m0), 0.0);
        for (core::Index j = 0; j < n0; ++j)
            xo[sz(j)] = (is_fixed[sz(j)] ? xfix[sz(j)] : xc[sz(new_col[sz(j)])]) * sc.dc[sz(j)];
        for (core::Index ii = 0; ii < k.m; ++ii) {
            const core::Index i = k.row_of[sz(ii)];
            yo[sz(i)] = -yc[sz(ii)] * sc.dr[sz(i)] / omega;
        }
    };
    qpc::OriginalKkt last;
    auto accept = [&](const std::vector<f64>& xc, const std::vector<f64>& yc) {
        std::vector<f64> xo, yo;
        to_original(xc, yc, xo, yo);
        last = qpc::kkt_original(p, ob, xo, yo);
        return qpc::kkt_ok(last, opts);
    };

    // An LP (Q = 0) also gets the original model's relative residuals, read
    // off the check `accept` just ran.
    const auto orig_rel = [&](f64& primal_rel, f64& dual_rel) {
        primal_rel = last.primal_rel;
        dual_rel = last.dual_rel;
    };
    const auto r = run_ipm(k, opts, t0, accept, orig_rel);
    to_original(r.x, r.y, raw.x, raw.y);
    last = qpc::kkt_original(p, ob, raw.x, raw.y);
    bool ok = qpc::kkt_ok(last, opts);
    // An interior point that stalled near the optimum usually already knows
    // the active set; polishing turns that into an exact solve, and the
    // original-units check decides whether to keep it.
    bool polished = false;
    // Polished in the Ruiz-scaled space, as OSQP does: unscaled, 10034 gave
    // a row regularisation of 4.8e9 and garbage solves.
    // Not once the time is spent: the loop stopped at the limit, and the
    // polish's active-set factorization is unbounded work (dfl001,
    // --time-limit 2: the interior point stopped at 1.2 s, the polish ran
    // another 4.9 s).
    const bool time_left = opts.time_limit_s <= 0.0 ||
        ms_since(t0) < opts.time_limit_s * 1000.0;
    if (!ok && opts.polish && time_left && !core::cancel_requested(opts.cancel))
        polished = ok = qpc::polish_scaled(p, ob, sc, ps, sb, opts, raw.x, raw.y, last);
    diag.iterations = static_cast<std::uint64_t>(r.iterations);
    diag.worst_solve_residual = r.worst_solve_residual;
    diag.primal_residual = last.primal;
    diag.stationarity = last.dual_res;
    diag.primal_residual_rel = last.primal_rel;
    diag.stationarity_rel = last.dual_rel;
    diag.primal_net = last.primal_net;
    diag.stationarity_net = last.dual_net;
    diag.gap_net = last.gap_net;
    diag.gap_rel = last.gap;
    diag.gap_finite = last.gap_finite;
    diag.objective = last.objective;
    raw.objective = last.objective;
    raw.iterations = diag.iterations;
    raw.proposed_status = ok ? core::Status::Optimal : core::Status::Interrupted;
    raw.proposed_level = ok ? core::ProofLevel::ProvedKKT : core::ProofLevel::None;
    raw.dual_bound = ok ? last.dual_bound : core::kNaN;
    raw.termination_reason = ok ? (polished ? "interior point + polish: original-units KKT satisfied"
                                            : r.reason)
                                : "interior point stopped: " + r.reason;
    diag.termination_reason = raw.termination_reason;
    diag.total_ms = ms_since(t0);
    return raw;
}

// ---- convex QCQP ------------------------------------------------------------
//
// The same iteration (run_ipm) on a Core that carries quadratic rows.  Why a
// nonlinear primal-dual IPM (Lagrangian Hessian in the KKT matrix) rather
// than a second-order-cone reformulation with Nesterov-Todd scaling:
//   * it is THIS solver's proven machinery -- the quasi-definite LDL', the
//     FGMRES refinement, Gondzio correctors, Ruiz scaling, the presolve and
//     the original-units claim all carry over; the only new objects are the
//     Jacobian rows a_i + Q_i x and the Hessian term eta_i Q_i.  An SOC
//     route needs a factor F_i of every Q_i (Q_i = F_i'F_i) to even state
//     the cone, dense scaling blocks per cone and a new conic IPM;
//   * convexity keeps the system quasi-definite: canonical rows are
//     "convex <= level", their multipliers are sign-definite, so
//     Q0 + sum eta_i Q_i is PSD (quad_eval in run_ipm has the one place
//     where a transiently wrong-signed multiplier is clipped);
//   * the QPLIB convex-QC set is norm balls with 2-3 variables (LCD) and a
//     single dense PSD row (LMC portfolio risk): their Hessians are exactly
//     what a (1,1) block holds cheaply.
// Method: the primal-dual interior point for convex nonlinear programs of
// Nocedal & Wright ch. 19 / Vanderbei & Shanno (Comput. Optim. Appl. 13,
// 1999), specialised to quadratic functions, where the constraint Hessians
// are constant and the second-order term of each row is exact (the
// corrector's `soc`).
//
// Weak duality, for the claim's gap and the branch-and-bound's bound: with
// y_i Q_i PSD for every quadratic row (PDHCG sign, y > 0 at an upper
// bound), the Lagrangian f(x) + sum_i y_i g_i(x) is convex, so its tangent
// plane at any x underestimates it; minimising the tangent over the box and
// the row bounds gives
//   D = -1/2 x'Q0x - 1/2 sum_i y_i x'Q_i x - px - py
// (kkt_original's px, py), and primal - D = x'Q0x + c'x + sum_i y_i x'Q_i x/2
// + px + py.
namespace ipmc {

void build_qcqp_core(const QcqpProblem& p0, const std::vector<std::int8_t>* orientation,
                     const QpOptions& opts, QcqpBuilt& B) {
    const auto& lp0 = p0.qp.linear;
    const core::Index n0 = lp0.n_cols(), m0 = lp0.n_rows();
    // Canonical copy: concave-above rows negated (exact in floating point),
    // so every quadratic row is "convex <= level".
    QcqpProblem p = p0;
    B.row_sign.assign(sz(m0), 1);
    for (std::size_t t = 0; orientation && t < p.quad.size(); ++t) {
        if ((*orientation)[t] >= 0) continue;
        const core::Index i = p.quad[t].row;
        B.row_sign[sz(i)] = -1;
        auto& L = p.qp.linear;
        const auto& rp = L.A.pattern.row_ptr();
        for (auto e = rp[sz(i)]; e < rp[sz(i) + 1]; ++e) L.A.vals[sz(e)] = -L.A.vals[sz(e)];
        const f64 lo = L.row_lo[sz(i)], hi = L.row_hi[sz(i)];
        L.row_lo[sz(i)] = -hi;
        L.row_hi[sz(i)] = -lo;
        for (f64& v : p.quad[t].v) v = -v;
    }
    const auto& lp = p.qp.linear;
    B.sc = opts.scale ? qpc::ruiz(p.qp, opts.ruiz_iterations, &p.quad)
                      : qpc::Scaling{std::vector<f64>(sz(n0), 1.0), std::vector<f64>(sz(m0), 1.0)};
    const auto& sc = B.sc;
    const QpProblem ps = qpc::apply_scaling(p.qp, sc);
    const auto sb = qpc::scale_bounds({lp.col_lo, lp.col_hi, lp.row_lo, lp.row_hi}, sc);
    // Scaled Hessians: row i scaled by dr_i, x = Dc x~.
    std::vector<QuadRow> qs = p.quad;
    for (auto& q : qs)
        for (std::size_t e = 0; e < q.v.size(); ++e)
            q.v[e] *= sc.dr[sz(q.row)] * sc.dc[sz(q.r[e])] * sc.dc[sz(q.c[e])];

    Core& k = B.k;
    B.xfix.assign(sz(n0), 0.0);
    B.is_fixed.assign(sz(n0), 0);
    auto& xfix = B.xfix;
    auto& is_fixed = B.is_fixed;
    for (core::Index j = 0; j < n0; ++j)
        if (sb.col_lo[sz(j)] == sb.col_hi[sz(j)]) { is_fixed[sz(j)] = 1; xfix[sz(j)] = sb.col_lo[sz(j)]; }
    B.new_col.assign(sz(n0), -1);
    auto& new_col = B.new_col;
    for (core::Index j = 0; j < n0; ++j)
        if (!is_fixed[sz(j)]) { new_col[sz(j)] = k.n++; k.col_of.push_back(j); }
    // Objective: exactly solve_qp_ipm's substitution.
    std::vector<f64> qfx(sz(n0), 0.0);
    const bool sparse_q = qpc::has_sparse_q(ps);
    if (sparse_q) {
        const auto& qp_ = ps.q_matrix.pattern.row_ptr();
        const auto& qi = ps.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n0; ++i)
            for (auto t = qp_[sz(i)]; t < qp_[sz(i) + 1]; ++t)
                if (is_fixed[sz(qi[sz(t)])]) qfx[sz(i)] += ps.q_matrix.vals[sz(t)] * xfix[sz(qi[sz(t)])];
    } else {
        for (core::Index j = 0; j < n0 && sz(j) < ps.q_diag.size(); ++j) qfx[sz(j)] = ps.q_diag[sz(j)] * xfix[sz(j)];
    }
    k.offset = ps.linear.obj_offset;
    for (core::Index j = 0; j < n0; ++j)
        if (is_fixed[sz(j)])
            k.offset += ps.linear.c[sz(j)] * xfix[sz(j)] + 0.5 * xfix[sz(j)] * qfx[sz(j)];
    k.qcol.assign(sz(k.n), {});
    k.c.resize(sz(k.n));
    for (core::Index j = 0; j < n0; ++j) {
        if (is_fixed[sz(j)]) continue;
        const core::Index jj = new_col[sz(j)];
        k.c[sz(jj)] = ps.linear.c[sz(j)] + qfx[sz(j)];
        k.l.push_back(sb.col_lo[sz(j)]);
        k.u.push_back(sb.col_hi[sz(j)]);
    }
    if (sparse_q) {
        const auto& qp_ = ps.q_matrix.pattern.row_ptr();
        const auto& qi = ps.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n0; ++i) {
            if (is_fixed[sz(i)]) continue;
            for (auto t = qp_[sz(i)]; t < qp_[sz(i) + 1]; ++t) {
                const core::Index j = qi[sz(t)];
                if (is_fixed[sz(j)]) continue;
                const core::Index a = new_col[sz(i)], b = new_col[sz(j)];
                if (a <= b) k.qcol[sz(b)].push_back({a, ps.q_matrix.vals[sz(t)]});
            }
        }
    } else {
        for (core::Index j = 0; j < n0 && sz(j) < ps.q_diag.size(); ++j)
            if (!is_fixed[sz(j)] && ps.q_diag[sz(j)] != 0.0)
                k.qcol[sz(new_col[sz(j)])].push_back({new_col[sz(j)], ps.q_diag[sz(j)]});
    }
    // Rows.  A quadratic row with some columns fixed splits three ways:
    // both fixed -> a constant (shifts the bounds), one fixed -> a linear
    // term on the other (joins the row's A entries), none -> stays in Q.
    std::vector<std::size_t> quad_of(sz(m0), qs.size());
    for (std::size_t t = 0; t < qs.size(); ++t) quad_of[sz(qs[t].row)] = t;
    const auto& rp = ps.linear.A.pattern.row_ptr();
    const auto& ci = ps.linear.A.pattern.col_idx();
    std::vector<core::Index> tr, tc;
    std::vector<f64> tv;
    for (core::Index i = 0; i < m0; ++i) {
        if (!std::isfinite(sb.row_lo[sz(i)]) && !std::isfinite(sb.row_hi[sz(i)])) continue;
        f64 shift = 0.0;
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
            if (is_fixed[sz(ci[sz(t)])]) shift += ps.linear.A.vals[sz(t)] * xfix[sz(ci[sz(t)])];
        const core::Index ii = k.m++;
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
            if (!is_fixed[sz(ci[sz(t)])]) {
                tr.push_back(ii); tc.push_back(new_col[sz(ci[sz(t)])]); tv.push_back(ps.linear.A.vals[sz(t)]);
            }
        if (const std::size_t t = quad_of[sz(i)]; t < qs.size()) {
            const auto& q = qs[t];
            Core::Quad cq;
            cq.row = ii;
            for (std::size_t e = 0; e < q.v.size(); ++e) {
                const core::Index r = q.r[e], c = q.c[e];
                const bool fr = is_fixed[sz(r)] != 0, fc = is_fixed[sz(c)] != 0;
                if (fr && fc) {
                    shift += (r == c ? 0.5 : 1.0) * q.v[e] * xfix[sz(r)] * xfix[sz(c)];
                } else if (fr || fc) {
                    // r != c here (a diagonal entry has both or neither fixed).
                    const core::Index freej = fr ? c : r, fixj = fr ? r : c;
                    tr.push_back(ii); tc.push_back(new_col[sz(freej)]); tv.push_back(q.v[e] * xfix[sz(fixj)]);
                } else {
                    cq.r.push_back(new_col[sz(r)]);
                    cq.c.push_back(new_col[sz(c)]);
                    cq.v.push_back(q.v[e]);
                }
            }
            if (!cq.v.empty()) k.quad.push_back(std::move(cq));
        }
        k.row_of.push_back(i);
        k.rl.push_back(sb.row_lo[sz(i)] - shift);
        k.ru.push_back(sb.row_hi[sz(i)] - shift);
    }
    k.A = sparse::from_triplets(k.m, k.n, tr, tc, tv);

    // Objective scaling, as solve_qp_ipm: the objective only -- constraint
    // Hessians are not objective terms; their multipliers absorb omega.
    f64 mx = 0.0;
    for (f64 v : k.c) mx = std::max(mx, std::fabs(v));
    for (const auto& col : k.qcol) for (const auto& e : col) mx = std::max(mx, std::fabs(e.second));
    if (mx > 0.0 && std::isfinite(mx)) B.omega = 1.0 / mx;
    for (f64& v : k.c) v *= B.omega;
    for (auto& col : k.qcol) for (auto& e : col) e.second *= B.omega;
}

}  // namespace ipmc

core::RawResult solve_qcqp_ipm(const QcqpProblem& p0, const QpOptions& opts, QpDiagnostics& diag,
                               const QcqpConvexity* cert_in) {
    const auto t0 = Clock::now();
    diag = QpDiagnostics{};
    core::RawResult raw;
    raw.engine = "qcqp_ipm";
    raw.backend = "cpu";
    const auto& lp = p0.qp.linear;
    auto fail = [&](core::Status s, const std::string& why) {
        raw.proposed_status = s;
        raw.termination_reason = why;
        diag.termination_reason = why;
        diag.total_ms = ms_since(t0);
        return raw;
    };
    if (lp.maximize) return fail(core::Status::Unsupported, "convex QCQP supports minimization only");
    try { p0.validate(); } catch (const std::exception& e) {
        return fail(core::Status::Unsupported, std::string("invalid QCQP: ") + e.what());
    }
    bool has_int = false;
    for (std::size_t j = 0; j < lp.is_integer.size(); ++j) has_int = has_int || lp.is_integer[j];
    if (!p0.has_quadratic_constraints() && !has_int) {
        // No quadratic row: exactly the QP interior point and its claim.
        return solve_qp_ipm(p0.qp, opts, diag);
    }
    // Convexity first: a nonconvex row is the more fundamental refusal (no
    // engine here would take the model), and its reason names the row.
    QcqpConvexity own;
    const QcqpConvexity* cert = cert_in;
    if (!cert) {
        own = certify_qcqp_convex(p0, opts);
        cert = &own;
    }
    if (!cert->ok || cert->orientation.size() != p0.quad.size())
        return fail(core::Status::Unsupported, "convex QCQP interior point (model has quadratic constraints): " +
                    (cert->ok ? std::string("certificate does not match the model") : cert->reason));
    if (has_int)
        return fail(core::Status::Unsupported,
                    "the QCQP interior point is continuous; the model has integer columns "
                    "(the mixed-integer QCQP branch-and-bound takes them)");
    diag.convexity_certified = true;

    ipmc::QcqpBuilt B;
    ipmc::build_qcqp_core(p0, &cert->orientation, opts, B);
    const core::Index n0 = lp.n_cols(), m0 = lp.n_rows();
    const auto& sc = B.sc;
    const Core& k = B.k;
    const backend::LaneBounds ob{lp.col_lo, lp.col_hi, lp.row_lo, lp.row_hi};
    // Core iterate -> ORIGINAL units, PDHCG's dual sign, original row sign.
    auto to_original = [&](const std::vector<f64>& xc, const std::vector<f64>& yc,
                           std::vector<f64>& xo, std::vector<f64>& yo) {
        xo.assign(sz(n0), 0.0);
        yo.assign(sz(m0), 0.0);
        for (core::Index j = 0; j < n0; ++j)
            xo[sz(j)] = (B.is_fixed[sz(j)] ? B.xfix[sz(j)] : xc[sz(B.new_col[sz(j)])]) * sc.dc[sz(j)];
        for (core::Index ii = 0; ii < k.m; ++ii) {
            const core::Index i = k.row_of[sz(ii)];
            yo[sz(i)] = -yc[sz(ii)] * sc.dr[sz(i)] / B.omega * B.row_sign[sz(i)];
        }
    };
    qpc::OriginalKkt last;
    auto accept = [&](const std::vector<f64>& xc, const std::vector<f64>& yc) {
        std::vector<f64> xo, yo;
        to_original(xc, yc, xo, yo);
        last = qpc::kkt_original_qcqp(p0, ob, xo, yo);
        return qpc::kkt_ok(last, opts);
    };
    const auto r = run_ipm(k, opts, t0, accept);
    to_original(r.x, r.y, raw.x, raw.y);
    last = qpc::kkt_original_qcqp(p0, ob, raw.x, raw.y);
    // No polish: its active-set solve is for linear rows; an active
    // quadratic row would need a nonlinear equality solve.
    const bool ok = qpc::kkt_ok(last, opts);
    diag.iterations = static_cast<std::uint64_t>(r.iterations);
    diag.worst_solve_residual = r.worst_solve_residual;
    diag.primal_residual = last.primal;
    diag.stationarity = last.dual_res;
    diag.primal_residual_rel = last.primal_rel;
    diag.stationarity_rel = last.dual_rel;
    diag.primal_net = last.primal_net;
    diag.stationarity_net = last.dual_net;
    diag.gap_net = last.gap_net;
    diag.gap_rel = last.gap;
    diag.gap_finite = last.gap_finite;
    diag.objective = last.objective;
    raw.objective = last.objective;
    raw.iterations = diag.iterations;
    raw.proposed_status = ok ? core::Status::Optimal : core::Status::Interrupted;
    raw.proposed_level = ok ? core::ProofLevel::ProvedKKT : core::ProofLevel::None;
    raw.dual_bound = ok ? last.dual_bound : core::kNaN;
    raw.termination_reason = ok ? "QCQP interior point: original-units KKT satisfied (quadratic rows included)"
                                : "QCQP interior point stopped: " + r.reason;
    diag.termination_reason = raw.termination_reason;
    diag.total_ms = ms_since(t0);
    return raw;
}

}  // namespace forge::engines
