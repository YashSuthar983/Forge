// SOR — solution polishing for convex QP (Track 1, node A4, first half).
//
// LAYER L3.  Method: Stellato, Banjac, Goulart, Bemporad & Boyd, "OSQP: an
// operator splitting solver for quadratic programs", Math. Prog. Comp. 12
// (2020), s.5 "solution polishing".  Implemented from the paper; no solver
// source consulted.
//
// WHY.  Both of our convex engines can end with the RIGHT answer and a
// residual that will not go below tolerance: the IPM on QPLIB_8845 has the
// objective right to 1e-9 and a stationarity residual stuck at 0.49; on
// _8906 it stalls at 2.8e-5.  An approximate point usually already knows
// which constraints are active.  Fixing that guess turns the problem into an
// equality-constrained QP, whose solution is ONE linear solve -- exact up to
// that solve's accuracy, not up to an iteration count.
//
// HOW.
//   1. Guess the active set from (x, y) with the primal-dual indicator: a
//      bound is active when its slack is smaller than its multiplier.
//   2. Fix active bounds at their value; treat active rows as equalities.
//   3. Solve the quasi-definite system
//          [ Q_FF + delta I   C_F'     ] [x_F]   [ -(c_F + Q_FB x_B)      ]
//          [ C_F             -delta I  ] [lam] = [ b_act - C_B x_B        ]
//      over the free columns F, solved for delta = 0 by FGMRES with the
//      regularised factor as preconditioner (the paper refines; see
//      qp_krylov.cpp for why that was not enough), on our sparse LDL'.
//   4. Recover row multipliers (y = lam on active rows, 0 elsewhere).
//   5. If the check fails, update the guess from the polished point --
//      primal-dual active set (Hintermueller, Ito & Kunisch 2002): fix what
//      left its box, release what has a wrong-signed multiplier; and when
//      the active rows cannot be met with the pinned columns (the reduced
//      row residual stagnates), release the pinned columns whose multiplier
//      is nearest zero -- and re-solve, up to 10 rounds or until the guess
//      stops changing.
//
// THE CLAIM.  Polishing proves nothing by itself: a wrong active-set guess
// gives a clean-looking wrong point.  The polished (x, y) is handed to the
// SAME original-units KKT check the engines use (qpc::kkt_original), and is
// kept only if it passes there.  Otherwise the caller keeps its own point.
#include <cstdio>
#include "sor/engines/qp.hpp"
#include "sor/la/ldlt.hpp"

#include "qp_common.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sor::engines {

namespace qpc {

bool polish(const QpProblem& p, const backend::LaneBounds& b, const QpOptions& opts,
            std::vector<f64>& x, std::vector<f64>& y, OriginalKkt& out, const PolishJudge* judge) {
    const auto& lp = p.linear;
    const core::Index n = lp.n_cols(), m = lp.n_rows();
    if (x.size() != sz(n) || y.size() != sz(m)) return false;

    // r = Q x + c + A' y  (PDHCG's sign: at an optimum r_j > 0 pushes x_j
    // to its lower bound, r_j < 0 to its upper; y_i > 0 means row i sits at
    // its upper bound, y_i < 0 at its lower).
    std::vector<f64> qx(sz(n), 0.0), aty(sz(n), 0.0), ax(sz(m), 0.0);
    const auto& arp = lp.A.pattern.row_ptr();
    const auto& aci = lp.A.pattern.col_idx();
    for (core::Index i = 0; i < m; ++i)
        for (auto t = arp[sz(i)]; t < arp[sz(i) + 1]; ++t) {
            ax[sz(i)] += lp.A.vals[sz(t)] * x[sz(aci[sz(t)])];
            aty[sz(aci[sz(t)])] += lp.A.vals[sz(t)] * y[sz(i)];
        }
    const bool sq = has_sparse_q(p);
    if (sq) {
        const auto& qrp = p.q_matrix.pattern.row_ptr();
        const auto& qci = p.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n; ++i)
            for (auto t = qrp[sz(i)]; t < qrp[sz(i) + 1]; ++t)
                qx[sz(i)] += p.q_matrix.vals[sz(t)] * x[sz(qci[sz(t)])];
    } else {
        for (core::Index j = 0; j < n; ++j) qx[sz(j)] = p.q_diag[sz(j)] * x[sz(j)];
    }

    // ---- 1. active-set guess ----
    //
    // The primal-dual indicator (El-Bakry, Tapia & Zhang, "A study of
    // indicators for identifying zero variables in interior-point methods",
    // SIAM Review 36, 1994): a bound is active when its SLACK is smaller than
    // its MULTIPLIER.  Scale-free, and right much earlier in an interior-point
    // run than a sign test on the multiplier alone -- which is what OSQP uses
    // on its more accurate ADMM iterates, and which guessed wrong on a
    // 3-iteration IPM point (the independent check then rejected it, as it
    // should).
    std::vector<int> col(sz(n), -1);      // -1 free, 0 fixed at lower, 1 at upper
    std::vector<f64> xfix(sz(n), 0.0);
    for (core::Index j = 0; j < n; ++j) {
        const f64 lo = b.col_lo[sz(j)], hi = b.col_hi[sz(j)];
        if (lo == hi) { col[sz(j)] = 0; xfix[sz(j)] = lo; continue; }
        const f64 r = qx[sz(j)] + lp.c[sz(j)] + aty[sz(j)];   // >0 pushes to lower
        if (std::isfinite(lo) && r > 0.0 && x[sz(j)] - lo < r) {
            col[sz(j)] = 0; xfix[sz(j)] = lo;
        } else if (std::isfinite(hi) && r < 0.0 && hi - x[sz(j)] < -r) {
            col[sz(j)] = 1; xfix[sz(j)] = hi;
        }
    }
    std::vector<int> row(sz(m), -1);      // -1 inactive, 0 at lower, 1 at upper
    std::vector<f64> rhs_row(sz(m), 0.0);
    for (core::Index i = 0; i < m; ++i) {
        const f64 lo = b.row_lo[sz(i)], hi = b.row_hi[sz(i)];
        if (lo == hi) { row[sz(i)] = 0; rhs_row[sz(i)] = lo; continue; }
        const f64 yi = y[sz(i)];                              // >0 pushes to upper
        if (std::isfinite(hi) && yi > 0.0 && hi - ax[sz(i)] < yi) {
            row[sz(i)] = 1; rhs_row[sz(i)] = hi;
        } else if (std::isfinite(lo) && yi < 0.0 && ax[sz(i)] - lo < -yi) {
            row[sz(i)] = 0; rhs_row[sz(i)] = lo;
        }
    }

    // ---- 2-4. one reduced solve for the current guess (col, row) ----
    std::vector<f64> xp, yp;
    // Proximal center of the reduced solve (see below): the incoming point.
    std::vector<f64> xc = x, yc = y;
    OriginalKkt kres;
    // Set by each attempt: the reduced row system had no solution for the
    // current guess (see the degenerate-column release in the rounds).
    bool rows_inconsistent = false;
    // 1 = accepted, 0 = solved but rejected, -1 = could not solve.
    auto attempt = [&](int round) -> int {
        // ---- 2-3. the reduced equality-constrained system ----
        std::vector<core::Index> fcol, arow;              // free columns, active rows
        std::vector<core::Index> fpos(sz(n), -1);
        for (core::Index j = 0; j < n; ++j)
            if (col[sz(j)] < 0) { fpos[sz(j)] = static_cast<core::Index>(fcol.size()); fcol.push_back(j); }
        for (core::Index i = 0; i < m; ++i)
            if (row[sz(i)] >= 0) arow.push_back(i);
        const core::Index nf = static_cast<core::Index>(fcol.size());
        const core::Index na = static_cast<core::Index>(arow.size());
        const core::Index N = nf + na;
        auto give_up = [&](const char* why) {
            if (opts.verbose) std::fprintf(stderr, "polish: %s\n", why);
            return -1;
        };
        if (N == 0) return give_up("empty reduced system");

        // Upper-triangle columns of K.  Column k < nf: Q_FF upper + diagonal.
        // Column nf + a: active row a's entries on free columns + diagonal.
        // delta = 1e-9 was too small: on QPLIB_9002 (Q diagonal ~1e-11, so delta
        // dominates Q) the 1/delta fill made terms ~1e9 whose rounding (~1e-7)
        // swamped delta itself -- 95 pivots came out with the wrong sign and the
        // factorization overflowed.  1e-8 is the interior point's own level, and
        // FGMRES removes it from the answer.
        const f64 delta = 1e-8;
        la::SymCsc K;
        K.n = N;
        K.col_ptr.assign(sz(N) + 1, 0);
        std::vector<std::vector<std::pair<core::Index, f64>>> cols(sz(N));
        if (sq) {
            const auto& qrp = p.q_matrix.pattern.row_ptr();
            const auto& qci = p.q_matrix.pattern.col_idx();
            for (core::Index i = 0; i < n; ++i) {
                if (fpos[sz(i)] < 0) continue;
                for (auto t = qrp[sz(i)]; t < qrp[sz(i) + 1]; ++t) {
                    const core::Index j = qci[sz(t)];
                    if (fpos[sz(j)] < 0) continue;
                    const core::Index a = fpos[sz(i)], c = fpos[sz(j)];
                    if (a <= c) cols[sz(c)].push_back({a, p.q_matrix.vals[sz(t)]});
                }
            }
        } else {
            for (core::Index k = 0; k < nf; ++k)
                cols[sz(k)].push_back({k, p.q_diag[sz(fcol[sz(k)])]});
        }
        for (core::Index a = 0; a < na; ++a) {
            const core::Index i = arow[sz(a)];
            for (auto t = arp[sz(i)]; t < arp[sz(i) + 1]; ++t) {
                const core::Index j = aci[sz(t)];
                if (fpos[sz(j)] >= 0) cols[sz(nf + a)].push_back({fpos[sz(j)], lp.A.vals[sz(t)]});
            }
        }
        // Diagonals: +delta on the Q block, -delta on the row block.
        std::vector<f64> diag_true(sz(N), 0.0);
        for (core::Index k = 0; k < N; ++k) {
            bool has = false;
            for (auto& e : cols[sz(k)])
                if (e.first == k) { diag_true[sz(k)] = e.second; has = true; }
            if (!has) cols[sz(k)].push_back({k, 0.0});
        }
        std::vector<std::int8_t> sign(sz(N), 1);
        for (core::Index a = 0; a < na; ++a) sign[sz(nf + a)] = -1;
        // The row block's regularisation scales with its Schur complement
        // -C D^-1 C' (D = diag of the Q block).  With linearly dependent active
        // rows that complement is singular in some directions; its entries can
        // be ~1/delta (4e8 on QPLIB_9002), and a fixed -delta = -1e-8 is below
        // their rounding, so pivots came out with the wrong sign.  FGMRES on
        // the unregularised system removes whatever is added here.
        // Per row, so one row's large complement does not over-regularise
        // the rest (10034's zero-curvature columns gave one max of 4.8e9).
        //
        // HOW MUCH.  What the pivot signs need is a regularisation above the
        // ROUNDING of the Schur diagonal, ~u * nnz_row * acc = 1e-14 acc.  The
        // first rule here was delta * acc = 1e-8 acc -- 1e6 times that, and
        // it broke the preconditioner: FGMRES's M^-1 K has an eigenvalue
        // S_ww / (S_ww + w'D w) along every direction w where the true Schur
        // complement S is small, and after the degenerate-column release on
        // QPLIB_8500 (below) S_ww = (A_j' w)^2 / delta = 1.6e-6 against a
        // regularisation of 2.2 -- a 1e-6 outlier that 4 x GMRES(30) never
        // resolved (row residual 2.5e-9 -> 7.3e-6, x_j at 113.8 of [0, 10],
        // 123 columns outside their box; reproduced in scipy with the same
        // matrix).  With 1e-12 acc (regularisation 2e-4 along w) GMRES(30)
        // converges in 15 iterations to 8.9e-15; 1e-10 acc needs all 30.
        // 1e-12 keeps a 100x margin over the rounding on 9002 (acc 4e8 ->
        // 4e-4 against 4e-6 of rounding) and 10034 (4.8e9 -> 4.8e-3).
        std::vector<f64> drow(sz(na), delta);
        f64 delta_row = delta;
        for (core::Index a = 0; a < na; ++a) {
            f64 acc = 0.0;
            for (auto& e : cols[sz(nf + a)])
                if (e.first < nf) acc += e.second * e.second / (std::max(diag_true[sz(e.first)], 0.0) + delta);
            drow[sz(a)] = std::max(delta, 1e-12 * acc);
            delta_row = std::max(delta_row, drow[sz(a)]);
        }
        for (core::Index k = 0; k < N; ++k) {
            for (auto& e : cols[sz(k)]) {
                K.row_idx.push_back(e.first);
                K.vals.push_back(e.first == k ? e.second + (k < nf ? delta : -drow[sz(k - nf)]) : e.second);
            }
            K.col_ptr[sz(k) + 1] = static_cast<la::Offset>(K.row_idx.size());
        }

        // Right-hand side, with the fixed columns moved across.
        std::vector<f64> rhs(sz(N), 0.0);
        for (core::Index k = 0; k < nf; ++k) rhs[sz(k)] = -lp.c[sz(fcol[sz(k)])];
        std::vector<f64> xb(sz(n), 0.0);
        for (core::Index j = 0; j < n; ++j) if (col[sz(j)] >= 0) xb[sz(j)] = xfix[sz(j)];
        {
            std::vector<f64> qxb(sz(n), 0.0);
            if (sq) {
                const auto& qrp = p.q_matrix.pattern.row_ptr();
                const auto& qci = p.q_matrix.pattern.col_idx();
                for (core::Index i = 0; i < n; ++i)
                    for (auto t = qrp[sz(i)]; t < qrp[sz(i) + 1]; ++t)
                        qxb[sz(i)] += p.q_matrix.vals[sz(t)] * xb[sz(qci[sz(t)])];
            } else {
                for (core::Index j = 0; j < n; ++j) qxb[sz(j)] = p.q_diag[sz(j)] * xb[sz(j)];
            }
            for (core::Index k = 0; k < nf; ++k) rhs[sz(k)] -= qxb[sz(fcol[sz(k)])];
            for (core::Index a = 0; a < na; ++a) {
                const core::Index i = arow[sz(a)];
                f64 s = rhs_row[sz(i)];
                for (auto t = arp[sz(i)]; t < arp[sz(i) + 1]; ++t) s -= lp.A.vals[sz(t)] * xb[sz(aci[sz(t)])];
                rhs[sz(nf + a)] = s;
            }
        }

        la::Ldlt fac;
        try {
            fac.analyze(K);
        } catch (const std::exception&) {
            return give_up("analyze failed");
        }
        if (!fac.factorize(K, sign, 1e-12)) return give_up("non-finite pivot in the reduced system");

        // Solve the delta = 0 system with the regularised factor as the
        // preconditioner (the paper refines; FGMRES converges where refinement
        // does not -- see qp_krylov.cpp).
        auto mul_true = [&](const std::vector<f64>& v, std::vector<f64>& o) {
            o.assign(sz(N), 0.0);
            for (core::Index c = 0; c < N; ++c)
                for (auto t = K.col_ptr[sz(c)]; t < K.col_ptr[sz(c) + 1]; ++t) {
                    const core::Index r = K.row_idx[sz(t)];
                    const f64 val = (r == c) ? diag_true[sz(c)] : K.vals[sz(t)];
                    o[sz(r)] += val * v[sz(c)];
                    if (r != c) o[sz(c)] += val * v[sz(r)];
                }
        };
        // PROXIMAL CENTER.  With R = diag(delta I, D_row) the factored matrix is
        // K + R (up to the row block's sign), so one factor solve from z_k is
        // one step of the proximal point method on the KKT operator
        // (Rockafellar, "Monotone operators and the proximal point algorithm",
        // SIAM J. Control Optim. 14(5), 1976; for regularised QP KKT systems,
        // the proximal method of multipliers of Friedlander & Orban, Math.
        // Prog. Comp. 4, 2012): it converges to a solution whenever one
        // exists, for ANY positive R -- singular K included.  When K is
        // singular (linearly dependent active rows; free columns with no
        // curvature inside null(C_F), as on QPLIB_8500 / _10034) the solution
        // is not unique, and which one comes out is decided by the start:
        // for consistent K z = rhs, every Krylov correction M^-1 A^j r0 is
        // R-orthogonal to null(K) (for w in null(K), M^-T R w = w and
        // w' r = 0 for r in range(K)), so the iterate keeps the null-space
        // component of its start.  The old start z = M^-1 rhs was the
        // solution nearest ZERO in that metric: on 10034 its x_F left the box
        // by 2.3 and its multipliers lost the signs the fixed columns need.
        // Starting from the incoming point instead returns the solution
        // nearest the interior-point iterate -- inside the box, multipliers
        // of dependent rows split as the IPM had them -- so dependent rows
        // need no detection and no dropping.
        std::vector<f64> sol(sz(N), 0.0);
        for (core::Index k = 0; k < nf; ++k) {
            const core::Index j = fcol[sz(k)];
            sol[sz(k)] = std::min(std::max(xc[sz(j)], b.col_lo[sz(j)]), b.col_hi[sz(j)]);
        }
        for (core::Index a = 0; a < na; ++a) sol[sz(nf + a)] = yc[sz(arow[sz(a)])];
        f64 bn = 0.0;
        for (f64 v : rhs) bn = std::max(bn, std::fabs(v));
        auto true_resid = [&](const std::vector<f64>& z, std::vector<f64>& r) {
            mul_true(z, r);
            f64 rn = 0.0;
            for (core::Index c = 0; c < N; ++c) {
                r[sz(c)] = rhs[sz(c)] - r[sz(c)];
                rn = std::max(rn, std::fabs(r[sz(c)]));
            }
            return rn;
        };
        std::vector<f64> res;
        f64 rnorm = true_resid(sol, res);
        // Restarted FGMRES on the correction; a restart from the current
        // iterate keeps the proximal property above.
        for (int pass = 0; pass < 4 && rnorm > 1e-14 * (1.0 + bn); ++pass) {
            std::vector<f64> d = res;
            fgmres_solve(fac, mul_true, d, 30, 1e-14);
            std::vector<f64> trial(sz(N)), rt;
            for (core::Index c = 0; c < N; ++c) trial[sz(c)] = sol[sz(c)] + d[sz(c)];
            const f64 rn = true_resid(trial, rt);
            if (opts.verbose) {
                f64 rx = 0, ry = 0, r0x = 0, r0y = 0;
                for (core::Index c = 0; c < N; ++c) {
                    (c < nf ? rx : ry) = std::max(c < nf ? rx : ry, std::fabs(rt[sz(c)]));
                    (c < nf ? r0x : r0y) = std::max(c < nf ? r0x : r0y, std::fabs(res[sz(c)]));
                }
                std::fprintf(stderr, "polish pass %d: resid %.2e (x %.2e row %.2e) -> %.2e (x %.2e row %.2e) bn %.2e\n",
                             pass, rnorm, r0x, r0y, rn, rx, ry, bn);
            }
            if (!(rn < 0.5 * rnorm)) {
                if (rn < rnorm) { sol.swap(trial); res.swap(rt); rnorm = rn; }
                break;
            }
            sol.swap(trial);
            res.swap(rt);
            rnorm = rn;
        }
        const f64 solve_rel = rnorm / (1.0 + bn);
        for (core::Index c = 0; c < N; ++c)
            if (!std::isfinite(sol[sz(c)])) return give_up("non-finite solution");
        // A row-block residual FGMRES could not remove (the loop above stops
        // at 1e-14 (1 + |rhs|) when it can) means the active rows are not
        // satisfiable with the fixed columns where the guess pinned them;
        // the rounds below read this.
        {
            f64 rrow = 0.0;
            for (core::Index a = 0; a < na; ++a) rrow = std::max(rrow, std::fabs(res[sz(nf + a)]));
            rows_inconsistent = rrow > 1e-12 * (1.0 + bn);
        }

        // ---- 4. assemble and hand to the independent check ----
        xp.assign(sz(n), 0.0);
        yp.assign(sz(m), 0.0);
        for (core::Index j = 0; j < n; ++j)
            xp[sz(j)] = col[sz(j)] >= 0 ? xfix[sz(j)] : sol[sz(fpos[sz(j)])];
        // K's row block reads C x_F - delta*lam = b, and the column block
        // Q x + c + C' lam = 0 on free columns: lam is already PDHCG's y sign.
        for (core::Index a = 0; a < na; ++a) yp[sz(arow[sz(a)])] = sol[sz(nf + a)];

        auto check = [&](const std::vector<f64>& xv, const std::vector<f64>& yv, OriginalKkt& k) {
            if (judge) return (*judge)(xv, yv, k);
            k = kkt_original(p, b, xv, yv);
            return kkt_ok(k, opts);
        };
        bool pass = check(xp, yp, kres);

        // ---- 4b. the multiplier freedom the dependent active rows leave ----
        //
        // WHY THIS EXISTS.  On QPLIB_8500 every one of the 250498 rows is an
        // active equality and C_F (the active rows on the free columns) has a
        // left null vector w: measured |C_F' w|_inf = 2.8e-17 with |w|_2 = 1,
        // spread over 250495 of the rows.  Two consequences.
        //
        // (a) The reduced row system C_F x_F = b_act - C_B x_B is solvable only
        //     if w'(b_act - C_B x_B) = 0.  The fixed columns are AT their bounds,
        //     w'b = -1.05e-3 and w'A_B is nonzero on all 500 of them, so that
        //     scalar is only as small as the interior point made it: the row
        //     residual keeps an irreducible component s*w, and FGMRES rightly
        //     refuses to move it (1.29e-9 -> 1.37e-9 on 8500).  Least squares --
        //     which is what GMRES returns on a singular symmetric system, since
        //     symmetry gives range-symmetry and the residual then stagnates in
        //     null(K); Brown & Walker, "GMRES on (nearly) singular systems",
        //     SIAM J. Matrix Anal. Appl. 18(1), 1997 -- spreads that scalar over
        //     all the rows, which is the best a max-norm feasibility check can
        //     get: dropping dependent rows instead would concentrate the same
        //     inconsistency into the dropped ones (s/w_i ~ 3e-6 on 8500).
        //
        // (b) The multiplier is not determined: y + t*w satisfies the same
        //     stationarity on the free columns for every t, because C_F' w = 0.
        //
        // And the duality gap this leaves is exactly
        //     gap = sum_j (bnd_j - x_j) z_j + sum_i (bnd_i - a_i) y_i,
        // whose second term is <s*w, y>.  So the SAME degeneracy that forbids
        // (a) hands us (b) to cancel it with: pick t so the residual we cannot
        // remove is orthogonal to the multipliers, and the gap goes with it.
        // 8500's gap is 2.3e-7 = s * (w'y) and nothing else -- its primal and
        // stationarity residuals are already 2.5e-9.
        //
        // The direction is free: the leftover residual IS the null direction
        // (that is what stagnation means), so no null-space basis is computed.
        // gap(t) is affine in t while no fixed column's multiplier changes sign,
        // so one probe gives the slope and one step lands on the root.  Nothing
        // here is trusted: every candidate goes through the same independent
        // original-units check, which is also what rejects a sign flip.
        if (!pass && kres.gap_finite && kres.primal_net <= opts.feas_tol &&
            kres.dual_net <= opts.stationarity_tol && kres.gap_net > opts.gap_tol) {
            std::vector<f64> dy(sz(m), 0.0);
            f64 dmax = 0.0;
            for (core::Index a = 0; a < na; ++a) dmax = std::max(dmax, std::fabs(res[sz(nf + a)]));
            if (dmax > 0.0) {
                for (core::Index a = 0; a < na; ++a) dy[sz(arow[sz(a)])] = res[sz(nf + a)] / dmax;
                // How far the direction is from null(C_F'), for the record: the
                // free columns' stationarity moves by t * C_F' dy.
                f64 cross = 0.0;
                {
                    std::vector<f64> e(sz(N), 0.0), ke;
                    for (core::Index a = 0; a < na; ++a) e[sz(nf + a)] = dy[sz(arow[sz(a)])];
                    mul_true(e, ke);
                    for (core::Index k = 0; k < nf; ++k) cross = std::max(cross, std::fabs(ke[sz(k)]));
                }
                // The signed gap numerator: objective - dual_bound, both of
                // which carry the same objective offset, so it cancels.
                auto num_of = [](const OriginalKkt& k) { return k.objective - k.dual_bound; };
                const f64 num0 = num_of(kres);
                std::vector<f64> yt(sz(m));
                OriginalKkt kt;
                auto at = [&](f64 t) {
                    for (core::Index i = 0; i < m; ++i) yt[sz(i)] = yp[sz(i)] + t * dy[sz(i)];
                    return check(xp, yt, kt);
                };
                // Probe: the smallest decade that moves the gap measurably.
                // t is not dimensionless (dy is a residual, y a multiplier), so
                // its scale is found rather than guessed; each probe is one
                // sparse pass over the model.
                f64 t = 1e-14, slope = 0.0;
                bool probed = false;
                for (int k = 0; k < 30 && !probed; ++k, t *= 10.0) {
                    if (at(t)) { yp.swap(yt); kres = kt; pass = true; break; }
                    if (std::fabs(num_of(kt) - num0) > 0.25 * std::fabs(num0)) {
                        slope = (num_of(kt) - num0) / t;
                        probed = true;
                    }
                }
                // Secant on the affine model; re-solved rather than trusted,
                // and retried twice in case a multiplier did change sign.
                for (int k = 0; k < 3 && !pass && probed && slope != 0.0; ++k) {
                    const f64 step = -num0 / slope;
                    if (!std::isfinite(step)) break;
                    if (at(step)) { yp.swap(yt); kres = kt; pass = true; break; }
                    const f64 n1 = num_of(kt);
                    if (!std::isfinite(n1) || n1 == num0) break;
                    slope = (n1 - num0) / step;
                }
                if (opts.verbose)
                    std::fprintf(stderr, "polish round %d: dependent active rows -- gap %.2e, "
                                 "null-direction |C_F' d| %.1e, multiplier shift %s\n",
                                 round, kres.gap, cross, pass ? "accepted" : "did not close it");
            }
        }
        if (opts.verbose)
            std::fprintf(stderr, "polish round %d: free %lld of %lld cols, active %lld of %lld rows; "
                         "solve residual %.1e, max delta_row %.1e; primal %.2e stationarity %.2e gap %.2e -> %s\n",
                         round, static_cast<long long>(nf), static_cast<long long>(n),
                         static_cast<long long>(na), static_cast<long long>(m), solve_rel, delta_row,
                         kres.primal, kres.dual_res, kres.gap_finite ? kres.gap : -1.0,
                         pass ? "accepted" : "rejected");
        return pass ? 1 : 0;
    };

    // ---- 5. primal-dual active-set rounds ----
    //
    // One polish is one step of the primal-dual active set method
    // (Hintermueller, Ito & Kunisch, "The primal-dual active set strategy as
    // a semismooth Newton method", SIAM J. Optim. 13(3), 2002).  When the
    // guess is wrong, the polished point says how: a free variable that
    // left its box belongs at that bound, and a fixed one whose multiplier
    // has the wrong sign should be released.  Re-solving with that update
    // is a semismooth Newton step -- locally superlinear -- and is what an
    // interior point stuck short of a moving front needs (QPLIB_10034).
    // Every round is still judged only by the original-units check.
    constexpr int kRounds = 10;
    // Columns the degenerate-column rule below has already released once;
    // a column it released that then left its box is re-fixed by the box
    // rule and must not be released again, or the rounds cycle.
    std::vector<char> tried(sz(n), 0);
    int stuck_rounds = 0;
    for (int round = 0; round < kRounds; ++round) {
        const int got = attempt(round);
        if (got < 0) return false;
        if (got == 1) {
            x.swap(xp);
            y.swap(yp);
            out = kres;
            return true;
        }
        std::vector<f64> qx2(sz(n), 0.0), aty2(sz(n), 0.0), ax2(sz(m), 0.0);
        for (core::Index i = 0; i < m; ++i)
            for (auto t = arp[sz(i)]; t < arp[sz(i) + 1]; ++t) {
                ax2[sz(i)] += lp.A.vals[sz(t)] * xp[sz(aci[sz(t)])];
                aty2[sz(aci[sz(t)])] += lp.A.vals[sz(t)] * yp[sz(i)];
            }
        if (sq) {
            const auto& qrp = p.q_matrix.pattern.row_ptr();
            const auto& qci = p.q_matrix.pattern.col_idx();
            for (core::Index i = 0; i < n; ++i)
                for (auto t = qrp[sz(i)]; t < qrp[sz(i) + 1]; ++t)
                    qx2[sz(i)] += p.q_matrix.vals[sz(t)] * xp[sz(qci[sz(t)])];
        } else {
            for (core::Index j = 0; j < n; ++j) qx2[sz(j)] = p.q_diag[sz(j)] * xp[sz(j)];
        }
        const f64 ft = opts.feas_tol, st = opts.stationarity_tol;
        // Feasibility restoration for the fixed columns.  The wrong-sign test
        // below assumes the guess admits a point at all; it does not always.
        // With the columns of a row all fixed, the reduced row block is EMPTY
        // and its equation unsatisfiable -- then the row's multiplier is not
        // determined by anything, the sign test reads whatever the solve
        // happened to leave (the proximal centre leaves the incoming y, which
        // is usually right-signed), and the rounds stop with nothing changed.
        // The honest signal there is the residual: with r_i = b_i - (A x)_i on
        // the active rows, d(0.5|r|^2)/dx_j = -sum_i A_ij r_i, so a column at
        // its LOWER bound can reduce the infeasibility exactly when that sum
        // is positive, and one at its upper bound when it is negative.  This
        // is the same step a phase-1 method would take, restricted to the
        // columns the guess pinned.  Only when the active rows really cannot
        // be met (rres > feas_tol) and the column really moves them.
        std::vector<f64> grad(sz(n), 0.0);
        f64 rres = 0.0;
        for (core::Index i = 0; i < m; ++i) {
            if (row[sz(i)] < 0) continue;
            const f64 ri = rhs_row[sz(i)] - ax2[sz(i)];
            rres = std::max(rres, std::fabs(ri));
            for (auto t = arp[sz(i)]; t < arp[sz(i) + 1]; ++t)
                grad[sz(aci[sz(t)])] += lp.A.vals[sz(t)] * ri;
        }
        bool changed = false;
        for (core::Index j = 0; j < n; ++j) {
            const f64 lo = b.col_lo[sz(j)], hi = b.col_hi[sz(j)];
            if (lo == hi) continue;
            const f64 r = qx2[sz(j)] + lp.c[sz(j)] + aty2[sz(j)];   // >0 pushes to lower
            const f64 g = grad[sz(j)];
            int& cj = col[sz(j)];
            if (cj < 0) {
                if (std::isfinite(lo) && xp[sz(j)] < lo - ft) { cj = 0; xfix[sz(j)] = lo; changed = true; }
                else if (std::isfinite(hi) && xp[sz(j)] > hi + ft) { cj = 1; xfix[sz(j)] = hi; changed = true; }
            } else if ((cj == 0 && r < -st) || (cj == 1 && r > st) ||
                       (rres > ft && std::fabs(g) > ft && ((cj == 0 && g > 0.0) || (cj == 1 && g < 0.0)))) {
                cj = -1;
                changed = true;
            }
        }
        for (core::Index i = 0; i < m; ++i) {
            const f64 lo = b.row_lo[sz(i)], hi = b.row_hi[sz(i)];
            if (lo == hi) continue;
            int& ri = row[sz(i)];
            const f64 yi = yp[sz(i)];                                 // >0 pushes to upper
            if (ri < 0) {
                if (std::isfinite(hi) && ax2[sz(i)] > hi + ft) { ri = 1; rhs_row[sz(i)] = hi; changed = true; }
                else if (std::isfinite(lo) && ax2[sz(i)] < lo - ft) { ri = 0; rhs_row[sz(i)] = lo; changed = true; }
            } else if ((ri == 1 && yi < -st) || (ri == 0 && yi > st)) {
                ri = -1;
                changed = true;
            }
        }
        // DEGENERATE COLUMNS.  The row system can be unsatisfiable with every
        // fixed multiplier right-signed and the residual under feas_tol, and
        // then nothing above moves -- QPLIB_8500: 250498 active equality rows
        // on 250497 free columns, the reduced residual stagnates at the left
        // null vector w of C_F (|C_F' w| = 3e-17), primal and stationarity
        // are 2.5e-9 and the gap <r, y> = 2.4e-7 is what fails.  The point is
        // NOT optimal: the true solution has some pinned column strictly
        // inside its box (measured: column 155120 at 8.989 of [0, 10], the
        // objective 2.4e-7 lower).  Which one?  The one the primal-dual
        // indicator could not classify.  Hintermueller, Ito & Kunisch's
        // active set is {j : lambda_j + c (x_j - bound_j) has the active
        // sign}; at a degenerate bound (x_j at its bound AND lambda_j ~ 0)
        // that test reads rounding, and the IPM's own complementarity
        // (x_j - l_j) z_j = 2e-23 says the same.  Such a column costs nothing
        // to release (its multiplier is the first-order price of moving it),
        // so it is the cheapest way to make the rows satisfiable; the
        // strongly coupled columns are NOT the answer -- on 8500 releasing
        // the column with the largest |A_j' w| (multiplier 0.115) sends it
        // outward by 3.7e-5.  So: released in order of increasing |z_j|,
        // 2^k at the k-th inconsistent round (one is what 8500 needs; a
        // guess that pinned many degenerate columns gets them all within the
        // round budget), each at most once (a release that leaves the box is
        // re-fixed above and stays fixed).  Only while the row system is
        // inconsistent; the released column's own direction is not guessed
        // from the residual gradient, because on the weakly coupled columns
        // that gradient is below FGMRES's own noise (|C_F' d| ~ 5e-6 of the
        // residual against couplings of 1e-7 on 8500) -- the exact re-solve
        // decides it, and the original-units check judges the result.
        if (rows_inconsistent) {
            std::vector<std::pair<f64, core::Index>> cand;
            for (core::Index j = 0; j < n; ++j) {
                if (col[sz(j)] < 0 || tried[sz(j)] || b.col_lo[sz(j)] == b.col_hi[sz(j)]) continue;
                cand.push_back({std::fabs(qx2[sz(j)] + lp.c[sz(j)] + aty2[sz(j)]), j});
            }
            const std::size_t take = std::min(cand.size(), std::size_t{1} << std::min(stuck_rounds, 20));
            std::partial_sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(take), cand.end());
            for (std::size_t k = 0; k < take; ++k) {
                const core::Index j = cand[k].second;
                col[sz(j)] = -1;
                tried[sz(j)] = 1;
                changed = true;
                if (opts.verbose)
                    std::fprintf(stderr, "polish round %d: rows inconsistent -- releasing degenerate column %lld (|z| %.2e)\n",
                                 round, static_cast<long long>(j), cand[k].first);
            }
            ++stuck_rounds;
        }
        if (!changed) return false;
    }
    return false;
}

bool polish_scaled(const QpProblem& p, const backend::LaneBounds& b, const Scaling& sc,
                   const QpProblem& ps, const backend::LaneBounds& sb, const QpOptions& opts,
                   std::vector<f64>& x, std::vector<f64>& y, OriginalKkt& out) {
    // x = Dc x~ and y = Dr y~; the scaled stationarity residual is Dc times
    // the original, so every sign the active-set rules read is preserved.
    std::vector<f64> xs(x.size()), ys(y.size());
    for (std::size_t j = 0; j < x.size(); ++j) xs[j] = x[j] / sc.dc[j];
    for (std::size_t i = 0; i < y.size(); ++i) ys[i] = y[i] / sc.dr[i];
    auto unscale = [&](const std::vector<f64>& xv, const std::vector<f64>& yv,
                       std::vector<f64>& xo, std::vector<f64>& yo) {
        xo.resize(xv.size());
        yo.resize(yv.size());
        for (std::size_t j = 0; j < xv.size(); ++j) xo[j] = xv[j] * sc.dc[j];
        for (std::size_t i = 0; i < yv.size(); ++i) yo[i] = yv[i] * sc.dr[i];
    };
    const PolishJudge judge = [&](const std::vector<f64>& xv, const std::vector<f64>& yv,
                                  OriginalKkt& k) {
        std::vector<f64> xo, yo;
        unscale(xv, yv, xo, yo);
        k = kkt_original(p, b, xo, yo);
        return kkt_ok(k, opts);
    };
    if (!polish(ps, sb, opts, xs, ys, out, &judge)) return false;
    unscale(xs, ys, x, y);
    return true;
}

}  // namespace qpc

}  // namespace sor::engines
