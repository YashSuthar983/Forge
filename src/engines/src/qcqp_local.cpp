// SOR — local solver for general (nonconvex) QCQP.  See qcqp.hpp.
//
// METHOD.  The barrier primal-dual interior point of the filter line-search method (Waechter &
// Biegler, "On the implementation of an interior-point filter line-search
// algorithm for large-scale nonlinear programming", Math. Program. 106,
// 2006), on the same presolved, scaled problem the convex IPM runs on
// (ipm_core.hpp) and with the same variables and sign conventions as
// qp_ipm.cpp's run_ipm:
//     min f(x) = 1/2 x'Q0 x + c'x   s.t.  g(x) - w = 0,  l <= x <= u,  rl <= w <= ru,
//     g_i(x) = a_i'x + 1/2 x'Q_i x,
// every finite bound with a log barrier, multipliers lam (lower), nu (upper),
// y (rows; stationarity  grad f - J'y - lam_x + nu_x = 0,  y - lam_w + nu_w = 0).
//
// What differs from the convex IPM, and why:
//   * The Hessian is the TRUE Lagrangian Hessian Q0 - sum_i y_i Q_i, of any
//     inertia.  INERTIA CORRECTION (Waechter & Biegler s.3.1): delta_w I is
//     added to it until the reduced KKT system is quasi-definite -- detected
//     exactly as the LDL' reports it: with the x pivots expected negative
//     and the y pivots positive, any pivot it had to regularise means the
//     (1,1) block was not negative definite in that elimination order.
//     delta_w starts at 0, then from 1e-4 (or a third of the last value) and
//     grows x8 (x100 the first time), the paper's schedule.
//   * GLOBALISATION: a backtracking line search on the l1 merit function
//     phi = f - mu sum log(slacks) + nu_pen ||g(x) - w||_1 with an Armijo
//     test (Nocedal & Wright, Numerical Optimization, 2nd ed., s.19.4 and
//     Thm 18.2 for the penalty rule nu_pen > ||y||_inf).  The paper uses a
//     filter instead; the merit version is the simpler of the two its paper
//     compares, and is what LOQO uses (Vanderbei & Shanno, Comput. Optim.
//     Appl. 13, 1999).
//   * The barrier parameter is MONOTONE (Fiacco-McCormick; Waechter & Biegler s.2.1):
//     solve each barrier problem to kappa_eps * mu, then
//     mu <- max(tol/10, min(kappa_mu mu, mu^theta_mu)).  Mehrotra's
//     predictor-corrector has no descent guarantee away from convexity.
//   * The primal step (x, w, y) and the dual step (lam, nu) have separate
//     fraction-to-boundary lengths, as in the paper.
//   * RESTORATION PHASE (Waechter & Biegler s.3.3, LocalIpm::run_restoration):
//     when the main direction stalls (line search fails repeatedly, inertia
//     correction can't be fixed, or feasibility progress plateaus),
//     `maybe_restart` tries a subproblem solve from the current point --
//     minimise the l1 constraint violation via per-row p/n slacks plus a
//     proximity term to the current point -- before falling back to a
//     random jump.  Built from the restoration-phase formulation itself
//     (Waechter & Biegler 2006 s.3.3: subproblem eq. 29a/31a, exit test
//     eq. 18a/18b), not just the paper's prose description.
// It finds LOCAL solutions.  Nothing here proves anything: the result is
// judged only by evaluate_qcqp on the ORIGINAL model (feasibility within
// feas_tol), and reported as Status::Feasible at best.
#include "sor/engines/dual_simplex.hpp"
#include "sor/engines/qcqp.hpp"
#include "sor/la/ldlt.hpp"
#include "sor/model/lp.hpp"
#include "sor/sparse/csr.hpp"

#include "ipm_core.hpp"
#include "qp_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace sor::engines {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
using qpc::sz;
using ipmc::Core;
constexpr f64 kInf = std::numeric_limits<f64>::infinity();
constexpr f64 kUlp = 2.220446049250313e-16;

// PURE-LINEAR EQUALITY ROWS: rows with no quadratic term at all whose
// range is a single point (rl == ru).  On several LCQ instances close to
// half of every row is exactly this shape (QPLIB_2703: 399 of 800), and a
// box-uniform random start satisfies essentially none of them by chance.
// Precomputed once per Core (both `k` and its feasibility-only twin `kf`
// share the same A/rows, so one list serves both engines).
std::vector<core::Index> linear_eq_rows(const Core& k) {
    std::vector<std::uint8_t> has_quad(sz(k.m), 0);
    for (const auto& q : k.quad) has_quad[sz(q.row)] = 1;
    std::vector<core::Index> rows;
    for (core::Index i = 0; i < k.m; ++i)
        if (!has_quad[sz(i)] && std::isfinite(k.rl[sz(i)]) && k.rl[sz(i)] == k.ru[sz(i)])
            rows.push_back(i);
    return rows;
}

// LCQ far bucket (QPLIB_2819/2834/3089/3120/3147/3225/3338 and
// friends, 80-90% quadratic rows by count): the Kaczmarz projection above
// only ever touches PURE-LINEAR equality rows, which these instances barely
// have, so every start is still effectively a blind uniform-random guess on
// the rows that actually matter.  This builds a McCormick (1976) /
// Al-Khayyal & Falk (1983) bilinear-envelope LP relaxation and solves it
// with our own dual simplex, giving a structurally-informed point instead.
// It is the SAME recipe search/global_qp.cpp's McCormickLp uses for its
// spatial branch-and-bound's root LP bound (four envelope inequalities per
// bilinear product, secant + one tangent for a square), generalised from
// "linearise the objective's quadratic part" to "linearise EVERY quadratic
// row's part, in that row" -- global_qp.cpp only ever had an objective
// Hessian to deal with (engines::QpProblem), never a QCQP's per-row Q_i
// (Core::Quad here), so its class was not reusable as-is; the envelope math
// itself is copied verbatim (not re-derived) from that file's `update()`
// (see its L1/L2/U1/U2/Secant/TanMid comments for the algebra).
//
// This is a RELAXATION, not a heuristic guess dressed up: any point x with
// w_ij set to the true product x_i x_j satisfies every envelope row and
// every row bound exactly (original units, modulo the row's own tolerance),
// so the LP is never infeasible unless the instance itself has no feasible
// point at all -- unlike a random start, it uses EVERY row, linear or
// quadratic.  Only ONE tangent (at the box midpoint, not the tightest of
// three) is used for a square term x_i^2's lower envelope: cheaper to build
// and still valid, and this is a start point, not a certified bound, so the
// tighter variants global_qp.cpp needs for pruning buy nothing here.
//
// Returns an empty vector (caller falls back to the existing origin/random
// start, unchanged) when: there is nothing quadratic to relax, the product
// count is too large for a cheap one-off LP, the deadline has no time left,
// or the LP solve does not report Optimal/Feasible.
std::vector<f64> mccormick_core_start(const Core& k, Clock::time_point deadline) {
    const double budget = std::chrono::duration<double>(deadline - Clock::now()).count();
    if (!(budget > 0.0)) return {};

    std::map<std::pair<core::Index, core::Index>, core::Index> pidx;
    std::vector<std::pair<core::Index, core::Index>> prod;   // i <= j
    auto idx_of = [&](core::Index a, core::Index b) -> core::Index {
        const auto key = std::make_pair(std::min(a, b), std::max(a, b));
        auto it = pidx.find(key);
        if (it != pidx.end()) return it->second;
        const core::Index t = static_cast<core::Index>(prod.size());
        prod.push_back(key);
        pidx[key] = t;
        return t;
    };
    // row < 0: objective.  Coefficient convention matches Core::Quad /
    // QuadRow / q_mul exactly: v*x_i*x_j when i != j, 0.5*v*x_i^2 when i==j.
    struct RTerm { core::Index row, t; f64 coef; };
    std::vector<RTerm> rterms;
    for (core::Index j = 0; j < k.n; ++j)
        for (const auto& [i, v] : k.qcol[sz(j)])
            rterms.push_back({-1, idx_of(i, j), i == j ? 0.5 * v : v});
    for (const auto& q : k.quad)
        for (std::size_t t = 0; t < q.r.size(); ++t)
            rterms.push_back({q.row, idx_of(q.r[t], q.c[t]),
                              q.r[t] == q.c[t] ? 0.5 * q.v[t] : q.v[t]});
    if (prod.empty()) return {};
    const core::Index T = static_cast<core::Index>(prod.size());
    if (T > 20000) return {};   // too large for a cheap one-off LP; keep the plain start

    std::vector<core::Index> tr, tc;
    std::vector<f64> tv;
    std::vector<f64> row_lo, row_hi;
    const auto& rp = k.A.pattern.row_ptr();
    const auto& ci = k.A.pattern.col_idx();
    for (core::Index r = 0; r < k.m; ++r) {
        for (auto e = rp[sz(r)]; e < rp[sz(r) + 1]; ++e) {
            tr.push_back(r); tc.push_back(ci[sz(e)]); tv.push_back(k.A.vals[sz(e)]);
        }
        row_lo.push_back(k.rl[sz(r)]);
        row_hi.push_back(k.ru[sz(r)]);
    }
    for (const auto& rt : rterms)
        if (rt.row >= 0) { tr.push_back(rt.row); tc.push_back(k.n + rt.t); tv.push_back(rt.coef); }

    core::Index row = k.m;
    // w - alpha x_i - beta x_j (>= or <=) -gamma, with a tiny outward slack
    // so a rounding-exact boundary point is never spuriously cut off (this
    // is a start point, not a proof: unlike global_qp.cpp's bound-grade
    // charge, a fixed relative epsilon is enough here).
    const auto add_env = [&](core::Index w, core::Index i, core::Index j, f64 alpha, f64 beta,
                             f64 gamma, bool ge) {
        tr.push_back(row); tc.push_back(k.n + w); tv.push_back(1.0);
        tr.push_back(row); tc.push_back(i); tv.push_back(-alpha);
        if (j != i) { tr.push_back(row); tc.push_back(j); tv.push_back(-beta); }
        const f64 slack = 1e-9 * std::max(1.0, std::fabs(gamma));
        if (ge) { row_lo.push_back(-gamma - slack); row_hi.push_back(kInf); }
        else    { row_lo.push_back(-kInf); row_hi.push_back(-gamma + slack); }
        ++row;
    };
    std::vector<f64> plo(sz(T), -kInf), phi(sz(T), kInf);
    for (core::Index t = 0; t < T; ++t) {
        const auto [i, j] = prod[sz(t)];
        const f64 li = k.l[sz(i)], ui = k.u[sz(i)];
        if (i != j) {
            const f64 lj = k.l[sz(j)], uj = k.u[sz(j)];
            if (std::isfinite(li) && std::isfinite(ui) && std::isfinite(lj) && std::isfinite(uj)) {
                const f64 p[4] = {li * lj, li * uj, ui * lj, ui * uj};
                plo[sz(t)] = std::min({p[0], p[1], p[2], p[3]});
                phi[sz(t)] = std::max({p[0], p[1], p[2], p[3]});
            }
            if (std::isfinite(li) && std::isfinite(lj)) add_env(t, i, j, lj, li, li * lj, true);
            if (std::isfinite(ui) && std::isfinite(uj)) add_env(t, i, j, uj, ui, ui * uj, true);
            if (std::isfinite(li) && std::isfinite(uj)) add_env(t, i, j, uj, li, li * uj, false);
            if (std::isfinite(ui) && std::isfinite(lj)) add_env(t, i, j, lj, ui, ui * lj, false);
        } else if (std::isfinite(li) && std::isfinite(ui)) {
            plo[sz(t)] = (li <= 0.0 && ui >= 0.0) ? 0.0 : std::min(li * li, ui * ui);
            phi[sz(t)] = std::max(li * li, ui * ui);
            const f64 s = li + ui, g = li * ui, a0 = 0.5 * s;
            tr.push_back(row); tc.push_back(k.n + t); tv.push_back(1.0);
            tr.push_back(row); tc.push_back(i); tv.push_back(-s);
            row_lo.push_back(-kInf);
            row_hi.push_back(-g + 1e-9 * std::max(1.0, std::fabs(g))); ++row;
            tr.push_back(row); tc.push_back(k.n + t); tv.push_back(1.0);
            tr.push_back(row); tc.push_back(i); tv.push_back(-2.0 * a0);
            row_lo.push_back(-a0 * a0 - 1e-9 * std::max(1.0, a0 * a0));
            row_hi.push_back(kInf); ++row;
        } else {
            plo[sz(t)] = 0.0;
            if (std::isfinite(li) && li >= 0.0) plo[sz(t)] = li * li;
            if (std::isfinite(ui) && ui <= 0.0) plo[sz(t)] = ui * ui;
        }
    }

    model::LpProblem lp;
    lp.name = "mccormick_start";
    lp.c.assign(sz(k.n + T), 0.0);
    for (core::Index j = 0; j < k.n; ++j) lp.c[sz(j)] = k.c[sz(j)];
    for (const auto& rt : rterms)
        if (rt.row < 0) lp.c[sz(k.n) + sz(rt.t)] += rt.coef;
    lp.col_lo.assign(sz(k.n + T), -kInf);
    lp.col_hi.assign(sz(k.n + T), kInf);
    for (core::Index j = 0; j < k.n; ++j) { lp.col_lo[sz(j)] = k.l[sz(j)]; lp.col_hi[sz(j)] = k.u[sz(j)]; }
    for (core::Index t = 0; t < T; ++t) {
        lp.col_lo[sz(k.n) + sz(t)] = plo[sz(t)];
        lp.col_hi[sz(k.n) + sz(t)] = phi[sz(t)];
    }
    lp.is_integer.assign(sz(k.n + T), false);
    lp.row_lo = row_lo;
    lp.row_hi = row_hi;
    lp.A = sparse::from_triplets(row, k.n + T, tr, tc, tv);

    SimplexOptions sx;
    sx.method = SimplexMethod::Dual;
    sx.time_limit_s = std::min(3.0, budget);
    sx.max_iterations = 20000;
    SimplexDiagnostics sd;
    const auto raw = solve_dual_simplex(lp, sx, sd);
    if (raw.x.size() != sz(k.n + T)) return {};
    if (raw.proposed_status != core::Status::Optimal && raw.proposed_status != core::Status::Feasible)
        return {};
    return std::vector<f64>(raw.x.begin(), raw.x.begin() + k.n);
}

// Kaczmarz (1937) alternating projection of a start point onto those rows'
// hyperplanes: x <- x + (b_i - a_i'x) / ||a_i||^2 * a_i, swept row by row.
// Converges to the intersection for a consistent, well-posed system; a
// handful of sweeps gets most of the way there even when it doesn't (the
// barrier Newton step still has to satisfy every QUADRATIC row exactly,
// and can polish off any linear residual left over -- this is a cheap,
// matrix-free PRE-conditioning of the start, not a solve).  Kaczmarz knows
// nothing about the box, so the final clamp reapplies it; a badly
// conditioned row can otherwise walk a column outside its bounds.
void kaczmarz_project(const Core& k, const std::vector<core::Index>& rows, std::vector<f64>& xs,
                      int sweeps) {
    if (rows.empty()) return;
    const auto& rp = k.A.pattern.row_ptr();
    const auto& ci = k.A.pattern.col_idx();
    for (int s = 0; s < sweeps; ++s) {
        for (core::Index i : rows) {
            f64 ax = 0.0, nrm2 = 0.0;
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
                ax += k.A.vals[sz(t)] * xs[sz(ci[sz(t)])];
                nrm2 += k.A.vals[sz(t)] * k.A.vals[sz(t)];
            }
            if (nrm2 < 1e-300) continue;
            const f64 step = (k.rl[sz(i)] - ax) / nrm2;
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
                xs[sz(ci[sz(t)])] += step * k.A.vals[sz(t)];
        }
    }
    for (core::Index j = 0; j < k.n; ++j) {
        if (std::isfinite(k.l[sz(j)])) xs[sz(j)] = std::max(xs[sz(j)], k.l[sz(j)]);
        if (std::isfinite(k.u[sz(j)])) xs[sz(j)] = std::min(xs[sz(j)], k.u[sz(j)]);
    }
}

struct LocalRun {
    std::vector<f64> x, y;          // core space
    int iterations = 0;
    bool converged = false;
    std::string reason;
    f64 final_theta = kInf;         // set by run_restoration only; original-units l1 row violation of res.x
};

// The Newton system of the barrier problem and everything needed to form it.
class LocalIpm {
public:
    LocalIpm(const Core& k, bool verbose, std::uint64_t seed = 1, bool nested = false)
        : k_(k), verbose_(verbose), rng_(seed), nested_(nested) { build_pattern(); }

    // One local solve from the core-space start x0.  `on_iterate(x, y)` is
    // called after every iteration with the current core iterate; returning
    // true requests an early stop (used by run_restoration's exit test —
    // see its comment).  A plain multi-start caller's hook always returns
    // false.
    LocalRun run(std::vector<f64> x0, int max_it, f64 tol, Clock::time_point deadline,
                 const std::function<bool(const std::vector<f64>&, const std::vector<f64>&)>& on_iterate);

    // JOINT feasibility polish: Levenberg-Marquardt on the whole row-
    // residual vector (every row's signed distance outside [rl,ru], 0 if
    // inside), instead of the barrier Newton step's row-by-row-coupled-
    // through-the-dual view.  See its definition for the trust-region
    // radius update rule (Nielsen, IMM Tech. Report 1999-05).
    // `on_accept(x)` is called with the core-space point every time a step
    // is accepted (x actually changes) -- the caller's SCALED cost here and
    // the caller's judge()'s ORIGINAL-units max-violation are related but
    // not monotonic in each other (row-dependent Ruiz scale factors; the
    // same reason the barrier step's ep and a row's original-units
    // violation disagree, see the near-feasible fix above), so the FINAL
    // x this function returns is not guaranteed to be its own best
    // original-units point -- the caller needs to see intermediate points
    // to judge that itself, the same reason run()'s on_iterate exists.
    LocalRun run_feasibility_lm(std::vector<f64> x0, int max_it, Clock::time_point deadline,
                                const std::function<void(const std::vector<f64>&)>& on_accept);

    // RESTORATION PHASE (Waechter & Biegler 2006 s.3.3).  See its
    // definition for the exact subproblem (eq. 29a/31a) and the exit test
    // (eq. 18a/18b).  `xref` is the core-space
    // point the caller's main iteration got stuck at; `mu_main` is that
    // run's CURRENT barrier parameter (used for the exit test's phi and the
    // subproblem's proximity weight, matching the paper exactly).  `nested_`
    // guards against this spawning ANOTHER restoration phase inside itself
    // (see nested_'s comment).  `count_restorations` is 1 on the FIRST
    // restoration call of THIS start's trajectory, 2/3/... on later ones in
    // the same start (the rho warm-up keys off it; see the
    // definition for why -- the published infeasibility-detection variant keys
    // off the identical count_restorations_==1 condition, IpRestoMinC_
    // 1Nrm.cpp, though for a different variable).
    LocalRun run_restoration(const std::vector<f64>& xref, f64 mu_main, Clock::time_point deadline, f64 tol,
                             int count_restorations);

private:
    const Core& k_;
    bool verbose_;
    std::mt19937_64 rng_;
    // True for the nested LocalIpm run_restoration constructs internally to
    // solve ITS OWN augmented subproblem: restoration must not recursively
    // trigger restoration-of-restoration on its own stalls (unbounded
    // recursion / runaway compute) -- the published method has the identical guard
    // (`resto.start_with_resto = no`, IpRestoMinC_1Nrm.cpp
    // InitializeImpl), forced for exactly this reason.  A nested run that
    // stalls just falls back to this file's existing random-restart, same
    // as before this phase existed.
    bool nested_ = false;
    core::Index n_ = 0, m_ = 0, N_ = 0;
    std::size_t nq_ = 0;
    std::vector<std::uint8_t> has_l_, has_u_, fixed_w_;
    // Quadratic rows: support, local entry positions, Q_t x on the support.
    std::vector<std::vector<core::Index>> qsup_, qlr_, qlc_;
    std::vector<std::vector<f64>> qix_;
    // KKT pattern: upper CSC over (x, y); slots of every data entry.
    la::SymCsc K_;
    la::Ldlt fac_;
    std::vector<std::int8_t> sign_;
    std::vector<core::Offset> diag_slot_, q0_slot_, a_slot_;
    std::vector<std::vector<core::Offset>> qslot_, jslot_;

    f64 lo(core::Index c) const { return c < n_ ? k_.l[sz(c)] : k_.rl[sz(c - n_)]; }
    f64 hi(core::Index c) const { return c < n_ ? k_.u[sz(c)] : k_.ru[sz(c - n_)]; }

    void build_pattern();
    // A fresh point uniform in each column's box; unbounded columns are
    // clipped to +-10 around `near`'s entry (same convention as the
    // caller's outer multi-start over `s>0`) -- used both for that and for
    // an in-run restart after a stall (see run()).
    std::vector<f64> random_start(const std::vector<f64>& near);
    // g(x) into gx (linear + quadratic), and qix_ at x.
    void eval_rows(const std::vector<f64>& x, std::vector<f64>& gx);
    f64 objective(const std::vector<f64>& x) const;
    void grad_f(const std::vector<f64>& x, std::vector<f64>& g) const;
    // J'y (J at the x qix_ was last formed at).
    void jac_t(const std::vector<f64>& y, std::vector<f64>& out) const;
    void jac(const std::vector<f64>& dx, std::vector<f64>& out) const;
    void fill(const std::vector<f64>& y, const std::vector<f64>& tdiag, f64 delta_w, f64 delta_c);
    // theta(x) = l1 row violation past [rl,ru] (the paper's ||c(x)||_1,
    // generalised to a two-sided range); phi(x,mu) = f(x) - mu * sum_j
    // log(x's own box slack).  ORIGINAL problem units (k_, not an augmented
    // restoration Core) -- run_restoration's exit test is the only caller.
    void theta_phi(const std::vector<f64>& x, f64 mu_, f64& theta, f64& phi);
};

void LocalIpm::build_pattern() {
    n_ = k_.n;
    m_ = k_.m;
    N_ = n_ + m_;
    nq_ = k_.quad.size();
    has_l_.assign(sz(N_), 0);
    has_u_.assign(sz(N_), 0);
    fixed_w_.assign(sz(m_), 0);
    for (core::Index c = 0; c < N_; ++c) {
        has_l_[sz(c)] = std::isfinite(lo(c));
        has_u_[sz(c)] = std::isfinite(hi(c));
    }
    for (core::Index i = 0; i < m_; ++i)
        if (k_.rl[sz(i)] == k_.ru[sz(i)]) {
            fixed_w_[sz(i)] = 1;
            has_l_[sz(n_ + i)] = has_u_[sz(n_ + i)] = 0;
        }
    qsup_.assign(nq_, {});
    qlr_.assign(nq_, {});
    qlc_.assign(nq_, {});
    qix_.assign(nq_, {});
    for (std::size_t t = 0; t < nq_; ++t) {
        const auto& q = k_.quad[t];
        auto& S = qsup_[t];
        for (std::size_t e = 0; e < q.v.size(); ++e) { S.push_back(q.r[e]); S.push_back(q.c[e]); }
        std::sort(S.begin(), S.end());
        S.erase(std::unique(S.begin(), S.end()), S.end());
        for (std::size_t e = 0; e < q.v.size(); ++e) {
            qlr_[t].push_back(static_cast<core::Index>(std::lower_bound(S.begin(), S.end(), q.r[e]) - S.begin()));
            qlc_[t].push_back(static_cast<core::Index>(std::lower_bound(S.begin(), S.end(), q.c[e]) - S.begin()));
        }
        qix_[t].assign(S.size(), 0.0);
    }
    // Columns: x block = Q0 upper u every Q_t upper u diagonal; column n+i =
    // supp(a_i) u supp(Q_i) u diagonal.  Every data entry records its slot.
    K_.n = N_;
    K_.col_ptr.assign(sz(N_) + 1, 0);
    diag_slot_.assign(sz(N_), 0);
    std::vector<std::vector<core::Index>> rows(sz(N_));
    for (core::Index j = 0; j < n_; ++j) {
        for (const auto& [i, v] : k_.qcol[sz(j)]) { (void)v; rows[sz(j)].push_back(i); }
        rows[sz(j)].push_back(j);
    }
    for (const auto& q : k_.quad)
        for (std::size_t e = 0; e < q.v.size(); ++e) rows[sz(q.c[e])].push_back(q.r[e]);
    const auto& rp = k_.A.pattern.row_ptr();
    const auto& ci = k_.A.pattern.col_idx();
    for (core::Index i = 0; i < m_; ++i) {
        auto& col = rows[sz(n_ + i)];
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) col.push_back(ci[sz(t)]);
        col.push_back(n_ + i);
    }
    for (std::size_t t = 0; t < nq_; ++t)
        for (core::Index j : qsup_[t]) rows[sz(n_ + k_.quad[t].row)].push_back(j);
    for (core::Index c = 0; c < N_; ++c) {
        auto& col = rows[sz(c)];
        std::sort(col.begin(), col.end());
        col.erase(std::unique(col.begin(), col.end()), col.end());
        for (core::Index r : col) K_.row_idx.push_back(r);
        K_.col_ptr[sz(c) + 1] = static_cast<core::Offset>(K_.row_idx.size());
    }
    K_.vals.assign(K_.row_idx.size(), 0.0);
    auto slot = [&](core::Index r, core::Index c) {
        const auto b = K_.row_idx.begin() + K_.col_ptr[sz(c)];
        const auto e = K_.row_idx.begin() + K_.col_ptr[sz(c) + 1];
        return static_cast<core::Offset>(std::lower_bound(b, e, r) - K_.row_idx.begin());
    };
    for (core::Index c = 0; c < N_; ++c) diag_slot_[sz(c)] = slot(c, c);
    for (core::Index j = 0; j < n_; ++j)
        for (const auto& [i, v] : k_.qcol[sz(j)]) { (void)v; q0_slot_.push_back(slot(i, j)); }
    for (core::Index i = 0; i < m_; ++i)
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) a_slot_.push_back(slot(ci[sz(t)], n_ + i));
    qslot_.assign(nq_, {});
    jslot_.assign(nq_, {});
    for (std::size_t t = 0; t < nq_; ++t) {
        const auto& q = k_.quad[t];
        for (std::size_t e = 0; e < q.v.size(); ++e) qslot_[t].push_back(slot(q.r[e], q.c[e]));
        for (core::Index j : qsup_[t]) jslot_[t].push_back(slot(j, n_ + q.row));
    }
    sign_.assign(sz(N_), 1);
    for (core::Index j = 0; j < n_; ++j) sign_[sz(j)] = -1;
    fac_.analyze(K_);
}

std::vector<f64> LocalIpm::random_start(const std::vector<f64>& near) {
    std::vector<f64> xs(sz(n_));
    for (core::Index j = 0; j < n_; ++j) {
        const f64 base = static_cast<std::size_t>(j) < near.size() ? near[sz(j)] : 0.0;
        const f64 a = std::isfinite(lo(j)) ? lo(j) : base - 10.0;
        const f64 b = std::isfinite(hi(j)) ? hi(j) : base + 10.0;
        std::uniform_real_distribution<f64> U(std::min(a, b), std::max(a, b));
        xs[sz(j)] = U(rng_);
    }
    return xs;
}

void LocalIpm::eval_rows(const std::vector<f64>& x, std::vector<f64>& gx) {
    gx.assign(sz(m_), 0.0);
    const auto& rp = k_.A.pattern.row_ptr();
    const auto& ci = k_.A.pattern.col_idx();
    for (core::Index i = 0; i < m_; ++i)
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) gx[sz(i)] += k_.A.vals[sz(t)] * x[sz(ci[sz(t)])];
    for (std::size_t t = 0; t < nq_; ++t) {
        const auto& q = k_.quad[t];
        auto& w = qix_[t];
        std::fill(w.begin(), w.end(), 0.0);
        f64 val = 0.0;
        for (std::size_t e = 0; e < q.v.size(); ++e) {
            const f64 xr = x[sz(q.r[e])], xc = x[sz(q.c[e])];
            if (q.r[e] == q.c[e]) {
                w[sz(qlr_[t][e])] += q.v[e] * xr;
                val += 0.5 * q.v[e] * xr * xr;
            } else {
                w[sz(qlr_[t][e])] += q.v[e] * xc;
                w[sz(qlc_[t][e])] += q.v[e] * xr;
                val += q.v[e] * xr * xc;
            }
        }
        gx[sz(q.row)] += val;
    }
}

f64 LocalIpm::objective(const std::vector<f64>& x) const {
    f64 s = 0.0;
    for (core::Index j = 0; j < n_; ++j) {
        s += k_.c[sz(j)] * x[sz(j)];
        for (const auto& [i, v] : k_.qcol[sz(j)]) s += (i == j ? 0.5 : 1.0) * v * x[sz(i)] * x[sz(j)];
    }
    return s;
}

void LocalIpm::grad_f(const std::vector<f64>& x, std::vector<f64>& g) const {
    g.assign(sz(n_), 0.0);
    for (core::Index j = 0; j < n_; ++j) {
        g[sz(j)] += k_.c[sz(j)];
        for (const auto& [i, v] : k_.qcol[sz(j)]) {
            g[sz(i)] += v * x[sz(j)];
            if (i != j) g[sz(j)] += v * x[sz(i)];
        }
    }
}

void LocalIpm::theta_phi(const std::vector<f64>& x, f64 mu_, f64& theta, f64& phi) {
    std::vector<f64> gx;
    eval_rows(x, gx);   // also forms qix_ at x -- fine, no caller of theta_phi relies on qix_ afterward
    theta = 0.0;
    for (core::Index i = 0; i < m_; ++i) {
        const f64 rl = k_.rl[sz(i)], ru = k_.ru[sz(i)];
        if (std::isfinite(rl) && gx[sz(i)] < rl) theta += rl - gx[sz(i)];
        else if (std::isfinite(ru) && gx[sz(i)] > ru) theta += gx[sz(i)] - ru;
    }
    phi = objective(x);
    for (core::Index j = 0; j < n_; ++j) {
        if (has_l_[sz(j)]) phi -= mu_ * std::log(std::max(x[sz(j)] - lo(j), kUlp));
        if (has_u_[sz(j)]) phi -= mu_ * std::log(std::max(hi(j) - x[sz(j)], kUlp));
    }
}

void LocalIpm::jac_t(const std::vector<f64>& y, std::vector<f64>& out) const {
    out.assign(sz(n_), 0.0);
    const auto& rp = k_.A.pattern.row_ptr();
    const auto& ci = k_.A.pattern.col_idx();
    for (core::Index i = 0; i < m_; ++i)
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) out[sz(ci[sz(t)])] += k_.A.vals[sz(t)] * y[sz(i)];
    for (std::size_t t = 0; t < nq_; ++t) {
        const f64 yr = y[sz(k_.quad[t].row)];
        for (std::size_t s = 0; s < qsup_[t].size(); ++s) out[sz(qsup_[t][s])] += qix_[t][s] * yr;
    }
}

void LocalIpm::jac(const std::vector<f64>& dx, std::vector<f64>& out) const {
    out.assign(sz(m_), 0.0);
    const auto& rp = k_.A.pattern.row_ptr();
    const auto& ci = k_.A.pattern.col_idx();
    for (core::Index i = 0; i < m_; ++i)
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) out[sz(i)] += k_.A.vals[sz(t)] * dx[sz(ci[sz(t)])];
    for (std::size_t t = 0; t < nq_; ++t) {
        f64 s2 = 0.0;
        for (std::size_t s = 0; s < qsup_[t].size(); ++s) s2 += qix_[t][s] * dx[sz(qsup_[t][s])];
        out[sz(k_.quad[t].row)] += s2;
    }
}

// K = [ -(H + T_x + delta_w + rho)   J'              ]
//     [   J                          T_w^-1 + delta_c ]
// H = Q0 - sum_t y_t Q_t.
void LocalIpm::fill(const std::vector<f64>& y, const std::vector<f64>& tdiag, f64 delta_w,
                     f64 delta_c) {
    constexpr f64 rho = 1e-8;
    std::fill(K_.vals.begin(), K_.vals.end(), 0.0);
    std::size_t p = 0;
    for (core::Index j = 0; j < n_; ++j)
        for (const auto& [i, v] : k_.qcol[sz(j)]) { (void)i; K_.vals[sz(q0_slot_[p++])] -= v; }
    for (std::size_t t = 0; t < nq_; ++t) {
        const f64 yt = y[sz(k_.quad[t].row)];
        if (yt != 0.0)
            for (std::size_t e = 0; e < k_.quad[t].v.size(); ++e) K_.vals[sz(qslot_[t][e])] += yt * k_.quad[t].v[e];
    }
    for (core::Index j = 0; j < n_; ++j) K_.vals[sz(diag_slot_[sz(j)])] -= tdiag[sz(j)] + delta_w + rho;
    p = 0;
    for (core::Index i = 0; i < m_; ++i) {
        const auto& rp = k_.A.pattern.row_ptr();
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) K_.vals[sz(a_slot_[p++])] += k_.A.vals[sz(t)];
    }
    for (std::size_t t = 0; t < nq_; ++t)
        for (std::size_t s = 0; s < jslot_[t].size(); ++s) K_.vals[sz(jslot_[t][s])] += qix_[t][s];
    for (core::Index i = 0; i < m_; ++i) {
        const core::Index c = n_ + i;
        K_.vals[sz(diag_slot_[sz(c)])] += (fixed_w_[sz(i)] ? 0.0 : 1.0 / tdiag[sz(c)]) + delta_c;
    }
}

LocalRun LocalIpm::run(std::vector<f64> x, int max_it, f64 tol, Clock::time_point deadline,
                       const std::function<bool(const std::vector<f64>&, const std::vector<f64>&)>& on_iterate) {
    LocalRun res;
    const core::Index n = n_, m = m_, N = N_;
    // Constants from Waechter & Biegler 2006, Table 1 / s.2-3.
    constexpr f64 kappa_eps = 10.0, kappa_mu = 0.2, theta_mu = 1.5, tau_min = 0.99;
    constexpr f64 kappa_sigma = 1e10, eta_armijo = 1e-4;
    constexpr f64 delta_w_min = 1e-20, delta_w_0 = 1e-4, delta_w_max = 1e40;

    // ---- starting point (Waechter & Biegler s.3.6): x pushed inside its box, w = g(x)
    // pushed inside its range, y = 0, bound multipliers 1.
    auto push = [&](core::Index c, f64 v) {
        const f64 a = lo(c), b = hi(c);
        const f64 pa = std::isfinite(a) ? 1e-2 * std::max(1.0, std::fabs(a)) : 0.0;
        const f64 pb = std::isfinite(b) ? 1e-2 * std::max(1.0, std::fabs(b)) : 0.0;
        if (std::isfinite(a) && std::isfinite(b)) {
            const f64 w = b - a;
            const f64 qa = std::min(pa, 1e-2 * w), qb = std::min(pb, 1e-2 * w);
            if (w <= 0.0) return a;
            return std::clamp(v, a + qa, b - qb);
        }
        if (std::isfinite(a)) return std::max(v, a + pa);
        if (std::isfinite(b)) return std::min(v, b - pb);
        return v;
    };
    std::vector<f64> z(sz(N)), y(sz(m)), lam(sz(N)), nu(sz(N));
    std::vector<f64> gx;
    // (Re-)initialise the primal-dual point from a core-space x0: x pushed
    // inside its box, w = g(x) pushed inside its range, y dual-feasible for
    // the bound multipliers (see the comment below), bound multipliers 1.
    // A named, callable step (not just the function's prologue) because a
    // stalled run restarts from here rather than giving up outright --
    // see maybe_restart below.
    auto init_state = [&](const std::vector<f64>& x0) {
        for (core::Index j = 0; j < n; ++j) z[sz(j)] = push(j, x0[sz(j)]);
        std::vector<f64> xs(z.begin(), z.begin() + n);
        eval_rows(xs, gx);
        for (core::Index i = 0; i < m; ++i)
            z[sz(n + i)] = fixed_w_[sz(i)] ? k_.rl[sz(i)] : push(n + i, gx[sz(i)]);
        std::fill(y.begin(), y.end(), 0.0);
        std::fill(lam.begin(), lam.end(), 0.0);
        std::fill(nu.begin(), nu.end(), 0.0);
        for (core::Index c = 0; c < N; ++c) {
            if (has_l_[sz(c)]) lam[sz(c)] = 1.0;
            if (has_u_[sz(c)]) nu[sz(c)] = 1.0;
        }
        // Row multipliers dual-feasible for their slacks, y = lam_w - nu_w
        // (rdw = 0): a one-sided row starts with |y| = 1 and so contributes
        // its curvature -y Q_i to the Hessian from the first step.  With
        // y = 0 a norm-ball row seen from its centre has zero gradient AND
        // zero curvature, and the first Newton step in its columns is
        // bounded only by the 1e-8 proximal term (measured on QPLIB_2482:
        // steps cut to 1e-4 by the line search for hundreds of iterations).
        for (core::Index i = 0; i < m; ++i)
            if (!fixed_w_[sz(i)]) y[sz(i)] = lam[sz(n + i)] - nu[sz(n + i)];
    };
    const std::vector<f64> x_center = x;   // caller's start, for any restart's unbounded columns
    init_state(x);
    f64 mu = 0.1, nu_pen = 1.0, delta_last = 0.0;
    int ls_fail = 0, tiny_alpha_count = 0;
    // NO-PROGRESS STALL.  Neither ls_fail nor tiny_alpha_count sees this
    // one: a step can be Armijo-"accepted" every iteration (phi improves
    // because the dual/penalty terms move) while primal infeasibility (ep)
    // barely moves, because the accepted direction is dominated by a huge,
    // ill-conditioned dual Newton step rather than a real feasibility
    // improvement.  Measured on QPLIB_2823 (LCQ, 283 quadratic equality
    // rows) even after the delta_c fix above: ep sat at 21.3 -> 20.9 over
    // 30 iterations (delta_w swinging 1e5-1e7 every single iteration, ed
    // growing into the thousands) -- technically all "accepted" steps,
    // burning the full 100000-iteration cap with no feasibility gained.
    // Track ep every kStallWindow iterations; if it hasn't dropped by at
    // least kStallMinDecrease relative and is still meaningfully
    // infeasible, restart is a better bet than more of the same Newton
    // direction.
    f64 ep_stall_ref = kInf;
    int stall_window_start = 0;
    // kStallWindow and the restoration-phase per-call cap below are
    // env-var-tunable (SOR_STALL_WINDOW, SOR_RESTO_MS) purely so a sweep can
    // scan several values without a rebuild per value -- read once per run()
    // call, negligible cost, no behavioural difference from a constexpr when
    // the env var is unset (defaults match the prior hardcoded values).
    // try a larger restoration cap (500ms -> 2s/5s) and
    // earlier stall entry (smaller window) on the 12 near-feasible LCQ
    // instances; whichever wins on the FULL 52-sweep gets hardcoded and the
    // env-var plumbing can be deleted then.
    const int kStallWindow = [] {
        if (const char* e = std::getenv("SOR_STALL_WINDOW")) { try { return std::stoi(e); } catch (...) {} }
        return 40;
    }();
    constexpr f64 kStallMinDecrease = 0.10;
    // RESTART ON STALL.  The paper's answer to a line search that keeps
    // failing (or an un-recoverable inertia/Newton failure) is a feasibility
    // restoration phase -- a second barrier problem, minimising constraint
    // violation, that needs its own KKT machinery.  The cheap stand-in used
    // here: jump to a fresh random point (same distribution as the caller's
    // outer multi-start) and re-run from mu = 0.1.  Measured on QPLIB_2881
    // (LCQ): the default start stalls at 19 iterations after ~30 ms, well
    // inside any real time budget -- restarting spends the REST of that
    // budget on new attempts instead of returning a mediocre point early.
    // Bounded by max_restarts so a pathological instance that fails on
    // every attempt still respects max_it/the deadline, not a runaway loop
    // -- tied to max_it (every restart burns at least one `it`, so this can
    // never be the binding constraint; max_it and the deadline already are).
    int restarts_used = 0;
    const int max_restarts = max_it;
    // Per-START restoration call counter (resets to 0 every run() call,
    // i.e. every multi-start attempt) -- the rho warm-up (see
    // run_restoration) only softens the FIRST restoration call of a given
    // start, mirroring the published first-restoration special case.
    int resto_calls = 0;
    // Errors declared here (not inside the loop where they're computed) so
    // maybe_restart -- defined once, below, before the loop reaches its own
    // errors() call for the first time -- can log the exact ep/ed/ec that
    // triggered a restart.  Lambda name lookup for a [&]-capture is
    // resolved at the lambda's OWN textual position, so these have to be in
    // scope before that lambda, not just before it is first CALLED.
    f64 ep = 0.0, ed = 0.0, ec = 0.0;
    f64 ep_best = kInf;   // best (lowest) scaled primal residual ever seen this run() call

    std::vector<f64> xv(sz(n)), gf, jty, rdx(sz(n)), rdw(sz(m)), rp(sz(m));
    std::vector<f64> sl(sz(N)), su(sz(N)), tdiag(sz(N)), rhs(sz(N));
    std::vector<f64> dz(sz(N)), dy(sz(m)), dlam(sz(N)), dnu(sz(N)), hw(sz(m)), jdx;
    auto slacks = [&](const std::vector<f64>& zz, std::vector<f64>& l_, std::vector<f64>& u_) {
        for (core::Index c = 0; c < N; ++c) {
            l_[sz(c)] = has_l_[sz(c)] ? std::max(zz[sz(c)] - lo(c), kUlp * std::max(1.0, std::fabs(lo(c)))) : 0.0;
            u_[sz(c)] = has_u_[sz(c)] ? std::max(hi(c) - zz[sz(c)], kUlp * std::max(1.0, std::fabs(hi(c)))) : 0.0;
        }
    };
    // Barrier merit: f - mu sum log s + nu_pen ||g(x) - w||_1 (evaluated with
    // its own g, so the line search can call it at trial points).
    std::vector<f64> gtrial, ltrial(sz(N)), utrial(sz(N));
    auto merit = [&](const std::vector<f64>& zz, f64 mu_, f64 pen, f64& infeas) {
        std::vector<f64> xx(zz.begin(), zz.begin() + n);
        f64 phi = objective(xx);
        eval_rows(xx, gtrial);
        infeas = 0.0;
        for (core::Index i = 0; i < m; ++i) infeas += std::fabs(gtrial[sz(i)] - zz[sz(n + i)]);
        slacks(zz, ltrial, utrial);
        for (core::Index c = 0; c < N; ++c) {
            if (has_l_[sz(c)]) phi -= mu_ * std::log(ltrial[sz(c)]);
            if (has_u_[sz(c)]) phi -= mu_ * std::log(utrial[sz(c)]);
        }
        return phi + pen * infeas;
    };

    for (int it = 0; it < max_it; ++it) {
        res.iterations = it + 1;
        if (Clock::now() > deadline) { res.reason = "time limit"; break; }
        // On an unrecoverable failure this iteration, restart from a fresh
        // point instead of giving up, if the budget allows one more attempt.
        // `why` becomes res.reason only if no restart is left to take.
        auto maybe_restart = [&](const char* why) {
            // Logged unconditionally on verbose (not gated on WHICH reason)
            // so a --verbose trace shows the ep/ed/ec this run had actually
            // reached at the moment it gave the trajectory up -- the
            // question asked of the 14 near-feasible
            // instances: is the no-progress-stall restart cutting off real
            // refinement (ep already small, still falling) or discarding a
            // trajectory that had genuinely plateaued (ep not small, not
            // falling)?  See the near-feasible fix below for what this
            // showed on QPLIB_2703/2650/3177/8810/9004.
            if (verbose_)
                std::fprintf(stderr,
                    "local restart it=%d reason=\"%s\" ep=%.3e ed=%.3e ec=%.3e ep_best=%.3e restarts_used=%d\n",
                    it + 1, why, ep, ed, ec, ep_best, restarts_used + 1);
            if (restarts_used >= max_restarts || Clock::now() > deadline || it + 1 >= max_it) {
                res.reason = why;
                return false;
            }
            ++restarts_used;
            // RESTORATION FIRST (Waechter & Biegler 2006 s.3.3;
            // see LocalIpm::run_restoration).  Every one of this function's
            // four callers is exactly the situation the published restoration
            // phase exists for: the main barrier direction stopped making
            // real progress toward feasibility.  Try the PRINCIPLED
            // recovery -- a subproblem solve from THIS trajectory's current
            // point, not a blind jump -- before falling back to the random
            // restart that is all this engine had before this phase.
            // `nested_` blocks this on the restoration subproblem's own
            // internal LocalIpm (no restoration-of-restoration).  Capped to
            // a modest slice of whatever time is left so one stubborn start
            // cannot eat the whole multi-start budget.
            if (!nested_) {
                static const long kRestoMs = [] {
                    if (const char* e = std::getenv("SOR_RESTO_MS")) { try { return std::stol(e); } catch (...) {} }
                    return 500L;
                }();
                const auto resto_dl = std::min(deadline, Clock::now() + std::chrono::milliseconds(kRestoMs));
                std::vector<f64> xcur(z.begin(), z.begin() + n);
                LocalRun rr = run_restoration(xcur, mu, resto_dl, tol, ++resto_calls);
                if (!rr.x.empty() && rr.final_theta < kInf) {
                    f64 theta_before;
                    { f64 phi_unused; theta_phi(xcur, mu, theta_before, phi_unused); }
                    // Only adopt the restoration endpoint if it did not
                    // make things WORSE (its own subproblem minimises p+n,
                    // so this should hold whenever it ran at all, but a
                    // time-truncated, never-converged solve is not
                    // guaranteed monotone in theta -- verify, don't assume).
                    if (rr.final_theta <= theta_before) {
                        init_state(rr.x);
                        mu = 0.1; nu_pen = 1.0; delta_last = 0.0; ls_fail = 0; tiny_alpha_count = 0;
                        ep_stall_ref = kInf; stall_window_start = it + 1;
                        if (verbose_)
                            std::fprintf(stderr, "local restart it=%d restoration accepted theta %.3e -> %.3e (%s)\n",
                                         it + 1, theta_before, rr.final_theta, rr.reason.c_str());
                        return true;
                    }
                }
            }
            init_state(random_start(x_center));
            mu = 0.1; nu_pen = 1.0; delta_last = 0.0; ls_fail = 0; tiny_alpha_count = 0;
            ep_stall_ref = kInf; stall_window_start = it + 1;
            return true;
        };
        std::copy(z.begin(), z.begin() + n, xv.begin());
        eval_rows(xv, gx);           // also forms qix_ at x: J(x)
        grad_f(xv, gf);
        jac_t(y, jty);
        for (core::Index j = 0; j < n; ++j) rdx[sz(j)] = gf[sz(j)] - jty[sz(j)] - lam[sz(j)] + nu[sz(j)];
        for (core::Index i = 0; i < m; ++i) {
            rdw[sz(i)] = fixed_w_[sz(i)] ? 0.0 : y[sz(i)] - lam[sz(n + i)] + nu[sz(n + i)];
            rp[sz(i)] = gx[sz(i)] - z[sz(n + i)];
        }
        slacks(z, sl, su);
        if (on_iterate(xv, y)) { res.reason = "early stop requested"; break; }

        // Optimality error (Waechter & Biegler s.2.1: dual and complementarity scaled by
        // the multipliers' size, s_max = 100).
        auto errors = [&](f64 mu_, f64& e_p, f64& e_d, f64& e_c) {
            e_p = e_d = e_c = 0.0;
            f64 msum = 0.0;
            int mcnt = 0;
            for (f64 v : rp) e_p = std::max(e_p, std::fabs(v));
            for (f64 v : rdx) e_d = std::max(e_d, std::fabs(v));
            for (f64 v : rdw) e_d = std::max(e_d, std::fabs(v));
            for (f64 v : y) { msum += std::fabs(v); ++mcnt; }
            for (core::Index c = 0; c < N; ++c) {
                if (has_l_[sz(c)]) { e_c = std::max(e_c, std::fabs(sl[sz(c)] * lam[sz(c)] - mu_)); msum += lam[sz(c)]; ++mcnt; }
                if (has_u_[sz(c)]) { e_c = std::max(e_c, std::fabs(su[sz(c)] * nu[sz(c)] - mu_)); msum += nu[sz(c)]; ++mcnt; }
            }
            const f64 sd = std::max(100.0, mcnt > 0 ? msum / mcnt : 0.0) / 100.0;
            e_d /= sd;
            e_c /= sd;
        };
        errors(0.0, ep, ed, ec);
        ep_best = std::min(ep_best, ep);
        if (std::max({ep, ed, ec}) <= tol) {
            res.converged = true;
            res.reason = "local KKT point (scaled tolerance)";
            break;
        }
        // TRIED AND REVERTED: exempting the no-progress-stall restart once
        // ep is within a small factor of ep_best.
        // MEASURED HARMFUL on the full LCQ sweep: QPLIB_2416 (feasible in
        // every prior sweep) regressed to NoSolutionFound, final viol
        // 2.53e-2, restarts suppressed the whole run because ep never
        // strayed more than 3x from a ep_best that was ITSELF stuck around
        // 0.02 -- "close to this run's own best" and "about to converge"
        // turned out to be the same signal as "stuck at a bad plateau",
        // exactly the ambiguity flagged as unresolved when this was first
        // written.  ep_best is kept
        // (maybe_restart's log line below still reports it -- useful for
        // diagnosis) but no longer gates restarting.  The near-feasible
        // bucket is instead handled by the LM joint feasibility polish
        // below (LocalIpm::run_feasibility_lm), which operates on every
        // start's endpoint regardless of how the barrier step's OWN
        // restart policy behaved -- so restarting freely, trying more
        // diverse starts, only gives it more endpoints to polish.
        if (it - stall_window_start >= kStallWindow) {
            if (ep_stall_ref < kInf && ep > 1e-3 && ep > (1.0 - kStallMinDecrease) * ep_stall_ref) {
                if (maybe_restart("no feasibility progress over a window of iterations")) continue;
                break;
            }
            ep_stall_ref = ep;
            stall_window_start = it;
        }
        // Barrier update: solve each barrier problem to kappa_eps * mu.
        for (int guard = 0; guard < 20; ++guard) {
            errors(mu, ep, ed, ec);
            if (std::max({ep, ed, ec}) > kappa_eps * mu || mu <= tol / 10.0) break;
            mu = std::max(tol / 10.0, std::min(kappa_mu * mu, std::pow(mu, theta_mu)));
        }

        for (core::Index c = 0; c < N; ++c)
            tdiag[sz(c)] = (has_l_[sz(c)] ? lam[sz(c)] / sl[sz(c)] : 0.0) +
                           (has_u_[sz(c)] ? nu[sz(c)] / su[sz(c)] : 0.0);
        // Inertia correction (Waechter & Biegler s.3.1): delta_w alone fixes
        // a (1,1) block (the Lagrangian Hessian) that isn't negative
        // definite enough.  It does NOT fix a (2,2)/Jacobian block that is
        // singular or near it -- a quadratic equality row's Jacobian row
        // (a_i + Q_i x) is exactly the zero vector at any x on that row's
        // own critical set, and plenty of QPLIB's bilinear-equality-heavy
        // LCQ instances (e.g. QPLIB_2823, 283 such rows) pass through
        // points near one.  No amount of delta_w repairs that: it only acts
        // on the x-block.  The paper's answer is delta_c on the (2,2)
        // block, tried only once delta_w has been pushed to delta_w_max
        // without producing a clean factorization (regularized_pivots()
        // counts entries the LDL' had to bump for sign OR magnitude, so a
        // persistent nonzero count here even at delta_w_max is the paper's
        // signal to blame the Jacobian, not the Hessian).  delta_c starts
        // at the old fixed constant (so a well-conditioned iterate behaves
        // exactly as before) and grows x100 per outer retry, each retry
        // re-running the FULL delta_w ladder from scratch since the right
        // delta_w generally shrinks once the Jacobian block is regularized.
        f64 delta_w = 0.0, delta_c = 1e-8;
        bool ok = false;
        for (int outer = 0; outer < 5 && !ok; ++outer) {
            delta_w = 0.0;
            for (int attempt = 0; attempt < 60; ++attempt) {
                fill(y, tdiag, delta_w, delta_c);
                const bool fin = fac_.factorize(K_, sign_, 1e-12);
                if (fin && fac_.regularized_pivots() == 0) { ok = true; break; }
                if (delta_w == 0.0) delta_w = delta_last == 0.0 ? delta_w_0 : std::max(delta_w_min, delta_last / 3.0);
                else delta_w *= delta_last == 0.0 ? 100.0 : 8.0;
                if (delta_w > delta_w_max) break;
            }
            if (!ok) delta_c *= 100.0;
        }
        if (!ok) { if (maybe_restart("inertia correction failed")) continue; break; }
        delta_last = delta_w;

        // Newton direction of the barrier problem (targets mu).
        for (core::Index j = 0; j < n; ++j) {
            const f64 h = (has_l_[sz(j)] ? -mu / sl[sz(j)] + lam[sz(j)] : 0.0) +
                          (has_u_[sz(j)] ? mu / su[sz(j)] - nu[sz(j)] : 0.0);
            rhs[sz(j)] = rdx[sz(j)] + h;
        }
        for (core::Index i = 0; i < m; ++i) {
            const core::Index c = n + i;
            if (fixed_w_[sz(i)]) { hw[sz(i)] = 0.0; rhs[sz(c)] = -rp[sz(i)]; continue; }
            hw[sz(i)] = (has_l_[sz(c)] ? -mu / sl[sz(c)] + lam[sz(c)] : 0.0) +
                        (has_u_[sz(c)] ? mu / su[sz(c)] - nu[sz(c)] : 0.0);
            rhs[sz(c)] = -rp[sz(i)] - (rdw[sz(i)] + hw[sz(i)]) / tdiag[sz(c)];
        }
        fac_.solve(rhs);
        bool finite = true;
        for (f64 v : rhs) finite = finite && std::isfinite(v);
        if (!finite) { if (maybe_restart("non-finite Newton direction")) continue; break; }
        for (core::Index j = 0; j < n; ++j) dz[sz(j)] = rhs[sz(j)];
        for (core::Index i = 0; i < m; ++i) {
            dy[sz(i)] = rhs[sz(n + i)];
            dz[sz(n + i)] = fixed_w_[sz(i)] ? 0.0 : (-rdw[sz(i)] - hw[sz(i)] - dy[sz(i)]) / tdiag[sz(n + i)];
        }
        for (core::Index c = 0; c < N; ++c) {
            dlam[sz(c)] = has_l_[sz(c)] ? (mu - sl[sz(c)] * lam[sz(c)] - lam[sz(c)] * dz[sz(c)]) / sl[sz(c)] : 0.0;
            dnu[sz(c)] = has_u_[sz(c)] ? (mu - su[sz(c)] * nu[sz(c)] + nu[sz(c)] * dz[sz(c)]) / su[sz(c)] : 0.0;
        }
        // Fraction to the boundary, separately for primal and bound duals.
        const f64 tau = std::max(tau_min, 1.0 - mu);
        f64 ap = 1.0, ad = 1.0;
        for (core::Index c = 0; c < N; ++c) {
            if (has_l_[sz(c)] && dz[sz(c)] < 0.0) ap = std::min(ap, -tau * sl[sz(c)] / dz[sz(c)]);
            if (has_u_[sz(c)] && dz[sz(c)] > 0.0) ap = std::min(ap, tau * su[sz(c)] / dz[sz(c)]);
            if (has_l_[sz(c)] && dlam[sz(c)] < 0.0) ad = std::min(ad, -tau * lam[sz(c)] / dlam[sz(c)]);
            if (has_u_[sz(c)] && dnu[sz(c)] < 0.0) ad = std::min(ad, -tau * nu[sz(c)] / dnu[sz(c)]);
        }
        // Merit: directional derivative of the barrier part, and of the l1
        // infeasibility from the linearisation J dx - dw.
        f64 dbar = 0.0;
        for (core::Index j = 0; j < n; ++j) dbar += gf[sz(j)] * dz[sz(j)];
        for (core::Index c = 0; c < N; ++c) {
            if (has_l_[sz(c)]) dbar -= mu * dz[sz(c)] / sl[sz(c)];
            if (has_u_[sz(c)]) dbar += mu * dz[sz(c)] / su[sz(c)];
        }
        std::vector<f64> dxv(dz.begin(), dz.begin() + n);
        jac(dxv, jdx);
        f64 dinf = 0.0, inf1 = 0.0;
        for (core::Index i = 0; i < m; ++i) {
            const f64 lin = jdx[sz(i)] - dz[sz(n + i)];
            inf1 += std::fabs(rp[sz(i)]);
            dinf += rp[sz(i)] > 0.0 ? lin : (rp[sz(i)] < 0.0 ? -lin : std::fabs(lin));
        }
        // Penalty: above the new multipliers (N&W Thm 18.2), and large
        // enough that the step is a descent direction when infeasible.
        // Capped: nu_pen only ever grows (never shrinks back down within a
        // run), so ONE spurious huge |y+dy| -- a bilinear equality-row
        // Jacobian is rank-deficient or near it at plenty of interior
        // points, which the Newton solve answers with a wild dual step --
        // locks in a penalty that cripples every later step, permanently:
        // measured on QPLIB_2823 (LCQ, 283 equality quadratic rows), pen
        // hit 1.3e9 by iteration 3 and every accepted step for the next 497
        // was microscopic enough that ep DRIFTED UP (21 -> 34), not down.
        // The exact-penalty descent guarantee this is meant to buy (N&W
        // Thm 18.2) needs nu_pen > ||y*||_inf at the TRUE point, not at a
        // transient bad iterate; this code never claims that guarantee
        // (status is always Feasible, never Optimal) so trading it for
        // "the search can still move" is the right side of that tradeoff.
        constexpr f64 kNuPenCap = 1e8;
        f64 ymax = 0.0;
        for (core::Index i = 0; i < m; ++i) ymax = std::max(ymax, std::fabs(y[sz(i)] + dy[sz(i)]));
        nu_pen = std::min(kNuPenCap, std::max(nu_pen, 1.1 * ymax + 1e-6));
        if (dbar + nu_pen * dinf >= 0.0 && dinf < 0.0)
            nu_pen = std::min(kNuPenCap, std::max(nu_pen, -2.0 * dbar / dinf + 1e-6));
        const f64 dphi = dbar + nu_pen * dinf;
        f64 inf0 = 0.0;
        const f64 phi0 = merit(z, mu, nu_pen, inf0);
        f64 alpha = ap;
        std::vector<f64> ztrial(sz(N));
        bool accepted = false;
        for (int bt = 0; bt < 40; ++bt) {
            for (core::Index c = 0; c < N; ++c)
                ztrial[sz(c)] = (c >= n && fixed_w_[sz(c - n)]) ? z[sz(c)] : z[sz(c)] + alpha * dz[sz(c)];
            f64 inft = 0.0;
            const f64 phit = merit(ztrial, mu, nu_pen, inft);
            const bool armijo = dphi < 0.0 ? phit <= phi0 + eta_armijo * alpha * dphi : phit < phi0;
            if (std::isfinite(phit) && armijo) { accepted = true; break; }
            alpha *= 0.5;
        }
        if (!accepted) {
            // Take the (tiny) last step and make the next Newton step more
            // conservative through the Hessian shift; if that keeps failing,
            // maybe_restart takes over (see its comment above the loop).
            ++ls_fail;
            delta_last = std::max(delta_last * 10.0, delta_w_0);
            if (ls_fail >= 10) {
                if (maybe_restart("line search failed 10 times in a row")) continue;
                break;
            }
        } else {
            ls_fail = 0;
        }
        // A step this line search calls "accepted" can still be a stall the
        // rejection counter above never sees: fraction-to-boundary (ap, set
        // before backtracking even starts) can itself be ~1e-12 when one
        // column's slack is nearly exhausted, and a step that small usually
        // satisfies Armijo trivially (phit ~ phi0).  Measured on QPLIB_8815
        // (QCD, 30010 vars): alpha pinned at 9e-13 for 14 straight ACCEPTED
        // iterations, ls_fail never leaving 0, while delta_w climbed
        // geometrically without ever recovering a useful step. Track tiny
        // alpha on its own and restart once it has persisted.
        constexpr f64 kTinyAlpha = 1e-8;
        if (alpha < kTinyAlpha) ++tiny_alpha_count; else tiny_alpha_count = 0;
        if (tiny_alpha_count >= 5) {
            if (maybe_restart("step length collapsed (fraction-to-boundary stall)")) continue;
            break;
        }
        for (core::Index c = 0; c < N; ++c)
            if (!(c >= n && fixed_w_[sz(c - n)])) z[sz(c)] += alpha * dz[sz(c)];
        for (core::Index i = 0; i < m; ++i) y[sz(i)] += alpha * dy[sz(i)];
        for (core::Index c = 0; c < N; ++c) {
            lam[sz(c)] += ad * dlam[sz(c)];
            nu[sz(c)] += ad * dnu[sz(c)];
        }
        // Keep each bound multiplier within kappa_sigma of mu / s (Waechter & Biegler
        // eq. 16): the primal-dual Hessian then cannot drift arbitrarily far
        // from the primal barrier Hessian.
        slacks(z, sl, su);
        for (core::Index c = 0; c < N; ++c) {
            if (has_l_[sz(c)])
                lam[sz(c)] = std::clamp(lam[sz(c)], mu / (kappa_sigma * sl[sz(c)]), kappa_sigma * mu / sl[sz(c)]);
            if (has_u_[sz(c)])
                nu[sz(c)] = std::clamp(nu[sz(c)], mu / (kappa_sigma * su[sz(c)]), kappa_sigma * mu / su[sz(c)]);
        }
        if (verbose_)
            std::fprintf(stderr, "local it=%d mu=%.2e ep=%.2e ed=%.2e ec=%.2e dw=%.1e a=%.2e ad=%.2e pen=%.1e obj=%.8e\n",
                         it + 1, mu, ep, ed, ec, delta_w, alpha, ad, nu_pen, objective(xv));
        if (it + 1 == max_it) res.reason = "iteration limit";
    }
    res.x.assign(z.begin(), z.begin() + n);
    res.y = y;
    return res;
}

// JOINT feasibility solve for the bilinear-equality-heavy rows the barrier
// Newton step above stalls on (LCQ's "far" bucket:
// QPLIB_2823 and its 22 siblings never got within 1e-2 of feasible from any
// of 200 barrier starts -- see the measurement notes).  The
// barrier step's direction is always filtered through the dual coupling
// H = Q0 - sum y_t Q_t and the l1 penalty merit; this instead runs
// Levenberg-Marquardt directly on the row-residual vector
//   r_i(x) = g_i(x) - rl_i   if g_i(x) < rl_i
//          = g_i(x) - ru_i   if g_i(x) > ru_i
//          = 0               otherwise (row already satisfied),
// i.e. classic nonlinear least squares on min 0.5||r(x)||^2 -- undistracted
// by any objective or dual variable, the same reason the feasibility-only
// restoration engine (kf/feas_ipm, see solve_qcqp_local below) helps.
//
// TRUST-REGION DAMPING CONTROL.  The radius update on an accepted step,
//   radius /= max(1/3, 1 - (2*step_quality - 1)^3),  decrease_factor = 2,
// and on a rejected step,
//   radius /= decrease_factor,  decrease_factor *= 2,
// is Nielsen's damping-parameter rule (H.B. Nielsen, "Damping Parameter in
// Marquardt's Method", IMM Tech. Report 1999-05, Technical University of
// Denmark) -- control logic, not linear algebra, so it survives the switch
// below unchanged.
//
// A DELIBERATE DEPARTURE: the standard formulation forms an explicit
// Jacobian and solves the
// per-column-scaled augmented system (J'J + diag(clamp(diag(J'J),lo,hi))/
// radius) dx = -J'r with a pluggable direct/iterative LinearSolver.  SOR's
// Jacobian here is matrix-free by the same design choice as everywhere
// else in this file (jac/jac_t; see eval_rows/jac/jac_t above) -- an
// explicit J for a few-hundred-row instance with hundreds of bilinear
// terms is exactly the kind of dense-looking-from-sparse-parts object this
// codebase avoids materialising.  So the damped normal equations here are
// (J'J + I/radius) dx = -J'r -- UNSCALED (Levenberg's 1944 original
// version, not Marquardt's 1963 per-column-scaled one: "A Method for the
// Solution of Certain Non-Linear Problems in Least Squares", Quart. Appl.
// Math. 2, 1944) -- solved approximately with plain conjugate gradient
// (Hestenes & Stiefel 1952) on the two matrix-free operators, a textbook
// substitution (Nocedal & Wright, Numerical Optimization 2nd ed., s.5.1)
// for an explicit-Jacobian linear solve, not itself taken from
// anywhere. Row activity (which rows are outside their range, hence
// contribute to r and J) is refixed at the start of every outer LM
// iteration (an active-set relinearisation, cheap here since eval_rows is
// already O(nnz)); CG only ever sees the CURRENT active set for that
// iteration's inner solve.
LocalRun LocalIpm::run_feasibility_lm(std::vector<f64> x, int max_it, Clock::time_point deadline,
                                      const std::function<void(const std::vector<f64>&)>& on_accept) {
    LocalRun res;
    const core::Index n = n_, m = m_;
    auto sqnorm = [](const std::vector<f64>& v) {
        f64 s = 0.0;
        for (f64 e : v) s += e * e;
        return s;
    };
    auto clamp_box = [&](std::vector<f64>& xx) {
        for (core::Index j = 0; j < n; ++j) {
            if (has_l_[sz(j)] && xx[sz(j)] < lo(j)) xx[sz(j)] = lo(j);
            if (has_u_[sz(j)] && xx[sz(j)] > hi(j)) xx[sz(j)] = hi(j);
        }
    };
    clamp_box(x);
    std::vector<f64> gx, r(sz(m));
    std::vector<std::uint8_t> active(sz(m), 0);
    auto residual = [&](const std::vector<f64>& xx) {
        eval_rows(xx, gx);   // also forms qix_ at xx, which jac/jac_t below read
        for (core::Index i = 0; i < m; ++i) {
            const f64 rl = k_.rl[sz(i)], ru = k_.ru[sz(i)];
            f64 ri = 0.0;
            std::uint8_t a = 0;
            if (std::isfinite(rl) && gx[sz(i)] < rl) { ri = gx[sz(i)] - rl; a = 1; }
            else if (std::isfinite(ru) && gx[sz(i)] > ru) { ri = gx[sz(i)] - ru; a = 1; }
            r[sz(i)] = ri;
            active[sz(i)] = a;
        }
    };
    residual(x);
    f64 cost = 0.5 * sqnorm(r);
    // Standard trust-region strategy defaults:
    // initial_trust_region_radius = 1e4, max_trust_region_radius = 1e32.
    f64 radius = 1e4, decrease_factor = 2.0;
    constexpr f64 kMaxRadius = 1e32, kMinRadius = 1e-64;
    constexpr f64 kMinRelativeDecrease = 1e-3;   // standard min_relative_decrease
    constexpr f64 kGradTol = 1e-12;
    std::vector<f64> jr(sz(n)), jv(sz(m)), masked(sz(m)), dx(sz(n));
    std::vector<f64> cg_r, cg_p, cg_ap(sz(n));
    std::vector<f64> xtrial(sz(n)), dxc(sz(n)), jdx(sz(m));
    for (int it = 0; it < max_it; ++it) {
        res.iterations = it + 1;
        if (Clock::now() > deadline) { res.reason = "time limit"; break; }
        if (cost <= 1e-28) { res.converged = true; res.reason = "residual ~0 (scaled)"; break; }
        const f64 lambda = 1.0 / radius;
        auto matvec = [&](const std::vector<f64>& v, std::vector<f64>& out) {
            jac(v, jv);
            for (core::Index i = 0; i < m; ++i) masked[sz(i)] = active[sz(i)] ? jv[sz(i)] : 0.0;
            jac_t(masked, out);
            for (core::Index j = 0; j < n; ++j) out[sz(j)] += lambda * v[sz(j)];
        };
        jac_t(r, jr);   // r is already 0 on inactive rows: jr = J_active' r
        f64 gmax = 0.0;
        for (f64 v : jr) gmax = std::max(gmax, std::fabs(v));
        if (gmax <= kGradTol) { res.reason = "stationary point of the residual (not necessarily feasible)"; break; }
        // CG on (J_active'J_active + I/radius) dx = -jr, dx0 = 0.
        dx.assign(sz(n), 0.0);
        cg_r.assign(sz(n), 0.0);
        for (core::Index j = 0; j < n; ++j) cg_r[sz(j)] = -jr[sz(j)];
        cg_p = cg_r;
        f64 rs_old = sqnorm(cg_r);
        const f64 rs_stop = std::max(1e-28, 1e-10 * rs_old);
        const int cg_max = std::clamp<int>(static_cast<int>(n), 10, 200);
        for (int c = 0; c < cg_max && rs_old > rs_stop; ++c) {
            matvec(cg_p, cg_ap);
            f64 pap = 0.0;
            for (core::Index j = 0; j < n; ++j) pap += cg_p[sz(j)] * cg_ap[sz(j)];
            if (!(pap > 0.0) || !std::isfinite(pap)) break;
            const f64 alpha = rs_old / pap;
            for (core::Index j = 0; j < n; ++j) {
                dx[sz(j)] += alpha * cg_p[sz(j)];
                cg_r[sz(j)] -= alpha * cg_ap[sz(j)];
            }
            const f64 rs_new = sqnorm(cg_r);
            if (!std::isfinite(rs_new)) break;
            const f64 beta = rs_new / rs_old;
            for (core::Index j = 0; j < n; ++j) cg_p[sz(j)] = cg_r[sz(j)] + beta * cg_p[sz(j)];
            rs_old = rs_new;
        }
        bool finite_dx = true;
        for (f64 v : dx) finite_dx = finite_dx && std::isfinite(v);
        if (!finite_dx) { res.reason = "non-finite LM step"; break; }
        xtrial = x;
        for (core::Index j = 0; j < n; ++j) xtrial[sz(j)] += dx[sz(j)];
        clamp_box(xtrial);
        for (core::Index j = 0; j < n; ++j) dxc[sz(j)] = xtrial[sz(j)] - x[sz(j)];
        // Predicted reduction of 0.5||r||^2 under the CURRENT linearisation
        // (active set fixed at this iteration's start), from the CLAMPED
        // step so it's judged against what the step actually did, not the
        // unclamped Newton step CG solved for.
        jac(dxc, jdx);
        f64 dot_r_jdx = 0.0, jdx_sq = 0.0;
        for (core::Index i = 0; i < m; ++i)
            if (active[sz(i)]) { dot_r_jdx += r[sz(i)] * jdx[sz(i)]; jdx_sq += jdx[sz(i)] * jdx[sz(i)]; }
        const f64 predicted_reduction = -(dot_r_jdx + 0.5 * jdx_sq);
        residual(xtrial);   // now r/active/gx/qix_ are AT xtrial regardless of accept/reject below
        const f64 cost_trial = 0.5 * sqnorm(r);
        const f64 actual_reduction = cost - cost_trial;
        const bool valid = predicted_reduction > 0.0 && std::isfinite(cost_trial);
        const f64 rho = valid ? actual_reduction / predicted_reduction : -1.0;
        if (verbose_)
            std::fprintf(stderr, "lm it=%d cost=%.6e cost_trial=%.6e rho=%.3e radius=%.3e gmax=%.3e\n",
                         it + 1, cost, cost_trial, rho, radius, gmax);
        if (valid && rho > kMinRelativeDecrease) {
            x = xtrial;
            cost = cost_trial;
            // Nielsen's rule, accepted step (IMM Tech. Report 1999-05).
            radius = std::min(kMaxRadius, radius / std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3)));
            decrease_factor = 2.0;
            if (on_accept) on_accept(x);
        } else {
            // Rejected: r/active/gx/qix_ were just overwritten by the
            // probe at xtrial above -- restore them at the retained x
            // before the next iteration reads r or calls jac/jac_t (which
            // depends on qix_ having been formed AT x, not xtrial).
            residual(x);
            // Nielsen's rule, rejected step (IMM Tech. Report 1999-05).
            radius /= decrease_factor;
            decrease_factor *= 2.0;
            if (radius < kMinRadius) { res.reason = "trust region collapsed"; break; }
        }
        if (it + 1 == max_it) res.reason = "iteration limit";
    }
    res.x = x;
    return res;
}

// RESTORATION PHASE (Waechter & Biegler 2006 s.3.3): the subproblem is
// equation 29a/31a of that paper and the exit test equation 18a/18b.
//
// The published method drives the subproblem as a separate recursive solve
// carrying its own filter, warm-start and output machinery.  None of that
// fits this engine's matrix-free, single-Core design, so what is taken is
// the MATH alone: the
// subproblem is expressed as an AUGMENTED Core (original x, plus per-row
// p_i/n_i slacks) handed to this engine's OWN LocalIpm -- the identical
// "problem transformation, not a new solver" pattern already used for `kf`
// (the feasibility-only engine, solve_qcqp_local below) -- and the exit
// test is expressed as a plain comparison against the single
// (theta_ref, phi_ref) pair at restoration's entry point, since this file
// has no running FILTER (a growing set of prior (theta,phi) pairs) to check
// against, only the l1 merit line search above.  That is a RELAXATION of
// the paper's own test (checking against one pair can only accept a point the
// full filter might still reject for cycling back near an old iterate,
// never the reverse) -- an honest, documented simplification, not a
// silent narrowing of what "acceptable" means.
//
// Subproblem (equation 29a, with the constraint terms of 31a):
//   min_{x,p,n}  rho * sum_i (p_i + n_i)  +  (Eta(mu)/2) sum_j dr_j^2 (x_j - xref_j)^2
//   s.t.  g_i(x) + n_i - p_i - w_i = 0,   rl_i <= w_i <= ru_i,   p_i, n_i >= 0,
//         l_j <= x_j <= u_j   (the ORIGINAL box, unchanged)
// rho = 1e3 (the penalty parameter), Eta(mu) = eta_factor*sqrt(mu) with
// eta_factor = 1 (the proximity weight), and dr_j = 1 / max(1, |xref_j|).
// p_i/n_i absorb
// whatever row violation the real objective's search could not resolve;
// minimising their l1 sum while staying close to xref (the proximity term)
// is exactly the mechanism Waechter & Biegler design to find a NEARBY
// feasible point rather than an arbitrary one.  This is a genuine QCQP in
// SOR's own Core representation (same quadratic rows on the original x
// columns, unchanged; p/n only ever appear linearly), so the existing
// LocalIpm -- inertia correction, delta_c, the merit line search, all of
// it -- applies to it completely unmodified; nothing new was written for
// "how to solve this problem", only what problem to hand the same solver.
//
// RHO WARM-UP.
// Earlier work left the far LCQ bucket (12 instances, theta_ref 35..4.2e5 at
// entry) unmoved: restoration visibly runs on them (iteration counts
// change) but the endpoint never beats whatever best_any_viol the run
// already had (the measurement notes).
// The published infeasibility-detection variant turns out to NOT touch
// rho or Eta at all -- on the
// first restoration call of a run, when already meaningfully infeasible,
// it tightens `required_infeasibility_reduction` (a filter-style ACCEPT
// gate) from its default 0.9 down to 1e-3, i.e. it forces MANY MORE
// Newton iterations before letting restoration return, specifically to
// get an early, reliable signal on whether the problem is genuinely
// infeasible.  Porting that gate verbatim would make restoration calls
// use more of their existing 500ms cap on average -- the identical "more
// time in restoration" trade against start diversity was already
// measured net-negative three separate ways (flat cap raise, earlier/
// more-frequent entry, severity-gated cap; the measurement notes
// the item-3 closure).  So that gate is deliberately NOT ported.
// What follows instead is SOR's OWN conditioning heuristic, in the
// direction the own notes speculated ("start rho smaller when
// theta_ref is itself enormous... without costing any extra wall time"):
// at the fixed rho=1e3 above, a row off by theta_ref ~ 1e5 needs p_i or
// n_i to ALSO reach ~1e5 to satisfy the artificial equality, while p_i/
// n_i start pushed near 0 by init_state and carry mu's log-barrier
// curvature (mu/p_i^2) -- a many-orders-of-magnitude gap between the
// penalty's fixed strength and the natural scale of the correction it is
// asking the very first Newton step to make, the classic ill-conditioned
// start of an exact-penalty method with a too-large fixed penalty
// (Nocedal & Wright, ch. 17-19: penalty/multiplier methods are usually
// warmed up, not started at full strength).  So kRho SOFTENS (never below
// a floor, never above the unmodified default) on ONLY the first
// restoration call of a start when theta_ref is already large; every
// later call in the same start (theta_ref hopefully smaller after real
// progress) uses the unmodified 1e3, and every start whose first stall
// happens at a small theta_ref (the near bucket, and ordinary LMQ calls)
// sees byte-identical behaviour to before this change. No cap, entry-
// timing, or target width touched anywhere -- same 500ms deadline, same
// 40-iteration stall trigger, same lenient on_it exit test below.
LocalRun LocalIpm::run_restoration(const std::vector<f64>& xref, f64 mu_main,
                                   Clock::time_point deadline, f64 tol,
                                   int count_restorations) {
    LocalRun res;
    const core::Index n = n_, m = m_;
    f64 theta_ref, phi_ref;
    theta_phi(xref, mu_main, theta_ref, phi_ref);
    if (theta_ref <= tol) {
        res.x = xref; res.converged = true; res.final_theta = theta_ref;
        res.reason = "already feasible, restoration not needed";
        return res;
    }

    Core kr;
    kr.n = n + 2 * m;
    kr.m = m;
    kr.rl = k_.rl; kr.ru = k_.ru;
    kr.l.assign(sz(kr.n), 0.0);
    kr.u.assign(sz(kr.n), kInf);
    for (core::Index j = 0; j < n; ++j) { kr.l[sz(j)] = k_.l[sz(j)]; kr.u[sz(j)] = k_.u[sz(j)]; }
    kr.quad = k_.quad;   // identical rows, referencing only the original x columns (0..n)
    kr.offset = 0.0;

    // A: original entries unchanged, plus -1 on column n+i (p_i) and +1 on
    // column n+m+i (n_i) for row i -- g_i(x) + n_i - p_i - w_i = 0, matching
    // the subproblem's "orig_c + n_c - p_c" convention exactly.
    std::vector<core::Index> rows, cols;
    std::vector<f64> vals;
    {
        const auto& rp = k_.A.pattern.row_ptr();
        const auto& ci = k_.A.pattern.col_idx();
        for (core::Index i = 0; i < m; ++i) {
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
                rows.push_back(i); cols.push_back(ci[sz(t)]); vals.push_back(k_.A.vals[sz(t)]);
            }
            rows.push_back(i); cols.push_back(n + i); vals.push_back(-1.0);
            rows.push_back(i); cols.push_back(n + m + i); vals.push_back(1.0);
        }
    }
    kr.A = sparse::from_triplets(m, kr.n, rows, cols, vals);

    // Objective: rho*(p+n) linear; the proximity term expanded (dropping
    // the xref_j^2 constant, which cannot move the minimiser) into a
    // diagonal quadratic + linear pair on the ORIGINAL x columns only.
    constexpr f64 kRhoDefault = 1e3, kEtaFactor = 1.0;
    // Warm-up window: theta_ref <= kWarmupThetaThresh gets the unmodified
    // default (the near bucket, 7.3e-6..0.28, never sees any change here).
    // Above it, kRho = kRhoDefault * kWarmupThetaThresh / theta_ref, floored
    // at kWarmupFloor -- continuous at the threshold (equals kRhoDefault
    // exactly when theta_ref == kWarmupThetaThresh), softening smoothly as
    // theta_ref grows past it, never below 100x under the default.
    constexpr f64 kWarmupThetaThresh = 10.0, kWarmupFloor = 10.0;
    const f64 kRho = (count_restorations == 1 && theta_ref > kWarmupThetaThresh)
                          ? std::max(kWarmupFloor, kRhoDefault * kWarmupThetaThresh / theta_ref)
                          : kRhoDefault;
    const f64 eta = kEtaFactor * std::sqrt(std::max(mu_main, 1e-12));
    kr.c.assign(sz(kr.n), 0.0);
    kr.qcol.assign(sz(kr.n), {});
    kr.qdiag.assign(sz(kr.n), 0.0);
    for (core::Index j = 0; j < n; ++j) {
        const f64 drj = 1.0 / std::max(1.0, std::fabs(xref[sz(j)]));
        const f64 h = eta * drj * drj;
        kr.qcol[sz(j)].push_back({j, h});
        kr.qdiag[sz(j)] = h;
        kr.c[sz(j)] = -h * xref[sz(j)];
    }
    for (core::Index i = 0; i < m; ++i) { kr.c[sz(n + i)] = kRho; kr.c[sz(n + m + i)] = kRho; }

    LocalIpm resto_ipm(kr, verbose_, rng_(), /*nested=*/true);
    std::vector<f64> x0(sz(kr.n), 0.0);
    for (core::Index j = 0; j < n; ++j) x0[sz(j)] = xref[sz(j)];   // p, n start at 0, pushed interior by init_state

    // Exit test ("equation 18"): stop the instant a resto ITERATE (not
    // necessarily its own converged optimum) is acceptable back in the
    // ORIGINAL problem's units -- sufficient decrease in theta, or in the
    // barrier objective phi weighted by theta_ref, exactly the filter
    // acceptance pair, checked against the single (theta_ref, phi_ref) this
    // restoration call was entered with (see the relaxation note above).
    bool accepted = false;
    std::vector<f64> accepted_x;
    f64 accepted_theta = kInf;
    auto on_it = [&](const std::vector<f64>& xv, const std::vector<f64>&) -> bool {
        std::vector<f64> xo(xv.begin(), xv.begin() + n);
        f64 theta_t, phi_t;
        theta_phi(xo, mu_main, theta_t, phi_t);
        constexpr f64 kGammaTheta = 1e-5, kGammaPhi = 1e-8;
        if (theta_t <= (1.0 - kGammaTheta) * theta_ref || phi_t <= phi_ref - kGammaPhi * theta_ref) {
            accepted = true; accepted_x = xo; accepted_theta = theta_t;
            return true;
        }
        return false;
    };
    const auto rr = resto_ipm.run(x0, 3000, tol, deadline, on_it);
    if (accepted) {
        res.x = accepted_x; res.converged = true; res.final_theta = accepted_theta;
        res.reason = "restoration: filter-acceptable point";
        return res;
    }
    // Never passed the exit test (ran out of iterations/time, or its own
    // barrier solve gave up) -- still hand back its endpoint's x part and
    // the theta actually reached there.  The caller (maybe_restart) decides
    // whether that is good enough to adopt; this function never claims
    // acceptance it did not earn.
    res.x.assign(rr.x.begin(), rr.x.begin() + n);
    f64 phi_unused;
    theta_phi(res.x, mu_main, res.final_theta, phi_unused);
    res.converged = false;
    res.reason = std::string("restoration: ") + rr.reason;
    return res;
}

}  // namespace

core::RawResult solve_qcqp_local(const QcqpProblem& p0, const QcqpLocalOptions& opts,
                                 QcqpLocalDiagnostics& diag) {
    const auto t0 = Clock::now();
    diag = QcqpLocalDiagnostics{};
    core::RawResult raw;
    raw.engine = "qcqp_local";
    raw.backend = "cpu";
    auto fail = [&](core::Status s, const std::string& why) {
        raw.proposed_status = s;
        raw.termination_reason = why;
        diag.reason = why;
        diag.total_ms = ms_since(t0);
        return raw;
    };
    const auto& lp = p0.qp.linear;
    if (lp.maximize) return fail(core::Status::Unsupported, "local QCQP solver takes a minimisation");
    try { p0.validate(); } catch (const std::exception& e) {
        return fail(core::Status::Unsupported, std::string("invalid QCQP: ") + e.what());
    }
    const core::Index n0 = lp.n_cols();
    QpOptions bo;   // builder: Ruiz scaling as the IPM does
    ipmc::QcqpBuilt B;
    ipmc::build_qcqp_core(p0, nullptr, bo, B);
    const Core& k = B.k;
    const auto deadline = opts.time_limit_s > 0.0
        ? t0 + std::chrono::microseconds(static_cast<long long>(opts.time_limit_s * 1e6))
        : Clock::time_point::max();

    // Original-units judge: rows and bounds (integrality is the caller's).
    auto to_original = [&](const std::vector<f64>& xc) {
        std::vector<f64> xo(sz(n0));
        for (core::Index j = 0; j < n0; ++j)
            xo[sz(j)] = (B.is_fixed[sz(j)] ? B.xfix[sz(j)] : xc[sz(B.new_col[sz(j)])]) * B.sc.dc[sz(j)];
        return xo;
    };
    std::vector<f64> best_x, best_any;
    // Core-space twin of best_any -- kept ONLY so the round-8 final-polish
    // phase below (see kNearGateViol) can hand the actual best point found
    // so far to run_feasibility_lm without inverting to_original() (lossy:
    // fixed columns are dropped from core space, see B.is_fixed above).
    // Every call site of judge() already has the matching core-space vector
    // in hand (xc from on_it/lm_on_accept, r.x/lr.x after a run), so this
    // costs nothing extra to capture.
    std::vector<f64> best_any_core;
    f64 best_obj = kInf, best_viol = kInf, best_any_viol = kInf;
    auto judge = [&](const std::vector<f64>& xo, const std::vector<f64>& xc, int start) {
        const auto e = evaluate_qcqp(p0, xo);
        const f64 viol = std::max(e.max_row_violation, e.max_bound_violation);
        if (!std::isfinite(viol) || !std::isfinite(e.objective_min)) return;
        if (viol <= opts.feas_tol) {
            if (e.objective_min < best_obj) {
                best_obj = e.objective_min;
                best_viol = viol;
                best_x = xo;
                diag.best_start = start;
            }
        } else if (best_x.empty() && viol < best_any_viol) {
            best_any_viol = viol;
            best_any = xo;
            best_any_core = xc;
        }
    };

    LocalIpm ipm(k, opts.verbose, opts.seed);
    // FEASIBILITY-ONLY RESTORATION FALLBACK.  Coverage (finding SOME
    // original-units feasible point) is a harder and more important goal
    // than objective quality for this engine: a linear/convex objective
    // gains nothing from a clever direction if the search never lands in
    // the feasible region at all, and QPLIB's quadratic-equality-heavy
    // classes (LCQ especially) stall the real-objective search for exactly
    // that reason (see the delta_c and no-progress-stall comments in
    // LocalIpm::run above) -- the Newton direction is dominated by the
    // dual/Hessian coupling -sum y_t Q_t an objective with real duals
    // induces, not by a genuine path toward g(x) = w.  kf is the SAME
    // rows/quad/bounds with the real objective zeroed (c, qcol, qdiag),
    // so feas_ipm below reuses every mechanism ipm has -- inertia/delta_c
    // correction, the stall restart, the line search -- unchanged, just
    // searching for ANY constraint-satisfying point, undistracted by an
    // objective. Every point either engine reaches is judged (below)
    // against the REAL objective via evaluate_qcqp on the original p0, so
    // a feasibility-only start that happens to land somewhere with a good
    // true objective is never wasted -- and once any start (from either
    // engine) has found ONE feasible point, later starts switch back to
    // the real objective to work on quality instead of continuing to
    // search blind.
    Core kf = k;
    std::fill(kf.c.begin(), kf.c.end(), 0.0);
    for (auto& col : kf.qcol) col.clear();
    std::fill(kf.qdiag.begin(), kf.qdiag.end(), 0.0);
    kf.offset = 0.0;
    LocalIpm feas_ipm(kf, opts.verbose, opts.seed + 0x9E3779B97F4A7C15ULL);
    // Rows with no quadratic term and a fixed range: the Kaczmarz
    // pre-projection below only ever touches these (see its comment).
    const auto kLinEqRows = linear_eq_rows(k);
    // computed ONCE (not per start) when the caller opted in and
    // did not already supply its own x0 -- see mccormick_core_start above
    // and QcqpLocalOptions::mccormick_start.  Empty (silently) on any
    // failure; the ordinary origin/random start below is unchanged.
    const std::vector<f64> mc_x0 = (opts.mccormick_start && opts.x0.empty())
        ? mccormick_core_start(k, deadline) : std::vector<f64>{};
    std::mt19937_64 rng(opts.seed);
    const int starts = std::max(1, opts.starts);
    // The periodic best-point polish (see its definition below) only
    // re-fires once best_any_viol has genuinely improved past this floor --
    // a fixed point a prior attempt already gave up on must not be retried
    // every single start.
    f64 last_polish_attempt_viol = kInf;
    for (int s = 0; s < starts; ++s) {
        if (Clock::now() > deadline) break;
        // Start: the caller's point, else the McCormick relaxation point
        // (if asked for and available), else the origin (pushed inside the
        // box by the solver); later starts: uniform in the box, the box
        // clipped to +-10 (core units) around the default where unbounded.
        std::vector<f64> xs(sz(k.n), 0.0);
        if (!opts.x0.empty() && opts.x0.size() == sz(n0))
            for (core::Index jj = 0; jj < k.n; ++jj) {
                const core::Index j = k.col_of[sz(jj)];
                xs[sz(jj)] = opts.x0[sz(j)] / B.sc.dc[sz(j)];
            }
        else if (mc_x0.size() == sz(k.n))
            xs = mc_x0;
        if (s > 0)
            for (core::Index jj = 0; jj < k.n; ++jj) {
                const f64 a = std::isfinite(k.l[sz(jj)]) ? k.l[sz(jj)] : xs[sz(jj)] - 10.0;
                const f64 b = std::isfinite(k.u[sz(jj)]) ? k.u[sz(jj)] : xs[sz(jj)] + 10.0;
                std::uniform_real_distribution<f64> U(std::min(a, b), std::max(a, b));
                xs[sz(jj)] = U(rng);
            }
        // Pre-condition every start (including s == 0) by pulling it onto
        // the model's pure-linear equality rows first: it can only get
        // closer to a requirement that must hold regardless of which
        // engine runs next.  (A bilinear alternating Gauss-Seidel sweep --
        // holding every column but one fixed, solving each quadratic
        // equality row exactly for its best-conditioned column -- was
        // tried here too and measured OFF: on QPLIB_2823 it converges the
        // RAW preconditioned point to a WORSE global max-violation than
        // Kaczmarz alone (12.1 -> 21.3, undamped; damping to 0.3x the
        // step made no difference, same fixed point), and repeating it
        // never recovers -- zeroing one row's residual exactly routinely
        // walks the shared variable away from OTHER rows' feasibility,
        // and nothing here accounts for that global effect.  See
        // the measurement notes for the measurement; reverted.)
        kaczmarz_project(k, kLinEqRows, xs, 15);
        // Remaining budget shared evenly by the remaining starts -- capped
        // to at most kSliceDivisorCap-way division.  --starts is also how a
        // caller asks for "as many restarts as fit in the time limit"
        // (a large count); dividing literally by every one of those planned
        // starts would shrink each slice to a sliver long before the clock
        // runs out.  A start that finishes early (converged, or gave up
        // after its in-run restarts) just returns and the NEXT slice is
        // recomputed from what's actually left, so the cap only bounds the
        // minimum slice size, never the number of starts actually run.
        auto dl = deadline;
        if (opts.time_limit_s > 0.0) {
            constexpr int kSliceDivisorCap = 20;
            const auto left = deadline - Clock::now();
            const int divisor = std::max(1, std::min(starts - s, kSliceDivisorCap));
            dl = Clock::now() + left / divisor;
        }
        int cnt = 0;
        auto on_it = [&](const std::vector<f64>& xc, const std::vector<f64>&) -> bool {
            // Every 5th iterate is judged (O(nnz) each): a feasible iterate
            // is kept even if the start never converges.  This driver loop
            // never requests an early stop (that's run_restoration's own
            // internal on_iterate, a different closure -- see LocalIpm).
            if (++cnt % 5 == 0) judge(to_original(xc), xc, s);
            return false;
        };
        // s == 0 (the caller's point, or the origin) always gets a real-
        // objective attempt: it is often already feasible and cheap to
        // confirm.  Every later start runs the feasibility-only restoration
        // engine until SOME start (either engine) has found an
        // original-units feasible point; once coverage is secured, later
        // starts go back to the real objective to search for a better one.
        LocalIpm& engine = (s > 0 && best_x.empty()) ? feas_ipm : ipm;
        const auto r = engine.run(xs, opts.max_iterations, opts.tol, dl, on_it);
        diag.iterations += static_cast<std::uint64_t>(r.iterations);
        ++diag.starts_run;
        const auto xo = to_original(r.x);
        const f64 before = best_obj;
        judge(xo, r.x, s);
        // JOINT feasibility polish (2026-09-22 scope:
        // LCQ coverage only).  Hand the barrier step's endpoint to
        // Levenberg-Marquardt on the row-residual vector -- see
        // LocalIpm::run_feasibility_lm above for the method and
        // the radius rule it uses.  Skipped
        // once coverage is secured (best_x non-empty): this phase only
        // targets feasibility, and a start that already converged the real
        // objective has nothing left for a feasibility-only polish to fix.
        if (best_x.empty() && Clock::now() < deadline) {
            // 300 iterations was measured too tight a cap on QPLIB_3177: it
            // hit "iteration limit" every time well inside the ms budget
            // below, mid-improvement (cost still falling, no sign of having
            // plateaued) -- 20000 makes the wall-clock budget the real
            // limiter instead, matching how the barrier step above is
            // governed (max_iterations is generous; the deadline decides).
            // 1000ms was measured to cost QPLIB_2590/2693 their coverage:
            // both only completed 28 starts_run in 60s (vs. up to 200
            // elsewhere) with the 1000ms cap, down from whatever they got
            // without LM at all -- fewer independent random starts for two
            // instances whose feasibility apparently depended on start-
            // count diversity, not single-trajectory refinement. 250ms
            // still gave ~1900 LM iterations on QPLIB_3177 in the earlier
            // trace (7605 iterations/1000ms), which is what flipped
            // QPLIB_2698/2758 -- trying that as a smaller tax on start
            // count.
            const auto lm_dl = std::min(deadline, Clock::now() + std::chrono::milliseconds(250));
            int lm_cnt = 0;
            auto lm_on_accept = [&](const std::vector<f64>& xc) {
                // Every 20th ACCEPTED step (not every iteration -- most LM
                // iterations near a plateau are rejected probes that don't
                // move x at all, see the run_feasibility_lm chattering
                // comment): judge() is O(nnz) on the ORIGINAL problem, LM
                // iterations can be sub-millisecond, thousands of them fit
                // in the 1s budget above -- judging every single accept
                // would make the judge overhead dominate the polish itself.
                if (++lm_cnt % 20 == 0) judge(to_original(xc), xc, s);
            };
            const auto lr = engine.run_feasibility_lm(r.x, 20000, lm_dl, lm_on_accept);
            diag.iterations += static_cast<std::uint64_t>(lr.iterations);
            judge(to_original(lr.x), lr.x, s);
            if (opts.verbose)
                std::fprintf(stderr, "qcqp_local: start %d: LM polish %d iterations, %s\n", s,
                             lr.iterations, lr.reason.c_str());
        }
        if (best_obj < before && r.converged) diag.kkt_converged = true;
        if (opts.verbose)
            std::fprintf(stderr, "qcqp_local: start %d: %d iterations, %s; best feasible %.10e\n", s,
                         r.iterations, r.reason.c_str(), best_obj);
        // PERIODIC POLISH ON THE GLOBAL BEST POINT (near-feasible
        // LCQ coverage).  Diagnosed first (see the measurement notes
        // measurement notes): for the near-feasible bucket the violation is
        // NOT concentrated in one or two rows -- QPLIB_2430 has 92/92 rows
        // violated, QPLIB_2823 353/386, QPLIB_2703 519/800, all clustered in
        // a similar small-magnitude band -- so the right tool is exactly the
        // JOINT LM polish above (it already does a Gauss-Newton step over
        // every active row simultaneously). What it lacks is budget: the
        // block above gives EVERY start's own endpoint only a fresh 250ms
        // before the next random start discards it, even when that
        // endpoint is already the best point the whole run has found.
        // kNearGateViol is a generous, not a tight, cutoff: the far
        // bucket's best-ever violation is orders of magnitude above it
        // (1e1-1e5 in every sweep this project has run), so this can only
        // fire on a point already shaped like the near bucket.
        //
        // FIRST ATTEMPT (measured, reverted): give this ALL remaining
        // budget in one shot and break out of the starts loop entirely.
        // Full 52-sweep result: net +1 (27->28) -- gained 5 (2 of them
        // real near-bucket instances, 2540 and 3334, both driven to true
        // feasibility) but REGRESSED 4 that used to succeed on start
        // diversity alone (2416, 2650, 2693, 2698; QPLIB_2416.log etc.
        // showed total_ms well under the 60s budget with starts_run=4-5 --
        // the dedicated LM call hit "trust region collapsed" or a
        // stationary point WITHOUT reaching feasibility, in a few seconds,
        // and the unconditional break then threw away the rest of the
        // clock instead of falling back to more starts). Root cause: an
        // all-or-nothing commit to one point is exactly as risky as the
        // closed-off "bigger restoration cap" dead ends when the point
        // turns out to be a dead end itself -- the difference that makes
        // THIS lever work is spending a BOUNDED slice per attempt and
        // falling back to ordinary random-start diversity when it doesn't
        // pay off, not the single unconditional commit.  So: no break,
        // capped per-attempt time, and re-fire only once best_any_viol
        // has genuinely improved since the last attempt (a fixed point
        // that didn't move must not be retried every single start).
        constexpr f64 kNearGateViol = 1.0;
        constexpr int kMinStartsBeforePolish = 3;
        if (best_x.empty() && s >= kMinStartsBeforePolish && !best_any_core.empty() &&
            best_any_viol <= kNearGateViol && best_any_viol < last_polish_attempt_viol &&
            Clock::now() < deadline) {
            const f64 entry_viol = best_any_viol;
            last_polish_attempt_viol = entry_viol;
            const auto polish_dl = std::min(deadline, Clock::now() + std::chrono::milliseconds(3000));
            int fp_cnt = 0;
            auto fp_on_accept = [&](const std::vector<f64>& xc) {
                if (++fp_cnt % 20 == 0) judge(to_original(xc), xc, s);
            };
            const auto fr = engine.run_feasibility_lm(best_any_core, 500000, polish_dl, fp_on_accept);
            diag.iterations += static_cast<std::uint64_t>(fr.iterations);
            judge(to_original(fr.x), fr.x, s);
            if (opts.verbose)
                std::fprintf(stderr,
                             "qcqp_local: start %d: PERIODIC POLISH (entry viol %.6e <= gate) %d "
                             "iterations, %s; best viol now %.6e (feasible=%d)\n",
                             s, entry_viol, fr.iterations, fr.reason.c_str(),
                             best_x.empty() ? best_any_viol : best_viol, !best_x.empty());
            // No break: whether or not this closed the gap, later starts
            // still run (either more diversity, or -- once best_x is set
            // -- the real-objective engine for quality). A failed attempt
            // is cheap (capped at 3s) and last_polish_attempt_viol stops
            // it from being retried on the same stuck point.
        }
    }
    if (!best_x.empty()) {
        raw.x = best_x;
        raw.objective = best_obj;
        diag.objective = best_obj;
        diag.max_violation = best_viol;
        diag.feasible = true;
        raw.proposed_status = core::Status::Feasible;
        raw.proposed_level = core::ProofLevel::FeasibleOnly;
        raw.termination_reason = "local solution, feasible in original units (not proved optimal)";
        diag.reason = raw.termination_reason;
    } else {
        raw.x = best_any;
        diag.max_violation = best_any_viol;
        raw.proposed_status = core::Status::NoSolutionFound;
        raw.proposed_level = core::ProofLevel::None;
        raw.termination_reason = "no start reached an original-units feasible point";
        diag.reason = raw.termination_reason;
    }
    diag.total_ms = ms_since(t0);
    return raw;
}

}  // namespace sor::engines
