// SOR — FGMRES on a symmetric system, preconditioned by an LDL' factor of a
// REGULARISED copy of it.  Shared by the interior point and by polishing:
// both factor K + reg (for stability of the factorization) but need the
// solution of K itself.
//
// Why not plain iterative refinement: it contracts by roughly
// reg * ||K^-1|| per step, which nears or exceeds 1 exactly when it matters
// -- near an IPM optimum (refinement stalled at a relative residual ~1e-8
// and held QPLIB_10034's |rp| at 3e-5), and in polishing an almost-LP (on
// QPLIB_9002, Q's diagonal is ~1e-11 against a regularisation of 1e-8, a
// contraction of ~1e3: refinement diverges).  FGMRES preconditioned by the
// same factor (Arioli, Duff, Gratton & Pralet, SIAM J. Sci. Comput. 29(5),
// 2007) costs one factor solve and one product per iteration, like a
// refinement step, but converges whenever the preconditioned spectrum is
// clustered.  Implemented from the textbook algorithm (Saad, Iterative
// Methods for Sparse Linear Systems, 2nd ed., s.9.4.1).
#include "qp_common.hpp"
#include "sor/core/env_switches.hpp"

#include "sor/core/parallel.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace sor::engines::qpc {
namespace {
using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
}  // namespace

f64 fgmres_solve(const la::Ldlt& fac, const SymMul& mul, std::vector<f64>& rhs,
                 int max_krylov, f64 tol_rel, const char* phase) {
    const std::size_t nn = rhs.size();
    core::ThreadPool& pool = core::global_pool();
    // Fixed chunk size for deterministic dot product reductions (independent of thread count).
    const core::Offset chunk_size = 4096;

    // Timing instrumentation for preconditioner vs. matrix products.
    Clock::time_point solve_start = Clock::now();
    f64 prec_ms = 0.0, matvec_ms = 0.0;

    auto dot = [&](const std::vector<f64>& u, const std::vector<f64>& v) {
        // Always use deterministic_sum for bit-identical results at any thread count.
        // Small vectors use serial execution; large vectors parallelize within chunks.
        return core::deterministic_sum(pool, static_cast<core::Offset>(nn), chunk_size,
                                       [&](core::Offset i) { return u[i] * v[i]; });
    };
    auto resid = [&](const std::vector<f64>& x, std::vector<f64>& r) {
        std::vector<f64> kv;
        mul(x, kv);
        r.resize(nn);
        f64 rn = 0.0;
        for (std::size_t i = 0; i < nn; ++i) {
            r[i] = rhs[i] - kv[i];
            rn = std::max(rn, std::fabs(r[i]));
        }
        return rn;
    };
    f64 bn = 0.0, b2 = 0.0;
    for (f64 v : rhs) {
        bn = std::max(bn, std::fabs(v));
        b2 += v * v;
    }
    b2 = std::sqrt(b2);

    std::vector<f64> x0 = rhs, r0;
    {
        Clock::time_point t0 = Clock::now();
        fac.solve(x0);
        prec_ms += ms_since(t0);
    }
    f64 best = resid(x0, r0);
    std::vector<f64> best_x = x0;
    if (!std::isfinite(best)) best = std::numeric_limits<f64>::infinity();
    std::size_t kr = 0;
    if (best > 0.1 * tol_rel * (1.0 + bn) && std::isfinite(best)) {
        const f64 beta = std::sqrt(dot(r0, r0));
        const auto K = static_cast<std::size_t>(max_krylov);
        std::vector<std::vector<f64>> V, Z;
        std::vector<std::vector<f64>> H(K + 1, std::vector<f64>(K, 0.0));
        std::vector<f64> cs(K, 0.0), sn(K, 0.0), g(K + 1, 0.0);
        g[0] = beta;
        V.push_back(r0);
        for (auto& v : V[0]) v /= beta;
        kr = 0;
        for (; kr < K; ++kr) {
            std::vector<f64> zk = V[kr];
            {
                Clock::time_point t0 = Clock::now();
                fac.solve(zk);
                prec_ms += ms_since(t0);
            }
            bool finite = true;
            for (std::size_t i = 0; i < nn && finite; ++i) finite = std::isfinite(zk[i]);
            if (!finite) break;
            std::vector<f64> w;
            {
                Clock::time_point t0 = Clock::now();
                mul(zk, w);
                matvec_ms += ms_since(t0);
            }
            Z.push_back(std::move(zk));
            for (std::size_t i = 0; i <= kr; ++i) {   // modified Gram-Schmidt
                const f64 h = dot(w, V[i]);
                H[i][kr] = h;
                // Thread the elementwise update w[t] -= h * V[i][t]
                if (nn > 1000 && pool.size() > 1) {
                    const core::Offset nc = (static_cast<core::Offset>(nn) + chunk_size - 1) / chunk_size;
                    pool.parallel_for(nc, [&](core::Offset chunk, int) {
                        const core::Offset lo = chunk * chunk_size;
                        const core::Offset hi = lo + chunk_size < static_cast<core::Offset>(nn) ? lo + chunk_size : static_cast<core::Offset>(nn);
                        for (core::Offset t = lo; t < hi; ++t) w[t] -= h * V[i][t];
                    });
                } else {
                    for (std::size_t t = 0; t < nn; ++t) w[t] -= h * V[i][t];
                }
            }
            const f64 hn = std::sqrt(dot(w, w));
            H[kr + 1][kr] = hn;
            for (std::size_t i = 0; i < kr; ++i) {    // previous Givens rotations
                const f64 t = cs[i] * H[i][kr] + sn[i] * H[i + 1][kr];
                H[i + 1][kr] = -sn[i] * H[i][kr] + cs[i] * H[i + 1][kr];
                H[i][kr] = t;
            }
            const f64 den = std::hypot(H[kr][kr], H[kr + 1][kr]);
            if (!(den > 0.0)) break;
            cs[kr] = H[kr][kr] / den;
            sn[kr] = H[kr + 1][kr] / den;
            H[kr][kr] = den;
            H[kr + 1][kr] = 0.0;
            g[kr + 1] = -sn[kr] * g[kr];
            g[kr] = cs[kr] * g[kr];
            if (std::fabs(g[kr + 1]) <= tol_rel * (1.0 + b2) || !(hn > 0.0)) { ++kr; break; }
            V.emplace_back(std::move(w));
            for (auto& v : V.back()) v /= hn;
        }
        if (kr > 0) {
            std::vector<f64> coef(kr, 0.0);
            for (std::size_t ii = kr; ii-- > 0;) {
                f64 t = g[ii];
                for (std::size_t j = ii + 1; j < kr; ++j) t -= H[ii][j] * coef[j];
                coef[ii] = t / H[ii][ii];
            }
            std::vector<f64> x = x0, r;
            for (std::size_t j = 0; j < kr; ++j) {
                // Thread the elementwise accumulation x[t] += coef[j] * Z[j][t]
                if (nn > 1000 && pool.size() > 1) {
                    const core::Offset nc = (static_cast<core::Offset>(nn) + chunk_size - 1) / chunk_size;
                    pool.parallel_for(nc, [&](core::Offset chunk, int) {
                        const core::Offset lo = chunk * chunk_size;
                        const core::Offset hi = lo + chunk_size < static_cast<core::Offset>(nn) ? lo + chunk_size : static_cast<core::Offset>(nn);
                        for (core::Offset t = lo; t < hi; ++t) x[t] += coef[j] * Z[j][t];
                    });
                } else {
                    for (std::size_t t = 0; t < nn; ++t) x[t] += coef[j] * Z[j][t];
                }
            }
            // The Krylov estimate is a 2-norm; the TRUE max-norm residual
            // decides, and the plain factor solve is kept if it was better.
            const f64 rn = resid(x, r);
            if (rn < best) { best = rn; best_x.swap(x); }
        }
    }
    rhs.swap(best_x);

    // Report timing breakdown if significant time was spent (debug instrumentation).
    // The split (fac.solve vs mul) is: preconditioner ~70-72%, matrix product ~15-20%.
    // This is measured on QPLIB_10038 and QPLIB_8547; preconditioner is dominant.
    // Off unless SOR_FGMRES_PROFILE is set: the interior point calls this
    // dozens of times per solve, so an unconditional print would bury the
    // solver's own output.
    static const bool profile = core::env_switches().fgmres_profile;
    const f64 total_ms = ms_since(solve_start);
    if (profile && total_ms > 1.0 && (prec_ms > 0.0 || matvec_ms > 0.0)) {
        if (phase) {
            fprintf(stderr, "FGMRES [%s] %zu iters: total %.1f ms (precond %.1f ms, matvec %.1f ms, other %.1f ms)\n",
                    phase, kr, total_ms, prec_ms, matvec_ms, total_ms - prec_ms - matvec_ms);
        } else {
            fprintf(stderr, "FGMRES %zu iters: total %.1f ms (precond %.1f ms, matvec %.1f ms, other %.1f ms)\n",
                    kr, total_ms, prec_ms, matvec_ms, total_ms - prec_ms - matvec_ms);
        }
    }

    return best / (1.0 + bn);
}

}  // namespace sor::engines::qpc
