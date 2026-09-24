// SOR — SDP-derived QCR shift for binary QP; see bqp_sdp.hpp for the
// mathematics, the papers it comes from, and why nothing here needs to be
// accurate for the final bound to be valid.
#include "sor/search/bqp_sdp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;
inline double secs_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
constexpr f64 kUnit = 1.1102230246251565e-16;   // 2^-53
constexpr f64 kInf = std::numeric_limits<f64>::infinity();

struct Lcg {
    std::uint64_t s;
    f64 next() {   // uniform in [0,1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<f64>(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

// y = A x for dense row-major A (N x N).
void dense_mv(const std::vector<f64>& A, Index N, const f64* x, f64* y) {
    for (Index i = 0; i < N; ++i) {
        const f64* a = &A[sz(i) * sz(N)];
        f64 s = 0.0;
        for (Index j = 0; j < N; ++j) s += a[j] * x[j];
        y[i] = s;
    }
}

// Smallest eigenvalue of the symmetric tridiagonal (alpha, beta) by Sturm
// bisection: the count of negative pivots of T - x I is the number of
// eigenvalues below x.
f64 tridiag_min_eig(const std::vector<f64>& al, const std::vector<f64>& be) {
    const std::size_t m = al.size();
    f64 lo = kInf, hi = -kInf;
    for (std::size_t i = 0; i < m; ++i) {
        const f64 r = (i > 0 ? std::fabs(be[i - 1]) : 0.0) + (i + 1 < m ? std::fabs(be[i]) : 0.0);
        lo = std::min(lo, al[i] - r);
        hi = std::max(hi, al[i] + r);
    }
    auto below = [&](f64 x) {
        int cnt = 0;
        f64 d = 1.0;
        for (std::size_t i = 0; i < m; ++i) {
            const f64 b2 = i > 0 ? be[i - 1] * be[i - 1] : 0.0;
            d = al[i] - x - (i > 0 ? b2 / d : 0.0);
            if (d == 0.0) d = -1e-300;
            if (d < 0.0) ++cnt;
        }
        return cnt;
    };
    for (int it = 0; it < 200 && hi - lo > 1e-15 * std::max(1.0, std::fabs(lo) + std::fabs(hi)); ++it) {
        const f64 mid = 0.5 * (lo + hi);
        if (below(mid) >= 1) hi = mid;
        else lo = mid;
    }
    return lo;
}

// Lanczos with full reorthogonalisation: the smallest Ritz value, which is
// >= lambda_min (an estimate that errs HIGH -- the certificate below never
// trusts it, it only uses it to place the Cholesky shift).
f64 lanczos_min(const std::vector<f64>& A, Index N, int steps) {
    steps = std::max(1, std::min<int>(steps, static_cast<int>(N)));
    std::vector<std::vector<f64>> Vb;
    std::vector<f64> al, be, v(sz(N)), w(sz(N));
    Lcg g{12345};
    f64 nv = 0.0;
    for (auto& a : v) { a = g.next() - 0.5; nv += a * a; }
    nv = std::sqrt(nv);
    for (auto& a : v) a /= nv;
    for (int j = 0; j < steps; ++j) {
        Vb.push_back(v);
        dense_mv(A, N, v.data(), w.data());
        f64 a = 0.0;
        for (Index i = 0; i < N; ++i) a += w[sz(i)] * v[sz(i)];
        al.push_back(a);
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& q : Vb) {
                f64 d = 0.0;
                for (Index i = 0; i < N; ++i) d += w[sz(i)] * q[sz(i)];
                for (Index i = 0; i < N; ++i) w[sz(i)] -= d * q[sz(i)];
            }
        f64 b = 0.0;
        for (f64 t : w) b += t * t;
        b = std::sqrt(b);
        if (!(b > 1e-12 * (std::fabs(a) + 1e-300)) || j + 1 == steps) break;
        be.push_back(b);
        for (Index i = 0; i < N; ++i) v[sz(i)] = w[sz(i)] / b;
    }
    be.resize(al.size() > 0 ? al.size() - 1 : 0);
    return tridiag_min_eig(al, be);
}

// One certified trial: lower bound on lambda_min(A) from a completed
// Cholesky of fl(A - sigma I), or false if a pivot is not positive.
bool cholesky_trial(const std::vector<f64>& A, Index N, f64 sigma, f64& lower) {
    std::vector<f64> L(A);
    f64 emax = 0.0;
    for (Index i = 0; i < N; ++i) {
        f64& d = L[sz(i) * sz(N) + sz(i)];
        d = d - sigma;
        emax = std::max(emax, kUnit * std::fabs(d));   // |E_ii| <= u |fl(a - sigma)|
    }
    // Right-looking-free, row-oriented Cholesky (lower triangle, in place).
    for (Index j = 0; j < N; ++j) {
        f64* Lj = &L[sz(j) * sz(N)];
        f64 d = Lj[j];
        for (Index k = 0; k < j; ++k) d -= Lj[k] * Lj[k];
        if (!(d > 0.0) || !std::isfinite(d)) return false;
        d = std::sqrt(d);
        Lj[j] = d;
        const f64 inv = 1.0 / d;
        for (Index i = j + 1; i < N; ++i) {
            f64* Li = &L[sz(i) * sz(N)];
            f64 s = Li[j];
            for (Index k = 0; k < j; ++k) s -= Li[k] * Lj[k];
            Li[j] = s * inv;
        }
    }
    // ||L||_F^2 over the lower triangle; a sum of N^2 nonnegative terms, so
    // its own rounding is at most gamma_{N^2} relative -- charged below.
    f64 fro = 0.0;
    for (Index i = 0; i < N; ++i)
        for (Index k = 0; k <= i; ++k) fro += L[sz(i) * sz(N) + sz(k)] * L[sz(i) * sz(N) + sz(k)];
    const f64 Nf = static_cast<f64>(N);
    const f64 gam = (Nf + 1.0) * kUnit / (1.0 - (Nf + 1.0) * kUnit);
    const f64 charge = gam * fro * (1.0 + 2.0 * Nf * Nf * kUnit) + emax;
    // Final subtraction and product rounded: widen by a few ulps.
    lower = sigma - charge - 4.0 * kUnit * (std::fabs(sigma) + charge);
    return std::isfinite(lower);
}

}  // namespace

bool dense_lambda_min_lower(const std::vector<f64>& A, Index N, f64& lower, f64* estimate) {
    if (N <= 0) { lower = 0.0; return true; }
    f64 gers = kInf, scale = 0.0;
    for (Index i = 0; i < N; ++i) {
        f64 off = 0.0;
        for (Index j = 0; j < N; ++j) {
            const f64 a = A[sz(i) * sz(N) + sz(j)];
            scale = std::max(scale, std::fabs(a));
            if (j != i) off += std::fabs(a);
        }
        gers = std::min(gers, A[sz(i) * sz(N) + sz(i)] - off);
    }
    scale = std::max(scale, 1e-300);
    const f64 theta = lanczos_min(A, N, 80);
    if (estimate) *estimate = theta;
    const f64 base = std::max(scale, std::fabs(theta));
    for (f64 rel : {1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1}) {
        const f64 sigma = theta - rel * base;
        if (cholesky_trial(A, N, sigma, lower)) return true;
    }
    // Last resort: a Gershgorin-safe shift always factors in exact
    // arithmetic; the margin covers the roundoff.
    return cholesky_trial(A, N, gers - 1e-6 * base - 1e-3 * std::fabs(gers), lower);
}

std::vector<ProductIdentity> binary_product_identities(const engines::QpProblem& base) {
    std::vector<ProductIdentity> out;
    const auto& lp = base.linear;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index r = 0; r < lp.n_rows(); ++r) {
        if (rp[sz(r) + 1] - rp[sz(r)] != 2) continue;
        const Index i = ci[sz(rp[sz(r)])], j = ci[sz(rp[sz(r)] + 1)];
        const f64 a = lp.A.vals[sz(rp[sz(r)])], b = lp.A.vals[sz(rp[sz(r)] + 1)];
        if (i == j) continue;
        const f64 lo = lp.row_lo[sz(r)], hi = lp.row_hi[sz(r)];
        // Which corners of {0,1}^2 the row admits.  A margin keeps a corner
        // that sits on the boundary within roundoff feasible: excluding a
        // point that is actually feasible would make the identity invalid.
        bool ok[2][2];
        int nok = 0;
        for (int xi = 0; xi < 2; ++xi)
            for (int xj = 0; xj < 2; ++xj) {
                const f64 v = a * xi + b * xj;
                const f64 tol = 1e-9 * (1.0 + std::fabs(a) + std::fabs(b));
                ok[xi][xj] = v >= lo - tol && v <= hi + tol;
                nok += ok[xi][xj];
            }
        if (nok != 3) continue;   // 4: no information; <= 2: a fixing, not a product
        ProductIdentity p;
        p.i = i;
        p.j = j;
        if (!ok[1][1]) { /* x_i x_j = 0 */ }
        else if (!ok[0][0]) { p.l0 = -1.0; p.li = 1.0; p.lj = 1.0; }   // (1-xi)(1-xj) = 0
        else if (!ok[1][0]) { p.li = 1.0; }                            // xi (1-xj) = 0
        else { p.lj = 1.0; }                                           // xj (1-xi) = 0
        out.push_back(p);
    }
    return out;
}

bool bqp_sdp_shift(const engines::QpProblem& base, const BqpSdpOptions& opts,
                   BqpSdpResult& out) {
    const auto t0 = Clock::now();
    out = BqpSdpResult{};
    const auto& lp = base.linear;
    const Index n = lp.n_cols();
    if (n == 0) { out.reason = "empty problem"; return false; }
    if (n > opts.dense_limit) {
        out.reason = "n above the dense SDP limit";
        return false;
    }
    for (Index j = 0; j < n; ++j)
        if (lp.col_lo[sz(j)] != 0.0 || lp.col_hi[sz(j)] != 1.0) {
            out.reason = "SDP shift needs a [0,1] box";
            return false;
        }
    const Index N = n + 1;
    const std::size_t NN = sz(N);

    // Dense objective in x-space.
    std::vector<f64> Q0(sz(n) * sz(n), 0.0), c0(lp.c);
    {
        const auto& rp = base.q_matrix.pattern.row_ptr();
        const auto& ci = base.q_matrix.pattern.col_idx();
        for (Index i = 0; i < base.q_matrix.n_rows(); ++i)
            for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                Q0[sz(i) * sz(n) + sz(ci[sz(k)])] += base.q_matrix.vals[sz(k)];
    }
    const f64 off0 = lp.obj_offset;
    f64 obj_scale = 0.0;
    for (f64 v : Q0) obj_scale = std::max(obj_scale, std::fabs(v));
    for (f64 v : c0) obj_scale = std::max(obj_scale, std::fabs(v));
    obj_scale = std::max(obj_scale, 1e-12);

    // Equality rows (dense copies; n <= dense_limit and eq rows are few).
    std::vector<std::vector<std::pair<Index, f64>>> eq;
    std::vector<f64> eqb;
    {
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        for (Index r = 0; r < lp.n_rows(); ++r) {
            if (!(lp.row_lo[sz(r)] == lp.row_hi[sz(r)]) || !std::isfinite(lp.row_lo[sz(r)])) continue;
            std::vector<std::pair<Index, f64>> row;
            for (auto k = rp[sz(r)]; k < rp[sz(r) + 1]; ++k)
                if (lp.A.vals[sz(k)] != 0.0) row.push_back({ci[sz(k)], lp.A.vals[sz(k)]});
            if (row.empty()) continue;
            eq.push_back(std::move(row));
            eqb.push_back(lp.row_lo[sz(r)]);
        }
    }
    out.products = binary_product_identities(base);
    const std::size_t P = out.products.size();
    out.beta.assign(P, 0.0);

    f64 amax2 = 0.0;
    for (const auto& row : eq) {
        f64 s = 0.0;
        for (const auto& e : row) s += e.second * e.second;
        amax2 = std::max(amax2, s);
    }
    const f64 rho0 = eq.empty() ? 0.0 : obj_scale / std::max(amax2, 1e-12);

    // C(rho, beta) and K, as described in bqp_sdp.hpp.
    std::vector<f64> C(NN * NN), Qt, ct;
    f64 K = 0.0;
    auto build = [&](f64 rho, const std::vector<f64>& beta) {
        Qt = Q0;
        ct = c0;
        f64 off = off0;
        for (std::size_t r = 0; r < eq.size(); ++r) {
            const auto& row = eq[r];
            for (const auto& a : row) {
                for (const auto& b : row) Qt[sz(a.first) * sz(n) + sz(b.first)] += 2.0 * rho * a.second * b.second;
                ct[sz(a.first)] -= 2.0 * rho * eqb[r] * a.second;
            }
            off += rho * eqb[r] * eqb[r];
        }
        for (std::size_t p = 0; p < P; ++p) {
            const auto& pr = out.products[p];
            const f64 bt = beta[p];
            if (bt == 0.0) continue;
            Qt[sz(pr.i) * sz(n) + sz(pr.j)] += bt;
            Qt[sz(pr.j) * sz(n) + sz(pr.i)] += bt;
            ct[sz(pr.i)] -= bt * pr.li;
            ct[sz(pr.j)] -= bt * pr.lj;
            off -= bt * pr.l0;
        }
        std::fill(C.begin(), C.end(), 0.0);
        f64 sumQ = 0.0, sumc = 0.0;
        for (Index i = 0; i < n; ++i) {
            f64 rs = 0.0;
            for (Index j = 0; j < n; ++j) {
                const f64 q = Qt[sz(i) * sz(n) + sz(j)];
                C[sz(i + 1) * NN + sz(j + 1)] = 0.125 * q;
                rs += q;
            }
            const f64 qi = 0.25 * rs + 0.5 * ct[sz(i)];
            C[sz(i + 1)] = 0.5 * qi;
            C[sz(i + 1) * NN] = 0.5 * qi;
            sumQ += rs;
            sumc += ct[sz(i)];
        }
        K = 0.125 * sumQ + 0.5 * sumc + off;
    };

    // Low-rank factor V (N x k), unit rows.
    int k = opts.rank > 0 ? opts.rank
                          : static_cast<int>(std::ceil(std::sqrt(2.0 * static_cast<f64>(N)))) + 1;
    k = std::max(2, std::min<int>(k, static_cast<int>(N)));
    out.rank = k;
    const std::size_t kk = static_cast<std::size_t>(k);
    std::vector<f64> V(NN * kk);
    {
        Lcg g{0x5eed};
        for (std::size_t i = 0; i < NN; ++i) {
            f64 nv = 0.0;
            for (std::size_t c = 0; c < kk; ++c) {
                V[i * kk + c] = g.next() - 0.5;
                nv += V[i * kk + c] * V[i * kk + c];
            }
            nv = std::sqrt(nv);
            for (std::size_t c = 0; c < kk; ++c) V[i * kk + c] /= nv;
        }
    }
    std::vector<f64> gbuf(kk);
    // G = C V, and the primal objective <C, VV'>.
    auto primal_obj = [&]() {
        f64 f = 0.0;
        for (std::size_t i = 0; i < NN; ++i) {
            std::fill(gbuf.begin(), gbuf.end(), 0.0);
            const f64* ci = &C[i * NN];
            for (std::size_t j = 0; j < NN; ++j) {
                const f64 a = ci[j];
                if (a == 0.0) continue;
                const f64* vj = &V[j * kk];
                for (std::size_t c = 0; c < kk; ++c) gbuf[c] += a * vj[c];
            }
            for (std::size_t c = 0; c < kk; ++c) f += gbuf[c] * V[i * kk + c];
        }
        return f;
    };
    // The mixing method: exact minimisation over one row at a time.
    auto mix = [&](int max_sweeps) {
        f64 f = primal_obj();
        for (int sw = 0; sw < max_sweeps; ++sw) {
            f64 dec = 0.0;
            for (std::size_t i = 0; i < NN; ++i) {
                std::fill(gbuf.begin(), gbuf.end(), 0.0);
                const f64* ci = &C[i * NN];
                for (std::size_t j = 0; j < NN; ++j) {
                    if (j == i) continue;
                    const f64 a = ci[j];
                    if (a == 0.0) continue;
                    const f64* vj = &V[j * kk];
                    for (std::size_t c = 0; c < kk; ++c) gbuf[c] += a * vj[c];
                }
                f64 ng = 0.0, gv = 0.0;
                for (std::size_t c = 0; c < kk; ++c) {
                    ng += gbuf[c] * gbuf[c];
                    gv += gbuf[c] * V[i * kk + c];
                }
                ng = std::sqrt(ng);
                if (!(ng > 0.0)) continue;
                for (std::size_t c = 0; c < kk; ++c) V[i * kk + c] = -gbuf[c] / ng;
                dec += 2.0 * (ng + gv);   // f_old - f_new >= 0
            }
            ++out.sweeps;
            f -= dec;
            if (dec <= opts.tol * (std::fabs(f) + std::fabs(K) + 1e-9 * obj_scale)) break;
            if ((sw & 15) == 15 && secs_since(t0) > opts.time_limit_s) break;
        }
        return primal_obj();
    };
    // Dual estimate y_i = C_ii - ||g_i|| at the current V.
    std::vector<f64> y(NN), S(NN * NN);
    auto dual_y = [&]() {
        for (std::size_t i = 0; i < NN; ++i) {
            std::fill(gbuf.begin(), gbuf.end(), 0.0);
            const f64* ci = &C[i * NN];
            for (std::size_t j = 0; j < NN; ++j) {
                if (j == i) continue;
                const f64 a = ci[j];
                if (a == 0.0) continue;
                const f64* vj = &V[j * kk];
                for (std::size_t c = 0; c < kk; ++c) gbuf[c] += a * vj[c];
            }
            f64 ng = 0.0;
            for (f64 t : gbuf) ng += t * t;
            y[i] = C[i * NN + i] - std::sqrt(ng);
        }
        S = C;
        for (std::size_t i = 0; i < NN; ++i) S[i * NN + i] -= y[i];
    };
    auto sum_y = [&]() {
        f64 s = 0.0;
        for (f64 t : y) s += t;
        return s;
    };
    // Lifted residual of product p under Y = V V' (the supergradient).
    auto vdot = [&](std::size_t a, std::size_t b) {
        f64 s = 0.0;
        for (std::size_t c = 0; c < kk; ++c) s += V[a * kk + c] * V[b * kk + c];
        return s;
    };

    struct Best {
        f64 est = -kInf;
        f64 rho = 0.0;
        std::vector<f64> beta, y;
        f64 primal = 0.0;
    } best;

    const int stages = eq.empty() ? 1 : std::max(1, opts.penalty_stages);
    const int rounds = P == 0 ? 1 : std::max(1, opts.product_rounds);
    std::vector<f64> beta(P, 0.0), grad(P, 0.0);
    f64 step = 0.0;
    int since_gain = 0;
    const int total = std::max(stages, rounds);
    for (int t = 0; t < total; ++t) {
        if (t > 0 && secs_since(t0) > opts.time_limit_s) break;
        const int st = std::min(t, stages - 1);
        const f64 rho = rho0 * std::pow(4.0, static_cast<f64>(st));
        build(rho, beta);
        const f64 fp = mix(t == 0 ? opts.max_sweeps : std::max(50, opts.max_sweeps / 4)) + K;
        dual_y();
        const f64 theta = lanczos_min(S, N, 60);
        const f64 est = sum_y() + K + static_cast<f64>(N) * theta;
        ++out.stages;
        if (opts.verbose)
            std::fprintf(stderr, "sdp: t=%d rho=%.3e primal=%.10e dual~%.10e theta=%.3e sweeps=%lld %.2fs\n",
                         t, rho, fp, est, theta, out.sweeps, secs_since(t0));
        const bool gain = est > best.est + 1e-6 * (std::fabs(best.est) + 1.0);
        if (est > best.est) {
            best.est = est;
            best.rho = rho;
            best.beta = beta;
            best.y = y;
            best.primal = fp;
        }
        if (P > 0) {
            // Supergradient step on beta; on a loss, halve back towards the
            // best point.
            if (t > 0 && !gain) {
                step *= 0.5;
                beta = best.beta;
            } else if (t > 0) {
                step *= 1.3;
            }
            f64 gmax = 0.0;
            for (std::size_t p = 0; p < P; ++p) {
                const auto& pr = out.products[p];
                const std::size_t i = sz(pr.i + 1), j = sz(pr.j + 1);
                const f64 y0i = vdot(0, i), y0j = vdot(0, j), yij = vdot(i, j);
                grad[p] = 0.25 * (1.0 + y0i + y0j + yij) - pr.l0 -
                          0.5 * pr.li * (1.0 + y0i) - 0.5 * pr.lj * (1.0 + y0j);
                gmax = std::max(gmax, std::fabs(grad[p]));
            }
            if (t == 0) step = gmax > 0.0 ? 0.5 * obj_scale / gmax : 0.0;
            for (std::size_t p = 0; p < P; ++p) beta[p] += step * grad[p];
        }
        if (!gain && t > 0) ++since_gain;
        else since_gain = 0;
        // Penalty stages alone: stop when raising rho stops paying.
        if (P == 0 && t > 0 && !gain) break;
        if (P > 0 && since_gain >= 8) break;
    }

    // Certify the best dual: lambda_min(S) >= ell, then shift y by ell (and
    // a hair more, so the QCR Hessian is strictly PD for the downstream
    // factorizations) -- S - ell I is PSD.
    build(best.rho, best.beta);
    y = best.y;
    S = C;
    for (std::size_t i = 0; i < NN; ++i) S[i * NN + i] -= y[i];
    f64 ell = 0.0, theta = 0.0;
    if (!dense_lambda_min_lower(S, N, ell, &theta)) {
        out.reason = "could not certify lambda_min of the SDP dual slack";
        return false;
    }
    out.sdp_dual = sum_y() + K + static_cast<f64>(N) * ell;
    out.sdp_primal = best.primal;
    out.rho = best.rho;
    out.beta = best.beta;
    f64 cscale = 0.0;
    for (f64 v : C) cscale = std::max(cscale, std::fabs(v));
    const f64 margin = 1e-9 * std::max(cscale, 1e-300);
    out.u.assign(sz(n), 0.0);
    for (Index i = 0; i < n; ++i) out.u[sz(i)] = -4.0 * (y[sz(i + 1)] + ell - margin);
    out.valid = true;
    out.ms = secs_since(t0) * 1000.0;
    if (opts.verbose)
        std::fprintf(stderr, "sdp: certified lambda_min(S) >= %.3e (lanczos %.3e); dual %.10e primal %.10e\n",
                     ell, theta, out.sdp_dual, out.sdp_primal);
    return true;
}

}  // namespace sor::search
