// SOR — QCR bound; see qcr.hpp for the mathematics and why it is sound.
#include "sor/search/qcr.hpp"

#include "sor/search/bqp_sdp.hpp"
#include "sor/search/qplib_qp.hpp"
#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
constexpr f64 kUnit = 1.1102230246251565e-16;   // 2^-53, unit roundoff

// max_i (Q_ii + sum_{j!=i} |Q_ij|): an upper bound on lambda_max(Q), used as
// the power-iteration shift so B = sigma I - Q is PSD and its dominant
// eigenvalue is sigma - lambda_min(Q).
f64 gershgorin_upper(const sparse::CsrMatrix& Q) {
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    f64 up = -std::numeric_limits<f64>::infinity();
    for (Index i = 0; i < Q.n_rows(); ++i) {
        f64 d = 0.0, off = 0.0;
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (ci[sz(k)] == i) d += Q.vals[sz(k)];
            else off += std::fabs(Q.vals[sz(k)]);
        }
        up = std::max(up, d + off);
    }
    return std::isfinite(up) ? up : 0.0;
}

void spmv(const sparse::CsrMatrix& Q, const std::vector<f64>& x, std::vector<f64>& y) {
    y.assign(sz(Q.n_rows()), 0.0);
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    for (Index i = 0; i < Q.n_rows(); ++i)
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            y[sz(i)] += Q.vals[sz(k)] * x[sz(ci[sz(k)])];
}

// Power iteration on B = sigma I - Q.  The Rayleigh quotient never exceeds
// lambda_max(B), so the returned sigma - rho is >= the true lambda_min(Q):
// an estimate that errs toward too SMALL a shift, which is why the shift is
// always proved afterwards and never trusted.
f64 lambda_min_estimate(const sparse::CsrMatrix& Q) {
    const Index n = Q.n_rows();
    if (n == 0) return 0.0;
    const f64 sigma = gershgorin_upper(Q);
    std::vector<f64> v(sz(n)), w;
    for (Index i = 0; i < n; ++i)   // deterministic, not orthogonal to much
        v[sz(i)] = 1.0 + 0.01 * static_cast<f64>((i * 7919) % 101);
    f64 rho = 0.0, prev = -1.0;
    int stable = 0;
    for (int it = 0; it < 5000; ++it) {
        f64 nv = 0.0;
        for (f64 a : v) nv += a * a;
        nv = std::sqrt(nv);
        if (!(nv > 0.0)) break;
        for (f64& a : v) a /= nv;
        spmv(Q, v, w);
        rho = 0.0;
        for (Index i = 0; i < n; ++i) {
            w[sz(i)] = sigma * v[sz(i)] - w[sz(i)];   // w = B v
            rho += v[sz(i)] * w[sz(i)];
        }
        if (std::fabs(rho - prev) <= 1e-13 * std::max(1.0, std::fabs(rho))) {
            if (++stable >= 20) break;
        } else {
            stable = 0;
        }
        prev = rho;
        v.swap(w);
    }
    return sigma - rho;
}

// The convexified objective, as fresh problem data:
//
//   f(x) + sum_p beta_p (x_i x_j - l_p(x)) + rho sum_eq (a'x - b)^2
//        + sum_i u_i (x_i^2 - x_i)
//
// which equals f at every feasible binary point (bqp_sdp.hpp).  Forming it in
// floating point perturbs every coefficient a little, so the identity only
// holds up to that perturbation.  Each entry is a sum of t terms, each a
// product of at most three factors, so its error is at most
// gamma_{t+3} * (sum of |terms|); over the box [0,1] the whole objective can
// then move by at most 0.5 sum|dH_ij| + sum|dc_i| + |d offset|.  That is
// `charge`, and every bound computed on the result must subtract it.
engines::QpProblem convexified(const engines::QpProblem& base, const std::vector<f64>& u,
                               f64 rho, const std::vector<ProductIdentity>& prods,
                               const std::vector<f64>& beta, f64& charge) {
    engines::QpProblem p = base;
    const auto& lp = base.linear;
    const Index n = lp.n_cols();
    std::vector<Index> r, c;
    std::vector<f64> v;
    f64 habs = 0.0, cabs = 0.0, oabs = std::fabs(lp.obj_offset);
    std::size_t terms = 1;
    {
        const auto& rp = base.q_matrix.pattern.row_ptr();
        const auto& ci = base.q_matrix.pattern.col_idx();
        for (Index i = 0; i < n; ++i)
            for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                r.push_back(i); c.push_back(ci[sz(k)]); v.push_back(base.q_matrix.vals[sz(k)]);
                habs += std::fabs(base.q_matrix.vals[sz(k)]);
            }
    }
    for (Index i = 0; i < n; ++i) {
        cabs += std::fabs(lp.c[sz(i)]);
        if (u[sz(i)] == 0.0) continue;
        r.push_back(i); c.push_back(i); v.push_back(2.0 * u[sz(i)]);
        p.linear.c[sz(i)] -= u[sz(i)];
        habs += 2.0 * std::fabs(u[sz(i)]);
        cabs += std::fabs(u[sz(i)]);
    }
    terms += 1;
    if (rho != 0.0) {
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        for (Index row = 0; row < lp.n_rows(); ++row) {
            const f64 b = lp.row_lo[sz(row)];
            if (!(b == lp.row_hi[sz(row)]) || !std::isfinite(b)) continue;
            ++terms;
            for (auto k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
                const Index i = ci[sz(k)];
                const f64 ai = lp.A.vals[sz(k)];
                for (auto l = rp[sz(row)]; l < rp[sz(row) + 1]; ++l) {
                    const f64 t = 2.0 * rho * ai * lp.A.vals[sz(l)];
                    r.push_back(i); c.push_back(ci[sz(l)]); v.push_back(t);
                    habs += std::fabs(t);
                }
                const f64 t = 2.0 * rho * b * ai;
                p.linear.c[sz(i)] -= t;
                cabs += std::fabs(t);
            }
            p.linear.obj_offset += rho * b * b;
            oabs += std::fabs(rho * b * b);
        }
    }
    for (std::size_t q = 0; q < prods.size(); ++q) {
        const f64 bt = q < beta.size() ? beta[q] : 0.0;
        if (bt == 0.0) continue;
        const auto& pr = prods[q];
        r.push_back(pr.i); c.push_back(pr.j); v.push_back(bt);
        r.push_back(pr.j); c.push_back(pr.i); v.push_back(bt);
        habs += 2.0 * std::fabs(bt);
        p.linear.c[sz(pr.i)] -= bt * pr.li;
        p.linear.c[sz(pr.j)] -= bt * pr.lj;
        cabs += std::fabs(bt * pr.li) + std::fabs(bt * pr.lj);
        p.linear.obj_offset -= bt * pr.l0;
        oabs += std::fabs(bt * pr.l0);
        ++terms;
    }
    p.q_matrix = sparse::from_triplets(n, n, r, c, v);
    const f64 t = static_cast<f64>(terms + 3);
    const f64 gam = 2.0 * t * kUnit;   // gamma_t, doubled for the abs sums' own rounding
    charge = gam * (0.5 * habs + cabs + oabs);
    return p;
}

// Dense copy of a CSR matrix (row-major), for the rigorous PSD certificate.
std::vector<f64> dense_of(const sparse::CsrMatrix& Q) {
    const Index n = Q.n_rows();
    std::vector<f64> D(sz(n) * sz(n), 0.0);
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    for (Index i = 0; i < n; ++i)
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            D[sz(i) * sz(n) + sz(ci[sz(k)])] += Q.vals[sz(k)];
    return D;
}

// PSD slack for the relaxed Hessian: the engine's own certificate must pass
// (PDHCG-II refuses the solve otherwise), and when the matrix is small
// enough for a dense factorization the slack is the rigorous one from
// dense_lambda_min_lower(), which accounts for the Cholesky's roundoff.
bool certify_relaxed(const engines::QpProblem& p, const QcrOptions& opts, std::string& why,
                     f64& slack) {
    if (!engines::certify_qp_convex(p, opts.qp, why, slack)) return false;
    const Index n = p.linear.n_cols();
    if (n <= opts.dense_certificate_limit) {
        f64 lower = 0.0;
        if (!dense_lambda_min_lower(dense_of(p.q_matrix), n, lower)) {
            why = "dense PSD certificate failed";
            return false;
        }
        slack = std::max(0.0, -lower);
    }
    return true;
}

}  // namespace

bool wolfe_bound(const engines::QpProblem& p, const std::vector<f64>& x,
                 const std::vector<f64>& y, f64 psd_slack, f64& bound, f64& raw,
                 f64& charge_psd, f64& charge_fp) {
    const auto& lp = p.linear;
    const Index n = lp.n_cols(), m = lp.n_rows();
    if (x.size() != sz(n) || y.size() != sz(m)) return false;

    std::vector<f64> qx(sz(n), 0.0), qxa(sz(n), 0.0), aty(sz(n), 0.0), atya(sz(n), 0.0);
    {
        const auto& Q = p.q_matrix;
        const auto& rp = Q.pattern.row_ptr();
        const auto& ci = Q.pattern.col_idx();
        for (Index i = 0; i < Q.n_rows(); ++i)
            for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const f64 t = Q.vals[sz(k)] * x[sz(ci[sz(k)])];
                qx[sz(i)] += t;
                qxa[sz(i)] += std::fabs(t);
            }
    }
    {
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        for (Index i = 0; i < m; ++i)
            for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const f64 t = lp.A.vals[sz(k)] * y[sz(i)];
                aty[sz(ci[sz(k)])] += t;
                atya[sz(ci[sz(k)])] += std::fabs(t);
            }
    }

    // Every quantity below is accompanied by the sum of absolute values of
    // the terms that formed it; gamma times that sum bounds its error.
    f64 xqx = 0.0, xqx_abs = 0.0, px = 0.0, px_abs = 0.0, py = 0.0, py_abs = 0.0;
    f64 width2 = 0.0;
    for (Index j = 0; j < n; ++j) {
        const std::size_t s = sz(j);
        const f64 lo = lp.col_lo[s], hi = lp.col_hi[s];
        xqx += x[s] * qx[s];
        xqx_abs += std::fabs(x[s]) * qxa[s];
        const f64 r = qx[s] + lp.c[s] + aty[s];
        const f64 r_abs = qxa[s] + std::fabs(lp.c[s]) + atya[s];
        // supp_box(-r) = max over u in [lo,hi] of -r u.  If r is too small to
        // trust its sign, either bound may be the maximiser, so both must be
        // finite; otherwise only the one the sign selects.
        const bool sign_sure = std::fabs(r) > 4.0 * kUnit * r_abs;
        f64 bmax = 0.0;
        if (!sign_sure) {
            if (!std::isfinite(lo) || !std::isfinite(hi)) {
                if (r_abs != 0.0) return false;
            } else {
                bmax = std::max(std::fabs(lo), std::fabs(hi));
            }
        }
        if (r < 0.0) {
            if (!std::isfinite(hi)) return false;
            px += -r * hi;
            bmax = std::max(bmax, std::fabs(hi));
        } else if (r > 0.0) {
            if (!std::isfinite(lo)) return false;
            px += -r * lo;
            bmax = std::max(bmax, std::fabs(lo));
        }
        px_abs += (std::fabs(r) + r_abs) * bmax;
        if (psd_slack > 0.0) {
            if (!std::isfinite(lo) || !std::isfinite(hi)) return false;
            const f64 d = std::max(std::fabs(x[s] - lo), std::fabs(hi - x[s]));
            width2 += d * d;
        }
    }
    for (Index i = 0; i < m; ++i) {
        const std::size_t s = sz(i);
        const f64 yi = y[s];
        if (yi > 0.0) {
            if (!std::isfinite(lp.row_hi[s])) return false;
            py += yi * lp.row_hi[s];
            py_abs += std::fabs(yi * lp.row_hi[s]);
        } else if (yi < 0.0) {
            if (!std::isfinite(lp.row_lo[s])) return false;
            py += yi * lp.row_lo[s];
            py_abs += std::fabs(yi * lp.row_lo[s]);
        }
    }

    raw = -0.5 * xqx - px - py + lp.obj_offset;
    // First-order bound on the rounding of every sum and matvec that fed
    // raw: products and recursive sums over at most N terms each, charged at
    // gamma_N = N u with N the largest count involved, doubled for margin.
    const f64 N = static_cast<f64>(p.q_matrix.nnz() + lp.A.nnz() + n + m + 8);
    const f64 gamma = 2.0 * N * kUnit;
    charge_fp = gamma * (0.5 * xqx_abs + px_abs + py_abs + std::fabs(lp.obj_offset));
    charge_psd = 0.5 * psd_slack * width2;
    bound = raw - charge_fp - charge_psd;
    return std::isfinite(bound);
}

bool qcr_relaxation(const io::QplibInstance& inst, const QcrOptions& opts,
                    engines::QpProblem& relaxed, bool& negated, QcrDiagnostics& diag) {
    diag = QcrDiagnostics{};
    if (!inst.all_binary() &&
        std::any_of(inst.var_type.begin(), inst.var_type.end(),
                    [](io::QplibVarType t) { return t != io::QplibVarType::Binary; })) {
        diag.reason = "QCR needs every variable binary";
        return false;
    }
    engines::QpProblem base;
    negated = false;
    QplibToQpOptions conv;
    conv.relax_binary = true;
    if (!qplib_to_qp(inst, conv, base, negated, diag.reason)) return false;
    const Index n = base.linear.n_cols();
    const auto& Q = base.q_matrix;

    bool have = false;
    std::string why;
    const std::vector<ProductIdentity> no_prods;
    // SDP-derived shift first (Billionnet & Elloumi 2007; Billionnet,
    // Elloumi & Plateau 2009): the strongest diagonal shift there is, at the
    // price of a dense (n+1)^2 SDP solve.  Any failure falls through to the
    // cheaper shifts below -- never to no bound.
    const bool try_sdp = opts.shift == QcrShift::Sdp ||
                         ((opts.shift == QcrShift::Auto || opts.shift == QcrShift::Best) &&
                          n <= opts.sdp.dense_limit);
    if (try_sdp) {
        BqpSdpResult sr;
        if (bqp_sdp_shift(base, opts.sdp, sr)) {
            diag.sdp_dual = sr.sdp_dual;
            diag.sdp_primal = sr.sdp_primal;
            diag.sdp_rho = sr.rho;
            diag.sdp_products = sr.products.size();
            diag.sdp_rank = sr.rank;
            diag.sdp_sweeps = sr.sweeps;
            diag.sdp_ms = sr.ms;
            // The shift was made strictly PD in exact arithmetic; if the
            // formed matrix still fails a certificate, grow a uniform margin
            // (each step costs at most margin * n / 4 of bound).
            std::vector<f64> u = sr.u;
            f64 uscale = 1.0;
            for (f64 t : u) uscale = std::max(uscale, std::fabs(t));
            for (f64 rel : {0.0, 1e-9, 1e-7, 1e-5, 1e-3}) {
                std::vector<f64> uu(u);
                for (f64& t : uu) t += rel * uscale;
                relaxed = convexified(base, uu, sr.rho, sr.products, sr.beta, diag.charge_form);
                if (!certify_relaxed(relaxed, opts, why, diag.psd_slack)) continue;
                // The equality penalty is ZERO on the relaxation's feasible
                // set (the rows are kept), so it only has to make the
                // Hessian PSD -- and every unit of rho beyond that worsens
                // the conditioning of each node solve and inflates the
                // floating-point charges.  Take the smallest rho (to a
                // factor 2) that still certifies.
                if (sr.rho > 0.0) {
                    f64 hi = sr.rho, lo = 0.0;
                    f64 slack = 0.0;
                    std::string w2;
                    f64 cf = 0.0;
                    auto passes = [&](f64 r) {
                        const auto t = convexified(base, uu, r, sr.products, sr.beta, cf);
                        return certify_relaxed(t, opts, w2, slack);
                    };
                    if (passes(0.0)) {
                        hi = 0.0;
                    } else {
                        lo = sr.rho * 1e-12;
                        for (int it = 0; it < 14 && hi > 2.0 * lo; ++it) {
                            const f64 mid = std::sqrt(lo * hi);
                            if (passes(mid)) hi = mid;
                            else lo = mid;
                        }
                        hi = std::min(sr.rho, 2.0 * hi);
                    }
                    if (hi < sr.rho) {
                        auto t = convexified(base, uu, hi, sr.products, sr.beta, cf);
                        if (certify_relaxed(t, opts, w2, slack)) {
                            relaxed = std::move(t);
                            diag.charge_form = cf;
                            diag.psd_slack = slack;
                            diag.sdp_rho_relaxation = hi;
                        }
                    }
                }
                diag.shift_used = "sdp";
                diag.u = std::move(uu);
                have = true;
                break;
            }
            if (!have) diag.sdp_reason = "sdp shift failed its certificate: " + why;
        } else {
            diag.sdp_reason = sr.reason;
        }
        if (!have && opts.shift == QcrShift::Sdp) {
            diag.reason = diag.sdp_reason;
            return false;
        }
    }
    // Uniform (eigenvalue) shift.  No size gate: the PSD certificate is a
    // sparse LDL' now, not a dense Cholesky, so a large shifted Q can still
    // be proved.
    const bool try_eig = !have && opts.shift != QcrShift::DiagonalDominance;
    if (try_eig) {
        diag.lambda_min_estimate = lambda_min_estimate(Q);
        const f64 scale = std::max(1.0, std::fabs(gershgorin_upper(Q)));
        // Grow the margin until the Cholesky certificate accepts the shift.
        for (f64 rel : {1e-9, 1e-7, 1e-5, 1e-3}) {
            const f64 s = std::max(0.0, -diag.lambda_min_estimate) + rel * scale;
            std::vector<f64> u(sz(n), 0.5 * s);
            relaxed = convexified(base, u, 0.0, no_prods, {}, diag.charge_form);
            if (certify_relaxed(relaxed, opts, why, diag.psd_slack)) {
                diag.uniform_shift = s;
                diag.shift_used = "min-eigenvalue";
                diag.u = std::move(u);
                have = true;
                break;
            }
        }
    }
    if (!have && opts.shift == QcrShift::MinEigenvalue) {
        diag.reason = "min-eigenvalue shift could not be certified PSD: " + why;
        return false;
    }
    if (!have) {
        std::vector<f64> u(sz(n), 0.0);
        const auto& rp = Q.pattern.row_ptr();
        const auto& ci = Q.pattern.col_idx();
        for (Index i = 0; i < n; ++i) {
            f64 d = 0.0, off = 0.0;
            for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                if (ci[sz(k)] == i) d += Q.vals[sz(k)];
                else off += std::fabs(Q.vals[sz(k)]);
            }
            u[sz(i)] = 0.5 * std::max(0.0, off - d);
        }
        relaxed = convexified(base, u, 0.0, no_prods, {}, diag.charge_form);
        if (!certify_relaxed(relaxed, opts, why, diag.psd_slack)) {
            diag.reason = "diagonal-dominance shift failed its certificate: " + why;
            return false;
        }
        diag.shift_used = "diagonal-dominance";
        diag.u = std::move(u);
    }
    return true;
}

bool qcr_candidates(const io::QplibInstance& inst, const QcrOptions& opts,
                    std::vector<QcrCandidate>& out, bool& negated, std::string& why) {
    out.clear();
    if (opts.shift == QcrShift::Best) {
        for (auto s : {QcrShift::Sdp, QcrShift::MinEigenvalue}) {
            QcrOptions o = opts;
            o.shift = s;
            QcrCandidate c;
            bool neg = false;
            if (qcr_relaxation(inst, o, c.relaxed, neg, c.diag)) {
                negated = neg;
                out.push_back(std::move(c));
            } else {
                why = c.diag.reason;
            }
        }
        if (!out.empty()) return true;
        // Neither certified: fall back to Auto, whose last resort (diagonal
        // dominance by Gershgorin) cannot fail.
    }
    QcrOptions o = opts;
    if (o.shift == QcrShift::Best) o.shift = QcrShift::Auto;
    QcrCandidate c;
    bool neg = false;
    if (!qcr_relaxation(inst, o, c.relaxed, neg, c.diag)) {
        why = c.diag.reason;
        return false;
    }
    negated = neg;
    out.push_back(std::move(c));
    return true;
}

QcrBound qcr_bound(const io::QplibInstance& inst, const QcrOptions& opts,
                   backend::PdhcgDevice& device, QcrDiagnostics& diag) {
    QcrBound out;
    diag = QcrDiagnostics{};
    std::vector<QcrCandidate> cands;
    bool negated = false;
    if (!qcr_candidates(inst, opts, cands, negated, diag.reason)) return out;

    // Each candidate's value is valid on its own (weak duality at its own
    // certified PSD slack), so the best of them is valid too.
    f64 best = -std::numeric_limits<f64>::infinity();
    for (auto& cand : cands) {
        const Index n = cand.relaxed.linear.n_cols();
        auto& d = cand.diag;
        const auto raw = engines::solve_qp_pdhcg(cand.relaxed, opts.qp, device, d.relaxation);
        if (raw.x.size() != sz(n)) {
            d.reason = "relaxation solve failed: " + raw.termination_reason;
            continue;
        }
        f64 b = 0.0;
        if (!wolfe_bound(cand.relaxed, raw.x, raw.y, d.psd_slack, b, d.bound_raw,
                         d.charge_psd, d.charge_fp)) {
            d.reason = "relaxation point gives no finite bound";
            continue;
        }
        b -= d.charge_form;
        if (!out.valid || b > best) {
            best = b;
            diag = d;
            out.valid = true;
        }
    }
    if (!out.valid) {
        diag = cands.front().diag;
        return out;
    }
    out.bound = negated ? -best : best;
    return out;
}

}  // namespace sor::search
