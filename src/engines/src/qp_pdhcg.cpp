#include "sor/engines/qp.hpp"
#include "sor/backend/batched_pdhcg_device.hpp"
#include "sor/backend/pdhcg_device.hpp"
#include "sor/la/ldlt.hpp"
#include "sor/sparse/csc.hpp"
#include "qp_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sor::engines {
namespace {

// Algorithmic reference (equations 4, 6, 7, 10 and 11--15):
//   Li, Huang, Liu, Ge & Ye, "PDHCG-II: An Enhanced Version of PDHCG for
//   Large-Scale Convex QP", arXiv:2602.23967 (2026).

using Clock = std::chrono::steady_clock;
inline std::size_t sz(core::Index i) { return static_cast<std::size_t>(i); }
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

bool has_sparse_q(const QpProblem& p) {
    return p.q_matrix.n_rows() != 0 || p.q_matrix.n_cols() != 0 ||
           p.q_matrix.nnz() != 0;
}

f64 matrix_row_sum_norm(const sparse::CsrMatrix& A) {
    f64 out = 0.0;
    const auto& rp = A.pattern.row_ptr();
    for (core::Index i = 0; i < A.n_rows(); ++i) {
        f64 sum = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            sum += std::fabs(A.vals[sz(k)]);
        out = std::max(out, sum);
    }
    return out;
}

// sqrt(||A||_1 ||A||_inf) is a deterministic upper bound on ||A||_2, so the
// resulting PDHG step is safe rather than dependent on a power-iteration
// underestimate.
f64 matrix_two_norm_upper(const sparse::CsrMatrix& A) {
    if (A.n_rows() == 0 || A.n_cols() == 0 || A.nnz() == 0) return 0.0;
    std::vector<f64> col_sum(sz(A.n_cols()), 0.0);
    f64 row_max = 0.0;
    const auto& rp = A.pattern.row_ptr();
    const auto& ci = A.pattern.col_idx();
    for (core::Index i = 0; i < A.n_rows(); ++i) {
        f64 row_sum = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 a = std::fabs(A.vals[sz(k)]);
            row_sum += a;
            col_sum[sz(ci[sz(k)])] += a;
        }
        row_max = std::max(row_max, row_sum);
    }
    f64 col_max = 0.0;
    for (f64 a : col_sum) col_max = std::max(col_max, a);
    return std::sqrt(row_max * col_max);
}

f64 sparse_value(const sparse::CsrMatrix& Q, core::Index i, core::Index j) {
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    const auto first = ci.begin() + static_cast<std::ptrdiff_t>(rp[sz(i)]);
    const auto last = ci.begin() + static_cast<std::ptrdiff_t>(rp[sz(i) + 1]);
    const auto it = std::lower_bound(first, last, j);
    if (it == last || *it != j) return 0.0;
    return Q.vals[static_cast<std::size_t>(it - ci.begin())];
}

bool symmetric_q(const sparse::CsrMatrix& Q, f64 tol, std::string& reason) {
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    for (core::Index i = 0; i < Q.n_rows(); ++i) {
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const core::Index j = ci[sz(k)];
            const f64 a = Q.vals[sz(k)];
            const f64 b = sparse_value(Q, j, i);
            if (!std::isfinite(a) || std::fabs(a - b) > tol * (1.0 + std::fabs(a))) {
                reason = "Q must be finite and symmetric (store both triangles)";
                return false;
            }
        }
    }
    return true;
}

// Numerical PSD certificate.  For modest Q, Cholesky of Q + delta I accepts
// positive semidefinite matrices up to a scale-aware roundoff tolerance.  For
// large Q we use the sparse Gershgorin lower bound and otherwise require the
// caller to explicitly declare PSD through QpOptions::assume_psd.
//
// `slack` receives the delta actually certified: the proof is that Q + slack*I
// is PSD, i.e. lambda_min(Q) >= -slack.  Callers that turn a dual point into
// a bound (QCR) must charge for it; a convex solve does not need to.
bool certify_psd(const QpProblem& p, const QpOptions& opts, std::string& reason,
                 f64& slack) {
    const core::Index n = p.linear.n_cols();
    slack = 0.0;
    if (!has_sparse_q(p)) {
        for (f64 q : p.q_diag) {
            if (!std::isfinite(q) || q < -1e-12) {
                reason = "diagonal Q is not positive semidefinite";
                return false;
            }
            slack = std::max(slack, -q);
        }
        return true;
    }

    if (!symmetric_q(p.q_matrix, 1e-11, reason)) return false;
    if (opts.assume_psd) {
        // Trusted, not proved: no slack can be stated.  QCR never sets this.
        slack = core::kPosInf;
        return true;
    }

    const auto& rp = p.q_matrix.pattern.row_ptr();
    const auto& ci = p.q_matrix.pattern.col_idx();
    f64 scale = 1.0;
    f64 gershgorin_min = std::numeric_limits<f64>::infinity();
    for (core::Index i = 0; i < n; ++i) {
        f64 d = 0.0, off = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 a = p.q_matrix.vals[sz(k)];
            scale = std::max(scale, std::fabs(a));
            if (ci[sz(k)] == i) d += a;
            else off += std::fabs(a);
        }
        gershgorin_min = std::min(gershgorin_min, d - off);
    }
    const f64 delta = 1e-11 * scale;
    if (gershgorin_min >= -delta) {
        slack = std::max(0.0, -gershgorin_min);
        return true;
    }
    // Sparse LDL' of Q + delta I.  All pivots positive proves that matrix
    // positive definite, hence lambda_min(Q) >= -delta, which is exactly the
    // certificate the dense Cholesky used to give -- at any size, instead of
    // only for n <= convexity_dense_limit.  QPLIB_8515 (n = 16002) and
    // QPLIB_8906 (n = 5223) are convex and were refused purely for being
    // bigger than that limit.
    la::SymCsc K;
    K.n = n;
    K.col_ptr.assign(sz(n) + 1, 0);
    {
        // Upper triangle by column, with the diagonal always present.
        std::vector<std::vector<std::pair<core::Index, f64>>> cols(sz(n));
        const auto& qrp = p.q_matrix.pattern.row_ptr();
        const auto& qci = p.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n; ++i)
            for (core::Offset k = qrp[sz(i)]; k < qrp[sz(i) + 1]; ++k) {
                const core::Index j = qci[sz(k)];
                if (i <= j) cols[sz(j)].push_back({i, p.q_matrix.vals[sz(k)]});
            }
        for (core::Index j = 0; j < n; ++j) {
            bool diag = false;
            for (const auto& e : cols[sz(j)]) diag = diag || e.first == j;
            if (!diag) cols[sz(j)].push_back({j, 0.0});
            for (const auto& e : cols[sz(j)]) {
                K.row_idx.push_back(e.first);
                K.vals.push_back(e.first == j ? e.second + delta : e.second);
            }
            K.col_ptr[sz(j) + 1] = static_cast<la::Offset>(K.row_idx.size());
        }
    }
    la::Ldlt fac;
    try {
        fac.analyze(K);
    } catch (const std::exception&) {
        reason = "PSD certificate: could not order Q";
        return false;
    }
    // Guard the memory the factor would take before filling it.
    constexpr la::Offset kMaxFactorEntries = 200000000;   // ~1.6 GB of values
    if (fac.nnz_l() > kMaxFactorEntries) {
        reason = "PSD not certifiable: the factor of Q would exceed the fill "
                 "budget; set assume_psd only for a trusted convex model";
        return false;
    }
    // reg = 0 with the expected sign positive: any pivot that is not
    // positive is counted, and one such pivot means not PSD at this delta.
    if (!fac.factorize(K, std::vector<std::int8_t>(sz(n), 1), 0.0) ||
        fac.regularized_pivots() > 0) {
        reason = "Q is indefinite (convex QP requires Q positive semidefinite)";
        return false;
    }
    slack = delta;
    return true;
}

bool fast_path_eligible(const QpProblem& p) {
    if (has_sparse_q(p) || p.q_diag.size() != sz(p.linear.n_cols())) return false;
    for (f64 q : p.q_diag) if (!(q > 0.0)) return false;
    for (core::Index i = 0; i < p.linear.n_rows(); ++i) {
        const f64 lo = p.linear.row_lo[sz(i)], hi = p.linear.row_hi[sz(i)];
        if ((lo > -model::kInf || hi < model::kInf) && lo != hi) return false;
    }
    return true;
}

// The one PDHCG-II loop.  K lanes share A, Q, c and the step sizes (which
// depend only on those) and differ in bounds; each lane keeps its own inner
// tolerance, Barzilai-Borwein step, convergence state and diagnostics, and
// leaves the inner loop -- or the solve -- on its own tests.  A single solve
// is K = 1 (solve_qp_general below), so there is exactly one copy of the
// algorithm whatever the batch width.
std::vector<core::RawResult> solve_batch_core(const QpProblem& p,
                                              const std::vector<backend::LaneBounds>& lanes,
                                              const QpOptions& opts,
                                              backend::BatchedPdhcgDevice& dev,
                                              std::vector<QpDiagnostics>& diags) {
    const auto t0 = Clock::now();
    const auto deadline = (opts.time_limit_s > 0.0)
        ? t0 + std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(opts.time_limit_s))
        : Clock::time_point::max();
    const std::size_t K = lanes.size();
    std::vector<core::RawResult> raws(K);
    diags.assign(K, QpDiagnostics{});
    for (std::size_t l = 0; l < K; ++l) {
        raws[l].engine = "qp_pdhcg2";
        raws[l].backend = std::string(dev.name());
        diags[l].used_general_path = true;
    }

    const auto& lp = p.linear;
    const core::Index n = lp.n_cols(), m = lp.n_rows();
    auto finish_error = [&](core::Status status, const std::string& why) {
        for (std::size_t l = 0; l < K; ++l) {
            raws[l].proposed_status = status;
            raws[l].termination_reason = why;
            diags[l].termination_reason = why;
            diags[l].total_ms = ms_since(t0);
        }
        return raws;
    };
    if (K == 0 || (dev.lanes() != 0 && dev.lanes() != K))
        return finish_error(core::Status::Unsupported, "batch width does not match the device");
    if (lp.maximize)
        return finish_error(core::Status::Unsupported,
                            "convex QP supports minimization only");
    try {
        lp.validate();
    } catch (const std::exception& e) {
        return finish_error(core::Status::Unsupported,
                            std::string("invalid QP: ") + e.what());
    }
    for (const auto& b : lanes) {
        if (b.col_lo.size() != sz(n) || b.col_hi.size() != sz(n) ||
            b.row_lo.size() != sz(m) || b.row_hi.size() != sz(m))
            return finish_error(core::Status::Unsupported, "lane bounds have the wrong length");
        for (core::Index j = 0; j < n; ++j)
            if (!(b.col_lo[sz(j)] <= b.col_hi[sz(j)]))
                return finish_error(core::Status::Unsupported, "lane has col_lo > col_hi");
        for (core::Index i = 0; i < m; ++i)
            if (!(b.row_lo[sz(i)] <= b.row_hi[sz(i)]))
                return finish_error(core::Status::Unsupported, "lane has row_lo > row_hi");
    }
    if (has_sparse_q(p)) {
        if (p.q_matrix.n_rows() != n || p.q_matrix.n_cols() != n)
            return finish_error(core::Status::Unsupported, "Q must be n_cols by n_cols");
        if (p.q_matrix.vals.size() != sz(p.q_matrix.nnz()))
            return finish_error(core::Status::Unsupported,
                                "Q values length must equal its sparse nnz");
    } else if (p.q_diag.size() != sz(n)) {
        return finish_error(core::Status::Unsupported, "q_diag length must equal n_cols");
    }
    if (!(opts.step_safety > 0.0 && opts.step_safety < 1.0) ||
        opts.inner_max_iterations <= 0 || opts.inner_epoch <= 0)
        return finish_error(core::Status::Unsupported,
                            "QP options require 0 < step_safety < 1 and positive inner "
                            "iterations and inner_epoch");
    std::string convexity_reason;
    f64 psd_slack = 0.0;
    if (!certify_psd(p, opts, convexity_reason, psd_slack))
        return finish_error(core::Status::Unsupported, convexity_reason);

    if (Clock::now() >= deadline)
        return finish_error(core::Status::Interrupted, "PDHCG-II time limit");

    const f64 anorm = matrix_two_norm_upper(lp.A);
    const f64 qnorm = has_sparse_q(p) ? matrix_row_sum_norm(p.q_matrix)
                                      : (p.q_diag.empty() ? 0.0
                                           : *std::max_element(p.q_diag.begin(),
                                                               p.q_diag.end()));
    for (auto& d : diags) {
        d.convexity_certified = true;
        d.matrix_norm_estimate = anorm;
    }
    const f64 eta = anorm > 0.0 ? opts.step_safety / anorm
                                : 1.0 / std::max<f64>(1.0, qnorm);
    // Step sizes: tau * sigma = eta^2 < 1/||A||^2 whatever the primal weight
    // omega, so rebalancing them never breaks the convergence condition.
    // Per lane: a lane in a batch takes the steps it would take alone.
    std::vector<f64> omega(K, 1.0), tau(K, eta), sigma(K, eta), lipschitz(K, qnorm + 1.0 / eta);

    // The device owns every vector; this function owns every decision.
    // See backend/pdhcg_device.hpp for where the split falls and why.
    backend::PdhcgData data;
    data.A_csr = lp.A;
    data.A_csc = sparse::to_csc(lp.A);
    data.diagonal = !has_sparse_q(p);
    if (data.diagonal) data.q_diag = p.q_diag;
    else data.Q_csr = p.q_matrix;
    data.c = lp.c;
    dev.reset_stats();
    dev.upload(data, lanes);
    dev.init();

    std::vector<f64> inner_tol(K, std::max(opts.inner_tolerance_min, 1e-3));
    std::vector<f64> certified_dual_bound(K, core::kNaN);
    std::vector<std::uint8_t> converged(K, 0);

    // Scores the lanes in `mask`; returns which of them converged.
    // One evaluation's verdict.  `err` is the single number the restart rule
    // compares: the worst of the primal residual, the natural-map residual
    // and (when a Wolfe bound exists) the relative gap.
    struct Score {
        bool ok = false;
        f64 err = core::kPosInf;
    };
    auto score_of = [&](const backend::PdhcgDevice::Eval& e, std::size_t l, bool record) {
        const f64 xqx = e.xqx, ctx = e.ctx, px = e.px, py = e.py;
        const bool finite_gap = e.support_finite;
        f64 gap = core::kPosInf;
        f64 bound = core::kNaN;
        if (finite_gap) {
            const f64 num = std::fabs(xqx + ctx + px + py);
            const f64 dual_expr = 0.5 * xqx + px + py;
            gap = num / (1.0 + std::max(std::fabs(0.5 * xqx + ctx), std::fabs(dual_expr)));
            bound = -dual_expr + lp.obj_offset;
        }
        if (record) {
            auto& diag = diags[l];
            certified_dual_bound[l] = bound;
            diag.gap_finite = finite_gap;
            diag.primal_residual = e.primal;
            diag.stationarity = e.dual_res;
            diag.gap_rel = gap;
            diag.objective = 0.5 * xqx + ctx + lp.obj_offset;
        }
        Score sc;
        sc.ok = e.primal <= opts.feas_tol && e.dual_res <= opts.stationarity_tol &&
                (!finite_gap || gap <= opts.gap_tol);
        sc.err = std::max({e.primal, e.dual_res, finite_gap ? gap : 0.0});
        return sc;
    };
    // Scores the lanes in `mask` at the current iterate (recorded in the
    // diagnostics) or at the running average (compared only).
    auto evaluate = [&](const backend::LaneMask& mask, bool at_average = false) {
        const auto es = dev.evaluate(at_average, mask);
        std::vector<Score> out(K);
        for (std::size_t l = 0; l < K; ++l)
            if (mask[l]) out[l] = score_of(es[l], l, !at_average);
        return out;
    };

    // Adaptive restarts (Applegate et al., "Practical large-scale linear
    // programming using primal-dual hybrid gradient", NeurIPS 2021, s.3-4):
    // restart from the better of (current, average) when its KKT error has
    // fallen far enough below the error at the last restart, when progress
    // has stalled below a looser threshold, or after an artificial fraction
    // of the total iterations.  Per lane.
    constexpr f64 kSufficient = 0.2, kNecessary = 0.8, kArtificial = 0.36;
    std::vector<f64> mu_ref(K, core::kPosInf), last_cand(K, core::kPosInf);
    std::vector<std::uint64_t> since_restart(K, 0);

    backend::LaneMask active(K, 1);
    bool timed_out = false;
    for (std::uint64_t k = 0; k < opts.max_iterations; ++k) {
        if (Clock::now() >= deadline) { timed_out = true; break; }
        for (std::size_t l = 0; l < K; ++l)
            if (active[l]) diags[l].iterations = k + 1;
        dev.outer_begin(active);

        if (data.diagonal) {
            dev.diag_prox(tau, active);
            for (std::size_t l = 0; l < K; ++l)
                if (active[l]) ++diags[l].inner_iterations;
        } else {
            dev.inner_begin(active);
            std::vector<f64> alpha(K);
            for (std::size_t l = 0; l < K; ++l) alpha[l] = 1.0 / lipschitz[l];
            backend::LaneMask inner = active;
            // Epoch batching (measured 2026-09-23; see
            // PdhcgDevice::inner_advance_blind and vk_pdhcg_device.cpp's
            // file header for the design and why it is safe).  Every "it"
            // below is the SAME step the loop always took: one
            // inner_grad() stop test, then -- if not converged -- one
            // inner_trial() BB update, in that order, nothing else between
            // them.  The only change opts.inner_epoch > 1 makes is that the
            // (epoch - 1) iterations before each such step are recorded
            // blind first (fixed alpha, no stop test, one call to
            // inner_advance_blind()) instead of individually with their own
            // host round trip.  opts.inner_epoch == 1 (the default) makes
            // `blind` always 0, so this is byte-for-byte the pre-epoch loop.
            const int epoch = std::max(1, opts.inner_epoch);
            for (int it = 0; it < opts.inner_max_iterations;) {
                if (Clock::now() >= deadline) { timed_out = true; break; }
                const int remaining = opts.inner_max_iterations - it;
                const int blind = std::min(epoch - 1, remaining - 1);
                if (blind > 0) {
                    dev.inner_advance_blind(alpha, tau, inner, blind);
                    for (std::size_t l = 0; l < K; ++l)
                        if (inner[l])
                            diags[l].inner_iterations += static_cast<std::uint64_t>(blind);
                    it += blind;
                }
                for (std::size_t l = 0; l < K; ++l)
                    if (inner[l]) ++diags[l].inner_iterations;
                const auto pg = dev.inner_grad(tau, inner);
                bool any = false;
                for (std::size_t l = 0; l < K; ++l) {
                    if (inner[l] && pg[l] <= inner_tol[l]) inner[l] = 0;
                    any = any || inner[l];
                }
                ++it;
                if (!any) break;
                const auto bb = dev.inner_trial(alpha, tau, inner);
                for (std::size_t l = 0; l < K; ++l)
                    if (inner[l] && bb[l].sty > 0.0)
                        alpha[l] = std::clamp(bb[l].sts / bb[l].sty, 1e-12, 1.9 / lipschitz[l]);
            }
            dev.inner_end(active);
        }

        if (timed_out) break;
        const f64 a = static_cast<f64>(k + 1) / static_cast<f64>(k + 2);
        // Only the sparse path's inner tolerance consumes the movement, so the
        // diagonal path skips it and a device never has to read it back.
        const auto movement2 = dev.dual_and_advance(
            sigma, opts.use_reflected_halpern, a, opts.halpern_theta, !data.diagonal, active);
        if (!data.diagonal)
            for (std::size_t l = 0; l < K; ++l)
                if (active[l])
                    inner_tol[l] = std::min(inner_tol[l],
                        std::max(opts.inner_tolerance_min,
                                 opts.inner_tolerance_scale * std::sqrt(movement2[l]) / tau[l]));
        if (opts.adaptive_restart) {
            dev.average_add(active);
            for (std::size_t l = 0; l < K; ++l)
                if (active[l]) ++since_restart[l];
        }

        if (opts.check_every > 0 && (k + 1) % opts.check_every == 0) {
            const auto cur = evaluate(active);
            std::vector<Score> avg;
            if (opts.adaptive_restart) avg = evaluate(active, true);

            backend::LaneMask restart_mask(K, 0);
            std::vector<std::uint8_t> to_avg(K, 0);
            backend::LaneMask converged_on_avg(K, 0);
            for (std::size_t l = 0; l < K; ++l) {
                if (!active[l]) continue;
                if (cur[l].ok) { converged[l] = 1; active[l] = 0; continue; }
                if (!opts.adaptive_restart) continue;
                if (avg[l].ok) {                  // the average converged first
                    restart_mask[l] = 1;
                    to_avg[l] = 1;
                    converged_on_avg[l] = 1;
                    continue;
                }
                const bool use_avg = avg[l].err < cur[l].err;
                const f64 cand = use_avg ? avg[l].err : cur[l].err;
                if (!std::isfinite(mu_ref[l])) { mu_ref[l] = cand; last_cand[l] = cand; continue; }
                const bool fire =
                    cand <= kSufficient * mu_ref[l] ||
                    (cand <= kNecessary * mu_ref[l] && cand > last_cand[l]) ||
                    static_cast<f64>(since_restart[l]) >= kArtificial * static_cast<f64>(k + 1);
                if (fire) {
                    restart_mask[l] = 1;
                    to_avg[l] = use_avg ? 1 : 0;
                    mu_ref[l] = cand;
                    last_cand[l] = core::kPosInf;
                    since_restart[l] = 0;
                } else {
                    last_cand[l] = cand;
                }
            }
            bool any_restart = false;
            for (std::size_t l = 0; l < K; ++l) any_restart = any_restart || restart_mask[l];
            if (any_restart) {
                // Primal weight (same reference, s.3.3): rebalance tau and
                // sigma toward the observed ratio of dual to primal movement,
                // lane by lane, measured before the restart moves the iterate.
                if (opts.primal_weight) {
                    const auto mv = dev.movement_since_restart(restart_mask);
                    for (std::size_t l = 0; l < K; ++l) {
                        if (!restart_mask[l] || converged_on_avg[l]) continue;
                        if (mv[l].dx > 1e-10 && mv[l].dy > 1e-10) {
                            omega[l] = std::exp(0.5 * std::log(mv[l].dy / mv[l].dx) +
                                                0.5 * std::log(omega[l]));
                            omega[l] = std::clamp(omega[l], 1e-4, 1e4);
                            tau[l] = eta / omega[l];
                            sigma[l] = eta * omega[l];
                            lipschitz[l] = qnorm + 1.0 / tau[l];
                        }
                    }
                }
                dev.restart(to_avg, restart_mask);
                for (std::size_t l = 0; l < K; ++l)
                    if (restart_mask[l]) ++diags[l].restarts;
                // A lane whose average passed: re-score it as the iterate and stop it.
                bool any_avg = false;
                for (std::size_t l = 0; l < K; ++l) any_avg = any_avg || converged_on_avg[l];
                if (any_avg) {
                    const auto re = evaluate(converged_on_avg);
                    for (std::size_t l = 0; l < K; ++l)
                        if (converged_on_avg[l] && re[l].ok) { converged[l] = 1; active[l] = 0; }
                }
            }
            bool any = false;
            for (std::size_t l = 0; l < K; ++l) any = any || active[l];
            if (!any) break;
            if (opts.verbose && ((k + 1) % (opts.check_every * 100) == 0))
                for (std::size_t l = 0; l < K; ++l)
                    if (active[l])
                        std::fprintf(stderr, "qp lane=%zu iter=%llu primal=%.3e dual=%.3e gap=%.3e\n",
                                     l, static_cast<unsigned long long>(k + 1),
                                     diags[l].primal_residual, diags[l].stationarity,
                                     diags[l].gap_rel);
            if (Clock::now() >= deadline) { timed_out = true; break; }
        }
    }
    // Lanes still running get one last check, as the single solve always had.
    {
        backend::LaneMask last(K, 0);
        bool any = false;
        for (std::size_t l = 0; l < K; ++l)
            if (!converged[l] || diags[l].iterations == 0) { last[l] = 1; any = true; }
        if (any) {
            const auto ok = evaluate(last);
            for (std::size_t l = 0; l < K; ++l)
                if (last[l]) converged[l] = ok[l].ok;
        }
    }

    const auto stats = dev.transfer_stats();
    for (std::size_t l = 0; l < K; ++l) {
        auto& raw = raws[l];
        auto& diag = diags[l];
        dev.download(l, raw.x, raw.y);
        diag.device_stats = stats;
        raw.objective = diag.objective;
        raw.iterations = diag.iterations;
        const bool proved = converged[l] && !timed_out;
        raw.proposed_status = proved ? core::Status::Optimal : core::Status::Interrupted;
        raw.proposed_level = proved ? core::ProofLevel::ProvedKKT
                                          : core::ProofLevel::None;
        raw.dual_bound = proved ? certified_dual_bound[l] : core::kNaN;
        raw.termination_reason = proved ? "PDHCG-II KKT and Wolfe gap satisfied"
                                              : (timed_out ? "PDHCG-II time limit" : "PDHCG-II iteration limit");
        diag.termination_reason = raw.termination_reason;
        diag.total_ms = ms_since(t0);
    }
    return raws;
}

std::vector<core::RawResult> solve_batch_impl(const QpProblem& p,
                                              const std::vector<backend::LaneBounds>& lanes,
                                              const QpOptions& opts,
                                              backend::BatchedPdhcgDevice& dev,
                                              std::vector<QpDiagnostics>& diags) {
    if (!opts.scale || opts.ruiz_iterations <= 0 || p.linear.maximize)
        return solve_batch_core(p, lanes, opts, dev, diags);
    try {
        p.linear.validate();
    } catch (const std::exception&) {
        return solve_batch_core(p, lanes, opts, dev, diags);   // reports the error
    }
    if (has_sparse_q(p) ? p.q_matrix.n_rows() != p.linear.n_cols()
                        : p.q_diag.size() != sz(p.linear.n_cols()))
        return solve_batch_core(p, lanes, opts, dev, diags);   // reports the error

    // Convexity is certified on the ORIGINAL Q.  Dc Q Dc is congruent to Q,
    // so the proof carries over; certifying the scaled Q instead would lose
    // cheap certificates (a diagonally dominant Q need not stay so after
    // scaling -- QPLIB_8559 was refused that way).
    {
        std::string why;
        f64 slack = 0.0;
        if (!certify_psd(p, opts, why, slack))
            return solve_batch_core(p, lanes, opts, dev, diags);   // reports it
    }
    const auto t0 = Clock::now();
    const qpc::Scaling sc = qpc::ruiz(p, opts.ruiz_iterations);
    const QpProblem ps = qpc::apply_scaling(p, sc);
    std::vector<backend::LaneBounds> ls;
    for (const auto& b : lanes) ls.push_back(qpc::scale_bounds(b, sc));

    QpOptions o = opts;
    o.assume_psd = true;   // proved above, on the original Q
    std::vector<core::RawResult> raws;
    std::vector<qpc::OriginalKkt> kk(lanes.size());   // per lane, original units
    for (int attempt = 0;; ++attempt) {
        if (opts.time_limit_s > 0.0) {
            o.time_limit_s = opts.time_limit_s - ms_since(t0) / 1000.0;
            if (o.time_limit_s <= 0.0 && attempt > 0) break;
            o.time_limit_s = std::max(o.time_limit_s, std::numeric_limits<f64>::min());
        }
        raws = solve_batch_core(ps, ls, o, dev, diags);
        bool retry = false;
        for (std::size_t l = 0; l < lanes.size(); ++l) {
            auto& r = raws[l];
            if (r.proposed_status == core::Status::Unsupported ||
                r.x.size() != sz(p.linear.n_cols())) continue;
            for (std::size_t j = 0; j < r.x.size(); ++j) r.x[j] *= sc.dc[j];
            for (std::size_t i = 0; i < r.y.size(); ++i) r.y[i] *= sc.dr[i];
            kk[l] = qpc::kkt_original(p, lanes[l], r.x, r.y);
            // Scaled says done but the user's units disagree: tighten.
            if (r.proposed_status == core::Status::Optimal && !qpc::kkt_ok(kk[l], opts))
                retry = true;
        }
        if (!retry || attempt >= opts.tighten_retries) break;
        o.feas_tol *= 0.01;
        o.stationarity_tol *= 0.01;
        o.gap_tol *= 0.01;
    }

    for (std::size_t l = 0; l < lanes.size(); ++l) {
        auto& r = raws[l];
        auto& d = diags[l];
        if (r.proposed_status == core::Status::Unsupported ||
            r.x.size() != sz(p.linear.n_cols())) continue;
        auto& k = kk[l];
        bool ok = qpc::kkt_ok(k, opts);
        bool polished = false;
        if (!ok && opts.polish && !r.x.empty() &&
            (opts.time_limit_s <= 0.0 || ms_since(t0) < opts.time_limit_s * 1000.0))
            polished = ok = qpc::polish_scaled(p, lanes[l], sc, ps, ls[l], opts, r.x, r.y, k);
        d.primal_residual = k.primal;
        d.stationarity = k.dual_res;
        d.primal_residual_rel = k.primal_rel;
        d.stationarity_rel = k.dual_rel;
        d.primal_net = k.primal_net;
        d.stationarity_net = k.dual_net;
        d.gap_net = k.gap_net;
        d.gap_rel = k.gap;
        d.gap_finite = k.gap_finite;
        d.objective = k.objective;
        d.total_ms = ms_since(t0);
        r.objective = k.objective;
        r.proposed_status = ok ? core::Status::Optimal : core::Status::Interrupted;
        r.proposed_level = ok ? core::ProofLevel::ProvedKKT : core::ProofLevel::None;
        r.dual_bound = ok ? k.dual_bound : core::kNaN;
        r.termination_reason = ok ? (polished ? "PDHCG-II + polish: original-units KKT satisfied"
                                              : "PDHCG-II KKT and Wolfe gap satisfied (original units)")
                                  : "PDHCG-II limit before the original-units KKT test passed";
        d.termination_reason = r.termination_reason;
    }
    return raws;
}

core::RawResult solve_qp_general(const QpProblem& p, const QpOptions& opts,
                                 backend::PdhcgDevice& dev, QpDiagnostics& diag) {
    backend::LaneBounds b{p.linear.col_lo, p.linear.col_hi, p.linear.row_lo,
                          p.linear.row_hi};
    auto one = backend::make_lanes_view({&dev});
    std::vector<QpDiagnostics> diags;
    auto raws = solve_batch_impl(p, {b}, opts, *one, diags);
    diag = diags.front();
    return std::move(raws.front());
}

}  // namespace

core::RawResult solve_qp(const QpProblem& problem, const QpOptions& opts,
                         QpDiagnostics& diag) {
    diag = QpDiagnostics{};
    if (fast_path_eligible(problem)) {
        auto out = solve_qp_diag(problem, opts, diag);
        diag.convexity_certified = out.proposed_status != core::Status::Unsupported;
        if (out.proposed_status == core::Status::Optimal) {
            diag.gap_rel = 0.0;
            diag.gap_finite = true;
        }
        return out;
    }
    auto cpu = backend::make_cpu_pdhcg_device();
    return solve_qp_general(problem, opts, *cpu, diag);
}

std::vector<core::RawResult> solve_qp_pdhcg_batch(
    const QpProblem& shared, const std::vector<backend::LaneBounds>& lanes,
    const QpOptions& opts, backend::BatchedPdhcgDevice& device,
    std::vector<QpDiagnostics>& diags) {
    return solve_batch_impl(shared, lanes, opts, device, diags);
}

bool certify_qp_convex(const QpProblem& problem, const QpOptions& opts,
                       std::string& reason, f64& slack) {
    return certify_psd(problem, opts, reason, slack);
}

core::RawResult solve_qp_pdhcg(const QpProblem& problem, const QpOptions& opts,
                               backend::PdhcgDevice& device, QpDiagnostics& diag) {
    diag = QpDiagnostics{};
    return solve_qp_general(problem, opts, device, diag);
}

}  // namespace sor::engines
