#include "sor/search/kernel_pump.hpp"

#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline bool is_int_val(f64 v, f64 tol) {
    return std::fabs(v - std::round(v)) <= tol;
}

inline bool binary_col(const model::LpProblem& lp, Index j) {
    if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) return false;
    return lp.col_lo[sz(j)] > -1e-9 && lp.col_lo[sz(j)] < 1e-9 &&
           lp.col_hi[sz(j)] > 1.0 - 1e-9 && lp.col_hi[sz(j)] < 1.0 + 1e-9;
}

inline f64 frac_dist(f64 v) {
    return std::fabs(v - std::round(v));
}

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

bool point_integer_feasible(const model::LpProblem& lp, const std::vector<f64>& x,
                            f64 int_tol, f64 feas_tol) {
    if (static_cast<Index>(x.size()) != lp.n_cols()) return false;
    if (lp.max_row_violation(x) > feas_tol) return false;
    if (lp.max_bound_violation(x) > feas_tol) return false;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (!is_int_val(x[sz(j)], int_tol)) return false;
    }
    return true;
}

// Round integer coordinates of `current` into `target`.
void round_integers(const model::LpProblem& lp, const std::vector<f64>& current,
                    f64 int_tol, std::vector<f64>& target) {
    target = current;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        f64 v = std::round(current[sz(j)]);
        v = std::min(std::max(v, lp.col_lo[sz(j)]), lp.col_hi[sz(j)]);
        if (is_int_val(v, int_tol)) target[sz(j)] = v;
    }
}

}  // namespace

bool build_kernel_buckets(const model::LpProblem& mip,
                          const std::vector<f64>& x_lp,
                          const std::vector<f64>* reduced_cost,
                          int kappa,
                          KernelBuckets& out) {
    out = KernelBuckets{};
    const Index n = mip.n_cols();
    if (static_cast<Index>(x_lp.size()) != n) return false;
    if (kappa < 1) kappa = 1;

    std::vector<Index> binaries;
    f64 d_min = std::numeric_limits<f64>::infinity();
    f64 d_max = 0.0;
    for (Index j = 0; j < n; ++j) {
        if (!binary_col(mip, j)) continue;
        binaries.push_back(j);
        const f64 d = frac_dist(x_lp[sz(j)]);
        d_min = std::min(d_min, d);
        d_max = std::max(d_max, d);
    }
    if (binaries.empty()) return false;

    const f64 span = std::max(d_max - d_min, 1e-12);
    const f64 theta = span / static_cast<f64>(kappa);
    const bool have_rc =
        reduced_cost != nullptr &&
        static_cast<Index>(reduced_cost->size()) == n;

    // Ordered groups: for each fractionality range, three RC sign buckets.
    struct Group {
        int range = 0;
        int rc_rank = 1;  // 0 = improving (-), 1 = zero, 2 = degrading (+)
        std::vector<Index> cols;
    };
    std::vector<Group> groups;
    groups.reserve(static_cast<std::size_t>(3 * (kappa + 1)));
    for (int r = 0; r <= kappa; ++r) {
        for (int s = 0; s < 3; ++s) {
            Group g;
            g.range = r;
            g.rc_rank = s;
            groups.push_back(g);
        }
    }

    for (Index j : binaries) {
        const f64 d = frac_dist(x_lp[sz(j)]);
        int range = kappa;
        if (d < d_max - 1e-15) {
            range = static_cast<int>(std::floor((d - d_min) / theta));
            if (range < 0) range = 0;
            if (range > kappa) range = kappa;
        }
        int rc_rank = 1;
        if (have_rc) {
            const f64 rc = (*reduced_cost)[sz(j)];
            // Minimization convention: negative RC prefers raising the var.
            if (rc < -1e-9) rc_rank = 0;
            else if (rc > 1e-9) rc_rank = 2;
        } else {
            // Without RC: prefer variables closer to 1 over those near 0
            // inside the same fractionality band.
            if (x_lp[sz(j)] >= 0.5) rc_rank = 0;
            else rc_rank = 2;
        }
        groups[static_cast<std::size_t>(range * 3 + rc_rank)].cols.push_back(j);
    }

    bool first = true;
    for (auto& g : groups) {
        if (g.cols.empty()) continue;
        if (first) {
            out.kernel = std::move(g.cols);
            first = false;
        } else {
            out.buckets.push_back(std::move(g.cols));
        }
    }
    if (out.kernel.empty() && !out.buckets.empty()) {
        out.kernel = std::move(out.buckets.front());
        out.buckets.erase(out.buckets.begin());
    }
    return !out.kernel.empty();
}

model::LpProblem apply_kernel_restriction(const model::LpProblem& mip,
                                          const std::vector<Index>& active) {
    model::LpProblem sub = mip;
    std::vector<char> on(static_cast<std::size_t>(mip.n_cols()), 0);
    for (Index j : active) {
        if (j >= 0 && j < mip.n_cols()) on[sz(j)] = 1;
    }
    for (Index j = 0; j < mip.n_cols(); ++j) {
        if (!binary_col(mip, j)) continue;
        if (on[sz(j)]) continue;
        // Fix excluded binaries to 0 (paper §3).
        const f64 lo = std::max(0.0, mip.col_lo[sz(j)]);
        const f64 hi = std::min(0.0, mip.col_hi[sz(j)]);
        if (lo <= hi + 1e-12) {
            sub.col_lo[sz(j)] = 0.0;
            sub.col_hi[sz(j)] = 0.0;
        } else {
            // Box excludes 0: fix to lower bound instead (still a restriction).
            sub.col_lo[sz(j)] = mip.col_lo[sz(j)];
            sub.col_hi[sz(j)] = mip.col_lo[sz(j)];
        }
    }
    return sub;
}

bool feasibility_pump_restricted(const model::LpProblem& mip,
                                 const std::vector<f64>& x_start,
                                 const KernelPumpOptions& opts,
                                 double time_limit_s,
                                 int max_pumps,
                                 std::vector<f64>& x_out,
                                 KernelPumpDiagnostics& diag) {
    if (static_cast<Index>(x_start.size()) != mip.n_cols()) return false;
    if (max_pumps <= 0 || time_limit_s <= 0.0) return false;

    const auto t0 = Clock::now();
    auto over = [&]() {
        return ms_since(t0) >= time_limit_s * 1000.0;
    };

    std::vector<f64> current = x_start;
    // Clip start into the (possibly restricted) box.
    for (Index j = 0; j < mip.n_cols(); ++j) {
        current[sz(j)] = std::min(std::max(current[sz(j)], mip.col_lo[sz(j)]),
                                  mip.col_hi[sz(j)]);
    }
    if (point_integer_feasible(mip, current, opts.int_tol, opts.feas_tol)) {
        x_out = std::move(current);
        return true;
    }

    std::vector<f64> previous_target;
    for (int round = 0; round < max_pumps && !over(); ++round) {
        ++diag.pumps;
        std::vector<f64> target;
        round_integers(mip, current, opts.int_tol, target);
        if (point_integer_feasible(mip, target, opts.int_tol, opts.feas_tol)) {
            // Target may violate rows; still try a cheap acceptance.
            if (mip.max_row_violation(target) <= opts.feas_tol &&
                mip.max_bound_violation(target) <= opts.feas_tol) {
                x_out = std::move(target);
                return true;
            }
        }
        if (target == previous_target) break;
        previous_target = target;

        // L1 projection LP toward the integer target (same construction as
        // bab's classical FP, scoped to this module for clean layering).
        const Index n = mip.n_cols();
        Index ni = 0;
        for (Index j = 0; j < n; ++j)
            if (!mip.is_integer.empty() && mip.is_integer[sz(j)]) ++ni;
        if (ni == 0) return false;

        model::LpProblem pump = mip;
        pump.name = mip.name + "_kp_fp";
        pump.maximize = false;
        pump.obj_offset = 0.0;
        pump.c.assign(static_cast<std::size_t>(n + ni), 0.0);
        pump.col_lo = mip.col_lo;
        pump.col_hi = mip.col_hi;
        pump.col_lo.resize(static_cast<std::size_t>(n + ni), 0.0);
        pump.col_hi.resize(static_cast<std::size_t>(n + ni), model::kInf);
        pump.is_integer.assign(static_cast<std::size_t>(n + ni), false);
        pump.row_lo = mip.row_lo;
        pump.row_hi = mip.row_hi;
        // Keep name vectors sized to the expanded LP. Presolve indexes
        // col_names/row_names by the live dimensions; leaving them short
        // (copying mip then adding aux vars/rows) is an OOB crash.
        pump.row_names = mip.row_names;
        if (!pump.row_names.empty() || !mip.row_names.empty())
            pump.row_names.resize(static_cast<std::size_t>(mip.n_rows() + 2 * ni));
        pump.col_names = mip.col_names;
        if (!pump.col_names.empty() || !mip.col_names.empty())
            pump.col_names.resize(static_cast<std::size_t>(n + ni));

        std::vector<Index> rows, cols;
        std::vector<f64> vals;
        const auto& rp = mip.A.pattern.row_ptr();
        const auto& ci = mip.A.pattern.col_idx();
        for (Index i = 0; i < mip.n_rows(); ++i) {
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                rows.push_back(i);
                cols.push_back(ci[sz(k)]);
                vals.push_back(mip.A.vals[sz(k)]);
            }
        }
        Index d = 0;
        for (Index j = 0; j < n; ++j) {
            if (mip.is_integer.empty() || !mip.is_integer[sz(j)]) continue;
            const Index dj = n + d;
            pump.c[sz(dj)] = 1.0;
            if (!pump.col_names.empty())
                pump.col_names[sz(dj)] = "KP_D_" + std::to_string(j);
            const Index r1 = mip.n_rows() + 2 * d;
            const Index r2 = r1 + 1;
            pump.row_lo.push_back(-model::kInf);
            pump.row_hi.push_back(target[sz(j)]);
            pump.row_lo.push_back(-model::kInf);
            pump.row_hi.push_back(-target[sz(j)]);
            if (!pump.row_names.empty()) {
                pump.row_names[sz(r1)] = "KP_FP_P_" + std::to_string(j);
                pump.row_names[sz(r2)] = "KP_FP_N_" + std::to_string(j);
            }
            rows.push_back(r1); cols.push_back(j);  vals.push_back(1.0);
            rows.push_back(r1); cols.push_back(dj); vals.push_back(-1.0);
            rows.push_back(r2); cols.push_back(j);  vals.push_back(-1.0);
            rows.push_back(r2); cols.push_back(dj); vals.push_back(-1.0);
            ++d;
        }
        pump.A = sparse::from_triplets(mip.n_rows() + 2 * ni, n + ni,
                                       rows, cols, vals);
        pump.validate();

        engines::SimplexOptions so;
        so.method = engines::SimplexMethod::Primal;
        so.presolve = true;
        so.max_iterations = opts.max_lp_iterations;
        const double left =
            time_limit_s - ms_since(t0) / 1000.0;
        so.time_limit_s = std::max(0.01, left);
        so.primal_feas_tol = opts.feas_tol;
        so.dual_feas_tol = std::max(opts.feas_tol, 1e-7);
        engines::SimplexDiagnostics sd;
        const auto pumped = engines::solve_simplex(pump, so, sd, nullptr);
        ++diag.lp_solves;
        diag.lp_iterations += sd.iterations;
        if (pumped.proposed_status != core::Status::Optimal &&
            pumped.proposed_status != core::Status::Feasible)
            break;
        if (static_cast<Index>(pumped.x.size()) < n) break;
        current.assign(pumped.x.begin(), pumped.x.begin() + n);
        if (point_integer_feasible(mip, current, opts.int_tol, opts.feas_tol)) {
            x_out = std::move(current);
            return true;
        }
    }
    // Best effort: return least-fractional current if somehow integral later.
    if (point_integer_feasible(mip, current, opts.int_tol, opts.feas_tol)) {
        x_out = std::move(current);
        return true;
    }
    return false;
}

bool kernel_pump(const model::LpProblem& mip,
                 const std::vector<f64>& x_lp,
                 const std::vector<f64>* reduced_cost,
                 const KernelPumpOptions& opts,
                 std::vector<f64>& x_out,
                 KernelPumpDiagnostics& diag) {
    diag = KernelPumpDiagnostics{};
    const auto t0 = Clock::now();
    if (!opts.enabled || opts.time_limit_s <= 0.0) {
        diag.ms = ms_since(t0);
        return false;
    }
    if (static_cast<Index>(x_lp.size()) != mip.n_cols()) {
        diag.ms = ms_since(t0);
        return false;
    }

    KernelBuckets kb;
    if (!build_kernel_buckets(mip, x_lp, reduced_cost, opts.kappa, kb)) {
        // No binaries: one restricted FP on the full model.
        const bool ok = feasibility_pump_restricted(
            mip, x_lp, opts, opts.time_limit_s, opts.max_pumps_total, x_out,
            diag);
        diag.found = ok;
        diag.ms = ms_since(t0);
        return ok;
    }
    diag.kernel_size = kb.kernel.size();
    diag.buckets = kb.buckets.size();

    // Optional kernel refinement (paper §3.1.1): min sum of out-of-kernel
    // binaries s.t. LP feasibility — add any positive ones into the kernel.
    if (opts.refine_kernel && opts.time_limit_s > 0.2) {
        std::vector<char> in_k(static_cast<std::size_t>(mip.n_cols()), 0);
        for (Index j : kb.kernel) in_k[sz(j)] = 1;
        model::LpProblem refine = mip;
        refine.maximize = false;
        refine.obj_offset = 0.0;
        refine.c.assign(static_cast<std::size_t>(mip.n_cols()), 0.0);
        for (Index j = 0; j < mip.n_cols(); ++j) {
            if (binary_col(mip, j) && !in_k[sz(j)]) refine.c[sz(j)] = 1.0;
            if (!refine.is_integer.empty())
                refine.is_integer[sz(j)] = false;  // pure LP
        }
        engines::SimplexOptions so;
        so.method = engines::SimplexMethod::Dual;
        so.presolve = true;
        so.max_iterations = opts.max_lp_iterations;
        so.time_limit_s = std::min(0.25, opts.time_limit_s * 0.15);
        so.primal_feas_tol = opts.feas_tol;
        engines::SimplexDiagnostics sd;
        const auto rr = engines::solve_simplex(refine, so, sd, nullptr);
        ++diag.lp_solves;
        diag.lp_iterations += sd.iterations;
        if ((rr.proposed_status == core::Status::Optimal ||
             rr.proposed_status == core::Status::Feasible) &&
            static_cast<Index>(rr.x.size()) == mip.n_cols()) {
            for (Index j = 0; j < mip.n_cols(); ++j) {
                if (!binary_col(mip, j) || in_k[sz(j)]) continue;
                if (rr.x[sz(j)] > 1e-6) {
                    kb.kernel.push_back(j);
                    in_k[sz(j)] = 1;
                }
            }
            // Strip refined vars from later buckets.
            for (auto& b : kb.buckets) {
                b.erase(std::remove_if(b.begin(), b.end(),
                                       [&](Index j) { return in_k[sz(j)]; }),
                        b.end());
            }
            diag.kernel_size = kb.kernel.size();
        }
    }

    const std::size_t n_stages = 1 + kb.buckets.size();
    const double per =
        opts.time_limit_s / static_cast<double>(std::max<std::size_t>(n_stages, 1));
    int pumps_left = opts.max_pumps_total;
    std::vector<Index> active = kb.kernel;
    std::vector<f64> warm = x_lp;

    auto run_stage = [&](double budget, bool last) -> bool {
        if (budget <= 0.01 || pumps_left <= 0) return false;
        const int cap = last ? pumps_left
                             : std::min(pumps_left, opts.max_pumps_per_bucket);
        const std::uint64_t pumps_before = diag.pumps;
        model::LpProblem sub = apply_kernel_restriction(mip, active);
        std::vector<f64> cand;
        const bool ok = feasibility_pump_restricted(sub, warm, opts, budget,
                                                    cap, cand, diag);
        const int used =
            static_cast<int>(diag.pumps - pumps_before);
        pumps_left = std::max(0, pumps_left - std::max(used, 1));
        if (static_cast<Index>(cand.size()) == mip.n_cols()) warm = cand;
        if (!ok) return false;
        // Re-check on the ORIGINAL model: fixing binaries to 0 is a
        // restriction, so any feasible point of `sub` is feasible for mip.
        if (!point_integer_feasible(mip, cand, opts.int_tol, opts.feas_tol))
            return false;
        x_out = std::move(cand);
        return true;
    };

    const double spent0 = ms_since(t0) / 1000.0;
    if (run_stage(std::max(0.05, per - spent0 * 0.5), kb.buckets.empty())) {
        diag.found = true;
        diag.ms = ms_since(t0);
        return true;
    }

    for (std::size_t b = 0; b < kb.buckets.size(); ++b) {
        const double spent = ms_since(t0) / 1000.0;
        if (spent >= opts.time_limit_s) break;
        active.insert(active.end(), kb.buckets[b].begin(), kb.buckets[b].end());
        const bool last = (b + 1 == kb.buckets.size());
        const double budget =
            last ? std::max(0.05, opts.time_limit_s - spent)
                 : std::max(0.05, std::min(per, opts.time_limit_s - spent));
        if (run_stage(budget, last)) {
            diag.found = true;
            diag.ms = ms_since(t0);
            return true;
        }
    }

    diag.found = false;
    diag.ms = ms_since(t0);
    return false;
}

}  // namespace sor::search
