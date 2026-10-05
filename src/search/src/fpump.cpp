#include "sor/search/fpump.hpp"

#include "sor/engines/simplex.hpp"
#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

struct Rng {
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed ? seed : 1u) {}
    std::uint32_t next() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }
    // Uniform in [0, 1).
    double uniform() { return (next() >> 8) * (1.0 / 16777216.0); }
    int range(int lo, int hi) {
        if (hi <= lo) return lo;
        return lo + static_cast<int>(next() % static_cast<std::uint32_t>(hi - lo + 1));
    }
};

}  // namespace

bool feasibility_pump(const model::LpProblem& lp,
                      const std::vector<f64>& col_lo,
                      const std::vector<f64>& col_hi,
                      const std::vector<f64>& x_start,
                      const FeasPumpOptions& opts,
                      std::vector<f64>& x_out,
                      FeasPumpDiagnostics& diag) {
    const auto t0 = Clock::now();
    const auto elapsed_s = [&]() {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    };
    const auto finish = [&](bool ok) {
        diag.ms = elapsed_s() * 1000.0;
        return ok;
    };
    const Index n = lp.n_cols();
    if (static_cast<Index>(x_start.size()) != n ||
        static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n || lp.is_integer.empty())
        return finish(false);

    std::vector<Index> ints;
    for (Index j = 0; j < n; ++j)
        if (lp.is_integer[sz(j)] && col_hi[sz(j)] > col_lo[sz(j)]) ints.push_back(j);
    if (ints.empty()) return finish(false);

    // Objective in minimisation form, and the scale that makes it comparable
    // with the distance term (Achterberg & Berthold: sqrt(|B|) / ||c||).
    const f64 sense = lp.maximize ? -1.0 : 1.0;
    f64 cnorm = 0.0;
    for (Index j = 0; j < n; ++j) cnorm += lp.c[sz(j)] * lp.c[sz(j)];
    cnorm = std::sqrt(cnorm);
    const f64 obj_scale = cnorm > 0.0
        ? std::sqrt(static_cast<f64>(ints.size())) / cnorm : 0.0;

    // Improvement mode: the objective must reach the cutoff (original sense).
    const bool has_cutoff = std::isfinite(opts.objective_cutoff);
    const auto within_cutoff = [&](const std::vector<f64>& p) {
        if (!has_cutoff) return true;
        const f64 v = lp.objective(p);
        return lp.maximize ? v >= opts.objective_cutoff - 1e-9 * (1.0 + std::fabs(v))
                           : v <= opts.objective_cutoff + 1e-9 * (1.0 + std::fabs(v));
    };

    Rng rng(opts.seed);
    std::vector<f64> x = x_start;
    std::vector<std::vector<f64>> history;   // last targets, oldest first
    f64 alpha = opts.alpha_start;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();

    // One prepared projection LP for every round whose target has no interior
    // integer (no auxiliary distance variables): only the objective changes
    // between such rounds, so the matrix, scaling and tolerances are built once
    // and each round is a warm primal simplex from the previous basis. Rounds
    // with interior targets change the LP's shape and use a fresh solve.
    std::unique_ptr<engines::PrimalCostSession> session;
    engines::SimplexBasis session_basis;
    bool session_warm = false;
    bool session_failed = false;
    for (int round = 0; round < opts.max_rounds; ++round) {
        if (opts.time_limit_s > 0.0 && elapsed_s() >= opts.time_limit_s) {
            diag.stopped_on_time = true;
            break;
        }
        diag.rounds = round + 1;

        // Round the LP point on the integer columns.
        std::vector<f64> target(sz(n), 0.0);
        f64 dist = 0.0;
        for (const Index j : ints) {
            f64 t = std::round(x[sz(j)]);
            t = std::min(std::max(t, col_lo[sz(j)]), col_hi[sz(j)]);
            target[sz(j)] = t;
            dist += std::fabs(x[sz(j)] - t);
        }
        diag.last_distance = dist;

        // Integral LP point: feasible for every row, continuous columns
        // included. Snap the integers and confirm before reporting.
        bool integral = true;
        for (const Index j : ints)
            if (std::fabs(x[sz(j)] - target[sz(j)]) > opts.int_tol) { integral = false; break; }
        if (integral) {
            std::vector<f64> cand = x;
            for (const Index j : ints) cand[sz(j)] = target[sz(j)];
            if (lp.max_row_violation(cand) <= 10.0 * opts.feas_tol &&
                lp.max_bound_violation(cand) <= 10.0 * opts.feas_tol &&
                within_cutoff(cand)) {
                x_out = std::move(cand);
                return finish(true);
            }
        }

        // Cycle handling on the TARGET, not on the LP point.
        if (!history.empty()) {
            const bool same_as_last = history.back() == target;
            bool longer_cycle = false;
            for (std::size_t h = 0; h + 1 < history.size(); ++h)
                if (history[h] == target) longer_cycle = true;
            if (longer_cycle) {
                // Randomised restart: flip where the rounding was not clear.
                ++diag.restarts;
                for (const Index j : ints) {
                    const f64 rho = -0.3 + rng.uniform();   // in [-0.3, 0.7)
                    const f64 dev = std::fabs(x[sz(j)] - target[sz(j)]);
                    if (dev + std::max(rho, 0.0) > 0.5) {
                        const f64 alt = x[sz(j)] > target[sz(j)] ? target[sz(j)] + 1.0
                                                                 : target[sz(j)] - 1.0;
                        if (alt >= col_lo[sz(j)] && alt <= col_hi[sz(j)])
                            target[sz(j)] = alt;
                    }
                }
            } else if (same_as_last) {
                // Flip the T columns whose LP value is farthest from target.
                const int flips = std::min<int>(
                    static_cast<int>(ints.size()),
                    rng.range(opts.flip_min, opts.flip_max));
                std::vector<Index> order = ints;
                std::partial_sort(order.begin(), order.begin() + flips, order.end(),
                                  [&](Index a, Index b) {
                                      return std::fabs(x[sz(a)] - target[sz(a)]) >
                                             std::fabs(x[sz(b)] - target[sz(b)]);
                                  });
                for (int q = 0; q < flips; ++q) {
                    const Index j = order[sz(q)];
                    if (std::fabs(x[sz(j)] - target[sz(j)]) <= 1e-9) break;
                    const f64 alt = x[sz(j)] > target[sz(j)] ? target[sz(j)] + 1.0
                                                             : target[sz(j)] - 1.0;
                    if (alt >= col_lo[sz(j)] && alt <= col_hi[sz(j)]) {
                        target[sz(j)] = alt;
                        ++diag.flips;
                    }
                }
            }
        }
        history.push_back(target);
        if (history.size() > 3) history.erase(history.begin());

        // Projection LP: same rows, tightened objective. Distance is linear
        // on columns whose target sits at a bound; a column whose target is
        // strictly inside gets an auxiliary variable and two rows.
        std::vector<Index> interior;
        for (const Index j : ints) {
            const f64 t = target[sz(j)];
            const bool at_lo = std::isfinite(col_lo[sz(j)]) && t <= col_lo[sz(j)];
            const bool at_hi = std::isfinite(col_hi[sz(j)]) && t >= col_hi[sz(j)];
            if (!at_lo && !at_hi) interior.push_back(j);
        }
        const Index na = static_cast<Index>(interior.size());
        if (na == 0 && !session_failed) {
            const f64 beta0 = 1.0 - alpha;
            std::vector<f64> costs(sz(n), 0.0);
            for (Index j = 0; j < n; ++j) costs[sz(j)] = alpha * obj_scale * sense * lp.c[sz(j)];
            for (const Index j : ints) {
                const f64 t = target[sz(j)];
                if (std::isfinite(col_lo[sz(j)]) && t <= col_lo[sz(j)]) costs[sz(j)] += beta0;
                else if (std::isfinite(col_hi[sz(j)]) && t >= col_hi[sz(j)]) costs[sz(j)] -= beta0;
            }
            engines::SimplexOptions so;
            so.presolve = false;
            so.primal_feas_tol = opts.feas_tol;
            so.max_iterations = opts.lp_max_iterations;
            const double left0 = opts.time_limit_s > 0.0 ? opts.time_limit_s - elapsed_s()
                                                         : opts.lp_time_limit_s;
            if (left0 <= 0.0) { diag.stopped_on_time = true; break; }
            so.time_limit_s = std::min(opts.lp_time_limit_s, left0);
            if (!session) {
                model::LpProblem base;
                base.name = lp.name + "_fpump";
                base.maximize = false;
                base.obj_offset = 0.0;
                base.c.assign(sz(n), 0.0);
                base.col_lo = col_lo;
                base.col_hi = col_hi;
                base.is_integer.assign(sz(n), false);
                base.row_lo = lp.row_lo;
                base.row_hi = lp.row_hi;
                std::vector<Index> rr, cc;
                std::vector<f64> vv;
                rr.reserve(sz(lp.nnz()) + sz(n));
                cc.reserve(rr.capacity());
                vv.reserve(rr.capacity());
                for (Index i = 0; i < lp.n_rows(); ++i)
                    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                        rr.push_back(i);
                        cc.push_back(ci[sz(k)]);
                        vv.push_back(lp.A.vals[sz(k)]);
                    }
                Index rows_total = lp.n_rows();
                if (has_cutoff) {
                    base.row_lo.push_back(-model::kInf);
                    base.row_hi.push_back(sense * (opts.objective_cutoff - lp.obj_offset));
                    for (Index j = 0; j < n; ++j)
                        if (lp.c[sz(j)] != 0.0) {
                            rr.push_back(rows_total);
                            cc.push_back(j);
                            vv.push_back(sense * lp.c[sz(j)]);
                        }
                    ++rows_total;
                }
                base.A = sparse::from_triplets(rows_total, n, rr, cc, vv);
                session = std::make_unique<engines::PrimalCostSession>(base, so);
                ++diag.session_builds;
            }
            session->set_costs(costs);
            engines::SimplexDiagnostics sd;
            engines::SimplexBasis next_basis;
            const auto pumped = session->solve(so, sd, &next_basis,
                                               session_warm ? &session_basis : nullptr);
            ++diag.lp_solves;
            diag.lp_iterations += sd.iterations;
            if (session_warm) ++diag.warm_solves;
            if ((pumped.proposed_status == core::Status::Optimal ||
                 pumped.proposed_status == core::Status::Feasible) &&
                static_cast<Index>(pumped.x.size()) >= n) {
                session_basis = std::move(next_basis);
                session_warm = !session_basis.basic.empty();
                x.assign(pumped.x.begin(), pumped.x.begin() + n);
                alpha *= opts.alpha_decay;
                continue;
            }
            // The persistent path failed (numerics, limit): drop it for this
            // call and let the ordinary cold solve below answer.
            session.reset();
            session_warm = false;
            session_failed = true;
        }
        model::LpProblem pump;
        pump.name = lp.name + "_fpump";
        pump.maximize = false;
        pump.obj_offset = 0.0;
        pump.c.assign(sz(n + na), 0.0);
        pump.col_lo = col_lo;
        pump.col_hi = col_hi;
        pump.col_lo.resize(sz(n + na), 0.0);
        pump.col_hi.resize(sz(n + na), model::kInf);
        pump.is_integer.assign(sz(n + na), false);
        pump.row_lo = lp.row_lo;
        pump.row_hi = lp.row_hi;
        // The objective row (improvement mode): sense*c'x <= sense*cutoff -
        // sense*offset, appended as the LAST original row so the layout below
        // (rows first, then the distance rows) stays as it is.
        const Index cutoff_rows = has_cutoff ? 1 : 0;
        if (has_cutoff) {
            pump.row_lo.push_back(-model::kInf);
            pump.row_hi.push_back(sense * (opts.objective_cutoff - lp.obj_offset));
        }
        for (Index j = 0; j < n; ++j)
            pump.c[sz(j)] = alpha * obj_scale * sense * lp.c[sz(j)];
        const f64 beta = 1.0 - alpha;
        for (const Index j : ints) {
            const f64 t = target[sz(j)];
            if (std::isfinite(col_lo[sz(j)]) && t <= col_lo[sz(j)])
                pump.c[sz(j)] += beta;                       // x_j - lo_j
            else if (std::isfinite(col_hi[sz(j)]) && t >= col_hi[sz(j)])
                pump.c[sz(j)] -= beta;                       // hi_j - x_j
        }
        if (na == 0 && cutoff_rows == 0) {
            pump.A = lp.A;
        } else {
            std::vector<Index> rows, cols;
            std::vector<f64> vals;
            rows.reserve(sz(lp.nnz()) + sz(4 * na) + sz(n));
            cols.reserve(rows.capacity());
            vals.reserve(rows.capacity());
            for (Index i = 0; i < lp.n_rows(); ++i)
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    rows.push_back(i);
                    cols.push_back(ci[sz(k)]);
                    vals.push_back(lp.A.vals[sz(k)]);
                }
            if (cutoff_rows == 1)
                for (Index j = 0; j < n; ++j)
                    if (lp.c[sz(j)] != 0.0) {
                        rows.push_back(lp.n_rows());
                        cols.push_back(j);
                        vals.push_back(sense * lp.c[sz(j)]);
                    }
            const Index dist_base = lp.n_rows() + cutoff_rows;
            for (Index q = 0; q < na; ++q) {
                const Index j = interior[sz(q)], d = n + q;
                const Index r1 = dist_base + 2 * q, r2 = r1 + 1;
                pump.c[sz(d)] = beta;
                // x_j - d <= t   and   -x_j - d <= -t
                pump.row_lo.push_back(-model::kInf); pump.row_hi.push_back(target[sz(j)]);
                pump.row_lo.push_back(-model::kInf); pump.row_hi.push_back(-target[sz(j)]);
                rows.push_back(r1); cols.push_back(j); vals.push_back(1.0);
                rows.push_back(r1); cols.push_back(d); vals.push_back(-1.0);
                rows.push_back(r2); cols.push_back(j); vals.push_back(-1.0);
                rows.push_back(r2); cols.push_back(d); vals.push_back(-1.0);
            }
            pump.A = sparse::from_triplets(dist_base + 2 * na, n + na, rows, cols, vals);
        }
        engines::SimplexOptions so;
        so.presolve = true;
        so.primal_feas_tol = opts.feas_tol;
        so.max_iterations = opts.lp_max_iterations;
        double left = opts.time_limit_s > 0.0 ? opts.time_limit_s - elapsed_s() : opts.lp_time_limit_s;
        // No floor above what is left: an LP started past the deadline would
        // spend time the caller does not have.
        if (left <= 0.0) { diag.stopped_on_time = true; break; }
        so.time_limit_s = std::min(opts.lp_time_limit_s, left);
        engines::SimplexDiagnostics sd;
        const auto pumped = engines::solve_simplex(pump, so, sd, nullptr);
        ++diag.lp_solves;
        diag.lp_iterations += sd.iterations;
        if ((pumped.proposed_status != core::Status::Optimal &&
             pumped.proposed_status != core::Status::Feasible) ||
            static_cast<Index>(pumped.x.size()) < n)
            break;
        x.assign(pumped.x.begin(), pumped.x.begin() + n);
        alpha *= opts.alpha_decay;
    }
    // The last projection may itself be integral: the loop only tests the
    // point at the top of a round, so a round-limit exit used to discard it.
    {
        bool integral = true;
        std::vector<f64> cand = x;
        for (const Index j : ints) {
            const f64 t = std::min(std::max(std::round(x[sz(j)]), col_lo[sz(j)]), col_hi[sz(j)]);
            if (std::fabs(x[sz(j)] - t) > opts.int_tol) { integral = false; break; }
            cand[sz(j)] = t;
        }
        if (integral && lp.max_row_violation(cand) <= 10.0 * opts.feas_tol &&
            lp.max_bound_violation(cand) <= 10.0 * opts.feas_tol && within_cutoff(cand)) {
            x_out = std::move(cand);
            return finish(true);
        }
    }
    return finish(false);
}

}  // namespace sor::search
