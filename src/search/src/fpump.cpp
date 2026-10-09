#include "forge/search/fpump.hpp"
#include "forge/search/l1_projection.hpp"
#include "heuristic_lp.hpp"
#include "forge/core/work_clock.hpp"

#include "forge/engines/simplex.hpp"
#include "forge/sparse/csr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>

namespace forge::search {
namespace {

using Clock = forge::core::SolverClock;
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
    core::WorkScope work_scope_(core::WorkCat::Heuristic);
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
    // One matrix for every target, including interior general-integer targets.
    std::unique_ptr<L1ProjectionSession> session;
    for (const Index j : ints)
        if (std::ceil(col_lo[sz(j)]) > std::floor(col_hi[sz(j)]))
            return finish(false);
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
            t = std::min(std::max(t, std::ceil(col_lo[sz(j)])), std::floor(col_hi[sz(j)]));
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

        // The epigraph layout depends on the box, never on this target.
        engines::SimplexOptions so;
        heuristic_lp_options(so);
        so.presolve = false;
        so.primal_feas_tol = opts.feas_tol;
        so.max_iterations = opts.lp_max_iterations;
        double left = opts.time_limit_s > 0.0 ? opts.time_limit_s - elapsed_s() : opts.lp_time_limit_s;
        // No floor above what is left: an LP started past the deadline would
        // spend time the caller does not have.
        if (left <= 0.0) { diag.stopped_on_time = true; break; }
        so.time_limit_s = std::min(opts.lp_time_limit_s, left);
        engines::SimplexDiagnostics sd;
        if (!session) {
            session = std::make_unique<L1ProjectionSession>(
                lp, col_lo, col_hi, ints, so, opts.objective_cutoff);
            ++diag.session_builds;
        }
        if (session->has_basis()) ++diag.warm_solves;
        const auto pumped = session->solve(target, alpha * obj_scale, 1.0 - alpha, so, sd);
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
            const f64 t = std::min(std::max(std::round(x[sz(j)]), std::ceil(col_lo[sz(j)])), std::floor(col_hi[sz(j)]));
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

}  // namespace forge::search
