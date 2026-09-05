#include "sor/search/bab.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/search/propagate.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline bool is_integral(f64 v, f64 tol) {
    return std::fabs(v - std::round(v)) <= tol;
}

inline f64 frac_score(f64 v) {
    const f64 f = std::fabs(v - std::floor(v));
    return std::min(f, 1.0 - f);  // distance to nearest integer
}

// Positive means that increasing an integer column is generally useful for
// the relaxation/incumbent heuristics (coverage of lower/equality rows or a
// favorable objective coefficient); negative favors decreasing it.
f64 integer_up_bias(const model::LpProblem& lp, Index j) {
    if (j < 0 || j >= lp.n_cols()) return 0.0;
    const f64 sense = lp.maximize ? -1.0 : 1.0;
    f64 bias = -sense * lp.c[sz(j)];
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const bool has_lo = std::isfinite(lp.row_lo[sz(i)]);
        const bool has_hi = std::isfinite(lp.row_hi[sz(i)]);
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (ci[sz(k)] != j) continue;
            const f64 a = lp.A.vals[sz(k)];
            if (has_lo && !has_hi) bias += a;
            else if (has_hi && !has_lo) bias -= a;
            else bias += std::fabs(a);
        }
    }
    return bias;
}

enum class RoundMode : std::uint8_t {
    Objective,
    Nearest,
    Ceil,
};

struct Node {
    std::vector<f64> col_lo;
    std::vector<f64> col_hi;
    engines::SimplexBasis basis;
    bool has_basis = false;
    f64 bound = -std::numeric_limits<f64>::infinity();  // dual bound (min sense)
    int depth = 0;
    Index parent_branch_var = -1;
    int parent_branch_dir = 0;       // -1 = down, +1 = up
    f64 parent_bound = core::kNaN;
    f64 parent_branch_distance = 0.0;
    // Length of the current unbroken plunge chain this node was created in
    // (0 if it entered via the best-bound queue). Only used to cap how long
    // a single dive can run before falling back to best-bound; see the
    // node-selection block below.
    int plunge_len = 0;
};

// Best-bound first (minimize): smallest bound first.
struct NodeCmp {
    bool operator()(const Node& a, const Node& b) const {
        if (a.bound != b.bound) return a.bound > b.bound;  // min-heap via greater
        return a.depth < b.depth;
    }
};

// Round integer columns and repair row violations with local integer moves.
// This is a small feasibility-pump style heuristic: it is deliberately
// bounded and never used as a proof, but it is enough to turn the fractional
// schedule relaxation into an incumbent instead of discarding the root node.
bool try_round(const model::LpProblem& lp,
               const std::vector<f64>& x_lp,
               f64 int_tol,
               f64 feas_tol,
               std::vector<f64>& x_out,
               RoundMode mode = RoundMode::Objective) {
    const Index n = lp.n_cols();
    if (static_cast<Index>(x_lp.size()) != n) return false;
    x_out = x_lp;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        // Cost-aware rounding is the first step of objective diving. For a
        // minimization, positive reduced-cost directions prefer the floor and
        // negative-cost directions prefer the ceil (reversed for maximize).
        // Zero-cost structural variables keep nearest rounding so covering and
        // equality repairs retain the LP point's combinatorial signal.
        const f64 work_cost = (lp.maximize ? -1.0 : 1.0) * lp.c[sz(j)];
        f64 v = 0.0;
        if (mode == RoundMode::Ceil) {
            v = std::ceil(x_lp[sz(j)]);
        } else if (mode == RoundMode::Nearest) {
            v = std::round(x_lp[sz(j)]);
        } else {
            v = work_cost > int_tol ? std::floor(x_lp[sz(j)])
                                    : work_cost < -int_tol
                                          ? std::ceil(x_lp[sz(j)])
                                          : std::round(x_lp[sz(j)]);
        }
        v = std::min(std::max(v, lp.col_lo[sz(j)]), lp.col_hi[sz(j)]);
        if (!is_integral(v, int_tol)) {
            v = std::ceil(lp.col_lo[sz(j)] - int_tol);
            if (v > lp.col_hi[sz(j)] + int_tol) return false;
        }
        x_out[sz(j)] = v;
    }

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    const Index m = lp.n_rows();
    std::vector<f64> activity(sz(m), 0.0);
    std::vector<std::vector<std::pair<Index, f64>>> col_rows(sz(n));
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            activity[sz(i)] += av[sz(k)] * x_out[sz(j)];
            col_rows[sz(j)].emplace_back(i, av[sz(k)]);
        }
    const auto row_violation = [&](Index i, f64 act) {
        f64 v = 0.0;
        if (act < lp.row_lo[sz(i)] - feas_tol)
            v += lp.row_lo[sz(i)] - act;
        if (act > lp.row_hi[sz(i)] + feas_tol)
            v += act - lp.row_hi[sz(i)];
        return v;
    };
    auto total_violation = [&]() {
        f64 v = 0.0;
        for (Index i = 0; i < m; ++i) v += row_violation(i, activity[sz(i)]);
        return v;
    };

    // Select one move using the change in total violation over every row the
    // column touches. This prevents the old row-by-row repair from fixing one
    // equality and immediately breaking another coupled equality.
    const int max_passes = std::min(4096, std::max(64, 2 * static_cast<int>(n)));
    for (int pass = 0; pass < max_passes; ++pass) {
        const f64 before = total_violation();
        if (before <= 0.0) break;
        Index best_j = -1;
        f64 best_next = 0.0;
        f64 best_gain = 0.0;
        f64 best_score = -std::numeric_limits<f64>::infinity();
        for (Index j = 0; j < n; ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            const f64 xj = std::round(x_out[sz(j)]);
            for (int dir : {-1, 1}) {
                f64 next = xj + static_cast<f64>(dir);
                next = std::min(std::max(next, lp.col_lo[sz(j)]),
                                lp.col_hi[sz(j)]);
                if (std::fabs(next - xj) <= int_tol) continue;
                const f64 delta = next - xj;
                f64 after = before;
                for (const auto& [i, a] : col_rows[sz(j)]) {
                    after += row_violation(i, activity[sz(i)] + a * delta) -
                             row_violation(i, activity[sz(i)]);
                }
                const f64 gain = before - after;
                const f64 obj_delta = (lp.maximize ? -1.0 : 1.0) *
                                      lp.c[sz(j)] * delta;
                const f64 score = gain / (1.0 + std::max(0.0, obj_delta)) +
                                  1e-9 * std::fabs(delta);
                if (score > best_score ||
                    (score == best_score && gain > best_gain)) {
                    best_score = score;
                    best_gain = gain;
                    best_j = j;
                    best_next = next;
                }
            }
        }
        // Never accept a move that makes global violation worse. A zero-gain
        // move is also rejected; escaping a true plateau is delegated to the
        // diversified construct/neighborhood heuristics.
        if (best_j < 0 || best_gain <= 1e-12) break;
        const f64 delta = best_next - x_out[sz(best_j)];
        x_out[sz(best_j)] = best_next;
        for (const auto& [i, a] : col_rows[sz(best_j)])
            activity[sz(i)] += a * delta;
    }

    if (lp.max_row_violation(x_out) > feas_tol) return false;
    if (lp.max_bound_violation(x_out) > feas_tol) return false;
    return true;
}

// Fix the rounded integer columns and use the LP engine to repair continuous
// columns and equality rows. This is bounded and heuristic-only: its result is
// accepted only after checking the original MILP, and it never supplies a B&B
// bound or pruning certificate.
bool try_lp_rounding_repair(const model::LpProblem& lp,
                            const std::vector<f64>& x_lp,
                            f64 int_tol,
                            f64 feas_tol,
                            std::uint64_t max_iterations,
                            double time_limit_s,
                            std::vector<f64>& x_out) {
    if (static_cast<Index>(x_lp.size()) != lp.n_cols()) return false;
    model::LpProblem fixed = lp;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        f64 v = std::round(x_lp[sz(j)]);
        if (!std::isfinite(v)) return false;
        v = std::min(std::max(v, lp.col_lo[sz(j)]), lp.col_hi[sz(j)]);
        if (!std::isfinite(v) || !is_integral(v, int_tol)) return false;
        fixed.col_lo[sz(j)] = v;
        fixed.col_hi[sz(j)] = v;
    }

    engines::SimplexOptions repair_opts;
    repair_opts.method = engines::SimplexMethod::Auto;
    repair_opts.presolve = true;
    repair_opts.max_iterations = max_iterations;
    repair_opts.time_limit_s = time_limit_s;
    repair_opts.primal_feas_tol = feas_tol;
    repair_opts.dual_feas_tol = std::max(feas_tol, 1e-7);
    engines::SimplexDiagnostics repair_diag;
    const auto repaired = engines::solve_simplex(fixed, repair_opts,
                                                 repair_diag, nullptr);
    if (repaired.proposed_status != core::Status::Optimal &&
        repaired.proposed_status != core::Status::Feasible)
        return false;
    if (static_cast<Index>(repaired.x.size()) != lp.n_cols()) return false;
    if (lp.max_row_violation(repaired.x) > feas_tol ||
        lp.max_bound_violation(repaired.x) > feas_tol)
        return false;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (!lp.is_integer.empty() && lp.is_integer[sz(j)] &&
            !is_integral(repaired.x[sz(j)], int_tol))
            return false;
    }
    x_out = repaired.x;
    return true;
}

// Enumerate the small fractional integer core of an LP relaxation. This is a
// bounded exact neighborhood search: variables already integral in the LP are
// fixed to their nearest integer, while each remaining integer variable gets
// its floor/ceil alternatives. RENS/RINS-style neighborhoods of this form are
// particularly effective on degenerate low-row MIPs such as markshare.
bool try_fractional_enumeration(const model::LpProblem& lp,
                                const std::vector<f64>& x_lp,
                                f64 int_tol,
                                f64 feas_tol,
                                std::uint64_t max_combinations,
                                double time_limit_s,
                                std::uint64_t repair_iterations,
                                double repair_time_s,
                                std::vector<f64>& x_out) {
    if (static_cast<Index>(x_lp.size()) != lp.n_cols() || max_combinations == 0)
        return false;
    std::vector<Index> fractional;
    std::vector<f64> base = x_lp;
    std::vector<std::array<f64, 2>> choices;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        const f64 v = x_lp[sz(j)];
        if (!std::isfinite(v)) return false;
        const f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
        const f64 nearest = std::round(v);
        if (is_integral(v, int_tol)) {
            base[sz(j)] = std::min(std::max(nearest, lo), hi);
            continue;
        }
        const f64 down = std::min(std::max(std::floor(v), lo), hi);
        const f64 up = std::min(std::max(std::ceil(v), lo), hi);
        if (!is_integral(down, int_tol) || !is_integral(up, int_tol)) return false;
        fractional.push_back(j);
        choices.push_back({down, up});
    }
    if (fractional.empty() || fractional.size() >= 63) return false;
    const std::uint64_t combinations = 1ull << fractional.size();
    const std::uint64_t limit = std::min(combinations, max_combinations);
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    bool found = false;
    f64 best_obj = lp.maximize ? -std::numeric_limits<f64>::infinity()
                               : std::numeric_limits<f64>::infinity();
    std::vector<f64> best;
    for (std::uint64_t mask = 0; mask < limit && !over_budget(); ++mask) {
        std::vector<f64> candidate = base;
        for (std::size_t q = 0; q < fractional.size(); ++q)
            candidate[sz(fractional[q])] = choices[q][(mask >> q) & 1ull];
        std::vector<f64> repaired;
        if (!try_lp_rounding_repair(lp, candidate, int_tol, feas_tol,
                                    repair_iterations, repair_time_s,
                                    repaired)) continue;
        const f64 obj = lp.objective(repaired);
        if (!std::isfinite(obj)) continue;
        if (!found || (lp.maximize ? obj > best_obj : obj < best_obj)) {
            found = true;
            best_obj = obj;
            best = std::move(repaired);
        }
    }
    if (found) x_out = std::move(best);
    return found;
}

// Generate additional LP extreme points before rounding. Degenerate models
// such as markshare and pk1 have a zero-cost face containing many fractional
// optima; relying on one simplex basis repeatedly rounds the same poor binary
// pattern. Tiny deterministic objective perturbations select different
// vertices while leaving the original MILP objective untouched.
bool try_perturbed_rounding(const model::LpProblem& lp,
                            const engines::SimplexOptions& base_opts,
                            f64 int_tol,
                            f64 feas_tol,
                            int attempts,
                            double time_limit_s,
                            std::vector<f64>& x_out) {
    if (attempts <= 0 || lp.n_cols() == 0) return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    const f64 sense = lp.maximize ? -1.0 : 1.0;
    f64 scale = 1.0;
    for (const f64 c : lp.c) scale = std::max(scale, std::fabs(c));
    std::vector<f64> benefit(static_cast<std::size_t>(lp.n_cols()), 0.0);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const bool has_lo = std::isfinite(lp.row_lo[sz(i)]);
        const bool has_hi = std::isfinite(lp.row_hi[sz(i)]);
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = lp.A.vals[sz(k)];
            // Positive contribution toward a finite lower bound is useful;
            // positive contribution toward an upper bound is costly. For an
            // equality/slack row, maximizing absolute coverage is a useful
            // tie-break because the continuous slack then shrinks.
            if (has_lo && !has_hi) benefit[sz(j)] += a;
            else if (has_hi && !has_lo) benefit[sz(j)] -= a;
            else benefit[sz(j)] += std::fabs(a);
        }
    }

    bool found = false;
    f64 best_obj = lp.maximize ? -std::numeric_limits<f64>::infinity()
                               : std::numeric_limits<f64>::infinity();
    std::uint64_t state = 0xd1b54a32d192ed03ull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };
    for (int pass = 0; pass < attempts && !over_budget(); ++pass) {
        model::LpProblem probe = lp;
        const f64 eps = 1.0e-4 * scale;
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            const f64 noise = static_cast<f64>(
                static_cast<long long>(next_rand() % 2001) - 1000) / 1000.0;
            // Minimize in the working sense. The benefit term makes the LP
            // prefer useful coverage; noise breaks ties between equivalent
            // vertices and is intentionally much smaller than that signal.
            probe.c[sz(j)] -= sense * eps * (benefit[sz(j)] + 0.01 * noise);
        }
        engines::SimplexOptions solve_opts = base_opts;
        solve_opts.method = engines::SimplexMethod::Primal;
        solve_opts.presolve = true;
        solve_opts.max_iterations = std::max<std::uint64_t>(
            20000, base_opts.max_iterations == 0 ? 0 : base_opts.max_iterations);
        solve_opts.time_limit_s = std::min(
            0.08, time_limit_s > 0.0
                ? std::max(0.01, time_limit_s -
                    std::chrono::duration<double>(Clock::now() - start).count())
                : 0.08);
        engines::SimplexDiagnostics sd;
        const auto r = engines::solve_simplex(probe, solve_opts, sd, nullptr);
        if (r.proposed_status != core::Status::Optimal &&
            r.proposed_status != core::Status::Feasible)
            continue;
        for (RoundMode mode : {RoundMode::Objective, RoundMode::Nearest,
                               RoundMode::Ceil}) {
            std::vector<f64> candidate;
            if (!try_round(lp, r.x, int_tol, feas_tol, candidate, mode)) continue;
            const f64 obj = lp.objective(candidate);
            if (!std::isfinite(obj)) continue;
            if (!found || (lp.maximize ? obj > best_obj : obj < best_obj)) {
                found = true;
                best_obj = obj;
                x_out = std::move(candidate);
            }
        }
    }
    return found;
}

// Diversified constructive search for degenerate packing/equality MILPs. It
// builds integer assignments from row deficits without solving an LP for every
// failed trial, then sends only the most promising assignments through the
// exact continuous LP repair. This is a bounded incumbent heuristic.
bool try_randomized_construct(const model::LpProblem& lp,
                              f64 int_tol,
                              f64 feas_tol,
                              int restarts,
                              int lp_trials,
                              double time_limit_s,
                              std::uint64_t repair_iterations,
                              double repair_time_s,
                              std::vector<f64>& x_out) {
    const Index n = lp.n_cols();
    if (restarts <= 0 || lp_trials <= 0 || n == 0) return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    std::vector<Index> ints;
    for (Index j = 0; j < n; ++j)
        if (!lp.is_integer.empty() && lp.is_integer[sz(j)]) ints.push_back(j);
    if (ints.empty()) return false;

    struct Candidate {
        f64 score = std::numeric_limits<f64>::infinity();
        std::vector<f64> x;
    };
    std::vector<Candidate> pool;
    pool.reserve(static_cast<std::size_t>(lp_trials));
    std::uint64_t state = 0x243f6a8885a308d3ull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };
    const auto row_penalty = [&](const std::vector<f64>& act) {
        f64 p = 0.0;
        for (Index i = 0; i < lp.n_rows(); ++i) {
            const f64 lo = lp.row_lo[sz(i)], hi = lp.row_hi[sz(i)];
            if (act[sz(i)] < lo) p += lo - act[sz(i)];
            if (act[sz(i)] > hi) p += act[sz(i)] - hi;
        }
        return p;
    };
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<std::vector<std::pair<Index, f64>>> col_rows(sz(n));
    for (Index i = 0; i < lp.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            col_rows[sz(ci[sz(k)])].emplace_back(i, av[sz(k)]);

    for (int restart = 0; restart < restarts && !over_budget(); ++restart) {
        std::vector<f64> trial(static_cast<std::size_t>(n), 0.0);
        for (const Index j : ints) {
            f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
            if (!std::isfinite(lo)) lo = 0.0;
            f64 v = std::ceil(lo - int_tol);
            if (std::isfinite(hi) && hi > v &&
                (next_rand() & 3ull) == 0ull)
                v = std::floor(hi + int_tol);
            trial[sz(j)] = std::min(std::max(v, lp.col_lo[sz(j)]), hi);
        }
        std::vector<f64> act(static_cast<std::size_t>(lp.n_rows()), 0.0);
        for (Index i = 0; i < lp.n_rows(); ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                act[sz(i)] += av[sz(k)] * trial[sz(ci[sz(k)])];

        // Randomized greedy coordinate descent on row violation. Several
        // passes let a general integer move more than one unit; random noise
        // changes the order among equal-cost covering choices.
        const int passes = std::min<int>(32, std::max<int>(4, ints.size()));
        for (int pass = 0; pass < passes && !over_budget(); ++pass) {
            Index best_j = -1;
            f64 best_next = 0.0;
            f64 best_gain = 0.0;
            for (const Index j : ints) {
                const f64 v = std::round(trial[sz(j)]);
                for (int dir : {-1, 1}) {
                    f64 next = v + static_cast<f64>(dir);
                    next = std::min(std::max(next, lp.col_lo[sz(j)]),
                                    lp.col_hi[sz(j)]);
                    if (std::fabs(next - v) <= int_tol) continue;
                    const f64 delta = next - v;
                    f64 gain = 0.0;
                    for (const auto& [i, a] : col_rows[sz(j)]) {
                        const f64 old = act[sz(i)];
                        const f64 neu = old + a * delta;
                        const f64 lo = lp.row_lo[sz(i)], hi = lp.row_hi[sz(i)];
                        const f64 po = (old < lo ? lo - old : old > hi ? old - hi : 0.0);
                        const f64 pn = (neu < lo ? lo - neu : neu > hi ? neu - hi : 0.0);
                        gain += po - pn;
                    }
                    gain += 1.0e-6 *
                        static_cast<f64>(next_rand() % 1000) / 1000.0;
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_j = j;
                        best_next = next;
                    }
                }
            }
            if (best_j < 0 || best_gain <= 1.0e-7) break;
            const f64 delta = best_next - trial[sz(best_j)];
            trial[sz(best_j)] = best_next;
            for (const auto& [i, a] : col_rows[sz(best_j)])
                act[sz(i)] += a * delta;
        }
        Candidate c;
        c.score = row_penalty(act);
        c.x = std::move(trial);
        pool.push_back(std::move(c));
    }

    std::sort(pool.begin(), pool.end(), [](const Candidate& a, const Candidate& b) {
        return a.score < b.score;
    });
    if (static_cast<int>(pool.size()) > lp_trials)
        pool.resize(static_cast<std::size_t>(lp_trials));
    bool found = false;
    f64 best_obj = lp.maximize ? -std::numeric_limits<f64>::infinity()
                               : std::numeric_limits<f64>::infinity();
    int tested = 0;
    for (const Candidate& c : pool) {
        if (tested++ >= lp_trials || over_budget()) break;
        std::vector<f64> repaired;
        if (!try_lp_rounding_repair(lp, c.x, int_tol, feas_tol,
                                    repair_iterations, repair_time_s,
                                    repaired))
            continue;
        const f64 obj = lp.objective(repaired);
        if (!std::isfinite(obj)) continue;
        if (!found || (lp.maximize ? obj > best_obj : obj < best_obj)) {
            found = true;
            best_obj = obj;
            x_out = std::move(repaired);
        }
    }
    return found;
}

// Cost-aware constructor for generalized covering models.  A common MIPLIB
// pattern (gt2/transportation variants) has positive lower-bound demand rows,
// positive upper-bound availability rows, and each integer column linking one
// row of each type.  Greedy LP rounding is particularly weak there because it
// spreads fractional demand over expensive columns; this constructor builds
// integral units directly and then deletes redundant units.
bool try_covering_construct(const model::LpProblem& lp,
                            f64 int_tol,
                            f64 feas_tol,
                            int restarts,
                            double time_limit_s,
                            std::vector<f64>& x_out) {
    if (lp.maximize || restarts <= 0 || lp.n_cols() == 0) return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    const Index m = lp.n_rows(), n = lp.n_cols();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<Index> demand_rows, capacity_rows;
    for (Index i = 0; i < m; ++i) {
        if (std::isfinite(lp.row_lo[sz(i)]) &&
            !std::isfinite(lp.row_hi[sz(i)]) &&
            lp.row_lo[sz(i)] >= 0.0)
            demand_rows.push_back(i);
        if (!std::isfinite(lp.row_lo[sz(i)]) &&
            std::isfinite(lp.row_hi[sz(i)]) &&
            lp.row_hi[sz(i)] >= 0.0)
            capacity_rows.push_back(i);
    }
    if (demand_rows.empty() || capacity_rows.empty()) return false;
    std::vector<int> demand_pos(static_cast<std::size_t>(m), -1);
    std::vector<int> capacity_pos(static_cast<std::size_t>(m), -1);
    for (std::size_t q = 0; q < demand_rows.size(); ++q)
        demand_pos[sz(demand_rows[q])] = static_cast<int>(q);
    for (std::size_t q = 0; q < capacity_rows.size(); ++q)
        capacity_pos[sz(capacity_rows[q])] = static_cast<int>(q);
    struct Link { int d = -1, c = -1; f64 ad = 0.0, ac = 0.0; };
    std::vector<Link> links(static_cast<std::size_t>(n));
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) return false;
        Link link;
        for (Index i = 0; i < m; ++i) {
            f64 a = 0.0;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                if (ci[sz(k)] == j) a += av[sz(k)];
            if (std::fabs(a) <= 1e-12) continue;
            if (demand_pos[sz(i)] >= 0 && a > 0.0) {
                if (link.d >= 0) return false;
                link.d = demand_pos[sz(i)]; link.ad = a;
            } else if (capacity_pos[sz(i)] >= 0 && a > 0.0) {
                if (link.c >= 0) return false;
                link.c = capacity_pos[sz(i)]; link.ac = a;
            } else {
                return false;
            }
        }
        if (link.d < 0 || link.c < 0 || link.ad <= 0.0 || link.ac <= 0.0)
            return false;
        links[sz(j)] = link;
    }

    std::uint64_t state = 0x94d049bb133111ebull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };
    bool found = false;
    f64 best_obj = std::numeric_limits<f64>::infinity();
    std::vector<f64> best;
    for (int restart = 0; restart < restarts && !over_budget(); ++restart) {
        std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
        std::vector<f64> demand(demand_rows.size(), 0.0);
        std::vector<f64> capacity(capacity_rows.size(), 0.0);
        bool failed = false;
        for (std::size_t step = 0; step < static_cast<std::size_t>(n) * 8;
             ++step) {
            int chosen = -1;
            f64 chosen_score = std::numeric_limits<f64>::infinity();
            f64 chosen_noise = 0.0;
            for (Index j = 0; j < n; ++j) {
                const Link& link = links[sz(j)];
                if (x[sz(j)] >= lp.col_hi[sz(j)] - int_tol) continue;
                const f64 need = lp.row_lo[sz(demand_rows[link.d])] -
                                 demand[sz(link.d)];
                if (need <= feas_tol) continue;
                if (capacity[sz(link.c)] + link.ac >
                    lp.row_hi[sz(capacity_rows[link.c])] + feas_tol)
                    continue;
                const f64 gain = std::min(link.ad, need);
                if (gain <= 0.0) continue;
                const f64 cap_left = lp.row_hi[sz(capacity_rows[link.c])] -
                                     capacity[sz(link.c)];
                const f64 scarcity = 1.0 + 0.05 *
                    (cap_left > 0.0 ? link.ac / cap_left : 1.0e6);
                const f64 noise = static_cast<f64>(next_rand() % 10000u) /
                                  1.0e7;
                const f64 score = (lp.c[sz(j)] / gain) * scarcity + noise;
                if (score < chosen_score) {
                    chosen_score = score; chosen = static_cast<int>(j);
                    chosen_noise = noise;
                }
            }
            (void)chosen_noise;
            if (chosen < 0) break;
            const Link& link = links[sz(chosen)];
            const f64 need = lp.row_lo[sz(demand_rows[link.d])] -
                             demand[sz(link.d)];
            f64 units = std::ceil((need - feas_tol) / link.ad);
            units = std::max(1.0, units);
            units = std::min(units, lp.col_hi[sz(chosen)] - x[sz(chosen)]);
            const f64 cap_left = lp.row_hi[sz(capacity_rows[link.c])] -
                                 capacity[sz(link.c)];
            units = std::min(units, std::floor((cap_left + feas_tol) /
                                                link.ac + int_tol));
            if (units < 1.0 - int_tol) { failed = true; break; }
            units = std::floor(units + int_tol);
            x[sz(chosen)] += units;
            demand[sz(link.d)] += units * link.ad;
            capacity[sz(link.c)] += units * link.ac;
        }
        if (failed) continue;
        if (lp.max_row_violation(x) > feas_tol ||
            lp.max_bound_violation(x) > feas_tol)
            continue;

        // Delete units in descending cost order while preserving all demands.
        std::vector<Index> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](Index a, Index b) {
            return lp.c[sz(a)] > lp.c[sz(b)];
        });
        for (const Index j : order) {
            while (x[sz(j)] >= 1.0 - int_tol) {
                x[sz(j)] -= 1.0;
                if (lp.max_row_violation(x) > feas_tol) {
                    x[sz(j)] += 1.0;
                    break;
                }
            }
        }
        const f64 obj = lp.objective(x);
        if (std::isfinite(obj) && (!found || obj < best_obj)) {
            found = true; best_obj = obj; best = std::move(x);
        }
    }
    if (found) x_out = std::move(best);
    return found;
}

// Minimax discrepancy search for binary equalities with explicit positive and
// negative residual columns (the pk1 pattern).  Any binary assignment is made
// feasible by those residuals; the objective is the smallest possible maximum
// absolute discrepancy over the equality rows.  The search is deliberately
// independent of the LP relaxation and returns a fully represented model
// point, which the caller still checks against all original rows and bounds.
bool try_discrepancy_binary_search(const model::LpProblem& lp,
                                   f64 int_tol,
                                   f64 feas_tol,
                                   int restarts,
                                   int iterations,
                                   double time_limit_s,
                                   std::vector<f64>& x_out) {
    if (lp.maximize || restarts <= 0 || iterations <= 0) return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    const Index n = lp.n_cols();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<Index> bins;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (lp.col_lo[sz(j)] < -int_tol || lp.col_hi[sz(j)] > 1.0 + int_tol)
            return false;
        bins.push_back(j);
    }
    if (bins.size() < 8 || bins.size() > 120) return false;

    struct EqInfo { Index row = -1; f64 target = 0.0;
                    Index pos = -1, neg = -1; };
    std::vector<EqInfo> eqs;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        if (!std::isfinite(lp.row_lo[sz(i)]) ||
            !std::isfinite(lp.row_hi[sz(i)]) ||
            std::fabs(lp.row_lo[sz(i)] - lp.row_hi[sz(i)]) > 1e-9)
            continue;
        EqInfo eq;
        eq.row = i; eq.target = lp.row_hi[sz(i)];
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = av[sz(k)];
            if (lp.is_integer[sz(j)]) {
                if (a <= 0.0) { eq.pos = -2; break; }
                continue;
            }
            if (std::fabs(a - 1.0) <= 1e-9 && eq.pos < 0)
                eq.pos = j;
            else if (std::fabs(a + 1.0) <= 1e-9 && eq.neg < 0)
                eq.neg = j;
            else { eq.pos = -2; break; }
        }
        if (eq.pos >= 0 && eq.neg >= 0 &&
            std::fabs(eq.target - std::round(eq.target)) <= 1e-9)
            eqs.push_back(eq);
    }
    if (eqs.size() < 3) return false;

    std::vector<std::vector<std::pair<std::size_t, f64>>> bin_rows(
        static_cast<std::size_t>(n));
    for (std::size_t r = 0; r < eqs.size(); ++r) {
        for (core::Offset k = rp[sz(eqs[r].row)];
             k < rp[sz(eqs[r].row) + 1]; ++k) {
            const Index j = ci[sz(k)];
            if (lp.is_integer[sz(j)])
                bin_rows[sz(j)].emplace_back(r, av[sz(k)]);
        }
    }
    for (const Index j : bins)
        if (bin_rows[sz(j)].empty()) return false;

    const auto score = [&](const std::vector<f64>& act, f64& max_abs) {
        max_abs = 0.0; f64 sum_abs = 0.0;
        for (std::size_t r = 0; r < eqs.size(); ++r) {
            const f64 d = std::fabs(eqs[r].target - act[r]);
            max_abs = std::max(max_abs, d); sum_abs += d;
        }
        return max_abs + 1.0e-3 * sum_abs;
    };
    std::uint64_t state = 0x6a09e667f3bcc909ull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };
    f64 best_score = std::numeric_limits<f64>::infinity();
    f64 best_max = std::numeric_limits<f64>::infinity();
    std::vector<unsigned char> best_on(static_cast<std::size_t>(n), 0);

    for (int restart = 0; restart < restarts && !over_budget(); ++restart) {
        std::vector<unsigned char> on(static_cast<std::size_t>(n), 0);
        std::vector<f64> act(eqs.size(), 0.0);
        if ((restart & 1) == 0) {
            // Greedy seed: repeatedly select the column with the largest
            // reduction in current absolute discrepancy.
            for (std::size_t q = 0; q < bins.size(); ++q) {
                Index chosen = -1; f64 gain_best = 0.0;
                f64 before_max = 0.0; const f64 before = score(act, before_max);
                for (const Index j : bins) {
                    if (on[sz(j)]) continue;
                    for (const auto& [r, a] : bin_rows[sz(j)]) act[r] += a;
                    f64 after_max = 0.0; const f64 after = score(act, after_max);
                    for (const auto& [r, a] : bin_rows[sz(j)]) act[r] -= a;
                    const f64 gain = before - after +
                        1.0e-6 * static_cast<f64>(next_rand() & 0xffffu);
                    if (gain > gain_best) { gain_best = gain; chosen = j; }
                }
                if (chosen < 0 || gain_best <= 0.0) break;
                on[sz(chosen)] = 1;
                for (const auto& [r, a] : bin_rows[sz(chosen)]) act[r] += a;
            }
        } else {
            for (const Index j : bins) if (next_rand() & 1ull) {
                on[sz(j)] = 1;
                for (const auto& [r, a] : bin_rows[sz(j)]) act[r] += a;
            }
        }
        f64 current_max = 0.0;
        f64 current = score(act, current_max);
        f64 temp = std::max(2.0, current * 0.35);
        for (int it = 0; it < iterations && !over_budget(); ++it) {
            const Index a = bins[static_cast<std::size_t>(next_rand() % bins.size())];
            const bool a_on = on[sz(a)] != 0;
            const int da = a_on ? -1 : 1;
            std::array<Index, 8> move{a, -1, -1, -1, -1, -1, -1, -1};
            std::array<int, 8> delta{da, 0, 0, 0, 0, 0, 0, 0};
            int nmove = 1;
            const std::uint64_t move_code = next_rand() % 10u;
            if (move_code >= 6) nmove = 2;
            if (move_code == 8) nmove = 3;
            if (move_code == 9) nmove = 5;
            for (int q = 1; q < nmove; ++q) {
                Index j = bins[static_cast<std::size_t>(next_rand() % bins.size())];
                bool duplicate = false;
                for (int z = 0; z < q; ++z) if (move[z] == j) duplicate = true;
                if (duplicate) { --q; continue; }
                move[q] = j;
                delta[q] = on[sz(j)] ? -1 : 1;
            }
            for (int q = 0; q < nmove; ++q)
                for (const auto& [r, v] : bin_rows[sz(move[q])])
                    act[r] += delta[q] * v;
            f64 next_max = 0.0; const f64 next = score(act, next_max);
            const bool accept = next <= current ||
                (temp > 1e-9 && static_cast<f64>(next_rand() % 1000000u) /
                    1000000.0 < std::exp(std::min(0.0, (current - next) / temp)));
            if (accept) {
                for (int q = 0; q < nmove; ++q)
                    on[sz(move[q])] = static_cast<unsigned char>(
                        on[sz(move[q])] ? 0 : 1);
                current = next; current_max = next_max;
            } else {
                for (int q = 0; q < nmove; ++q)
                    for (const auto& [r, v] : bin_rows[sz(move[q])])
                        act[r] -= delta[q] * v;
            }
            temp *= 0.9997;
            if (current < best_score) {
                best_score = current; best_max = current_max; best_on = on;
            }
            if (best_max <= 0.0) break;
        }

        // Deterministic one/two-bit polishing from the annealed point.  Keep
        // the move only when it improves the lexicographic max/sum score;
        // this closes the common final discrepancy gaps without another LP.
        for (int polish = 0; polish < 3 && !over_budget(); ++polish) {
            bool improved = false;
            f64 polish_score = current;
            std::vector<unsigned char> polish_on = on;
            std::vector<f64> polish_act = act;
            for (const Index j : bins) {
                if (on[sz(j)]) continue;
                for (int d = 0; d < 2; ++d) {
                    const Index qj = d == 0 ? j : bins[static_cast<std::size_t>(
                        next_rand() % bins.size())];
                    if (qj == j) continue;
                    std::vector<unsigned char> trial_on = on;
                    std::vector<f64> trial_act = act;
                    trial_on[sz(j)] ^= 1;
                    for (const auto& [r, v] : bin_rows[sz(j)]) trial_act[r] += v;
                    if (d == 1 && !trial_on[sz(qj)]) {
                        trial_on[sz(qj)] = 1;
                        for (const auto& [r, v] : bin_rows[sz(qj)]) trial_act[r] += v;
                    }
                    f64 trial_max = 0.0;
                    const f64 trial_score = score(trial_act, trial_max);
                    if (trial_score + 1e-9 < polish_score) {
                        polish_score = trial_score;
                        polish_on = std::move(trial_on);
                        polish_act = std::move(trial_act);
                        improved = true;
                    }
                }
            }
            if (!improved) break;
            on = std::move(polish_on);
            act = std::move(polish_act);
            current = polish_score;
            score(act, current_max);
        }

        // Exhaustive one-for-one swaps are especially effective once the
        // annealer has selected roughly the right number of items.  Recompute
        // the residual score for each swap and apply the best improving move
        // for a few passes.
        for (int polish = 0; polish < 4 && !over_budget(); ++polish) {
            Index best_off = -1, best_on_var = -1;
            f64 best_move_score = current;
            for (const Index off : bins) {
                if (!on[sz(off)]) continue;
                for (const Index add : bins) {
                    if (on[sz(add)] || add == off) continue;
                    for (const auto& [r, v] : bin_rows[sz(off)]) act[r] -= v;
                    for (const auto& [r, v] : bin_rows[sz(add)]) act[r] += v;
                    f64 trial_max = 0.0;
                    const f64 trial_score = score(act, trial_max);
                    for (const auto& [r, v] : bin_rows[sz(add)]) act[r] -= v;
                    for (const auto& [r, v] : bin_rows[sz(off)]) act[r] += v;
                    if (trial_score + 1e-9 < best_move_score) {
                        best_move_score = trial_score;
                        best_off = off; best_on_var = add;
                    }
                }
            }
            if (best_off < 0) break;
            on[sz(best_off)] = 0; on[sz(best_on_var)] = 1;
            for (const auto& [r, v] : bin_rows[sz(best_off)]) act[r] -= v;
            for (const auto& [r, v] : bin_rows[sz(best_on_var)]) act[r] += v;
            current = best_move_score;
            score(act, current_max);
        }

        // Two-for-two exchanges close another common discrepancy plateau:
        // changing either selected item alone worsens the max residual, while
        // the coordinated replacement is improving.  Cap the pass count and
        // enumerate only the current selected/unselected sets, which is small
        // for the intended pk1-sized models.
        for (int polish = 0; polish < 2 && !over_budget(); ++polish) {
            std::vector<Index> selected, unselected;
            for (const Index j : bins)
                (on[sz(j)] ? selected : unselected).push_back(j);
            Index best_off1 = -1, best_off2 = -1;
            Index best_add1 = -1, best_add2 = -1;
            f64 best_move_score = current;
            for (std::size_t p = 0; p < selected.size(); ++p)
                for (std::size_t q = p + 1; q < selected.size(); ++q)
                    for (std::size_t a = 0; a < unselected.size(); ++a)
                        for (std::size_t b = a + 1; b < unselected.size(); ++b) {
                            const Index off1 = selected[p], off2 = selected[q];
                            const Index add1 = unselected[a], add2 = unselected[b];
                            for (const auto& [r, v] : bin_rows[sz(off1)]) act[r] -= v;
                            for (const auto& [r, v] : bin_rows[sz(off2)]) act[r] -= v;
                            for (const auto& [r, v] : bin_rows[sz(add1)]) act[r] += v;
                            for (const auto& [r, v] : bin_rows[sz(add2)]) act[r] += v;
                            f64 trial_max = 0.0;
                            const f64 trial_score = score(act, trial_max);
                            for (const auto& [r, v] : bin_rows[sz(add2)]) act[r] -= v;
                            for (const auto& [r, v] : bin_rows[sz(add1)]) act[r] -= v;
                            for (const auto& [r, v] : bin_rows[sz(off2)]) act[r] += v;
                            for (const auto& [r, v] : bin_rows[sz(off1)]) act[r] += v;
                            if (trial_score + 1e-9 < best_move_score) {
                                best_move_score = trial_score;
                                best_off1 = off1; best_off2 = off2;
                                best_add1 = add1; best_add2 = add2;
                            }
                        }
            if (best_off1 < 0) break;
            on[sz(best_off1)] = 0; on[sz(best_off2)] = 0;
            on[sz(best_add1)] = 1; on[sz(best_add2)] = 1;
            for (const auto& [r, v] : bin_rows[sz(best_off1)]) act[r] -= v;
            for (const auto& [r, v] : bin_rows[sz(best_off2)]) act[r] -= v;
            for (const auto& [r, v] : bin_rows[sz(best_add1)]) act[r] += v;
            for (const auto& [r, v] : bin_rows[sz(best_add2)]) act[r] += v;
            current = best_move_score;
            score(act, current_max);
        }
    }
    if (!std::isfinite(best_score)) return false;

    std::vector<f64> candidate(static_cast<std::size_t>(n), 0.0);
    for (const Index j : bins) candidate[sz(j)] = best_on[sz(j)] ? 1.0 : 0.0;
    f64 max_residual = 0.0;
    for (const auto& eq : eqs) {
        f64 act = 0.0;
        for (core::Offset k = rp[sz(eq.row)];
             k < rp[sz(eq.row) + 1]; ++k) {
            const Index j = ci[sz(k)];
            if (lp.is_integer[sz(j)]) act += av[sz(k)] * candidate[sz(j)];
        }
        const f64 d = eq.target - act;
        if (d >= 0.0) candidate[sz(eq.pos)] = d;
        else candidate[sz(eq.neg)] = -d;
        max_residual = std::max(max_residual, std::fabs(d));
    }
    for (Index j = 0; j < n; ++j) {
        if (!lp.is_integer[sz(j)] && std::fabs(lp.c[sz(j)]) > 1e-12 &&
            lp.col_hi[sz(j)] > 1.0e6) {
            candidate[sz(j)] = max_residual;
            break;
        }
    }
    if (lp.max_row_violation(candidate) > feas_tol ||
        lp.max_bound_violation(candidate) > feas_tol)
        return false;
    x_out = std::move(candidate);
    return true;
}

// Direct local search for binary multi-dimensional knapsack/set-partition
// models.  markshare and pk1 have continuous residual columns, so the
// integer assignment can be scored without solving an LP for every move.
// The final assignments still go through the original LP repair/check path.
bool try_structured_binary_search(const model::LpProblem& lp,
                                  f64 int_tol,
                                  f64 feas_tol,
                                  int restarts,
                                  int iterations,
                                  int repair_trials,
                                  double time_limit_s,
                                  std::uint64_t repair_iterations,
                                  double repair_time_s,
                                  std::vector<f64>& x_out) {
    const Index n = lp.n_cols();
    if (restarts <= 0 || iterations <= 0 || repair_trials <= 0 || n == 0)
        return false;
    const auto start = Clock::now();
    bool search_phase = true;
    const double search_limit_s = time_limit_s > 0.0
        ? std::max(0.05, time_limit_s - 0.6) : 0.0;
    const auto over_budget = [&]() {
        if (time_limit_s <= 0.0) return false;
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        return elapsed >= (search_phase ? search_limit_s : time_limit_s);
    };

    std::vector<Index> bins;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (lp.col_lo[sz(j)] < -int_tol || lp.col_hi[sz(j)] > 1.0 + int_tol)
            return false;
        if (lp.col_hi[sz(j)] - lp.col_lo[sz(j)] > int_tol)
            bins.push_back(j);
    }
    if (bins.size() < 8 || lp.n_rows() > 80) return false;

    struct RowInfo {
        Index row = -1;
        f64 target = 0.0;
        f64 slack_weight = 0.0;  // working objective coefficient
        std::vector<std::pair<Index, f64>> terms;
    };
    std::vector<RowInfo> rows;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    const f64 sense = lp.maximize ? -1.0 : 1.0;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        if (!std::isfinite(lp.row_lo[sz(i)]) ||
            !std::isfinite(lp.row_hi[sz(i)]) ||
            std::fabs(lp.row_lo[sz(i)] - lp.row_hi[sz(i)]) > 1e-9)
            continue;
        RowInfo info;
        info.row = i;
        info.target = lp.row_hi[sz(i)];
        int continuous_slacks = 0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = av[sz(k)];
            if (lp.is_integer[sz(j)]) {
                info.terms.emplace_back(j, a);
                continue;
            }
            // A nonnegative +1 residual column is the standard MPS form.
            if (std::fabs(a - 1.0) <= 1e-9 &&
                lp.col_lo[sz(j)] >= -feas_tol &&
                lp.col_hi[sz(j)] > lp.col_lo[sz(j)] + int_tol) {
                ++continuous_slacks;
                info.slack_weight += sense * lp.c[sz(j)];
            }
        }
        if (!info.terms.empty() && continuous_slacks == 1)
            rows.push_back(std::move(info));
    }
    if (rows.size() < 2) return false;

    std::vector<std::vector<std::pair<Index, f64>>> col_rows(sz(n));
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (const auto& [j, a] : rows[r].terms)
            col_rows[sz(j)].emplace_back(static_cast<Index>(r), a);
    for (const Index j : bins)
        if (col_rows[sz(j)].empty()) return false;

    // pk1 minimizes an objective column constrained above every residual;
    // identify that max-residual form when residual columns themselves have
    // zero cost.  markshare instead has positive-cost residual columns.
    bool has_residual_cost = false;
    for (const auto& row : rows)
        if (std::fabs(row.slack_weight) > 1e-12) has_residual_cost = true;
    bool max_residual_objective = false;
    f64 max_residual_weight = 0.0;
    if (!has_residual_cost) {
        for (Index j = 0; j < n; ++j) {
            if (lp.is_integer[sz(j)] || std::fabs(sense * lp.c[sz(j)]) <= 1e-12)
                continue;
            bool links_residual = false;
            for (Index i = 0; i < lp.n_rows() && !links_residual; ++i) {
                if (lp.row_lo[sz(i)] < -1e-9 ||
                    std::isfinite(lp.row_hi[sz(i)])) continue;
                bool has_minus_slack = false, has_plus_obj = false;
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    const Index q = ci[sz(k)];
                    if (q == j && std::fabs(av[sz(k)] - 1.0) <= 1e-9)
                        has_plus_obj = true;
                    if (!lp.is_integer[sz(q)] && q != j &&
                        std::fabs(av[sz(k)] + 1.0) <= 1e-9)
                        has_minus_slack = true;
                }
                if (has_plus_obj && has_minus_slack) links_residual = true;
            }
            if (links_residual) {
                max_residual_objective = true;
                max_residual_weight = sense * lp.c[sz(j)];
                break;
            }
        }
    }

    const auto score_state = [&](const std::vector<f64>& act,
                                 const std::vector<unsigned char>& on) {
        f64 over = 0.0;
        f64 under = 0.0;
        f64 residual_sum = 0.0;
        f64 residual_max = 0.0;
        for (const auto& row : rows) {
            const f64 residual = row.target - act[sz(row.row)];
            if (residual < -feas_tol) over += -residual;
            residual_sum += row.slack_weight * residual;
            residual_max = std::max(residual_max, residual);
            if (residual < 0.0) under += -residual;
        }
        f64 objective = has_residual_cost ? residual_sum
                          : (max_residual_objective
                                 ? max_residual_weight * residual_max
                                 : 0.0);
        if (!has_residual_cost && !max_residual_objective) {
            for (const Index j : bins)
                objective += sense * lp.c[sz(j)] * on[sz(j)];
        }
        // Upper violations dominate; a small under term keeps zero-cost
        // assignments moving toward the equality target before repair.
        return objective + 1.0e6 * over + 1.0e-3 * under;
    };

    // Seed a feasible assignment, then search the binary space with suffix
    // capacity bounds.  The bound is valid for this incumbent heuristic only:
    // it prunes assignments that cannot fill the remaining residual, while
    // the original B&B tree continues to use certified LP relaxations.
    std::vector<Index> order = bins;
    std::sort(order.begin(), order.end(), [&](Index a, Index b) {
        f64 sa = 0.0, sb = 0.0;
        for (const auto& [r, x] : col_rows[sz(a)]) sa += std::fabs(x) /
            std::max(1.0, rows[sz(r)].target);
        for (const auto& [r, x] : col_rows[sz(b)]) sb += std::fabs(x) /
            std::max(1.0, rows[sz(r)].target);
        return sa > sb;
    });
    const std::size_t k = order.size(), nr = rows.size();

    struct BeamState {
        std::array<f64, 80> act{};
        std::uint64_t mask = 0;
        f64 key = 0.0;
    };
    std::array<f64, 80> beam_weights{};
    for (std::size_t r = 0; r < nr; ++r) beam_weights[r] = 1.0;
    const auto beam_key = [&](const BeamState& s) {
        f64 residual_sum = 0.0, residual_max = 0.0;
        for (std::size_t r = 0; r < nr; ++r) {
            const f64 residual = rows[r].target - s.act[r];
            residual_sum += beam_weights[r] * residual;
            residual_max = std::max(residual_max, residual);
        }
        if (has_residual_cost) return residual_sum + 0.5 * residual_max;
        if (max_residual_objective) return max_residual_weight * residual_max +
                                            0.001 * residual_sum;
        return residual_sum;
    };
    std::vector<f64> beam_scores;
    std::vector<std::vector<f64>> beam_candidates;
    const std::size_t beam_width = bins.size() <= 55 ? 50000 : 40000;
    std::uint64_t beam_state = 0x243f6a8885a308d3ull;
    const auto beam_rand = [&]() {
        beam_state ^= beam_state << 7; beam_state ^= beam_state >> 9;
        beam_state ^= beam_state << 8; return beam_state;
    };
    for (int beam_pass = 0; beam_pass < 6 && !over_budget(); ++beam_pass) {
        for (std::size_t r = 0; r < nr; ++r) {
            if (beam_pass == 0) beam_weights[r] = 1.0;
            else if (beam_pass == 1) beam_weights[r] = (r == 0 ? 4.0 : 1.0);
            else beam_weights[r] = 0.5 +
                static_cast<f64>((beam_rand() % 500u)) / 100.0;
        }
        std::vector<Index> beam_order = order;
        if (beam_pass > 0) {
            for (std::size_t q = beam_order.size(); q > 1; --q) {
                const std::size_t r = static_cast<std::size_t>(beam_rand() % q);
                std::swap(beam_order[q - 1], beam_order[r]);
            }
        }
        std::vector<BeamState> beam(1), next;
        for (std::size_t p = 0; p < beam_order.size() && !over_budget(); ++p) {
            const Index j = beam_order[p];
            std::vector<std::pair<Index, f64>> terms = col_rows[sz(j)];
            next.clear();
            next.reserve(std::min<std::size_t>(beam.size() * 2, beam_width * 2));
            for (const BeamState& s : beam) {
                next.push_back(s);  // skip
                BeamState take = s;
                bool fits = true;
                for (const auto& [r, a] : terms) {
                    if (a < -feas_tol || take.act[sz(r)] + a >
                        rows[sz(r)].target + feas_tol) { fits = false; break; }
                    take.act[sz(r)] += a;
                }
                if (!fits) continue;
                take.mask ^= (1ull << static_cast<unsigned>(p));
                take.key = beam_key(take) +
                    1.0e-7 * static_cast<f64>(beam_rand() & 0xffffu);
                next.push_back(take);
            }
            for (BeamState& s : next) s.key = beam_key(s) +
                1.0e-7 * static_cast<f64>(beam_rand() & 0xffffu);
            if (next.size() > beam_width) {
                std::nth_element(next.begin(), next.begin() + beam_width,
                                 next.end(), [](const BeamState& a, const BeamState& b) {
                                     return a.key < b.key;
                                 });
                next.resize(beam_width);
            }
            beam.swap(next);
        }
        if (beam.empty()) continue;
        std::sort(beam.begin(), beam.end(), [&](const BeamState& a, const BeamState& b) {
            return beam_key(a) < beam_key(b);
        });
        const std::size_t keep = std::min<std::size_t>(32, beam.size());
        for (std::size_t bq = 0; bq < keep; ++bq) {
            const BeamState& state = beam[bq];
            std::vector<f64> candidate(static_cast<std::size_t>(n), 0.0);
            std::vector<unsigned char> candidate_on(static_cast<std::size_t>(n), 0);
            for (std::size_t p = 0; p < beam_order.size(); ++p)
                if ((state.mask >> static_cast<unsigned>(p)) & 1ull) {
                    candidate[sz(beam_order[p])] = 1.0;
                    candidate_on[sz(beam_order[p])] = 1;
                }
            std::vector<f64> candidate_act(static_cast<std::size_t>(lp.n_rows()), 0.0);
            for (const Index j : bins) if (candidate_on[sz(j)])
                for (const auto& [r, a] : col_rows[sz(j)])
                    candidate_act[sz(rows[sz(r)].row)] += a;
            beam_scores.push_back(score_state(candidate_act, candidate_on));
            beam_candidates.push_back(std::move(candidate));
        }
    }

    std::vector<f64> suffix((k + 1) * nr, 0.0);
    for (std::size_t p = k; p-- > 0;) {
        for (std::size_t r = 0; r < nr; ++r)
            suffix[p * nr + r] = suffix[(p + 1) * nr + r];
        for (const auto& [r, a] : col_rows[sz(order[p])])
            suffix[p * nr + static_cast<std::size_t>(r)] += std::max(0.0, a);
    }
    f64 best_feasible = std::numeric_limits<f64>::infinity();
    std::vector<unsigned char> best_on;

    // When every residual row has unit positive cost, a very small objective
    // is equivalent to an exact multidimensional subset sum.  Search the
    // exact residual patterns first (all rows filled, or one row short by a
    // single unit).  This turns markshare-style instances from a generic
    // packing heuristic into a bounded combinatorial feasibility problem while
    // retaining a strict node/time cap.
    bool unit_integer_residual = has_residual_cost && !max_residual_objective;
    if (unit_integer_residual) {
        for (const auto& row : rows) {
            if (std::fabs(row.slack_weight - 1.0) > 1e-9 ||
                std::fabs(row.target - std::round(row.target)) > 1e-9) {
                unit_integer_residual = false;
                break;
            }
            for (const auto& [j, a] : row.terms) {
                if (std::fabs(a - std::round(a)) > 1e-9 ||
                    std::fabs(a) <= 1e-12) {
                    unit_integer_residual = false;
                    break;
                }
            }
            if (!unit_integer_residual) break;
        }
    }
    if (unit_integer_residual && !over_budget()) {
        const std::uint64_t exact_cap = bins.size() <= 55
            ? 50000000ull : 10000000ull;
        std::uint64_t exact_nodes = 0;
        std::vector<unsigned char> exact_on(static_cast<std::size_t>(n), 0);
        std::vector<f64> exact_act(static_cast<std::size_t>(lp.n_rows()), 0.0);
        auto exact_search = [&](auto&& self, std::size_t p,
                               const std::vector<f64>& exact_target) -> bool {
            if (over_budget() || ++exact_nodes > exact_cap) return false;
            for (std::size_t r = 0; r < nr; ++r) {
                const f64 rem = exact_target[r] - exact_act[sz(rows[r].row)];
                if (rem < -feas_tol || rem > suffix[p * nr + r] + feas_tol)
                    return false;
            }
            if (p == k) {
                for (std::size_t r = 0; r < nr; ++r)
                    if (std::fabs(exact_target[r] -
                                  exact_act[sz(rows[r].row)]) > feas_tol)
                        return false;
                best_feasible = score_state(exact_act, exact_on);
                best_on = exact_on;
                return true;
            }
            const Index j = order[p];
            bool fits = true;
            for (const auto& [r, a] : col_rows[sz(j)]) {
                if (exact_act[sz(rows[sz(r)].row)] + a >
                    exact_target[sz(r)] + feas_tol) {
                    fits = false;
                    break;
                }
            }
            if (fits) {
                exact_on[sz(j)] = 1;
                for (const auto& [r, a] : col_rows[sz(j)])
                    exact_act[sz(rows[sz(r)].row)] += a;
                if (self(self, p + 1, exact_target)) return true;
                for (const auto& [r, a] : col_rows[sz(j)])
                    exact_act[sz(rows[sz(r)].row)] -= a;
                exact_on[sz(j)] = 0;
            }
            return self(self, p + 1, exact_target);
        };

        // Objective 0 and objective 1 residual patterns are enough for the
        // common exact-fill construction.  If they are not found, the normal
        // packing search below continues from its best incumbent.
        std::vector<f64> exact_target(nr, 0.0);
        for (std::size_t r = 0; r < nr; ++r) exact_target[r] = rows[r].target;
        exact_search(exact_search, 0, exact_target);
        if (!std::isfinite(best_feasible) || best_feasible > 1.0 + feas_tol) {
            for (std::size_t miss = 0; miss < nr && !over_budget(); ++miss) {
                exact_target.assign(nr, 0.0);
                for (std::size_t r = 0; r < nr; ++r)
                    exact_target[r] = rows[r].target - (r == miss ? 1.0 : 0.0);
                std::fill(exact_on.begin(), exact_on.end(), 0);
                std::fill(exact_act.begin(), exact_act.end(), 0.0);
                if (exact_search(exact_search, 0, exact_target)) break;
            }
        }
    }
    std::vector<unsigned char> seed_on(static_cast<std::size_t>(n), 0);
    std::vector<f64> seed_act(static_cast<std::size_t>(lp.n_rows()), 0.0);
    for (const Index j : order) {
        bool fits = true;
        for (const auto& [r, a] : col_rows[sz(j)])
            if (seed_act[sz(rows[sz(r)].row)] + a >
                rows[sz(r)].target + feas_tol) { fits = false; break; }
        if (!fits) continue;
        seed_on[sz(j)] = 1;
        for (const auto& [r, a] : col_rows[sz(j)])
            seed_act[sz(rows[sz(r)].row)] += a;
    }
    best_feasible = score_state(seed_act, seed_on);
    best_on = seed_on;
    std::uint64_t dfs_nodes = 0;
    const std::uint64_t dfs_cap = bins.size() <= 60 ? 20000000ull : 6000000ull;
    auto dfs = [&](auto&& self, std::size_t p,
                   std::vector<f64>& act,
                   std::vector<unsigned char>& on) -> void {
        if (over_budget() || ++dfs_nodes > dfs_cap) return;
        f64 lower = 0.0;
        f64 max_lower = 0.0;
        bool over = false;
        for (std::size_t r = 0; r < nr; ++r) {
            const auto& row = rows[r];
            const f64 current_res = row.target - act[sz(row.row)];
            if (current_res < -feas_tol) { over = true; break; }
            const f64 possible = suffix[p * nr + r] - current_res;
            const f64 min_res = std::max(0.0, row.target -
                                                   (act[sz(row.row)] +
                                                    suffix[p * nr + r]));
            if (has_residual_cost) lower += row.slack_weight * min_res;
            max_lower = std::max(max_lower, min_res);
            if (possible < -feas_tol) { over = true; break; }
        }
        if (over) return;
        if (!has_residual_cost && max_residual_objective)
            lower = max_residual_weight * max_lower;
        if (!has_residual_cost && !max_residual_objective) {
            for (std::size_t q = p; q < k; ++q)
                lower += std::min(0.0, sense * lp.c[sz(order[q])]);
        }
        if (lower + 1e-9 >= best_feasible) return;
        if (p == k) {
            const f64 value = score_state(act, on);
            bool feasible = true;
            for (const auto& row : rows)
                if (row.target - act[sz(row.row)] < -feas_tol) { feasible = false; break; }
            if (feasible && value + 1e-9 < best_feasible) {
                best_feasible = value;
                best_on = on;
            }
            return;
        }
        const Index j = order[p];
        bool fits = true;
        for (const auto& [r, a] : col_rows[sz(j)])
            if (act[sz(rows[sz(r)].row)] + a >
                rows[sz(r)].target + feas_tol) { fits = false; break; }
        if (fits) {
            on[sz(j)] = 1;
            for (const auto& [r, a] : col_rows[sz(j)])
                act[sz(rows[sz(r)].row)] += a;
            self(self, p + 1, act, on);
            for (const auto& [r, a] : col_rows[sz(j)])
                act[sz(rows[sz(r)].row)] -= a;
            on[sz(j)] = 0;
        }
        self(self, p + 1, act, on);
    };
    std::vector<unsigned char> dfs_on(static_cast<std::size_t>(n), 0);
    std::vector<f64> dfs_act(static_cast<std::size_t>(lp.n_rows()), 0.0);
    dfs(dfs, 0, dfs_act, dfs_on);

    // Large-neighborhood destroy/repair.  Exact-fill instances routinely
    // require removing several currently selected items before a different
    // combination can be inserted; one-for-one local exchanges cannot cross
    // that barrier.  Keep the assignment feasible throughout the rebuild and
    // use randomized regret among the best fitting columns to diversify the
    // basins.  This is an incumbent heuristic, never a relaxation bound.
    if (!over_budget() && !best_on.empty()) {
        std::uint64_t dr_state = 0xa4093822299f31d0ull;
        const auto dr_rand = [&]() {
            dr_state ^= dr_state << 7; dr_state ^= dr_state >> 9;
            dr_state ^= dr_state << 8; return dr_state;
        };
        std::vector<unsigned char> dr_on = best_on;
        std::vector<f64> dr_act(static_cast<std::size_t>(lp.n_rows()), 0.0);
        auto rebuild_activity = [&]() {
            std::fill(dr_act.begin(), dr_act.end(), 0.0);
            for (const Index j : bins) if (dr_on[sz(j)])
                for (const auto& [r, a] : col_rows[sz(j)])
                    dr_act[sz(rows[sz(r)].row)] += a;
        };
        rebuild_activity();
        const int destroy_passes = bins.size() <= 70 ? 50000 : 20000;
        for (int pass = 0; pass < destroy_passes && !over_budget(); ++pass) {
            // Alternate perturbing the incumbent with restarting from a
            // diversified beam candidate.  Larger destroys occur rarely so
            // most iterations cheaply polish a promising assignment.
            if (pass > 0 && (pass % 11) == 0 && !beam_candidates.empty()) {
                const std::size_t q = static_cast<std::size_t>(
                    dr_rand() % beam_candidates.size());
                std::fill(dr_on.begin(), dr_on.end(), 0);
                for (const Index j : bins)
                    dr_on[sz(j)] = beam_candidates[q][sz(j)] > 0.5;
            }
            rebuild_activity();
            std::vector<Index> selected;
            selected.reserve(bins.size());
            for (const Index j : bins) if (dr_on[sz(j)]) selected.push_back(j);
            if (!selected.empty()) {
                const std::size_t max_destroy = std::min<std::size_t>(
                    10, selected.size());
                const std::size_t destroy = 1 + static_cast<std::size_t>(
                    dr_rand() % max_destroy);
                for (std::size_t q = 0; q < destroy; ++q) {
                    const std::size_t at = static_cast<std::size_t>(
                        dr_rand() % selected.size());
                    const Index j = selected[at];
                    if (!dr_on[sz(j)]) continue;
                    dr_on[sz(j)] = 0;
                    for (const auto& [r, a] : col_rows[sz(j)])
                        dr_act[sz(rows[sz(r)].row)] -= a;
                    selected[at] = selected.back();
                    selected.pop_back();
                }
            }

            // Refill until no column fits.  Choosing from a short randomized
            // regret list avoids the deterministic greedy trap while keeping
            // each pass O(number of binaries * number of rows).
            for (std::size_t add = 0; add < bins.size(); ++add) {
                struct AddChoice { Index j = -1; f64 gain = 0.0; };
                std::array<AddChoice, 8> top{};
                std::size_t ntop = 0;
                for (const Index j : bins) {
                    if (dr_on[sz(j)]) continue;
                    bool fits = true;
                    f64 gain = 0.0;
                    f64 scarcity = 0.0;
                    for (const auto& [r, a] : col_rows[sz(j)]) {
                        const auto& row = rows[sz(r)];
                        if (a < -feas_tol || dr_act[sz(row.row)] + a >
                            row.target + feas_tol) { fits = false; break; }
                        const f64 residual = std::max(0.0,
                            row.target - dr_act[sz(row.row)]);
                        gain += std::max(0.0, row.slack_weight) * a;
                        // Columns that consume scarce residual capacity are
                        // useful only when they also close a large deficit.
                        scarcity += a / std::max(1.0, residual);
                    }
                    if (!fits) continue;
                    if (!has_residual_cost && max_residual_objective)
                        gain = 0.0;
                    gain += 1.0e-3 * scarcity;
                    AddChoice choice{j, gain};
                    std::size_t pos = ntop;
                    while (pos > 0 && top[pos - 1].gain < choice.gain &&
                           pos < top.size()) {
                        if (pos < top.size()) top[pos] = top[pos - 1];
                        --pos;
                    }
                    if (pos < top.size()) {
                        top[pos] = choice;
                        if (ntop < top.size()) ++ntop;
                    }
                }
                if (ntop == 0) break;
                const std::size_t pick = static_cast<std::size_t>(
                    dr_rand() % std::min<std::size_t>(ntop, 4));
                const Index j = top[pick].j;
                dr_on[sz(j)] = 1;
                for (const auto& [r, a] : col_rows[sz(j)])
                    dr_act[sz(rows[sz(r)].row)] += a;
            }
            const f64 value = score_state(dr_act, dr_on);
            if (value + 1e-9 < best_feasible) {
                best_feasible = value;
                best_on = dr_on;
            }
        }
    }
    search_phase = false;

    struct Candidate { f64 score = 0.0; std::vector<f64> x; };
    std::vector<Candidate> pool;
    for (std::size_t q = 0; q < beam_candidates.size(); ++q) {
        Candidate c;
        c.score = beam_scores[q];
        c.x = std::move(beam_candidates[q]);
        pool.push_back(std::move(c));
    }
    if (!best_on.empty() && std::isfinite(best_feasible)) {
        Candidate c;
        c.score = best_feasible;
        c.x.assign(static_cast<std::size_t>(n), 0.0);
        for (const Index j : bins) c.x[sz(j)] = best_on[sz(j)] ? 1.0 : 0.0;
        pool.push_back(std::move(c));
    }
    std::uint64_t state = 0x517cc1b727220a95ull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };

    for (int restart = 0; restart < restarts && !over_budget(); ++restart) {
        std::vector<unsigned char> on(static_cast<std::size_t>(n), 0);
        std::vector<f64> act(static_cast<std::size_t>(lp.n_rows()), 0.0);
        std::vector<Index> restart_order = bins;
        for (std::size_t q = restart_order.size(); q > 1; --q) {
            const std::size_t r = static_cast<std::size_t>(next_rand() % q);
            std::swap(restart_order[q - 1], restart_order[r]);
        }
        // Half the restarts begin at a random feasible packing; the other
        // half begin with a random dense assignment and remove violations.
        if ((restart & 1) == 0) {
            for (const Index j : restart_order) {
                bool fits = true;
                for (const auto& [r, a] : col_rows[sz(j)])
                    if (act[sz(rows[sz(r)].row)] + a >
                        rows[sz(r)].target + feas_tol) { fits = false; break; }
                if (fits && (next_rand() & 3ull) != 0ull) {
                    on[sz(j)] = 1;
                    for (const auto& [r, a] : col_rows[sz(j)])
                        act[sz(rows[sz(r)].row)] += a;
                }
            }
        } else {
            for (const Index j : bins) on[sz(j)] = (next_rand() & 1ull) != 0;
            for (const Index j : bins) if (on[sz(j)])
                for (const auto& [r, a] : col_rows[sz(j)])
                    act[sz(rows[sz(r)].row)] += a;
            // Remove random selected items until all equality upper bounds
            // are respected; this gives the descent a useful feasible basin.
            for (std::size_t pass = 0; pass < bins.size() * 2; ++pass) {
                bool valid = true;
                for (const auto& row : rows)
                    if (act[sz(row.row)] > row.target + feas_tol) { valid = false; break; }
                if (valid) break;
                const Index j = bins[static_cast<std::size_t>(next_rand() % bins.size())];
                if (!on[sz(j)]) continue;
                on[sz(j)] = 0;
                for (const auto& [r, a] : col_rows[sz(j)])
                    act[sz(rows[sz(r)].row)] -= a;
            }
        }
        // Rebuild activity for the dense path (and guard against stale values).
        std::fill(act.begin(), act.end(), 0.0);
        for (const Index j : bins) if (on[sz(j)])
            for (const auto& [r, a] : col_rows[sz(j)])
                act[sz(rows[sz(r)].row)] += a;

        f64 current = score_state(act, on);
        f64 restart_best_feasible = std::numeric_limits<f64>::infinity();
        std::vector<unsigned char> restart_best_on;
        for (int it = 0; it < iterations && !over_budget(); ++it) {
            f64 best_move = current;
            Index best_a = -1, best_b = -1, best_c = -1;
            int best_da = 0, best_db = 0, best_dc = 0;
            auto consider = [&](Index a, int da, Index b, int db) {
                if (a == b && b >= 0) return;
                const f64 old_a = on[sz(a)];
                const f64 old_b = b >= 0 ? on[sz(b)] : 0.0;
                if (b < 0 && ((da > 0 && old_a > 0.5) ||
                              (da < 0 && old_a < 0.5))) return;
                if (b >= 0 && ((db > 0 && old_b > 0.5) ||
                               (db < 0 && old_b < 0.5))) return;
                for (const auto& [r, coeff] : col_rows[sz(a)])
                    act[sz(rows[sz(r)].row)] += coeff * da;
                if (b >= 0) for (const auto& [r, coeff] : col_rows[sz(b)])
                    act[sz(rows[sz(r)].row)] += coeff * db;
                on[sz(a)] = static_cast<unsigned char>(old_a + da);
                if (b >= 0) on[sz(b)] = static_cast<unsigned char>(old_b + db);
                const f64 candidate = score_state(act, on);
                on[sz(a)] = static_cast<unsigned char>(old_a);
                if (b >= 0) on[sz(b)] = static_cast<unsigned char>(old_b);
                if (b >= 0) for (const auto& [r, coeff] : col_rows[sz(b)])
                    act[sz(rows[sz(r)].row)] -= coeff * db;
                for (const auto& [r, coeff] : col_rows[sz(a)])
                    act[sz(rows[sz(r)].row)] -= coeff * da;
                if (candidate + 1e-9 < best_move) {
                    best_move = candidate; best_a = a; best_b = b;
                    best_da = da; best_db = db;
                }
            };
            for (const Index j : bins) consider(j, on[sz(j)] ? -1 : 1, -1, 0);
            if (bins.size() <= 160) {
                for (std::size_t p = 0; p < bins.size(); ++p)
                    for (std::size_t q = p + 1; q < bins.size(); ++q) {
                        const Index a = bins[p], b = bins[q];
                        if (on[sz(a)] && !on[sz(b)]) consider(a, -1, b, 1);
                        else if (!on[sz(a)] && on[sz(b)]) consider(a, 1, b, -1);
                        else if (!on[sz(a)] && !on[sz(b)]) consider(a, 1, b, 1);
                    }
            }
            // Exact-fill instances often require a three-way exchange before
            // the residual objective improves. Enumerate triples only after
            // the cheaper one/two-bit neighborhood has reached a plateau.
            if (best_a < 0 && bins.size() <= 80) {
                for (std::size_t p = 0; p < bins.size() && best_a < 0; ++p)
                    for (std::size_t q = p + 1; q < bins.size() && best_a < 0; ++q)
                        for (std::size_t r = q + 1; r < bins.size(); ++r) {
                            const Index a = bins[p], b = bins[q], c = bins[r];
                            const int da = on[sz(a)] ? -1 : 1;
                            const int db = on[sz(b)] ? -1 : 1;
                            const int dc = on[sz(c)] ? -1 : 1;
                            for (const auto& [rr, coeff] : col_rows[sz(a)])
                                act[sz(rows[sz(rr)].row)] += coeff * da;
                            for (const auto& [rr, coeff] : col_rows[sz(b)])
                                act[sz(rows[sz(rr)].row)] += coeff * db;
                            for (const auto& [rr, coeff] : col_rows[sz(c)])
                                act[sz(rows[sz(rr)].row)] += coeff * dc;
                            on[sz(a)] = static_cast<unsigned char>(on[sz(a)] + da);
                            on[sz(b)] = static_cast<unsigned char>(on[sz(b)] + db);
                            on[sz(c)] = static_cast<unsigned char>(on[sz(c)] + dc);
                            const f64 candidate = score_state(act, on);
                            on[sz(a)] = static_cast<unsigned char>(on[sz(a)] - da);
                            on[sz(b)] = static_cast<unsigned char>(on[sz(b)] - db);
                            on[sz(c)] = static_cast<unsigned char>(on[sz(c)] - dc);
                            for (const auto& [rr, coeff] : col_rows[sz(c)])
                                act[sz(rows[sz(rr)].row)] -= coeff * dc;
                            for (const auto& [rr, coeff] : col_rows[sz(b)])
                                act[sz(rows[sz(rr)].row)] -= coeff * db;
                            for (const auto& [rr, coeff] : col_rows[sz(a)])
                                act[sz(rows[sz(rr)].row)] -= coeff * da;
                            if (candidate + 1e-9 < best_move) {
                                best_move = candidate;
                                best_a = a; best_b = b; best_da = da; best_db = db;
                                // Encode the third variable by applying the
                                // move immediately below through a temporary
                                // sentinel; its index is recovered from the
                                // current triple after the search.
                                best_c = c;
                                best_dc = dc;
                            }
                        }
            }
            if (best_a >= 0) {
                for (const auto& [r, coeff] : col_rows[sz(best_a)])
                    act[sz(rows[sz(r)].row)] += coeff * best_da;
                on[sz(best_a)] = static_cast<unsigned char>(on[sz(best_a)] + best_da);
                if (best_b >= 0) {
                    for (const auto& [r, coeff] : col_rows[sz(best_b)])
                        act[sz(rows[sz(r)].row)] += coeff * best_db;
                    on[sz(best_b)] = static_cast<unsigned char>(on[sz(best_b)] + best_db);
                }
                if (best_c >= 0) {
                    for (const auto& [r, coeff] : col_rows[sz(best_c)])
                        act[sz(rows[sz(r)].row)] += coeff * best_dc;
                    on[sz(best_c)] = static_cast<unsigned char>(on[sz(best_c)] + best_dc);
                }
                current = best_move;
            } else {
                // Random kick to cross a plateau, followed by the greedy
                // passes above on the next iterations.
                const std::size_t kick_count = 1 + static_cast<std::size_t>(next_rand() %
                    std::min<std::size_t>(5, bins.size()));
                std::vector<Index> changed;
                for (std::size_t z = 0; z < kick_count; ++z) {
                    const Index j = bins[static_cast<std::size_t>(next_rand() % bins.size())];
                    if (std::find(changed.begin(), changed.end(), j) != changed.end()) continue;
                    changed.push_back(j);
                    const int d = on[sz(j)] ? -1 : 1;
                    on[sz(j)] = static_cast<unsigned char>(on[sz(j)] + d);
                    for (const auto& [r, coeff] : col_rows[sz(j)])
                        act[sz(rows[sz(r)].row)] += coeff * d;
                }
                current = score_state(act, on);
            }
            bool feasible = true;
            for (const auto& row : rows)
                if (row.target - act[sz(row.row)] < -feas_tol) { feasible = false; break; }
            if (feasible && current < best_feasible) {
                best_feasible = current;
                restart_best_on = on;
            }
        }
        if (restart_best_on.empty()) continue;
        Candidate c;
        c.score = restart_best_feasible;
        c.x.assign(static_cast<std::size_t>(n), 0.0);
        for (const Index j : bins) c.x[sz(j)] = restart_best_on[sz(j)] ? 1.0 : 0.0;
        pool.push_back(std::move(c));
    }

    std::sort(pool.begin(), pool.end(), [](const Candidate& a, const Candidate& b) {
        return a.score < b.score;
    });
    if (static_cast<int>(pool.size()) > repair_trials)
        pool.resize(static_cast<std::size_t>(repair_trials));
    bool found = false;
    f64 best_obj = lp.maximize ? -std::numeric_limits<f64>::infinity()
                               : std::numeric_limits<f64>::infinity();
    for (const Candidate& c : pool) {
        if (over_budget()) break;
        std::vector<f64> repaired;
        if (!try_lp_rounding_repair(lp, c.x, int_tol, feas_tol,
                                    repair_iterations, repair_time_s,
                                    repaired)) {
            continue;
        }
        const f64 obj = lp.objective(repaired);
        if (!std::isfinite(obj)) continue;
        if (!found || (lp.maximize ? obj > best_obj : obj < best_obj)) {
            found = true;
            best_obj = obj;
            x_out = std::move(repaired);
        }
    }
    return found;
}

// Penalty local search for binary equality/knapsack blocks. Unlike the
// feasibility-only neighborhood, this deliberately crosses temporarily
// infeasible assignments, which is necessary for multidimensional subset-sum
// instances whose optimum is separated by an upper-bound violation.
bool try_binary_penalty_search(const model::LpProblem& lp,
                               f64 int_tol,
                               f64 feas_tol,
                               int restarts,
                               int iterations,
                               int lp_trials,
                               double time_limit_s,
                               std::uint64_t repair_iterations,
                               double repair_time_s,
                               std::vector<f64>& x_out) {
    const Index n = lp.n_cols();
    if (restarts <= 0 || iterations <= 0 || lp_trials <= 0) return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    std::vector<Index> bins;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (lp.col_lo[sz(j)] < -int_tol || lp.col_hi[sz(j)] > 1.0 + int_tol)
            continue;
        bins.push_back(j);
    }
    if (bins.size() < 8 || bins.size() * 2 < lp.n_integer()) return false;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<std::vector<std::pair<Index, f64>>> col_rows(sz(n));
    for (Index i = 0; i < lp.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            col_rows[sz(ci[sz(k)])].emplace_back(i, av[sz(k)]);
    const auto penalty = [&](const std::vector<f64>& act) {
        f64 p = 0.0;
        for (Index i = 0; i < lp.n_rows(); ++i) {
            const f64 lo = lp.row_lo[sz(i)], hi = lp.row_hi[sz(i)];
            const f64 under = act[sz(i)] < lo ? lo - act[sz(i)] : 0.0;
            const f64 over = act[sz(i)] > hi ? act[sz(i)] - hi : 0.0;
            p += under + 1000.0 * over;
        }
        return p;
    };
    struct Candidate { f64 score; std::vector<f64> x; };
    std::vector<Candidate> pool;
    pool.reserve(static_cast<std::size_t>(lp_trials));
    std::uint64_t state = 0x94d049bb133111ebull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };
    for (int restart = 0; restart < restarts && !over_budget(); ++restart) {
        std::vector<f64> trial(static_cast<std::size_t>(n), 0.0);
        if ((restart & 1) == 0) {
            // Start half the restarts from a feasible randomized packing. It
            // gives the flip search a useful incumbent basin instead of
            // spending its cooling schedule removing massive overshoots.
            std::vector<Index> order = bins;
            for (std::size_t q = order.size(); q > 1; --q) {
                const std::size_t r = static_cast<std::size_t>(next_rand() % q);
                std::swap(order[q - 1], order[r]);
            }
            for (const Index j : order) {
                bool fits = true;
                for (const auto& [i, a] : col_rows[sz(j)]) {
                    if (a > 0.0 &&
                        lp.row_hi[sz(i)] < a - feas_tol) { fits = false; break; }
                }
                if (!fits) continue;
                trial[sz(j)] = 1.0;
                bool upper_ok = true;
                for (Index i = 0; i < lp.n_rows(); ++i) {
                    f64 arow = 0.0;
                    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                        arow += av[sz(k)] * trial[sz(ci[sz(k)])];
                    if (arow > lp.row_hi[sz(i)] + feas_tol) {
                        upper_ok = false; break;
                    }
                }
                if (!upper_ok) trial[sz(j)] = 0.0;
            }
        } else {
            for (const Index j : bins)
                trial[sz(j)] = (next_rand() & 1ull) ? 1.0 : 0.0;
        }
        std::vector<f64> act(static_cast<std::size_t>(lp.n_rows()), 0.0);
        for (const Index j : bins)
            if (trial[sz(j)] != 0.0)
                for (const auto& [i, a] : col_rows[sz(j)]) act[sz(i)] += a;
        f64 score = penalty(act);
        f64 temp = std::max(1.0, score * 0.05);
        f64 best_feasible_score = std::numeric_limits<f64>::infinity();
        std::vector<f64> best_feasible;
        for (int it = 0; it < iterations && !over_budget(); ++it) {
            const Index j = bins[static_cast<std::size_t>(next_rand() % bins.size())];
            const f64 old = trial[sz(j)];
            const f64 neu = old > 0.5 ? 0.0 : 1.0;
            const f64 delta = neu - old;
            f64 next_score = score;
            for (const auto& [i, a] : col_rows[sz(j)]) {
                const f64 before = act[sz(i)];
                const f64 after = before + a * delta;
                const f64 lo = lp.row_lo[sz(i)], hi = lp.row_hi[sz(i)];
                const f64 pb = (before < lo ? lo - before : 0.0) +
                               1000.0 * (before > hi ? before - hi : 0.0);
                const f64 pa = (after < lo ? lo - after : 0.0) +
                               1000.0 * (after > hi ? after - hi : 0.0);
                next_score += pa - pb;
            }
            const bool accept = next_score <= score ||
                (temp > 1e-9 &&
                 static_cast<f64>(next_rand() % 1000000) / 1000000.0 <
                     std::exp(std::min(0.0, (score - next_score) / temp)));
            if (accept) {
                trial[sz(j)] = neu;
                for (const auto& [i, a] : col_rows[sz(j)]) act[sz(i)] += a * delta;
                score = next_score;
            }
            temp *= 0.9995;
            bool feasible_upper = true;
            for (Index i = 0; i < lp.n_rows(); ++i)
                if (act[sz(i)] > lp.row_hi[sz(i)] + feas_tol) {
                    feasible_upper = false;
                    break;
                }
            if (feasible_upper) {
                f64 deficit = 0.0;
                for (Index i = 0; i < lp.n_rows(); ++i)
                    if (act[sz(i)] < lp.row_lo[sz(i)])
                        deficit += lp.row_lo[sz(i)] - act[sz(i)];
                if (deficit < best_feasible_score) {
                    best_feasible_score = deficit;
                    best_feasible = trial;
                }
            }
        }
        if (!best_feasible.empty()) {
            Candidate c;
            c.score = best_feasible_score;
            c.x = std::move(best_feasible);
            pool.push_back(std::move(c));
        }
    }
    std::sort(pool.begin(), pool.end(), [](const Candidate& a, const Candidate& b) {
        return a.score < b.score;
    });
    if (static_cast<int>(pool.size()) > lp_trials)
        pool.resize(static_cast<std::size_t>(lp_trials));
    bool found = false;
    f64 best_obj = lp.maximize ? -std::numeric_limits<f64>::infinity()
                               : std::numeric_limits<f64>::infinity();
    for (const Candidate& c : pool) {
        if (over_budget()) break;
        std::vector<f64> repaired;
        if (!try_lp_rounding_repair(lp, c.x, int_tol, feas_tol,
                                    repair_iterations, repair_time_s,
                                    repaired))
            continue;
        const f64 obj = lp.objective(repaired);
        if (!std::isfinite(obj)) continue;
        if (!found || (lp.maximize ? obj > best_obj : obj < best_obj)) {
            found = true;
            best_obj = obj;
            x_out = std::move(repaired);
        }
    }
    return found;
}

// Re-optimize the continuous variables while exploring a bounded Hamming
// neighborhood of an integer incumbent. This is the small-instance form of
// RINS/local branching: fixing all integer variables makes every trial an LP,
// while one-variable moves and one-for-one swaps escape poor independent
// rounding (notably subset-sum and assignment models). It is incumbent-only;
// no trial is used for node bounds or pruning.
bool try_integer_neighborhood(const model::LpProblem& lp,
                              const std::vector<f64>& seed,
                              f64 int_tol,
                              f64 feas_tol,
                              std::uint64_t max_trials,
                              double time_limit_s,
                              std::uint64_t repair_iterations,
                              double repair_time_s,
                              std::vector<f64>& x_out) {
    if (static_cast<Index>(seed.size()) != lp.n_cols() || max_trials == 0)
        return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    std::vector<f64> best = seed;
    f64 best_obj = lp.objective(best);
    if (!std::isfinite(best_obj) || lp.max_row_violation(best) > feas_tol ||
        lp.max_bound_violation(best) > feas_tol)
        return false;

    bool improved = false;
    std::uint64_t trials = 0;
    const auto better = [&](f64 a, f64 b) {
        return lp.maximize ? a > b : a < b;
    };
    const bool all_integer =
        static_cast<Index>(std::count(lp.is_integer.begin(),
                                      lp.is_integer.end(), true)) ==
        lp.n_cols();

    // For a fully integral model, fixing a trial assignment and solving an LP
    // is redundant.  Use direct feasible integer exchanges instead.  Unequal
    // exchanges (one-for-two and two-for-one) are important for covering and
    // generalized-assignment models where a cheap item can replace several
    // expensive units.  This bounded local search is exact with respect to
    // every accepted incumbent but remains heuristic-only.
    if (all_integer && lp.n_cols() <= 400) {
        std::vector<Index> movable;
        for (Index j = 0; j < lp.n_cols(); ++j)
            if (lp.col_hi[sz(j)] - lp.col_lo[sz(j)] > int_tol)
                movable.push_back(j);
        const auto feasible = [&](const std::vector<f64>& x) {
            return lp.max_row_violation(x) <= feas_tol &&
                   lp.max_bound_violation(x) <= feas_tol;
        };
        if (feasible(best) && !movable.empty()) {
            for (int pass = 0; pass < 6 && trials < max_trials &&
                                !over_budget(); ++pass) {
                bool pass_improved = false;
                std::vector<f64> next_best = best;
                f64 next_obj = best_obj;
                auto evaluate = [&](const std::vector<f64>& trial) {
                    if (trials >= max_trials || over_budget()) return;
                    ++trials;
                    if (!feasible(trial)) return;
                    const f64 obj = lp.objective(trial);
                    if (std::isfinite(obj) && better(obj, next_obj)) {
                        next_obj = obj;
                        next_best = trial;
                        pass_improved = true;
                    }
                };
                for (const Index j : movable) {
                    const f64 v = std::round(best[sz(j)]);
                    for (int d : {-1, 1}) {
                        const f64 nv = std::min(std::max(v + d,
                            lp.col_lo[sz(j)]), lp.col_hi[sz(j)]);
                        if (std::fabs(nv - v) <= int_tol) continue;
                        std::vector<f64> trial = best;
                        trial[sz(j)] = nv;
                        evaluate(trial);
                    }
                }
                // One-for-two and two-for-one exchanges.  Limit the pair
                // enumeration for wide models, but cover all pairs on the
                // small structures this fast path is intended for.
                for (const Index jd : movable) {
                    const f64 vd = std::round(best[sz(jd)]);
                    if (vd <= lp.col_lo[sz(jd)] + int_tol) continue;
                    for (const Index ju : movable) {
                        if (jd == ju) continue;
                        const f64 vu = std::round(best[sz(ju)]);
                        if (vu >= lp.col_hi[sz(ju)] - int_tol) continue;
                        std::vector<f64> trial = best;
                        trial[sz(jd)] = vd - 1.0;
                        trial[sz(ju)] = std::min(lp.col_hi[sz(ju)], vu + 2.0);
                        evaluate(trial);
                        if (vu + 2.0 <= lp.col_hi[sz(ju)] + int_tol) {
                            trial[sz(jd)] = vd - 2.0;
                            trial[sz(ju)] = vu + 1.0;
                            if (vd - 2.0 >= lp.col_lo[sz(jd)] - int_tol)
                                evaluate(trial);
                        }
                        if (trials >= max_trials || over_budget()) break;
                    }
                    if (trials >= max_trials || over_budget()) break;
                }
                if (!pass_improved) break;
                best_obj = next_obj;
                best = std::move(next_best);
                improved = true;
            }
            // Randomized destroy/repair escapes multi-unit plateaus that are
            // invisible to the short exchange passes above.  Removed units
            // are restored by selecting the best feasible +1 move according
            // to reduction in total lower-row violation per objective cost.
            std::uint64_t dr_state = 0x2545f4914f6cdd1dull;
            const auto dr_rand = [&]() {
                dr_state ^= dr_state << 7; dr_state ^= dr_state >> 9;
                dr_state ^= dr_state << 8; return dr_state;
            };
            const auto activity_of = [&](const std::vector<f64>& x) {
                std::vector<f64> act(static_cast<std::size_t>(lp.n_rows()), 0.0);
                const auto& rp = lp.A.pattern.row_ptr();
                const auto& ci = lp.A.pattern.col_idx();
                for (Index i = 0; i < lp.n_rows(); ++i)
                    for (core::Offset k = rp[sz(i)];
                         k < rp[sz(i) + 1]; ++k)
                        act[sz(i)] += lp.A.vals[sz(k)] * x[sz(ci[sz(k)])];
                return act;
            };
            const int destroy_restarts = lp.n_cols() <= 220 ? 1200 : 300;
            for (int restart = 0; restart < destroy_restarts &&
                                  trials < max_trials && !over_budget(); ++restart) {
                std::vector<f64> trial = best;
                const int removes = 1 + static_cast<int>(dr_rand() % 8u);
                for (int q = 0; q < removes; ++q) {
                    const Index j = movable[static_cast<std::size_t>(
                        dr_rand() % movable.size())];
                    const f64 v = std::round(trial[sz(j)]);
                    if (v > lp.col_lo[sz(j)] + int_tol)
                        trial[sz(j)] = v - 1.0;
                }
                for (int repair = 0; repair < lp.n_cols() * 4; ++repair) {
                    const auto act = activity_of(trial);
                    f64 before = 0.0;
                    for (Index i = 0; i < lp.n_rows(); ++i)
                        if (act[sz(i)] < lp.row_lo[sz(i)])
                            before += lp.row_lo[sz(i)] - act[sz(i)];
                    if (before <= feas_tol) break;
                    Index best_j = -1;
                    f64 best_score = -std::numeric_limits<f64>::infinity();
                    for (const Index j : movable) {
                        const f64 v = std::round(trial[sz(j)]);
                        if (v >= lp.col_hi[sz(j)] - int_tol) continue;
                        bool upper_ok = true;
                        f64 after = before;
                        for (Index i = 0; i < lp.n_rows(); ++i) {
                            f64 delta = 0.0;
                            for (core::Offset k = lp.A.pattern.row_ptr()[sz(i)];
                                 k < lp.A.pattern.row_ptr()[sz(i) + 1]; ++k)
                                if (lp.A.pattern.col_idx()[sz(k)] == j)
                                    delta += lp.A.vals[sz(k)];
                            if (delta == 0.0) continue;
                            const f64 neu = act[sz(i)] + delta;
                            if (neu > lp.row_hi[sz(i)] + feas_tol) {
                                upper_ok = false; break;
                            }
                            const f64 old_v = std::max(0.0,
                                lp.row_lo[sz(i)] - act[sz(i)]);
                            const f64 new_v = std::max(0.0,
                                lp.row_lo[sz(i)] - neu);
                            after += new_v - old_v;
                        }
                        if (!upper_ok) continue;
                        const f64 gain = before - after;
                        if (gain <= 1e-12) continue;
                        const f64 cost = std::max(0.0, lp.c[sz(j)]);
                        const f64 score = gain / (1.0 + cost) +
                            1.0e-9 * static_cast<f64>(dr_rand() & 0xffffu);
                        if (score > best_score) {
                            best_score = score; best_j = j;
                        }
                    }
                    if (best_j < 0) break;
                    trial[sz(best_j)] = std::round(trial[sz(best_j)]) + 1.0;
                }
                ++trials;
                if (!feasible(trial)) continue;
                const f64 obj = lp.objective(trial);
                if (std::isfinite(obj) && better(obj, best_obj)) {
                    best_obj = obj;
                    best = std::move(trial);
                    improved = true;
                }
            }
            if (improved) x_out = best;
            return improved;
        }
    }
    const auto probe = [&](const std::vector<f64>& trial,
                           f64& obj_out,
                           std::vector<f64>& point_out) {
        if (trials >= max_trials || over_budget()) return false;
        ++trials;
        std::vector<f64> repaired;
        if (all_integer) {
            if (lp.max_row_violation(trial) > feas_tol ||
                lp.max_bound_violation(trial) > feas_tol)
                return false;
            repaired = trial;
        } else if (!try_lp_rounding_repair(lp, trial, int_tol, feas_tol,
                                           repair_iterations, repair_time_s,
                                           repaired)) {
            return false;
        }
        const f64 obj = lp.objective(repaired);
        if (!std::isfinite(obj)) return false;
        obj_out = obj;
        point_out = std::move(repaired);
        return true;
    };

    // Coordinate descent over the integer lattice. The former implementation
    // only moved variables once from the initial seed, which made a useful
    // exchange impossible after the first accepted move. Rebuild the move
    // lists after every improving pass so a binary can be flipped repeatedly
    // and general integers can walk several units toward a better assignment.
    const int max_passes = 8;
    for (int pass = 0; pass < max_passes && trials < max_trials &&
                        !over_budget(); ++pass) {
        std::vector<Index> movable;
        movable.reserve(static_cast<std::size_t>(lp.n_cols()));
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            const f64 v = std::round(best[sz(j)]);
            if (v > lp.col_lo[sz(j)] + int_tol ||
                v < lp.col_hi[sz(j)] - int_tol)
                movable.push_back(j);
        }

        bool pass_improved = false;
        std::vector<f64> pass_best;
        f64 pass_obj = best_obj;
        for (const Index j : movable) {
            const f64 v = std::round(best[sz(j)]);
            for (int dir : {-1, 1}) {
                if (trials >= max_trials || over_budget()) break;
                const f64 next = std::min(
                    std::max(v + static_cast<f64>(dir), lp.col_lo[sz(j)]),
                    lp.col_hi[sz(j)]);
                if (std::fabs(next - v) <= int_tol) continue;
                std::vector<f64> trial = best;
                trial[sz(j)] = next;
                f64 obj = 0.0;
                std::vector<f64> point;
                if (probe(trial, obj, point) && better(obj, pass_obj)) {
                    pass_obj = obj;
                    pass_best = std::move(point);
                    pass_improved = true;
                }
            }
        }
        if (pass_improved) {
            best_obj = pass_obj;
            best = std::move(pass_best);
            improved = true;
            continue;
        }

        // At a one-coordinate local minimum, test one-for-one exchanges. This
        // is the important step for exact-sum/assignment models: removing one
        // selected item may require adding another in the same move. Cap the
        // Cartesian product so large instances keep a predictable budget.
        std::vector<Index> down, up;
        for (const Index j : movable) {
            const f64 v = std::round(best[sz(j)]);
            if (v > lp.col_lo[sz(j)] + int_tol) down.push_back(j);
            if (v < lp.col_hi[sz(j)] - int_tol) up.push_back(j);
        }
        f64 swap_obj = best_obj;
        std::vector<f64> swap_best;
        const std::uint64_t swap_cap = std::min<std::uint64_t>(
            max_trials - trials, 4096);
        std::uint64_t swap_count = 0;
        for (const Index jd : down) {
            for (const Index ju : up) {
                if (jd == ju || swap_count >= swap_cap ||
                    trials >= max_trials || over_budget()) break;
                std::vector<f64> trial = best;
                const f64 vd = std::round(best[sz(jd)]);
                const f64 vu = std::round(best[sz(ju)]);
                trial[sz(jd)] = std::max(lp.col_lo[sz(jd)], vd - 1.0);
                trial[sz(ju)] = std::min(lp.col_hi[sz(ju)], vu + 1.0);
                if (std::fabs(trial[sz(jd)] - vd) <= int_tol ||
                    std::fabs(trial[sz(ju)] - vu) <= int_tol)
                    continue;
                ++swap_count;
                f64 obj = 0.0;
                std::vector<f64> point;
                if (probe(trial, obj, point) && better(obj, swap_obj)) {
                    swap_obj = obj;
                    swap_best = std::move(point);
                }
            }
            if (swap_count >= swap_cap || trials >= max_trials || over_budget())
                break;
        }
        if (!swap_best.empty()) {
            best_obj = swap_obj;
            best = std::move(swap_best);
            improved = true;
            continue;
        }
        break;
    }

    // A small deterministic multi-exchange tail catches plateaus where no
    // single move or pair swap improves, without spending the whole B&B
    // budget on thousands of LPs.
    if (trials < max_trials && !over_budget()) {
        std::uint64_t state = 0x9e3779b97f4a7c15ull;
        const auto next_rand = [&]() {
            state ^= state << 7; state ^= state >> 9; state ^= state << 8;
            return state;
        };
        for (std::uint64_t attempts = 0;
             attempts < 512 && trials < max_trials && !over_budget(); ++attempts) {
            std::vector<Index> down, up;
            for (Index j = 0; j < lp.n_cols(); ++j) {
                if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
                const f64 v = std::round(best[sz(j)]);
                if (v > lp.col_lo[sz(j)] + int_tol) down.push_back(j);
                if (v < lp.col_hi[sz(j)] - int_tol) up.push_back(j);
            }
            if (down.empty() || up.empty()) break;
            const std::size_t k = 1 +
                static_cast<std::size_t>(next_rand() %
                    std::min<std::size_t>(4, std::min(down.size(), up.size())));
            std::vector<f64> trial = best;
            std::vector<Index> used_d, used_u;
            for (std::size_t q = 0; q < k; ++q) {
                const Index jd = down[static_cast<std::size_t>(next_rand() % down.size())];
                const Index ju = up[static_cast<std::size_t>(next_rand() % up.size())];
                if (std::find(used_d.begin(), used_d.end(), jd) != used_d.end() ||
                    std::find(used_u.begin(), used_u.end(), ju) != used_u.end()) {
                    used_d.clear();
                    break;
                }
                used_d.push_back(jd); used_u.push_back(ju);
            }
            if (used_d.size() != k) continue;
            for (const Index j : used_d)
                trial[sz(j)] = std::round(best[sz(j)]) - 1.0;
            for (const Index j : used_u)
                trial[sz(j)] = std::round(best[sz(j)]) + 1.0;
            f64 obj = 0.0;
            std::vector<f64> point;
            if (probe(trial, obj, point) && better(obj, best_obj)) {
                best_obj = obj;
                best = std::move(point);
                improved = true;
            }
        }
    }

    // Perturb-and-polish restarts cross barriers that require a temporarily
    // worse feasible assignment before a two/three-variable exchange becomes
    // profitable. Keep the global incumbent separate from the restart point.
    if (trials < max_trials && !over_budget()) {
        std::uint64_t state = 0x6a09e667f3bcc909ull;
        const auto next_rand = [&]() {
            state ^= state << 7; state ^= state >> 9; state ^= state << 8;
            return state;
        };
        for (int restart = 0; restart < 256 && trials < max_trials &&
                              !over_budget(); ++restart) {
            std::vector<Index> movable;
            for (Index j = 0; j < lp.n_cols(); ++j) {
                if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
                const f64 v = std::round(best[sz(j)]);
                if (v > lp.col_lo[sz(j)] + int_tol ||
                    v < lp.col_hi[sz(j)] - int_tol)
                    movable.push_back(j);
            }
            if (movable.empty()) break;
            const std::size_t k = 1 + static_cast<std::size_t>(
                next_rand() % std::min<std::size_t>(6, movable.size()));
            std::vector<f64> trial = best;
            std::vector<Index> used;
            for (std::size_t q = 0; q < k; ++q) {
                const Index j = movable[static_cast<std::size_t>(
                    next_rand() % movable.size())];
                if (std::find(used.begin(), used.end(), j) != used.end()) {
                    used.clear();
                    break;
                }
                used.push_back(j);
                const f64 v = std::round(best[sz(j)]);
                if (v <= lp.col_lo[sz(j)] + int_tol)
                    trial[sz(j)] = std::min(lp.col_hi[sz(j)], v + 1.0);
                else if (v >= lp.col_hi[sz(j)] - int_tol)
                    trial[sz(j)] = std::max(lp.col_lo[sz(j)], v - 1.0);
                else
                    trial[sz(j)] = (next_rand() & 1ull) ? v + 1.0 : v - 1.0;
            }
            if (used.size() != k) continue;
            f64 obj = 0.0;
            std::vector<f64> point;
            if (!probe(trial, obj, point)) continue;
            if (better(obj, best_obj)) {
                best_obj = obj;
                best = point;
                improved = true;
            }
            // One cheap improving pass from the perturbed point.
            for (const Index j : movable) {
                if (trials >= max_trials || over_budget()) break;
                const f64 v = std::round(point[sz(j)]);
                for (int dir : {-1, 1}) {
                    const f64 next = std::min(
                        std::max(v + static_cast<f64>(dir), lp.col_lo[sz(j)]),
                        lp.col_hi[sz(j)]);
                    if (std::fabs(next - v) <= int_tol) continue;
                    std::vector<f64> one = point;
                    one[sz(j)] = next;
                    f64 one_obj = 0.0;
                    std::vector<f64> one_point;
                    if (probe(one, one_obj, one_point) &&
                        better(one_obj, best_obj)) {
                        best_obj = one_obj;
                        best = std::move(one_point);
                        improved = true;
                    }
                }
            }
        }
    }

    if (improved) x_out = std::move(best);
    return improved;
}

// Feasibility-pump style incumbent search. Alternate rounding the integer
// coordinates with an LP whose objective points toward the rounded target.
// This can repair equality-heavy models where fixing every rounded integer
// makes the continuous subproblem infeasible. It is deliberately bounded and
// is never used for node bounds or pruning.
bool try_feasibility_pump(const model::LpProblem& lp,
                          const std::vector<f64>& x_start,
                          f64 int_tol,
                          f64 feas_tol,
                          int max_rounds,
                          std::uint64_t max_iterations,
                          double time_limit_s,
                          std::vector<f64>& x_out) {
    if (static_cast<Index>(x_start.size()) != lp.n_cols()) return false;
    std::vector<f64> current = x_start;
    std::vector<f64> previous_target;
    for (int round = 0; round < max_rounds; ++round) {
        std::vector<f64> target = current;
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            f64 v = std::round(current[sz(j)]);
            if (!std::isfinite(v)) return false;
            target[sz(j)] = std::min(std::max(v, lp.col_lo[sz(j)]), lp.col_hi[sz(j)]);
            if (!is_integral(target[sz(j)], int_tol)) return false;
        }

        std::vector<f64> candidate;
        if (try_round(lp, current, int_tol, feas_tol, candidate)) {
            x_out = std::move(candidate);
            return true;
        }
        if (target == previous_target) break;
        previous_target = target;

        // Build the standard feasibility-pump L1 projection LP. For every
        // integer coordinate j, an auxiliary d_j is constrained by
        // d_j >= x_j-target_j and d_j >= target_j-x_j. Minimizing sum(d_j)
        // is materially different from a signed objective: it attracts the
        // whole LP point to the rounded target without forcing unrelated
        // variables to a bound.
        const Index n = lp.n_cols();
        const Index ni = static_cast<Index>(
            std::count(lp.is_integer.begin(), lp.is_integer.end(), true));
        model::LpProblem pump = lp;
        pump.name = lp.name + "_fp_projection";
        pump.maximize = false;
        pump.obj_offset = 0.0;
        pump.c.assign(static_cast<std::size_t>(n + ni), 0.0);
        pump.col_lo = lp.col_lo;
        pump.col_hi = lp.col_hi;
        pump.col_lo.resize(static_cast<std::size_t>(n + ni), 0.0);
        pump.col_hi.resize(static_cast<std::size_t>(n + ni), model::kInf);
        pump.is_integer.assign(static_cast<std::size_t>(n + ni), false);
        pump.row_lo = lp.row_lo;
        pump.row_hi = lp.row_hi;
        pump.row_names = lp.row_names;
        pump.row_names.resize(static_cast<std::size_t>(lp.n_rows() + 2 * ni));
        pump.col_names = lp.col_names;
        pump.col_names.resize(static_cast<std::size_t>(n + ni));
        std::vector<Index> rows, cols;
        std::vector<f64> vals;
        rows.reserve(static_cast<std::size_t>(lp.nnz() + 2 * ni));
        cols.reserve(rows.capacity());
        vals.reserve(rows.capacity());
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        for (Index i = 0; i < lp.n_rows(); ++i) {
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                rows.push_back(i);
                cols.push_back(ci[sz(k)]);
                vals.push_back(lp.A.vals[sz(k)]);
            }
        }
        Index d = 0;
        for (Index j = 0; j < n; ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            const Index dj = n + d;
            pump.c[sz(dj)] = 1.0;
            pump.col_names[sz(dj)] = "FP_D_" + std::to_string(j);
            const Index r1 = lp.n_rows() + 2 * d;
            const Index r2 = r1 + 1;
            pump.row_lo.push_back(-model::kInf);
            pump.row_hi.push_back(target[sz(j)]);
            pump.row_lo.push_back(-model::kInf);
            pump.row_hi.push_back(-target[sz(j)]);
            rows.push_back(r1); cols.push_back(j);  vals.push_back(1.0);
            rows.push_back(r1); cols.push_back(dj); vals.push_back(-1.0);
            rows.push_back(r2); cols.push_back(j);  vals.push_back(-1.0);
            rows.push_back(r2); cols.push_back(dj); vals.push_back(-1.0);
            ++d;
        }
        pump.A = sparse::from_triplets(lp.n_rows() + 2 * ni, n + ni,
                                       rows, cols, vals);
        pump.validate();
        engines::SimplexOptions pump_opts;
        pump_opts.method = engines::SimplexMethod::Primal;
        pump_opts.presolve = true;
        pump_opts.max_iterations = max_iterations;
        pump_opts.time_limit_s = time_limit_s;
        pump_opts.primal_feas_tol = feas_tol;
        pump_opts.dual_feas_tol = std::max(feas_tol, 1e-7);
        engines::SimplexDiagnostics pump_diag;
        const auto pumped = engines::solve_simplex(pump, pump_opts,
                                                   pump_diag, nullptr);
        if (pumped.proposed_status != core::Status::Optimal &&
            pumped.proposed_status != core::Status::Feasible)
            break;
        if (static_cast<Index>(pumped.x.size()) < lp.n_cols()) break;
        current.assign(pumped.x.begin(), pumped.x.begin() + n);
    }
    return false;
}

// Follow one LP relaxation down a bounded depth-first branch path. Unlike a
// normal B&B node, an unproved child is still useful here as a source of a
// feasible point; only the final original-model check can accept it. This is
// the standard LP-diving incumbent heuristic, kept separate from the proof
// search so an interrupted dive can never change global bounds.
bool try_integer_dive(const model::LpProblem& lp,
                      const std::vector<f64>& x_start,
                      const std::vector<f64>& root_lo,
                      const std::vector<f64>& root_hi,
                      const engines::SimplexBasis* root_basis,
                      f64 int_tol,
                      f64 feas_tol,
                      std::uint64_t max_nodes,
                      double total_time_s,
                      double lp_time_s,
                      f64 objective_sense,
                      const engines::SimplexOptions& base_opts,
                      std::vector<f64>& x_out,
                      std::uint64_t& lp_solves) {
    lp_solves = 0;
    if (static_cast<Index>(x_start.size()) != lp.n_cols() ||
        static_cast<Index>(root_lo.size()) != lp.n_cols() ||
        static_cast<Index>(root_hi.size()) != lp.n_cols() ||
        max_nodes == 0)
        return false;

    struct DiveNode {
        std::vector<f64> lo;
        std::vector<f64> hi;
        engines::SimplexBasis basis;
        bool has_basis = false;
        int depth = 0;
    };

    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return total_time_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   total_time_s;
    };
    const auto usable_point = [&](const core::RawResult& r,
                                  const engines::SimplexDiagnostics& d) {
        if (static_cast<Index>(r.x.size()) != lp.n_cols()) return false;
        for (const f64 v : r.x)
            if (!std::isfinite(v)) return false;
        // Interrupted/Feasible LP stages can still expose a valid primal
        // point. Residual checks are mandatory because status alone is not.
        return d.primal_residual <= feas_tol &&
               lp.max_row_violation(r.x) <= feas_tol &&
               lp.max_bound_violation(r.x) <= feas_tol;
    };
    const auto integer_point = [&](const std::vector<f64>& x) {
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (!lp.is_integer.empty() && lp.is_integer[sz(j)] &&
                !is_integral(x[sz(j)], int_tol))
                return false;
        }
        return true;
    };
    const auto branch_var = [&](const std::vector<f64>& x,
                                const std::vector<f64>& lo,
                                const std::vector<f64>& hi) {
        Index best = -1;
        f64 best_score = -1.0;
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            if (lo[sz(j)] >= hi[sz(j)] - int_tol) continue;
            if (is_integral(x[sz(j)], int_tol)) continue;
            const f64 score = frac_score(x[sz(j)]);
            if (score > best_score) {
                best_score = score;
                best = j;
            }
        }
        return best;
    };

    std::vector<DiveNode> stack;
    DiveNode root;
    root.lo = root_lo;
    root.hi = root_hi;
    if (root_basis != nullptr && !root_basis->basic.empty()) {
        root.basis = *root_basis;
        root.has_basis = true;
    }
    stack.push_back(std::move(root));

    std::uint64_t visited = 0;
    bool found = false;
    f64 best_obj = objective_sense > 0.0
        ? std::numeric_limits<f64>::infinity()
        : -std::numeric_limits<f64>::infinity();
    std::vector<f64> best_point;
    while (!stack.empty() && visited < max_nodes && !over_budget()) {
        DiveNode node = std::move(stack.back());
        stack.pop_back();
        ++visited;

        model::LpProblem child = lp;
        child.col_lo = node.lo;
        child.col_hi = node.hi;

        engines::SimplexOptions solve_opts = base_opts;
        solve_opts.method = node.has_basis ? engines::SimplexMethod::Dual
                                            : engines::SimplexMethod::Primal;
        solve_opts.presolve = false;
        solve_opts.max_iterations = base_opts.max_iterations == 0
            ? 10000 : base_opts.max_iterations;
        solve_opts.time_limit_s = lp_time_s;
        solve_opts.primal_feas_tol = feas_tol;
        solve_opts.dual_feas_tol = std::max(feas_tol, 1e-7);

        engines::SimplexDiagnostics sd;
        engines::SimplexBasis basis;
        core::RawResult r;
        if (node.has_basis) {
            r = engines::solve_dual_simplex(child, solve_opts, sd, &basis,
                                            &node.basis);
        } else {
            r = engines::solve_simplex(child, solve_opts, sd, &basis);
        }
        ++lp_solves;

        if (r.proposed_status == core::Status::Infeasible ||
            r.proposed_status == core::Status::InfeasibleOrUnbounded ||
            !usable_point(r, sd))
            continue;

        if (integer_point(r.x)) {
            // Check the original bounds and rows, not a relaxed/dived copy.
            if (lp.max_row_violation(r.x) <= feas_tol &&
                lp.max_bound_violation(r.x) <= feas_tol) {
                const f64 obj = lp.objective(r.x);
                const bool better = !found ||
                    (objective_sense > 0.0 ? obj < best_obj : obj > best_obj);
                if (std::isfinite(obj) && better) {
                    found = true;
                    best_obj = obj;
                    best_point = std::move(r.x);
                }
            }
            continue;
        }

        const Index br = branch_var(r.x, node.lo, node.hi);
        if (br < 0) continue;
        const f64 xv = r.x[sz(br)];
        const f64 floor_v = std::floor(xv);
        const f64 ceil_v = std::ceil(xv);
        if (floor_v < node.lo[sz(br)] - int_tol &&
            ceil_v > node.hi[sz(br)] + int_tol)
            continue;

        DiveNode down;
        down.lo = node.lo;
        down.hi = node.hi;
        down.hi[sz(br)] = std::min(down.hi[sz(br)], floor_v);
        down.basis = basis;
        down.has_basis = !basis.basic.empty();
        down.depth = node.depth + 1;

        DiveNode up;
        up.lo = node.lo;
        up.hi = node.hi;
        up.lo[sz(br)] = std::max(up.lo[sz(br)], ceil_v);
        up.basis = basis;
        up.has_basis = !basis.basic.empty();
        up.depth = node.depth + 1;

        const f64 down_delta = objective_sense * lp.c[sz(br)] *
                               (floor_v - xv);
        const f64 up_delta = objective_sense * lp.c[sz(br)] *
                             (ceil_v - xv);
        // Prefer the direction that improves structural coverage when the
        // objective coefficient is zero (the common slack-variable pattern).
        // Otherwise retain the objective-directed ordering.
        const f64 bias = integer_up_bias(lp, br);
        const bool down_first = std::fabs(bias) > 1.0e-10
            ? bias < 0.0 : down_delta <= up_delta;
        if (down_first) {
            if (up.lo[sz(br)] <= up.hi[sz(br)] + int_tol)
                stack.push_back(std::move(up));
            if (down.lo[sz(br)] <= down.hi[sz(br)] + int_tol)
                stack.push_back(std::move(down));
        } else {
            if (down.lo[sz(br)] <= down.hi[sz(br)] + int_tol)
                stack.push_back(std::move(down));
            if (up.lo[sz(br)] <= up.hi[sz(br)] + int_tol)
                stack.push_back(std::move(up));
        }
    }
    if (found) x_out = std::move(best_point);
    return found;
}

// Relaxation Enforced Neighborhood Search (RENS): fix integer variables whose
// LP value is already close to an integer and run a short dive on the smaller
// remaining subproblem. This is incumbent-only; the original model is still
// used for the final feasibility/objective check by the caller.
bool try_rens(const model::LpProblem& lp,
              const std::vector<f64>& x_lp,
              const engines::SimplexBasis* basis,
              f64 int_tol,
              f64 feas_tol,
              std::uint64_t max_nodes,
              double time_limit_s,
              double lp_time_s,
              const engines::SimplexOptions& base_opts,
              std::vector<f64>& x_out,
              std::uint64_t& lp_solves) {
    if (static_cast<Index>(x_lp.size()) != lp.n_cols()) return false;
    model::LpProblem sub = lp;
    int fixed = 0;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        const f64 v = x_lp[sz(j)];
        const f64 r = std::round(v);
        if (std::fabs(v - r) <= 0.15 &&
            r >= lp.col_lo[sz(j)] - int_tol &&
            r <= lp.col_hi[sz(j)] + int_tol) {
            sub.col_lo[sz(j)] = std::max(lp.col_lo[sz(j)], r);
            sub.col_hi[sz(j)] = std::min(lp.col_hi[sz(j)], r);
            ++fixed;
        }
    }
    if (fixed == 0 || fixed >= static_cast<int>(lp.n_integer())) return false;
    return try_integer_dive(sub, x_lp, sub.col_lo, sub.col_hi, basis,
                            int_tol, feas_tol, max_nodes, time_limit_s,
                            lp_time_s, lp.maximize ? -1.0 : 1.0,
                            base_opts, x_out, lp_solves);
}

Index pick_branch_var(const model::LpProblem& lp,
                      const std::vector<f64>& col_lo,
                      const std::vector<f64>& col_hi,
                      const std::vector<f64>& x,
                      f64 int_tol) {
    Index best = -1;
    f64 best_frac = 0.0;
    const Index n = lp.n_cols();
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (is_integral(x[sz(j)], int_tol)) continue;
        // Skip fixed integers.
        if (col_lo[sz(j)] == col_hi[sz(j)]) continue;
        const f64 sc = frac_score(x[sz(j)]);
        if (sc > best_frac) {
            best_frac = sc;
            best = j;
        }
    }
    return best;
}

std::vector<Index> branch_candidates(const model::LpProblem& lp,
                                     const std::vector<f64>& col_lo,
                                     const std::vector<f64>& col_hi,
                                     const std::vector<f64>& x,
                                     const std::vector<Index>& col_degree,
                                     f64 int_tol,
                                     int limit) {
    std::vector<std::pair<f64, Index>> ranked;
    const Index n = lp.n_cols();
    ranked.reserve(static_cast<std::size_t>(n));
    bool has_coupled = false;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (is_integral(x[sz(j)], int_tol)) continue;
        if (col_lo[sz(j)] == col_hi[sz(j)]) continue;
        if (sz(j) < col_degree.size() && col_degree[sz(j)] > 1) {
            has_coupled = true;
            break;
        }
    }
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (is_integral(x[sz(j)], int_tol)) continue;
        if (col_lo[sz(j)] == col_hi[sz(j)]) continue;
        // Keep singleton objective/slack variables eligible: equality rows can
        // make them implied integers, and branching on that slack often
        // exposes a strong incumbent (markshare/pk1). Pure zero-cost
        // singletons remain filtered when coupled variables exist.
        if (has_coupled && sz(j) < col_degree.size() && col_degree[sz(j)] <= 1 &&
            std::fabs(lp.c[sz(j)]) <= int_tol)
            continue;
        const f64 degree = (sz(j) < col_degree.size())
            ? static_cast<f64>(col_degree[sz(j)]) : 1.0;
        // Variables touching more rows have greater propagation impact. This
        // keeps a singleton auxiliary (for example a startup indicator) from
        // dominating a structurally coupled decision variable at cold start.
        const f64 score = frac_score(x[sz(j)]) *
                          (1.0 + 0.5 * std::min(degree, 8.0)) +
                          (std::fabs(lp.c[sz(j)]) > int_tol ? 0.25 : 0.0);
        ranked.emplace_back(score, j);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });
    if (limit > 0 && static_cast<int>(ranked.size()) > limit)
        ranked.resize(static_cast<std::size_t>(limit));
    std::vector<Index> out;
    out.reserve(ranked.size());
    for (const auto& p : ranked) out.push_back(p.second);
    return out;
}

model::LpProblem tighten_integral_rows(const model::LpProblem& in,
                                       std::uint64_t& tightened_bounds) {
    model::LpProblem out = in;
    const Index m = in.n_rows();
    const Index n = in.n_cols();
    const auto& rp = in.A.pattern.row_ptr();
    const auto& ci = in.A.pattern.col_idx();
    tightened_bounds = 0;

    for (Index i = 0; i < m; ++i) {
        bool integral_image = true;
        bool has_term = false;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = in.A.vals[sz(k)];
            if (std::fabs(a) <= 1e-12) continue;
            has_term = true;
            if (j < 0 || j >= n || in.is_integer.empty() ||
                !in.is_integer[sz(j)] ||
                std::fabs(a - std::round(a)) > 1e-12) {
                integral_image = false;
                break;
            }
        }
        if (!has_term || !integral_image) continue;

        const f64 tol_lo = 1e-9 * (1.0 + std::fabs(in.row_lo[sz(i)]));
        if (std::isfinite(in.row_lo[sz(i)])) {
            const f64 tightened = std::ceil(in.row_lo[sz(i)] - tol_lo);
            if (tightened > in.row_lo[sz(i)] + tol_lo) {
                out.row_lo[sz(i)] = tightened;
                ++tightened_bounds;
            }
        }
        const f64 tol_hi = 1e-9 * (1.0 + std::fabs(in.row_hi[sz(i)]));
        if (std::isfinite(in.row_hi[sz(i)])) {
            const f64 tightened = std::floor(in.row_hi[sz(i)] + tol_hi);
            if (tightened < in.row_hi[sz(i)] - tol_hi) {
                out.row_hi[sz(i)] = tightened;
                ++tightened_bounds;
            }
        }
    }
    return out;
}

// Add a bounded set of valid binary knapsack-cover cuts. Besides pure binary
// <= rows, an equality/<= row may contain nonnegative continuous residuals;
// dropping those residuals gives a valid binary capacity projection.
model::LpProblem add_binary_cover_cuts(const model::LpProblem& in,
                                       std::uint64_t& added_cuts) {
    model::LpProblem out = in;
    const Index m = in.n_rows(), n = in.n_cols();
    const auto& rp = in.A.pattern.row_ptr();
    const auto& ci = in.A.pattern.col_idx();
    const auto& av = in.A.vals;
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    rows.reserve(static_cast<std::size_t>(in.nnz() + 1024));
    cols.reserve(rows.capacity());
    vals.reserve(rows.capacity());
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            rows.push_back(i); cols.push_back(ci[sz(k)]); vals.push_back(av[sz(k)]);
        }
    added_cuts = 0;
    const std::uint64_t cut_limit = 256;
    for (Index i = 0; i < m && added_cuts < cut_limit; ++i) {
        if (!std::isfinite(in.row_hi[sz(i)]) || in.row_hi[sz(i)] < 0.0)
            continue;
        std::vector<std::pair<Index, f64>> terms;
        bool projectable = true;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = av[sz(k)];
            if (std::fabs(a) <= 1e-12) continue;
            if (!in.is_integer.empty() && in.is_integer[sz(j)] &&
                a > 0.0 && in.col_lo[sz(j)] >= -1e-9 &&
                in.col_hi[sz(j)] <= 1.0 + 1e-9) {
                terms.emplace_back(j, a);
                continue;
            }
            // A continuous nonnegative residual can be dropped from an upper
            // capacity row. Negative residuals or free-sign variables would
            // invalidate the projection and are rejected.
            if (a < 0.0 || in.col_lo[sz(j)] < -1e-9) {
                projectable = false;
                break;
            }
        }
        if (!projectable || terms.size() < 2) continue;
        std::sort(terms.begin(), terms.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        // Pair cover cuts catch the common conflict structure cheaply.
        for (std::size_t a = 0; a < terms.size() && added_cuts < cut_limit; ++a)
            for (std::size_t b = a + 1; b < terms.size() && added_cuts < cut_limit; ++b)
                if (terms[a].second + terms[b].second > in.row_hi[sz(i)] + 1e-9) {
                    const Index r = m + static_cast<Index>(added_cuts);
                    rows.push_back(r); cols.push_back(terms[a].first); vals.push_back(1.0);
                    rows.push_back(r); cols.push_back(terms[b].first); vals.push_back(1.0);
                    out.row_lo.push_back(-model::kInf);
                    out.row_hi.push_back(1.0);
                    ++added_cuts;
                }
        // One greedy cardinality cover per row adds a stronger, still-valid
        // inequality when the largest coefficients alone exceed capacity.
        if (added_cuts >= cut_limit) break;
        f64 accum = 0.0;
        std::vector<Index> cover;
        for (const auto& [j, a] : terms) {
            accum += a;
            cover.push_back(j);
            if (accum > in.row_hi[sz(i)] + 1e-9) {
                const Index r = m + static_cast<Index>(added_cuts);
                for (const Index q : cover) {
                    rows.push_back(r); cols.push_back(q); vals.push_back(1.0);
                }
                out.row_lo.push_back(-model::kInf);
                out.row_hi.push_back(static_cast<f64>(cover.size() - 1));
                ++added_cuts;
                break;
            }
        }
    }
    if (added_cuts == 0) return out;
    out.A = sparse::from_triplets(m + static_cast<Index>(added_cuts), n,
                                  rows, cols, vals);
    out.row_names.resize(static_cast<std::size_t>(m + added_cuts));
    for (std::uint64_t q = 0; q < added_cuts; ++q)
        out.row_names[static_cast<std::size_t>(m + q)] =
            "COVER_" + std::to_string(q);
    return out;
}

// Equality-row propagation: a continuous column with coefficient +/-1 in an
// integer equality is itself integral whenever all other terms are integral.
// The inferred domain is used only by B&B; incumbent heuristics continue to
// validate against the caller's declared model.
std::size_t infer_implied_integers(model::LpProblem& p) {
    const Index m = p.n_rows(), n = p.n_cols();
    if (p.is_integer.size() != static_cast<std::size_t>(n))
        p.is_integer.assign(static_cast<std::size_t>(n), false);
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    const auto& av = p.A.vals;
    std::size_t added = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (Index j = 0; j < n; ++j) {
            if (p.is_integer[sz(j)]) continue;
            bool appears = false, implied = true;
            for (Index i = 0; i < m && implied; ++i) {
                f64 aj = 0.0;
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                    if (ci[sz(k)] == j) aj += av[sz(k)];
                if (std::fabs(aj) <= 1e-12) continue;
                appears = true;
                if (!std::isfinite(p.row_lo[sz(i)]) ||
                    !std::isfinite(p.row_hi[sz(i)]) ||
                    std::fabs(p.row_lo[sz(i)] - p.row_hi[sz(i)]) > 1e-9 ||
                    std::fabs(std::fabs(aj) - 1.0) > 1e-9 ||
                    std::fabs(p.row_lo[sz(i)] - std::round(p.row_lo[sz(i)])) > 1e-9) {
                    implied = false;
                    break;
                }
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    const Index q = ci[sz(k)];
                    if (q == j) continue;
                    const bool fixed =
                        std::isfinite(p.col_lo[sz(q)]) &&
                        std::isfinite(p.col_hi[sz(q)]) &&
                        std::fabs(p.col_lo[sz(q)] - p.col_hi[sz(q)]) <= 1e-9 &&
                        std::fabs(p.col_lo[sz(q)] - std::round(p.col_lo[sz(q)])) <= 1e-9;
                    if ((!p.is_integer[sz(q)] && !fixed) ||
                        std::fabs(av[sz(k)] - std::round(av[sz(k)])) > 1e-9) {
                        implied = false;
                        break;
                    }
                }
            }
            if (appears && implied) {
                p.is_integer[sz(j)] = true;
                ++added;
                changed = true;
            }
        }
    }
    return added;
}

bool relaxation_proved(const core::RawResult& r,
                       const engines::SimplexDiagnostics& d,
                       const engines::SimplexOptions& opts) {
    return r.proposed_status == core::Status::Optimal &&
           d.primal_residual <= opts.primal_feas_tol &&
           d.dual_residual <= opts.dual_feas_tol &&
           d.dual_bound_finite && d.gap_rel <= opts.gap_tol;
}

}  // namespace

core::RawResult solve_milp(const model::LpProblem& problem,
                           const BabOptions& opts,
                           BabDiagnostics& diag) {
    const auto t0 = Clock::now();
    diag = BabDiagnostics{};

    core::RawResult raw;
    raw.engine = "milp_bab";
    raw.backend = "cpu";
    raw.proposed_status = core::Status::NoSolutionFound;
    raw.proposed_level = core::ProofLevel::None;

    model::LpProblem mip = problem;
    infer_implied_integers(mip);

    if (mip.n_integer() == 0) {
        // Pure LP — just call simplex. The raw result carries the LP's own
        // proposed status, but the MILP evidence machinery reads THIS diag,
        // so it must reflect what actually happened: a certified relaxation
        // of a tree with a single (root) node is exactly the "tree
        // exhausted, every LP proved" case, and without these fields
        // finalize_result would downgrade the LP's Optimal to
        // NoSolutionFound (observed on industrial blend_lp instances fed to
        // --engine milp). An uncertified LP stays honestly uncertified.
        engines::SimplexDiagnostics sd;
        raw = engines::solve_simplex(problem, opts.lp, sd, nullptr);
        diag.lp_solves = 1;
        diag.nodes = 1;
        if (relaxation_proved(raw, sd, opts.lp)) {
            diag.globally_proved = true;
            diag.incumbent = raw.objective;
            diag.dual_bound = raw.objective;
            diag.gap_rel = 0.0;
        }
        diag.total_ms = ms_since(t0);
        diag.termination_reason = "no integer columns; LP solve";
        return raw;
    }

    const Index n = mip.n_cols();
    const f64 sense = mip.maximize ? -1.0 : 1.0;
    // Strong branching is highly effective on small MIPs but can dominate the
    // solve on wide models. Disable it there; every probe remains advisory and
    // certified node LPs still control correctness.
    const bool use_reliability = opts.reliability_branching &&
        n <= 2000 && mip.nnz() <= 10000;
    std::uint64_t strong_branch_budget = use_reliability
        ? opts.strong_branch_nodes : 0;
    std::vector<Index> col_degree(sz(n), 0);
    for (const Index j : mip.A.pattern.col_idx()) {
        if (j >= 0 && j < n) ++col_degree[sz(j)];
    }

    // Work in minimize sense for bounds: lower dual bound is valid.
    // Incumbent stored as original-sense objective.
    f64 best_incumbent = mip.maximize ? -std::numeric_limits<f64>::infinity()
                                          :  std::numeric_limits<f64>::infinity();
    std::vector<f64> best_x;
    bool have_incumbent = false;

    // Directional pseudocosts store observed LP-bound improvement per unit of
    // integer movement. They are learned only from certified relaxations.
    std::vector<f64> pc_down_sum(sz(n), 0.0), pc_up_sum(sz(n), 0.0);
    std::vector<std::uint32_t> pc_down_count(sz(n), 0), pc_up_count(sz(n), 0);

    std::priority_queue<Node, std::vector<Node>, NodeCmp> open;
    // Hybrid node selection (item 16): a bounded depth-first "plunge" stack
    // that runs alongside the best-bound queue. Every node created goes into
    // exactly one of these two containers, so both the tree-exhausted check
    // and the final dual-bound drain below must account for both -- nothing
    // is dropped, this only changes visitation ORDER.
    std::vector<Node> plunge_stack;
    // Root basis seeding (set below): when the cut loop ran, its final
    // proved basis warm-starts the root node LP, which would otherwise
    // re-solve the identical relaxation cold -- measured as a full ~20-28s
    // duplicate solve on schedule_milp HUGE.
    engines::SimplexBasis cut_loop_basis;
    bool have_cut_loop_basis = false;
    std::uint64_t tightened_row_bounds = 0;
    model::LpProblem search_problem = mip;
    if (opts.integer_row_rounding)
        search_problem = tighten_integral_rows(mip, tightened_row_bounds);
    diag.integer_row_roundings = tightened_row_bounds;
    std::uint64_t cover_cuts = 0;
    search_problem = add_binary_cover_cuts(search_problem, cover_cuts);
    diag.binary_cover_cuts = cover_cuts;

    auto timed_out = [&]() {
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - t0).count() >
                   opts.time_limit_s;
    };

    // Branch-and-Cut: a root-node cutting loop (Achterberg thesis 2007
    // Ch.3-4; Gomory Mixed-Integer separator, Ch.8.2-8.3 -- see
    // sor/search/cuts.hpp). Repeatedly solve the LP, separate cuts from the
    // optimal tableau, and reoptimize until integer-feasible, diminishing
    // returns, or the round cap. Per-node/local cuts are not supported yet:
    // adding rows mid-tree would invalidate the warm-start bases the B&B
    // loop below relies on (SimplexBasis is sized to the problem and a
    // dimension mismatch silently falls back to a cold start).
    if (opts.cuts_enabled) {
        CutDiagnostics cut_diag;
        CutPool cut_pool(opts.cut);
        f64 prev_bound = core::kNaN;
        // Warm continuation across cut rounds: cuts only ADD ROWS, so the
        // previous round's basis extends naturally (each new row's logical
        // basic in that row) and the dual re-optimizes from it in a handful
        // of pivots instead of a full cold solve. Measured on
        // schedule_milp HUGE: each cold round cost ~28s; the warm rounds
        // are near-free. A failed warm solve falls back to the cold path.
        engines::SimplexBasis& prior_basis = cut_loop_basis;
        bool& have_prior = have_cut_loop_basis;
        for (int round = 0; round < opts.cut.max_rounds; ++round) {
            if (timed_out()) break;
            engines::SimplexOptions cut_lp_opts = opts.lp;
            // separate_gomory_mi() reconstructs the basis matrix from the
            // ORIGINAL row/col bounds and reads AtLower/AtUpper against
            // them. A presolved solve can lift a basis whose nonbasic
            // statuses reflect a presolve-tightened bound (e.g. a singleton
            // row folded into a column bound), which would silently
            // mismatch lp.col_lo/col_hi here. Solve unpresolved so the
            // returned SimplexBasis is exactly in this function's bound
            // space.
            cut_lp_opts.presolve = false;
            if (opts.time_limit_s > 0.0) {
                const double left = opts.time_limit_s -
                    std::chrono::duration<double>(Clock::now() - t0).count();
                if (left <= 0.02) break;
                cut_lp_opts.time_limit_s = left;
            }
            engines::SimplexDiagnostics cut_sd;
            engines::SimplexBasis cut_basis;
            core::RawResult cut_lp_raw;
            bool warm_used = false;
            if (have_prior) {
                // Row extension: ns is unchanged and old indices are stable
                // (apply_cuts appends rows only).
                engines::SimplexBasis ext = prior_basis;
                const Index ns_ = search_problem.n_cols();
                const Index m_old =
                    static_cast<Index>(prior_basis.basic.size());
                const Index m_new = search_problem.n_rows();
                if (m_new >= m_old &&
                    static_cast<Index>(ext.status.size()) == ns_ + m_old) {
                    ext.status.resize(sz(ns_ + m_new));
                    for (Index i = m_old; i < m_new; ++i) {
                        ext.basic.push_back(ns_ + i);
                        ext.status[sz(ns_ + i)] = engines::NonbasicStatus::Basic;
                    }
                    cut_lp_raw = engines::solve_dual_simplex(
                        search_problem, cut_lp_opts, cut_sd, &cut_basis, &ext);
                    ++diag.lp_solves;
                    warm_used = relaxation_proved(cut_lp_raw, cut_sd, cut_lp_opts);
                }
            }
            if (!warm_used) {
                if (have_prior) {
                    // The warm dual failed to prove; redo the round cold.
                    // (The wasted warm attempt is charged to lp_solves.)
                    cut_sd = engines::SimplexDiagnostics{};
                    cut_basis = engines::SimplexBasis{};
                }
                cut_lp_raw = engines::solve_simplex(
                    search_problem, cut_lp_opts, cut_sd, &cut_basis);
                ++diag.lp_solves;
            }
            if (!relaxation_proved(cut_lp_raw, cut_sd, cut_lp_opts)) break;
            if (round == 0) diag.root_bound_before_cuts = cut_lp_raw.objective;
            diag.root_bound_after_cuts = cut_lp_raw.objective;
            prior_basis = cut_basis;
            have_prior = true;
            (void)0;

            bool integer_ok = true;
            for (Index j = 0; j < search_problem.n_cols(); ++j) {
                if (!search_problem.is_integer.empty() &&
                    search_problem.is_integer[sz(j)] &&
                    !is_integral(cut_lp_raw.x[sz(j)], opts.int_tol)) {
                    integer_ok = false;
                    break;
                }
            }
            if (integer_ok) break;

            if (round > 0 && std::isfinite(prev_bound)) {
                const f64 gain = std::fabs(cut_lp_raw.objective - prev_bound);
                const f64 scale = 1.0 + std::fabs(prev_bound);
                if (gain / scale < opts.cut.min_progress_rel) break;
            }
            prev_bound = cut_lp_raw.objective;

            cut_pool.start_round(cut_diag);
            const auto candidates = separate_gomory_mi(
                search_problem, cut_lp_raw.x, cut_basis, opts.cut, cut_diag);
            cut_pool.add(candidates, cut_diag);
            const auto cuts = cut_pool.select_violated(cut_lp_raw.x, cut_diag);
            if (cuts.empty()) break;
            search_problem = apply_cuts(search_problem, cuts);
            diag.gmi_cuts_added += cuts.size();
            ++diag.cut_rounds;
        }
        diag.cut_pool_inserted = cut_diag.pool_inserted;
        diag.cut_pool_duplicates = cut_diag.pool_duplicates;
        diag.cut_pool_dominated = cut_diag.pool_dominated;
        diag.cut_pool_parallel_rejections = cut_diag.pool_rejected_parallel;
        diag.cut_pool_aged_out = cut_diag.pool_aged_out;
        diag.cut_pool_evicted = cut_diag.pool_evicted;
    }

    {
        Node root;
        root.col_lo = mip.col_lo;
        root.col_hi = mip.col_hi;
        root.bound = -std::numeric_limits<f64>::infinity();
        root.depth = 0;
        if (have_cut_loop_basis) {
            // Extend the cut loop's final basis to search_problem's current
            // dimensions (rows may have been added after the last solved
            // round) and seed the root with it.
            const Index ns_ = search_problem.n_cols();
            const Index m_old =
                static_cast<Index>(cut_loop_basis.basic.size());
            const Index m_new = search_problem.n_rows();
            if (static_cast<Index>(cut_loop_basis.status.size()) == ns_ + m_old &&
                m_new >= m_old) {
                cut_loop_basis.status.resize(sz(ns_ + m_new));
                for (Index i = m_old; i < m_new; ++i) {
                    cut_loop_basis.basic.push_back(ns_ + i);
                    cut_loop_basis.status[sz(ns_ + i)] =
                        engines::NonbasicStatus::Basic;
                }
                root.basis = std::move(cut_loop_basis);
                root.has_basis = true;
            }
        }
        open.push(std::move(root));
    }

    // The matrix and static model data never change during B&B. Reuse one
    // problem object and replace only its bound vectors for each node; copying
    // the complete CSR matrix at every node is pure overhead.
    model::LpProblem node_lp = std::move(search_problem);

    std::string reason = "node limit";
    bool stopped_early = false;
    bool all_lp_proven = true;
    while (!open.empty() || !plunge_stack.empty()) {
        if (diag.nodes >= opts.max_nodes) {
            reason = "node limit (" + std::to_string(opts.max_nodes) + ")";
            break;
        }
        if (timed_out()) {
            reason = "time limit";
            break;
        }

        // Node selection: continue the current plunge if its inherited bound
        // isn't meaningfully worse than the best open node's, otherwise (or
        // once one side is empty) fall back to best-bound. Best-bound is the
        // ONLY thing that prunes or proves anything below -- this choice
        // only reorders which already-valid node gets expanded next.
        bool from_plunge = false;
        if (opts.hybrid_node_selection && !plunge_stack.empty()) {
            if (open.empty()) {
                from_plunge = true;
            } else {
                const f64 ob = open.top().bound;
                const f64 pb = plunge_stack.back().bound;
                from_plunge = !std::isfinite(ob) || !std::isfinite(pb) ||
                    pb <= ob + opts.plunge_bound_slack_rel * (1.0 + std::fabs(ob));
            }
        } else if (open.empty()) {
            from_plunge = true;  // hybrid disabled mid-run or open drained early
        }

        Node node;
        if (from_plunge) {
            node = std::move(plunge_stack.back());
            plunge_stack.pop_back();
            ++diag.plunge_nodes;
        } else {
            node = open.top();
            open.pop();
        }
        ++diag.nodes;

        // Bound prune (minimize working objective = sense * original).
        if (have_incumbent) {
            const f64 inc_min = sense * best_incumbent;
            if (node.bound > inc_min + opts.gap_tol * (1.0 + std::fabs(inc_min)))
                continue;
        }

        // Domain propagation (Achterberg thesis 2007, Ch.10.4): tighten this
        // node's bounds using the row structure BEFORE paying for an LP
        // solve. node_lp's rows are shared/static across the whole tree
        // (root-level cuts included), so propagating against it is always
        // valid regardless of what node_lp.col_lo/col_hi currently hold --
        // only node.col_lo/col_hi (passed by reference) are read and
        // tightened. An empty box prunes the node for free.
        if (opts.domain_propagation) {
            const auto prop = propagate_bounds(node_lp, node.col_lo, node.col_hi,
                                               opts.primal_feas_tol,
                                               opts.propagation_max_rounds);
            diag.propagation_tightenings += prop.tightened;
            if (!prop.feasible) {
                ++diag.propagation_prunes;
                continue;
            }
        }

        node_lp.col_lo = node.col_lo;
        node_lp.col_hi = node.col_hi;
        engines::SimplexOptions lp_opts = opts.lp;
        lp_opts.verbose = false;
        // The simplex default iteration cap is sized for a single standalone
        // LP.  A warm-started B&B child can need substantially more pivots
        // after a bound change; let the wall-clock budget, rather than that
        // small per-LP cap, be the primary limiter.
        if (lp_opts.max_iterations == 0) {
            const std::uint64_t work_size =
                static_cast<std::uint64_t>(node_lp.n_rows()) +
                static_cast<std::uint64_t>(node_lp.n_cols()) +
                static_cast<std::uint64_t>(node_lp.n_rows());
            lp_opts.max_iterations = std::max<std::uint64_t>(100000,
                                                               200ull * work_size);
        }
        if (opts.time_limit_s > 0.0) {
            const double elapsed =
                std::chrono::duration<double>(Clock::now() - t0).count();
            lp_opts.time_limit_s = std::max(0.05, opts.time_limit_s - elapsed);
        }
        engines::SimplexDiagnostics sd;
        engines::SimplexBasis node_basis;
        core::RawResult lp_raw;
        if (!node.has_basis || lp_opts.method == engines::SimplexMethod::Primal) {
            // The root has no compatible warm start. Let the normal simplex
            // dispatcher use presolve and its dual/primal fallback, then lift
            // the resulting basis back to the original model indices.
            if (node.has_basis)
                lp_opts.presolve = false;
            lp_raw = engines::solve_simplex(node_lp, lp_opts, sd, &node_basis);
        } else {
            // Bound changes preserve the row/column structure, so warm-start
            // child nodes with dual simplex and the parent's basis.
            lp_opts.presolve = false;
            if (lp_opts.method == engines::SimplexMethod::Auto)
                lp_opts.method = engines::SimplexMethod::Dual;
            lp_raw = engines::solve_dual_simplex(
                node_lp, lp_opts, sd, &node_basis, &node.basis);

            // A node bound is usable only when the LP has passed all three
            // optimality checks. A feasible point or an unproved "Optimal" is
            // not a lower bound and must never drive pruning/branching.
            const bool needs_fallback = !relaxation_proved(lp_raw, sd, lp_opts) &&
                lp_raw.proposed_status != core::Status::Infeasible &&
                lp_raw.proposed_status != core::Status::InfeasibleOrUnbounded &&
                (!timed_out() || lp_raw.proposed_status != core::Status::Interrupted);
            if (needs_fallback) {
                engines::SimplexOptions fallback_opts = lp_opts;
                fallback_opts.method = engines::SimplexMethod::Primal;
                // A warm-started dual failure can be numerical rather than
                // structural. Retry the same bounded LP with a fresh primal
                // solve and the caller's presolve preference; this is still
                // accepted only after an independent certificate check.
                fallback_opts.presolve = opts.lp.presolve;
                if (opts.time_limit_s > 0.0) {
                    const double left = opts.time_limit_s -
                        std::chrono::duration<double>(Clock::now() - t0).count();
                    fallback_opts.time_limit_s = std::max(0.05, left);
                }
                engines::SimplexDiagnostics fallback_sd;
                engines::SimplexBasis fallback_basis;
                auto fallback_raw = engines::solve_simplex(
                    node_lp, fallback_opts, fallback_sd, &fallback_basis);
                ++diag.lp_fallbacks;
                if (relaxation_proved(fallback_raw, fallback_sd, fallback_opts) ||
                    fallback_raw.proposed_status == core::Status::Infeasible) {
                    lp_raw = std::move(fallback_raw);
                    sd = std::move(fallback_sd);
                    node_basis = std::move(fallback_basis);
                }
            }
        }
        ++diag.lp_solves;

        if (lp_raw.proposed_status == core::Status::Infeasible ||
            lp_raw.proposed_status == core::Status::InfeasibleOrUnbounded) {
            continue;  // prune
        }
        // An interrupted/numerically unproved relaxation is not evidence that
        // the node is infeasible.  If it did produce a checked primal point,
        // keep branching from that point with an explicitly invalid bound.
        // This preserves correctness: the node can generate incumbents, but
        // its objective is never used for pruning or a global proof.  Only an
        // LP with no usable point forces an honest early stop.
        const bool lp_point_feasible =
            static_cast<Index>(lp_raw.x.size()) == node_lp.n_cols() &&
            std::all_of(lp_raw.x.begin(), lp_raw.x.end(),
                        [](f64 v) { return std::isfinite(v); }) &&
            node_lp.max_row_violation(lp_raw.x) <= opts.primal_feas_tol &&
            node_lp.max_bound_violation(lp_raw.x) <= opts.primal_feas_tol;
        if (lp_raw.proposed_status == core::Status::Interrupted &&
            (!lp_point_feasible || timed_out())) {
            reason = timed_out() ? "time limit" : "node LP interrupted";
            stopped_early = true;
            break;
        }
        const bool node_lp_proved = relaxation_proved(lp_raw, sd, lp_opts);
        if (!node_lp_proved) {
            all_lp_proven = false;
            if (!lp_point_feasible) {
                reason = "node LP unproved";
                stopped_early = true;
                break;
            }
        }

        const f64 lp_obj = lp_raw.objective;          // original sense
        const f64 lp_obj_min = sense * lp_obj;        // minimize sense
        // Only certified LP optima are valid node lower bounds.  A feasible
        // but unproved point gets -inf so it remains searchable without ever
        // causing an unsound incumbent prune.
        node.bound = node_lp_proved && std::isfinite(lp_obj_min)
                         ? lp_obj_min
                         : -std::numeric_limits<f64>::infinity();
        if (opts.verbose && diag.nodes <= 12)
            std::printf("  [milp] node %llu lp_obj %.10e frac_branch %d\n",
                        static_cast<unsigned long long>(diag.nodes), lp_obj,
                        static_cast<int>(pick_branch_var(
                            mip, node.col_lo, node.col_hi, lp_raw.x,
                            opts.int_tol)));

        if (node.parent_branch_var >= 0 &&
            node.parent_branch_var < n &&
            std::isfinite(node.parent_bound) &&
            node.parent_branch_distance > opts.int_tol &&
            std::isfinite(lp_obj_min)) {
            const f64 gain = std::max(0.0, lp_obj_min - node.parent_bound);
            const f64 unit_gain = gain / node.parent_branch_distance;
            if (node.parent_branch_dir < 0) {
                pc_down_sum[sz(node.parent_branch_var)] += unit_gain;
                ++pc_down_count[sz(node.parent_branch_var)];
            } else if (node.parent_branch_dir > 0) {
                pc_up_sum[sz(node.parent_branch_var)] += unit_gain;
                ++pc_up_count[sz(node.parent_branch_var)];
            }
            ++diag.pseudocost_updates;
        }
        if (have_incumbent && node_lp_proved) {
            const f64 inc_min = sense * best_incumbent;
            if (lp_obj_min > inc_min + opts.gap_tol * (1.0 + std::fabs(inc_min)))
                continue;
        }

        // Integer feasibility?
        bool integer_ok = true;
        for (Index j = 0; j < n; ++j) {
            if (!mip.is_integer.empty() && mip.is_integer[sz(j)] &&
                !is_integral(lp_raw.x[sz(j)], opts.int_tol)) {
                integer_ok = false;
                break;
            }
        }

        if (integer_ok) {
            // Reject non-finite LP objectives (can appear on malformed/relaxed nodes).
            if (std::isfinite(lp_obj)) {
                ++diag.integer_feasible;
                const bool better =
                    !have_incumbent ||
                    (mip.maximize ? (lp_obj > best_incumbent)
                                      : (lp_obj < best_incumbent));
                if (better) {
                    have_incumbent = true;
                    best_incumbent = lp_obj;
                    best_x = lp_raw.x;
                    if (opts.verbose) {
                        std::printf("  [milp] incumbent %.10e at node %llu\n",
                                    best_incumbent,
                                    static_cast<unsigned long long>(diag.nodes));
                    }
                }
            }
            continue;  // no branch
        }

        // Rounding heuristic.
        if (opts.rounding_heuristic) {
            std::vector<f64> xh;
            bool rounded = false;
            f64 rounded_obj = mip.maximize
                ? -std::numeric_limits<f64>::infinity()
                : std::numeric_limits<f64>::infinity();
            const auto consider_round = [&](RoundMode mode) {
                std::vector<f64> candidate;
                if (!try_round(problem, lp_raw.x, opts.int_tol,
                               opts.primal_feas_tol, candidate, mode))
                    return;
                const f64 obj = mip.objective(candidate);
                if (!std::isfinite(obj)) return;
                const bool better = !rounded ||
                    (mip.maximize ? obj > rounded_obj : obj < rounded_obj);
                if (better) {
                    rounded = true;
                    rounded_obj = obj;
                    xh = std::move(candidate);
                }
            };
            // Keep the objective-aware choice, but also test nearest and ceil
            // rounding. Lower-bound demand rows frequently need ceilings,
            // while signed-cost models can prefer the nearest point.
            consider_round(RoundMode::Objective);
            consider_round(RoundMode::Nearest);
            consider_round(RoundMode::Ceil);
            // Implied integrality is used by the search, but keep a fallback
            // rounding pass on the caller's declared domains. A continuous
            // slack can be integral in every feasible solution yet appear a
            // few ulps outside an integer at an unfinished LP vertex.
            if (!rounded) {
                for (RoundMode mode : {RoundMode::Objective,
                                       RoundMode::Nearest, RoundMode::Ceil}) {
                    std::vector<f64> candidate;
                    if (!try_round(problem, lp_raw.x, opts.int_tol,
                                   opts.primal_feas_tol, candidate, mode))
                        continue;
                    const f64 obj = problem.objective(candidate);
                    if (std::isfinite(obj) &&
                        (!rounded || (problem.maximize ? obj > rounded_obj
                                                        : obj < rounded_obj))) {
                        rounded = true;
                        rounded_obj = obj;
                        xh = std::move(candidate);
                    }
                }
            }
            if (diag.nodes == 1 && mip.n_cols() <= 1000 &&
                mip.nnz() <= 10000) {
                std::vector<f64> perturbed;
                if (try_perturbed_rounding(problem, opts.lp, opts.int_tol,
                                           opts.primal_feas_tol, 12, 0.8,
                                           perturbed)) {
                    const f64 pobj = mip.objective(perturbed);
                    if (std::isfinite(pobj) &&
                        (!rounded || (mip.maximize ? pobj > rounded_obj
                                                        : pobj < rounded_obj))) {
                        rounded = true;
                        rounded_obj = pobj;
                        xh = std::move(perturbed);
                    }
                }
            }
            if (diag.nodes == 1 && mip.n_cols() <= 3000 &&
                mip.nnz() <= 20000) {
                double enum_budget = 1.0;
                if (opts.time_limit_s > 0.0)
                    enum_budget = std::min(enum_budget, std::max(0.0,
                        opts.time_limit_s -
                        std::chrono::duration<double>(Clock::now() - t0).count()));
                std::vector<f64> enumerated;
                if (enum_budget > 0.02 &&
                    try_fractional_enumeration(
                        problem, lp_raw.x, opts.int_tol,
                        opts.primal_feas_tol, 65536, enum_budget,
                        opts.lp_rounding_repair_max_iterations, 0.01,
                        enumerated)) {
                    const f64 eobj = problem.objective(enumerated);
                    if (std::isfinite(eobj) &&
                        (!rounded || (mip.maximize ? eobj > rounded_obj
                                                    : eobj < rounded_obj))) {
                        rounded = true;
                        rounded_obj = eobj;
                        xh = std::move(enumerated);
                    }
                }
            }
            // LP points on highly symmetric 0/1 models can round into a
            // basin from which the local repair cannot escape. Retry from the
            // lower-bound corner; this is deterministic and often gives the
            // covering rows a much better starting point (p0201/markshare).
            if (!rounded) {
                std::vector<f64> lower(mip.n_cols(), 0.0);
                for (Index j = 0; j < mip.n_cols(); ++j) {
                    if (mip.col_lo[sz(j)] > -model::kInf)
                        lower[sz(j)] = mip.col_lo[sz(j)];
                }
                rounded = try_round(problem, lower, opts.int_tol,
                                    opts.primal_feas_tol, xh,
                                    RoundMode::Objective);
            }
            // Equality-heavy MILPs often need continuous columns to move after
            // integer rounding. Try a bounded LP repair at cold start and then
            // periodically while the tree has no incumbent.
            const bool repair_due = diag.nodes <= 8 || (diag.nodes % 256 == 0);
            if (!rounded && opts.lp_rounding_repair && repair_due) {
                ++diag.lp_repair_attempts;
                rounded = try_lp_rounding_repair(
                    problem, lp_raw.x, opts.int_tol,
                    opts.primal_feas_tol, opts.lp_rounding_repair_max_iterations,
                    opts.lp_rounding_repair_time_s, xh);
                if (rounded) ++diag.lp_repair_hits;
            }
            if (repair_due && mip.n_cols() <= 1000 &&
                mip.nnz() <= 10000) {
                ++diag.feasibility_pump_attempts;
                std::vector<f64> pumped;
                const bool pump_ok = try_feasibility_pump(
                    problem, lp_raw.x, opts.int_tol,
                    opts.primal_feas_tol, 12, 5000, 0.08, pumped);
                if (pump_ok) {
                    const f64 pobj = mip.objective(pumped);
                    if (std::isfinite(pobj) &&
                        (!rounded || (mip.maximize ? pobj > rounded_obj
                                                        : pobj < rounded_obj))) {
                        rounded = true;
                        rounded_obj = pobj;
                        xh = std::move(pumped);
                    }
                    ++diag.feasibility_pump_hits;
                }
            }
            if (diag.nodes == 1 && mip.n_cols() <= 3000 &&
                mip.nnz() <= 15000) {
                std::vector<f64> constructed;
                const int construct_restarts = mip.n_cols() > 1000 ? 48 : 96;
                const int construct_trials = mip.n_cols() > 1000 ? 24 : 48;
                if (try_randomized_construct(problem, opts.int_tol,
                                             opts.primal_feas_tol,
                                             construct_restarts, construct_trials,
                                             mip.n_cols() > 1000 ? 1.0 : 0.8,
                                             opts.lp_rounding_repair_max_iterations,
                                             opts.lp_rounding_repair_time_s,
                                             constructed)) {
                    const f64 cobj = mip.objective(constructed);
                    if (std::isfinite(cobj) &&
                        (!rounded || (mip.maximize ? cobj > rounded_obj
                                                        : cobj < rounded_obj))) {
                        rounded = true;
                        rounded_obj = cobj;
                        xh = std::move(constructed);
                    }
                }
                if (diag.nodes == 1 && problem.n_cols() <= 500 &&
                    problem.nnz() <= 5000) {
                    const double construct_budget = opts.time_limit_s > 0.0
                        ? std::min(0.8, std::max(0.0, opts.time_limit_s -
                            std::chrono::duration<double>(Clock::now() - t0).count()))
                        : 0.8;
                    std::vector<f64> covering;
                    if (construct_budget > 0.02 &&
                        try_covering_construct(problem, opts.int_tol,
                                               opts.primal_feas_tol, 32,
                                               construct_budget, covering)) {
                        const f64 cobj = problem.objective(covering);
                        if (opts.verbose)
                            std::printf("  [milp] covering heuristic %.10e\n", cobj);
                        if (std::isfinite(cobj) &&
                            (!rounded || (mip.maximize ? cobj > rounded_obj
                                                        : cobj < rounded_obj))) {
                            rounded = true;
                            rounded_obj = cobj;
                            xh = std::move(covering);
                        }
                    }
                }
                if (diag.nodes == 1 && problem.n_cols() <= 200 &&
                    problem.nnz() <= 5000) {
                    const double discrepancy_budget = opts.time_limit_s > 0.0
                        ? std::min(5.0, std::max(0.0, opts.time_limit_s -
                            std::chrono::duration<double>(Clock::now() - t0).count()))
                        : 1.8;
                    std::vector<f64> discrepancy;
                    if (discrepancy_budget > 0.02 &&
                        try_discrepancy_binary_search(
                            problem, opts.int_tol, opts.primal_feas_tol,
                            64, 100000, discrepancy_budget, discrepancy)) {
                        const f64 dobj = problem.objective(discrepancy);
                        if (opts.verbose)
                            std::printf("  [milp] discrepancy heuristic %.10e\n", dobj);
                        if (std::isfinite(dobj) &&
                            (!rounded || (mip.maximize ? dobj > rounded_obj
                                                        : dobj < rounded_obj))) {
                            rounded = true;
                            rounded_obj = dobj;
                            xh = std::move(discrepancy);
                        }
                    }
                }
                // Equality-heavy binary models (notably markshare and pk1)
                // need coordinated exchanges rather than independent flips.
                // Search their residual objective directly, then use the LP
                // repair only for the best few assignments.
                if (problem.n_rows() <= 80 && problem.n_cols() <= 200 &&
                    problem.nnz() <= 10000) {
                    // Small low-row binary models benefit disproportionately
                    // from spending time in the direct combinatorial search:
                    // its beam/DFS work is much cheaper than thousands of
                    // cold LP repairs, and it does not weaken proof logic.
                    double structured_budget =
                        (problem.n_cols() <= 100 && problem.n_rows() <= 10)
                            ? 6.0 : 1.5;
                    if (opts.time_limit_s > 0.0)
                        structured_budget = std::min(structured_budget,
                            std::max(0.0, opts.time_limit_s -
                                std::chrono::duration<double>(Clock::now() - t0).count()));
                    std::vector<f64> structured;
                    if (structured_budget > 0.02 &&
                        try_structured_binary_search(
                            problem, opts.int_tol, opts.primal_feas_tol,
                            4, 80, 32, structured_budget,
                            opts.lp_rounding_repair_max_iterations,
                            opts.lp_rounding_repair_time_s, structured)) {
                        const f64 sobj = problem.objective(structured);
                        if (std::isfinite(sobj) &&
                            (!rounded || (mip.maximize ? sobj > rounded_obj
                                                        : sobj < rounded_obj))) {
                            rounded = true;
                            rounded_obj = sobj;
                            xh = std::move(structured);
                        }
                    }
                }
                std::vector<f64> binary_search;
                if (problem.n_rows() <= 10 && try_binary_penalty_search(
                        problem, opts.int_tol, opts.primal_feas_tol,
                        24, 12000, 32, 1.2,
                        opts.lp_rounding_repair_max_iterations,
                        opts.lp_rounding_repair_time_s, binary_search)) {
                    const f64 bobj = problem.objective(binary_search);
                    if (std::isfinite(bobj) &&
                        (!rounded || (problem.maximize ? bobj > rounded_obj
                                                        : bobj < rounded_obj))) {
                        rounded = true;
                        rounded_obj = bobj;
                        xh = std::move(binary_search);
                    }
                }
            }
            // Independent rounding can select far too many structural
            // binaries. Explore a bounded Hamming neighborhood and re-solve
            // the continuous subproblem after each move (RINS/local search).
            // This is only an incumbent heuristic and is never used in a
            // node bound or pruning decision.
            if (rounded && opts.integer_neighborhood && diag.nodes == 1 &&
                mip.n_cols() <= 3000 && mip.nnz() <= 20000) {
                ++diag.integer_neighborhood_attempts;
                std::vector<f64> polished;
                const bool improved = try_integer_neighborhood(
                    problem, xh, opts.int_tol, opts.primal_feas_tol,
                    opts.integer_neighborhood_max_trials,
                    opts.integer_neighborhood_time_s,
                    opts.lp_rounding_repair_max_iterations,
                    opts.integer_neighborhood_lp_time_s, polished);
                if (improved) {
                    xh = std::move(polished);
                    ++diag.integer_neighborhood_hits;
                }
                // The helper's trial count is bounded by the configured cap;
                // expose that cap as a conservative diagnostic when enabled.
                diag.integer_neighborhood_trials +=
                    opts.integer_neighborhood_max_trials;
            }
                if (rounded) {
                    f64 hobj = problem.objective(xh);
                if (std::isfinite(hobj)) {
                    const bool better =
                        !have_incumbent ||
                        (mip.maximize ? (hobj > best_incumbent)
                                          : (hobj < best_incumbent));
                    if (better) {
                        // Run one bounded local-improvement pass as soon as
                        // the first incumbent appears.  Waiting for the root
                        // misses incumbents discovered later in the tree (for
                        // example gt2/assignment models).
                        if (opts.integer_neighborhood &&
                            diag.integer_neighborhood_attempts == 0 &&
                            mip.n_cols() <= 3000 && mip.nnz() <= 20000) {
                            ++diag.integer_neighborhood_attempts;
                            std::vector<f64> polished;
                            const bool all_integer_model =
                                static_cast<Index>(std::count(
                                    problem.is_integer.begin(),
                                    problem.is_integer.end(), true)) ==
                                problem.n_cols();
                            double left = all_integer_model
                                ? std::max(2.5, opts.integer_neighborhood_time_s)
                                : opts.integer_neighborhood_time_s;
                            if (opts.time_limit_s > 0.0) {
                                left = std::min(left, std::max(0.0,
                                    opts.time_limit_s -
                                    std::chrono::duration<double>(Clock::now() - t0).count()));
                            }
                            if (left > 0.0 && try_integer_neighborhood(
                                    problem, xh, opts.int_tol,
                                    opts.primal_feas_tol,
                                    all_integer_model
                                        ? std::max<std::uint64_t>(
                                            100000, opts.integer_neighborhood_max_trials)
                                        : opts.integer_neighborhood_max_trials,
                                    left,
                                    opts.lp_rounding_repair_max_iterations,
                                    opts.integer_neighborhood_lp_time_s,
                                    polished)) {
                                xh = std::move(polished);
                                ++diag.integer_neighborhood_hits;
                                hobj = problem.objective(xh);
                            }
                            diag.integer_neighborhood_trials +=
                                all_integer_model
                                    ? std::max<std::uint64_t>(
                                        100000, opts.integer_neighborhood_max_trials)
                                    : opts.integer_neighborhood_max_trials;
                        }
                        have_incumbent = true;
                        best_incumbent = hobj;
                        best_x = std::move(xh);
                        ++diag.heuristic_hits;
                        if (opts.verbose) {
                            std::printf("  [milp] heuristic incumbent %.10e at node %llu\n",
                                        best_incumbent,
                                        static_cast<unsigned long long>(diag.nodes));
                        }
                    }
                }
            }
        }

        // A local repair can get trapped by coupled rows even when a nearby
        // integer branch path is easy. Give small/medium models one bounded
        // LP dive from the current relaxation before committing to the global
        // best-bound tree. The dive is incumbent-only and is skipped once a
        // valid incumbent already exists.
        // Always give the bounded dive a chance at the root. A rounded
        // incumbent can be far worse than a feasible point reached by
        // following the relaxation down a few branch decisions (the classic
        // weakness of independent rounding on markshare/assignment models).
        if (opts.integer_dive && diag.nodes == 1 &&
            problem.n_cols() <= 3000 && problem.nnz() <= 15000) {
            ++diag.integer_dive_attempts;
            double dive_budget = opts.integer_dive_time_s;
            if (problem.n_cols() > 1000)
                dive_budget = std::min(dive_budget, 3.0);
            if (opts.time_limit_s > 0.0) {
                const double left = opts.time_limit_s -
                    std::chrono::duration<double>(Clock::now() - t0).count();
                dive_budget = std::min(dive_budget, std::max(0.0, left));
            }
            std::vector<f64> xd;
            std::uint64_t dive_solves = 0;
            const bool dived = dive_budget > 0.0 &&
                try_integer_dive(node_lp, lp_raw.x, node.col_lo, node.col_hi,
                                 &node_basis, opts.int_tol,
                                 opts.primal_feas_tol,
                                 problem.n_cols() > 1000
                                     ? std::min<std::uint64_t>(1024, opts.integer_dive_max_nodes)
                                     : opts.integer_dive_max_nodes,
                                 dive_budget,
                                 problem.n_cols() > 1000
                                     ? std::min(0.01, opts.integer_dive_lp_time_s)
                                     : opts.integer_dive_lp_time_s,
                                 sense, opts.lp, xd,
                                 dive_solves);
            diag.integer_dive_lp_solves += dive_solves;
            if (dived && problem.max_row_violation(xd) <= opts.primal_feas_tol &&
                problem.max_bound_violation(xd) <= opts.primal_feas_tol) {
                if (opts.integer_neighborhood &&
                    mip.n_cols() <= 3000 && mip.nnz() <= 20000) {
                    ++diag.integer_neighborhood_attempts;
                    std::vector<f64> polished;
                    double left = opts.integer_neighborhood_time_s;
                    if (opts.time_limit_s > 0.0) {
                        left = std::min(left, std::max(0.0, opts.time_limit_s -
                            std::chrono::duration<double>(Clock::now() - t0).count()));
                    }
                    if (left > 0.0 && try_integer_neighborhood(
                            problem, xd, opts.int_tol, opts.primal_feas_tol,
                            opts.integer_neighborhood_max_trials, left,
                            opts.lp_rounding_repair_max_iterations,
                            opts.integer_neighborhood_lp_time_s, polished)) {
                        xd = std::move(polished);
                        ++diag.integer_neighborhood_hits;
                    }
                    diag.integer_neighborhood_trials +=
                        opts.integer_neighborhood_max_trials;
                }
                const f64 dobj = mip.objective(xd);
                if (std::isfinite(dobj) &&
                    (!have_incumbent ||
                    (mip.maximize ? dobj > best_incumbent
                                       : dobj < best_incumbent))) {
                    have_incumbent = true;
                    best_incumbent = dobj;
                    best_x = std::move(xd);
                    ++diag.integer_dive_hits;
                    ++diag.heuristic_hits;
                    if (opts.verbose)
                        std::printf("  [milp] dive incumbent %.10e at node %llu\n",
                                    best_incumbent,
                                    static_cast<unsigned long long>(diag.nodes));
                }
            }
        }

        if (opts.integer_dive && diag.nodes == 1 &&
            problem.n_cols() <= 3000 && problem.nnz() <= 15000) {
            double rens_budget = 0.8;
            if (opts.time_limit_s > 0.0)
                rens_budget = std::min(rens_budget, std::max(0.0,
                    opts.time_limit_s - std::chrono::duration<double>(Clock::now() - t0).count()));
            if (rens_budget > 0.02) {
                ++diag.rens_attempts;
                std::vector<f64> xr;
                std::uint64_t rens_solves = 0;
                const bool found_rens = try_rens(
                    node_lp, lp_raw.x, &node_basis, opts.int_tol,
                    opts.primal_feas_tol,
                    problem.n_cols() > 1000 ? 256 : 512,
                    rens_budget,
                    problem.n_cols() > 1000 ? 0.01 : opts.integer_dive_lp_time_s,
                    opts.lp, xr, rens_solves);
                diag.rens_lp_solves += rens_solves;
                if (found_rens && problem.max_row_violation(xr) <= opts.primal_feas_tol &&
                    problem.max_bound_violation(xr) <= opts.primal_feas_tol) {
                    const f64 robj = problem.objective(xr);
                    if (std::isfinite(robj) &&
                        (!have_incumbent ||
                         (mip.maximize ? robj > best_incumbent
                                       : robj < best_incumbent))) {
                        have_incumbent = true;
                        best_incumbent = robj;
                        best_x = std::move(xr);
                        ++diag.rens_hits;
                        ++diag.heuristic_hits;
                    }
                }
            }
        }

        // Branch. Reliability branching uses exact child-LP gains to seed
        // pseudocosts, then switches to cheap learned estimates. Probes are
        // deliberately bounded and advisory: they can improve the tree, but
        // only certified relaxations are ever used for bounds or pruning.
        Index br = -1;
        f64 best_branch_score = -std::numeric_limits<f64>::infinity();
        const auto candidates = branch_candidates(
            mip, node.col_lo, node.col_hi, lp_raw.x, col_degree, opts.int_tol,
            use_reliability ? opts.strong_branch_candidates : 0);

        const auto choose_score = [&](Index j, f64 xv) {
            const f64 down_dist = xv - std::floor(xv);
            const f64 up_dist = std::ceil(xv) - xv;
            const bool down_ready =
                pc_down_count[sz(j)] >= static_cast<std::uint32_t>(
                    std::max(1, opts.reliability_threshold));
            const bool up_ready =
                pc_up_count[sz(j)] >= static_cast<std::uint32_t>(
                    std::max(1, opts.reliability_threshold));
            const f64 down_est = down_ready
                ? (pc_down_sum[sz(j)] / pc_down_count[sz(j)]) * down_dist : 0.0;
            const f64 up_est = up_ready
                ? (pc_up_sum[sz(j)] / pc_up_count[sz(j)]) * up_dist : 0.0;
            if (down_ready && up_ready)
                return std::min(down_est, up_est) + 0.1 * std::max(down_est, up_est);
            // Fractionality remains the safe cold-start ordering. Existing
            // one-sided pseudocost information is only a deterministic tie
            // breaker until both directions are reliable.
            return frac_score(xv) + 1e-9 * (down_est + up_est);
        };

        for (const Index j : candidates) {
            const f64 xv = lp_raw.x[sz(j)];
            const f64 score = choose_score(j, xv);
            if (score > best_branch_score) {
                best_branch_score = score;
                br = j;
            }
        }
        if (br < 0)
            br = pick_branch_var(mip, node.col_lo, node.col_hi,
                                 lp_raw.x, opts.int_tol);

        if (use_reliability && node_lp_proved &&
            diag.strong_branch_solves < strong_branch_budget &&
            !candidates.empty() && !node_basis.basic.empty()) {
            f64 strong_best = -std::numeric_limits<f64>::infinity();
            Index strong_var = -1;
            const engines::SimplexBasis* warm = &node_basis;

            for (const Index j : candidates) {
                const f64 xv = lp_raw.x[sz(j)];
                const f64 floor_v = std::floor(xv);
                const f64 ceil_v = std::ceil(xv);
                const f64 down_dist = xv - floor_v;
                const f64 up_dist = ceil_v - xv;
                if (down_dist <= opts.int_tol || up_dist <= opts.int_tol)
                    continue;

                f64 down_gain = std::numeric_limits<f64>::quiet_NaN();
                f64 up_gain = std::numeric_limits<f64>::quiet_NaN();
                bool down_ok = false, up_ok = false;
                for (int dir : {-1, 1}) {
                    if (diag.strong_branch_solves >= strong_branch_budget)
                        break;
                    node_lp.col_lo = node.col_lo;
                    node_lp.col_hi = node.col_hi;
                    if (dir < 0)
                        node_lp.col_hi[sz(j)] = std::min(node_lp.col_hi[sz(j)], floor_v);
                    else
                        node_lp.col_lo[sz(j)] = std::max(node_lp.col_lo[sz(j)], ceil_v);

                    engines::SimplexOptions probe_opts = lp_opts;
                    probe_opts.presolve = false;
                    probe_opts.method = engines::SimplexMethod::Dual;
                    probe_opts.max_iterations = std::min<std::uint64_t>(
                        probe_opts.max_iterations, 5000);
                    probe_opts.time_limit_s = opts.strong_branch_time_s;
                    if (opts.time_limit_s > 0.0) {
                        const double left = opts.time_limit_s -
                            std::chrono::duration<double>(Clock::now() - t0).count();
                        if (left <= 0.005) break;
                        probe_opts.time_limit_s = std::min(probe_opts.time_limit_s,
                                                           std::max(0.005, left));
                    }
                    engines::SimplexDiagnostics probe_diag;
                    engines::SimplexBasis probe_basis;
                    const auto probe_raw = engines::solve_dual_simplex(
                        node_lp, probe_opts, probe_diag, &probe_basis, warm);
                    ++diag.strong_branch_solves;
                    const bool proved = relaxation_proved(probe_raw, probe_diag,
                                                          probe_opts);
                    const bool infeasible =
                        probe_raw.proposed_status == core::Status::Infeasible;
                    if (dir < 0) {
                        down_ok = proved || infeasible;
                        if (proved)
                            down_gain = std::max(0.0,
                                sense * probe_raw.objective - lp_obj_min);
                        else if (infeasible)
                            down_gain = std::numeric_limits<f64>::infinity();
                    } else {
                        up_ok = proved || infeasible;
                        if (proved)
                            up_gain = std::max(0.0,
                                sense * probe_raw.objective - lp_obj_min);
                        else if (infeasible)
                            up_gain = std::numeric_limits<f64>::infinity();
                    }
                    if (proved) {
                        const f64 unit_gain = std::max(0.0,
                            (dir < 0 ? down_gain : up_gain) /
                            (dir < 0 ? down_dist : up_dist));
                        if (dir < 0) {
                            pc_down_sum[sz(j)] += unit_gain;
                            ++pc_down_count[sz(j)];
                        } else {
                            pc_up_sum[sz(j)] += unit_gain;
                            ++pc_up_count[sz(j)];
                        }
                        ++diag.pseudocost_updates;
                    }
                }
                if (down_ok && up_ok) {
                    const f64 score = std::min(down_gain, up_gain) +
                                      0.1 * std::max(down_gain, up_gain);
                    if (score > strong_best) {
                        strong_best = score;
                        strong_var = j;
                    }
                }
            }
            node_lp.col_lo = node.col_lo;
            node_lp.col_hi = node.col_hi;
            if (strong_var >= 0)
                br = strong_var;
        }

        if (br < 0) continue;

        const f64 xv = lp_raw.x[sz(br)];
        const f64 floor_v = std::floor(xv);
        const f64 ceil_v = std::ceil(xv);

        Node down = node;
        down.col_hi[sz(br)] = std::min(down.col_hi[sz(br)], floor_v);
        down.bound = node.bound;
        down.depth = node.depth + 1;
        down.basis = node_basis;
        down.has_basis = !node_basis.basic.empty();
        down.parent_branch_var = br;
        down.parent_branch_dir = -1;
        down.parent_bound = node_lp_proved && std::isfinite(lp_obj_min)
                                ? lp_obj_min : core::kNaN;
        down.parent_branch_distance = xv - floor_v;

        Node up = node;
        up.col_lo[sz(br)] = std::max(up.col_lo[sz(br)], ceil_v);
        up.bound = node.bound;
        up.depth = node.depth + 1;
        up.basis = node_basis;
        up.has_basis = !node_basis.basic.empty();
        up.parent_branch_var = br;
        up.parent_branch_dir = +1;
        up.parent_bound = node_lp_proved && std::isfinite(lp_obj_min)
                              ? lp_obj_min : core::kNaN;
        up.parent_branch_distance = ceil_v - xv;

        // Best-bound remains the proof ordering; integer_up_bias only picks
        // which child is the more promising integer direction on degenerate
        // zero-cost faces. Hybrid node selection (item 16) sends that
        // preferred child straight to the plunge stack (continuing a bounded
        // depth-first dive) and the other child to the best-bound queue as a
        // fallback -- both still get visited eventually either way.
        const bool down_first = integer_up_bias(mip, br) < 0.0;
        Node& preferred = down_first ? down : up;
        Node& fallback = down_first ? up : down;
        const bool preferred_range =
            preferred.col_lo[sz(br)] <= preferred.col_hi[sz(br)] + 1e-12;
        const bool fallback_range =
            fallback.col_lo[sz(br)] <= fallback.col_hi[sz(br)] + 1e-12;

        const bool plunge_this = opts.hybrid_node_selection &&
                                 node.plunge_len < opts.plunge_max_depth &&
                                 preferred_range;
        if (plunge_this) {
            preferred.plunge_len = node.plunge_len + 1;
            plunge_stack.push_back(std::move(preferred));
            if (fallback_range) {
                fallback.plunge_len = 0;
                open.push(std::move(fallback));
            }
        } else {
            if (preferred_range) {
                preferred.plunge_len = 0;
                open.push(std::move(preferred));
            }
            if (fallback_range) {
                fallback.plunge_len = 0;
                open.push(std::move(fallback));
            }
        }
    }

    const bool tree_exhausted = !stopped_early && open.empty() && plunge_stack.empty();
    if (tree_exhausted && have_incumbent)
        reason = "tree exhausted";
    else if (tree_exhausted && !have_incumbent)
        reason = "tree exhausted with no integer feasible point";

    // Dual bound: for a complete tree with incumbent, dual = incumbent.
    // Otherwise, for minimize, take the min LP bound among remaining open nodes
    // (and consider proved if gap small).
    f64 dual_bound_min = std::numeric_limits<f64>::infinity();
    if (have_incumbent)
        dual_bound_min = std::min(dual_bound_min, sense * best_incumbent);
    // Drain open and the plunge stack for the actual global bound
    // (destructive OK at end). Nodes retain their parent's proved LP bound
    // until they are processed, and every node lives in exactly one of these
    // two containers, so both must be drained or the bound would be
    // unsoundly optimistic.
    while (!open.empty()) {
        dual_bound_min = std::min(dual_bound_min, open.top().bound);
        open.pop();
    }
    while (!plunge_stack.empty()) {
        dual_bound_min = std::min(dual_bound_min, plunge_stack.back().bound);
        plunge_stack.pop_back();
    }
    if (!have_incumbent && !std::isfinite(dual_bound_min))
        dual_bound_min = std::numeric_limits<f64>::quiet_NaN();

    const f64 dual_orig = std::isfinite(dual_bound_min) ? sense * dual_bound_min
                                                        : core::kNaN;

    diag.incumbent = have_incumbent ? best_incumbent : core::kNaN;
    diag.dual_bound = dual_orig;
    if (have_incumbent && std::isfinite(dual_orig)) {
        diag.gap_rel = std::fabs(best_incumbent - dual_orig) /
                       (1.0 + std::fabs(best_incumbent));
    }
    diag.total_ms = ms_since(t0);
    diag.termination_reason = reason;

    // A node's .bound is only ever a certified LP bound (see the assignment
    // above: unproven nodes get -infinity, never a value that could look
    // artificially tight), so dual_bound_min -- drained from EVERY remaining
    // open/plunge node -- is a sound global lower bound regardless of
    // whether the tree was formally exhausted. If it already closes the gap
    // to within opts.gap_tol, that IS a complete proof of optimality; the
    // whole point of a gap tolerance is to allow exactly this early exit,
    // not to require full exhaustion anyway. all_lp_proven is still required
    // for BOTH paths, conservatively: it is not strictly needed for the
    // gap-based path's own soundness, but there's no case in these MIPLIB
    // runs where relaxing it mattered, and keeping it costs nothing while
    // avoiding a subtler argument about which parts of the tree an unproven
    // node's failure could have contaminated.
    const bool gap_proved = have_incumbent && std::isfinite(dual_orig) &&
                            diag.gap_rel <= opts.gap_tol;
    diag.globally_proved = all_lp_proven && (tree_exhausted || gap_proved);

    raw.iterations = diag.nodes;
    raw.termination_reason = reason;
    raw.dual_bound = dual_orig;

    if (have_incumbent) {
        raw.x = std::move(best_x);
        raw.objective = best_incumbent;
        if (diag.globally_proved) {
            raw.proposed_status = core::Status::Optimal;
            raw.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
        } else {
            raw.proposed_status = core::Status::Feasible;
            raw.proposed_level = core::ProofLevel::FeasibleWithGap;
        }
    } else if (tree_exhausted) {
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::BoundOnly;
    } else {
        raw.proposed_status = core::Status::Interrupted;
        raw.proposed_level = core::ProofLevel::None;
    }

    return raw;
}

core::ProofEvidence milp_evidence(const BabDiagnostics& diag,
                                  const BabOptions& opts) {
    core::ProofEvidence ev;
    ev.has_basis = false;
    ev.max_primal_violation = 0.0;
    // Read diag.globally_proved directly (set once, in solve_milp()) rather
    // than re-deriving it here from termination_reason -- a STRING match on
    // "tree exhausted" duplicated the proof condition in two places and
    // silently required full tree exhaustion even when the gap had already
    // closed to within opts.gap_tol, which is a complete proof on its own.
    const bool globally_proved = diag.globally_proved;
    ev.max_dual_violation = globally_proved ? 0.0 : core::kPosInf;
    ev.gap_rel = diag.gap_rel;
    ev.primal_feas_tol = opts.primal_feas_tol;
    ev.dual_feas_tol = opts.primal_feas_tol;
    ev.gap_tol = opts.gap_tol;
    ev.checker_passed = std::isfinite(diag.incumbent);
    if (globally_proved)
        ev.claimed_level = core::ProofLevel::ProvedGlobalEpsilon;
    else if (std::isfinite(diag.incumbent))
        ev.claimed_level = core::ProofLevel::FeasibleWithGap;
    else
        ev.claimed_level = core::ProofLevel::None;
    return ev;
}

}  // namespace sor::search
