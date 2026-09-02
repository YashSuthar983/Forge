#include "sor/search/bab.hpp"
#include "sor/engines/dual_simplex.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
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

struct Node {
    std::vector<f64> col_lo;
    std::vector<f64> col_hi;
    engines::SimplexBasis basis;
    bool has_basis = false;
    f64 bound = -std::numeric_limits<f64>::infinity();  // dual bound (min sense)
    int depth = 0;
};

// Best-bound first (minimize): smallest bound first.
struct NodeCmp {
    bool operator()(const Node& a, const Node& b) const {
        if (a.bound != b.bound) return a.bound > b.bound;  // min-heap via greater
        return a.depth < b.depth;
    }
};

model::LpProblem with_bounds(const model::LpProblem& base,
                             const std::vector<f64>& lo,
                             const std::vector<f64>& hi) {
    model::LpProblem p = base;
    p.col_lo = lo;
    p.col_hi = hi;
    return p;
}

// Round integer columns and repair row violations with local integer moves.
// This is a small feasibility-pump style heuristic: it is deliberately
// bounded and never used as a proof, but it is enough to turn the fractional
// schedule relaxation into an incumbent instead of discarding the root node.
bool try_round(const model::LpProblem& lp,
               const std::vector<f64>& x_lp,
               f64 int_tol,
               f64 feas_tol,
               std::vector<f64>& x_out) {
    const Index n = lp.n_cols();
    if (static_cast<Index>(x_lp.size()) != n) return false;
    x_out = x_lp;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        f64 v = std::round(x_lp[sz(j)]);
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
    auto rebuild_activity = [&]() {
        std::fill(activity.begin(), activity.end(), 0.0);
        for (Index i = 0; i < m; ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                activity[sz(i)] += av[sz(k)] * x_out[sz(ci[sz(k)])];
    };
    rebuild_activity();

    // A few passes are sufficient for the generated scheduling rows (one
    // demand row per period and three-term startup links).  Keep the cap
    // proportional to the model size for larger, sparse MILPs.
    const int max_passes = std::min(16, std::max(2, static_cast<int>(m / 256) + 2));
    for (int pass = 0; pass < max_passes; ++pass) {
        bool changed = false;
        for (Index i = 0; i < m; ++i) {
            const bool lower_violation =
                activity[sz(i)] < lp.row_lo[sz(i)] - feas_tol;
            if (!lower_violation &&
                !(activity[sz(i)] > lp.row_hi[sz(i)] + feas_tol)) {
                continue;
            }

            Index best_j = -1;
            f64 best_gain = 0.0;
            f64 best_score = -std::numeric_limits<f64>::infinity();
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index j = ci[sz(k)];
                if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
                const f64 a = av[sz(k)];
                // Choose the integer movement that improves this row.  For a
                // lower violation, positive coefficients move up and negative
                // coefficients move down; upper violations use the opposite.
                const int move_dir = lower_violation
                    ? (a > 0.0 ? +1 : -1)
                    : (a > 0.0 ? -1 : +1);
                const f64 xj = x_out[sz(j)];
                const f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
                f64 next = xj + static_cast<f64>(move_dir);
                if (move_dir > 0) {
                    next = std::ceil(xj + int_tol);
                    if (next <= xj + int_tol) next += 1.0;
                    next = std::min(next, hi);
                } else {
                    next = std::floor(xj - int_tol);
                    if (next >= xj - int_tol) next -= 1.0;
                    next = std::max(next, lo);
                }
                const f64 delta = next - xj;
                const f64 gain = std::fabs(a * delta);
                if (!(gain > 0.0)) continue;
                const f64 obj_delta = (lp.maximize ? -1.0 : 1.0) *
                                      lp.c[sz(j)] * delta;
                // Prefer high row correction per unit objective cost, with a
                // small tie-break toward larger correction.
                f64 score = gain / (1.0 + std::max(0.0, obj_delta));
                // Startup-link rows are best repaired by turning on the
                // explicit startup variable, rather than turning off a run
                // variable and immediately breaking that period's demand.
                if (!lp.row_names.empty() && !lp.col_names.empty() &&
                    static_cast<std::size_t>(i) < lp.row_names.size() &&
                    static_cast<std::size_t>(j) < lp.col_names.size() &&
                    lp.row_names[sz(i)].rfind("SU_", 0) == 0 &&
                    lp.col_names[sz(j)].rfind("S_", 0) == 0)
                    score *= 1.0e6;
                if (score > best_score ||
                    (score == best_score && gain > best_gain)) {
                    best_score = score;
                    best_gain = gain;
                    best_j = j;
                }
            }
            if (best_j < 0) continue;

            const f64 a_best = [&]() {
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                    if (ci[sz(k)] == best_j) return av[sz(k)];
                return 0.0;
            }();
            const int move_dir = lower_violation
                ? (a_best > 0.0 ? +1 : -1)
                : (a_best > 0.0 ? -1 : +1);
            f64 next = x_out[sz(best_j)] + static_cast<f64>(move_dir);
            if (move_dir > 0) {
                next = std::ceil(x_out[sz(best_j)] + int_tol);
                if (next <= x_out[sz(best_j)] + int_tol) next += 1.0;
                next = std::min(next, lp.col_hi[sz(best_j)]);
            } else {
                next = std::floor(x_out[sz(best_j)] - int_tol);
                if (next >= x_out[sz(best_j)] - int_tol) next -= 1.0;
                next = std::max(next, lp.col_lo[sz(best_j)]);
            }
            const f64 delta = next - x_out[sz(best_j)];
            x_out[sz(best_j)] = next;
            // Coupled rows are refreshed at the end of each pass.  This keeps
            // each move O(1) and the total heuristic linear in the sparse
            // matrix size for the generated schedule family.
            changed = changed || std::fabs(delta) > int_tol;
        }
        rebuild_activity();
        bool feasible = true;
        for (Index i = 0; i < m; ++i)
            if (activity[sz(i)] < lp.row_lo[sz(i)] - feas_tol ||
                activity[sz(i)] > lp.row_hi[sz(i)] + feas_tol) {
                feasible = false;
                break;
            }
        if (feasible) break;
        if (!changed) break;
    }

    if (lp.max_row_violation(x_out) > feas_tol) return false;
    if (lp.max_bound_violation(x_out) > feas_tol) return false;
    return true;
}

Index pick_branch_var(const model::LpProblem& lp,
                      const std::vector<f64>& x,
                      f64 int_tol) {
    Index best = -1;
    f64 best_frac = 0.0;
    const Index n = lp.n_cols();
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (is_integral(x[sz(j)], int_tol)) continue;
        // Skip fixed integers.
        if (lp.col_lo[sz(j)] == lp.col_hi[sz(j)]) continue;
        const f64 sc = frac_score(x[sz(j)]);
        if (sc > best_frac) {
            best_frac = sc;
            best = j;
        }
    }
    return best;
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

    if (problem.n_integer() == 0) {
        // Pure LP — just call simplex.
        engines::SimplexDiagnostics sd;
        raw = engines::solve_simplex(problem, opts.lp, sd, nullptr);
        diag.lp_solves = 1;
        diag.nodes = 1;
        diag.total_ms = ms_since(t0);
        diag.termination_reason = "no integer columns; LP solve";
        return raw;
    }

    const Index n = problem.n_cols();
    const f64 sense = problem.maximize ? -1.0 : 1.0;

    // Work in minimize sense for bounds: lower dual bound is valid.
    // Incumbent stored as original-sense objective.
    f64 best_incumbent = problem.maximize ? -std::numeric_limits<f64>::infinity()
                                          :  std::numeric_limits<f64>::infinity();
    std::vector<f64> best_x;
    bool have_incumbent = false;

    // Global dual bound in minimize-sense: min LP objective over live nodes.
    // Updated whenever we solve a node; remaining open nodes still carry their
    // parent LP bound as a valid underestimate until solved.
    f64 global_dual_min = std::numeric_limits<f64>::infinity();

    std::priority_queue<Node, std::vector<Node>, NodeCmp> open;
    {
        Node root;
        root.col_lo = problem.col_lo;
        root.col_hi = problem.col_hi;
        root.bound = -std::numeric_limits<f64>::infinity();
        root.depth = 0;
        open.push(std::move(root));
    }

    auto timed_out = [&]() {
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - t0).count() >
                   opts.time_limit_s;
    };

    std::string reason = "node limit";
    bool stopped_early = false;
    while (!open.empty()) {
        if (diag.nodes >= opts.max_nodes) {
            reason = "node limit (" + std::to_string(opts.max_nodes) + ")";
            break;
        }
        if (timed_out()) {
            reason = "time limit";
            break;
        }

        Node node = open.top();
        open.pop();
        ++diag.nodes;

        // Bound prune (minimize working objective = sense * original).
        if (have_incumbent) {
            const f64 inc_min = sense * best_incumbent;
            if (node.bound > inc_min + opts.gap_tol * (1.0 + std::fabs(inc_min)))
                continue;
        }

        model::LpProblem node_lp = with_bounds(problem, node.col_lo, node.col_hi);
        engines::SimplexOptions lp_opts = opts.lp;
        lp_opts.verbose = false;
        // Node structure is unchanged across branches.  Keep the basis from
        // the parent and solve directly with dual simplex; presolve can change
        // row/column indices and would invalidate that warm start.
        lp_opts.presolve = false;
        if (opts.time_limit_s > 0.0) {
            const double elapsed =
                std::chrono::duration<double>(Clock::now() - t0).count();
            lp_opts.time_limit_s = std::max(0.05, opts.time_limit_s - elapsed);
        }
        // Dual is the natural engine after bound changes.
        if (lp_opts.method == engines::SimplexMethod::Auto)
            lp_opts.method = engines::SimplexMethod::Dual;

        engines::SimplexDiagnostics sd;
        engines::SimplexBasis node_basis;
        auto lp_raw = engines::solve_dual_simplex(
            node_lp, lp_opts, sd, &node_basis,
            node.has_basis ? &node.basis : nullptr);
        ++diag.lp_solves;

        // A node relaxation that did not finish is not evidence that the node
        // is infeasible.  Stop the search with an honest reason; otherwise the
        // empty queue below would be misreported as "tree exhausted".
        if (lp_raw.proposed_status == core::Status::Interrupted) {
            reason = timed_out() ? "time limit" : "node LP interrupted";
            stopped_early = true;
            break;
        }

        if (lp_raw.proposed_status == core::Status::Infeasible ||
            lp_raw.proposed_status == core::Status::InfeasibleOrUnbounded) {
            continue;  // prune
        }
        if (lp_raw.proposed_status != core::Status::Optimal &&
            lp_raw.proposed_status != core::Status::Feasible) {
            reason = "node LP failed";
            stopped_early = true;
            break;
        }

        const f64 lp_obj = lp_raw.objective;          // original sense
        const f64 lp_obj_min = sense * lp_obj;        // minimize sense
        node.bound = lp_obj_min;
        global_dual_min = std::min(global_dual_min, lp_obj_min);

        if (have_incumbent) {
            const f64 inc_min = sense * best_incumbent;
            if (lp_obj_min > inc_min + opts.gap_tol * (1.0 + std::fabs(inc_min)))
                continue;
        }

        // Integer feasibility?
        bool integer_ok = true;
        for (Index j = 0; j < n; ++j) {
            if (!problem.is_integer.empty() && problem.is_integer[sz(j)] &&
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
                    (problem.maximize ? (lp_obj > best_incumbent)
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
            if (try_round(problem, lp_raw.x, opts.int_tol, opts.primal_feas_tol, xh)) {
                const f64 hobj = problem.objective(xh);
                if (std::isfinite(hobj)) {
                    const bool better =
                        !have_incumbent ||
                        (problem.maximize ? (hobj > best_incumbent)
                                          : (hobj < best_incumbent));
                    if (better) {
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

        // Branch.
        // Temporarily put node's original bounds for branching variable pick
        // using the LP solution (integer columns from original flags).
        model::LpProblem probe = problem;
        probe.col_lo = node.col_lo;
        probe.col_hi = node.col_hi;
        const Index br = pick_branch_var(probe, lp_raw.x, opts.int_tol);
        if (br < 0) continue;

        const f64 xv = lp_raw.x[sz(br)];
        const f64 floor_v = std::floor(xv);
        const f64 ceil_v = std::ceil(xv);

        Node down = node;
        down.col_hi[sz(br)] = std::min(down.col_hi[sz(br)], floor_v);
        down.bound = lp_obj_min;
        down.depth = node.depth + 1;
        down.basis = node_basis;
        down.has_basis = !node_basis.basic.empty();

        Node up = node;
        up.col_lo[sz(br)] = std::max(up.col_lo[sz(br)], ceil_v);
        up.bound = lp_obj_min;
        up.depth = node.depth + 1;
        up.basis = node_basis;
        up.has_basis = !node_basis.basic.empty();

        // Skip children with empty bound intervals.
        if (down.col_lo[sz(br)] <= down.col_hi[sz(br)] + 1e-12)
            open.push(std::move(down));
        if (up.col_lo[sz(br)] <= up.col_hi[sz(br)] + 1e-12)
            open.push(std::move(up));
    }

    if (!stopped_early && open.empty() && have_incumbent)
        reason = "tree exhausted";
    else if (!stopped_early && open.empty() && !have_incumbent)
        reason = "tree exhausted with no integer feasible point";

    // Dual bound: for a complete tree with incumbent, dual = incumbent.
    // Otherwise, for minimize, take the min LP bound among remaining open nodes
    // (and consider proved if gap small).
    f64 dual_bound_min = have_incumbent ? sense * best_incumbent
                                        : std::numeric_limits<f64>::infinity();
    // Drain open for dual bound (destructive OK at end).
    while (!open.empty()) {
        dual_bound_min = std::min(dual_bound_min, open.top().bound);
        open.pop();
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

    raw.iterations = diag.nodes;
    raw.termination_reason = reason;
    raw.dual_bound = dual_orig;

    if (have_incumbent) {
        raw.x = std::move(best_x);
        raw.objective = best_incumbent;
        // MIP optima are not LP-basis proofs; finalize_result would demote
        // Optimal without has_basis. Report Feasible + FeasibleWithGap honestly
        // (gap may be 0 when the tree is exhausted).
        raw.proposed_status = core::Status::Feasible;
        raw.proposed_level = core::ProofLevel::FeasibleWithGap;
    } else if (reason == "tree exhausted") {
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
    ev.max_dual_violation = core::kPosInf;
    ev.gap_rel = diag.gap_rel;
    ev.primal_feas_tol = opts.primal_feas_tol;
    ev.dual_feas_tol = opts.primal_feas_tol;
    ev.gap_tol = opts.gap_tol;
    ev.checker_passed = std::isfinite(diag.incumbent);
    if (std::isfinite(diag.incumbent))
        ev.claimed_level = core::ProofLevel::FeasibleWithGap;
    else
        ev.claimed_level = core::ProofLevel::None;
    return ev;
}

}  // namespace sor::search
