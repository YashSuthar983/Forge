#include "sor/search/lns.hpp"

#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

inline bool integral_col(const model::LpProblem& lp, Index j) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)];
}

inline bool is_int_value(f64 v, f64 tol) {
    return std::fabs(v - std::round(v)) <= tol;
}

// A 0/1 column under the box being restricted. Local branching and crossover
// are only defined over binaries; a general integer has no "flip".
inline bool binary_col(const model::LpProblem& lp, const std::vector<f64>& lo,
                       const std::vector<f64>& hi, Index j) {
    return integral_col(lp, j) && lo[sz(j)] > -1e-9 && lo[sz(j)] < 1e-9 &&
           hi[sz(j)] > 1.0 - 1e-9 && hi[sz(j)] < 1.0 + 1e-9;
}

// xorshift32: a few bits of determinism without dragging in <random> state that
// would have to be threaded through every call.
inline std::uint32_t next_rand(std::uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
inline double rand_unit(std::uint32_t& s) {
    return static_cast<double>(next_rand(s) >> 8) / 16777216.0;
}

}  // namespace

const char* to_string(Neighborhood n) noexcept {
    switch (n) {
        case Neighborhood::Rens: return "rens";
        case Neighborhood::Rins: return "rins";
        case Neighborhood::Mutation: return "mutation";
        case Neighborhood::Crossover: return "crossover";
        case Neighborhood::LocalBranching: return "localbranch";
        case Neighborhood::Proximity: return "proximity";
    }
    return "?";
}

// ------------------------------------------------------------- scheduler ---

AlnsScheduler::AlnsScheduler(const LnsOptions& opts) : opts_(opts) {
    for (const Neighborhood k :
         {Neighborhood::Rens, Neighborhood::Rins, Neighborhood::Mutation,
          Neighborhood::Crossover, Neighborhood::LocalBranching,
          Neighborhood::Proximity}) {
        LnsArmStats a;
        a.kind = k;
        a.fixing_rate = opts.fixing_rate_init;
        a.node_budget = opts.call_nodes;
        arms_.push_back(a);
    }
}

int AlnsScheduler::select(const std::vector<bool>& usable) {
    // UCB1: mean reward plus a confidence term that decays as an arm is tried.
    // An arm never tried has infinite priority, so the first round is a sweep
    // over whatever is currently usable -- which is the right opening move when
    // nothing is known about the instance.
    int best = -1;
    f64 best_score = -std::numeric_limits<f64>::infinity();
    const f64 log_n = std::log(static_cast<f64>(std::max<std::uint64_t>(total_calls_, 1)));
    for (std::size_t a = 0; a < arms_.size(); ++a) {
        if (a < usable.size() && !usable[a]) continue;
        const auto& arm = arms_[a];
        f64 score;
        if (arm.calls == 0) {
            score = std::numeric_limits<f64>::infinity();
        } else {
            const f64 mean = arm.reward_sum / static_cast<f64>(arm.calls);
            score = mean + opts_.ucb_alpha *
                               std::sqrt(log_n / static_cast<f64>(arm.calls));
        }
        if (score > best_score) {
            best_score = score;
            best = static_cast<int>(a);
        }
    }
    return best;
}

void AlnsScheduler::reward(int arm, LnsOutcome outcome, double seconds) {
    if (arm < 0 || arm >= static_cast<int>(arms_.size())) return;
    auto& a = arms_[sz(static_cast<Index>(arm))];
    // A neighborhood that could not be built cost nothing and taught nothing;
    // scoring it would punish an arm for the state of the search rather than
    // for its own behaviour.
    if (outcome == LnsOutcome::NotBuilt) return;

    ++a.calls;
    ++total_calls_;
    a.seconds += seconds;

    f64 r = opts_.reward_nothing;
    switch (outcome) {
        case LnsOutcome::NewBest:   r = opts_.reward_new_best; ++a.hits; break;
        case LnsOutcome::Feasible:  r = opts_.reward_feasible; break;
        case LnsOutcome::Exhausted: r = opts_.reward_exhausted; break;
        default: break;
    }
    // Effort discount: two arms that both found a new best are not equally
    // good if one took ten times as long. Normalising against the per-call
    // budget keeps the discount in [0.5, 1] rather than letting a slow success
    // score below a fast failure.
    if (opts_.call_time_s > 0.0) {
        const f64 frac = std::min(1.0, seconds / opts_.call_time_s);
        r *= 1.0 / (1.0 + frac);
    }
    a.reward_sum += r;

    // Adaptive fixing rate: failure makes the next neighborhood smaller (fix
    // more), success makes it larger (fix less).
    const bool success = outcome == LnsOutcome::NewBest ||
                         outcome == LnsOutcome::Feasible;
    if (success) a.fixing_rate -= opts_.fixing_rate_step_down;
    else a.fixing_rate += opts_.fixing_rate_step_up;
    a.fixing_rate = std::min(std::max(a.fixing_rate, opts_.fixing_rate_min),
                             opts_.fixing_rate_max);

    // Node budget adaptation. An arm that improved the incumbent gets a bigger
    // sub-MIP next time; one that ran out of budget without finding anything
    // gets a smaller one, on the reasoning that it was probably too large to
    // settle rather than too small to be interesting. An arm that EXHAUSTED its
    // neighbourhood keeps its budget: the size was fine, the neighbourhood was
    // simply empty of improvements.
    const std::uint64_t cap = opts_.call_nodes * opts_.call_nodes_max_growth;
    if (outcome == LnsOutcome::NewBest)
        a.node_budget = std::min(cap, a.node_budget * 2);
    else if (outcome == LnsOutcome::Nothing)
        a.node_budget = std::max<std::uint64_t>(opts_.call_nodes / 4,
                                                a.node_budget / 2);
}

// ----------------------------------------------------------------- pool ----

void SolutionPool::add(const std::vector<f64>& x, f64 objective, bool maximize) {
    if (x.empty()) return;
    for (const auto& e : entries_) {
        if (e.x.size() != x.size()) continue;
        bool same = true;
        for (std::size_t k = 0; k < x.size() && same; ++k)
            if (std::fabs(e.x[k] - x[k]) > 1e-9) same = false;
        if (same) return;
    }
    entries_.push_back({x, objective});
    std::sort(entries_.begin(), entries_.end(),
              [maximize](const Entry& a, const Entry& b) {
                  return maximize ? a.obj > b.obj : a.obj < b.obj;
              });
    if (entries_.size() > cap_) entries_.resize(cap_);
}

// -------------------------------------------------------- neighborhoods ---

bool build_neighborhood(Neighborhood kind,
                        const model::LpProblem& mip,
                        const std::vector<f64>& node_lo,
                        const std::vector<f64>& node_hi,
                        const std::vector<f64>& x_relax,
                        const std::vector<f64>& x_inc,
                        const SolutionPool& pool,
                        f64 fixing_rate,
                        f64 int_tol,
                        std::uint32_t& rng_state,
                        NeighborhoodProblem& out) {
    const Index n = mip.n_cols();
    if (static_cast<Index>(node_lo.size()) != n ||
        static_cast<Index>(node_hi.size()) != n)
        return false;

    out = NeighborhoodProblem{};
    out.col_lo = node_lo;
    out.col_hi = node_hi;

    const bool have_relax = static_cast<Index>(x_relax.size()) == n;
    const bool have_inc = static_cast<Index>(x_inc.size()) == n;

    // Fixes column j at v, if v is inside the box. Returns false when it is
    // not, which means the caller's reference point is not in this box and the
    // neighborhood should be abandoned rather than silently distorted.
    auto fix_at = [&](Index j, f64 v) {
        const f64 r = std::round(v);
        if (r < out.col_lo[sz(j)] - int_tol || r > out.col_hi[sz(j)] + int_tol)
            return false;
        out.col_lo[sz(j)] = r;
        out.col_hi[sz(j)] = r;
        ++out.fixed;
        return true;
    };

    switch (kind) {
        case Neighborhood::Rens: {
            if (!have_relax) return false;
            for (Index j = 0; j < n; ++j) {
                if (!integral_col(mip, j)) continue;
                const f64 v = x_relax[sz(j)];
                if (!std::isfinite(v)) continue;
                if (is_int_value(v, int_tol)) {
                    if (!fix_at(j, v)) return false;
                } else {
                    out.col_lo[sz(j)] = std::max(out.col_lo[sz(j)], std::floor(v));
                    out.col_hi[sz(j)] = std::min(out.col_hi[sz(j)], std::ceil(v));
                    ++out.free_integer;
                }
            }
            break;
        }
        case Neighborhood::Rins: {
            if (!have_relax || !have_inc) return false;
            for (Index j = 0; j < n; ++j) {
                if (!integral_col(mip, j)) continue;
                const f64 a = x_inc[sz(j)], b = x_relax[sz(j)];
                if (!std::isfinite(a) || !std::isfinite(b)) continue;
                if (std::fabs(a - b) <= int_tol) {
                    if (!fix_at(j, a)) return false;
                } else {
                    ++out.free_integer;
                }
            }
            break;
        }
        case Neighborhood::Mutation: {
            // Fix a random share of the integer columns at the incumbent. The
            // share IS the fixing rate, so this arm is the one the adaptive
            // rate steers most directly.
            if (!have_inc) return false;
            for (Index j = 0; j < n; ++j) {
                if (!integral_col(mip, j)) continue;
                if (!std::isfinite(x_inc[sz(j)])) continue;
                if (rand_unit(rng_state) < fixing_rate) {
                    if (!fix_at(j, x_inc[sz(j)])) return false;
                } else {
                    ++out.free_integer;
                }
            }
            break;
        }
        case Neighborhood::Crossover: {
            // Fix where several pooled solutions already agree: the columns
            // every good solution shares are unlikely to be where the next
            // improvement lives.
            if (pool.size() < 2) return false;
            const std::size_t take = std::min<std::size_t>(pool.size(), 3);
            for (Index j = 0; j < n; ++j) {
                if (!integral_col(mip, j)) continue;
                const f64 v = pool.at(0)[sz(j)];
                if (!std::isfinite(v)) continue;
                bool agree = true;
                for (std::size_t s = 1; s < take && agree; ++s)
                    if (std::fabs(pool.at(s)[sz(j)] - v) > int_tol) agree = false;
                if (agree) {
                    if (!fix_at(j, v)) return false;
                } else {
                    ++out.free_integer;
                }
            }
            break;
        }
        case Neighborhood::LocalBranching: {
            // A Hamming ball around the incumbent, over binaries only:
            //   sum_{inc=0} x_j + sum_{inc=1} (1 - x_j)  <=  k
            // which is linear once the constant is moved to the right side.
            if (!have_inc) return false;
            std::uint64_t bins = 0;
            f64 rhs_const = 0.0;
            for (Index j = 0; j < n; ++j) {
                if (!binary_col(mip, node_lo, node_hi, j)) continue;
                if (!std::isfinite(x_inc[sz(j)])) continue;
                ++bins;
                if (x_inc[sz(j)] > 0.5) {
                    out.distance_cols.push_back(j);
                    out.distance_vals.push_back(-1.0);
                    rhs_const -= 1.0;
                } else {
                    out.distance_cols.push_back(j);
                    out.distance_vals.push_back(1.0);
                }
                ++out.free_integer;
            }
            if (bins < 4) return false;
            // The ball radius shrinks as the fixing rate rises, so the same
            // adaptive rule that shrinks a fix-based neighborhood shrinks this
            // one too.
            const f64 k = std::max(
                2.0, std::floor(static_cast<f64>(bins) * (1.0 - fixing_rate)));
            out.distance_rhs = k + rhs_const;
            out.has_distance_row = true;
            // Nothing is literally fixed here; report the ball as the fixed
            // share so the caller's "did this restrict anything" test passes.
            out.fixed = bins - static_cast<std::uint64_t>(k);
            break;
        }
        case Neighborhood::Proximity: {
            // Fischetti & Monaci: drop the true objective, demand a strictly
            // better one as a constraint, and minimise the Hamming distance to
            // the incumbent instead. The sub-MIP then hunts for the NEAREST
            // improvement, which is usually far easier to find than the best.
            if (!have_inc) return false;
            std::uint64_t bins = 0;
            out.objective.assign(sz(n), 0.0);
            for (Index j = 0; j < n; ++j) {
                if (!binary_col(mip, node_lo, node_hi, j)) continue;
                if (!std::isfinite(x_inc[sz(j)])) continue;
                ++bins;
                out.objective[sz(j)] = x_inc[sz(j)] > 0.5 ? -1.0 : 1.0;
                ++out.free_integer;
            }
            if (bins < 4) return false;
            out.replace_objective = true;
            const f64 inc_obj = mip.objective(x_inc);
            if (!std::isfinite(inc_obj)) return false;
            const f64 delta = 1e-4 * (1.0 + std::fabs(inc_obj));
            out.objective_cutoff = mip.maximize ? inc_obj + delta : inc_obj - delta;
            out.has_objective_cutoff = true;
            out.fixed = bins;  // the cutoff row is the restriction here
            break;
        }
    }

    // Degenerate either way: nothing held still, or nothing left to search.
    if (out.fixed == 0 || out.free_integer == 0) return false;
    for (Index j = 0; j < n; ++j)
        if (out.col_lo[sz(j)] > out.col_hi[sz(j)]) return false;
    return true;
}

model::LpProblem apply_neighborhood(const model::LpProblem& mip,
                                    const NeighborhoodProblem& np) {
    model::LpProblem sub = mip;
    sub.col_lo = np.col_lo;
    sub.col_hi = np.col_hi;
    if (np.replace_objective && np.objective.size() == sz(mip.n_cols())) {
        sub.c = np.objective;
        sub.obj_offset = 0.0;
        sub.maximize = false;
    }

    const int extra = (np.has_distance_row ? 1 : 0) +
                      (np.has_objective_cutoff ? 1 : 0);
    if (extra == 0) return sub;

    // Same CSR rebuild pattern as bab.cpp's add_binary_cover_cuts / apply_cuts:
    // rows are appended, columns are untouched, so every existing index stays
    // valid.
    const Index m = mip.n_rows(), n = mip.n_cols();
    const auto& rp = mip.A.pattern.row_ptr();
    const auto& ci = mip.A.pattern.col_idx();
    const auto& av = mip.A.vals;
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    rows.reserve(sz(mip.nnz()) + np.distance_cols.size() + sz(n));
    cols.reserve(rows.capacity());
    vals.reserve(rows.capacity());
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            rows.push_back(i);
            cols.push_back(ci[sz(k)]);
            vals.push_back(av[sz(k)]);
        }

    Index next = m;
    if (np.has_distance_row) {
        for (std::size_t k = 0; k < np.distance_cols.size(); ++k) {
            rows.push_back(next);
            cols.push_back(np.distance_cols[k]);
            vals.push_back(np.distance_vals[k]);
        }
        sub.row_lo.push_back(-kInf);
        sub.row_hi.push_back(np.distance_rhs);
        sub.row_names.push_back("LNS_DIST");
        ++next;
    }
    if (np.has_objective_cutoff) {
        // The cutoff is stated against the ORIGINAL objective, which the
        // proximity arm has just removed from `sub.c` -- so it has to be
        // written out as a row over mip.c, not read back off sub.
        bool any = false;
        for (Index j = 0; j < n; ++j) {
            if (mip.c[sz(j)] == 0.0) continue;
            rows.push_back(next);
            cols.push_back(j);
            vals.push_back(mip.c[sz(j)]);
            any = true;
        }
        if (any) {
            if (mip.maximize) {
                sub.row_lo.push_back(np.objective_cutoff - mip.obj_offset);
                sub.row_hi.push_back(kInf);
            } else {
                sub.row_lo.push_back(-kInf);
                sub.row_hi.push_back(np.objective_cutoff - mip.obj_offset);
            }
            sub.row_names.push_back("LNS_CUTOFF");
            ++next;
        }
    }

    if (next == m) return sub;
    sub.A = sparse::from_triplets(next, n, rows, cols, vals);
    sub.row_names.resize(sz(next));
    return sub;
}

}  // namespace sor::search
