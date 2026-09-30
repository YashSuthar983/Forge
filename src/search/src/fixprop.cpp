// Fix-Propagate-Repair - see sor/search/fixprop.hpp for why this exists and
// what it is implementing (Salvagnin, MPC 2024).

#include "sor/search/fixprop.hpp"

#include "sor/core/route_debug.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/search/propagate.hpp"
#include "sor/search/prop_trail.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline bool is_integral(f64 v, f64 tol) {
    return std::fabs(v - std::round(v)) <= tol;
}

// One decision in the dive. `trail_mark` is where the trail stood before the
// decision, so undoing means restoring every entry above it.
struct Level {
    std::size_t trail_mark = 0;
    Index var = -1;
    f64 old_lo = 0.0;
    f64 old_hi = 0.0;
    f64 value = 0.0;
    bool alternative_used = false;
};

struct Strategy {
    FixPropVarRule var;
    FixPropValRule val;
    // true  -> on conflict, SHIFT already-fixed columns to clear the violation
    //          and carry on (Salvagnin's `dive`: repair, no backtrack)
    // false -> chronological backtracking (`dfs`)
    // Both are in the portfolio because they fail on different models: plain
    // dfs thrashes on the hard ones (measured: csched007 48k backtracks,
    // deepest 10%) while repair cannot undo an early bad decision.
    bool repair;
};

// The portfolio. Order matters: the cheapest and most generally effective
// rules run first, so a short budget still gets the best single shot. Rules
// needing a reference point are interleaved rather than front-loaded, because
// `lp_x` is frequently absent exactly on the instances this heuristic is for
// (their root LP never finished).
const Strategy kPortfolio[] = {
    {FixPropVarRule::Locks,           FixPropValRule::ZeroElse,  true},
    {FixPropVarRule::Locks,           FixPropValRule::ZeroElse,  false},
    {FixPropVarRule::SmallestDomain,  FixPropValRule::Lower,     true},
    {FixPropVarRule::LeastFractional, FixPropValRule::Reference, true},
    {FixPropVarRule::SmallestDomain,  FixPropValRule::Lower,     false},
    {FixPropVarRule::Locks,           FixPropValRule::Lower,     true},
    {FixPropVarRule::MostFractional,  FixPropValRule::Reference, true},
    {FixPropVarRule::Random,          FixPropValRule::ZeroElse,  true},
    {FixPropVarRule::SmallestDomain,  FixPropValRule::Upper,     true},
    {FixPropVarRule::Random,          FixPropValRule::Lower,     true},
    {FixPropVarRule::Locks,           FixPropValRule::Upper,     true},
    {FixPropVarRule::LeastFractional, FixPropValRule::ZeroElse,  false},
};
constexpr int kPortfolioSize =
    static_cast<int>(sizeof(kPortfolio) / sizeof(kPortfolio[0]));

class Dive {
public:
    Dive(const model::LpProblem& lp, const std::vector<f64>* lp_x,
         const FixPropOptions& opts, FixPropDiagnostics& diag)
        : lp_(lp), lp_x_(lp_x), opts_(opts), diag_(diag),
          n_(lp.n_cols()), m_(lp.n_rows()) {}

    // Locks, computed once from the CSR pattern (no CSC needed): a row can be
    // broken by raising x_j when the coefficient pushes activity toward a
    // finite row_hi, and by lowering it when it pushes toward a finite row_lo.
    void build() {
        up_locks_.assign(sz(n_), 0);
        down_locks_.assign(sz(n_), 0);
        const auto& rp = lp_.A.pattern.row_ptr();
        const auto& ci = lp_.A.pattern.col_idx();
        const auto& av = lp_.A.vals;
        for (Index i = 0; i < m_; ++i) {
            const bool fin_lo = std::isfinite(lp_.row_lo[sz(i)]);
            const bool fin_hi = std::isfinite(lp_.row_hi[sz(i)]);
            if (!fin_lo && !fin_hi) continue;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const f64 a = av[sz(k)];
                if (a == 0.0) continue;
                const Index j = ci[sz(k)];
                if (a > 0.0) {
                    if (fin_hi) ++up_locks_[sz(j)];
                    if (fin_lo) ++down_locks_[sz(j)];
                } else {
                    if (fin_lo) ++up_locks_[sz(j)];
                    if (fin_hi) ++down_locks_[sz(j)];
                }
            }
        }
        for (Index j = 0; j < n_; ++j)
            if (!lp_.is_integer.empty() && lp_.is_integer[sz(j)])
                int_cols_.push_back(j);

        // CSC by counting sort over the CSR pattern.
        col_ptr_.assign(sz(n_) + 1, 0);
        for (Index i = 0; i < m_; ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                ++col_ptr_[sz(ci[sz(k)]) + 1];
        for (Index j = 0; j < n_; ++j) col_ptr_[sz(j) + 1] += col_ptr_[sz(j)];
        col_row_.assign(sz(col_ptr_[sz(n_)]), 0);
        col_val_.assign(sz(col_ptr_[sz(n_)]), 0.0);
        std::vector<core::Offset> cursor(col_ptr_.begin(), col_ptr_.end() - 1);
        for (Index i = 0; i < m_; ++i) {
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index j = ci[sz(k)];
                const core::Offset dst = cursor[sz(j)]++;
                col_row_[sz(dst)] = i;
                col_val_[sz(dst)] = av[sz(k)];
            }
        }
    }

    const std::vector<Index>& int_cols() const { return int_cols_; }

    bool run(const Strategy& s, const std::vector<f64>& base_lo,
             const std::vector<f64>& base_hi, std::uint64_t seed,
             Clock::time_point t0, std::vector<f64>& x_out);

private:
    bool over_budget(Clock::time_point t0) const {
        if (opts_.time_limit_s > 0.0 &&
            ms_since(t0) > opts_.time_limit_s * 1000.0)
            return true;
        return work_limit_ > 0 && diag_.work > work_limit_;
    }

    bool abort_if_over_budget() {
        if (!over_budget(run_start_)) return false;
        if (!aborted_) {
#ifdef SOR_ROUTE_DEBUG
            char fields[160];
            const bool time_exhausted = opts_.time_limit_s > 0.0 &&
                ms_since(run_start_) > opts_.time_limit_s * 1000.0;
            std::snprintf(fields, sizeof fields,
                          "\"reason\":\"%s\",\"elapsed_ms\":%.3f,\"work\":%llu",
                          time_exhausted ? "deadline" : "work_limit",
                          ms_since(run_start_),
                          static_cast<unsigned long long>(diag_.work));
            SOR_ROUTE_PATH(1, "fixprop", "heuristic", "abort", fields);
#endif
        }
        aborted_ = true;
        return true;
    }

    void undo_to(std::size_t mark) {
        const auto& e = trail_.entries();
        for (std::size_t k = e.size(); k-- > mark;) {
            const auto& en = e[k];
            if (en.dir == BoundDir::Lower) lo_[sz(en.var)] = en.old_bound;
            else                           hi_[sz(en.var)] = en.old_bound;
        }
        trail_.truncate(mark);
    }

    // Propagate after a decision, charging deterministic work.
    bool propagate(int depth) {
        ++diag_.propagations;
        const auto pr = propagate_bounds_trail(lp_, lo_, hi_, &trail_, depth,
                                               opts_.feas_tol,
                                               opts_.propagation_rounds);
        // Charge an upper bound on the matrix traffic the call could have done.
        diag_.work += static_cast<std::uint64_t>(std::max(1, pr.rounds)) *
                      static_cast<std::uint64_t>(lp_.nnz());
        return pr.feasible;
    }

    bool resolve_conflict(std::vector<Level>& levels,
                          std::uint64_t& backtracks, Clock::time_point t0);
    void compute_activities();
    void bump_activities(Index j, f64 old_lo, f64 old_hi);
    bool tighten_row(Index i);
    bool propagate_incremental(int depth);
    f64 row_violation(Index i) const;
    void collect_violations();
    void shift_fixed(Index j, f64 new_value);
    std::uint64_t damage_of(Index j, f64 new_value) const;
    bool repair_walk(std::mt19937_64& rng, Clock::time_point t0);
    Index pick_var(FixPropVarRule rule, std::mt19937_64& rng) const;
    f64 pick_val(FixPropValRule rule, Index j) const;
    f64 alternative_val(Index j, f64 tried) const;
    bool finish_bottom(std::vector<f64>& x_out);

    const model::LpProblem& lp_;
    const std::vector<f64>* lp_x_;
    const FixPropOptions& opts_;
    FixPropDiagnostics& diag_;
    Index n_, m_;
    std::vector<Index> int_cols_;
    std::vector<std::uint32_t> up_locks_, down_locks_;
    // Column-wise view, built once: repair needs "which rows does j appear
    // in", which the CSR pattern cannot answer.
    std::vector<core::Offset> col_ptr_;
    std::vector<Index> col_row_;
    std::vector<f64> col_val_;
    // Row activity bounds under the CURRENT box. A row is violated when
    // [minact, maxact] misses [row_lo, row_hi] entirely.
    std::vector<f64> minact_, maxact_;
    std::vector<Index> viol_rows_;
    std::vector<f64> base_lo_, base_hi_;
    std::vector<Index> queue_;
    std::vector<char> queued_;
    int cur_depth_ = 0;
    std::vector<f64> lo_, hi_;
    PropTrail trail_;
    std::uint64_t work_limit_ = 0;
    Clock::time_point run_start_{};
    bool aborted_ = false;

public:
    void set_work_limit(std::uint64_t w) { work_limit_ = w; }
};


// --- Incremental propagation -------------------------------------------
//
// The shared propagate_bounds() sweeps every row up to max_rounds times, which
// is O(nnz) per fixing. A dive performs thousands of fixings, so that cost is
// what actually stopped this heuristic: measured before this existed,
// tbfp-network managed 405 fixings per second and s250r10 just 67. Only the
// rows containing a changed column can deduce anything new, so this keeps a
// dirty-row queue and the activity bounds it needs, updating both in place.

void Dive::bump_activities(Index j, f64 old_lo, f64 old_hi) {
    SOR_FN();
    const f64 dlo = lo_[sz(j)] - old_lo;
    const f64 dhi = hi_[sz(j)] - old_hi;
    if (dlo == 0.0 && dhi == 0.0) return;
    for (core::Offset k = col_ptr_[sz(j)]; k < col_ptr_[sz(j) + 1]; ++k) {
        if ((k - col_ptr_[sz(j)]) % 1024 == 0 && abort_if_over_budget())
            return;
        const Index i = col_row_[sz(k)];
        const f64 a = col_val_[sz(k)];
        if (a > 0.0) { minact_[sz(i)] += a * dlo; maxact_[sz(i)] += a * dhi; }
        else         { minact_[sz(i)] += a * dhi; maxact_[sz(i)] += a * dlo; }
        if (!queued_[sz(i)]) { queued_[sz(i)] = 1; queue_.push_back(i); }
    }
    diag_.work += static_cast<std::uint64_t>(col_ptr_[sz(j) + 1] -
                                             col_ptr_[sz(j)]);
}

// Derive bounds for every column of row i from the residual activity of the
// others. Returns false when the row cannot be satisfied inside the box.
bool Dive::tighten_row(Index i) {
    SOR_FN();
    const f64 rl = lp_.row_lo[sz(i)], rh = lp_.row_hi[sz(i)];
    if (!std::isfinite(rl) && !std::isfinite(rh)) return true;
    if (std::isfinite(rh) && minact_[sz(i)] > rh + 1e-7) return false;
    if (std::isfinite(rl) && maxact_[sz(i)] < rl - 1e-7) return false;

    const auto& rp = lp_.A.pattern.row_ptr();
    const auto& ci = lp_.A.pattern.col_idx();
    const auto& av = lp_.A.vals;
    diag_.work += static_cast<std::uint64_t>(rp[sz(i) + 1] - rp[sz(i)]);

    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        if ((k - rp[sz(i)]) % 1024 == 0 && abort_if_over_budget())
            return false;
        const f64 a = av[sz(k)];
        if (a == 0.0) continue;
        const Index j = ci[sz(k)];
        const f64 l = lo_[sz(j)], h = hi_[sz(j)];
        if (h - l <= 0.0) continue;                 // already fixed
        // Residual activity of row i with column j removed.
        const f64 jmin = (a > 0.0) ? a * l : a * h;
        const f64 jmax = (a > 0.0) ? a * h : a * l;
        const f64 rmin = minact_[sz(i)] - jmin;
        const f64 rmax = maxact_[sz(i)] - jmax;

        f64 new_lo = l, new_hi = h;
        if (std::isfinite(rh) && std::isfinite(rmin)) {
            const f64 slack = rh - rmin;            // a * x_j <= slack
            if (a > 0.0) new_hi = std::min(new_hi, slack / a);
            else         new_lo = std::max(new_lo, slack / a);
        }
        if (std::isfinite(rl) && std::isfinite(rmax)) {
            const f64 need = rl - rmax;             // a * x_j >= need
            if (a > 0.0) new_lo = std::max(new_lo, need / a);
            else         new_hi = std::min(new_hi, need / a);
        }
        if (!lp_.is_integer.empty() && lp_.is_integer[sz(j)]) {
            if (std::isfinite(new_lo)) new_lo = std::ceil(new_lo - 1e-6);
            if (std::isfinite(new_hi)) new_hi = std::floor(new_hi + 1e-6);
        }
        if (new_lo > new_hi + 1e-9) return false;

        if (new_lo > l + 1e-12 || new_hi < h - 1e-12) {
            const f64 old_l = l, old_h = h;
            if (new_lo > l + 1e-12) {
                trail_.push(j, BoundDir::Lower, new_lo, l, ReasonKind::Row, i,
                            cur_depth_);
                lo_[sz(j)] = new_lo;
            }
            if (new_hi < h - 1e-12) {
                trail_.push(j, BoundDir::Upper, new_hi, h, ReasonKind::Row, i,
                            cur_depth_);
                hi_[sz(j)] = new_hi;
            }
            bump_activities(j, old_l, old_h);
            if (aborted_) return false;
        }
    }
    return true;
}

bool Dive::propagate_incremental(int depth) {
    SOR_FN();
    ++diag_.propagations;
    cur_depth_ = depth;
    std::size_t head = 0;
    bool ok = true;
    while (head < queue_.size()) {
        if ((head & 63u) == 0 && abort_if_over_budget()) {
            ok = false;
            break;
        }
        const Index i = queue_[head++];
        queued_[sz(i)] = 0;
        if (!tighten_row(i)) { ok = false; break; }
        if (abort_if_over_budget()) { ok = false; break; }
    }
    for (std::size_t k = head; k < queue_.size(); ++k)
        queued_[sz(queue_[k])] = 0;
    queue_.clear();
    return ok;
}

// ---------------------------------------------------------------------------
// Repair (Salvagnin, MPC 2024, Section 4) -- a WalkSAT generalisation that
// operates on the PARTIAL assignment rather than a complete one. The state is
// the current box; a row's violation is the distance between its activity
// interval [minact, maxact] and [row_lo, row_hi].

void Dive::compute_activities() {
    minact_.assign(sz(m_), 0.0);
    maxact_.assign(sz(m_), 0.0);
    const auto& rp = lp_.A.pattern.row_ptr();
    const auto& ci = lp_.A.pattern.col_idx();
    const auto& av = lp_.A.vals;
    for (Index i = 0; i < m_; ++i) {
        f64 lo_sum = 0.0, hi_sum = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 a = av[sz(k)];
            if (a == 0.0) continue;
            const Index j = ci[sz(k)];
            const f64 l = lo_[sz(j)], h = hi_[sz(j)];
            lo_sum += (a > 0.0) ? a * l : a * h;
            hi_sum += (a > 0.0) ? a * h : a * l;
        }
        minact_[sz(i)] = lo_sum;
        maxact_[sz(i)] = hi_sum;
    }
    diag_.work += static_cast<std::uint64_t>(lp_.nnz());
    queued_.assign(sz(m_), 0);
    queue_.clear();
}

f64 Dive::row_violation(Index i) const {
    const f64 rl = lp_.row_lo[sz(i)], rh = lp_.row_hi[sz(i)];
    // No assignment inside the box can reach the row's range.
    if (std::isfinite(rh) && minact_[sz(i)] > rh + opts_.feas_tol)
        return minact_[sz(i)] - rh;
    if (std::isfinite(rl) && maxact_[sz(i)] < rl - opts_.feas_tol)
        return rl - maxact_[sz(i)];
    return 0.0;
}

void Dive::collect_violations() {
    viol_rows_.clear();
    for (Index i = 0; i < m_; ++i)
        if (row_violation(i) > opts_.feas_tol) viol_rows_.push_back(i);
}

// Move a FIXED column to a new value, keeping activities in step. Both
// activity ends move by a*delta because a fixed column contributes a*v to each.
void Dive::shift_fixed(Index j, f64 new_value) {
    const f64 delta = new_value - lo_[sz(j)];
    if (delta == 0.0) return;
    lo_[sz(j)] = hi_[sz(j)] = new_value;
    for (core::Offset k = col_ptr_[sz(j)]; k < col_ptr_[sz(j) + 1]; ++k) {
        const Index i = col_row_[sz(k)];
        const f64 d = col_val_[sz(k)] * delta;
        minact_[sz(i)] += d;
        maxact_[sz(i)] += d;
        if (!queued_[sz(i)]) { queued_[sz(i)] = 1; queue_.push_back(i); }
    }
    diag_.work += static_cast<std::uint64_t>(col_ptr_[sz(j) + 1] -
                                             col_ptr_[sz(j)]);
}

// WalkSAT scores a flip by the DAMAGE it does only -- the rows it newly breaks
// -- deliberately ignoring what it fixes. Keeping that asymmetry matters; it is
// what stops the walk from greedily chasing the row it just picked.
std::uint64_t Dive::damage_of(Index j, f64 new_value) const {
    const f64 delta = new_value - lo_[sz(j)];
    std::uint64_t broken = 0;
    for (core::Offset k = col_ptr_[sz(j)]; k < col_ptr_[sz(j) + 1]; ++k) {
        const Index i = col_row_[sz(k)];
        const f64 d = col_val_[sz(k)] * delta;
        const f64 nmin = minact_[sz(i)] + d, nmax = maxact_[sz(i)] + d;
        const f64 rl = lp_.row_lo[sz(i)], rh = lp_.row_hi[sz(i)];
        const bool was_ok = row_violation(i) <= opts_.feas_tol;
        const bool now_bad =
            (std::isfinite(rh) && nmin > rh + opts_.feas_tol) ||
            (std::isfinite(rl) && nmax < rl - opts_.feas_tol);
        if (was_ok && now_bad) ++broken;
    }
    return broken;
}

bool Dive::repair_walk(std::mt19937_64& rng, Clock::time_point t0) {
    compute_activities();
    collect_violations();
    constexpr double kNoise = 0.20;          // WalkSAT noise parameter
    const std::uint64_t max_steps = 200;
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    for (std::uint64_t step = 0; step < max_steps; ++step) {
        if (viol_rows_.empty()) return true;
        if (over_budget(t0)) return false;

        std::uniform_int_distribution<std::size_t> pick(0, viol_rows_.size() - 1);
        const Index i = viol_rows_[pick(rng)];
        const f64 v = row_violation(i);
        if (v <= opts_.feas_tol) { collect_violations(); continue; }

        const f64 rh = lp_.row_hi[sz(i)];
        const bool over = std::isfinite(rh) && minact_[sz(i)] > rh + opts_.feas_tol;

        // Candidate shifts: only FIXED columns, moved within their ORIGINAL
        // domain, and only in the direction that reduces this row's violation.
        Index best_j = -1;
        f64 best_val = 0.0;
        std::uint64_t best_damage = ~0ull;
        std::size_t seen = 0;
        Index rand_j = -1;
        f64 rand_val = 0.0;

        const auto& rp = lp_.A.pattern.row_ptr();
        const auto& ci = lp_.A.pattern.col_idx();
        const auto& av = lp_.A.vals;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 a = av[sz(k)];
            if (a == 0.0) continue;
            const Index j = ci[sz(k)];
            if (hi_[sz(j)] - lo_[sz(j)] > 0.5) continue;   // not fixed
            const f64 cur = lo_[sz(j)];
            // Need activity to move DOWN when over, UP when under.
            const f64 want = over ? -v : v;
            f64 step_val = cur + want / a;
            if (!std::isfinite(step_val)) continue;
            const bool integral = !lp_.is_integer.empty() && lp_.is_integer[sz(j)];
            if (integral) {
                // Round in the direction that actually helps.
                step_val = (want / a > 0.0) ? std::ceil(step_val - opts_.int_tol)
                                            : std::floor(step_val + opts_.int_tol);
            }
            step_val = std::min(std::max(step_val, base_lo_[sz(j)]),
                                base_hi_[sz(j)]);
            if (std::fabs(step_val - cur) < (integral ? 0.5 : opts_.feas_tol))
                continue;
            ++seen;
            std::uniform_int_distribution<std::size_t> r(1, seen);
            if (r(rng) == 1) { rand_j = j; rand_val = step_val; }
            const std::uint64_t dmg = damage_of(j, step_val);
            if (dmg < best_damage) { best_damage = dmg; best_j = j; best_val = step_val; }
        }
        if (best_j < 0 && rand_j < 0) return false;   // nothing can move

        const bool take_random = unit(rng) < kNoise && rand_j >= 0;
        const Index j = take_random ? rand_j : best_j;
        const f64 val = take_random ? rand_val : best_val;
        shift_fixed(j, val);
        if (aborted_) return false;
        collect_violations();
    }
    return viol_rows_.empty();
}

Index Dive::pick_var(FixPropVarRule rule, std::mt19937_64& rng) const {
    Index best = -1;
    f64 best_score = 0.0;
    std::uint64_t seen = 0;
    for (const Index j : int_cols_) {
        const f64 l = lo_[sz(j)], h = hi_[sz(j)];
        if (h - l < 0.5) continue;  // already fixed (integral domain)
        f64 score = 0.0;
        switch (rule) {
            case FixPropVarRule::SmallestDomain:
                score = -(h - l);
                break;
            case FixPropVarRule::Locks:
                score = -static_cast<f64>(
                    std::min(up_locks_[sz(j)], down_locks_[sz(j)]));
                break;
            case FixPropVarRule::LeastFractional:
            case FixPropVarRule::MostFractional: {
                if (lp_x_ == nullptr || static_cast<Index>(lp_x_->size()) != n_) {
                    score = -(h - l);  // fall back to first-fail
                    break;
                }
                const f64 v = (*lp_x_)[sz(j)];
                const f64 frac = std::fabs(v - std::round(v));
                score = (rule == FixPropVarRule::LeastFractional) ? -frac : frac;
                break;
            }
            case FixPropVarRule::Random: {
                // Reservoir sampling: one uniform pick in a single pass.
                ++seen;
                std::uniform_int_distribution<std::uint64_t> d(1, seen);
                if (d(rng) == 1) best = j;
                continue;
            }
        }
        if (best < 0 || score > best_score) { best = j; best_score = score; }
    }
    return best;
}

f64 Dive::pick_val(FixPropValRule rule, Index j) const {
    const f64 l = lo_[sz(j)], h = hi_[sz(j)];
    f64 v = l;
    switch (rule) {
        case FixPropValRule::Lower: v = l; break;
        case FixPropValRule::Upper: v = h; break;
        case FixPropValRule::ZeroElse:
            v = (l <= 0.0 && 0.0 <= h) ? 0.0
                                       : (std::fabs(l) <= std::fabs(h) ? l : h);
            break;
        case FixPropValRule::Reference:
            if (lp_x_ != nullptr && static_cast<Index>(lp_x_->size()) == n_) {
                v = std::round((*lp_x_)[sz(j)]);
                v = std::min(std::max(v, l), h);
            } else {
                v = (l <= 0.0 && 0.0 <= h) ? 0.0 : l;
            }
            break;
    }
    if (!std::isfinite(v)) v = std::isfinite(l) ? l : (std::isfinite(h) ? h : 0.0);
    v = std::round(v);
    return std::min(std::max(v, l), h);
}

// The one retry a level gets: jump to the opposite end of the domain, which is
// the most different decision available, rather than nudging by one.
f64 Dive::alternative_val(Index j, f64 tried) const {
    const f64 l = lo_[sz(j)], h = hi_[sz(j)];
    if (!(h > l + 0.5)) return tried;
    const f64 far = (std::fabs(tried - l) >= std::fabs(h - tried)) ? l : h;
    const f64 v = std::round(far);
    if (std::fabs(v - tried) < 0.5) {
        const f64 alt = (tried + 1.0 <= h) ? tried + 1.0 : tried - 1.0;
        return (alt >= l && alt <= h) ? alt : tried;
    }
    return v;
}

// Every integer column is fixed. Continuous columns still need values; get
// them from one LP with the integers pinned. Integral models skip the LP.
bool Dive::finish_bottom(std::vector<f64>& x_out) {
    bool has_continuous = false;
    for (Index j = 0; j < n_; ++j) {
        if (lp_.is_integer.empty() || !lp_.is_integer[sz(j)]) {
            has_continuous = true;
            break;
        }
    }

    if (!has_continuous || opts_.bottom_lp_time_s <= 0.0) {
        std::vector<f64> x(sz(n_));
        for (Index j = 0; j < n_; ++j) {
            // Propagation leaves a (possibly narrowed) box; take the lower end,
            // which is integral for integer columns after the ceil/floor it
            // applies.
            f64 v = lo_[sz(j)];
            if (!std::isfinite(v)) v = std::isfinite(hi_[sz(j)]) ? hi_[sz(j)] : 0.0;
            x[sz(j)] = v;
        }
        if (lp_.max_row_violation(x) > opts_.feas_tol ||
            lp_.max_bound_violation(x) > opts_.feas_tol)
            return false;
        x_out = std::move(x);
        return true;
    }

    model::LpProblem fixed = lp_;
    for (Index j = 0; j < n_; ++j) {
        if (lp_.is_integer.empty() || !lp_.is_integer[sz(j)]) continue;
        f64 v = std::round(lo_[sz(j)]);
        if (!std::isfinite(v)) return false;
        v = std::min(std::max(v, lp_.col_lo[sz(j)]), lp_.col_hi[sz(j)]);
        if (!is_integral(v, opts_.int_tol)) return false;
        fixed.col_lo[sz(j)] = v;
        fixed.col_hi[sz(j)] = v;
    }
    engines::SimplexOptions so;
    so.method = engines::SimplexMethod::Auto;
    so.presolve = true;
    so.max_iterations = opts_.bottom_lp_iterations;
    so.time_limit_s = opts_.bottom_lp_time_s;
    so.primal_feas_tol = opts_.feas_tol;
    so.dual_feas_tol = std::max(opts_.feas_tol, 1e-7);
    engines::SimplexDiagnostics sd;
    ++diag_.bottom_lps;
    const auto r = engines::solve_simplex(fixed, so, sd, nullptr);
    if (r.proposed_status != core::Status::Optimal &&
        r.proposed_status != core::Status::Feasible)
        return false;
    if (static_cast<Index>(r.x.size()) != n_) return false;
    x_out = r.x;
    return true;
}

namespace {
inline std::size_t lv_trail_mark_of(const std::vector<Level>& levels) {
    return levels.empty() ? 0 : levels.back().trail_mark;
}
}  // namespace

bool Dive::run(const Strategy& s, const std::vector<f64>& base_lo,
               const std::vector<f64>& base_hi, std::uint64_t seed,
               Clock::time_point t0, std::vector<f64>& x_out) {
    SOR_FN();
    run_start_ = t0;
    aborted_ = false;
    lo_ = base_lo;
    hi_ = base_hi;
    base_lo_ = base_lo;
    base_hi_ = base_hi;
    trail_.clear();
    compute_activities();
    if (abort_if_over_budget()) return false;
    for (Index i = 0; i < m_; ++i) { queued_[sz(i)] = 1; queue_.push_back(i); }
    if (!propagate_incremental(0)) return false;   // root box already empty
    std::mt19937_64 rng(seed);
    std::vector<Level> levels;

    const std::size_t total_int = int_cols_.size();
    std::uint64_t backtracks = 0;

    for (;;) {
        if (aborted_ || over_budget(t0)) return false;

        const Index j = pick_var(s.var, rng);
        if (j < 0) {
            const int pct = 100;
            diag_.best_depth_pct = std::max(diag_.best_depth_pct, pct);
            return finish_bottom(x_out);
        }

        if (total_int > 0) {
            const int pct = static_cast<int>(100 * levels.size() / total_int);
            diag_.best_depth_pct = std::max(diag_.best_depth_pct, pct);
        }

        Level lv;
        lv.trail_mark = trail_.size();
        lv.var = j;
        lv.old_lo = lo_[sz(j)];
        lv.old_hi = hi_[sz(j)];
        lv.value = pick_val(s.val, j);
        lo_[sz(j)] = hi_[sz(j)] = lv.value;
        bump_activities(j, lv.old_lo, lv.old_hi);
        ++diag_.fixings;
        levels.push_back(lv);

        if (!propagate_incremental(static_cast<int>(levels.size()))) {
            if (aborted_) return false;
            ++diag_.conflicts;
            if (s.repair) {
                // Undo only what propagation deduced; the decision itself
                // stays. Repair then shifts previously fixed columns until no
                // row is violated, and propagation runs again on the result.
                undo_to(lv_trail_mark_of(levels));
                ++diag_.repairs;
                if (!repair_walk(rng, t0)) return false;
                if (!propagate_incremental(static_cast<int>(levels.size())))
                    return false;
                ++diag_.repairs_ok;
            } else if (!resolve_conflict(levels, backtracks, t0)) {
                return false;
            }
        }
    }
}

// Unwind until some level accepts its alternative value and propagation is
// clean again. Each level gets exactly one retry (the far end of its domain);
// once that is spent the level is dropped and its parent is retried.
bool Dive::resolve_conflict(std::vector<Level>& levels,
                            std::uint64_t& backtracks,
                            Clock::time_point t0) {
    while (!levels.empty()) {
        if (over_budget(t0)) return false;

        Level& top = levels.back();
        undo_to(top.trail_mark);
        lo_[sz(top.var)] = top.old_lo;
        hi_[sz(top.var)] = top.old_hi;

        if (!top.alternative_used) {
            const f64 alt = alternative_val(top.var, top.value);
            if (std::fabs(alt - top.value) >= 0.5) {
                top.alternative_used = true;
                top.value = alt;
                lo_[sz(top.var)] = hi_[sz(top.var)] = alt;
                ++diag_.fixings;
                // undo_to() rewound bounds but not activities; they are
                // derived state, so rebuild them before propagating again.
                compute_activities();
                for (Index i = 0; i < m_; ++i) {
                    queued_[sz(i)] = 1;
                    queue_.push_back(i);
                }
                if (propagate_incremental(static_cast<int>(levels.size())))
                    return true;
                if (aborted_) return false;
                ++diag_.conflicts;
                continue;  // alternative also failed; the level is now spent
            }
        }

        levels.pop_back();
        ++diag_.backtracks;
        if (++backtracks > opts_.max_backtracks) return false;
    }
    return false;
}

}  // namespace

bool fix_and_propagate(const model::LpProblem& lp,
                       const std::vector<f64>& col_lo,
                       const std::vector<f64>& col_hi,
                       const std::vector<f64>* lp_x,
                       const FixPropOptions& opts,
                       std::vector<f64>& x_out,
                       FixPropDiagnostics& diag) {
    SOR_FN();
    SOR_ROUTE_PATH(1, "fixprop", "heuristic", "start");
    diag = FixPropDiagnostics{};
    const auto t0 = Clock::now();
    const Index n = lp.n_cols();
    if (n <= 0 || static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return false;

    Dive dive(lp, lp_x, opts, diag);
    dive.build();
    if (dive.int_cols().empty()) return false;  // nothing to fix; not our job
    dive.set_work_limit(opts.work_limit_nnz_multiple *
                        static_cast<std::uint64_t>(lp.nnz()));

    // Replace infinite bounds with a finite box so propagation always has
    // finite activities to work with. This is NOT a valid reformulation, which
    // is exactly why the result is re-validated against `lp` below.
    std::vector<f64> base_lo = col_lo, base_hi = col_hi;
    for (Index j = 0; j < n; ++j) {
        if (!std::isfinite(base_lo[sz(j)]))
            base_lo[sz(j)] = std::max(-opts.artificial_bound,
                                      -std::fabs(opts.artificial_bound));
        if (!std::isfinite(base_hi[sz(j)]))
            base_hi[sz(j)] = opts.artificial_bound;
        if (base_lo[sz(j)] > base_hi[sz(j)]) return false;
        if (!lp.is_integer.empty() && lp.is_integer[sz(j)]) {
            base_lo[sz(j)] = std::ceil(base_lo[sz(j)] - opts.int_tol);
            base_hi[sz(j)] = std::floor(base_hi[sz(j)] + opts.int_tol);
            if (base_lo[sz(j)] > base_hi[sz(j)]) return false;
        }
    }

    const int dives = std::max(1, opts.max_dives);
    for (int d = 0; d < dives; ++d) {
        if ((opts.time_limit_s > 0.0 &&
             ms_since(t0) > opts.time_limit_s * 1000.0) ||
            (opts.work_limit_nnz_multiple > 0 &&
             diag.work > opts.work_limit_nnz_multiple *
                             static_cast<std::uint64_t>(lp.nnz())))
            break;
        ++diag.dives;
        const Strategy& s = kPortfolio[d % kPortfolioSize];
        std::vector<f64> cand;
        if (!dive.run(s, base_lo, base_hi,
                      opts.seed + static_cast<std::uint64_t>(d) * 7919u, t0,
                      cand))
            continue;

        // Validate against the ORIGINAL model. The dive ran against an
        // artificially bounded box and a propagation tolerance, so "the dive
        // says feasible" is a reason to test the point, not to trust it.
        if (static_cast<Index>(cand.size()) != n) continue;
        bool ok = true;
        for (Index j = 0; j < n && ok; ++j) {
            const f64 v = cand[sz(j)];
            if (!std::isfinite(v)) ok = false;
            else if (v < lp.col_lo[sz(j)] - opts.feas_tol ||
                     v > lp.col_hi[sz(j)] + opts.feas_tol) ok = false;
            else if (!lp.is_integer.empty() && lp.is_integer[sz(j)] &&
                     !is_integral(v, opts.int_tol)) ok = false;
        }
        if (!ok) continue;
        if (lp.max_row_violation(cand) > opts.feas_tol ||
            lp.max_bound_violation(cand) > opts.feas_tol)
            continue;

        x_out = std::move(cand);
        diag.found = true;
        diag.ms = ms_since(t0);
        SOR_ROUTE_PATH(1, "fixprop", "heuristic", "found");
        return true;
    }

    diag.ms = ms_since(t0);
    SOR_ROUTE_PATH(1, "fixprop", "heuristic", "stop");
    return false;
}

}  // namespace sor::search
