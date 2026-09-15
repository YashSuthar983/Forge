#include "sor/search/feasjump.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <utility>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// The soft objective row is carried as one extra row past the model's own, so
// every loop below treats it identically to a constraint and there is no
// separate objective branch anywhere in the search.
struct Search {
    const model::LpProblem& lp;
    const std::vector<f64>& lo;
    const std::vector<f64>& hi;
    const FeasJumpOptions& opts;

    Index n = 0;
    Index m = 0;          // model rows
    Index m_all = 0;      // model rows + optional objective row
    bool has_obj_row = false;

    // Column-major adjacency (the row-major CSR alone cannot answer "which rows
    // does column j touch", which is the inner loop of every score below).
    std::vector<core::Offset> col_ptr;
    std::vector<Index> col_row;
    std::vector<f64> col_val;

    std::vector<f64> x;
    std::vector<f64> act;      // row activity, size m_all
    std::vector<f64> weight;   // row weight, size m_all
    std::vector<f64> rlo, rhi; // row bounds, size m_all

    // Violated-row membership, maintained incrementally: scanning all rows at
    // every local minimum would make weight updates the dominant cost.
    std::vector<int> viol_pos;      // index into viol_list, or -1
    std::vector<Index> viol_list;

    // Cached jump target and score per column, with a version stamp so the
    // priority queue can hold stale entries and discard them on pop instead of
    // paying for a decrease-key structure.
    std::vector<f64> jump_to;
    std::vector<f64> score;
    std::vector<std::uint32_t> stamp;

    std::mt19937 rng;
    // Reused breakpoint scratch, so the inner loop does not allocate.
    std::vector<std::pair<f64, f64>> bp_;

    Search(const model::LpProblem& p, const std::vector<f64>& l,
           const std::vector<f64>& h, const FeasJumpOptions& o)
        : lp(p), lo(l), hi(h), opts(o), rng(o.seed) {}

    bool integral_col(Index j) const {
        return !lp.is_integer.empty() && lp.is_integer[sz(j)];
    }

    f64 row_viol(Index i, f64 a) const {
        f64 v = 0.0;
        if (a < rlo[sz(i)]) v += rlo[sz(i)] - a;
        if (a > rhi[sz(i)]) v += a - rhi[sz(i)];
        return v;
    }

    void set_violated(Index i, bool on) {
        const bool was = viol_pos[sz(i)] >= 0;
        if (was == on) return;
        if (on) {
            viol_pos[sz(i)] = static_cast<int>(viol_list.size());
            viol_list.push_back(i);
        } else {
            const auto p = static_cast<std::size_t>(viol_pos[sz(i)]);
            const Index moved = viol_list.back();
            viol_list[p] = moved;
            viol_pos[sz(moved)] = static_cast<int>(p);
            viol_list.pop_back();
            viol_pos[sz(i)] = -1;
        }
    }

    void refresh_violated(Index i) {
        set_violated(i, row_viol(i, act[sz(i)]) > opts.feas_tol);
    }

    void build();
    void seed(const std::vector<f64>* x_start);
    void recompute_activity();
    // Weighted violation of the rows touching j, if j took value v.
    f64 local_cost(Index j, f64 v) const;
    // Best value for j other than its current one, and the score gain.
    void compute_jump(Index j);
    void apply_move(Index j, f64 v);
    bool run(std::vector<f64>& x_out, FeasJumpDiagnostics& diag);
};

void Search::build() {
    n = lp.n_cols();
    m = lp.n_rows();
    has_obj_row = std::isfinite(opts.objective_cutoff);
    m_all = m + (has_obj_row ? 1 : 0);

    rlo.assign(sz(m_all), -kInf);
    rhi.assign(sz(m_all), kInf);
    for (Index i = 0; i < m; ++i) {
        rlo[sz(i)] = lp.row_lo[sz(i)];
        rhi[sz(i)] = lp.row_hi[sz(i)];
    }

    // Column adjacency by counting sort over the CSR.
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<core::Offset> count(sz(n) + 1, 0);
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            ++count[sz(ci[sz(k)]) + 1];
    if (has_obj_row)
        for (Index j = 0; j < n; ++j)
            if (lp.c[sz(j)] != 0.0) ++count[sz(j) + 1];
    for (Index j = 0; j < n; ++j) count[sz(j) + 1] += count[sz(j)];
    col_ptr = count;
    col_row.assign(sz(col_ptr[sz(n)]), 0);
    col_val.assign(col_row.size(), 0.0);
    std::vector<core::Offset> fill = col_ptr;
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            col_row[sz(fill[sz(j)])] = i;
            col_val[sz(fill[sz(j)])] = av[sz(k)];
            ++fill[sz(j)];
        }
    if (has_obj_row) {
        // The objective row is written in the ORIGINAL sense: for a maximize
        // model the cutoff is a lower bound on c'x, not an upper one.
        const f64 off = lp.obj_offset;
        if (lp.maximize) {
            rlo[sz(m)] = opts.objective_cutoff - off;
            rhi[sz(m)] = kInf;
        } else {
            rlo[sz(m)] = -kInf;
            rhi[sz(m)] = opts.objective_cutoff - off;
        }
        for (Index j = 0; j < n; ++j) {
            if (lp.c[sz(j)] == 0.0) continue;
            col_row[sz(fill[sz(j)])] = m;
            col_val[sz(fill[sz(j)])] = lp.c[sz(j)];
            ++fill[sz(j)];
        }
    }

    weight.assign(sz(m_all), 1.0);
    if (has_obj_row) weight[sz(m)] = opts.objective_weight;
    act.assign(sz(m_all), 0.0);
    viol_pos.assign(sz(m_all), -1);
    jump_to.assign(sz(n), 0.0);
    score.assign(sz(n), 0.0);
    stamp.assign(sz(n), 0);
}

void Search::seed(const std::vector<f64>* x_start) {
    x.assign(sz(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        f64 v;
        if (x_start != nullptr && static_cast<Index>(x_start->size()) == n) {
            v = (*x_start)[sz(j)];
        } else {
            // No LP to start from: take the bound nearest zero, which is the
            // paper's default and is what makes the heuristic usable on a model
            // whose relaxation has not been solved.
            v = 0.0;
        }
        v = std::min(std::max(v, lo[sz(j)]), hi[sz(j)]);
        if (integral_col(j)) {
            v = std::round(v);
            v = std::min(std::max(v, std::ceil(lo[sz(j)] - 1e-9)),
                         std::floor(hi[sz(j)] + 1e-9));
        }
        if (!std::isfinite(v)) v = 0.0;
        x[sz(j)] = v;
    }
}

void Search::recompute_activity() {
    std::fill(act.begin(), act.end(), 0.0);
    for (Index j = 0; j < n; ++j)
        for (core::Offset k = col_ptr[sz(j)]; k < col_ptr[sz(j) + 1]; ++k)
            act[sz(col_row[sz(k)])] += col_val[sz(k)] * x[sz(j)];
    viol_list.clear();
    std::fill(viol_pos.begin(), viol_pos.end(), -1);
    for (Index i = 0; i < m_all; ++i) refresh_violated(i);
}

f64 Search::local_cost(Index j, f64 v) const {
    const f64 d = v - x[sz(j)];
    f64 c = 0.0;
    for (core::Offset k = col_ptr[sz(j)]; k < col_ptr[sz(j) + 1]; ++k) {
        const Index i = col_row[sz(k)];
        c += weight[sz(i)] * row_viol(i, act[sz(i)] + col_val[sz(k)] * d);
    }
    return c;
}

// The exact jump target, in O(deg log deg).
//
// The weighted violation restricted to column j is
//     f(v) = sum_i w_i * viol_i( act_i + a_ij * (v - x_j) ),
// and viol_i is convex in its argument while the argument is affine in v, so
// EVERY term is convex in v and so is f. That is the whole trick: a convex
// piecewise-linear function is minimised at the first breakpoint where its
// slope turns non-negative, which one sorted sweep finds exactly. Each row
// contributes at most two breakpoints -- the values of v that put that row's
// activity precisely on each of its finite bounds -- and each contributes the
// same |a_ij| * w_i to the slope there.
//
// The obvious alternative, enumerating those breakpoints as candidate values
// and evaluating f at each, is what this function used to do, and it is
// O(deg^2). On mcsched (two rows of ~1500 nonzeros, columns of degree ~30)
// that held the whole search to 13k moves in 3 s against timtab1's 150k/s.
void Search::compute_jump(Index j) {
    ++stamp[sz(j)];
    score[sz(j)] = 0.0;
    jump_to[sz(j)] = x[sz(j)];
    const f64 l = lo[sz(j)], h = hi[sz(j)];
    if (!(h > l + 1e-12)) return;  // fixed column

    const bool integral = integral_col(j);
    f64 dlo = l, dhi = h;
    if (integral) {
        dlo = std::ceil(l - 1e-9);
        dhi = std::floor(h + 1e-9);
        if (dhi < dlo) return;
    }
    const f64 cur = x[sz(j)];

    // Breakpoints and the slope as v -> -infinity.
    bp_.clear();
    f64 slope = 0.0;
    for (core::Offset k = col_ptr[sz(j)]; k < col_ptr[sz(j) + 1]; ++k) {
        const Index i = col_row[sz(k)];
        const f64 a = col_val[sz(k)];
        if (a == 0.0) continue;
        // [L, R] is the v-interval on which row i is satisfied. A negative
        // coefficient reverses which row bound gives which end.
        f64 L, R;
        if (a > 0.0) {
            L = std::isfinite(rlo[sz(i)]) ? cur + (rlo[sz(i)] - act[sz(i)]) / a
                                          : -kInf;
            R = std::isfinite(rhi[sz(i)]) ? cur + (rhi[sz(i)] - act[sz(i)]) / a
                                          : kInf;
        } else {
            L = std::isfinite(rhi[sz(i)]) ? cur + (rhi[sz(i)] - act[sz(i)]) / a
                                          : -kInf;
            R = std::isfinite(rlo[sz(i)]) ? cur + (rlo[sz(i)] - act[sz(i)]) / a
                                          : kInf;
        }
        const f64 g = std::fabs(a) * weight[sz(i)];
        if (!(g > 0.0)) continue;
        // Left of L the row is violated and f decreases at rate g; it flattens
        // at L and starts increasing again at R.
        if (std::isfinite(L)) {
            slope -= g;
            bp_.emplace_back(L, g);
        }
        if (std::isfinite(R)) bp_.emplace_back(R, g);
    }

    f64 v_star;
    if (slope >= 0.0) {
        v_star = dlo;  // non-decreasing everywhere: take the smallest value
    } else {
        std::sort(bp_.begin(), bp_.end());
        v_star = dhi;  // decreasing all the way out: take the largest
        f64 s = slope;
        for (const auto& [v, g] : bp_) {
            s += g;
            if (s >= 0.0) { v_star = v; break; }
        }
    }
    // Clamping a convex function's unconstrained minimiser into the domain
    // gives the constrained minimiser.
    v_star = std::min(std::max(v_star, dlo), dhi);

    const f64 base = local_cost(j, cur);
    f64 best_v = cur, best_c = base;
    auto consider = [&](f64 v) {
        if (!std::isfinite(v)) return;
        v = std::min(std::max(v, dlo), dhi);
        if (std::fabs(v - cur) <= 1e-12) return;
        const f64 c = local_cost(j, v);
        if (c < best_c - 1e-12 ||
            (c < best_c + 1e-12 && std::fabs(v - cur) < std::fabs(best_v - cur))) {
            best_c = c;
            best_v = v;
        }
    };
    if (integral) {
        // The integer minimiser of a convex function is one of the two
        // integers bracketing the continuous one.
        consider(std::floor(v_star));
        consider(std::ceil(v_star));
    } else {
        consider(v_star);
    }

    if (std::fabs(best_v - cur) <= 1e-12) return;
    jump_to[sz(j)] = best_v;
    score[sz(j)] = base - best_c;
}

void Search::apply_move(Index j, f64 v) {
    const f64 d = v - x[sz(j)];
    x[sz(j)] = v;
    for (core::Offset k = col_ptr[sz(j)]; k < col_ptr[sz(j) + 1]; ++k) {
        const Index i = col_row[sz(k)];
        act[sz(i)] += col_val[sz(k)] * d;
        refresh_violated(i);
    }
}

bool Search::run(std::vector<f64>& x_out, FeasJumpDiagnostics& diag) {
    const auto t0 = Clock::now();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();

    struct Item {
        f64 s;
        Index j;
        std::uint32_t st;
        bool operator<(const Item& o) const { return s < o.s; }
    };
    std::priority_queue<Item> pq;
    auto refresh = [&](Index j) {
        compute_jump(j);
        if (score[sz(j)] > 1e-12) pq.push({score[sz(j)], j, stamp[sz(j)]});
    };
    // Stale entries are discarded on pop, but a move can push hundreds of
    // refreshed columns while popping one, so pushes outrun pops and the queue
    // would otherwise grow without bound. Rebuilding from the live scores once
    // it exceeds a few times the column count bounds both the memory and the
    // log factor, at one O(n) sweep amortised over those pushes.
    //
    // This is a memory bound, not a measured speedup: adding it changed
    // mcsched's move count by under 10%, so unbounded queue growth was not
    // what was limiting that instance.
    const std::size_t pq_cap = std::max<std::size_t>(1024, 4 * sz(n));
    auto compact = [&]() {
        if (pq.size() <= pq_cap) return;
        pq = std::priority_queue<Item>();
        for (Index j = 0; j < n; ++j)
            if (score[sz(j)] > 1e-12) pq.push({score[sz(j)], j, stamp[sz(j)]});
    };

    std::vector<f64> best_x = x;
    std::size_t best_viol = viol_list.size();
    std::uint64_t stalls = 0;
    // Marks columns already queued for refresh in this step, so a column shared
    // by several of the moved column's rows is recomputed once, not once per
    // shared row.
    std::vector<std::uint32_t> touched(sz(n), 0);
    std::uint32_t touch_mark = 0;
    // Rotating offset into over-long rows, so the bounded refresh window walks
    // across the row over successive moves instead of always re-examining the
    // same prefix.
    std::uint64_t row_cursor = 0;
    // Collects columns of row i that need refreshing, drawing from a shared
    // per-move budget and starting at a rotating offset so a long row is
    // covered across successive moves rather than always from its front.
    std::size_t refresh_budget = 0;
    auto gather_row = [&](Index i, std::vector<Index>& dirty) {
        const core::Offset beg = rp[sz(i)], end = rp[sz(i) + 1];
        const auto len = static_cast<std::size_t>(end - beg);
        if (len == 0 || refresh_budget == 0) return;
        const std::size_t take = std::min(len, refresh_budget);
        const std::size_t start =
            len > take ? static_cast<std::size_t>(row_cursor % len) : 0;
        for (std::size_t q = 0; q < take; ++q) {
            const auto k = beg + static_cast<core::Offset>((start + q) % len);
            const Index j = ci[sz(k)];
            if (touched[sz(j)] == touch_mark) continue;
            touched[sz(j)] = touch_mark;
            dirty.push_back(j);
        }
        refresh_budget -= take;
        if (len > take) row_cursor += take;
    };

    for (Index j = 0; j < n; ++j) refresh(j);

    const bool timed = opts.time_limit_s > 0.0;
    while (diag.iterations < opts.max_iterations) {
        ++diag.iterations;
        if ((diag.iterations & 0x3FF) == 0 && timed &&
            ms_since(t0) > opts.time_limit_s * 1000.0)
            break;

        if (viol_list.empty()) break;  // weighted or not, nothing is violated

        // Pop the best candidate, and RE-DERIVE its jump before trusting it.
        //
        // The cached score can be stale, because max_refresh_per_move
        // deliberately leaves some columns un-refreshed after a move. Acting on
        // a stale positive score would apply a move that does not actually
        // reduce the weighted violation, and a walk doing that can cycle
        // between assignments without ever reaching a true local minimum -- so
        // the weights never rise and the escape mechanism never fires.
        // Recomputing here costs one O(deg log deg) sweep per pop and makes
        // staleness cost at most a wasted pop.
        //
        // This is a soundness-of-the-search argument, not a measured fix: it
        // was added on the hypothesis that it explained mcsched (many moves,
        // almost no reweights) and it did NOT -- mcsched behaves the same with
        // and without it. It is kept because the reasoning holds regardless of
        // whether that particular instance is an instance of it.
        Index pick = -1;
        while (!pq.empty()) {
            const Item it = pq.top();
            pq.pop();
            if (it.st != stamp[sz(it.j)]) continue;  // superseded entry
            compute_jump(it.j);
            if (score[sz(it.j)] <= 1e-12) continue;  // stale: not an improvement
            pick = it.j;
            break;
        }

        if (pick < 0) {
            // Local minimum: no single jump reduces the weighted violation.
            // Raise the weight of every violated row and continue -- this is
            // the whole mechanism, and it is why this does not stop where
            // bab.cpp's try_round() stops.
            ++diag.weight_updates;
            f64 wmax = 0.0;
            for (const Index i : viol_list) {
                weight[sz(i)] += opts.weight_increment;
                wmax = std::max(wmax, weight[sz(i)]);
            }
            if (wmax > opts.weight_rescale_at)
                for (f64& w : weight) w = 1.0 + w / wmax;

            // Only columns in violated rows changed score.
            ++touch_mark;
            refresh_budget = opts.max_refresh_per_move;
            std::vector<Index> dirty;
            for (const Index i : viol_list) {
                if (i < m) {
                    gather_row(i, dirty);
                } else {
                    for (Index j = 0; j < n; ++j) {
                        if (lp.c[sz(j)] == 0.0) continue;
                        if (touched[sz(j)] == touch_mark) continue;
                        touched[sz(j)] = touch_mark;
                        dirty.push_back(j);
                    }
                }
            }
            for (const Index j : dirty) refresh(j);
            compact();

            if (viol_list.size() < best_viol) {
                best_viol = viol_list.size();
                best_x = x;
                stalls = 0;
            } else if (++stalls >= opts.restart_after_stalls) {
                // Restart from the best point seen, perturbed, with the weight
                // landscape reset. Without this the walk can spend the whole
                // budget circling one basin it has already flattened.
                ++diag.restarts;
                stalls = 0;
                x = best_x;
                std::fill(weight.begin(), weight.end(), 1.0);
                if (has_obj_row) weight[sz(m)] = opts.objective_weight;
                std::uniform_real_distribution<double> u(0.0, 1.0);
                for (Index j = 0; j < n; ++j) {
                    if (u(rng) > opts.restart_perturb_frac) continue;
                    const f64 l = lo[sz(j)], h = hi[sz(j)];
                    if (!(h > l + 1e-12)) continue;
                    const f64 span = std::isfinite(h) && std::isfinite(l)
                                         ? h - l : 4.0;
                    f64 v = (std::isfinite(l) ? l : x[sz(j)] - span * 0.5) +
                            u(rng) * span;
                    v = std::min(std::max(v, l), h);
                    if (integral_col(j)) v = std::round(v);
                    if (std::isfinite(v)) x[sz(j)] = v;
                }
                recompute_activity();
                pq = std::priority_queue<Item>();
                for (Index j = 0; j < n; ++j) refresh(j);
            }
            continue;
        }

        const f64 target = jump_to[sz(pick)];
        apply_move(pick, target);
        ++diag.moves;

        // Every column sharing a row with `pick` may have a different jump now.
        ++touch_mark;
        refresh_budget = opts.max_refresh_per_move;
        touched[sz(pick)] = touch_mark;
        std::vector<Index> dirty{pick};
        for (core::Offset k = col_ptr[sz(pick)]; k < col_ptr[sz(pick) + 1]; ++k) {
            const Index i = col_row[sz(k)];
            if (i >= m) continue;  // the objective row has no CSR entry
            gather_row(i, dirty);
        }
        for (const Index j : dirty) refresh(j);
        compact();

        if (viol_list.size() < best_viol) {
            best_viol = viol_list.size();
            best_x = x;
            stalls = 0;
        }
    }

    diag.ms = ms_since(t0);
    diag.best_violated_rows = std::min(best_viol, viol_list.size());

    // Accept only what actually checks out against the ORIGINAL model. The
    // search runs on a weighted surrogate with its own tolerance, so "the
    // surrogate says zero" is a reason to test the point, not to trust it.
    for (const std::vector<f64>* cand : {&x, &best_x}) {
        bool ok = true;
        for (Index j = 0; j < n && ok; ++j) {
            const f64 v = (*cand)[sz(j)];
            if (!std::isfinite(v)) ok = false;
            else if (v < lo[sz(j)] - opts.feas_tol ||
                     v > hi[sz(j)] + opts.feas_tol) ok = false;
            else if (integral_col(j) &&
                     std::fabs(v - std::round(v)) > 1e-6) ok = false;
        }
        if (ok && lp.max_row_violation(*cand) <= opts.feas_tol &&
            lp.max_bound_violation(*cand) <= opts.feas_tol) {
            x_out = *cand;
            diag.found = true;
            return true;
        }
    }
    return false;
}

}  // namespace

bool feasibility_jump(const model::LpProblem& lp,
                      const std::vector<f64>& col_lo,
                      const std::vector<f64>& col_hi,
                      const std::vector<f64>* x_start,
                      const FeasJumpOptions& opts,
                      std::vector<f64>& x_out,
                      FeasJumpDiagnostics& diag) {
    diag = FeasJumpDiagnostics{};
    const Index n = lp.n_cols();
    if (n <= 0 || static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return false;
    for (Index j = 0; j < n; ++j)
        if (col_lo[sz(j)] > col_hi[sz(j)]) return false;  // empty box

    Search s(lp, col_lo, col_hi, opts);
    s.build();
    s.seed(x_start);
    s.recompute_activity();
    return s.run(x_out, diag);
}

}  // namespace sor::search
