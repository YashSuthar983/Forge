#include "sor/search/spp_repair.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

struct Rng {
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed ? seed : 1u) {}
    std::uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    std::uint32_t below(std::uint32_t n) { return n ? next() % n : 0; }
    f64 uniform() { return (next() >> 8) * (1.0 / 16777216.0); }
};

bool is_binary_col(const model::LpProblem& lp, Index j, const std::vector<f64>& lo,
                   const std::vector<f64>& hi) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)] &&
           std::fabs(lo[sz(j)]) < 1e-9 && std::fabs(hi[sz(j)] - 1.0) < 1e-9;
}

enum class RowKind : std::uint8_t { Other, Partition, Packing, Covering };

}  // namespace

SppStructure detect_spp_structure(const model::LpProblem& lp,
                                  const std::vector<f64>& col_lo,
                                  const std::vector<f64>& col_hi) {
    SppStructure st;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    std::vector<char> seen(sz(lp.n_cols()), 0);
    for (Index i = 0; i < lp.n_rows(); ++i) {
        bool ok = rp[sz(i) + 1] > rp[sz(i)];
        for (core::Offset k = rp[sz(i)]; ok && k < rp[sz(i) + 1]; ++k)
            ok = lp.A.vals[sz(k)] == 1.0 && is_binary_col(lp, ci[sz(k)], col_lo, col_hi);
        if (!ok) continue;
        const f64 l = lp.row_lo[sz(i)], h = lp.row_hi[sz(i)];
        if (l == 1.0 && h == 1.0) ++st.partition_rows;
        else if (h == 1.0 && l <= 0.0) ++st.packing_rows;
        else if (l == 1.0 && !std::isfinite(h)) ++st.covering_rows;
        else continue;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            if (!seen[sz(ci[sz(k)])]) { seen[sz(ci[sz(k)])] = 1; ++st.movable_cols; }
    }
    return st;
}

bool spp_repair(const model::LpProblem& lp, const std::vector<f64>& col_lo,
                const std::vector<f64>& col_hi, const std::vector<f64>& x_start,
                const SppRepairOptions& opts, std::vector<f64>& x_out,
                SppRepairDiagnostics& diag) {
    const auto t0 = Clock::now();
    const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    const auto finish = [&](bool ok) { diag.ms = elapsed() * 1000.0; return ok; };
    const Index n = lp.n_cols(), m = lp.n_rows();
    if (static_cast<Index>(x_start.size()) != n || static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n || lp.is_integer.empty() || m == 0)
        return finish(false);
    diag.structure = detect_spp_structure(lp, col_lo, col_hi);
    if (diag.structure.rows() == 0) return finish(false);

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    // Column-wise view.
    std::vector<core::Offset> cp(sz(n) + 1, 0);
    for (core::Offset k = 0; k < rp[sz(m)]; ++k) ++cp[sz(ci[sz(k)]) + 1];
    for (Index j = 0; j < n; ++j) cp[sz(j) + 1] += cp[sz(j)];
    std::vector<Index> crow(sz(static_cast<Index>(cp[sz(n)])));
    std::vector<f64> cval(crow.size());
    {
        std::vector<core::Offset> fill(cp.begin(), cp.end() - 1);
        for (Index i = 0; i < m; ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index j = ci[sz(k)];
                crow[sz(static_cast<Index>(fill[sz(j)]))] = i;
                cval[sz(static_cast<Index>(fill[sz(j)]))] = av[sz(k)];
                ++fill[sz(j)];
            }
    }
    // Row kinds, movable columns.
    std::vector<RowKind> kind(sz(m), RowKind::Other);
    for (Index i = 0; i < m; ++i) {
        bool ok = rp[sz(i) + 1] > rp[sz(i)];
        for (core::Offset k = rp[sz(i)]; ok && k < rp[sz(i) + 1]; ++k)
            ok = av[sz(k)] == 1.0 && is_binary_col(lp, ci[sz(k)], col_lo, col_hi);
        if (!ok) continue;
        const f64 l = lp.row_lo[sz(i)], h = lp.row_hi[sz(i)];
        if (l == 1.0 && h == 1.0) kind[sz(i)] = RowKind::Partition;
        else if (h == 1.0 && l <= 0.0) kind[sz(i)] = RowKind::Packing;
        else if (l == 1.0 && !std::isfinite(h)) kind[sz(i)] = RowKind::Covering;
    }
    std::vector<char> movable(sz(n), 0);
    for (Index j = 0; j < n; ++j) movable[sz(j)] = is_binary_col(lp, j, col_lo, col_hi);

    // Objective in minimisation form, scaled to be small next to a violation.
    const f64 sense = lp.maximize ? -1.0 : 1.0;
    f64 cmax = 0.0;
    for (Index j = 0; j < n; ++j) cmax = std::max(cmax, std::fabs(lp.c[sz(j)]));
    const f64 cscale = cmax > 0.0 ? opts.objective_weight / cmax : 0.0;

    // Start: integers rounded into the box, movable ones to {0,1}.
    std::vector<f64> x0 = x_start;
    for (Index j = 0; j < n; ++j) {
        if (!std::isfinite(x0[sz(j)])) return finish(false);
        if (lp.is_integer[sz(j)]) {
            x0[sz(j)] = std::min(std::max(std::round(x0[sz(j)]), col_lo[sz(j)]), col_hi[sz(j)]);
        }
    }

    const f64 tol = opts.feas_tol;
    std::vector<f64> x = x0, act(sz(m), 0.0), w(sz(m), 1.0);
    const auto recompute_act = [&] {
        for (Index i = 0; i < m; ++i) {
            long double s = 0.0L;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                s += static_cast<long double>(av[sz(k)]) * x[sz(ci[sz(k)])];
            act[sz(i)] = static_cast<f64>(s);
        }
    };
    const auto viol_of = [&](Index i, f64 a) {
        const f64 lo = lp.row_lo[sz(i)], hi = lp.row_hi[sz(i)];
        if (a < lo - tol) return lo - a;
        if (a > hi + tol) return a - hi;
        return 0.0;
    };
    // Violated-row set with O(1) insert/erase.
    std::vector<Index> vlist;
    std::vector<Index> vpos(sz(m), -1);
    const auto set_violated = [&](Index i, bool v) {
        if (v && vpos[sz(i)] < 0) { vpos[sz(i)] = static_cast<Index>(vlist.size()); vlist.push_back(i); }
        else if (!v && vpos[sz(i)] >= 0) {
            const Index p = vpos[sz(i)];
            vlist[sz(p)] = vlist.back();
            vpos[sz(vlist[sz(p)])] = p;
            vlist.pop_back();
            vpos[sz(i)] = -1;
        }
    };
    const auto refresh_all = [&] {
        recompute_act();
        vlist.clear();
        std::fill(vpos.begin(), vpos.end(), -1);
        f64 total = 0.0;
        for (Index i = 0; i < m; ++i) {
            const f64 v = viol_of(i, act[sz(i)]);
            if (v > 0.0) { set_violated(i, true); total += v; }
        }
        return total;
    };
    const auto total_violation = [&] {
        f64 t = 0.0;
        for (const Index i : vlist) t += viol_of(i, act[sz(i)]);
        return t;
    };

    // Apply a flip of column j to value t, updating rows; returns the change in
    // weighted violation.
    const auto apply_flip = [&](Index j, f64 t) {
        const f64 d = t - x[sz(j)];
        f64 delta = 0.0;
        for (core::Offset q = cp[sz(j)]; q < cp[sz(j) + 1]; ++q) {
            const Index i = crow[sz(static_cast<Index>(q))];
            const f64 before = viol_of(i, act[sz(i)]);
            act[sz(i)] += cval[sz(static_cast<Index>(q))] * d;
            const f64 after = viol_of(i, act[sz(i)]);
            delta += w[sz(i)] * (after - before);
            set_violated(i, after > 0.0);
        }
        x[sz(j)] = t;
        return delta + sense * lp.c[sz(j)] * d * cscale;
    };

    Rng rng(opts.seed);
    diag.initial_violation = refresh_all();
    if (vlist.empty()) {
        // Already satisfied by the rounded start: validate and return.
    }
    f64 best_total = diag.initial_violation;
    std::vector<f64> best_x = x;
    std::vector<std::uint64_t> tabu_until(sz(n), 0);
    std::uint64_t iter = 0, since_improve = 0;
    int restart = 0;

    const auto valid = [&](const std::vector<f64>& p) {
        for (Index j = 0; j < n; ++j) {
            if (p[sz(j)] < col_lo[sz(j)] - tol || p[sz(j)] > col_hi[sz(j)] + tol) return false;
            if (lp.is_integer[sz(j)] && std::fabs(p[sz(j)] - std::round(p[sz(j)])) > opts.int_tol)
                return false;
        }
        return lp.max_row_violation(p) <= 10.0 * tol;
    };

    while (true) {
        if (vlist.empty()) {
            if (valid(x)) {
                x_out = x;
                diag.final_violation = 0.0;
                return finish(true);
            }
            break;   // satisfied by our tolerance but not by the model's: give up
        }
        if (diag.moves >= opts.max_moves) break;
        if ((iter & 63u) == 0 && elapsed() >= opts.time_limit_s) {
            diag.stopped_on_time = true;
            break;
        }
        ++iter;
        const Index r = vlist[rng.below(static_cast<std::uint32_t>(vlist.size()))];
        const f64 a_r = act[sz(r)];
        const bool need_up = a_r < lp.row_lo[sz(r)] - tol;

        // Candidates: movable columns of r that move its activity the right way.
        struct Cand { Index j; f64 delta; f64 target; std::vector<std::pair<Index, f64>> ejected; };
        Cand best{-1, std::numeric_limits<f64>::infinity(), 0.0, {}};
        std::uint32_t ties = 0;
        for (core::Offset k = rp[sz(r)]; k < rp[sz(r) + 1]; ++k) {
            const Index j = ci[sz(k)];
            if (!movable[sz(j)]) continue;
            const f64 a = av[sz(k)];
            const f64 target = (need_up == (a > 0.0)) ? 1.0 : 0.0;
            if (x[sz(j)] == target) continue;
            const bool is_tabu = tabu_until[sz(j)] > iter;

            // Trial: the flip, then eject the holders of every packing or
            // partitioning row it over-covers.
            std::vector<std::pair<Index, f64>> log;   // (column, previous value)
            f64 delta = 0.0;
            log.emplace_back(j, x[sz(j)]);
            delta += apply_flip(j, target);
            if (target == 1.0) {
                int ejected = 0;
                for (core::Offset q = cp[sz(j)]; q < cp[sz(j) + 1] && ejected < opts.max_ejections; ++q) {
                    const Index i = crow[sz(static_cast<Index>(q))];
                    if (kind[sz(i)] != RowKind::Partition && kind[sz(i)] != RowKind::Packing)
                        continue;
                    if (act[sz(i)] <= lp.row_hi[sz(i)] + tol) continue;
                    // Eject the currently selected columns of this row, one at
                    // a time, until it is no longer over-covered.
                    for (core::Offset u = rp[sz(i)];
                         u < rp[sz(i) + 1] && act[sz(i)] > lp.row_hi[sz(i)] + tol &&
                         ejected < opts.max_ejections; ++u) {
                        const Index e = ci[sz(u)];
                        if (e == j || x[sz(e)] != 1.0 || !movable[sz(e)]) continue;
                        log.emplace_back(e, 1.0);
                        delta += apply_flip(e, 0.0);
                        ++ejected;
                    }
                }
            }
            // Undo the trial exactly.
            for (std::size_t z = log.size(); z-- > 0;) apply_flip(log[z].first, log[z].second);
            ++diag.undone;
            if (is_tabu && !(total_violation() + delta < best_total - 1e-12)) continue;
            if (delta < best.delta - 1e-12) {
                best.j = j; best.delta = delta; best.target = target;
                best.ejected.assign(log.begin() + 1, log.end());
                ties = 1;
            } else if (std::fabs(delta - best.delta) <= 1e-12 && rng.below(++ties) == 0) {
                best.j = j; best.target = target;
                best.ejected.assign(log.begin() + 1, log.end());
            }
        }
        if (best.j < 0) {
            // No usable move for this row (all tabu or none): raise its weight.
            w[sz(r)] += 1.0;
            ++since_improve;
        } else {
            const f64 target = best.target;
            apply_flip(best.j, target);
            tabu_until[sz(best.j)] = iter + static_cast<std::uint64_t>(opts.tabu_tenure) + rng.below(5);
            for (const auto& [e, prev] : best.ejected) {
                (void)prev;
                apply_flip(e, 0.0);
                tabu_until[sz(e)] = iter + static_cast<std::uint64_t>(opts.tabu_tenure) + rng.below(5);
            }
            ++diag.moves;
            if (!best.ejected.empty()) {
                ++diag.compound_moves;
                diag.ejections += best.ejected.size();
            }
            if (best.delta >= -1e-12)
                for (const Index i : vlist) w[sz(i)] += 1.0;   // local minimum: reweight
        }
        const f64 cur = total_violation();
        if (cur < best_total - 1e-12) { best_total = cur; best_x = x; since_improve = 0; }
        else ++since_improve;

        if (since_improve > 4000 + 4ull * static_cast<std::uint64_t>(n)) {
            if (++restart > opts.max_restarts) break;
            ++diag.restarts;
            // Restart from the best point with a random burst of flips.
            x = best_x;
            const std::uint32_t burst = std::max<std::uint32_t>(3, static_cast<std::uint32_t>(n) / 200);
            for (std::uint32_t b = 0; b < burst; ++b) {
                const Index j = static_cast<Index>(rng.below(static_cast<std::uint32_t>(n)));
                if (movable[sz(j)]) x[sz(j)] = 1.0 - x[sz(j)];
            }
            std::fill(w.begin(), w.end(), 1.0);
            std::fill(tabu_until.begin(), tabu_until.end(), 0);
            refresh_all();
            since_improve = 0;
        }
    }
    diag.final_violation = best_total;
    return finish(false);
}

}  // namespace sor::search
