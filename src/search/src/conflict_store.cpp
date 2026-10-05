#include "sor/search/conflict_store.hpp"

#include "sor/certify/finalize.hpp"
#include "sor/search/prop_trail.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
constexpr f64 kTol = 1e-9;

// A literal is false when the box excludes it, true when the box entails it.
inline bool lit_false(const ConflictLiteral& l, const std::vector<f64>& lo,
                      const std::vector<f64>& hi) {
    return l.upper ? lo[sz(l.var)] > l.bound + kTol
                   : hi[sz(l.var)] < l.bound - kTol;
}
inline bool lit_true(const ConflictLiteral& l, const std::vector<f64>& lo,
                     const std::vector<f64>& hi) {
    return l.upper ? hi[sz(l.var)] <= l.bound + kTol
                   : lo[sz(l.var)] >= l.bound - kTol;
}

std::uint64_t hash_lits(const std::vector<ConflictLiteral>& lits) {
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&](std::uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 1099511628211ull;
    };
    for (const auto& l : lits) {
        mix(static_cast<std::uint64_t>(l.var));
        mix(l.upper ? 1u : 0u);
        // Integral doubles can exceed int64. Hash their canonical bits rather
        // than invoking an out-of-range floating-to-integer conversion.
        mix(std::bit_cast<std::uint64_t>(l.bound == 0.0 ? 0.0 : l.bound));
    }
    return h;
}

}  // namespace

ConflictStore::ConflictStore(Index n_cols, std::size_t byte_budget, std::size_t max_len)
    : n_(n_cols), budget_(byte_budget), max_len_(max_len),
      by_var_(static_cast<std::size_t>(n_cols)) {}

std::vector<std::vector<ConflictLiteral>> ConflictStore::export_clauses() const {
    std::vector<std::vector<ConflictLiteral>> out;
    for (const auto& c : clauses_)
        if (c.live) out.push_back(c.lits);
    return out;
}

void ConflictStore::clear() {
    clauses_.clear();
    for (auto& v : by_var_) v.clear();
    seen_.clear();
    stats_.live = 0;
    stats_.bytes = 0;
}

void ConflictStore::index_clause(std::size_t c) {
    for (const auto& l : clauses_[c].lits)
        by_var_[sz(l.var)].push_back(static_cast<std::uint32_t>(c));
}

bool ConflictStore::add(std::vector<ConflictLiteral> lits,
                        const std::vector<f64>& root_lo,
                        const std::vector<f64>& root_hi) {
    return add_typed(std::move(lits), root_lo, root_hi) == ClauseResult::Inserted;
}

ClauseResult ConflictStore::add_typed(std::vector<ConflictLiteral> lits,
                                      const std::vector<f64>& root_lo,
                                      const std::vector<f64>& root_hi,
                                      std::vector<ConflictLiteral>* normalized) {
    if (normalized != nullptr) normalized->clear();
    // Defensive bound on normalisation work only; the STORAGE policy (max_len_)
    // is applied to the normalised clause below.
    const std::size_t input_limit = std::max<std::size_t>(4096, 16 * max_len_);
    if (n_ <= 0 || lits.empty() || lits.size() > input_limit) {
        ++stats_.rejected;
        return ClauseResult::Rejected;
    }
    // Normalise: integral bounds on in-range columns; merge per (var, side)
    // keeping the WEAKER literal of a disjunction (x <= 3 or x <= 5 is x <= 5);
    // drop literals the root box already excludes; refuse a vacuous clause.
    std::vector<ConflictLiteral> out;
    for (const auto& l : lits) {
        if (l.var < 0 || l.var >= n_ || !std::isfinite(l.bound) ||
            l.bound != std::trunc(l.bound)) {
            ++stats_.rejected;
            return ClauseResult::Rejected;
        }
        const ConflictLiteral probe = l;
        if (lit_true(probe, root_lo, root_hi)) {   // always true: vacuous
            ++stats_.rejected;
            return ClauseResult::Redundant;
        }
        if (lit_false(probe, root_lo, root_hi)) continue;   // never true
        bool merged = false;
        for (auto& o : out)
            if (o.var == l.var && o.upper == l.upper) {
                o.bound = l.upper ? std::max(o.bound, l.bound)
                                  : std::min(o.bound, l.bound);
                merged = true;
                break;
            }
        if (!merged) out.push_back(l);
    }
    if (out.empty()) {
        // Every literal is false at the root: the model has no point that
        // satisfies what this clause was learned under. That is a fact for the
        // caller to draw from its own evidence, not something to record.
        ++stats_.rejected;
        return ClauseResult::Contradiction;
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        if (a.var != b.var) return a.var < b.var;
        if (a.upper != b.upper) return a.upper < b.upper;
        return a.bound < b.bound;
    });
    if (normalized != nullptr) *normalized = out;
    if (out.size() == 1) return ClauseResult::Unit;
    if (out.size() > max_len_) {
        ++stats_.rejected;
        return ClauseResult::Rejected;
    }
    if (out.size() * sizeof(ConflictLiteral) + sizeof(Clause) > budget_) {
        ++stats_.rejected;
        return ClauseResult::Rejected;
    }
    const std::uint64_t h = hash_lits(out);
    const auto it = seen_.find(h);
    if (it != seen_.end() && clauses_[it->second].live &&
        clauses_[it->second].lits == out) {
        ++stats_.rejected;
        return ClauseResult::AlreadyPresent;
    }
    Clause c;
    c.lits = std::move(out);
    c.hash = h;
    c.born = ++tick_;
    stats_.bytes += c.lits.size() * sizeof(ConflictLiteral) + sizeof(Clause);
    clauses_.push_back(std::move(c));
    const std::size_t idx = clauses_.size() - 1;
    seen_[h] = idx;
    index_clause(idx);
    ++stats_.added;
    ++stats_.live;
    if (stats_.bytes > budget_) {
        const auto born = tick_;
        evict_to_budget();
        if (std::none_of(clauses_.begin(), clauses_.end(),
                         [&](const Clause& cl) { return cl.born == born; })) {
            ++stats_.rejected;
            return ClauseResult::Rejected;
        }
    }
    return ClauseResult::Inserted;
}

void ConflictStore::evict_to_budget() {
    // Least active first, oldest first among equals, down to 3/4 of budget.
    std::vector<std::size_t> order;
    for (std::size_t c = 0; c < clauses_.size(); ++c)
        if (clauses_[c].live) order.push_back(c);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (clauses_[a].activity != clauses_[b].activity)
            return clauses_[a].activity < clauses_[b].activity;
        return clauses_[a].born < clauses_[b].born;
    });
    const std::size_t target = budget_ / 4 * 3;
    for (const std::size_t c : order) {
        if (stats_.bytes <= target) break;
        Clause& cl = clauses_[c];
        stats_.bytes -= std::min(stats_.bytes,
                                 cl.lits.size() * sizeof(ConflictLiteral) + sizeof(Clause));
        cl.live = false;
        cl.lits.clear();
        cl.lits.shrink_to_fit();
        --stats_.live;
        ++stats_.evicted;
    }
    // Rebuild the compact form and the indexes.
    std::vector<Clause> kept;
    kept.reserve(stats_.live);
    for (auto& c : clauses_)
        if (c.live) kept.push_back(std::move(c));
    clauses_ = std::move(kept);
    for (auto& v : by_var_) v.clear();
    seen_.clear();
    for (std::size_t c = 0; c < clauses_.size(); ++c) {
        seen_[clauses_[c].hash] = c;
        index_clause(c);
    }
}

bool ConflictStore::propagate(std::vector<f64>& lo, std::vector<f64>& hi,
                              const std::vector<f64>& root_lo,
                              const std::vector<f64>& root_hi, bool* converged) {
    if (converged != nullptr) *converged = true;
    ++stats_.scans;
    if (clauses_.empty()) return true;
    // Back off while the store is not paying: a long clause almost never
    // becomes unit, and each scan costs time at every node. After 2000 scans
    // with fewer than one useful outcome per thousand, scan one node in 16
    // (still sound: skipping a scan only forgoes deductions).
    if (stats_.scans > 2000 &&
        (stats_.propagations + stats_.conflicts) * 1000 < stats_.scans &&
        (stats_.scans & 15u) != 0) {
        if (converged != nullptr) *converged = false;
        return true;
    }
    // A clause can only force or conflict when some literal is false, which
    // needs a column whose box is tighter than the root's. Start from those
    // columns and follow the bounds the clauses themselves move.
    // Scratch is kept between calls: this runs at every node.
    std::vector<Index>& queue = scratch_queue_;
    queue.clear();
    if (queued_.size() != sz(n_)) queued_.assign(sz(n_), 0);
    std::vector<char>& queued = queued_;
    for (Index j = 0; j < n_; ++j)
        if (!by_var_[sz(j)].empty() &&
            (lo[sz(j)] > root_lo[sz(j)] + kTol || hi[sz(j)] < root_hi[sz(j)] - kTol)) {
            queue.push_back(j);
            queued[sz(j)] = 1;
        }
    std::size_t head = 0;
    if (seen_stamp_.size() < clauses_.size()) seen_stamp_.resize(clauses_.size(), 0);
    std::vector<std::uint64_t>& seen_stamp = seen_stamp_;
    std::uint64_t& stamp = stamp_;
    std::uint64_t budget = 8ull * clauses_.size() + 1000ull;
    while (head < queue.size()) {
        const Index j = queue[head++];
        queued[sz(j)] = 0;
        ++stamp;
        for (const std::uint32_t ci : by_var_[sz(j)]) {
            if (seen_stamp[ci] == stamp) continue;
            seen_stamp[ci] = stamp;
            if (budget == 0) {
                if (converged != nullptr) *converged = false;
                finish_scan(queue, head);
                return true;
            }
            --budget;
            Clause& cl = clauses_[ci];
            const ConflictLiteral* open = nullptr;
            std::size_t n_open = 0;
            bool satisfied = false;
            for (const auto& l : cl.lits) {
                if (lit_true(l, lo, hi)) { satisfied = true; break; }
                if (!lit_false(l, lo, hi)) {
                    if (++n_open == 1) open = &l;
                    else break;
                }
            }
            if (satisfied || n_open >= 2) continue;
            ++cl.activity;
            if (n_open == 0) {
                ++stats_.conflicts;
                finish_scan(queue, head);
                return false;
            }
            // Exactly one literal can still hold: it must.
            ++stats_.propagations;
            const Index v = open->var;
            if (open->upper) {
                if (open->bound < hi[sz(v)] - kTol) {
                    hi[sz(v)] = open->bound;
                    ++stats_.tightenings;
                }
            } else if (open->bound > lo[sz(v)] + kTol) {
                lo[sz(v)] = open->bound;
                ++stats_.tightenings;
            }
            if (lo[sz(v)] > hi[sz(v)] + kTol) {
                ++stats_.conflicts;
                finish_scan(queue, head);
                return false;
            }
            if (!queued[sz(v)]) { queued[sz(v)] = 1; queue.push_back(v); }
        }
    }
    finish_scan(queue, head);
    return true;
}

// Leaves the reusable "queued" flags all clear.
void ConflictStore::finish_scan(const std::vector<Index>& queue, std::size_t head) {
    for (std::size_t q = head; q < queue.size(); ++q) queued_[sz(queue[q])] = 0;
}

std::vector<ConflictLiteral> negated_branch_decisions(
    const PropTrail& trail, const std::vector<char>& is_int,
    const std::vector<f64>& root_lo, const std::vector<f64>& root_hi) {
    struct Side { bool have = false; f64 bound = 0.0; };
    std::vector<std::pair<Index, std::pair<Side, Side>>> per;   // var -> (upper, lower)
    const auto slot = [&](Index v) -> std::pair<Side, Side>& {
        for (auto& p : per)
            if (p.first == v) return p.second;
        per.push_back({v, {}});
        return per.back().second;
    };
    for (const auto& e : trail.entries()) {
        if (e.kind != ReasonKind::Branch) continue;
        if (e.var < 0 || sz(e.var) >= is_int.size() || !is_int[sz(e.var)] ||
            !std::isfinite(e.new_bound) || e.new_bound != std::trunc(e.new_bound))
            return {};
        auto& s = slot(e.var);
        if (e.dir == BoundDir::Upper) {
            // decision x <= b: keep the smallest b
            if (!s.first.have || e.new_bound < s.first.bound) s.first = {true, e.new_bound};
        } else {
            if (!s.second.have || e.new_bound > s.second.bound) s.second = {true, e.new_bound};
        }
    }
    std::vector<ConflictLiteral> lits;
    for (const auto& p : per) {
        const Index v = p.first;
        if (p.second.first.have) {           // decision x <= b  ->  x >= b + 1
            const f64 b = p.second.first.bound;
            if (b < root_hi[sz(v)] - kTol)
                lits.push_back({v, false, b + 1.0});
        }
        if (p.second.second.have) {          // decision x >= b  ->  x <= b - 1
            const f64 b = p.second.second.bound;
            if (b > root_lo[sz(v)] + kTol)
                lits.push_back({v, true, b - 1.0});
        }
    }
    return lits;
}

std::vector<ConflictLiteral> explain_conflict_clause(
    const PropTrail& trail, const model::LpProblem& lp, Index n_global_rows,
    const std::vector<char>& is_int, const std::vector<f64>& root_lo,
    const std::vector<f64>& root_hi, const std::vector<f64>& cur_lo,
    const std::vector<f64>& cur_hi, Index conflict_var) {
    const auto& es = trail.entries();
    const Index n = lp.n_cols();
    if (conflict_var < 0 || conflict_var >= n || es.empty()) return {};
    // Series of trail entries per (column, side), chronological.
    // side 0 = Lower, 1 = Upper.
    std::vector<std::vector<std::uint32_t>> series(static_cast<std::size_t>(2 * n));
    for (std::size_t k = 0; k < es.size(); ++k) {
        const auto& e = es[k];
        if (e.var < 0 || e.var >= n) return {};
        series[2 * sz(e.var) + (e.dir == BoundDir::Upper ? 1 : 0)].push_back(
            static_cast<std::uint32_t>(k));
    }
    // A series is trustworthy when it chains from the root bound to the box's
    // current bound with no untraced move in between.
    std::vector<signed char> chain(static_cast<std::size_t>(2 * n), -1);   // -1 unknown
    const auto chained = [&](Index v, int side) -> bool {
        signed char& c = chain[2 * sz(v) + side];
        if (c >= 0) return c == 1;
        const auto& sr = series[2 * sz(v) + side];
        const f64 root = side == 0 ? root_lo[sz(v)] : root_hi[sz(v)];
        const f64 cur = side == 0 ? cur_lo[sz(v)] : cur_hi[sz(v)];
        bool ok = true;
        f64 prev = root;
        for (const std::uint32_t k : sr) {
            if (std::fabs(es[k].old_bound - prev) > kTol) { ok = false; break; }
            prev = es[k].new_bound;
        }
        if (ok && std::fabs(prev - cur) > kTol) ok = false;
        c = ok ? 1 : 0;
        return ok;
    };
    // Index of the last entry of (v, side) strictly before `before`, -1 for
    // "the root bound was in force".
    const auto latest_before = [&](Index v, int side, std::size_t before) -> long {
        const auto& sr = series[2 * sz(v) + side];
        long found = -1;
        for (const std::uint32_t k : sr) {
            if (k >= before) break;
            found = static_cast<long>(k);
        }
        return found;
    };
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    std::vector<char> visited(es.size(), 0);
    std::vector<std::size_t> work;
    std::vector<std::size_t> decisions;
    const auto need = [&](Index v, int side, std::size_t before) -> bool {
        if (!chained(v, side)) return false;
        const long k = latest_before(v, side, before);
        if (k >= 0 && !visited[sz(static_cast<Index>(k))]) {
            visited[sz(static_cast<Index>(k))] = 1;
            work.push_back(static_cast<std::size_t>(k));
        }
        return true;
    };
    // The emptied domain: its lower and its upper bound.
    if (!need(conflict_var, 0, es.size()) || !need(conflict_var, 1, es.size()))
        return {};
    while (!work.empty()) {
        const std::size_t k = work.back();
        work.pop_back();
        const auto& e = es[k];
        if (e.kind == ReasonKind::Branch) {
            decisions.push_back(k);
            continue;
        }
        if (e.kind != ReasonKind::Row || e.reason_id < 0 ||
            e.reason_id >= n_global_rows || e.reason_id >= lp.n_rows())
            return {};
        const Index r = e.reason_id;
        // Coefficient of the tightened column in its reason row.
        f64 a_v = 0.0;
        for (core::Offset q = rp[sz(r)]; q < rp[sz(r) + 1]; ++q)
            if (ci[sz(q)] == e.var) a_v += av[sz(q)];
        if (a_v == 0.0) return {};
        // Upper bound from a positive coefficient (or lower from a negative
        // one) comes from the row's minimum activity; the other two cases from
        // its maximum. Minimum activity uses lower bounds of positive
        // coefficients and upper bounds of negative ones; maximum the reverse.
        const bool use_min = (e.dir == BoundDir::Upper) == (a_v > 0.0);
        for (core::Offset q = rp[sz(r)]; q < rp[sz(r) + 1]; ++q) {
            const Index c = ci[sz(q)];
            if (c == e.var || av[sz(q)] == 0.0) continue;
            const bool positive = av[sz(q)] > 0.0;
            const int side = (use_min == positive) ? 0 : 1;   // 0: lower, 1: upper
            if (!need(c, side, k)) return {};
        }
    }
    if (decisions.empty()) return {};
    // Negate the decisions found: tightest per (column, side).
    std::vector<ConflictLiteral> lits;
    for (const std::size_t k : decisions) {
        const auto& e = es[k];
        if (sz(e.var) >= is_int.size() || !is_int[sz(e.var)] ||
            !std::isfinite(e.new_bound) || e.new_bound != std::trunc(e.new_bound))
            return {};
        ConflictLiteral l;
        l.var = e.var;
        if (e.dir == BoundDir::Upper) {        // decision x <= b  ->  x >= b + 1
            l.upper = false;
            l.bound = e.new_bound + 1.0;
            if (e.new_bound >= root_hi[sz(e.var)] - kTol) continue;
        } else {                               // decision x >= b  ->  x <= b - 1
            l.upper = true;
            l.bound = e.new_bound - 1.0;
            if (e.new_bound <= root_lo[sz(e.var)] + kTol) continue;
        }
        bool merged = false;
        for (auto& o : lits)
            if (o.var == l.var && o.upper == l.upper) {
                // Several decisions on one side: their conjunction is the
                // tightest one, and its negation is the disjunction of the
                // negations, i.e. the weakest literal (largest upper bound,
                // smallest lower bound).
                o.bound = l.upper ? std::max(o.bound, l.bound) : std::min(o.bound, l.bound);
                merged = true;
                break;
            }
        if (!merged) lits.push_back(l);
    }
    return lits;
}

}  // namespace sor::search

namespace sor::search {

std::vector<ConflictLiteral> farkas_conflict_clause(
    const model::LpProblem& lp, const std::vector<f64>& ray, f64 tolerance,
    const std::vector<char>& is_int, const std::vector<f64>& root_lo,
    const std::vector<f64>& root_hi, std::size_t* relaxed_bounds) {
    std::vector<ConflictLiteral> out;
    const Index n = lp.n_cols();
    if (!(tolerance >= 0.0) || !std::isfinite(tolerance) ||
        static_cast<Index>(root_lo.size()) != n || static_cast<Index>(root_hi.size()) != n ||
        static_cast<Index>(is_int.size()) != n)
        return out;
    const auto checked = certify::check_dual_farkas_ray(lp, ray, tolerance);
    if (!checked.certified || checked.multipliers.size() != static_cast<std::size_t>(lp.n_rows()))
        return out;
    const auto sz = [](Index v) { return static_cast<std::size_t>(v); };

    // d = A'y with the ray the checker actually certified (normalised, repaired).
    std::vector<long double> d(sz(n), 0.0L);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const f64 y = checked.multipliers[sz(i)];
        if (y == 0.0) continue;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            d[sz(ci[sz(k)])] += static_cast<long double>(y) * lp.A.vals[sz(k)];
    }
    // Slack of the certificate, kept conservative: half of what the checker
    // measured beyond its own tolerance, so round-off in this recomputation
    // cannot turn a relaxed bound into an invalid clause.
    long double lower = 0.0L;
    for (Index j = 0; j < n; ++j) {
        if (d[sz(j)] > 0.0L) lower += d[sz(j)] * lp.col_lo[sz(j)];
        else if (d[sz(j)] < 0.0L) lower += d[sz(j)] * lp.col_hi[sz(j)];
    }
    long double upper = 0.0L;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const f64 y = checked.multipliers[sz(i)];
        if (y > 0.0) upper += static_cast<long double>(y) * lp.row_hi[sz(i)];
        else if (y < 0.0) upper += static_cast<long double>(y) * lp.row_lo[sz(i)];
    }
    long double remaining = 0.5L * (lower - upper) -
        static_cast<long double>(tolerance) *
            (1.0L + std::fabs(static_cast<double>(lower)) + std::fabs(static_cast<double>(upper)));
    if (!(remaining > 0.0L) || !std::isfinite(static_cast<double>(remaining))) return out;

    struct Tight {
        Index col;
        bool lower_bound;
        f64 room, bound;
        long double coef;
        bool removed = false;
    };
    std::vector<Tight> tight;
    for (Index j = 0; j < n; ++j) {
        const long double dj = d[sz(j)];
        if (dj == 0.0L) continue;
        if (dj > 0.0L) {
            const f64 b = lp.col_lo[sz(j)];
            if (!std::isfinite(b)) return {};
            if (b > root_lo[sz(j)] + 1e-9) {
                if (!is_int[sz(j)] || std::fabs(b - std::round(b)) > 1e-9) return {};
                tight.push_back({j, true, std::round(b) - root_lo[sz(j)], std::round(b), dj});
            }
        } else {
            const f64 b = lp.col_hi[sz(j)];
            if (!std::isfinite(b)) return {};
            if (b < root_hi[sz(j)] - 1e-9) {
                if (!is_int[sz(j)] || std::fabs(b - std::round(b)) > 1e-9) return {};
                tight.push_back({j, false, root_hi[sz(j)] - std::round(b), std::round(b), -dj});
            }
        }
    }
    if (tight.empty()) return out;
    // Minimise the number of reasons first. Removing one bound costs its
    // coefficient TIMES its full distance to the root, not just its coefficient.
    // Spending the margin on part of a distant, small-coefficient bound can
    // prevent several nearby reasons from disappearing. Infinite-root bounds
    // cannot disappear, but can still be weakened in the second pass.
    std::stable_sort(tight.begin(), tight.end(),
                     [](const Tight& a, const Tight& b) {
                         return a.coef * a.room < b.coef * b.room;
                     });
    std::size_t relaxed = 0;
    for (Tight& t : tight) {
        const long double cost = t.coef * t.room;
        if (std::isfinite(cost) && cost <= remaining) {
            remaining -= cost;
            t.removed = true;
            ++relaxed;
        }
    }
    std::stable_sort(tight.begin(), tight.end(),
                     [](const Tight& a, const Tight& b) { return a.coef < b.coef; });
    for (const Tight& t : tight) {
        if (t.removed) continue;
        const long double units = std::floor(std::min<long double>(t.room, remaining / t.coef));
        const f64 take = static_cast<f64>(std::max(0.0L, units));
        remaining -= t.coef * take;
        const f64 b = t.lower_bound ? t.bound - take : t.bound + take;
        // Complements need a representable adjacent integer. Outside that
        // range, the caller retains its ordinary decision explanation.
        if (!std::isfinite(b) || std::fabs(b) >= 9007199254740991.0) return {};
        // lower bound x >= b  ->  literal x <= b - 1 ;  upper bound x <= b  ->  x >= b + 1
        out.push_back({t.col, t.lower_bound, t.lower_bound ? b - 1.0 : b + 1.0});
    }
    if (out.empty()) return out;
    // Verify precisely the box described by the complement of the clause.
    // Every OTHER bound is a root bound, including continuous columns and
    // sides unused by the initial ray. No sensitivity estimate or rounding
    // decision is by itself permission to publish a learned clause.
    model::LpProblem explanation = lp;
    explanation.col_lo = root_lo;
    explanation.col_hi = root_hi;
    for (const ConflictLiteral& l : out) {
        if (l.upper) explanation.col_lo[sz(l.var)] = std::max(root_lo[sz(l.var)], l.bound + 1.0);
        else explanation.col_hi[sz(l.var)] = std::min(root_hi[sz(l.var)], l.bound - 1.0);
    }
    if (!certify::check_dual_farkas_ray(explanation, checked.multipliers, tolerance).certified)
        return {};
    if (relaxed_bounds != nullptr) *relaxed_bounds += relaxed;
    return out;
}

}  // namespace sor::search
