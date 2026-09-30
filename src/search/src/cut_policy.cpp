#include "sor/search/cut_policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

struct CandUnit {
    CutRow cut;
    std::vector<f64> unit_vals;
    f64 rhs_unit = 0.0;
    f64 efficacy = 0.0;
    f64 score = 0.0;
};

bool normalized_le(const CutRow& cut, std::vector<f64>& unit_vals,
                   f64& rhs_unit) {
    if (cut.cols.size() != cut.vals.size() || cut.cols.empty()) return false;
    const bool has_lo = std::isfinite(cut.row_lo);
    const bool has_hi = std::isfinite(cut.row_hi);
    if (has_lo == has_hi) return false;
    const f64 sign = has_hi ? 1.0 : -1.0;
    f64 norm2 = 0.0;
    for (const f64 v : cut.vals) {
        if (!std::isfinite(v)) return false;
        norm2 += v * v;
    }
    if (!(norm2 > 0.0)) return false;
    const f64 norm = std::sqrt(norm2);
    unit_vals.resize(cut.vals.size());
    for (std::size_t k = 0; k < cut.vals.size(); ++k)
        unit_vals[k] = sign * cut.vals[k] / norm;
    rhs_unit = sign * (has_hi ? cut.row_hi : cut.row_lo) / norm;
    return std::isfinite(rhs_unit);
}

struct ScoreCtx {
    Index n_cols = 0;
    std::vector<f64> obj_unit;
    std::vector<bool> is_integer;
    std::vector<f64> locks;
    f64 max_locks = 0.0;
};

ScoreCtx build_score_ctx(const model::LpProblem& lp) {
    ScoreCtx ctx;
    ctx.n_cols = lp.n_cols();
    ctx.is_integer = lp.is_integer;
    ctx.obj_unit.assign(sz(lp.n_cols()), 0.0);
    f64 norm = 0.0;
    for (const f64 v : lp.c) norm += v * v;
    if (norm > 0.0) {
        norm = std::sqrt(norm);
        for (std::size_t j = 0; j < lp.c.size() && j < ctx.obj_unit.size(); ++j)
            ctx.obj_unit[j] = lp.c[j] / norm;
    } else {
        ctx.obj_unit.clear();
    }
    ctx.locks.assign(sz(lp.n_cols()), 0.0);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const bool has_lo = std::isfinite(lp.row_lo[sz(i)]);
        const bool has_hi = std::isfinite(lp.row_hi[sz(i)]);
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 a = av[sz(k)];
            if (a == 0.0) continue;
            const Index j = ci[sz(k)];
            if ((a > 0.0 && has_lo) || (a < 0.0 && has_hi)) ctx.locks[sz(j)] += 1.0;
            if ((a > 0.0 && has_hi) || (a < 0.0 && has_lo)) ctx.locks[sz(j)] += 1.0;
        }
    }
    for (const f64 v : ctx.locks) ctx.max_locks = std::max(ctx.max_locks, v);
    return ctx;
}

f64 composite_score(const CandUnit& c, const ScoreCtx& ctx, f64 max_eff,
                    const CutOptions& opts) {
    f64 score = max_eff > 0.0 ? c.efficacy / max_eff : 0.0;
    if (!ctx.obj_unit.empty()) {
        f64 dot = 0.0;
        for (std::size_t k = 0; k < c.cut.cols.size(); ++k) {
            const Index j = c.cut.cols[k];
            if (j >= 0 && sz(j) < ctx.obj_unit.size())
                dot += c.unit_vals[k] * ctx.obj_unit[sz(j)];
        }
        score += opts.pool_weight_objective_parallelism * std::fabs(dot);
    }
    if (!ctx.is_integer.empty() && !c.cut.cols.empty()) {
        std::size_t ints = 0;
        for (const Index j : c.cut.cols)
            if (j >= 0 && sz(j) < ctx.is_integer.size() && ctx.is_integer[sz(j)])
                ++ints;
        score += opts.pool_weight_integer_support *
                 (static_cast<f64>(ints) / static_cast<f64>(c.cut.cols.size()));
    }
    if (ctx.n_cols > 0) {
        const f64 density =
            static_cast<f64>(c.cut.cols.size()) / static_cast<f64>(ctx.n_cols);
        const f64 end = std::max(1e-9, opts.pool_sparsity_end_density);
        score += opts.pool_weight_sparsity * std::max(0.0, 1.0 - density / end);
    }
    if (ctx.max_locks > 0.0 && !c.cut.cols.empty()) {
        f64 sum = 0.0;
        for (const Index j : c.cut.cols)
            if (j >= 0 && sz(j) < ctx.locks.size()) sum += ctx.locks[sz(j)];
        const f64 mean = sum / static_cast<f64>(c.cut.cols.size());
        score += opts.pool_weight_low_locks *
                 std::max(0.0, 1.0 - mean / ctx.max_locks);
    }
    return score;
}

}  // namespace

void apply_auto_cuts_policy(BabOptions& o) {
    if (!milp_policy_is_latest(o.policy)) return;

    CutOptions& c = o.cut;
    // Turner et al. consensus parameters (arXiv:2307.07322 §2).
    c.pool_max_density = 0.45;
    c.pool_min_dense_nnz = 200;
    c.pool_nnz_budget_factor = 1.0;
    c.pool_min_nnz_budget = 200;
    c.pool_parallel_hard_filter = false;
    c.pool_parallelism_penalty = 0.5;
    c.pool_parallelism_penalty_min = 0.3;
    c.pool_weight_sparsity = 0.15;
    c.pool_weight_low_locks = 0.1;
    c.pool_weight_objective_parallelism = 0.2;
    c.pool_sparsity_end_density = 0.4;
    c.max_cuts_per_round = std::min(c.max_cuts_per_round, 120);
    c.max_candidates_per_round = std::min(c.max_candidates_per_round, 400);

    o.mir_cuts = true;
    o.lifted_cover_cuts = true;
    o.zerohalf_cuts = true;
    // Flow-cover derivation has a known, gross counterexample on blend2.
    // Auto policy must never make an unverified separator available.
    o.flow_cover_cuts = false;
    // Clique cuts: valid but regressed misc03 node LP cost; keep opt-in only.
}

std::vector<CutRow> filter_cut_candidates_for_round(
    std::vector<CutRow> candidates, const model::LpProblem& lp,
    const std::vector<f64>& x, const CutOptions& opts,
    CutFilterStats* stats) {
    CutFilterStats local;
    CutFilterStats& st = stats != nullptr ? *stats : local;
    if (candidates.empty()) return candidates;
    const int cap = opts.max_candidates_per_round;
    if (cap <= 0) {
        st.rejected_budget += candidates.size();
        return {};
    }

    const ScoreCtx ctx = build_score_ctx(lp);
    std::vector<CandUnit> pool;
    pool.reserve(candidates.size());
    f64 max_eff = 0.0;

    const std::size_t dense_nnz =
        ctx.n_cols > 0 && opts.pool_max_density > 0.0
            ? std::max<std::size_t>(
                  opts.pool_min_dense_nnz,
                  static_cast<std::size_t>(opts.pool_max_density *
                                           static_cast<f64>(ctx.n_cols)))
            : std::numeric_limits<std::size_t>::max();

    for (auto& cut : candidates) {
        CandUnit c;
        c.cut = std::move(cut);
        if (!normalized_le(c.cut, c.unit_vals, c.rhs_unit)) {
            ++st.rejected_malformed;
            continue;
        }
        if (c.cut.cols.size() > dense_nnz) {
            ++st.rejected_dense;
            continue;
        }
        f64 activity = 0.0;
        bool ok = true;
        for (std::size_t k = 0; k < c.cut.cols.size(); ++k) {
            const Index j = c.cut.cols[k];
            if (j < 0 || sz(j) >= x.size()) {
                ok = false;
                break;
            }
            activity += c.unit_vals[k] * x[sz(j)];
        }
        if (!ok) {
            ++st.rejected_malformed;
            continue;
        }
        c.efficacy = activity - c.rhs_unit;
        if (c.efficacy < opts.pool_efficacy_min) {
            ++st.rejected_efficacy;
            continue;
        }
        max_eff = std::max(max_eff, c.efficacy);
        pool.push_back(std::move(c));
    }
    if (pool.empty()) return {};

    for (auto& c : pool) c.score = composite_score(c, ctx, max_eff, opts);

    // The prefilter screens for validity and numerics and caps the candidate
    // list generously; it does NOT take the selection decisions. Diversity
    // (parallelism penalties) and the nonzero budget are applied once, by the
    // selecting CutPool, which sees every survivor. Only exact / near
    // duplicates (cosine above pool_parallelism_max) are dropped here, since
    // the pool can gain nothing from two copies of one cut.
    //
    // (This function used to run its own greedy selection and computed each
    // pick's parallelism AFTER moving the pick into the output: the moved-from
    // cut had no support, every cosine was 0, and no parallel candidate was
    // ever rejected.)
    std::vector<std::size_t> order(pool.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return pool[a].score > pool[b].score;
    });
    const std::size_t keep_cap =
        static_cast<std::size_t>(cap) * static_cast<std::size_t>(std::max(1, opts.prefilter_cap_factor));
    std::vector<std::size_t> accepted;
    accepted.reserve(std::min(order.size(), keep_cap));
    // Duplicate detection is EXACT (canonical form), not a cosine threshold:
    // both rows are put in one sense (<=, unit norm by positive scaling), and
    // two rows are the same inequality only when they have the same support and
    // the same normalised coefficients. Then the one with the smaller
    // normalised right-hand side is the stronger and is kept; the other is
    // dominated. Opposite-facing rows (x <= 0 and x >= 1) have opposite
    // normalised coefficients and are never duplicates, and neither are
    // near-parallel independent rows: how much a pick should discourage
    // similar candidates is the pool's soft score, not a rejection here.
    const auto same_lhs = [&](const CandUnit& a, const CandUnit& b) {
        if (a.cut.cols.size() != b.cut.cols.size()) return false;
        for (std::size_t k = 0; k < a.cut.cols.size(); ++k) {
            if (a.cut.cols[k] != b.cut.cols[k]) return false;
            if (std::fabs(a.unit_vals[k] - b.unit_vals[k]) > 1e-9) return false;
        }
        return true;
    };
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> by_support;   // hash -> slots in `accepted`
    const auto support_hash = [&](const CandUnit& c) {
        std::uint64_t h = 1469598103934665603ull;
        for (const Index j : c.cut.cols) { h ^= static_cast<std::uint64_t>(j) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); h *= 1099511628211ull; }
        return h;
    };
    std::uint64_t duplicates = 0, dominated = 0;
    for (const std::size_t i : order) {
        if (accepted.size() >= keep_cap) break;
        auto& slots = by_support[support_hash(pool[i])];
        bool handled = false;
        for (const std::size_t slot : slots) {
            const std::size_t a = accepted[slot];
            if (!same_lhs(pool[i], pool[a])) continue;
            handled = true;
            if (pool[i].rhs_unit < pool[a].rhs_unit - 1e-12) {
                accepted[slot] = i;   // strictly stronger: replaces the kept one
                ++dominated;
            } else if (pool[i].rhs_unit > pool[a].rhs_unit + 1e-12) {
                ++dominated;
            } else {
                ++duplicates;
            }
            break;
        }
        if (handled) continue;
        slots.push_back(accepted.size());
        accepted.push_back(i);
    }
    std::vector<CutRow> out;
    out.reserve(accepted.size());
    for (const std::size_t i : accepted) out.push_back(std::move(pool[i].cut));
    st.rejected_duplicate += duplicates;
    st.rejected_dominated += dominated;
    st.rejected_budget += pool.size() - accepted.size() - duplicates - dominated;
    return out;
}

}  // namespace sor::search
