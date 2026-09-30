#include "sor/search/cut_policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

f64 sparse_dot(const CutRow& a, const std::vector<f64>& av, const CutRow& b,
               const std::vector<f64>& bv) {
    std::size_t p = 0, q = 0;
    f64 dot = 0.0;
    while (p < a.cols.size() && q < b.cols.size()) {
        if (a.cols[p] == b.cols[q]) {
            dot += av[p] * bv[q];
            ++p;
            ++q;
        } else if (a.cols[p] < b.cols[q]) {
            ++p;
        } else {
            ++q;
        }
    }
    return dot;
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
    const std::vector<f64>& x, const CutOptions& opts) {
    if (candidates.empty()) return candidates;
    const int cap = opts.max_candidates_per_round;
    if (cap <= 0) return {};

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
        if (!normalized_le(c.cut, c.unit_vals, c.rhs_unit)) continue;
        if (c.cut.cols.size() > dense_nnz) continue;
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
        if (!ok) continue;
        c.efficacy = activity - c.rhs_unit;
        if (c.efficacy < opts.pool_efficacy_min) continue;
        max_eff = std::max(max_eff, c.efficacy);
        pool.push_back(std::move(c));
    }
    if (pool.empty()) return {};

    for (auto& c : pool) c.score = composite_score(c, ctx, max_eff, opts);

    const f64 nnz_budget =
        opts.pool_nnz_budget_factor > 0.0 && ctx.n_cols > 0
            ? std::max(static_cast<f64>(opts.pool_min_nnz_budget),
                       opts.pool_nnz_budget_factor * static_cast<f64>(ctx.n_cols))
            : static_cast<f64>(cap) * static_cast<f64>(ctx.n_cols);

    std::vector<char> alive(pool.size(), 1);
    std::vector<CutRow> out;
    f64 nnz_used = 0.0;
    constexpr f64 kStopScore = 0.04;

    while (static_cast<int>(out.size()) < cap) {
        std::size_t best = pool.size();
        f64 best_score = -std::numeric_limits<f64>::infinity();
        for (std::size_t i = 0; i < pool.size(); ++i) {
            if (!alive[i]) continue;
            if (nnz_used + static_cast<f64>(pool[i].cut.cols.size()) > nnz_budget)
                continue;
            if (pool[i].score > best_score) {
                best_score = pool[i].score;
                best = i;
            }
        }
        if (best == pool.size() || best_score < kStopScore) break;

        alive[best] = 0;
        nnz_used += static_cast<f64>(pool[best].cut.cols.size());
        out.push_back(std::move(pool[best].cut));

        for (std::size_t i = 0; i < pool.size(); ++i) {
            if (!alive[i]) continue;
            const f64 cosine = std::fabs(
                sparse_dot(pool[i].cut, pool[i].unit_vals, pool[best].cut,
                           pool[best].unit_vals));
            if (opts.pool_parallel_hard_filter) {
                if (cosine > opts.pool_parallelism_max) alive[i] = 0;
                continue;
            }
            if (cosine < opts.pool_parallelism_penalty_min) continue;
            pool[i].score -= opts.pool_parallelism_penalty * cosine;
            if (pool[i].score <= 0.0) alive[i] = 0;
        }
    }
    return out;
}

}  // namespace sor::search
