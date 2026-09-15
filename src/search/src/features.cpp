#include "sor/search/features.hpp"
#include "sor/search/cuts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline f64 safe_div(f64 a, f64 b) {
    const f64 d = std::fabs(b) < 1e-12 ? 1.0 : b;
    return a / d;
}

inline f64 frac_part(f64 v) {
    const f64 f = v - std::floor(v);
    return std::min(f, 1.0 - f);
}

f64 obj_scale(const model::LpProblem& lp) {
    f64 m = 0.0;
    for (f64 c : lp.c) m = std::max(m, std::fabs(c));
    return 1.0 + m;
}

void coef_stats(const model::LpProblem& lp, Index j,
                f64& mean_abs, f64& max_abs, f64& min_abs, int& nnz) {
    mean_abs = 0.0;
    max_abs = 0.0;
    min_abs = 0.0;
    nnz = 0;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    f64 sum = 0.0;
    f64 mn = std::numeric_limits<f64>::infinity();
    for (Index i = 0; i < lp.n_rows(); ++i) {
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (ci[sz(k)] != j) continue;
            const f64 a = std::fabs(lp.A.vals[sz(k)]);
            if (a <= 1e-15) continue;
            ++nnz;
            sum += a;
            max_abs = std::max(max_abs, a);
            mn = std::min(mn, a);
        }
    }
    if (nnz > 0) {
        mean_abs = sum / static_cast<f64>(nnz);
        min_abs = mn;
    }
}

}  // namespace

int branch_feature_stratum(const BranchFeatureVec& f) noexcept {
    // f[0] = min(down,up) fractionality in [0,0.5]; f[12] = binary flag.
    const f64 frac = std::max(0.0, std::min(0.5, f[0]));
    int bin = 0;
    if (frac >= 0.375) bin = 3;
    else if (frac >= 0.25) bin = 2;
    else if (frac >= 0.125) bin = 1;
    const int binary = (f[12] >= 0.5) ? 1 : 0;
    return std::min(kBranchStratumCount - 1, bin + 4 * binary);
}

bool fill_branch_features(const BranchFeatureContext& ctx,
                          Index j,
                          BranchFeatureVec& out) {
    out.fill(0.0);
    if (!ctx.lp || !ctx.col_lo || !ctx.col_hi || !ctx.x) return false;
    const auto& lp = *ctx.lp;
    if (j < 0 || j >= lp.n_cols()) return false;
    if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) return false;
    const f64 lo = (*ctx.col_lo)[sz(j)];
    const f64 hi = (*ctx.col_hi)[sz(j)];
    if (lo >= hi - 1e-12) return false;
    const f64 xv = (*ctx.x)[sz(j)];
    if (std::fabs(xv - std::round(xv)) <= ctx.int_tol) return false;

    const f64 down = xv - std::floor(xv);
    const f64 up = std::ceil(xv) - xv;
    const f64 frac = std::min(down, up);
    const f64 osc = obj_scale(lp);
    f64 mean_a = 0.0, max_a = 0.0, min_a = 0.0;
    int nnz = 0;
    coef_stats(lp, j, mean_a, max_a, min_a, nnz);
    const f64 degree = (ctx.col_degree && sz(j) < ctx.col_degree->size())
        ? static_cast<f64>((*ctx.col_degree)[sz(j)])
        : static_cast<f64>(nnz);
    const f64 span = hi - lo;
    const bool binary =
        std::fabs(lo) <= ctx.int_tol && std::fabs(hi - 1.0) <= ctx.int_tol;

    f64 pc_d = 0.0, pc_u = 0.0;
    if (ctx.pc_down && ctx.pc_down_count &&
        sz(j) < ctx.pc_down->size() &&
        (*ctx.pc_down_count)[sz(j)] > 0) {
        pc_d = (*ctx.pc_down)[sz(j)];
    }
    if (ctx.pc_up && ctx.pc_up_count &&
        sz(j) < ctx.pc_up->size() &&
        (*ctx.pc_up_count)[sz(j)] > 0) {
        pc_u = (*ctx.pc_up)[sz(j)];
    }
    const f64 pc_score =
        std::min(pc_d * down, pc_u * up) + 0.1 * std::max(pc_d * down, pc_u * up);

    out[0] = frac;
    out[1] = xv;
    out[2] = lp.c[sz(j)] / osc;
    out[3] = safe_div(xv - lo, span);
    out[4] = span / (1.0 + std::fabs(lo) + std::fabs(hi));
    out[5] = pc_d;
    out[6] = pc_u;
    out[7] = pc_score;
    out[8] = safe_div(degree, static_cast<f64>(std::max(Index{1}, lp.n_rows())));
    out[9] = mean_a / (1.0 + max_a);
    out[10] = max_a / (1.0 + max_a);
    out[11] = min_a / (1.0 + max_a);
    out[12] = binary ? 1.0 : 0.0;
    out[13] = std::fabs(lp.c[sz(j)]) / osc;
    out[14] = down;
    out[15] = up;

    // [16..23] bipartite neighbor summaries (Sparse-SB / Gasse spirit).
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    f64 mean_slack = 0.0, mean_dens = 0.0, mean_edge = 0.0, mean_sense = 0.0;
    f64 mean_tight = 0.0, mean_abss = 0.0, max_slack = 0.0;
    int neigh = 0;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        f64 a_ij = 0.0;
        bool hit = false;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (ci[sz(k)] != j) continue;
            a_ij = lp.A.vals[sz(k)];
            hit = true;
            break;
        }
        if (!hit || std::fabs(a_ij) <= 1e-15) continue;
        ++neigh;
        const int rnnz = static_cast<int>(rp[sz(i) + 1] - rp[sz(i)]);
        f64 activity = 0.0;
        f64 abs_sum = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index jj = ci[sz(k)];
            abs_sum += std::fabs(lp.A.vals[sz(k)]);
            if (jj >= 0 && jj < lp.n_cols())
                activity += lp.A.vals[sz(k)] * (*ctx.x)[sz(jj)];
        }
        f64 slack = 0.0;
        if (std::isfinite(lp.row_hi[sz(i)]))
            slack = std::max(slack, activity - lp.row_hi[sz(i)]);
        if (std::isfinite(lp.row_lo[sz(i)]))
            slack = std::max(slack, lp.row_lo[sz(i)] - activity);
        f64 sense = 0.0;
        if (std::isfinite(lp.row_hi[sz(i)]) && !std::isfinite(lp.row_lo[sz(i)]))
            sense = -1.0;
        else if (std::isfinite(lp.row_lo[sz(i)]) && !std::isfinite(lp.row_hi[sz(i)]))
            sense = 1.0;
        else if (std::isfinite(lp.row_lo[sz(i)]) && std::isfinite(lp.row_hi[sz(i)]) &&
                 std::fabs(lp.row_lo[sz(i)] - lp.row_hi[sz(i)]) > 1e-12)
            sense = 0.5;
        const f64 dens =
            static_cast<f64>(rnnz) /
            static_cast<f64>(std::max(Index{1}, lp.n_cols()));
        const f64 rhs =
            std::isfinite(lp.row_hi[sz(i)]) ? lp.row_hi[sz(i)]
            : (std::isfinite(lp.row_lo[sz(i)]) ? lp.row_lo[sz(i)] : 0.0);
        const f64 tight = 1.0 - std::min(1.0, std::fabs(activity - rhs) /
                                                  (1.0 + std::fabs(rhs)));
        mean_slack += slack;
        max_slack = std::max(max_slack, slack);
        mean_dens += dens;
        mean_edge += std::fabs(a_ij) / (1.0 + std::fabs(a_ij));
        mean_sense += sense;
        mean_tight += tight;
        mean_abss += abs_sum / (1.0 + abs_sum);
    }
    if (neigh > 0) {
        const f64 inv = 1.0 / static_cast<f64>(neigh);
        mean_slack *= inv;
        mean_dens *= inv;
        mean_edge *= inv;
        mean_sense *= inv;
        mean_tight *= inv;
        mean_abss *= inv;
    }
    const f64 gap = (ctx.have_incumbent && std::isfinite(ctx.incumbent) &&
                     std::isfinite(ctx.dual_bound))
                        ? std::fabs(ctx.incumbent - ctx.dual_bound) /
                              (1.0 + std::fabs(ctx.incumbent))
                        : 0.0;
    out[16] = mean_slack;
    out[17] = max_slack;
    out[18] = mean_dens;
    out[19] = mean_edge;
    out[20] = mean_sense;
    out[21] = mean_tight;
    out[22] = mean_abss;
    out[23] = std::min(1.0, static_cast<f64>(ctx.depth) / 64.0) + 0.1 * gap;
    return true;
}

bool fill_lifted_features(const BranchFeatureContext& ctx,
                          Index j,
                          LiftedFeatureVec& out) {
    out.fill(0.0);
    BranchFeatureVec base{};
    if (!fill_branch_features(ctx, j, base)) return false;
    for (int i = 0; i < kBranchFeatureDim; ++i)
        out[static_cast<std::size_t>(i)] = base[static_cast<std::size_t>(i)];

    // Extra lifts: per-candidate message-pass lite (mean/max of neighbor
    // constraint features weighted by |aij|), matching Lifted Branching /
    // Gasse bipartite spirit without a full GCNN.
    if (!ctx.lp || !ctx.x) return true;
    const auto& lp = *ctx.lp;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    f64 w_sum = 0.0;
    f64 lift_obj = 0.0, lift_dual = 0.0, lift_bias = 0.0, lift_age = 0.0;
    f64 lift_tight = 0.0, lift_nnz = 0.0, lift_edge = 0.0, lift_viol = 0.0;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        f64 a_ij = 0.0;
        bool hit = false;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (ci[sz(k)] != j) continue;
            a_ij = lp.A.vals[sz(k)];
            hit = true;
            break;
        }
        if (!hit || std::fabs(a_ij) <= 1e-15) continue;
        const f64 w = std::fabs(a_ij);
        w_sum += w;
        f64 activity = 0.0;
        f64 abs_sum = 0.0;
        f64 obj_dot = 0.0;
        const int rnnz = static_cast<int>(rp[sz(i) + 1] - rp[sz(i)]);
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index jj = ci[sz(k)];
            const f64 a = lp.A.vals[sz(k)];
            abs_sum += std::fabs(a);
            if (jj >= 0 && jj < lp.n_cols()) {
                activity += a * (*ctx.x)[sz(jj)];
                obj_dot += a * lp.c[sz(jj)];
            }
        }
        f64 slack = 0.0;
        if (std::isfinite(lp.row_hi[sz(i)]))
            slack = std::max(slack, activity - lp.row_hi[sz(i)]);
        if (std::isfinite(lp.row_lo[sz(i)]))
            slack = std::max(slack, lp.row_lo[sz(i)] - activity);
        const f64 rhs =
            std::isfinite(lp.row_hi[sz(i)]) ? lp.row_hi[sz(i)]
            : (std::isfinite(lp.row_lo[sz(i)]) ? lp.row_lo[sz(i)] : 0.0);
        const f64 bias = rhs / (1.0 + abs_sum);
        const f64 osc = obj_scale(lp);
        const f64 obj_sim =
            (abs_sum > 0.0)
                ? obj_dot / ((1.0 + abs_sum) * osc)
                : 0.0;
        // Dual surrogate: normalized row residual (no dual vector required).
        const f64 dual_surr = slack / (1.0 + std::fabs(rhs));
        lift_obj += w * obj_sim;
        lift_dual += w * dual_surr;
        lift_bias += w * bias;
        lift_age += w * (static_cast<f64>(i + 1) /
                         static_cast<f64>(std::max(Index{1}, lp.n_rows())));
        lift_tight +=
            w * (1.0 - std::min(1.0, std::fabs(activity - rhs) /
                                        (1.0 + std::fabs(rhs))));
        lift_nnz += w * (static_cast<f64>(rnnz) /
                         static_cast<f64>(std::max(Index{1}, lp.n_cols())));
        lift_edge += w * (a_ij / (1.0 + std::fabs(a_ij)));
        lift_viol += w * slack;
    }
    if (w_sum > 0.0) {
        const f64 inv = 1.0 / w_sum;
        out[kBranchFeatureDim + 0] = lift_obj * inv;
        out[kBranchFeatureDim + 1] = lift_dual * inv;
        out[kBranchFeatureDim + 2] = lift_bias * inv;
        out[kBranchFeatureDim + 3] = lift_age * inv;
        out[kBranchFeatureDim + 4] = lift_tight * inv;
        out[kBranchFeatureDim + 5] = lift_nnz * inv;
        out[kBranchFeatureDim + 6] = lift_edge * inv;
        out[kBranchFeatureDim + 7] = lift_viol * inv;
    }
    return true;
}

void fill_branch_features_batch(const BranchFeatureContext& ctx,
                                std::vector<Index>& candidates,
                                std::vector<BranchFeatureVec>& feats) {
    candidates.clear();
    feats.clear();
    if (!ctx.lp || !ctx.col_lo || !ctx.col_hi || !ctx.x) return;
    const Index n = ctx.lp->n_cols();
    candidates.reserve(sz(n));
    feats.reserve(sz(n));
    for (Index j = 0; j < n; ++j) {
        BranchFeatureVec v{};
        if (!fill_branch_features(ctx, j, v)) continue;
        candidates.push_back(j);
        feats.push_back(v);
    }
}

bool fill_cut_features(const CutFeatureContext& ctx,
                       const CutRow& cut,
                       CutFeatureVec& out) {
    out.fill(0.0);
    if (!ctx.lp || !ctx.x) return false;
    const auto& lp = *ctx.lp;
    const auto& x = *ctx.x;
    if (cut.cols.size() != cut.vals.size() || cut.cols.empty()) return false;

    f64 activity = 0.0;
    f64 norm2 = 0.0;
    f64 max_abs = 0.0;
    f64 min_abs = std::numeric_limits<f64>::infinity();
    int int_support = 0;
    for (std::size_t k = 0; k < cut.cols.size(); ++k) {
        const Index j = cut.cols[k];
        if (j < 0 || j >= lp.n_cols()) return false;
        const f64 a = cut.vals[k];
        activity += a * x[sz(j)];
        norm2 += a * a;
        const f64 aa = std::fabs(a);
        max_abs = std::max(max_abs, aa);
        if (aa > 1e-15) min_abs = std::min(min_abs, aa);
        if (!lp.is_integer.empty() && lp.is_integer[sz(j)]) ++int_support;
    }
    if (!(norm2 > 0.0)) return false;
    const f64 norm = std::sqrt(norm2);
    f64 viol = 0.0;
    if (std::isfinite(cut.row_hi))
        viol = std::max(viol, activity - cut.row_hi);
    if (std::isfinite(cut.row_lo))
        viol = std::max(viol, cut.row_lo - activity);
    const f64 efficacy = viol / norm;

    f64 obj_dot = 0.0, obj_n2 = 0.0;
    for (std::size_t k = 0; k < cut.cols.size(); ++k) {
        const Index j = cut.cols[k];
        obj_dot += cut.vals[k] * lp.c[sz(j)];
        obj_n2 += lp.c[sz(j)] * lp.c[sz(j)];
    }
    const f64 obj_par =
        (norm > 0.0 && obj_n2 > 0.0) ? obj_dot / (norm * std::sqrt(obj_n2)) : 0.0;
    const f64 density =
        static_cast<f64>(cut.cols.size()) /
        static_cast<f64>(std::max(Index{1}, lp.n_cols()));
    const f64 int_sup =
        static_cast<f64>(int_support) /
        static_cast<f64>(std::max<std::size_t>(1, cut.cols.size()));
    const f64 dynamism =
        (std::isfinite(min_abs) && min_abs > 0.0) ? max_abs / min_abs : 0.0;

    out[0] = efficacy;
    out[1] = obj_par;
    out[2] = density;
    out[3] = int_sup;
    out[4] = viol;
    out[5] = std::log1p(dynamism);
    out[6] = static_cast<f64>(cut.cols.size());
    out[7] = norm;
    return true;
}

BipartiteGraphSnapshot build_bipartite_snapshot(
    const model::LpProblem& lp,
    const std::vector<f64>& col_lo,
    const std::vector<f64>& col_hi,
    const std::vector<f64>* x) {
    BipartiteGraphSnapshot g;
    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    g.vars.resize(sz(n));
    g.cons.resize(sz(m));
    const f64 osc = obj_scale(lp);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();

    for (Index j = 0; j < n; ++j) {
        VarNodeFeatureVec& v = g.vars[sz(j)];
        v.fill(0.0);
        const f64 lo = col_lo[sz(j)];
        const f64 hi = col_hi[sz(j)];
        const f64 xv = (x && sz(j) < x->size()) ? (*x)[sz(j)] : 0.5 * (lo + hi);
        const bool integer = !lp.is_integer.empty() && lp.is_integer[sz(j)];
        const bool fixed = lo >= hi - 1e-12;
        f64 mean_a = 0.0, max_a = 0.0, min_a = 0.0;
        int nnz = 0;
        coef_stats(lp, j, mean_a, max_a, min_a, nnz);
        v[0] = integer ? 1.0 : 0.0;
        v[1] = fixed ? 0.0 : 1.0;
        v[2] = lp.c[sz(j)] / osc;
        v[3] = xv;
        v[4] = safe_div(xv - lo, hi - lo);
        v[5] = integer ? frac_part(xv) : 0.0;
        v[6] = safe_div(static_cast<f64>(nnz),
                        static_cast<f64>(std::max(Index{1}, m)));
        v[7] = mean_a / (1.0 + max_a);
    }

    for (Index i = 0; i < m; ++i) {
        ConNodeFeatureVec& c = g.cons[sz(i)];
        c.fill(0.0);
        f64 sense = 0.0;
        if (std::isfinite(lp.row_lo[sz(i)]) && std::isfinite(lp.row_hi[sz(i)]) &&
            std::fabs(lp.row_lo[sz(i)] - lp.row_hi[sz(i)]) <= 1e-12)
            sense = 0.0;  // equality
        else if (std::isfinite(lp.row_hi[sz(i)]) && !std::isfinite(lp.row_lo[sz(i)]))
            sense = -1.0;  // <=
        else if (std::isfinite(lp.row_lo[sz(i)]) && !std::isfinite(lp.row_hi[sz(i)]))
            sense = 1.0;   // >=
        else
            sense = 0.5;   // ranged
        const int nnz = static_cast<int>(rp[sz(i) + 1] - rp[sz(i)]);
        f64 abs_sum = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            abs_sum += std::fabs(lp.A.vals[sz(k)]);
        f64 activity = 0.0;
        if (x) {
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index j = ci[sz(k)];
                if (j >= 0 && j < n) activity += lp.A.vals[sz(k)] * (*x)[sz(j)];
            }
        }
        f64 slack = 0.0;
        if (std::isfinite(lp.row_hi[sz(i)]))
            slack = std::max(slack, activity - lp.row_hi[sz(i)]);
        if (std::isfinite(lp.row_lo[sz(i)]))
            slack = std::max(slack, lp.row_lo[sz(i)] - activity);
        c[0] = sense;
        c[1] = safe_div(static_cast<f64>(nnz),
                        static_cast<f64>(std::max(Index{1}, n)));
        c[2] = abs_sum / (1.0 + abs_sum);
        c[3] = activity;
        c[4] = slack;
        c[5] = std::isfinite(lp.row_hi[sz(i)])
            ? lp.row_hi[sz(i)] / (1.0 + std::fabs(lp.row_hi[sz(i)]))
            : 0.0;
    }

    g.edges.reserve(static_cast<std::size_t>(lp.nnz()));
    for (Index i = 0; i < m; ++i) {
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            if (j < 0 || j >= n) continue;
            GraphEdge e;
            e.var = j;
            e.con = i;
            const f64 a = lp.A.vals[sz(k)];
            e.feats[0] = a / (1.0 + std::fabs(a));
            e.feats[1] = std::fabs(a) / (1.0 + std::fabs(a));
            g.edges.push_back(e);
        }
    }
    return g;
}

TripartiteGraphSnapshot build_tripartite_snapshot(
    const model::LpProblem& lp,
    const std::vector<f64>& col_lo,
    const std::vector<f64>& col_hi,
    const std::vector<CutRow>& cut_rows,
    const std::vector<f64>* x) {
    TripartiteGraphSnapshot g;
    g.bipartite = build_bipartite_snapshot(lp, col_lo, col_hi, x);
    g.cuts.resize(cut_rows.size());
    CutFeatureContext cctx{&lp, x};
    for (std::size_t t = 0; t < cut_rows.size(); ++t) {
        CutFeatureVec cf{};
        fill_cut_features(cctx, cut_rows[t], cf);
        CutNodeFeatureVec& cn = g.cuts[t];
        cn.fill(0.0);
        cn[0] = cf[0];  // efficacy
        cn[1] = cf[1];  // obj parallelism
        cn[2] = cf[2];  // density
        cn[3] = cf[3];  // integer support
        cn[4] = cf[4];  // violation
        cn[5] = cf[5];  // dynamism
        for (std::size_t k = 0; k < cut_rows[t].cols.size(); ++k) {
            GraphEdge e;
            e.var = cut_rows[t].cols[k];
            e.con = static_cast<Index>(t);  // cut-local index
            const f64 a = cut_rows[t].vals[k];
            e.feats[0] = a / (1.0 + std::fabs(a));
            e.feats[1] = std::fabs(a) / (1.0 + std::fabs(a));
            g.cut_edges.push_back(e);
        }
    }
    // Cons↔Cut edges for non-trivial feature alignment (HGTSM similar_to_*).
    constexpr f64 kSimTol = 0.15;
    for (std::size_t t = 0; t < g.cuts.size(); ++t) {
        for (std::size_t i = 0; i < g.bipartite.cons.size(); ++i) {
            const f64 sim =
                std::fabs(g.cuts[t][1] * g.bipartite.cons[i][0]) +
                0.1 * (1.0 - std::fabs(g.cuts[t][2] - g.bipartite.cons[i][1]));
            if (sim < kSimTol) continue;
            GraphEdge e;
            e.var = static_cast<Index>(i);  // constraint
            e.con = static_cast<Index>(t);  // cut
            e.feats[0] = sim / (1.0 + sim);
            e.feats[1] = g.cuts[t][2];
            g.cut_con_edges.push_back(e);
        }
    }
    return g;
}

}  // namespace sor::search
