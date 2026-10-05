// Learned-branching features (P2): the cached, column-local extractor must
// equal the direct row-scan definition bit for bit; the cache must notice a
// different model or revision; and the default selector, which reads no
// features, must build none.
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/features.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using sor::core::Index;
using sor::core::f64;
using sor::model::LpProblem;
using sor::search::BranchFeatureCache;
using sor::search::BranchFeatureContext;
using sor::search::BranchFeatureVec;
using sor::search::LiftedFeatureVec;

namespace {

using sor::search::kBranchFeatureDim;
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

// ---- reference: the direct row-scan definitions -------------------------
// Transcribed from the extractor before caching, except that pseudocost
// features are the averages sum / count (the old code read the sums).

f64 ref_obj_scale(const LpProblem& lp) {
    f64 m = 0.0;
    for (f64 c : lp.c) m = std::max(m, std::fabs(c));
    return 1.0 + m;
}

void ref_coef_stats(const LpProblem& lp, Index j, f64& mean_abs, f64& max_abs,
                    f64& min_abs, int& nnz) {
    mean_abs = max_abs = min_abs = 0.0;
    nnz = 0;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    f64 sum = 0.0, mn = std::numeric_limits<f64>::infinity();
    for (Index i = 0; i < lp.n_rows(); ++i)
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (ci[sz(k)] != j) continue;
            const f64 a = std::fabs(lp.A.vals[sz(k)]);
            if (a <= 1e-15) continue;
            ++nnz; sum += a; max_abs = std::max(max_abs, a); mn = std::min(mn, a);
        }
    if (nnz > 0) { mean_abs = sum / static_cast<f64>(nnz); min_abs = mn; }
}

struct RowScan {
    f64 a_ij = 0.0, activity = 0.0, abs_sum = 0.0, obj_dot = 0.0;
    int rnnz = 0;
    bool hit = false;
};

RowScan scan_row(const LpProblem& lp, Index i, Index j, const std::vector<f64>& x) {
    RowScan r;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        if (ci[sz(k)] != j) continue;
        r.a_ij = lp.A.vals[sz(k)];
        r.hit = true;
        break;
    }
    r.rnnz = static_cast<int>(rp[sz(i) + 1] - rp[sz(i)]);
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        const Index jj = ci[sz(k)];
        const f64 a = lp.A.vals[sz(k)];
        r.abs_sum += std::fabs(a);
        if (jj >= 0 && jj < lp.n_cols()) {
            r.activity += a * x[sz(jj)];
            r.obj_dot += a * lp.c[sz(jj)];
        }
    }
    return r;
}

f64 row_rhs(const LpProblem& lp, Index i) {
    return std::isfinite(lp.row_hi[sz(i)]) ? lp.row_hi[sz(i)]
         : (std::isfinite(lp.row_lo[sz(i)]) ? lp.row_lo[sz(i)] : 0.0);
}

f64 row_slack(const LpProblem& lp, Index i, f64 activity) {
    f64 slack = 0.0;
    if (std::isfinite(lp.row_hi[sz(i)])) slack = std::max(slack, activity - lp.row_hi[sz(i)]);
    if (std::isfinite(lp.row_lo[sz(i)])) slack = std::max(slack, lp.row_lo[sz(i)] - activity);
    return slack;
}

f64 safe_div(f64 a, f64 b) { return a / (std::fabs(b) < 1e-12 ? 1.0 : b); }

BranchFeatureVec ref_branch(const BranchFeatureContext& ctx, Index j) {
    BranchFeatureVec out{};
    const auto& lp = *ctx.lp;
    const auto& x = *ctx.x;
    const f64 lo = (*ctx.col_lo)[sz(j)], hi = (*ctx.col_hi)[sz(j)], xv = x[sz(j)];
    const f64 down = xv - std::floor(xv), up = std::ceil(xv) - xv;
    const f64 osc = ref_obj_scale(lp);
    f64 mean_a, max_a, min_a;
    int nnz;
    ref_coef_stats(lp, j, mean_a, max_a, min_a, nnz);
    const f64 degree = static_cast<f64>(nnz);
    const f64 span = hi - lo;
    const bool binary = std::fabs(lo) <= ctx.int_tol && std::fabs(hi - 1.0) <= ctx.int_tol;
    f64 pc_d = 0.0, pc_u = 0.0;
    if ((*ctx.pc_down_count)[sz(j)] > 0)
        pc_d = (*ctx.pc_down)[sz(j)] / static_cast<f64>((*ctx.pc_down_count)[sz(j)]);
    if ((*ctx.pc_up_count)[sz(j)] > 0)
        pc_u = (*ctx.pc_up)[sz(j)] / static_cast<f64>((*ctx.pc_up_count)[sz(j)]);
    out[0] = std::min(down, up);
    out[1] = xv;
    out[2] = lp.c[sz(j)] / osc;
    out[3] = safe_div(xv - lo, span);
    out[4] = span / (1.0 + std::fabs(lo) + std::fabs(hi));
    out[5] = pc_d;
    out[6] = pc_u;
    out[7] = std::min(pc_d * down, pc_u * up) + 0.1 * std::max(pc_d * down, pc_u * up);
    out[8] = safe_div(degree, static_cast<f64>(std::max(Index{1}, lp.n_rows())));
    out[9] = mean_a / (1.0 + max_a);
    out[10] = max_a / (1.0 + max_a);
    out[11] = min_a / (1.0 + max_a);
    out[12] = binary ? 1.0 : 0.0;
    out[13] = std::fabs(lp.c[sz(j)]) / osc;
    out[14] = down;
    out[15] = up;
    f64 ms = 0, md = 0, me = 0, mse = 0, mt = 0, ma = 0, mx = 0;
    int neigh = 0;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const RowScan r = scan_row(lp, i, j, x);
        if (!r.hit || std::fabs(r.a_ij) <= 1e-15) continue;
        ++neigh;
        const f64 slack = row_slack(lp, i, r.activity);
        f64 sense = 0.0;
        if (std::isfinite(lp.row_hi[sz(i)]) && !std::isfinite(lp.row_lo[sz(i)])) sense = -1.0;
        else if (std::isfinite(lp.row_lo[sz(i)]) && !std::isfinite(lp.row_hi[sz(i)])) sense = 1.0;
        else if (std::isfinite(lp.row_lo[sz(i)]) && std::isfinite(lp.row_hi[sz(i)]) &&
                 std::fabs(lp.row_lo[sz(i)] - lp.row_hi[sz(i)]) > 1e-12) sense = 0.5;
        const f64 dens = static_cast<f64>(r.rnnz) / static_cast<f64>(std::max(Index{1}, lp.n_cols()));
        const f64 rhs = row_rhs(lp, i);
        const f64 tight = 1.0 - std::min(1.0, std::fabs(r.activity - rhs) / (1.0 + std::fabs(rhs)));
        ms += slack; mx = std::max(mx, slack); md += dens;
        me += std::fabs(r.a_ij) / (1.0 + std::fabs(r.a_ij));
        mse += sense; mt += tight; ma += r.abs_sum / (1.0 + r.abs_sum);
    }
    if (neigh > 0) {
        const f64 inv = 1.0 / static_cast<f64>(neigh);
        ms *= inv; md *= inv; me *= inv; mse *= inv; mt *= inv; ma *= inv;
    }
    const f64 gap = (ctx.have_incumbent && std::isfinite(ctx.incumbent) &&
                     std::isfinite(ctx.dual_bound))
        ? std::fabs(ctx.incumbent - ctx.dual_bound) / (1.0 + std::fabs(ctx.incumbent)) : 0.0;
    out[16] = ms; out[17] = mx; out[18] = md; out[19] = me;
    out[20] = mse; out[21] = mt; out[22] = ma;
    out[23] = std::min(1.0, static_cast<f64>(ctx.depth) / 64.0) + 0.1 * gap;
    return out;
}

LiftedFeatureVec ref_lifted(const BranchFeatureContext& ctx, Index j) {
    LiftedFeatureVec out{};
    const BranchFeatureVec base = ref_branch(ctx, j);
    for (int i = 0; i < kBranchFeatureDim; ++i) out[sz(i)] = base[sz(i)];
    const auto& lp = *ctx.lp;
    f64 w_sum = 0, lo = 0, ld = 0, lb = 0, la = 0, lt = 0, ln = 0, le = 0, lv = 0;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const RowScan r = scan_row(lp, i, j, *ctx.x);
        if (!r.hit || std::fabs(r.a_ij) <= 1e-15) continue;
        const f64 w = std::fabs(r.a_ij);
        w_sum += w;
        const f64 slack = row_slack(lp, i, r.activity);
        const f64 rhs = row_rhs(lp, i);
        const f64 osc = ref_obj_scale(lp);
        lo += w * (r.abs_sum > 0.0 ? r.obj_dot / ((1.0 + r.abs_sum) * osc) : 0.0);
        ld += w * (slack / (1.0 + std::fabs(rhs)));
        lb += w * (rhs / (1.0 + r.abs_sum));
        la += w * (static_cast<f64>(i + 1) / static_cast<f64>(std::max(Index{1}, lp.n_rows())));
        lt += w * (1.0 - std::min(1.0, std::fabs(r.activity - rhs) / (1.0 + std::fabs(rhs))));
        ln += w * (static_cast<f64>(r.rnnz) / static_cast<f64>(std::max(Index{1}, lp.n_cols())));
        le += w * (r.a_ij / (1.0 + std::fabs(r.a_ij)));
        lv += w * slack;
    }
    if (w_sum > 0.0) {
        const f64 inv = 1.0 / w_sum;
        const f64 v[8] = {lo, ld, lb, la, lt, ln, le, lv};
        for (int k = 0; k < 8; ++k) out[sz(kBranchFeatureDim + k)] = v[k] * inv;
    }
    return out;
}

// ---- tests --------------------------------------------------------------

void test_cached_matches_reference() {
    std::uint64_t compared = 0;
    for (std::uint32_t seed = 1; seed <= 300; ++seed) {
        const LpProblem lp = sor::test::oracle::make_random_milp(seed);
        std::uint32_t st = seed * 2654435761u + 7u;
        std::vector<f64> x(sz(lp.n_cols()));
        std::vector<f64> pcd(x.size()), pcu(x.size());
        std::vector<std::uint32_t> ncd(x.size()), ncu(x.size());
        for (std::size_t j = 0; j < x.size(); ++j) {
            const f64 u = static_cast<f64>(sor::test::oracle::next(st) % 1000) / 1000.0;
            x[j] = lp.col_lo[j] + u * (lp.col_hi[j] - lp.col_lo[j]);
            pcd[j] = static_cast<f64>(sor::test::oracle::next(st) % 50) / 7.0;
            pcu[j] = static_cast<f64>(sor::test::oracle::next(st) % 50) / 3.0;
            ncd[j] = sor::test::oracle::next(st) % 4;
            ncu[j] = sor::test::oracle::next(st) % 4;
        }
        BranchFeatureContext ctx;
        ctx.lp = &lp;
        ctx.col_lo = &lp.col_lo;
        ctx.col_hi = &lp.col_hi;
        ctx.x = &x;
        ctx.pc_down = &pcd;
        ctx.pc_up = &pcu;
        ctx.pc_down_count = &ncd;
        ctx.pc_up_count = &ncu;
        ctx.depth = static_cast<int>(seed % 70);
        ctx.dual_bound = 1.5;
        ctx.incumbent = 4.0;
        ctx.have_incumbent = (seed & 1u) != 0;
        BranchFeatureCache cache;
        cache.build(lp, 0);
        cache.set_point(x);
        for (Index j = 0; j < lp.n_cols(); ++j) {
            BranchFeatureVec direct{}, cached{};
            const bool ok_direct = sor::search::fill_branch_features(ctx, j, direct);
            BranchFeatureContext cctx = ctx;
            cctx.cache = &cache;
            const bool ok_cached = sor::search::fill_branch_features(cctx, j, cached);
            CHECK(ok_direct == ok_cached);
            if (!ok_cached) continue;
            ++compared;
            const BranchFeatureVec ref = ref_branch(ctx, j);
            for (int k = 0; k < kBranchFeatureDim; ++k) {
                CHECK(cached[sz(k)] == ref[sz(k)]);
                CHECK(direct[sz(k)] == ref[sz(k)]);
            }
            LiftedFeatureVec lf{};
            CHECK(sor::search::fill_lifted_features(cctx, j, lf));
            const LiftedFeatureVec lref = ref_lifted(ctx, j);
            for (std::size_t k = 0; k < lf.size(); ++k) CHECK(lf[k] == lref[k]);
        }
    }
    CHECK(compared > 200);
}

void test_cache_identity_and_invalidation() {
    LpProblem a = sor::test::oracle::make_random_milp(11);
    LpProblem b = a;
    BranchFeatureCache cache;
    CHECK(!cache.matches(a, 0));
    cache.build(a, 0);
    CHECK(cache.matches(a, 0));
    CHECK(!cache.matches(a, 1));  // caller advanced the revision
    CHECK(!cache.matches(b, 0));  // equal content, different model object
    // An objective change must reach obj_scale after a rebuild.
    const f64 before = cache.obj_scale;
    a.c[0] = 1000.0;
    cache.build(a, 1);
    CHECK(cache.matches(a, 1));
    CHECK(cache.obj_scale == 1001.0 && cache.obj_scale != before);
    CHECK(cache.builds == 2);
}

void test_default_branching_builds_no_features() {
    const std::string path = std::string(SOR_SOURCE_DIR) +
                             "/benchmarks/miplib-easy/mps/p0201.mps";
    if (!sor::test::data_available(path)) {
        sor::test::skip("test_default_branching_builds_no_features", path);
        return;
    }
    std::ifstream in(path);
    sor::io::MpsReadReport rep;
    const auto lp = sor::io::read_mps(in, rep);
    sor::search::BabOptions o;
    o.para_bab.threads = 1;
    o.max_nodes = 60;
    o.time_limit_s = 1000.0;
    sor::search::BabDiagnostics d;
    sor::search::solve_milp(lp, o, d);
    CHECK(d.rb_nodes > 0);                 // the default selector branched
    CHECK(d.branch_feature_vectors == 0);  // and read no feature vector
    CHECK(d.ms_branch_features == 0.0);

    // A learned selector does read them, so they are still built for it.
    o.paper_reliability = true;
    o.branch_strategy = sor::search::BranchStrategy::ScMilp;
    o.sc_milp.enabled = true;
    o.sc_milp.use_heuristic_without_model = true;
    sor::search::BabDiagnostics l;
    sor::search::solve_milp(lp, o, l);
    CHECK(l.branch_feature_vectors > 0);
}

}  // namespace

int main() {
    test_cached_matches_reference();
    test_cache_identity_and_invalidation();
    test_default_branching_builds_no_features();
    return sor::test::finish("test_branch_features");
}
