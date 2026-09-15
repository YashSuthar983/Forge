#include "sor/search/l2sep.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline f64 sigmoid(f64 z) {
    if (z > 30.0) return 1.0;
    if (z < -30.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

inline int clamp_budget(f64 v, int lo, int hi) {
    if (!std::isfinite(v)) return lo;
    const int iv = static_cast<int>(std::lround(v));
    return std::max(lo, std::min(hi, iv));
}

}  // namespace

void L2SepModel::clear() {
    feature_dim = kL2SepFeatureDim;
    for (int i = 0; i < kSepFamilyCount; ++i) intercept_allow[i] = 0.0;
    intercept_budget_scale = 0.0;
    terms.clear();
    loaded = false;
}

void L2SepModel::apply(const L2SepFeatureVec& x, DynSepOptions& base,
                       int budget_floor, int budget_cap) const {
    std::array<f64, kSepFamilyCount> logits{};
    for (int f = 0; f < kSepFamilyCount; ++f)
        logits[static_cast<std::size_t>(f)] = intercept_allow[f];
    f64 budget_logit = intercept_budget_scale;
    for (const L2SepTerm& t : terms) {
        if (t.feature < 0 || t.feature >= kL2SepFeatureDim) continue;
        const f64 v = x[static_cast<std::size_t>(t.feature)];
        if (t.family < 0) {
            budget_logit += t.coef * v;
        } else if (t.family < kSepFamilyCount) {
            logits[static_cast<std::size_t>(t.family)] += t.coef * v;
        }
    }
    auto set_allow = [&](SepFamily f, bool& flag) {
        flag = sigmoid(logits[static_cast<std::size_t>(f)]) >= 0.5;
    };
    set_allow(SepFamily::Gmi, base.allow_gmi);
    set_allow(SepFamily::Mir, base.allow_mir);
    set_allow(SepFamily::Cover, base.allow_cover);
    set_allow(SepFamily::Clique, base.allow_clique);
    set_allow(SepFamily::ImpliedBound, base.allow_ib);
    set_allow(SepFamily::ZeroHalf, base.allow_zerohalf);
    set_allow(SepFamily::FlowCover, base.allow_flowcover);
    // Always keep cheap baseline arms available when the model is uncertain.
    base.allow_gmi = base.allow_gmi || base.always_gmi;
    base.allow_ib = base.allow_ib || base.always_ib;

    const f64 scale = std::exp(std::max(-2.0, std::min(2.0, budget_logit)));
    auto scale_b = [&](int& b) {
        b = clamp_budget(static_cast<f64>(b) * scale, budget_floor, budget_cap);
    };
    scale_b(base.budget_gmi);
    scale_b(base.budget_mir);
    scale_b(base.budget_cover);
    scale_b(base.budget_clique);
    scale_b(base.budget_ib);
    scale_b(base.budget_zerohalf);
    scale_b(base.budget_flowcover);
}

void fill_l2sep_features(const model::LpProblem& lp, const SeparatorState& sep,
                         const std::vector<f64>* x, L2SepFeatureVec& out) {
    out.fill(0.0);
    const Index m = lp.n_rows();
    const Index n = lp.n_cols();
    const auto nnz = lp.A.nnz();
    const f64 mf = static_cast<f64>(std::max(Index{1}, m));
    const f64 nf = static_cast<f64>(std::max(Index{1}, n));
    const f64 area = mf * nf;

    out[0] = std::log1p(mf) / 10.0;
    out[1] = std::log1p(nf) / 10.0;
    out[2] = area > 0.0 ? static_cast<f64>(nnz) / area : 0.0;
    out[3] = std::min(10.0, nf / mf);

    Index n_int = 0, n_bin = 0, n_eq = 0, n_boxed = 0, n_obj = 0;
    for (Index j = 0; j < n; ++j) {
        if (!lp.is_integer.empty() && lp.is_integer[sz(j)]) ++n_int;
        const f64 lo = lp.col_lo[sz(j)];
        const f64 hi = lp.col_hi[sz(j)];
        if (std::isfinite(lo) && std::isfinite(hi)) {
            ++n_boxed;
            if (!lp.is_integer.empty() && lp.is_integer[sz(j)] &&
                std::fabs(lo) <= 1e-9 && std::fabs(hi - 1.0) <= 1e-9)
                ++n_bin;
        }
        if (std::fabs(lp.c[sz(j)]) > 1e-15) ++n_obj;
    }
    for (Index i = 0; i < m; ++i) {
        if (std::isfinite(lp.row_lo[sz(i)]) && std::isfinite(lp.row_hi[sz(i)]) &&
            std::fabs(lp.row_lo[sz(i)] - lp.row_hi[sz(i)]) <= 1e-12)
            ++n_eq;
    }
    out[4] = static_cast<f64>(n_int) / nf;
    out[5] = static_cast<f64>(n_bin) / nf;
    out[6] = static_cast<f64>(n_eq) / mf;
    out[7] = static_cast<f64>(nnz) / mf / nf;  // mean row degree / n
    out[13] = std::log1p(static_cast<f64>(nnz)) / 12.0;
    out[14] = static_cast<f64>(n_obj) / nf;
    out[15] = static_cast<f64>(n_boxed) / nf;

    f64 mean_frac = 0.0;
    Index n_frac = 0;
    if (x && x->size() >= sz(n)) {
        for (Index j = 0; j < n; ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            const f64 xv = (*x)[sz(j)];
            const f64 f = xv - std::floor(xv);
            const f64 frac = std::min(f, 1.0 - f);
            if (frac > 1e-6) {
                mean_frac += frac;
                ++n_frac;
            }
        }
        if (n_frac > 0) mean_frac /= static_cast<f64>(n_frac);
    }
    out[8] = static_cast<f64>(n_frac) / std::max(1.0, static_cast<f64>(n_int));
    out[9] = mean_frac;
    out[10] = static_cast<f64>(sep.last_round) / 20.0;
    out[11] = sep.last_candidates > 0
                  ? static_cast<f64>(sep.last_accepted) /
                        static_cast<f64>(sep.last_candidates)
                  : 0.0;
    const f64 scale = 1.0 + std::fabs(sep.last_bound_before);
    out[12] = std::isfinite(sep.last_bound_after) &&
                      std::isfinite(sep.last_bound_before)
                  ? std::fabs(sep.last_bound_after - sep.last_bound_before) / scale
                  : 0.0;
}

void apply_l2sep_builtin_prior(const L2SepFeatureVec& x, DynSepOptions& base,
                               int budget_floor, int budget_cap) {
    const f64 density = x[2];
    const f64 int_frac = x[4];
    const f64 bin_frac = x[5];
    const f64 eq_frac = x[6];
    const f64 frac_share = x[8];
    const f64 mean_frac = x[9];

    // Sparse-ish binary → covers / cliques worth trying. Density of a single
    // knapsack row is 1.0 by construction, so gate on binary share + not an
    // equality-heavy system rather than raw matrix density alone.
    if (bin_frac >= 0.35 && eq_frac < 0.5) {
        base.allow_cover = true;
        base.allow_clique = true;
        base.budget_cover = clamp_budget(60.0 + 80.0 * bin_frac, budget_floor,
                                         budget_cap);
        base.budget_clique = clamp_budget(40.0 + 40.0 * bin_frac, budget_floor,
                                          budget_cap);
    }
    // Fractional general-integer → ZH / flow with modest budgets.
    if (int_frac >= 0.2 && frac_share >= 0.05) {
        base.allow_zerohalf = true;
        base.allow_flowcover = true;
        base.budget_zerohalf =
            clamp_budget(30.0 + 40.0 * mean_frac, budget_floor, budget_cap);
        base.budget_flowcover =
            clamp_budget(30.0 + 30.0 * frac_share, budget_floor, budget_cap);
    }
    // Dense equality systems: shrink expensive arms, keep GMI/IB.
    if (density > 0.08 || eq_frac > 0.6) {
        base.allow_mir = false;
        base.max_optional_arms = std::min(base.max_optional_arms, 1);
        base.budget_gmi =
            clamp_budget(0.7 * static_cast<f64>(base.budget_gmi), budget_floor,
                         budget_cap);
        base.budget_zerohalf =
            clamp_budget(0.6 * static_cast<f64>(base.budget_zerohalf),
                         budget_floor, budget_cap);
    }
    // Very fractional root → slightly larger GMI budget.
    if (frac_share > 0.3)
        base.budget_gmi =
            clamp_budget(1.25 * static_cast<f64>(base.budget_gmi), budget_floor,
                         budget_cap);

    base.allow_gmi = true;
    base.allow_ib = base.allow_ib || base.always_ib;
}

bool save_l2sep_model(const std::string& path, const L2SepModel& model) {
    std::ofstream out(path);
    if (!out) return false;
    out << "SOR_L2SEP 1\n";
    out << "feature_dim " << model.feature_dim << "\n";
    out << "intercept_budget_scale " << model.intercept_budget_scale << "\n";
    out << "intercept_allow";
    for (int i = 0; i < kSepFamilyCount; ++i)
        out << " " << model.intercept_allow[i];
    out << "\n";
    out << "nnz " << model.terms.size() << "\n";
    for (const L2SepTerm& t : model.terms)
        out << t.feature << " " << t.family << " " << t.coef << "\n";
    return static_cast<bool>(out);
}

bool load_l2sep_model(const std::string& path, L2SepModel& model) {
    model.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string tag;
    int ver = 0;
    if (!(in >> tag >> ver) || tag != "SOR_L2SEP" || ver != 1) return false;
    std::string key;
    int nnz = 0;
    while (in >> key) {
        if (key == "feature_dim") {
            in >> model.feature_dim;
        } else if (key == "intercept_budget_scale") {
            in >> model.intercept_budget_scale;
        } else if (key == "intercept_allow") {
            for (int i = 0; i < kSepFamilyCount; ++i)
                in >> model.intercept_allow[i];
        } else if (key == "nnz") {
            in >> nnz;
            break;
        } else {
            return false;
        }
    }
    if (model.feature_dim != kL2SepFeatureDim) return false;
    model.terms.resize(static_cast<std::size_t>(std::max(0, nnz)));
    for (int k = 0; k < nnz; ++k) {
        L2SepTerm t;
        if (!(in >> t.feature >> t.family >> t.coef)) return false;
        model.terms[static_cast<std::size_t>(k)] = t;
    }
    model.loaded = true;
    return true;
}

void configure_dynsep_from_l2sep(const L2SepOptions& opts,
                                 const L2SepModel& model,
                                 const model::LpProblem& lp,
                                 const SeparatorState& sep,
                                 const std::vector<f64>* x,
                                 DynSepOptions& dynsep,
                                 L2SepDiagnostics& diag,
                                 bool mid_tree) {
    if (!opts.enabled) return;
    L2SepFeatureVec feats{};
    fill_l2sep_features(lp, sep, x, feats);
    if (model.loaded) {
        model.apply(feats, dynsep, opts.budget_floor, opts.budget_cap);
        diag.used_builtin_prior = false;
        diag.model_loaded = true;
    } else {
        apply_l2sep_builtin_prior(feats, dynsep, opts.budget_floor,
                                  opts.budget_cap);
        diag.used_builtin_prior = true;
        diag.model_loaded = false;
    }
    ++diag.applies;
    if (mid_tree) ++diag.mid_tree_applies;
}

}  // namespace sor::search
