// SOR — sparse linear / quadratic strong-branching score predictor
// (Bayramoğlu, Nemhauser & Sahinidis, arXiv:2604.00094) + ranking spirit
// from Khalil et al. 2016 (ranking SVM over branching features).
//
// Clean-room: lasso OR pairwise ranking loss over Khalil/Gasse-style variable
// features with optional quadratic expansions and bipartite neighbor aggregates.
// Predicts SB product scores; never writes dual bounds.
// Missing / unloaded model → caller falls back to reliability branching.
#pragma once

#include "sor/search/features.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

enum class SparseSbLoss : std::uint8_t {
    Lasso = 0,
    Ranking = 1,  // pairwise logistic ranking (default train path)
};

struct SparseSbOptions {
    // When true under milp.policy=latest, prefer model scores when loaded.
    bool enabled = true;
    // Optional on-disk model (text format SOR_SPARSE_SB). Empty = none.
    std::string model_path;
    // Collect (features, normalized SB score) during strong-branch probes.
    bool collect_labels = false;
    bool use_quadratic = true;
    int max_nonzero = 500;
    f64 lasso_lambda = 1e-3;
    int lasso_max_iter = 200;
    SparseSbLoss prefer_loss = SparseSbLoss::Ranking;
    // Cap samples kept in the in-solve collector.
    std::size_t collect_max_samples = 50000;
};

struct SparseSbTerm {
    int i = 0;     // base feature index
    int j = -1;    // -1 = linear; j==i = square; else i<j product
    f64 coef = 0.0;
};

struct SparseSbModel {
    int base_dim = kBranchFeatureDim;
    bool quadratic = true;
    SparseSbLoss loss = SparseSbLoss::Lasso;
    f64 intercept = 0.0;
    std::vector<SparseSbTerm> terms;
    bool loaded = false;

    void clear();
    f64 predict(const BranchFeatureVec& x) const;
    // Expand base features into the design vector used by fit/predict internals.
    static void expand_features(const BranchFeatureVec& x,
                                bool quadratic,
                                std::vector<f64>& out,
                                int base_dim = kBranchFeatureDim);
};

struct SparseSbSample {
    BranchFeatureVec feats{};
    f64 sb_score = 0.0;  // raw product score (pre-normalization per batch OK)
};

struct SparseSbCollector {
    std::vector<SparseSbSample> samples;
    std::size_t max_samples = 50000;

    void clear() { samples.clear(); }
    void add(const BranchFeatureVec& feats, f64 sb_score);
};

struct SparseSbFitOptions {
    bool use_quadratic = true;
    f64 lasso_lambda = 1e-3;
    int max_iter = 200;
    int max_nonzero = 500;
    f64 intercept = 0.0;  // if NaN, estimate mean(y)
    SparseSbLoss loss = SparseSbLoss::Ranking;
    f64 ranking_lr = 0.05;
    int ranking_epochs = 40;
    f64 ranking_l1 = 1e-3;
    // Cap pairwise comparisons per epoch (0 = all pairs, capped internally).
    int ranking_pair_cap = 20000;
};

// Coordinate-descent lasso on expanded features. Labels are L2-normalized
// across the sample set (paper §4.4). Returns empty/unloaded on failure.
SparseSbModel fit_sparse_sb_lasso(const std::vector<SparseSbSample>& samples,
                                  const SparseSbFitOptions& opts);

// Pairwise logistic ranking over expanded features with proximal L1
// (Khalil ranking spirit + Sparse-SB quadratic expansions).
SparseSbModel fit_sparse_sb_ranking(const std::vector<SparseSbSample>& samples,
                                    const SparseSbFitOptions& opts);

// Dispatch on opts.loss (default Ranking).
SparseSbModel fit_sparse_sb(const std::vector<SparseSbSample>& samples,
                            const SparseSbFitOptions& opts);

bool save_sparse_sb_model(const std::string& path, const SparseSbModel& model);
bool load_sparse_sb_model(const std::string& path, SparseSbModel& model);

// Pick argmax predicted SB score among candidates. Returns -1 if none.
Index pick_sparse_sb_branch(const SparseSbModel& model,
                            const std::vector<Index>& candidates,
                            const std::vector<BranchFeatureVec>& feats);

// Product SB score used as the imitation label (Achterberg product with eps).
inline f64 sb_product_score(f64 down_gain, f64 up_gain, f64 eps = 1e-6) {
    const f64 d = std::max(down_gain, eps);
    const f64 u = std::max(up_gain, eps);
    return d * u;
}

}  // namespace sor::search
