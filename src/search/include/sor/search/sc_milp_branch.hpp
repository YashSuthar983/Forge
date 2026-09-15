// SOR — SC-MILP-style stratified contrastive branching scores
// (arXiv:2511.21107 clean-room, CPU linear/MLP-lite).
//
// Paper spirit: (1) stratified node grouping, (2) dynamic stratified contrastive
// loss (positives in-stratum, negatives across with distance weights),
// (3) supervised expert imitation (strong-branch choice). Scores candidate
// VARIABLES only. Never writes dual bounds.
// Missing model → heuristic stratum score; caller may fall back to sparse-SB
// then reliability branching.
#pragma once

#include "sor/search/features.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

inline constexpr int kScMilpEmbedDim = 8;

struct ScMilpOptions {
    bool enabled = true;
    std::string model_path;
    bool collect_labels = false;
    // When no on-disk model is loaded, still score via stratified heuristics.
    bool use_heuristic_without_model = true;
    f64 fit_lr = 0.05;
    int fit_epochs = 40;
    f64 contrastive_weight = 0.5;
    f64 contrastive_tau = 0.08;
    f64 stratum_alpha = 0.75;
    std::size_t collect_max_samples = 50000;
};

struct ScMilpModel {
    int base_dim = kBranchFeatureDim;
    int n_strata = kBranchStratumCount;
    int embed_dim = kScMilpEmbedDim;
    f64 intercept = 0.0;
    // Linear score head over raw features (supervised imitation).
    std::array<f64, kBranchFeatureDim> weights{};
    std::array<f64, kBranchStratumCount> stratum_bias{};
    // Projection for stratified contrastive embeddings: embed = P · x (row-major).
    std::array<f64, kScMilpEmbedDim * kBranchFeatureDim> proj{};
    bool loaded = false;

    void clear();
    // True when every weight / bias / proj entry is finite (safe to score).
    bool finite() const;
    // Contrastive / ranking score (higher = prefer to branch).
    f64 score(const BranchFeatureVec& x) const;
    void embed(const BranchFeatureVec& x, std::array<f64, kScMilpEmbedDim>& z) const;
};

struct ScMilpSample {
    BranchFeatureVec feats{};
    f64 preference = 0.0;  // higher preferred (e.g. SB product score)
    int decision_id = -1;  // same node / probe round for expert CE
};

struct ScMilpCollector {
    std::vector<ScMilpSample> samples;
    std::size_t max_samples = 50000;

    void clear() { samples.clear(); }
    void add(const BranchFeatureVec& feats, f64 preference, int decision_id = -1);
};

struct ScMilpFitOptions {
    f64 lr = 0.02;  // lower default: 0.05 + contrastive proj diverged to NaN
    int epochs = 40;
    f64 contrastive_weight = 0.25;  // λ on L_cons; L = L_sup + λ L_cons
    f64 contrastive_tau = 0.08;
    f64 stratum_alpha = 0.75;  // α in w(g,g') = σ(α |g-g'|)
    int pair_cap = 8000;
    int contrastive_cap = 4000;
};

// Stratified contrastive + expert imitation fit (paper §4.4 spirit).
ScMilpModel fit_sc_milp_contrastive(const std::vector<ScMilpSample>& samples,
                                    const ScMilpFitOptions& opts);

bool save_sc_milp_model(const std::string& path, const ScMilpModel& model);
bool load_sc_milp_model(const std::string& path, ScMilpModel& model);

// Stratified heuristic score when no fitted model is available (Auto path).
f64 heuristic_sc_milp_score(const BranchFeatureVec& x);

// Argmax score among candidates. Uses model if loaded, else heuristic if
// allow_heuristic. Returns -1 if none.
Index pick_sc_milp_branch(const ScMilpModel& model,
                          const std::vector<Index>& candidates,
                          const std::vector<BranchFeatureVec>& feats,
                          bool allow_heuristic = true);

}  // namespace sor::search
