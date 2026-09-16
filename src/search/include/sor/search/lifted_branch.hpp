// SOR - Lifted Branching (Renard-Louveaux-Fortz EJOR 2026 clean-room).
//
// Iteratively improves an imitation expert from strong-branch / lifted scores.
// Features: Gasse-style variable + aggregated constraint lifts
// (LiftedFeatureVec). Expert is a sparse/ ranking linear model over lifted
// features. Cold start falls back to RB / sparse-SB. Variable choice only -
// never writes dual bounds.
#pragma once

#include "sor/search/features.hpp"
#include "sor/search/sparse_sb.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct LiftedBranchOptions {
    bool enabled = true;
    // Refit expert after this many new labeled probes (since last fit).
    int refit_every = 32;
    int min_samples = 8;
    bool use_quadratic = false;
    f64 lasso_lambda = 1e-3;
    int lasso_max_iter = 80;
    int max_nonzero = 200;
    // Ranking loss preferred (imitation of lifted/SB scores).
    SparseSbLoss loss = SparseSbLoss::Ranking;
    f64 ranking_lr = 0.05;
    int ranking_epochs = 30;
    // Optional warm-start expert path (SOR_LIFTED_SB format; Sparse-SB also OK
    // if dims match via branch-prefix projection).
    std::string expert_path;
};

struct LiftedSbTerm {
    int i = 0;
    int j = -1;
    f64 coef = 0.0;
};

struct LiftedSbModel {
    int dim = kLiftedFeatureDim;
    bool quadratic = false;
    SparseSbLoss loss = SparseSbLoss::Ranking;
    f64 intercept = 0.0;
    std::vector<LiftedSbTerm> terms;
    bool loaded = false;

    void clear();
    f64 predict(const LiftedFeatureVec& x) const;
};

struct LiftedSbSample {
    LiftedFeatureVec feats{};
    f64 score = 0.0;  // SB product or lifted approximation label
};

struct LiftedSbCollector {
    std::vector<LiftedSbSample> samples;
    std::size_t max_samples = 50000;
    void clear() { samples.clear(); }
    void add(const LiftedFeatureVec& feats, f64 score);
};

struct LiftedBranchState {
    LiftedSbModel expert;
    LiftedSbCollector buffer;
    int iterations = 0;  // successful refits (δ in paper LLB_δ)
    int samples_since_fit = 0;
    bool seeded_from_file = false;

    void clear();
    void observe(const LiftedFeatureVec& feats, f64 sb_score);
    // Refit when enough new labels accumulated. Returns true on a new fit.
    bool maybe_refit(const LiftedBranchOptions& opts);
    bool ready() const { return expert.loaded; }
};

LiftedSbModel fit_lifted_sb(const std::vector<LiftedSbSample>& samples,
                            const LiftedBranchOptions& opts);

bool save_lifted_sb_model(const std::string& path, const LiftedSbModel& model);
bool load_lifted_sb_model(const std::string& path, LiftedSbModel& model);

// Pick via current expert; -1 if expert not ready.
Index pick_lifted_branch(const LiftedBranchState& state,
                         const std::vector<Index>& candidates,
                         const std::vector<LiftedFeatureVec>& feats);

}  // namespace sor::search
