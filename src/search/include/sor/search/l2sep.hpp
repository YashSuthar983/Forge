// SOR - L2Sep instance-aware separator configuration (WP-E2).
//
// L2Sep (arXiv:2311.05650 spirit): map a compact feature vector from the root
// LP / model stats (+ optional SeparatorState) to DynSepOptions allow_* flags
// and per-family budgets. Clean-room sparse linear / logistic over features -
// not the paper's full learned policy. Applied once at the root under
// milp.policy=latest; optionally re-applied mid-tree. Classical ignores.
#pragma once

#include "sor/model/lp.hpp"
#include "sor/search/dynsep.hpp"
#include "sor/search/features.hpp"
#include "sor/search/milp_policy.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

// Stable dimension for on-disk models (breaking if changed).
inline constexpr int kL2SepFeatureDim = 16;
using L2SepFeatureVec = std::array<f64, kL2SepFeatureDim>;

struct L2SepOptions {
    // Latest default on; Classical forces off via apply_l2sep_policy().
    bool enabled = true;
    // Optional on-disk sparse logistic (SOR_L2SEP). Empty = built-in prior.
    std::string model_path;
    // Re-apply mid-tree every N processed nodes (0 = root only).
    std::uint64_t mid_tree_every_nodes = 500;
    // Soft-cap budgets after the logistic map (safety).
    int budget_cap = 400;
    int budget_floor = 10;
};

struct L2SepTerm {
    int feature = 0;
    int family = 0;  // SepFamily index, or -1 for shared budget scale
    f64 coef = 0.0;
};

struct L2SepModel {
    int feature_dim = kL2SepFeatureDim;
    f64 intercept_allow[kSepFamilyCount]{};
    f64 intercept_budget_scale = 0.0;  // log-scale multiplier on all budgets
    std::vector<L2SepTerm> terms;
    bool loaded = false;

    void clear();
    // Sigmoid logits → allow flags + budget scales applied onto `base`.
    void apply(const L2SepFeatureVec& x, DynSepOptions& base,
               int budget_floor, int budget_cap) const;
};

struct L2SepDiagnostics {
    std::uint64_t applies = 0;
    std::uint64_t mid_tree_applies = 0;
    bool used_builtin_prior = true;
    bool model_loaded = false;
};

inline void apply_l2sep_policy(MilpPolicy policy, L2SepOptions& o) {
    if (policy == MilpPolicy::Classical) o.enabled = false;
}

// RouteFeatures-like + SeparatorState stats. Safe with null x / default sep.
void fill_l2sep_features(const model::LpProblem& lp,
                         const SeparatorState& sep,
                         const std::vector<f64>* x,
                         L2SepFeatureVec& out);

// Built-in sparse prior when no model is loaded (hand-tuned logistic spirit).
void apply_l2sep_builtin_prior(const L2SepFeatureVec& x, DynSepOptions& base,
                               int budget_floor, int budget_cap);

bool save_l2sep_model(const std::string& path, const L2SepModel& model);
bool load_l2sep_model(const std::string& path, L2SepModel& model);

// Convenience: load (optional) + fill features + apply prior or model.
void configure_dynsep_from_l2sep(const L2SepOptions& opts,
                                 const L2SepModel& model,
                                 const model::LpProblem& lp,
                                 const SeparatorState& sep,
                                 const std::vector<f64>* x,
                                 DynSepOptions& dynsep,
                                 L2SepDiagnostics& diag,
                                 bool mid_tree);

}  // namespace sor::search
