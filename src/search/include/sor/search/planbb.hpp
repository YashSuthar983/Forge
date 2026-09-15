// SOR — PlanB&B (AAAI 2026 / arXiv:2511.09219 clean-room).
//
// Paper-complete path (MuZero-style MBRL over B&B):
//   * Representation h — BranchFeatureVec + pooled bipartite state → latent
//   * Dynamics g — latent + action → child dual gains / prune / next latent
//   * Prediction f — policy prior (PUCT), value, branchability
//   * Full MCTS — selection / expansion / simulation / backpropagation
//
// Lite fallback (v1): linear PlanBbPolicy + one-step / shallow UCT over probe
// gains when paper mode is off and no paper model is loaded.
//
// VARIABLE-CHOICE ONLY — never writes dual bounds.
#pragma once

#include "sor/search/features.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

inline constexpr int kPlanBbLatentDim = 32;
inline constexpr int kPlanBbHiddenDim = 64;
inline constexpr int kPlanBbGraphPoolDim = 16;

using PlanBbLatent = std::array<f64, kPlanBbLatentDim>;
using PlanBbGraphPool = std::array<f64, kPlanBbGraphPoolDim>;

struct PlanBbOptions {
    bool enabled = true;
    // Force paper MBRL path (full MCTS + dynamics). Also auto-on when a
    // paper model is loaded from model_path.
    bool paper_mode = false;
    // Full paper model (SOR_PLANBB_PAPER). Prefer over legacy policy_path.
    std::string model_path;
    // Legacy linear stub (SOR_PLANBB 1) for lite path / warm priors.
    std::string policy_path;
    // Shallow one-step lookahead over probe child gains (lite).
    bool shallow_lookahead = true;
    // Bounded MCTS. Lite = probe-gain UCT; paper = model-based rollouts.
    bool use_mcts = true;
    int mcts_sims = 48;
    int mcts_depth = 3;
    f64 mcts_c_puct = 1.4;
    int top_k = 6;
    // Mix for lite path: (1-alpha)*policy + alpha*lookahead_value.
    f64 lookahead_alpha = 0.7;
    f64 eps = 1e-6;
    // Collect SB-expert + dynamics labels during strong-branch probes.
    bool collect_labels = false;
    std::size_t collect_max_samples = 50000;
};

// Mean-pool bipartite snapshot into a fixed state vector (additive helper).
void pool_bipartite_for_planbb(const BipartiteGraphSnapshot& g,
                               PlanBbGraphPool& out);
PlanBbGraphPool zero_graph_pool();

// ---- Legacy linear policy (lite) ------------------------------------------

struct PlanBbPolicy {
    int base_dim = kBranchFeatureDim;
    f64 intercept = 0.0;
    std::array<f64, kBranchFeatureDim> weights{};
    bool loaded = false;

    void clear();
    f64 predict(const BranchFeatureVec& x) const;
};

// Per-candidate strong-branch child estimates (NaN if unavailable).
struct PlanBbChildEstimate {
    f64 down_gain = std::numeric_limits<f64>::quiet_NaN();
    f64 up_gain = std::numeric_limits<f64>::quiet_NaN();
    bool ok = false;
};

struct PlanBbFitOptions {
    f64 lr = 0.05;
    int epochs = 30;
};

struct PlanBbSample {
    BranchFeatureVec feats{};
    f64 target = 0.0;  // e.g. SB product or Achterberg score
};

PlanBbPolicy fit_planbb_policy(const std::vector<PlanBbSample>& samples,
                               const PlanBbFitOptions& opts);

bool save_planbb_policy(const std::string& path, const PlanBbPolicy& policy);
bool load_planbb_policy(const std::string& path, PlanBbPolicy& policy);

// ---- Paper model (MLP over feats + bipartite pool) ------------------------

// Two-layer MLP: y = W2 relu(W1 x + b1) + b2. Flat storage for save/load.
struct PlanBbMlp {
    int in_dim = 0;
    int hidden = 0;
    int out_dim = 0;
    std::vector<f64> w1;  // hidden * in_dim, row-major
    std::vector<f64> b1;
    std::vector<f64> w2;  // out_dim * hidden
    std::vector<f64> b2;

    void clear();
    void init(int in_d, int hid, int out_d, f64 scale = 0.05);
    bool valid() const;
    void forward(const f64* x, f64* y) const;
    // Forward + store hidden activations for a cheap SGD step.
    void forward_train(const f64* x, f64* y, std::vector<f64>& h_out) const;
    void sgd_step(const f64* x, const std::vector<f64>& h,
                  const f64* dy, f64 lr);
};

struct PlanBbTransition {
    f64 down_gain = 0.0;
    f64 up_gain = 0.0;
    f64 prune_prob = 0.0;  // P(child unbranchable / early fathom)
    PlanBbLatent next{};   // imagined child latent
};

struct PlanBbModel {
    int latent_dim = kPlanBbLatentDim;
    int hidden_dim = kPlanBbHiddenDim;
    PlanBbMlp repr;      // [feats|graph] → latent
    PlanBbMlp dynamics;  // [latent|feats] → [log1p(down), log1p(up), prune_logit, next_latent]
    PlanBbMlp policy;    // [feats|graph] → logit
    PlanBbMlp value;     // latent → scalar subtree value (higher = better dual progress)
    bool loaded = false;

    void clear();
    bool ready() const { return loaded && repr.valid() && dynamics.valid() &&
                                policy.valid() && value.valid(); }

    PlanBbLatent encode(const BranchFeatureVec& feats,
                        const PlanBbGraphPool& graph) const;
    f64 policy_logit(const BranchFeatureVec& feats,
                     const PlanBbGraphPool& graph) const;
    f64 value_of(const PlanBbLatent& z) const;
    PlanBbTransition transition(const PlanBbLatent& z,
                                const BranchFeatureVec& action_feats) const;
};

struct PlanBbDynSample {
    BranchFeatureVec feats{};
    PlanBbGraphPool graph{};
    f64 down_gain = 0.0;
    f64 up_gain = 0.0;
    f64 prune = 0.0;  // 1 if either child infinite / infeasible proxy
};

struct PlanBbCollector {
    std::vector<PlanBbDynSample> samples;
    std::size_t max_samples = 50000;
    void clear() { samples.clear(); }
    void add(const BranchFeatureVec& feats, const PlanBbGraphPool& graph,
             f64 down_gain, f64 up_gain, bool pruned);
};

struct PlanBbPaperFitOptions {
    f64 lr = 0.02;
    int epochs = 40;
    f64 dynamics_w = 1.0;
    f64 policy_w = 1.0;
    f64 value_w = 0.5;
};

PlanBbModel fit_planbb_paper(const std::vector<PlanBbDynSample>& samples,
                             const PlanBbPaperFitOptions& opts);

bool save_planbb_model(const std::string& path, const PlanBbModel& model);
bool load_planbb_model(const std::string& path, PlanBbModel& model);

// One-step lookahead value from child gains (product-style, finite only).
f64 planbb_lookahead_value(const PlanBbChildEstimate& est, f64 eps = 1e-6);

// Lite MCTS over probe estimates (fallback).
Index pick_planbb_mcts(const PlanBbPolicy& policy,
                       const PlanBbOptions& opts,
                       const std::vector<Index>& candidates,
                       const std::vector<BranchFeatureVec>& feats,
                       const std::vector<PlanBbChildEstimate>& estimates,
                       std::uint64_t* sims_used = nullptr);

// Paper MCTS: model-based rollouts; probes refine root transitions when present.
Index pick_planbb_mcts_paper(const PlanBbModel& model,
                             const PlanBbPolicy& prior_stub,
                             const PlanBbOptions& opts,
                             const std::vector<Index>& candidates,
                             const std::vector<BranchFeatureVec>& feats,
                             const PlanBbGraphPool& graph,
                             const std::vector<PlanBbChildEstimate>* estimates,
                             std::uint64_t* sims_used = nullptr);

// Unified pick. Paper path when opts.paper_mode || model.ready(); else lite.
Index pick_planbb_branch(const PlanBbPolicy& policy,
                         const PlanBbOptions& opts,
                         const std::vector<Index>& candidates,
                         const std::vector<BranchFeatureVec>& feats,
                         const std::vector<PlanBbChildEstimate>* estimates,
                         std::uint64_t* mcts_sims_used = nullptr);

Index pick_planbb_branch(const PlanBbModel* model,
                         const PlanBbPolicy& policy,
                         const PlanBbOptions& opts,
                         const std::vector<Index>& candidates,
                         const std::vector<BranchFeatureVec>& feats,
                         const PlanBbGraphPool* graph,
                         const std::vector<PlanBbChildEstimate>* estimates,
                         std::uint64_t* mcts_sims_used = nullptr);

inline bool planbb_use_paper_path(const PlanBbOptions& opts,
                                  const PlanBbModel* model) {
    return opts.paper_mode || (model && model->ready());
}

}  // namespace sor::search
