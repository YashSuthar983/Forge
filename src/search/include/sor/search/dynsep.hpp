// SOR — DynSep paper-complete separator configuration (WP-E1 / NeurIPS 2025).
//
// DynSep (Ye et al., NeurIPS 2025): dynamic RL separator configuration on an
// *incremental triplet graph* (vars / cons / separator nodes). The paper encodes
// each round's Δ-subgraph with a GCN, tokenizes graph + separator embeddings,
// and uses a decoder-only Transformer + PPO to predict (i) max separation
// rounds and (ii) per-separator activation ∈ {-1,0,+1}.
//
// This file is a clean-room, CPU-only deployment of that control surface:
//   • Heterogeneous MPNN (2–3 layers, mean agg, ReLU) over a compact
//     bipartite/tripartite round graph: problem(+delta) nodes ↔ separator nodes.
//   • Policy heads: multi-label separator on/off + per-family budget scales.
//   • Online/offline imitation (logistic / CE) from efficacy/bound labels;
//     weights in dense float vectors, SOR_DYNSEP save/load.
//   • UCB1 bandit remains Classical / missing-model fallback
//     (`dynsep.backend = Gnn | Ucb | Auto`).
//
// Paper deltas still open (documented, not blockers for product path):
//   full V/C node graphs (we pool LP/round features), decoder-only Transformer
//   over tokenized history, PPO/online RL, SCIP-side sep frequencies {-1,0,+1}.
// L2Sep (arXiv:2311.05650) still supplies instance-aware allow_*/budget priors
// into DynSepOptions before the controller runs.
//
// Measurement-gated under milp.policy=latest: DynSep is ON; MIR/cover/clique
// stay off unless opted in or rewarded. Classical disables DynSep.
#pragma once

#include "sor/search/flowcover.hpp"
#include "sor/search/milp_policy.hpp"
#include "sor/search/zerohalf.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;

enum class SepFamily : std::uint8_t {
    Gmi = 0,
    Mir = 1,
    Cover = 2,
    Clique = 3,
    ImpliedBound = 4,
    ZeroHalf = 5,
    FlowCover = 6,
    Count = 7,
};

inline constexpr int kSepFamilyCount = static_cast<int>(SepFamily::Count);

inline const char* sep_family_name(SepFamily f) noexcept {
    switch (f) {
    case SepFamily::Gmi: return "gmi";
    case SepFamily::Mir: return "mir";
    case SepFamily::Cover: return "cover";
    case SepFamily::Clique: return "clique";
    case SepFamily::ImpliedBound: return "ib";
    case SepFamily::ZeroHalf: return "zerohalf";
    case SepFamily::FlowCover: return "flowcover";
    case SepFamily::Count: break;
    }
    return "unknown";
}

// Auto: prefer GNN when a valid model is loaded, else UCB.
enum class DynSepBackend : std::uint8_t { Auto = 0, Gnn = 1, Ucb = 2 };

inline constexpr int kDynSepStateFeatDim = 12;
inline constexpr int kDynSepSepFeatDim = 12;
inline constexpr int kDynSepEmbDim = 8;
inline constexpr int kDynSepMsgLayers = 3;

using DynSepStateFeat = std::array<f64, kDynSepStateFeatDim>;
using DynSepSepFeat = std::array<f64, kDynSepSepFeatDim>;

struct DynSepOptions {
    // Latest default on; Classical forces off via apply_dynsep_policy().
    bool enabled = true;
    DynSepBackend backend = DynSepBackend::Auto;
    // Optional on-disk GNN (SOR_DYNSEP). Empty = UCB / Auto→UCB.
    std::string model_path;
    bool collect_labels = false;
    std::size_t collect_max_samples = 50000;
    // UCB1 exploration constant (fallback / Classical path).
    f64 ucb_c = 1.25;
    // GNN decision thresholds / training.
    f64 gnn_on_threshold = 0.45;
    int emb_dim = kDynSepEmbDim;
    int n_msg_layers = kDynSepMsgLayers;
    int sgd_epochs = 40;
    f64 sgd_lr = 0.05;
    // Always schedule these when allowed (cheap / historically useful baseline).
    bool always_gmi = true;
    bool always_ib = true;
    // Arms the bandit / GNN may enable. MIR/cover/clique stay opt-in:
    // enabling MIR+cover by default stalled dual proofs on gt2 (Latest
    // Feasible @1.4% gap; --no-dynsep Optimal in 15s). ZH/flow remain on.
    bool allow_gmi = true;
    bool allow_mir = false;
    bool allow_cover = false;
    bool allow_clique = false;
    bool allow_ib = true;
    bool allow_zerohalf = true;
    bool allow_flowcover = true;
    // Soft cap on how many optional arms run in one round (excl. always_*).
    int max_optional_arms = 2;
    // Per-family cut budgets when selected.
    int budget_gmi = 200;
    int budget_mir = 80;
    int budget_cover = 80;
    int budget_clique = 50;
    int budget_ib = 100;
    // Paper-complete ZH / flow engines: slightly larger default budgets so
    // DynSep can surface GF(2) aggregates and SI-lifted flow covers.
    int budget_zerohalf = 60;
    int budget_flowcover = 60;
    // Optional arms (MIR/cover/…) only near the root — tree DynSep stays GMI/IB.
    int max_depth_optional = 1;
    ZeroHalfOptions zerohalf;
    FlowCoverOptions flowcover;
};

struct DynSepDecision {
    bool run_gmi = false;
    bool run_mir = false;
    bool run_cover = false;
    bool run_clique = false;
    bool run_ib = false;
    bool run_zerohalf = false;
    bool run_flowcover = false;
    int budget_gmi = 0;
    int budget_mir = 0;
    int budget_cover = 0;
    int budget_clique = 0;
    int budget_ib = 0;
    int budget_zerohalf = 0;
    int budget_flowcover = 0;
    // GNN may propose a soft max-round hint (paper 𝑎ₜ⁽¹⁾); 0 = unused.
    int max_rounds_hint = 0;
};

struct DynSepRoundInput {
    int round = 0;
    int depth = 0;
    bool at_root = true;
    // Recent bound improvement from the previous separation round (relative).
    f64 last_bound_gain_rel = 0.0;
    // User / BabOptions force-on flags (opt-in families).
    bool force_mir = false;
    bool force_cover = false;
    bool force_clique = false;
    bool force_ib = true;
    bool force_zerohalf = false;
    bool force_flowcover = false;
    bool have_conflict_graph = false;
    bool have_basis = true;
    // Optional LP / frac context (filled by BaB when available).
    f64 frac_share = 0.0;
    f64 mean_frac = 0.0;
    f64 gap_rel = 1.0;
    int n_cuts_last = 0;
};

struct DynSepObserve {
    SepFamily family = SepFamily::Gmi;
    int cuts_generated = 0;
    int cuts_selected = 0;
    f64 total_efficacy = 0.0;
};

struct DynSepDiagnostics {
    std::uint64_t decisions = 0;
    std::uint64_t arms_scheduled = 0;
    std::uint64_t gnn_decisions = 0;
    std::uint64_t ucb_decisions = 0;
    std::array<std::uint64_t, kSepFamilyCount> plays{};
    std::array<std::uint64_t, kSepFamilyCount> cuts_from{};
    std::array<f64, kSepFamilyCount> reward_sum{};
    bool model_loaded = false;
    bool used_gnn = false;
};

// Dense GNN weights (float vectors; no torch / Eigen).
struct DynSepGnnWeights {
    int emb_dim = kDynSepEmbDim;
    int n_msg_layers = kDynSepMsgLayers;
    int state_dim = kDynSepStateFeatDim;
    int sep_dim = kDynSepSepFeatDim;
    int n_sep = kSepFamilyCount;

    // Input projections.
    std::vector<f64> W_state;   // emb * state_dim
    std::vector<f64> b_state;
    std::vector<f64> W_sep;     // emb * sep_dim
    std::vector<f64> b_sep;
    std::vector<f64> W_delta;   // emb * state_dim  (incremental node)
    std::vector<f64> b_delta;

    // Edge-type messages (3 types × layers × emb×emb):
    //   0: state→sep, 1: sep→state, 2: sep→sep (mean), 3: delta→sep, 4: sep→delta
    std::vector<f64> W_msg;     // 5 * n_msg_layers * emb * emb
    std::vector<f64> W_self;    // n_msg_layers * 3 * emb * emb  (state/sep/delta)
    std::vector<f64> b_upd;     // n_msg_layers * 3 * emb

    // Heads: on/off logit and log-budget-scale per separator embedding.
    std::vector<f64> W_on;      // emb
    f64 b_on = 0.0;
    std::vector<f64> W_bud;     // emb
    f64 b_bud = 0.0;
    // Max-rounds categorical proxy: pooled graph → scalar in (0,1) * T_max.
    std::vector<f64> W_rounds;  // emb
    f64 b_rounds = 0.0;
    int t_max = 5;

    void clear();
    void init_xavier(int emb, int n_layers, std::uint64_t seed);
    bool valid() const;
};

struct DynSepModel {
    DynSepGnnWeights gnn;
    bool loaded = false;

    void clear();
};

// Compact incremental round graph (problem + delta + separators).
struct DynSepRoundGraph {
    DynSepStateFeat state{};
    DynSepStateFeat delta{};  // incremental features vs previous round
    std::array<DynSepSepFeat, kSepFamilyCount> seps{};
};

struct DynSepSample {
    DynSepRoundGraph graph;
    // Multi-label: 1 if family helped (selected cuts / positive efficacy).
    std::array<f64, kSepFamilyCount> label_on{};
    // Soft budget targets in [0,1] (relative to option caps); unused → 0.
    std::array<f64, kSepFamilyCount> label_budget{};
};

struct DynSepCollector {
    std::vector<DynSepSample> samples;
    std::size_t max_samples = 50000;
    void clear() { samples.clear(); }
    void add(DynSepSample s);
};

struct DynSepFitOptions {
    int emb_dim = kDynSepEmbDim;
    int n_msg_layers = kDynSepMsgLayers;
    int sgd_epochs = 40;
    f64 sgd_lr = 0.05;
    std::uint64_t seed = 0x44594e534550ull;  // "DYNSEP"
};

inline void apply_dynsep_policy(MilpPolicy policy, DynSepOptions& o) {
    if (policy == MilpPolicy::Classical) {
        o.enabled = false;
        o.allow_zerohalf = false;
        o.allow_flowcover = false;
        o.backend = DynSepBackend::Ucb;
        o.collect_labels = false;
    }
}

void fill_dynsep_state_feats(const DynSepRoundInput& in,
                             const DynSepOptions& opts,
                             DynSepStateFeat& out);

void fill_dynsep_sep_feats(SepFamily f, const DynSepRoundInput& in,
                           const DynSepOptions& opts, f64 mean_reward,
                           f64 plays_norm, f64 cuts_norm, bool allowed,
                           DynSepSepFeat& out);

// Forward: logits_on[k], budget_scale[k] in ~[0.25, 2], max_rounds_hint.
void dynsep_forward(const DynSepGnnWeights& g, const DynSepRoundGraph& graph,
                    std::array<f64, kSepFamilyCount>& logits_on,
                    std::array<f64, kSepFamilyCount>& budget_scale,
                    int& max_rounds_hint);

DynSepModel fit_dynsep_imitation(const DynSepCollector& collector,
                                 const DynSepFitOptions& opts);

bool save_dynsep_model(const std::string& path, const DynSepModel& model);
bool load_dynsep_model(const std::string& path, DynSepModel& model);

bool dynsep_prefer_gnn(const DynSepModel& model, const DynSepOptions& opts);

// Online adapter: GNN when model present (Latest/Auto), else UCB1.
class DynSepController {
public:
    explicit DynSepController(const DynSepOptions& opts);

    const DynSepOptions& options() const { return opts_; }
    DynSepDiagnostics& diagnostics() { return diag_; }
    const DynSepDiagnostics& diagnostics() const { return diag_; }
    DynSepCollector& collector() { return collector_; }
    const DynSepCollector& collector() const { return collector_; }
    const DynSepModel& model() const { return model_; }

    void set_model(DynSepModel model);
    // L2Sep may rewrite allow_*/budgets mid-solve; bandit counts are kept.
    void update_options(const DynSepOptions& opts);

    DynSepDecision decide(const DynSepRoundInput& in);
    void observe(const DynSepObserve& obs);
    // Convenience: attribute selected cuts by name prefix after pool select.
    void observe_selected_names(const std::vector<std::string>& names,
                                f64 mean_efficacy);
    // Call after a round's decisions+observations to record an imitation sample.
    void collect_round_labels(const DynSepDecision& decided);

private:
    DynSepOptions opts_;
    DynSepDiagnostics diag_;
    DynSepModel model_;
    DynSepCollector collector_;
    std::array<std::uint64_t, kSepFamilyCount> plays_{};
    std::array<f64, kSepFamilyCount> reward_sum_{};
    std::uint64_t total_plays_ = 0;

    DynSepRoundGraph last_graph_{};
    DynSepDecision last_decision_{};
    bool have_last_graph_ = false;
    DynSepStateFeat prev_state_{};

    f64 ucb(SepFamily f) const;
    bool allowed(SepFamily f, const DynSepRoundInput& in) const;
    DynSepDecision decide_ucb(const DynSepRoundInput& in);
    DynSepDecision decide_gnn(const DynSepRoundInput& in);
    void build_graph(const DynSepRoundInput& in, DynSepRoundGraph& g) const;
};

}  // namespace sor::search
