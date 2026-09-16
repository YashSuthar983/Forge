// SOR - HGTSM paper-complete cut sequence scoring (arXiv:2410.03112).
//
// Clean-room Heterogeneous-Graph-Transformer Sequence Model:
//   - Encode LP + candidates as a heterogeneous tripartite graph
//     (Vars / Cons / Cuts) via features::build_tripartite_snapshot.
//   - Type-specific message passing (HGT-lite) over six directed edge types.
//   - Permutation-invariant sequence head (Transformer-lite or GRU) over cut
//     embeddings - no positional encodings (paper §3.4).
//   - Higher-level ratio head (mean-pooled cuts → keep fraction).
//
// When a graph model is loaded it is the default scorer for CutPool under
// milp.policy=latest. The linear joint scorer remains the fallback (unloaded
// model, classical policy, or explicit prefer_linear). Fit/save/load cover
// both linear (SOR_HGTSM 1) and full graph (SOR_HGTSM 2) payloads.
#pragma once

#include "sor/search/features.hpp"
#include "sor/search/milp_policy.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

inline constexpr int kHgtsmLpStateDim = 8;
inline constexpr int kHgtsmJointDim = kCutFeatureDim + kHgtsmLpStateDim;  // 16
inline constexpr int kHgtsmEmbDim = 8;
inline constexpr int kHgtsmMsgLayers = 2;

using HgtsmLpStateVec = std::array<f64, kHgtsmLpStateDim>;
using HgtsmJointVec = std::array<f64, kHgtsmJointDim>;

enum class HgtsmSequenceKind : int {
    TransformerLite = 0,
    Gru = 1,
};

struct HgtsmOptions {
    // Latest default on; Classical forces off via apply_hgtsm_policy().
    bool enabled = true;
    // Optional on-disk model (SOR_HGTSM). Empty = built-in linear prior.
    std::string model_path;
    bool collect_labels = false;
    std::size_t collect_max_samples = 50000;
    f64 lasso_lambda = 1e-3;
    int lasso_max_iter = 200;
    int max_nonzero = 200;
    // When a graph payload is loaded, use it unless prefer_linear is set.
    // Default linear: tripartite scoring is costly in the tree; graph only when
    // explicitly wanted (`prefer_linear=false` + loaded graph model).
    bool prefer_linear = true;
    HgtsmSequenceKind sequence = HgtsmSequenceKind::TransformerLite;
    int emb_dim = kHgtsmEmbDim;
    int n_msg_layers = kHgtsmMsgLayers;
    int sgd_epochs = 40;
    f64 sgd_lr = 0.05;
};

struct HgtsmTerm {
    int i = 0;   // joint feature index
    int j = -1;  // -1 = linear; else i<=j product (optional quad)
    f64 coef = 0.0;
};

// Flattened dense weights for the graph path (clean-room HGT + sequence).
struct HgtsmGraphWeights {
    int emb_dim = kHgtsmEmbDim;
    int n_msg_layers = kHgtsmMsgLayers;
    int seq_kind = 0;  // 0 = Transformer-lite, 1 = GRU

    // Type-specific input projections: out = W * feat (+ bias).
    std::vector<f64> W_var;   // emb * kVarNodeFeatureDim
    std::vector<f64> b_var;
    std::vector<f64> W_con;
    std::vector<f64> b_con;
    std::vector<f64> W_cut;
    std::vector<f64> b_cut;

    // Per edge-type message: msg = W_msg[e] * h_src  (emb*emb each).
    // Order: VarCon, ConVar, VarCut, CutVar, ConCut, CutCon.
    std::vector<f64> W_msg;   // 6 * n_msg_layers * emb * emb
    std::vector<f64> W_self;  // n_msg_layers * 3 * emb * emb (var/con/cut)
    std::vector<f64> b_upd;   // n_msg_layers * 3 * emb

    // Sequence: Transformer-lite (Q,K,V,O) or GRU (Wz,Wr,Wh + biases).
    std::vector<f64> W_qkv;   // 3 * emb * emb  (transformer)
    std::vector<f64> W_o;     // emb * emb
    std::vector<f64> W_ff1;   // emb * emb
    std::vector<f64> W_ff2;   // emb * emb
    std::vector<f64> b_ff1;
    std::vector<f64> b_ff2;
    std::vector<f64> W_gru;   // 3 * emb * (emb+emb)  [z,r,h] on [x|h]
    std::vector<f64> b_gru;   // 3 * emb

    // Heads.
    std::vector<f64> W_score;  // emb
    f64 b_score = 0.0;
    std::vector<f64> W_ratio;  // emb → 2 (mu, log_sigma proxy); use [0]
    f64 b_ratio = 0.0;

    void clear();
    void init_xavier(int emb, int n_layers, int seq_kind, std::uint64_t seed);
    bool valid() const;
};

struct HgtsmModel {
    int joint_dim = kHgtsmJointDim;
    bool quadratic = false;
    f64 intercept = 0.0;
    std::vector<HgtsmTerm> terms;
    bool loaded = false;

    // Graph path (paper-complete). When has_graph && loaded, prefer this.
    bool has_graph = false;
    HgtsmGraphWeights graph;

    void clear();
    static void pack_joint(const CutFeatureVec& cut, const HgtsmLpStateVec& lp,
                           HgtsmJointVec& out);
    f64 predict_linear(const CutFeatureVec& cut,
                       const HgtsmLpStateVec& lp) const;
    // Built-in prior when unloaded: efficacy + light obj/sparsity terms.
    static f64 builtin_score(const CutFeatureVec& cut,
                             const HgtsmLpStateVec& lp);
    // Backward-compatible alias.
    f64 predict(const CutFeatureVec& cut, const HgtsmLpStateVec& lp) const {
        return predict_linear(cut, lp);
    }
};

struct HgtsmSample {
    HgtsmJointVec feats{};
    f64 label = 0.0;  // efficacy / bound-delta pseudo-label
};

// One separation round for sequence / graph fitting.
struct HgtsmRoundSample {
    TripartiteGraphSnapshot graph;
    std::vector<f64> labels;  // parallel to graph.cuts
};

struct HgtsmCollector {
    std::vector<HgtsmSample> samples;
    std::vector<HgtsmRoundSample> rounds;
    std::size_t max_samples = 50000;
    std::size_t max_rounds = 2000;
    void clear() {
        samples.clear();
        rounds.clear();
    }
    void add(const CutFeatureVec& cut, const HgtsmLpStateVec& lp, f64 label);
    void add_round(TripartiteGraphSnapshot graph, std::vector<f64> labels);
};

struct HgtsmFitOptions {
    bool use_quadratic = false;
    bool fit_graph = true;
    HgtsmSequenceKind sequence = HgtsmSequenceKind::TransformerLite;
    f64 lasso_lambda = 1e-3;
    int max_iter = 200;
    int max_nonzero = 200;
    int emb_dim = kHgtsmEmbDim;
    int n_msg_layers = kHgtsmMsgLayers;
    int sgd_epochs = 40;
    f64 sgd_lr = 0.05;
    std::uint64_t seed = 0x484754534dull;  // "HGTSM"
};

struct HgtsmDiagnostics {
    std::uint64_t selects = 0;
    std::uint64_t scored_cuts = 0;
    std::uint64_t graph_selects = 0;
    bool used_builtin = true;
    bool model_loaded = false;
    bool used_graph = false;
};

inline void apply_hgtsm_policy(MilpPolicy policy, HgtsmOptions& o) {
    if (policy == MilpPolicy::Classical) o.enabled = false;
}

// LP-state side of the heterogeneous pair (depth, frac, gap proxy, ...).
void fill_hgtsm_lp_state(Index n_cols, Index n_int, Index n_frac,
                         f64 mean_frac, f64 gap_rel, int depth,
                         std::size_t pool_size, f64 last_bound_gain,
                         HgtsmLpStateVec& out);

// Score used by CutPool when HGTSM is armed (linear / builtin path).
f64 hgtsm_score(const HgtsmModel& model, const CutFeatureVec& cut,
                const HgtsmLpStateVec& lp);

// Paper path: score all cuts jointly via HGT + sequence. Writes one score per
// cut node. Optional ratio_out in [0,1] from the higher-level head.
// Falls back to per-cut linear/builtin if !model.has_graph.
void hgtsm_score_sequence(const HgtsmModel& model,
                          const TripartiteGraphSnapshot& graph,
                          std::vector<f64>& scores_out,
                          f64* ratio_out = nullptr);

bool hgtsm_prefer_graph(const HgtsmModel& model, const HgtsmOptions& opts);

HgtsmModel fit_hgtsm_lasso(const std::vector<HgtsmSample>& samples,
                           const HgtsmFitOptions& opts);

// Fit linear + (optional) full graph/sequence from round samples.
HgtsmModel fit_hgtsm(const HgtsmCollector& collector,
                     const HgtsmFitOptions& opts);

bool save_hgtsm_model(const std::string& path, const HgtsmModel& model);
bool load_hgtsm_model(const std::string& path, HgtsmModel& model);

// Greedy sequence order over already-scored candidates (indices into a score
// vector). Ties broken by index. Does not mutate cuts - CutPool owns filters.
std::vector<std::size_t> hgtsm_sequence_order(const std::vector<f64>& scores,
                                              int max_keep);

}  // namespace sor::search
