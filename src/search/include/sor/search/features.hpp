// SOR — feature export APIs for latest MILP policies (WP-L0).
//
// Clean-room feature set inspired by Khalil et al. 2016 / Gasse et al. 2019 /
// Bayramoğlu–Nemhauser–Sahinidis (arXiv:2604.00094): variable, constraint, and
// cut descriptors with stable dimensions so learners can load without scraping
// SCIP-style internals. Features never tighten dual bounds.
#pragma once

#include "sor/model/lp.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

// Forward decl — avoid a cuts.hpp ↔ features.hpp ↔ hgtsm.hpp include cycle.
struct CutRow;

// Stable dimensions — bumping is a soft ABI break; loaders accept legacy dims
// by zero-padding missing coordinates (see sparse_sb / sc_milp / lifted load).
//
// Branch layout (Khalil/Gasse/Sparse-SB spirit):
//   [0..15] core variable / pseudocost / column stats (legacy v1)
//   [16..23] bipartite neighbor aggregates (constraint + edge summaries)
inline constexpr int kBranchFeatureDimLegacy = 16;
inline constexpr int kBranchFeatureDim = 24;
inline constexpr int kCutFeatureDim = 8;
inline constexpr int kVarNodeFeatureDim = 8;
inline constexpr int kConNodeFeatureDim = 6;
inline constexpr int kCutNodeFeatureDim = 6;
inline constexpr int kEdgeFeatureDim = 2;
// Lifted Branching (Renard–Louveaux–Fortz): branch vec + extra var/con lifts.
inline constexpr int kLiftedExtraDim = 8;
inline constexpr int kLiftedFeatureDim = kBranchFeatureDim + kLiftedExtraDim;

using BranchFeatureVec = std::array<f64, kBranchFeatureDim>;
using LiftedFeatureVec = std::array<f64, kLiftedFeatureDim>;
using CutFeatureVec = std::array<f64, kCutFeatureDim>;
using VarNodeFeatureVec = std::array<f64, kVarNodeFeatureDim>;
using ConNodeFeatureVec = std::array<f64, kConNodeFeatureDim>;
using CutNodeFeatureVec = std::array<f64, kCutNodeFeatureDim>;
using EdgeFeatureVec = std::array<f64, kEdgeFeatureDim>;

struct BranchFeatureContext {
    const model::LpProblem* lp = nullptr;
    const std::vector<f64>* col_lo = nullptr;
    const std::vector<f64>* col_hi = nullptr;
    const std::vector<f64>* x = nullptr;
    const std::vector<f64>* pc_down = nullptr;   // unit gains; may be null
    const std::vector<f64>* pc_up = nullptr;
    const std::vector<std::uint32_t>* pc_down_count = nullptr;
    const std::vector<std::uint32_t>* pc_up_count = nullptr;
    const std::vector<Index>* col_degree = nullptr;
    f64 int_tol = 1e-6;
    int depth = 0;
    f64 dual_bound = 0.0;
    f64 incumbent = 0.0;
    bool have_incumbent = false;
};

// SC-MILP-style discrete stratum over branch features (stable bucket count).
// Layout: 4 fractionality bins × binary/general → 8 strata.
inline constexpr int kBranchStratumCount = 8;
int branch_feature_stratum(const BranchFeatureVec& f) noexcept;

// Fill one candidate variable's branching feature vector. Returns false if j
// is not a free fractional integer under the context.
bool fill_branch_features(const BranchFeatureContext& ctx,
                          Index j,
                          BranchFeatureVec& out);

// Lifted Branching / Gasse-style var+constraint lift: branch features plus
// extra aggregated constraint/edge lifts into LiftedFeatureVec.
bool fill_lifted_features(const BranchFeatureContext& ctx,
                          Index j,
                          LiftedFeatureVec& out);

// Fill features for every free fractional integer; candidates[k] <-> feats[k].
void fill_branch_features_batch(const BranchFeatureContext& ctx,
                                std::vector<Index>& candidates,
                                std::vector<BranchFeatureVec>& feats);

struct CutFeatureContext {
    const model::LpProblem* lp = nullptr;
    const std::vector<f64>* x = nullptr;
};

bool fill_cut_features(const CutFeatureContext& ctx,
                       const CutRow& cut,
                       CutFeatureVec& out);

struct GraphEdge {
    Index var = -1;
    Index con = -1;
    EdgeFeatureVec feats{};
};

struct BipartiteGraphSnapshot {
    std::vector<VarNodeFeatureVec> vars;
    std::vector<ConNodeFeatureVec> cons;
    std::vector<GraphEdge> edges;
};

struct TripartiteGraphSnapshot {
    BipartiteGraphSnapshot bipartite;
    std::vector<CutNodeFeatureVec> cuts;
    // Edges from cut nodes into variables (cut index in `con` field).
    std::vector<GraphEdge> cut_edges;
    // Cons↔Cut similarity edges (HGTSM meta-relations similar_to_*).
    // `var` = constraint index, `con` = cut index.
    std::vector<GraphEdge> cut_con_edges;
};

struct SeparatorState {
    bool gmi = true;
    bool lifted_cover = false;
    bool mir = false;
    bool clique = false;
    bool implied_bound = true;
    int last_round = 0;
    std::uint64_t last_candidates = 0;
    std::uint64_t last_accepted = 0;
    f64 last_bound_before = 0.0;
    f64 last_bound_after = 0.0;
};

// Build a bipartite var↔constraint snapshot at the current node domains.
// Fixed variables (lo==hi) are still listed so indices stay aligned with the
// model; their `active` flag is encoded in var features.
BipartiteGraphSnapshot build_bipartite_snapshot(
    const model::LpProblem& lp,
    const std::vector<f64>& col_lo,
    const std::vector<f64>& col_hi,
    const std::vector<f64>* x = nullptr);

TripartiteGraphSnapshot build_tripartite_snapshot(
    const model::LpProblem& lp,
    const std::vector<f64>& col_lo,
    const std::vector<f64>& col_hi,
    const std::vector<CutRow>& cut_rows,
    const std::vector<f64>* x = nullptr);

}  // namespace sor::search
