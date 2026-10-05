// SOR - tree / local cut lifecycle + GCS paper-complete selection (WP-C / E3).
//
// Under milp.policy=latest, separation is allowed past the root on a depth
// schedule. Cuts generated from a node tableau with local bounds are LOCAL:
// they are valid in that node's subtree and must not leak to siblings.
//
// Global Cut Selection (GCS, arXiv:2503.15847 spirit, clean-room):
//   - Tree-wide candidate pool with multi-node violation history
//   - Global scoring via node embeddings + learned promote/reinject policy
//   - Optional cut×node bipartite GNN scorer (SOR_GCS)
//   - Local (!globally_valid) cuts are never promoted into the global LP
//
// Classical efficacy scoring remains the ablation fallback when no model is
// loaded (heuristic multi-node score only).
#pragma once

#include "sor/search/cuts.hpp"
#include "sor/search/features.hpp"
#include "sor/search/milp_policy.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>
#include "sor/core/route_debug.hpp"

namespace sor::search {

using core::f64;
using core::Index;

inline constexpr int kGcsNodeFeatDim = 6;
inline constexpr int kGcsCutFeatDim = 8;
inline constexpr int kGcsEmbDim = 8;

using GcsNodeFeat = std::array<f64, kGcsNodeFeatDim>;
using GcsCutFeat = std::array<f64, kGcsCutFeatDim>;

struct TreeCutOptions {
    // Default on for Latest; Classical forces off in apply_policy_tree_cuts().
    bool enabled = true;
    // Separate at every node with depth <= always_depth.
    int always_depth = 1;
    // Deeper nodes: separate when depth % every_k == 0 (every_k <= 0 disables).
    int every_k = 12;
    // Skip separation when the dual bound has stalled this many nodes.
    std::uint64_t stall_skip_nodes = 4000;
    int max_cuts_per_node = 3;
    int max_rounds_per_node = 1;
    // Re-solve the node LP with the cuts just selected (warm dual simplex from
    // the node basis, rows appended non-binding). Every cut was derived over
    // this node's box, so the re-solved objective is a valid bound for the
    // whole subtree: it is folded into node.bound, which both children
    // inherit, and a node whose new bound reaches the cutoff is pruned. The
    // rows themselves are not carried into the children's LPs.
    bool resolve_with_local = true;
    // Iteration cap for that re-solve, as a multiple of the node LP's rows.
    f64 resolve_iter_per_row = 2.0;
    // A node lineage carries at most this many inherited local rows. Zero (the
    // default) keeps them out of descendants: measured on exp-1-500-5-5 and
    // beasleyC3, a subtree solved with inherited rows cannot use the prepared
    // LP session or the checkpoint cache (their rows differ from global_lp),
    // and throughput fell 4x (7923 -> 1834 nodes, worse bound in 20 s). The
    // machinery is complete and oracle-tested; it needs sessions keyed on the
    // local row set before it can pay.
    int max_local_rows = 0;
    // --- GCS ---------------------------------------------------------------
    bool gcs_enabled = true;
    int gcs_promote_max = 10;
    f64 gcs_min_score = 1e-6;
    // Cap stored candidates (evict lowest score when exceeded).
    int gcs_pool_max = 2000;
    // Re-check every stored globally-valid cut for violation at sep nodes.
    bool gcs_aggressive_observe = true;
    // Periodic re-injection of top GCS cuts into global_lp (0 = promote-only
    // at the separation site, no extra reinject cadence).
    std::uint64_t gcs_reinject_every_nodes = 50;
    int gcs_reinject_max = 5;
    // Optional on-disk GCS policy (SOR_GCS). Empty = heuristic global score.
    std::string gcs_model_path;
    bool gcs_prefer_heuristic = false;
    bool gcs_use_gnn = true;
};

// The COMPLETE derivation scope of a node-generated cut: every way it can
// depend on the subtree it was born in. A cut may leave that subtree only when
// all four hold, and each is recorded separately so a failure names its cause.
//
// The historical bug was testing only the first: bounds were checked, rows were
// not, and the separator's own derivation was trusted by name.
struct CutScopeConditions {
    // No column the derivation touched was branch-tightened: node bounds are
    // still the root bounds.
    bool bounds_are_root = false;
    // The model it was derived from carried no subtree-local row. node_lp is
    // global_lp plus the node's active local cuts, so a separator can consume
    // a locally-valid row while touching no tightened bound at all -- support
    // and bound provenance are both blind to that.
    bool rows_are_global = false;
    // The separator's derivation is established for global use. Name-based,
    // but a whitelist of what has been PROVEN, not an assumption that a family
    // is sound because it is row-derived.
    bool derivation_trusted = false;
    // Cheap last-line necessary condition: a globally valid cut admits every
    // integer-feasible point, so it admits the incumbent. Never sufficient.
    bool admits_incumbent = true;
};

// Single decision point for "may this cut leave the subtree that made it".
inline bool cut_may_leave_subtree(const CutScopeConditions& s) {
    return s.bounds_are_root && s.rows_are_global && s.derivation_trusted &&
           s.admits_incumbent;
}

// Whether a separator family's NODE-side derivation is established enough to
// hand a cut to the global model.
//
// ZH_ / MIR_ / FC_ / GMI_ stay off this whitelist. An earlier n5-3 campaign
// reported cuts that excluded a reference optimum, but that campaign is not a
// derivation proof: one sweep deleted the detector, and the later flags were
// references lying outside the generating node's bounds. A local cut may
// exclude a point outside its subtree. A node-feasible point excluded by a
// cut would refute that cut; failing to find one does not prove validity.
// These families stay quarantined until their derivations and scope are
// established for global use.
inline bool cut_family_derivation_trusted(const std::string& name) {
    return name.rfind("COV_", 0) == 0 || name.rfind("COVPC_", 0) == 0 ||
           name.rfind("COVGNS_", 0) == 0;
}

struct ManagedCut {
    CutRow row;
    bool global = false;
    int created_depth = 0;
    std::uint64_t created_node = 0;
    // Fingerprint for sibling / pool identity checks in tests.
    std::string id;
};

inline void apply_policy_tree_cuts(MilpPolicy policy, TreeCutOptions& tc) {
    if (policy == MilpPolicy::Classical)
        tc.enabled = false;
}

// Depth / stall schedule for per-node separation.
bool should_separate_at_node(int depth,
                             std::uint64_t nodes_since_dual_improve,
                             const TreeCutOptions& opts);

// Append `neu` onto `active` for a child node. Does not mutate sibling state.
void inherit_local_cuts(const std::vector<ManagedCut>& parent_active,
                        const std::vector<ManagedCut>& neu,
                        std::vector<ManagedCut>& child_active);

// True iff every cut in `cuts` with matching id appears in `active`.
bool local_cut_present(const std::vector<ManagedCut>& active,
                       const std::string& id);

// Extract CutRows for apply_cuts().
std::vector<CutRow> managed_to_rows(const std::vector<ManagedCut>& cuts);

// Content key so the same inequality accumulates multi-node stats.
// Printed with six significant digits, so it is not an exact identity.
// GcsPool treats a disagreement on this key as "keep the weaker scope".
// That is a conservative policy: a later local sighting does not
// mathematically invalidate an earlier global derivation of a truly
// identical inequality. Relaxing the policy needs an exact comparison
// and retained provenance.
std::string cut_content_id(const CutRow& row);

// --- GCS (global selection over tree-collected candidates) -----------------

struct GcsObservation {
    int depth = 0;
    f64 violation = 0.0;
    f64 gap_rel = 0.0;
    f64 dual_bound = 0.0;
};

struct GcsCandidate {
    CutRow row;
    CutFeatureVec feats{};
    f64 efficacy_sum = 0.0;
    int violation_nodes = 0;   // nodes where the cut was violated
    int seen_nodes = 0;
    bool globally_valid = false;
    std::string id;
    // Multi-node violation history + node embedding aggregates.
    std::vector<GcsObservation> history;
    f64 depth_sum = 0.0;
    int first_depth = -1;
    f64 best_violation = 0.0;
    f64 promote_label = 0.0;  // training: bound gain after promote
};

// Heuristic global score (fallback): mean_eff * (1 + log1p(violation_nodes)).
f64 gcs_heuristic_score(const GcsCandidate& c);

// Packed cut features for the scorer / GNN.
void fill_gcs_cut_feat(const GcsCandidate& c, GcsCutFeat& out);

// Node-side embedding features from a single observation site.
void fill_gcs_node_feat(const GcsObservation& ob, int pool_seen,
                        GcsNodeFeat& out);

struct GcsGraphWeights {
    int emb_dim = kGcsEmbDim;
    std::vector<f64> W_cut;   // emb * kGcsCutFeatDim
    std::vector<f64> b_cut;
    std::vector<f64> W_node;  // emb * kGcsNodeFeatDim
    std::vector<f64> b_node;
    std::vector<f64> W_msg_cn;  // emb * emb  cut←node
    std::vector<f64> W_msg_nc;  // emb * emb  node←cut
    std::vector<f64> W_policy;  // emb
    f64 b_policy = 0.0;
    void clear();
    void init_xavier(int emb, std::uint64_t seed);
    bool valid() const;
};

struct GcsModel {
    bool loaded = false;
    bool use_gnn = true;
    // Linear promote policy over GcsCutFeat (always present when loaded).
    std::vector<f64> W_linear;  // kGcsCutFeatDim
    f64 b_linear = 0.0;
    GcsGraphWeights gnn;

    void clear();
    f64 score_linear(const GcsCutFeat& f) const;
    f64 score_gnn(const GcsCandidate& c) const;
    // Model score if loaded; else heuristic.
    f64 score(const GcsCandidate& c, bool prefer_heuristic = false) const;
};

struct GcsSample {
    GcsCutFeat feats{};
    f64 label = 0.0;  // promote usefulness / bound delta
};

struct GcsCollector {
    std::vector<GcsSample> samples;
    std::size_t max_samples = 50000;
    void clear() { samples.clear(); }
    void add(const GcsCutFeat& f, f64 label);
};

struct GcsFitOptions {
    bool fit_gnn = true;
    int emb_dim = kGcsEmbDim;
    int sgd_epochs = 50;
    f64 sgd_lr = 0.05;
    f64 lasso_lambda = 1e-3;
    std::uint64_t seed = 0x47435321ull;  // "GCS!"
};

struct GcsDiagnostics {
    std::uint64_t promotes = 0;
    std::uint64_t reinjects = 0;
    std::uint64_t scored = 0;
    bool model_loaded = false;
    bool used_gnn = false;
    bool used_heuristic = true;
};

GcsModel fit_gcs(const GcsCollector& collector, const GcsFitOptions& opts);
bool save_gcs_model(const std::string& path, const GcsModel& model);
bool load_gcs_model(const std::string& path, GcsModel& model);

struct GcsPool {
    std::vector<GcsCandidate> cands;
    std::unordered_set<std::string> promoted_ids;
    int pool_max = 2000;
    GcsModel model;
    GcsDiagnostics diag;
    bool prefer_heuristic = false;

    void observe(const CutRow& row,
                 const CutFeatureVec& feats,
                 f64 efficacy,
                 bool violated,
                 bool globally_valid,
                 const std::string& id,
                 int depth = 0,
                 f64 gap_rel = 0.0,
                 f64 dual_bound = 0.0);
    // Re-evaluate stored globally-valid cuts at the current LP point.
    void touch_point(const std::vector<f64>& x, f64 viol_tol = 1e-6,
                     int depth = 0, f64 gap_rel = 0.0, f64 dual_bound = 0.0);
    // Rank globally-valid candidates by GCS score. Never returns local cuts.
    // Skips ids already in promoted_ids when skip_promoted is true.
    std::vector<CutRow> select_global(int max_keep, f64 min_score,
                                      bool skip_promoted = false);
    void mark_promoted(const std::vector<CutRow>& rows);
    void trim();
};

}  // namespace sor::search
