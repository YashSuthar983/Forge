// SOR - MRENS: Multi-Reference Relaxation Enforced Neighborhood Search
// (Bolusani, Mexi, Besançon & Turner, arXiv:2408.00718). Clean-room.
//
// LAYER L5. Builds a restricted box from several fractional LP references;
// never writes dual bounds. The caller solves the sub-MIP / dive and must
// re-validate any point against the original model.
//
// Also hosts BTBS-LNS-v1 (ICLR 2025 spirit) and CL-TLNS-v1 (arXiv:2412.08206
// spirit): destroy operators that emit NeighborhoodProblem boxes for Balans;
// repair is the existing sub-MIP path in bab.cpp - primal-only.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/lns.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct MrensOptions {
    bool enabled = true;
    double time_limit_s = 1.0;
    std::uint64_t max_nodes = 2000;
    // Require at least this fraction of integer columns fixed (or tightly
    // bracketed to a singleton) before the neighborhood is worth solving.
    f64 min_fix_frac = 0.25;
    int max_refs = 3;
    f64 int_tol = 1e-6;
    std::uint64_t min_interval = 400;  // nodes between bab call sites
};

struct MrensDiagnostics {
    std::uint64_t attempts = 0;
    std::uint64_t built = 0;
    std::uint64_t hits = 0;
    std::uint64_t refs_used = 0;
    std::uint64_t fixed = 0;
    std::uint64_t free_integer = 0;
    double seconds = 0.0;
};

// Box describing the MRENS sub-MILP (paper eq. (4)).
struct MrensNeighborhood {
    std::vector<f64> col_lo;
    std::vector<f64> col_hi;
    std::uint64_t fixed = 0;
    std::uint64_t free_integer = 0;
};

// Build the multi-reference RENS box. `refs` should contain 1-max_refs LP
// points of size n. When only one ref is supplied this reduces to classical
// RENS domain tightening. Returns false when degenerate or below min_fix_frac.
bool build_mrens_neighborhood(const model::LpProblem& mip,
                              const std::vector<std::vector<f64>>& refs,
                              const std::vector<f64>& node_lo,
                              const std::vector<f64>& node_hi,
                              const MrensOptions& opts,
                              MrensNeighborhood& out);

// Materialise as an LpProblem (box only - no extra rows).
model::LpProblem apply_mrens(const model::LpProblem& mip,
                             const MrensNeighborhood& nb);

// Synthesize extra reference points by diversified rounding of a single LP
// point when bab has no Lagromory / snapshot store yet.
void synthesize_mrens_refs(const model::LpProblem& mip,
                           const std::vector<f64>& x_lp,
                           int max_refs,
                           f64 int_tol,
                           std::uint32_t& rng,
                           std::vector<std::vector<f64>>& refs_out);

// ---------------------------------------------------------------------------
// BTBS-LNS-v1 (ICLR 2025 spirit, clean-room)
//
// Destroy: rank free integers by importance (caller proxies: frac / obj /
// pseudocost scores in `importance`, else |c_j| + LP fractionality); free a
// beam of size destroy_frac; tighten the rest around the incumbent with a
// binarized half-range scheme (Alg. 1 spirit). Repair: caller's sub-MIP.
// ---------------------------------------------------------------------------

struct BtbsOptions {
    bool enabled = true;
    double time_limit_s = 0.5;
    std::uint64_t max_nodes = 400;
    // Fraction of free integer columns left fully open (the "beam").
    f64 destroy_frac = 0.3;
    // Half-range tighten passes for non-beam integers (binarized tightening).
    int tighten_bits = 2;
    f64 int_tol = 1e-6;
    std::uint64_t min_interval = 400;
};

struct BtbsDiagnostics {
    std::uint64_t attempts = 0;
    std::uint64_t built = 0;
    std::uint64_t hits = 0;
    std::uint64_t fixed = 0;
    std::uint64_t free_integer = 0;
    std::uint64_t tightened = 0;
    double seconds = 0.0;
};

// `x_relax` / `importance` may be empty; when present they must match n cols.
bool build_btbs_neighborhood(const model::LpProblem& mip,
                             const std::vector<f64>& node_lo,
                             const std::vector<f64>& node_hi,
                             const std::vector<f64>& x_inc,
                             const std::vector<f64>& x_relax,
                             const std::vector<f64>& importance,
                             const BtbsOptions& opts,
                             NeighborhoodProblem& out);

// ---------------------------------------------------------------------------
// CL-TLNS-v1 (arXiv:2412.08206 spirit, clean-room)
//
// Contrastive / two-layer destroy: free the disagreement set between incumbent
// and LP (and optional second reference), then grow a margin up to destroy_frac.
// Repair: caller's capped sub-MIP (inner layer). Never dual-tightens.
// ---------------------------------------------------------------------------

struct ClTlnsOptions {
    bool enabled = true;
    double time_limit_s = 0.5;
    std::uint64_t max_nodes = 400;
    // Target free fraction after disagreement + margin.
    f64 destroy_frac = 0.25;
    f64 int_tol = 1e-6;
    f64 disagree_tol = 1e-6;
    std::uint64_t min_interval = 400;
};

struct ClTlnsDiagnostics {
    std::uint64_t attempts = 0;
    std::uint64_t built = 0;
    std::uint64_t hits = 0;
    std::uint64_t fixed = 0;
    std::uint64_t free_integer = 0;
    std::uint64_t disagree = 0;
    double seconds = 0.0;
};

// `x_ref2` may be empty. `x_lp` should be the current LP (size n).
bool build_cl_tlns_neighborhood(const model::LpProblem& mip,
                                const std::vector<f64>& node_lo,
                                const std::vector<f64>& node_hi,
                                const std::vector<f64>& x_inc,
                                const std::vector<f64>& x_lp,
                                const std::vector<f64>& x_ref2,
                                const ClTlnsOptions& opts,
                                NeighborhoodProblem& out);

inline const char* btbs_note() noexcept {
    return "BTBS-LNS-v1: beam free + binarized bound tightening "
           "(ICLR 2025 spirit); repair via sub-MIP";
}
inline const char* cl_tlns_note() noexcept {
    return "CL-TLNS-v1: contrastive disagreement+margin destroy "
           "(arXiv:2412.08206 spirit); repair via sub-MIP";
}

// Back-compat aliases used by older notes/tests.
inline const char* btbs_stub_note() noexcept { return btbs_note(); }
inline const char* cl_tlns_stub_note() noexcept { return cl_tlns_note(); }

}  // namespace sor::search
