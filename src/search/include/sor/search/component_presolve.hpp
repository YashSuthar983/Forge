// SOR - disconnected-component MIP root reductions (WP-F).
//
// Columns are connected when they share a constraint with nonzero coefficient.
// Independent components can be tightened separately (dual-fix + FBBT, and
// optional hull enumeration on tiny pure-binary components) and the resulting
// boxes merged. Never removes a globally feasible integer point.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct ComponentPresolveOptions {
    bool enabled = true;
    // Independent dual-fix + FBBT per component with ≥2 columns.
    bool tighten = true;
    int dual_fix_rounds = 4;
    int fbbt_rounds = 12;
    // Enumerate feasible 0/1 assignments of tiny all-binary components and
    // take the coordinate hull (safe domain shrink).
    bool enumerate_tiny = true;
    Index max_enum_bins = 12;
    f64 tol = 1e-9;
};

struct ComponentPresolveDiagnostics {
    std::uint64_t n_components = 0;
    std::uint64_t multi_col_components = 0;
    std::uint64_t largest_component = 0;
    std::uint64_t dual_fixings = 0;
    std::uint64_t fbbt_tightenings = 0;
    std::uint64_t enum_components = 0;
    std::uint64_t enum_fixings = 0;
    bool infeasible = false;
    bool disconnected = false;  // true iff ≥2 multi-column components
};

// Detects column components via bipartite constraint connectivity, optionally
// tightens each independently, merges into col_lo/hi.
ComponentPresolveDiagnostics apply_component_presolve(
    const model::LpProblem& lp, std::vector<f64>& col_lo,
    std::vector<f64>& col_hi, const ComponentPresolveOptions& opts);

}  // namespace sor::search
