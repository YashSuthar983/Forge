#pragma once

#include "sor/model/lp.hpp"
#include <cstdint>
#include <vector>

namespace sor::search {

class ConflictGraph;

// Integer-domain equalities. complement means first = 1 - second;
// otherwise first = second. These are never asserted for LP relaxations.
struct BinaryRelation {
    core::Index first = -1, second = -1;
    bool complement = false;
};

struct RowSupportProbeOptions {
    bool enabled = true;
    int max_row_binaries = 8;
    int min_row_binaries = 2;
    std::uint64_t max_total_row_visits = 2000000;
    std::uint64_t max_row_visits_per_branch = 10000;
    std::uint64_t max_graph_visits_per_branch = 20000;
    std::uint64_t max_total_graph_visits = 4000000;
    std::uint64_t max_candidate_rows = 400;
    int max_consecutive_fails = 50;
    // Avoid spending a pass on many nearly identical binary groups. One
    // retains exact-set deduplication only; one half allows at most half of
    // the smaller group to overlap any previously enumerated group.
    double max_row_overlap = 1.0;
    double time_limit_s = 0.25;
    core::f64 tol = 1e-9;
};

struct RowSupportProbeDiagnostics {
    std::uint64_t rows_candidate = 0, rows_enumerated = 0;
    std::uint64_t assignments = 0, assignments_infeasible = 0;
    std::uint64_t rows_overlap_skipped = 0;
    std::uint64_t fixings = 0, tightenings = 0, row_visits = 0;
    std::uint64_t graph_visits = 0;
    bool truncated = false, infeasible = false;
    double ms = 0.0;
    std::vector<BinaryRelation> relations;
};

// Branch only on a short binary row, propagate through the complete model,
// undo every branch, then publish the union of the surviving boxes. Budget
// interruption discards the incomplete row's hull and relations. A truncated
// propagation is an overestimate, so it may lose deductions but cannot turn
// an unproved branch into an infeasibility claim.
RowSupportProbeDiagnostics probe_row_supports(const model::LpProblem &lp,
                                              std::vector<core::f64> &lo,
                                              std::vector<core::f64> &hi,
                                              const RowSupportProbeOptions &opts,
                                              const ConflictGraph* graph = nullptr);

} // namespace sor::search
