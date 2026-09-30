// Optional bound snapshots and invocation provenance for MILP diagnostics.
#pragma once

#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::search::detail {

std::uint64_t next_milp_invocation_id();
void record_milp_bounds(const model::LpProblem& model, std::uint64_t invocation,
                        std::uint64_t parent, int sub_mip_depth, const char* phase,
                        const char* snapshot_base, const std::vector<core::f64>& lo,
                        const std::vector<core::f64>& hi);

}  // namespace sor::search::detail
