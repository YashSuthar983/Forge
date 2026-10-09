#pragma once
#include "forge/core/cancel.hpp"
#include "forge/model/lp.hpp"
#include <optional>

namespace forge::search {
struct ConditionalComponent {
  model::LpProblem model;
  // Owned free columns precede zero-cost copies of fixed/boundary columns.
  std::vector<core::Index> columns;
  std::size_t owned = 0;
  // (index in the separator tuple, index in this model).
  std::vector<std::pair<std::size_t, core::Index>> boundary;
};
struct ConditionalDecomposition {
  std::vector<core::Index> separator;
  std::vector<ConditionalComponent> components;
  std::vector<core::Index> independent_cost_columns;
  std::vector<core::Index> constant_rows;
};
// Compile finite integer articulation / row-boundary candidates once. The
// returned row ownership is exact: no residual row crosses two components.
// Boundary columns stay explicit, preserving sides without rounded shifts.
std::optional<ConditionalDecomposition>
compile_conditional_components(const model::LpProblem &,
                               double analysis_time_limit_s,
                               const core::CancelToken *cancel = nullptr);
// Necessary projected rows in the boundary coordinates. Non-boundary
// activities are eliminated with exact sums and outward-rounded sides;
// these rows propagate whole master regions before any complete context.
model::LpProblem compile_conditional_master(const model::LpProblem&,
                                            const std::vector<core::Index>& separator);
} // namespace forge::search
