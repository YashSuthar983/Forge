#pragma once
#include "sor/model/lp.hpp"
#include "sor/search/cuts.hpp"

namespace sor::search {
// A binary articulation of the exact row-column incidence graph. Fixed
// columns are constants; every nonzero coefficient is an incidence edge.
// Iterative Tarjan traversal, O(rows + columns + nnz), no dense row cliques.
core::Index binary_separator(const model::LpProblem &lp);
// Finite required-integer articulation, with representable consecutive values.
core::Index integer_separator(const model::LpProblem &lp);
// Conservation cutsets, with exact source-row addition. Both orientations
// are supplied so a MIR/flow separator can use the boundary geometry.
std::vector<CutRow> network_cutset_rows(const model::LpProblem &,
                                        std::uint64_t work_budget);
std::vector<std::vector<core::Index>>
binary_product_blocks(const model::LpProblem &,
                      const std::vector<char> &integer, int max_columns);
} // namespace sor::search
