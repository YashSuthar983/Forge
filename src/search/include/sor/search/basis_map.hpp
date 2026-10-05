// Carrying a simplex basis from a parent LP onto a child LP that has the same
// columns and a subset of the parent's rows: rows [0, base_rows) unchanged, then
// `fresh_rows` rows the parent never had (their slacks start basic), then the
// parent rows named in `extra_parent_rows` (in that order). Rows of the parent
// that the child does not have are dropped.
//
// Dropping a row whose slack was BASIC loses one basic variable and the child's
// row count drops with it, so the counts still match. Dropping a BINDING row
// (slack nonbasic) leaves one basic variable too many: that many structural
// basics -- those nearest a bound in the parent's point -- are made nonbasic at
// that bound. The result is a starting basis only: the child's simplex repairs
// it (it may be primal infeasible under the child's tighter box, which is what
// the dual simplex is for), and a singular choice falls to its basis repair.
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/simplex.hpp"

#include <optional>
#include <vector>

namespace sor::search {

std::optional<engines::SimplexBasis> map_basis_to_child(
    const engines::SimplexBasis& parent, core::Index n_struct, core::Index parent_rows,
    core::Index base_rows, core::Index fresh_rows,
    const std::vector<core::Index>& extra_parent_rows,
    const std::vector<core::f64>& parent_x, const std::vector<core::f64>& col_lo,
    const std::vector<core::f64>& col_hi);

}  // namespace sor::search
