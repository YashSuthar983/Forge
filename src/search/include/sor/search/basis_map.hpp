// Carrying a simplex basis from a parent LP onto a child LP that has the same
// columns and a subset of the parent's rows: rows [0, base_rows) unchanged, then
// `fresh_rows` rows the parent never had (their slacks start basic), then the
// parent rows named in `extra_parent_rows` (in that order). Rows of the parent
// that the child does not have are dropped.
//
// Dropping a row whose slack was BASIC loses one basic variable and the child's
// row count drops with it, so the counts still match. Dropping a BINDING row
// (slack nonbasic) leaves one basic variable too many. Which ones may leave is
// a property of the matrix, not of the point: the remaining columns must be
// nonsingular over the child's rows, so the choice is made by LU over the
// child (basis_rank_selection, the same rule as fit_stored_basis). A leaving
// column is made nonbasic at the child bound nearer its parent value.
// The result is a starting basis: the child's simplex repairs primal
// infeasibility under the child's tighter box (that is what the dual simplex
// is for); it is singular only if the child's rows are numerically rank
// deficient on it, and then the vacant rows' logicals complete it.
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"

#include <optional>
#include <vector>

namespace sor::search {

struct MilpPresolveResult;
// Transfer candidate basis columns through the transformation journal and
// select independent images on the actual reduced matrix. No parent factor
// or point is reused after a structural transformation.
std::optional<engines::SimplexBasis> map_basis_through_presolve(
    const engines::SimplexBasis& parent, const MilpPresolveResult& pre);

std::optional<engines::SimplexBasis> map_basis_to_child(
    const engines::SimplexBasis& parent, core::Index n_struct, core::Index parent_rows,
    core::Index base_rows, core::Index fresh_rows,
    const std::vector<core::Index>& extra_parent_rows,
    const std::vector<core::f64>& parent_x, const model::LpProblem& child);

// A stored warm start describes the rows it was solved over. Its status vector
// has one entry per column of [A | -I] over those rows; its basic list may hold
// MORE columns than rows: when rows were deleted while their logicals were
// nonbasic (binding), every basic column stays listed and the selection of
// the ones to leave is deferred to fit_stored_basis(), which is where the
// matrix is at hand. Such a basis never reaches the engine unfitted: the
// engine's size check would turn it into a cold start.
//
// The basis was solved over [g_b global rows][L local rows], where those g_b
// rows are the first g_b of `lp`'s first `n_global` rows (global rows are only
// appended, and a deletion rewrites stored bases) and the L local rows are
// `lp`'s last L rows. fit_stored_basis rewrites it for exactly `lp`:
//   - logicals of the global rows appended since (g_b..n_global) enter basic:
//     a unit column on a new row keeps any nonsingular basis nonsingular;
//   - surplus basic columns leave. Deleting rows R from a nonsingular B leaves
//     its columns with full row rank, so exactly |R| of them must leave, and a
//     set K may leave iff (B^-1)[K,R] is nonsingular. LU of the basic columns
//     over the current rows, padded with |R| empty rows (which cannot pivot),
//     reports such a K as its singular slots without B or the deleted rows.
//     Each leaving column becomes nonbasic at a finite bound (the lower one
//     when both are; the dual simplex flips a boxed column for free).
// A numerically rank-deficient remainder is completed with logicals of the
// rows LU left vacant, the engine's own singular repair. Returns false (and
// leaves `b` untouched) when the layout does not match `lp`.
struct BasisFitStats {
    core::Index appended = 0;   // logicals of new global rows made basic
    core::Index dropped = 0;    // surplus basics made nonbasic
    core::Index repaired = 0;   // logicals substituted for numerically singular columns
};
bool fit_stored_basis(const model::LpProblem& lp, core::Index n_global, core::Index local_rows,
                      engines::SimplexBasis& b, const std::vector<core::f64>& col_lo,
                      const std::vector<core::f64>& col_hi, BasisFitStats* stats = nullptr);

// Columns of `basic` (indices into [A | -I] over lp's first `rows` rows) that
// LU over those rows, padded with basic.size() - rows empty rows, could not
// pivot, and the real rows it left vacant. Exposed for tests.
void basis_rank_selection(const model::LpProblem& lp, core::Index rows,
                          const std::vector<core::Index>& basic,
                          std::vector<core::Index>& leaving_slots,
                          std::vector<core::Index>& vacant_rows);

}  // namespace sor::search
