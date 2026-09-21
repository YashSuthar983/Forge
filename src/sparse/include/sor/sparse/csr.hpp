// SOR - sparse CSR pattern and matrix.
//
// LAYER L1. Depends only on sor_core.
//
// The pattern is deliberately separated from the values: batched kernels share
// one pattern across N value/vector sets, and the
// first-order engine reuses a fixed pattern across every iteration.
#pragma once

#include "sor/core/result.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace sor::sparse {

using core::f64;
using core::Index;
using core::Offset;

// Compressed-sparse-row pattern: row_ptr has n_rows+1 entries.
class SparsePattern {
public:
    SparsePattern() = default;

    SparsePattern(Index n_rows, Index n_cols,
                  std::vector<Offset> row_ptr,
                  std::vector<Index> col_idx)
        : n_rows_(n_rows), n_cols_(n_cols),
          row_ptr_(std::move(row_ptr)), col_idx_(std::move(col_idx)) {
        validate();
    }

    Index n_rows() const noexcept { return n_rows_; }
    Index n_cols() const noexcept { return n_cols_; }
    Offset nnz()   const noexcept { return static_cast<Offset>(col_idx_.size()); }

    const std::vector<Offset>& row_ptr() const noexcept { return row_ptr_; }
    const std::vector<Index>&  col_idx() const noexcept { return col_idx_; }

    // Append one row. `cols` must already be sorted ascending with no
    // duplicates, and every index in [0, n_cols_). O(|cols|).
    void append_row(const std::vector<Index>& cols);

    // Throws std::invalid_argument on a malformed pattern. Called on
    // construction so no kernel ever sees an inconsistent pattern.
    void validate() const;

private:
    Index n_rows_ = 0;
    Index n_cols_ = 0;
    std::vector<Offset> row_ptr_;
    std::vector<Index>  col_idx_;
};

// A CSR matrix is a pattern plus values, in pattern order.
struct CsrMatrix {
    SparsePattern     pattern;
    std::vector<f64>  vals;

    Index n_rows() const noexcept { return pattern.n_rows(); }
    Index n_cols() const noexcept { return pattern.n_cols(); }
    Offset nnz()   const noexcept { return pattern.nnz(); }

    // Append one row in place. Columns are sorted and duplicates summed, so
    // callers may pass unsorted cut supports. O(k log k) in the row nnz -
    // not O(matrix nnz) - which is what makes per-nogood learning affordable.
    void append_row(std::vector<Index> cols, std::vector<f64> row_vals);
};

// Build CSR from unordered triplets, summing duplicate (row, col) entries.
// Column indices within each row come out ascending, which the hypersparse
// solves in Phase 1 will rely on.
CsrMatrix from_triplets(Index n_rows, Index n_cols,
                        const std::vector<Index>& rows,
                        const std::vector<Index>& cols,
                        const std::vector<f64>& vals);

}  // namespace sor::sparse
