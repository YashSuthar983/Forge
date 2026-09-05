// SOR — compressed-sparse-column pattern and matrix.
//
// LAYER L1. Depends only on sor_core.
//
// CSR answers "what is in row i". CSC answers "what is in column j", and the
// simplex method asks only the second question: the basis matrix is assembled
// from the columns of the basic variables, and the entering column is the input
// to FTRAN. With a row-major-only matrix, fetching one column costs a scan of
// the entire matrix, so a simplex iteration would be O(nnz) in column access
// alone.
//
// The first-order engine wants this too, for a different reason: its spmv_t is
// currently a scatter over CSR (y[col] += ...), which needs atomics to run in
// parallel. The same product over CSC is a gather, which does not.
// See docs/architecture.md §3.2 (FO / SpMV path).
#pragma once

#include "sor/sparse/csr.hpp"

#include <utility>
#include <vector>

namespace sor::sparse {

struct CscMatrix;
class CscPattern;
CscMatrix to_csc(const CsrMatrix& a);

// Compressed-sparse-column pattern: col_ptr has n_cols+1 entries.
class CscPattern {
public:
    CscPattern() = default;

    CscPattern(Index n_rows, Index n_cols,
               std::vector<Offset> col_ptr,
               std::vector<Index> row_idx)
        : n_rows_(n_rows), n_cols_(n_cols),
          col_ptr_(std::move(col_ptr)), row_idx_(std::move(row_idx)) {
        validate();
    }

    Index n_rows() const noexcept { return n_rows_; }
    Index n_cols() const noexcept { return n_cols_; }
    Offset nnz()   const noexcept { return static_cast<Offset>(row_idx_.size()); }

    const std::vector<Offset>& col_ptr() const noexcept { return col_ptr_; }
    const std::vector<Index>&  row_idx() const noexcept { return row_idx_; }

    // Throws std::invalid_argument on a malformed pattern.
    void validate() const;

private:
    struct TrustedCsrTransposeTag {};
    CscPattern(Index n_rows, Index n_cols,
               std::vector<Offset> col_ptr,
               std::vector<Index> row_idx,
               TrustedCsrTransposeTag)
        : n_rows_(n_rows), n_cols_(n_cols),
          col_ptr_(std::move(col_ptr)), row_idx_(std::move(row_idx)) {}

    friend CscMatrix to_csc(const CsrMatrix& a);

    Index n_rows_ = 0;
    Index n_cols_ = 0;
    std::vector<Offset> col_ptr_;
    std::vector<Index>  row_idx_;
};

struct CscMatrix {
    CscPattern       pattern;
    std::vector<f64> vals;

    Index n_rows() const noexcept { return pattern.n_rows(); }
    Index n_cols() const noexcept { return pattern.n_cols(); }
    Offset nnz()   const noexcept { return pattern.nnz(); }
};

// Transpose a CSR matrix into CSC. Row indices within each column come out
// ascending, which the triangular solves rely on for cache behaviour and which
// makes the result canonical (so two equal matrices transpose identically —
// commitment C3, determinism).
CscMatrix to_csc(const CsrMatrix& a);

}  // namespace sor::sparse
