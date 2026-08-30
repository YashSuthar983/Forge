#include "sor/sparse/csc.hpp"

#include <stdexcept>
#include <string>

namespace sor::sparse {

void CscPattern::validate() const {
    if (n_rows_ < 0 || n_cols_ < 0)
        throw std::invalid_argument("CscPattern: negative dimension");
    if (col_ptr_.size() != static_cast<std::size_t>(n_cols_) + 1)
        throw std::invalid_argument("CscPattern: col_ptr must have n_cols+1 entries");
    if (col_ptr_.front() != 0)
        throw std::invalid_argument("CscPattern: col_ptr[0] must be 0");
    if (col_ptr_.back() != static_cast<Offset>(row_idx_.size()))
        throw std::invalid_argument("CscPattern: col_ptr.back() must equal nnz");

    for (Index j = 0; j < n_cols_; ++j) {
        const Offset b = col_ptr_[static_cast<std::size_t>(j)];
        const Offset e = col_ptr_[static_cast<std::size_t>(j) + 1];
        if (e < b)
            throw std::invalid_argument("CscPattern: col_ptr not non-decreasing at column " +
                                        std::to_string(j));
        for (Offset k = b; k < e; ++k) {
            const Index i = row_idx_[static_cast<std::size_t>(k)];
            if (i < 0 || i >= n_rows_)
                throw std::invalid_argument("CscPattern: row index out of range in column " +
                                            std::to_string(j));
        }
    }
}

CscMatrix to_csc(const CsrMatrix& a) {
    const Index nr = a.n_rows();
    const Index nc = a.n_cols();
    const auto& rp = a.pattern.row_ptr();
    const auto& ci = a.pattern.col_idx();

    if (a.vals.size() != static_cast<std::size_t>(a.nnz()))
        throw std::invalid_argument("to_csc: vals size does not match pattern nnz");

    // Counting sort by column: one pass to count, one to place. Because the CSR
    // rows are visited in ascending order, the row indices land in each column
    // in ascending order without a second sort.
    std::vector<Offset> col_ptr(static_cast<std::size_t>(nc) + 1, 0);
    for (const Index j : ci) ++col_ptr[static_cast<std::size_t>(j) + 1];
    for (Index j = 0; j < nc; ++j)
        col_ptr[static_cast<std::size_t>(j) + 1] += col_ptr[static_cast<std::size_t>(j)];

    std::vector<Index> row_idx(static_cast<std::size_t>(a.nnz()));
    std::vector<f64>   vals(static_cast<std::size_t>(a.nnz()));
    std::vector<Offset> next(col_ptr.begin(), col_ptr.end() - 1);

    for (Index i = 0; i < nr; ++i) {
        for (Offset k = rp[static_cast<std::size_t>(i)];
             k < rp[static_cast<std::size_t>(i) + 1]; ++k) {
            const auto j = static_cast<std::size_t>(ci[static_cast<std::size_t>(k)]);
            const auto dst = static_cast<std::size_t>(next[j]++);
            row_idx[dst] = i;
            vals[dst]    = a.vals[static_cast<std::size_t>(k)];
        }
    }

    CscMatrix out;
    out.pattern = CscPattern(nr, nc, std::move(col_ptr), std::move(row_idx));
    out.vals    = std::move(vals);
    return out;
}

}  // namespace sor::sparse
