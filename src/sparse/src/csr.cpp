#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <numeric>

namespace sor::sparse {

void SparsePattern::validate() const {
    if (n_rows_ < 0 || n_cols_ < 0)
        throw std::invalid_argument("SparsePattern: negative dimension");
    if (row_ptr_.size() != static_cast<std::size_t>(n_rows_) + 1)
        throw std::invalid_argument("SparsePattern: row_ptr must have n_rows+1 entries");
    if (row_ptr_.front() != 0)
        throw std::invalid_argument("SparsePattern: row_ptr[0] must be 0");
    if (row_ptr_.back() != static_cast<Offset>(col_idx_.size()))
        throw std::invalid_argument("SparsePattern: row_ptr.back() must equal nnz");
    for (Index r = 0; r < n_rows_; ++r) {
        if (row_ptr_[r] > row_ptr_[r + 1])
            throw std::invalid_argument("SparsePattern: row_ptr not nondecreasing");
    }
    for (Index c : col_idx_) {
        if (c < 0 || c >= n_cols_)
            throw std::invalid_argument("SparsePattern: column index out of range");
    }
}

CsrMatrix from_triplets(Index n_rows, Index n_cols,
                        const std::vector<Index>& rows,
                        const std::vector<Index>& cols,
                        const std::vector<f64>& vals) {
    if (rows.size() != cols.size() || rows.size() != vals.size())
        throw std::invalid_argument("from_triplets: mismatched triplet lengths");

    // Count per row.
    std::vector<Offset> row_ptr(static_cast<std::size_t>(n_rows) + 1, 0);
    for (std::size_t k = 0; k < rows.size(); ++k) {
        const Index r = rows[k];
        if (r < 0 || r >= n_rows)
            throw std::invalid_argument("from_triplets: row index out of range");
        if (cols[k] < 0 || cols[k] >= n_cols)
            throw std::invalid_argument("from_triplets: column index out of range");
        ++row_ptr[static_cast<std::size_t>(r) + 1];
    }
    std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());

    // Scatter.
    std::vector<Index> col_idx(rows.size());
    std::vector<f64>   v(rows.size());
    std::vector<Offset> cursor(row_ptr.begin(), row_ptr.end() - 1);
    for (std::size_t k = 0; k < rows.size(); ++k) {
        const Offset dst = cursor[static_cast<std::size_t>(rows[k])]++;
        col_idx[static_cast<std::size_t>(dst)] = cols[k];
        v[static_cast<std::size_t>(dst)]       = vals[k];
    }

    // Sort each row by column, summing duplicates in place.
    std::vector<Offset> out_ptr(static_cast<std::size_t>(n_rows) + 1, 0);
    std::vector<Index>  out_col;
    std::vector<f64>    out_val;
    out_col.reserve(col_idx.size());
    out_val.reserve(v.size());

    std::vector<std::size_t> order;
    for (Index r = 0; r < n_rows; ++r) {
        const Offset b = row_ptr[static_cast<std::size_t>(r)];
        const Offset e = row_ptr[static_cast<std::size_t>(r) + 1];
        order.resize(static_cast<std::size_t>(e - b));
        std::iota(order.begin(), order.end(), static_cast<std::size_t>(b));
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t c) { return col_idx[a] < col_idx[c]; });

        for (std::size_t i = 0; i < order.size();) {
            const Index c = col_idx[order[i]];
            f64 acc = 0.0;
            std::size_t j = i;
            while (j < order.size() && col_idx[order[j]] == c) {
                acc += v[order[j]];
                ++j;
            }
            out_col.push_back(c);
            out_val.push_back(acc);
            i = j;
        }
        out_ptr[static_cast<std::size_t>(r) + 1] =
            static_cast<Offset>(out_col.size());
    }

    CsrMatrix m;
    m.pattern = SparsePattern(n_rows, n_cols, std::move(out_ptr), std::move(out_col));
    m.vals    = std::move(out_val);
    return m;
}

}  // namespace sor::sparse
