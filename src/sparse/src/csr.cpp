#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <numeric>
#include "sor/core/route_debug.hpp"

namespace sor::sparse {

void SparsePattern::validate() const {
    SOR_FN();
    if (n_rows_ < 0 || n_cols_ < 0)
        throw std::invalid_argument("SparsePattern: negative dimension");
    if (storage_->row_ptr.size() != static_cast<std::size_t>(n_rows_) + 1)
        throw std::invalid_argument("SparsePattern: row_ptr must have n_rows+1 entries");
    if (storage_->row_ptr.front() != 0)
        throw std::invalid_argument("SparsePattern: row_ptr[0] must be 0");
    if (storage_->row_ptr.back() != static_cast<Offset>(storage_->col_idx.size()))
        throw std::invalid_argument("SparsePattern: row_ptr.back() must equal nnz");
    for (Index r = 0; r < n_rows_; ++r) {
        if (storage_->row_ptr[r] > storage_->row_ptr[r + 1])
            throw std::invalid_argument("SparsePattern: row_ptr not nondecreasing");
    }
    for (Index c : storage_->col_idx) {
        if (c < 0 || c >= n_cols_)
            throw std::invalid_argument("SparsePattern: column index out of range");
    }
}

void SparsePattern::append_row(const std::vector<Index>& cols) {
    SOR_FN();
    detach_structure();
    // Empty pattern (default-constructed) has no row_ptr sentinel yet.
    if (storage_->row_ptr.empty()) {
        if (n_rows_ != 0)
            throw std::invalid_argument("SparsePattern::append_row: empty row_ptr");
        storage_->row_ptr.push_back(0);
    }
    for (const Index c : cols) {
        if (c < 0 || c >= n_cols_)
            throw std::invalid_argument("SparsePattern::append_row: column out of range");
    }
    for (std::size_t k = 1; k < cols.size(); ++k) {
        if (cols[k] <= cols[k - 1])
            throw std::invalid_argument(
                "SparsePattern::append_row: cols must be strictly ascending");
    }
    storage_->col_idx.insert(storage_->col_idx.end(), cols.begin(), cols.end());
    ++n_rows_;
    storage_->row_ptr.push_back(static_cast<Offset>(storage_->col_idx.size()));
}

void CsrMatrix::append_row(std::vector<Index> cols, std::vector<f64> row_vals) {
    SOR_FN();
    if (cols.size() != row_vals.size())
        throw std::invalid_argument("CsrMatrix::append_row: mismatched lengths");
    // Sort by column and sum duplicates, same contract as from_triplets.
    // Callers must start from a real LP (n_cols known via from_triplets).
    std::vector<std::size_t> order(cols.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { SOR_FN(); return cols[a] < cols[b]; });
    std::vector<Index> out_cols;
    std::vector<f64> out_vals;
    out_cols.reserve(cols.size());
    out_vals.reserve(cols.size());
    for (std::size_t i = 0; i < order.size();) {
        const Index c = cols[order[i]];
        f64 acc = 0.0;
        std::size_t j = i;
        while (j < order.size() && cols[order[j]] == c) {
            acc += row_vals[order[j]];
            ++j;
        }
        out_cols.push_back(c);
        out_vals.push_back(acc);
        i = j;
    }
    pattern.append_row(out_cols);
    vals.insert(vals.end(), out_vals.begin(), out_vals.end());
}

void SparsePattern::truncate_rows(Index keep) {
    SOR_FN();
    if (keep < 0 || keep > n_rows_)
        throw std::invalid_argument("SparsePattern::truncate_rows: keep out of range");
    if (keep == n_rows_) return;
    detach_structure();
    // storage_->row_ptr is non-decreasing, so the entry at `keep` is exactly the nnz of
    // the retained prefix; everything at or past it belongs to a dropped row.
    const Offset cut = storage_->row_ptr[static_cast<std::size_t>(keep)];
    storage_->col_idx.resize(static_cast<std::size_t>(cut));
    storage_->row_ptr.resize(static_cast<std::size_t>(keep) + 1);
    n_rows_ = keep;
}

void CsrMatrix::truncate_rows(Index keep) {
    SOR_FN();
    pattern.truncate_rows(keep);
    vals.resize(static_cast<std::size_t>(pattern.nnz()));
}

CsrMatrix from_triplets(Index n_rows, Index n_cols,
                        const std::vector<Index>& rows,
                        const std::vector<Index>& cols,
                        const std::vector<f64>& vals) {
    SOR_FN();
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
                  [&](std::size_t a, std::size_t c) { SOR_FN(); return col_idx[a] < col_idx[c]; });

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
