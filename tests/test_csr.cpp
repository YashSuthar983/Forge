#include "sor/sparse/csr.hpp"
#include "test_helpers.hpp"
#include "sor/core/route_debug.hpp"

using namespace sor;
using sor::core::f64;
using sor::core::Index;

int main() {
    SOR_FN();
    // Duplicate (row, col) entries must be summed, and columns sorted.
    {
        std::vector<Index> rows{0, 0, 1, 0, 2, 1};
        std::vector<Index> cols{2, 0, 1, 2, 0, 1};
        std::vector<f64>   vals{1.0, 2.0, 3.0, 0.5, 4.0, -3.0};
        const auto m = sparse::from_triplets(3, 3, rows, cols, vals);

        CHECK(m.n_rows() == 3);
        CHECK(m.n_cols() == 3);
        // row0: col0=2.0, col2=1.5 ; row1: col1=0.0 ; row2: col0=4.0
        CHECK(m.nnz() == 4);
        CHECK(m.pattern.col_idx()[0] == 0);
        CHECK(m.pattern.col_idx()[1] == 2);
        CHECK_NEAR(m.vals[0], 2.0, 1e-15);
        CHECK_NEAR(m.vals[1], 1.5, 1e-15);
        CHECK_NEAR(m.vals[2], 0.0, 1e-15);   // 3.0 + (-3.0)
        CHECK_NEAR(m.vals[3], 4.0, 1e-15);
    }

    // Bound/cost/scaling copies share structure, but cut edits detach and
    // preserve every existing snapshot and its version token.
    {
        const auto original = sparse::from_triplets(2, 3, {0, 1}, {0, 2}, {1, 2});
        auto overlay = original;
        const auto token = original.pattern.structure_token();
        CHECK(overlay.pattern.structure_token() == token);
        CHECK(overlay.pattern.row_ptr().data() == original.pattern.row_ptr().data());
        overlay.vals[0] = 7;
        CHECK(original.vals[0] == 1);
        CHECK(overlay.pattern.structure_token() == token);
        overlay.append_row({1}, {3});
        CHECK(overlay.pattern.structure_token() != token);
        CHECK(original.n_rows() == 2 && original.nnz() == 2);
        auto snapshot = overlay;
        const auto appended_token = snapshot.pattern.structure_token();
        overlay.truncate_rows(1);
        CHECK(overlay.pattern.structure_token() != appended_token);
        CHECK(snapshot.n_rows() == 3 && snapshot.nnz() == 3);
        original.pattern.validate(); snapshot.pattern.validate(); overlay.pattern.validate();
    }

    // Malformed patterns must be rejected at construction.
    {
        CHECK_THROWS(sparse::SparsePattern(2, 2, {0, 1}, {0}));          // short row_ptr
        CHECK_THROWS(sparse::SparsePattern(1, 2, {1, 1}, {}));           // row_ptr[0] != 0
        CHECK_THROWS(sparse::SparsePattern(1, 2, {0, 1}, {5}));          // col out of range
        CHECK_THROWS(sparse::SparsePattern(2, 2, {0, 2, 1}, {0, 1}));    // not nondecreasing
    }

    // Empty matrix is legal.
    {
        const auto m = sparse::from_triplets(0, 0, {}, {}, {});
        CHECK(m.nnz() == 0);
    }

    // In-place row append must grow CSR without rebuilding prior rows.
    {
        auto m = sparse::from_triplets(1, 3, {0, 0}, {0, 2}, {1.0, 2.0});
        m.append_row({1, 0}, {4.0, 3.0});  // unsorted; sorted to col0, col1
        CHECK(m.n_rows() == 2);
        CHECK(m.n_cols() == 3);
        CHECK(m.nnz() == 4);
        CHECK(m.pattern.row_ptr()[0] == 0);
        CHECK(m.pattern.row_ptr()[1] == 2);
        CHECK(m.pattern.row_ptr()[2] == 4);
        CHECK(m.pattern.col_idx()[2] == 0);
        CHECK(m.pattern.col_idx()[3] == 1);
        CHECK_NEAR(m.vals[2], 3.0, 1e-15);
        CHECK_NEAR(m.vals[3], 4.0, 1e-15);
        m.append_row({1, 1}, {1.0, 5.0});  // duplicate col summed
        CHECK(m.n_rows() == 3);
        CHECK(m.nnz() == 5);
        CHECK(m.pattern.col_idx()[4] == 1);
        CHECK_NEAR(m.vals[4], 6.0, 1e-15);
    }

    // truncate_rows is the exact inverse of append_row, including the
    // validate() invariants: append N rows, truncate back, and the pattern is
    // bit-identical to the one before.
    {
        auto m = sparse::from_triplets(2, 3, {0, 1, 1}, {0, 1, 2},
                                       {1.0, 2.0, 3.0});
        const auto rp0 = m.pattern.row_ptr();
        const auto ci0 = m.pattern.col_idx();
        const auto v0  = m.vals;

        m.append_row({0, 2}, {7.0, 8.0});
        m.append_row({1}, {9.0});
        CHECK(m.n_rows() == 4);
        CHECK(m.nnz() == 6);

        m.truncate_rows(2);
        CHECK(m.n_rows() == 2);
        CHECK(m.n_cols() == 3);
        CHECK(m.nnz() == 3);
        CHECK(m.pattern.row_ptr() == rp0);
        CHECK(m.pattern.col_idx() == ci0);
        CHECK(m.vals == v0);
        m.pattern.validate();

        // Truncating to the current size is a no-op, and to zero empties it
        // without destroying the row_ptr sentinel.
        m.truncate_rows(2);
        CHECK(m.n_rows() == 2 && m.nnz() == 3);
        m.truncate_rows(0);
        CHECK(m.n_rows() == 0);
        CHECK(m.nnz() == 0);
        CHECK(m.pattern.row_ptr().size() == 1);
        m.pattern.validate();
        // ...and the emptied matrix still accepts a fresh row.
        m.append_row({2}, {5.0});
        CHECK(m.n_rows() == 1 && m.nnz() == 1);
        CHECK_NEAR(m.vals[0], 5.0, 1e-15);

        CHECK_THROWS(m.truncate_rows(-1));
        CHECK_THROWS(m.truncate_rows(9));
    }

    return sor::test::finish("test_csr");
}
