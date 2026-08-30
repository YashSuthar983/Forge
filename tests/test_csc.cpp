// CSC transpose: values, ordering, and the pattern validator.
#include "sor/sparse/csc.hpp"

#include "test_helpers.hpp"

#include <random>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::core::Offset;
using sor::sparse::CscPattern;
using sor::sparse::from_triplets;
using sor::sparse::to_csc;

namespace {

void test_small() {
    //  [ 1 0 2 ]
    //  [ 0 3 0 ]
    const auto a = from_triplets(2, 3, {0, 0, 1}, {0, 2, 1}, {1.0, 2.0, 3.0});
    const auto c = to_csc(a);

    CHECK(c.n_rows() == 2);
    CHECK(c.n_cols() == 3);
    CHECK(c.nnz() == 3);
    const std::vector<Offset> want_ptr{0, 1, 2, 3};
    CHECK(c.pattern.col_ptr() == want_ptr);
    const std::vector<Index> want_row{0, 1, 0};
    CHECK(c.pattern.row_idx() == want_row);
    CHECK_NEAR(c.vals[0], 1.0, 0.0);
    CHECK_NEAR(c.vals[1], 3.0, 0.0);
    CHECK_NEAR(c.vals[2], 2.0, 0.0);
}

// Every entry of A must appear exactly once in the CSC, with row indices
// ascending inside each column. Ascending order is not cosmetic: the LU's
// row lookups binary-search, and determinism (commitment C3) requires that two
// equal matrices transpose to byte-identical arrays.
void test_random_roundtrip() {
    std::mt19937 rng(4242u);
    std::uniform_real_distribution<f64> val(-5.0, 5.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    for (const Index nr : {1, 3, 40}) {
        for (const Index nc : {1, 7, 33}) {
            std::vector<Index> rows, cols;
            std::vector<f64> vals;
            for (Index i = 0; i < nr; ++i)
                for (Index j = 0; j < nc; ++j)
                    if (u(rng) < 0.2) {
                        rows.push_back(i); cols.push_back(j);
                        vals.push_back(val(rng));
                    }
            const auto a = from_triplets(nr, nc, rows, cols, vals);
            const auto c = to_csc(a);
            CHECK(c.nnz() == a.nnz());

            // Dense comparison against A, built from the CSR side.
            std::vector<f64> dense_csr(static_cast<std::size_t>(nr) *
                                       static_cast<std::size_t>(nc), 0.0);
            const auto& rp = a.pattern.row_ptr();
            const auto& ci = a.pattern.col_idx();
            for (Index i = 0; i < nr; ++i)
                for (Offset k = rp[static_cast<std::size_t>(i)];
                     k < rp[static_cast<std::size_t>(i) + 1]; ++k)
                    dense_csr[static_cast<std::size_t>(i) *
                                  static_cast<std::size_t>(nc) +
                              static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])] =
                        a.vals[static_cast<std::size_t>(k)];

            const auto& cp = c.pattern.col_ptr();
            const auto& ri = c.pattern.row_idx();
            for (Index j = 0; j < nc; ++j) {
                Index prev = -1;
                for (Offset k = cp[static_cast<std::size_t>(j)];
                     k < cp[static_cast<std::size_t>(j) + 1]; ++k) {
                    const Index i = ri[static_cast<std::size_t>(k)];
                    CHECK(i > prev);          // strictly ascending, no duplicates
                    prev = i;
                    CHECK_NEAR(c.vals[static_cast<std::size_t>(k)],
                               dense_csr[static_cast<std::size_t>(i) *
                                             static_cast<std::size_t>(nc) +
                                         static_cast<std::size_t>(j)],
                               0.0);
                }
            }
        }
    }
}

void test_validator_rejects_malformed() {
    CHECK_THROWS(CscPattern(2, 2, {0, 1}, {0}));            // col_ptr too short
    CHECK_THROWS(CscPattern(2, 2, {1, 1, 2}, {0, 1}));      // col_ptr[0] != 0
    CHECK_THROWS(CscPattern(2, 2, {0, 1, 5}, {0, 1}));      // back() != nnz
    CHECK_THROWS(CscPattern(2, 2, {0, 2, 1}, {0, 1}));      // decreasing
    CHECK_THROWS(CscPattern(2, 2, {0, 1, 2}, {0, 9}));      // row out of range
}

void test_empty() {
    const auto a = from_triplets(0, 0, {}, {}, {});
    const auto c = to_csc(a);
    CHECK(c.nnz() == 0);
    CHECK(c.n_rows() == 0);
    CHECK(c.n_cols() == 0);
}

}  // namespace

int main() {
    test_small();
    test_random_roundtrip();
    test_validator_rejects_malformed();
    test_empty();
    return sor::test::finish("test_csc");
}
