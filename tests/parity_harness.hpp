// Shared backend parity harness.
//
// Exercises every KernelBackend method on fixed, seeded inputs and compares a
// candidate backend against the CPU reference. test_backend_parity.cpp uses it
// for CPU-vs-CPU (a self-consistency and API check).
//
// docs/architecture.md §3.3: "A CUDA kernel that disagrees with the CPU
// reference fails the build."
#pragma once

#include "sor/backend/kernel_backend.hpp"
#include "sor/sparse/csr.hpp"
#include "test_helpers.hpp"

#include <random>
#include <vector>

namespace sor::test {

using backend::BatchView;
using backend::DeviceBuffer;
using backend::KernelBackend;
using core::f64;
using core::Index;
using sparse::CsrMatrix;

// Deterministic sparse matrix: mt19937 with a fixed seed is reproducible across
// platforms, so parity failures are always reproducible.
inline CsrMatrix make_test_matrix(Index n_rows, Index n_cols, int nnz_per_row,
                                  unsigned seed = 20260828u) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<Index> col(0, n_cols - 1);
    std::uniform_real_distribution<f64> val(-2.0, 2.0);

    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    rows.reserve(static_cast<std::size_t>(n_rows) * nnz_per_row);
    for (Index r = 0; r < n_rows; ++r) {
        // Guarantee a diagonal-ish entry so no row is empty.
        rows.push_back(r);
        cols.push_back(r % n_cols);
        vals.push_back(1.0 + val(rng) * 0.1);
        for (int k = 1; k < nnz_per_row; ++k) {
            rows.push_back(r);
            cols.push_back(col(rng));
            vals.push_back(val(rng));
        }
    }
    return sparse::from_triplets(n_rows, n_cols, rows, cols, vals);
}

inline std::vector<f64> make_vector(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f64> d(-1.0, 1.0);
    std::vector<f64> v(n);
    for (auto& e : v) e = d(rng);
    return v;
}

// Runs every kernel on both backends and compares. `tol` is relative.
inline void run_parity(KernelBackend& ref, KernelBackend& cand, f64 tol) {
    constexpr Index kRows = 137;
    constexpr Index kCols = 91;
    const CsrMatrix A = make_test_matrix(kRows, kCols, 5);

    DeviceBuffer<f64> vals(A.vals.data(), A.vals.size());

    // ---- spmv ----
    {
        const auto xv = make_vector(static_cast<std::size_t>(kCols), 11u);
        DeviceBuffer<f64> x(xv.data(), xv.size());
        DeviceBuffer<f64> y_ref, y_cand;
        ref.spmv(A.pattern, vals, x, y_ref);
        cand.spmv(A.pattern, vals, x, y_cand);
        CHECK(y_ref.size() == y_cand.size());
        for (std::size_t i = 0; i < y_ref.size(); ++i)
            CHECK_NEAR(y_cand[i], y_ref[i], tol);
    }

    // ---- spmv_t ----
    {
        const auto xv = make_vector(static_cast<std::size_t>(kRows), 12u);
        DeviceBuffer<f64> x(xv.data(), xv.size());
        DeviceBuffer<f64> y_ref, y_cand;
        ref.spmv_t(A.pattern, vals, x, y_ref);
        cand.spmv_t(A.pattern, vals, x, y_cand);
        CHECK(y_ref.size() == y_cand.size());
        for (std::size_t i = 0; i < y_ref.size(); ++i)
            CHECK_NEAR(y_cand[i], y_ref[i], tol);
    }

    // ---- project_box ----
    {
        const auto xv = make_vector(static_cast<std::size_t>(kCols), 13u);
        std::vector<f64> lo(xv.size(), -0.4), hi(xv.size(), 0.35);
        DeviceBuffer<f64> lob(lo.data(), lo.size()), hib(hi.data(), hi.size());
        DeviceBuffer<f64> a(xv.data(), xv.size()), b(xv.data(), xv.size());
        ref.project_box(a, lob, hib);
        cand.project_box(b, lob, hib);
        for (std::size_t i = 0; i < a.size(); ++i) CHECK_NEAR(b[i], a[i], tol);
    }

    // ---- dot ----
    {
        const auto av = make_vector(64, 14u);
        const auto bv = make_vector(64, 15u);
        DeviceBuffer<f64> a(av.data(), av.size()), b(bv.data(), bv.size());
        CHECK_NEAR(cand.dot(a, b), ref.dot(a, b), tol);
    }

    // ---- spmv_batched ----
    {
        constexpr std::size_t kBatch = 7;
        DeviceBuffer<f64> X(kBatch * static_cast<std::size_t>(kCols));
        const auto flat = make_vector(X.size(), 16u);
        for (std::size_t i = 0; i < X.size(); ++i) X[i] = flat[i];

        DeviceBuffer<f64> Yr(kBatch * static_cast<std::size_t>(kRows));
        DeviceBuffer<f64> Yc(kBatch * static_cast<std::size_t>(kRows));
        auto Xv  = BatchView<f64>::from_buffer(X, kBatch);
        auto Yrv = BatchView<f64>::from_buffer(Yr, kBatch);
        auto Ycv = BatchView<f64>::from_buffer(Yc, kBatch);

        ref.spmv_batched(A.pattern, vals, Xv, Yrv);
        cand.spmv_batched(A.pattern, vals, Xv, Ycv);
        for (std::size_t i = 0; i < Yr.size(); ++i) CHECK_NEAR(Yc[i], Yr[i], tol);

        // Batched must agree with kernel-by-kernel single spmv.
        for (std::size_t b = 0; b < kBatch; ++b) {
            DeviceBuffer<f64> xi(Xv.item(b), static_cast<std::size_t>(kCols));
            DeviceBuffer<f64> yi;
            ref.spmv(A.pattern, vals, xi, yi);
            for (std::size_t r = 0; r < yi.size(); ++r)
                CHECK_NEAR(Ycv.item(b)[r], yi[r], tol);
        }
    }

    // ---- project_box_batched ----
    {
        constexpr std::size_t kBatch = 5;
        const std::size_t len = 32;
        DeviceBuffer<f64> A1(kBatch * len), A2(kBatch * len);
        DeviceBuffer<f64> LO(kBatch * len), HI(kBatch * len);
        const auto flat = make_vector(kBatch * len, 17u);
        for (std::size_t i = 0; i < flat.size(); ++i) {
            A1[i] = A2[i] = flat[i];
            // Per-item boxes differ, which is the point of the batched form.
            LO[i] = -0.5 + 0.05 * static_cast<f64>(i / len);
            HI[i] =  0.5 - 0.05 * static_cast<f64>(i / len);
        }
        auto V1 = BatchView<f64>::from_buffer(A1, kBatch);
        auto V2 = BatchView<f64>::from_buffer(A2, kBatch);
        auto LOv = BatchView<f64>::from_buffer(LO, kBatch);
        auto HIv = BatchView<f64>::from_buffer(HI, kBatch);
        ref.project_box_batched(V1, LOv, HIv);
        cand.project_box_batched(V2, LOv, HIv);
        for (std::size_t i = 0; i < A1.size(); ++i) CHECK_NEAR(A2[i], A1[i], tol);
    }

    // Transfer accounting must be populated, so a timing table can never be
    // reported without it.
    CHECK(cand.transfer_stats().calls > 0);
}

}  // namespace sor::test
