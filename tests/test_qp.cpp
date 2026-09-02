// Convex diagonal-QP active-set: unconstrained-with-bounds and equality dispatch.
#include "sor/certify/finalize.hpp"
#include "sor/engines/qp.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <vector>

using sor::core::ProofLevel;
using sor::core::Status;
using sor::engines::QpDiagnostics;
using sor::engines::QpOptions;
using sor::engines::QpProblem;

namespace {

// min 1/2 (x-1)^2 + 1/2 (y-2)^2  (= 1/2 x^2 - x + 1/2 y^2 - 2y + const)
// s.t. 0 <= x,y <= 10
// Optimum (1,2)
void test_box_qp() {
    QpProblem qp;
    qp.linear.A = sor::sparse::from_triplets(0, 2, {}, {}, {});
    qp.linear.c = {-1.0, -2.0};
    qp.linear.col_lo = {0.0, 0.0};
    qp.linear.col_hi = {10.0, 10.0};
    qp.linear.row_lo.clear();
    qp.linear.row_hi.clear();
    qp.q_diag = {1.0, 1.0};

    QpOptions opts;
    QpDiagnostics diag;
    auto raw = sor::engines::solve_qp_diag(qp, opts, diag);
    const auto ev = sor::engines::qp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal);
    CHECK(r.proof == ProofLevel::ProvedKKT);
    CHECK_NEAR(r.x[0], 1.0, 1e-6);
    CHECK_NEAR(r.x[1], 2.0, 1e-6);
}

// Economic dispatch toy:
// min 0.1 p1^2 + 5 p1 + 0.2 p2^2 + 3 p2
// s.t. p1 + p2 = 10,  0 <= p <= 8
void test_dispatch_qp() {
    QpProblem qp;
    std::vector<sor::core::Index> rows = {0, 0};
    std::vector<sor::core::Index> cols = {0, 1};
    std::vector<double> vals = {1.0, 1.0};
    qp.linear.A = sor::sparse::from_triplets(1, 2, rows, cols, vals);
    qp.linear.c = {5.0, 3.0};
    qp.linear.row_lo = {10.0};
    qp.linear.row_hi = {10.0};
    qp.linear.col_lo = {0.0, 0.0};
    qp.linear.col_hi = {8.0, 8.0};
    qp.q_diag = {0.2, 0.4};  // Q = diag(0.2, 0.4) for 1/2 x'Qx with coeffs matching
    // obj = 1/2 * 0.2 p1^2 + 5 p1 + 1/2 * 0.4 p2^2 + 3 p2
    //     = 0.1 p1^2 + 5 p1 + 0.2 p2^2 + 3 p2  ✓

    QpOptions opts;
    QpDiagnostics diag;
    auto raw = sor::engines::solve_qp_diag(qp, opts, diag);
    const auto ev = sor::engines::qp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.x[0] + r.x[1], 10.0, 1e-6);
    CHECK(r.x[0] >= -1e-8 && r.x[0] <= 8.0 + 1e-8);
    CHECK(r.x[1] >= -1e-8 && r.x[1] <= 8.0 + 1e-8);
    CHECK(diag.stationarity < 1e-6);
}

}  // namespace

int main() {
    test_box_qp();
    test_dispatch_qp();
    return 0;
}
