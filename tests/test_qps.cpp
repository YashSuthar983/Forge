#include "sor/certify/finalize.hpp"
#include "sor/engines/qp.hpp"
#include "sor/io/qps.hpp"
#include "sor/io/write_mps.hpp"

#include "test_helpers.hpp"

#include <cstdio>
#include <fstream>
#include <string>

using sor::core::ProofLevel;
using sor::core::Status;

namespace {

void test_qps_roundtrip_dispatch() {
    sor::model::LpProblem lp;
    lp.name = "TQ";
    lp.c = {5.0, 3.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {8.0, 8.0};
    lp.row_lo = {10.0};
    lp.row_hi = {10.0};
    lp.col_names = {"P1", "P2"};
    lp.row_names = {"DEMAND"};
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    const std::vector<double> q = {0.2, 0.4};

    const std::string path = "/tmp/sor_test_dispatch.qps";
    sor::io::write_qps_file(path, lp, q);

    sor::io::QpsReadReport rep;
    auto loaded = sor::io::read_qps_file(path, rep);
    CHECK(loaded.linear.n_cols() == 2);
    CHECK(loaded.q_diag.size() == 2);
    CHECK_NEAR(loaded.q_diag[0], 0.2, 1e-12);
    CHECK_NEAR(loaded.q_diag[1], 0.4, 1e-12);
    CHECK(!rep.has_off_diagonal);

    sor::engines::QpProblem qp;
    qp.linear = loaded.linear;
    qp.q_diag = loaded.q_diag;
    sor::engines::QpOptions opts;
    sor::engines::QpDiagnostics diag;
    auto raw = sor::engines::solve_qp_diag(qp, opts, diag);
    const auto ev = sor::engines::qp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal);
    CHECK(r.proof == ProofLevel::ProvedKKT);
    CHECK_NEAR(r.x[0] + r.x[1], 10.0, 1e-6);
}

}  // namespace

int main() {
    test_qps_roundtrip_dispatch();
    return 0;
}
