// Executable acceptance gate for the nine Must categories.
//
// This is intentionally a fast coverage test, not a performance benchmark and
// not a claim that the frontier item 30 of the product checklist is complete. It proves that every PS
// category has a working vertical slice in the current build and that required
// public-suite/industrial inputs and external-baseline harnesses are present.
#include "forge/certify/finalize.hpp"
#include "forge/engines/qp.hpp"
#include "forge/engines/simplex.hpp"
#include "forge/io/mps.hpp"
#include "forge/io/qps.hpp"
#include "forge/la/lu.hpp"
#include "forge/search/bab.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using forge::core::ProofLevel;
using forge::core::Status;

forge::model::LpProblem small_lp(bool integer) {
    // min x + 2y, x+y >= 1, 0 <= x,y <= 1.
    forge::model::LpProblem lp;
    lp.name = integer ? "must_milp" : "must_lp";
    lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {1.0, 2.0};
    lp.row_lo = {1.0};
    lp.row_hi = {forge::model::kInf};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {integer, integer};
    lp.validate();
    return lp;
}

void test_m1_clean_room_runtime_link_map() {
#if defined(__linux__)
    std::ifstream maps("/proc/self/maps");
    CHECK(static_cast<bool>(maps));
    std::string text((std::istreambuf_iterator<char>(maps)),
                     std::istreambuf_iterator<char>());
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const char* forbidden : {"libhighs", "libcbc", "libscip", "libglpk",
                                  "libortools", "libcuopt"})
        CHECK(text.find(forbidden) == std::string::npos);
#endif
}

void test_m2_lp_and_m6_certification() {
    const auto lp = small_lp(false);
    forge::engines::SimplexOptions opts;
    opts.method = forge::engines::SimplexMethod::Auto;
    forge::engines::SimplexDiagnostics diag;
    auto raw = forge::engines::solve_simplex(lp, opts, diag, nullptr);
    const auto result = forge::certify::finalize_result(
        std::move(raw), forge::engines::simplex_evidence(diag, opts));
    CHECK(result.status == Status::Optimal);
    CHECK(result.proof == ProofLevel::ProvedOptimalFP);
    CHECK_NEAR(result.objective, 1.0, 1e-8);
    CHECK(result.max_primal_violation <= opts.primal_feas_tol);
    CHECK(result.max_dual_violation <= opts.dual_feas_tol);
}

void test_m3_milp() {
    const auto mip = small_lp(true);
    forge::search::BabOptions opts;
    opts.max_nodes = 100;
    forge::search::BabDiagnostics diag;
    auto raw = forge::search::solve_milp(mip, opts, diag);
    const auto result = forge::certify::finalize_result(
        std::move(raw), forge::search::milp_evidence(diag, opts));
    CHECK(result.status == Status::Optimal);
    CHECK_NEAR(result.objective, 1.0, 1e-8);
    // The proof either came from the tree or, when an early incumbent
    // already matches the certified root LP bound, before any node was
    // popped -- from that root certificate.
    CHECK(diag.nodes >= 1 || std::isfinite(diag.root_certified_bound));
}

void test_m4_qp() {
    forge::engines::QpProblem qp;
    qp.linear.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    qp.linear.c = {0.0, 0.0};
    qp.linear.row_lo = {1.0};
    qp.linear.row_hi = {1.0};
    qp.linear.col_lo = {0.0, 0.0};
    qp.linear.col_hi = {1.0, 1.0};
    qp.q_diag = {1.0, 1.0};
    forge::engines::QpOptions opts;
    forge::engines::QpDiagnostics diag;
    auto raw = forge::engines::solve_qp_diag(qp, opts, diag);
    const auto result = forge::certify::finalize_result(
        std::move(raw), forge::engines::qp_evidence(diag, opts));
    CHECK(result.status == Status::Optimal);
    CHECK(result.proof == ProofLevel::ProvedKKT);
    CHECK_NEAR(result.x[0], 0.5, 1e-7);
    CHECK_NEAR(result.x[1], 0.5, 1e-7);
}

void test_m5_sparse_linear_algebra() {
    // B = [4 1; 2 3], stored by columns. Check B(B^-1 b)=b.
    forge::la::BasisFactor factor;
    const std::vector<forge::core::Offset> ptr{0, 2, 4};
    const std::vector<forge::core::Index> row{0, 1, 0, 1};
    const std::vector<double> val{4.0, 2.0, 1.0, 3.0};
    CHECK(factor.factorize(2, ptr, row, val, forge::la::LuOptions{}));
    std::vector<double> x{1.0, 2.0};
    factor.ftran(x);
    CHECK_NEAR(4.0 * x[0] + x[1], 1.0, 1e-12);
    CHECK_NEAR(2.0 * x[0] + 3.0 * x[1], 2.0, 1e-12);
}

void test_m7_cli_and_m8_public_harnesses() {
    const fs::path src(FORGE_SOURCE_DIR), bin(FORGE_BINARY_DIR);
    CHECK(fs::is_regular_file(bin / "forge_solve"));
    CHECK(fs::is_regular_file(bin / "forge_check"));
    CHECK(fs::is_regular_file(src / "scripts/fetch_benchmarks.py"));
    CHECK(fs::is_regular_file(src / "benchmarks/miplib2017/benchmark-v2.test"));
    // M8 requires a public harness that measures FORGE against independent
    // reference solvers. That used to be one script per baseline
    // (run_highs_baseline.py, run_cbc_baseline.py, ...); they are now a single
    // tool that drives HiGHS, CBC, SciPy and Gurobi as external processes and
    // prints the runs side by side.
    CHECK(fs::is_regular_file(src / "scripts/compare.py"));
}

void test_m9_industrial_vertical_slices() {
    const fs::path root(FORGE_SOURCE_DIR);
    forge::io::MpsReadReport blend_report, schedule_report;
    const auto blend = forge::io::read_mps_file_auto(
        (root / "examples/crude_blending/blend_s42.mps").string(), blend_report);
    const auto schedule = forge::io::read_mps_file_auto(
        (root / "examples/scheduling/schedule_s42.mps").string(), schedule_report);
    forge::io::QpsReadReport dispatch_report;
    const auto dispatch = forge::io::read_qps_file(
        (root / "examples/dispatch/dispatch_s42.qps").string(), dispatch_report);
    CHECK(blend.n_rows() > 0 && blend.n_cols() > 0);
    CHECK(schedule.n_integer() > 0);
    CHECK(dispatch.linear.n_cols() > 0 && !dispatch.q_diag.empty());
    CHECK(fs::is_regular_file(root / "examples/MANIFEST.json"));
}

}  // namespace

int main() {
    test_m1_clean_room_runtime_link_map();
    test_m2_lp_and_m6_certification();
    test_m3_milp();
    test_m4_qp();
    test_m5_sparse_linear_algebra();
    test_m7_cli_and_m8_public_harnesses();
    test_m9_industrial_vertical_slices();
    return forge::test::finish("test_ps_must");
}
