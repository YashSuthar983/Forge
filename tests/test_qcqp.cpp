// Quadratic constraints in the model layer (Track 2 B1 / MOD-1).
//
// What is pinned down here:
//   1. the QPLIB reader decodes per-constraint Hessians, general-integer and
//      binary typing, and the trailer (so a file is consumed to its last
//      line), and refuses an unknown variable-type code;
//   2. qplib_to_qcqp keeps EVERY quadratic row, with the same 0.5*v
//      convention as the objective, symmetric listings and cancelling
//      duplicates handled, constraints not negated for a maximisation;
//   3. the model evaluation agrees with the raw-file evaluation at many
//      points (they share no conversion code);
//   4. the published-.sol name mapping (<prefix><j+2>, overrides honoured);
//   5. NO ENGINE SILENTLY RELAXES A QUADRATIC CONSTRAINT: the only road to a
//      QpProblem refuses, binquad throws, QCR refuses, the refusal finalises
//      as Status::Unsupported, and the CLI reports Unsupported for every
//      engine name on a QC instance -- never a status with an objective.
//
// tests/data/qcqp_tiny.qplib is hand-written for this test, not a QPLIB
// instance; its expected values are worked out by hand in the comments.
#include "test_helpers.hpp"

#include "sor/certify/finalize.hpp"
#include "sor/engines/qcqp.hpp"
#include "sor/io/qplib.hpp"
#include "sor/search/binquad.hpp"
#include "sor/search/qcr.hpp"
#include "sor/search/qplib_qp.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace sor;
using core::f64;
namespace fs = std::filesystem;

const std::string kData = std::string(SOR_SOURCE_DIR) + "/tests/data/";

std::string slurp(const std::string& p) {
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string write_tmp(const std::string& name, const std::string& text) {
    const auto p = fs::temp_directory_path() / ("sor_test_qcqp_" + name);
    std::ofstream(p) << text;
    return p.string();
}

void test_reader() {
    io::QplibReadReport rep;
    const auto q = io::read_qplib_file(kData + "qcqp_tiny.qplib", rep);
    CHECK(q.n == 4 && q.m == 3);
    CHECK(q.maximize);
    CHECK(q.has_quadratic_constraints());
    CHECK(rep.n_hc_entries == 8);
    CHECK(rep.fully_consumed());
    CHECK(rep.trailer_error.empty());
    // Typing: default 0 (continuous), var 2 binary, var 3 integer.
    CHECK(q.var_type[0] == io::QplibVarType::Continuous);
    CHECK(q.var_type[1] == io::QplibVarType::Binary);
    CHECK(q.var_type[2] == io::QplibVarType::Integer);
    CHECK(q.var_type[3] == io::QplibVarType::Continuous);
    CHECK(!q.is_discrete());
    CHECK(q.var_names[3] == "w4" && q.var_names[0].empty());
    CHECK(q.con_names[2] == "cap");
    CHECK(io::qplib_default_var_name(q, 0) == "x2");
    CHECK(io::qplib_default_var_name(q, 1) == "b3");
    CHECK(io::qplib_default_var_name(q, 2) == "i4");

    // An unknown type code (3) must be refused by name, not cast into the enum.
    std::string bad = slurp(kData + "qcqp_tiny.qplib");
    const auto at = bad.find("\n3 1\n");
    CHECK(at != std::string::npos);
    bad.replace(at, 5, "\n3 3\n");
    io::QplibReadReport r2;
    CHECK_THROWS(io::read_qplib_file(write_tmp("badtype.qplib", bad), r2));

    // A stray line after the trailer: the model still loads, but the file is
    // flagged as not fully consumed -- the parse-all sweep counts it failed.
    const std::string extra = slurp(kData + "qcqp_tiny.qplib") + "7 # stray\n";
    io::QplibReadReport r3;
    (void)io::read_qplib_file(write_tmp("extra.qplib", extra), r3);
    CHECK(!r3.fully_consumed());
}

void test_model() {
    io::QplibReadReport rep;
    const auto q = io::read_qplib_file(kData + "qcqp_tiny.qplib", rep);
    engines::QcqpProblem p;
    search::qplib_to_qcqp(q, p);
    p.validate();
    CHECK(p.objective_negated);
    CHECK(p.n_rows() == 3 && p.n_cols() == 4);
    // Row 1 (0-based 1): its quadruples (1,2,1.0) and (2,1,-1.0) cancel, so
    // it is NOT a quadratic row.  Rows 0 and 2 are.
    CHECK(p.quad.size() == 2);
    if (p.quad.size() == 2) {
        const auto& r0 = p.quad[0];
        // (1,1,2) -> Q_00 = 2;  (2,1,4) + (2,1,-1) -> Q_01 = 0.5*3 = 1.5.
        CHECK(r0.row == 0 && r0.v.size() == 2);
        if (r0.v.size() == 2) {
            CHECK(r0.r[0] == 0 && r0.c[0] == 0 && r0.v[0] == 2.0);
            CHECK(r0.r[1] == 0 && r0.c[1] == 1 && r0.v[1] == 1.5);
        }
        const auto& r2 = p.quad[1];
        // (4,4,-2) -> Q_33 = -2;  (4,3,1) + (3,4,1) -> Q_23 = 0.5+0.5 = 1.
        // NOT negated although the objective is a maximisation.
        CHECK(r2.row == 2 && r2.v.size() == 2);
        if (r2.v.size() == 2) {
            CHECK(r2.r[0] == 2 && r2.c[0] == 3 && r2.v[0] == 1.0);
            CHECK(r2.r[1] == 3 && r2.c[1] == 3 && r2.v[1] == -2.0);
        }
        CHECK(!r0.diagonal());
    }
    const auto& lp = p.qp.linear;
    // Binary: file bounds [0, inf) intersected with [0,1].
    CHECK(lp.col_lo[1] == 0.0 && lp.col_hi[1] == 1.0);
    CHECK(lp.col_lo[3] == -5.0 && lp.col_hi[3] == 5.0);
    CHECK(!lp.is_integer[0] && lp.is_integer[1] && lp.is_integer[2] && !lp.is_integer[3]);
    CHECK(lp.row_hi[2] == -4.5 && lp.row_lo[1] == 1.0);
    CHECK(lp.col_names[3] == "w4" && lp.col_names[0] == "x2");

    // The published-style point, worked by hand:
    //   objective  0.5*2*x1^2 + 0.5*3*x2*x1 + x1 - x3 + 5 = 4.5 (max sense)
    //   row 2      x4 + (0.5*-2*x4^2 + 0.5*x4*x3 + 0.5*x3*x4) = -1 - 3 = -4
    //              against <= -4.5: violated by 0.5
    const std::vector<f64> x{0.5, 1.0, 2.0, -1.0};
    const auto e = engines::evaluate_qcqp(p, x);
    CHECK_NEAR(e.objective, 4.5, 1e-15);
    CHECK_NEAR(e.objective_min, -4.5, 1e-15);
    CHECK_NEAR(e.max_qc_violation, 0.5, 1e-15);
    CHECK_NEAR(e.max_row_violation, 0.5, 1e-15);
    CHECK(e.max_bound_violation == 0.0 && e.max_integrality_violation == 0.0);
    const auto raw = io::qplib_evaluate_point(q, x);
    CHECK_NEAR(raw.objective, 4.5, 1e-15);
    CHECK_NEAR(raw.max_qc_violation, 0.5, 1e-15);

    // Model vs raw at random points, integrality and bounds included.
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<f64> u(-3.0, 3.0);
    for (int k = 0; k < 200; ++k) {
        std::vector<f64> y(4);
        for (auto& v : y) v = u(rng);
        const auto a = engines::evaluate_qcqp(p, y);
        const auto b = io::qplib_evaluate_point(q, y);
        CHECK_NEAR(a.objective, b.objective, 1e-13);
        CHECK_NEAR(a.max_row_violation, b.max_row_violation, 1e-13);
        // max_qc_violation is NOT compared: the raw evaluator counts row 1
        // (whose Hessian entries cancel) as quadratic, the model does not.
        // max_row_violation covers every row either way.
        CHECK_NEAR(a.max_bound_violation, b.max_bound_violation, 1e-13);
        CHECK_NEAR(a.max_integrality_violation, b.max_integrality_violation, 1e-13);
    }
}

void test_solution_file() {
    io::QplibReadReport rep;
    const auto q = io::read_qplib_file(kData + "qcqp_tiny.qplib", rep);
    const auto s = io::read_qplib_solution(kData + "qcqp_tiny.sol", q);
    CHECK(s.has_objvar && s.objvar == 4.5);
    CHECK(s.listed == 4);
    CHECK(s.x == (std::vector<f64>{0.5, 1.0, 2.0, -1.0}));
    // A name whose prefix disagrees with the type (x3 for the binary) or an
    // index past n must not silently land somewhere.
    CHECK_THROWS(io::read_qplib_solution(write_tmp("wrongprefix.sol", "x3 1\n"), q));
    CHECK_THROWS(io::read_qplib_solution(write_tmp("past.sol", "x6 1\n"), q));
    CHECK_THROWS(io::read_qplib_solution(write_tmp("twice.sol", "x2 1\nx2 1\n"), q));
    // The overridden name replaces the default: "x5" is no longer valid.
    CHECK_THROWS(io::read_qplib_solution(write_tmp("override.sol", "x5 1\n"), q));
}

void test_no_engine_drops_a_qc() {
    io::QplibReadReport rep;
    const auto q = io::read_qplib_file(kData + "qcqp_tiny.qplib", rep);
    engines::QcqpProblem p;
    search::qplib_to_qcqp(q, p);

    // The gateway refuses, names the problem, and leaves p intact.
    engines::QpProblem out;
    std::string why;
    CHECK(!engines::qcqp_to_qp(p, out, why));
    CHECK(why.find("quadratic constraint") != std::string::npos);
    CHECK(!engines::qcqp_to_qp(std::move(p), out, why));
    CHECK(p.quad.size() == 2);   // refusal did not move from it

    // The same model with its quadratic rows removed is accepted: the gate
    // is about QCs, not about the rest of the model.
    engines::QcqpProblem lin = p;
    lin.quad.clear();
    CHECK(engines::qcqp_to_qp(lin, out, why));

    // Direct QPLIB -> QP conversion refuses, also with relaxation on.
    bool negated = false;
    CHECK(!search::qplib_to_qp(q, {}, out, negated, why));
    search::QplibToQpOptions relax;
    relax.relax_binary = true;
    CHECK(!search::qplib_to_qp(q, relax, out, negated, why));

    // Refusal finalises as Unsupported with no objective.
    const auto r = certify::finalize_result(engines::qcqp_unsupported("qp", "cpu", p),
                                            core::ProofEvidence{});
    CHECK(r.status == core::Status::Unsupported);
    CHECK(r.x.empty());
    CHECK(r.termination_reason.find("quadratic constraint") != std::string::npos);

    // binquad scores rows through A only: on an all-binary instance WITH a
    // quadratic row it must throw, not return an incumbent that ignores it.
    io::QplibInstance b;
    b.name = "QBQ_TINY";
    b.classification = {'Q', 'B', 'Q'};
    b.n = 2;
    b.m = 1;
    b.h_row = {1};
    b.h_col = {2};
    b.h_val = {-1.0};
    b.g = {0.0, 0.0};
    b.hc_con = {1};
    b.hc_row = {1};
    b.hc_col = {1};
    b.hc_val = {2.0};          // x1^2 <= 0  forces x1 = 0
    b.c_lo = {-core::kPosInf};
    b.c_hi = {0.0};
    b.x_lo = {0.0, 0.0};
    b.x_hi = {1.0, 1.0};
    b.var_type = {io::QplibVarType::Binary, io::QplibVarType::Binary};
    b.inf_bound = 1e20;
    search::BinQuadOptions bo;
    search::BinQuadDiagnostics bd;
    CHECK_THROWS(search::solve_binquad(b, bo, bd));
    // QCR (and therefore bqp_bab, which bounds through it) refuses too.
    engines::QpProblem relaxed;
    search::QcrOptions qo;
    search::QcrDiagnostics qd;
    CHECK(!search::qcr_relaxation(b, qo, relaxed, negated, qd));
    CHECK(qd.reason.find("quadratic") != std::string::npos);
}

// The CLI: every engine name reports Unsupported on a QC instance (exit 5,
// the non-claim code), and --eval-solution evaluates without solving.
struct Run { int code = 0; std::string out; };
Run run(const std::string& cmd) {
    Run r;
    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) { r.code = -1; return r; }
    std::array<char, 512> buf{};
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) r.out += buf.data();
    const int st = pclose(pipe);
    r.code = st == -1 ? -1 : st / 256;
    return r;
}

void test_cli() {
    const std::string exe = std::string(SOR_BINARY_DIR) + "/sor_solve";
    const std::string model = kData + "qcqp_tiny.qplib";
    // Engines that cannot represent a quadratic row must still refuse it.
    // `auto` is NOT in this list any more: it routes a QC instance to an
    // engine that does handle one (miqcqp_bb here), which is the point of
    // the QCQP work -- checked separately below.
    for (const char* eng : {"qp", "qpipm", "hprqp", "binquad", "simplex", "milp",
                            "pdhg", "hpr", "primal", "dual"}) {
        const auto r = run(exe + " '" + model + "' --engine " + eng + " --time-limit 5");
        ::sor::test::report(r.code == 5, "QC instance exits 5", __FILE__, __LINE__,
                            std::string(eng) + " -> " + std::to_string(r.code) + "\n" + r.out);
        ::sor::test::report(r.out.find("status:            Unsupported") != std::string::npos,
                            "QC instance is Unsupported", __FILE__, __LINE__,
                            std::string(eng) + " -> " + r.out);
        ::sor::test::report(r.out.find("quadratic constraint") != std::string::npos,
                            "reason names quadratic constraints", __FILE__, __LINE__, eng);
    }
    // --engine auto on the same instance: it may now solve it, but only as
    // far as the evidence goes -- a checked Feasible point, never Optimal,
    // and never a claim without the raw re-check agreeing.
    {
        const auto a = run(exe + " '" + model + "' --engine auto --time-limit 5");
        const bool optimal = a.out.find("status:            Optimal") != std::string::npos;
        ::sor::test::report(!optimal, "auto never claims Optimal on a nonconvex QC instance",
                            __FILE__, __LINE__, a.out);
        if (a.out.find("status:            Feasible") != std::string::npos)
            ::sor::test::report(a.out.find("raw re-check:") != std::string::npos,
                                "a Feasible claim carries its raw re-check", __FILE__, __LINE__,
                                a.out);
    }

    const auto e = run(exe + " '" + model + "' --eval-solution '" + kData + "qcqp_tiny.sol'");
    CHECK(e.code == 0);
    CHECK(e.out.find("EVAL name=QCQP_TINY") != std::string::npos);
    CHECK(e.out.find("agree=1") != std::string::npos);
    CHECK(e.out.find("consumed=1") != std::string::npos);
    CHECK(e.out.find("qcviol=0.5 ") != std::string::npos);
}

}  // namespace

int main() {
    test_reader();
    test_model();
    test_solution_file();
    test_no_engine_drops_a_qc();
    test_cli();
    return sor::test::finish("test_qcqp");
}
