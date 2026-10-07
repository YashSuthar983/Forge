#include "sor/certify/finalize.hpp"
#include "sor/engines/qp.hpp"
#include "sor/io/gzip.hpp"
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

void test_qps_off_diagonal_is_preserved_and_solved() {
    const std::string path = "/tmp/sor_test_offdiag.qps";
    std::ofstream out(path);
    out << "NAME OFFDIAG\n"
           "ROWS\n"
           " N OBJ\n"
           " G LIMIT\n"
           "COLUMNS\n"
           " X OBJ -3 LIMIT 1\n"
           " Y OBJ -3 LIMIT 1\n"
           "RHS\n"
           " RHS1 LIMIT 3\n"
           "BOUNDS\n"
           " LO BND X 0\n"
           " UP BND X 5\n"
           " LO BND Y 0\n"
           " UP BND Y 5\n"
           "QUADOBJ\n"
           " X X 2\n"
           " Y X 1\n"
           " Y Y 2\n"
           "ENDATA\n";
    out.close();

    sor::io::QpsReadReport rep;
    auto loaded = sor::io::read_qps_file(path, rep);
    CHECK(rep.has_off_diagonal);
    CHECK(loaded.q_matrix.n_rows() == 2);
    CHECK(loaded.q_matrix.nnz() == 4);

    sor::engines::QpProblem qp;
    qp.linear = std::move(loaded.linear);
    qp.q_diag = std::move(loaded.q_diag);
    qp.q_matrix = std::move(loaded.q_matrix);
    sor::engines::QpOptions opts;
    opts.feas_tol = opts.stationarity_tol = opts.gap_tol = 1e-7;
    sor::engines::QpDiagnostics diag;
    auto raw = sor::engines::solve_qp(qp, opts, diag);
    const auto r = sor::certify::finalize_result(
        std::move(raw), sor::engines::qp_evidence(diag, opts));
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.x[0], 1.5, 2e-5);
    CHECK_NEAR(r.x[1], 1.5, 2e-5);
}

}  // namespace

// The quadratic sections are read in the MPS reader's single pass. The old
// second scan (a plain ifstream) found nothing in a gzip file and the QP was
// solved as an LP; it mirrored QMATRIX's full matrix, doubling every
// off-diagonal; and it read "2x" as 0, skipped short lines and dropped
// entries for unknown columns with only a warning.
void test_qps_quadratic_sections_read_faithfully() {
    const std::string head =
        "NAME          QSEC\nROWS\n N  OBJ\n L  R1\nCOLUMNS\n"
        "    X         OBJ       1.0        R1        1.0\n"
        "    Y         OBJ       1.0        R1        1.0\n"
        "RHS\n    RHS       R1        4.0\n";
    const std::string path = "/tmp/sor_test_qsec.qps";
    const auto read = [&](const std::string& body) {
        { std::ofstream f(path, std::ios::binary); f << body; }
        sor::io::QpsReadReport rep;
        return sor::io::read_qps_file(path, rep);
    };
    const auto off = [](const sor::io::QpsProblem& q) {
        const auto& rp = q.q_matrix.pattern.row_ptr();
        const auto& ci = q.q_matrix.pattern.col_idx();
        for (auto k = rp[0]; k < rp[1]; ++k)
            if (ci[static_cast<std::size_t>(k)] == 1) return q.q_matrix.vals[static_cast<std::size_t>(k)];
        return 0.0;
    };
    const std::string triangle = head + "QUADOBJ\n    X  X  2\n    X  Y  0.5\n    Y  Y  4\nENDATA\n";
    const std::string full = head + "QMATRIX\n    X  X  2\n    X  Y  0.5\n    Y  X  0.5\n    Y  Y  4\nENDATA\n";
    const auto t = read(triangle);
    const auto m = read(full);
    CHECK(t.q_diag[0] == 2.0 && t.q_diag[1] == 4.0 && off(t) == 0.5);
    CHECK(m.q_diag[0] == 2.0 && m.q_diag[1] == 4.0 && off(m) == 0.5);  // not 1.0

    // gzip: a stored-block member holding the same text.
    {
        std::string gz;
        gz += '\x1f'; gz += '\x8b'; gz += '\x08'; gz += '\x00';
        gz.append(4, '\x00'); gz += '\x00'; gz += '\x03'; gz += '\x01';
        const auto n = static_cast<unsigned>(triangle.size());
        gz += static_cast<char>(n & 0xff); gz += static_cast<char>((n >> 8) & 0xff);
        gz += static_cast<char>(~n & 0xff); gz += static_cast<char>((~n >> 8) & 0xff);
        gz += triangle;
        const auto crc = sor::io::crc32(triangle.data(), triangle.size());
        for (int i = 0; i < 4; ++i) gz += static_cast<char>((crc >> (8 * i)) & 0xff);
        for (int i = 0; i < 4; ++i) gz += static_cast<char>((n >> (8 * i)) & 0xff);
        const auto z = read(gz);
        CHECK(z.q_diag[0] == 2.0 && z.q_diag[1] == 4.0 && off(z) == 0.5);
    }
    // Repeated entries are summed exactly.
    const auto twice = read(head + "QUADOBJ\n    X  X  1.5\n    X  X  0.5\nENDATA\n");
    CHECK(twice.q_diag[0] == 2.0);
    // Refused: both triangles under QUADOBJ, an asymmetric QMATRIX, both
    // sections, a bad number, a short line, an unknown column.
    for (const std::string& bad : {
             head + "QUADOBJ\n    X  Y  0.5\n    Y  X  0.5\nENDATA\n",
             head + "QMATRIX\n    X  Y  0.5\n    Y  X  0.25\nENDATA\n",
             head + "QMATRIX\n    X  Y  0.5\nENDATA\n",
             head + "QUADOBJ\n    X  X  2\nQMATRIX\n    Y  Y  4\nENDATA\n",
             head + "QUADOBJ\n    X  X  2x\nENDATA\n",
             head + "QUADOBJ\n    X  2\nENDATA\n",
             head + "QUADOBJ\n    X  Z  2\nENDATA\n"}) {
        bool threw = false;
        try { (void)read(bad); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }
    std::remove(path.c_str());
}

int main() {
    test_qps_quadratic_sections_read_faithfully();
    test_qps_roundtrip_dispatch();
    test_qps_off_diagonal_is_preserved_and_solved();
    return sor::test::finish("test_qps");
}
