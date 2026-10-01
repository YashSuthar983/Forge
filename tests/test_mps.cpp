#include "sor/io/gzip.hpp"
#include "sor/io/mps.hpp"
#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <sstream>

using namespace sor;
using sor::model::kInf;

namespace {

// Dense lookup so the test does not depend on CSR ordering.
double coeff(const model::LpProblem& p, core::Index r, core::Index c) {
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    for (auto k = rp[static_cast<std::size_t>(r)];
         k < rp[static_cast<std::size_t>(r) + 1]; ++k) {
        if (ci[static_cast<std::size_t>(k)] == c)
            return p.A.vals[static_cast<std::size_t>(k)];
    }
    return 0.0;
}

int index_of(const std::vector<std::string>& v, const std::string& n) {
    for (std::size_t i = 0; i < v.size(); ++i) if (v[i] == n) return static_cast<int>(i);
    return -1;
}

}  // namespace

int main() {
    // ---- basic LP ----
    {
        std::istringstream in(sor::test::kTestLpMps);
        io::MpsReadReport rep;
        const auto p = io::read_mps(in, rep);

        CHECK(p.name == "TESTLP");
        CHECK(p.n_rows() == 2);
        CHECK(p.n_cols() == 2);
        CHECK(rep.n_integer == 0);
        CHECK(!p.maximize);

        const int x1 = index_of(p.col_names, "X1");
        const int x2 = index_of(p.col_names, "X2");
        CHECK(x1 >= 0 && x2 >= 0);

        CHECK_NEAR(p.c[static_cast<std::size_t>(x1)], -1.0, 1e-15);
        CHECK_NEAR(p.c[static_cast<std::size_t>(x2)], -2.0, 1e-15);

        const int r1 = index_of(p.row_names, "R1");
        const int r2 = index_of(p.row_names, "R2");
        CHECK(r1 >= 0 && r2 >= 0);
        CHECK_NEAR(coeff(p, r1, x1), 1.0, 1e-15);
        CHECK_NEAR(coeff(p, r1, x2), 1.0, 1e-15);
        CHECK_NEAR(coeff(p, r2, x1), 1.0, 1e-15);
        CHECK_NEAR(coeff(p, r2, x2), 3.0, 1e-15);

        // L rows: (-inf, rhs]
        CHECK(std::isinf(p.row_lo[static_cast<std::size_t>(r1)]));
        CHECK(p.row_lo[static_cast<std::size_t>(r1)] < 0);
        CHECK_NEAR(p.row_hi[static_cast<std::size_t>(r1)], 4.0, 1e-15);
        CHECK_NEAR(p.row_hi[static_cast<std::size_t>(r2)], 6.0, 1e-15);

        // UP bounds with default lower 0.
        CHECK_NEAR(p.col_lo[static_cast<std::size_t>(x1)], 0.0, 1e-15);
        CHECK_NEAR(p.col_hi[static_cast<std::size_t>(x1)], 3.0, 1e-15);

        // Sanity: the known optimum is feasible and hits the objective.
        std::vector<double> xopt(2, 0.0);
        xopt[static_cast<std::size_t>(x1)] = sor::test::kTestLpX1;
        xopt[static_cast<std::size_t>(x2)] = sor::test::kTestLpX2;
        CHECK_NEAR(p.max_row_violation(xopt), 0.0, 1e-15);
        CHECK_NEAR(p.max_bound_violation(xopt), 0.0, 1e-15);
        CHECK_NEAR(p.objective(xopt), sor::test::kTestLpOptimum, 1e-15);
    }

    // ---- RANGES, bound types, MARKER, extra free row, objective constant ----
    {
        std::istringstream in(sor::test::kFeaturesMps);
        io::MpsReadReport rep;
        const auto p = io::read_mps(in, rep);

        CHECK(p.n_rows() == 3);      // EQ1, GE1, LE1 -- FREEROW is not a constraint
        CHECK(p.n_cols() == 3);
        CHECK(rep.had_ranges);
        CHECK(rep.n_integer == 1);   // only B is inside the MARKER block

        const int a = index_of(p.col_names, "A");
        const int b = index_of(p.col_names, "B");
        const int c = index_of(p.col_names, "C");

        // Coefficients on the FREE second N row must be discarded.
        CHECK_NEAR(p.c[static_cast<std::size_t>(a)], 1.0, 1e-15);
        CHECK_NEAR(p.c[static_cast<std::size_t>(b)], 2.0, 1e-15);
        CHECK_NEAR(p.c[static_cast<std::size_t>(c)], -1.0, 1e-15);

        // RHS on the objective row is the negative of the constant.
        CHECK_NEAR(p.obj_offset, 7.0, 1e-15);

        const int eq = index_of(p.row_names, "EQ1");
        const int ge = index_of(p.row_names, "GE1");
        const int le = index_of(p.row_names, "LE1");

        // E row without a range: lo == hi == rhs
        CHECK_NEAR(p.row_lo[static_cast<std::size_t>(eq)], 5.0, 1e-15);
        CHECK_NEAR(p.row_hi[static_cast<std::size_t>(eq)], 5.0, 1e-15);
        // G row with range R: [rhs, rhs + |R|]
        CHECK_NEAR(p.row_lo[static_cast<std::size_t>(ge)], 1.0, 1e-15);
        CHECK_NEAR(p.row_hi[static_cast<std::size_t>(ge)], 4.0, 1e-15);
        // L row with range R: [rhs - |R|, rhs]
        CHECK_NEAR(p.row_lo[static_cast<std::size_t>(le)], 6.0, 1e-15);
        CHECK_NEAR(p.row_hi[static_cast<std::size_t>(le)], 10.0, 1e-15);

        // MI: lower -inf, upper stays at the +inf default
        CHECK(std::isinf(p.col_lo[static_cast<std::size_t>(a)]) &&
              p.col_lo[static_cast<std::size_t>(a)] < 0);
        // FX: both bounds pinned
        CHECK_NEAR(p.col_lo[static_cast<std::size_t>(b)], 2.0, 1e-15);
        CHECK_NEAR(p.col_hi[static_cast<std::size_t>(b)], 2.0, 1e-15);
        // FR: free both ways
        CHECK(std::isinf(p.col_lo[static_cast<std::size_t>(c)]));
        CHECK(std::isinf(p.col_hi[static_cast<std::size_t>(c)]));
        CHECK(p.is_integer[static_cast<std::size_t>(b)]);
        CHECK(!p.is_integer[static_cast<std::size_t>(a)]);
    }

    // ---- OBJSENSE MAX ----
    {
        const std::string mps =
            "NAME          M\n"
            "OBJSENSE\n"
            "    MAX\n"
            "ROWS\n"
            " N  OBJ\n"
            " L  R1\n"
            "COLUMNS\n"
            "    X         OBJ       1.0        R1        1.0\n"
            "RHS\n"
            "    RHS       R1        2.0\n"
            "ENDATA\n";
        std::istringstream in(mps);
        io::MpsReadReport rep;
        const auto p = io::read_mps(in, rep);
        CHECK(p.maximize);
        CHECK(rep.had_objsense_max);
    }

    // Integer markers default to binary only when no BOUNDS record is present.
    {
        const std::string mps =
            "NAME MARKERBOUNDS\n"
            "ROWS\n"
            " N OBJ\n"
            " L R1\n"
            "COLUMNS\n"
            " MARK1 'MARKER' 'INTORG'\n"
            " BINARY OBJ 1 R1 1\n"
            " GENERAL OBJ 1 R1 1\n"
            " MARK2 'MARKER' 'INTEND'\n"
            "RHS\n"
            " RHS1 R1 2\n"
            "BOUNDS\n"
            " LO BND GENERAL 0\n"
            "ENDATA\n";
        std::istringstream in(mps);
        io::MpsReadReport rep;
        const auto p = io::read_mps(in, rep);
        const int binary = index_of(p.col_names, "BINARY");
        const int general = index_of(p.col_names, "GENERAL");
        CHECK(binary >= 0 && general >= 0);
        CHECK_NEAR(p.col_hi[static_cast<std::size_t>(binary)], 1.0, 1e-15);
        CHECK(std::isinf(p.col_hi[static_cast<std::size_t>(general)]));
        CHECK(p.is_integer[static_cast<std::size_t>(binary)]);
        CHECK(p.is_integer[static_cast<std::size_t>(general)]);
    }

    // ---- malformed input must throw with a line number, not be guessed at ----
    {
        const std::string bad =
            "ROWS\n N  OBJ\n L  R1\nCOLUMNS\n    X   NOSUCHROW   1.0\nENDATA\n";
        std::istringstream in(bad);
        io::MpsReadReport rep;
        CHECK_THROWS(io::read_mps(in, rep));
    }
    {
        const std::string bad =
            "ROWS\n Z  OBJ\nENDATA\n";   // unknown row kind
        std::istringstream in(bad);
        io::MpsReadReport rep;
        CHECK_THROWS(io::read_mps(in, rep));
    }

    // ---- number tokens read exactly as strtod reads them ----
    // A leading '+', exponent forms and hex are accepted; a doubled sign,
    // trailing text and values strtod reports as out of range (overflow, and
    // underflow into the subnormals) are rejected. The from_chars fast path
    // must not widen this.
    {
        const auto coefficient = [](const std::string& token, double& out) {
            const std::string mps =
                "NAME T\nROWS\n N  OBJ\n L  R1\nCOLUMNS\n    X  OBJ  1  R1  " + token +
                "\nRHS\n    RHS  R1  1\nENDATA\n";
            std::istringstream in(mps);
            io::MpsReadReport rep;
            try {
                const auto p = io::read_mps(in, rep);
                out = p.A.vals.empty() ? 0.0 : p.A.vals[0];
                return true;
            } catch (const std::exception&) {
                return false;
            }
        };
        double v = 0.0;
        CHECK(coefficient("+5", v) && v == 5.0);
        CHECK(coefficient("-2.5e+3", v) && v == -2500.0);
        CHECK(coefficient(".5", v) && v == 0.5);
        CHECK(coefficient("0.1", v) && v == 0.1);
        CHECK(coefficient("0x10", v) && v == 16.0);
        CHECK(coefficient("1e-300", v) && v == 1e-300);
        CHECK(!coefficient("+-5", v));
        CHECK(!coefficient("--5", v));
        CHECK(!coefficient("5x", v));
        CHECK(!coefficient("1e400", v));
        CHECK(!coefficient("1e-320", v));
        CHECK(!coefficient("inf", v));
        CHECK(!coefficient("nan", v));
    }

    // ---- integrality relaxation ----
    //
    // A MIPLIB root LP relaxation must be formed from the ORIGINAL file, by
    // dropping integrality and nothing else. The report still states what the
    // file carried, so a manifest can record that the model really was integer.
    {
        const std::string mip =
            "NAME          RELAXME\n"
            "ROWS\n"
            " N  OBJ\n"
            " L  R1\n"
            "COLUMNS\n"
            "    MARKER                 'MARKER'                 'INTORG'\n"
            "    X         OBJ       1.0        R1        1.0\n"
            "    MARKER                 'MARKER'                 'INTEND'\n"
            "    Y         OBJ       2.0        R1        1.0\n"
            "RHS\n"
            "    RHS       R1        4.0\n"
            "BOUNDS\n"
            " UP BND       X         3.0\n"
            "ENDATA\n";
        io::MpsReadReport rep;
        std::istringstream in(mip);
        const auto p = io::read_mps(in, rep);
        CHECK(rep.n_integer == 1);
        CHECK(!rep.relaxed_integrality);
        CHECK(p.n_integer() == 1);

        io::MpsReadOptions opt;
        opt.relax_integrality = true;
        io::MpsReadReport rep2;
        std::istringstream in2(mip);
        const auto q = io::read_mps(in2, rep2, opt);
        CHECK(rep2.relaxed_integrality);
        CHECK(rep2.n_integer == 1);    // what the FILE carried
        CHECK(q.n_integer() == 0);     // what the model now is
        // Nothing else may change: same shape, bounds and coefficients.
        CHECK(q.n_rows() == p.n_rows());
        CHECK(q.n_cols() == p.n_cols());
        CHECK(q.nnz() == p.nnz());
        CHECK_NEAR(q.col_hi[0], 3.0, 1e-15);

        // Relaxing a model with no integer column is a no-op, and must not
        // claim otherwise.
        const std::string pure =
            "ROWS\n N  OBJ\n L  R1\n"
            "COLUMNS\n    X         OBJ       1.0        R1        1.0\n"
            "RHS\n    RHS       R1        2.0\nENDATA\n";
        io::MpsReadReport rep3;
        std::istringstream in3(pure);
        const auto r = io::read_mps(in3, rep3, opt);
        CHECK(!rep3.relaxed_integrality);
        CHECK(r.n_integer() == 0);
    }

    // ---- small-matrix-value threshold ----
    //
    // Both solvers in a comparison must receive the same matrix. The oracle
    // discards coefficients at or below its own threshold, so SOR needs the
    // identical policy -- and needs to report exactly what it dropped.
    {
        const std::string tiny =
            "NAME          TINY\n"
            "ROWS\n"
            " N  OBJ\n"
            " L  R1\n"
            " L  R2\n"
            "COLUMNS\n"
            "    X         OBJ       1.0        R1        1.0\n"
            "    Y         R1        1.0e-11    R2        2.0\n"
            "RHS\n"
            "    RHS       R1        4.0        R2        5.0\n"
            "ENDATA\n";
        io::MpsReadReport keep;
        std::istringstream in(tiny);
        const auto p = io::read_mps(in, keep);
        CHECK(p.nnz() == 3);
        CHECK(keep.small_values_dropped == 0);

        io::MpsReadOptions opt;
        opt.small_matrix_value = 1e-9;
        io::MpsReadReport drop;
        std::istringstream in2(tiny);
        const auto q = io::read_mps(in2, drop, opt);
        CHECK(q.nnz() == 2);
        CHECK(drop.small_values_dropped == 1);
        CHECK_NEAR(drop.largest_small_value_dropped, 1.0e-11, 1e-20);
        CHECK(!drop.warnings.empty());
        // The COLUMN survives; only the coefficient went.
        CHECK(q.n_cols() == 2);
        CHECK_NEAR(coeff(q, 1, 1), 2.0, 1e-15);
        // A coefficient exactly AT the threshold is dropped ("at or below").
        io::MpsReadOptions at_opt;
        at_opt.small_matrix_value = 1.0e-11;
        io::MpsReadReport at_rep;
        std::istringstream in3(tiny);
        const auto s = io::read_mps(in3, at_rep, at_opt);
        CHECK(at_rep.small_values_dropped == 1);
        CHECK(s.nnz() == 2);
    }

    // ---- gzip (via system zlib) ----
    //
    // Content sniffing rather than extension matching, round-tripping, and
    // loud failure on a corrupt stream rather than a silently truncated model.
    {
        CHECK(!io::has_gzip_magic("", 0));
        CHECK(!io::has_gzip_magic("NAME", 4));
        const char magic[2] = {'\x1f', '\x8b'};
        CHECK(io::has_gzip_magic(magic, 2));

        // A stored (uncompressed) DEFLATE block inside a minimal gzip
        // container. Building it by hand keeps the test free of any
        // compressor.
        const std::string payload = "ROWS\n N  OBJ\nENDATA\n";
        std::string gz;
        gz += '\x1f'; gz += '\x8b'; gz += '\x08';   // magic, CM = DEFLATE
        gz += '\x00';                                // no flags
        gz.append(4, '\x00');                        // mtime
        gz += '\x00'; gz += '\x03';                  // XFL, OS
        gz += '\x01';                                // BFINAL=1, BTYPE=00
        const auto n = static_cast<unsigned>(payload.size());
        gz += static_cast<char>(n & 0xff);
        gz += static_cast<char>((n >> 8) & 0xff);
        gz += static_cast<char>(~n & 0xff);
        gz += static_cast<char>((~n >> 8) & 0xff);
        gz += payload;
        const auto crc = io::crc32(payload.data(), payload.size());
        for (int i = 0; i < 4; ++i) gz += static_cast<char>((crc >> (8 * i)) & 0xff);
        for (int i = 0; i < 4; ++i) gz += static_cast<char>((n >> (8 * i)) & 0xff);

        CHECK(io::gunzip(gz.data(), gz.size()) == payload);

        // A corrupt payload must be REJECTED, not returned. A silently wrong
        // model is far worse than a failed read.
        std::string broken = gz;
        broken[broken.size() - 12] = 'Z';   // inside the stored payload
        CHECK_THROWS(io::gunzip(broken.data(), broken.size()));

        // Truncation likewise.
        const std::string cut = gz.substr(0, gz.size() - 4);
        CHECK_THROWS(io::gunzip(cut.data(), cut.size()));

        // Not gzip at all.
        CHECK_THROWS(io::gunzip(payload.data(), payload.size()));

        // A stream that carries the gzip magic but is too short to hold one
        // member is TRUNCATED, not empty. Returning "" here would hand the MPS
        // reader a model with no rows and let it be solved.
        const std::string stub = gz.substr(0, 12);
        CHECK_THROWS(io::gunzip(stub.data(), stub.size()));
        const char only_magic[2] = {'\x1f', '\x8b'};
        CHECK_THROWS(io::gunzip(only_magic, 2));
    }

    return sor::test::finish("test_mps");
}
