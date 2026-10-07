#include "sor/io/gzip.hpp"
#include "sor/io/mps.hpp"
#include "sor/io/write_mps.hpp"
#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
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
    // ---- crossed bounds read as an infeasible model, with a warning ----
    // The reader used to throw "col_lo > col_hi", so the CLI reported a parse
    // failure for a model whose answer is Infeasible.
    {
        std::istringstream in(
            "NAME          CROSSED\n"
            "ROWS\n"
            " N  COST\n"
            " L  R1\n"
            "COLUMNS\n"
            "    X         COST      1.0        R1        1.0\n"
            "RHS\n"
            "    RHS       R1        10.0\n"
            "BOUNDS\n"
            " LO BND       X          3.0\n"
            " UP BND       X          2.0\n"
            "ENDATA\n");
        io::MpsReadReport rep;
        const auto p = io::read_mps(in, rep);
        CHECK(p.col_lo[0] == 3.0 && p.col_hi[0] == 2.0);
        bool warned = false;
        for (const auto& w : rep.warnings)
            warned |= w == "column 'X' has lower bound 3 above upper bound 2; "
                           "the model is infeasible";
        CHECK(warned);
    }

    // ---- duplicate entries are summed exactly; exact zeros are not stored ----
    {
        // X: 0.1 + 0.2 - 0.3 on R1, which in floating point is 2^-54 in this
        // order and 2^-55 in another; exactly it is 2^-55. Y: 1 and -1 on R1
        // cancel. Z: an explicit 0 on R2. W: the only real entry of R2.
        std::istringstream in(
            "NAME          DUPS\n"
            "ROWS\n"
            " N  COST\n"
            " L  R1\n"
            " L  R2\n"
            "COLUMNS\n"
            "    X         R1        0.1\n"
            "    X         R1        0.2\n"
            "    X         R1        -0.3\n"
            "    Y         R1        1.0        R1        -1.0\n"
            "    Z         R2        0.0\n"
            "    W         R2        2.0\n"
            "RHS\n"
            "    RHS       R1        1.0        R2        1.0\n"
            "ENDATA\n");
        io::MpsReadReport rep;
        const auto p = io::read_mps(in, rep);
        CHECK(p.nnz() == 2);
        CHECK(p.A.vals[0] == std::ldexp(1.0, -55));
        CHECK(p.A.vals[1] == 2.0);
        CHECK(rep.duplicate_entries_summed == 3);
        CHECK(rep.zero_entries_dropped == 2);
        bool warned = false;
        for (const auto& w : rep.warnings)
            warned |= w.find("3 duplicate matrix entries summed") != std::string::npos;
        CHECK(warned);
    }

    // ---- tiny coefficients are kept, counted and warned about ----
    {
        const std::string text =
            "NAME          TINY\n"
            "ROWS\n"
            " N  COST\n"
            " L  R1\n"
            "COLUMNS\n"
            "    X         R1        1e-12\n"
            "    Y         R1        3e-10      COST      1.0\n"
            "    Z         R1        1.0\n"
            "RHS\n"
            "    RHS       R1        1.0\n"
            "ENDATA\n";
        std::istringstream in(text);
        io::MpsReadReport rep;
        const auto p = io::read_mps(in, rep);
        CHECK(p.nnz() == 3);
        CHECK(rep.tiny_entries == 2);
        CHECK(rep.smallest_entry == 1e-12);
        bool warned = false;
        for (const auto& w : rep.warnings)
            warned |= w.find("2 matrix coefficient(s) below 1e-9") != std::string::npos &&
                      w.find("--small-matrix-value") != std::string::npos;
        CHECK(warned);

        std::istringstream again(text);
        io::MpsReadReport dropped;
        io::MpsReadOptions o;
        o.small_matrix_value = 1e-9;
        CHECK(io::read_mps(again, dropped, o).nnz() == 1);
        CHECK(dropped.tiny_entries == 0 && dropped.small_values_dropped == 2);
    }

    // ---- free-format data lines may start in column 1 ----
    // Only a keyword alone (or NAME/OBJSENSE with an argument) is a header.
    // These lines used to be "unknown section" headers: strict reading
    // refused the file, lenient reading dropped the rest of the section.
    {
        std::istringstream in(
            "NAME          COLONE\n"
            "OBJSENSE\n"
            "MAX\n"
            "ROWS\n"
            "N  COST\n"
            "L  R1\n"
            "COLUMNS\n"
            "X         COST      1.0        R1        1.0\n"
            "RHS       COST      2.0        R1        1.0\n"
            "RHS\n"
            "RHS       R1        4.0\n"
            "BOUNDS\n"
            "UP BND       X          3.0\n"
            "ENDATA\n");
        io::MpsReadReport rep;
        const auto p = io::read_mps(in, rep);
        CHECK(p.maximize);
        CHECK(p.n_rows() == 1 && p.n_cols() == 2);
        CHECK(p.col_names[1] == "RHS");  // a column named like a section
        CHECK(p.c[0] == 1.0 && p.c[1] == 2.0);
        CHECK(p.row_hi[0] == 4.0);
        CHECK(p.col_hi[0] == 3.0);
        CHECK(rep.column_one_data_lines == 7);
    }

    // ---- several RHS/RANGES/BOUNDS sets: the first named one is used ----
    // All of them used to be applied, the last write winning. A valueless
    // bound without a set name ("FR Y") used to be "too short".
    {
        std::istringstream in(
            "NAME          SETS\n"
            "ROWS\n"
            " N  COST\n"
            " L  R1\n"
            " G  R2\n"
            "COLUMNS\n"
            "    X         COST      1.0        R1        1.0\n"
            "    Y         R2        1.0\n"
            "RHS\n"
            "    RHS1      R1        4.0\n"
            "    RHS2      R1        9.0        R2        9.0\n"
            "    R2        1.0\n"
            "    RHS1      R2        2.0\n"
            "RANGES\n"
            "    RNG1      R1        1.0\n"
            "    RNG2      R1        5.0\n"
            "BOUNDS\n"
            " UP BND1      X          3.0\n"
            " UP BND2      X          7.0\n"
            " UP BND1      X          2.0\n"
            " FR Y\n"
            "ENDATA\n");
        io::MpsReadReport rep;
        const auto p = io::read_mps(in, rep);
        CHECK(p.row_hi[0] == 4.0 && p.row_lo[0] == 3.0);  // RHS1 and RNG1
        CHECK(p.row_lo[1] == 2.0);   // unnamed 1.0, then RHS1's 2.0
        CHECK(p.col_hi[0] == 2.0);   // BND1, its last value
        CHECK(p.col_lo[1] == -sor::model::kInf && p.col_hi[1] == sor::model::kInf);
        CHECK(rep.ignored_set_lines == 3);
        CHECK(rep.repeated_entries == 2);  // R2 in RHS1, X's UP in BND1
        int set_warnings = 0;
        for (const auto& w : rep.warnings)
            set_warnings += w.find("is the one used") != std::string::npos;
        CHECK(set_warnings == 3);
    }

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

    // ---- infinity tokens where infinity is meaningful ----
    // Writers spell an absent bound as "Inf", "Infinity" or "-inf" (any case).
    // Rejecting the token made such files unreadable; a matrix coefficient or
    // cost of infinity stays an error (checked above and below).
    {
        const auto read = [](const std::string& body, sor::model::LpProblem& out) {
            std::istringstream in("NAME T\nROWS\n N  OBJ\n G  R1\n L  R2\n E  R3\n"
                                  "COLUMNS\n    X  OBJ  1  R1  1\n    X  R2  1  R3  1\n"
                                  "    Y  OBJ  1  R1  1\n" + body + "ENDATA\n");
            io::MpsReadReport rep;
            try { out = io::read_mps(in, rep); return true; }
            catch (const std::exception&) { return false; }
        };
        sor::model::LpProblem p;
        CHECK(read("RHS\n    RHS  R1  -Infinity  R2  INF\n    RHS  R3  2\n"
                   "RANGES\n    RNG  R3  inf\n"
                   "BOUNDS\n UP BND  X  Infinity\n LO BND  Y  -inf\n UP BND  Y  +Inf\n", p));
        CHECK(p.row_lo[0] == -sor::model::kInf && p.row_hi[0] == sor::model::kInf);
        CHECK(p.row_lo[1] == -sor::model::kInf && p.row_hi[1] == sor::model::kInf);
        CHECK(p.row_lo[2] == 2.0 && p.row_hi[2] == sor::model::kInf);
        CHECK(p.col_lo[0] == 0.0 && p.col_hi[0] == sor::model::kInf);
        CHECK(p.col_lo[1] == -sor::model::kInf && p.col_hi[1] == sor::model::kInf);
        // Infinity where no finite model can carry it is still refused.
        CHECK(!read("RHS\n    RHS  R3  inf\n", p));            // equality side
        CHECK(!read("BOUNDS\n LO BND  X  inf\n", p));            // lower bound +inf
        CHECK(!read("BOUNDS\n FX BND  X  -inf\n", p));           // fixed at infinity
        CHECK(!read("BOUNDS\n UP BND  X  nan\n", p));
    }

    // ---- writer round trip: every number and bound reads back exactly ----
    // The writer used six significant digits, gave integer (-inf, u] columns
    // no MI record (read back with lower bound 0), gave integer [0, +inf)
    // columns no record at all (read back as binary), and wrote a free row as
    // an equation with no RHS (read back as "= 0").
    {
        sor::model::LpProblem w;
        w.name = "RT";
        w.maximize = true;
        w.obj_offset = 1.0 / 7.0;
        w.A = sparse::from_triplets(4, 4, {0,0,1,1,2,3,3}, {0,1,1,2,3,0,3},
                                    {1.0 / 3.0, 0.1, 2.0, -1.0 / 7.0, 1.0, 3.0, 1e-7});
        w.c = {0.1, -1.0 / 3.0, 0.0, 2.5};
        w.col_lo = {0.0, -kInf, -2.0, -kInf};
        w.col_hi = {kInf, 5.0, 1.0 / 3.0, 7.25};
        w.is_integer = {true, true, false, false};
        w.row_lo = {-kInf, 0.3, -1.5, -kInf};
        w.row_hi = {2.0 / 3.0, 0.3, 2.5, kInf};       // L, E, exact range, free
        std::ostringstream text;
        io::write_mps(text, w);
        std::istringstream in(text.str());
        io::MpsReadReport rep;
        const auto r = io::read_mps(in, rep);
        CHECK(r.maximize && r.obj_offset == w.obj_offset);
        CHECK(r.n_rows() == 4 && r.n_cols() == 4);
        CHECK(r.c == w.c && r.A.vals == w.A.vals);
        CHECK(r.col_lo == w.col_lo && r.col_hi == w.col_hi);
        CHECK(r.row_lo == w.row_lo && r.row_hi == w.row_hi);
        CHECK(r.is_integer == w.is_integer);
        CHECK(text.precision() == 6);          // the caller's stream state is kept
    }

    // ---- quadratic sections: an error when strict, unless a caller reads them ----
    {
        const std::string qp =
            "NAME T\nROWS\n N  OBJ\nCOLUMNS\n    X  OBJ  -4\nBOUNDS\n UP BND  X  5\n"
            "QUADOBJ\n    X  X  2\nENDATA\n";
        {
            std::istringstream in(qp);
            io::MpsReadReport rep;
            io::MpsReadOptions strict;
            strict.strict = true;
            CHECK_THROWS(io::read_mps(in, rep, strict));
        }
        {
            std::istringstream in(qp);
            io::MpsReadReport rep;
            io::MpsReadOptions handled;
            io::QuadraticTerms terms;
            handled.strict = true;
            handled.quadratic = &terms;
            CHECK(io::read_mps(in, rep, handled).n_cols() == 1);
            CHECK(rep.warnings.empty());
            CHECK(terms.vals.size() == 1 && terms.vals[0] == 2.0 && !terms.full_matrix);
        }
        {
            std::istringstream in(qp);
            io::MpsReadReport rep;
            CHECK(io::read_mps(in, rep).n_cols() == 1);     // lenient: warn
            CHECK(rep.warnings.size() == 1);
        }
    }

    // ---- 1e20 and beyond spell "no bound" (CPLEX/HiGHS convention) ----
    // Kept finite, UP 1e30 is a box the simplex flips to, and every basic
    // value it touches becomes ~1e30. Only bounds and row sides convert;
    // a lower side converts only downward and an upper side only upward.
    {
        const std::string body =
            "NAME T\nROWS\n N  OBJ\n L  R1\n G  R2\nCOLUMNS\n"
            "    X  OBJ  1  R1  1\n    X  R2  1\n    Y  OBJ  1  R1  1e30\n"
            "RHS\n    RHS  R1  1e20  R2  -1e30\n"
            "BOUNDS\n UP BND  X  1e30\n LO BND  Y  -1e25\n UP BND  Y  9.9e19\nENDATA\n";
        {
            std::istringstream in(body);
            io::MpsReadReport rep;
            const auto p = io::read_mps(in, rep);
            CHECK(p.row_hi[0] == kInf && p.row_lo[1] == -kInf);
            CHECK(p.col_hi[0] == kInf);
            CHECK(p.col_lo[1] == -kInf && p.col_hi[1] == 9.9e19);   // below threshold
            CHECK(p.A.vals[1] == 1e30);   // row R1 = (X, Y): coefficients untouched
            CHECK(rep.infinite_bounds_read == 4);
        }
        {
            std::istringstream in(body);
            io::MpsReadReport rep;
            io::MpsReadOptions literal;
            literal.infinite_bound = 0.0;
            const auto p = io::read_mps(in, rep, literal);
            CHECK(p.row_hi[0] == 1e20 && p.col_hi[0] == 1e30 && p.col_lo[1] == -1e25);
            CHECK(rep.infinite_bounds_read == 0);
        }
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

        // The file reader inflates as it reads instead of holding the
        // compressed and decompressed file (it used to keep about five
        // copies at once). Same results, same refusals, and an error inside
        // the stream reaches the caller rather than a quiet end of file.
        const auto member = [](const std::string& text) {
            std::string m;
            m += '\x1f'; m += '\x8b'; m += '\x08'; m += '\x00';
            m.append(4, '\x00');
            m += '\x00'; m += '\x03'; m += '\x01';
            const auto len = static_cast<unsigned>(text.size());
            m += static_cast<char>(len & 0xff);
            m += static_cast<char>((len >> 8) & 0xff);
            m += static_cast<char>(~len & 0xff);
            m += static_cast<char>((~len >> 8) & 0xff);
            m += text;
            const auto c = io::crc32(text.data(), text.size());
            for (int i = 0; i < 4; ++i) m += static_cast<char>((c >> (8 * i)) & 0xff);
            for (int i = 0; i < 4; ++i) m += static_cast<char>((len >> (8 * i)) & 0xff);
            return m;
        };
        const auto path = std::filesystem::temp_directory_path() / "sor_test_stream.gz";
        const auto write = [&](const std::string& bytes) {
            std::ofstream f(path, std::ios::binary);
            f << bytes;
        };
        const auto slurp = [&]() {
            io::GzipFileStream in(path.string());
            return std::string(std::istreambuf_iterator<char>(in),
                               std::istreambuf_iterator<char>());
        };
        write(gz);
        CHECK(slurp() == payload);
        // Three 60000-byte members: crosses both the 64 KiB read chunks and
        // the member boundaries.
        std::string big, big_gz;
        for (int k = 0; k < 3; ++k) {
            std::string part(60000, static_cast<char>('a' + k));
            big += part;
            big_gz += member(part);
        }
        write(big_gz);
        CHECK(slurp() == big);
        for (const std::string& bad : {broken, cut, stub, std::string(only_magic, 2),
                                       gz + "junk"}) {
            write(bad);
            CHECK_THROWS(slurp());
        }
        // Through the MPS reader: a model, and a truncated one.
        write(member("NAME          GZ\nROWS\n N  OBJ\n L  R1\nCOLUMNS\n"
                     "    X         OBJ       1.0        R1        1.0\n"
                     "RHS\n    RHS       R1        2.0\nENDATA\n"));
        {
            io::MpsReadReport rep;
            const auto p = io::read_mps_file(path.string(), rep);
            CHECK(rep.used_gzip && p.n_rows() == 1 && p.n_cols() == 1);
        }
        write(cut);
        {
            io::MpsReadReport rep;
            CHECK_THROWS(io::read_mps_file(path.string(), rep));
        }
        std::filesystem::remove(path);
    }

    return sor::test::finish("test_mps");
}
