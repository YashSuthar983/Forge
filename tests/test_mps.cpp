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

    return sor::test::finish("test_mps");
}
