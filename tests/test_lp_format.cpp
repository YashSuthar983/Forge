// CPLEX LP format reader: structure, spellings, bounds, integer sections,
// rejected constructs, and dispatch from the shared MPS auto loader.
#include "sor/io/lp_format.hpp"
#include "sor/io/mps.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace sor;
using sor::model::kInf;

namespace {

double coeff(const model::LpProblem& p, core::Index r, core::Index c) {
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    for (auto k = rp[static_cast<std::size_t>(r)]; k < rp[static_cast<std::size_t>(r) + 1]; ++k)
        if (ci[static_cast<std::size_t>(k)] == c) return p.A.vals[static_cast<std::size_t>(k)];
    return 0.0;
}

int index_of(const std::vector<std::string>& v, const std::string& n) {
    for (std::size_t i = 0; i < v.size(); ++i) if (v[i] == n) return static_cast<int>(i);
    return -1;
}

model::LpProblem parse(const std::string& text, const io::MpsReadOptions& opt = {}) {
    std::istringstream in(text);
    io::MpsReadReport rep;
    return io::read_lp(in, rep, opt);
}

bool rejects(const std::string& text, const std::string& needle) {
    try {
        (void)parse(text);
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

}  // namespace

int main() {
    // ---- structure, both sides of a row, ranges, bounds, integer sections ----
    {
        const auto p = parse(R"(\ a comment line
Maximize
 obj: 3 x + 2 y + z - w + 5
Subject To
 c1: x + y + z <= 10
 c2: x - y >= -2
 c3: 2 x + 3 w = 6
 c4: -4 <= y + z - w <= 8
 c5: x + 2 >= y
Bounds
 0 <= x <= 4
 y >= -1
 z free
 w = 1.5
 1 <= v <= inf
General
 y
Binary
 b
End
)");
        CHECK(p.name == "obj");
        CHECK(p.maximize);
        CHECK(p.n_rows() == 5);
        CHECK(p.n_cols() == 6);
        CHECK(std::fabs(p.obj_offset - 5.0) < 1e-15);
        const int x = index_of(p.col_names, "x"), y = index_of(p.col_names, "y"),
                  z = index_of(p.col_names, "z"), w = index_of(p.col_names, "w"),
                  v = index_of(p.col_names, "v"), b = index_of(p.col_names, "b");
        CHECK(x == 0 && y == 1 && z == 2 && w == 3);   // order of first appearance
        CHECK(p.c[x] == 3.0 && p.c[y] == 2.0 && p.c[z] == 1.0 && p.c[w] == -1.0);

        CHECK(p.row_lo[0] == -kInf && p.row_hi[0] == 10.0);
        CHECK(p.row_lo[1] == -2.0 && p.row_hi[1] == kInf);
        CHECK(p.row_lo[2] == 6.0 && p.row_hi[2] == 6.0);
        CHECK(p.row_lo[3] == -4.0 && p.row_hi[3] == 8.0);
        CHECK(coeff(p, 3, y) == 1.0 && coeff(p, 3, z) == 1.0 && coeff(p, 3, w) == -1.0);
        // "x + 2 >= y"  ->  x - y >= -2
        CHECK(coeff(p, 4, x) == 1.0 && coeff(p, 4, y) == -1.0);
        CHECK(p.row_lo[4] == -2.0 && p.row_hi[4] == kInf);

        CHECK(p.col_lo[x] == 0.0 && p.col_hi[x] == 4.0);
        CHECK(p.col_lo[y] == -1.0 && p.col_hi[y] == kInf && p.is_integer[y]);
        CHECK(p.col_lo[z] == -kInf && p.col_hi[z] == kInf);
        CHECK(p.col_lo[w] == 1.5 && p.col_hi[w] == 1.5);
        CHECK(p.col_lo[v] == 1.0 && p.col_hi[v] == kInf);
        CHECK(p.col_lo[b] == 0.0 && p.col_hi[b] == 1.0 && p.is_integer[b]);
        CHECK(p.n_integer() == 2);
    }

    // ---- multi-line rows, comments, alternative spellings, duplicate terms ----
    {
        const auto p = parse(
            "Minimise\n obj: x\n + y \\ trailing comment\n"
            "Such That\n"
            " r1: x + y\n   >= 2\n"
            " r2: x < 5\n"
            " r3: y => 1\n"
            " r4: x + x + y =< 9\n"
            "END\n");
        CHECK(!p.maximize);
        CHECK(p.n_rows() == 4);
        CHECK(p.row_lo[0] == 2.0 && p.row_hi[0] == kInf);
        CHECK(p.row_lo[1] == -kInf && p.row_hi[1] == 5.0);
        CHECK(p.row_lo[2] == 1.0 && p.row_hi[2] == kInf);
        CHECK(coeff(p, 3, 0) == 2.0);                 // x + x summed
        CHECK(p.c[0] == 1.0 && p.c[1] == 1.0);
    }

    // ---- unlabeled rows: a new row starting with a sign or a name is not an rhs term ----
    {
        const auto p = parse(
            "Minimize\n obj: x + y\nSubject To\n"
            " x + y >= 2\n"
            " -x + y <= 3\n"
            " x - y = 0\n"
            "End\n");
        CHECK(p.n_rows() == 3);
        CHECK(p.row_names[0] == "R1" && p.row_names[2] == "R3");
        CHECK(coeff(p, 1, 0) == -1.0 && p.row_hi[1] == 3.0);
        CHECK(p.row_lo[2] == 0.0 && p.row_hi[2] == 0.0);
    }

    // ---- exponents, coefficient forms, infinity ----
    {
        const auto p = parse(
            "Minimize\n obj: 1.5e2 x + .5 y + 2*z\nSubject To\n"
            " c: 3e-1 x + y - z >= -1e+2\n"
            " d: x <= 1e30\n"
            "Bounds\n x >= -infinity\n y <= +inf\nEnd\n");
        CHECK(p.c[0] == 150.0 && p.c[1] == 0.5 && p.c[2] == 2.0);
        CHECK(std::fabs(coeff(p, 0, 0) - 0.3) < 1e-15);
        CHECK(p.row_lo[0] == -100.0);
        CHECK(p.row_hi[1] == kInf);                    // 1e30 is "no bound" by default
        CHECK(p.col_lo[0] == -kInf);
    }
    // With the infinite-bound convention disabled the number is taken literally.
    {
        io::MpsReadOptions literal;
        literal.infinite_bound = 0.0;
        const auto p = parse("Minimize\n obj: x\nSubject To\n d: x <= 1e30\n"
                             " e: x >= -1e20\nEnd\n", literal);
        CHECK(p.row_hi[0] == 1e30);
        CHECK(p.row_lo[1] == -1e20);
        // Just below the default threshold stays finite with it enabled.
        const auto q = parse("Minimize\n obj: x\nSubject To\n d: x <= 9.9e19\nEnd\n");
        CHECK(q.row_hi[0] == 9.9e19);
    }

    // ---- reader options ----
    {
        const auto p = parse("Minimize\n obj: b + v\nBounds\n b = 1\n v <= 1e30\nBinary\n b v\nEnd\n");
        CHECK(p.col_lo[0] == 1.0 && p.col_hi[0] == 1.0);
        CHECK(p.col_hi[1] == kInf && p.n_integer() == 2);
        const auto wrapped = parse("Minimize\n obj: 2\n x + y\nSubject To\n bounds: 3\n x + y >= 4\nEnd\n");
        CHECK(wrapped.c[0] == 2.0 && coeff(wrapped, 0, 0) == 3.0);
    }
    {
        const std::string text =
            "Minimize\n obj: x + 0.0000001 y\nSubject To\n c: x + 0.0000001 y >= 1\n"
            "General\n x\nEnd\n";
        io::MpsReadOptions o;
        o.relax_integrality = true;
        CHECK(parse(text, o).n_integer() == 0);
        CHECK(parse(text).n_integer() == 1);
        io::MpsReadOptions d;
        d.small_matrix_value = 1e-6;
        const auto p = parse(text, d);
        CHECK(p.nnz() == 1);
    }

    // ---- crossed bounds: a well-formed model that is infeasible ----
    // The reader used to throw, so the CLI said "parse failed" for a model
    // the solver should report Infeasible. It now reads it and warns.
    {
        std::istringstream in("Minimize\n obj: x\nSubject To\n c: 3 <= x + y <= 2\n"
                              "Bounds\n x <= -1\nEnd\n");
        io::MpsReadReport rep;
        const auto p = io::read_lp(in, rep);
        CHECK(p.col_lo[0] == 0.0 && p.col_hi[0] == -1.0);
        CHECK(p.row_lo[0] == 3.0 && p.row_hi[0] == 2.0);
        const auto empty = p.find_empty_domain();
        CHECK(!empty.is_row && empty.index == 0);
        bool column_warned = false, row_warned = false;
        for (const auto& w : rep.warnings) {
            column_warned |= w.find("variable 'x' has lower bound") != std::string::npos &&
                             w.find("negative upper") != std::string::npos;
            row_warned |= w.find("constraint 'c' has lower side above upper side") !=
                          std::string::npos;
        }
        CHECK(column_warned);
        CHECK(row_warned);
    }

    // ---- rejected constructs are errors, never silently dropped ----
    CHECK(rejects("Minimize\n obj: 1e-999 x\nEnd\n", "underflows"));
    CHECK(rejects("Minimize\n obj: x\nSubject To\n c: x <= 1e308 + 1e308\nEnd\n", "overflows"));
    CHECK(rejects("Minimize\n obj: x\nSubject To\n c: x - 1e308 <= 1e308\nEnd\n", "overflows"));
    CHECK(rejects("Minimize\n obj: x\nSubject To\n c: -1e308 <= x + 1e308 <= 1e308\nEnd\n", "overflows"));
    CHECK(rejects("Minimize\n obj: x + [ x ^ 2 ] / 2\nSubject To\n c: x >= 1\nEnd\n", "quadratic"));
    CHECK(rejects("Minimize\n obj: x\nSubject To\n c: x >= 1\nSOS\n s1: S1:: x:1\nEnd\n", "not supported"));
    CHECK(rejects("Minimize\n obj: x\nSubject To\n c: x >= 1\nSemi-continuous\n x\nEnd\n", "not supported"));
    CHECK(rejects("Minimize\n obj: x\nSubject To\n c: b = 1 -> x >= 1\nEnd\n", "indicator"));
    CHECK(rejects("Minimize\n obj: x\nSubject To\n c: x * y >= 1\nEnd\n", "products"));
    CHECK(rejects("Subject To\n c: x >= 1\nEnd\n", "Minimize or Maximize"));
    CHECK(rejects("Minimize\n obj: x\nSubject To\n c: x 1\nEnd\n", "expected"));
    CHECK(rejects("Minimize\n obj: x\nSubject To\n c: 1 <= x >= 3\nEnd\n", "ranged"));
    CHECK(rejects("Minimize\n obj: x\nBounds\n 1 <= x >= 3\nEnd\n", "ranged bound"));
    CHECK(rejects("Minimize\n obj: x\nBounds\n x <= 1e999\nEnd\n", "overflows"));
    CHECK(rejects("Minimize\n obj: x + inf\nEnd\n", "finite"));
    CHECK(rejects("Minimize\n obj: x\nEnd\n x >= 5\n", "after End"));
    CHECK(rejects("Minimize\n obj: x\nSubject To\n c: x >= 1\n", "missing End"));
    CHECK(rejects("Minimize\n obj: 1e308 x + 1e308 x\nEnd\n", "overflows"));

    // ---- extension test and dispatch through the shared MPS auto loader ----
    CHECK(io::has_lp_extension("a/b/model.lp"));
    CHECK(io::has_lp_extension("model.LP.GZ"));
    CHECK(!io::has_lp_extension("model.mps"));
    {
        const auto path = std::filesystem::temp_directory_path() / "sor_test_lp_format.lp";
        {
            std::ofstream f(path);
            f << "Minimize\n obj: x + y\nSubject To\n c: x + y >= 2\nBounds\n x <= 5\nEnd\n";
        }
        io::MpsReadReport rep;
        const auto p = io::read_mps_file_auto(path.string(), rep);
        CHECK(p.n_rows() == 1 && p.n_cols() == 2 && rep.n_rows == 1);
        std::filesystem::remove(path);
    }

    return sor::test::finish("test_lp_format");
}
