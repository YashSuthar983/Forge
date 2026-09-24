// --engine auto (.qplib) dispatch table: for each QPLIB 3-letter
// classification, does the CLI print the engine this solver actually has
// the best answer for today?
//
// This shells out to the real sor_solve binary (there is no library entry
// point for CLI argument routing) and reads back the "auto engine:" line it
// prints right after resolving the classification -- before the chosen
// engine gets anywhere near actually solving the instance.  Some of the
// synthetic instances below are deliberately tiny and some skip the QPLIB
// trailer entirely; that is fine here, because what is under test is which
// branch the CLI takes, not whether that branch reaches Optimal. The 453-
// instance sweep (scripts/qplib_all_sweep.py) is what measures the solves.
//
// Two real sample files already checked into benchmarks/qplib/ cover the
// binquad and qcqplocal routes; the rest of the table (qpauto, global,
// miqp via linear rows, qpipm, miqp via quadratic rows, and the Unsupported
// fallback for an unrecognised class) is covered with minimal instances
// built directly in QPLIB's own section order (see sor/io/qplib.hpp).
#include "test_helpers.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Run {
    int exit_code = 0;
    std::string output;
};

std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (const char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

Run run(const std::vector<std::string>& argv) {
    std::string cmd;
    for (const auto& a : argv) cmd += shell_quote(a) + " ";
    cmd += "2>&1";
    Run r;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        r.exit_code = -1;
        return r;
    }
    std::array<char, 512> buf{};
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr)
        r.output += buf.data();
    const int status = pclose(pipe);
    r.exit_code = (status == -1) ? -1 : (status / 256);
    return r;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string solve_exe;
std::string tmp_dir;

std::string write_file(const std::string& name, const std::string& content) {
    const std::string path = (fs::path(tmp_dir) / name).string();
    std::ofstream f(path);
    f << content;
    return path;
}

// Runs --engine auto on `path` and checks it routes to `expect_engine`. A
// tight time limit keeps every case fast regardless of what the chosen
// engine then does with the (possibly minimal / trailer-less) instance.
void expect_route(const std::string& label, const std::string& path,
                  const std::string& expect_engine) {
    const Run r = run({solve_exe, path, "--engine", "auto", "--time-limit", "0.2"});
    const std::string want = "auto engine:       " + expect_engine;
    ::sor::test::report(contains(r.output, want), "routes to the expected engine",
                        __FILE__, __LINE__,
                        label + " -> wanted \"" + want + "\", got:\n" + r.output);
}

}  // namespace

int main() {
    const fs::path src(SOR_SOURCE_DIR), bin(SOR_BINARY_DIR);
    solve_exe = (bin / "sor_solve").string();
    if (!fs::is_regular_file(solve_exe)) {
        std::fprintf(stderr, "missing sor_solve; nothing to validate\n");
        return 1;
    }
    tmp_dir = (fs::temp_directory_path() / "sor_qplib_auto_engine_test").string();
    fs::create_directories(tmp_dir);

    // ---- real library samples, when present ----------------------------
    // Instance data is not tracked (QP.md); these routes are checked only when
    // the files have been fetched. The synthetic cases below cover every
    // classification letter regardless, so the route table is still tested.
    const auto qplib_dir = src / "benchmarks" / "qplib";
    if (fs::is_regular_file(qplib_dir / "QPLIB_2430.qplib")) {
        expect_route("LCQ (real sample)", (qplib_dir / "QPLIB_2430.qplib").string(),
                    "qcqplocal");
        expect_route("QCQ (real sample)", (qplib_dir / "QPLIB_1157.qplib").string(),
                    "qcqplocal");
        expect_route("QBB (real sample)", (qplib_dir / "QPLIB_3565.qplib").string(),
                    "binquad");
        expect_route("QBL (real sample)", (qplib_dir / "QPLIB_3834.qplib").string(),
                    "binquad");
    }

    // ---- CCB: convex continuous, box-only (no A/m block at all) --------
    // qpauto: interior point, first-order fallback with the remaining
    // budget. Same minimal, previously-verified CCB layout as
    // tests/test_qplib.cpp's infinity-sentinel case.
    expect_route("CCB (synthetic)", write_file("ccb.qplib",
        "TINY_CCB\nCCB\nminimize\n2\n"
        "1\n1 1 2.0\n"          // nnz(H)=1: H(1,1)=2.0
        "0.0\n1\n2 -1.0\n"      // g default 0, override g(2)=-1
        "0.0\n"                 // f
        "1e20\n"                // infinity sentinel
        "-1e20\n0\n"            // x_l default, no overrides
        "3.0\n0\n"), "qpauto"); // x_u default 3, no overrides

    // ---- QCL: nonconvex continuous objective, linear constraint --------
    // global: spatial branch-and-bound over an indefinite diagonal H.
    expect_route("QCL (synthetic)", write_file("qcl.qplib",
        "TINY_QCL\nQCL\nminimize\n2\n1\n"
        "2\n1 1 2.0\n2 2 -1.0\n"      // nnz(H)=2, indefinite diagonal
        "0.0\n0\n"                    // g default 0, no overrides
        "0.0\n"                       // f
        "2\n1 1 1.0\n1 2 1.0\n"       // nnz(A)=2: row1 = x1 + x2
        "1e20\n"                      // infinity sentinel
        "0.0\n0\n5.0\n0\n"            // c_lo default 0, c_hi default 5
        "-1e20\n0\n1e20\n0\n"),       // x_l default, x_u default (var C)
        "global");

    // ---- CML: convex objective, mixed binary+continuous, linear row ----
    // miqp: B&B over convex IPM node relaxations (no quadratic constraint
    // rows here, so this is the plain-linear-rows MIQP branch).
    expect_route("CML (synthetic)", write_file("cml.qplib",
        "TINY_CML\nCML\nminimize\n2\n1\n"
        "2\n1 1 2.0\n2 2 2.0\n"       // nnz(H)=2, convex diagonal
        "0.0\n0\n"                    // g default 0
        "0.0\n"                       // f
        "2\n1 1 1.0\n1 2 1.0\n"       // nnz(A)=2: row1 = x1 + x2
        "1e20\n"
        "0.0\n0\n5.0\n0\n"            // c_lo/c_hi (row <= 5)
        "0.0\n0\n1.0\n0\n"            // x_l/x_u default [0,1] (var M)
        "0\n1\n1 2\n"),               // vartype default Continuous(0),
        "miqp");                      // override: var 1 -> Binary(2)

    // ---- LCD: linear objective, continuous vars, diagonal convex row ---
    // qpipm: the IPM's own QCQP path, the one class this table treats as
    // convex on the classification alone.
    expect_route("LCD (synthetic)", write_file("lcd.qplib",
        "TINY_LCD\nLCD\nminimize\n2\n1\n"
        // obj is 'L': no objective Hessian block.
        "0.0\n2\n1 1.0\n2 1.0\n"      // g default 0, overrides g(1)=g(2)=1
        "0.0\n"                       // f
        "2\n1 1 1 2.0\n1 2 2 2.0\n"   // per-row H: row1 has x1^2 + x2^2 (PSD)
        "0\n"                         // nnz(A)=0 (no linear part in the row)
        "1e20\n"
        "-1e20\n0\n4.0\n0\n"          // c_lo default -inf, c_hi default 4
        "-1e20\n0\n1e20\n0\n"),       // x_l/x_u default (var C)
        "qpipm");

    // ---- LMQ: linear objective, mixed binary+continuous, quadratic row -
    // miqp: same B&B, but through the miqcqp_bb (quadratic-row) branch
    // rather than the plain-linear-rows one CML exercised above.
    expect_route("LMQ (synthetic)", write_file("lmq.qplib",
        "TINY_LMQ\nLMQ\nminimize\n2\n1\n"
        "1.0\n0\n"                    // g default 1, no overrides
        "0.0\n"                       // f
        "1\n1 1 1 2.0\n"              // per-row H: row1 has x1^2 (PSD)
        "0\n"                         // nnz(A)=0
        "1e20\n"
        "-1e20\n0\n4.0\n0\n"          // c_lo/c_hi
        "0.0\n0\n1.0\n0\n"            // x_l/x_u default [0,1] (var M)
        "0\n1\n1 2\n"),               // var 1 -> Binary
        "miqp");

    // ---- an unrecognised classification: Unsupported, not a guess ------
    {
        const std::string path = write_file("bogus.qplib", "TINY_BOGUS\nXCB\n");
        const Run r = run({solve_exe, path, "--engine", "auto", "--time-limit", "0.2"});
        ::sor::test::report(contains(r.output, "model class:       XCB"),
                            "prints the unrecognised class before refusing",
                            __FILE__, __LINE__, r.output);
        ::sor::test::report(contains(r.output, "Unsupported"),
                            "refuses by status, not a crash or a silent guess",
                            __FILE__, __LINE__, r.output);
        ::sor::test::report(contains(r.output, "not a recognised"),
                            "names the reason", __FILE__, __LINE__, r.output);
        ::sor::test::report(!contains(r.output, "auto engine:"),
                            "never prints a chosen engine for an unroutable class",
                            __FILE__, __LINE__, r.output);
    }

    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
    return sor::test::finish("test_qplib_auto_engine");
}
