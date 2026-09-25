// CLI argument validation for sor_solve and sor_check.
//
// These options used to be parsed with strtod/strtoull and a null end pointer,
// which is silent about everything that matters: "abc" parses as 0, "-5" wraps
// to a huge unsigned, and "nan"/"inf" are taken at face value. A mistyped flag
// therefore produced a DIFFERENT RUN rather than an error -- `--tol abc` solved
// at tolerance exactly 0 and still exited 0.
//
// So the contract this file pins down is deliberately narrow and behavioural:
//   * a rejected value exits NONZERO,
//   * the message names the offending OPTION and echoes the offending VALUE,
//     so the user can see which of several numeric flags they got wrong,
//   * and a valid value on the same option is still accepted, which is what
//     stops the validation from being "reject everything".
//
// The test shells out to the real binaries; there is no library entry point for
// argument parsing, and the exit code is part of what is under test.
#include "test_helpers.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Run {
    int exit_code = 0;
    std::string output;   // stdout and stderr together
};

// Quote an argument for /bin/sh. Values under test include things like
// "1e-6; echo" and "--", so this must not rely on them being tame.
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

std::string solve_exe, check_exe, model, solution_file, crash_model;

// A rejected value must exit nonzero AND say which option and which value.
void reject(const char* option, const std::string& value) {
    const Run r = run({solve_exe, model, option, value, "--no-presolve"});
    const std::string where = std::string(option) + " " + value;
    ::sor::test::report(r.exit_code != 0, "nonzero exit", __FILE__, __LINE__,
                        where + " -> exit " + std::to_string(r.exit_code));
    ::sor::test::report(contains(r.output, option), "message names the option",
                        __FILE__, __LINE__, where + " -> " + r.output);
    ::sor::test::report(contains(r.output, value), "message echoes the value",
                        __FILE__, __LINE__, where + " -> " + r.output);
}

// The counterweight: the same option with a sane value must still work.
void accept(const char* option, const std::string& value) {
    const Run r = run({solve_exe, model, option, value, "--no-presolve",
                       "--time-limit", "20"});
    ::sor::test::report(r.exit_code == 0, "accepts a valid value", __FILE__,
                        __LINE__, std::string(option) + " " + value +
                                      " -> exit " + std::to_string(r.exit_code) +
                                      "\n" + r.output);
}

// ---------------------------------------------------------------------------

// Real-valued flags. Each is checked against the whole family of ways a value
// can be wrong, because they all funnel through one parser and a regression in
// it would otherwise show up on only one flag.
void test_real_flags_reject_malformed_values() {
    for (const char* option : {"--tol", "--time-limit"}) {
        reject(option, "nan");
        reject(option, "-nan");
        reject(option, "inf");
        reject(option, "-inf");
        reject(option, "infinity");
        reject(option, "abc");
        reject(option, "");            // empty string is not a number
        reject(option, "1e-6xyz");     // trailing garbage must not be ignored
        reject(option, "1,5");         // comma decimal separator
        reject(option, "--no-presolve");  // a forgotten value eats the next flag
        // Deliberately NOT asserted either way: std::stod accepts C99 hex float
        // notation, so "0x10" parses cleanly as 16. It is a strange tolerance,
        // but unlike "abc" nothing is silently reinterpreted, so pinning a
        // behaviour here would only make the parser harder to change.
    }
}

// --tol and --time-limit are strictly positive: zero is not a usable value for
// either, and it is exactly what strtod produced from a typo.
void test_strictly_positive_flags_reject_zero_and_negatives() {
    for (const char* option : {"--tol", "--time-limit"}) {
        reject(option, "0");
        reject(option, "-0.0");
        reject(option, "-1");
        reject(option, "-1e-9");
    }
}

// The nonnegative reals accept 0 (it means "off") but nothing below it.
void test_nonnegative_real_flags() {
    for (const char* option : {"--refactor-eta-ratio", "--refactor-work-ratio",
                               "--dual-cost-perturbation",
                               "--small-matrix-value"}) {
        reject(option, "nan");
        reject(option, "inf");
        reject(option, "abc");
        reject(option, "-1");
        reject(option, "-0.5");
        accept(option, "0");
    }
}

void test_integer_flags_reject_malformed_values() {
    for (const char* option : {"--max-iter", "--refactor-interval",
                               "--dual-resync-interval"}) {
        reject(option, "abc");
        reject(option, "");
        reject(option, "1.5");      // a real where an integer is required
        reject(option, "5abc");
        reject(option, "1e6");      // scientific notation is not an integer here
        reject(option, "nan");
        reject(option, "inf");
        // stoull WRAPS a negative into a huge positive, so this is the one that
        // silently produced an effectively infinite iteration cap.
        reject(option, "-5");
        reject(option, "-1");
    }
}

void test_max_iter_rejects_zero_but_intervals_accept_it() {
    // A zero iteration cap can never finish; a zero refactor interval means
    // "no interval trigger", which is a legitimate configuration.
    reject("--max-iter", "0");
    accept("--refactor-interval", "0");
    accept("--dual-resync-interval", "0");
}

void test_valid_values_still_run() {
    accept("--tol", "1e-7");
    accept("--time-limit", "30");
    accept("--max-iter", "100000");
    accept("--refactor-interval", "64");
    accept("--refactor-eta-ratio", "2.5");
    accept("--refactor-work-ratio", "1.0");
    accept("--dual-resync-interval", "100");
    accept("--dual-cost-perturbation", "1e-7");
}

// A flag at the very end of the command line has no value to consume. This
// must be a usage error, not a read past the end of argv.
void test_missing_values_at_end_of_argv() {
    for (const char* option : {"--tol", "--time-limit", "--max-iter",
                               "--refactor-interval", "--refactor-eta-ratio",
                               "--refactor-work-ratio", "--dual-resync-interval",
                               "--dual-cost-perturbation", "--engine", "--method",
                               "--pricing", "--basis-update", "--backend",
                               "--solution-out",
                               "--small-matrix-value", "--q-diag",
                               "--auto-budget-split"}) {
        const Run r = run({solve_exe, model, option});
        ::sor::test::report(r.exit_code != 0, "missing value exits nonzero",
                            __FILE__, __LINE__,
                            std::string(option) + " -> exit " +
                                std::to_string(r.exit_code));
        ::sor::test::report(contains(r.output, option) &&
                                contains(r.output, "needs a value"),
                            "missing value names the option", __FILE__, __LINE__,
                            std::string(option) + " -> " + r.output);
    }
}

// Enumerated options were already validated; this keeps that from regressing
// while the numeric parsing around them changes.
void test_enum_flags_reject_unknown_names() {
    const std::array<std::pair<const char*, const char*>, 5> cases{{
        {"--engine", "quantum"},
        {"--method", "sideways"},
        {"--pricing", "vibes"},
        {"--basis-update", "magic"},
        {"--backend", "abacus"},
    }};
    for (const auto& [option, value] : cases) {
        const Run r = run({solve_exe, model, option, value});
        ::sor::test::report(r.exit_code != 0, "unknown enum exits nonzero",
                            __FILE__, __LINE__,
                            std::string(option) + " " + value + " -> exit " +
                                std::to_string(r.exit_code));
        ::sor::test::report(contains(r.output, value), "unknown enum echoes value",
                            __FILE__, __LINE__,
                            std::string(option) + " -> " + r.output);
    }
}

void test_q_diag_rejects_malformed_entries() {
    for (const char* value : {"1.0,abc,2.0", "1.0,nan", "1.0,inf", "1.0,,2.0x"}) {
        const Run r = run({solve_exe, model, "--engine", "qp", "--q-diag", value});
        ::sor::test::report(r.exit_code != 0, "bad --q-diag exits nonzero",
                            __FILE__, __LINE__,
                            std::string(value) + " -> exit " +
                                std::to_string(r.exit_code));
    }
}

void test_unknown_option_and_no_arguments() {
    const Run unknown = run({solve_exe, model, "--not-an-option"});
    CHECK(unknown.exit_code != 0);
    CHECK(contains(unknown.output, "--not-an-option"));

    const Run none = run({solve_exe});
    CHECK(none.exit_code != 0);
    CHECK(contains(none.output, "usage"));

    // --help is the one non-model invocation that must SUCCEED.
    const Run help = run({solve_exe, "--help"});
    CHECK(help.exit_code == 0);
    CHECK(contains(help.output, "usage"));
}

void test_simplex_cli_crash_default_and_opt_out() {
    const auto crash_count = [](const Run& r) {
        const auto pos = r.output.find("primal crash");
        if (pos == std::string::npos) return -1LL;
        unsigned long long count = 0;
        if (std::sscanf(r.output.c_str() + pos,
                        "primal crash %llu", &count) != 1)
            return -1LL;
        return static_cast<long long>(count);
    };
    const std::vector<std::string> common{
        solve_exe, crash_model, "--engine", "simplex", "--method", "primal",
        "--no-presolve", "--verbose"};
    const Run default_run = run(common);
    auto disabled_args = common;
    disabled_args.push_back("--no-primal-crash");
    const Run disabled = run(disabled_args);
    auto explicit_args = common;
    explicit_args.push_back("--primal-crash");
    const Run explicit_run = run(explicit_args);

    CHECK(default_run.exit_code == 0);
    CHECK(disabled.exit_code == 0);
    CHECK(explicit_run.exit_code == 0);
    CHECK(crash_count(default_run) == 1);
    CHECK(crash_count(disabled) == 0);
    CHECK(crash_count(explicit_run) == 1);
}

void test_missing_model_file_is_reported() {
    const Run r = run({solve_exe, "/no/such/model.mps"});
    CHECK(r.exit_code != 0);
    CHECK(contains(r.output, "no/such/model.mps"));
}

void test_lp_dispatch_and_diagnostic_flags() {
    for (const char* engine : {"auto", "primal", "dual"}) {
        const Run r = run({solve_exe, model, "--engine", engine,
                           "--no-presolve", "--time-limit", "20"});
        ::sor::test::report(r.exit_code == 0, "LP engine name is accepted",
                            __FILE__, __LINE__,
                            std::string(engine) + " -> " + r.output);
    }

    for (const char* flag : {"--relax-integrality", "--fo-polish",
                             "--no-fo-polish", "--fo-certificates",
                             "--no-fo-certificates", "--fo-crossover",
                             "--no-fo-crossover"}) {
        const Run r = run({solve_exe, model, flag, "--no-presolve",
                           "--time-limit", "20"});
        ::sor::test::report(r.exit_code == 0, "LP boolean flag is accepted",
                            __FILE__, __LINE__,
                            std::string(flag) + " -> " + r.output);
    }

}

void test_unavailable_backends_are_not_silently_replaced() {
    const Run hpr = run({solve_exe, model, "--engine", "hpr",
                         "--backend", "cuda"});
    CHECK(hpr.exit_code != 0);
    CHECK(contains(hpr.output, "Unsupported"));
    CHECK(contains(hpr.output, "unavailable for HPR"));
    CHECK(!contains(hpr.output, "using cpu"));

    const Run pdhg = run({solve_exe, model, "--engine", "pdhg",
                          "--backend", "vulkan"});
    CHECK(pdhg.exit_code != 0);
    CHECK(contains(pdhg.output, "Unsupported"));
    CHECK(contains(pdhg.output, "unavailable for PDHG"));
    CHECK(!contains(pdhg.output, "using cpu"));
}

void test_auto_budget_split_is_a_closed_protocol_set() {
    for (const char* split : {"60/25/15", "70/20/10", "80/15/5"}) {
        const Run accepted = run({solve_exe, model, "--engine", "auto",
                                  "--auto-budget-split", split,
                                  "--time-limit", "20"});
        ::sor::test::report(accepted.exit_code == 0,
                            "approved Auto split is accepted", __FILE__,
                            __LINE__, split);
    }
    const Run rejected = run({solve_exe, model, "--engine", "auto",
                              "--auto-budget-split", "65/20/15"});
    CHECK(rejected.exit_code != 0);
    CHECK(contains(rejected.output, "--auto-budget-split"));
    CHECK(contains(rejected.output, "65/20/15"));
}

// sor_check has its own --tol with the same defect.
void test_sor_check_tolerance_validation() {
    for (const char* value : {"nan", "inf", "abc", "-1", "0", "1e-7xyz"}) {
        const Run r = run({check_exe, model, solution_file, "--tol", value});
        ::sor::test::report(r.exit_code != 0, "sor_check rejects bad --tol",
                            __FILE__, __LINE__,
                            std::string(value) + " -> exit " +
                                std::to_string(r.exit_code) + "\n" + r.output);
        ::sor::test::report(contains(r.output, value),
                            "sor_check echoes the bad --tol value", __FILE__,
                            __LINE__, std::string(value) + " -> " + r.output);
    }
    const Run missing = run({check_exe, model, solution_file, "--tol"});
    CHECK(missing.exit_code != 0);

    // And a good tolerance still checks the solution.
    const Run ok = run({check_exe, model, solution_file, "--tol", "1e-6"});
    ::sor::test::report(ok.exit_code == 0, "sor_check accepts a valid --tol",
                        __FILE__, __LINE__, ok.output);
}

}  // namespace


// Unreadable or model-less input must never produce a CLAIM.
//
// This is the failure these cases exist to prevent: the readers return an EMPTY
// model rather than an error, an empty LP is trivially optimal, and the solver
// therefore reported "Optimal / ProvedOptimalFP / objective 0" for a directory,
// an empty file, or a file that did not parse. A confident proof for input that
// was never read is worse than any crash, and worse than a plain error.
void test_unusable_input_is_refused() {
    const fs::path tmp = fs::temp_directory_path();
    const std::string missing = (tmp / "sor_cli_no_such_model.mps").string();
    fs::remove(missing);
    const std::string empty_file = (tmp / "sor_cli_empty.mps").string();
    { std::ofstream out(empty_file); }
    const std::string nomodel = (tmp / "sor_cli_nomodel.mps").string();
    {
        std::ofstream out(nomodel);
        out << "NAME          NOTHING\nROWS\nCOLUMNS\nRHS\nBOUNDS\nENDATA\n";
    }

    struct Case { std::string path; const char* what; };
    const Case cases[] = {
        {missing, "missing file"},
        {tmp.string(), "a directory"},
        {empty_file, "an empty file"},
        {nomodel, "a file with no columns"},
    };
    for (const auto& c : cases) {
        const Run r = run({solve_exe, c.path, "--engine", "auto"});
        // nonzero exit, and above all NOT a claim of optimality
        CHECK(r.exit_code != 0);
        CHECK(!contains(r.output, "ProvedOptimalFP"));
        CHECK(!contains(r.output, "status:            Optimal"));
        CHECK(contains(r.output, "error:"));
        if (r.exit_code == 0 || contains(r.output, "ProvedOptimalFP"))
            std::fprintf(stderr, "  %s was not refused\n", c.what);
    }
    fs::remove(empty_file);
    fs::remove(nomodel);
}

// An output path that cannot be written is refused BEFORE solving, so a long
// solve does not finish into a file that was never openable.
void test_unwritable_solution_out_is_refused() {
    const Run r = run({solve_exe, model, "--engine", "auto", "--time-limit", "30",
                       "--solution-out", "/sor_no_such_dir/x.sol"});
    CHECK(r.exit_code != 0);
    CHECK(contains(r.output, "--solution-out"));
    // refused up front: the run must not have got as far as reporting a status
    CHECK(!contains(r.output, "status:"));
}

// A mistyped engine option is fatal, not ignored: a dropped option would make a
// tuning run attribute its number to a setting that never applied.
void test_bad_engine_option_is_refused() {
    for (const auto& kv : {std::string("--qp-opt"), std::string("--qcqp-opt"),
                           std::string("--miqp-opt"), std::string("--global-opt")}) {
        const Run r = run({solve_exe, model, kv, "definitely_not_an_option=1"});
        CHECK(r.exit_code != 0);
        CHECK(contains(r.output, "error:"));
        CHECK(!contains(r.output, "status:"));
    }
    // a whole-number option refuses a fractional value rather than truncating
    const Run frac = run({solve_exe, model, "--miqp-opt", "max_nodes=2.5"});
    CHECK(frac.exit_code != 0);
    CHECK(contains(frac.output, "whole number"));
}


// A zero-column model must not be reported as solved by ANY engine.
//
// Raised in review on the QP routes: a zero-column .qps still came back
// "Optimal / ProvedKKT". The first fix guarded only the LP path, so the four
// quadratic engines still claimed it -- which is why the guard now sits in the
// shared QP loader and in the QPLIB reader rather than at each call site.
//
// Sweeping every engine rather than the reported one also turned up a
// zero-variable QPLIB instance that made the mixed-integer path claim
// ProvedGlobalEpsilon and made the binary-quadratic path SEGFAULT. A
// per-engine loop is the only assertion shape that catches those.
void test_zero_column_model_is_refused_by_every_engine() {
    const fs::path tmp = fs::temp_directory_path();
    const std::string qps = (tmp / "sor_cli_zerocol.qps").string();
    {
        std::ofstream out(qps);
        out << "NAME          ZEROCOL\nROWS\n N  COST\nCOLUMNS\nRHS\nBOUNDS\nENDATA\n";
    }
    const std::string qplib = (tmp / "sor_cli_zerovar.qplib").string();
    {
        std::ofstream out(qplib);
        out << "ZEROVAR\nLCL\nminimize\n0\n0\n0.0\n0\n0.0\n0\n1e20\n";
    }

    const char* engines[] = {"simplex", "auto", "qp", "qpipm", "qpauto",
                             "hprqp", "binquad", "qcqplocal", "global", "miqp"};
    for (const std::string& input : {qps, qplib}) {
        for (const char* eng : engines) {
            const Run r = run({solve_exe, input, "--engine", eng,
                               "--time-limit", "3"});
            // no claim of optimality, however the engine chooses to decline
            const bool claimed = contains(r.output, "ProvedOptimal") ||
                                 contains(r.output, "ProvedKKT") ||
                                 contains(r.output, "ProvedGlobal") ||
                                 contains(r.output, "status:            Optimal");
            CHECK(!claimed);
            // and it must not die on a signal
            CHECK(r.exit_code >= 0 && r.exit_code < 128);
            if (claimed || r.exit_code >= 128)
                std::fprintf(stderr, "  %s on %s: exit %d\n", eng,
                             input.c_str(), r.exit_code);
        }
    }
    fs::remove(qps);
    fs::remove(qplib);
}

int main() {
    const fs::path src(SOR_SOURCE_DIR), bin(SOR_BINARY_DIR);
    solve_exe = (bin / "sor_solve").string();
    check_exe = (bin / "sor_check").string();
    model = (src / "examples/testlp.mps").string();

    if (!fs::is_regular_file(solve_exe) || !fs::is_regular_file(check_exe) ||
        !fs::is_regular_file(model)) {
        std::fprintf(stderr, "missing binaries or model; nothing to validate\n");
        return 1;
    }

    test_unusable_input_is_refused();
    test_zero_column_model_is_refused_by_every_engine();
    test_unwritable_solution_out_is_refused();
    test_bad_engine_option_is_refused();

    // One good solve produces the solution file the sor_check cases need.
    solution_file = (fs::temp_directory_path() / "sor_cli_validation.sol").string();
    crash_model =
        (fs::temp_directory_path() / "sor_cli_primal_crash.mps").string();
    {
        std::ofstream out(crash_model);
        out << "NAME          CRASHCLI\n"
               "ROWS\n"
               " N  COST\n"
               " E  R1\n"
               "COLUMNS\n"
               "    X1        COST      0.0        R1        1.0\n"
               "RHS\n"
               "    RHS       R1        1.0\n"
               "ENDATA\n";
    }
    const Run seed = run({solve_exe, model, "--no-presolve", "--solution-out",
                          solution_file});
    ::sor::test::report(seed.exit_code == 0, "seed solve succeeded", __FILE__,
                        __LINE__, seed.output);

    test_real_flags_reject_malformed_values();
    test_strictly_positive_flags_reject_zero_and_negatives();
    test_nonnegative_real_flags();
    test_integer_flags_reject_malformed_values();
    test_max_iter_rejects_zero_but_intervals_accept_it();
    test_valid_values_still_run();
    test_missing_values_at_end_of_argv();
    test_enum_flags_reject_unknown_names();
    test_q_diag_rejects_malformed_entries();
    test_unknown_option_and_no_arguments();
    test_simplex_cli_crash_default_and_opt_out();
    test_missing_model_file_is_reported();
    test_lp_dispatch_and_diagnostic_flags();
    test_unavailable_backends_are_not_silently_replaced();
    test_auto_budget_split_is_a_closed_protocol_set();
    test_sor_check_tolerance_validation();

    std::error_code ec;
    fs::remove(solution_file, ec);
    fs::remove(crash_model, ec);
    return sor::test::finish("test_cli_validation");
}
