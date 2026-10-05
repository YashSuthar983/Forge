// GPU-track G1 call site (A2): BabOptions::gpu_binary_heuristic wires
// try_binquad_milp_heuristic() into solve_milp as a FALLBACK, run only when
// Feasibility Jump and Fix-Propagate-Repair found no incumbent -- see
// bab.cpp, immediately after the cold try_fixprop call. Every test below
// disables both of those so the fallback path is actually exercised instead
// of being skipped by !have_incumbent (which is what a plain small model
// would do: FJ alone solves these in well under 0.1s).
//
// Covers the eligible case (a pure-binary knapsack, found==1, final answer
// matches brute force), the ineligible case (a continuous column present,
// eligible==0, gpu_bin_init_ms==0 -- no device is even requested -- and the
// option changes nothing about the final answer), and a forced exception in
// device construction (SOR_TEST_BINQUAD_THROW) being caught with a correct
// cpu fallback rather than propagating out of solve_milp.
//
// The device is cached process-wide (see shared_binquad_device_locked() in
// bab.cpp): once any test below runs the fallback path for the first time,
// every later test in this binary reuses that SAME device (cpu, if the
// exception test ran first, as it does here) rather than creating a new
// one. That's the feature under test, so the exception test runs FIRST in
// main() -- correctness of the later tests does not depend on which device
// backend actually got cached, only on cpu being a fully valid backend.
#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"

#include "test_helpers.hpp"

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using sor::core::Status;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

// One knapsack row, N binary columns, minimize -sum(value_i * x_i) so the
// LP/MIP convention (minimize) matches a "maximize value" knapsack. The
// reader is whitespace-tolerant (see the other inline MPS fixtures in this
// suite), so exact column alignment is not required.
std::string build_knapsack_mps(const std::vector<double>& value,
                               const std::vector<double>& weight,
                               double capacity) {
    std::ostringstream out;
    out << "NAME KNAP12\nROWS\n N  OBJ\n L  R1\nCOLUMNS\n";
    out << " MARK0000 'MARKER' 'INTORG'\n";
    for (std::size_t i = 0; i < value.size(); ++i)
        out << " X" << (i + 1) << " OBJ " << -value[i] << " R1 " << weight[i] << "\n";
    out << " MARK0001 'MARKER' 'INTEND'\n";
    out << "RHS\n RHS R1 " << capacity << "\n";
    out << "BOUNDS\n";
    for (std::size_t i = 0; i < value.size(); ++i)
        out << " UP BND X" << (i + 1) << " 1\n";
    out << "ENDATA\n";
    return out.str();
}

// A binary column plus a continuous one: ineligible for the pure-binary
// heuristic regardless of presolve.
const char* kMixedContinuous = R"(NAME          MIXEDCONT
ROWS
 N  OBJ
 L  R1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        OBJ       -1             R1        1
    MARK0001  'MARKER'                 'INTEND'
    X2        OBJ       -1             R1        2
RHS
    RHS       R1        3
BOUNDS
 UP BND       X1        1
 UP BND       X2        2
ENDATA
)";

void test_gpu_binary_heuristic_matches_brute_force() {
    const std::vector<double> value = {10, 40, 30, 50, 35, 20, 15, 45, 25, 55, 5, 60};
    const std::vector<double> weight = {2, 3, 4, 5, 3, 2, 1, 6, 4, 7, 1, 8};
    const double capacity = 20.0;
    auto lp = read_text(build_knapsack_mps(value, weight, capacity));
    CHECK(static_cast<std::size_t>(lp.n_integer()) == value.size());

    BabOptions opts;
    opts.gpu_binary_heuristic = true;
    // Isolate the fallback: with FJ/fixprop on, they solve this trivial
    // knapsack before the GPU heuristic's turn (!have_incumbent would never
    // hold), which would test placement, not the heuristic.
    opts.feasibility_jump = false;
    opts.fixprop = false;
    opts.time_limit_s = 10.0;
    opts.max_nodes = 100000;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);

    CHECK(diag.gpu_bin_attempted == 1);
    CHECK(diag.gpu_bin_found == 1);
    CHECK(diag.gpu_bin_eligible == 1);

    double best_value = 0.0;
    const int n = static_cast<int>(value.size());
    for (int mask = 0; mask < (1 << n); ++mask) {
        double w = 0.0, v = 0.0;
        for (int i = 0; i < n; ++i)
            if (mask & (1 << i)) { w += weight[i]; v += value[i]; }
        if (w <= capacity + 1e-9 && v > best_value) best_value = v;
    }
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -best_value, 1e-6);
}

void test_continuous_column_is_ineligible_and_unchanged() {
    auto lp = read_text(kMixedContinuous);

    BabOptions off;
    off.feasibility_jump = false;
    off.fixprop = false;
    off.time_limit_s = 5.0;
    BabDiagnostics diag_off;
    auto raw_off = sor::search::solve_milp(lp, off, diag_off);
    const auto ev_off = sor::search::milp_evidence(diag_off, off);
    const auto r_off = sor::certify::finalize_result(std::move(raw_off), ev_off);

    BabOptions on = off;
    on.gpu_binary_heuristic = true;
    BabDiagnostics diag_on;
    auto raw_on = sor::search::solve_milp(lp, on, diag_on);
    const auto ev_on = sor::search::milp_evidence(diag_on, on);
    const auto r_on = sor::certify::finalize_result(std::move(raw_on), ev_on);

    CHECK(diag_on.gpu_bin_attempted == 1);
    CHECK(diag_on.gpu_bin_eligible == 0);
    CHECK(diag_on.gpu_bin_found == 0);
    // Ineligible: the eligibility check (a presolve pass only) is what
    // returned false, so no device was ever requested.
    CHECK(diag_on.gpu_bin_init_ms == 0.0);

    CHECK(r_off.status == r_on.status);
    CHECK_NEAR(r_off.objective, r_on.objective, 1e-9);
}

// A2: make_binquad_device("vulkan") is caught, not just null-checked, since
// the try/catch inside it wraps only device construction, not
// vk::Context::create() itself. SOR_TEST_BINQUAD_THROW forces that first
// attempt to throw; solve_milp must not propagate it, and the cpu fallback
// must still solve correctly.
void test_forced_device_exception_falls_back_to_cpu() {
    setenv("SOR_TEST_BINQUAD_THROW", "1", 1);
    const std::vector<double> value = {10, 40, 30, 50, 35, 20, 15, 45, 25, 55, 5, 60};
    const std::vector<double> weight = {2, 3, 4, 5, 3, 2, 1, 6, 4, 7, 1, 8};
    const double capacity = 20.0;
    auto lp = read_text(build_knapsack_mps(value, weight, capacity));

    BabOptions opts;
    opts.gpu_binary_heuristic = true;
    opts.feasibility_jump = false;
    opts.fixprop = false;
    opts.time_limit_s = 10.0;
    opts.max_nodes = 100000;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    unsetenv("SOR_TEST_BINQUAD_THROW");
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);

    CHECK(diag.gpu_bin_eligible == 1);
    CHECK(diag.gpu_bin_found == 1);

    double best_value = 0.0;
    const int n = static_cast<int>(value.size());
    for (int mask = 0; mask < (1 << n); ++mask) {
        double w = 0.0, v = 0.0;
        for (int i = 0; i < n; ++i)
            if (mask & (1 << i)) { w += weight[i]; v += value[i]; }
        if (w <= capacity + 1e-9 && v > best_value) best_value = v;
    }
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -best_value, 1e-6);
}

}  // namespace

int main() {
    // Must run first: the device is cached process-wide on first use (see
    // the file header comment), and this test is the one that depends on
    // controlling which backend that first use resolves to.
    test_forced_device_exception_falls_back_to_cpu();
    test_gpu_binary_heuristic_matches_brute_force();
    test_continuous_column_is_ineligible_and_unchanged();
    return sor::test::finish("test_bab_gpu_binary_heuristic");
}
