// MILP measurement contract: counters reconcile, supported options reach the
// code they name, and the capability inventory matches what runs. These are
// operation-based assertions -- they read counts and identities, never wall
// time thresholds -- so they hold on a loaded machine.
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"

#include "test_helpers.hpp"

#include <cstring>
#include <fstream>
#include <set>
#include <string>

using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::CapabilityStatus;

namespace {

const std::string kDir = std::string(SOR_SOURCE_DIR) + "/benchmarks/miplib-easy/mps/";

bool load(const std::string& name, sor::model::LpProblem& out) {
    const std::string path = kDir + name + ".mps";
    if (!sor::test::data_available(path)) return false;
    std::ifstream in(path);
    sor::io::MpsReadReport rep;
    out = sor::io::read_mps(in, rep);
    return true;
}

// Deterministic configuration: the node limit, not the clock, ends the run,
// and the heuristics whose outcome depends on a wall-clock budget are off.
BabOptions deterministic_options(std::uint64_t nodes) {
    BabOptions o;
    o.max_nodes = nodes;
    o.time_limit_s = 1000.0;
    o.strong_branch_time_s = 0.0;
    o.rb_startup_ms = 1e12; // The deterministic test uses pivot/node budgets.
    o.para_bab.threads = 1;
    o.feasibility_jump = false;
    o.fixprop = false;
    o.sub_mip_lns = false;
    o.balans.enabled = false;
    o.kernel_pump.enabled = false;
    o.mrens.enabled = false;
    o.integer_dive = false;
    o.integer_neighborhood = false;
    o.lp_rounding_repair = false;
    return o;
}

void check_cut_identities(const BabDiagnostics& d) {
    CHECK(d.root_prefilter.total() == d.root_cuts_prefilter_rejected);
    CHECK(d.root_cuts_prefilter_rejected <= d.root_cuts_generated);
    CHECK(d.root_cuts_gate_dropped <= d.root_cuts_selected);
    // A selected cut becomes one appended row, one tightened row, or is
    // refused by apply_cuts (duplicate / parallel to an existing row).
    CHECK(d.root_cut_rows_appended + d.root_cut_rows_tightened <=
          d.root_cuts_selected - d.root_cuts_gate_dropped);
    CHECK(d.root_cut_rows_active <= d.root_cut_rows_appended);
    CHECK(d.cut_lp_warm_hits <= d.cut_lp_warm_attempts);
    CHECK(d.tree_cuts_prefilter_rejected <= d.tree_cuts_generated);
    CHECK(d.tree_local_cuts_selected <=
          d.tree_cuts_generated - d.tree_cuts_prefilter_rejected);
    // Local rows are not carried into descendants by default (max_local_rows = 0).
    CHECK(d.tree_local_cuts_inserted == 0);
}

void check_probe_identities(const BabDiagnostics& d, bool batch_lp) {
    if (!batch_lp)
        CHECK(d.strong_branch_proved + d.strong_branch_infeasible +
                  d.strong_branch_unproved == d.strong_branch_solves);
    if (d.strong_branch_proved > 0) CHECK(d.strong_branch_iterations > 0);
    CHECK(d.rb_nodes_with_sb <= d.rb_nodes);
    CHECK(d.ms_branch_candidates + d.ms_branch_features <= d.ms_branching + 1e-6);
}

void test_counters_reconcile() {
    sor::model::LpProblem lp;
    if (!load("p0201", lp)) {
        sor::test::skip("test_counters_reconcile", kDir + "p0201.mps");
        return;
    }
    BabOptions o = deterministic_options(150);
    BabDiagnostics d;
    sor::search::solve_milp(lp, o, d);
    CHECK(d.nodes > 1);
    CHECK(d.root_cuts_generated > 0);
    CHECK(d.strong_branch_solves > 0);
    check_cut_identities(d);
    check_probe_identities(d, o.batch_lp_strong_branch);
}

// The measurement fields must not perturb the search: two identical runs
// agree on every decision-bearing count and on the answer.
void test_deterministic_repeat() {
    sor::model::LpProblem lp;
    if (!load("misc03", lp)) {
        sor::test::skip("test_deterministic_repeat", kDir + "misc03.mps");
        return;
    }
    const BabOptions o = deterministic_options(120);
    BabDiagnostics a, b;
    const auto ra = sor::search::solve_milp(lp, o, a);
    const auto rb = sor::search::solve_milp(lp, o, b);
    CHECK(a.nodes == b.nodes);
    CHECK(a.lp_iterations == b.lp_iterations);
    CHECK(a.cut_rounds == b.cut_rounds);
    CHECK(a.root_cut_rows_active == b.root_cut_rows_active);
    CHECK(a.strong_branch_iterations == b.strong_branch_iterations);
    CHECK(a.rb_nodes == b.rb_nodes);
    CHECK(ra.proposed_status == rb.proposed_status);
    CHECK(ra.objective == rb.objective || (std::isnan(ra.objective) &&
                                           std::isnan(rb.objective)));
}

// root_cut_share was a declared field the loop never read (it used a
// hard-coded 35%). Round 0 -- the root LP and the separation from it -- is
// exempt from the cap, so a zero share must stop the loop before round 1
// re-solves: at most the one round of cuts round 0 applied.
void test_root_cut_share_reaches_loop() {
    sor::model::LpProblem lp;
    if (!load("p0201", lp)) {
        sor::test::skip("test_root_cut_share_reaches_loop", kDir + "p0201.mps");
        return;
    }
    BabOptions o = deterministic_options(1);
    BabDiagnostics normal;
    sor::search::solve_milp(lp, o, normal);
    CHECK(normal.cut_rounds > 1);
    CHECK(normal.cut_loop_time_capped == 0);

    o.root_cut_share = 0.0;
    BabDiagnostics capped;
    sor::search::solve_milp(lp, o, capped);
    CHECK(capped.cut_rounds <= 1);
    CHECK(capped.cut_loop_time_capped >= 1);
    CHECK(std::isfinite(capped.root_bound_before_cuts));  // round 0 still ran
}

// root_reduction_share gates the root setup phases; zero skips symmetry.
void test_root_reduction_share_reaches_setup() {
    sor::model::LpProblem lp;
    if (!load("p0201", lp)) {
        sor::test::skip("test_root_reduction_share_reaches_setup", kDir + "p0201.mps");
        return;
    }
    BabOptions o = deterministic_options(1);
    BabDiagnostics normal;
    sor::search::solve_milp(lp, o, normal);
    CHECK(normal.symmetry_diag.aborted_on_time == 0);

    o.root_reduction_share = 0.0;
    BabDiagnostics none;
    sor::search::solve_milp(lp, o, none);
    CHECK(none.symmetry_diag.aborted_on_time == 1);
}

void test_capability_inventory() {
    const auto& inv = sor::search::milp_capability_inventory();
    CHECK(!inv.empty());
    std::set<std::string> names;
    for (const auto& c : inv) {
        CHECK(c.name != nullptr && std::strlen(c.name) > 0);
        CHECK(c.note != nullptr && std::strlen(c.note) > 0);
        CHECK(names.insert(c.name).second);
    }
    const auto status_of = [&](const char* name) {
        for (const auto& c : inv)
            if (std::strcmp(c.name, name) == 0) return c.status;
        CHECK(false);
        return CapabilityStatus::Implemented;
    };
    // Empty sources and never-read options: must not be reported working.
    // Implemented since P11 (conflict_store.cpp is no longer empty).
    CHECK(status_of("persistent conflict store (bound disjunctions)") !=
          CapabilityStatus::Unavailable);
    CHECK(status_of("objective-face search") == CapabilityStatus::Implemented);
    CHECK(status_of("set-partitioning / assignment repair") !=
          CapabilityStatus::Unavailable);
    CHECK(status_of("independent component solving") !=
          CapabilityStatus::Unavailable);
    CHECK(status_of("reliability branching") == CapabilityStatus::Implemented);
    CHECK(status_of("tree / local cuts") != CapabilityStatus::Implemented);
}

}  // namespace

// Primal work runs from the first proved root LP (not only after the root
// node).
void test_root_primal_pass() {
    sor::model::LpProblem lp;
    if (!load("p0201", lp)) { sor::test::skip("test_root_primal_pass", kDir + "p0201.mps"); return; }
    auto o = deterministic_options(50);
    o.root_primal_early = true;
    o.objective_face = true;
    o.sub_mip_lns = true;      // the objective face rides on the sub-MIP machinery
    o.time_limit_s = 60.0;
    BabDiagnostics d;
    (void)sor::search::solve_milp(lp, o, d);
    CHECK(d.root_primal_passes >= 1);
    BabOptions off = o;
    off.root_primal_early = false;
    BabDiagnostics d0;
    (void)sor::search::solve_milp(lp, off, d0);
    CHECK(d0.root_primal_passes == 0);
}

// Budget accounting: every cap is a REMAINING amount, so an allowance can never
// exceed what the category or the enclosing operation has left; nothing left
// means zero, never "unlimited".
void test_budget_allowance() {
    using sor::search::budget_allowance;
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(budget_allowance(5.0, {10.0, 8.0, inf}) == 5.0);
    CHECK(budget_allowance(5.0, {10.0, 1.5, inf}) == 1.5);      // category has 1.5 s left
    CHECK(budget_allowance(5.0, {0.2, 8.0, inf}) == 0.2);       // deadline is close
    CHECK(budget_allowance(5.0, {10.0, 8.0, 0.3}) == 0.3);      // enclosing pass has 0.3 s left
    CHECK(budget_allowance(5.0, {10.0, -2.0, inf}) == 0.0);     // overspent category: nothing
    CHECK(budget_allowance(5.0, {std::nan(""), 8.0}) == 0.0);   // unknown cap: nothing
    CHECK(budget_allowance(-1.0, {10.0}) == 0.0);
    CHECK(budget_allowance(inf, {4.0, 9.0}) == 4.0);
    // Nested: a pass with 1 s left runs two attempts; the second gets the rest.
    double pass_left = 1.0;
    const double first = budget_allowance(0.7, {10.0, pass_left});
    pass_left -= first;
    const double second = budget_allowance(0.7, {10.0, pass_left});
    CHECK(first == 0.7);
    CHECK(std::fabs(second - 0.3) < 1e-12);
    pass_left -= second;
    CHECK(budget_allowance(0.7, {10.0, pass_left}) < 1e-12);
}

int main() {
    test_budget_allowance();
    test_root_primal_pass();
    test_capability_inventory();
    test_counters_reconcile();
    test_deterministic_repeat();
    test_root_cut_share_reaches_loop();
    test_root_reduction_share_reaches_setup();
    return sor::test::finish("test_milp_diagnostics");
}
