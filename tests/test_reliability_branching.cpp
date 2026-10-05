// Reliability branching on the node-LP session (P10). With every candidate
// treated as unreliable, strong branching runs at every node; certified dead
// directions become bound tightenings and a node with both directions dead is
// closed. Checked against exhaustive enumeration on random MILPs (min/max,
// objective offsets, general integers): the answer, the status and the bound
// must agree, and the deduction paths must actually be exercised.
#include "sor/certify/finalize.hpp"
#include "sor/search/bab.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>

using sor::core::Status;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::test::oracle::make_random_milp;
using sor::test::oracle::solve_oracle;

namespace {

using sor::test::oracle::make_branching_ip;

void test_probing_everywhere_matches_oracle() {
    std::uint64_t probes = 0, deductions = 0, closed = 0, session_nodes = 0,
                  optimal = 0, infeasible = 0, inference_probes = 0, probe_clauses = 0;
    for (std::uint32_t seed = 1; seed <= 3000; ++seed) {
        // Alternate the two families: mixed models (continuous columns,
        // equalities) and pure-integer branching models.
        auto lp = (seed & 1u) ? make_random_milp(seed) : make_branching_ip(seed);
        if (seed & 1u) lp.obj_offset = static_cast<double>(static_cast<int>(seed % 7) - 3);
        const auto oracle = solve_oracle(lp);
        BabOptions o;
        o.para_bab.threads = 1;
        o.structural_presolve.enabled = false;
        o.gap_tol = o.abs_gap_tol = 1e-9;
        o.max_nodes = 5000;
        o.time_limit_s = 0.0;
        // Force a real tree: no cuts, no primal heuristics, no root reductions.
        o.cuts_enabled = false;
        o.feasibility_jump = false;
        o.fixprop = false;
        o.sub_mip_lns = false;
        o.balans.enabled = false;
        o.kernel_pump.enabled = false;
        o.mrens.enabled = false;
        o.integer_dive = false;
        o.integer_neighborhood = false;
        o.rounding_heuristic = false;
        o.lp_rounding_repair = false;
        o.mip_presolve = o.probing = o.symmetry = false;
        o.integer_row_rounding = false;
        o.tree_cut.enabled = false;
        o.reliability_threshold = 1000;   // nothing is reliable: probe everywhere
        o.rb_max_probed = 50;
        o.rb_lookahead_candidates = 50;
        o.rb_lp_time_share = 1e9;         // budget never binds
        o.rb_startup_ms = 1e9;
        BabDiagnostics d;
        auto raw = sor::search::solve_milp(lp, o, d);
        const auto ev = sor::search::milp_evidence(d, o);
        const auto res = sor::certify::finalize_result(raw, ev);
        probes += d.strong_branch_solves;
        deductions += d.sb_domain_reductions;
        closed += d.sb_nodes_closed;
        session_nodes += d.rb_reliable_nodes;
        inference_probes += d.domain_probe_runs;
        probe_clauses += d.farkas_probe_clauses;
        const double sense = lp.maximize ? -1.0 : 1.0;
        if (res.status == Status::Optimal) {
            ++optimal;
            const bool close = oracle.feasible &&
                std::fabs(res.objective - oracle.objective) <=
                    1e-6 * (1.0 + std::fabs(oracle.objective));
            if (!close)
                std::cout << "WRONG OPTIMAL seed=" << seed << " got " << res.objective
                          << " oracle " << oracle.objective << '\n';
            CHECK(close);
        }
        if (res.status == Status::Infeasible) {
            ++infeasible;
            if (oracle.feasible)
                std::cout << "WRONG INFEASIBLE seed=" << seed << '\n';
            CHECK(!oracle.feasible);
        }
        if (oracle.feasible) {
            CHECK(res.status != Status::Infeasible);
            if (std::isfinite(d.dual_bound))
                CHECK(sense * d.dual_bound <=
                      sense * oracle.objective + 1e-7 * (1.0 + std::fabs(oracle.objective)));
        } else {
            CHECK(res.status != Status::Optimal);
        }
    }
    std::cout << "RELIABILITY_STRESS probes=" << probes << " deductions=" << deductions
              << " nodes_closed=" << closed << " session_selected_nodes=" << session_nodes
              << " optimal=" << optimal << " infeasible=" << infeasible << '\n';
    std::cout << "INFERENCE_PROBES runs=" << inference_probes
              << " farkas_clauses=" << probe_clauses << '\n';
    CHECK(probes > 500);
    CHECK(deductions > 20);
    CHECK(closed > 5);
    CHECK(inference_probes > 0);   // LP observations/updated scores must not suppress them
    CHECK(probe_clauses > 0);       // certified probe rays must reach persistent learning
    CHECK(optimal > 500);
}

}  // namespace

int main() {
    test_probing_everywhere_matches_oracle();
    return sor::test::finish("test_reliability_branching");
}
