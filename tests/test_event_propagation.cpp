// Event-driven node propagation (P4): a child is propagated from its
// parent's fixpoint by visiting only the rows of the column it branched on.
// Solved twice per model -- event mode and the full sweep -- against
// exhaustive enumeration: both must give the oracle's optimum, status and a
// valid bound, and the event path must actually run.
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
using sor::test::oracle::make_branching_ip;
using sor::test::oracle::make_random_milp;
using sor::test::oracle::solve_oracle;

namespace {

BabOptions tree_options(bool event) {
    BabOptions o;
    o.para_bab.threads = 1;
    o.structural_presolve.enabled = false;
    o.gap_tol = o.abs_gap_tol = 1e-9;
    o.max_nodes = 20000;
    o.time_limit_s = 0.0;
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
    o.event_propagation = event;
    return o;
}

void test_event_matches_full_and_oracle() {
    std::uint64_t event_nodes = 0, full_nodes = 0, optimal = 0;
    for (std::uint32_t seed = 1; seed <= 3000; ++seed) {
        auto lp = (seed & 1u) ? make_random_milp(seed) : make_branching_ip(seed);
        if (seed & 1u) lp.obj_offset = static_cast<double>(static_cast<int>(seed % 7) - 3);
        const auto oracle = solve_oracle(lp);
        for (const bool event : {true, false}) {
            const BabOptions o = tree_options(event);
            BabDiagnostics d;
            auto raw = sor::search::solve_milp(lp, o, d);
            const auto ev = sor::search::milp_evidence(d, o);
            const auto res = sor::certify::finalize_result(raw, ev);
            if (event) event_nodes += d.prop_event_nodes; else full_nodes += d.prop_full_nodes;
            if (!event) CHECK(d.prop_event_nodes == 0);
            const double sense = lp.maximize ? -1.0 : 1.0;
            if (res.status == Status::Optimal) {
                ++optimal;
                const bool close = oracle.feasible &&
                    std::fabs(res.objective - oracle.objective) <=
                        1e-6 * (1.0 + std::fabs(oracle.objective));
                if (!close)
                    std::cout << "WRONG OPTIMAL seed=" << seed << " event=" << event
                              << " got " << res.objective << " oracle "
                              << oracle.objective << '\n';
                CHECK(close);
            }
            if (res.status == Status::Infeasible) {
                if (oracle.feasible)
                    std::cout << "WRONG INFEASIBLE seed=" << seed << " event=" << event << '\n';
                CHECK(!oracle.feasible);
            }
            if (oracle.feasible) {
                CHECK(res.status != Status::Infeasible);
                if (std::isfinite(d.dual_bound))
                    CHECK(sense * d.dual_bound <= sense * oracle.objective +
                                                       1e-7 * (1.0 + std::fabs(oracle.objective)));
            }
        }
    }
    std::cout << "EVENT_PROPAGATION event_nodes=" << event_nodes
              << " full_nodes=" << full_nodes << " optimal_claims=" << optimal << '\n';
    CHECK(event_nodes > 1000);
    CHECK(optimal > 1000);
}

}  // namespace

int main() {
    test_event_matches_full_and_oracle();
    return sor::test::finish("test_event_propagation");
}
