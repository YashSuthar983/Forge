// P9: cuts separated at a tree node are put into that node's LP for a warm
// re-solve; the certified objective raises the node's bound and can close it.
// Soundness is checked against the exhaustive oracle on random models with
// root cuts off (so the tree has fractional nodes to cut) and tree
// separation forced at every node, with and without the re-solve: the proven
// answer must be the oracle's and identical either way, and no reported bound
// may pass the optimum.
#include "forge/certify/finalize.hpp"
#include "forge/search/bab.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>

using forge::core::Status;
using forge::search::BabDiagnostics;
using forge::search::BabOptions;
using forge::search::MilpPolicy;
using forge::test::oracle::make_branching_ip;
using forge::test::oracle::make_random_milp;
using forge::test::oracle::solve_oracle;

namespace {

BabOptions tree_options(bool resolve) {
    BabOptions o;
    o.structural_presolve.binary_row_max_work = 0;  // Exercise the tree service, not the exact root oracle.
    o.para_bab.threads = 1;
    o.policy = MilpPolicy::Latest;
    o.tree_cut.enabled = true;
    o.tree_cut.always_depth = 64;
    o.tree_cut.every_k = 1;
    o.tree_cut.stall_skip_nodes = 0;
    o.tree_cut.max_cuts_per_node = 6;
    o.tree_cut.resolve_with_local = resolve;
    o.tree_cut.max_local_rows = 60;   // exercise inheritance by descendants
    // The opt-in cut variants are exercised at solve level here too.
    o.cut.tableau_cmir = true;
    o.cut.relax_small_terms = true;
    o.mir.relax_small_terms = true;
    o.cuts_enabled = true;    // authorize tree cuts
    o.cut.max_rounds = 0;     // keep the root cut loop out of this component test
    o.structural_presolve.enabled = false;
    o.mip_presolve = o.probing = o.symmetry = false;
    o.integer_row_rounding = false;
    o.feasibility_jump = false;
    o.sub_mip_lns = false;
    o.rounding_heuristic = false;
    o.lp_rounding_repair = false;
    o.integer_dive = false;
    o.integer_neighborhood = false;
    o.balans.enabled = false;
    o.kernel_pump.enabled = false;
    o.mrens.enabled = false;
    o.hgtsm.enabled = false;
    o.time_limit_s = 20.0;
    o.max_nodes = 20000;
    o.gap_tol = o.abs_gap_tol = 1e-9;
    return o;
}

struct Totals {
    std::uint64_t models = 0, optimal = 0, resolves = 0, proved = 0, raises = 0, prunes = 0;
    std::uint64_t inherited = 0, local_nodes = 0, basis_kept = 0;
};

template <class Make>
void run(const char* label, Make make, std::uint32_t first, std::uint32_t last,
         Totals& tot) {
    for (std::uint32_t seed = first; seed <= last; ++seed) {
        const auto lp = make(seed);
        const auto oracle = solve_oracle(lp);
        for (const bool resolve : {false, true}) {
            auto o = tree_options(resolve);
            BabDiagnostics d;
            auto raw = forge::search::solve_milp(lp, o, d);
            const auto res = forge::certify::finalize_result(
                std::move(raw), forge::search::milp_evidence(d, o));
            if (resolve) {
                ++tot.models;
                tot.resolves += d.node_cut_resolves;
                tot.proved += d.node_cut_resolves_proved;
                tot.raises += d.node_cut_bound_raises;
                tot.prunes += d.node_cut_prunes;
                tot.inherited += d.node_cuts_inherited;
                tot.local_nodes += d.node_local_cut_nodes;
                tot.basis_kept += d.node_local_cut_basis_kept;
            } else {
                CHECK(d.node_cut_resolves == 0);
                CHECK(d.node_cuts_inherited == 0);
                // Certified probe geometry may be scoped even without a cut re-solve.
            }
            if (res.status == Status::Optimal) {
                if (resolve) ++tot.optimal;
                CHECK(oracle.feasible);
                const bool same = !oracle.feasible ||
                    std::fabs(res.objective - oracle.objective) <=
                        1e-6 * (1.0 + std::fabs(oracle.objective));
                if (!same)
                    std::cout << "WRONG " << label << " seed=" << seed << " resolve=" << resolve
                              << " got " << res.objective << " oracle " << oracle.objective << '\n';
                CHECK(same);
            } else if (res.status == Status::Infeasible) {
                CHECK(!oracle.feasible);
            }
            if (oracle.feasible && std::isfinite(d.dual_bound)) {
                const double sense = lp.maximize ? -1.0 : 1.0;
                const bool ok = sense * d.dual_bound <= sense * oracle.objective + 1e-6;
                if (!ok)
                    std::cout << "BOUND PAST OPTIMUM " << label << " seed=" << seed
                              << " resolve=" << resolve << " bound " << d.dual_bound
                              << " oracle " << oracle.objective << '\n';
                CHECK(ok);
            }
        }
    }
}

}  // namespace

int main() {
    Totals ip, mixed;
    run("branching_ip", make_branching_ip, 1, 120, ip);
    run("random_milp", make_random_milp, 1, 300, mixed);
    std::cout << "NODE_CUT_RESOLVE ip: models=" << ip.models << " optimal=" << ip.optimal
              << " resolves=" << ip.resolves << " proved=" << ip.proved
              << " raises=" << ip.raises << " prunes=" << ip.prunes
              << " inherited=" << ip.inherited << " local_nodes=" << ip.local_nodes
              << " basis_kept=" << ip.basis_kept
              << " | mixed: models=" << mixed.models << " optimal=" << mixed.optimal
              << " resolves=" << mixed.resolves << " proved=" << mixed.proved
              << " raises=" << mixed.raises << " prunes=" << mixed.prunes << '\n';
    CHECK(ip.resolves + mixed.resolves > 0);
    // Descendants must actually solve with the inherited rows, warm.
    CHECK(ip.inherited > 0);
    CHECK(ip.local_nodes > 0);
    CHECK(ip.basis_kept > 0);
    CHECK(ip.optimal + mixed.optimal > 100);
    return forge::test::finish("test_node_cut_resolve");
}
