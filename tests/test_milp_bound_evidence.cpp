// Proof-bound preservation (P1). A bound certified for a region must survive
// every later event that does not refute it: an interrupted or unproved
// re-solve, a cached LP being replaced, a cut leaving the model, a restart.
// And no unproved LP value may ever stand in for a bound. The unit checks pin
// the combination rules exactly; the oracle stress forces unproved node LPs
// (a pivot cap of a few iterations) and checks every claim the solver still
// makes against exhaustive enumeration.
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::BoundCertificate;
using sor::search::RelaxationOutcome;
using sor::test::oracle::make_random_milp;
using sor::test::oracle::solve_oracle;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void test_certificate_rules() {
    BoundCertificate c;
    CHECK(c.value == -kInf);
    CHECK(c.source == BoundCertificate::Source::None);
    CHECK(c.raise(3.0, BoundCertificate::Source::LpOptimal));
    CHECK(c.value == 3.0);
    // A weaker or non-finite value never lowers or replaces it.
    CHECK(!c.raise(2.5, BoundCertificate::Source::Lagrangian));
    CHECK(!c.raise(-kInf, BoundCertificate::Source::Lagrangian));
    CHECK(!c.raise(std::nan(""), BoundCertificate::Source::Lagrangian));
    CHECK(c.value == 3.0 && c.source == BoundCertificate::Source::LpOptimal);
    CHECK(c.raise(4.0, BoundCertificate::Source::Frontier));
    CHECK(c.source == BoundCertificate::Source::Frontier);

    // No evidence at all (interrupted, no multipliers): the inherited bound.
    RelaxationOutcome none;
    none.point_valid = true;  // a feasible point is not a bound
    CHECK(sor::search::node_bound_after_relaxation(7.0, none) == 7.0);
    CHECK(sor::search::node_bound_after_relaxation(-kInf, none) == -kInf);
    // Unproved but with a Lagrangian certificate: raised, never lowered.
    RelaxationOutcome lag;
    lag.lagrangian_bound = 9.0;
    CHECK(sor::search::node_bound_after_relaxation(7.0, lag) == 9.0);
    lag.lagrangian_bound = 5.0;
    CHECK(sor::search::node_bound_after_relaxation(7.0, lag) == 7.0);
    // Proved LP below the inherited bound (roundoff, different local rows):
    // both are valid for the node, keep the stronger.
    RelaxationOutcome proved;
    proved.lp_optimal = true;
    proved.lp_bound = 6.999;
    CHECK(sor::search::node_bound_after_relaxation(7.0, proved) == 7.0);
    proved.lp_bound = 8.0;
    CHECK(sor::search::node_bound_after_relaxation(7.0, proved) == 8.0);
}

// Every claim must still be true when node LPs are routinely cut off after
// a handful of pivots: unproved LPs keep their inherited bound, integral
// points from them do not close their subtree, and no unproved objective
// becomes a bound. Maximisation and objective offsets are in the mix.
void test_interrupted_node_lps_stay_sound() {
    std::uint64_t optimal = 0, infeasible = 0, bounded = 0, kept = 0, folded = 0;
    std::uint64_t deferred = 0, retried = 0, more_nodes = 0;
    // Dual simplex stopped early rarely holds a primal-feasible point; primal
    // simplex stopped in phase 2 does, which is what reaches the "integral
    // point from an unproved LP" path. Run both.
    for (std::uint32_t run = 1; run <= 1600; ++run) {
        const bool primal = run > 800;
        const std::uint32_t seed = primal ? run - 800 : run;
        auto lp = make_random_milp(seed);
        lp.obj_offset = static_cast<double>(static_cast<int>(seed % 7) - 3);
        const auto oracle = solve_oracle(lp);
        BabOptions o;
        o.para_bab.threads = 1;
        o.structural_presolve.enabled = false;
        o.time_limit_s = 0.0;           // node LPs end on pivots, not on time
        o.max_nodes = 400;
        o.lp.max_iterations = primal ? 2 + seed % 3 : 1 + seed % 2;
        o.lp.presolve = false;          // presolve alone solves these LPs
        if (primal) o.lp.method = sor::engines::SimplexMethod::Primal;
        o.gap_tol = o.abs_gap_tol = 1e-9;
        BabDiagnostics d;
        auto raw = sor::search::solve_milp(lp, o, d);
        const auto ev = sor::search::milp_evidence(d, o);
        const auto res = sor::certify::finalize_result(std::move(raw), ev);
        kept += d.unproved_bounds_kept;
        folded += d.unproved_regions_folded;
        deferred += d.node_lp_deferred;
        retried += d.node_lp_retries;
        {
            // The same run with the old behaviour (an unusable LP ends the
            // search) can only have explored fewer nodes.
            BabOptions stop = o;
            stop.node_lp_max_retries = 0;
            BabDiagnostics sd;
            (void)sor::search::solve_milp(lp, stop, sd);
            CHECK(d.nodes >= sd.nodes);
            if (d.nodes > sd.nodes) ++more_nodes;
        }
        const double sense = lp.maximize ? -1.0 : 1.0;
        if (res.status == sor::core::Status::Optimal) {
            ++optimal;
            CHECK(oracle.feasible);
            if (oracle.feasible && std::fabs(res.objective - oracle.objective) > 1e-6)
                std::cout << "WRONG OPTIMAL seed=" << seed << " got " << res.objective
                          << " oracle " << oracle.objective << '\n';
            CHECK(!oracle.feasible ||
                  std::fabs(res.objective - oracle.objective) <= 1e-6);
        }
        if (res.status == sor::core::Status::Infeasible) {
            ++infeasible;
            CHECK(!oracle.feasible);
        }
        if (oracle.feasible && std::isfinite(d.dual_bound)) {
            ++bounded;
            if (sense * d.dual_bound > sense * oracle.objective + 1e-7)
                std::cout << "BOUND PAST OPTIMUM seed=" << seed << " bound "
                          << d.dual_bound << " oracle " << oracle.objective << '\n';
            CHECK(sense * d.dual_bound <= sense * oracle.objective + 1e-7);
        }
        // The root certificate is a bound on the whole problem too.
        if (oracle.feasible && std::isfinite(d.root_certified_bound))
            CHECK(sense * d.root_certified_bound <= sense * oracle.objective + 1e-7);
    }
    std::cout << "INTERRUPTED_STRESS optimal=" << optimal << " infeasible=" << infeasible
              << " bounded=" << bounded << " unproved_bounds_kept=" << kept
              << " regions_folded=" << folded << '\n';
    // The stress must actually reach the paths it is about.
    CHECK(kept > 0);
    CHECK(folded > 0);
    // An unusable node LP no longer ends the search: nodes are set aside,
    // retried, and the rest of the frontier is still searched.
    CHECK(deferred > 0);
    CHECK(retried > 0);
    CHECK(more_nodes > 0);
    CHECK(optimal > 0);
}

// A global bound, once certified at the root, is never undercut by the
// reported final bound: every open node inherits at least the root
// certificate, whatever happened to later relaxations.
void test_root_certificate_floors_final_bound() {
    const std::string path = std::string(SOR_SOURCE_DIR) +
                             "/benchmarks/miplib-easy/mps/p0201.mps";
    if (!sor::test::data_available(path)) {
        sor::test::skip("test_root_certificate_floors_final_bound", path);
        return;
    }
    std::ifstream in(path);
    sor::io::MpsReadReport rep;
    const auto lp = sor::io::read_mps(in, rep);
    for (const std::uint64_t nodes : {1ull, 2ull, 25ull}) {
        BabOptions o;
        o.para_bab.threads = 1;
        o.max_nodes = nodes;
        o.time_limit_s = 1000.0;
        BabDiagnostics d;
        sor::search::solve_milp(lp, o, d);
        const double sense = lp.maximize ? -1.0 : 1.0;
        CHECK(std::isfinite(d.root_certified_bound));
        CHECK(d.root_bound_raises > 0);
        if (std::isfinite(d.dual_bound) && std::isfinite(d.root_certified_bound))
            CHECK(sense * d.dual_bound >= sense * d.root_certified_bound -
                                              1e-9 * (1.0 + std::fabs(d.dual_bound)));
    }
}

}  // namespace

int main() {
    test_certificate_rules();
    test_interrupted_node_lps_stay_sound();
    test_root_certificate_floors_final_bound();
    return sor::test::finish("test_milp_bound_evidence");
}
