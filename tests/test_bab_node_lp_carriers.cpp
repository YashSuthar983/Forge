// S3.2: solve_milp's serial node loop passes a persistent DualEdgeWeightCarrier
// and FactorCarrier (see dual_simplex.hpp) across warm-started node LPs,
// tokened to &global_lp and cleared whenever global_lp.n_rows() changes.
// FactorCarrier's own adoption/rejection rules are covered by
// test_factor_reuse.cpp. This test checks the bab.cpp integration: both
// carriers must be adopted in the node loop and the answer must match a
// brute-force optimum. In particular, the factor carrier needs the global
// LP's row, column and nonzero counts as well as its matrix token.
#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"

#include "test_helpers.hpp"

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

std::string build_knapsack_mps(const std::vector<double>& value,
                               const std::vector<double>& weight,
                               double capacity) {
    std::ostringstream out;
    out << "NAME KNAPCARR\nROWS\n N  OBJ\n L  R1\nCOLUMNS\n";
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

void test_node_lp_carriers_reuse_weights_and_match_brute_force() {
    const std::vector<double> value = {94, 48, 59, 39, 24, 73, 76, 97, 43, 75, 99, 20, 78,
                                       24, 84, 60, 62, 95, 49, 54, 70, 67, 21, 42, 67};
    const std::vector<double> weight = {24, 39, 29, 21, 21, 11, 40, 40, 28, 34, 16, 7, 10,
                                        12, 6, 30, 12, 14, 35, 28, 34, 16, 34, 36, 29};
    const double capacity = 272.7;
    auto lp = read_text(build_knapsack_mps(value, weight, capacity));

    BabOptions opts;
    opts.time_limit_s = 20.0;
    opts.max_nodes = 2000000;
    // This test is about carrying factors and weights between node LPs, which
    // needs a tree; the opt-in tableau c-MIR would close this knapsack at the
    // root, so pin it off.
    opts.cut.tableau_cmir = false;
    opts.root_primal_early = false;   // a tree is needed to carry LP state
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);

    CHECK(diag.nodes > 10);
    CHECK(diag.node_lp_dse_reuses > 0);
    CHECK(diag.node_lp_factor_reuses > 0);
    CHECK(diag.node_lp_factor_reuse_rows_mismatch == 0);

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
    test_node_lp_carriers_reuse_weights_and_match_brute_force();
    return sor::test::finish("test_bab_node_lp_carriers");
}
