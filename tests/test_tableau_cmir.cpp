// Tableau c-MIR and small-term relaxation (P8). A cut is valid only if no
// feasible mixed-integer point violates it, so every cut the separator emits
// from the root LP basis of a random model is checked by minimising its own
// left-hand side over the model with the exhaustive oracle (integer
// assignments enumerated, continuous remainder solved by certified simplex):
// the minimum must not fall below the cut's right-hand side.
#include "sor/engines/simplex.hpp"
#include "sor/search/cuts.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>

using sor::test::oracle::make_random_milp;
using sor::test::oracle::solve_oracle;

namespace {

struct Totals {
    std::uint64_t models = 0, cuts = 0, checked = 0, tableau = 0, repaired = 0;
};

// Every cut against the oracle. `dynamism_max` small forces the repair path.
void run(const char* label, double dynamism_max, bool tableau_cmir,
         bool relax, Totals& tot) {
    for (std::uint32_t seed = 1; seed <= 700; ++seed) {
        auto lp = make_random_milp(seed);
        sor::engines::SimplexOptions so;
        so.presolve = false;
        sor::engines::SimplexDiagnostics sd;
        sor::engines::SimplexBasis basis;
        const auto raw = sor::engines::solve_simplex(lp, so, sd, &basis);
        if (raw.proposed_status != sor::core::Status::Optimal) continue;
        sor::search::CutOptions o;
        o.dynamism_max = dynamism_max;
        o.tableau_cmir = tableau_cmir;
        o.relax_small_terms = relax;
        o.violation_min = 1e-6;
        sor::search::CutDiagnostics d;
        const auto cuts = sor::search::separate_gomory_mi(lp, raw.x, basis, o, d);
        ++tot.models;
        tot.tableau += d.tableau_cmir_cuts;
        tot.repaired += d.dynamism_repaired;
        for (const auto& cut : cuts) {
            ++tot.cuts;
            // min of the cut's left-hand side over every feasible point.
            auto probe = lp;
            probe.maximize = false;
            probe.obj_offset = 0.0;
            probe.c.assign(static_cast<std::size_t>(lp.n_cols()), 0.0);
            for (std::size_t k = 0; k < cut.cols.size(); ++k)
                probe.c[static_cast<std::size_t>(cut.cols[k])] = cut.vals[k];
            const auto oracle = solve_oracle(probe);
            if (!oracle.feasible) continue;
            ++tot.checked;
            const bool ok = oracle.objective >= cut.row_lo - 1e-6 * (1.0 + std::fabs(cut.row_lo));
            if (!ok)
                std::cout << "INVALID " << label << " seed=" << seed << " " << cut.name
                          << " min " << oracle.objective << " < rhs " << cut.row_lo << '\n';
            CHECK(ok);
        }
    }
}

}  // namespace

int main() {
    Totals a, b, c;
    run("tableau", 1e2, true, true, a);
    run("tableau-tight-dynamism", 4.0, true, true, b);
    run("gmi-repair-only", 4.0, false, true, c);
    std::cout << "TABLEAU_CMIR models=" << a.models << " cuts=" << a.cuts
              << " checked=" << a.checked << " tableau=" << a.tableau
              << " | tight: cuts=" << b.cuts << " tableau=" << b.tableau
              << " repaired=" << b.repaired << " | gmi-repair: cuts=" << c.cuts
              << " repaired=" << c.repaired << '\n';
    // The test must actually reach the paths it is about.
    CHECK(a.tableau > 0);
    CHECK(b.repaired > 0 || c.repaired > 0);
    CHECK(a.checked > 100);
    return sor::test::finish("test_tableau_cmir");
}
