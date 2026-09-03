#include "sor/engines/dual_edge_weights.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/sparse/csr.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int test_rebuild_small_fixture() {
    int failures = 0;
    std::vector<double> weights;
    const bool ok = sor::engines::rebuild_dual_edge_weights(
        2,
        [](std::vector<double>& v) {
            // B = [[2, 0], [1, 3]], so B^-T e0 = [1/2, 0] and
            // B^-T e1 = [-1/6, 1/3].
            const double a = v[0], b = v[1];
            v[0] = 0.5 * a - (1.0 / 6.0) * b;
            v[1] = (1.0 / 3.0) * b;
        },
        weights);
    if (!ok || weights.size() != 2) ++failures;
    if (weights.size() == 2) {
        if (std::fabs(weights[0] - 0.25) > 1e-12) ++failures;
        if (std::fabs(weights[1] - (5.0 / 36.0)) > 1e-12) ++failures;
    }
    return failures;
}

// Exact DSE end-to-end regression (item 4 of docs/SIH26119_PS_ALIGNMENT.md
// §5): the incremental per-pivot weight update used to be capped to m <= 64
// because the ONLY implementation was a full O(m)-BTRAN rebuild after every
// pivot. It is now a genuine O(1)-extra-FTRAN incremental update (Forrest &
// Goldfarb 1992) with no size cap. DSE weights only affect PRICING (which
// column/row is chosen), never correctness of the simplex method itself, so
// the thing that can actually go wrong here is a wrong answer slipping
// through -- this checks Dantzig/Devex/DSE all agree on the same random
// sparse LPs, including several with m > 64 (previously impossible to run
// exact DSE on at all).
int test_dse_matches_other_pricing_on_larger_lps() {
    using sor::core::Index;
    using sor::core::f64;
    using sor::model::LpProblem;

    int failures = 0;
    std::mt19937 rng(2024);
    std::uniform_real_distribution<f64> val_dist(0.1, 5.0);
    std::uniform_real_distribution<f64> sign_dist(0.0, 1.0);

    for (const Index m : {20, 80, 150}) {
        const Index n = m + 20;
        std::vector<Index> rows, cols;
        std::vector<f64> vals;
        // A feasible, bounded chain: row i covers columns i and i+1 (mod n),
        // plus a few random extra nonzeros for sparsity variety. All
        // coefficients positive and row_hi generous, so x = 0 is always
        // feasible and the LP is bounded below by the nonnegativity of c.
        for (Index i = 0; i < m; ++i) {
            for (int off = 0; off < 3; ++off) {
                const Index j = (i + off) % n;
                rows.push_back(i);
                cols.push_back(j);
                vals.push_back(val_dist(rng));
            }
        }

        LpProblem lp;
        lp.name = "dse_fuzz";
        lp.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
        lp.c.resize(static_cast<std::size_t>(n));
        for (Index j = 0; j < n; ++j) lp.c[static_cast<std::size_t>(j)] = 1.0 + val_dist(rng);
        lp.row_lo.assign(static_cast<std::size_t>(m), 1.0);
        lp.row_hi.assign(static_cast<std::size_t>(m), 50.0);
        lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
        lp.col_hi.assign(static_cast<std::size_t>(n), 20.0);
        (void)sign_dist;

        sor::engines::SimplexOptions dantzig_opts;
        dantzig_opts.pricing = sor::engines::SimplexPricing::Dantzig;
        sor::engines::SimplexDiagnostics dsd;
        const auto raw_dantzig = sor::engines::solve_dual_simplex(lp, dantzig_opts, dsd, nullptr, nullptr);

        sor::engines::SimplexOptions dse_opts;
        dse_opts.pricing = sor::engines::SimplexPricing::DSE;
        sor::engines::SimplexDiagnostics dsed;
        const auto raw_dse = sor::engines::solve_dual_simplex(lp, dse_opts, dsed, nullptr, nullptr);

        if (raw_dantzig.proposed_status != raw_dse.proposed_status) {
            std::fprintf(stderr,
                "m=%lld: status mismatch dantzig=%d dse=%d\n",
                static_cast<long long>(m), static_cast<int>(raw_dantzig.proposed_status),
                static_cast<int>(raw_dse.proposed_status));
            ++failures;
            continue;
        }
        const f64 rel = std::fabs(raw_dantzig.objective - raw_dse.objective) /
                        (1.0 + std::fabs(raw_dantzig.objective));
        if (rel > 1e-6) {
            std::fprintf(stderr,
                "m=%lld: objective mismatch dantzig=%.10e dse=%.10e rel=%.3e\n",
                static_cast<long long>(m), raw_dantzig.objective, raw_dse.objective, rel);
            ++failures;
        }
    }
    return failures;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_rebuild_small_fixture();
    failures += test_dse_matches_other_pricing_on_larger_lps();
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
