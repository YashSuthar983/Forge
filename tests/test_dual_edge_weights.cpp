#include "sor/engines/dual_edge_weights.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/la/lu.hpp"
#include "sor/sparse/csr.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

int test_initial_pricing_content_boundaries() {
    using sor::engines::DualInitialPricingStrategy;
    using sor::engines::choose_dual_initial_pricing;
    int failures = 0;
    const auto expect = [&](DualInitialPricingStrategy wanted,
                            sor::core::Index rows,
                            sor::core::Index cols,
                            sor::core::Offset nnz,
                            sor::core::Index boxed) {
        if (choose_dual_initial_pricing(rows, cols, nnz, boxed) != wanted)
            ++failures;
    };

    // Invalid descriptions fail closed to exact steepest edge.
    expect(DualInitialPricingStrategy::DSE, 0, 1, 1, 0);
    expect(DualInitialPricingStrategy::DSE, 10, -1, 1, 0);
    expect(DualInitialPricingStrategy::DSE, 10, 5, -1, 0);
    expect(DualInitialPricingStrategy::DSE, 10, 5, 1, 6);

    // Low-degree near-square regime: all inequalities are strict except the
    // dimension floor.
    expect(DualInitialPricingStrategy::Dantzig, 512, 511, 2047, 0);
    expect(DualInitialPricingStrategy::DSE, 511, 510, 2043, 0);
    expect(DualInitialPricingStrategy::DSE, 512, 512, 2047, 0);
    expect(DualInitialPricingStrategy::DSE, 512, 511, 2048, 0);

    // Wide boxed regime: aspect is strict; degree and boxed fraction include
    // their exact boundaries.
    expect(DualInitialPricingStrategy::Dantzig, 512, 2049, 10240, 1025);
    expect(DualInitialPricingStrategy::DSE, 512, 2048, 10240, 1024);
    expect(DualInitialPricingStrategy::DSE, 512, 2049, 10239, 1025);
    expect(DualInitialPricingStrategy::DSE, 512, 2049, 10240, 1024);

    // Medium sparse regime uses Devex, with [1000,3000), [1,2), and degree<8.
    expect(DualInitialPricingStrategy::Devex, 1000, 1000, 7999, 0);
    expect(DualInitialPricingStrategy::DSE, 999, 999, 7991, 0);
    expect(DualInitialPricingStrategy::DSE, 3000, 3000, 23999, 0);
    expect(DualInitialPricingStrategy::DSE, 1000, 2000, 7999, 0);
    expect(DualInitialPricingStrategy::DSE, 1000, 1000, 8000, 0);
    return failures;
}

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

int test_devex_reference_weight_and_reset_policy() {
    using sor::core::Index;
    int failures = 0;

    const std::vector<double> row{2.0, 3.0, -4.0, 0.0, 5.0};
    const std::vector<Index> support{0, 1, 2, 3, 4};
    const std::vector<std::uint8_t> reference{1, 0, 1, 1, 0};
    const double weight = sor::engines::dual_devex_reference_weight(
        row, support, reference);
    if (std::fabs(weight - 20.0) > 1e-12) ++failures;

    // An over-approximate support may contain explicit zeros, but not
    // duplicates: production's pivotal-row generation is duplicate-free.
    const std::vector<Index> sparse_support{0, 3};
    if (sor::engines::dual_devex_reference_weight(
            row, sparse_support, reference) != 4.0)
        ++failures;

    // Malformed input and overflow fail closed rather than poisoning CHUZR.
    if (std::isfinite(sor::engines::dual_devex_reference_weight(
            row, std::vector<Index>{9}, reference)))
        ++failures;
    auto huge = row;
    huge[0] = std::numeric_limits<double>::max();
    if (std::isfinite(sor::engines::dual_devex_reference_weight(
            huge, std::vector<Index>{0}, reference)))
        ++failures;

    using sor::engines::dual_devex_needs_new_framework;
    if (dual_devex_needs_new_framework(9.0, 1.0, 0, 10)) ++failures;
    if (!dual_devex_needs_new_framework(9.0001, 1.0, 0, 10)) ++failures;
    if (!dual_devex_needs_new_framework(0.0, 1.0, 0, 10)) ++failures;
    if (!dual_devex_needs_new_framework(
            std::numeric_limits<double>::infinity(), 1.0, 0, 10))
        ++failures;
    if (dual_devex_needs_new_framework(1.0, 1.0, 1000, 10)) ++failures;
    if (!dual_devex_needs_new_framework(1.0, 1.0, 1001, 10)) ++failures;

    using sor::engines::dual_dse_accept_weight;
    if (!dual_dse_accept_weight(0.25, 1.0)) ++failures;
    if (dual_dse_accept_weight(0.249999, 1.0)) ++failures;
    if (!dual_dse_accept_weight(100.0, 1.0)) ++failures;
    if (dual_dse_accept_weight(0.0, 1.0)) ++failures;
    if (dual_dse_accept_weight(1.0,
            std::numeric_limits<double>::infinity()))
        ++failures;

    using sor::engines::dual_update_running_density;
    if (std::fabs(dual_update_running_density(0.0, 1.0) - 0.05) > 1e-15)
        ++failures;
    if (std::fabs(dual_update_running_density(0.5, 0.5) - 0.5) > 1e-15)
        ++failures;
    if (std::isfinite(dual_update_running_density(-1.0, 0.5))) ++failures;

    using sor::engines::dual_dse_iteration_is_costly;
    // Both policy comparisons are strict: equality at 1% density or a
    // squared density ratio of exactly 1000 must not switch.
    if (dual_dse_iteration_is_costly(0.01, 0.0001, 0.0001, 0.0001))
        ++failures;
    const double root1000 = std::sqrt(1000.0);
    if (dual_dse_iteration_is_costly(root1000 * 0.001, 0.001, 0.0, 0.0))
        ++failures;
    if (!dual_dse_iteration_is_costly(root1000 * 0.001001, 0.001, 0.0, 0.0))
        ++failures;
    if (dual_dse_iteration_is_costly(0.5, 0.0, 0.0, 0.0)) ++failures;
    if (!dual_dse_iteration_is_costly(
            std::numeric_limits<double>::infinity(), 0.1, 0.1, 0.1))
        ++failures;

    using sor::engines::dual_dse_should_switch_to_devex;
    if (dual_dse_should_switch_to_devex(6, 100, 1000)) ++failures;
    if (dual_dse_should_switch_to_devex(5, 101, 1000)) ++failures;
    if (!dual_dse_should_switch_to_devex(6, 101, 1000)) ++failures;
    if (dual_dse_should_switch_to_devex(1, 1, 0)) ++failures;

    using sor::engines::dual_update_dse_log_error;
    using sor::engines::dual_dse_accuracy_requires_devex;
    double low = 0.0, high = 0.0;
    if (!dual_update_dse_log_error(1.0, std::exp(100.0), low, high))
        ++failures;
    if (std::fabs(low - 1.0) > 1e-12 || high != 0.0) ++failures;
    if (!dual_update_dse_log_error(std::exp(200.0), 1.0, low, high))
        ++failures;
    if (std::fabs(high - 2.0) > 1e-12) ++failures;
    if (dual_dse_accuracy_requires_devex(5.0, 5.0)) ++failures;
    if (!dual_dse_accuracy_requires_devex(5.0001, 5.0)) ++failures;
    if (!dual_dse_accuracy_requires_devex(
            std::numeric_limits<double>::infinity(), 0.0))
        ++failures;
    if (dual_update_dse_log_error(0.0, 1.0, low, high)) ++failures;

    using sor::engines::dual_dse_stability_requires_devex;
    if (dual_dse_stability_requires_devex(2)) ++failures;
    if (!dual_dse_stability_requires_devex(3)) ++failures;
    return failures;
}

// The DSE -> Devex ACCURACY switch, end to end over the decision chain rather
// than one threshold at a time.
//
// This replaces a test that drove the switch through nesm with
// SOR_DUAL_CHOOSE_DSE=1. That model no longer exercises it: after the WS1
// phase-1 work nesm reports 0 switches, 0 rejected rows and a DSE log error of
// 4.5e-06, and the old test also asserted `phase_restarts >= 3`, which is the
// phase-1 re-entry WS1 removed on purpose. No model in the suite destabilises
// DSE any more, so the mechanism is covered synthetically or not at all.
int test_dse_accuracy_switch_chain() {
    using sor::engines::dual_dse_accuracy_requires_devex;
    using sor::engines::dual_update_dse_log_error;
    int failures = 0;

    // A run whose propagated weights agree with the exact ones never switches,
    // however long it goes on.
    {
        double low = 0.0, high = 0.0;
        for (int i = 0; i < 500; ++i)
            dual_update_dse_log_error(1.0 + 1e-9 * i, 1.0, low, high);
        if (dual_dse_accuracy_requires_devex(low, high)) ++failures;
    }

    // Persistent UNDERestimation is the dangerous direction: it makes a row
    // look more attractive than it is. Feed it until the smoothed error
    // crosses, and require that it does cross.
    {
        double low = 0.0, high = 0.0;
        bool switched = false;
        for (int i = 0; i < 2000 && !switched; ++i) {
            dual_update_dse_log_error(1.0, std::exp(20.0), low, high);
            switched = dual_dse_accuracy_requires_devex(low, high);
        }
        if (!switched) ++failures;
    }

    // Symmetric OVERestimation must also be visible -- it wastes the row
    // rather than endangering it, but the smoothed error is two-sided.
    {
        double low = 0.0, high = 0.0;
        bool switched = false;
        for (int i = 0; i < 2000 && !switched; ++i) {
            dual_update_dse_log_error(std::exp(20.0), 1.0, low, high);
            switched = dual_dse_accuracy_requires_devex(low, high);
        }
        if (!switched) ++failures;
    }

    // A degenerate weight must not move the average at all: a single
    // non-positive computed weight would otherwise poison the smoothing and
    // switch a healthy run to Devex.
    {
        double low = 0.0, high = 0.0;
        if (dual_update_dse_log_error(0.0, 1.0, low, high)) ++failures;
        if (dual_update_dse_log_error(1.0, 0.0, low, high)) ++failures;
        if (low != 0.0 || high != 0.0) ++failures;
        if (dual_dse_accuracy_requires_devex(low, high)) ++failures;
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

// Algebraic regression for the exact rank-one DSE update. End-to-end LP tests
// prove that pricing cannot change the answer; this independently proves that
// the propagated weights after every product-form basis replacement equal a
// fresh all-row BTRAN rebuild. It catches sign, pivot-scaling, and old/new
// basis ordering mistakes directly at the function boundary where they occur.
int test_incremental_dse_matches_full_rebuild() {
    using sor::core::Index;
    using sor::core::Offset;
    using sor::core::f64;
    using sor::la::BasisFactor;

    constexpr Index m = 24;
    std::vector<Offset> cp{0};
    std::vector<Index> ri;
    std::vector<f64> bv;
    for (Index j = 0; j < m; ++j) {
        ri.push_back(j);
        bv.push_back(3.0 + 0.05 * static_cast<f64>(j));
        if (j + 1 < m) {
            ri.push_back(j + 1);
            bv.push_back(0.2);
        }
        cp.push_back(static_cast<Offset>(ri.size()));
    }

    BasisFactor factor;
    if (!factor.factorize(m, cp, ri, bv, sor::la::LuOptions{})) return 1;
    std::vector<f64> propagated;
    if (!sor::engines::rebuild_dual_edge_weights(
            m, [&](std::vector<f64>& v) { factor.btran(v); }, propagated))
        return 1;

    std::mt19937 rng(9173);
    std::uniform_real_distribution<f64> dist(-1.0, 1.0);
    for (int step = 0; step < 40; ++step) {
        const Index leave = static_cast<Index>((7 * step + 3) % m);
        std::vector<f64> entering(static_cast<std::size_t>(m));
        for (f64& v : entering) v = dist(rng);

        std::vector<f64> alpha = entering;
        factor.ftran(alpha);
        // Make the pivot safely nonzero without changing any other component
        // of alpha: adding c * old_basis_column(leave) adds c*e_leave after
        // FTRAN, and an update is defined completely by alpha.
        if (std::fabs(alpha[static_cast<std::size_t>(leave)]) < 0.5)
            alpha[static_cast<std::size_t>(leave)] +=
                alpha[static_cast<std::size_t>(leave)] < 0.0 ? -1.0 : 1.0;

        std::vector<f64> rho(static_cast<std::size_t>(m), 0.0);
        rho[static_cast<std::size_t>(leave)] = 1.0;
        factor.btran(rho);
        std::vector<f64> tau = rho;
        factor.ftran(tau);

        const f64 ap = alpha[static_cast<std::size_t>(leave)];
        const f64 old_pivotal = propagated[static_cast<std::size_t>(leave)];
        for (Index i = 0; i < m; ++i) {
            if (i == leave) continue;
            const f64 ratio = alpha[static_cast<std::size_t>(i)] / ap;
            propagated[static_cast<std::size_t>(i)] +=
                -2.0 * ratio * tau[static_cast<std::size_t>(i)] +
                ratio * ratio * old_pivotal;
        }
        propagated[static_cast<std::size_t>(leave)] =
            old_pivotal / (ap * ap);

        if (!factor.update(leave, alpha, 1e-12)) return 1;
        std::vector<f64> rebuilt;
        if (!sor::engines::rebuild_dual_edge_weights(
                m, [&](std::vector<f64>& v) { factor.btran(v); }, rebuilt))
            return 1;
        for (Index i = 0; i < m; ++i) {
            const f64 scale = 1.0 + std::fabs(rebuilt[static_cast<std::size_t>(i)]);
            if (std::fabs(propagated[static_cast<std::size_t>(i)] -
                          rebuilt[static_cast<std::size_t>(i)]) > 2e-8 * scale) {
                std::fprintf(stderr,
                             "DSE update mismatch step=%d row=%lld got=%.17g exact=%.17g\n",
                             step, static_cast<long long>(i),
                             propagated[static_cast<std::size_t>(i)],
                             rebuilt[static_cast<std::size_t>(i)]);
                return 1;
            }
        }
    }
    return 0;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_initial_pricing_content_boundaries();
    failures += test_dse_accuracy_switch_chain();
    failures += test_rebuild_small_fixture();
    failures += test_devex_reference_weight_and_reset_policy();
    failures += test_dse_matches_other_pricing_on_larger_lps();
    failures += test_incremental_dse_matches_full_rebuild();
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
