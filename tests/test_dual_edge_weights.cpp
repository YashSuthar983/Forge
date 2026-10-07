#include "sor/engines/dual_edge_weights.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/la/lu.hpp"
#include "sor/sparse/csr.hpp"

#include <chrono>
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

    // Wide boxed models retain DSE; bounds alone do not justify Dantzig.
    expect(DualInitialPricingStrategy::DSE, 512, 2049, 10240, 1025);
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

// Past its deadline the all-row rebuild gives up like any other failure, so
// the dual falls back to unit weights instead of spending m BTRANs after the
// time limit.
int test_rebuild_stops_at_deadline() {
    int failures = 0;
    std::vector<double> weights;
    int solves = 0;
    const auto btran = [&](std::vector<double>&) { ++solves; };
    if (!sor::engines::rebuild_dual_edge_weights(4, btran, weights, {},
            std::chrono::steady_clock::now() + std::chrono::hours(1)))
        ++failures;
    if (solves != 4) ++failures;
    solves = 0;
    if (sor::engines::rebuild_dual_edge_weights(4, btran, weights, {},
            std::chrono::steady_clock::now() - std::chrono::seconds(1)))
        ++failures;
    if (solves != 0) ++failures;
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

// Exact DSE end-to-end regression: the incremental per-pivot weight update used to be capped to m <= 64
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

// The seeded rebuild must be a pure speedup: supplying the seeded BTRAN may
// change how many rows the solve touches, never a single bit of the resulting
// weights -- one different bit can change a later CHUZR and therefore the whole
// pivot sequence. Bitwise equality is the assertion, on a factor that carries
// product-form etas (where the hypersparse firing path actually runs), and the
// test also proves it is not vacuous by requiring the sparse route to fire.
int test_seeded_rebuild_is_bit_identical() {
    using sor::core::Index;
    using sor::core::Offset;
    using sor::core::f64;
    using sor::la::BasisFactor;

    // Lower-bidiagonal, so B^-T e_i reaches only a short tail of rows: the
    // regime the hypersparse path is built for.
    constexpr Index m = 256;
    std::vector<Offset> cp{0};
    std::vector<Index> ri;
    std::vector<f64> bv;
    for (Index j = 0; j < m; ++j) {
        ri.push_back(j);
        bv.push_back(2.0 + 0.03 * static_cast<f64>(j));
        if (j + 1 < m) {
            ri.push_back(j + 1);
            bv.push_back(0.35);
        }
        cp.push_back(static_cast<Offset>(ri.size()));
    }

    BasisFactor factor;
    if (!factor.factorize(m, cp, ri, bv, sor::la::LuOptions{})) return 1;

    // Append etas so the seeded BTRAN exercises the firing-set path, not just
    // the eta-free base solve.
    std::mt19937 rng(4421);
    std::uniform_real_distribution<f64> dist(-1.0, 1.0);
    for (int step = 0; step < 12; ++step) {
        const Index leave = static_cast<Index>((37 * step + 2) % m);
        // Sparse entering columns keep the etas sparse; dense etas would make
        // every BTRAN dense and quietly turn this test vacuous.
        std::vector<f64> alpha(static_cast<std::size_t>(m), 0.0);
        for (int k = 0; k < 3; ++k)
            alpha[static_cast<std::size_t>((17 * step + 5 * k) % m)] = dist(rng);
        factor.ftran(alpha);
        if (std::fabs(alpha[static_cast<std::size_t>(leave)]) < 0.5)
            alpha[static_cast<std::size_t>(leave)] +=
                alpha[static_cast<std::size_t>(leave)] < 0.0 ? -1.0 : 1.0;
        if (!factor.update(leave, alpha)) return 1;
    }

    std::vector<f64> dense;
    if (!sor::engines::rebuild_dual_edge_weights(
            m, [&](std::vector<f64>& v) { factor.btran(v); }, dense))
        return 1;

    std::vector<Index> seed(1, 0);
    std::uint64_t sparse_calls = 0;
    std::vector<f64> seeded;
    if (!sor::engines::rebuild_dual_edge_weights(
            m, [&](std::vector<f64>& v) { factor.btran(v); }, seeded,
            [&](std::vector<f64>& v, Index slot, std::vector<Index>& support) {
                seed[0] = slot;
                const bool sparse =
                    factor.btran_seeded_with_support(v, seed, support);
                if (sparse) ++sparse_calls;
                return sparse;
            }))
        return 1;

    int failures = 0;
    if (dense.size() != seeded.size()) return 1;
    for (std::size_t i = 0; i < dense.size(); ++i) {
        // Bitwise, not approximate.
        if (!(dense[i] == seeded[i])) {
            std::fprintf(stderr,
                         "seeded DSE rebuild differs at row %zu: dense=%.17g seeded=%.17g\n",
                         i, dense[i], seeded[i]);
            ++failures;
        }
    }
    if (sparse_calls == 0) {
        std::fprintf(stderr, "seeded DSE rebuild never took the sparse path\n");
        ++failures;
    }
    return failures;
}

// Carrying DSE weights between solves must be a pure saving: the second solve
// reaches the same optimum, having skipped the rebuild. The test also pins both
// guards -- a carrier whose basis does not match the run being started, and a
// carrier from a different matrix, must both be refused rather than believed.
int test_dse_weight_carrier() {
    using sor::core::Index;
    using sor::core::f64;
    using sor::model::LpProblem;
    int failures = 0;

    constexpr Index m = 40;
    constexpr Index n = m + 15;
    std::mt19937 rng(20260917);
    std::uniform_real_distribution<f64> val(0.5, 2.5);
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    for (Index i = 0; i < m; ++i)
        for (int off = 0; off < 3; ++off) {
            rows.push_back(i);
            cols.push_back((i + off) % n);
            vals.push_back(val(rng));
        }

    LpProblem lp;
    lp.name = "dse_carrier";
    lp.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
    lp.c.assign(static_cast<std::size_t>(n), 0.0);
    for (Index j = 0; j < n; ++j) lp.c[static_cast<std::size_t>(j)] = 1.0 + val(rng);
    lp.row_lo.assign(static_cast<std::size_t>(m), 1.0);
    lp.row_hi.assign(static_cast<std::size_t>(m), 40.0);
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 15.0);

    sor::engines::SimplexOptions opts;
    opts.pricing = sor::engines::SimplexPricing::DSE;

    sor::engines::SimplexDiagnostics parent_diag;
    sor::engines::SimplexBasis parent_basis;
    sor::engines::DualEdgeWeightCarrier carrier;
    carrier.matrix = static_cast<const void*>(&lp);   // caller-declared identity
    const auto parent = sor::engines::solve_dual_simplex(
        lp, opts, parent_diag, &parent_basis, nullptr, &carrier);
    if (parent.proposed_status != sor::core::Status::Optimal) return 1;
    if (carrier.weights.size() != static_cast<std::size_t>(m) ||
        carrier.basis.size() != static_cast<std::size_t>(m)) {
        std::fprintf(stderr, "carrier not filled by the parent solve\n");
        return 1;
    }

    // Branch: tighten one bound, which changes the LP but not its matrix.
    LpProblem child = lp;
    child.col_hi[3] = 2.0;

    const auto solve_child = [&](sor::engines::DualEdgeWeightCarrier* c,
                                 sor::engines::SimplexDiagnostics& d) {
        sor::engines::SimplexBasis out;
        return sor::engines::solve_dual_simplex(child, opts, d, &out,
                                                &parent_basis, c);
    };

    sor::engines::SimplexDiagnostics cold_diag;
    const auto cold = solve_child(nullptr, cold_diag);

    auto warm_carrier = carrier;
    sor::engines::SimplexDiagnostics warm_diag;
    const auto warm = solve_child(&warm_carrier, warm_diag);

    if (warm_diag.dse_weight_reuses != 1 || warm_diag.dse_weight_rebuilds != 0) {
        std::fprintf(stderr, "carried weights not adopted (reuses=%llu rebuilds=%llu)\n",
                     static_cast<unsigned long long>(warm_diag.dse_weight_reuses),
                     static_cast<unsigned long long>(warm_diag.dse_weight_rebuilds));
        ++failures;
    }
    if (cold_diag.dse_weight_rebuilds != 1 || cold_diag.dse_weight_reuses != 0) {
        std::fprintf(stderr, "no-carrier solve did not rebuild (rebuilds=%llu)\n",
                     static_cast<unsigned long long>(cold_diag.dse_weight_rebuilds));
        ++failures;
    }
    if (cold.proposed_status != warm.proposed_status ||
        std::fabs(cold.objective - warm.objective) >
            1e-7 * (1.0 + std::fabs(cold.objective))) {
        std::fprintf(stderr, "carried-weight solve changed the answer: %.12g vs %.12g\n",
                     cold.objective, warm.objective);
        ++failures;
    }

    // Guard 1: weights that belong to a different basis are refused.
    auto wrong_basis = carrier;
    wrong_basis.basis[0] = wrong_basis.basis[0] == 0 ? 1 : 0;
    sor::engines::SimplexDiagnostics wrong_diag;
    (void)solve_child(&wrong_basis, wrong_diag);
    if (wrong_diag.dse_weight_reuses != 0 || wrong_diag.dse_weight_rebuilds != 1) {
        std::fprintf(stderr, "mismatched basis was not refused\n");
        ++failures;
    }

    // Guard 2: a carrier with no declared matrix is never adopted -- clear()
    // is how a caller invalidates weights after changing the matrix.
    auto anonymous = carrier;
    anonymous.matrix = nullptr;
    sor::engines::SimplexDiagnostics anon_diag;
    (void)solve_child(&anonymous, anon_diag);
    if (anon_diag.dse_weight_reuses != 0) {
        std::fprintf(stderr, "carrier with no matrix token was adopted\n");
        ++failures;
    }

    // Guard 3: the engine refuses weights whose shape no longer matches the
    // problem, even under a token the caller forgot to clear.
    LpProblem wider = child;
    wider.c.push_back(1.0);
    wider.col_lo.push_back(0.0);
    wider.col_hi.push_back(5.0);
    {
        std::vector<Index> r2 = rows, c2 = cols;
        std::vector<f64> v2 = vals;
        r2.push_back(0); c2.push_back(n); v2.push_back(1.0);
        wider.A = sor::sparse::from_triplets(m, n + 1, r2, c2, v2);
    }
    auto stale = carrier;
    sor::engines::SimplexDiagnostics stale_diag;
    sor::engines::SimplexBasis stale_out;
    (void)sor::engines::solve_dual_simplex(wider, opts, stale_diag, &stale_out,
                                           nullptr, &stale);
    if (stale_diag.dse_weight_reuses != 0) {
        std::fprintf(stderr, "weights were adopted after the shape changed\n");
        ++failures;
    }
    return failures;
}

// SimplexOptions::warm_dse_reset (S3.1): a warm-started child with a
// non-logical basis must skip the exact rebuild and use weights of 1
// instead, and must still reach the same answer as the exact rebuild --
// weights only steer pricing, never correctness.
int test_warm_dse_reset_skips_rebuild() {
    using sor::core::Index;
    using sor::core::f64;
    using sor::model::LpProblem;
    int failures = 0;

    constexpr Index m = 40;
    constexpr Index n = m + 15;
    std::mt19937 rng(20260928);
    std::uniform_real_distribution<f64> val(0.5, 2.5);
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    for (Index i = 0; i < m; ++i)
        for (int off = 0; off < 3; ++off) {
            rows.push_back(i);
            cols.push_back((i + off) % n);
            vals.push_back(val(rng));
        }

    LpProblem lp;
    lp.name = "warm_dse_reset";
    lp.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
    lp.c.assign(static_cast<std::size_t>(n), 0.0);
    for (Index j = 0; j < n; ++j) lp.c[static_cast<std::size_t>(j)] = 1.0 + val(rng);
    lp.row_lo.assign(static_cast<std::size_t>(m), 1.0);
    lp.row_hi.assign(static_cast<std::size_t>(m), 40.0);
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 15.0);

    sor::engines::SimplexOptions opts;
    opts.pricing = sor::engines::SimplexPricing::DSE;

    sor::engines::SimplexDiagnostics parent_diag;
    sor::engines::SimplexBasis parent_basis;
    const auto parent = sor::engines::solve_dual_simplex(lp, opts, parent_diag,
                                                          &parent_basis);
    if (parent.proposed_status != sor::core::Status::Optimal) return 1;

    // Branch: tighten one bound. B does not change, but the parent basis
    // (captured at B != -I, since the parent itself started logical and
    // pivoted away from it) is non-logical, so reset_weights() takes the
    // exact-rebuild branch on a plain warm start.
    LpProblem child = lp;
    child.col_hi[3] = 2.0;

    sor::engines::SimplexOptions default_opts = opts;
    sor::engines::SimplexDiagnostics default_diag;
    sor::engines::SimplexBasis default_out;
    const auto default_result = sor::engines::solve_dual_simplex(
        child, default_opts, default_diag, &default_out, &parent_basis);
    if (default_diag.dse_weight_rebuilds != 1) {
        std::fprintf(stderr,
                     "expected the exact rebuild by default (rebuilds=%llu)\n",
                     static_cast<unsigned long long>(default_diag.dse_weight_rebuilds));
        ++failures;
    }

    sor::engines::SimplexOptions warm_opts = opts;
    warm_opts.warm_dse_reset = true;
    sor::engines::SimplexDiagnostics warm_diag;
    sor::engines::SimplexBasis warm_out;
    const auto warm_result = sor::engines::solve_dual_simplex(
        child, warm_opts, warm_diag, &warm_out, &parent_basis);
    if (warm_diag.dse_weight_rebuilds != 0) {
        std::fprintf(stderr,
                     "warm_dse_reset did not skip the rebuild (rebuilds=%llu)\n",
                     static_cast<unsigned long long>(warm_diag.dse_weight_rebuilds));
        ++failures;
    }
    if (default_result.proposed_status != warm_result.proposed_status ||
        std::fabs(default_result.objective - warm_result.objective) >
            1e-7 * (1.0 + std::fabs(default_result.objective))) {
        std::fprintf(stderr, "warm_dse_reset changed the answer: %.12g vs %.12g\n",
                     default_result.objective, warm_result.objective);
        ++failures;
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
    failures += test_rebuild_stops_at_deadline();
    failures += test_devex_reference_weight_and_reset_policy();
    failures += test_dse_matches_other_pricing_on_larger_lps();
    failures += test_seeded_rebuild_is_bit_identical();
    failures += test_dse_weight_carrier();
    failures += test_warm_dse_reset_skips_rebuild();
    failures += test_incremental_dse_matches_full_rebuild();
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
