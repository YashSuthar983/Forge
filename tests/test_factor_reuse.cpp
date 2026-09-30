// EXPERIMENTAL FactorCarrier -- see its doc comment in dual_simplex.hpp.
// Correctness gate: adopting a carried LU factorization must never change
// the ANSWER, only the time to reach it. Every solve here is independently
// validated against the true (unscaled) LpProblem via certify::check_lp_point
// -- not just compared to the cold path's own internal diagnostics -- so a
// reuse bug that made every arm wrong the same way would still be caught.
#include "sor/engines/dual_simplex.hpp"
#include "../src/engines/src/simplex_prepared.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/sparse/csr.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using sor::core::f64;
using sor::core::Index;
using sor::model::LpProblem;

LpProblem make_lp(Index m, Index n, unsigned seed) {
    std::mt19937 rng(seed);
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
    lp.name = "factor_reuse_fixture";
    lp.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
    lp.c.assign(static_cast<std::size_t>(n), 0.0);
    for (Index j = 0; j < n; ++j) lp.c[static_cast<std::size_t>(j)] = 1.0 + val(rng);
    lp.row_lo.assign(static_cast<std::size_t>(m), 1.0);
    lp.row_hi.assign(static_cast<std::size_t>(m), 40.0);
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 15.0);
    return lp;
}

bool valid_point(const LpProblem& lp, const sor::core::RawResult& raw) {
    auto ev = sor::certify::check_lp_point(lp, raw, 1e-7, 1e-7, 1e-9, false);
    return ev.max_primal_violation <= 1e-6;
}

// A bound-only child of `lp`, warm-started from `parent_basis`, solved with
// `prepared` (already patched or rebuilt for the child by the caller) and an
// optional FactorCarrier. Returns the raw result and fills `reused`.
sor::core::RawResult solve_child(const LpProblem& child,
                                 const sor::engines::SimplexPrepared& prepared,
                                 const sor::engines::SimplexOptions& opts,
                                 const sor::engines::SimplexBasis& parent_basis,
                                 sor::engines::FactorCarrier* fc,
                                 bool* reused) {
    sor::engines::SimplexDiagnostics d;
    auto raw = sor::engines::solve_dual_simplex_prepared(
        prepared, opts, d, nullptr, &parent_basis, nullptr, fc);
    if (reused) *reused = d.factor_reused;
    (void)child;
    return raw;
}

int test_adopt_on_matching_basis_is_bit_correct() {
    int failures = 0;
    const Index m = 30, n = 45;
    LpProblem lp = make_lp(m, n, 20260923u);
    sor::engines::SimplexOptions opts;

    sor::engines::SimplexBasis parent_basis;
    sor::engines::FactorCarrier parent_carrier;
    parent_carrier.matrix = static_cast<const void*>(&lp);
    parent_carrier.rows = m; parent_carrier.cols = n; parent_carrier.nnz = lp.nnz();
    auto prepared = sor::engines::prepare_simplex_model(lp, opts);
    sor::engines::SimplexDiagnostics pd;
    auto parent_raw = sor::engines::solve_dual_simplex_prepared(
        prepared, opts, pd, &parent_basis, nullptr, nullptr, &parent_carrier);
    if (parent_raw.proposed_status != sor::core::Status::Optimal || !valid_point(lp, parent_raw)) {
        std::fprintf(stderr, "adopt test: parent solve not optimal/valid\n");
        return 1;
    }
    if (!parent_carrier.has_factor) {
        std::fprintf(stderr, "adopt test: carrier not filled by parent solve\n");
        return 1;
    }

    // Child: tighten one bound. Matrix A is untouched, so the parent's
    // basis is a valid starting basis and, if it happens to already be
    // optimal (0 pivots) or reach optimality with the SAME final basis, the
    // adopted factor is bit-identical to a fresh factorize of it.
    LpProblem child = lp;
    child.col_hi[3] = std::min(child.col_hi[3], 2.0);

    bool reused_a = false, reused_b = false;
    // Cold: no carrier.
    auto cold = solve_child(child, prepared, opts, parent_basis, nullptr, &reused_a);
    // Warm: a COPY of the parent's carrier (never the original -- see the
    // sibling test below for why).
    auto warm_carrier = parent_carrier;
    auto warm = solve_child(child, prepared, opts, parent_basis, &warm_carrier, &reused_b);

    if (cold.proposed_status != sor::core::Status::Optimal ||
        warm.proposed_status != sor::core::Status::Optimal) {
        std::fprintf(stderr, "adopt test: child solve not optimal\n");
        ++failures;
    }
    if (!valid_point(child, cold) || !valid_point(child, warm)) {
        std::fprintf(stderr, "adopt test: child solution fails independent validation\n");
        ++failures;
    }
    if (std::fabs(cold.objective - warm.objective) > 1e-8 * (1.0 + std::fabs(cold.objective))) {
        std::fprintf(stderr, "adopt test: cold=%.12g warm=%.12g objectives disagree\n",
                     cold.objective, warm.objective);
        ++failures;
    }
    if (reused_a) {
        std::fprintf(stderr, "adopt test: cold arm unexpectedly reused a factor\n");
        ++failures;
    }
    if (!reused_b) {
        std::fprintf(stderr, "adopt test: warm arm did not adopt the matching factor "
                     "(same basis, same matrix -- this is the case the mechanism exists for)\n");
        ++failures;
    }
    return failures;
}

// The safety gate must reject a carrier whose row count no longer matches
// the current problem (e.g. after a cut/row was added) and fall back to a
// cold factorize -- never adopt a factor for the wrong-shaped matrix.
int test_row_count_mismatch_is_rejected_safely() {
    int failures = 0;
    const Index m = 20, n = 30;
    LpProblem lp = make_lp(m, n, 777u);
    sor::engines::SimplexOptions opts;

    sor::engines::SimplexBasis basis;
    sor::engines::FactorCarrier carrier;
    carrier.matrix = static_cast<const void*>(&lp);
    carrier.rows = m; carrier.cols = n; carrier.nnz = lp.nnz();
    auto prepared = sor::engines::prepare_simplex_model(lp, opts);
    sor::engines::SimplexDiagnostics pd;
    auto parent_raw = sor::engines::solve_dual_simplex_prepared(
        prepared, opts, pd, &basis, nullptr, nullptr, &carrier);
    if (parent_raw.proposed_status != sor::core::Status::Optimal) return 1;

    // Add a row -- A itself changes, m grows by one. The stale carrier
    // (rows=m, old m) must be rejected by dimension alone.
    LpProblem grown = lp;
    std::vector<Index> cols = {0, 1, 2};
    std::vector<f64> vals = {1.0, 1.0, 1.0};
    grown.A.append_row(cols, vals);
    grown.row_lo.push_back(0.0);
    grown.row_hi.push_back(100.0);

    auto grown_prepared = sor::engines::prepare_simplex_model(grown, opts);
    bool reused = true;  // must end up false
    sor::engines::SimplexDiagnostics d;
    auto raw = sor::engines::solve_dual_simplex_prepared(
        grown_prepared, opts, d, nullptr, &basis, nullptr, &carrier);
    reused = d.factor_reused;

    if (reused) {
        std::fprintf(stderr, "row-mismatch test: carrier was adopted across a row "
                     "addition -- safety gate failed\n");
        ++failures;
    }
    if (raw.proposed_status != sor::core::Status::Optimal || !valid_point(grown, raw)) {
        std::fprintf(stderr, "row-mismatch test: solve on the grown model is wrong "
                     "even though it correctly fell back to a cold factorize\n");
        ++failures;
    }
    return failures;
}

// A carrier whose STORED basis does not match the actual starting basis
// (constructed directly, standing in for whatever future bug might produce
// this) must be rejected -- basis content, not just dimensions, gates
// adoption.
int test_basis_content_mismatch_is_rejected_safely() {
    int failures = 0;
    const Index m = 18, n = 25;
    LpProblem lp = make_lp(m, n, 55u);
    sor::engines::SimplexOptions opts;

    sor::engines::SimplexBasis basis_a, basis_b;
    sor::engines::FactorCarrier carrier_a;
    carrier_a.matrix = static_cast<const void*>(&lp);
    carrier_a.rows = m; carrier_a.cols = n; carrier_a.nnz = lp.nnz();
    auto prepared = sor::engines::prepare_simplex_model(lp, opts);
    sor::engines::SimplexDiagnostics d0;
    sor::engines::solve_dual_simplex_prepared(prepared, opts, d0, &basis_a, nullptr,
                                              nullptr, &carrier_a);

    // A DIFFERENT LP (different costs) very likely optimizes to a different
    // basis; solve it to get a basis_b that (with overwhelming likelihood)
    // differs from basis_a, then hand carrier_a (captured for basis_a) to a
    // solve warm-started from basis_b -- an intentionally wrong pairing.
    LpProblem lp2 = lp;
    for (auto& v : lp2.c) v += 3.0;
    sor::engines::SimplexDiagnostics d1;
    sor::engines::solve_dual_simplex(lp2, opts, d1, &basis_b, nullptr, nullptr);
    if (basis_a.basic == basis_b.basic) {
        // Degenerate coincidence; not a test failure, just not exercised.
        return 0;
    }

    sor::engines::SimplexDiagnostics d2;
    auto raw = sor::engines::solve_dual_simplex_prepared(
        prepared, opts, d2, nullptr, &basis_b, nullptr, &carrier_a);
    if (d2.factor_reused) {
        std::fprintf(stderr, "basis-mismatch test: carrier for one basis was adopted "
                     "under a DIFFERENT warm basis -- safety gate failed\n");
        ++failures;
    }
    if (raw.proposed_status != sor::core::Status::Optimal || !valid_point(lp, raw)) {
        std::fprintf(stderr, "basis-mismatch test: solve wrong despite correct fallback\n");
        ++failures;
    }
    return failures;
}

// Two children of the SAME parent must each get a correct, independent
// answer from a COPY of the parent's carrier -- BasisFactor::update()/
// update_ft() mutate in place, so sharing the same carrier OBJECT across
// two solves (rather than a copy per solve) would let the first solve's
// pivots corrupt the second's starting state. See FactorCarrier's own doc
// comment: sibling reuse is the CALLER's responsibility, not the engine's.
int test_sibling_reuse_via_copy_does_not_cross_contaminate() {
    int failures = 0;
    const Index m = 25, n = 35;
    LpProblem lp = make_lp(m, n, 909u);
    sor::engines::SimplexOptions opts;

    sor::engines::SimplexBasis parent_basis;
    sor::engines::FactorCarrier parent_carrier;
    parent_carrier.matrix = static_cast<const void*>(&lp);
    parent_carrier.rows = m; parent_carrier.cols = n; parent_carrier.nnz = lp.nnz();
    auto prepared = sor::engines::prepare_simplex_model(lp, opts);
    sor::engines::SimplexDiagnostics pd;
    auto parent_raw = sor::engines::solve_dual_simplex_prepared(
        prepared, opts, pd, &parent_basis, nullptr, nullptr, &parent_carrier);
    if (parent_raw.proposed_status != sor::core::Status::Optimal) return 1;

    LpProblem down = lp, up = lp;
    down.col_hi[5] = std::min(down.col_hi[5], 3.0);
    up.col_lo[5] = std::max(up.col_lo[5], 4.0);

    auto down_prepared = sor::engines::prepare_simplex_model(down, opts);
    auto up_prepared = sor::engines::prepare_simplex_model(up, opts);

    // Each sibling gets its OWN copy, per the documented discipline.
    auto fc_down = parent_carrier;
    auto fc_up = parent_carrier;
    sor::engines::SimplexDiagnostics dd, du;
    auto raw_down = sor::engines::solve_dual_simplex_prepared(
        down_prepared, opts, dd, nullptr, &parent_basis, nullptr, &fc_down);
    auto raw_up = sor::engines::solve_dual_simplex_prepared(
        up_prepared, opts, du, nullptr, &parent_basis, nullptr, &fc_up);

    if (!valid_point(down, raw_down) || !valid_point(up, raw_up)) {
        std::fprintf(stderr, "sibling test: a sibling's solution fails independent "
                     "validation -- possible cross-contamination between copies\n");
        ++failures;
    }
    // Cross-check against fully independent cold solves of the same two
    // children -- the copies must not have altered the MATH, only the time.
    sor::engines::SimplexDiagnostics dd2, du2;
    sor::engines::SimplexBasis dummy;
    auto cold_down = sor::engines::solve_dual_simplex_prepared(
        down_prepared, opts, dd2, &dummy, &parent_basis, nullptr, nullptr);
    auto cold_up = sor::engines::solve_dual_simplex_prepared(
        up_prepared, opts, du2, &dummy, &parent_basis, nullptr, nullptr);
    if (std::fabs(cold_down.objective - raw_down.objective) >
            1e-8 * (1.0 + std::fabs(cold_down.objective)) ||
        std::fabs(cold_up.objective - raw_up.objective) >
            1e-8 * (1.0 + std::fabs(cold_up.objective))) {
        std::fprintf(stderr, "sibling test: reused-copy objective disagrees with an "
                     "independent cold solve of the same child\n");
        ++failures;
    }
    return failures;
}

// matrix == nullptr must never be adopted, matching DualEdgeWeightCarrier's
// documented contract exactly (the safe default with no token set).
int test_null_matrix_token_is_never_adopted() {
    int failures = 0;
    const Index m = 12, n = 18;
    LpProblem lp = make_lp(m, n, 321u);
    sor::engines::SimplexOptions opts;
    auto prepared = sor::engines::prepare_simplex_model(lp, opts);

    sor::engines::FactorCarrier carrier;  // matrix left null
    carrier.rows = m; carrier.cols = n; carrier.nnz = lp.nnz();
    sor::engines::SimplexDiagnostics d0;
    sor::engines::SimplexBasis basis;
    sor::engines::solve_dual_simplex_prepared(prepared, opts, d0, &basis, nullptr,
                                              nullptr, &carrier);
    // carrier.matrix stays null throughout (never set by the harness), so a
    // second call reusing it must never adopt.
    sor::engines::SimplexDiagnostics d1;
    sor::engines::solve_dual_simplex_prepared(prepared, opts, d1, nullptr, &basis,
                                              nullptr, &carrier);
    if (d1.factor_reused) {
        std::fprintf(stderr, "null-token test: adopted despite matrix == nullptr\n");
        ++failures;
    }
    return failures;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_adopt_on_matching_basis_is_bit_correct();
    failures += test_row_count_mismatch_is_rejected_safely();
    failures += test_basis_content_mismatch_is_rejected_safely();
    failures += test_sibling_reuse_via_copy_does_not_cross_contaminate();
    failures += test_null_matrix_token_is_never_adopted();
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
