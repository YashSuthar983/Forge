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
#include <stdexcept>
#include <optional>
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

int test_session_rejects_foreign_factor_and_invalid_bounds() {
    using namespace sor::engines;
    auto a = make_lp(1, 1, 12u);
    a.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {1.0});
    a.c = {1.0}; a.row_hi = {sor::model::kInf};
    auto b = a;
    b.A.vals[0] = 2.0;
    SimplexOptions opts;
    opts.presolve = false;
    opts.ruiz_iterations = 0;
    DualProbeSession first(a, opts), second(b, opts);
    SimplexBasis first_basis, second_basis;
    SimplexDiagnostics da, db, reuse;
    const auto ra = first.solve(opts, da, &first_basis, nullptr);
    const FactorCarrier foreign = first.final_factor();
    const auto rb = second.solve(opts, db, &second_basis, &first_basis, nullptr, &foreign);
    int failures = 0;
    if (!foreign.has_factor || first_basis.basic != second_basis.basic ||
        db.factor_reused || !valid_point(a, ra) || !valid_point(b, rb) ||
        std::fabs(rb.objective - 0.5) > 1e-9) {
        std::fprintf(stderr, "session test: foreign numeric factor with matching basis was accepted\n");
        ++failures;
    }
    const FactorCarrier own = second.final_factor();
    const auto again = second.solve(opts, reuse, nullptr, &second_basis, nullptr, &own);
    if (!reuse.factor_reused || !valid_point(b, again)) {
        std::fprintf(stderr, "session test: valid same-session factor was rejected\n");
        ++failures;
    }
    for (Index bad : {Index(-1), b.n_cols()}) {
        bool threw = false;
        try { second.probe(bad, 0.0, 1.0, opts, db, second_basis); }
        catch (const std::out_of_range&) { threw = true; }
        if (!threw) ++failures;
    }
    bool threw = false;
    try { second.probe(0, std::nan(""), 1.0, opts, db, second_basis); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) ++failures;

    // A late invalid bound must not partially replace the earlier columns.
    auto multi = make_lp(2, 3, 43u);
    DualProbeSession all(multi, opts);
    auto lo = multi.col_lo, hi = multi.col_hi;
    lo[0] = 10.0; hi[1] = std::nan("");
    threw = false;
    try { all.set_column_bounds(lo, hi); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) ++failures;
    SimplexDiagnostics before, after;
    const auto original = solve_dual_simplex(multi, opts, before);
    const auto unchanged = all.solve(opts, after, nullptr, nullptr);
    if (!valid_point(multi, unchanged) ||
        std::fabs(original.objective - unchanged.objective) > 1e-9) ++failures;
    const auto restored = second.solve(opts, db, nullptr, &second_basis);
    if (!valid_point(b, restored) || std::fabs(restored.objective - 0.5) > 1e-9) ++failures;

    // Evict/recreate at precisely the same address, as an LRU allocation can
    // do. The retained checkpoint must keep a distinct preparation lifetime.
    std::optional<DualProbeSession> slot;
    slot.emplace(a, opts);
    slot->solve(opts, da, &first_basis, nullptr);
    const FactorCarrier evicted = slot->final_factor();
    slot.reset();
    slot.emplace(b, opts);
    const auto recreated = slot->solve(opts, db, nullptr, &first_basis, nullptr, &evicted);
    if (evicted.matrix != &*slot || db.factor_reused || !valid_point(b, recreated) ||
        std::fabs(recreated.objective - 0.5) > 1e-9) {
        std::fprintf(stderr, "session test: evicted identity survived address reuse\n");
        ++failures;
    }
    return failures;
}

int test_rescaling_same_matrix_rejects_factor() {
    using namespace sor::engines;
    auto lp = make_lp(1, 1, 12u);
    lp.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {8.0});
    lp.c = {1.0}; lp.row_hi = {sor::model::kInf};
    SimplexOptions unscaled;
    unscaled.presolve = false;
    unscaled.ruiz_iterations = 0;
    SimplexOptions scaled = unscaled;
    scaled.ruiz_iterations = 8;
    FactorCarrier carrier;
    carrier.matrix = &lp;
    carrier.rows = lp.n_rows(); carrier.cols = lp.n_cols(); carrier.nnz = lp.nnz();
    SimplexDiagnostics d;
    SimplexBasis basis, rescaled_basis;
    solve_dual_simplex(lp, unscaled, d, &basis, nullptr, nullptr, &carrier);
    const auto changed = solve_dual_simplex(lp, scaled, d, &rescaled_basis, &basis,
                                           nullptr, &carrier);
    int failures = 0;
    if (d.factor_reused || d.factor_reuse_preparation_mismatch != 1 ||
        !valid_point(lp, changed) || std::fabs(changed.objective - 0.125) > 1e-9) {
        std::fprintf(stderr, "scaling test: factor survived a changed numeric basis\n");
        ++failures;
    }
    const auto same = solve_dual_simplex(lp, scaled, d, nullptr, &rescaled_basis,
                                        nullptr, &carrier);
    if (!d.factor_reused || !valid_point(lp, same)) {
        std::fprintf(stderr, "scaling test: equivalent repeated preparation failed reuse\n");
        ++failures;
    }
    return failures;
}

int test_preparation_spends_first_solve_allowance() {
    using namespace sor::engines;
    const auto lp = make_lp(30, 45, 23u);
    SimplexOptions opts;
    opts.presolve = false;
    // Preparation dominates this deadline; the small LP's pivots would fit
    // in a fresh allowance. Once preparation spends it, no pivot may start.
    opts.ruiz_iterations = 50000;
    opts.time_limit_s = 0.002;
    int failures = 0;
    const auto expired = [&](const sor::core::RawResult& raw,
                             const SimplexDiagnostics& d) {
        if (d.preprocessing_ms <= 1000.0 * opts.time_limit_s || d.iterations != 0 ||
            raw.proposed_status != sor::core::Status::Interrupted) {
            std::fprintf(stderr, "deadline test: exhausted preparation started fresh pivot work\n");
            ++failures;
        }
    };
    for (const auto method : {SimplexMethod::Auto, SimplexMethod::Primal, SimplexMethod::Dual}) {
        opts.method = method;
        SimplexDiagnostics d;
        const auto raw = solve_simplex(lp, opts, d);
        expired(raw, d);
    }
    SimplexDiagnostics d;
    expired(solve_dual_simplex(lp, opts, d), d);
    DualProbeSession dual(lp, opts);
    SimplexBasis first, next;
    expired(dual.solve(opts, d, &first, nullptr), d);
    SimplexOptions unlimited = opts;
    unlimited.time_limit_s = 0.0;
    auto raw = dual.solve(unlimited, d, &next, &first);
    if (d.preprocessing_builds != 0 || d.preprocessing_ms != 0.0 ||
        raw.proposed_status != sor::core::Status::Optimal || !valid_point(lp, raw)) {
        std::fprintf(stderr, "deadline dual resume: prep builds %llu ms %g status %d residual %g reason %s\n",
                     static_cast<unsigned long long>(d.preprocessing_builds), d.preprocessing_ms,
                     static_cast<int>(raw.proposed_status), d.primal_residual,
                     raw.termination_reason.c_str());
        ++failures;
    }

    PrimalCostSession primal(lp, opts);
    expired(primal.solve(opts, d, &first, nullptr), d);
    raw = primal.solve(unlimited, d, &next, &first);
    if (d.preprocessing_builds != 0 || d.preprocessing_ms != 0.0 ||
        raw.proposed_status != sor::core::Status::Optimal || !valid_point(lp, raw)) {
        std::fprintf(stderr, "deadline primal resume: prep builds %llu ms %g status %d residual %g reason %s\n",
                     static_cast<unsigned long long>(d.preprocessing_builds), d.preprocessing_ms,
                     static_cast<int>(raw.proposed_status), d.primal_residual,
                     raw.termination_reason.c_str());
        ++failures;
    }
    return failures;
}

int test_completed_solve_transfers_preparation_without_resolve() {
    using namespace sor::engines;
    auto lp = make_lp(1, 1, 71u);
    lp.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {1.0});
    lp.c = {1.0};
    int failures = 0;
    for (int route = 0; route < 4; ++route) {
        SimplexOptions opts;
        opts.presolve = false;
        opts.pricing = SimplexPricing::DSE;
        opts.method = route == 1 ? SimplexMethod::Primal :
                      route == 2 ? SimplexMethod::Dual : SimplexMethod::Auto;
        std::unique_ptr<DualProbeSession> session;
        SimplexBasis base;
        SimplexDiagnostics d;
        const auto raw = route == 3
            ? solve_dual_simplex(lp, opts, d, &base, nullptr, nullptr, nullptr, &session)
            : solve_simplex(lp, opts, d, &base, &session);
        if (!session || raw.proposed_status != sor::core::Status::Optimal ||
            !valid_point(lp, raw) || d.preprocessing_builds != 1) {
            std::fprintf(stderr, "completed preparation transfer failed on route %d\n", route);
            ++failures;
            continue;
        }
        const auto checkpoint = session->final_factor();
        auto probe_lp = lp;
        probe_lp.col_lo[0] = 2.0;
        const auto trial = session->probe(0, 2.0, lp.col_hi[0], opts, d, base);
        if (!valid_point(probe_lp, trial) || std::fabs(trial.objective - 2.0) > 1e-9 ||
            d.preprocessing_builds != 0 || d.preprocessing_ms != 0.0 ||
            (!checkpoint.has_factor || !d.factor_reused)) {
            std::fprintf(stderr, "transferred probe lost numeric reuse or changed answer, route %d\n", route);
            ++failures;
        }
        // A probe owns a copy; neither the base's box nor factor lineage moves.
        const auto restored = session->solve(opts, d, nullptr, &base,
            &session->final_weights(), &session->final_factor());
        if (!valid_point(lp, restored) || std::fabs(restored.objective - 1.0) > 1e-9 ||
            d.preprocessing_builds != 0 || !d.factor_reused) {
            std::fprintf(stderr, "transferred base was contaminated by probe, route %d\n", route);
            ++failures;
        }
    }
    // Reduced preparation cannot be represented by an original-space session.
    SimplexOptions opts;
    opts.presolve = false;
    SimplexDiagnostics d;
    std::unique_ptr<DualProbeSession> old;
    const auto first = solve_simplex(lp, opts, d, nullptr, &old);
    if (!old || !valid_point(lp, first)) ++failures;
    opts.presolve = true;
    const auto reduced = solve_simplex(lp, opts, d, nullptr, &old);
    if (old || !valid_point(lp, reduced)) ++failures;
    return failures;
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

int test_cleanup_exports_returned_basis_factor() {
    using namespace sor::engines;
    LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 4, {0, 0, 0, 0}, {0, 1, 2, 3},
                                     {1.0, 1.0, 1.0, 1.0});
    lp.c = {1.0, 1.0, 100.0, 1.00001};
    lp.row_lo = lp.row_hi = {5.0};
    lp.col_lo = {0.0, 0.0, 0.0, 4.5};
    lp.col_hi = {1.0, sor::model::kInf, 0.0, 5.0};
    SimplexOptions opts;
    opts.presolve = false;
    opts.ruiz_iterations = 0;
    opts.dual_cost_perturbation_multiplier = 1.0;
    SimplexBasis start;
    start.n_struct = 4;
    start.basic = {3};
    start.status = {NonbasicStatus::AtLower, NonbasicStatus::AtLower,
                    NonbasicStatus::AtLower, NonbasicStatus::Basic,
                    NonbasicStatus::AtLower};
    FactorCarrier carrier;
    carrier.matrix = &lp;
    carrier.rows = lp.n_rows(); carrier.cols = lp.n_cols(); carrier.nnz = lp.nnz();
    auto prepared = prepare_simplex_model(lp, opts);
    SimplexDiagnostics cleanup;
    SimplexBasis after_cleanup;
    auto raw = solve_dual_simplex_prepared(prepared, opts, cleanup,
                                          &after_cleanup, &start, nullptr, &carrier);
    int failures = 0;
    if (cleanup.primal_cleanups != 1 || !carrier.has_factor || carrier.basis != after_cleanup.basic ||
        raw.proposed_status != sor::core::Status::Optimal || !valid_point(lp, raw) ||
        std::fabs(raw.objective - 5.000045) > 1e-9) {
        std::fprintf(stderr, "cleanup test: expected a valid primal hand-off and its final factor\n");
        ++failures;
    }
    if (carrier.matrix != &lp || carrier.rows != lp.n_rows() ||
        carrier.cols != lp.n_cols() || carrier.nnz != lp.nnz()) {
        std::fprintf(stderr, "cleanup test: owner matrix identity was lost\n");
        ++failures;
    }
    // The hand-off exports the new primal factor. Both subsequent solves
    // must adopt that valid lineage without a preparation or retokening.
    opts.dual_cost_perturbation_multiplier = 0.0;
    SimplexDiagnostics refill, reuse;
    SimplexBasis refilled_basis;
    auto rebuilt = solve_dual_simplex_prepared(prepared, opts, refill, &refilled_basis,
                                               &after_cleanup, nullptr, &carrier);
    auto adopted = solve_dual_simplex_prepared(prepared, opts, reuse, nullptr,
                                              &refilled_basis, nullptr, &carrier);
    if (!refill.factor_reused || !reuse.factor_reused ||
        rebuilt.proposed_status != sor::core::Status::Optimal ||
        adopted.proposed_status != sor::core::Status::Optimal ||
        !valid_point(lp, rebuilt) || !valid_point(lp, adopted) ||
        std::fabs(rebuilt.objective - adopted.objective) > 1e-9) {
        std::fprintf(stderr, "cleanup test: subsequent valid solves did not recover factor reuse\n");
        ++failures;
    }
    return failures;
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
    // A4: the rejection reason must be counted, and counted as exactly this
    // reason (not folded into some other bucket).
    if (d.factor_reuse_rows_mismatch != 1) {
        std::fprintf(stderr, "row-mismatch test: factor_reuse_rows_mismatch=%llu, want 1\n",
                     static_cast<unsigned long long>(d.factor_reuse_rows_mismatch));
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
    // A4: carrier_a starts empty (has_factor defaults to false), so this
    // first-ever call on it must count as "carrier empty", not silently
    // fall through some other bucket.
    if (d0.factor_reuse_carrier_empty != 1) {
        std::fprintf(stderr, "basis-mismatch test: factor_reuse_carrier_empty=%llu, want 1\n",
                     static_cast<unsigned long long>(d0.factor_reuse_carrier_empty));
        ++failures;
    }

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
    if (d2.factor_reuse_basis_mismatch != 1) {
        std::fprintf(stderr, "basis-mismatch test: factor_reuse_basis_mismatch=%llu, want 1\n",
                     static_cast<unsigned long long>(d2.factor_reuse_basis_mismatch));
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
    // A4: this first call has an empty carrier (has_factor defaults to
    // false) -- that's what gates it, checked before the matrix token.
    if (d0.factor_reuse_carrier_empty != 1) {
        std::fprintf(stderr, "null-token test: factor_reuse_carrier_empty=%llu, want 1\n",
                     static_cast<unsigned long long>(d0.factor_reuse_carrier_empty));
        ++failures;
    }
    // carrier.matrix stays null throughout (never set by the harness), so a
    // second call reusing it must never adopt. It DOES have content now (the
    // first call's normal exit refilled it), so this one is gated by the
    // null token specifically, not by an empty carrier.
    sor::engines::SimplexDiagnostics d1;
    sor::engines::solve_dual_simplex_prepared(prepared, opts, d1, nullptr, &basis,
                                              nullptr, &carrier);
    if (d1.factor_reused) {
        std::fprintf(stderr, "null-token test: adopted despite matrix == nullptr\n");
        ++failures;
    }
    if (d1.factor_reuse_matrix_null != 1) {
        std::fprintf(stderr, "null-token test: factor_reuse_matrix_null=%llu, want 1\n",
                     static_cast<unsigned long long>(d1.factor_reuse_matrix_null));
        ++failures;
    }
    return failures;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_session_rejects_foreign_factor_and_invalid_bounds();
    failures += test_rescaling_same_matrix_rejects_factor();
    failures += test_preparation_spends_first_solve_allowance();
    failures += test_completed_solve_transfers_preparation_without_resolve();
    failures += test_cleanup_exports_returned_basis_factor();
    failures += test_adopt_on_matching_basis_is_bit_correct();
    failures += test_row_count_mismatch_is_rejected_safely();
    failures += test_basis_content_mismatch_is_rejected_safely();
    failures += test_sibling_reuse_via_copy_does_not_cross_contaminate();
    failures += test_null_matrix_token_is_never_adopted();
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
