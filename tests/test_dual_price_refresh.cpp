// Task 3 verification: working-price refresh contract and approximate-zero
// stationarity through later pivots / resync / probe reuse.
#include "forge/certify/finalize.hpp"
#include "forge/engines/dual_ratio_test.hpp"
#include "forge/engines/dual_simplex.hpp"
#include "forge/engines/simplex.hpp"
#include "forge/io/mps.hpp"

#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

using forge::core::f64;
using forge::core::Index;
using forge::core::Status;
using forge::engines::DualProbeSession;
using forge::engines::NonbasicStatus;
using forge::engines::SimplexDiagnostics;
using forge::engines::SimplexMethod;
using forge::engines::SimplexOptions;
using forge::engines::WorkingPriceRefreshAction;
using forge::engines::WorkingPriceRefreshKind;
using forge::engines::apply_working_price_endpoint_flip;
using forge::engines::classify_working_price_refresh_column;
using forge::engines::model_requires_primal_cleanup_after_restore;
using forge::engines::solve_simplex;

namespace {

constexpr f64 kInf = std::numeric_limits<f64>::infinity();
constexpr f64 kTol = 1e-7;

WorkingPriceRefreshAction classify(int phase, NonbasicStatus st, f64 d,
                                   f64 lo, f64 hi, f64 dtol = kTol) {
    return classify_working_price_refresh_column(phase, st, d, lo, hi, dtol);
}

// Regression: a post-refresh reduced cost that is beyond-tol for the current
// endpoint must be classified as a repair (phase 1) or shift (phase 2), and
// applying the phase-1 flip must update status AND value.
void test_beyond_tol_refresh_creates_repair_invariant() {
    // Phase 1: AtLower with d = -1e-2 ≪ -tol → must flip to upper and set value=hi.
    {
        const auto a = classify(1, NonbasicStatus::AtLower, -1e-2, 0.0, 5.0);
        CHECK(a.kind == WorkingPriceRefreshKind::FlipEndpoint);
        CHECK(a.new_status == NonbasicStatus::AtUpper);
        CHECK_NEAR(a.new_value, 5.0, 0.0);
        NonbasicStatus st = NonbasicStatus::AtLower;
        f64 val = 0.0;
        CHECK(apply_working_price_endpoint_flip(a, st, val));
        CHECK(st == NonbasicStatus::AtUpper);
        CHECK_NEAR(val, 5.0, 0.0);
    }
    // Phase 1 upper mirror.
    {
        const auto a = classify(1, NonbasicStatus::AtUpper, +1e-2, 0.0, 5.0);
        CHECK(a.kind == WorkingPriceRefreshKind::FlipEndpoint);
        NonbasicStatus st = NonbasicStatus::AtUpper;
        f64 val = 5.0;
        CHECK(apply_working_price_endpoint_flip(a, st, val));
        CHECK(st == NonbasicStatus::AtLower);
        CHECK_NEAR(val, 0.0, 0.0);
    }
    // Phase 2 one-sided: beyond-tol → shift (caller must honor refusal).
    {
        const auto a = classify(2, NonbasicStatus::AtLower, -1e-2, 0.0, kInf);
        CHECK(a.kind == WorkingPriceRefreshKind::ShiftOneSided);
        NonbasicStatus st = NonbasicStatus::AtLower;
        f64 val = 0.0;
        CHECK(!apply_working_price_endpoint_flip(a, st, val)); // not a flip
        CHECK(st == NonbasicStatus::AtLower);
        CHECK_NEAR(val, 0.0, 0.0);
    }
}

void test_within_tol_is_noop_and_boxed_preserved() {
    // Within-tol wrong sign → NoOp (must not re-park).
    CHECK(classify(1, NonbasicStatus::AtLower, -0.5 * kTol, 0.0, 5.0).kind ==
          WorkingPriceRefreshKind::NoOp);
    CHECK(classify(1, NonbasicStatus::AtUpper, +0.5 * kTol, 0.0, 5.0).kind ==
          WorkingPriceRefreshKind::NoOp);
    CHECK(classify(2, NonbasicStatus::AtLower, -0.5 * kTol, 0.0, kInf).kind ==
          WorkingPriceRefreshKind::NoOp);
    // Phase-2 boxed beyond-tol: mismatch is visible but refresh must NOT flip.
    {
        const auto a = classify(2, NonbasicStatus::AtLower, -1e-2, 0.0, 5.0);
        CHECK(a.kind == WorkingPriceRefreshKind::BoxedMismatchIgnored);
        NonbasicStatus st = NonbasicStatus::AtLower;
        f64 val = 0.0;
        CHECK(!apply_working_price_endpoint_flip(a, st, val));
        CHECK(st == NonbasicStatus::AtLower);
        CHECK_NEAR(val, 0.0, 0.0);
    }
    {
        const auto a = classify(2, NonbasicStatus::AtUpper, +1e-2, 0.0, 5.0);
        CHECK(a.kind == WorkingPriceRefreshKind::BoxedMismatchIgnored);
    }
}

// Engine: forced periodic refresh on a dual-phase-1 model must refresh, may
// repair beyond-tol parking, and must still prove. Repairs imply the post-
// refresh contract ran after a price rebuild that exposed a violation.
void test_engine_refresh_repairs_restore_and_prove() {
    const std::string mps = R"(NAME          DP1
ROWS
 N  COST
 L  R1
 L  R2
COLUMNS
    X1        COST      -3.0       R1        2.0
    X1        R2        1.0
    X2        COST      -2.0       R1        1.0
    X2        R2        3.0
RHS
    RHS       R1        12.0       R2        15.0
ENDATA
)";
    SimplexOptions opts;
    opts.method = SimplexMethod::Dual;
    opts.presolve = false;
    opts.dual_resync_interval = 1;
    forge::io::MpsReadReport rep;
    std::istringstream in(mps);
    auto problem = forge::io::read_mps(in, rep);
    SimplexDiagnostics diag;
    auto raw = solve_simplex(problem, opts, diag);
    auto ev = forge::engines::simplex_evidence(diag, opts);
    ev = forge::certify::check_lp_result(problem, raw, ev);
    auto r = forge::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal);
    CHECK(diag.working_price_refreshes > 0);
    CHECK(diag.phase1_iterations > 0);
    // Path must remain dual-feasible enough to prove after any beyond-tol
    // parking repairs the refresh contract applied.
    CHECK(diag.primal_residual <= 1e-6);
}

// Refused one-sided shifts after a refresh must not be ignored: with shifts
// disabled, recovery is primal cleanup and/or an explicit non-Optimal dual
// handoff that Auto/dual+cleanup still resolves, or a proved Optimal via
// numerical-zero / within-tol paths. Never a silent Optimal with unrepaired
// refused one-sided residual and zero cleanups when refreshes refused shifts.
void test_refused_onesided_refresh_has_explicit_recovery() {
    SimplexOptions opts;
    opts.method = SimplexMethod::Dual;
    opts.presolve = false;
    opts.allow_cost_shifts = false;
    opts.dual_resync_interval = 1;
    forge::io::MpsReadReport rep;
    std::istringstream in(forge::test::kTestLpMps);
    auto problem = forge::io::read_mps(in, rep);
    SimplexDiagnostics diag;
    auto raw = solve_simplex(problem, opts, diag);
    auto ev = forge::engines::simplex_evidence(diag, opts);
    ev = forge::certify::check_lp_result(problem, raw, ev);
    auto r = forge::certify::finalize_result(std::move(raw), ev);
    const bool recovered =
        r.status == Status::Optimal ||
        diag.primal_cleanups > 0 ||
        diag.harmful_shift_fresh_factor_blocks > 0 ||
        diag.refused_cost_shifts > 0;
    CHECK(recovered);
    if (r.status == Status::Optimal) {
        CHECK(r.proof == forge::core::ProofLevel::ProvedOptimalFP ||
              r.proof == forge::core::ProofLevel::ProvedKKT);
    }
    // Approximate-zero residual must respect the applicable dual tolerance.
    CHECK(diag.numerical_zero_stationarity_max <= opts.dual_feas_tol + 1e-18);
}

// Task 4 cancel decision: after model restore, zero nonflippable residual
// with only boxed issues on a primal-infeasible point continues dual; a
// nonflippable residual or boxed residual on a feasible point commits cleanup.
void test_cleanup_reeval_cancel_vs_commit_decision() {
    CHECK(!model_requires_primal_cleanup_after_restore(0.0, 0.0, 0.0));
    CHECK(!model_requires_primal_cleanup_after_restore(0.0, 0.0, 1e-3));
    // Boxed residual + primal infeasible → dual may flip; do not hand off.
    CHECK(!model_requires_primal_cleanup_after_restore(0.0, 1e-3, 1e-3));
    // Restore cleared working obstruction (model dual feasible).
    CHECK(!model_requires_primal_cleanup_after_restore(0.0, 0.0, 0.0));
    // Nonflippable residual → commit.
    CHECK(model_requires_primal_cleanup_after_restore(1e-3, 0.0, 0.0));
    CHECK(model_requires_primal_cleanup_after_restore(1e-3, 0.0, 1.0));
    // Boxed residual on primal-feasible point → commit (preserve the point).
    CHECK(model_requires_primal_cleanup_after_restore(0.0, 1e-3, 0.0));
}

// Approximate-zero contract through subsequent pivots, a forced resync cadence,
// and DualProbeSession reuse: stationarity approx stays within dual_feas_tol
// on the base solve and on a sibling probe/resume.
void test_approx_zero_through_pivots_resync_and_probe() {
    SimplexOptions opts;
    opts.method = SimplexMethod::Dual;
    opts.presolve = false;
    opts.allow_cost_shifts = false;
    opts.dual_resync_interval = 1;
    opts.pricing = forge::engines::SimplexPricing::DSE;

    forge::io::MpsReadReport rep;
    std::istringstream in(forge::test::kTestLpMps);
    auto lp = forge::io::read_mps(in, rep);

    std::unique_ptr<DualProbeSession> session;
    forge::engines::SimplexBasis base;
    SimplexDiagnostics diag;
    auto raw = solve_simplex(lp, opts, diag, &base, &session);
    auto ev = forge::engines::simplex_evidence(diag, opts);
    ev = forge::certify::check_lp_result(lp, raw, ev);
    auto r = forge::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal);
    CHECK(diag.numerical_zero_stationarity_max <= opts.dual_feas_tol + 1e-18);
    const f64 base_stat = diag.numerical_zero_stationarity_max;
    const auto base_zero_steps = diag.numerical_zero_dual_steps;

    CHECK(session != nullptr);
    // Sibling probe (bound nudge) then restore: refresh/zero contracts must
    // still hold under factor/weight reuse.
    SimplexDiagnostics probe_diag;
    auto probe_raw = session->probe(0, lp.col_lo[0], lp.col_hi[0], opts,
                                    probe_diag, base);
    CHECK(probe_diag.numerical_zero_stationarity_max <=
          opts.dual_feas_tol + 1e-18);

    SimplexDiagnostics resume_diag;
    auto resume = session->solve(opts, resume_diag, nullptr, &base,
                                 &session->final_weights(),
                                 &session->final_factor());
    auto rev = forge::engines::simplex_evidence(resume_diag, opts);
    rev = forge::certify::check_lp_result(lp, resume, rev);
    auto rr = forge::certify::finalize_result(std::move(resume), rev);
    CHECK(rr.status == Status::Optimal);
    CHECK(resume_diag.numerical_zero_stationarity_max <=
          opts.dual_feas_tol + 1e-18);
    // Residual never exceeds the tolerance that authorized the approximation.
    CHECK(base_stat <= opts.dual_feas_tol + 1e-18);
    (void)base_zero_steps;
    (void)probe_raw;
}

}  // namespace

int main() {
    test_beyond_tol_refresh_creates_repair_invariant();
    test_within_tol_is_noop_and_boxed_preserved();
    test_cleanup_reeval_cancel_vs_commit_decision();
    test_engine_refresh_repairs_restore_and_prove();
    test_refused_onesided_refresh_has_explicit_recovery();
    test_approx_zero_through_pivots_resync_and_probe();
    return forge::test::finish("test_dual_price_refresh");
}
