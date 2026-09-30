#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "test_helpers.hpp"

#include <sstream>
#include <limits>

using namespace sor;

namespace {

model::LpProblem read(const char* text) {
    std::istringstream input(text);
    io::MpsReadReport report;
    return io::read_mps(input, report);
}

}  // namespace

// certify::safe_lagrangian_lower_bound: weak duality from ANY multipliers.
void test_safe_lagrangian_bound() {
    // min x + y  s.t.  x + y >= 2,  x in [0, inf), y in [0, 3].
    const auto lp = read(R"(NAME SAFE
ROWS
 N OBJ
 G R1
COLUMNS
    X OBJ 1 R1 1
    Y OBJ 1 R1 1
RHS
    RHS R1 2
BOUNDS
 UP BND Y 3
ENDATA
)");
    // Exact dual y = 1: d = 0 everywhere, bound = 2.
    auto b = certify::safe_lagrangian_lower_bound(lp, {1.0}, lp.col_lo, lp.col_hi);
    CHECK(b.finite);
    CHECK_NEAR(b.value, 2.0, 1e-9);
    CHECK(b.value <= 2.0);
    // Round-off dual y = 1 + 1e-14 leaves d_x = -1e-14 on [0, inf): the
    // multiplier correction must restore a finite bound (not wave it through).
    b = certify::safe_lagrangian_lower_bound(lp, {1.0 + 1e-14}, lp.col_lo, lp.col_hi);
    CHECK(b.finite);
    CHECK(b.value <= 2.0);
    CHECK(b.value >= 2.0 - 1e-6);
    // Any y is valid: y = 0.5 gives L = 1 + 0.5*0 ... = 1 (weak duality).
    b = certify::safe_lagrangian_lower_bound(lp, {0.5}, lp.col_lo, lp.col_hi);
    CHECK(b.finite);
    CHECK(b.value <= 1.0 + 1e-12);
}

void test_safe_bound_refuses_unbounded_direction() {
    // min -x  s.t.  y - x >= -1 (never caps x),  x, y in [0, inf).
    // LP is unbounded below; y = 1e-8 leaves d_x = -1 + 1e-8 < 0 on x with
    // no finite upper bound anywhere: the bound must be -inf.
    const auto lp = read(R"(NAME UNB
ROWS
 N OBJ
 G R1
COLUMNS
    X OBJ -1 R1 -1
    Y OBJ 0 R1 1
RHS
    RHS R1 -1
ENDATA
)");
    const auto b = certify::safe_lagrangian_lower_bound(lp, {1e-8}, lp.col_lo, lp.col_hi);
    CHECK(!b.finite);
}

void test_small_cost_cannot_erase_unbounded_direction() {
    // The cost is inside the KKT tolerance, but its product with the
    // unbounded upper box is still -infinity. A fake zero gap is unsound.
    const auto source = read(R"(NAME TINY
ROWS
 N OBJ
COLUMNS
 X OBJ -0.000000001
ENDATA
)");
    for (bool maximize : {false, true}) {
        auto lp = source;
        lp.maximize = maximize;
        lp.c[0] = maximize ? 1e-9 : -1e-9;
        lp.obj_offset = 5.0;
        core::RawResult fake;
        fake.x = {0.0};
        fake.objective = fake.dual_bound = 5.0;
        fake.proposed_status = core::Status::Optimal;
        fake.proposed_level = core::ProofLevel::ProvedOptimalFP;
        auto checked = certify::check_lp_point(lp, fake, 1e-7, 1e-7, 1e-9, true);
        CHECK(checked.checker_passed); // The primal point itself is feasible.
        CHECK(!std::isfinite(checked.gap_rel));
        CHECK(certify::finalize_result(fake, checked).status != core::Status::Optimal);

        engines::SimplexOptions opts;
        opts.presolve = false;
        opts.ruiz_iterations = 0;
        opts.method = engines::SimplexMethod::Primal;
        opts.time_limit_s = 1.0;
        opts.max_iterations = 32;
        for (bool dual : {false, true}) {
            engines::SimplexDiagnostics diag;
            auto raw = dual ? engines::solve_dual_simplex(lp, opts, diag)
                            : engines::solve_simplex(lp, opts, diag);
            CHECK(!diag.dual_bound_finite);
            checked = certify::check_lp_point(lp, raw, 1e-7, 1e-7, 1e-9, true);
            CHECK(certify::finalize_result(raw, checked).status != core::Status::Optimal);
        }
    }
}

void test_checked_repair_preserves_sense_and_offset() {
    auto lp = read(R"(NAME REPAIR
ROWS
 N OBJ
 G R1
COLUMNS
 X OBJ 1 R1 1
RHS
 RHS R1 2
ENDATA
)");
    for (bool maximize : {false, true}) {
        lp.maximize = maximize;
        lp.c[0] = maximize ? -1.0 : 1.0;
        lp.obj_offset = 5.0;
        core::RawResult raw;
        raw.x = {2.0};
        raw.y = {maximize ? -(1.0 + 1e-14) : 1.0 + 1e-14};
        raw.objective = maximize ? 3.0 : 7.0;
        raw.dual_bound = raw.objective;
        raw.proposed_status = core::Status::Optimal;
        raw.proposed_level = core::ProofLevel::ProvedOptimalFP;
        const auto checked = certify::check_lp_point(lp, raw, 1e-7, 1e-7, 1e-9, true);
        CHECK(checked.checker_passed);
        CHECK(checked.gap_rel < 1e-9);
        CHECK(certify::finalize_result(raw, checked).status == core::Status::Optimal);
        engines::SimplexOptions opts;
        opts.presolve = false;
        opts.ruiz_iterations = 0;
        opts.method = engines::SimplexMethod::Primal;
        for (bool dual : {false, true}) {
            engines::SimplexDiagnostics diag;
            auto solved = dual ? engines::solve_dual_simplex(lp, opts, diag)
                               : engines::solve_simplex(lp, opts, diag);
            CHECK(diag.dual_bound_finite);
            CHECK_NEAR(solved.dual_bound, raw.objective, 1e-9);
            CHECK(maximize ? solved.dual_bound >= raw.objective
                           : solved.dual_bound <= raw.objective);
        }
    }
    core::RawResult nonfinite;
    nonfinite.x = {std::numeric_limits<double>::quiet_NaN()};
    CHECK(!certify::check_lp_point(lp, nonfinite, 1e-7, 1e-7, 1e-9, false).checker_passed);
    nonfinite.x = {sor::model::kInf};
    CHECK(!certify::check_lp_point(lp, nonfinite, 1e-7, 1e-7, 1e-9, false).checker_passed);
}

int main() {
    test_safe_lagrangian_bound();
    test_safe_bound_refuses_unbounded_direction();
    test_small_cost_cannot_erase_unbounded_direction();
    test_checked_repair_preserves_sense_and_offset();
    const auto infeasible = read(R"(NAME INF
ROWS
 N OBJ
 L R1
COLUMNS
 X OBJ 0 R1 1
RHS
 RHS R1 1
BOUNDS
 LO BND X 2
ENDATA
)");
    const auto dual = certify::check_dual_farkas_ray(infeasible, {1.0}, 1e-7);
    CHECK(dual.certified);
    CHECK(dual.contradiction > 0.9);
    CHECK(!certify::check_dual_farkas_ray(infeasible, {0.0}, 1e-7).certified);

    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.dual_farkas_ray = dual;
        core::ProofEvidence ev;
        ev.claimed_level = core::ProofLevel::BoundOnly;
        ev.dual_farkas_violation = dual.max_sign_residual;
        ev.dual_farkas_contradiction = dual.contradiction;
        const auto result = certify::finalize_result(std::move(raw), ev);
        CHECK(result.status == core::Status::Infeasible);
        CHECK(result.dual_farkas_ray.certified);
        CHECK(result.ray_certified);
        CHECK(result.ray == result.dual_farkas_ray.multipliers);
    }

    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Infeasible;
        raw.dual_farkas_ray.multipliers = {1.0};
        core::ProofEvidence ev;
        const auto result = certify::finalize_result(std::move(raw), ev);
        CHECK(result.status == core::Status::NoSolutionFound);
        CHECK(!result.downgrade_reason.empty());
    }

    const auto unbounded = read(R"(NAME UNB
ROWS
 N OBJ
COLUMNS
 X OBJ -1
BOUNDS
 LO BND X 0
ENDATA
)");
    const auto primal = certify::check_primal_ray(unbounded, {1.0}, 1e-7);
    CHECK(primal.certified);
    CHECK(primal.objective_direction < -0.9);
    CHECK(!certify::check_primal_ray(unbounded, {-1.0}, 1e-7).certified);

    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Unbounded;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.primal_ray = primal;
        core::ProofEvidence ev;
        ev.claimed_level = core::ProofLevel::BoundOnly;
        ev.primal_ray_violation = std::max(primal.max_row_residual,
                                           primal.max_bound_sign_residual);
        ev.primal_ray_objective = primal.objective_direction;
        const auto result = certify::finalize_result(std::move(raw), ev);
        CHECK(result.status == core::Status::Unbounded);
        CHECK(result.primal_ray.certified);
    }

    const auto feasible = read(R"(NAME FEAS
ROWS
 N OBJ
 L R1
COLUMNS
 X OBJ 0 R1 1
RHS
 RHS R1 2
BOUNDS
 LO BND X 2
ENDATA
)");

    // The aggregate checker, used by the CLI, must ignore producer residuals
    // and certify from the original model.  Its objective convention remains
    // correct for maximization as well as minimization.
    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Unbounded;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.primal_ray.direction = {1.0};
        core::ProofEvidence proposed;
        proposed.claimed_level = core::ProofLevel::BoundOnly;
        proposed.primal_feas_tol = 1e-7;
        auto checked = certify::check_lp_result(unbounded, raw, proposed);
        const auto result = certify::finalize_result(std::move(raw), checked);
        CHECK(result.status == core::Status::Unbounded);
        CHECK(result.primal_ray.certified);
    }

    {
        auto maximize_unbounded = unbounded;
        maximize_unbounded.maximize = true;
        maximize_unbounded.c = {1.0};
        core::RawResult raw;
        raw.proposed_status = core::Status::Unbounded;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.primal_ray.direction = {1.0};
        core::ProofEvidence proposed;
        proposed.claimed_level = core::ProofLevel::BoundOnly;
        auto checked = certify::check_lp_result(
            maximize_unbounded, raw, proposed);
        const auto result = certify::finalize_result(std::move(raw), checked);
        CHECK(result.status == core::Status::Unbounded);
        CHECK(result.primal_ray.certified);
    }

    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.dual_farkas_ray.multipliers = {1.0};
        core::ProofEvidence proposed;
        proposed.claimed_level = core::ProofLevel::BoundOnly;
        // Deliberately lie in the producer evidence; independent checking of
        // the feasible model below must still reject the claim.
        proposed.dual_farkas_violation = 0.0;
        proposed.dual_farkas_contradiction = 100.0;
        auto checked = certify::check_lp_result(feasible, raw, proposed);
        const auto result = certify::finalize_result(std::move(raw), checked);
        CHECK(result.status == core::Status::NoSolutionFound);
    }

    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Unbounded;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.primal_ray.direction = {-1.0};
        core::ProofEvidence proposed;
        proposed.claimed_level = core::ProofLevel::BoundOnly;
        // Producer metadata claims a perfect improving direction, but the
        // direction violates the lower-bound recession cone.  The independent
        // original-model checker must reject it.
        proposed.primal_ray_violation = 0.0;
        proposed.primal_ray_objective = -1.0;
        auto checked = certify::check_lp_result(unbounded, raw, proposed);
        const auto result = certify::finalize_result(std::move(raw), checked);
        CHECK(result.status == core::Status::NoSolutionFound);
        CHECK(!result.primal_ray.certified);
    }

    // A zero-separation near-certificate must not certify infeasibility.
    CHECK(!certify::check_dual_farkas_ray(feasible, {1.0}, 1e-7).certified);

    // Finite support terms are never tolerance-zeroed.  This LP is feasible
    // at x=-1e8.  Dropping the small 1e-8 A'y term before multiplying it by
    // the large finite lower bound would manufacture lower=0 > upper=-1 and
    // falsely certify infeasibility.
    {
        model::LpProblem scaled_support;
        scaled_support.name = "finite-support-near-certificate";
        scaled_support.A = sparse::from_triplets(
            1, 1, {0}, {0}, {1e-8});
        scaled_support.c = {0.0};
        scaled_support.col_lo = {-1e9};
        scaled_support.col_hi = {1e9};
        scaled_support.row_lo = {-model::kInf};
        scaled_support.row_hi = {-1.0};
        const auto near = certify::check_dual_farkas_ray(
            scaled_support, {1.0}, 1e-7);
        CHECK(!near.certified);
        CHECK(near.contradiction < 0.0);

        core::RawResult raw;
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.dual_farkas_ray.multipliers = {1.0};
        core::ProofEvidence lie;
        lie.claimed_level = core::ProofLevel::BoundOnly;
        lie.dual_farkas_violation = 0.0;
        lie.dual_farkas_contradiction = 1.0;
        const auto checked = certify::check_lp_result(
            scaled_support, raw, lie);
        const auto result = certify::finalize_result(std::move(raw), checked);
        CHECK(result.status == core::Status::NoSolutionFound);
        CHECK(!result.ray_certified);
    }

    // Even a coefficient below feasibility tolerance cannot be discarded
    // when its variable is unbounded: x=-1e8 satisfies 1e-8*x <= -1.
    // Ignoring that term would fabricate a positive Farkas contradiction.
    {
        model::LpProblem unbounded_support;
        unbounded_support.name = "unbounded-support-near-certificate";
        unbounded_support.A = sparse::from_triplets(
            1, 1, {0}, {0}, {1e-8});
        unbounded_support.c = {0.0};
        unbounded_support.col_lo = {-model::kInf};
        unbounded_support.col_hi = {model::kInf};
        unbounded_support.row_lo = {-model::kInf};
        unbounded_support.row_hi = {-1.0};
        CHECK(!certify::check_dual_farkas_ray(
            unbounded_support, {1.0}, 1e-7).certified);
    }

    // An approximate ray for x >= 1, x <= 0 with x free leaves a small A'y
    // residual on the free column. The checker may repair it into the exact
    // certificate, but it must re-verify the repaired ray, and it must not
    // repair into a multiplier whose sign faces an infinite row side.
    {
        model::LpProblem near_free;
        near_free.name = "near-free-column-farkas";
        near_free.A = sparse::from_triplets(2, 1, {0, 1}, {0, 0}, {1.0, 1.0});
        near_free.c = {0.0};
        near_free.col_lo = {-model::kInf};
        near_free.col_hi = {model::kInf};
        near_free.row_lo = {1.0, -model::kInf};
        near_free.row_hi = {model::kInf, 0.0};
        const auto repaired = certify::check_dual_farkas_ray(
            near_free, {-1.0, 1.0 - 1e-11}, 1e-7);
        CHECK(repaired.certified);
        // The feasible variant (x >= 0, x <= 1) has no certificate at all.
        near_free.row_lo = {0.0, -model::kInf};
        near_free.row_hi = {model::kInf, 1.0};
        CHECK(!certify::check_dual_farkas_ray(
            near_free, {-1.0, 1.0 - 1e-11}, 1e-7).certified);
    }

    // Ranged rows and free columns need the same original-space separation
    // convention.  Two opposing one-sided rows make the free-variable model
    // infeasible while cancelling A'y exactly.
    {
        model::LpProblem free_column;
        free_column.name = "free-column-farkas";
        free_column.A = sparse::from_triplets(
            2, 1, {0, 1}, {0, 0}, {1.0, 1.0});
        free_column.c = {0.0};
        free_column.col_lo = {-model::kInf};
        free_column.col_hi = {model::kInf};
        free_column.row_lo = {1.0, -model::kInf};
        free_column.row_hi = {model::kInf, 0.0};
        CHECK(certify::check_dual_farkas_ray(
            free_column, {-1.0, 1.0}, 1e-7).certified);

        model::LpProblem ranged;
        ranged.name = "ranged-row-farkas";
        ranged.A = sparse::from_triplets(1, 1, {0}, {0}, {1.0});
        ranged.c = {0.0};
        ranged.col_lo = {0.0};
        ranged.col_hi = {0.0};
        ranged.row_lo = {2.0};
        ranged.row_hi = {3.0};
        CHECK(certify::check_dual_farkas_ray(
            ranged, {-1.0}, 1e-7).certified);
    }

    return test::finish("test_lp_certificates");
}
