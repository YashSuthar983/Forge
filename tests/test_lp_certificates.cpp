#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "test_helpers.hpp"

#include <sstream>

using namespace sor;

namespace {

model::LpProblem read(const char* text) {
    std::istringstream input(text);
    io::MpsReadReport report;
    return io::read_mps(input, report);
}

}  // namespace

int main() {
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
