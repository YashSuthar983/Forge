#include "sor/presolve/presolve.hpp"
#include "test_helpers.hpp"
#include <cmath>

int main() {
    sor::model::LpProblem p;
    p.A = sor::sparse::from_triplets(3, 3, {0,0,1,1,1,2,2,2}, {0,1,0,1,2,0,1,2}, {1,1,2,2,1,3,3,1});
    p.c = {1,2,1}; p.col_lo = {0,0,0}; p.col_hi = {10,10,10};
    p.row_lo = {4, 9, 13}; p.row_hi = p.row_lo;
    sor::presolve::PresolveOptions options; options.equation_sparsification = true;
    const auto outcome = sor::presolve::presolve(p, options);
    CHECK(outcome.map.stats.equation_sparsifications > 0);
    CHECK(outcome.map.stats.sparsification_nnz_removed > 0);
    const auto lift_known = [](const sor::presolve::PresolveMap& map) {
        const std::vector<double> known{4,0,1};
        std::vector<double> reduced;
        for (auto original : map.new_to_orig) reduced.push_back(known[static_cast<std::size_t>(original)]);
        return sor::presolve::postsolve(map, reduced);
    };
    const auto x = lift_known(outcome.map);
    CHECK(p.max_row_violation(x) < 1e-10);
    CHECK(p.max_bound_violation(x) < 1e-10);
    // Non-dyadic row operations must not be rounded into a different model.
    p.A = sor::sparse::from_triplets(2, 3, {0,0,1,1,1}, {0,1,0,1,2}, {3,6,1,2,1});
    p.row_lo = {12,5}; p.row_hi = p.row_lo;
    const auto fractional = sor::presolve::presolve(p, options);
    const auto recovered = lift_known(fractional.map);
    CHECK(p.max_row_violation(recovered) < 1e-9);
    // General dependency combines three equations; pairwise proportional
    // row detection cannot remove this redundant equation.
    p.A = sor::sparse::from_triplets(3,3,{0,0,1,1,2,2,2},{0,1,1,2,0,1,2},{1,1,1,1,1,2,1});
    p.c = {0,0,0}; p.row_lo = p.row_hi = {2,2,4};
    const auto dependent = sor::presolve::presolve(p, options);
    CHECK(dependent.map.stats.linear_dependencies_removed == 1);
    // Probe budget is bounded and continuous domains are never integer-rounded.
    options.domain_probing = true; options.max_domain_probes = 1;
    p.A = sor::sparse::from_triplets(2,2,{0,0,1,1},{0,1,0,1},{1,1,1,-1});
    p.c = {0,0}; p.col_lo = {0,0}; p.col_hi = {1,1};
    p.row_lo = {1.5,0}; p.row_hi = {sor::model::kInf,sor::model::kInf};
    const auto probed = sor::presolve::presolve(p, options);
    CHECK(probed.map.stats.domain_probes == 2);
    CHECK(probed.map.stats.bounds_tightened > 0);
    // Exact chronological primal-ray lifting retains a non-dyadic quotient.
    p.A = sor::sparse::from_triplets(2,3,{0,0,1,1},{0,1,1,2},{3,-1,1,-1});
    p.c = {0,-1,0}; p.col_lo = {0,0,0}; p.col_hi = {sor::model::kInf,sor::model::kInf,sor::model::kInf};
    p.row_lo = {0,-sor::model::kInf}; p.row_hi = {0,0};
    options = {}; options.equation_sparsification = true;
    const auto ray_map = sor::presolve::presolve(p,options).map;
    sor::core::PrimalRay reduced_ray;
    for (auto col : ray_map.new_to_orig) {
        reduced_ray.direction.push_back(col == 0 ? 1.0/3 : 1);
        reduced_ray.exact_direction.push_back(col == 0 ? "1/3" : "1");
    }
    const auto lifted_ray = sor::presolve::recover_primal_ray(p,ray_map,reduced_ray,1e-7);
    CHECK(lifted_ray.certified);
    CHECK(!lifted_ray.exact_direction.empty());
    // Aggregation changes y_source = -sum(mu*y_target), while retaining
    // the target multipliers. This catches the former inverse-row update.
    p.A = sor::sparse::from_triplets(3,2,{0,0,1,2},{0,1,0,1},{3,-1,3,1});
    p.c = {0,0}; p.col_lo = {-sor::model::kInf,-sor::model::kInf};
    p.col_hi = {sor::model::kInf,sor::model::kInf};
    p.row_lo = {0,3,-sor::model::kInf}; p.row_hi = {0,sor::model::kInf,0};
    sor::presolve::PresolveMap map;
    map.problem.A = sor::sparse::from_triplets(2,1,{0,1},{0,0},{1,1});
    map.row_new_to_orig = {1,2};
    sor::presolve::EqualityAggregation aggregation;
    aggregation.row = 0; aggregation.col = 0; aggregation.coeff = 3;
    aggregation.affected_rows = {1}; aggregation.row_multipliers = {1};
    map.equality_aggregations.push_back(aggregation);
    sor::presolve::DualRecoveryStep journal;
    journal.kind = sor::presolve::DualRecoveryKind::EqualityAggregation;
    journal.row = 0; journal.record = 0; map.recovery_steps.push_back(journal);
    sor::core::DualFarkasRay reduced_farkas;
    reduced_farkas.multipliers = {-1,1}; reduced_farkas.exact_multipliers = {"-1","1"};
    CHECK(sor::presolve::recover_dual_farkas_ray(p,map,reduced_farkas,1e-7).certified);
    return sor::test::finish("test_advanced_presolve");
}
