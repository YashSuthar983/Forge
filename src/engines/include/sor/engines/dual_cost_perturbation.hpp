#pragma once

#include "sor/core/result.hpp"

#include <cstdint>
#include <vector>

namespace sor::engines {

struct DualCostPerturbationStats {
    core::f64 structural_base = 0.0;
    std::uint64_t changed_structural = 0;
    std::uint64_t changed_logical = 0;
};

// Stateless index hash in [0, 1): the same index always yields the same
// fraction, so every deterministic "random" margin in the dual (perturbation
// magnitudes, cost-shift margins) is reproducible across runs and platforms.
core::f64 deterministic_fraction(std::uint64_t index);

// Build the deterministic working-cost vector used by dual simplex. The first
// n_struct entries are structural variables; the remainder are row logicals.
// Infinite bounds use the model convention in `infinity`. Returns false for
// malformed/non-finite input and leaves `perturbed` equal to `cost` when the
// multiplier is zero.
bool build_dual_perturbed_costs(
    core::Index n_struct,
    const std::vector<core::f64>& cost,
    const std::vector<core::f64>& lower,
    const std::vector<core::f64>& upper,
    core::f64 multiplier,
    core::f64 infinity,
    std::vector<core::f64>& perturbed,
    DualCostPerturbationStats* stats = nullptr, std::uint64_t seed = 0);

// Koberstein (2005 thesis, §6.3.1) cost perturbation magnitude and sign for
// one structural column, steps 1-4 of the thesis:
//   1. xi = 100*eps_D + psi*|c_j|, psi = 1e-5;
//   2. xi <- -0.5 xi (1+mu) if the column's dual-feasible direction is
//      downward (thesis: u_j < inf), +0.5 xi (1+mu) otherwise, mu in [0,1];
//   3. xi <- w[nu_j] xi, w = (1e-2,1e-1,1,2,5,10,20,30,40,100) indexed by the
//      column's nonzero count nu_j (w_10 for nu_j >= 10);
//   4. |xi| multiplied by 0.1 or 10 until it lies in [xi_min, xi_max],
//      xi_min = min(1e-2 eps_D, psi), xi_max = max(1e3 eps_D, psi*10*mean|c|).
// `downward` selects the sign; `mu` is the random number of step 2.
core::f64 koberstein_perturbation(core::f64 cost, bool downward, core::f64 mu,
                                  core::Index column_nonzeros,
                                  core::f64 dual_tol, core::f64 mean_abs_cost);

// Thesis §6.3.1 up-front test: the problem is treated as significantly dual
// degenerate when the structural cost vector has fewer than n/4 distinct
// values.
bool koberstein_perturb_at_start(const std::vector<core::f64>& cost,
                                 core::Index n_struct);

}  // namespace sor::engines
