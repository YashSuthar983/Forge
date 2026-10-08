// FORGE - GF(2) / mod-2 subsystem reductions for MIP root presolve (WP-F).
//
// Clean-room spirit of PaPILO/SCIP XOR equation handling: collect binary
// parity equalities ∑ x_j ≡ b (mod 2), Gaussian-eliminate over GF(2), and
// propagate singleton / empty equations into fixings. Domains only shrink;
// every integer-feasible point that satisfied the original equalities still
// does. Pairing with zero-half: tighter binary boxes improve ZH bases later.
#pragma once

#include "forge/core/result.hpp"
#include "forge/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace forge::search {

using core::f64;
using core::Index;

struct Gf2PresolveOptions {
    bool enabled = true;
    // Max free binaries participating in the GF(2) system (cost ~ O(e^2)).
    Index max_vars = 512;
    // Max equality / forced-equality rows admitted.
    Index max_equations = 1024;
    int max_rounds = 8;
    f64 tol = 1e-9;
    // Columns integral at EVERY feasible point (the caller's integers). A
    // parity deduction is a feasibility fact, so it may rest only on those:
    // an implied-integer mark that only guarantees an integral optimum does
    // not make x + y - 2k = 1 a parity relation at every point. Null: the
    // model's is_integer.
    const std::vector<char>* integer_mask = nullptr;
};

struct Gf2PresolveDiagnostics {
    std::uint64_t equations = 0;
    std::uint64_t vars = 0;
    std::uint64_t pivots = 0;
    std::uint64_t fixings = 0;
    std::uint64_t substitutions = 0;
    bool infeasible = false;
    bool truncated = false;
};

// Detects mod-2 structure on binary rows, eliminates, and tightens col_lo/hi
// in place. Returns diagnostics; infeasible ⇒ no binary-feasible point.
Gf2PresolveDiagnostics apply_gf2_presolve(const model::LpProblem& lp,
                                          std::vector<f64>& col_lo,
                                          std::vector<f64>& col_hi,
                                          const Gf2PresolveOptions& opts);

}  // namespace forge::search
