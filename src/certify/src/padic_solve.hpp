// SOR - exact sparse integer linear solve by p-adic (Dixon) lifting.
//
// Internal to sor_certify. The exact basis certificate needs y with E y = b
// for an integer E (a basis, rows cleared of their binary64 denominators).
// Fraction-free elimination pays big-integer arithmetic on every update:
// on pilot only ~750 structural entry updates were needed, yet rows grew to
// ~28k bits and content GCDs took 17.6 s of a 19 s elimination.
//
// Dixon (1982): factor E once modulo a word prime p, then lift the p-adic
// expansion of y one word digit per iteration with EXACT integer residuals
// r <- (r - E z) / p, whose size stays bounded by the data. The solution's
// rational form is recovered from y mod p^K by a half-extended Euclid on one
// component (the denominator is shared, so the others follow by a multiply
// and a symmetric residue) once p^K is large enough; attempts follow a
// geometric schedule. Nothing is trusted: a candidate is returned only after
// E n = b D holds exactly, so every failure (E singular mod p, budget,
// deadline, no verified candidate) just leaves the caller's exact
// elimination to run.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/exact.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <vector>

namespace sor::certify::detail {

using PadicInteger = boost::multiprecision::cpp_int;

struct PadicStats {
    std::uint64_t operations = 0;   // structural updates of the modular elimination
    std::uint64_t lifting_steps = 0;
    std::uint64_t reconstruction_attempts = 0;
    std::size_t denominator_bits = 0;
    bool declined = false;   // structural work below the caller's minimum
    // The (unique) solution needs more precision than any size-policy
    // compliant one: exact elimination would produce the same oversized
    // witness, so callers need not try.
    bool exceeds_policy = false;
};

// rows[r] is equation r: sum_j rows[r][j] * y_j = rhs[q][r], no zero entries.
// On success every solutions[q] satisfies its system exactly.
//
// Lifting costs about m K^2 word operations for K digits (~2.4x the largest
// denominator in bits) whatever the sparsity, while fraction-free
// elimination costs its structural updates W times the integer size. The
// modular factorization measures W in milliseconds; below
// `minimum_operations` the solve declines (stats.declined) and the caller's
// elimination is the cheaper exact method. Measured: pilot W/m = 200
// (elimination 25 s, lifting 11.6 s), d2q06c 10 (4.0 s vs 11.6 s),
// greenbea 2 (0.45 s vs 6.2 s), 80bau3b 0.25 (0.09 s vs 0.39 s).
bool padic_solve(const std::vector<std::map<core::Index, PadicInteger>>& rows,
                 const std::vector<std::vector<PadicInteger>>& rhs,
                 std::vector<std::vector<model::Rational>>& solutions,
                 std::uint64_t max_operations, unsigned max_bits,
                 const std::function<bool()>& expired, PadicStats* stats = nullptr,
                 std::uint64_t minimum_operations = 0,
                 const std::function<bool(std::size_t, const std::vector<std::vector<model::Rational>>&)>&
                     wanted = {});
// `wanted(q, solved)`, when given, is asked before right-hand side q > 0 with
// the solutions found so far; a declined system is returned as all zeros.

}  // namespace sor::certify::detail
