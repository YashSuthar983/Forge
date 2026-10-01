// SOR - plain-text solution file: sor_solve's --solution-out format, read
// back by sor_check (the independent checker, item 30 of the SIH26119
// checklist). Deliberately dumb (one field per line, whitespace-separated
// vectors) so a from-scratch reader has nothing subtle to get wrong -- the
// whole point of this file is to hand a claim to code that never ran the
// solve, so the format itself must not be a place for shared bugs to hide.
#pragma once

#include "sor/core/result.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace sor::io {

using core::f64;

struct SolutionFile {
    core::Status status = core::Status::NotSolved;
    core::ProofLevel proof = core::ProofLevel::None;
    f64 objective = core::kNaN;
    std::vector<f64> x;
    std::vector<f64> y;
    std::vector<std::string> exact_dual;
    std::vector<std::string> exact_dual_farkas;
    std::vector<f64> ray;  // Farkas certificate; empty unless status == Infeasible
    std::vector<f64> primal_ray;
    std::vector<std::string> exact_primal_ray;
    std::vector<f64> dual_farkas_ray;
};

// Format:
//   status <name>
//   proof <name>
//   objective <double>
//   x <n> <v0> <v1> ... <v_{n-1}>
//   y <m> <v0> ... <v_{m-1}>
//   ray <k> <v0> ... <v_{k-1}>     (k == 0 line still present when empty)
//   primal_ray <k> ...              (unboundedness certificate)
//   dual_farkas_ray <k> ...         (infeasibility certificate)
// Optional exact witnesses, with the corresponding floating vector's size:
//   exact_dual <m> <rational> ...
//   exact_dual_farkas <m> <rational> ...
//   exact_primal_ray <n> <rational> ...
void write_solution(std::ostream& out, const core::SolveResult& r);

// Throws std::runtime_error on a malformed file (missing field, size
// mismatch between a declared count and the values that follow, or an
// unrecognized status/proof name).
SolutionFile read_solution(std::istream& in);

}  // namespace sor::io
