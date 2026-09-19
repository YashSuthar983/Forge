// SOR - bounded-variable dual revised simplex (augmented [A|-I]; Harris;
// Devex; EXPAND). Shares SimplexOptions / SimplexDiagnostics / SimplexBasis
// with the primal engine. Evidence is engines::simplex_evidence().
#pragma once

#include "sor/engines/simplex.hpp"

namespace sor::engines {

// Exact dual steepest-edge weights carried BETWEEN solves.
//
// w_i = ||B^-T e_i||^2 depends only on the basis matrix B. Changing variable
// BOUNDS -- which is all a branch-and-bound child, a strong-branch probe or a
// dive does -- leaves B untouched, so a solve that starts from the basis a
// previous solve ended on already knows its weights exactly and does not need
// the m dense BTRANs of a rebuild.
//
// The caller declares WHICH matrix the weights belong to by setting `matrix`
// to any stable token of its own (typically the address of the problem it
// reuses). Only the caller can know that -- an LpProblem may be copied, moved
// or rebuilt in place, so no address the engine can reach is a reliable
// identity. A carrier with `matrix == nullptr` is never adopted, so the safe
// way to invalidate weights after changing the matrix is `clear()`.
//
// Given that token, the engine decides the rest: dimensions, the starting
// basis being exactly the one the weights describe, exact-DSE pricing actually
// selected, and every weight finite and positive. Anything else silently falls
// back to the rebuild. Passing nullptr restores the old behaviour exactly.
//
// Weights only steer CHUZR, so a stale carrier could at worst cost pivots --
// never correctness. The guards exist to keep that "at worst" from happening,
// not to protect the answer.
struct DualEdgeWeightCarrier {
    const void* matrix = nullptr;    // caller's identity for the matrix
    core::Index rows = 0;
    core::Index cols = 0;
    core::Offset nnz = 0;
    std::vector<core::Index> basis;  // basis (slot order) these weights are for
    std::vector<core::f64> weights;  // ||B^-T e_i||^2 per basis slot

    void clear() {
        matrix = nullptr;
        rows = 0;
        cols = 0;
        nnz = 0;
        basis.clear();
        weights.clear();
    }
};

// `warm` (optional): a basis captured from an earlier run on the SAME problem
// (e.g. the Auto dispatcher's dual probe). The engine continues from it
// instead of the all-logical cold start, so a probe followed by a committed
// run costs one solve, not two. Ignored when its dimensions do not match.
//
// `weights` (optional): see DualEdgeWeightCarrier. Read at the first
// factorization, overwritten with this run's final weights on exit.
core::RawResult solve_dual_simplex(const model::LpProblem& problem,
                                   const SimplexOptions& opts,
                                   SimplexDiagnostics& diag,
                                   SimplexBasis* out_basis = nullptr,
                                   const SimplexBasis* warm = nullptr,
                                   DualEdgeWeightCarrier* weights = nullptr);

}  // namespace sor::engines
