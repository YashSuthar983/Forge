// SOR — QPLIB instance -> engines::QpProblem (min form, symmetric Q).
//
// LAYER L5.  One conversion, used by the CLI's convex QP engines and by the
// QCR relaxation, so the objective convention lives in exactly one place:
//
//   QPLIB stores 0.5 x'Hx with one triangle of H used AS-IS (see
//   io/qplib.hpp and binquad.cpp; verified against QPLIB's published
//   optima).  An off-diagonal entry v at (r,c) therefore contributes 0.5*v to
//   the coefficient of x_r x_c, which in the engines' symmetric 0.5 x'Qx form
//   is Q_rc = Q_cr = 0.5*v.  A diagonal entry maps straight to Q_rr.
//   from_triplets sums duplicates, so a file listing both (r,c) and (c,r)
//   still lands on 0.5*(H_rc + H_cr).
//
// A maximisation is negated into a minimisation; `negated` reports it so a
// caller can map objective values and bounds back.
#pragma once

#include "sor/engines/qcqp.hpp"
#include "sor/engines/qp.hpp"
#include "sor/io/qplib.hpp"

#include <string>

namespace sor::search {

using core::f64;
using core::Index;

struct QplibToQpOptions {
    // Binary variables become continuous on [0,1] instead of being refused.
    // Only a caller that relaxes on purpose (QCR) sets this.
    bool relax_binary = false;
    // Binary and general-integer variables are kept, typed is_integer, for
    // the MIQP branch-and-bound (--engine miqp).  Takes precedence over
    // relax_binary.
    bool keep_integer = false;
};

// Every QPLIB instance -> the QCQP model, exactly (no refusal: all six
// constraint letters and all five variable letters are representable).
//   * objective and constraint Hessians use the same convention: a listed
//     entry v at (r,c), r != c, contributes 0.5*v*x_r*x_c, i.e. Q_rc = Q_cr
//     = 0.5*v in the symmetric form; a diagonal entry v gives Q_rr = v.
//     Verified for constraints against QPLIB's published solution points
//     (benchmarks/results/qplib-parse-all-*.md), not assumed.
//   * integer and binary variables set is_integer; a binary's bounds are
//     intersected with [0,1] (QPLIB defines binary as integer on {0,1}).
//   * a maximisation is negated into min form; out.objective_negated says so.
//   * col_names/row_names carry the file's names (defaults filled in), so a
//     published .sol, which names variables, maps back onto indices.
void qplib_to_qcqp(const io::QplibInstance& q, engines::QcqpProblem& out);

// False, with `why`, when the instance is outside what a continuous,
// linearly constrained QP can represent: quadratic constraints, or discrete
// variables that the options do not allow relaxing.
bool qplib_to_qp(const io::QplibInstance& q, const QplibToQpOptions& opts,
                 engines::QpProblem& out, bool& negated, std::string& why);

}  // namespace sor::search
