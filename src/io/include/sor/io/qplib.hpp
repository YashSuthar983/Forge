// SOR — QPLIB reader (qplib.zib.de / github.com/ralna/QPLIB native format).
//
// LAYER L2 — does not depend on engines. The CLI maps this into the engine
// problem types.
//
// This is a plain data format: numbers in a fixed section order, with a
// free-form comment allowed after the last value on a line. Decoded from the
// published format description and from real instances; no reader
// implementation was consulted. Same category as the MPS/QPS readers beside
// it -- read the input, never borrow solve logic.
//
// SECTION ORDER (blank lines are insignificant):
//   name
//   3-letter classification          e.g. "QBL"
//   Minimize | Maximize
//   n                                variables
//   m                                general linear constraints
//   nnz(H) then that many (row col value)   upper triangle of H
//   default(g), count, then (index value) pairs
//   f                                objective constant
//   nnz(A) then that many (row col value)
//   infinity sentinel                typically 1e19 or 1e20
//   default(c_l), count, pairs   ·   default(c_u), count, pairs
//   then, CONDITIONAL on classification[1] (the variable-type letter):
//     'B'      x_l/x_u/vartype ALL omitted; bounds are [0,1], all binary
//     'C','I'  x_l/x_u present, vartype omitted (all continuous / all integer)
//     'G','M'  x_l/x_u present AND a vartype block (default + overrides)
//   then the TRAILER (read so the whole file is accounted for, see below):
//     starting x: default, count, pairs            (length n)
//     starting constraint duals: default, count, pairs  -- ONLY when m exists
//     starting bound duals: default, count, pairs  (length n)
//     count, then (index name) variable-name overrides
//     count, then (index name) constraint-name overrides  -- present even
//       when m is absent (verified on CCB files, where it is "0")
//
// OBJECTIVE CONVENTION: 0.5 x'Hx + g'x + f, with H given as its UPPER
// TRIANGLE only. Callers wanting a full symmetric Q must mirror it.
//
// THREE SECTIONS ARE CONDITIONAL, and each one was verified against real
// instances rather than assumed. Getting any of them wrong does not produce
// an error -- it silently desyncs the line cursor and every later section is
// read from the wrong offset.
//
//   1. The OBJECTIVE Hessian block (nnz(H) + triplets) is ABSENT when the
//      objective letter is 'L' (linear objective). Verified on LCQ files,
//      where the g block follows m directly.
//   2. The PER-CONSTRAINT Hessian block (nnz + (constraint,row,col,value)
//      QUADRUPLES) is PRESENT when the constraint letter is D, C or Q, and
//      sits between f and the A matrix. Verified on LCQ and QCQ.
//   3. m, the A matrix and the constraint-bound blocks are ABSENT when the
//      constraint letter is N or B. Verified on QBB/QBN.
#pragma once

#include "sor/core/result.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace sor::io {

using core::f64;
using core::Index;

enum class QplibVarType : std::uint8_t { Continuous = 0, Integer = 1, Binary = 2 };

struct QplibInstance {
    std::string name;
    std::array<char, 3> classification{};  // [0] objective [1] variables [2] constraints
    bool maximize = false;

    Index n = 0;  // variables
    Index m = 0;  // general linear constraints

    // Upper triangle of H, 1-based row/col as they appear in the file.
    std::vector<Index> h_row, h_col;
    std::vector<f64> h_val;

    std::vector<f64> g;     // objective gradient, length n
    f64 f_const = 0.0;      // objective constant

    std::vector<Index> a_row, a_col;  // 1-based
    std::vector<f64> a_val;

    // Per-constraint Hessians, as flat quadruples: constraint hc_con[t] has
    // H entry (hc_row[t], hc_col[t]) = hc_val[t], all 1-based, upper triangle.
    // Same objective convention as H: the stored triangle is used AS-IS, so
    // the coefficient of x_r x_c in constraint i is 0.5 * hc_val, NOT hc_val.
    // Empty when the instance has no quadratic constraints.
    std::vector<Index> hc_con, hc_row, hc_col;
    std::vector<f64> hc_val;

    f64 inf_bound = 1e20;   // the file's own infinity sentinel
    std::vector<f64> c_lo, c_hi;      // length m
    std::vector<f64> x_lo, x_hi;      // length n
    std::vector<QplibVarType> var_type;  // length n

    // Name overrides from the trailer; empty string = the file's default
    // name.  Only needed to map a published .sol (which names variables)
    // back onto indices; see qplib_default_var_name().
    std::vector<std::string> var_names;  // length n
    std::vector<std::string> con_names;  // length m

    bool all_binary() const noexcept { return classification[1] == 'B'; }
    // True when any constraint carries a quadratic term.
    bool has_quadratic_constraints() const noexcept { return !hc_val.empty(); }
    // True when the objective has a quadratic term.
    bool has_quadratic_objective() const noexcept { return classification[0] != 'L'; }
    // True when every variable is binary or integer -- no continuous part.
    bool is_discrete() const noexcept;
};

struct QplibReadReport {
    std::size_t lines_consumed = 0;
    std::size_t n_h_entries = 0;
    std::size_t n_a_entries = 0;
    std::size_t n_hc_entries = 0;   // per-constraint Hessian nonzeros
    bool h_has_off_diagonal = false;
    // The trailer (starting point, duals, names) is parsed AFTER the model
    // is complete and never makes the read fail: an instance whose model
    // sections decode is usable even if its trailer is odd.  But a trailer
    // that does not decode, or lines left over after it, means the cursor
    // may have drifted earlier -- the parse-all sweep treats either as a
    // failure of the whole instance.
    std::string trailer_error;          // empty = trailer decoded
    std::size_t lines_total = 0;        // significant lines in the file
    bool fully_consumed() const noexcept {
        return trailer_error.empty() && lines_consumed == lines_total;
    }
};

// Name QPLIB uses for variable j (0-based) when nothing better is known:
// "b<j+2>" for a binary (an integer bounded [0, 1]), "i<j+2>" for a general
// integer, "x<j+2>" for a continuous variable (the +2 is explained in
// qplib.cpp).  This is a fallback for ONE of the three naming families in
// the library; prefer apply_qplib_varnames_file().
std::string qplib_default_var_name(const QplibInstance& q, Index j);

// Fill q.var_names from "<instance path minus .qplib>.varnames", written by
// scripts/fetch_qplib_varnames.py out of the instance's own published GAMS
// source: the name of variable 1..n, one per line, in index order.  Returns
// false (leaving q untouched) when the file is not there.  Throws when it is
// there and does not hold exactly n names -- a silently shifted name map
// would evaluate a DIFFERENT point and still look plausible.
//
// WHY a sidecar and not a rule: the published .sol files name variables, most
// .qplib files carry no names, and the mapping is not one rule.  QPLIB_0031's
// GAMS model declares objvar,x2..x61 for n = 60 (instance variable k is GAMS
// index k+1); QPLIB_10035's declares objvar,x2..x40401 for n = 40401 (the
// objective variable IS variable 1); QPLIB_10057's declares b1..b200,objvar
// for n = 200 (objvar last, model variables from index 1).  One guessed rule
// resolved 273 of 453 instances.
bool apply_qplib_varnames_file(const std::string& instance_path, QplibInstance& q);

// Raw evaluation of a point against the file data (no model conversion):
// objective in the instance's ORIGINAL sense, and the worst violation of
// the constraint rows (linear + quadratic), of the variable bounds, and of
// integrality -- the same three terms as QPLIB's SOLINFEASIBILITY.  Every
// listed Hessian entry, objective or constraint, contributes
// 0.5 * v * x_r * x_c (the format's lower-triangle, NOT symmetrised,
// convention).  Shares no code with the model conversion in
// search/qplib_qp.cpp, so it can check it.
struct QplibPointEval {
    f64 objective = 0.0;
    f64 max_row_violation = 0.0;    // linear and quadratic rows
    f64 max_qc_violation = 0.0;     // rows with any listed Hessian entry
    f64 max_bound_violation = 0.0;
    f64 max_integrality_violation = 0.0;
    f64 max_violation() const noexcept;
};
QplibPointEval qplib_evaluate_point(const QplibInstance& q, const std::vector<f64>& x);

// A published QPLIB solution file (qplib.zib.de/sol/QPLIB_xxxx.sol): lines
// "name value", an "objvar" line carrying the published objective, and
// every variable not listed at zero.  Names are resolved through the
// instance's name overrides, falling back to qplib_default_var_name().
struct QplibSolution {
    std::vector<f64> x;          // length n
    bool has_objvar = false;
    f64 objvar = 0.0;
    std::size_t listed = 0;      // variable lines read
};
// Throws std::runtime_error on an unknown name, a repeated name, or a
// malformed line.  A name whose prefix disagrees with the variable's type
// is accepted only when it matches an explicit override.
QplibSolution read_qplib_solution(const std::string& path, const QplibInstance& q);


// Throws std::runtime_error with the instance name and line number on any
// malformed or out-of-scope input.
QplibInstance read_qplib_file(const std::string& path, QplibReadReport& rep);

}  // namespace sor::io
