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
//   (initial points and name overrides follow; not read -- not needed to
//    build a solvable problem)
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
};

// Throws std::runtime_error with the instance name and line number on any
// malformed or out-of-scope input.
QplibInstance read_qplib_file(const std::string& path, QplibReadReport& rep);

}  // namespace sor::io
