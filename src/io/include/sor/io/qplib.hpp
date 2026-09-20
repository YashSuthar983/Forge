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
// SCOPE: classification[2] (the constraint-type letter) must be one of
// N (none), B (bounds only) or L (linear). Types D/C/Q are quadratic
// constraints, which carry extra per-constraint Hessian sections this
// reader does not decode. It fails fast and specifically on those rather
// than desyncing the line cursor and producing a confusing downstream
// error. That is a stated scope limit, not a silent gap.
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

    f64 inf_bound = 1e20;   // the file's own infinity sentinel
    std::vector<f64> c_lo, c_hi;      // length m
    std::vector<f64> x_lo, x_hi;      // length n
    std::vector<QplibVarType> var_type;  // length n

    bool all_binary() const noexcept { return classification[1] == 'B'; }
    // True when every variable is binary or integer -- no continuous part.
    bool is_discrete() const noexcept;
};

struct QplibReadReport {
    std::size_t lines_consumed = 0;
    std::size_t n_h_entries = 0;
    std::size_t n_a_entries = 0;
    bool h_has_off_diagonal = false;
};

// Throws std::runtime_error with the instance name and line number on any
// malformed or out-of-scope input.
QplibInstance read_qplib_file(const std::string& path, QplibReadReport& rep);

}  // namespace sor::io
