// SOR - human-readable solution export, keyed by the ORIGINAL names in the
// model file.
//
// LAYER L2.
//
// The --solution-out format next door is deliberately dumb because sor_check
// has to re-read it; it writes bare index vectors. That is unreadable to the
// person who wrote the model: a planner who asks "how much goes to blend A"
// gets position 847 of a whitespace-separated list. This writer answers in
// their vocabulary instead -- one row per named variable and per named
// constraint, with the bound each one sits against -- and emits CSV so it
// opens directly in the spreadsheet where planning models actually live.
//
// It is an EXPORT, not an interchange format: nothing in this codebase reads
// it back. Claims are still verified from the solution file, not from here.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <iosfwd>

namespace sor::io {

// Emits, in order:
//   summary rows  - status, proof level, objective (in the file's own sense)
//   variable rows - name, value, lower, upper, which bound it sits on
//   constraint rows - name, row activity (Ax)_i, lower, upper, bound hit
//
// Columns are kind,name,value,lower,upper,at_bound. Infinite bounds print as
// empty cells rather than "inf", because a spreadsheet treats an empty cell as
// "no limit" and would treat the text "inf" as a label.
//
// A variable with no name in the file (possible when a reader synthesises
// columns) is written as its index, so every row is still addressable.
void write_solution_csv(std::ostream& out, const model::LpProblem& p,
                        const core::SolveResult& r);

}  // namespace sor::io
