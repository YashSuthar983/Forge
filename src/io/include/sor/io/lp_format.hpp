// SOR - CPLEX-style LP file format reader.
//
// LAYER L2. Written from the published LP format description (objective,
// Subject To, Bounds, General / Integer / Binary sections, backslash comments); no
// other solver's reader was consulted.
//
// SUPPORTED: linear objective (with constant), linear rows with <=, >=, =
// (and the <, >, =<, => spellings), ranged rows `lo <= expr <= hi`, variables
// on either side of a row, multi-line rows, free / fixed / boxed bounds with
// `inf` / `infinity`, General / Integer / Binary sections, gzip input.
//
// NOT SUPPORTED, and rejected with an error instead of being ignored: quadratic
// terms `[ ... ]`, SOS sections, semi-continuous sections, indicator
// constraints, lazy constraints and user cuts. A solve that silently dropped
// one of those would answer a different model.
//
// The reader reuses MpsReadOptions / MpsReadReport so every loader in the CLI
// has one contract (small-value dropping, integrality relaxation, warnings).
#pragma once

#include "sor/io/mps.hpp"
#include "sor/model/lp.hpp"

#include <istream>
#include <string>

namespace sor::io {

// True for a path ending in ".lp" or ".lp.gz" (case-insensitive).
bool has_lp_extension(const std::string& path);

// Throws std::runtime_error naming the line on malformed or unsupported input.
model::LpProblem read_lp(std::istream&, MpsReadReport&,
                         const MpsReadOptions& = {});
// Detects gzip from the content, like read_mps_file.
model::LpProblem read_lp_file(const std::string& path, MpsReadReport&,
                              const MpsReadOptions& = {});

}  // namespace sor::io
