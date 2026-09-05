// SOR — MPS reader (free and fixed format).
//
// LAYER L2. Written from the published MPS format description (IBM MPSX
// convention as documented by Netlib and the MIPLIB/QPLIB format notes).
// No reader source from any solver was consulted -- see docs/clean_room_policy.md.
#pragma once

#include "sor/model/lp.hpp"

#include <istream>
#include <string>

namespace sor::io {

struct MpsReadOptions {
    // Fixed-format MPS pads names into fixed columns; free format is whitespace
    // delimited. Free format handles the vast majority of real files, so it is
    // the default. Names containing SPACES require fixed format -- Netlib's
    // `forplan` is the canonical example ("DEDO3 1R", "DEDO3 11").
    bool fixed_format = false;
    // Fail instead of warning when an unrecognised section appears.
    bool strict = false;
};

struct MpsReadReport {
    std::size_t n_rows = 0, n_cols = 0, n_integer = 0;
    core::Offset nnz = 0;
    bool had_ranges = false;
    bool had_objsense_max = false;
    bool used_fixed_format = false;   // set by read_mps_auto on fallback
    std::vector<std::string> warnings;
};

// Throws std::runtime_error with a line number on malformed input.
model::LpProblem read_mps(std::istream&, MpsReadReport&,
                          const MpsReadOptions& = {});
model::LpProblem read_mps_file(const std::string& path, MpsReadReport&,
                               const MpsReadOptions& = {});

// Tries free format, and on ANY parse failure retries in fixed format. This is
// the behaviour a benchmark harness wants: real-world MPS corpora mix both.
// Reports which one succeeded via MpsReadReport::used_fixed_format.
model::LpProblem read_mps_file_auto(const std::string& path, MpsReadReport&,
                                    bool strict = false);

}  // namespace sor::io
