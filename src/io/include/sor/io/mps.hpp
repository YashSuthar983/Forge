// SOR - MPS reader (free and fixed format).
//
// LAYER L2. Written from the published MPS format description (IBM MPSX
// convention as documented by Netlib and the MIPLIB/QPLIB format notes).
// No reader source from another solver was used; see docs/architecture.md §8.
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

    // Drop matrix coefficients with |a| <= small_matrix_value, and report how
    // many were dropped. A benchmark that compares two solvers must give both
    // the same matrix: the external oracle discards coefficients at or below
    // its own 1e-9 threshold, so a SOR that keeps them is not solving the same
    // model. Set to 0.0 to keep every coefficient the file contains.
    core::f64 small_matrix_value = 0.0;

    // Read the model as a continuous LP: keep the columns and bounds but
    // discard integrality. This is how a MIPLIB root LP relaxation is formed,
    // and it must happen at READ time from the ORIGINAL file -- never from a
    // model some other solver rewrote.
    bool relax_integrality = false;
};

struct MpsReadReport {
    // Reader self-timing (ms) and line count. See read_mps() for why the
    // reader measures itself rather than relying on a profiler.
    double ms_parse = 0.0;
    double ms_assemble = 0.0;
    std::size_t lines_read = 0;
    std::size_t n_rows = 0, n_cols = 0, n_integer = 0;
    core::Offset nnz = 0;
    bool had_ranges = false;
    bool had_objsense_max = false;
    bool used_fixed_format = false;   // set by read_mps_auto on fallback
    bool used_gzip = false;           // input was a gzip stream
    bool relaxed_integrality = false; // integrality was present and discarded
    // Coefficients dropped by MpsReadOptions::small_matrix_value, and the
    // largest magnitude among them, so a manifest can record exactly what the
    // threshold removed rather than just that it was enabled.
    std::size_t small_values_dropped = 0;
    core::f64 largest_small_value_dropped = 0.0;
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
model::LpProblem read_mps_file_auto(const std::string& path, MpsReadReport&,
                                    const MpsReadOptions&);

}  // namespace sor::io
