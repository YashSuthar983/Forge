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

// Quadratic objective entries exactly as the file wrote them, by column
// index. QUADOBJ lists one triangle of Q; QMATRIX lists the full symmetric
// matrix. A file may use one of them, not both. read_qps_file assembles Q.
struct QuadraticTerms {
    bool full_matrix = false;  // the entries came from QMATRIX
    std::vector<core::Index> rows, cols;
    std::vector<core::f64> vals;
    std::vector<std::size_t> lines;  // source line of each entry, for errors
    void clear() { *this = QuadraticTerms{}; }
};

struct MpsReadOptions {
    // Fixed-format MPS pads names into fixed columns; free format is whitespace
    // delimited. Free format handles the vast majority of real files, so it is
    // the default. Names containing SPACES require fixed format -- Netlib's
    // `forplan` is the canonical example ("DEDO3 1R", "DEDO3 11").
    bool fixed_format = false;
    // Fail instead of warning when an unrecognised section appears, or a
    // quadratic section (QUADOBJ/QMATRIX) that no caller is going to read.
    // Any route that reports a result about the file must read strictly: a
    // dropped section means the solved model is not the file's model.
    bool strict = false;
    // When set, the quadratic objective sections are read into it (see
    // QuadraticTerms) in the same pass as the rest of the file, gzip
    // included. When null they are refused (strict) or dropped with a warning.
    QuadraticTerms* quadratic = nullptr;

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
    // Bound and row-side magnitudes at or beyond this read as infinite: a
    // lower side <= -infinite_bound becomes -inf and an upper side >=
    // infinite_bound becomes +inf. Files routinely spell "no bound" as 1e20
    // or 1e30; kept finite, such a bound is a huge box that the simplex
    // flips to and that poisons every value it touches. 1e20 is the CPLEX
    // and HiGHS convention. Zero (or a non-finite value) disables it, so the
    // file's numbers are taken literally. Matrix coefficients are unaffected.
    core::f64 infinite_bound = 1e20;
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
    // Finite bounds and row sides read as infinite by
    // MpsReadOptions::infinite_bound.
    std::size_t infinite_bounds_read = 0;
    // Matrix entries given more than once for the same row and column (each
    // extra one counts), summed exactly and rounded once; and entries that
    // were exactly zero, as written or after that sum, which are not stored.
    std::size_t duplicate_entries_summed = 0;
    std::size_t zero_entries_dropped = 0;
    // Stored matrix coefficients below kTinyMatrixValue in magnitude, and the
    // smallest magnitude stored. They are kept (the solved model is the file's
    // model), but equilibration scales a column holding only such entries by
    // their inverse, so the reader warns and names --small-matrix-value.
    // Free-format data lines that started in column 1 (read as data, not as
    // an unknown section header).
    std::size_t column_one_data_lines = 0;
    // RHS, RANGES and BOUNDS lines of a second (or later) named set, which a
    // single model does not use; and values given twice for the same row or
    // column side within the set used (the last one stands).
    std::size_t ignored_set_lines = 0;
    std::size_t repeated_entries = 0;
    std::size_t tiny_entries = 0;
    core::f64 smallest_entry = 0.0;
    std::vector<std::string> warnings;
};

// Throws std::runtime_error with a line number on malformed input.
model::LpProblem read_mps(std::istream&, MpsReadReport&,
                          const MpsReadOptions& = {});
model::LpProblem read_mps_file(const std::string& path, MpsReadReport&,
                               const MpsReadOptions& = {});

// Routes .lp/.lp.gz to the LP reader. For MPS, tries free format, and on ANY
// parse failure retries in fixed format. This is
// the behaviour a benchmark harness wants: real-world MPS corpora mix both.
// Reports which one succeeded via MpsReadReport::used_fixed_format.
model::LpProblem read_mps_file_auto(const std::string& path, MpsReadReport&,
                                    bool strict = false);
model::LpProblem read_mps_file_auto(const std::string& path, MpsReadReport&,
                                    const MpsReadOptions&);

// The constraint matrix from (row, column, value) triplets as a reader
// collected them. Repeated (row, column) pairs are summed exactly and rounded
// once, so the value does not depend on the order they appeared in, and
// exact zeros are dropped rather than stored as entries: a structural zero
// makes an empty column look occupied and costs every sparse kernel. Counts
// go to the report, with a warning for duplicates, which are almost always a
// modelling or export error. Shared by the MPS and LP-format readers.
inline constexpr core::f64 kTinyMatrixValue = 1e-9;

sparse::CsrMatrix assemble_matrix(core::Index n_rows, core::Index n_cols,
                                  const std::vector<core::Index>& rows,
                                  const std::vector<core::Index>& cols,
                                  const std::vector<core::f64>& vals,
                                  MpsReadReport& rep);

// Apply MpsReadOptions::infinite_bound to the column bounds and row sides of
// a freshly read model, counting the conversions in the report (and adding a
// warning when there were any). Shared by the MPS and LP-format readers so
// both read the same file content as the same model.
void apply_infinite_bound(model::LpProblem& p, const MpsReadOptions& opt,
                          MpsReadReport& rep);

}  // namespace sor::io
