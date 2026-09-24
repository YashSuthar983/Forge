// SOR — internal: the presolved, scaled problem the interior points run on
// (qp_ipm.cpp's IPM and qcqp_local.cpp's nonconvex local solver), and the
// QCQP -> Core builder they share.  Not a public header: engines/src only.
#pragma once

#include "sor/engines/qcqp.hpp"
#include "sor/sparse/csr.hpp"

#include "qp_common.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace sor::engines::ipmc {

// The presolved, scaled problem the iteration actually runs on.
struct Core {
    core::Index n = 0, m = 0;
    std::vector<core::Index> col_of;      // core col -> original col
    std::vector<core::Index> row_of;      // core row -> original row
    sparse::CsrMatrix A;                  // m x n
    // Q upper triangle incl. diagonal, by column: (row, val) lists
    std::vector<std::vector<std::pair<core::Index, f64>>> qcol;
    std::vector<f64> qdiag;               // for Q x products (full symmetric via qcol)
    std::vector<f64> c, l, u, rl, ru;
    f64 offset = 0.0;
    // Quadratic rows (empty for a QP).  For the convex IPM they are
    // canonical -- every one is  a_i'x + 1/2 x'Q_i x <= ru_i  with Q_i PSD
    // (build_qcqp_core negates the concave-above rows; the certificate
    // refused everything else) -- so its Lagrangian Hessian stays PSD.  The
    // nonconvex local solver takes them as they come, bounds and signs.
    // Upper triangle, QuadRow's convention.
    struct Quad {
        core::Index row = 0;              // core row
        std::vector<core::Index> r, c;    // core columns, r <= c
        std::vector<f64> v;
    };
    std::vector<Quad> quad;
};

// A QCQP presolved and scaled into a Core, and the maps back.
struct QcqpBuilt {
    Core k;
    qpc::Scaling sc;
    std::vector<f64> xfix;
    std::vector<std::uint8_t> is_fixed;
    std::vector<core::Index> new_col;
    std::vector<std::int8_t> row_sign;   // -1 where the core row is the negated one
    f64 omega = 1.0;                     // objective scale: core objective = omega * original
};

// Ruiz scaling (with the quadratic rows), fixed-column substitution (a
// Hessian entry on a fixed column becomes a linear term or a constant),
// free rows dropped, objective normalised.  `orientation` (per quad entry,
// may be null): rows with -1 are negated first, so a certified concave
// ">= level" row becomes "convex <= level" (the convex IPM's canonical
// form); the nonconvex local solver passes null and keeps every row as is.
void build_qcqp_core(const QcqpProblem& p0, const std::vector<std::int8_t>* orientation,
                     const QpOptions& opts, QcqpBuilt& B);

}  // namespace sor::engines::ipmc
