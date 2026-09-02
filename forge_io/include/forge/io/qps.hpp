// FORGE — QPS reader (MPS + QUADOBJ). Diagonal Q only for the current QP engine.
//
// LAYER L2 — does not depend on engines. The CLI maps this into engines::QpProblem.
#pragma once

#include "forge/io/mps.hpp"
#include "forge/model/lp.hpp"

#include <string>
#include <vector>

namespace forge::io {

struct QpsProblem {
    model::LpProblem linear;
    std::vector<core::f64> q_diag;  // Q for obj = 1/2 x'Qx + c'x
};

struct QpsReadReport : MpsReadReport {
    std::size_t n_quad_entries = 0;
    bool has_off_diagonal = false;
};

QpsProblem read_qps_file(const std::string& path, QpsReadReport& rep,
                         const MpsReadOptions& = {});

}  // namespace forge::io
