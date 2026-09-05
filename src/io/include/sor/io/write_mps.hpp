// SOR — write an LpProblem as free-format MPS.
#pragma once

#include "sor/model/lp.hpp"

#include <ostream>
#include <string>

namespace sor::io {

void write_mps(std::ostream& out, const model::LpProblem& p);
void write_mps_file(const std::string& path, const model::LpProblem& p);

// QPS = MPS + QUADOBJ diagonal (obj = 1/2 x' Q x + c'x, Q = diag(q_diag)).
void write_qps(std::ostream& out, const model::LpProblem& p,
               const std::vector<core::f64>& q_diag);
void write_qps_file(const std::string& path, const model::LpProblem& p,
                    const std::vector<core::f64>& q_diag);

}  // namespace sor::io
