#pragma once

#include "forge/model/lp.hpp"

namespace forge::engines {

// True when every column has at most two nonzero entries with opposite signs,
// as in a node-arc incidence matrix (up to scaling).
bool detect_network_structure(const model::LpProblem& problem);

}  // namespace forge::engines
