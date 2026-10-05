#pragma once
#include "forge/search/milp_presolve.hpp"

namespace forge::search::detail {
MilpPresolveResult substitute_binary_relations(const model::LpProblem &lp,
                                               const std::vector<BinaryRelation> &relations,
                                               MilpPresolveStats &stats);
}
