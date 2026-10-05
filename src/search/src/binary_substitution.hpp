#pragma once
#include "sor/search/milp_presolve.hpp"

namespace sor::search::detail {
MilpPresolveResult substitute_binary_relations(const model::LpProblem &lp,
                                               const std::vector<BinaryRelation> &relations,
                                               MilpPresolveStats &stats);
}
