#pragma once
#include "sor/model/lp.hpp"
#include "sor/search/conflict_store.hpp"
namespace sor::search {
struct ObjectiveCoreResult {
  f64 lower_bound = -model::kInf;
  // A master pattern is guidance, not an original-model incumbent.
  std::vector<Index> free_paid_columns;
};
ObjectiveCoreResult
price_objective_cores(const model::LpProblem &,
                      const std::vector<char> &integer,
                      const std::vector<std::vector<ConflictLiteral>> &clauses,
                      std::uint64_t work_budget);
} // namespace sor::search
