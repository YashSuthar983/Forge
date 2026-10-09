#pragma once
#include "forge/engines/simplex.hpp"
#include "forge/search/certificate_rows.hpp"
#include "forge/search/mir.hpp"
#include <memory>
namespace forge::search {
// A solve-owned multiplier template. The LP proposes joint nonnegative
// weights; exact source reconstruction certifies the aggregate separately.
class AggregationService {
public:
  void build(const model::LpProblem &, const std::vector<f64> &point,
             const MirOptions &, const engines::SimplexOptions &);
  std::vector<CutRow> separate(const model::LpProblem &,
                               const std::vector<f64> &, const MirOptions &,
                               const engines::SimplexOptions &,
                               MirDiagnostics &);

private:
  struct Source {
    Index row;
    f64 sign;
  };
  model::LpProblem source_, multiplier_;
  std::vector<Source> rows_;
  std::vector<Index> targets_;
  std::vector<std::vector<f64>> witnesses_;
  std::unique_ptr<engines::PrimalCostSession> session_;
  engines::SimplexBasis basis_;
};
} // namespace forge::search
