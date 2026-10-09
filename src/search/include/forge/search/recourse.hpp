#pragma once
#include "forge/engines/simplex.hpp"
#include "forge/search/certificate_rows.hpp"
#include <memory>
namespace forge::search {
// Original-space Benders service. The monolithic LP remains the coverage
// authority; recourse points are guidance, while dual/ray aggregates produce
// supporting rows checked against the GLOBAL recourse domain.
class RecourseService {
public:
  void build(const model::LpProblem &, const std::vector<char> &integer,
             const engines::SimplexOptions &);
  std::vector<CutRow> separate(const std::vector<f64> &master_point,
                               const engines::SimplexOptions &,
                               f64 violation_min);

private:
  model::LpProblem model_;
  std::vector<char> master_;
  std::unique_ptr<engines::PrimalCostSession> session_;
  engines::SimplexBasis basis_;
};
} // namespace forge::search
