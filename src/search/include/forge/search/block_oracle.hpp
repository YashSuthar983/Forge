#pragma once
#include "forge/search/cuts.hpp"
namespace forge::search {
// Finite projected outer relaxations of the admitted row-feasibility band. A
// block owns its coordinates; it only projects original rows using exact
// outside box activities. Rows/costs are not summed across overlapping blocks.
// Each pricing result independently certifies an inequality in the original
// column space.
class FiniteBlockOracle {
public:
  void build(const model::LpProblem &lp,
             const std::vector<char> &required_integer, int max_columns,
             std::uint64_t work_budget, f64 feasibility_tolerance = 0.0);
  bool infeasible() const { return infeasible_; }
  std::vector<CutRow> separate(const model::LpProblem &lp,
                               const std::vector<f64> &x, int max_cuts,
                               f64 violation_min,
                               std::uint64_t work_budget) const;
  // Joint integer moves, not a kernel-basis optimality claim. Each full
  // candidate is checked against all original rows, bounds and costs.
  bool augment(const model::LpProblem &, std::vector<f64> &point,
               std::uint64_t work_budget) const;

private:
  struct Block {
    std::vector<Index> cols;
    std::vector<std::vector<f64>> points;
  };
  std::vector<Block> blocks_;
  bool infeasible_ = false;
};
} // namespace forge::search
