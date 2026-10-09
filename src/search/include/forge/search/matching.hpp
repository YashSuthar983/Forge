#pragma once
#include "forge/model/lp.hpp"
#include "forge/search/prop_trail.hpp"
namespace forge::search {
class MatchingService {
public:
  void build(const model::LpProblem &, const std::vector<char> &integer);
  bool propagate(std::vector<f64> &lo, std::vector<f64> &hi,
                 std::uint64_t &tightened, std::uint64_t work_budget,
                 PropTrail *trail = nullptr, int depth = 0,
                 bool *converged = nullptr, ConflictAntecedents* conflict = nullptr) const;

private:
  struct Edge {
    Index col, value;
  };
  struct Block {
    Index values = 0;
    std::vector<std::vector<Edge>> jobs;
  };
  std::vector<Block> blocks_;
};
} // namespace forge::search
