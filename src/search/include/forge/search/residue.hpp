#pragma once
#include "forge/model/lp.hpp"
#include "forge/search/prop_trail.hpp"
#include <array>
#include <map>
#include <memory>
namespace forge::search {
// Immutable sparse field-elimination journal. Its equations remain available
// after interval tightening: interior residue holes are never lost at children.
struct ResidueState {
  std::shared_ptr<const int> generation;
  std::vector<std::array<unsigned char, 4>> masks;
  std::vector<f64> lo, hi;
};
class ResidueSystem {
public:
  void build(const model::LpProblem &, const std::vector<char> &integer,
             std::uint64_t work_budget);
  bool infeasible() const { return infeasible_; }
  bool propagate(std::vector<f64> &lo, std::vector<f64> &hi,
                 std::uint64_t &tightened, PropTrail *trail = nullptr,
                 int depth = 0, const std::vector<Index> *seeds = nullptr,
                 ResidueState *state = nullptr,
                 ConflictAntecedents* conflict = nullptr) const;

private:
  struct Equation {
    int prime = 2, rhs = 0;
    std::map<Index, int> terms;
  };
  std::vector<Equation> equations_;
  std::vector<std::vector<Index>> incident_;
  bool infeasible_ = false;
  std::vector<Index> columns_, position_;
  std::shared_ptr<const int> generation_;
};
} // namespace forge::search
