#pragma once
#include "forge/core/cancel.hpp"
#include "forge/search/prop_trail.hpp"
#include "forge/search/symmetry.hpp"

namespace forge::search {
// Verified generators of a finite permutation group of the supplied model.
// Color cells are proposals; an individualization/refinement search finds
// coupled row/column actions and the exact verifier authorizes every action.
// Every orbit has a lexicographically maximal point in ONE common order.
// Enforcing x >=lex g(x) for every generator therefore preserves an optimum.
// These are representative restrictions, never global feasibility reasons.
class VerifiedSymmetryGroup {
public:
  void build(const model::LpProblem &, const std::vector<Orbit> &candidates,
             double time_limit_s, std::vector<Index> order = {},
             const core::CancelToken *cancel = nullptr);
  bool empty() const { return generators_.empty(); }
  const std::vector<std::vector<Index>> &generators() const {
    return generators_;
  }
  bool propagate(std::vector<f64> &lo, std::vector<f64> &hi, PropTrail *trail,
                 int depth, std::uint64_t &moves) const;
  // Full setwise stabilizer of the retained verified group, using Schreier
  // generators of the orbit of the box coloring. Compositions can preserve
  // the box even when no supplied generator does. Interruption returns the
  // proved partial subgroup and sets complete=false; it never adds a color
  // class as an orbit without group evidence.
  std::vector<Orbit> stabilizer_orbits(const std::vector<f64> &lo,
                                       const std::vector<f64> &hi,
                                       double time_limit_s = 0,
                                       const core::CancelToken* cancel = nullptr,
                                       bool* complete = nullptr) const;

private:
  std::vector<Index> order_;
  std::vector<std::vector<Index>> generators_;
  std::vector<std::vector<std::pair<Index, Index>>> leaders_;
};
} // namespace forge::search
