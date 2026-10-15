#pragma once
#include "sor/model/lp.hpp"
#include <memory>

namespace sor::search {

// Exact formulation automorphisms, independent of candidate color refinement.
// Action: (Tx)_j = sign[j] * x[image[j]] + shift[j]. Images are bijective,
// signs are +/-1. Checks the live box, types, objective (including its affine
// constant), and the complete row set. Owns its input snapshot.
class SymmetryVerifier {
public:
  explicit SymmetryVerifier(const model::LpProblem &model);
  ~SymmetryVerifier();
  bool verifies(const std::vector<core::Index> &image,
                const std::vector<int> &sign,
                const std::vector<core::f64> &shift) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace sor::search
