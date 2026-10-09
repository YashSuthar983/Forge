#pragma once

#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"
#include <memory>

namespace sor::search {

// A primal-guidance workspace, never a source of tree bounds. Its matrix is
// compiled once: coordinates that can have an interior integer target have a
// permanent epigraph column. Moving a target only edits row sides and costs.
class L1ProjectionSession {
public:
  L1ProjectionSession(const model::LpProblem &model,
                      const std::vector<core::f64> &lo,
                      const std::vector<core::f64> &hi,
                      const std::vector<core::Index> &integers,
                      const engines::SimplexOptions &options,
                      core::f64 objective_cutoff = core::kNaN);

  // Objective is objective_weight * sense*c'x + distance_weight*L1(x,target).
  // Target-independent constants are omitted. Returns the expanded point;
  // consumers validate its original-column prefix against their input model.
  core::RawResult solve(const std::vector<core::f64> &target,
                        core::f64 objective_weight, core::f64 distance_weight,
                        const engines::SimplexOptions &options,
                        engines::SimplexDiagnostics &diagnostics);
  bool has_basis() const { return !basis_.basic.empty(); }

private:
  model::LpProblem problem_;
  std::vector<core::f64> objective_;
  std::vector<core::Index> integers_, auxiliary_;
  core::Index original_columns_ = 0, distance_row_base_ = 0;
  std::unique_ptr<engines::PrimalCostSession> session_;
  engines::SimplexBasis basis_;
};

} // namespace sor::search
