#include "forge/search/l1_projection.hpp"
#include "forge/model/exact.hpp"
#include "forge/sparse/csr.hpp"
#include <cmath>
#include <stdexcept>

namespace forge::search {
using core::f64;
using core::Index;
namespace {
std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
} // namespace

L1ProjectionSession::L1ProjectionSession(const model::LpProblem &input,
                                         const std::vector<f64> &lo,
                                         const std::vector<f64> &hi,
                                         const std::vector<Index> &integers,
                                         const engines::SimplexOptions &options,
                                         f64 objective_cutoff)
    : problem_(input), objective_(input.c), integers_(integers),
      original_columns_(input.n_cols()) {
  const Index n = original_columns_;
  if (lo.size() != sz(n) || hi.size() != sz(n))
    throw std::invalid_argument("L1 projection: box dimensions");
  const f64 sense = input.maximize ? -1.0 : 1.0;
  for (auto &c : objective_)
    c *= sense;
  std::vector<char> seen(sz(n), false);
  for (const Index j : integers_) {
    if (j < 0 || j >= n || seen[sz(j)] || std::isnan(lo[sz(j)]) ||
        std::isnan(hi[sz(j)]) || std::ceil(lo[sz(j)]) > std::floor(hi[sz(j)]))
      throw std::invalid_argument("L1 projection: integer domain");
    seen[sz(j)] = true;
    // First integer strictly above the real lower side. If it lies below
    // the upper side, targets can be interior and need an epigraph.
    const f64 first_inside =
        std::isfinite(lo[sz(j)]) ? std::floor(lo[sz(j)]) + 1.0 : -model::kInf;
    if (!std::isfinite(lo[sz(j)]) || !std::isfinite(hi[sz(j)]) ||
        first_inside < hi[sz(j)])
      auxiliary_.push_back(j);
  }
  const Index na = static_cast<Index>(auxiliary_.size());
  problem_.name += "_l1_projection";
  problem_.maximize = false;
  problem_.obj_offset = 0;
  problem_.c.assign(sz(n + na), 0);
  problem_.col_lo = lo;
  problem_.col_hi = hi;
  problem_.col_lo.resize(sz(n + na), 0);
  problem_.col_hi.resize(sz(n + na), model::kInf);
  problem_.is_integer.assign(sz(n + na), false);
  std::vector<Index> rows, cols;
  std::vector<f64> values;
  rows.reserve(static_cast<std::size_t>(input.nnz()) + sz(n + 4 * na));
  cols.reserve(rows.capacity());
  values.reserve(rows.capacity());
  const auto &rp = input.A.pattern.row_ptr();
  const auto &ci = input.A.pattern.col_idx();
  for (Index i = 0; i < input.n_rows(); ++i)
    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
      rows.push_back(i);
      cols.push_back(ci[static_cast<std::size_t>(k)]);
      values.push_back(input.A.vals[static_cast<std::size_t>(k)]);
    }
  Index m = input.n_rows();
  if (std::isfinite(objective_cutoff)) {
    model::DyadicSum side;
    side.add(sense * objective_cutoff);
    side.add(-sense * input.obj_offset);
    problem_.row_lo.push_back(-model::kInf);
    problem_.row_hi.push_back(side.up());
    for (Index j = 0; j < n; ++j)
      if (objective_[sz(j)] != 0) {
        rows.push_back(m);
        cols.push_back(j);
        values.push_back(objective_[sz(j)]);
      }
    ++m;
  }
  distance_row_base_ = m;
  for (Index q = 0; q < na; ++q) {
    const Index j = auxiliary_[sz(q)], d = n + q;
    for (Index sign = 0; sign < 2; ++sign) {
      problem_.row_lo.push_back(-model::kInf);
      problem_.row_hi.push_back(0);
      rows.push_back(m);
      cols.push_back(j);
      values.push_back(sign == 0 ? 1 : -1);
      rows.push_back(m);
      cols.push_back(d);
      values.push_back(-1);
      ++m;
    }
  }
  problem_.A = sparse::from_triplets(m, n + na, rows, cols, values);
  if (!problem_.col_names.empty())
    problem_.col_names.resize(sz(n + na));
  if (!problem_.row_names.empty())
    problem_.row_names.resize(sz(m));
  problem_.validate();
  session_ = std::make_unique<engines::PrimalCostSession>(problem_, options);
}

core::RawResult L1ProjectionSession::solve(
    const std::vector<f64> &target, f64 objective_weight, f64 distance_weight,
    const engines::SimplexOptions &options, engines::SimplexDiagnostics &diag) {
  const Index n = original_columns_;
  if (target.size() != sz(n) || !std::isfinite(objective_weight) ||
      !std::isfinite(distance_weight) || distance_weight < 0)
    throw std::invalid_argument(
        "L1 projection: objective or target dimensions");
  for (const Index j : integers_)
    if (!std::isfinite(target[sz(j)]) ||
        target[sz(j)] != std::round(target[sz(j)]) ||
        target[sz(j)] < problem_.col_lo[sz(j)] ||
        target[sz(j)] > problem_.col_hi[sz(j)])
      throw std::invalid_argument("L1 projection: illegal integer target");
  for (Index j = 0; j < n; ++j)
    problem_.c[sz(j)] = objective_weight * objective_[sz(j)];
  std::size_t q = 0;
  for (const Index j : integers_) {
    if (q < auxiliary_.size() && auxiliary_[q] == j) {
      problem_.c[sz(n) + q] = distance_weight;
      problem_.row_hi[sz(distance_row_base_) + 2 * q] = target[sz(j)];
      problem_.row_hi[sz(distance_row_base_) + 2 * q + 1] = -target[sz(j)];
      ++q;
    } else {
      problem_.c[sz(j)] += target[sz(j)] == problem_.col_lo[sz(j)]
                               ? distance_weight
                               : -distance_weight;
    }
  }
  session_->set_costs(problem_.c);
  session_->set_row_bounds(problem_.row_lo, problem_.row_hi);
  engines::SimplexBasis next;
  auto result =
      session_->solve(options, diag, &next, has_basis() ? &basis_ : nullptr);
  basis_ = std::move(next);
  return result;
}

} // namespace forge::search
