#include "forge/search/recourse.hpp"
#include <cmath>
namespace forge::search {
namespace {
std::size_t sz(Index j) { return static_cast<std::size_t>(j); }
} // namespace
void RecourseService::build(const model::LpProblem &lp,
                            const std::vector<char> &integer,
                            const engines::SimplexOptions &opts) {
  session_.reset();
  basis_ = {};
  master_ = integer;
  model_ = lp;
  bool have_master = false, have_recourse = false;
  for (Index j = 0; j < lp.n_cols(); ++j) {
    have_master = have_master || integer[sz(j)];
    have_recourse = have_recourse || !integer[sz(j)];
    model_.c[sz(j)] = integer[sz(j)] ? 0 : (lp.maximize ? -1 : 1) * lp.c[sz(j)];
  }
  model_.maximize = false;
  model_.obj_offset = 0;
  if (have_master && have_recourse)
    session_ = std::make_unique<engines::PrimalCostSession>(model_, opts);
}
std::vector<CutRow>
RecourseService::separate(const std::vector<f64> &x,
                          const engines::SimplexOptions &opts, f64 violation) {
  std::vector<CutRow> cuts;
  if (!session_ || x.size() != model_.c.size())
    return cuts;
  auto lo = model_.col_lo, hi = model_.col_hi;
  for (std::size_t j = 0; j < master_.size(); ++j)
    if (master_[j]) {
      if (!std::isfinite(x[j]) || x[j] < lo[j] || x[j] > hi[j])
        return cuts;
      lo[j] = hi[j] = x[j];
    }
  session_->set_column_bounds(lo, hi);
  engines::SimplexDiagnostics diag;
  engines::SimplexBasis next;
  auto raw = session_->solve(opts, diag, &next,
                             basis_.basic.empty() ? nullptr : &basis_);
  basis_ = std::move(next);
  const bool ray = raw.proposed_status == core::Status::Infeasible &&
                   raw.ray.size() == sz(model_.n_rows());
  const auto &y = ray ? raw.ray : raw.y;
  std::vector<model::DyadicSum> g;
  model::DyadicSum rhs;
  if (!aggregate_geq_exact(model_, y, g, rhs))
    return cuts;
  for (Index j = 0; j < model_.n_cols(); ++j)
    if (!master_[sz(j)]) {
      // Project feasibility, or substitute the recourse objective for g_j.
      // The residual's minimum is over the original recourse box, never
      // over a completion point or a master-conditioned guessed bound.
      model::DyadicSum residual = g[sz(j)];
      residual.negate();
      if (!ray)
        residual.add(model_.c[sz(j)]);
      const f64 endpoint =
          residual.sign() > 0 ? model_.col_lo[sz(j)] : model_.col_hi[sz(j)];
      if (!residual.is_zero()) {
        if (!std::isfinite(endpoint))
          return cuts;
        rhs.add_sum_product(residual, endpoint);
      }
      g[sz(j)] = {};
      if (!ray)
        g[sz(j)].add(model_.c[sz(j)]);
    }
  CutRow cut;
  if (!compile_geq_exact(g, rhs, model_.col_lo, model_.col_hi, cut) ||
      cut.cols.empty())
    return cuts;
  model::DyadicSum activity;
  for (std::size_t q = 0; q < cut.cols.size(); ++q)
    activity.add_product(cut.vals[q], x[sz(cut.cols[q])]);
  if (activity.compare(cut.row_lo - violation) < 0) {
    cut.name = ray ? "BDF_" : "BDO_";
    cut.used_local_bound = false;
    cuts.push_back(std::move(cut));
  }
  return cuts;
}
} // namespace forge::search
