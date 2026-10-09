#include "forge/search/aggregation.hpp"
#include <algorithm>
#include <cmath>
namespace forge::search {
namespace {
std::size_t sz(Index j) { return static_cast<std::size_t>(j); }
} // namespace
void AggregationService::build(const model::LpProblem &lp,
                               const std::vector<f64> &x,
                               const MirOptions &opts,
                               const engines::SimplexOptions &engine) {
  session_.reset();
  basis_ = {};
  rows_.clear();
  targets_.clear();
  witnesses_.clear();
  source_ = lp;
  if (!opts.aggregate || x.size() != lp.c.size())
    return;
  std::vector<std::pair<f64, Index>> targets;
  for (Index j = 0; j < lp.n_cols(); ++j)
    if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) {
      const f64 distance =
          std::min(x[sz(j)] - lp.col_lo[sz(j)], lp.col_hi[sz(j)] - x[sz(j)]);
      if (std::isfinite(distance) && distance > opts.bad_variable_min_distance)
        targets.emplace_back(distance, j);
    }
  std::sort(targets.begin(), targets.end(), std::greater<>());
  for (const auto &[distance, j] : targets) {
    (void)distance;
    if (static_cast<int>(targets_.size()) >= opts.max_aggregations)
      break;
    targets_.push_back(j);
  }
  if (targets_.empty())
    return;
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  std::vector<std::pair<f64, Source>> candidates;
  for (Index i = 0; i < lp.n_rows(); ++i) {
    if (static_cast<std::size_t>(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len)
      continue;
    bool useful = false;
    f64 activity = 0;
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
      activity += lp.A.vals[sz(k)] * x[sz(ci[sz(k)])];
      useful = useful || std::find(targets_.begin(), targets_.end(),
                                   ci[sz(k)]) != targets_.end();
    }
    if (!useful)
      continue;
    for (f64 sign : {1.0, -1.0}) {
      const f64 rhs = sign > 0 ? lp.row_lo[sz(i)] : -lp.row_hi[sz(i)];
      if (std::isfinite(rhs))
        candidates.push_back({std::max(0.0, sign * activity - rhs), {i, sign}});
    }
  }
  std::stable_sort(
      candidates.begin(), candidates.end(),
      [](const auto &a, const auto &b) { return a.first < b.first; });
  for (const auto &candidate : candidates) {
    if (static_cast<int>(rows_.size()) >= opts.max_aggregations)
      break;
    rows_.push_back(candidate.second);
  }
  if (rows_.size() < 2)
    return;
  const Index weights = static_cast<Index>(rows_.size()),
              n = weights + static_cast<Index>(targets_.size());
  std::vector<Index> tr, tc;
  std::vector<f64> tv;
  for (Index q = 0; q < weights; ++q) {
    tr.push_back(0);
    tc.push_back(q);
    tv.push_back(1);
  }
  for (Index t = 0; t < static_cast<Index>(targets_.size()); ++t) {
    for (Index q = 0; q < weights; ++q) {
      const auto src = rows_[sz(q)];
      f64 a = 0;
      for (auto k = rp[sz(src.row)]; k < rp[sz(src.row) + 1]; ++k)
        if (ci[sz(k)] == targets_[sz(t)])
          a += src.sign * lp.A.vals[sz(k)];
      if (a != 0)
        for (int sign : {1, -1}) {
          tr.push_back(1 + 2 * t + (sign < 0));
          tc.push_back(q);
          tv.push_back(sign * a);
        }
    }
    for (Index r : {1 + 2 * t, 2 + 2 * t}) {
      tr.push_back(r);
      tc.push_back(weights + t);
      tv.push_back(-1);
    }
  }
  multiplier_ = {};
  multiplier_.A = sparse::from_triplets(
      1 + 2 * static_cast<Index>(targets_.size()), n, tr, tc, tv);
  multiplier_.c.assign(sz(n), 0);
  multiplier_.col_lo.assign(sz(n), 0);
  multiplier_.col_hi.assign(sz(n), model::kInf);
  multiplier_.row_lo.assign(sz(multiplier_.n_rows()), -model::kInf);
  multiplier_.row_hi.assign(sz(multiplier_.n_rows()), 0);
  multiplier_.row_lo[0] = multiplier_.row_hi[0] = 1;
  auto options = engine;
  options.presolve = false;
  session_ = std::make_unique<engines::PrimalCostSession>(multiplier_, options);
}
std::vector<CutRow>
AggregationService::separate(const model::LpProblem &lp,
                             const std::vector<f64> &x, const MirOptions &opts,
                             const engines::SimplexOptions &engine,
                             MirDiagnostics &diagnostics) {
  std::vector<CutRow> cuts;
  if (!session_ || lp.n_cols() != source_.n_cols())
    return cuts;
  const auto &rp = source_.A.pattern.row_ptr();
  const auto &ci = source_.A.pattern.col_idx();
  for (std::size_t q = 0; q < rows_.size(); ++q) {
    const auto src = rows_[q];
    f64 activity = 0;
    for (auto k = rp[sz(src.row)]; k < rp[sz(src.row) + 1]; ++k)
      activity += src.sign * source_.A.vals[sz(k)] * x[sz(ci[sz(k)])];
    const f64 rhs = src.sign > 0 ? source_.row_lo[sz(src.row)]
                                 : -source_.row_hi[sz(src.row)];
    multiplier_.c[q] = std::max(0.0, activity - rhs);
  }
  for (std::size_t t = 0; t < targets_.size(); ++t) {
    const auto j = sz(targets_[t]);
    multiplier_.c[rows_.size() + t] =
        std::max(0.0, std::min(x[j] - lp.col_lo[j], lp.col_hi[j] - x[j]));
  }
  session_->set_costs(multiplier_.c);
  engines::SimplexDiagnostics diag;
  engines::SimplexBasis next;
  auto options = engine;
  options.presolve = false;
  const auto raw = session_->solve(options, diag, &next,
                                   basis_.basic.empty() ? nullptr : &basis_);
  basis_ = std::move(next);
  if (raw.x.size() == multiplier_.c.size()) {
    std::vector<f64> y(sz(source_.n_rows()), 0);
    bool finite = true;
    for (std::size_t q = 0; q < rows_.size(); ++q) {
      const f64 weight = raw.x[q];
      if (!std::isfinite(weight)) {
        finite = false;
        break;
      }
      if (weight > 0)
        y[sz(rows_[q].row)] += rows_[q].sign * weight;
    }
    // An equality admits both signs. A ranged row can have both positive
    // and negative proposals: netting them discards a valid constant slack,
    // yielding a weaker but still valid signed row combination.
    if (finite &&
        std::find(witnesses_.begin(), witnesses_.end(), y) == witnesses_.end())
      witnesses_.push_back(std::move(y));
  }
  for (const auto &y : witnesses_) {
    std::vector<model::DyadicSum> g;
    model::DyadicSum rhs;
    CutRow base, cut;
    if (!aggregate_geq_exact(source_, y, g, rhs) ||
        !compile_geq_exact(g, rhs, lp.col_lo, lp.col_hi, base) ||
        base.cols.empty())
      continue;
    if (apply_cmir_geq(lp, base.cols, base.vals, base.row_lo, x, lp.col_lo,
                       lp.col_hi, opts, true, cut.cols, cut.vals, cut.row_lo,
                       diagnostics)) {
      cut.name = "MIR_JOINT_";
      cut.used_local_bound = false;
      cuts.push_back(std::move(cut));
    }
  }
  return cuts;
}
} // namespace forge::search
