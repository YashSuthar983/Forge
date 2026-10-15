#include "sor/search/column_dominance.hpp"
#include "sor/model/dyadic.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace sor::search {
namespace {
std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

// Both exchange endpoints must belong to the integer box. In particular,
// an approximately-zero positive lower bound requires x=1, not x=0.
bool free_binary(const model::LpProblem &lp, const std::vector<f64> &lo,
                 const std::vector<f64> &hi, Index j) {
  return !lp.is_integer.empty() && lp.is_integer[sz(j)] && lo[sz(j)] > -1.0 &&
         lo[sz(j)] <= 0.0 && hi[sz(j)] >= 1.0 && hi[sz(j)] < 2.0;
}

struct Outside {
  model::DyadicSum minimum, maximum;
  bool min_finite = true, max_finite = true;
  bool initialized = false;
};
Outside outside_activity(const model::LpProblem &lp, Index i, Index j, Index k,
                         f64 aj, f64 ak, const std::vector<f64> &lo,
                         const std::vector<f64> &hi,
                         std::vector<Outside> &activities) {
  auto &cached = activities[sz(i)];
  if (!cached.initialized) {
    const auto &rp = lp.A.pattern.row_ptr();
    const auto &ci = lp.A.pattern.col_idx();
    for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
      const Index col = ci[sz(t)];
      const f64 a = lp.A.vals[sz(t)];
      if (a == 0.0)
        continue;
      const f64 l = a > 0.0 ? lo[sz(col)] : hi[sz(col)];
      const f64 h = a > 0.0 ? hi[sz(col)] : lo[sz(col)];
      if (std::isfinite(l))
        cached.minimum.add_product(a, l);
      else
        cached.min_finite = false;
      if (std::isfinite(h))
        cached.maximum.add_product(a, h);
      else
        cached.max_finite = false;
    }
    cached.initialized = true;
  }
  // Candidate binary endpoints are finite, so removing them cannot change
  // the outside finite-activity flags. Cached rows are updated on every
  // accepted restriction before another pair is considered.
  Outside result = cached;
  result.minimum.add_product(-aj, aj > 0.0 ? lo[sz(j)] : hi[sz(j)]);
  result.maximum.add_product(-aj, aj > 0.0 ? hi[sz(j)] : lo[sz(j)]);
  result.minimum.add_product(-ak, ak > 0.0 ? lo[sz(k)] : hi[sz(k)]);
  result.maximum.add_product(-ak, ak > 0.0 ? hi[sz(k)] : lo[sz(k)]);
  return result;
}

// A feasible point with k=1 maps to k=0 by swapping (j,k)=(0,1)
// to (1,0), or by dropping k from (1,1). Each case needs its own theorem.
// Cost/coefficient comparisons are proof premises, never tolerance tests.
bool exchange_preserves_optimum(const model::LpProblem &lp, Index j, Index k,
                                const std::vector<f64> &lo,
                                const std::vector<f64> &hi,
                                const std::vector<std::vector<Index>> &col_rows,
                                const std::vector<std::vector<f64>> &col_vals,
                                std::vector<Outside> &activities) {
  const f64 sense = lp.maximize ? -1.0 : 1.0;
  const f64 cj = sense * lp.c[sz(j)], ck = sense * lp.c[sz(k)];
  if (cj > ck)
    return false;
  std::map<Index, std::pair<f64, f64>> coef;
  for (std::size_t t = 0; t < col_rows[sz(j)].size(); ++t)
    coef[col_rows[sz(j)][t]].first = col_vals[sz(j)][t];
  for (std::size_t t = 0; t < col_rows[sz(k)].size(); ++t)
    coef[col_rows[sz(k)][t]].second = col_vals[sz(k)][t];

  bool droppable = ck >= 0.0, both_excluded = false;
  for (const auto &[i, pair] : coef) {
    const auto [aj, ak] = pair;
    const bool lower = std::isfinite(lp.row_lo[sz(i)]);
    const bool upper = std::isfinite(lp.row_hi[sz(i)]);
    if ((upper && aj > ak) || (lower && aj < ak))
      return false;
    if (!lower && !upper)
      continue;
    auto outside = outside_activity(lp, i, j, k, aj, ak, lo, hi, activities);
    // In the dropping case, a side harmed by -a_k must already hold
    // throughout the remaining box with j=1, k=0. Ranged/covering rows
    // can participate only when this is actually proved.
    outside.minimum.add(aj);
    outside.maximum.add(aj);
    if ((upper && ak < 0.0 &&
         (!outside.max_finite ||
          outside.maximum.compare(lp.row_hi[sz(i)]) > 0)) ||
        (lower && ak > 0.0 &&
         (!outside.min_finite ||
          outside.minimum.compare(lp.row_lo[sz(i)]) < 0)))
      droppable = false;
    outside.minimum.add(ak);
    outside.maximum.add(ak);
    if ((upper && outside.min_finite &&
         outside.minimum.compare(lp.row_hi[sz(i)]) > 0) ||
        (lower && outside.max_finite &&
         outside.maximum.compare(lp.row_lo[sz(i)]) < 0))
      both_excluded = true;
  }
  return droppable || both_excluded;
}
} // namespace

std::uint64_t apply_binary_column_dominance(const model::LpProblem &lp,
                                            std::vector<f64> &col_lo,
                                            std::vector<f64> &col_hi,
                                            const ColumnDominanceOptions &opts,
                                            ColumnDominanceDiagnostics *diag) {
  if (!opts.enabled || lp.n_cols() == 0)
    return 0;
  if (col_lo.size() != sz(lp.n_cols()) || col_hi.size() != sz(lp.n_cols()))
    return 0;
  ColumnDominanceDiagnostics local;
  auto &d = diag != nullptr ? *diag : local;
  std::vector<std::vector<Index>> col_rows(sz(lp.n_cols()));
  std::vector<std::vector<f64>> col_vals(sz(lp.n_cols()));
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  for (Index i = 0; i < lp.n_rows(); ++i)
    for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
      if (lp.A.vals[sz(t)] == 0.0)
        continue;
      col_rows[sz(ci[sz(t)])].push_back(i);
      col_vals[sz(ci[sz(t)])].push_back(lp.A.vals[sz(t)]);
    }
  std::map<std::vector<Index>, std::vector<Index>> buckets;
  for (Index j = 0; j < lp.n_cols(); ++j) {
    if (!free_binary(lp, col_lo, col_hi, j))
      continue;
    if (col_rows[sz(j)].size() > opts.max_bucket)
      continue;
    buckets[col_rows[sz(j)]].push_back(j);
  }
  std::vector<Outside> activities(sz(lp.n_rows()));
  std::uint64_t fixed = 0;
  for (const auto &[support, group] : buckets) {
    (void)support;
    for (const Index j : group) {
      if (!free_binary(lp, col_lo, col_hi, j))
        continue;
      for (const Index k : group) {
        if (j == k || !free_binary(lp, col_lo, col_hi, k))
          continue;
        ++d.pairs_compared;
        if (!exchange_preserves_optimum(lp, j, k, col_lo, col_hi, col_rows,
                                        col_vals, activities))
          continue;
        // Serialize on the current restricted box. Even tied columns
        // cannot then fix each other: the removed column is never a
        // later exchange representative. Independently collected
        // existential restrictions must not be applied as a batch.
        const f64 old_hi = col_hi[sz(k)];
        for (std::size_t t = 0; t < col_rows[sz(k)].size(); ++t) {
          auto &activity = activities[sz(col_rows[sz(k)][t])];
          if (!activity.initialized)
            continue;
          const f64 ak = col_vals[sz(k)][t];
          (ak > 0.0 ? activity.maximum : activity.minimum)
              .add_product(-ak, old_hi);
        }
        col_hi[sz(k)] = 0.0;
        ++fixed;
        ++d.dominated_fixed;
      }
    }
  }
  return fixed;
}
} // namespace sor::search
