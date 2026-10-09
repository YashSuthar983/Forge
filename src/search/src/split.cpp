#include "forge/search/split.hpp"
#include "forge/model/dyadic.hpp"
#include <cmath>
#include <cstdint>
#include <numeric>
namespace forge::search {
std::optional<IntegerSplit>
sparse_integer_split(const model::LpProblem &lp,
                     const std::vector<core::f64> &x,
                     const std::vector<char> &required, core::f64 tol) {
  const auto sz = [](core::Index i) { return static_cast<std::size_t>(i); };
  if (x.size() != sz(lp.n_cols()) || required.size() != x.size())
    return std::nullopt;
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  std::optional<IntegerSplit> best;
  double best_score = 0.0;
  for (core::Index i = 0; i < lp.n_rows(); ++i) {
    IntegerSplit candidate;
    std::int64_t gcd = 0;
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
      const auto j = ci[sz(k)];
      const double a = lp.A.vals[sz(k)];
      if (!required[sz(j)] || a == 0.0 || lp.col_lo[sz(j)] == lp.col_hi[sz(j)])
        continue;
      if (!std::isfinite(a) || a != std::trunc(a) || std::fabs(a) >= 0x1p52)
        continue;
      candidate.direction.cols.push_back(j);
      candidate.direction.vals.push_back(a);
      gcd = std::gcd(gcd, static_cast<std::int64_t>(std::fabs(a)));
    }
    if (candidate.direction.cols.size() < 2 || gcd == 0)
      continue;
    model::DyadicSum activity;
    for (std::size_t k = 0; k < candidate.direction.cols.size(); ++k) {
      candidate.direction.vals[k] /= static_cast<double>(gcd);
      activity.add_product(candidate.direction.vals[k],
                           x[sz(candidate.direction.cols[k])]);
    }
    const double v = activity.nearest();
    if (!std::isfinite(v) || std::fabs(v) >= 0x1p52)
      continue;
    double floor = std::floor(v);
    if (activity.compare(floor) < 0)
      --floor;
    else if (activity.compare(floor + 1.0) >= 0)
      ++floor;
    candidate.floor = floor;
    candidate.fraction = v - floor;
    const double fraction =
        std::min(candidate.fraction, 1.0 - candidate.fraction);
    if (!(fraction > tol))
      continue;
    const double score =
        fraction /
        std::sqrt(static_cast<double>(candidate.direction.cols.size()));
    if (score > best_score) {
      candidate.direction.exact = candidate.direction.compiled = true;
      candidate.direction.name = "SPLIT";
      best_score = score;
      best = std::move(candidate);
    }
  }
  return best;
}
} // namespace forge::search
