#include "sor/search/certificate_rows.hpp"
#include <cmath>
namespace sor::search {
bool compile_geq_exact(const std::vector<model::DyadicSum> &g,
                       model::DyadicSum rhs, const std::vector<f64> &lo,
                       const std::vector<f64> &hi, CutRow &row) {
  if (g.size() != lo.size() || g.size() != hi.size())
    return false;
  row = CutRow{};
  for (std::size_t j = 0; j < g.size(); ++j) {
    if (g[j].is_zero())
      continue;
    f64 a;
    if (!g[j].representable(a)) {
      a = std::isfinite(lo[j]) ? g[j].up() : g[j].down();
      if (!std::isfinite(a))
        return false;
      auto delta = g[j];
      delta.negate();
      delta.add(a);
      const f64 endpoint = delta.sign() > 0 ? lo[j] : hi[j];
      if (!std::isfinite(endpoint))
        return false;
      rhs.add_sum_product(delta, endpoint);
    }
    if (!std::isfinite(a))
      return false;
    if (a != 0) {
      row.cols.push_back(static_cast<Index>(j));
      row.vals.push_back(a);
    }
  }
  row.row_lo = rhs.down();
  row.exact = true;
  return std::isfinite(row.row_lo);
}
bool aggregate_geq_exact(const model::LpProblem &lp, const std::vector<f64> &y,
                         std::vector<model::DyadicSum> &g,
                         model::DyadicSum &beta) {
  if (y.size() != static_cast<std::size_t>(lp.n_rows()))
    return false;
  g.assign(static_cast<std::size_t>(lp.n_cols()), {});
  beta = {};
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  for (Index i = 0; i < lp.n_rows(); ++i) {
    const auto q = static_cast<std::size_t>(i);
    const f64 yi = y[q];
    if (!std::isfinite(yi))
      return false;
    if (yi == 0)
      continue;
    const f64 side = yi > 0 ? lp.row_lo[q] : lp.row_hi[q];
    if (!std::isfinite(side))
      continue; // zero this unsupported multiplier
    beta.add_product(yi, side);
    for (auto k = rp[q]; k < rp[q + 1]; ++k)
      g[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])].add_product(
          yi, lp.A.vals[static_cast<std::size_t>(k)]);
  }
  return true;
}
} // namespace sor::search
