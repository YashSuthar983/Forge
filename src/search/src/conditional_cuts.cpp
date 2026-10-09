#include "sor/search/conditional_cuts.hpp"
#include "sor/model/dyadic.hpp"
#include <cmath>
namespace sor::search {
namespace {
std::size_t sz(Index j) { return static_cast<std::size_t>(j); }
} // namespace
bool objective_probe_cut(const model::LpProblem &lp, Index binary, f64 b0,
                         f64 b1, CutRow &cut) {
  if (binary < 0 || binary >= lp.n_cols() || lp.is_integer.empty() ||
      !lp.is_integer[sz(binary)] || lp.col_lo[sz(binary)] < 0 ||
      lp.col_hi[sz(binary)] > 1 || !std::isfinite(b0) || !std::isfinite(b1))
    return false;
  model::DyadicSum slope;
  slope.add(b1);
  slope.add(-b0);
  const f64 s = slope.down();
  if (!std::isfinite(s))
    return false;
  // a = c_min - s*e_j. Round this one coefficient up, then
  // lower the RHS by the worst endpoint error. Other coefficients are
  // copied exactly from the modeled objective.
  const f64 sense = lp.maximize ? -1 : 1;
  model::DyadicSum coefficient;
  coefficient.add(sense * lp.c[sz(binary)]);
  coefficient.add(-s);
  const f64 a = coefficient.up();
  if (!std::isfinite(a))
    return false;
  model::DyadicSum rhs;
  rhs.add(b0);
  rhs.add(-sense * lp.obj_offset);
  // Upward coefficient rounding on a nonnegative binary only relaxes >=.
  cut = CutRow{};
  for (Index j = 0; j < lp.n_cols(); ++j) {
    const f64 v = j == binary ? a : sense * lp.c[sz(j)];
    if (v == 0)
      continue;
    cut.cols.push_back(j);
    cut.vals.push_back(v);
  }
  cut.row_lo = rhs.down();
  cut.exact = true;
  cut.used_local_bound = true;
  cut.name = "GEO_";
  return !cut.cols.empty() && std::isfinite(cut.row_lo);
}
std::vector<CutRow> strengthen_conditional_rows(
    const model::LpProblem &lp, const std::vector<f64> &x,
    const std::vector<char> &integer, int cap, f64 violation) {
  std::vector<CutRow> cuts;
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  for (Index i = 0; i < lp.n_rows() && static_cast<int>(cuts.size()) < cap; ++i)
    for (int side : {1, -1}) {
      const f64 rhs = side == 1 ? lp.row_hi[sz(i)] : -lp.row_lo[sz(i)];
      if (!std::isfinite(rhs))
        continue;
      model::DyadicSum all_upper;
      int infinite = 0;
      for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = side * lp.A.vals[sz(k)];
        if (a == 0)
          continue;
        const f64 bound = a > 0 ? lp.col_hi[sz(j)] : lp.col_lo[sz(j)];
        if (!std::isfinite(bound))
          ++infinite;
        else
          all_upper.add_product(a, bound);
      }
      if (infinite)
        continue;
      for (auto kk = rp[sz(i)];
           kk < rp[sz(i) + 1] && static_cast<int>(cuts.size()) < cap; ++kk) {
        const Index b = ci[sz(kk)];
        const f64 a = side * lp.A.vals[sz(kk)];
        if (sz(b) >= integer.size() || !integer[sz(b)] ||
            lp.col_lo[sz(b)] != 0 || lp.col_hi[sz(b)] != 1 || a == 0)
          continue;
        model::DyadicSum upper = all_upper;
        if (a > 0)
          upper.add(-a);
        // For a>0, binary=0 is the relaxed endpoint; for a<0 it
        // is binary=1. Keep the other endpoint's original restriction.
        model::DyadicSum inactive = upper;
        if (a < 0)
          inactive.add(a);
        if (inactive.compare(rhs) >= 0)
          continue;
        model::DyadicSum delta;
        delta.add(rhs);
        inactive.negate();
        delta.add_sum(inactive);
        const f64 d = std::min(std::fabs(a), delta.down());
        if (!(d > 0))
          continue;
        model::DyadicSum changed;
        changed.add(a);
        changed.add(a > 0 ? -d : d);
        const f64 slope =
            changed.down(); // <= : a smaller binary coefficient relaxes
        model::DyadicSum delivered;
        delivered.add(rhs);
        if (a > 0)
          delivered.add(-d);
        // RHS down could strengthen accidentally. Use its upper endpoint.
        CutRow cut;
        cut.row_hi = delivered.up();
        cut.exact = true;
        cut.used_local_bound = true;
        cut.name = "LCM_";
        model::DyadicSum activity;
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
          const Index j = ci[sz(k)];
          const f64 v = k == kk ? slope : side * lp.A.vals[sz(k)];
          if (v == 0)
            continue;
          cut.cols.push_back(j);
          cut.vals.push_back(v);
          activity.add_product(v, x[sz(j)]);
        }
        if (activity.compare(cut.row_hi + violation) > 0)
          cuts.push_back(std::move(cut));
      }
    }
  return cuts;
}
} // namespace sor::search
