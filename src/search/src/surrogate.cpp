#include "sor/search/surrogate.hpp"
#include "sor/model/dyadic.hpp"
#include <algorithm>
#include <cmath>
namespace sor::search {
namespace {
std::size_t sz(Index j) { return static_cast<std::size_t>(j); }
} // namespace
std::vector<CutRow>
separate_surrogate_knapsacks(const model::LpProblem &lp,
                             const std::vector<f64> &x,
                             const std::vector<char> &integer, int cap,
                             f64 violation, std::uint64_t budget) {
  std::vector<CutRow> cuts;
  const f64 sense = lp.maximize ? -1 : 1;
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  struct Item {
    Index col;
    f64 weight, cost;
    bool complement;
  };
  struct State {
    model::DyadicSum cost;
    bool reachable = false;
  };
  for (Index row = 0;
       row < lp.n_rows() && budget > 0 && static_cast<int>(cuts.size()) < cap;
       ++row)
    for (int sign : {1, -1}) {
      const f64 side = sign == 1 ? lp.row_lo[sz(row)] : -lp.row_hi[sz(row)];
      if (!std::isfinite(side))
        continue;
      std::vector<Item> items;
      model::DyadicSum rhs, constant;
      rhs.add(side);
      bool eligible = true;
      for (auto k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = sign * lp.A.vals[sz(k)];
        if (a == 0)
          continue;
        if (integer[sz(j)] && lp.col_lo[sz(j)] == 0 && lp.col_hi[sz(j)] == 1) {
          const f64 c = sense * lp.c[sz(j)];
          const bool complement = a < 0;
          if (complement) {
            rhs.add(-a);
            constant.add(c);
          }
          items.push_back({j, std::fabs(a), complement ? -c : c, complement});
        } else {
          // The omitted activity can take its maximum, a necessary
          // projection. No incumbent-conditioned recourse is substituted.
          const f64 endpoint = a > 0 ? lp.col_hi[sz(j)] : lp.col_lo[sz(j)];
          if (!std::isfinite(endpoint)) {
            eligible = false;
            break;
          }
          rhs.add_product(-a, endpoint);
        }
      }
      if (!eligible || items.size() < 2 || rhs.sign() <= 0)
        continue;
      const std::uint64_t states = std::min<std::uint64_t>(
          budget / (2 * items.size()), (32u * 1024u * 1024u) / sizeof(State));
      if (states < 2)
        continue;
      // Power-of-two scaling affects only oracle width, never validity.
      const f64 quotient = rhs.quotient_up(static_cast<f64>(states - 1));
      int exponent = 0;
      std::frexp(quotient, &exponent);
      const f64 scale = std::ldexp(1.0, exponent);
      if (!(scale > 0) || !std::isfinite(scale))
        continue;
      const f64 target_value = std::ceil(rhs.quotient_down(scale));
      if (!(target_value > 0) || target_value >= static_cast<f64>(states))
        continue;
      const auto target = static_cast<std::size_t>(target_value);
      const auto work = 2 * items.size() * (target + 1);
      if (work > budget)
        continue;
      budget -= work;
      std::vector<State> dp(target + 1), next(target + 1);
      dp[0].reachable = true;
      for (const auto &item : items) {
        model::DyadicSum weight;
        weight.add(item.weight);
        const auto amount = static_cast<std::size_t>(
            std::min(target_value, std::ceil(weight.quotient_up(scale))));
        next = dp;
        for (std::size_t k = 0; k <= target; ++k)
          if (dp[k].reachable) {
            const auto to = std::min(target, k + amount);
            auto cost = dp[k].cost;
            cost.add(item.cost);
            auto difference = cost;
            auto old = next[to].cost;
            old.negate();
            difference.add_sum(old);
            if (!next[to].reachable || difference.sign() < 0) {
              next[to].cost = std::move(cost);
              next[to].reachable = true;
            }
          }
        dp.swap(next);
      }
      if (!dp[target].reachable)
        continue; // scalar infeasibility is handled by domain/tree coverage
      constant.add_sum(dp[target].cost);
      CutRow cut;
      cut.row_lo = constant.down();
      cut.exact = true;
      cut.name = "SKP_";
      cut.used_local_bound = false;
      model::DyadicSum activity;
      for (const auto &item : items) {
        const f64 c = sense * lp.c[sz(item.col)];
        if (c == 0)
          continue;
        cut.cols.push_back(item.col);
        cut.vals.push_back(c);
        activity.add_product(c, x[sz(item.col)]);
      }
      if (!cut.cols.empty() && std::isfinite(cut.row_lo) &&
          activity.compare(cut.row_lo - violation) < 0)
        cuts.push_back(std::move(cut));
    }
  return cuts;
}
} // namespace sor::search
