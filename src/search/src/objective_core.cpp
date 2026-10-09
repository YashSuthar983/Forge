#include "forge/search/objective_core.hpp"
#include "forge/model/dyadic.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
namespace forge::search {
namespace {
std::size_t sz(Index j) { return static_cast<std::size_t>(j); }
} // namespace
ObjectiveCoreResult
price_objective_cores(const model::LpProblem &lp,
                      const std::vector<char> &integer,
                      const std::vector<std::vector<ConflictLiteral>> &clauses,
                      std::uint64_t budget) {
  ObjectiveCoreResult result;
  std::vector<Index> position(sz(lp.n_cols()), -1), columns;
  std::vector<f64> costs;
  model::DyadicSum constant;
  const f64 sense = lp.maximize ? -1 : 1;
  constant.add(sense * lp.obj_offset);
  for (Index j = 0; j < lp.n_cols(); ++j) {
    const f64 c = sense * lp.c[sz(j)];
    if (c == 0)
      continue;
    if (integer[sz(j)] && lp.col_lo[sz(j)] == 0 && lp.col_hi[sz(j)] == 1) {
      position[sz(j)] = static_cast<Index>(columns.size());
      columns.push_back(j);
      costs.push_back(std::fabs(c));
      if (c < 0)
        constant.add(c);
    } else {
      const f64 endpoint = c > 0 ? lp.col_lo[sz(j)] : lp.col_hi[sz(j)];
      if (!std::isfinite(endpoint))
        return result;
      constant.add_product(c, endpoint);
    }
  }
  std::vector<std::vector<Index>> cores;
  for (const auto &clause : clauses) {
    std::vector<Index> core;
    bool eligible = true;
    for (const auto &lit : clause) {
      if (lit.var < 0 || lit.var >= lp.n_cols() || position[sz(lit.var)] < 0) {
        eligible = false;
        break;
      }
      const bool positive = sense * lp.c[sz(lit.var)] > 0;
      const bool paid = positive
                            ? (!lit.upper && lit.bound > 0 && lit.bound <= 1)
                            : (lit.upper && lit.bound >= 0 && lit.bound < 1);
      if (!paid) {
        eligible = false;
        break;
      }
      core.push_back(position[sz(lit.var)]);
    }
    if (eligible && !core.empty()) {
      std::sort(core.begin(), core.end());
      core.erase(std::unique(core.begin(), core.end()), core.end());
      cores.push_back(std::move(core));
    }
  }
  if (cores.empty())
    return result;
  struct State {
    std::vector<signed char> value;
    f64 bound;
  };
  const auto compare = [](const State &a, const State &b) {
    return a.bound > b.bound;
  };
  std::priority_queue<State, std::vector<State>, decltype(compare)> open(
      compare);
  f64 best = model::kInf;
  std::vector<signed char> best_pattern;
  const auto analyze = [&](State &state, std::vector<Index> &branch) {
    model::DyadicSum bound = constant;
    for (std::size_t j = 0; j < costs.size(); ++j)
      if (state.value[j] == 1)
        bound.add(costs[j]);
    std::vector<char> charged(costs.size(), 0);
    bool complete = true;
    for (const auto &core : cores) {
      bool hit = false;
      std::vector<Index> available;
      for (Index j : core) {
        hit = hit || state.value[sz(j)] == 1;
        if (state.value[sz(j)] < 0)
          available.push_back(j);
      }
      if (hit)
        continue;
      if (available.empty())
        return false;
      complete = false;
      if (branch.empty() || available.size() < branch.size())
        branch = available;
      bool disjoint = true;
      f64 minimum = model::kInf;
      for (Index j : available) {
        disjoint = disjoint && !charged[sz(j)];
        minimum = std::min(minimum, costs[sz(j)]);
      }
      if (disjoint) {
        bound.add(minimum);
        for (Index j : available)
          charged[sz(j)] = 1;
      }
    }
    state.bound = bound.down();
    if (complete && state.bound < best) {
      best = state.bound;
      best_pattern = state.value;
    }
    return !complete;
  };
  State initial;
  initial.value.assign(costs.size(), -1);
  std::vector<Index> branch;
  if (analyze(initial, branch))
    open.push(std::move(initial));
  const std::uint64_t work =
      costs.size() + std::accumulate(cores.begin(), cores.end(),
                                     std::uint64_t{0},
                                     [](std::uint64_t n, const auto &c) {
                                       return n + c.size();
                                     });
  while (!open.empty() && budget > 0) {
    if (work > budget)
      break;
    budget -= work;
    State state = open.top();
    open.pop();
    branch.clear();
    if (state.bound >= best || !analyze(state, branch))
      continue;
    // The children partition this unsatisfied core: first paid position.
    // Overlapping cores are constraints, never independently charged costs.
    for (Index j : branch) {
      if (work > budget) {
        open.push(state);
        break;
      }
      budget -= work;
      State child = state;
      child.value[sz(j)] = 1;
      std::vector<Index> next;
      if (analyze(child, next) && child.bound < best)
        open.push(std::move(child));
      state.value[sz(j)] = 0;
    }
  }
  result.lower_bound =
      std::min(best, open.empty() ? model::kInf : open.top().bound);
  for (std::size_t j = 0; j < best_pattern.size(); ++j)
    if (best_pattern[j] == 1)
      result.free_paid_columns.push_back(columns[j]);
  return result;
}
} // namespace forge::search
