#include "forge/search/structure.hpp"
#include "forge/search/certificate_rows.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace forge::search {
namespace {
core::Index articulation_separator(const model::LpProblem &lp,
                                   bool binary_only) {
  using core::Index;
  const auto sz = [](Index v) { return static_cast<std::size_t>(v); };
  const Index n = lp.n_cols(), count = n + lp.n_rows();
  std::vector<std::vector<Index>> adj(sz(count));
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  for (Index i = 0; i < lp.n_rows(); ++i)
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
      const Index j = ci[sz(k)];
      if (lp.A.vals[sz(k)] == 0.0 || lp.col_lo[sz(j)] == lp.col_hi[sz(j)])
        continue;
      adj[sz(j)].push_back(n + i);
      adj[sz(n + i)].push_back(j);
    }
  std::vector<Index> disc(sz(count), -1), low(sz(count)), parent(sz(count), -1),
      subtree(sz(count), 0), separated(sz(count), 0), largest(sz(count), 0),
      children(sz(count), 0);
  std::vector<std::size_t> cursor(sz(count), 0);
  Index serial = 0, best = -1, best_balance = 0;
  for (Index root = 0; root < count; ++root) {
    if (disc[sz(root)] >= 0 || adj[sz(root)].empty())
      continue;
    std::vector<Index> stack{root}, members;
    disc[sz(root)] = low[sz(root)] = serial++;
    subtree[sz(root)] = root < n ? 1 : 0;
    while (!stack.empty()) {
      const Index v = stack.back();
      if (cursor[sz(v)] < adj[sz(v)].size()) {
        const Index w = adj[sz(v)][cursor[sz(v)]++];
        if (w == parent[sz(v)])
          continue;
        if (disc[sz(w)] < 0) {
          parent[sz(w)] = v;
          ++children[sz(v)];
          disc[sz(w)] = low[sz(w)] = serial++;
          subtree[sz(w)] = w < n ? 1 : 0;
          stack.push_back(w);
        } else
          low[sz(v)] = std::min(low[sz(v)], disc[sz(w)]);
      } else {
        members.push_back(v);
        stack.pop_back();
        const Index p = parent[sz(v)];
        if (p >= 0) {
          subtree[sz(p)] += subtree[sz(v)];
          low[sz(p)] = std::min(low[sz(p)], low[sz(v)]);
          if (low[sz(v)] >= disc[sz(p)]) {
            separated[sz(p)] += subtree[sz(v)];
            largest[sz(p)] = std::max(largest[sz(p)], subtree[sz(v)]);
          }
        }
      }
    }
    const Index size = subtree[sz(root)];
    for (const Index j : members) {
      if (j >= n || lp.is_integer.empty() || !lp.is_integer[sz(j)] ||
          !std::isfinite(lp.col_lo[sz(j)]) || !std::isfinite(lp.col_hi[sz(j)]))
        continue;
      const double lower = std::ceil(lp.col_lo[sz(j)]),
                   upper = std::floor(lp.col_hi[sz(j)]);
      // Consecutive conditioning values must be representable in this model's
      // binary64 coordinates. Other domains remain with ordinary branching.
      if (!(lower < upper) || std::fabs(lower) >= 0x1p52 ||
          std::fabs(upper) >= 0x1p52 ||
          (binary_only && (lower != 0.0 || upper != 1.0)))
        continue;
      if (j == root && children[sz(j)] < 2)
        continue;
      const Index maxpart =
          std::max(largest[sz(j)], size - 1 - separated[sz(j)]);
      const Index balance = size - 1 - maxpart;
      if (balance > best_balance) {
        best_balance = balance;
        best = j;
      }
    }
  }
  return best;
}
} // namespace
core::Index binary_separator(const model::LpProblem &lp) {
  return articulation_separator(lp, true);
}
core::Index integer_separator(const model::LpProblem &lp) {
  return articulation_separator(lp, false);
}
std::vector<CutRow> network_cutset_rows(const model::LpProblem &lp,
                                        std::uint64_t budget) {
  const auto sz = [](Index j) { return static_cast<std::size_t>(j); };
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  std::vector<std::vector<std::pair<Index, f64>>> incidence(sz(lp.n_cols()));
  std::vector<char> verified(sz(lp.n_rows()), 0), seen(sz(lp.n_rows()), 0);
  for (Index i = 0; i < lp.n_rows(); ++i) {
    if (!std::isfinite(lp.row_lo[sz(i)]) ||
        lp.row_lo[sz(i)] != lp.row_hi[sz(i)])
      continue;
    bool good = true;
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
      if (lp.A.vals[sz(k)] != 0 && std::fabs(lp.A.vals[sz(k)]) != 1) {
        good = false;
        break;
      }
    if (!good)
      continue;
    verified[sz(i)] = 1;
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
      if (lp.A.vals[sz(k)] != 0)
        incidence[sz(ci[sz(k)])].emplace_back(i, lp.A.vals[sz(k)]);
  }
  std::vector<std::vector<Index>> adjacency(sz(lp.n_rows()));
  for (const auto &arc : incidence)
    if (arc.size() == 2 && arc[0].second == -arc[1].second) {
      adjacency[sz(arc[0].first)].push_back(arc[1].first);
      adjacency[sz(arc[1].first)].push_back(arc[0].first);
    }
  std::vector<CutRow> rows;
  for (Index seed = 0; seed < lp.n_rows() && budget > 0; ++seed)
    if (verified[sz(seed)] && !seen[sz(seed)] && !adjacency[sz(seed)].empty()) {
      std::vector<Index> queue{seed};
      seen[sz(seed)] = 1;
      std::map<Index, std::int64_t> boundary;
      model::DyadicSum rhs;
      for (std::size_t head = 0; head < queue.size(); ++head) {
        const Index i = queue[head];
        const auto work =
            static_cast<std::uint64_t>(rp[sz(i) + 1] - rp[sz(i)]) +
            boundary.size();
        if (work > budget)
          break;
        budget -= work;
        rhs.add(lp.row_lo[sz(i)]);
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
          const Index j = ci[sz(k)];
          boundary[j] += static_cast<std::int64_t>(lp.A.vals[sz(k)]);
          if (boundary[j] == 0)
            boundary.erase(j);
        }
        for (Index r : adjacency[sz(i)])
          if (!seen[sz(r)]) {
            seen[sz(r)] = 1;
            queue.push_back(r);
          }
        if (head == 0 || boundary.size() < 2)
          continue;
        CutRow row;
        row.row_lo = rhs.down();
        row.row_hi = rhs.up();
        row.exact = true;
        row.used_local_bound = false;
        row.name = "NET_";
        for (const auto &[j, a] : boundary) {
          row.cols.push_back(j);
          row.vals.push_back(static_cast<f64>(a));
        }
        rows.push_back(std::move(row));
      }
    }
  return rows;
}
std::vector<std::vector<Index>>
binary_product_blocks(const model::LpProblem &lp,
                      const std::vector<char> &integer, int max_columns) {
  const auto sz = [](Index j) { return static_cast<std::size_t>(j); };
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  std::set<std::pair<Index, Index>> upper;
  struct Product {
    Index x, y, z;
  };
  std::vector<Product> products;
  const auto binary = [&](Index j) {
    return integer[sz(j)] && lp.col_lo[sz(j)] >= 0 && lp.col_hi[sz(j)] <= 1;
  };
  for (Index i = 0; i < lp.n_rows(); ++i)
    for (f64 sign : {1.0, -1.0}) {
      const f64 rhs = sign > 0 ? lp.row_hi[sz(i)] : -lp.row_lo[sz(i)];
      if (rhs != 0 && rhs != 1)
        continue;
      std::vector<Index> positive, negative;
      bool good = true;
      for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = sign * lp.A.vals[sz(k)];
        if (a == 0)
          continue;
        if (!binary(j) || std::fabs(a) != 1) {
          good = false;
          break;
        }
        (a > 0 ? positive : negative).push_back(j);
      }
      if (!good)
        continue;
      if (rhs == 0 && positive.size() == 1 && negative.size() == 1)
        upper.emplace(positive[0], negative[0]);
      if (rhs == 1 && positive.size() == 2 && negative.size() == 1)
        products.push_back({positive[0], positive[1], negative[0]});
    }
  std::map<Index, std::vector<Product>> stars;
  for (const auto &product : products)
    if (upper.count({product.z, product.x}) &&
        upper.count({product.z, product.y})) {
      stars[product.x].push_back(product);
      stars[product.y].push_back(product);
    }
  std::vector<std::vector<Index>> blocks;
  for (const auto &[anchor, star] : stars) {
    std::set<Index> support{anchor};
    for (const auto &product : star) {
      auto next = support;
      next.insert(product.x);
      next.insert(product.y);
      next.insert(product.z);
      if (static_cast<int>(next.size()) > max_columns)
        continue;
      support = std::move(next);
    }
    if (support.size() > 3)
      blocks.emplace_back(support.begin(), support.end());
  }
  return blocks;
}
} // namespace forge::search
