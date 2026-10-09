#include "forge/search/conditional_components.hpp"
#include "forge/core/work_clock.hpp"
#include "forge/search/structure.hpp"
#include "forge/model/dyadic.hpp"
#include <stdexcept>
#include <limits>
#include "forge/sparse/csr.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <set>

namespace forge::search {
namespace {
using core::f64;
using core::Index;
std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
struct Partition {
  std::vector<std::vector<Index>> columns, rows;
  std::vector<Index> outside, constant_rows;
};
Partition partition(const model::LpProblem &lp,
                    const std::vector<Index> &boundary) {
  const Index n = lp.n_cols();
  std::vector<char> excluded(sz(n), false), in_row(sz(n), false);
  for (Index j : boundary)
    excluded[sz(j)] = true;
  for (Index j = 0; j < n; ++j)
    if (lp.col_lo[sz(j)] == lp.col_hi[sz(j)])
      excluded[sz(j)] = true;
  std::vector<Index> parent(sz(n));
  std::iota(parent.begin(), parent.end(), 0);
  const auto find = [&](Index v) {
    while (parent[sz(v)] != v) {
      parent[sz(v)] = parent[sz(parent[sz(v)])];
      v = parent[sz(v)];
    }
    return v;
  };
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  for (Index i = 0; i < lp.n_rows(); ++i) {
    Index first = -1;
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
      const Index j = ci[static_cast<std::size_t>(k)];
      if (lp.A.vals[static_cast<std::size_t>(k)] == 0 || excluded[sz(j)])
        continue;
      in_row[sz(j)] = true;
      if (first < 0)
        first = j;
      else
        parent[sz(find(j))] = find(first);
    }
  }
  Partition result;
  std::map<Index, std::size_t> component;
  for (Index j = 0; j < n; ++j) {
    if (!in_row[sz(j)]) {
      result.outside.push_back(j);
      continue;
    }
    auto [it, added] = component.try_emplace(find(j), component.size());
    if (added) {
      result.columns.emplace_back();
      result.rows.emplace_back();
    }
    result.columns[it->second].push_back(j);
  }
  for (Index i = 0; i < lp.n_rows(); ++i) {
    Index first = -1;
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
      if (lp.A.vals[static_cast<std::size_t>(k)] != 0 &&
          in_row[sz(ci[static_cast<std::size_t>(k)])]) {
        first = ci[static_cast<std::size_t>(k)];
        break;
      }
    if (first < 0)
      result.constant_rows.push_back(i);
    else
      result.rows[component.at(find(first))].push_back(i);
  }
  core::work_add(static_cast<std::uint64_t>(lp.nnz()) * 2 + sz(n),
                 core::WorkTag::MilpPresolve);
  return result;
}
// A finite-integer vertex cut of the exact incidence graph. Capacities are
// integer domain bits, used only to propose a boundary; exact partitioning
// below independently checks every row's ownership before it can be used.
std::vector<Index> graph_separator(const model::LpProblem& lp,
                                  const auto& finite_integer,
                                  const auto& expired) {
  const Index n = lp.n_cols();
  const auto wide_vertices = static_cast<std::uint64_t>(n) + lp.n_rows();
  if (wide_vertices > static_cast<std::uint64_t>(std::numeric_limits<Index>::max() / 2)) return {};
  const Index vertices = static_cast<Index>(wide_vertices);
  std::vector<std::vector<Index>> incidence(sz(vertices));
  const auto& rp = lp.A.pattern.row_ptr(); const auto& ci = lp.A.pattern.col_idx();
  Index first = -1;
  for (Index i = 0; i < lp.n_rows(); ++i) {
    if (expired()) return {};
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
      const Index j = ci[static_cast<std::size_t>(k)];
      if (lp.A.vals[static_cast<std::size_t>(k)] == 0 || lp.col_lo[sz(j)] == lp.col_hi[sz(j)]) continue;
      incidence[sz(j)].push_back(n + i); incidence[sz(n + i)].push_back(j);
      if (first < 0) first = j;
    }
  }
  if (first < 0) return {};
  const auto farthest = [&](Index source) {
    std::vector<Index> distance(sz(vertices), -1), queue{source};
    distance[sz(source)] = 0; Index result = source;
    for (std::size_t q = 0; q < queue.size(); ++q) {
      if (expired()) return Index{-1};
      const Index v = queue[q];
      if (v < n && distance[sz(v)] > distance[sz(result)]) result = v;
      for (Index w : incidence[sz(v)]) if (distance[sz(w)] < 0) {
        distance[sz(w)] = distance[sz(v)] + 1; queue.push_back(w);
      }
      core::work_add(incidence[sz(v)].size() + 1, core::WorkTag::MilpPresolve);
    }
    return result;
  };
  const Index source = farthest(first);
  if (source < 0) return {};
  const Index sink = farthest(source);
  if (sink < 0 || source == sink) return {};
  std::vector<std::uint64_t> cost(sz(vertices), 0);
  std::uint64_t total_cost = 0;
  for (Index j = 0; j < n; ++j) if (j != source && j != sink && finite_integer(j)) {
    cost[sz(j)] = static_cast<std::uint64_t>(std::ceil(std::log2(
        std::floor(lp.col_hi[sz(j)]) - std::ceil(lp.col_lo[sz(j)]) + 1)));
    total_cost += cost[sz(j)];
  }
  if (total_cost == 0) return {};
  const std::uint64_t infinite = total_cost + 1;
  struct Edge { Index to; std::size_t reverse; std::uint64_t residual; };
  std::vector<std::vector<Edge>> graph(sz(2 * vertices));
  const auto add = [&](Index a, Index b, std::uint64_t capacity) {
    const auto ar = graph[sz(a)].size(), br = graph[sz(b)].size();
    graph[sz(a)].push_back({b, br, capacity}); graph[sz(b)].push_back({a, ar, 0});
  };
  for (Index v = 0; v < vertices; ++v) {
    if (expired()) return {};
    add(2 * v, 2 * v + 1, cost[sz(v)] ? cost[sz(v)] : infinite);
    for (Index w : incidence[sz(v)]) add(2 * v + 1, 2 * w, infinite);
  }
  const Index start = 2 * source + 1, end = 2 * sink;
  std::uint64_t flow = 0;
  std::vector<Index> parent(graph.size());
  std::vector<std::size_t> edge_of(graph.size());
  for (;;) {
    std::fill(parent.begin(), parent.end(), -1);
    std::vector<Index> queue{start}; parent[sz(start)] = start;
    for (std::size_t q = 0; q < queue.size() && parent[sz(end)] < 0; ++q) {
      if (expired()) return {};
      const Index v = queue[q];
      for (std::size_t e = 0; e < graph[sz(v)].size(); ++e) {
        const auto& edge = graph[sz(v)][e];
        if (edge.residual && parent[sz(edge.to)] < 0) {
          parent[sz(edge.to)] = v; edge_of[sz(edge.to)] = e; queue.push_back(edge.to);
        }
      }
      core::work_add(graph[sz(v)].size() + 1, core::WorkTag::MilpPresolve);
    }
    if (parent[sz(end)] < 0) break;
    std::uint64_t amount = infinite;
    for (Index v = end; v != start; v = parent[sz(v)])
      amount = std::min(amount, graph[sz(parent[sz(v)])][edge_of[sz(v)]].residual);
    for (Index v = end; v != start; v = parent[sz(v)]) {
      auto& edge = graph[sz(parent[sz(v)])][edge_of[sz(v)]];
      edge.residual -= amount; graph[sz(v)][edge.reverse].residual += amount;
    }
    flow += amount;
    if (flow >= infinite) return {}; // no cut using only finite-integer vertices
  }
  std::vector<Index> separator;
  for (Index j = 0; j < n; ++j)
    if (cost[sz(j)] && parent[sz(2 * j)] >= 0 && parent[sz(2 * j + 1)] < 0) separator.push_back(j);
  return separator;
}
} // namespace
std::optional<ConditionalDecomposition>
compile_conditional_components(const model::LpProblem &lp,
                               double analysis_time_limit_s,
                               const core::CancelToken *cancel) {
  if (core::cancel_requested(cancel))
    return std::nullopt;
  const auto started = core::SolverClock::now();
  const Index n = lp.n_cols();
  const auto finite_integer = [&](Index j) {
    if (lp.is_integer.empty() || !lp.is_integer[sz(j)] ||
        !std::isfinite(lp.col_lo[sz(j)]) || !std::isfinite(lp.col_hi[sz(j)]))
      return false;
    const f64 lo = std::ceil(lp.col_lo[sz(j)]),
              hi = std::floor(lp.col_hi[sz(j)]);
    return lo < hi && std::fabs(lo) < 0x1p52 && std::fabs(hi) < 0x1p52;
  };
  const auto expired = [&] {
    return core::cancel_requested(cancel) || (analysis_time_limit_s > 0 &&
        std::chrono::duration<double>(core::SolverClock::now() - started).count() >= analysis_time_limit_s);
  };
  Partition best = partition(lp, {});
  const bool disconnected = best.columns.size() >= 2;
  std::set<std::vector<Index>> candidates;
  const Index articulation = disconnected ? -1 : integer_separator(lp);
  if (articulation >= 0)
    candidates.insert({articulation});
  else if (!disconnected) {
    auto vertex_cut = graph_separator(lp, finite_integer, expired);
    if (!vertex_cut.empty()) candidates.insert(std::move(vertex_cut));
  }
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  for (Index i = 0; i < lp.n_rows() && !disconnected; ++i) {
    if (core::cancel_requested(cancel))
      return std::nullopt;
    std::vector<Index> boundary;
    bool eligible = true;
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
      const Index j = ci[static_cast<std::size_t>(k)];
      if (lp.A.vals[static_cast<std::size_t>(k)] == 0 ||
          lp.col_lo[sz(j)] == lp.col_hi[sz(j)])
        continue;
      if (!finite_integer(j)) {
        eligible = false;
        break;
      }
      boundary.push_back(j);
    }
    if (eligible && !boundary.empty()) {
      std::sort(boundary.begin(), boundary.end());
      boundary.erase(std::unique(boundary.begin(), boundary.end()),
                     boundary.end());
      candidates.insert(std::move(boundary));
    }
  }
  std::vector<Index> selected;
  f64 best_score = 0;
  for (const auto &boundary : candidates) {
    if (core::cancel_requested(cancel))
      return std::nullopt;
    if (analysis_time_limit_s > 0 &&
        std::chrono::duration<double>(core::SolverClock::now() - started)
                .count() >= analysis_time_limit_s)
      break;
    auto part = partition(lp, boundary);
    if (part.columns.size() < 2)
      continue;
    std::size_t count = 0, largest = 0;
    for (const auto &c : part.columns) {
      count += c.size();
      largest = std::max(largest, c.size());
    }
    f64 domain_bits = 0;
    for (Index j : boundary)
      domain_bits += std::log2(std::floor(lp.col_hi[sz(j)]) -
                               std::ceil(lp.col_lo[sz(j)]) + 1);
    // Routing evidence, never proof: favor balanced residual work and
    // charge the boundary's entire finite-domain size in the comparison.
    const f64 score = static_cast<f64>(count - largest) / (1 + domain_bits);
    if (score > best_score) {
      best_score = score;
      selected = boundary;
      best = std::move(part);
    }
  }
  if (!disconnected && selected.empty())
    return std::nullopt;
  ConditionalDecomposition result;
  result.separator = selected;
  result.independent_cost_columns = std::move(best.outside);
  result.constant_rows = std::move(best.constant_rows);
  std::vector<Index> local(sz(n), -1), separator_index(sz(n), -1);
  for (std::size_t q = 0; q < selected.size(); ++q)
    separator_index[sz(selected[q])] = static_cast<Index>(q);
  for (std::size_t b = 0; b < best.columns.size(); ++b) {
    if (core::cancel_requested(cancel))
      return std::nullopt;
    ConditionalComponent block;
    block.columns = std::move(best.columns[b]);
    block.owned = block.columns.size();
    for (std::size_t q = 0; q < block.columns.size(); ++q)
      local[sz(block.columns[q])] = static_cast<Index>(q);
    for (Index i : best.rows[b])
      for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        const Index j = ci[static_cast<std::size_t>(k)];
        if (lp.A.vals[static_cast<std::size_t>(k)] != 0 && local[sz(j)] < 0) {
          local[sz(j)] = static_cast<Index>(block.columns.size());
          block.columns.push_back(j);
        }
      }
    auto &sub = block.model;
    sub.name = lp.name + "_conditional_block";
    sub.maximize = lp.maximize;
    for (std::size_t q = 0; q < block.columns.size(); ++q) {
      const Index j = block.columns[q];
      sub.c.push_back(q < block.owned ? lp.c[sz(j)] : 0);
      sub.col_lo.push_back(lp.col_lo[sz(j)]);
      sub.col_hi.push_back(lp.col_hi[sz(j)]);
      sub.is_integer.push_back(!lp.is_integer.empty() && lp.is_integer[sz(j)]);
      if (separator_index[sz(j)] >= 0)
        block.boundary.emplace_back(sz(separator_index[sz(j)]),
                                    static_cast<Index>(q));
    }
    std::vector<Index> row, col;
    std::vector<f64> val;
    for (std::size_t r = 0; r < best.rows[b].size(); ++r) {
      const Index i = best.rows[b][r];
      for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
        if (lp.A.vals[static_cast<std::size_t>(k)] != 0) {
          row.push_back(static_cast<Index>(r));
          col.push_back(local[sz(ci[static_cast<std::size_t>(k)])]);
          val.push_back(lp.A.vals[static_cast<std::size_t>(k)]);
        }
      sub.row_lo.push_back(lp.row_lo[sz(i)]);
      sub.row_hi.push_back(lp.row_hi[sz(i)]);
    }
    sub.A = sparse::from_triplets(static_cast<Index>(best.rows[b].size()),
                                  static_cast<Index>(block.columns.size()), row,
                                  col, val);
    for (Index j : block.columns)
      local[sz(j)] = -1;
    result.components.push_back(std::move(block));
  }
  return result;
}
model::LpProblem compile_conditional_master(const model::LpProblem& lp,
                                             const std::vector<Index>& separator) {
  lp.validate(true);
  model::LpProblem master;
  master.name = lp.name + "_boundary_master";
  std::vector<Index> position(sz(lp.n_cols()), -1);
  for (std::size_t q = 0; q < separator.size(); ++q) {
    const Index j = separator[q];
    if (j < 0 || j >= lp.n_cols() || position[sz(j)] >= 0 ||
        lp.is_integer.empty() || !lp.is_integer[sz(j)])
      throw std::invalid_argument("conditional master: invalid boundary");
    position[sz(j)] = static_cast<Index>(q);
    master.c.push_back(0); master.is_integer.push_back(true);
    master.col_lo.push_back(std::ceil(lp.col_lo[sz(j)]));
    master.col_hi.push_back(std::floor(lp.col_hi[sz(j)]));
  }
  // A nonempty real interval may contain no integer; projection of an
  // empty domain is empty regardless of whether that column occurs in a row.
  for (Index j = 0; j < lp.n_cols(); ++j) {
    if (!lp.is_integer.empty() && lp.is_integer[sz(j)] &&
        std::ceil(lp.col_lo[sz(j)]) > std::floor(lp.col_hi[sz(j)])) {
      master.A = sparse::from_triplets(1, static_cast<Index>(separator.size()), {}, {}, {});
      master.row_lo = {1}; master.row_hi = {model::kInf};
      // Boundary domains remain well formed; the empty row encodes the proof.
      for (std::size_t q = 0; q < master.col_lo.size(); ++q)
        if (master.col_lo[q] > master.col_hi[q]) master.col_lo[q] = master.col_hi[q] = 0;
      return master;
    }
  }
  std::vector<Index> row, col;
  std::vector<f64> val;
  const auto& rp = lp.A.pattern.row_ptr(); const auto& ci = lp.A.pattern.col_idx();
  for (Index i = 0; i < lp.n_rows(); ++i) {
    model::DyadicSum minimum, maximum;
    bool finite_min = true, finite_max = true;
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
      const Index j = ci[static_cast<std::size_t>(k)]; const f64 a = lp.A.vals[static_cast<std::size_t>(k)];
      if (a == 0) continue;
      if (position[sz(j)] >= 0) {
        row.push_back(i); col.push_back(position[sz(j)]); val.push_back(a);
      } else {
        f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
        if (!lp.is_integer.empty() && lp.is_integer[sz(j)]) { lo = std::ceil(lo); hi = std::floor(hi); }
        const f64 mn = a > 0 ? lo : hi, mx = a > 0 ? hi : lo;
        if (std::isfinite(mn)) minimum.add_product(a, mn); else finite_min = false;
        if (std::isfinite(mx)) maximum.add_product(a, mx); else finite_max = false;
      }
    }
    f64 lower = -model::kInf, upper = model::kInf;
    if (finite_max && std::isfinite(lp.row_lo[sz(i)])) {
      maximum.negate(); maximum.add(lp.row_lo[sz(i)]); lower = maximum.down();
    }
    if (finite_min && std::isfinite(lp.row_hi[sz(i)])) {
      minimum.negate(); minimum.add(lp.row_hi[sz(i)]); upper = minimum.up();
    }
    master.row_lo.push_back(lower); master.row_hi.push_back(upper);
    core::work_add(static_cast<std::uint64_t>(rp[sz(i) + 1] - rp[sz(i)]) + 1, core::WorkTag::MilpPresolve);
  }
  master.A = sparse::from_triplets(lp.n_rows(), static_cast<Index>(separator.size()), row, col, val);
  return master;
}
} // namespace forge::search
