#include "forge/search/domain_writer.hpp"
#include "forge/search/matching.hpp"
#include <algorithm>
#include <functional>
#include <queue>
namespace forge::search {
namespace {
std::size_t sz(Index j) { return static_cast<std::size_t>(j); }
} // namespace
void MatchingService::build(const model::LpProblem &lp,
                            const std::vector<char> &integer) {
  blocks_.clear();
  const Index m = lp.n_rows(), n = lp.n_cols();
  std::vector<char> eligible(sz(m), 0), equality(sz(m), 0), bad(sz(m), 0);
  std::vector<std::vector<Index>> columns(sz(n)), row_cols(sz(m));
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  for (Index i = 0; i < m; ++i) {
    if (lp.row_hi[sz(i)] != 1 ||
        !(lp.row_lo[sz(i)] <= 0 || lp.row_lo[sz(i)] == 1))
      continue;
    bool good = true;
    for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
      const Index j = ci[sz(k)];
      const f64 a = lp.A.vals[sz(k)];
      if (a == 0)
        continue;
      if (a != 1 || !integer[sz(j)] || lp.col_lo[sz(j)] < 0 ||
          lp.col_hi[sz(j)] > 1) {
        good = false;
        break;
      }
      row_cols[sz(i)].push_back(j);
    }
    if (!good || row_cols[sz(i)].empty()) {
      row_cols[sz(i)].clear();
      continue;
    }
    eligible[sz(i)] = 1;
    equality[sz(i)] = lp.row_lo[sz(i)] == 1;
    for (Index j : row_cols[sz(i)])
      columns[sz(j)].push_back(i);
  }
  std::vector<std::vector<std::pair<Index, Index>>> adj(sz(m));
  for (Index j = 0; j < n; ++j) {
    if (columns[sz(j)].size() != 2) {
      for (Index i : columns[sz(j)])
        bad[sz(i)] = 1;
      continue;
    }
    const Index a = columns[sz(j)][0], b = columns[sz(j)][1];
    adj[sz(a)].emplace_back(b, j);
    adj[sz(b)].emplace_back(a, j);
  }
  std::vector<Index> color(sz(m), -1), local(sz(m), -1);
  for (Index seed = 0; seed < m; ++seed)
    if (eligible[sz(seed)] && color[sz(seed)] < 0) {
      std::vector<Index> component;
      std::queue<Index> queue;
      queue.push(seed);
      color[sz(seed)] = 0;
      bool verified = true;
      while (!queue.empty()) {
        const Index i = queue.front();
        queue.pop();
        component.push_back(i);
        verified = verified && !bad[sz(i)];
        for (const auto &[r, j] : adj[sz(i)]) {
          (void)j;
          if (color[sz(r)] < 0) {
            color[sz(r)] = 1 - color[sz(i)];
            queue.push(r);
          } else if (color[sz(r)] == color[sz(i)])
            verified = false;
        }
      }
      if (!verified)
        continue;
      for (Index orientation = 0; orientation < 2; ++orientation) {
        bool jobs_exact = true;
        Index jobs = 0, values = 0;
        for (Index i : component)
          if (color[sz(i)] == orientation) {
            jobs_exact = jobs_exact && equality[sz(i)];
            local[sz(i)] = jobs++;
          } else
            local[sz(i)] = values++;
        if (!jobs_exact || jobs == 0 || values == 0)
          continue;
        Block block;
        block.values = values;
        block.jobs.resize(sz(jobs));
        for (Index i : component)
          if (color[sz(i)] == orientation)
            for (const auto &[r, j] : adj[sz(i)])
              block.jobs[sz(local[sz(i)])].push_back({j, local[sz(r)]});
        blocks_.push_back(std::move(block));
      }
    }
}
bool MatchingService::propagate(std::vector<f64> &lo, std::vector<f64> &hi,
                                std::uint64_t &tightened, std::uint64_t budget,
                                PropTrail *trail, int depth,
                                bool *converged, ConflictAntecedents* conflict_reason) const {
  if (conflict_reason) conflict_reason->reset();
  if (converged)
    *converged = true;
  const auto pending = [&]() {
    if (converged)
      *converged = false;
    return true;
  };
  for (const auto &block : blocks_) {
    const Index jobs = static_cast<Index>(block.jobs.size()),
                count = jobs + block.values;
    // The filtering theorem uses exactly this block's edge domains. The
    // immutable explanation is independent of budgets and worker store IDs.
    auto refs = trail || conflict_reason ? std::make_shared<std::vector<BoundRef>>() : nullptr;
    if (refs) for (const auto& edges : block.jobs) for (const auto& edge : edges) {
      refs->push_back({edge.col, false}); refs->push_back({edge.col, true});
    }
    const DomainReason reason{ReasonKind::Antecedents, -1, depth, {}, refs};
    const auto conflict = [&] { if (conflict_reason) *conflict_reason = refs; return false; };
    std::vector<Index> owner(sz(block.values), -1), match(sz(jobs), -1),
        selected(sz(jobs), -1);
    std::vector<char> pinned(sz(jobs), 0);
    for (Index job = 0; job < jobs; ++job)
      for (const auto &edge : block.jobs[sz(job)]) {
        if (!budget)
          return pending();
        --budget;
        if (lo[sz(edge.col)] > 0) {
          if (hi[sz(edge.col)] < 1 || selected[sz(job)] >= 0 ||
              owner[sz(edge.value)] >= 0)
            return conflict();
          selected[sz(job)] = edge.col;
          match[sz(job)] = edge.value;
          owner[sz(edge.value)] = job;
          pinned[sz(job)] = 1;
        }
      }
    for (Index start = 0; start < jobs; ++start)
      if (match[sz(start)] < 0) {
        std::vector<Index> parent(sz(block.values), -1), queue{start};
        Index terminal = -1;
        for (std::size_t head = 0; head < queue.size() && terminal < 0;
             ++head) {
          const Index job = queue[head];
          for (const auto &edge : block.jobs[sz(job)]) {
            if (!budget)
              return pending();
            --budget;
            if (hi[sz(edge.col)] < 1 || parent[sz(edge.value)] >= 0)
              continue;
            parent[sz(edge.value)] = job;
            const Index other = owner[sz(edge.value)];
            if (other < 0) {
              terminal = edge.value;
              break;
            }
            if (!pinned[sz(other)])
              queue.push_back(other);
          }
        }
        if (terminal < 0)
          return conflict(); // complete augmenting-path search: Hall witness
        while (terminal >= 0) {
          const Index job = parent[sz(terminal)], previous = match[sz(job)];
          owner[sz(terminal)] = job;
          match[sz(job)] = terminal;
          terminal = previous;
        }
      }
    // One matching plus alternating reachability/SCC filters every edge.
    // This avoids rebuilding a matching separately for each proposed edge.
    std::vector<std::vector<Index>> graph(sz(count)), reverse(sz(count));
    const auto remove = [&](Index col) {
      if (hi[sz(col)] == 0)
        return;
      write_domain_bound(lo, hi, col, BoundDir::Upper, 0,
                         trail, reason);
      ++tightened;
    };
    for (Index job = 0; job < jobs; ++job)
      for (const auto &edge : block.jobs[sz(job)]) {
        if (!budget)
          return pending();
        --budget;
        if (hi[sz(edge.col)] < 1)
          continue;
        const Index other = owner[sz(edge.value)];
        if ((pinned[sz(job)] && edge.col != selected[sz(job)]) ||
            (other >= 0 && pinned[sz(other)] && other != job)) {
          remove(edge.col);
          continue;
        }
        if (pinned[sz(job)])
          continue;
        const Index value = jobs + edge.value;
        const Index from = match[sz(job)] == edge.value ? value : job;
        const Index to = match[sz(job)] == edge.value ? job : value;
        graph[sz(from)].push_back(to);
        reverse[sz(to)].push_back(from);
      }
    // Reserve the complete linear alternating-graph pass before filtering.
    // Partial graph exploration cannot justify deleting an edge.
    std::uint64_t edges = 0;
    for (const auto &row : graph)
      edges += row.size();
    std::uint64_t candidates = 0;
    for (const auto &row : block.jobs)
      candidates += row.size();
    const std::uint64_t remaining_work = 4 * edges + 4 * sz(count) + candidates;
    if (budget < remaining_work)
      return pending();
    budget -= remaining_work;
    std::vector<char> reach(sz(count), 0), seen(sz(count), 0);
    std::vector<Index> queue, order, component(sz(count), -1);
    for (Index v = 0; v < block.values; ++v)
      if (owner[sz(v)] < 0) {
        queue.push_back(jobs + v);
        reach[sz(jobs + v)] = 1;
      }
    for (std::size_t h = 0; h < queue.size(); ++h)
      for (Index u : reverse[sz(queue[h])])
        if (!reach[sz(u)]) {
          reach[sz(u)] = 1;
          queue.push_back(u);
        }
    for (Index seed = 0; seed < count; ++seed)
      if (!seen[sz(seed)]) {
        std::vector<std::pair<Index, std::size_t>> stack{{seed, 0}};
        seen[sz(seed)] = 1;
        while (!stack.empty()) {
          auto &frame = stack.back();
          if (frame.second < graph[sz(frame.first)].size()) {
            const Index u = graph[sz(frame.first)][frame.second++];
            if (!seen[sz(u)]) {
              seen[sz(u)] = 1;
              stack.emplace_back(u, 0);
            }
          } else {
            order.push_back(frame.first);
            stack.pop_back();
          }
        }
      }
    Index serial = 0;
    for (auto it = order.rbegin(); it != order.rend(); ++it)
      if (component[sz(*it)] < 0) {
        queue.assign(1, *it);
        component[sz(*it)] = serial;
        for (std::size_t h = 0; h < queue.size(); ++h)
          for (Index u : reverse[sz(queue[h])])
            if (component[sz(u)] < 0) {
              component[sz(u)] = serial;
              queue.push_back(u);
            }
        ++serial;
      }
    for (Index job = 0; job < jobs; ++job)
      for (const auto &edge : block.jobs[sz(job)]) {
        if (hi[sz(edge.col)] < 1 || match[sz(job)] == edge.value ||
            pinned[sz(job)])
          continue;
        const Index value = jobs + edge.value;
        if (!reach[sz(value)] && component[sz(job)] != component[sz(value)])
          remove(edge.col);
      }
  }
  return true;
}
} // namespace forge::search
