#include "sor/search/domain_writer.hpp"
#include "sor/search/residue.hpp"
#include "sor/model/dyadic.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
namespace sor::search {
namespace {
std::size_t sz(Index j) { return static_cast<std::size_t>(j); }
int mod(f64 v, int p) {
  int r = static_cast<int>(std::fmod(v, p));
  return r < 0 ? r + p : r;
}
int inverse(int a, int p) {
  for (int i = 1; i < p; ++i)
    if (a * i % p == 1)
      return i;
  return 0;
}
unsigned sum(unsigned a, unsigned b, int p) {
  unsigned c = 0;
  for (int i = 0; i < p; ++i)
    if (a & (1u << i))
      for (int j = 0; j < p; ++j)
        if (b & (1u << j))
          c |= 1u << ((i + j) % p);
  return c;
}
unsigned domain(f64 lo, f64 hi, int a, int p) {
  if (!std::isfinite(lo) || !std::isfinite(hi) || hi - lo >= p - 1 ||
      std::fabs(lo) >= 0x1p52 || std::fabs(hi) >= 0x1p52)
    return (1u << p) - 1;
  unsigned mask = 0;
  for (f64 x = std::ceil(lo); x <= std::floor(hi); x += 1)
    mask |= 1u << (a * mod(x, p) % p);
  return mask;
}
} // namespace
void ResidueSystem::build(const model::LpProblem &lp,
                          const std::vector<char> &integer,
                          std::uint64_t budget) {
  generation_ = std::make_shared<const int>(0);
  equations_.clear();
  columns_.clear();
  position_.assign(static_cast<std::size_t>(lp.n_cols()), -1);
  incident_.assign(sz(lp.n_cols()), {});
  infeasible_ = false;
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  // Each prime is an independent exact relaxation. Sparse elimination can
  // terminate on its work allowance; all retained equations are still valid.
  for (int p : {2, 3, 5, 7}) {
    std::map<Index, Equation> pivots;
    for (Index i = 0; i < lp.n_rows() && budget > 0; ++i) {
      if (!std::isfinite(lp.row_lo[sz(i)]) ||
          lp.row_lo[sz(i)] != lp.row_hi[sz(i)])
        continue;
      Equation e;
      e.prime = p;
      model::DyadicSum rhs;
      rhs.add(lp.row_lo[sz(i)]);
      bool eligible = true;
      for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        if (!budget) {
          eligible = false;
          break;
        }
        --budget;
        const Index j = ci[sz(k)];
        const f64 a = lp.A.vals[sz(k)];
        if (a == 0)
          continue;
        if (lp.col_lo[sz(j)] == lp.col_hi[sz(j)] &&
            std::isfinite(lp.col_lo[sz(j)])) {
          rhs.add_product(-a, lp.col_lo[sz(j)]);
          continue;
        }
        if (sz(j) >= integer.size() || !integer[sz(j)] || !std::isfinite(a) ||
            a != std::floor(a)) {
          eligible = false;
          break;
        }
        const int v = (e.terms[j] + mod(a, p)) % p;
        if (v)
          e.terms[j] = v;
        else
          e.terms.erase(j);
      }
      if (!eligible || rhs.down() != rhs.up() || !std::isfinite(rhs.down()) ||
          rhs.down() != std::floor(rhs.down()))
        continue;
      e.rhs = mod(rhs.down(), p);
      while (!e.terms.empty()) {
        const auto first = *e.terms.begin();
        const auto found = pivots.find(first.first);
        if (found == pivots.end())
          break;
        if (found->second.terms.size() > budget) {
          eligible = false;
          break;
        }
        budget -= found->second.terms.size();
        const int factor = first.second;
        e.rhs = (e.rhs - factor * found->second.rhs % p + p) % p;
        for (const auto &[j, v] : found->second.terms) {
          const int value = (e.terms[j] - factor * v % p + p) % p;
          if (value)
            e.terms[j] = value;
          else
            e.terms.erase(j);
        }
      }
      if (!eligible)
        continue;
      if (e.terms.empty()) {
        if (e.rhs) {
          infeasible_ = true;
          return;
        }
        continue;
      }
      const int inv = inverse(e.terms.begin()->second, p);
      for (auto &[j, v] : e.terms) {
        (void)j;
        v = v * inv % p;
      }
      e.rhs = e.rhs * inv % p;
      pivots.emplace(e.terms.begin()->first, e);
      equations_.push_back(std::move(e));
    }
  }
  for (Index i = 0; i < static_cast<Index>(equations_.size()); ++i)
    for (const auto &[j, v] : equations_[sz(i)].terms) {
      (void)v;
      incident_[sz(j)].push_back(i);
    }
  for (Index j = 0; j < lp.n_cols(); ++j)
    if (!incident_[sz(j)].empty()) {
      position_[sz(j)] = static_cast<Index>(columns_.size());
      columns_.push_back(j);
    }
}
bool ResidueSystem::propagate(std::vector<f64> &lo, std::vector<f64> &hi,
                              std::uint64_t &tightened, PropTrail *trail,
                              int depth, const std::vector<Index> *seeds,
                              ResidueState *saved, ConflictAntecedents* conflict_reason) const {
  if (infeasible_)
    return false;
  static constexpr std::array<int, 4> primes{2, 3, 5, 7};
  ResidueState local;
  auto &state = saved ? *saved : local;
  bool compatible =
      state.generation == generation_ && state.masks.size() == columns_.size() &&
      state.lo.size() == columns_.size() && state.hi.size() == columns_.size();
  if (compatible)
    for (std::size_t k = 0; k < columns_.size(); ++k) {
      const Index j = columns_[k];
      if (lo[sz(j)] < state.lo[k] || hi[sz(j)] > state.hi[k]) {
        compatible = false;
        break;
      }
    }
  if (!compatible) {
    state.generation = generation_;
    state.masks.assign(columns_.size(), {3, 7, 31, 127});
    state.lo.resize(columns_.size());
    state.hi.resize(columns_.size());
    seeds = nullptr;
  }
  // Every mask computed in this call rests on at least this entry box.
  // Anchor before any early conflict exit; failed probes cannot leave masks
  // whose recorded scope is wider than the box that justified them.
  auto refs = trail || conflict_reason ? std::make_shared<std::vector<BoundRef>>() : nullptr;
  for (std::size_t k = 0; k < columns_.size(); ++k) {
    const Index j = columns_[k];
    state.lo[k] = lo[sz(j)]; state.hi[k] = hi[sz(j)];
    if (refs) { refs->push_back({j, false}); refs->push_back({j, true}); }
  }
  const DomainReason reason{ReasonKind::Antecedents, -1, depth, {}, refs};
  if (conflict_reason) conflict_reason->reset();
  const auto conflict = [&] { if (conflict_reason) *conflict_reason = refs; return false; };
  std::deque<Index> queue;
  std::vector<char> queued(equations_.size(), 0);
  const auto enqueue = [&](Index i) {
    if (!queued[sz(i)]) {
      queued[sz(i)] = 1;
      queue.push_back(i);
    }
  };
  if (seeds) {
    for (Index j : *seeds)
      if (j >= 0 && sz(j) < incident_.size())
        for (Index i : incident_[sz(j)])
          enqueue(i);
  } else
    for (Index i = 0; i < static_cast<Index>(equations_.size()); ++i)
      enqueue(i);
  for (std::size_t k = 0; k < columns_.size(); ++k) {
    const Index j = columns_[k];
    bool changed = false;
    for (std::size_t q = 0; q < primes.size(); ++q) {
      const unsigned mask =
          state.masks[k][q] & domain(lo[sz(j)], hi[sz(j)], 1, primes[q]);
      if (!mask)
        return conflict();
      changed = changed || mask != state.masks[k][q];
      state.masks[k][q] = static_cast<unsigned char>(mask);
    }
    if (changed)
      for (Index i : incident_[sz(j)])
        enqueue(i);
  }
  while (!queue.empty()) {
    const Index id = queue.front();
    queue.pop_front();
    queued[sz(id)] = 0;
    const auto &e = equations_[sz(id)];
    const int p = e.prime;
    std::vector<unsigned> pre(e.terms.size() + 1, 1),
        post(e.terms.size() + 1, 1), domains;
    const auto prime_index = static_cast<std::size_t>(
        std::find(primes.begin(), primes.end(), p) - primes.begin());
    for (const auto &[j, a] : e.terms) {
      unsigned transformed = 0;
      const auto mask = state.masks[sz(position_[sz(j)])][prime_index];
      for (int r = 0; r < p; ++r)
        if (mask & (1u << r))
          transformed |= 1u << (a * r % p);
      domains.push_back(transformed);
    }
    for (std::size_t k = 0; k < domains.size(); ++k)
      pre[k + 1] = sum(pre[k], domains[k], p);
    if (!(pre.back() & (1u << e.rhs)))
      return conflict();
    for (std::size_t k = domains.size(); k > 0; --k)
      post[k - 1] = sum(domains[k - 1], post[k], p);
    std::size_t k = 0;
    for (const auto &[j, a] : e.terms) {
      const unsigned others = sum(pre[k], post[k + 1], p);
      ++k;
      auto &masks = state.masks[sz(position_[sz(j)])];
      unsigned supported = 0;
      for (int r = 0; r < p; ++r)
        if ((masks[prime_index] & (1u << r)) &&
            (others & (1u << ((e.rhs - a * r % p + p) % p))))
          supported |= 1u << r;
      if (!supported)
        return conflict();
      bool changed = supported != masks[prime_index];
      masks[prime_index] = static_cast<unsigned char>(supported);
      const auto allowed = [&](f64 x) {
        for (std::size_t q = 0; q < primes.size(); ++q)
          if (!(masks[q] & (1u << mod(x, primes[q]))))
            return conflict();
        return true;
      };
      f64 l = lo[sz(j)], h = hi[sz(j)];
      if (std::isfinite(l) && std::fabs(l) < 0x1p52) {
        l = std::ceil(l);
        int step = 0;
        while (step < 210 && !allowed(l)) {
          l += 1;
          ++step;
        }
        if (step == 210)
          return conflict();
      }
      if (std::isfinite(h) && std::fabs(h) < 0x1p52) {
        h = std::floor(h);
        int step = 0;
        while (step < 210 && !allowed(h)) {
          h -= 1;
          ++step;
        }
        if (step == 210)
          return conflict();
      }
      if (l > h)
        return conflict();
      if (l > lo[sz(j)]) {
        write_domain_bound(lo, hi, j, BoundDir::Lower, l,
                           trail, reason);
        ++tightened;
        changed = true;
      }
      if (h < hi[sz(j)]) {
        write_domain_bound(lo, hi, j, BoundDir::Upper, h,
                           trail, reason);
        ++tightened;
        changed = true;
      }
      if (changed) {
        for (std::size_t q = 0; q < primes.size(); ++q)
          masks[q] &= static_cast<unsigned char>(
              domain(lo[sz(j)], hi[sz(j)], 1, primes[q]));
        for (Index row : incident_[sz(j)])
          enqueue(row);
      }
    }
  }
  for (std::size_t k = 0; k < columns_.size(); ++k) {
    state.lo[k] = lo[sz(columns_[k])];
    state.hi[k] = hi[sz(columns_[k])];
  }
  return true;
}
} // namespace sor::search
