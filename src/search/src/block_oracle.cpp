#include "forge/search/block_oracle.hpp"
#include "forge/model/dyadic.hpp"
#include "forge/search/structure.hpp"
#include "forge/sparse/csc.hpp"
#include <algorithm>
#include <cmath>
#include <set>

namespace forge::search {
namespace {
std::size_t sz(Index j) { return static_cast<std::size_t>(j); }
model::DyadicSum dot(const std::vector<f64> &a, const std::vector<f64> &b) {
  model::DyadicSum result;
  for (std::size_t j = 0; j < a.size(); ++j)
    result.add_product(a[j], b[j]);
  return result;
}
} // namespace
void FiniteBlockOracle::build(const model::LpProblem &lp,
                              const std::vector<char> &required,
                              int max_columns, std::uint64_t budget,
                              f64 feasibility_tolerance) {
  blocks_.clear();
  infeasible_ = false;
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  std::set<std::vector<Index>> seen;
  const auto columns = sparse::to_csc(lp.A);
  std::vector<Index> local(sz(lp.n_cols()), -1);
  std::vector<std::uint32_t> row_seen(sz(lp.n_rows()), 0);
  std::uint32_t generation = 0;
  const auto product_blocks = binary_product_blocks(lp, required, max_columns);
  for (Index candidate = 0;
       candidate < static_cast<Index>(product_blocks.size()) + lp.n_rows() &&
       budget > 0;
       ++candidate) {
    const Index anchor = candidate - static_cast<Index>(product_blocks.size());
    Block block;
    bool eligible = true;
    if (anchor < 0)
      block.cols = product_blocks[sz(candidate)];
    else
      for (auto k = rp[sz(anchor)]; k < rp[sz(anchor) + 1]; ++k) {
        const Index j = ci[sz(k)];
        if (lp.A.vals[sz(k)] == 0.0 || lp.col_lo[sz(j)] == lp.col_hi[sz(j)])
          continue;
        if (sz(j) >= required.size() || !required[sz(j)] ||
            !std::isfinite(lp.col_lo[sz(j)]) ||
            !std::isfinite(lp.col_hi[sz(j)])) {
          eligible = false;
          break;
        }
        block.cols.push_back(j);
      }
    if (!eligible || block.cols.size() < 2 ||
        static_cast<int>(block.cols.size()) > max_columns ||
        !seen.insert(block.cols).second)
      continue;
    std::vector<f64> lower, upper;
    std::uint64_t states = 1;
    for (const Index j : block.cols) {
      const f64 lo = std::ceil(lp.col_lo[sz(j)]),
                hi = std::floor(lp.col_hi[sz(j)]);
      if (lo > hi) {
        infeasible_ = true;
        return;
      }
      const f64 width = hi - lo + 1.0;
      // Enumeration needs distinct, representable consecutive integers.
      if (std::fabs(lo) >= 0x1p52 || std::fabs(hi) >= 0x1p52 ||
          width > static_cast<f64>(budget / states)) {
        eligible = false;
        break;
      }
      states *= static_cast<std::uint64_t>(width);
      lower.push_back(lo);
      upper.push_back(hi);
    }
    if (!eligible)
      continue;
    struct Projection {
      Index row;
      std::vector<std::pair<Index, f64>> terms;
      model::DyadicSum outside_min, outside_max;
      bool min_finite = true, max_finite = true;
    };
    std::vector<Projection> rows;
    for (std::size_t q = 0; q < block.cols.size(); ++q)
      local[sz(block.cols[q])] = static_cast<Index>(q);
    std::vector<Index> touched;
    ++generation;
    for (const Index j : block.cols)
      for (auto k = columns.pattern.col_ptr()[sz(j)];
           k < columns.pattern.col_ptr()[sz(j) + 1]; ++k) {
        const Index i = columns.pattern.row_idx()[sz(k)];
        if (row_seen[sz(i)] != generation) {
          row_seen[sz(i)] = generation;
          touched.push_back(i);
        }
      }
    std::uint64_t nnz = 0;
    for (Index i : touched) {
      const auto len = static_cast<std::uint64_t>(rp[sz(i) + 1] - rp[sz(i)]);
      if (len > budget) {
        eligible = false;
        break;
      }
      budget -= len;
      Projection row;
      row.row = i;
      for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = lp.A.vals[sz(k)];
        if (a == 0)
          continue;
        if (local[sz(j)] >= 0) {
          row.terms.emplace_back(local[sz(j)], a);
          continue;
        }
        const f64 l = a > 0 ? lp.col_lo[sz(j)] : lp.col_hi[sz(j)];
        const f64 h = a > 0 ? lp.col_hi[sz(j)] : lp.col_lo[sz(j)];
        if (std::isfinite(l))
          row.outside_min.add_product(a, l);
        else
          row.min_finite = false;
        if (std::isfinite(h))
          row.outside_max.add_product(a, h);
        else
          row.max_finite = false;
      }
      if ((row.min_finite && std::isfinite(lp.row_hi[sz(i)])) ||
          (row.max_finite && std::isfinite(lp.row_lo[sz(i)]))) {
        nnz += row.terms.size() + 2;
        rows.push_back(std::move(row));
      }
    }
    for (const Index j : block.cols)
      local[sz(j)] = -1;
    if (!eligible || nnz == 0 || states > budget / nnz)
      continue;
    budget -= states * nnz;
    std::vector<f64> point = lower;
    for (std::uint64_t state = 0; state < states; ++state) {
      bool feasible = true;
      for (const auto &row : rows) {
        model::DyadicSum activity;
        for (const auto &[j, a] : row.terms)
          activity.add_product(a, point[sz(j)]);
        auto lower_activity = activity;
        lower_activity.add_sum(row.outside_min);
        auto upper_activity = activity;
        upper_activity.add_sum(row.outside_max);
        // The enumerated domain must contain every point admitted by the
        // solver's row-feasibility contract. Exact binary arithmetic alone
        // would reject integral decimal equalities (e.g. .9 * 60 + 6 = 60)
        // whose residual is smaller than that contract. Keep this allowance
        // in the projected domain, before pricing; a margin on the final
        // cut cannot restore patterns already discarded.
        lower_activity.add(-feasibility_tolerance);
        upper_activity.add(feasibility_tolerance);
        if ((row.max_finite && std::isfinite(lp.row_lo[sz(row.row)]) &&
             upper_activity.compare(lp.row_lo[sz(row.row)]) < 0) ||
            (row.min_finite && std::isfinite(lp.row_hi[sz(row.row)]) &&
             lower_activity.compare(lp.row_hi[sz(row.row)]) > 0)) {
          feasible = false;
          break;
        }
      }
      if (feasible)
        block.points.push_back(point);
      for (std::size_t q = 0; q < point.size(); ++q) {
        if (point[q] < upper[q]) {
          point[q] += 1.0;
          break;
        }
        point[q] = lower[q];
      }
    }
    if (block.points.empty()) {
      infeasible_ = true;
      return;
    }
    blocks_.push_back(std::move(block));
  }
}
std::vector<CutRow> FiniteBlockOracle::separate(const model::LpProblem &lp,
                                                const std::vector<f64> &x,
                                                int max_cuts, f64 violation,
                                                std::uint64_t budget) const {
  std::vector<CutRow> cuts;
  for (const auto &block : blocks_) {
    if (static_cast<int>(cuts.size()) >= max_cuts)
      break;
    std::vector<f64> target, projected = block.points.front(), direction;
    for (const Index j : block.cols)
      target.push_back(x[sz(j)]);
    // First price the true objective, then a geometric projection with
    // exact linear pricing. Approximate projection is fine: each returned
    // support inequality is certified over every enumerated pattern.
    for (int iteration = 0; iteration < 17; ++iteration) {
      const auto work = block.points.size() * block.cols.size();
      if (work > budget)
        break;
      budget -= work;
      direction.resize(target.size());
      for (std::size_t q = 0; q < direction.size(); ++q)
        direction[q] = iteration == 0 ? (lp.maximize ? -1.0 : 1.0) *
                                            lp.c[sz(block.cols[q])]
                                      : projected[q] - target[q];
      std::size_t minimizer = 0;
      auto theta = dot(direction, block.points.front());
      for (std::size_t q = 1; q < block.points.size(); ++q) {
        auto value = dot(direction, block.points[q]);
        auto difference = theta;
        difference.negate();
        difference.add_sum(value);
        if (difference.sign() < 0) {
          theta = std::move(value);
          minimizer = q;
        }
      }
      auto gap = dot(direction, target);
      gap.negate();
      gap.add_sum(theta);
      if (gap.compare(violation) > 0) {
        CutRow cut;
        cut.cols = block.cols;
        cut.vals = direction;
        cut.row_lo = theta.down();
        cut.exact = true;
        cut.used_local_bound = false;
        cut.name = "BLK_" + std::to_string(cuts.size());
        cuts.push_back(std::move(cut));
        break;
      }
      if (iteration == 0)
        continue;
      const auto &vertex = block.points[minimizer];
      long double numerator = 0.0L, denominator = 0.0L;
      for (std::size_t q = 0; q < projected.size(); ++q) {
        const long double step = vertex[q] - projected[q];
        numerator += (target[q] - projected[q]) * step;
        denominator += step * step;
      }
      if (!(denominator > 0.0L) || !(numerator > 0.0L))
        break;
      const f64 alpha =
          static_cast<f64>(std::clamp(numerator / denominator, 0.0L, 1.0L));
      for (std::size_t q = 0; q < projected.size(); ++q)
        projected[q] += alpha * (vertex[q] - projected[q]);
    }
  }
  return cuts;
}
bool FiniteBlockOracle::augment(const model::LpProblem &lp, std::vector<f64> &x,
                                std::uint64_t budget) const {
  if (x.size() != lp.c.size())
    return false;
  const f64 sense = lp.maximize ? -1 : 1;
  bool changed = false;
  const auto &rp = lp.A.pattern.row_ptr();
  const auto &ci = lp.A.pattern.col_idx();
  auto objective = [&](const std::vector<f64> &point) {
    model::DyadicSum value;
    for (Index j = 0; j < lp.n_cols(); ++j)
      value.add_product(sense * lp.c[sz(j)], point[sz(j)]);
    return value;
  };
  auto best = objective(x);
  for (const auto &block : blocks_)
    for (const auto &pattern : block.points) {
      const std::uint64_t work =
          static_cast<std::uint64_t>(lp.nnz() + lp.n_cols());
      if (work > budget)
        return changed;
      budget -= work;
      auto candidate = x;
      for (std::size_t q = 0; q < block.cols.size(); ++q)
        candidate[sz(block.cols[q])] = pattern[q];
      auto value = objective(candidate);
      auto difference = value;
      auto old = best;
      old.negate();
      difference.add_sum(old);
      if (difference.sign() >= 0)
        continue;
      bool feasible = true;
      for (Index j = 0; j < lp.n_cols(); ++j)
        if (candidate[sz(j)] < lp.col_lo[sz(j)] ||
            candidate[sz(j)] > lp.col_hi[sz(j)]) {
          feasible = false;
          break;
        }
      for (Index i = 0; i < lp.n_rows() && feasible; ++i) {
        model::DyadicSum activity;
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
          activity.add_product(lp.A.vals[sz(k)], candidate[sz(ci[sz(k)])]);
        if ((std::isfinite(lp.row_lo[sz(i)]) &&
             activity.compare(lp.row_lo[sz(i)]) < 0) ||
            (std::isfinite(lp.row_hi[sz(i)]) &&
             activity.compare(lp.row_hi[sz(i)]) > 0))
          feasible = false;
      }
      if (feasible) {
        x = std::move(candidate);
        best = std::move(value);
        changed = true;
      }
    }
  return changed;
}

} // namespace forge::search
