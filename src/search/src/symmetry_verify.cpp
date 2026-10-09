#include "sor/search/symmetry_verify.hpp"
#include "sor/model/exact.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace sor::search {
using core::f64;
using core::Index;
using model::Rational;
namespace {
struct Side {
  int infinity = 0;
  Rational value;
  explicit Side(f64 v)
      : infinity(std::isfinite(v) ? 0 : (v < 0 ? -1 : 1)),
        value(infinity == 0 ? Rational(v) : Rational(0)) {}
  bool operator==(const Side &) const = default;
  bool operator<(const Side &other) const {
    return infinity != other.infinity ? infinity < other.infinity
                                      : value < other.value;
  }
  Side transformed(int sign, f64 shift) const {
    Side out = *this;
    out.infinity *= sign;
    if (infinity == 0)
      out.value = sign * value + Rational(shift);
    return out;
  }
  void subtract(const Rational &offset) {
    if (infinity == 0)
      value -= offset;
  }
};
struct Row {
  Side lo, hi;
  std::vector<std::pair<Index, Rational>> entries;
  Row(f64 lower, f64 upper) : lo(lower), hi(upper) {}
  bool operator<(const Row &other) const {
    if (!(lo == other.lo))
      return lo < other.lo;
    if (!(hi == other.hi))
      return hi < other.hi;
    return entries < other.entries;
  }
};
std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
} // namespace
struct SymmetryVerifier::Impl {
  model::LpProblem model;
  std::set<Row> rows;
  explicit Impl(const model::LpProblem &input) : model(input) {
    const auto &rp = model.A.pattern.row_ptr();
    const auto &ci = model.A.pattern.col_idx();
    for (Index i = 0; i < model.n_rows(); ++i) {
      Row row(model.row_lo[sz(i)], model.row_hi[sz(i)]);
      std::map<Index, Rational> entries;
      for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
        entries[ci[static_cast<std::size_t>(k)]] +=
            Rational(model.A.vals[static_cast<std::size_t>(k)]);
      for (auto &[j, a] : entries)
        if (a != 0)
          row.entries.emplace_back(j, std::move(a));
      rows.insert(std::move(row));
    }
  }
};
SymmetryVerifier::SymmetryVerifier(const model::LpProblem &model)
    : impl_(std::make_unique<Impl>(model)) {}
SymmetryVerifier::~SymmetryVerifier() = default;
bool SymmetryVerifier::verifies(const std::vector<Index> &image,
                                const std::vector<int> &sign,
                                const std::vector<f64> &shift) const {
  const auto &lp = impl_->model;
  const Index n = lp.n_cols();
  if (image.size() != sz(n) || sign.size() != sz(n) || shift.size() != sz(n))
    return false;
  std::vector<char> seen(sz(n), false);
  Rational constant = 0;
  for (Index j = 0; j < n; ++j) {
    const Index t = image[sz(j)];
    if (t < 0 || t >= n || seen[sz(t)] ||
        (sign[sz(j)] != 1 && sign[sz(j)] != -1) || !std::isfinite(shift[sz(j)]))
      return false;
    seen[sz(t)] = true;
    if (!lp.is_integer.empty() &&
        (lp.is_integer[sz(j)] != lp.is_integer[sz(t)] ||
         (lp.is_integer[sz(j)] && shift[sz(j)] != std::round(shift[sz(j)]))))
      return false;
    if (lp.c[sz(j)] * sign[sz(j)] != lp.c[sz(t)])
      return false;
    constant += Rational(lp.c[sz(j)]) * Rational(shift[sz(j)]);
    const Side lo(sign[sz(j)] > 0 ? lp.col_lo[sz(t)] : lp.col_hi[sz(t)]);
    const Side hi(sign[sz(j)] > 0 ? lp.col_hi[sz(t)] : lp.col_lo[sz(t)]);
    if (!(lo.transformed(sign[sz(j)], shift[sz(j)]) ==
          Side(lp.col_lo[sz(j)])) ||
        !(hi.transformed(sign[sz(j)], shift[sz(j)]) == Side(lp.col_hi[sz(j)])))
      return false;
  }
  if (constant != 0)
    return false;
  for (const auto &row : impl_->rows) {
    Row transformed = row;
    Rational offset = 0;
    for (auto &[j, a] : transformed.entries) {
      offset += a * Rational(shift[sz(j)]);
      a *= sign[sz(j)];
      j = image[sz(j)];
    }
    std::sort(transformed.entries.begin(), transformed.entries.end());
    transformed.lo.subtract(offset);
    transformed.hi.subtract(offset);
    if (!impl_->rows.contains(transformed))
      return false;
  }
  // The invertible affine action is injective on row data. An injection of
  // this finite row set into itself is onto, so the inverse preserves it too.
  return true;
}
} // namespace sor::search
