// Focused invariants for dual-cutoff caching and MIP binary dominance.
#include "sor/model/lp.hpp"
#include "sor/search/column_dominance.hpp"
#include "sor/search/dual_cutoffs.hpp"
#include "sor/sparse/csr.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using sor::certify::RcEnclosure;
using sor::core::f64;
using sor::core::Index;
using sor::search::apply_binary_column_dominance;
using sor::search::ColumnDominanceDiagnostics;
using sor::search::ColumnDominanceOptions;
using sor::search::DualCutoffDiagnostics;
using sor::search::DualCutoffStore;

static int fails = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++fails;                                                                 \
    }                                                                          \
  } while (0)

static void test_dual_cutoff_propagates() {
  DualCutoffStore store;
  DualCutoffDiagnostics diag;
  RcEnclosure enc;
  enc.finite = true;
  enc.bound = 0.0;
  enc.lo = {2.0, 0.0};
  enc.hi = {2.0, 0.0};
  std::vector<f64> box_lo = {0.0, 0.0}, box_hi = {10.0, 1.0};
  CHECK(store.consider(enc, box_lo, box_hi, /*generation=*/1,
                       /*root_scope=*/true, &diag));
  CHECK(diag.retained == 1);

  std::vector<f64> lo = {0.0, 0.0}, hi = {10.0, 1.0};
  std::vector<char> mask = {1, 1};
  // cutoff_min = 1: room = 1, so x0 <= 0 + floor(1/2) = 0.
  CHECK(store.propagate(lo, hi, /*cutoff_min=*/1.0, mask, /*generation=*/1,
                        nullptr, 0, &diag));
  CHECK(hi[0] <= 0.0 + 1e-12);
  CHECK(diag.tightenings >= 1);

  // Sibling box (not nested): must not apply.
  std::vector<f64> sib_lo = {0.0, 0.0}, sib_hi = {10.0, 1.0};
  sib_lo[0] = -1.0; // wider than the cert box
  DualCutoffDiagnostics sib;
  const f64 hi0 = hi[0];
  CHECK(store.propagate(sib_lo, sib_hi, 1.0, mask, /*generation=*/1, nullptr, 0,
                        &sib));
  CHECK(sib.rejected_stale >= 1);

  // Older global witnesses remain necessary after valid row-side edits.
  DualCutoffDiagnostics stale;
  CHECK(
      store.propagate(lo, hi, 1.0, mask, /*generation=*/2, nullptr, 0, &stale));
  CHECK(hi[0] == hi0);
  CHECK(stale.rejected_stale == 0);
}

static void test_binary_dominance_fixes() {
  sor::model::LpProblem lp;
  lp.maximize = false;
  lp.c = {1.0, 2.0}; // x0 cheaper than x1
  lp.col_lo = {0.0, 0.0};
  lp.col_hi = {1.0, 1.0};
  lp.is_integer = {true, true};
  lp.row_lo = {-sor::model::kInf};
  lp.row_hi = {1.0};
  // x0 + x1 <= 1: one-sided packing, x0 dominates x1 on cost with equal coeffs.
  lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});

  ColumnDominanceOptions opts;
  ColumnDominanceDiagnostics diag;
  const auto n =
      apply_binary_column_dominance(lp, lp.col_lo, lp.col_hi, opts, &diag);
  CHECK(n == 1);
  CHECK(lp.col_hi[1] <= 0.0 + 1e-12);
  CHECK(diag.dominated_fixed == 1);

  // Equality support must not fix a "dominated" binary: (0,1,1) is the only
  // feasible point of 3x+2y+2z = 4, and y does not dominate z there.
  sor::model::LpProblem eq;
  eq.maximize = false;
  eq.c = {-1., 1., 2.};
  eq.col_lo = {0., 0., 0.};
  eq.col_hi = {1., 1., 1.};
  eq.is_integer = {true, true, true};
  eq.row_lo = eq.row_hi = {4.};
  eq.A = sor::sparse::from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {3., 2., 2.});
  ColumnDominanceDiagnostics eqd;
  CHECK(apply_binary_column_dominance(eq, eq.col_lo, eq.col_hi, opts, &eqd) ==
        0);
  CHECK(eq.col_hi[1] > 0.5 && eq.col_hi[2] > 0.5);
}

static void test_cached_anchors_and_joint_witnesses() {
  DualCutoffStore store;
  RcEnclosure enc;
  enc.finite = true;
  enc.bound = 0;
  enc.lo = {2, 3};
  enc.hi = enc.lo;
  CHECK(store.consider(enc, {0, 0}, {10, 10}, 1, true));
  std::vector<f64> lo{2, 1}, hi{10, 10};
  // 2x+3y<12, with x>=2,y>=1: x<=4,y<=2. The original
  // corners are needed to charge the seven objective units already spent.
  CHECK(store.propagate(lo, hi, 12, {1, 1}, 2, nullptr, 0));
  CHECK(hi[0] == 4 && hi[1] == 2);
  lo = {5, 1};
  hi = {10, 10};
  CHECK(!store.propagate(lo, hi, 12, {1, 1}, 2, nullptr, 0));
  // Exact nesting: a slightly wider box is still a different domain.
  lo = {std::nextafter(0., -1.), 0};
  hi = {10, 10};
  DualCutoffDiagnostics diag;
  CHECK(store.propagate(lo, hi, 12, {1, 1}, 2, nullptr, 0, &diag));
  CHECK(diag.rejected_stale == 1 && hi[0] == 10);

  DualCutoffStore joint;
  enc.lo = {2, 0};
  enc.hi = enc.lo;
  CHECK(joint.consider(enc, {0, 0}, {10, 10}, 1, true));
  enc.lo = {0, 3};
  enc.hi = enc.lo;
  CHECK(joint.consider(enc, {0, 0}, {10, 10}, 2, true));
  CHECK(joint.size() == 2);
  lo = {0, 0};
  hi = {10, 10};
  CHECK(joint.propagate(lo, hi, 7, {1, 1}, 3, nullptr, 0));
  CHECK(hi[0] == 3 && hi[1] == 2);
  // Continuous endpoints are outward, too; no integer-only inference cap.
  lo = {0, 0};
  hi = {10, 10};
  CHECK(joint.propagate(lo, hi, 7, {0, 0}, 3, nullptr, 0));
  CHECK(hi[0] >= 3.5 && hi[0] <= std::nextafter(3.5, 4.));
  CHECK(hi[1] >= 7. / 3. && hi[1] <= std::nextafter(7. / 3., 3.));
}

static f64 binary_optimum(const sor::model::LpProblem &lp,
                          const std::vector<f64> &lo,
                          const std::vector<f64> &hi) {
  f64 best = std::numeric_limits<f64>::infinity();
  for (int mask = 0; mask < 8; ++mask) {
    std::vector<f64> x(3);
    bool inside = true;
    for (int j = 0; j < 3; ++j) {
      x[j] = (mask >> j) & 1;
      inside = inside && x[j] >= lo[j] && x[j] <= hi[j];
    }
    if (inside && lp.max_row_violation(x) == 0)
      best = std::min(best, (lp.maximize ? -1. : 1.) * lp.objective(x));
  }
  return best;
}
static void test_exchange_theorem_over_binary_domains() {
  std::mt19937 rng(781);
  for (int trial = 0; trial < 1000; ++trial) {
    sor::model::LpProblem lp;
    lp.c.resize(3);
    lp.is_integer.assign(3, true);
    lp.col_lo.assign(3, 0);
    lp.col_hi.assign(3, 1);
    lp.maximize = (trial % 2) != 0;
    for (auto &c : lp.c)
      c = static_cast<int>(rng() % 7) - 3;
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        rows.push_back(i);
        cols.push_back(j);
        vals.push_back(static_cast<int>(rng() % 7) - 3);
      }
      const double b = static_cast<int>(rng() % 9) - 4;
      lp.row_lo.push_back(i == trial % 3 ? b : -sor::model::kInf);
      lp.row_hi.push_back(i == (trial + 1) % 3 ? b : sor::model::kInf);
    }
    lp.A = sor::sparse::from_triplets(3, 3, rows, cols, vals);
    const f64 original = binary_optimum(lp, lp.col_lo, lp.col_hi);
    auto lo = lp.col_lo, hi = lp.col_hi;
    apply_binary_column_dominance(lp, lo, hi, {}, nullptr);
    CHECK(binary_optimum(lp, lo, hi) == original);
  }
  // A negative-cost dominated packing column cannot just be dropped:
  // both columns belong to the unique optimum when capacity permits both.
  sor::model::LpProblem lp;
  lp.c = {-2, -1, 0};
  lp.col_lo = {0, 0, 0};
  lp.col_hi = {1, 1, 0};
  lp.is_integer = {true, true, true};
  lp.row_lo = {-sor::model::kInf};
  lp.row_hi = {2};
  lp.A = sor::sparse::from_triplets(1, 3, {0, 0}, {0, 1}, {1, 1});
  auto lo = lp.col_lo, hi = lp.col_hi;
  CHECK(apply_binary_column_dominance(lp, lo, hi, {}, nullptr) == 0);
  lp.row_hi = {1};
  lo = lp.col_lo;
  hi = lp.col_hi;
  CHECK(apply_binary_column_dominance(lp, lo, hi, {}, nullptr) == 1);
  CHECK(hi[0] == 1 && hi[1] == 0);
  // Proof premises cannot accept an approximately equal worse cost.
  lp.c = {-1, -1 - 0x1p-40, 0};
  lo = lp.col_lo;
  hi = lp.col_hi;
  apply_binary_column_dominance(lp, lo, hi, {}, nullptr);
  CHECK(hi[0] == 0 && hi[1] == 1);
}

int main() {
  test_dual_cutoff_propagates();
  test_binary_dominance_fixes();
  test_cached_anchors_and_joint_witnesses();
  test_exchange_theorem_over_binary_domains();
  if (fails) {
    std::fprintf(stderr, "%d checks failed\n", fails);
    return 1;
  }
  std::printf("test_dual_cutoffs_dominance: ok\n");
  return 0;
}
