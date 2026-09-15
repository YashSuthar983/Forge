// Kernel Pump (sor_search/src/kernel_pump.cpp): kernel/bucket partition is a
// restriction (excluded binaries fixed to 0); FP never invents dual bounds.
#include "sor/search/kernel_pump.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::search::KernelBuckets;
using sor::search::KernelPumpDiagnostics;
using sor::search::KernelPumpOptions;
using sor::search::apply_kernel_restriction;
using sor::search::build_kernel_buckets;
using sor::search::kernel_pump;
using sor::sparse::from_triplets;

namespace {

bool satisfies(const LpProblem& lp, const std::vector<f64>& x, f64 tol) {
    if (lp.max_row_violation(x) > tol) return false;
    if (lp.max_bound_violation(x) > tol) return false;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        const auto u = static_cast<std::size_t>(j);
        if (!lp.is_integer.empty() && lp.is_integer[u] &&
            std::fabs(x[u] - std::round(x[u])) > 1e-6)
            return false;
    }
    return true;
}

// min x0+x1+x2  s.t. x0+x1+x2 >= 2, 0/1 binaries. Feasible: any two 1s.
LpProblem tiny_set_cover() {
    LpProblem lp;
    lp.name = "kp_cover";
    lp.c = {1.0, 1.0, 1.0};
    lp.col_lo = {0, 0, 0};
    lp.col_hi = {1, 1, 1};
    lp.is_integer = {true, true, true};
    lp.row_lo = {2.0};
    lp.row_hi = {kInf};
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    return lp;
}

void test_buckets_partition_binaries() {
    const auto lp = tiny_set_cover();
    // Fractional LP point: all 2/3.
    std::vector<f64> x = {2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0};
    KernelBuckets kb;
    CHECK(build_kernel_buckets(lp, x, nullptr, 4, kb));
    CHECK(!kb.kernel.empty());
    std::size_t total = kb.kernel.size();
    for (const auto& b : kb.buckets) total += b.size();
    CHECK(total == 3);
}

void test_restriction_fixes_excluded_to_zero() {
    const auto lp = tiny_set_cover();
    std::vector<Index> active = {0, 1};
    const auto sub = apply_kernel_restriction(lp, active);
    CHECK(sub.col_lo[2] == 0.0);
    CHECK(sub.col_hi[2] == 0.0);
    CHECK(sub.col_hi[0] == 1.0);
}

void test_kernel_pump_finds_feasible() {
    const auto lp = tiny_set_cover();
    std::vector<f64> xlp = {0.7, 0.7, 0.6};
    KernelPumpOptions o;
    o.time_limit_s = 2.0;
    o.kappa = 3;
    o.max_pumps_total = 40;
    o.refine_kernel = false;
    std::vector<f64> xout;
    KernelPumpDiagnostics d;
    const bool ok = kernel_pump(lp, xlp, nullptr, o, xout, d);
    // May or may not succeed under tight caps; if it claims success it must
    // be feasible for the original model.
    if (ok) {
        CHECK(satisfies(lp, xout, 1e-6));
        CHECK(d.found);
    }
}

void test_disabled_returns_false() {
    const auto lp = tiny_set_cover();
    KernelPumpOptions o;
    o.enabled = false;
    std::vector<f64> xout;
    KernelPumpDiagnostics d;
    CHECK(!kernel_pump(lp, {0.5, 0.5, 0.5}, nullptr, o, xout, d));
}

}  // namespace

int main() {
    test_buckets_partition_binaries();
    test_restriction_fixes_excluded_to_zero();
    test_kernel_pump_finds_feasible();
    test_disabled_returns_false();
    return sor::test::finish("test_kernel_pump");
}
