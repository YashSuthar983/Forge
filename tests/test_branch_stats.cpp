// Branching evidence bookkeeping (search/branch_stats.hpp): what counts as an
// observation, once-per-branch consumption, and the batch allowance rule.
#include "sor/search/branch_stats.hpp"

#include "test_helpers.hpp"

#include <cmath>

using sor::search::batch_has_allowance;
using sor::search::BranchKey;
using sor::search::BranchOutcome;
using sor::search::branch_key;
using sor::search::feasibility_means;
using sor::search::inference_unobserved;
using sor::search::key_better;
using sor::search::score_bucket;
using sor::search::BranchStats;
using sor::core::f64;

namespace {

static double core_nan() { return std::nan(""); }
BranchStats fresh(int n = 4) {
    BranchStats s;
    s.resize(n);
    return s;
}

void test_proved_zero_gain_counts_once() {
    auto s = fresh();
    bool consumed = false;
    f64 unit = std::nan("");
    CHECK(s.record_child(1, -1, 0.5, BranchOutcome::ProvedOptimal, 0.0, 3.0, consumed, unit));
    CHECK(consumed);
    CHECK(s.pc_count[0][1] == 1);
    CHECK_NEAR(s.pc_sum[0][1], 0.0, 1e-15);
    CHECK(s.objective_samples == 1);
    // The same incoming branch again (requeued after a deduction, stronger
    // relaxation): the count stays 1 and the sample is REPLACED, not added.
    CHECK(s.record_child(1, -1, 0.5, BranchOutcome::ProvedOptimal, 2.0, 3.0, consumed, unit));
    CHECK(s.pc_count[0][1] == 1);
    CHECK(s.replaced == 1);
    CHECK_NEAR(s.pc_sum[0][1], 4.0, 1e-12);
    CHECK(s.objective_samples == 1);
    // A repeat that carries no objective sample changes nothing.
    CHECK(!s.record_child(1, -1, 0.5, BranchOutcome::CertifiedInfeasible, core_nan(), 3.0,
                          consumed, unit));
    CHECK(s.ignored_repeat == 1);
    CHECK_NEAR(s.pc_sum[0][1], 4.0, 1e-12);
}

void test_incomplete_contributes_nothing() {
    auto s = fresh();
    bool consumed = false;
    f64 unit = std::nan("");
    CHECK(!s.record_child(2, +1, 0.4, BranchOutcome::Incomplete, 5.0, 1.0, consumed, unit));
    CHECK(!consumed);                       // still available for a real outcome
    CHECK(s.pc_count[1][2] == 0 && s.observed[1][2] == 0);
    CHECK(s.ignored_incomplete == 1);
    // The later, proved solve of the same node is the one sample.
    CHECK(s.record_child(2, +1, 0.4, BranchOutcome::ProvedOptimal, 1.0, 1.0, consumed, unit));
    CHECK(s.pc_count[1][2] == 1);
    CHECK_NEAR(s.pc_sum[1][2], 2.5, 1e-12);
}

void test_certified_closures_are_separate_statistics() {
    auto s = fresh();
    CHECK(s.record(0, -1, 0.5, BranchOutcome::CertifiedInfeasible, core_nan(), -1.0));
    CHECK(s.record(0, -1, 0.5, BranchOutcome::ProvedOptimal, 1.0, 4.0));
    CHECK(s.closed[0][0] == 1 && s.observed[0][0] == 2);
    CHECK_NEAR(s.closure_rate(0, -1), 0.5, 1e-12);
    CHECK(s.pc_count[0][0] == 1);           // the closure is not an objective sample
    CHECK_NEAR(s.inference_rate(0, -1), 4.0, 1e-12);   // only the measured one
    CHECK(std::isnan(s.closure_rate(0, +1)));
    // A cutoff after a proved solve is both a closure and an objective sample.
    CHECK(s.record(3, +1, 0.5, BranchOutcome::CertifiedCutoff, 3.0, 0.0));
    CHECK(s.closed[1][3] == 1 && s.pc_count[1][3] == 1);
    CHECK_NEAR(s.pc_sum[1][3], 6.0, 1e-12);
}

void test_rates_not_counts() {
    auto s = fresh();
    for (int i = 0; i < 100; ++i) s.record(0, -1, 0.5, BranchOutcome::ProvedOptimal, 0.0, 0.0);
    s.record(1, -1, 0.5, BranchOutcome::CertifiedInfeasible, core_nan(), 0.0);
    // Column 0 was visited 100x, column 1 once and always closed.
    CHECK(s.closure_rate(1, -1) > s.closure_rate(0, -1));
}

void test_batch_allowance_counts_running_batch() {
    // Earlier batches used 10 ms of a 12 ms allowance (share 0.1 of 100 ms + 2 ms).
    CHECK(batch_has_allowance(10.0, 0.0, 0.1, 100.0, 2.0));
    CHECK(batch_has_allowance(10.0, 2.0, 0.1, 100.0, 2.0));
    CHECK(!batch_has_allowance(10.0, 2.5, 0.1, 100.0, 2.0));   // this batch alone exhausted it
    CHECK(!batch_has_allowance(13.0, 0.0, 0.1, 100.0, 2.0));
}

void test_branch_key_ordering() {
    auto s = fresh(6);
    // Column 0: visited 100x, never closed. Column 1: seen once, closed. Column 2: unobserved.
    for (int i = 0; i < 100; ++i) s.record(0, -1, 0.5, BranchOutcome::ProvedOptimal, 0.0, 0.0);
    s.record(1, -1, 0.5, BranchOutcome::CertifiedInfeasible, core_nan(), 5.0);
    const auto m = feasibility_means(s);
    const double flat = 1e-12;   // identical objective scores
    const auto k0 = branch_key(s, 0, flat, m), k1 = branch_key(s, 1, flat, m), k2 = branch_key(s, 2, flat, m);
    CHECK(k0.bucket == k1.bucket && k1.bucket == k2.bucket);
    CHECK(key_better(k1, k0));            // rate beats exposure
    CHECK(key_better(k1, k2));            // observed-closing beats unknown (= model mean)
    CHECK(key_better(k2, k0));            // unknown (mean) beats observed-never-closing
    // A clearly better objective score always wins, whatever the feasibility history.
    CHECK(key_better(branch_key(s, 0, 1.0, m), k1));
    // Scores equal to numerical resolution share a bucket; distinct scores do not.
    CHECK(score_bucket(1.0) == score_bucket(1.0 + 1e-12));
    CHECK(score_bucket(1.0) != score_bucket(1.001));
    CHECK(score_bucket(std::numeric_limits<double>::infinity()) ==
          std::numeric_limits<long long>::max());
    CHECK(score_bucket(std::numeric_limits<double>::max()) > score_bucket(1.));
    CHECK(score_bucket(core_nan()) == score_bucket(0.));
    CHECK(!inference_unobserved(s, 0));
    CHECK(inference_unobserved(s, 2));
    // An LP observation without a propagation measurement must leave the
    // candidate eligible for an inference probe, even in both directions.
    s.record(2, -1, .5, BranchOutcome::ProvedOptimal, 0., -1.);
    s.record(2, +1, .5, BranchOutcome::ProvedOptimal, 0., -1.);
    CHECK(inference_unobserved(s, 2));
    s.record(2, -1, .5, BranchOutcome::DomainOnly, core_nan(), 0.);
    CHECK(!inference_unobserved(s, 2));
    // The order is strict: no key is better than itself.
    CHECK(!key_better(k0, k0));
}

}  // namespace

int main() {
    test_branch_key_ordering();
    test_proved_zero_gain_counts_once();
    test_incomplete_contributes_nothing();
    test_certified_closures_are_separate_statistics();
    test_rates_not_counts();
    test_batch_allowance_counts_running_batch();
    return sor::test::finish("test_branch_stats");
}
