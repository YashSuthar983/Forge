// Dual ratio test (CHUZC): the quantities the caller pivots and shifts on.
//
// dual_ratio_test() had no unit test of its own. It gained one when the dual's
// wrong-sign entering-column shift started reading `row_max_alpha` from it to
// decide whether a wrong-sign pivot can do observable damage: if that number
// is wrong, the shift fires on the wrong iterations, and the only signal is a
// model somewhere losing its proof.
#include "sor/engines/dual_ratio_test.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::engines::DualRatioInput;
using sor::engines::DualRatioResult;
using sor::engines::DualRatioWorkspace;
using sor::engines::NonbasicStatus;

namespace {

constexpr f64 kInf = std::numeric_limits<f64>::infinity();

struct Row {
    std::vector<Index> j;
    std::vector<f64> alpha;
    std::vector<f64> dual;
    std::vector<NonbasicStatus> status;
    std::vector<f64> lo;
    std::vector<f64> hi;
};

DualRatioResult run(const Row& r, f64 delta_primal, f64 srow,
                    f64 pivot_tol = 1e-7, f64 rel_pivot_tol = 1e-9) {
    DualRatioInput in;
    in.cand_j = &r.j;
    in.cand_alpha = &r.alpha;
    in.cand_dual = &r.dual;
    in.status = &r.status;
    in.lo = &r.lo;
    in.hi = &r.hi;
    in.delta_primal = delta_primal;
    in.srow = srow;
    in.slack = 1e-9;
    in.pivot_tol = pivot_tol;
    in.rel_pivot_tol = rel_pivot_tol;
    in.allow_flips = true;
    DualRatioWorkspace ws;
    return dual_ratio_test(in, ws);
}

// row_max_alpha is the magnitude of the whole pivotal row, measured BEFORE the
// relative floor drops anything -- it is what the floor itself is derived from.
// A candidate that the floor then excludes must still have counted, otherwise
// the row would appear smaller than it is exactly when it is most spread out,
// which is the case the caller's damage estimate exists for.
void test_row_max_alpha_spans_the_whole_row() {
    Row r;
    r.j = {0, 1, 2};
    r.alpha = {-1.0e-6, 40.0, 2.0};
    r.dual = {1.0, 3.0, 0.5};
    r.status = {NonbasicStatus::AtLower, NonbasicStatus::AtLower,
                NonbasicStatus::AtLower};
    r.lo = {0.0, 0.0, 0.0};
    r.hi = {kInf, kInf, kInf};

    // Leaving to lower with srow = -1 makes every negative-alpha column
    // eligible; the caller has already applied that filter, so the test simply
    // hands all three over.
    const auto out = run(r, 1.0, -1.0, 1e-9, 0.1);
    CHECK(out.ok);
    CHECK_NEAR(out.row_max_alpha, 40.0, 0.0);
    // The relative floor is 0.1 * 40 = 4, so both the 1e-6 and the 2.0 entry
    // are dropped as unreliable -- and the row is still 40 wide.
    CHECK(out.excluded_small == 2);
    CHECK(out.pivot == 1);
    CHECK_NEAR(std::fabs(out.alpha_enter), 40.0, 0.0);
}

// The same row with the floor out of the way: the reported magnitude does not
// depend on how many candidates survive.
void test_row_max_alpha_is_independent_of_the_floor() {
    Row r;
    r.j = {0, 1, 2};
    r.alpha = {-1.0e-6, 40.0, 2.0};
    r.dual = {1.0, 3.0, 0.5};
    r.status = {NonbasicStatus::AtLower, NonbasicStatus::AtLower,
                NonbasicStatus::AtLower};
    r.lo = {0.0, 0.0, 0.0};
    r.hi = {kInf, kInf, kInf};

    const auto out = run(r, 1.0, -1.0, 1e-12, 1e-12);
    CHECK(out.ok);
    CHECK_NEAR(out.row_max_alpha, 40.0, 0.0);
    CHECK(out.excluded_small == 0);
}

// An empty candidate list cannot report a magnitude, and must not pretend to.
// The caller's damage estimate divides by alpha_enter, so a bogus non-zero
// here would be read as a real row.
void test_no_candidates_reports_no_row() {
    Row r;
    const auto out = run(r, 1.0, 1.0);
    CHECK(!out.ok);
    CHECK(out.pivot == -1);
    CHECK(out.row_max_alpha == 0.0);
}

// A wrong-sign entering column is reported as such, with the signed alpha and
// the unshifted reduced cost the caller needs to size the damage. Here the
// only candidate has a reduced cost on the infeasible side of zero.
void test_wrong_sign_entering_is_reported_with_its_row_entry() {
    Row r;
    r.j = {0};
    r.alpha = {-3.0};
    r.dual = {-1.0e-9};   // AtLower wants d >= 0
    r.status = {NonbasicStatus::AtLower};
    r.lo = {0.0};
    r.hi = {kInf};

    const auto out = run(r, 1.0, -1.0);
    CHECK(out.ok);
    CHECK(out.pivot == 0);
    CHECK(out.enter_wrong_sign);
    CHECK_NEAR(out.d_enter, -1.0e-9, 0.0);
    CHECK_NEAR(out.alpha_enter, -3.0, 0.0);
    CHECK_NEAR(out.row_max_alpha, 3.0, 0.0);
}

// The O(k) path exists to stop sorting the pivotal row on every iteration.
// A row of one-sided candidates can never pass a group, so the first group is
// always final and nothing may be sorted at all.
void test_one_sided_row_sorts_nothing() {
    Row r;
    for (Index j = 0; j < 64; ++j) {
        r.j.push_back(j);
        r.alpha.push_back(-(1.0 + static_cast<f64>(j % 7)));
        r.dual.push_back(0.5 * static_cast<f64>(j % 5));
        r.status.push_back(NonbasicStatus::AtLower);
        r.lo.push_back(0.0);
        r.hi.push_back(kInf);        // one-sided: never flippable
    }
    const auto out = run(r, 3.0, -1.0);
    CHECK(out.ok);
    CHECK(out.sorted_candidates == 0);
    CHECK(out.flip_count == 0);
    CHECK(out.groups == 1);
}

// ...and the same row with flips allowed but every candidate non-flippable
// still sorts nothing, because the first group holds a non-flippable member.
void test_boxed_row_still_sorts_nothing_when_the_first_group_blocks() {
    Row r;
    for (Index j = 0; j < 16; ++j) {
        r.j.push_back(j);
        r.alpha.push_back(-2.0);
        r.dual.push_back(1.0);        // identical ratios: one big first group
        r.status.push_back(NonbasicStatus::AtLower);
        r.lo.push_back(0.0);
        r.hi.push_back(kInf);
    }
    const auto out = run(r, 100.0, -1.0);
    CHECK(out.ok);
    CHECK(out.sorted_candidates == 0);
}

// Harris's band is invariant: no candidate may end a step below -slack. Column
// 0 sits inside the band at -0.9 slack; column 1 is feasible with the larger
// |alpha|. Clamping column 0's reduced cost to 0 before adding the slack gave it
// a full slack of room, put both columns in one group and pivoted on column 1
// (step 2.5e-10), leaving column 0 at -1.15e-9 -- outside the band, and a step
// further out on every later pivot. With the raw bound column 0 blocks first.
void test_harris_band_is_invariant_for_wrong_signed_candidates() {
    Row r;
    r.j = {0, 1};
    r.alpha = {-1.0, -2.0};
    r.dual = {-0.9e-9, 0.5e-9};
    r.status = {NonbasicStatus::AtLower, NonbasicStatus::AtLower};
    r.lo = {0.0, 0.0};
    r.hi = {kInf, kInf};
    const auto out = run(r, 1.0, -1.0);   // slack 1e-9
    CHECK(out.ok);
    CHECK(out.pivot == 0);
    // Whatever enters, the step must keep every other candidate in the band.
    const f64 t = out.theta;
    for (std::size_t c = 0; c < r.j.size(); ++c) {
        if (r.j[c] == out.pivot) continue;
        CHECK(r.dual[c] - t * std::fabs(r.alpha[c]) >= -1e-9);
    }
}

// The differential test. A speedup that changes a pivot is not a speedup, and
// the only way to know is to run both implementations on the same input.
// Rows are randomized over the shapes that matter: mixed bound classes, ties
// in ratio, wrong-sign candidates, tiny alphas near the relative floor, and a
// slope large enough that the long step passes several groups.
void test_fast_paths_agree_with_the_exhaustive_reference() {
    std::mt19937 rng(20260908u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int with_flips = 0, sorted_free = 0;

    for (int trial = 0; trial < 4000; ++trial) {
        const Index n = 1 + static_cast<Index>(rng() % 40);
        Row r;
        for (Index j = 0; j < n; ++j) {
            r.j.push_back(j);
            // Mix magnitudes so the relative pivot floor excludes some.
            const double mag = (rng() % 8 == 0) ? 1e-8 + 1e-9 * unit(rng)
                                                : 0.05 + 4.0 * unit(rng);
            r.alpha.push_back(-mag);
            // Ties matter: quantize some duals so ratios collide.
            r.dual.push_back((rng() % 3 == 0) ? 0.25 * static_cast<double>(rng() % 4)
                                              : 2.0 * unit(rng) - 0.25);
            r.status.push_back(NonbasicStatus::AtLower);
            r.lo.push_back(0.0);
            switch (rng() % 3) {
                case 0:  r.hi.push_back(kInf); break;               // one-sided
                case 1:  r.hi.push_back(0.1 + 2.0 * unit(rng)); break;  // boxed
                default: r.hi.push_back(0.5); break;                 // boxed, tied range
            }
        }
        const double delta = (rng() % 4 == 0) ? 50.0 * unit(rng) : 2.0 * unit(rng);
        const bool allow = (rng() % 8) != 0;

        DualRatioWorkspace ws_fast, ws_ref;
        DualRatioInput in;
        in.cand_j = &r.j;
        in.cand_alpha = &r.alpha;
        in.cand_dual = &r.dual;
        in.status = &r.status;
        in.lo = &r.lo;
        in.hi = &r.hi;
        in.delta_primal = delta;
        in.srow = -1.0;
        in.slack = 1e-7;
        in.pivot_tol = 1e-7;
        in.rel_pivot_tol = 1e-9;
        in.allow_flips = allow;

        DualRatioInput ref = in;
        ref.exhaustive_reference = true;

        const auto a = dual_ratio_test(in, ws_fast);
        const auto b = dual_ratio_test(ref, ws_ref);

        // Every field a pivot decision is made from, bit for bit.
        CHECK(a.ok == b.ok);
        CHECK(a.pivot == b.pivot);
        CHECK(a.pivot_dir == b.pivot_dir);
        CHECK(a.d_enter == b.d_enter);
        CHECK(a.alpha_enter == b.alpha_enter);
        CHECK(a.theta == b.theta);
        CHECK(a.enter_wrong_sign == b.enter_wrong_sign);
        CHECK(a.row_max_alpha == b.row_max_alpha);
        CHECK(a.excluded_small == b.excluded_small);
        CHECK(a.backed_off == b.backed_off);
        CHECK(a.exhausted == b.exhausted);
        CHECK(a.flip_count == b.flip_count);
        CHECK(ws_fast.flips == ws_ref.flips);
        if (a.flip_count > 0) ++with_flips;
        if (a.sorted_candidates == 0) ++sorted_free;
        // The fast paths must never sort MORE than the reference would.
        CHECK(a.sorted_candidates <= b.sorted_candidates);
    }
    // The trial mix has to actually exercise both regimes, or the agreement
    // above is agreement about nothing.
    CHECK(with_flips > 100);
    CHECK(sorted_free > 100);
}

}  // namespace

int main() {
    test_row_max_alpha_spans_the_whole_row();
    test_row_max_alpha_is_independent_of_the_floor();
    test_no_candidates_reports_no_row();
    test_wrong_sign_entering_is_reported_with_its_row_entry();
    test_one_sided_row_sorts_nothing();
    test_boxed_row_still_sorts_nothing_when_the_first_group_blocks();
    test_harris_band_is_invariant_for_wrong_signed_candidates();
    test_fast_paths_agree_with_the_exhaustive_reference();
    return sor::test::finish("test_dual_ratio_test");
}
