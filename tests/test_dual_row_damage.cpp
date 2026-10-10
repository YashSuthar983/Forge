// Full-row dual-step damage oracle: eligible-candidate |alpha| envelopes do
// not bound updates on ineligible nonbasics (Agentsol / work-order 2026-10-09).
#include "sor/core/env_switches.hpp"
#include "sor/engines/dual_ratio_test.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::engines::DualEnterShieldKind;
using sor::engines::NonbasicStatus;
using sor::engines::classify_dual_enter_shield;
using sor::engines::dual_row_step_worsens_feasibility;
using sor::engines::dual_sign_violation;

namespace {

constexpr f64 kInf = std::numeric_limits<f64>::infinity();
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

void test_sign_violation_senses() {
    CHECK_NEAR(dual_sign_violation(NonbasicStatus::AtLower, -1e-8, 1e-7), 0.0, 0.0);
    CHECK(dual_sign_violation(NonbasicStatus::AtLower, -0.01, 1e-7) > 0.0);
    CHECK(dual_sign_violation(NonbasicStatus::AtUpper, 0.01, 1e-7) > 0.0);
    CHECK_NEAR(dual_sign_violation(NonbasicStatus::AtUpper, 1e-8, 1e-7), 0.0, 0.0);
    CHECK(dual_sign_violation(NonbasicStatus::AtZeroFree, -0.02, 1e-7) > 0.0);
}

// Agentsol / review counterexample (unscaled, all AtLower, leave toward lower):
// eligible q: alpha=-1, d=-5e-8; excluded h: alpha=+1e6, d=1e-8.
// Candidate-only envelope |d|*1/1 = 5e-8 <= 1e-7 ("harmless"); signed theta =
// d/alpha = 5e-8 drives h to about -0.05.
void test_ineligible_column_damage_underbound() {
    const Index q = 0;
    const Index h = 1;
    const Index leave_var = 2;
    std::vector<Index> prow_idx = {q, h};
    std::vector<f64> prow(3, 0.0);
    prow[sz(q)] = -1.0;
    prow[sz(h)] = 1.0e6;
    std::vector<f64> redcost = {-5.0e-8, 1.0e-8, 0.0};
    std::vector<NonbasicStatus> status = {
        NonbasicStatus::AtLower, NonbasicStatus::AtLower, NonbasicStatus::Basic};
    std::vector<Index> flips;
    std::vector<f64> lo = {0.0, 0.0, 0.0};
    std::vector<f64> hi = {kInf, kInf, kInf};
    std::vector<f64> dtol = {1e-7, 1e-7, 1e-7};

    const f64 theta = redcost[sz(q)] / prow[sz(q)];  // (+5e-8)
    CHECK_NEAR(theta, 5.0e-8, 1e-20);

    const f64 candidate_only =
        std::fabs(redcost[sz(q)]) * 1.0 / std::fabs(prow[sz(q)]);
    CHECK(candidate_only <= 1e-7);  // old estimate: "harmless"

    const f64 d_h_new = redcost[sz(h)] - theta * prow[sz(h)];
    CHECK(d_h_new < -1e-2);

    CHECK(dual_row_step_worsens_feasibility(
        prow_idx, prow, redcost, status, flips, lo, hi, dtol, q, leave_var,
        NonbasicStatus::AtLower, theta));
}

void test_mirrored_upper_bound() {
    const Index q = 0;
    const Index h = 1;
    const Index leave_var = 2;
    std::vector<Index> prow_idx = {q, h};
    std::vector<f64> prow(3, 0.0);
    prow[sz(q)] = 1.0;
    prow[sz(h)] = -1.0e6;
    std::vector<f64> redcost = {5.0e-8, -1.0e-8, 0.0};
    std::vector<NonbasicStatus> status = {
        NonbasicStatus::AtUpper, NonbasicStatus::AtUpper, NonbasicStatus::Basic};
    std::vector<Index> flips;
    std::vector<f64> lo = {-kInf, -kInf, 0.0};
    std::vector<f64> hi = {0.0, 0.0, kInf};
    std::vector<f64> dtol = {1e-7, 1e-7, 1e-7};
    const f64 theta = redcost[sz(q)] / prow[sz(q)];
    CHECK(dual_row_step_worsens_feasibility(
        prow_idx, prow, redcost, status, flips, lo, hi, dtol, q, leave_var,
        NonbasicStatus::AtUpper, theta));
}

void test_free_column_harm() {
    const Index q = 0;
    const Index h = 1;
    const Index leave_var = 2;
    std::vector<Index> prow_idx = {q, h};
    std::vector<f64> prow(3, 0.0);
    prow[sz(q)] = -1.0;
    prow[sz(h)] = 1.0e5;
    std::vector<f64> redcost = {-1.0e-8, 0.0, 0.0};
    std::vector<NonbasicStatus> status = {
        NonbasicStatus::AtLower, NonbasicStatus::AtZeroFree,
        NonbasicStatus::Basic};
    std::vector<Index> flips;
    std::vector<f64> lo = {0.0, -kInf, 0.0};
    std::vector<f64> hi = {kInf, kInf, kInf};
    std::vector<f64> dtol = {1e-7, 1e-7, 1e-7};
    const f64 theta = redcost[sz(q)] / prow[sz(q)];
    CHECK(dual_row_step_worsens_feasibility(
        prow_idx, prow, redcost, status, flips, lo, hi, dtol, q, leave_var,
        NonbasicStatus::AtLower, theta));
}

void test_planned_flip_changes_sense() {
    // Column h is AtLower before flips; BFRT will flip it to AtUpper. A step
    // that looks fine for AtLower can violate AtUpper after the flip.
    const Index q = 0;
    const Index h = 1;
    const Index leave_var = 2;
    std::vector<Index> prow_idx = {q, h};
    std::vector<f64> prow(3, 0.0);
    prow[sz(q)] = -1.0;
    prow[sz(h)] = 10.0;
    std::vector<f64> redcost = {-1.0e-8, 1.0e-8, 0.0};
    std::vector<NonbasicStatus> status = {
        NonbasicStatus::AtLower, NonbasicStatus::AtLower, NonbasicStatus::Basic};
    std::vector<Index> flips = {h};
    std::vector<f64> lo = {0.0, 0.0, 0.0};
    std::vector<f64> hi = {1.0, 1.0, kInf};
    std::vector<f64> dtol = {1e-7, 1e-7, 1e-7};
    const f64 theta = redcost[sz(q)] / prow[sz(q)];  // +1e-8
    // After flip to AtUpper: d_new = 1e-8 - (1e-8)*10 = -9e-8 still inside
    // AtUpper tol. Scale alpha so it violates.
    prow[sz(h)] = 1.0e7;
    const f64 d_new = redcost[sz(h)] - theta * prow[sz(h)];
    CHECK(d_new < -0.05);
    // AtUpper needs d <= +tol; large negative is fine for AtUpper — use a
    // positive push instead.
    redcost[sz(h)] = -1.0e-8;
    prow[sz(h)] = -1.0e7;  // d_new = -1e-8 - (1e-8)*(-1e7) = 0.1 > tol on AtUpper
    CHECK(dual_row_step_worsens_feasibility(
        prow_idx, prow, redcost, status, flips, lo, hi, dtol, q, leave_var,
        NonbasicStatus::AtLower, theta));
}

void test_harmless_when_row_small() {
    const Index q = 0;
    const Index h = 1;
    const Index leave_var = 2;
    std::vector<Index> prow_idx = {q, h};
    std::vector<f64> prow(3, 0.0);
    prow[sz(q)] = -1.0;
    prow[sz(h)] = 0.1;
    std::vector<f64> redcost = {-1.0e-9, 0.0, 0.0};
    std::vector<NonbasicStatus> status = {
        NonbasicStatus::AtLower, NonbasicStatus::AtLower, NonbasicStatus::Basic};
    std::vector<Index> flips;
    std::vector<f64> lo = {0.0, 0.0, 0.0};
    std::vector<f64> hi = {kInf, kInf, kInf};
    std::vector<f64> dtol = {1e-7, 1e-7, 1e-7};
    const f64 theta = redcost[sz(q)] / prow[sz(q)];
    CHECK(!dual_row_step_worsens_feasibility(
        prow_idx, prow, redcost, status, flips, lo, hi, dtol, q, leave_var,
        NonbasicStatus::AtLower, theta));
}

void test_leaving_column_residual() {
    const Index q = 0;
    const Index leave_var = 1;
    std::vector<Index> prow_idx = {q};
    std::vector<f64> prow(2, 0.0);
    prow[sz(q)] = -1e-12;  // tiny enter alpha => huge |theta|
    std::vector<f64> redcost = {-1e-7, 0.0};
    std::vector<NonbasicStatus> status = {
        NonbasicStatus::AtLower, NonbasicStatus::Basic};
    std::vector<Index> flips;
    std::vector<f64> lo = {0.0, 0.0};
    std::vector<f64> hi = {kInf, kInf};
    std::vector<f64> dtol = {1e-7, 1e-7};
    const f64 theta = redcost[sz(q)] / prow[sz(q)];  // huge
    CHECK(dual_row_step_worsens_feasibility(
        prow_idx, prow, redcost, status, flips, lo, hi, dtol, q, leave_var,
        NonbasicStatus::AtLower, theta));
}

void test_classify_shield_needs_shift_on_ineligible_damage() {
    const Index q = 0;
    const Index h = 1;
    const Index leave_var = 2;
    std::vector<Index> prow_idx = {q, h};
    std::vector<f64> prow(3, 0.0);
    prow[sz(q)] = -1.0;
    prow[sz(h)] = 1.0e6;
    std::vector<f64> redcost = {-5.0e-8, 1.0e-8, 0.0};
    std::vector<NonbasicStatus> status = {
        NonbasicStatus::AtLower, NonbasicStatus::AtLower, NonbasicStatus::Basic};
    std::vector<Index> flips;
    std::vector<f64> lo = {0.0, 0.0, 0.0};
    std::vector<f64> hi = {kInf, kInf, kInf};
    std::vector<f64> dtol = {1e-7, 1e-7, 1e-7};
    const auto dec = classify_dual_enter_shield(
        true, redcost[sz(q)], prow[sz(q)], dtol[sz(q)], /*allow_cost_shifts=*/true,
        prow_idx, prow, redcost, status, flips, lo, hi, dtol, q, leave_var,
        NonbasicStatus::AtLower);
    CHECK(dec.harmful);
    CHECK(dec.kind == DualEnterShieldKind::NeedsCostShift);
}

class ScopedEnvironment {
public:
    ScopedEnvironment(const char* name, const char* value) : name_(name) {
        if (const char* old = std::getenv(name)) {
            had_old_ = true;
            old_ = old;
        }
        if (value) ::setenv(name, value, 1);
        else ::unsetenv(name);
        sor::core::reload_env_switches();
    }
    ~ScopedEnvironment() {
        if (had_old_) ::setenv(name_.c_str(), old_.c_str(), 1);
        else ::unsetenv(name_.c_str());
        sor::core::reload_env_switches();
    }
    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;
private:
    std::string name_;
    std::string old_;
    bool had_old_ = false;
};

void test_env_flag_disable_spellings() {
    {
        ScopedEnvironment on("SOR_DUAL_BFRT_DAMAGE_GUARD", "1");
        CHECK(sor::core::env_switches().dual_bfrt_damage_guard);
    }
    {
        ScopedEnvironment off("SOR_DUAL_BFRT_DAMAGE_GUARD", "0");
        CHECK(!sor::core::env_switches().dual_bfrt_damage_guard);
    }
    {
        ScopedEnvironment off("SOR_DUAL_BFRT_DAMAGE_GUARD", "false");
        CHECK(!sor::core::env_switches().dual_bfrt_damage_guard);
    }
    {
        ScopedEnvironment off("SOR_DUAL_BFRT_PREFER_SAFE", "off");
        CHECK(!sor::core::env_switches().dual_bfrt_prefer_safe);
    }
}

void test_classify_shield_zero_step_when_shifts_disabled() {
    // Same ineligible-damage geometry as above, but shifts disabled and
    // |d_enter| inside dtol → Harris zero-step approximation, never a
    // silent harmful backward authorization.
    const Index q = 0;
    const Index h = 1;
    const Index leave_var = 2;
    std::vector<Index> prow_idx = {q, h};
    std::vector<f64> prow(3, 0.0);
    prow[sz(q)] = -1.0;
    prow[sz(h)] = 1.0e6;
    std::vector<f64> redcost = {-5.0e-8, 1.0e-8, 0.0};
    std::vector<NonbasicStatus> status = {
        NonbasicStatus::AtLower, NonbasicStatus::AtLower, NonbasicStatus::Basic};
    std::vector<Index> flips;
    std::vector<f64> lo = {0.0, 0.0, 0.0};
    std::vector<f64> hi = {kInf, kInf, kInf};
    std::vector<f64> dtol = {1e-7, 1e-7, 1e-7};
    const auto dec = classify_dual_enter_shield(
        true, redcost[sz(q)], prow[sz(q)], dtol[sz(q)], /*allow_cost_shifts=*/false,
        prow_idx, prow, redcost, status, flips, lo, hi, dtol, q, leave_var,
        NonbasicStatus::AtLower);
    CHECK(dec.harmful);
    CHECK(dec.kind == DualEnterShieldKind::ZeroStepApprox);
}

}  // namespace

int main() {
    test_sign_violation_senses();
    test_ineligible_column_damage_underbound();
    test_mirrored_upper_bound();
    test_free_column_harm();
    test_planned_flip_changes_sense();
    test_harmless_when_row_small();
    test_leaving_column_residual();
    test_classify_shield_needs_shift_on_ineligible_damage();
    test_classify_shield_zero_step_when_shifts_disabled();
    test_env_flag_disable_spellings();
    std::printf("PASS test_dual_row_damage\n");
    return 0;
}
