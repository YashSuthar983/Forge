// Balans bandit-ALNS (sor_search/src/balans.cpp): UCB schedule over Neighborhood
// + meta-arms (Kernel Pump / MRENS / FeasJump / BTBS / CL-TLNS).
#include "sor/search/balans.hpp"

#include "test_helpers.hpp"

#include <string>
#include <vector>

using sor::search::BalansArm;
using sor::search::BalansOptions;
using sor::search::BalansScheduler;
using sor::search::LnsOutcome;
using sor::search::Neighborhood;
using sor::search::balans_arm_is_stub;
using sor::search::balans_to_neighborhood;
using sor::search::to_string;

namespace {

void test_explores_every_arm_first() {
    BalansOptions o;
    o.include_btbs = false;
    o.include_cl_tlns = false;
    BalansScheduler s(o);
    const std::vector<bool> all(static_cast<std::size_t>(s.n_arms()), true);
    std::vector<int> seen(static_cast<std::size_t>(s.n_arms()), 0);
    for (int i = 0; i < s.n_arms(); ++i) {
        const int a = s.select(all);
        CHECK(a >= 0);
        ++seen[static_cast<std::size_t>(a)];
        s.reward(a, LnsOutcome::Nothing, 0.1);
    }
    for (const int c : seen) CHECK(c == 1);
}

void test_favours_paying_arm() {
    BalansOptions o;
    o.include_kernel_pump = false;
    o.include_mrens = false;
    o.include_feasjump = false;
    o.include_btbs = false;
    o.include_cl_tlns = false;
    BalansScheduler s(o);
    const std::vector<bool> all(static_cast<std::size_t>(s.n_arms()), true);
    const int lucky = 1;
    int lucky_picks = 0;
    for (int i = 0; i < 200; ++i) {
        const int a = s.select(all);
        if (a == lucky) ++lucky_picks;
        s.reward(a, a == lucky ? LnsOutcome::NewBest : LnsOutcome::Nothing, 0.05);
    }
    CHECK(lucky_picks > 100);
}

void test_not_built_unscored() {
    BalansOptions o;
    BalansScheduler s(o);
    s.reward(0, LnsOutcome::NotBuilt, 0.5);
    CHECK(s.arms()[0].calls == 0);
}

void test_arms_documented() {
    CHECK(!balans_arm_is_stub(BalansArm::BtbsLns));
    CHECK(!balans_arm_is_stub(BalansArm::ClTlns));
    CHECK(!balans_arm_is_stub(BalansArm::Rens));
    Neighborhood n;
    CHECK(balans_to_neighborhood(BalansArm::Rins, n));
    CHECK(n == Neighborhood::Rins);
    CHECK(!balans_to_neighborhood(BalansArm::KernelPump, n));
    CHECK(!balans_to_neighborhood(BalansArm::BtbsLns, n));
    CHECK(std::string(to_string(BalansArm::KernelPump)) == "kernel_pump");
    CHECK(std::string(to_string(BalansArm::BtbsLns)) == "btbs_lns");
    CHECK(std::string(to_string(BalansArm::ClTlns)) == "cl_tlns");
}

void test_includes_btbs_cl_by_default() {
    BalansOptions o;
    BalansScheduler s(o);
    bool saw_btbs = false, saw_cl = false;
    for (const auto& a : s.arms()) {
        if (a.kind == BalansArm::BtbsLns) saw_btbs = true;
        if (a.kind == BalansArm::ClTlns) saw_cl = true;
    }
    CHECK(saw_btbs);
    CHECK(saw_cl);
}

void test_fixing_rate_adapts() {
    BalansOptions o;
    o.include_kernel_pump = false;
    o.include_mrens = false;
    o.include_feasjump = false;
    o.include_btbs = false;
    o.include_cl_tlns = false;
    BalansScheduler s(o);
    const auto start = s.fixing_rate(0);
    s.reward(0, LnsOutcome::Nothing, 0.1);
    CHECK(s.fixing_rate(0) > start);
    for (int i = 0; i < 10; ++i) s.reward(0, LnsOutcome::NewBest, 0.1);
    CHECK(s.fixing_rate(0) < start);
}

}  // namespace

int main() {
    test_explores_every_arm_first();
    test_favours_paying_arm();
    test_not_built_unscored();
    test_arms_documented();
    test_includes_btbs_cl_by_default();
    test_fixing_rate_adapts();
    return sor::test::finish("test_balans");
}
