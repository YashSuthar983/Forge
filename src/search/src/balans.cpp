#include "sor/search/balans.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sor::search {
namespace {

inline std::size_t sz(int i) { return static_cast<std::size_t>(i); }

}  // namespace

const char* to_string(BalansArm a) noexcept {
    switch (a) {
        case BalansArm::Rens: return "rens";
        case BalansArm::Rins: return "rins";
        case BalansArm::Mutation: return "mutation";
        case BalansArm::Crossover: return "crossover";
        case BalansArm::LocalBranching: return "localbranch";
        case BalansArm::Proximity: return "proximity";
        case BalansArm::KernelPump: return "kernel_pump";
        case BalansArm::Mrens: return "mrens";
        case BalansArm::FeasJump: return "feasjump";
        case BalansArm::BtbsLns: return "btbs_lns";
        case BalansArm::ClTlns: return "cl_tlns";
    }
    return "?";
}

bool balans_to_neighborhood(BalansArm a, Neighborhood& out) noexcept {
    switch (a) {
        case BalansArm::Rens: out = Neighborhood::Rens; return true;
        case BalansArm::Rins: out = Neighborhood::Rins; return true;
        case BalansArm::Mutation: out = Neighborhood::Mutation; return true;
        case BalansArm::Crossover: out = Neighborhood::Crossover; return true;
        case BalansArm::LocalBranching:
            out = Neighborhood::LocalBranching;
            return true;
        case BalansArm::Proximity: out = Neighborhood::Proximity; return true;
        default: return false;
    }
}

bool balans_arm_is_stub(BalansArm /*a*/) noexcept {
    return false;  // BTBS / CL-TLNS are real arms as of v1
}

BalansScheduler::BalansScheduler(const BalansOptions& opts) : opts_(opts) {
    const BalansArm core[] = {
        BalansArm::Rens,         BalansArm::Rins,      BalansArm::Mutation,
        BalansArm::Crossover,    BalansArm::LocalBranching,
        BalansArm::Proximity};
    for (BalansArm k : core) {
        BalansArmStats a;
        a.kind = k;
        a.fixing_rate = opts.fixing_rate_init;
        a.node_budget = opts.call_nodes;
        arms_.push_back(a);
    }
    auto add = [&](BalansArm k, bool on) {
        if (!on) return;
        BalansArmStats a;
        a.kind = k;
        a.fixing_rate = opts.fixing_rate_init;
        a.node_budget = opts.call_nodes;
        arms_.push_back(a);
    };
    add(BalansArm::KernelPump, opts.include_kernel_pump);
    add(BalansArm::Mrens, opts.include_mrens);
    add(BalansArm::FeasJump, opts.include_feasjump);
    add(BalansArm::BtbsLns, opts.include_btbs);
    add(BalansArm::ClTlns, opts.include_cl_tlns);
}

LnsOptions BalansScheduler::as_lns_options() const {
    LnsOptions o;
    o.enabled = opts_.enabled;
    o.budget_frac = opts_.budget_frac;
    o.call_time_s = opts_.call_time_s;
    o.call_nodes = opts_.call_nodes;
    o.call_nodes_max_growth = opts_.call_nodes_max_growth;
    o.min_interval = opts_.min_interval;
    o.fixing_rate_init = opts_.fixing_rate_init;
    o.fixing_rate_min = opts_.fixing_rate_min;
    o.fixing_rate_max = opts_.fixing_rate_max;
    o.fixing_rate_step_up = opts_.fixing_rate_step_up;
    o.fixing_rate_step_down = opts_.fixing_rate_step_down;
    o.ucb_alpha = opts_.ucb_alpha;
    o.reward_new_best = opts_.reward_new_best;
    o.reward_feasible = opts_.reward_feasible;
    o.reward_exhausted = opts_.reward_exhausted;
    o.reward_nothing = opts_.reward_nothing;
    o.pool_size = opts_.pool_size;
    o.seed = opts_.seed;
    return o;
}

int BalansScheduler::select(const std::vector<bool>& usable) {
    int best = -1;
    f64 best_score = -std::numeric_limits<f64>::infinity();
    const f64 log_n =
        std::log(static_cast<f64>(std::max<std::uint64_t>(total_calls_, 1)));
    for (std::size_t a = 0; a < arms_.size(); ++a) {
        if (a < usable.size() && !usable[a]) continue;
        const auto& arm = arms_[a];
        f64 score;
        if (arm.calls == 0) {
            score = std::numeric_limits<f64>::infinity();
        } else {
            const f64 mean = arm.reward_sum / static_cast<f64>(arm.calls);
            score = mean + opts_.ucb_alpha *
                               std::sqrt(log_n / static_cast<f64>(arm.calls));
        }
        if (score > best_score) {
            best_score = score;
            best = static_cast<int>(a);
        }
    }
    return best;
}

void BalansScheduler::reward(int arm, LnsOutcome outcome, double seconds) {
    if (arm < 0 || arm >= static_cast<int>(arms_.size())) return;
    auto& a = arms_[sz(arm)];
    if (outcome == LnsOutcome::NotBuilt) return;

    ++a.calls;
    ++total_calls_;
    a.seconds += seconds;

    f64 r = opts_.reward_nothing;
    switch (outcome) {
        case LnsOutcome::NewBest:
            r = opts_.reward_new_best;
            ++a.hits;
            break;
        case LnsOutcome::Feasible: r = opts_.reward_feasible; break;
        case LnsOutcome::Exhausted: r = opts_.reward_exhausted; break;
        default: break;
    }
    if (opts_.call_time_s > 0.0) {
        const f64 frac = std::min(1.0, seconds / opts_.call_time_s);
        r *= 1.0 / (1.0 + frac);
    }
    a.reward_sum += r;

    const bool success = outcome == LnsOutcome::NewBest ||
                         outcome == LnsOutcome::Feasible;
    if (success) a.fixing_rate -= opts_.fixing_rate_step_down;
    else a.fixing_rate += opts_.fixing_rate_step_up;
    a.fixing_rate = std::min(std::max(a.fixing_rate, opts_.fixing_rate_min),
                             opts_.fixing_rate_max);

    const std::uint64_t cap = opts_.call_nodes * opts_.call_nodes_max_growth;
    if (outcome == LnsOutcome::NewBest)
        a.node_budget = std::min(cap, a.node_budget * 2);
    else if (outcome == LnsOutcome::Nothing)
        a.node_budget = std::max<std::uint64_t>(opts_.call_nodes / 4,
                                                a.node_budget / 2);
}

}  // namespace sor::search
