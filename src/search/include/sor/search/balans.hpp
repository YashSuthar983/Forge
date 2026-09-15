// SOR — Balans: online multi-armed bandit ALNS for MILP primal search
// (Kilinc-Karzan et al., IJCAI 2025). Clean-room: papers + own code.
//
// LAYER L5. Decision-only: selects which neighborhood / meta-heuristic arm to
// try next and records tiered rewards. Never writes dual bounds; only the
// caller may accept incumbents after validating against the original model.
//
// Under milp.policy=latest this is the primary primal controller over the
// existing Neighborhood portfolio (plus optional Kernel Pump / MRENS /
// FeasJump / BTBS-LNS / CL-TLNS meta-arms). Classical keeps the fixed
// AlnsScheduler in lns.hpp.
#pragma once

#include "sor/core/result.hpp"
#include "sor/search/lns.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;

// Arms Balans can schedule. The first six map 1:1 onto Neighborhood; the rest
// are meta-heuristics whose execution lives in bab / sibling modules.
enum class BalansArm : std::uint8_t {
    Rens = 0,
    Rins,
    Mutation,
    Crossover,
    LocalBranching,
    Proximity,
    KernelPump,
    Mrens,
    FeasJump,
    BtbsLns,
    ClTlns,
};

const char* to_string(BalansArm a) noexcept;

// Map a Balans arm onto an LNS Neighborhood when applicable. Returns false for
// meta-arms that are not Neighborhood builders.
bool balans_to_neighborhood(BalansArm a, Neighborhood& out) noexcept;

// Historically true for BTBS / CL-TLNS stubs. Always false now that both arms
// ship working destroy/repair paths.
bool balans_arm_is_stub(BalansArm a) noexcept;

struct BalansOptions {
    bool enabled = true;
    // Share of wall clock the Balans-controlled layer may consume.
    double budget_frac = 0.25;
    double call_time_s = 0.6;
    std::uint64_t call_nodes = 800;
    std::uint64_t call_nodes_max_growth = 8;
    std::uint64_t min_interval = 150;
    f64 fixing_rate_init = 0.5;
    f64 fixing_rate_min = 0.2;
    f64 fixing_rate_max = 0.9;
    f64 fixing_rate_step_up = 0.10;
    f64 fixing_rate_step_down = 0.05;
    f64 ucb_alpha = 1.0;
    f64 reward_new_best = 1.0;
    f64 reward_feasible = 0.35;
    f64 reward_exhausted = 0.15;
    f64 reward_nothing = 0.0;
    std::size_t pool_size = 8;
    std::uint32_t seed = 20260913u;
    // Meta-arms. Neighborhood arms are always present.
    bool include_kernel_pump = true;
    bool include_mrens = true;
    bool include_feasjump = true;
    bool include_btbs = true;
    bool include_cl_tlns = true;
};

struct BalansArmStats {
    BalansArm kind = BalansArm::Rens;
    std::uint64_t calls = 0;
    std::uint64_t hits = 0;
    f64 reward_sum = 0.0;
    f64 fixing_rate = 0.5;
    std::uint64_t node_budget = 800;
    double seconds = 0.0;
};

struct BalansDiagnostics {
    std::uint64_t attempts = 0;
    std::uint64_t built = 0;
    std::uint64_t hits = 0;
    std::uint64_t budget_blocks = 0;
    double seconds = 0.0;
    std::vector<BalansArmStats> arms;
};

// Online UCB1 bandit over BalansArm. Deterministic given the reward sequence.
class BalansScheduler {
public:
    explicit BalansScheduler(const BalansOptions& opts);

    int select(const std::vector<bool>& usable);
    void reward(int arm, LnsOutcome outcome, double seconds);

    f64 fixing_rate(int arm) const {
        return arms_[static_cast<std::size_t>(arm)].fixing_rate;
    }
    std::uint64_t node_budget(int arm) const {
        return arms_[static_cast<std::size_t>(arm)].node_budget;
    }
    BalansArm kind(int arm) const {
        return arms_[static_cast<std::size_t>(arm)].kind;
    }
    int n_arms() const { return static_cast<int>(arms_.size()); }
    const std::vector<BalansArmStats>& arms() const noexcept { return arms_; }

    // Convenience: LnsOptions view for Neighborhood builders that still take
    // the classical options struct.
    LnsOptions as_lns_options() const;

private:
    BalansOptions opts_;
    std::vector<BalansArmStats> arms_;
    std::uint64_t total_calls_ = 0;
};

}  // namespace sor::search
