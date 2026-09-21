// SOR - Adaptive Large Neighborhood Search: a bandit-scheduled portfolio of
// sub-MIP neighborhoods.
//
// LAYER L5 (search), sibling of bab.hpp. Implementation spec:
//   Hendel, "Adaptive large neighborhood search for mixed integer programming",
//     Mathematical Programming Computation 14, 2022 - ALNS as a bandit over LNS
//     neighborhoods, with an adaptive fixing rate.
//   Kilinc-Karzan et al., "BALANS: Multi-Armed Bandits-based Adaptive Large
//     Neighborhood Search for MIP", IJCAI 2025 - the arm/reward formulation used
//     below (tiered outcome rewards, one arm per neighborhood configuration).
//   Per-neighborhood sources: RENS (Berthold, MPC 6, 2014), RINS (Danna,
//     Rothberg & Le Pape, Oper. Res. 53(5), 2005), Local Branching (Fischetti &
//     Lodi, Math. Prog. 98, 2003), Proximity Search (Fischetti & Monaci, 2014),
//     Mutation and Crossover (Rothberg, INFORMS J. Comp. 19(4), 2007).
//
// WHY THIS EXISTS. Measured on this host, MILP wall time is dominated by
// heuristics, not by the tree: node relaxations are 13% of the solve on pk1,
// 14% on p0201, 19% on misc03. Warm-starting is healthy (2.8-18 simplex
// iterations per node), so those instances are not slow because their LPs are
// slow -- they are slow because every heuristic runs on a FIXED schedule
// whether or not it has ever paid off, and the search gets whatever is left.
// Hand-tuning each schedule is the wrong shape of fix; the right one is to let
// the solver learn which neighborhood is worth its time on THIS instance and
// to bound what the whole heuristic layer may spend.
//
// This header owns the two things that are pure decision-making -- which
// neighborhood to try next, and what box it implies -- and deliberately does
// NOT own the sub-MIP solve. The solve needs solve_milp(), which lives in
// bab.cpp; keeping it there avoids a circular dependency and keeps this file
// testable without a solver.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

// One neighborhood family. Each is a way of deciding which integer columns to
// hold still, so that what remains is a small enough MILP to be solved outright
// in a fraction of a second.
enum class Neighborhood {
    Rens,            // fix integral columns of the relaxation, bracket the rest
    Rins,            // fix where incumbent and relaxation already agree
    Mutation,        // fix a random share of columns at their incumbent values
    Crossover,       // fix where several pooled solutions agree
    LocalBranching,  // Hamming ball around the incumbent (adds one row)
    Proximity,       // objective replaced by distance to the incumbent
};

const char* to_string(Neighborhood n) noexcept;

// What a neighborhood asks the caller to solve. `col_lo`/`col_hi` is the
// restricted box; the optional extras describe the two neighborhoods that need
// more than a box.
struct NeighborhoodProblem {
    std::vector<f64> col_lo, col_hi;
    // Local branching: sum over binaries of |x_j - incumbent_j|, expressed
    // linearly, must be <= distance_max. Empty when unused.
    std::vector<Index> distance_cols;
    std::vector<f64> distance_vals;
    f64 distance_rhs = 0.0;
    bool has_distance_row = false;
    // Proximity search: replace the objective with this one (minimised), and
    // add the original objective as a cutoff row. Empty when unused.
    std::vector<f64> objective;
    bool replace_objective = false;
    f64 objective_cutoff = core::kPosInf;   // in the ORIGINAL sense
    bool has_objective_cutoff = false;
    // How many integer columns ended up fixed, and how many stayed free. The
    // scheduler needs both to tell "this neighborhood was too big to solve"
    // apart from "this neighborhood was empty".
    std::uint64_t fixed = 0;
    std::uint64_t free_integer = 0;
};

// How a sub-MIP call turned out, in the tiers BALANS scores against.
enum class LnsOutcome {
    NewBest,      // improved the incumbent
    Feasible,     // found a point, but not an improvement
    Exhausted,    // solved the neighborhood without improving (informative)
    Nothing,      // no point, budget spent
    NotBuilt,     // the neighborhood was degenerate; costs no time and no reward
};

struct LnsOptions {
    bool enabled = true;
    // The whole point: a hard ceiling on the share of wall clock the LNS layer
    // may consume, whatever the bandit would like to do.
    double budget_frac = 0.20;
    // Per-call budgets, deliberately SMALL to start. A bandit that only gets
    // two samples cannot learn anything: measured on pk1, a 2 s call budget
    // against a 20 s solve meant the LNS layer spent its entire share on two
    // calls that found nothing and never revisited the decision. Small calls
    // buy samples, and the per-arm node budget below lets an arm that pays
    // earn a bigger one.
    double call_time_s = 0.6;
    std::uint64_t call_nodes = 800;
    std::uint64_t call_nodes_max_growth = 8;   // arm node budget ceiling, x this
    std::uint64_t min_interval = 200;  // nodes between attempts
    // Adaptive fixing rate (Hendel §4.3): a neighborhood that fails is made
    // SMALLER next time by fixing more, and one that succeeds is made larger.
    // Each arm carries its own rate, so a neighborhood that suits this instance
    // widens while the others shrink out of the way.
    f64 fixing_rate_init = 0.5;
    f64 fixing_rate_min = 0.2;
    f64 fixing_rate_max = 0.9;
    f64 fixing_rate_step_up = 0.10;    // applied after a failure
    f64 fixing_rate_step_down = 0.05;  // applied after a success
    // UCB1 exploration weight. Rewards below are already in [0,1], so this is
    // the textbook constant rather than a tuned one.
    f64 ucb_alpha = 1.0;
    // Tiered rewards (BALANS, IJCAI 2025, linear setting normalised to [0,1]).
    f64 reward_new_best = 1.0;
    f64 reward_feasible = 0.35;
    f64 reward_exhausted = 0.15;
    f64 reward_nothing = 0.0;
    // Solutions kept for Crossover.
    std::size_t pool_size = 8;
    std::uint32_t seed = 20260907u;
};

struct LnsArmStats {
    Neighborhood kind = Neighborhood::Rens;
    std::uint64_t calls = 0;
    std::uint64_t hits = 0;
    f64 reward_sum = 0.0;
    f64 fixing_rate = 0.5;
    // Node budget for THIS arm's sub-MIPs. Grows when the arm delivers and
    // shrinks when it does not, so a neighbourhood that suits the instance is
    // allowed to search harder while the others stay cheap to sample.
    std::uint64_t node_budget = 800;
    double seconds = 0.0;
};

struct LnsDiagnostics {
    std::uint64_t attempts = 0;
    std::uint64_t built = 0;
    std::uint64_t hits = 0;
    std::uint64_t sub_nodes = 0;
    double seconds = 0.0;
    std::uint64_t budget_blocks = 0;   // times the ceiling refused a call
    std::vector<LnsArmStats> arms;
};

// The bandit. Deterministic given the same sequence of rewards, which is what
// keeps a benchmark run reproducible.
class AlnsScheduler {
public:
    explicit AlnsScheduler(const LnsOptions& opts);

    // UCB1 over the arms, restricted to those `usable` allows (Crossover needs
    // a populated pool; everything but RENS needs an incumbent).
    int select(const std::vector<bool>& usable);

    void reward(int arm, LnsOutcome outcome, double seconds);

    f64 fixing_rate(int arm) const { return arms_[static_cast<std::size_t>(arm)].fixing_rate; }
    std::uint64_t node_budget(int arm) const {
        return arms_[static_cast<std::size_t>(arm)].node_budget;
    }
    Neighborhood kind(int arm) const { return arms_[static_cast<std::size_t>(arm)].kind; }
    int n_arms() const { return static_cast<int>(arms_.size()); }
    const std::vector<LnsArmStats>& arms() const noexcept { return arms_; }

private:
    LnsOptions opts_;
    std::vector<LnsArmStats> arms_;
    std::uint64_t total_calls_ = 0;
};

// A bounded store of the best distinct solutions found, for Crossover.
class SolutionPool {
public:
    explicit SolutionPool(std::size_t cap) : cap_(cap) {}
    // Keeps the pool sorted best-first and drops duplicates.
    void add(const std::vector<f64>& x, f64 objective, bool maximize);
    std::size_t size() const noexcept { return entries_.size(); }
    const std::vector<f64>& at(std::size_t i) const { return entries_[i].x; }

private:
    struct Entry { std::vector<f64> x; f64 obj; };
    std::size_t cap_;
    std::vector<Entry> entries_;
};

// Builds the restricted sub-problem for `kind`. Returns false when the
// neighborhood is degenerate -- nothing fixed, or nothing left free -- which
// the caller must treat as NotBuilt rather than as a failure, since it cost no
// time and says nothing about the neighborhood's value.
//
// `node_lo`/`node_hi` is the box to restrict (normally the root box, so the
// result is globally valid). `x_relax` is the current relaxation point and
// `x_inc` the incumbent; either may be empty when the neighborhood does not
// need it.
bool build_neighborhood(Neighborhood kind,
                        const model::LpProblem& mip,
                        const std::vector<f64>& node_lo,
                        const std::vector<f64>& node_hi,
                        const std::vector<f64>& x_relax,
                        const std::vector<f64>& x_inc,
                        const SolutionPool& pool,
                        f64 fixing_rate,
                        f64 int_tol,
                        std::uint32_t& rng_state,
                        NeighborhoodProblem& out);

// Materialises `np` as a solvable model: applies the box, and appends the
// local-branching distance row and/or the proximity objective and cutoff row
// when present. The result is a restriction of `mip`, so any point it admits is
// a point of `mip` -- the caller still re-validates before accepting.
model::LpProblem apply_neighborhood(const model::LpProblem& mip,
                                    const NeighborhoodProblem& np);

}  // namespace sor::search
