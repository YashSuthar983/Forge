// SOR - Para-B&B deterministic parallel branch-and-bound (arXiv:2604.09556).
//
// Clean-room from the paper's architecture, not a port of HiGHS:
//   - State replication: each worker expands its own frontier slice
//   - Orchestrated phases + barrier sync (NOT work-stealing)
//   - Deterministic merge: children / incumbents combined in fixed order
//
// C3: within a phase workers are independent; after barrier the global open
// set and incumbent are identical for a given seed regardless of scheduling.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstddef>
#include <cstdint>

namespace sor::search {

using core::f64;

struct ParaBabOptions {
    // 1 = serial bab path. >1 enables Para-B&B phases.
    int threads = 1;
    // Nodes each worker may expand before the next barrier. 0 → 1.
    int nodes_per_worker_per_phase = 1;
    // Soft cap on phases (0 = unlimited until tree/time/node limits).
    std::uint64_t max_phases = 0;

    bool enabled() const noexcept { return threads > 1; }
};

// ---------------------------------------------------------------------------
// Cost model for CPU tree parallelism
// ---------------------------------------------------------------------------
//
// Parallelism and plunging compete for the SAME resource: the node frontier.
// Plunging dives depth-first toward an incumbent and deliberately keeps the
// frontier thin; a Para-B&B phase needs a fat best-bound frontier to fill k
// workers. They cannot both be on.
//
// That conflict used to be resolved statically -- `threads > 1` disabled
// plunging for the entire solve. Measured cost of that rule (2026-09-19,
// 4 workers on distinct physical cores):
//
//     stein27   43.13 s -> 4.40 s   (9.8x FASTER, tree-bound)
//     enigma     4.75 s -> 34.64 s  (7.3x SLOWER, incumbent-bound)
//
// enigma is incumbent-bound: it finishes in under 5 s precisely BECAUSE it
// plunges, and it never runs long enough for parallelism to amortise the
// ramp-up. No static rule can separate those two cases, because which one an
// instance is only becomes observable after the search has started.
//
// So the model defers the decision and reads it off the search itself: stay
// serial-with-plunging until the search demonstrates it is TREE-bound rather
// than INCUMBENT-bound, then switch once and never switch back. A short
// instance simply finishes before the switch and pays nothing.
struct ParaBabCostModel {
    // false → legacy behaviour: parallel from node 0, plunging off throughout.
    bool adaptive = true;

    // --- ramp: do not pay thread + cache cost before the search has shown it
    // will last long enough to repay it. Either signal is enough.
    f64 ramp_seconds = 0.75;
    std::uint64_t ramp_nodes = 150;

    // --- occupancy: a phase with idle workers is pure overhead. Para-B&B
    // reports 34.7% mean idle even with its load balancer, so require a
    // frontier that can actually fill the workers before paying for them.
    int min_frontier_per_worker = 2;

    // --- plunge productivity: while diving is still improving the incumbent
    // it is out-earning parallelism, because a better incumbent prunes the
    // tree that parallelism would otherwise have to explore. Only once diving
    // has gone quiet for this many nodes is the frontier worth more than the
    // dive. Counted from the start of search when no incumbent exists yet, so
    // an instance that never finds one still eventually switches.
    std::uint64_t plunge_quiet_nodes = 250;

    // On activation, should plunging be switched OFF for the rest of the solve?
    //
    // The original Para-B&B wiring did exactly that, on the reasoning that a
    // persistent plunge stack starves phases. But the phase gate ALREADY
    // requires `plunge_stack.empty()`, so the two can simply interleave: dive
    // when there is a dive to do, run a parallel phase otherwise.
    //
    // Killing plunging outright cost blend2 9016 -> 14180 nodes (+57%) and
    // 29.4s -> 96.1s, because plunging keeps the TREE small on that model even
    // during stretches when it is not producing incumbent improvements -- which
    // is precisely what plunge_quiet_nodes cannot see.
    bool disable_plunge_on_activate = false;
};

// Everything the model reads, sampled from the live search.
struct ParaBabSearchState {
    f64 elapsed_s = 0.0;
    std::uint64_t nodes_done = 0;
    std::size_t open_frontier = 0;
    // Nodes since the incumbent last improved (or since search start if there
    // is no incumbent yet).
    std::uint64_t nodes_since_incumbent = 0;
    int threads = 1;
};

// True once parallel phases are expected to pay for themselves. Monotone in
// intent: callers latch the first `true` and stop asking.
bool para_bab_should_activate(const ParaBabCostModel& model,
                              const ParaBabSearchState& state);

struct ParaBabDiagnostics {
    std::uint64_t phases = 0;
    std::uint64_t parallel_expansions = 0;
    std::uint64_t syncs = 0;
    int threads_used = 0;
    // Cost-model telemetry: when the switch happened, so a run can be read
    // back afterwards instead of guessed at.
    bool activated = false;
    f64 activated_at_s = 0.0;
    std::uint64_t activated_at_node = 0;
    std::uint64_t serial_nodes = 0;
};

// Resolve worker count: Classical → 1; threads>0 → as set; threads==0 + Latest
// → min(8, max(2, hardware_concurrency)).
int resolve_para_bab_threads(const ParaBabOptions& opts, bool latest_policy);

}  // namespace sor::search
