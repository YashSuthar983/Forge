#include "sor/search/para_bab.hpp"

#include <algorithm>
#include <thread>

namespace sor::search {

int resolve_para_bab_threads(const ParaBabOptions& opts, bool latest_policy) {
    if (!latest_policy) return 1;
    if (opts.threads > 0) return opts.threads;
    const unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) return 4;
    return static_cast<int>(std::min(8u, std::max(2u, hc)));
}

bool para_bab_should_activate(const ParaBabCostModel& model,
                              const ParaBabSearchState& state) {
    if (state.threads <= 1) return false;
    // Legacy path: parallel from node 0, which is what cost enigma 7.3x.
    if (!model.adaptive) return true;

    // 1. Ramp. Either signal is enough -- a model with very slow node LPs can
    //    trip the clock long before the node count, and a model with very fast
    //    ones the other way round.
    const bool ramped = state.elapsed_s >= model.ramp_seconds ||
                        state.nodes_done >= model.ramp_nodes;
    if (!ramped) return false;

    // 2. Occupancy. Workers with no node to expand are pure overhead, so do
    //    not start phases until the frontier can fill them.
    const std::size_t needed = static_cast<std::size_t>(state.threads) *
        static_cast<std::size_t>(std::max(1, model.min_frontier_per_worker));
    if (state.open_frontier < needed) return false;

    // 3. Plunge productivity. While diving still improves the incumbent it is
    //    worth more than parallelism, because each improvement prunes tree
    //    that the workers would otherwise have to explore.
    if (state.nodes_since_incumbent < model.plunge_quiet_nodes) return false;

    return true;
}

}  // namespace sor::search
