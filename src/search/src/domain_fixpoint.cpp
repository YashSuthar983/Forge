#include "sor/search/domain_fixpoint.hpp"

#include <algorithm>
#include <utility>

namespace sor::search {

int DomainFixpoint::add(std::string name, Step step) {
    names_.push_back(std::move(name));
    steps_.push_back(std::move(step));
    return static_cast<int>(steps_.size()) - 1;
}

FixpointRun DomainFixpoint::run(int max_steps, const std::vector<char>& initially_stale) const {
    const std::size_t k = steps_.size();
    FixpointRun out;
    std::vector<char> stale(k, 1);
    std::vector<char> unfinished(k, 0);   // ran, not converged, made no change
    if (initially_stale.size() == k) stale = initially_stale;
    if (max_steps <= 0) max_steps = static_cast<int>(64 * (k > 0 ? k : 1));

    // Round-robin over the stale ones, so a cheap propagator is not starved by
    // an expensive one that keeps invalidating it.
    std::size_t cursor = 0;
    while (out.steps < max_steps) {
        std::size_t pick = k;
        for (std::size_t t = 0; t < k; ++t) {
            const std::size_t i = (cursor + t) % k;
            if (stale[i]) { pick = i; break; }
        }
        if (pick == k) {                       // nothing stale left to run
            bool any_unfinished = false;
            for (const char u : unfinished) any_unfinished = any_unfinished || u;
            out.status = any_unfinished ? FixpointStatus::Pending : FixpointStatus::Stable;
            out.stale = std::move(unfinished);   // resume set (all zero when Stable)
            return out;
        }
        stale[pick] = 0;
        unfinished[pick] = 0;
        cursor = pick + 1;
        const FixpointStep r = steps_[pick]();
        ++out.steps;
        if (!r.feasible) {
            out.status = FixpointStatus::Infeasible;
            out.infeasible_id = static_cast<int>(pick);
            out.stale.assign(k, 0);
            return out;
        }
        if (r.changed) {
            ++out.changing_steps;
            // New information: every other propagator (including one that was
            // unfinished) gets another look.
            for (std::size_t i = 0; i < k; ++i)
                if (i != pick) { stale[i] = 1; unfinished[i] = 0; }
            if (!r.converged) stale[pick] = 1;
        } else if (!r.converged) {
            // Unfinished but no progress on this domain: re-running it verbatim
            // cannot help. It stays unfinished (Pending) until something else
            // changes the domain.
            unfinished[pick] = 1;
        }
    }
    bool any = false;
    for (std::size_t i = 0; i < k; ++i) {
        stale[i] = stale[i] || unfinished[i];
        any = any || stale[i];
    }
    out.status = any ? FixpointStatus::Pending : FixpointStatus::Stable;
    out.stale = std::move(stale);
    return out;
}

std::size_t DomainEventLog::record_diff(const std::vector<double>& lo0,
                                        const std::vector<double>& hi0,
                                        const std::vector<double>& lo1,
                                        const std::vector<double>& hi1, int level,
                                        int source) {
    const std::size_t n = std::min({lo0.size(), hi0.size(), lo1.size(), hi1.size()});
    std::size_t made = 0;
    for (std::size_t j = 0; j < n; ++j) {
        if (lo1[j] != lo0[j]) {
            events_.push_back({static_cast<std::int32_t>(j), false, lo0[j], lo1[j], level, source});
            ++made;
        }
        if (hi1[j] != hi0[j]) {
            events_.push_back({static_cast<std::int32_t>(j), true, hi0[j], hi1[j], level, source});
            ++made;
        }
    }
    return made;
}

std::vector<std::int32_t> DomainEventLog::vars_since(std::size_t cursor) const {
    std::vector<std::int32_t> out;
    std::vector<char> seen;
    for (std::size_t e = cursor; e < events_.size(); ++e) {
        const auto v = static_cast<std::size_t>(events_[e].var);
        if (v >= seen.size()) seen.resize(v + 1, 0);
        if (seen[v]) continue;
        seen[v] = 1;
        out.push_back(events_[e].var);
    }
    return out;
}

}  // namespace sor::search
