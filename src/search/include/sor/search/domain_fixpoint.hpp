// Shared domain fixpoint: one driver that runs a set of domain propagators
// (row bounds, conflict graph, learned clauses, ...) until none of them can
// change the domain any more, and says so explicitly.
//
// The propagators feed each other -- a clique forces a binary, which is a new
// bound for the rows it sits in, which can force a clause's last literal -- so
// no fixed order is a fixpoint. Each propagator here is stale until it has run
// on the current domain; a step that CHANGES the domain makes every other
// propagator stale again (and itself too, unless it reports it ran to its own
// fixpoint). The outcome is one of three states, never an implicit one:
//   Stable      every propagator has run on the final domain and changed nothing
//   Infeasible  a propagator proved the domain empty (which one is reported)
//   Pending     the step allowance ran out with work still stale, OR a propagator
//               reported it is unfinished without being able to make progress
//               on the current domain; the domain is valid (only ever narrowed
//               soundly) but not a fixpoint, and the unfinished set is returned
//               so the caller can resume it later. An unfinished propagator is
//               never mistaken for a finished one: it is re-run only when
//               another propagator changes the domain (new information), not in
//               a loop that cannot help.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sor::search {

enum class FixpointStatus { Stable, Infeasible, Pending };

struct FixpointStep {
    bool feasible = true;    // false: the domain is empty
    bool changed = false;    // the step narrowed the domain
    bool converged = true;   // it ran to its own fixpoint (nothing left for it to do)
};

// Every accepted bound update, with where it came from. Propagators subscribe by
// keeping a cursor: `vars_since(cursor)` is the set of columns other steps moved
// since that consumer last looked, which is all an incremental propagator needs
// to revisit.
struct DomainEvent {
    std::int32_t var = -1;
    bool upper = false;        // which side moved
    double old_bound = 0.0;
    double new_bound = 0.0;
    std::int32_t level = 0;    // decision level (node depth)
    std::int32_t source = -1;  // propagator id that made the update
};

class DomainEventLog {
public:
    // Records every side of every column that differs between the before and
    // after boxes. Returns the number of updates.
    std::size_t record_diff(const std::vector<double>& lo0, const std::vector<double>& hi0,
                            const std::vector<double>& lo1, const std::vector<double>& hi1,
                            int level, int source);
    std::size_t size() const { return events_.size(); }
    const std::vector<DomainEvent>& events() const { return events_; }
    // Distinct columns with an update at index >= cursor (order of first update).
    std::vector<std::int32_t> vars_since(std::size_t cursor) const;
    void clear() { events_.clear(); }

private:
    std::vector<DomainEvent> events_;
};

struct FixpointRun {
    FixpointStatus status = FixpointStatus::Stable;
    int steps = 0;              // propagator invocations
    int changing_steps = 0;     // invocations that narrowed the domain
    int infeasible_id = -1;     // propagator that proved infeasibility
    std::vector<char> stale;    // per propagator; all zero unless Pending (stale or unfinished)
};

class DomainFixpoint {
public:
    using Step = std::function<FixpointStep()>;

    // Propagators run in registration order among the stale ones.
    int add(std::string name, Step step);
    int size() const { return static_cast<int>(steps_.size()); }
    const std::string& name(int id) const { return names_[static_cast<std::size_t>(id)]; }

    // `initially_stale` (size() entries, or empty for all stale) marks which
    // propagators have not yet seen the current domain. `max_steps` bounds the
    // invocations (<= 0: only the trivial bound of 64 per propagator).
    FixpointRun run(int max_steps, const std::vector<char>& initially_stale = {}) const;

private:
    std::vector<std::string> names_;
    std::vector<Step> steps_;
};

}  // namespace sor::search
