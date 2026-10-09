#pragma once

#include "sor/search/domain_fixpoint.hpp"
#include "sor/search/prop_trail.hpp"
#include "sor/search/domain_checkpoint.hpp"
#include <cmath>
#include <stdexcept>
#include <map>
#include <utility>

namespace sor::search {
struct DomainReason {
    ReasonKind kind = ReasonKind::Unknown;
    Index id = -1;
    int depth = 0;
    std::shared_ptr<const std::vector<ConflictLiteral>> clause;
    std::shared_ptr<const std::vector<BoundRef>> antecedents;
    DomainReason() = default;
    DomainReason(ReasonKind k, Index i, int d,
                 std::shared_ptr<const std::vector<ConflictLiteral>> c = {},
                 std::shared_ptr<const std::vector<BoundRef>> a = {})
        : kind(k), id(i), depth(d), clause(std::move(c)), antecedents(std::move(a)) {}
};

// Publish both journals atomically before committing the bound. An allocation
// failure cannot leave a proof/event describing a change that never happened.
inline void record_domain_change(Index var, BoundDir dir, f64 old, f64 value,
                                 PropTrail* trail, const DomainReason& reason,
                                 DomainEventLog* events, int source) {
    const auto mark = trail ? trail->size() : 0;
    try {
        if (trail) trail->push(var, dir, value, old, reason.kind, reason.id,
                              reason.depth, reason.clause, reason.antecedents);
        if (events) events->record(var, dir == BoundDir::Upper, old, value,
                                   reason.depth, source, reason.kind, reason.id);
    } catch (...) {
        if (trail) trail->truncate(mark);
        throw;
    }
}

// The bound and its explanation have one writer. Mathematical endpoint
// derivation/rounding belongs to the caller; this operation only intersects
// the domain and records exactly the update it commits, including crossings.
// Unknown explanations stay explicit and cannot authorize conflict resolution.
inline bool write_domain_bound(std::vector<f64>& lo, std::vector<f64>& hi,
                               Index var, BoundDir dir, f64 value,
                               PropTrail* trail, DomainReason reason,
                               DomainEventLog* events = nullptr, int source = -1) {
    if (lo.size() != hi.size() || var < 0 || static_cast<std::size_t>(var) >= lo.size() ||
        std::isnan(value)) throw std::invalid_argument("domain writer: malformed bound");
    f64& side = (dir == BoundDir::Upper ? hi : lo)[static_cast<std::size_t>(var)];
    if (dir == BoundDir::Upper ? value >= side : value <= side) return false;
    const f64 old = side;
    record_domain_change(var, dir, old, value, trail, reason, events, source);
    side = value;
    return true;
}

// The same transaction applies to a paged frontier domain. Page allocation
// precedes proof publication; committing the prepared checkpoint is noexcept.
inline bool write_domain_bound(DomainCheckpoint& box, Index var, BoundDir dir,
                               f64 value, PropTrail* trail, DomainReason reason,
                               DomainEventLog* events = nullptr, int source = -1) {
    if (var < 0 || static_cast<std::size_t>(var) >= box.size() || std::isnan(value))
        throw std::invalid_argument("domain writer: malformed checkpoint bound");
    const f64 old = dir == BoundDir::Upper ? box.upper(var) : box.lower(var);
    if (dir == BoundDir::Upper ? value >= old : value <= old) return false;
    auto next = box;
    if (dir == BoundDir::Upper) next.set_upper(var, value); else next.set_lower(var, value);
    record_domain_change(var, dir, old, value, trail, reason, events, source);
    box = std::move(next);
    return true;
}

// Replay a proved probe's deductions as one checkpoint transaction. Journal
// entries retain the actual committed old endpoint and the owned explanation;
// duplicate writes are intersections, not fresh unreasoned assignments.
inline std::vector<Index> write_domain_bounds(DomainCheckpoint& box, PropTrail& trail,
                                              const std::vector<PropTrailEntry>& proof,
                                              int depth) {
    std::map<Index, std::pair<f64, f64>> current;
    std::vector<DomainCheckpoint::Edit> edits;
    std::vector<PropTrailEntry> committed;
    for (const auto& e : proof) {
        if (e.var < 0 || static_cast<std::size_t>(e.var) >= box.size() || std::isnan(e.new_bound))
            throw std::invalid_argument("domain proof replay: malformed bound");
        auto [it, inserted] = current.try_emplace(e.var);
        if (inserted) it->second = {box.lower(e.var), box.upper(e.var)};
        auto& side = e.dir == BoundDir::Upper ? it->second.second : it->second.first;
        if (e.dir == BoundDir::Upper ? e.new_bound >= side : e.new_bound <= side) continue;
        auto entry = e;
        entry.old_bound = side;
        entry.depth = depth;
        committed.push_back(std::move(entry));
        edits.push_back({static_cast<std::size_t>(e.var), e.new_bound, e.dir == BoundDir::Upper});
        side = e.new_bound;
    }
    auto next = box;
    next.apply(edits);
    std::vector<Index> moved;
    moved.reserve(committed.size());
    for (const auto& e : committed) moved.push_back(e.var);
    const auto mark = trail.size();
    try {
        for (const auto& e : committed)
            trail.push(e.var, e.dir, e.new_bound, e.old_bound, e.kind, e.reason_id,
                       e.depth, e.clause, e.antecedents);
    } catch (...) { trail.truncate(mark); throw; }
    box = std::move(next);
    return moved;
}

// Probes own one trail and restore its exact endpoints in reverse order.
// Undo itself is an event: persistent activities and watches must see widening
// before another deduction can use their retained state.
inline void undo_domain(std::vector<f64>& lo, std::vector<f64>& hi,
                        PropTrail& trail, std::size_t mark,
                        DomainEventLog* events = nullptr) {
    if (mark > trail.size() || lo.size() != hi.size())
        throw std::invalid_argument("domain undo: malformed checkpoint");
    for (std::size_t k = 0; k < trail.size() - mark; ++k) {
        const auto& e = trail.at_reverse(k);
        auto& side = (e.dir == BoundDir::Upper ? hi : lo).at(static_cast<std::size_t>(e.var));
        if (side != e.new_bound) throw std::logic_error("domain undo: incomplete writer trail");
        if (events) events->record(e.var, e.dir == BoundDir::Upper, side, e.old_bound,
                                   e.depth, -1, e.kind, e.reason_id);
        side = e.old_bound;
    }
    trail.truncate(mark);
}
} // namespace sor::search
