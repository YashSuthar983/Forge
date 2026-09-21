// SOR - propagation reason trail (WP-A). Records bound changes for Mexi-style
// reverse walk; does not affect dual bounds by itself.
#pragma once

#include "sor/core/result.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

enum class ReasonKind : std::uint8_t {
    Row = 0,
    Cut = 1,
    Conflict = 2,
    Branch = 3,
    Unknown = 4,
};

enum class BoundDir : std::uint8_t {
    Lower = 0,
    Upper = 1,
};

struct PropTrailEntry {
    Index var = -1;
    BoundDir dir = BoundDir::Lower;
    f64 new_bound = 0.0;
    f64 old_bound = 0.0;
    ReasonKind kind = ReasonKind::Unknown;
    Index reason_id = -1;  // row / cut / conflict index when applicable
    int depth = 0;
    std::uint32_t trail_index = 0;
};

class PropTrail {
public:
    void clear() { entries_.clear(); next_index_ = 0; }
    std::size_t size() const { return entries_.size(); }
    const std::vector<PropTrailEntry>& entries() const { return entries_; }

    // Drop everything pushed after `n`. A backtracking search records
    // trail.size() before a decision and undoes back to that mark by
    // restoring each entry's old_bound in reverse order; see fixprop.cpp.
    void truncate(std::size_t n) {
        if (n < entries_.size()) entries_.resize(n);
    }

    void push(Index var, BoundDir dir, f64 new_bound, f64 old_bound,
              ReasonKind kind, Index reason_id, int depth);

    // Reverse chronological order (newest first).
    const PropTrailEntry& at_reverse(std::size_t k) const {
        return entries_[entries_.size() - 1 - k];
    }

private:
    std::vector<PropTrailEntry> entries_;
    std::uint32_t next_index_ = 0;
};

}  // namespace sor::search
