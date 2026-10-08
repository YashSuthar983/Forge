// SOR - propagation reason trail (WP-A). Records bound changes for Mexi-style
// reverse walk; does not affect dual bounds by itself.
#pragma once

#include "sor/core/result.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

// What a Row-family reason names must not depend on when it is read. The
// global LP's rows are append-only except for the tree working-set purge,
// which rewrites every stored trail; a node's local cut rows sit after the
// global rows of ITS LP, so their position means a different row once a global
// row is appended. Hence three kinds:
//   Row     - global row `reason_id` of the current global LP;
//   Cut     - a subtree-local cut row of the LP the bound was derived on:
//             reason_id is its position after that LP's global rows. Valid in
//             the node's subtree only; no global explanation may resolve it;
//   Retired - a global row the purge deleted while a trail still named it:
//             reason_id indexes ReasonStore::rows. Still a valid global
//             inequality, so explanations read it there;
//   Antecedents - a deduction that is no single row's propagation (the
//             objective cutoff row with its clique groups, reduced-cost
//             fixing from LP duals): reason_id indexes
//             ReasonStore::antecedents, the bounds it rests on beside global
//             facts (rows, cliques, the incumbent cutoff). Listing a bound
//             that was not needed only weakens an explanation.
enum class ReasonKind : std::uint8_t {
    Row = 0,
    Cut = 1,
    Conflict = 2,
    Branch = 3,
    Unknown = 4,
    // A conflict-graph implication: reason_id is the literal (2j + v) whose
    // truth made this bound's literal false (a global pair conflict).
    Implication = 5,
    // A conflict-store clause forced this bound: reason_id is the clause's
    // stable id (ConflictStore::clause_by_id); the other literals were false.
    Clause = 6,
    Retired = 7,
    Antecedents = 8,
    // A symmetry-handling restriction (orbitopal fixing): valid only for
    // the representatives the search keeps, so no explanation resolves it.
    Symmetry = 9,
};

// Content of a retired global row (see ReasonKind::Retired): lo <= a'x <= hi.
struct RetiredRow {
    std::vector<Index> cols;
    std::vector<f64> vals;
    f64 lo = 0.0;
    f64 hi = 0.0;
};
// One bound of a column: its lower (upper = false) or upper bound.
struct BoundRef {
    Index var = -1;
    bool upper = false;
};
// What Retired and Antecedents reasons name. Owned by the solve; entries are
// shared by every trail that copied the reason, and only grow between
// compactions that rewrite those trails.
struct ReasonStore {
    std::vector<RetiredRow> rows;
    std::vector<std::vector<BoundRef>> antecedents;
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

// A node's trail is its ancestors' entries followed by its own. Children
// copy their parent's node, so the history is shared instead of copied: a
// copy freezes the source's own entries into an immutable segment that both
// trails then reference, and each trail appends only its own entries. A
// branch used to copy every entry back to the root into both children (mas76:
// the largest single memcpy source of the search once deductions carried
// reasons). Reading is unchanged: entries() is the whole history in order
// (flattened into a cache when there is a shared prefix), since(mark) the
// entries pushed after a mark without flattening.
class PropTrail {
public:
    PropTrail() = default;
    // Shares `o`'s history; `o`'s own entries are frozen into a segment.
    PropTrail(const PropTrail& o) { share_from(o); }
    PropTrail& operator=(const PropTrail& o) {
        if (this != &o) share_from(o);
        return *this;
    }
    PropTrail(PropTrail&&) noexcept = default;
    PropTrail& operator=(PropTrail&&) noexcept = default;

    void clear() {
        prefix_.reset();
        prefix_size_ = 0;
        own_.clear();
        flat_.clear();
        flat_valid_ = false;
        next_index_ = 0;
    }
    std::size_t size() const { return prefix_size_ + own_.size(); }
    // The whole history, oldest first.
    const std::vector<PropTrailEntry>& entries() const {
        if (!prefix_) return own_;
        if (!flat_valid_) {
            flat_.clear();
            flat_.reserve(size());
            append_segment(prefix_.get(), flat_);
            flat_.insert(flat_.end(), own_.begin(), own_.end());
            flat_valid_ = true;
        }
        return flat_;
    }
    // Entries [mark, size()) as a contiguous range.
    struct Range {
        const PropTrailEntry* first = nullptr;
        const PropTrailEntry* last = nullptr;
        const PropTrailEntry* begin() const { return first; }
        const PropTrailEntry* end() const { return last; }
        std::size_t size() const { return static_cast<std::size_t>(last - first); }
    };
    Range since(std::size_t mark) const {
        if (mark >= prefix_size_) {
            const std::size_t off = std::min(mark - prefix_size_, own_.size());
            return {own_.data() + off, own_.data() + own_.size()};
        }
        const auto& all = entries();
        const std::size_t off = std::min(mark, all.size());
        return {all.data() + off, all.data() + all.size()};
    }
    // Visits every entry, oldest first, without flattening.
    template <class Fn>
    void for_each(Fn&& fn) const {
        visit_segment(prefix_.get(), fn);
        for (const auto& e : own_) fn(e);
    }

    // Drop everything pushed after `n`. A backtracking search records
    // trail.size() before a decision and undoes back to that mark by
    // restoring each entry's old_bound in reverse order; see fixprop.cpp.
    void truncate(std::size_t n) {
        if (n >= size()) return;
        if (n < prefix_size_) materialize();
        own_.resize(n - prefix_size_);
        flat_valid_ = false;
    }

    void push(Index var, BoundDir dir, f64 new_bound, f64 old_bound,
              ReasonKind kind, Index reason_id, int depth);

    // Reverse chronological order (newest first).
    const PropTrailEntry& at_reverse(std::size_t k) const {
        if (k < own_.size()) return own_[own_.size() - 1 - k];
        const auto& all = entries();
        return all[all.size() - 1 - k];
    }

    // Rows were deleted from the global model: a Row reason moves to
    // new_row[r]; a deleted row's reason becomes Retired, its index given by
    // retire(r) (which stores the row's content once per row). Cut reasons
    // (local rows) are positions in their own LP's local block and do not move.
    // `epoch` names the remap pass: a segment shared by several trails is
    // rewritten once per pass, however many trails reach it.
    template <class Retire>
    void remap_row_reasons(const std::vector<Index>& new_row, Retire&& retire,
                           std::uint64_t epoch) {
        const auto fix = [&](PropTrailEntry& e) {
            if (e.kind != ReasonKind::Row || e.reason_id < 0 ||
                static_cast<std::size_t>(e.reason_id) >= new_row.size())
                return;
            const Index r = new_row[static_cast<std::size_t>(e.reason_id)];
            if (r >= 0) {
                e.reason_id = r;
            } else {
                e.kind = ReasonKind::Retired;
                e.reason_id = retire(e.reason_id);
            }
        };
        rewrite(fix, epoch);
    }

    // ReasonStore compaction: Retired / Antecedents ids move to
    // row_id[id] / ante_id[id] (both >= 0 for every id a trail names).
    void remap_store_reasons(const std::vector<Index>& row_id,
                             const std::vector<Index>& ante_id, std::uint64_t epoch) {
        const auto fix = [&](PropTrailEntry& e) {
            if (e.reason_id < 0) return;
            if (e.kind == ReasonKind::Retired)
                e.reason_id = row_id[static_cast<std::size_t>(e.reason_id)];
            else if (e.kind == ReasonKind::Antecedents)
                e.reason_id = ante_id[static_cast<std::size_t>(e.reason_id)];
        };
        rewrite(fix, epoch);
    }

private:
    struct Segment {
        std::vector<PropTrailEntry> entries;
        std::shared_ptr<Segment> parent;
        std::uint64_t epoch = 0;   // last remap pass applied
    };
    static void append_segment(const Segment* s, std::vector<PropTrailEntry>& out) {
        if (s == nullptr) return;
        append_segment(s->parent.get(), out);
        out.insert(out.end(), s->entries.begin(), s->entries.end());
    }
    template <class Fn>
    static void visit_segment(const Segment* s, Fn& fn) {
        if (s == nullptr) return;
        visit_segment(s->parent.get(), fn);
        for (const auto& e : s->entries) fn(e);
    }
    template <class Fix>
    void rewrite(Fix& fix, std::uint64_t epoch) {
        for (Segment* s = prefix_.get(); s != nullptr && s->epoch != epoch; s = s->parent.get()) {
            for (auto& e : s->entries) fix(e);
            s->epoch = epoch;
        }
        for (auto& e : own_) fix(e);
        flat_valid_ = false;
    }
    // Freeze `o`'s own entries (logically const: its history is unchanged)
    // and share the result.
    void share_from(const PropTrail& o) {
        if (!o.own_.empty()) {
            auto seg = std::make_shared<Segment>();
            seg->entries = std::move(o.own_);
            seg->parent = o.prefix_;
            o.prefix_size_ += seg->entries.size();
            o.prefix_ = std::move(seg);
            o.own_.clear();
        }
        prefix_ = o.prefix_;
        prefix_size_ = o.prefix_size_;
        own_.clear();
        flat_.clear();
        flat_valid_ = false;
        next_index_ = o.next_index_;
    }
    // Own copy of the whole history (a truncate into the shared prefix).
    void materialize() {
        std::vector<PropTrailEntry> all = entries();
        prefix_.reset();
        prefix_size_ = 0;
        own_ = std::move(all);
        flat_.clear();
        flat_valid_ = false;
    }

    mutable std::shared_ptr<Segment> prefix_;
    mutable std::size_t prefix_size_ = 0;
    mutable std::vector<PropTrailEntry> own_;
    mutable std::vector<PropTrailEntry> flat_;
    mutable bool flat_valid_ = false;
    std::uint32_t next_index_ = 0;
};

}  // namespace sor::search
