#include "sor/search/prop_trail.hpp"

namespace sor::search {

void PropTrail::push(Index var, BoundDir dir, f64 new_bound, f64 old_bound,
                     ReasonKind kind, Index reason_id, int depth) {
    PropTrailEntry e;
    e.var = var;
    e.dir = dir;
    e.new_bound = new_bound;
    e.old_bound = old_bound;
    e.kind = kind;
    e.reason_id = reason_id;
    e.depth = depth;
    e.trail_index = next_index_++;
    entries_.push_back(e);
}

}  // namespace sor::search
