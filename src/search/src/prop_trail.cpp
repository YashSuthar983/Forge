#include "sor/search/prop_trail.hpp"

namespace sor::search {

void PropTrail::push(Index var, BoundDir dir, f64 new_bound, f64 old_bound,
                     ReasonKind kind, Index reason_id, int depth,
                     std::shared_ptr<const std::vector<ConflictLiteral>> clause,
                     std::shared_ptr<const std::vector<BoundRef>> antecedents) {
    PropTrailEntry e;
    e.var = var;
    e.dir = dir;
    e.new_bound = new_bound;
    e.old_bound = old_bound;
    e.kind = kind;
    e.reason_id = reason_id;
    e.depth = depth;
    e.trail_index = next_index_++;
    e.clause = std::move(clause);
    e.antecedents = std::move(antecedents);
    own_.push_back(e);
    flat_valid_ = false;
}

}  // namespace sor::search
