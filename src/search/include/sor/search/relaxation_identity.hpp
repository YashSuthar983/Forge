// What relaxation an LP snapshot is. Identity is the model's CONTENT -- column
// space, the exact matrix and integrality, row sides, the column box and the
// objective -- never a proxy such as the row count or the optimal value: two
// different cut systems can agree on both and still have different points,
// bases and implications. The bound and the incumbent target are attributes of
// a snapshot (scheduling observations), not part of its identity.
#pragma once

#include "sor/model/lp.hpp"
#include "sor/search/conflict.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace sor::search {

struct RelaxationIdentity {
    core::Index cols = -1;
    core::Index rows = -1;
    std::uint64_t matrix = 0;      // matrix + integrality + row sides
    std::uint64_t box = 0;         // column bounds
    std::uint64_t objective = 0;   // cost vector
    bool operator==(const RelaxationIdentity& o) const {
        return cols == o.cols && rows == o.rows && matrix == o.matrix && box == o.box &&
               objective == o.objective;
    }
    bool operator!=(const RelaxationIdentity& o) const { return !(*this == o); }
};

namespace detail {
inline std::uint64_t hash_doubles(std::uint64_t h, const std::vector<core::f64>& v) {
    for (const core::f64 x : v) {
        std::uint64_t b;
        std::memcpy(&b, &x, sizeof b);
        h = (h ^ b) * 1099511628211ull;
    }
    return h;
}
}  // namespace detail

// `box_lo/box_hi` are the column bounds the relaxation is solved over (the root
// box, which may be tighter than lp's own).
inline RelaxationIdentity relaxation_identity(const model::LpProblem& lp,
                                              const std::vector<core::f64>& box_lo,
                                              const std::vector<core::f64>& box_hi) {
    RelaxationIdentity id;
    id.cols = lp.n_cols();
    id.rows = lp.n_rows();
    std::uint64_t h = matrix_fingerprint(lp);
    h = detail::hash_doubles(h, lp.row_lo);
    h = detail::hash_doubles(h, lp.row_hi);
    id.matrix = h;
    id.box = detail::hash_doubles(detail::hash_doubles(1469598103934665603ull, box_lo), box_hi);
    id.objective = detail::hash_doubles(1469598103934665603ull, lp.c) ^
                   (lp.maximize ? 0x5bd1e995ull : 0ull);
    return id;
}

}  // namespace sor::search
