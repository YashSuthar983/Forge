// SOR - mixing inequalities over a shared conditional activity.
//
// LAYER L5, search. Several rows can make one nonnegative activity z depend
// on different binary literals,
//     z >= h_i (1 - l_i),   h_i > 0,  l_i a binary literal (y or 1 - y).
// Each row alone is already exact at its two binary endpoints, and tightening
// its h cannot remove a fractional point such as z = 1, l_1 = 0, l_2 = 0.5
// for h = (1, 2): the rows interact only jointly. For a chain
// h_{s_1} <= ... <= h_{s_k} of them (h_{s_0} = 0)
//     z + sum_t (h_{s_t} - h_{s_{t-1}}) l_{s_t} >= h_{s_k}
// holds at every integer point: if every literal of the chain is 1 the
// coefficients telescope to h_{s_k} (z >= 0 does the rest); otherwise, with
// s_j the last literal at 0, z >= h_{s_j} and the literals after it give
// h_{s_k} - h_{s_j}, the earlier ones nothing negative (the mixing set,
// Günlük & Pochet 2001). Rounding a coefficient up keeps the proof (the
// literal is nonnegative), so differences are rounded up, the side of a
// complemented literal's constant moved down, and every row is exact.
//
// Separation is the chain recurrence over v_i = 1 - l_i(x), sorted by h:
//     F(i) = max( h_i v_i,  max_{j < i} F(j) + (h_i - h_j) v_i ),
// and a chain with F(i) > z(x) is violated by F(i) - z(x).
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/cuts.hpp"

#include <vector>

namespace sor::search {

struct MixingFamily {
    Index z = -1;
    // Sorted by h ascending; lit[i] = 2 * column + v (literal "column = v").
    std::vector<Index> lit;
    std::vector<f64> h;
};

struct MixingIndex {
    std::vector<MixingFamily> families;
    bool empty() const noexcept { return families.empty(); }
};

// Families of `lp`'s two-entry rows a z + b y >= r (or <= after negation)
// with z >= 0 by its own bound and y binary, read off the model's data: the
// threshold h is r / a (b > 0, 0 < r <= b: z >= h (1 - y)) or |b| / a
// (b < 0, r = 0: z >= h y), each rounded down (a weaker premise stays true).
// Families with fewer than two literals are dropped.
MixingIndex build_mixing_index(const model::LpProblem& lp);

// Violated mixing rows at x, at most `max_cuts`, most violated first.
std::vector<CutRow> separate_mixing_cuts(const MixingIndex& index,
                                         const std::vector<f64>& x,
                                         int max_cuts, f64 violation_min);

}  // namespace sor::search
