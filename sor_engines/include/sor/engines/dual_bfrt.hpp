#pragma once

#include "sor/engines/simplex.hpp"

#include <cstddef>
#include <vector>

namespace sor::engines {

// Bound-flipping (long-step) dual ratio test. Koberstein & Suhl (2007);
// Huangfu & Hall (2018). HiGHS HEkkDualRow consulted for understanding only.
struct DualBfrtResult {
    core::Index               pivot     = -1;
    int                       pivot_dir = 0;
    core::f64                 d_enter   = 0.0;
    core::f64                 theta_dual = 0.0;  // dual long-step (≠ primal t_step)
    std::vector<core::Index>  flips;
    bool                      ok        = false;
};

// Reusable storage for the long-step ratio sweep. Keeping the arrays owned by
// the dual solve removes several heap allocations per pivot without changing
// the candidate arithmetic or tie-breaking.
struct DualBfrtWorkspace {
    std::vector<core::Index> order;
    std::vector<core::Index> column;
    std::vector<core::f64> alpha;
    std::vector<core::f64> dual;
    std::vector<core::f64> range;
    std::vector<core::f64> ratio;
};

// Harris two-pass step (pass 1) on pivotal-row candidates.
core::f64 dual_harris_theta(const std::vector<core::Index>& cand_j,
                            const std::vector<core::f64>& cand_aj,
                            const std::vector<core::f64>& cand_d,
                            const std::vector<NonbasicStatus>& st,
                            core::f64 srow, core::f64 dual_slack);

// Legacy single-step ratio test: max |alpha_rj| among te <= harris_theta.
DualBfrtResult dual_legacy_ratio(const std::vector<core::Index>& cand_j,
                                 const std::vector<core::f64>& cand_aj,
                                 const std::vector<core::f64>& cand_d,
                                 const std::vector<NonbasicStatus>& st,
                                 core::f64 srow, core::f64 harris_theta,
                                 core::f64 pivot_tol);

// Full BFRT (chooseFinal). Returns ok=false → caller should use dual_legacy_ratio.
// `dual_slack` is the Harris relaxation; the long step is capped at the
// slack-adjusted ratio of the first NON-flippable candidate.
DualBfrtResult dual_bfrt_choose(const std::vector<core::Index>& cand_j,
                                 const std::vector<core::f64>& cand_aj,
                                 const std::vector<core::f64>& cand_d,
                                 const std::vector<NonbasicStatus>& st,
                                 const std::vector<core::f64>& lo,
                                 const std::vector<core::f64>& hi,
                                 core::f64 delta_primal, core::f64 srow,
                                 core::f64 harris_theta,
                                 core::f64 pivot_tol, core::f64 dual_tol,
                                 core::f64 dual_slack,
                                 DualBfrtWorkspace* workspace = nullptr);

void flip_nonbasic(NonbasicStatus& st, core::f64& val,
                   core::f64 lo_j, core::f64 hi_j);

inline int nonbasic_move(NonbasicStatus s) {
    switch (s) {
        case NonbasicStatus::AtLower: return +1;
        case NonbasicStatus::AtUpper: return -1;
        default: return 0;
    }
}

}  // namespace sor::engines
