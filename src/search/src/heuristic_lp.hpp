#pragma once
// Simplex options for LP solves whose only product is a primal point.
#include "forge/engines/simplex.hpp"

namespace forge::search {

// A heuristic LP's point is checked against the original model before it is
// used, and its bound and rays prove nothing. The engine's defaults certify
// every terminal bound and ray exactly (dual repair, exact Farkas witnesses),
// which the proof needs from node LPs and no heuristic reads: on mas76 that
// repair was 48% of a 20 s run, in LP rounding repair (from the integer
// neighbourhood and fractional enumeration) and the feasibility pump. A solve
// without it still returns Optimal on a basis whose residuals are within
// tolerance, the standard every floating-point LP solver stops at.
inline void heuristic_lp_options(engines::SimplexOptions& so) {
    so.certify_terminal = false;
    so.exact_proof = false;
    so.certify_rays = false;
}

}  // namespace forge::search
