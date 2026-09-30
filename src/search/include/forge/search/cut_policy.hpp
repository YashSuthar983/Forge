// Context-aware cut selection / optional-separator gating (Turner, Berthold &
// Besançon, arXiv:2307.07322). Clean-room: scoring and stopping rules from the
// paper, not SCIP/HiGHS source.
#pragma once

#include "forge/search/bab.hpp"
#include "forge/search/cuts.hpp"
#include "forge/search/milp_policy.hpp"

#include <vector>

namespace forge::search {

// When true, BabOptions::auto_cuts enables the Turner pool weights, pre-pool
// candidate filtering, and the optional separator subset (MIR, lifted cover,
// zero-half). Flow cover has a known validity counterexample and stays off.
// Clique cuts stay off unless explicitly
// requested (--clique-cuts).
void apply_auto_cuts_policy(BabOptions& o);

// Pre-pool greedy rank/limit: normalised efficacy + sparsity / locks / objective
// alignment, parallelism penalties, nnz budget - mirrors CutPool selection but
// on a single round's candidate batch so separators can stay generous.
// When `stats` is non-null every dropped candidate is counted there by reason.
std::vector<CutRow> filter_cut_candidates_for_round(
    std::vector<CutRow> candidates, const model::LpProblem& lp,
    const std::vector<f64>& x, const CutOptions& opts,
    CutFilterStats* stats = nullptr);

}  // namespace forge::search
