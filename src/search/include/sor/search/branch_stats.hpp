// Branching evidence: what each branch of each variable was observed to do.
//
// Two kinds of evidence are kept apart on purpose.
//   * Objective pseudocosts: LP-bound gain per unit of integer movement, from
//     PROVED optimal relaxations only. A proved zero gain is a real sample (the
//     bound really does not move); an incomplete LP contributes nothing.
//   * Feasibility usefulness, per direction: how often a branch was certified
//     closed (infeasible, or cut off by the incumbent) and how many bound
//     deductions propagation derived in it. Rates, not counts, so a variable
//     that is merely visited often does not win by exposure.
// Every observation is recorded through record(); an ordinary child's single
// observation is consumed once per incoming branch (a requeued node resolves
// again but is the same sample).
#pragma once

#include "sor/core/result.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sor::search {

enum class BranchOutcome {
    ProvedOptimal,        // relaxation solved to a certified optimum
    CertifiedInfeasible,  // branch proved empty (propagation or verified ray)
    CertifiedCutoff,      // branch proved no better than the incumbent
    DomainOnly,           // survived propagation: feasibility evidence only, no LP
    Incomplete,           // no certified result: not evidence
};

struct BranchStats {
    using f64 = core::f64;
    using Index = core::Index;

    // Objective pseudocosts (index 0 = down, 1 = up).
    std::vector<f64> pc_sum[2];
    std::vector<std::uint32_t> pc_count[2];
    // Feasibility usefulness.
    std::vector<std::uint32_t> closed[2];     // certified infeasible / cutoff
    std::vector<std::uint32_t> observed[2];   // certified outcomes (closed or optimal)
    std::vector<f64> inference_sum[2];        // bound deductions in the branch
    std::vector<std::uint32_t> inference_n[2];

    std::uint64_t objective_samples = 0;
    std::uint64_t closure_samples = 0;
    std::uint64_t ignored_incomplete = 0;
    std::uint64_t ignored_repeat = 0;
    std::uint64_t replaced = 0;   // a re-solved child's sample replaced its earlier one

    void resize(Index n) {
        for (int d = 0; d < 2; ++d) {
            pc_sum[d].assign(static_cast<std::size_t>(n), 0.0);
            pc_count[d].assign(static_cast<std::size_t>(n), 0);
            closed[d].assign(static_cast<std::size_t>(n), 0);
            observed[d].assign(static_cast<std::size_t>(n), 0);
            inference_sum[d].assign(static_cast<std::size_t>(n), 0.0);
            inference_n[d].assign(static_cast<std::size_t>(n), 0);
        }
    }
    Index size() const { return static_cast<Index>(pc_sum[0].size()); }

    // One observation of branch (var, dir) of integer movement `dist`. For
    // ProvedOptimal `gain` is the proved child bound minus the parent
    // reference (clamped at 0). `inferences` is the number of bound deductions
    // propagation made inside that branch (negative: not measured).
    // Returns whether anything was recorded.
    bool record(Index var, int dir, f64 dist, BranchOutcome outcome, f64 gain,
                f64 inferences, f64 int_tol = 1e-6) {
        if (var < 0 || var >= size() || dir == 0) return false;
        const auto j = static_cast<std::size_t>(var);
        const int d = dir < 0 ? 0 : 1;
        switch (outcome) {
        case BranchOutcome::Incomplete:
            ++ignored_incomplete;
            return false;
        case BranchOutcome::ProvedOptimal:
            if (!(dist > int_tol) || !(gain == gain)) return false;
            pc_sum[d][j] += (gain > 0.0 ? gain : 0.0) / dist;
            ++pc_count[d][j];
            ++objective_samples;
            break;
        case BranchOutcome::CertifiedCutoff:
            // The relaxation was solved (finite gain) and then found no better
            // than the incumbent: a closure, and also a real objective sample.
            if (dist > int_tol && gain == gain) {
                pc_sum[d][j] += (gain > 0.0 ? gain : 0.0) / dist;
                ++pc_count[d][j];
                ++objective_samples;
            }
            ++closed[d][j];
            ++closure_samples;
            break;
        case BranchOutcome::CertifiedInfeasible:
            ++closed[d][j];
            ++closure_samples;
            break;
        case BranchOutcome::DomainOnly:
            break;
        }
        ++observed[d][j];
        if (inferences >= 0.0) {
            inference_sum[d][j] += inferences;
            ++inference_n[d][j];
        }
        return true;
    }

    // An ordinary child's observation: consumed at most once per incoming
    // branch. `consumed` lives on the node; a requeue keeps it.
    //
    // A node re-solved after a deduction tightened its box (strong branching
    // fixed a bound and requeued it) is the SAME child with a stronger
    // relaxation: its new objective sample REPLACES the earlier one in the sums
    // (count unchanged) instead of being a second independent sample.
    // `recorded_unit` is the unit gain this child last contributed (NaN if none).
    bool record_child(Index var, int dir, f64 dist, BranchOutcome outcome, f64 gain,
                      f64 inferences, bool& consumed, f64& recorded_unit,
                      f64 int_tol = 1e-6) {
        if (consumed) {
            const bool objective = outcome == BranchOutcome::ProvedOptimal ||
                                   outcome == BranchOutcome::CertifiedCutoff;
            if (objective && var >= 0 && var < size() && dir != 0 && dist > int_tol &&
                gain == gain && recorded_unit == recorded_unit) {
                const int d = dir < 0 ? 0 : 1;
                const f64 unit = (gain > 0.0 ? gain : 0.0) / dist;
                pc_sum[d][static_cast<std::size_t>(var)] += unit - recorded_unit;
                recorded_unit = unit;
                ++replaced;
                return true;
            }
            ++ignored_repeat;
            return false;
        }
        const bool ok = record(var, dir, dist, outcome, gain, inferences, int_tol);
        if (ok) {
            consumed = true;
            if ((outcome == BranchOutcome::ProvedOptimal ||
                 outcome == BranchOutcome::CertifiedCutoff) && dist > int_tol && gain == gain)
                recorded_unit = (gain > 0.0 ? gain : 0.0) / dist;
        }
        return ok;
    }

    // Observed rate at which branches of (var, dir) were certified closed,
    // and the mean inference count; NaN when never observed.
    f64 closure_rate(Index var, int dir) const {
        const auto j = static_cast<std::size_t>(var);
        const int d = dir < 0 ? 0 : 1;
        return observed[d][j] ? static_cast<f64>(closed[d][j]) / observed[d][j]
                              : static_cast<f64>(0.0) / 0.0;
    }
    f64 inference_rate(Index var, int dir) const {
        const auto j = static_cast<std::size_t>(var);
        const int d = dir < 0 ? 0 : 1;
        return inference_n[d][j] ? inference_sum[d][j] / inference_n[d][j]
                                 : static_cast<f64>(0.0) / 0.0;
    }
};

// Ordering key of a branching candidate: objective score to numerical
// resolution first, then the OBSERVED certified-closure rate, then the observed
// inference rate. Exposure does not help a candidate (rates, not counts); a
// direction never observed counts as the model-wide average so it neither wins
// nor loses by being new. Remaining ties are the caller's stable order.
struct BranchKey {
    long long bucket = 0;
    core::f64 closure = 0.0;
    core::f64 inference = 0.0;
};

inline long long score_bucket(core::f64 score) {
    if (!(score > 1e-300)) score = 1e-300;
    // A product of finite gains can overflow. Ranking may saturate, but
    // converting log(+inf) to an integer is undefined behaviour.
    if (!std::isfinite(score)) return std::numeric_limits<long long>::max();
    return static_cast<long long>(std::floor(std::log(score) * 1e9));
}

// True if a is strictly preferred to b.
inline bool key_better(const BranchKey& a, const BranchKey& b) {
    if (a.bucket != b.bucket) return a.bucket > b.bucket;
    if (std::fabs(a.closure - b.closure) > 1e-12) return a.closure > b.closure;
    if (std::fabs(a.inference - b.inference) > 1e-12) return a.inference > b.inference;
    return false;
}

// Model-wide averages used for directions with no observation.
struct FeasibilityMeans {
    core::f64 closure = 0.0;
    core::f64 inference = 0.0;
};

inline FeasibilityMeans feasibility_means(const BranchStats& s) {
    core::f64 c = 0.0, i = 0.0;
    std::uint64_t co = 0, in = 0;
    for (int d = 0; d < 2; ++d)
        for (std::size_t j = 0; j < s.closed[d].size(); ++j) {
            c += s.closed[d][j];
            co += s.observed[d][j];
            i += s.inference_sum[d][j];
            in += s.inference_n[d][j];
        }
    return {co ? c / static_cast<core::f64>(co) : 0.0,
            in ? i / static_cast<core::f64>(in) : 0.0};
}

inline BranchKey branch_key(const BranchStats& s, core::Index var, core::f64 score,
                            const FeasibilityMeans& m) {
    const auto j = static_cast<std::size_t>(var);
    core::f64 c = 0.0, i = 0.0;
    for (int d = 0; d < 2; ++d) {
        c += s.observed[d][j] ? static_cast<core::f64>(s.closed[d][j]) / s.observed[d][j] : m.closure;
        i += s.inference_n[d][j] ? s.inference_sum[d][j] / s.inference_n[d][j] : m.inference;
    }
    return {score_bucket(score), 0.5 * c, 0.5 * i};
}

// Has neither direction had its propagation deductions measured? LP probes
// contribute closure/objective evidence, but with inferences = -1 they must
// not prevent the domain probe that measures this separate kind of evidence.
inline bool inference_unobserved(const BranchStats& s, core::Index var) {
    const auto j = static_cast<std::size_t>(var);
    return s.inference_n[0][j] == 0 && s.inference_n[1][j] == 0;
}

// Does a strong-branching batch that has already run `batch_elapsed_ms` still
// have allowance? `accumulated_ms` is what earlier batches spent; the batch's
// own running time counts against the allowance BEFORE it is added to the
// total, otherwise one batch can overshoot by its whole length.
inline bool batch_has_allowance(core::f64 accumulated_ms, core::f64 batch_elapsed_ms,
                                core::f64 lp_share, core::f64 lp_ms,
                                core::f64 startup_ms) {
    return accumulated_ms + batch_elapsed_ms <= lp_share * lp_ms + startup_ms;
}

}  // namespace sor::search
