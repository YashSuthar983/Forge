#include "sor/engines/dual_bfrt.hpp"

#include "sor/model/lp.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace sor::engines {
namespace {

using core::f64;
using core::Index;
using model::kInf;

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

constexpr f64 kInfTheta = std::numeric_limits<f64>::infinity();

struct Cand {
    Index j     = -1;
    f64   alpha = 0.0;   // |a_rj| > 0
    f64   dual  = 0.0;   // d_j (signed)
    f64   range = 0.0;   // hi - lo > 0 (boxed only)
    f64   te    = 0.0;   // |d_j| / |a_rj| >= 0, the dual ratio
};

}  // namespace

void flip_nonbasic(NonbasicStatus& st, f64& val, f64 lo_j, f64 hi_j) {
    if (st == NonbasicStatus::AtLower) {
        st = NonbasicStatus::AtUpper;
        val = hi_j;
    } else if (st == NonbasicStatus::AtUpper) {
        st = NonbasicStatus::AtLower;
        val = lo_j;
    }
}

f64 dual_harris_theta(const std::vector<Index>& cand_j,
                      const std::vector<f64>& cand_aj,
                      const std::vector<f64>& cand_d,
                      const std::vector<NonbasicStatus>& st,
                      const f64 srow, const f64 dual_slack) {
    f64 t_max = kInf;
    for (std::size_t c = 0; c < cand_j.size(); ++c) {
        const f64 sa = srow * cand_aj[c];
        const f64 d  = cand_d[c];
        f64 tb = kInf;
        switch (st[sz(cand_j[c])]) {
            case NonbasicStatus::AtLower:
                tb = (d + dual_slack) / (-sa);
                break;
            case NonbasicStatus::AtUpper:
                tb = (dual_slack - d) / sa;
                break;
            case NonbasicStatus::AtZeroFree:
                tb = 0.0;
                break;
            default:
                break;
        }
        if (tb < t_max) t_max = tb;
    }
    if (t_max < 0.0) t_max = 0.0;
    return t_max;
}

DualBfrtResult dual_legacy_ratio(const std::vector<Index>& cand_j,
                                 const std::vector<f64>& cand_aj,
                                 const std::vector<f64>& cand_d,
                                 const std::vector<NonbasicStatus>& st,
                                 const f64 srow, const f64 harris_theta,
                                 const f64 pivot_tol) {
    DualBfrtResult out;
    f64 best_piv = 0.0;
    for (std::size_t c = 0; c < cand_j.size(); ++c) {
        const f64 aj  = cand_aj[c];
        const f64 mag = std::fabs(aj);
        if (mag <= pivot_tol) continue;
        const f64 d  = cand_d[c];
        const f64 sa = srow * aj;
        f64 te = kInf;
        int dir = 0;
        switch (st[sz(cand_j[c])]) {
            case NonbasicStatus::AtLower:
                te = d / (-sa);
                dir = +1;
                break;
            case NonbasicStatus::AtUpper:
                te = (-d) / sa;
                dir = -1;
                break;
            case NonbasicStatus::AtZeroFree:
                te = 0.0;
                dir = (sa < 0.0) ? +1 : -1;
                break;
            default:
                break;
        }
        if (dir == 0) continue;
        if (!(te < kInf) || te > harris_theta + 1e-20) continue;
        if (mag > best_piv) {
            best_piv      = mag;
            out.pivot     = cand_j[c];
            out.pivot_dir = dir;
            out.d_enter   = d;
        }
    }
    out.ok = out.pivot >= 0;
    return out;
}

// Bound-flipping ratio test (Koberstein 2008 §5.3.5; Huangfu & Hall 2015).
//
// Semantics: candidates are the eligible nonbasics of the pivotal row, in
// |d_j| / |a_rj| ratio order. Stepping the duals by theta passes every
// candidate whose ratio is below theta:
//   * a BOXED candidate is passed by FLIPPING it (its reduced cost changes
//     sign, and it has another bound to sit on) -- always legal;
//   * a one-sided/free candidate can only be passed within the Harris slack,
//     so it CAPS the step at its slack-adjusted ratio.
// The sweep keeps flipping while the accumulated change at the leaving row is
// smaller than |delta_primal| (the distance the leaving variable must travel):
// every unit of extra theta improves the dual objective, and each flip moves
// the leaving row toward its target for free. When the accumulation covers
// |delta_primal| the sweep stops; the batch where it stopped supplies the
// pivot (largest |a_rj|), and everything strictly before it flips.
DualBfrtResult dual_bfrt_choose(const std::vector<Index>& cand_j,
                                const std::vector<f64>& cand_aj,
                                const std::vector<f64>& cand_d,
                                const std::vector<NonbasicStatus>& st,
                                const std::vector<f64>& lo,
                                const std::vector<f64>& hi,
                                const f64 delta_primal, const f64 srow,
                                const f64 harris_theta,
                                const f64 pivot_tol, const f64 dual_tol,
                                const f64 dual_slack,
                                DualBfrtWorkspace* workspace) {
    DualBfrtResult out;
    if (!(harris_theta > 0.0) || cand_j.empty()) return out;

    const f64 total_delta = std::fabs(delta_primal);

    // Step cap from the candidates that cannot flip. Their slack-adjusted
    // ratio te + slack/|a| is the furthest the dual step may reach without
    // manufacturing dual infeasibility. (harris_theta is the same bound taken
    // over ALL candidates; restricting to non-flippables is what licenses the
    // long step past the boxed ones.)
    f64 theta_cap = kInfTheta;
    DualBfrtWorkspace local;
    DualBfrtWorkspace& ws = workspace ? *workspace : local;
    ws.order.clear();
    ws.column.clear();
    ws.alpha.clear();
    ws.dual.clear();
    ws.range.clear();
    ws.ratio.clear();
    ws.order.reserve(cand_j.size());
    ws.column.reserve(cand_j.size());
    ws.alpha.reserve(cand_j.size());
    ws.dual.reserve(cand_j.size());
    ws.range.reserve(cand_j.size());
    ws.ratio.reserve(cand_j.size());
    for (std::size_t c = 0; c < cand_j.size(); ++c) {
        const Index j = cand_j[c];
        const f64 a_rj = cand_aj[c];
        const f64 alpha = std::fabs(a_rj);
        if (alpha <= pivot_tol) continue;
        const bool boxed =
            lo[sz(j)] > -kInf && hi[sz(j)] < kInf && lo[sz(j)] != hi[sz(j)];
        const f64 te = std::fabs(cand_d[c]) / alpha;
        if (!boxed) {
            const f64 tb = te + dual_slack / alpha;
            if (tb < theta_cap) theta_cap = tb;
            continue;
        }
        ws.order.push_back(static_cast<Index>(ws.order.size()));
        ws.column.push_back(j);
        ws.alpha.push_back(alpha);
        ws.dual.push_back(cand_d[c]);
        ws.range.push_back(hi[sz(j)] - lo[sz(j)]);
        ws.ratio.push_back(te);
    }
    if (ws.order.empty()) return out;

    // Ratio order. Quad-sort (HiGHS) avoids the full sort; the sort is not the
    // bottleneck at these sizes and correctness beats the constant here.
    std::sort(ws.order.begin(), ws.order.end(),
              [&](Index a, Index b) { return ws.ratio[sz(a)] < ws.ratio[sz(b)]; });

    // Sweep. Batches are runs of candidates reached by the current theta;
    // theta advances to the next candidate's ratio, never past theta_cap.
    f64 theta = harris_theta;
    std::size_t flip_end = 0;    // [0, flip_end) will be flipped
    std::size_t batch_start = 0; // start of the batch the pivot comes from
    f64 acc = 0.0;
    bool crossed = false;
    while (flip_end < ws.order.size()) {
        std::size_t nxt = flip_end;
        while (nxt < ws.order.size()) {
            const Index k = ws.order[nxt];
            if (ws.ratio[sz(k)] > theta + dual_tol / ws.alpha[sz(k)]) break;
            ++nxt;
        }
        if (nxt == flip_end) {
            // Nothing at the current theta: extend to the next candidate.
            const Index k = ws.order[flip_end];
            if (ws.ratio[sz(k)] > theta_cap) break;
            theta = ws.ratio[sz(k)];
            continue;
        }
        batch_start = flip_end;
        for (std::size_t k = flip_end; k < nxt; ++k)
            acc += ws.alpha[sz(ws.order[k])] * ws.range[sz(ws.order[k])];
        flip_end = nxt;
        if (acc >= total_delta) { crossed = true; break; }
        if (flip_end == ws.order.size()) break;
        const f64 next_te = ws.ratio[sz(ws.order[flip_end])];
        if (next_te > theta_cap) break;
        theta = next_te;
    }
    if (flip_end == 0) return out;   // nobody to pivot on: legacy decides
    // If the sweep stopped without crossing (cap or exhaustion), the last
    // batch is the pivot batch and everything before it still flips.
    (void)crossed;

    // Pivot: largest |a_rj| inside the pivot batch [batch_start, flip_end).
    std::size_t best = batch_start;
    for (std::size_t k = batch_start + 1; k < flip_end; ++k)
        if (ws.alpha[sz(ws.order[k])] > ws.alpha[sz(ws.order[best])]) best = k;
    const Index piv_k = ws.order[best];

    out.pivot     = ws.column[sz(piv_k)];
    out.d_enter   = ws.dual[sz(piv_k)];
    out.pivot_dir = 0;
    for (std::size_t c = 0; c < cand_j.size(); ++c) {
        if (cand_j[c] != out.pivot) continue;
        switch (st[sz(out.pivot)]) {
            case NonbasicStatus::AtLower: out.pivot_dir = +1; break;
            case NonbasicStatus::AtUpper: out.pivot_dir = -1; break;
            case NonbasicStatus::AtZeroFree:
                out.pivot_dir = (srow * cand_aj[c] < 0.0) ? +1 : -1;
                break;
            default: break;
        }
        break;
    }
    if (out.pivot_dir == 0) return out;

    // The dual step is SIGNED and matches the legacy engine's formula,
    // theta_D = d_q / a_rq; its sign follows from the leaving direction.
    f64 a_rq = 0.0;
    for (std::size_t c = 0; c < cand_j.size(); ++c)
        if (cand_j[c] == out.pivot) { a_rq = cand_aj[c]; break; }
    if (std::fabs(a_rq) <= pivot_tol) return out;
    out.theta_dual = (ws.dual[sz(piv_k)] != 0.0) ? ws.dual[sz(piv_k)] / a_rq : 0.0;
    if (out.theta_dual == 0.0) { out.ok = true; return out; }

    for (std::size_t k = 0; k < batch_start; ++k)
        out.flips.push_back(ws.column[sz(ws.order[k])]);

    out.ok = true;
    return out;
}

}  // namespace sor::engines
