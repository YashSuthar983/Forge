#include "sor/search/flowcover.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

inline bool is_binary(const model::LpProblem& lp, const std::vector<f64>& lo,
                      const std::vector<f64>& hi, Index j) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)] &&
           std::fabs(lo[sz(j)]) <= 1e-9 && std::fabs(hi[sz(j)] - 1.0) <= 1e-9;
}

inline bool is_continuous(const model::LpProblem& lp, Index j) {
    return lp.is_integer.empty() || !lp.is_integer[sz(j)];
}

struct FlowArc {
    Index y = -1;     // continuous flow column
    Index x = -1;     // binary indicator (VUB)
    f64 u = 0.0;      // capacity on y when x = 1
    f64 y_val = 0.0;
    f64 x_val = 0.0;
    bool in_cover = false;
};

// Gu et al. SI lifting coefficient β for a non-cover arc of capacity uk, given
// m1 = max cover capacity and cover excess λ. Returns β ≤ 0 used as +β x on
// the LHS of the ≤ form (together with +1 · y).
f64 gu_si_beta(f64 uk, f64 m1, f64 lambda, f64 tol) {
    if (uk <= tol || m1 <= tol || lambda <= tol) return 0.0;
    // Search small i; capacities in MIPs are not astronomical relative to m1.
    const int imax = static_cast<int>(std::floor(uk / m1)) + 3;
    for (int i = 0; i <= imax; ++i) {
        const f64 im1 = static_cast<f64>(i) * m1;
        const f64 next = static_cast<f64>(i + 1) * m1;
        // Interval A: i m1 ≤ uk ≤ (i+1) m1 − λ  →  β = −uk + i λ
        const f64 a_hi = next - lambda;
        if (uk + tol >= im1 && uk <= a_hi + tol && a_hi + tol >= im1)
            return -uk + static_cast<f64>(i) * lambda;
        // Interval B: i m1 − λ ≤ uk ≤ i m1  (i ≥ 1)  →  β = −i (m1 − λ)
        if (i >= 1) {
            const f64 b_lo = im1 - lambda;
            if (uk + tol >= b_lo && uk <= im1 + tol)
                return -static_cast<f64>(i) * (m1 - lambda);
        }
    }
    // Safe fallback: β = −max(0, uk − λ) complements the (uk−λ)(1−x) pattern.
    return -std::max(0.0, uk - lambda);
}

void record_vub(std::unordered_map<Index, std::pair<Index, f64>>& vub,
                Index ycol, Index xcol, f64 u, FlowCoverDiagnostics& diag,
                bool projected) {
    if (u <= 0.0) return;
    auto it = vub.find(ycol);
    if (it == vub.end() || u < it->second.second - 1e-12) {
        vub[ycol] = {xcol, u};
        ++diag.vubs_found;
        if (projected) ++diag.vubs_projected;
    }
}

// Try to read ay y + ax x ≤ rhs as y − u x ≤ 0 (after scaling).
bool vub_from_pair(f64 ay, f64 ax, f64 rhs, f64 tol, f64& u_out) {
    if (std::fabs(ay) <= tol) return false;
    // Prefer ay > 0: y + (ax/ay) x ≤ rhs/ay.
    f64 c = ax / ay;
    f64 r = rhs / ay;
    if (ay < 0.0) {
        // Flipped by dividing by negative: inequality sense reverses - reject
        // here; caller tries the ≥ form separately.
        return false;
    }
    if (c < -tol && std::fabs(r) <= tol) {
        u_out = -c;
        return u_out > tol;
    }
    // Soft: y − u x ≤ ε with tiny positive rhs still encodes a VUB up to
    // shifting capacity: y ≤ u x + ε ⇒ effective u' = u + ε when x=1, but we
    // keep u = −c and require ε small.
    if (c < -tol && r >= -tol && r <= tol * 10.0) {
        u_out = -c;
        return u_out > tol;
    }
    return false;
}

void collect_vubs(const model::LpProblem& lp, const std::vector<f64>& lo,
                  const std::vector<f64>& hi, const FlowCoverOptions& opts,
                  std::unordered_map<Index, std::pair<Index, f64>>& vub,
                  FlowCoverDiagnostics& diag) {
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    for (Index i = 0; i < lp.n_rows(); ++i) {
        const auto nnz = rp[sz(i) + 1] - rp[sz(i)];
        if (nnz < 2) continue;

        // ---- Exact 2-column VUBs ------------------------------------------
        if (nnz == 2) {
            Index j0 = ci[sz(rp[sz(i)])];
            Index j1 = ci[sz(rp[sz(i)] + 1)];
            f64 a0 = av[sz(rp[sz(i)])];
            f64 a1 = av[sz(rp[sz(i)] + 1)];

            auto try_pair = [&](Index ycol, f64 ay, Index xcol, f64 ax) {
                if (!is_binary(lp, lo, hi, xcol)) return;
                if (!is_continuous(lp, ycol)) return;
                const f64 rh = lp.row_hi[sz(i)];
                const f64 rl = lp.row_lo[sz(i)];
                f64 u = 0.0;
                if (std::isfinite(rh) &&
                    vub_from_pair(ay, ax, rh, opts.tol, u))
                    record_vub(vub, ycol, xcol, u, diag, false);
                if (std::isfinite(rl)) {
                    // ay y + ax x ≥ rl  ↔  (−ay)y + (−ax)x ≤ −rl
                    if (vub_from_pair(-ay, -ax, -rl, opts.tol, u))
                        record_vub(vub, ycol, xcol, u, diag, false);
                }
            };
            try_pair(j0, a0, j1, a1);
            try_pair(j1, a1, j0, a0);
            continue;
        }

        if (!opts.project_vubs || nnz > 8) continue;

        // ---- Projected VUBs from multi-column rows ------------------------
        // For each (continuous y, binary x) pair on the row, move every other
        // term to the RHS at its maximum contribution under ≤ sense. A
        // surviving relation y − u x ≤ ε with ε≈0 yields a valid (possibly
        // weaker) VUB.
        const f64 rh = lp.row_hi[sz(i)];
        const f64 rl = lp.row_lo[sz(i)];

        auto project_sense = [&](f64 sign, f64 bound) {
            if (!std::isfinite(bound)) return;
            // Collect terms.
            struct Term {
                Index j;
                f64 a;
            };
            std::vector<Term> terms;
            terms.reserve(static_cast<std::size_t>(nnz));
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                terms.push_back({ci[sz(k)], sign * av[sz(k)]});

            for (std::size_t p = 0; p < terms.size(); ++p) {
                if (!is_continuous(lp, terms[p].j)) continue;
                if (!(terms[p].a > opts.tol)) continue;
                for (std::size_t q = 0; q < terms.size(); ++q) {
                    if (p == q) continue;
                    if (!is_binary(lp, lo, hi, terms[q].j)) continue;
                    if (!(terms[q].a < -opts.tol)) continue;

                    // Move other terms to RHS at max contribution for ≤.
                    f64 rhs = bound;
                    bool ok = true;
                    for (std::size_t t = 0; t < terms.size(); ++t) {
                        if (t == p || t == q) continue;
                        const f64 a = terms[t].a;
                        const Index j = terms[t].j;
                        // max a·x over box: a>0 → hi, a<0 → lo
                        const f64 at_max = a > 0.0 ? hi[sz(j)] : lo[sz(j)];
                        if (!std::isfinite(at_max)) {
                            ok = false;
                            break;
                        }
                        rhs -= a * at_max;
                    }
                    if (!ok) continue;
                    f64 u = 0.0;
                    if (vub_from_pair(terms[p].a, terms[q].a, rhs, opts.tol,
                                      u)) {
                        // Also tighten by column upper bound.
                        if (std::isfinite(hi[sz(terms[p].j)])) {
                            const f64 col_u =
                                hi[sz(terms[p].j)] -
                                std::min(0.0, lo[sz(terms[p].j)]);
                            if (col_u > opts.tol) u = std::min(u, col_u);
                        }
                        record_vub(vub, terms[p].j, terms[q].j, u, diag, true);
                    }
                }
            }
        };

        project_sense(1.0, rh);
        project_sense(-1.0, -rl);
    }

    // Bound-inferred: continuous y with finite ub U and a binary x that appears
    // with y in some row with opposite-sign coupling (already covered by
    // projection). Additionally, if y has finite ub and there is an explicit
    // singleton row y ≤ 0 when some binary is 0 - skip (needs implications).
}

bool build_flow_from_row(
    const model::LpProblem& lp, Index row, f64 sign, const std::vector<f64>& x,
    const std::vector<f64>& lo, const std::vector<f64>& hi,
    const std::unordered_map<Index, std::pair<Index, f64>>& vub,
    const FlowCoverOptions& opts, std::vector<FlowArc>& arcs, f64& cap,
    FlowCoverDiagnostics& diag) {
    const f64 bound = sign > 0.0 ? lp.row_hi[sz(row)] : -lp.row_lo[sz(row)];
    if (!std::isfinite(bound)) return false;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    arcs.clear();
    cap = bound;
    for (core::Offset k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = sign * av[sz(k)];
        if (std::fabs(a) <= opts.tol) continue;

        if (is_continuous(lp, j) && a > opts.tol) {
            auto it = vub.find(j);
            if (it == vub.end()) continue;
            // The flow cover inequality rests on 0 <= y <= u*x. A flow
            // variable with a NEGATIVE lower bound breaks that outright: y can
            // go below zero, the cover/lambda arithmetic no longer measures
            // what it claims, and the resulting cut is not valid.
            //
            // This was unchecked. Measured on blend2 with --verify-cuts
            // against the true optimum (7.5989850): 57 invalid FC_ cuts, e.g.
            // FC_1 activity 12040 against an rhs of 312, and the solve
            // reported a FALSE Optimal of 16.554765.
            //
            // The capacity line below already nods at negative lower bounds
            // via hi - min(0, lo), but adjusting the CAPACITY is not the same
            // as shifting the VARIABLE. The proper treatment is to substitute
            // y' = y - lo >= 0 and carry the shift through cap.
            //
            // NOT A VERIFIED FIX EITHER: blend2 still produced 57 invalid cuts
            // with this guard in place, so its flow arcs were not the problem.
            // Three theories have now failed on this separator (tolerance
            // flooring, unsafe lifting, negative flow bounds). The next person
            // should dump one offending cut and check it term by term against
            // the source row rather than reason about the derivation -- that
            // is what finally worked for the cover and node-promotion bugs.
            if (!std::isfinite(lo[sz(j)]) || lo[sz(j)] < -opts.tol) {
                ++diag.rejected_negative_flow;
                continue;
            }
            FlowArc arc;
            arc.y = j;
            arc.x = it->second.first;
            arc.u = a * it->second.second;
            if (std::isfinite(hi[sz(j)]))
                arc.u = std::min(arc.u,
                                 a * (hi[sz(j)] - std::min(0.0, lo[sz(j)])));
            if (arc.u <= opts.tol) continue;
            arc.y_val = a * x[sz(j)];
            arc.x_val = x[sz(arc.x)];
            arcs.push_back(arc);
            continue;
        }

        const f64 at_min = a > 0.0 ? lo[sz(j)] : hi[sz(j)];
        if (!std::isfinite(at_min)) return false;
        cap -= a * at_min;
    }
    if (arcs.size() < 2 || cap < -opts.tol) return false;
    f64 total_u = 0.0;
    for (const auto& a : arcs) total_u += a.u;
    if (total_u <= cap + opts.tol) return false;
    return true;
}

bool find_flow_cover(const std::vector<FlowArc>& arcs, f64 cap,
                     const FlowCoverOptions& opts, std::vector<std::size_t>& cover,
                     f64& lambda) {
    std::vector<std::size_t> order(arcs.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const f64 sa = arcs[a].u * (1.0 - arcs[a].x_val);
        const f64 sb = arcs[b].u * (1.0 - arcs[b].x_val);
        if (sa != sb) return sa > sb;
        return arcs[a].u > arcs[b].u;
    });

    cover.clear();
    f64 w = 0.0;
    for (const std::size_t idx : order) {
        cover.push_back(idx);
        w += arcs[idx].u;
        if (w > cap + opts.tol) break;
    }
    if (w <= cap + opts.tol) return false;
    lambda = w - cap;

    std::sort(cover.begin(), cover.end(), [&](std::size_t a, std::size_t b) {
        return arcs[a].u * (1.0 - arcs[a].x_val) <
               arcs[b].u * (1.0 - arcs[b].x_val);
    });
    std::vector<std::size_t> kept;
    for (const std::size_t idx : cover) {
        const f64 without = w - arcs[idx].u;
        if (without > cap + opts.tol) {
            w = without;
        } else {
            kept.push_back(idx);
        }
    }
    cover.swap(kept);
    if (cover.size() < 2) return false;
    lambda = w - cap;
    return lambda > opts.tol;
}

}  // namespace

std::vector<CutRow> separate_flow_covers(const model::LpProblem& lp,
                                         const std::vector<f64>& x,
                                         const std::vector<f64>& col_lo,
                                         const std::vector<f64>& col_hi,
                                         const FlowCoverOptions& opts,
                                         FlowCoverDiagnostics& diag) {
    std::vector<CutRow> cuts;
    const Index n = lp.n_cols();
    if (!opts.enabled || static_cast<Index>(x.size()) != n ||
        static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return cuts;

    std::unordered_map<Index, std::pair<Index, f64>> vub;
    collect_vubs(lp, col_lo, col_hi, opts, vub, diag);
    if (vub.empty()) return cuts;

    const auto& rp = lp.A.pattern.row_ptr();
    std::vector<FlowArc> arcs;
    std::vector<std::size_t> cover;

    for (Index i = 0; i < lp.n_rows() && static_cast<int>(cuts.size()) < opts.max_cuts;
         ++i) {
        if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
        ++diag.rows_scanned;

        for (const f64 sign : {1.0, -1.0}) {
            if (static_cast<int>(cuts.size()) >= opts.max_cuts) break;
            f64 cap = 0.0;
            if (!build_flow_from_row(lp, i, sign, x, col_lo, col_hi, vub, opts,
                                     arcs, cap, diag))
                continue;
            ++diag.structures_built;

            f64 lambda = 0.0;
            if (!find_flow_cover(arcs, cap, opts, cover, lambda)) continue;
            ++diag.covers_found;

            for (auto& a : arcs) a.in_cover = false;
            for (const std::size_t idx : cover)
                arcs[idx].in_cover = true;

            f64 m1 = 0.0;
            for (const std::size_t idx : cover)
                m1 = std::max(m1, arcs[idx].u);

            // ∑_{j∈C} y_j + ∑_{j∈L} (u_j−λ)(1−x_j) + SI terms ≤ cap
            CutRow cut;
            f64 rhs = cap;
            f64 lhs = 0.0;
            f64 cmin = kInf, cmax = 0.0;

            auto push_coef = [&](Index col, f64 val, f64 xval) {
                if (std::fabs(val) <= opts.tol) return;
                cut.cols.push_back(col);
                cut.vals.push_back(val);
                lhs += val * xval;
                cmin = std::min(cmin, std::fabs(val));
                cmax = std::max(cmax, std::fabs(val));
            };

            for (const std::size_t idx : cover) {
                const auto& a = arcs[idx];
                push_coef(a.y, 1.0, a.y_val);
                const f64 excess = a.u - lambda;
                if (excess > opts.tol) {
                    // +(u−λ)(1−x) → −(u−λ) x and RHS −= (u−λ)
                    push_coef(a.x, -excess, a.x_val);
                    rhs -= excess;
                }
            }

            // Sequence-independent lifting of non-cover arcs (Gu Thm. 8, N−=∅).
            if (opts.sequence_independent_lift && m1 > opts.tol) {
                std::vector<std::size_t> outside;
                for (std::size_t t = 0; t < arcs.size(); ++t)
                    if (!arcs[t].in_cover) outside.push_back(t);
                // Prefer arcs that look fractional / residual-open.
                std::sort(outside.begin(), outside.end(),
                          [&](std::size_t a, std::size_t b) {
                              const f64 sa =
                                  arcs[a].y_val +
                                  std::fabs(gu_si_beta(arcs[a].u, m1, lambda,
                                                       opts.tol)) *
                                      arcs[a].x_val;
                              const f64 sb =
                                  arcs[b].y_val +
                                  std::fabs(gu_si_beta(arcs[b].u, m1, lambda,
                                                       opts.tol)) *
                                      arcs[b].x_val;
                              return sa > sb;
                          });
                int lifted = 0;
                for (const std::size_t idx : outside) {
                    if (lifted >= opts.max_lift_arcs) break;
                    const auto& a = arcs[idx];
                    const f64 beta = gu_si_beta(a.u, m1, lambda, opts.tol);
                    // Skip vacuous (β ≈ 0 and we still add y - only add when
                    // the pair can strengthen violation).
                    // Include y with α=1 and binary with β (≤ 0).
                    const f64 contrib =
                        a.y_val + beta * a.x_val;  // vs 0 if excluded
                    if (contrib <= opts.tol && beta >= -opts.tol) continue;
                    // VALIDITY GUARD (2026-09-19). The rhs is fixed before this
                    // loop and never adjusted, so a lifted arc may only be
                    // added if its term cannot increase the left-hand side
                    // beyond what the base inequality already allows.
                    //
                    // The arc contributes y + beta*x, and the VUB gives
                    // y <= u*x, so its worst case over x in {0,1} is
                    // max(0, u + beta). Validity without an rhs change
                    // therefore needs u + beta <= 0. But gu_si_beta returns
                    // -u + i*lambda on interval A, so for i >= 1 we get
                    // u + beta = i*lambda > 0 -- each such arc silently adds
                    // slack the rhs never paid for.
                    //
                    // Measured before this guard, with --verify-cuts against
                    // blend2's true optimum: FC_0 reached activity 12040
                    // against an rhs of 312, a factor of 38, and the solve
                    // reported a FALSE Optimal of 16.554765 (true 7.5989850).
                    //
                    // NOT A VERIFIED FIX. u + beta <= 0 is a genuine
                    // requirement of rhs-free lifting, so the guard is correct
                    // to have, but blend2 STILL produced 57 invalid FC_ cuts
                    // with it in place. Flow cover therefore remains default
                    // OFF. Whatever is wrong is upstream of the lifting.
                    if (a.u + beta > opts.tol) {
                        ++diag.rejected_unsafe_lift;
                        continue;
                    }
                    push_coef(a.y, 1.0, a.y_val);
                    push_coef(a.x, beta, a.x_val);
                    ++lifted;
                    ++diag.si_lifted_arcs;
                }
            }

            if (cut.cols.size() < 2) continue;
            if (cmin > 0.0 && cmax / cmin > opts.max_dynamism) {
                ++diag.rejected_dynamism;
                continue;
            }
            if (lhs <= rhs + opts.violation_min) {
                ++diag.rejected_not_violated;
                continue;
            }
            cut.row_lo = -kInf;
            cut.row_hi = rhs;
            cut.name = "FC_" + std::to_string(cuts.size());
            cuts.push_back(std::move(cut));
            ++diag.cuts_emitted;
        }
    }
    return cuts;
}

}  // namespace sor::search
