#include "forge/search/flowcover.hpp"
#include "forge/core/work_clock.hpp"
#include "forge/model/dyadic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace forge::search {
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
    f64 scale = 1.0;  // positive coefficient of y in the capacity row
    f64 u = 0.0;      // capacity on y when x = 1
    f64 y_val = 0.0;
    f64 x_val = 0.0;
    bool in_cover = false;
};

// Upward-rounded a * b (a, b finite).
f64 mul_up(f64 a, f64 b) {
    model::DyadicSum d;
    d.add_product(a, b);
    return d.up();
}

// Lifting function of the simple flow cover (C, lambda) for N- empty: with
// the cover arcs of capacity above lambda sorted u_1 >= ... >= u_r and
// M_d = u_1 + ... + u_d (M_0 = 0),
//     H(t) = max_{d = 0..r} min(lambda + t - d lambda, M_d - d lambda).
// Derivation: with a new arc carrying flow t the cover arcs keep b - t of
// the capacity; closing a set D of them gains U(D) - |D| lambda while it
// still fits (U(D) <= lambda + t) and t - (|D| - 1) lambda once it no longer
// does, and both are at most min(lambda + t, M_|D|) - |D| lambda. So an arc
// lifted as  y_k - g x_k  with g >= H(u_k) keeps the row valid, and H is
// subadditive (Gu, Nemhauser & Savelsbergh 1999, the case N- = empty), which
// lets every lifted arc use its own g at once. Exactly, rounded up.
f64 lifting_h_up(f64 t, const std::vector<f64>& caps_desc, const model::DyadicSum& lambda) {
    model::DyadicSum best;   // d = 0 gives min(lambda + t, 0) = 0
    model::DyadicSum M;      // M_d
    for (std::size_t d = 1; d <= caps_desc.size(); ++d) {
        M.add(caps_desc[d - 1]);
        model::DyadicSum a = lambda;                 // lambda + t - d lambda
        a.add(t);
        a.add_sum_product(lambda, -static_cast<f64>(d));
        model::DyadicSum b = M;                      // M_d - d lambda
        b.add_sum_product(lambda, -static_cast<f64>(d));
        model::DyadicSum diff = a;
        model::DyadicSum nb = b;
        nb.negate();
        diff.add_sum(nb);
        const model::DyadicSum& m = diff.sign() <= 0 ? a : b;   // min(a, b)
        model::DyadicSum gap = m;
        model::DyadicSum nbest = best;
        nbest.negate();
        gap.add_sum(nbest);
        if (gap.sign() > 0) best = m;
    }
    return best.up();
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

// Read ay y + ax x <= rhs (ay > 0, ax < 0, rhs <= 0, decided exactly) as the
// VUB y <= u x with u = -ax / ay rounded UP: a capacity may only be
// overstated. A positive residual, however small, is y <= u x + epsilon and
// cannot be discarded in a validity proof for a cut; a negative one only
// strengthens the row and is dropped.
bool vub_from_pair(f64 ay, f64 ax, const model::DyadicSum& rhs, f64 tol, f64& u_out) {
    if (!(ay > 0.0) || !(ax < 0.0)) return false;
    if (rhs.sign() > 0) return false;
    model::DyadicSum num;
    num.add(-ax);
    u_out = num.quotient_up(ay);
    return u_out > tol && std::isfinite(u_out);
}
bool vub_from_pair(f64 ay, f64 ax, f64 rhs, f64 tol, f64& u_out) {
    model::DyadicSum r;
    r.add(rhs);
    return vub_from_pair(ay, ax, r, tol, u_out);
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
        // term to the RHS at its minimum contribution under ≤ sense. A
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

                    // For a <= row, the largest possible y occurs when the
                    // omitted terms have their MINIMUM activity. Using their
                    // maximum would invent a stronger, invalid VUB. Summed
                    // exactly: the residual's sign decides validity.
                    model::DyadicSum rhs;
                    rhs.add(bound);
                    bool ok = true;
                    for (std::size_t t = 0; t < terms.size(); ++t) {
                        if (t == p || t == q) continue;
                        const f64 a = terms[t].a;
                        const Index j = terms[t].j;
                        const f64 at_min = a > 0.0 ? lo[sz(j)] : hi[sz(j)];
                        if (!std::isfinite(at_min)) {
                            ok = false;
                            break;
                        }
                        rhs.add_product(-a, at_min);
                    }
                    if (!ok) continue;
                    f64 u = 0.0;
                    if (vub_from_pair(terms[p].a, terms[q].a, rhs, opts.tol,
                                      u)) {
                        // y <= hi_y as well (y >= 0 is required of every arc
                        // later), so u may be reduced to hi_y.
                        if (std::isfinite(hi[sz(terms[p].j)]) && hi[sz(terms[p].j)] > opts.tol)
                            u = std::min(u, hi[sz(terms[p].j)]);
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

// The single-node flow set of a row: arcs a y (y >= 0 with a VUB y <= u x,
// capacity a u rounded up) and the capacity b = side minus every other
// term's minimum activity, summed exactly and rounded up. Overstating a
// capacity or b only relaxes the flow set, so every cut of the rounded set is
// valid for the row.
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
    model::DyadicSum cap_sum;
    cap_sum.add(bound);
    for (core::Offset k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = sign * av[sz(k)];
        if (std::fabs(a) <= opts.tol) continue;

        if (is_continuous(lp, j) && a > opts.tol) {
            auto it = vub.find(j);
            if (it != vub.end()) {
                // The inequality needs 0 <= y <= u*x. A negative lower
                // bound cannot be used as a flow arc without shifting y.
                // The historical blend2 run still had invalid cuts after
                // this guard; rejection alone was not a complete repair.
                if (!std::isfinite(lo[sz(j)]) || lo[sz(j)] < -opts.tol) {
                    ++diag.rejected_negative_flow;
                } else {
                    FlowArc arc;
                    arc.y = j;
                    arc.x = it->second.first;
                    arc.scale = a;
                    arc.u = mul_up(a, it->second.second);
                    if (std::isfinite(hi[sz(j)]))
                        arc.u = std::min(arc.u, mul_up(a, hi[sz(j)]));
                    if (arc.u > opts.tol) {
                        arc.y_val = a * x[sz(j)];
                        arc.x_val = x[sz(arc.x)];
                        arcs.push_back(arc);
                        continue;
                    }
                }
            }
        }

        // A term not used as a flow arc still affects capacity. In particular,
        // a negative lower bound on a positive continuous term increases the
        // available capacity. Dropping it produced invalid cuts.
        const f64 at_min = a > 0.0 ? lo[sz(j)] : hi[sz(j)];
        if (!std::isfinite(at_min)) return false;
        cap_sum.add_product(-a, at_min);
    }
    cap = cap_sum.up();
    if (arcs.size() < 2 || cap < 0.0) return false;
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
    core::WorkScope work_scope_(core::WorkCat::Separation);
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

            // lambda = sum_C u - b, exactly; the cover needs it positive.
            model::DyadicSum lambda_x;
            for (const std::size_t idx : cover) lambda_x.add(arcs[idx].u);
            lambda_x.add(-cap);
            if (lambda_x.sign() <= 0) continue;

            // sum_{j in C} a_j y_j + sum_{j in C+} e_j (1 - x_j) <= b with
            // e_j = u_j - lambda > 0 (exact sign), each e_j rounded DOWN (a
            // smaller multiplier of 1 - x_j >= 0 only weakens the row) and
            // the side b - sum e_j recomputed from the rounded values,
            // rounded UP. Terms on one column are merged, the merged x
            // coefficient rounded down. The row is then exact.
            std::vector<std::pair<Index, model::DyadicSum>> terms;   // col -> coefficient
            const auto add_term = [&](Index col, f64 v) {
                for (auto& [c, d] : terms)
                    if (c == col) { d.add(v); return; }
                model::DyadicSum d;
                d.add(v);
                terms.emplace_back(col, std::move(d));
            };
            model::DyadicSum side;
            side.add(cap);
            std::vector<f64> caps_plus;   // capacities of C+ (for lifting)
            for (const std::size_t idx : cover) {
                const auto& a = arcs[idx];
                add_term(a.y, a.scale);
                model::DyadicSum e;
                e.add(a.u);
                model::DyadicSum nl = lambda_x;
                nl.negate();
                e.add_sum(nl);
                if (e.sign() <= 0) continue;
                caps_plus.push_back(a.u);
                const f64 e_down = e.down();
                if (!(e_down > 0.0)) continue;
                add_term(a.x, -e_down);
                side.add(-e_down);
            }

            // Lifting of the arcs outside the cover: a_k y_k - g_k x_k with
            // g_k = H(u_k) rounded up (lifting_h_up). Taken only when the
            // term adds violation at x.
            if (opts.sequence_independent_lift && !caps_plus.empty()) {
                std::sort(caps_plus.begin(), caps_plus.end(), std::greater<f64>());
                std::vector<std::pair<f64, std::size_t>> outside;
                for (std::size_t t = 0; t < arcs.size(); ++t) {
                    if (arcs[t].in_cover) continue;
                    const f64 g = lifting_h_up(arcs[t].u, caps_plus, lambda_x);
                    const f64 gain = arcs[t].y_val - g * arcs[t].x_val;
                    if (gain > opts.tol) outside.emplace_back(gain, t);
                }
                std::sort(outside.begin(), outside.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                int lifted = 0;
                for (const auto& [gain, t] : outside) {
                    if (lifted >= opts.max_lift_arcs) break;
                    (void)gain;
                    const auto& a = arcs[t];
                    const f64 g = lifting_h_up(a.u, caps_plus, lambda_x);
                    add_term(a.y, a.scale);
                    if (g > 0.0) add_term(a.x, -g);
                    ++lifted;
                    ++diag.si_lifted_arcs;
                }
            }

            CutRow cut;
            f64 lhs = 0.0;
            f64 cmin = kInf, cmax = 0.0;
            for (const auto& [col, d] : terms) {
                // y coefficients are exact row data; a merged x coefficient
                // is rounded down (x >= 0: a smaller coefficient weakens).
                const f64 v = d.down();
                if (v == 0.0) continue;
                cut.cols.push_back(col);
                cut.vals.push_back(v);
                lhs += v * x[sz(col)];
                cmin = std::min(cmin, std::fabs(v));
                cmax = std::max(cmax, std::fabs(v));
            }
            const f64 rhs = side.up();
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
            cut.exact = true;
            cut.name = "FC_" + std::to_string(cuts.size());
            cuts.push_back(std::move(cut));
            ++diag.cuts_emitted;
        }
    }
    return cuts;
}

}  // namespace forge::search
