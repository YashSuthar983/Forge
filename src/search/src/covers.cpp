#include "sor/search/covers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

// One binary of the knapsack relaxation. `complemented` records that the item
// stands for 1 - x_j rather than x_j, which is how a negative coefficient is
// turned into a positive weight.
struct Item {
    Index col = -1;
    f64 weight = 0.0;      // > 0
    f64 value = 0.0;       // the LP value of the item's literal, in [0,1]
    bool complemented = false;
    int alpha = 0;         // cut coefficient, filled during lifting
};

inline bool is_binary(const model::LpProblem& lp, const std::vector<f64>& lo,
                      const std::vector<f64>& hi, Index j) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)] &&
           std::fabs(lo[sz(j)]) <= 1e-9 && std::fabs(hi[sz(j)] - 1.0) <= 1e-9;
}

// Builds the knapsack relaxation `sum w_k z_k <= cap` of one row, in the
// direction `sign` (+1 uses row_hi, -1 uses -row_lo).
//
// Non-binary terms are moved to the right-hand side at their MINIMUM possible
// contribution, which is a relaxation of the row and therefore keeps every
// inequality derived downstream valid. A term whose minimum is unbounded
// defeats that and the row is abandoned.
bool build_knapsack(const model::LpProblem& lp, Index row, f64 sign,
                    const std::vector<f64>& x, const std::vector<f64>& lo,
                    const std::vector<f64>& hi, const CoverOptions& opts,
                    std::vector<Item>& items, f64& cap,
                    CoverDiagnostics& diag) {
    const f64 bound = sign > 0.0 ? lp.row_hi[sz(row)] : -lp.row_lo[sz(row)];
    if (!std::isfinite(bound)) return false;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    items.clear();
    cap = bound;
    for (core::Offset k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = sign * av[sz(k)];
        if (std::fabs(a) <= opts.tol) continue;

        if (is_binary(lp, lo, hi, j)) {
            Item it;
            it.col = j;
            if (a > 0.0) {
                it.weight = a;
                it.complemented = false;
                it.value = x[sz(j)];
            } else {
                // a*x_j = a*(1 - z) = a - a*z, so the item weighs -a > 0 and
                // the constant a moves to the right-hand side.
                it.weight = -a;
                it.complemented = true;
                it.value = 1.0 - x[sz(j)];
                cap -= a;
            }
            it.value = std::min(1.0, std::max(0.0, it.value));
            items.push_back(it);
            continue;
        }

        // Non-binary: subtract the smallest value the term can take.
        const f64 at_min = a > 0.0 ? lo[sz(j)] : hi[sz(j)];
        if (!std::isfinite(at_min)) {
            ++diag.rejected_unbounded_term;
            return false;
        }
        cap -= a * at_min;
    }

    if (items.size() < 2) return false;
    // A negative capacity means the relaxed row is already infeasible, which is
    // propagation's business rather than a cut's. A capacity that covers every
    // item admits no cover at all.
    if (cap < -opts.tol) return false;
    f64 total = 0.0;
    for (const auto& it : items) total += it.weight;
    if (total <= cap + opts.tol) return false;
    ++diag.knapsacks_built;
    return true;
}

// Greedy separation (Gu-Nemhauser-Savelsbergh): the cover inequality over C is
// violated exactly when sum_{j in C} (1 - z*_j) < 1, so the search is for a
// cover minimising that sum. Taking items in increasing (1 - z*) order is the
// standard heuristic; the second pass makes the cover MINIMAL, which is what
// makes lifting meaningful.
bool find_violated_cover(const std::vector<Item>& items, f64 cap,
                         const CoverOptions& opts, std::vector<std::size_t>& cover,
                         f64& violation) {
    std::vector<std::size_t> order(items.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const f64 ca = 1.0 - items[a].value, cb = 1.0 - items[b].value;
        if (ca != cb) return ca < cb;
        return items[a].weight > items[b].weight;  // heavier first breaks ties
    });

    cover.clear();
    f64 w = 0.0;
    for (const std::size_t idx : order) {
        cover.push_back(idx);
        w += items[idx].weight;
        if (w > cap + opts.tol) break;
    }
    if (w <= cap + opts.tol) return false;  // never became a cover

    // Minimalise: drop the costliest members while the set stays a cover.
    std::sort(cover.begin(), cover.end(), [&](std::size_t a, std::size_t b) {
        return (1.0 - items[a].value) > (1.0 - items[b].value);
    });
    std::vector<std::size_t> kept;
    for (std::size_t p = 0; p < cover.size(); ++p) {
        const f64 without = w - items[cover[p]].weight;
        if (without > cap + opts.tol) {
            w = without;  // still a cover without it
        } else {
            kept.push_back(cover[p]);
        }
    }
    cover.swap(kept);
    if (cover.size() < 2) return false;

    f64 slack = 0.0;
    for (const std::size_t idx : cover) slack += 1.0 - items[idx].value;
    violation = 1.0 - slack;
    return violation > opts.violation_min;
}

}  // namespace

std::vector<CutRow> separate_lifted_covers(const model::LpProblem& lp,
                                           const std::vector<f64>& x,
                                           const std::vector<f64>& col_lo,
                                           const std::vector<f64>& col_hi,
                                           const CoverOptions& opts,
                                           CoverDiagnostics& diag) {
    std::vector<CutRow> cuts;
    const Index n = lp.n_cols();
    if (!opts.enabled || static_cast<Index>(x.size()) != n ||
        static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return cuts;

    const auto& rp = lp.A.pattern.row_ptr();
    std::vector<Item> items;
    std::vector<std::size_t> cover;
    std::vector<f64> minw;

    for (Index i = 0; i < lp.n_rows() && static_cast<int>(cuts.size()) < opts.max_cuts;
         ++i) {
        if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
        ++diag.rows_scanned;

        for (const f64 sign : {1.0, -1.0}) {
            if (static_cast<int>(cuts.size()) >= opts.max_cuts) break;
            f64 cap = 0.0;
            if (!build_knapsack(lp, i, sign, x, col_lo, col_hi, opts, items, cap,
                                diag))
                continue;

            f64 violation = 0.0;
            if (!find_violated_cover(items, cap, opts, cover, violation)) continue;
            ++diag.covers_found;

            // ---- sequential up-lifting -------------------------------------
            // Start from the cover inequality sum_{j in C} z_j <= |C| - 1 and
            // lift the variables outside C one at a time. The coefficient for j
            // is beta - max{ sum alpha_k z_k : sum w_k z_k <= cap - w_j } over
            // the variables already in the cut, which is the largest value that
            // keeps the inequality valid when z_j is switched on.
            for (auto& it : items) it.alpha = 0;
            for (const std::size_t idx : cover) items[idx].alpha = 1;
            int beta = static_cast<int>(cover.size()) - 1;

            const bool can_lift = items.size() <= opts.max_lift_items;
            if (can_lift && beta > 0) {
                // minw[v] = least knapsack weight achieving cut value exactly v
                // over the variables currently in the cut. Indexing the DP by
                // VALUE rather than weight is what makes it exact with real
                // weights.
                int vmax = beta;
                minw.assign(sz(static_cast<Index>(vmax) + 1), kInf);
                minw[0] = 0.0;
                for (const std::size_t idx : cover) {
                    for (int v = vmax; v >= 1; --v)
                        if (std::isfinite(minw[sz(static_cast<Index>(v) - 1)]))
                            minw[sz(static_cast<Index>(v))] =
                                std::min(minw[sz(static_cast<Index>(v))],
                                         minw[sz(static_cast<Index>(v) - 1)] +
                                             items[idx].weight);
                }

                // Lift the heaviest LP values first: those are the variables
                // whose coefficients most affect whether the cut bites here.
                std::vector<std::size_t> outside;
                for (std::size_t k = 0; k < items.size(); ++k)
                    if (items[k].alpha == 0) outside.push_back(k);
                std::sort(outside.begin(), outside.end(),
                          [&](std::size_t a, std::size_t b) {
                              return items[a].value > items[b].value;
                          });

                for (const std::size_t k : outside) {
                    if (vmax >= opts.lift_value_cap) break;
                    const f64 room = cap - items[k].weight;
                    int alpha;
                    if (room < -opts.tol) {
                        // z_k = 1 alone overruns the knapsack, so no feasible
                        // point has it set; any coefficient is valid and beta
                        // is the strongest finite one.
                        alpha = beta;
                    } else {
                        int best_v = 0;
                        for (int v = vmax; v >= 0; --v)
                            if (minw[sz(static_cast<Index>(v))] <= room + opts.tol) {
                                best_v = v;
                                break;
                            }
                        alpha = beta - best_v;
                    }
                    if (alpha <= 0) continue;
                    items[k].alpha = alpha;
                    ++diag.lifted_coefficients;

                    // Fold the new variable into the DP so later lifts account
                    // for it -- this is what makes the lifting SEQUENTIAL and
                    // therefore valid, rather than an independent guess.
                    const int new_vmax =
                        std::min(vmax + alpha, opts.lift_value_cap);
                    minw.resize(sz(static_cast<Index>(new_vmax) + 1), kInf);
                    for (int v = new_vmax; v >= alpha; --v)
                        if (std::isfinite(minw[sz(static_cast<Index>(v - alpha))]))
                            minw[sz(static_cast<Index>(v))] =
                                std::min(minw[sz(static_cast<Index>(v))],
                                         minw[sz(static_cast<Index>(v - alpha))] +
                                             items[k].weight);
                    vmax = new_vmax;
                }
            }

            // ---- back to the original variables ----------------------------
            // A complemented item stands for 1 - x_j, so alpha*(1 - x_j)
            // contributes -alpha to the column and alpha to the right-hand side.
            CutRow cut;
            f64 rhs = static_cast<f64>(beta);
            f64 lhs_at_x = 0.0;
            for (const auto& it : items) {
                if (it.alpha == 0) continue;
                const f64 a = static_cast<f64>(it.alpha);
                if (it.complemented) {
                    cut.cols.push_back(it.col);
                    cut.vals.push_back(-a);
                    rhs -= a;
                } else {
                    cut.cols.push_back(it.col);
                    cut.vals.push_back(a);
                }
                lhs_at_x += a * it.value;
            }
            if (cut.cols.size() < 2) continue;
            // Lifting can only raise the left-hand side at the current point,
            // so the cut should still bite -- but check rather than assume.
            if (lhs_at_x <= static_cast<f64>(beta) + opts.violation_min) {
                ++diag.rejected_not_violated;
                continue;
            }
            cut.row_lo = -kInf;
            cut.row_hi = rhs;
            cut.name = "COV_" + std::to_string(cuts.size());
            cuts.push_back(std::move(cut));
            ++diag.cuts_emitted;
        }
    }
    return cuts;
}

}  // namespace sor::search
