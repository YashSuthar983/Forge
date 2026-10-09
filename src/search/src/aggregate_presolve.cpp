#include "aggregate_presolve.hpp"

#include "forge/core/work_clock.hpp"
#include "forge/model/dyadic.hpp"
#include "forge/sparse/csr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace forge::search::detail {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

// (num / den) when it is exactly a double, the numerator given exactly.
std::optional<f64> exact_quotient(const model::DyadicSum& num, f64 den) {
    if (den == 0.0 || !std::isfinite(den)) return std::nullopt;
    if (num.is_zero()) return 0.0;
    const f64 lo = num.quotient_down(den), hi = num.quotient_up(den);
    if (lo != hi || !std::isfinite(lo)) return std::nullopt;
    return lo;
}

// (a * d - b * e) / d exactly: the rewrite of a coefficient a when a
// multiple b/d of the pivot row (entry e) is subtracted.
std::optional<f64> exact_rewrite(f64 a, f64 b, f64 e, f64 d) {
    model::DyadicSum num;
    num.add_product(a, d);
    num.add_product(-b, e);
    return exact_quotient(num, d);
}

bool integral(f64 v) { return std::isfinite(v) && std::fabs(v) < 0x1p52 && v == std::trunc(v); }

}  // namespace

MilpPresolveResult aggregate_equalities(const model::LpProblem& lp,
                                        const MilpPresolveOptions& opts,
                                        MilpPresolveStats& stats) {
    core::WorkScope work_scope_(core::WorkCat::Presolve);
    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    MilpPresolveResult out;
    out.reduced_col.assign(sz(n), -1);
    out.reduced_row.assign(sz(m), -1);
    out.eliminated.assign(sz(n), 0);
    out.fixed_value.assign(sz(n), 0.0);

    using Entry = std::pair<Index, f64>;
    std::vector<std::vector<Entry>> rows(sz(m));
    std::vector<std::vector<Index>> col_rows(sz(n));
    {
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        for (Index i = 0; i < m; ++i) {
            auto& row = rows[sz(i)];
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                if (lp.A.vals[sz(k)] != 0.0) row.emplace_back(ci[sz(k)], lp.A.vals[sz(k)]);
            std::sort(row.begin(), row.end(),
                      [](const Entry& a, const Entry& b) { return a.first < b.first; });
            for (const auto& e : row) col_rows[sz(e.first)].push_back(i);
        }
    }
    std::vector<f64> c = lp.c, lo = lp.col_lo, hi = lp.col_hi;
    std::vector<f64> row_lo = lp.row_lo, row_hi = lp.row_hi;
    std::vector<char> col_dead(sz(n), 0), row_dead(sz(m), 0);
    const auto is_int = [&](Index j) {
        return !lp.is_integer.empty() && lp.is_integer[sz(j)];
    };
    model::DyadicSum offset;
    offset.add(lp.obj_offset);

    const auto find = [&](Index i, Index j) -> f64* {
        auto& row = rows[sz(i)];
        auto it = std::lower_bound(row.begin(), row.end(), j,
                                   [](const Entry& e, Index v) { return e.first < v; });
        return (it != row.end() && it->first == j) ? &it->second : nullptr;
    };
    // Live rows holding column j (col_rows may name rows j has left).
    const auto live_rows_of = [&](Index j, std::vector<Index>& outv) {
        outv.clear();
        for (const Index i : col_rows[sz(j)])
            if (!row_dead[sz(i)] && find(i, j) != nullptr) outv.push_back(i);
        std::sort(outv.begin(), outv.end());
        outv.erase(std::unique(outv.begin(), outv.end()), outv.end());
    };

    const int max_row = std::max(2, opts.aggregation_max_row);
    const int max_col = std::max(1, opts.aggregation_max_col_rows);
    std::vector<Index> others;
    struct Rewrite {
        Index row;
        std::vector<Entry> entries;   // the row after substitution
        f64 new_lo, new_hi;
    };
    std::vector<Rewrite> rewrites;
    std::vector<std::pair<Index, f64>> cost_rewrites;
    std::uint64_t work = 0;

    for (int pass = 0; pass < 3; ++pass) {
        bool changed = false;
        for (Index r = 0; r < m; ++r) {
            if (core::cancel_requested(opts.cancel)) break;
            if (row_dead[sz(r)]) continue;
            const f64 b = row_lo[sz(r)];
            if (!(b == row_hi[sz(r)]) || !std::isfinite(b)) continue;
            const auto& prow = rows[sz(r)];
            const Index len = static_cast<Index>(prow.size());
            if (len < 2 || len > max_row) continue;
            work += static_cast<std::uint64_t>(len);
            f64 amax = 0.0;
            for (const auto& e : prow) amax = std::max(amax, std::fabs(e.second));
            // Integer pivots need the row's other data integral.
            bool row_integral = integral(b);
            for (const auto& e : prow)
                if (!is_int(e.first) || !integral(e.second)) row_integral = false;

            struct Choice {
                Index col = -1;
                f64 a = 0.0;
                bool drop_row = false;
                long delta_nnz = 0;
                f64 keep_lo = 0.0, keep_hi = 0.0;
            } best;
            bool have_best = false;
            for (const auto& [j, a] : prow) {
                if (col_dead[sz(j)] || !(lo[sz(j)] < hi[sz(j)])) continue;
                if (std::fabs(a) < 0.01 * amax) continue;
                if (is_int(j) && !(std::fabs(a) == 1.0 && row_integral)) continue;
                live_rows_of(j, others);
                const Index count = static_cast<Index>(others.size());
                if (count > max_col) continue;
                // Does the row imply x_j's bounds? a x_j = b - sum_{k != j} a_k x_k.
                // Lower bound of x_j (a > 0) is (b - max)/a; compare exactly.
                const auto implied_side = [&](bool lower_of_xj) -> bool {
                    const f64 bound = lower_of_xj ? lo[sz(j)] : hi[sz(j)];
                    if (!std::isfinite(bound)) return true;   // nothing to imply
                    // x_j >= bound  <=>  a x_j >= a bound (a > 0) or <= (a < 0).
                    // a x_j = b - S, so we need the extreme of S on the right side.
                    const bool need_max_s = (a > 0.0) == lower_of_xj;
                    model::DyadicSum d;
                    d.add(b);
                    for (const auto& [k, ak] : prow) {
                        if (k == j) continue;
                        const f64 v = ((ak > 0.0) == need_max_s) ? hi[sz(k)] : lo[sz(k)];
                        if (!std::isfinite(v)) return false;
                        d.add_product(-ak, v);
                    }
                    d.add_product(-a, bound);   // (b - S_ext) - a * bound
                    // lower_of_xj, a > 0: need b - S_max >= a lo  -> d >= 0
                    // lower_of_xj, a < 0: need b - S_min <= a lo  -> d <= 0
                    // upper (hi), a > 0:  need b - S_min <= a hi  -> d <= 0
                    // upper (hi), a < 0:  need b - S_max >= a hi  -> d >= 0
                    const bool want_nonneg = lower_of_xj == (a > 0.0);
                    return want_nonneg ? d.sign() >= 0 : d.sign() <= 0;
                };
                const bool drop_row = implied_side(true) && implied_side(false);
                f64 keep_lo = -kInf, keep_hi = kInf;
                if (!drop_row) {
                    // The row stays without x_j: sum_{k != j} a_k x_k = b - a x_j
                    // over x_j in [lo_j, hi_j].
                    const auto side = [&](f64 xj) -> std::optional<f64> {
                        if (!std::isfinite(xj)) return a * xj > 0 ? -kInf : kInf;
                        model::DyadicSum d;
                        d.add(b);
                        d.add_product(-a, xj);
                        f64 v = 0.0;
                        if (!d.representable(v)) return std::nullopt;
                        return v;
                    };
                    const auto s_lo = side(lo[sz(j)]), s_hi = side(hi[sz(j)]);
                    if (!s_lo || !s_hi) { ++stats.aggregation_inexact_skipped; continue; }
                    keep_lo = std::min(*s_lo, *s_hi);
                    keep_hi = std::max(*s_lo, *s_hi);
                }
                // Net nonzeros: entries the other rows gain, minus x_j's
                // entries leaving them, minus the pivot row (or x_j in it).
                long gain = 0;
                for (const Index i : others) {
                    if (i == r) continue;
                    for (const auto& e : prow)
                        if (e.first != j && find(i, e.first) == nullptr) ++gain;
                }
                const long delta = gain - static_cast<long>(count - 1) -
                                   (drop_row ? static_cast<long>(len) : 1L);
                if (delta > 0) continue;
                const bool better = !have_best || delta < best.delta_nnz ||
                    (delta == best.delta_nnz && std::fabs(a) > std::fabs(best.a));
                if (!better) continue;
                best = Choice{j, a, drop_row, delta, keep_lo, keep_hi};
                have_best = true;
            }
            if (!have_best) continue;
            const Index j = best.col;
            const f64 a = best.a;
            // Dry run: every rewrite exact, or nothing changes.
            rewrites.clear();
            cost_rewrites.clear();
            bool exact = true;
            live_rows_of(j, others);
            for (const Index i : others) {
                if (i == r) continue;
                const f64* aij_p = find(i, j);
                if (aij_p == nullptr) continue;
                const f64 aij = *aij_p;
                Rewrite rw;
                rw.row = i;
                // Merge row i (without j) with -aij/a times the pivot row (without j).
                const auto& ri = rows[sz(i)];
                std::size_t p = 0, q = 0;
                while ((p < ri.size() || q < prow.size()) && exact) {
                    const Index ci = p < ri.size() ? ri[p].first : std::numeric_limits<Index>::max();
                    const Index cq = q < prow.size() ? prow[q].first : std::numeric_limits<Index>::max();
                    if (ci == j) { ++p; continue; }
                    if (cq == j) { ++q; continue; }
                    if (ci < cq) { rw.entries.push_back(ri[p++]); continue; }
                    const f64 old = ci == cq ? ri[p].second : 0.0;
                    const auto nv = exact_rewrite(old, aij, prow[q].second, a);
                    if (!nv) { exact = false; break; }
                    if (*nv != 0.0) rw.entries.emplace_back(cq, *nv);
                    if (ci == cq) ++p;
                    ++q;
                }
                if (!exact) break;
                const auto side = [&](f64 s) -> std::optional<f64> {
                    if (!std::isfinite(s)) return s;
                    return exact_rewrite(s, aij, b, a);
                };
                const auto nlo = side(row_lo[sz(i)]), nhi = side(row_hi[sz(i)]);
                if (!nlo || !nhi) { exact = false; break; }
                rw.new_lo = *nlo;
                rw.new_hi = *nhi;
                rewrites.push_back(std::move(rw));
            }
            if (exact && c[sz(j)] != 0.0) {
                for (const auto& [k, ak] : prow) {
                    if (k == j) continue;
                    const auto nc = exact_rewrite(c[sz(k)], c[sz(j)], ak, a);
                    if (!nc) { exact = false; break; }
                    cost_rewrites.emplace_back(k, *nc);
                }
                model::DyadicSum t;
                t.add_product(c[sz(j)], b);
                const auto shift = exact_quotient(t, a);
                // The objective constant stays a double: the reduced
                // objective must equal the original one at every mapped
                // point, so a substitution whose shift would leave a sum no
                // double holds is declined (as an inexact rewrite is).
                f64 next_offset = 0.0;
                model::DyadicSum trial = offset;
                if (shift) trial.add(*shift);
                if (!shift || !trial.representable(next_offset)) exact = false;
                else if (exact) offset = std::move(trial);
            }
            if (!exact) { ++stats.aggregation_inexact_skipped; continue; }

            // Apply.
            LinearSubstitution ls;
            ls.col = j;
            ls.coef = a;
            ls.rhs = b;
            for (const auto& [k, ak] : prow)
                if (k != j) { ls.cols.push_back(k); ls.vals.push_back(ak); }
            for (auto& rw : rewrites) {
                for (const auto& e : rw.entries) col_rows[sz(e.first)].push_back(rw.row);
                rows[sz(rw.row)] = std::move(rw.entries);
                row_lo[sz(rw.row)] = rw.new_lo;
                row_hi[sz(rw.row)] = rw.new_hi;
                work += static_cast<std::uint64_t>(rows[sz(rw.row)].size());
            }
            for (const auto& [k, nc] : cost_rewrites) c[sz(k)] = nc;
            c[sz(j)] = 0.0;
            if (best.drop_row) {
                row_dead[sz(r)] = 1;
                ++stats.aggregation_rows_removed;
            } else {
                auto& keep = rows[sz(r)];
                keep.erase(std::remove_if(keep.begin(), keep.end(),
                                          [&](const Entry& e) { return e.first == j; }),
                           keep.end());
                row_lo[sz(r)] = best.keep_lo;
                row_hi[sz(r)] = best.keep_hi;
            }
            col_dead[sz(j)] = 1;
            out.eliminated[sz(j)] = 1;
            out.journal.push_back({PostsolveKind::Linear, out.linear_substitutions.size()});
            out.linear_substitutions.push_back(std::move(ls));
            ++stats.aggregated_cols;
            changed = true;
        }
        if (!changed || core::cancel_requested(opts.cancel)) break;
    }
    core::work_add(work + static_cast<std::uint64_t>(lp.nnz()), core::WorkTag::MilpPresolve);

    // The reduced model: live columns and rows, in their original order.
    model::LpProblem& red = out.reduced;
    red = model::LpProblem{};
    red.name = lp.name;
    red.maximize = lp.maximize;
    std::vector<Index> new_col(sz(n), -1);
    Index nn = 0;
    for (Index j = 0; j < n; ++j) {
        if (col_dead[sz(j)]) continue;
        new_col[sz(j)] = nn;
        out.reduced_col[sz(j)] = nn++;
        red.c.push_back(c[sz(j)]);
        red.col_lo.push_back(lo[sz(j)]);
        red.col_hi.push_back(hi[sz(j)]);
        if (!lp.is_integer.empty()) red.is_integer.push_back(lp.is_integer[sz(j)]);
        if (!lp.col_names.empty()) red.col_names.push_back(lp.col_names[sz(j)]);
    }
    std::vector<Index> ti, tj;
    std::vector<f64> tv;
    Index mm = 0;
    for (Index i = 0; i < m; ++i) {
        if (row_dead[sz(i)]) continue;
        out.reduced_row[sz(i)] = mm;
        for (const auto& [k, v] : rows[sz(i)]) {
            if (new_col[sz(k)] < 0 || v == 0.0) continue;
            ti.push_back(mm);
            tj.push_back(new_col[sz(k)]);
            tv.push_back(v);
        }
        red.row_lo.push_back(row_lo[sz(i)]);
        red.row_hi.push_back(row_hi[sz(i)]);
        if (!lp.row_names.empty()) red.row_names.push_back(lp.row_names[sz(i)]);
        ++mm;
    }
    red.A = sparse::from_triplets(mm, nn, ti, tj, tv);
    // A sum of doubles kept representable at every accepted step.
    if (!offset.representable(red.obj_offset))
        throw std::logic_error("aggregation: objective constant not representable");
    return out;
}

}  // namespace forge::search::detail
