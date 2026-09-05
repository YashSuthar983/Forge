#include "sor/presolve/presolve.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sor::presolve {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

}  // namespace

PresolveMap presolve_lp(const model::LpProblem& in) {
    PresolveMap out;
    out.stats = PresolveStats{};
    out.problem = in;

    const Index m = in.n_rows();
    const Index n = in.n_cols();

    std::vector<char> row_live(sz(m), 1);
    std::vector<char> col_live(sz(n), 1);
    std::vector<f64> fixed(n, 0.0);
    std::vector<f64> work_lo = in.col_lo;
    std::vector<f64> work_hi = in.col_hi;
    std::vector<Index> col_nnz(sz(n), 0);
    for (Index i = 0; i < m; ++i)
        for (Offset k = in.A.pattern.row_ptr()[sz(i)];
             k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k)
            if (in.A.vals[sz(k)] != 0.0)
                ++col_nnz[sz(in.A.pattern.col_idx()[sz(k)])];

    const auto add_interval = [](f64& lo, f64& hi, f64 a, f64 l, f64 u) {
        if (a >= 0.0) {
            lo += a * l;
            hi += a * u;
        } else {
            lo += a * u;
            hi += a * l;
        }
    };

    bool changed = true;
    for (int pass = 0; pass < 64 && changed; ++pass) {
        changed = false;

        // Fixed columns (lo == hi).
        for (Index j = 0; j < n; ++j) {
            if (!col_live[sz(j)]) continue;
            if (work_lo[sz(j)] == work_hi[sz(j)]) {
                col_live[sz(j)] = 0;
                fixed[sz(j)] = work_lo[sz(j)];
                ++out.stats.cols_fixed;
                changed = true;
            }
        }

        // Structurally empty columns can be fixed at an objective-minimizing
        // bound. Use the precomputed sparse counts: scanning every row for
        // every column made this rule quadratic on wide models.
        for (Index j = 0; j < n; ++j) {
            if (!col_live[sz(j)] || col_nnz[sz(j)] != 0) continue;
            const f64 mc = in.maximize ? -in.c[sz(j)] : in.c[sz(j)];
            f64 v = 0.0;
            if (mc > 0.0 && work_lo[sz(j)] > -model::kInf) v = work_lo[sz(j)];
            else if (mc < 0.0 && work_hi[sz(j)] < model::kInf) v = work_hi[sz(j)];
            else if (mc == 0.0) {
                if (work_lo[sz(j)] > -model::kInf) v = work_lo[sz(j)];
                else if (work_hi[sz(j)] < model::kInf) v = work_hi[sz(j)];
                else v = 0.0;
            } else {
                continue;
            }
            col_live[sz(j)] = 0;
            fixed[sz(j)] = v;
            ++out.stats.cols_fixed;
            changed = true;
        }

        // Empty rows after substituting fixed columns.
        for (Index i = 0; i < m; ++i) {
            if (!row_live[sz(i)]) continue;
            f64 activity = 0.0;
            bool has_live = false;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                if (col_live[sz(j)]) {
                    has_live = true;
                    break;
                }
                activity += in.A.vals[sz(k)] * fixed[sz(j)];
            }
            if (!has_live && in.row_lo[sz(i)] <= activity && activity <= in.row_hi[sz(i)]) {
                row_live[sz(i)] = 0;
                ++out.stats.rows_removed;
                changed = true;
            }
        }

        // Singleton-column tightening is reserved for the full reversible
        // dual-recovery stack. Applying it without that stack can produce a
        // valid primal point but an invalid lifted dual certificate.

        // Remove rows whose full interval of possible activity already lies
        // inside their bounds. This is a reversible redundant-row reduction;
        // the lifted multiplier is zero, which is valid by construction.
        for (Index i = 0; i < m; ++i) {
            if (!row_live[sz(i)]) continue;
            f64 act_lo = 0.0, act_hi = 0.0;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                const f64 a = in.A.vals[sz(k)];
                if (!col_live[sz(j)]) {
                    act_lo += a * fixed[sz(j)];
                    act_hi += a * fixed[sz(j)];
                } else {
                    add_interval(act_lo, act_hi, a, work_lo[sz(j)], work_hi[sz(j)]);
                }
            }
            if (act_lo >= in.row_lo[sz(i)] && act_hi <= in.row_hi[sz(i)]) {
                row_live[sz(i)] = 0;
                ++out.stats.rows_removed;
                changed = true;
            }
        }

        // Singleton-row bound tightening/fixing. A row with
        // exactly one live entry a*x_j = rhs pins x_j at rhs/a once every
        // other entry is fixed. Equality-only is the sound subset:
        //   * the row's logical is FIXED (lo == hi), so its multiplier is
        //     exempt from every sign condition in the certificate check;
        //   * the pinned column can land at an INTERIOR point of its own
        //     bounds, which demands reduced cost ~0 -- the postsolve dual
        //     recovery (eq_row_/eq_col_/eq_coeff_ triples) sets the row's
        //     multiplier to exactly that.
        // Inequality singletons only tighten bounds; equality rows can fix a
        // variable exactly and carry the existing multiplier recovery record.
        // The effective row bounds subtract already-fixed contributions: the
        // ORIGINAL bounds are wrong for cascaded rows (x+y=0 with x fixed at
        // 5 pins y at -5, not 0).
        for (Index i = 0; i < m; ++i) {
            if (!row_live[sz(i)]) continue;
            Index col = -1;
            f64 a = 0.0;
            int cnt = 0;
            f64 shift = 0.0;   // contribution of already-fixed columns
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                if (!col_live[sz(j)]) {
                    shift += in.A.vals[sz(k)] * fixed[sz(j)];
                    continue;
                }
                ++cnt;
                col = j;
                a = in.A.vals[sz(k)];
            }
            if (cnt != 1 || a == 0.0) continue;
            const f64 clo = work_lo[sz(col)];
            const f64 chi = work_hi[sz(col)];
            if (in.row_lo[sz(i)] == in.row_hi[sz(i)]) {
                const f64 rhs = in.row_lo[sz(i)] - shift;
                const f64 v = rhs / a;
                if (v < clo - 1e-12 || v > chi + 1e-12) continue;
                if (clo == chi) continue;
                col_live[sz(col)] = 0;
                fixed[sz(col)] = v;
                work_lo[sz(col)] = v;
                work_hi[sz(col)] = v;
                ++out.stats.cols_fixed;
                out.eq_row_.push_back(i);
                out.eq_col_.push_back(col);
                out.eq_coeff_.push_back(a);
                changed = true;
                continue;
            }

            f64 implied_lo = -model::kInf, implied_hi = model::kInf;
            if (a > 0.0) {
                if (in.row_lo[sz(i)] > -model::kInf) implied_lo = (in.row_lo[sz(i)] - shift) / a;
                if (in.row_hi[sz(i)] <  model::kInf) implied_hi = (in.row_hi[sz(i)] - shift) / a;
            } else {
                if (in.row_hi[sz(i)] <  model::kInf) implied_lo = (in.row_hi[sz(i)] - shift) / a;
                if (in.row_lo[sz(i)] > -model::kInf) implied_hi = (in.row_lo[sz(i)] - shift) / a;
            }
            const f64 new_lo = std::max(clo, implied_lo);
            const f64 new_hi = std::min(chi, implied_hi);
            if (new_lo > new_hi + 1e-12 * (1.0 + std::max(std::fabs(new_lo), std::fabs(new_hi))))
                continue;
            if (new_lo != clo || new_hi != chi) {
                work_lo[sz(col)] = new_lo;
                work_hi[sz(col)] = new_hi;
                out.bound_changes.push_back({col, i, a, clo, chi, new_lo, new_hi});
                ++out.stats.bounds_tightened;
                changed = true;
            }
        }
    }

    out.orig_to_new.assign(sz(n), -1);
    out.fixed_value.assign(sz(n), 0.0);
    Index new_j = 0;
    for (Index j = 0; j < n; ++j) {
        if (!col_live[sz(j)]) {
            out.fixed_value[sz(j)] = fixed[sz(j)];
            ++out.stats.cols_removed;
            continue;
        }
        out.orig_to_new[sz(j)] = new_j;
        out.new_to_orig.push_back(j);
        ++new_j;
    }

    const Index new_n = new_j;
    Index new_i = 0;
    out.row_orig_to_new.assign(sz(m), -1);
    out.row_new_to_orig.clear();
    for (Index i = 0; i < m; ++i) {
        if (!row_live[sz(i)]) continue;
        out.row_orig_to_new[sz(i)] = new_i++;
        out.row_new_to_orig.push_back(i);
    }
    const std::vector<Index>& row_map = out.row_orig_to_new;

    model::LpProblem red;
    red.name = in.name;
    red.c.assign(sz(new_n), 0.0);
    red.col_lo.assign(sz(new_n), 0.0);
    red.col_hi.assign(sz(new_n), 0.0);
    red.row_lo.assign(sz(new_i), 0.0);
    red.row_hi.assign(sz(new_i), 0.0);
    red.maximize = in.maximize;
    red.obj_offset = in.obj_offset;
    red.is_integer.assign(sz(new_n), false);
    red.row_names.reserve(sz(new_i));
    red.col_names.reserve(sz(new_n));

    for (Index nj = 0; nj < new_n; ++nj) {
        const Index oj = out.new_to_orig[sz(nj)];
        red.c[sz(nj)] = in.c[sz(oj)];
        red.col_lo[sz(nj)] = work_lo[sz(oj)];
        red.col_hi[sz(nj)] = work_hi[sz(oj)];
        if (!in.is_integer.empty()) red.is_integer[sz(nj)] = in.is_integer[sz(oj)];
        if (!in.col_names.empty()) red.col_names.push_back(in.col_names[sz(oj)]);
    }
    for (Index i = 0; i < m; ++i) {
        if (row_map[sz(i)] < 0) continue;
        const Index ni = row_map[sz(i)];
        // A fixed column's contribution to this row is a CONSTANT: the row
        // sum_live x_j + const must lie in [row_lo, row_hi], so the reduced row
        // must carry the SHIFTED bounds [row_lo - const, row_hi - const].
        // Keeping the original bounds silently changes the feasible set: the
        // reduced LP is over-constrained whenever const != 0, which reported
        // bore3d/capri/nesm/... (18 Netlib instances) as Infeasible and left
        // standata "optimal" at a point violating the original rows by 171.
        f64 shift = 0.0;
        for (Offset k = in.A.pattern.row_ptr()[sz(i)];
             k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
            const Index j = in.A.pattern.col_idx()[sz(k)];
            if (!col_live[sz(j)]) shift += in.A.vals[sz(k)] * fixed[sz(j)];
        }
        // IEEE: -inf - finite stays -inf, so infinite bounds survive the shift.
        red.row_lo[sz(ni)] = in.row_lo[sz(i)] - shift;
        red.row_hi[sz(ni)] = in.row_hi[sz(i)] - shift;
        if (!in.row_names.empty()) red.row_names.push_back(in.row_names[sz(i)]);
    }

    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    for (Index i = 0; i < m; ++i) {
        if (row_map[sz(i)] < 0) continue;
        const Index ni = row_map[sz(i)];
        for (Offset k = in.A.pattern.row_ptr()[sz(i)];
             k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
            const Index j = in.A.pattern.col_idx()[sz(k)];
            if (!col_live[sz(j)]) continue;
            const Index nj = out.orig_to_new[sz(j)];
            rows.push_back(ni);
            cols.push_back(nj);
            vals.push_back(in.A.vals[sz(k)]);
        }
    }
    red.A = sparse::from_triplets(new_i, new_n, rows, cols, vals);

    out.problem = std::move(red);
    return out;
}

std::vector<f64> postsolve(const PresolveMap& map, const std::vector<f64>& x_reduced) {
    const Index n = static_cast<Index>(map.orig_to_new.size());
    const auto n_red = static_cast<Index>(x_reduced.size());
    std::vector<f64> x(sz(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        const Index nj = map.orig_to_new[sz(j)];
        if (nj < 0 || nj >= n_red) x[sz(j)] = map.fixed_value[sz(j)];
        else                       x[sz(j)] = x_reduced[sz(nj)];
    }
    return x;
}

}  // namespace sor::presolve
