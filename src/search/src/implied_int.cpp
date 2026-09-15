#include "sor/search/implied_int.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline bool nearly_int(f64 v, f64 tol) {
    return std::fabs(v - std::round(v)) <= tol;
}

inline bool is_eq_row(const model::LpProblem& lp, Index i, f64 tol) {
    return std::isfinite(lp.row_lo[sz(i)]) && std::isfinite(lp.row_hi[sz(i)]) &&
           std::fabs(lp.row_lo[sz(i)] - lp.row_hi[sz(i)]) <= tol &&
           nearly_int(lp.row_lo[sz(i)], tol);
}

inline bool fixed_int(const model::LpProblem& lp, Index j, f64 tol) {
    if (!std::isfinite(lp.col_lo[sz(j)]) || !std::isfinite(lp.col_hi[sz(j)]))
        return false;
    if (std::fabs(lp.col_lo[sz(j)] - lp.col_hi[sz(j)]) > tol) return false;
    return nearly_int(lp.col_lo[sz(j)], tol);
}

std::uint64_t infer_equality_pm1(model::LpProblem& p, f64 tol) {
    const Index m = p.n_rows(), n = p.n_cols();
    if (p.is_integer.size() != sz(n))
        p.is_integer.assign(sz(n), false);
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    const auto& av = p.A.vals;
    std::uint64_t added = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (Index j = 0; j < n; ++j) {
            if (p.is_integer[sz(j)]) continue;
            bool appears = false, implied = true;
            for (Index i = 0; i < m && implied; ++i) {
                f64 aj = 0.0;
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                    if (ci[sz(k)] == j) aj += av[sz(k)];
                if (std::fabs(aj) <= tol) continue;
                appears = true;
                if (!is_eq_row(p, i, tol) || std::fabs(std::fabs(aj) - 1.0) > tol) {
                    implied = false;
                    break;
                }
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    const Index q = ci[sz(k)];
                    if (q == j) continue;
                    const bool ok_int =
                        p.is_integer[sz(q)] || fixed_int(p, q, tol);
                    if (!ok_int || !nearly_int(av[sz(k)], tol)) {
                        implied = false;
                        break;
                    }
                }
            }
            if (appears && implied) {
                p.is_integer[sz(j)] = true;
                ++added;
                changed = true;
            }
        }
    }
    return added;
}

// Column is network-like: nonzeros ⊆ {+1}, {-1}, or one +1 and one -1.
bool network_column(const model::LpProblem& lp, Index j, f64 tol,
                    int& n_plus, int& n_minus, bool& bad) {
    n_plus = n_minus = 0;
    bad = false;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    // Scan all rows for column j (CSR: walk each row).
    for (Index i = 0; i < lp.n_rows(); ++i) {
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (ci[sz(k)] != j) continue;
            const f64 a = av[sz(k)];
            if (std::fabs(a) <= tol) continue;
            if (std::fabs(a - 1.0) <= tol) {
                ++n_plus;
            } else if (std::fabs(a + 1.0) <= tol) {
                ++n_minus;
            } else {
                bad = true;
                return false;
            }
        }
    }
    return n_plus <= 1 && n_minus <= 1 && (n_plus + n_minus) >= 1;
}

std::uint64_t infer_network(model::LpProblem& p, f64 tol) {
    const Index m = p.n_rows(), n = p.n_cols();
    if (p.is_integer.size() != sz(n))
        p.is_integer.assign(sz(n), false);

    // Equality rows with integer RHS only.
    std::vector<char> eq(sz(m), 0);
    for (Index i = 0; i < m; ++i) eq[sz(i)] = is_eq_row(p, i, tol) ? 1 : 0;

    // A continuous column is network-implied integer if:
    //  - every row it appears in is an integer equality,
    //  - every column in those rows is network-structured (±1 incidence),
    //  - every other column in those rows is already integer or fixed int,
    //  - finite bounds on j are integer-valued.
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    const auto& av = p.A.vals;

    std::vector<char> col_network(sz(n), 0);
    for (Index j = 0; j < n; ++j) {
        int np = 0, nm = 0;
        bool bad = false;
        if (network_column(p, j, tol, np, nm, bad) && !bad)
            col_network[sz(j)] = 1;
    }

    std::uint64_t added = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (Index j = 0; j < n; ++j) {
            if (p.is_integer[sz(j)]) continue;
            if (!col_network[sz(j)]) continue;
            if (std::isfinite(p.col_lo[sz(j)]) &&
                !nearly_int(p.col_lo[sz(j)], tol))
                continue;
            if (std::isfinite(p.col_hi[sz(j)]) &&
                !nearly_int(p.col_hi[sz(j)], tol))
                continue;

            bool appears = false, ok = true;
            for (Index i = 0; i < m && ok; ++i) {
                f64 aj = 0.0;
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                    if (ci[sz(k)] == j) aj += av[sz(k)];
                if (std::fabs(aj) <= tol) continue;
                appears = true;
                if (!eq[sz(i)]) {
                    ok = false;
                    break;
                }
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    const Index q = ci[sz(k)];
                    if (std::fabs(av[sz(k)]) <= tol) continue;
                    if (!col_network[sz(q)]) {
                        ok = false;
                        break;
                    }
                    if (q == j) continue;
                    if (!p.is_integer[sz(q)] && !fixed_int(p, q, tol)) {
                        // Other continuous network cols: allow if they also
                        // qualify in the same pass (mark jointly below).
                        // Require at least integer RHS already checked.
                        if (std::isfinite(p.col_lo[sz(q)]) &&
                            !nearly_int(p.col_lo[sz(q)], tol)) {
                            ok = false;
                            break;
                        }
                        if (std::isfinite(p.col_hi[sz(q)]) &&
                            !nearly_int(p.col_hi[sz(q)], tol)) {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if (appears && ok) {
                // Mark all continuous network columns that share these equality
                // rows (closed subsystem).
                std::vector<Index> stack = {j};
                std::vector<char> seen(sz(n), 0);
                seen[sz(j)] = 1;
                while (!stack.empty()) {
                    const Index u = stack.back();
                    stack.pop_back();
                    if (!p.is_integer[sz(u)]) {
                        p.is_integer[sz(u)] = true;
                        ++added;
                        changed = true;
                    }
                    for (Index i = 0; i < m; ++i) {
                        bool hit = false;
                        for (core::Offset k = rp[sz(i)];
                             k < rp[sz(i) + 1]; ++k)
                            if (ci[sz(k)] == u &&
                                std::fabs(av[sz(k)]) > tol) {
                                hit = true;
                                break;
                            }
                        if (!hit || !eq[sz(i)]) continue;
                        for (core::Offset k = rp[sz(i)];
                             k < rp[sz(i) + 1]; ++k) {
                            const Index q = ci[sz(k)];
                            if (std::fabs(av[sz(k)]) <= tol) continue;
                            if (!col_network[sz(q)] || seen[sz(q)]) continue;
                            seen[sz(q)] = 1;
                            stack.push_back(q);
                        }
                    }
                }
            }
        }
    }
    return added;
}

// Consecutive-ones: 0/1 coefficients, ones in each column form an interval
// of row indices (current order). Equality rows only.
std::uint64_t infer_consecutive_ones(model::LpProblem& p, f64 tol) {
    const Index m = p.n_rows(), n = p.n_cols();
    if (p.is_integer.size() != sz(n))
        p.is_integer.assign(sz(n), false);

    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    const auto& av = p.A.vals;

    std::vector<char> eq(sz(m), 0);
    Index n_eq = 0;
    for (Index i = 0; i < m; ++i) {
        if (is_eq_row(p, i, tol)) {
            eq[sz(i)] = 1;
            ++n_eq;
        }
    }
    if (n_eq == 0) return 0;

    // Per column: list of equality rows with nonzero; must be 0/1 and interval.
    std::vector<char> c1_col(sz(n), 0);
    for (Index j = 0; j < n; ++j) {
        Index first = -1, last = -1;
        int nnz = 0;
        bool bad = false;
        for (Index i = 0; i < m; ++i) {
            if (!eq[sz(i)]) {
                // Non-equality row must not contain j.
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    if (ci[sz(k)] == j && std::fabs(av[sz(k)]) > tol) {
                        bad = true;
                        break;
                    }
                }
                if (bad) break;
                continue;
            }
            f64 aj = 0.0;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                if (ci[sz(k)] == j) aj += av[sz(k)];
            if (std::fabs(aj) <= tol) continue;
            if (std::fabs(aj - 1.0) > tol) {
                bad = true;
                break;
            }
            if (first < 0) first = i;
            last = i;
            ++nnz;
        }
        if (bad || nnz == 0 || first < 0) continue;
        // Interval: every equality row between first and last must have the 1.
        int expect = 0;
        for (Index i = first; i <= last; ++i) {
            if (!eq[sz(i)]) continue;
            f64 aj = 0.0;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                if (ci[sz(k)] == j) aj += av[sz(k)];
            if (std::fabs(aj - 1.0) <= tol)
                ++expect;
            else if (std::fabs(aj) > tol) {
                bad = true;
                break;
            } else {
                bad = true;  // gap in equality block
                break;
            }
        }
        if (!bad && expect == nnz) c1_col[sz(j)] = 1;
    }

    // Subsystem of C1 columns: all continuous with integer finite bounds
    // and only C1/integer companions → mark.
    std::uint64_t added = 0;
    for (Index j = 0; j < n; ++j) {
        if (p.is_integer[sz(j)] || !c1_col[sz(j)]) continue;
        if (std::isfinite(p.col_lo[sz(j)]) && !nearly_int(p.col_lo[sz(j)], tol))
            continue;
        if (std::isfinite(p.col_hi[sz(j)]) && !nearly_int(p.col_hi[sz(j)], tol))
            continue;

        bool ok = true, appears = false;
        for (Index i = 0; i < m && ok; ++i) {
            f64 aj = 0.0;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                if (ci[sz(k)] == j) aj += av[sz(k)];
            if (std::fabs(aj) <= tol) continue;
            appears = true;
            if (!eq[sz(i)]) {
                ok = false;
                break;
            }
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index q = ci[sz(k)];
                if (std::fabs(av[sz(k)]) <= tol) continue;
                if (!c1_col[sz(q)] && !p.is_integer[sz(q)] &&
                    !fixed_int(p, q, tol)) {
                    ok = false;
                    break;
                }
            }
        }
        if (appears && ok) {
            p.is_integer[sz(j)] = true;
            ++added;
        }
    }
    return added;
}

void snap_integer_bounds(model::LpProblem& lp, std::vector<f64>* col_lo,
                         std::vector<f64>* col_hi, f64 tol,
                         std::uint64_t& snapped) {
    const Index n = lp.n_cols();
    auto& lo = col_lo ? *col_lo : lp.col_lo;
    auto& hi = col_hi ? *col_hi : lp.col_hi;
    if (static_cast<Index>(lo.size()) != n ||
        static_cast<Index>(hi.size()) != n)
        return;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (std::isfinite(lo[sz(j)])) {
            const f64 nl = std::ceil(lo[sz(j)] - tol);
            if (nl > lo[sz(j)] + tol) {
                lo[sz(j)] = nl;
                ++snapped;
            }
        }
        if (std::isfinite(hi[sz(j)])) {
            const f64 nh = std::floor(hi[sz(j)] + tol);
            if (nh < hi[sz(j)] - tol) {
                hi[sz(j)] = nh;
                ++snapped;
            }
        }
    }
    if (col_lo) lp.col_lo = lo;
    if (col_hi) lp.col_hi = hi;
}

}  // namespace

ImpliedIntDiagnostics infer_implied_integers_ex(
    model::LpProblem& lp, const ImpliedIntOptions& opts,
    std::vector<f64>* col_lo, std::vector<f64>* col_hi) {
    ImpliedIntDiagnostics diag;
    if (!opts.enabled) return diag;
    if (opts.equality_pm1)
        diag.equality_pm1 = infer_equality_pm1(lp, opts.tol);
    if (opts.network) diag.network = infer_network(lp, opts.tol);
    if (opts.consecutive_ones)
        diag.consecutive_ones = infer_consecutive_ones(lp, opts.tol);
    // Cascade: TU marks may unlock more ±1 inferences.
    if (opts.equality_pm1 && (diag.network + diag.consecutive_ones) > 0)
        diag.equality_pm1 += infer_equality_pm1(lp, opts.tol);
    diag.total =
        diag.equality_pm1 + diag.network + diag.consecutive_ones;
    snap_integer_bounds(lp, col_lo, col_hi, opts.tol, diag.bounds_snapped);
    return diag;
}

}  // namespace sor::search
