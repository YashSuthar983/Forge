#include "sor/search/implied_int.hpp"

#include <chrono>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sor::search {

namespace {
// Deadline checked inside each rule's outer loop, not merely between rules.
//
// Between-rule checks are not enough: a SINGLE rule ran 49.2 s on MIPLIB2017
// netdiversion (119589 x 129180) against a 10 s solver limit -- 49204 ms ->
// 52.6 ms with --no-implied-int. Checks have to be where the loop is.
struct IiDeadline {
    std::chrono::steady_clock::time_point t0{};
    double limit = 0.0;
    bool over() const {
        return limit > 0.0 &&
               std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count() > limit;
    }
};
// Poll every 256 columns: frequent enough to bound a quadratic rule, rare
// enough that the clock read never shows up.
constexpr std::uint64_t kIiPollMask = 0xFF;
}  // namespace

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

std::uint64_t infer_equality_pm1(model::LpProblem& p, f64 tol,
                    const IiDeadline& dl) {
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
            if ((static_cast<std::uint64_t>(j) & kIiPollMask) == 0 && dl.over())
                break;
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

std::uint64_t infer_network(model::LpProblem& p, f64 tol,
                    const IiDeadline& dl) {
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
        if ((static_cast<std::uint64_t>(j) & kIiPollMask) == 0 && dl.over())
            break;
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
            if ((static_cast<std::uint64_t>(j) & kIiPollMask) == 0 && dl.over())
                break;
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
std::uint64_t infer_consecutive_ones(model::LpProblem& p, f64 tol,
                    const IiDeadline& dl) {
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
        if ((static_cast<std::uint64_t>(j) & kIiPollMask) == 0 && dl.over())
            break;
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
        if ((static_cast<std::uint64_t>(j) & kIiPollMask) == 0 && dl.over())
            break;
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

// --- Paper arXiv:2504.07209 §3.1 / §7.1 (Theorem 3.2, Cor. 3.3, Algorithm 1) ---

inline bool coeff_integral(f64 a, f64 tol) {
    return std::fabs(a) <= tol || nearly_int(a, tol);
}

// Row has only integer coefficients and integer right-hand side data for TU
// detection (Algorithm 1, line 2: incompatible rows R).
inline bool row_integral_data(const model::LpProblem& lp, Index i, f64 tol) {
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& av = lp.A.vals;
    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        if (!coeff_integral(av[sz(k)], tol)) return false;
    }
    if (is_eq_row(lp, i, tol)) return true;
    if (std::isfinite(lp.row_lo[sz(i)]) &&
        !nearly_int(lp.row_lo[sz(i)], tol))
        return false;
    if (std::isfinite(lp.row_hi[sz(i)]) &&
        !nearly_int(lp.row_hi[sz(i)], tol))
        return false;
    return true;
}

// Row is usable in a ≤-form TU block (Theorem 3.2): equality or one-sided ≤/≥.
inline bool row_tu_sense_ok(const model::LpProblem& lp, Index i, f64 tol) {
    if (is_eq_row(lp, i, tol)) return true;
    const bool lo = std::isfinite(lp.row_lo[sz(i)]);
    const bool hi = std::isfinite(lp.row_hi[sz(i)]);
    return lo != hi;
}

// Network column restricted to row set MK (±1, ≤1 of each sign per column).
bool network_column_on_rows(const model::LpProblem& lp, Index j,
                            const std::vector<Index>& mk, f64 tol, int& n_plus,
                            int& n_minus) {
    n_plus = n_minus = 0;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (Index i : mk) {
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (ci[sz(k)] != j) continue;
            const f64 a = av[sz(k)];
            if (std::fabs(a) <= tol) continue;
            if (std::fabs(a - 1.0) <= tol) {
                ++n_plus;
            } else if (std::fabs(a + 1.0) <= tol) {
                ++n_minus;
            } else {
                return false;
            }
        }
    }
    return n_plus <= 1 && n_minus <= 1 && (n_plus + n_minus) >= 1;
}

// Transposed-network row: ±1 pattern on column set nk (paper §7.1, row aug).
bool network_row_on_cols(const model::LpProblem& lp, Index i,
                         const std::vector<Index>& nk, f64 tol) {
    int n_plus = 0, n_minus = 0;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        const Index j = ci[sz(k)];
        bool in = false;
        for (Index q : nk) {
            if (q == j) {
                in = true;
                break;
            }
        }
        if (!in) continue;
        const f64 a = av[sz(k)];
        if (std::fabs(a) <= tol) continue;
        if (std::fabs(a - 1.0) <= tol) {
            ++n_plus;
        } else if (std::fabs(a + 1.0) <= tol) {
            ++n_minus;
        } else {
            return false;
        }
    }
    return n_plus <= 1 && n_minus <= 1 && (n_plus + n_minus) >= 1;
}

inline bool continuous_int_bounds(const model::LpProblem& lp, Index j, f64 tol) {
    if (std::isfinite(lp.col_lo[sz(j)]) && !nearly_int(lp.col_lo[sz(j)], tol))
        return false;
    if (std::isfinite(lp.col_hi[sz(j)]) && !nearly_int(lp.col_hi[sz(j)], tol))
        return false;
    return true;
}

struct UnionFind {
    std::vector<Index> p, r;
    explicit UnionFind(Index n) : p(sz(n)), r(sz(n), 0) {
        for (Index i = 0; i < n; ++i) p[sz(i)] = i;
    }
    Index find(Index x) {
        while (p[sz(x)] != x) {
            p[sz(x)] = p[sz(p[sz(x)])];
            x = p[sz(x)];
        }
        return x;
    }
    void unite(Index a, Index b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (r[sz(a)] < r[sz(b)]) std::swap(a, b);
        p[sz(b)] = a;
        if (r[sz(a)] == r[sz(b)]) ++r[sz(a)];
    }
};

// Corollary 3.3 (dual detection): scale rows where column k appears so that
// a_jk = ±1 and all other coefficients / RHS are integral.
std::uint64_t infer_dual_rational(model::LpProblem& p, f64 tol,
                    const IiDeadline& dl) {
    const Index m = p.n_rows(), n = p.n_cols();
    if (p.is_integer.size() != sz(n))
        p.is_integer.assign(sz(n), false);

    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    const auto& av = p.A.vals;

    std::uint64_t added = 0;
    for (Index k = 0; k < n; ++k) {
        if ((static_cast<std::uint64_t>(k) & kIiPollMask) == 0 && dl.over())
            break;
        if (p.is_integer[sz(k)] || fixed_int(p, k, tol)) continue;

        bool appears = false, ok = true;
        for (Index i = 0; i < m && ok; ++i) {
            f64 ak = 0.0;
            for (core::Offset off = rp[sz(i)]; off < rp[sz(i) + 1]; ++off)
                if (ci[sz(off)] == k) ak += av[sz(off)];
            if (std::fabs(ak) <= tol) continue;
            appears = true;
            if (!row_integral_data(p, i, tol) || !row_tu_sense_ok(p, i, tol)) {
                ok = false;
                break;
            }
            const f64 pivot = ak;
            f64 rhs = 0.0;
            if (is_eq_row(p, i, tol)) {
                rhs = p.row_lo[sz(i)];
            } else if (!std::isfinite(p.row_lo[sz(i)])) {
                rhs = p.row_hi[sz(i)];
            } else if (!std::isfinite(p.row_hi[sz(i)])) {
                rhs = p.row_lo[sz(i)];
            } else {
                ok = false;
                break;
            }
            if (!nearly_int(rhs / pivot, tol)) {
                ok = false;
                break;
            }
            for (core::Offset off = rp[sz(i)]; off < rp[sz(i) + 1]; ++off) {
                const Index s = ci[sz(off)];
                if (std::fabs(av[sz(off)]) <= tol) continue;
                if (!nearly_int(av[sz(off)] / pivot, tol)) {
                    ok = false;
                    break;
                }
                if (s == k) continue;
                if (!p.is_integer[sz(s)] && !fixed_int(p, s, tol)) {
                    ok = false;
                    break;
                }
            }
        }
        if (appears && ok) {
            p.is_integer[sz(k)] = true;
            ++added;
        }
    }
    return added;
}

// GrowNetwork test (paper §7.1): each column of a network matrix has at most
// one +1 and one −1; any such column set forms a network matrix (TU).
bool grow_network_add_column(const model::LpProblem& lp,
                             const std::vector<Index>& mk, Index j, f64 tol) {
    int np = 0, nm = 0;
    return network_column_on_rows(lp, j, mk, tol, np, nm);
}

// Algorithm 1 column-wise pass on continuous connected components.
std::uint64_t infer_tu_network_block(model::LpProblem& p, f64 tol,
                    const IiDeadline& dl) {
    const Index m = p.n_rows(), n = p.n_cols();
    if (p.is_integer.size() != sz(n))
        p.is_integer.assign(sz(n), false);

    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    const auto& av = p.A.vals;

    std::vector<char> blocked_row(sz(m), 0);
    for (Index i = 0; i < m; ++i) {
        if (!row_integral_data(p, i, tol) || !row_tu_sense_ok(p, i, tol))
            blocked_row[sz(i)] = 1;
    }

    std::vector<Index> cont_cols;
    for (Index j = 0; j < n; ++j) {
        if ((static_cast<std::uint64_t>(j) & kIiPollMask) == 0 && dl.over())
            break;
        if (p.is_integer[sz(j)] || fixed_int(p, j, tol)) continue;
        cont_cols.push_back(j);
    }
    if (cont_cols.empty()) return 0;

    std::vector<Index> col_id(sz(n), -1);
    for (Index t = 0; t < static_cast<Index>(cont_cols.size()); ++t)
        col_id[sz(cont_cols[static_cast<std::size_t>(t)])] = t;

    UnionFind uf(static_cast<Index>(cont_cols.size()));
    for (Index i = 0; i < m; ++i) {
        if (blocked_row[sz(i)]) continue;
        Index prev = -1;
        for (core::Offset off = rp[sz(i)]; off < rp[sz(i) + 1]; ++off) {
            const Index j = ci[sz(off)];
            if (std::fabs(av[sz(off)]) <= tol) continue;
            if (p.is_integer[sz(j)] || fixed_int(p, j, tol)) continue;
            const Index id = col_id[sz(j)];
            if (id < 0) continue;
            if (prev >= 0)
                uf.unite(prev, id);
            else
                prev = id;
        }
    }

    std::vector<std::vector<Index>> comps(static_cast<std::size_t>(cont_cols.size()));
    for (Index t = 0; t < static_cast<Index>(cont_cols.size()); ++t) {
        const Index root = uf.find(t);
        comps[sz(root)].push_back(cont_cols[static_cast<std::size_t>(t)]);
    }

    std::uint64_t added = 0;
    for (const auto& nk : comps) {
        if (nk.empty()) continue;

        std::vector<char> in_comp(sz(n), 0);
        for (Index j : nk) in_comp[sz(j)] = 1;

        std::vector<Index> mk;
        for (Index i = 0; i < m; ++i) {
            if (blocked_row[sz(i)]) continue;
            bool hit = false;
            for (core::Offset off = rp[sz(i)]; off < rp[sz(i) + 1]; ++off) {
                const Index j = ci[sz(off)];
                if (in_comp[sz(j)] && std::fabs(av[sz(off)]) > tol) {
                    hit = true;
                    break;
                }
            }
            if (hit) mk.push_back(i);
        }
        if (mk.empty()) continue;

        bool coupling_ok = true;
        for (Index i : mk) {
            for (core::Offset off = rp[sz(i)]; off < rp[sz(i) + 1]; ++off) {
                const Index j = ci[sz(off)];
                if (std::fabs(av[sz(off)]) <= tol) continue;
                if (in_comp[sz(j)]) continue;
                if (!p.is_integer[sz(j)] && !fixed_int(p, j, tol)) {
                    coupling_ok = false;
                    break;
                }
            }
            if (!coupling_ok) break;
        }
        if (!coupling_ok) continue;

        std::vector<Index> tk;
        for (Index j : nk) {
            if (!grow_network_add_column(p, mk, j, tol)) break;
            tk.push_back(j);
        }
        if (tk.size() != nk.size()) continue;

        for (Index j : tk) {
            if (p.is_integer[sz(j)]) continue;
            if (!continuous_int_bounds(p, j, tol)) continue;
            p.is_integer[sz(j)] = true;
            ++added;
        }
    }
    return added;
}

// Algorithm 1 transposed-network pass (row-wise growth, paper §7.1).
std::uint64_t infer_tu_network_transpose(model::LpProblem& p, f64 tol,
                    const IiDeadline& dl) {
    const Index m = p.n_rows(), n = p.n_cols();
    if (p.is_integer.size() != sz(n))
        p.is_integer.assign(sz(n), false);

    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    const auto& av = p.A.vals;

    std::vector<char> blocked_row(sz(m), 0);
    for (Index i = 0; i < m; ++i) {
        if (!row_integral_data(p, i, tol) || !row_tu_sense_ok(p, i, tol))
            blocked_row[sz(i)] = 1;
    }

    std::vector<Index> cont_cols;
    for (Index j = 0; j < n; ++j) {
        if ((static_cast<std::uint64_t>(j) & kIiPollMask) == 0 && dl.over())
            break;
        if (p.is_integer[sz(j)] || fixed_int(p, j, tol)) continue;
        cont_cols.push_back(j);
    }
    if (cont_cols.empty()) return 0;

    std::vector<Index> col_id(sz(n), -1);
    for (Index t = 0; t < static_cast<Index>(cont_cols.size()); ++t)
        col_id[sz(cont_cols[static_cast<std::size_t>(t)])] = t;

    UnionFind uf(static_cast<Index>(cont_cols.size()));
    for (Index i = 0; i < m; ++i) {
        if (blocked_row[sz(i)]) continue;
        Index prev = -1;
        for (core::Offset off = rp[sz(i)]; off < rp[sz(i) + 1]; ++off) {
            const Index j = ci[sz(off)];
            if (std::fabs(av[sz(off)]) <= tol) continue;
            if (p.is_integer[sz(j)] || fixed_int(p, j, tol)) continue;
            const Index id = col_id[sz(j)];
            if (id < 0) continue;
            if (prev >= 0)
                uf.unite(prev, id);
            else
                prev = id;
        }
    }

    std::vector<std::vector<Index>> comps(static_cast<std::size_t>(cont_cols.size()));
    for (Index t = 0; t < static_cast<Index>(cont_cols.size()); ++t) {
        const Index root = uf.find(t);
        comps[sz(root)].push_back(cont_cols[static_cast<std::size_t>(t)]);
    }

    std::uint64_t added = 0;
    for (const auto& nk : comps) {
        if (nk.empty()) continue;

        std::vector<char> in_comp(sz(n), 0);
        for (Index j : nk) in_comp[sz(j)] = 1;

        std::vector<Index> mk;
        for (Index i = 0; i < m; ++i) {
            if (blocked_row[sz(i)]) continue;
            bool hit = false;
            for (core::Offset off = rp[sz(i)]; off < rp[sz(i) + 1]; ++off) {
                const Index j = ci[sz(off)];
                if (in_comp[sz(j)] && std::fabs(av[sz(off)]) > tol) {
                    hit = true;
                    break;
                }
            }
            if (hit) mk.push_back(i);
        }
        if (mk.empty()) continue;

        bool coupling_ok = true;
        for (Index i : mk) {
            for (core::Offset off = rp[sz(i)]; off < rp[sz(i) + 1]; ++off) {
                const Index j = ci[sz(off)];
                if (std::fabs(av[sz(off)]) <= tol) continue;
                if (in_comp[sz(j)]) continue;
                if (!p.is_integer[sz(j)] && !fixed_int(p, j, tol)) {
                    coupling_ok = false;
                    break;
                }
            }
            if (!coupling_ok) break;
        }
        if (!coupling_ok) continue;

        std::vector<Index> tr;
        for (Index i : mk) {
            if (!network_row_on_cols(p, i, nk, tol)) break;
            tr.push_back(i);
        }
        if (tr.size() != mk.size()) continue;

        for (Index j : nk) {
            if (p.is_integer[sz(j)]) continue;
            if (!continuous_int_bounds(p, j, tol)) continue;
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
    // Deadline between rules. Each rule is a full pass over the matrix, so a
    // per-rule check is both cheap and the right granularity.
    const auto ii_t0 = std::chrono::steady_clock::now();
    const IiDeadline dl{ii_t0, opts.time_limit_s};
    const auto ii_over = [&]() {
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - ii_t0).count() >
                   opts.time_limit_s;
    };
    if (ii_over()) { diag.aborted_on_time = 1; return diag; }
    if (opts.equality_pm1)
        diag.equality_pm1 = infer_equality_pm1(lp, opts.tol, dl);
    if (ii_over()) { diag.aborted_on_time = 1; return diag; }
    if (opts.network) diag.network = infer_network(lp, opts.tol, dl);
    if (ii_over()) { diag.aborted_on_time = 1; return diag; }
    if (opts.consecutive_ones)
        diag.consecutive_ones = infer_consecutive_ones(lp, opts.tol, dl);
    if (ii_over()) { diag.aborted_on_time = 1; return diag; }
    if (opts.dual_rational)
        diag.dual_rational = infer_dual_rational(lp, opts.tol, dl);
    if (ii_over()) { diag.aborted_on_time = 1; return diag; }
    if (opts.tu_network_block) {
        diag.tu_network_block = infer_tu_network_block(lp, opts.tol, dl);
        diag.tu_network_transpose = infer_tu_network_transpose(lp, opts.tol, dl);
    }
    // Cascade: TU marks may unlock more ±1 / dual inferences.
    const std::uint64_t tu_extra = diag.network + diag.consecutive_ones +
                                   diag.tu_network_block +
                                   diag.tu_network_transpose;
    if (opts.equality_pm1 && tu_extra > 0)
        diag.equality_pm1 += infer_equality_pm1(lp, opts.tol, dl);
    if (opts.dual_rational && tu_extra > 0)
        diag.dual_rational += infer_dual_rational(lp, opts.tol, dl);
    diag.total = diag.equality_pm1 + diag.network + diag.consecutive_ones +
                 diag.dual_rational + diag.tu_network_block +
                 diag.tu_network_transpose;
    snap_integer_bounds(lp, col_lo, col_hi, opts.tol, diag.bounds_snapped);
    return diag;
}

}  // namespace sor::search
