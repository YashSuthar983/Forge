#include "sor/search/implied_int.hpp"
#include "sor/core/work_clock.hpp"
#include "sor/sparse/csc.hpp"

#include <chrono>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sor::search {

namespace {
// Deadline checked inside each rule's outer loop, not merely between rules.
//
// Between-rule checks are not enough: a SINGLE rule ran 49.2 s on MIPLIB2017
// netdiversion (119589 x 129180) against a 10 s solver limit -- 49204 ms ->
// 52.6 ms with --no-implied-int. Checks have to be where the loop is.
struct IiDeadline {
    sor::core::SolverClock::time_point t0{};
    double limit = 0.0;
    const core::CancelToken* cancel = nullptr;
    bool over() const {
        return core::cancel_requested(cancel) || (limit > 0.0 &&
               std::chrono::duration<double>(
                   sor::core::SolverClock::now() - t0).count() > limit);
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

inline std::size_t szo(core::Offset k) { return static_cast<std::size_t>(k); }

// The matrix with duplicate (row, column) entries summed and exact zeros
// dropped, in both orientations. Every rule below reads coefficients from
// here, so a column split over two entries of one row is judged by its sum,
// and no rule rescans the matrix per column: each one is O(nnz). The rules
// used to find column j's coefficients by walking every row of the matrix,
// O(n * nnz), and spent their whole time budget on large models finding
// nothing.
struct MergedMatrix {
    std::vector<core::Offset> rptr, cptr;
    std::vector<Index> rcol, crow;
    std::vector<f64> rval, cval;
};

MergedMatrix build_merged(const model::LpProblem& lp) {
    const Index m = lp.n_rows(), n = lp.n_cols();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    MergedMatrix M;
    M.rptr.assign(sz(m) + 1, 0);
    M.rcol.reserve(av.size());
    M.rval.reserve(av.size());
    std::vector<Index> stamp(sz(n), -1);
    std::vector<std::size_t> pos(sz(n), 0);
    for (Index i = 0; i < m; ++i) {
        const std::size_t start = M.rcol.size();
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[szo(k)];
            if (stamp[sz(j)] == i) {
                M.rval[pos[sz(j)]] += av[szo(k)];
            } else {
                stamp[sz(j)] = i;
                pos[sz(j)] = M.rcol.size();
                M.rcol.push_back(j);
                M.rval.push_back(av[szo(k)]);
            }
        }
        std::size_t w = start;
        for (std::size_t t = start; t < M.rcol.size(); ++t) {
            if (M.rval[t] == 0.0) continue;
            M.rcol[w] = M.rcol[t];
            M.rval[w] = M.rval[t];
            ++w;
        }
        M.rcol.resize(w);
        M.rval.resize(w);
        M.rptr[sz(i) + 1] = static_cast<core::Offset>(w);
    }
    M.cptr.assign(sz(n) + 1, 0);
    for (const Index j : M.rcol) ++M.cptr[sz(j) + 1];
    for (Index j = 0; j < n; ++j) M.cptr[sz(j) + 1] += M.cptr[sz(j)];
    M.crow.resize(M.rcol.size());
    M.cval.resize(M.rcol.size());
    std::vector<core::Offset> fill(M.cptr.begin(), M.cptr.end() - 1);
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = M.rptr[sz(i)]; k < M.rptr[sz(i) + 1]; ++k) {
            const Index j = M.rcol[szo(k)];
            const std::size_t at = szo(fill[sz(j)]++);
            M.crow[at] = i;      // ascending row order within each column
            M.cval[at] = M.rval[szo(k)];
        }
    return M;
}

// The box the rules reason about: the caller's working bounds when it passes
// them (mip_presolve's box, already tighter than lp.col_lo/col_hi), else the
// model's own. Integrality is implied for the problem actually being solved,
// so its bounds are the ones that have to be integral.
struct Box {
    const std::vector<f64>& lo;
    const std::vector<f64>& hi;
    f64 tol;
    bool fixed_int(Index j) const {
        const f64 l = lo[sz(j)], h = hi[sz(j)];
        return std::isfinite(l) && std::isfinite(h) && std::fabs(l - h) <= tol &&
               nearly_int(l, tol);
    }
    // Every finite bound integral (an infinite one constrains nothing).
    bool int_bounds(Index j) const {
        const f64 l = lo[sz(j)], h = hi[sz(j)];
        return (!std::isfinite(l) || nearly_int(l, tol)) &&
               (!std::isfinite(h) || nearly_int(h, tol));
    }
};

// Equality rows with an integral right-hand side, x_j with coefficient +-1
// and every other term an integral multiple of an integer (or integer-fixed)
// column: x_j = rhs - (integer) is integral at EVERY feasible point.
//
// Incremental: bad[i] counts row i's terms that are not an integral
// coefficient on an integer-like column. A row serves column j when it is
// such an equality, j's coefficient is +-1 and j is its only bad term; j is
// marked once every row it appears in serves it. Marking j removes one bad
// term from each of its rows, and a row whose count drops to one starts
// serving its last bad column. Each row reaches a count of one at most once,
// so the whole cascade is O(nnz).
std::uint64_t infer_equality_pm1(model::LpProblem& p, const MergedMatrix& M,
                                 const Box& b, const std::vector<char>& eq,
                                 f64 tol, const IiDeadline& dl) {
    const Index m = p.n_rows(), n = p.n_cols();
    const auto int_like = [&](Index j) { return p.is_integer[sz(j)] || b.fixed_int(j); };
    const auto candidate = [&](Index j) { return !p.is_integer[sz(j)] && !b.fixed_int(j); };
    const auto deg = [&](Index j) { return M.cptr[sz(j) + 1] - M.cptr[sz(j)]; };
    std::vector<Index> bad(sz(m), 0);
    for (Index i = 0; i < m; ++i)
        for (core::Offset t = M.rptr[sz(i)]; t < M.rptr[sz(i) + 1]; ++t)
            if (!int_like(M.rcol[szo(t)]) || !nearly_int(M.rval[szo(t)], tol)) ++bad[sz(i)];
    const auto serves = [&](Index i, f64 a) {
        return eq[sz(i)] && std::fabs(std::fabs(a) - 1.0) <= tol && bad[sz(i)] == 1;
    };
    std::vector<core::Offset> served(sz(n), 0);
    std::vector<Index> queue;
    for (Index j = 0; j < n; ++j) {
        if (!candidate(j) || deg(j) == 0) continue;
        for (core::Offset k = M.cptr[sz(j)]; k < M.cptr[sz(j) + 1]; ++k)
            if (serves(M.crow[szo(k)], M.cval[szo(k)])) ++served[sz(j)];
        if (served[sz(j)] == deg(j)) queue.push_back(j);
    }
    std::uint64_t added = 0;
    for (std::size_t head = 0; head < queue.size(); ++head) {
        if ((head & kIiPollMask) == 0 && dl.over()) break;
        const Index j = queue[head];
        if (p.is_integer[sz(j)]) continue;
        p.is_integer[sz(j)] = true;
        ++added;
        // j's terms are +-1 on a now-integer column: no longer bad.
        for (core::Offset k = M.cptr[sz(j)]; k < M.cptr[sz(j) + 1]; ++k) {
            const Index i = M.crow[szo(k)];
            if (--bad[sz(i)] != 1) continue;
            for (core::Offset t = M.rptr[sz(i)]; t < M.rptr[sz(i) + 1]; ++t) {
                const Index q = M.rcol[szo(t)];
                if (int_like(q) && nearly_int(M.rval[szo(t)], tol)) continue;
                // The row's one remaining bad term.
                if (candidate(q) && serves(i, M.rval[szo(t)]) &&
                    ++served[sz(q)] == deg(q))
                    queue.push_back(q);
                break;
            }
        }
    }
    return added;
}

struct ComponentMarks {
    std::uint64_t network = 0;
    std::uint64_t consecutive_ones = 0;
};

// Network and interval (consecutive-ones) matrices are totally unimodular.
// Both rules are about a WHOLE equality-connected continuous subsystem:
// collect the continuous columns reachable from a seed through equality rows,
// and the rows they touch. With the integer (and integer-fixed) columns at
// any integral values, those continuous columns solve
//     N x = b - (integer terms),   lo <= x <= hi,
// and if N is TU and the right-hand side and every bound are integral, an
// optimal vertex is integral: marking the whole subsystem integer keeps an
// optimum. That needs, for the subsystem as a unit,
//   - every touched row an equality with an integral side,
//   - every term outside the subsystem an integral coefficient on an integer
//     or integer-fixed column,
//   - every finite bound of every subsystem column integral (working box),
//   - N network (each column at most one +1 and one -1) or interval (each
//     column all +1 on a run of consecutive equality rows).
// A column checked only against its own rows is not enough: a companion with
// a fractional bound, a fractional coefficient on an integer column, or a
// further row with a non-TU continuous column all force fractional values.
// The consecutive-ones rule used to check row by row and marked such columns
// (x + 0.5 y + z = 3 with y binary: x = 2.5 at the optimum).
ComponentMarks infer_tu_components(model::LpProblem& p, const MergedMatrix& M,
                                   const Box& b, const std::vector<char>& eq,
                                   bool want_network, bool want_c1, f64 tol,
                                   const IiDeadline& dl) {
    ComponentMarks out;
    if (!want_network && !want_c1) return out;
    const Index m = p.n_rows(), n = p.n_cols();
    const auto int_like = [&](Index j) { return p.is_integer[sz(j)] || b.fixed_int(j); };
    const auto candidate = [&](Index j) { return !p.is_integer[sz(j)] && !b.fixed_int(j); };
    std::vector<Index> eq_rank(sz(m), -1);
    {
        Index r = 0;
        for (Index i = 0; i < m; ++i)
            if (eq[sz(i)]) eq_rank[sz(i)] = r++;
    }
    std::vector<Index> comp_of(sz(n), -1), row_seen(sz(m), -1);
    std::vector<Index> columns, rows;
    for (Index seed = 0; seed < n; ++seed) {
        if ((static_cast<std::uint64_t>(seed) & kIiPollMask) == 0 && dl.over()) break;
        if (comp_of[sz(seed)] >= 0 || !candidate(seed)) continue;
        columns.assign(1, seed);
        rows.clear();
        comp_of[sz(seed)] = seed;
        for (std::size_t head = 0; head < columns.size(); ++head) {
            const Index j = columns[head];
            for (core::Offset k = M.cptr[sz(j)]; k < M.cptr[sz(j) + 1]; ++k) {
                const Index i = M.crow[szo(k)];
                if (row_seen[sz(i)] == seed) continue;
                row_seen[sz(i)] = seed;
                rows.push_back(i);
                if (!eq[sz(i)]) continue;   // touched, and invalidates below
                for (core::Offset t = M.rptr[sz(i)]; t < M.rptr[sz(i) + 1]; ++t) {
                    const Index q = M.rcol[szo(t)];
                    if (comp_of[sz(q)] >= 0 || !candidate(q)) continue;
                    comp_of[sz(q)] = seed;
                    columns.push_back(q);
                }
            }
        }
        bool valid = !rows.empty();
        for (const Index i : rows) {
            if (!eq[sz(i)]) { valid = false; break; }
            for (core::Offset t = M.rptr[sz(i)]; t < M.rptr[sz(i) + 1] && valid; ++t) {
                const Index q = M.rcol[szo(t)];
                if (comp_of[sz(q)] == seed) continue;
                if (!int_like(q) || !nearly_int(M.rval[szo(t)], tol)) valid = false;
            }
            if (!valid) break;
        }
        for (const Index j : columns)
            if (valid && !b.int_bounds(j)) valid = false;
        if (!valid) continue;
        bool network = want_network, interval = want_c1;
        for (const Index j : columns) {
            int plus = 0, minus = 0;
            bool other = false;
            Index rmin = std::numeric_limits<Index>::max(), rmax = -1, count = 0;
            for (core::Offset k = M.cptr[sz(j)]; k < M.cptr[sz(j) + 1]; ++k) {
                const f64 a = M.cval[szo(k)];
                if (std::fabs(a - 1.0) <= tol) ++plus;
                else if (std::fabs(a + 1.0) <= tol) ++minus;
                else other = true;
                const Index r = eq_rank[sz(M.crow[szo(k)])];
                rmin = std::min(rmin, r);
                rmax = std::max(rmax, r);
                ++count;
            }
            if (other || plus > 1 || minus > 1 || plus + minus == 0) network = false;
            if (other || minus > 0 || count == 0 || rmax - rmin + 1 != count) interval = false;
            if (!network && !interval) break;
        }
        if (!network && !interval) continue;
        for (const Index j : columns) p.is_integer[sz(j)] = true;
        if (network) out.network += columns.size();
        else out.consecutive_ones += columns.size();
    }
    return out;
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

inline f64 gcd_integral(f64 a, f64 b) {
    // Exact for integral doubles below 2^53 (fmod of such values is exact).
    while (b != 0.0) {
        const f64 t = std::fmod(a, b);
        a = b;
        b = t;
    }
    return a;
}

// Corollary 3.3 (dual detection). Continuous x_k whose every row is an
// equality or a one-sided row with integral data, in which k's coefficient
// divides every coefficient and the side (|a_k| = gcd of the row), and whose
// every other term sits on an integer or integer-fixed column. With those
// columns at integral values each row bounds x_k by an integer, and an
// equality pins it to one. Without an equality, x_k ranges over an interval
// whose endpoints are those row bounds AND x_k's own bounds; an optimum sits
// at an endpoint (any integer of the interval when c_k = 0), so x_k is
// integral at some optimum only when its own finite bounds are integral too.
// That check was missing: x_k in [0.5, 10] with x_k >= y (y integer) and
// c_k > 0 was marked, its lower bound snapped to 1, and the optimum 0.5 cut
// off (the netlib d2q06c wrong optimum that the pure-LP guard in bab.cpp
// papered over).
//
// Incremental like infer_equality_pm1: cont[i] counts row i's terms on
// columns that are not integer-like; a usable row serves k when k is its
// only such term.
std::uint64_t infer_dual_rational(model::LpProblem& p, const MergedMatrix& M,
                                  const Box& b, const std::vector<char>& eq,
                                  f64 tol, const IiDeadline& dl) {
    const Index m = p.n_rows(), n = p.n_cols();
    const auto int_like = [&](Index j) { return p.is_integer[sz(j)] || b.fixed_int(j); };
    const auto candidate = [&](Index j) { return !p.is_integer[sz(j)] && !b.fixed_int(j); };
    const auto deg = [&](Index j) { return M.cptr[sz(j) + 1] - M.cptr[sz(j)]; };
    constexpr f64 kExactInt = 9007199254740992.0;   // 2^53
    std::vector<char> usable(sz(m), 0);
    std::vector<f64> g(sz(m), 0.0), side(sz(m), 0.0);
    for (Index i = 0; i < m; ++i) {
        const f64 rl = p.row_lo[sz(i)], rh = p.row_hi[sz(i)];
        f64 s;
        if (eq[sz(i)]) s = rl;
        else if (std::isfinite(rl) && !std::isfinite(rh)) s = rl;
        else if (!std::isfinite(rl) && std::isfinite(rh)) s = rh;
        else continue;   // ranged (or free) row
        if (!nearly_int(s, tol)) continue;
        f64 gg = 0.0;
        bool ok = M.rptr[sz(i) + 1] > M.rptr[sz(i)];
        for (core::Offset t = M.rptr[sz(i)]; t < M.rptr[sz(i) + 1] && ok; ++t) {
            const f64 a = M.rval[szo(t)];
            if (!nearly_int(a, tol) || std::fabs(a) >= kExactInt) ok = false;
            else gg = gcd_integral(gg, std::fabs(std::round(a)));
        }
        if (!ok || gg == 0.0) continue;
        usable[sz(i)] = 1;
        g[sz(i)] = gg;
        side[sz(i)] = s;
    }
    std::vector<Index> cont(sz(m), 0);
    for (Index i = 0; i < m; ++i)
        for (core::Offset t = M.rptr[sz(i)]; t < M.rptr[sz(i) + 1]; ++t)
            if (!int_like(M.rcol[szo(t)])) ++cont[sz(i)];
    const auto serves = [&](Index i, f64 a) {
        return usable[sz(i)] && cont[sz(i)] == 1 &&
               std::fabs(std::fabs(std::round(a)) - g[sz(i)]) <= tol &&
               nearly_int(side[sz(i)] / a, tol);
    };
    // Own-bound condition: integral finite bounds, or an equality row pins
    // x_k to an integer at every feasible point anyway.
    const auto bounds_ok = [&](Index k) {
        if (b.int_bounds(k)) return true;
        for (core::Offset t = M.cptr[sz(k)]; t < M.cptr[sz(k) + 1]; ++t)
            if (eq[sz(M.crow[szo(t)])]) return true;
        return false;
    };
    std::vector<core::Offset> served(sz(n), 0);
    std::vector<Index> queue;
    for (Index k = 0; k < n; ++k) {
        if (!candidate(k) || deg(k) == 0) continue;
        for (core::Offset t = M.cptr[sz(k)]; t < M.cptr[sz(k) + 1]; ++t)
            if (serves(M.crow[szo(t)], M.cval[szo(t)])) ++served[sz(k)];
        if (served[sz(k)] == deg(k) && bounds_ok(k)) queue.push_back(k);
    }
    std::uint64_t added = 0;
    for (std::size_t head = 0; head < queue.size(); ++head) {
        if ((head & kIiPollMask) == 0 && dl.over()) break;
        const Index k = queue[head];
        if (p.is_integer[sz(k)]) continue;
        p.is_integer[sz(k)] = true;
        ++added;
        for (core::Offset t = M.cptr[sz(k)]; t < M.cptr[sz(k) + 1]; ++t) {
            const Index i = M.crow[szo(t)];
            if (--cont[sz(i)] != 1) continue;
            for (core::Offset u = M.rptr[sz(i)]; u < M.rptr[sz(i) + 1]; ++u) {
                const Index q = M.rcol[szo(u)];
                if (int_like(q)) continue;
                // The row's one remaining continuous term.
                if (candidate(q) && serves(i, M.rval[szo(u)]) &&
                    ++served[sz(q)] == deg(q) && bounds_ok(q))
                    queue.push_back(q);
                break;
            }
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
    // Deadline between rules and inside each rule's loop.
    const auto ii_t0 = sor::core::SolverClock::now();
    const IiDeadline dl{ii_t0, opts.time_limit_s, opts.cancel};
    const auto ii_over = [&]() {
        return core::cancel_requested(opts.cancel) || (opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(
                   sor::core::SolverClock::now() - ii_t0).count() >
                   opts.time_limit_s);
    };
    if (ii_over()) { diag.aborted_on_time = 1; return diag; }
    const Index n = lp.n_cols(), m = lp.n_rows();
    if (lp.is_integer.size() != sz(n)) lp.is_integer.assign(sz(n), false);
    // These rules change the discrete feasible set. A near-zero coefficient,
    // near-unit matrix entry, near-integer RHS, or near-equality is not an
    // exact TU/equality pattern: over a wide domain the discarded difference
    // can force a fractional value. Use exact representable data for the
    // structural proof; opts.tol remains only for conservative bound snaps.
    constexpr f64 proof_tol = 0.0;
    const bool own_box = col_lo != nullptr && col_hi != nullptr &&
                         col_lo->size() == sz(n) && col_hi->size() == sz(n);
    const Box box{own_box ? *col_lo : lp.col_lo, own_box ? *col_hi : lp.col_hi,
                  proof_tol};
    const MergedMatrix M = build_merged(lp);
    std::vector<char> eq(sz(m), 0);
    for (Index i = 0; i < m; ++i) eq[sz(i)] = is_eq_row(lp, i, proof_tol) ? 1 : 0;

    if (opts.equality_pm1)
        diag.equality_pm1 = infer_equality_pm1(lp, M, box, eq, proof_tol, dl);
    if (ii_over()) { diag.aborted_on_time = 1; return diag; }
    if (opts.network || opts.consecutive_ones) {
        const auto c = infer_tu_components(lp, M, box, eq, opts.network,
                                           opts.consecutive_ones, proof_tol, dl);
        diag.network = c.network;
        diag.consecutive_ones = c.consecutive_ones;
    }
    if (ii_over()) { diag.aborted_on_time = 1; return diag; }
    if (opts.dual_rational)
        diag.dual_rational = infer_dual_rational(lp, M, box, eq, proof_tol, dl);
    if (ii_over()) { diag.aborted_on_time = 1; return diag; }
    if (opts.tu_network_block) {
        diag.tu_network_block = infer_tu_network_block(lp, proof_tol, dl);
        diag.tu_network_transpose = infer_tu_network_transpose(lp, proof_tol, dl);
    }
    // Cascade: subsystem marks may unlock more +-1 / dual inferences.
    const std::uint64_t tu_extra = diag.network + diag.consecutive_ones +
                                   diag.tu_network_block +
                                   diag.tu_network_transpose;
    if (opts.equality_pm1 && tu_extra > 0 && !ii_over())
        diag.equality_pm1 += infer_equality_pm1(lp, M, box, eq, proof_tol, dl);
    if (opts.dual_rational && tu_extra > 0 && !ii_over())
        diag.dual_rational += infer_dual_rational(lp, M, box, eq, proof_tol, dl);
    diag.total = diag.equality_pm1 + diag.network + diag.consecutive_ones +
                 diag.dual_rational + diag.tu_network_block +
                 diag.tu_network_transpose;
    snap_integer_bounds(lp, col_lo, col_hi, opts.tol, diag.bounds_snapped);
    return diag;
}

}  // namespace sor::search
