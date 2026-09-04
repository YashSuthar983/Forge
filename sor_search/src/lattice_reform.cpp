// SOR — lattice reformulation implementation (AHL / Aardal–Wolsey).
#include "sor/search/lattice_reform.hpp"

#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sor::search {
namespace {

using core::f64;
using core::Index;
using i64 = std::int64_t;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
using i128 = __int128;
#pragma GCC diagnostic pop
#else
#error "lattice_reform requires __int128"
#endif

constexpr long double kDelta = 0.75L;  // classical LLL Lovász parameter

i64 i128_to_i64_exact(i128 v) {
    if (v > static_cast<i128>(std::numeric_limits<i64>::max()) ||
        v < static_cast<i128>(std::numeric_limits<i64>::min())) {
        throw std::overflow_error("lattice: int64 overflow");
    }
    return static_cast<i64>(v);
}

bool near_integer(f64 v, f64 tol = 1e-9) {
    if (!std::isfinite(v)) return false;
    return std::fabs(v - std::round(v)) <= tol;
}

i64 round_exact(f64 v) {
    return static_cast<i64>(std::llround(v));
}

// Dense integer matrix stored column-major: cols[j][i] = entry (i,j).
using DenseCols = std::vector<std::vector<i64>>;

long double dot_ld(const std::vector<long double>& a,
                   const std::vector<long double>& b) {
    long double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// Classical LLL on integer column vectors (Lenstra–Lenstra–Lovász 1982).
// Gram–Schmidt in long double; basis updates exact int64.
void lll_reduce(DenseCols& B, long double delta = kDelta) {
    const int n_rows = static_cast<int>(B.empty() ? 0 : B[0].size());
    const int n_cols = static_cast<int>(B.size());
    if (n_cols <= 1 || n_rows == 0) return;

    std::vector<std::vector<long double>> bstar(
        static_cast<std::size_t>(n_cols),
        std::vector<long double>(static_cast<std::size_t>(n_rows), 0));
    std::vector<std::vector<long double>> mu(
        static_cast<std::size_t>(n_cols),
        std::vector<long double>(static_cast<std::size_t>(n_cols), 0));
    std::vector<long double> Bnorm(static_cast<std::size_t>(n_cols), 0);

    auto recompute_gs = [&](int from) {
        for (int i = from; i < n_cols; ++i) {
            for (int r = 0; r < n_rows; ++r)
                bstar[static_cast<std::size_t>(i)][static_cast<std::size_t>(r)] =
                    static_cast<long double>(
                        B[static_cast<std::size_t>(i)][static_cast<std::size_t>(r)]);
            for (int j = 0; j < i; ++j) {
                const long double num = dot_ld(
                    bstar[static_cast<std::size_t>(i)],
                    bstar[static_cast<std::size_t>(j)]);
                const long double den = Bnorm[static_cast<std::size_t>(j)];
                mu[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    (den == 0) ? 0 : num / den;
                for (int r = 0; r < n_rows; ++r) {
                    bstar[static_cast<std::size_t>(i)][static_cast<std::size_t>(r)] -=
                        mu[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                        bstar[static_cast<std::size_t>(j)][static_cast<std::size_t>(r)];
                }
            }
            Bnorm[static_cast<std::size_t>(i)] =
                dot_ld(bstar[static_cast<std::size_t>(i)],
                       bstar[static_cast<std::size_t>(i)]);
        }
    };

    auto size_reduce = [&](int k, int j) {
        const long double m =
            mu[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
        if (std::fabs(m) <= 0.5L) return;
        const i64 q = static_cast<i64>(std::llround(static_cast<double>(m)));
        if (q == 0) return;
        for (int r = 0; r < n_rows; ++r) {
            // q * B[j][r] as i64*i64 is signed-overflow UB for large bases;
            // compute in i128 and narrow exactly (throws -> ahl_extract
            // catches -> nullopt -> caller falls back, never a wrong basis).
            const i128 v = static_cast<i128>(q) *
                           static_cast<i128>(
                               B[static_cast<std::size_t>(j)][static_cast<std::size_t>(r)]);
            B[static_cast<std::size_t>(k)][static_cast<std::size_t>(r)] =
                i128_to_i64_exact(
                    static_cast<i128>(
                        B[static_cast<std::size_t>(k)][static_cast<std::size_t>(r)]) -
                    v);
        }
        mu[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] -=
            static_cast<long double>(q);
        for (int i = 0; i < j; ++i) {
            mu[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] -=
                static_cast<long double>(q) *
                mu[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
        }
    };

    recompute_gs(0);
    int k = 1;
    int guard = 0;
    const int guard_max = std::max(1000, 50 * n_cols * n_cols);
    while (k < n_cols) {
        if (++guard > guard_max)
            throw std::runtime_error("lattice: LLL iteration cap");
        size_reduce(k, k - 1);
        const long double muk =
            mu[static_cast<std::size_t>(k)][static_cast<std::size_t>(k - 1)];
        const long double left = Bnorm[static_cast<std::size_t>(k)];
        const long double right =
            (delta - muk * muk) * Bnorm[static_cast<std::size_t>(k - 1)];
        if (left >= right * (1 - 1e-15L)) {
            for (int j = k - 2; j >= 0; --j) size_reduce(k, j);
            ++k;
        } else {
            std::swap(B[static_cast<std::size_t>(k)],
                      B[static_cast<std::size_t>(k - 1)]);
            recompute_gs(std::max(0, k - 1));
            k = std::max(1, k - 1);
        }
    }
}

struct Extracted {
    DenseCols Q;          // n × k  (k = n - m)
    std::vector<i64> x0;  // n
    i64 N1_sign = 1;      // ±N1 observed
    // false + !proven_infeasible: extraction was inconsistent (insufficient
    // N1/N2) -- caller should grow the penalties and retry.
    bool feasible = true;
    // false + true: the AHL infeasibility signal (paper eq. (1), §1.1) was
    // observed AND cross-verified against the bottom block -- Ax=b has no
    // integer solution, period. Retrying with larger N1/N2 cannot change a
    // gcd obstruction, so the caller must not retry this case.
    bool proven_infeasible = false;
};

// Build AHL augmented matrix and extract Q, x0 after LLL.
// A is m × n (row-major flat), b length m. All entries integer.
std::optional<Extracted>
ahl_extract(const std::vector<i64>& A_flat, int m, int n,
            const std::vector<i64>& b, i64 N1, i64 N2) {
    if (m <= 0 || n <= m) return std::nullopt;
    const int nrows = n + m + 1;
    const int ncols = n + 1;
    DenseCols B(static_cast<std::size_t>(ncols),
                std::vector<i64>(static_cast<std::size_t>(nrows), 0));

    // Columns 0..n-1: [e_j ; 0 ; N2 * A_{:j}]
    for (int j = 0; j < n; ++j) {
        B[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)] = 1;
        for (int i = 0; i < m; ++i) {
            const i128 v = static_cast<i128>(N2) *
                           static_cast<i128>(A_flat[static_cast<std::size_t>(i * n + j)]);
            B[static_cast<std::size_t>(j)][static_cast<std::size_t>(n + 1 + i)] =
                i128_to_i64_exact(v);
        }
    }
    // Last column: [0 ; N1 ; -N2 * b]
    B[static_cast<std::size_t>(n)][static_cast<std::size_t>(n)] = N1;
    for (int i = 0; i < m; ++i) {
        const i128 v = -static_cast<i128>(N2) *
                       static_cast<i128>(b[static_cast<std::size_t>(i)]);
        B[static_cast<std::size_t>(n)][static_cast<std::size_t>(n + 1 + i)] =
            i128_to_i64_exact(v);
    }

    try {
        lll_reduce(B);
    } catch (...) {
        return std::nullopt;
    }

    // Identify the particular-solution column: row n equals ±N1, and the
    // bottom m block equals ±N2-scaled? Paper: first n-m+1 short columns give
    // [[Q, x0],[0,±N1],[0,0...]] with bottom A-block zero for kernel cols.
    // After reduction, scan for the column with |B[n]| == N1.
    int x0_col = -1;
    int infeas_col = -1;
    i64 infeas_k = 0;
    for (int j = 0; j < ncols; ++j) {
        const i64 mid = B[static_cast<std::size_t>(j)][static_cast<std::size_t>(n)];
        if (mid == N1 || mid == -N1) {
            x0_col = j;
            break;
        }
        // Paper's infeasibility signal: no x0 with A x0 = b exists iff this
        // entry comes out as ±k*N1 for some integer k > 1 (eq. (1), §1.1).
        // Cross-check against the bottom block (must equal k * (-N2*b), the
        // same multiple) before trusting it, rather than treating any
        // coincidental multiple of a (possibly still-too-small) N1 as proof
        // -- retrying with larger N1/N2 is pointless for a genuine gcd
        // obstruction, but a false positive here would wrongly claim
        // Infeasible, so this stays conservative: verified match or nothing.
        if (mid != 0 && N1 != 0 && mid % N1 == 0) {
            const i64 kk = mid / N1;
            if (kk > 1 || kk < -1) {
                bool bottom_matches = true;
                for (int i = 0; i < m; ++i) {
                    const i128 want = -static_cast<i128>(kk) *
                                      static_cast<i128>(N2) *
                                      static_cast<i128>(b[static_cast<std::size_t>(i)]);
                    if (static_cast<i128>(B[static_cast<std::size_t>(j)]
                                            [static_cast<std::size_t>(n + 1 + i)]) != want) {
                        bottom_matches = false;
                        break;
                    }
                }
                if (bottom_matches && infeas_col < 0) {
                    infeas_col = j;
                    infeas_k = kk;
                }
            }
        }
    }
    if (x0_col < 0) {
        if (infeas_col >= 0) {
            Extracted out;
            out.feasible = false;
            out.proven_infeasible = true;  // genuine: Ax=b has no integer solution
            (void)infeas_k;
            return out;
        }
        return std::nullopt;
    }

    Extracted out;
    out.N1_sign =
        B[static_cast<std::size_t>(x0_col)][static_cast<std::size_t>(n)];
    // Feasibility: paper says if no integer solution, mid becomes ±k·N1, k>1.
    // We already required |mid|==N1, so k=1. Also check bottom matches.
    out.x0.assign(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        out.x0[static_cast<std::size_t>(i)] =
            B[static_cast<std::size_t>(x0_col)][static_cast<std::size_t>(i)];
    }
    // Verify A x0 = b exactly.
    for (int i = 0; i < m; ++i) {
        i128 s = 0;
        for (int j = 0; j < n; ++j) {
            s += static_cast<i128>(A_flat[static_cast<std::size_t>(i * n + j)]) *
                 static_cast<i128>(out.x0[static_cast<std::size_t>(j)]);
        }
        if (s != static_cast<i128>(b[static_cast<std::size_t>(i)])) {
            out.feasible = false;
            return out;  // signal: extracted but inconsistent → retry
        }
    }

    // Kernel columns: remaining columns with mid-row 0 and bottom m block 0.
    const int kdim = n - m;
    out.Q.assign(static_cast<std::size_t>(n),
                 std::vector<i64>(static_cast<std::size_t>(kdim), 0));
    int kfill = 0;
    for (int j = 0; j < ncols && kfill < kdim; ++j) {
        if (j == x0_col) continue;
        const i64 mid = B[static_cast<std::size_t>(j)][static_cast<std::size_t>(n)];
        if (mid != 0) continue;
        bool bottom_zero = true;
        for (int i = 0; i < m; ++i) {
            if (B[static_cast<std::size_t>(j)][static_cast<std::size_t>(n + 1 + i)] != 0) {
                bottom_zero = false;
                break;
            }
        }
        if (!bottom_zero) continue;
        for (int i = 0; i < n; ++i) {
            out.Q[static_cast<std::size_t>(i)][static_cast<std::size_t>(kfill)] =
                B[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
        }
        // Verify A Q_col = 0.
        for (int i = 0; i < m; ++i) {
            i128 s = 0;
            for (int r = 0; r < n; ++r) {
                s += static_cast<i128>(A_flat[static_cast<std::size_t>(i * n + r)]) *
                     static_cast<i128>(
                         out.Q[static_cast<std::size_t>(r)][static_cast<std::size_t>(kfill)]);
            }
            if (s != 0) {
                return std::nullopt;
            }
        }
        ++kfill;
    }
    if (kfill != kdim) return std::nullopt;
    return out;
}

i64 max_abs_entry(const std::vector<i64>& A_flat, const std::vector<i64>& b) {
    i64 m = 1;
    for (i64 v : A_flat) m = std::max(m, v < 0 ? -v : v);
    for (i64 v : b) m = std::max(m, v < 0 ? -v : v);
    return m;
}

}  // namespace

std::optional<LatticeReformMap>
try_lattice_reform(const model::LpProblem& lp, const LatticeReformOptions& opts) {
    const Index m_all = lp.n_rows();
    const Index n_all = lp.n_cols();
    if (m_all <= 0 || n_all <= 0) return std::nullopt;
    if (static_cast<int>(lp.is_integer.size()) != n_all) return std::nullopt;

    // All rows must be equalities.
    for (Index i = 0; i < m_all; ++i) {
        if (!(lp.row_lo[static_cast<std::size_t>(i)] ==
              lp.row_hi[static_cast<std::size_t>(i)])) {
            return std::nullopt;
        }
        if (!near_integer(lp.row_lo[static_cast<std::size_t>(i)])) return std::nullopt;
    }

    LatticeReformMap map;
    map.n_rows_eq = m_all;

    std::vector<bool> is_fixed(static_cast<std::size_t>(n_all), false);
    std::vector<f64> fixed_val(static_cast<std::size_t>(n_all), 0);
    for (Index j = 0; j < n_all; ++j) {
        const f64 lo = lp.col_lo[static_cast<std::size_t>(j)];
        const f64 hi = lp.col_hi[static_cast<std::size_t>(j)];
        if (lo == hi) {
            if (!near_integer(lo)) return std::nullopt;
            is_fixed[static_cast<std::size_t>(j)] = true;
            fixed_val[static_cast<std::size_t>(j)] = lo;
            map.fixed_cols.push_back(j);
            map.fixed_vals.push_back(lo);
        }
    }

    // RHS after eliminating fixed columns: b' = b - A_fixed x_fixed
    std::vector<f64> b_work(static_cast<std::size_t>(m_all));
    for (Index i = 0; i < m_all; ++i)
        b_work[static_cast<std::size_t>(i)] = lp.row_lo[static_cast<std::size_t>(i)];

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (Index i = 0; i < m_all; ++i) {
        for (core::Offset p = rp[static_cast<std::size_t>(i)];
             p < rp[static_cast<std::size_t>(i) + 1]; ++p) {
            const Index j = ci[static_cast<std::size_t>(p)];
            if (is_fixed[static_cast<std::size_t>(j)]) {
                b_work[static_cast<std::size_t>(i)] -=
                    av[static_cast<std::size_t>(p)] *
                    fixed_val[static_cast<std::size_t>(j)];
            }
        }
    }
    for (f64& v : b_work) {
        if (!near_integer(v)) return std::nullopt;
        v = std::round(v);
    }

    // Partition free columns.
    std::vector<Index> free_int, free_cont;
    for (Index j = 0; j < n_all; ++j) {
        if (is_fixed[static_cast<std::size_t>(j)]) continue;
        if (lp.is_integer[static_cast<std::size_t>(j)])
            free_int.push_back(j);
        else
            free_cont.push_back(j);
    }

    if (!free_cont.empty()) {
        if (!opts.allow_force_continuous_zero) return std::nullopt;
        // Force continuous free columns to zero only when doing so is
        // objective-safe for proving an exact cover (c_j has the "prefer 0" sign).
        for (Index j : free_cont) {
            const f64 cj = lp.c[static_cast<std::size_t>(j)];
            if (lp.maximize) {
                if (cj > 1e-12) return std::nullopt;  // would want to increase
            } else {
                if (cj < -1e-12) return std::nullopt;  // would want to decrease
            }
            // Must be able to set to 0.
            const f64 lo = lp.col_lo[static_cast<std::size_t>(j)];
            const f64 hi = lp.col_hi[static_cast<std::size_t>(j)];
            if (!(lo <= 0.0 && 0.0 <= hi)) return std::nullopt;
            map.forced_zero_cols.push_back(j);
        }
    }
    map.exact_equivalence = map.forced_zero_cols.empty();

    // The μ-reformulation requires finite original bounds: the transformed
    // rows only bound μ through [l - x0, u - x0], so an infinite l/u leaves
    // μ genuinely unbounded and any starter box we invent would silently
    // bound an unbounded problem (a restriction masquerading as a proof).
    for (Index j : free_int) {
        if (!std::isfinite(lp.col_lo[static_cast<std::size_t>(j)]) ||
            !std::isfinite(lp.col_hi[static_cast<std::size_t>(j)])) {
            return std::nullopt;
        }
    }

    const int n = static_cast<int>(free_int.size());
    const int m = static_cast<int>(m_all);
    if (n <= m || n > opts.max_columns) return std::nullopt;

    // Build dense integer A on free integer columns.
    std::vector<i64> A_flat(static_cast<std::size_t>(m * n), 0);
    std::vector<i64> b_int(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i)
        b_int[static_cast<std::size_t>(i)] =
            round_exact(b_work[static_cast<std::size_t>(i)]);

    for (Index i = 0; i < m_all; ++i) {
        for (core::Offset p = rp[static_cast<std::size_t>(i)];
             p < rp[static_cast<std::size_t>(i) + 1]; ++p) {
            const Index j = ci[static_cast<std::size_t>(p)];
            if (is_fixed[static_cast<std::size_t>(j)]) continue;
            if (!lp.is_integer[static_cast<std::size_t>(j)]) {
                // Continuous column must not appear if we force it to 0 —
                // its contribution is already excluded from b_work only if
                // value is 0; verify coeff is finite and we'll set x=0.
                continue;
            }
            if (!near_integer(av[static_cast<std::size_t>(p)])) return std::nullopt;
            // Find local index.
            const auto it = std::find(free_int.begin(), free_int.end(), j);
            if (it == free_int.end()) return std::nullopt;
            const int jj = static_cast<int>(it - free_int.begin());
            A_flat[static_cast<std::size_t>(i * n + jj)] =
                round_exact(av[static_cast<std::size_t>(p)]);
        }
    }

    map.free_int_cols = free_int;
    map.kernel_dim = n - m;

    // Trivially-infeasible-but-well-formed 2-row marker on mu_0 (two
    // contradictory equalities): validate() requires lo <= hi on each row
    // INDIVIDUALLY, which this satisfies, while the row SYSTEM is jointly
    // unsatisfiable, so solve_milp on it reports Infeasible immediately
    // through the ordinary pipeline. Used both when Ax=b itself has no
    // integer solution (AHL's own ±k*N1 signal) and when the mu-projection
    // LP later proves the real relaxation empty.
    const int k_dim = map.kernel_dim;
    auto build_infeasible_marker = [&](const char* reason) {
        map.proven_lp_infeasible = true;
        model::LpProblem marker;
        marker.name = lp.name + "_lattice";
        marker.maximize = false;
        marker.A = sparse::from_triplets(
            2, static_cast<Index>(k_dim), {0, 1}, {0, 0}, {1.0, 1.0});
        marker.row_lo = {1.0, 0.0};
        marker.row_hi = {1.0, 0.0};
        marker.col_lo.assign(static_cast<std::size_t>(k_dim), 0.0);
        marker.col_hi.assign(static_cast<std::size_t>(k_dim), 1.0);
        marker.is_integer.assign(static_cast<std::size_t>(k_dim), true);
        marker.c.assign(static_cast<std::size_t>(k_dim), 0.0);
        marker.row_names = {"infeas_a", "infeas_b"};
        marker.col_names.resize(static_cast<std::size_t>(k_dim));
        for (int t = 0; t < k_dim; ++t)
            marker.col_names[static_cast<std::size_t>(t)] = "mu" + std::to_string(t);
        map.note += reason;
        map.transformed = std::move(marker);
        return map;
    };

    const i64 amax = max_abs_entry(A_flat, b_int);
    i64 N1 = static_cast<i64>(n - m + 1);
    i64 N2 = std::max<i64>(10, 10 * static_cast<i64>(n) * amax);

    std::optional<Extracted> extracted;
    for (int attempt = 0; attempt < opts.max_retries; ++attempt) {
        auto got = ahl_extract(A_flat, m, n, b_int, N1, N2);
        if (got && got->feasible) {
            extracted = std::move(got);
            map.note = "AHL N1=" + std::to_string(N1) + " N2=" + std::to_string(N2) +
                       " attempt=" + std::to_string(attempt + 1);
            break;
        }
        if (got && got->proven_infeasible) {
            // Verified gcd obstruction: Ax=b has no integer solution at all.
            // No amount of retrying with larger N1/N2 changes this.
            return build_infeasible_marker(
                "; Ax=b has no integer solution (AHL infeasibility signal)");
        }
        // Grow penalties and retry.
        if (N1 > std::numeric_limits<i64>::max() / 4) break;
        if (N2 > std::numeric_limits<i64>::max() / (4 * std::max<i64>(1, amax))) break;
        N1 *= 2;
        N2 *= 2;
    }
    if (!extracted) return std::nullopt;

    map.x0 = extracted->x0;
    map.Q = extracted->Q;  // n × kdim as row vectors of length kdim

    // Build transformed problem: k = n-m integer vars μ.
    // Constraints: for each free int col i: lo - x0_i <= Q_i·μ <= hi - x0_i
    const int k = map.kernel_dim;
    std::vector<Index> t_rows, t_cols;
    std::vector<f64> t_vals;
    std::vector<f64> row_lo, row_hi;
    row_lo.reserve(static_cast<std::size_t>(n));
    row_hi.reserve(static_cast<std::size_t>(n));

    Index row_id = 0;
    for (int i = 0; i < n; ++i) {
        const Index oj = free_int[static_cast<std::size_t>(i)];
        const f64 lo = lp.col_lo[static_cast<std::size_t>(oj)] -
                       static_cast<f64>(map.x0[static_cast<std::size_t>(i)]);
        const f64 hi = lp.col_hi[static_cast<std::size_t>(oj)] -
                       static_cast<f64>(map.x0[static_cast<std::size_t>(i)]);
        // Skip vacuous infinite free rows with all-zero Q row.
        bool any = false;
        for (int t = 0; t < k; ++t) {
            const i64 q =
                map.Q[static_cast<std::size_t>(i)][static_cast<std::size_t>(t)];
            if (q == 0) continue;
            any = true;
            t_rows.push_back(row_id);
            t_cols.push_back(static_cast<Index>(t));
            t_vals.push_back(static_cast<f64>(q));
        }
        if (!any) {
            // Q row zero: need lo <= 0 <= hi for feasibility of this coordinate.
            if (!(lo <= 0.0 && 0.0 <= hi)) return std::nullopt;
            continue;
        }
        row_lo.push_back(lo);
        row_hi.push_back(hi);
        ++row_id;
    }

    model::LpProblem tp;
    tp.name = lp.name + "_lattice";
    tp.maximize = lp.maximize;
    tp.A = sparse::from_triplets(row_id, static_cast<Index>(k), t_rows, t_cols, t_vals);
    tp.row_lo = std::move(row_lo);
    tp.row_hi = std::move(row_hi);

    // Exact per-coordinate mu bounds: LP-project the polytope
    // {row_lo <= Q*mu <= row_hi} onto each mu_t. Every INTEGER solution of
    // the (equality-eliminated) restricted system satisfies these real-
    // valued constraints, so the resulting box provably contains every
    // integer point of the restricted system -- this is the exact
    // projection, not a heuristic guess (replaces a fixed +-64 starter box,
    // which on markshare's ~10^80-point box was nowhere near tight enough
    // for B&B to find anything). Full-column-rank Q + the finite original
    // bounds enforced above guarantee this polytope is bounded, so a
    // generous finite dummy box for the projection LPs cannot silently
    // truncate it; hitting the dummy is treated as that guarantee having
    // failed somehow, and refuses the reform rather than risk a truncated
    // box reaching B&B.
    constexpr f64 kDummyBox = 1.0e7;
    model::LpProblem proj = tp;
    proj.col_lo.assign(static_cast<std::size_t>(k), -kDummyBox);
    proj.col_hi.assign(static_cast<std::size_t>(k), kDummyBox);
    proj.is_integer.assign(static_cast<std::size_t>(k), false);
    proj.c.assign(static_cast<std::size_t>(k), 0.0);
    proj.row_names.assign(static_cast<std::size_t>(row_id), "");
    proj.col_names.assign(static_cast<std::size_t>(k), "");

    tp.col_lo.assign(static_cast<std::size_t>(k), 0.0);
    tp.col_hi.assign(static_cast<std::size_t>(k), 0.0);
    bool lp_infeasible = false;
    engines::SimplexOptions sopts;
    for (int t = 0; t < k && !lp_infeasible; ++t) {
        f64 lo_v = 0.0, hi_v = 0.0;
        for (int dir = 0; dir < 2 && !lp_infeasible; ++dir) {
            proj.c.assign(static_cast<std::size_t>(k), 0.0);
            proj.c[static_cast<std::size_t>(t)] = 1.0;
            proj.maximize = (dir == 1);
            engines::SimplexDiagnostics sdiag;
            auto r = engines::solve_simplex(proj, sopts, sdiag);
            if (r.proposed_status == core::Status::Infeasible) {
                lp_infeasible = true;
                break;
            }
            if (r.proposed_status != core::Status::Optimal ||
                !std::isfinite(r.objective) ||
                std::fabs(r.objective) >= 0.5 * kDummyBox) {
                return std::nullopt;  // boundedness guarantee failed -- refuse
            }
            (dir == 0 ? lo_v : hi_v) = r.objective;
        }
        if (!lp_infeasible) {
            tp.col_lo[static_cast<std::size_t>(t)] = std::ceil(lo_v - 1e-7);
            tp.col_hi[static_cast<std::size_t>(t)] = std::floor(hi_v + 1e-7);
        }
    }

    if (lp_infeasible) {
        // The restricted system has no real point at all (proven directly,
        // independent of integrality) -- same marker-problem construction as
        // the AHL-level infeasibility signal above.
        return build_infeasible_marker(
            "; restricted LP relaxation infeasible (no real point)");
    }

    // A few rounds of row-wise bound propagation as a cheap, sound extra
    // tightening on top of the exact per-coordinate LP projection (a joint
    // polytope constraint can still imply more than any single coordinate's
    // own projection alone). Can only tighten further, never invalidate the
    // LP-derived box.
    for (int pass = 0; pass < 3; ++pass) {
        for (Index i = 0; i < row_id; ++i) {
            const auto& trp = tp.A.pattern.row_ptr();
            const auto& tci = tp.A.pattern.col_idx();
            const auto& tav = tp.A.vals;
            for (core::Offset p = trp[static_cast<std::size_t>(i)];
                 p < trp[static_cast<std::size_t>(i) + 1]; ++p) {
                const Index t = tci[static_cast<std::size_t>(p)];
                const f64 q = tav[static_cast<std::size_t>(p)];
                if (q == 0.0) continue;
                f64 others_max = 0.0, others_min = 0.0;
                for (core::Offset p2 = trp[static_cast<std::size_t>(i)];
                     p2 < trp[static_cast<std::size_t>(i) + 1]; ++p2) {
                    const Index j = tci[static_cast<std::size_t>(p2)];
                    if (j == t) continue;
                    const f64 qj = tav[static_cast<std::size_t>(p2)];
                    const f64 loj = tp.col_lo[static_cast<std::size_t>(j)];
                    const f64 hij = tp.col_hi[static_cast<std::size_t>(j)];
                    others_max += std::max(qj * loj, qj * hij);
                    others_min += std::min(qj * loj, qj * hij);
                }
                const f64 L = tp.row_lo[static_cast<std::size_t>(i)] - others_max;
                const f64 U = tp.row_hi[static_cast<std::size_t>(i)] - others_min;
                f64 a = L / q, b = U / q;
                if (a > b) std::swap(a, b);
                tp.col_lo[static_cast<std::size_t>(t)] =
                    std::max(tp.col_lo[static_cast<std::size_t>(t)], a);
                tp.col_hi[static_cast<std::size_t>(t)] =
                    std::min(tp.col_hi[static_cast<std::size_t>(t)], b);
            }
        }
    }
    for (int t = 0; t < k; ++t) {
        if (!(tp.col_lo[static_cast<std::size_t>(t)] <=
              tp.col_hi[static_cast<std::size_t>(t)])) {
            // The LP projection proved this coordinate's range nonempty;
            // integer-only row-propagation tightening it to empty here would
            // mean either a genuine integer infeasibility the propagation
            // found (which the LP box alone would not show) or a numerical
            // inconsistency -- either way, not safe to paper over with a
            // silently-widened box. Refuse the reform; the plain B&B path
            // handles the instance instead.
            return std::nullopt;
        }
        // Integer bounds: floor/ceil (idempotent -- already exact integers
        // from the LP projection step, this only guards the propagation
        // pass's own floating-point tightenings).
        tp.col_lo[static_cast<std::size_t>(t)] =
            std::ceil(tp.col_lo[static_cast<std::size_t>(t)] - 1e-9);
        tp.col_hi[static_cast<std::size_t>(t)] =
            std::floor(tp.col_hi[static_cast<std::size_t>(t)] + 1e-9);
    }
    tp.is_integer.assign(static_cast<std::size_t>(k), true);
    tp.c.assign(static_cast<std::size_t>(k), 0.0);
    // c_μ = Q^T c_free ; offset += c_free · x0 (+ fixed contrib already in lp)
    f64 offset = lp.obj_offset;
    for (std::size_t fi = 0; fi < map.fixed_cols.size(); ++fi) {
        const Index j = map.fixed_cols[fi];
        offset += lp.c[static_cast<std::size_t>(j)] * map.fixed_vals[fi];
    }
    for (std::size_t ii = 0; ii < free_int.size(); ++ii) {
        const Index oj = free_int[ii];
        const f64 cj = lp.c[static_cast<std::size_t>(oj)];
        offset += cj * static_cast<f64>(map.x0[ii]);
        for (int t = 0; t < k; ++t) {
            tp.c[static_cast<std::size_t>(t)] +=
                cj * static_cast<f64>(map.Q[ii][static_cast<std::size_t>(t)]);
        }
    }
    // Forced-zero continuous contribute 0.
    tp.obj_offset = offset;
    tp.col_names.resize(static_cast<std::size_t>(k));
    for (int t = 0; t < k; ++t)
        tp.col_names[static_cast<std::size_t>(t)] = "mu" + std::to_string(t);
    tp.row_names.resize(static_cast<std::size_t>(row_id));
    for (Index r = 0; r < row_id; ++r)
        tp.row_names[static_cast<std::size_t>(r)] = "bnd" + std::to_string(r);

    try {
        tp.validate();
    } catch (...) {
        return std::nullopt;
    }
    map.transformed = std::move(tp);
    return map;
}

std::vector<f64>
lattice_postsolve(const LatticeReformMap& map, const std::vector<f64>& mu) {
    const Index n_all = static_cast<Index>(
        map.free_int_cols.size() + map.forced_zero_cols.size() +
        map.fixed_cols.size());
    // Recover original dimension from max index.
    Index n_orig = 0;
    auto consider = [&](const std::vector<Index>& v) {
        for (Index j : v) n_orig = std::max(n_orig, static_cast<Index>(j + 1));
    };
    consider(map.free_int_cols);
    consider(map.forced_zero_cols);
    consider(map.fixed_cols);
    (void)n_all;

    std::vector<f64> x(static_cast<std::size_t>(n_orig), 0.0);
    for (std::size_t i = 0; i < map.fixed_cols.size(); ++i) {
        x[static_cast<std::size_t>(map.fixed_cols[i])] = map.fixed_vals[i];
    }
    for (Index j : map.forced_zero_cols) {
        x[static_cast<std::size_t>(j)] = 0.0;
    }
    const int n = static_cast<int>(map.free_int_cols.size());
    const int k = map.kernel_dim;
    if (static_cast<int>(mu.size()) < k) {
        throw std::invalid_argument("lattice_postsolve: mu too short");
    }
    // Round μ to integers (B&B should already be integral).
    std::vector<i64> mu_i(static_cast<std::size_t>(k));
    for (int t = 0; t < k; ++t) {
        if (!near_integer(mu[static_cast<std::size_t>(t)], 1e-6)) {
            throw std::runtime_error("lattice_postsolve: non-integer mu");
        }
        mu_i[static_cast<std::size_t>(t)] =
            round_exact(mu[static_cast<std::size_t>(t)]);
    }
    for (int i = 0; i < n; ++i) {
        i128 s = map.x0[static_cast<std::size_t>(i)];
        for (int t = 0; t < k; ++t) {
            s += static_cast<i128>(
                     map.Q[static_cast<std::size_t>(i)][static_cast<std::size_t>(t)]) *
                 static_cast<i128>(mu_i[static_cast<std::size_t>(t)]);
        }
        x[static_cast<std::size_t>(map.free_int_cols[static_cast<std::size_t>(i)])] =
            static_cast<f64>(i128_to_i64_exact(s));
    }
    return x;
}

LatticeSolveOutcome
solve_milp_lattice(const model::LpProblem& lp,
                   const BabOptions& bab,
                   bool attempt_reform,
                   const LatticeReformOptions& lropts) {
    LatticeSolveOutcome out;
    if (!attempt_reform) {
        out.raw = solve_milp(lp, bab, out.diag);
        return out;
    }

    auto map = try_lattice_reform(lp, lropts);
    if (!map) {
        out.note = "guard/verify failed";
        out.raw = solve_milp(lp, bab, out.diag);
        return out;
    }
    out.reform_applied = true;
    out.exact_equivalence = map->exact_equivalence;
    out.kernel_dim = map->kernel_dim;
    out.forced_zero_cols = map->forced_zero_cols.size();
    out.fixed_cols = map->fixed_cols.size();
    out.note = map->note;

    out.raw = solve_milp(map->transformed, bab, out.diag);

    // Postsolve any incumbent into original x-space (exact integer map).
    bool have_incumbent = false;
    core::RawResult kept;
    if (!out.raw.x.empty()) {
        try {
            kept = out.raw;
            kept.x = lattice_postsolve(*map, out.raw.x);
            kept.objective = lp.objective(kept.x);
            kept.dual_bound = core::kNaN;
            have_incumbent = true;
        } catch (const std::exception& ex) {
            out.note += "; postsolve failed: " + std::string(ex.what());
        }
    }

    const bool terminal = out.raw.proposed_status == core::Status::Optimal ||
                          out.raw.proposed_status == core::Status::Infeasible;

    if (map->exact_equivalence) {
        // The transform is an exact bijection (mu bounds are the exact LP
        // projection of the original box through Q, no forced-zero columns
        // excluded): the transformed problem IS the original in different
        // coordinates, so every status and bound from solving it transfers
        // directly. No certification argument or fallback re-solve needed.
        if (have_incumbent) {
            kept.proposed_status = out.raw.proposed_status;
            kept.proposed_level = out.raw.proposed_level;
            kept.dual_bound = out.diag.dual_bound;  // same objective scale
            out.raw = std::move(kept);
        } else {
            out.raw.dual_bound = out.diag.dual_bound;
        }
        return out;
    }

    if (!terminal) {
        if (have_incumbent) {
            kept.proposed_status = core::Status::Feasible;
            kept.proposed_level = core::ProofLevel::FeasibleOnly;
            out.raw = std::move(kept);
        }
        // A transformed-space dual bound is a bound on the RESTRICTED
        // problem (forced-zero continuous columns excluded), not on the
        // original — NaN it on every non-fallback path so no downstream
        // gap/printout can over-read it.
        out.raw.dual_bound = core::kNaN;
        out.diag.dual_bound = core::kNaN;
        out.diag.gap_rel = core::kNaN;
        return out;
    }

    // Terminal status under a genuine restriction (forced-zero continuous
    // columns excluded). Try to CERTIFY it against the ORIGINAL problem's
    // own LP relaxation bound before falling back: if the restricted
    // optimum V_r exactly matches the original LP bound V_LP, then
    // V_LP <= V_orig <= V_r = V_LP forces V_orig = V_r, so the postsolved
    // incumbent is provably optimal for the ORIGINAL problem — a rigorous
    // argument (not a heuristic), and it is what lets a solved restriction
    // (e.g. "does an exact 0/1 split exist" for markshare) actually close
    // the original instance instead of always being thrown away.
    if (out.raw.proposed_status == core::Status::Optimal && have_incumbent) {
        engines::SimplexOptions sopts = bab.lp;
        sopts.presolve = true;
        engines::SimplexDiagnostics sdiag;
        auto lp_relax = engines::solve_simplex(lp, sopts, sdiag);
        if (lp_relax.proposed_status == core::Status::Optimal &&
            std::isfinite(lp_relax.objective)) {
            const f64 V_LP = lp_relax.objective;
            const f64 V_r = kept.objective;
            const f64 tol = 1e-6 * (1.0 + std::fabs(V_LP));
            const bool matches =
                lp.maximize ? (V_r >= V_LP - tol) : (V_r <= V_LP + tol);
            bool exact_feasible = true;
            for (Index j : map->free_int_cols) {
                const f64 xj = kept.x[static_cast<std::size_t>(j)];
                if (xj < lp.col_lo[static_cast<std::size_t>(j)] - 1e-6 ||
                    xj > lp.col_hi[static_cast<std::size_t>(j)] + 1e-6) {
                    exact_feasible = false;
                    break;
                }
            }
            if (matches && exact_feasible) {
                out.certified = true;
                out.note +=
                    "; certified optimal (incumbent matches original LP bound)";
                kept.proposed_status = core::Status::Optimal;
                kept.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
                kept.dual_bound = V_LP;
                out.raw = std::move(kept);
                out.diag.incumbent = V_r;
                out.diag.dual_bound = V_LP;
                out.diag.gap_rel = 0.0;
                out.diag.globally_proved = true;
                out.diag.termination_reason =
                    "lattice-certified: incumbent matches original LP relaxation bound";
                return out;
            }
        }
    }

    // Not certifiable (restricted Infeasible — no exact split exists but the
    // original may still be feasible with the forced columns active; or a
    // restricted Optimal that does not match the LP bound — the forced
    // columns were genuinely needed): nothing about the original follows
    // from the restricted run. Re-solve the original in full.
    out.fell_back = true;
    BabDiagnostics diag2;
    auto raw2 = solve_milp(lp, bab, diag2);
    if (!raw2.x.empty()) {
        out.raw = std::move(raw2);
        out.diag = std::move(diag2);
    } else if (have_incumbent) {
        kept.proposed_status = core::Status::Feasible;
        kept.proposed_level = core::ProofLevel::FeasibleOnly;
        kept.dual_bound = core::kNaN;
        out.raw = std::move(kept);
        out.diag.dual_bound = core::kNaN;
        out.diag.gap_rel = core::kNaN;
    } else {
        out.raw = std::move(raw2);
        out.diag = std::move(diag2);
    }
    return out;
}

}  // namespace sor::search
