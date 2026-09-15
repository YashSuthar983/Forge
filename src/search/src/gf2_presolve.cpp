#include "sor/search/gf2_presolve.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline bool nearly_fixed(f64 lo, f64 hi, f64 tol) {
    return std::isfinite(lo) && std::isfinite(hi) && hi - lo <= tol;
}

inline bool is_binary_col(const model::LpProblem& lp, Index j,
                          const std::vector<f64>& lo, const std::vector<f64>& hi,
                          f64 tol) {
    if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) return false;
    return std::fabs(lo[sz(j)]) <= tol && std::fabs(hi[sz(j)] - 1.0) <= tol;
}

inline bool nearly_integer(f64 v, f64 tol) {
    return std::fabs(v - std::round(v)) <= tol;
}

// Odd integer coefficient? Even coeffs vanish mod 2.
inline bool odd_int_coef(f64 a, f64 tol) {
    if (!nearly_integer(a, tol)) return false;
    const long long k = static_cast<long long>(std::llround(a));
    return (k & 1LL) != 0;
}

struct Eq {
    std::vector<Index> cols;  // free binary columns with odd coef
    int rhs = 0;              // mod 2
};

bool collect_equation(const model::LpProblem& lp, Index row,
                      const std::vector<f64>& col_lo,
                      const std::vector<f64>& col_hi, f64 tol, Eq& out) {
    const f64 rlo = lp.row_lo[sz(row)];
    const f64 rhi = lp.row_hi[sz(row)];
    // Need an equality (or activity-forced equality handled by caller via
    // matching lo/hi).
    if (!std::isfinite(rlo) || !std::isfinite(rhi) ||
        std::fabs(rlo - rhi) > tol)
        return false;
    if (!nearly_integer(rlo, tol)) return false;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    int rhs = static_cast<int>(std::llround(rlo)) & 1;
    out.cols.clear();

    for (core::Offset k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = av[sz(k)];
        if (std::fabs(a) <= tol) continue;

        if (nearly_fixed(col_lo[sz(j)], col_hi[sz(j)], tol)) {
            if (!nearly_integer(col_lo[sz(j)], tol)) return false;
            if (!nearly_integer(a, tol)) return false;
            const long long ak = static_cast<long long>(std::llround(a));
            const long long xk =
                static_cast<long long>(std::llround(col_lo[sz(j)]));
            if ((ak & 1LL) && (xk & 1LL)) rhs ^= 1;
            continue;
        }

        // Free term must be binary with odd integer coefficient.
        if (!is_binary_col(lp, j, col_lo, col_hi, tol)) return false;
        if (!odd_int_coef(a, tol)) return false;
        out.cols.push_back(j);
    }

    // Dedup columns (sum of two odds on same var → even → drop).
    std::sort(out.cols.begin(), out.cols.end());
    std::vector<Index> uniq;
    for (Index j : out.cols) {
        if (!uniq.empty() && uniq.back() == j) {
            uniq.pop_back();  // cancel mod 2
        } else {
            uniq.push_back(j);
        }
    }
    out.cols = std::move(uniq);
    out.rhs = rhs;
    return true;
}

}  // namespace

Gf2PresolveDiagnostics apply_gf2_presolve(const model::LpProblem& lp,
                                          std::vector<f64>& col_lo,
                                          std::vector<f64>& col_hi,
                                          const Gf2PresolveOptions& opts) {
    Gf2PresolveDiagnostics diag;
    if (!opts.enabled) return diag;

    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    if (static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return diag;

    for (int round = 0; round < opts.max_rounds; ++round) {
        std::vector<Eq> eqs;
        eqs.reserve(sz(std::min(m, opts.max_equations)));

        for (Index i = 0; i < m; ++i) {
            if (static_cast<Index>(eqs.size()) >= opts.max_equations) {
                diag.truncated = true;
                break;
            }
            Eq e;
            if (!collect_equation(lp, i, col_lo, col_hi, opts.tol, e))
                continue;
            if (e.cols.empty()) {
                if (e.rhs != 0) {
                    diag.infeasible = true;
                    return diag;
                }
                continue;
            }
            eqs.push_back(std::move(e));
        }

        if (eqs.empty()) break;
        diag.equations = std::max(
            diag.equations, static_cast<std::uint64_t>(eqs.size()));

        // Map free vars appearing in eqs.
        std::vector<Index> vars;
        for (const Eq& e : eqs)
            for (Index j : e.cols) vars.push_back(j);
        std::sort(vars.begin(), vars.end());
        vars.erase(std::unique(vars.begin(), vars.end()), vars.end());
        if (static_cast<Index>(vars.size()) > opts.max_vars) {
            vars.resize(sz(opts.max_vars));
            diag.truncated = true;
        }
        const Index nv = static_cast<Index>(vars.size());
        if (nv == 0) break;
        diag.vars = std::max(diag.vars, static_cast<std::uint64_t>(nv));

        std::vector<Index> col_to_v(sz(n), -1);
        for (Index t = 0; t < nv; ++t) col_to_v[sz(vars[sz(t)])] = t;

        // Bitsets as uint64 limbs — nv may exceed 64; use vector<uint64_t> per row.
        const Index nlimbs = (nv + 63) / 64;
        auto bit_get = [&](const std::vector<std::uint64_t>& row, Index v) {
            return (row[sz(v >> 6)] >> (v & 63)) & 1ULL;
        };
        auto bit_xor_into = [&](std::vector<std::uint64_t>& a,
                                const std::vector<std::uint64_t>& b) {
            for (Index L = 0; L < nlimbs; ++L) a[sz(L)] ^= b[sz(L)];
        };
        auto bit_set = [&](std::vector<std::uint64_t>& row, Index v) {
            row[sz(v >> 6)] |= (1ULL << (v & 63));
        };
        auto row_empty = [&](const std::vector<std::uint64_t>& row) {
            for (Index L = 0; L < nlimbs; ++L)
                if (row[sz(L)]) return false;
            return true;
        };
        auto first_bit = [&](const std::vector<std::uint64_t>& row) -> Index {
            for (Index L = 0; L < nlimbs; ++L) {
                std::uint64_t w = row[sz(L)];
                if (!w) continue;
                return static_cast<Index>(L * 64 +
                                          __builtin_ctzll(w));
            }
            return -1;
        };

        std::vector<std::vector<std::uint64_t>> mat;
        std::vector<int> rhs;
        mat.reserve(eqs.size());
        for (const Eq& e : eqs) {
            std::vector<std::uint64_t> row(sz(nlimbs), 0);
            bool skip = false;
            for (Index j : e.cols) {
                const Index v = col_to_v[sz(j)];
                if (v < 0) {
                    skip = true;
                    break;
                }
                bit_set(row, v);
            }
            if (skip) continue;
            mat.push_back(std::move(row));
            rhs.push_back(e.rhs);
        }

        const Index ne = static_cast<Index>(mat.size());
        std::vector<Index> pivot_of_var(sz(nv), -1);
        Index rank = 0;
        for (Index col = 0; col < nv && rank < ne; ++col) {
            Index piv = -1;
            for (Index r = rank; r < ne; ++r) {
                if (bit_get(mat[sz(r)], col)) {
                    piv = r;
                    break;
                }
            }
            if (piv < 0) continue;
            if (piv != rank) {
                std::swap(mat[sz(piv)], mat[sz(rank)]);
                std::swap(rhs[sz(piv)], rhs[sz(rank)]);
            }
            for (Index r = 0; r < ne; ++r) {
                if (r == rank) continue;
                if (bit_get(mat[sz(r)], col)) {
                    bit_xor_into(mat[sz(r)], mat[sz(rank)]);
                    rhs[sz(r)] ^= rhs[sz(rank)];
                }
            }
            pivot_of_var[sz(col)] = rank;
            ++rank;
            ++diag.pivots;
        }

        // Check inconsistency / extract singleton fixings.
        bool any_fix = false;
        for (Index r = 0; r < ne; ++r) {
            if (row_empty(mat[sz(r)])) {
                if (rhs[sz(r)] != 0) {
                    diag.infeasible = true;
                    return diag;
                }
                continue;
            }
            // Count set bits; if exactly one, fix that var to rhs.
            Index only = -1;
            int cnt = 0;
            for (Index L = 0; L < nlimbs; ++L) {
                std::uint64_t w = mat[sz(r)][sz(L)];
                while (w) {
                    const int b = __builtin_ctzll(w);
                    const Index v = static_cast<Index>(L * 64 + b);
                    if (v >= nv) break;
                    ++cnt;
                    only = v;
                    w &= w - 1;
                }
            }
            if (cnt == 1 && only >= 0) {
                const Index j = vars[sz(only)];
                const f64 val = static_cast<f64>(rhs[sz(r)] & 1);
                if (col_lo[sz(j)] > val + opts.tol ||
                    col_hi[sz(j)] < val - opts.tol) {
                    diag.infeasible = true;
                    return diag;
                }
                if (!nearly_fixed(col_lo[sz(j)], col_hi[sz(j)], opts.tol) ||
                    std::fabs(col_lo[sz(j)] - val) > opts.tol) {
                    col_lo[sz(j)] = val;
                    col_hi[sz(j)] = val;
                    ++diag.fixings;
                    any_fix = true;
                }
            } else if (cnt >= 2) {
                // Substitution count: pivot rows with leading 1 and ≥1 other.
                const Index lead = first_bit(mat[sz(r)]);
                if (lead >= 0 && pivot_of_var[sz(lead)] == r) ++diag.substitutions;
            }
        }

        if (!any_fix) break;
    }
    return diag;
}

}  // namespace sor::search
