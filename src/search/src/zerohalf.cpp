#include "sor/search/zerohalf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

struct BaseRow {
    std::vector<Index> cols;
    std::vector<f64> coef;  // integer-valued after construction
    f64 rhs = 0.0;          // integer-valued
    f64 slack = 0.0;        // rhs − a·x* at build time
};

inline bool exact_integer(f64 v) {
    return std::isfinite(v) && std::fabs(v) <= 9007199254740992.0 &&
           v == std::round(v);
}

// Build ∑ a_j x_j ≤ b from an exact-integer, nonnegative-variable row.
// Mixed rows and nonintegral coefficients require a directed-rounding
// derivation that this separator does not yet provide; abstain instead of
// rounding small coefficients away or promoting a cut with a false premise.
bool build_integer_base(const model::LpProblem& lp, Index row, f64 sign,
                        const std::vector<f64>& lo,
                        const std::vector<f64>& x,
                        BaseRow& out, ZeroHalfDiagnostics& diag) {
    const f64 bound = sign > 0.0 ? lp.row_hi[sz(row)] : -lp.row_lo[sz(row)];
    if (!std::isfinite(bound)) return false;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    out.cols.clear();
    out.coef.clear();
    out.rhs = bound;
    f64 activity = 0.0;

    for (core::Offset k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = sign * av[sz(k)];
        if (a == 0.0) continue;

        const bool is_int =
            !lp.is_integer.empty() && lp.is_integer[sz(j)];
        if (!is_int) return false;
        // floor(a/2) x <= floor(b/2) uses x >= 0 when rounding a downward.
        if (!std::isfinite(lo[sz(j)]) || lo[sz(j)] < 0.0) return false;
        if (!exact_integer(a)) return false;
        out.cols.push_back(j);
        out.coef.push_back(a);
        activity += a * x[sz(j)];
    }
    if (out.cols.empty()) return false;
    if (!exact_integer(out.rhs)) return false;
    out.slack = out.rhs - activity;
    ++diag.bases_built;
    return true;
}

bool merge_bases(const BaseRow& a, const BaseRow& b, BaseRow& out) {
    out.cols = a.cols;
    out.coef = a.coef;
    out.rhs = a.rhs + b.rhs;
    for (std::size_t t = 0; t < b.cols.size(); ++t) {
        const Index j = b.cols[t];
        bool found = false;
        for (std::size_t s = 0; s < out.cols.size(); ++s) {
            if (out.cols[s] == j) {
                out.coef[s] += b.coef[t];
                found = true;
                break;
            }
        }
        if (!found) {
            out.cols.push_back(j);
            out.coef.push_back(b.coef[t]);
        }
    }
    return !out.cols.empty();
}

// Aggregate bases selected by a bit-mask over an index list.
bool aggregate_selected(const std::vector<BaseRow>& bases,
                        const std::vector<int>& idxs, BaseRow& out) {
    if (idxs.empty()) return false;
    out = bases[static_cast<std::size_t>(idxs[0])];
    for (std::size_t t = 1; t < idxs.size(); ++t) {
        BaseRow merged;
        if (!merge_bases(out, bases[static_cast<std::size_t>(idxs[t])], merged))
            return false;
        out = std::move(merged);
    }
    return !out.cols.empty();
}

// CG cut with λ = 1/2 on an integer base ∑ a x ≤ b.
bool emit_half_cg(const BaseRow& base, const std::vector<f64>& x,
                  const ZeroHalfOptions& opts, CutRow& cut, f64& viol) {
    if (!exact_integer(base.rhs)) return false;
    for (const f64 aj : base.coef)
        if (!exact_integer(aj)) return false;
    cut.cols.clear();
    cut.vals.clear();
    f64 rhs = std::floor(base.rhs * 0.5);
    f64 lhs = 0.0;
    f64 cmin = kInf, cmax = 0.0;
    for (std::size_t t = 0; t < base.cols.size(); ++t) {
        const f64 aj = base.coef[t];
        const f64 cj = std::floor(aj * 0.5);
        if (std::fabs(cj) <= opts.tol) continue;
        cut.cols.push_back(base.cols[t]);
        cut.vals.push_back(cj);
        lhs += cj * x[sz(base.cols[t])];
        cmin = std::min(cmin, std::fabs(cj));
        cmax = std::max(cmax, std::fabs(cj));
    }
    if (cut.cols.empty()) return false;
    if (cmin > 0.0 && cmax / cmin > opts.max_dynamism) return false;
    viol = lhs - rhs;
    if (viol <= opts.violation_min) return false;
    cut.row_lo = -kInf;
    cut.row_hi = rhs;
    return true;
}

// ---- GF(2) bit matrix -------------------------------------------------------

using BitWord = std::uint64_t;
constexpr int kWordBits = 64;

inline int words_for(int nbits) {
    return (nbits + kWordBits - 1) / kWordBits;
}

struct Mod2Row {
    std::vector<BitWord> bits;  // odd-support over column universe
    int rhs_parity = 0;
    // Which original bases XOR into this row (bitmask over local index, or
    // dense list when > 64 bases).
    std::vector<BitWord> combo;  // over base-index universe
    int odd_weight = 0;
};

inline void bits_xor(std::vector<BitWord>& a, const std::vector<BitWord>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) a[i] ^= b[i];
}

inline bool bit_get(const std::vector<BitWord>& v, int j) {
    return (v[static_cast<std::size_t>(j / kWordBits)] >> (j % kWordBits)) & 1ull;
}

inline void bit_set(std::vector<BitWord>& v, int j) {
    v[static_cast<std::size_t>(j / kWordBits)] |= (1ull << (j % kWordBits));
}

inline int popcount_bits(const std::vector<BitWord>& v) {
    int c = 0;
    for (BitWord w : v) {
#if defined(__GNUG__) || defined(__clang__)
        c += __builtin_popcountll(w);
#else
        while (w) {
            w &= w - 1;
            ++c;
        }
#endif
    }
    return c;
}

void gaussian_elim_gf2(std::vector<Mod2Row>& rows, int ncols,
                       ZeroHalfDiagnostics& diag) {
    const int m = static_cast<int>(rows.size());
    std::vector<int> pivot_of_col(static_cast<std::size_t>(ncols), -1);
    int rank = 0;
    for (int col = 0; col < ncols && rank < m; ++col) {
        int piv = -1;
        for (int r = rank; r < m; ++r) {
            if (bit_get(rows[static_cast<std::size_t>(r)].bits, col)) {
                piv = r;
                break;
            }
        }
        if (piv < 0) continue;
        if (piv != rank) std::swap(rows[static_cast<std::size_t>(rank)],
                                   rows[static_cast<std::size_t>(piv)]);
        pivot_of_col[static_cast<std::size_t>(col)] = rank;
        ++diag.mod2_pivots;
        for (int r = 0; r < m; ++r) {
            if (r == rank) continue;
            if (!bit_get(rows[static_cast<std::size_t>(r)].bits, col)) continue;
            bits_xor(rows[static_cast<std::size_t>(r)].bits,
                     rows[static_cast<std::size_t>(rank)].bits);
            bits_xor(rows[static_cast<std::size_t>(r)].combo,
                     rows[static_cast<std::size_t>(rank)].combo);
            rows[static_cast<std::size_t>(r)].rhs_parity ^=
                rows[static_cast<std::size_t>(rank)].rhs_parity;
        }
        ++rank;
    }
    for (auto& r : rows) r.odd_weight = popcount_bits(r.bits);
}

}  // namespace

std::vector<CutRow> separate_zerohalf(const model::LpProblem& lp,
                                      const std::vector<f64>& x,
                                      const std::vector<f64>& col_lo,
                                      const std::vector<f64>& col_hi,
                                      const ZeroHalfOptions& opts,
                                      ZeroHalfDiagnostics& diag) {
    std::vector<CutRow> cuts;
    const Index n = lp.n_cols();
    if (!opts.enabled || static_cast<Index>(x.size()) != n ||
        static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return cuts;

    const auto& rp = lp.A.pattern.row_ptr();
    std::vector<BaseRow> bases;
    bases.reserve(static_cast<std::size_t>(lp.n_rows()));

    for (Index i = 0; i < lp.n_rows(); ++i) {
        if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
        ++diag.rows_scanned;
        for (const f64 sign : {1.0, -1.0}) {
            BaseRow b;
            if (!build_integer_base(lp, i, sign, col_lo, x, b,
                                    diag))
                continue;
            bases.push_back(std::move(b));
        }
    }

    auto try_emit = [&](const BaseRow& base) {
        if (static_cast<int>(cuts.size()) >= opts.max_cuts) return;
        CutRow cut;
        f64 viol = 0.0;
        if (!emit_half_cg(base, x, opts, cut, viol)) {
            if (viol <= opts.violation_min) ++diag.rejected_not_violated;
            else ++diag.rejected_dynamism;
            return;
        }
        cut.name = "ZH_" + std::to_string(cuts.size());
        cuts.push_back(std::move(cut));
        ++diag.cuts_emitted;
    };

    // Cheap warm-up: single-row CG.
    for (const auto& b : bases) try_emit(b);

    // Legacy pair aggregation.
    if (opts.max_pair_tries > 0 && static_cast<int>(cuts.size()) < opts.max_cuts) {
        const int nbase = static_cast<int>(bases.size());
        int tries = 0;
        for (int i = 0; i < nbase && tries < opts.max_pair_tries; ++i) {
            for (int k = i + 1; k < nbase && tries < opts.max_pair_tries; ++k) {
                ++tries;
                BaseRow merged;
                if (!merge_bases(bases[static_cast<std::size_t>(i)],
                                 bases[static_cast<std::size_t>(k)], merged))
                    continue;
                ++diag.aggregations;
                try_emit(merged);
                if (static_cast<int>(cuts.size()) >= opts.max_cuts) break;
            }
        }
    }

    // ---- Koster-style mod-2 Gaussian + violated-cut enumeration ------------
    if (!opts.use_mod2_gaussian || bases.empty() ||
        static_cast<int>(cuts.size()) >= opts.max_cuts)
        return cuts;

    // Select near-tight bases; build column universe of odd supports.
    std::vector<int> selected;
    selected.reserve(static_cast<std::size_t>(opts.max_mod2_rows));
    for (int i = 0; i < static_cast<int>(bases.size()); ++i) {
        if (static_cast<int>(selected.size()) >= opts.max_mod2_rows) break;
        if (bases[static_cast<std::size_t>(i)].slack > opts.max_slack + opts.tol)
            continue;
        selected.push_back(i);
    }
    if (selected.empty()) return cuts;

    // Map global col -> local odd-column index.
    std::vector<int> col_to_local(static_cast<std::size_t>(n), -1);
    std::vector<Index> local_to_col;
    auto ensure_local = [&](Index j) {
        int& slot = col_to_local[sz(j)];
        if (slot < 0) {
            slot = static_cast<int>(local_to_col.size());
            local_to_col.push_back(j);
        }
        return slot;
    };

    const int nbase_sel = static_cast<int>(selected.size());
    std::vector<Mod2Row> mat;
    mat.reserve(static_cast<std::size_t>(nbase_sel));

    // First pass: discover odd columns.
    for (int si = 0; si < nbase_sel; ++si) {
        const BaseRow& b = bases[static_cast<std::size_t>(selected[si])];
        for (std::size_t t = 0; t < b.cols.size(); ++t) {
            const long long aj =
                static_cast<long long>(std::llround(b.coef[t]));
            if (aj & 1LL) ensure_local(b.cols[t]);
        }
    }
    const int ncols = static_cast<int>(local_to_col.size());
    if (ncols == 0) return cuts;

    const int nw_cols = words_for(ncols);
    const int nw_base = words_for(nbase_sel);

    for (int si = 0; si < nbase_sel; ++si) {
        const BaseRow& b = bases[static_cast<std::size_t>(selected[si])];
        Mod2Row r;
        r.bits.assign(static_cast<std::size_t>(nw_cols), 0);
        r.combo.assign(static_cast<std::size_t>(nw_base), 0);
        bit_set(r.combo, si);
        for (std::size_t t = 0; t < b.cols.size(); ++t) {
            const long long aj =
                static_cast<long long>(std::llround(b.coef[t]));
            if (aj & 1LL) {
                const int lc = col_to_local[sz(b.cols[t])];
                if (lc >= 0) bit_set(r.bits, lc);
            }
        }
        const long long br = static_cast<long long>(std::llround(b.rhs));
        r.rhs_parity = static_cast<int>(br & 1LL);
        r.odd_weight = popcount_bits(r.bits);
        mat.push_back(std::move(r));
    }
    diag.mod2_rows = static_cast<std::uint64_t>(mat.size());

    gaussian_elim_gf2(mat, ncols, diag);

    // Map selected-local combo indices back to bases[] indices.
    auto bases_from_red = [&](const std::vector<int>& red_idxs, BaseRow& out) {
        std::vector<BitWord> combo(static_cast<std::size_t>(nw_base), 0);
        for (int ri : red_idxs)
            bits_xor(combo, mat[static_cast<std::size_t>(ri)].combo);
        std::vector<int> idxs;
        for (int si = 0; si < nbase_sel; ++si) {
            if (bit_get(combo, si)) idxs.push_back(selected[si]);
        }
        return aggregate_selected(bases, idxs, out);
    };

    // Degree-1: each nonzero reduced row.
    const int nred = static_cast<int>(mat.size());
    for (int r = 0; r < nred && static_cast<int>(cuts.size()) < opts.max_cuts;
         ++r) {
        if (mat[static_cast<std::size_t>(r)].odd_weight == 0 &&
            mat[static_cast<std::size_t>(r)].rhs_parity == 0)
            continue;
        BaseRow agg;
        if (!bases_from_red({r}, agg)) continue;
        ++diag.aggregations;
        try_emit(agg);
    }

    // Higher-degree enumeration of reduced-row combinations.
    const int deg = std::max(1, opts.max_enum_degree);
    if (deg >= 2 && static_cast<int>(cuts.size()) < opts.max_cuts) {
        // Prefer light reduced rows (few odd entries) - Caprara-Fischetti /
        // Koster heuristic: sparse mod-2 supports yield denser CG cuts less
        // often, but small-weight rows separate well.
        std::vector<int> order(static_cast<std::size_t>(nred));
        for (int i = 0; i < nred; ++i) order[static_cast<std::size_t>(i)] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return mat[static_cast<std::size_t>(a)].odd_weight <
                   mat[static_cast<std::size_t>(b)].odd_weight;
        });
        const int pool = std::min(nred, 48);

        auto enum_degree = [&](int d) {
            if (d == 2) {
                for (int a = 0; a < pool; ++a) {
                    for (int b = a + 1; b < pool; ++b) {
                        if (static_cast<int>(cuts.size()) >= opts.max_cuts)
                            return;
                        if (static_cast<int>(diag.enum_combos) >=
                            opts.max_enum_combos)
                            return;
                        ++diag.enum_combos;
                        BaseRow agg;
                        if (!bases_from_red(
                                {order[static_cast<std::size_t>(a)],
                                 order[static_cast<std::size_t>(b)]},
                                agg))
                            continue;
                        ++diag.aggregations;
                        try_emit(agg);
                    }
                }
            } else if (d == 3) {
                for (int a = 0; a < pool; ++a) {
                    for (int b = a + 1; b < pool; ++b) {
                        for (int c = b + 1; c < pool; ++c) {
                            if (static_cast<int>(cuts.size()) >= opts.max_cuts)
                                return;
                            if (static_cast<int>(diag.enum_combos) >=
                                opts.max_enum_combos)
                                return;
                            ++diag.enum_combos;
                            BaseRow agg;
                            if (!bases_from_red(
                                    {order[static_cast<std::size_t>(a)],
                                     order[static_cast<std::size_t>(b)],
                                     order[static_cast<std::size_t>(c)]},
                                    agg))
                                continue;
                            ++diag.aggregations;
                            try_emit(agg);
                        }
                    }
                }
            }
        };
        for (int d = 2; d <= deg; ++d) enum_degree(d);
    }

    return cuts;
}

}  // namespace sor::search
