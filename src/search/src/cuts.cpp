#include "sor/search/cuts.hpp"

#include "sor/la/lu.hpp"
#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }

bool canonicalize(CutRow& cut) {
    if (cut.cols.size() != cut.vals.size()) return false;
    std::vector<std::pair<Index, f64>> terms;
    terms.reserve(cut.cols.size());
    for (std::size_t k = 0; k < cut.cols.size(); ++k) {
        if (cut.cols[k] < 0 || !std::isfinite(cut.vals[k])) return false;
        if (cut.vals[k] != 0.0) terms.emplace_back(cut.cols[k], cut.vals[k]);
    }
    std::sort(terms.begin(), terms.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    cut.cols.clear();
    cut.vals.clear();
    for (const auto& [col, val] : terms) {
        if (!cut.cols.empty() && cut.cols.back() == col) {
            cut.vals.back() += val;
        } else {
            cut.cols.push_back(col);
            cut.vals.push_back(val);
        }
    }
    for (std::size_t k = cut.vals.size(); k-- > 0;)
        if (cut.vals[k] == 0.0) {
            cut.vals.erase(cut.vals.begin() + static_cast<std::ptrdiff_t>(k));
            cut.cols.erase(cut.cols.begin() + static_cast<std::ptrdiff_t>(k));
        }
    return !cut.cols.empty();
}

bool normalized_le(const CutRow& cut, std::vector<f64>& unit_vals,
                   f64& rhs_unit) {
    if (cut.cols.size() != cut.vals.size() || cut.cols.empty()) return false;
    const bool has_lo = std::isfinite(cut.row_lo);
    const bool has_hi = std::isfinite(cut.row_hi);
    if (has_lo == has_hi) return false;  // only one-sided inequalities belong here
    const f64 sign = has_hi ? 1.0 : -1.0;
    f64 norm2 = 0.0;
    for (const f64 v : cut.vals) {
        if (!std::isfinite(v)) return false;
        norm2 += v * v;
    }
    if (!(norm2 > 0.0) || !std::isfinite(norm2)) return false;
    const f64 norm = std::sqrt(norm2);
    unit_vals.resize(cut.vals.size());
    for (std::size_t k = 0; k < cut.vals.size(); ++k)
        unit_vals[k] = sign * cut.vals[k] / norm;
    rhs_unit = sign * (has_hi ? cut.row_hi : cut.row_lo) / norm;
    return std::isfinite(rhs_unit);
}

f64 sparse_dot(const CutRow& a, const std::vector<f64>& av,
               const CutRow& b, const std::vector<f64>& bv) {
    std::size_t p = 0, q = 0;
    f64 dot = 0.0;
    while (p < a.cols.size() && q < b.cols.size()) {
        if (a.cols[p] == b.cols[q]) {
            dot += av[p] * bv[q];
            ++p;
            ++q;
        } else if (a.cols[p] < b.cols[q]) {
            ++p;
        } else {
            ++q;
        }
    }
    return dot;
}

}  // namespace

std::size_t CutPool::active_size() const {
    return static_cast<std::size_t>(std::count_if(
        entries_.begin(), entries_.end(), [](const Entry& e) { return e.active; }));
}

void CutPool::start_round(CutDiagnostics& diag) {
    for (auto& entry : entries_)
        if (!entry.active) ++entry.age;
    const auto old_size = entries_.size();
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& e) {
                                      return !e.active && e.age > opts_.pool_max_age;
                                  }),
                   entries_.end());
    diag.pool_aged_out += old_size - entries_.size();
}

void CutPool::add(const std::vector<CutRow>& candidates,
                  CutDiagnostics& diag) {
    constexpr f64 kCosRoundoff = 1e-12;
    for (const auto& cut : candidates) {
        Entry incoming;
        incoming.cut = cut;
        if (!canonicalize(incoming.cut) ||
            !normalized_le(incoming.cut, incoming.unit_vals, incoming.rhs_unit))
            continue;

        bool reject = false;
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->cut.cols != incoming.cut.cols) {
                ++it;
                continue;
            }
            const f64 cosine = sparse_dot(it->cut, it->unit_vals,
                                          incoming.cut, incoming.unit_vals);
            if (cosine < 1.0 - opts_.pool_duplicate_tol - kCosRoundoff) {
                ++it;
                continue;
            }
            const f64 rhs_delta = incoming.rhs_unit - it->rhs_unit;
            if (std::fabs(rhs_delta) <= opts_.pool_duplicate_tol) {
                ++diag.pool_duplicates;
                reject = true;
                break;
            }
            // In normalized <= form, the smaller rhs is the stronger cut.
            if (rhs_delta > 0.0) {
                ++diag.pool_dominated;
                reject = true;
                break;
            }
            if (!it->active) {
                it = entries_.erase(it);
                ++diag.pool_dominated;
                continue;
            }
            // The weaker row is already in the LP and cannot be removed here;
            // retain its fingerprint and allow the strengthening to be added.
            ++it;
        }
        if (reject) continue;
        entries_.push_back(std::move(incoming));
        ++diag.pool_inserted;
    }

    while (entries_.size() > opts_.pool_max_size) {
        auto victim = std::max_element(
            entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) {
                if (a.active != b.active) return a.active; // inactive is preferred
                if (a.age != b.age) return a.age < b.age;
                return a.last_efficacy > b.last_efficacy;
            });
        if (victim == entries_.end()) break;
        entries_.erase(victim);
        ++diag.pool_evicted;
    }
}

std::vector<CutRow> CutPool::select_violated(const std::vector<f64>& x,
                                             CutDiagnostics& diag) {
    if (opts_.max_cuts_per_round <= 0) return {};
    std::vector<std::size_t> order;
    order.reserve(entries_.size());
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        auto& entry = entries_[i];
        if (entry.active) continue;
        f64 activity = 0.0;
        bool valid_point = true;
        for (std::size_t k = 0; k < entry.cut.cols.size(); ++k) {
            const Index j = entry.cut.cols[k];
            if (j < 0 || sz(j) >= x.size()) {
                valid_point = false;
                break;
            }
            activity += entry.unit_vals[k] * x[sz(j)];
        }
        entry.last_efficacy = valid_point ? activity - entry.rhs_unit : 0.0;
        if (entry.last_efficacy >= opts_.pool_efficacy_min) order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return entries_[a].last_efficacy > entries_[b].last_efficacy;
    });

    std::vector<std::size_t> chosen;
    std::vector<CutRow> out;
    for (const std::size_t idx : order) {
        bool too_parallel = false;
        for (const std::size_t prev : chosen) {
            const f64 cosine = std::fabs(sparse_dot(
                entries_[idx].cut, entries_[idx].unit_vals,
                entries_[prev].cut, entries_[prev].unit_vals));
            if (cosine > opts_.pool_parallelism_max) {
                too_parallel = true;
                break;
            }
        }
        if (too_parallel) {
            ++diag.pool_rejected_parallel;
            continue;
        }
        entries_[idx].active = true;
        entries_[idx].age = 0;
        chosen.push_back(idx);
        out.push_back(entries_[idx].cut);
        ++diag.pool_selected;
        if (static_cast<int>(out.size()) >= opts_.max_cuts_per_round) break;
    }
    return out;
}

// Derives Gomory Mixed-Integer cuts from the optimal simplex tableau (Wolsey;
// Achterberg thesis 2007 Ch. 8.2-8.3). The public solve_simplex()/
// solve_dual_simplex() API returns a SimplexBasis (index bookkeeping only,
// never the LU factors used internally), so the basis matrix is re-assembled
// here from `lp.A` and refactorized independently with sor::la::BasisFactor.
// This duplicates one factorization per cutting round -- acceptable for a
// root-level loop bounded by CutOptions::max_rounds.
std::vector<CutRow> separate_gomory_mi(const model::LpProblem& lp,
                                       const std::vector<f64>& x,
                                       const engines::SimplexBasis& basis,
                                       const CutOptions& opts,
                                       CutDiagnostics& diag) {
    const Index m = lp.n_rows();
    const Index ns = lp.n_cols();
    const Index nt = ns + m;
    if (basis.n_struct != ns || static_cast<Index>(basis.basic.size()) != m ||
        static_cast<Index>(basis.status.size()) != nt ||
        static_cast<Index>(x.size()) != ns)
        return {};

    // ---- assemble and factorize the m x m basis matrix, in CSC form ------
    const auto Acsc = sparse::to_csc(lp.A);
    const auto& acp = Acsc.pattern.col_ptr();
    const auto& ari = Acsc.pattern.row_idx();
    const auto& acv = Acsc.vals;

    std::vector<core::Offset> bcol_ptr(sz(m) + 1, 0);
    std::vector<Index> brow_idx;
    std::vector<f64> bvals;
    brow_idx.reserve(sz(lp.nnz()));
    bvals.reserve(sz(lp.nnz()));
    for (Index slot = 0; slot < m; ++slot) {
        const Index bj = basis.basic[sz(slot)];
        if (bj >= 0 && bj < ns) {
            for (core::Offset k = acp[sz(bj)]; k < acp[sz(bj) + 1]; ++k) {
                brow_idx.push_back(ari[sz(k)]);
                bvals.push_back(acv[sz(k)]);
            }
        } else if (bj >= ns && bj < nt) {
            brow_idx.push_back(bj - ns);
            bvals.push_back(-1.0);
        } else {
            return {};  // malformed basis
        }
        bcol_ptr[sz(slot) + 1] = static_cast<core::Offset>(brow_idx.size());
    }

    la::BasisFactor factor;
    la::LuOptions lu_opts;
    std::vector<Index> singular_slots, vacant_rows;
    const bool factored = factor.factorize(m, bcol_ptr, brow_idx, bvals, lu_opts,
                                           &singular_slots, &vacant_rows);
    // A partially-repaired factorization does not represent the true B^-1;
    // skip this round rather than derive a cut from the wrong inverse.
    if (!factored || !factor.is_valid()) return {};

    // ---- per-column bookkeeping over the augmented [A | -I] space --------
    std::vector<char> is_basic(sz(nt), 0);
    for (Index slot = 0; slot < m; ++slot) is_basic[sz(basis.basic[sz(slot)])] = 1;

    std::vector<f64> lo(sz(nt)), hi(sz(nt));
    for (Index j = 0; j < ns; ++j) {
        lo[sz(j)] = lp.col_lo[sz(j)];
        hi[sz(j)] = lp.col_hi[sz(j)];
    }
    for (Index i = 0; i < m; ++i) {
        lo[sz(ns + i)] = lp.row_lo[sz(i)];
        hi[sz(ns + i)] = lp.row_hi[sz(i)];
    }

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    std::vector<CutRow> cuts;
    constexpr f64 kZeroTol = 1e-11;

    for (Index slot = 0; slot < m &&
                        static_cast<int>(cuts.size()) < opts.max_candidates_per_round;
        ++slot) {
        const Index bj = basis.basic[sz(slot)];
        if (bj < 0 || bj >= ns) continue;  // cut only on structural basic vars
        if (lp.is_integer.empty() || !lp.is_integer[sz(bj)]) continue;

        const f64 beta = x[sz(bj)];
        const f64 f0 = beta - std::floor(beta);
        if (f0 < opts.frac_min || f0 > 1.0 - opts.frac_min) continue;
        ++diag.candidates_considered;

        // y = B^-T e_slot: row `slot` of the tableau is y' A (structural)
        // and -y (slack), computed with one CSR sweep below.
        std::vector<f64> y(sz(m), 0.0);
        y[sz(slot)] = 1.0;
        factor.btran(y);

        std::vector<f64> tab(sz(nt), 0.0);
        for (Index i = 0; i < m; ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                tab[sz(ci[sz(k)])] += y[sz(i)] * av[sz(k)];
        for (Index i = 0; i < m; ++i) tab[sz(ns + i)] = -y[sz(i)];

        std::vector<f64> struct_coef(sz(ns), 0.0);
        f64 rhs = 1.0;
        bool reject_free = false;
        for (Index j = 0; j < nt; ++j) {
            if (is_basic[sz(j)]) continue;
            const f64 alpha = tab[sz(j)];
            if (std::fabs(alpha) <= kZeroTol) continue;
            const auto st = basis.status[sz(j)];
            if (st == engines::NonbasicStatus::AtZeroFree) {
                reject_free = true;
                ++diag.rejected_free_nonbasic;
                break;
            }
            const bool at_lower = (st == engines::NonbasicStatus::AtLower);
            const f64 gamma = at_lower ? alpha : -alpha;
            const bool is_int_var =
                (j < ns) && !lp.is_integer.empty() && lp.is_integer[sz(j)];

            f64 coeff;
            if (is_int_var) {
                const f64 fj = gamma - std::floor(gamma);
                coeff = (fj <= f0) ? (fj / f0) : ((1.0 - fj) / (1.0 - f0));
            } else {
                coeff = (gamma >= 0.0) ? (gamma / f0) : (-gamma / (1.0 - f0));
            }
            if (coeff <= kZeroTol) continue;

            const f64 local = at_lower ? coeff : -coeff;
            if (at_lower) rhs += coeff * lo[sz(j)];
            else          rhs -= coeff * hi[sz(j)];

            if (j < ns) {
                struct_coef[sz(j)] += local;
            } else {
                const Index srow = j - ns;
                for (core::Offset k = rp[sz(srow)]; k < rp[sz(srow) + 1]; ++k)
                    struct_coef[sz(ci[sz(k)])] += local * av[sz(k)];
            }
        }
        if (reject_free) continue;

        std::vector<Index> cols;
        std::vector<f64> vals_out;
        f64 max_abs = 0.0, min_abs = std::numeric_limits<f64>::infinity();
        for (Index k = 0; k < ns; ++k) {
            const f64 v = struct_coef[sz(k)];
            if (std::fabs(v) <= kZeroTol) continue;
            cols.push_back(k);
            vals_out.push_back(v);
            max_abs = std::max(max_abs, std::fabs(v));
            min_abs = std::min(min_abs, std::fabs(v));
        }
        if (cols.empty()) continue;
        if (max_abs / std::max(min_abs, 1e-300) > opts.dynamism_max) {
            ++diag.rejected_dynamism;
            continue;
        }

        f64 activity = 0.0;
        for (std::size_t q = 0; q < cols.size(); ++q)
            activity += vals_out[q] * x[sz(cols[q])];
        const f64 violation = rhs - activity;
        if (violation < opts.violation_min) {
            ++diag.rejected_violation;
            continue;
        }

        CutRow cut;
        cut.cols = std::move(cols);
        cut.vals = std::move(vals_out);
        cut.row_lo = rhs;
        cut.row_hi = model::kInf;
        cut.name = "GMI_" + std::to_string(diag.gmi_cuts_added);
        cuts.push_back(std::move(cut));
        ++diag.gmi_cuts_added;
    }
    return cuts;
}

model::LpProblem apply_cuts(const model::LpProblem& lp,
                            const std::vector<CutRow>& cuts) {
    model::LpProblem out = lp;
    if (cuts.empty()) return out;

    const Index m = lp.n_rows();
    const Index n = lp.n_cols();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    rows.reserve(sz(lp.nnz()) + cuts.size() * 8);
    cols.reserve(rows.capacity());
    vals.reserve(rows.capacity());
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            rows.push_back(i);
            cols.push_back(ci[sz(k)]);
            vals.push_back(av[sz(k)]);
        }

    for (std::size_t q = 0; q < cuts.size(); ++q) {
        const Index r = m + static_cast<Index>(q);
        for (std::size_t t = 0; t < cuts[q].cols.size(); ++t) {
            rows.push_back(r);
            cols.push_back(cuts[q].cols[t]);
            vals.push_back(cuts[q].vals[t]);
        }
        out.row_lo.push_back(cuts[q].row_lo);
        out.row_hi.push_back(cuts[q].row_hi);
    }
    out.A = sparse::from_triplets(m + static_cast<Index>(cuts.size()), n,
                                  rows, cols, vals);
    out.row_names.resize(sz(m) + cuts.size());
    for (std::size_t q = 0; q < cuts.size(); ++q) {
        out.row_names[sz(m) + q] =
            cuts[q].name.empty() ? ("CUT_" + std::to_string(q)) : cuts[q].name;
    }
    return out;
}

}  // namespace sor::search
