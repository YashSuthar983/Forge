#include "sor/search/cuts.hpp"

#include "sor/la/lu.hpp"
#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>

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
        // Reject cuts whose support is outside the scoring context (stale
        // after a column remap / empty context with absurd indices).
        if (n_cols_ > 0) {
            bool oob = false;
            for (const Index j : incoming.cut.cols) {
                if (j < 0 || j >= n_cols_) { oob = true; break; }
            }
            if (oob) continue;
        }

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

void CutPool::set_scoring_context(const model::LpProblem& lp) {
    lp_ctx_ = &lp;
    is_integer_ = lp.is_integer;
    n_cols_ = lp.n_cols();

    // Locks (Achterberg 2007): how many rows would be violated by moving a
    // column down, and how many by moving it up. A cut on columns that few
    // rows already restrict is more likely to say something new, which is why
    // the score below uses the COMPLEMENT of this measure.
    locks_.assign(sz(lp.n_cols()), 0.0);
    {
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        const auto& av = lp.A.vals;
        for (Index i = 0; i < lp.n_rows(); ++i) {
            const bool has_lo = std::isfinite(lp.row_lo[sz(i)]);
            const bool has_hi = std::isfinite(lp.row_hi[sz(i)]);
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const f64 a = av[sz(k)];
                if (a == 0.0) continue;
                const Index j = ci[sz(k)];
                // Decreasing x_j lowers the activity when a > 0, which can
                // break a finite lower bound; and raises it when a < 0.
                if ((a > 0.0 && has_lo) || (a < 0.0 && has_hi)) locks_[sz(j)] += 1.0;
                if ((a > 0.0 && has_hi) || (a < 0.0 && has_lo)) locks_[sz(j)] += 1.0;
            }
        }
        max_locks_ = 0.0;
        for (const f64 v : locks_) max_locks_ = std::max(max_locks_, v);
    }

    obj_unit_.assign(sz(lp.n_cols()), 0.0);
    f64 norm = 0.0;
    for (const f64 v : lp.c) norm += v * v;
    if (!(norm > 0.0)) { obj_unit_.clear(); return; }
    norm = std::sqrt(norm);
    for (std::size_t j = 0; j < lp.c.size() && j < obj_unit_.size(); ++j)
        obj_unit_[j] = lp.c[j] / norm;
}

void CutPool::set_external_scorer(ExternalScoreFn fn) {
    external_score_ = std::move(fn);
}

void CutPool::set_external_batch_scorer(ExternalBatchScoreFn fn) {
    external_batch_score_ = std::move(fn);
}

std::vector<CutRow> CutPool::select_violated(const std::vector<f64>& x,
                                             CutDiagnostics& diag) {
    if (opts_.max_cuts_per_round <= 0) return {};
    std::vector<std::size_t> order;
    order.reserve(entries_.size());
    f64 max_eff = 0.0;
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
        if (entry.last_efficacy < opts_.pool_efficacy_min) continue;
        max_eff = std::max(max_eff, entry.last_efficacy);
        order.push_back(i);
    }

    // Efficacy is a DISTANCE and can be arbitrarily large, while every other
    // term below is a ratio in [0, 1]. Turner et al. normalise it within the
    // separation round for exactly this reason: without that, the weights and
    // the parallelism penalty are measured against an unknown scale and mean
    // nothing. This is not cosmetic -- an unnormalised efficacy of 1.5 shrugs
    // off a 0.5 penalty that is meant to be decisive.
    //
    // HGTSM (Latest): batch graph/sequence scorer takes precedence; else
    // per-cut external scorer; else efficacy composite. Violation gate above
    // still rejects invalid / non-cutting rows; density and parallelism
    // filters below stay.
    const bool use_batch = static_cast<bool>(external_batch_score_);
    const bool use_ext = !use_batch && static_cast<bool>(external_score_);
    if (use_batch && !order.empty()) {
        std::vector<CutRow> batch;
        batch.reserve(order.size());
        for (const std::size_t i : order) batch.push_back(entries_[i].cut);
        std::vector<f64> batch_scores;
        external_batch_score_(batch, x, batch_scores);
        if (batch_scores.size() == order.size()) {
            for (std::size_t k = 0; k < order.size(); ++k)
                entries_[order[k]].last_score = batch_scores[k];
        } else {
            // Malformed batch — fall back to efficacy so selection still runs.
            for (const std::size_t i : order) {
                auto& entry = entries_[i];
                entry.last_score =
                    max_eff > 0.0 ? entry.last_efficacy / max_eff : 0.0;
            }
        }
    } else {
    for (const std::size_t i : order) {
        auto& entry = entries_[i];
        f64 score = 0.0;
        if (use_ext) {
            score = external_score_(entry.cut, x);
        } else {
            score = max_eff > 0.0 ? entry.last_efficacy / max_eff : 0.0;
            // unit_vals is already the cut normalised to unit length, so the
            // objective term is a plain dot product and lands in [0, 1], as does
            // the integer-support fraction.
            if (!obj_unit_.empty()) {
                f64 dot = 0.0;
                for (std::size_t k = 0; k < entry.cut.cols.size(); ++k) {
                    const Index j = entry.cut.cols[k];
                    if (j >= 0 && sz(j) < obj_unit_.size())
                        dot += entry.unit_vals[k] * obj_unit_[sz(j)];
                }
                score += opts_.pool_weight_objective_parallelism * std::fabs(dot);
            }
            if (!is_integer_.empty() && !entry.cut.cols.empty()) {
                std::size_t ints = 0;
                for (const Index j : entry.cut.cols)
                    if (j >= 0 && sz(j) < is_integer_.size() && is_integer_[sz(j)])
                        ++ints;
                score += opts_.pool_weight_integer_support *
                         (static_cast<f64>(ints) /
                          static_cast<f64>(entry.cut.cols.size()));
            }
            if (n_cols_ > 0) {
                // Sparsity score (Turner et al. 2.1.3): a linear ramp rewarding
                // sparse cuts, reaching zero at `pool_sparsity_end_density`. A
                // dense cut is paid for on every node LP that follows it, so
                // density is a cost the violation alone does not price.
                const f64 density = static_cast<f64>(entry.cut.cols.size()) /
                                    static_cast<f64>(n_cols_);
                const f64 end = std::max(1e-9, opts_.pool_sparsity_end_density);
                score += opts_.pool_weight_sparsity *
                         std::max(0.0, 1.0 - density / end);
            }
            if (max_locks_ > 0.0 && !entry.cut.cols.empty()) {
                // Lock score, COMPLEMENTED (Turner et al. 2.1.2): a cut on columns
                // that few other rows already constrain is more likely to add
                // something the relaxation does not already know.
                f64 sum = 0.0;
                for (const Index j : entry.cut.cols)
                    if (j >= 0 && sz(j) < locks_.size()) sum += locks_[sz(j)];
                const f64 mean = sum / static_cast<f64>(entry.cut.cols.size());
                score += opts_.pool_weight_low_locks *
                         std::max(0.0, 1.0 - mean / max_locks_);
            }
        }
        entry.last_score = score;
    }
    }  // !use_batch
    // Density filter, applied BEFORE selection (Turner et al. 2.2.1): a cut
    // denser than the threshold is removed outright, because its cost lands on
    // every subsequent node LP whatever its score says.
    if (n_cols_ > 0 && opts_.pool_max_density > 0.0) {
        std::vector<std::size_t> dense_ok;
        dense_ok.reserve(order.size());
        const std::size_t dense_nnz = std::max<std::size_t>(
            opts_.pool_min_dense_nnz,
            static_cast<std::size_t>(opts_.pool_max_density *
                                     static_cast<f64>(n_cols_)));
        for (const std::size_t i : order) {
            if (entries_[i].cut.cols.size() <= dense_nnz) dense_ok.push_back(i);
            else ++diag.pool_rejected_dense;
        }
        order.swap(dense_ok);
    }

    // Greedy selection with parallelism PENALTIES rather than hard filtering
    // (Turner et al. 2.2.2), under a budget on nonzeros added this round rather
    // than on the number of cuts (2.3).
    std::vector<f64> score(entries_.size(), 0.0);
    for (const std::size_t i : order) score[i] = entries_[i].last_score;
    std::vector<char> alive(entries_.size(), 0);
    for (const std::size_t i : order) alive[i] = 1;

    const f64 nnz_budget = opts_.pool_nnz_budget_factor > 0.0 && n_cols_ > 0
        ? std::max(static_cast<f64>(opts_.pool_min_nnz_budget),
                   opts_.pool_nnz_budget_factor * static_cast<f64>(n_cols_))
        : std::numeric_limits<f64>::infinity();
    f64 nnz_used = 0.0;

    std::vector<std::size_t> chosen;
    std::vector<CutRow> out;
    while (static_cast<int>(out.size()) < opts_.max_cuts_per_round) {
        std::size_t best = entries_.size();
        f64 best_score = -std::numeric_limits<f64>::infinity();
        for (const std::size_t i : order) {
            if (!alive[i]) continue;
            if (nnz_used + static_cast<f64>(entries_[i].cut.cols.size()) > nnz_budget)
                continue;
            if (score[i] > best_score) { best_score = score[i]; best = i; }
        }
        if (best == entries_.size()) break;

        alive[best] = 0;
        entries_[best].active = true;
        entries_[best].age = 0;
        chosen.push_back(best);
        nnz_used += static_cast<f64>(entries_[best].cut.cols.size());
        out.push_back(entries_[best].cut);
        ++diag.pool_selected;

        for (const std::size_t i : order) {
            if (!alive[i]) continue;
            const f64 cosine = std::fabs(sparse_dot(
                entries_[i].cut, entries_[i].unit_vals,
                entries_[best].cut, entries_[best].unit_vals));
            if (opts_.pool_parallel_hard_filter) {
                if (cosine > opts_.pool_parallelism_max) {
                    alive[i] = 0;
                    ++diag.pool_rejected_parallel;
                }
                continue;
            }
            if (cosine < opts_.pool_parallelism_penalty_min) continue;
            ++diag.pool_penalized_parallel;
            score[i] -= opts_.pool_parallelism_penalty * cosine;
            if (score[i] <= 0.0) {
                alive[i] = 0;
                ++diag.pool_rejected_parallel;
            }
        }
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

namespace {

// A scale- and sign-invariant signature of a row's SUPPORT AND SHAPE, so two
// rows that constrain the same linear form up to a positive multiple hash the
// same. `scale` is what the row was divided by, carrying the sign, which is
// what lets a bound be moved between the two representations exactly.
std::string row_signature(const std::vector<Index>& cols,
                          const std::vector<f64>& vals, f64& scale) {
    scale = 0.0;
    for (std::size_t k = 0; k < vals.size(); ++k)
        if (vals[k] != 0.0) { scale = vals[k]; break; }
    if (scale == 0.0) return {};
    std::string key;
    key.reserve(cols.size() * 20);
    char buf[48];
    for (std::size_t k = 0; k < cols.size(); ++k) {
        if (vals[k] == 0.0) continue;
        std::snprintf(buf, sizeof buf, "%d:%.9g|", static_cast<int>(cols[k]),
                      vals[k] / scale);
        key += buf;
    }
    return key;
}

}  // namespace

// Largest |cos| between `cut` and any model row sharing a column with it.
// Rows that share no column are orthogonal by construction, so the column
// index keeps this proportional to the cut's own support rather than to m.
f64 max_row_parallelism(const model::LpProblem& lp,
                        const std::vector<std::vector<Index>>& col_rows,
                        const CutRow& cut) {
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    f64 cn = 0.0;
    for (const f64 v : cut.vals) cn += v * v;
    if (!(cn > 0.0)) return 0.0;
    cn = std::sqrt(cn);

    const Index n = lp.n_cols();
    if (cut.cols.size() != cut.vals.size()) return 0.0;
    for (const Index j : cut.cols)
        if (j < 0 || j >= n) return 0.0;

    std::vector<f64> dense(sz(n), 0.0);
    for (std::size_t k = 0; k < cut.cols.size(); ++k)
        dense[sz(cut.cols[k])] = cut.vals[k];

    std::vector<char> seen;
    std::vector<Index> touched;
    seen.assign(sz(lp.n_rows()), 0);
    for (const Index j : cut.cols) {
        if (sz(j) >= col_rows.size()) continue;
        for (const Index i : col_rows[sz(j)])
            if (i >= 0 && i < lp.n_rows() && !seen[sz(i)]) {
                seen[sz(i)] = 1;
                touched.push_back(i);
            }
    }

    f64 best = 0.0;
    for (const Index i : touched) {
        f64 dot = 0.0, rn = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index cj = ci[sz(k)];
            if (cj < 0 || cj >= n) continue;
            dot += av[sz(k)] * dense[sz(cj)];
            rn += av[sz(k)] * av[sz(k)];
        }
        if (!(rn > 0.0)) continue;
        best = std::max(best, std::fabs(dot) / (cn * std::sqrt(rn)));
    }
    return best;
}

model::LpProblem apply_cuts(const model::LpProblem& lp,
                            const std::vector<CutRow>& cuts,
                            const CutOptions& opts) {
    model::LpProblem out = lp;
    if (cuts.empty()) return out;

    std::vector<std::vector<Index>> col_rows(sz(lp.n_cols()));
    {
        const auto& rp0 = lp.A.pattern.row_ptr();
        const auto& ci0 = lp.A.pattern.col_idx();
        for (Index i = 0; i < lp.n_rows(); ++i)
            for (core::Offset k = rp0[sz(i)]; k < rp0[sz(i) + 1]; ++k)
                col_rows[sz(ci0[sz(k)])].push_back(i);
    }

    const Index m = lp.n_rows();
    const Index n = lp.n_cols();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    // Index the existing rows by shape. A cut that constrains a linear form the
    // model ALREADY has a row for must not be appended as a second, parallel
    // row: the two are linearly dependent, one of them is immediately
    // redundant, and the pair is a direct source of dual degeneracy. Measured
    // on misc03, two such cuts -- 3-term set-packing rows x_a+x_b+x_c <= 1
    // whose support already appeared as a weaker cardinality cover <= 2 --
    // took the node relaxation from 11.8 to 80.3 simplex iterations per node
    // and cost the instance its proof. Tightening the existing row instead
    // keeps every bit of the cut's strength and adds no dependence at all.
    std::unordered_map<std::string, Index> shape_of;
    shape_of.reserve(sz(m) * 2);
    std::vector<Index> rcols;
    std::vector<f64> rvals;
    for (Index i = 0; i < m; ++i) {
        rcols.clear();
        rvals.clear();
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            rcols.push_back(ci[sz(k)]);
            rvals.push_back(av[sz(k)]);
        }
        f64 sc = 0.0;
        const std::string key = row_signature(rcols, rvals, sc);
        if (!key.empty()) shape_of.emplace(key, i);
    }

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

    // Pending appends are not yet in `lp.A` / `out.row_*`. shape_of may point
    // either at an existing model row (< m) or at a pending slot (>= m). Merging
    // into a pending slot must tighten that CutRow's bounds — never index `rp`.
    std::vector<CutRow> pending;
    pending.reserve(cuts.size());
    auto cut_cols_in_range = [&](const CutRow& cut) -> bool {
        if (cut.cols.size() != cut.vals.size()) return false;
        for (const Index j : cut.cols)
            if (j < 0 || j >= n) return false;
        return true;
    };
    for (const auto& cut : cuts) {
        if (!cut_cols_in_range(cut)) continue;
        f64 cs = 0.0;
        const std::string key = row_signature(cut.cols, cut.vals, cs);
        const auto it = key.empty() ? shape_of.end() : shape_of.find(key);
        if (it != shape_of.end() && cs != 0.0) {
            // Existing/pending row constrains (row_scale/cs) times the same
            // form as the cut. Map cut bounds into that scaling — FLIP when
            // the ratio is negative — and keep the tighter side of each.
            const Index i = it->second;
            f64 rsc = 0.0;
            if (i < m) {
                rcols.clear();
                rvals.clear();
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    rcols.push_back(ci[sz(k)]);
                    rvals.push_back(av[sz(k)]);
                }
                row_signature(rcols, rvals, rsc);
            } else {
                const std::size_t q = static_cast<std::size_t>(i - m);
                if (q >= pending.size()) continue;
                row_signature(pending[q].cols, pending[q].vals, rsc);
            }
            if (rsc == 0.0) continue;
            const f64 ratio = rsc / cs;
            f64 lo = cut.row_lo * ratio, hi = cut.row_hi * ratio;
            if (ratio < 0.0) std::swap(lo, hi);
            if (i < m) {
                if (std::isfinite(lo))
                    out.row_lo[sz(i)] = std::max(out.row_lo[sz(i)], lo);
                if (std::isfinite(hi))
                    out.row_hi[sz(i)] = std::min(out.row_hi[sz(i)], hi);
            } else {
                CutRow& dest = pending[static_cast<std::size_t>(i - m)];
                if (std::isfinite(lo))
                    dest.row_lo = std::max(dest.row_lo, lo);
                if (std::isfinite(hi))
                    dest.row_hi = std::min(dest.row_hi, hi);
            }
            continue;
        }
        // Near-parallel to an existing row: not exactly proportional, so the
        // exact merge above cannot fold it in, but close enough that adding it
        // buys almost no new direction while contributing near-dependence.
        if (max_row_parallelism(lp, col_rows, cut) > opts.parallel_to_row_max)
            continue;
        pending.push_back(cut);
        if (!key.empty())
            shape_of.emplace(key, m + static_cast<Index>(pending.size()) - 1);
    }

    if (pending.empty()) return out;
    for (std::size_t q = 0; q < pending.size(); ++q) {
        const Index r = m + static_cast<Index>(q);
        for (std::size_t t = 0; t < pending[q].cols.size(); ++t) {
            rows.push_back(r);
            cols.push_back(pending[q].cols[t]);
            vals.push_back(pending[q].vals[t]);
        }
        out.row_lo.push_back(pending[q].row_lo);
        out.row_hi.push_back(pending[q].row_hi);
    }
    out.A = sparse::from_triplets(m + static_cast<Index>(pending.size()), n,
                                  rows, cols, vals);
    // Only materialize names when the source already had them, or when we
    // need labels for the new cuts; keep vector length == n_rows().
    if (!out.row_names.empty() || !pending.empty()) {
        out.row_names.resize(sz(m) + pending.size());
        for (std::size_t q = 0; q < pending.size(); ++q) {
            out.row_names[sz(m) + q] = pending[q].name.empty()
                ? ("CUT_" + std::to_string(q)) : pending[q].name;
        }
    }
    return out;
}

}  // namespace sor::search
