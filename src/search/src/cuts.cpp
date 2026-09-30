#include "sor/search/cuts.hpp"
#include "sor/search/mir.hpp"
#include <bit>

#include "sor/la/lu.hpp"
#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include "sor/core/route_debug.hpp"

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { SOR_FN(); return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { SOR_FN(); return static_cast<std::size_t>(i); }

bool canonicalize(CutRow& cut) {
    SOR_FN();
    if (cut.cols.size() != cut.vals.size()) return false;
    std::vector<std::pair<Index, f64>> terms;
    terms.reserve(cut.cols.size());
    for (std::size_t k = 0; k < cut.cols.size(); ++k) {
        if (cut.cols[k] < 0 || !std::isfinite(cut.vals[k])) return false;
        if (cut.vals[k] != 0.0) terms.emplace_back(cut.cols[k], cut.vals[k]);
    }
    std::sort(terms.begin(), terms.end(),
              [](const auto& a, const auto& b) { SOR_FN(); return a.first < b.first; });
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
    std::size_t write = 0;
    for (std::size_t read = 0; read < cut.vals.size(); ++read) {
        if (cut.vals[read] != 0.0) {
            cut.vals[write] = cut.vals[read];
            cut.cols[write] = cut.cols[read];
            ++write;
        }
    }
    cut.vals.resize(write);
    cut.cols.resize(write);
    return !cut.cols.empty();
}

bool normalized_le(const CutRow& cut, std::vector<f64>& unit_vals,
                   f64& rhs_unit) {
    SOR_FN();
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
    SOR_FN();
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
    SOR_FN();
    return static_cast<std::size_t>(std::count_if(
        entries_.begin(), entries_.end(), [](const Entry& e) { SOR_FN(); return e.active; }));
}

void CutPool::start_round(CutDiagnostics& diag) {
    SOR_FN();
    for (auto& entry : entries_)
        if (!entry.active) ++entry.age;
    const auto old_size = entries_.size();
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& e) {
                                      SOR_FN();
                                      return !e.active && e.age > opts_.pool_max_age;
                                  }),
                   entries_.end());
    diag.pool_aged_out += old_size - entries_.size();
}

void CutPool::add(const std::vector<CutRow>& candidates,
                  CutDiagnostics& diag) {
    SOR_FN();
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
                std::swap(*it, entries_.back());
                entries_.pop_back();
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
                SOR_FN();
                if (a.active != b.active) return a.active; // inactive is preferred
                if (a.age != b.age) return a.age < b.age;
                return a.last_efficacy > b.last_efficacy;
            });
        if (victim == entries_.end()) break;
        std::swap(*victim, entries_.back());
        entries_.pop_back();
        ++diag.pool_evicted;
    }
}

void CutPool::set_scoring_context(const model::LpProblem& lp) {
    SOR_FN();
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
    SOR_FN();
    external_score_ = std::move(fn);
}

void CutPool::set_external_batch_scorer(ExternalBatchScoreFn fn) {
    SOR_FN();
    external_batch_score_ = std::move(fn);
}

std::vector<CutRow> CutPool::select_violated(const std::vector<f64>& x,
                                             CutDiagnostics& diag) {
    SOR_FN();
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
            // Malformed batch - fall back to efficacy so selection still runs.
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
            // Signed: an opposite-facing cut (anti-parallel normal) is not a
            // near-copy of the pick and is not penalised for it.
            const f64 cosine = std::max(0.0, sparse_dot(
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
bool relax_small_terms(std::vector<Index>& cols, std::vector<f64>& vals, f64& rhs,
                       bool geq, const std::vector<f64>& lo,
                       const std::vector<f64>& hi, f64 dynamism_max,
                       std::vector<std::pair<Index, f64>>* used_bounds) {
    if (cols.size() != vals.size() || cols.empty() || !(dynamism_max > 1.0))
        return false;
    f64 max_abs = 0.0;
    for (const f64 v : vals) max_abs = std::max(max_abs, std::fabs(v));
    if (!(max_abs > 0.0) || !std::isfinite(max_abs)) return false;
    const f64 thr = max_abs / dynamism_max;
    std::vector<Index> kept_cols;
    std::vector<f64> kept_vals;
    std::vector<std::pair<Index, f64>> used;
    f64 new_rhs = rhs;
    for (std::size_t k = 0; k < cols.size(); ++k) {
        const f64 v = vals[k];
        if (std::fabs(v) >= thr) {
            kept_cols.push_back(cols[k]);
            kept_vals.push_back(v);
            continue;
        }
        if (v == 0.0) continue;
        // The rest must still imply the cut, so the dropped term is replaced by
        // its LARGEST value in a >= cut (hi if v>0, else lo) and by its
        // SMALLEST in a <= cut (lo if v>0, else hi).
        const bool use_lower = geq ? v < 0.0 : v > 0.0;
        const f64 bound = use_lower ? lo[sz(cols[k])] : hi[sz(cols[k])];
        if (!std::isfinite(bound)) return false;
        new_rhs -= v * bound;
        used.emplace_back(cols[k], bound);
    }
    if (kept_cols.empty() || !std::isfinite(new_rhs)) return false;
    cols = std::move(kept_cols);
    vals = std::move(kept_vals);
    rhs = new_rhs;
    if (used_bounds != nullptr)
        used_bounds->insert(used_bounds->end(), used.begin(), used.end());
    return true;
}

namespace {

// One nonbasic (or the basic) variable of a tableau row, in the augmented
// [A | -I] space: z_v with coefficient a_v in  z_basic + sum a_v z_v = 0.
struct TabVar {
    Index z;          // column of the augmented space
    f64 a;            // row coefficient
    bool at_upper;    // shifted onto its upper bound (t = u - z)
    f64 bound;        // the bound the variable is shifted onto
    f64 t;            // shifted value at the LP point (>= 0)
    bool integral;    // t integral: integer variable, integral bound
    bool fixed = false;   // lo == hi: a constant, no term in the cut
};

// c-MIR on the tableau row in z-space. For each orientation sigma of the
// equality row  sigma * (z_b + sum a_v z_v) <= 0  and each scaling delta the
// Marchand-Wolsey inequality is formed over t = z - lo (or up - z) >= 0 and
// mapped back to z. Returns the best inequality  sum w[z] z <= rhs  by
// efficacy in z-space, or false if none is violated. Continuous terms with a
// positive coefficient are dropped (t >= 0), which is what makes the base a
// relaxation, exactly as in the row separator.
bool tableau_cmir_z(const std::vector<TabVar>& vars, Index basic_pos, f64 min_frac,
                    int max_scalings, std::vector<std::pair<Index, f64>>& w_out,
                    f64& rhs_out) {
    constexpr f64 kTol = 1e-9;
    std::vector<f64> deltas{1.0};
    {
        std::vector<f64> mags;
        for (const auto& v : vars)
            if (v.integral && std::fabs(v.a) > kTol) mags.push_back(std::fabs(v.a));
        std::sort(mags.begin(), mags.end(), std::greater<f64>());
        for (const f64 m : mags) {
            if (static_cast<int>(deltas.size()) > max_scalings) break;
            const f64 d = 1.0 / m;
            bool dup = false;
            for (const f64 e : deltas)
                if (std::fabs(e - d) <= 1e-9 * std::max(1.0, e)) { dup = true; break; }
            if (!dup) deltas.push_back(d);
        }
    }
    struct Best { f64 eff = -1.0; f64 sigma = 0.0, delta = 0.0; } best;
    const auto build = [&](f64 sigma, f64 delta, bool write,
                           std::vector<std::pair<Index, f64>>* w, f64* rhs_z) -> f64 {
        // Base: sum_v c_v t_v <= R over t >= 0.
        f64 R = 0.0;
        for (const auto& v : vars) {
            const f64 c0 = sigma * v.a;
            R -= c0 * v.bound;
        }
        R *= delta;
        const f64 fl = std::floor(R + kTol);
        const f64 f = R - fl;
        if (f < min_frac || f > 1.0 - min_frac) return -1.0;
        const f64 inv = 1.0 / (1.0 - f);
        f64 lhs = 0.0, norm2 = 0.0, rz = fl;
        for (std::size_t q = 0; q < vars.size(); ++q) {
            const auto& v = vars[q];
            const f64 c = delta * sigma * v.a * (v.at_upper ? -1.0 : 1.0);
            f64 g;
            if (v.integral) {
                const f64 fc = std::floor(c + kTol);
                g = fc + std::max(0.0, (c - fc) - f) * inv;
            } else {
                g = c < 0.0 ? c * inv : 0.0;
            }
            if (v.fixed || std::fabs(g) <= kTol) continue;
            if (!std::isfinite(g)) return -1.0;
            lhs += g * v.t;
            // t = z - bound (lower) or bound - z (upper).
            const f64 wz = v.at_upper ? -g : g;
            rz += v.at_upper ? -g * v.bound : g * v.bound;
            norm2 += wz * wz;
            if (write) w->emplace_back(v.z, wz);
        }
        (void)basic_pos;
        if (!(norm2 > 0.0)) return -1.0;
        const f64 viol = lhs - fl;
        if (write) *rhs_z = rz;
        return viol > 0.0 ? viol / std::sqrt(norm2) : -1.0;
    };
    const auto consider = [&](f64 sigma, f64 delta) {
        const f64 e = build(sigma, delta, false, nullptr, nullptr);
        if (e > best.eff) { best.eff = e; best.sigma = sigma; best.delta = delta; }
    };
    for (const f64 sigma : {1.0, -1.0})
        for (const f64 d : deltas) consider(sigma, d);
    if (best.eff <= 0.0) return false;
    const f64 star = best.delta, sig = best.sigma;
    for (const f64 mult : {2.0, 4.0, 8.0}) consider(sig, star * mult);
    w_out.clear();
    return build(best.sigma, best.delta, true, &w_out, &rhs_out) > 0.0;
}

}  // namespace

std::vector<CutRow> separate_gomory_mi(const model::LpProblem& lp,
                                       const std::vector<f64>& x,
                                       const engines::SimplexBasis& basis,
                                       const CutOptions& opts,
                                       CutDiagnostics& diag) {
    SOR_FN();
    const Index m = lp.n_rows();
    const Index ns = lp.n_cols();
    const Index nt = ns + m;
    if (basis.n_struct != ns || static_cast<Index>(basis.basic.size()) != m ||
        static_cast<Index>(basis.status.size()) != nt ||
        static_cast<Index>(x.size()) != ns) {
        ++diag.gmi_missing_basis;
        return {};
    }

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
    if (!factored || !factor.is_valid()) {
        ++diag.gmi_invalid_factor;
        return {};
    }

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

    std::vector<char> integral_activity(sz(m), 0);
    if (opts.integer_slack_gmi && lp.is_integer.size() == sz(ns)) {
        for (Index i = 0; i < m; ++i) {
            bool exact = true;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index j = ci[sz(k)];
                const f64 a = av[sz(k)];
                if (a == 0.0) continue;
                if (!lp.is_integer[sz(j)] || !std::isfinite(a) ||
                    std::fabs(a) > 0x1p52 || a != std::trunc(a)) {
                    exact = false;
                    break;
                }
            }
            integral_activity[sz(i)] = exact;
            if (exact) ++diag.integral_activity_rows;
        }
    }

    std::vector<CutRow> cuts;
    constexpr f64 kZeroTol = 1e-11;
    int cmir_attempts = 0;

    struct TableauCandidate {
        Index slot;
        f64 fraction;
        core::Offset column_nnz;
        f64 beta;
    };
    std::vector<TableauCandidate> tableau_candidates;
    for (Index slot = 0; slot < m; ++slot) {
        const Index bj = basis.basic[sz(slot)];
        if (bj < 0 || bj >= nt) continue;
        const bool basic_activity = bj >= ns;
        if (basic_activity ? (!opts.integer_activity_basic_gmi ||
                              !integral_activity[sz(bj - ns)])
                           : (lp.is_integer.empty() || !lp.is_integer[sz(bj)]))
            continue;

        f64 beta = 0.0;
        if (basic_activity) {
            const Index i = bj - ns;
            long double activity = 0.0L;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                activity += static_cast<long double>(av[sz(k)]) *
                            static_cast<long double>(x[sz(ci[sz(k)])]);
            beta = static_cast<f64>(activity);
        } else {
            beta = x[sz(bj)];
        }
        if (!std::isfinite(beta)) continue;
        const f64 f0 = beta - std::floor(beta);
        if (f0 < opts.frac_min || f0 > 1.0 - opts.frac_min) continue;
        const core::Offset column_nnz = basic_activity ? 1 : acp[sz(bj) + 1] - acp[sz(bj)];
        tableau_candidates.push_back({slot, f0, column_nnz, beta});
    }
    if (opts.rank_gmi_candidates) {
        // Fractions away from either integer need less MIR amplification.
        // Prefer the cheaper structural column on ties; no cut is accepted
        // without the unchanged validity and numerical filters below.
        std::stable_sort(tableau_candidates.begin(), tableau_candidates.end(),
            [](const TableauCandidate& a, const TableauCandidate& b) {
                const f64 fa = std::min(a.fraction, 1.0 - a.fraction);
                const f64 fb = std::min(b.fraction, 1.0 - b.fraction);
                if (fa != fb) return fa > fb;
                return a.column_nnz < b.column_nnz;
            });
    }
    int tableau_trials = 0;
    const auto gmi_started = std::chrono::steady_clock::now();
    for (const auto& candidate : tableau_candidates) {
        if (opts.time_limit_s > 0.0 && (tableau_trials & 7) == 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - gmi_started).count() >=
                opts.time_limit_s) {
            ++diag.time_stops;
            break;
        }
        if (static_cast<int>(cuts.size()) >= opts.max_candidates_per_round ||
            (opts.gmi_max_tableau_trials > 0 &&
             tableau_trials >= opts.gmi_max_tableau_trials)) break;
        const Index slot = candidate.slot;
        const Index bj = basis.basic[sz(slot)];
        const bool basic_activity = bj >= ns;
        const f64 f0 = candidate.fraction;
        const f64 beta = candidate.beta;
        ++tableau_trials;
        ++diag.candidates_considered;
        if (basic_activity) ++diag.integer_activity_candidates;

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

        CutRow gmi_cut;
        bool have_gmi = false;
        f64 gmi_eff = -1.0;
        // Outer-level `continue` below leaves this block (the condition is
        // false), so a refused GMI still falls through to the c-MIR attempt.
        do {
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
            const f64 active_bound = at_lower ? lo[sz(j)] : hi[sz(j)];
            // Complementing an integer variable about a fractional bound
            // makes the transformed variable nonintegral. Such a term must
            // use the continuous MIR coefficient, including for structural
            // integer columns that arrive with unsnapped bounds.
            const bool integral_bound = std::isfinite(active_bound) &&
                std::fabs(active_bound) <= 0x1p52 &&
                active_bound == std::trunc(active_bound);
            if (j < ns && lp.is_integer.size() == sz(ns) &&
                lp.is_integer[sz(j)] &&
                !integral_bound)
                ++diag.fractional_integer_bound_terms;
            const bool is_int_var = j < ns
                ? (lp.is_integer.size() == sz(ns) && lp.is_integer[sz(j)] &&
                   integral_bound)
                : (integral_bound && integral_activity[sz(j - ns)]);
            if (j >= ns && is_int_var) ++diag.integer_activity_terms;

            f64 coeff;
            if (is_int_var) {
                const f64 fj = gamma - std::floor(gamma);
                coeff = (fj <= f0) ? (fj / f0) : ((1.0 - fj) / (1.0 - f0));
            } else {
                coeff = (gamma >= 0.0) ? (gamma / f0) : (-gamma / (1.0 - f0));
            }
            if (coeff <= kZeroTol) {
                // A5: dropping this term silently is exact only when the
                // coeff*bound contribution it would have added to rhs
                // (lines below, for a kept term) is itself exactly 0. A
                // rounded-to-~0 coefficient times a large or infinite bound
                // is not negligible, and omitting it can produce a cut that
                // excludes a feasible point. Refuse the candidate rather
                // than risk that -- cheaper and safer than deriving the
                // exact relaxation for every case here.
                const f64 bound = at_lower ? lo[sz(j)] : hi[sz(j)];
                if (bound != 0.0) {
                    reject_free = true;
                    ++diag.rejected_dropped_term_bound;
                    break;
                }
                continue;
            }

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
            if (std::fabs(v) <= kZeroTol) {
                if (v == 0.0) continue;
                // Dropping v*x from a >= cut requires subtracting its
                // maximum box contribution from the right-hand side.
                const f64 bound = v > 0.0 ? lp.col_hi[sz(k)] : lp.col_lo[sz(k)];
                if (!std::isfinite(bound)) {
                    reject_free = true;
                    ++diag.rejected_dropped_term_bound;
                    break;
                }
                rhs -= v * bound;
                continue;
            }
            cols.push_back(k);
            vals_out.push_back(v);
            max_abs = std::max(max_abs, std::fabs(v));
            min_abs = std::min(min_abs, std::fabs(v));
        }
        if (reject_free) continue;
        if (cols.empty()) {
            ++diag.gmi_empty_rows;
            continue;
        }
        if (max_abs / std::max(min_abs, 1e-300) > opts.dynamism_max &&
            opts.relax_small_terms &&
            relax_small_terms(cols, vals_out, rhs, true, lp.col_lo, lp.col_hi,
                              opts.dynamism_max)) {
            ++diag.dynamism_repaired;
            max_abs = 0.0;
            min_abs = std::numeric_limits<f64>::infinity();
            for (const f64 v : vals_out) {
                max_abs = std::max(max_abs, std::fabs(v));
                min_abs = std::min(min_abs, std::fabs(v));
            }
        }
        if (max_abs / std::max(min_abs, 1e-300) > opts.dynamism_max) {
            bool recovered = false;
            if (opts.gmi_cmir_recovery &&
                cmir_attempts < opts.max_cmir_attempts_per_round) {
                ++cmir_attempts;
                ++diag.cmir_attempted;
                MirOptions mir;
                mir.max_dynamism = opts.dynamism_max;
                mir.violation_min = opts.violation_min;
                MirDiagnostics md;
                std::vector<Index> new_cols;
                std::vector<f64> new_vals;
                f64 new_rhs = 0.0;
                // The completed GMI is valid over this LP's integer hull.
                // Re-rounding that inequality and model VUBs preserves its
                // validity here; CutRow's default local scope is retained.
                recovered = apply_cmir_geq(lp, cols, vals_out, rhs, x,
                    lp.col_lo, lp.col_hi, mir, true, new_cols, new_vals,
                    new_rhs, md);
                if (recovered) {
                    cols = std::move(new_cols);
                    vals_out = std::move(new_vals);
                    rhs = new_rhs;
                    ++diag.cmir_recovered;
                }
            }
            if (!recovered) {
                ++diag.rejected_dynamism;
                continue;
            }
        }

        f64 activity = 0.0;
        for (std::size_t q = 0; q < cols.size(); ++q)
            activity += vals_out[q] * x[sz(cols[q])];
        const f64 violation = rhs - activity;
        if (violation < opts.violation_min) {
            ++diag.rejected_violation;
            continue;
        }

        f64 gnorm2 = 0.0;
        for (const f64 v : vals_out) gnorm2 += v * v;
        gmi_cut.cols = std::move(cols);
        gmi_cut.vals = std::move(vals_out);
        gmi_cut.row_lo = rhs;
        gmi_cut.row_hi = model::kInf;
        gmi_eff = violation / std::sqrt(gnorm2);
        have_gmi = true;
        } while (false);

        if (opts.tableau_cmir) {
            // The same tableau row as a c-MIR base over the shifted variables.
            std::vector<TabVar> tv;
            bool ok = true;
            const auto integral_z = [&](Index z) {
                return z < ns ? (lp.is_integer.size() == sz(ns) && lp.is_integer[sz(z)])
                              : static_cast<bool>(integral_activity[sz(z - ns)]);
            };
            const auto is_integral_bound = [](f64 b) {
                return std::isfinite(b) && std::fabs(b) <= 0x1p52 && b == std::trunc(b);
            };
            {
                // Basic variable: coefficient 1, shifted onto its nearer bound.
                const f64 l = lo[sz(bj)], u = hi[sz(bj)];
                TabVar b{};
                b.z = bj; b.a = 1.0;
                if (std::isfinite(l) && std::isfinite(u)) b.at_upper = (u - beta) < (beta - l);
                else if (std::isfinite(l)) b.at_upper = false;
                else if (std::isfinite(u)) b.at_upper = true;
                else ok = false;
                if (ok) {
                    b.bound = b.at_upper ? u : l;
                    b.t = std::max(0.0, b.at_upper ? u - beta : beta - l);
                    b.integral = integral_z(bj) && is_integral_bound(b.bound);
                    tv.push_back(b);
                }
            }
            for (Index j = 0; ok && j < nt; ++j) {
                if (is_basic[sz(j)]) continue;
                const f64 alpha = tab[sz(j)];
                // Terms are kept exactly, however small: a tiny coefficient on
                // a wide column still moves the base, and the cut's own
                // dynamism repair decides (with finite bounds) whether it may
                // be relaxed away. Only roundoff noise is ignored.
                if (std::fabs(alpha) <= 1e-13) continue;
                const auto st = basis.status[sz(j)];
                if (st == engines::NonbasicStatus::AtZeroFree) { ok = false; break; }
                TabVar v{};
                v.z = j; v.a = alpha;
                v.at_upper = (st != engines::NonbasicStatus::AtLower);
                v.bound = v.at_upper ? hi[sz(j)] : lo[sz(j)];
                if (!std::isfinite(v.bound)) { ok = false; break; }
                v.t = 0.0;
                v.fixed = lo[sz(j)] == hi[sz(j)];
                v.integral = integral_z(j) && is_integral_bound(v.bound);
                tv.push_back(v);
            }
            std::vector<std::pair<Index, f64>> wz;
            f64 rz = 0.0;
            if (ok && tv.size() >= 2 &&
                tableau_cmir_z(tv, 0, std::max(opts.frac_min, 1e-4),
                               opts.tableau_cmir_max_scalings, wz, rz)) {
                ++diag.tableau_cmir_tried;
                std::vector<f64> sc(sz(ns), 0.0);
                for (const auto& [z, w] : wz) {
                    if (z < ns) { sc[sz(z)] += w; continue; }
                    const Index srow = z - ns;
                    for (core::Offset k = rp[sz(srow)]; k < rp[sz(srow) + 1]; ++k)
                        sc[sz(ci[sz(k)])] += w * av[sz(k)];
                }
                // sum sc x <= rz  ->  sum (-sc) x >= -rz.
                std::vector<Index> cc;
                std::vector<f64> vv;
                for (Index k = 0; k < ns; ++k)
                    if (sc[sz(k)] != 0.0) { cc.push_back(k); vv.push_back(-sc[sz(k)]); }
                f64 crhs = -rz;
                bool good = !cc.empty();
                if (good) {
                    f64 mx = 0.0, mn = std::numeric_limits<f64>::infinity();
                    for (const f64 v : vv) { mx = std::max(mx, std::fabs(v)); mn = std::min(mn, std::fabs(v)); }
                    if (mx / mn > opts.dynamism_max)
                        good = opts.relax_small_terms &&
                               relax_small_terms(cc, vv, crhs, true, lp.col_lo,
                                                 lp.col_hi, opts.dynamism_max);
                }
                if (good) {
                    f64 act = 0.0, n2 = 0.0;
                    for (std::size_t q = 0; q < cc.size(); ++q) {
                        act += vv[q] * x[sz(cc[q])];
                        n2 += vv[q] * vv[q];
                    }
                    const f64 viol = crhs - act;
                    const f64 eff = n2 > 0.0 ? viol / std::sqrt(n2) : -1.0;
                    if (viol >= opts.violation_min && eff > gmi_eff * (1.0 + 1e-9)) {
                        ++diag.tableau_cmir_cuts;
                        if (have_gmi) ++diag.tableau_cmir_won;
                        else ++diag.tableau_cmir_only;
                        gmi_cut.cols = std::move(cc);
                        gmi_cut.vals = std::move(vv);
                        gmi_cut.row_lo = crhs;
                        gmi_cut.row_hi = model::kInf;
                        gmi_eff = eff;
                        have_gmi = true;
                    }
                }
            }
        }
        if (!have_gmi) continue;
        gmi_cut.name = "GMI_" + std::to_string(diag.gmi_cuts_added);
        cuts.push_back(std::move(gmi_cut));
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
    SOR_FN();
    scale = vals.empty() ? 0.0 : vals.front();
    if (scale == 0.0 || !std::isfinite(scale) || cols.size() != vals.size()) return {};
    std::string key;
    key.reserve(cols.size() * (sizeof(Index) + sizeof(std::uint64_t)));
    for (std::size_t k = 0; k < cols.size(); ++k) {
        const f64 normalized = vals[k] / scale;
        if (!std::isfinite(normalized)) return {};
        const auto bits = std::bit_cast<std::uint64_t>(normalized);
        key.append(reinterpret_cast<const char*>(&cols[k]), sizeof(Index));
        key.append(reinterpret_cast<const char*>(&bits), sizeof(bits));
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
    SOR_FN();
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

void apply_cuts_inplace(model::LpProblem& lp,
                        const std::vector<CutRow>& cuts,
                        const CutOptions& opts,
                        CutUndo* undo) {
    SOR_FN();
    if (undo != nullptr) {
        *undo = CutUndo{};
        undo->rows_before = lp.n_rows();
        undo->names_were_empty = lp.row_names.empty();
    }
    if (cuts.empty()) return;

    std::vector<std::vector<Index>> col_rows(sz(lp.n_cols()));
    {
        const auto& rp0 = lp.A.pattern.row_ptr();
        const auto& ci0 = lp.A.pattern.col_idx();
        for (Index i = 0; i < lp.n_rows(); ++i)
            for (core::Offset k = rp0[sz(i)]; k < rp0[sz(i) + 1]; ++k)
                col_rows[sz(ci0[sz(k)])].push_back(i);
    }

    const Index m0 = lp.n_rows();
    const Index n = lp.n_cols();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    // Only the appended cut suffix can participate in a merge. Original
    // rows retain both their coefficients and their bounds byte for byte.
    const Index first_cut = opts.original_rows < 0 ? m0
        : std::clamp(opts.original_rows, Index{0}, m0);
    std::unordered_map<std::string, Index> shape_of;
    shape_of.reserve(sz(m0) * 2);
    std::vector<Index> rcols;
    std::vector<f64> rvals;
    for (Index i = first_cut; i < m0; ++i) {
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

    // Pending appends are not yet in `lp.A` / `lp.row_*`. shape_of may point
    // either at an existing cut row (< m0) or at a pending slot (>= m0).
    // Merging into a pending slot must tighten that CutRow's bounds.
    std::vector<CutRow> pending;
    pending.reserve(cuts.size());
    auto cut_cols_in_range = [&](const CutRow& cut) -> bool {
        SOR_FN();
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
            // form as the cut. Map cut bounds into that scaling - FLIP when
            // the ratio is negative - and keep the tighter side of each.
            const Index i = it->second;
            f64 rsc = 0.0;
            if (i < m0) {
                rcols.clear();
                rvals.clear();
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    rcols.push_back(ci[sz(k)]);
                    rvals.push_back(av[sz(k)]);
                }
                row_signature(rcols, rvals, rsc);
            } else {
                const std::size_t q = static_cast<std::size_t>(i - m0);
                if (q >= pending.size()) continue;
                row_signature(pending[q].cols, pending[q].vals, rsc);
            }
            if (rsc == 0.0) continue;
            const f64 ratio = rsc / cs;
            f64 lo = cut.row_lo * ratio, hi = cut.row_hi * ratio;
            if (ratio < 0.0) std::swap(lo, hi);
            if (i < m0) {
                if (undo != nullptr) {
                    // First touch of this row in this round: save the bounds
                    // it had on entry. A later cut may tighten the same row
                    // again; the entry recorded here is still the pre-round
                    // state, which is what a retraction has to restore.
                    bool seen = false;
                    for (const Index r : undo->tightened_rows)
                        if (r == i) { seen = true; break; }
                    if (!seen) {
                        undo->tightened_rows.push_back(i);
                        undo->tightened_lo.push_back(lp.row_lo[sz(i)]);
                        undo->tightened_hi.push_back(lp.row_hi[sz(i)]);
                    }
                }
                if (std::isfinite(lo))
                    lp.row_lo[sz(i)] = std::max(lp.row_lo[sz(i)], lo);
                if (std::isfinite(hi))
                    lp.row_hi[sz(i)] = std::min(lp.row_hi[sz(i)], hi);
            } else {
                CutRow& dest = pending[static_cast<std::size_t>(i - m0)];
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
            shape_of.emplace(key, m0 + static_cast<Index>(pending.size()) - 1);
    }

    if (pending.empty()) return;
    if (undo != nullptr)
        undo->rows_added_ = static_cast<Index>(pending.size());
    for (std::size_t q = 0; q < pending.size(); ++q) {
        lp.A.append_row(pending[q].cols, pending[q].vals);
        lp.row_lo.push_back(pending[q].row_lo);
        lp.row_hi.push_back(pending[q].row_hi);
    }
    // Only materialize names when the source already had them, or when we
    // need labels for the new cuts; keep vector length == n_rows().
    if (!lp.row_names.empty() || !pending.empty()) {
        lp.row_names.resize(sz(m0) + pending.size());
        for (std::size_t q = 0; q < pending.size(); ++q) {
            lp.row_names[sz(m0) + q] = pending[q].name.empty()
                ? ("CUT_" + std::to_string(q)) : pending[q].name;
        }
    }
}

model::LpProblem apply_cuts(const model::LpProblem& lp,
                            const std::vector<CutRow>& cuts,
                            const CutOptions& opts) {
    SOR_FN();
    model::LpProblem out = lp;
    apply_cuts_inplace(out, cuts, opts);
    return out;
}

void retract_cuts_inplace(model::LpProblem& lp, const CutUndo& undo) {
    SOR_FN();
    // Bounds first: these rows are all in the retained prefix, so the order
    // relative to the truncation does not matter, but doing it first keeps the
    // model consistent at every point.
    for (std::size_t k = 0; k < undo.tightened_rows.size(); ++k) {
        const std::size_t r = static_cast<std::size_t>(undo.tightened_rows[k]);
        if (r >= lp.row_lo.size()) continue;
        lp.row_lo[r] = undo.tightened_lo[k];
        lp.row_hi[r] = undo.tightened_hi[k];
    }
    const Index keep = undo.rows_before;
    if (lp.n_rows() <= keep) return;
    lp.A.truncate_rows(keep);
    lp.row_lo.resize(sz(keep));
    lp.row_hi.resize(sz(keep));
    // apply_cuts_inplace materializes row_names when it appends, even for a
    // model that had none. Put that back too, or the name vector outlives the
    // rows it was created for.
    if (undo.names_were_empty) lp.row_names.clear();
    else if (!lp.row_names.empty()) lp.row_names.resize(sz(keep));
}

}  // namespace sor::search
