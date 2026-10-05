#include "sor/search/conflict_cut.hpp"

#include "sor/search/mir.hpp"
#include "sor/search/propagate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <vector>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

f64 max_activity_geq(const GeqConstraint& c,
                     const std::vector<f64>& lo,
                     const std::vector<f64>& hi) {
    f64 s = 0.0;
    for (std::size_t k = 0; k < c.cols.size(); ++k) {
        const Index j = c.cols[k];
        const f64 a = c.vals[k];
        if (j < 0 || j >= static_cast<Index>(lo.size())) continue;
        s += (a >= 0.0) ? a * hi[sz(j)] : a * lo[sz(j)];
    }
    return s;
}

f64 min_activity_geq(const GeqConstraint& c,
                     const std::vector<f64>& lo,
                     const std::vector<f64>& hi) {
    f64 s = 0.0;
    for (std::size_t k = 0; k < c.cols.size(); ++k) {
        const Index j = c.cols[k];
        const f64 a = c.vals[k];
        if (j < 0 || j >= static_cast<Index>(lo.size())) continue;
        s += (a >= 0.0) ? a * lo[sz(j)] : a * hi[sz(j)];
    }
    return s;
}

bool geq_infeasible(const GeqConstraint& c,
                    const std::vector<f64>& lo,
                    const std::vector<f64>& hi,
                    f64 tol) {
    return max_activity_geq(c, lo, hi) < c.rhs - tol;
}

bool is_integer_col(const model::LpProblem& lp, Index j) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)];
}

bool is_binary_col(const model::LpProblem& lp, Index j, f64 tol) {
    if (!is_integer_col(lp, j)) return false;
    return lp.col_lo[sz(j)] >= -tol && lp.col_hi[sz(j)] <= 1.0 + tol;
}

bool reason_all_binary(const GeqConstraint& reason,
                       const model::LpProblem& lp,
                       f64 tol) {
    for (Index j : reason.cols) {
        if (!is_binary_col(lp, j, tol)) return false;
    }
    return !reason.cols.empty();
}

// Paper §2.1: xj relaxable for C under ρ iff actmax unchanged when local
// bounds of j are replaced by global. Equivalent: aj=0 or
// (aj>0 ∧ uρj=uj) or (aj<0 ∧ ℓρj=ℓj).
bool is_relaxable(const GeqConstraint& c, Index j,
                  const std::vector<f64>& glo, const std::vector<f64>& ghi,
                  const std::vector<f64>& lo, const std::vector<f64>& hi,
                  f64 tol) {
    const f64 a = [&]() {
        for (std::size_t k = 0; k < c.cols.size(); ++k)
            if (c.cols[k] == j) return c.vals[k];
        return 0.0;
    }();
    if (std::fabs(a) <= tol) return true;
    if (a > tol) return std::fabs(hi[sz(j)] - ghi[sz(j)]) <= tol;
    return std::fabs(lo[sz(j)] - glo[sz(j)]) <= tol;
}

bool extract_row_geq_upper(const model::LpProblem& lp, Index row,
                           GeqConstraint& out) {
    out.cols.clear();
    out.vals.clear();
    out.rhs = 0.0;
    if (row < 0 || row >= lp.n_rows()) return false;
    if (!std::isfinite(lp.row_hi[sz(row)])) return false;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (core::Offset k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = lp.A.vals[sz(k)];
        if (std::fabs(a) <= 1e-15) continue;
        out.cols.push_back(j);
        out.vals.push_back(-a);
    }
    out.rhs = -lp.row_hi[sz(row)];
    return !out.cols.empty();
}

bool extract_row_geq_lower(const model::LpProblem& lp, Index row,
                           GeqConstraint& out) {
    out.cols.clear();
    out.vals.clear();
    out.rhs = 0.0;
    if (row < 0 || row >= lp.n_rows()) return false;
    if (!std::isfinite(lp.row_lo[sz(row)])) return false;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (core::Offset k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = lp.A.vals[sz(k)];
        if (std::fabs(a) <= 1e-15) continue;
        out.cols.push_back(j);
        out.vals.push_back(a);
    }
    out.rhs = lp.row_lo[sz(row)];
    return !out.cols.empty();
}

f64 coeff_at(const GeqConstraint& c, Index j) {
    for (std::size_t k = 0; k < c.cols.size(); ++k)
        if (c.cols[k] == j) return c.vals[k];
    return 0.0;
}

// Orient the model row so that BoundDir matches propagation sign in >= form:
// Lower ⇒ positive coeff on var; Upper ⇒ negative coeff.
bool extract_reason_geq(const model::LpProblem& lp, Index row, Index var,
                        BoundDir dir, GeqConstraint& out) {
    GeqConstraint up, lo;
    const bool has_up = extract_row_geq_upper(lp, row, up);
    const bool has_lo = extract_row_geq_lower(lp, row, lo);
    const f64 want = (dir == BoundDir::Lower) ? 1.0 : -1.0;
    if (has_up) {
        const f64 c = coeff_at(up, var);
        if (c * want > 0.0) {
            out = std::move(up);
            return true;
        }
    }
    if (has_lo) {
        const f64 c = coeff_at(lo, var);
        if (c * want > 0.0) {
            out = std::move(lo);
            return true;
        }
    }
    // Fallback: any finite side that mentions var.
    if (has_up && std::fabs(coeff_at(up, var)) > 0.0) {
        out = std::move(up);
        return true;
    }
    if (has_lo && std::fabs(coeff_at(lo, var)) > 0.0) {
        out = std::move(lo);
        return true;
    }
    return false;
}

bool branch_as_geq(const PropTrailEntry& e, GeqConstraint& out) {
    out.cols = {e.var};
    if (e.dir == BoundDir::Lower) {
        out.vals = {1.0};
        out.rhs = e.new_bound;
    } else {
        out.vals = {-1.0};
        out.rhs = -e.new_bound;
    }
    return e.var >= 0;
}

bool entry_as_reason(const model::LpProblem& lp, const PropTrailEntry& e,
                     GeqConstraint& out) {
    if (e.kind == ReasonKind::Row || e.kind == ReasonKind::Cut ||
        e.kind == ReasonKind::Conflict) {
        return extract_reason_geq(lp, e.reason_id, e.var, e.dir, out);
    }
    if (e.kind == ReasonKind::Branch) return branch_as_geq(e, out);
    return false;
}

GeqConstraint add_geq(const GeqConstraint& a, f64 sa,
                      const GeqConstraint& b, f64 sb) {
    std::map<Index, f64> acc;
    for (std::size_t i = 0; i < a.cols.size(); ++i)
        acc[a.cols[i]] += sa * a.vals[i];
    for (std::size_t i = 0; i < b.cols.size(); ++i)
        acc[b.cols[i]] += sb * b.vals[i];
    GeqConstraint out;
    out.rhs = sa * a.rhs + sb * b.rhs;
    for (const auto& p : acc) {
        if (!std::isfinite(p.second)) continue;  // inf*0-style NaN guard
        if (std::fabs(p.second) <= 1e-12) continue;
        out.cols.push_back(p.first);
        out.vals.push_back(p.second);
    }
    return out;
}

// True iff every coefficient and the rhs are finite and of sane magnitude.
// Resolution / tightening must never propagate non-finite values: gen-ip002
// (2026-09-14) showed inf/NaN leaking into learned cuts and poisoning duals.
bool geq_finite(const GeqConstraint& c) {
    if (!std::isfinite(c.rhs) || std::fabs(c.rhs) > 1e12) return false;
    for (f64 v : c.vals)
        if (!std::isfinite(v) || std::fabs(v) > 1e9) return false;
    return true;
}

bool resolve_geq(GeqConstraint& learn, const GeqConstraint& reason, Index var,
                 f64 tol) {
    const f64 cr = coeff_at(reason, var);
    const f64 lr = coeff_at(learn, var);
    if (std::fabs(cr) <= tol || std::fabs(lr) <= tol) return false;
    // Two >= inequalities combine into a valid consequence that eliminates
    // `var` only via a nonnegative Farkas combination after rewriting as <=,
    // which requires opposite signs on `var`. Same-sign learn-(lr/cr)*reason
    // is not a valid implication and can cut off feasible MIP points.
    if (lr * cr > 0.0) return false;
    const f64 mult = lr / cr;  // strictly negative
    GeqConstraint out = add_geq(learn, 1.0, reason, -mult);
    if (!geq_finite(out)) return false;  // non-finite resolvent: refuse
    learn = std::move(out);
    return true;
}

void compact_geq(GeqConstraint& c, f64 tol) {
    GeqConstraint out;
    out.rhs = c.rhs;
    for (std::size_t k = 0; k < c.cols.size(); ++k) {
        if (std::fabs(c.vals[k]) <= tol) continue;
        out.cols.push_back(c.cols[k]);
        out.vals.push_back(c.vals[k]);
    }
    c = std::move(out);
}

// Definition 1 (Brearley / Mexi): tighten integer coeffs using global bounds.
// Sound only when every bound the substitution touches is FINITE: with an
// unbounded side the (a - btilde) * bound adjustment diverges and the
// "tightened" row stops being implied (gen-ip002 P0, 2026-09-14). Columns
// without a finite needed bound are simply not tightened - capping is an
// optional strengthening, skipping it is always sound.
void apply_coef_tightening(GeqConstraint& c,
                           const model::LpProblem& lp,
                           f64 tol) {
    const f64 actmin = min_activity_geq(c, lp.col_lo, lp.col_hi);
    if (!std::isfinite(actmin)) return;
    if (!(actmin < c.rhs - tol)) return;
    if (!std::isfinite(c.rhs)) return;
    const f64 btilde = c.rhs - actmin;
    if (!(btilde > tol)) return;

    f64 rhs_adj = 0.0;
    for (std::size_t k = 0; k < c.cols.size(); ++k) {
        const Index j = c.cols[k];
        if (!is_integer_col(lp, j)) continue;
        f64 a = c.vals[k];
        if (a > tol) {
            const f64 ell = lp.col_lo[sz(j)];
            if (a > btilde + tol) {
                if (!std::isfinite(ell)) continue;  // unbounded below: skip
                rhs_adj += (a - btilde) * ell;
                c.vals[k] = btilde;
            }
        } else if (a < -tol) {
            const f64 u = lp.col_hi[sz(j)];
            const f64 ap = -a;
            if (ap > btilde + tol) {
                if (!std::isfinite(u)) continue;  // unbounded above: skip
                rhs_adj += (ap - btilde) * (-u);
                c.vals[k] = -btilde;
            }
        }
    }
    if (!std::isfinite(rhs_adj)) return;  // belt and braces: refuse divergence
    c.rhs -= rhs_adj;
}

// Paper §2.4: weaken(C, xs) := drop xs, rhs -= max{as us, as ℓs}.
bool weaken_var(GeqConstraint& reason, Index s,
                const model::LpProblem& lp, f64 tol) {
    for (std::size_t k = 0; k < reason.cols.size(); ++k) {
        if (reason.cols[k] != s) continue;
        const f64 a = reason.vals[k];
        if (std::fabs(a) <= tol) {
            reason.vals[k] = 0.0;
            compact_geq(reason, tol);
            return true;
        }
        const f64 ell = lp.col_lo[sz(s)];
        const f64 u = lp.col_hi[sz(s)];
        const f64 m = std::max(a * u, a * ell);
        // An unbounded side makes the sound weakening value divergent; the
        // old code subtracted +-inf and produced a vacuous (-inf rhs)
        // constraint. Refuse instead - the caller picks another variable or
        // stops, which is always sound.
        if (!std::isfinite(m)) return false;
        reason.rhs -= m;
        reason.vals[k] = 0.0;
        compact_geq(reason, tol);
        return true;
    }
    return false;
}

// Algorithm 2: while resolve(reason, confl, xr) feasible, weaken a relaxable
// var ≠ xr and coefTight.
GeqConstraint coef_tighten_reduce(const GeqConstraint& reason_in,
                                  const GeqConstraint& conflict,
                                  Index var,
                                  const model::LpProblem& lp,
                                  const std::vector<f64>& lo,
                                  const std::vector<f64>& hi,
                                  f64 tol) {
    GeqConstraint reason = reason_in;
    for (int iter = 0; iter < 64; ++iter) {
        GeqConstraint tmp = conflict;
        if (!resolve_geq(tmp, reason, var, tol)) return reason;
        if (geq_infeasible(tmp, lo, hi, tol)) return reason;

        Index xs = -1;
        for (Index j : reason.cols) {
            if (j == var) continue;
            if (!is_integer_col(lp, j) &&
                !is_relaxable(reason, j, lp.col_lo, lp.col_hi, lo, hi, tol))
                continue;
            if (is_relaxable(reason, j, lp.col_lo, lp.col_hi, lo, hi, tol)) {
                xs = j;
                break;
            }
        }
        // Also allow weakening fixed integers that contribute to max activity.
        if (xs < 0) {
            for (std::size_t k = 0; k < reason.cols.size(); ++k) {
                const Index j = reason.cols[k];
                if (j == var) continue;
                if (!is_integer_col(lp, j)) continue;
                if (hi[sz(j)] > lo[sz(j)] + tol) continue;
                xs = j;
                break;
            }
        }
        if (xs < 0) {
            apply_coef_tightening(reason, lp, tol);
            tmp = conflict;
            if (!resolve_geq(tmp, reason, var, tol)) return reason;
            if (geq_infeasible(tmp, lo, hi, tol)) return reason;
            break;
        }
        if (!weaken_var(reason, xs, lp, tol)) {
            // No sound weakening for this variable (unbounded side): stop
            // rather than produce a vacuous or divergent reason.
            break;
        }
        apply_coef_tightening(reason, lp, tol);
    }
    return reason;
}

// Proposition 2 (pure binary cMIR), arXiv:2410.15110 §4.2:
// Normalize to Assumption 1 (coeff(xr)=1, aj≥0), let P={j: uρj=1},
// b̃ = b − Σ_{j∈P} aj (0<b̃<1 under non-tight prop), then
//   CcMIR : xr + Σ_{j∉P} ψ(aj) xj − Σ_{j∈P} ψ(−aj) xj
//         ≥ 1 − Σ_{j∈P} ψ(−aj)
// with ψ(a)=⌊a⌋ + min{1, f(a)/f(b̃)}.
bool cmir_prop2_binary(GeqConstraint reason,
                       Index xr,
                       const model::LpProblem& lp,
                       const std::vector<f64>& /*lo*/,
                       const std::vector<f64>& hi,
                       f64 tol,
                       GeqConstraint& out) {
    if (!reason_all_binary(reason, lp, tol)) return false;
    f64 ar = coeff_at(reason, xr);
    if (std::fabs(ar) <= tol) return false;

    const f64 s = 1.0 / ar;
    for (f64& v : reason.vals) v *= s;
    reason.rhs *= s;

    // Complement negative-coeff binaries into Assumption-1 form.
    for (std::size_t k = 0; k < reason.cols.size(); ++k) {
        if (reason.cols[k] == xr) continue;
        if (reason.vals[k] >= -tol) continue;
        // aj xj → aj(1 − x̄) = −aj x̄ + aj  ⇒ coeff ← −aj, rhs ← rhs − aj
        reason.rhs -= reason.vals[k];
        reason.vals[k] = -reason.vals[k];
    }

    ar = coeff_at(reason, xr);
    if (std::fabs(ar - 1.0) > 1e-6) return false;

    std::vector<char> in_P(static_cast<std::size_t>(lp.n_cols()), 0);
    f64 btilde = reason.rhs;
    for (std::size_t k = 0; k < reason.cols.size(); ++k) {
        const Index j = reason.cols[k];
        if (j == xr) continue;
        const f64 a = reason.vals[k];
        if (a < -tol) return false;
        // Fixed at local upper bound 1 (paper P = {j ∈ J : uρj = 1}).
        if (hi[sz(j)] >= 1.0 - tol) {
            in_P[sz(j)] = 1;
            btilde -= a;
        }
    }
    if (!(btilde > tol && btilde < 1.0 - tol)) return false;
    const f64 fb = btilde;

    auto psi = [&](f64 a) -> f64 {
        const f64 fl = std::floor(a + tol);
        const f64 fa = a - fl;
        return fl + std::min(1.0, (fb > tol ? fa / fb : 0.0));
    };

    out.cols.clear();
    out.vals.clear();
    out.cols.push_back(xr);
    out.vals.push_back(1.0);
    f64 sum_psi_neg = 0.0;
    for (std::size_t k = 0; k < reason.cols.size(); ++k) {
        const Index j = reason.cols[k];
        if (j == xr) continue;
        const f64 a = reason.vals[k];
        if (in_P[sz(j)]) {
            const f64 p = psi(-a);
            out.cols.push_back(j);
            out.vals.push_back(-p);
            sum_psi_neg += p;
        } else {
            const f64 p = psi(a);
            if (std::fabs(p) <= tol) continue;
            out.cols.push_back(j);
            out.vals.push_back(p);
        }
    }
    out.rhs = 1.0 - sum_psi_neg;
    compact_geq(out, tol);
    return !out.cols.empty();
}

// General cMIR on the reason. Validity (2026-09-14 P0 fix): the Marchand-
// Wolsey bound substitution must use GLOBAL bounds - substituting at LOCAL
// (node) bounds yields a cut valid only inside that subtree, and applying it
// to global_lp cut off feasible points in sibling subtrees (markshare1
// claimed Optimal 19 vs MIPLIB opt 1). The vertex x must be finite as well;
// with an unbounded support column no activity-max vertex exists, so the
// reduction is refused rather than fed ±inf into the MIR formula (gen-ip002).
bool apply_general_cmir(const GeqConstraint& reason,
                        const model::LpProblem& lp,
                        f64 tol,
                        GeqConstraint& out) {
    const std::vector<f64>& glo = lp.col_lo;
    const std::vector<f64>& ghi = lp.col_hi;
    std::vector<f64> x(static_cast<std::size_t>(lp.n_cols()), 0.0);
    for (Index j = 0; j < lp.n_cols(); ++j) {
        const f64 a = coeff_at(reason, j);
        const f64 bound = (a >= 0.0) ? ghi[sz(j)] : glo[sz(j)];
        if (std::fabs(a) <= tol || !std::isfinite(bound)) {
            // Free/unbounded side on a support column: no finite vertex.
            if (std::fabs(a) > tol) return false;
            x[sz(j)] = std::isfinite(glo[sz(j)]) ? glo[sz(j)] : 0.0;
            continue;
        }
        x[sz(j)] = bound;
    }
    MirOptions mo;
    mo.enabled = true;
    mo.max_cuts = 1;
    mo.aggregate = false;
    mo.violation_min = 0.0;
    mo.min_fractionality = 1e-6;
    MirDiagnostics md;
    std::vector<Index> ocols;
    std::vector<f64> ovals;
    f64 orhs = 0.0;
    if (!apply_cmir_geq(lp, reason.cols, reason.vals, reason.rhs, x, glo, ghi,
                        mo, /*require_violation=*/false, ocols, ovals, orhs,
                        md))
        return false;
    // Non-finite cMIR output is refused outright (fail-closed).
    if (!std::isfinite(orhs) || std::fabs(orhs) > 1e12) return false;
    for (f64 v : ovals)
        if (!std::isfinite(v) || std::fabs(v) > 1e9) return false;
    out.cols = std::move(ocols);
    out.vals = std::move(ovals);
    out.rhs = orhs;
    return !out.cols.empty();
}

// Algorithm 3: resolve non-relaxable continuous variables from Creason using
// earlier trail reasons (full history walk), then binary/general reduce.
struct ReduceResult {
    GeqConstraint reason;
    bool became_conflict = false;  // Remark 3 early infeas
};

ReduceResult reduce_mixed_binary(GeqConstraint reason,
                                 const GeqConstraint& conflict,
                                 Index xr,
                                 std::size_t reason_trail_pos,  // chrono index
                                 const model::LpProblem& lp,
                                 const std::vector<f64>& lo,
                                 const std::vector<f64>& hi,
                                 const PropTrail& trail,
                                 const ConflictCutOptions& opts,
                                 ConflictCutDiagnostics& diag) {
    ReduceResult res;
    res.reason = std::move(reason);
    const f64 tol = opts.tol;

    // Reconstruct local bounds just before the reason propagation.
    std::vector<f64> blo = lp.col_lo;
    std::vector<f64> bhi = lp.col_hi;
    const auto& ents = trail.entries();
    for (std::size_t t = 0; t < reason_trail_pos && t < ents.size(); ++t) {
        const auto& e = ents[t];
        if (e.dir == BoundDir::Lower) blo[sz(e.var)] = e.new_bound;
        else bhi[sz(e.var)] = e.new_bound;
    }

    for (int guard = 0; guard < opts.max_resolve_steps; ++guard) {
        Index cont = -1;
        std::size_t cont_pos = 0;
        // nr(Creason, ρ): non-relaxable continuous under blo/bhi.
        for (Index j : res.reason.cols) {
            if (is_integer_col(lp, j)) continue;
            if (is_relaxable(res.reason, j, lp.col_lo, lp.col_hi, blo, bhi,
                             tol))
                continue;
            // Latest trail entry for j with (p,q) ≺ reason state.
            for (std::size_t t = reason_trail_pos; t > 0; --t) {
                const auto& e = ents[t - 1];
                if (e.var != j) continue;
                if (e.kind == ReasonKind::Unknown) continue;
                cont = j;
                cont_pos = t - 1;
                break;
            }
            if (cont >= 0) break;
        }
        if (cont < 0) break;

        GeqConstraint ccont;
        if (!entry_as_reason(lp, ents[cont_pos], ccont)) {
            res.became_conflict = false;
            return res;  // caller may abort
        }
        if (!resolve_geq(res.reason, ccont, cont, tol)) break;
        ++diag.continuous_resolved;
        compact_geq(res.reason, tol);

        // Remark 3: aggregated reason may already be infeasible in predecessor.
        std::vector<f64> pred_lo = blo;
        std::vector<f64> pred_hi = bhi;
        // Predecessor of reason state ≈ bounds before reason entry.
        if (geq_infeasible(res.reason, pred_lo, pred_hi, tol)) {
            res.became_conflict = true;
            return res;
        }
        // Also update working bounds to the continuous's predecessor for nr().
        blo = lp.col_lo;
        bhi = lp.col_hi;
        for (std::size_t t = 0; t < cont_pos && t < ents.size(); ++t) {
            const auto& e = ents[t];
            if (e.dir == BoundDir::Lower) blo[sz(e.var)] = e.new_bound;
            else bhi[sz(e.var)] = e.new_bound;
        }
        reason_trail_pos = cont_pos;
    }

    // Prefer Prop. 2 cMIR on pure binary; else general cMIR; else Alg 2.
    const bool use_cmir = opts.use_cmirror || opts.use_cmirror_binary;
    if (use_cmir && reason_all_binary(res.reason, lp, tol)) {
        GeqConstraint mir;
        if (cmir_prop2_binary(res.reason, xr, lp, lo, hi, tol, mir)) {
            res.reason = std::move(mir);
            ++diag.cmir_applied;
            return res;
        }
        if (apply_general_cmir(res.reason, lp, tol, mir)) {
            res.reason = std::move(mir);
            ++diag.cmir_applied;
            return res;
        }
    } else if (use_cmir) {
        GeqConstraint mir;
        if (apply_general_cmir(res.reason, lp, tol, mir)) {
            // Keep if resolvent with mir stays infeasible or improves.
            GeqConstraint trial = conflict;
            if (resolve_geq(trial, mir, xr, tol) &&
                geq_infeasible(trial, lo, hi, tol)) {
                res.reason = std::move(mir);
                ++diag.cmir_applied;
                return res;
            }
            // §7: do NOT keep a cMIR that failed the local-resolvent check -
            // an over-strengthened reason can still leave Clearn locally
            // infeasible while cutting off globally feasible points.
        }
    }

    res.reason = coef_tighten_reduce(res.reason, conflict, xr, lp, lo, hi, tol);
    return res;
}

CutRow geq_to_cut(const GeqConstraint& c) {
    CutRow row;
    row.cols = c.cols;
    row.vals = c.vals;
    row.row_lo = c.rhs;
    row.row_hi = kInf;
    row.name = "conflict_cut";
    return row;
}

int current_decision_depth(const PropTrail& trail) {
    int d = 0;
    for (const auto& e : trail.entries()) d = std::max(d, e.depth);
    return d;
}

int deepest_bound_depth(const PropTrail& trail, Index j) {
    int d = -1;
    for (const auto& e : trail.entries())
        if (e.var == j) d = std::max(d, e.depth);
    return d;
}

// FUIP / asserting (paper Alg 1): at most one non-relaxable variable whose
// deepest bound change sits at the current decision level. Also treat a
// single free variable under local bounds as asserting (unit propagation).
bool is_asserting(const GeqConstraint& c,
                  const model::LpProblem& lp,
                  const std::vector<f64>& lo,
                  const std::vector<f64>& hi,
                  const PropTrail& trail,
                  f64 tol) {
    const int cur = current_decision_depth(trail);
    int nonrel_cur = 0;
    int unbound = 0;
    for (std::size_t k = 0; k < c.cols.size(); ++k) {
        const Index j = c.cols[k];
        if (std::fabs(c.vals[k]) <= tol) continue;
        if (hi[sz(j)] > lo[sz(j)] + tol) ++unbound;
        if (is_relaxable(c, j, lp.col_lo, lp.col_hi, lo, hi, tol)) continue;
        const int dj = deepest_bound_depth(trail, j);
        if (dj >= cur) ++nonrel_cur;
    }
    if (nonrel_cur <= 1) return true;
    return unbound <= 1;
}

// Algorithm 1 line 3: earliest trail index after which Clearn is infeasible.
// Returns chronological index into trail.entries(), or npos if never.
std::size_t earliest_infeasible_index(const GeqConstraint& learn,
                                      const model::LpProblem& lp,
                                      const PropTrail& trail,
                                      f64 tol) {
    std::vector<f64> lo = lp.col_lo;
    std::vector<f64> hi = lp.col_hi;
    if (geq_infeasible(learn, lo, hi, tol)) return 0;  // already at root
    const auto& ents = trail.entries();
    for (std::size_t t = 0; t < ents.size(); ++t) {
        const auto& e = ents[t];
        if (e.dir == BoundDir::Lower) lo[sz(e.var)] = e.new_bound;
        else hi[sz(e.var)] = e.new_bound;
        if (geq_infeasible(learn, lo, hi, tol)) return t;
    }
    return static_cast<std::size_t>(-1);
}

}  // namespace

bool build_conflict_constraint(const ConflictAnalysisContext& ctx,
                               GeqConstraint& out) {
    if (!ctx.lp || !ctx.col_lo || !ctx.col_hi) return false;
    if (ctx.conflict_row >= 0) {
        // Prefer the side that is infeasible under local bounds.
        GeqConstraint up, lo;
        const bool has_up = extract_row_geq_upper(*ctx.lp, ctx.conflict_row, up);
        const bool has_lo = extract_row_geq_lower(*ctx.lp, ctx.conflict_row, lo);
        if (has_up && geq_infeasible(up, *ctx.col_lo, *ctx.col_hi, 1e-9)) {
            out = std::move(up);
            return true;
        }
        if (has_lo && geq_infeasible(lo, *ctx.col_lo, *ctx.col_hi, 1e-9)) {
            out = std::move(lo);
            return true;
        }
        if (has_up) {
            out = std::move(up);
            return true;
        }
        if (has_lo) {
            out = std::move(lo);
            return true;
        }
        return false;
    }
    if (ctx.conflict_var >= 0) {
        out.cols = {ctx.conflict_var};
        out.vals = {0.0};
        out.rhs = 1.0;  // 0 >= 1 under empty domain
        return true;
    }
    return false;
}

std::optional<CutRow> analyze_conflict_cuts(const ConflictAnalysisContext& ctx,
                                            const ConflictCutOptions& opts,
                                            ConflictCutDiagnostics& diag) {
    ++diag.attempts;
    if (!opts.enabled || !ctx.lp || !ctx.col_lo || !ctx.col_hi || !ctx.trail)
        return std::nullopt;

    // Refuse analysis that depends on node-local cut rows. Tree GMI (and
    // similar) derived under local bounds are subtree-valid only; seeding a
    // global conflict cut from them yields invalid duals (flugpl P0).
    const Index n_glob =
        (ctx.n_global_rows >= 0) ? ctx.n_global_rows : ctx.lp->n_rows();
    auto row_is_local = [&](Index r) {
        return r >= 0 && r >= n_glob;
    };
    if (row_is_local(ctx.conflict_row)) {
        ++diag.aborted;
        ++diag.aborted_local_scope;
        return std::nullopt;
    }
    for (const auto& e : ctx.trail->entries()) {
        if ((e.kind == ReasonKind::Row || e.kind == ReasonKind::Cut ||
             e.kind == ReasonKind::Conflict) &&
            row_is_local(e.reason_id)) {
            ++diag.aborted;
            ++diag.aborted_local_scope;
            return std::nullopt;
        }
    }

    const bool paper = (opts.mode == ConflictCutMode::Paper);
    GeqConstraint learn;
    if (!build_conflict_constraint(ctx, learn)) {
        ++diag.aborted;
        ++diag.aborted_seed;
        return std::nullopt;
    }
    if (!geq_infeasible(learn, *ctx.col_lo, *ctx.col_hi, opts.tol)) {
        ++diag.aborted;
        ++diag.aborted_seed;
        return std::nullopt;
    }

    const PropTrail& trail = *ctx.trail;
    const auto& ents = trail.entries();

    for (int step = 0; step < opts.max_resolve_steps; ++step) {
        // Global ⊥
        if (geq_infeasible(learn, ctx.lp->col_lo, ctx.lp->col_hi, opts.tol)) {
            ++diag.global_infeas_proofs;
            break;
        }
        if (is_asserting(learn, *ctx.lp, *ctx.col_lo, *ctx.col_hi, trail,
                         opts.tol)) {
            ++diag.fuip_stops;
            break;
        }

        const std::size_t idx =
            earliest_infeasible_index(learn, *ctx.lp, trail, opts.tol);
        if (idx == static_cast<std::size_t>(-1) || idx >= ents.size()) {
            ++diag.aborted;
            ++diag.aborted_trail;
            std::vector<f64> replay_lo = ctx.lp->col_lo;
            std::vector<f64> replay_hi = ctx.lp->col_hi;
            for (const auto& entry : ents) {
                if (entry.dir == BoundDir::Lower)
                    replay_lo[sz(entry.var)] = entry.new_bound;
                else
                    replay_hi[sz(entry.var)] = entry.new_bound;
            }
            for (Index j : learn.cols) {
                if ((*ctx.col_lo)[sz(j)] > replay_lo[sz(j)] + opts.tol ||
                    (*ctx.col_hi)[sz(j)] < replay_hi[sz(j)] - opts.tol) {
                    ++diag.aborted_trail_missing_bound;
                    break;
                }
            }
            return std::nullopt;
        }
        const PropTrailEntry& e = ents[idx];

        // Branch decision at earliest infeasible state ⇒ should be asserting;
        // if not, stop safely rather than spinning.
        if (e.kind == ReasonKind::Branch) {
            ++diag.fuip_stops;
            break;
        }

        GeqConstraint reason;
        if (!entry_as_reason(*ctx.lp, e, reason)) {
            ++diag.aborted;
            ++diag.aborted_reason;
            return std::nullopt;
        }

        bool has_nonbinary = false;
        bool has_continuous = false;
        for (Index j : reason.cols) {
            if (!is_integer_col(*ctx.lp, j)) {
                has_continuous = true;
            } else if (!is_binary_col(*ctx.lp, j, opts.tol)) {
                has_nonbinary = true;
            }
        }
        if (has_nonbinary) ++diag.general_int_reasons;

        if (!paper) {
            // SafeLimited: abort on continuous; skip cMIR on non-binary.
            if (has_continuous) {
                ++diag.aborted;
                ++diag.aborted_reason;
                return std::nullopt;
            }
            if (has_nonbinary && !opts.allow_general_integer) {
                ++diag.aborted;
                ++diag.aborted_reason;
                return std::nullopt;
            }
            const bool use_cmir =
                opts.use_cmirror || opts.use_cmirror_binary;
            if (use_cmir && e.kind == ReasonKind::Row &&
                reason_all_binary(reason, *ctx.lp, opts.tol)) {
                GeqConstraint mir;
                if (cmir_prop2_binary(reason, e.var, *ctx.lp, *ctx.col_lo,
                                      *ctx.col_hi, opts.tol, mir) ||
                    apply_general_cmir(reason, *ctx.lp, opts.tol, mir)) {
                    reason = std::move(mir);
                    ++diag.cmir_applied;
                }
            } else if (use_cmir && has_nonbinary) {
                ++diag.cmir_skipped_nonbinary;
            }
            reason = coef_tighten_reduce(reason, learn, e.var, *ctx.lp,
                                         *ctx.col_lo, *ctx.col_hi, opts.tol);
        } else {
            // Paper path: Alg 3 continuous resolve + cMIR / coef reduce.
            if (has_nonbinary && !opts.allow_general_integer) {
                ++diag.aborted;
                ++diag.aborted_reason;
                return std::nullopt;
            }
            auto red = reduce_mixed_binary(std::move(reason), learn, e.var, idx,
                                           *ctx.lp, *ctx.col_lo, *ctx.col_hi,
                                           trail, opts, diag);
            if (red.became_conflict) {
                learn = std::move(red.reason);
                apply_coef_tightening(learn, *ctx.lp, opts.tol);
                if (!geq_infeasible(learn, *ctx.col_lo, *ctx.col_hi, opts.tol)) {
                    ++diag.aborted;
                    ++diag.aborted_resolution;
                    return std::nullopt;
                }
                continue;
            }
            reason = std::move(red.reason);
        }

        if (!resolve_geq(learn, reason, e.var, opts.tol)) {
            ++diag.aborted;
            ++diag.aborted_resolution;
            return std::nullopt;
        }
        apply_coef_tightening(learn, *ctx.lp, opts.tol);  // Alg 1 strengthen
        compact_geq(learn, opts.tol);

        // Invariant: resolvent must stay locally infeasible (§7 abort-safe).
        if (!geq_infeasible(learn, *ctx.col_lo, *ctx.col_hi, opts.tol)) {
            ++diag.aborted;
            ++diag.aborted_resolution;
            return std::nullopt;
        }
    }

    if (!geq_infeasible(learn, *ctx.col_lo, *ctx.col_hi, opts.tol)) {
        ++diag.aborted;
        ++diag.aborted_final;
        return std::nullopt;
    }

    GeqConstraint clean;
    clean.rhs = learn.rhs;
    for (std::size_t k = 0; k < learn.cols.size(); ++k) {
        if (std::fabs(learn.vals[k]) <= opts.tol) continue;
        clean.cols.push_back(learn.cols[k]);
        clean.vals.push_back(learn.vals[k]);
    }
    if (clean.cols.empty()) {
        if (clean.rhs > opts.tol) {
            ++diag.learned;
            return geq_to_cut(clean);
        }
        ++diag.aborted;
        ++diag.aborted_final;
        return std::nullopt;
    }

    // Final validity gate: cut must be satisfied by all points in the global
    // box that meet integrality on a tiny exhaustive check when possible;
    // otherwise trust derivation but refuse empty nonsense.
    ++diag.learned;
    return geq_to_cut(clean);
}

CutValidity conflict_cut_check_binary(const model::LpProblem& lp,
                                      const CutRow& cut,
                                      f64 tol) {
    const Index n = lp.n_cols();
    // Complete enumeration is possible only when the whole box is decided:
    // every integer column binary, every continuous column fixed. Anything
    // else leaves room for a violating point outside the sweep, so the
    // result there is at most a refutation, never a verification.
    std::vector<Index> bins;
    bins.reserve(sz(n));
    bool complete = true;
    for (Index j = 0; j < n; ++j) {
        const bool is_int =
            !lp.is_integer.empty() && lp.is_integer[sz(j)];
        if (is_int) {
            if (lp.col_lo[sz(j)] >= -tol && lp.col_hi[sz(j)] <= 1.0 + tol) {
                bins.push_back(j);
            } else {
                complete = false;  // general integer freedom
            }
        } else if (lp.col_lo[sz(j)] != lp.col_hi[sz(j)]) {
            complete = false;  // free continuous column
        }
    }

    auto cut_violated_at = [&](const std::vector<f64>& x) -> bool {
        f64 lhs = 0.0;
        for (std::size_t t = 0; t < cut.cols.size(); ++t) {
            const Index j = cut.cols[t];
            if (j < 0 || j >= n) continue;
            lhs += cut.vals[t] * x[sz(j)];
        }
        if (std::isfinite(cut.row_lo) && lhs + tol < cut.row_lo) return true;
        if (std::isfinite(cut.row_hi) && lhs > cut.row_hi + tol) return true;
        return false;
    };

    // Large pure-binary box: 2^n exceeds the full-enum budget, but FUIP cuts
    // typically have tiny support. Sound Verified path: every support
    // assignment that violates the cut must be domain-infeasible under
    // bound propagation (prop-infeasible ⇒ truly infeasible). If prop
    // cannot rule a violator out, stay Unverified (fail-closed) unless a
    // concrete LP-feasible witness refutes.
    auto verify_large_pure_binary_by_support = [&]() -> CutValidity {
        std::vector<Index> support;
        support.reserve(cut.cols.size());
        for (Index j : cut.cols) {
            if (j < 0 || j >= n) return CutValidity::Unverified;
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)])
                return CutValidity::Unverified;
            if (lp.col_lo[sz(j)] < -tol || lp.col_hi[sz(j)] > 1.0 + tol)
                return CutValidity::Unverified;
            support.push_back(j);
        }
        std::sort(support.begin(), support.end());
        support.erase(std::unique(support.begin(), support.end()),
                      support.end());
        // Cap by WORK, not by support size. Each mask costs a full
        // max_row_violation() + max_bound_violation() pass, i.e. O(nnz), so a
        // flat 2^16 bound is 1.2e9 operations on an 18k-nnz model -- measured
        // at ~1 s per learned nogood on app1-1, invisible to every timer.
        // Same reasoning as the general checker: this is defense-in-depth on
        // a cut that is sound by construction and it is fail-closed only on
        // REFUTATION, so a smaller sweep costs refutation power, never
        // soundness.
        {
            const std::size_t nnz_cost =
                std::max<std::size_t>(static_cast<std::size_t>(lp.nnz()), 1);
            const std::size_t kEnumWorkCap = 2000000;
            std::size_t max_support = 0;
            while (max_support < 16 &&
                   (static_cast<std::size_t>(1) << (max_support + 1)) <=
                       kEnumWorkCap / nnz_cost)
                ++max_support;
            if (support.size() > max_support) return CutValidity::Unverified;
        }

        if (support.empty()) {
            // 0 >= rhs (or 0 <= rhs): check whether any domain-feasible
            // point exists; if none, the contradiction cut is vacuously
            // valid for the integer box.
            std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
            const auto pr = propagate_bounds(lp, lo, hi, tol, 32);
            if (!pr.feasible) return CutValidity::Verified;
            std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
            for (Index j = 0; j < n; ++j) {
                if (lo[sz(j)] == hi[sz(j)]) x[sz(j)] = lo[sz(j)];
                else if (lo[sz(j)] > 0.0) x[sz(j)] = lo[sz(j)];
                else if (hi[sz(j)] < 0.0) x[sz(j)] = hi[sz(j)];
            }
            if (lp.max_row_violation(x) <= tol &&
                lp.max_bound_violation(x) <= tol && cut_violated_at(x))
                return CutValidity::Refuted;
            return CutValidity::Unverified;
        }

        const std::size_t ns = support.size();
        const std::size_t total = static_cast<std::size_t>(1) << ns;
        for (std::size_t mask = 0; mask < total; ++mask) {
            std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
            for (Index j = 0; j < n; ++j) {
                if (lp.col_lo[sz(j)] == lp.col_hi[sz(j)])
                    x[sz(j)] = lp.col_lo[sz(j)];
            }
            for (std::size_t b = 0; b < ns; ++b)
                x[sz(support[b])] = static_cast<f64>((mask >> b) & 1);
            if (!cut_violated_at(x)) continue;

            std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
            for (std::size_t b = 0; b < ns; ++b) {
                const f64 v = static_cast<f64>((mask >> b) & 1);
                lo[sz(support[b])] = v;
                hi[sz(support[b])] = v;
            }
            const auto pr = propagate_bounds(lp, lo, hi, tol, 32);
            if (!pr.feasible) continue;  // violator ruled out

            // Prop did not prove empty: try a concrete witness on the
            // post-prop box (fixed support + lo on free cols).
            for (Index j = 0; j < n; ++j) {
                if (lo[sz(j)] == hi[sz(j)]) x[sz(j)] = lo[sz(j)];
                else x[sz(j)] = lo[sz(j)];
            }
            if (lp.max_row_violation(x) <= tol &&
                lp.max_bound_violation(x) <= tol && cut_violated_at(x))
                return CutValidity::Refuted;
            return CutValidity::Unverified;
        }
        return CutValidity::Verified;
    };

    // Refutation sweep over a candidate set: points built from enumerated
    // columns with everything else at its fixed value (0 when free) are
    // genuine LP points, so a violating one is a definitive witness. Used
    // for the full binary set when complete, else for the cut support.
    // One work cap for every exhaustive sweep in this file. Each mask costs a
    // full max_row_violation() + max_bound_violation() pass, i.e. O(nnz), so
    // bounding the EXPONENT alone bounds nothing: a flat 2^16 is 1.2e9
    // operations on an 18k-nnz model. Measured on app1-1 -- 42 nodes, and
    // 19,126 ms of a 30 s budget inside this one function, attributed to no
    // timer at all.
    const std::size_t nnz_cost =
        std::max<std::size_t>(static_cast<std::size_t>(lp.nnz()), 1);
    const std::size_t kEnumWorkCap = 2000000;
    std::size_t max_sweep_bits = 0;
    while (max_sweep_bits < 16 &&
           (static_cast<std::size_t>(1) << (max_sweep_bits + 1)) <=
               kEnumWorkCap / nnz_cost)
        ++max_sweep_bits;

    std::vector<Index> sweep;
    if (complete) {
        if (bins.size() > max_sweep_bits)
            return verify_large_pure_binary_by_support();
        sweep = bins;
    } else {
        // Mixed MIP / general integers: full-box Verified is impossible, but
        // FUIP cuts almost always have tiny pure-binary support. The
        // support+prop path is sound there (every cut-violating support
        // assignment must be domain-infeasible). Without this, misc03-class
        // models abort every Mexi cut at Unverified despite valid derivation.
        {
            const CutValidity via = verify_large_pure_binary_by_support();
            if (via != CutValidity::Unverified) return via;
        }
        // Support-only sweep: sound for refutation only.
        if (!cut.cols.empty() && cut.cols.size() <= 20) {
            for (Index j : cut.cols) {
                if (j < 0 || j >= n) continue;
                if (!lp.is_integer.empty() && lp.is_integer[sz(j)] &&
                    lp.col_lo[sz(j)] >= -tol && lp.col_hi[sz(j)] <= 1.0 + tol)
                    sweep.push_back(j);
            }
        }
        if (sweep.empty()) return CutValidity::Unverified;
    }

    // The support-only sweep allowed 2^20 masks. At O(nnz) per mask that is
    // 1.9e10 operations on an 18k-nnz model -- the dominant cost of the whole
    // solve on app1-1 (18,975 ms of 30 s) and invisible to every timer. Same
    // work cap as the other sweeps in this file.
    if (sweep.size() > max_sweep_bits) return CutValidity::Unverified;

    const std::size_t nb = sweep.size();
    const std::size_t total = static_cast<std::size_t>(1) << nb;
    std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        if (lp.col_lo[sz(j)] == lp.col_hi[sz(j)])
            x[sz(j)] = lp.col_lo[sz(j)];
    }
    for (std::size_t mask = 0; mask < total; ++mask) {
        for (std::size_t b = 0; b < nb; ++b)
            x[sz(sweep[b])] = static_cast<f64>((mask >> b) & 1);
        if (lp.max_row_violation(x) > tol || lp.max_bound_violation(x) > tol)
            continue;
        if (cut_violated_at(x)) return CutValidity::Refuted;
    }
    return complete ? CutValidity::Verified : CutValidity::Unverified;
}

CutValidity conflict_cut_check_general(const model::LpProblem& lp,
                                       const CutRow& cut,
                                       f64 tol,
                                       std::size_t max_points,
                                       bool* enumerated) {
    if (enumerated) *enumerated = false;
    // WORK cap, not just a POINT cap.
    //
    // Each enumerated point costs a full pass over the model:
    // max_row_violation() + max_bound_violation() are O(nnz). Capping the
    // number of POINTS alone therefore bounds nothing -- on app1-1
    // (4926 x 2480, 18275 nnz) the default 2^16 points meant 1.2e9
    // operations, and this check cost 1.4 SECONDS per learned nogood.
    // Traced end to end: node LP infeasible -> prune -> learn nogood ->
    // validate here -> 21 nogoods x 1.4 s = 29 s of a 30 s budget, none of it
    // visible in any timer, while the search branched 44 nodes.
    //
    // This is a defense-in-depth check on a cut that is sound by construction
    // and it is fail-closed only on REFUTATION, so running it on fewer points
    // costs refutation power, never soundness. Bounding points*nnz keeps it
    // free on the small models where exhaustive enumeration is genuinely
    // decisive, and stops it dominating a solve on anything larger.
    const std::size_t nnz_cost =
        std::max<std::size_t>(static_cast<std::size_t>(lp.nnz()), 1);
    const std::size_t kEnumWorkCap = 2000000;   // ~2e7 was still 100+ ms/cut
    const std::size_t eff_max_points = std::max<std::size_t>(
        64, std::min(max_points, kEnumWorkCap / nnz_cost));

    const Index n = lp.n_cols();
    std::vector<Index> ints;
    std::vector<int> lo_i, hi_i;
    std::size_t product = 1;
    std::size_t n_integer_cols = 0;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) {
            // A free continuous column puts points outside any enumeration:
            // unverifiable (fail-closed), NOT silently valid.
            if (lp.col_lo[sz(j)] != lp.col_hi[sz(j)])
                return CutValidity::Unverified;
            continue;
        }
        ++n_integer_cols;
        const f64 lo = lp.col_lo[sz(j)];
        const f64 hi = lp.col_hi[sz(j)];
        // Unbounded / non-finite domains cannot be cast through int safely
        // (floor(+inf) → UB) and must not be skipped: older code did
        // `int uj = (int)floor(hi)` then `if (uj < lj) continue`, which
        // dropped every free integer column and returned Verified on an
        // empty sweep - false Optimal on gen-ip002 (2026-09-14).
        if (!std::isfinite(lo) || !std::isfinite(hi))
            return CutValidity::Unverified;
        if (hi - lo > static_cast<f64>(eff_max_points) + 1.0)
            return CutValidity::Unverified;
        const f64 lj_f = std::ceil(lo - tol);
        const f64 uj_f = std::floor(hi + tol);
        if (!std::isfinite(lj_f) || !std::isfinite(uj_f))
            return CutValidity::Unverified;
        if (uj_f < lj_f) continue;
        // Stay inside int range before casting.
        if (lj_f < static_cast<f64>(std::numeric_limits<int>::min()) ||
            uj_f > static_cast<f64>(std::numeric_limits<int>::max()))
            return CutValidity::Unverified;
        const int lj = static_cast<int>(lj_f);
        const int uj = static_cast<int>(uj_f);
        const std::size_t span = static_cast<std::size_t>(uj - lj + 1);
        if (span == 0 || product > eff_max_points / std::max<std::size_t>(span, 1)) {
            if (enumerated) *enumerated = false;
            return CutValidity::Unverified;
        }
        product *= span;
        ints.push_back(j);
        lo_i.push_back(lj);
        hi_i.push_back(uj);
    }
    // Every integer column must appear in the sweep. Skipped empty domains
    // (uj < lj) with no other ints would otherwise vacuously "verify".
    if (n_integer_cols > 0 && ints.size() != n_integer_cols)
        return CutValidity::Unverified;
    if (ints.empty()) {
        // No integer columns: nothing to certify for a MIP cut.
        return CutValidity::Unverified;
    }
    if (enumerated) *enumerated = true;

    std::vector<int> cur(ints.size());
    for (std::size_t i = 0; i < ints.size(); ++i) cur[i] = lo_i[i];
    std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        if (lp.col_lo[sz(j)] == lp.col_hi[sz(j)])
            x[sz(j)] = lp.col_lo[sz(j)];
    }

    auto advance = [&]() -> bool {
        for (std::size_t i = 0; i < cur.size(); ++i) {
            if (cur[i] < hi_i[i]) {
                ++cur[i];
                return true;
            }
            cur[i] = lo_i[i];
        }
        return false;
    };

    for (;;) {
        for (std::size_t i = 0; i < ints.size(); ++i)
            x[sz(ints[i])] = static_cast<f64>(cur[i]);
        if (lp.max_row_violation(x) <= tol &&
            lp.max_bound_violation(x) <= tol) {
            f64 lhs = 0.0;
            for (std::size_t t = 0; t < cut.cols.size(); ++t)
                lhs += cut.vals[t] * x[sz(cut.cols[t])];
            if (std::isfinite(cut.row_lo) && lhs + tol < cut.row_lo)
                return CutValidity::Refuted;
            if (std::isfinite(cut.row_hi) && lhs > cut.row_hi + tol)
                return CutValidity::Refuted;
        }
        if (!advance()) break;
    }
    return CutValidity::Verified;
}

bool conflict_cut_near_empty(const CutRow& cut, f64 tol) {
    if (cut.cols.size() != cut.vals.size()) return true;
    for (f64 v : cut.vals)
        if (std::fabs(v) > tol) return false;
    // Empty / all-zero support with a contradictory rhs is a global ⊥
    // proof (0 >= positive), not "no cut" - allow apply to see it.
    if (std::isfinite(cut.row_lo) && cut.row_lo > tol) return false;
    if (std::isfinite(cut.row_hi) && cut.row_hi < -tol) return false;
    return true;
}

std::optional<CutRow> build_nogood_from_branch_trail(const PropTrail& trail,
                                                    const model::LpProblem& lp,
                                                    f64 tol) {
    const Index n = lp.n_cols();
    // Last Branch assignment wins per variable (deeper decisions override).
    std::vector<int> assign(static_cast<std::size_t>(n), -1);  // -1 unset, 0/1
    for (const auto& e : trail.entries()) {
        if (e.kind != ReasonKind::Branch) continue;
        if (e.var < 0 || e.var >= n) continue;
        // SOUNDNESS (2026-09-14): the nogood may only exclude the branch
        // ASSIGNMENT if that assignment fully represents the node's box.
        // A branch on a non-binary column (general integer x >= 2, x <= 5,
        // or any continuous branching) constrains the box in ways a 0/1
        // assignment row cannot express; skipping such entries (the old
        // behavior) excluded points the node never ruled out -> invalid
        // global cut on mixed models. Refuse instead.
        if (lp.is_integer.empty() || !lp.is_integer[sz(e.var)]) return
            std::nullopt;
        if (lp.col_lo[sz(e.var)] < -tol || lp.col_hi[sz(e.var)] > 1.0 + tol)
            return std::nullopt;
        // Branch to 1: raise lower bound to 1. Branch to 0: drop upper to 0.
        if (e.dir == BoundDir::Lower && e.new_bound >= 1.0 - tol)
            assign[sz(e.var)] = 1;
        else if (e.dir == BoundDir::Upper && e.new_bound <= tol)
            assign[sz(e.var)] = 0;
        else
            return std::nullopt;  // partial bound (should not happen on bins)
    }
    CutRow row;
    row.name = "nogood";
    int n_true = 0;
    for (Index j = 0; j < n; ++j) {
        const int a = assign[sz(j)];
        if (a < 0) continue;
        row.cols.push_back(j);
        if (a == 0) {
            row.vals.push_back(1.0);  // x_j
        } else {
            row.vals.push_back(-1.0);  // -x_j from (1 - x_j)
            ++n_true;
        }
    }
    if (row.cols.empty()) return std::nullopt;
    // sum False x + sum True (1-x) >= 1  ⇒  ... >= 1 - n_true
    row.row_lo = 1.0 - static_cast<f64>(n_true);
    row.row_hi = model::kInf;
    return row;
}

}  // namespace sor::search
