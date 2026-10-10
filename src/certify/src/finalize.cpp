#include "sor/certify/finalize.hpp"
#include "sor/core/env_switches.hpp"
#include "sor/core/work_clock.hpp"
#include "sor/model/dyadic.hpp"
#include "sor/model/exact.hpp"

#include <algorithm>
#include <cstdint>
#include <bit>
#include <optional>
#include <cstring>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "sor/core/route_debug.hpp"

namespace sor::certify {
namespace {

inline std::size_t sz(core::Index i) { SOR_FN(); return static_cast<std::size_t>(i); }

f64 scaled_tol(f64 tol, f64 magnitude) {
    SOR_FN();
    return tol * (1.0 + std::fabs(magnitude));
}

// Column-major view of the CSR entries: for column j, entries
// [ptr[j], ptr[j+1]) list (row, CSR entry index). Exact column sums then need
// one live accumulator instead of one per column.
struct ColumnEntries {
    std::vector<core::Offset> ptr;
    std::vector<core::Index> row;
    std::vector<core::Offset> entry;
};
ColumnEntries column_entries(const model::LpProblem& problem) {
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    const core::Index m = problem.n_rows(), n = problem.n_cols();
    ColumnEntries out;
    out.ptr.assign(sz(n) + 1, 0);
    for (core::Offset k = 0; k < rp[sz(m)]; ++k) ++out.ptr[sz(ci[sz(k)]) + 1];
    for (core::Index j = 0; j < n; ++j) out.ptr[sz(j) + 1] += out.ptr[sz(j)];
    out.row.resize(static_cast<std::size_t>(rp[sz(m)]));
    out.entry.resize(out.row.size());
    std::vector<core::Offset> fill(out.ptr.begin(), out.ptr.end() - 1);
    for (core::Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const auto at = static_cast<std::size_t>(fill[sz(ci[sz(k)])]++);
            out.row[at] = i;
            out.entry[at] = k;
        }
    return out;
}

bool residuals_within_tolerance(const ProofEvidence& ev) {
    SOR_FN();
    return std::isfinite(ev.max_primal_violation) &&
           std::isfinite(ev.max_dual_violation) &&
           ev.max_primal_violation <= ev.primal_feas_tol &&
           ev.max_dual_violation <= ev.dual_feas_tol;
}

bool lp_optimality_within_tolerance(const ProofEvidence& ev) {
    SOR_FN();
    return residuals_within_tolerance(ev) &&
           std::isfinite(ev.gap_rel) && ev.gap_rel <= ev.gap_tol;
}

// The highest level the evidence actually supports.
ProofLevel supported_level(const ProofEvidence& ev) {
    SOR_FN();
    const bool feasible = ev.checker_passed &&
        std::isfinite(ev.max_primal_violation) &&
        ev.max_primal_violation <= ev.primal_feas_tol;
    if (!feasible) return std::isfinite(ev.checked_dual_bound)
        ? ProofLevel::BoundOnly : ProofLevel::None;
    // A basis whose independently checked residuals and gap close is
    // ProvedOptimalFP even when the producer claimed only ProvedKKT (e.g.
    // CLI exact_proof off: engine gap used the unsnapped dual bound).
    if (ev.has_basis && lp_optimality_within_tolerance(ev) &&
        (ev.claimed_level >= ProofLevel::ProvedOptimalFP ||
         ev.claimed_level == ProofLevel::ProvedKKT)) {
        if (ev.vipr_verified) return ProofLevel::ProvedOptimalCertified;
        if (ev.rational_verified) return ProofLevel::ProvedOptimalExact;
        return ProofLevel::ProvedOptimalFP;
    }
    if ((ev.claimed_level == ProofLevel::ProvedKKT ||
         ev.claimed_level == ProofLevel::ProvedGlobalFP ||
         ev.claimed_level == ProofLevel::ProvedGlobalEpsilon) &&
        residuals_within_tolerance(ev)) return ev.claimed_level;
    return std::isfinite(ev.gap_rel) ? ProofLevel::FeasibleWithGap
                                    : ProofLevel::FeasibleOnly;
}


// Column bounds every feasible point of `problem` (with column bounds
// [lo, hi]) satisfies, by bound propagation over the rows to a fixpoint (at
// most 20 rounds). Each round derives, row by row, bounds implied by the
// bounds so far, so the limit is valid for every feasible point. Bounds are
// computed exactly from binary64 inputs, then converted outward. Approximate
// screening may decline a tightening; it never supplies a published bound.
void row_implied_bounds_uncached(const model::LpProblem& problem,
                        const std::vector<f64>& lo_in,
                        const std::vector<f64>& hi_in,
                        std::vector<f64>& imp_lo, std::vector<f64>& imp_hi) {
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    const core::Index m = problem.n_rows();
    imp_lo = lo_in;
    imp_hi = hi_in;
    // Rows to revisit: all at first, then only rows holding a column whose
    // implied bound moved in the previous round.
    ColumnEntries columns;   // built on the first round that moves a bound
    std::vector<char> queued(sz(m), 1), next(sz(m), 0);
    std::vector<core::Index> active(sz(m));
    for (core::Index i = 0; i < m; ++i) active[sz(i)] = i;
    std::vector<f64> row_min, row_max;
    for (int round = 0; round < 20 && !active.empty(); ++round) {
        std::vector<core::Index> touched_cols;
        for (const core::Index i : active) {
            queued[sz(i)] = 0;
            model::DyadicSum min_sum, max_sum;
            row_min.clear(); row_max.clear();
            int nmin = 0, nmax = 0;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const f64 a = problem.A.vals[sz(k)];
                const core::Index c = ci[sz(k)];
                const f64 bmin = a > 0.0 ? imp_lo[sz(c)] : imp_hi[sz(c)];
                const f64 bmax = a > 0.0 ? imp_hi[sz(c)] : imp_lo[sz(c)];
                row_min.push_back(bmin);
                row_max.push_back(bmax);
                if (a == 0.0) continue;
                if (std::isfinite(bmin)) min_sum.add_product(a, bmin); else ++nmin;
                if (std::isfinite(bmax)) max_sum.add_product(a, bmax); else ++nmax;
            }
            // Screening reads the exact sums approximately; exact quotients
            // are formed only for a promising entry.
            const long double amin_hint = min_sum.approx(), amax_hint = max_sum.approx();
            const f64 lo = problem.row_lo[sz(i)], hi = problem.row_hi[sz(i)];
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const f64 a = problem.A.vals[sz(k)];
                if (a == 0.0) continue;
                const core::Index c = ci[sz(k)];
                const f64 bmin = row_min[static_cast<std::size_t>(k - rp[sz(i)])];
                const f64 bmax = row_max[static_cast<std::size_t>(k - rp[sz(i)])];
                // Screening only saves work. A cancellation error here can
                // miss a useful tightening, but cannot create an invalid one.
                bool useful = false;
                const auto screen = [&](long double value, bool lower) {
                    const long double current = lower ? imp_lo[sz(c)] : imp_hi[sz(c)];
                    return lower ? value > current + 1e-9L * (1 + std::fabs(value))
                                 : value < current - 1e-9L * (1 + std::fabs(value));
                };
                if (std::isfinite(lo) && (nmax == 0 || (!std::isfinite(bmax) && nmax == 1)))
                    useful |= screen((static_cast<long double>(lo) - amax_hint +
                        (std::isfinite(bmax) ? static_cast<long double>(a) * bmax : 0)) / a, a > 0);
                if (std::isfinite(hi) && (nmin == 0 || (!std::isfinite(bmin) && nmin == 1)))
                    useful |= screen((static_cast<long double>(hi) - amin_hint +
                        (std::isfinite(bmin) ? static_cast<long double>(a) * bmin : 0)) / a, a < 0);
                if (!useful) continue;
                const bool fmin = std::isfinite(bmin) ? nmin == 0 : nmin == 1;
                const bool fmax = std::isfinite(bmax) ? nmax == 0 : nmax == 1;
                bool moved = false;
                // a x_c in [lo - rmax, hi - rmin], rmax = amax - a bmax (the
                // rest of the row), so x_c's bound is (side - amax + a b) / a,
                // rounded outward exactly.
                const auto side_quotient = [&](f64 side, const model::DyadicSum& whole,
                                               f64 own, bool down) {
                    model::DyadicSum v = whole;
                    v.negate();
                    v.add(side);
                    if (std::isfinite(own)) v.add_product(a, own);
                    return down ? v.quotient_down(a) : v.quotient_up(a);
                };
                if (std::isfinite(lo) && fmax) {
                    if (a > 0.0) {
                        const f64 nb = side_quotient(lo, max_sum, bmax, true);
                        if (nb > imp_lo[sz(c)] + 1e-9 * (1.0 + std::fabs(nb))) { imp_lo[sz(c)] = nb; moved = true; }
                    } else {
                        const f64 nb = side_quotient(lo, max_sum, bmax, false);
                        if (nb < imp_hi[sz(c)] - 1e-9 * (1.0 + std::fabs(nb))) { imp_hi[sz(c)] = nb; moved = true; }
                    }
                }
                if (std::isfinite(hi) && fmin) {
                    if (a > 0.0) {
                        const f64 nb = side_quotient(hi, min_sum, bmin, false);
                        if (nb < imp_hi[sz(c)] - 1e-9 * (1.0 + std::fabs(nb))) { imp_hi[sz(c)] = nb; moved = true; }
                    } else {
                        const f64 nb = side_quotient(hi, min_sum, bmin, true);
                        if (nb > imp_lo[sz(c)] + 1e-9 * (1.0 + std::fabs(nb))) { imp_lo[sz(c)] = nb; moved = true; }
                    }
                }
                if (moved) touched_cols.push_back(c);
            }
        }
        if (touched_cols.empty()) break;
        if (columns.ptr.empty()) columns = column_entries(problem);
        std::vector<core::Index> upcoming;
        for (const core::Index c : touched_cols)
            for (core::Offset e = columns.ptr[sz(c)]; e < columns.ptr[sz(c) + 1]; ++e) {
                const core::Index i = columns.row[static_cast<std::size_t>(e)];
                if (!next[sz(i)]) { next[sz(i)] = 1; upcoming.push_back(i); }
            }
        for (const core::Index i : upcoming) next[sz(i)] = 0;
        std::sort(upcoming.begin(), upcoming.end());
        active = std::move(upcoming);
    }
}

// Implied bounds depend only on the matrix, the row sides and the column
// box, and one solve asks for them many times (pilot: 7 calls of ~0.8 s each
// across the dual bound, the point check and certificate repair). A small
// per-thread cache keyed by a content hash returns the identical result.
void row_implied_bounds(const model::LpProblem& problem,
                        const std::vector<f64>& lo_in,
                        const std::vector<f64>& hi_in,
                        std::vector<f64>& imp_lo, std::vector<f64>& imp_hi) {
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    const auto mix_values = [&](const std::vector<f64>& v) {
        mix(v.size());
        for (const f64 x : v) mix(std::bit_cast<std::uint64_t>(x));
    };
    mix(static_cast<std::uint64_t>(problem.n_rows()));
    mix(static_cast<std::uint64_t>(problem.n_cols()));
    for (const auto o : problem.A.pattern.row_ptr()) mix(static_cast<std::uint64_t>(o));
    for (const auto c : problem.A.pattern.col_idx()) mix(static_cast<std::uint64_t>(c));
    mix_values(problem.A.vals);
    mix_values(problem.row_lo);
    mix_values(problem.row_hi);
    mix_values(lo_in);
    mix_values(hi_in);
    // The hash only selects a candidate; a hit is confirmed by comparing the
    // full inputs, so a collision can never return another model's bounds.
    struct Entry {
        std::uint64_t key;
        std::vector<core::Offset> row_ptr;
        std::vector<core::Index> col_idx;
        std::vector<f64> vals, row_lo, row_hi, col_lo, col_hi, lo, hi;
    };
    thread_local std::vector<Entry> cache;
    const auto same_bits = [](const std::vector<f64>& a, const std::vector<f64>& b) {
        return a.size() == b.size() &&
               (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(f64)) == 0);
    };
    for (const auto& e : cache)
        if (e.key == h && e.row_ptr == problem.A.pattern.row_ptr() &&
            e.col_idx == problem.A.pattern.col_idx() && same_bits(e.vals, problem.A.vals) &&
            same_bits(e.row_lo, problem.row_lo) && same_bits(e.row_hi, problem.row_hi) &&
            same_bits(e.col_lo, lo_in) && same_bits(e.col_hi, hi_in)) {
            imp_lo = e.lo; imp_hi = e.hi; return;
        }
    row_implied_bounds_uncached(problem, lo_in, hi_in, imp_lo, imp_hi);
    if (cache.size() >= 4) cache.erase(cache.begin());
    cache.push_back({h, problem.A.pattern.row_ptr(), problem.A.pattern.col_idx(), problem.A.vals,
                     problem.row_lo, problem.row_hi, lo_in, hi_in, imp_lo, imp_hi});
}

}  // namespace

std::pair<std::vector<f64>, std::vector<f64>> implied_lp_column_bounds(
    const model::LpProblem& problem) {
    std::pair<std::vector<f64>, std::vector<f64>> bounds;
    row_implied_bounds(problem, problem.col_lo, problem.col_hi, bounds.first, bounds.second);
    return bounds;
}

ProofEvidence check_lp_point(const model::LpProblem& problem,
                             const core::RawResult& raw,
                             f64 primal_feas_tol,
                             f64 dual_feas_tol,
                             f64 gap_tol,
                             bool has_basis) {
    SOR_FN();
    ProofEvidence ev;
    ev.has_basis = has_basis;
    ev.primal_feas_tol = primal_feas_tol;
    ev.dual_feas_tol = dual_feas_tol;
    ev.gap_tol = gap_tol;
    ev.claimed_level = raw.proposed_level;
    ev.empty_domain = problem.find_empty_domain().index >= 0;
    ev.lp_values_checked = true;

    const auto m = static_cast<std::size_t>(problem.n_rows());
    const auto n = static_cast<std::size_t>(problem.n_cols());
    if (raw.x.size() != n) return ev;
    for (f64 v : raw.x) if (!std::isfinite(v)) return ev;

    ev.checked_objective = problem.objective(raw.x);
    long double objective_magnitude = std::fabs(static_cast<long double>(problem.obj_offset));
    for (std::size_t j = 0; j < n; ++j)
        objective_magnitude += std::fabs(static_cast<long double>(problem.c[j]) * raw.x[j]);
    const long double objective_envelope = 64 * std::numeric_limits<f64>::epsilon() *
        (1 + objective_magnitude);
    if (std::isfinite(raw.objective) &&
        std::fabs(static_cast<long double>(raw.objective) - ev.checked_objective) > objective_envelope)
        ev.reported_values_consistent = false;
    ev.max_primal_violation = std::max(problem.max_row_violation(raw.x),
                                       problem.max_bound_violation(raw.x));
    ev.checker_passed = std::isfinite(ev.max_primal_violation) &&
                        ev.max_primal_violation <= primal_feas_tol;

    // A point without row multipliers can still be independently checked for
    // primal feasibility, but it cannot claim a dual bound or KKT proof.
    if (raw.y.size() != m) {
        if (ev.checker_passed && ev.claimed_level < ProofLevel::FeasibleOnly)
            ev.claimed_level = ProofLevel::FeasibleOnly;
        return ev;
    }

    // Public row multipliers follow the simplex convention: in minimization
    // form reduced costs are c - A'y.  Transform only objective sense here;
    // HPR/PDHG negate their internal proximal multiplier when exporting it.
    for (f64 v : raw.y) if (!std::isfinite(v)) {
        ev.checker_passed = false;
        return ev;
    }
    const f64 sense = problem.maximize ? -1.0 : 1.0;
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    const auto columns = column_entries(problem);

    f64 dres = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        // Exact reduced cost c_j - (A'y)_j, read as the nearest double.
        model::DyadicSum reduced;
        reduced.add(sense * problem.c[j]);
        for (core::Offset e = columns.ptr[j]; e < columns.ptr[j + 1]; ++e)
            reduced.add_product(-(sense * raw.y[sz(columns.row[static_cast<std::size_t>(e)])]),
                                problem.A.vals[static_cast<std::size_t>(columns.entry[static_cast<std::size_t>(e)])]);
        const f64 r = reduced.nearest();
        const f64 btol = scaled_tol(primal_feas_tol, raw.x[j]);
        const bool at_lo = std::isfinite(problem.col_lo[j]) &&
                           raw.x[j] <= problem.col_lo[j] + btol;
        const bool at_hi = std::isfinite(problem.col_hi[j]) &&
                           raw.x[j] >= problem.col_hi[j] - btol;
        if (at_lo && !at_hi) dres = std::max(dres, std::max(0.0, -r));
        else if (at_hi && !at_lo) dres = std::max(dres, std::max(0.0, r));
        else if (!at_lo && !at_hi) dres = std::max(dres, std::fabs(r));
    }
    for (std::size_t i = 0; i < m; ++i) {
        model::DyadicSum activity;
        for (core::Offset k = rp[i]; k < rp[i + 1]; ++k)
            activity.add_product(problem.A.vals[sz(k)], raw.x[sz(ci[sz(k)])]);
        const f64 value = activity.nearest();
        const f64 multiplier = sense * raw.y[i];
        const f64 btol = scaled_tol(primal_feas_tol, value);
        const bool at_lo = std::isfinite(problem.row_lo[i]) &&
                           value <= problem.row_lo[i] + btol;
        const bool at_hi = std::isfinite(problem.row_hi[i]) &&
                           value >= problem.row_hi[i] - btol;
        if (at_lo && !at_hi)
            dres = std::max(dres, std::max(0.0, -multiplier));
        else if (at_hi && !at_lo)
            dres = std::max(dres, std::max(0.0, multiplier));
        else if (!at_lo && !at_hi)
            dres = std::max(dres, std::fabs(multiplier));
    }
    ev.max_dual_violation = dres;

    std::vector<f64> y_min(m);
    for (std::size_t i = 0; i < m; ++i) y_min[i] = sense * raw.y[i];
    auto bound = safe_lagrangian_lower_bound(problem, y_min, problem.col_lo,
                                             problem.col_hi, dual_feas_tol);
    if (!raw.exact_dual.empty()) {
        const auto exact = exact_dual_lower_bound(problem, raw.exact_dual);
        if (exact.finite && (!bound.finite || exact.value > bound.value)) bound = exact;
    }
    if (bound.finite) {
        ev.checked_dual_bound = sense * bound.value;
        if (std::isfinite(raw.dual_bound) && std::fabs(raw.dual_bound - ev.checked_dual_bound) >
            std::max(1e-9, gap_tol) * (1 + std::fabs(ev.checked_dual_bound))) ev.reported_values_consistent = false;
        ev.gap_rel = std::fabs(ev.checked_objective - ev.checked_dual_bound) /
                     (1.0 + std::fabs(ev.checked_objective));
    }
    // checker_passed means full KKT at the stated tolerances when duals
    // are present, not merely a finite dual residual.
    ev.checker_passed = ev.checker_passed &&
                        std::isfinite(ev.max_dual_violation) &&
                        ev.max_dual_violation <= dual_feas_tol;
    return ev;
}

core::PrimalRay check_primal_ray(const model::LpProblem& problem,
                                 const std::vector<f64>& direction,
                                 f64 tolerance) {
    SOR_FN();
    core::PrimalRay out;
    if (direction.size() != static_cast<std::size_t>(problem.n_cols())) return out;

    f64 norm_inf = 0.0;
    for (f64 v : direction) {
        if (!std::isfinite(v)) return out;
        norm_inf = std::max(norm_inf, std::fabs(v));
    }
    if (!(norm_inf > 0.0) || !std::isfinite(norm_inf)) return out;
    out.direction = direction;
    for (f64& v : out.direction) v /= norm_inf;

    std::vector<model::ExactSum> ad(static_cast<std::size_t>(problem.n_rows()));
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            ad[sz(i)].add_product(problem.A.vals[sz(k)], out.direction[sz(ci[sz(k)])]);

    f64 row_res = 0.0;
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const auto v = ad[sz(i)].value();
        const bool has_lo = std::isfinite(problem.row_lo[sz(i)]);
        const bool has_hi = std::isfinite(problem.row_hi[sz(i)]);
        if (has_lo && has_hi) row_res = std::max(row_res, model::rounded_up(v < 0 ? -v : v));
        else if (has_hi) row_res = std::max(row_res, model::rounded_up(std::max(model::Rational(0), v)));
        else if (has_lo) row_res = std::max(row_res, model::rounded_up(std::max(model::Rational(0), model::Rational(-v))));
    }

    f64 bound_res = 0.0;
    for (std::size_t j = 0; j < out.direction.size(); ++j) {
        const f64 v = out.direction[j];
        const bool has_lo = std::isfinite(problem.col_lo[j]);
        const bool has_hi = std::isfinite(problem.col_hi[j]);
        if (has_lo && has_hi) bound_res = std::max(bound_res, std::fabs(v));
        else if (has_hi) bound_res = std::max(bound_res, std::max(0.0, v));
        else if (has_lo) bound_res = std::max(bound_res, std::max(0.0, -v));
    }

    model::ExactSum slope;
    for (std::size_t j = 0; j < out.direction.size(); ++j)
        slope.add_product(problem.c[j], out.direction[j]);
    out.objective_direction = problem.maximize ? model::rounded_down(slope.value())
                                               : model::rounded_up(slope.value());
    out.max_row_residual = row_res;
    out.max_bound_sign_residual = bound_res;
    const f64 improving = problem.maximize ? out.objective_direction
                                           : -out.objective_direction;
    out.certified = row_res == 0.0 && bound_res == 0.0 &&
                    improving > tolerance;
    return out;
}

DualRayEnclosure enclose_dual_ray(const model::LpProblem& problem,
                                  const std::vector<f64>& multipliers) {
    constexpr long double u = std::numeric_limits<long double>::epsilon() / 2.0L;
    const auto gamma = [&](long double k) { return (k * u) / (1.0L - k * u); };
    DualRayEnclosure out;
    const auto n = static_cast<std::size_t>(problem.n_cols());
    out.d.assign(n, 0.0L);
    out.d_rad.assign(n, 0.0L);
    out.d_terms.assign(n, 0);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    long double upper_abs = 0.0L;
    std::uint64_t upper_terms = 0;
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 y = multipliers[sz(i)];
        if (y == 0.0) continue;
        const f64 side = y > 0.0 ? problem.row_hi[sz(i)] : problem.row_lo[sz(i)];
        if (!std::isfinite(side)) out.upper_finite = false;
        else {
            const long double t = static_cast<long double>(y) * side;
            out.upper += t;
            upper_abs += std::fabs(t);
            ++upper_terms;
        }
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const auto j = sz(ci[sz(k)]);
            const long double p = static_cast<long double>(y) * problem.A.vals[sz(k)];
            out.d[j] += p;
            out.d_rad[j] += std::fabs(p);   // the magnitude sum, scaled below
            ++out.d_terms[j];
        }
    }
    for (std::size_t j = 0; j < n; ++j)
        out.d_rad[j] *= gamma(2.0L * out.d_terms[j] + 2.0L);
    out.upper_rad = gamma(2.0L * static_cast<long double>(upper_terms) + 2.0L) * upper_abs;
    core::work_add(static_cast<std::uint64_t>(problem.A.vals.size()) + n +
                       static_cast<std::uint64_t>(problem.n_rows()),
                   core::WorkTag::Interval);
    return out;
}

RayLowerEnclosure enclose_ray_lower(const DualRayEnclosure& ray,
                                    const std::vector<f64>& lo,
                                    const std::vector<f64>& hi) {
    constexpr long double u = std::numeric_limits<long double>::epsilon() / 2.0L;
    const auto gamma = [&](long double k) { return (k * u) / (1.0L - k * u); };
    RayLowerEnclosure out;
    long double sum_lo = 0.0L, sum_hi = 0.0L, mag = 0.0L;
    std::uint64_t terms = 0;
    for (std::size_t j = 0; j < ray.d.size(); ++j) {
        if (ray.d_terms[j] == 0) continue;   // d_j = 0 exactly
        const long double lo_d = ray.d[j] - ray.d_rad[j], hi_d = ray.d[j] + ray.d_rad[j];
        const f64 l = lo[j], h = hi[j];
        // Each column's term d_j b_j, bracketed over the interval (over both
        // signs when it meets 0).
        long double t_lo, t_hi;
        if (lo_d > 0.0L) {
            if (!std::isfinite(l)) { out.needs_infinite = true; return out; }
            t_lo = l >= 0.0 ? lo_d * l : hi_d * l;
            t_hi = l >= 0.0 ? hi_d * l : lo_d * l;
        } else if (hi_d < 0.0L) {
            if (!std::isfinite(h)) { out.needs_infinite = true; return out; }
            t_lo = h >= 0.0 ? lo_d * h : hi_d * h;
            t_hi = h >= 0.0 ? hi_d * h : lo_d * h;
        } else {
            if (!std::isfinite(l) || !std::isfinite(h)) return out;   // undecided
            t_lo = std::min(std::min(lo_d * l, lo_d * h), std::min(hi_d * l, hi_d * h));
            t_hi = std::max(std::max(lo_d * l, lo_d * h), std::max(hi_d * l, hi_d * h));
        }
        sum_lo += t_lo;
        sum_hi += t_hi;
        mag += std::max(std::fabs(t_lo), std::fabs(t_hi));
        ++terms;
    }
    const long double rad = gamma(2.0L * static_cast<long double>(terms) + 2.0L) * mag;
    out.lo = sum_lo - rad;
    out.hi = sum_hi + rad;
    out.decided = std::isfinite(static_cast<double>(out.lo)) &&
                  std::isfinite(static_cast<double>(out.hi));
    return out;
}

core::DualFarkasRay check_dual_farkas_ray(
    const model::LpProblem& problem,
    const std::vector<f64>& multipliers,
    f64 tolerance) {
    core::WorkScope work_scope_(core::WorkCat::Exact);
    SOR_FN();
    core::DualFarkasRay out;
    if (multipliers.size() != static_cast<std::size_t>(problem.n_rows())) return out;
    f64 norm_inf = 0.0;
    for (f64 v : multipliers) {
        if (!std::isfinite(v)) return out;
        norm_inf = std::max(norm_inf, std::fabs(v));
    }
    if (!(norm_inf > 0.0) || !std::isfinite(norm_inf)) return out;
    out.multipliers = multipliers;
    for (f64& v : out.multipliers) v /= norm_inf;
    // A multiplier facing an infinite row side can never contribute to a
    // valid proof. Rather than rejecting the whole ray for round-off noise
    // there, drop such entries: the result is a different candidate ray,
    // and everything below checks that candidate exactly.
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        f64& y = out.multipliers[sz(i)];
        if ((y > 0.0 && !std::isfinite(problem.row_hi[sz(i)])) ||
            (y < 0.0 && !std::isfinite(problem.row_lo[sz(i)])))
            y = 0.0;
    }

    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    std::vector<long double> aty;
    const auto compute_aty = [&]() {
        aty.assign(static_cast<std::size_t>(problem.n_cols()), 0.0L);
        for (core::Index i = 0; i < problem.n_rows(); ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                aty[sz(ci[sz(k)])] +=
                    static_cast<long double>(out.multipliers[sz(i)]) *
                    problem.A.vals[sz(k)];
    };
    compute_aty();

    // A'y must vanish exactly in every unbounded direction. An approximate
    // ray (first-order iterate, or simplex round-off) leaves tiny residuals
    // there. Repair: for each such column, shift the largest multiplier in
    // that column to cancel the residual, provided the shifted multiplier
    // still faces a finite row side. The repaired ray is a new candidate and
    // is checked exactly below; the repair itself is never trusted.
    const auto unbounded_residual = [&](core::Index j) {
        const long double d = aty[sz(j)];
        return (d > 0.0L && !std::isfinite(problem.col_lo[sz(j)])) ||
               (d < 0.0L && !std::isfinite(problem.col_hi[sz(j)]));
    };
    constexpr int kMaxRepairColumns = 64;
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<char> bad(static_cast<std::size_t>(problem.n_cols()), 0);
        int n_bad = 0;
        for (core::Index j = 0; j < problem.n_cols(); ++j)
            if (unbounded_residual(j)) { bad[sz(j)] = 1; ++n_bad; }
        if (n_bad == 0 || n_bad > kMaxRepairColumns) break;
        // Best row per offending column: largest |y_i| with a_ij != 0.
        std::vector<core::Index> best_row(static_cast<std::size_t>(problem.n_cols()), -1);
        std::vector<f64> best_a(static_cast<std::size_t>(problem.n_cols()), 0.0);
        for (core::Index i = 0; i < problem.n_rows(); ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const core::Index j = ci[sz(k)];
                if (!bad[sz(j)] || problem.A.vals[sz(k)] == 0.0) continue;
                const core::Index b = best_row[sz(j)];
                if (b < 0 || std::fabs(out.multipliers[sz(i)]) >
                                 std::fabs(out.multipliers[sz(b)])) {
                    best_row[sz(j)] = i;
                    best_a[sz(j)] = problem.A.vals[sz(k)];
                }
            }
        for (core::Index j = 0; j < problem.n_cols(); ++j) {
            const core::Index i = best_row[sz(j)];
            if (i < 0) continue;
            const f64 y = static_cast<f64>(
                out.multipliers[sz(i)] - aty[sz(j)] / best_a[sz(j)]);
            const bool compatible =
                y == 0.0 ||
                (y > 0.0 && std::isfinite(problem.row_hi[sz(i)])) ||
                (y < 0.0 && std::isfinite(problem.row_lo[sz(i)]));
            if (compatible) out.multipliers[sz(i)] = y;
        }
        compute_aty();
    }

    std::vector<f64> imp_lo, imp_hi;
    const auto implied_bounds = [&]() {
        row_implied_bounds(problem, problem.col_lo, problem.col_hi, imp_lo, imp_hi);
    };

    // Interval first pass: the certificate below, decided in long double
    // with every rounding bounded (enclose_dual_ray). It certifies only when
    // the pessimistic contradiction still clears the separation tolerance,
    // compared as the exact rule compares (contradiction rounded down,
    // tolerance rounded up, the tolerance's magnitude taken from above), so
    // whatever it certifies the exact evaluation would certify too.
    // Anything else -- a d_j whose interval meets 0 on a column missing a
    // bound either sign could need, a narrow margin -- goes to the exact
    // evaluation below, which alone ever refuses. Every infeasible node LP's
    // ray is checked by the engine and again by each consumer; in exact
    // arithmetic that was 15% of mas76's search.
    {
        const auto ray = enclose_dual_ray(problem, out.multipliers);
        // Declared bounds, else row-implied ones, as the exact rule charges.
        std::vector<f64> lo_b = problem.col_lo, hi_b = problem.col_hi;
        bool want_implied = false;
        for (std::size_t j = 0; j < lo_b.size() && !want_implied; ++j)
            if (ray.d_terms[j] != 0 &&
                ((ray.d[j] + ray.d_rad[j] >= 0.0L && !std::isfinite(lo_b[j])) ||
                 (ray.d[j] - ray.d_rad[j] <= 0.0L && !std::isfinite(hi_b[j]))))
                want_implied = true;
        if (want_implied) {
            implied_bounds();
            for (std::size_t j = 0; j < lo_b.size(); ++j) {
                if (!std::isfinite(lo_b[j])) lo_b[j] = imp_lo[j];
                if (!std::isfinite(hi_b[j])) hi_b[j] = imp_hi[j];
            }
        }
        const auto lower = enclose_ray_lower(ray, lo_b, hi_b);
        if (ray.upper_finite && lower.decided && !lower.needs_infinite) {
            constexpr long double u = std::numeric_limits<long double>::epsilon() / 2.0L;
            const long double upper_hi = ray.upper + ray.upper_rad;
            const long double diff = lower.lo - upper_hi;
            const long double contradiction_lo =
                diff - 2.0L * u * (std::fabs(lower.lo) + std::fabs(upper_hi));
            const long double mag = std::max(
                std::max(std::fabs(lower.lo), std::fabs(lower.hi)),
                std::fabs(ray.upper) + ray.upper_rad);
            const long double sep =
                static_cast<long double>(tolerance) * (1.0L + mag) * (1.0L + 8.0L * u);
            f64 c = static_cast<f64>(contradiction_lo);
            if (static_cast<long double>(c) > contradiction_lo)
                c = std::nextafter(c, -std::numeric_limits<f64>::infinity());
            f64 sep_d = static_cast<f64>(sep);
            if (static_cast<long double>(sep_d) < sep)
                sep_d = std::nextafter(sep_d, std::numeric_limits<f64>::infinity());
            if (std::isfinite(c) && std::isfinite(sep_d) && c > sep_d) {
                out.max_homogeneous_residual = 0.0;
                out.max_sign_residual = 0.0;
                out.contradiction = c;
                out.certified = true;
                return out;
            }
        }
    }

    // A residual left on a column with no bound on the side it needs cannot
    // be charged to the column's declared bounds. It can be charged to a
    // bound the ROWS imply: for a_ij x_j + sum_{k!=j} a_ik x_k in [lo, hi],
    // every feasible point satisfies x_j >= (lo - max rest)/a_ij (a_ij > 0)
    // etc., using only declared bounds of the other columns. That keeps the
    // certificate exact (the contribution d_j x_j is bounded for every
    // feasible x) instead of waving the residual through as round-off; a
    // column with no finite implied bound on the needed side still refutes
    // the ray. One pass, no propagation chains.

    f64 sign_res = 0.0;
    bool incompatible_unbounded_support = false;
    std::vector<model::ExactSum> exact_sums(static_cast<std::size_t>(problem.n_cols()));
    for (core::Index i = 0; i < problem.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            exact_sums[sz(ci[sz(k)])].add_product(out.multipliers[sz(i)], problem.A.vals[sz(k)]);
    model::Rational lower = 0;
    for (core::Index j = 0; j < problem.n_cols(); ++j) {
        const model::Rational d = exact_sums[sz(j)].value();
        if (d > 0.0) {
            f64 b = problem.col_lo[sz(j)];
            if (!std::isfinite(b)) {
                if (imp_lo.empty()) implied_bounds();
                b = imp_lo[sz(j)];
            }
            if (!std::isfinite(b)) {
                incompatible_unbounded_support = true;
                sign_res = std::max(sign_res, d.convert_to<f64>());
                if (core::env_switches().farkas_debug)
                    std::fprintf(stderr, "[farkas] col %d d=%.3Lg bounds [%g,%g]\n",
                                 (int)j, d.convert_to<long double>(), problem.col_lo[sz(j)], problem.col_hi[sz(j)]);
                continue;
            }
            lower += d * model::Rational(b);
        } else if (d < 0.0) {
            f64 b = problem.col_hi[sz(j)];
            if (!std::isfinite(b)) {
                if (imp_hi.empty()) implied_bounds();
                b = imp_hi[sz(j)];
            }
            if (!std::isfinite(b)) {
                incompatible_unbounded_support = true;
                sign_res = std::max(sign_res, (-d).convert_to<f64>());
                if (core::env_switches().farkas_debug)
                    std::fprintf(stderr, "[farkas] col %d d=%.3Lg bounds [%g,%g]\n",
                                 (int)j, d.convert_to<long double>(), problem.col_lo[sz(j)], problem.col_hi[sz(j)]);
                continue;
            }
            lower += d * model::Rational(b);
        }
    }

    model::Rational upper = 0;
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 y = out.multipliers[sz(i)];
        if (y > 0.0)
            upper += model::Rational(y) * model::Rational(problem.row_hi[sz(i)]);
        else if (y < 0.0)
            upper += model::Rational(y) * model::Rational(problem.row_lo[sz(i)]);
    }

    out.max_homogeneous_residual = sign_res;
    out.max_sign_residual = sign_res;
    out.contradiction = model::rounded_down(lower - upper);
    const model::Rational scale = 1 + std::max(model::Rational(abs(lower)), model::Rational(abs(upper)));
    const f64 separation_tol = model::rounded_up(model::Rational(tolerance) * scale);
    out.certified = !incompatible_unbounded_support &&
                    sign_res <= tolerance &&
                    std::isfinite(out.contradiction) &&
                    out.contradiction > separation_tol;
    return out;
}

ProofEvidence check_lp_result(const model::LpProblem& problem,
                              const core::RawResult& raw,
                              const ProofEvidence& proposed) {
    SOR_FN();
    ProofEvidence checked = check_lp_point(
        problem, raw, proposed.primal_feas_tol, proposed.dual_feas_tol,
        proposed.gap_tol, proposed.has_basis);
    checked.claimed_level = proposed.claimed_level;
    checked.rational_verified = proposed.rational_verified;
    checked.vipr_verified = proposed.vipr_verified;

    const auto& primal_direction = raw.primal_ray.direction;
    if (!primal_direction.empty() || !raw.primal_ray.exact_direction.empty()) {
        const core::PrimalRay ray = !raw.primal_ray.exact_direction.empty()
            ? check_exact_primal_ray(problem, raw.primal_ray.exact_direction, proposed.primal_feas_tol)
            : check_primal_ray(problem, primal_direction, proposed.primal_feas_tol);
        checked.primal_ray_violation = ray.certified ? std::max(
            ray.max_row_residual, ray.max_bound_sign_residual) : core::kPosInf;
        // Evidence uses minimization convention so finalize_result can apply
        // one sign-independent rule; PrimalRay retains the original-model
        // objective direction for users.
        checked.primal_ray_objective = problem.maximize
            ? -ray.objective_direction : ray.objective_direction;
    }

    const std::vector<f64>* dual_multipliers = nullptr;
    if (!raw.dual_farkas_ray.multipliers.empty())
        dual_multipliers = &raw.dual_farkas_ray.multipliers;
    else if (!raw.ray.empty())
        dual_multipliers = &raw.ray;
    if (dual_multipliers != nullptr || !raw.dual_farkas_ray.exact_multipliers.empty()) {
        const core::DualFarkasRay ray = !raw.dual_farkas_ray.exact_multipliers.empty()
            ? check_exact_dual_farkas_ray(problem, raw.dual_farkas_ray.exact_multipliers, proposed.primal_feas_tol)
            : check_dual_farkas_ray(problem, *dual_multipliers, proposed.primal_feas_tol);
        checked.dual_farkas_contradiction = ray.contradiction;
        // A finite residual plus contradiction > tol is not sufficient: the
        // checker also applies a scale-aware separation threshold.  Publish
        // an acceptable violation only after that full test passes, otherwise
        // finalize_result could re-accept a near-certificate using its legacy
        // scalar fields.
        if (ray.certified) {
            checked.dual_farkas_violation = std::max(
                ray.max_homogeneous_residual, ray.max_sign_residual);
            checked.ray_violation = checked.dual_farkas_violation;
        }
    } else {
        // One-release compatibility for simplex's existing independently
        // recomputed Farkas evidence when it did not export row multipliers.
        checked.ray_violation = proposed.ray_violation;
    }
    return checked;
}

CheckedLpResult check_lp_candidate(const model::LpProblem& problem,
                                   RawResult raw, const ProofEvidence& proposed) {
    problem.validate(/*allow_empty_domains=*/true);
    if (!raw.dual_farkas_ray.exact_multipliers.empty()) {
        const auto ray = check_exact_dual_farkas_ray(problem,
            raw.dual_farkas_ray.exact_multipliers, proposed.primal_feas_tol);
        if (ray.certified) {
            raw.dual_farkas_ray.multipliers = ray.multipliers;
            raw.ray = ray.multipliers;
        }
    }
    if (!raw.primal_ray.exact_direction.empty()) {
        const auto ray = check_exact_primal_ray(problem, raw.primal_ray.exact_direction, proposed.primal_feas_tol);
        if (ray.certified) raw.primal_ray.direction = ray.direction;
    }
    const auto checked = check_lp_result(problem, raw, proposed);
    return CheckedLpResult(std::move(raw), checked);
}
SolveResult finalize_result(CheckedLpResult checked) {
    return finalize_result(std::move(checked.raw_), checked.evidence_);
}

SolveResult finalize_result(RawResult raw, const ProofEvidence& ev) {
    SOR_FN();
    SolveResult r;
    r.objective          = ev.lp_values_checked ? ev.checked_objective : raw.objective;
    r.dual_bound         = ev.lp_values_checked ? ev.checked_dual_bound : raw.dual_bound;
    r.x                  = std::move(raw.x);
    r.y                  = std::move(raw.y);
    r.exact_dual         = std::move(raw.exact_dual);
    r.ray                = std::move(raw.ray);
    r.primal_ray         = std::move(raw.primal_ray);
    r.dual_farkas_ray    = std::move(raw.dual_farkas_ray);
    r.iterations         = raw.iterations;
    r.engine             = std::move(raw.engine);
    r.backend            = std::move(raw.backend);
    r.termination_reason = std::move(raw.termination_reason);

    r.max_primal_violation = ev.max_primal_violation;
    r.max_dual_violation   = ev.max_dual_violation;
    r.gap_rel              = ev.gap_rel;

    r.proof  = supported_level(ev);
    r.status = raw.proposed_status;

    // Same rule as Status::Optimal below, applied to the Farkas certificate:
    // an engine PROPOSES a ray, only this function may certify it, and only
    // once it is independently re-verified against the unscaled model
    // (ev.ray_violation, computed by farkas_violation() -- never the
    // engine's own view of its termination). No ray, or one that doesn't
    // clear the tolerance, leaves ray_certified false and r.ray cleared: an
    // honest "infeasible, no proof" rather than a wrong claim.
    if (r.dual_farkas_ray.multipliers.empty() && !r.ray.empty())
        r.dual_farkas_ray.multipliers = r.ray;
    if (r.ray.empty() && !r.dual_farkas_ray.multipliers.empty())
        r.ray = r.dual_farkas_ray.multipliers;

    const bool checked_dual_ray =
        ((!r.dual_farkas_ray.multipliers.empty() ||
          !r.dual_farkas_ray.exact_multipliers.empty()) &&
         std::isfinite(ev.dual_farkas_violation) &&
         ev.dual_farkas_violation <= ev.primal_feas_tol &&
         ev.dual_farkas_contradiction > ev.primal_feas_tol);
    if (r.status == Status::Infeasible && checked_dual_ray) {
        r.ray_certified = true;
        r.dual_farkas_ray.certified = true;
        r.dual_farkas_ray.max_homogeneous_residual =
            std::min(ev.dual_farkas_violation, ev.ray_violation);
        r.dual_farkas_ray.contradiction = ev.dual_farkas_contradiction;
    } else if (r.status != Status::Infeasible) {
        r.ray.clear();
        r.dual_farkas_ray = core::DualFarkasRay{};
    }

    // Unbounded needs a feasible point + improving ray. Do NOT require
    // checker_passed (full KKT): an unbounded LP is dual-infeasible, so the
    // dual residual check that now gates checker_passed would always fail.
    const bool checked_primal_ray =
        r.status == Status::Unbounded && !r.primal_ray.direction.empty() &&
        ev.max_primal_violation <= ev.primal_feas_tol &&
        std::isfinite(ev.primal_ray_violation) &&
        ev.primal_ray_violation <= ev.primal_feas_tol &&
        std::isfinite(ev.primal_ray_objective) &&
        ev.primal_ray_objective < -ev.primal_feas_tol;
    if (checked_primal_ray) r.primal_ray.certified = true;
    else if (r.status != Status::Unbounded)
        r.primal_ray = core::PrimalRay{};

    // A globally complete discrete search is a different proof object from
    // an LP Farkas ray.  Preserve that route while refusing an uncertified LP
    // terminal claim.
    const bool global_proof = (ev.claimed_level == ProofLevel::ProvedGlobalEpsilon ||
                               ev.claimed_level == ProofLevel::ProvedGlobalFP) &&
                              residuals_within_tolerance(ev);
    if (r.status == Status::Infeasible && global_proof)
        r.proof = ev.claimed_level;
    if (r.status == Status::Infeasible && !checked_dual_ray && !global_proof &&
        !ev.empty_domain) {
        r.status = Status::NoSolutionFound;
        r.ray.clear();
        r.dual_farkas_ray = core::DualFarkasRay{};
        r.downgrade_reason =
            "Infeasible rejected: no independently checked dual Farkas ray";
    }
    if (r.status == Status::Unbounded && !checked_primal_ray) {
        r.status = Status::NoSolutionFound;
        r.primal_ray = core::PrimalRay{};
        r.downgrade_reason =
            "Unbounded rejected: no independently checked primal ray";
    }

    if (raw.proposed_status == Status::Optimal) {
        // LP needs ProvedOptimalFP (basis). Convex QP may claim Optimal at
        // ProvedKKT, and a complete branch-and-bound tree may claim Optimal at
        // ProvedGlobalEpsilon when every relaxation bound is proved, or at
        // ProvedGlobalFP when some were tolerance-optimal LP values. Anything
        // weaker is demoted.
        const bool strong_enough =
            r.proof >= ProofLevel::ProvedOptimalFP ||
            r.proof == ProofLevel::ProvedKKT ||
            r.proof == ProofLevel::ProvedGlobalFP ||
            r.proof == ProofLevel::ProvedGlobalEpsilon;
        if (!strong_enough) {
            // The load-bearing rule of the whole codebase.
            r.status = (r.proof >= ProofLevel::FeasibleOnly) ? Status::Feasible
                                                             : Status::NoSolutionFound;
            r.downgrade_reason =
                "Optimal rejected: evidence supports only " +
                std::string(core::to_string(r.proof));
        } else if (!ev.checker_passed || !std::isfinite(r.objective) ||
                   (r.proof >= ProofLevel::ProvedOptimalFP &&
                    !std::isfinite(r.dual_bound))) {
            r.status = Status::NumericalFailure;
            r.downgrade_reason = !ev.checker_passed
                ? "Optimal rejected: independent checker did not pass"
                : "Optimal rejected: objective or dual bound is not finite";
        }
    }
    if (ev.lp_values_checked && !ev.reported_values_consistent &&
        raw.proposed_status == Status::Optimal) {
        r.status = Status::NumericalFailure;
        r.downgrade_reason = "Optimal rejected: reported objective or bound disagrees with checked values";
    }
    if (r.status == Status::Feasible &&
        (!ev.checker_passed || ev.max_primal_violation > ev.primal_feas_tol)) {
        r.status = Status::NoSolutionFound;
        r.downgrade_reason = "Feasible rejected: independent primal check failed";
    }
    return r;
}

// Rigorous lower bound for min{c'x : row_lo <= Ax <= row_hi, lo <= x <= hi}
// from ANY multipliers y (weak duality, no optimality assumed):
//   c'x = (c - A'y)'x + y'Ax >= sum_j min_{x_j in box} d_j x_j
//                               + sum_i min_{a_i in [row_lo,row_hi]} y_i a_i.
// y_i facing an infinite side is set to 0 (still a valid multiplier) and d is
// recomputed from the y actually used. A nonzero d_j that needs an infinite
// column bound is charged to a row-implied bound (valid for every feasible
// point); if none exists the bound is -inf -- never treated as zero. The sum
// is accumulated in long double and reduced by a conservative bound on its
// own rounding error.
namespace {
// Round reduced costs that sit within dual_feas_tol on a free/infinite bound
// side to zero so a dual-feasible point within tolerance keeps a finite bound.
// Enclosures often straddle 0 as [-e, e] from rounding; the one-sided tests
// alone miss that and leave the interval path with no finite chargeable bound
// even though every point in the enclosure is dual-feasible within tolerance.
void snap_infinite_bound_reduced_cost(long double& lo_d, long double& hi_d,
                                     f64 col_lo, f64 col_hi, f64 dual_feas_tol) {
    const bool free_below = !std::isfinite(col_lo);
    const bool free_above = !std::isfinite(col_hi);
    if (!free_below && !free_above) return;
    f64 scale = 1.0;
    if (std::isfinite(col_lo)) scale = std::max(scale, std::fabs(col_lo));
    if (std::isfinite(col_hi)) scale = std::max(scale, std::fabs(col_hi));
    const f64 tol = dual_feas_tol * (1.0 + scale);
    // Whole enclosure inside the dual-feasibility band around 0.
    if (static_cast<f64>(hi_d) <= tol && static_cast<f64>(-lo_d) <= tol) {
        lo_d = hi_d = 0.0L;
        return;
    }
    if (free_below && lo_d > 0.0L && static_cast<f64>(lo_d) <= tol)
        lo_d = hi_d = 0.0L;
    if (free_above && hi_d < 0.0L && static_cast<f64>(-hi_d) <= tol)
        lo_d = hi_d = 0.0L;
}

void snap_infinite_bound_reduced_cost(long double& d, f64 col_lo, f64 col_hi,
                                      f64 dual_feas_tol) {
    long double lo = d, hi = d;
    snap_infinite_bound_reduced_cost(lo, hi, col_lo, col_hi, dual_feas_tol);
    d = lo;
}

// The bound itself, optionally exporting the reduced costs d = c - A'y of the
// (possibly corrected) multipliers it was computed from.
SafeLpBound lagrangian_bound_impl(const model::LpProblem& problem,
                                  const std::vector<f64>& y_min,
                                  const std::vector<f64>& col_lo,
                                  const std::vector<f64>& col_hi,
                                  std::vector<model::Rational>* d_out,
                                  f64 dual_feas_tol) {
    core::WorkScope work_scope_(core::WorkCat::Exact);
    SafeLpBound out;
    const core::Index m = problem.n_rows(), n = problem.n_cols();
    if (y_min.size() != sz(m) || col_lo.size() != sz(n) || col_hi.size() != sz(n))
        return out;
    const f64 sense = problem.maximize ? -1.0 : 1.0;
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    std::vector<long double> d(sz(n));
    for (core::Index j = 0; j < n; ++j) d[sz(j)] = sense * problem.c[sz(j)];
    for (core::Index i = 0; i < m; ++i) {
        long double yi = y_min[sz(i)];
        if (!std::isfinite(static_cast<double>(yi))) return out;
        if (yi > 0.0L && !std::isfinite(problem.row_lo[sz(i)])) yi = 0.0L;
        if (yi < 0.0L && !std::isfinite(problem.row_hi[sz(i)])) yi = 0.0L;
        if (yi == 0.0L) continue;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const long double t2 = yi * problem.A.vals[sz(k)];
            d[sz(ci[sz(k)])] -= t2;
        }
    }
    for (core::Index j = 0; j < n; ++j)
        snap_infinite_bound_reduced_cost(d[sz(j)], col_lo[sz(j)], col_hi[sz(j)],
                                         dual_feas_tol);
    std::vector<f64> imp_lo, imp_hi;
    const auto ensure_implied = [&]() {
        if (imp_lo.empty()) row_implied_bounds(problem, col_lo, col_hi, imp_lo, imp_hi);
    };
    // The finite bound column j's reduced cost d would be charged to, or
    // +/-inf when there is none (declared first, then row-implied).
    const auto needed_bound = [&](core::Index j, long double dj) -> f64 {
        if (dj == 0.0L) return 0.0;
        f64 b = dj > 0.0L ? col_lo[sz(j)] : col_hi[sz(j)];
        if (std::isfinite(b)) return b;
        ensure_implied();
        return dj > 0.0L ? imp_lo[sz(j)] : imp_hi[sz(j)];
    };
    // Multiplier correction. The simplex's y leaves round-off reduced costs
    // (|d| ~ 1e-14) on basic columns; where such a column has no finite bound
    // on the side its sign needs, the bound for THIS y is -inf. Choose a
    // different y instead: shift one y_i, in a row whose other columns all
    // stay chargeable to finite bounds, so d_j lands just on the side j's
    // finite bound can absorb. Weak duality holds for any y; the bound below
    // is computed exactly from the corrected y and d.
    // Column-major entries, built on first need and shared with the exact
    // pass below. A vector of (row, value) vectors per column was rebuilt by
    // every node LP whose duals left a column without a finite bound: about
    // 12% of nu25-pr12's tree search was its allocations.
    ColumnEntries columns;
    const auto ensure_columns = [&]() {
        if (columns.ptr.empty()) columns = column_entries(problem);
    };
    std::vector<f64> y(y_min.begin(), y_min.end());
    for (core::Index i = 0; i < m; ++i) {
        if (y[sz(i)] > 0.0 && !std::isfinite(problem.row_lo[sz(i)])) y[sz(i)] = 0.0;
        if (y[sz(i)] < 0.0 && !std::isfinite(problem.row_hi[sz(i)])) y[sz(i)] = 0.0;
    }
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<core::Index> bad;
        for (core::Index j = 0; j < n; ++j)
            if (d[sz(j)] != 0.0L && !std::isfinite(needed_bound(j, d[sz(j)])))
                bad.push_back(j);
        if (bad.empty() || bad.size() > 256) break;
        ensure_columns();
        for (const core::Index j : bad) {
            const auto& dj = d[sz(j)];
            if (dj == 0.0L || std::isfinite(needed_bound(j, dj))) continue;
            // Target sign: toward the side that has a finite bound.
            const bool lo_ok = std::isfinite(col_lo[sz(j)]) ||
                               (ensure_implied(), std::isfinite(imp_lo[sz(j)]));
            const bool hi_ok = std::isfinite(col_hi[sz(j)]) ||
                               (ensure_implied(), std::isfinite(imp_hi[sz(j)]));
            for (core::Offset e = columns.ptr[sz(j)]; e < columns.ptr[sz(j) + 1]; ++e) {
                const core::Index i = columns.row[static_cast<std::size_t>(e)];
                const f64 aij = problem.A.vals[static_cast<std::size_t>(
                    columns.entry[static_cast<std::size_t>(e)])];
                if (aij == 0.0) continue;
                const long double scale = 1.0L + std::fabs(static_cast<long double>(y[sz(i)]) * aij);
                const long double eta = 1e-11L * scale;
                const long double target = lo_ok ? eta : hi_ok ? -eta : 0.0L;
                // d_j - delta * a_ij = target
                const long double delta = (dj - target) / aij;
                const long double yi = y[sz(i)] + delta;
                if (yi > 0.0L && !std::isfinite(problem.row_lo[sz(i)])) continue;
                if (yi < 0.0L && !std::isfinite(problem.row_hi[sz(i)])) continue;
                bool ok = true;
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1] && ok; ++k) {
                    const core::Index c = ci[sz(k)];
                    if (c == j) continue;
                    const long double nd = d[sz(c)] - delta * problem.A.vals[sz(k)];
                    if (nd != 0.0L && !std::isfinite(needed_bound(c, nd))) ok = false;
                }
                if (!ok) continue;
                y[sz(i)] = static_cast<f64>(yi);
                // Recompute this row's contribution to d exactly from the
                // stored (rounded) multiplier.
                const long double applied = static_cast<long double>(y[sz(i)]) - (yi - delta);
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                    d[sz(ci[sz(k)])] -= applied * problem.A.vals[sz(k)];
                ++out.multiplier_corrections;
                break;
            }
        }
    }
    // Interval fast path (no exact reduced costs requested). The same
    // Lagrangian, evaluated in long double with a rigorous bound on every
    // rounding: each reduced cost d_j lies in [d~_j - e_j, d~_j + e_j] with
    // e_j = gamma(k_j + 2) * (|c_j| + sum |y_i a_ij|) (k_j accumulated terms,
    // gamma(k) = k u / (1 - k u), u = 2^-64), each column contributes the
    // minimum of d * x over that interval and its box (the four corners), and
    // the sum is lowered by gamma(terms + 2) * sum |terms|. Any column whose
    // interval meets an infinite bound it would need is left to the exact
    // pass below (with its row-implied bounds). The exact pass evaluated
    // every node LP's bound in multiprecision: 14% of drayage-25-23's search.
    if (d_out == nullptr) {
        constexpr long double u = std::numeric_limits<long double>::epsilon() / 2.0L;
        const auto gamma = [&](long double k) { return (k * u) / (1.0L - k * u); };
        std::vector<long double> dl(sz(n)), ab(sz(n));
        std::vector<std::uint32_t> cnt(sz(n), 1);
        for (core::Index j = 0; j < n; ++j) {
            dl[sz(j)] = static_cast<long double>(sense * problem.c[sz(j)]);
            ab[sz(j)] = std::fabs(dl[sz(j)]);
        }
        long double L = static_cast<long double>(sense * problem.obj_offset);
        long double absL = std::fabs(L);
        std::uint64_t terms = 1;
        bool fast_ok = true;
        for (core::Index i = 0; i < m && fast_ok; ++i) {
            const f64 yi = y[sz(i)];
            if (yi == 0.0) continue;
            const f64 b = yi > 0.0 ? problem.row_lo[sz(i)] : problem.row_hi[sz(i)];
            if (!std::isfinite(b)) { fast_ok = false; break; }
            const long double t = static_cast<long double>(yi) * b;
            L += t;
            absL += std::fabs(t);
            ++terms;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const core::Index j = ci[sz(k)];
                const long double t2 = static_cast<long double>(yi) * problem.A.vals[sz(k)];
                dl[sz(j)] -= t2;
                ab[sz(j)] += std::fabs(t2);
                ++cnt[sz(j)];
            }
        }
        core::work_add(static_cast<std::uint64_t>(problem.A.vals.size()) +
                           static_cast<std::uint64_t>(n) + static_cast<std::uint64_t>(m),
                       core::WorkTag::Interval);
        for (core::Index j = 0; j < n && fast_ok; ++j) {
            const long double e = gamma(static_cast<long double>(cnt[sz(j)]) + 2.0L) * ab[sz(j)];
            long double lo_d = dl[sz(j)] - e, hi_d = dl[sz(j)] + e;
            const f64 l = col_lo[sz(j)], h = col_hi[sz(j)];
            snap_infinite_bound_reduced_cost(lo_d, hi_d, l, h, dual_feas_tol);
            long double contrib;
            if (lo_d == 0.0L && hi_d == 0.0L) continue;
            if (lo_d >= 0.0L) {
                if (!std::isfinite(l)) { fast_ok = false; break; }   // needs the lower bound
                contrib = l >= 0.0 ? lo_d * l : hi_d * l;
            } else if (hi_d <= 0.0L) {
                if (!std::isfinite(h)) { fast_ok = false; break; }   // needs the upper bound
                contrib = h >= 0.0 ? lo_d * h : hi_d * h;
            } else {
                // Sign undecided: both bounds must be finite.
                if (!std::isfinite(l) || !std::isfinite(h)) { fast_ok = false; break; }
                contrib = std::min(std::min(lo_d * l, lo_d * h), std::min(hi_d * l, hi_d * h));
            }
            L += contrib;
            absL += std::fabs(contrib);
            ++terms;
        }
        if (fast_ok) {
            const long double low = L - gamma(static_cast<long double>(terms) + 2.0L) * absL;
            f64 v = static_cast<f64>(low);
            if (static_cast<long double>(v) > low)
                v = std::nextafter(v, -std::numeric_limits<f64>::infinity());
            if (std::isfinite(v)) {
                out.value = v;
                out.finite = true;
                return out;
            }
        }
    }
    // Only this final exact pass publishes a bound. The rounded arithmetic
    // above proposes multiplier corrections; it cannot establish a proof.
    std::vector<model::Rational> exact_d(d_out ? sz(n) : 0);
    model::DyadicSum exact_L;
    exact_L.add(sense * problem.obj_offset);
    for (core::Index i = 0; i < m; ++i) {
        if (y[sz(i)] == 0.0) continue;
        const f64 b = y[sz(i)] > 0.0 ? problem.row_lo[sz(i)] : problem.row_hi[sz(i)];
        if (!std::isfinite(b)) return out;
        exact_L.add_product(y[sz(i)], b);
    }
    ensure_columns();
    for (core::Index j = 0; j < n; ++j) {
        model::DyadicSum dj;
        dj.add(sense * problem.c[sz(j)]);
        for (core::Offset e = columns.ptr[sz(j)]; e < columns.ptr[sz(j) + 1]; ++e) {
            const f64 yi = y[sz(columns.row[static_cast<std::size_t>(e)])];
            if (yi != 0.0)
                dj.add_product(-yi, problem.A.vals[static_cast<std::size_t>(columns.entry[static_cast<std::size_t>(e)])]);
        }
        if (d_out) exact_d[sz(j)] = dj.value();
        const int sign = dj.sign();
        if (sign == 0) continue;
        f64 b = sign > 0 ? col_lo[sz(j)] : col_hi[sz(j)];
        if (!std::isfinite(b)) {
            ensure_implied();
            b = sign > 0 ? imp_lo[sz(j)] : imp_hi[sz(j)];
            ++out.implied_bound_uses;
        }
        if (!std::isfinite(b)) {
            // Same tolerance snap as the interval path: a dual-feasible
            // residual on an infinite side is charged as zero, not -inf.
            f64 scale = 1.0;
            if (std::isfinite(col_lo[sz(j)]))
                scale = std::max(scale, std::fabs(col_lo[sz(j)]));
            if (std::isfinite(col_hi[sz(j)]))
                scale = std::max(scale, std::fabs(col_hi[sz(j)]));
            const f64 tol = dual_feas_tol * (1.0 + scale);
            const f64 mag = std::fabs(dj.nearest());
            if (std::isfinite(mag) && mag <= tol) continue;
            return out;
        }
        exact_L.add_sum_product(dj, b);
    }
    out.value = exact_L.down();
    out.finite = std::isfinite(out.value);
    if (d_out != nullptr) *d_out = std::move(exact_d);
    return out;
}
}  // namespace

SafeLpBound safe_lagrangian_lower_bound(const model::LpProblem& problem,
                                        const std::vector<f64>& y_min,
                                        const std::vector<f64>& col_lo,
                                        const std::vector<f64>& col_hi,
                                        f64 dual_feas_tol) {
    return lagrangian_bound_impl(problem, y_min, col_lo, col_hi, nullptr,
                                 dual_feas_tol);
}

RcEnclosure enclose_reduced_costs_from_sums(const model::LpProblem& problem,
                                            const std::vector<f64>& y_min,
                                            const std::vector<long double>& aty,
                                            const std::vector<long double>& ab,
                                            const std::vector<std::uint32_t>& cnt,
                                            const std::vector<f64>& col_lo,
                                            const std::vector<f64>& col_hi,
                                            f64 dual_feas_tol) {
    core::WorkScope work_scope_(core::WorkCat::Exact);
    RcEnclosure out;
    const core::Index m = problem.n_rows(), n = problem.n_cols();
    if (y_min.size() != sz(m) || aty.size() != sz(n) || ab.size() != sz(n) ||
        cnt.size() != sz(n) || col_lo.size() != sz(n) || col_hi.size() != sz(n))
        return out;
    constexpr long double u = std::numeric_limits<long double>::epsilon() / 2.0L;
    const auto gamma = [&](long double k) { return (k * u) / (1.0L - k * u); };
    const auto down = [](long double v) {
        f64 f = static_cast<f64>(v);
        if (static_cast<long double>(f) > v) f = std::nextafter(f, -std::numeric_limits<f64>::infinity());
        return f;
    };
    const auto up = [](long double v) {
        f64 f = static_cast<f64>(v);
        if (static_cast<long double>(f) < v) f = std::nextafter(f, std::numeric_limits<f64>::infinity());
        return f;
    };
    const f64 sense = problem.maximize ? -1.0 : 1.0;
    long double L = static_cast<long double>(sense * problem.obj_offset);
    long double absL = std::fabs(L);
    std::uint64_t terms = 1;
    // A multiplier on a side its row does not have (round-off around 0)
    // is taken as 0: weak duality holds for any y. Its contribution, already
    // in the sums, then widens the reduced costs of that row's columns.
    thread_local std::vector<long double> widen;
    bool widened = false;
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < m; ++i) {
        const f64 yi = y_min[sz(i)];
        if (yi == 0.0) continue;
        if (!std::isfinite(yi)) return out;
        const f64 b = yi > 0.0 ? problem.row_lo[sz(i)] : problem.row_hi[sz(i)];
        if (!std::isfinite(b)) {
            if (!widened) { widen.assign(sz(n), 0.0L); widened = true; }
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                widen[sz(ci[sz(k)])] += std::fabs(static_cast<long double>(yi) *
                                                  problem.A.vals[sz(k)]) * (1.0L + 4.0L * u);
            continue;
        }
        const long double t = static_cast<long double>(yi) * b;
        L += t;
        absL += std::fabs(t);
        ++terms;
    }
    out.lo.resize(sz(n));
    out.hi.resize(sz(n));
    for (core::Index j = 0; j < n; ++j) {
        const long double c = static_cast<long double>(sense * problem.c[sz(j)]);
        const long double d = c - aty[sz(j)];
        const long double e = gamma(static_cast<long double>(cnt[sz(j)]) + 2.0L) *
                              (std::fabs(c) + ab[sz(j)]) +
                              (widened ? widen[sz(j)] * (1.0L + 4.0L * u) : 0.0L);
        long double lo_d = d - e, hi_d = d + e;
        const f64 l = col_lo[sz(j)], h = col_hi[sz(j)];
        snap_infinite_bound_reduced_cost(lo_d, hi_d, l, h, dual_feas_tol);
        out.lo[sz(j)] = down(lo_d);
        out.hi[sz(j)] = up(hi_d);
        if (lo_d == 0.0L && hi_d == 0.0L) continue;
        long double contrib;
        if (lo_d >= 0.0L) {
            if (!std::isfinite(l)) return out;
            contrib = l >= 0.0 ? lo_d * l : hi_d * l;
        } else if (hi_d <= 0.0L) {
            if (!std::isfinite(h)) return out;
            contrib = h >= 0.0 ? lo_d * h : hi_d * h;
        } else {
            if (!std::isfinite(l) || !std::isfinite(h)) return out;
            contrib = std::min(std::min(lo_d * l, lo_d * h), std::min(hi_d * l, hi_d * h));
        }
        L += contrib;
        absL += std::fabs(contrib);
        ++terms;
    }
    const f64 bound = down(L - gamma(static_cast<long double>(terms) + 2.0L) * absL);
    if (!std::isfinite(bound)) return out;
    out.bound = bound;
    out.finite = true;
    return out;
}

RcEnclosure enclose_reduced_costs(const model::LpProblem& problem,
                                  const std::vector<f64>& y_min,
                                  const std::vector<f64>& col_lo,
                                  const std::vector<f64>& col_hi) {
    const core::Index m = problem.n_rows(), n = problem.n_cols();
    if (y_min.size() != sz(m)) return {};
    // Per-thread scratch: called for every node LP without a carried enclosure.
    thread_local std::vector<long double> aty, ab;
    thread_local std::vector<std::uint32_t> cnt;
    aty.assign(sz(n), 0.0L);
    ab.assign(sz(n), 0.0L);
    cnt.assign(sz(n), 0);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    // Multipliers on sides their rows do not have are zeroed here, before
    // summing, so they cost no widening.
    thread_local std::vector<f64> y;
    y.assign(y_min.begin(), y_min.end());
    for (core::Index i = 0; i < m; ++i) {
        if (y[sz(i)] > 0.0 && !std::isfinite(problem.row_lo[sz(i)])) y[sz(i)] = 0.0;
        if (y[sz(i)] < 0.0 && !std::isfinite(problem.row_hi[sz(i)])) y[sz(i)] = 0.0;
    }
    for (core::Index i = 0; i < m; ++i) {
        const f64 yi = y[sz(i)];
        if (yi == 0.0) continue;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const auto j = sz(ci[sz(k)]);
            const long double t = static_cast<long double>(yi) * problem.A.vals[sz(k)];
            aty[j] += t;
            ab[j] += std::fabs(t);
            ++cnt[j];
        }
    }
    core::work_add(static_cast<std::uint64_t>(problem.A.vals.size()) +
                       static_cast<std::uint64_t>(n) + static_cast<std::uint64_t>(m),
                   core::WorkTag::Interval);
    return enclose_reduced_costs_from_sums(problem, y, aty, ab, cnt, col_lo, col_hi);
}

IntervalRcFixing reduced_cost_tightenings_from(const std::vector<f64>& rc_lo,
                                               const std::vector<f64>& rc_hi, f64 bound,
                                               const std::vector<f64>& col_lo,
                                               const std::vector<f64>& col_hi,
                                               f64 cutoff_min,
                                               const std::vector<char>& integer_mask,
                                               std::vector<unsigned char>* sides) {
    IntervalRcFixing out;
    const std::size_t n = col_lo.size();
    if (rc_lo.size() != n || rc_hi.size() != n || col_hi.size() != n ||
        integer_mask.size() < n || !std::isfinite(bound) || !std::isfinite(cutoff_min))
        return out;
    out.finite = true;
    out.bound = bound;
    if (sides != nullptr) {
        sides->assign(n, 0);
        for (std::size_t j = 0; j < n; ++j) {
            if (rc_lo[j] == 0.0 && rc_hi[j] == 0.0) continue;
            (*sides)[j] = rc_lo[j] >= 0.0 ? 1 : rc_hi[j] <= 0.0 ? 2 : 3;
        }
    }
    if (!(cutoff_min >= bound)) return out;
    constexpr long double u = std::numeric_limits<long double>::epsilon() / 2.0L;
    // room = cutoff - L rounded up; the quotient room / |d| rounded up.
    const long double room_ld = static_cast<long double>(cutoff_min) - static_cast<long double>(bound);
    const long double room = room_ld + std::fabs(room_ld) * u * 2.0L;
    const auto integral = [](f64 v) {
        return std::isfinite(v) && std::fabs(v) < 0x1p52 && v == std::trunc(v);
    };
    for (std::size_t j = 0; j < n; ++j) {
        if (!integer_mask[j]) continue;
        const f64 lo = col_lo[j], hi = col_hi[j];
        if (!(hi > lo)) continue;
        const bool up = rc_lo[j] > 0.0;
        const bool down = rc_hi[j] < 0.0;
        if (!up && !down) continue;
        const f64 base = up ? lo : hi;
        if (!integral(base)) continue;
        const long double mag = up ? static_cast<long double>(rc_lo[j])
                                   : -static_cast<long double>(rc_hi[j]);
        long double q = room / mag;
        q += std::fabs(q) * u * 2.0L;   // the quotient, rounded up
        if (!std::isfinite(static_cast<double>(q))) continue;
        const long double steps = std::floor(q);
        if (steps >= static_cast<long double>(hi) - static_cast<long double>(lo)) continue;
        const f64 nb = up ? static_cast<f64>(static_cast<long double>(base) + steps)
                          : static_cast<f64>(static_cast<long double>(base) - steps);
        if (!integral(nb)) continue;   // the sum must be exact to be the bound
        if (up ? !(nb < hi) : !(nb > lo)) continue;
        out.changes.push_back(RcBoundChange{static_cast<core::Index>(j), up, nb});
    }
    return out;
}

IntervalRcFixing interval_reduced_cost_tightenings(
    const model::LpProblem& problem, const std::vector<f64>& y_min,
    const std::vector<f64>& col_lo, const std::vector<f64>& col_hi,
    f64 cutoff_min, const std::vector<char>* integer_mask,
    std::vector<unsigned char>* sides) {
    const auto enc = enclose_reduced_costs(problem, y_min, col_lo, col_hi);
    if (!enc.finite) return {};
    std::vector<char> mask;
    if (integer_mask == nullptr) {
        mask.assign(col_lo.size(), 0);
        for (std::size_t j = 0; j < mask.size() && j < problem.is_integer.size(); ++j)
            mask[j] = problem.is_integer[j] ? 1 : 0;
        integer_mask = &mask;
    }
    return reduced_cost_tightenings_from(enc.lo, enc.hi, enc.bound, col_lo, col_hi, cutoff_min,
                                         *integer_mask, sides);
}

std::vector<RcBoundChange> safe_reduced_cost_tightenings(
    const model::LpProblem& problem, const std::vector<f64>& y_min,
    const std::vector<f64>& col_lo, const std::vector<f64>& col_hi,
    f64 cutoff_min, SafeLpBound* bound_out, std::uint64_t* candidates_out) {
    std::vector<RcBoundChange> out;
    if (candidates_out != nullptr) *candidates_out = 0;
    const core::Index n = problem.n_cols();
    std::vector<model::Rational> d;
    const SafeLpBound lb =
        lagrangian_bound_impl(problem, y_min, col_lo, col_hi, &d, 1e-7);
    if (bound_out != nullptr) *bound_out = lb;
    if (!lb.finite || !std::isfinite(cutoff_min) || d.size() != sz(n) ||
        problem.is_integer.size() != sz(n))
        return out;
    // Room between the certified bound and the cutoff. Every excluded region
    // below must raise the bound by MORE than this.
    const model::Rational room = model::Rational(cutoff_min) - model::Rational(lb.value);
    if (room < 0.0L) return out;   // the whole box is already cut off
    const auto integral = [](f64 v) {
        return std::isfinite(v) && std::fabs(v) < 0x1p52 && v == std::trunc(v);
    };
    for (core::Index j = 0; j < n; ++j) {
        if (!problem.is_integer[sz(j)]) continue;
        const auto& dj = d[sz(j)];
        const f64 lo = col_lo[sz(j)], hi = col_hi[sz(j)];
        if (!(hi > lo)) continue;
        // lb charged d_j to lo (d_j > 0) or hi (d_j < 0), both finite here.
        // Restricting x_j to [lo + t, hi] raises that one term by |d_j| t;
        // every other term can only rise, since a smaller box only tightens
        // the row-implied bounds charged to columns without declared ones.
        // So the region is cut off once |d_j| t > room; t is the smallest
        // integer step past room / |d_j|. Both the division and floor are
        // exact; convert the retained endpoint outward.
        const model::Rational ad = dj < 0 ? -dj : dj;
        if (!(ad > 0.0L)) continue;
        const bool up = dj > 0.0L;
        const f64 base = up ? lo : hi;
        if (!integral(base)) continue;
        if (candidates_out != nullptr) ++*candidates_out;
        const model::Rational steps = room / ad;
        if (std::isfinite(hi) && std::isfinite(lo) &&
            steps >= model::Rational(hi) - model::Rational(lo)) continue;
        const boost::multiprecision::cpp_int keep = numerator(steps) / denominator(steps);
        const model::Rational endpoint = up ? model::Rational(base) + model::Rational(keep)
                                           : model::Rational(base) - model::Rational(keep);
        const f64 nb = up ? model::rounded_up(endpoint) : model::rounded_down(endpoint);
        if (up ? !(nb < hi) : !(nb > lo)) continue;
        out.push_back(RcBoundChange{j, up, nb});
    }
    return out;
}


std::vector<FarkasBoundUse> farkas_conflict_bounds(
    const model::LpProblem& problem, const std::vector<f64>& multipliers,
    const std::vector<f64>& col_lo, const std::vector<f64>& col_hi,
    const std::vector<f64>& root_lo, const std::vector<f64>& root_hi,
    f64 tolerance) {
    std::vector<FarkasBoundUse> out;
    const core::Index m = problem.n_rows(), n = problem.n_cols();
    if (multipliers.size() != sz(m) || col_lo.size() != sz(n) ||
        root_lo.size() != sz(n))
        return out;
    // Proof: for all x in the box, y'Ax >= sum_j min(d_j x_j) =: lower, while
    // the rows force y'Ax <= sum_i max over row sides of y_i a_i =: upper.
    // lower > upper is the contradiction.
    std::vector<long double> d(sz(n), 0.0L);
    long double upper = 0.0L;
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < m; ++i) {
        const long double yi = multipliers[sz(i)];
        if (yi == 0.0L) continue;
        const f64 side = yi > 0.0L ? problem.row_hi[sz(i)] : problem.row_lo[sz(i)];
        if (!std::isfinite(side)) return out;
        upper += yi * side;
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            d[sz(ci[sz(k)])] += yi * problem.A.vals[sz(k)];
    }
    long double lower = 0.0L;
    struct Use { core::Index col; bool upper; f64 value; long double relax_cost; };
    std::vector<Use> uses;
    for (core::Index j = 0; j < n; ++j) {
        const auto& dj = d[sz(j)];
        if (dj == 0.0L) continue;
        const bool use_hi = dj < 0.0L;
        const f64 b = use_hi ? col_hi[sz(j)] : col_lo[sz(j)];
        if (!std::isfinite(b)) return out;
        lower += dj * b;
        const f64 rb = use_hi ? root_hi[sz(j)] : root_lo[sz(j)];
        if (b == rb) continue;               // root bound: not an assumption
        // Relaxing to the root bound lowers `lower` by |d_j| * |rb - b|
        // (infinite when the root bound is infinite: essential).
        const long double cost = std::isfinite(rb)
            ? std::fabs(dj) * std::fabs(static_cast<long double>(rb) - b)
            : std::numeric_limits<long double>::infinity();
        uses.push_back({j, use_hi, b, cost});
    }
    long double slack = lower - upper;
    const long double margin = tolerance * (1.0L + std::fabs(lower) + std::fabs(upper));
    if (!(slack > margin)) return out;       // ray does not certify here
    std::sort(uses.begin(), uses.end(),
              [](const Use& a, const Use& b) { return a.relax_cost < b.relax_cost; });
    for (const auto& u : uses) {
        if (slack - u.relax_cost > margin) {
            slack -= u.relax_cost;           // this bound is not needed
            continue;
        }
        out.push_back({u.col, u.upper, u.value});
    }
    return out;
}

}  // namespace sor::certify
