#include "sor/search/binquad.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

// Column-major adjacency: for each variable, the (other variable, coefficient)
// pairs it couples to. Built once from H's upper triangle and mirrored, since
// M is symmetric. This is the structure a flip walks.
struct Adjacency {
    std::vector<Index> start;   // n+1
    std::vector<Index> idx;
    std::vector<f64> val;
};

Adjacency build_adjacency(const io::QplibInstance& q, std::vector<f64>& lin) {
    const Index n = q.n;
    lin.assign(sz(n), 0.0);
    for (Index j = 0; j < n; ++j) lin[sz(j)] = q.g[sz(j)];

    std::vector<Index> count(sz(n), 0);
    for (std::size_t t = 0; t < q.h_val.size(); ++t) {
        const Index r = q.h_row[t] - 1, c = q.h_col[t] - 1;
        if (r == c) {
            // x_i^2 == x_i for binary x, so the diagonal is linear. The 0.5
            // is the objective's own 0.5 x'Hx convention.
            lin[sz(r)] += 0.5 * q.h_val[t];
        } else {
            ++count[sz(r)];
            ++count[sz(c)];
        }
    }

    Adjacency adj;
    adj.start.assign(sz(n) + 1, 0);
    for (Index j = 0; j < n; ++j) adj.start[sz(j) + 1] = adj.start[sz(j)] + count[sz(j)];
    adj.idx.resize(sz(adj.start[sz(n)]));
    adj.val.resize(adj.idx.size());

    std::vector<Index> cursor(adj.start.begin(), adj.start.end() - 1);
    for (std::size_t t = 0; t < q.h_val.size(); ++t) {
        const Index r = q.h_row[t] - 1, c = q.h_col[t] - 1;
        if (r == c) continue;
        // QPLIB's objective is 0.5 x'Hx with H stored as an upper triangle
        // that is used AS-IS -- the stored entries are NOT mirrored into the
        // lower triangle. So the coefficient of x_i x_j in the objective is
        // 0.5*H_ij, not H_ij. Verified against QPLIB's own shipped solution
        // point on QPLIB_3834: mirroring gives 6962.459179 at the optimum,
        // the as-stored halving gives 3760.715066, which is QPLIB's published
        // value to every printed digit.
        const f64 v = 0.5 * q.h_val[t];
        adj.idx[sz(cursor[sz(r)])] = c;
        adj.val[sz(cursor[sz(r)])] = v;
        ++cursor[sz(r)];
        adj.idx[sz(cursor[sz(c)])] = r;
        adj.val[sz(cursor[sz(c)])] = v;
        ++cursor[sz(c)];
    }
    return adj;
}

// Column-major A, for the same reason: a flip touches one column.
struct ColMajorA {
    std::vector<Index> start;  // n+1
    std::vector<Index> row;
    std::vector<f64> val;
};

ColMajorA build_col_major_a(const io::QplibInstance& q) {
    ColMajorA a;
    a.start.assign(sz(q.n) + 1, 0);
    if (q.a_val.empty()) return a;

    std::vector<Index> count(sz(q.n), 0);
    for (std::size_t t = 0; t < q.a_val.size(); ++t) ++count[sz(q.a_col[t] - 1)];
    for (Index j = 0; j < q.n; ++j) a.start[sz(j) + 1] = a.start[sz(j)] + count[sz(j)];
    a.row.resize(sz(a.start[sz(q.n)]));
    a.val.resize(a.row.size());

    std::vector<Index> cursor(a.start.begin(), a.start.end() - 1);
    for (std::size_t t = 0; t < q.a_val.size(); ++t) {
        const Index c = q.a_col[t] - 1;
        a.row[sz(cursor[sz(c)])] = q.a_row[t] - 1;
        a.val[sz(cursor[sz(c)])] = q.a_val[t];
        ++cursor[sz(c)];
    }
    return a;
}

// True when a row is "sum of a subset of x == k": every coefficient 1 and an
// equality bound. A single flip changes such a row by exactly +/-1, so a
// flip-only neighbourhood can never move between feasible points -- it thrashes
// on the boundary. Detecting this is what earns the swap neighbourhood below.
bool is_cardinality_row(const io::QplibInstance& q, Index row,
                        const std::vector<Index>& row_start,
                        const std::vector<f64>& row_val) {
    if (q.c_lo[static_cast<std::size_t>(row)] != q.c_hi[static_cast<std::size_t>(row)]) return false;
    if (!std::isfinite(q.c_lo[static_cast<std::size_t>(row)])) return false;
    for (Index k = row_start[static_cast<std::size_t>(row)];
         k < row_start[static_cast<std::size_t>(row) + 1]; ++k) {
        if (row_val[static_cast<std::size_t>(k)] != 1.0) return false;
    }
    return true;
}

inline f64 row_violation(f64 activity, f64 lo, f64 hi) {
    if (activity < lo) return lo - activity;
    if (activity > hi) return activity - hi;
    return 0.0;
}

}  // namespace

BinQuadResult solve_binquad(const io::QplibInstance& q,
                            const BinQuadOptions& opts,
                            BinQuadDiagnostics& diag) {
    const auto t0 = Clock::now();
    for (auto t : q.var_type) {
        if (t != io::QplibVarType::Binary) {
            throw std::invalid_argument(
                "solve_binquad: every variable must be binary");
        }
    }
    // The search scores rows through A only.  A quadratic constraint would be
    // invisible to it -- the incumbent could violate it and still be called
    // feasible -- so refuse rather than drop it (QBQ/QBC/QBD instances).
    if (q.has_quadratic_constraints())
        throw std::invalid_argument("solve_binquad: quadratic constraints are not supported");

    const Index n = q.n, m = q.m;
    std::vector<f64> lin;
    const Adjacency adj = build_adjacency(q, lin);
    const ColMajorA A = build_col_major_a(q);

    // Work internally as a MINIMISATION. A maximise instance is negated on the
    // way in and the reported objective is un-negated on the way out, so the
    // search logic below never has to carry the sense.
    const f64 sense = q.maximize ? -1.0 : 1.0;
    if (q.maximize) {
        for (f64& v : lin) v = -v;
    }
    std::vector<f64> adjval = adj.val;
    if (q.maximize) {
        for (f64& v : adjval) v = -v;
    }

    // Row-major A, only to classify rows (the search itself uses columns).
    bool has_cardinality = false;
    {
        std::vector<Index> rs(sz(m) + 1, 0), rcount(sz(m), 0);
        for (std::size_t t = 0; t < q.a_val.size(); ++t) ++rcount[sz(q.a_row[t] - 1)];
        for (Index i = 0; i < m; ++i) rs[sz(i) + 1] = rs[sz(i)] + rcount[sz(i)];
        std::vector<f64> rv(q.a_val.size(), 0.0);
        std::vector<Index> cur(rs.begin(), rs.end() - 1);
        for (std::size_t t = 0; t < q.a_val.size(); ++t) {
            const Index i = q.a_row[t] - 1;
            rv[sz(cur[sz(i)])] = q.a_val[t];
            ++cur[sz(i)];
        }
        for (Index i = 0; i < m && !has_cardinality; ++i)
            has_cardinality = is_cardinality_row(q, i, rs, rv);
    }

    std::mt19937_64 rng(opts.seed);
    std::vector<std::uint8_t> x(sz(n), 0);
    for (Index j = 0; j < n; ++j) x[sz(j)] = static_cast<std::uint8_t>(rng() & 1ULL);

    // h = M x, and r = A x, both maintained incrementally from here on.
    std::vector<f64> h(sz(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        if (!x[sz(j)]) continue;
        for (Index k = adj.start[sz(j)]; k < adj.start[sz(j) + 1]; ++k)
            h[sz(adj.idx[sz(k)])] += adjval[sz(k)];
    }
    std::vector<f64> r(sz(m), 0.0);
    for (Index j = 0; j < n; ++j) {
        if (!x[sz(j)]) continue;
        for (Index k = A.start[sz(j)]; k < A.start[sz(j) + 1]; ++k)
            r[sz(A.row[sz(k)])] += A.val[sz(k)];
    }
    // row_viol[i] == row_violation(r[i], lo, hi), kept in sync by apply_flip
    // (the sole mutator of r) below. The gain sweep calls violation_delta for
    // every candidate column, and every one of those calls used to recompute
    // this SAME "current violation of row i" from scratch -- it does not
    // depend on which candidate column is being evaluated, only on the row
    // and the current r. Caching it turns that from an O(nnz(A)) recompute
    // per sweep into an O(nnz of the one flipped column) update per flip.
    // Profiled 2026-09-24 (callgrind, QPLIB_3307): row_violation's two calls
    // inside violation_delta -- one of them this now-redundant "before" --
    // were together the single largest cost in the engine, ~35% of
    // instructions self+inclusive.
    std::vector<f64> row_viol(sz(m));
    for (Index i = 0; i < m; ++i)
        row_viol[sz(i)] = row_violation(r[sz(i)], q.c_lo[sz(i)], q.c_hi[sz(i)]);

    auto objective_of = [&](const std::vector<std::uint8_t>& v) {
        f64 o = 0.0;
        for (Index j = 0; j < n; ++j) {
            if (!v[sz(j)]) continue;
            o += lin[sz(j)];
            for (Index k = adj.start[sz(j)]; k < adj.start[sz(j) + 1]; ++k)
                if (v[sz(adj.idx[sz(k)])]) o += 0.5 * adjval[sz(k)];
        }
        return o;
    };
    auto total_violation = [&](const std::vector<f64>& act) {
        f64 s = 0.0;
        for (Index i = 0; i < m; ++i)
            s += row_violation(act[sz(i)], q.c_lo[sz(i)], q.c_hi[sz(i)]);
        return s;
    };
    auto max_violation = [&](const std::vector<f64>& act) {
        f64 s = 0.0;
        for (Index i = 0; i < m; ++i)
            s = std::max(s, row_violation(act[sz(i)], q.c_lo[sz(i)], q.c_hi[sz(i)]));
        return s;
    };

    f64 obj = objective_of(x);
    f64 viol = total_violation(r);

    // Penalty scaled to the objective's own magnitude, so it means the same
    // thing on an instance with coefficients of 1e5 as on one with 1e0.
    f64 scale = 0.0;
    for (f64 v : lin) scale = std::max(scale, std::fabs(v));
    for (f64 v : adjval) scale = std::max(scale, std::fabs(v));
    if (scale <= 0.0) scale = 1.0;
    f64 penalty = opts.penalty0 * scale;

    std::vector<std::uint8_t> best_x = x;
    f64 best_obj = viol <= 0.0 ? obj : std::numeric_limits<f64>::infinity();
    f64 best_viol = viol;
    bool have_feasible = viol <= 0.0;

    std::vector<std::uint64_t> tabu_until(sz(n), 0);
    std::uniform_int_distribution<Index> tenure_dist(
        opts.tenure_min, opts.tenure_min + std::max<Index>(1, opts.tenure_span) - 1);
    std::uniform_int_distribution<Index> var_dist(0, n > 0 ? n - 1 : 0);

    // The change in total violation if variable j flips. O(nnz of column j).
    // "before" is row_viol[i] -- the row's CURRENT violation, already known,
    // not recomputed. Only "after" (the hypothetical post-flip value) needs
    // an actual row_violation call, since that state is never materialised
    // unless this candidate is the one taken.
    auto violation_delta = [&](Index j, f64 delta) {
        f64 dv = 0.0;
        for (Index k = A.start[sz(j)]; k < A.start[sz(j) + 1]; ++k) {
            const Index i = A.row[sz(k)];
            const f64 after = row_violation(r[sz(i)] + delta * A.val[sz(k)],
                                            q.c_lo[sz(i)], q.c_hi[sz(i)]);
            dv += after - row_viol[sz(i)];
        }
        return dv;
    };

    auto apply_flip = [&](Index j) {
        const f64 delta = x[sz(j)] ? -1.0 : 1.0;
        for (Index k = adj.start[sz(j)]; k < adj.start[sz(j) + 1]; ++k)
            h[sz(adj.idx[sz(k)])] += delta * adjval[sz(k)];
        // r is mutated ONLY here, so this is also the only place row_viol
        // needs to be refreshed to stay in sync with it.
        for (Index k = A.start[sz(j)]; k < A.start[sz(j) + 1]; ++k) {
            const Index i = A.row[sz(k)];
            r[sz(i)] += delta * A.val[sz(k)];
            row_viol[sz(i)] = row_violation(r[sz(i)], q.c_lo[sz(i)], q.c_hi[sz(i)]);
        }
        x[sz(j)] ^= 1U;
        ++diag.flips;
    };

    // Scratch for the swap neighbourhood below, hoisted out of the iteration
    // loop. These used to be declared fresh (and, for outs/ins, reserved to
    // n; for coupling, allocated AND zero-filled) on every single iteration
    // that took the cardinality-swap path -- i.e. a malloc/free and an O(n)
    // fill per iteration on top of the O(n) work the block already does.
    // coupling's own reset loop (below) always leaves it all-zero again
    // before the block exits, so zero-filling it here once is sufficient;
    // outs/ins only need their size reset, not their capacity.
    std::vector<std::pair<f64, Index>> outs, ins;
    outs.reserve(sz(n)); ins.reserve(sz(n));
    std::vector<f64> coupling(sz(n), 0.0);

    std::uint64_t since_improve = 0;
    for (diag.iterations = 0; diag.iterations < opts.max_iterations; ++diag.iterations) {
        if (opts.time_limit_s > 0.0 && (diag.iterations & 0xFFU) == 0 &&
            ms_since(t0) > opts.time_limit_s * 1000.0) {
            break;
        }

        // --- gain sweep: O(1) per candidate, exactly the GPU gain kernel ---
        Index best_j = -1;
        f64 best_score = std::numeric_limits<f64>::infinity();
        for (Index j = 0; j < n; ++j) {
            const f64 delta = x[sz(j)] ? -1.0 : 1.0;
            const f64 d_obj = delta * (lin[sz(j)] + h[sz(j)]);
            const f64 d_viol = m > 0 ? violation_delta(j, delta) : 0.0;
            const f64 score = d_obj + penalty * d_viol;

            const bool is_tabu = tabu_until[sz(j)] > diag.iterations;
            // Aspiration: a tabu move is allowed when it would produce a
            // feasible point better than anything seen so far.
            const bool aspires = is_tabu && (viol + d_viol) <= 0.0 &&
                                 (obj + d_obj) < best_obj - 1e-12;
            if (is_tabu && !aspires) continue;
            if (score < best_score) {
                best_score = score;
                best_j = j;
            }
        }
        if (best_j < 0) {  // everything tabu; kick one at random
            best_j = var_dist(rng);
        }

        // --- swap neighbourhood, for equality-cardinality instances --------
        // A swap (i out, j in) leaves every all-ones equality row unchanged,
        // so it moves between feasible points instead of across the boundary.
        // Exact gain: -(L_i + h_i) + (L_j + h_j) - M_ij, where the last term
        // corrects for the coupling the two single-flip gains double-count.
        // Restricted to the best few candidates on each side, which keeps the
        // cost at K*(nnz + K) and is the same shape a GPU would evaluate.
        if (has_cardinality && viol <= 1e-9) {
            constexpr Index kCand = 24;
            outs.clear(); ins.clear();
            for (Index j = 0; j < n; ++j) {
                if (tabu_until[sz(j)] > diag.iterations) continue;
                const f64 gain_remove = -(lin[sz(j)] + h[sz(j)]);
                const f64 gain_insert = (lin[sz(j)] + h[sz(j)]);
                if (x[sz(j)]) outs.emplace_back(gain_remove, j);
                else ins.emplace_back(gain_insert, j);
            }
            const Index n_out = std::min<Index>(kCand, static_cast<Index>(outs.size()));
            const Index n_in = std::min<Index>(kCand, static_cast<Index>(ins.size()));
            if (n_out > 0 && n_in > 0) {
                // Only the SET of the n_out/n_in best candidates matters
                // below: the (a, b) loop scans the full cross product
                // regardless of order, nothing downstream indexes outs/ins
                // expecting sorted position. So this needs a SELECTION, not
                // a SORT -- nth_element instead of partial_sort avoids
                // paying to order the kept elements against each other, and
                // is skipped entirely when there is nothing to select down
                // to (n_out == outs.size(), the common case for "outs": a
                // cardinality row keeps few variables at 1, usually fewer
                // than kCand). Profiled 2026-09-24 (callgrind, QPLIB_3834,
                // n=50, cardinality=10): this pair of partial_sort calls,
                // run on every iteration once the search is feasible, was
                // ~32% of the engine's instructions -- the "outs" side never
                // truncates on this instance (10 <= kCand), so half of that
                // was sorting an array not being cut down at all.
                if (n_out < static_cast<Index>(outs.size()))
                    std::nth_element(outs.begin(), outs.begin() + (n_out - 1), outs.end());
                if (n_in < static_cast<Index>(ins.size()))
                    std::nth_element(ins.begin(), ins.begin() + (n_in - 1), ins.end());

                // Take the BEST available swap even when it worsens the
                // objective -- that is what makes this a tabu search rather
                // than a descent, and the tabu tenure is what stops it
                // cycling straight back. Restricting to improving swaps
                // stalls at the first swap-local optimum (measured: QPLIB_3834
                // frozen at 6962.46 across the whole budget).
                f64 best_swap = std::numeric_limits<f64>::infinity();
                Index si = -1, sj = -1;
                for (Index a = 0; a < n_out; ++a) {
                    const Index i = outs[sz(a)].second;
                    for (Index k = adj.start[sz(i)]; k < adj.start[sz(i) + 1]; ++k)
                        coupling[sz(adj.idx[sz(k)])] = adjval[sz(k)];
                    for (Index b = 0; b < n_in; ++b) {
                        const Index j = ins[sz(b)].second;
                        const f64 d = outs[sz(a)].first + ins[sz(b)].first - coupling[sz(j)];
                        if (d < best_swap - 1e-12) { best_swap = d; si = i; sj = j; }
                    }
                    for (Index k = adj.start[sz(i)]; k < adj.start[sz(i) + 1]; ++k)
                        coupling[sz(adj.idx[sz(k)])] = 0.0;
                }
                if (si >= 0) {
                    obj += -(lin[sz(si)] + h[sz(si)]);
                    apply_flip(si);
                    obj += (lin[sz(sj)] + h[sz(sj)]);
                    apply_flip(sj);
                    viol = total_violation(r);
                    tabu_until[sz(si)] = diag.iterations + static_cast<std::uint64_t>(tenure_dist(rng));
                    tabu_until[sz(sj)] = diag.iterations + static_cast<std::uint64_t>(tenure_dist(rng));
                    if (viol <= 1e-9 && obj < best_obj - 1e-12) {
                        best_obj = obj; best_x = x; best_viol = 0.0;
                        have_feasible = true; since_improve = 0; ++diag.improvements;
                    } else {
                        ++since_improve;
                    }
                    continue;  // swap taken; skip the single flip this iteration
                }
            }
        }

        const f64 delta = x[sz(best_j)] ? -1.0 : 1.0;
        obj += delta * (lin[sz(best_j)] + h[sz(best_j)]);
        viol += m > 0 ? violation_delta(best_j, delta) : 0.0;
        apply_flip(best_j);
        tabu_until[sz(best_j)] = diag.iterations + static_cast<std::uint64_t>(tenure_dist(rng));

        if (viol <= 1e-9 && obj < best_obj - 1e-12) {
            best_obj = obj;
            best_x = x;
            best_viol = 0.0;
            have_feasible = true;
            since_improve = 0;
            ++diag.improvements;
        } else if (!have_feasible && viol < best_viol) {
            best_viol = viol;
            best_x = x;
            since_improve = 0;
        } else {
            ++since_improve;
        }

        // Adapt the penalty: push harder while infeasible, relax when not.
        if (viol > 1e-9) penalty *= 1.0005;
        else penalty = std::max(penalty * 0.999, 1e-6 * scale);

        if (since_improve >= opts.stagnation_limit) {
            const Index kicks = std::max<Index>(1, n / 10);
            for (Index t = 0; t < kicks; ++t) {
                const Index j = var_dist(rng);
                const f64 d = x[sz(j)] ? -1.0 : 1.0;
                obj += d * (lin[sz(j)] + h[sz(j)]);
                viol += m > 0 ? violation_delta(j, d) : 0.0;
                apply_flip(j);
            }
            std::fill(tabu_until.begin(), tabu_until.end(), 0);
            since_improve = 0;
            ++diag.restarts;
        }
    }

    // Recompute from the stored point rather than trusting the incremental
    // accumulators -- floating-point drift over 200k updates is real, and the
    // reported number has to be the one an independent checker would get.
    std::vector<f64> r_best(sz(m), 0.0);
    for (Index j = 0; j < n; ++j) {
        if (!best_x[sz(j)]) continue;
        for (Index k = A.start[sz(j)]; k < A.start[sz(j) + 1]; ++k)
            r_best[sz(A.row[sz(k)])] += A.val[sz(k)];
    }
    const f64 final_obj_min = objective_of(best_x);
    const f64 final_maxviol = max_violation(r_best);

    BinQuadResult out;
    out.x = best_x;
    out.objective = sense * final_obj_min + q.f_const;
    out.violation = final_maxviol;
    out.feasible = final_maxviol <= 1e-6;

    diag.found_feasible = out.feasible;
    diag.best_objective = out.objective;
    diag.best_violation = final_maxviol;
    diag.total_ms = ms_since(t0);
    if (opts.verbose) {
        std::fprintf(stderr,
                     "binquad: iters=%llu flips=%llu restarts=%llu obj=%.10g viol=%.3e %.1fms\n",
                     (unsigned long long)diag.iterations, (unsigned long long)diag.flips,
                     (unsigned long long)diag.restarts, out.objective, final_maxviol,
                     diag.total_ms);
    }
    return out;
}

backend::BqData binquad_device_data(const io::QplibInstance& q, f64& scale) {
    std::vector<f64> lin;
    const Adjacency adj = build_adjacency(q, lin);
    const ColMajorA A = build_col_major_a(q);
    const f64 sense = q.maximize ? -1.0 : 1.0;

    backend::BqData d;
    d.n = q.n;
    d.m = q.m;
    for (f64 v : lin) d.lin.push_back(sense * v);
    for (Index v : adj.start) d.m_start.push_back(v);
    for (Index v : adj.idx) d.m_idx.push_back(v);
    for (f64 v : adj.val) d.m_val.push_back(sense * v);
    d.a_cstart.assign(A.start.begin(), A.start.end());   // always n+1 entries
    d.a_crow.assign(A.row.begin(), A.row.end());
    d.a_cval = A.val;

    d.a_rstart.assign(sz(q.m) + 1, 0);
    for (std::size_t t = 0; t < q.a_val.size(); ++t) ++d.a_rstart[sz(q.a_row[t])];
    for (Index i = 0; i < q.m; ++i) d.a_rstart[sz(i) + 1] += d.a_rstart[sz(i)];
    d.a_rcol.resize(q.a_val.size());
    d.a_rval.resize(q.a_val.size());
    std::vector<std::int32_t> cur(d.a_rstart.begin(), d.a_rstart.end() - 1);
    for (std::size_t t = 0; t < q.a_val.size(); ++t) {
        const auto i = sz(q.a_row[t] - 1);
        d.a_rcol[sz(cur[i])] = q.a_col[t] - 1;
        d.a_rval[sz(cur[i])] = q.a_val[t];
        ++cur[i];
    }
    d.c_lo = q.c_lo;
    d.c_hi = q.c_hi;

    scale = 0.0;
    for (f64 v : d.lin) scale = std::max(scale, std::fabs(v));
    for (f64 v : d.m_val) scale = std::max(scale, std::fabs(v));
    if (scale <= 0.0) scale = 1.0;
    return d;
}

BinQuadResult solve_binquad_parallel(const io::QplibInstance& q,
                                     const BinQuadParallelOptions& opts,
                                     backend::BinQuadDevice& dev,
                                     BinQuadDiagnostics& diag) {
    const auto t0 = Clock::now();
    diag = BinQuadDiagnostics{};
    for (auto t : q.var_type)
        if (t != io::QplibVarType::Binary)
            throw std::invalid_argument("solve_binquad_parallel: every variable must be binary");
    if (q.has_quadratic_constraints())   // same reason as solve_binquad
        throw std::invalid_argument(
            "solve_binquad_parallel: quadratic constraints are not supported");
    if (opts.elite_size == 0 || opts.elite_size > 16 || opts.searches == 0)
        throw std::invalid_argument("solve_binquad_parallel: need 1..16 elites, >= 1 search");

    f64 scale = 1.0;
    const backend::BqData data = binquad_device_data(q, scale);
    backend::BqParams bp;
    bp.searches = opts.searches;
    bp.tenure_min = opts.tenure_min;
    bp.tenure_span = opts.tenure_span;
    bp.stagnation_limit = opts.stagnation_limit;
    bp.penalty0 = opts.penalty0 * scale;
    bp.penalty_floor = 1e-6 * scale;
    bp.seed = opts.seed;
    dev.upload(data, bp);

    const std::uint32_t P = opts.searches;
    std::uint32_t epoch = 0;
    dev.restart(std::vector<std::uint8_t>(P, 1), std::vector<std::int32_t>(P, -1), 0.5,
                epoch);

    // Elite pool, best first, distinct points only; min-sense objectives.
    struct Elite { f64 obj; std::vector<std::uint8_t> x; };
    std::vector<Elite> elite;
    std::vector<std::uint8_t> least_viol_x;
    f64 least_viol = std::numeric_limits<f64>::infinity();
    std::mt19937 rng(opts.seed);
    std::vector<backend::BqSearch> ss;
    std::vector<std::uint8_t> xs;

    const auto time_up = [&] {
        return opts.time_limit_s > 0.0 && ms_since(t0) > opts.time_limit_s * 1000.0;
    };
    while (epoch < opts.max_epochs && !time_up()) {
        dev.run(opts.epoch_iterations);
        diag.iterations += static_cast<std::uint64_t>(opts.epoch_iterations) * P;
        ++epoch;
        dev.read_searches(ss);

        bool pool_changed = false;
        for (std::uint32_t s = 0; s < P; ++s) {
            if (ss[s].have_feasible) {
                const bool room = elite.size() < opts.elite_size;
                if (!room && !(ss[s].best_obj < elite.back().obj - 1e-12)) continue;
                dev.read_best(s, xs);
                bool dup = false;
                for (const auto& e : elite) dup = dup || e.x == xs;
                if (dup) continue;
                elite.push_back({ss[s].best_obj, xs});
                std::sort(elite.begin(), elite.end(),
                          [](const Elite& a, const Elite& b) { return a.obj < b.obj; });
                if (elite.size() > opts.elite_size) elite.pop_back();
                pool_changed = true;
                ++diag.improvements;
            } else if (elite.empty() && ss[s].best_viol < least_viol) {
                least_viol = ss[s].best_viol;
                dev.read_best(s, least_viol_x);
            }
        }
        if (pool_changed) {
            std::vector<std::vector<std::uint8_t>> pool;
            for (const auto& e : elite) pool.push_back(e.x);
            dev.set_elite(pool);
        }

        std::vector<std::uint8_t> restart(P, 0);
        std::vector<std::int32_t> base(P, -1);
        bool any = false;
        for (std::uint32_t s = 0; s < P; ++s) {
            if (!ss[s].stalled) continue;
            restart[s] = 1;
            any = true;
            if (!elite.empty())
                base[s] = static_cast<std::int32_t>(rng() % elite.size());
        }
        if (any) {
            // No elite yet: restart from scratch; otherwise perturb an elite.
            dev.restart(restart, base, elite.empty() ? 0.5 : opts.restart_flip, epoch);
            diag.restarts += static_cast<std::uint64_t>(
                std::count(restart.begin(), restart.end(), std::uint8_t{1}));
        }
    }

    // Re-score from the raw instance; the device's running sums are not trusted.
    const std::vector<std::uint8_t> best_x =
        !elite.empty() ? elite.front().x
                       : (!least_viol_x.empty() ? least_viol_x
                                                : std::vector<std::uint8_t>(sz(q.n), 0));
    f64 o = q.f_const;
    for (std::size_t t = 0; t < q.h_val.size(); ++t)
        o += 0.5 * q.h_val[t] * best_x[sz(q.h_row[t] - 1)] * best_x[sz(q.h_col[t] - 1)];
    for (Index j = 0; j < q.n; ++j) o += q.g[sz(j)] * best_x[sz(j)];
    std::vector<f64> ax(sz(q.m), 0.0);
    for (std::size_t t = 0; t < q.a_val.size(); ++t)
        ax[sz(q.a_row[t] - 1)] += q.a_val[t] * best_x[sz(q.a_col[t] - 1)];
    f64 viol = 0.0;
    for (Index i = 0; i < q.m; ++i)
        viol = std::max(viol, row_violation(ax[sz(i)], q.c_lo[sz(i)], q.c_hi[sz(i)]));

    BinQuadResult out;
    out.x = best_x;
    out.objective = o;
    out.violation = viol;
    out.feasible = viol <= 1e-6;
    diag.found_feasible = out.feasible;
    diag.best_objective = o;
    diag.best_violation = viol;
    diag.total_ms = ms_since(t0);
    return out;
}

}  // namespace sor::search
