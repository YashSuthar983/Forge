// SOR — binary-QP branch-and-bound over batched QCR relaxations; see
// bqp_bab.hpp for why every prune is sound.
#include "sor/search/bqp_bab.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <limits>
#include <queue>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = std::numeric_limits<f64>::infinity();

// -1 free, 0 or 1 fixed.
struct Node {
    f64 bound;                       // min form, certified
    std::vector<std::int8_t> fix;
};
struct Worse {
    bool operator()(const Node& a, const Node& b) const { return a.bound > b.bound; }
};

// Raw-instance score of a binary point: objective (ORIGINAL sense) and the
// max row violation.  Shares no code with the relaxation.
void score(const io::QplibInstance& q, const std::vector<std::uint8_t>& x, f64& obj,
           f64& viol) {
    obj = q.f_const;
    for (std::size_t t = 0; t < q.h_val.size(); ++t)
        obj += 0.5 * q.h_val[t] * x[sz(q.h_row[t] - 1)] * x[sz(q.h_col[t] - 1)];
    for (Index j = 0; j < q.n; ++j) obj += q.g[sz(j)] * x[sz(j)];
    std::vector<f64> ax(sz(q.m), 0.0);
    for (std::size_t t = 0; t < q.a_val.size(); ++t)
        ax[sz(q.a_row[t] - 1)] += q.a_val[t] * x[sz(q.a_col[t] - 1)];
    viol = 0.0;
    for (Index i = 0; i < q.m; ++i) {
        const f64 a = ax[sz(i)];
        if (q.c_lo[sz(i)] > -q.inf_bound && a < q.c_lo[sz(i)]) viol = std::max(viol, q.c_lo[sz(i)] - a);
        if (q.c_hi[sz(i)] < q.inf_bound && a > q.c_hi[sz(i)]) viol = std::max(viol, a - q.c_hi[sz(i)]);
    }
}

}  // namespace

BqpBabResult solve_bqp_bab(const io::QplibInstance& q, const BinQuadResult* warm,
                           const BqpBabOptions& opts, backend::BatchedPdhcgDevice& dev) {
    const auto t0 = Clock::now();
    BqpBabResult res;
    QcrDiagnostics& qd = res.qcr;
    std::vector<QcrCandidate> cands;
    bool negated = false;
    if (!qcr_candidates(q, opts.qcr, cands, negated, qd.reason)) {
        res.reason = qd.reason;
        return res;
    }
    const f64 sense = negated ? -1.0 : 1.0;   // min form = sense * original
    const Index n = q.n;
    constexpr f64 kFeasTol = 1e-6;

    // Incumbent, min form.
    f64 U = kInf;
    auto offer = [&](const std::vector<std::uint8_t>& x) {
        f64 obj = 0.0, viol = 0.0;
        score(q, x, obj, viol);
        if (viol <= kFeasTol && sense * obj < U) {
            U = sense * obj;
            res.x = x;
            res.incumbent = obj;
            res.have_incumbent = true;
        }
    };
    if (warm && warm->x.size() == sz(n)) offer(warm->x);

    // A node pruned within the tolerance may still have a bound a little
    // BELOW the incumbent; the reported global bound must include it, or the
    // bound would claim more than was proved.
    f64 pruned_floor = kInf;
    const auto prunable = [&](f64 bound) {
        const bool p = bound >= U - opts.gap_tol * std::max(1.0, std::fabs(U));
        if (p) pruned_floor = std::min(pruned_floor, bound);
        return p;
    };
    // The deadline the CURRENT phase must respect.  The root phase gets a
    // slice of the run so that a root solved with a huge iteration budget
    // cannot swallow the whole tree; everything after it gets the rest.
    double phase_deadline_s = opts.time_limit_s;
    const auto out_of_time = [&] {
        return phase_deadline_s > 0.0 && ms_since(t0) > phase_deadline_s * 1000.0;
    };

    // Bound a list of variable fixings against one relaxation.  The device
    // takes dev.lanes() problems at a time (0 = any width), so the list is
    // chunked; callers may hand over as many as they like.  Each result is a
    // certified bound in min form (-inf when the solve gave nothing usable)
    // plus the relaxation point, which branching reads.
    const bool use_ipm = opts.node_solver == BqpNodeSolver::Ipm ||
                         (opts.node_solver == BqpNodeSolver::Auto && !dev.is_accelerated());
    res.node_solver = use_ipm ? "ipm" : "pdhcg";
    auto bound_nodes = [&](const engines::QpProblem& rel, const QcrDiagnostics& d,
                           const std::vector<const std::vector<std::int8_t>*>& fixes,
                           std::uint64_t max_iterations,
                           std::vector<f64>& bounds,
                           std::vector<std::vector<f64>>& xs) {
        bounds.assign(fixes.size(), -kInf);
        xs.assign(fixes.size(), std::vector<f64>{});
        if (use_ipm) {
            // One host solve per fixing.  Q + 2 diag(u) is already certified
            // PSD by the candidate (d.psd_slack is that certificate, and
            // wolfe_bound charges for it), so the engine's own certificate is
            // skipped: it proves nothing the bound relies on.
            engines::QpProblem node = rel;
            for (std::size_t t = 0; t < fixes.size(); ++t) {
                if (t != 0 && out_of_time()) break;
                const auto& f = *fixes[t];
                node.linear.col_lo = rel.linear.col_lo;
                node.linear.col_hi = rel.linear.col_hi;
                for (Index j = 0; j < n; ++j)
                    if (f[sz(j)] >= 0) node.linear.col_lo[sz(j)] = node.linear.col_hi[sz(j)] = f[sz(j)];
                engines::QpOptions qp = opts.qcr.qp;
                qp.assume_psd = true;
                qp.max_iterations = 300;      // the IPM's own cap; iterations here are Newton steps
                if (phase_deadline_s > 0.0) {
                    const double left = phase_deadline_s - ms_since(t0) / 1000.0;
                    qp.time_limit_s = qp.time_limit_s > 0.0 ? std::min(qp.time_limit_s, left) : left;
                    qp.time_limit_s = std::max(qp.time_limit_s, 1e-3);
                }
                engines::QpDiagnostics diag;
                const double tb = ms_since(t0);
                const auto r = engines::solve_qp_ipm(node, qp, diag);
                ++res.batches;
                if (opts.verbose)
                    std::fprintf(stderr, "bab: ipm node %llu iters %llu %.0f ms (t=%.1fs) %s\n",
                                 static_cast<unsigned long long>(res.batches),
                                 static_cast<unsigned long long>(diag.iterations),
                                 ms_since(t0) - tb, ms_since(t0) / 1000.0,
                                 diag.termination_reason.c_str());
                if (r.x.size() != sz(n) || r.y.size() != sz(rel.linear.n_rows())) continue;
                f64 b = 0.0, raw = 0.0, cp = 0.0, cf = 0.0;
                if (wolfe_bound(node, r.x, r.y, d.psd_slack, b, raw, cp, cf))
                    bounds[t] = b - d.charge_form;
                xs[t] = r.x;
            }
            return;
        }
        const std::size_t width = dev.lanes() != 0 ? dev.lanes()
                                                   : std::max<std::size_t>(1, opts.batch);
        for (std::size_t off = 0; off < fixes.size(); off += width) {
            if (off != 0 && out_of_time()) break;
            const std::size_t real = std::min(width, fixes.size() - off);
            std::vector<backend::LaneBounds> lanes;
            for (std::size_t t = 0; t < real; ++t) {
                const auto& f = *fixes[off + t];
                backend::LaneBounds b{rel.linear.col_lo, rel.linear.col_hi,
                                      rel.linear.row_lo, rel.linear.row_hi};
                for (Index j = 0; j < n; ++j)
                    if (f[sz(j)] >= 0) b.col_lo[sz(j)] = b.col_hi[sz(j)] = f[sz(j)];
                lanes.push_back(std::move(b));
            }
            // A batch must not outlive the run's budget: give it what is
            // left.  Its bounds stay valid when cut short -- just weaker.
            engines::QpOptions qp = opts.qcr.qp;
            if (max_iterations != 0) qp.max_iterations = max_iterations;
            if (phase_deadline_s > 0.0) {
                const double left = phase_deadline_s - ms_since(t0) / 1000.0;
                qp.time_limit_s = qp.time_limit_s > 0.0 ? std::min(qp.time_limit_s, left) : left;
                qp.time_limit_s = std::max(qp.time_limit_s, 1e-3);
            }
            // A device built for a fixed width (the CPU lanes adapter)
            // refuses any other: pad with copies of the last lane and ignore
            // their results.  Without this every batch narrower than the
            // device -- the root and the first levels of the tree -- came
            // back empty, and those nodes inherited their parents' -inf
            // bound.
            if (dev.lanes() != 0)
                while (lanes.size() < dev.lanes()) lanes.push_back(lanes.back());
            std::vector<engines::QpDiagnostics> diags;
            const double tb = ms_since(t0);
            const auto raws = engines::solve_qp_pdhcg_batch(rel, lanes, qp, dev, diags);
            ++res.batches;
            if (opts.verbose) {
                std::uint64_t it = 0;
                for (std::size_t l = 0; l < real && l < diags.size(); ++l)
                    it = std::max(it, diags[l].iterations);
                std::fprintf(stderr, "bab: pass %llu lanes %zu iters<=%llu %.0f ms (t=%.1fs)\n",
                             static_cast<unsigned long long>(res.batches), real,
                             static_cast<unsigned long long>(it), ms_since(t0) - tb,
                             ms_since(t0) / 1000.0);
            }
            for (std::size_t l = 0; l < real && l < raws.size(); ++l) {
                const auto& r = raws[l];
                if (r.x.size() != sz(n)) continue;
                engines::QpProblem node = rel;
                node.linear.col_lo = lanes[l].col_lo;
                node.linear.col_hi = lanes[l].col_hi;
                f64 b = 0.0, raw = 0.0, cp = 0.0, cf = 0.0;
                if (wolfe_bound(node, r.x, r.y, d.psd_slack, b, raw, cp, cf))
                    bounds[off + l] = b - d.charge_form;
                xs[off + l] = r.x;
            }
        }
    };

    // ---- root: bound it once per candidate shift and keep the winner ----
    // With QcrShift::Best there are two candidates.  Each bound is valid on
    // its own (weak duality at that relaxation's own certified PSD slack), so
    // keeping the larger is valid -- and which one is larger is genuinely
    // instance-dependent, see qcr.hpp.
    const std::vector<std::int8_t> root_fix(sz(n), -1);
    // Root share of the budget.  0.35 is a choice, NOT a measured optimum:
    // enough for a 2e5-iteration root on the mid-size instances, while still
    // leaving most of the run to the tree.  Not yet swept against 0.15 / 1.0.
    if (opts.time_limit_s > 0.0) phase_deadline_s = 0.35 * opts.time_limit_s;
    std::size_t winner = 0;
    f64 root_bound = -kInf;
    std::vector<f64> root_x;
    std::vector<f64> cand_bound(cands.size(), -kInf);
    for (std::size_t c = 0; c < cands.size(); ++c) {
        std::vector<f64> bs;
        std::vector<std::vector<f64>> xs;
        bound_nodes(cands[c].relaxed, cands[c].diag, {&root_fix},
                    opts.root_iterations, bs, xs);
        cand_bound[c] = bs[0];
        if (c == 0 || bs[0] > root_bound) {
            root_bound = bs[0];
            winner = c;
            root_x = xs[0];
        }
    }
    if (cands.size() > 1) {   // publish the runner-up, so the choice is auditable
        const std::size_t other = winner == 0 ? 1 : 0;
        res.shift_alt = cands[other].diag.shift_used;
        res.root_bound_alt = sense * cand_bound[other];
        res.root_bound_alt_valid = std::isfinite(cand_bound[other]);
    }
    phase_deadline_s = opts.time_limit_s;
    qd = cands[winner].diag;
    const engines::QpProblem relaxed = std::move(cands[winner].relaxed);
    cands.clear();
    res.root_bound = sense * root_bound;
    res.root_bound_valid = std::isfinite(root_bound);
    res.nodes = 1;

    std::priority_queue<Node, std::vector<Node>, Worse> open;
    // Rounding heuristic: cheap, and all a primal point needs is to pass the
    // raw-data check in offer().
    auto round_and_offer = [&](const std::vector<f64>& x) {
        if (!opts.rounding || x.size() != sz(n)) return;
        std::vector<std::uint8_t> xr(sz(n));
        for (Index j = 0; j < n; ++j) xr[sz(j)] = x[sz(j)] >= 0.5 ? 1 : 0;
        offer(xr);
    };

    // Pseudocosts, in units of bound gained per unit of fractionality closed
    // (Benichou et al. 1971).  Kept across the whole tree, per variable.
    struct PseudoCost {
        f64 dn_sum = 0.0, up_sum = 0.0;
        int dn_cnt = 0, up_cnt = 0;
    };
    std::vector<PseudoCost> pc(sz(n));

    // Branch a batch of nodes at once.  Without strong branching this is the
    // old rule (the free variable closest to 1/2).  With it, every probe
    // needed by every node in the batch goes to the device together, so the
    // batched relaxation solver is used for branching decisions as well as
    // for bounds.
    auto branch_many = [&](const std::vector<Node>& nodes,
                           const std::vector<std::vector<f64>>& xs) {
        struct Choice {
            Index j = -1;
            f64 val = 0.5;                // x_j in the parent's relaxation point
            f64 frac = 0.0;               // min(x_j, 1 - x_j): how fractional
            f64 dn = -kInf, up = -kInf;   // probe bounds, min form
        };
        std::vector<std::vector<Choice>> per_node(nodes.size());
        std::vector<std::vector<std::int8_t>> probe_fix;
        struct Req { std::size_t node, cand; bool up; };
        std::vector<Req> reqs;

        for (std::size_t t = 0; t < nodes.size(); ++t) {
            const auto& nd = nodes[t];
            const auto& x = xs[t];
            std::vector<Choice> cs;
            for (Index j = 0; j < n; ++j) {
                if (nd.fix[sz(j)] >= 0) continue;
                const f64 v = x.size() == sz(n) ? x[sz(j)] : 0.5;
                Choice c;
                c.j = j;
                c.val = std::min(std::max(v, 0.0), 1.0);
                c.frac = std::min(c.val, 1.0 - c.val);
                cs.push_back(c);
            }
            if (cs.empty()) continue;
            std::sort(cs.begin(), cs.end(), [](const Choice& a, const Choice& b) {
                return a.frac > b.frac;
            });
            if (opts.sb_candidates > 0 && cs.size() > opts.sb_candidates)
                cs.resize(opts.sb_candidates);
            per_node[t] = std::move(cs);
            if (opts.sb_candidates == 0) continue;
            for (std::size_t k = 0; k < per_node[t].size(); ++k) {
                const Index j = per_node[t][k].j;
                const auto& p = pc[sz(j)];
                if (p.dn_cnt >= opts.reliability && p.up_cnt >= opts.reliability)
                    continue;             // trusted: no probe
                for (bool up : {false, true}) {
                    probe_fix.push_back(nd.fix);
                    probe_fix.back()[sz(j)] = up ? 1 : 0;
                    reqs.push_back({t, k, up});
                }
            }
        }

        if (!reqs.empty() && !out_of_time()) {
            std::vector<const std::vector<std::int8_t>*> ptrs;
            ptrs.reserve(probe_fix.size());
            for (const auto& f : probe_fix) ptrs.push_back(&f);
            std::vector<f64> pb;
            std::vector<std::vector<f64>> px;
            bound_nodes(relaxed, qd, ptrs, opts.sb_iterations, pb, px);
            res.probes += ptrs.size();
            for (std::size_t r = 0; r < reqs.size(); ++r) {
                const auto& rq = reqs[r];
                auto& c = per_node[rq.node][rq.cand];
                (rq.up ? c.up : c.dn) = pb[r];
                round_and_offer(px[r]);
                // Pseudocost update: bound gained per unit of fractionality
                // closed on that side.  A probe that returned nothing teaches
                // nothing.
                const f64 parent = nodes[rq.node].bound;
                if (!std::isfinite(pb[r]) || !std::isfinite(parent)) continue;
                const f64 xj = c.val;
                const f64 step = std::max(rq.up ? 1.0 - xj : xj, 1e-6);
                const f64 gain = std::max(0.0, pb[r] - parent) / step;
                auto& p = pc[sz(c.j)];
                if (rq.up) { p.up_sum += gain; ++p.up_cnt; }
                else       { p.dn_sum += gain; ++p.dn_cnt; }
            }
        }

        for (std::size_t t = 0; t < nodes.size(); ++t) {
            const auto& nd = nodes[t];
            const auto& cs = per_node[t];
            if (cs.empty()) continue;
            std::size_t pick = 0;
            if (opts.sb_candidates > 0) {
                // Product rule (Achterberg, Koch & Martin 2005): score the
                // pair of gains, with a floor so that a zero on one side does
                // not erase the other.
                constexpr f64 kEps = 1e-6;
                f64 best = -kInf;
                for (std::size_t k = 0; k < cs.size(); ++k) {
                    const auto& p = pc[sz(cs[k].j)];
                    const f64 xj = cs[k].val;
                    const f64 gdn = std::isfinite(cs[k].dn)
                        ? std::max(0.0, cs[k].dn - nd.bound)
                        : (p.dn_cnt ? p.dn_sum / p.dn_cnt * std::max(xj, 1e-6) : 0.0);
                    const f64 gup = std::isfinite(cs[k].up)
                        ? std::max(0.0, cs[k].up - nd.bound)
                        : (p.up_cnt ? p.up_sum / p.up_cnt * std::max(1.0 - xj, 1e-6) : 0.0);
                    const f64 s = std::max(gdn, kEps) * std::max(gup, kEps);
                    if (s > best) { best = s; pick = k; }
                }
            }
            const Choice& c = cs[pick];
            for (bool up : {false, true}) {
                Node child{nd.bound, nd.fix};
                child.fix[sz(c.j)] = up ? 1 : 0;
                // A probe bound is certified exactly like a node bound, so
                // the child may start from it.
                const f64 probed = up ? c.up : c.dn;
                if (std::isfinite(probed)) child.bound = std::max(child.bound, probed);
                open.push(std::move(child));
            }
        }
    };

    round_and_offer(root_x);
    if (n == 0) {
        offer({});
        ++res.leaves;
    } else if (prunable(root_bound)) {
        ++res.pruned;
    } else {
        branch_many({Node{root_bound, root_fix}}, {root_x});
    }
    const std::size_t K = std::max<std::size_t>(1, opts.batch);

    while (!open.empty()) {
        if (out_of_time() || res.nodes >= opts.max_nodes) {
            res.reason = "limit reached with open nodes";
            break;
        }
        // Take up to K nodes that can still matter.
        std::vector<Node> batch;
        while (!open.empty() && batch.size() < K) {
            Node nd = open.top();
            open.pop();
            if (prunable(nd.bound)) { ++res.pruned; continue; }
            // A leaf is a single point: score it exactly, no relaxation.
            if (std::find(nd.fix.begin(), nd.fix.end(), std::int8_t{-1}) == nd.fix.end()) {
                std::vector<std::uint8_t> x(sz(n));
                for (Index j = 0; j < n; ++j) x[sz(j)] = static_cast<std::uint8_t>(nd.fix[sz(j)]);
                offer(x);
                ++res.leaves;
                ++res.nodes;
                continue;
            }
            batch.push_back(std::move(nd));
        }
        if (batch.empty()) continue;

        std::vector<const std::vector<std::int8_t>*> fixes;
        fixes.reserve(batch.size());
        for (const auto& nd : batch) fixes.push_back(&nd.fix);
        std::vector<f64> bounds;
        std::vector<std::vector<f64>> xs;
        bound_nodes(relaxed, qd, fixes, 0, bounds, xs);
        res.nodes += batch.size();

        std::vector<Node> live;
        std::vector<std::vector<f64>> live_x;
        for (std::size_t l = 0; l < batch.size(); ++l) {
            Node& nd = batch[l];
            round_and_offer(xs[l]);
            nd.bound = std::max(nd.bound, bounds[l]);
            if (prunable(nd.bound)) { ++res.pruned; continue; }
            live.push_back(std::move(nd));
            live_x.push_back(std::move(xs[l]));
        }
        if (!live.empty()) branch_many(live, live_x);
    }

    // Global bound: the smallest of the open bounds, the bounds of nodes
    // pruned within tolerance, and the incumbent.
    f64 L = std::min(U, pruned_floor);
    if (!open.empty()) L = std::min(L, open.top().bound);
    res.bound_valid = std::isfinite(L);
    res.bound = sense * L;
    res.proved = open.empty() && res.have_incumbent;
    if (res.proved) res.reason = "tree closed";
    else if (open.empty() && !res.have_incumbent) res.reason = "tree closed, no feasible point";
    res.total_ms = ms_since(t0);
    return res;
}

}  // namespace sor::search
