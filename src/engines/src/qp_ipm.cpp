// FORGE — primal-dual interior-point method for convex QP.
//
// LAYER L3.  The exact path for convex QP: Newton on the perturbed KKT
// conditions converges in tens of iterations almost regardless of scaling,
// where a first-order method on QPLIB's badly scaled instances did not
// converge at all.  Method: Mehrotra's predictor-corrector (Mehrotra 1992)
// for the QP form in Nocedal & Wright ch. 16 / Vanderbei ch. 18; implemented
// from those texts, no solver source consulted.
//
// FORM.  min 1/2 x'Qx + c'x  s.t.  A x - w = 0,  l <= x <= u,  rl <= w <= ru.
// Row activities w are variables, so ranged, one-sided and equality rows
// are all "a bounded variable".  Every finite bound gets a slack and a
// multiplier: s_l = z - l, s_u = u - z, lam, nu >= 0.  Lagrangian sign:
//   Q x + c - A'y - lam_x + nu_x = 0,     y - lam_w + nu_w = 0.
// Newton, eliminating the bound multipliers and then dw, gives the reduced
// system, QUASI-DEFINITE in any order (Vanderbei 1995):
//   [ -(Q + Tx + rho I)   A'        ] [dx]   [ r_dx + h_x                  ]
//   [   A                 Tw^-1 + d ] [dy] = [ -r_p - Tw^-1 (r_dw + h_w)   ]
// with T = lam/s_l + nu/s_u per component and h the complementarity terms.
// Equality rows have w fixed (Tw^-1 = 0, regularised by d); free columns have
// T = 0 (regularised by rho).  rho and d act as proximal terms (Friedlander &
// Orban 2012), so the regularised system is the one solved.
//
// LP (Q = 0, no quadratic rows).  Eliminating dx from the augmented system
// leaves the normal equation of order m,
//   M = A D A^T + G,   D_j = 1 / (T_j + rho),
//   G_i = delta on equalities and 1/T_i + delta on inequalities.
// The sparsity pattern of the sparse-column product is analysed once; each
// Newton iteration only reassembles its values and refactorises.  The (up to
// 256) longest columns, those with at least 64 nonzeros, are a low-rank
// Sherman-Morrison-Woodbury update on top of the factor of the rest (fit2p).
// The augmented LDL' stays as the fallback -- chosen up front when M is the
// costlier factor, taken over mid-run if the normal equations break down --
// and is the only path for Q != 0.
//
// CENTRALITY.  A pair (s, lam) can drift to s ~ 0 with lam ~ 1/s while its
// product stays near mu: "converged" complementarity, hopeless conditioning.
// Its T = lam/s then reaches 1e17, which FREEZES that variable -- Newton can
// no longer move it, and for a row slack the primal residual is left for the
// dual regularisation to absorb as a huge dy (measured on QPLIB_8845: row
// 511 stuck at rp = 1.2 with T = 1e17 and dy = 1e8, for 100 iterations).
// Gondzio's multiple centrality correctors (Gondzio 1996) are the fix: after
// the Mehrotra corrector, retarget the products that sit outside
// [beta_min*mu, beta_max*mu] and re-solve with the SAME factorization,
// keeping the new direction only if it buys a longer step.
//
// PRESOLVE, the minimum the form needs: fixed columns are substituted out
// (branch-and-bound fixes variables, so this is common) and free rows are
// dropped.  Their duals are reconstructed afterwards.
//
// THE CLAIM.  The IPM's own residuals are in scaled, presolved units.  The
// solution is mapped back and re-checked by qpc::kkt_original on the
// ORIGINAL data -- the same independent check PDHCG-II's claim rests on --
// and only that check can return Optimal.
#include "forge/engines/qp.hpp"
#include "forge/core/env_switches.hpp"
#include "forge/engines/qcqp.hpp"
#include "forge/la/ldlt.hpp"
#include "forge/sparse/csr.hpp"

#include "forge/core/parallel.hpp"
#include "forge/core/process_memory.hpp"
#include "ipm_core.hpp"
#include "qp_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <new>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

namespace forge::engines {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
using qpc::sz;
constexpr f64 kInf = std::numeric_limits<f64>::infinity();

using ipmc::Core;

// Thrown inside run_ipm when the time limit (or a race cancel) arrives
// during the setup -- the normal-equation or augmented analysis, whose
// ordering of a large system can outlast a limit on its own.  Caught at
// the end of run_ipm: the run reports "time limit" like any other.
struct DeadlineReached {};

// Full symmetric Q x from the upper-triangle column lists.
void q_mul(const Core& k, const std::vector<f64>& x, std::vector<f64>& out) {
    out.assign(sz(k.n), 0.0);
    for (core::Index j = 0; j < k.n; ++j)
        for (const auto& [i, v] : k.qcol[sz(j)]) {
            out[sz(i)] += v * x[sz(j)];
            if (i != j) out[sz(j)] += v * x[sz(i)];
        }
}

struct Result {
    std::vector<f64> x, y;   // core space; y in the Lagrangian sign above
    int iterations = 0;
    bool converged = false;
    f64 worst_solve_residual = 0.0;
    std::string reason;
    bool stopped_on_time = false;
};

// Normal equations for the LP barrier (Q = 0).  The regularised augmented
// system is
//   [ -(T_x + rho)   A'  ] [dx]   [r1]
//   [     A          G   ] [dy] = [r2],   G_i = delta (+ 1/T_w,i on an inequality row),
// and either unknown can be eliminated.  Eliminating dx leaves the order-m
// PRIMAL form
//   M dy = r2 + A D r1,     M = A D A' + G,   D_j = 1 / (T_j + rho),
//   dx = D (A' dy - r1);
// eliminating dy instead leaves the order-n DUAL form
//   N dx = A' H r2 - r1,    N = A' H A + (T_x + rho),   H = G^-1,
//   dy = H (r2 - A dx).
// Both are SPD and have the same shape, B S B' + diag(g) for B = A or A'.
// Which is cheaper is a property of the model, not of the method: rmine15
// has 358k rows and 42k columns, ex10 70k rows of 17 entries against 18k
// columns of 66 -- their A D A' is dense (ex10: a 5 GB factor) while A' H A
// is small.  The dual form's equality rows carry H = 1/delta, exactly as a
// free column carries D = 1/rho in the primal form, and the same Krylov
// refinement against the true system covers both.  The form is chosen by
// the analysed factor cost (choose_form).
//
// The matrix is built from the SPARSE columns of B only; the few dense
// ones are a low-rank update applied with the Sherman-Morrison-Woodbury
// identity on top of one Cholesky-type (LDL', positive pivots)
// factorisation of the sparse part.  The sparsity pattern is analysed
// once; each Newton iteration only reassembles the values (column by
// column, in parallel) and refactorises.
struct NormalEq {
    // A by column and by row, as views into flat CSC/CSR arrays (duplicate
    // entries merged).  They used to be separate vectors -- on s100 (364k
    // columns of ~5 entries) the products with A and A' that wrap every
    // solve then cost more than the solve itself.  NormalEq is never copied
    // or moved after build(), so the views stay valid.
    template <class T> struct Span {
        const T* p = nullptr;
        std::size_t len = 0;
        std::size_t size() const { return len; }
        bool empty() const { return len == 0; }
        const T& operator[](std::size_t i) const { return p[i]; }
        const T* data() const { return p; }
        const T* begin() const { return p; }
        const T* end() const { return p + len; }
    };
    struct Col {
        Span<core::Index> rows;   // ascending, duplicates merged
        Span<f64> vals;
    };
    std::vector<core::Offset> ccp;           // A by column
    std::vector<core::Index> crow;
    std::vector<f64> cval;
    std::vector<core::Offset> rrp;           // A by row (same merged entries)
    std::vector<core::Index> rcj;
    std::vector<f64> rcv;
    std::vector<Col> acols, arows;

    core::Index n = 0, m = 0;                // A is m x n
    // The form: B = A (primal, system order m) or B = A' (dual, order n).
    bool transposed = false;
    core::Index P = 0, Q = 0;                // system order, columns of B
    const std::vector<Col>* bcols = nullptr; // columns of B
    const std::vector<Col>* brows = nullptr; // rows of B
    std::vector<core::Index> dense_of;       // B columns taken out into the Schur update
    std::vector<std::uint8_t> is_dense;
    int dense_candidates = 0;                // B columns long enough to be considered
    bool schur = false;
    // The system held as its full upper triangle (dense_mode): see build().
    bool dense_mode = false;
    bool ok = true;                          // false: no form fits, stay augmented
    bool interrupted = false;                // the deadline ended the analysis
    bool verbose = false;                    // per-trial analysis lines on stderr

    la::SymCsc M;
    std::vector<core::Offset> diag_slot;
    std::vector<std::int8_t> sign;
    la::Ldlt fac;
    int reg_pivots = 0;
    double assemble_ms = 0.0, ldlt_ms = 0.0, schur_ms = 0.0;   // cumulative, for --verbose
    f64 assemble_work = 0.0;                                     // multiply-adds per assembly

    std::vector<f64> dscale;                 // per B column: D_j (primal) or H_i (dual)
    std::vector<f64> gdiag;                  // per system row: G_i (primal) or T_j + rho (dual)
    std::vector<f64> sroot;                  // sqrt(dscale) of the dense columns
    std::vector<f64> Y;                      // P * p, column-major, M^{-1} B_dense
    std::vector<f64> schur_l;                // p * p row-major lower Cholesky of I + S^1/2 B'M^-1B S^1/2
    std::vector<f64> bwork, zwork, wwork;
    std::vector<std::vector<f64>> wscratch;  // per-worker dense accumulator (assemble)

    // Memory: the factor budget is what the machine has left, less the
    // pattern and the Schur vectors, at the time of the analysis.  The old
    // fixed 40% of physical memory ignored everything else the process and
    // the machine held (a2864 and five other lpopt models were OOM-killed
    // with the factor "within budget").
    static std::uint64_t available_memory_bytes() {
        std::uint64_t avail = 0;
        if (std::FILE* fp = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            while (std::fgets(line, sizeof line, fp)) {
                unsigned long long kb = 0;
                if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { avail = kb * 1024ULL; break; }
            }
            std::fclose(fp);
        }
        if (avail == 0) {
            const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
            if (pages > 0 && page > 0)
                avail = static_cast<std::uint64_t>(0.4 * static_cast<double>(pages) * static_cast<double>(page));
        }
        // An address-space limit (ulimit -v) binds before the machine does:
        // what is left of it is the current virtual size subtracted from it.
        struct rlimit rl {};
        if (getrlimit(RLIMIT_AS, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) {
            unsigned long long vm_pages = 0;
            if (std::FILE* fp = std::fopen("/proc/self/statm", "r")) {
                if (std::fscanf(fp, "%llu", &vm_pages) != 1) vm_pages = 0;
                std::fclose(fp);
            }
            const long page = sysconf(_SC_PAGE_SIZE);
            const std::uint64_t used = vm_pages * static_cast<std::uint64_t>(page > 0 ? page : 4096);
            const std::uint64_t lim = static_cast<std::uint64_t>(rl.rlim_cur);
            avail = std::min(avail, lim > used ? lim - used : 0);
        }
        return avail;
    }
    static void budget_factor(la::Ldlt& f, std::uint64_t reserved, const std::function<bool()>* stop) {
        la::LdltOptions o = f.options();
        // Three quarters of what is free, less what this system's own
        // arrays take, leaves the iteration's vectors and the OS some room.
        const std::uint64_t avail = available_memory_bytes();
        const std::uint64_t usable = static_cast<std::uint64_t>(0.75 * static_cast<double>(avail));
        o.max_factor_bytes = usable > reserved ? usable - reserved : 1;
        if (stop) o.should_stop = *stop;
        f.set_options(o);
    }
    static std::uint64_t pattern_bytes(std::size_t nnz) {
        return static_cast<std::uint64_t>(nnz) * (sizeof(core::Index) + sizeof(f64));
    }

    static constexpr std::size_t kKeepPatternEntries = 20000000;

    // One analysed candidate: a form and a dense split.
    struct Candidate {
        bool valid = false;
        bool transposed = false;
        std::vector<std::uint8_t> dense;      // per B column
        std::vector<core::Index> dense_of;
        la::SymCsc M;
        std::vector<core::Offset> dslot;
        la::Ldlt fac;
        f64 cost = kInf;                       // per Newton iteration
        f64 probe_lb = kInf;                   // flops (lower bound) of the first trial analysed
        f64 probe_nnzl = 0.0;                  // and its nnz(L) (lower bound)
        int candidates = 0;
    };

    // Both orientations of A, then the form.
    void build(const ipmc::Core& k, const std::function<bool()>* stop = nullptr) {
        n = k.n;
        m = k.m;
        {
            // Rows are visited in ascending order, so each column's entries
            // arrive sorted and a duplicate (i, j) is the previous entry.
            const auto& rp = k.A.pattern.row_ptr();
            const auto& ci = k.A.pattern.col_idx();
            std::vector<core::Offset> cnt(sz(n) + 1, 0);
            for (core::Index i = 0; i < m; ++i)
                for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) ++cnt[sz(ci[sz(t)]) + 1];
            for (core::Index j = 0; j < n; ++j) cnt[sz(j) + 1] += cnt[sz(j)];
            std::vector<core::Offset> cur(cnt.begin(), cnt.end() - 1);
            std::vector<core::Index> rtmp(sz(cnt[sz(n)]));
            std::vector<f64> vtmp(sz(cnt[sz(n)]));
            for (core::Index i = 0; i < m; ++i)
                for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
                    const core::Index j = ci[sz(t)];
                    const core::Offset at = cur[sz(j)];
                    if (at > cnt[sz(j)] && rtmp[sz(at - 1)] == i) { vtmp[sz(at - 1)] += k.A.vals[sz(t)]; continue; }
                    rtmp[sz(at)] = i;
                    vtmp[sz(at)] = k.A.vals[sz(t)];
                    ++cur[sz(j)];
                }
            ccp.assign(sz(n) + 1, 0);
            for (core::Index j = 0; j < n; ++j) ccp[sz(j) + 1] = ccp[sz(j)] + (cur[sz(j)] - cnt[sz(j)]);
            crow.resize(sz(ccp[sz(n)]));
            cval.resize(sz(ccp[sz(n)]));
            for (core::Index j = 0; j < n; ++j) {
                const core::Offset len = cur[sz(j)] - cnt[sz(j)];
                std::copy_n(rtmp.begin() + cnt[sz(j)], len, crow.begin() + ccp[sz(j)]);
                std::copy_n(vtmp.begin() + cnt[sz(j)], len, cval.begin() + ccp[sz(j)]);
            }
            acols.assign(sz(n), {});
            for (core::Index j = 0; j < n; ++j) {
                const auto len = static_cast<std::size_t>(ccp[sz(j) + 1] - ccp[sz(j)]);
                acols[sz(j)].rows = {crow.data() + ccp[sz(j)], len};
                acols[sz(j)].vals = {cval.data() + ccp[sz(j)], len};
            }
        }
        // The same merged entries by row (columns ascending within a row).
        rrp.assign(sz(m) + 1, 0);
        for (core::Index j = 0; j < n; ++j)
            for (core::Index r : acols[sz(j)].rows) ++rrp[sz(r) + 1];
        for (core::Index i = 0; i < m; ++i) rrp[sz(i) + 1] += rrp[sz(i)];
        rcj.assign(sz(rrp[sz(m)]), 0);
        rcv.assign(sz(rrp[sz(m)]), 0.0);
        {
            std::vector<core::Offset> cur(rrp.begin(), rrp.end() - 1);
            for (core::Index j = 0; j < n; ++j) {
                const auto& c = acols[sz(j)];
                for (std::size_t e = 0; e < c.rows.size(); ++e) {
                    const auto at = static_cast<std::size_t>(cur[sz(c.rows[e])]++);
                    rcj[at] = j;
                    rcv[at] = c.vals[e];
                }
            }
        }
        arows.assign(sz(m), {});
        for (core::Index i = 0; i < m; ++i) {
            const auto len = static_cast<std::size_t>(rrp[sz(i) + 1] - rrp[sz(i)]);
            arows[sz(i)].rows = {rcj.data() + rrp[sz(i)], len};
            arows[sz(i)].vals = {rcv.data() + rrp[sz(i)], len};
        }
        is_dense.clear();
        ok = false;
        interrupted = false;
        if (m == 0) { set_form(false); ok = true; return; }
        choose_form(stop);
    }

    void set_form(bool t) {
        transposed = t;
        P = t ? n : m;
        Q = t ? m : n;
        bcols = t ? &arows : &acols;
        brows = t ? &acols : &arows;
    }

    // The split candidates of a form: B columns long enough to be taken
    // out into the Schur update, longest first, and how many at most.
    struct SplitPlan {
        std::vector<core::Index> cand;
        std::size_t cap = 0;      // most columns a trial splits
        std::size_t p0 = 0;       // columns every trial splits (>= P/4 rows)
    };
    SplitPlan split_plan(bool t) const {
        constexpr core::Index kDenseNnz = 64;
        constexpr std::size_t kMaxSchur = 256;
        constexpr std::size_t kMaxSchurStore = 30000000;   // P * p doubles
        const auto& cols = t ? arows : acols;
        const core::Index Pt = t ? n : m;
        SplitPlan sp;
        for (core::Index q = 0; q < static_cast<core::Index>(cols.size()); ++q)
            if (static_cast<core::Index>(cols[sz(q)].rows.size()) >= kDenseNnz) sp.cand.push_back(q);
        std::stable_sort(sp.cand.begin(), sp.cand.end(), [&](core::Index a, core::Index b) {
            return cols[sz(a)].rows.size() > cols[sz(b)].rows.size();
        });
        sp.cap = std::min({sp.cand.size(), kMaxSchur,
                           static_cast<std::size_t>(std::max<core::Index>(Pt / 2, 1)),
                           std::max<std::size_t>(kMaxSchurStore / sz(std::max<core::Index>(Pt, 1)), 1)});
        // Columns over a quarter of the rows would make M dense: always split
        // (analysing that M only to reject it costs more than the solve).
        while (sp.p0 < sp.cap && static_cast<core::Index>(cols[sz(sp.cand[sp.p0])].rows.size()) >= Pt / 4) ++sp.p0;
        return sp;
    }

    // Upper bound on the pattern of the form's sparse part when its longest
    // columns are split off (as many as a trial may split): the cliques of
    // the remaining columns of B.  Without the split, a few dense rows of A
    // made the dual form look as dense as A'A (rmine15).
    f64 clique_estimate(bool t) const {
        const auto& cols = t ? arows : acols;
        const SplitPlan sp = split_plan(t);
        std::vector<std::uint8_t> out(cols.size(), 0);
        for (std::size_t q = 0; q < sp.cap; ++q) out[sz(sp.cand[q])] = 1;
        f64 e = 0.0;
        for (std::size_t q = 0; q < cols.size(); ++q) {
            if (out[q]) continue;
            const f64 l = static_cast<f64>(cols[q].rows.size());
            e += 0.5 * l * (l + 1.0);
        }
        const f64 p = static_cast<f64>(t ? n : m);
        return std::min(e, 0.5 * p * (p + 1.0));
    }

    // Analyse the primal and the dual form -- the one whose pattern bound
    // is smaller first, the other only when the first is expensive and the
    // other's bound is not far larger -- and keep the cheaper per Newton
    // iteration.  Each analysis holds symbolic structures only (the factor
    // storage is allocated at the first factorisation), so comparing them
    // costs no factor memory.
    void choose_form(const std::function<bool()>* stop) {
        // Branch and bound over (form, split).  The most-split trial of a
        // form has the smallest pattern -- the quickest to order -- and,
        // fill being monotone in the pattern, about the fewest multiply-adds
        // that form can reach: its flops are a lower bound on the form's
        // cost.  So: probe both forms at their most-split trial (smaller
        // pattern bound first), then refine the split sizes only of a form
        // whose bound is below the best cost found (pilot87: the dual
        // form's probe bound 6.4e8 against the primal's analysed 1.5e8 ends
        // it; fit2d: the dual probe is diagonal but pays 25 Schur columns,
        // the primal's 25x25 system costs 5e3).  A second form whose
        // pattern bound is large (> 1e7 entries) and far above both the
        // first's bound and its factor is not probed (rmine15's primal);
        // on a small model every probe takes milliseconds.
        const f64 ep = clique_estimate(false), ed = clique_estimate(true);
        const bool dual_first = ed < ep;
        if (verbose)
            std::fprintf(stderr, "ipm ne-forms pattern bound primal %.3e dual %.3e -> %s first\n",
                         ep, ed, dual_first ? "dual" : "primal");
        Candidate best;
        f64 lb[2] = {kInf, kInf}, lbn[2] = {0.0, 0.0};
        bool probed[2] = {false, false};
        for (int round = 0; round < 2; ++round) {
            const bool t = (round == 0) == dual_first;
            if (round == 1) {
                if (stop && (*stop)()) { interrupted = true; break; }
                const f64 first = dual_first ? ed : ep, second = dual_first ? ep : ed;
                constexpr f64 kSmallPattern = 1e7;
                if (best.valid && second > kSmallPattern &&
                    second > 4.0 * std::max(first, static_cast<f64>(best.fac.nnz_l()))) break;
            }
            Candidate c = analyse_form(t, stop, best.cost, TrialSet::Probe);
            if (interrupted) break;
            probed[t ? 1 : 0] = true;
            lb[t ? 1 : 0] = c.probe_lb;
            lbn[t ? 1 : 0] = c.probe_nnzl;
            if (c.valid && c.cost < best.cost) best = std::move(c);
        }
        // Refine, the form with the smaller bound first.
        for (int k = 0; k < 2 && !interrupted; ++k) {
            const bool t = (k == 0) == (lb[1] < lb[0]);
            if (!probed[t ? 1 : 0] || !(lb[t ? 1 : 0] < best.cost)) continue;
            if (stop && (*stop)()) { interrupted = true; break; }
            Candidate c = analyse_form(t, stop, best.cost, TrialSet::Rest, lb[t ? 1 : 0], lbn[t ? 1 : 0]);
            if (interrupted) break;
            if (c.valid && c.cost < best.cost) best = std::move(c);
        }
        if (!best.valid) return;
        set_form(best.transposed);
        is_dense = std::move(best.dense);
        dense_of = std::move(best.dense_of);
        dense_candidates = best.candidates;
        M = std::move(best.M);
        diag_slot = std::move(best.dslot);
        if (M.n != P && P > 0) {
            // Dropped during the trials (kKeepPatternEntries): rebuild it.
            const bool rebuilt = pattern(is_dense, M, diag_slot);
            if (!rebuilt) return;
        }
        fac = std::move(best.fac);
        schur = !dense_of.empty();
        ok = true;
        // DENSE NORMAL EQUATIONS.  When the factor is mostly dense anyway,
        // keep the system as its full upper triangle: every column -- the
        // long ones included, so no Schur update -- is added as an outer
        // product into the columns it reaches, reading each column of B
        // once, the updates landing in one contiguous column at a time.
        // The sparse assembly pulled a prefix of every column of B through
        // each row of M (square41, m = 1754, L 90% dense: 8.9e8
        // multiply-adds per assembly at 0.55 G/s, two thirds of the
        // barrier), and the 256 split columns cost another 0.6 s per
        // iteration in Schur solves.
        constexpr f64 kDenseEntries = 2e7;
        const f64 full = 0.5 * static_cast<f64>(P) * (static_cast<f64>(P) + 1.0);
        if (full <= kDenseEntries && static_cast<f64>(fac.nnz_l()) >= 0.5 * full) {
            la::SymCsc Md;
            Md.n = P;
            Md.col_ptr.assign(sz(P) + 1, 0);
            Md.row_idx.resize(static_cast<std::size_t>(full));
            std::vector<core::Offset> dslot(sz(P), 0);
            for (core::Index q = 0; q < P; ++q) {
                const core::Offset base = static_cast<core::Offset>(q) * (q + 1) / 2;
                Md.col_ptr[sz(q) + 1] = base + q + 1;
                for (core::Index r = 0; r <= q; ++r) Md.row_idx[sz(base + r)] = r;
                dslot[sz(q)] = base + q;
            }
            la::Ldlt fd;
            budget_factor(fd, pattern_bytes(Md.row_idx.size()), stop);
            fd.analyze(Md);
            if (fd.interrupted()) { interrupted = true; ok = false; return; }
            if (fd.fits()) {
                dense_mode = true;
                schur = false;
                dense_of.clear();
                is_dense.assign(sz(Q), 0);
                M = std::move(Md);
                diag_slot = std::move(dslot);
                fac = std::move(fd);
            }
        }
        M.vals.assign(M.row_idx.size(), 0.0);
        sign.assign(sz(P), 1);
        assemble_work = 0.0;  // multiply-adds of one assembly
        for (core::Index q = 0; q < Q; ++q)
            if (!is_dense[sz(q)]) {
                const f64 l = static_cast<f64>((*bcols)[sz(q)].rows.size());
                assemble_work += 0.5 * l * (l + 1.0);
            }
    }

    // A column this long fills a dense patch of B S B'.  The longest of
    // them may be cheaper as a Schur update than inside the sparse factor,
    // but each costs one solve with M per factorisation and P doubles of
    // storage, so whether and how many is decided by trying: the pattern
    // is analysed for a few split sizes and the cheapest per Newton
    // iteration wins (pilot87's 178 columns of 64+ nonzeros: not split, 3x
    // faster; fit2p's few thousand-nonzero columns: split).
    enum class TrialSet { All, Probe, Rest };
    // probe_flops / probe_nnzl (TrialSet::Rest): the probe's factor, a lower
    // bound on every less-split trial's (their patterns contain the probe's).
    Candidate analyse_form(bool t, const std::function<bool()>* stop, f64 cost_cap, TrialSet which = TrialSet::All,
                           f64 probe_flops = 0.0, f64 probe_nnzl = 0.0) {
        set_form(t);
        Candidate best;
        best.transposed = t;
        const SplitPlan sp = split_plan(t);
        const std::vector<core::Index>& cand = sp.cand;
        best.candidates = static_cast<int>(cand.size());
        // Split sizes p0, 4 p0, 16 p0, ..., cap -- analysed LARGEST first:
        // the most-split pattern is the smallest and the quickest to order,
        // and its cost then caps the others (an ordering is abandoned once
        // its factor is provably costlier than the best in hand, AmdCaps).
        std::vector<std::size_t> trials;
        trials.push_back(sp.p0);
        for (std::size_t tr = std::max<std::size_t>(sp.p0, 1) * 4; tr < sp.cap; tr *= 4) trials.push_back(tr);
        if (sp.cap > sp.p0) trials.push_back(sp.cap);
        std::reverse(trials.begin(), trials.end());
        if (which == TrialSet::Probe) trials.resize(1);
        else if (which == TrialSet::Rest) trials.erase(trials.begin());
        for (const std::size_t p : trials) {
            if (stop && (*stop)()) { interrupted = true; return best; }
            std::vector<std::uint8_t> dense(sz(Q), 0);
            for (std::size_t q = 0; q < p; ++q) dense[sz(cand[q])] = 1;
            // This trial's factor flops must stay under what would make it
            // lose to the best candidate so far (its split's dense work is
            // known before the analysis; the solve term is left out, so the
            // cap only drops trials that cannot win).
            // A less-split trial's pattern contains the probe's, so its
            // nnz(L) and multiply-adds are at least the probe's: its split
            // penalty is bounded below before any analysis, and a trial
            // whose penalty plus the probe's multiply-adds already reaches
            // the bound is not analysed (degen3's dual: 64 and 16 columns).
            const f64 bound = std::min(cost_cap, best.cost);
            const f64 flop_cap = std::isfinite(bound)
                ? bound - static_cast<f64>(p) * (8.0 * probe_nnzl + 100.0 * static_cast<f64>(P)) : 0.0;
            if (std::isfinite(bound) && flop_cap <= probe_flops) continue;
            la::SymCsc Mt;
            std::vector<core::Offset> dslot;
            bool formed = false;
            try {
                formed = pattern(dense, Mt, dslot, flop_cap);
            } catch (const std::bad_alloc&) {
                Mt = la::SymCsc{};
                dslot.clear();
            }
            if (!formed) {
                if (verbose)
                    std::fprintf(stderr, "ipm ne-trial %s split=%zu dropped while forming its pattern (%zu entries)\n",
                                 t ? "dual" : "primal", p, Mt.row_idx.size());
                // Smaller splits only grow the pattern.
                break;
            }
            // The ordering's own working memory (amd_order): the symmetric
            // adjacency as one CSR of both triangles (8 bytes per entry of M)
            // and the variable lists copied from it (8 more) while both
            // exist; afterwards the quotient graph, whose variable, element
            // and membership lists together never exceed twice the original
            // adjacency, with growth slack -- 24 bytes per entry bounds it --
            // plus ~160 bytes of per-vertex arrays.  The symbolic phases
            // after it (permuted copies, row structure) take less.  It is
            // held to the share of free memory the factor itself may take
            // (budget_factor).  The earlier bound (32 bytes, half of what is
            // free) was written for the vector-per-row ordering and skipped
            // set-cover-model's only normal-equation candidate under a 5 GB
            // address-space limit (1.4 GB "needed", 2.3 GB free): the run
            // then fell back to ordering the 1.1M-node augmented system and
            // spent the whole budget in it.
            {
                const std::uint64_t amd_bytes = 24ULL * Mt.row_idx.size() + 160ULL * sz(P);
                const std::uint64_t avail = available_memory_bytes();
                if (static_cast<double>(amd_bytes) > 0.75 * static_cast<double>(avail)) {
                    if (verbose)
                        std::fprintf(stderr, "ipm ne-trial %s split=%zu skipped: ordering needs ~%.1f GB, %.1f GB free\n",
                                     t ? "dual" : "primal", p, static_cast<double>(amd_bytes) / 1e9,
                                     static_cast<double>(avail) / 1e9);
                    continue;
                }
            }
            la::Ldlt ft;
            budget_factor(ft, pattern_bytes(Mt.row_idx.size()) + static_cast<std::uint64_t>(p) * sz(P) * sizeof(f64),
                          stop);
            {
                la::LdltOptions o = ft.options();
                o.max_flops = flop_cap;
                ft.set_options(o);
            }
            const auto ta = Clock::now();
            try {
                ft.analyze(Mt);
            } catch (const std::bad_alloc&) {
                // A candidate whose analysis does not fit in memory is one
                // that does not fit: the estimate above is a bound, not a
                // guarantee, and an address-space limit makes the failure an
                // exception rather than the OOM killer.
                if (verbose)
                    std::fprintf(stderr, "ipm ne-trial %s split=%zu: analysis out of memory, skipped\n",
                                 t ? "dual" : "primal", p);
                continue;
            }
            if (ft.interrupted()) { interrupted = true; return best; }
            if (p == trials.front()) {
                best.probe_lb = ft.flops();
                best.probe_nnzl = static_cast<f64>(ft.nnz_l());
            }
            // Factor multiply-adds, plus per split column one solve with M
            // (two passes over L, memory-bound: weighted x4) and the dense
            // work of applying the update ~100 times per iteration.
            const f64 cost = ft.flops() + static_cast<f64>(p) * (8.0 * static_cast<f64>(ft.nnz_l()) +
                                                              100.0 * static_cast<f64>(P));
            if (verbose)
                std::fprintf(stderr, "ipm ne-trial %s split=%zu nnzM=%zu nnzL=%lld flops=%.3e cost=%.3e fits=%d%s analyse %.0f ms"
                             " (memory GB: rss %.2f peak %.2f)\n",
                             t ? "dual" : "primal", p, Mt.row_idx.size(), static_cast<long long>(ft.nnz_l()),
                             ft.flops(), cost, static_cast<int>(ft.fits()),
                             ft.fits() ? "" : ft.over_cost() ? " (over cost cap; bounds)" : " (over memory; bounds)",
                             ms_since(ta), core::process_memory().rss_gb, core::process_memory().peak_gb);
            if (!ft.fits()) {
                // The remaining trials split fewer columns, so their patterns
                // CONTAIN this one and their fill and multiply-adds are, for
                // a minimum-fill ordering, no smaller; they save at most the
                // split penalty.  When this factor's multiply-adds alone are
                // already within 10% of the best cost, they cannot win by
                // more than noise: stop (ex10's primal form: four more
                // 3.4 s orderings; supportcase10's: three 72M-entry patterns).
                if (ft.over_cost() && std::isfinite(bound) && ft.flops() >= 0.9 * bound) break;
                continue;
            }
            // Ties go to the smaller split (fewer Schur columns).
            if (cost < best.cost || (cost == best.cost && p < best.dense_of.size())) {
                best.cost = cost;
                best.valid = true;
                best.dense = std::move(dense);
                // A large pattern is not held while other trials run (it is
                // rethe winner, deterministically): two of them at
                // once were most of a2864's peak memory.
                if (Mt.row_idx.size() <= kKeepPatternEntries) {
                    best.M = std::move(Mt);
                    best.dslot = std::move(dslot);
                } else {
                    best.M = la::SymCsc{};
                    best.dslot.clear();
                }
                best.fac = std::move(ft);
                best.dense_of.assign(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(p));
                std::sort(best.dense_of.begin(), best.dense_of.end());
            }
        }
        return best;
    }

    // Frees the factor and the Schur vectors once the normal equations are
    // abandoned for the augmented system, so the two factors are never held
    // at once (the augmented one was budgeted before this one was allocated).
    void release() {
        fac = la::Ldlt{};
        M = la::SymCsc{};
        diag_slot = {};
        Y = {};
        wscratch = {};
    }

    // Cost of one Newton iteration with the chosen form (for the comparison
    // with the augmented system).
    f64 iteration_cost() const {
        return fac.flops() + 4.0 * static_cast<f64>(dense_of.size()) * static_cast<f64>(fac.nnz_l());
    }

    // Pattern of the sparse part for a given dense set: column q of the upper
    // triangle holds every r <= q sharing a sparse column of B with row q,
    // plus the diagonal.  False when it would be too large to hold.
    // flop_cap > 0: give up once the pattern alone proves the factor
    // costlier -- L holds at least nnz(M) entries in at most P columns, so
    // its multiply-adds sum c_j^2 >= nnz(M)^2 / P (Cauchy-Schwarz).
    bool pattern(const std::vector<std::uint8_t>& dense, la::SymCsc& Mo,
                 std::vector<core::Offset>& dslot, f64 flop_cap = 0.0) const {
        constexpr std::size_t kMaxNnzM = 150000000;
        const f64 nnz_cap = flop_cap > 0.0 ? std::sqrt(flop_cap * static_cast<f64>(P)) : kInf;
        Mo.n = P;
        Mo.col_ptr.assign(sz(P) + 1, 0);
        dslot.assign(sz(P), 0);
        std::vector<core::Index> mark(sz(P), -1), tmp;
        for (core::Index q = 0; q < P; ++q) {
            tmp.clear();
            const auto& row = (*brows)[sz(q)];
            for (std::size_t e = 0; e < row.rows.size(); ++e) {
                const core::Index j = row.rows[e];
                if (dense[sz(j)]) continue;
                for (core::Index r : (*bcols)[sz(j)].rows) {
                    if (r > q) break;
                    if (mark[sz(r)] != q) { mark[sz(r)] = q; tmp.push_back(r); }
                }
            }
            if (mark[sz(q)] != q) tmp.push_back(q);
            std::sort(tmp.begin(), tmp.end());
            for (core::Index r : tmp) Mo.row_idx.push_back(r);
            Mo.col_ptr[sz(q) + 1] = static_cast<core::Offset>(Mo.row_idx.size());
            dslot[sz(q)] = Mo.col_ptr[sz(q) + 1] - 1;   // row == q is last
            // (L's counts are strictly below the diagonal: compare M's.)
            if (Mo.row_idx.size() > kMaxNnzM ||
                static_cast<f64>(Mo.row_idx.size()) - static_cast<f64>(q + 1) > nnz_cap) return false;
        }
        return true;
    }

    // T is length n+m (x then w).  rho, dreg are the proximal terms already
    // scaled by any regularisation boost.
    void assemble(const std::vector<f64>& tdiag, f64 rho, f64 dreg,
                  const std::vector<std::uint8_t>& fixed_w) {
        const auto ta0 = Clock::now();
        struct Acc { double& a; Clock::time_point t; ~Acc() { a += ms_since(t); } } acc{assemble_ms, ta0};
        dscale.resize(sz(Q));
        gdiag.resize(sz(P));
        // G_i = delta (+ 1/T_w on an inequality row): the primal form's
        // diagonal, the dual form's column scale (inverted).
        auto g_of = [&](core::Index i) {
            f64 g = dreg;
            const f64 tw = tdiag[sz(n + i)];
            if (!fixed_w[sz(i)] && tw > 0.0) g += 1.0 / tw;
            return g;
        };
        if (!transposed) {
            for (core::Index j = 0; j < n; ++j) dscale[sz(j)] = 1.0 / (tdiag[sz(j)] + rho);
            for (core::Index i = 0; i < m; ++i) gdiag[sz(i)] = g_of(i);
        } else {
            for (core::Index i = 0; i < m; ++i) dscale[sz(i)] = 1.0 / g_of(i);
            for (core::Index j = 0; j < n; ++j) gdiag[sz(j)] = tdiag[sz(j)] + rho;
        }
        if (m == 0) return;
        if (dense_mode) { assemble_dense(); return; }
        // Column q of M reads only B and S, so columns are independent tasks;
        // every entry is summed in the same order whatever the schedule.
        auto column = [&](core::Index q, std::vector<f64>& w) {
            const auto& row = (*brows)[sz(q)];
            for (std::size_t t = 0; t < row.rows.size(); ++t) {
                const core::Index j = row.rows[t];
                if (is_dense[sz(j)]) continue;
                const f64 coef = row.vals[t] * dscale[sz(j)];
                const auto& c = (*bcols)[sz(j)];
                for (std::size_t e = 0; e < c.rows.size(); ++e) {
                    if (c.rows[e] > q) break;
                    w[sz(c.rows[e])] += coef * c.vals[e];
                }
            }
            for (auto p = M.col_ptr[sz(q)]; p < M.col_ptr[sz(q) + 1]; ++p) {
                const auto r = sz(M.row_idx[sz(p)]);
                M.vals[sz(p)] = w[r];
                w[r] = 0.0;
            }
            M.vals[sz(diag_slot[sz(q)])] += gdiag[sz(q)];
        };
        core::ThreadPool& pool = core::global_pool();
        constexpr core::Offset kChunk = 64;
        const core::Offset nchunks = (P + kChunk - 1) / kChunk;
        const int nw = std::max(1, pool.size());
        if (wscratch.size() != sz(nw)) wscratch.assign(sz(nw), {});
        for (auto& w : wscratch) if (w.size() != sz(P)) w.assign(sz(P), 0.0);
        auto body = [&](core::Offset chunk, int worker) {
            auto& w = wscratch[sz(worker)];
            const core::Index lo = static_cast<core::Index>(chunk * kChunk);
            const core::Index hi = static_cast<core::Index>(std::min<core::Offset>(P, (chunk + 1) * kChunk));
            for (core::Index q = lo; q < hi; ++q) column(q, w);
        };
        if (nw == 1 || nchunks <= 1 || P < 512) {
            for (core::Offset c = 0; c < nchunks; ++c) body(c, 0);
        } else {
            pool.parallel_for(nchunks, body);
        }
    }

    // Dense mode: column q of M (rows 0..q) at offset q(q+1)/2.  Column j of
    // B adds s_j b_j b_j' -- for each of its rows q, the products with its
    // rows r <= q into column q.  Columns of M are split into ranges of
    // about equal triangle area, one per task; within a range the columns
    // of B are taken in ascending order, so every entry is summed in the
    // same order whatever the thread count.  (Tiling M into L2-sized blocks
    // was measured slower: the cost is the indexed update itself, ~3 cycles,
    // not the traffic to M.)
    void assemble_dense() {
        core::ThreadPool& pool = core::global_pool();
        const int nw = std::max(1, pool.size());
        const core::Offset ntask = nw == 1 || P < 512 ? 1 : 4 * static_cast<core::Offset>(nw);
        std::vector<core::Index> cut(sz(ntask) + 1, P);
        cut[0] = 0;
        for (core::Offset t = 1; t < ntask; ++t)
            cut[sz(t)] = static_cast<core::Index>(static_cast<f64>(P) *
                                                  std::sqrt(static_cast<f64>(t) / static_cast<f64>(ntask)));
        f64* V = M.vals.data();
        auto body = [&](core::Offset t, int) {
            const core::Index q0 = cut[sz(t)], q1 = cut[sz(t) + 1];
            if (q0 >= q1) return;
            std::fill(V + static_cast<core::Offset>(q0) * (q0 + 1) / 2,
                      V + static_cast<core::Offset>(q1) * (q1 + 1) / 2, 0.0);
            for (core::Index j = 0; j < Q; ++j) {
                const auto& c = (*bcols)[sz(j)];
                const core::Index* rows = c.rows.data();
                const f64* vals = c.vals.data();
                const std::size_t k = c.rows.size();
                const std::size_t lo = static_cast<std::size_t>(std::lower_bound(rows, rows + k, q0) - rows);
                const std::size_t hi = static_cast<std::size_t>(std::lower_bound(rows + lo, rows + k, q1) - rows);
                const f64 dj = dscale[sz(j)];
                for (std::size_t a = lo; a < hi; ++a) {
                    const core::Index q = rows[a];
                    f64* col = V + static_cast<core::Offset>(q) * (q + 1) / 2;
                    const f64 coef = vals[a] * dj;
                    for (std::size_t b = 0; b <= a; ++b) col[rows[b]] += coef * vals[b];
                }
            }
            for (core::Index q = q0; q < q1; ++q) V[sz(diag_slot[sz(q)])] += gdiag[sz(q)];
        };
        if (ntask == 1) body(0, 0);
        else pool.parallel_for(ntask, body);
    }

    bool factorize() {
        reg_pivots = 0;
        if (m == 0) return true;
        auto t = Clock::now();
        const bool fok = fac.factorize(M, sign, 1e-12);
        ldlt_ms += ms_since(t);
        if (!fok) return false;
        reg_pivots = static_cast<int>(fac.regularized_pivots());
        t = Clock::now();
        const bool sok = !schur || prepare_schur();
        schur_ms += ms_since(t);
        return sok;
    }

    // K_reg [dx; dy] = rhs, in place.  rhs is length n+m.
    std::uint64_t n_apply = 0;               // preconditioner applications (for --verbose)
    double apply_ms = 0.0, msolve_ms = 0.0;
    void apply(std::vector<f64>& rhs) {
        ++n_apply;
        const auto tap = Clock::now();
        struct Acc { double& a; Clock::time_point t; ~Acc() { a += ms_since(t); } } acc_{apply_ms, tap};
        if (m == 0) {
            for (core::Index j = 0; j < n; ++j) rhs[sz(j)] = -dscale[sz(j)] * rhs[sz(j)];
            return;
        }
        if (!transposed) {
            // M dy = r2 + A D r1;  dx = D (A' dy - r1).
            bwork.assign(sz(m), 0.0);
            for (core::Index i = 0; i < m; ++i) bwork[sz(i)] = rhs[sz(n + i)];
            for (core::Index j = 0; j < n; ++j) {
                const f64 t = dscale[sz(j)] * rhs[sz(j)];
                const auto& c = acols[sz(j)];
                for (std::size_t e = 0; e < c.rows.size(); ++e)
                    bwork[sz(c.rows[e])] += c.vals[e] * t;
            }
            {
                const auto tms = Clock::now();
                solve_m(bwork);
                msolve_ms += ms_since(tms);
            }
            for (core::Index j = 0; j < n; ++j) {
                f64 atd = 0.0;
                const auto& c = acols[sz(j)];
                for (std::size_t e = 0; e < c.rows.size(); ++e) atd += c.vals[e] * bwork[sz(c.rows[e])];
                rhs[sz(j)] = dscale[sz(j)] * (atd - rhs[sz(j)]);
            }
            for (core::Index i = 0; i < m; ++i) rhs[sz(n + i)] = bwork[sz(i)];
            return;
        }
        // N dx = A' H r2 - r1;  dy = H (r2 - A dx).
        bwork.assign(sz(n), 0.0);
        for (core::Index j = 0; j < n; ++j) bwork[sz(j)] = -rhs[sz(j)];
        for (core::Index i = 0; i < m; ++i) {
            const f64 t = dscale[sz(i)] * rhs[sz(n + i)];
            const auto& r = arows[sz(i)];
            for (std::size_t e = 0; e < r.rows.size(); ++e)
                bwork[sz(r.rows[e])] += r.vals[e] * t;
        }
        {
            const auto tms = Clock::now();
            solve_m(bwork);
            msolve_ms += ms_since(tms);
        }
        for (core::Index i = 0; i < m; ++i) {
            f64 ax = 0.0;
            const auto& r = arows[sz(i)];
            for (std::size_t e = 0; e < r.rows.size(); ++e) ax += r.vals[e] * bwork[sz(r.rows[e])];
            rhs[sz(n + i)] = dscale[sz(i)] * (rhs[sz(n + i)] - ax);
        }
        for (core::Index j = 0; j < n; ++j) rhs[sz(j)] = bwork[sz(j)];
    }

private:
    static core::Index static_core_index(std::size_t n) { return static_cast<core::Index>(n); }

    // Full matrix S = M + B_d S_d B_d^T.  With s = sqrt(S_d) and Y = M^-1 B_d,
    //   S^-1 = M^-1 - Y s C^-1 s Y^T,   C = I + s B_d^T M^-1 B_d s,
    // C symmetric with eigenvalues >= 1, so its Cholesky needs no pivoting
    // however large S_d is (a free column carries D = 1/rho).
    bool prepare_schur() {
        const core::Index p = static_core_index(dense_of.size());
        Y.assign(sz(P) * sz(p), 0.0);
        schur_l.assign(sz(p) * sz(p), 0.0);
        sroot.assign(sz(p), 0.0);
        // Y = M^-1 B_d, all p columns in one blocked solve: one solve with
        // a large factor is a pass over L at memory speed, so p separate
        // solves cost p passes (ex10, dual form: 128 columns, 3.8 s of a
        // 7.2 s Newton step).
        for (core::Index k = 0; k < p; ++k) {
            sroot[sz(k)] = std::sqrt(dscale[sz(dense_of[sz(k)])]);
            const auto& c = (*bcols)[sz(dense_of[sz(k)])];
            f64* yk = Y.data() + sz(k) * sz(P);
            for (std::size_t e = 0; e < c.rows.size(); ++e) yk[sz(c.rows[e])] = c.vals[e];
        }
        fac.solve_many(Y.data(), p);
        for (core::Index i = 0; i < p; ++i) {
            const auto& ci = (*bcols)[sz(dense_of[sz(i)])];
            for (core::Index k = 0; k < p; ++k) {
                f64 s = 0.0;
                const f64* yk = Y.data() + sz(k) * sz(P);
                for (std::size_t e = 0; e < ci.rows.size(); ++e)
                    s += ci.vals[e] * yk[sz(ci.rows[e])];
                schur_l[sz(i) * sz(p) + sz(k)] = s * sroot[sz(i)] * sroot[sz(k)];
            }
        }
        for (core::Index i = 0; i < p; ++i)
            for (core::Index k = i + 1; k < p; ++k) {
                const f64 s = 0.5 * (schur_l[sz(i) * sz(p) + sz(k)] + schur_l[sz(k) * sz(p) + sz(i)]);
                schur_l[sz(i) * sz(p) + sz(k)] = schur_l[sz(k) * sz(p) + sz(i)] = s;
            }
        for (core::Index i = 0; i < p; ++i) schur_l[sz(i) * sz(p) + sz(i)] += 1.0;
        for (core::Index j = 0; j < p; ++j) {
            f64 s = schur_l[sz(j) * sz(p) + sz(j)];
            for (core::Index k = 0; k < j; ++k) {
                const f64 l = schur_l[sz(j) * sz(p) + sz(k)];
                s -= l * l;
            }
            if (!std::isfinite(s)) return false;
            if (!(s > 1e-12)) { s = 1e-12; ++reg_pivots; }
            const f64 ljj = std::sqrt(s);
            schur_l[sz(j) * sz(p) + sz(j)] = ljj;
            for (core::Index i = j + 1; i < p; ++i) {
                f64 t = schur_l[sz(i) * sz(p) + sz(j)];
                for (core::Index k = 0; k < j; ++k)
                    t -= schur_l[sz(i) * sz(p) + sz(k)] * schur_l[sz(j) * sz(p) + sz(k)];
                schur_l[sz(i) * sz(p) + sz(j)] = t / ljj;
            }
        }
        return true;
    }

    void solve_schur(std::vector<f64>& w) const {
        const core::Index p = static_core_index(dense_of.size());
        for (core::Index j = 0; j < p; ++j) {
            f64 s = w[sz(j)];
            for (core::Index k = 0; k < j; ++k) s -= schur_l[sz(j) * sz(p) + sz(k)] * w[sz(k)];
            w[sz(j)] = s / schur_l[sz(j) * sz(p) + sz(j)];
        }
        for (core::Index j = p; j-- > 0; ) {
            f64 s = w[sz(j)];
            for (core::Index k = j + 1; k < p; ++k) s -= schur_l[sz(k) * sz(p) + sz(j)] * w[sz(k)];
            w[sz(j)] = s / schur_l[sz(j) * sz(p) + sz(j)];
        }
    }

    // Overwrites b with S^{-1} b, S = M_sparse + B_d S_d B_d^T.
    void solve_m(std::vector<f64>& b) {
        if (!schur) { fac.solve(b); return; }
        const core::Index p = static_core_index(dense_of.size());
        zwork = b;
        fac.solve(zwork);
        wwork.assign(sz(p), 0.0);
        for (core::Index k = 0; k < p; ++k) {
            const auto& c = (*bcols)[sz(dense_of[sz(k)])];
            f64 s = 0.0;
            for (std::size_t e = 0; e < c.rows.size(); ++e) s += c.vals[e] * zwork[sz(c.rows[e])];
            wwork[sz(k)] = s * sroot[sz(k)];
        }
        solve_schur(wwork);
        b = zwork;
        for (core::Index k = 0; k < p; ++k) {
            const f64 wk = wwork[sz(k)] * sroot[sz(k)];
            const f64* yk = Y.data() + sz(k) * sz(P);
            for (core::Index r = 0; r < P; ++r) b[sz(r)] -= yk[sz(r)] * wk;
        }
    }
};

// ---- the iteration --------------------------------------------------------
// orig_rel (optional, LP only): the ORIGINAL model's relative primal and dual
// residuals at the point `accept` was last called on.
Result run_ipm(const Core& k, const QpOptions& opts, Clock::time_point t0,
               const std::function<bool(const std::vector<f64>&, const std::vector<f64>&)>& accept,
               const std::function<void(f64&, f64&)>& orig_rel = nullptr) {
    const core::Index n = k.n, m = k.m, N = n + m;
    Result res;
    bool has_q = false;
    int stall_count = 0;
    f64 stall_rp = kInf, stall_rd = kInf, mu_start = -1.0;
    // LP: the iterate with the smallest original-model relative merit, which is
    // what comes back when the run ends stalled or out of budget.
    f64 best_merit = kInf;
    std::vector<f64> best_z, best_y;
    std::vector<f64> dist_hist, time_hist;   // per step: distance to the tolerances, ms (soft limit)
    std::vector<f64> z, lam, nu, y;
    f64 worst_solve_run = 0.0;          // max solve residual over the whole run (reported)
    for (const auto& col : k.qcol)
        for (const auto& e : col) has_q = has_q || e.second != 0.0;
    // A quadratic row makes the stationarity depend on x through J(x)'y and
    // the Hessian on y, so the equal-step argument below applies to it too.
    has_q = has_q || !k.quad.empty();
    // The caller's deadline, polled by the setup's orderings as well as by
    // the iteration.
    const std::function<bool()> stop_fn = [&]() {
        return (opts.time_limit_s > 0.0 && ms_since(t0) > opts.time_limit_s * 1000.0) ||
               core::cancel_requested(opts.cancel);
    };
    // Normal equations are the LP path.  A quadratic objective or a quadratic
    // row keeps the quasi-definite augmented factor.
    const bool lp = !has_q;
    bool use_ne = lp;
    NormalEq ne;
    try {
    if (opts.verbose) std::fprintf(stderr, "ipm setup: run_ipm entered at %.0f ms\n", ms_since(t0));
    if (lp) {
        ne.verbose = opts.verbose;
        ne.build(k, &stop_fn);
        if (opts.verbose) {
            const core::ProcessMemory pm = core::process_memory();
            std::fprintf(stderr, "ipm setup: normal equations analysed at %.0f ms (memory GB: rss %.2f peak %.2f virtual %.2f)\n",
                         ms_since(t0), pm.rss_gb, pm.peak_gb, pm.vm_gb);
        }
        if (ne.interrupted) throw DeadlineReached{};
        use_ne = ne.ok;
    }
    // Primal-dual regularisation as a PROXIMAL term (Friedlander & Orban,
    // Math. Prog. Comp. 4, 2012): the regularised system is the exact Newton
    // system of a proximal-point subproblem, so it is solved as is -- the
    // refinement below only removes round-off.  An earlier version refined
    // against the UNregularised matrix; on near-flat directions (QPLIB_9002:
    // Q entries down to 1e-11, no linear term) that re-solved a nearly
    // singular system and took a first step of ~1e9.
    // Fixed proximal regularisation.  Letting it follow mu down (rho = mu/100)
    // was tried, on the theory that a fixed 1e-8 term floors the residuals it
    // perturbs; measured, it was a net loss -- QPLIB_8906 went from matching
    // its published objective to 100x out -- so it stays fixed.
    // rho (primal) and the row term d (dual) share one value.  An LP tapers it
    // with mu (lp_prox below); prox_cur is the value in force.
    const f64 rho0 = 1e-8;
    f64 prox_cur = rho0;
    f64 reg_boost = 1.0;
    constexpr f64 kUlp = 2.220446049250313e-16;

    // Bound structure: component c in [0, n+m): x then w.
    auto bound_lo = [&](core::Index c) {
        return c < n ? k.l[sz(c)] : k.rl[sz(c - n)];
    };
    auto bound_hi = [&](core::Index c) {
        return c < n ? k.u[sz(c)] : k.ru[sz(c - n)];
    };
    std::vector<std::uint8_t> has_l(sz(N)), has_u(sz(N)), fixed_w(sz(m));
    for (core::Index c = 0; c < N; ++c) {
        has_l[sz(c)] = std::isfinite(bound_lo(c));
        has_u[sz(c)] = std::isfinite(bound_hi(c));
    }
    for (core::Index i = 0; i < m; ++i) {
        fixed_w[sz(i)] = k.rl[sz(i)] == k.ru[sz(i)];
        if (fixed_w[sz(i)]) { has_l[sz(n + i)] = 0; has_u[sz(n + i)] = 0; }
    }

    z.assign(sz(N), 0.0);
    lam.assign(sz(N), 0.0);
    nu.assign(sz(N), 0.0);
    y.assign(sz(m), 0.0);

    // ---- quadratic rows: supports and scratch ------------------------------
    // Per quadratic row t: its support (sorted columns), each Hessian entry's
    // local (row, col) in that support, Q_t x on the support (qix), and
    // 1/2 x'Q_t x (qval) at the current x.  eta_t is the row's weight in
    // the Lagrangian Hessian (sign: see quad_eval).
    const std::size_t nq = k.quad.size();
    std::vector<std::vector<core::Index>> qsup(nq), qlr(nq), qlc(nq);
    std::vector<std::vector<f64>> qix(nq);
    std::vector<f64> qval(nq, 0.0), eta(nq, 0.0);
    for (std::size_t t = 0; t < nq; ++t) {
        const auto& q = k.quad[t];
        auto& S = qsup[t];
        for (std::size_t e = 0; e < q.v.size(); ++e) { S.push_back(q.r[e]); S.push_back(q.c[e]); }
        std::sort(S.begin(), S.end());
        S.erase(std::unique(S.begin(), S.end()), S.end());
        for (std::size_t e = 0; e < q.v.size(); ++e) {
            qlr[t].push_back(static_cast<core::Index>(std::lower_bound(S.begin(), S.end(), q.r[e]) - S.begin()));
            qlc[t].push_back(static_cast<core::Index>(std::lower_bound(S.begin(), S.end(), q.c[e]) - S.begin()));
        }
        qix[t].assign(S.size(), 0.0);
    }
    // Adds each quadratic row's part to a row activity and a J'y at x: the
    // row value 1/2 x'Q_t x to ax, (Q_t x) y_row to aty -- the IPM's
    // Lagrangian sign (stationarity grad f - J'y - lam + nu = 0) -- and
    // refreshes qix/qval/eta for the next KKT fill.
    //
    // The Hessian weight: the Lagrangian Hessian is Q0 - sum_t y_t Q_t.  A
    // canonical row is upper-bounded only, so at dual feasibility
    // y_t = -nu_t <= 0 and -y_t Q_t is PSD.  In the infeasible iteration y_t
    // can transiently carry the wrong sign; the weight is then nu_t instead
    // (see below), which keeps the (1,1) block negative definite on the
    // row's support -- the system QUASI-DEFINITE, which the static-pivot
    // LDL' relies on (Vanderbei 1995) -- and the weight equals the exact
    // Hessian's whenever the sign is right, which it is near the solution.
    auto quad_eval = [&](const std::vector<f64>& xv, const std::vector<f64>& yv,
                         std::vector<f64>& ax_out, std::vector<f64>* aty_out) {
        for (std::size_t t = 0; t < nq; ++t) {
            const auto& q = k.quad[t];
            auto& w = qix[t];
            std::fill(w.begin(), w.end(), 0.0);
            f64 val = 0.0;
            for (std::size_t e = 0; e < q.v.size(); ++e) {
                const f64 xr = xv[sz(q.r[e])], xc = xv[sz(q.c[e])];
                if (q.r[e] == q.c[e]) {
                    w[sz(qlr[t][e])] += q.v[e] * xr;
                    val += 0.5 * q.v[e] * xr * xr;
                } else {
                    w[sz(qlr[t][e])] += q.v[e] * xc;
                    w[sz(qlc[t][e])] += q.v[e] * xr;
                    val += q.v[e] * xr * xc;
                }
            }
            qval[t] = val;
            ax_out[sz(q.row)] += val;
            const f64 yr = yv[sz(q.row)];
            // -y when it has the convex sign, else the slack's own
            // multiplier nu (> 0 in the interior; = -y at dual feasibility),
            // so the row never loses its curvature to a transient sign.
            eta[t] = -yr > 0.0 ? -yr : nu[sz(n + q.row)];
            if (aty_out)
                for (std::size_t s = 0; s < qsup[t].size(); ++s)
                    (*aty_out)[sz(qsup[t][s])] += w[s] * yr;
        }
    };

    // ---- KKT pattern (upper triangle): x block (Q upper + diag), then A' ----
    // With quadratic rows the x block also holds every Q_t's upper triangle
    // (the Lagrangian Hessian) and column n+i holds row i's Jacobian support
    // supp(a_i) u supp(Q_i).  The extra entries are APPENDED after the QP
    // pattern of each column, so a QP's pattern and values are exactly what
    // they were before quadratic rows existed.
    la::SymCsc K;
    std::vector<core::Offset> diag_slot(sz(N));
    std::vector<std::vector<core::Offset>> qpos(nq), jpos(nq);   // K slots: Q_t entries / J_t support
    std::vector<std::int8_t> sign;
    la::Ldlt fac;
    bool aug_built = false;
    bool aug_fits = true;
    const auto too_large = [&]() {
        return std::runtime_error("interior point: no Newton-system factor fits in memory (" +
                                  std::to_string(fac.factor_bytes() / 1000000) + " MB needed)");
    };
    f64 aug_flop_cap = 0.0;
    std::function<void()> analyse_aug;
    auto build_aug = [&]() {
    if (aug_built) return;
    aug_built = true;
    K.n = N;
    K.col_ptr.assign(sz(N) + 1, 0);
    {
        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> quad_by_col(sz(n));
        for (std::size_t t = 0; t < nq; ++t) {
            qpos[t].assign(k.quad[t].v.size(), 0);
            for (std::size_t e = 0; e < k.quad[t].v.size(); ++e)
                quad_by_col[sz(k.quad[t].c[e])].push_back({t, e});
        }
        std::vector<core::Offset> slot_of(sz(N), -1);   // row -> slot, current column only
        // Column j (x): Q upper entries (i <= j) + quad extras + diagonal.
        for (core::Index j = 0; j < n; ++j) {
            bool diag = false;
            const auto start = static_cast<core::Offset>(K.row_idx.size());
            for (const auto& [i, v] : k.qcol[sz(j)]) {
                (void)v;
                K.row_idx.push_back(i);
                slot_of[sz(i)] = static_cast<core::Offset>(K.row_idx.size()) - 1;
                if (i == j) { diag = true; diag_slot[sz(j)] = static_cast<core::Offset>(K.row_idx.size()) - 1; }
            }
            if (!quad_by_col[sz(j)].empty()) {
                std::vector<core::Index> extra;
                for (const auto& [t, e] : quad_by_col[sz(j)])
                    if (slot_of[sz(k.quad[t].r[e])] < 0) extra.push_back(k.quad[t].r[e]);
                std::sort(extra.begin(), extra.end());
                extra.erase(std::unique(extra.begin(), extra.end()), extra.end());
                for (core::Index i : extra) {
                    K.row_idx.push_back(i);
                    slot_of[sz(i)] = static_cast<core::Offset>(K.row_idx.size()) - 1;
                    if (i == j) { diag = true; diag_slot[sz(j)] = static_cast<core::Offset>(K.row_idx.size()) - 1; }
                }
                for (const auto& [t, e] : quad_by_col[sz(j)]) qpos[t][e] = slot_of[sz(k.quad[t].r[e])];
            }
            if (!diag) { K.row_idx.push_back(j); diag_slot[sz(j)] = static_cast<core::Offset>(K.row_idx.size()) - 1; }
            for (auto t = start; t < static_cast<core::Offset>(K.row_idx.size()); ++t) slot_of[sz(K.row_idx[sz(t)])] = -1;
            K.col_ptr[sz(j) + 1] = static_cast<core::Offset>(K.row_idx.size());
        }
        // Column n+i (y): A row i entries (x indices) + quad support extras + diagonal.
        std::vector<std::size_t> quad_of_row(sz(m), nq);
        for (std::size_t t = 0; t < nq; ++t) quad_of_row[sz(k.quad[t].row)] = t;
        const auto& rp = k.A.pattern.row_ptr();
        const auto& ci = k.A.pattern.col_idx();
        for (core::Index i = 0; i < m; ++i) {
            const auto start = static_cast<core::Offset>(K.row_idx.size());
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
                K.row_idx.push_back(ci[sz(t)]);
                slot_of[sz(ci[sz(t)])] = static_cast<core::Offset>(K.row_idx.size()) - 1;
            }
            if (const std::size_t t = quad_of_row[sz(i)]; t < nq) {
                for (core::Index j : qsup[t])
                    if (slot_of[sz(j)] < 0) {
                        K.row_idx.push_back(j);
                        slot_of[sz(j)] = static_cast<core::Offset>(K.row_idx.size()) - 1;
                    }
                for (core::Index j : qsup[t]) jpos[t].push_back(slot_of[sz(j)]);
            }
            for (auto t = start; t < static_cast<core::Offset>(K.row_idx.size()); ++t) slot_of[sz(K.row_idx[sz(t)])] = -1;
            K.row_idx.push_back(n + i);
            diag_slot[sz(n + i)] = static_cast<core::Offset>(K.row_idx.size()) - 1;
            K.col_ptr[sz(n + i) + 1] = static_cast<core::Offset>(K.row_idx.size());
        }
        K.vals.assign(K.row_idx.size(), 0.0);
    }
    sign.assign(sz(N), 1);
    for (core::Index j = 0; j < n; ++j) sign[sz(j)] = -1;
    analyse_aug();
    };
    // The analysis of K, capped at aug_flop_cap multiply-adds when the
    // augmented system is only being compared with the normal equations.
    analyse_aug = [&]() {
    NormalEq::budget_factor(fac, NormalEq::pattern_bytes(K.row_idx.size()), &stop_fn);
    {
        la::LdltOptions o = fac.options();
        o.max_flops = aug_flop_cap;
        fac.set_options(o);
    }
    fac.analyze(K);
    if (fac.interrupted()) throw DeadlineReached{};
    aug_fits = fac.fits();
    if (opts.verbose)
        std::fprintf(stderr, "ipm kkt n=%lld nnzL=%lld supernodes=%lld flops=%.3e fits=%d (analysed at %.0f ms)\n",
                     static_cast<long long>(K.n), static_cast<long long>(fac.nnz_l()),
                     static_cast<long long>(fac.supernodes()), fac.flops(), static_cast<int>(aug_fits),
                     ms_since(t0));
    };
    if (!use_ne) {
        build_aug();
        if (!aug_fits) throw too_large();
    } else if (ne.fac.flops() > 2e9 &&
               ne.dense_candidates > static_cast<int>(ne.dense_of.size())) {
        // The normal-equation factor is big enough for the choice to
        // matter: keep the augmented system when it is the cheaper one.
        // The augmented factor wins only below half the normal equations'
        // cost, so its ordering is abandoned past that (AmdCaps).
        //
        // What the augmented ordering can do that neither normal-equation
        // form does is keep LONG columns of B out of the fill: eliminating
        // every column node first IS the primal form, every row node first
        // the dual form, and both forms already take their long columns
        // (>= kDenseNnz entries) out into a Schur update when that is
        // cheaper.  So the comparison is only made when the chosen form
        // still carries long columns in its sparse part.  Without that its
        // ordering of the (n+m)-node system bought nothing and cost more
        // than the solve's setup: scpm1 101 s (505k nodes; the 5000-row
        // normal equations are dense, 1.3 s a factorization), s82 21 s.
        const f64 ne_cost = ne.iteration_cost();
        aug_flop_cap = 0.5 * ne_cost;
        build_aug();
        use_ne = !aug_fits || ne_cost <= 2.0 * fac.flops();
        if (!use_ne) ne.release();
    }
    if (use_ne && opts.verbose) {
        std::fprintf(stderr,
                     "ipm normal-eq %s order=%lld nnzM=%lld nnzL=%lld dense_cols=%d schur=%d flops=%.3e%s\n",
                     ne.transposed ? "dual-form (A'HA)" : "primal-form (ADA')",
                     static_cast<long long>(ne.P),
                     static_cast<long long>(ne.m == 0 ? 0 : ne.M.row_idx.size()),
                     static_cast<long long>(ne.m == 0 ? 0 : ne.fac.nnz_l()),
                     ne.dense_candidates, static_cast<int>(ne.dense_of.size()),
                     ne.m == 0 ? 0.0 : ne.fac.flops(), ne.dense_mode ? " (dense M)" : "");
    }

    // Fill K's values for the current T.  Two diagonals are kept: the
    // REGULARISED one that is factorized, and the UNREGULARISED one that the
    // refinement targets -- the factor is then a preconditioner for the true
    // Newton system, which is how a regularised IPM recovers the accuracy the
    // proximal term costs it (Saunders; Friedlander & Orban 2012).
    std::vector<f64> tdiag(sz(N)), kdiag_reg(sz(N)), kdiag_true(sz(N));
    auto fill_K = [&]() {
        core::Offset pos = 0;
        // Q = 0 (LP barrier): qcol is empty; skip Hessian value writes and
        // only clear the diagonal / structural slots the pattern reserved.
        for (core::Index j = 0; j < n; ++j) {
            if (has_q) {
                for (const auto& [i, v] : k.qcol[sz(j)])
                    K.vals[sz(pos++)] = -v;
            }
            // explicit diagonal slot, and any quad-only Hessian slots
            while (pos < K.col_ptr[sz(j) + 1]) K.vals[sz(pos++)] = 0.0;
        }
        const auto& rp = k.A.pattern.row_ptr();
        for (core::Index i = 0; i < m; ++i) {
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) K.vals[sz(pos++)] = k.A.vals[sz(t)];
            while (pos < K.col_ptr[sz(n + i) + 1]) K.vals[sz(pos++)] = 0.0;
        }
        // Quadratic rows: -eta_t Q_t into the Hessian block, Q_t x into the
        // Jacobian row (row t's J = a_t + Q_t x at the current x).
        for (std::size_t t = 0; t < nq; ++t) {
            if (eta[t] != 0.0)
                for (std::size_t e = 0; e < k.quad[t].v.size(); ++e)
                    K.vals[sz(qpos[t][e])] -= eta[t] * k.quad[t].v[e];
            for (std::size_t s2 = 0; s2 < jpos[t].size(); ++s2) K.vals[sz(jpos[t][s2])] += qix[t][s2];
        }
        for (core::Index j = 0; j < n; ++j) {
            const f64 base = K.vals[sz(diag_slot[sz(j)])];
            kdiag_true[sz(j)] = base - tdiag[sz(j)];
            kdiag_reg[sz(j)] = kdiag_true[sz(j)] - prox_cur * reg_boost;
            K.vals[sz(diag_slot[sz(j)])] = kdiag_reg[sz(j)];
        }
        for (core::Index i = 0; i < m; ++i) {
            const core::Index c = n + i;
            kdiag_true[sz(c)] = fixed_w[sz(i)] ? 0.0 : 1.0 / tdiag[sz(c)];
            kdiag_reg[sz(c)] = kdiag_true[sz(c)] + prox_cur * reg_boost;
            K.vals[sz(diag_slot[sz(c)])] = kdiag_reg[sz(c)];
        }
    };
    // K v with the UNREGULARISED diagonal: the system refinement targets.
    // Parallelized with fixed-size column chunks and thread-local accumulators.
    auto K_true_mul = [&](const std::vector<f64>& v, std::vector<f64>& out) {
        out.assign(sz(N), 0.0);
        core::ThreadPool& pool = core::global_pool();
        const int nthreads = pool.size();
        const core::Offset chunk_size = 16384;  // Fixed chunk independent of thread count
        const core::Offset nchunks = (N + chunk_size - 1) / chunk_size;

        // Thread-local output buffers
        std::vector<std::vector<f64>> thread_out(nthreads, std::vector<f64>(sz(N), 0.0));

        // Process chunks of columns in parallel
        auto body = [&](core::Offset chunk, int worker) {
            const core::Index col_start = chunk * chunk_size;
            const core::Index col_end = col_start + chunk_size < N ? col_start + chunk_size : N;
            auto& out_local = thread_out[sz(worker)];

            for (core::Index j = col_start; j < col_end; ++j)
                for (auto t = K.col_ptr[sz(j)]; t < K.col_ptr[sz(j) + 1]; ++t) {
                    const core::Index i = K.row_idx[sz(t)];
                    f64 val = K.vals[sz(t)];
                    if (i == j) val = kdiag_true[sz(j)];
                    out_local[sz(i)] += val * v[sz(j)];
                    if (i != j) out_local[sz(j)] += val * v[sz(i)];
                }
        };

        if (nchunks <= 1 || pool.size() == 1) {
            // Serial path: process all chunks serially with fixed order
            for (core::Offset chunk = 0; chunk < nchunks; ++chunk) body(chunk, 0);
        } else {
            pool.parallel_for(nchunks, body);
        }

        // Merge thread-local buffers into output in deterministic order
        for (int w = 0; w < nthreads; ++w)
            for (core::Index i = 0; i < N; ++i)
                out[sz(i)] += thread_out[sz(w)][sz(i)];
    };
    f64 fac_ms = 0.0, solve_ms = 0.0;   // reported under --verbose
    int gz_tried = 0, gz_kept = 0;      // Gondzio correctors solved / kept
    f64 worst_solve_rel = 0.0;          // max ||K_true x - b|| / (1+||b||) this iteration
    // Solve K_true x = rhs with the REGULARISED factor as a preconditioner
    // (FGMRES; qp_krylov.cpp has why plain refinement was not enough).
    // A fixed 1e-12.  Tying it to mu (clamp(1e-4 mu, 1e-12, 1e-6), inexact
    // Newton) was tried: 5-25% faster on the instances it already proved,
    // but 10034 went from stalling at iteration 101 with primal 5.4e-5 to
    // the 300-iteration limit at 6.6e-3.  Accuracy before speed.
    const f64 solve_tol = 1e-12;
    // Unregularised augmented product for Q = 0, without materialising K.
    // (Refining against the REGULARISED system instead was measured: 43%
    // fewer factor applications on Netlib, but s100 took 72 Newton steps
    // instead of 62 for the same time, so the true system stays the target.)
    auto K_true_mul_lp = [&](const std::vector<f64>& v, std::vector<f64>& out) {
        out.assign(sz(N), 0.0);
        for (core::Index j = 0; j < n; ++j) out[sz(j)] = -tdiag[sz(j)] * v[sz(j)];
        for (core::Index i = 0; i < m; ++i) {
            const f64 tw = tdiag[sz(n + i)];
            const f64 g = (!fixed_w[sz(i)] && tw > 0.0) ? 1.0 / tw : 0.0;
            out[sz(n + i)] = g * v[sz(n + i)];
        }
        for (core::Index j = 0; j < n; ++j) {
            const f64 dx = v[sz(j)];
            const auto& col = ne.acols[sz(j)];
            for (std::size_t e = 0; e < col.rows.size(); ++e) {
                const core::Index i = col.rows[e];
                const f64 a = col.vals[e];
                out[sz(n + i)] += a * dx;
                out[sz(j)] += a * v[sz(n + i)];
            }
        }
    };
    auto solve_refined = [&](std::vector<f64>& rhs, const char* phase = nullptr) {
        const auto ts0 = Clock::now();
        const f64 rel = use_ne
            ? qpc::fgmres_solve([&](std::vector<f64>& b) { ne.apply(b); }, K_true_mul_lp, rhs,
                                20, solve_tol, phase)
            : qpc::fgmres_solve(fac, K_true_mul, rhs, 20, solve_tol, phase);
        worst_solve_rel = std::max(worst_solve_rel, rel);
        worst_solve_run = std::max(worst_solve_run, rel);
        solve_ms += ms_since(ts0);
    };
    // LP proximal terms taper from 1e-8 toward 1e-10 as mu falls through
    // [1e-4, 1e-8], so the regularised Newton system approaches the true one
    // near termination.  QP keeps the fixed 1e-8 (following mu was a net loss
    // on QPLIB_8906).
    auto lp_prox = [](f64 mu_now) {
        const f64 prox_hi_v = 1e-8;
        const f64 prox_lo_v = 1e-10;
        if (!(mu_now > 0.0) || mu_now >= 1e-4) return prox_hi_v;
        if (mu_now <= 1e-8) return prox_lo_v;
        const f64 u = (std::log10(mu_now) + 8.0) / 4.0;
        return std::pow(10.0, std::log10(prox_lo_v) +
                                  u * (std::log10(prox_hi_v) - std::log10(prox_lo_v)));
    };
    // Factor the regularised Newton matrix for the current T: M (normal
    // equations) or the augmented K (whose values fill_K wrote with prox_cur).
    auto factor_newton = [&](f64 prox) {
        if (!use_ne) return fac.factorize(K, sign, 1e-12);
        ne.assemble(tdiag, prox * reg_boost, prox * reg_boost, fixed_w);
        return ne.factorize();
    };
    // The normal equations broke down (non-finite factor, or the Krylov
    // refinement could not reach a usable residual): continue on the
    // augmented quasi-definite LDL', one way.
    constexpr f64 kNeSolveFail = 1e-6;
    auto fall_back_to_aug = [&](const char* why) {
        build_aug();
        if (fac.over_cost()) {
            // Analysed only for the comparison, and abandoned there.
            aug_flop_cap = 0.0;
            analyse_aug();
        }
        if (!aug_fits) throw too_large();
        use_ne = false;
        ne.release();
        worst_solve_rel = 0.0;
        reg_boost = 1.0;
        if (opts.verbose)
            std::fprintf(stderr, "ipm normal equations abandoned (%s); augmented LDL' from here\n", why);
    };


    // ---- starting point (Mehrotra 1992, adapted to bounded variables) ----
    //
    // The primal-dual pair comes from ONE solve of the KKT system with unit
    // bound weights (T = I), which is the least-squares point that balances
    // the objective against the constraints -- not a guess.  It is then
    // pushed strictly inside the box, and the bound multipliers are seeded
    // from the reduced costs there so the starting dual residual is small.
    //
    // The classical Mehrotra shift moves each slack independently; a bounded
    // variable has TWO slacks tied to one z, so the shift here is on the
    // multipliers only, with z placed by the box geometry.  Offsets are
    // relative to each bound's magnitude: a fixed offset of 1 sits on the
    // boundary when the bound is 1e6.
    auto interior = [&](core::Index c, f64 v) {
        const f64 a = bound_lo(c), b = bound_hi(c);
        const f64 ta = std::isfinite(a) ? std::max(1.0, 1e-2 * std::fabs(a)) : 0.0;
        const f64 tb = std::isfinite(b) ? std::max(1.0, 1e-2 * std::fabs(b)) : 0.0;
        if (std::isfinite(a) && std::isfinite(b)) {
            const f64 w = b - a;
            if (w <= 2.0 * std::min(ta, tb)) return a + 0.5 * w;
            return std::clamp(v, a + std::min(ta, 0.25 * w), b - std::min(tb, 0.25 * w));
        }
        if (std::isfinite(a)) return std::max(v, a + ta);
        if (std::isfinite(b)) return std::min(v, b - tb);
        return v;
    };
    {
        std::vector<f64> ls(sz(N), 0.0);
        std::fill(tdiag.begin(), tdiag.end(), 1.0);
        if (!use_ne) fill_K();
        bool start_ok = factor_newton(rho0);
        if (!start_ok && use_ne) {
            fall_back_to_aug("non-finite factor at the starting point");
            fill_K();
            start_ok = factor_newton(rho0);
        }
        if (start_ok) {
            // Stationarity residual at (0, 0) is c; the rows contribute their
            // fixed right-hand side.
            auto start_solve = [&]() {
                for (core::Index j = 0; j < n; ++j) ls[sz(j)] = k.c[sz(j)];
                for (core::Index i = 0; i < m; ++i)
                    ls[sz(n + i)] = fixed_w[sz(i)] ? k.rl[sz(i)] : 0.0;
                solve_refined(ls, "startup");
            };
            start_solve();
            if (use_ne && !(worst_solve_rel <= kNeSolveFail)) {
                fall_back_to_aug("starting-point solve did not converge");
                fill_K();
                if (factor_newton(rho0)) start_solve();
            }
            bool ok = true;
            for (core::Index c = 0; c < N && ok; ++c) ok = std::isfinite(ls[sz(c)]);
            if (!ok) std::fill(ls.begin(), ls.end(), 0.0);
        }
        worst_solve_rel = 0.0;
        for (core::Index j = 0; j < n; ++j) z[sz(j)] = interior(j, ls[sz(j)]);
        for (core::Index i = 0; i < m; ++i) y[sz(i)] = ls[sz(n + i)];

        std::vector<f64> ax(sz(m), 0.0), aty(sz(n), 0.0), q0;
        const auto& rp0 = k.A.pattern.row_ptr();
        const auto& ci0 = k.A.pattern.col_idx();
        for (core::Index i = 0; i < m; ++i)
            for (auto t = rp0[sz(i)]; t < rp0[sz(i) + 1]; ++t) {
                ax[sz(i)] += k.A.vals[sz(t)] * z[sz(ci0[sz(t)])];
                aty[sz(ci0[sz(t)])] += k.A.vals[sz(t)] * y[sz(i)];
            }
        if (nq > 0) {
            // Quadratic rows' activity at the starting x, so w starts at the
            // row's value pushed inside, as for a linear row.
            const std::vector<f64> xs(z.begin(), z.begin() + n);
            quad_eval(xs, y, ax, &aty);
        }
        for (core::Index i = 0; i < m; ++i)
            z[sz(n + i)] = fixed_w[sz(i)] ? k.rl[sz(i)] : interior(n + i, ax[sz(i)]);

        std::vector<f64> x0(z.begin(), z.begin() + n);
        q_mul(k, x0, q0);
        f64 rmax = 0.0;
        for (core::Index j = 0; j < n; ++j)
            rmax = std::max(rmax, std::fabs(q0[sz(j)] + k.c[sz(j)] - aty[sz(j)]));
        // Keep mu0 commensurate with starting infeasibility. A hard floor of
        // 1.0 inflated early mu on well-scaled LPs (extra Newton steps).
        const f64 floor_ = (nq == 0) ? std::max(1e-2, rmax)
                                    : std::max(1.0, rmax);
        for (core::Index j = 0; j < n; ++j) {
            const f64 r = q0[sz(j)] + k.c[sz(j)] - aty[sz(j)];   // wants lam - nu = r
            if (has_l[sz(j)]) lam[sz(j)] = std::max(floor_, has_u[sz(j)] ? std::max(r, 0.0) : r);
            if (has_u[sz(j)]) nu[sz(j)] = std::max(floor_, has_l[sz(j)] ? std::max(-r, 0.0) : -r);
        }
        for (core::Index i = 0; i < m; ++i) {
            const core::Index c = n + i;
            const f64 r = -y[sz(i)];                             // wants lam - nu = -y
            if (has_l[sz(c)]) lam[sz(c)] = std::max(floor_, has_u[sz(c)] ? std::max(r, 0.0) : r);
            if (has_u[sz(c)]) nu[sz(c)] = std::max(floor_, has_l[sz(c)] ? std::max(-r, 0.0) : -r);
        }
        // Centre: raise every product to at least the average.
        f64 mu0 = 0.0;
        int np = 0;
        for (core::Index c = 0; c < N; ++c) {
            if (has_l[sz(c)]) { mu0 += (z[sz(c)] - bound_lo(c)) * lam[sz(c)]; ++np; }
            if (has_u[sz(c)]) { mu0 += (bound_hi(c) - z[sz(c)]) * nu[sz(c)]; ++np; }
        }
        mu0 = np > 0 ? mu0 / np : 1.0;
        for (core::Index c = 0; c < N; ++c) {
            if (has_l[sz(c)]) lam[sz(c)] = std::max(lam[sz(c)], mu0 / (z[sz(c)] - bound_lo(c)));
            if (has_u[sz(c)]) nu[sz(c)] = std::max(nu[sz(c)], mu0 / (bound_hi(c) - z[sz(c)]));
        }
        // A quadratic row's multiplier from the least-squares solve above
        // carries no curvature information (that solve saw J = A at x = 0),
        // and a wrong sign would start the iteration with no Hessian term:
        // on a linear objective over free columns the first Newton step is
        // then limited only by the 1e-8 proximal term (measured: |rp| 1e31
        // after one step on min -x-y s.t. (x-1)^2 + y^2 <= 1).  Start it
        // dual feasible for its slack instead: y = -nu (rdw = 0), eta = nu.
        for (std::size_t t = 0; t < nq; ++t) {
            const core::Index i = k.quad[t].row;
            y[sz(i)] = lam[sz(n + i)] - nu[sz(n + i)];
        }
    }

    std::vector<f64> qx, rdx(sz(n)), rdw(sz(m)), rp(sz(m));
    std::vector<f64> dz(sz(N)), dlam(sz(N)), dnu(sz(N)), dy(sz(m)), rhs(sz(N));
    // Second-order term of the quadratic rows in the corrector (zero for a
    // linear row, and zero throughout for a QP).
    std::vector<f64> soc(sz(m), 0.0);
    std::vector<f64> sl(sz(N)), su(sz(N)), tl(sz(N)), tu(sz(N));
    // The caller's cap, but never more than 300: an interior point that has
    // not converged in 300 Newton steps is not going to, and polishing is
    // the better use of the time.
    const int max_it = static_cast<int>(std::min<std::uint64_t>(300, std::max<std::uint64_t>(1, opts.max_iterations)));
    for (int it = 0; it < max_it; ++it) {
        const auto it_t0 = Clock::now();
        const f64 fac_at_it = fac_ms, sol_at_it = solve_ms;
        if (opts.time_limit_s > 0.0 && ms_since(t0) > opts.time_limit_s * 1000.0) {
            res.reason = "time limit";
            res.stopped_on_time = true;
            break;
        }
        if (core::cancel_requested(opts.cancel)) {
            res.reason = "cancelled (concurrent race lost)";
            break;
        }
        // Counted once it runs: a step stopped before any work is none.
        res.iterations = it + 1;
        // Residuals.
        std::vector<f64> x(z.begin(), z.begin() + n);
        q_mul(k, x, qx);
        std::vector<f64> aty(sz(n), 0.0), ax(sz(m), 0.0);
        {
            const auto& rpp = k.A.pattern.row_ptr();
            const auto& ci = k.A.pattern.col_idx();
            for (core::Index i = 0; i < m; ++i)
                for (auto t = rpp[sz(i)]; t < rpp[sz(i) + 1]; ++t) {
                    ax[sz(i)] += k.A.vals[sz(t)] * x[sz(ci[sz(t)])];
                    aty[sz(ci[sz(t)])] += k.A.vals[sz(t)] * y[sz(i)];
                }
        }
        if (nq > 0) {
            quad_eval(x, y, ax, &aty);   // g(x), J(x)'y, and the next K's Q_t x, eta
            // Slack reset (Byrd, Hribar & Nocedal, SIAM J. Optim. 9, 1999,
            // s.3): where a quadratic row's true value is BELOW its slack
            // variable, move the slack down to it.  w stays strictly inside
            // (g < w < ru), the row's residual becomes 0 instead of being
            // left for later Newton steps, and ru - w only grows.
            // Resetting in both directions (w = g whenever g < ru) was tried:
            // 2784 and 2456 went from proved to stalled (stationarity 4e-7,
            // 5e-8), 2468 unchanged -- shrinking a slack under its multiplier
            // wrecks centrality.  One direction only.
            for (std::size_t t = 0; t < nq; ++t) {
                const core::Index i = k.quad[t].row;
                if (ax[sz(i)] < z[sz(n + i)]) z[sz(n + i)] = ax[sz(i)];
            }
        }
        for (core::Index j = 0; j < n; ++j)
            rdx[sz(j)] = qx[sz(j)] + k.c[sz(j)] - aty[sz(j)] - lam[sz(j)] + nu[sz(j)];
        for (core::Index i = 0; i < m; ++i) {
            rdw[sz(i)] = fixed_w[sz(i)] ? 0.0 : y[sz(i)] - lam[sz(n + i)] + nu[sz(n + i)];
            rp[sz(i)] = ax[sz(i)] - z[sz(n + i)];
        }
        f64 mu = 0.0;
        int pairs = 0;
        for (core::Index c = 0; c < N; ++c) {
            // z - lo is computed, not stored: once the slack falls below the
            // rounding unit of the bound it can come out as exactly 0 and
            // make lam/s infinite.  A slack is never allowed below one ulp of
            // its bound (10038 hit this at mu ~ 1e-22).
            sl[sz(c)] = has_l[sz(c)] ? std::max(z[sz(c)] - bound_lo(c), kUlp * std::max(1.0, std::abs(bound_lo(c)))) : 0.0;
            su[sz(c)] = has_u[sz(c)] ? std::max(bound_hi(c) - z[sz(c)], kUlp * std::max(1.0, std::abs(bound_hi(c)))) : 0.0;
            if (has_l[sz(c)]) { mu += sl[sz(c)] * lam[sz(c)]; ++pairs; }
            if (has_u[sz(c)]) { mu += su[sz(c)] * nu[sz(c)]; ++pairs; }
        }
        mu = pairs > 0 ? mu / pairs : 0.0;

        // The claim is the caller's: stop as soon as the original-units
        // check passes.  Checked every iteration -- it is O(nnz).
        if (accept(x, y)) {
            res.converged = true;
            res.reason = "interior point: original-units KKT satisfied";
            break;
        }

        // LP barrier: stop and stall on the ORIGINAL model's RELATIVE
        // residuals.  The claim above stays absolute and only advisory here
        // (crossover and the checker decide what the point proves), but the
        // scaled residuals this loop minimises can sit at their floor while
        // the caller's units are converged (d2q06c: |rd| 5e-7 scaled, 3e-9
        // relative) or the reverse, and a mu floor measured against mu_start
        // says nothing about either.  The complementarity term is the usual
        // relative gap, pairs*mu / (1 + |c'x|).
        f64 orig_p = kInf, orig_d = kInf, orig_c = kInf;
        const bool lp_rel = lp && static_cast<bool>(orig_rel);
        if (lp_rel) {
            orig_rel(orig_p, orig_d);
            f64 pobj = 0.0;
            for (core::Index j = 0; j < n; ++j) pobj += k.c[sz(j)] * x[sz(j)];
            orig_c = mu * static_cast<f64>(pairs) / (1.0 + std::fabs(pobj));
            const f64 merit = std::max({orig_p, orig_d, orig_c});
            if (std::isfinite(merit) && merit < best_merit) {
                best_merit = merit;
                best_z = z;
                best_y = y;
            }
            // A margin under the tolerances: crossover starts from this point.
            constexpr f64 kRelMargin = 0.1;
            if (orig_p <= kRelMargin * opts.feas_tol && orig_d <= kRelMargin * opts.stationarity_tol &&
                orig_c <= opts.gap_tol) {
                res.converged = true;
                res.reason = "interior point: original-model relative KKT satisfied";
                break;
            }
            // Past the soft limit: continue only while on course to finish.
            // dist > 1 is how far the test above is from passing (as a
            // factor); over the last kRateWin steps it shrank by rho per
            // step, so about log(dist)/log(rho) more steps of the recent
            // length remain.  Projected beyond the hard limit (with a 25%
            // margin), or not shrinking at all, the run stops here: the
            // point is then handed on as an unconverged one.
            const f64 dist = std::max({orig_p / (kRelMargin * opts.feas_tol),
                                       orig_d / (kRelMargin * opts.stationarity_tol), orig_c / opts.gap_tol});
            dist_hist.push_back(std::isfinite(dist) ? dist : kInf);
            time_hist.push_back(ms_since(t0));
            if (opts.soft_time_limit_s > 0.0 && opts.time_limit_s > opts.soft_time_limit_s &&
                time_hist.back() > opts.soft_time_limit_s * 1000.0) {
                constexpr std::size_t kRateWin = 5;
                bool on_course = false;
                f64 projected_ms = kInf;
                if (dist_hist.size() > kRateWin) {
                    const std::size_t a = dist_hist.size() - 1 - kRateWin, b = dist_hist.size() - 1;
                    const f64 d0 = dist_hist[a], d1 = dist_hist[b];
                    if (std::isfinite(d0) && std::isfinite(d1) && d1 > 0.0 && d0 > d1) {
                        const f64 log_rho = std::log(d0 / d1) / static_cast<f64>(kRateWin);
                        const f64 steps = std::max(0.0, std::log(d1)) / log_rho;
                        const f64 step_ms = (time_hist[b] - time_hist[a]) / static_cast<f64>(kRateWin);
                        projected_ms = time_hist[b] + 1.25 * steps * step_ms;
                        on_course = projected_ms <= opts.time_limit_s * 1000.0;
                    }
                }
                if (opts.verbose)
                    std::fprintf(stderr, "ipm past soft limit: distance %.2e, projected end %.0f ms of %.0f -> %s\n",
                                 dist, projected_ms, opts.time_limit_s * 1000.0, on_course ? "continue" : "stop");
                if (!on_course) {
                    res.reason = "time limit (not on course to converge within the budget)";
                    res.stopped_on_time = true;
                    break;
                }
            }
        }

        // Stalled: complementarity at its floor and neither residual improving
        // by 10% (against the best so far, not the last: iterate-to-iterate
        // noise otherwise resets the count) over 5 iterations.  Nothing
        // further comes out of Newton steps here (8500 repeated one iterate
        // from step 35 to its time limit at 98), and polishing is what can
        // still certify, so stop.  An LP reads both off the original model's
        // relative residuals; a QP off the scaled ones and a mu floor measured
        // from where mu started ("at the floor" relative to mu_start: an
        // absolute 1e-14 never fired on 8567, mu0 2.6e5).
        {
            f64 rpn = 0.0, rdn = 0.0;
            for (f64 v : rp) rpn = std::max(rpn, std::fabs(v));
            for (f64 v : rdx) rdn = std::max(rdn, std::fabs(v));
            for (f64 v : rdw) rdn = std::max(rdn, std::fabs(v));
            const f64 sp = lp_rel ? orig_p : rpn, sd = lp_rel ? orig_d : rdn;
            const bool better = sp < 0.9 * stall_rp || sd < 0.9 * stall_rd;
            stall_rp = std::min(stall_rp, sp);
            stall_rd = std::min(stall_rd, sd);
            if (mu_start < 0.0) mu_start = mu;
            const bool at_floor = lp_rel ? orig_c <= opts.gap_tol
                                         : mu <= (nq > 0 ? 1e-20 : 1e-14) * std::max(1.0, mu_start);
            if (at_floor && !better) {
                if (++stall_count >= 5) {
                    res.reason = lp_rel ? "stalled: original-model relative residuals stopped improving"
                                        : "stalled at the mu floor";
                    break;
                }
            } else {
                stall_count = 0;
            }
        }

        for (core::Index c = 0; c < N; ++c)
            tdiag[sz(c)] = (has_l[sz(c)] ? lam[sz(c)] / sl[sz(c)] : 0.0) +
                           (has_u[sz(c)] ? nu[sz(c)] / su[sz(c)] : 0.0);
        // The proximal term in force: an LP tapers it with mu (B3), a QP keeps 1e-8.
        const f64 prox = lp ? lp_prox(mu) : rho0;
        prox_cur = prox;
        if (!use_ne) fill_K();
        // A non-finite pivot near the optimum comes from complementarity
        // ratios lam/s spanning ~1e20: more regularisation keeps the pivots
        // bounded, and refinement against the true system restores accuracy.
        // Same budget as the non-finite-direction guard below.
        const auto tf0 = Clock::now();
        const bool fac_ok = factor_newton(prox);
        fac_ms += ms_since(tf0);
        // Developer hook: FORGE_DUMP_KKT=<path> writes the first in-loop system
        // (upper CSC + pivot signs; the normal-equation matrix M for an LP on
        // that path) for apps/forge_ldlt_bench, so the factorization can be
        // timed and checked in isolation on a real instance's system.  Read
        // once; costs nothing when unset.
        if (it == 0)
            if (!core::env_switches().dump_kkt.empty())
                la::write_sym_csc(core::env_switches().dump_kkt.c_str(),
                                  use_ne ? ne.M : K, use_ne ? ne.sign : sign);
        if (!fac_ok) {
            if (use_ne && reg_boost >= 1e4) {
                fall_back_to_aug("non-finite pivot");
                continue;
            }
            if (reg_boost >= 1e4) {
                res.reason = "factorization produced a non-finite pivot even at 1e4x regularisation";
                break;
            }
            reg_boost *= 100.0;
            continue;
        }

        // Solve for a direction given complementarity targets tl, tu.
        auto direction = [&](const char* phase = nullptr) {
            for (core::Index j = 0; j < n; ++j) {
                const f64 h = (has_l[sz(j)] ? -tl[sz(j)] / sl[sz(j)] + lam[sz(j)] : 0.0) +
                              (has_u[sz(j)] ? tu[sz(j)] / su[sz(j)] - nu[sz(j)] : 0.0);
                rhs[sz(j)] = rdx[sz(j)] + h;
            }
            std::vector<f64> hw(sz(m), 0.0);
            for (core::Index i = 0; i < m; ++i) {
                const core::Index c = n + i;
                if (fixed_w[sz(i)]) { rhs[sz(c)] = -(rp[sz(i)] + soc[sz(i)]); continue; }
                hw[sz(i)] = (has_l[sz(c)] ? -tl[sz(c)] / sl[sz(c)] + lam[sz(c)] : 0.0) +
                            (has_u[sz(c)] ? tu[sz(c)] / su[sz(c)] - nu[sz(c)] : 0.0);
                rhs[sz(c)] = -(rp[sz(i)] + soc[sz(i)]) - (rdw[sz(i)] + hw[sz(i)]) / tdiag[sz(c)];
            }
            solve_refined(rhs, phase);
            for (core::Index j = 0; j < n; ++j) dz[sz(j)] = rhs[sz(j)];
            for (core::Index i = 0; i < m; ++i) {
                dy[sz(i)] = rhs[sz(n + i)];
                const core::Index c = n + i;
                dz[sz(c)] = fixed_w[sz(i)] ? 0.0
                                           : (-rdw[sz(i)] - hw[sz(i)] - dy[sz(i)]) / tdiag[sz(c)];
            }
            for (core::Index c = 0; c < N; ++c) {
                dlam[sz(c)] = has_l[sz(c)]
                    ? (tl[sz(c)] - sl[sz(c)] * lam[sz(c)] - lam[sz(c)] * dz[sz(c)]) / sl[sz(c)] : 0.0;
                dnu[sz(c)] = has_u[sz(c)]
                    ? (tu[sz(c)] - su[sz(c)] * nu[sz(c)] + nu[sz(c)] * dz[sz(c)]) / su[sz(c)] : 0.0;
            }
        };
        auto max_steps = [&](f64& ap, f64& ad) {
            ap = 1.0; ad = 1.0;
            for (core::Index c = 0; c < N; ++c) {
                if (has_l[sz(c)] && dz[sz(c)] < 0.0) ap = std::min(ap, -sl[sz(c)] / dz[sz(c)]);
                if (has_u[sz(c)] && dz[sz(c)] > 0.0) ap = std::min(ap, su[sz(c)] / dz[sz(c)]);
                if (has_l[sz(c)] && dlam[sz(c)] < 0.0) ad = std::min(ad, -lam[sz(c)] / dlam[sz(c)]);
                if (has_u[sz(c)] && dnu[sz(c)] < 0.0) ad = std::min(ad, -nu[sz(c)] / dnu[sz(c)]);
            }
            // One step length for both sides when Q != 0.  The stationarity
            // residual contains Qx, so after a step it is
            //     rd+ = (1 - ad) rd + (ap - ad) Q dx,
            // and with separate lengths the second term keeps re-injecting
            // dual infeasibility: on QPLIB_10034 (ap 0.995, ad 0.05) |rd| sat
            // at 0.2 for 250 iterations while mu fell to 1e-7.  Separate
            // lengths are an LP device; with Q = 0 the term vanishes and
            // they are kept (Wright, Primal-Dual Interior-Point Methods,
            // 1997, ch. 11 makes the same distinction).
            if (has_q) ap = ad = std::min(ap, ad);
        };

        // Predictor (affine scaling).
        std::fill(soc.begin(), soc.end(), 0.0);   // the predictor is purely linear
        std::fill(tl.begin(), tl.end(), 0.0);
        std::fill(tu.begin(), tu.end(), 0.0);
        direction("predictor");
        if (use_ne && !(worst_solve_rel <= kNeSolveFail)) {
            // More proximal regularisation first (as for a non-finite pivot);
            // the augmented factor once that budget is spent.
            if (reg_boost >= 1e4) {
                char why[64];
                std::snprintf(why, sizeof why, "Newton solve residual %.1e", worst_solve_rel);
                fall_back_to_aug(why);
            } else {
                if (opts.verbose)
                    std::fprintf(stderr, "ipm normal-equation solve residual %.1e: regularisation x100\n",
                                 worst_solve_rel);
                reg_boost *= 100.0;
                worst_solve_rel = 0.0;
            }
            continue;
        }
        f64 ap = 1.0, ad = 1.0;
        max_steps(ap, ad);
        f64 mu_aff = 0.0;
        for (core::Index c = 0; c < N; ++c) {
            if (has_l[sz(c)]) mu_aff += (sl[sz(c)] + ap * dz[sz(c)]) * (lam[sz(c)] + ad * dlam[sz(c)]);
            if (has_u[sz(c)]) mu_aff += (su[sz(c)] - ap * dz[sz(c)]) * (nu[sz(c)] + ad * dnu[sz(c)]);
        }
        mu_aff = pairs > 0 ? mu_aff / pairs : 0.0;
        const f64 sigma = mu > 0.0 ? std::pow(std::clamp(mu_aff / mu, 0.0, 1.0), 3.0) : 0.0;

        // Corrector: centring plus Mehrotra's second-order term.  For a
        // quadratic row the constraint itself is second order too:
        //   g(x + dx) = g(x) + J dx + 1/2 dx'Q dx   exactly,
        // so the corrector's linearised row carries the predictor's
        // 1/2 dx_aff'Q dx_aff, the constraint analogue of Mehrotra's
        // dz*dlam term (the second-order correction of Mehrotra 1992 applied
        // to the constraint instead of the complementarity).
        std::fill(soc.begin(), soc.end(), 0.0);
        for (std::size_t t = 0; t < nq; ++t) {
            const auto& q = k.quad[t];
            f64 v2 = 0.0;
            for (std::size_t e = 0; e < q.v.size(); ++e)
                v2 += (q.r[e] == q.c[e] ? 0.5 : 1.0) * q.v[e] * dz[sz(q.r[e])] * dz[sz(q.c[e])];
            soc[sz(q.row)] = v2;
        }
        for (core::Index c = 0; c < N; ++c) {
            tl[sz(c)] = has_l[sz(c)] ? sigma * mu - dz[sz(c)] * dlam[sz(c)] : 0.0;
            tu[sz(c)] = has_u[sz(c)] ? sigma * mu + dz[sz(c)] * dnu[sz(c)] : 0.0;
        }
        direction("corrector");
        max_steps(ap, ad);

        // Gondzio centrality correctors: re-target the products that sit
        // outside the band, re-solve with the SAME factorization, and keep
        // the result only when it buys a longer step.  This is what stops a
        // pair reaching s ~ 0 with lam ~ 1/s and freezing its variable.
        {
            constexpr f64 bmin = 0.1, bmax = 10.0;
            std::vector<f64> bz(dz), blam(dlam), bnu(dnu), bdy(dy);
            f64 bap = ap, bad = ad;
            const int gondzio_limit = std::max(0, opts.gondzio_correctors);
            for (int corr = 0; corr < gondzio_limit && mu > 0.0; ++corr) {
                const f64 ta = std::min(1.0, bap + 0.2), td = std::min(1.0, bad + 0.2);
                bool any = false;
                for (core::Index c = 0; c < N; ++c) {
                    if (has_l[sz(c)]) {
                        const f64 v = (sl[sz(c)] + ta * bz[sz(c)]) *
                                      (lam[sz(c)] + td * blam[sz(c)]);
                        const f64 want = std::clamp(v, bmin * mu, bmax * mu);
                        tl[sz(c)] = sigma * mu - bz[sz(c)] * blam[sz(c)] + (want - v);
                        any = any || want != v;
                    }
                    if (has_u[sz(c)]) {
                        const f64 v = (su[sz(c)] - ta * bz[sz(c)]) *
                                      (nu[sz(c)] + td * bnu[sz(c)]);
                        const f64 want = std::clamp(v, bmin * mu, bmax * mu);
                        tu[sz(c)] = sigma * mu + bz[sz(c)] * bnu[sz(c)] + (want - v);
                        any = any || want != v;
                    }
                }
                if (!any) break;
                direction("gondzio");
                ++gz_tried;
                f64 nap = 1.0, nad = 1.0;
                max_steps(nap, nad);
                if (nap < bap + 0.05 && nad < bad + 0.05) break;
                ++gz_kept;
                bz = dz; blam = dlam; bnu = dnu; bdy = dy;
                bap = nap; bad = nad;
            }
            dz = bz; dlam = blam; dnu = bnu; dy = bdy;
            ap = bap; ad = bad;
        }

        // A direction with a non-finite entry is never stepped along: raise
        // the regularisation and redo the iteration.
        {
            bool finite = true;
            for (core::Index c = 0; c < N && finite; ++c)
                finite = std::isfinite(dz[sz(c)]) && std::isfinite(dlam[sz(c)]) &&
                         std::isfinite(dnu[sz(c)]);
            for (core::Index i = 0; i < m && finite; ++i) finite = std::isfinite(dy[sz(i)]);
            if (!finite) {
                if (reg_boost >= 1e4) {
                    res.reason = "direction not finite even at 1e4x regularisation";
                    break;
                }
                reg_boost *= 100.0;
                continue;
            }
        }
        // Fraction-to-boundary grows toward 0.9995 as mu falls (Mehrotra /
        // Andersen–Andersen). Frozen 0.995 left late iterations short.
        const f64 eta_ftb = (nq == 0)
            ? std::min(0.9995, std::max(0.99, 1.0 - mu))
            : 0.995;
        ap = std::min(1.0, eta_ftb * ap);
        ad = std::min(1.0, eta_ftb * ad);
        // Quadratic rows: step to the boundary of the NONLINEAR constraint.
        // Along the step a row's value is exactly quadratic,
        //   g(x + a dx) = g(x) + a J dx + a^2/2 dx'Q dx,
        // while the linearised system only saw the first two terms.  Where
        // an inactive row is all that bounds a direction, the Newton step in
        // that direction is long (its curvature eta Q -> 0 with the
        // multiplier) and the true row value overshoots by a^2/2 dx'Q dx:
        // measured on QPLIB_2784, |rp| 9e-4 -> 2e2 in one step, after which
        // the iteration crawled at step 0.01 into the mu floor.  So a row
        // that is feasible now, g(x) < ru, keeps at least 0.5% of its slack
        // ru - g(x): the fraction-to-boundary rule applied to the true
        // constraint -- what a conic IPM's step-to-boundary of the cone does
        // for a second-order-cone reformulation of the same row.  The
        // largest such step solves 1/2 q2 a^2 + q1 a = tau s0, taken in the
        // cancellation-free form 2c / (b + sqrt(b^2 + 4ac)).
        if (nq > 0) {
            const auto& rpa = k.A.pattern.row_ptr();
            const auto& cia = k.A.pattern.col_idx();
            f64 amax = ap;
            for (std::size_t t = 0; t < nq; ++t) {
                const core::Index i = k.quad[t].row;
                const f64 s0 = k.ru[sz(i)] - ax[sz(i)];
                if (!(s0 > 0.0)) continue;   // infeasible now: the residual path decides
                f64 q1 = 0.0, q2 = 0.0;
                for (auto e = rpa[sz(i)]; e < rpa[sz(i) + 1]; ++e) q1 += k.A.vals[sz(e)] * dz[sz(cia[sz(e)])];
                for (std::size_t s2 = 0; s2 < qsup[t].size(); ++s2) q1 += qix[t][s2] * dz[sz(qsup[t][s2])];
                const auto& q = k.quad[t];
                for (std::size_t e = 0; e < q.v.size(); ++e)
                    q2 += (q.r[e] == q.c[e] ? 1.0 : 2.0) * q.v[e] * dz[sz(q.r[e])] * dz[sz(q.c[e])];
                q2 = std::max(q2, 0.0);   // PSD: negative only by rounding
                const f64 cc = 0.995 * s0;
                const f64 den = q1 + std::sqrt(q1 * q1 + 2.0 * q2 * cc);
                if (den > 0.0) amax = std::min(amax, 2.0 * cc / den);
            }
            ap = ad = std::min(ap, amax);
        }
        for (core::Index c = 0; c < N; ++c) {
            if (!(c >= n && fixed_w[sz(c - n)])) z[sz(c)] += ap * dz[sz(c)];
            lam[sz(c)] += ad * dlam[sz(c)];
            nu[sz(c)] += ad * dnu[sz(c)];
        }
        for (core::Index i = 0; i < m; ++i) y[sz(i)] += ad * dy[sz(i)];
        if (opts.verbose)
        {
            f64 rpn = 0.0, rdn = 0.0;
            core::Index rpi = -1;
            for (core::Index i = 0; i < m; ++i)
                if (std::fabs(rp[sz(i)]) > rpn) { rpn = std::fabs(rp[sz(i)]); rpi = i; }
            for (f64 v : rdx) rdn = std::max(rdn, std::fabs(v));
            for (f64 v : rdw) rdn = std::max(rdn, std::fabs(v));
            if (nq > 0 && rpi >= 0) std::fprintf(stderr, "ipm   worst rp row %lld (core)\n", static_cast<long long>(rpi));
            const int nreg = use_ne ? ne.reg_pivots : static_cast<int>(fac.regularized_pivots());
            std::fprintf(stderr, "ipm it=%d mu=%.3e |rp|=%.3e |rd|=%.3e sigma=%.2f ap=%.3f ad=%.3f reg=%d solve_res=%.1e iter %.0fms (factor %.0f, solves %.0f) total %.0fms rss %.2fGB\n",
                         it + 1, mu, rpn, rdn, sigma, ap, ad, nreg,
                         worst_solve_rel, ms_since(it_t0), fac_ms - fac_at_it, solve_ms - sol_at_it,
                         ms_since(t0), core::process_memory().rss_gb);
            if (lp_rel)
                std::fprintf(stderr, "ipm   original model: primal_rel=%.2e dual_rel=%.2e compl_rel=%.2e\n",
                             orig_p, orig_d, orig_c);
        }
        worst_solve_rel = 0.0;
        if (it + 1 == max_it) res.reason = "iteration limit";
    }
    if (opts.verbose)
        std::fprintf(stderr, "ipm gondzio correctors: %d solved, %d kept\n", gz_tried, gz_kept);
    if (opts.verbose && use_ne)
        std::fprintf(stderr, "ipm normal-eq time: assemble %.0f ms (%.2e madds each), ldlt %.0f ms, schur %.0f ms, solves %.0f ms (%llu M-solves)\n",
                     ne.assemble_ms, ne.assemble_work, ne.ldlt_ms, ne.schur_ms, solve_ms,
                     static_cast<unsigned long long>(ne.n_apply));
    if (opts.verbose && use_ne)
        std::fprintf(stderr, "ipm apply time %.0f ms of which M-solve %.0f ms\n", ne.apply_ms, ne.msolve_ms);
    } catch (const DeadlineReached&) {
        res.reason = core::cancel_requested(opts.cancel) ? "cancelled (concurrent race lost)"
                                                         : "time limit";
        res.stopped_on_time = !core::cancel_requested(opts.cancel);
        res.converged = false;
        if (opts.verbose) std::fprintf(stderr, "ipm setup stopped: %s\n", res.reason.c_str());
    }
    if (z.size() != sz(N)) z.assign(sz(N), 0.0);
    if (y.size() != sz(m)) y.assign(sz(m), 0.0);
    // An LP run that ended without meeting a test (stalled, iteration or time
    // limit) returns its best point, not the last one.
    if (!res.converged && !best_z.empty()) {
        z = best_z;
        y = best_y;
    }
    res.x.assign(z.begin(), z.begin() + n);
    res.y = y;
    res.worst_solve_residual = worst_solve_run;
    return res;
}

}  // namespace

core::RawResult solve_qp_ipm(const QpProblem& p, const QpOptions& opts, QpDiagnostics& diag) {
    const auto t0 = Clock::now();
    diag = QpDiagnostics{};
    core::RawResult raw;
    raw.engine = "qp_ipm";
    raw.backend = "cpu";
    const auto& lp = p.linear;
    auto fail = [&](core::Status s, const std::string& why) {
        raw.proposed_status = s;
        raw.termination_reason = why;
        diag.termination_reason = why;
        diag.total_ms = ms_since(t0);
        return raw;
    };
    if (lp.maximize) return fail(core::Status::Unsupported, "convex QP supports minimization only");
    try { lp.validate(); } catch (const std::exception& e) {
        return fail(core::Status::Unsupported, std::string("invalid QP: ") + e.what());
    }
    std::string why;
    f64 slack = 0.0;
    if (!certify_qp_convex(p, opts, why, slack)) return fail(core::Status::Unsupported, why);
    diag.convexity_certified = true;

    // Scale (Ruiz), exactly as the first-order path does.
    const auto sc = opts.scale ? qpc::ruiz(p, opts.ruiz_iterations)
                               : qpc::Scaling{std::vector<f64>(qpc::sz(lp.n_cols()), 1.0),
                                              std::vector<f64>(qpc::sz(lp.n_rows()), 1.0)};
    const QpProblem ps = qpc::apply_scaling(p, sc);
    const auto sb = qpc::scale_bounds({lp.col_lo, lp.col_hi, lp.row_lo, lp.row_hi}, sc);
    const core::Index n0 = lp.n_cols(), m0 = lp.n_rows();

    // Presolve: substitute fixed columns, drop free rows.
    Core k;
    std::vector<f64> xfix(sz(n0), 0.0);
    std::vector<std::uint8_t> is_fixed(sz(n0), 0);
    for (core::Index j = 0; j < n0; ++j)
        if (sb.col_lo[sz(j)] == sb.col_hi[sz(j)]) { is_fixed[sz(j)] = 1; xfix[sz(j)] = sb.col_lo[sz(j)]; }
    std::vector<core::Index> new_col(sz(n0), -1);
    for (core::Index j = 0; j < n0; ++j)
        if (!is_fixed[sz(j)]) { new_col[sz(j)] = k.n++; k.col_of.push_back(j); }
    // Q x_fixed contributions and constant.
    std::vector<f64> qfx(sz(n0), 0.0);
    if (qpc::has_sparse_q(ps)) {
        const auto& qp_ = ps.q_matrix.pattern.row_ptr();
        const auto& qi = ps.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n0; ++i)
            for (auto t = qp_[sz(i)]; t < qp_[sz(i) + 1]; ++t)
                if (is_fixed[sz(qi[sz(t)])]) qfx[sz(i)] += ps.q_matrix.vals[sz(t)] * xfix[sz(qi[sz(t)])];
    } else {
        for (core::Index j = 0; j < n0; ++j) qfx[sz(j)] = ps.q_diag[sz(j)] * xfix[sz(j)];
    }
    k.offset = ps.linear.obj_offset;
    for (core::Index j = 0; j < n0; ++j)
        if (is_fixed[sz(j)])
            k.offset += ps.linear.c[sz(j)] * xfix[sz(j)] + 0.5 * xfix[sz(j)] * qfx[sz(j)];
    k.qcol.assign(sz(k.n), {});
    k.c.resize(sz(k.n));
    for (core::Index j = 0; j < n0; ++j) {
        if (is_fixed[sz(j)]) continue;
        const core::Index jj = new_col[sz(j)];
        k.c[sz(jj)] = ps.linear.c[sz(j)] + qfx[sz(j)];
        k.l.push_back(sb.col_lo[sz(j)]);
        k.u.push_back(sb.col_hi[sz(j)]);
    }
    if (qpc::has_sparse_q(ps)) {
        const auto& qp_ = ps.q_matrix.pattern.row_ptr();
        const auto& qi = ps.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n0; ++i) {
            if (is_fixed[sz(i)]) continue;
            for (auto t = qp_[sz(i)]; t < qp_[sz(i) + 1]; ++t) {
                const core::Index j = qi[sz(t)];
                if (is_fixed[sz(j)]) continue;
                const core::Index a = new_col[sz(i)], b = new_col[sz(j)];
                if (a <= b) k.qcol[sz(b)].push_back({a, ps.q_matrix.vals[sz(t)]});
            }
        }
    } else {
        for (core::Index j = 0; j < n0; ++j)
            if (!is_fixed[sz(j)] && ps.q_diag[sz(j)] != 0.0)
                k.qcol[sz(new_col[sz(j)])].push_back({new_col[sz(j)], ps.q_diag[sz(j)]});
    }
    // Rows: shift by fixed columns; drop free rows.
    std::vector<f64> row_shift(sz(m0), 0.0);
    {
        const auto& rp = ps.linear.A.pattern.row_ptr();
        const auto& ci = ps.linear.A.pattern.col_idx();
        std::vector<core::Index> tr, tc;
        std::vector<f64> tv;
        for (core::Index i = 0; i < m0; ++i) {
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
                if (is_fixed[sz(ci[sz(t)])]) row_shift[sz(i)] += ps.linear.A.vals[sz(t)] * xfix[sz(ci[sz(t)])];
            if (!std::isfinite(sb.row_lo[sz(i)]) && !std::isfinite(sb.row_hi[sz(i)])) continue;
            const core::Index ii = k.m++;
            k.row_of.push_back(i);
            k.rl.push_back(sb.row_lo[sz(i)] - row_shift[sz(i)]);
            k.ru.push_back(sb.row_hi[sz(i)] - row_shift[sz(i)]);
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
                if (!is_fixed[sz(ci[sz(t)])]) {
                    tr.push_back(ii); tc.push_back(new_col[sz(ci[sz(t)])]); tv.push_back(ps.linear.A.vals[sz(t)]);
                }
        }
        k.A = sparse::from_triplets(k.m, k.n, tr, tc, tv);
    }

    // Objective scaling: normalise so the largest objective coefficient is
    // O(1).  Primal unchanged; duals come back divided by omega.
    f64 omega = 1.0;
    {
        f64 mx = 0.0;
        for (f64 v : k.c) mx = std::max(mx, std::fabs(v));
        for (const auto& col : k.qcol) for (const auto& e : col) mx = std::max(mx, std::fabs(e.second));
        if (mx > 0.0 && std::isfinite(mx)) omega = 1.0 / mx;
        for (f64& v : k.c) v *= omega;
        for (auto& col : k.qcol) for (auto& e : col) e.second *= omega;
    }

    // Primal scaling (LP): x = beta x', beta the largest finite row bound
    // (the column bounds when every row bound is zero), so that the iterate
    // is O(1) whatever the model's units.  The objective is already O(1)
    // (omega), the duals and c are unchanged by it, and only the bounds are
    // divided.  The proximal terms and the fraction-to-boundary rule are
    // absolute; on agg (x ~ 1e5) the rho*dx they leave held the dual residual
    // at 5e-6 for 80 iterations, and Netlib-93 took 3622 Newton steps where
    // the same iteration takes 1852 on the scaled iterate.  A QP keeps
    // beta = 1: scaling x would scale Q by beta as well.
    f64 beta = 1.0;
    {
        bool has_q = false;
        for (const auto& col : k.qcol) has_q = has_q || !col.empty();
        if (!has_q) {
            f64 bmax = 0.0;
            for (core::Index i = 0; i < k.m; ++i)
                for (f64 v : {k.rl[sz(i)], k.ru[sz(i)]})
                    if (std::isfinite(v)) bmax = std::max(bmax, std::fabs(v));
            if (bmax == 0.0)
                for (core::Index j = 0; j < k.n; ++j)
                    for (f64 v : {k.l[sz(j)], k.u[sz(j)]})
                        if (std::isfinite(v)) bmax = std::max(bmax, std::fabs(v));
            if (bmax > 1.0 && std::isfinite(bmax)) beta = bmax;
            if (beta != 1.0) {
                for (f64& v : k.l) v /= beta;
                for (f64& v : k.u) v /= beta;
                for (f64& v : k.rl) v /= beta;
                for (f64& v : k.ru) v /= beta;
            }
        }
    }

    // Map a core iterate to ORIGINAL units, in PDHCG's dual sign (y_p = -y).
    const backend::LaneBounds ob{lp.col_lo, lp.col_hi, lp.row_lo, lp.row_hi};
    auto to_original = [&](const std::vector<f64>& xc, const std::vector<f64>& yc,
                           std::vector<f64>& xo, std::vector<f64>& yo) {
        xo.assign(sz(n0), 0.0);
        yo.assign(sz(m0), 0.0);
        for (core::Index j = 0; j < n0; ++j)
            xo[sz(j)] = (is_fixed[sz(j)] ? xfix[sz(j)] : beta * xc[sz(new_col[sz(j)])]) * sc.dc[sz(j)];
        for (core::Index ii = 0; ii < k.m; ++ii) {
            const core::Index i = k.row_of[sz(ii)];
            yo[sz(i)] = -yc[sz(ii)] * sc.dr[sz(i)] / omega;
        }
    };
    qpc::OriginalKkt last;
    auto accept = [&](const std::vector<f64>& xc, const std::vector<f64>& yc) {
        std::vector<f64> xo, yo;
        to_original(xc, yc, xo, yo);
        last = qpc::kkt_original(p, ob, xo, yo);
        return qpc::kkt_ok(last, opts);
    };

    // An LP (Q = 0) also gets the original model's relative residuals, read
    // off the check `accept` just ran.
    const auto orig_rel = [&](f64& primal_rel, f64& dual_rel) {
        primal_rel = last.primal_rel;
        dual_rel = last.dual_rel;
    };
    Result r;
    try {
        r = run_ipm(k, opts, t0, accept, orig_rel);
    } catch (const std::runtime_error& e) {
        // The Newton system cannot be factored within the memory budget:
        // no point to report; the LP route falls back to the simplex.
        return fail(core::Status::NumericalFailure, e.what());
    }
    to_original(r.x, r.y, raw.x, raw.y);
    last = qpc::kkt_original(p, ob, raw.x, raw.y);
    bool ok = qpc::kkt_ok(last, opts);
    // An interior point that stalled near the optimum usually already knows
    // the active set; polishing turns that into an exact solve, and the
    // original-units check decides whether to keep it.
    bool polished = false;
    // Polished in the Ruiz-scaled space, as OSQP does: unscaled, 10034 gave
    // a row regularisation of 4.8e9 and garbage solves.
    // Not once the time is spent: the loop stopped at the limit, and the
    // polish's active-set factorization is unbounded work (dfl001,
    // --time-limit 2: the interior point stopped at 1.2 s, the polish ran
    // another 4.9 s).
    const bool time_left = opts.time_limit_s <= 0.0 ||
        ms_since(t0) < opts.time_limit_s * 1000.0;
    if (!ok && opts.polish && time_left && !core::cancel_requested(opts.cancel)) {
        const auto tp0 = Clock::now();
        polished = ok = qpc::polish_scaled(p, ob, sc, ps, sb, opts, raw.x, raw.y, last);
        if (opts.verbose)
            std::fprintf(stderr, "ipm polish %s in %.0f ms\n", polished ? "accepted" : "rejected",
                         ms_since(tp0));
    }
    diag.iterations = static_cast<std::uint64_t>(r.iterations);
    diag.worst_solve_residual = r.worst_solve_residual;
    diag.primal_residual = last.primal;
    diag.stationarity = last.dual_res;
    diag.primal_residual_rel = last.primal_rel;
    diag.stationarity_rel = last.dual_rel;
    diag.primal_net = last.primal_net;
    diag.stationarity_net = last.dual_net;
    diag.gap_net = last.gap_net;
    diag.gap_rel = last.gap;
    diag.gap_finite = last.gap_finite;
    diag.objective = last.objective;
    raw.objective = last.objective;
    raw.iterations = diag.iterations;
    raw.proposed_status = ok ? core::Status::Optimal : core::Status::Interrupted;
    raw.proposed_level = ok ? core::ProofLevel::ProvedKKT : core::ProofLevel::None;
    raw.dual_bound = ok ? last.dual_bound : core::kNaN;
    raw.termination_reason = ok ? (polished ? "interior point + polish: original-units KKT satisfied"
                                            : r.reason)
                                : "interior point stopped: " + r.reason;
    diag.termination_reason = raw.termination_reason;
    diag.stopped_on_time = !ok && r.stopped_on_time;
    diag.total_ms = ms_since(t0);
    return raw;
}

// ---- convex QCQP ------------------------------------------------------------
//
// The same iteration (run_ipm) on a Core that carries quadratic rows.  Why a
// nonlinear primal-dual IPM (Lagrangian Hessian in the KKT matrix) rather
// than a second-order-cone reformulation with Nesterov-Todd scaling:
//   * it is THIS solver's proven machinery -- the quasi-definite LDL', the
//     FGMRES refinement, Gondzio correctors, Ruiz scaling, the presolve and
//     the original-units claim all carry over; the only new objects are the
//     Jacobian rows a_i + Q_i x and the Hessian term eta_i Q_i.  An SOC
//     route needs a factor F_i of every Q_i (Q_i = F_i'F_i) to even state
//     the cone, dense scaling blocks per cone and a new conic IPM;
//   * convexity keeps the system quasi-definite: canonical rows are
//     "convex <= level", their multipliers are sign-definite, so
//     Q0 + sum eta_i Q_i is PSD (quad_eval in run_ipm has the one place
//     where a transiently wrong-signed multiplier is clipped);
//   * the QPLIB convex-QC set is norm balls with 2-3 variables (LCD) and a
//     single dense PSD row (LMC portfolio risk): their Hessians are exactly
//     what a (1,1) block holds cheaply.
// Method: the primal-dual interior point for convex nonlinear programs of
// Nocedal & Wright ch. 19 / Vanderbei & Shanno (Comput. Optim. Appl. 13,
// 1999), specialised to quadratic functions, where the constraint Hessians
// are constant and the second-order term of each row is exact (the
// corrector's `soc`).
//
// Weak duality, for the claim's gap and the branch-and-bound's bound: with
// y_i Q_i PSD for every quadratic row (PDHCG sign, y > 0 at an upper
// bound), the Lagrangian f(x) + sum_i y_i g_i(x) is convex, so its tangent
// plane at any x underestimates it; minimising the tangent over the box and
// the row bounds gives
//   D = -1/2 x'Q0x - 1/2 sum_i y_i x'Q_i x - px - py
// (kkt_original's px, py), and primal - D = x'Q0x + c'x + sum_i y_i x'Q_i x/2
// + px + py.
namespace ipmc {

void build_qcqp_core(const QcqpProblem& p0, const std::vector<std::int8_t>* orientation,
                     const QpOptions& opts, QcqpBuilt& B) {
    const auto& lp0 = p0.qp.linear;
    const core::Index n0 = lp0.n_cols(), m0 = lp0.n_rows();
    // Canonical copy: concave-above rows negated (exact in floating point),
    // so every quadratic row is "convex <= level".
    QcqpProblem p = p0;
    B.row_sign.assign(sz(m0), 1);
    for (std::size_t t = 0; orientation && t < p.quad.size(); ++t) {
        if ((*orientation)[t] >= 0) continue;
        const core::Index i = p.quad[t].row;
        B.row_sign[sz(i)] = -1;
        auto& L = p.qp.linear;
        const auto& rp = L.A.pattern.row_ptr();
        for (auto e = rp[sz(i)]; e < rp[sz(i) + 1]; ++e) L.A.vals[sz(e)] = -L.A.vals[sz(e)];
        const f64 lo = L.row_lo[sz(i)], hi = L.row_hi[sz(i)];
        L.row_lo[sz(i)] = -hi;
        L.row_hi[sz(i)] = -lo;
        for (f64& v : p.quad[t].v) v = -v;
    }
    const auto& lp = p.qp.linear;
    B.sc = opts.scale ? qpc::ruiz(p.qp, opts.ruiz_iterations, &p.quad)
                      : qpc::Scaling{std::vector<f64>(sz(n0), 1.0), std::vector<f64>(sz(m0), 1.0)};
    const auto& sc = B.sc;
    const QpProblem ps = qpc::apply_scaling(p.qp, sc);
    const auto sb = qpc::scale_bounds({lp.col_lo, lp.col_hi, lp.row_lo, lp.row_hi}, sc);
    // Scaled Hessians: row i scaled by dr_i, x = Dc x~.
    std::vector<QuadRow> qs = p.quad;
    for (auto& q : qs)
        for (std::size_t e = 0; e < q.v.size(); ++e)
            q.v[e] *= sc.dr[sz(q.row)] * sc.dc[sz(q.r[e])] * sc.dc[sz(q.c[e])];

    Core& k = B.k;
    B.xfix.assign(sz(n0), 0.0);
    B.is_fixed.assign(sz(n0), 0);
    auto& xfix = B.xfix;
    auto& is_fixed = B.is_fixed;
    for (core::Index j = 0; j < n0; ++j)
        if (sb.col_lo[sz(j)] == sb.col_hi[sz(j)]) { is_fixed[sz(j)] = 1; xfix[sz(j)] = sb.col_lo[sz(j)]; }
    B.new_col.assign(sz(n0), -1);
    auto& new_col = B.new_col;
    for (core::Index j = 0; j < n0; ++j)
        if (!is_fixed[sz(j)]) { new_col[sz(j)] = k.n++; k.col_of.push_back(j); }
    // Objective: exactly solve_qp_ipm's substitution.
    std::vector<f64> qfx(sz(n0), 0.0);
    const bool sparse_q = qpc::has_sparse_q(ps);
    if (sparse_q) {
        const auto& qp_ = ps.q_matrix.pattern.row_ptr();
        const auto& qi = ps.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n0; ++i)
            for (auto t = qp_[sz(i)]; t < qp_[sz(i) + 1]; ++t)
                if (is_fixed[sz(qi[sz(t)])]) qfx[sz(i)] += ps.q_matrix.vals[sz(t)] * xfix[sz(qi[sz(t)])];
    } else {
        for (core::Index j = 0; j < n0 && sz(j) < ps.q_diag.size(); ++j) qfx[sz(j)] = ps.q_diag[sz(j)] * xfix[sz(j)];
    }
    k.offset = ps.linear.obj_offset;
    for (core::Index j = 0; j < n0; ++j)
        if (is_fixed[sz(j)])
            k.offset += ps.linear.c[sz(j)] * xfix[sz(j)] + 0.5 * xfix[sz(j)] * qfx[sz(j)];
    k.qcol.assign(sz(k.n), {});
    k.c.resize(sz(k.n));
    for (core::Index j = 0; j < n0; ++j) {
        if (is_fixed[sz(j)]) continue;
        const core::Index jj = new_col[sz(j)];
        k.c[sz(jj)] = ps.linear.c[sz(j)] + qfx[sz(j)];
        k.l.push_back(sb.col_lo[sz(j)]);
        k.u.push_back(sb.col_hi[sz(j)]);
    }
    if (sparse_q) {
        const auto& qp_ = ps.q_matrix.pattern.row_ptr();
        const auto& qi = ps.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n0; ++i) {
            if (is_fixed[sz(i)]) continue;
            for (auto t = qp_[sz(i)]; t < qp_[sz(i) + 1]; ++t) {
                const core::Index j = qi[sz(t)];
                if (is_fixed[sz(j)]) continue;
                const core::Index a = new_col[sz(i)], b = new_col[sz(j)];
                if (a <= b) k.qcol[sz(b)].push_back({a, ps.q_matrix.vals[sz(t)]});
            }
        }
    } else {
        for (core::Index j = 0; j < n0 && sz(j) < ps.q_diag.size(); ++j)
            if (!is_fixed[sz(j)] && ps.q_diag[sz(j)] != 0.0)
                k.qcol[sz(new_col[sz(j)])].push_back({new_col[sz(j)], ps.q_diag[sz(j)]});
    }
    // Rows.  A quadratic row with some columns fixed splits three ways:
    // both fixed -> a constant (shifts the bounds), one fixed -> a linear
    // term on the other (joins the row's A entries), none -> stays in Q.
    std::vector<std::size_t> quad_of(sz(m0), qs.size());
    for (std::size_t t = 0; t < qs.size(); ++t) quad_of[sz(qs[t].row)] = t;
    const auto& rp = ps.linear.A.pattern.row_ptr();
    const auto& ci = ps.linear.A.pattern.col_idx();
    std::vector<core::Index> tr, tc;
    std::vector<f64> tv;
    for (core::Index i = 0; i < m0; ++i) {
        if (!std::isfinite(sb.row_lo[sz(i)]) && !std::isfinite(sb.row_hi[sz(i)])) continue;
        f64 shift = 0.0;
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
            if (is_fixed[sz(ci[sz(t)])]) shift += ps.linear.A.vals[sz(t)] * xfix[sz(ci[sz(t)])];
        const core::Index ii = k.m++;
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
            if (!is_fixed[sz(ci[sz(t)])]) {
                tr.push_back(ii); tc.push_back(new_col[sz(ci[sz(t)])]); tv.push_back(ps.linear.A.vals[sz(t)]);
            }
        if (const std::size_t t = quad_of[sz(i)]; t < qs.size()) {
            const auto& q = qs[t];
            Core::Quad cq;
            cq.row = ii;
            for (std::size_t e = 0; e < q.v.size(); ++e) {
                const core::Index r = q.r[e], c = q.c[e];
                const bool fr = is_fixed[sz(r)] != 0, fc = is_fixed[sz(c)] != 0;
                if (fr && fc) {
                    shift += (r == c ? 0.5 : 1.0) * q.v[e] * xfix[sz(r)] * xfix[sz(c)];
                } else if (fr || fc) {
                    // r != c here (a diagonal entry has both or neither fixed).
                    const core::Index freej = fr ? c : r, fixj = fr ? r : c;
                    tr.push_back(ii); tc.push_back(new_col[sz(freej)]); tv.push_back(q.v[e] * xfix[sz(fixj)]);
                } else {
                    cq.r.push_back(new_col[sz(r)]);
                    cq.c.push_back(new_col[sz(c)]);
                    cq.v.push_back(q.v[e]);
                }
            }
            if (!cq.v.empty()) k.quad.push_back(std::move(cq));
        }
        k.row_of.push_back(i);
        k.rl.push_back(sb.row_lo[sz(i)] - shift);
        k.ru.push_back(sb.row_hi[sz(i)] - shift);
    }
    k.A = sparse::from_triplets(k.m, k.n, tr, tc, tv);

    // Objective scaling, as solve_qp_ipm: the objective only -- constraint
    // Hessians are not objective terms; their multipliers absorb omega.
    f64 mx = 0.0;
    for (f64 v : k.c) mx = std::max(mx, std::fabs(v));
    for (const auto& col : k.qcol) for (const auto& e : col) mx = std::max(mx, std::fabs(e.second));
    if (mx > 0.0 && std::isfinite(mx)) B.omega = 1.0 / mx;
    for (f64& v : k.c) v *= B.omega;
    for (auto& col : k.qcol) for (auto& e : col) e.second *= B.omega;
}

}  // namespace ipmc

core::RawResult solve_qcqp_ipm(const QcqpProblem& p0, const QpOptions& opts, QpDiagnostics& diag,
                               const QcqpConvexity* cert_in) {
    const auto t0 = Clock::now();
    diag = QpDiagnostics{};
    core::RawResult raw;
    raw.engine = "qcqp_ipm";
    raw.backend = "cpu";
    const auto& lp = p0.qp.linear;
    auto fail = [&](core::Status s, const std::string& why) {
        raw.proposed_status = s;
        raw.termination_reason = why;
        diag.termination_reason = why;
        diag.total_ms = ms_since(t0);
        return raw;
    };
    if (lp.maximize) return fail(core::Status::Unsupported, "convex QCQP supports minimization only");
    try { p0.validate(); } catch (const std::exception& e) {
        return fail(core::Status::Unsupported, std::string("invalid QCQP: ") + e.what());
    }
    bool has_int = false;
    for (std::size_t j = 0; j < lp.is_integer.size(); ++j) has_int = has_int || lp.is_integer[j];
    if (!p0.has_quadratic_constraints() && !has_int) {
        // No quadratic row: exactly the QP interior point and its claim.
        return solve_qp_ipm(p0.qp, opts, diag);
    }
    // Convexity first: a nonconvex row is the more fundamental refusal (no
    // engine here would take the model), and its reason names the row.
    QcqpConvexity own;
    const QcqpConvexity* cert = cert_in;
    if (!cert) {
        own = certify_qcqp_convex(p0, opts);
        cert = &own;
    }
    if (!cert->ok || cert->orientation.size() != p0.quad.size())
        return fail(core::Status::Unsupported, "convex QCQP interior point (model has quadratic constraints): " +
                    (cert->ok ? std::string("certificate does not match the model") : cert->reason));
    if (has_int)
        return fail(core::Status::Unsupported,
                    "the QCQP interior point is continuous; the model has integer columns "
                    "(the mixed-integer QCQP branch-and-bound takes them)");
    diag.convexity_certified = true;

    ipmc::QcqpBuilt B;
    ipmc::build_qcqp_core(p0, &cert->orientation, opts, B);
    const core::Index n0 = lp.n_cols(), m0 = lp.n_rows();
    const auto& sc = B.sc;
    const Core& k = B.k;
    const backend::LaneBounds ob{lp.col_lo, lp.col_hi, lp.row_lo, lp.row_hi};
    // Core iterate -> ORIGINAL units, PDHCG's dual sign, original row sign.
    auto to_original = [&](const std::vector<f64>& xc, const std::vector<f64>& yc,
                           std::vector<f64>& xo, std::vector<f64>& yo) {
        xo.assign(sz(n0), 0.0);
        yo.assign(sz(m0), 0.0);
        for (core::Index j = 0; j < n0; ++j)
            xo[sz(j)] = (B.is_fixed[sz(j)] ? B.xfix[sz(j)] : xc[sz(B.new_col[sz(j)])]) * sc.dc[sz(j)];
        for (core::Index ii = 0; ii < k.m; ++ii) {
            const core::Index i = k.row_of[sz(ii)];
            yo[sz(i)] = -yc[sz(ii)] * sc.dr[sz(i)] / B.omega * B.row_sign[sz(i)];
        }
    };
    qpc::OriginalKkt last;
    auto accept = [&](const std::vector<f64>& xc, const std::vector<f64>& yc) {
        std::vector<f64> xo, yo;
        to_original(xc, yc, xo, yo);
        last = qpc::kkt_original_qcqp(p0, ob, xo, yo);
        return qpc::kkt_ok(last, opts);
    };
    const auto r = run_ipm(k, opts, t0, accept);
    to_original(r.x, r.y, raw.x, raw.y);
    last = qpc::kkt_original_qcqp(p0, ob, raw.x, raw.y);
    // No polish: its active-set solve is for linear rows; an active
    // quadratic row would need a nonlinear equality solve.
    const bool ok = qpc::kkt_ok(last, opts);
    diag.iterations = static_cast<std::uint64_t>(r.iterations);
    diag.worst_solve_residual = r.worst_solve_residual;
    diag.primal_residual = last.primal;
    diag.stationarity = last.dual_res;
    diag.primal_residual_rel = last.primal_rel;
    diag.stationarity_rel = last.dual_rel;
    diag.primal_net = last.primal_net;
    diag.stationarity_net = last.dual_net;
    diag.gap_net = last.gap_net;
    diag.gap_rel = last.gap;
    diag.gap_finite = last.gap_finite;
    diag.objective = last.objective;
    raw.objective = last.objective;
    raw.iterations = diag.iterations;
    raw.proposed_status = ok ? core::Status::Optimal : core::Status::Interrupted;
    raw.proposed_level = ok ? core::ProofLevel::ProvedKKT : core::ProofLevel::None;
    raw.dual_bound = ok ? last.dual_bound : core::kNaN;
    raw.termination_reason = ok ? "QCQP interior point: original-units KKT satisfied (quadratic rows included)"
                                : "QCQP interior point stopped: " + r.reason;
    diag.termination_reason = raw.termination_reason;
    diag.total_ms = ms_since(t0);
    return raw;
}

}  // namespace forge::engines
