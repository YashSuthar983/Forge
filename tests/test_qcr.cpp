// QCR bound validity.  The one property that matters is that the bound is
// NEVER on the wrong side of the true optimum, so it is checked against
// exhaustive enumeration on instances small enough to enumerate, and against
// QPLIB's published optimum on a real instance.
//
// Also pinned: an approximate or even arbitrary (x, y) may weaken the bound
// but must not invalidate it -- that is what lets the relaxation run on a
// first-order method, on any device, with a time limit.
#include "sor/backend/pdhcg_device.hpp"
#include "sor/io/qplib.hpp"
#include "sor/search/qcr.hpp"
#include "sor/search/qplib_qp.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace sor;
using core::f64;
using core::Index;

namespace {

struct Lcg {
    std::uint64_t s;
    f64 next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<f64>(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

// A random binary QBL/QBN instance built directly in QPLIB's own form, so
// the test goes through the same reader-facing conversion as real files.
io::QplibInstance make_binary(Index n, Index m, bool maximize, std::uint64_t seed) {
    Lcg g{seed};
    io::QplibInstance q;
    q.name = "RAND_" + std::to_string(seed);
    q.classification = {'Q', 'B', m > 0 ? 'L' : 'N'};
    q.maximize = maximize;
    q.n = n;
    q.m = m;
    for (Index i = 1; i <= n; ++i)
        for (Index j = i; j <= n; ++j)
            if (i == j || g.next() < 0.6) {   // indefinite, fairly dense
                q.h_row.push_back(i);
                q.h_col.push_back(j);
                q.h_val.push_back(10.0 * (2.0 * g.next() - 1.0));
            }
    for (Index j = 0; j < n; ++j) q.g.push_back(5.0 * (2.0 * g.next() - 1.0));
    q.f_const = 1.5;
    for (Index i = 1; i <= m; ++i) {
        for (Index j = 1; j <= n; ++j)
            if (g.next() < 0.5) {
                q.a_row.push_back(i);
                q.a_col.push_back(j);
                q.a_val.push_back(1.0);
            }
        q.c_lo.push_back(1.0);                        // at least one ...
        q.c_hi.push_back(static_cast<f64>(n) - 2.0);  // ... not all
    }
    q.inf_bound = 1e20;
    q.x_lo.assign(static_cast<std::size_t>(n), 0.0);
    q.x_hi.assign(static_cast<std::size_t>(n), 1.0);
    q.var_type.assign(static_cast<std::size_t>(n), io::QplibVarType::Binary);
    return q;
}

// Exhaustive optimum straight from the raw triplets (0.5 * H as stored).
f64 brute_force(const io::QplibInstance& q, bool& any_feasible) {
    const Index n = q.n;
    f64 best = q.maximize ? -std::numeric_limits<f64>::infinity()
                          : std::numeric_limits<f64>::infinity();
    any_feasible = false;
    std::vector<f64> x(static_cast<std::size_t>(n));
    for (std::uint32_t mask = 0; mask < (1u << n); ++mask) {
        for (Index j = 0; j < n; ++j) x[static_cast<std::size_t>(j)] = (mask >> j) & 1u;
        std::vector<f64> ax(static_cast<std::size_t>(q.m), 0.0);
        for (std::size_t t = 0; t < q.a_val.size(); ++t)
            ax[static_cast<std::size_t>(q.a_row[t] - 1)] +=
                q.a_val[t] * x[static_cast<std::size_t>(q.a_col[t] - 1)];
        bool ok = true;
        for (Index i = 0; i < q.m; ++i)
            ok = ok && ax[static_cast<std::size_t>(i)] >= q.c_lo[static_cast<std::size_t>(i)] &&
                 ax[static_cast<std::size_t>(i)] <= q.c_hi[static_cast<std::size_t>(i)];
        if (!ok) continue;
        f64 o = q.f_const;
        for (std::size_t t = 0; t < q.h_val.size(); ++t)
            o += 0.5 * q.h_val[t] * x[static_cast<std::size_t>(q.h_row[t] - 1)] *
                 x[static_cast<std::size_t>(q.h_col[t] - 1)];
        for (Index j = 0; j < n; ++j) o += q.g[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
        any_feasible = true;
        best = q.maximize ? std::max(best, o) : std::min(best, o);
    }
    return best;
}

bool on_right_side(const io::QplibInstance& q, f64 bound, f64 opt) {
    const f64 slack = 1e-9 * (1.0 + std::fabs(opt));
    return q.maximize ? bound >= opt - slack : bound <= opt + slack;
}

}  // namespace

int main() {
    auto cpu = backend::make_cpu_pdhcg_device();

    // ---- exhaustive: both shifts, min and max, with and without rows ----
    int checked = 0;
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        const bool maximize = seed % 2 == 0;
        const Index m = seed % 3 == 0 ? 0 : 2;
        const auto q = make_binary(12, m, maximize, seed);
        bool feas = false;
        const f64 opt = brute_force(q, feas);
        CHECK(feas);
        for (auto shift : {search::QcrShift::Best, search::QcrShift::Sdp,
                           search::QcrShift::MinEigenvalue,
                           search::QcrShift::DiagonalDominance}) {
            search::QcrOptions o;
            o.shift = shift;
            o.qp.max_iterations = 20000;
            search::QcrDiagnostics d;
            const auto b = search::qcr_bound(q, o, *cpu, d);
            ::sor::test::report(b.valid, "QCR produced a bound", __FILE__, __LINE__,
                                q.name + " " + d.reason);
            ::sor::test::report(!b.valid || on_right_side(q, b.bound, opt),
                                "QCR bound on the right side of the optimum",
                                __FILE__, __LINE__,
                                q.name + " bound " + std::to_string(b.bound) +
                                    " opt " + std::to_string(opt));
            ++checked;
        }
    }
    CHECK(checked == 24);

    // ---- an approximate SDP dual is still a valid shift ----
    // The mixing method is stopped after 1 or 3 sweeps, and with a single
    // penalty stage, so the dual it hands over is far from optimal and its
    // S = Q + 2 diag(u) is nowhere near PSD on its own.  The shift is only
    // used after the lambda_min correction and the PSD certificate, so the
    // bound must still sit on the right side of the exhaustive optimum --
    // weaker, never wrong.
    {
        int approx_checked = 0;
        for (std::uint64_t seed = 1; seed <= 6; ++seed) {
            const bool maximize = seed % 2 == 0;
            const Index m = seed % 3 == 0 ? 0 : 2;
            const auto q = make_binary(12, m, maximize, seed);
            bool feas = false;
            const f64 opt = brute_force(q, feas);
            for (int sweeps : {1, 3}) {
                search::QcrOptions o;
                o.shift = search::QcrShift::Sdp;
                o.qp.max_iterations = 20000;
                o.sdp.max_sweeps = sweeps;
                o.sdp.penalty_stages = 1;
                search::QcrDiagnostics d;
                const auto b = search::qcr_bound(q, o, *cpu, d);
                ::sor::test::report(b.valid, "truncated-SDP QCR produced a bound", __FILE__,
                                    __LINE__, q.name + " sweeps " + std::to_string(sweeps) +
                                                   " " + d.reason);
                ::sor::test::report(!b.valid || on_right_side(q, b.bound, opt),
                                    "truncated-SDP QCR bound on the right side",
                                    __FILE__, __LINE__,
                                    q.name + " sweeps " + std::to_string(sweeps) + " bound " +
                                        std::to_string(b.bound) + " opt " + std::to_string(opt) +
                                        " slack " + std::to_string(d.psd_slack));
                ++approx_checked;
            }
        }
        CHECK(approx_checked == 12);
    }

    // ---- arbitrary (x, y): weaker, never wrong ----
    // wolfe_bound at points nowhere near optimal, including infeasible x and
    // badly signed y, must still sit below the true minimum whenever it
    // returns a bound at all.
    {
        const auto q = make_binary(10, 2, false, 99);
        bool feas = false;
        const f64 opt = brute_force(q, feas);
        engines::QpProblem base;
        bool neg = false;
        std::string why;
        search::QplibToQpOptions conv;
        conv.relax_binary = true;
        CHECK(search::qplib_to_qp(q, conv, base, neg, why));
        // Convexify crudely by a large uniform shift; any PSD shift is valid.
        auto p = base;
        {
            std::vector<Index> r, c;
            std::vector<f64> v;
            const auto& rp = base.q_matrix.pattern.row_ptr();
            const auto& ci = base.q_matrix.pattern.col_idx();
            for (Index i = 0; i < 10; ++i) {
                for (auto k = rp[static_cast<std::size_t>(i)];
                     k < rp[static_cast<std::size_t>(i) + 1]; ++k) {
                    r.push_back(i); c.push_back(ci[static_cast<std::size_t>(k)]);
                    v.push_back(base.q_matrix.vals[static_cast<std::size_t>(k)]);
                }
                r.push_back(i); c.push_back(i); v.push_back(200.0);   // 2u_i
                p.linear.c[static_cast<std::size_t>(i)] -= 100.0;     // -u_i
            }
            p.q_matrix = sparse::from_triplets(10, 10, r, c, v);
        }
        Lcg g{7};
        int bounded = 0;
        for (int trial = 0; trial < 200; ++trial) {
            std::vector<f64> x(10), y(2);
            for (auto& a : x) a = 3.0 * g.next() - 1.0;    // outside [0,1] too
            for (auto& a : y) a = 20.0 * (2.0 * g.next() - 1.0);
            f64 b = 0.0, raw = 0.0, cp = 0.0, cf = 0.0;
            if (search::wolfe_bound(p, x, y, 0.0, b, raw, cp, cf)) {
                ++bounded;
                CHECK(b <= opt + 1e-9 * (1.0 + std::fabs(opt)));
            }
        }
        CHECK(bounded > 0);
    }

    // ---- QPLIB_3834 against its published optimum ----
    // Skipped when the instance is absent: it is fetched, not tracked.
    if (sor::test::data_available(std::string(SOR_TEST_DATA_DIR) +
                                 "/qplib/QPLIB_3834.qplib")) {
        io::QplibReadReport rep;
        const auto q = io::read_qplib_file(
            std::string(SOR_TEST_DATA_DIR) + "/qplib/QPLIB_3834.qplib", rep);
        search::QcrOptions o;
        o.qp.max_iterations = 3000;   // truncated on purpose: still valid
        search::QcrDiagnostics d;
        const auto b = search::qcr_bound(q, o, *cpu, d);
        CHECK(b.valid);
        CHECK(b.bound <= 3760.715066);
        CHECK(d.shift_used == "sdp");   // Auto prefers the SDP shift at this size
        std::printf("  QPLIB_3834 bound %.6f after %llu iterations (published opt 3760.715066)\n",
                    b.bound, static_cast<unsigned long long>(d.relaxation.iterations));
    }

    return ::sor::test::finish("test_qcr");
}
