// Binary-QP branch-and-bound: a PROOF is only as good as its agreement with
// exhaustive enumeration.  On instances small enough to enumerate, a run that
// reports `proved` must land on the brute-force optimum (to gap_tol), and
// its bound must never sit on the wrong side of it.  Run over CPU lanes and,
// when present, the Vulkan batched device.
#include "sor/backend/batched_pdhcg_device.hpp"
#include "sor/io/qplib.hpp"
#include "sor/search/bqp_bab.hpp"
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


void check(const std::string& label, backend::BatchedPdhcgDevice& dev, std::uint64_t seeds,
           bool rounding, search::BqpNodeSolver solver = search::BqpNodeSolver::Auto,
           std::size_t strong = 0) {
    for (std::uint64_t seed = 1; seed <= seeds; ++seed) {
        const bool maximize = seed % 2 == 0;
        const Index m = seed % 3 == 0 ? 0 : 2;
        const auto q = make_binary(12, m, maximize, seed);
        bool feas = false;
        const f64 opt = brute_force(q, feas);
        search::BqpBabOptions o;
        o.batch = 4;
        o.time_limit_s = 120.0;
        o.qcr.qp.max_iterations = 3000;
        o.rounding = rounding;
        o.node_solver = solver;
        o.sb_candidates = strong;
        // Probe every candidate at least once before trusting a pseudocost,
        // so the probe-bound inheritance path is actually exercised.
        o.reliability = 1;
        const auto r = search::solve_bqp_bab(q, nullptr, o, dev);
        const f64 tol = 1e-6 * std::max(1.0, std::fabs(opt));
        ::sor::test::report(r.proved, "tree closes", __FILE__, __LINE__, label + " " + q.name);
        ::sor::test::report(r.have_incumbent && std::fabs(r.incumbent - opt) <= tol,
                            "proved optimum equals brute force", __FILE__, __LINE__,
                            label + " " + q.name + " got " + std::to_string(r.incumbent) +
                                " opt " + std::to_string(opt));
        ::sor::test::report(r.bound_valid && (maximize ? r.bound >= opt - tol : r.bound <= opt + tol),
                            "bound on the right side", __FILE__, __LINE__, label + " " + q.name);
        std::printf("  %-14s %-8s %s opt %12.4f  nodes %5llu batches %4llu pruned %5llu leaves %4llu"
                    " probes %4llu (%s)\n",
                    label.c_str(), q.name.c_str(), maximize ? "max" : "min", opt,
                    static_cast<unsigned long long>(r.nodes),
                    static_cast<unsigned long long>(r.batches),
                    static_cast<unsigned long long>(r.pruned),
                    static_cast<unsigned long long>(r.leaves),
                    static_cast<unsigned long long>(r.probes), r.node_solver.c_str());
    }
}

}  // namespace

int main() {
    {
        std::vector<std::unique_ptr<backend::PdhcgDevice>> devs;
        for (int l = 0; l < 4; ++l) devs.push_back(backend::make_cpu_pdhcg_device());
        auto cpu = backend::make_lanes_device(std::move(devs));
        using NS = search::BqpNodeSolver;
        // Both node solvers, each with and without rounding.  No rounding:
        // the optimum must be reached as a leaf, so any bound that overstates
        // itself -- node or strong-branching probe -- prunes it away and
        // fails the brute-force check.  The PDHCG lanes are what the Vulkan
        // batch is checked against, so they stay under test even though the
        // CPU default is now the IPM.
        check("ipm", *cpu, 6, true, NS::Ipm);
        check("ipm-leaf", *cpu, 6, false, NS::Ipm);
        check("ipm-strong-leaf", *cpu, 6, false, NS::Ipm, 3);
        check("pdhcg", *cpu, 6, true, NS::Pdhcg);
        check("pdhcg-leaf", *cpu, 6, false, NS::Pdhcg);
        check("pdhcg-strong", *cpu, 3, true, NS::Pdhcg, 3);
    }
    // Two seeds on the GPU keep ctest quick; the CPU lanes cover all six.
    if (auto vk = backend::make_vulkan_batched_pdhcg_device()) check("vulkan", *vk, 2, true);
    else std::printf("SKIP vulkan: no fp64 Vulkan device\n");
    return ::sor::test::finish("test_bqp_bab");
}
