// Parallel binquad: the Vulkan BinQuadDevice against the CPU oracle.
//
// On integer-valued instances every quantity the search computes is exact
// in double (halves included), so the two devices must follow IDENTICAL
// trajectories: this test compares every per-search state field and every
// search's best point bit for bit, across epochs, restarts and elite
// perturbation.  It also checks the host driver end to end and that the
// point it returns is re-scored from the raw instance.  Skips (passes) when
// no fp64 Vulkan device exists.
#include "sor/backend/binquad_device.hpp"
#include "sor/io/qplib.hpp"
#include "sor/search/binquad.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
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
    int integer(int lo, int hi) { return lo + static_cast<int>(next() * (hi - lo + 1)); }
};

// Integer QBL: indefinite H, a cardinality-style equality row and two
// knapsack rows, so penalties, aspiration and infeasible phases all occur.
io::QplibInstance integer_qbl(Index n, std::uint64_t seed) {
    Lcg g{seed};
    io::QplibInstance q;
    q.name = "INT_QBL";
    q.classification = {'Q', 'B', 'L'};
    q.n = n;
    q.m = 3;
    for (Index i = 1; i <= n; ++i)
        for (Index j = i; j <= n; ++j)
            if (i == j || g.next() < 0.3) {
                q.h_row.push_back(i); q.h_col.push_back(j);
                q.h_val.push_back(g.integer(-20, 20));
            }
    for (Index j = 0; j < n; ++j) q.g.push_back(g.integer(-10, 10));
    for (Index j = 1; j <= n; ++j) {
        q.a_row.push_back(1); q.a_col.push_back(j); q.a_val.push_back(1.0);
        if (g.next() < 0.5) { q.a_row.push_back(2); q.a_col.push_back(j); q.a_val.push_back(g.integer(1, 9)); }
        if (g.next() < 0.5) { q.a_row.push_back(3); q.a_col.push_back(j); q.a_val.push_back(g.integer(1, 9)); }
    }
    q.c_lo = {static_cast<f64>(n / 3), -1e20, 10.0};
    q.c_hi = {static_cast<f64>(n / 3), 60.0, 1e20};
    q.inf_bound = 1e20;
    q.x_lo.assign(static_cast<std::size_t>(n), 0.0);
    q.x_hi.assign(static_cast<std::size_t>(n), 1.0);
    q.var_type.assign(static_cast<std::size_t>(n), io::QplibVarType::Binary);
    return q;
}

bool same(const backend::BqSearch& a, const backend::BqSearch& b) {
    return a.obj == b.obj && a.viol == b.viol && a.penalty == b.penalty &&
           a.best_obj == b.best_obj && a.best_viol == b.best_viol && a.iter == b.iter &&
           a.since_improve == b.since_improve && a.have_feasible == b.have_feasible &&
           a.stalled == b.stalled;
}

void lockstep(const std::string& label, const io::QplibInstance& q,
              backend::BinQuadDevice& vk) {
    auto cpu = backend::make_cpu_binquad_device();
    f64 scale = 1.0;
    const auto data = search::binquad_device_data(q, scale);
    backend::BqParams p;
    p.searches = 24;
    p.stagnation_limit = 60;     // small, so restarts actually happen
    p.penalty0 = scale;
    p.penalty_floor = 1e-6 * scale;
    p.seed = 7;
    cpu->upload(data, p);
    vk.upload(data, p);

    const std::uint32_t P = p.searches;
    cpu->restart(std::vector<std::uint8_t>(P, 1), std::vector<std::int32_t>(P, -1), 0.5, 0);
    vk.restart(std::vector<std::uint8_t>(P, 1), std::vector<std::int32_t>(P, -1), 0.5, 0);

    std::vector<backend::BqSearch> a, b;
    std::vector<std::uint8_t> xa, xb;
    int mismatches = 0, restarts = 0;
    for (std::uint32_t epoch = 1; epoch <= 8; ++epoch) {
        cpu->run(50);
        vk.run(50);
        cpu->read_searches(a);
        vk.read_searches(b);
        for (std::uint32_t s = 0; s < P; ++s) {
            if (!same(a[s], b[s])) ++mismatches;
            cpu->read_best(s, xa);
            vk.read_best(s, xb);
            if (xa != xb) ++mismatches;
        }
        // Elite = the first two searches' bests; restart every stalled one.
        cpu->read_best(0, xa);
        cpu->read_best(1, xb);
        cpu->set_elite({xa, xb});
        vk.set_elite({xa, xb});
        std::vector<std::uint8_t> rs(P, 0);
        std::vector<std::int32_t> base(P, -1);
        for (std::uint32_t s = 0; s < P; ++s)
            if (a[s].stalled) { rs[s] = 1; base[s] = static_cast<std::int32_t>(s % 2); ++restarts; }
        cpu->restart(rs, base, 0.15, epoch);
        vk.restart(rs, base, 0.15, epoch);
    }
    ::sor::test::report(mismatches == 0, "bit-identical CPU/GPU search trajectories",
                        __FILE__, __LINE__, label + ": " + std::to_string(mismatches) +
                                                " mismatching search/epochs");
    ::sor::test::report(restarts > 0, "restarts exercised", __FILE__, __LINE__, label);
    std::printf("  %-10s lockstep: %d mismatches, %d restarts over 8 epochs x %u searches\n",
                label.c_str(), mismatches, restarts, P);
}

}  // namespace

int main() {
    CHECK(backend::make_binquad_device("cpu") != nullptr);
    CHECK(backend::make_binquad_device("no_such") == nullptr);

    auto vk = backend::make_vulkan_binquad_device();
    if (!vk) {
        std::printf("SKIP vulkan binquad parity: no fp64 Vulkan device\n");
        return ::sor::test::finish("test_binquad_device");
    }
    CHECK(vk->is_accelerated());

    lockstep("INT_QBL", integer_qbl(70, 3), *vk);
    // Skipped when the instance is absent: it is fetched, not tracked.
    if (sor::test::data_available(std::string(SOR_TEST_DATA_DIR) +
                                  "/qplib/QPLIB_3565.qplib")) {
        io::QplibReadReport rep;
        const auto q = io::read_qplib_file(
            std::string(SOR_TEST_DATA_DIR) + "/qplib/QPLIB_3565.qplib", rep);
        lockstep("QPLIB_3565", q, *vk);   // QBB, maximise, integer data
    }

    // Host driver, both devices, fixed epochs: identical results, and the
    // returned objective is the raw-data re-score of the returned point.
    {
        const auto q = integer_qbl(70, 5);
        search::BinQuadParallelOptions o;
        o.searches = 32;
        o.epoch_iterations = 100;
        o.max_epochs = 20;
        o.time_limit_s = 0.0;
        o.stagnation_limit = 200;
        auto cpu = backend::make_cpu_binquad_device();
        search::BinQuadDiagnostics dc, dv;
        const auto rc = search::solve_binquad_parallel(q, o, *cpu, dc);
        const auto rv = search::solve_binquad_parallel(q, o, *vk, dv);
        CHECK(rc.feasible && rv.feasible);
        CHECK(rc.x == rv.x);
        CHECK(rc.objective == rv.objective);
        f64 o2 = q.f_const;
        for (std::size_t t = 0; t < q.h_val.size(); ++t)
            o2 += 0.5 * q.h_val[t] * rv.x[static_cast<std::size_t>(q.h_row[t] - 1)] *
                  rv.x[static_cast<std::size_t>(q.h_col[t] - 1)];
        for (Index j = 0; j < q.n; ++j) o2 += q.g[static_cast<std::size_t>(j)] * rv.x[static_cast<std::size_t>(j)];
        CHECK(o2 == rv.objective);
        std::printf("  driver: cpu %.1f  gpu %.1f  (restarts %llu / %llu)\n", rc.objective,
                    rv.objective, static_cast<unsigned long long>(dc.restarts),
                    static_cast<unsigned long long>(dv.restarts));
    }

    return ::sor::test::finish("test_binquad_device");
}
