// PDHCG-II on a device: the Vulkan PdhcgDevice against the CPU one, run
// through the real engine (solve_qp_pdhcg) so the host decisions -- inner
// stop test, Barzilai-Borwein step, tolerance update, convergence check --
// are exercised on the scalars each device actually returns.
//
// Covers both Q forms (sparse off-diagonal; diagonal with ranged rows, which
// the active-set fast path cannot take), the reflected-Halpern variant, free
// columns (no finite Wolfe bound), and one size large enough that the
// reductions stride past kMaxPartials workgroups.  Skips when no fp64
// Vulkan device exists.
#include "sor/backend/pdhcg_device.hpp"
#include "sor/engines/qp.hpp"
#include "sor/sparse/csr.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace sor;
using core::f64;
using core::Index;

namespace {

constexpr f64 kInf = model::kInf;

struct Lcg {
    std::uint64_t s;
    f64 next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<f64>(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

// Convex by construction: Q symmetric, diagonally dominant, positive diagonal.
engines::QpProblem make_qp(Index m, Index n, bool diagonal, bool free_cols,
                           std::uint64_t seed) {
    Lcg g{seed};
    engines::QpProblem p;
    auto& lp = p.linear;
    std::vector<Index> r, c;
    std::vector<f64> v;
    for (Index i = 0; i < m; ++i)
        for (int k = 0; k < 3; ++k) {
            r.push_back(i);
            c.push_back(static_cast<Index>(g.next() * static_cast<f64>(n)) % n);
            v.push_back(2.0 * g.next() - 1.0);
        }
    lp.A = sparse::from_triplets(m, n, r, c, v);
    for (Index j = 0; j < n; ++j) {
        lp.c.push_back(2.0 * g.next() - 1.0);
        if (free_cols && j % 5 == 4) {
            lp.col_lo.push_back(-kInf);
            lp.col_hi.push_back(kInf);
        } else {
            lp.col_lo.push_back(-2.0);
            lp.col_hi.push_back(2.0 + g.next());
        }
    }
    for (Index i = 0; i < m; ++i) {
        const f64 b = 0.5 * (2.0 * g.next() - 1.0);
        if (i % 2 == 0) { lp.row_lo.push_back(b); lp.row_hi.push_back(b); }
        else { lp.row_lo.push_back(b - 1.0); lp.row_hi.push_back(b + 1.0); }
    }
    lp.is_integer.assign(static_cast<std::size_t>(n), false);

    if (diagonal) {
        for (Index j = 0; j < n; ++j) p.q_diag.push_back(0.5 + g.next());
    } else {
        std::vector<Index> qr, qc;
        std::vector<f64> qv, rowsum(static_cast<std::size_t>(n), 0.0);
        for (Index j = 0; j + 1 < n; j += 2) {
            const f64 o = 0.4 * (2.0 * g.next() - 1.0);
            qr.push_back(j); qc.push_back(j + 1); qv.push_back(o);
            qr.push_back(j + 1); qc.push_back(j); qv.push_back(o);
            rowsum[static_cast<std::size_t>(j)] += std::fabs(o);
            rowsum[static_cast<std::size_t>(j + 1)] += std::fabs(o);
        }
        for (Index j = 0; j < n; ++j) {
            qr.push_back(j); qc.push_back(j);
            qv.push_back(rowsum[static_cast<std::size_t>(j)] + 0.2 + g.next());
        }
        p.q_matrix = sparse::from_triplets(n, n, qr, qc, qv);
    }
    return p;
}

f64 max_rel(const std::vector<f64>& a, const std::vector<f64>& b) {
    if (a.size() != b.size()) return kInf;
    f64 scale = 1.0, d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        scale = std::max(scale, std::fabs(b[i]));
        d = std::max(d, std::fabs(a[i] - b[i]));
    }
    return d / scale;
}

void parity(const std::string& label, const engines::QpProblem& p,
            engines::QpOptions opts, backend::PdhcgDevice& vk) {
    auto cpu = backend::make_cpu_pdhcg_device();
    engines::QpDiagnostics dc, dv;
    const auto rc = engines::solve_qp_pdhcg(p, opts, *cpu, dc);
    const auto rv = engines::solve_qp_pdhcg(p, opts, vk, dv);

    const auto fail = [&](bool ok, const char* what) {
        ::sor::test::report(ok, what, __FILE__, __LINE__, label);
    };
    fail(rc.proposed_status == rv.proposed_status, "status parity");
    // Same host decisions on both devices: identical iteration counts.
    fail(dc.iterations == dv.iterations, "outer iteration parity");
    fail(dc.inner_iterations == dv.inner_iterations, "inner iteration parity");
    fail(max_rel(rv.x, rc.x) < 1e-9, "x parity");
    fail(max_rel(rv.y, rc.y) < 1e-9, "y parity");
    fail(std::fabs(dv.objective - dc.objective) <= 1e-9 * (1.0 + std::fabs(dc.objective)),
         "objective parity");
    fail(dc.gap_finite == dv.gap_finite, "Wolfe-bound finiteness parity");
    fail(dv.used_general_path, "device run took the PDHCG path");
    std::printf("  %-22s iters %llu/%llu  inner %llu/%llu  status %d  obj %.12e\n",
                label.c_str(), static_cast<unsigned long long>(dc.iterations),
                static_cast<unsigned long long>(dv.iterations),
                static_cast<unsigned long long>(dc.inner_iterations),
                static_cast<unsigned long long>(dv.inner_iterations),
                static_cast<int>(rv.proposed_status), dv.objective);
}

}  // namespace

int main() {
    CHECK(backend::make_pdhcg_device("cpu") != nullptr);
    CHECK(backend::make_pdhcg_device("no_such") == nullptr);

    auto vk = backend::make_vulkan_pdhcg_device();
    if (!vk) {
        std::printf("SKIP vulkan PDHCG parity: no fp64 Vulkan device\n");
        return ::sor::test::finish("test_pdhcg_device");
    }
    CHECK(vk->is_accelerated());

    engines::QpOptions opts;
    opts.max_iterations = 4000;
    opts.feas_tol = opts.stationarity_tol = opts.gap_tol = 1e-8;

    parity("sparse Q", make_qp(30, 60, false, false, 11), opts, *vk);
    parity("sparse Q, free cols", make_qp(40, 80, false, true, 12), opts, *vk);
    parity("diagonal Q, ranged", make_qp(30, 60, true, false, 13), opts, *vk);
    {
        // Halpern is off by default and does not converge here; 500 capped
        // iterations still pin every step of the reflected update.
        auto o = opts;
        o.max_iterations = 500;
        o.use_reflected_halpern = true;
        o.halpern_theta = 0.3;
        parity("sparse Q, Halpern", make_qp(30, 60, false, false, 14), o, *vk);
    }
    {
        // 300k columns: reductions clamp to 1024 groups and grid-stride.
        auto o = opts;
        o.max_iterations = 30;
        parity("grid-stride, capped", make_qp(2000, 300000, true, false, 15), o, *vk);
    }

    // Epoch batching (measured 2026-09-23): inner_epoch == 1 must be
    // the untouched path (redundant with the "sparse Q" case above, but
    // explicit -- this is the exact claim the epoch code makes about
    // itself), and inner_epoch > 1 is a real change to the iterate sequence
    // that must still land CPU and Vulkan on the same iteration counts and
    // the same point, because the blind interior iterations are recorded
    // identically on both devices (PdhcgDevice::inner_advance_blind's
    // default on the CPU, VulkanPdhcgDevice's override on the GPU -- same
    // formulas, same order, the override only skips readbacks).
    {
        auto o = opts;
        o.inner_epoch = 1;
        parity("epoch=1 (explicit)", make_qp(30, 60, false, false, 11), o, *vk);
    }
    {
        // epoch=3: most outer iterations need an epoch of 2 blind + 1
        // boundary; some need only the boundary (remaining < epoch near the
        // inner iteration cap).  "sparse Q" above averages ~4.6 inner
        // iterations per outer step, so this instance exercises both.
        auto o = opts;
        o.inner_epoch = 3;
        parity("epoch=3", make_qp(30, 60, false, false, 11), o, *vk);
        parity("epoch=3, free cols", make_qp(40, 80, false, true, 12), o, *vk);
    }
    {
        // epoch=25 versus inner_max_iterations=100 (opts' default): most
        // epochs are cut short by "remaining - 1" well before 24 blind
        // iterations, the case the min() in qp_pdhcg.cpp's loop exists for.
        auto o = opts;
        o.inner_epoch = 25;
        parity("epoch=25", make_qp(30, 60, false, false, 11), o, *vk);
    }

    return ::sor::test::finish("test_pdhcg_device");
}
