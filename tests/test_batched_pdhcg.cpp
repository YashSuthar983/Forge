// Batched PDHCG-II: K sibling solves in one pass.
//
// Phase 4 gate: lane l of a batch equals a separate solve with lane l's
// bounds.  Checked two ways:
//   * CPU lanes (make_lanes_device) vs K sequential solve_qp_pdhcg: must be
//     BIT-identical -- same loop, same decisions, same arithmetic;
//   * Vulkan batched device vs CPU lanes: same iteration counts per lane and
//     agreement to rounding.
// Lanes differ as branch-and-bound children do, by fixed variables, and are
// built so they converge at different iterations -- the masking is the
// thing being tested.  Vulkan part skips without an fp64 device.
#include "sor/backend/batched_pdhcg_device.hpp"
#include "sor/engines/qp.hpp"
#include "sor/sparse/csc.hpp"
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

struct Lcg {
    std::uint64_t s;
    f64 next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<f64>(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

engines::QpProblem make_qp(Index m, Index n, bool diagonal, std::uint64_t seed) {
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
        lp.col_lo.push_back(0.0);
        lp.col_hi.push_back(1.0);
    }
    for (Index i = 0; i < m; ++i) {
        const f64 b = 0.3 * (2.0 * g.next() - 1.0);
        lp.row_lo.push_back(b - 1.5);
        lp.row_hi.push_back(b + 1.5);
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

// Lane 0 is the root; lane l fixes l variables to 0 or 1, as a B&B child.
std::vector<backend::LaneBounds> children(const engines::QpProblem& p, std::size_t K) {
    std::vector<backend::LaneBounds> out;
    for (std::size_t l = 0; l < K; ++l) {
        backend::LaneBounds b{p.linear.col_lo, p.linear.col_hi, p.linear.row_lo,
                              p.linear.row_hi};
        for (std::size_t t = 0; t < l; ++t) {
            const std::size_t j = (t * 7 + l) % b.col_lo.size();
            const f64 val = ((t + l) % 2) ? 1.0 : 0.0;
            b.col_lo[j] = b.col_hi[j] = val;
        }
        out.push_back(std::move(b));
    }
    return out;
}

f64 max_rel(const std::vector<f64>& a, const std::vector<f64>& b) {
    if (a.size() != b.size()) return 1e300;
    f64 scale = 1.0, d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        scale = std::max(scale, std::fabs(b[i]));
        d = std::max(d, std::fabs(a[i] - b[i]));
    }
    return d / scale;
}

void run(const std::string& label, bool diagonal, bool halpern,
         backend::BatchedPdhcgDevice* vk) {
    const auto p = make_qp(25, 50, diagonal, diagonal ? 21 : 22);
    const std::size_t K = 6;
    const auto lanes = children(p, K);
    engines::QpOptions o;
    o.max_iterations = 5000;
    if (halpern) {
        // With the reflection on, a lane's x depends on x_prev and the anchor,
        // so any step applied to a finished lane shows up in its result.
        o.use_reflected_halpern = true;
        o.halpern_theta = 0.2;
        o.max_iterations = 600;
    }

    // K sequential single solves: the reference.
    std::vector<core::RawResult> seq;
    std::vector<engines::QpDiagnostics> seq_d(K);
    for (std::size_t l = 0; l < K; ++l) {
        auto q = p;
        q.linear.col_lo = lanes[l].col_lo;
        q.linear.col_hi = lanes[l].col_hi;
        auto dev = backend::make_cpu_pdhcg_device();
        seq.push_back(engines::solve_qp_pdhcg(q, o, *dev, seq_d[l]));
    }

    // CPU lanes: bit-identical.
    std::vector<std::unique_ptr<backend::PdhcgDevice>> devs;
    for (std::size_t l = 0; l < K; ++l) devs.push_back(backend::make_cpu_pdhcg_device());
    auto cpu = backend::make_lanes_device(std::move(devs));
    std::vector<engines::QpDiagnostics> bd;
    const auto br = engines::solve_qp_pdhcg_batch(p, lanes, o, *cpu, bd);
    std::uint64_t lo_it = ~0ull, hi_it = 0;
    for (std::size_t l = 0; l < K; ++l) {
        const bool same = br[l].x == seq[l].x && br[l].y == seq[l].y &&
                          bd[l].iterations == seq_d[l].iterations &&
                          bd[l].inner_iterations == seq_d[l].inner_iterations &&
                          br[l].proposed_status == seq[l].proposed_status;
        ::sor::test::report(same, "CPU batch lane == sequential solve (bitwise)", __FILE__,
                            __LINE__, label + " lane " + std::to_string(l));
        lo_it = std::min(lo_it, bd[l].iterations);
        hi_it = std::max(hi_it, bd[l].iterations);
    }
    if (!halpern)
        ::sor::test::report(lo_it < hi_it, "lanes finish at different iterations", __FILE__,
                            __LINE__, label);
    std::printf("  %-8s K=%zu  lane iterations %llu..%llu\n", label.c_str(), K,
                static_cast<unsigned long long>(lo_it), static_cast<unsigned long long>(hi_it));

    if (!vk) return;
    std::vector<engines::QpDiagnostics> vd;
    const auto vr = engines::solve_qp_pdhcg_batch(p, lanes, o, *vk, vd);
    for (std::size_t l = 0; l < K; ++l) {
        const bool ok = vd[l].iterations == bd[l].iterations &&
                        vd[l].inner_iterations == bd[l].inner_iterations &&
                        vr[l].proposed_status == br[l].proposed_status &&
                        max_rel(vr[l].x, br[l].x) < 1e-9 && max_rel(vr[l].y, br[l].y) < 1e-9 &&
                        std::fabs(vd[l].objective - bd[l].objective) <=
                            1e-9 * (1.0 + std::fabs(bd[l].objective));
        ::sor::test::report(ok, "Vulkan batch lane matches CPU lane", __FILE__, __LINE__,
                            label + " lane " + std::to_string(l) + " iters " +
                                std::to_string(vd[l].iterations) + "/" +
                                std::to_string(bd[l].iterations));
    }
}

// Masking, tested directly: result comparisons cannot see it, because a
// finished lane sits at a near-fixed point and extra steps barely move it.
// So run every step with lane 0 masked OUT and demand lane 0's iterate be
// bit-for-bit unchanged while lane 1 (masked in) moves.
void masking(backend::BatchedPdhcgDevice& dev, bool diagonal) {
    const auto p = make_qp(25, 50, diagonal, 31);
    const auto lanes = children(p, 3);
    backend::PdhcgData d;
    d.A_csr = p.linear.A;
    d.A_csc = sparse::to_csc(p.linear.A);
    d.diagonal = diagonal;
    if (diagonal) d.q_diag = p.q_diag; else d.Q_csr = p.q_matrix;
    d.c = p.linear.c;
    dev.upload(d, lanes);
    dev.init();
    const backend::LaneMask all(3, 1), not0 = {0, 1, 1};
    // A few full steps first so every lane is somewhere non-trivial.
    auto step = [&](const backend::LaneMask& m) {
        dev.outer_begin(m);
        if (diagonal) {
            dev.diag_prox(std::vector<f64>(3, 0.1), m);
        } else {
            dev.inner_begin(m);
            for (int it = 0; it < 3; ++it) {
                (void)dev.inner_grad(std::vector<f64>(3, 0.1), m);
                (void)dev.inner_trial(std::vector<f64>(3, 0.05), std::vector<f64>(3, 0.1), m);
            }
            dev.inner_end(m);
        }
        (void)dev.dual_and_advance(std::vector<f64>(3, 0.1), true, 0.5, 0.2, true, m);
        (void)dev.evaluate(false, m);
    };
    for (int k = 0; k < 5; ++k) step(all);
    std::vector<f64> x0a, y0a, x1a, y1a, x0b, y0b, x1b, y1b;
    dev.download(0, x0a, y0a);
    dev.download(1, x1a, y1a);
    for (int k = 0; k < 5; ++k) step(not0);
    dev.download(0, x0b, y0b);
    dev.download(1, x1b, y1b);
    ::sor::test::report(x0a == x0b && y0a == y0b, "masked-out lane is untouched", __FILE__,
                        __LINE__, diagonal ? "diag Q" : "sparse Q");
    ::sor::test::report(x1a != x1b, "masked-in lane moves", __FILE__, __LINE__,
                        diagonal ? "diag Q" : "sparse Q");
}

}  // namespace

int main() {
    auto vk = backend::make_vulkan_batched_pdhcg_device();
    if (!vk) std::printf("SKIP vulkan batched parity: no fp64 Vulkan device\n");
    run("sparse Q", false, false, vk.get());
    run("diag Q", true, false, vk.get());
    run("Halpern", false, true, vk.get());
    {
        std::vector<std::unique_ptr<backend::PdhcgDevice>> devs;
        for (int l = 0; l < 3; ++l) devs.push_back(backend::make_cpu_pdhcg_device());
        auto cpu = backend::make_lanes_device(std::move(devs));
        masking(*cpu, false);
        masking(*cpu, true);
    }
    if (vk) {
        masking(*vk, false);
        masking(*vk, true);
    }
    return ::sor::test::finish("test_batched_pdhcg");
}
