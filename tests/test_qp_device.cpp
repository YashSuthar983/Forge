// QpDevice parity: the Vulkan device against cpu_qp_device.cpp, which is the
// oracle by definition.  Skips (passes) when no fp64 Vulkan device exists.
//
// The dispatch ladder only has one-row, diagonal-Q instances, so this
// covers what it cannot: many rows, off-diagonal Q, Q = 0, free and
// one-sided bounds (which drive the infinite-support-term flags in the
// dual objective), and vectors long enough that the reductions need more
// than one workgroup in pass 1 and grid-stride past kMaxPartials groups.
#include "sor/backend/qp_device.hpp"
#include "sor/sparse/csc.hpp"
#include "sor/sparse/csr.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace sor;
using core::f64;
using core::Index;

namespace {

constexpr f64 kInf = std::numeric_limits<f64>::infinity();

// Deterministic, platform-independent generator (no <random> distributions,
// whose output is implementation-defined).
struct Lcg {
    std::uint64_t s;
    f64 next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<f64>(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

struct Case {
    backend::ScaledQp qp;
    backend::QpStepParams p;
};

Case make_case(Index m, Index n, bool with_q, int row_nnz, std::uint64_t seed) {
    Lcg g{seed};
    Case c;
    std::vector<Index> r, cc;
    std::vector<f64> v;
    f64 frob2 = 0.0;
    for (Index i = 0; i < m; ++i)
        for (int k = 0; k < row_nnz; ++k) {
            const Index j = static_cast<Index>(g.next() * static_cast<f64>(n)) % n;
            const f64 a = 2.0 * g.next() - 1.0;
            r.push_back(i); cc.push_back(j); v.push_back(a);
            frob2 += a * a;
        }
    c.qp.A_csr = sparse::from_triplets(m, n, r, cc, v);
    c.qp.A_csc = sparse::to_csc(c.qp.A_csr);

    f64 gersh = 0.0;
    if (with_q) {
        // Symmetric, diagonally dominant with positive diagonal => PSD.
        std::vector<Index> qr, qc;
        std::vector<f64> qv;
        std::vector<f64> rowsum(static_cast<std::size_t>(n), 0.0);
        for (Index j = 0; j + 1 < n; j += 2) {
            const f64 o = 0.5 * (2.0 * g.next() - 1.0);
            qr.push_back(j); qc.push_back(j + 1); qv.push_back(o);
            qr.push_back(j + 1); qc.push_back(j); qv.push_back(o);
            rowsum[static_cast<std::size_t>(j)] += std::fabs(o);
            rowsum[static_cast<std::size_t>(j + 1)] += std::fabs(o);
        }
        for (Index j = 0; j < n; ++j) {
            const f64 d = rowsum[static_cast<std::size_t>(j)] + 0.1 + g.next();
            qr.push_back(j); qc.push_back(j); qv.push_back(d);
            gersh = std::max(gersh, d + rowsum[static_cast<std::size_t>(j)]);
        }
        c.qp.Q_csr = sparse::from_triplets(n, n, qr, qc, qv);
    }

    for (Index j = 0; j < n; ++j) {
        c.qp.c.push_back(2.0 * g.next() - 1.0);
        switch (j % 4) {   // boxed, lower only, upper only, free
            case 0: c.qp.col_lo.push_back(-1.0); c.qp.col_hi.push_back(2.0); break;
            case 1: c.qp.col_lo.push_back(0.0);  c.qp.col_hi.push_back(kInf); break;
            case 2: c.qp.col_lo.push_back(-kInf); c.qp.col_hi.push_back(1.0); break;
            default: c.qp.col_lo.push_back(-kInf); c.qp.col_hi.push_back(kInf); break;
        }
    }
    for (Index i = 0; i < m; ++i) {
        const f64 b = 2.0 * g.next() - 1.0;
        switch (i % 3) {   // equality, ranged, one-sided
            case 0: c.qp.row_lo.push_back(b); c.qp.row_hi.push_back(b); break;
            case 1: c.qp.row_lo.push_back(b - 1.0); c.qp.row_hi.push_back(b + 1.0); break;
            default: c.qp.row_lo.push_back(-kInf); c.qp.row_hi.push_back(b); break;
        }
    }
    c.p.sigma = 0.7;
    c.p.lambda_Q = with_q ? gersh : 1.0;
    // ||A||_F^2 bounds lambda_1(A A'); doubled because from_triplets sums
    // duplicate entries, which can exceed the sum of their squares.
    c.p.lambda_A = std::max(1.0, 2.0 * frob2);
    return c;
}

bool near(f64 a, f64 b, f64 tol) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
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

void run_parity(const std::string& label, const Case& c, backend::QpDevice& vk) {
    constexpr f64 tol = 1e-9;
    auto cpu = backend::make_cpu_qp_device();
    cpu->upload(c.qp);
    vk.upload(c.qp);
    cpu->init_zero();
    vk.init_zero();

    // Three epochs with a restart between each: exercises the anchors and
    // the inner Halpern counter reset, not just a single run of steps.
    for (int epoch = 0; epoch < 3; ++epoch) {
        cpu->hpr_qp_steps(40, c.p);
        vk.hpr_qp_steps(40, c.p);

        const auto a = vk.reduce_kkt(c.p);
        const auto b = cpu->reduce_kkt(c.p);
        const bool ok =
            near(a.primal_res, b.primal_res, tol) && near(a.dual_res, b.dual_res, tol) &&
            near(a.primal_obj, b.primal_obj, tol) && near(a.dual_obj, b.dual_obj, tol) &&
            near(a.gap_rel, b.gap_rel, tol) &&
            near(a.restart_metric, b.restart_metric, tol) &&
            a.dual_bound_finite == b.dual_bound_finite;
        ::sor::test::report(ok, "reduce_kkt parity", __FILE__, __LINE__,
                            label + " epoch " + std::to_string(epoch));

        const auto sa = vk.sigma_coefficients(c.p);
        const auto sb = cpu->sigma_coefficients(c.p);
        ::sor::test::report(near(sa.theta1, sb.theta1, tol) &&
                                near(sa.theta2, sb.theta2, tol) &&
                                near(sa.theta3, sb.theta3, tol),
                            "sigma_coefficients parity", __FILE__, __LINE__, label);
        cpu->restart();
        vk.restart();
    }

    backend::QpSolution sa, sb;
    vk.download(sa);
    cpu->download(sb);
    ::sor::test::report(max_rel(sa.x, sb.x) < tol && max_rel(sa.y, sb.y) < tol &&
                            max_rel(sa.z, sb.z) < tol && max_rel(sa.w, sb.w) < tol,
                        "download parity", __FILE__, __LINE__, label);

    CHECK(vk.transfer_stats().h2d_bytes > 0);
}

}  // namespace

int main() {
    CHECK(backend::make_qp_device("cpu") != nullptr);
    CHECK(backend::make_qp_device("no_such") == nullptr);

    auto vk = backend::make_vulkan_qp_device();
    if (!vk) {
        std::printf("SKIP vulkan parity: no fp64 Vulkan device\n");
        return ::sor::test::finish("test_qp_device");
    }
    CHECK(vk->is_accelerated());
    const auto caps = vk->capabilities();
    CHECK(caps.fused_steps && caps.device_reduction && caps.restart &&
          caps.sigma_coefficients);

    run_parity("small Q", make_case(7, 12, true, 3, 1), *vk);
    run_parity("Q = 0", make_case(9, 15, false, 4, 2), *vk);
    run_parity("multi-group", make_case(700, 2000, true, 6, 3), *vk);
    // 300k columns: pass 1 wants 1172 groups, clamps to 1024 and strides.
    run_parity("grid-stride", make_case(5, 300000, true, 2, 4), *vk);

    // Scalar-only KKT: a check after upload moves 17 doubles, not vectors.
    {
        const Case c = make_case(50, 400, true, 4, 5);
        vk->upload(c.qp);
        vk->init_zero();
        vk->hpr_qp_steps(5, c.p);
        vk->reset_stats();
        (void)vk->reduce_kkt(c.p);
        CHECK(vk->transfer_stats().d2h_bytes == 17 * sizeof(f64));
        CHECK(vk->transfer_stats().h2d_bytes == 0);
    }

    return ::sor::test::finish("test_qp_device");
}
