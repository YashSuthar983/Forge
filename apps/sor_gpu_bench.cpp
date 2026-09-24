// sor_gpu_bench — hardware characterisation of THIS machine's GPU through
// SOR's own Vulkan path (vkc::Compute, the plumbing every QP device uses).
//
// LAYER L8 (tool).  Not in any solve path.  Measures, in order:
//   1. fp32 vs fp64 FMA throughput (bench_fma_{f32,f64}.comp)
//   2. achieved device-memory bandwidth (vec_axpby, vec_reduce, buffer copy)
//   3. submit / dispatch / host-decision latency, as the solver pays it
//      (Compute::rec + flush, including descriptor-set lookup)
//   4. host<->device transfer bandwidth: the solver's path (fresh staging
//      buffer per call, what TransferStats charges) and the raw DMA path
//      (pre-mapped host buffer, copy only)
//   5. vector-op crossover: per-call GPU axpby vs single-thread CPU axpby
//      over n, the GPU side charged its submit when it stands alone
//
// Every time here is host wall clock around a completed submit (the solver's
// own accounting), median of repeats unless stated.  Run with SOR_VK_PROFILE=1
// to get the device-side timestamp view of the same kernels on stderr.
// A number printed here belongs to the machine it ran on.
#include "vk_compute.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using sor::backend::vkc::Buf;
using sor::backend::vkc::Compute;
using sor::backend::vkc::Kernel;
namespace vkc = sor::backend::vkc;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

template <class F>
double median_ms(int reps, F&& f) {
    std::vector<double> t;
    t.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        f();
        t.push_back(ms_since(t0));
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Keeps the CPU reference loop from being optimised away.
volatile double g_sink = 0.0;

void cpu_axpby(std::size_t n, double a, const double* x, double b, const double* y,
               double* z) {
    for (std::size_t i = 0; i < n; ++i) z[i] = a * x[i] + b * y[i];
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    const int reps = quick ? 5 : 21;

    auto ctx = sor::backend::vk::Context::create(-1);
    if (!ctx) {
        std::fprintf(stderr, "sor_gpu_bench: no Vulkan device\n");
        return 2;
    }
    if (!ctx->info().shader_float64) {
        std::fprintf(stderr, "sor_gpu_bench: device lacks shaderFloat64\n");
        return 2;
    }
    Compute vk(std::move(ctx));
    vk.alloc_reduction_scratch();
    std::printf("device: %s\n", vk.ctx().info().name.c_str());

    // ------------------------------------------------ 1. FMA throughput
    {
        std::printf("\n[1] FMA throughput (8 independent chains/thread, 16 flop/iter)\n");
        const uint32_t groups = 8192;                  // 2,097,152 threads
        const std::size_t threads = std::size_t(groups) * 256;
        Buf out = vk.dev_zero(threads);                // fits f64 and f32
        Kernel k32 = vk.make_kernel("bench_fma_f32", 1, 8);
        Kernel k64 = vk.make_kernel("bench_fma_f64", 1, 8);
        auto run = [&](const Kernel& k, uint32_t iters) {
            struct { uint32_t iters, pad; } pc{iters, 0};
            vk.rec(k, {&out}, &pc, sizeof(pc), 0, groups);
            vk.flush();
        };
        run(k32, 16);  // warm-up (pipeline first use)
        run(k64, 16);
        for (int prec = 0; prec < 2; ++prec) {
            const Kernel& k = prec == 0 ? k32 : k64;
            // Grow the loop until one dispatch takes >= 50 ms, so the submit
            // (~tens of us) is noise in the figure.
            uint32_t iters = 64;
            double t = 0.0;
            for (;;) {
                t = median_ms(3, [&] { run(k, iters); });
                if (t >= 50.0 || iters >= (1u << 22)) break;
                iters *= 2;
            }
            t = median_ms(quick ? 3 : 7, [&] { run(k, iters); });
            const double flops = double(threads) * iters * 16.0;
            std::printf("  %s  iters %8u  %9.3f ms  %9.1f GFLOP/s\n",
                        prec == 0 ? "fp32" : "fp64", iters, t, flops / (t * 1e6));
        }
        vk.kill_kernel(k32);
        vk.kill_kernel(k64);
        vk.destroy_buf(out);
        vk.reset_sets();
    }

    // ------------------------------------------------ 2. device bandwidth
    {
        std::printf("\n[2] device-memory bandwidth (n = 2^25 doubles = 256 MiB/vector)\n");
        const std::size_t n = std::size_t(1) << 25;
        Buf x = vk.dev_zero(n), y = vk.dev_zero(n), z = vk.dev_zero(n);
        const int inner = 10;   // dispatches per submit
        auto bw = [&](const char* what, double bytes_per, auto&& body) {
            body();
            vk.flush();   // warm-up
            const double t = median_ms(quick ? 3 : 7, [&] {
                for (int i = 0; i < inner; ++i) body();
                vk.flush();
            }) / inner;
            std::printf("  %-34s %8.3f ms/op  %7.1f GB/s\n", what, t,
                        bytes_per / (t * 1e6));
        };
        bw("axpby z=ax+by (2R+1W)", 24.0 * n, [&] { vk.axpby(n, 1.0, x, 2.0, y, z); });
        bw("dot reduce (2R, 2-pass)", 16.0 * n,
           [&] { vk.reduce(vkc::kOpDot, n, x, y, 0); });
        bw("vkCmdCopyBuffer (1R+1W)", 16.0 * n, [&] { vk.copy(x, z); });
        vk.destroy_buf(x);
        vk.destroy_buf(y);
        vk.destroy_buf(z);
        vk.reset_sets();
    }

    // ------------------------------------------------ 3. latency
    {
        std::printf("\n[3] latency, as the solver pays it (median of %d)\n", reps * 10);
        Buf a = vk.dev_zero(256), b = vk.dev_zero(256), c = vk.dev_zero(256);
        for (int i = 0; i < 20; ++i) { vk.fill_zero(a); vk.flush(); }
        const double t_fill = median_ms(reps * 10, [&] { vk.fill_zero(a); vk.flush(); });
        const double t_disp = median_ms(reps * 10, [&] {
            vk.axpby(1, 1.0, a, 1.0, b, c);
            vk.flush();
        });
        const double t_decide = median_ms(reps * 10, [&] {
            vk.reduce(vkc::kOpDot, 256, a, b, 0);
            std::vector<double> s(1);
            vk.read_scalars(s);
        });
        const int burst = 2000;
        const double t_burst = median_ms(quick ? 3 : 9, [&] {
            for (int i = 0; i < burst; ++i) vk.axpby(1, 1.0, a, 1.0, b, c);
            vk.flush();
        });
        std::printf("  submit+wait, 1 fill (min roundtrip)   %8.1f us\n", t_fill * 1e3);
        std::printf("  submit+wait, 1 tiny dispatch          %8.1f us\n", t_disp * 1e3);
        std::printf("  host decision (2-pass reduce + read)  %8.1f us\n", t_decide * 1e3);
        std::printf("  in-stream dispatch+barrier (burst %d) %6.2f us each\n", burst,
                    t_burst * 1e3 / burst);
        vk.destroy_buf(a);
        vk.destroy_buf(b);
        vk.destroy_buf(c);
        vk.reset_sets();
    }

    // ------------------------------------------------ 4. transfer
    {
        std::printf("\n[4] host<->device transfer\n");
        std::printf("  %12s  %14s %14s   %14s %14s\n", "bytes", "solver h2d", "solver d2h",
                    "raw DMA h2d", "raw DMA d2h");
        const std::size_t max_n = std::size_t(1) << 25;   // 256 MiB
        Buf dev = vk.dev_zero(max_n);
        Buf host = vk.host_vec(max_n);
        std::vector<double> hv(max_n, 1.5);
        for (std::size_t n = 1; n <= max_n; n *= 16) {
            std::vector<double> v(hv.begin(), hv.begin() + std::ptrdiff_t(n));
            const int r = n >= (std::size_t(1) << 21) ? (quick ? 3 : 5) : reps;
            const double t_up = median_ms(r, [&] { vk.upload(dev, v); });
            const double t_dn = median_ms(r, [&] { vk.download_range(dev, 0, v); });
            // Raw: the host buffer is already mapped; charge the memcpy into
            // it (h2d) / out of it (d2h) plus the DMA copy and its submit.
            Buf dev_view = dev;  dev_view.size = n * 8;   // copy() takes min size
            Buf host_view = host; host_view.size = n * 8;
            const double t_rup = median_ms(r, [&] {
                std::memcpy(host.mapped, v.data(), n * 8);
                vk.copy(host_view, dev_view);
                vk.flush();
            });
            const double t_rdn = median_ms(r, [&] {
                vk.copy(dev_view, host_view);
                vk.flush();
                std::memcpy(v.data(), host.mapped, n * 8);
            });
            const double bytes = double(n) * 8.0;
            auto gbs = [&](double t) { return bytes / (t * 1e6); };
            std::printf("  %12.0f  %7.3fms %5.2fGB/s %7.3fms %5.2fGB/s   "
                        "%7.3fms %5.2fGB/s %7.3fms %5.2fGB/s\n",
                        bytes, t_up, gbs(t_up), t_dn, gbs(t_dn), t_rup, gbs(t_rup), t_rdn,
                        gbs(t_rdn));
        }
        vk.destroy_buf(dev);
        vk.destroy_buf(host);
        vk.reset_sets();
    }

    // ------------------------------------------------ 5. vector-op crossover
    {
        std::printf("\n[5] axpby crossover: GPU vs single-thread CPU, per call\n");
        std::printf("  %10s  %12s %12s %12s   %s\n", "n", "CPU us", "GPU in-stream",
                    "GPU +submit", "GPU/CPU (in-stream)");
        const std::size_t max_n = std::size_t(1) << 25;
        Buf x = vk.dev_zero(max_n), y = vk.dev_zero(max_n), z = vk.dev_zero(max_n);
        std::vector<double> hx(max_n, 1.0), hy(max_n, 2.0), hz(max_n, 0.0);
        for (std::size_t n = 1024; n <= max_n; n *= 4) {
            const int inner = int(std::clamp<std::size_t>((std::size_t(1) << 26) / n, 4, 500));
            const double t_cpu = median_ms(quick ? 3 : 7, [&] {
                for (int i = 0; i < inner; ++i)
                    cpu_axpby(n, 1.0, hx.data(), 2.0 + i * 1e-9, hy.data(), hz.data());
                g_sink = g_sink + hz[n / 2];
            }) / inner;
            vk.axpby(n, 1.0, x, 2.0, y, z);
            vk.flush();
            const double t_gpu = median_ms(quick ? 3 : 7, [&] {
                for (int i = 0; i < inner; ++i) vk.axpby(n, 1.0, x, 2.0, y, z);
                vk.flush();
            }) / inner;
            const double t_one = median_ms(reps, [&] {
                vk.axpby(n, 1.0, x, 2.0, y, z);
                vk.flush();
            });
            std::printf("  %10zu  %12.2f %12.2f %12.2f   %8.2fx %s\n", n, t_cpu * 1e3,
                        t_gpu * 1e3, t_one * 1e3, t_gpu / t_cpu,
                        t_gpu < t_cpu ? "GPU faster" : "CPU faster");
        }
        vk.destroy_buf(x);
        vk.destroy_buf(y);
        vk.destroy_buf(z);
        vk.reset_sets();
    }
    return 0;
}
