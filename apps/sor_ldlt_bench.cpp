// sor_ldlt_bench — time and check the sparse LDL' on a dumped KKT system.
//
// Developer tool, not a solver front end.  Get a system with
//     SOR_DUMP_KKT=/path/k.bin sor_solve --engine qpipm model.qplib
// then
//     sor_ldlt_bench /path/k.bin [--threads 1,2,4,8] [--reps 3] [--solves 10]
//                    [--no-amalg] [--relax SMALL,ZEROS,MAXCOLS]
// Reports analyze time, fill, factorization and solve times per thread count
// (median of --reps), the machine load (/proc/loadavg) next to every timing,
// and whether the factor and the solves are BIT-IDENTICAL across thread
// counts (the determinism contract of sor::la::Ldlt).  Exit status 1 when
// they are not.
#include "sor/core/parallel.hpp"
#include "sor/la/ldlt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace sor;
using la::Index;
using la::Offset;
using Clock = std::chrono::steady_clock;

namespace {
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
std::string loadavg() {
    std::ifstream f("/proc/loadavg");
    std::string a, b, c;
    f >> a >> b >> c;
    return a + " " + b + " " + c;
}
std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::atoi(t.c_str()));
    return v;
}
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
std::uint64_t hash_vec(const std::vector<double>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (double x : v) {
        std::uint64_t u;
        std::memcpy(&u, &x, sizeof u);
        h ^= u;
        h *= 1099511628211ULL;
    }
    return h;
}
std::vector<double> matvec(const la::SymCsc& a, const std::vector<double>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (Index j = 0; j < a.n; ++j)
        for (Offset t = a.col_ptr[static_cast<std::size_t>(j)]; t < a.col_ptr[static_cast<std::size_t>(j) + 1]; ++t) {
            const auto i = static_cast<std::size_t>(a.row_idx[static_cast<std::size_t>(t)]);
            const double v = a.vals[static_cast<std::size_t>(t)];
            y[i] += v * x[static_cast<std::size_t>(j)];
            if (i != static_cast<std::size_t>(j)) y[static_cast<std::size_t>(j)] += v * x[i];
        }
    return y;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s kkt.bin [--threads 1,2,4,8] [--reps 3] [--solves 10] "
                             "[--no-amalg] [--relax SMALL,ZEROS,MAXCOLS]\n", argv[0]);
        return 2;
    }
    std::vector<int> threads{1, 2, 4, 8};
    int reps = 3, solves = 10;
    la::LdltOptions opt;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--threads") threads = parse_list(next());
        else if (a == "--reps") reps = std::atoi(next().c_str());
        else if (a == "--solves") solves = std::atoi(next().c_str());
        else if (a == "--no-amalg") opt.amalgamate = false;
        else if (a == "--relax") {
            const std::string v = next();
            std::stringstream ss(v);
            std::string t;
            std::getline(ss, t, ','); opt.relax_small = std::atoi(t.c_str());
            std::getline(ss, t, ','); opt.relax_zeros = std::atof(t.c_str());
            std::getline(ss, t, ','); opt.relax_max_cols = std::atoi(t.c_str());
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    la::SymCsc K;
    std::vector<std::int8_t> sign;
    la::read_sym_csc(argv[1], K, sign);
    std::printf("kkt %s: n=%d nnz(upper)=%zu\n", argv[1], K.n, K.row_idx.size());
    std::printf("amalgamation: %s small=%d zeros=%.3f maxcols=%d\n", opt.amalgamate ? "on" : "off",
                opt.relax_small, opt.relax_zeros, opt.relax_max_cols);

    la::Ldlt f;
    f.set_options(opt);
    auto t0 = Clock::now();
    f.analyze(K);
    auto t1 = Clock::now();
    std::printf("analyze %.0f ms  nnzL=%lld stored=%lld (+%.1f%%) supernodes=%d flops=%.3e stored_flops=%.3e"
                "  mem=%.0f MB  load %s\n",
                ms(t0, t1), static_cast<long long>(f.nnz_l()), static_cast<long long>(f.stored_l()),
                100.0 * static_cast<double>(f.stored_l() - f.nnz_l()) / static_cast<double>(f.nnz_l()),
                f.supernodes(), f.flops(), f.stored_flops(), static_cast<double>(f.memory_bytes()) / 1e6,
                loadavg().c_str());

    std::vector<double> xtrue(static_cast<std::size_t>(K.n));
    std::uint64_t s = 12345;
    for (auto& v : xtrue) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
    }
    const auto b0 = matvec(K, xtrue);

    std::uint64_t ref_fp = 0, ref_sol = 0;
    double t_fac1 = 0.0, t_sol1 = 0.0;
    bool identical = true;
    for (std::size_t ti = 0; ti < threads.size(); ++ti) {
        const int p = threads[ti];
        core::set_global_threads(p);
        std::vector<double> tf, ts;
        std::uint64_t fp = 0, sh = 0;
        double resid = 0.0;
        int nreg = 0;
        const std::string load_before = loadavg();
        for (int r = 0; r < reps; ++r) {
            auto a0 = Clock::now();
            const bool ok = f.factorize(K, sign, 1e-12);
            auto a1 = Clock::now();
            if (!ok) std::printf("  factorize returned false (non-finite pivot)\n");
            tf.push_back(ms(a0, a1));
            fp = f.fingerprint();
            nreg = f.regularized_pivots();
            std::vector<double> x;
            auto c0 = Clock::now();
            for (int k = 0; k < solves; ++k) {
                x = b0;
                f.solve(x);
            }
            auto c1 = Clock::now();
            ts.push_back(ms(c0, c1) / std::max(1, solves));
            sh = hash_vec(x);
            const auto kx = matvec(K, x);
            double rn = 0.0, bn = 0.0;
            for (std::size_t i = 0; i < kx.size(); ++i) {
                rn = std::max(rn, std::fabs(kx[i] - b0[i]));
                bn = std::max(bn, std::fabs(b0[i]));
            }
            resid = rn / (1.0 + bn);
        }
        const double mf = median(tf), msol = median(ts);
        if (ti == 0) {
            ref_fp = fp;
            ref_sol = sh;
            t_fac1 = mf;
            t_sol1 = msol;
        }
        const bool same = fp == ref_fp && sh == ref_sol;
        identical = identical && same;
        std::printf("threads=%d  factor %.0f ms (x%.2f)  solve %.1f ms (x%.2f)  reg=%d  resid=%.1e"
                    "  factor_fp=%016llx solve_fp=%016llx %s  load %s -> %s\n",
                    p, mf, t_fac1 / mf, msol, t_sol1 / msol, nreg, resid,
                    static_cast<unsigned long long>(fp), static_cast<unsigned long long>(sh),
                    same ? "IDENTICAL" : "DIFFERENT", load_before.c_str(), loadavg().c_str());
        std::fflush(stdout);
    }
    std::printf("bit-identical across thread counts: %s\n", identical ? "yes" : "NO");
    return identical ? 0 : 1;
}
