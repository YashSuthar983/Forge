// Why is dual_bound_finite false on converged HPR runs?
#include "sor/engines/hpr.hpp"
#include "sor/io/mps.hpp"
#include "sor/sparse/csc.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
using namespace sor;
static const double kInf = std::numeric_limits<double>::infinity();

int main(int argc, char** argv) {
    for (int a = 1; a < argc; ++a) {
        std::ifstream in(argv[a]); io::MpsReadReport rep;
        auto p = io::read_mps(in, rep);
        engines::HprOptions o; o.max_iterations = 200000; o.time_limit_s = 20.0;
        auto dev = backend::make_cpu_lp_device();
        engines::HprDiagnostics d;
        auto r = engines::solve_hpr(p, o, *dev, d);

        // reduced costs in ORIGINAL space: rc = c + A^T y
        const auto nc = (std::size_t)p.n_cols();
        std::vector<double> rc(p.c.begin(), p.c.end());
        const auto& rp = p.A.pattern.row_ptr(); const auto& ci = p.A.pattern.col_idx();
        for (std::size_t i = 0; i < (std::size_t)p.n_rows(); ++i)
            for (auto k = rp[i]; k < rp[i+1]; ++k)
                rc[(std::size_t)ci[(std::size_t)k]] += p.A.vals[(std::size_t)k] * r.y[i];

        int trip = 0; double worst = 0.0; int exact_zero = 0;
        for (std::size_t j = 0; j < nc; ++j) {
            const double b = (rc[j] >= 0.0) ? p.col_lo[j] : p.col_hi[j];
            if (rc[j] == 0.0) { ++exact_zero; continue; }
            if (std::isinf(b)) { ++trip; worst = std::max(worst, std::fabs(rc[j])); }
        }
        std::string nm(argv[a]); nm = nm.substr(nm.find_last_of('/')+1);
        std::printf("%-12s cols=%4zu dual_bound_finite=%s | cols tripping inf-bound test: %d "
                    "(largest |rc| among them = %.3e, exact-zero rc = %d)\n",
                    nm.c_str(), nc, d.dual_bound_finite ? "yes" : "NO", trip, worst, exact_zero);
    }
}
