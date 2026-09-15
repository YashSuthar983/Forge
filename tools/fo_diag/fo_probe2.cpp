// Test: (a) primal-weight sign, (b) restart granularity, (c) Halpern warmup=0.
#include "sor/engines/hpr.hpp"
#include "sor/io/mps.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
using namespace sor;

struct Cfg {
    const char* name; bool hal, wgt, rst; double kp; std::uint64_t chk; std::uint64_t warm;
};

int main(int argc, char** argv) {
    const std::vector<Cfg> cfgs = {
        {"rst/chk200",    false,false,true,  0.15, 200, 400},
        {"rst/chk40",     false,false,true,  0.15,  40, 400},
        {"rst/chk10",     false,false,true,  0.15,  10, 400},
        {"wgt +0.15",     false,true, false, 0.15, 200, 400},
        {"wgt -0.15(flip)",false,true,false,-0.15, 200, 400},
        {"wgt-flip+rst",  false,true, true, -0.15, 200, 400},
        {"hal warm=0",    true, false,false, 0.15, 200,   0},
        {"hal warm=0+rst",true, false,true,  0.15, 200,   0},
        {"hal w0+rst/chk40",true,false,true, 0.15,  40,   0},
    };
    for (int a = 1; a < argc; ++a) {
        std::ifstream in(argv[a]); io::MpsReadReport rep;
        auto p = io::read_mps(in, rep);
        std::string nm(argv[a]); nm = nm.substr(nm.find_last_of('/') + 1);
        for (const auto& c : cfgs) {
            engines::HprOptions o;
            o.max_iterations = 200000; o.time_limit_s = 20.0;
            o.check_every = c.chk; o.halpern_warmup = c.warm;
            o.use_halpern = c.hal; o.use_primal_weight = c.wgt; o.use_restart = c.rst;
            o.pid_kp = c.kp;
            auto dev = backend::make_cpu_lp_device();
            engines::HprDiagnostics d;
            engines::solve_hpr(p, o, *dev, d);
            std::printf("%-12s %-18s it=%8llu pres=%9.2e dres=%9.2e rst=%4llu w=%8.2e obj=%.6e %s\n",
                nm.c_str(), c.name, (unsigned long long)d.iterations,
                d.primal_residual, d.dual_residual, (unsigned long long)d.restarts,
                d.final_primal_weight, d.primal_objective, d.termination_reason.c_str());
        }
        std::printf("\n");
    }
}
