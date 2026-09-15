// Isolate which HPR feature breaks the ladder.
#include "sor/engines/hpr.hpp"
#include "sor/io/mps.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace sor;

struct Cfg { const char* name; bool halpern, weight, restart, adaptive; };

int main(int argc, char** argv) {
    const std::vector<Cfg> cfgs = {
        {"vanilla",     false, false, false, false},
        {"restart",     false, false, true,  false},
        {"halpern",     true,  false, false, false},
        {"hal+rst",     true,  false, true,  false},
        {"weight",      false, true,  false, false},
        {"wgt+rst",     false, true,  true,  false},
        {"full(h+w+r)", true,  true,  true,  false},
        {"adaptive",    false, false, true,  true },
    };
    for (int a = 1; a < argc; ++a) {
        std::ifstream in(argv[a]);
        io::MpsReadReport rep;
        auto p = io::read_mps(in, rep);
        std::string nm(argv[a]);
        nm = nm.substr(nm.find_last_of('/') + 1);
        for (const auto& c : cfgs) {
            engines::HprOptions o;
            o.max_iterations = 200000;
            o.check_every = 200;
            o.time_limit_s = 20.0;
            o.use_halpern = c.halpern;
            o.use_primal_weight = c.weight;
            o.use_restart = c.restart;
            o.use_adaptive_step = c.adaptive;
            auto dev = backend::make_cpu_lp_device();
            engines::HprDiagnostics d;
            auto r = engines::solve_hpr(p, o, *dev, d);
            std::printf("%-12s %-12s it=%8llu pres=%9.2e dres=%9.2e restarts=%4llu w=%8.2e obj=%.6e %s\n",
                nm.c_str(), c.name,
                (unsigned long long)d.iterations, d.primal_residual, d.dual_residual,
                (unsigned long long)d.restarts, d.final_primal_weight, d.primal_objective,
                d.termination_reason.c_str());
        }
        std::printf("\n");
    }
    return 0;
}
