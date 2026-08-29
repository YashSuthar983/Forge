// sor_solve — prototype CLI.
//
// LAYER L8.
//
//   sor_solve model.mps [--backend cpu|julia_gpu] [--engine pdhg]
//                       [--max-iter N] [--tol T] [--verbose]
//
// Prints status, proof level, objective, residuals, and a timing breakdown that
// always includes host<->device transfer and IPC cost.
#include "sor/backend/kernel_backend.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/engines/pdhg.hpp"
#include "sor/io/mps.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace {

void usage() {
    std::fputs(
        "usage: sor_solve MODEL.mps [options]\n"
        "  --backend NAME   cpu (default) | julia_gpu (EXPERIMENTAL)\n"
        "  --engine NAME    pdhg (default; only engine in this prototype)\n"
        "  --max-iter N     iteration limit (default 100000)\n"
        "  --tol T          primal/dual tolerance (default 1e-6)\n"
        "  --fixed-mps      force fixed-column MPS parsing\n"
        "  --free-mps       force free-format MPS parsing\n"
        "                   (default: try free, fall back to fixed)\n"
        "  --verbose        print iteration log\n",
        stderr);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }

    std::string path, backend_name = "cpu", engine_name = "pdhg";
    sor::engines::PdhgOptions opts;
    sor::io::MpsReadOptions mps_opts;
    bool mps_format_forced = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (a == "--backend")  backend_name = next("--backend");
        else if (a == "--engine")   engine_name  = next("--engine");
        else if (a == "--max-iter") opts.max_iterations =
                                        std::strtoull(next("--max-iter").c_str(), nullptr, 10);
        else if (a == "--tol") {
            const double t = std::strtod(next("--tol").c_str(), nullptr);
            opts.primal_tol = opts.dual_tol = t;
        }
        else if (a == "--fixed-mps") { mps_opts.fixed_format = true;
                                       mps_format_forced = true; }
        else if (a == "--free-mps")  { mps_opts.fixed_format = false;
                                       mps_format_forced = true; }
        else if (a == "--verbose")   opts.verbose = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            return 2;
        }
        else path = a;
    }

    if (path.empty()) { usage(); return 2; }
    if (engine_name != "pdhg") {
        std::fprintf(stderr,
                     "error: engine '%s' is not implemented in this prototype "
                     "(only 'pdhg'). Refusing rather than substituting.\n",
                     engine_name.c_str());
        return 3;
    }

    try {
        sor::io::MpsReadReport rep;
        const auto problem = mps_format_forced
            ? sor::io::read_mps_file(path, rep, mps_opts)
            : sor::io::read_mps_file_auto(path, rep, mps_opts.strict);
        for (const auto& w : rep.warnings)
            std::fprintf(stderr, "warning: %s\n", w.c_str());

        std::printf("model:             %s\n",
                    problem.name.empty() ? path.c_str() : problem.name.c_str());
        std::printf("rows x cols:       %d x %d   nnz %lld\n",
                    problem.n_rows(), problem.n_cols(),
                    static_cast<long long>(problem.nnz()));
        if (rep.n_integer > 0) {
            std::printf("integer columns:   %zu\n", rep.n_integer);
            std::printf("NOTE:              solving the LP RELAXATION - this "
                        "prototype has no branch-and-bound\n");
        }

        auto be = sor::backend::make_backend(backend_name);
        if (!be) {
            std::fprintf(stderr,
                         "warning: backend '%s' unavailable; falling back to cpu\n",
                         backend_name.c_str());
            be = sor::backend::make_cpu_backend();
        }
        std::printf("backend:           %s (accelerated=%s)\n",
                    std::string(be->name()).c_str(),
                    be->is_accelerated() ? "yes" : "no");

        sor::engines::PdhgDiagnostics diag;
        auto raw = sor::engines::solve_pdhg(problem, opts, *be, diag);
        const auto ev = sor::engines::pdhg_evidence(diag, opts);
        const auto r = sor::certify::finalize_result(std::move(raw), ev);

        std::printf("\nstatus:            %s\n",
                    std::string(sor::core::to_string(r.status)).c_str());
        std::printf("proof_level:       %s\n",
                    std::string(sor::core::to_string(r.proof)).c_str());
        std::printf("                   %s\n",
                    std::string(sor::core::human_line(r.status, r.proof)).c_str());
        if (!r.downgrade_reason.empty())
            std::printf("downgrade:         %s\n", r.downgrade_reason.c_str());

        std::printf("objective:         %.10e\n", r.objective);
        if (diag.dual_bound_finite) {
            std::printf("dual bound:        %.10e\n", r.dual_bound);
            std::printf("rel gap:           %.3e\n", r.gap_rel);
        } else {
            std::printf("dual bound:        none (first-order dual not repaired "
                        "to feasibility)\n");
        }
        std::printf("max row violation: %.3e\n", r.max_primal_violation);
        std::printf("dual residual:     %.3e\n", r.max_dual_violation);
        std::printf("iterations:        %llu\n",
                    static_cast<unsigned long long>(r.iterations));
        std::printf("||A|| estimate:    %.6e\n", diag.matrix_norm_estimate);
        std::printf("termination:       %s\n", r.termination_reason.c_str());

        const auto& s = diag.kernel_stats;
        std::printf("\ntiming (ms)\n");
        std::printf("  total            %10.3f\n", diag.total_ms);
        std::printf("  scaling          %10.3f\n", diag.scaling_ms);
        std::printf("  norm estimate    %10.3f\n", diag.norm_ms);
        std::printf("  pdhg loop        %10.3f\n", diag.loop_ms);
        std::printf("  kernel           %10.3f\n", s.kernel_ms);
        std::printf("  host->device     %10.3f  (%llu bytes)\n", s.h2d_ms,
                    static_cast<unsigned long long>(s.h2d_bytes));
        std::printf("  device->host     %10.3f  (%llu bytes)\n", s.d2h_ms,
                    static_cast<unsigned long long>(s.d2h_bytes));
        std::printf("  ipc overhead     %10.3f\n", s.ipc_ms);
        std::printf("  kernel calls     %10llu\n",
                    static_cast<unsigned long long>(s.calls));

        // Non-zero exit for anything that is not a usable answer.
        switch (r.status) {
            case sor::core::Status::Optimal:
            case sor::core::Status::Feasible:
                return 0;
            case sor::core::Status::Interrupted:
                return 4;
            default:
                return 5;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
