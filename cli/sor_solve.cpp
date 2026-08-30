// sor_solve — prototype CLI.
//
// LAYER L8.
//
//   sor_solve model.mps [--engine simplex|pdhg] [--backend cpu|julia_gpu]
//                       [--max-iter N] [--tol T] [--time-limit S] [--verbose]
//
// Prints status, proof level, objective, residuals, and a timing breakdown that
// always includes host<->device transfer and IPC cost.
#include "sor/backend/kernel_backend.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/engines/pdhg.hpp"
#include "sor/engines/simplex.hpp"
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
        "  --engine NAME    simplex (default) | pdhg\n"
        "                   simplex: primal revised simplex, can prove Optimal\n"
        "                   pdhg:    first-order, cannot prove Optimal by design\n"
        "  --backend NAME   cpu (default) | julia_gpu (EXPERIMENTAL); pdhg only\n"
        "  --max-iter N     iteration limit (0 = automatic)\n"
        "  --tol T          primal/dual tolerance (default 1e-6 pdhg, 1e-7 simplex)\n"
        "  --time-limit S   wall-clock limit in seconds; returns the best point\n"
        "                   found so far (simplex only)\n"
        "  --no-scaling     skip Ruiz equilibration\n"
        "  --fixed-mps      force fixed-column MPS parsing\n"
        "  --free-mps       force free-format MPS parsing\n"
        "                   (default: try free, fall back to fixed)\n"
        "  --verbose        print iteration log and per-phase timings\n",
        stderr);
}

// Everything after the status block is common to both engines, so the two
// branches below differ only in how they produce a SolveResult.
void print_result(const sor::core::SolveResult& r) {
    std::printf("\nstatus:            %s\n",
                std::string(sor::core::to_string(r.status)).c_str());
    std::printf("proof_level:       %s\n",
                std::string(sor::core::to_string(r.proof)).c_str());
    std::printf("                   %s\n",
                std::string(sor::core::human_line(r.status, r.proof)).c_str());
    if (!r.downgrade_reason.empty())
        std::printf("downgrade:         %s\n", r.downgrade_reason.c_str());
    std::printf("objective:         %.10e\n", r.objective);
}

int exit_code_for(sor::core::Status s) {
    switch (s) {
        case sor::core::Status::Optimal:
        case sor::core::Status::Feasible:
            return 0;
        case sor::core::Status::Interrupted:
            return 4;
        default:
            return 5;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }

    std::string path, backend_name = "cpu", engine_name = "simplex";
    sor::engines::PdhgOptions pdhg_opts;
    sor::engines::SimplexOptions sx_opts;
    sor::io::MpsReadOptions mps_opts;
    bool mps_format_forced = false;
    bool tol_given = false;
    double tol = 0.0;

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
        else if (a == "--max-iter") {
            const auto n = std::strtoull(next("--max-iter").c_str(), nullptr, 10);
            pdhg_opts.max_iterations = n;
            sx_opts.max_iterations   = n;
        }
        else if (a == "--tol") { tol = std::strtod(next("--tol").c_str(), nullptr);
                                 tol_given = true; }
        else if (a == "--time-limit")
            sx_opts.time_limit_s = std::strtod(next("--time-limit").c_str(), nullptr);
        else if (a == "--no-scaling") { sx_opts.ruiz_iterations = 0;
                                        pdhg_opts.ruiz_iterations = 0; }
        else if (a == "--fixed-mps") { mps_opts.fixed_format = true;
                                       mps_format_forced = true; }
        else if (a == "--free-mps")  { mps_opts.fixed_format = false;
                                       mps_format_forced = true; }
        else if (a == "--verbose")   { pdhg_opts.verbose = true;
                                       sx_opts.verbose = true; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            return 2;
        }
        else path = a;
    }

    if (path.empty()) { usage(); return 2; }
    if (engine_name != "pdhg" && engine_name != "simplex") {
        std::fprintf(stderr,
                     "error: engine '%s' is not implemented (have 'simplex' and "
                     "'pdhg'). Refusing rather than substituting.\n",
                     engine_name.c_str());
        return 3;
    }
    if (tol_given) {
        pdhg_opts.primal_tol = pdhg_opts.dual_tol = tol;
        sx_opts.primal_feas_tol = sx_opts.dual_feas_tol = tol;
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
        std::printf("engine:            %s\n", engine_name.c_str());

        if (engine_name == "simplex") {
            sor::engines::SimplexDiagnostics diag;
            auto raw = sor::engines::solve_simplex(problem, sx_opts, diag, nullptr);
            const auto ev = sor::engines::simplex_evidence(diag, sx_opts);
            const auto r = sor::certify::finalize_result(std::move(raw), ev);

            print_result(r);
            if (diag.dual_bound_finite) {
                std::printf("dual bound:        %.10e\n", r.dual_bound);
                std::printf("rel gap:           %.3e\n", r.gap_rel);
            } else {
                std::printf("dual bound:        none (no finite Lagrangian value "
                            "at this basis)\n");
            }
            std::printf("max primal viol:   %.3e   (rows and bounds, recomputed "
                        "on the unscaled model)\n", r.max_primal_violation);
            std::printf("dual residual:     %.3e\n", r.max_dual_violation);
            std::printf("iterations:        %llu  (phase 1: %llu, phase 2: %llu)\n",
                        static_cast<unsigned long long>(diag.iterations),
                        static_cast<unsigned long long>(diag.phase1_iterations),
                        static_cast<unsigned long long>(diag.phase2_iterations));
            std::printf("bound flips:       %llu\n",
                        static_cast<unsigned long long>(diag.bound_flips));
            std::printf("refactorizations:  %llu\n",
                        static_cast<unsigned long long>(diag.refactorizations));
            std::printf("degenerate steps:  %llu  (bland iterations %llu)\n",
                        static_cast<unsigned long long>(diag.degenerate_steps),
                        static_cast<unsigned long long>(diag.bland_iterations));
            if (diag.basis_repairs > 0)
                std::printf("basis repairs:     %llu  (singular basis replaced by "
                            "logicals)\n",
                            static_cast<unsigned long long>(diag.basis_repairs));
            std::printf("LU nnz:            %lld   max multiplier %.3e\n",
                        static_cast<long long>(diag.factor_nnz),
                        diag.largest_multiplier);
            std::printf("termination:       %s\n", r.termination_reason.c_str());

            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", diag.total_ms);
            std::printf("  scaling          %10.3f\n", diag.scaling_ms);
            std::printf("  factorization    %10.3f\n", diag.factor_ms);
            std::printf("  simplex loop     %10.3f\n", diag.loop_ms);
            if (sx_opts.verbose) {
                std::printf("  pricing          %10.3f\n", diag.price_ms);
                std::printf("  ftran + btran    %10.3f\n", diag.solve_ms);
            } else {
                std::printf("  pricing          (--verbose; clock reads cost "
                            "~1.3us each on this host)\n");
            }
            return exit_code_for(r.status);
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
        auto raw = sor::engines::solve_pdhg(problem, pdhg_opts, *be, diag);
        const auto ev = sor::engines::pdhg_evidence(diag, pdhg_opts);
        const auto r = sor::certify::finalize_result(std::move(raw), ev);

        print_result(r);
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

        return exit_code_for(r.status);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
