// sor_solve — CLI for LP / MILP / first-order engines.
//
// LAYER L8.
#include "sor/backend/kernel_backend.hpp"
#include "sor/backend/lp_device.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/engines/pdhg.hpp"
#include "sor/engines/qp.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/io/mps.hpp"
#include "sor/io/qps.hpp"
#include "sor/io/solution.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/lattice_reform.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <string>

namespace {

void usage() {
    std::fputs(
        "usage: sor_solve MODEL.mps [options]\n"
        "  --engine NAME    simplex (default) | pdhg | hpr | milp | qp\n"
        "  --q-diag LIST    comma-separated diagonal of Q (if not using .qps)\n"
        "  --backend NAME   cpu (default) | vulkan | julia_gpu\n"
        "  --method NAME    auto | primal | dual   (simplex and MILP node LPs)\n"
        "  --basis-update NAME  product | ft       (simplex basis updates)\n"
        "  --collective-ft  fold pending product updates into L/U at cleanup\n"
        "  --refactor-interval N  maximum basis updates between refactors\n"
        "  --refactor-eta-ratio R refactor when eta nnz exceeds R*factor nnz\n"
        "  --refactor-work-ratio R refactor when solve work exceeds R*factor nnz\n"
        "  --max-iter N     iteration / node limit\n"
        "  --tol T          feasibility tolerance\n"
        "  --time-limit S   wall-clock limit in seconds\n"
        "  --no-scaling     skip Ruiz equilibration\n"
        "  --no-presolve    skip presolve (simplex/milp)\n"
        "  --lattice-reform  opt-in AHL lattice reform for pure integer equalities\n"
        "  --verbose        iteration / node log\n"
        "  --hpr-vanilla | --hpr-full\n"
        "  --solution-out PATH   write a plain-text solution file for sor_check\n",
        stderr);
}

// No-op when `path` is empty (the common case: --solution-out wasn't given).
void write_solution_out(const std::string& path, const sor::core::SolveResult& r) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "warning: could not open '%s' for --solution-out\n",
                     path.c_str());
        return;
    }
    sor::io::write_solution(out, r);
}

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

void print_transfer(const sor::backend::TransferStats& s) {
    std::printf("  kernel           %10.3f\n", s.kernel_ms);
    std::printf("  host->device     %10.3f  (%llu bytes)\n", s.h2d_ms,
                static_cast<unsigned long long>(s.h2d_bytes));
    std::printf("  device->host     %10.3f  (%llu bytes)\n", s.d2h_ms,
                static_cast<unsigned long long>(s.d2h_bytes));
    std::printf("  ipc overhead     %10.3f\n", s.ipc_ms);
    std::printf("  kernel calls     %10llu\n",
                static_cast<unsigned long long>(s.calls));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }

    std::string path, backend_name = "cpu", engine_name = "simplex";
    std::string q_diag_arg;
    std::string solution_out;
    sor::engines::PdhgOptions pdhg_opts;
    sor::engines::HprOptions hpr_opts;
    sor::engines::SimplexOptions sx_opts;
    sor::io::MpsReadOptions mps_opts;
    bool mps_format_forced = false;
    bool tol_given = false;
    bool max_iter_given = false;
    bool lattice_reform = false;
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
        else if (a == "--q-diag")   q_diag_arg   = next("--q-diag");
        else if (a == "--max-iter") {
            const auto n = std::strtoull(next("--max-iter").c_str(), nullptr, 10);
            pdhg_opts.max_iterations = n;
            hpr_opts.max_iterations  = n;
            sx_opts.max_iterations   = n;
            max_iter_given = true;
        }
        else if (a == "--tol") {
            tol = std::strtod(next("--tol").c_str(), nullptr);
            tol_given = true;
        }
        else if (a == "--time-limit") {
            const double t = std::strtod(next("--time-limit").c_str(), nullptr);
            sx_opts.time_limit_s = t;
            hpr_opts.time_limit_s = t;
            pdhg_opts.time_limit_s = t;
            // PDHG/HPR's max_iterations is a fixed default (100000/200000),
            // not the "0 = auto-scale to problem size" convention the
            // simplex engines use -- so with a time budget also given, an
            // UNCHANGED default iteration cap was usually the thing that
            // actually stopped the loop (100000 PDHG iterations run in well
            // under a second on most Netlib instances), silently discarding
            // most of the requested time budget. When a time limit is
            // explicitly requested, let it be the real constraint.
            if (!max_iter_given) {
                pdhg_opts.max_iterations = std::numeric_limits<std::uint64_t>::max();
                hpr_opts.max_iterations = std::numeric_limits<std::uint64_t>::max();
            }
        }
        else if (a == "--method") {
            const std::string m = next("--method");
            if      (m == "auto")   sx_opts.method = sor::engines::SimplexMethod::Auto;
            else if (m == "primal") sx_opts.method = sor::engines::SimplexMethod::Primal;
            else if (m == "dual")   sx_opts.method = sor::engines::SimplexMethod::Dual;
            else { std::fprintf(stderr, "error: unknown method '%s'\n", m.c_str()); return 2; }
        }
        else if (a == "--basis-update") {
            const std::string method = next("--basis-update");
            if (method == "product")
                sx_opts.update_method = sor::la::UpdateMethod::ProductForm;
            else if (method == "ft")
                sx_opts.update_method = sor::la::UpdateMethod::ForrestTomlin;
            else {
                std::fprintf(stderr, "error: unknown basis update '%s'\n", method.c_str());
                return 2;
            }
        }
        else if (a == "--collective-ft") sx_opts.collective_ft = true;
        else if (a == "--refactor-interval")
            sx_opts.refactor_interval =
                static_cast<int>(std::strtol(next("--refactor-interval").c_str(), nullptr, 10));
        else if (a == "--refactor-eta-ratio")
            sx_opts.refactor_eta_ratio =
                std::strtod(next("--refactor-eta-ratio").c_str(), nullptr);
        else if (a == "--refactor-work-ratio")
            sx_opts.refactor_work_ratio =
                std::strtod(next("--refactor-work-ratio").c_str(), nullptr);
        else if (a == "--no-scaling") {
            sx_opts.ruiz_iterations = 0;
            pdhg_opts.ruiz_iterations = 0;
            hpr_opts.ruiz_iterations = 0;
        }
        else if (a == "--no-presolve") sx_opts.presolve = false;
        else if (a == "--lattice-reform") lattice_reform = true;
        else if (a == "--fixed-mps") { mps_opts.fixed_format = true; mps_format_forced = true; }
        else if (a == "--free-mps")  { mps_opts.fixed_format = false; mps_format_forced = true; }
        else if (a == "--verbose") {
            pdhg_opts.verbose = true;
            hpr_opts.verbose = true;
            sx_opts.verbose = true;
        }
        else if (a == "--hpr-vanilla") {
            hpr_opts.use_primal_weight = false;
            hpr_opts.use_restart = false;
            hpr_opts.use_halpern = false;
        }
        else if (a == "--hpr-full") {
            hpr_opts.use_primal_weight = true;
            hpr_opts.use_restart = true;
            hpr_opts.use_halpern = true;
        }
        else if (a == "--solution-out") solution_out = next("--solution-out");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            return 2;
        }
        else path = a;
    }

    if (path.empty()) { usage(); return 2; }
    if (engine_name != "pdhg" && engine_name != "simplex" && engine_name != "hpr" &&
        engine_name != "milp" && engine_name != "qp") {
        std::fprintf(stderr,
                     "error: engine '%s' not implemented "
                     "(have simplex|pdhg|hpr|milp|qp)\n",
                     engine_name.c_str());
        return 3;
    }
    if (tol_given) {
        pdhg_opts.primal_tol = pdhg_opts.dual_tol = tol;
        hpr_opts.primal_tol = hpr_opts.dual_tol = hpr_opts.gap_tol = tol;
        sx_opts.primal_feas_tol = sx_opts.dual_feas_tol = tol;
    }

    try {
        const bool path_is_qps = path.size() >= 4 &&
            (path.compare(path.size() - 4, 4, ".qps") == 0 ||
             path.compare(path.size() - 4, 4, ".QPS") == 0);

        if (engine_name == "qp") {
            sor::engines::QpProblem qp;
            if (!q_diag_arg.empty()) {
                sor::io::MpsReadReport rep;
                qp.linear = mps_format_forced
                    ? sor::io::read_mps_file(path, rep, mps_opts)
                    : sor::io::read_mps_file_auto(path, rep, mps_opts.strict);
                for (const auto& w : rep.warnings)
                    std::fprintf(stderr, "warning: %s\n", w.c_str());
                qp.q_diag.clear();
                std::size_t start = 0;
                while (start <= q_diag_arg.size()) {
                    const auto comma = q_diag_arg.find(',', start);
                    const auto tok = q_diag_arg.substr(
                        start, comma == std::string::npos ? std::string::npos
                                                          : comma - start);
                    if (!tok.empty())
                        qp.q_diag.push_back(std::strtod(tok.c_str(), nullptr));
                    if (comma == std::string::npos) break;
                    start = comma + 1;
                }
            } else if (path_is_qps) {
                sor::io::QpsReadReport rep;
                auto loaded = sor::io::read_qps_file(path, rep, mps_opts);
                for (const auto& w : rep.warnings)
                    std::fprintf(stderr, "warning: %s\n", w.c_str());
                qp.linear = std::move(loaded.linear);
                qp.q_diag = std::move(loaded.q_diag);
                qp.q_matrix = std::move(loaded.q_matrix);
            } else {
                std::fprintf(stderr,
                             "error: --engine qp needs a .qps file or --q-diag\n");
                return 2;
            }

            std::printf("model:             %s\n",
                        qp.linear.name.empty() ? path.c_str()
                                               : qp.linear.name.c_str());
            std::printf("rows x cols:       %d x %d   nnz %lld\n",
                        qp.linear.n_rows(), qp.linear.n_cols(),
                        static_cast<long long>(qp.linear.nnz()));
            std::printf("engine:            qp\n");

            sor::engines::QpOptions qopts;
            qopts.max_iterations = pdhg_opts.max_iterations;
            qopts.time_limit_s = sx_opts.time_limit_s;
            qopts.verbose = sx_opts.verbose;
            if (tol_given) {
                qopts.feas_tol = tol;
                qopts.stationarity_tol = tol;
                qopts.gap_tol = tol;
            }
            sor::engines::QpDiagnostics diag;
            auto raw = sor::engines::solve_qp(qp, qopts, diag);
            const auto ev = sor::engines::qp_evidence(diag, qopts);
            const auto r = sor::certify::finalize_result(std::move(raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            std::printf("stationarity:      %.3e\n", diag.stationarity);
            std::printf("max primal viol:   %.3e\n", diag.primal_residual);
            std::printf("relative gap:      %.3e\n", diag.gap_rel);
            std::printf("iterations:        %llu\n",
                        static_cast<unsigned long long>(diag.iterations));
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", diag.total_ms);
            return exit_code_for(r.status);
        }

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
        if (rep.n_integer > 0)
            std::printf("integer columns:   %zu\n", rep.n_integer);
        std::printf("engine:            %s\n", engine_name.c_str());

        if (engine_name == "milp") {
            if (rep.n_integer == 0) {
                std::fprintf(stderr,
                             "warning: no integer columns; milp reduces to LP\n");
            }
            sor::search::BabOptions bab;
            bab.lp = sx_opts;
            bab.verbose = sx_opts.verbose;
            // --max-iter is the public MILP node limit.  A node LP must not
            // inherit that value as its own pivot cap: doing so interrupted
            // the schedule root relaxation at 5000 pivots and left B&B with
            // no incumbent.  The B&B driver applies the global wall-clock
            // budget to each relaxation instead.
            bab.lp.max_iterations = 0;
            if (sx_opts.max_iterations != 0)
                bab.max_nodes = sx_opts.max_iterations;
            if (sx_opts.time_limit_s > 0.0)
                bab.time_limit_s = sx_opts.time_limit_s;
            if (tol_given) {
                bab.primal_feas_tol = tol;
                bab.int_tol = std::max(tol, 1e-9);
            }
            auto out =
                sor::search::solve_milp_lattice(problem, bab, lattice_reform);
            if (lattice_reform) {
                if (out.reform_applied) {
                    std::printf("lattice reform:    applied (%s), kernel dim %d, "
                                "forced-zero cont %zu, fixed %zu, %s\n",
                                out.note.c_str(), out.kernel_dim,
                                out.forced_zero_cols, out.fixed_cols,
                                out.exact_equivalence ? "exact equivalence"
                                                      : "restriction");
                    if (out.certified) {
                        std::printf("lattice reform:    certified optimal against "
                                    "original LP relaxation bound\n");
                    } else if (out.fell_back) {
                        std::printf("lattice reform:    restricted run not "
                                    "certifiable; original re-solved\n");
                    }
                } else {
                    std::printf("lattice reform:    skipped (%s)\n",
                                out.note.c_str());
                }
            }
            const auto& diag = out.diag;
            const auto ev = sor::search::milp_evidence(diag, bab);
            const auto r = sor::certify::finalize_result(std::move(out.raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            if (std::isfinite(diag.dual_bound)) {
                std::printf("dual bound:        %.10e\n", diag.dual_bound);
                std::printf("mip gap:           %.3e\n", diag.gap_rel);
            }
            std::printf("nodes:             %llu\n",
                        static_cast<unsigned long long>(diag.nodes));
            std::printf("lp solves:         %llu\n",
                        static_cast<unsigned long long>(diag.lp_solves));
            std::printf("lp fallbacks:      %llu\n",
                        static_cast<unsigned long long>(diag.lp_fallbacks));
            std::printf("integer row roundings: %llu\n",
                        static_cast<unsigned long long>(diag.integer_row_roundings));
            std::printf("binary cover cuts: %llu\n",
                        static_cast<unsigned long long>(diag.binary_cover_cuts));
            std::printf("GMI cuts:          %llu in %d rounds\n",
                        static_cast<unsigned long long>(diag.gmi_cuts_added),
                        diag.cut_rounds);
            std::printf("cut pool:          %llu inserted, %llu duplicate, "
                        "%llu dominated, %llu parallel-rejected, %llu aged, "
                        "%llu evicted\n",
                        static_cast<unsigned long long>(diag.cut_pool_inserted),
                        static_cast<unsigned long long>(diag.cut_pool_duplicates),
                        static_cast<unsigned long long>(diag.cut_pool_dominated),
                        static_cast<unsigned long long>(diag.cut_pool_parallel_rejections),
                        static_cast<unsigned long long>(diag.cut_pool_aged_out),
                        static_cast<unsigned long long>(diag.cut_pool_evicted));
            std::printf("strong branch LPs: %llu  (pseudocost updates %llu)\n",
                        static_cast<unsigned long long>(diag.strong_branch_solves),
                        static_cast<unsigned long long>(diag.pseudocost_updates));
            std::printf("integer feas:      %llu  (heuristic hits %llu)\n",
                        static_cast<unsigned long long>(diag.integer_feasible),
                        static_cast<unsigned long long>(diag.heuristic_hits));
            std::printf("LP repair:         %llu attempts, %llu hits\n",
                        static_cast<unsigned long long>(diag.lp_repair_attempts),
                        static_cast<unsigned long long>(diag.lp_repair_hits));
            std::printf("feasibility pump:  %llu attempts, %llu hits\n",
                        static_cast<unsigned long long>(diag.feasibility_pump_attempts),
                        static_cast<unsigned long long>(diag.feasibility_pump_hits));
            std::printf("integer dive:      %llu attempts, %llu LPs, %llu hits\n",
                        static_cast<unsigned long long>(diag.integer_dive_attempts),
                        static_cast<unsigned long long>(diag.integer_dive_lp_solves),
                        static_cast<unsigned long long>(diag.integer_dive_hits));
            std::printf("RENS:              %llu attempts, %llu LPs, %llu hits\n",
                        static_cast<unsigned long long>(diag.rens_attempts),
                        static_cast<unsigned long long>(diag.rens_lp_solves),
                        static_cast<unsigned long long>(diag.rens_hits));
            std::printf("integer neighborhood: %llu attempts, %llu trials, %llu hits\n",
                        static_cast<unsigned long long>(diag.integer_neighborhood_attempts),
                        static_cast<unsigned long long>(diag.integer_neighborhood_trials),
                        static_cast<unsigned long long>(diag.integer_neighborhood_hits));
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", diag.total_ms);
            return exit_code_for(r.status);
        }

        if (engine_name == "simplex") {
            if (rep.n_integer > 0) {
                std::printf("NOTE:              solving the LP RELAXATION "
                            "(use --engine milp for branch-and-bound)\n");
            }
            sor::engines::SimplexDiagnostics diag;
            auto raw = sor::engines::solve_simplex(problem, sx_opts, diag, nullptr);
            const auto ev = sor::engines::simplex_evidence(diag, sx_opts);
            const auto r = sor::certify::finalize_result(std::move(raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            if (diag.dual_bound_finite) {
                std::printf("dual bound:        %.10e\n", r.dual_bound);
                std::printf("rel gap:           %.3e\n", r.gap_rel);
            }
            std::printf("max primal viol:   %.3e\n", r.max_primal_violation);
            std::printf("dual residual:     %.3e\n", r.max_dual_violation);
            std::printf("iterations:        %llu  (phase1 %llu, phase2 %llu)\n",
                        static_cast<unsigned long long>(diag.iterations),
                        static_cast<unsigned long long>(diag.phase1_iterations),
                        static_cast<unsigned long long>(diag.phase2_iterations));
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", diag.total_ms);
            std::printf("  scaling          %10.3f\n", diag.scaling_ms);
            std::printf("  CSR->CSC         %10.3f\n", diag.csc_ms);
            std::printf("  preparation      %10.3f\n", diag.preprocessing_ms);
            std::printf("  presolve         %10.3f\n", diag.presolve_ms);
            std::printf("  factorization    %10.3f\n", diag.factor_ms);
            std::printf("  pricing          %10.3f\n", diag.price_ms);
            std::printf("  triangular solve %10.3f\n", diag.solve_ms);
            std::printf("    FTRAN          %10.3f  (%llu calls)\n", diag.ftran_ms,
                        static_cast<unsigned long long>(diag.ftran_calls));
            std::printf("    BTRAN          %10.3f  (%llu calls)\n", diag.btran_ms,
                        static_cast<unsigned long long>(diag.btran_calls));
            std::printf("    pivotal rows   %10.3f\n", diag.pivotal_row_ms);
            std::printf("    ratio tests    %10.3f\n", diag.ratio_test_ms);
            std::printf("  basis updates    %10.3f  (%llu calls)\n", diag.basis_update_ms,
                        static_cast<unsigned long long>(diag.basis_update_calls));
            std::printf("  refactors/cFT    %10llu / %llu  (%llu guarded skips)\n",
                        static_cast<unsigned long long>(diag.refactorizations),
                        static_cast<unsigned long long>(diag.collective_ft_collapses),
                        static_cast<unsigned long long>(diag.collective_ft_skips));
            std::printf("  simplex loop     %10.3f\n", diag.loop_ms);
            std::printf("  auto stages/builds %8llu / %llu\n",
                        static_cast<unsigned long long>(diag.stages),
                        static_cast<unsigned long long>(diag.preprocessing_builds));
            return exit_code_for(r.status);
        }

        if (engine_name == "hpr") {
            auto dev = sor::backend::make_lp_device(backend_name);
            if (!dev) {
                std::fprintf(stderr, "warning: LpDevice '%s' unavailable; using cpu\n",
                             backend_name.c_str());
                dev = sor::backend::make_cpu_lp_device();
            }
            std::printf("backend:           %s (accelerated=%s)\n",
                        std::string(dev->name()).c_str(),
                        dev->is_accelerated() ? "yes" : "no");
            sor::engines::HprDiagnostics diag;
            auto raw = sor::engines::solve_hpr(problem, hpr_opts, *dev, diag);
            const auto ev = sor::engines::hpr_evidence(diag, hpr_opts);
            const auto r = sor::certify::finalize_result(std::move(raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            std::printf("max row violation: %.3e\n", r.max_primal_violation);
            std::printf("dual residual:     %.3e\n", r.max_dual_violation);
            std::printf("iterations:        %llu\n",
                        static_cast<unsigned long long>(r.iterations));
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", diag.total_ms);
            print_transfer(diag.device_stats);
            return exit_code_for(r.status);
        }

        auto be = sor::backend::make_backend(backend_name);
        if (!be) {
            std::fprintf(stderr, "warning: backend '%s' unavailable; using cpu\n",
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
        write_solution_out(solution_out, r);
        std::printf("max row violation: %.3e\n", r.max_primal_violation);
        std::printf("dual residual:     %.3e\n", r.max_dual_violation);
        std::printf("iterations:        %llu\n",
                    static_cast<unsigned long long>(r.iterations));
        std::printf("termination:       %s\n", r.termination_reason.c_str());
        std::printf("\ntiming (ms)\n");
        std::printf("  total            %10.3f\n", diag.total_ms);
        print_transfer(diag.kernel_stats);
        return exit_code_for(r.status);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
