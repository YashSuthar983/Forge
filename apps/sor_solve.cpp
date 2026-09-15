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
        "  --pricing NAME   choose | dantzig | devex | dse  (simplex pricing)\n"
        "  --basis-update NAME  product | ft       (simplex basis updates)\n"
        "  --collective-ft  fold pending product updates into L/U at cleanup\n"
        "  --refactor-interval N  maximum basis updates between refactors\n"
        "  --refactor-eta-ratio R refactor when eta nnz exceeds R*factor nnz\n"
        "  --refactor-work-ratio R refactor when solve work exceeds R*factor nnz\n"
        "  --refactor-u-nnz-ratio R  FT: refactor when nnz(U) exceeds R*factor nnz\n"
        "  --ft-update-limit N       FT: refactor after N row etas\n"
        "  --dual-resync-interval N rebuild dual/reduced costs every N pivots (0=off)\n"
        "  --dual-cost-perturbation M deterministic dual perturbation multiplier\n"
        "  --[no-]primal-crash  feasibility-reducing crash (simplex CLI default: on)\n"
        "  --max-iter N     iteration / node limit\n"
        "  --tol T          feasibility tolerance\n"
        "  --time-limit S   wall-clock limit in seconds\n"
        "  --no-scaling     skip Ruiz equilibration\n"
        "  --pow2-scaling   round Ruiz factors to powers of two (exact scaling)\n"
        "  --no-presolve    skip presolve (simplex/milp)\n"
        "  --implied-slack  presolve: drop zero-cost singleton columns as slacks\n"
        "  --lattice-reform  opt-in AHL lattice reform for pure integer equalities\n"
        "  --no-probing     skip MILP root probing (conflict graph, implied bounds)\n"
        "  --no-feasjump    skip the Feasibility Jump primal heuristic\n"
        "  --no-sub-mip     skip the bandit-scheduled LNS portfolio\n"
        "  --heuristic-budget F  share of the time limit the heuristic layer may use\n"
        "  --clique-cuts    opt-in clique cut separation in the root cut loop\n"
        "  --no-vub-cuts    skip implied-bound (variable-bound) cut separation\n"
        "  --cover-cuts     opt-in lifted knapsack cover cut separation\n"
        "  --mir-cuts       opt-in mixed-integer rounding cut separation\n"
        "  --no-aggregation skip MIR row aggregation (single-row bases only)\n"
        "  --cut-nnz-budget F   nonzeros added per cut round, as a multiple of n (0=off)\n"
        "  --cut-max-density F  reject cuts denser than this fraction of n (>1=off)\n"
        "  --cut-parallel-penalty F  score penalty for parallel cuts (0=off)\n"
        "  --cut-extra-scores F  weight of the sparsity and low-lock score terms\n"
        "  --no-conflict-prop  skip conflict-graph propagation at nodes\n"
        "  --verbose        iteration / node log\n"
        "  --hpr-vanilla | --hpr-full\n"
        "  --solution-out PATH   write a plain-text solution file for sor_check\n",
        stderr);
}

// ---------------------------------------------------------------------------
// Numeric option parsing.
//
// Every numeric flag used to go straight through strtod/strtoull with a null
// end pointer, which accepts far too much in silence: "abc" parses as 0,
// "-5" as a huge unsigned, and "nan"/"inf" as themselves. A mistyped flag
// therefore produced a DIFFERENT RUN instead of an error -- `--tol abc` ran at
// tolerance exactly 0, `--max-iter -5` ran with an effectively infinite cap,
// and both exited 0 as though nothing was wrong. These parse the WHOLE token,
// reject non-finite values, and enforce each option's own admissible range.
[[noreturn]] void bad_value(const char* what, const std::string& got,
                            const std::string& expected) {
    std::fprintf(stderr, "error: %s expects %s, got '%s'\n", what,
                 expected.c_str(), got.c_str());
    std::exit(2);
}

// std::to_string(0.0) is "0.000000", which reads badly in an error message.
std::string compact(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

std::string range_text(double lo, double hi, bool lo_open) {
    std::string s = "a finite number ";
    s += lo_open ? "greater than " : "at least ";
    s += compact(lo);
    if (hi < std::numeric_limits<double>::max()) s += " and at most " + compact(hi);
    return s;
}

double parse_real(const std::string& text, const char* what, double lo,
                  double hi = std::numeric_limits<double>::max(),
                  bool lo_open = false) {
    std::size_t used = 0;
    double v = 0.0;
    try {
        v = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != text.size() || text.empty() || !std::isfinite(v))
        bad_value(what, text, range_text(lo, hi, lo_open));
    if (lo_open ? !(v > lo) : !(v >= lo)) bad_value(what, text, range_text(lo, hi, lo_open));
    if (!(v <= hi)) bad_value(what, text, range_text(lo, hi, lo_open));
    return v;
}

// A leading '-' is rejected before stoull sees it: stoull WRAPS a negative
// value into a huge positive one rather than failing.
unsigned long long parse_uint(const std::string& text, const char* what,
                              unsigned long long lo, unsigned long long hi) {
    const std::string expected = "an integer from " + std::to_string(lo) +
                                 " to " + std::to_string(hi);
    if (text.empty() || text.find('-') != std::string::npos)
        bad_value(what, text, expected);
    std::size_t used = 0;
    unsigned long long v = 0;
    try {
        v = std::stoull(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != text.size() || v < lo || v > hi) bad_value(what, text, expected);
    return v;
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
    bool primal_crash_given = false;
    bool lattice_reform = false;
    // MILP conflict-graph switches. Three independent consumers of one root
    // probing pass, kept separately switchable so a benchmark run can attribute
    // a change to the reduction, the cuts, or the node propagation.
    bool probing = true;
    bool feasibility_jump = true;
    bool sub_mip_lns = true;
    double heuristic_budget = -1.0;   // <0 keeps the library default
    bool clique_cuts = false;   // opt-in: see BabOptions::clique_cuts
    bool implied_bound_cuts = true;
    bool lifted_cover_cuts = false;   // opt-in: see BabOptions
    bool mir_cuts = false;
    bool mir_aggregate = true;
    double cut_nnz_budget = -1.0;    // <0 keeps the library default
    double cut_max_density = -1.0;
    double cut_par_penalty = -1.0;
    double cut_extra_scores = -1.0;
    bool conflict_propagation = true;
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
            const auto n = parse_uint(next("--max-iter"), "--max-iter", 1,
                                      std::numeric_limits<std::uint64_t>::max());
            pdhg_opts.max_iterations = n;
            hpr_opts.max_iterations  = n;
            sx_opts.max_iterations   = n;
            max_iter_given = true;
        }
        else if (a == "--tol") {
            tol = parse_real(next("--tol"), "--tol", 0.0,
                             std::numeric_limits<double>::max(), true);
            tol_given = true;
        }
        else if (a == "--time-limit") {
            const double t = parse_real(next("--time-limit"), "--time-limit", 0.0,
                                        std::numeric_limits<double>::max(), true);
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
        else if (a == "--pricing") {
            const std::string pricing = next("--pricing");
            if (pricing == "choose")
                sx_opts.pricing = sor::engines::SimplexPricing::Choose;
            else if (pricing == "dantzig")
                sx_opts.pricing = sor::engines::SimplexPricing::Dantzig;
            else if (pricing == "devex")
                sx_opts.pricing = sor::engines::SimplexPricing::Devex;
            else if (pricing == "dse")
                sx_opts.pricing = sor::engines::SimplexPricing::DSE;
            else {
                std::fprintf(stderr, "error: unknown simplex pricing '%s'\n",
                             pricing.c_str());
                return 2;
            }
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
            sx_opts.refactor_interval = static_cast<int>(parse_uint(
                next("--refactor-interval"), "--refactor-interval", 0,
                static_cast<unsigned long long>(std::numeric_limits<int>::max())));
        else if (a == "--refactor-eta-ratio")
            sx_opts.refactor_eta_ratio =
                parse_real(next("--refactor-eta-ratio"), "--refactor-eta-ratio", 0.0);
        else if (a == "--refactor-work-ratio")
            sx_opts.refactor_work_ratio =
                parse_real(next("--refactor-work-ratio"), "--refactor-work-ratio", 0.0);
        else if (a == "--refactor-u-nnz-ratio")
            sx_opts.refactor_u_nnz_ratio =
                parse_real(next("--refactor-u-nnz-ratio"),
                           "--refactor-u-nnz-ratio", 0.0);
        else if (a == "--ft-update-limit")
            sx_opts.ft_update_limit = static_cast<int>(parse_uint(
                next("--ft-update-limit"), "--ft-update-limit", 0,
                static_cast<unsigned long long>(std::numeric_limits<int>::max())));
        else if (a == "--dual-resync-interval")
            sx_opts.dual_resync_interval = static_cast<int>(parse_uint(
                next("--dual-resync-interval"), "--dual-resync-interval", 0,
                static_cast<unsigned long long>(std::numeric_limits<int>::max())));
        else if (a == "--dual-cost-perturbation")
            sx_opts.dual_cost_perturbation_multiplier = parse_real(
                next("--dual-cost-perturbation"), "--dual-cost-perturbation", 0.0);
        else if (a == "--primal-crash") {
            sx_opts.primal_crash = true;
            primal_crash_given = true;
        }
        else if (a == "--no-primal-crash") {
            sx_opts.primal_crash = false;
            primal_crash_given = true;
        }
        else if (a == "--pow2-scaling") sx_opts.ruiz_power_of_two = true;
        else if (a == "--no-scaling") {
            sx_opts.ruiz_iterations = 0;
            pdhg_opts.ruiz_iterations = 0;
            hpr_opts.ruiz_iterations = 0;
        }
        else if (a == "--no-presolve") sx_opts.presolve = false;
        else if (a == "--implied-slack") sx_opts.presolve_implied_slack = true;
        else if (a == "--lattice-reform") lattice_reform = true;
        else if (a == "--no-probing") probing = false;
        else if (a == "--no-feasjump") feasibility_jump = false;
        else if (a == "--no-sub-mip") sub_mip_lns = false;
        else if (a == "--heuristic-budget")
            heuristic_budget = parse_real(next("--heuristic-budget"),
                                         "--heuristic-budget", 0.0, 1.0);
        else if (a == "--clique-cuts") clique_cuts = true;
        else if (a == "--no-vub-cuts") implied_bound_cuts = false;
        else if (a == "--cover-cuts") lifted_cover_cuts = true;
        else if (a == "--mir-cuts") mir_cuts = true;
        else if (a == "--no-aggregation") mir_aggregate = false;
        else if (a == "--cut-nnz-budget")
            cut_nnz_budget = parse_real(next("--cut-nnz-budget"), "--cut-nnz-budget", 0.0);
        else if (a == "--cut-max-density")
            cut_max_density = parse_real(next("--cut-max-density"), "--cut-max-density", 0.0);
        else if (a == "--cut-extra-scores")
            cut_extra_scores = parse_real(next("--cut-extra-scores"), "--cut-extra-scores", 0.0);
        else if (a == "--cut-parallel-penalty")
            cut_par_penalty = parse_real(next("--cut-parallel-penalty"), "--cut-parallel-penalty", 0.0);
        else if (a == "--no-conflict-prop") conflict_propagation = false;
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
    // A backend that is merely UNAVAILABLE warns and falls back to cpu further
    // down -- that is deliberate, since vulkan/julia_gpu depend on the machine.
    // A backend NAME that does not exist is a different thing, and it used to
    // be accepted in total silence: the simplex path returns before any backend
    // is constructed, so `--backend abacus` simply did nothing.
    if (backend_name != "cpu" && backend_name != "vulkan" &&
        backend_name != "julia_gpu") {
        std::fprintf(stderr,
                     "error: unknown backend '%s' (have cpu|vulkan|julia_gpu)\n",
                     backend_name.c_str());
        return 2;
    }
    // The crash has an independent 93/93 Netlib proof gate and improves the
    // LP CLI's aggregate pivot count. Keep SimplexOptions' library default off:
    // MILP owns its node-LP policy separately and must not change merely
    // because the standalone LP default did.
    if (engine_name == "simplex" && !primal_crash_given)
        sx_opts.primal_crash = true;
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
                        qp.q_diag.push_back(parse_real(
                            tok, "--q-diag",
                            -std::numeric_limits<double>::max()));
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
            bab.probing = probing;
            bab.feasibility_jump = feasibility_jump;
            bab.sub_mip_lns = sub_mip_lns;
            if (heuristic_budget >= 0.0) {
                bab.heuristic_budget_frac = heuristic_budget;
                bab.heuristic_budget_frac_no_incumbent =
                    std::max(heuristic_budget, 0.80);
            }
            bab.clique_cuts = clique_cuts;
            bab.implied_bound_cuts = implied_bound_cuts;
            bab.lifted_cover_cuts = lifted_cover_cuts;
            bab.mir_cuts = mir_cuts;
            bab.mir.aggregate = mir_aggregate;
            if (cut_nnz_budget >= 0.0) bab.cut.pool_nnz_budget_factor = cut_nnz_budget;
            if (cut_max_density >= 0.0) bab.cut.pool_max_density = cut_max_density;
            if (cut_par_penalty >= 0.0) bab.cut.pool_parallelism_penalty = cut_par_penalty;
            if (cut_extra_scores >= 0.0) {
                bab.cut.pool_weight_sparsity = cut_extra_scores;
                bab.cut.pool_weight_low_locks = cut_extra_scores;
            }
            bab.conflict_propagation = conflict_propagation;
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
            std::printf("node LP:            %llu iterations in %.1f ms "
                        "(%.1f iters/node, %.3f ms/node)\n",
                        static_cast<unsigned long long>(diag.lp_iterations),
                        diag.lp_ms,
                        diag.nodes ? double(diag.lp_iterations) / double(diag.nodes) : 0.0,
                        diag.nodes ? diag.lp_ms / double(diag.nodes) : 0.0);
            if (std::isfinite(diag.root_bound_before_cuts))
                std::printf("root bound:        %.10e -> %.10e after cuts\n",
                            diag.root_bound_before_cuts, diag.root_bound_after_cuts);
            std::printf("integer row roundings: %llu\n",
                        static_cast<unsigned long long>(diag.integer_row_roundings));
            std::printf("binary cover cuts: %llu\n",
                        static_cast<unsigned long long>(diag.binary_cover_cuts));
            std::printf("GMI cuts:          %llu in %d rounds\n",
                        static_cast<unsigned long long>(diag.gmi_cuts_added),
                        diag.cut_rounds);
            std::printf("LNS (bandit):      %llu hits / %llu built / %llu attempts, "
                        "%llu child nodes, %llu budget blocks (%.1f ms)\n",
                        static_cast<unsigned long long>(diag.lns.hits),
                        static_cast<unsigned long long>(diag.lns.built),
                        static_cast<unsigned long long>(diag.lns.attempts),
                        static_cast<unsigned long long>(diag.sub_mip_nodes),
                        static_cast<unsigned long long>(diag.lns.budget_blocks),
                        diag.sub_mip_ms);
            for (const auto& a : diag.lns.arms) {
                if (a.calls == 0) continue;
                std::printf("  arm %-12s calls %-5llu hits %-4llu "
                            "mean reward %.3f  fixing rate %.2f  %.2fs\n",
                            sor::search::to_string(a.kind),
                            static_cast<unsigned long long>(a.calls),
                            static_cast<unsigned long long>(a.hits),
                            a.calls ? a.reward_sum / double(a.calls) : 0.0,
                            a.fixing_rate, a.seconds);
            }
            std::printf("heuristics:        %.1f ms total, %llu budget blocks\n",
                        diag.heuristic_ms + diag.feasjump_ms + diag.sub_mip_ms,
                        static_cast<unsigned long long>(diag.heuristic_budget_blocks));
            std::printf("feasibility jump:  %llu attempts, %llu hits, %llu moves, "
                        "%llu reweights, %llu restarts, best %zu violated rows "
                        "(%.1f ms)\n",
                        static_cast<unsigned long long>(diag.feasjump_attempts),
                        static_cast<unsigned long long>(diag.feasjump_hits),
                        static_cast<unsigned long long>(diag.feasjump_moves),
                        static_cast<unsigned long long>(diag.feasjump_weight_updates),
                        static_cast<unsigned long long>(diag.feasjump_restarts),
                        diag.feasjump_best_violated_rows,
                        diag.feasjump_ms);
            std::printf("probing:           %llu probes, %llu fixings, "
                        "%llu implications, %llu bound tightenings%s (%.1f ms)\n",
                        static_cast<unsigned long long>(diag.conflict.probes),
                        static_cast<unsigned long long>(diag.conflict.probe_fixings),
                        static_cast<unsigned long long>(diag.conflict.probe_implications),
                        static_cast<unsigned long long>(diag.conflict.probe_tightenings),
                        diag.conflict.probing_truncated ? " [truncated]" : "",
                        diag.conflict.probe_ms);
            std::printf("conflict graph:    %llu row cliques, %llu edges\n",
                        static_cast<unsigned long long>(diag.conflict.row_cliques),
                        static_cast<unsigned long long>(diag.conflict.edges));
            std::printf("cut loop:          %.1f ms of the %.0f s budget\n",
                        diag.cut_loop_ms, bab.time_limit_s);
            std::printf("MIR cuts:          %llu of %llu candidates added; "
                        "%llu bases, rejected %llu frac / %llu dyn / %llu unviolated\n",
                        static_cast<unsigned long long>(diag.mir_cuts_added),
                        static_cast<unsigned long long>(diag.mir_candidates),
                        static_cast<unsigned long long>(diag.mir.bases_built),
                        static_cast<unsigned long long>(diag.mir.rejected_fractionality),
                        static_cast<unsigned long long>(diag.mir.rejected_dynamism),
                        static_cast<unsigned long long>(diag.mir.rejected_not_violated));
            std::printf("lifted covers:     %llu of %llu candidates added; "
                        "%llu covers from %llu knapsacks, %llu lifted coefs\n",
                        static_cast<unsigned long long>(diag.lifted_cover_cuts_added),
                        static_cast<unsigned long long>(diag.lifted_cover_candidates),
                        static_cast<unsigned long long>(diag.cover.covers_found),
                        static_cast<unsigned long long>(diag.cover.knapsacks_built),
                        static_cast<unsigned long long>(diag.cover.lifted_coefficients));
            std::printf("implied bounds:    %llu recorded; %llu of %llu VUB "
                        "cut candidates added\n",
                        static_cast<unsigned long long>(diag.conflict.implied_bounds),
                        static_cast<unsigned long long>(diag.implied_bound_cuts_added),
                        static_cast<unsigned long long>(diag.implied_bound_cut_candidates));
            std::printf("clique cuts:       %llu of %llu candidates added; "
                        "node propagation %llu tightenings, %llu prunes\n",
                        static_cast<unsigned long long>(diag.clique_cuts_added),
                        static_cast<unsigned long long>(diag.clique_cut_candidates),
                        static_cast<unsigned long long>(diag.conflict_prop_tightenings),
                        static_cast<unsigned long long>(diag.conflict_prop_prunes));
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
            std::printf("    reductions     %10d rows / %d cols\n",
                        static_cast<int>(diag.presolve_rows_removed),
                        static_cast<int>(diag.presolve_cols_removed));
            std::printf("    singleton cols %10d\n",
                        static_cast<int>(
                            diag.presolve_singleton_columns_removed));
            std::printf("    forcing rows  %10d  (%d columns fixed)\n",
                        static_cast<int>(diag.presolve_forcing_rows_removed),
                        static_cast<int>(diag.presolve_forcing_columns_fixed));
            std::printf("    aggregations  %10d  (%lld positive fill)\n",
                        static_cast<int>(diag.presolve_equality_aggregations),
                        static_cast<long long>(diag.presolve_aggregation_fill));
            std::printf("    retries       %10llu\n",
                        static_cast<unsigned long long>(diag.presolve_retries));
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
            std::printf("  phase restarts   %10llu  (%llu dual resyncs)\n",
                        static_cast<unsigned long long>(diag.phase_restarts),
                        static_cast<unsigned long long>(diag.dual_resyncs));
            std::printf("  phase1 cost delta %9llu / %llu changed, max %llu\n",
                        static_cast<unsigned long long>(
                            diag.phase1_cost_change_iterations),
                        static_cast<unsigned long long>(diag.phase1_cost_changes),
                        static_cast<unsigned long long>(diag.phase1_cost_change_max));
            std::printf("  primal BTRAN     %10llu / %llu sparse, avg support %llu\n",
                        static_cast<unsigned long long>(diag.primal_btran_sparse),
                        static_cast<unsigned long long>(
                            diag.primal_btran_sparse + diag.primal_btran_dense),
                        static_cast<unsigned long long>(
                            diag.primal_btran_sparse
                                ? diag.primal_btran_support_entries /
                                      diag.primal_btran_sparse
                                : 0));
            std::printf("  phase1 composite %10llu  (%llu sparse / %llu dense / %llu fallback), avg support %llu\n",
                        static_cast<unsigned long long>(
                            diag.phase1_composite_updates),
                        static_cast<unsigned long long>(
                            diag.phase1_composite_sparse),
                        static_cast<unsigned long long>(
                            diag.phase1_composite_dense),
                        static_cast<unsigned long long>(
                            diag.phase1_composite_fallbacks),
                        static_cast<unsigned long long>(
                            diag.phase1_composite_sparse
                                ? diag.phase1_composite_support_entries /
                                      diag.phase1_composite_sparse
                                : 0));
            std::printf("  composite error %11.3g\n",
                        diag.phase1_composite_max_abs_error);
            std::printf("  Devex frameworks %10llu  (%llu exact checks)\n",
                        static_cast<unsigned long long>(diag.devex_frameworks),
                        static_cast<unsigned long long>(diag.devex_weight_checks));
            std::printf("  DSE weight checks %10llu  (%llu rejected rows)\n",
                        static_cast<unsigned long long>(diag.dse_weight_checks),
                        static_cast<unsigned long long>(diag.dse_weight_rejections));
            std::printf("  DSE->Devex       %10llu  (%llu accuracy, %llu stability, %llu costly DSE iters)\n",
                        static_cast<unsigned long long>(diag.dse_to_devex_switches),
                        static_cast<unsigned long long>(diag.dse_accuracy_switches),
                        static_cast<unsigned long long>(diag.dse_stability_switches),
                        static_cast<unsigned long long>(diag.costly_dse_iterations));
            std::printf("  DSE log error    %10.3g\n", diag.dse_log_weight_error);
            std::printf("  paired FTRAN     %10llu\n",
                        static_cast<unsigned long long>(diag.dual_paired_ftrans));
            std::printf("  pivotal support  %10llu -> %llu entries\n",
                        static_cast<unsigned long long>(
                            diag.dual_pivotal_entries_full),
                        static_cast<unsigned long long>(
                            diag.dual_pivotal_entries_kept));
            std::printf("  dual pricing init D:%llu V:%llu S:%llu\n",
                        static_cast<unsigned long long>(diag.dual_dantzig_starts),
                        static_cast<unsigned long long>(diag.dual_devex_starts),
                        static_cast<unsigned long long>(diag.dual_dse_starts));
            std::printf("  cost perturb     %10llu  (%llu cleanups)\n",
                        static_cast<unsigned long long>(diag.perturbed_costs),
                        static_cast<unsigned long long>(diag.perturbation_cleanups));
            std::printf("  cost shifts      %10llu  (%llu wrong-sign entering, max %.3e)\n",
                        static_cast<unsigned long long>(diag.cost_shifts),
                        static_cast<unsigned long long>(
                            diag.wrong_sign_entering_shifts),
                        diag.cost_shift_max);
            std::printf("  primal cleanup   %10llu  (%llu pivots, handed dual "
                        "infeas %.3e / primal infeas %.3e)\n",
                        static_cast<unsigned long long>(diag.primal_cleanups),
                        static_cast<unsigned long long>(
                            diag.primal_cleanup_iterations),
                        diag.cleanup_dual_infeasibility,
                        diag.cleanup_primal_infeasibility);
            std::printf("  trouble refactor %10llu  (%llu shifts refused)\n",
                        static_cast<unsigned long long>(
                            diag.numerical_trouble_refactors),
                        static_cast<unsigned long long>(
                            diag.refused_cost_shifts));
            std::printf("  ratio sorted     %10llu candidates\n",
                        static_cast<unsigned long long>(
                            diag.ratio_sorted_candidates));
            std::printf("  dual ratio test  %llu groups, %llu back-offs, %llu exhausted, %llu tiny excluded\n",
                        static_cast<unsigned long long>(diag.ratio_groups),
                        static_cast<unsigned long long>(diag.ratio_backoffs),
                        static_cast<unsigned long long>(diag.ratio_exhausted),
                        static_cast<unsigned long long>(
                            diag.ratio_small_pivot_exclusions));
            std::printf("  simplex loop     %10.3f\n", diag.loop_ms);
            std::printf("  flip batches     %10.3f  (%llu batches)\n", diag.flip_ms,
                        static_cast<unsigned long long>(diag.flip_batches));
            std::printf("  apply_pivot      %10.3f\n", diag.pivot_apply_ms);
            std::printf("  alpha sparse     %llu / %llu iters, avg support %llu\n",
                        static_cast<unsigned long long>(diag.alpha_sparse_iters),
                        static_cast<unsigned long long>(
                            diag.alpha_sparse_iters + diag.alpha_dense_iters),
                        static_cast<unsigned long long>(
                            diag.alpha_sparse_iters
                                ? diag.alpha_support_entries / diag.alpha_sparse_iters
                                : 0));
            std::printf("  primal FTRAN switches %6llu\n",
                        static_cast<unsigned long long>(
                            diag.primal_ftran_dense_switches));
            std::printf("  primal crash     %10llu  (infeas %.6g -> %.6g)\n",
                        static_cast<unsigned long long>(diag.primal_crash_columns),
                        diag.primal_crash_infeasibility_before,
                        diag.primal_crash_infeasibility_after);
            std::printf("  auto stages/builds %8llu / %llu\n",
                        static_cast<unsigned long long>(diag.stages),
                        static_cast<unsigned long long>(diag.preprocessing_builds));
            std::printf("  stage path       P:%llu D:%llu cold:%llu basis-restart:%llu\n",
                        static_cast<unsigned long long>(diag.primal_stages),
                        static_cast<unsigned long long>(diag.dual_stages),
                        static_cast<unsigned long long>(diag.cold_stages),
                        static_cast<unsigned long long>(diag.basis_restarts));
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
