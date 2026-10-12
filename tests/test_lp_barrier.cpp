// LP barrier route: interior point on the (presolved) LP, crossover to a
// basis, and the independent original-model checker. The IPM's own claim is
// never trusted, so every accepted proof here comes from the LP checker.
#include "sor/certify/finalize.hpp"
#include "sor/engines/lp.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/io/mps.hpp"
#include "sor/model/lp.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <fstream>
#include <string>

using namespace sor;

namespace {

std::string netlib_path(const std::string& name) {
    return std::string(SOR_SOURCE_DIR) + "/benchmarks/netlib/mps/" + name + ".mps";
}

model::LpProblem read_netlib(const std::string& name) {
    std::ifstream input(netlib_path(name));
    io::MpsReadReport report;
    return io::read_mps(input, report);
}

core::SolveResult solve(const model::LpProblem& problem, bool crossover, bool presolve,
                        core::LpDiagnostics& diagnostics) {
    core::LpOptions options;
    options.strategy = core::LpStrategy::Barrier;
    options.time_limit_s = 60;
    options.primal_feas_tol = 1e-7;
    options.dual_feas_tol = 1e-7;
    options.gap_tol = 1e-8;
    options.fo_crossover = crossover;
    options.presolve = presolve;
    core::ProofEvidence evidence;
    auto raw = engines::solve_lp(problem, options, diagnostics, &evidence);
    return certify::finalize_result(certify::check_lp_candidate(problem, std::move(raw), evidence));
}

}  // namespace

int main() {
    if (!test::data_available(netlib_path("afiro")))
        return test::skip("test_lp_barrier", netlib_path("afiro"));
    // afiro: optimum -464.7531428571 (Netlib). Barrier plus crossover must
    // reach a checked basis proof, with and without presolve.
    for (const bool presolve : {false, true}) {
        const auto afiro = read_netlib("afiro");
        core::LpDiagnostics diagnostics;
        const auto result = solve(afiro, true, presolve, diagnostics);
        CHECK(result.status == core::Status::Optimal);
        CHECK(result.proof == core::ProofLevel::ProvedOptimalFP);
        CHECK(std::fabs(result.objective + 464.7531428571) < 1e-6);
        CHECK(diagnostics.routed_strategy == core::LpStrategy::Barrier);
        CHECK(diagnostics.fo_iterations > 0);
        CHECK(diagnostics.crossover_attempted);
    }
    // The same model maximized with negated costs: the IPM only minimizes,
    // so the route's sense and dual-sign mapping must hold end to end.
    {
        auto afiro = read_netlib("afiro");
        afiro.maximize = true;
        for (auto& c : afiro.c) c = -c;
        core::LpDiagnostics diagnostics;
        const auto result = solve(afiro, true, false, diagnostics);
        CHECK(result.status == core::Status::Optimal);
        CHECK(result.proof == core::ProofLevel::ProvedOptimalFP);
        CHECK(std::fabs(result.objective - 464.7531428571) < 1e-6);
    }
    // Without crossover, the interior point alone is checked on the model:
    // a primal point within tolerance and a finite checked gap, no basis.
    {
        const auto afiro = read_netlib("afiro");
        core::LpDiagnostics diagnostics;
        const auto result = solve(afiro, false, false, diagnostics);
        CHECK(std::fabs(result.objective + 464.7531428571) < 1e-5);
        CHECK(result.proof != core::ProofLevel::ProvedOptimalFP);
        CHECK(!diagnostics.crossover_attempted);
    }
    // Without the exact proof (as the CLI runs) the cold simplex fallback
    // after a crossover that found no valid basis claims tolerance-level KKT
    // optimality. The route must accept it: it used to require the exact
    // dual bound and kept the interior point instead, Feasible on scsd6.
    if (test::data_available(netlib_path("scsd6"))) {
        const auto scsd6 = read_netlib("scsd6");
        core::LpOptions options;
        options.strategy = core::LpStrategy::Barrier;
        options.time_limit_s = 60;
        engines::SimplexOptions policy;
        policy.exact_proof = false;
        core::LpDiagnostics diagnostics;
        core::ProofEvidence evidence;
        auto raw = engines::solve_lp(scsd6, options, diagnostics, &evidence, &policy);
        const auto result = certify::finalize_result(
            certify::check_lp_candidate(scsd6, std::move(raw), evidence));
        CHECK(result.status == core::Status::Optimal);
        CHECK(result.proof == core::ProofLevel::ProvedKKT ||
              result.proof == core::ProofLevel::ProvedOptimalFP);
        CHECK(std::fabs(result.objective - 50.5) < 1e-6);
        CHECK(diagnostics.crossover_attempted);
    }
    return test::finish("test_lp_barrier");
}
