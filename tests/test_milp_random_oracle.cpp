
#include "sor/certify/finalize.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/portfolio.hpp"
#include "sor/engines/simplex.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>

using sor::core::Index;
using sor::core::Status;
using sor::model::LpProblem;

namespace {

using sor::test::oracle::next;
using sor::test::oracle::make_random_milp;
using sor::test::oracle::solve_oracle;
using sor::test::oracle::OracleResult;

LpProblem make_separator_milp(std::uint32_t seed, bool zh) {
    std::uint32_t state = seed * 747796405u + 2891336453u;
    Index n = zh ? 3 + 2 * (next(state) % 3) : 6 + next(state) % 3;
    Index m = zh ? n : 3;
    LpProblem p;
    p.maximize = true;
    p.name = std::string(zh ? "odd_" : "knapsack_") + std::to_string(seed);
    p.c.assign(n, 1.0);
    p.col_lo.assign(n, 0.0);
    p.col_hi.assign(n, zh ? 2.0 : 1.0);
    p.is_integer.assign(n, true);
    p.row_lo.assign(m, -sor::model::kInf);
    p.row_hi.resize(m);
    std::vector<Index> rows, cols;
    std::vector<double> vals;
    if (zh) {
        const double a = 3 + 2 * (next(state) % 5);
        for (Index i = 0; i < m; ++i) {
            rows.insert(rows.end(), {i, i});
            cols.insert(cols.end(), {i, (i + 1) % n});
            vals.insert(vals.end(), {a, a});
            p.row_hi[i] = a;
        }
    } else {
        for (Index i = 0; i < m; ++i) {
            double sum = 0.0;
            for (Index j = 0; j < n; ++j) {
                const double a = 5 + next(state) % 26;
                rows.push_back(i); cols.push_back(j); vals.push_back(a); sum += a;
                if (i == 0) p.c[j] = a;
            }
            p.row_hi[i] = std::floor(sum * (40 + next(state) % 21) / 100.0);
        }
    }
    p.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
    p.validate();
    return p;
}
void test_random_milps() {
    std::array<std::uint64_t, 4> cut_models{}, cut_totals{};
    for (std::uint32_t seed = 1; seed <= 2000; ++seed) {
        const LpProblem lp = make_random_milp(seed);
        const OracleResult oracle = solve_oracle(lp);
        
        std::vector<sor::search::BabOptions> configs;
        
        // default options
        sor::search::BabOptions default_opts;
        default_opts.para_bab.threads = 1;
        default_opts.structural_presolve.enabled = false;
        default_opts.cuts_enabled = true;
        if (oracle.feasible) {
            CHECK(lp.max_row_violation(oracle.x) <= 1e-7);
            CHECK(lp.max_bound_violation(oracle.x) <= 1e-7);
            default_opts.cut_reference_point = &oracle.x;
        }
        const auto default_gap_opts = default_opts;
        default_opts.gap_tol = 1e-9;
        default_opts.abs_gap_tol = 1e-9;
        configs.push_back(default_opts);
        
        // mir_cuts
        auto mir_opts = default_opts;
        mir_opts.mir_cuts = true;
        mir_opts.lifted_cover_cuts = false;
        mir_opts.zerohalf_cuts = false;
        mir_opts.clique_cuts = false;
        configs.push_back(mir_opts);
        
        // lifted_cover_cuts
        auto cov_opts = default_opts;
        cov_opts.mir_cuts = false;
        cov_opts.lifted_cover_cuts = true;
        cov_opts.zerohalf_cuts = false;
        cov_opts.clique_cuts = false;
        configs.push_back(cov_opts);
        
        // zerohalf_cuts
        auto zh_opts = default_opts;
        zh_opts.mir_cuts = false;
        zh_opts.lifted_cover_cuts = false;
        zh_opts.zerohalf_cuts = true;
        zh_opts.clique_cuts = false;
        configs.push_back(zh_opts);
        
        // clique_cuts
        auto cq_opts = default_opts;
        cq_opts.mir_cuts = false;
        cq_opts.lifted_cover_cuts = false;
        cq_opts.zerohalf_cuts = false;
        cq_opts.clique_cuts = true;
        configs.push_back(cq_opts);
        
        // Numerical objective checks above are exact; keep a separate run
        // that permits the solver's default MIP gap and checks its bound.
        configs.push_back(default_gap_opts);
        for (std::size_t config = 0; config < configs.size(); ++config) {
            const auto& opts = configs[config];
            const bool default_gap = config + 1 == configs.size();
            sor::search::BabDiagnostics diag;
            auto raw = sor::search::solve_milp(lp, opts, diag);
            if (config >= 1 && config <= 4) {
                CHECK(diag.invalid_cuts_detected == 0);
                CHECK(diag.node_cuts_invalid_inserted == 0);
                const std::array<std::uint64_t, 4> added{
                    diag.mir_cuts_added, diag.lifted_cover_cuts_added,
                    diag.zerohalf_cuts_added, diag.clique_cuts_added};
                cut_models[config - 1] += added[config - 1] > 0;
                cut_totals[config - 1] += added[config - 1];
            }
            const auto ev = sor::search::milp_evidence(diag, opts);
            const auto result = sor::certify::finalize_result(std::move(raw), ev);
            
            if (oracle.feasible) {
                const double allowed_error = default_gap
                    ? std::max(1e-6, 1e-4 * std::fabs(oracle.objective)) : 1e-6;
                if (result.status != Status::Optimal ||
                    std::abs(result.objective - oracle.objective) > allowed_error) {
                    std::cout << "FAIL: seed=" << seed << " oracle.obj=" << oracle.objective 
                              << " solver.status=" << (int)result.status << " solver.obj=" << result.objective << std::endl;
                }
                CHECK(result.status == Status::Optimal);
                CHECK(std::isfinite(result.objective) &&
                      std::fabs(result.objective - oracle.objective) <= allowed_error);
                if (default_gap) {
                    CHECK(std::isfinite(diag.dual_bound));
                    CHECK(lp.maximize ? diag.dual_bound >= oracle.objective - 1e-7
                                      : diag.dual_bound <= oracle.objective + 1e-7);
                }
            } else {
                if (result.status != Status::Infeasible) {
                    std::cout << "FAIL: seed=" << seed << " oracle.feas=false"
                              << " solver.status=" << (int)result.status << " solver.obj=" << result.objective << std::endl;
                }
                CHECK(result.status == Status::Infeasible);
            }
        }
    }
    constexpr std::array<const char*, 4> families{"mir", "lifted_cover", "zerohalf", "clique"};
    for (std::size_t k = 0; k < families.size(); ++k)
        std::cout << "CUT_COUNTS " << families[k] << " models=" << cut_models[k]
                  << " cuts=" << cut_totals[k] << '\n';

}


// Direct separation checks cover candidates that selection or duplicate
// filtering may omit from the solver's cut-reference diagnostic.
void test_separator_families() {
    for (bool odd : {false, true}) {
        std::uint64_t generated_models = 0, generated_cuts = 0;
        std::uint64_t inserted_models = 0, inserted_cuts = 0;
        for (std::uint32_t seed = 1; seed <= 200; ++seed) {
            const auto lp = make_separator_milp(seed, odd);
            const auto oracle = solve_oracle(lp);
            CHECK(oracle.feasible);
            if (!oracle.feasible) continue;
            sor::engines::SimplexOptions sx;
            sx.presolve = false;
            sor::engines::SimplexDiagnostics sd;
            auto relaxation = sor::engines::solve_simplex(lp, sx, sd);
            const auto evidence = sor::certify::check_lp_result(lp, relaxation,
                sor::engines::simplex_evidence(sd, sx));
            auto checked = sor::certify::finalize_result(std::move(relaxation), evidence);
            CHECK(checked.status == Status::Optimal);
            if (checked.status != Status::Optimal) continue;
            std::vector<sor::search::CutRow> cuts;
            if (odd) {
                sor::search::ZeroHalfOptions options;
                sor::search::ZeroHalfDiagnostics diagnostics;
                cuts = sor::search::separate_zerohalf(lp, checked.x,
                    lp.col_lo, lp.col_hi, options, diagnostics);
            } else {
                sor::search::CoverOptions options;
                sor::search::CoverDiagnostics diagnostics;
                cuts = sor::search::separate_lifted_covers(lp, checked.x,
                    lp.col_lo, lp.col_hi, options, diagnostics);
            }
            generated_models += !cuts.empty();
            generated_cuts += cuts.size();
            std::uint64_t assignments = 1;
            for (double hi : lp.col_hi) assignments *= static_cast<std::uint64_t>(hi) + 1;
            std::vector<double> point(lp.n_cols(), 0.0);
            for (std::uint64_t assignment = 0; assignment < assignments; ++assignment) {
                auto code = assignment;
                for (Index j = 0; j < lp.n_cols(); ++j) {
                    const auto radix = static_cast<std::uint64_t>(lp.col_hi[j]) + 1;
                    point[j] = static_cast<double>(code % radix);
                    code /= radix;
                }
                if (lp.max_row_violation(point) > 1e-9) continue;
                for (const auto& cut : cuts)
                    CHECK(sor::search::cut_admits_point(cut.cols, cut.vals,
                          cut.row_lo, cut.row_hi, point, 1e-9));
            }
            sor::search::BabOptions options;
            options.para_bab.threads = 1;
            options.auto_cuts = false;
            options.mip_presolve = options.probing = options.symmetry = false;
            options.integer_row_rounding = false;
            options.structural_presolve.enabled = false;
            options.cuts_enabled = true;
            options.cut_reference_point = &oracle.x;
            options.lifted_cover_cuts = !odd;
            options.zerohalf_cuts = odd;
            options.gap_tol = options.abs_gap_tol = 1e-9;
            sor::search::BabDiagnostics diag;
            auto raw = sor::search::solve_milp(lp, options, diag);
            const auto result = sor::certify::finalize_result(std::move(raw),
                                sor::search::milp_evidence(diag, options));
            CHECK(result.status == Status::Optimal);
            CHECK_NEAR(result.objective, oracle.objective, 1e-6);
            CHECK(diag.invalid_cuts_detected == 0);
            CHECK(diag.node_cuts_invalid_inserted == 0);
            const auto inserted = odd ? diag.zerohalf_cuts_added
                                      : diag.lifted_cover_cuts_added;
            inserted_models += inserted > 0;
            inserted_cuts += inserted;
        }
        std::cout << "SEPARATOR_FAMILY " << (odd ? "zerohalf_odd_cycle" : "lifted_cover_knapsack")
                  << " generated_models=" << generated_models << " generated_cuts=" << generated_cuts
                  << " inserted_models=" << inserted_models << " inserted_cuts=" << inserted_cuts << '\n';
        CHECK(generated_models >= 50);
    }
}

} // namespace

int main() {
    test_random_milps();
    test_separator_families();
    return sor::test::finish("test_milp_random_oracle");
}
