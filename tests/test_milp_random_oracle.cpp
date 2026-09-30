
#include "sor/certify/finalize.hpp"
#include "sor/search/bab.hpp"
#include "sor/engines/simplex.hpp"

#include "test_helpers.hpp"

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

std::uint32_t next(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

LpProblem make_random_milp(std::uint32_t seed) {
    std::uint32_t state = seed * 747796405u + 2891336453u;
    
    Index n = 3 + (next(state) % 6); // 3 to 8 columns
    Index m = 2 + (next(state) % 5); // 2 to 6 rows

    LpProblem lp;
    lp.name = "rand_" + std::to_string(seed);
    lp.maximize = (next(state) & 1u) != 0;
    
    lp.c.assign(n, 0.0);
    lp.col_lo.assign(n, 0.0);
    lp.col_hi.assign(n, 0.0);
    lp.is_integer.assign(n, false);
    lp.col_names.assign(n, "");
    
    for (Index j = 0; j < n; ++j) {
        lp.col_names[j] = "x" + std::to_string(j);
        lp.c[j] = static_cast<int>(next(state) % 11) - 5;
        
        int type = next(state) % 3;
        if (type == 0) { // binary
            lp.col_lo[j] = 0.0;
            lp.col_hi[j] = 1.0;
            lp.is_integer[j] = true;
        } else if (type == 1) { // integer
            lp.col_lo[j] = static_cast<int>(next(state) % 4) - 2;
            lp.col_hi[j] = lp.col_lo[j] + 1.0 + static_cast<int>(next(state) % 4);
            lp.is_integer[j] = true;
        } else { // continuous
            lp.col_lo[j] = 0.0;
            lp.col_hi[j] = 2.0 + static_cast<int>(next(state) % 5);
            lp.is_integer[j] = false;
        }
    }

    std::vector<Index> rows, cols;
    std::vector<double> vals;
    lp.row_lo.resize(m);
    lp.row_hi.resize(m);
    lp.row_names.resize(m);
    
    for (Index i = 0; i < m; ++i) {
        lp.row_names[i] = "r" + std::to_string(i);
        
        for (Index j = 0; j < n; ++j) {
            if ((next(state) % 3) != 0) {
                int a = static_cast<int>(next(state) % 11) - 5;
                if (a != 0) {
                    rows.push_back(i);
                    cols.push_back(j);
                    vals.push_back(static_cast<double>(a));
                }
            }
        }
        
        int sense = next(state) % 3;
        double b = static_cast<int>(next(state) % 21) - 10;
        if (sense == 0) { // <=
            lp.row_lo[i] = -sor::model::kInf;
            lp.row_hi[i] = b;
        } else if (sense == 1) { // >=
            lp.row_lo[i] = b;
            lp.row_hi[i] = sor::model::kInf;
        } else { // ==
            lp.row_lo[i] = b;
            lp.row_hi[i] = b;
        }
    }
    
    lp.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
    lp.validate();
    return lp;
}

struct OracleResult {
    bool feasible = false;
    double objective = 0.0;
};

OracleResult solve_oracle(const LpProblem& lp) {
    std::vector<Index> int_cols;
    std::vector<Index> cont_cols;
    for (Index j = 0; j < static_cast<Index>(lp.c.size()); ++j) {
        if (lp.is_integer[j]) int_cols.push_back(j);
        else cont_cols.push_back(j);
    }
    
    OracleResult best;
    best.feasible = false;
    best.objective = lp.maximize ? -sor::model::kInf : sor::model::kInf;
    
    // Build combinations
    std::vector<std::vector<double>> int_assignments;
    int_assignments.push_back({}); // start with empty
    
    for (Index j : int_cols) {
        std::vector<std::vector<double>> next_assignments;
        int lo = static_cast<int>(std::ceil(lp.col_lo[j]));
        int hi = static_cast<int>(std::floor(lp.col_hi[j]));
        for (const auto& asn : int_assignments) {
            for (int val = lo; val <= hi; ++val) {
                auto next_asn = asn;
                next_asn.push_back(static_cast<double>(val));
                next_assignments.push_back(next_asn);
            }
        }
        int_assignments = std::move(next_assignments);
    }
    
    // For each assignment, solve continuous LP remainder
    for (const auto& asn : int_assignments) {
        LpProblem sub = lp;
        for (size_t k = 0; k < int_cols.size(); ++k) {
            Index j = int_cols[k];
            sub.col_lo[j] = asn[k];
            sub.col_hi[j] = asn[k];
        }
        
        sor::engines::SimplexOptions opts;
        sor::engines::SimplexDiagnostics diag;
        auto raw = sor::engines::solve_simplex(sub, opts, diag);
        auto ev = sor::engines::simplex_evidence(diag, opts);
        auto res = sor::certify::finalize_result(std::move(raw), ev);
        
        if (res.status == Status::Optimal) {
            if (!best.feasible) {
                best.feasible = true;
                best.objective = res.objective;
            } else {
                if (lp.maximize && res.objective > best.objective) best.objective = res.objective;
                if (!lp.maximize && res.objective < best.objective) best.objective = res.objective;
            }
        }
    }
    return best;
}

void test_random_milps() {
    for (std::uint32_t seed = 1; seed <= 2000; ++seed) {
        const LpProblem lp = make_random_milp(seed);
        const OracleResult oracle = solve_oracle(lp);
        
        std::vector<sor::search::BabOptions> configs;
        
        // default options
        sor::search::BabOptions default_opts;
        default_opts.para_bab.threads = 1;
        default_opts.structural_presolve.enabled = false;
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
        
        for (const auto& opts : configs) {
            sor::search::BabDiagnostics diag;
            auto raw = sor::search::solve_milp(lp, opts, diag);
            const auto ev = sor::search::milp_evidence(diag, opts);
            const auto result = sor::certify::finalize_result(std::move(raw), ev);
            
            if (oracle.feasible) {
                if (result.status != Status::Optimal || std::abs(result.objective - oracle.objective) > 1e-5) {
                    std::cout << "FAIL: seed=" << seed << " oracle.obj=" << oracle.objective 
                              << " solver.status=" << (int)result.status << " solver.obj=" << result.objective << std::endl;
                }
                CHECK(result.status == Status::Optimal);
                CHECK_NEAR(result.objective, oracle.objective, 1e-6);
            } else {
                if (result.status != Status::Infeasible) {
                    std::cout << "FAIL: seed=" << seed << " oracle.feas=false"
                              << " solver.status=" << (int)result.status << " solver.obj=" << result.objective << std::endl;
                }
                CHECK(result.status == Status::Infeasible);
            }
        }
    }
}

} // namespace

int main() {
    test_random_milps();
    return sor::test::finish("test_milp_random_oracle");
}
