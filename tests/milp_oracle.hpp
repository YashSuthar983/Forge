// Random small MILPs and an exhaustive oracle for them, shared by the
// oracle-based MILP tests. Every integer assignment is enumerated and the
// continuous remainder of each is solved by certified simplex, so the oracle
// answer does not depend on any branch-and-bound code under test.
#pragma once

#include "sor/certify/finalize.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace sor::test::oracle {

using sor::core::Index;
using sor::core::Status;
using sor::model::LpProblem;

inline std::uint32_t next(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline LpProblem make_random_milp(std::uint32_t seed) {
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
    std::vector<double> x;
};

inline OracleResult solve_oracle(const LpProblem& lp) {
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
    
    // Build the continuous remainder once. Integer assignments shift only
    // row bounds and the objective offset; integer columns are eliminated.
    LpProblem remainder;
    remainder.maximize = lp.maximize;
    remainder.row_lo = lp.row_lo;
    remainder.row_hi = lp.row_hi;
    std::vector<Index> position(lp.n_cols(), -1);
    for (Index j : cont_cols) {
        position[j] = static_cast<Index>(remainder.c.size());
        remainder.c.push_back(lp.c[j]);
        remainder.col_lo.push_back(lp.col_lo[j]);
        remainder.col_hi.push_back(lp.col_hi[j]);
    }
    std::vector<Index> rows, cols;
    std::vector<double> vals;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < lp.n_rows(); ++i)
        for (auto k = rp[i]; k < rp[i + 1]; ++k)
            if (position[ci[k]] >= 0) {
                rows.push_back(i); cols.push_back(position[ci[k]]);
                vals.push_back(lp.A.vals[k]);
            }
    remainder.A = sor::sparse::from_triplets(lp.n_rows(),
        static_cast<Index>(cont_cols.size()), rows, cols, vals);
    std::vector<double> assigned(lp.n_cols(), 0.0);
    for (const auto& asn : int_assignments) {
        LpProblem sub = remainder;
        sub.obj_offset = lp.obj_offset;
        for (std::size_t k = 0; k < int_cols.size(); ++k) {
            assigned[int_cols[k]] = asn[k];
            sub.obj_offset += lp.c[int_cols[k]] * asn[k];
        }
        bool feasible = true;
        for (Index i = 0; i < lp.n_rows(); ++i) {
            double shift = 0.0;
            for (auto k = rp[i]; k < rp[i + 1]; ++k)
                if (position[ci[k]] < 0) shift += lp.A.vals[k] * assigned[ci[k]];
            sub.row_lo[i] -= shift;
            sub.row_hi[i] -= shift;
            // Exact interval contradiction: all generated coefficients,
            // assignment values and box endpoints are small integers.
            double lower_activity = 0.0, upper_activity = 0.0;
            for (auto k = remainder.A.pattern.row_ptr()[i];
                 k < remainder.A.pattern.row_ptr()[i + 1]; ++k) {
                const auto j = remainder.A.pattern.col_idx()[k];
                const double a = remainder.A.vals[k];
                lower_activity += a * (a > 0.0 ? sub.col_lo[j] : sub.col_hi[j]);
                upper_activity += a * (a > 0.0 ? sub.col_hi[j] : sub.col_lo[j]);
            }
            if (lower_activity > sub.row_hi[i] + 1e-9 ||
                upper_activity < sub.row_lo[i] - 1e-9) feasible = false;
        }
        if (!feasible) continue;
        double objective = sub.obj_offset;
        if (!cont_cols.empty()) {
            sor::engines::SimplexOptions opts;
            opts.presolve = false; // Require original-remainder terminal certificates.
            sor::engines::SimplexDiagnostics diag;
            auto raw = sor::engines::solve_simplex(sub, opts, diag);
            auto ev = sor::certify::check_lp_result(sub, raw,
                sor::engines::simplex_evidence(diag, opts));
            auto res = sor::certify::finalize_result(std::move(raw), ev);
            if (res.status != Status::Optimal && res.status != Status::Infeasible) {
                std::cerr << "UNRESOLVED ORACLE LP: " << lp.name << " status="
                          << static_cast<int>(res.status) << " " << res.downgrade_reason << '\n';
                CHECK(false);
                continue; // The failed assertion forbids treating this as a proof.
            }
            feasible = res.status == Status::Optimal;
            objective = res.objective;
            if (feasible) {
                CHECK(res.x.size() == cont_cols.size());
                for (std::size_t k = 0; k < cont_cols.size(); ++k)
                    assigned[cont_cols[k]] = res.x[k];
            }
        }
        if (feasible && (!best.feasible || (lp.maximize ? objective > best.objective
                                                       : objective < best.objective))) {
            best.feasible = true;
            best.objective = objective;
            best.x = assigned;
        }
    }
    return best;
}

// Pure-integer models with many fractional LP variables: 8-11 binary/small
// general integer columns and 3-5 dense knapsack-type rows, so the tree
// branches on many candidates and dead probe directions are common. No
// continuous columns keeps the oracle a plain enumeration.
inline sor::model::LpProblem make_branching_ip(std::uint32_t seed) {
    using sor::core::Index;
    std::uint32_t st = seed * 2654435761u + 40503u;
    const Index n = 8 + next(st) % 4;
    const Index m = 3 + next(st) % 3;
    sor::model::LpProblem lp;
    lp.name = "ip_" + std::to_string(seed);
    lp.maximize = (next(st) & 1u) != 0;
    lp.obj_offset = static_cast<double>(static_cast<int>(next(st) % 9) - 4);
    lp.c.assign(n, 0.0);
    lp.col_lo.assign(n, 0.0);
    lp.col_hi.assign(n, 1.0);
    lp.is_integer.assign(n, true);
    for (Index j = 0; j < n; ++j) {
        lp.c[j] = 1.0 + next(st) % 20;
        if (next(st) % 4 == 0) lp.col_hi[j] = 2.0;
    }
    std::vector<Index> rows, cols;
    std::vector<double> vals;
    lp.row_lo.assign(m, -sor::model::kInf);
    lp.row_hi.assign(m, sor::model::kInf);
    for (Index i = 0; i < m; ++i) {
        double sum = 0.0;
        for (Index j = 0; j < n; ++j) {
            const double a = 2.0 + next(st) % 9;
            rows.push_back(i); cols.push_back(j); vals.push_back(a);
            sum += a * lp.col_hi[j];
        }
        const double b = std::floor(sum * (30 + next(st) % 30) / 100.0);
        if (next(st) % 5 == 0) lp.row_lo[i] = std::floor(sum * 0.15);
        lp.row_hi[i] = b;
    }
    lp.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
    lp.validate();
    return lp;
}

}  // namespace sor::test::oracle
