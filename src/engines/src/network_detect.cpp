#include "forge/engines/network_detect.hpp"
#include "forge/core/result.hpp"

#include <cmath>
#include <vector>

namespace forge::engines {

using core::f64;

bool detect_network_structure(const model::LpProblem& problem) {
    problem.validate(/*allow_empty_domains=*/true);
    const auto nc = static_cast<std::size_t>(problem.n_cols());
    if (nc == 0 || problem.nnz() == 0) return false;

    struct ColEntry {
        int count = 0;
        f64 pos = 0.0;
        f64 neg = 0.0;
        core::Index pos_row = -1;
        core::Index neg_row = -1;
    };
    std::vector<ColEntry> cols(nc);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        for (core::Offset k = rp[static_cast<std::size_t>(i)];
             k < rp[static_cast<std::size_t>(i) + 1]; ++k) {
            const auto j = static_cast<std::size_t>(ci[static_cast<std::size_t>(k)]);
            const f64 a = problem.A.vals[static_cast<std::size_t>(k)];
            if (a == 0.0) continue;
            ColEntry& col = cols[j];
            ++col.count;
            if (col.count > 2) return false;
            if (a > 0.0) {
                if (col.pos != 0.0) return false;
                col.pos = a;
                col.pos_row = i;
            } else {
                if (col.neg != 0.0) return false;
                col.neg = a;
                col.neg_row = i;
            }
        }
    }
    for (const ColEntry& col : cols) {
        if (col.count == 0) continue;
        if (col.count == 1) continue;
        if (col.pos <= 0.0 || col.neg >= 0.0) return false;
        if (col.pos_row == col.neg_row) return false;
    }
    return true;
}

}  // namespace forge::engines
