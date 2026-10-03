// Live presolve rules against direct solves. Random small LPs built to
// trigger doubleton equalities (including chains, where one substitution
// fills a column into rows a later reduction rewrites) and parallel rows
// (both orientations, either row supplying the tighter side), doubletons
// whose eliminated column's bounds move to the kept column, and cost-tight
// singleton columns that make their inequality an equation. For every
// model with an optimum: presolve with the live rules, solve the reduced
// model, lift, and require the lift to meet the primal and dual residual
// tolerances on the ORIGINAL model with the direct optimum, and the lifted
// basis to name the lifted vertex: one basic variable per row, every
// nonbasic variable at the finite bound its status names, every basic
// variable priced at zero by the lifted duals. The default rule set alone
// (forcing rows, singleton rows made bounds) is held to the same.
#include "sor/engines/simplex.hpp"
#include "sor/presolve/presolve.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <vector>

using sor::core::Index;
using sor::model::LpProblem;

namespace {

struct Rng {
    std::uint64_t s;
    std::uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>(s >> 33);
    }
    int pick(int n) { return static_cast<int>(next() % static_cast<std::uint32_t>(n)); }
};

LpProblem make_lp(std::uint32_t seed) {
    Rng r{seed * 2654435761ull + 17};
    const double inf = sor::model::kInf;
    const Index n = 5 + r.pick(6);
    LpProblem lp;
    lp.c.resize(n);
    lp.col_lo.assign(n, 0.0);
    lp.col_hi.assign(n, 5.0);
    for (Index j = 0; j < n; ++j) {
        lp.c[j] = r.pick(7) - 3;
        if (r.pick(4) == 0) { lp.col_lo[j] = -inf; lp.col_hi[j] = inf; }
    }
    const double vals[] = {1.0, -1.0, 2.0, -2.0, 0.5, 3.0};
    std::vector<std::vector<std::pair<Index, double>>> rows;
    std::vector<double> lo, hi;
    const Index m = 4 + r.pick(6);
    for (Index i = 0; i < m; ++i) {
        std::vector<std::pair<Index, double>> row;
        const int kind = r.pick(10);
        if (kind < 4) {                       // doubleton equality
            const Index a = r.pick(n);
            Index b = r.pick(n);
            if (b == a) b = (a + 1) % n;
            row = {{std::min(a, b), vals[r.pick(6)]}, {std::max(a, b), vals[r.pick(6)]}};
            rows.push_back(row);
            const double rhs = r.pick(5) - 1;
            lo.push_back(rhs); hi.push_back(rhs);
            continue;
        }
        if (kind < 6 && !rows.empty()) {      // parallel copy of an earlier row
            const auto& src = rows[static_cast<std::size_t>(r.pick(static_cast<int>(rows.size())))];
            const double s = (r.pick(3) == 0) ? -2.0 : (r.pick(2) ? 2.0 : 0.5);
            for (const auto& [j, a] : src) row.push_back({j, s * a});
            rows.push_back(row);
            const double side = r.pick(9) - 2;
            if (r.pick(2)) { lo.push_back(-inf); hi.push_back(side); }
            else           { lo.push_back(side); hi.push_back(inf); }
            continue;
        }
        std::set<Index> cols;                 // general sparse inequality
        const int len = 2 + r.pick(3);
        while (static_cast<int>(cols.size()) < len) cols.insert(r.pick(n));
        for (const Index j : cols) row.push_back({j, vals[r.pick(6)]});
        rows.push_back(row);
        const double side = r.pick(11) - 1;
        if (r.pick(2)) { lo.push_back(-inf); hi.push_back(side); }
        else           { lo.push_back(side); hi.push_back(inf); }
    }
    // Cost-tight singletons: a new column x appearing only in one
    // two-entry inequality with a larger coefficient than its partner
    // (a capacity-style row, f - cap x <= 0 and its mirror images).
    const int singles = r.pick(4);
    for (int t = 0; t < singles; ++t) {
        const Index x = static_cast<Index>(lp.c.size());
        lp.c.push_back((r.pick(3) + 1) * (r.pick(2) ? 1.0 : -1.0));
        lp.col_lo.push_back(0.0);
        lp.col_hi.push_back(r.pick(2) ? 1.0 : 2.0);
        const Index f = r.pick(n);
        const double ax = (r.pick(2) ? 1.0 : -1.0) * (3 + r.pick(3));
        const double af = vals[r.pick(6)];
        rows.push_back({{f, af}, {x, ax}});
        const double side = r.pick(5) - 2;
        if (r.pick(2)) { lo.push_back(-inf); hi.push_back(side); }
        else           { lo.push_back(side); hi.push_back(inf); }
    }
    const Index ncols = static_cast<Index>(lp.c.size());
    std::vector<Index> ri, ci;
    std::vector<double> v;
    for (Index i = 0; i < static_cast<Index>(rows.size()); ++i)
        for (const auto& [j, a] : rows[static_cast<std::size_t>(i)]) {
            ri.push_back(i); ci.push_back(j); v.push_back(a);
        }
    lp.A = sor::sparse::from_triplets(static_cast<Index>(rows.size()), ncols, ri, ci, v);
    lp.row_lo = lo;
    lp.row_hi = hi;
    return lp;
}

}  // namespace

int main() {
    using sor::presolve::DualRecoveryKind;
    int optimal = 0, lifted_with_doubleton = 0, lifted_with_parallel = 0, chains = 0;
    int lifted_with_transfer = 0, lifted_with_tight_row = 0;
    int default_lifts = 0, default_forcing = 0, default_tightened = 0;
    // The live rules, then the default rule set alone (forcing rows and
    // singleton rows turned into bounds) on the same models.
    for (const bool live : {true, false})
    for (std::uint32_t seed = 1; seed <= 3000; ++seed) {
        const LpProblem lp = make_lp(seed);
        sor::engines::SimplexOptions direct;
        direct.presolve = false;
        direct.exact_proof = false;
        sor::engines::SimplexDiagnostics dd;
        const auto ref = sor::engines::solve_simplex(lp, direct, dd, nullptr);
        if (ref.proposed_status != sor::core::Status::Optimal) continue;
        optimal += live;

        sor::presolve::PresolveOptions po;
        po.live_reductions = live;
        po.parallel_rows = live;
        const auto out = sor::presolve::presolve(lp, po);
        if (out.status != sor::presolve::PresolveStatus::Reduced &&
            out.status != sor::presolve::PresolveStatus::Solved) {
            ::sor::test::report(false, "presolve concluded on a model with an optimum",
                                __FILE__, __LINE__, "seed " + std::to_string(seed));
            continue;
        }
        int doubletons = 0, parallels = 0, tight_rows = 0, transfers = 0;
        int forcing = 0, tightened = 0;
        for (const auto& step : out.map.recovery_steps) {
            doubletons += step.kind == DualRecoveryKind::DoubletonEquality;
            parallels += step.kind == DualRecoveryKind::ParallelRowMerge;
            tight_rows += step.kind == DualRecoveryKind::RowSideFixed;
            forcing += step.kind == DualRecoveryKind::ForcingRow;
            tightened += step.kind == DualRecoveryKind::BoundTightening;
        }
        for (const auto& rec : out.map.doubleton_equalities) transfers += rec.transferred;
        if (live ? doubletons == 0 && parallels == 0 : out.map.recovery_steps.empty()) continue;

        sor::presolve::PresolveReducedSolve rs;
        if (out.map.problem.n_rows() > 0 || out.map.problem.n_cols() > 0) {
            sor::engines::SimplexDiagnostics rd;
            sor::engines::SimplexBasis basis;
            const auto red = sor::engines::solve_simplex(out.map.problem, direct, rd, &basis);
            if (red.proposed_status != sor::core::Status::Optimal) {
                ::sor::test::report(false, "reduced model lost its optimum", __FILE__, __LINE__,
                                    "seed " + std::to_string(seed));
                continue;
            }
            rs.x = red.x;
            rs.y = red.y;
            rs.has_basis = !basis.status.empty();
            rs.basis.n_struct = basis.n_struct;
            rs.basis.basic = basis.basic;
            for (const auto s : basis.status)
                rs.basis.status.push_back(static_cast<sor::presolve::PostsolveNonbasicStatus>(s));
        }
        sor::presolve::PresolveRecoveryOptions ro;
        ro.certificate_time_limit_s = -1;
        ro.gap_tol = 1e-7;
        const auto rec = sor::presolve::recover_solution(lp, out.map, rs, ro);
        const std::string where = "seed " + std::to_string(seed) +
            " doubletons " + std::to_string(doubletons) + " parallel " + std::to_string(parallels) +
            " primal " + std::to_string(rec.evidence.max_primal_violation) +
            " dual " + std::to_string(rec.evidence.max_dual_violation) +
            " gap " + std::to_string(rec.evidence.gap_rel);
        // Residuals, not `validated`: that also demands a finite safe dual
        // bound, which floating multipliers on free columns often do not
        // give (the same reason the default result is ProvedKKT).
        ::sor::test::report(rec.evidence.max_primal_violation <= 1e-7 &&
                                rec.evidence.max_dual_violation <= 1e-7,
                            "live-presolve lift satisfies primal and dual residuals",
                            __FILE__, __LINE__, where);
        ::sor::test::report(std::fabs(rec.raw.objective - ref.objective) <=
                                1e-7 * (1.0 + std::fabs(ref.objective)),
                            "lifted objective equals the direct optimum", __FILE__, __LINE__,
                            where + " lifted " + std::to_string(rec.raw.objective) +
                                " direct " + std::to_string(ref.objective));
        if (rs.has_basis) {
            std::set<Index> basic(rec.basis.basic.begin(), rec.basis.basic.end());
            ::sor::test::report(rec.basis.basic.size() == static_cast<std::size_t>(lp.n_rows()) &&
                                    basic.size() == rec.basis.basic.size() && !basic.count(-1),
                                "lifted basis has one distinct basic variable per row",
                                __FILE__, __LINE__, where);
            // Every nonbasic variable sits at the finite original bound its
            // status names, at the lifted point: otherwise the basis names a
            // different vertex than the lifted solution (or none at all).
            using PS = sor::presolve::PostsolveNonbasicStatus;
            const Index n = lp.n_cols();
            std::vector<double> ax(static_cast<std::size_t>(lp.n_rows()), 0.0);
            const auto& rp = lp.A.pattern.row_ptr();
            const auto& ci = lp.A.pattern.col_idx();
            for (Index i = 0; i < lp.n_rows(); ++i)
                for (auto k = rp[i]; k < rp[i + 1]; ++k)
                    ax[static_cast<std::size_t>(i)] += lp.A.vals[k] * rec.raw.x[ci[k]];
            bool consistent = rec.basis.status.size() == static_cast<std::size_t>(n + lp.n_rows());
            for (Index j = 0; consistent && j < n + lp.n_rows(); ++j) {
                const auto st = rec.basis.status[static_cast<std::size_t>(j)];
                if (st == PS::Basic) continue;
                const double lo = j < n ? lp.col_lo[j] : lp.row_lo[j - n];
                const double hi = j < n ? lp.col_hi[j] : lp.row_hi[j - n];
                const double v = j < n ? rec.raw.x[j] : ax[static_cast<std::size_t>(j - n)];
                const double b = st == PS::AtLower ? lo : st == PS::AtUpper ? hi : 0.0;
                consistent = std::isfinite(b) && std::fabs(v - b) <= 1e-7 * (1.0 + std::fabs(b));
            }
            ::sor::test::report(consistent,
                                "lifted nonbasic variables sit at their named finite bounds",
                                __FILE__, __LINE__, where);
            // And the lifted duals price every basic variable at zero, so the
            // basis and the multipliers describe the same optimal vertex.
            std::vector<double> d(lp.c.begin(), lp.c.end());
            for (Index i = 0; i < lp.n_rows(); ++i)
                for (auto k = rp[i]; k < rp[i + 1]; ++k)
                    d[static_cast<std::size_t>(ci[k])] -= lp.A.vals[k] * rec.raw.y[i];
            bool complementary = true;
            for (Index j = 0; j < n + lp.n_rows(); ++j) {
                if (rec.basis.status[static_cast<std::size_t>(j)] != PS::Basic) continue;
                const double dj = j < n ? d[static_cast<std::size_t>(j)] : rec.raw.y[j - n];
                if (std::fabs(dj) > 1e-7) complementary = false;
            }
            ::sor::test::report(complementary,
                                "lifted duals price the lifted basic variables at zero",
                                __FILE__, __LINE__, where);
        }
        if (!live) {
            ++default_lifts;
            default_forcing += forcing > 0;
            default_tightened += tightened > 0;
            continue;
        }
        lifted_with_doubleton += doubletons > 0;
        lifted_with_transfer += transfers > 0;
        lifted_with_tight_row += tight_rows > 0;
        lifted_with_parallel += parallels > 0;
        chains += doubletons > 1;
    }
    std::cout << "optimal " << optimal << ", lifts with doubletons " << lifted_with_doubleton
              << " (chains " << chains << ", bound transfers " << lifted_with_transfer
              << ", cost-tight rows " << lifted_with_tight_row << "), with parallel rows "
              << lifted_with_parallel << "\n";
    std::cout << "default rules: lifts " << default_lifts << " (forcing rows "
              << default_forcing << ", tightened singleton rows " << default_tightened << ")\n";
    CHECK(default_forcing >= 50);
    CHECK(default_tightened >= 20);
    CHECK(lifted_with_transfer >= 50);
    CHECK(lifted_with_tight_row >= 50);
    CHECK(lifted_with_doubleton >= 100);
    CHECK(lifted_with_parallel >= 50);
    CHECK(chains >= 20);
    return sor::test::finish("test_presolve_live_lift");
}
