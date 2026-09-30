// Independent-component solving (P12): a model whose rows split into
// disconnected parts is solved part by part and combined. Checked against the
// exhaustive oracle applied to each part (their sum is the optimum of the
// union), including an infeasible part, columns in no row, and a timing
// budget that reaches every part.
#include "sor/certify/finalize.hpp"
#include "sor/search/bab.hpp"
#include "sor/sparse/csr.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using sor::core::Status;
using sor::model::LpProblem;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::test::oracle::make_random_milp;
using sor::test::oracle::solve_oracle;

namespace {

LpProblem as_minimize(LpProblem p) {
    if (p.maximize) {
        for (auto& v : p.c) v = -v;
        p.obj_offset = -p.obj_offset;
        p.maximize = false;
    }
    return p;
}

// Block-diagonal union of parts, plus `extra_cols` columns in no row.
LpProblem union_of(const std::vector<LpProblem>& parts, int extra_cols,
                   const std::vector<double>& extra_cost) {
    LpProblem u;
    std::vector<sor::core::Index> rows, cols;
    std::vector<double> vals;
    sor::core::Index r0 = 0, c0 = 0;
    for (const auto& p : parts) {
        const auto& rp = p.A.pattern.row_ptr();
        const auto& ci = p.A.pattern.col_idx();
        for (sor::core::Index i = 0; i < p.n_rows(); ++i)
            for (auto k = rp[i]; k < rp[i + 1]; ++k) {
                rows.push_back(r0 + i);
                cols.push_back(c0 + ci[k]);
                vals.push_back(p.A.vals[k]);
            }
        for (sor::core::Index j = 0; j < p.n_cols(); ++j) {
            u.c.push_back(p.c[j]);
            u.col_lo.push_back(p.col_lo[j]);
            u.col_hi.push_back(p.col_hi[j]);
            u.is_integer.push_back(p.is_integer[j]);
        }
        for (sor::core::Index i = 0; i < p.n_rows(); ++i) {
            u.row_lo.push_back(p.row_lo[i]);
            u.row_hi.push_back(p.row_hi[i]);
        }
        u.obj_offset += p.obj_offset;
        r0 += p.n_rows();
        c0 += p.n_cols();
    }
    for (int e = 0; e < extra_cols; ++e) {
        u.c.push_back(extra_cost[static_cast<std::size_t>(e)]);
        u.col_lo.push_back(-2.0);
        u.col_hi.push_back(3.0);
        u.is_integer.push_back(e % 2 == 0);
    }
    u.A = sor::sparse::from_triplets(r0, c0 + extra_cols, rows, cols, vals);
    return u;
}

BabOptions options() {
    BabOptions o;
    o.para_bab.threads = 1;
    o.structural_presolve.enabled = false;
    o.mip_presolve = o.probing = o.symmetry = false;
    o.time_limit_s = 20.0;
    o.max_nodes = 20000;
    o.gap_tol = o.abs_gap_tol = 1e-9;
    return o;
}

void stress() {
    std::uint64_t used = 0, optimal = 0, infeasible = 0, models = 0;
    for (std::uint32_t seed = 1; seed <= 150; ++seed) {
        std::vector<LpProblem> parts;
        double expect = 0.0;
        bool feasible = true;
        const int n_parts = 2 + static_cast<int>(seed % 2);
        for (int q = 0; q < n_parts; ++q) {
            // Most models should be feasible (so Optimal is exercised); every
            // fifth keeps whatever the generator gives, infeasible parts included.
            std::uint32_t s2 = seed * 7 + static_cast<std::uint32_t>(q);
            auto part = as_minimize(make_random_milp(s2));
            auto o = solve_oracle(part);
            for (int tries = 0; seed % 5 != 0 && !o.feasible && tries < 60; ++tries) {
                s2 += 1000;
                part = as_minimize(make_random_milp(s2));
                o = solve_oracle(part);
            }
            if (!o.feasible) feasible = false;
            else expect += o.objective;
            parts.push_back(std::move(part));
        }
        // Isolated columns: minimise cost over the box [-2, 3] (integers on
        // even positions): the optimum takes the bound the cost points to.
        const std::vector<double> extra{1.0 + seed % 3, -1.0 - seed % 2, 0.0};
        const int n_extra = static_cast<int>(seed % 4);   // 0..3
        for (int e = 0; e < std::min(n_extra, 3); ++e) {
            const double c = extra[static_cast<std::size_t>(e)];
            expect += c > 0 ? -2.0 * c : (c < 0 ? 3.0 * c : 0.0);
        }
        auto lp = union_of(parts, std::min(n_extra, 3), extra);
        ++models;
        for (const bool comps : {true, false}) {
            auto o = options();
            o.component_solve = comps;
            BabDiagnostics d;
            auto raw = sor::search::solve_milp(lp, o, d);
            const auto res = sor::certify::finalize_result(
                std::move(raw), sor::search::milp_evidence(d, o));
            if (comps && d.component_count >= 2) ++used;
            if (!comps) CHECK(d.component_count == 0);
            if (!feasible) {
                CHECK(res.status != Status::Optimal);
                if (res.status == Status::Infeasible) { if (comps) ++infeasible; }
                continue;
            }
            if (res.status == Status::Optimal) {
                if (comps) ++optimal;
                const bool same = std::fabs(res.objective - expect) <=
                                  1e-6 * (1.0 + std::fabs(expect));
                if (!same)
                    std::cout << "WRONG seed=" << seed << " comps=" << comps << " got "
                              << res.objective << " want " << expect << '\n';
                CHECK(same);
            }
            // The bound may never pass the optimum.
            if (std::isfinite(d.dual_bound)) CHECK(d.dual_bound <= expect + 1e-6);
        }
    }
    std::cout << "COMPONENT_SOLVE models=" << models << " decomposed=" << used
              << " optimal=" << optimal << " infeasible=" << infeasible << '\n';
    CHECK(used > 50);
    CHECK(optimal > 30);
}

// A budget too small for the big part still yields an honest partial answer:
// never Optimal without every part proved.
void test_partial_is_not_optimal() {
    LpProblem a = as_minimize(make_random_milp(3)), b = as_minimize(make_random_milp(9));
    auto lp = union_of({a, b}, 0, {});
    auto o = options();
    o.max_nodes = 1;
    BabDiagnostics d;
    auto raw = sor::search::solve_milp(lp, o, d);
    if (raw.proposed_status == Status::Optimal) CHECK(d.globally_proved);
}

// The parent's proof is judged on the PARENT's objective. An offset that
// cancels the components' objectives makes the parent objective tiny while
// each child's absolute gap is large; whatever is claimed Optimal must still
// have a certified bound within the parent's own tolerance, and no reported
// bound may pass the optimum.
void test_parent_gap_uses_parent_objective() {
    std::uint64_t optimal = 0;
    for (std::uint32_t seed = 1; seed <= 400; ++seed) {
        LpProblem a = as_minimize(make_random_milp(seed)), b = as_minimize(make_random_milp(seed + 500));
        if (!solve_oracle(a).feasible || !solve_oracle(b).feasible) continue;
        for (auto* p : {&a, &b})
            for (auto& v : p->c) v *= 1e5;               // large child objectives
        auto lp = union_of({a, b}, 0, {});
        auto o = options();
        o.gap_tol = 1e-2;
        o.abs_gap_tol = 1e-6;
        // Make the parent objective small: cancel the components' optimum.
        auto probe = options();
        probe.component_solve = false;
        BabDiagnostics pd;
        auto pr = sor::search::solve_milp(lp, probe, pd);
        if (pr.proposed_status != Status::Optimal) continue;
        lp.obj_offset = -pr.objective + 1.0;
        BabDiagnostics d;
        auto raw = sor::search::solve_milp(lp, o, d);
        if (raw.proposed_status == Status::Optimal) {
            ++optimal;
            CHECK(std::isfinite(d.dual_bound));
            const double allowance = std::max(o.abs_gap_tol, o.gap_tol * (1.0 + std::fabs(raw.objective)));
            CHECK(std::fabs(raw.objective - d.dual_bound) <= allowance + 1e-9);
        }
        if (std::isfinite(d.dual_bound)) CHECK(d.dual_bound <= 1.0 + 1e-6);   // true optimum is 1
    }
    CHECK(optimal > 10);
}

}  // namespace

int main() {
    test_parent_gap_uses_parent_objective();
    stress();
    test_partial_is_not_optimal();
    return sor::test::finish("test_component_solve");
}
