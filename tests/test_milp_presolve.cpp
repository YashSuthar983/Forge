// Structural MIP presolve (forge/search/milp_presolve.hpp): fixed-column
// substitution, singleton-row bound tightening, and the postsolve round trip
// that must exactly undo both.
#include "forge/io/mps.hpp"
#include "forge/search/bab.hpp"
#include "forge/search/milp_presolve.hpp"
#include "forge/certify/finalize.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <chrono>
#include <functional>
#include <random>
#include <sstream>
#include <string>

using forge::core::Status;
using forge::core::Index;
using forge::core::f64;
using forge::search::MilpPresolveOptions;
using forge::search::MilpPresolveStats;
using forge::search::run_structural_presolve;
using forge::search::postsolve_point;

namespace {

forge::model::LpProblem read_text(const std::string& mps) {
    forge::io::MpsReadReport rep;
    std::istringstream in(mps);
    return forge::io::read_mps(in, rep);
}

void test_short_binary_row_support_fixes_and_postsolves() {
    // Activity bounds alone cannot see that 3x + 2y + 2z = 4 has only the
    // binary assignment (0,1,1). No search or whole-model probing is needed.
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {3., 2., 2.});
    lp.c = {-1., 1., 2.};
    lp.col_lo = {0., 0., 0.}; lp.col_hi = {1., 1., 1.};
    lp.is_integer = {1, 1, 1};
    lp.row_lo = lp.row_hi = {4.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    MilpPresolveStats stats;
    const auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(pre.reduced.n_cols() == 0);
    CHECK(stats.fixed_cols == 3);
    if (pre.reduced.n_cols() == 0) {
        const auto x = postsolve_point(pre, {});
        CHECK(x == std::vector<double>({0., 1., 1.}));
        CHECK_NEAR(pre.reduced.obj_offset, lp.objective(x), 1e-12);
        CHECK(lp.max_row_violation(x) <= 1e-9);
    }
}

void test_global_propagation_prepares_structural_support() {
    // The first row is not pure binary until global propagation proves
    // w=t=0 from the second row. Neither row is initially a singleton.
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(2, 5,
        {0,0,0,0,1,1}, {0,1,2,3,3,4}, {3.,2.,2.,1.,1.,1.});
    lp.c = {-1.,1.,2.,0.,0.};
    lp.col_lo.assign(5,0.); lp.col_hi.assign(5,1.);
    lp.is_integer = {1,1,1,0,0};
    lp.row_lo = {4.,-forge::model::kInf}; lp.row_hi = {4.,0.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    MilpPresolveStats stats;
    const auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(pre.reduced.n_cols() == 0);
    if (pre.reduced.n_cols() == 0) {
        const auto x = postsolve_point(pre, {});
        CHECK(x == std::vector<double>({0.,1.,1.,0.,0.}));
        CHECK(lp.max_row_violation(x) <= 1e-9);
        CHECK_NEAR(pre.reduced.objective({}),lp.objective(x),1e-12);
    }
    CHECK(stats.structural_fbbt_tightenings >= 2);
    CHECK(stats.structural_fbbt_work <= opts.structural_fbbt_max_work);
    opts.structural_fbbt_max_work = 0;
    const auto skipped = run_structural_presolve(lp, opts, stats);
    CHECK(skipped.reduced.n_cols() == 5);
    CHECK(stats.structural_fbbt_work == 0);
}

void test_structural_joint_assignments_detect_integer_infeasibility() {
    // Each row has two binary supports and FBBT cannot tighten any bound.
    // Jointly the three equalities force x=y=z=1/2, excluding every integer
    // point. Assignment propagation must inspect the other rows.
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(3,3,
        {0,0,1,1,2,2}, {0,1,0,2,1,2}, {1.,1.,1.,1.,1.,1.});
    lp.c.assign(3,1.); lp.col_lo.assign(3,0.); lp.col_hi.assign(3,1.);
    lp.is_integer.assign(3,1); lp.row_lo=lp.row_hi={1.,1.,1.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    MilpPresolveStats stats;
    const auto pre=run_structural_presolve(lp,opts,stats);
    CHECK(pre.infeasible);
}

void test_structural_binary_relations_postsolve() {
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(2,3,
        {0,0,1,1}, {0,2,1,2}, {1.,1.,1.,1.});
    lp.c={-1.,2.,3.};lp.col_lo.assign(3,0.);lp.col_hi.assign(3,1.);
    lp.is_integer.assign(3,1);lp.row_lo=lp.row_hi={1.,1.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    MilpPresolveStats stats;
    const auto pre=run_structural_presolve(lp,opts,stats);
    CHECK(!pre.infeasible);
    CHECK(pre.reduced.n_cols()==1);
    if(pre.reduced.n_cols()==1)for(double v:{0.,1.}) {
        const auto x=postsolve_point(pre,{v});
        CHECK(x[0]==x[1]);CHECK(x[0]+x[2]==1.);
        CHECK(lp.max_row_violation(x)<=1e-9);
        CHECK_NEAR(pre.reduced.objective({v}),lp.objective(x),1e-12);
    }
}

void test_conflict_components_become_reversible_relations() {
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(2, 3,
        {0,0,1,1}, {0,1,1,2}, {1.,-1.,1.,1.});
    lp.c = {2.,3.,5.};
    lp.col_lo = {0.,0.,0.}; lp.col_hi = {1.,1.,1.};
    lp.is_integer = {1,1,1};
    lp.row_lo = lp.row_hi = {0.,1.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    opts.structural_fbbt = false;
    opts.binary_row_support = false;
    opts.row_probe.enabled = false;
    opts.graph_relation_probe = true;
    opts.graph_probe.enabled = false;  // row-derived conflicts alone suffice
    MilpPresolveStats stats;
    const auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(pre.reduced.n_cols() == 1);
    CHECK(stats.graph_probe_relations >= 2);
    if (pre.reduced.n_cols() == 1) {
        for (double value : {0., 1.}) {
            const auto x = postsolve_point(pre, {value});
            CHECK(lp.max_row_violation(x) <= 1e-9);
            CHECK(x[0] == x[1] && x[1] == 1. - x[2]);
            CHECK_NEAR(lp.objective(x), pre.reduced.objective({value}), 1e-12);
        }
    }
}

void test_structural_relation_budgets_and_fixed_representative() {
    forge::model::LpProblem lp;
    lp.A=forge::sparse::from_triplets(2,2,{0,0,1,1},{0,1,0,1},{1.,-1.,1.,1.});
    lp.c={-1.,3.};lp.col_lo={0.,0.};lp.col_hi={1.,1.};lp.is_integer={1,1};
    lp.row_lo={0.,1.};lp.row_hi={0.,forge::model::kInf};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    opts.row_probe.max_row_visits_per_branch=2;
    MilpPresolveStats stats;
    auto pre=run_structural_presolve(lp,opts,stats);
    CHECK(!pre.infeasible);CHECK(pre.reduced.n_cols()==0);
    CHECK(stats.binary_substitutions>=1);
    if(pre.reduced.n_cols()==0) {
        CHECK(postsolve_point(pre,{})==std::vector<double>({1.,1.}));
        CHECK_NEAR(pre.reduced.objective({}),2.,1e-12);
    }
    opts.row_probe.max_total_row_visits=1;
    opts.structural_fbbt=false;opts.binary_row_support=false;
    pre=run_structural_presolve(lp,opts,stats);
    CHECK(!pre.infeasible);CHECK(pre.reduced.n_cols()==2);
    CHECK(pre.binary_substitution_steps.empty());
    CHECK(stats.row_probe_visits<=opts.row_probe.max_total_row_visits);
    CHECK(pre.reduced.col_lo==lp.col_lo);CHECK(pre.reduced.col_hi==lp.col_hi);
}

void test_structural_relations_reject_rounded_rewrites() {
    forge::model::LpProblem lp;
    lp.A=forge::sparse::from_triplets(2,2,{0,0,1,1},{0,1,0,1},{1.,-1.,1e16,1.});
    lp.c={0.,0.};lp.col_lo={0.,0.};lp.col_hi={1.,1.};lp.is_integer={1,1};
    lp.row_lo={0.,-forge::model::kInf};lp.row_hi={0.,2e16};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    // Isolates the substitution's numerical guard: coefficient strengthening
    // would drop the 1e16 row as activity-redundant before it is reached.
    opts.coefficient_strengthening=false;
    MilpPresolveStats stats;
    auto pre=run_structural_presolve(lp,opts,stats);
    CHECK(!pre.infeasible);CHECK(pre.reduced.n_cols()==2);
    CHECK(stats.binary_substitution_numerical_rejects>=1);
    CHECK(pre.binary_substitution_steps.empty());
    for(double v:{0.,1.})
        CHECK(postsolve_point(pre,{v,v})==std::vector<double>({v,v}));
    lp.A=forge::sparse::from_triplets(1,2,{0,0},{0,1},{1.,1.});
    lp.row_lo=lp.row_hi={1.};lp.c={1e16,1.};
    pre=run_structural_presolve(lp,opts,stats);
    CHECK(!pre.infeasible);CHECK(pre.reduced.n_cols()==2);
    CHECK(stats.binary_substitution_numerical_rejects>=1);
}

void test_joint_support_mixed_domain_retention() {
    // With one continuous column, enumerating the binary box and both ends
    // of its remaining continuous interval checks every linear-objective
    // extreme point. Check the mapping in both directions, including cost.
    auto vertices=[](const forge::model::LpProblem& p) {
        std::vector<int> binary;
        int continuous=-1;
        for(int j=0;j<p.n_cols();++j) {
            if(p.is_integer[j])binary.push_back(j);else continuous=j;
        }
        std::vector<std::vector<double>> result;
        const auto& rp=p.A.pattern.row_ptr();const auto& ci=p.A.pattern.col_idx();
        for(unsigned mask=0;mask<(1u<<binary.size());++mask) {
            std::vector<double> x(p.n_cols(),0.);
            for(std::size_t b=0;b<binary.size();++b)x[binary[b]]=(mask>>b)&1u;
            bool feasible=true;
            for(int j:binary)if(x[j]<p.col_lo[j]-1e-9 || x[j]>p.col_hi[j]+1e-9)feasible=false;
            double lo=continuous<0?0.:p.col_lo[continuous];
            double hi=continuous<0?0.:p.col_hi[continuous];
            for(int i=0;i<p.n_rows() && feasible;++i) {
                double activity=0.,a=0.;
                for(auto k=rp[i];k<rp[i+1];++k)
                    if(ci[k]==continuous)a=p.A.vals[k];else activity+=p.A.vals[k]*x[ci[k]];
                if(a==0.) {
                    if(activity<p.row_lo[i]-1e-9 || activity>p.row_hi[i]+1e-9)feasible=false;
                } else {
                    double rlo=(p.row_lo[i]-activity)/a,rhi=(p.row_hi[i]-activity)/a;
                    if(a<0.)std::swap(rlo,rhi);
                    lo=std::max(lo,rlo);hi=std::min(hi,rhi);
                }
            }
            if(!feasible || lo>hi+1e-9)continue;
            if(continuous>=0)x[continuous]=lo;
            result.push_back(x);
            if(continuous>=0 && hi>lo) {x[continuous]=hi;result.push_back(x);}
        }
        return result;
    };
    std::mt19937 rng(20260929u);
    for(int example=0;example<60;++example) {
        forge::model::LpProblem lp;
        std::vector<forge::core::Index> ri,ci;
        std::vector<double> av;
        const unsigned planted=rng()%16;
        const double continuous=.25*(1+rng()%3);
        for(int i=0;i<5;++i) {
            double activity=0.;
            for(int j=0;j<5;++j) {
                double a=static_cast<int>(rng()%7)-3;
                if(j==4 && i<2)a=0.;
                if(a==0.)continue;
                ri.push_back(i);ci.push_back(j);av.push_back(a);
                activity+=a*(j==4?continuous:((planted>>j)&1u));
            }
            lp.row_lo.push_back(activity-rng()%3);lp.row_hi.push_back(activity+rng()%3);
        }
        lp.A=forge::sparse::from_triplets(5,5,ri,ci,av);
        lp.col_lo.assign(5,0.);lp.col_hi.assign(5,1.);lp.is_integer={1,1,1,1,0};
        lp.maximize=example%2;lp.obj_offset=.5;
        for(int j=0;j<5;++j)lp.c.push_back(static_cast<int>(rng()%9)-4);
        MilpPresolveOptions opts;
        opts.monotone_binary_pairs = false;  // this oracle requires EVERY feasible point
        opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
        opts.row_probe.max_row_visits_per_branch=example%2?2:10000;
        opts.row_probe.max_row_overlap=example%3?1.0:0.5;
        opts.graph_relation_probe = example % 2 == 0;
        opts.graph_support_propagation = example % 3 == 0;
        opts.graph_probe.max_binaries_probed = 4;
        MilpPresolveStats stats;
        const auto pre=run_structural_presolve(lp,opts,stats);
        CHECK(!pre.infeasible);if(pre.infeasible)continue;
        CHECK(stats.row_probe_visits<=opts.row_probe.max_total_row_visits);
        for(const auto& x:vertices(lp)) {
            std::vector<double> y(pre.reduced.n_cols());
            for(int j=0;j<5;++j)if(pre.reduced_col[j]>=0)y[pre.reduced_col[j]]=x[j];
            const auto lifted=postsolve_point(pre,y);
            CHECK(pre.reduced.max_row_violation(y)<=1e-8);
            CHECK(pre.reduced.max_bound_violation(y)<=1e-8);
            for(int j=0;j<5;++j)CHECK_NEAR(lifted[j],x[j],1e-8);
            CHECK_NEAR(pre.reduced.objective(y),lp.objective(x),1e-8);
        }
        for(const auto& y:vertices(pre.reduced)) {
            const auto x=postsolve_point(pre,y);
            CHECK(lp.max_row_violation(x)<=1e-8);CHECK(lp.max_bound_violation(x)<=1e-8);
            CHECK_NEAR(lp.objective(x),pre.reduced.objective(y),1e-8);
        }
    }
}

void test_monotone_pairs_preserve_optimum_and_postsolve() {
    // Exhaustive integer boxes and scalar continuous vertices, independent of B&B. Vary
    // sense, costs, pair scaling/sign, and competing row sides. Checking only
    // a returned incumbent would miss deletion of a better original point.
    auto points = [](const forge::model::LpProblem& p) {
        std::vector<std::vector<double>> out;
        std::vector<double> x(p.n_cols());
        std::function<void(int)> visit = [&](int j) {
            if (j == p.n_cols()) {
                if (p.max_row_violation(x) <= 1e-10) out.push_back(x);
                return;
            }
            const bool integer = p.is_integer[j];
            for (double v = integer ? std::ceil(p.col_lo[j]) : p.col_lo[j];
                 v <= p.col_hi[j]; v += integer ? 1. : .5) {
                x[j] = v;
                visit(j + 1);
            }
        };
        visit(0);
        return out;
    };
    std::mt19937 rng(41093);
    int saturated = 0;
    for (int example = 0; example < 240; ++example) {
        forge::model::LpProblem lp;
        const double a = example % 2 ? -2. : 2.;
        std::vector<forge::core::Index> ri{0, 0}, ci{0, 1};
        std::vector<double> av{a, a};
        lp.row_lo = {a < 0. ? a : -forge::model::kInf};
        lp.row_hi = {a > 0. ? a : forge::model::kInf};
        for (int i = 1; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const double coef = j < 2 && example % 3 == 0 ? -2. :
                    static_cast<int>(rng() % 5) - 2.;
                if (coef == 0.) continue;
                ri.push_back(i); ci.push_back(j); av.push_back(coef);
            }
            lp.row_lo.push_back(example % 3 == 0 ? -forge::model::kInf : -2.);
            lp.row_hi.push_back(2. + rng() % 3);
        }
        lp.A = forge::sparse::from_triplets(4, 4, ri, ci, av);
        lp.col_lo = {0., 0., 0., 0.}; lp.col_hi = {1., 1., 2., 2.};
        lp.is_integer = {1, 1, example % 2 != 0, 1}; lp.maximize = example % 2;
        lp.obj_offset = .5;
        const double good_cost = lp.maximize ? 1. : -1.;
        lp.c = {example % 3 == 0 ? good_cost : static_cast<int>(rng() % 5) - 2.,
                0., 1., -1.};
        MilpPresolveOptions opts;
        opts.row_probe.enabled = false; opts.probing_presolve = false;
        opts.binary_row_support = false; opts.structural_fbbt = false;
        opts.coefficient_strengthening = false; opts.merge_duplicate_columns = false;
        MilpPresolveStats stats;
        const auto pre = run_structural_presolve(lp, opts, stats);
        saturated += stats.monotone_pairs_saturated;
        const auto original = points(lp);
        CHECK(!pre.infeasible || original.empty());
        if (pre.infeasible) continue;
        const auto reduced = points(pre.reduced);
        CHECK(original.empty() == reduced.empty());
        const double sense = lp.maximize ? -1. : 1.;
        double before = forge::model::kInf, after = forge::model::kInf;
        for (const auto& x : original) before = std::min(before, sense * lp.objective(x));
        for (const auto& y : reduced) {
            const auto x = postsolve_point(pre, y);
            CHECK(lp.max_row_violation(x) <= 1e-10);
            CHECK(lp.max_bound_violation(x) <= 1e-10);
            CHECK_NEAR(lp.objective(x), pre.reduced.objective(y), 1e-12);
            after = std::min(after, sense * pre.reduced.objective(y));
        }
        if (!original.empty()) CHECK_NEAR(before, after, 1e-12);
    }
    CHECK(saturated >= 80);
    // Opt-out retains (0,0); an objective-worsening fill must also be refused.
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1., 1.});
    lp.col_lo = {0., 0.}; lp.col_hi = {1., 1.}; lp.is_integer = {1, 1};
    lp.row_lo = {-forge::model::kInf}; lp.row_hi = {1.}; lp.c = {1., 1.};
    MilpPresolveOptions opts;
    opts.row_probe.enabled = false; opts.probing_presolve = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(stats.monotone_pairs_saturated == 0);
    const std::vector<double> reduced_zero(pre.reduced.n_cols(), 0.);
    CHECK(pre.reduced.max_row_violation(reduced_zero) == 0.);
    const auto original_zero = postsolve_point(pre, reduced_zero);
    CHECK(lp.max_row_violation(original_zero) == 0.);
    CHECK(lp.max_bound_violation(original_zero) == 0.);
    CHECK_NEAR(lp.objective(original_zero), 0., 0.);
    opts.monotone_binary_pairs = false; lp.c = {0., 0.};
    opts.merge_duplicate_columns = false;
    pre = run_structural_presolve(lp, opts, stats);
    CHECK(stats.monotone_pairs_saturated == 0);
    CHECK(pre.reduced.n_cols() == 2);
}

void test_short_binary_rows_retain_every_feasible_assignment() {
    std::mt19937 rng(20260928u);
    for (int example = 0; example < 80; ++example) {
        constexpr int n = 6, m = 4;
        const unsigned planted = rng() % (1u << n);
        forge::model::LpProblem lp;
        std::vector<forge::core::Index> rows, cols;
        std::vector<double> vals;
        for (int i = 0; i < m; ++i) {
            double activity = 0.;
            for (int j = 0; j < n; ++j) {
                const double a = static_cast<int>(rng() % 11) - 5;
                if (a == 0.) continue;
                rows.push_back(i); cols.push_back(j); vals.push_back(a);
                activity += a * ((planted >> j) & 1u);
            }
            lp.row_lo.push_back(activity - rng() % 3);
            lp.row_hi.push_back(activity + rng() % 3);
        }
        lp.A = forge::sparse::from_triplets(m, n, rows, cols, vals);
        lp.col_lo.assign(n, 0.); lp.col_hi.assign(n, 1.);
        lp.is_integer.assign(n, 1); lp.maximize = (example % 2) != 0;
        for (int j = 0; j < n; ++j) lp.c.push_back(static_cast<int>(rng() % 9) - 4);
        MilpPresolveOptions opts;
        opts.monotone_binary_pairs = false;  // isolate primal support retention
        opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
        MilpPresolveStats stats;
        const auto pre = run_structural_presolve(lp, opts, stats);
        CHECK(!pre.infeasible);
        if (pre.infeasible) continue;
        for (unsigned mask = 0; mask < (1u << n); ++mask) {
            std::vector<double> x(n);
            for (int j = 0; j < n; ++j) x[j] = (mask >> j) & 1u;
            if (lp.max_row_violation(x) > 1e-9) continue;
            std::vector<double> reduced(pre.reduced.n_cols());
            for (int j = 0; j < n; ++j)
                if (pre.reduced_col[j] >= 0) reduced[pre.reduced_col[j]] = x[j];
            CHECK(pre.reduced.max_row_violation(reduced) <= 1e-9);
            CHECK(pre.reduced.max_bound_violation(reduced) <= 1e-9);
            CHECK(postsolve_point(pre, reduced) == x);
            CHECK_NEAR(pre.reduced.objective(reduced), lp.objective(x), 1e-12);
        }
    }
}

void test_binary_row_support_budget_and_continuous_guard() {
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1., 2.});
    lp.c = {1., 1.}; lp.col_lo = {0., 0.}; lp.col_hi = {1., 1.};
    lp.is_integer = {1, 0}; lp.row_lo = lp.row_hi = {1.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    opts.structural_fbbt = false;  // isolate the binary support budget/guard
    opts.row_probe.enabled = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(pre.reduced.n_cols() == 2);
    CHECK(postsolve_point(pre, {0., .5}) == std::vector<double>({0., .5}));
    lp.is_integer = {1, 1};
    opts.binary_row_max_work = 0;
    pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(pre.reduced.n_cols() == 2);
    CHECK(stats.binary_row_fixings == 0);
    CHECK(stats.binary_row_work == 0);
    opts.binary_row_max_work = 1000;
    lp.row_lo = lp.row_hi = {1. + 5e-10};
    pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);  // outward numerical relaxation admits (1,0)
    lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {6., 4.});
    lp.row_lo = lp.row_hi = {3.};
    pre = run_structural_presolve(lp, opts, stats);
    CHECK(pre.infeasible);
}

// x1 fixed at 3 (LO == UP). Row R1: x1 + x2 <= 10 folds to x2 <= 7.
// min x2  =>  optimum x2 = -inf in principle, so bound it: 0 <= x2 <= 7.
const char* kFixedColumn = R"(NAME          FIXEDCOL
ROWS
 N  OBJ
 L  R1
COLUMNS
    X1        OBJ       0              R1        1
    X2        OBJ       1              R1        1
RHS
    RHS       R1        10
BOUNDS
 FX BND       X1        3
 LO BND       X2        0
ENDATA
)";

// Exactness of structural presolve (audit section 2). Each model is the
// audit's hand-derived counterexample; every number in it is a double.
forge::model::LpProblem exact_model(std::vector<double> c, std::vector<double> lo,
                                  std::vector<double> hi, std::vector<char> integer,
                                  std::vector<double> row_lo, std::vector<double> row_hi,
                                  std::vector<forge::core::Index> ri,
                                  std::vector<forge::core::Index> rj,
                                  std::vector<double> rv) {
    forge::model::LpProblem lp;
    lp.c = std::move(c);
    lp.col_lo = std::move(lo);
    lp.col_hi = std::move(hi);
    lp.is_integer.assign(integer.begin(), integer.end());
    lp.row_lo = std::move(row_lo);
    lp.row_hi = std::move(row_hi);
    lp.A = forge::sparse::from_triplets(static_cast<forge::core::Index>(lp.row_lo.size()),
                                      static_cast<forge::core::Index>(lp.c.size()), ri, rj, rv);
    lp.validate();
    return lp;
}

// The optimum of the reduced model when it has no rows: every column at its
// cheaper bound; postsolved and priced in the original model.
double box_optimum_after_presolve(const forge::model::LpProblem& lp) {
    MilpPresolveOptions opts;
    opts.probing_presolve = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(pre.reduced.n_rows() == 0);
    std::vector<double> xr(static_cast<std::size_t>(pre.reduced.n_cols()));
    for (std::size_t k = 0; k < xr.size(); ++k)
        xr[k] = pre.reduced.c[k] >= 0.0 ? pre.reduced.col_lo[k] : pre.reduced.col_hi[k];
    const auto x = postsolve_point(pre, xr);
    CHECK(lp.max_bound_violation(x) == 0.0);
    return lp.objective(x);
}

void test_narrow_continuous_column_keeps_its_optimum() {
    // x in [0, 2^-40] with cost -2^40: optimum -1 at the upper bound. Fixing
    // the narrow column at its midpoint gave -1/2.
    const auto lp = exact_model({-0x1p40, 0.0}, {0.0, 0.0}, {0x1p-40, 0.0}, {0, 1},
                                {}, {}, {}, {}, {});
    CHECK(box_optimum_after_presolve(lp) == -1.0);
}

void test_fixed_column_substitution_is_exact() {
    // p, r, q fixed to 1, M = 2^54: M p - M r + q = 1 holds exactly. A
    // rounded running subtraction of the sides left 0 = -1 (infeasible).
    const double M = 0x1p54;
    const auto lp = exact_model({0.0, 0.0, 0.0, 1.0}, {1.0, 1.0, 1.0, 0.0},
                                {1.0, 1.0, 1.0, 5.0}, {1, 1, 1, 0},
                                {1.0, 0.0}, {1.0, forge::model::kInf},
                                {0, 0, 0, 1}, {0, 1, 2, 3}, {M, -M, 1.0, 1.0});
    MilpPresolveOptions opts;
    opts.probing_presolve = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
}

void test_duplicate_column_merge_needs_an_exact_domain() {
    // Integer x in [0, 2^53], y in [0, 1], identical columns and costs: the
    // merged domain [0, 2^53 + 1] is not a double. Merging into [0, 2^53]
    // loses (2^53, 1), the optimum (-2^53 - 1).
    // x + y - z = 0 with z continuous in [0, 2^54] keeps the columns
    // nonempty and the row from being redundant.
    const auto lp = exact_model({-1.0, -1.0, 0.0}, {0.0, 0.0, 0.0}, {0x1p53, 1.0, 0x1p54},
                                {1, 1, 0}, {0.0}, {0.0}, {0, 0, 0}, {0, 1, 2},
                                {1.0, 1.0, -1.0});
    MilpPresolveOptions opts;
    opts.probing_presolve = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(stats.merged_cols == 0);
    CHECK(stats.merge_inexact_skipped >= 1);
    // A pair whose domain sum is exact still merges.
    const auto lp2 = exact_model({-1.0, -1.0, 0.0}, {0.0, 0.0, 0.0}, {4.0, 1.0, 16.0},
                                 {1, 1, 0}, {0.0}, {0.0}, {0, 0, 0}, {0, 1, 2},
                                 {1.0, 1.0, -1.0});
    MilpPresolveStats stats2;
    auto pre2 = run_structural_presolve(lp2, opts, stats2);
    CHECK(stats2.merged_cols == 1);
}

void test_singleton_row_bound_is_rounded_outward() {
    // 3 x <= 1 on continuous x: the bound is 1/3, whose nearest double lies
    // below it. Rounded to nearest the new bound excluded x = 1/3's
    // neighbourhood exactly at the side; it must round up.
    const auto lp = exact_model({-1.0, 1.0}, {0.0, 0.0}, {10.0, 1.0}, {0, 1},
                                {-forge::model::kInf}, {1.0}, {0}, {0}, {3.0});
    MilpPresolveOptions opts;
    opts.probing_presolve = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    const auto k = pre.reduced_col[0];
    CHECK(k >= 0);
    if (k >= 0) CHECK(pre.reduced.col_hi[static_cast<std::size_t>(k)] > 1.0 / 3.0);
}

void test_fixed_column_eliminated_and_folded_into_row() {
    auto lp = read_text(kFixedColumn);
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(stats.fixed_cols == 1);
    CHECK(pre.reduced.n_cols() == 1);
    // Folding X1 out of R1 leaves x2 <= 7, which is itself now a singleton
    // row: the fixed-point loop folds THAT into x2's own bound in the same
    // pass, so the row disappears too, not just X1's column.
    CHECK(pre.reduced.n_rows() == 0);
    CHECK(stats.singleton_rows == 1);
    CHECK_NEAR(pre.reduced.col_hi[0], 7.0, 1e-12);

    // Round trip: a feasible reduced point expands to a feasible original
    // point with X1 exactly at its fixed value.
    const std::vector<forge::core::f64> reduced_x = {4.0};  // x2 = 4
    auto x = postsolve_point(pre, reduced_x);
    CHECK(x.size() == 2);
    CHECK_NEAR(x[0], 3.0, 1e-12);
    CHECK_NEAR(x[1], 4.0, 1e-12);
    CHECK(lp.max_row_violation(x) <= 1e-9);
    CHECK(lp.max_bound_violation(x) <= 1e-9);
}

// A singleton row 3 x1 <= 9 tightens x1's upper bound from 100 to 3, and the
// row itself is then redundant and dropped.
const char* kSingletonRow = R"(NAME          SINGLETON
ROWS
 N  OBJ
 L  R1
 G  R2
COLUMNS
    X1        OBJ       -1             R1        3
    X1        R2        1
    X2        OBJ       1              R2        1
RHS
    RHS       R1        9              R2        0
BOUNDS
 UP BND       X1        100
 UP BND       X2        100
ENDATA
)";

// Regression: bound propagation closes x0 onto 4 only asymptotically
// ([3.99999999906, 4] after 20 sweeps). Fixing a "nearly fixed" continuous
// column at its lower bound rewrote the other rows with a value off by ~1e-9
// times their coefficients and the presolve declared a feasible model
// infeasible. Must stay feasible with the unique optimum intact.
void test_asymptotic_bounds_are_not_fixed() {
    // max 3 x2 - x3 - 2 x4 + 5 x5 + x1 (x0 free of cost); x0,x1 continuous.
    forge::model::LpProblem lp;
    lp.maximize = true;
    lp.obj_offset = 0.0;
    lp.c = {0.0, 1.0, 3.0, -1.0, -2.0, 5.0};
    lp.col_lo = {0., 0., 0., 0., 0., 0.};
    lp.col_hi = {5., 2., 1., 2., 2., 1.};
    lp.is_integer = {0, 0, 1, 1, 1, 1};
    lp.row_lo = {-9., -3., 3., -forge::model::kInf};
    lp.row_hi = {-9., forge::model::kInf, 3., -10.};
    lp.A = forge::sparse::from_triplets(
        4, 6,
        {0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3},
        {0, 1, 3, 4, 5, 0, 1, 5, 2, 3, 4, 0, 1},
        {1., -5., -2., 3., -1., 5., -4., -2., -2., 5., -4., -3., 1.});
    lp.validate();
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    // The point x = (4, 2, 1, 1, 0, 1) is feasible for the ORIGINAL rows and
    // must survive (postsolve of any reduced point that fixes the same
    // integers reproduces it).
    const std::vector<double> x = {4., 2., 1., 1., 0., 1.};
    CHECK(lp.max_row_violation(x) <= 1e-9);
    CHECK(lp.max_bound_violation(x) <= 1e-9);
}

void test_singleton_row_tightens_bound_and_is_dropped() {
    auto lp = read_text(kSingletonRow);
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    // R2 (x1 + x2 >= 0, columns >= 0) is activity-redundant too; isolate the
    // singleton-row behaviour under test.
    opts.coefficient_strengthening = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(stats.singleton_rows == 1);
    CHECK(stats.redundant_rows == 1);
    // R1 (the singleton) is gone; R2 (x1 + x2 >= 0, still two live columns)
    // survives, so one row and both columns remain.
    CHECK(pre.reduced.n_rows() == 1);
    CHECK(pre.reduced.n_cols() == 2);
    const auto j1 = pre.reduced_col[0];
    CHECK(j1 >= 0);
    CHECK_NEAR(pre.reduced.col_hi[static_cast<std::size_t>(j1)], 3.0, 1e-12);
}

// Two singleton rows pin the same integer column to disjoint ranges: the
// fixed-point loop must catch the resulting infeasibility itself, not hand a
// broken reduced problem to the solver.
const char* kSingletonInfeasible = R"(NAME          SINGLETONINFEAS
ROWS
 N  OBJ
 G  R1
 L  R2
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        OBJ       1              R1        1
    X1        R2        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        5              R2        2
BOUNDS
 UP BND       X1        10
ENDATA
)";

void test_conflicting_singleton_rows_detected_infeasible() {
    auto lp = read_text(kSingletonInfeasible);
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(pre.infeasible);
    CHECK(stats.infeasible);
}

// A chain: fixing x1 makes row R1 a singleton in x2, tightening x2 down to a
// fixed value, which folds into R2 leaving a singleton in x3, and so on --
// the fixed-point loop, not a single pass, is what fully reduces this.
const char* kChainedElimination = R"(NAME          CHAIN
ROWS
 N  OBJ
 E  R1
 E  R2
 L  R3
COLUMNS
    X1        OBJ       0              R1        1
    X2        OBJ       0              R1        1
    X2        R2        1
    X3        OBJ       1              R2        1
    X3        R3        1
    MARK0000  'MARKER'                 'INTORG'
    X4        OBJ       1              R3        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        5              R2        5
    RHS       R3        20
BOUNDS
 FX BND       X1        5
 UP BND       X3        100
 UP BND       X4        100
ENDATA
)";

void test_chained_elimination_reaches_fixed_point() {
    auto lp = read_text(kChainedElimination);
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    opts.aggregation = false;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    // X1 fixed (given). R1: X1+X2=5 => X2 fixed at 0 (singleton after X1
    // drops out). R2: X2+X3=5 => X3 fixed at 5 (singleton after X2 drops
    // out). R3: X3+X4<=20 => X4<=15 (singleton after X3 drops out), and
    // since X4 has no other row, that fold empties the model down to a
    // single free column and zero rows.
    CHECK(stats.fixed_cols == 3);
    CHECK(pre.reduced.n_cols() == 1);
    CHECK(pre.reduced.n_rows() == 0);
    CHECK_NEAR(pre.reduced.col_hi[0], 15.0, 1e-9);

    const std::vector<forge::core::f64> reduced_x = {10.0};
    auto x = postsolve_point(pre, reduced_x);
    CHECK(x.size() == 4);
    CHECK_NEAR(x[0], 5.0, 1e-9);
    CHECK_NEAR(x[1], 0.0, 1e-9);
    CHECK_NEAR(x[2], 5.0, 1e-9);
    CHECK_NEAR(x[3], 10.0, 1e-9);
    CHECK(lp.max_row_violation(x) <= 1e-9);
}

// End-to-end: solving the reduced problem and postsolving must give the same
// objective as solving the original directly (the reduction is exact, not a
// heuristic relaxation).
void test_presolved_solve_matches_direct_solve() {
    auto lp = read_text(kChainedElimination);
    // X1..X3 are pinned by the chain; only X4 is free, minimizing X3 + X4.
    forge::search::BabOptions opts;
    opts.structural_presolve.enabled = false;
    forge::search::BabDiagnostics diag_direct;
    auto raw_direct = forge::search::solve_milp(lp, opts, diag_direct);
    const auto ev_direct = forge::search::milp_evidence(diag_direct, opts);
    const auto r_direct =
        forge::certify::finalize_result(std::move(raw_direct), ev_direct);

    MilpPresolveOptions pre_opts;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, pre_opts, stats);
    CHECK(!pre.infeasible);
    forge::search::BabDiagnostics diag_reduced;
    auto raw_reduced = forge::search::solve_milp(pre.reduced, opts, diag_reduced);
    const auto ev_reduced = forge::search::milp_evidence(diag_reduced, opts);
    const auto r_reduced =
        forge::certify::finalize_result(std::move(raw_reduced), ev_reduced);

    CHECK(r_direct.status == Status::Optimal);
    CHECK(r_reduced.status == Status::Optimal);
    CHECK_NEAR(r_direct.objective, r_reduced.objective, 1e-7);

    auto x = postsolve_point(pre, r_reduced.x);
    CHECK(lp.max_row_violation(x) <= 1e-7);
    CHECK(lp.max_bound_violation(x) <= 1e-7);
    CHECK_NEAR(lp.objective(x), r_direct.objective, 1e-7);
}


void test_wrapper_total_time_includes_presolve() {
    forge::model::LpProblem lp;
    constexpr forge::core::Index fixed = 30000;
    lp.A = forge::sparse::from_triplets(0, fixed + 1, {}, {}, {});
    lp.c.assign(fixed + 1, 0.0);
    lp.c.back() = 1.0;
    lp.col_lo.assign(fixed + 1, 0.0);
    lp.col_hi.assign(fixed + 1, 0.0);
    lp.col_hi.back() = 1.0;
    lp.is_integer.assign(fixed + 1, true);
    forge::search::BabOptions opts;
    opts.structural_presolve.enabled = true;
    opts.para_bab.threads = 1;
    opts.time_limit_s = 10.0;
    opts.feasibility_jump = opts.fixprop = opts.rounding_heuristic = false;
    opts.sub_mip_lns = false;
    opts.balans.enabled = opts.kernel_pump.enabled = opts.mrens.enabled = false;
    forge::search::BabDiagnostics diag;
    const auto started = std::chrono::steady_clock::now();
    auto raw = forge::search::solve_milp(lp, opts, diag);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    CHECK(raw.proposed_status == Status::Optimal);
    CHECK(raw.x.size() == static_cast<std::size_t>(fixed + 1));
    CHECK(diag.structural_presolve.fixed_cols == fixed);
    CHECK(diag.total_ms >= diag.structural_presolve.ms);
    CHECK(wall <= 10.5);

    opts.time_limit_s = 1e-12;
    forge::search::BabDiagnostics timed;
    const auto interrupted = forge::search::solve_milp(lp, opts, timed);
    CHECK(interrupted.proposed_status == Status::Interrupted);
    CHECK(interrupted.termination_reason == "time limit exhausted during structural presolve");
    CHECK(timed.lp_solves == 0);
    CHECK(timed.total_ms >= timed.structural_presolve.ms);
}

void test_wrapper_projects_cut_reference_point() {
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, 3, {0, 0}, {1, 2}, {2.0, 3.0});
    lp.row_lo = {2.5};
    lp.row_hi = {forge::model::kInf};
    lp.c = {0.0, 1.0, 1.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {0.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};
    const std::vector<double> reference{0.0, 0.0, 1.0};
    forge::search::BabOptions opts;
    opts.structural_presolve.enabled = true;
    // This component test needs a fractional root so cuts consume the
    // projected reference. Binary support would force X2=1 before any cuts;
    // support/postsolve correctness is tested independently above.
    opts.structural_presolve.binary_row_support = false;
    opts.structural_presolve.structural_fbbt = false;
    opts.structural_presolve.row_probe.enabled = false;
    // Strengthening 2x1+3x2>=2.5 to 2x1+2.5x2>=2.5 makes the root LP
    // integral, leaving no cut to project the reference through.
    opts.structural_presolve.coefficient_strengthening = false;
    // Likewise rounding its side to 3 (gcd 1 over integer data).
    opts.structural_presolve.side_rounding = false;
    opts.para_bab.threads = 1;
    opts.lp.presolve = false;
    opts.mip_presolve = opts.probing = opts.symmetry = false;
    opts.integer_row_rounding = false;
    opts.cut_reference_point = &reference;
    forge::search::BabDiagnostics diag;
    auto raw = forge::search::solve_milp(lp, opts, diag);
    CHECK(diag.structural_presolve.fixed_cols == 1);
    CHECK(diag.gmi_cuts_added + diag.mir_cuts_added > 0);
    CHECK(diag.invalid_cuts_detected == 0);
    CHECK(diag.node_cuts_invalid_inserted == 0);
    CHECK_NEAR(raw.objective, 1.0, 1e-6);
    CHECK(forge::search::milp_point_max_violation(lp, raw.x, opts.int_tol) <= opts.primal_feas_tol);
}

void test_wrapper_postsolves_a_zero_column_optimum() {
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(0, 1, {}, {}, {});
    lp.c = {4.0};
    lp.col_lo = lp.col_hi = {3.0};
    lp.is_integer = {true};
    forge::search::BabOptions opts;
    opts.structural_presolve.enabled = true;
    opts.para_bab.threads = 1;
    forge::search::BabDiagnostics diag;
    const auto raw = forge::search::solve_milp(lp, opts, diag);
    CHECK(raw.proposed_status == Status::Optimal);
    CHECK_NEAR(raw.objective, 12.0, 1e-9);
    CHECK(raw.x.size() == 1);
    if (raw.x.size() == 1) CHECK_NEAR(raw.x[0], 3.0, 1e-9);
    CHECK(forge::search::milp_point_max_violation(lp, raw.x, opts.int_tol) <= opts.primal_feas_tol);
}

void test_wrapper_keeps_portfolio_points_in_original_space() {
    // The first column is fixed; the 16 surviving binaries need a tree.
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> value(1000, 9999), weight(10, 99);
    forge::model::LpProblem lp;
    lp.maximize = true;
    lp.c.assign(17, 0.0);
    lp.col_lo.assign(17, 0.0);
    lp.col_hi.assign(17, 1.0);
    lp.col_hi[0] = 0.0;
    lp.is_integer.assign(17, true);
    for (int j = 1; j <= 16; ++j) lp.c[j] = value(rng) / 1000.0;
    std::vector<forge::core::Index> rows, cols;
    std::vector<double> vals;
    lp.row_lo.assign(5, -forge::model::kInf);
    lp.row_hi.assign(5, 0.0);
    for (int i = 0; i < 5; ++i)
        for (int j = 1; j <= 16; ++j) {
            const double w = weight(rng);
            rows.push_back(i); cols.push_back(j); vals.push_back(w);
            lp.row_hi[i] += 0.5 * w;
        }
    lp.A = forge::sparse::from_triplets(5, 17, rows, cols, vals);
    forge::search::BabOptions opts;
    opts.structural_presolve.enabled = true;
    opts.time_limit_s = 10.0;
    opts.para_bab.threads = 1;
    opts.cuts_enabled = false;
    opts.mip_presolve = opts.probing = opts.symmetry = false;
    opts.integer_row_rounding = false;
    opts.sub_mip_lns = false;
    opts.balans.enabled = opts.kernel_pump.enabled = opts.mrens.enabled = false;
    opts.gap_tol = 0.005;
    forge::search::PortfolioPool pool(true);
    opts.pool = &pool;
    forge::search::BabDiagnostics diag;
    auto raw = forge::search::solve_milp(lp, opts, diag);
    CHECK(diag.nodes > 1);
    CHECK(raw.x.size() == 17);
    const auto shared = pool.best_point();
    CHECK(shared.empty() || forge::search::portfolio_point_is_feasible(lp, shared,
          opts.primal_feas_tol, opts.int_tol));
    CHECK(forge::search::milp_point_max_violation(lp, raw.x, opts.int_tol) <= opts.primal_feas_tol);

    opts.pool = nullptr;
    forge::search::PortfolioOptions portfolio;
    portfolio.workers = 2;
    portfolio.ramp_seconds = 0.0;
    forge::search::PortfolioDiagnostics portfolio_diag;
    forge::search::BabDiagnostics winner;
    const auto raced = forge::search::solve_milp_portfolio(lp, opts, portfolio, winner, portfolio_diag);
    CHECK(portfolio_diag.workers_used == 2);
    CHECK(raced.x.size() == 17);
    CHECK(forge::search::portfolio_point_is_feasible(lp, raced.x, opts.primal_feas_tol, opts.int_tol));
}

// Presolve probing: z = 0 forces p = q = 1, which p + q <= 1 forbids, so only
// probing (not one row at a time) proves z = 1. The fixing goes back through
// the reductions: z disappears, and the two rows it satisfied are dropped.
void test_probing_fixes_and_shrinks() {
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(3, 3, {0, 0, 1, 1, 2, 2}, {2, 0, 2, 1, 0, 1},
                                      {1., 1., 1., 1., 1., 1.});   // cols: p q z
    lp.row_lo = {1., 1., -forge::model::kInf};
    lp.row_hi = {forge::model::kInf, forge::model::kInf, 1.};
    lp.c = {1., 2., 3.};
    lp.col_lo.assign(3, 0.); lp.col_hi.assign(3, 1.);
    lp.is_integer = {1, 1, 1};
    MilpPresolveOptions off;
    off.probing_presolve = false;
    off.row_probe.enabled = false;      // the joint row-support probe would find it too
    off.binary_row_max_cols = 0;
    MilpPresolveStats s0;
    const auto without = run_structural_presolve(lp, off, s0);
    CHECK(without.reduced.n_cols() == 3);
    MilpPresolveOptions on;
    on.row_probe.enabled = false;
    on.binary_row_max_cols = 0;
    MilpPresolveStats s1;
    const auto pre = run_structural_presolve(lp, on, s1);
    CHECK(!pre.infeasible);
    CHECK(s1.probing_fixings >= 1);
    CHECK(pre.reduced.n_cols() < 3);
    CHECK(pre.reduced.n_rows() < 3);
    // Every reduced point lifts to a feasible point of the original with z = 1.
    const auto k = static_cast<unsigned>(pre.reduced.n_cols());
    unsigned feasible_reduced = 0;
    for (unsigned mask = 0; mask < (1u << k); ++mask) {
        std::vector<double> y(k);
        for (unsigned j = 0; j < k; ++j) y[j] = (mask >> j) & 1u;
        if (pre.reduced.max_row_violation(y) > 1e-9) continue;
        ++feasible_reduced;
        const auto x = postsolve_point(pre, y);
        CHECK(lp.max_row_violation(x) <= 1e-9);
        CHECK(x[2] == 1.0);
    }
    CHECK(feasible_reduced > 0);
}

}  // namespace

// Side rounding (structural presolve step 2d): integer columns with integer
// coefficients make every activity a multiple of the coefficients' gcd.
forge::model::LpProblem side_rounding_lp(double a1, double a2, double lo, double hi,
                                       bool second_integer) {
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {a1, a2});
    lp.row_lo = {lo};
    lp.row_hi = {hi};
    lp.c = {-1.0, -1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {10.0, 10.0};
    lp.is_integer = {true, second_integer};
    return lp;
}

void test_side_rounding_moves_sides_to_the_gcd_lattice() {
    const auto hi0 = [](const forge::search::MilpPresolveResult& pre) {
        return pre.reduced.n_rows() > 0 ? pre.reduced.row_hi[0] : -1.0;
    };
    const auto lo0 = [](const forge::search::MilpPresolveResult& pre) {
        return pre.reduced.n_rows() > 0 ? pre.reduced.row_lo[0] : -1.0;
    };
    MilpPresolveOptions opts;
    opts.probing_presolve = false;
    opts.structural_fbbt = false;
    opts.binary_row_support = false;
    opts.coefficient_strengthening = false;
    // Identical columns would be merged and the row turned into a bound.
    opts.merge_duplicate_columns = false;
    {   // 2x + 4y <= 7 holds for the same integer points as 2x + 4y <= 6.
        MilpPresolveStats stats;
        auto pre = run_structural_presolve(
            side_rounding_lp(2.0, 4.0, -forge::model::kInf, 7.0, true), opts, stats);
        CHECK(!pre.infeasible);
        CHECK(stats.sides_rounded == 1);
        CHECK(pre.reduced.n_rows() == 1);
        CHECK_NEAR(hi0(pre), 6.0, 0.0);
    }
    {   // 3x + 6y >= 1 becomes >= 3; both sides of a ranged row move.
        MilpPresolveStats stats;
        auto pre = run_structural_presolve(
            side_rounding_lp(3.0, 6.0, 1.0, 10.0, true), opts, stats);
        CHECK(!pre.infeasible);
        CHECK_NEAR(lo0(pre), 3.0, 0.0);
        CHECK_NEAR(hi0(pre), 9.0, 0.0);
    }
    {   // 2x + 4y = 3 has no integer point (3 is not a multiple of 2).
        MilpPresolveStats stats;
        auto pre = run_structural_presolve(
            side_rounding_lp(2.0, 4.0, 3.0, 3.0, true), opts, stats);
        CHECK(pre.infeasible);
    }
    {   // A continuous column leaves the row alone.
        MilpPresolveStats stats;
        auto pre = run_structural_presolve(
            side_rounding_lp(2.0, 4.0, -forge::model::kInf, 7.0, false), opts, stats);
        CHECK(stats.sides_rounded == 0);
        CHECK_NEAR(hi0(pre), 7.0, 0.0);
    }
    {   // A side within round-off of a multiple is not pushed below it:
        // 2x + 4y <= 5.9999999999 keeps the points with activity 6 out only
        // through the feasibility tolerance, never by rounding to 4.
        MilpPresolveStats stats;
        auto pre = run_structural_presolve(
            side_rounding_lp(2.0, 4.0, -forge::model::kInf, 5.9999999999, true), opts, stats);
        CHECK(!pre.infeasible);
        CHECK(stats.sides_rounded == 0);
        CHECK(hi0(pre) >= 5.9999999999);
    }
    {   // Switched off: untouched.
        MilpPresolveOptions off = opts;
        off.side_rounding = false;
        MilpPresolveStats stats;
        auto pre = run_structural_presolve(
            side_rounding_lp(2.0, 4.0, -forge::model::kInf, 7.0, true), off, stats);
        CHECK(stats.sides_rounded == 0);
        CHECK_NEAR(hi0(pre), 7.0, 0.0);
    }
}

// Aggregation (aggregate_presolve.hpp). x0, x1 integer in [0,5], y
// continuous in [ylo, yhi]; r0: x0 + x1 - y = 0, r1: 2 x0 + y <= 8,
// objective -x0 - 2 x1 + y/2. y is substituted out (y = x0 + x1): r1 becomes
// 3 x0 + x1 <= 8 and the costs -1/2, -3/2. With y in [0,10] the row implies
// y's bounds and leaves with it; with y in [0,3] it stays as 0 <= x0+x1 <= 3.
// Every integer (x0, x1) must be feasible in the reduced model exactly when
// (x0, x1, x0 + x1) is in the original, with the same objective, and
// postsolve must give back y exactly.
void test_aggregation_substitutes_exactly() {
    for (const double yhi : {10.0, 3.0}) {
        forge::model::LpProblem lp;
        lp.A = forge::sparse::from_triplets(2, 3, {0, 0, 0, 1, 1}, {0, 1, 2, 0, 2},
                                          {1., 1., -1., 2., 1.});
        lp.c = {-1., -2., .5};
        lp.col_lo = {0., 0., 0.};
        lp.col_hi = {5., 5., yhi};
        lp.is_integer = {1, 1, 0};
        lp.row_lo = {0., -forge::model::kInf};
        lp.row_hi = {0., 8.};
        MilpPresolveOptions opts;
        opts.probing_presolve = false;
        opts.structural_fbbt = false;
        opts.coefficient_strengthening = false;
        opts.row_probe.enabled = false;
        MilpPresolveStats stats;
        auto pre = run_structural_presolve(lp, opts, stats);
        CHECK(!pre.infeasible);
        CHECK(stats.aggregated_cols == 1);
        CHECK(stats.aggregation_rows_removed == (yhi == 10.0 ? 1 : 0));
        CHECK(pre.reduced.n_cols() == 2);
        CHECK(pre.eliminated[2] == 1);
        if (pre.reduced.n_cols() != 2) continue;
        for (int x0 = 0; x0 <= 5; ++x0)
            for (int x1 = 0; x1 <= 5; ++x1) {
                const std::vector<double> x{double(x0), double(x1), double(x0 + x1)};
                std::vector<double> red(2);
                for (int j = 0; j < 2; ++j)
                    if (pre.reduced_col[j] >= 0) red[pre.reduced_col[j]] = x[j];
                const bool orig_ok = lp.max_row_violation(x) == 0.0 &&
                                     lp.max_bound_violation(x) == 0.0;
                const bool red_ok = pre.reduced.max_row_violation(red) == 0.0 &&
                                    pre.reduced.max_bound_violation(red) == 0.0;
                CHECK(orig_ok == red_ok);
                if (!red_ok) continue;
                CHECK(postsolve_point(pre, red) == x);
                CHECK(pre.reduced.objective(red) == lp.objective(x));
            }
    }
}

// Integrality survives exactly or nothing happens: an integer column leaves
// only through a unit coefficient on an all-integer row (2 x0 + 3 x1 = 7 has
// none), and a rewrite that is not exactly representable is declined
// (x0 + x1 - 3 y = 0 would give y's cost 1 as 1/3 on x0 and x1).
void test_aggregation_declines_what_it_cannot_keep_exact() {
    MilpPresolveOptions opts;
    opts.probing_presolve = false;
    opts.structural_fbbt = false;
    opts.coefficient_strengthening = false;
    opts.side_rounding = false;
    opts.row_probe.enabled = false;
    opts.binary_row_support = false;
    opts.merge_duplicate_columns = false;   // x0, x1 below are duplicates
    {
        forge::model::LpProblem lp;
        lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {2., 3.});
        lp.c = {1., 1.};
        lp.col_lo = {0., 0.};
        lp.col_hi = {10., 10.};
        lp.is_integer = {1, 1};
        lp.row_lo = lp.row_hi = {7.};
        MilpPresolveStats stats;
        const auto pre = run_structural_presolve(lp, opts, stats);
        CHECK(stats.aggregated_cols == 0);
        CHECK(pre.reduced.n_cols() == 2);
    }
    {
        forge::model::LpProblem lp;
        lp.A = forge::sparse::from_triplets(2, 3, {0, 0, 0, 1, 1}, {0, 1, 2, 0, 1},
                                          {1., 1., -3., 1., 1.});
        lp.c = {0., 0., 1.};
        lp.col_lo = {0., 0., 0.};
        lp.col_hi = {4., 4., 10.};
        lp.is_integer = {1, 1, 0};
        lp.row_lo = {0., -forge::model::kInf};
        lp.row_hi = {0., 6.};
        MilpPresolveStats stats;
        const auto pre = run_structural_presolve(lp, opts, stats);
        CHECK(stats.aggregated_cols == 0);
        CHECK(stats.aggregation_inexact_skipped >= 1);
        CHECK(pre.reduced.n_cols() == 3);
    }
}

// Presolve is an EXACT equivalence (the search runs optimality reductions such
// as dual fixing on the reduced model), never a relaxation of the original:
// four hand-traced compositions from the 2026-10-08 follow-up audit, each
// solved through the presolve wrapper. M = 2^54.
forge::model::LpProblem exactness_model(Index n, const std::vector<f64>& c,
                                      const std::vector<f64>& lo, const std::vector<f64>& hi,
                                      const std::vector<Index>& ri, const std::vector<Index>& cj,
                                      const std::vector<f64>& v, const std::vector<f64>& rlo,
                                      const std::vector<f64>& rhi, f64 offset = 0.0) {
    forge::model::LpProblem lp;
    lp.name = "exactness";
    lp.A = forge::sparse::from_triplets(static_cast<Index>(rlo.size()), n, ri, cj, v);
    lp.c = c;
    lp.col_lo = lo;
    lp.col_hi = hi;
    lp.is_integer.assign(static_cast<std::size_t>(n), true);
    lp.row_lo = rlo;
    lp.row_hi = rhi;
    lp.obj_offset = offset;
    return lp;
}

f64 solve_presolved(const forge::model::LpProblem& lp, Status& st) {
    forge::search::BabOptions opts;
    opts.structural_presolve.enabled = true;
    opts.para_bab.threads = 1;
    opts.gap_tol = opts.abs_gap_tol = 0.0;
    forge::search::BabDiagnostics diag;
    auto raw = forge::search::solve_milp(lp, opts, diag);
    const auto res = forge::certify::finalize_result(std::move(raw), forge::search::milp_evidence(diag, opts));
    st = res.status;
    return res.objective;
}

void test_presolve_keeps_rows_cancellation_makes_look_redundant() {
    // M x + y - M z <= 0, min M (z - x) - 2 y, x, y binary, z in [1, 2].
    // The maximizing corner (1, 1, 1) sums to 0 in binary64 (exactly 1).
    const f64 M = 0x1p54;
    const auto lp = exactness_model(3, {-M, -2.0, M}, {0, 0, 1}, {1, 1, 2}, {0, 0, 0},
                                    {0, 1, 2}, {M, 1.0, -M}, {-forge::model::kInf}, {0.0});
    // The reduced model excludes the corner and keeps the optimum (1, 0, 1).
    MilpPresolveOptions popts;
    popts.probing_presolve = false;
    MilpPresolveStats stats;
    const auto pre = run_structural_presolve(lp, popts, stats);
    CHECK(!pre.infeasible);
    CHECK(stats.redundant_by_activity == 0);
    if (!pre.infeasible) {
        const auto image = [&](const std::vector<f64>& xo) {
            std::vector<f64> xr(static_cast<std::size_t>(pre.reduced.n_cols()), 0.0);
            for (Index j = 0; j < 3; ++j) {
                const Index k = pre.reduced_col[static_cast<std::size_t>(j)];
                if (k >= 0) xr[static_cast<std::size_t>(k)] = xo[static_cast<std::size_t>(j)];
            }
            return xr;
        };
        CHECK(pre.reduced.max_row_violation(image({1.0, 1.0, 1.0})) +
                  pre.reduced.max_bound_violation(image({1.0, 1.0, 1.0})) > 0.5);
        CHECK(pre.reduced.max_row_violation(image({1.0, 0.0, 1.0})) == 0.0);
    }
    // Solved at the default gap (at gap 0 the LP optimum z = 1 + 2^-54 is
    // beyond binary64 and the node LP cannot be certified: an LP precision
    // limit, not a presolve one), the optimum 0 is found and proved.
    forge::search::BabOptions opts;
    opts.structural_presolve.enabled = true;
    opts.para_bab.threads = 1;
    forge::search::BabDiagnostics diag;
    auto raw = forge::search::solve_milp(lp, opts, diag);
    const auto res = forge::certify::finalize_result(std::move(raw), forge::search::milp_evidence(diag, opts));
    CHECK(res.status == Status::Optimal);
    CHECK(res.objective == 0.0);   // (1, 0, 1)
}

void test_presolve_keeps_singleton_rows_no_exact_bound_states() {
    // M x <= M - 2, min -x, x binary: x = 1 violates by 2, optimum 0.
    const f64 M = 0x1p54;
    const auto lp = exactness_model(1, {-1.0}, {0}, {1}, {0}, {0}, {M},
                                    {-forge::model::kInf}, {M - 2.0});
    Status st;
    const f64 obj = solve_presolved(lp, st);
    CHECK(st == Status::Optimal);
    CHECK(obj == 0.0);
}

void test_presolve_keeps_fixed_terms_an_inexact_side_needs() {
    // M x + r <= M with r fixed at 1, min -x: the residual side M - 1 is no
    // double; rounded to M it admitted x = 1. Optimum 0.
    const f64 M = 0x1p54;
    const auto lp = exactness_model(2, {-1.0, 0.0}, {0, 1}, {1, 1}, {0, 0}, {0, 1},
                                    {M, 1.0}, {-forge::model::kInf}, {M});
    Status st;
    const f64 obj = solve_presolved(lp, st);
    CHECK(st == Status::Optimal);
    CHECK(obj == 0.0);
    // The reduced model itself must exclude x = 1.
    MilpPresolveOptions popts;
    popts.probing_presolve = false;
    MilpPresolveStats stats;
    const auto pre = run_structural_presolve(lp, popts, stats);
    CHECK(!pre.infeasible);
    if (!pre.infeasible) {
        const Index k = pre.reduced_col[0];
        if (k >= 0) {
            std::vector<f64> x(static_cast<std::size_t>(pre.reduced.n_cols()), 0.0);
            for (Index j = 0; j < pre.reduced.n_cols(); ++j)
                x[static_cast<std::size_t>(j)] = pre.reduced.col_lo[static_cast<std::size_t>(j)];
            x[static_cast<std::size_t>(k)] = 1.0;
            CHECK(pre.reduced.col_hi[static_cast<std::size_t>(k)] < 1.0 ||
                  pre.reduced.max_row_violation(x) > 0.5);
        }
    }
}

void test_presolve_objective_constant_is_exact() {
    // q = M and r = 1 fixed, x binary; objective q - r - M x; row
    // q / M - r + x >= 0 (x >= 0). The constant M - 1 is no double: rounded
    // to M, the reduced optimum read 0 against the true -1.
    const f64 M = 0x1p54;
    const auto lp = exactness_model(3, {1.0, -1.0, -M}, {M, 1, 0}, {M, 1, 1}, {0, 0, 0},
                                    {0, 1, 2}, {1.0 / M, -1.0, 1.0}, {0.0}, {forge::model::kInf});
    Status st;
    const f64 obj = solve_presolved(lp, st);
    CHECK(st == Status::Optimal);
    CHECK(obj == -1.0);
    MilpPresolveOptions popts;
    popts.probing_presolve = false;
    MilpPresolveStats stats;
    const auto pre = run_structural_presolve(lp, popts, stats);
    CHECK(!pre.infeasible);
    if (!pre.infeasible) {
        // Reduced and original objectives agree exactly at the optimum.
        std::vector<f64> xo{M, 1.0, 1.0};
        std::vector<f64> xr(static_cast<std::size_t>(pre.reduced.n_cols()), 0.0);
        for (Index j = 0; j < 3; ++j) {
            const Index k = pre.reduced_col[static_cast<std::size_t>(j)];
            if (k >= 0) xr[static_cast<std::size_t>(k)] = xo[static_cast<std::size_t>(j)];
        }
        CHECK(pre.reduced.objective(xr) == lp.objective(xo));
    }
}

int main() {
    test_presolve_keeps_rows_cancellation_makes_look_redundant();
    test_presolve_keeps_singleton_rows_no_exact_bound_states();
    test_presolve_keeps_fixed_terms_an_inexact_side_needs();
    test_presolve_objective_constant_is_exact();
    test_aggregation_substitutes_exactly();
    test_aggregation_declines_what_it_cannot_keep_exact();
    test_narrow_continuous_column_keeps_its_optimum();
    test_fixed_column_substitution_is_exact();
    test_duplicate_column_merge_needs_an_exact_domain();
    test_singleton_row_bound_is_rounded_outward();
    test_monotone_pairs_preserve_optimum_and_postsolve();
    test_probing_fixes_and_shrinks();
    test_joint_support_mixed_domain_retention();
    test_structural_relation_budgets_and_fixed_representative();
    test_structural_relations_reject_rounded_rewrites();
    test_structural_joint_assignments_detect_integer_infeasibility();
    test_structural_binary_relations_postsolve();
    test_conflict_components_become_reversible_relations();
    test_global_propagation_prepares_structural_support();
    test_short_binary_row_support_fixes_and_postsolves();
    test_short_binary_rows_retain_every_feasible_assignment();
    test_binary_row_support_budget_and_continuous_guard();
    test_wrapper_total_time_includes_presolve();
    test_wrapper_projects_cut_reference_point();
    test_wrapper_postsolves_a_zero_column_optimum();
    test_wrapper_keeps_portfolio_points_in_original_space();
    test_fixed_column_eliminated_and_folded_into_row();
    test_asymptotic_bounds_are_not_fixed();
    test_singleton_row_tightens_bound_and_is_dropped();
    test_conflicting_singleton_rows_detected_infeasible();
    test_chained_elimination_reaches_fixed_point();
    test_presolved_solve_matches_direct_solve();
    test_side_rounding_moves_sides_to_the_gcd_lattice();
    return forge::test::finish("test_milp_presolve");
}
