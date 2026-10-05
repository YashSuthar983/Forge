// Structural MIP presolve (sor/search/milp_presolve.hpp): fixed-column
// substitution, singleton-row bound tightening, and the postsolve round trip
// that must exactly undo both.
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/milp_presolve.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <chrono>
#include <functional>
#include <random>
#include <sstream>
#include <string>

using sor::core::Status;
using sor::search::MilpPresolveOptions;
using sor::search::MilpPresolveStats;
using sor::search::run_structural_presolve;
using sor::search::postsolve_point;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

void test_short_binary_row_support_fixes_and_postsolves() {
    // Activity bounds alone cannot see that 3x + 2y + 2z = 4 has only the
    // binary assignment (0,1,1). No search or whole-model probing is needed.
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {3., 2., 2.});
    lp.c = {-1., 1., 2.};
    lp.col_lo = {0., 0., 0.}; lp.col_hi = {1., 1., 1.};
    lp.is_integer = {1, 1, 1};
    lp.row_lo = lp.row_hi = {4.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
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
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(2, 5,
        {0,0,0,0,1,1}, {0,1,2,3,3,4}, {3.,2.,2.,1.,1.,1.});
    lp.c = {-1.,1.,2.,0.,0.};
    lp.col_lo.assign(5,0.); lp.col_hi.assign(5,1.);
    lp.is_integer = {1,1,1,0,0};
    lp.row_lo = {4.,-sor::model::kInf}; lp.row_hi = {4.,0.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
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
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(3,3,
        {0,0,1,1,2,2}, {0,1,0,2,1,2}, {1.,1.,1.,1.,1.,1.});
    lp.c.assign(3,1.); lp.col_lo.assign(3,0.); lp.col_hi.assign(3,1.);
    lp.is_integer.assign(3,1); lp.row_lo=lp.row_hi={1.,1.,1.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
    MilpPresolveStats stats;
    const auto pre=run_structural_presolve(lp,opts,stats);
    CHECK(pre.infeasible);
}

void test_structural_binary_relations_postsolve() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(2,3,
        {0,0,1,1}, {0,2,1,2}, {1.,1.,1.,1.});
    lp.c={-1.,2.,3.};lp.col_lo.assign(3,0.);lp.col_hi.assign(3,1.);
    lp.is_integer.assign(3,1);lp.row_lo=lp.row_hi={1.,1.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
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
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(2, 3,
        {0,0,1,1}, {0,1,1,2}, {1.,-1.,1.,1.});
    lp.c = {2.,3.,5.};
    lp.col_lo = {0.,0.,0.}; lp.col_hi = {1.,1.,1.};
    lp.is_integer = {1,1,1};
    lp.row_lo = lp.row_hi = {0.,1.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
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
    sor::model::LpProblem lp;
    lp.A=sor::sparse::from_triplets(2,2,{0,0,1,1},{0,1,0,1},{1.,-1.,1.,1.});
    lp.c={-1.,3.};lp.col_lo={0.,0.};lp.col_hi={1.,1.};lp.is_integer={1,1};
    lp.row_lo={0.,1.};lp.row_hi={0.,sor::model::kInf};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
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
    sor::model::LpProblem lp;
    lp.A=sor::sparse::from_triplets(2,2,{0,0,1,1},{0,1,0,1},{1.,-1.,1e16,1.});
    lp.c={0.,0.};lp.col_lo={0.,0.};lp.col_hi={1.,1.};lp.is_integer={1,1};
    lp.row_lo={0.,-sor::model::kInf};lp.row_hi={0.,2e16};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
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
    lp.A=sor::sparse::from_triplets(1,2,{0,0},{0,1},{1.,1.});
    lp.row_lo=lp.row_hi={1.};lp.c={1e16,1.};
    pre=run_structural_presolve(lp,opts,stats);
    CHECK(!pre.infeasible);CHECK(pre.reduced.n_cols()==2);
    CHECK(stats.binary_substitution_numerical_rejects>=1);
}

void test_joint_support_mixed_domain_retention() {
    // With one continuous column, enumerating the binary box and both ends
    // of its remaining continuous interval checks every linear-objective
    // extreme point. Check the mapping in both directions, including cost.
    auto vertices=[](const sor::model::LpProblem& p) {
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
        sor::model::LpProblem lp;
        std::vector<sor::core::Index> ri,ci;
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
        lp.A=sor::sparse::from_triplets(5,5,ri,ci,av);
        lp.col_lo.assign(5,0.);lp.col_hi.assign(5,1.);lp.is_integer={1,1,1,1,0};
        lp.maximize=example%2;lp.obj_offset=.5;
        for(int j=0;j<5;++j)lp.c.push_back(static_cast<int>(rng()%9)-4);
        MilpPresolveOptions opts;
        opts.monotone_binary_pairs = false;  // this oracle requires EVERY feasible point
        opts.probing_presolve = false;   // these tests isolate one reduction at a time
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
    auto points = [](const sor::model::LpProblem& p) {
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
        sor::model::LpProblem lp;
        const double a = example % 2 ? -2. : 2.;
        std::vector<sor::core::Index> ri{0, 0}, ci{0, 1};
        std::vector<double> av{a, a};
        lp.row_lo = {a < 0. ? a : -sor::model::kInf};
        lp.row_hi = {a > 0. ? a : sor::model::kInf};
        for (int i = 1; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const double coef = j < 2 && example % 3 == 0 ? -2. :
                    static_cast<int>(rng() % 5) - 2.;
                if (coef == 0.) continue;
                ri.push_back(i); ci.push_back(j); av.push_back(coef);
            }
            lp.row_lo.push_back(example % 3 == 0 ? -sor::model::kInf : -2.);
            lp.row_hi.push_back(2. + rng() % 3);
        }
        lp.A = sor::sparse::from_triplets(4, 4, ri, ci, av);
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
        double before = sor::model::kInf, after = sor::model::kInf;
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
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1., 1.});
    lp.col_lo = {0., 0.}; lp.col_hi = {1., 1.}; lp.is_integer = {1, 1};
    lp.row_lo = {-sor::model::kInf}; lp.row_hi = {1.}; lp.c = {1., 1.};
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
        sor::model::LpProblem lp;
        std::vector<sor::core::Index> rows, cols;
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
        lp.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
        lp.col_lo.assign(n, 0.); lp.col_hi.assign(n, 1.);
        lp.is_integer.assign(n, 1); lp.maximize = (example % 2) != 0;
        for (int j = 0; j < n; ++j) lp.c.push_back(static_cast<int>(rng() % 9) - 4);
        MilpPresolveOptions opts;
        opts.monotone_binary_pairs = false;  // isolate primal support retention
        opts.probing_presolve = false;   // these tests isolate one reduction at a time
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
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1., 2.});
    lp.c = {1., 1.}; lp.col_lo = {0., 0.}; lp.col_hi = {1., 1.};
    lp.is_integer = {1, 0}; lp.row_lo = lp.row_hi = {1.};
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
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
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {6., 4.});
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

void test_fixed_column_eliminated_and_folded_into_row() {
    auto lp = read_text(kFixedColumn);
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
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
    const std::vector<sor::core::f64> reduced_x = {4.0};  // x2 = 4
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
    sor::model::LpProblem lp;
    lp.maximize = true;
    lp.obj_offset = 0.0;
    lp.c = {0.0, 1.0, 3.0, -1.0, -2.0, 5.0};
    lp.col_lo = {0., 0., 0., 0., 0., 0.};
    lp.col_hi = {5., 2., 1., 2., 2., 1.};
    lp.is_integer = {0, 0, 1, 1, 1, 1};
    lp.row_lo = {-9., -3., 3., -sor::model::kInf};
    lp.row_hi = {-9., sor::model::kInf, 3., -10.};
    lp.A = sor::sparse::from_triplets(
        4, 6,
        {0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3},
        {0, 1, 3, 4, 5, 0, 1, 5, 2, 3, 4, 0, 1},
        {1., -5., -2., 3., -1., 5., -4., -2., -2., 5., -4., -3., 1.});
    lp.validate();
    MilpPresolveOptions opts;
    opts.probing_presolve = false;   // these tests isolate one reduction at a time
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

    const std::vector<sor::core::f64> reduced_x = {10.0};
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
    sor::search::BabOptions opts;
    opts.structural_presolve.enabled = false;
    sor::search::BabDiagnostics diag_direct;
    auto raw_direct = sor::search::solve_milp(lp, opts, diag_direct);
    const auto ev_direct = sor::search::milp_evidence(diag_direct, opts);
    const auto r_direct =
        sor::certify::finalize_result(std::move(raw_direct), ev_direct);

    MilpPresolveOptions pre_opts;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, pre_opts, stats);
    CHECK(!pre.infeasible);
    sor::search::BabDiagnostics diag_reduced;
    auto raw_reduced = sor::search::solve_milp(pre.reduced, opts, diag_reduced);
    const auto ev_reduced = sor::search::milp_evidence(diag_reduced, opts);
    const auto r_reduced =
        sor::certify::finalize_result(std::move(raw_reduced), ev_reduced);

    CHECK(r_direct.status == Status::Optimal);
    CHECK(r_reduced.status == Status::Optimal);
    CHECK_NEAR(r_direct.objective, r_reduced.objective, 1e-7);

    auto x = postsolve_point(pre, r_reduced.x);
    CHECK(lp.max_row_violation(x) <= 1e-7);
    CHECK(lp.max_bound_violation(x) <= 1e-7);
    CHECK_NEAR(lp.objective(x), r_direct.objective, 1e-7);
}


void test_wrapper_total_time_includes_presolve() {
    sor::model::LpProblem lp;
    constexpr sor::core::Index fixed = 30000;
    lp.A = sor::sparse::from_triplets(0, fixed + 1, {}, {}, {});
    lp.c.assign(fixed + 1, 0.0);
    lp.c.back() = 1.0;
    lp.col_lo.assign(fixed + 1, 0.0);
    lp.col_hi.assign(fixed + 1, 0.0);
    lp.col_hi.back() = 1.0;
    lp.is_integer.assign(fixed + 1, true);
    sor::search::BabOptions opts;
    opts.structural_presolve.enabled = true;
    opts.para_bab.threads = 1;
    opts.time_limit_s = 10.0;
    opts.feasibility_jump = opts.fixprop = opts.rounding_heuristic = false;
    opts.sub_mip_lns = false;
    opts.balans.enabled = opts.kernel_pump.enabled = opts.mrens.enabled = false;
    sor::search::BabDiagnostics diag;
    const auto started = std::chrono::steady_clock::now();
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    CHECK(raw.proposed_status == Status::Optimal);
    CHECK(raw.x.size() == static_cast<std::size_t>(fixed + 1));
    CHECK(diag.structural_presolve.fixed_cols == fixed);
    CHECK(diag.total_ms >= diag.structural_presolve.ms);
    CHECK(wall <= 10.5);

    opts.time_limit_s = 1e-12;
    sor::search::BabDiagnostics timed;
    const auto interrupted = sor::search::solve_milp(lp, opts, timed);
    CHECK(interrupted.proposed_status == Status::Interrupted);
    CHECK(interrupted.termination_reason == "time limit exhausted during structural presolve");
    CHECK(timed.lp_solves == 0);
    CHECK(timed.total_ms >= timed.structural_presolve.ms);
}

void test_wrapper_projects_cut_reference_point() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 3, {0, 0}, {1, 2}, {2.0, 3.0});
    lp.row_lo = {2.5};
    lp.row_hi = {sor::model::kInf};
    lp.c = {0.0, 1.0, 1.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {0.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};
    const std::vector<double> reference{0.0, 0.0, 1.0};
    sor::search::BabOptions opts;
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
    opts.para_bab.threads = 1;
    opts.lp.presolve = false;
    opts.mip_presolve = opts.probing = opts.symmetry = false;
    opts.integer_row_rounding = false;
    opts.cut_reference_point = &reference;
    sor::search::BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.structural_presolve.fixed_cols == 1);
    CHECK(diag.gmi_cuts_added + diag.mir_cuts_added > 0);
    CHECK(diag.invalid_cuts_detected == 0);
    CHECK(diag.node_cuts_invalid_inserted == 0);
    CHECK_NEAR(raw.objective, 1.0, 1e-6);
    CHECK(sor::search::milp_point_max_violation(lp, raw.x, opts.int_tol) <= opts.primal_feas_tol);
}

void test_wrapper_postsolves_a_zero_column_optimum() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(0, 1, {}, {}, {});
    lp.c = {4.0};
    lp.col_lo = lp.col_hi = {3.0};
    lp.is_integer = {true};
    sor::search::BabOptions opts;
    opts.structural_presolve.enabled = true;
    opts.para_bab.threads = 1;
    sor::search::BabDiagnostics diag;
    const auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(raw.proposed_status == Status::Optimal);
    CHECK_NEAR(raw.objective, 12.0, 1e-9);
    CHECK(raw.x.size() == 1);
    if (raw.x.size() == 1) CHECK_NEAR(raw.x[0], 3.0, 1e-9);
    CHECK(sor::search::milp_point_max_violation(lp, raw.x, opts.int_tol) <= opts.primal_feas_tol);
}

void test_wrapper_keeps_portfolio_points_in_original_space() {
    // The first column is fixed; the 16 surviving binaries need a tree.
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> value(1000, 9999), weight(10, 99);
    sor::model::LpProblem lp;
    lp.maximize = true;
    lp.c.assign(17, 0.0);
    lp.col_lo.assign(17, 0.0);
    lp.col_hi.assign(17, 1.0);
    lp.col_hi[0] = 0.0;
    lp.is_integer.assign(17, true);
    for (int j = 1; j <= 16; ++j) lp.c[j] = value(rng) / 1000.0;
    std::vector<sor::core::Index> rows, cols;
    std::vector<double> vals;
    lp.row_lo.assign(5, -sor::model::kInf);
    lp.row_hi.assign(5, 0.0);
    for (int i = 0; i < 5; ++i)
        for (int j = 1; j <= 16; ++j) {
            const double w = weight(rng);
            rows.push_back(i); cols.push_back(j); vals.push_back(w);
            lp.row_hi[i] += 0.5 * w;
        }
    lp.A = sor::sparse::from_triplets(5, 17, rows, cols, vals);
    sor::search::BabOptions opts;
    opts.structural_presolve.enabled = true;
    opts.time_limit_s = 10.0;
    opts.para_bab.threads = 1;
    opts.cuts_enabled = false;
    opts.mip_presolve = opts.probing = opts.symmetry = false;
    opts.integer_row_rounding = false;
    opts.sub_mip_lns = false;
    opts.balans.enabled = opts.kernel_pump.enabled = opts.mrens.enabled = false;
    opts.gap_tol = 0.005;
    sor::search::PortfolioPool pool(true);
    opts.pool = &pool;
    sor::search::BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.nodes > 1);
    CHECK(raw.x.size() == 17);
    const auto shared = pool.best_point();
    CHECK(shared.empty() || sor::search::portfolio_point_is_feasible(lp, shared,
          opts.primal_feas_tol, opts.int_tol));
    CHECK(sor::search::milp_point_max_violation(lp, raw.x, opts.int_tol) <= opts.primal_feas_tol);

    opts.pool = nullptr;
    sor::search::PortfolioOptions portfolio;
    portfolio.workers = 2;
    portfolio.ramp_seconds = 0.0;
    sor::search::PortfolioDiagnostics portfolio_diag;
    sor::search::BabDiagnostics winner;
    const auto raced = sor::search::solve_milp_portfolio(lp, opts, portfolio, winner, portfolio_diag);
    CHECK(portfolio_diag.workers_used == 2);
    CHECK(raced.x.size() == 17);
    CHECK(sor::search::portfolio_point_is_feasible(lp, raced.x, opts.primal_feas_tol, opts.int_tol));
}

// Presolve probing: z = 0 forces p = q = 1, which p + q <= 1 forbids, so only
// probing (not one row at a time) proves z = 1. The fixing goes back through
// the reductions: z disappears, and the two rows it satisfied are dropped.
void test_probing_fixes_and_shrinks() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(3, 3, {0, 0, 1, 1, 2, 2}, {2, 0, 2, 1, 0, 1},
                                      {1., 1., 1., 1., 1., 1.});   // cols: p q z
    lp.row_lo = {1., 1., -sor::model::kInf};
    lp.row_hi = {sor::model::kInf, sor::model::kInf, 1.};
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

int main() {
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
    return sor::test::finish("test_milp_presolve");
}
