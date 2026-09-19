#include "sor/search/bab.hpp"
#include "sor/search/cut_policy.hpp"
#include "sor/search/cuts.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

using sor::sparse::from_triplets;

#include <cmath>
#include <limits>
#include <string>
#include <vector>

using sor::core::Index;
using sor::search::apply_auto_cuts_policy;
using sor::search::BabOptions;
using sor::search::CutOptions;
using sor::search::CutRow;
using sor::search::filter_cut_candidates_for_round;
using sor::search::MilpPolicy;

namespace {

sor::search::CutRow le_cut(std::vector<Index> cols, std::vector<double> vals,
                           double rhs, const char* name) {
    sor::search::CutRow c;
    c.cols = std::move(cols);
    c.vals = std::move(vals);
    c.row_hi = rhs;
    c.row_lo = -sor::model::kInf;
    c.name = name;
    return c;
}

void test_auto_policy_enables_safe_separators_not_clique() {
    BabOptions o;
    o.policy = MilpPolicy::Latest;
    CHECK(!o.mir_cuts);
    CHECK(!o.clique_cuts);
    apply_auto_cuts_policy(o);
    CHECK(o.mir_cuts);
    CHECK(o.lifted_cover_cuts);
    CHECK(o.zerohalf_cuts);
    CHECK(o.flow_cover_cuts);
    CHECK(!o.clique_cuts);
    CHECK(o.cut.pool_nnz_budget_factor > 0.0);
    CHECK(o.cut.pool_weight_sparsity > 0.0);
    CHECK(!o.cut.pool_parallel_hard_filter);
}

void test_filter_drops_parallel_low_value() {
    sor::model::LpProblem lp;
    lp.A = from_triplets(0, 2, {}, {}, {});
    lp.c = {1.0, 0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {10.0, 10.0};
    lp.is_integer = {true, true};

    CutOptions opts;
    opts.max_candidates_per_round = 4;
    opts.pool_efficacy_min = 1e-9;
    opts.pool_parallel_hard_filter = false;
    opts.pool_parallelism_penalty = 0.5;
    opts.pool_parallelism_penalty_min = 0.0;
    opts.pool_nnz_budget_factor = 10.0;

    std::vector<CutRow> in;
    in.push_back(le_cut({0}, {1.0}, 1.5, "strong"));
    in.push_back(le_cut({0, 1}, {1.0, 0.01}, 1.6, "near_par"));
    in.push_back(le_cut({1}, {1.0}, 0.5, "orth"));

    const std::vector<double> x = {2.0, 2.0};
    const auto out =
        filter_cut_candidates_for_round(std::move(in), lp, x, opts);
    CHECK(out.size() >= 1);
    CHECK(out.size() <= static_cast<std::size_t>(opts.max_candidates_per_round));
    bool has_orth = false;
    for (const auto& c : out)
        if (c.name == "orth") has_orth = true;
    CHECK(has_orth);
}

}  // namespace

int main() {
    test_auto_policy_enables_safe_separators_not_clique();
    test_filter_drops_parallel_low_value();
    return sor::test::finish("test_cut_policy");
}
