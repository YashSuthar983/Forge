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
    CHECK(!o.flow_cover_cuts);
    CHECK(!o.clique_cuts);
    CHECK(o.cut.pool_nnz_budget_factor > 0.0);
    CHECK(o.cut.pool_weight_sparsity > 0.0);
    CHECK(!o.cut.pool_parallel_hard_filter);
}

void test_known_invalid_flow_cover_option_fails_closed() {
    sor::model::LpProblem lp;
    lp.c = {1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {1.0};
    lp.is_integer = {true};
    lp.A = from_triplets(0, 1, {}, {}, {});
    BabOptions opts;
    opts.flow_cover_cuts = true;
    sor::search::BabDiagnostics diag;
    const auto result = sor::search::solve_milp(lp, opts, diag);
    CHECK(result.proposed_status == sor::core::Status::Unsupported);
    CHECK(result.termination_reason.find("known invalid-cut") != std::string::npos);
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
    sor::search::CutFilterStats st;
    const auto out =
        filter_cut_candidates_for_round(std::move(in), lp, x, opts, &st);
    // A near-parallel but DIFFERENT row (cosine 0.99995 with "strong") is not a
    // duplicate: it survives the prefilter, and the pool's soft score decides
    // how much it is worth next to the pick.
    bool has_strong = false, has_orth = false, has_near = false;
    for (const auto& c : out) {
        has_strong |= c.name == "strong";
        has_orth |= c.name == "orth";
        has_near |= c.name == "near_par";
    }
    CHECK(out.size() == 3);
    CHECK(has_strong && has_orth && has_near);
    CHECK(st.rejected_duplicate == 0 && st.rejected_dominated == 0);
}

// Duplicates are exact and canonical: scaled copies collapse to one, the
// stronger right-hand side wins, and opposite-facing rows are kept.
void test_exact_duplicates_dominance_and_opposites() {
    sor::model::LpProblem lp;
    lp.A = from_triplets(0, 2, {}, {}, {});
    lp.c = {1.0, 0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {10.0, 10.0};
    lp.is_integer = {true, true};
    CutOptions opts;
    opts.max_candidates_per_round = 8;
    opts.pool_efficacy_min = 1e-9;
    opts.pool_nnz_budget_factor = 10.0;
    const std::vector<double> x = {1.0, 1.0};   // violates every <= row below

    // x <= 0 and x >= 1 face opposite ways: both must survive.
    {
        std::vector<CutRow> in;
        in.push_back(le_cut({0}, {1.0}, 0.0, "up_to_0"));
        auto ge = le_cut({0}, {1.0}, 0.0, "at_least_1");
        ge.row_lo = 1.0;
        ge.row_hi = sor::model::kInf;
        in.push_back(std::move(ge));
        const std::vector<double> xm = {0.5, 0.0};
        sor::search::CutFilterStats st;
        const auto out = filter_cut_candidates_for_round(std::move(in), lp, xm, opts, &st);
        CHECK(out.size() == 2);
        CHECK(st.rejected_duplicate == 0 && st.rejected_dominated == 0);
    }
    // 2x + 2y <= 3 and x + y <= 1.5 are one inequality (positive scaling):
    // an exact duplicate collapses to one row.
    {
        std::vector<CutRow> in;
        in.push_back(le_cut({0, 1}, {2.0, 2.0}, 3.0, "scaled"));
        in.push_back(le_cut({0, 1}, {1.0, 1.0}, 1.5, "plain"));
        sor::search::CutFilterStats st;
        const auto out = filter_cut_candidates_for_round(std::move(in), lp, x, opts, &st);
        CHECK(out.size() == 1);
        CHECK(st.rejected_duplicate == 1);
    }
    // Same left-hand side, different right-hand sides: the stronger stays.
    {
        std::vector<CutRow> in;
        in.push_back(le_cut({0, 1}, {1.0, 1.0}, 1.2, "weaker"));
        in.push_back(le_cut({0, 1}, {1.0, 1.0}, 0.8, "stronger"));
        sor::search::CutFilterStats st;
        const auto out = filter_cut_candidates_for_round(std::move(in), lp, x, opts, &st);
        CHECK(out.size() == 1);
        if (out.size() == 1) CHECK(out[0].name == "stronger");
        CHECK(st.rejected_dominated == 1);
    }
    // Different support is never a duplicate, even when the coefficients agree.
    {
        std::vector<CutRow> in;
        in.push_back(le_cut({0}, {1.0}, 0.2, "on_x"));
        in.push_back(le_cut({1}, {1.0}, 0.2, "on_y"));
        const auto out = filter_cut_candidates_for_round(std::move(in), lp, x, opts);
        CHECK(out.size() == 2);
    }
}

// The prefilter caps generously and leaves selection to the pool: more
// candidates than max_candidates_per_round (up to the factor) get through.
void test_prefilter_cap_is_generous() {
    sor::model::LpProblem lp;
    lp.A = from_triplets(0, 6, {}, {}, {});
    lp.c.assign(6, 0.0);
    lp.col_lo.assign(6, 0.0);
    lp.col_hi.assign(6, 10.0);
    lp.is_integer.assign(6, true);
    CutOptions opts;
    opts.max_candidates_per_round = 2;
    opts.prefilter_cap_factor = 4;
    opts.pool_efficacy_min = 1e-9;
    std::vector<CutRow> in;
    for (int j = 0; j < 6; ++j)
        in.push_back(le_cut({j}, {1.0}, 1.0 + 0.1 * j, "c"));
    const std::vector<double> x(6, 2.0);
    const auto out = filter_cut_candidates_for_round(std::move(in), lp, x, opts);
    CHECK(out.size() == 6);
}

// The cut-round progress measure does not depend on constants in the objective.
void test_round_progress_ignores_objective_constants() {
    using sor::search::cut_round_progress;
    const double nan = std::nan("");
    // No incumbent: a 3% move of the offset-free scale, wherever the constant sits.
    const double p0 = cut_round_progress(100.0, 103.0, nan, 0.0);
    CHECK(std::fabs(p0 - 0.03) < 1e-12);
    // Adding a constant K to the objective shifts bounds and offset together.
    for (const double k : {1e3, -50.0, 1e6}) {
        const double p = cut_round_progress(100.0 + k, 103.0 + k, nan, k);
        CHECK(std::fabs(p - p0) < 1e-9);
    }
    // With an incumbent it is the share of the gap closed: shift-invariant too.
    const double g0 = cut_round_progress(90.0, 95.0, 100.0, 0.0);
    CHECK(std::fabs(g0 - 0.5) < 1e-12);
    CHECK(std::fabs(cut_round_progress(90.0 + 7e5, 95.0 + 7e5, 100.0 + 7e5, 7e5) - g0) < 1e-9);
    // A closed gap has nothing left to gain.
    CHECK(cut_round_progress(100.0, 100.0, 100.0, 0.0) == 0.0);
    // Unknown previous bound counts as full progress (no basis to call it stalled).
    CHECK(cut_round_progress(nan, 5.0, nan, 0.0) == 1.0);
}

}  // namespace

int main() {
    test_auto_policy_enables_safe_separators_not_clique();
    test_known_invalid_flow_cover_option_fails_closed();
    test_filter_drops_parallel_low_value();
    test_exact_duplicates_dominance_and_opposites();
    test_prefilter_cap_is_generous();
    test_round_progress_ignores_objective_constants();
    return sor::test::finish("test_cut_policy");
}
