// Slice 0 - milp.policy + feature export APIs.
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/features.hpp"
#include "sor/search/milp_policy.hpp"

#include "test_helpers.hpp"

#include <sstream>
#include <string>

using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::MilpPolicy;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

const char* kTiny = R"(NAME          TINY
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      -5             CAP       4
    X2        COST      -3             CAP       2
    X3        COST      -2             CAP       1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       5
BOUNDS
 UI BND       X1        1
 UI BND       X2        1
 UI BND       X3        1
ENDATA
)";

void test_policy_parse_and_default() {
    CHECK(sor::search::kDefaultMilpPolicy == MilpPolicy::Latest);
    BabOptions opts;
    CHECK(opts.policy == MilpPolicy::Latest);

    MilpPolicy p = MilpPolicy::Classical;
    CHECK(sor::search::parse_milp_policy("latest", p));
    CHECK(p == MilpPolicy::Latest);
    CHECK(sor::search::parse_milp_policy("CLASSICAL", p));
    CHECK(p == MilpPolicy::Classical);
    CHECK(!sor::search::parse_milp_policy("bogus", p));
    CHECK(std::string(sor::search::milp_policy_name(MilpPolicy::Latest)) ==
          "latest");
}

void test_feature_dims_stable() {
    CHECK(sor::search::kBranchFeatureDim == 24);
    CHECK(sor::search::kBranchFeatureDimLegacy == 16);
    CHECK(sor::search::kLiftedFeatureDim == 32);
    CHECK(sor::search::kCutFeatureDim == 8);
    CHECK(sor::search::kVarNodeFeatureDim == 8);
    CHECK(sor::search::kConNodeFeatureDim == 6);
    CHECK(sor::search::kCutNodeFeatureDim == 6);
    CHECK(sor::search::kEdgeFeatureDim == 2);
}

void test_features_on_tiny_milp() {
    auto lp = read_text(kTiny);
    CHECK(lp.n_integer() == 3);
    std::vector<sor::core::f64> x = {0.5, 0.75, 0.25};
    sor::search::BranchFeatureContext ctx;
    ctx.lp = &lp;
    ctx.col_lo = &lp.col_lo;
    ctx.col_hi = &lp.col_hi;
    ctx.x = &x;
    ctx.int_tol = 1e-6;

    std::vector<sor::core::Index> cands;
    std::vector<sor::search::BranchFeatureVec> feats;
    sor::search::fill_branch_features_batch(ctx, cands, feats);
    CHECK(cands.size() == 3);
    CHECK(feats.size() == 3);
    for (const auto& f : feats) {
        CHECK(f.size() == static_cast<std::size_t>(sor::search::kBranchFeatureDim));
        CHECK(f[0] > 0.0);  // fractionality
    }

    auto bip = sor::search::build_bipartite_snapshot(lp, lp.col_lo, lp.col_hi, &x);
    CHECK(bip.vars.size() == static_cast<std::size_t>(lp.n_cols()));
    CHECK(bip.cons.size() == static_cast<std::size_t>(lp.n_rows()));
    CHECK(!bip.edges.empty());

    sor::search::CutRow cut;
    cut.cols = {0, 1};
    cut.vals = {1.0, 1.0};
    cut.row_hi = 1.0;
    sor::search::CutFeatureVec cf{};
    sor::search::CutFeatureContext cctx{&lp, &x};
    CHECK(sor::search::fill_cut_features(cctx, cut, cf));
    CHECK(cf[4] > 0.0);  // violation of x1+x2 <= 1 at (0.5,0.75)

    auto tri = sor::search::build_tripartite_snapshot(
        lp, lp.col_lo, lp.col_hi, {cut}, &x);
    CHECK(tri.cuts.size() == 1);
    CHECK(tri.cut_edges.size() == 2);
}

void test_classical_vs_latest_selectable() {
    auto lp = read_text(kTiny);
    BabOptions latest;
    latest.policy = MilpPolicy::Latest;
    latest.max_nodes = 200;
    latest.feasibility_jump = false;
    latest.sub_mip_lns = false;
    latest.probing = false;
    BabDiagnostics d1;
    auto r1 = sor::search::solve_milp(lp, latest, d1);
    CHECK(d1.policy_used == MilpPolicy::Latest);
    CHECK(r1.proposed_status == sor::core::Status::Optimal ||
          r1.proposed_status == sor::core::Status::Feasible ||
          d1.nodes > 0);

    BabOptions classical = latest;
    classical.policy = MilpPolicy::Classical;
    BabDiagnostics d2;
    auto r2 = sor::search::solve_milp(lp, classical, d2);
    CHECK(d2.policy_used == MilpPolicy::Classical);
    (void)r2;
}

}  // namespace

int main() {
    test_policy_parse_and_default();
    test_feature_dims_stable();
    test_features_on_tiny_milp();
    test_classical_vs_latest_selectable();
    return sor::test::finish("test_milp_policy");
}
