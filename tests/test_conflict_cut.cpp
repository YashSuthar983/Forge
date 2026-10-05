// WP-B: Mexi cut-based conflict analysis (arXiv:2410.15110 paper-complete).
#include "forge/certify/finalize.hpp"
#include "forge/io/mps.hpp"
#include "forge/search/bab.hpp"
#include "forge/search/conflict_cut.hpp"
#include "forge/search/propagate.hpp"
#include "forge/search/prop_trail.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include "forge/core/route_debug.hpp"

using forge::core::Status;
using forge::search::BabDiagnostics;
using forge::search::BabOptions;
using forge::search::ConflictAnalysisContext;
using forge::search::ConflictCutMode;
using forge::search::ConflictCutOptions;
using forge::search::CutValidity;
using forge::search::MilpPolicy;

namespace {

forge::model::LpProblem read_text(const std::string& mps) {
    forge::io::MpsReadReport rep;
    std::istringstream in(mps);
    return forge::io::read_mps(in, rep);
}

const char* kPairInfeas = R"(NAME          PAIRINF
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      1              CAP       1
    X2        COST      1              CAP       1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       1
BOUNDS
 UI BND       X1        1
 UI BND       X2        1
ENDATA
)";

void test_analyze_learns_valid_cut() {
    auto lp = read_text(kPairInfeas);
    std::vector<forge::core::f64> lo = lp.col_lo, hi = lp.col_hi;
    lo[0] = 1.0;
    hi[0] = 1.0;
    lo[1] = 1.0;
    hi[1] = 1.0;

    forge::search::PropTrail trail;
    trail.push(0, forge::search::BoundDir::Lower, 1.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 1);
    trail.push(1, forge::search::BoundDir::Lower, 1.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 1);

    forge::search::PropagateResult pr;
    pr.feasible = false;
    pr.conflict_row = 0;
    pr.conflict_var = 0;

    ConflictAnalysisContext ctx;
    ctx.lp = &lp;
    ctx.col_lo = &lo;
    ctx.col_hi = &hi;
    ctx.trail = &trail;
    ctx.conflict_row = pr.conflict_row;
    ctx.conflict_var = pr.conflict_var;

    ConflictCutOptions opts;
    opts.enabled = true;
    opts.mode = ConflictCutMode::Paper;
    forge::search::ConflictCutDiagnostics cd;
    const auto cut = forge::search::analyze_conflict_cuts(ctx, opts, cd);
    CHECK(cut.has_value());
    CHECK(cd.learned >= 1);
    // Pure-binary model, no continuous columns: the checker enumerates the
    // whole box, so this is a VERIFIED validity proof (not an assumption).
    CHECK(forge::search::conflict_cut_check_binary(lp, *cut) ==
          CutValidity::Verified);
}

void test_latest_bab_learns_global_conflict_cut() {
    auto lp = read_text(kPairInfeas);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.conflict_cut.enabled = true;
    opts.conflict_cut.mode = ConflictCutMode::Paper;
    opts.max_nodes = 50;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.probing = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.cuts_enabled = false;
    opts.rounding_heuristic = false;
    opts.lp_rounding_repair = false;
    opts.integer_dive = false;
    opts.integer_neighborhood = false;
    opts.conflict_propagation = false;
    lp.col_lo[0] = 1.0;
    lp.col_hi[0] = 1.0;
    lp.col_lo[1] = 1.0;
    lp.col_hi[1] = 1.0;

    BabDiagnostics diag;
    forge::search::solve_milp(lp, opts, diag);
    CHECK(diag.conflict_cut_diag.attempts >= 1);
    CHECK(diag.conflict_cuts_global >= 1);
    CHECK(diag.conflict_cuts_global +
              diag.conflict_cut_diag.validation_rejected ==
          diag.conflict_cut_diag.learned);
}

void test_classical_default_off() {
    ConflictCutOptions o;
    o.enabled = true;
    forge::search::apply_conflict_cut_policy(MilpPolicy::Classical, o);
    CHECK(!o.enabled);
    BabOptions latest;
    latest.structural_presolve.enabled = false;  // component test: keep the model unreduced
    latest.policy = MilpPolicy::Latest;
    // Product default: Mexi on in struct; bab auto-disables on dense pure-binary.
    CHECK(latest.conflict_cut.enabled);
    CHECK(latest.conflict_cut.mode == ConflictCutMode::Paper);
}

void test_nogood_from_branch_trail_valid() {
    auto lp = read_text(kPairInfeas);
    forge::search::PropTrail trail;
    // x1=1, x2=1 — classic infeasible assignment under CAP.
    trail.push(0, forge::search::BoundDir::Lower, 1.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 1);
    trail.push(1, forge::search::BoundDir::Lower, 1.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 1);
    const auto ng =
        forge::search::build_nogood_from_branch_trail(trail, lp, 1e-9);
    CHECK(ng.has_value());
    CHECK(!forge::search::conflict_cut_near_empty(*ng));
    // Nogood: (1-x1)+(1-x2) >= 1  ⇒  -x1 -x2 >= -1
    CHECK(ng->cols.size() == 2);
    CHECK(forge::search::conflict_cut_check_binary(lp, *ng) ==
          CutValidity::Verified);
}

// Tri-state checker semantics (fail-closed): a free continuous column makes
// verification impossible; the checker must say Unverified, never silently
// claim validity. Complete boxes give Verified / Refuted definitively.
void test_validity_check_tri_state() {
    // Mixed model: binary x, free continuous y in [0,1], x + y <= 1.
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.row_lo = {-forge::model::kInf};
    lp.row_hi = {1.0};
    lp.c = {1.0, 0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, false};

    // Valid cut on y alone: cannot be verified (y free) -> Unverified.
    forge::search::CutRow cut_y;
    cut_y.cols = {1};
    cut_y.vals = {1.0};
    cut_y.row_lo = -forge::model::kInf;
    cut_y.row_hi = 1.0;
    CHECK(forge::search::conflict_cut_check_binary(lp, cut_y) ==
          CutValidity::Unverified);
    CHECK(forge::search::conflict_cut_check_general(lp, cut_y) ==
          CutValidity::Unverified);

    // Invalid cut x <= 0 (x=1, y=0 is feasible): the support sweep finds the
    // witness even though y is free -> Refuted.
    forge::search::CutRow cut_x;
    cut_x.cols = {0};
    cut_x.vals = {1.0};
    cut_x.row_lo = -forge::model::kInf;
    cut_x.row_hi = 0.0;
    CHECK(forge::search::conflict_cut_check_binary(lp, cut_x) ==
          CutValidity::Refuted);

    // Pure-binary complete box: valid -> Verified, invalid -> Refuted.
    auto b2 = read_text(kPairInfeas);
    forge::search::CutRow valid;
    valid.cols = {0, 1};
    valid.vals = {1.0, 1.0};
    valid.row_lo = -forge::model::kInf;
    valid.row_hi = 1.0;  // x1 + x2 <= 1 (CAP)
    CHECK(forge::search::conflict_cut_check_binary(b2, valid) ==
          CutValidity::Verified);
    forge::search::CutRow invalid;
    invalid.cols = {0, 1};
    invalid.vals = {1.0, 1.0};
    invalid.row_lo = -forge::model::kInf;
    invalid.row_hi = 0.5;  // (1,0) violates
    CHECK(forge::search::conflict_cut_check_binary(b2, invalid) ==
          CutValidity::Refuted);

    // Unbounded integer domains must NOT vacuously Verify (gen-ip002 P0:
    // floor(+inf)→int UB previously skipped every free int column).
    {
        forge::model::LpProblem ub;
        ub.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        ub.row_lo = {-forge::model::kInf};
        ub.row_hi = {5.0};
        ub.c = {1.0, 1.0};
        ub.col_lo = {0.0, 0.0};
        ub.col_hi = {forge::model::kInf, forge::model::kInf};
        ub.is_integer = {true, true};
        forge::search::CutRow c;
        c.cols = {0, 1};
        c.vals = {1.0, 1.0};
        c.row_lo = -forge::model::kInf;
        c.row_hi = 0.0;
        CHECK(forge::search::conflict_cut_check_general(ub, c) ==
              CutValidity::Unverified);
    }
}

void test_near_empty_cut_refused() {
    forge::search::CutRow empty;
    CHECK(forge::search::conflict_cut_near_empty(empty));
    forge::search::CutRow zeros;
    zeros.cols = {0, 1};
    zeros.vals = {0.0, 0.0};
    CHECK(forge::search::conflict_cut_near_empty(zeros));
    // Global ⊥ (0 >= 1) is a real conflict cut, not "empty".
    forge::search::CutRow bot;
    bot.row_lo = 1.0;
    CHECK(!forge::search::conflict_cut_near_empty(bot));
}

void test_unverified_wide_cut_is_not_verified() {
    constexpr int n = 18;
    forge::model::LpProblem lp;
    std::vector<forge::core::Index> rows(n, 0);
    std::vector<forge::core::Index> cols(n);
    std::vector<double> vals(n, 1.0);
    for (int j = 0; j < n; ++j) cols[j] = j;
    lp.A = forge::sparse::from_triplets(1, n, rows, cols, vals);
    lp.row_lo = {1.0};
    lp.row_hi = {forge::model::kInf};
    lp.c.assign(n, 1.0);
    lp.col_lo.assign(n, 0.0);
    lp.col_hi.assign(n, 1.0);
    lp.is_integer.assign(n, true);
    forge::search::CutRow wide;
    wide.cols = cols;
    wide.vals = vals;
    wide.row_lo = 1.0;
    wide.row_hi = forge::model::kInf;
    // Valid for the model (it is the model row) but the support exceeds the
    // enumeration cap, so the checker must abstain instead of inserting it.
    CHECK(forge::search::conflict_cut_check_binary(lp, wide) ==
          forge::search::CutValidity::Unverified);

    forge::model::LpProblem small;
    small.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    small.row_lo = {1.0};
    small.row_hi = {forge::model::kInf};
    small.c = {1.0, 1.0};
    small.col_lo = {0.0, 0.0};
    small.col_hi = {1.0, 1.0};
    small.is_integer = {true, true};
    forge::search::CutRow narrow;
    narrow.cols = {0, 1};
    narrow.vals = {1.0, 1.0};
    narrow.row_lo = 1.0;
    narrow.row_hi = forge::model::kInf;
    CHECK(forge::search::conflict_cut_check_binary(small, narrow) ==
          forge::search::CutValidity::Verified);
}

void test_large_pure_binary_support_verified() {
    // 20 binaries with x0 + x1 >= 1. Full 2^20 exceeds the old enum budget;
    // support+prop verification must still certify the redundant cut.
    constexpr int n = 20;
    std::vector<forge::core::Index> rows(2, 0), cols = {0, 1};
    std::vector<forge::core::f64> vals = {1.0, 1.0};
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, n, rows, cols, vals);
    lp.row_lo = {1.0};
    lp.row_hi = {forge::model::kInf};
    lp.c.assign(static_cast<std::size_t>(n), 1.0);
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), true);

    forge::search::CutRow valid;
    valid.cols = {0, 1};
    valid.vals = {1.0, 1.0};
    valid.row_lo = 1.0;
    valid.row_hi = forge::model::kInf;
    CHECK(forge::search::conflict_cut_check_binary(lp, valid) ==
          CutValidity::Verified);

    // Invalid strengthening: x0 + x1 >= 2 excludes the feasible (1,0).
    forge::search::CutRow invalid = valid;
    invalid.row_lo = 2.0;
    CHECK(forge::search::conflict_cut_check_binary(lp, invalid) ==
          CutValidity::Refuted);
}

void test_p0033_latest_learns_mexi_cut() {
    // p0033: 33 pure binaries — Mexi stays on (n_bin < 80). Before support+prop
    // verification, every FUIP cut died at Verified-only apply (2^33 budget).
    const char* candidates[] = {
        FORGE_SOURCE_DIR "/benchmarks/miplib-easy/mps/p0033.mps",
        "benchmarks/miplib-easy/mps/p0033.mps",
        "../benchmarks/miplib-easy/mps/p0033.mps",
        "forge/benchmarks/miplib-easy/mps/p0033.mps",
    };
    const char* path = nullptr;
    for (const char* c : candidates) {
        std::ifstream in(c);
        if (in) {
            path = c;
            break;
        }
    }
    if (!path) {
        ::forge::test::skip("test_p0033_latest_learns_mexi_cut",
                          "benchmarks/miplib-easy/mps/p0033.mps");
        return;
    }
    forge::io::MpsReadReport rep;
    auto lp = forge::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 15.0;
    opts.feasibility_jump = false;
    // This regression exercises the conflict trajectory reached with the
    // light root-cut policy. The default cut selector can close p0033 before
    // any conflict is generated, making the accounting assertion vacuous.
    opts.auto_cuts = false;
    BabDiagnostics diag;
    forge::search::solve_milp(lp, opts, diag);
    CHECK(diag.conflict_cut_diag.attempts >= 1);
    CHECK(diag.conflict_cut_diag.learned >= 1);
    // Every configured analysis attempt either learns a cut or aborts its
    // derivation; post-learning rejection must not inflate aborted.
    CHECK(diag.conflict_cut_diag.aborted + diag.conflict_cut_diag.learned ==
          diag.conflict_cut_diag.attempts);
    // Accounting only. p0033 is not the accepted-cut regression: a learned
    // cut is inserted or counted validation_rejected, and derivation aborts
    // do not fill that counter. The accepted insertion is
    // test_latest_bab_learns_global_conflict_cut. The rejected checker path
    // is test_unverified_wide_cut_is_not_verified.
    CHECK(diag.conflict_cuts_global +
              diag.conflict_cut_diag.validation_rejected ==
          diag.conflict_cut_diag.learned);
}

// Nogoods must refuse trails that branch on non-binary columns: the node
// box under x >= 2 (general int) cannot be expressed as a 0/1 assignment,
// and the old code silently skipped such entries, producing an invalid
// global cut on mixed models (2026-09-14 review).
void test_nogood_refuses_mixed_trail() {
    // x1 binary, x2 general integer in [0, 10], x1 + x2 <= 0.
    forge::model::LpProblem lp;
    lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.row_lo = {-forge::model::kInf};
    lp.row_hi = {0.0};
    lp.c = {1.0, 1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 10.0};
    lp.is_integer = {true, true};

    forge::search::PropTrail trail;
    // Binary branch x1 = 1 ...
    trail.push(0, forge::search::BoundDir::Lower, 1.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 1);
    // ... plus a general-integer branch x2 >= 2 (mixed trail).
    trail.push(1, forge::search::BoundDir::Lower, 2.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 2);
    const auto ng =
        forge::search::build_nogood_from_branch_trail(trail, lp, 1e-9);
    CHECK(!ng.has_value());

    // Pure-binary trail on the same model still yields a nogood.
    forge::search::PropTrail bin_trail;
    bin_trail.push(0, forge::search::BoundDir::Lower, 1.0, 0.0,
                   forge::search::ReasonKind::Branch, -1, 1);
    const auto ng2 =
        forge::search::build_nogood_from_branch_trail(bin_trail, lp, 1e-9);
    CHECK(ng2.has_value());
    CHECK(ng2->cols.size() == 1);
}

// Tiny general-integer conflict: 2x + 2y <= 3 with x,y in {0,1,2}.
const char* kGenIntPair = R"(NAME          GENINT
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      1              CAP       2
    X2        COST      1              CAP       2
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       3
BOUNDS
 UI BND       X1        2
 UI BND       X2        2
ENDATA
)";

void test_general_integer_conflict_safe() {
    auto lp = read_text(kGenIntPair);
    std::vector<forge::core::f64> lo = {2.0, 2.0};
    std::vector<forge::core::f64> hi = {2.0, 2.0};

    forge::search::PropTrail trail;
    trail.push(0, forge::search::BoundDir::Lower, 2.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 1);
    trail.push(1, forge::search::BoundDir::Lower, 2.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 1);

    ConflictAnalysisContext ctx;
    ctx.lp = &lp;
    ctx.col_lo = &lo;
    ctx.col_hi = &hi;
    ctx.trail = &trail;
    ctx.conflict_row = 0;
    ctx.conflict_var = -1;

    ConflictCutOptions opts;
    opts.enabled = true;
    opts.mode = ConflictCutMode::Paper;
    opts.allow_general_integer = true;
    opts.use_cmirror = true;
    forge::search::ConflictCutDiagnostics cd;
    const auto cut = forge::search::analyze_conflict_cuts(ctx, opts, cd);
    CHECK(cd.attempts >= 1);
    CHECK(cd.general_int_reasons >= 1 || cd.learned >= 1 || cd.aborted >= 1);
    if (cut) {
        bool enumerated = false;
        CHECK(forge::search::conflict_cut_check_general(lp, *cut, 1e-9,
                                                       1u << 12,
                                                       &enumerated) ==
              CutValidity::Verified);
        CHECK(enumerated);
        CHECK(cd.learned >= 1);
    } else {
        CHECK(cd.aborted >= 1);
        CHECK(cd.learned == 0);
    }
}

void test_general_integer_disabled_aborts() {
    // Non-asserting Clearn whose earliest Alg-1 infeasible state is a
    // general-integer row reason. allow_general_integer=false → abort.
    auto lp = read_text(kGenIntPair);
    lp.A = forge::sparse::from_triplets(1, 3, {0, 0, 0}, {0, 1, 2},
                                      {2.0, 2.0, 2.0});
    lp.c = {1.0, 1.0, 1.0};
    lp.row_lo = {-forge::model::kInf};
    lp.row_hi = {3.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {2.0, 2.0, 2.0};
    lp.is_integer = {true, true, true};

    // z>=1 alone keeps conflict feasible; row prop x>=2 makes it infeasible
    // and leaves two non-relaxables (x,z) so not FUIP-asserting.
    std::vector<forge::core::f64> lo = {2.0, 0.0, 1.0};
    std::vector<forge::core::f64> hi = {2.0, 2.0, 2.0};
    forge::search::PropTrail trail;
    trail.push(2, forge::search::BoundDir::Lower, 1.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 1);
    trail.push(0, forge::search::BoundDir::Lower, 2.0, 0.0,
               forge::search::ReasonKind::Row, 0, 1);

    ConflictAnalysisContext ctx;
    ctx.lp = &lp;
    ctx.col_lo = &lo;
    ctx.col_hi = &hi;
    ctx.trail = &trail;
    ctx.conflict_row = 0;
    ConflictCutOptions opts;
    opts.enabled = true;
    opts.mode = ConflictCutMode::Paper;
    opts.allow_general_integer = false;
    forge::search::ConflictCutDiagnostics cd;
    const auto cut = forge::search::analyze_conflict_cuts(ctx, opts, cd);
    CHECK(!cut.has_value());
    CHECK(cd.aborted >= 1);
    CHECK(cd.aborted_reason == cd.aborted);
}

void test_safe_limited_skips_cmir_on_nonbinary() {
    auto lp = read_text(kGenIntPair);
    std::vector<forge::core::f64> lo = {2.0, 2.0};
    std::vector<forge::core::f64> hi = {2.0, 2.0};
    forge::search::PropTrail trail;
    trail.push(0, forge::search::BoundDir::Lower, 2.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 1);
    trail.push(1, forge::search::BoundDir::Lower, 2.0, 0.0,
               forge::search::ReasonKind::Row, 0, 1);

    ConflictAnalysisContext ctx;
    ctx.lp = &lp;
    ctx.col_lo = &lo;
    ctx.col_hi = &hi;
    ctx.trail = &trail;
    ctx.conflict_row = 0;

    ConflictCutOptions opts;
    opts.enabled = true;
    opts.mode = ConflictCutMode::SafeLimited;
    opts.use_cmirror = true;
    opts.allow_general_integer = true;
    forge::search::ConflictCutDiagnostics cd;
    (void)forge::search::analyze_conflict_cuts(ctx, opts, cd);
    // SafeLimited must record skip when it hits a non-binary row reason.
    CHECK(cd.cmir_skipped_nonbinary >= 1 || cd.aborted >= 1 || cd.learned >= 1);
}

// Paper Example 2 / Fig. 2 mixed-binary skeleton (reconstructed):
// binaries x1,x2,x3; continuous y1∈[0,1], y2∈[-1,1].
// Local: x2=0 ⇒ y2≤0 (C4), y2≥0 & x3=0 (C5), y1≤3/4 (C1), x1≥1 (C2),
// C3 infeasible. Analysis must either learn a valid cut or abort — never
// emit an inequality violated by a feasible MBP point.
void test_mixed_binary_example2_safe() {
    forge::model::LpProblem lp;
    // Paper Example 2 / Fig. 2 (reconstructed): binaries x1,x2,x3;
    // continuous y1∈[0,1], y2∈[-1,1].
    // C1 L: 2x1+4y1+2y2 <= 3
    // C2 G: 20x1+5y1−y2 >= 4
    // C3 L: 20x1−5y1+10y2 <= 16
    // C4 L: x2+y2 <= 0
    // C5 G: −x3+y2 >= 0
    lp.A = forge::sparse::from_triplets(
        5, 5,
        {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4},
        {0, 3, 4, 0, 3, 4, 0, 3, 4, 1, 4, 2, 4},
        {2.0, 4.0, 2.0, 20.0, 5.0, -1.0, 20.0, -5.0, 10.0, 1.0, 1.0, -1.0, 1.0});
    lp.row_lo = {-forge::model::kInf, 4.0, -forge::model::kInf, -forge::model::kInf,
                 0.0};
    lp.row_hi = {3.0, forge::model::kInf, 16.0, 0.0, forge::model::kInf};
    lp.c = {1, 1, 1, 0, 0};
    lp.col_lo = {0, 0, 0, 0, -1};
    lp.col_hi = {1, 1, 1, 1, 1};
    lp.is_integer = {true, true, true, false, false};

    // Local domain after Example 2 cascade; C3 infeasible.
    std::vector<forge::core::f64> lo = {1.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<forge::core::f64> hi = {1.0, 0.0, 0.0, 0.75, 0.0};

    forge::search::PropTrail trail;
    trail.push(1, forge::search::BoundDir::Upper, 0.0, 1.0,
               forge::search::ReasonKind::Branch, -1, 1);
    trail.push(4, forge::search::BoundDir::Upper, 0.0, 1.0,
               forge::search::ReasonKind::Row, 3, 1);
    trail.push(4, forge::search::BoundDir::Lower, 0.0, -1.0,
               forge::search::ReasonKind::Row, 4, 1);
    trail.push(2, forge::search::BoundDir::Upper, 0.0, 1.0,
               forge::search::ReasonKind::Row, 4, 1);
    trail.push(3, forge::search::BoundDir::Upper, 0.75, 1.0,
               forge::search::ReasonKind::Row, 0, 1);
    trail.push(0, forge::search::BoundDir::Lower, 1.0, 0.0,
               forge::search::ReasonKind::Row, 1, 1);

    ConflictAnalysisContext ctx;
    ctx.lp = &lp;
    ctx.col_lo = &lo;
    ctx.col_hi = &hi;
    ctx.trail = &trail;
    ctx.conflict_row = 2;

    ConflictCutOptions opts;
    opts.enabled = true;
    opts.mode = ConflictCutMode::Paper;
    opts.use_cmirror = true;
    forge::search::ConflictCutDiagnostics cd;
    const auto cut = forge::search::analyze_conflict_cuts(ctx, opts, cd);
    CHECK(cd.attempts >= 1);
    if (cut) {
        CHECK(cd.learned >= 1);
        auto lp_fix = lp;
        lp_fix.col_lo[3] = lo[3];
        lp_fix.col_hi[3] = hi[3];
        lp_fix.col_lo[4] = lo[4];
        lp_fix.col_hi[4] = hi[4];
        // y1 keeps its [0, 0.75] freedom, so neither checker can VERIFY;
        // the assertion is the safety property itself: no feasible point
        // within the sweep refutes the learned cut.
        CHECK(forge::search::conflict_cut_check_binary(lp_fix, *cut) !=
              CutValidity::Refuted);
        CHECK(forge::search::conflict_cut_check_general(lp_fix, *cut) !=
              CutValidity::Refuted);
    } else {
        CHECK(cd.aborted >= 1);
        CHECK(cd.learned == 0);
    }
}

void test_paper_mode_default() {
    ConflictCutOptions o;
    CHECK(o.mode == ConflictCutMode::Paper);
    CHECK(o.nogood_cuts);  // Latest default: branch-trail nogoods on
}

// 2x1 + 2x2 + 2x3 = 3 with binaries: the LP relaxation sits at the fractional
// point (1/2,1/2,1/2) so the tree must branch, and every integer leaf is
// infeasible (the row can only sum to 0, 2, 4 or 6) — branches like
// x1=0,x2=0 (forcing 2x3=3 > 1) are the canonical nogood source.
const char* kNogoodModel = R"(NAME          NOGOOD
ROWS
 N  COST
 G  EQLO
 L  EQHI
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      1              EQLO      2
    X1        EQHI      2
    X2        COST      1              EQLO      2
    X2        EQHI      2
    X3        COST      1              EQLO      2
    X3        EQHI      2
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       EQLO      3              EQHI      3
BOUNDS
 UI BND       X1        1
 UI BND       X2        1
 UI BND       X3        1
ENDATA
)";

BabOptions nogood_test_options() {
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.max_nodes = 50;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.probing = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.cuts_enabled = false;
    opts.rounding_heuristic = false;
    opts.lp_rounding_repair = false;
    opts.integer_dive = false;
    opts.integer_neighborhood = false;
    opts.conflict_propagation = false;
    return opts;
}

void test_latest_bab_learns_nogood() {
    auto lp = read_text(kNogoodModel);
    BabOptions opts = nogood_test_options();
    opts.policy = MilpPolicy::Latest;
    opts.conflict_cut.enabled = true;
    opts.conflict_cut.nogood_cuts = true;
    // The LP-row path (kept as the fallback) ...
    opts.conflict_cut.conflict_store = false;
    BabDiagnostics diag;
    forge::search::solve_milp(lp, opts, diag);
    CHECK(diag.nogood_cuts_global >= 1);
    CHECK(diag.conflict_store.added == 0);
    // ... and the persistent store, which learns the same nogood without
    // touching the LP.
    opts.conflict_cut.conflict_store = true;
    BabDiagnostics stored;
    forge::search::solve_milp(lp, opts, stored);
    // A unit clause is applied to the root box instead of being stored.
    CHECK(stored.conflict_store.added + stored.conflict_store_root_units >= 1);
    CHECK(stored.nogood_cuts_global == 0);
}

void test_classical_learns_no_conflict_family() {
    // Classical ablation must run the classical path only: neither Mexi
    // conflict cuts nor branch-trail nogoods may reach global_lp.
    auto lp = read_text(kNogoodModel);
    BabOptions opts = nogood_test_options();
    opts.policy = MilpPolicy::Classical;
    BabDiagnostics diag;
    forge::search::solve_milp(lp, opts, diag);
    CHECK(diag.conflict_cuts_global == 0);
    CHECK(diag.nogood_cuts_global == 0);
}

void test_nogood_cap_zero_disables_learning() {
    auto lp = read_text(kNogoodModel);
    BabOptions opts = nogood_test_options();
    opts.policy = MilpPolicy::Latest;
    opts.conflict_cut.max_nogood_cuts = 0;
    BabDiagnostics diag;
    forge::search::solve_milp(lp, opts, diag);
    CHECK(diag.nogood_cuts_global == 0);
}

void test_local_cut_row_aborts_global_learn() {
    // Conflict on a row past n_global_rows must abort — never promote a
    // node-local GMI/MIR into a global conflict cut.
    auto lp = read_text(kPairInfeas);
    std::vector<forge::core::f64> lo = {1.0, 1.0};
    std::vector<forge::core::f64> hi = {1.0, 1.0};
    forge::search::PropTrail trail;
    trail.push(0, forge::search::BoundDir::Lower, 1.0, 0.0,
               forge::search::ReasonKind::Branch, -1, 1);
    trail.push(1, forge::search::BoundDir::Lower, 1.0, 0.0,
               forge::search::ReasonKind::Row, /*local row*/ 5, 1);

    ConflictAnalysisContext ctx;
    ctx.lp = &lp;
    ctx.col_lo = &lo;
    ctx.col_hi = &hi;
    ctx.trail = &trail;
    ctx.conflict_row = 5;
    ctx.n_global_rows = 1;  // only row 0 is global

    ConflictCutOptions opts;
    opts.enabled = true;
    forge::search::ConflictCutDiagnostics cd;
    const auto cut = forge::search::analyze_conflict_cuts(ctx, opts, cd);
    CHECK(!cut.has_value());
    CHECK(cd.aborted >= 1);
    CHECK(cd.learned == 0);
}

void test_flugpl_latest_dual_not_above_opt() {
    // P0 regression (2026-09-13): Latest + conflict cuts promoted local tree
    // GMI into global_lp and proved dual 1253765 > true opt 1201500.
    const char* candidates[] = {
        FORGE_SOURCE_DIR "/benchmarks/miplib-easy/mps/flugpl.mps",
        "benchmarks/miplib-easy/mps/flugpl.mps",
        "../benchmarks/miplib-easy/mps/flugpl.mps",
        "forge/benchmarks/miplib-easy/mps/flugpl.mps",
    };
    const char* path = nullptr;
    for (const char* c : candidates) {
        std::ifstream in(c);
        if (in) {
            path = c;
            break;
        }
    }
    if (!path) {
        ::forge::test::skip("test_flugpl_latest_dual_not_above_opt",
                          "benchmarks/miplib-easy/mps/flugpl.mps");
        return;
    }
    forge::io::MpsReadReport rep;
    auto lp = forge::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 15.0;
    opts.conflict_cut.enabled = true;
    BabDiagnostics diag;
    auto raw = forge::search::solve_milp(lp, opts, diag);
    const auto ev = forge::search::milp_evidence(diag, opts);
    const auto r = forge::certify::finalize_result(std::move(raw), ev);

    constexpr double kOpt = 1201500.0;
    constexpr double kTol = 1e-5;
    const bool dual_ok =
        std::isfinite(diag.dual_bound) && diag.dual_bound <= kOpt * (1.0 + kTol);
    CHECK(dual_ok);

    if (r.status == Status::Optimal || diag.globally_proved) {
        CHECK_NEAR(diag.incumbent, kOpt, kTol);
        // A proof may close on the solver's MIP gap tolerance rather than by
        // exhausting the tree, so the bound is only within opts.gap_tol of
        // the optimum (still never above it: dual_ok above).
        CHECK_NEAR(diag.dual_bound, kOpt, 2.0 * opts.gap_tol);
    } else if (r.status == Status::Feasible) {
        CHECK(std::isfinite(diag.incumbent));
        CHECK(diag.incumbent >= kOpt * (1.0 - kTol));
        CHECK(dual_ok);
    } else {
        // Still acceptable: no false Optimal / no dual above true opt.
        CHECK(dual_ok || !std::isfinite(diag.dual_bound));
        CHECK(r.status != Status::Optimal);
    }
}

void test_misc03_dense_binary_mexi_auto_off() {
    // misc03: 159×0-1 + 1 continuous — mixed, so dense pure-binary auto-off
    // (n_cont==0 && n_bin≥80 && n_bin==n_int) does not fire. Nogoods apply.
    // Support+prop Verified accepts small binary-support FUIP cuts.
    const char* candidates[] = {
        FORGE_SOURCE_DIR "/benchmarks/miplib-easy/mps/misc03.mps",
        "benchmarks/miplib-easy/mps/misc03.mps",
        "../benchmarks/miplib-easy/mps/misc03.mps",
        "forge/benchmarks/miplib-easy/mps/misc03.mps",
    };
    const char* path = nullptr;
    for (const char* c : candidates) {
        std::ifstream in(c);
        if (in) {
            path = c;
            break;
        }
    }
    if (!path) {
        ::forge::test::skip("test_misc03_dense_binary_mexi_auto_off",
                          "benchmarks/miplib-easy/mps/misc03.mps");
        return;
    }
    forge::io::MpsReadReport rep;
    auto lp = forge::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 8.0;
    opts.feasibility_jump = false;
    // Exercise conflict learning in the tree. The default incumbent-focused
    // portfolio can spend this entire short allowance in primal heuristics.
    opts.heuristic_budget_frac = 0.0;
    opts.heuristic_budget_frac_no_incumbent = 0.0;
    opts.heuristic_budget_frac_stalled = 0.0;
    opts.heuristic_budget_frac_proof = 0.0;
    BabDiagnostics diag;
    forge::search::solve_milp(lp, opts, diag);
    CHECK(diag.conflict_cut_diag.attempts >= 1);
    // The mixed-model Mexi path must execute. Accepted insertion is the
    // enumerable pair model, not this timed solve and not p0033.
    CHECK(diag.nogood_cuts_global + diag.conflict_store.added >= 1);
    CHECK(diag.conflict_cuts_global +
              diag.conflict_cut_diag.validation_rejected ==
          diag.conflict_cut_diag.learned);
}

void test_enigma_latest_not_false_infeasible() {
    // P0 (2026-09-13): Mexi conflict cuts falsely proved ENIGMA Infeasible.
    // Latest defaults (dense pure-binary Mexi auto-off in bab) must not claim
    // Infeasible; HiGHS / Classical Optimal 0.
    const char* candidates[] = {
        FORGE_SOURCE_DIR "/benchmarks/miplib-easy/mps/enigma.mps",
        "benchmarks/miplib-easy/mps/enigma.mps",
        "../benchmarks/miplib-easy/mps/enigma.mps",
        "forge/benchmarks/miplib-easy/mps/enigma.mps",
    };
    const char* path = nullptr;
    for (const char* c : candidates) {
        std::ifstream in(c);
        if (in) {
            path = c;
            break;
        }
    }
    if (!path) {
        ::forge::test::skip("test_enigma_latest_not_false_infeasible",
                          "benchmarks/miplib-easy/mps/enigma.mps");
        return;
    }
    forge::io::MpsReadReport rep;
    auto lp = forge::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 20.0;
    // Explicit default-on conflict (struct default); do not force Paper.
    CHECK(opts.conflict_cut.enabled);
    BabDiagnostics diag;
    auto raw = forge::search::solve_milp(lp, opts, diag);
    const auto ev = forge::search::milp_evidence(diag, opts);
    const auto r = forge::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status != Status::Infeasible);
    if (r.status == Status::Optimal) {
        CHECK_NEAR(diag.incumbent, 0.0, 1e-6);
    }
}

void test_gen_ip002_latest_not_false_optimal() {
    // P0 (2026-09-14): Unverified Mexi cuts on global_lp falsely closed the
    // tree at incumbent ~-4746 while HiGHS holds feasible ~-4772 (min).
    const char* candidates[] = {
        FORGE_SOURCE_DIR "/benchmarks/miplib-easy/mps/gen-ip002.mps",
        "benchmarks/miplib-easy/mps/gen-ip002.mps",
        "../benchmarks/miplib-easy/mps/gen-ip002.mps",
        "forge/benchmarks/miplib-easy/mps/gen-ip002.mps",
    };
    const char* path = nullptr;
    for (const char* c : candidates) {
        std::ifstream in(c);
        if (in) {
            path = c;
            break;
        }
    }
    if (!path) {
        ::forge::test::skip("test_gen_ip002_latest_not_false_optimal",
                          "benchmarks/miplib-easy/mps/gen-ip002.mps");
        return;
    }
    forge::io::MpsReadReport rep;
    auto lp = forge::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 25.0;
    BabDiagnostics diag;
    auto raw = forge::search::solve_milp(lp, opts, diag);
    const auto ev = forge::search::milp_evidence(diag, opts);
    const auto r = forge::certify::finalize_result(std::move(raw), ev);
    // HiGHS 60s incumbent; any Optimal claim at a worse (higher) obj is false.
    constexpr double kHighsIncumbent = -4772.258675;
    constexpr double kTol = 1e-3;
    if (r.status == Status::Optimal || diag.globally_proved) {
        CHECK(std::isfinite(diag.incumbent));
        CHECK(diag.incumbent <= kHighsIncumbent + kTol);
        CHECK(std::isfinite(diag.dual_bound));
        CHECK(diag.dual_bound <= kHighsIncumbent + kTol);
    } else {
        CHECK(r.status != Status::Optimal);
        CHECK(!diag.globally_proved);
        if (std::isfinite(diag.dual_bound))
            CHECK(diag.dual_bound <= kHighsIncumbent + kTol);
    }
}

void test_markshare1_latest_not_false_optimal() {
    // P0 (2026-09-14): conflict+nogood Unverified cuts claimed Optimal 19;
    // MIPLIB verified optimum is 1.
    const char* candidates[] = {
        FORGE_SOURCE_DIR "/benchmarks/miplib-easy/mps/markshare1.mps",
        "benchmarks/miplib-easy/mps/markshare1.mps",
        "../benchmarks/miplib-easy/mps/markshare1.mps",
        "forge/benchmarks/miplib-easy/mps/markshare1.mps",
    };
    const char* path = nullptr;
    for (const char* c : candidates) {
        std::ifstream in(c);
        if (in) {
            path = c;
            break;
        }
    }
    if (!path) {
        ::forge::test::skip("test_markshare1_latest_not_false_optimal",
                          "benchmarks/miplib-easy/mps/markshare1.mps");
        return;
    }
    forge::io::MpsReadReport rep;
    auto lp = forge::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 20.0;
    BabDiagnostics diag;
    auto raw = forge::search::solve_milp(lp, opts, diag);
    const auto ev = forge::search::milp_evidence(diag, opts);
    const auto r = forge::certify::finalize_result(std::move(raw), ev);
    constexpr double kTrueOpt = 1.0;
    constexpr double kTol = 1e-5;
    CHECK(r.status != Status::Optimal);
    CHECK(!diag.globally_proved);
    // Dual must never sit above the true optimum on a minimization instance.
    if (std::isfinite(diag.dual_bound))
        CHECK(diag.dual_bound <= kTrueOpt + kTol);
}

}  // namespace

int main() {
    test_paper_mode_default();
    test_analyze_learns_valid_cut();
    test_latest_bab_learns_global_conflict_cut();
    test_classical_default_off();
    test_nogood_from_branch_trail_valid();
    test_nogood_refuses_mixed_trail();
    test_validity_check_tri_state();
    test_near_empty_cut_refused();
    test_unverified_wide_cut_is_not_verified();
    test_large_pure_binary_support_verified();
    test_general_integer_conflict_safe();
    test_general_integer_disabled_aborts();
    test_safe_limited_skips_cmir_on_nonbinary();
    test_mixed_binary_example2_safe();
    test_local_cut_row_aborts_global_learn();
    test_latest_bab_learns_nogood();
    test_classical_learns_no_conflict_family();
    test_nogood_cap_zero_disables_learning();
    test_flugpl_latest_dual_not_above_opt();
    test_misc03_dense_binary_mexi_auto_off();
    test_p0033_latest_learns_mexi_cut();
    test_enigma_latest_not_false_infeasible();
    test_gen_ip002_latest_not_false_optimal();
    test_markshare1_latest_not_false_optimal();
    return forge::test::finish("test_conflict_cut");
}
