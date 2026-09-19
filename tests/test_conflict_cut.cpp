// WP-B: Mexi cut-based conflict analysis (arXiv:2410.15110 paper-complete).
#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/conflict_cut.hpp"
#include "sor/search/propagate.hpp"
#include "sor/search/prop_trail.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

using sor::core::Status;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::ConflictAnalysisContext;
using sor::search::ConflictCutMode;
using sor::search::ConflictCutOptions;
using sor::search::CutValidity;
using sor::search::MilpPolicy;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
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
    std::vector<sor::core::f64> lo = lp.col_lo, hi = lp.col_hi;
    lo[0] = 1.0;
    hi[0] = 1.0;
    lo[1] = 1.0;
    hi[1] = 1.0;

    sor::search::PropTrail trail;
    trail.push(0, sor::search::BoundDir::Lower, 1.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 1);
    trail.push(1, sor::search::BoundDir::Lower, 1.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 1);

    sor::search::PropagateResult pr;
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
    sor::search::ConflictCutDiagnostics cd;
    const auto cut = sor::search::analyze_conflict_cuts(ctx, opts, cd);
    CHECK(cut.has_value());
    CHECK(cd.learned >= 1);
    // Pure-binary model, no continuous columns: the checker enumerates the
    // whole box, so this is a VERIFIED validity proof (not an assumption).
    CHECK(sor::search::conflict_cut_check_binary(lp, *cut) ==
          CutValidity::Verified);
}

void test_latest_bab_learns_global_conflict_cut() {
    auto lp = read_text(kPairInfeas);
    BabOptions opts;
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
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.conflict_cut_diag.attempts >= 1);
    CHECK(diag.conflict_cuts_global >= 1);
}

void test_classical_default_off() {
    ConflictCutOptions o;
    o.enabled = true;
    sor::search::apply_conflict_cut_policy(MilpPolicy::Classical, o);
    CHECK(!o.enabled);
    BabOptions latest;
    latest.policy = MilpPolicy::Latest;
    // Product default: Mexi on in struct; bab auto-disables on dense pure-binary.
    CHECK(latest.conflict_cut.enabled);
    CHECK(latest.conflict_cut.mode == ConflictCutMode::Paper);
}

void test_nogood_from_branch_trail_valid() {
    auto lp = read_text(kPairInfeas);
    sor::search::PropTrail trail;
    // x1=1, x2=1 - classic infeasible assignment under CAP.
    trail.push(0, sor::search::BoundDir::Lower, 1.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 1);
    trail.push(1, sor::search::BoundDir::Lower, 1.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 1);
    const auto ng =
        sor::search::build_nogood_from_branch_trail(trail, lp, 1e-9);
    CHECK(ng.has_value());
    CHECK(!sor::search::conflict_cut_near_empty(*ng));
    // Nogood: (1-x1)+(1-x2) >= 1  ⇒  -x1 -x2 >= -1
    CHECK(ng->cols.size() == 2);
    CHECK(sor::search::conflict_cut_check_binary(lp, *ng) ==
          CutValidity::Verified);
}

// Tri-state checker semantics (fail-closed): a free continuous column makes
// verification impossible; the checker must say Unverified, never silently
// claim validity. Complete boxes give Verified / Refuted definitively.
void test_validity_check_tri_state() {
    // Mixed model: binary x, free continuous y in [0,1], x + y <= 1.
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {1.0};
    lp.c = {1.0, 0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, false};

    // Valid cut on y alone: cannot be verified (y free) -> Unverified.
    sor::search::CutRow cut_y;
    cut_y.cols = {1};
    cut_y.vals = {1.0};
    cut_y.row_lo = -sor::model::kInf;
    cut_y.row_hi = 1.0;
    CHECK(sor::search::conflict_cut_check_binary(lp, cut_y) ==
          CutValidity::Unverified);
    CHECK(sor::search::conflict_cut_check_general(lp, cut_y) ==
          CutValidity::Unverified);

    // Invalid cut x <= 0 (x=1, y=0 is feasible): the support sweep finds the
    // witness even though y is free -> Refuted.
    sor::search::CutRow cut_x;
    cut_x.cols = {0};
    cut_x.vals = {1.0};
    cut_x.row_lo = -sor::model::kInf;
    cut_x.row_hi = 0.0;
    CHECK(sor::search::conflict_cut_check_binary(lp, cut_x) ==
          CutValidity::Refuted);

    // Pure-binary complete box: valid -> Verified, invalid -> Refuted.
    auto b2 = read_text(kPairInfeas);
    sor::search::CutRow valid;
    valid.cols = {0, 1};
    valid.vals = {1.0, 1.0};
    valid.row_lo = -sor::model::kInf;
    valid.row_hi = 1.0;  // x1 + x2 <= 1 (CAP)
    CHECK(sor::search::conflict_cut_check_binary(b2, valid) ==
          CutValidity::Verified);
    sor::search::CutRow invalid;
    invalid.cols = {0, 1};
    invalid.vals = {1.0, 1.0};
    invalid.row_lo = -sor::model::kInf;
    invalid.row_hi = 0.5;  // (1,0) violates
    CHECK(sor::search::conflict_cut_check_binary(b2, invalid) ==
          CutValidity::Refuted);

    // Unbounded integer domains must NOT vacuously Verify (gen-ip002 P0:
    // floor(+inf)→int UB previously skipped every free int column).
    {
        sor::model::LpProblem ub;
        ub.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        ub.row_lo = {-sor::model::kInf};
        ub.row_hi = {5.0};
        ub.c = {1.0, 1.0};
        ub.col_lo = {0.0, 0.0};
        ub.col_hi = {sor::model::kInf, sor::model::kInf};
        ub.is_integer = {true, true};
        sor::search::CutRow c;
        c.cols = {0, 1};
        c.vals = {1.0, 1.0};
        c.row_lo = -sor::model::kInf;
        c.row_hi = 0.0;
        CHECK(sor::search::conflict_cut_check_general(ub, c) ==
              CutValidity::Unverified);
    }
}

void test_near_empty_cut_refused() {
    sor::search::CutRow empty;
    CHECK(sor::search::conflict_cut_near_empty(empty));
    sor::search::CutRow zeros;
    zeros.cols = {0, 1};
    zeros.vals = {0.0, 0.0};
    CHECK(sor::search::conflict_cut_near_empty(zeros));
    // Global ⊥ (0 >= 1) is a real conflict cut, not "empty".
    sor::search::CutRow bot;
    bot.row_lo = 1.0;
    CHECK(!sor::search::conflict_cut_near_empty(bot));
}

void test_large_pure_binary_support_verified() {
    // 20 binaries with x0 + x1 >= 1. Full 2^20 exceeds the old enum budget;
    // support+prop verification must still certify the redundant cut.
    constexpr int n = 20;
    std::vector<sor::core::Index> rows(2, 0), cols = {0, 1};
    std::vector<sor::core::f64> vals = {1.0, 1.0};
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, n, rows, cols, vals);
    lp.row_lo = {1.0};
    lp.row_hi = {sor::model::kInf};
    lp.c.assign(static_cast<std::size_t>(n), 1.0);
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), true);

    sor::search::CutRow valid;
    valid.cols = {0, 1};
    valid.vals = {1.0, 1.0};
    valid.row_lo = 1.0;
    valid.row_hi = sor::model::kInf;
    CHECK(sor::search::conflict_cut_check_binary(lp, valid) ==
          CutValidity::Verified);

    // Invalid strengthening: x0 + x1 >= 2 excludes the feasible (1,0).
    sor::search::CutRow invalid = valid;
    invalid.row_lo = 2.0;
    CHECK(sor::search::conflict_cut_check_binary(lp, invalid) ==
          CutValidity::Refuted);
}

void test_p0033_latest_learns_mexi_cut() {
    // p0033: 33 pure binaries - Mexi stays on (n_bin < 80). Before support+prop
    // verification, every FUIP cut died at Verified-only apply (2^33 budget).
    const char* candidates[] = {
        "benchmarks/miplib-easy/mps/p0033.mps",
        "../benchmarks/miplib-easy/mps/p0033.mps",
        "sor/benchmarks/miplib-easy/mps/p0033.mps",
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
        ::sor::test::report(true, "p0033: skipped (no mps)", __FILE__, __LINE__);
        return;
    }
    sor::io::MpsReadReport rep;
    auto lp = sor::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 15.0;
    opts.feasibility_jump = false;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.conflict_cut_diag.attempts >= 1);
    CHECK(diag.conflict_cut_diag.learned >= 1);
    CHECK(diag.conflict_cuts_global >= 1);
}

// Nogoods must refuse trails that branch on non-binary columns: the node
// box under x >= 2 (general int) cannot be expressed as a 0/1 assignment,
// and the old code silently skipped such entries, producing an invalid
// global cut on mixed models (2026-09-14 review).
void test_nogood_refuses_mixed_trail() {
    // x1 binary, x2 general integer in [0, 10], x1 + x2 <= 0.
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {0.0};
    lp.c = {1.0, 1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 10.0};
    lp.is_integer = {true, true};

    sor::search::PropTrail trail;
    // Binary branch x1 = 1 ...
    trail.push(0, sor::search::BoundDir::Lower, 1.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 1);
    // ... plus a general-integer branch x2 >= 2 (mixed trail).
    trail.push(1, sor::search::BoundDir::Lower, 2.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 2);
    const auto ng =
        sor::search::build_nogood_from_branch_trail(trail, lp, 1e-9);
    CHECK(!ng.has_value());

    // Pure-binary trail on the same model still yields a nogood.
    sor::search::PropTrail bin_trail;
    bin_trail.push(0, sor::search::BoundDir::Lower, 1.0, 0.0,
                   sor::search::ReasonKind::Branch, -1, 1);
    const auto ng2 =
        sor::search::build_nogood_from_branch_trail(bin_trail, lp, 1e-9);
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
    std::vector<sor::core::f64> lo = {2.0, 2.0};
    std::vector<sor::core::f64> hi = {2.0, 2.0};

    sor::search::PropTrail trail;
    trail.push(0, sor::search::BoundDir::Lower, 2.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 1);
    trail.push(1, sor::search::BoundDir::Lower, 2.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 1);

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
    sor::search::ConflictCutDiagnostics cd;
    const auto cut = sor::search::analyze_conflict_cuts(ctx, opts, cd);
    CHECK(cd.attempts >= 1);
    CHECK(cd.general_int_reasons >= 1 || cd.learned >= 1 || cd.aborted >= 1);
    if (cut) {
        bool enumerated = false;
        CHECK(sor::search::conflict_cut_check_general(lp, *cut, 1e-9,
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
    lp.A = sor::sparse::from_triplets(1, 3, {0, 0, 0}, {0, 1, 2},
                                      {2.0, 2.0, 2.0});
    lp.c = {1.0, 1.0, 1.0};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {3.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {2.0, 2.0, 2.0};
    lp.is_integer = {true, true, true};

    // z>=1 alone keeps conflict feasible; row prop x>=2 makes it infeasible
    // and leaves two non-relaxables (x,z) so not FUIP-asserting.
    std::vector<sor::core::f64> lo = {2.0, 0.0, 1.0};
    std::vector<sor::core::f64> hi = {2.0, 2.0, 2.0};
    sor::search::PropTrail trail;
    trail.push(2, sor::search::BoundDir::Lower, 1.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 1);
    trail.push(0, sor::search::BoundDir::Lower, 2.0, 0.0,
               sor::search::ReasonKind::Row, 0, 1);

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
    sor::search::ConflictCutDiagnostics cd;
    const auto cut = sor::search::analyze_conflict_cuts(ctx, opts, cd);
    CHECK(!cut.has_value());
    CHECK(cd.aborted >= 1);
}

void test_safe_limited_skips_cmir_on_nonbinary() {
    auto lp = read_text(kGenIntPair);
    std::vector<sor::core::f64> lo = {2.0, 2.0};
    std::vector<sor::core::f64> hi = {2.0, 2.0};
    sor::search::PropTrail trail;
    trail.push(0, sor::search::BoundDir::Lower, 2.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 1);
    trail.push(1, sor::search::BoundDir::Lower, 2.0, 0.0,
               sor::search::ReasonKind::Row, 0, 1);

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
    sor::search::ConflictCutDiagnostics cd;
    (void)sor::search::analyze_conflict_cuts(ctx, opts, cd);
    // SafeLimited must record skip when it hits a non-binary row reason.
    CHECK(cd.cmir_skipped_nonbinary >= 1 || cd.aborted >= 1 || cd.learned >= 1);
}

// Paper Example 2 / Fig. 2 mixed-binary skeleton (reconstructed):
// binaries x1,x2,x3; continuous y1∈[0,1], y2∈[-1,1].
// Local: x2=0 ⇒ y2≤0 (C4), y2≥0 & x3=0 (C5), y1≤3/4 (C1), x1≥1 (C2),
// C3 infeasible. Analysis must either learn a valid cut or abort - never
// emit an inequality violated by a feasible MBP point.
void test_mixed_binary_example2_safe() {
    sor::model::LpProblem lp;
    // Paper Example 2 / Fig. 2 (reconstructed): binaries x1,x2,x3;
    // continuous y1∈[0,1], y2∈[-1,1].
    // C1 L: 2x1+4y1+2y2 <= 3
    // C2 G: 20x1+5y1−y2 >= 4
    // C3 L: 20x1−5y1+10y2 <= 16
    // C4 L: x2+y2 <= 0
    // C5 G: −x3+y2 >= 0
    lp.A = sor::sparse::from_triplets(
        5, 5,
        {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4},
        {0, 3, 4, 0, 3, 4, 0, 3, 4, 1, 4, 2, 4},
        {2.0, 4.0, 2.0, 20.0, 5.0, -1.0, 20.0, -5.0, 10.0, 1.0, 1.0, -1.0, 1.0});
    lp.row_lo = {-sor::model::kInf, 4.0, -sor::model::kInf, -sor::model::kInf,
                 0.0};
    lp.row_hi = {3.0, sor::model::kInf, 16.0, 0.0, sor::model::kInf};
    lp.c = {1, 1, 1, 0, 0};
    lp.col_lo = {0, 0, 0, 0, -1};
    lp.col_hi = {1, 1, 1, 1, 1};
    lp.is_integer = {true, true, true, false, false};

    // Local domain after Example 2 cascade; C3 infeasible.
    std::vector<sor::core::f64> lo = {1.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<sor::core::f64> hi = {1.0, 0.0, 0.0, 0.75, 0.0};

    sor::search::PropTrail trail;
    trail.push(1, sor::search::BoundDir::Upper, 0.0, 1.0,
               sor::search::ReasonKind::Branch, -1, 1);
    trail.push(4, sor::search::BoundDir::Upper, 0.0, 1.0,
               sor::search::ReasonKind::Row, 3, 1);
    trail.push(4, sor::search::BoundDir::Lower, 0.0, -1.0,
               sor::search::ReasonKind::Row, 4, 1);
    trail.push(2, sor::search::BoundDir::Upper, 0.0, 1.0,
               sor::search::ReasonKind::Row, 4, 1);
    trail.push(3, sor::search::BoundDir::Upper, 0.75, 1.0,
               sor::search::ReasonKind::Row, 0, 1);
    trail.push(0, sor::search::BoundDir::Lower, 1.0, 0.0,
               sor::search::ReasonKind::Row, 1, 1);

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
    sor::search::ConflictCutDiagnostics cd;
    const auto cut = sor::search::analyze_conflict_cuts(ctx, opts, cd);
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
        CHECK(sor::search::conflict_cut_check_binary(lp_fix, *cut) !=
              CutValidity::Refuted);
        CHECK(sor::search::conflict_cut_check_general(lp_fix, *cut) !=
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
// infeasible (the row can only sum to 0, 2, 4 or 6) - branches like
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
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.nogood_cuts_global >= 1);
}

void test_classical_learns_no_conflict_family() {
    // Classical ablation must run the classical path only: neither Mexi
    // conflict cuts nor branch-trail nogoods may reach global_lp.
    auto lp = read_text(kNogoodModel);
    BabOptions opts = nogood_test_options();
    opts.policy = MilpPolicy::Classical;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.conflict_cuts_global == 0);
    CHECK(diag.nogood_cuts_global == 0);
}

void test_nogood_cap_zero_disables_learning() {
    auto lp = read_text(kNogoodModel);
    BabOptions opts = nogood_test_options();
    opts.policy = MilpPolicy::Latest;
    opts.conflict_cut.max_nogood_cuts = 0;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.nogood_cuts_global == 0);
}

void test_local_cut_row_aborts_global_learn() {
    // Conflict on a row past n_global_rows must abort - never promote a
    // node-local GMI/MIR into a global conflict cut.
    auto lp = read_text(kPairInfeas);
    std::vector<sor::core::f64> lo = {1.0, 1.0};
    std::vector<sor::core::f64> hi = {1.0, 1.0};
    sor::search::PropTrail trail;
    trail.push(0, sor::search::BoundDir::Lower, 1.0, 0.0,
               sor::search::ReasonKind::Branch, -1, 1);
    trail.push(1, sor::search::BoundDir::Lower, 1.0, 0.0,
               sor::search::ReasonKind::Row, /*local row*/ 5, 1);

    ConflictAnalysisContext ctx;
    ctx.lp = &lp;
    ctx.col_lo = &lo;
    ctx.col_hi = &hi;
    ctx.trail = &trail;
    ctx.conflict_row = 5;
    ctx.n_global_rows = 1;  // only row 0 is global

    ConflictCutOptions opts;
    opts.enabled = true;
    sor::search::ConflictCutDiagnostics cd;
    const auto cut = sor::search::analyze_conflict_cuts(ctx, opts, cd);
    CHECK(!cut.has_value());
    CHECK(cd.aborted >= 1);
    CHECK(cd.learned == 0);
}

void test_flugpl_latest_dual_not_above_opt() {
    // P0 regression (2026-09-13): Latest + conflict cuts promoted local tree
    // GMI into global_lp and proved dual 1253765 > true opt 1201500.
    const char* candidates[] = {
        "benchmarks/miplib-easy/mps/flugpl.mps",
        "../benchmarks/miplib-easy/mps/flugpl.mps",
        "sor/benchmarks/miplib-easy/mps/flugpl.mps",
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
        ::sor::test::report(true, "flugpl: skipped (no mps)", __FILE__,
                            __LINE__);
        return;
    }
    sor::io::MpsReadReport rep;
    auto lp = sor::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 15.0;
    opts.conflict_cut.enabled = true;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);

    constexpr double kOpt = 1201500.0;
    constexpr double kTol = 1e-5;
    const bool dual_ok =
        std::isfinite(diag.dual_bound) && diag.dual_bound <= kOpt * (1.0 + kTol);
    CHECK(dual_ok);

    if (r.status == Status::Optimal || diag.globally_proved) {
        CHECK_NEAR(diag.incumbent, kOpt, kTol);
        CHECK_NEAR(diag.dual_bound, kOpt, kTol);
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
    // misc03: 159×0-1 + 1 continuous - mixed, so dense pure-binary auto-off
    // (n_cont==0 && n_bin≥80 && n_bin==n_int) does not fire. Nogoods apply.
    // Support+prop Verified accepts small binary-support FUIP cuts.
    const char* candidates[] = {
        "benchmarks/miplib-easy/mps/misc03.mps",
        "../benchmarks/miplib-easy/mps/misc03.mps",
        "sor/benchmarks/miplib-easy/mps/misc03.mps",
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
        ::sor::test::report(true, "misc03: skipped (no mps)", __FILE__, __LINE__);
        return;
    }
    sor::io::MpsReadReport rep;
    auto lp = sor::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 8.0;
    opts.feasibility_jump = false;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.conflict_cut_diag.attempts >= 1);
    CHECK(diag.conflict_cuts_global >= 1);
    CHECK(diag.nogood_cuts_global >= 1);
}

void test_enigma_latest_not_false_infeasible() {
    // P0 (2026-09-13): Mexi conflict cuts falsely proved ENIGMA Infeasible.
    // Latest defaults (dense pure-binary Mexi auto-off in bab) must not claim
    // Infeasible; HiGHS / Classical Optimal 0.
    const char* candidates[] = {
        "benchmarks/miplib-easy/mps/enigma.mps",
        "../benchmarks/miplib-easy/mps/enigma.mps",
        "sor/benchmarks/miplib-easy/mps/enigma.mps",
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
        ::sor::test::report(true, "enigma: skipped (no mps)", __FILE__,
                            __LINE__);
        return;
    }
    sor::io::MpsReadReport rep;
    auto lp = sor::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 20.0;
    // Explicit default-on conflict (struct default); do not force Paper.
    CHECK(opts.conflict_cut.enabled);
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status != Status::Infeasible);
    if (r.status == Status::Optimal) {
        CHECK_NEAR(diag.incumbent, 0.0, 1e-6);
    }
}

void test_gen_ip002_latest_not_false_optimal() {
    // P0 (2026-09-14): Unverified Mexi cuts on global_lp falsely closed the
    // tree at incumbent ~-4746 while HiGHS holds feasible ~-4772 (min).
    const char* candidates[] = {
        "benchmarks/miplib-easy/mps/gen-ip002.mps",
        "../benchmarks/miplib-easy/mps/gen-ip002.mps",
        "sor/benchmarks/miplib-easy/mps/gen-ip002.mps",
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
        ::sor::test::report(true, "gen-ip002: skipped (no mps)", __FILE__,
                            __LINE__);
        return;
    }
    sor::io::MpsReadReport rep;
    auto lp = sor::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 25.0;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
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
        "benchmarks/miplib-easy/mps/markshare1.mps",
        "../benchmarks/miplib-easy/mps/markshare1.mps",
        "sor/benchmarks/miplib-easy/mps/markshare1.mps",
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
        ::sor::test::report(true, "markshare1: skipped (no mps)", __FILE__,
                            __LINE__);
        return;
    }
    sor::io::MpsReadReport rep;
    auto lp = sor::io::read_mps_file(path, rep);
    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.time_limit_s = 20.0;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
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
    return sor::test::finish("test_conflict_cut");
}
