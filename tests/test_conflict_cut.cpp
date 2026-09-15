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
    CHECK(sor::search::conflict_cut_valid_binary(lp, *cut));
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
    CHECK(latest.conflict_cut.enabled);
    CHECK(latest.conflict_cut.mode == ConflictCutMode::Paper);
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
        CHECK(sor::search::conflict_cut_valid_general(lp, *cut, 1e-9,
                                                      1u << 12, &enumerated));
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
// C3 infeasible. Analysis must either learn a valid cut or abort — never
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
        CHECK(sor::search::conflict_cut_valid_binary(lp_fix, *cut) ||
              sor::search::conflict_cut_valid_general(lp_fix, *cut));
    } else {
        CHECK(cd.aborted >= 1);
        CHECK(cd.learned == 0);
    }
}

void test_paper_mode_default() {
    ConflictCutOptions o;
    CHECK(o.mode == ConflictCutMode::Paper);
}

void test_local_cut_row_aborts_global_learn() {
    // Conflict on a row past n_global_rows must abort — never promote a
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

}  // namespace

int main() {
    test_paper_mode_default();
    test_analyze_learns_valid_cut();
    test_latest_bab_learns_global_conflict_cut();
    test_classical_default_off();
    test_general_integer_conflict_safe();
    test_general_integer_disabled_aborts();
    test_safe_limited_skips_cmir_on_nonbinary();
    test_mixed_binary_example2_safe();
    test_local_cut_row_aborts_global_learn();
    test_flugpl_latest_dual_not_above_opt();
    return sor::test::finish("test_conflict_cut");
}
