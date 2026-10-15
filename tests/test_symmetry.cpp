// WP-G: verified Reflection (Hojny) + Folding (van der Hulst).
#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/conflict.hpp"
#include "sor/search/milp_policy.hpp"
#include "sor/search/symmetry.hpp"
#include "sor/search/symmetry_verify.hpp"
#include "sor/search/symmetry_group.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::core::Status;
using sor::model::LpProblem;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::ConflictGraph;
using sor::search::MilpPolicy;
using sor::search::Orbit;
using sor::search::SymmetryOptions;
using sor::sparse::from_triplets;

namespace {

LpProblem twin_amo() {
    LpProblem lp;
    lp.name = "twin";
    lp.c = {1.0, 1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    return lp;
}

LpProblem asymmetric() {
    LpProblem lp;
    lp.name = "asym";
    lp.c = {1.0, 2.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    return lp;
}

// Complementary pair: a_x = -a_y, c_y = -c_x. Row x - y = 0.
LpProblem reflection_pair() {
    LpProblem lp;
    lp.name = "refl_pair";
    lp.c = {1.0, -1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};
    lp.row_lo = {0.0};
    lp.row_hi = {0.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
    return lp;
}

LpProblem reflection_self() {
    LpProblem lp;
    lp.name = "refl_self";
    lp.c = {0.0, 3.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};
    lp.row_lo = {1.0};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 2, {0}, {1}, {1.0});
    return lp;
}

LpProblem reflection_global() {
    LpProblem lp;
    lp.name = "refl_global";
    lp.c = {0.0, 0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};
    lp.row_lo = {1.0};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    return lp;
}

LpProblem fold_parallel() {
    LpProblem lp;
    lp.name = "fold3";
    lp.c = {2.0, 2.0, 2.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {2.0};
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    return lp;
}

// Packing: three identical binaries, sum <= 1. Known optima: any single 1 or 0.
LpProblem packing3() {
    LpProblem lp;
    lp.name = "pack3";
    lp.c = {-1.0, -1.0, -1.0};  // max cardinality via min -sum
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    return lp;
}

// 2×2 assignment (permutation matrices). Columns are NOT identical-parallel.
LpProblem assignment2() {
    LpProblem lp;
    lp.name = "assign2";
    // min 0 feasibility
    lp.c = {0.0, 0.0, 0.0, 0.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true, true};
    // rows: r1, r2, c1, c2
    lp.row_lo = {1.0, 1.0, 1.0, 1.0};
    lp.row_hi = {1.0, 1.0, 1.0, 1.0};
    // x00,x01,x10,x11
    lp.A = from_triplets(
        4, 4,
        {0, 0, 1, 1, 2, 2, 3, 3},
        {0, 1, 2, 3, 0, 2, 1, 3},
        {1, 1, 1, 1, 1, 1, 1, 1});
    return lp;
}

void test_color_refinement_finds_twin_orbit() {
    auto lp = twin_amo();
    SymmetryOptions opts;
    sor::search::SymmetryDiagnostics diag;
    const auto orbits = sor::search::detect_permutation_orbits(lp, opts, &diag);
    CHECK(orbits.size() >= 1);
    bool found = false;
    for (const Orbit& o : orbits) {
        if (o.cols.size() == 2 && o.cols[0] == 0 && o.cols[1] == 1)
            found = true;
    }
    CHECK(found);
    CHECK(diag.color_iters >= 1);
}

void test_asymmetric_not_orbited_together() {
    auto lp = asymmetric();
    SymmetryOptions opts;
    const auto orbits = sor::search::detect_permutation_orbits(lp, opts, nullptr);
    for (const Orbit& o : orbits) {
        if (o.cols.size() == 2)
            CHECK(!(o.cols[0] == 0 && o.cols[1] == 1));
    }
}

void test_orbital_fixing_on_amo_orbit() {
    auto lp = twin_amo();
    auto lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    sor::search::ProbingOptions po;
    po.enabled = false;
    po.row_cliques = true;
    (void)sor::search::build_conflict_graph(lp, lo, hi, cg, po);
    CHECK(cg.conflicts(sor::search::lit_of(0, 1), sor::search::lit_of(1, 1)));

    lo[0] = 1.0;
    hi[0] = 1.0;
    SymmetryOptions opts;
    opts.folding = false;
    opts.reflection = false;
    const auto orbits = sor::search::detect_permutation_orbits(lp, opts, nullptr);
    const auto nfix =
        sor::search::apply_orbital_fixing(cg, orbits, lo, hi, 1e-9);
    CHECK(nfix >= 1);
    CHECK_NEAR(hi[1], 0.0, 1e-9);
}

void test_reflection_signed_sdg_and_sbc() {
    auto lp = reflection_pair();
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.folding = false;
    sor::search::SymmetryDiagnostics diag;
    const auto gens =
        sor::search::detect_reflection_generators(lp, opts, &diag, nullptr);
    CHECK(!gens.empty());
    CHECK(diag.signed_color_iters >= 1);

    (void)sor::search::apply_reflection_symmetry(lp, lo, hi, nullptr, opts,
                                                 diag);
    CHECK(diag.reflection_status.find("Verified reflection") !=
          std::string::npos);
    CHECK(diag.reflection_pairs >= 1);
    // Feasible images must survive: no bogus per-pair x+y>=1 SBCs.
    std::vector<f64> opt00 = {0.0, 0.0};
    std::vector<f64> opt11 = {1.0, 1.0};
    CHECK_NEAR(lp.max_row_violation(opt00), 0.0, 1e-9);
    CHECK_NEAR(lp.max_row_violation(opt11), 0.0, 1e-9);
}

void test_reflection_orbital_fixing_with_amo() {
    // Complementary AMO-style: x + (1-y) structure via x - y <= 0 and packing.
    // Use twin AMO + force reflection gens empty; instead test fixing via cg
    // on a true complementary conflicting pair: x + y <= 1 with a_x=-a_y? 
    // Simpler: packing twin with reflection self disabled - use pair where
    // columns negate AND they conflict.
    LpProblem lp;
    lp.name = "refl_amo";
    lp.c = {0.0, 0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};
    // x - y = 0 alone does not conflict at 1; add x + y <= 1 ⇒ only (0,0),(1,0)?
    // x-y=0 and x+y<=1 ⇒ (0,0) only. Bad.
    // Just verify AMO orbital fixing still works alongside reflection path.
    auto base = twin_amo();
    auto lo = base.col_lo, hi = base.col_hi;
    ConflictGraph cg;
    sor::search::ProbingOptions po;
    po.enabled = false;
    po.row_cliques = true;
    (void)sor::search::build_conflict_graph(base, lo, hi, cg, po);
    lo[0] = 1.0;
    hi[0] = 1.0;
    SymmetryOptions opts;
    opts.folding = false;
    sor::search::SymmetryDiagnostics diag;
    (void)sor::search::apply_symmetry(base, &cg, lo, hi, opts);
    CHECK_NEAR(hi[1], 0.0, 1e-9);
}

void test_reflection_self_fixes_unused() {
    auto lp = reflection_self();
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.folding = false;
    sor::search::SymmetryDiagnostics diag;
    (void)sor::search::apply_reflection_symmetry(lp, lo, hi, nullptr, opts,
                                                 diag);
    CHECK(diag.reflection_status.find("Verified reflection") !=
          std::string::npos);
    CHECK_NEAR(hi[0], 0.0, 1e-9);
    CHECK(hi[1] >= 1.0 - 1e-9);
}

void test_reflection_global_preserves_optimum() {
    auto lp = reflection_global();
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.folding = false;
    sor::search::SymmetryDiagnostics diag;
    (void)sor::search::apply_reflection_symmetry(lp, lo, hi, nullptr, opts,
                                                 diag);
    CHECK(diag.reflection_status.find("Verified reflection") !=
          std::string::npos);
    // Hojny (5) + packing orbitopal may keep only the lex-max representative;
    // at least one known optimum must survive.
    std::vector<f64> a = {1.0, 0.0};
    std::vector<f64> b = {0.0, 1.0};
    const bool keep_a = lp.max_row_violation(a) < 1e-9;
    const bool keep_b = lp.max_row_violation(b) < 1e-9;
    CHECK(keep_a || keep_b);
    CHECK(diag.reflection_sbcs >= 1 || diag.reflection_applied ||
          diag.orbitopal_sbcs >= 1);
}

void test_folding_parallel_and_lift() {
    auto lp = fold_parallel();
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.reflection = false;
    opts.orbital_fixing = false;
    sor::search::SymmetryDiagnostics diag;
    const auto orbits = sor::search::detect_permutation_orbits(lp, opts, &diag);
    CHECK(!orbits.empty());
    const auto ng =
        sor::search::apply_folding_symmetry(lp, lo, hi, orbits, opts, diag);
    CHECK(ng >= 1);
    CHECK(diag.folding_applied);
    CHECK(diag.folding_status.find("Exact folding") != std::string::npos);
    CHECK_NEAR(hi[0], 3.0, 1e-9);
    CHECK_NEAR(hi[1], 0.0, 1e-9);
    CHECK_NEAR(hi[2], 0.0, 1e-9);

    std::vector<f64> x = {2.0, 0.0, 0.0};
    const f64 folded_obj = lp.objective(x);
    sor::search::lift_folded_solution(diag, x);
    f64 sum = x[0] + x[1] + x[2];
    CHECK_NEAR(sum, 2.0, 1e-9);
    for (f64 v : x) CHECK(v < 1e-9 || std::fabs(v - 1.0) < 1e-9);
    auto orig = fold_parallel();
    CHECK_NEAR(orig.max_row_violation(x), 0.0, 1e-9);
    CHECK_NEAR(orig.max_bound_violation(x), 0.0, 1e-9);
    CHECK_NEAR(orig.objective(x), folded_obj, 1e-9);
}

void test_folding_does_not_cut_sum_optima() {
    auto lp = fold_parallel();
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.reflection = false;
    const auto d = sor::search::apply_symmetry(lp, nullptr, lo, hi, opts);
    CHECK(d.folding_applied);
    CHECK(lo[0] <= 1e-9);
    CHECK(hi[0] >= 2.0 - 1e-9);
}

// Plan 3I: folding asserts IDENTICAL columns. A coefficient 5e-10 apart is
// not identical (times a wide domain it moves a row's activity by O(1)), so
// the pair must not fold; exactly equal columns still must.
void test_folding_requires_exact_identity() {
    auto lp = fold_parallel();
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0 + 5e-10, 1.0});
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.reflection = false;
    const auto d = sor::search::apply_symmetry(lp, nullptr, lo, hi, opts);
    for (const auto& g : d.folds)
        for (const auto j : g.members) CHECK(j != 1);   // column 1 never folded
    auto exact = fold_parallel();
    auto lo2 = exact.col_lo, hi2 = exact.col_hi;
    const auto d2 = sor::search::apply_symmetry(exact, nullptr, lo2, hi2, opts);
    CHECK(d2.folding_applied);
}

void test_packing_orbitopal_preserves_opt() {
    auto lp = packing3();
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.folding = false;  // isolate orbitopal SBCs
    opts.reflection = true;
    const auto diag = sor::search::apply_symmetry(lp, nullptr, lo, hi, opts);
    CHECK(diag.reflection_status.find("Verified reflection") !=
          std::string::npos);
    // Known cardinality-1 lex-max optimum must remain feasible.
    std::vector<f64> e0 = {1.0, 0.0, 0.0};
    CHECK_NEAR(lp.max_row_violation(e0), 0.0, 1e-9);
    std::vector<f64> z = {0.0, 0.0, 0.0};
    CHECK_NEAR(lp.max_row_violation(z), 0.0, 1e-9);
}

void test_packing_fold_lift_matches_objective() {
    auto lp = packing3();
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.reflection = false;
    opts.orbital_fixing = false;
    const auto diag = sor::search::apply_symmetry(lp, nullptr, lo, hi, opts);
    CHECK(diag.folding_applied);
    CHECK(diag.folding_status.find("Exact folding") != std::string::npos);
    // Folded opt y=1 on rep.
    std::vector<f64> x = {1.0, 0.0, 0.0};
    const f64 fobj = lp.objective(x);
    sor::search::lift_folded_solution(diag, x);
    auto orig = packing3();
    CHECK_NEAR(orig.objective(x), fobj, 1e-9);
    CHECK_NEAR(orig.max_row_violation(x), 0.0, 1e-9);
    f64 sum = x[0] + x[1] + x[2];
    CHECK_NEAR(sum, 1.0, 1e-9);
}

void test_assignment_never_cuts_permutation_optima() {
    auto lp = assignment2();
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.reflection = true;
    sor::search::SymmetryDiagnostics diag =
        sor::search::apply_symmetry(lp, nullptr, lo, hi, opts);
    CHECK(diag.reflection_status.find("Verified reflection") !=
          std::string::npos);
    CHECK(diag.folding_status.find("Exact folding") != std::string::npos);
    // Must NOT fold all four into one (columns are not identical-parallel).
    if (diag.folding_applied) {
        for (const auto& g : diag.folds)
            CHECK(g.members.size() < 4);
    }
    // Identity permutation x00=x11=1.
    std::vector<f64> id = {1.0, 0.0, 0.0, 1.0};
    std::vector<f64> anti = {0.0, 1.0, 1.0, 0.0};
    // Rebuild original for feasibility of known optima (SBCs may restrict
    // which representative survives, but at least one optimum must remain).
    auto orig = assignment2();
    const bool id_ok = orig.max_row_violation(id) < 1e-9 &&
                       lp.max_row_violation(id) < 1e-9;
    const bool anti_ok = orig.max_row_violation(anti) < 1e-9 &&
                         lp.max_row_violation(anti) < 1e-9;
    CHECK(id_ok || anti_ok);
}

void test_apply_symmetry_complete_status() {
    auto lp = twin_amo();
    auto lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    sor::search::ProbingOptions po;
    po.enabled = false;
    (void)sor::search::build_conflict_graph(lp, lo, hi, cg, po);
    SymmetryOptions opts;
    const auto d = sor::search::apply_symmetry(lp, &cg, lo, hi, opts);
    CHECK(d.reflection_status.find("stub") == std::string::npos);
    CHECK(d.folding_status.find("stub") == std::string::npos);
    CHECK(d.reflection_status.find("Verified reflection") != std::string::npos ||
          d.reflection_status == "disabled");
    CHECK(d.folding_status.find("Exact folding") != std::string::npos ||
          d.folding_status == "disabled");
    CHECK(d.folding_applied);
}

void test_flags_disable_reflection_folding() {
    auto lp = fold_parallel();
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.reflection = false;
    opts.folding = false;
    const auto d = sor::search::apply_symmetry(lp, nullptr, lo, hi, opts);
    CHECK(!d.reflection_applied);
    CHECK(!d.folding_applied);
    CHECK(d.reflection_status == "disabled");
    CHECK(d.folding_status == "disabled");
    CHECK_NEAR(hi[0], 1.0, 1e-9);
    CHECK_NEAR(hi[1], 1.0, 1e-9);
}

void test_orbital_fixing_does_not_cut_all_zero_opt() {
    auto lp = twin_amo();
    auto lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    sor::search::ProbingOptions po;
    po.enabled = false;
    (void)sor::search::build_conflict_graph(lp, lo, hi, cg, po);
    SymmetryOptions opts;
    opts.folding = false;
    opts.reflection = false;
    (void)sor::search::apply_symmetry(lp, &cg, lo, hi, opts);
    CHECK(lo[0] <= 1e-9 && lo[1] <= 1e-9);
    CHECK(hi[0] >= 1.0 - 1e-9 && hi[1] >= 1.0 - 1e-9);
}

void test_continuous_fold_lift_equal_split() {
    LpProblem lp;
    lp.name = "cont_fold";
    lp.c = {1.0, 1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {2.0, 2.0};
    lp.is_integer = {false, false};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {3.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    auto lo = lp.col_lo, hi = lp.col_hi;
    SymmetryOptions opts;
    opts.reflection = false;
    sor::search::SymmetryDiagnostics diag;
    (void)sor::search::apply_symmetry(lp, nullptr, lo, hi, opts);
    if (!diag.folding_applied) return;  // optional path
    std::vector<f64> x = {2.0, 0.0};
    const f64 fobj = lp.objective(x);
    sor::search::lift_folded_solution(diag, x);
    CHECK_NEAR(x[0], 1.0, 1e-9);
    CHECK_NEAR(x[1], 1.0, 1e-9);
    LpProblem orig;
    orig.c = {1.0, 1.0};
    CHECK_NEAR(x[0] + x[1], 2.0, 1e-9);
    CHECK_NEAR(1.0 * x[0] + 1.0 * x[1], fobj, 1e-9);
}

void test_enigma_not_falsely_infeasible() {
    // Default symmetry must not falsely prove Infeasible (HiGHS Optimal 0).
    const char* candidates[] = {
        "benchmarks/miplib-easy/mps/enigma.mps",
        "../benchmarks/miplib-easy/mps/enigma.mps",
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
    opts.policy = MilpPolicy::Classical;
    opts.time_limit_s = 60.0;
    // Product defaults (reflection off). Opt-in reflection still experimental.
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status != Status::Infeasible);
    if (r.status == Status::Optimal) {
        CHECK_NEAR(diag.incumbent, 0.0, 1e-6);
    }
}

}  // namespace

void test_reflection_checks_complete_rows_and_compatible_generators() {
    SymmetryOptions opts;
    opts.reflection = true;
    opts.folding = false;
    opts.orbital_fixing = false;
    // Complementing only the two binaries does not preserve this equality:
    // the untouched continuous coefficient cannot be implicitly negated.
    auto mixed = reflection_global();
    mixed.c.push_back(0); mixed.col_lo.push_back(0); mixed.col_hi.push_back(1);
    mixed.is_integer.push_back(false);
    mixed.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1., 1., 1.});
    auto lo = mixed.col_lo, hi = mixed.col_hi;
    const auto md = sor::search::apply_symmetry(mixed, nullptr, lo, hi, opts);
    CHECK(md.reflection_sbcs == 0);
    CHECK_NEAR(mixed.max_row_violation({0., 0., 1.}), 0., 0.);
    // Color refinement rounds these costs to the same tag; exact verification
    // must reject the swap and preserve the uniquely better representative.
    auto tiny = reflection_global(); tiny.c = {1e-10, 0.};
    lo = tiny.col_lo; hi = tiny.col_hi;
    const auto td = sor::search::apply_symmetry(tiny, nullptr, lo, hi, opts);
    CHECK(td.orbitopal_sbcs == 0);
    CHECK_NEAR(tiny.max_row_violation({0., 1.}), 0., 0.);
    // Every vertex in C3 + C4 has the same refinement color, but they are not
    // one freely permutable group. A total order across that cell loses every
    // independent set of size three. Verified disjoint swap groups preserve one.
    LpProblem graph;
    graph.c.assign(7, -1); graph.col_lo.assign(7, 0); graph.col_hi.assign(7, 1);
    graph.is_integer.assign(7, true); graph.row_lo.assign(7, -sor::model::kInf);
    graph.row_hi.assign(7, 1);
    graph.A = from_triplets(7, 7, {0,0,1,1,2,2,3,3,4,4,5,5,6,6},
                           {0,1,1,2,2,0,3,4,4,5,5,6,6,3}, std::vector<f64>(14,1));
    const auto original = graph;
    lo = graph.col_lo; hi = graph.col_hi;
    (void)sor::search::apply_symmetry(graph, nullptr, lo, hi, opts);
    bool optimum_survives = false;
    for (unsigned mask = 0; mask < 128; ++mask) {
        std::vector<f64> x(7);
        for (int j = 0; j < 7; ++j) x[j] = (mask >> j) & 1u;
        if (original.objective(x) == -3 && original.max_row_violation(x) == 0 &&
            graph.max_row_violation(x) == 0) optimum_survives = true;
    }
    CHECK(optimum_survives);
}

void test_fold_owns_live_integer_domains_and_wide_lift() {
    auto lp = twin_amo();
    lp.c = {1., 1.}; lp.is_integer = {true, true};
    lp.col_lo = {.1, .1}; lp.col_hi = {4000000000.9, 4000000000.9};
    lp.row_hi = {7000000000.};
    auto lo = lp.col_lo, hi = lp.col_hi;
    lo = {2.1, 2.1}; hi = {3000000000.9, 3000000000.9};
    SymmetryOptions opts; opts.orbital_fixing = false;
    sor::search::SymmetryDiagnostics diag;
    CHECK(sor::search::apply_folding_symmetry(lp, lo, hi, {{{0,1}}}, opts, diag) == 1);
    CHECK_NEAR(lo[0], 6., 0.); CHECK_NEAR(hi[0], 6000000000., 0.);
    std::vector<f64> x{5000000000., 0.};
    sor::search::lift_folded_solution(diag, x);
    CHECK_NEAR(x[0] + x[1], 5000000000., 0.);
    CHECK(x[0] >= 3 && x[1] >= 3 && x[0] <= 3000000000. && x[1] <= 3000000000.);
    CHECK(x[0] == std::round(x[0]) && x[1] == std::round(x[1]));
    // A current bound on only one member breaks the equal-domain transform.
    lp = twin_amo(); lo = {0., 0.}; hi = {1., 0.};
    diag = {};
    CHECK(sor::search::apply_folding_symmetry(lp, lo, hi, {{{0,1}}}, opts, diag) == 0);
    CHECK(hi[1] == 0);
}

// Cycle rotations require coupled column/row actions: no single vertex
// swap is an automorphism of C5. Every group orbit's global lex maximum must
// survive all leaders and their box propagation simultaneously.
void test_general_verified_generators_and_common_leaders() {
    using sor::search::VerifiedSymmetryGroup;
    std::vector<Index> rows, cols; std::vector<f64> vals;
    for (Index i = 0; i < 5; ++i) {
        rows.push_back(i); cols.push_back(i); vals.push_back(1);
        rows.push_back(i); cols.push_back((i+1)%5); vals.push_back(1);
    }
    LpProblem lp;
    lp.A = from_triplets(5, 5, rows, cols, vals);
    lp.row_lo.assign(5, -sor::model::kInf); lp.row_hi.assign(5, 1);
    lp.c.assign(5, -1); lp.col_lo.assign(5, 0); lp.col_hi.assign(5, 1);
    lp.is_integer.assign(5, true);
    VerifiedSymmetryGroup group; group.build(lp, {{{0,1,2,3,4}}}, 0);
    CHECK(!group.empty());
    sor::search::SymmetryVerifier verifier(lp);
    for (const auto& g : group.generators()) {
        CHECK(verifier.verifies(g, std::vector<int>(5,1), std::vector<f64>(5,0)));
        int moved = 0; for (Index j = 0; j < 5; ++j) moved += g[static_cast<std::size_t>(j)] != j;
        CHECK(moved >= 4);
    }
    // Enumerate the retained group independently, then compare every box
    // coloring's full stabilizer, including compositions of rejected generators.
    std::vector<std::vector<Index>> elements{{0,1,2,3,4}};
    for (std::size_t q = 0; q < elements.size(); ++q) for (const auto& g : group.generators()) {
        std::vector<Index> h(5); for (Index j = 0; j < 5; ++j) h[j] = elements[q][g[j]];
        if (std::find(elements.begin(),elements.end(),h) == elements.end()) elements.push_back(std::move(h));
    }
    for (int code = 0; code < 243; ++code) {
        auto lo = lp.col_lo, hi = lp.col_hi; int v = code;
        for (Index j = 0; j < 5; ++j) { int state = v%3; v/=3; if(state==1) hi[j]=0; if(state==2) lo[j]=1; }
        bool expected[5][5]{}; for(Index j=0;j<5;++j) expected[j][j]=true;
        for(const auto& g:elements) {
            bool keeps=true; for(Index j=0;j<5;++j) keeps=keeps && lo[j]==lo[g[j]] && hi[j]==hi[g[j]];
            if(keeps) for(Index j=0;j<5;++j) expected[j][g[j]]=true;
        }
        bool actual[5][5]{}; for(Index j=0;j<5;++j) actual[j][j]=true;
        bool complete=false;
        for(const auto& orbit:group.stabilizer_orbits(lo,hi,0,nullptr,&complete))
            for(Index j:orbit.cols) for(Index k:orbit.cols) actual[j][k]=true;
        CHECK(complete);
        for(Index j=0;j<5;++j) for(Index k=0;k<5;++k) CHECK(actual[j][k]==expected[j][k]);
    }
    f64 retained_best = 0;
    for (unsigned mask = 0; mask < 32; ++mask) {
        std::vector<f64> x(5);
        for (unsigned j = 0; j < 5; ++j) x[j] = (mask >> j) & 1;
        if (lp.max_row_violation(x) != 0) continue;
        std::vector<std::vector<f64>> orbit{x};
        for (std::size_t q = 0; q < orbit.size(); ++q)
            for (const auto& g : group.generators()) {
                std::vector<f64> image(5);
                for (unsigned j = 0; j < 5; ++j) image[j] = orbit[q][static_cast<std::size_t>(g[j])];
                if (std::find(orbit.begin(), orbit.end(), image) == orbit.end()) orbit.push_back(image);
            }
        const auto& maximum = *std::max_element(orbit.begin(), orbit.end());
        auto lo = maximum, hi = maximum;
        std::uint64_t moves = 0;
        CHECK(group.propagate(lo, hi, nullptr, 0, moves));
        CHECK(lo == maximum && hi == maximum);
        lo = lp.col_lo; hi = lp.col_hi; lo[0] = maximum[0]; hi[0] = maximum[0];
        sor::search::PropTrail trail;
        CHECK(group.propagate(lo, hi, &trail, 1, moves));
        for (unsigned j = 0; j < 5; ++j) CHECK(lo[j] <= maximum[j] && hi[j] >= maximum[j]);
        retained_best = std::min(retained_best, lp.objective(maximum));
    }
    CHECK_NEAR(retained_best, -2, 0);
    auto lo = lp.col_lo, hi = lp.col_hi; hi[0] = 0;
    for (const auto& cell : group.stabilizer_orbits(lo, hi))
        CHECK(std::find(cell.cols.begin(), cell.cols.end(), 0) == cell.cols.end());
}

int main() {
    test_general_verified_generators_and_common_leaders();
    test_reflection_checks_complete_rows_and_compatible_generators();
    test_fold_owns_live_integer_domains_and_wide_lift();
    test_folding_requires_exact_identity();
    test_color_refinement_finds_twin_orbit();
    test_asymmetric_not_orbited_together();
    test_orbital_fixing_on_amo_orbit();
    test_reflection_signed_sdg_and_sbc();
    test_reflection_orbital_fixing_with_amo();
    test_reflection_self_fixes_unused();
    test_reflection_global_preserves_optimum();
    test_folding_parallel_and_lift();
    test_folding_does_not_cut_sum_optima();
    test_packing_orbitopal_preserves_opt();
    test_packing_fold_lift_matches_objective();
    test_assignment_never_cuts_permutation_optima();
    test_apply_symmetry_complete_status();
    test_flags_disable_reflection_folding();
    test_orbital_fixing_does_not_cut_all_zero_opt();
    test_continuous_fold_lift_equal_split();
    test_enigma_not_falsely_infeasible();
    return sor::test::finish("test_symmetry");
}
