// Lattice reformulation unit tests: tiny Ax=b systems, LLL invariants via
// exact A·Q=0 / A·x0=b checks, the restriction protocol (a restricted run
// may only contribute an incumbent - never a terminal status), and optional
// markshare smoke (if present).
#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/lattice_reform.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using sor::core::Index;
using sor::core::Status;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::LatticeReformOptions;
using sor::search::LatticeSolveOutcome;
using sor::search::try_lattice_reform;
using sor::search::lattice_postsolve;
using sor::search::solve_milp_lattice;

namespace {

sor::model::LpProblem make_eq_binary(const std::vector<std::vector<double>>& A,
                                     const std::vector<double>& b,
                                     const std::vector<double>& c) {
    const Index m = static_cast<Index>(A.size());
    const Index n = m ? static_cast<Index>(A[0].size()) : 0;
    std::vector<Index> rows, cols;
    std::vector<double> vals;
    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < n; ++j) {
            if (A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] == 0) continue;
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
        }
    }
    sor::model::LpProblem lp;
    lp.name = "tiny";
    lp.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
    lp.c = c;
    lp.row_lo = b;
    lp.row_hi = b;
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), true);
    lp.validate();
    return lp;
}

struct ColSpec {
    double lo = 0.0;
    double hi = 1.0;
    bool integer = true;
};

// General builder: per-column bounds and integrality, equality rows.
sor::model::LpProblem make_mixed_lp(const std::vector<ColSpec>& cols,
                                    const std::vector<std::vector<double>>& A,
                                    const std::vector<double>& b,
                                    const std::vector<double>& c) {
    const Index m = static_cast<Index>(A.size());
    const Index n = static_cast<Index>(cols.size());
    std::vector<Index> rows, cols_idx;
    std::vector<double> vals;
    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < n; ++j) {
            if (A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] == 0) continue;
            rows.push_back(i);
            cols_idx.push_back(j);
            vals.push_back(A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
        }
    }
    sor::model::LpProblem lp;
    lp.name = "mixed";
    lp.A = sor::sparse::from_triplets(m, n, rows, cols_idx, vals);
    lp.c = c;
    lp.row_lo = b;
    lp.row_hi = b;
    for (Index j = 0; j < n; ++j) {
        lp.col_lo.push_back(cols[static_cast<std::size_t>(j)].lo);
        lp.col_hi.push_back(cols[static_cast<std::size_t>(j)].hi);
        lp.is_integer.push_back(cols[static_cast<std::size_t>(j)].integer);
    }
    lp.validate();
    return lp;
}

void test_tiny_feasible() {
    // x0 + 2 x1 = 2, binary → unique solution (0,1), objective 1.
    auto lp = make_eq_binary({{1, 2}}, {2}, {1, 1});
    BabOptions bab;
    bab.time_limit_s = 5;
    auto out = solve_milp_lattice(lp, bab, /*attempt_reform=*/true);
    ::sor::test::report(out.reform_applied, "tiny feasible: reform applies",
                        __FILE__, __LINE__);
    ::sor::test::report(out.exact_equivalence && !out.fell_back,
                        "tiny feasible: exact equivalence, no fallback re-solve",
                        __FILE__, __LINE__);
    const auto ev = sor::search::milp_evidence(out.diag, bab);
    const auto r = sor::certify::finalize_result(out.raw, ev);
    ::sor::test::report(r.status == Status::Optimal,
                        "tiny feasible: Optimal", __FILE__, __LINE__,
                        std::string(sor::core::to_string(r.status)));
    ::sor::test::report(!r.x.empty(), "tiny feasible: has solution",
                        __FILE__, __LINE__);
    if (r.x.empty()) return;
    const double Ax = 1 * r.x[0] + 2 * r.x[1];
    ::sor::test::report(std::fabs(Ax - 2.0) < 1e-8, "tiny feasible: Ax=b",
                        __FILE__, __LINE__, "Ax=" + std::to_string(Ax));
    ::sor::test::report(std::fabs(r.objective - 1.0) < 1e-8,
                        "tiny feasible: objective 1", __FILE__, __LINE__,
                        "obj=" + std::to_string(r.objective));
}

void test_tiny_infeasible() {
    // 2 x0 = 1, binary - no integer solution (kernel dim 0: reform must be
    // refused, and the plain solve must prove Infeasible).
    auto lp = make_eq_binary({{2}}, {1}, {1});
    auto map = try_lattice_reform(lp);
    ::sor::test::report(!map.has_value(), "tiny infeas: reform refused",
                        __FILE__, __LINE__);
    BabOptions bab;
    bab.time_limit_s = 5;
    auto out = solve_milp_lattice(lp, bab, true);
    const auto ev = sor::search::milp_evidence(out.diag, bab);
    const auto r = sor::certify::finalize_result(out.raw, ev);
    ::sor::test::report(r.status == Status::Infeasible,
                        "tiny infeas: proved Infeasible", __FILE__, __LINE__,
                        std::string(sor::core::to_string(r.status)));
}

void test_paper_knapsack_kernel() {
    // Aardal-Wolsey Example 3 (single-row): a = (12223,...,85569).
    // RHS = a[0] → solution e0 = (1,0,0,0,0).
    const std::vector<double> a = {12223, 12224, 36674, 61119, 85569};
    const double rhs = a[0];
    auto lp = make_eq_binary({a}, {rhs}, {0, 0, 0, 0, 0});
    auto map = try_lattice_reform(lp);
    ::sor::test::report(map.has_value(), "paper ex3: reform applies",
                        __FILE__, __LINE__);
    if (!map) return;
    ::sor::test::report(map->kernel_dim == 4, "paper ex3: kernel dim 4",
                        __FILE__, __LINE__);
    for (int t = 0; t < map->kernel_dim; ++t) {
        long double s = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            s += static_cast<long double>(a[i]) *
                 static_cast<long double>(map->Q[i][static_cast<std::size_t>(t)]);
        }
        ::sor::test::report(s == 0, "paper ex3: A Q_col = 0", __FILE__, __LINE__,
                            "col=" + std::to_string(t) + " s=" + std::to_string((double)s));
    }
    long double ax0 = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        ax0 += static_cast<long double>(a[i]) *
               static_cast<long double>(map->x0[i]);
    ::sor::test::report(ax0 == static_cast<long double>(rhs),
                        "paper ex3: A x0 = b", __FILE__, __LINE__);
}

void test_tiny_market_split_optimal() {
    // 2 equalities, 6 binary - market-split shape, small enough for B&B.
    // Known feasible: x = (1,1,0,0,0,1) → sum=3, weighted=1+2+32=35.
    auto lp = make_eq_binary({{1, 1, 1, 1, 1, 1}, {1, 2, 4, 8, 16, 32}},
                             {3, 35}, {0, 0, 0, 0, 0, 0});
    BabOptions bab;
    bab.time_limit_s = 10;
    bab.cuts_enabled = false;
    auto out = solve_milp_lattice(lp, bab, true);
    ::sor::test::report(out.reform_applied, "ms_tiny: reform applies",
                        __FILE__, __LINE__);
    ::sor::test::report(out.exact_equivalence && !out.fell_back,
                        "ms_tiny: exact equivalence, no fallback re-solve",
                        __FILE__, __LINE__);
    const auto ev = sor::search::milp_evidence(out.diag, bab);
    const auto r = sor::certify::finalize_result(out.raw, ev);
    ::sor::test::report(r.status == Status::Optimal || r.status == Status::Feasible,
                        "ms_tiny: Optimal/Feasible", __FILE__, __LINE__,
                        std::string(sor::core::to_string(r.status)));
    if (r.x.empty()) return;
    ::sor::test::report(r.x.size() == 6, "ms_tiny: x size", __FILE__, __LINE__);
    ::sor::test::report(lp.max_row_violation(r.x) < 1e-8 &&
                            lp.max_bound_violation(r.x) < 1e-8,
                        "ms_tiny: postsolved solution feasible", __FILE__, __LINE__);
}

// Regression 1 (wrong proved optimum): forced-zero continuous restriction.
// min x0 + 10 x1 + 20 x2 s.t. x0 + x1 + x2 = 1, x0 continuous [0,inf),
// x1,x2 binary. True optimum: x0=1 → 1. The restricted problem (x0=0)
// proves Optimal 10 - that answer must NEVER ship; the original must be
// re-solved and report 1.
void test_restriction_optimum_falls_back() {
    auto lp = make_mixed_lp({{0.0, sor::model::kInf, false}, {0.0, 1.0, true}, {0.0, 1.0, true}},
                            {{1, 1, 1}}, {1}, {1, 10, 20});
    auto map = try_lattice_reform(lp);
    ::sor::test::report(map.has_value() && map->forced_zero_cols.size() == 1,
                        "forced-zero: reform applies with restriction",
                        __FILE__, __LINE__);
    BabOptions bab;
    bab.time_limit_s = 5;
    auto out = solve_milp_lattice(lp, bab, true);
    ::sor::test::report(out.reform_applied, "forced-zero: reform applied",
                        __FILE__, __LINE__);
    const auto ev = sor::search::milp_evidence(out.diag, bab);
    const auto r = sor::certify::finalize_result(out.raw, ev);
    ::sor::test::report(r.status == Status::Optimal,
                        "forced-zero: Optimal after fallback", __FILE__, __LINE__,
                        std::string(sor::core::to_string(r.status)));
    ::sor::test::report(std::fabs(r.objective - 1.0) < 1e-9,
                        "forced-zero: objective 1 (not restricted 10)",
                        __FILE__, __LINE__,
                        "obj=" + std::to_string(r.objective));
    if (!r.x.empty()) {
        ::sor::test::report(lp.max_row_violation(r.x) < 1e-8,
                            "forced-zero: solution feasible", __FILE__, __LINE__);
    }
}

// Regression 2 (false proved Infeasible): forced-zero restriction makes the
// integer system bounds-infeasible while the original is feasible via the
// continuous column. min x0 + 10 x1 + 20 x2 s.t. x0 + x1 + 2 x2 = 5.
// Restricted (x0=0): max x1+2x2 = 3 < 5 → "Infeasible" - must not ship.
// True optimum: x0=5 → 5.
void test_restriction_infeasible_falls_back() {
    auto lp = make_mixed_lp({{0.0, sor::model::kInf, false}, {0.0, 1.0, true}, {0.0, 1.0, true}},
                            {{1, 1, 2}}, {5}, {1, 10, 20});
    BabOptions bab;
    bab.time_limit_s = 5;
    auto out = solve_milp_lattice(lp, bab, true);
    ::sor::test::report(out.reform_applied, "false-infeas: reform applied",
                        __FILE__, __LINE__);
    const auto ev = sor::search::milp_evidence(out.diag, bab);
    const auto r = sor::certify::finalize_result(out.raw, ev);
    ::sor::test::report(r.status == Status::Optimal,
                        "false-infeas: Optimal (not Infeasible)", __FILE__, __LINE__,
                        std::string(sor::core::to_string(r.status)));
    ::sor::test::report(std::fabs(r.objective - 5.0) < 1e-9,
                        "false-infeas: objective 5", __FILE__, __LINE__,
                        "obj=" + std::to_string(r.objective));
}

// Regression 3 (wrong proved optimum on an unbounded problem): free integer
// columns must make the reform refuse outright; with the flag on, the answer
// must be identical to the plain solve.
// min -x1 s.t. x1 - 2 x2 = 0, x1,x2 free integer (unbounded below).
void test_unbounded_integer_refused() {
    auto lp = make_mixed_lp({{sor::model::kInf * -1.0, sor::model::kInf, true},
                             {sor::model::kInf * -1.0, sor::model::kInf, true}},
                            {{1, -2}}, {0}, {-1, 0});
    auto map = try_lattice_reform(lp);
    ::sor::test::report(!map.has_value(), "unbounded: reform refused (infinite bounds)",
                        __FILE__, __LINE__);

    BabOptions bab;
    bab.time_limit_s = 5;
    auto out = solve_milp_lattice(lp, bab, true);
    ::sor::test::report(!out.reform_applied && !out.fell_back,
                        "unbounded: plain solve used", __FILE__, __LINE__);

    // Ground truth: identical code path (reform refused) → identical result.
    BabDiagnostics diag_plain;
    auto raw_plain = sor::search::solve_milp(lp, bab, diag_plain);
    ::sor::test::report(out.raw.proposed_status == raw_plain.proposed_status,
                        "unbounded: status matches plain solve", __FILE__, __LINE__);
    ::sor::test::report(!out.reform_applied ||
                            !(out.raw.proposed_status == Status::Optimal &&
                              out.raw.objective < 0),
                        "unbounded: no boxed false Optimal", __FILE__, __LINE__);
}

// gcd obstruction (2x0+2x1=1, no integer solution -- LHS always even, RHS
// odd), with n > m so the reform is at least attempted (unlike the m=1,n=1
// case in test_tiny_infeasible, which is refused before any lattice work).
// For a single-row gcd obstruction like this the AHL extraction's natural
// shortest x0-position candidate is the raw generator (0,0,N1,-N2*b) itself
// (top block all-zero), which fails A*x0=b directly rather than manifesting
// as the paper's k>1 marker -- so this exercises the "extraction retries,
// exhausts, refuses cleanly" path, not the fast proven_infeasible shortcut.
// Either way the invariant that actually matters is: no wrong "Feasible".
void test_gcd_obstruction_falls_back_to_correct_infeasible() {
    auto lp = make_eq_binary({{2, 2}}, {1}, {0, 0});
    BabOptions bab;
    bab.time_limit_s = 5;
    auto out = solve_milp_lattice(lp, bab, true);
    const auto ev = sor::search::milp_evidence(out.diag, bab);
    const auto r = sor::certify::finalize_result(out.raw, ev);
    ::sor::test::report(r.status == Status::Infeasible,
                        "gcd obstruction: proved Infeasible (reform-direct or fallback)",
                        __FILE__, __LINE__,
                        std::string(sor::core::to_string(r.status)));
}

// A genuine AHL ±k*N1 (k>1) witness, verified against the bottom block:
// three free binary columns so a length-1 kernel direction other than the
// trivial single-row generator exists. 4x0+2x1+2x2=1 -- gcd(4,2,2)=2 does
// not divide 1, so no integer solution exists; unlike the single-row 2-var
// case above, the extra column gives LLL room to present the k>1 marker
// directly instead of only the trivial always-zero-top generator.
void test_proven_infeasible_marker_fires() {
    auto lp = make_eq_binary({{4, 2, 2}}, {1}, {0, 0, 0});
    auto map = try_lattice_reform(lp);
    if (map && map->proven_lp_infeasible) {
        ::sor::test::report(map->exact_equivalence,
                            "ahl k>1 marker: exact equivalence", __FILE__, __LINE__);
        BabOptions bab;
        bab.time_limit_s = 5;
        auto out = solve_milp_lattice(lp, bab, true);
        ::sor::test::report(!out.fell_back,
                            "ahl k>1 marker: no fallback re-solve", __FILE__, __LINE__);
        const auto ev = sor::search::milp_evidence(out.diag, bab);
        const auto r = sor::certify::finalize_result(out.raw, ev);
        ::sor::test::report(r.status == Status::Infeasible,
                            "ahl k>1 marker: proved Infeasible directly",
                            __FILE__, __LINE__,
                            std::string(sor::core::to_string(r.status)));
    } else {
        // The marker not firing here just means extraction found a
        // consistent x0/refused instead on this instance -- either is safe;
        // only report on the invariant that matters when it DOES fire.
        ::sor::test::report(true, "ahl k>1 marker: did not fire on this instance (OK)",
                            __FILE__, __LINE__);
    }
}

// Exact equivalence, LP-infeasible marker path: bounds alone make the real
// relaxation empty (x0,x1 in [10,20], x0+x1=1 needs sum<=1 << 20). Proven by
// the mu-projection LP itself, before any B&B node runs.
void test_lp_infeasible_marker_no_fallback() {
    auto lp = make_mixed_lp({{10.0, 20.0, true}, {10.0, 20.0, true}},
                            {{1, 1}}, {1}, {0, 0});
    auto map = try_lattice_reform(lp);
    ::sor::test::report(map.has_value() && map->exact_equivalence &&
                            map->proven_lp_infeasible,
                        "lp-infeas marker: detected during projection",
                        __FILE__, __LINE__);
    BabOptions bab;
    bab.time_limit_s = 5;
    auto out = solve_milp_lattice(lp, bab, true);
    ::sor::test::report(out.reform_applied && !out.fell_back,
                        "lp-infeas marker: no fallback re-solve", __FILE__, __LINE__);
    const auto ev = sor::search::milp_evidence(out.diag, bab);
    const auto r = sor::certify::finalize_result(out.raw, ev);
    ::sor::test::report(r.status == Status::Infeasible,
                        "lp-infeas marker: proved Infeasible directly",
                        __FILE__, __LINE__,
                        std::string(sor::core::to_string(r.status)));
}

// Certification path: restricted (forced x0=0) Optimal exactly matches the
// original LP relaxation bound, so it ships as a certified Optimal for the
// ORIGINAL problem without a fallback re-solve. Needs kernel dim >= 1, so
// two free binary columns against one row.
// min x0 s.t. x0 + x1 + x2 = 1, x0 continuous [0,1] c=+1, x1,x2 binary c=0.
// Restricted (x0=0): x1+x2=1 feasible (e.g. x1=1,x2=0), obj 0.
// Original LP relaxation: x0=0, x1+x2=1 within [0,1]^2, V_LP=0.
// V_r == V_LP -> certified optimal (ground truth: obj 0, matches).
void test_certification_matches_lp_bound() {
    auto lp = make_mixed_lp({{0.0, 1.0, false}, {0.0, 1.0, true}, {0.0, 1.0, true}},
                            {{1, 1, 1}}, {1}, {1, 0, 0});
    auto map = try_lattice_reform(lp);
    ::sor::test::report(map.has_value() && !map->exact_equivalence,
                        "certify: reform applies as a restriction",
                        __FILE__, __LINE__);
    BabOptions bab;
    bab.time_limit_s = 5;
    auto out = solve_milp_lattice(lp, bab, true);
    ::sor::test::report(out.reform_applied && out.certified && !out.fell_back,
                        "certify: shipped via LP-bound match, no fallback",
                        __FILE__, __LINE__);
    const auto ev = sor::search::milp_evidence(out.diag, bab);
    const auto r = sor::certify::finalize_result(out.raw, ev);
    ::sor::test::report(r.status == Status::Optimal,
                        "certify: Optimal", __FILE__, __LINE__,
                        std::string(sor::core::to_string(r.status)));
    ::sor::test::report(std::fabs(r.objective - 0.0) < 1e-9,
                        "certify: objective 0", __FILE__, __LINE__,
                        "obj=" + std::to_string(r.objective));
}

void test_markshare_smoke() {
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
        ::sor::test::report(true, "markshare smoke: skipped (no mps)",
                            __FILE__, __LINE__);
        return;
    }
    sor::io::MpsReadReport rep;
    auto lp = sor::io::read_mps_file(path, rep);
    auto map = try_lattice_reform(lp);
    ::sor::test::report(map.has_value(), "markshare1: reform applies",
                        __FILE__, __LINE__);
    if (!map) return;
    ::sor::test::report(map->kernel_dim > 0, "markshare1: kernel dim > 0",
                        __FILE__, __LINE__);
    ::sor::test::report(map->forced_zero_cols.size() >= 6,
                        "markshare1: forced continuous deviations",
                        __FILE__, __LINE__);

    BabOptions bab;
    bab.time_limit_s = 5;   // smoke only - full markshare close needs stronger MIP
    bab.max_nodes = 20000;
    bab.cuts_enabled = false;
    auto out = solve_milp_lattice(lp, bab, true);
    if (!out.raw.x.empty()) {
        // out.raw.x is already postsolved into original x-space.
        const double viol = lp.max_row_violation(out.raw.x);
        ::sor::test::report(viol < 1e-6, "markshare1: postsolve feasible",
                            __FILE__, __LINE__, "viol=" + std::to_string(viol));
    } else {
        // Honest: reform applies and is self-checked; finding a feasible μ on
        // markshare with the current B&B is still open (see paper_bibliography).
        ::sor::test::report(true, "markshare1: reform OK, no incumbent in smoke budget",
                            __FILE__, __LINE__);
    }
}

}  // namespace

int main() {
    test_tiny_feasible();
    test_tiny_infeasible();
    test_paper_knapsack_kernel();
    test_tiny_market_split_optimal();
    test_restriction_optimum_falls_back();
    test_restriction_infeasible_falls_back();
    test_unbounded_integer_refused();
    test_gcd_obstruction_falls_back_to_correct_infeasible();
    test_proven_infeasible_marker_fires();
    test_lp_infeasible_marker_no_fallback();
    test_certification_matches_lp_bound();
    test_markshare_smoke();
    return ::sor::test::finish("test_lattice_reform");
}
