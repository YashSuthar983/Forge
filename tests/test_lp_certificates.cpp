#include "forge/certify/finalize.hpp"
#include "forge/io/mps.hpp"
#include "forge/engines/dual_simplex.hpp"
#include "forge/engines/farkas.hpp"
#include "test_helpers.hpp"
#include "forge/model/exact.hpp"
#include <random>
#include <cmath>

#include <sstream>
#include <limits>

using namespace forge;

namespace {

model::LpProblem read(const char* text) {
    std::istringstream input(text);
    io::MpsReadReport report;
    return io::read_mps(input, report);
}

}  // namespace

// certify::safe_lagrangian_lower_bound: weak duality from ANY multipliers.
void test_safe_lagrangian_bound() {
    // min x + y  s.t.  x + y >= 2,  x in [0, inf), y in [0, 3].
    const auto lp = read(R"(NAME SAFE
ROWS
 N OBJ
 G R1
COLUMNS
    X OBJ 1 R1 1
    Y OBJ 1 R1 1
RHS
    RHS R1 2
BOUNDS
 UP BND Y 3
ENDATA
)");
    // Exact dual y = 1: d = 0 everywhere, bound = 2.
    auto b = certify::safe_lagrangian_lower_bound(lp, {1.0}, lp.col_lo, lp.col_hi);
    CHECK(b.finite);
    CHECK_NEAR(b.value, 2.0, 1e-9);
    CHECK(b.value <= 2.0);
    // Round-off dual y = 1 + 1e-14 leaves d_x = -1e-14 on [0, inf): the
    // multiplier correction must restore a finite bound (not wave it through).
    b = certify::safe_lagrangian_lower_bound(lp, {1.0 + 1e-14}, lp.col_lo, lp.col_hi);
    CHECK(b.finite);
    CHECK(b.value <= 2.0);
    CHECK(b.value >= 2.0 - 1e-6);
    // Any y is valid: y = 0.5 gives L = 1 + 0.5*0 ... = 1 (weak duality).
    b = certify::safe_lagrangian_lower_bound(lp, {0.5}, lp.col_lo, lp.col_hi);
    CHECK(b.finite);
    CHECK(b.value <= 1.0 + 1e-12);
}

void test_safe_bound_refuses_unbounded_direction() {
    // min -x  s.t.  y - x >= -1 (never caps x),  x, y in [0, inf).
    // LP is unbounded below; y = 1e-8 leaves d_x = -1 + 1e-8 < 0 on x with
    // no finite upper bound anywhere: the bound must be -inf.
    const auto lp = read(R"(NAME UNB
ROWS
 N OBJ
 G R1
COLUMNS
    X OBJ -1 R1 -1
    Y OBJ 0 R1 1
RHS
    RHS R1 -1
ENDATA
)");
    const auto b = certify::safe_lagrangian_lower_bound(lp, {1e-8}, lp.col_lo, lp.col_hi);
    CHECK(!b.finite);
}

void test_safe_bound_snaps_straddling_enclosure_on_infinite_side() {
    // Large |A'y| with near-cancellation against c leaves d ~ 0 but a wide
    // absolute sum: the interval path forms [-e, e] with e <= dual_feas_tol
    // on a free-upper column. That must snap to a finite bound (C9), not
    // fall through to -inf because the enclosure straddles 0.
    const auto lp = read(R"(NAME STRADDLE
ROWS
 N OBJ
 E R1
COLUMNS
    X OBJ 1000000 R1 1000000
RHS
    RHS R1 1
BOUNDS
 LO BND X 0
ENDATA
)");
    // Exact dual y = 1: d = 0, bound = 1.
    auto b = certify::safe_lagrangian_lower_bound(lp, {1.0}, lp.col_lo, lp.col_hi, 1e-7);
    CHECK(b.finite);
    CHECK_NEAR(b.value, 1.0, 1e-6);
    // Round-off dual: d_x ~ -1e-8 on [0, inf), within dual_feas_tol.
    b = certify::safe_lagrangian_lower_bound(lp, {1.0 + 1e-14}, lp.col_lo, lp.col_hi, 1e-7);
    CHECK(b.finite);
    CHECK(b.value <= 1.0 + 1e-6);
}

void test_small_cost_cannot_erase_unbounded_direction() {
    // A reduced cost outside dual_feas_tol on an infinite bound is a real
    // unbounded direction: snapping it to publish a zero gap would be unsound.
    // (Residuals inside dual_feas_tol are snapped by design — C9 — so the
    // cost here is larger than the dual feasibility tolerance.)
    const auto source = read(R"(NAME TINY
ROWS
 N OBJ
COLUMNS
 X OBJ -0.000001
ENDATA
)");
    for (bool maximize : {false, true}) {
        auto lp = source;
        lp.maximize = maximize;
        lp.c[0] = maximize ? 1e-6 : -1e-6;
        lp.obj_offset = 5.0;
        core::RawResult fake;
        fake.x = {0.0};
        fake.objective = fake.dual_bound = 5.0;
        fake.proposed_status = core::Status::Optimal;
        fake.proposed_level = core::ProofLevel::ProvedOptimalFP;
        auto checked = certify::check_lp_point(lp, fake, 1e-7, 1e-7, 1e-9, true);
        // Primal-feasible at x=0, but the structural reduced cost exceeds
        // dual_feas_tol on an infinite side, so the Lagrangian is -inf.
        CHECK(checked.max_primal_violation <= 1e-7);
        CHECK(!std::isfinite(checked.gap_rel));
        CHECK(certify::finalize_result(fake, checked).status != core::Status::Optimal);

        engines::SimplexOptions opts;
        opts.presolve = false;
        opts.ruiz_iterations = 0;
        opts.method = engines::SimplexMethod::Primal;
        opts.time_limit_s = 1.0;
        opts.max_iterations = 32;
        for (bool dual : {false, true}) {
            engines::SimplexDiagnostics diag;
            auto raw = dual ? engines::solve_dual_simplex(lp, opts, diag)
                            : engines::solve_simplex(lp, opts, diag);
            CHECK(!diag.dual_bound_finite);
            checked = certify::check_lp_point(lp, raw, 1e-7, 1e-7, 1e-9, true);
            CHECK(certify::finalize_result(raw, checked).status != core::Status::Optimal);
        }
    }
}

void test_checked_repair_preserves_sense_and_offset() {
    auto lp = read(R"(NAME REPAIR
ROWS
 N OBJ
 G R1
COLUMNS
 X OBJ 1 R1 1
RHS
 RHS R1 2
ENDATA
)");
    for (bool maximize : {false, true}) {
        lp.maximize = maximize;
        lp.c[0] = maximize ? -1.0 : 1.0;
        lp.obj_offset = 5.0;
        core::RawResult raw;
        raw.x = {2.0};
        raw.y = {maximize ? -(1.0 + 1e-14) : 1.0 + 1e-14};
        raw.objective = maximize ? 3.0 : 7.0;
        raw.dual_bound = raw.objective;
        raw.proposed_status = core::Status::Optimal;
        raw.proposed_level = core::ProofLevel::ProvedOptimalFP;
        const auto checked = certify::check_lp_point(lp, raw, 1e-7, 1e-7, 1e-9, true);
        CHECK(checked.checker_passed);
        CHECK(checked.gap_rel < 1e-9);
        CHECK(certify::finalize_result(raw, checked).status == core::Status::Optimal);
        engines::SimplexOptions opts;
        opts.presolve = false;
        opts.ruiz_iterations = 0;
        opts.method = engines::SimplexMethod::Primal;
        for (bool dual : {false, true}) {
            engines::SimplexDiagnostics diag;
            auto solved = dual ? engines::solve_dual_simplex(lp, opts, diag)
                               : engines::solve_simplex(lp, opts, diag);
            CHECK(diag.dual_bound_finite);
            CHECK_NEAR(solved.dual_bound, raw.objective, 1e-9);
            CHECK(maximize ? solved.dual_bound >= raw.objective
                           : solved.dual_bound <= raw.objective);
        }
    }
    core::RawResult nonfinite;
    nonfinite.x = {std::numeric_limits<double>::quiet_NaN()};
    CHECK(!certify::check_lp_point(lp, nonfinite, 1e-7, 1e-7, 1e-9, false).checker_passed);
    nonfinite.x = {forge::model::kInf};
    CHECK(!certify::check_lp_point(lp, nonfinite, 1e-7, 1e-7, 1e-9, false).checker_passed);
}

// Compare the common-denominator implementation with independent, per-term
// rational arithmetic. Coprime denominators force the LCM path; finite boxes
// make every arbitrary multiplier a valid lower bound, for either sense.
void test_exact_dual_rational_reference() {
    auto lp = read(R"(NAME FRACTIONS
ROWS
 N OBJ
 E R1
 E R2
 E R3
COLUMNS
 X OBJ 0.125 R1 10000000000
 X R2 1 R3 -10000000000
 Y OBJ -0.75 R1 0.0625
 Y R2 -2.75 R3 0.2
RHS
 RHS R1 3 R2 -7
 RHS R3 0.25
BOUNDS
 LO BND X -2
 UP BND X 5
 LO BND Y -3
 UP BND Y 7
ENDATA
)");
    std::mt19937 rng(26202);
    const int denominators[] = {3, 7, 11};
    using model::Rational;
    for (bool maximize : {false, true}) {
        lp.maximize = maximize;
        lp.obj_offset = 9.25;
        const int sense = maximize ? -1 : 1;
        for (int trial = 0; trial < 40; ++trial) {
            std::vector<Rational> y;
            std::vector<std::string> witness;
            for (int den : denominators) {
                y.push_back(Rational(static_cast<int>(rng() % 21) - 10) / den);
                witness.push_back(y.back().str());
            }
            Rational reference = Rational(sense) * Rational(lp.obj_offset);
            std::vector<Rational> reduced;
            for (double cost : lp.c) reduced.push_back(Rational(sense) * Rational(cost));
            const auto& rp = lp.A.pattern.row_ptr();
            const auto& ci = lp.A.pattern.col_idx();
            for (std::size_t i = 0; i < y.size(); ++i) {
                reference += y[i] * Rational(y[i] >= 0 ? lp.row_lo[i] : lp.row_hi[i]);
                for (auto k = rp[i]; k < rp[i+1]; ++k)
                    reduced[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])] -=
                        y[i] * Rational(lp.A.vals[static_cast<std::size_t>(k)]);
            }
            for (std::size_t j = 0; j < reduced.size(); ++j)
                reference += reduced[j] * Rational(reduced[j] >= 0 ? lp.col_lo[j] : lp.col_hi[j]);
            const auto bound = certify::exact_dual_lower_bound(lp, witness);
            CHECK(bound.finite);
            CHECK(bound.value == model::rounded_down(reference));
            CHECK(Rational(bound.value) <= reference);
        }
    }
    for (const auto& bad : {"", "--1", "1/0", "1/-3", "1/2/3", "nan", "/2"}) {
        CHECK(!certify::exact_dual_lower_bound(lp, {bad, "0", "0"}).finite);
        CHECK(certify::exact_dual_support_failure(lp, {bad, "0", "0"}).variable == -1);
    }
}

// Free basic columns with non-dyadic duals: floating duals leave round-off
// reduced costs on columns with no finite bound, so the bound for them is
// -inf; repair_dual_certificate must zero those exactly and publish a witness
// whose exact Lagrangian is finite and matches the optimum.
void test_targeted_dual_certificate() {
    std::mt19937 rng(91);
    for (int trial = 0; trial < 20; ++trial) {
        model::LpProblem lp;
        constexpr int n = 12;
        std::vector<core::Index> rows, cols;
        std::vector<double> vals;
        std::vector<model::Rational> y(n);
        for (int i = 0; i < n; ++i) y[static_cast<std::size_t>(i)] = model::Rational(static_cast<int>(rng() % 21) - 10, 3 + 2 * static_cast<int>(rng() % 5));
        std::vector<model::Rational> c(n, 0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const double a = i == j ? 7.0 + j : static_cast<int>(rng() % 9) - 4;
                if (a == 0) continue;
                rows.push_back(i); cols.push_back(j); vals.push_back(a);
                c[static_cast<std::size_t>(j)] += model::Rational(a) * y[static_cast<std::size_t>(i)];
            }
        lp.A = sparse::from_triplets(n, n, rows, cols, vals);
        // c is the rounded A'y: the exact duals of this basis are near y but
        // not dyadic, and every column is free.
        for (const auto& v : c) lp.c.push_back(v.convert_to<double>());
        lp.col_lo.assign(n, -model::kInf); lp.col_hi.assign(n, model::kInf);
        lp.row_lo.assign(n, 1.0); lp.row_hi.assign(n, 1.0);
        core::RawResult raw;
        for (int j = 0; j < n; ++j) raw.certificate_basis.push_back(j);
        // The basis point: A x = 1 by dense elimination with partial pivoting.
        {
            std::vector<double> dense(n * n, 0.0), rhs(n, 1.0);
            for (std::size_t k = 0; k < rows.size(); ++k)
                dense[static_cast<std::size_t>(rows[k]) * n + static_cast<std::size_t>(cols[k])] = vals[k];
            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[static_cast<std::size_t>(i)] = i;
            for (int col = 0; col < n; ++col) {
                int piv = col;
                for (int r = col + 1; r < n; ++r)
                    if (std::fabs(dense[static_cast<std::size_t>(r * n + col)]) > std::fabs(dense[static_cast<std::size_t>(piv * n + col)])) piv = r;
                for (int t = 0; t < n; ++t) std::swap(dense[static_cast<std::size_t>(col * n + t)], dense[static_cast<std::size_t>(piv * n + t)]);
                std::swap(rhs[static_cast<std::size_t>(col)], rhs[static_cast<std::size_t>(piv)]);
                for (int r = col + 1; r < n; ++r) {
                    const double f = dense[static_cast<std::size_t>(r * n + col)] / dense[static_cast<std::size_t>(col * n + col)];
                    for (int t = col; t < n; ++t) dense[static_cast<std::size_t>(r * n + t)] -= f * dense[static_cast<std::size_t>(col * n + t)];
                    rhs[static_cast<std::size_t>(r)] -= f * rhs[static_cast<std::size_t>(col)];
                }
            }
            raw.x.assign(n, 0.0);
            for (int r = n - 1; r >= 0; --r) {
                double v = rhs[static_cast<std::size_t>(r)];
                for (int t = r + 1; t < n; ++t) v -= dense[static_cast<std::size_t>(r * n + t)] * raw.x[static_cast<std::size_t>(t)];
                raw.x[static_cast<std::size_t>(r)] = v / dense[static_cast<std::size_t>(r * n + r)];
            }
        }
        CHECK(certify::repair_dual_certificate(lp, raw));
        CHECK(raw.exact_dual.size() == static_cast<std::size_t>(n));
        const auto bound = certify::exact_dual_lower_bound(lp, raw.exact_dual);
        CHECK(bound.finite);
        // The optimum is c'x with A x = 1, i.e. sum of the exact duals.
        const auto exact = certify::exact_dual_lower_bound(lp, raw.exact_dual);
        model::Rational sum = 0;
        for (const auto& t : raw.exact_dual) sum += model::Rational(t);
        CHECK(std::fabs(exact.value - sum.convert_to<double>()) <= 1e-9 * (1 + std::fabs(exact.value)));
    }
}

void test_sparse_integer_basis_certificate() {
    std::mt19937 rng(26207);
    for (int trial = 0; trial < 20; ++trial) {
        model::LpProblem lp;
        constexpr int n = 9;
        std::vector<core::Index> rows, cols;
        std::vector<double> vals, expected;
        lp.c.assign(n, 0);
        lp.col_lo.assign(n, -1); lp.col_hi.assign(n, 1);
        lp.row_lo.assign(n, 0); lp.row_hi.assign(n, 0);
        for (int i = 0; i < n; ++i)
            expected.push_back((static_cast<int>(rng() % 17) - 8) / 16.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const double a = i == j ? 100.0 : static_cast<int>(rng() % 7) - 3;
                if (a == 0) continue;
                rows.push_back(i); cols.push_back(j); vals.push_back(a);
                lp.c[static_cast<std::size_t>(j)] += a * expected[static_cast<std::size_t>(i)];
            }
        lp.A = sparse::from_triplets(n, n, rows, cols, vals);
        core::RawResult raw;
        for (int j = 0; j < n; ++j) raw.certificate_basis.push_back(j);
        CHECK(certify::repair_basis_certificate(lp, raw));
        CHECK(raw.exact_dual.size() == expected.size());
        for (std::size_t i = 0; i < expected.size() && i < raw.exact_dual.size(); ++i)
            CHECK(model::Rational(raw.exact_dual[i]) == model::Rational(expected[i]));
        CHECK(certify::exact_dual_lower_bound(lp, raw.exact_dual).finite);
        CHECK(!certify::repair_basis_certificate(lp, raw, {.max_operations = 0}));
        CHECK(raw.exact_dual.empty());
        CHECK(!certify::repair_basis_certificate(lp, raw, {.time_limit_s = std::numeric_limits<double>::min()}));
    }
    const auto third = read(R"(NAME THIRD
ROWS
 N OBJ
 E R1
COLUMNS
 X OBJ 1 R1 3
RHS
 RHS R1 1
ENDATA
)");
    core::RawResult raw;
    raw.certificate_basis = {0};
    CHECK(certify::repair_basis_certificate(third, raw));
    CHECK(raw.exact_dual.size() == 1);
    if (!raw.exact_dual.empty()) CHECK(model::Rational(raw.exact_dual[0]) == model::Rational(1) / 3);
}

void test_exact_terminal_rays() {
    model::LpProblem p;
    p.A = sparse::from_triplets(1,2,{0,0},{0,1},{3,-1});
    p.c = {0,-1}; p.col_lo = {0,0}; p.col_hi = {model::kInf,model::kInf};
    p.row_lo = p.row_hi = {0};
    // A rounded 1/3 leaves a nonzero recession drift on an equality.
    CHECK(!certify::check_primal_ray(p,{1.0/3,1},1e-7).certified);
    CHECK(certify::check_exact_primal_ray(p,{"1/3","1"},1e-7).certified);
    CHECK(!certify::check_exact_primal_ray(p,{"1/3","-1"},1e-7).certified);
    core::RawResult raw;
    raw.certificate_basis = {0};
    CHECK(certify::repair_basis_primal_ray(p,raw,1,1,{}));
    CHECK(certify::check_exact_primal_ray(p,raw.primal_ray.exact_direction,1e-7).certified);
    // A caller asking for a tighter terminal gate must get a witness even
    // when its slope is smaller than the default feasibility tolerance.
    p.c[1] = -1e-8;
    raw.primal_ray = {};
    CHECK(!certify::repair_basis_primal_ray(p, raw, 1, 1, {}));
    CHECK(certify::repair_basis_primal_ray(p, raw, 1, 1, {.ray_tolerance = 1e-10}));
    CHECK(certify::check_exact_primal_ray(p, raw.primal_ray.exact_direction, 1e-10).certified);
    for (int invalid_kind = 0; invalid_kind < 3; ++invalid_kind) {
        certify::ExactCertificatePolicy policy;
        if (invalid_kind == 0) policy.time_limit_s = core::kNaN;
        if (invalid_kind == 1) policy.max_bits = 0;
        if (invalid_kind == 2) policy.ray_tolerance = 0;
        bool rejected = false;
        try { certify::repair_basis_primal_ray(p, raw, 1, 1, policy); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
    }
    // A non-dyadic Farkas multiplier must cancel the free column exactly.
    p.A = sparse::from_triplets(2,1,{0,1},{0,0},{3,1});
    p.c = {0}; p.col_lo = {-model::kInf}; p.col_hi = {model::kInf};
    p.row_lo = {3,-model::kInf}; p.row_hi = {model::kInf,0};
    CHECK(certify::check_exact_dual_farkas_ray(p,{"-1/3","1"},1e-7).certified);
    CHECK(!certify::check_exact_dual_farkas_ray(p,{"-1/3","0"},1e-7).certified);
    raw = {}; raw.certificate_basis = {0,2};
    CHECK(certify::repair_basis_farkas_certificate(p,raw,1,-1,{}));
    CHECK(certify::check_exact_dual_farkas_ray(p,raw.dual_farkas_ray.exact_multipliers,1e-7).certified);
    p.row_lo[0] = 3e-8;
    raw.dual_farkas_ray = {}; raw.ray.clear();
    CHECK(!certify::repair_basis_farkas_certificate(p, raw, 1, -1, {}));
    CHECK(certify::repair_basis_farkas_certificate(p, raw, 1, -1, {.ray_tolerance = 1e-10}));
    CHECK(certify::check_exact_dual_farkas_ray(p,
        raw.dual_farkas_ray.exact_multipliers, 1e-10).certified);
}

void test_presolved_terminal_witnesses() {
    model::LpProblem unbounded;
    unbounded.A = sparse::from_triplets(2, 3, {0,0,1,1}, {0,1,1,2}, {3,-1,1,-1});
    unbounded.c = {0,-1,0};
    unbounded.col_lo = {0,0,0};
    unbounded.col_hi.assign(3, model::kInf);
    unbounded.row_lo = {0,-model::kInf}; unbounded.row_hi = {0,0};
    model::LpProblem infeasible;
    infeasible.A = sparse::from_triplets(3, 2, {0,0,1,2}, {0,1,0,1}, {3,-1,3,1});
    infeasible.c = {1,0};
    infeasible.col_lo.assign(2, -model::kInf); infeasible.col_hi.assign(2, model::kInf);
    infeasible.row_lo = {0,3,-model::kInf}; infeasible.row_hi = {0,model::kInf,0};
    for (bool presolve : {false, true}) {
        for (const auto method : {engines::SimplexMethod::Primal, engines::SimplexMethod::Dual}) {
            engines::SimplexOptions opts;
            opts.presolve = presolve; opts.method = method;
            opts.time_limit_s = 2; opts.max_iterations = 64;
            for (const auto* model : {&unbounded, &infeasible}) {
                engines::SimplexDiagnostics diag;
                const auto raw = engines::solve_simplex(*model, opts, diag);
                const auto expected = model == &unbounded ? core::Status::Unbounded : core::Status::Infeasible;
                const auto checked = certify::check_lp_result(*model, raw, engines::simplex_evidence(diag, opts));
                CHECK(certify::finalize_result(raw, checked).status == expected);
                CHECK(diag.iterations <= opts.max_iterations);
                if (expected == core::Status::Unbounded) {
                    CHECK(!raw.primal_ray.exact_direction.empty());
                    CHECK(certify::check_exact_primal_ray(*model, raw.primal_ray.exact_direction, 1e-7).certified);
                } else {
                    CHECK(!raw.dual_farkas_ray.exact_multipliers.empty());
                    CHECK(certify::check_exact_dual_farkas_ray(*model, raw.dual_farkas_ray.exact_multipliers, 1e-7).certified);
                }
            }
        }
    }
}

// Without the exact proof a terminal ray is checked in floating point, and the
// rational repair runs only if that check fails. Both rays here are exact in
// binary64, so each run must finalize with no exact certificate attached;
// with the exact proof requested the exact certificate is still produced.
void test_float_rays_skip_exact_repair_without_exact_proof() {
    model::LpProblem infeasible;   // x + y >= 3 and x + y <= 1
    infeasible.A = sparse::from_triplets(2, 2, {0,0,1,1}, {0,1,0,1}, {1,1,1,1});
    infeasible.c = {1, 1};
    infeasible.col_lo.assign(2, -model::kInf); infeasible.col_hi.assign(2, model::kInf);
    infeasible.row_lo = {3, -model::kInf}; infeasible.row_hi = {model::kInf, 1};
    model::LpProblem unbounded;    // min -x with x - y = 0, x, y >= 0
    unbounded.A = sparse::from_triplets(1, 2, {0,0}, {0,1}, {1,-1});
    unbounded.c = {-1, 0};
    unbounded.col_lo = {0, 0}; unbounded.col_hi.assign(2, model::kInf);
    unbounded.row_lo = unbounded.row_hi = {0};
    for (const bool exact : {false, true}) {
        for (const auto method : {engines::SimplexMethod::Primal, engines::SimplexMethod::Dual}) {
            for (const auto* model : {&infeasible, &unbounded}) {
                engines::SimplexOptions opts;
                opts.presolve = false;
                opts.method = method;
                opts.exact_proof = exact;
                engines::SimplexDiagnostics diag;
                const auto raw = engines::solve_simplex(*model, opts, diag);
                const auto checked = certify::check_lp_result(*model, raw, engines::simplex_evidence(diag, opts));
                const auto expected = model == &unbounded ? core::Status::Unbounded : core::Status::Infeasible;
                CHECK(certify::finalize_result(raw, checked).status == expected);
                const bool has_exact = model == &unbounded
                    ? !raw.primal_ray.exact_direction.empty()
                    : !raw.dual_farkas_ray.exact_multipliers.empty();
                CHECK(has_exact == exact);
            }
        }
    }
}

// The interval fast path of safe_lagrangian_lower_bound must never exceed the
// exactly evaluated Lagrangian of the same multipliers (the exact value comes
// from the d_out route of safe_reduced_cost_tightenings), and must stay within
// rounding of it. Random dense-ish LPs with mixed signs, magnitudes over many
// orders, finite boxes and both row senses; multipliers perturbed so reduced
// costs straddle zero.
void test_interval_lagrangian_matches_exact() {
    std::mt19937 rng(20261006u);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    int compared = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const core::Index m = 2 + static_cast<core::Index>(rng() % 6);
        const core::Index n = 2 + static_cast<core::Index>(rng() % 8);
        model::LpProblem lp;
        lp.maximize = (rng() & 1u) != 0;
        lp.obj_offset = U(rng) * 10.0;
        std::vector<core::Index> rows, cols;
        std::vector<double> vals;
        const double scale = std::pow(10.0, static_cast<double>(rng() % 7) - 3.0);
        for (core::Index i = 0; i < m; ++i)
            for (core::Index j = 0; j < n; ++j)
                if (rng() % 3 != 0) {
                    rows.push_back(i); cols.push_back(j);
                    vals.push_back(U(rng) * scale * std::pow(10.0, static_cast<double>(rng() % 5) - 2.0));
                }
        lp.A = sparse::from_triplets(m, n, rows, cols, vals);
        lp.c.resize(n); lp.col_lo.resize(n); lp.col_hi.resize(n);
        for (core::Index j = 0; j < n; ++j) {
            lp.c[j] = U(rng) * 5.0;
            lp.col_lo[j] = -1.0 - 3.0 * std::fabs(U(rng));
            lp.col_hi[j] = lp.col_lo[j] + 0.5 + 4.0 * std::fabs(U(rng));
        }
        lp.row_lo.resize(m); lp.row_hi.resize(m);
        for (core::Index i = 0; i < m; ++i) {
            lp.row_lo[i] = -2.0 + U(rng);
            lp.row_hi[i] = (rng() % 4 == 0) ? model::kInf : lp.row_lo[i] + 3.0;
        }
        std::vector<double> y(m);
        for (auto& v : y) v = U(rng) * std::pow(10.0, static_cast<double>(rng() % 5) - 2.0);
        const auto fast = certify::safe_lagrangian_lower_bound(lp, y, lp.col_lo, lp.col_hi);
        certify::SafeLpBound exact;
        (void)certify::safe_reduced_cost_tightenings(lp, y, lp.col_lo, lp.col_hi,
                                                     std::numeric_limits<double>::infinity(),
                                                     &exact, nullptr);
        CHECK(fast.finite == exact.finite);
        if (!fast.finite || !exact.finite) continue;
        ++compared;
        CHECK(fast.value <= exact.value);
        CHECK(exact.value - fast.value <= 1e-9 * (1.0 + std::fabs(exact.value)));
    }
    CHECK(compared > 300);
}

// interval_reduced_cost_tightenings may only be weaker than the exactly
// evaluated rule (safe_reduced_cost_tightenings with the same multipliers):
// for every column, its bound must contain the exact one. Integer columns
// with integral boxes; the cutoff a random room above the exact Lagrangian,
// so that both tight and loose fixings occur.
void test_interval_rc_never_tighter_than_exact() {
    std::mt19937 rng(20261008u);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    int fixed = 0, same = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const core::Index m = 2 + static_cast<core::Index>(rng() % 6);
        const core::Index n = 2 + static_cast<core::Index>(rng() % 8);
        model::LpProblem lp;
        lp.maximize = (rng() & 1u) != 0;
        lp.obj_offset = U(rng) * 10.0;
        std::vector<core::Index> rows, cols;
        std::vector<double> vals;
        for (core::Index i = 0; i < m; ++i)
            for (core::Index j = 0; j < n; ++j)
                if (rng() % 3 != 0) {
                    rows.push_back(i); cols.push_back(j);
                    vals.push_back(U(rng) * std::pow(10.0, static_cast<double>(rng() % 5) - 2.0));
                }
        lp.A = sparse::from_triplets(m, n, rows, cols, vals);
        lp.c.resize(n); lp.col_lo.resize(n); lp.col_hi.resize(n);
        lp.is_integer.assign(n, true);
        for (core::Index j = 0; j < n; ++j) {
            lp.c[j] = U(rng) * 5.0;
            lp.col_lo[j] = -static_cast<double>(rng() % 4);
            lp.col_hi[j] = lp.col_lo[j] + 1.0 + static_cast<double>(rng() % 6);
        }
        lp.row_lo.resize(m); lp.row_hi.resize(m);
        for (core::Index i = 0; i < m; ++i) {
            lp.row_lo[i] = -2.0 + U(rng);
            lp.row_hi[i] = (rng() % 4 == 0) ? model::kInf : lp.row_lo[i] + 3.0;
        }
        std::vector<double> y(m);
        for (auto& v : y) v = U(rng) * std::pow(10.0, static_cast<double>(rng() % 5) - 2.0);
        certify::SafeLpBound exact_lb;
        (void)certify::safe_reduced_cost_tightenings(lp, y, lp.col_lo, lp.col_hi,
                                                     std::numeric_limits<double>::infinity(),
                                                     &exact_lb, nullptr);
        if (!exact_lb.finite) continue;
        const double cutoff = exact_lb.value + std::fabs(U(rng)) * 6.0;
        const auto exact = certify::safe_reduced_cost_tightenings(lp, y, lp.col_lo, lp.col_hi,
                                                                  cutoff, nullptr, nullptr);
        const auto fast = certify::interval_reduced_cost_tightenings(
            lp, y, lp.col_lo, lp.col_hi, cutoff, nullptr);
        if (!fast.finite) continue;
        CHECK(fast.bound <= exact_lb.value);
        for (const auto& f : fast.changes) {
            ++fixed;
            bool found = false;
            for (const auto& e : exact) {
                if (e.col != f.col || e.upper != f.upper) continue;
                found = true;
                // Valid: the interval bound keeps everything the exact one keeps.
                CHECK(f.upper ? f.value >= e.value : f.value <= e.value);
                if (f.value == e.value) ++same;
            }
            CHECK(found);   // the exact rule tightens wherever the interval one does
        }
    }
    CHECK(fixed > 100);
    CHECK(same * 10 >= fixed * 9);   // the enclosure is tight: nearly always the same bound
}

// check_dual_farkas_ray certifies by interval arithmetic when that decides
// and exactly otherwise. Either way the verdict must be the exact rule's:
// lower - upper > tol * (1 + max(|lower|, |upper|)) (rounded down against
// rounded up), with lower = sum_j d_j b_j (d = A'y exactly, b the bound d_j's
// sign needs) and upper = sum_i y_i side_i, evaluated here in rationals on
// the multipliers the check returns. The rays are planted around the
// threshold: separation a log-uniform multiple (0.1x .. 10x) of the
// tolerance, coefficients across 1e-4 .. 1e4, so both verdicts and the
// undecided-interval fallback are exercised.
void test_farkas_check_matches_exact_rule() {
    std::mt19937 rng(20261006u + 7u);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    const double tol = 1e-7;
    int certified = 0, refused = 0;
    for (int trial = 0; trial < 800; ++trial) {
        const core::Index m = 1 + static_cast<core::Index>(rng() % 6);
        const core::Index n = 1 + static_cast<core::Index>(rng() % 8);
        std::vector<core::Index> rows, cols;
        std::vector<double> vals;
        for (core::Index i = 0; i < m; ++i)
            for (core::Index j = 0; j < n; ++j)
                if (rng() % 3 != 0) {
                    rows.push_back(i); cols.push_back(j);
                    vals.push_back(U(rng) * std::pow(10.0, static_cast<double>(rng() % 9) - 4.0));
                }
        model::LpProblem lp;
        lp.A = sparse::from_triplets(m, n, rows, cols, vals);
        lp.c.assign(static_cast<std::size_t>(n), 0.0);
        lp.col_lo.resize(static_cast<std::size_t>(n));
        lp.col_hi.resize(static_cast<std::size_t>(n));
        for (core::Index j = 0; j < n; ++j) {
            lp.col_lo[j] = -1.0 - 3.0 * std::fabs(U(rng));
            lp.col_hi[j] = lp.col_lo[j] + 0.5 + 4.0 * std::fabs(U(rng));
        }
        std::vector<double> y(static_cast<std::size_t>(m));
        for (auto& v : y) v = rng() % 5 == 0 ? 0.0 : U(rng);
        core::Index pivot = -1;
        for (core::Index i = 0; i < m; ++i) if (y[i] != 0.0) pivot = i;
        if (pivot < 0) continue;
        // lower for these multipliers, exactly.
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        const auto exact_lower = [&](const std::vector<double>& yy) {
            std::vector<model::Rational> d(static_cast<std::size_t>(n), 0);
            for (core::Index i = 0; i < m; ++i)
                for (auto k = rp[i]; k < rp[i + 1]; ++k)
                    d[ci[k]] += model::Rational(yy[i]) * model::Rational(lp.A.vals[k]);
            model::Rational lower = 0;
            for (core::Index j = 0; j < n; ++j) {
                if (d[j] > 0) lower += d[j] * model::Rational(lp.col_lo[j]);
                else if (d[j] < 0) lower += d[j] * model::Rational(lp.col_hi[j]);
            }
            return lower;
        };
        const double lower = exact_lower(y).convert_to<double>();
        // Sides: the one each multiplier's sign uses, the other open or not.
        lp.row_lo.resize(static_cast<std::size_t>(m));
        lp.row_hi.resize(static_cast<std::size_t>(m));
        double upper_rest = 0.0;
        for (core::Index i = 0; i < m; ++i) {
            const double side = 3.0 * U(rng);
            const double other = rng() % 2 ? model::kInf : 2.0 + std::fabs(U(rng));
            if (y[i] >= 0.0) { lp.row_hi[i] = side; lp.row_lo[i] = side - other; }
            else { lp.row_lo[i] = side; lp.row_hi[i] = side + other; }
            if (i != pivot) upper_rest += y[i] * side;
        }
        const double factor = std::pow(10.0, 2.0 * std::fabs(U(rng)) - 1.0);
        const double sign = rng() % 4 == 0 ? -1.0 : 1.0;   // some feasible-side rays
        const double target_upper =
            lower - sign * factor * tol * (1.0 + std::fabs(lower));
        const double side = (target_upper - upper_rest) / y[pivot];
        if (y[pivot] > 0.0) { lp.row_hi[pivot] = side; lp.row_lo[pivot] = -model::kInf; }
        else { lp.row_lo[pivot] = side; lp.row_hi[pivot] = model::kInf; }

        const auto got = certify::check_dual_farkas_ray(lp, y, tol);
        // The exact rule on the multipliers the check settled on.
        bool want = false;
        if (got.multipliers.size() == y.size()) {
            const auto& w = got.multipliers;
            const model::Rational lo_exact = exact_lower(w);
            model::Rational up_exact = 0;
            for (core::Index i = 0; i < m; ++i) {
                if (w[i] > 0.0) up_exact += model::Rational(w[i]) * model::Rational(lp.row_hi[i]);
                else if (w[i] < 0.0) up_exact += model::Rational(w[i]) * model::Rational(lp.row_lo[i]);
            }
            const double contradiction = model::rounded_down(lo_exact - up_exact);
            const model::Rational mag = boost::multiprecision::max(
                boost::multiprecision::abs(lo_exact), boost::multiprecision::abs(up_exact));
            const double sep = model::rounded_up(model::Rational(tol) * (1 + mag));
            want = std::isfinite(contradiction) && contradiction > sep;
        }
        CHECK(got.certified == want);
        // The engine's strict rule (no tolerance, declared bounds only) on
        // the multipliers as given.
        {
            const model::Rational lo_exact = exact_lower(y);
            model::Rational up_exact = 0;
            for (core::Index i = 0; i < m; ++i) {
                if (y[i] > 0.0) up_exact += model::Rational(y[i]) * model::Rational(lp.row_hi[i]);
                else if (y[i] < 0.0) up_exact += model::Rational(y[i]) * model::Rational(lp.row_lo[i]);
            }
            CHECK((engines::farkas_violation(lp, y) == 0.0) == (lo_exact > up_exact));
        }
        if (got.certified) {
            ++certified;
            CHECK(got.max_sign_residual == 0.0);
        } else {
            ++refused;
        }
    }
    CHECK(certified > 150);
    CHECK(refused > 150);

    // Cancellation below long double precision. Column x0's terms are
    // 1, 3*2^-66, -1, -2^-65: exactly d0 = +2^-66, but summed in long double
    // (1 + 3*2^-66 rounds to 1) it comes out -2^-65. With x0 in
    // [-2e20, -1e20] the sign decides the bound: the exact rule charges d0 to
    // the lower bound (-2.71) and refuses the ray against upper = -1.71;
    // trusting the computed sign would charge it to the upper bound (+2.71)
    // and certify. Against upper = -10 both verdicts certify.
    {
        const double t66 = std::ldexp(1.0, -66), t65 = std::ldexp(1.0, -65);
        model::LpProblem lp;
        lp.A = sparse::from_triplets(
            4, 2, {0, 0, 1, 2, 3}, {0, 1, 0, 0, 0},
            {1.0, 1.0, 3.0 * t66, -1.0, -t65});
        lp.c.assign(2, 0.0);
        lp.col_lo = {-2e20, 0.0};
        lp.col_hi = {-1e20, 1.0};
        lp.row_lo.assign(4, -model::kInf);
        lp.row_hi = {-1.71, 0.0, 0.0, 0.0};
        const std::vector<double> y(4, 1.0);
        CHECK(!certify::check_dual_farkas_ray(lp, y, tol).certified);
        CHECK(engines::farkas_violation(lp, y) == core::kPosInf);
        lp.row_hi[0] = -10.0;
        CHECK(certify::check_dual_farkas_ray(lp, y, tol).certified);
        CHECK(engines::farkas_violation(lp, y) == 0.0);
    }
}

int main() {
    test_interval_rc_never_tighter_than_exact();
    test_float_rays_skip_exact_repair_without_exact_proof();
    test_farkas_check_matches_exact_rule();
    test_presolved_terminal_witnesses();
    test_exact_terminal_rays();
    test_sparse_integer_basis_certificate();
    test_targeted_dual_certificate();    test_exact_dual_rational_reference();
    test_safe_lagrangian_bound();
    test_interval_lagrangian_matches_exact();
    test_safe_bound_refuses_unbounded_direction();
    test_safe_bound_snaps_straddling_enclosure_on_infinite_side();
    test_small_cost_cannot_erase_unbounded_direction();
    test_checked_repair_preserves_sense_and_offset();
    const auto infeasible = read(R"(NAME INF
ROWS
 N OBJ
 L R1
COLUMNS
 X OBJ 0 R1 1
RHS
 RHS R1 1
BOUNDS
 LO BND X 2
ENDATA
)");
    const auto dual = certify::check_dual_farkas_ray(infeasible, {1.0}, 1e-7);
    CHECK(dual.certified);
    CHECK(dual.contradiction > 0.9);
    CHECK(!certify::check_dual_farkas_ray(infeasible, {0.0}, 1e-7).certified);

    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.dual_farkas_ray = dual;
        core::ProofEvidence ev;
        ev.claimed_level = core::ProofLevel::BoundOnly;
        ev.dual_farkas_violation = dual.max_sign_residual;
        ev.dual_farkas_contradiction = dual.contradiction;
        const auto result = certify::finalize_result(std::move(raw), ev);
        CHECK(result.status == core::Status::Infeasible);
        CHECK(result.dual_farkas_ray.certified);
        CHECK(result.ray_certified);
        CHECK(result.ray == result.dual_farkas_ray.multipliers);
    }

    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Infeasible;
        raw.dual_farkas_ray.multipliers = {1.0};
        core::ProofEvidence ev;
        const auto result = certify::finalize_result(std::move(raw), ev);
        CHECK(result.status == core::Status::NoSolutionFound);
        CHECK(!result.downgrade_reason.empty());
    }

    const auto unbounded = read(R"(NAME UNB
ROWS
 N OBJ
COLUMNS
 X OBJ -1
BOUNDS
 LO BND X 0
ENDATA
)");
    const auto primal = certify::check_primal_ray(unbounded, {1.0}, 1e-7);
    CHECK(primal.certified);
    CHECK(primal.objective_direction < -0.9);
    CHECK(!certify::check_primal_ray(unbounded, {-1.0}, 1e-7).certified);

    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Unbounded;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.primal_ray = primal;
        raw.x = {0.0};
        core::ProofEvidence ev;
        ev.claimed_level = core::ProofLevel::BoundOnly;
        ev.checker_passed = true;
        ev.max_primal_violation = 0.0;
        ev.primal_ray_violation = std::max(primal.max_row_residual,
                                           primal.max_bound_sign_residual);
        ev.primal_ray_objective = primal.objective_direction;
        const auto result = certify::finalize_result(std::move(raw), ev);
        CHECK(result.status == core::Status::Unbounded);
        CHECK(result.primal_ray.certified);
    }

    const auto feasible = read(R"(NAME FEAS
ROWS
 N OBJ
 L R1
COLUMNS
 X OBJ 0 R1 1
RHS
 RHS R1 2
BOUNDS
 LO BND X 2
ENDATA
)");

    // The aggregate checker, used by the CLI, must ignore producer residuals
    // and certify from the original model.  Its objective convention remains
    // correct for maximization as well as minimization.
    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Unbounded;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.primal_ray.direction = {1.0};
        raw.x = {0.0};
        core::ProofEvidence proposed;
        proposed.claimed_level = core::ProofLevel::BoundOnly;
        proposed.primal_feas_tol = 1e-7;
        auto checked = certify::check_lp_result(unbounded, raw, proposed);
        const auto result = certify::finalize_result(std::move(raw), checked);
        CHECK(result.status == core::Status::Unbounded);
        CHECK(result.primal_ray.certified);
    }

    {
        auto maximize_unbounded = unbounded;
        maximize_unbounded.maximize = true;
        maximize_unbounded.c = {1.0};
        core::RawResult raw;
        raw.proposed_status = core::Status::Unbounded;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.primal_ray.direction = {1.0};
        raw.x = {0.0};
        core::ProofEvidence proposed;
        proposed.claimed_level = core::ProofLevel::BoundOnly;
        auto checked = certify::check_lp_result(
            maximize_unbounded, raw, proposed);
        const auto result = certify::finalize_result(std::move(raw), checked);
        CHECK(result.status == core::Status::Unbounded);
        CHECK(result.primal_ray.certified);
    }

    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.dual_farkas_ray.multipliers = {1.0};
        core::ProofEvidence proposed;
        proposed.claimed_level = core::ProofLevel::BoundOnly;
        // Deliberately lie in the producer evidence; independent checking of
        // the feasible model below must still reject the claim.
        proposed.dual_farkas_violation = 0.0;
        proposed.dual_farkas_contradiction = 100.0;
        auto checked = certify::check_lp_result(feasible, raw, proposed);
        const auto result = certify::finalize_result(std::move(raw), checked);
        CHECK(result.status == core::Status::NoSolutionFound);
    }

    {
        core::RawResult raw;
        raw.proposed_status = core::Status::Unbounded;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.primal_ray.direction = {-1.0};
        core::ProofEvidence proposed;
        proposed.claimed_level = core::ProofLevel::BoundOnly;
        // Producer metadata claims a perfect improving direction, but the
        // direction violates the lower-bound recession cone.  The independent
        // original-model checker must reject it.
        proposed.primal_ray_violation = 0.0;
        proposed.primal_ray_objective = -1.0;
        auto checked = certify::check_lp_result(unbounded, raw, proposed);
        const auto result = certify::finalize_result(std::move(raw), checked);
        CHECK(result.status == core::Status::NoSolutionFound);
        CHECK(!result.primal_ray.certified);
    }

    // A zero-separation near-certificate must not certify infeasibility.
    CHECK(!certify::check_dual_farkas_ray(feasible, {1.0}, 1e-7).certified);

    // Finite support terms are never tolerance-zeroed.  This LP is feasible
    // at x=-1e8.  Dropping the small 1e-8 A'y term before multiplying it by
    // the large finite lower bound would manufacture lower=0 > upper=-1 and
    // falsely certify infeasibility.
    {
        model::LpProblem scaled_support;
        scaled_support.name = "finite-support-near-certificate";
        scaled_support.A = sparse::from_triplets(
            1, 1, {0}, {0}, {1e-8});
        scaled_support.c = {0.0};
        scaled_support.col_lo = {-1e9};
        scaled_support.col_hi = {1e9};
        scaled_support.row_lo = {-model::kInf};
        scaled_support.row_hi = {-1.0};
        const auto near = certify::check_dual_farkas_ray(
            scaled_support, {1.0}, 1e-7);
        CHECK(!near.certified);
        CHECK(near.contradiction < 0.0);

        core::RawResult raw;
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.dual_farkas_ray.multipliers = {1.0};
        core::ProofEvidence lie;
        lie.claimed_level = core::ProofLevel::BoundOnly;
        lie.dual_farkas_violation = 0.0;
        lie.dual_farkas_contradiction = 1.0;
        const auto checked = certify::check_lp_result(
            scaled_support, raw, lie);
        const auto result = certify::finalize_result(std::move(raw), checked);
        CHECK(result.status == core::Status::NoSolutionFound);
        CHECK(!result.ray_certified);
    }

    // Even a coefficient below feasibility tolerance cannot be discarded
    // when its variable is unbounded: x=-1e8 satisfies 1e-8*x <= -1.
    // Ignoring that term would fabricate a positive Farkas contradiction.
    {
        model::LpProblem unbounded_support;
        unbounded_support.name = "unbounded-support-near-certificate";
        unbounded_support.A = sparse::from_triplets(
            1, 1, {0}, {0}, {1e-8});
        unbounded_support.c = {0.0};
        unbounded_support.col_lo = {-model::kInf};
        unbounded_support.col_hi = {model::kInf};
        unbounded_support.row_lo = {-model::kInf};
        unbounded_support.row_hi = {-1.0};
        CHECK(!certify::check_dual_farkas_ray(
            unbounded_support, {1.0}, 1e-7).certified);
    }

    // An approximate ray for x >= 1, x <= 0 with x free leaves a small A'y
    // residual on the free column. The checker may repair it into the exact
    // certificate, but it must re-verify the repaired ray, and it must not
    // repair into a multiplier whose sign faces an infinite row side.
    {
        model::LpProblem near_free;
        near_free.name = "near-free-column-farkas";
        near_free.A = sparse::from_triplets(2, 1, {0, 1}, {0, 0}, {1.0, 1.0});
        near_free.c = {0.0};
        near_free.col_lo = {-model::kInf};
        near_free.col_hi = {model::kInf};
        near_free.row_lo = {1.0, -model::kInf};
        near_free.row_hi = {model::kInf, 0.0};
        const auto repaired = certify::check_dual_farkas_ray(
            near_free, {-1.0, 1.0 - 1e-11}, 1e-7);
        CHECK(repaired.certified);
        // The feasible variant (x >= 0, x <= 1) has no certificate at all.
        near_free.row_lo = {0.0, -model::kInf};
        near_free.row_hi = {model::kInf, 1.0};
        CHECK(!certify::check_dual_farkas_ray(
            near_free, {-1.0, 1.0 - 1e-11}, 1e-7).certified);
    }

    // Ranged rows and free columns need the same original-space separation
    // convention.  Two opposing one-sided rows make the free-variable model
    // infeasible while cancelling A'y exactly.
    {
        model::LpProblem free_column;
        free_column.name = "free-column-farkas";
        free_column.A = sparse::from_triplets(
            2, 1, {0, 1}, {0, 0}, {1.0, 1.0});
        free_column.c = {0.0};
        free_column.col_lo = {-model::kInf};
        free_column.col_hi = {model::kInf};
        free_column.row_lo = {1.0, -model::kInf};
        free_column.row_hi = {model::kInf, 0.0};
        CHECK(certify::check_dual_farkas_ray(
            free_column, {-1.0, 1.0}, 1e-7).certified);

        model::LpProblem ranged;
        ranged.name = "ranged-row-farkas";
        ranged.A = sparse::from_triplets(1, 1, {0}, {0}, {1.0});
        ranged.c = {0.0};
        ranged.col_lo = {0.0};
        ranged.col_hi = {0.0};
        ranged.row_lo = {2.0};
        ranged.row_hi = {3.0};
        CHECK(certify::check_dual_farkas_ray(
            ranged, {-1.0}, 1e-7).certified);
    }

    // Audit F1/F2: raw numbers and an invalid point cannot create a proof.
    {
        model::LpProblem box;
        box.A = sparse::from_triplets(0, 1, {}, {}, {});
        box.c = {1}; box.col_lo = {0}; box.col_hi = {1};
        core::RawResult raw;
        raw.x = {0}; raw.objective = 123; raw.dual_bound = 456;
        raw.proposed_status = core::Status::Optimal;
        raw.proposed_level = core::ProofLevel::ProvedOptimalFP;
        core::ProofEvidence proposed;
        proposed.has_basis = true;
        proposed.claimed_level = raw.proposed_level;
        auto checked = certify::check_lp_result(box, raw, proposed);
        auto result = certify::finalize_result(raw, checked);
        CHECK_NEAR(result.objective, 0, 0);
        CHECK_NEAR(result.dual_bound, 0, 0);
        CHECK(result.status == core::Status::NumericalFailure);
        raw.x = {2};
        for (const auto status : {core::Status::Optimal, core::Status::Feasible}) {
            raw.proposed_status = status;
            checked = certify::check_lp_result(box, raw, proposed);
            result = certify::finalize_result(raw, checked);
            CHECK(result.status != core::Status::Feasible);
            CHECK(result.status != core::Status::Optimal);
            CHECK(result.proof < core::ProofLevel::FeasibleOnly);
        }
    }
    // Audit F3: an improving direction on an empty polyhedron has no anchor.
    {
        model::LpProblem p;
        p.A = sparse::from_triplets(2, 2, {0,1}, {0,0}, {1,1});
        p.c = {0,-1}; p.col_lo = {-model::kInf,0};
        p.col_hi = {model::kInf,model::kInf};
        p.row_lo = {1,-model::kInf}; p.row_hi = {model::kInf,0};
        core::RawResult raw;
        raw.x = {0,0}; raw.y = {0,0};
        raw.proposed_status = core::Status::Unbounded;
        raw.primal_ray.direction = {0,1};
        const auto ev = certify::check_lp_result(p, raw, {});
        const auto result = certify::finalize_result(raw, ev);
        CHECK(result.status == core::Status::NoSolutionFound);
        CHECK(!result.primal_ray.certified);
    }
    // Audit F16: exact binary64 cancellation must retain the unit term.
    {
        model::LpProblem p;
        p.A = sparse::from_triplets(1, 4, {0,0,0,0}, {0,1,2,3}, {1e10,1,-1e10,1});
        p.c = {0,0,0,1}; p.col_lo = {1e10,1,1e10,-model::kInf};
        p.col_hi = {1e10,1,1e10,model::kInf}; p.row_lo = {0}; p.row_hi = {0};
        CHECK_NEAR(p.max_row_violation({1e10,1,1e10,0}), 1, 0);
        CHECK_NEAR(p.max_row_violation({1e10,1,1e10,-1}), 0, 0);
        const auto bound = certify::safe_lagrangian_lower_bound(p, {0}, p.col_lo, p.col_hi);
        CHECK(bound.finite);
        CHECK(bound.value <= -1);
        CHECK(bound.value >= -1.00000001);
    }
    // Model boundaries reject each category of nonfinite data.
    {
        model::LpProblem box;
        box.A = sparse::from_triplets(0, 1, {}, {}, {});
        box.c = {1}; box.col_lo = {0}; box.col_hi = {1};
        for (int kind = 0; kind < 4; ++kind) {
            auto invalid = box;
            if (kind == 0) invalid.c[0] = core::kNaN;
            if (kind == 1) invalid.obj_offset = core::kPosInf;
            if (kind == 2) invalid.col_lo[0] = core::kPosInf;
            if (kind == 3) invalid.col_hi[0] = -core::kPosInf;
            bool rejected = false;
            try { invalid.validate(); } catch (const std::invalid_argument&) { rejected = true; }
            CHECK(rejected);
        }
    }
    return test::finish("test_lp_certificates");
}
