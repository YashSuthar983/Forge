#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "test_helpers.hpp"
#include "sor/model/exact.hpp"
#include <random>

#include <sstream>
#include <limits>

using namespace sor;

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

void test_small_cost_cannot_erase_unbounded_direction() {
    // The cost is inside the KKT tolerance, but its product with the
    // unbounded upper box is still -infinity. A fake zero gap is unsound.
    const auto source = read(R"(NAME TINY
ROWS
 N OBJ
COLUMNS
 X OBJ -0.000000001
ENDATA
)");
    for (bool maximize : {false, true}) {
        auto lp = source;
        lp.maximize = maximize;
        lp.c[0] = maximize ? 1e-9 : -1e-9;
        lp.obj_offset = 5.0;
        core::RawResult fake;
        fake.x = {0.0};
        fake.objective = fake.dual_bound = 5.0;
        fake.proposed_status = core::Status::Optimal;
        fake.proposed_level = core::ProofLevel::ProvedOptimalFP;
        auto checked = certify::check_lp_point(lp, fake, 1e-7, 1e-7, 1e-9, true);
        CHECK(checked.checker_passed); // The primal point itself is feasible.
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
    nonfinite.x = {sor::model::kInf};
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

int main() {
    test_presolved_terminal_witnesses();
    test_exact_terminal_rays();
    test_sparse_integer_basis_certificate();    test_exact_dual_rational_reference();
    test_safe_lagrangian_bound();
    test_safe_bound_refuses_unbounded_direction();
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
