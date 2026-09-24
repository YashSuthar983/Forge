// Convex QCQP in the interior point (Track 2 B9).
//
// Hand-checked optima for each way a quadratic row can be convex, the
// refusals for each way it can be nonconvex, the fixed-column presolve of a
// Hessian, infeasibility that must not be claimed, and the QCQP claim check
// against the QP one it extends.
#include "sor/certify/finalize.hpp"
#include "sor/engines/qcqp.hpp"
#include "sor/search/miqp_bb.hpp"
// Internal: the original-units check is compared with the QP one directly.
#include "../src/engines/src/qp_common.hpp"
#include "sor/sparse/csr.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <random>
#include <string>
#include <vector>

using namespace sor;
using core::f64;
using core::Index;
constexpr f64 kInf = model::kInf;

namespace {

struct Row {
    std::vector<std::pair<Index, f64>> lin;
    std::vector<std::tuple<Index, Index, f64>> quad;   // (r <= c, v): QuadRow convention
    f64 lo, hi;
};

// min 1/2 x'Q0x + c'x + off  s.t. rows, lo <= x <= hi.  Q0 as upper-triangle
// (r, c, v) with QuadRow's symmetric convention.
engines::QcqpProblem make(std::vector<f64> c, std::vector<f64> lo, std::vector<f64> hi,
                          std::vector<Row> rows,
                          std::vector<std::tuple<Index, Index, f64>> q0 = {}, f64 off = 0.0) {
    engines::QcqpProblem p;
    const Index n = static_cast<Index>(c.size()), m = static_cast<Index>(rows.size());
    auto& lp = p.qp.linear;
    std::vector<Index> r, cc;
    std::vector<f64> v;
    for (Index i = 0; i < m; ++i) {
        const auto& row = rows[static_cast<std::size_t>(i)];
        for (auto [j, a] : row.lin) { r.push_back(i); cc.push_back(j); v.push_back(a); }
        lp.row_lo.push_back(row.lo);
        lp.row_hi.push_back(row.hi);
        if (!row.quad.empty()) {
            engines::QuadRow q;
            q.row = i;
            for (auto [a, b, w] : row.quad) { q.r.push_back(a); q.c.push_back(b); q.v.push_back(w); }
            p.quad.push_back(q);
        }
    }
    lp.A = sparse::from_triplets(m, n, r, cc, v);
    lp.c = c;
    lp.obj_offset = off;
    lp.col_lo = lo;
    lp.col_hi = hi;
    lp.is_integer.assign(static_cast<std::size_t>(n), false);
    std::vector<Index> qr, qc;
    std::vector<f64> qv;
    for (auto [a, b, w] : q0) {
        qr.push_back(a); qc.push_back(b); qv.push_back(w);
        if (a != b) { qr.push_back(b); qc.push_back(a); qv.push_back(w); }
    }
    p.qp.q_matrix = sparse::from_triplets(n, n, qr, qc, qv);
    p.validate();
    return p;
}

void expect(const std::string& label, const engines::QcqpProblem& p, std::vector<f64> x_star, f64 obj) {
    engines::QpOptions o;
    engines::QpDiagnostics d;
    const auto r = engines::solve_qcqp_ipm(p, o, d);
    bool ok = r.proposed_status == core::Status::Optimal &&
              r.proposed_level == core::ProofLevel::ProvedKKT &&
              std::fabs(r.objective - obj) <= 1e-7 * (1.0 + std::fabs(obj));
    for (std::size_t j = 0; j < x_star.size() && ok; ++j)
        ok = std::fabs(r.x[j] - x_star[j]) <= 1e-6 * (1.0 + std::fabs(x_star[j]));
    ::sor::test::report(ok, "QCQP IPM hand-checked optimum", __FILE__, __LINE__,
                        label + ": status " + std::to_string(static_cast<int>(r.proposed_status)) +
                            " obj " + std::to_string(r.objective) + " want " + std::to_string(obj) +
                            " it " + std::to_string(d.iterations) + " " + r.termination_reason);
    // The claim finalises: evidence from the same diagnostics.
    auto raw = r;
    const auto fr = certify::finalize_result(std::move(raw), engines::qp_evidence(d, o));
    ::sor::test::report(!ok || fr.status == core::Status::Optimal, "claim finalises", __FILE__,
                        __LINE__, label);
}

void expect_refused(const std::string& label, const engines::QcqpProblem& p, const std::string& needle) {
    engines::QpOptions o;
    engines::QpDiagnostics d;
    const auto r = engines::solve_qcqp_ipm(p, o, d);
    ::sor::test::report(r.proposed_status == core::Status::Unsupported && r.x.empty() &&
                            r.termination_reason.find(needle) != std::string::npos,
                        "nonconvex QCQP refused", __FILE__, __LINE__,
                        label + ": " + r.termination_reason);
}

void test_hand_checked() {
    const f64 s2 = std::sqrt(2.0);
    // min x + y  s.t.  x^2 + y^2 <= 1  ->  (-1/sqrt2, -1/sqrt2), -sqrt2.
    // (QuadRow diagonal v contributes 1/2 v x^2, so v = 2 gives x^2.)
    expect("disk, linear objective",
           make({1, 1}, {-kInf, -kInf}, {kInf, kInf}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 1.0}}),
           {-1 / s2, -1 / s2}, -s2);
    // The same set written concave-above: -x^2 - y^2 >= -1 (NSD, lower bound).
    expect("disk written as a concave row >= level",
           make({1, 1}, {-kInf, -kInf}, {kInf, kInf}, {{{}, {{0, 0, -2.0}, {1, 1, -2.0}}, -1.0, kInf}}),
           {-1 / s2, -1 / s2}, -s2);
    // Maximise x + y = min -(x + y) over the shifted disk (x-1)^2 + y^2 <= 1:
    // x^2 - 2x + y^2 <= 0 -> x = 1 + 1/sqrt2, y = 1/sqrt2, value -(1 + sqrt2).
    expect("shifted disk, linear part in the row",
           make({-1, -1}, {-kInf, -kInf}, {kInf, kInf},
                {{{{0, -2.0}}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 0.0}}),
           {1 + 1 / s2, 1 / s2}, -(1 + s2));
    // Projection of (2,1) onto the unit disk: min (x-2)^2 + (y-1)^2 =
    // 1/2 x'(2I)x - 4x - 2y + 5  ->  (2,1)/sqrt5, value (sqrt5 - 1)^2.
    const f64 s5 = std::sqrt(5.0);
    expect("projection onto the disk (quadratic objective)",
           make({-4, -2}, {-kInf, -kInf}, {kInf, kInf}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 1.0}},
                {{0, 0, 2.0}, {1, 1, 2.0}}, 5.0),
           {2 / s5, 1 / s5}, (s5 - 1) * (s5 - 1));
    // Inactive quadratic row: min (x-0.1)^2 + (y-0.2)^2 in the unit disk.
    expect("inactive quadratic row",
           make({-0.2, -0.4}, {-kInf, -kInf}, {kInf, kInf}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 1.0}},
                {{0, 0, 2.0}, {1, 1, 2.0}}, 0.05),
           {0.1, 0.2}, 0.0);
    // Off-diagonal Hessian: x^2 + xy + y^2 <= 3 (PSD: eigenvalues 1/2, 3/2 of
    // [[1, .5], [.5, 1]]).  min -x - y: by symmetry x = y, 3x^2 = 3 -> x = 1.
    expect("off-diagonal PSD row",
           make({-1, -1}, {-kInf, -kInf}, {kInf, kInf},
                {{{}, {{0, 0, 2.0}, {0, 1, 1.0}, {1, 1, 2.0}}, -kInf, 3.0}}),
           {1.0, 1.0}, -2.0);
    // Two quadratic rows and a linear one, bounds on x: min -y s.t.
    // x^2 + y^2 <= 4, (x-2)^2 + y^2 <= 4, x <= 1.5 (linear), 0 <= x.
    // Lens of two radius-2 disks centred 0 and 2: top at x = 1, y = sqrt3.
    expect("lens of two disks",
           make({0, -1}, {0.0, -kInf}, {kInf, kInf},
                {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 4.0},
                 {{{0, -4.0}}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 0.0},
                 {{{0, 1.0}}, {}, -kInf, 1.5}}),
           {1.0, std::sqrt(3.0)}, -std::sqrt(3.0));
}

void test_fixed_columns() {
    // x fixed at 0.5 by its bounds; row x^2 + xy + y^2 <= 1 becomes
    // y^2 + 0.5 y + 0.25 <= 1: the fixed column feeds a LINEAR term (0.5 y)
    // and a constant.  max y -> y = (-0.5 + sqrt(3.25)) / 2.
    const f64 y = (-0.5 + std::sqrt(3.25)) / 2.0;
    expect("fixed column inside a Hessian",
           make({0, -1}, {0.5, -kInf}, {0.5, kInf},
                {{{}, {{0, 0, 2.0}, {0, 1, 1.0}, {1, 1, 2.0}}, -kInf, 1.0}}),
           {0.5, y}, -y);
    // Every column of the row fixed: it is a constant, checked, not solved.
    expect("quadratic row with all columns fixed",
           make({0, 1}, {0.5, 0.0}, {0.5, 3.0}, {{{}, {{0, 0, 2.0}}, -kInf, 1.0}}), {0.5, 0.0}, 0.0);
}

void test_refusals() {
    // x^2 + y^2 >= 1: a PSD Hessian bounded BELOW -- the outside of a disk.
    expect_refused("outside of a disk",
                   make({1, 1}, {-1, -1}, {1, 1}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, 1.0, kInf}}),
                   "quadratic constraint");
    // x^2 + y^2 = 1: a circle.
    expect_refused("circle (equality)",
                   make({1, 1}, {-1, -1}, {1, 1}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, 1.0, 1.0}}),
                   "ranged or an equality");
    // xy <= 1: indefinite.
    expect_refused("bilinear row", make({1, 1}, {-3, -3}, {3, 3}, {{{}, {{0, 1, 1.0}}, -kInf, 1.0}}),
                   "not certified convex");
    // Nonconvex objective with convex rows: refused on the objective.
    expect_refused("nonconvex objective",
                   make({0, 0}, {-1, -1}, {1, 1}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 1.0}},
                        {{0, 1, 1.0}}),
                   "objective");
    // Integer columns: the continuous IPM does not take them.
    auto p = make({1, 1}, {-2, -2}, {2, 2}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 1.0}});
    p.qp.linear.is_integer[0] = true;
    expect_refused("integer column", p, "integer");
}

void test_infeasible_not_claimed() {
    // x^2 + y^2 <= 1 with x >= 2: empty.  Whatever the IPM does, no Optimal.
    auto p = make({1, 1}, {2.0, -kInf}, {kInf, kInf}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 1.0}});
    engines::QpOptions o;
    o.time_limit_s = 5.0;
    engines::QpDiagnostics d;
    const auto r = engines::solve_qcqp_ipm(p, o, d);
    CHECK(r.proposed_status != core::Status::Optimal);
    CHECK(r.proposed_level != core::ProofLevel::ProvedKKT);
}

void test_check_extends_qp_check() {
    // With no quadratic rows the QCQP check is the QP check, bit for bit.
    std::mt19937 rng(7);
    std::uniform_real_distribution<f64> U(-2.0, 2.0);
    for (int trial = 0; trial < 20; ++trial) {
        auto p = make({U(rng), U(rng), U(rng)}, {-1, -kInf, 0}, {1, 2, kInf},
                      {{{{0, U(rng)}, {1, U(rng)}}, {}, -1.0, 1.0}, {{{1, U(rng)}, {2, 1.0}}, {}, -kInf, 2.0}},
                      {{0, 0, 2.0}, {0, 1, 0.5}, {2, 2, 1.0}});
        const std::vector<f64> x{U(rng), U(rng), U(rng)}, y{U(rng), U(rng)};
        const backend::LaneBounds b{p.qp.linear.col_lo, p.qp.linear.col_hi, p.qp.linear.row_lo,
                                    p.qp.linear.row_hi};
        const auto a = engines::qpc::kkt_original(p.qp, b, x, y);
        const auto q = engines::qpc::kkt_original_qcqp(p, b, x, y);
        CHECK(a.primal == q.primal && a.dual_res == q.dual_res && a.primal_net == q.primal_net &&
              a.dual_net == q.dual_net && a.objective == q.objective && a.gap_finite == q.gap_finite);
        CHECK(!a.gap_finite || (a.gap == q.gap && a.gap_net == q.gap_net && a.dual_bound == q.dual_bound));
    }
    // With a quadratic row: the activity and Jacobian are the Hessian's.
    // Row x^2 + y^2 <= 1 at x = (0.6, 0.8): activity exactly 1, gradient
    // (1.2, 1.6).  min x + y has r = c + y_row * (1.2, 1.6); y_row = -1/1.6
    // with... choose the KKT point of min -0.6x - 0.8y: y_row = 0.5 gives
    // r = (-0.6 + 0.6, -0.8 + 0.8) = 0: stationarity 0, primal 0, and the
    // gap: primal -1, dual -(0.5 * 1) - 0.5*1 (py = hi*y) = -1.
    auto p = make({-0.6, -0.8}, {-kInf, -kInf}, {kInf, kInf}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 1.0}});
    const backend::LaneBounds b{p.qp.linear.col_lo, p.qp.linear.col_hi, p.qp.linear.row_lo,
                                p.qp.linear.row_hi};
    const auto k = engines::qpc::kkt_original_qcqp(p, b, {0.6, 0.8}, {0.5});
    CHECK(k.primal <= 1e-15);
    CHECK(k.dual_res <= 1e-15);
    CHECK(k.gap_finite);
    CHECK_NEAR(k.objective, -1.0, 1e-15);
    CHECK_NEAR(k.dual_bound, -1.0, 1e-15);
    // A point outside the disk is primal infeasible by exactly its excess.
    const auto k2 = engines::qpc::kkt_original_qcqp(p, b, {1.2, 1.6}, {0.5});
    CHECK_NEAR(k2.primal, 3.0, 1e-14);
}

bool miqcqp_bb_ok(const engines::QcqpProblem& p, const search::MiqpBbOptions& o,
                  const search::MiqpBbDiagnostics& d, const core::RawResult& r) {
    return search::miqcqp_bb_evidence(p, o, d, r).checker_passed;
}

void test_miqcqp() {
    // min -x - y, x, y integer in [-3, 3], x^2 + y^2 <= 5: the lattice points
    // of the disk of radius sqrt5 maximising x + y are (1,2) and (2,1): -3.
    // (The continuous optimum is -sqrt10 = -3.162, so the tree must work.)
    {
        auto p = make({-1, -1}, {-3, -3}, {3, 3}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 5.0}});
        p.qp.linear.is_integer = {true, true};
        search::MiqpBbOptions o;
        search::MiqpBbDiagnostics d;
        const auto r = search::solve_miqcqp_bb(p, o, d);
        CHECK(r.proposed_status == core::Status::Optimal);
        CHECK_NEAR(r.objective, -3.0, 1e-9);
        CHECK(d.proved);
        const auto ev = search::miqcqp_bb_evidence(p, o, d, r);
        CHECK(ev.checker_passed);
        const auto e = engines::evaluate_qcqp(p, r.x);
        CHECK(e.max_violation() == 0.0);
    }
    // Mixed: min -x - 2y, x integer in [0, 3], y continuous, x^2 + y^2 <= 4.5.
    // x = 0: -2*sqrt4.5 = -4.2426;  x = 1: -1 - 2*sqrt3.5 = -4.7417;
    // x = 2: -2 - 2*sqrt0.5 = -3.4142.  Optimum x = 1, y = sqrt3.5.
    {
        auto p = make({-1, -2}, {0, -kInf}, {3, kInf}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 4.5}});
        p.qp.linear.is_integer = {true, false};
        search::MiqpBbOptions o;
        search::MiqpBbDiagnostics d;
        const auto r = search::solve_miqcqp_bb(p, o, d);
        CHECK(r.proposed_status == core::Status::Optimal);
        CHECK_NEAR(r.objective, -1.0 - 2.0 * std::sqrt(3.5), 1e-6);
        CHECK(r.x.size() == 2 && r.x[0] == 1.0);
        CHECK(miqcqp_bb_ok(p, o, d, r));
    }
    // Infeasible: x integer, x^2 <= 0.25 and x >= 0.5 by bounds [0.5, 3]:
    // the integer box is {1, 2, 3}, none with x^2 <= 0.25.
    {
        auto p = make({1}, {0.5}, {3}, {{{}, {{0, 0, 2.0}}, -kInf, 0.25}});
        p.qp.linear.is_integer = {true};
        search::MiqpBbOptions o;
        search::MiqpBbDiagnostics d;
        const auto r = search::solve_miqcqp_bb(p, o, d);
        CHECK(r.proposed_status == core::Status::Infeasible);
    }
    // A row nonconvex only on bounded INTEGER columns is relaxed exactly by
    // the secant shift: x^2 + y^2 >= 5 (outside of a disk), x, y integer in
    // [-3, 3], min x + y  ->  (-3, -3), -6 (the box corner is outside).
    {
        auto p = make({1, 1}, {-3, -3}, {3, 3}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, 5.0, kInf}});
        p.qp.linear.is_integer = {true, true};
        search::MiqpBbOptions o;
        search::MiqpBbDiagnostics d;
        const auto r = search::solve_miqcqp_bb(p, o, d);
        CHECK(r.proposed_status == core::Status::Optimal);
        CHECK_NEAR(r.objective, -6.0, 1e-9);
        CHECK(d.row_sigma_max > 0.0);
    }
    // Same row with x + y <= 1.5: the best integer points outside the disk
    // of radius sqrt5 with x + y <= 1 have x + y = 1 ((2,-1): 4 + 1 = 5).
    {
        auto p = make({-1, -1}, {-3, -3}, {3, 3},
                      {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, 5.0, kInf}, {{{0, 1.0}, {1, 1.0}}, {}, -kInf, 1.5}});
        p.qp.linear.is_integer = {true, true};
        search::MiqpBbOptions o;
        search::MiqpBbDiagnostics d;
        const auto r = search::solve_miqcqp_bb(p, o, d);
        CHECK(r.proposed_status == core::Status::Optimal);
        CHECK_NEAR(r.objective, -1.0, 1e-9);
    }
    // A row nonconvex on a CONTINUOUS column is outside this file's convex
    // machinery (a proof needs the spatial branch-and-bound), but the primal
    // heuristics find the actual optimum here by inspection: min x + y s.t.
    // x^2 + y^2 >= 1, x integer in [-3, 3], y continuous in [-3, 3] is
    // minimised at the unconstrained box corner (-3, -3) (value -6), which
    // trivially satisfies the quadratic row (18 >= 1) -- the root NLP
    // relaxation alone should land there, no search needed.
    {
        auto p = make({-1, -1}, {-3, -3}, {3, 3}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, 1.0, kInf}});
        p.qp.linear.is_integer = {true, false};
        search::MiqpBbOptions o;
        search::MiqpBbDiagnostics d;
        const auto r = search::solve_miqcqp_bb(p, o, d);
        CHECK(r.proposed_status == core::Status::Feasible);
        CHECK(r.x.size() == 2);
        CHECK_NEAR(r.objective, -6.0, 1e-6);
        CHECK(r.termination_reason.find("no convex relaxation") != std::string::npos);
    }
    // The bound is valid for ANY (x, y): on the disk problem, at random
    // points and multipliers it never exceeds the true continuous optimum
    // -sqrt2 (min x + y over x^2 + y^2 <= 1, box [-2, 2]^2).
    {
        auto p = make({1, 1}, {-2, -2}, {2, 2}, {{{}, {{0, 0, 2.0}, {1, 1, 2.0}}, -kInf, 1.0}});
        engines::QpOptions qo;
        const auto cert = engines::certify_qcqp_convex(p, qo);
        CHECK(cert.ok);
        std::mt19937 rng(11);
        std::uniform_real_distribution<f64> U(-2.0, 2.0), Y(-3.0, 3.0);
        f64 worst = -kInf;
        for (int t = 0; t < 2000; ++t) {
            const f64 b = search::miqcqp_lagrangian_bound(p, cert, {U(rng), U(rng)}, {Y(rng)});
            worst = std::max(worst, b);
        }
        CHECK(worst <= -std::sqrt(2.0) + 1e-12);
        // ... and is tight at the KKT point: x = -(1,1)/sqrt2, y = 1/sqrt2.
        const f64 s = 1.0 / std::sqrt(2.0);
        const f64 tight = search::miqcqp_lagrangian_bound(p, cert, {-s, -s}, {s});
        CHECK_NEAR(tight, -std::sqrt(2.0), 1e-12);
    }
}

}  // namespace

int main() {
    test_hand_checked();
    test_fixed_columns();
    test_refusals();
    test_infeasible_not_claimed();
    test_check_extends_qp_check();
    test_miqcqp();
    return sor::test::finish("test_qcqp_ipm");
}
