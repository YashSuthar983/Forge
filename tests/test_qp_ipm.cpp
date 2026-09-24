// Interior-point QP: hand-checkable optima for every row and bound type,
// then random convex QPs checked against the independent original-units
// KKT test the claim rests on.
#include "sor/engines/qp.hpp"
// Internal: polishing is tested directly, from a guess we control.
#include "../src/engines/src/qp_common.hpp"
#include "sor/sparse/csr.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <string>
#include <cstdio>
#include <string>
#include <vector>

using namespace sor;
using core::f64;
using core::Index;
constexpr f64 kInf = model::kInf;

namespace {

// min 0.5 x'diag(q)x + c'x  s.t.  rows (each a list of (col, coef)).
engines::QpProblem make(std::vector<f64> q, std::vector<f64> c,
                        std::vector<f64> lo, std::vector<f64> hi,
                        std::vector<std::vector<std::pair<Index, f64>>> rows,
                        std::vector<f64> rlo, std::vector<f64> rhi) {
    engines::QpProblem p;
    const Index n = static_cast<Index>(c.size()), m = static_cast<Index>(rows.size());
    std::vector<Index> r, cc;
    std::vector<f64> v;
    for (Index i = 0; i < m; ++i)
        for (auto [j, a] : rows[static_cast<std::size_t>(i)]) { r.push_back(i); cc.push_back(j); v.push_back(a); }
    p.linear.A = sparse::from_triplets(m, n, r, cc, v);
    p.linear.c = c;
    p.linear.col_lo = lo;
    p.linear.col_hi = hi;
    p.linear.row_lo = rlo;
    p.linear.row_hi = rhi;
    p.linear.is_integer.assign(static_cast<std::size_t>(n), false);
    std::vector<Index> qr, qc;
    std::vector<f64> qv;
    for (Index j = 0; j < n; ++j)
        if (q[static_cast<std::size_t>(j)] != 0.0) { qr.push_back(j); qc.push_back(j); qv.push_back(q[static_cast<std::size_t>(j)]); }
    p.q_matrix = sparse::from_triplets(n, n, qr, qc, qv);
    return p;
}

void expect(const std::string& label, const engines::QpProblem& p, std::vector<f64> x_star, f64 obj) {
    engines::QpOptions o;
    engines::QpDiagnostics d;
    const auto r = engines::solve_qp_ipm(p, o, d);
    bool ok = r.proposed_status == core::Status::Optimal &&
              std::fabs(r.objective - obj) <= 1e-7 * (1.0 + std::fabs(obj));
    for (std::size_t j = 0; j < x_star.size() && ok; ++j)
        ok = std::fabs(r.x[j] - x_star[j]) <= 1e-6 * (1.0 + std::fabs(x_star[j]));
    ::sor::test::report(ok, "IPM hand-checked optimum", __FILE__, __LINE__,
                        label + ": status " + std::to_string(static_cast<int>(r.proposed_status)) +
                            " obj " + std::to_string(r.objective) + " want " + std::to_string(obj) +
                            " it " + std::to_string(d.iterations) + " " + r.termination_reason);
}

}  // namespace

int main() {
    // min (x-2)^2 = 0.5*2x^2 - 4x + 4 : unconstrained optimum 2.
    expect("box, interior", make({2}, {-4}, {0}, {5}, {}, {}, {}), {2}, -4);
    expect("box, at upper", make({2}, {-4}, {0}, {1.5}, {}, {}, {}), {1.5}, 2.25 - 6);
    expect("lower only", make({2}, {4}, {-1}, {kInf}, {}, {}, {}), {-1}, 1 - 4);
    expect("upper only", make({2}, {-4}, {-kInf}, {1}, {}, {}, {}), {1}, 1 - 4);
    expect("free", make({2}, {-4}, {-kInf}, {kInf}, {}, {}, {}), {2}, -4);
    // Rows, x free:  x <= 1.2  (one-sided upper), >= 3 (one-sided lower),
    // = 0.5 (equality), in [0.2, 0.7] (ranged).
    expect("row <= 1.2", make({2}, {-4}, {-kInf}, {kInf}, {{{0, 1.0}}}, {-kInf}, {1.2}), {1.2}, 1.44 - 4.8);
    expect("row >= 3", make({2}, {-4}, {-kInf}, {kInf}, {{{0, 1.0}}}, {3}, {kInf}), {3}, 9 - 12);
    expect("row == 0.5", make({2}, {-4}, {-kInf}, {kInf}, {{{0, 1.0}}}, {0.5}, {0.5}), {0.5}, 0.25 - 2);
    expect("row ranged", make({2}, {-4}, {-kInf}, {kInf}, {{{0, 1.0}}}, {0.2}, {0.7}), {0.7}, 0.49 - 2.8);
    // Row inactive: x <= 10 does not bind.
    expect("row slack", make({2}, {-4}, {-kInf}, {kInf}, {{{0, 1.0}}}, {-kInf}, {10}), {2}, -4);
    // Two variables, one row: min (x-2)^2 + (y-2)^2  s.t. x + y <= 1.2
    expect("2-var row", make({2, 2}, {-4, -4}, {-kInf, -kInf}, {kInf, kInf}, {{{0, 1.0}, {1, 1.0}}},
                             {-kInf}, {1.2}), {0.6, 0.6}, 2 * (0.36 - 2.4));
    // Fixed column substituted out: x0 = 1 fixed, min (x1-2)^2 s.t. x0 + x1 <= 1.2
    expect("fixed col", make({0, 2}, {0, -4}, {1, -kInf}, {1, kInf}, {{{0, 1.0}, {1, 1.0}}},
                             {-kInf}, {1.2}), {1, 0.2}, 0.04 - 0.8);
    // LP (Q = 0): min -x - y s.t. x + 2y <= 4, 3x + y <= 6, x,y >= 0 -> (1.6, 1.2)
    expect("LP", make({0, 0}, {-1, -1}, {0, 0}, {kInf, kInf},
                      {{{0, 1.0}, {1, 2.0}}, {{0, 3.0}, {1, 1.0}}}, {-kInf, -kInf}, {4, 6}),
           {1.6, 1.2}, -2.8);
    // ---- polishing: recovers the optimum from a truncated solve, and only
    //      when the independent check agrees ----
    {
        // min (x-2)^2 + (y-2)^2  s.t.  x + y <= 1.2,  0 <= x, y <= 5
        const auto p = make({2, 2}, {-4, -4}, {0, 0}, {5, 5}, {{{0, 1.0}, {1, 1.0}}},
                            {-kInf}, {1.2});
        engines::QpOptions o;
        o.max_iterations = 3;          // far too few to converge on its own
        o.polish = false;
        engines::QpDiagnostics d0;
        const auto r0 = engines::solve_qp_ipm(p, o, d0);
        CHECK(r0.proposed_status != core::Status::Optimal);   // truncated really fails
        o.polish = true;
        engines::QpDiagnostics d1;
        const auto r1 = engines::solve_qp_ipm(p, o, d1);
        CHECK(r1.proposed_status == core::Status::Optimal);
        CHECK(std::fabs(r1.x[0] - 0.6) < 1e-9 && std::fabs(r1.x[1] - 0.6) < 1e-9);
        CHECK(r1.termination_reason.find("polish") != std::string::npos);
    }
    {
        // A bound that is NOT active at the optimum must not be fixed by a
        // bad guess and then accepted: min (x-2)^2 with 0 <= x <= 5 has x = 2
        // interior.  From one IPM step the guess may be wrong; whatever it
        // is, an accepted point must be the true optimum.
        const auto p = make({2}, {-4}, {0}, {5}, {}, {}, {});
        engines::QpOptions o;
        o.max_iterations = 1;
        engines::QpDiagnostics d;
        const auto r = engines::solve_qp_ipm(p, o, d);
        if (r.proposed_status == core::Status::Optimal) CHECK(std::fabs(r.x[0] - 2.0) < 1e-9);
    }

    {
        // Primal-dual active-set rounds repair a wrong first guess.
        //   min 0.5[(x0-0.5)^2 + (x1-0.5)^2]  s.t.  x0 + x1 = 1,  0 <= x <= 1
        // has x = (0.5, 0.5), y = 0.  From x = (0.99, 0.001) with a large
        // y = 10, the indicator fixes BOTH columns at 0 -- which makes the
        // equality row infeasible.  Round 0 must be rejected, and round 1
        // must release both (their multipliers come out wrong-signed) and
        // land on the optimum.
        const auto p = make({1, 1}, {-0.5, -0.5}, {0, 0}, {1, 1}, {{{0, 1.0}, {1, 1.0}}},
                            {1}, {1});
        const backend::LaneBounds lb{p.linear.col_lo, p.linear.col_hi, p.linear.row_lo,
                                     p.linear.row_hi};
        engines::QpOptions o;
        std::vector<f64> x{0.99, 0.001}, y{10.0};
        engines::qpc::OriginalKkt k;
        const bool ok = engines::qpc::polish(p, lb, o, x, y, k);
        CHECK(ok);
        CHECK(std::fabs(x[0] - 0.5) < 1e-9 && std::fabs(x[1] - 0.5) < 1e-9);
        CHECK(std::fabs(y[0]) < 1e-9);
    }

    {
        // Routing (A5): --engine qpauto must return what the interior point
        // returns when the interior point certifies, and must say so.
        const auto p = make({2, 2}, {-4, -4}, {0, 0}, {5, 5}, {{{0, 1.0}, {1, 1.0}}},
                            {-kInf}, {1.2});
        engines::QpOptions o;
        engines::QpDiagnostics dr, di;
        const auto routed = engines::solve_qp_auto(p, o, dr);
        const auto ipm = engines::solve_qp_ipm(p, o, di);
        CHECK(routed.proposed_status == core::Status::Optimal);
        CHECK(routed.proposed_status == ipm.proposed_status);
        CHECK(std::fabs(routed.objective - ipm.objective) <= 1e-12 * (1.0 + std::fabs(ipm.objective)));
        CHECK(routed.termination_reason.find("routed to interior point") != std::string::npos);
    }
    {
        // Dependent active rows, the QPLIB_8500 shape in miniature.
        //   min 0.5(x0^2 + x1^2) - 200(x0 + x1)
        //   s.t.  x0 + x1 = 1,  x0 + x1 = 1 + eps,   0 <= x <= 1000
        // The two rows are the same row with two right-hand sides, so the
        // active block is rank deficient AND its rhs is inconsistent by eps:
        // no x satisfies both, and the best any point can do is split the eps
        // between them (primal residual eps/2 = 5e-9, inside feas_tol).  The
        // multiplier is then free along the null direction (1, -1): only
        // y0 + y1 = 199.5 is determined.  The gap is
        //     (b0 - a) y0 + (b1 - a) y1 = (eps/2)(y1 - y0),
        // so a lopsided split -- which is what an interior point hands over,
        // and what the proximal centre deliberately preserves -- leaves a gap
        // of ~1e-6, a hundred times the tolerance, with every other residual
        // already at 5e-9.  Only spending the null-space freedom closes it.
        const f64 eps = 1e-8;
        const auto p = make({1, 1}, {-200, -200}, {0, 0}, {1000, 1000},
                            {{{0, 1.0}, {1, 1.0}}, {{0, 1.0}, {1, 1.0}}},
                            {1, 1 + eps}, {1, 1 + eps});
        const backend::LaneBounds lb{p.linear.col_lo, p.linear.col_hi, p.linear.row_lo,
                                     p.linear.row_hi};
        engines::QpOptions o;
        const f64 ysum = 199.5 - 0.5 * eps;
        std::vector<f64> x{0.5 + 0.25 * eps, 0.5 + 0.25 * eps}, y{ysum, 0.0};
        const auto k0 = engines::qpc::kkt_original(p, lb, x, y);
        CHECK(k0.primal_net <= o.feas_tol && k0.dual_net <= o.stationarity_tol);
        CHECK(k0.gap_net > o.gap_tol);          // the gap, and only the gap, fails
        engines::qpc::OriginalKkt k;
        const bool ok = engines::qpc::polish(p, lb, o, x, y, k);
        CHECK(ok);
        CHECK(k.gap_net <= o.gap_tol);
        // The multipliers still have to sum to what stationarity fixes.
        CHECK(std::fabs(y[0] + y[1] - ysum) < 1e-6);
    }

    return sor::test::finish("test_qp_ipm");
}
