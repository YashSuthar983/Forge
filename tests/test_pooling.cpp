// Refinery pooling problem -- the nonconvex core of blending, and the one
// model family on this project that has a PUBLISHED global optimum to be held
// against.
//
// Haverly (1978) built these three instances to show that a local method
// stops at the wrong point, so they are exactly the right regression: an
// engine that quietly degrades to a local search still returns a feasible
// blend, and only the published number catches it.  The assertions are
// therefore pinned to the published profits (400 / 600 / 750) and to the
// PROOF, not merely to feasibility -- solve_global_qcqp must close the tree,
// because "found it" and "proved it" are different claims and this project
// does not let one stand in for the other.
#include "sor/io/qplib.hpp"
#include "sor/search/global_qp.hpp"
#include "sor/search/qplib_qp.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace sor;

namespace {

std::string data_path(const char* name) {
    return std::string(SOR_TEST_DATA_DIR) + "/refinery/" + name;
}

// The emitted instance is a MINIMISATION of cost - revenue, so the refinery
// profit a reader cares about is the negated objective.
void check_haverly(const char* file, double published_profit) {
    io::QplibReadReport rep;
    const auto q = io::read_qplib_file(data_path(file), rep);
    CHECK(q.n > 0);
    CHECK(q.has_quadratic_constraints());   // the pooling bilinearity is present

    engines::QcqpProblem qcqp;
    search::qplib_to_qcqp(q, qcqp);

    search::GlobalQpOptions opts;
    opts.time_limit_s = 60.0;
    const auto g = search::solve_global_qcqp(qcqp, opts);

    CHECK(g.have_incumbent);
    CHECK(g.proved);            // the tree was closed, not merely explored
    CHECK(g.bound_valid);
    const double sgn = qcqp.objective_negated ? -1.0 : 1.0;
    const double profit = -(sgn * g.incumbent);
    CHECK(std::fabs(profit - published_profit) <= 1e-5 * (1.0 + published_profit));
}

// Mixed-integer pooling (QPLIB class LMQ: unit on/off decisions plus
// blending).  Before integer branching was added to solve_global_qcqp, NO
// engine in this project could prove one of these optimal: --engine miqp
// (miqcqp_bb) only certifies convex node relaxations and pooling's bilinear
// rows are always indefinite, while the spatial branch-and-bound had no
// integrality branching at all (see QP.md's refinery section
// section for the fuller account). Pinned on the smallest generated rung so
// the tree closes within a few hundred milliseconds; REF_MIP_medium proves
// too but is left out of ctest for runtime.
// The integer columns of the returned point must be EXACTLY integral, not
// merely within feas_tol -- the model does not admit a fractional point, and
// scripts/qplib_eval.py's own integrality check is exact equality.
void check_ref_mip_small() {
    io::QplibReadReport rep;
    const auto q = io::read_qplib_file(data_path("REF_MIP_small.qplib"), rep);
    CHECK(q.n > 0);
    CHECK(q.has_quadratic_constraints());

    engines::QcqpProblem qcqp;
    search::qplib_to_qcqp(q, qcqp);
    CHECK(qcqp.qp.linear.n_integer() > 0);

    search::GlobalQpOptions opts;
    opts.time_limit_s = 60.0;
    const auto g = search::solve_global_qcqp(qcqp, opts);

    CHECK(g.have_incumbent);
    CHECK(g.proved);
    CHECK(g.bound_valid);
    CHECK(g.gap_rel <= opts.gap_tol);
    CHECK(g.x.size() == qcqp.qp.linear.is_integer.size());

    const auto& is_int = qcqp.qp.linear.is_integer;
    for (std::size_t j = 0; j < is_int.size(); ++j)
        if (is_int[j]) CHECK(g.x[j] == std::round(g.x[j]));

    const auto ev = engines::evaluate_qcqp(qcqp, g.x);
    CHECK(ev.max_violation() <= opts.feas_tol);
    CHECK(ev.max_integrality_violation == 0.0);
}

}  // namespace

int main() {
    if (!sor::test::data_available(data_path("HAVERLY1.qplib")))
        return sor::test::skip("test_pooling", data_path("HAVERLY1.qplib"));
    check_haverly("HAVERLY1.qplib", 400.0);
    check_haverly("HAVERLY2.qplib", 600.0);
    check_haverly("HAVERLY3.qplib", 750.0);
    check_ref_mip_small();
    return sor::test::finish("test_pooling");
}
