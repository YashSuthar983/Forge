// sor_io/solution.{hpp,cpp}: round-trip the plain-text solution format
// sor_solve writes and sor_check reads. Nothing fancy -- the point of this
// file being dead simple is that sor_check must never doubt the format
// itself, only the claim inside it.
#include "sor/io/solution.hpp"

#include "test_helpers.hpp"

#include <sstream>

using sor::core::ProofLevel;
using sor::core::SolveResult;
using sor::core::Status;

namespace {

void test_round_trip_optimal() {
    SolveResult r;
    r.status = Status::Optimal;
    r.proof = ProofLevel::ProvedOptimalFP;
    r.objective = -464.75314285714285;
    r.x = {1.0, 2.5, -3.0, 0.0};
    r.y = {0.1, -0.2};

    std::ostringstream out;
    sor::io::write_solution(out, r);

    std::istringstream in(out.str());
    const auto s = sor::io::read_solution(in);
    CHECK(s.status == Status::Optimal);
    CHECK(s.proof == ProofLevel::ProvedOptimalFP);
    CHECK_NEAR(s.objective, r.objective, 1e-15);
    CHECK(s.x.size() == r.x.size());
    for (std::size_t i = 0; i < r.x.size(); ++i) CHECK_NEAR(s.x[i], r.x[i], 1e-15);
    CHECK(s.y.size() == r.y.size());
    for (std::size_t i = 0; i < r.y.size(); ++i) CHECK_NEAR(s.y[i], r.y[i], 1e-15);
    CHECK(s.ray.empty());
}

void test_round_trip_infeasible_with_ray() {
    SolveResult r;
    r.status = Status::Infeasible;
    r.proof = ProofLevel::BoundOnly;
    r.objective = 4.0;
    r.x = {2.0, 2.0};
    r.y = {0.0};
    r.ray = {1.0};

    std::ostringstream out;
    sor::io::write_solution(out, r);

    std::istringstream in(out.str());
    const auto s = sor::io::read_solution(in);
    CHECK(s.status == Status::Infeasible);
    CHECK(s.ray.size() == 1);
    CHECK_NEAR(s.ray[0], 1.0, 1e-15);
}

void test_round_trip_empty_vectors() {
    SolveResult r;
    r.status = Status::NoSolutionFound;
    r.proof = ProofLevel::None;
    r.objective = sor::core::kNaN;

    std::ostringstream out;
    sor::io::write_solution(out, r);
    std::istringstream in(out.str());
    const auto s = sor::io::read_solution(in);
    CHECK(s.status == Status::NoSolutionFound);
    CHECK(s.x.empty());
    CHECK(s.y.empty());
    CHECK(s.ray.empty());
}

void test_malformed_file_throws() {
    std::istringstream in("status Optimal\nproof ProvedOptimalFP\nobjective 1.0\n"
                          "x 3 1.0 2.0\n");  // declares 3, gives 2
    CHECK_THROWS(sor::io::read_solution(in));

    std::istringstream in2("status NotARealStatus\nproof None\nobjective 0\n"
                           "x 0\ny 0\nray 0\n");
    CHECK_THROWS(sor::io::read_solution(in2));
}

}  // namespace

int main() {
    test_round_trip_optimal();
    test_round_trip_infeasible_with_ray();
    test_round_trip_empty_vectors();
    test_malformed_file_throws();
    return sor::test::finish("test_solution_io");
}
