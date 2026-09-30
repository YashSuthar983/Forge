#include "sor/io/solution.hpp"

#include <cmath>
#include <cstdlib>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace sor::io {
namespace {

using core::ProofLevel;
using core::Status;

// operator<<(double) happily writes "nan"/"inf", but operator>>(double&) does
// NOT reliably parse those tokens back on every libstdc++ configuration
// (locale-dependent; verified failing on this build). objective legitimately
// carries NaN whenever a run found nothing (Interrupted, NumericalFailure,
// NotSolved all default to it), so this format needs its own non-finite
// round-trip instead of leaning on the stream operators for it.
void write_f64(std::ostream& out, f64 v) {
    if (std::isnan(v)) out << "nan";
    else if (v == core::kPosInf) out << "inf";
    else if (v == -core::kPosInf) out << "-inf";
    else out << v;
}

f64 read_f64_token(const std::string& tok) {
    if (tok == "nan" || tok == "-nan") return core::kNaN;
    if (tok == "inf" || tok == "+inf") return core::kPosInf;
    if (tok == "-inf") return -core::kPosInf;
    // std::stod throws out_of_range for ERANGE, including a representable
    // subnormal. Such values are emitted by the solver and must reach the
    // checker unchanged; reject malformed and overflowing tokens instead.
    char* end = nullptr;
    const f64 value = std::strtod(tok.c_str(), &end);
    if (end == tok.c_str() || *end != '\0' || !std::isfinite(value))
        throw std::runtime_error("solution file: not a number '" + tok + "'");
    return value;
}

Status status_from_string(const std::string& s) {
    if (s == "NotSolved") return Status::NotSolved;
    if (s == "Optimal") return Status::Optimal;
    if (s == "Infeasible") return Status::Infeasible;
    if (s == "Unbounded") return Status::Unbounded;
    if (s == "InfeasibleOrUnbounded") return Status::InfeasibleOrUnbounded;
    if (s == "Feasible") return Status::Feasible;
    if (s == "NoSolutionFound") return Status::NoSolutionFound;
    if (s == "Interrupted") return Status::Interrupted;
    if (s == "NumericalFailure") return Status::NumericalFailure;
    if (s == "Unsupported") return Status::Unsupported;
    throw std::runtime_error("solution file: unknown status '" + s + "'");
}

ProofLevel proof_from_string(const std::string& s) {
    if (s == "None") return ProofLevel::None;
    if (s == "BoundOnly") return ProofLevel::BoundOnly;
    if (s == "FeasibleOnly") return ProofLevel::FeasibleOnly;
    if (s == "FeasibleWithGap") return ProofLevel::FeasibleWithGap;
    if (s == "ProvedKKT") return ProofLevel::ProvedKKT;
    if (s == "ProvedGlobalEpsilon") return ProofLevel::ProvedGlobalEpsilon;
    if (s == "ProvedOptimalFP") return ProofLevel::ProvedOptimalFP;
    if (s == "ProvedOptimalExact") return ProofLevel::ProvedOptimalExact;
    if (s == "ProvedOptimalCertified") return ProofLevel::ProvedOptimalCertified;
    throw std::runtime_error("solution file: unknown proof level '" + s + "'");
}

void write_vec(std::ostream& out, const char* tag, const std::vector<f64>& v) {
    out << tag << ' ' << v.size();
    out.precision(17);
    for (const f64 x : v) { out << ' '; write_f64(out, x); }
    out << '\n';
}

std::vector<f64> read_vec(std::istream& in, const char* expect_tag) {
    std::string tag;
    std::size_t n = 0;
    if (!(in >> tag >> n) || tag != expect_tag)
        throw std::runtime_error(std::string("solution file: expected '") +
                                 expect_tag + "' line");
    std::vector<f64> v(n);
    std::string tok;
    for (std::size_t i = 0; i < n; ++i) {
        if (!(in >> tok))
            throw std::runtime_error(std::string("solution file: '") + expect_tag +
                                     "' declared " + std::to_string(n) +
                                     " values but fewer were present");
        v[i] = read_f64_token(tok);
    }
    return v;
}

}  // namespace

void write_solution(std::ostream& out, const core::SolveResult& r) {
    out << "status " << core::to_string(r.status) << '\n';
    out << "proof " << core::to_string(r.proof) << '\n';
    out.precision(17);
    out << "objective "; write_f64(out, r.objective); out << '\n';
    write_vec(out, "x", r.x);
    write_vec(out, "y", r.y);
    write_vec(out, "ray", r.ray);
    write_vec(out, "primal_ray", r.primal_ray.direction);
    write_vec(out, "dual_farkas_ray", r.dual_farkas_ray.multipliers);
}

SolutionFile read_solution(std::istream& in) {
    SolutionFile s;
    std::string tag, value;
    if (!(in >> tag >> value) || tag != "status")
        throw std::runtime_error("solution file: expected 'status' line");
    s.status = status_from_string(value);

    if (!(in >> tag >> value) || tag != "proof")
        throw std::runtime_error("solution file: expected 'proof' line");
    s.proof = proof_from_string(value);

    if (!(in >> tag >> value) || tag != "objective")
        throw std::runtime_error("solution file: expected 'objective' line");
    s.objective = read_f64_token(value);

    s.x = read_vec(in, "x");
    s.y = read_vec(in, "y");
    s.ray = read_vec(in, "ray");
    // The final two fields were added in claim protocol v2.  Accepting an old
    // file remains useful for local diagnostics, but such a file cannot carry
    // an unboundedness certificate and claim_run never emits the old form.
    in >> std::ws;
    if (in.peek() != std::char_traits<char>::eof()) {
        s.primal_ray = read_vec(in, "primal_ray");
        s.dual_farkas_ray = read_vec(in, "dual_farkas_ray");
    }
    return s;
}

}  // namespace sor::io
