#include "forge/certify/finalize.hpp"
#include "forge/engines/nlp.hpp"
#include "forge/search/minlp.hpp"
#include "forge/search/miqp.hpp"
#include "forge/sparse/csr.hpp"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {
void print(const char* kind, const forge::core::SolveResult& r) {
    std::printf("%-6s status=%s proof=%s objective=%.9g x=[", kind,
                std::string(forge::core::to_string(r.status)).c_str(),
                std::string(forge::core::to_string(r.proof)).c_str(), r.objective);
    for (std::size_t i = 0; i < r.x.size(); ++i)
        std::printf("%s%.7g", i ? "," : "", r.x[i]);
    std::printf("]\n");
}

forge::engines::NlpProblem nonlinear(bool integer_x) {
    forge::engines::NlpProblem p;
    p.linear.A = forge::sparse::from_triplets(0, 2, {}, {}, {});
    p.linear.c = {0.0, 0.0};
    p.linear.col_lo = {-2.0, -2.0};
    p.linear.col_hi = {3.0, 2.0};
    p.linear.is_integer = {integer_x, false};
    p.initial_x = {0.2, -0.5};
    p.convex = true;
    p.objective = [](const std::vector<double>& x) {
        const double a = x[0] - 1.0, b = x[1] - 0.25;
        return a * a + b * b + 0.1 * b * b * b * b;
    };
    p.gradient = [](const std::vector<double>& x, std::vector<double>& g) {
        const double a = x[0] - 1.0, b = x[1] - 0.25;
        g = {2.0 * a, 2.0 * b + 0.4 * b * b * b};
    };
    return p;
}
}

int main() {
    bool ok = true;
    forge::engines::QpProblem qp;
    qp.linear.A = forge::sparse::from_triplets(0, 1, {}, {}, {});
    qp.linear.c = {-1.2};
    qp.linear.col_lo = {0.0};
    qp.linear.col_hi = {3.0};
    qp.linear.is_integer = {true};
    qp.q_diag = {1.0};
    forge::search::MiqpOptions mio;
    forge::search::MiqpDiagnostics mid;
    auto mir = forge::search::solve_miqp(qp, mio, mid);
    const auto miev = forge::search::miqp_evidence(qp, mio, mid, mir);
    const auto mif = forge::certify::finalize_result(std::move(mir), miev);
    print("MIQP", mif);
    ok = ok && mif.status == forge::core::Status::Optimal;

    auto np = nonlinear(false);
    forge::engines::NlpOptions no;
    forge::engines::NlpDiagnostics nd;
    auto nr = forge::engines::solve_nlp(np, no, nd);
    const auto nev = forge::engines::nlp_evidence(np, no, nd, nr);
    const auto nf = forge::certify::finalize_result(std::move(nr), nev);
    print("NLP", nf);
    ok = ok && nf.status == forge::core::Status::Optimal;

    auto mp = nonlinear(true);
    forge::search::MinlpOptions mino;
    forge::search::MinlpDiagnostics mind;
    auto minr = forge::search::solve_minlp(mp, mino, mind);
    const auto minev = forge::search::minlp_evidence(mp, mino, mind, minr);
    const auto minf = forge::certify::finalize_result(std::move(minr), minev);
    print("MINLP", minf);
    ok = ok && minf.status == forge::core::Status::Optimal;
    return ok ? 0 : 1;
}
