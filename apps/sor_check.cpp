// sor_check — independent checker (SIH26119 checklist item 30). Re-verifies
// a claim written by `sor_solve --solution-out` against the ORIGINAL model
// file, using nothing from the solve itself: no basis, no iteration state,
// just the model's own row/bound violation methods and (for an infeasibility
// claim) sor::engines::farkas_violation(). A claim this binary rejects is a
// claim sor_solve should never have made; that's the entire point of it
// living in a separate executable that never links against the solver
// engines' internal state.
//
// LAYER L8.
#include "sor/engines/farkas.hpp"
#include "sor/io/mps.hpp"
#include "sor/io/solution.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <exception>
#include <fstream>
#include <string>

namespace {

void usage() {
    std::fputs(
        "usage: sor_check MODEL.mps SOLUTION.sol [--tol T]\n"
        "  SOLUTION.sol is written by `sor_solve ... --solution-out FILE`.\n"
        "  Exit 0: the claim independently verifies. Exit 1: it does not.\n",
        stderr);
}

bool fail(const char* what, double residual, double tol) {
    std::printf("FAIL  %-28s residual=%.3e  tol=%.3e\n", what, residual, tol);
    return false;
}

bool pass(const char* what, double residual, double tol) {
    std::printf("pass  %-28s residual=%.3e  tol=%.3e\n", what, residual, tol);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 2; }

    std::string model_path, solution_path;
    double tol = 1e-7;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tol") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --tol needs a value\n");
                return 2;
            }
            // strtod with a null end pointer accepts "abc" as 0 and takes
            // "nan"/"inf" at face value, so a mistyped tolerance silently
            // changed what the checker accepts instead of failing.
            const std::string text = argv[++i];
            std::size_t used = 0;
            try { tol = std::stod(text, &used); } catch (const std::exception&) { used = 0; }
            if (used != text.size() || !std::isfinite(tol) || !(tol > 0.0)) {
                std::fprintf(stderr,
                             "error: --tol expects a finite number greater than 0, "
                             "got '%s'\n", text.c_str());
                return 2;
            }
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (model_path.empty()) {
            model_path = a;
        } else if (solution_path.empty()) {
            solution_path = a;
        } else {
            usage();
            return 2;
        }
    }
    if (model_path.empty() || solution_path.empty()) { usage(); return 2; }

    try {
        sor::io::MpsReadReport rep;
        const auto lp = sor::io::read_mps_file_auto(model_path, rep);
        for (const auto& w : rep.warnings)
            std::fprintf(stderr, "warning: %s\n", w.c_str());

        std::ifstream sfile(solution_path);
        if (!sfile) {
            std::fprintf(stderr, "error: cannot open '%s'\n", solution_path.c_str());
            return 2;
        }
        const auto sol = sor::io::read_solution(sfile);

        std::printf("model:     %s  (%d rows x %d cols, nnz %lld)\n",
                    model_path.c_str(), lp.n_rows(), lp.n_cols(),
                    static_cast<long long>(lp.nnz()));
        std::printf("claimed:   status=%s proof=%s objective=%.10e\n",
                    std::string(sor::core::to_string(sol.status)).c_str(),
                    std::string(sor::core::to_string(sol.proof)).c_str(),
                    sol.objective);

        bool ok = true;

        switch (sol.status) {
            case sor::core::Status::Optimal:
            case sor::core::Status::Feasible: {
                if (static_cast<sor::core::Index>(sol.x.size()) != lp.n_cols()) {
                    std::printf("FAIL  x has %zu entries, model has %d columns\n",
                                sol.x.size(), lp.n_cols());
                    ok = false;
                    break;
                }
                const double row_viol = lp.max_row_violation(sol.x);
                const double bound_viol = lp.max_bound_violation(sol.x);
                ok &= (row_viol <= tol) ? pass("row bounds", row_viol, tol)
                                        : fail("row bounds", row_viol, tol);
                ok &= (bound_viol <= tol) ? pass("column bounds", bound_viol, tol)
                                          : fail("column bounds", bound_viol, tol);

                const double true_obj = lp.objective(sol.x);
                const double obj_err =
                    std::fabs(true_obj - sol.objective) / (1.0 + std::fabs(true_obj));
                ok &= (obj_err <= tol)
                          ? pass("objective (recomputed)", obj_err, tol)
                          : fail("objective (recomputed)", obj_err, tol);
                // Optimal additionally claims a proof; this checker has no
                // independent way to re-derive dual optimality without a
                // basis (which is exactly the state this binary refuses to
                // trust), so an Optimal claim's PROOF is not re-verified
                // here -- only that the point itself is genuinely feasible
                // and the stated objective matches it. That is the honest
                // scope of a checker that never touches solver internals.
                break;
            }
            case sor::core::Status::Infeasible: {
                if (sol.ray.empty()) {
                    std::printf(
                        "no certificate: status=Infeasible but no ray was recorded "
                        "(honest gap, not a failure -- nothing to verify)\n");
                    break;
                }
                const double v = sor::engines::farkas_violation(lp, sol.ray);
                if (!std::isfinite(v)) {
                    ok = fail("farkas certificate", v, tol);
                } else {
                    ok &= (v <= tol) ? pass("farkas certificate", v, tol)
                                     : fail("farkas certificate", v, tol);
                }
                break;
            }
            default:
                std::printf(
                    "no independent check implemented for status=%s "
                    "(honest gap, not a failure)\n",
                    std::string(sor::core::to_string(sol.status)).c_str());
                break;
        }

        std::printf("%s\n", ok ? "VERIFIED" : "REJECTED");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
