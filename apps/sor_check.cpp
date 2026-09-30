// sor_check - independent checker (SIH26119 checklist item 30). Re-verifies
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
#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/io/qps.hpp"
#include "sor/io/solution.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <exception>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#include "sor/core/route_debug.hpp"

namespace {

void usage() {
    std::fputs(
        "usage: sor_check MODEL.mps SOLUTION.sol [--tol T] [--strict-mps]\n"
        "                 [--relax-integrality] [--small-matrix-value V]\n"
        "                 [--fixed-mps|--free-mps]\n"
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

double diagonal_qp_kkt_residual(const sor::io::QpsProblem& qp,
                               const sor::io::SolutionFile& sol) {
    const auto& lp = qp.linear;
    if (static_cast<sor::core::Index>(sol.y.size()) != lp.n_rows() ||
        qp.q_matrix.n_rows() != 0 ||
        lp.maximize ||
        static_cast<sor::core::Index>(qp.q_diag.size()) != lp.n_cols())
        return sor::core::kPosInf;
    for (double q : qp.q_diag)
        if (!std::isfinite(q) || q < 0.0) return sor::core::kPosInf;

    const double sense = lp.maximize ? -1.0 : 1.0;
    std::vector<double> ax(lp.n_rows(), 0.0);
    std::vector<double> aty(lp.n_cols(), 0.0);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (sor::core::Index i = 0; i < lp.n_rows(); ++i) {
        if (!std::isfinite(sol.y[i])) return sor::core::kPosInf;
        for (sor::core::Offset k = rp[i]; k < rp[i + 1]; ++k) {
            ax[i] += lp.A.vals[k] * sol.x[ci[k]];
            aty[ci[k]] += lp.A.vals[k] * sol.y[i];
        }
    }
    double residual = 0.0;
    for (sor::core::Index i = 0; i < lp.n_rows(); ++i) {
        const double y_min = sense * sol.y[i];
        const double projected = std::clamp(ax[i] + y_min,
                                            lp.row_lo[i], lp.row_hi[i]);
        residual = std::max(residual, std::fabs(ax[i] - projected));
    }
    for (sor::core::Index j = 0; j < lp.n_cols(); ++j) {
        if (!std::isfinite(sol.x[j])) return sor::core::kPosInf;
        const double gradient = qp.q_diag[j] * sol.x[j] +
                                sense * (lp.c[j] + aty[j]);
        const double projected = std::clamp(sol.x[j] - gradient,
                                            lp.col_lo[j], lp.col_hi[j]);
        residual = std::max(residual, std::fabs(sol.x[j] - projected));
    }
    return std::isfinite(residual) ? residual : sor::core::kPosInf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 2; }

    std::string model_path, solution_path;
    double tol = 1e-7;
    sor::io::MpsReadOptions mps_opts;
    bool mps_format_forced = false;
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
        } else if (a == "--relax-integrality") {
            mps_opts.relax_integrality = true;
        } else if (a == "--strict-mps") {
            mps_opts.strict = true;
        } else if (a == "--small-matrix-value") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --small-matrix-value needs a value\n");
                return 2;
            }
            const std::string text = argv[++i];
            std::size_t used = 0;
            try {
                mps_opts.small_matrix_value = std::stod(text, &used);
            } catch (const std::exception&) {
                used = 0;
            }
            if (used != text.size() || !std::isfinite(mps_opts.small_matrix_value) ||
                !(mps_opts.small_matrix_value > 0.0)) {
                std::fprintf(stderr,
                             "error: --small-matrix-value expects a finite number "
                             "greater than 0, got '%s'\n", text.c_str());
                return 2;
            }
        } else if (a == "--fixed-mps") {
            mps_opts.fixed_format = true;
            mps_format_forced = true;
        } else if (a == "--free-mps") {
            mps_opts.fixed_format = false;
            mps_format_forced = true;
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
        const bool is_qps = model_path.size() >= 4 &&
            (model_path.compare(model_path.size() - 4, 4, ".qps") == 0 ||
             model_path.compare(model_path.size() - 4, 4, ".QPS") == 0);
        std::optional<sor::io::QpsProblem> qp;
        sor::model::LpProblem lp;
        if (is_qps) {
            sor::io::QpsReadReport qrep;
            qp = sor::io::read_qps_file(model_path, qrep, mps_opts);
            rep = qrep;
            lp = qp->linear;
        } else {
            lp = mps_format_forced
                     ? sor::io::read_mps_file(model_path, rep, mps_opts)
                     : sor::io::read_mps_file_auto(model_path, rep, mps_opts);
        }
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
        const bool has_integer = std::any_of(
            lp.is_integer.begin(), lp.is_integer.end(),
            [](char value) { return value != 0; });
        const char* validation_scope = "unsupported";

        switch (sol.status) {
            case sor::core::Status::Optimal:
            case sor::core::Status::Feasible: {
                validation_scope = is_qps
                    ? (has_integer ? "miqp_incumbent" : "qp_primal_point")
                    : (has_integer ? "milp_incumbent" : "lp_primal_point");
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

                if (has_integer) {
                    double integrality_viol = 0.0;
                    for (sor::core::Index j = 0; j < lp.n_cols(); ++j) {
                        if (!lp.is_integer[static_cast<std::size_t>(j)]) continue;
                        const double value = sol.x[static_cast<std::size_t>(j)];
                        integrality_viol = std::max(
                            integrality_viol, std::fabs(value - std::round(value)));
                    }
                    ok &= (integrality_viol <= tol)
                              ? pass("integrality", integrality_viol, tol)
                              : fail("integrality", integrality_viol, tol);
                }

                double true_obj = lp.objective(sol.x);
                if (qp) {
                    double xqx = 0.0;
                    if (qp->q_matrix.n_rows() != 0) {
                        const auto& Q = qp->q_matrix;
                        const auto& rp = Q.pattern.row_ptr();
                        const auto& ci = Q.pattern.col_idx();
                        for (sor::core::Index i = 0; i < lp.n_cols(); ++i)
                            for (sor::core::Offset k = rp[i]; k < rp[i + 1]; ++k)
                                xqx += sol.x[i] * Q.vals[k] * sol.x[ci[k]];
                    } else {
                        for (sor::core::Index j = 0; j < lp.n_cols(); ++j)
                            xqx += qp->q_diag[j] * sol.x[j] * sol.x[j];
                    }
                    true_obj += (lp.maximize ? -0.5 : 0.5) * xqx;
                }
                const double obj_err =
                    std::fabs(true_obj - sol.objective) / (1.0 + std::fabs(true_obj));
                ok &= (obj_err <= tol)
                          ? pass("objective (recomputed)", obj_err, tol)
                          : fail("objective (recomputed)", obj_err, tol);
                if (sol.status == sor::core::Status::Optimal && !has_integer &&
                    !is_qps) {
                    validation_scope = "lp_optimality_f64";
                    sor::core::RawResult raw;
                    raw.proposed_status = sol.status;
                    raw.proposed_level = sol.proof;
                    raw.objective = sol.objective;
                    raw.x = sol.x;
                    raw.y = sol.y;
                    const auto ev = sor::certify::check_lp_point(
                        lp, raw, tol, tol, tol, true);
                    ok &= (ev.max_dual_violation <= tol)
                              ? pass("dual/reduced costs",
                                     ev.max_dual_violation, tol)
                              : fail("dual/reduced costs",
                                     ev.max_dual_violation, tol);
                    ok &= (ev.gap_rel <= tol)
                              ? pass("primal-dual gap", ev.gap_rel, tol)
                              : fail("primal-dual gap", ev.gap_rel, tol);
                } else if (sol.status == sor::core::Status::Optimal &&
                           !has_integer && qp &&
                           qp->q_matrix.n_rows() == 0) {
                    // For a convex diagonal QP, primal feasibility and KKT
                    // stationarity/complementarity suffice for global
                    // optimality. Recompute them from the exported point and
                    // multipliers; never trust engine diagnostics here.
                    validation_scope = "qp_kkt_f64";
                    const double kkt = diagonal_qp_kkt_residual(*qp, sol);
                    ok &= (kkt <= tol) ? pass("quadratic KKT", kkt, tol)
                                       : fail("quadratic KKT", kkt, tol);
                }
                break;
            }
            case sor::core::Status::Infeasible: {
                validation_scope = is_qps
                    ? "qp_infeasible_via_lp_farkas_f64"
                    : has_integer
                    ? "milp_infeasible_via_lp_farkas_f64"
                    : "lp_farkas_f64";
                const auto& ray = sol.dual_farkas_ray.empty()
                                      ? sol.ray : sol.dual_farkas_ray;
                if (ray.empty()) {
                    std::printf("FAIL  status=Infeasible but no Farkas "
                                "certificate was recorded\n");
                    ok = false;
                    break;
                }
                const auto cert = sor::certify::check_dual_farkas_ray(lp, ray, tol);
                ok &= cert.certified
                          ? pass("farkas certificate",
                                 cert.max_homogeneous_residual, tol)
                          : fail("farkas certificate",
                                 cert.max_homogeneous_residual, tol);
                break;
            }
            case sor::core::Status::Unbounded: {
                if (is_qps) {
                    validation_scope = "qp_unbounded_unverified";
                    std::printf("FAIL  QP unboundedness requires a recession "
                                "direction with zero quadratic curvature\n");
                    ok = false;
                    break;
                }
                validation_scope = "lp_unbounded_point_and_ray_f64";
                if (static_cast<sor::core::Index>(sol.x.size()) != lp.n_cols()) {
                    std::printf("FAIL  unbounded claim has no full primal "
                                "feasible point (%zu entries, expected %d)\n",
                                sol.x.size(), lp.n_cols());
                    ok = false;
                } else {
                    const double row_viol = lp.max_row_violation(sol.x);
                    const double bound_viol = lp.max_bound_violation(sol.x);
                    ok &= (row_viol <= tol)
                              ? pass("primal point row bounds", row_viol, tol)
                              : fail("primal point row bounds", row_viol, tol);
                    ok &= (bound_viol <= tol)
                              ? pass("primal point column bounds", bound_viol, tol)
                              : fail("primal point column bounds", bound_viol, tol);
                }
                if (sol.primal_ray.empty()) {
                    std::printf("FAIL  status=Unbounded but no primal ray was "
                                "recorded\n");
                    ok = false;
                    break;
                }
                const auto cert = sor::certify::check_primal_ray(
                    lp, sol.primal_ray, tol);
                ok &= cert.certified
                          ? pass("primal ray", std::max(cert.max_row_residual,
                                                       cert.max_bound_sign_residual), tol)
                          : fail("primal ray", std::max(cert.max_row_residual,
                                                       cert.max_bound_sign_residual), tol);
                break;
            }
            default:
                std::printf(
                    "FAIL  no independent check implemented for status=%s\n",
                    std::string(sor::core::to_string(sol.status)).c_str());
                ok = false;
                break;
        }

        std::printf("validation: %s\n", validation_scope);
        std::printf("%s\n", ok ? "VERIFIED" : "REJECTED");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
