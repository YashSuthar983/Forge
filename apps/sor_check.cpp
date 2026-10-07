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
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include "sor/core/route_debug.hpp"

namespace {

void usage() {
    std::fputs(
        "usage: sor_check MODEL.{mps,lp,qps} SOLUTION.sol [--tol T] [--gap-tol G] [--strict-mps]\n"
        "                 [--relax-integrality] [--small-matrix-value V]\n"
        "                 [--fixed-mps|--free-mps] [--strict]\n"
        "  SOLUTION.sol is written by `sor_solve ... --solution-out FILE`.\n"
        "  An LP optimum is checked against what it claims: ProvedKKT (the default\n"
        "  simplex result) by its primal and dual residuals; ProvedOptimalFP and\n"
        "  above (sor_solve --exact-proof) also by a safe dual bound within the gap\n"
        "  tolerance. --strict demands that bound for every LP optimum.\n"
        "  Exit 0: the claim independently verifies. Exit 1: rejected.\n"
        "  Exit 3: unverified claim (insufficient certificate), or a status\n"
        "  that makes no claim (Interrupted, NoSolutionFound, ...).\n"
        "  The model is always read strictly: an unknown section, or a quadratic\n"
        "  objective in a non-.qps file, is an error (--strict-mps is the default).\n",
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

// The QPS reader stores both triangles of Q; the objective is
// c'x + 1/2 x'Qx. Symmetry and one of two sufficient PSD checks are required
// before a KKT point can prove global optimality. Diagonal dominance handles
// large sparse matrices; dense Cholesky handles other small positive-definite
// matrices. A failed sufficient check leaves the claim unverified.
bool sparse_qp_convexity_proved(const sor::io::QpsProblem& qp) {
    const auto& Q = qp.q_matrix;
    const auto& lp = qp.linear;
    const auto n = lp.n_cols();
    if (Q.n_rows() != n || Q.n_cols() != n) return false;
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    std::vector<double> diagonal(n, 0.0), off_diagonal(n, 0.0);
    for (sor::core::Index i = 0; i < n; ++i) {
        for (sor::core::Offset k = rp[i]; k < rp[i + 1]; ++k) {
            const auto j = ci[k];
            const double v = Q.vals[k];
            if (j < 0 || j >= n || !std::isfinite(v)) return false;
            const auto mirror = std::lower_bound(ci.begin() + rp[j],
                                                 ci.begin() + rp[j + 1], i);
            if (mirror == ci.begin() + rp[j + 1] || *mirror != i ||
                Q.vals[mirror - ci.begin()] != v)
                return false;
            if (i == j) diagonal[i] += v;
            else off_diagonal[i] += std::fabs(v);
        }
        if (!std::isfinite(diagonal[i]) || !std::isfinite(off_diagonal[i]))
            return false;
    }
    bool diagonally_dominant = true;
    for (sor::core::Index i = 0; i < n; ++i) {
        const double margin = 64.0 * std::numeric_limits<double>::epsilon() *
            static_cast<double>(rp[i + 1] - rp[i] + 1) *
            (1.0 + std::fabs(diagonal[i]) + off_diagonal[i]);
        diagonally_dominant &=
            (diagonal[i] == 0.0 && off_diagonal[i] == 0.0) ||
            diagonal[i] > off_diagonal[i] + margin;
    }
    if (diagonally_dominant) return true;  // symmetric Gershgorin PSD bound

    constexpr sor::core::Index kDenseCholeskyLimit = 512;
    if (n > kDenseCholeskyLimit) return false;
    const auto size = static_cast<std::size_t>(n);
    std::vector<double> lower(size * size, 0.0);
    for (sor::core::Index i = 0; i < n; ++i)
        for (sor::core::Offset k = rp[i]; k < rp[i + 1]; ++k)
            lower[static_cast<std::size_t>(i) * size + ci[k]] = Q.vals[k];
    double scale = 1.0;
    for (double value : diagonal)
        scale = std::max(scale, std::fabs(value));
    const double pivot_margin = 1e-10 * scale;
    for (sor::core::Index i = 0; i < n; ++i) {
        for (sor::core::Index j = 0; j <= i; ++j) {
            double pivot = lower[static_cast<std::size_t>(i) * size + j];
            for (sor::core::Index k = 0; k < j; ++k)
                pivot -= lower[static_cast<std::size_t>(i) * size + k] *
                         lower[static_cast<std::size_t>(j) * size + k];
            if (!std::isfinite(pivot)) return false;
            if (i == j) {
                if (!(pivot > pivot_margin)) return false;
                lower[static_cast<std::size_t>(i) * size + i] = std::sqrt(pivot);
            } else {
                lower[static_cast<std::size_t>(i) * size + j] =
                    pivot / lower[static_cast<std::size_t>(j) * size + j];
            }
        }
    }
    return true;
}

double qp_kkt_residual(const sor::io::QpsProblem& qp,
                       const sor::io::SolutionFile& sol) {
    const auto& lp = qp.linear;
    if (static_cast<sor::core::Index>(sol.y.size()) != lp.n_rows())
        return sor::core::kPosInf;
    for (double x : sol.x)
        if (!std::isfinite(x)) return sor::core::kPosInf;
    for (double y : sol.y)
        if (!std::isfinite(y)) return sor::core::kPosInf;

    std::vector<double> qx(lp.n_cols(), 0.0);
    if (qp.q_matrix.n_rows() != 0) {
        const auto& Q = qp.q_matrix;
        const auto& qr = Q.pattern.row_ptr();
        const auto& qc = Q.pattern.col_idx();
        for (sor::core::Index i = 0; i < lp.n_cols(); ++i)
            for (sor::core::Offset k = qr[i]; k < qr[i + 1]; ++k)
                qx[i] += Q.vals[k] * sol.x[qc[k]];
    } else {
        for (sor::core::Index i = 0; i < lp.n_cols(); ++i)
            qx[i] = qp.q_diag[i] * sol.x[i];
    }

    std::vector<double> ax(lp.n_rows(), 0.0);
    std::vector<double> aty(lp.n_cols(), 0.0);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (sor::core::Index i = 0; i < lp.n_rows(); ++i) {
        for (sor::core::Offset k = rp[i]; k < rp[i + 1]; ++k) {
            ax[i] += lp.A.vals[k] * sol.x[ci[k]];
            aty[ci[k]] += lp.A.vals[k] * sol.y[i];
        }
    }
    double residual = 0.0;
    for (sor::core::Index i = 0; i < lp.n_rows(); ++i) {
        if (!std::isfinite(ax[i])) return sor::core::kPosInf;
        const double projected = std::clamp(ax[i] + sol.y[i],
                                            lp.row_lo[i], lp.row_hi[i]);
        if (!std::isfinite(projected)) return sor::core::kPosInf;
        residual = std::max(residual, std::fabs(ax[i] - projected));
    }
    for (sor::core::Index j = 0; j < lp.n_cols(); ++j) {
        if (!std::isfinite(qx[j]) || !std::isfinite(aty[j]) ||
            !std::isfinite(lp.c[j])) return sor::core::kPosInf;
        // Exported sol.y has the solver's sign convention: adding A' sol.y
        // here is g - A' y_math in the usual KKT notation. Projection onto
        // column bounds accounts for the bound multiplier z.
        const double gradient = qx[j] + lp.c[j] + aty[j];
        if (!std::isfinite(gradient)) return sor::core::kPosInf;
        const double projected = std::clamp(sol.x[j] - gradient,
                                            lp.col_lo[j], lp.col_hi[j]);
        if (!std::isfinite(projected)) return sor::core::kPosInf;
        residual = std::max(residual, std::fabs(sol.x[j] - projected));
    }
    return std::isfinite(residual) ? residual : sor::core::kPosInf;
}

double sparse_qp_kkt_residual(const sor::io::QpsProblem& qp,
                              const sor::io::SolutionFile& sol) {
    return qp.q_matrix.n_rows() == 0 ? sor::core::kPosInf
                                     : qp_kkt_residual(qp, sol);
}

double diagonal_qp_kkt_residual(const sor::io::QpsProblem& qp,
                                const sor::io::SolutionFile& sol) {
    return qp.q_matrix.n_rows() != 0 ? sor::core::kPosInf
                                     : qp_kkt_residual(qp, sol);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 2; }

    std::string model_path, solution_path;
    double tol = 1e-7;
    std::optional<double> gap_tol;
    bool strict = false;
    // Read the model completely, as sor_solve does: a section the reader
    // would skip (QUADOBJ in a .mps, SOS, ...) means the claim cannot be
    // checked against the file's model. --strict-mps is kept as a no-op.
    sor::io::MpsReadOptions mps_opts;
    mps_opts.strict = true;
    bool mps_format_forced = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tol" || a == "--gap-tol") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", a.c_str());
                return 2;
            }
            // strtod with a null end pointer accepts "abc" as 0 and takes
            // "nan"/"inf" at face value, so a mistyped tolerance silently
            // changed what the checker accepts instead of failing.
            const std::string text = argv[++i];
            std::size_t used = 0;
            double parsed = 0;
            try { parsed = std::stod(text, &used); } catch (const std::exception&) { used = 0; }
            if (used != text.size() || !std::isfinite(parsed) || !(parsed > 0.0)) {
                std::fprintf(stderr,
                             "error: %s expects a finite number greater than 0, "
                             "got '%s'\n", a.c_str(), text.c_str());
                return 2;
            }
            if (a == "--gap-tol") gap_tol = parsed; else tol = parsed;
        } else if (a == "--strict") {
            strict = true;
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
        if (sol.integrality_relaxed && lp.n_integer() > 0) {
            // The claim answers the LP relaxation, and says so.
            std::printf("integrality relaxed: the claim answers the LP relaxation\n");
            lp.is_integer.assign(lp.is_integer.size(), false);
        }

        std::printf("model:     %s  (%d rows x %d cols, nnz %lld)\n",
                    model_path.c_str(), lp.n_rows(), lp.n_cols(),
                    static_cast<long long>(lp.nnz()));
        std::printf("claimed:   status=%s proof=%s objective=%.10e\n",
                    std::string(sor::core::to_string(sol.status)).c_str(),
                    std::string(sor::core::to_string(sol.proof)).c_str(),
                    sol.objective);

        bool ok = true;
        bool unverified = false;
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
                    // A ProvedKKT claim asserts tolerance-level optimality:
                    // original-model primal and dual residuals (multiplier
                    // signs, reduced costs, complementarity) within
                    // tolerance. It does not assert a safe dual bound, which
                    // from floating multipliers is often -inf (a round-off
                    // reduced cost on a column with no finite bound on that
                    // side), so the gap is then reported, not required.
                    const bool kkt_claim =
                        sol.proof == sor::core::ProofLevel::ProvedKKT && !strict;
                    validation_scope = kkt_claim ? "lp_kkt_f64" : "lp_optimality_f64";
                    sor::core::RawResult raw;
                    raw.proposed_status = sol.status;
                    raw.proposed_level = sol.proof;
                    raw.objective = sol.objective;
                    raw.x = sol.x;
                    raw.y = sol.y;
                    raw.exact_dual = sol.exact_dual;
                    const auto ev = sor::certify::check_lp_point(
                        lp, raw, tol, tol, gap_tol.value_or(tol), true);
                    ok &= (ev.max_dual_violation <= tol)
                              ? pass("dual/reduced costs",
                                     ev.max_dual_violation, tol)
                              : fail("dual/reduced costs",
                                     ev.max_dual_violation, tol);
                    if (kkt_claim)
                        std::printf("info  %-28s residual=%.3e  (not claimed; --strict "
                                    "requires <= %.3e)\n", "primal-dual gap", ev.gap_rel,
                                    gap_tol.value_or(tol));
                    else
                        ok &= (ev.gap_rel <= gap_tol.value_or(tol))
                                  ? pass("primal-dual gap", ev.gap_rel, gap_tol.value_or(tol))
                                  : fail("primal-dual gap", ev.gap_rel, gap_tol.value_or(tol));
                } else if (sol.status == sor::core::Status::Optimal &&
                           !has_integer && qp) {
                    bool convexity_proved = !lp.maximize;
                    if (qp->q_matrix.n_rows() != 0) {
                        convexity_proved &= sparse_qp_convexity_proved(*qp);
                    } else {
                        convexity_proved &=
                            static_cast<sor::core::Index>(qp->q_diag.size()) == lp.n_cols();
                        for (double q : qp->q_diag)
                            convexity_proved &= std::isfinite(q) && q >= 0.0;
                    }
                    if (!convexity_proved) {
                        validation_scope = "qp_optimality_unverified";
                        if (ok) {
                            std::printf("UNVERIFIED  QP maximization or convexity "
                                        "not independently proved\n");
                            unverified = true;
                        }
                    } else {
                        // Projected row and column residuals encode multiplier
                        // signs, stationarity and complementarity.
                        validation_scope = "qp_kkt_f64";
                        const double kkt = qp->q_matrix.n_rows() != 0
                            ? sparse_qp_kkt_residual(*qp, sol)
                            : diagonal_qp_kkt_residual(*qp, sol);
                        ok &= (kkt <= tol) ? pass("quadratic KKT", kkt, tol)
                                           : fail("quadratic KKT", kkt, tol);
                    }
                }
                break;
            }
            case sor::core::Status::Infeasible: {
                validation_scope = is_qps
                    ? "qp_infeasible_via_lp_farkas_f64"
                    : has_integer
                    ? "milp_infeasible_via_lp_farkas_f64"
                    : "lp_farkas_f64";
                // A crossed bound in the model's own data proves
                // infeasibility, of the LP and of any integer restriction.
                if (const auto empty = lp.find_empty_domain(); empty.index >= 0) {
                    validation_scope = "empty_domain";
                    std::printf("info  %s\n", lp.describe(empty).c_str());
                    pass("empty domain", 0.0, tol);
                    break;
                }
                const auto& ray = sol.dual_farkas_ray.empty()
                                      ? sol.ray : sol.dual_farkas_ray;
                if (ray.empty()) {
                    if (has_integer) {
                        validation_scope = is_qps
                            ? "miqp_infeasibility_unverified"
                            : "milp_infeasibility_unverified";
                        std::printf("UNVERIFIED  status=Infeasible but no LP "
                                    "Farkas certificate was recorded; this "
                                    "does not refute integer infeasibility\n");
                        unverified = true;
                        break;
                    }
                    std::printf("FAIL  status=Infeasible but no Farkas "
                                "certificate was recorded\n");
                    ok = false;
                    break;
                }
                const auto cert = sol.exact_dual_farkas.empty()
                    ? sor::certify::check_dual_farkas_ray(lp, ray, tol)
                    : sor::certify::check_exact_dual_farkas_ray(lp, sol.exact_dual_farkas, tol);
                if (!sol.exact_dual_farkas.empty()) validation_scope = "lp_farkas_exact";
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
                const auto cert = sol.exact_primal_ray.empty()
                    ? sor::certify::check_primal_ray(lp, sol.primal_ray, tol)
                    : sor::certify::check_exact_primal_ray(lp, sol.exact_primal_ray, tol);
                if (!sol.exact_primal_ray.empty()) validation_scope = "lp_primal_ray_exact";
                ok &= cert.certified
                          ? pass("primal ray", std::max(cert.max_row_residual,
                                                       cert.max_bound_sign_residual), tol)
                          : fail("primal ray", std::max(cert.max_row_residual,
                                                       cert.max_bound_sign_residual), tol);
                break;
            }
            // These statuses assert nothing about the model: a limit stop, a
            // search that found no point, a detected numerical failure, a
            // refusal. There is no claim to verify, and none to reject; a
            // point written with them (an interrupted iterate) is not
            // claimed feasible either.
            case sor::core::Status::NotSolved:
            case sor::core::Status::NoSolutionFound:
            case sor::core::Status::Interrupted:
            case sor::core::Status::NumericalFailure:
            case sor::core::Status::Unsupported:
                validation_scope = "no_claim";
                std::printf("UNVERIFIED  status=%s makes no claim to check\n",
                            std::string(sor::core::to_string(sol.status)).c_str());
                unverified = true;
                break;
            default:
                std::printf(
                    "FAIL  no independent check implemented for status=%s\n",
                    std::string(sor::core::to_string(sol.status)).c_str());
                ok = false;
                break;
        }

        std::printf("validation: %s\n", validation_scope);
        std::printf("%s\n", unverified ? "UNVERIFIED"
                                        : ok ? "VERIFIED" : "REJECTED");
        return unverified ? 3 : ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
