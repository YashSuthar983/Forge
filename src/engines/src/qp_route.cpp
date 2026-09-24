// SOR — convex QP engine routing (Track 1, node A5 / QP-4).
//
// The rule comes from the measured sweeps, not from taste.  On the 19 convex
// continuous QPLIB instances, 300 s, one core (Ryzen 7700X):
//
//   benchmarks/results/qplib-convex-qpipm-cpu-20260922-030440.md  IPM  13 proved
//   benchmarks/results/qplib-convex-qp-cpu-20260922-034002.md     PDHCG 8 proved
//   (the IPM reached 14 after the net-of-rounding claim, b867479)
//
// Every instance the first-order path proves, the interior point also proves,
// and on the eight they share the IPM is faster on six (8515 37 ms vs 300 s;
// 8616 85 ms vs 3.4 s; 8991 60 ms vs 300 s; 8785 7.3 s vs 300 s; 8792 152 ms
// vs 300 s; 8602 220 s vs 322 s), the first-order path on two, by little
// (8495 0.7 s vs 2.0 s; 8790 68 ms vs 212 ms).  So: interior point first.
//
// What the first-order path is FOR, then, is the cases the interior point
// cannot take:
//   * a Q that is not certified PSD -- the IPM refuses it outright;
//   * a factorization that does not fit (QPLIB_9008, ~1M x 1M, died with
//     std::bad_alloc at 30 GB);
//   * an interior point that ran out of time or stalled without certifying,
//     where the remaining budget is better spent on a different method than
//     on more Newton steps.
// The last two are only visible by trying, so this routes by OUTCOME rather
// than by a guessed cost model: run the IPM under the memory the caller
// allows, and hand what is left of the time budget to the first-order path
// if -- and only if -- the IPM did not come back with a proof.  The result
// is never worse than the better of the two, at the cost of at most one
// extra solve.  A guessed shape threshold would have to be re-measured on
// every machine; this does not.
#include "qp_common.hpp"

#include "sor/engines/qp.hpp"

#include <chrono>
#include <new>
#include <string>

namespace sor::engines {

namespace {
using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
}  // namespace

core::RawResult solve_qp_auto(const QpProblem& problem, const QpOptions& opts,
                              QpDiagnostics& diag) {
    const auto t0 = Clock::now();
    QpDiagnostics ipm_diag;
    core::RawResult ipm;
    bool ipm_ran = false;
    try {
        ipm = solve_qp_ipm(problem, opts, ipm_diag);
        ipm_ran = true;
    } catch (const std::bad_alloc&) {
        // The KKT factorization did not fit.  Not a failure of the claim
        // machinery, so it is not an error -- it is what the first-order
        // path exists for.
        ipm_diag = QpDiagnostics{};
        ipm_diag.termination_reason = "interior point: out of memory";
    }
    if (ipm_ran && ipm.proposed_status == core::Status::Optimal) {
        diag = ipm_diag;
        diag.total_ms = ms_since(t0);
        diag.termination_reason = "routed to interior point; " + diag.termination_reason;
        ipm.termination_reason = diag.termination_reason;
        return ipm;
    }

    // No proof from the IPM: spend what is left of the budget on the
    // first-order path.  Its own claim goes through the same original-units
    // check, so this cannot manufacture a proof the checker would refuse.
    QpOptions fo = opts;
    if (opts.time_limit_s > 0.0) {
        const f64 left = opts.time_limit_s - ms_since(t0) / 1000.0;
        if (left <= 0.0) {
            diag = ipm_diag;
            diag.total_ms = ms_since(t0);
            diag.termination_reason =
                "routed to interior point (no time left for the first-order path); " +
                diag.termination_reason;
            return ipm;
        }
        fo.time_limit_s = left;
    }
    QpDiagnostics fo_diag;
    auto first_order = solve_qp(problem, fo, fo_diag);
    if (first_order.proposed_status == core::Status::Optimal) {
        diag = fo_diag;
        diag.total_ms = ms_since(t0);
        diag.termination_reason =
            "routed to first-order after the interior point did not certify; " +
            diag.termination_reason;
        first_order.termination_reason = diag.termination_reason;
        return first_order;
    }

    // Neither proved.  Return the interior point's answer when it has one --
    // it is the more accurate of the two on this set -- and say both ran.
    if (ipm_ran) {
        diag = ipm_diag;
        diag.total_ms = ms_since(t0);
        diag.termination_reason = "routed: neither engine certified; interior point: " +
                                  ipm_diag.termination_reason + "; first-order: " +
                                  fo_diag.termination_reason;
        ipm.termination_reason = diag.termination_reason;
        return ipm;
    }
    diag = fo_diag;
    diag.total_ms = ms_since(t0);
    diag.termination_reason =
        "routed to first-order (the interior point ran out of memory); " +
        fo_diag.termination_reason;
    first_order.termination_reason = diag.termination_reason;
    return first_order;
}

}  // namespace sor::engines
