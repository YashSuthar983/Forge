// Option binding tables for the QP and QCQP engines.
//
// One table per options struct, used to set an option from "key=value", to list
// what exists with its default, and to generate the documented flag reference.
// Because all three read the same table, a new option becomes settable,
// listable and documented at once -- it cannot be added to the struct and
// forgotten in the help text.
#include "sor/engines/qcqp.hpp"
#include "sor/engines/qp.hpp"

namespace sor::engines {

std::vector<core::OptionBinding> qp_option_bindings(QpOptions& o) {
    std::vector<core::OptionBinding> t;
    t.push_back(core::bind_u64("max_iterations", o.max_iterations,
        "Iteration cap for the first-order path."));
    t.push_back(core::bind_real("time_limit_s", o.time_limit_s,
        "Wall-clock budget in seconds; 0 means no limit."));
    t.push_back(core::bind_u64("check_every", o.check_every,
        "Iterations between convergence checks. Each check costs a residual "
        "evaluation, so checking too often is its own slowdown."));
    t.push_back(core::bind_int("inner_max_iterations", o.inner_max_iterations,
        "Cap on the inner solve of one outer iteration."));
    t.push_back(core::bind_int("inner_epoch", o.inner_epoch,
        "Inner iterations batched per device launch. Larger batches cut launch "
        "overhead on a GPU and delay the convergence check."));
    t.push_back(core::bind_int("convexity_dense_limit", o.convexity_dense_limit,
        "Largest Q for which convexity is certified by a dense factorization."));
    t.push_back(core::bind_real("feas_tol", o.feas_tol,
        "Primal feasibility tolerance, original units."));
    t.push_back(core::bind_real("stationarity_tol", o.stationarity_tol,
        "Dual/stationarity tolerance, original units."));
    t.push_back(core::bind_real("gap_tol", o.gap_tol,
        "Relative duality-gap tolerance for a claim of optimality."));
    t.push_back(core::bind_real("step_safety", o.step_safety,
        "Fraction of the step to the boundary actually taken (<1)."));
    t.push_back(core::bind_real("inner_tolerance_min", o.inner_tolerance_min,
        "Floor on the inner solve tolerance."));
    t.push_back(core::bind_real("inner_tolerance_scale", o.inner_tolerance_scale,
        "Inner tolerance as a fraction of the current outer residual."));
    t.push_back(core::bind_flag("assume_psd", o.assume_psd,
        "Skip the convexity certificate and trust the caller. A nonconvex Q "
        "accepted here yields a KKT point, not an optimum, so the result can "
        "no longer be claimed optimal on this engine's own evidence."));
    t.push_back(core::bind_flag("use_reflected_halpern", o.use_reflected_halpern,
        "Reflected Halpern iteration instead of the default averaging."));
    t.push_back(core::bind_flag("adaptive_restart", o.adaptive_restart,
        "Restart the first-order sequence on measured stagnation."));
    t.push_back(core::bind_flag("primal_weight", o.primal_weight,
        "Rebalance primal and dual step sizes from observed residuals."));
    t.push_back(core::bind_flag("polish", o.polish,
        "Active-set polish of a first-order point. Accepted only if the "
        "original-units check then passes, so it cannot manufacture a claim."));
    t.push_back(core::bind_flag("scale", o.scale,
        "Ruiz equilibration before solving."));
    t.push_back(core::bind_int("ruiz_iterations", o.ruiz_iterations,
        "Ruiz scaling sweeps."));
    t.push_back(core::bind_int("tighten_retries", o.tighten_retries,
        "Retries with a tightened inner tolerance before giving up."));
    t.push_back(core::bind_real("halpern_theta", o.halpern_theta,
        "Halpern anchor weight; 0 selects the built-in schedule."));
    t.push_back(core::bind_flag("verbose", o.verbose, "Per-iteration logging."));
    return t;
}

bool set_qp_option(QpOptions& o, const std::string& kv, std::string& err) {
    auto t = qp_option_bindings(o);
    return core::apply_option(t, kv, err);
}

std::vector<core::OptionBinding> qcqp_local_option_bindings(QcqpLocalOptions& o) {
    std::vector<core::OptionBinding> t;
    t.push_back(core::bind_real("time_limit_s", o.time_limit_s,
        "Wall-clock budget shared by every start; 0 means no limit."));
    t.push_back(core::bind_int("max_iterations", o.max_iterations,
        "Barrier iteration cap per start."));
    t.push_back(core::bind_real("tol", o.tol,
        "Scaled KKT tolerance for local optimality."));
    t.push_back(core::bind_real("feas_tol", o.feas_tol,
        "Row and bound violation, original units, for a point to count feasible. "
        "This is also the bar the multi-start loop stops improving at, so a "
        "looser value here yields points a stricter consumer will reject."));
    t.push_back(core::bind_int("starts", o.starts,
        "Multi-start count; 1 uses the default start only. Time, not this, is "
        "normally the real limiter."));
    t.push_back(core::bind_u64("seed", o.seed,
        "Seed for the randomised starts, so a run is reproducible."));
    t.push_back(core::bind_flag("mccormick_start", o.mccormick_start,
        "Seed the first start from a McCormick relaxation instead of the "
        "origin; costs one extra LP solve per call."));
    t.push_back(core::bind_flag("verbose", o.verbose, "Per-iteration logging."));
    return t;
}

bool set_qcqp_local_option(QcqpLocalOptions& o, const std::string& kv, std::string& err) {
    auto t = qcqp_local_option_bindings(o);
    return core::apply_option(t, kv, err);
}

}  // namespace sor::engines
