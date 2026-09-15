#include "sor/backend/lp_device.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/sparse/csc.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

using namespace sor;

namespace {

backend::ScaledLp scalar_lp(double upper = 10.0) {
    backend::ScaledLp lp;
    lp.A_csr.pattern = sparse::SparsePattern(1, 1, {0, 0}, {});
    lp.A_csc = sparse::to_csc(lp.A_csr);
    lp.c = {-1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {upper};
    lp.row_lo = {-model::kInf};
    lp.row_hi = {model::kInf};
    lp.row_scale = {1.0};
    lp.col_scale = {1.0};
    return lp;
}

model::LpProblem controller_problem() {
    model::LpProblem p;
    p.A = sparse::from_triplets(1, 1, {0}, {0}, {1.0});
    p.c = {0.0};
    p.col_lo = {-10.0};
    p.col_hi = {10.0};
    p.row_lo = {-model::kInf};
    p.row_hi = {model::kInf};
    return p;
}

class ScriptedDevice final : public backend::LpDevice {
public:
    explicit ScriptedDevice(std::vector<Kkt> values,
                            std::vector<core::f64> primal_ray = {},
                            std::vector<core::f64> dual_ray = {},
                            backend::LpDeviceCapabilities capabilities =
                                {true, true, true, true, true})
        : values_(std::move(values)), primal_ray_(std::move(primal_ray)),
          dual_ray_(std::move(dual_ray)), capabilities_(capabilities) {}

    std::string_view name() const override { return "noncpu-scripted"; }
    bool is_accelerated() const override { return false; }
    backend::LpDeviceCapabilities capabilities() const override {
        return capabilities_;
    }
    void upload(const backend::ScaledLp& lp) override {
        rows_ = static_cast<std::size_t>(lp.n_rows());
        cols_ = static_cast<std::size_t>(lp.n_cols());
    }
    void hpr_steps(std::uint32_t k, const backend::StepParams& step) override {
        iterations_ += k;
        last_step_ = step;
    }
    Kkt reduce_kkt() override {
        if (values_.empty()) return {};
        const auto index = std::min(position_, values_.size() - 1);
        ++position_;
        return values_[index];
    }
    void snapshot_anchor() override { ++snapshots_; }
    bool snapshot_step_checkpoint() override {
        ++step_checkpoints_;
        return true;
    }
    bool restore_step_checkpoint() override {
        ++step_restores_;
        return true;
    }
    void restart_to(backend::RestartPoint point) override {
        CHECK(point == backend::RestartPoint::Current);
        ++restarts_;
    }
    void init_zero() override {}
    bool init_iterate(const std::vector<core::f64>&,
                      const std::vector<core::f64>&) override { return true; }
    void download(backend::LpSolution& solution) override {
        solution.x.assign(cols_, 0.0);
        solution.y.assign(rows_, 0.0);
        solution.x_avg = solution.x;
        solution.y_avg = solution.y;
        solution.primal_ray = primal_ray_;
        solution.dual_farkas_ray = dual_ray_;
    }
    backend::TransferStats transfer_stats() const override { return {}; }
    void reset_stats() override {}

    std::uint64_t restarts() const { return restarts_; }
    std::uint64_t step_checkpoints() const { return step_checkpoints_; }
    std::uint64_t step_restores() const { return step_restores_; }
    const backend::StepParams& last_step() const { return last_step_; }

private:
    std::vector<Kkt> values_;
    std::size_t position_ = 0;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::uint64_t iterations_ = 0;
    std::uint64_t restarts_ = 0;
    std::uint64_t snapshots_ = 0;
    std::uint64_t step_checkpoints_ = 0;
    std::uint64_t step_restores_ = 0;
    backend::StepParams last_step_{};
    std::vector<core::f64> primal_ray_;
    std::vector<core::f64> dual_ray_;
    backend::LpDeviceCapabilities capabilities_;
};

}  // namespace

int main() {
    auto reflected = backend::make_cpu_lp_device();
    CHECK(reflected->capabilities().reflected_operator);
    CHECK(reflected->capabilities().fixed_point_restart);
    CHECK(reflected->capabilities().warm_start);
    CHECK(reflected->capabilities().transactional_step);
    reflected->upload(scalar_lp(1.5));
    reflected->init_zero();
    backend::StepParams step;
    step.tau = 1.0;
    step.sigma = 1.0;
    step.primal_weight = 1.0;
    step.use_halpern = true;
    step.use_reflection = true;
    step.reflection_gamma = 1.0;
    step.update_average = false;
    reflected->hpr_steps(1, step);
    backend::LpSolution r1;
    reflected->download(r1);
    // beta_0=1/2 and (2T-I)(0)=2, hence x_1=1 exactly.
    CHECK_NEAR(r1.x[0], 1.0, 1e-14);
    reflected->hpr_steps(1, step);
    backend::LpSolution before_restart;
    reflected->download(before_restart);
    CHECK_NEAR(before_restart.x[0], 4.0 / 3.0, 1e-14);
    reflected->restart_to(backend::RestartPoint::Current);
    backend::LpSolution after_restart;
    reflected->download(after_restart);
    // Restart installs the latest T(z)=1.5, not the Halpern iterate 4/3.
    CHECK_NEAR(after_restart.x[0], 1.5, 1e-14);

    auto unreflected = backend::make_cpu_lp_device();
    unreflected->upload(scalar_lp());
    unreflected->init_zero();
    step.use_reflection = false;
    unreflected->hpr_steps(1, step);
    backend::LpSolution ordinary;
    unreflected->download(ordinary);
    CHECK_NEAR(ordinary.x[0], 0.5, 1e-14);
    const auto kkt = unreflected->reduce_kkt();
    CHECK_NEAR(kkt.restart_metric, 1.0, 1e-14);
    CHECK(kkt.operator_lhs <= kkt.operator_rhs);

    // Snapshotting begins a fresh Halpern epoch, so the next beta is 1/2
    // again.  The fixed-point residual uses the requested weighted norm.
    step.use_reflection = true;
    auto epoch_reset = backend::make_cpu_lp_device();
    epoch_reset->upload(scalar_lp());
    epoch_reset->init_zero();
    epoch_reset->hpr_steps(2, step);
    epoch_reset->snapshot_anchor();
    epoch_reset->hpr_steps(1, step);
    backend::LpSolution fresh_epoch;
    epoch_reset->download(fresh_epoch);
    CHECK_NEAR(fresh_epoch.x[0], 3.0, 1e-14);
    step.use_halpern = false;
    step.primal_weight = 4.0;
    auto weighted = backend::make_cpu_lp_device();
    weighted->upload(scalar_lp());
    weighted->init_zero();
    weighted->hpr_steps(1, step);
    CHECK_NEAR(weighted->reduce_kkt().restart_metric, 2.0, 1e-14);

    backend::ScaledLp residual_lp;
    residual_lp.A_csr = sparse::from_triplets(1, 1, {0}, {0}, {6.0});
    residual_lp.A_csc = sparse::to_csc(residual_lp.A_csr);
    residual_lp.c = {3.0};
    residual_lp.col_lo = {-model::kInf};
    residual_lp.col_hi = {model::kInf};
    residual_lp.row_lo = {2.0};
    residual_lp.row_hi = {2.0};
    residual_lp.row_scale = {2.0};
    residual_lp.col_scale = {3.0};
    auto residual_device = backend::make_cpu_lp_device();
    residual_device->upload(residual_lp);
    CHECK(residual_device->init_iterate({0.0}, {0.0}));
    const auto original_kkt = residual_device->reduce_kkt();
    CHECK_NEAR(original_kkt.primal_res, 1.0, 1e-14);
    CHECK_NEAR(original_kkt.dual_res, 1.0, 1e-14);

    // A non-CPU device must refuse an enabled feature it cannot execute.  It
    // is never permitted to run a weakened algorithm under the HPR name.
    const auto expect_unsupported = [&](backend::LpDeviceCapabilities caps,
                                        const char* capability) {
        ScriptedDevice limited({}, {}, {}, caps);
        engines::HprOptions options;
        options.max_iterations = 1;
        engines::HprDiagnostics diagnostics;
        const auto raw = engines::solve_hpr(
            controller_problem(), options, limited, diagnostics);
        CHECK(raw.proposed_status == core::Status::Unsupported);
        CHECK(raw.backend == "noncpu-scripted");
        CHECK(raw.termination_reason.find(capability) != std::string::npos);
    };
    expect_unsupported({false, true, true, true, true}, "reflected");
    expect_unsupported({true, false, true, true, true}, "restart");
    expect_unsupported({true, true, false, true, true}, "warm-started");
    expect_unsupported({true, true, true, false, true}, "certificate");
    expect_unsupported({true, true, true, true, false}, "transactional");

    // One Ruiz-free Pock--Chambolle alpha=1 pass has the published l1
    // inverse-square-root factors.
    model::LpProblem p;
    p.A = sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 3.0});
    p.c = {1.0, 1.0};
    p.col_lo = {0.0, 0.0};
    p.col_hi = {model::kInf, model::kInf};
    p.row_lo = {-model::kInf};
    p.row_hi = {4.0};
    auto scaled = engines::build_scaled_lp(p, 0, true, 1.0);
    CHECK_NEAR(scaled.row_scale[0], 0.5, 1e-14);
    CHECK_NEAR(scaled.col_scale[0], 1.0, 1e-14);
    CHECK_NEAR(scaled.col_scale[1], 1.0 / std::sqrt(3.0), 1e-14);
    CHECK_NEAR(scaled.A_csr.vals[0], 0.5, 1e-14);
    CHECK_NEAR(scaled.A_csr.vals[1], 3.0 / (2.0 * std::sqrt(3.0)), 1e-14);

    // Controller-level tests use scalar KKT scripts so they pin the restart
    // thresholds, log-space weight update, and operator backtracking without
    // depending on convergence behavior of a particular LP.
    backend::LpDevice::Kkt first;
    first.primal_res = first.dual_res = 1.0;
    first.restart_metric = 1.0;
    first.epoch_dx_norm = 1.0;
    first.epoch_dy_norm = 4.0;
    backend::LpDevice::Kkt decayed = first;
    decayed.restart_metric = 0.1;
    ScriptedDevice sufficient({first, decayed});
    engines::HprOptions hopts;
    hopts.max_iterations = 2;
    hopts.check_every = 1;
    hopts.use_halpern = false;
    hopts.halpern_warmup = 0;
    hopts.min_iters_between_restarts = 1;
    hopts.use_polishing = false;
    hopts.detect_certificates = false;
    hopts.use_adaptive_step = false;
    engines::HprDiagnostics hdiag;
    (void)engines::solve_hpr(controller_problem(), hopts, sufficient, hdiag);
    CHECK(hdiag.sufficient_restarts == 1);
    CHECK(hdiag.necessary_restarts == 0);
    CHECK_NEAR(hdiag.final_primal_weight, 2.0, 1e-14);
    CHECK_NEAR(sufficient.last_step().tau, 0.998, 1e-14);
    CHECK_NEAR(sufficient.last_step().sigma, 0.998, 1e-14);

    // Necessary/stall restart: the metric first improves, then rebounds while
    // remaining within 0.8 of the epoch start.  Artificial restart is moved
    // out of the way so this pins the intended rule rather than a coincidence.
    backend::LpDevice::Kkt improved = first;
    improved.restart_metric = 0.6;
    backend::LpDevice::Kkt rebounded = first;
    rebounded.restart_metric = 0.7;
    ScriptedDevice necessary({first, improved, rebounded});
    hopts.max_iterations = 3;
    hopts.artificial_restart_fraction = 10.0;
    engines::HprDiagnostics necessary_diag;
    (void)engines::solve_hpr(
        controller_problem(), hopts, necessary, necessary_diag);
    CHECK(necessary_diag.sufficient_restarts == 0);
    CHECK(necessary_diag.necessary_restarts == 1);

    backend::LpDevice::Kkt flat = first;
    flat.restart_metric = 1.0;
    ScriptedDevice artificial({flat, flat});
    hopts.use_primal_weight = false;
    hopts.artificial_restart_fraction = 0.36;
    engines::HprDiagnostics artificial_diag;
    (void)engines::solve_hpr(
        controller_problem(), hopts, artificial, artificial_diag);
    CHECK(artificial_diag.artificial_restarts == 1);

    backend::LpDevice::Kkt unstable = first;
    unstable.operator_lhs = 10.0;
    unstable.operator_rhs = 1.0;
    ScriptedDevice backtracking({unstable, unstable, unstable});
    hopts.max_iterations = 3;
    hopts.use_restart = false;
    hopts.use_adaptive_step = true;
    engines::HprDiagnostics backtrack_diag;
    (void)engines::solve_hpr(
        controller_problem(), hopts, backtracking, backtrack_diag);
    CHECK(backtrack_diag.step_backtracks == 3);
    CHECK(backtracking.restarts() == 0);
    CHECK(backtracking.step_checkpoints() == 3);
    CHECK(backtracking.step_restores() == 3);
    CHECK(backtrack_diag.final_step < 0.998);

    model::LpProblem unbounded;
    unbounded.A = sparse::from_triplets(1, 1, {0}, {0}, {0.0});
    unbounded.c = {-1.0};
    unbounded.col_lo = {0.0};
    unbounded.col_hi = {model::kInf};
    unbounded.row_lo = {-model::kInf};
    unbounded.row_hi = {model::kInf};
    backend::LpDevice::Kkt ray_check = first;
    ray_check.primal_ray_residual = 0.0;
    ray_check.primal_ray_objective = -1.0;
    hopts.max_iterations = 2;
    hopts.use_adaptive_step = false;
    hopts.detect_certificates = true;
    hopts.certificate_checks_required = 3;
    ScriptedDevice two_checks({ray_check, ray_check}, {1.0});
    engines::HprDiagnostics two_check_diag;
    const auto premature = engines::solve_hpr(
        unbounded, hopts, two_checks, two_check_diag);
    CHECK(two_check_diag.primal_ray_check_streak == 2);
    CHECK(premature.proposed_status != core::Status::Unbounded);

    hopts.max_iterations = 3;
    ScriptedDevice three_checks(
        {ray_check, ray_check, ray_check}, {1.0});
    engines::HprDiagnostics three_check_diag;
    const auto certified_candidate = engines::solve_hpr(
        unbounded, hopts, three_checks, three_check_diag);
    CHECK(three_check_diag.primal_ray_check_streak == 3);
    CHECK(certified_candidate.proposed_status == core::Status::Unbounded);
    CHECK(three_check_diag.primal_ray.certified);

    // A candidate streak cannot span an adaptive step change.
    {
        auto unstable_ray = ray_check;
        unstable_ray.operator_lhs = 10.0;
        unstable_ray.operator_rhs = 1.0;
        ScriptedDevice reset_by_step(
            {unstable_ray, ray_check, ray_check, ray_check}, {1.0});
        engines::HprOptions options;
        options.max_iterations = 4;
        options.check_every = 1;
        options.use_restart = false;
        options.use_halpern = false;
        options.use_polishing = false;
        options.use_adaptive_step = true;
        options.certificate_checks_required = 3;
        engines::HprDiagnostics diagnostics;
        const auto raw = engines::solve_hpr(
            unbounded, options, reset_by_step, diagnostics);
        CHECK(diagnostics.step_backtracks == 1);
        CHECK(raw.proposed_status == core::Status::Unbounded);
        CHECK(raw.iterations == 4);
    }

    // Arming Halpern installs a new fixed-point anchor and therefore starts a
    // new certificate epoch; the two pre-anchor samples do not count.
    {
        ScriptedDevice reset_by_anchor(
            {ray_check, ray_check, ray_check, ray_check, ray_check}, {1.0});
        engines::HprOptions options;
        options.max_iterations = 5;
        options.check_every = 1;
        options.use_restart = false;
        options.use_halpern = true;
        options.halpern_warmup = 2;
        options.use_polishing = false;
        options.use_adaptive_step = false;
        options.certificate_checks_required = 3;
        engines::HprDiagnostics diagnostics;
        const auto raw = engines::solve_hpr(
            unbounded, options, reset_by_anchor, diagnostics);
        CHECK(raw.proposed_status == core::Status::Unbounded);
        CHECK(raw.iterations == 5);
    }

    // A restart changes the fixed-step/fixed-weight epoch too.  Two valid
    // samples before the sufficient restart may not be combined with the
    // three samples after it.
    {
        auto sharp_decay = ray_check;
        sharp_decay.restart_metric = 0.1;
        auto post_restart_1 = ray_check;
        post_restart_1.restart_metric = 1.0;
        auto post_restart_2 = ray_check;
        post_restart_2.restart_metric = 0.9;
        auto post_restart_3 = ray_check;
        post_restart_3.restart_metric = 0.8;
        ScriptedDevice reset_by_restart(
            {ray_check, sharp_decay, post_restart_1, post_restart_2,
             post_restart_3},
            {1.0});
        engines::HprOptions options;
        options.max_iterations = 5;
        options.check_every = 1;
        options.use_restart = true;
        options.min_iters_between_restarts = 1;
        options.artificial_restart_fraction = 10.0;
        options.use_halpern = false;
        options.use_polishing = false;
        options.use_adaptive_step = false;
        options.certificate_checks_required = 3;
        engines::HprDiagnostics diagnostics;
        const auto raw = engines::solve_hpr(
            unbounded, options, reset_by_restart, diagnostics);
        CHECK(diagnostics.sufficient_restarts == 1);
        CHECK(raw.proposed_status == core::Status::Unbounded);
        CHECK(raw.iterations == 5);
    }

    return test::finish("test_r2hpdg");
}
