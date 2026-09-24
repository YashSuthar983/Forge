// SOR — BatchedPdhcgDevice over ordinary PdhcgDevices, one per lane.
//
// LAYER L1.  No arithmetic of its own: each call is forwarded to the lanes in
// the mask, so K = 1 over a device is exactly that device, and K CPU devices
// are the batched oracle.
#include "sor/backend/batched_pdhcg_device.hpp"

#include <stdexcept>
#include <string>

namespace sor::backend {
namespace {

class LanesDevice final : public BatchedPdhcgDevice {
public:
    LanesDevice(std::vector<std::unique_ptr<PdhcgDevice>> owned, std::vector<PdhcgDevice*> view)
        : owned_(std::move(owned)), lanes_(std::move(view)) {
        if (lanes_.empty()) throw std::invalid_argument("lanes device needs >= 1 lane");
        name_ = std::string(lanes_.front()->name()) + " lanes";
    }

    std::string_view name() const override {
        return lanes_.size() == 1 ? lanes_.front()->name() : std::string_view(name_);
    }
    bool is_accelerated() const override { return lanes_.front()->is_accelerated(); }
    std::size_t lanes() const override { return lanes_.size(); }

    void upload(const PdhcgData& shared, const std::vector<LaneBounds>& b) override {
        if (b.size() != lanes_.size())
            throw std::invalid_argument("lanes device: one LaneBounds per lane");
        for (std::size_t l = 0; l < lanes_.size(); ++l) {
            PdhcgData d = shared;
            d.col_lo = b[l].col_lo;
            d.col_hi = b[l].col_hi;
            d.row_lo = b[l].row_lo;
            d.row_hi = b[l].row_hi;
            lanes_[l]->upload(d);
        }
    }
    void init() override { for (auto* d : lanes_) d->init(); }

    void outer_begin(const LaneMask& m) override {
        each(m, [](PdhcgDevice& d, std::size_t) { d.outer_begin(); });
    }
    void diag_prox(const std::vector<f64>& tau, const LaneMask& m) override {
        each(m, [&](PdhcgDevice& d, std::size_t l) { d.diag_prox(tau[l]); });
    }
    void inner_begin(const LaneMask& m) override {
        each(m, [](PdhcgDevice& d, std::size_t) { d.inner_begin(); });
    }
    std::vector<f64> inner_grad(const std::vector<f64>& tau, const LaneMask& m) override {
        std::vector<f64> out(lanes_.size(), 0.0);
        each(m, [&](PdhcgDevice& d, std::size_t l) { out[l] = d.inner_grad(tau[l]); });
        return out;
    }
    std::vector<BbPair> inner_trial(const std::vector<f64>& alpha, const std::vector<f64>& tau,
                                    const LaneMask& m) override {
        std::vector<BbPair> out(lanes_.size());
        each(m, [&](PdhcgDevice& d, std::size_t l) { out[l] = d.inner_trial(alpha[l], tau[l]); });
        return out;
    }
    void inner_end(const LaneMask& m) override {
        each(m, [](PdhcgDevice& d, std::size_t) { d.inner_end(); });
    }
    void inner_advance_blind(const std::vector<f64>& alpha, const std::vector<f64>& tau,
                             const LaneMask& m, int j) override {
        each(m, [&](PdhcgDevice& d, std::size_t l) { d.inner_advance_blind(alpha[l], tau[l], j); });
    }
    std::vector<f64> dual_and_advance(const std::vector<f64>& sigma, bool halpern, f64 a,
                                      f64 theta, bool want_movement, const LaneMask& m) override {
        std::vector<f64> out(lanes_.size(), 0.0);
        each(m, [&](PdhcgDevice& d, std::size_t l) {
            out[l] = d.dual_and_advance(sigma[l], halpern, a, theta, want_movement);
        });
        return out;
    }
    std::vector<Eval> evaluate(bool at_average, const LaneMask& m) override {
        std::vector<Eval> out(lanes_.size());
        each(m, [&](PdhcgDevice& d, std::size_t l) { out[l] = d.evaluate(at_average); });
        return out;
    }
    void average_reset(const LaneMask& m) override {
        each(m, [](PdhcgDevice& d, std::size_t) { d.average_reset(); });
    }
    void average_add(const LaneMask& m) override {
        each(m, [](PdhcgDevice& d, std::size_t) { d.average_add(); });
    }
    void restart(const std::vector<std::uint8_t>& to_average, const LaneMask& m) override {
        each(m, [&](PdhcgDevice& d, std::size_t l) { d.restart(to_average[l] != 0); });
    }
    std::vector<Movement> movement_since_restart(const LaneMask& m) override {
        std::vector<Movement> out(lanes_.size());
        each(m, [&](PdhcgDevice& d, std::size_t l) { out[l] = d.movement_since_restart(); });
        return out;
    }
    void download(std::size_t lane, std::vector<f64>& x, std::vector<f64>& y) override {
        lanes_.at(lane)->download(x, y);
    }
    TransferStats transfer_stats() const override {
        TransferStats t{};
        for (auto* d : lanes_) {
            const auto s = d->transfer_stats();
            t.h2d_bytes += s.h2d_bytes; t.d2h_bytes += s.d2h_bytes;
            t.h2d_ms += s.h2d_ms; t.d2h_ms += s.d2h_ms; t.kernel_ms += s.kernel_ms;
            t.ipc_ms += s.ipc_ms; t.calls += s.calls;
        }
        return t;
    }
    void reset_stats() override { for (auto* d : lanes_) d->reset_stats(); }

private:
    template <class F>
    void each(const LaneMask& m, F&& f) {
        for (std::size_t l = 0; l < lanes_.size(); ++l)
            if (m[l]) f(*lanes_[l], l);
    }

    std::vector<std::unique_ptr<PdhcgDevice>> owned_;
    std::vector<PdhcgDevice*> lanes_;
    std::string name_;
};

}  // namespace

std::unique_ptr<BatchedPdhcgDevice> make_lanes_device(
    std::vector<std::unique_ptr<PdhcgDevice>> lanes) {
    std::vector<PdhcgDevice*> view;
    for (auto& d : lanes) view.push_back(d.get());
    return std::make_unique<LanesDevice>(std::move(lanes), std::move(view));
}

std::unique_ptr<BatchedPdhcgDevice> make_lanes_view(std::vector<PdhcgDevice*> lanes) {
    return std::make_unique<LanesDevice>(std::vector<std::unique_ptr<PdhcgDevice>>{},
                                         std::move(lanes));
}

}  // namespace sor::backend
