// SOR — CPU BinQuadDevice: the reference semantics of the parallel search.
//
// LAYER L1.  Each search is advanced exactly as the four device kernels do
// it (gain, select, apply, save_best), in the same arithmetic order, with the
// same hash and generator from binquad_device.hpp.  vk_binquad_device.cpp is
// checked against this file.
#include "sor/backend/binquad_device.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sor::backend {
namespace {

constexpr f64 kHuge = 1e300;          // "no incumbent yet" / excluded score
constexpr f64 kFeasTol = 1e-9;
constexpr f64 kImproveTol = 1e-12;

inline f64 row_viol(f64 v, f64 lo, f64 hi) {
    if (v < lo) return lo - v;
    if (v > hi) return v - hi;
    return 0.0;
}

class CpuBinQuadDevice final : public BinQuadDevice {
public:
    std::string_view name() const override { return "cpu"; }
    bool is_accelerated() const override { return false; }

    void upload(const BqData& d, const BqParams& p) override {
        d_ = d;
        p_ = p;
        n_ = static_cast<std::size_t>(d.n);
        m_ = static_cast<std::size_t>(d.m);
        const std::size_t P = p.searches;
        x_.assign(P * n_, 0.0);
        best_x_.assign(P * n_, 0.0);
        h_.assign(P * n_, 0.0);
        tabu_.assign(P * n_, 0.0);
        dobj_.assign(P * n_, 0.0);
        dviol_.assign(P * n_, 0.0);
        score_.assign(P * n_, 0.0);
        r_.assign(P * m_, 0.0);
        st_.assign(P, BqState{});
        rng_.resize(P);
        for (std::uint32_t s = 0; s < P; ++s) {
            st_[s].penalty = p.penalty0;
            rng_[s] = bq_hash(p.seed, s, 0x5EEDu, 1u) | 1u;
        }
        elite_.clear();
    }

    void set_elite(const std::vector<std::vector<std::uint8_t>>& elite) override {
        elite_ = elite;
    }

    void restart(const std::vector<std::uint8_t>& restart,
                 const std::vector<std::int32_t>& base, f64 flip_prob,
                 std::uint32_t epoch) override {
        for (std::uint32_t s = 0; s < p_.searches; ++s) {
            if (!restart[s]) continue;
            // perturb
            for (std::size_t j = 0; j < n_; ++j) {
                const std::uint32_t b =
                    base[s] >= 0 ? elite_[static_cast<std::size_t>(base[s])][j] : 0u;
                const f64 u = static_cast<f64>(bq_hash(p_.seed, epoch, s,
                                                       static_cast<std::uint32_t>(j)) >> 8) *
                              (1.0 / 16777216.0);
                x_[s * n_ + j] = static_cast<f64>(b ^ (u < flip_prob ? 1u : 0u));
            }
            // rebuild h, r; reset tabu
            for (std::size_t j = 0; j < n_; ++j) {
                f64 acc = 0.0;
                for (auto k = d_.m_start[j]; k < d_.m_start[j + 1]; ++k)
                    acc += d_.m_val[static_cast<std::size_t>(k)] *
                           x_[s * n_ + static_cast<std::size_t>(d_.m_idx[static_cast<std::size_t>(k)])];
                h_[s * n_ + j] = acc;
                tabu_[s * n_ + j] = 0.0;
            }
            for (std::size_t i = 0; i < m_; ++i) {
                f64 acc = 0.0;
                for (auto k = d_.a_rstart[i]; k < d_.a_rstart[i + 1]; ++k)
                    acc += d_.a_rval[static_cast<std::size_t>(k)] *
                           x_[s * n_ + static_cast<std::size_t>(d_.a_rcol[static_cast<std::size_t>(k)])];
                r_[s * m_ + i] = acc;
            }
            // rescore
            auto& t = st_[s];
            f64 obj = 0.0, viol = 0.0;
            for (std::size_t j = 0; j < n_; ++j)
                obj += x_[s * n_ + j] * (d_.lin[j] + 0.5 * h_[s * n_ + j]);
            for (std::size_t i = 0; i < m_; ++i)
                viol += row_viol(r_[s * m_ + i], d_.c_lo[i], d_.c_hi[i]);
            t.obj = obj;
            t.viol = viol;
            t.since = 0.0;
            t.stalled = false;
            record_if_improved(s);
        }
    }

    void run(std::uint32_t iterations) override {
        for (std::uint32_t it = 0; it < iterations; ++it)
            for (std::uint32_t s = 0; s < p_.searches; ++s) step(s);
    }

    void read_searches(std::vector<BqSearch>& out) override {
        out.resize(p_.searches);
        for (std::uint32_t s = 0; s < p_.searches; ++s) {
            const auto& t = st_[s];
            out[s] = BqSearch{t.obj, t.viol, t.penalty, t.best_obj, t.best_viol,
                              t.iter, t.since, t.have_feasible, t.stalled};
        }
    }

    void read_best(std::uint32_t s, std::vector<std::uint8_t>& x) override {
        x.resize(n_);
        for (std::size_t j = 0; j < n_; ++j)
            x[j] = best_x_[s * n_ + j] != 0.0 ? 1 : 0;
    }

    TransferStats transfer_stats() const override { return {}; }

private:
    struct BqState {
        f64 obj = 0.0, viol = 0.0, penalty = 0.0;
        f64 best_obj = kHuge, best_viol = kHuge;
        f64 iter = 0.0, since = 0.0;
        bool have_feasible = false, stalled = false;
    };

    // The incumbent rule, shared by a step and a restart.
    void record_if_improved(std::uint32_t s) {
        auto& t = st_[s];
        bool improved = false;
        if (t.viol <= kFeasTol && t.obj < t.best_obj - kImproveTol) {
            t.best_obj = t.obj;
            t.best_viol = 0.0;
            t.have_feasible = true;
            improved = true;
        } else if (!t.have_feasible && t.viol < t.best_viol) {
            t.best_viol = t.viol;
            improved = true;
        }
        if (improved)
            std::copy(x_.begin() + static_cast<std::ptrdiff_t>(s * n_),
                      x_.begin() + static_cast<std::ptrdiff_t>((s + 1) * n_),
                      best_x_.begin() + static_cast<std::ptrdiff_t>(s * n_));
        last_improved_ = improved;
    }

    void step(std::uint32_t s) {
        auto& t = st_[s];
        const std::size_t off = s * n_, roff = s * m_;
        // gain
        for (std::size_t j = 0; j < n_; ++j) {
            const f64 delta = x_[off + j] != 0.0 ? -1.0 : 1.0;
            const f64 dobj = delta * (d_.lin[j] + h_[off + j]);
            f64 dviol = 0.0;
            for (auto k = d_.a_cstart[j]; k < d_.a_cstart[j + 1]; ++k) {
                const auto i = static_cast<std::size_t>(d_.a_crow[static_cast<std::size_t>(k)]);
                const f64 a = d_.a_cval[static_cast<std::size_t>(k)];
                dviol += row_viol(r_[roff + i] + delta * a, d_.c_lo[i], d_.c_hi[i]) -
                         row_viol(r_[roff + i], d_.c_lo[i], d_.c_hi[i]);
            }
            const f64 score = dobj + t.penalty * dviol;
            const bool tabu = tabu_[off + j] > t.iter;
            const bool aspires = tabu && (t.viol + dviol) <= 0.0 &&
                                 (t.obj + dobj) < t.best_obj - kImproveTol;
            dobj_[off + j] = dobj;
            dviol_[off + j] = dviol;
            score_[off + j] = (tabu && !aspires) ? kHuge : score;
        }
        // select
        std::size_t best_j = 0;
        f64 best_score = kHuge;
        for (std::size_t j = 0; j < n_; ++j)
            if (score_[off + j] < best_score) { best_score = score_[off + j]; best_j = j; }
        if (!(best_score < kHuge)) best_j = bq_xorshift(rng_[s]) % static_cast<std::uint32_t>(n_);
        t.obj += dobj_[off + best_j];
        t.viol += dviol_[off + best_j];
        const std::uint32_t tenure =
            p_.tenure_min + bq_xorshift(rng_[s]) % std::max(1u, p_.tenure_span);
        tabu_[off + best_j] = t.iter + static_cast<f64>(tenure);
        // apply (before the incumbent test: best_x must be the flipped x)
        const f64 delta = x_[off + best_j] != 0.0 ? -1.0 : 1.0;
        for (auto k = d_.m_start[best_j]; k < d_.m_start[best_j + 1]; ++k)
            h_[off + static_cast<std::size_t>(d_.m_idx[static_cast<std::size_t>(k)])] +=
                delta * d_.m_val[static_cast<std::size_t>(k)];
        for (auto k = d_.a_cstart[best_j]; k < d_.a_cstart[best_j + 1]; ++k)
            r_[roff + static_cast<std::size_t>(d_.a_crow[static_cast<std::size_t>(k)])] +=
                delta * d_.a_cval[static_cast<std::size_t>(k)];
        x_[off + best_j] = x_[off + best_j] != 0.0 ? 0.0 : 1.0;
        // bookkeeping + save_best
        record_if_improved(s);
        t.since = last_improved_ ? 0.0 : t.since + 1.0;
        if (t.viol > kFeasTol) t.penalty *= 1.0005;
        else t.penalty = std::max(t.penalty * 0.999, p_.penalty_floor);
        t.iter += 1.0;
        if (t.since >= static_cast<f64>(p_.stagnation_limit)) t.stalled = true;
    }

    BqData d_;
    BqParams p_;
    std::size_t n_ = 0, m_ = 0;
    std::vector<f64> x_, best_x_, h_, tabu_, dobj_, dviol_, score_, r_;
    std::vector<BqState> st_;
    std::vector<std::uint32_t> rng_;
    std::vector<std::vector<std::uint8_t>> elite_;
    bool last_improved_ = false;
};

}  // namespace

std::unique_ptr<BinQuadDevice> make_cpu_binquad_device() {
    return std::make_unique<CpuBinQuadDevice>();
}

std::unique_ptr<BinQuadDevice> make_binquad_device(std::string_view name, int device) {
    if (name == "cpu") return make_cpu_binquad_device();
    if (name == "vulkan") return make_vulkan_binquad_device(device);
    return nullptr;
}

}  // namespace sor::backend
