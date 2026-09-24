// SOR — Vulkan BinQuadDevice: P parallel flip-tabu searches on the GPU.
//
// LAYER L1.  Semantics are cpu_binquad_device.cpp's; see binquad_device.hpp
// for the kernel split.  run(k) records k iterations x 4 dispatches into one
// command buffer and submits once: no host decision is needed inside an
// epoch, so nothing crosses the bus until the host asks for per-search state
// (16 doubles per search) or one search's best point.
#include "sor/backend/binquad_device.hpp"
#include "vk_compute.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sor::backend {
namespace {

using vkc::Buf;
using vkc::Kernel;

constexpr std::size_t kStride = 16;      // doubles of state per search
constexpr std::size_t kMaxElite = 16;
constexpr f64 kHuge = 1e300;

inline uint32_t u32(std::size_t v) { return static_cast<uint32_t>(v); }

class VulkanBinQuadDevice final : public BinQuadDevice {
public:
    explicit VulkanBinQuadDevice(std::unique_ptr<vk::Context> ctx) : vk_(std::move(ctx)) {
        k_gain_ = vk_.make_kernel("bq_gain", 14, 16);
        k_select_ = vk_.make_kernel("bq_select", 6, 24);
        k_apply_ = vk_.make_kernel("bq_apply", 10, 16);
        k_save_ = vk_.make_kernel("bq_save_best", 3, 8);
        k_perturb_ = vk_.make_kernel("bq_perturb", 4, 24);
        k_rh_ = vk_.make_kernel("bq_rebuild_h", 6, 8);
        k_rr_ = vk_.make_kernel("bq_rebuild_r", 6, 16);
        k_rescore_ = vk_.make_kernel("bq_rescore", 7, 16);
    }
    ~VulkanBinQuadDevice() override {
        vk_.flush();
        destroy_bufs();
        for (Kernel* k : {&k_gain_, &k_select_, &k_apply_, &k_save_, &k_perturb_, &k_rh_,
                          &k_rr_, &k_rescore_})
            vk_.kill_kernel(*k);
    }

    std::string_view name() const override { return "vulkan"; }
    bool is_accelerated() const override { return true; }

    void upload(const BqData& d, const BqParams& p) override {
        vk_.reset_sets();
        destroy_bufs();
        n_ = static_cast<std::size_t>(d.n);
        m_ = static_cast<std::size_t>(d.m);
        p_ = p;
        const std::size_t P = p.searches;
        if (P * n_ > 0x7FFFFFFFu || P * m_ > 0x7FFFFFFFu)
            throw std::runtime_error("VulkanBinQuadDevice: P x n exceeds 32-bit indexing");

        lin_ = vk_.dev_vec(d.lin);
        ms_ = vk_.dev_ints(d.m_start);
        mi_ = vk_.dev_ints(d.m_idx);
        mv_ = vk_.dev_vec(d.m_val);
        acs_ = vk_.dev_ints(d.a_cstart);
        acr_ = vk_.dev_ints(d.a_crow);
        acv_ = vk_.dev_vec(d.a_cval);
        ars_ = vk_.dev_ints(d.a_rstart);
        arc_ = vk_.dev_ints(d.a_rcol);
        arv_ = vk_.dev_vec(d.a_rval);
        clo_ = vk_.dev_vec(d.c_lo);
        chi_ = vk_.dev_vec(d.c_hi);

        for (Buf* b : {&x_, &best_x_, &h_, &tabu_, &dobj_, &dviol_, &score_})
            *b = vk_.dev_zero(P * n_);
        r_ = vk_.dev_zero(P * m_);
        elite_ = vk_.dev_zero(kMaxElite * n_);

        std::vector<f64> st(P * kStride, 0.0);
        std::vector<int32_t> rng(P);
        for (std::uint32_t s = 0; s < P; ++s) {
            st[s * kStride + 2] = p.penalty0;
            st[s * kStride + 3] = kHuge;
            st[s * kStride + 4] = kHuge;
            rng[s] = static_cast<int32_t>(bq_hash(p.seed, s, 0x5EEDu, 1u) | 1u);
        }
        st_ = vk_.dev_vec(st);
        rng_ = vk_.dev_ints(rng);
        uploaded_ = true;
    }

    void set_elite(const std::vector<std::vector<std::uint8_t>>& elite) override {
        if (elite.size() > kMaxElite) throw std::logic_error("set_elite: pool too large");
        std::vector<f64> flat(kMaxElite * n_, 0.0);
        for (std::size_t e = 0; e < elite.size(); ++e)
            for (std::size_t j = 0; j < n_; ++j) flat[e * n_ + j] = elite[e][j];
        vk_.upload(elite_, flat);
    }

    void restart(const std::vector<std::uint8_t>& restart,
                 const std::vector<std::int32_t>& base, f64 flip_prob,
                 std::uint32_t epoch) override {
        require_up();
        const std::size_t P = p_.searches;
        std::vector<f64> st(P * kStride);
        vk_.download(st_, st);
        for (std::size_t s = 0; s < P; ++s) {
            st[s * kStride + 11] = restart[s] ? 1.0 : 0.0;
            st[s * kStride + 12] = static_cast<f64>(base[s]);
        }
        vk_.upload(st_, st);

        const uint32_t n = u32(n_), m = u32(m_), Pu = u32(P);
        struct { uint32_t n, P, seed, epoch; double prob; } pp{n, Pu, p_.seed, epoch, flip_prob};
        vk_.rec(k_perturb_, {&x_, &elite_, &st_, &tabu_}, &pp, sizeof(pp), P * n_);
        struct { uint32_t n, P; } p2{n, Pu};
        vk_.rec(k_rh_, {&x_, &h_, &ms_, &mi_, &mv_, &st_}, &p2, sizeof(p2), P * n_);
        struct { uint32_t n, m, P, pad; } p4{n, m, Pu, 0};
        vk_.rec(k_rr_, {&x_, &r_, &ars_, &arc_, &arv_, &st_}, &p4, sizeof(p4), P * m_);
        vk_.rec(k_rescore_, {&x_, &h_, &lin_, &r_, &clo_, &chi_, &st_}, &p4, sizeof(p4), 0,
                Pu);
        vk_.rec(k_save_, {&x_, &best_x_, &st_}, &p2, sizeof(p2), P * n_);
        vk_.flush();
    }

    void run(std::uint32_t iterations) override {
        require_up();
        const std::size_t P = p_.searches;
        const uint32_t n = u32(n_), m = u32(m_), Pu = u32(P);
        struct { uint32_t n, m, P, pad; } p4{n, m, Pu, 0};
        struct { uint32_t n, tmin, tspan, stag; double floor; } ps{
            n, p_.tenure_min, p_.tenure_span, p_.stagnation_limit, p_.penalty_floor};
        struct { uint32_t n, P; } p2{n, Pu};
        for (std::uint32_t it = 0; it < iterations; ++it) {
            vk_.rec(k_gain_, {&x_, &h_, &tabu_, &lin_, &acs_, &acr_, &acv_, &r_, &clo_, &chi_,
                              &st_, &dobj_, &dviol_, &score_},
                    &p4, sizeof(p4), P * n_);
            vk_.rec(k_select_, {&score_, &dobj_, &dviol_, &tabu_, &st_, &rng_}, &ps,
                    sizeof(ps), 0, Pu);
            vk_.rec(k_apply_, {&x_, &h_, &r_, &ms_, &mi_, &mv_, &acs_, &acr_, &acv_, &st_},
                    &p4, sizeof(p4), 0, std::max(1u, (Pu + 63u) / 64u));
            vk_.rec(k_save_, {&x_, &best_x_, &st_}, &p2, sizeof(p2), P * n_);
        }
        vk_.flush();
    }

    void read_searches(std::vector<BqSearch>& out) override {
        require_up();
        const std::size_t P = p_.searches;
        std::vector<f64> st(P * kStride);
        vk_.download(st_, st);
        out.resize(P);
        for (std::size_t s = 0; s < P; ++s) {
            const f64* t = &st[s * kStride];
            out[s] = BqSearch{t[0], t[1], t[2], t[3], t[4], t[5], t[6],
                              t[7] != 0.0, t[10] != 0.0};
        }
    }

    void read_best(std::uint32_t s, std::vector<std::uint8_t>& x) override {
        require_up();
        std::vector<f64> v(n_);
        vk_.download_range(best_x_, static_cast<std::size_t>(s) * n_, v);
        x.resize(n_);
        for (std::size_t j = 0; j < n_; ++j) x[j] = v[j] != 0.0 ? 1 : 0;
    }

    TransferStats transfer_stats() const override {
        return const_cast<vkc::Compute&>(vk_).stats();
    }

private:
    void require_up() const {
        if (!uploaded_) throw std::logic_error("VulkanBinQuadDevice: upload() required");
    }
    void destroy_bufs() {
        for (Buf* b : {&lin_, &ms_, &mi_, &mv_, &acs_, &acr_, &acv_, &ars_, &arc_, &arv_,
                       &clo_, &chi_, &x_, &best_x_, &h_, &tabu_, &dobj_, &dviol_, &score_,
                       &r_, &elite_, &st_, &rng_})
            vk_.destroy_buf(*b);
        uploaded_ = false;
    }

    vkc::Compute vk_;
    bool uploaded_ = false;
    std::size_t n_ = 0, m_ = 0;
    BqParams p_;
    Kernel k_gain_, k_select_, k_apply_, k_save_, k_perturb_, k_rh_, k_rr_, k_rescore_;
    Buf lin_, ms_, mi_, mv_, acs_, acr_, acv_, ars_, arc_, arv_, clo_, chi_;
    Buf x_, best_x_, h_, tabu_, dobj_, dviol_, score_, r_, elite_, st_, rng_;
};

}  // namespace

std::unique_ptr<BinQuadDevice> make_vulkan_binquad_device(int device) {
    auto ctx = vk::Context::create(device);
    if (!ctx || !ctx->info().shader_float64) return nullptr;
    try {
        return std::make_unique<VulkanBinQuadDevice>(std::move(ctx));
    } catch (const std::exception&) {
        return nullptr;
    }
}

}  // namespace sor::backend
