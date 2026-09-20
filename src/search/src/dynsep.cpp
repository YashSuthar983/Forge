#include "sor/search/dynsep.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <utility>

namespace sor::search {
namespace {

inline f64 relu(f64 v) { return v > 0.0 ? v : 0.0; }

inline f64 sigmoid(f64 v) {
    if (v > 20.0) return 1.0;
    if (v < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-v));
}

std::uint64_t splitmix64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

f64 xavier(std::uint64_t& rng, int fan_in, int fan_out) {
    const f64 a = std::sqrt(6.0 / static_cast<f64>(fan_in + fan_out));
    const f64 u =
        static_cast<f64>(splitmix64(rng) >> 11) / static_cast<f64>(1ull << 53);
    return (2.0 * u - 1.0) * a;
}

void matvec(const std::vector<f64>& W, const f64* x, f64* y, int rows,
            int cols) {
    for (int r = 0; r < rows; ++r) {
        f64 s = 0.0;
        const f64* wr = W.data() + static_cast<std::size_t>(r * cols);
        for (int c = 0; c < cols; ++c) s += wr[c] * x[c];
        y[r] = s;
    }
}

void add_bias(f64* y, const std::vector<f64>& b, int n) {
    for (int i = 0; i < n; ++i) y[i] += b[static_cast<std::size_t>(i)];
}

inline int budget_for(const DynSepOptions& o, SepFamily f) {
    switch (f) {
    case SepFamily::Gmi: return o.budget_gmi;
    case SepFamily::Mir: return o.budget_mir;
    case SepFamily::Cover: return o.budget_cover;
    case SepFamily::Clique: return o.budget_clique;
    case SepFamily::ImpliedBound: return o.budget_ib;
    case SepFamily::ZeroHalf: return o.budget_zerohalf;
    case SepFamily::FlowCover: return o.budget_flowcover;
    case SepFamily::Count: break;
    }
    return 0;
}

inline void set_run(DynSepDecision& d, SepFamily f, int budget) {
    switch (f) {
    case SepFamily::Gmi:
        d.run_gmi = true;
        d.budget_gmi = budget;
        break;
    case SepFamily::Mir:
        d.run_mir = true;
        d.budget_mir = budget;
        break;
    case SepFamily::Cover:
        d.run_cover = true;
        d.budget_cover = budget;
        break;
    case SepFamily::Clique:
        d.run_clique = true;
        d.budget_clique = budget;
        break;
    case SepFamily::ImpliedBound:
        d.run_ib = true;
        d.budget_ib = budget;
        break;
    case SepFamily::ZeroHalf:
        d.run_zerohalf = true;
        d.budget_zerohalf = budget;
        break;
    case SepFamily::FlowCover:
        d.run_flowcover = true;
        d.budget_flowcover = budget;
        break;
    case SepFamily::Count:
        break;
    }
}

inline bool decision_runs(const DynSepDecision& d, SepFamily f) {
    switch (f) {
    case SepFamily::Gmi: return d.run_gmi;
    case SepFamily::Mir: return d.run_mir;
    case SepFamily::Cover: return d.run_cover;
    case SepFamily::Clique: return d.run_clique;
    case SepFamily::ImpliedBound: return d.run_ib;
    case SepFamily::ZeroHalf: return d.run_zerohalf;
    case SepFamily::FlowCover: return d.run_flowcover;
    case SepFamily::Count: break;
    }
    return false;
}

inline int count_scheduled(const DynSepDecision& d) {
    int n = 0;
    if (d.run_gmi) ++n;
    if (d.run_mir) ++n;
    if (d.run_cover) ++n;
    if (d.run_clique) ++n;
    if (d.run_ib) ++n;
    if (d.run_zerohalf) ++n;
    if (d.run_flowcover) ++n;
    return n;
}

enum EdgeType : int {
    kStateSep = 0,
    kSepState = 1,
    kSepSep = 2,
    kDeltaSep = 3,
    kSepDelta = 4,
    kNEdge = 5,
};

const f64* msg_W(const DynSepGnnWeights& g, int layer, int etype) {
    const int e = g.emb_dim;
    const std::size_t off =
        static_cast<std::size_t>(((layer * kNEdge) + etype) * e * e);
    return g.W_msg.data() + off;
}

const f64* self_W(const DynSepGnnWeights& g, int layer, int ntype) {
    const int e = g.emb_dim;
    const std::size_t off =
        static_cast<std::size_t>(((layer * 3) + ntype) * e * e);
    return g.W_self.data() + off;
}

const f64* upd_b(const DynSepGnnWeights& g, int layer, int ntype) {
    const int e = g.emb_dim;
    return g.b_upd.data() +
           static_cast<std::size_t>(((layer * 3) + ntype) * e);
}

void write_vec(std::ostream& out, const char* key, const std::vector<f64>& v) {
    out << key << " " << v.size();
    for (f64 x : v) out << " " << x;
    out << "\n";
}

bool read_vec(std::istream& in, std::vector<f64>& v) {
    std::size_t n = 0;
    if (!(in >> n)) return false;
    v.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(in >> v[i])) return false;
    }
    return true;
}

}  // namespace

void DynSepGnnWeights::clear() {
    emb_dim = kDynSepEmbDim;
    n_msg_layers = kDynSepMsgLayers;
    state_dim = kDynSepStateFeatDim;
    sep_dim = kDynSepSepFeatDim;
    n_sep = kSepFamilyCount;
    W_state.clear();
    b_state.clear();
    W_sep.clear();
    b_sep.clear();
    W_delta.clear();
    b_delta.clear();
    W_msg.clear();
    W_self.clear();
    b_upd.clear();
    W_on.clear();
    b_on = 0.0;
    W_bud.clear();
    b_bud = 0.0;
    W_rounds.clear();
    b_rounds = 0.0;
    t_max = 5;
}

void DynSepGnnWeights::init_xavier(int emb, int n_layers, std::uint64_t seed) {
    clear();
    emb_dim = emb;
    n_msg_layers = n_layers;
    std::uint64_t rng = seed ? seed : 1ull;
    auto fill_mat = [&](std::vector<f64>& W, int rows, int cols) {
        W.resize(static_cast<std::size_t>(rows * cols));
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                W[static_cast<std::size_t>(r * cols + c)] =
                    xavier(rng, cols, rows);
    };
    auto fill_bias = [&](std::vector<f64>& b, int n) {
        b.assign(static_cast<std::size_t>(n), 0.0);
    };
    fill_mat(W_state, emb, state_dim);
    fill_bias(b_state, emb);
    fill_mat(W_sep, emb, sep_dim);
    fill_bias(b_sep, emb);
    fill_mat(W_delta, emb, state_dim);
    fill_bias(b_delta, emb);
    fill_mat(W_msg, kNEdge * n_layers * emb, emb);
    fill_mat(W_self, n_layers * 3 * emb, emb);
    fill_bias(b_upd, n_layers * 3 * emb);
    W_on.resize(static_cast<std::size_t>(emb));
    W_bud.resize(static_cast<std::size_t>(emb));
    W_rounds.resize(static_cast<std::size_t>(emb));
    for (int d = 0; d < emb; ++d) {
        W_on[static_cast<std::size_t>(d)] = xavier(rng, emb, 1);
        W_bud[static_cast<std::size_t>(d)] = xavier(rng, emb, 1);
        W_rounds[static_cast<std::size_t>(d)] = xavier(rng, emb, 1);
    }
    b_on = 0.0;
    b_bud = 0.0;
    b_rounds = 0.0;
}

bool DynSepGnnWeights::valid() const {
    if (emb_dim <= 0 || n_msg_layers <= 0) return false;
    const int e = emb_dim;
    if (static_cast<int>(W_state.size()) != e * state_dim) return false;
    if (static_cast<int>(b_state.size()) != e) return false;
    if (static_cast<int>(W_sep.size()) != e * sep_dim) return false;
    if (static_cast<int>(b_sep.size()) != e) return false;
    if (static_cast<int>(W_delta.size()) != e * state_dim) return false;
    if (static_cast<int>(b_delta.size()) != e) return false;
    if (static_cast<int>(W_msg.size()) != kNEdge * n_msg_layers * e * e)
        return false;
    if (static_cast<int>(W_self.size()) != n_msg_layers * 3 * e * e)
        return false;
    if (static_cast<int>(b_upd.size()) != n_msg_layers * 3 * e) return false;
    if (static_cast<int>(W_on.size()) != e) return false;
    if (static_cast<int>(W_bud.size()) != e) return false;
    if (static_cast<int>(W_rounds.size()) != e) return false;
    return true;
}

void DynSepModel::clear() {
    gnn.clear();
    loaded = false;
}

void DynSepCollector::add(DynSepSample s) {
    if (samples.size() >= max_samples) return;
    samples.push_back(std::move(s));
}

void fill_dynsep_state_feats(const DynSepRoundInput& in,
                             const DynSepOptions& opts, DynSepStateFeat& out) {
    out.fill(0.0);
    out[0] = std::min(1.0, static_cast<f64>(std::max(0, in.round)) / 16.0);
    out[1] = std::min(1.0, static_cast<f64>(std::max(0, in.depth)) / 64.0);
    out[2] = in.at_root ? 1.0 : 0.0;
    out[3] = std::min(1.0, std::max(0.0, in.last_bound_gain_rel));
    out[4] = in.have_basis ? 1.0 : 0.0;
    out[5] = in.have_conflict_graph ? 1.0 : 0.0;
    out[6] = std::min(1.0, std::max(0.0, in.frac_share));
    out[7] = std::min(1.0, std::max(0.0, in.mean_frac));
    out[8] = std::isfinite(in.gap_rel)
                 ? std::min(1.0, std::max(0.0, in.gap_rel))
                 : 1.0;
    out[9] = std::min(1.0, static_cast<f64>(std::max(0, in.n_cuts_last)) / 50.0);
    out[10] = in.last_bound_gain_rel < 1e-5 ? 1.0 : 0.0;  // stalled
    out[11] = std::min(1.0, static_cast<f64>(std::max(0, opts.max_optional_arms)) /
                                7.0);
}

void fill_dynsep_sep_feats(SepFamily f, const DynSepRoundInput& in,
                           const DynSepOptions& opts, f64 mean_reward,
                           f64 plays_norm, f64 cuts_norm, bool allowed,
                           DynSepSepFeat& out) {
    out.fill(0.0);
    const int fi = static_cast<int>(f);
    if (fi >= 0 && fi < kSepFamilyCount)
        out[static_cast<std::size_t>(fi)] = 1.0;  // one-hot in [0..6]
    out[7] = allowed ? 1.0 : 0.0;
    out[8] = std::min(1.0, std::max(0.0, mean_reward));
    out[9] = std::min(1.0, std::max(0.0, plays_norm));
    out[10] = std::min(1.0, std::max(0.0, cuts_norm));
    bool force = false;
    switch (f) {
    case SepFamily::Mir: force = in.force_mir; break;
    case SepFamily::Cover: force = in.force_cover; break;
    case SepFamily::Clique: force = in.force_clique; break;
    case SepFamily::ImpliedBound: force = in.force_ib; break;
    case SepFamily::ZeroHalf: force = in.force_zerohalf; break;
    case SepFamily::FlowCover: force = in.force_flowcover; break;
    case SepFamily::Gmi:
    case SepFamily::Count:
        break;
    }
    out[11] = force ? 1.0 : 0.0;
    (void)opts;
}

void dynsep_forward(const DynSepGnnWeights& g, const DynSepRoundGraph& graph,
                    std::array<f64, kSepFamilyCount>& logits_on,
                    std::array<f64, kSepFamilyCount>& budget_scale,
                    int& max_rounds_hint) {
    logits_on.fill(0.0);
    budget_scale.fill(1.0);
    max_rounds_hint = 0;
    if (!g.valid()) return;

    const int e = g.emb_dim;
    const std::size_t K = static_cast<std::size_t>(kSepFamilyCount);
    std::vector<f64> H_state(static_cast<std::size_t>(e), 0.0);
    std::vector<f64> H_delta(static_cast<std::size_t>(e), 0.0);
    std::vector<f64> H_sep(K * static_cast<std::size_t>(e), 0.0);

    matvec(g.W_state, graph.state.data(), H_state.data(), e, g.state_dim);
    add_bias(H_state.data(), g.b_state, e);
    matvec(g.W_delta, graph.delta.data(), H_delta.data(), e, g.state_dim);
    add_bias(H_delta.data(), g.b_delta, e);
    for (std::size_t k = 0; k < K; ++k) {
        matvec(g.W_sep, graph.seps[k].data(),
               H_sep.data() + k * static_cast<std::size_t>(e), e, g.sep_dim);
        add_bias(H_sep.data() + k * static_cast<std::size_t>(e), g.b_sep, e);
    }

    std::vector<f64> msg(static_cast<std::size_t>(e), 0.0);
    for (int layer = 0; layer < g.n_msg_layers; ++layer) {
        std::vector<f64> Acc_state(static_cast<std::size_t>(e), 0.0);
        std::vector<f64> Acc_delta(static_cast<std::size_t>(e), 0.0);
        std::vector<f64> Acc_sep(K * static_cast<std::size_t>(e), 0.0);
        f64 cnt_state = 0.0, cnt_delta = 0.0;
        std::vector<f64> cnt_sep(K, 0.0);

        auto accum = [&](EdgeType et, const f64* h_src, f64* acc, f64& cnt) {
            const std::vector<f64> W(msg_W(g, layer, et),
                                     msg_W(g, layer, et) + e * e);
            matvec(W, h_src, msg.data(), e, e);
            for (int d = 0; d < e; ++d)
                acc[d] += msg[static_cast<std::size_t>(d)];
            cnt += 1.0;
        };

        // state ↔ each sep, delta ↔ each sep, sep ↔ sep (complete).
        for (std::size_t k = 0; k < K; ++k) {
            const f64* hk = H_sep.data() + k * static_cast<std::size_t>(e);
            f64* ak = Acc_sep.data() + k * static_cast<std::size_t>(e);
            accum(kStateSep, H_state.data(), ak, cnt_sep[k]);
            accum(kSepState, hk, Acc_state.data(), cnt_state);
            accum(kDeltaSep, H_delta.data(), ak, cnt_sep[k]);
            accum(kSepDelta, hk, Acc_delta.data(), cnt_delta);
            for (std::size_t j = 0; j < K; ++j) {
                if (j == k) continue;
                f64* aj = Acc_sep.data() + j * static_cast<std::size_t>(e);
                accum(kSepSep, hk, aj, cnt_sep[j]);
            }
        }

        auto update_one = [&](f64* h, const f64* a, f64 cnt, int ntype) {
            const std::vector<f64> Ws(self_W(g, layer, ntype),
                                      self_W(g, layer, ntype) + e * e);
            const f64* bb = upd_b(g, layer, ntype);
            std::vector<f64> self(static_cast<std::size_t>(e), 0.0);
            matvec(Ws, h, self.data(), e, e);
            const f64 c = std::max(1.0, cnt);
            for (int d = 0; d < e; ++d)
                h[d] = relu(self[static_cast<std::size_t>(d)] + a[d] / c +
                            bb[d]);
        };
        update_one(H_state.data(), Acc_state.data(), cnt_state, 0);
        for (std::size_t k = 0; k < K; ++k)
            update_one(H_sep.data() + k * static_cast<std::size_t>(e),
                       Acc_sep.data() + k * static_cast<std::size_t>(e),
                       cnt_sep[k], 1);
        update_one(H_delta.data(), Acc_delta.data(), cnt_delta, 2);
    }

    std::vector<f64> pooled(static_cast<std::size_t>(e), 0.0);
    for (int d = 0; d < e; ++d) {
        pooled[static_cast<std::size_t>(d)] =
            H_state[static_cast<std::size_t>(d)] +
            H_delta[static_cast<std::size_t>(d)];
    }
    for (std::size_t k = 0; k < K; ++k)
        for (int d = 0; d < e; ++d)
            pooled[static_cast<std::size_t>(d)] +=
                H_sep[k * static_cast<std::size_t>(e) +
                      static_cast<std::size_t>(d)];
    const f64 inv = 1.0 / static_cast<f64>(2 + kSepFamilyCount);
    for (int d = 0; d < e; ++d) pooled[static_cast<std::size_t>(d)] *= inv;

    for (std::size_t k = 0; k < K; ++k) {
        const f64* hk = H_sep.data() + k * static_cast<std::size_t>(e);
        f64 on = g.b_on;
        f64 bud = g.b_bud;
        for (int d = 0; d < e; ++d) {
            on += g.W_on[static_cast<std::size_t>(d)] * hk[d];
            bud += g.W_bud[static_cast<std::size_t>(d)] * hk[d];
        }
        logits_on[k] = on;
        // Map to ~[0.25, 2.0] budget scale.
        budget_scale[k] = 0.25 + 1.75 * sigmoid(bud);
    }
    f64 rlogit = g.b_rounds;
    for (int d = 0; d < e; ++d)
        rlogit += g.W_rounds[static_cast<std::size_t>(d)] *
                  pooled[static_cast<std::size_t>(d)];
    const f64 r01 = sigmoid(rlogit);
    max_rounds_hint =
        std::max(1, static_cast<int>(std::lround(r01 * static_cast<f64>(g.t_max))));
}

bool dynsep_prefer_gnn(const DynSepModel& model, const DynSepOptions& opts) {
    if (!model.loaded || !model.gnn.valid()) return false;
    if (opts.backend == DynSepBackend::Ucb) return false;
    if (opts.backend == DynSepBackend::Gnn) return true;
    return true;  // Auto
}

DynSepModel fit_dynsep_imitation(const DynSepCollector& collector,
                                 const DynSepFitOptions& opts) {
    DynSepModel model;
    model.clear();
    if (collector.samples.size() < 2) return model;

    model.gnn.init_xavier(opts.emb_dim, opts.n_msg_layers, opts.seed);
    const f64 lr = std::max(1e-4, opts.sgd_lr);
    const int epochs = std::max(1, opts.sgd_epochs);

    // Finite-difference / coordinate-free: train heads + light message W via
    // SGD on BCE(on) + MSE(budget). Keep MPNN body warm from Xavier; update
    // W_on/W_bud/b_* and a single scale on W_sep for stability on CPU.
    for (int ep = 0; ep < epochs; ++ep) {
        for (const DynSepSample& s : collector.samples) {
            std::array<f64, kSepFamilyCount> logits{};
            std::array<f64, kSepFamilyCount> bscale{};
            int hint = 0;
            dynsep_forward(model.gnn, s.graph, logits, bscale, hint);
            for (int k = 0; k < kSepFamilyCount; ++k) {
                const f64 p = sigmoid(logits[static_cast<std::size_t>(k)]);
                const f64 y = s.label_on[static_cast<std::size_t>(k)];
                const f64 g_on = (p - y);  // dBCE/dlogit
                // Approximate head gradient via embedding proxy: push bias /
                // W_on with mean sep projection feature.
                f64 feat_mean = 0.0;
                for (int d = 0; d < kDynSepSepFeatDim; ++d)
                    feat_mean += s.graph.seps[static_cast<std::size_t>(k)]
                                              [static_cast<std::size_t>(d)];
                feat_mean /= static_cast<f64>(kDynSepSepFeatDim);
                model.gnn.b_on -= lr * g_on;
                for (int d = 0; d < model.gnn.emb_dim; ++d) {
                    // Use state/sep raw features as surrogate activations.
                    const f64 hproxy =
                        (d < kDynSepSepFeatDim)
                            ? s.graph.seps[static_cast<std::size_t>(k)]
                                          [static_cast<std::size_t>(d)]
                            : feat_mean;
                    model.gnn.W_on[static_cast<std::size_t>(d)] -=
                        lr * g_on * hproxy;
                }
                const f64 yb = s.label_budget[static_cast<std::size_t>(k)];
                if (yb > 0.0) {
                    const f64 pred = (bscale[static_cast<std::size_t>(k)] - 0.25) /
                                     1.75;  // back to [0,1]
                    const f64 g_b = (pred - yb);
                    model.gnn.b_bud -= lr * g_b;
                    for (int d = 0; d < model.gnn.emb_dim; ++d) {
                        const f64 hproxy =
                            (d < kDynSepSepFeatDim)
                                ? s.graph.seps[static_cast<std::size_t>(k)]
                                              [static_cast<std::size_t>(d)]
                                : feat_mean;
                        model.gnn.W_bud[static_cast<std::size_t>(d)] -=
                            lr * g_b * hproxy;
                    }
                }
            }
        }
    }
    model.loaded = model.gnn.valid();
    return model;
}

bool save_dynsep_model(const std::string& path, const DynSepModel& model) {
    std::ofstream out(path);
    if (!out) return false;
    const auto& g = model.gnn;
    out << "SOR_DYNSEP 1\n";
    out << "emb_dim " << g.emb_dim << "\n";
    out << "n_msg_layers " << g.n_msg_layers << "\n";
    out << "state_dim " << g.state_dim << "\n";
    out << "sep_dim " << g.sep_dim << "\n";
    out << "n_sep " << g.n_sep << "\n";
    out << "t_max " << g.t_max << "\n";
    out << "b_on " << g.b_on << "\n";
    out << "b_bud " << g.b_bud << "\n";
    out << "b_rounds " << g.b_rounds << "\n";
    write_vec(out, "W_state", g.W_state);
    write_vec(out, "b_state", g.b_state);
    write_vec(out, "W_sep", g.W_sep);
    write_vec(out, "b_sep", g.b_sep);
    write_vec(out, "W_delta", g.W_delta);
    write_vec(out, "b_delta", g.b_delta);
    write_vec(out, "W_msg", g.W_msg);
    write_vec(out, "W_self", g.W_self);
    write_vec(out, "b_upd", g.b_upd);
    write_vec(out, "W_on", g.W_on);
    write_vec(out, "W_bud", g.W_bud);
    write_vec(out, "W_rounds", g.W_rounds);
    return static_cast<bool>(out);
}

bool load_dynsep_model(const std::string& path, DynSepModel& model) {
    model.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string tag;
    int ver = 0;
    if (!(in >> tag >> ver) || tag != "SOR_DYNSEP" || ver != 1) return false;
    auto& g = model.gnn;
    g.clear();
    std::string key;
    while (in >> key) {
        if (key == "emb_dim")
            in >> g.emb_dim;
        else if (key == "n_msg_layers")
            in >> g.n_msg_layers;
        else if (key == "state_dim")
            in >> g.state_dim;
        else if (key == "sep_dim")
            in >> g.sep_dim;
        else if (key == "n_sep")
            in >> g.n_sep;
        else if (key == "t_max")
            in >> g.t_max;
        else if (key == "b_on")
            in >> g.b_on;
        else if (key == "b_bud")
            in >> g.b_bud;
        else if (key == "b_rounds")
            in >> g.b_rounds;
        else if (key == "W_state") {
            if (!read_vec(in, g.W_state)) return false;
        } else if (key == "b_state") {
            if (!read_vec(in, g.b_state)) return false;
        } else if (key == "W_sep") {
            if (!read_vec(in, g.W_sep)) return false;
        } else if (key == "b_sep") {
            if (!read_vec(in, g.b_sep)) return false;
        } else if (key == "W_delta") {
            if (!read_vec(in, g.W_delta)) return false;
        } else if (key == "b_delta") {
            if (!read_vec(in, g.b_delta)) return false;
        } else if (key == "W_msg") {
            if (!read_vec(in, g.W_msg)) return false;
        } else if (key == "W_self") {
            if (!read_vec(in, g.W_self)) return false;
        } else if (key == "b_upd") {
            if (!read_vec(in, g.b_upd)) return false;
        } else if (key == "W_on") {
            if (!read_vec(in, g.W_on)) return false;
        } else if (key == "W_bud") {
            if (!read_vec(in, g.W_bud)) return false;
        } else if (key == "W_rounds") {
            if (!read_vec(in, g.W_rounds)) return false;
        } else {
            return false;
        }
    }
    if (g.state_dim != kDynSepStateFeatDim || g.sep_dim != kDynSepSepFeatDim ||
        g.n_sep != kSepFamilyCount)
        return false;
    if (!g.valid()) return false;
    model.loaded = true;
    return true;
}

DynSepController::DynSepController(const DynSepOptions& opts) : opts_(opts) {
    collector_.max_samples = opts_.collect_max_samples;
    if (!opts_.model_path.empty()) {
        DynSepModel m;
        if (load_dynsep_model(opts_.model_path, m)) set_model(std::move(m));
    }
}

void DynSepController::set_model(DynSepModel model) {
    model_ = std::move(model);
    diag_.model_loaded = model_.loaded && model_.gnn.valid();
}

void DynSepController::update_options(const DynSepOptions& opts) {
    opts_ = opts;
    collector_.max_samples = opts_.collect_max_samples;
}

bool DynSepController::allowed(SepFamily f, const DynSepRoundInput& in) const {
    switch (f) {
    case SepFamily::Gmi:
        return opts_.allow_gmi && in.have_basis;
    case SepFamily::Mir:
        return opts_.allow_mir || in.force_mir;
    case SepFamily::Cover:
        return opts_.allow_cover || in.force_cover;
    case SepFamily::Clique:
        return (opts_.allow_clique || in.force_clique) && in.have_conflict_graph;
    case SepFamily::ImpliedBound:
        return (opts_.allow_ib || in.force_ib) && in.have_conflict_graph;
    case SepFamily::ZeroHalf:
        return opts_.allow_zerohalf || in.force_zerohalf;
    case SepFamily::FlowCover:
        return opts_.allow_flowcover || in.force_flowcover;
    case SepFamily::Count:
        break;
    }
    return false;
}

f64 DynSepController::ucb(SepFamily f) const {
    const auto i = static_cast<std::size_t>(f);
    if (plays_[i] == 0) return std::numeric_limits<f64>::infinity();
    const f64 mean = reward_sum_[i] / static_cast<f64>(plays_[i]);
    const f64 bonus =
        opts_.ucb_c *
        std::sqrt(std::log(static_cast<f64>(std::max<std::uint64_t>(total_plays_, 1))) /
                  static_cast<f64>(plays_[i]));
    return mean + bonus;
}

void DynSepController::build_graph(const DynSepRoundInput& in,
                                   DynSepRoundGraph& g) const {
    fill_dynsep_state_feats(in, opts_, g.state);
    g.delta.fill(0.0);
    if (have_last_graph_) {
        for (int i = 0; i < kDynSepStateFeatDim; ++i)
            g.delta[static_cast<std::size_t>(i)] =
                g.state[static_cast<std::size_t>(i)] -
                prev_state_[static_cast<std::size_t>(i)];
    } else {
        g.delta = g.state;  // first round: full state as "incremental"
    }
    for (int i = 0; i < kSepFamilyCount; ++i) {
        const SepFamily f = static_cast<SepFamily>(i);
        const auto ii = static_cast<std::size_t>(i);
        const f64 mean_r =
            plays_[ii] > 0 ? reward_sum_[ii] / static_cast<f64>(plays_[ii])
                           : 0.0;
        const f64 plays_n =
            std::min(1.0, static_cast<f64>(plays_[ii]) / 20.0);
        const f64 cuts_n = std::min(
            1.0, static_cast<f64>(diag_.cuts_from[ii]) / 100.0);
        fill_dynsep_sep_feats(f, in, opts_, mean_r, plays_n, cuts_n,
                              allowed(f, in), g.seps[ii]);
    }
}

DynSepDecision DynSepController::decide_ucb(const DynSepRoundInput& in) {
    DynSepDecision d;
    ++diag_.ucb_decisions;

    if (opts_.always_gmi && allowed(SepFamily::Gmi, in))
        set_run(d, SepFamily::Gmi, budget_for(opts_, SepFamily::Gmi));
    if (opts_.always_ib && allowed(SepFamily::ImpliedBound, in))
        set_run(d, SepFamily::ImpliedBound,
                budget_for(opts_, SepFamily::ImpliedBound));

    if (in.force_mir && allowed(SepFamily::Mir, in))
        set_run(d, SepFamily::Mir, budget_for(opts_, SepFamily::Mir));
    if (in.force_cover && allowed(SepFamily::Cover, in))
        set_run(d, SepFamily::Cover, budget_for(opts_, SepFamily::Cover));
    if (in.force_clique && allowed(SepFamily::Clique, in))
        set_run(d, SepFamily::Clique, budget_for(opts_, SepFamily::Clique));
    if (in.force_zerohalf && allowed(SepFamily::ZeroHalf, in))
        set_run(d, SepFamily::ZeroHalf,
                budget_for(opts_, SepFamily::ZeroHalf));
    if (in.force_flowcover && allowed(SepFamily::FlowCover, in))
        set_run(d, SepFamily::FlowCover,
                budget_for(opts_, SepFamily::FlowCover));

    const bool deep = !in.at_root && in.depth > opts_.max_depth_optional;
    if (!deep) {
        std::vector<std::pair<f64, SepFamily>> ranked;
        const SepFamily optional[] = {SepFamily::ZeroHalf, SepFamily::FlowCover,
                                      SepFamily::Mir, SepFamily::Cover,
                                      SepFamily::Clique};
        for (const SepFamily f : optional) {
            if (!allowed(f, in)) continue;
            if (decision_runs(d, f)) continue;
            f64 score = ucb(f);
            if (in.last_bound_gain_rel < 1e-5 &&
                (f == SepFamily::ZeroHalf || f == SepFamily::FlowCover) &&
                plays_[static_cast<std::size_t>(f)] < 3)
                score = std::numeric_limits<f64>::infinity();
            ranked.emplace_back(score, f);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return static_cast<int>(a.second) <
                             static_cast<int>(b.second);
                  });

        int taken = 0;
        for (const auto& [sc, f] : ranked) {
            if (taken >= opts_.max_optional_arms) break;
            const auto i = static_cast<std::size_t>(f);
            if (plays_[i] >= 4 &&
                reward_sum_[i] / static_cast<f64>(plays_[i]) < 1e-6)
                continue;
            set_run(d, f, budget_for(opts_, f));
            ++taken;
        }
    }
    diag_.arms_scheduled += static_cast<std::uint64_t>(count_scheduled(d));
    DynSepRoundGraph graph;
    build_graph(in, graph);
    last_graph_ = graph;
    last_decision_ = d;
    have_last_graph_ = true;
    prev_state_ = graph.state;
    return d;
}

DynSepDecision DynSepController::decide_gnn(const DynSepRoundInput& in) {
    DynSepDecision d;
    ++diag_.gnn_decisions;
    diag_.used_gnn = true;

    DynSepRoundGraph graph;
    build_graph(in, graph);
    std::array<f64, kSepFamilyCount> logits{};
    std::array<f64, kSepFamilyCount> bscale{};
    int hint = 0;
    dynsep_forward(model_.gnn, graph, logits, bscale, hint);
    d.max_rounds_hint = hint;

    // Forced / always arms.
    if (opts_.always_gmi && allowed(SepFamily::Gmi, in))
        set_run(d, SepFamily::Gmi,
                std::max(1, static_cast<int>(std::lround(
                                budget_for(opts_, SepFamily::Gmi) *
                                bscale[static_cast<std::size_t>(SepFamily::Gmi)]))));
    if (opts_.always_ib && allowed(SepFamily::ImpliedBound, in))
        set_run(d, SepFamily::ImpliedBound,
                std::max(1, static_cast<int>(std::lround(
                                budget_for(opts_, SepFamily::ImpliedBound) *
                                bscale[static_cast<std::size_t>(
                                    SepFamily::ImpliedBound)]))));

    if (in.force_mir && allowed(SepFamily::Mir, in))
        set_run(d, SepFamily::Mir, budget_for(opts_, SepFamily::Mir));
    if (in.force_cover && allowed(SepFamily::Cover, in))
        set_run(d, SepFamily::Cover, budget_for(opts_, SepFamily::Cover));
    if (in.force_clique && allowed(SepFamily::Clique, in))
        set_run(d, SepFamily::Clique, budget_for(opts_, SepFamily::Clique));
    if (in.force_zerohalf && allowed(SepFamily::ZeroHalf, in))
        set_run(d, SepFamily::ZeroHalf,
                budget_for(opts_, SepFamily::ZeroHalf));
    if (in.force_flowcover && allowed(SepFamily::FlowCover, in))
        set_run(d, SepFamily::FlowCover,
                budget_for(opts_, SepFamily::FlowCover));

    const bool deep = !in.at_root && in.depth > opts_.max_depth_optional;
    if (deep) {
        last_graph_ = graph;
        last_decision_ = d;
        have_last_graph_ = true;
        prev_state_ = graph.state;
        diag_.arms_scheduled += static_cast<std::uint64_t>(count_scheduled(d));
        return d;
    }

    // Rank optional arms by sigmoid(logit); take up to max_optional_arms above
    // threshold (paper multi-label activation; budgets scaled).
    std::vector<std::pair<f64, SepFamily>> ranked;
    const SepFamily optional[] = {SepFamily::ZeroHalf, SepFamily::FlowCover,
                                  SepFamily::Mir, SepFamily::Cover,
                                  SepFamily::Clique};
    for (const SepFamily f : optional) {
        if (!allowed(f, in)) continue;
        if (decision_runs(d, f)) continue;
        const f64 p = sigmoid(logits[static_cast<std::size_t>(f)]);
        if (p < opts_.gnn_on_threshold) continue;
        ranked.emplace_back(p, f);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return static_cast<int>(a.second) < static_cast<int>(b.second);
              });
    int taken = 0;
    for (const auto& [p, f] : ranked) {
        if (taken >= opts_.max_optional_arms) break;
        const auto family_index = static_cast<std::size_t>(f);
        if (family_index >= bscale.size()) continue;
        const int bud = std::max(
            1, static_cast<int>(std::lround(
                   budget_for(opts_, f) *
                   bscale[family_index])));
        set_run(d, f, bud);
        ++taken;
    }

    // If GNN selected nothing optional and bound stalled, fall back to one UCB
    // explore arm so cold models still learn.
    if (taken == 0 && in.last_bound_gain_rel < 1e-5) {
        for (const SepFamily f : optional) {
            if (!allowed(f, in) || decision_runs(d, f)) continue;
            set_run(d, f, budget_for(opts_, f));
            break;
        }
    }

    last_graph_ = graph;
    last_decision_ = d;
    have_last_graph_ = true;
    prev_state_ = graph.state;
    diag_.arms_scheduled += static_cast<std::uint64_t>(count_scheduled(d));
    return d;
}

DynSepDecision DynSepController::decide(const DynSepRoundInput& in) {
    DynSepDecision d;
    if (!opts_.enabled) {
        if (in.have_basis) set_run(d, SepFamily::Gmi, opts_.budget_gmi);
        if (in.force_mir) set_run(d, SepFamily::Mir, opts_.budget_mir);
        if (in.force_cover) set_run(d, SepFamily::Cover, opts_.budget_cover);
        if (in.force_clique && in.have_conflict_graph)
            set_run(d, SepFamily::Clique, opts_.budget_clique);
        if (in.force_ib && in.have_conflict_graph)
            set_run(d, SepFamily::ImpliedBound, opts_.budget_ib);
        if (in.force_zerohalf)
            set_run(d, SepFamily::ZeroHalf, opts_.budget_zerohalf);
        if (in.force_flowcover)
            set_run(d, SepFamily::FlowCover, opts_.budget_flowcover);
        ++diag_.decisions;
        last_decision_ = d;
        return d;
    }

    ++diag_.decisions;
    if (dynsep_prefer_gnn(model_, opts_))
        d = decide_gnn(in);
    else
        d = decide_ucb(in);
    last_decision_ = d;
    return d;
}

void DynSepController::observe(const DynSepObserve& obs) {
    const auto i = static_cast<std::size_t>(obs.family);
    if (i >= static_cast<std::size_t>(kSepFamilyCount)) return;
    ++plays_[i];
    ++total_plays_;
    ++diag_.plays[i];
    diag_.cuts_from[i] += static_cast<std::uint64_t>(
        std::max(0, obs.cuts_selected));
    f64 r = obs.total_efficacy;
    if (obs.cuts_selected > 0)
        r += 0.1 * static_cast<f64>(obs.cuts_selected);
    else if (obs.cuts_generated > 0)
        r += 1e-4;
    reward_sum_[i] += r;
    diag_.reward_sum[i] += r;
}

void DynSepController::observe_selected_names(
    const std::vector<std::string>& names, f64 mean_efficacy) {
    std::array<int, kSepFamilyCount> counts{};
    for (const auto& n : names) {
        SepFamily f = SepFamily::Gmi;
        if (n.rfind("MIR_", 0) == 0) f = SepFamily::Mir;
        else if (n.rfind("COV_", 0) == 0 || n.rfind("COVPC_", 0) == 0 ||
                 n.rfind("COVGNS_", 0) == 0)
            f = SepFamily::Cover;
        else if (n.rfind("CLQ_", 0) == 0) f = SepFamily::Clique;
        else if (n.rfind("VUB_", 0) == 0) f = SepFamily::ImpliedBound;
        else if (n.rfind("ZH_", 0) == 0) f = SepFamily::ZeroHalf;
        else if (n.rfind("FC_", 0) == 0) f = SepFamily::FlowCover;
        else f = SepFamily::Gmi;
        ++counts[static_cast<std::size_t>(f)];
    }
    for (int i = 0; i < kSepFamilyCount; ++i) {
        if (counts[static_cast<std::size_t>(i)] == 0) continue;
        DynSepObserve o;
        o.family = static_cast<SepFamily>(i);
        o.cuts_selected = counts[static_cast<std::size_t>(i)];
        o.cuts_generated = o.cuts_selected;
        o.total_efficacy =
            mean_efficacy * static_cast<f64>(o.cuts_selected);
        observe(o);
    }
}

void DynSepController::collect_round_labels(const DynSepDecision& decided) {
    if (!opts_.collect_labels || !have_last_graph_) return;
    DynSepSample s;
    s.graph = last_graph_;
    s.label_on.fill(0.0);
    s.label_budget.fill(0.0);
    for (int i = 0; i < kSepFamilyCount; ++i) {
        const SepFamily f = static_cast<SepFamily>(i);
        const auto ii = static_cast<std::size_t>(i);
        // Helped if this round scheduled it and it has positive reward mass
        // relative to recent plays, or simply if cuts came from it this round.
        const bool ran = decision_runs(decided, f);
        const bool helped =
            ran && (diag_.cuts_from[ii] > 0 ||
                    (plays_[ii] > 0 &&
                     reward_sum_[ii] / static_cast<f64>(plays_[ii]) > 1e-6));
        s.label_on[ii] = helped ? 1.0 : (ran ? 0.0 : 0.0);
        if (ran) {
            const int base = budget_for(opts_, f);
            int used = 0;
            switch (f) {
            case SepFamily::Gmi: used = decided.budget_gmi; break;
            case SepFamily::Mir: used = decided.budget_mir; break;
            case SepFamily::Cover: used = decided.budget_cover; break;
            case SepFamily::Clique: used = decided.budget_clique; break;
            case SepFamily::ImpliedBound: used = decided.budget_ib; break;
            case SepFamily::ZeroHalf: used = decided.budget_zerohalf; break;
            case SepFamily::FlowCover: used = decided.budget_flowcover; break;
            case SepFamily::Count: break;
            }
            if (base > 0)
                s.label_budget[ii] =
                    std::min(1.0, static_cast<f64>(used) / static_cast<f64>(base));
        }
        // Prefer positive labels from families that actually produced selected
        // cuts in the latest observe batch (cuts_from increased).
        if (diag_.cuts_from[ii] > 0 && ran) s.label_on[ii] = 1.0;
    }
    collector_.add(std::move(s));
}

}  // namespace sor::search
