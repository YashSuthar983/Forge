#include "sor/search/hgtsm.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <utility>

namespace sor::search {
namespace {

inline f64 soft_threshold(f64 z, f64 lam) {
    if (z > lam) return z - lam;
    if (z < -lam) return z + lam;
    return 0.0;
}

inline f64 relu(f64 v) { return v > 0.0 ? v : 0.0; }

inline f64 tanh01(f64 v) { return 0.5 * std::tanh(v) + 0.5; }

std::uint64_t splitmix64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

f64 xavier(std::uint64_t& rng, int fan_in, int fan_out) {
    const f64 a = std::sqrt(6.0 / static_cast<f64>(fan_in + fan_out));
    const f64 u = static_cast<f64>(splitmix64(rng) >> 11) / static_cast<f64>(1ull << 53);
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

[[maybe_unused]] void matvec_add(const std::vector<f64>& W, const f64* x, f64* y,
                                 int rows, int cols, f64 scale = 1.0) {
    for (int r = 0; r < rows; ++r) {
        f64 s = 0.0;
        const f64* wr = W.data() + static_cast<std::size_t>(r * cols);
        for (int c = 0; c < cols; ++c) s += wr[c] * x[c];
        y[r] += scale * s;
    }
}

void add_bias(f64* y, const std::vector<f64>& b, int n) {
    for (int i = 0; i < n; ++i) y[i] += b[static_cast<std::size_t>(i)];
}

}  // namespace

void HgtsmGraphWeights::clear() {
    emb_dim = kHgtsmEmbDim;
    n_msg_layers = kHgtsmMsgLayers;
    seq_kind = 0;
    W_var.clear();
    b_var.clear();
    W_con.clear();
    b_con.clear();
    W_cut.clear();
    b_cut.clear();
    W_msg.clear();
    W_self.clear();
    b_upd.clear();
    W_qkv.clear();
    W_o.clear();
    W_ff1.clear();
    W_ff2.clear();
    b_ff1.clear();
    b_ff2.clear();
    W_gru.clear();
    b_gru.clear();
    W_score.clear();
    b_score = 0.0;
    W_ratio.clear();
    b_ratio = 0.0;
}

void HgtsmGraphWeights::init_xavier(int emb, int n_layers, int sk,
                                   std::uint64_t seed) {
    clear();
    emb_dim = emb;
    n_msg_layers = n_layers;
    seq_kind = sk;
    std::uint64_t rng = seed ? seed : 0x484754534dull;

    auto fill_mat = [&](std::vector<f64>& M, int rows, int cols) {
        M.resize(static_cast<std::size_t>(rows * cols));
        for (f64& v : M) v = xavier(rng, cols, rows);
    };
    auto fill_bias = [&](std::vector<f64>& b, int n) {
        b.assign(static_cast<std::size_t>(n), 0.0);
    };

    fill_mat(W_var, emb, kVarNodeFeatureDim);
    fill_bias(b_var, emb);
    fill_mat(W_con, emb, kConNodeFeatureDim);
    fill_bias(b_con, emb);
    fill_mat(W_cut, emb, kCutNodeFeatureDim);
    fill_bias(b_cut, emb);

    fill_mat(W_msg, 6 * n_layers * emb, emb);
    fill_mat(W_self, n_layers * 3 * emb, emb);
    fill_bias(b_upd, n_layers * 3 * emb);

    if (sk == 0) {
        fill_mat(W_qkv, 3 * emb, emb);
        fill_mat(W_o, emb, emb);
        fill_mat(W_ff1, emb, emb);
        fill_mat(W_ff2, emb, emb);
        fill_bias(b_ff1, emb);
        fill_bias(b_ff2, emb);
    } else {
        fill_mat(W_gru, 3 * emb, 2 * emb);
        fill_bias(b_gru, 3 * emb);
    }

    fill_mat(W_score, 1, emb);
    // flatten to vector of emb
    W_score.resize(static_cast<std::size_t>(emb));
    for (f64& v : W_score) v = xavier(rng, emb, 1);
    b_score = 0.0;
    fill_mat(W_ratio, 1, emb);
    W_ratio.resize(static_cast<std::size_t>(emb));
    for (f64& v : W_ratio) v = xavier(rng, emb, 1);
    b_ratio = 0.0;
}

bool HgtsmGraphWeights::valid() const {
    if (emb_dim <= 0 || n_msg_layers <= 0) return false;
    const int e = emb_dim;
    if (static_cast<int>(W_var.size()) != e * kVarNodeFeatureDim) return false;
    if (static_cast<int>(W_con.size()) != e * kConNodeFeatureDim) return false;
    if (static_cast<int>(W_cut.size()) != e * kCutNodeFeatureDim) return false;
    if (static_cast<int>(W_msg.size()) != 6 * n_msg_layers * e * e) return false;
    if (static_cast<int>(W_self.size()) != n_msg_layers * 3 * e * e) return false;
    if (static_cast<int>(b_upd.size()) != n_msg_layers * 3 * e) return false;
    if (static_cast<int>(W_score.size()) != e) return false;
    if (static_cast<int>(W_ratio.size()) != e) return false;
    if (seq_kind == 0) {
        if (static_cast<int>(W_qkv.size()) != 3 * e * e) return false;
        if (static_cast<int>(W_o.size()) != e * e) return false;
        if (static_cast<int>(W_ff1.size()) != e * e) return false;
        if (static_cast<int>(W_ff2.size()) != e * e) return false;
    } else {
        if (static_cast<int>(W_gru.size()) != 3 * e * (2 * e)) return false;
        if (static_cast<int>(b_gru.size()) != 3 * e) return false;
    }
    return true;
}

void HgtsmModel::clear() {
    joint_dim = kHgtsmJointDim;
    quadratic = false;
    intercept = 0.0;
    terms.clear();
    loaded = false;
    has_graph = false;
    graph.clear();
}

void HgtsmModel::pack_joint(const CutFeatureVec& cut, const HgtsmLpStateVec& lp,
                            HgtsmJointVec& out) {
    for (int i = 0; i < kCutFeatureDim; ++i)
        out[static_cast<std::size_t>(i)] = cut[static_cast<std::size_t>(i)];
    for (int i = 0; i < kHgtsmLpStateDim; ++i)
        out[static_cast<std::size_t>(kCutFeatureDim + i)] =
            lp[static_cast<std::size_t>(i)];
}

f64 HgtsmModel::builtin_score(const CutFeatureVec& cut,
                              const HgtsmLpStateVec& lp) {
    const f64 efficacy = cut[0];
    const f64 obj_par = std::fabs(cut[1]);
    const f64 density = cut[2];
    const f64 int_sup = cut[3];
    const f64 frac_share = lp[0];
    return efficacy + 0.15 * obj_par + 0.05 * int_sup - 0.25 * density +
           0.05 * frac_share;
}

f64 HgtsmModel::predict_linear(const CutFeatureVec& cut,
                               const HgtsmLpStateVec& lp) const {
    if (!loaded || terms.empty()) return builtin_score(cut, lp);
    HgtsmJointVec j{};
    pack_joint(cut, lp, j);
    f64 y = intercept;
    for (const HgtsmTerm& t : terms) {
        if (t.i < 0 || t.i >= kHgtsmJointDim) continue;
        f64 v = j[static_cast<std::size_t>(t.i)];
        if (t.j >= 0) {
            if (t.j >= kHgtsmJointDim) continue;
            v *= j[static_cast<std::size_t>(t.j)];
        }
        y += t.coef * v;
    }
    return y;
}

void HgtsmCollector::add(const CutFeatureVec& cut, const HgtsmLpStateVec& lp,
                         f64 label) {
    if (!std::isfinite(label)) return;
    if (samples.size() >= max_samples) return;
    HgtsmSample s;
    HgtsmModel::pack_joint(cut, lp, s.feats);
    s.label = label;
    samples.push_back(s);
}

void HgtsmCollector::add_round(TripartiteGraphSnapshot graph,
                               std::vector<f64> labels) {
    if (rounds.size() >= max_rounds) return;
    if (labels.size() != graph.cuts.size()) return;
    HgtsmRoundSample r;
    r.graph = std::move(graph);
    r.labels = std::move(labels);
    rounds.push_back(std::move(r));
}

void fill_hgtsm_lp_state(Index n_cols, Index n_int, Index n_frac,
                         f64 mean_frac, f64 gap_rel, int depth,
                         std::size_t pool_size, f64 last_bound_gain,
                         HgtsmLpStateVec& out) {
    out.fill(0.0);
    const f64 nc = static_cast<f64>(std::max(Index{1}, n_cols));
    const f64 ni = static_cast<f64>(std::max(Index{1}, n_int));
    out[0] = static_cast<f64>(n_frac) / ni;
    out[1] = mean_frac;
    out[2] = std::isfinite(gap_rel) ? std::min(1.0, std::max(0.0, gap_rel)) : 1.0;
    out[3] = std::min(1.0, static_cast<f64>(std::max(0, depth)) / 64.0);
    out[4] = std::min(1.0, static_cast<f64>(pool_size) / 500.0);
    out[5] = std::min(1.0, std::max(0.0, last_bound_gain));
    out[6] = ni / nc;
    out[7] = 1.0;
}

f64 hgtsm_score(const HgtsmModel& model, const CutFeatureVec& cut,
                const HgtsmLpStateVec& lp) {
    return model.predict_linear(cut, lp);
}

bool hgtsm_prefer_graph(const HgtsmModel& model, const HgtsmOptions& opts) {
    return model.loaded && model.has_graph && model.graph.valid() &&
           !opts.prefer_linear;
}

namespace {

enum EdgeType : int {
    kVarCon = 0,
    kConVar = 1,
    kVarCut = 2,
    kCutVar = 3,
    kConCut = 4,
    kCutCon = 5,
};

const f64* msg_W(const HgtsmGraphWeights& g, int layer, int etype) {
    const int e = g.emb_dim;
    const std::size_t off =
        static_cast<std::size_t>(((layer * 6) + etype) * e * e);
    return g.W_msg.data() + off;
}

const f64* self_W(const HgtsmGraphWeights& g, int layer, int ntype) {
    const int e = g.emb_dim;
    const std::size_t off =
        static_cast<std::size_t>(((layer * 3) + ntype) * e * e);
    return g.W_self.data() + off;
}

const f64* upd_b(const HgtsmGraphWeights& g, int layer, int ntype) {
    const int e = g.emb_dim;
    return g.b_upd.data() +
           static_cast<std::size_t>(((layer * 3) + ntype) * e);
}

void project_nodes(const HgtsmGraphWeights& g,
                   const TripartiteGraphSnapshot& snap,
                   std::vector<f64>& H_var, std::vector<f64>& H_con,
                   std::vector<f64>& H_cut) {
    const int e = g.emb_dim;
    const std::size_t nv = snap.bipartite.vars.size();
    const std::size_t nc = snap.bipartite.cons.size();
    const std::size_t nk = snap.cuts.size();
    H_var.assign(nv * static_cast<std::size_t>(e), 0.0);
    H_con.assign(nc * static_cast<std::size_t>(e), 0.0);
    H_cut.assign(nk * static_cast<std::size_t>(e), 0.0);

    for (std::size_t j = 0; j < nv; ++j) {
        matvec(g.W_var, snap.bipartite.vars[j].data(),
               H_var.data() + j * static_cast<std::size_t>(e), e,
               kVarNodeFeatureDim);
        add_bias(H_var.data() + j * static_cast<std::size_t>(e), g.b_var, e);
    }
    for (std::size_t i = 0; i < nc; ++i) {
        matvec(g.W_con, snap.bipartite.cons[i].data(),
               H_con.data() + i * static_cast<std::size_t>(e), e,
               kConNodeFeatureDim);
        add_bias(H_con.data() + i * static_cast<std::size_t>(e), g.b_con, e);
    }
    for (std::size_t t = 0; t < nk; ++t) {
        matvec(g.W_cut, snap.cuts[t].data(),
               H_cut.data() + t * static_cast<std::size_t>(e), e,
               kCutNodeFeatureDim);
        add_bias(H_cut.data() + t * static_cast<std::size_t>(e), g.b_cut, e);
    }
}

void message_layer(const HgtsmGraphWeights& g, int layer,
                   const TripartiteGraphSnapshot& snap,
                   std::vector<f64>& H_var, std::vector<f64>& H_con,
                   std::vector<f64>& H_cut) {
    const int e = g.emb_dim;
    const std::size_t nv = snap.bipartite.vars.size();
    const std::size_t nc = snap.bipartite.cons.size();
    const std::size_t nk = snap.cuts.size();

    std::vector<f64> Acc_var(nv * static_cast<std::size_t>(e), 0.0);
    std::vector<f64> Acc_con(nc * static_cast<std::size_t>(e), 0.0);
    std::vector<f64> Acc_cut(nk * static_cast<std::size_t>(e), 0.0);
    std::vector<f64> Cnt_var(nv, 0.0), Cnt_con(nc, 0.0), Cnt_cut(nk, 0.0);
    std::vector<f64> msg(static_cast<std::size_t>(e), 0.0);

    auto accum = [&](EdgeType et, const f64* h_src, f64* acc, f64& cnt) {
        const std::vector<f64> W(msg_W(g, layer, et),
                                 msg_W(g, layer, et) + e * e);
        matvec(W, h_src, msg.data(), e, e);
        for (int d = 0; d < e; ++d) acc[d] += msg[static_cast<std::size_t>(d)];
        cnt += 1.0;
    };

    // Var ↔ Con (bidirectional).
    for (const GraphEdge& ed : snap.bipartite.edges) {
        if (ed.var < 0 || static_cast<std::size_t>(ed.var) >= nv) continue;
        if (ed.con < 0 || static_cast<std::size_t>(ed.con) >= nc) continue;
        const std::size_t j = static_cast<std::size_t>(ed.var);
        const std::size_t i = static_cast<std::size_t>(ed.con);
        accum(kVarCon, H_var.data() + j * static_cast<std::size_t>(e),
              Acc_con.data() + i * static_cast<std::size_t>(e), Cnt_con[i]);
        accum(kConVar, H_con.data() + i * static_cast<std::size_t>(e),
              Acc_var.data() + j * static_cast<std::size_t>(e), Cnt_var[j]);
    }
    // Var ↔ Cut.
    for (const GraphEdge& ed : snap.cut_edges) {
        if (ed.var < 0 || static_cast<std::size_t>(ed.var) >= nv) continue;
        if (ed.con < 0 || static_cast<std::size_t>(ed.con) >= nk) continue;
        const std::size_t j = static_cast<std::size_t>(ed.var);
        const std::size_t t = static_cast<std::size_t>(ed.con);
        accum(kVarCut, H_var.data() + j * static_cast<std::size_t>(e),
              Acc_cut.data() + t * static_cast<std::size_t>(e), Cnt_cut[t]);
        accum(kCutVar, H_cut.data() + t * static_cast<std::size_t>(e),
              Acc_var.data() + j * static_cast<std::size_t>(e), Cnt_var[j]);
    }
    // Cons ↔ Cut similarity (paper meta-relations similar_to_*).
    for (const GraphEdge& ed : snap.cut_con_edges) {
        if (ed.var < 0 || static_cast<std::size_t>(ed.var) >= nc) continue;
        if (ed.con < 0 || static_cast<std::size_t>(ed.con) >= nk) continue;
        const std::size_t i = static_cast<std::size_t>(ed.var);  // con
        const std::size_t t = static_cast<std::size_t>(ed.con);  // cut
        accum(kConCut, H_con.data() + i * static_cast<std::size_t>(e),
              Acc_cut.data() + t * static_cast<std::size_t>(e), Cnt_cut[t]);
        accum(kCutCon, H_cut.data() + t * static_cast<std::size_t>(e),
              Acc_con.data() + i * static_cast<std::size_t>(e), Cnt_con[i]);
    }

    auto update = [&](std::vector<f64>& H, std::vector<f64>& Acc,
                      std::vector<f64>& Cnt, int ntype, std::size_t n) {
        const std::vector<f64> Ws(self_W(g, layer, ntype),
                                  self_W(g, layer, ntype) + e * e);
        const f64* bb = upd_b(g, layer, ntype);
        for (std::size_t i = 0; i < n; ++i) {
            f64* h = H.data() + i * static_cast<std::size_t>(e);
            f64* a = Acc.data() + i * static_cast<std::size_t>(e);
            const f64 c = std::max(1.0, Cnt[i]);
            std::vector<f64> self(static_cast<std::size_t>(e), 0.0);
            matvec(Ws, h, self.data(), e, e);
            for (int d = 0; d < e; ++d) {
                const f64 agg = a[d] / c;
                h[d] = relu(self[static_cast<std::size_t>(d)] + agg + bb[d]);
            }
        }
    };
    update(H_var, Acc_var, Cnt_var, 0, nv);
    update(H_con, Acc_con, Cnt_con, 1, nc);
    update(H_cut, Acc_cut, Cnt_cut, 2, nk);
}

void transformer_lite(const HgtsmGraphWeights& g, std::vector<f64>& H_cut,
                      std::size_t nk) {
    const int e = g.emb_dim;
    if (nk == 0) return;
    std::vector<f64> Q(nk * static_cast<std::size_t>(e));
    std::vector<f64> K(nk * static_cast<std::size_t>(e));
    std::vector<f64> V(nk * static_cast<std::size_t>(e));
    const f64* Wq = g.W_qkv.data();
    const f64* Wk = g.W_qkv.data() + static_cast<std::size_t>(e * e);
    const f64* Wv = g.W_qkv.data() + static_cast<std::size_t>(2 * e * e);
    std::vector<f64> Wq_v(Wq, Wq + e * e);
    std::vector<f64> Wk_v(Wk, Wk + e * e);
    std::vector<f64> Wv_v(Wv, Wv + e * e);
    for (std::size_t t = 0; t < nk; ++t) {
        const f64* h = H_cut.data() + t * static_cast<std::size_t>(e);
        matvec(Wq_v, h, Q.data() + t * static_cast<std::size_t>(e), e, e);
        matvec(Wk_v, h, K.data() + t * static_cast<std::size_t>(e), e, e);
        matvec(Wv_v, h, V.data() + t * static_cast<std::size_t>(e), e, e);
    }
    const f64 scale = 1.0 / std::sqrt(static_cast<f64>(e));
    std::vector<f64> Ctx(nk * static_cast<std::size_t>(e), 0.0);
    for (std::size_t i = 0; i < nk; ++i) {
        std::vector<f64> attn(nk, 0.0);
        f64 m = -std::numeric_limits<f64>::infinity();
        for (std::size_t j = 0; j < nk; ++j) {
            f64 dot = 0.0;
            for (int d = 0; d < e; ++d)
                dot += Q[i * static_cast<std::size_t>(e) +
                         static_cast<std::size_t>(d)] *
                       K[j * static_cast<std::size_t>(e) +
                         static_cast<std::size_t>(d)];
            attn[j] = dot * scale;
            m = std::max(m, attn[j]);
        }
        f64 z = 0.0;
        for (std::size_t j = 0; j < nk; ++j) {
            attn[j] = std::exp(attn[j] - m);
            z += attn[j];
        }
        if (z <= 0.0) z = 1.0;
        for (std::size_t j = 0; j < nk; ++j) {
            const f64 a = attn[j] / z;
            for (int d = 0; d < e; ++d)
                Ctx[i * static_cast<std::size_t>(e) +
                    static_cast<std::size_t>(d)] +=
                    a * V[j * static_cast<std::size_t>(e) +
                          static_cast<std::size_t>(d)];
        }
    }
    std::vector<f64> tmp(static_cast<std::size_t>(e));
    std::vector<f64> out(static_cast<std::size_t>(e));
    for (std::size_t i = 0; i < nk; ++i) {
        f64* h = H_cut.data() + i * static_cast<std::size_t>(e);
        matvec(g.W_o, Ctx.data() + i * static_cast<std::size_t>(e), tmp.data(),
               e, e);
        for (int d = 0; d < e; ++d)
            h[d] = relu(h[d] + tmp[static_cast<std::size_t>(d)]);
        matvec(g.W_ff1, h, tmp.data(), e, e);
        add_bias(tmp.data(), g.b_ff1, e);
        for (int d = 0; d < e; ++d)
            tmp[static_cast<std::size_t>(d)] =
                relu(tmp[static_cast<std::size_t>(d)]);
        matvec(g.W_ff2, tmp.data(), out.data(), e, e);
        add_bias(out.data(), g.b_ff2, e);
        for (int d = 0; d < e; ++d)
            h[d] = relu(h[d] + out[static_cast<std::size_t>(d)]);
    }
}

void gru_sequence(const HgtsmGraphWeights& g, std::vector<f64>& H_cut,
                  std::size_t nk) {
    // Permutation-invariant GRU: sort by L2 norm of embedding (content key),
    // run GRU, write hidden back in original order. No positional encodings.
    const int e = g.emb_dim;
    if (nk == 0) return;
    std::vector<std::pair<f64, std::size_t>> order;
    order.reserve(nk);
    for (std::size_t t = 0; t < nk; ++t) {
        f64 n2 = 0.0;
        const f64* h = H_cut.data() + t * static_cast<std::size_t>(e);
        for (int d = 0; d < e; ++d) n2 += h[d] * h[d];
        order.emplace_back(n2, t);
    }
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });

    std::vector<f64> hprev(static_cast<std::size_t>(e), 0.0);
    std::vector<f64> xh(static_cast<std::size_t>(2 * e));
    std::vector<f64> z(static_cast<std::size_t>(e));
    std::vector<f64> r(static_cast<std::size_t>(e));
    std::vector<f64> hn(static_cast<std::size_t>(e));
    std::vector<f64> out_store(nk * static_cast<std::size_t>(e));

    const f64* Wz = g.W_gru.data();
    const f64* Wr = g.W_gru.data() + static_cast<std::size_t>(e * 2 * e);
    const f64* Wh = g.W_gru.data() + static_cast<std::size_t>(2 * e * 2 * e);
    std::vector<f64> Wz_v(Wz, Wz + e * 2 * e);
    std::vector<f64> Wr_v(Wr, Wr + e * 2 * e);
    std::vector<f64> Wh_v(Wh, Wh + e * 2 * e);

    for (std::size_t step = 0; step < nk; ++step) {
        const std::size_t t = order[step].second;
        const f64* x = H_cut.data() + t * static_cast<std::size_t>(e);
        for (int d = 0; d < e; ++d) {
            xh[static_cast<std::size_t>(d)] = x[d];
            xh[static_cast<std::size_t>(e + d)] = hprev[static_cast<std::size_t>(d)];
        }
        matvec(Wz_v, xh.data(), z.data(), e, 2 * e);
        matvec(Wr_v, xh.data(), r.data(), e, 2 * e);
        for (int d = 0; d < e; ++d) {
            z[static_cast<std::size_t>(d)] =
                1.0 /
                (1.0 + std::exp(-(z[static_cast<std::size_t>(d)] +
                                  g.b_gru[static_cast<std::size_t>(d)])));
            r[static_cast<std::size_t>(d)] =
                1.0 /
                (1.0 + std::exp(-(r[static_cast<std::size_t>(d)] +
                                  g.b_gru[static_cast<std::size_t>(e + d)])));
        }
        for (int d = 0; d < e; ++d) {
            xh[static_cast<std::size_t>(d)] = x[d];
            xh[static_cast<std::size_t>(e + d)] =
                r[static_cast<std::size_t>(d)] *
                hprev[static_cast<std::size_t>(d)];
        }
        matvec(Wh_v, xh.data(), hn.data(), e, 2 * e);
        for (int d = 0; d < e; ++d) {
            const f64 cand =
                std::tanh(hn[static_cast<std::size_t>(d)] +
                          g.b_gru[static_cast<std::size_t>(2 * e + d)]);
            hprev[static_cast<std::size_t>(d)] =
                (1.0 - z[static_cast<std::size_t>(d)]) *
                    hprev[static_cast<std::size_t>(d)] +
                z[static_cast<std::size_t>(d)] * cand;
            out_store[t * static_cast<std::size_t>(e) +
                      static_cast<std::size_t>(d)] =
                hprev[static_cast<std::size_t>(d)];
        }
    }
    H_cut.swap(out_store);
}

void forward_graph(const HgtsmGraphWeights& g,
                   const TripartiteGraphSnapshot& snap,
                   std::vector<f64>& H_cut, f64* ratio_out) {
    std::vector<f64> H_var, H_con;
    project_nodes(g, snap, H_var, H_con, H_cut);
    for (int L = 0; L < g.n_msg_layers; ++L)
        message_layer(g, L, snap, H_var, H_con, H_cut);
    // Discard Vars/Cons; keep Cuts (paper §3.3).
    if (g.seq_kind == 0)
        transformer_lite(g, H_cut, snap.cuts.size());
    else
        gru_sequence(g, H_cut, snap.cuts.size());

    if (ratio_out) {
        const int e = g.emb_dim;
        const std::size_t nk = snap.cuts.size();
        std::vector<f64> pooled(static_cast<std::size_t>(e), 0.0);
        if (nk > 0) {
            for (std::size_t t = 0; t < nk; ++t)
                for (int d = 0; d < e; ++d)
                    pooled[static_cast<std::size_t>(d)] +=
                        H_cut[t * static_cast<std::size_t>(e) +
                              static_cast<std::size_t>(d)];
            for (int d = 0; d < e; ++d)
                pooled[static_cast<std::size_t>(d)] /= static_cast<f64>(nk);
        }
        f64 mu = g.b_ratio;
        for (int d = 0; d < e; ++d)
            mu += g.W_ratio[static_cast<std::size_t>(d)] *
                  pooled[static_cast<std::size_t>(d)];
        *ratio_out = tanh01(mu);
    }
}

f64 score_from_emb(const HgtsmGraphWeights& g, const f64* h) {
    f64 y = g.b_score;
    for (int d = 0; d < g.emb_dim; ++d)
        y += g.W_score[static_cast<std::size_t>(d)] * h[d];
    return y;
}

}  // namespace

void hgtsm_score_sequence(const HgtsmModel& model,
                          const TripartiteGraphSnapshot& graph,
                          std::vector<f64>& scores_out, f64* ratio_out) {
    scores_out.assign(graph.cuts.size(), 0.0);
    if (graph.cuts.empty()) {
        if (ratio_out) *ratio_out = 0.0;
        return;
    }
    if (!model.has_graph || !model.graph.valid()) {
        // Linear / builtin fallback per cut node features.
        HgtsmLpStateVec lp{};
        lp.fill(0.0);
        lp[7] = 1.0;
        for (std::size_t t = 0; t < graph.cuts.size(); ++t) {
            CutFeatureVec cf{};
            cf.fill(0.0);
            for (int i = 0; i < kCutNodeFeatureDim && i < kCutFeatureDim; ++i)
                cf[static_cast<std::size_t>(i)] =
                    graph.cuts[t][static_cast<std::size_t>(i)];
            scores_out[t] = model.predict_linear(cf, lp);
        }
        if (ratio_out) *ratio_out = 1.0;
        return;
    }
    std::vector<f64> H_cut;
    forward_graph(model.graph, graph, H_cut, ratio_out);
    const int e = model.graph.emb_dim;
    for (std::size_t t = 0; t < graph.cuts.size(); ++t)
        scores_out[t] =
            score_from_emb(model.graph,
                           H_cut.data() + t * static_cast<std::size_t>(e));
}

HgtsmModel fit_hgtsm_lasso(const std::vector<HgtsmSample>& samples,
                           const HgtsmFitOptions& opts) {
    HgtsmModel model;
    model.clear();
    if (samples.size() < 2) return model;

    const int d = kHgtsmJointDim;
    const std::size_t n = samples.size();
    std::vector<f64> y(n);
    f64 y_norm2 = 0.0;
    f64 intercept = 0.0;
    for (std::size_t r = 0; r < n; ++r) {
        y[r] = samples[r].label;
        y_norm2 += y[r] * y[r];
        intercept += y[r];
    }
    intercept /= static_cast<f64>(n);
    if (y_norm2 > 0.0) {
        const f64 inv = 1.0 / std::sqrt(y_norm2);
        for (f64& v : y) v *= inv;
        intercept *= inv;
    }

    std::vector<f64> beta(static_cast<std::size_t>(d), 0.0);
    std::vector<f64> residual(n);
    for (std::size_t r = 0; r < n; ++r) residual[r] = y[r] - intercept;

    std::vector<f64> col_norm2(static_cast<std::size_t>(d), 0.0);
    for (int j = 0; j < d; ++j) {
        f64 s = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            const f64 v = samples[r].feats[static_cast<std::size_t>(j)];
            s += v * v;
        }
        col_norm2[static_cast<std::size_t>(j)] = s;
    }

    const f64 lam = std::max(0.0, opts.lasso_lambda);
    for (int it = 0; it < opts.max_iter; ++it) {
        for (int j = 0; j < d; ++j) {
            const f64 cn = col_norm2[static_cast<std::size_t>(j)];
            if (cn <= 1e-18) continue;
            f64 rho = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                const f64 xij = samples[r].feats[static_cast<std::size_t>(j)];
                residual[r] += beta[static_cast<std::size_t>(j)] * xij;
                rho += xij * residual[r];
            }
            const f64 bj = soft_threshold(rho / cn, lam / cn);
            const f64 delta = bj - beta[static_cast<std::size_t>(j)];
            beta[static_cast<std::size_t>(j)] = bj;
            if (delta != 0.0) {
                for (std::size_t r = 0; r < n; ++r)
                    residual[r] -=
                        bj * samples[r].feats[static_cast<std::size_t>(j)];
            }
        }
    }

    std::vector<std::pair<f64, int>> ranked;
    for (int j = 0; j < d; ++j) {
        if (std::fabs(beta[static_cast<std::size_t>(j)]) > 1e-12)
            ranked.emplace_back(std::fabs(beta[static_cast<std::size_t>(j)]), j);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (opts.max_nonzero > 0 &&
        static_cast<int>(ranked.size()) > opts.max_nonzero)
        ranked.resize(static_cast<std::size_t>(opts.max_nonzero));

    model.joint_dim = d;
    model.quadratic = false;
    model.intercept = intercept;
    for (const auto& pr : ranked)
        model.terms.push_back(
            {pr.second, -1, beta[static_cast<std::size_t>(pr.second)]});
    model.loaded = !model.terms.empty() || std::fabs(intercept) > 0.0;
    return model;
}

HgtsmModel fit_hgtsm(const HgtsmCollector& collector,
                     const HgtsmFitOptions& opts) {
    HgtsmModel model = fit_hgtsm_lasso(collector.samples, opts);
    if (!opts.fit_graph) return model;

    model.graph.init_xavier(
        opts.emb_dim, opts.n_msg_layers,
        opts.sequence == HgtsmSequenceKind::Gru ? 1 : 0, opts.seed);
    model.has_graph = model.graph.valid();
    if (!model.has_graph) return model;

    // Train score / ratio heads (+ light touch on W_cut) with SGD on rounds
    // when available; otherwise on singleton graphs from cut features.
    std::vector<HgtsmRoundSample> local_rounds;
    const std::vector<HgtsmRoundSample>* rounds = &collector.rounds;
    if (rounds->empty() && !collector.samples.empty()) {
        local_rounds.reserve(std::min<std::size_t>(collector.samples.size(), 512));
        for (std::size_t i = 0; i < collector.samples.size() &&
                                local_rounds.size() < 512;
             ++i) {
            HgtsmRoundSample rs;
            rs.graph.cuts.resize(1);
            CutNodeFeatureVec& cn = rs.graph.cuts[0];
            cn.fill(0.0);
            for (int k = 0; k < kCutNodeFeatureDim && k < kCutFeatureDim; ++k)
                cn[static_cast<std::size_t>(k)] =
                    collector.samples[i].feats[static_cast<std::size_t>(k)];
            rs.labels = {collector.samples[i].label};
            local_rounds.push_back(std::move(rs));
        }
        rounds = &local_rounds;
    }
    if (rounds->empty()) {
        model.loaded = model.loaded || model.has_graph;
        return model;
    }

    const int e = model.graph.emb_dim;
    const f64 lr = opts.sgd_lr;
    for (int epoch = 0; epoch < opts.sgd_epochs; ++epoch) {
        for (const HgtsmRoundSample& rs : *rounds) {
            if (rs.labels.empty() || rs.labels.size() != rs.graph.cuts.size())
                continue;
            std::vector<f64> H_cut;
            f64 ratio = 0.0;
            forward_graph(model.graph, rs.graph, H_cut, &ratio);
            // Target ratio ≈ fraction of positive labels.
            f64 pos = 0.0;
            for (f64 lab : rs.labels)
                if (lab > 0.0) pos += 1.0;
            const f64 target_ratio =
                rs.labels.empty()
                    ? 0.5
                    : pos / static_cast<f64>(rs.labels.size());
            const f64 ratio_err = ratio - target_ratio;

            std::vector<f64> pooled(static_cast<std::size_t>(e), 0.0);
            for (std::size_t t = 0; t < rs.graph.cuts.size(); ++t)
                for (int d = 0; d < e; ++d)
                    pooled[static_cast<std::size_t>(d)] +=
                        H_cut[t * static_cast<std::size_t>(e) +
                              static_cast<std::size_t>(d)];
            if (!rs.graph.cuts.empty()) {
                for (int d = 0; d < e; ++d)
                    pooled[static_cast<std::size_t>(d)] /=
                        static_cast<f64>(rs.graph.cuts.size());
            }
            for (int d = 0; d < e; ++d)
                model.graph.W_ratio[static_cast<std::size_t>(d)] -=
                    lr * ratio_err * pooled[static_cast<std::size_t>(d)];
            model.graph.b_ratio -= lr * ratio_err;

            for (std::size_t t = 0; t < rs.graph.cuts.size(); ++t) {
                const f64* h =
                    H_cut.data() + t * static_cast<std::size_t>(e);
                const f64 pred = score_from_emb(model.graph, h);
                const f64 err = pred - rs.labels[t];
                for (int d = 0; d < e; ++d)
                    model.graph.W_score[static_cast<std::size_t>(d)] -=
                        lr * err * h[d];
                model.graph.b_score -= lr * err;
                // Light update of cut input projection toward label.
                for (int f = 0; f < kCutNodeFeatureDim; ++f) {
                    const f64 xf =
                        rs.graph.cuts[t][static_cast<std::size_t>(f)];
                    for (int d = 0; d < e; ++d) {
                        const std::size_t ix =
                            static_cast<std::size_t>(d * kCutNodeFeatureDim +
                                                    f);
                        model.graph.W_cut[ix] -=
                            lr * 0.1 * err *
                            model.graph.W_score[static_cast<std::size_t>(d)] *
                            xf;
                    }
                }
            }
        }
    }
    model.loaded = true;
    model.has_graph = true;
    return model;
}

namespace {

bool write_vec(std::ostream& out, const char* key, const std::vector<f64>& v) {
    out << key << " " << v.size();
    for (f64 x : v) out << " " << x;
    out << "\n";
    return static_cast<bool>(out);
}

bool read_vec(std::istream& in, std::vector<f64>& v) {
    std::size_t n = 0;
    if (!(in >> n)) return false;
    v.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        if (!(in >> v[i])) return false;
    return true;
}

}  // namespace

bool save_hgtsm_model(const std::string& path, const HgtsmModel& model) {
    std::ofstream out(path);
    if (!out) return false;
    const int ver = model.has_graph ? 2 : 1;
    out << "SOR_HGTSM " << ver << "\n";
    out << "joint_dim " << model.joint_dim << "\n";
    out << "quadratic " << (model.quadratic ? 1 : 0) << "\n";
    out << "intercept " << model.intercept << "\n";
    out << "nnz " << model.terms.size() << "\n";
    for (const HgtsmTerm& t : model.terms)
        out << t.i << " " << t.j << " " << t.coef << "\n";
    if (ver >= 2 && model.has_graph) {
        const auto& g = model.graph;
        out << "graph 1\n";
        out << "emb_dim " << g.emb_dim << "\n";
        out << "n_msg_layers " << g.n_msg_layers << "\n";
        out << "seq_kind " << g.seq_kind << "\n";
        out << "b_score " << g.b_score << "\n";
        out << "b_ratio " << g.b_ratio << "\n";
        write_vec(out, "W_var", g.W_var);
        write_vec(out, "b_var", g.b_var);
        write_vec(out, "W_con", g.W_con);
        write_vec(out, "b_con", g.b_con);
        write_vec(out, "W_cut", g.W_cut);
        write_vec(out, "b_cut", g.b_cut);
        write_vec(out, "W_msg", g.W_msg);
        write_vec(out, "W_self", g.W_self);
        write_vec(out, "b_upd", g.b_upd);
        write_vec(out, "W_qkv", g.W_qkv);
        write_vec(out, "W_o", g.W_o);
        write_vec(out, "W_ff1", g.W_ff1);
        write_vec(out, "W_ff2", g.W_ff2);
        write_vec(out, "b_ff1", g.b_ff1);
        write_vec(out, "b_ff2", g.b_ff2);
        write_vec(out, "W_gru", g.W_gru);
        write_vec(out, "b_gru", g.b_gru);
        write_vec(out, "W_score", g.W_score);
        write_vec(out, "W_ratio", g.W_ratio);
    }
    return static_cast<bool>(out);
}

bool load_hgtsm_model(const std::string& path, HgtsmModel& model) {
    model.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string tag;
    int ver = 0;
    if (!(in >> tag >> ver) || tag != "SOR_HGTSM" || (ver != 1 && ver != 2))
        return false;
    std::string key;
    int nnz = 0;
    while (in >> key) {
        if (key == "joint_dim") {
            in >> model.joint_dim;
        } else if (key == "quadratic") {
            int q = 0;
            in >> q;
            model.quadratic = q != 0;
        } else if (key == "intercept") {
            in >> model.intercept;
        } else if (key == "nnz") {
            in >> nnz;
            break;
        } else {
            return false;
        }
    }
    if (model.joint_dim != kHgtsmJointDim) return false;
    model.terms.resize(static_cast<std::size_t>(std::max(0, nnz)));
    for (int k = 0; k < nnz; ++k) {
        HgtsmTerm t;
        if (!(in >> t.i >> t.j >> t.coef)) return false;
        model.terms[static_cast<std::size_t>(k)] = t;
    }
    model.loaded = true;

    if (ver >= 2) {
        std::string gtag;
        int gflag = 0;
        if (!(in >> gtag >> gflag) || gtag != "graph") {
            // Linear-only v2 header is fine.
            return true;
        }
        if (!gflag) return true;
        auto& g = model.graph;
        g.clear();
        while (in >> key) {
            if (key == "emb_dim")
                in >> g.emb_dim;
            else if (key == "n_msg_layers")
                in >> g.n_msg_layers;
            else if (key == "seq_kind")
                in >> g.seq_kind;
            else if (key == "b_score")
                in >> g.b_score;
            else if (key == "b_ratio")
                in >> g.b_ratio;
            else if (key == "W_var") {
                if (!read_vec(in, g.W_var)) return false;
            } else if (key == "b_var") {
                if (!read_vec(in, g.b_var)) return false;
            } else if (key == "W_con") {
                if (!read_vec(in, g.W_con)) return false;
            } else if (key == "b_con") {
                if (!read_vec(in, g.b_con)) return false;
            } else if (key == "W_cut") {
                if (!read_vec(in, g.W_cut)) return false;
            } else if (key == "b_cut") {
                if (!read_vec(in, g.b_cut)) return false;
            } else if (key == "W_msg") {
                if (!read_vec(in, g.W_msg)) return false;
            } else if (key == "W_self") {
                if (!read_vec(in, g.W_self)) return false;
            } else if (key == "b_upd") {
                if (!read_vec(in, g.b_upd)) return false;
            } else if (key == "W_qkv") {
                if (!read_vec(in, g.W_qkv)) return false;
            } else if (key == "W_o") {
                if (!read_vec(in, g.W_o)) return false;
            } else if (key == "W_ff1") {
                if (!read_vec(in, g.W_ff1)) return false;
            } else if (key == "W_ff2") {
                if (!read_vec(in, g.W_ff2)) return false;
            } else if (key == "b_ff1") {
                if (!read_vec(in, g.b_ff1)) return false;
            } else if (key == "b_ff2") {
                if (!read_vec(in, g.b_ff2)) return false;
            } else if (key == "W_gru") {
                if (!read_vec(in, g.W_gru)) return false;
            } else if (key == "b_gru") {
                if (!read_vec(in, g.b_gru)) return false;
            } else if (key == "W_score") {
                if (!read_vec(in, g.W_score)) return false;
            } else if (key == "W_ratio") {
                if (!read_vec(in, g.W_ratio)) return false;
                break;
            } else {
                return false;
            }
        }
        model.has_graph = g.valid();
        if (!model.has_graph) return false;
    }
    return true;
}

std::vector<std::size_t> hgtsm_sequence_order(const std::vector<f64>& scores,
                                              int max_keep) {
    std::vector<std::pair<f64, std::size_t>> ranked;
    ranked.reserve(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i)
        ranked.emplace_back(scores[i], i);
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });
    if (max_keep > 0 && static_cast<int>(ranked.size()) > max_keep)
        ranked.resize(static_cast<std::size_t>(max_keep));
    std::vector<std::size_t> out;
    out.reserve(ranked.size());
    for (const auto& r : ranked) out.push_back(r.second);
    return out;
}

}  // namespace sor::search
