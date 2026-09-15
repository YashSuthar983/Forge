#include "sor/search/planbb.hpp"

#include "sor/search/sparse_sb.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
namespace sor::search {
namespace {

inline f64 relu(f64 x) { return x > 0.0 ? x : 0.0; }
inline f64 sigmoid(f64 x) {
    if (x >= 20.0) return 1.0;
    if (x <= -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}
inline f64 softplus_inv_log1p(f64 y) {
    // y = log1p(gain) approx; recover gain = expm1(y) clipped.
    if (!std::isfinite(y)) return 0.0;
    return std::max(0.0, std::expm1(std::min(20.0, y)));
}

f64 policy_prior_stub(const PlanBbPolicy& policy, const BranchFeatureVec& f) {
    if (policy.loaded) return policy.predict(f);
    return f[0] + 0.1 * f[7];
}

f64 prune_mass(const PlanBbChildEstimate& e, f64 eps) {
    if (!e.ok) return 0.0;
    const f64 g =
        std::min(e.down_gain, e.up_gain) + 0.1 * std::max(e.down_gain, e.up_gain);
    if (!std::isfinite(g) || g < 0.0) return 0.0;
    return 1.0 - std::exp(-g / (1.0 + eps));
}

f64 leaf_value_lite(std::uint32_t avail,
                    const std::vector<f64>& priors,
                    const std::vector<PlanBbChildEstimate>& est,
                    f64 eps) {
    f64 best = -std::numeric_limits<f64>::infinity();
    for (std::size_t i = 0; avail; ++i, avail >>= 1) {
        if ((avail & 1u) == 0) continue;
        f64 s = priors[i];
        const f64 lv = planbb_lookahead_value(est[i], eps);
        if (std::isfinite(lv)) s = 0.3 * s + 0.7 * lv;
        if (s > best) best = s;
    }
    return std::isfinite(best) ? best : 0.0;
}

void pack_feats_graph(const BranchFeatureVec& feats,
                      const PlanBbGraphPool& graph,
                      std::vector<f64>& x) {
    x.resize(static_cast<std::size_t>(kBranchFeatureDim + kPlanBbGraphPoolDim));
    for (int i = 0; i < kBranchFeatureDim; ++i)
        x[static_cast<std::size_t>(i)] = feats[static_cast<std::size_t>(i)];
    for (int i = 0; i < kPlanBbGraphPoolDim; ++i)
        x[static_cast<std::size_t>(kBranchFeatureDim + i)] =
            graph[static_cast<std::size_t>(i)];
}

void pack_latent_feats(const PlanBbLatent& z,
                       const BranchFeatureVec& feats,
                       std::vector<f64>& x) {
    x.resize(static_cast<std::size_t>(kPlanBbLatentDim + kBranchFeatureDim));
    for (int i = 0; i < kPlanBbLatentDim; ++i)
        x[static_cast<std::size_t>(i)] = z[static_cast<std::size_t>(i)];
    for (int i = 0; i < kBranchFeatureDim; ++i)
        x[static_cast<std::size_t>(kPlanBbLatentDim + i)] =
            feats[static_cast<std::size_t>(i)];
}

PlanBbTransition heuristic_transition(const BranchFeatureVec& feats,
                                      const PlanBbChildEstimate* est,
                                      f64 eps) {
    PlanBbTransition t;
    t.next.fill(0.0);
    if (est && est->ok && std::isfinite(est->down_gain) &&
        std::isfinite(est->up_gain)) {
        t.down_gain = std::max(0.0, est->down_gain);
        t.up_gain = std::max(0.0, est->up_gain);
        t.prune_prob = prune_mass(*est, eps);
    } else {
        // Feature proxy when probes unavailable (paper: learned model fills this).
        const f64 frac = std::max(0.0, feats[0]);
        const f64 pc = std::max(0.0, feats[7]);
        t.down_gain = frac * (0.5 + pc);
        t.up_gain = (0.5 - std::min(0.5, frac)) * (0.5 + pc);
        t.prune_prob = 1.0 - std::exp(-(t.down_gain + t.up_gain));
    }
    // Seed next latent from action features (identity-ish).
    for (int i = 0; i < kPlanBbLatentDim; ++i) {
        const int src = i % kBranchFeatureDim;
        t.next[static_cast<std::size_t>(i)] =
            feats[static_cast<std::size_t>(src)] * (1.0 - 0.5 * t.prune_prob);
    }
    return t;
}

struct LiteMctsNode {
    std::uint32_t avail = 0;
    int depth_left = 0;
    std::vector<f64> child_q;
    std::vector<std::uint32_t> child_n;
    std::vector<int> child_id;
    std::uint32_t n = 0;
};

struct PaperMctsNode {
    PlanBbLatent latent{};
    std::uint32_t avail = 0;
    int depth_left = 0;
    bool terminal = false;
    std::vector<f64> child_q;
    std::vector<std::uint32_t> child_n;
    std::vector<int> child_id;  // expanded child node id per action
    std::vector<f64> child_prior;
    std::uint32_t n = 0;
};

}  // namespace

// ---- Graph pool -----------------------------------------------------------

PlanBbGraphPool zero_graph_pool() {
    PlanBbGraphPool g{};
    g.fill(0.0);
    return g;
}

void pool_bipartite_for_planbb(const BipartiteGraphSnapshot& g,
                               PlanBbGraphPool& out) {
    out.fill(0.0);
    // Mean-pool var (8) + con (6) + edge (2) → 16. Light GNN-style state.
    if (!g.vars.empty()) {
        for (const auto& v : g.vars)
            for (int d = 0; d < kVarNodeFeatureDim; ++d)
                out[static_cast<std::size_t>(d)] += v[static_cast<std::size_t>(d)];
        const f64 inv = 1.0 / static_cast<f64>(g.vars.size());
        for (int d = 0; d < kVarNodeFeatureDim; ++d)
            out[static_cast<std::size_t>(d)] *= inv;
    }
    if (!g.cons.empty()) {
        for (const auto& c : g.cons)
            for (int d = 0; d < kConNodeFeatureDim; ++d)
                out[static_cast<std::size_t>(kVarNodeFeatureDim + d)] +=
                    c[static_cast<std::size_t>(d)];
        const f64 inv = 1.0 / static_cast<f64>(g.cons.size());
        for (int d = 0; d < kConNodeFeatureDim; ++d)
            out[static_cast<std::size_t>(kVarNodeFeatureDim + d)] *= inv;
    }
    if (!g.edges.empty()) {
        const int base = kVarNodeFeatureDim + kConNodeFeatureDim;
        for (const auto& e : g.edges)
            for (int d = 0; d < kEdgeFeatureDim; ++d)
                out[static_cast<std::size_t>(base + d)] +=
                    e.feats[static_cast<std::size_t>(d)];
        const f64 inv = 1.0 / static_cast<f64>(g.edges.size());
        for (int d = 0; d < kEdgeFeatureDim; ++d)
            out[static_cast<std::size_t>(base + d)] *= inv;
    }
}

// ---- MLP ------------------------------------------------------------------

void PlanBbMlp::clear() {
    in_dim = hidden = out_dim = 0;
    w1.clear();
    b1.clear();
    w2.clear();
    b2.clear();
}

void PlanBbMlp::init(int in_d, int hid, int out_d, f64 scale) {
    clear();
    in_dim = in_d;
    hidden = hid;
    out_dim = out_d;
    w1.assign(static_cast<std::size_t>(hid * in_d), 0.0);
    b1.assign(static_cast<std::size_t>(hid), 0.0);
    w2.assign(static_cast<std::size_t>(out_d * hid), 0.0);
    b2.assign(static_cast<std::size_t>(out_d), 0.0);
    // Deterministic small init (no RNG dependency).
    for (std::size_t i = 0; i < w1.size(); ++i)
        w1[i] = scale * (1.0 - 2.0 * (static_cast<f64>((i * 17) % 100) / 100.0));
    for (std::size_t i = 0; i < w2.size(); ++i)
        w2[i] = scale * (1.0 - 2.0 * (static_cast<f64>((i * 29) % 100) / 100.0));
}

bool PlanBbMlp::valid() const {
    return in_dim > 0 && hidden > 0 && out_dim > 0 &&
           w1.size() == static_cast<std::size_t>(hidden * in_dim) &&
           b1.size() == static_cast<std::size_t>(hidden) &&
           w2.size() == static_cast<std::size_t>(out_dim * hidden) &&
           b2.size() == static_cast<std::size_t>(out_dim);
}

void PlanBbMlp::forward(const f64* x, f64* y) const {
    std::vector<f64> h;
    forward_train(x, y, h);
}

void PlanBbMlp::forward_train(const f64* x, f64* y,
                              std::vector<f64>& h_out) const {
    h_out.assign(static_cast<std::size_t>(hidden), 0.0);
    for (int i = 0; i < hidden; ++i) {
        f64 s = b1[static_cast<std::size_t>(i)];
        const f64* row = w1.data() + static_cast<std::size_t>(i * in_dim);
        for (int j = 0; j < in_dim; ++j) s += row[j] * x[j];
        h_out[static_cast<std::size_t>(i)] = relu(s);
    }
    for (int o = 0; o < out_dim; ++o) {
        f64 s = b2[static_cast<std::size_t>(o)];
        const f64* row = w2.data() + static_cast<std::size_t>(o * hidden);
        for (int i = 0; i < hidden; ++i)
            s += row[i] * h_out[static_cast<std::size_t>(i)];
        y[o] = s;
    }
}

void PlanBbMlp::sgd_step(const f64* x, const std::vector<f64>& h,
                         const f64* dy, f64 lr) {
    // Backprop through 2-layer MLP with ReLU.
    std::vector<f64> dh(static_cast<std::size_t>(hidden), 0.0);
    for (int o = 0; o < out_dim; ++o) {
        const f64 g = dy[o];
        b2[static_cast<std::size_t>(o)] += lr * g;
        f64* row = w2.data() + static_cast<std::size_t>(o * hidden);
        for (int i = 0; i < hidden; ++i) {
            dh[static_cast<std::size_t>(i)] += g * row[i];
            row[i] += lr * g * h[static_cast<std::size_t>(i)];
        }
    }
    for (int i = 0; i < hidden; ++i) {
        if (h[static_cast<std::size_t>(i)] <= 0.0) continue;
        const f64 g = dh[static_cast<std::size_t>(i)];
        b1[static_cast<std::size_t>(i)] += lr * g;
        f64* row = w1.data() + static_cast<std::size_t>(i * in_dim);
        for (int j = 0; j < in_dim; ++j) row[j] += lr * g * x[j];
    }
}

// ---- Legacy linear policy -------------------------------------------------

void PlanBbPolicy::clear() {
    base_dim = kBranchFeatureDim;
    intercept = 0.0;
    weights.fill(0.0);
    loaded = false;
}

f64 PlanBbPolicy::predict(const BranchFeatureVec& x) const {
    f64 s = intercept;
    for (int i = 0; i < kBranchFeatureDim; ++i)
        s += weights[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
    return s;
}

f64 planbb_lookahead_value(const PlanBbChildEstimate& est, f64 eps) {
    if (!est.ok) return -std::numeric_limits<f64>::infinity();
    if (!std::isfinite(est.down_gain) || !std::isfinite(est.up_gain))
        return -std::numeric_limits<f64>::infinity();
    return sb_product_score(est.down_gain, est.up_gain, eps);
}

PlanBbPolicy fit_planbb_policy(const std::vector<PlanBbSample>& samples,
                               const PlanBbFitOptions& opts) {
    PlanBbPolicy p;
    p.clear();
    if (samples.size() < 2) return p;

    f64 mean = 0.0;
    for (const auto& s : samples) mean += s.target;
    mean /= static_cast<f64>(samples.size());
    f64 var = 0.0;
    for (const auto& s : samples) {
        const f64 d = s.target - mean;
        var += d * d;
    }
    const f64 inv =
        (var > 1e-18) ? (1.0 / std::sqrt(var / static_cast<f64>(samples.size())))
                      : 1.0;

    const f64 lr = std::max(1e-4, opts.lr);
    for (int epoch = 0; epoch < opts.epochs; ++epoch) {
        for (const auto& s : samples) {
            const f64 y = (s.target - mean) * inv;
            const f64 pred = p.predict(s.feats);
            const f64 err = y - pred;
            p.intercept += lr * err;
            for (int i = 0; i < kBranchFeatureDim; ++i)
                p.weights[static_cast<std::size_t>(i)] +=
                    lr * err * s.feats[static_cast<std::size_t>(i)];
        }
    }
    p.loaded = true;
    return p;
}

bool save_planbb_policy(const std::string& path, const PlanBbPolicy& policy) {
    std::ofstream out(path);
    if (!out) return false;
    out << "SOR_PLANBB 1\n";
    out << "base_dim " << policy.base_dim << "\n";
    out << "intercept " << policy.intercept << "\n";
    out << "weights";
    for (int i = 0; i < kBranchFeatureDim; ++i)
        out << " " << policy.weights[static_cast<std::size_t>(i)];
    out << "\n";
    return static_cast<bool>(out);
}

bool load_planbb_policy(const std::string& path, PlanBbPolicy& policy) {
    policy.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string tag;
    int ver = 0;
    if (!(in >> tag >> ver) || tag != "SOR_PLANBB" || ver != 1) return false;
    std::string key;
    while (in >> key) {
        if (key == "base_dim") {
            in >> policy.base_dim;
        } else if (key == "intercept") {
            in >> policy.intercept;
        } else if (key == "weights") {
            policy.weights.fill(0.0);
            const int n_w = std::min(policy.base_dim > 0 ? policy.base_dim
                                                         : kBranchFeatureDim,
                                     kBranchFeatureDim);
            for (int i = 0; i < n_w; ++i)
                in >> policy.weights[static_cast<std::size_t>(i)];
            break;
        } else {
            return false;
        }
    }
    if (policy.base_dim <= 0 || policy.base_dim > kBranchFeatureDim) return false;
    policy.loaded = true;
    return true;
}

// ---- Paper model ----------------------------------------------------------

void PlanBbModel::clear() {
    latent_dim = kPlanBbLatentDim;
    hidden_dim = kPlanBbHiddenDim;
    repr.clear();
    dynamics.clear();
    policy.clear();
    value.clear();
    loaded = false;
}

PlanBbLatent PlanBbModel::encode(const BranchFeatureVec& feats,
                                 const PlanBbGraphPool& graph) const {
    PlanBbLatent z{};
    z.fill(0.0);
    if (!repr.valid()) return z;
    std::vector<f64> x;
    pack_feats_graph(feats, graph, x);
    std::vector<f64> y(static_cast<std::size_t>(repr.out_dim), 0.0);
    repr.forward(x.data(), y.data());
    for (int i = 0; i < latent_dim && i < repr.out_dim; ++i)
        z[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(i)];
    return z;
}

f64 PlanBbModel::policy_logit(const BranchFeatureVec& feats,
                              const PlanBbGraphPool& graph) const {
    if (!policy.valid()) return feats[0] + 0.1 * feats[7];
    std::vector<f64> x;
    pack_feats_graph(feats, graph, x);
    f64 y = 0.0;
    policy.forward(x.data(), &y);
    return y;
}

f64 PlanBbModel::value_of(const PlanBbLatent& z) const {
    if (!value.valid()) return 0.0;
    f64 y = 0.0;
    value.forward(z.data(), &y);
    return y;
}

PlanBbTransition PlanBbModel::transition(
    const PlanBbLatent& z, const BranchFeatureVec& action_feats) const {
    PlanBbTransition t;
    t.next.fill(0.0);
    if (!dynamics.valid())
        return heuristic_transition(action_feats, nullptr, 1e-6);
    std::vector<f64> x;
    pack_latent_feats(z, action_feats, x);
    std::vector<f64> y(static_cast<std::size_t>(dynamics.out_dim), 0.0);
    dynamics.forward(x.data(), y.data());
    t.down_gain = softplus_inv_log1p(y[0]);
    t.up_gain = softplus_inv_log1p(y[1]);
    t.prune_prob = sigmoid(y[2]);
    for (int i = 0; i < latent_dim && (3 + i) < dynamics.out_dim; ++i)
        t.next[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(3 + i)];
    return t;
}

void PlanBbCollector::add(const BranchFeatureVec& feats,
                          const PlanBbGraphPool& graph, f64 down_gain,
                          f64 up_gain, bool pruned) {
    if (samples.size() >= max_samples) return;
    PlanBbDynSample s;
    s.feats = feats;
    s.graph = graph;
    s.down_gain = down_gain;
    s.up_gain = up_gain;
    s.prune = pruned ? 1.0 : 0.0;
    samples.push_back(s);
}

PlanBbModel fit_planbb_paper(const std::vector<PlanBbDynSample>& samples,
                             const PlanBbPaperFitOptions& opts) {
    PlanBbModel m;
    m.clear();
    if (samples.size() < 4) return m;

    const int in_fg = kBranchFeatureDim + kPlanBbGraphPoolDim;
    const int in_zf = kPlanBbLatentDim + kBranchFeatureDim;
    const int dyn_out = 3 + kPlanBbLatentDim;
    m.repr.init(in_fg, kPlanBbHiddenDim, kPlanBbLatentDim);
    m.dynamics.init(in_zf, kPlanBbHiddenDim, dyn_out);
    m.policy.init(in_fg, kPlanBbHiddenDim, 1);
    m.value.init(kPlanBbLatentDim, kPlanBbHiddenDim / 2, 1);

    // Normalize SB products for policy/value targets.
    f64 mean_t = 0.0;
    for (const auto& s : samples) {
        mean_t += sb_product_score(std::max(0.0, s.down_gain),
                                   std::max(0.0, s.up_gain));
    }
    mean_t /= static_cast<f64>(samples.size());
    f64 var_t = 0.0;
    for (const auto& s : samples) {
        const f64 t = sb_product_score(std::max(0.0, s.down_gain),
                                       std::max(0.0, s.up_gain));
        const f64 d = t - mean_t;
        var_t += d * d;
    }
    const f64 inv =
        (var_t > 1e-18)
            ? (1.0 / std::sqrt(var_t / static_cast<f64>(samples.size())))
            : 1.0;

    const f64 lr = std::max(1e-4, opts.lr);
    std::vector<f64> x_fg, x_zf, h, y_dyn, y_pol, y_val, y_repr;
    y_dyn.resize(static_cast<std::size_t>(dyn_out));
    y_repr.resize(static_cast<std::size_t>(kPlanBbLatentDim));
    y_pol.resize(1);
    y_val.resize(1);
    std::vector<f64> dy(static_cast<std::size_t>(dyn_out), 0.0);

    for (int epoch = 0; epoch < opts.epochs; ++epoch) {
        for (const auto& s : samples) {
            pack_feats_graph(s.feats, s.graph, x_fg);

            // Representation → latent
            m.repr.forward_train(x_fg.data(), y_repr.data(), h);
            PlanBbLatent z{};
            for (int i = 0; i < kPlanBbLatentDim; ++i)
                z[static_cast<std::size_t>(i)] =
                    y_repr[static_cast<std::size_t>(i)];

            // Dynamics targets
            const f64 td = std::log1p(std::max(0.0, s.down_gain));
            const f64 tu = std::log1p(std::max(0.0, s.up_gain));
            const f64 tp = s.prune;  // match sigmoid via logit MSE on sigmoid out
            pack_latent_feats(z, s.feats, x_zf);
            m.dynamics.forward_train(x_zf.data(), y_dyn.data(), h);
            dy.assign(dy.size(), 0.0);
            dy[0] = opts.dynamics_w * (td - y_dyn[0]);
            dy[1] = opts.dynamics_w * (tu - y_dyn[1]);
            const f64 pr = sigmoid(y_dyn[2]);
            dy[2] = opts.dynamics_w * (tp - pr) * pr * (1.0 - pr);
            // Next latent consistency: pull toward scaled action features.
            for (int i = 0; i < kPlanBbLatentDim; ++i) {
                const f64 tgt =
                    s.feats[static_cast<std::size_t>(i % kBranchFeatureDim)] *
                    (1.0 - 0.5 * tp);
                dy[static_cast<std::size_t>(3 + i)] =
                    0.1 * opts.dynamics_w *
                    (tgt - y_dyn[static_cast<std::size_t>(3 + i)]);
            }
            m.dynamics.sgd_step(x_zf.data(), h, dy.data(), lr);

            // Policy: imitate normalized SB product
            const f64 sb = sb_product_score(std::max(0.0, s.down_gain),
                                            std::max(0.0, s.up_gain));
            const f64 y_sb = (sb - mean_t) * inv;
            m.policy.forward_train(x_fg.data(), y_pol.data(), h);
            const f64 dpol = opts.policy_w * (y_sb - y_pol[0]);
            m.policy.sgd_step(x_fg.data(), h, &dpol, lr);

            // Value: same SB proxy (higher dual progress → better)
            m.value.forward_train(z.data(), y_val.data(), h);
            const f64 dval = opts.value_w * (y_sb - y_val[0]);
            m.value.sgd_step(z.data(), h, &dval, lr);

            // Light repr nudge from value residual via encode path.
            std::vector<f64> drepr(static_cast<std::size_t>(kPlanBbLatentDim),
                                   0.0);
            for (int i = 0; i < kPlanBbLatentDim; ++i)
                drepr[static_cast<std::size_t>(i)] = 0.05 * dval;
            m.repr.forward_train(x_fg.data(), y_repr.data(), h);
            m.repr.sgd_step(x_fg.data(), h, drepr.data(), lr);
        }
    }
    m.loaded = true;
    return m;
}

namespace {

bool write_mlp(std::ostream& out, const char* name, const PlanBbMlp& m) {
    if (!m.valid()) return false;
    out << "mlp " << name << " " << m.in_dim << " " << m.hidden << " "
        << m.out_dim << "\n";
    out << "w1";
    for (f64 v : m.w1) out << " " << v;
    out << "\nb1";
    for (f64 v : m.b1) out << " " << v;
    out << "\nw2";
    for (f64 v : m.w2) out << " " << v;
    out << "\nb2";
    for (f64 v : m.b2) out << " " << v;
    out << "\n";
    return static_cast<bool>(out);
}

}  // namespace

bool save_planbb_model(const std::string& path, const PlanBbModel& model) {
    std::ofstream out(path);
    if (!out || !model.ready()) return false;
    out << "SOR_PLANBB_PAPER 1\n";
    out << "latent_dim " << model.latent_dim << "\n";
    out << "hidden_dim " << model.hidden_dim << "\n";
    if (!write_mlp(out, "repr", model.repr)) return false;
    if (!write_mlp(out, "dynamics", model.dynamics)) return false;
    if (!write_mlp(out, "policy", model.policy)) return false;
    if (!write_mlp(out, "value", model.value)) return false;
    return static_cast<bool>(out);
}

bool load_planbb_model(const std::string& path, PlanBbModel& model) {
    model.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string tag;
    int ver = 0;
    if (!(in >> tag >> ver) || tag != "SOR_PLANBB_PAPER" || ver != 1)
        return false;
    std::string key;
    int got = 0;
    while (got < 4 && (in >> key)) {
        if (key == "latent_dim") {
            in >> model.latent_dim;
        } else if (key == "hidden_dim") {
            in >> model.hidden_dim;
        } else if (key == "mlp") {
            std::string name;
            int in_d = 0, hid = 0, out_d = 0;
            if (!(in >> name >> in_d >> hid >> out_d)) return false;
            PlanBbMlp tmp;
            tmp.init(in_d, hid, out_d, 0.0);
            std::string wkey;
            if (!(in >> wkey) || wkey != "w1") return false;
            for (auto& v : tmp.w1) in >> v;
            if (!(in >> wkey) || wkey != "b1") return false;
            for (auto& v : tmp.b1) in >> v;
            if (!(in >> wkey) || wkey != "w2") return false;
            for (auto& v : tmp.w2) in >> v;
            if (!(in >> wkey) || wkey != "b2") return false;
            for (auto& v : tmp.b2) in >> v;
            if (!tmp.valid()) return false;
            if (name == "repr")
                model.repr = std::move(tmp);
            else if (name == "dynamics")
                model.dynamics = std::move(tmp);
            else if (name == "policy")
                model.policy = std::move(tmp);
            else if (name == "value")
                model.value = std::move(tmp);
            else
                return false;
            ++got;
        } else {
            return false;
        }
    }
    if (!model.repr.valid() || !model.dynamics.valid() || !model.policy.valid() ||
        !model.value.valid())
        return false;
    model.loaded = true;
    return true;
}

// ---- Lite MCTS ------------------------------------------------------------

Index pick_planbb_mcts(const PlanBbPolicy& policy, const PlanBbOptions& opts,
                       const std::vector<Index>& candidates,
                       const std::vector<BranchFeatureVec>& feats,
                       const std::vector<PlanBbChildEstimate>& estimates,
                       std::uint64_t* sims_used) {
    if (sims_used) *sims_used = 0;
    if (candidates.empty() || candidates.size() != feats.size() ||
        estimates.size() != candidates.size())
        return -1;

    const int k_all = static_cast<int>(candidates.size());
    const int k_lim = opts.top_k > 0 ? std::min(opts.top_k, k_all) : k_all;
    if (k_lim <= 0 || k_lim > 32) return -1;

    std::vector<std::size_t> order(static_cast<std::size_t>(k_all));
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::vector<f64> all_priors(static_cast<std::size_t>(k_all));
    for (int i = 0; i < k_all; ++i)
        all_priors[static_cast<std::size_t>(i)] =
            policy_prior_stub(policy, feats[static_cast<std::size_t>(i)]);
    std::partial_sort(order.begin(), order.begin() + k_lim, order.end(),
                      [&](std::size_t a, std::size_t b) {
                          return all_priors[a] > all_priors[b];
                      });

    std::vector<std::size_t> win(static_cast<std::size_t>(k_lim));
    std::vector<f64> priors(static_cast<std::size_t>(k_lim));
    std::vector<PlanBbChildEstimate> est(static_cast<std::size_t>(k_lim));
    std::vector<f64> look(static_cast<std::size_t>(k_lim), 0.0);
    bool any_ok = false;
    for (int i = 0; i < k_lim; ++i) {
        win[static_cast<std::size_t>(i)] = order[static_cast<std::size_t>(i)];
        priors[static_cast<std::size_t>(i)] =
            all_priors[order[static_cast<std::size_t>(i)]];
        est[static_cast<std::size_t>(i)] =
            estimates[order[static_cast<std::size_t>(i)]];
        const f64 lv =
            planbb_lookahead_value(est[static_cast<std::size_t>(i)], opts.eps);
        look[static_cast<std::size_t>(i)] = std::isfinite(lv) ? lv : 0.0;
        if (est[static_cast<std::size_t>(i)].ok) any_ok = true;
    }
    if (!any_ok) return -1;

    const int depth_cap = std::max(1, opts.mcts_depth);
    const int sims = std::max(1, opts.mcts_sims);
    const f64 c_puct = std::max(1e-6, opts.mcts_c_puct);
    const f64 eps = opts.eps;
    const f64 gamma = 0.5;

    f64 pmax = *std::max_element(priors.begin(), priors.end());
    f64 psum = 0.0;
    for (f64& p : priors) {
        p = std::exp(std::min(20.0, p - pmax));
        psum += p;
    }
    if (psum <= 0.0) {
        for (f64& p : priors) p = 1.0 / static_cast<f64>(k_lim);
    } else {
        for (f64& p : priors) p /= psum;
    }

    std::vector<LiteMctsNode> pool;
    pool.reserve(static_cast<std::size_t>(sims) * 2 + 4);

    auto make_node = [&](std::uint32_t avail, int depth_left) -> int {
        LiteMctsNode n;
        n.avail = avail;
        n.depth_left = depth_left;
        n.child_q.assign(static_cast<std::size_t>(k_lim), 0.0);
        n.child_n.assign(static_cast<std::size_t>(k_lim), 0);
        n.child_id.assign(static_cast<std::size_t>(k_lim), -1);
        pool.push_back(std::move(n));
        return static_cast<int>(pool.size()) - 1;
    };

    const std::uint32_t full_mask =
        (k_lim == 32) ? 0xffffffffu
                      : ((1u << static_cast<unsigned>(k_lim)) - 1u);
    const int root_id = make_node(full_mask, depth_cap);

    auto select_action = [&](LiteMctsNode& node) -> int {
        int best_a = -1;
        f64 best = -std::numeric_limits<f64>::infinity();
        const f64 sqrt_n = std::sqrt(static_cast<f64>(std::max(1u, node.n)));
        for (int a = 0; a < k_lim; ++a) {
            if (((node.avail >> a) & 1u) == 0) continue;
            const f64 n_a =
                static_cast<f64>(node.child_n[static_cast<std::size_t>(a)]);
            if (n_a <= 0.0) return a;
            const f64 q = node.child_q[static_cast<std::size_t>(a)];
            const f64 u =
                c_puct * priors[static_cast<std::size_t>(a)] * sqrt_n /
                (1.0 + n_a);
            const f64 score = q + u;
            if (score > best) {
                best = score;
                best_a = a;
            }
        }
        return best_a;
    };

    std::function<f64(int)> simulate = [&](int nid) -> f64 {
        LiteMctsNode& node = pool[static_cast<std::size_t>(nid)];
        if (node.depth_left <= 0 || node.avail == 0)
            return leaf_value_lite(node.avail, priors, est, eps);

        const int a = select_action(node);
        if (a < 0) return leaf_value_lite(node.avail, priors, est, eps);

        const f64 imm = look[static_cast<std::size_t>(a)];
        const f64 pm = prune_mass(est[static_cast<std::size_t>(a)], eps);

        f64 cont = 0.0;
        if (pm < 0.95 && node.depth_left > 1) {
            const std::uint32_t next_avail =
                node.avail & ~(1u << static_cast<unsigned>(a));
            int& cid = node.child_id[static_cast<std::size_t>(a)];
            if (cid < 0) cid = make_node(next_avail, node.depth_left - 1);
            cont = (1.0 - pm) * simulate(cid);
        }

        const f64 ret = imm + gamma * cont;
        auto& n_a = node.child_n[static_cast<std::size_t>(a)];
        auto& q_a = node.child_q[static_cast<std::size_t>(a)];
        ++n_a;
        q_a += (ret - q_a) / static_cast<f64>(n_a);
        ++node.n;
        return ret;
    };

    for (int s = 0; s < sims; ++s) simulate(root_id);
    if (sims_used) *sims_used = static_cast<std::uint64_t>(sims);

    const LiteMctsNode& root = pool[static_cast<std::size_t>(root_id)];
    int best_a = -1;
    std::uint32_t best_n = 0;
    f64 best_q = -std::numeric_limits<f64>::infinity();
    for (int a = 0; a < k_lim; ++a) {
        if (((root.avail >> a) & 1u) == 0) continue;
        const std::uint32_t n_a = root.child_n[static_cast<std::size_t>(a)];
        const f64 q = root.child_q[static_cast<std::size_t>(a)];
        if (q > best_q + 1e-12 ||
            (std::fabs(q - best_q) <= 1e-12 && n_a > best_n)) {
            best_n = n_a;
            best_q = q;
            best_a = a;
        }
    }
    if (best_a < 0) return -1;
    return candidates[win[static_cast<std::size_t>(best_a)]];
}

// ---- Paper MCTS (select / expand / simulate / backprop) -------------------

Index pick_planbb_mcts_paper(const PlanBbModel& model,
                             const PlanBbPolicy& prior_stub,
                             const PlanBbOptions& opts,
                             const std::vector<Index>& candidates,
                             const std::vector<BranchFeatureVec>& feats,
                             const PlanBbGraphPool& graph,
                             const std::vector<PlanBbChildEstimate>* estimates,
                             std::uint64_t* sims_used) {
    if (sims_used) *sims_used = 0;
    if (candidates.empty() || candidates.size() != feats.size()) return -1;

    const int k_all = static_cast<int>(candidates.size());
    const int k_lim = opts.top_k > 0 ? std::min(opts.top_k, k_all) : k_all;
    if (k_lim <= 0 || k_lim > 32) return -1;

    const bool have_est =
        estimates && estimates->size() == candidates.size();
    const bool model_ok = model.ready();

    std::vector<std::size_t> order(static_cast<std::size_t>(k_all));
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::vector<f64> all_priors(static_cast<std::size_t>(k_all));
    for (int i = 0; i < k_all; ++i) {
        if (model_ok)
            all_priors[static_cast<std::size_t>(i)] =
                model.policy_logit(feats[static_cast<std::size_t>(i)], graph);
        else
            all_priors[static_cast<std::size_t>(i)] =
                policy_prior_stub(prior_stub, feats[static_cast<std::size_t>(i)]);
    }
    std::partial_sort(order.begin(), order.begin() + k_lim, order.end(),
                      [&](std::size_t a, std::size_t b) {
                          return all_priors[a] > all_priors[b];
                      });

    std::vector<std::size_t> win(static_cast<std::size_t>(k_lim));
    std::vector<f64> priors(static_cast<std::size_t>(k_lim));
    std::vector<BranchFeatureVec> win_feats(static_cast<std::size_t>(k_lim));
    std::vector<PlanBbChildEstimate> win_est(static_cast<std::size_t>(k_lim));
    for (int i = 0; i < k_lim; ++i) {
        win[static_cast<std::size_t>(i)] = order[static_cast<std::size_t>(i)];
        priors[static_cast<std::size_t>(i)] =
            all_priors[order[static_cast<std::size_t>(i)]];
        win_feats[static_cast<std::size_t>(i)] =
            feats[order[static_cast<std::size_t>(i)]];
        if (have_est)
            win_est[static_cast<std::size_t>(i)] =
                (*estimates)[order[static_cast<std::size_t>(i)]];
    }

    // Softmax priors for PUCT.
    f64 pmax = *std::max_element(priors.begin(), priors.end());
    f64 psum = 0.0;
    for (f64& p : priors) {
        p = std::exp(std::min(20.0, p - pmax));
        psum += p;
    }
    if (psum <= 0.0) {
        for (f64& p : priors) p = 1.0 / static_cast<f64>(k_lim);
    } else {
        for (f64& p : priors) p /= psum;
    }

    const int depth_cap = std::max(1, opts.mcts_depth);
    const int sims = std::max(1, opts.mcts_sims);
    const f64 c_puct = std::max(1e-6, opts.mcts_c_puct);
    const f64 eps = opts.eps;

    // Root latent: mean encode of window candidates (Gasse-style local obs).
    PlanBbLatent root_z{};
    root_z.fill(0.0);
    if (model_ok) {
        for (int i = 0; i < k_lim; ++i) {
            const auto zi = model.encode(win_feats[static_cast<std::size_t>(i)],
                                         graph);
            for (int d = 0; d < kPlanBbLatentDim; ++d)
                root_z[static_cast<std::size_t>(d)] +=
                    zi[static_cast<std::size_t>(d)];
        }
        const f64 inv = 1.0 / static_cast<f64>(k_lim);
        for (int d = 0; d < kPlanBbLatentDim; ++d)
            root_z[static_cast<std::size_t>(d)] *= inv;
    }

    std::vector<PaperMctsNode> pool;
    pool.reserve(static_cast<std::size_t>(sims) * 4 + 8);

    auto make_node = [&](const PlanBbLatent& z, std::uint32_t avail,
                         int depth_left, bool terminal) -> int {
        PaperMctsNode n;
        n.latent = z;
        n.avail = avail;
        n.depth_left = depth_left;
        n.terminal = terminal;
        n.child_q.assign(static_cast<std::size_t>(k_lim), 0.0);
        n.child_n.assign(static_cast<std::size_t>(k_lim), 0);
        n.child_id.assign(static_cast<std::size_t>(k_lim), -1);
        n.child_prior = priors;
        pool.push_back(std::move(n));
        return static_cast<int>(pool.size()) - 1;
    };

    const std::uint32_t full_mask =
        (k_lim == 32) ? 0xffffffffu
                      : ((1u << static_cast<unsigned>(k_lim)) - 1u);
    const int root_id = make_node(root_z, full_mask, depth_cap, false);

    auto predict_transition = [&](const PlanBbLatent& z, int a,
                                  int depth_from_root) -> PlanBbTransition {
        const PlanBbChildEstimate* est_ptr = nullptr;
        // Prefer real probes only at the root expansion (true LP children).
        if (depth_from_root == 0 && have_est &&
            win_est[static_cast<std::size_t>(a)].ok)
            est_ptr = &win_est[static_cast<std::size_t>(a)];
        if (model_ok && !est_ptr)
            return model.transition(z, win_feats[static_cast<std::size_t>(a)]);
        if (model_ok && est_ptr) {
            // Blend: keep probe gains, use model for next latent.
            auto t = model.transition(z, win_feats[static_cast<std::size_t>(a)]);
            t.down_gain = std::max(0.0, est_ptr->down_gain);
            t.up_gain = std::max(0.0, est_ptr->up_gain);
            t.prune_prob = prune_mass(*est_ptr, eps);
            return t;
        }
        return heuristic_transition(win_feats[static_cast<std::size_t>(a)],
                                    est_ptr, eps);
    };

    auto select_action = [&](PaperMctsNode& node) -> int {
        int best_a = -1;
        f64 best = -std::numeric_limits<f64>::infinity();
        const f64 sqrt_n = std::sqrt(static_cast<f64>(std::max(1u, node.n)));
        for (int a = 0; a < k_lim; ++a) {
            if (((node.avail >> a) & 1u) == 0) continue;
            const f64 n_a =
                static_cast<f64>(node.child_n[static_cast<std::size_t>(a)]);
            const f64 p = node.child_prior[static_cast<std::size_t>(a)];
            if (n_a <= 0.0) {
                // Expand unvisited first (classic).
                const f64 u = c_puct * p * sqrt_n;
                if (u > best || best_a < 0) {
                    best = u;
                    best_a = a;
                }
                continue;
            }
            const f64 q = node.child_q[static_cast<std::size_t>(a)];
            const f64 u = c_puct * p * sqrt_n / (1.0 + n_a);
            const f64 score = q + u;
            if (score > best) {
                best = score;
                best_a = a;
            }
        }
        return best_a;
    };

    // One simulation: selection → expansion → evaluate → backprop.
    for (int s = 0; s < sims; ++s) {
        std::vector<std::pair<int, int>> path;  // (node_id, action)
        int nid = root_id;
        int depth_from_root = 0;

        // Selection
        while (true) {
            PaperMctsNode& node = pool[static_cast<std::size_t>(nid)];
            if (node.terminal || node.depth_left <= 0 || node.avail == 0) break;
            const int a = select_action(node);
            if (a < 0) break;
            path.emplace_back(nid, a);
            int& cid = node.child_id[static_cast<std::size_t>(a)];
            if (cid < 0) {
                // Expansion via dynamics
                const auto tr =
                    predict_transition(node.latent, a, depth_from_root);
                const bool pruned = tr.prune_prob >= 0.9;
                const std::uint32_t next_avail =
                    node.avail & ~(1u << static_cast<unsigned>(a));
                cid = make_node(tr.next, next_avail, node.depth_left - 1,
                                pruned || next_avail == 0);
                // Immediate dual-progress reward proxy + value backup.
                f64 leaf = sb_product_score(tr.down_gain, tr.up_gain, eps);
                if (model_ok && !pruned)
                    leaf = 0.5 * leaf + 0.5 * model.value_of(tr.next);
                // Paper R=-1 per step; we maximize dual progress, so use
                // progress - depth_penalty as the backed-up return.
                f64 G = leaf - 0.05 * static_cast<f64>(depth_from_root + 1);
                // Backprop along path
                for (int pi = static_cast<int>(path.size()) - 1; pi >= 0; --pi) {
                    const int pn = path[static_cast<std::size_t>(pi)].first;
                    const int pa = path[static_cast<std::size_t>(pi)].second;
                    auto& pn_node = pool[static_cast<std::size_t>(pn)];
                    auto& n_a = pn_node.child_n[static_cast<std::size_t>(pa)];
                    auto& q_a = pn_node.child_q[static_cast<std::size_t>(pa)];
                    ++n_a;
                    q_a += (G - q_a) / static_cast<f64>(n_a);
                    ++pn_node.n;
                    G -= 0.05;  // accumulate step cost (tree-size proxy)
                }
                goto next_sim;
            }
            nid = cid;
            ++depth_from_root;
        }

        // Simulation from existing leaf: evaluate value head / heuristic.
        {
            const PaperMctsNode& leaf_n = pool[static_cast<std::size_t>(nid)];
            f64 G = model_ok ? model.value_of(leaf_n.latent) : 0.0;
            G -= 0.05 * static_cast<f64>(depth_from_root);
            for (int pi = static_cast<int>(path.size()) - 1; pi >= 0; --pi) {
                const int pn = path[static_cast<std::size_t>(pi)].first;
                const int pa = path[static_cast<std::size_t>(pi)].second;
                auto& pn_node = pool[static_cast<std::size_t>(pn)];
                auto& n_a = pn_node.child_n[static_cast<std::size_t>(pa)];
                auto& q_a = pn_node.child_q[static_cast<std::size_t>(pa)];
                ++n_a;
                q_a += (G - q_a) / static_cast<f64>(n_a);
                ++pn_node.n;
                G -= 0.05;
            }
        }
    next_sim:;
    }

    if (sims_used) *sims_used = static_cast<std::uint64_t>(sims);

    const PaperMctsNode& root = pool[static_cast<std::size_t>(root_id)];
    int best_a = -1;
    std::uint32_t best_n = 0;
    f64 best_q = -std::numeric_limits<f64>::infinity();
    for (int a = 0; a < k_lim; ++a) {
        if (((root.avail >> a) & 1u) == 0) continue;
        const std::uint32_t n_a = root.child_n[static_cast<std::size_t>(a)];
        const f64 q = root.child_q[static_cast<std::size_t>(a)];
        // Prefer visit count (PUCT/Gumbel spirit); break ties by Q.
        if (n_a > best_n || (n_a == best_n && q > best_q)) {
            best_n = n_a;
            best_q = q;
            best_a = a;
        }
    }
    if (best_a < 0) return -1;
    return candidates[win[static_cast<std::size_t>(best_a)]];
}

// ---- Unified pick ---------------------------------------------------------

Index pick_planbb_branch(const PlanBbPolicy& policy, const PlanBbOptions& opts,
                         const std::vector<Index>& candidates,
                         const std::vector<BranchFeatureVec>& feats,
                         const std::vector<PlanBbChildEstimate>* estimates,
                         std::uint64_t* mcts_sims_used) {
    return pick_planbb_branch(nullptr, policy, opts, candidates, feats, nullptr,
                              estimates, mcts_sims_used);
}

Index pick_planbb_branch(const PlanBbModel* model, const PlanBbPolicy& policy,
                         const PlanBbOptions& opts,
                         const std::vector<Index>& candidates,
                         const std::vector<BranchFeatureVec>& feats,
                         const PlanBbGraphPool* graph,
                         const std::vector<PlanBbChildEstimate>* estimates,
                         std::uint64_t* mcts_sims_used) {
    if (mcts_sims_used) *mcts_sims_used = 0;
    if (candidates.empty() || candidates.size() != feats.size()) return -1;

    const bool paper = planbb_use_paper_path(opts, model);
    const PlanBbGraphPool g = graph ? *graph : zero_graph_pool();

    if (paper && opts.use_mcts) {
        PlanBbModel empty;
        const PlanBbModel& m = (model && model->ready()) ? *model : empty;
        const Index pb = pick_planbb_mcts_paper(
            m, policy, opts, candidates, feats, g, estimates, mcts_sims_used);
        if (pb >= 0) return pb;
        // Fall through to lite.
    }

    const bool have_est = estimates && estimates->size() == candidates.size();

    if (!paper && opts.use_mcts && have_est) {
        const Index mcts = pick_planbb_mcts(policy, opts, candidates, feats,
                                            *estimates, mcts_sims_used);
        if (mcts >= 0) return mcts;
    }

    const bool use_look = opts.shallow_lookahead && have_est;

    struct Ranked {
        std::size_t idx;
        f64 policy_s;
    };
    std::vector<Ranked> ranked;
    ranked.reserve(candidates.size());
    for (std::size_t k = 0; k < candidates.size(); ++k) {
        f64 ps = 0.0;
        if (model && model->ready())
            ps = model->policy_logit(feats[k], g);
        else if (policy.loaded)
            ps = policy.predict(feats[k]);
        else
            ps = feats[k][0] + 0.1 * feats[k][7];
        ranked.push_back({k, ps});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const Ranked& a, const Ranked& b) {
                  return a.policy_s > b.policy_s;
              });

    const int k_lim = opts.top_k > 0
                          ? std::min(opts.top_k, static_cast<int>(ranked.size()))
                          : static_cast<int>(ranked.size());

    Index best = -1;
    f64 best_s = -std::numeric_limits<f64>::infinity();
    const f64 alpha = std::max(0.0, std::min(1.0, opts.lookahead_alpha));

    for (int r = 0; r < k_lim; ++r) {
        const std::size_t k = ranked[static_cast<std::size_t>(r)].idx;
        f64 s = ranked[static_cast<std::size_t>(r)].policy_s;
        if (use_look) {
            const f64 lv = planbb_lookahead_value((*estimates)[k], opts.eps);
            if (std::isfinite(lv))
                s = (1.0 - alpha) * s + alpha * lv;
            else if (!(policy.loaded || (model && model->ready())))
                continue;
        }
        if (s > best_s) {
            best_s = s;
            best = candidates[k];
        }
    }
    return best;
}

}  // namespace sor::search
