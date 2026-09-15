#include "sor/search/tree_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace sor::search {
namespace {

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

f64 soft_threshold(f64 z, f64 lam) {
    if (z > lam) return z - lam;
    if (z < -lam) return z + lam;
    return 0.0;
}

f64 sigmoid(f64 z) {
    if (z > 20.0) return 1.0;
    if (z < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
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

bool should_separate_at_node(int depth,
                             std::uint64_t nodes_since_dual_improve,
                             const TreeCutOptions& opts) {
    if (!opts.enabled) return false;
    if (depth < 0) return false;
    if (opts.stall_skip_nodes > 0 &&
        nodes_since_dual_improve >= opts.stall_skip_nodes)
        return false;
    if (depth <= opts.always_depth) return true;
    if (opts.every_k <= 0) return false;
    return depth % opts.every_k == 0;
}

void inherit_local_cuts(const std::vector<ManagedCut>& parent_active,
                        const std::vector<ManagedCut>& neu,
                        std::vector<ManagedCut>& child_active) {
    child_active = parent_active;
    child_active.insert(child_active.end(), neu.begin(), neu.end());
}

bool local_cut_present(const std::vector<ManagedCut>& active,
                       const std::string& id) {
    for (const ManagedCut& c : active)
        if (c.id == id) return true;
    return false;
}

std::vector<CutRow> managed_to_rows(const std::vector<ManagedCut>& cuts) {
    std::vector<CutRow> rows;
    rows.reserve(cuts.size());
    for (const ManagedCut& c : cuts) rows.push_back(c.row);
    return rows;
}

std::string cut_content_id(const CutRow& row) {
    std::ostringstream os;
    os << "c";
    char buf[64];
    for (std::size_t k = 0; k < row.cols.size() && k < row.vals.size(); ++k) {
        std::snprintf(buf, sizeof buf, "|%d:%.6g",
                      static_cast<int>(row.cols[k]), row.vals[k]);
        os << buf;
    }
    std::snprintf(buf, sizeof buf, "#%.6g:%.6g", row.row_lo, row.row_hi);
    os << buf;
    return os.str();
}

f64 gcs_heuristic_score(const GcsCandidate& c) {
    if (c.seen_nodes <= 0) return 0.0;
    const f64 mean_eff = c.efficacy_sum / static_cast<f64>(c.seen_nodes);
    const f64 breadth =
        1.0 + std::log1p(static_cast<f64>(c.violation_nodes));
    const f64 depth_boost =
        c.first_depth >= 0
            ? 1.0 + 0.05 * std::min(1.0, c.depth_sum /
                                             static_cast<f64>(
                                                 std::max(1, c.seen_nodes)) /
                                             32.0)
            : 1.0;
    return mean_eff * breadth * depth_boost;
}

void fill_gcs_cut_feat(const GcsCandidate& c, GcsCutFeat& out) {
    out.fill(0.0);
    for (int i = 0; i < kCutFeatureDim && i < kGcsCutFeatDim; ++i)
        out[static_cast<std::size_t>(i)] = c.feats[static_cast<std::size_t>(i)];
    if (c.seen_nodes > 0) {
        out[0] = c.efficacy_sum / static_cast<f64>(c.seen_nodes);
        out[6] = std::min(
            1.0, static_cast<f64>(c.violation_nodes) /
                     static_cast<f64>(std::max(1, c.seen_nodes)));
        out[7] = std::min(1.0, std::log1p(static_cast<f64>(c.seen_nodes)) / 5.0);
    }
}

void fill_gcs_node_feat(const GcsObservation& ob, int pool_seen,
                        GcsNodeFeat& out) {
    out.fill(0.0);
    out[0] = std::min(1.0, static_cast<f64>(std::max(0, ob.depth)) / 64.0);
    out[1] = std::min(1.0, std::max(0.0, ob.violation));
    out[2] = std::isfinite(ob.gap_rel)
                 ? std::min(1.0, std::max(0.0, ob.gap_rel))
                 : 1.0;
    out[3] = std::isfinite(ob.dual_bound)
                 ? ob.dual_bound / (1.0 + std::fabs(ob.dual_bound))
                 : 0.0;
    out[4] = std::min(1.0, static_cast<f64>(std::max(0, pool_seen)) / 100.0);
    out[5] = 1.0;
}

void GcsGraphWeights::clear() {
    emb_dim = kGcsEmbDim;
    W_cut.clear();
    b_cut.clear();
    W_node.clear();
    b_node.clear();
    W_msg_cn.clear();
    W_msg_nc.clear();
    W_policy.clear();
    b_policy = 0.0;
}

void GcsGraphWeights::init_xavier(int emb, std::uint64_t seed) {
    clear();
    emb_dim = emb;
    std::uint64_t rng = seed ? seed : 0x47435321ull;
    auto fill_mat = [&](std::vector<f64>& M, int rows, int cols) {
        M.resize(static_cast<std::size_t>(rows * cols));
        for (f64& v : M) v = xavier(rng, cols, rows);
    };
    fill_mat(W_cut, emb, kGcsCutFeatDim);
    b_cut.assign(static_cast<std::size_t>(emb), 0.0);
    fill_mat(W_node, emb, kGcsNodeFeatDim);
    b_node.assign(static_cast<std::size_t>(emb), 0.0);
    fill_mat(W_msg_cn, emb, emb);
    fill_mat(W_msg_nc, emb, emb);
    W_policy.resize(static_cast<std::size_t>(emb));
    for (f64& v : W_policy) v = xavier(rng, emb, 1);
    b_policy = 0.0;
}

bool GcsGraphWeights::valid() const {
    if (emb_dim <= 0) return false;
    const int e = emb_dim;
    if (static_cast<int>(W_cut.size()) != e * kGcsCutFeatDim) return false;
    if (static_cast<int>(b_cut.size()) != e) return false;
    if (static_cast<int>(W_node.size()) != e * kGcsNodeFeatDim) return false;
    if (static_cast<int>(b_node.size()) != e) return false;
    if (static_cast<int>(W_msg_cn.size()) != e * e) return false;
    if (static_cast<int>(W_msg_nc.size()) != e * e) return false;
    if (static_cast<int>(W_policy.size()) != e) return false;
    return true;
}

void GcsModel::clear() {
    loaded = false;
    use_gnn = true;
    W_linear.clear();
    b_linear = 0.0;
    gnn.clear();
}

f64 GcsModel::score_linear(const GcsCutFeat& f) const {
    if (W_linear.size() != static_cast<std::size_t>(kGcsCutFeatDim))
        return 0.0;
    f64 y = b_linear;
    for (int i = 0; i < kGcsCutFeatDim; ++i)
        y += W_linear[static_cast<std::size_t>(i)] *
             f[static_cast<std::size_t>(i)];
    return y;
}

f64 GcsModel::score_gnn(const GcsCandidate& c) const {
    if (!gnn.valid()) {
        GcsCutFeat f{};
        fill_gcs_cut_feat(c, f);
        return score_linear(f);
    }
    const int e = gnn.emb_dim;
    GcsCutFeat cf{};
    fill_gcs_cut_feat(c, cf);
    std::vector<f64> h_cut(static_cast<std::size_t>(e), 0.0);
    matvec(gnn.W_cut, cf.data(), h_cut.data(), e, kGcsCutFeatDim);
    for (int d = 0; d < e; ++d)
        h_cut[static_cast<std::size_t>(d)] =
            std::max(0.0, h_cut[static_cast<std::size_t>(d)] +
                              gnn.b_cut[static_cast<std::size_t>(d)]);

    // Aggregate messages from observation-site (node) embeddings.
    std::vector<f64> acc(static_cast<std::size_t>(e), 0.0);
    int n_msg = 0;
    // Use up to last 16 observations as bipartite neighbors.
    const std::size_t start =
        c.history.size() > 16 ? c.history.size() - 16 : 0;
    for (std::size_t i = start; i < c.history.size(); ++i) {
        GcsNodeFeat nf{};
        fill_gcs_node_feat(c.history[i], c.seen_nodes, nf);
        std::vector<f64> h_node(static_cast<std::size_t>(e), 0.0);
        matvec(gnn.W_node, nf.data(), h_node.data(), e, kGcsNodeFeatDim);
        for (int d = 0; d < e; ++d)
            h_node[static_cast<std::size_t>(d)] =
                std::max(0.0, h_node[static_cast<std::size_t>(d)] +
                                  gnn.b_node[static_cast<std::size_t>(d)]);
        std::vector<f64> msg(static_cast<std::size_t>(e), 0.0);
        matvec(gnn.W_msg_cn, h_node.data(), msg.data(), e, e);
        for (int d = 0; d < e; ++d)
            acc[static_cast<std::size_t>(d)] +=
                msg[static_cast<std::size_t>(d)];
        ++n_msg;
    }
    if (n_msg > 0) {
        for (int d = 0; d < e; ++d) {
            h_cut[static_cast<std::size_t>(d)] = std::max(
                0.0, h_cut[static_cast<std::size_t>(d)] +
                         acc[static_cast<std::size_t>(d)] /
                             static_cast<f64>(n_msg));
        }
    }
    f64 y = gnn.b_policy;
    for (int d = 0; d < e; ++d)
        y += gnn.W_policy[static_cast<std::size_t>(d)] *
             h_cut[static_cast<std::size_t>(d)];
    return sigmoid(y);
}

f64 GcsModel::score(const GcsCandidate& c, bool prefer_heuristic) const {
    if (!c.globally_valid) return -1.0;  // never promote local
    if (!loaded || prefer_heuristic) return gcs_heuristic_score(c);
    if (use_gnn && gnn.valid()) return score_gnn(c);
    GcsCutFeat f{};
    fill_gcs_cut_feat(c, f);
    return score_linear(f);
}

void GcsCollector::add(const GcsCutFeat& f, f64 label) {
    if (!std::isfinite(label)) return;
    if (samples.size() >= max_samples) return;
    samples.push_back({f, label});
}

GcsModel fit_gcs(const GcsCollector& collector, const GcsFitOptions& opts) {
    GcsModel model;
    model.clear();
    if (collector.samples.size() < 2) return model;

    const std::size_t n = collector.samples.size();
    std::vector<f64> y(n);
    f64 intercept = 0.0;
    f64 y_norm2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = collector.samples[i].label;
        intercept += y[i];
        y_norm2 += y[i] * y[i];
    }
    intercept /= static_cast<f64>(n);
    if (y_norm2 > 0.0) {
        const f64 inv = 1.0 / std::sqrt(y_norm2);
        for (f64& v : y) v *= inv;
        intercept *= inv;
    }

    std::vector<f64> beta(static_cast<std::size_t>(kGcsCutFeatDim), 0.0);
    std::vector<f64> residual(n);
    for (std::size_t i = 0; i < n; ++i) residual[i] = y[i] - intercept;
    std::vector<f64> col_n2(static_cast<std::size_t>(kGcsCutFeatDim), 0.0);
    for (int j = 0; j < kGcsCutFeatDim; ++j) {
        f64 s = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            s += collector.samples[i].feats[static_cast<std::size_t>(j)] *
                 collector.samples[i].feats[static_cast<std::size_t>(j)];
        col_n2[static_cast<std::size_t>(j)] = s;
    }
    const f64 lam = std::max(0.0, opts.lasso_lambda);
    for (int it = 0; it < 200; ++it) {
        for (int j = 0; j < kGcsCutFeatDim; ++j) {
            const f64 cn = col_n2[static_cast<std::size_t>(j)];
            if (cn <= 1e-18) continue;
            f64 rho = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const f64 xij =
                    collector.samples[i].feats[static_cast<std::size_t>(j)];
                residual[i] += beta[static_cast<std::size_t>(j)] * xij;
                rho += xij * residual[i];
            }
            const f64 bj = soft_threshold(rho / cn, lam / cn);
            beta[static_cast<std::size_t>(j)] = bj;
            for (std::size_t i = 0; i < n; ++i)
                residual[i] -=
                    bj * collector.samples[i].feats[static_cast<std::size_t>(j)];
        }
    }
    model.W_linear = beta;
    model.b_linear = intercept;
    model.loaded = true;

    if (opts.fit_gnn) {
        model.gnn.init_xavier(opts.emb_dim, opts.seed);
        model.use_gnn = model.gnn.valid();
        // Distill linear policy into GNN policy head via SGD on cut feats
        // (node side uses synthetic single observation from feat[6..7]).
        const f64 lr = opts.sgd_lr;
        for (int ep = 0; ep < opts.sgd_epochs && model.use_gnn; ++ep) {
            for (const GcsSample& s : collector.samples) {
                GcsCandidate tmp;
                tmp.feats.fill(0.0);
                for (int i = 0; i < kCutFeatureDim && i < kGcsCutFeatDim; ++i)
                    tmp.feats[static_cast<std::size_t>(i)] =
                        s.feats[static_cast<std::size_t>(i)];
                tmp.efficacy_sum = s.feats[0];
                tmp.seen_nodes = 1;
                tmp.violation_nodes = s.feats[6] > 0.5 ? 1 : 0;
                tmp.globally_valid = true;
                GcsObservation ob;
                ob.depth = static_cast<int>(s.feats[7] * 64.0);
                ob.violation = s.feats[0];
                ob.gap_rel = 0.5;
                tmp.history.push_back(ob);

                const f64 pred = model.score_gnn(tmp);
                const f64 target = sigmoid(s.label);
                const f64 err = pred - target;
                // Update policy head only (stable distill).
                GcsCutFeat cf{};
                fill_gcs_cut_feat(tmp, cf);
                std::vector<f64> h_cut(
                    static_cast<std::size_t>(model.gnn.emb_dim), 0.0);
                matvec(model.gnn.W_cut, cf.data(), h_cut.data(),
                       model.gnn.emb_dim, kGcsCutFeatDim);
                for (int d = 0; d < model.gnn.emb_dim; ++d) {
                    h_cut[static_cast<std::size_t>(d)] = std::max(
                        0.0, h_cut[static_cast<std::size_t>(d)] +
                                 model.gnn.b_cut[static_cast<std::size_t>(d)]);
                    model.gnn.W_policy[static_cast<std::size_t>(d)] -=
                        lr * err * h_cut[static_cast<std::size_t>(d)];
                }
                model.gnn.b_policy -= lr * err;
            }
        }
    }
    return model;
}

bool save_gcs_model(const std::string& path, const GcsModel& model) {
    std::ofstream out(path);
    if (!out) return false;
    out << "SOR_GCS 1\n";
    out << "use_gnn " << (model.use_gnn ? 1 : 0) << "\n";
    out << "b_linear " << model.b_linear << "\n";
    write_vec(out, "W_linear", model.W_linear);
    if (model.use_gnn && model.gnn.valid()) {
        out << "gnn 1\n";
        out << "emb_dim " << model.gnn.emb_dim << "\n";
        out << "b_policy " << model.gnn.b_policy << "\n";
        write_vec(out, "W_cut", model.gnn.W_cut);
        write_vec(out, "b_cut", model.gnn.b_cut);
        write_vec(out, "W_node", model.gnn.W_node);
        write_vec(out, "b_node", model.gnn.b_node);
        write_vec(out, "W_msg_cn", model.gnn.W_msg_cn);
        write_vec(out, "W_msg_nc", model.gnn.W_msg_nc);
        write_vec(out, "W_policy", model.gnn.W_policy);
    } else {
        out << "gnn 0\n";
    }
    return static_cast<bool>(out);
}

bool load_gcs_model(const std::string& path, GcsModel& model) {
    model.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string tag;
    int ver = 0;
    if (!(in >> tag >> ver) || tag != "SOR_GCS" || ver != 1) return false;
    std::string key;
    int use_gnn = 1;
    if (!(in >> key >> use_gnn) || key != "use_gnn") return false;
    model.use_gnn = use_gnn != 0;
    if (!(in >> key >> model.b_linear) || key != "b_linear") return false;
    if (!(in >> key) || key != "W_linear") return false;
    if (!read_vec(in, model.W_linear)) return false;
    if (static_cast<int>(model.W_linear.size()) != kGcsCutFeatDim) return false;

    int gnn_flag = 0;
    if (!(in >> key >> gnn_flag) || key != "gnn") return false;
    if (gnn_flag) {
        auto& g = model.gnn;
        g.clear();
        while (in >> key) {
            if (key == "emb_dim")
                in >> g.emb_dim;
            else if (key == "b_policy")
                in >> g.b_policy;
            else if (key == "W_cut") {
                if (!read_vec(in, g.W_cut)) return false;
            } else if (key == "b_cut") {
                if (!read_vec(in, g.b_cut)) return false;
            } else if (key == "W_node") {
                if (!read_vec(in, g.W_node)) return false;
            } else if (key == "b_node") {
                if (!read_vec(in, g.b_node)) return false;
            } else if (key == "W_msg_cn") {
                if (!read_vec(in, g.W_msg_cn)) return false;
            } else if (key == "W_msg_nc") {
                if (!read_vec(in, g.W_msg_nc)) return false;
            } else if (key == "W_policy") {
                if (!read_vec(in, g.W_policy)) return false;
                break;
            } else {
                return false;
            }
        }
        if (!g.valid()) return false;
    }
    model.loaded = true;
    return true;
}

void GcsPool::trim() {
    if (pool_max <= 0 || static_cast<int>(cands.size()) <= pool_max) return;
    struct Ranked {
        f64 score;
        std::size_t idx;
    };
    std::vector<Ranked> ranked;
    ranked.reserve(cands.size());
    for (std::size_t i = 0; i < cands.size(); ++i) {
        ranked.push_back(
            {model.score(cands[i], /*prefer_heuristic=*/!model.loaded), i});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const Ranked& a, const Ranked& b) {
                  if (a.score != b.score) return a.score > b.score;
                  return a.idx < b.idx;
              });
    ranked.resize(static_cast<std::size_t>(pool_max));
    std::vector<GcsCandidate> keep;
    keep.reserve(ranked.size());
    for (const Ranked& r : ranked) keep.push_back(std::move(cands[r.idx]));
    cands.swap(keep);
}

void GcsPool::observe(const CutRow& row,
                      const CutFeatureVec& feats,
                      f64 efficacy,
                      bool violated,
                      bool globally_valid,
                      const std::string& id,
                      int depth,
                      f64 gap_rel,
                      f64 dual_bound) {
    GcsObservation ob;
    ob.depth = depth;
    ob.violation = std::max(0.0, efficacy);
    ob.gap_rel = gap_rel;
    ob.dual_bound = dual_bound;

    for (GcsCandidate& c : cands) {
        if (c.id == id) {
            c.feats = feats;
            c.efficacy_sum += efficacy;
            ++c.seen_nodes;
            if (violated) ++c.violation_nodes;
            c.globally_valid = c.globally_valid || globally_valid;
            if (c.row.cols.empty()) c.row = row;
            c.depth_sum += static_cast<f64>(std::max(0, depth));
            if (c.first_depth < 0) c.first_depth = depth;
            c.best_violation = std::max(c.best_violation, ob.violation);
            if (c.history.size() < 64) c.history.push_back(ob);
            return;
        }
    }
    GcsCandidate c;
    c.row = row;
    c.feats = feats;
    c.efficacy_sum = efficacy;
    c.seen_nodes = 1;
    c.violation_nodes = violated ? 1 : 0;
    c.globally_valid = globally_valid;
    c.id = id;
    c.depth_sum = static_cast<f64>(std::max(0, depth));
    c.first_depth = depth;
    c.best_violation = ob.violation;
    c.history.push_back(ob);
    cands.push_back(std::move(c));
    trim();
}

void GcsPool::touch_point(const std::vector<f64>& x, f64 viol_tol, int depth,
                          f64 gap_rel, f64 dual_bound) {
    for (GcsCandidate& c : cands) {
        if (!c.globally_valid || c.row.cols.empty()) continue;
        f64 activity = 0.0;
        bool ok = true;
        for (std::size_t k = 0; k < c.row.cols.size(); ++k) {
            const Index j = c.row.cols[k];
            if (j < 0 || static_cast<std::size_t>(j) >= x.size() ||
                k >= c.row.vals.size()) {
                ok = false;
                break;
            }
            activity += c.row.vals[k] * x[static_cast<std::size_t>(j)];
        }
        if (!ok) continue;
        f64 viol = 0.0;
        if (std::isfinite(c.row.row_hi))
            viol = std::max(viol, activity - c.row.row_hi);
        if (std::isfinite(c.row.row_lo))
            viol = std::max(viol, c.row.row_lo - activity);
        ++c.seen_nodes;
        c.efficacy_sum += std::max(0.0, viol);
        if (viol > viol_tol) ++c.violation_nodes;
        c.depth_sum += static_cast<f64>(std::max(0, depth));
        c.best_violation = std::max(c.best_violation, viol);
        GcsObservation ob;
        ob.depth = depth;
        ob.violation = viol;
        ob.gap_rel = gap_rel;
        ob.dual_bound = dual_bound;
        if (c.history.size() < 64) c.history.push_back(ob);
    }
}

std::vector<CutRow> GcsPool::select_global(int max_keep, f64 min_score,
                                           bool skip_promoted) {
    struct Ranked {
        f64 score;
        std::size_t idx;
    };
    std::vector<Ranked> ranked;
    ranked.reserve(cands.size());
    const bool prefer_h = prefer_heuristic || !model.loaded;
    diag.model_loaded = model.loaded;
    diag.used_heuristic = prefer_h;
    diag.used_gnn =
        !prefer_h && model.loaded && model.use_gnn && model.gnn.valid();
    for (std::size_t i = 0; i < cands.size(); ++i) {
        const GcsCandidate& c = cands[i];
        // Paper / product invariant: never promote local cuts.
        if (!c.globally_valid || c.seen_nodes <= 0) continue;
        if (skip_promoted && promoted_ids.count(c.id)) continue;
        const f64 score = model.score(c, prefer_h);
        ++diag.scored;
        if (score < min_score) continue;
        ranked.push_back({score, i});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const Ranked& a, const Ranked& b) {
                  if (a.score != b.score) return a.score > b.score;
                  return a.idx < b.idx;
              });
    if (max_keep > 0 && static_cast<int>(ranked.size()) > max_keep)
        ranked.resize(static_cast<std::size_t>(max_keep));
    std::vector<CutRow> out;
    out.reserve(ranked.size());
    for (const Ranked& r : ranked) out.push_back(cands[r.idx].row);
    return out;
}

void GcsPool::mark_promoted(const std::vector<CutRow>& rows) {
    for (const CutRow& r : rows) promoted_ids.insert(cut_content_id(r));
}

}  // namespace sor::search
