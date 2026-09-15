#include "sor/search/sc_milp_branch.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sor::search {
namespace {

f64 sigmoid(f64 z) {
    if (z >= 20.0) return 1.0;
    if (z <= -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

f64 cosine_sim(const std::array<f64, kScMilpEmbedDim>& a,
               const std::array<f64, kScMilpEmbedDim>& b) {
    f64 dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < kScMilpEmbedDim; ++i) {
        dot += a[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
        na += a[static_cast<std::size_t>(i)] * a[static_cast<std::size_t>(i)];
        nb += b[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
    }
    const f64 d = std::sqrt(na) * std::sqrt(nb);
    if (d <= 1e-18) return 0.0;
    return dot / d;
}

}  // namespace

void ScMilpModel::clear() {
    base_dim = kBranchFeatureDim;
    n_strata = kBranchStratumCount;
    embed_dim = kScMilpEmbedDim;
    intercept = 0.0;
    weights.fill(0.0);
    stratum_bias.fill(0.0);
    proj.fill(0.0);
    loaded = false;
}

void ScMilpModel::embed(const BranchFeatureVec& x,
                        std::array<f64, kScMilpEmbedDim>& z) const {
    z.fill(0.0);
    const int dim = std::min(base_dim, kBranchFeatureDim);
    for (int r = 0; r < embed_dim; ++r) {
        f64 s = 0.0;
        for (int c = 0; c < dim; ++c) {
            s += proj[static_cast<std::size_t>(r * kBranchFeatureDim + c)] *
                 x[static_cast<std::size_t>(c)];
        }
        // tanh nonlinearity (MLP-lite head without a second layer).
        z[static_cast<std::size_t>(r)] = std::tanh(s);
    }
}

f64 ScMilpModel::score(const BranchFeatureVec& x) const {
    f64 s = intercept;
    const int dim = std::min(base_dim, kBranchFeatureDim);
    for (int i = 0; i < dim; ++i)
        s += weights[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
    const int st = branch_feature_stratum(x);
    if (st >= 0 && st < n_strata)
        s += stratum_bias[static_cast<std::size_t>(st)];
    return s;
}

void ScMilpCollector::add(const BranchFeatureVec& feats, f64 preference,
                          int decision_id) {
    if (!std::isfinite(preference)) return;
    if (samples.size() >= max_samples) return;
    ScMilpSample s;
    s.feats = feats;
    s.preference = preference;
    s.decision_id = decision_id;
    samples.push_back(s);
}

f64 heuristic_sc_milp_score(const BranchFeatureVec& x) {
    const int st = branch_feature_stratum(x);
    const f64 stratum_boost = 0.05 * static_cast<f64>(st % 4);
    f64 s = 2.0 * x[0] + 1.5 * x[7] + 0.5 * x[13] + 0.25 * x[8] +
            0.35 * x[14] * x[15] + stratum_boost + 0.1 * x[12];
    // Prefer candidates with tighter bipartite neighborhoods when present.
    if (kBranchFeatureDim > 21) s += 0.15 * x[21] + 0.1 * x[19];
    return s;
}

ScMilpModel fit_sc_milp_contrastive(const std::vector<ScMilpSample>& samples,
                                    const ScMilpFitOptions& opts) {
    ScMilpModel model;
    model.clear();
    if (samples.size() < 2) return model;

    // Initialize projection as a scaled identity-ish map for stability.
    for (int r = 0; r < kScMilpEmbedDim; ++r) {
        const int c = r % kBranchFeatureDim;
        model.proj[static_cast<std::size_t>(r * kBranchFeatureDim + c)] = 0.1;
    }

    // Normalize preferences for stable pairwise margins.
    f64 mean = 0.0;
    for (const auto& s : samples) mean += s.preference;
    mean /= static_cast<f64>(samples.size());
    f64 var = 0.0;
    for (const auto& s : samples) {
        const f64 d = s.preference - mean;
        var += d * d;
    }
    const f64 scale =
        (var > 1e-18) ? (1.0 / std::sqrt(var / static_cast<f64>(samples.size())))
                      : 1.0;
    auto pref = [&](std::size_t i) {
        return (samples[i].preference - mean) * scale;
    };

    // Expert winners per decision_id (imitation CE / pairwise within node).
    std::unordered_map<int, std::vector<std::size_t>> by_dec;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i].decision_id >= 0)
            by_dec[samples[i].decision_id].push_back(i);
    }

    const f64 lr = std::max(1e-4, opts.lr);
    const f64 lam = std::max(0.0, opts.contrastive_weight);
    const f64 tau = std::max(1e-3, opts.contrastive_tau);
    const f64 alpha = opts.stratum_alpha;

    for (int epoch = 0; epoch < opts.epochs; ++epoch) {
        // --- L_sup: pairwise ranking + within-decision expert push ---
        int pair_count = 0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            for (std::size_t j = i + 1; j < samples.size(); ++j) {
                if (pair_count >= opts.pair_cap) break;
                const f64 pi = pref(i);
                const f64 pj = pref(j);
                if (std::fabs(pi - pj) < 1e-9) continue;
                const std::size_t pos = (pi > pj) ? i : j;
                const std::size_t neg = (pi > pj) ? j : i;
                const f64 margin =
                    model.score(samples[pos].feats) -
                    model.score(samples[neg].feats);
                const f64 g = (1.0 - sigmoid(margin));
                ++pair_count;
                for (int k = 0; k < kBranchFeatureDim; ++k) {
                    const f64 dx =
                        samples[pos].feats[static_cast<std::size_t>(k)] -
                        samples[neg].feats[static_cast<std::size_t>(k)];
                    model.weights[static_cast<std::size_t>(k)] += lr * g * dx;
                }
                const int st_pos = branch_feature_stratum(samples[pos].feats);
                const int st_neg = branch_feature_stratum(samples[neg].feats);
                if (st_pos >= 0 && st_pos < kBranchStratumCount)
                    model.stratum_bias[static_cast<std::size_t>(st_pos)] +=
                        lr * g;
                if (st_neg >= 0 && st_neg < kBranchStratumCount)
                    model.stratum_bias[static_cast<std::size_t>(st_neg)] -=
                        lr * g;
            }
            if (pair_count >= opts.pair_cap) break;
        }

        // Softmax imitation within decision groups (expert = max preference).
        for (const auto& kv : by_dec) {
            const auto& idxs = kv.second;
            if (idxs.size() < 2) continue;
            std::size_t expert = idxs[0];
            f64 best_p = samples[expert].preference;
            for (std::size_t t = 1; t < idxs.size(); ++t) {
                if (samples[idxs[t]].preference > best_p) {
                    best_p = samples[idxs[t]].preference;
                    expert = idxs[t];
                }
            }
            // Log-softmax gradient: push expert up, others down.
            f64 max_s = -std::numeric_limits<f64>::infinity();
            std::vector<f64> scores(idxs.size(), 0.0);
            for (std::size_t t = 0; t < idxs.size(); ++t) {
                scores[t] = model.score(samples[idxs[t]].feats);
                max_s = std::max(max_s, scores[t]);
            }
            f64 Z = 0.0;
            for (f64& s : scores) {
                s = std::exp(s - max_s);
                Z += s;
            }
            if (!(Z > 0.0)) continue;
            for (std::size_t t = 0; t < idxs.size(); ++t) {
                const f64 p = scores[t] / Z;
                const f64 target = (idxs[t] == expert) ? 1.0 : 0.0;
                const f64 g = lr * (target - p);
                for (int k = 0; k < kBranchFeatureDim; ++k)
                    model.weights[static_cast<std::size_t>(k)] +=
                        g * samples[idxs[t]].feats[static_cast<std::size_t>(k)];
                const int st = branch_feature_stratum(samples[idxs[t]].feats);
                if (st >= 0 && st < kBranchStratumCount)
                    model.stratum_bias[static_cast<std::size_t>(st)] += g;
            }
        }

        // --- L_cons: dynamic stratified contrastive on embeddings ---
        if (lam <= 0.0) continue;
        std::vector<std::array<f64, kScMilpEmbedDim>> Z(samples.size());
        std::vector<int> G(samples.size(), 0);
        for (std::size_t i = 0; i < samples.size(); ++i) {
            model.embed(samples[i].feats, Z[i]);
            G[i] = branch_feature_stratum(samples[i].feats);
        }
        int cons_count = 0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (cons_count >= opts.contrastive_cap) break;
            // Collect positives (same stratum) and a few negatives.
            std::vector<std::size_t> pos;
            std::vector<std::size_t> all_other;
            for (std::size_t j = 0; j < samples.size(); ++j) {
                if (j == i) continue;
                if (G[j] == G[i]) pos.push_back(j);
                all_other.push_back(j);
            }
            if (pos.empty() || all_other.empty()) continue;
            // Soft contrastive: maximize sim to positives vs weighted negatives.
            for (std::size_t pidx : pos) {
                if (cons_count >= opts.contrastive_cap) break;
                const f64 sim_p = cosine_sim(Z[i], Z[pidx]) / tau;
                // Pick one negative with largest stratum distance if possible.
                std::size_t nidx = all_other[0];
                int best_dist = -1;
                for (std::size_t j : all_other) {
                    if (G[j] == G[i]) continue;
                    const int dist = std::abs(G[j] - G[i]);
                    if (dist > best_dist) {
                        best_dist = dist;
                        nidx = j;
                    }
                }
                if (G[nidx] == G[i]) continue;
                const f64 w =
                    sigmoid(alpha * static_cast<f64>(std::abs(G[nidx] - G[i])));
                const f64 sim_n = w * cosine_sim(Z[i], Z[nidx]) / tau;
                // Gradient on score of (sim_p - sim_n): push embeds apart/together.
                const f64 margin = sim_p - sim_n;
                const f64 g = lam * (1.0 - sigmoid(margin)) * lr;
                // Update projection rows via feature outer products (approx).
                for (int r = 0; r < kScMilpEmbedDim; ++r) {
                    const f64 zi = Z[i][static_cast<std::size_t>(r)];
                    const f64 zp = Z[pidx][static_cast<std::size_t>(r)];
                    const f64 zn = Z[nidx][static_cast<std::size_t>(r)];
                    for (int c = 0; c < kBranchFeatureDim; ++c) {
                        const f64 xi =
                            samples[i].feats[static_cast<std::size_t>(c)];
                        const f64 xp =
                            samples[pidx].feats[static_cast<std::size_t>(c)];
                        const f64 xn =
                            samples[nidx].feats[static_cast<std::size_t>(c)];
                        // d/dP_rc of tanh(P x): (1-z^2) x_c * message.
                        const f64 gi = (1.0 - zi * zi) * xi;
                        const f64 gp = (1.0 - zp * zp) * xp;
                        const f64 gn = (1.0 - zn * zn) * xn;
                        model.proj[static_cast<std::size_t>(
                            r * kBranchFeatureDim + c)] +=
                            g * (gi * zp + gp * zi - w * (gi * zn + gn * zi));
                    }
                }
                ++cons_count;
            }
        }
    }

    model.loaded = true;
    return model;
}

bool save_sc_milp_model(const std::string& path, const ScMilpModel& model) {
    std::ofstream out(path);
    if (!out) return false;
    out << "SOR_SC_MILP 2\n";
    out << "base_dim " << model.base_dim << "\n";
    out << "n_strata " << model.n_strata << "\n";
    out << "embed_dim " << model.embed_dim << "\n";
    out << "intercept " << model.intercept << "\n";
    out << "weights";
    for (int i = 0; i < kBranchFeatureDim; ++i)
        out << " " << model.weights[static_cast<std::size_t>(i)];
    out << "\nstratum_bias";
    for (int i = 0; i < kBranchStratumCount; ++i)
        out << " " << model.stratum_bias[static_cast<std::size_t>(i)];
    out << "\nproj";
    for (f64 v : model.proj) out << " " << v;
    out << "\n";
    return static_cast<bool>(out);
}

bool load_sc_milp_model(const std::string& path, ScMilpModel& model) {
    model.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string tag;
    int ver = 0;
    if (!(in >> tag >> ver) || tag != "SOR_SC_MILP" || (ver != 1 && ver != 2))
        return false;
    std::string key;
    while (in >> key) {
        if (key == "base_dim") {
            in >> model.base_dim;
        } else if (key == "n_strata") {
            in >> model.n_strata;
        } else if (key == "embed_dim") {
            in >> model.embed_dim;
        } else if (key == "intercept") {
            in >> model.intercept;
        } else if (key == "weights") {
            model.weights.fill(0.0);
            const int n_w = (ver == 1)
                                ? std::min(model.base_dim, kBranchFeatureDim)
                                : kBranchFeatureDim;
            for (int i = 0; i < n_w; ++i) {
                f64 v = 0.0;
                if (!(in >> v)) return false;
                model.weights[static_cast<std::size_t>(i)] = v;
            }
        } else if (key == "stratum_bias") {
            for (int i = 0; i < kBranchStratumCount; ++i)
                in >> model.stratum_bias[static_cast<std::size_t>(i)];
            if (ver == 1) break;
        } else if (key == "proj") {
            model.proj.fill(0.0);
            for (std::size_t i = 0; i < model.proj.size(); ++i) {
                if (!(in >> model.proj[i])) {
                    // Truncated proj is OK if partially present.
                    break;
                }
            }
            break;
        } else {
            return false;
        }
    }
    if (model.base_dim <= 0 || model.base_dim > kBranchFeatureDim) return false;
    if (model.n_strata != kBranchStratumCount) return false;
    model.loaded = true;
    return true;
}

Index pick_sc_milp_branch(const ScMilpModel& model,
                          const std::vector<Index>& candidates,
                          const std::vector<BranchFeatureVec>& feats,
                          bool allow_heuristic) {
    if (candidates.empty() || candidates.size() != feats.size()) return -1;
    if (!model.loaded && !allow_heuristic) return -1;
    Index best = -1;
    f64 best_s = -std::numeric_limits<f64>::infinity();
    for (std::size_t k = 0; k < candidates.size(); ++k) {
        const f64 s = model.loaded ? model.score(feats[k])
                                   : heuristic_sc_milp_score(feats[k]);
        if (s > best_s) {
            best_s = s;
            best = candidates[k];
        }
    }
    return best;
}

}  // namespace sor::search
