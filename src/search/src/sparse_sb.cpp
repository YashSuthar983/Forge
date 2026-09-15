#include "sor/search/sparse_sb.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace sor::search {
namespace {

inline f64 soft_threshold(f64 z, f64 lam) {
    if (z > lam) return z - lam;
    if (z < -lam) return z + lam;
    return 0.0;
}

inline f64 sigmoid(f64 z) {
    if (z >= 20.0) return 1.0;
    if (z <= -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

int expanded_dim(int base, bool quadratic) {
    if (!quadratic) return base;
    return base + base + (base * (base - 1)) / 2;
}

void push_term_from_expanded(int exp_idx, int p, bool quad, f64 coef,
                             std::vector<SparseSbTerm>& terms) {
    if (std::fabs(coef) <= 1e-14) return;
    SparseSbTerm t;
    if (exp_idx < p) {
        t.i = exp_idx;
        t.j = -1;
    } else if (!quad) {
        return;
    } else if (exp_idx < 2 * p) {
        t.i = exp_idx - p;
        t.j = t.i;
    } else {
        int rem = exp_idx - 2 * p;
        for (int i = 0; i < p; ++i) {
            const int row = p - 1 - i;
            if (rem < row) {
                t.i = i;
                t.j = i + 1 + rem;
                break;
            }
            rem -= row;
        }
    }
    t.coef = coef;
    terms.push_back(t);
}

SparseSbModel sparsify_beta(const std::vector<f64>& beta, int p, bool quad,
                            f64 intercept, SparseSbLoss loss, int max_nonzero) {
    SparseSbModel model;
    model.clear();
    const int d = static_cast<int>(beta.size());
    std::vector<std::pair<f64, int>> ranked;
    ranked.reserve(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
        const f64 c = beta[static_cast<std::size_t>(j)];
        if (std::fabs(c) > 1e-14)
            ranked.emplace_back(std::fabs(c), j);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (max_nonzero > 0 && static_cast<int>(ranked.size()) > max_nonzero)
        ranked.resize(static_cast<std::size_t>(max_nonzero));

    model.base_dim = p;
    model.quadratic = quad;
    model.loss = loss;
    model.intercept = intercept;
    for (const auto& pr : ranked)
        push_term_from_expanded(pr.second, p, quad,
                                beta[static_cast<std::size_t>(pr.second)],
                                model.terms);
    model.loaded = !model.terms.empty() || std::fabs(intercept) > 0.0;
    return model;
}

}  // namespace

void SparseSbModel::clear() {
    base_dim = kBranchFeatureDim;
    quadratic = true;
    loss = SparseSbLoss::Lasso;
    intercept = 0.0;
    terms.clear();
    loaded = false;
}

void SparseSbModel::expand_features(const BranchFeatureVec& x,
                                    bool quadratic,
                                    std::vector<f64>& out,
                                    int base_dim) {
    const int p = std::max(1, std::min(base_dim, kBranchFeatureDim));
    out.assign(static_cast<std::size_t>(expanded_dim(p, quadratic)), 0.0);
    for (int i = 0; i < p; ++i)
        out[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)];
    if (!quadratic) return;
    int t = p;
    for (int i = 0; i < p; ++i)
        out[static_cast<std::size_t>(t++)] =
            x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
    for (int i = 0; i < p; ++i) {
        for (int j = i + 1; j < p; ++j)
            out[static_cast<std::size_t>(t++)] =
                x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(j)];
    }
}

f64 SparseSbModel::predict(const BranchFeatureVec& x) const {
    f64 y = intercept;
    const int dim_cap = std::min(base_dim, kBranchFeatureDim);
    for (const SparseSbTerm& t : terms) {
        if (t.i < 0 || t.i >= dim_cap) continue;
        f64 v = x[static_cast<std::size_t>(t.i)];
        if (t.j >= 0) {
            if (t.j >= dim_cap) continue;
            v *= x[static_cast<std::size_t>(t.j)];
        }
        y += t.coef * v;
    }
    return y;
}

void SparseSbCollector::add(const BranchFeatureVec& feats, f64 sb_score) {
    if (!(sb_score >= 0.0) || !std::isfinite(sb_score)) return;
    if (samples.size() >= max_samples) return;
    SparseSbSample s;
    s.feats = feats;
    s.sb_score = sb_score;
    samples.push_back(s);
}

SparseSbModel fit_sparse_sb_lasso(const std::vector<SparseSbSample>& samples,
                                  const SparseSbFitOptions& opts) {
    SparseSbModel model;
    model.clear();
    if (samples.size() < 2) return model;

    const int p = kBranchFeatureDim;
    const bool quad = opts.use_quadratic;
    const int d = expanded_dim(p, quad);
    const std::size_t n = samples.size();

    std::vector<std::vector<f64>> X(n);
    std::vector<f64> y(n, 0.0);
    f64 y_norm2 = 0.0;
    for (std::size_t r = 0; r < n; ++r) {
        SparseSbModel::expand_features(samples[r].feats, quad, X[r], p);
        y[r] = samples[r].sb_score;
        y_norm2 += y[r] * y[r];
    }
    if (y_norm2 > 0.0) {
        const f64 inv = 1.0 / std::sqrt(y_norm2);
        for (f64& v : y) v *= inv;
    }

    f64 intercept = opts.intercept;
    if (!std::isfinite(intercept)) {
        intercept = 0.0;
        for (f64 v : y) intercept += v;
        intercept /= static_cast<f64>(n);
    }

    std::vector<f64> beta(static_cast<std::size_t>(d), 0.0);
    std::vector<f64> residual(n);
    for (std::size_t r = 0; r < n; ++r)
        residual[r] = y[r] - intercept;

    const f64 lam = std::max(0.0, opts.lasso_lambda);
    std::vector<f64> col_norm2(static_cast<std::size_t>(d), 0.0);
    for (int j = 0; j < d; ++j) {
        f64 s = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            const f64 v = X[r][static_cast<std::size_t>(j)];
            s += v * v;
        }
        col_norm2[static_cast<std::size_t>(j)] = s;
    }

    for (int it = 0; it < opts.max_iter; ++it) {
        for (int j = 0; j < d; ++j) {
            const f64 cj = col_norm2[static_cast<std::size_t>(j)];
            if (cj <= 1e-18) continue;
            const f64 bj = beta[static_cast<std::size_t>(j)];
            if (bj != 0.0) {
                for (std::size_t r = 0; r < n; ++r)
                    residual[r] += bj * X[r][static_cast<std::size_t>(j)];
            }
            f64 rho = 0.0;
            for (std::size_t r = 0; r < n; ++r)
                rho += X[r][static_cast<std::size_t>(j)] * residual[r];
            const f64 bj_new = soft_threshold(rho / cj, lam / cj);
            beta[static_cast<std::size_t>(j)] = bj_new;
            if (bj_new != 0.0) {
                for (std::size_t r = 0; r < n; ++r)
                    residual[r] -= bj_new * X[r][static_cast<std::size_t>(j)];
            }
        }
    }

    return sparsify_beta(beta, p, quad, intercept, SparseSbLoss::Lasso,
                         opts.max_nonzero);
}

SparseSbModel fit_sparse_sb_ranking(const std::vector<SparseSbSample>& samples,
                                    const SparseSbFitOptions& opts) {
    SparseSbModel model;
    model.clear();
    if (samples.size() < 2) return model;

    const int p = kBranchFeatureDim;
    const bool quad = opts.use_quadratic;
    const int d = expanded_dim(p, quad);
    const std::size_t n = samples.size();

    std::vector<std::vector<f64>> X(n);
    for (std::size_t r = 0; r < n; ++r)
        SparseSbModel::expand_features(samples[r].feats, quad, X[r], p);

    // Build preference pairs (higher SB score preferred).
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    pairs.reserve(static_cast<std::size_t>(
        std::min<int>(opts.ranking_pair_cap > 0 ? opts.ranking_pair_cap : 20000,
                      20000)));
    for (std::size_t i = 0; i < n && static_cast<int>(pairs.size()) <
                                         (opts.ranking_pair_cap > 0
                                              ? opts.ranking_pair_cap
                                              : 20000);
         ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const f64 di = samples[i].sb_score - samples[j].sb_score;
            if (std::fabs(di) < 1e-12) continue;
            if (di > 0.0)
                pairs.emplace_back(i, j);
            else
                pairs.emplace_back(j, i);
            if (opts.ranking_pair_cap > 0 &&
                static_cast<int>(pairs.size()) >= opts.ranking_pair_cap)
                break;
        }
    }
    if (pairs.empty()) {
        // Degenerate labels — fall back to lasso path.
        SparseSbFitOptions alt = opts;
        alt.loss = SparseSbLoss::Lasso;
        return fit_sparse_sb_lasso(samples, alt);
    }

    std::vector<f64> beta(static_cast<std::size_t>(d), 0.0);
    f64 intercept = 0.0;
    const f64 lr = std::max(1e-4, opts.ranking_lr);
    const f64 l1 = std::max(0.0, opts.ranking_l1);
    const int epochs = std::max(1, opts.ranking_epochs);

    for (int ep = 0; ep < epochs; ++ep) {
        for (const auto& pr : pairs) {
            const std::size_t pos = pr.first;
            const std::size_t neg = pr.second;
            f64 s_pos = intercept;
            f64 s_neg = intercept;
            for (int k = 0; k < d; ++k) {
                s_pos += beta[static_cast<std::size_t>(k)] *
                         X[pos][static_cast<std::size_t>(k)];
                s_neg += beta[static_cast<std::size_t>(k)] *
                         X[neg][static_cast<std::size_t>(k)];
            }
            const f64 g = 1.0 - sigmoid(s_pos - s_neg);
            intercept += lr * g;
            for (int k = 0; k < d; ++k) {
                const f64 dx = X[pos][static_cast<std::size_t>(k)] -
                               X[neg][static_cast<std::size_t>(k)];
                beta[static_cast<std::size_t>(k)] += lr * g * dx;
            }
        }
        // Proximal L1 after each epoch (Sparse-SB sparsity spirit).
        if (l1 > 0.0) {
            for (f64& b : beta) b = soft_threshold(b, l1);
        }
    }

    return sparsify_beta(beta, p, quad, intercept, SparseSbLoss::Ranking,
                         opts.max_nonzero);
}

SparseSbModel fit_sparse_sb(const std::vector<SparseSbSample>& samples,
                            const SparseSbFitOptions& opts) {
    if (opts.loss == SparseSbLoss::Ranking)
        return fit_sparse_sb_ranking(samples, opts);
    return fit_sparse_sb_lasso(samples, opts);
}

bool save_sparse_sb_model(const std::string& path, const SparseSbModel& model) {
    std::ofstream out(path);
    if (!out) return false;
    out << "SOR_SPARSE_SB 2\n";
    out << "base_dim " << model.base_dim << "\n";
    out << "quadratic " << (model.quadratic ? 1 : 0) << "\n";
    out << "loss " << (model.loss == SparseSbLoss::Ranking ? "ranking" : "lasso")
        << "\n";
    out << "intercept " << model.intercept << "\n";
    out << "nnz " << model.terms.size() << "\n";
    for (const SparseSbTerm& t : model.terms)
        out << t.i << " " << t.j << " " << t.coef << "\n";
    return static_cast<bool>(out);
}

bool load_sparse_sb_model(const std::string& path, SparseSbModel& model) {
    model.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string tag;
    int ver = 0;
    if (!(in >> tag >> ver) || tag != "SOR_SPARSE_SB" || (ver != 1 && ver != 2))
        return false;
    std::string key;
    int nnz = 0;
    while (in >> key) {
        if (key == "base_dim") {
            in >> model.base_dim;
        } else if (key == "quadratic") {
            int q = 1;
            in >> q;
            model.quadratic = q != 0;
        } else if (key == "loss") {
            std::string loss;
            in >> loss;
            model.loss = (loss == "ranking") ? SparseSbLoss::Ranking
                                             : SparseSbLoss::Lasso;
        } else if (key == "intercept") {
            in >> model.intercept;
        } else if (key == "nnz") {
            in >> nnz;
            break;
        } else {
            return false;
        }
    }
    // Accept legacy 16-dim and current 24-dim models.
    if (model.base_dim <= 0 || model.base_dim > kBranchFeatureDim) return false;
    model.terms.resize(static_cast<std::size_t>(std::max(0, nnz)));
    for (int k = 0; k < nnz; ++k) {
        SparseSbTerm t;
        if (!(in >> t.i >> t.j >> t.coef)) return false;
        model.terms[static_cast<std::size_t>(k)] = t;
    }
    model.loaded = true;
    return true;
}

Index pick_sparse_sb_branch(const SparseSbModel& model,
                            const std::vector<Index>& candidates,
                            const std::vector<BranchFeatureVec>& feats) {
    if (!model.loaded || candidates.empty() ||
        candidates.size() != feats.size())
        return -1;
    Index best = -1;
    f64 best_s = -std::numeric_limits<f64>::infinity();
    for (std::size_t k = 0; k < candidates.size(); ++k) {
        const f64 s = model.predict(feats[k]);
        if (s > best_s) {
            best_s = s;
            best = candidates[k];
        }
    }
    return best;
}

}  // namespace sor::search
