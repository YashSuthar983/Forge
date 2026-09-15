#include "sor/search/lifted_branch.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>

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

void expand_lifted(const LiftedFeatureVec& x, bool quadratic, int dim,
                   std::vector<f64>& out) {
    const int p = std::max(1, std::min(dim, kLiftedFeatureDim));
    out.assign(static_cast<std::size_t>(expanded_dim(p, quadratic)), 0.0);
    for (int i = 0; i < p; ++i)
        out[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)];
    if (!quadratic) return;
    int t = p;
    for (int i = 0; i < p; ++i)
        out[static_cast<std::size_t>(t++)] =
            x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
    for (int i = 0; i < p; ++i)
        for (int j = i + 1; j < p; ++j)
            out[static_cast<std::size_t>(t++)] =
                x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(j)];
}

void push_term(int exp_idx, int p, bool quad, f64 coef,
               std::vector<LiftedSbTerm>& terms) {
    if (std::fabs(coef) <= 1e-14) return;
    LiftedSbTerm t;
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

}  // namespace

void LiftedSbModel::clear() {
    dim = kLiftedFeatureDim;
    quadratic = false;
    loss = SparseSbLoss::Ranking;
    intercept = 0.0;
    terms.clear();
    loaded = false;
}

f64 LiftedSbModel::predict(const LiftedFeatureVec& x) const {
    f64 y = intercept;
    const int dim_cap = std::min(dim, kLiftedFeatureDim);
    for (const LiftedSbTerm& t : terms) {
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

void LiftedSbCollector::add(const LiftedFeatureVec& feats, f64 score) {
    if (!(score >= 0.0) || !std::isfinite(score)) return;
    if (samples.size() >= max_samples) return;
    LiftedSbSample s;
    s.feats = feats;
    s.score = score;
    samples.push_back(s);
}

void LiftedBranchState::clear() {
    expert.clear();
    buffer.clear();
    iterations = 0;
    samples_since_fit = 0;
    seeded_from_file = false;
}

void LiftedBranchState::observe(const LiftedFeatureVec& feats, f64 sb_score) {
    const std::size_t before = buffer.samples.size();
    // Optional iterative lift: blend SB label with current expert preference so
    // later δ iterations imitate an improved oracle (paper LLB_δ spirit).
    f64 label = sb_score;
    if (expert.loaded) {
        const f64 pred = expert.predict(feats);
        if (std::isfinite(pred))
            label = 0.7 * sb_score + 0.3 * std::max(0.0, pred);
    }
    buffer.add(feats, label);
    if (buffer.samples.size() > before) ++samples_since_fit;
}

LiftedSbModel fit_lifted_sb(const std::vector<LiftedSbSample>& samples,
                            const LiftedBranchOptions& opts) {
    LiftedSbModel model;
    model.clear();
    if (samples.size() < 2) return model;

    const int p = kLiftedFeatureDim;
    const bool quad = opts.use_quadratic;
    const int d = expanded_dim(p, quad);
    const std::size_t n = samples.size();

    std::vector<std::vector<f64>> X(n);
    for (std::size_t r = 0; r < n; ++r)
        expand_lifted(samples[r].feats, quad, p, X[r]);

    std::vector<f64> beta(static_cast<std::size_t>(d), 0.0);
    f64 intercept = 0.0;

    if (opts.loss == SparseSbLoss::Lasso) {
        std::vector<f64> y(n, 0.0);
        f64 y_norm2 = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            y[r] = samples[r].score;
            y_norm2 += y[r] * y[r];
        }
        if (y_norm2 > 0.0) {
            const f64 inv = 1.0 / std::sqrt(y_norm2);
            for (f64& v : y) v *= inv;
        }
        for (f64 v : y) intercept += v;
        intercept /= static_cast<f64>(n);
        std::vector<f64> residual(n);
        for (std::size_t r = 0; r < n; ++r) residual[r] = y[r] - intercept;
        std::vector<f64> col_norm2(static_cast<std::size_t>(d), 0.0);
        for (int j = 0; j < d; ++j) {
            f64 s = 0.0;
            for (std::size_t r = 0; r < n; ++r)
                s += X[r][static_cast<std::size_t>(j)] *
                     X[r][static_cast<std::size_t>(j)];
            col_norm2[static_cast<std::size_t>(j)] = s;
        }
        const f64 lam = std::max(0.0, opts.lasso_lambda);
        for (int it = 0; it < opts.lasso_max_iter; ++it) {
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
                        residual[r] -=
                            bj_new * X[r][static_cast<std::size_t>(j)];
                }
            }
        }
        model.loss = SparseSbLoss::Lasso;
    } else {
        // Pairwise ranking on lifted features (imitation of lifted SB scores).
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        const int cap = 12000;
        for (std::size_t i = 0; i < n && static_cast<int>(pairs.size()) < cap;
             ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const f64 di = samples[i].score - samples[j].score;
                if (std::fabs(di) < 1e-12) continue;
                if (di > 0.0)
                    pairs.emplace_back(i, j);
                else
                    pairs.emplace_back(j, i);
                if (static_cast<int>(pairs.size()) >= cap) break;
            }
        }
        const f64 lr = std::max(1e-4, opts.ranking_lr);
        const f64 l1 = std::max(0.0, opts.lasso_lambda);
        for (int ep = 0; ep < opts.ranking_epochs; ++ep) {
            for (const auto& pr : pairs) {
                f64 s_pos = intercept;
                f64 s_neg = intercept;
                for (int k = 0; k < d; ++k) {
                    s_pos += beta[static_cast<std::size_t>(k)] *
                             X[pr.first][static_cast<std::size_t>(k)];
                    s_neg += beta[static_cast<std::size_t>(k)] *
                             X[pr.second][static_cast<std::size_t>(k)];
                }
                const f64 g = 1.0 - sigmoid(s_pos - s_neg);
                intercept += lr * g;
                for (int k = 0; k < d; ++k) {
                    const f64 dx = X[pr.first][static_cast<std::size_t>(k)] -
                                   X[pr.second][static_cast<std::size_t>(k)];
                    beta[static_cast<std::size_t>(k)] += lr * g * dx;
                }
            }
            if (l1 > 0.0)
                for (f64& b : beta) b = soft_threshold(b, l1);
        }
        model.loss = SparseSbLoss::Ranking;
    }

    std::vector<std::pair<f64, int>> ranked;
    for (int j = 0; j < d; ++j) {
        const f64 c = beta[static_cast<std::size_t>(j)];
        if (std::fabs(c) > 1e-14) ranked.emplace_back(std::fabs(c), j);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (opts.max_nonzero > 0 &&
        static_cast<int>(ranked.size()) > opts.max_nonzero)
        ranked.resize(static_cast<std::size_t>(opts.max_nonzero));

    model.dim = p;
    model.quadratic = quad;
    model.intercept = intercept;
    for (const auto& pr : ranked)
        push_term(pr.second, p, quad, beta[static_cast<std::size_t>(pr.second)],
                  model.terms);
    model.loaded = !model.terms.empty() || std::fabs(intercept) > 0.0;
    return model;
}

bool LiftedBranchState::maybe_refit(const LiftedBranchOptions& opts) {
    if (static_cast<int>(buffer.samples.size()) < opts.min_samples)
        return false;
    if (samples_since_fit < opts.refit_every && expert.loaded) return false;

    LiftedSbModel next = fit_lifted_sb(buffer.samples, opts);
    if (!next.loaded) return false;
    expert = std::move(next);
    samples_since_fit = 0;
    ++iterations;
    return true;
}

bool save_lifted_sb_model(const std::string& path, const LiftedSbModel& model) {
    std::ofstream out(path);
    if (!out) return false;
    out << "SOR_LIFTED_SB 1\n";
    out << "dim " << model.dim << "\n";
    out << "quadratic " << (model.quadratic ? 1 : 0) << "\n";
    out << "loss " << (model.loss == SparseSbLoss::Ranking ? "ranking" : "lasso")
        << "\n";
    out << "intercept " << model.intercept << "\n";
    out << "nnz " << model.terms.size() << "\n";
    for (const LiftedSbTerm& t : model.terms)
        out << t.i << " " << t.j << " " << t.coef << "\n";
    return static_cast<bool>(out);
}

bool load_lifted_sb_model(const std::string& path, LiftedSbModel& model) {
    model.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string tag;
    int ver = 0;
    if (!(in >> tag >> ver) || tag != "SOR_LIFTED_SB" || ver != 1) {
        // Fallback: try Sparse-SB and project into branch-prefix of lifted vec.
        SparseSbModel sb;
        if (!load_sparse_sb_model(path, sb) || !sb.loaded) return false;
        model.dim = kLiftedFeatureDim;
        model.quadratic = sb.quadratic;
        model.loss = sb.loss;
        model.intercept = sb.intercept;
        for (const auto& t : sb.terms) {
            if (t.i >= kBranchFeatureDim) continue;
            if (t.j >= kBranchFeatureDim) continue;
            model.terms.push_back({t.i, t.j, t.coef});
        }
        model.loaded = !model.terms.empty() || std::fabs(model.intercept) > 0.0;
        return model.loaded;
    }
    std::string key;
    int nnz = 0;
    while (in >> key) {
        if (key == "dim") {
            in >> model.dim;
        } else if (key == "quadratic") {
            int q = 0;
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
    if (model.dim <= 0 || model.dim > kLiftedFeatureDim) return false;
    model.terms.resize(static_cast<std::size_t>(std::max(0, nnz)));
    for (int k = 0; k < nnz; ++k) {
        LiftedSbTerm t;
        if (!(in >> t.i >> t.j >> t.coef)) return false;
        model.terms[static_cast<std::size_t>(k)] = t;
    }
    model.loaded = true;
    return true;
}

Index pick_lifted_branch(const LiftedBranchState& state,
                         const std::vector<Index>& candidates,
                         const std::vector<LiftedFeatureVec>& feats) {
    if (!state.ready() || candidates.empty() ||
        candidates.size() != feats.size())
        return -1;
    Index best = -1;
    f64 best_s = -std::numeric_limits<f64>::infinity();
    for (std::size_t k = 0; k < candidates.size(); ++k) {
        const f64 s = state.expert.predict(feats[k]);
        if (s > best_s) {
            best_s = s;
            best = candidates[k];
        }
    }
    return best;
}

}  // namespace sor::search
