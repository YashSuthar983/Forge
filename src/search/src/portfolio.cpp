#include "sor/search/portfolio.hpp"

#include <cmath>
#include <cstddef>

namespace sor::search {

using core::Index;

bool PortfolioPool::better(f64 candidate, f64 current) const {
    if (!std::isfinite(candidate)) return false;
    if (!std::isfinite(current)) return true;  // nothing published yet
    return maximize_ ? candidate > current : candidate < current;
}

bool PortfolioPool::publish_incumbent(f64 objective,
                                      const std::vector<f64>& x) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!better(objective, best_obj_)) return false;
    best_obj_ = objective;
    best_x_ = x;
    ++updates_;
    return true;
}

f64 PortfolioPool::best_objective() const {
    std::lock_guard<std::mutex> lock(mu_);
    return best_obj_;
}

std::vector<f64> PortfolioPool::best_point() const {
    std::lock_guard<std::mutex> lock(mu_);
    return best_x_;
}

void PortfolioPool::publish_dual_bound(f64 bound) {
    std::lock_guard<std::mutex> lock(mu_);
    // The best dual bound is the TIGHTEST one, i.e. the largest for a
    // minimisation problem -- the opposite direction to the incumbent.
    if (!std::isfinite(bound)) return;
    if (!std::isfinite(best_dual_) ||
        (maximize_ ? bound < best_dual_ : bound > best_dual_)) {
        best_dual_ = bound;
    }
}

f64 PortfolioPool::best_dual_bound() const {
    std::lock_guard<std::mutex> lock(mu_);
    return best_dual_;
}

std::uint64_t PortfolioPool::incumbent_updates() const {
    std::lock_guard<std::mutex> lock(mu_);
    return updates_;
}

// Is `x` a genuinely feasible integer point OF THIS PROBLEM?
//
// Used to gate what a portfolio arm may publish. Arms run with different
// presolve/implied-integrality settings, so each holds its OWN transformed
// copy of the model; a point that is valid in one arm's space is meaningless
// in another's. Validating against the arm's internal copy let blend2 publish
// a point that scored -33.0 in the caller's space against a true optimum of
// 7.598985, and the portfolio then reported it as Optimal.
//
// So the pool is a CROSS-ARM channel and its entry check must be done in the
// one space every arm shares: the caller's original problem. Bounds, rows and
// integrality are all re-checked -- an objective match alone proves nothing,
// since a point feasible under one arm's tightened bounds can score better
// than the true optimum under the original ones.
bool portfolio_point_is_feasible(const model::LpProblem& p,
                                  const std::vector<f64>& x,
                                  f64 feas_tol, f64 int_tol) {
    const Index n = p.n_cols();
    if (static_cast<Index>(x.size()) != n) return false;
    for (Index j = 0; j < n; ++j) {
        const f64 v = x[static_cast<std::size_t>(j)];
        if (!std::isfinite(v)) return false;
        const f64 lo = p.col_lo[static_cast<std::size_t>(j)], hi = p.col_hi[static_cast<std::size_t>(j)];
        const f64 scale = 1.0 + std::fabs(v);
        if (std::isfinite(lo) && v < lo - feas_tol * scale) return false;
        if (std::isfinite(hi) && v > hi + feas_tol * scale) return false;
        if (!p.is_integer.empty() && p.is_integer[static_cast<std::size_t>(j)] &&
            std::fabs(v - std::round(v)) > int_tol)
            return false;
    }
    const Index m = p.n_rows();
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    for (Index i = 0; i < m; ++i) {
        f64 act = 0.0;
        for (core::Offset k = rp[static_cast<std::size_t>(i)]; k < rp[static_cast<std::size_t>(i) + 1]; ++k)
            act += p.A.vals[static_cast<std::size_t>(k)] * x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
        if (!std::isfinite(act)) return false;
        const f64 scale = 1.0 + std::fabs(act);
        const f64 lo = p.row_lo[static_cast<std::size_t>(i)], hi = p.row_hi[static_cast<std::size_t>(i)];
        if (std::isfinite(lo) && act < lo - feas_tol * scale) return false;
        if (std::isfinite(hi) && act > hi + feas_tol * scale) return false;
    }
    return true;
}

// Does `cut` admit the point `x`?
//
// A globally valid cut must be satisfied by EVERY integer-feasible point, so
// in particular by any incumbent. This is the cheapest universal check there
// is against an invalid separator: it does not care which family produced the
// cut, or by what derivation, only whether the cut contradicts a point we
// already know is feasible. O(cut nnz).
//
// It cannot catch a cut that removes an optimum we have not found yet -- root
// cuts are generated before any incumbent exists -- so it is a net, not a
// proof. But an invalid cut that survives to the global pool while an
// incumbent is in hand is exactly the failure that produces a false Optimal,
// and this refuses it.
bool cut_admits_point(const std::vector<Index>& cols,
                      const std::vector<f64>& vals, f64 row_lo, f64 row_hi,
                      const std::vector<f64>& x, f64 tol) {
    if (x.empty()) return true;  // nothing to check against
    f64 act = 0.0;
    for (std::size_t k = 0; k < cols.size(); ++k) {
        const std::size_t j = static_cast<std::size_t>(cols[k]);
        if (j >= x.size()) return true;  // shape mismatch: not ours to judge
        act += vals[k] * x[j];
    }
    if (!std::isfinite(act)) return true;
    const f64 scale = tol * (1.0 + std::fabs(act));
    if (std::isfinite(row_lo) && act < row_lo - scale) return false;
    if (std::isfinite(row_hi) && act > row_hi + scale) return false;
    return true;
}

}  // namespace sor::search
