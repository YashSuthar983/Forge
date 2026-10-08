#include "forge/search/mixing.hpp"

#include "forge/core/work_clock.hpp"
#include "forge/model/dyadic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace forge::search {
namespace {
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

bool is_binary_col(const model::LpProblem& lp, Index j) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)] && lp.col_lo[sz(j)] == 0.0 &&
           lp.col_hi[sz(j)] == 1.0;
}
}  // namespace

MixingIndex build_mixing_index(const model::LpProblem& lp) {
    MixingIndex out;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    // z -> (literal -> best threshold)
    std::map<Index, std::map<Index, f64>> by_z;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        if (rp[sz(i) + 1] - rp[sz(i)] != 2) continue;
        const bool ge = std::isfinite(lp.row_lo[sz(i)]) && !std::isfinite(lp.row_hi[sz(i)]);
        const bool le = std::isfinite(lp.row_hi[sz(i)]) && !std::isfinite(lp.row_lo[sz(i)]);
        if (!ge && !le) continue;
        const f64 sg = ge ? 1.0 : -1.0;
        const f64 r = sg * (ge ? lp.row_lo[sz(i)] : lp.row_hi[sz(i)]);
        for (int zk = 0; zk < 2; ++zk) {
            const core::Offset kz = rp[sz(i)] + zk, ky = rp[sz(i)] + (1 - zk);
            const Index z = ci[sz(kz)], y = ci[sz(ky)];
            const f64 a = sg * av[sz(kz)], b = sg * av[sz(ky)];
            if (z == y || !(a > 0.0) || b == 0.0) continue;
            if (!is_binary_col(lp, y) || is_binary_col(lp, z)) continue;
            if (!(lp.col_lo[sz(z)] >= 0.0)) continue;
            model::DyadicSum num;
            Index lit = -1;
            if (b > 0.0 && r > 0.0 && r <= b) {          // z >= (r/a)(1 - y)
                num.add(r);
                lit = 2 * y + 1;
            } else if (b < 0.0 && r == 0.0) {            // z >= (|b|/a) y = h (1 - (1 - y))
                num.add(-b);
                lit = 2 * y;
            } else {
                continue;
            }
            const f64 h = num.quotient_down(a);
            if (!(h > 0.0) || !std::isfinite(h)) continue;
            f64& best = by_z[z][lit];
            best = std::max(best, h);
        }
    }
    for (auto& [z, lits] : by_z) {
        if (lits.size() < 2) continue;
        MixingFamily f;
        f.z = z;
        std::vector<std::pair<f64, Index>> v;
        for (const auto& [l, h] : lits) v.emplace_back(h, l);
        std::sort(v.begin(), v.end());
        for (const auto& [h, l] : v) {
            f.h.push_back(h);
            f.lit.push_back(l);
        }
        out.families.push_back(std::move(f));
    }
    return out;
}

std::vector<CutRow> separate_mixing_cuts(const MixingIndex& index, const std::vector<f64>& x,
                                         int max_cuts, f64 violation_min) {
    core::WorkScope work_scope_(core::WorkCat::Separation);
    std::vector<std::pair<f64, CutRow>> found;
    if (max_cuts <= 0) return {};
    std::vector<f64> F, v;
    std::vector<Index> pred;
    for (const MixingFamily& f : index.families) {
        const std::size_t k = f.lit.size();
        if (sz(f.z) >= x.size()) continue;
        core::work_add(static_cast<std::uint64_t>(k * k), core::WorkTag::Untagged);
        v.assign(k, 0.0);
        for (std::size_t i = 0; i < k; ++i) {
            const Index col = f.lit[i] >> 1;
            if (sz(col) >= x.size()) { v.clear(); break; }
            const f64 lv = (f.lit[i] & 1) ? x[sz(col)] : 1.0 - x[sz(col)];
            v[i] = std::clamp(1.0 - lv, 0.0, 1.0);
        }
        if (v.empty()) continue;
        F.assign(k, 0.0);
        pred.assign(k, -1);
        std::size_t best = k;
        f64 best_val = -kInf;
        for (std::size_t i = 0; i < k; ++i) {
            F[i] = f.h[i] * v[i];
            for (std::size_t j = 0; j < i; ++j) {
                const f64 cand = F[j] + (f.h[i] - f.h[j]) * v[i];
                if (cand > F[i]) { F[i] = cand; pred[i] = static_cast<Index>(j); }
            }
            if (F[i] > best_val) { best_val = F[i]; best = i; }
        }
        if (best == k) continue;
        // Recover the chain (ascending h).
        std::vector<std::size_t> chain;
        for (Index t = static_cast<Index>(best); t >= 0; t = pred[sz(t)]) chain.push_back(sz(t));
        std::reverse(chain.begin(), chain.end());
        // z + sum c_t l_t >= h_last with c_t rounded up; a complemented
        // literal (l = 1 - y) moves its c_t to the side, rounded down.
        CutRow cut;
        cut.cols.push_back(f.z);
        cut.vals.push_back(1.0);
        model::DyadicSum side;
        side.add(f.h[chain.back()]);
        f64 prev = 0.0;
        f64 act = x[sz(f.z)];
        bool ok = true;
        for (const std::size_t t : chain) {
            model::DyadicSum d;
            d.add(f.h[t]);
            d.add(-prev);
            const f64 c = d.up();
            prev = f.h[t];
            if (!(c >= 0.0) || !std::isfinite(c)) { ok = false; break; }
            if (c == 0.0) continue;
            const Index col = f.lit[t] >> 1;
            if (f.lit[t] & 1) {                 // l = y
                cut.cols.push_back(col);
                cut.vals.push_back(c);
                act += c * x[sz(col)];
            } else {                            // l = 1 - y
                cut.cols.push_back(col);
                cut.vals.push_back(-c);
                side.add(-c);
                act -= c * x[sz(col)];
            }
        }
        if (!ok || cut.cols.size() < 3) continue;   // a chain of one is the row itself
        cut.row_lo = side.down();
        cut.row_hi = kInf;
        f64 norm2 = 0.0;
        for (const f64 a : cut.vals) norm2 += a * a;
        const f64 viol = (cut.row_lo - act) / std::sqrt(norm2);
        if (!(viol > violation_min)) continue;
        cut.exact = true;
        cut.used_local_bound = false;   // the model's own rows and bounds
        found.emplace_back(viol, std::move(cut));
    }
    std::sort(found.begin(), found.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<CutRow> out;
    for (auto& [viol, c] : found) {
        if (static_cast<int>(out.size()) >= max_cuts) break;
        (void)viol;
        c.name = "MIX_" + std::to_string(out.size());
        out.push_back(std::move(c));
    }
    return out;
}

}  // namespace forge::search
