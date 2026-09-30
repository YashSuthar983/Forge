#include "binary_substitution.hpp"
#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace sor::search::detail {
namespace {
// Error-free TwoSum: reject a rewrite when its exact sum is not binary64.
// Keep the original row when a rewrite would round an inequality inwards.
// No feasibility tolerance is used to justify coefficient deletion.
bool exact_add(double a, double b, double &sum) {
    sum = a + b;
    if (!std::isfinite(sum))
        return false;
    const double bv = sum - a, av = sum - bv;
    return (a - av) + (b - bv) == 0.0;
}
} // namespace

MilpPresolveResult substitute_binary_relations(const model::LpProblem &lp,
                                               const std::vector<BinaryRelation> &relations,
                                               MilpPresolveStats &stats) {
    const Index n = lp.n_cols();
    auto identity = [&]() {
        MilpPresolveResult out;
        out.reduced = lp;
        out.reduced_col.resize(static_cast<std::size_t>(n));
        std::iota(out.reduced_col.begin(), out.reduced_col.end(), 0);
        out.eliminated.assign(static_cast<std::size_t>(n), 0);
        out.fixed_value.assign(static_cast<std::size_t>(n), 0.0);
        return out;
    };
    auto rejected = [&]() {
        ++stats.binary_substitution_numerical_rejects;
        return identity();
    };
    if (relations.empty())
        return identity();
    std::vector<Index> parent(static_cast<std::size_t>(n));
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<char> parity(static_cast<std::size_t>(n), 0);
    auto find = [&](Index j) {
        Index r = j;
        int value = 0;
        while (parent[r] != r) {
            value ^= parity[r];
            r = parent[r];
        }
        const int result = value;
        while (parent[j] != j) {
            const Index next = parent[j];
            const int edge = parity[j];
            parent[j] = r;
            parity[j] = static_cast<char>(value);
            value ^= edge;
            j = next;
        }
        return std::pair<Index, int>{r, result};
    };
    const auto binary = [&](Index j) {
        return j >= 0 && j < n && !lp.is_integer.empty() && lp.is_integer[j] &&
               lp.col_lo[j] >= 0.0 && lp.col_hi[j] <= 1.0;
    };
    for (const auto &relation : relations) {
        if (!binary(relation.first) || !binary(relation.second))
            continue;
        const auto [a, pa] = find(relation.first);
        const auto [b, pb] = find(relation.second);
        const int edge = pa ^ pb ^ static_cast<int>(relation.complement);
        if (a == b) {
            if (edge) {
                auto out = identity();
                out.infeasible = true;
                return out;
            }
            continue;
        }
        const Index drop = std::max(a, b), keep = std::min(a, b);
        parent[drop] = keep;
        parity[drop] = static_cast<char>(edge);
    }
    std::vector<Index> root(static_cast<std::size_t>(n)), remap(static_cast<std::size_t>(n), -1);
    std::vector<int> members(static_cast<std::size_t>(n), 0),
        domain(static_cast<std::size_t>(n), 3);
    Index nr = 0;
    for (Index j = 0; j < n; ++j) {
        const auto [r, p] = find(j);
        root[j] = r;
        parity[j] = static_cast<char>(p);
        ++members[r];
        if (r == j)
            remap[j] = nr++;
        if (binary(j)) {
            int allowed = (lp.col_lo[j] <= 0.0 && lp.col_hi[j] >= 0.0 ? 1 : 0) |
                          (lp.col_lo[j] <= 1.0 && lp.col_hi[j] >= 1.0 ? 2 : 0);
            if (p)
                allowed = ((allowed & 1) << 1) | ((allowed & 2) >> 1);
            domain[r] &= allowed;
        }
    }
    if (nr == n)
        return identity();
    for (Index j = 0; j < n; ++j)
        if (root[j] == j && members[j] > 1 && domain[j] == 0) {
            auto out = identity();
            out.infeasible = true;
            return out;
        }

    MilpPresolveResult out;
    out.reduced = lp;
    auto &reduced = out.reduced;
    reduced.c.assign(static_cast<std::size_t>(nr), 0.0);
    reduced.col_lo.resize(static_cast<std::size_t>(nr));
    reduced.col_hi.resize(static_cast<std::size_t>(nr));
    reduced.is_integer.resize(static_cast<std::size_t>(nr));
    if (!lp.col_names.empty())
        reduced.col_names.resize(static_cast<std::size_t>(nr));
    out.reduced_col.assign(static_cast<std::size_t>(n), -1);
    out.eliminated.assign(static_cast<std::size_t>(n), 0);
    out.fixed_value.assign(static_cast<std::size_t>(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        const Index r = root[j], k = remap[r];
        if (parity[j] && !exact_add(reduced.obj_offset, lp.c[j], reduced.obj_offset))
            return rejected();
        if (!exact_add(reduced.c[k], parity[j] ? -lp.c[j] : lp.c[j], reduced.c[k]))
            return rejected();
        if (r != j) {
            out.eliminated[j] = 1;
            out.binary_substitution_steps.push_back({j, r, parity[j] != 0});
        } else {
            out.reduced_col[j] = k;
            reduced.col_lo[k] = lp.col_lo[j];
            reduced.col_hi[k] = lp.col_hi[j];
            if (members[j] > 1) {
                reduced.col_lo[k] = domain[j] == 2 ? 1.0 : 0.0;
                reduced.col_hi[k] = domain[j] == 1 ? 0.0 : 1.0;
            }
            reduced.is_integer[k] = lp.is_integer[j];
            if (!lp.col_names.empty())
                reduced.col_names[k] = lp.col_names[j];
        }
    }
    std::vector<Index> ti, tj;
    std::vector<double> tv;
    const auto &rp = lp.A.pattern.row_ptr();
    const auto &ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < lp.n_rows(); ++i) {
        std::vector<std::pair<Index, double>> row;
        row.reserve(static_cast<std::size_t>(rp[i + 1] - rp[i]));
        double shift = 0.0;
        for (auto t = rp[i]; t < rp[i + 1]; ++t) {
            const Index j = ci[t];
            const double a = lp.A.vals[t];
            if (parity[j] && !exact_add(shift, a, shift))
                return rejected();
            row.emplace_back(remap[root[j]], parity[j] ? -a : a);
        }
        std::sort(row.begin(), row.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        for (std::size_t t = 0; t < row.size();) {
            const Index j = row[t].first;
            double value = 0.0;
            do {
                if (!exact_add(value, row[t++].second, value))
                    return rejected();
            } while (t < row.size() && row[t].first == j);
            if (value != 0.0) {
                ti.push_back(i);
                tj.push_back(j);
                tv.push_back(value);
            }
        }
        if (std::isfinite(reduced.row_lo[i]) && !exact_add(lp.row_lo[i], -shift, reduced.row_lo[i]))
            return rejected();
        if (std::isfinite(reduced.row_hi[i]) && !exact_add(lp.row_hi[i], -shift, reduced.row_hi[i]))
            return rejected();
    }
    reduced.A = sparse::from_triplets(lp.n_rows(), nr, ti, tj, tv);
    stats.binary_substitutions += n - nr;
    return out;
}
} // namespace sor::search::detail
