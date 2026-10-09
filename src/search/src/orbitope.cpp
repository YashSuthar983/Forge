#include "forge/search/domain_writer.hpp"
#include "forge/search/orbitope.hpp"

#include "forge/core/work_clock.hpp"
#include "forge/search/prop_trail.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <unordered_map>

namespace forge::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
using Clock = core::SolverClock;

std::uint64_t mix(std::uint64_t h, std::uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h * 0xff51afd7ed558ccdull;
}
std::uint64_t bits(f64 v) {
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return v == 0.0 ? 0 : b;   // +0 and -0 hash alike
}

struct UnionFind {
    std::vector<Index> p;
    explicit UnionFind(std::size_t n) : p(n) { std::iota(p.begin(), p.end(), 0); }
    Index find(Index x) {
        while (p[sz(x)] != x) { p[sz(x)] = p[sz(p[sz(x)])]; x = p[sz(x)]; }
        return x;
    }
    void unite(Index a, Index b) { a = find(a); b = find(b); if (a != b) p[sz(a)] = b; }
};

// One row as data: sides and its (column, coefficient) list sorted by column.
struct RowKey {
    f64 lo = 0.0, hi = 0.0;
    std::vector<std::pair<Index, f64>> entries;
    bool operator==(const RowKey& o) const {
        return lo == o.lo && hi == o.hi && entries == o.entries;
    }
};
std::uint64_t hash_row(const RowKey& r) {
    std::uint64_t h = mix(bits(r.lo), bits(r.hi));
    for (const auto& [c, v] : r.entries) h = mix(mix(h, static_cast<std::uint64_t>(c)), bits(v));
    return h;
}

}  // namespace

std::vector<Orbitope> detect_orbitopes(const model::LpProblem& lp,
                                       const std::vector<Orbit>& classes,
                                       const OrbitopeOptions& opts,
                                       OrbitopeDiagnostics& diag) {
    core::WorkScope work_scope_(core::WorkCat::Symmetry);
    const auto t0 = Clock::now();
    std::vector<Orbitope> out;
    if (!opts.enabled || lp.is_integer.empty()) return out;
    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    const auto binary = [&](Index j) {
        return lp.is_integer[sz(j)] && lp.col_lo[sz(j)] == 0.0 && lp.col_hi[sz(j)] == 1.0;
    };
    const auto over_time = [&]() {
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - t0).count() > opts.time_limit_s;
    };
    // Every row as data, for the image lookups of verification.
    std::vector<RowKey> rows(sz(m));
    std::unordered_multimap<std::uint64_t, Index> by_hash;
    by_hash.reserve(sz(m) * 2);
    std::vector<std::vector<Index>> col_rows(sz(n));
    for (Index i = 0; i < m; ++i) {
        RowKey& r = rows[sz(i)];
        r.lo = lp.row_lo[sz(i)];
        r.hi = lp.row_hi[sz(i)];
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            if (av[sz(k)] != 0.0) r.entries.emplace_back(ci[sz(k)], av[sz(k)]);
        std::sort(r.entries.begin(), r.entries.end());
        for (const auto& e : r.entries) col_rows[sz(e.first)].push_back(i);
        by_hash.emplace(hash_row(r), i);
    }
    core::work_add(4 * static_cast<std::uint64_t>(lp.nnz()), core::WorkTag::Symmetry);
    // Color of every column (color refinement); columns in no class of two or
    // more are alone in theirs.
    std::vector<Index> color(sz(n), -1);
    {
        Index next = 0;
        for (const Orbit& o : classes) {
            for (const Index j : o.cols)
                if (j >= 0 && j < n) color[sz(j)] = next;
            ++next;
        }
        for (Index j = 0; j < n; ++j)
            if (color[sz(j)] < 0) color[sz(j)] = next++;
    }
    // Candidate orbitope rows: packing / partitioning rows (unit
    // coefficients, sides <= 1 and >= 1 or open) over k >= 2 binaries of one
    // color. The block structure is then read off the other rows. Grouped by
    // k, smallest first.
    struct Cand { Index row; bool partition; };
    std::unordered_map<std::size_t, std::vector<Cand>> by_k;
    for (Index i = 0; i < m; ++i) {
        const RowKey& r = rows[sz(i)];
        const std::size_t k = r.entries.size();
        if (k < static_cast<std::size_t>(std::max(2, opts.min_blocks))) continue;
        if (!(r.hi <= 1.0) || r.hi < 1.0) continue;          // sum <= 1 exactly
        bool ok = true;
        for (const auto& [c, v] : r.entries)
            ok = ok && v == 1.0 && binary(c) && color[sz(c)] == color[sz(r.entries[0].first)];
        if (!ok) continue;
        by_k[k].push_back({i, r.lo >= 1.0});
    }
    std::vector<std::size_t> sizes;
    for (const auto& kv : by_k) sizes.push_back(kv.first);
    std::sort(sizes.begin(), sizes.end());
    std::vector<char> used(sz(n), 0);   // a column belongs to at most one orbitope
    std::uint64_t work = 0;

    for (const std::size_t k : sizes) {
        if (over_time() || work > opts.max_verify_work) break;
        auto& cands = by_k[k];
        ++diag.candidate_class_groups;
        // Candidate rows sharing a column cannot both be orbitope rows; keep
        // the first.
        std::vector<Index> cand_of(sz(n), -1);
        std::vector<Cand> kept;
        for (const Cand& cd : cands) {
            bool clash = false;
            for (const auto& e : rows[sz(cd.row)].entries)
                clash = clash || cand_of[sz(e.first)] >= 0 || used[sz(e.first)];
            if (clash) continue;
            for (const auto& e : rows[sz(cd.row)].entries)
                cand_of[sz(e.first)] = static_cast<Index>(kept.size());
            kept.push_back(cd);
        }
        if (kept.empty()) continue;
        // Columns that share a row holding at most one column of each
        // candidate row lie in one block.
        UnionFind uf(sz(n));
        std::vector<Index> seen;
        for (Index i = 0; i < m; ++i) {
            const RowKey& r = rows[sz(i)];
            seen.clear();
            Index first = -1;
            bool block_row = true;
            int members = 0;
            for (const auto& e : r.entries) {
                const Index c = cand_of[sz(e.first)];
                if (c < 0) continue;
                if (std::find(seen.begin(), seen.end(), c) != seen.end()) { block_row = false; break; }
                seen.push_back(c);
                ++members;
                if (first < 0) first = e.first;
            }
            work += r.entries.size();
            core::work_add(r.entries.size(), core::WorkTag::Symmetry);
            if (!block_row || members < 1) continue;
            // Every column of such a row travels with that block (a bin's
            // capacity row carries its bin-used indicator too). A row with a
            // single candidate column still ties its other columns to it.
            for (const auto& e : r.entries) uf.unite(first, e.first);
        }
        // A candidate qualifies when its k columns fall in k distinct blocks
        // and every qualifying candidate uses the same k blocks.
        std::vector<Index> blocks;
        std::vector<std::vector<Index>> var;
        std::vector<char> var_partition;
        for (const Cand& cd : kept) {
            const auto& es = rows[sz(cd.row)].entries;
            std::vector<Index> comp(es.size());
            for (std::size_t q = 0; q < es.size(); ++q) comp[q] = uf.find(es[q].first);
            std::vector<Index> sorted = comp;
            std::sort(sorted.begin(), sorted.end());
            if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) continue;
            if (blocks.empty()) blocks = comp;
            std::vector<Index> row(k, -1);
            bool fits = true;
            for (std::size_t q = 0; q < es.size() && fits; ++q) {
                const auto it = std::find(blocks.begin(), blocks.end(), comp[q]);
                if (it == blocks.end()) fits = false;
                else row[static_cast<std::size_t>(it - blocks.begin())] = es[q].first;
            }
            if (!fits) continue;
            var.push_back(std::move(row));
            var_partition.push_back(cd.partition ? 1 : 0);
        }
        if (var.empty()) { ++diag.blocks_rejected; continue; }
        // The blocks' other columns (e.g. a bin-used indicator): a block
        // transposition must move them too. Every column joined to a block
        // that is not in an orbitope row travels with its block; it is
        // paired with the column of the same color in the other block.
        std::vector<std::vector<Index>> extra(k);   // per block, sorted by color then index
        {
            std::vector<Index> block_of_root(sz(n), -1);
            for (std::size_t b = 0; b < k; ++b) block_of_root[sz(blocks[b])] = static_cast<Index>(b);
            for (Index j = 0; j < n; ++j) {
                if (cand_of[sz(j)] >= 0) continue;
                const Index root = uf.find(j);
                if (root == j && block_of_root[sz(j)] < 0) continue;   // a singleton set
                const Index b = block_of_root[sz(root)];
                if (b >= 0) extra[sz(b)].push_back(j);
            }
            for (auto& e : extra)
                std::sort(e.begin(), e.end(), [&](Index a, Index c) {
                    return color[sz(a)] != color[sz(c)] ? color[sz(a)] < color[sz(c)] : a < c;
                });
        }
        bool extras_match = true;
        for (std::size_t b = 1; b < k && extras_match; ++b) {
            if (extra[b].size() != extra[0].size()) { extras_match = false; break; }
            for (std::size_t q = 0; q < extra[b].size(); ++q)
                extras_match = extras_match && color[sz(extra[b][q])] == color[sz(extra[0][q])];
        }
        // Existing canonical orders may only be combined for disjoint action
        // supports. Marked extras travel with earlier blocks too, even when
        // they do not appear in an orbitope's displayed packing rows.
        for (const auto& e : extra) for (Index j : e)
            if (used[sz(j)]) extras_match = false;
        if (!extras_match) { ++diag.blocks_rejected; continue; }
        // Verify every adjacent block transposition as an automorphism.
        bool verified = true;
        std::vector<Index> image(sz(n), -1);
        RowKey img;
        for (std::size_t b = 0; b + 1 < k && verified; ++b) {
            std::vector<Index> moved;
            const auto pair = [&](Index u, Index v) {
                image[sz(u)] = v;
                image[sz(v)] = u;
                moved.push_back(u);
                moved.push_back(v);
            };
            for (const auto& row : var) pair(row[b], row[b + 1]);
            for (std::size_t q = 0; q < extra[b].size(); ++q) pair(extra[b][q], extra[b + 1][q]);
            for (const Index j : moved) {
                const Index t = image[sz(j)];
                if (lp.c[sz(j)] != lp.c[sz(t)] || lp.col_lo[sz(j)] != lp.col_lo[sz(t)] ||
                    lp.col_hi[sz(j)] != lp.col_hi[sz(t)] ||
                    lp.is_integer[sz(j)] != lp.is_integer[sz(t)])
                    verified = false;
            }
            std::vector<Index> touched;
            for (const Index j : moved)
                for (const Index i : col_rows[sz(j)]) touched.push_back(i);
            std::sort(touched.begin(), touched.end());
            touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
            for (const Index i : touched) {
                if (!verified) break;
                const RowKey& r = rows[sz(i)];
                img.lo = r.lo;
                img.hi = r.hi;
                img.entries.clear();
                for (const auto& [c, v] : r.entries)
                    img.entries.emplace_back(image[sz(c)] >= 0 ? image[sz(c)] : c, v);
                std::sort(img.entries.begin(), img.entries.end());
                work += 2 * img.entries.size();
                core::work_add(2 * img.entries.size(), core::WorkTag::Symmetry);
                bool found = false;
                const auto range = by_hash.equal_range(hash_row(img));
                for (auto it = range.first; it != range.second && !found; ++it)
                    found = rows[sz(it->second)] == img;
                if (!found) verified = false;
            }
            for (const Index j : moved) image[sz(j)] = -1;
            if (verified) ++diag.transpositions_verified;
            else ++diag.transpositions_refuted;
            if (work > opts.max_verify_work || over_time()) verified = false;
        }
        if (!verified) continue;
        Orbitope orb;
        orb.var = std::move(var);
        orb.partitioning = std::all_of(var_partition.begin(), var_partition.end(),
                                       [](char c) { return c != 0; });
        // The lexicographic rule reads rows in a fixed order; the original
        // column order of each row's first block is as good as any.
        std::sort(orb.var.begin(), orb.var.end(),
                  [](const std::vector<Index>& a, const std::vector<Index>& b) { return a[0] < b[0]; });
        for (const auto& row : orb.var)
            for (const Index j : row) used[sz(j)] = 1;
        for (const auto& e : extra)
            for (const Index j : e) used[sz(j)] = 1;
        ++diag.orbitopes;
        diag.orbitope_rows += orb.var.size();
        diag.orbitope_blocks += k;
        out.push_back(std::move(orb));
    }

    diag.ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return out;
}

bool orbitopal_fixing(const Orbitope& orb, std::vector<f64>& lo, std::vector<f64>& hi,
                      PropTrail* trail, int depth, std::vector<Index>* changed,
                      std::uint64_t& fixed) {
    // alpha_max: the furthest column any earlier row can use (-1: none).
    Index alpha_max = -1;
    for (const auto& row : orb.var) {
        const Index k = static_cast<Index>(row.size());
        const Index bound = std::min<Index>(k - 1, alpha_max + 1);
        Index alpha = -1;
        for (Index c = 0; c < k; ++c) {
            const Index j = row[sz(c)];
            if (c > bound) {
                if (lo[sz(j)] > 0.5) return false;   // its one sits past the bound
                if (hi[sz(j)] > 0.5) {
                    write_domain_bound(lo, hi, j, BoundDir::Upper, 0.0,
                                       trail, {ReasonKind::Symmetry, -1, depth});
                    ++fixed;
                    if (changed) changed->push_back(j);
                }
                continue;
            }
            if (hi[sz(j)] > 0.5) alpha = c;
        }
        if (alpha < 0 && orb.partitioning) return false;   // no column left for a one
        alpha_max = std::max(alpha_max, alpha);
    }
    return true;
}

}  // namespace forge::search
