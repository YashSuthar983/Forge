#include "live_matrix.hpp"
#include "sor/model/dyadic.hpp"
#include "sor/model/exact.hpp"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wpedantic"   // unsigned __int128 for GF(2^61-1)
#endif
#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <map>

namespace sor::presolve::detail {
namespace {
using model::Rational;
bool representable(const Rational& value, f64& out) {
    out = value.convert_to<f64>();
    return std::isfinite(out) && Rational(out) == value;
}
// A contradiction here is exact on the current binary64 matrix and box.
// Propagation endpoints round OUTWARD, so exhaustion never implies infeasibility.
bool propagate(const LiveMatrix& matrix, const std::vector<char>& active,
               std::vector<f64>& lo, std::vector<f64>& hi, int passes) {
    for (int pass = 0; pass < passes; ++pass) {
        bool changed = false;
        for (Index i = 0; i < matrix.m; ++i) {
            if (!active[sz(i)]) continue;
            const auto& row = matrix.rows[sz(i)];
            model::ExactIntervalSum activity;
            for (const auto& [j, a] : row) activity.add(a, lo[sz(j)], hi[sz(j)]);
            if ((std::isfinite(matrix.row_hi[sz(i)]) && activity.finite_minimum() &&
                 activity.exact_minimum() > Rational(matrix.row_hi[sz(i)])) ||
                (std::isfinite(matrix.row_lo[sz(i)]) && activity.finite_maximum() &&
                 activity.exact_maximum() < Rational(matrix.row_lo[sz(i)]))) return false;
            for (const auto& [j, a] : row) {
                if (a == 0) continue;
                model::ExactIntervalSum rest;
                for (const auto& [k, b] : row) if (k != j) rest.add(b, lo[sz(k)], hi[sz(k)]);
                f64 lower = lo[sz(j)], upper = hi[sz(j)];
                if (std::isfinite(matrix.row_lo[sz(i)]) && rest.finite_maximum()) {
                    const Rational bound = (Rational(matrix.row_lo[sz(i)])-rest.exact_maximum())/Rational(a);
                    if (a > 0) lower = std::max(lower, model::rounded_down(bound));
                    else upper = std::min(upper, model::rounded_up(bound));
                }
                if (std::isfinite(matrix.row_hi[sz(i)]) && rest.finite_minimum()) {
                    const Rational bound = (Rational(matrix.row_hi[sz(i)])-rest.exact_minimum())/Rational(a);
                    if (a > 0) upper = std::min(upper, model::rounded_up(bound));
                    else lower = std::max(lower, model::rounded_down(bound));
                }
                if (lower > upper) return false;
                changed |= lower != lo[sz(j)] || upper != hi[sz(j)];
                lo[sz(j)] = lower; hi[sz(j)] = upper;
            }
        }
        if (!changed) break;
    }
    return true;
}
}

void advanced_reductions(LiveMatrix& matrix, std::vector<char>& row_live,
    const std::vector<char>& col_live, std::vector<f64>& lo, std::vector<f64>& hi,
    PresolveMap& map, const PresolveOptions& options) {
    if (options.equation_sparsification) {
        // Dependent equalities. A sparse row echelon over GF(p), p = 2^61-1,
        // nominates candidates in word arithmetic; each nominated row is then
        // verified EXACTLY, with rational elimination over only the earlier
        // rows its modular reduction used (and rhs consistency), before it is
        // removed. The previous exact echelon over every equality spent its
        // 200k-update cap before reaching any dependency: 350 ms and zero rows
        // removed on brandy, 25fv47, fit1p (most of their presolve time).
        // A modular false negative only keeps a redundant row (safe); a false
        // positive is rejected by the exact check.
        constexpr std::uint64_t P = (std::uint64_t{1} << 61) - 1;
        const auto mod_add = [](std::uint64_t x, std::uint64_t y) {
            const std::uint64_t t = x + y; return t >= P ? t - P : t;
        };
        const auto mod_mul = [](std::uint64_t x, std::uint64_t y) {
            const unsigned __int128 t = static_cast<unsigned __int128>(x) * y;
            std::uint64_t r = static_cast<std::uint64_t>(t & P) + static_cast<std::uint64_t>(t >> 61);
            return r >= P ? r - P : r;
        };
        const auto mod_pow2 = [&](int e) {   // 2^61 = 1 (mod P)
            int k = e % 61; if (k < 0) k += 61;
            return std::uint64_t{1} << k;
        };
        const auto mod_inv = [&](std::uint64_t x) {
            std::uint64_t r = 1, base = x, e = P - 2;
            while (e) { if (e & 1) r = mod_mul(r, base); base = mod_mul(base, base); e >>= 1; }
            return r;
        };
        // A binary64 value is M 2^e exactly with |M| < 2^53.
        const auto to_mod = [&](f64 v) -> std::uint64_t {
            if (v == 0) return 0;
            int e = 0;
            const f64 f = std::frexp(std::fabs(v), &e);
            const auto mantissa = static_cast<std::uint64_t>(std::ldexp(f, 53));
            const std::uint64_t r = mod_mul(mantissa % P, mod_pow2(e - 53));
            return v < 0 ? (r == 0 ? 0 : P - r) : r;
        };
        // Elimination order. The active equations are eliminated right-looking
        // with Markowitz pivots: the next pivot (row, column) minimizes
        // (row length - 1) * (column count - 1) over the active submatrix.
        // Over GF(p) every nonzero is an acceptable pivot, so sparsity is the
        // only criterion and no stability threshold is needed. The previous
        // left-looking echelon took the rows in index order and pivoted on
        // each row's smallest column; on qap10 that filled the assignment
        // rows until the operation budget ran out after 28 of its 272
        // dependent equations.
        //
        // Each row records its eliminations (pivot row, multiplier): one
        // entry per update. The row as a combination of original rows is
        // expanded only when the row becomes empty, through the pivot rows'
        // own records, each pivot row expanded at most once. Tracking every
        // row's combination eagerly made each update cost the length of the
        // target's combination as well; on highschool1-aigio that was 2.9 s
        // for 44k updates, almost all of it for rows that were never empty.
        //
        // Pivot rows are never removed, and a row that becomes empty is a
        // combination of itself and pivot rows only, so the rows removed as
        // dependent are combinations of rows that stay.
        struct ActiveRow {
            std::vector<std::pair<Index, std::uint64_t>> entries;        // ascending columns
            std::vector<std::pair<Index, std::uint64_t>> eliminations;   // (pivot slot, factor)
            // A row that fills past its original length plus the substitution
            // fill allowance leaves the active submatrix and stays in the
            // model (a missed dependency at worst). Without it a linking row
            // of highschool1-aigio filled to 55,570 entries and every later
            // pivot merged into it: 976M entry visits, 2.4 s.
            std::size_t fill_limit = 0;
            bool active = true;
        };
        constexpr std::size_t kProvenanceCap = 256;   // support of the rational fallback
        std::vector<Index> origin;   // slot -> original row
        std::vector<ActiveRow> active_rows;
        std::vector<std::vector<Index>> column_slots(sz(matrix.n));   // may hold stale slots
        std::vector<Index> column_count(sz(matrix.n), 0);              // exact
        std::uint64_t equality_nnz = 0;
        for (Index i = 0; i < matrix.m; ++i) {
            if (!row_live[sz(i)] || matrix.row_lo[sz(i)] != matrix.row_hi[sz(i)] ||
                !std::isfinite(matrix.row_lo[sz(i)]) || matrix.rows[sz(i)].size() >
                    static_cast<std::size_t>(options.max_aggregation_row_nnz)) continue;
            ActiveRow row;
            for (const auto& [j, a] : matrix.rows[sz(i)])
                if (a != 0) row.entries.emplace_back(j, to_mod(a));
            if (row.entries.empty()) continue;
            std::sort(row.entries.begin(), row.entries.end());
            row.fill_limit = row.entries.size() +
                static_cast<std::size_t>(std::max<Offset>(0, options.max_substitution_fill));
            const Index slot = static_cast<Index>(active_rows.size());
            for (const auto& [j, v] : row.entries) {
                column_slots[sz(j)].push_back(slot);
                ++column_count[sz(j)];
            }
            equality_nnz += row.entries.size();
            origin.push_back(i);
            active_rows.push_back(std::move(row));
        }
        // Linear budget in the equality block: presolve's share of a small
        // solve must stay proportional to its input. It counts every entry a
        // merge visits (both rows) and every term of an expansion. Measured
        // need per equality nonzero: qap10 55, satellites2-40 16, dfl001 and
        // huahum 6, highschool1-aigio 0.5.
        std::uint64_t operations = 0;
        const std::uint64_t operation_cap = 64 * equality_nnz + 100'000;
        const auto value_at = [&](const ActiveRow& row, Index j) -> std::uint64_t {
            const auto it = std::lower_bound(row.entries.begin(), row.entries.end(),
                std::make_pair(j, std::uint64_t{0}),
                [](const auto& x, const auto& y) { return x.first < y.first; });
            return it != row.entries.end() && it->first == j ? it->second : 0;
        };
        // Buckets by current row length and column count. Entries go stale
        // when a count changes (the new count gets a fresh entry) and are
        // validated when read.
        std::vector<std::vector<Index>> row_bucket(1), column_bucket(1);
        const auto bucket_push = [](std::vector<std::vector<Index>>& b, std::size_t k, Index x) {
            if (b.size() <= k) b.resize(k + 1);
            b[k].push_back(x);
        };
        for (Index s = 0; s < static_cast<Index>(active_rows.size()); ++s)
            bucket_push(row_bucket, active_rows[sz(s)].entries.size(), s);
        for (Index j = 0; j < matrix.n; ++j)
            if (column_count[sz(j)] > 0) bucket_push(column_bucket, sz(column_count[sz(j)]), j);
        std::size_t remaining = active_rows.size();

        // Exact and cheap: the modular combination sum_r c_r row_r = 0 names
        // rational weights c_r = n_r/d_r (rational reconstruction, |n|,|d| <
        // 2^30). Scaled to integers below 2^53 they are binary64 numbers, so
        // sum_r w_r a_rj = 0 for every column and the rhs is checked exactly
        // with DyadicSum, in time linear in the rows. Such a sum is a proof of
        // dependence; when reconstruction or the check fails, the rational
        // elimination below decides (huahum: 24700 dependent rows, 6 s of
        // rational elimination before).
        const auto reconstruct = [&](std::uint64_t c, std::int64_t& num, std::int64_t& den) {
            constexpr std::int64_t bound = std::int64_t{1} << 30;
            std::int64_t r0 = static_cast<std::int64_t>(P), r1 = static_cast<std::int64_t>(c);
            std::int64_t t0 = 0, t1 = 1;
            while (r1 >= bound) {
                const std::int64_t q = r0 / r1;
                std::int64_t r2 = r0 - q * r1; r0 = r1; r1 = r2;
                const __int128 t2 = static_cast<__int128>(t0) - static_cast<__int128>(q) * t1;
                if (t2 > (static_cast<__int128>(1) << 62) || t2 < -(static_cast<__int128>(1) << 62)) return false;
                t0 = t1; t1 = static_cast<std::int64_t>(t2);
            }
            if (t1 == 0 || t1 >= bound || t1 <= -bound) return false;
            num = t1 < 0 ? -r1 : r1;
            den = t1 < 0 ? -t1 : t1;
            // num == c * den (mod P)
            const std::uint64_t lhs = num < 0 ? P - static_cast<std::uint64_t>(-num) % P
                                              : static_cast<std::uint64_t>(num) % P;
            return lhs % P == mod_mul(c, static_cast<std::uint64_t>(den));
        };
        const auto combination_proves_dependent =
            [&](const std::vector<std::pair<Index, std::uint64_t>>& combination) {
            std::vector<std::pair<Index, std::int64_t>> weights;   // row, n_r
            std::vector<std::int64_t> dens;
            std::int64_t common = 1;
            for (const auto& [r, c] : combination) {
                if (c == 0) continue;
                std::int64_t num = 0, den = 1;
                if (!reconstruct(c, num, den)) return false;
                const std::int64_t g = std::gcd(common, den);
                const __int128 l = static_cast<__int128>(common / g) * den;
                if (l >= (static_cast<__int128>(1) << 52)) return false;
                common = static_cast<std::int64_t>(l);
                weights.emplace_back(r, num);
                dens.push_back(den);
            }
            model::DyadicSum rhs;
            std::vector<std::pair<Index, std::pair<f64, f64>>> products;
            for (std::size_t q = 0; q < weights.size(); ++q) {
                const __int128 w = static_cast<__int128>(weights[q].second) * (common / dens[q]);
                if (w >= (static_cast<__int128>(1) << 53) || w <= -(static_cast<__int128>(1) << 53)) return false;
                const f64 wd = static_cast<f64>(static_cast<std::int64_t>(w));
                const Index r = weights[q].first;
                if (!row_live[sz(r)] || matrix.row_lo[sz(r)] != matrix.row_hi[sz(r)] ||
                    !std::isfinite(matrix.row_lo[sz(r)])) return false;
                rhs.add_product(wd, matrix.row_lo[sz(r)]);
                for (const auto& [j, a] : matrix.rows[sz(r)])
                    if (a != 0) products.push_back({j, {wd, a}});
            }
            if (rhs.sign() != 0) return false;
            std::sort(products.begin(), products.end(),
                      [](const auto& x, const auto& y) { return x.first < y.first; });
            for (std::size_t q = 0; q < products.size();) {
                model::DyadicSum column;
                std::size_t e = q;
                for (; e < products.size() && products[e].first == products[q].first; ++e)
                    column.add_product(products[e].second.first, products[e].second.second);
                if (column.sign() != 0) return false;
                q = e;
            }
            return true;
        };
        // Exact: is row i a rational combination of rows `support`, rhs included?
        const auto exactly_dependent = [&](Index i, const std::vector<Index>& support) {
            struct Equation { std::map<Index, Rational> entries; Rational rhs; };
            std::map<Index, Equation> basis;
            const auto reduce = [&](Equation& e) {
                while (!e.entries.empty()) {
                    const auto source = basis.find(e.entries.begin()->first);
                    if (source == basis.end()) return;
                    const Rational multiplier = e.entries.begin()->second / source->second.entries.begin()->second;
                    e.rhs -= multiplier * source->second.rhs;
                    for (const auto& [j, a] : source->second.entries) {
                        auto& value = e.entries[j];
                        value -= multiplier * a;
                        if (value == 0) e.entries.erase(j);
                    }
                }
            };
            const auto load = [&](Index r) {
                Equation e;
                e.rhs = Rational(matrix.row_lo[sz(r)]);
                for (const auto& [j, a] : matrix.rows[sz(r)]) if (a != 0) e.entries[j] = Rational(a);
                return e;
            };
            for (const Index r : support) {
                if (!row_live[sz(r)]) continue;
                auto e = load(r);
                reduce(e);
                if (!e.entries.empty()) {
                    const Index pivot = e.entries.begin()->first;
                    basis.emplace(pivot, std::move(e));
                }
            }
            auto target = load(i);
            reduce(target);
            return target.entries.empty() && target.rhs == 0;
        };
        // Pivot search over columns of count k and rows of length k for
        // k = 1, 2, ..., a few candidates each (Markowitz with a search limit).
        constexpr int kSearch = 4;
        std::vector<std::uint32_t> slot_stamp(active_rows.size(), 0);
        std::uint32_t stamp = 0;
        std::vector<std::pair<Index, std::uint64_t>> merged;
        // Expansion of rows into original-row combinations (see ActiveRow).
        std::vector<std::vector<std::pair<Index, std::uint64_t>>> expanded(active_rows.size());
        std::vector<char> is_expanded(active_rows.size(), 0);
        std::vector<Index> pivot_sequence(active_rows.size(), 0);
        Index pivots_made = 0;
        std::vector<std::uint64_t> weight(sz(matrix.m), 0);
        std::vector<Index> weight_rows;
        std::vector<char> queued(active_rows.size(), 0);
        // Returns false when the operation budget runs out first.
        const auto expand = [&](Index slot, std::vector<std::pair<Index, std::uint64_t>>& out) {
            // Pivot rows this one depends on, expanded oldest first: a pivot
            // row's eliminations name only earlier pivots.
            std::vector<Index> needed, stack{slot};
            while (!stack.empty()) {
                const Index s = stack.back();
                stack.pop_back();
                for (const auto& [p, f] : active_rows[sz(s)].eliminations) {
                    (void)f;
                    if (is_expanded[sz(p)] || queued[sz(p)]) continue;
                    queued[sz(p)] = 1;
                    needed.push_back(p);
                    stack.push_back(p);
                }
            }
            std::sort(needed.begin(), needed.end(), [&](Index x, Index y) {
                return pivot_sequence[sz(x)] < pivot_sequence[sz(y)];
            });
            const auto combine = [&](Index s, std::vector<std::pair<Index, std::uint64_t>>& result) {
                weight_rows.assign(1, origin[sz(s)]);
                weight[sz(origin[sz(s)])] = 1;
                for (const auto& [p, f] : active_rows[sz(s)].eliminations) {
                    operations += expanded[sz(p)].size();
                    for (const auto& [r, c] : expanded[sz(p)]) {
                        if (weight[sz(r)] == 0) weight_rows.push_back(r);
                        weight[sz(r)] = mod_add(weight[sz(r)], mod_mul(f, c));
                        if (weight[sz(r)] == 0) weight[sz(r)] = P;   // cancelled, still listed
                    }
                }
                std::sort(weight_rows.begin(), weight_rows.end());
                weight_rows.erase(std::unique(weight_rows.begin(), weight_rows.end()), weight_rows.end());
                result.clear();
                for (const Index r : weight_rows) {
                    const std::uint64_t c = weight[sz(r)] == P ? 0 : weight[sz(r)];
                    if (c != 0) result.emplace_back(r, c);
                    weight[sz(r)] = 0;
                }
            };
            bool within_budget = true;
            for (const Index p : needed) {
                queued[sz(p)] = 0;
                if (!within_budget) continue;
                combine(p, expanded[sz(p)]);
                is_expanded[sz(p)] = 1;
                within_budget = operations < operation_cap;
            }
            if (!within_budget) return false;
            combine(slot, out);
            return operations < operation_cap;
        };
        std::vector<std::pair<Index, std::uint64_t>> combination;
        while (remaining > 0 && operations < operation_cap) {
            check_deadline(options);
            Index best_slot = -1, best_column = -1;
            std::uint64_t best_cost = std::numeric_limits<std::uint64_t>::max();
            int examined = 0;
            const std::size_t k_end = std::max(row_bucket.size(), column_bucket.size());
            for (std::size_t k = 1; k < k_end; ++k) {
                if (k < column_bucket.size()) {
                    auto& bucket = column_bucket[k];
                    for (std::size_t t = 0; t < bucket.size() && examined < kSearch;) {
                        const Index j = bucket[t];
                        if (sz(column_count[sz(j)]) != k) {
                            bucket[t] = bucket.back();
                            bucket.pop_back();
                            continue;
                        }
                        ++t;
                        ++examined;
                        for (const Index s : column_slots[sz(j)]) {
                            const auto& row = active_rows[sz(s)];
                            if (!row.active || value_at(row, j) == 0) continue;
                            const std::uint64_t cost = (row.entries.size() - 1) * (k - 1);
                            if (cost < best_cost) { best_cost = cost; best_slot = s; best_column = j; }
                        }
                    }
                }
                if (best_slot >= 0 && best_cost <= (k - 1) * (k - 1)) break;
                if (k < row_bucket.size()) {
                    auto& bucket = row_bucket[k];
                    for (std::size_t t = 0; t < bucket.size() && examined < 2 * kSearch;) {
                        const Index s = bucket[t];
                        const auto& row = active_rows[sz(s)];
                        if (!row.active || row.entries.size() != k) {
                            bucket[t] = bucket.back();
                            bucket.pop_back();
                            continue;
                        }
                        ++t;
                        ++examined;
                        for (const auto& [j, v] : row.entries) {
                            const std::uint64_t cost = (k - 1) * (sz(column_count[sz(j)]) - 1);
                            if (cost < best_cost) { best_cost = cost; best_slot = s; best_column = j; }
                        }
                    }
                }
                if (best_slot >= 0 && (best_cost <= (k - 1) * k || examined >= 2 * kSearch)) break;
            }
            if (best_slot < 0) break;

            ActiveRow& pivot = active_rows[sz(best_slot)];
            const std::uint64_t inverse = mod_inv(value_at(pivot, best_column));
            ++stamp;
            slot_stamp[sz(best_slot)] = stamp;
            std::vector<Index> targets;
            for (const Index s : column_slots[sz(best_column)]) {
                if (slot_stamp[sz(s)] == stamp) continue;
                slot_stamp[sz(s)] = stamp;
                if (active_rows[sz(s)].active && value_at(active_rows[sz(s)], best_column) != 0)
                    targets.push_back(s);
            }
            // The pivot row leaves the active submatrix.
            pivot.active = false;
            pivot_sequence[sz(best_slot)] = pivots_made++;
            --remaining;
            for (const auto& [j, v] : pivot.entries) {
                if (--column_count[sz(j)] > 0 && j != best_column)
                    bucket_push(column_bucket, sz(column_count[sz(j)]), j);
            }
            for (const Index t : targets) {
                ActiveRow& row = active_rows[sz(t)];
                const std::uint64_t negated =
                    P - mod_mul(value_at(row, best_column), inverse);   // nonzero
                merged.clear();
                std::size_t a = 0, b = 0;
                while (a < row.entries.size() || b < pivot.entries.size()) {
                    if (b == pivot.entries.size() ||
                        (a < row.entries.size() && row.entries[a].first < pivot.entries[b].first)) {
                        merged.push_back(row.entries[a++]);
                    } else if (a == row.entries.size() ||
                               pivot.entries[b].first < row.entries[a].first) {
                        // Fill: a product of nonzeros is nonzero in a field.
                        const Index j = pivot.entries[b].first;
                        merged.emplace_back(j, mod_mul(negated, pivot.entries[b++].second));
                        column_slots[sz(j)].push_back(t);
                        bucket_push(column_bucket, sz(++column_count[sz(j)]), j);
                    } else {
                        const Index j = row.entries[a].first;
                        const std::uint64_t v = mod_add(row.entries[a++].second,
                                                        mod_mul(negated, pivot.entries[b++].second));
                        if (v != 0) {
                            merged.emplace_back(j, v);
                        } else if (--column_count[sz(j)] > 0 && j != best_column) {
                            bucket_push(column_bucket, sz(column_count[sz(j)]), j);
                        }
                    }
                }
                operations += row.entries.size() + pivot.entries.size();
                row.entries.swap(merged);
                row.eliminations.emplace_back(best_slot, negated);
                if (row.entries.size() > row.fill_limit) {
                    row.active = false;
                    --remaining;
                    for (const auto& [j, v] : row.entries)
                        if (--column_count[sz(j)] > 0 && j != best_column)
                            bucket_push(column_bucket, sz(column_count[sz(j)]), j);
                    continue;
                }
                if (!row.entries.empty()) {
                    bucket_push(row_bucket, row.entries.size(), t);
                    continue;
                }
                // Empty modulo P: a candidate, verified exactly before removal.
                row.active = false;
                --remaining;
                const Index i = origin[sz(t)];
                if (!expand(t, combination)) continue;
                if (!combination_proves_dependent(combination)) {
                    // The rational fallback grows superlinearly in its support;
                    // the combination check above is linear and has no cap.
                    if (combination.size() > kProvenanceCap) continue;
                    std::vector<Index> support;
                    for (const auto& [r, c] : combination)
                        if (r != i && c != 0) support.push_back(r);
                    if (!exactly_dependent(i, support)) continue;
                }
                for (const auto& entry : matrix.rows[sz(i)]) matrix.col_rows[sz(entry.first)].erase(i);
                matrix.rows[sz(i)].clear(); row_live[sz(i)] = 0;
                ++map.stats.rows_removed; ++map.stats.linear_dependencies_removed;
            }
            column_slots[sz(best_column)].clear();
        }
    }
    if (options.equation_sparsification) {
        for (int pass = 0; pass < options.sparsification_passes; ++pass) {
            bool changed = false;
            for (Index source = 0; source < matrix.m; ++source) {
                check_deadline(options);
                if (!row_live[sz(source)] || matrix.row_lo[sz(source)] != matrix.row_hi[sz(source)] ||
                    !std::isfinite(matrix.row_lo[sz(source)])) continue;
                const auto equation = matrix.rows[sz(source)];
                if (equation.size() < 2 || equation.size() > static_cast<std::size_t>(options.max_aggregation_row_nnz)) continue;
                // Use the sparsest incidence as a bounded candidate generator.
                Index pivot = -1; std::size_t degree = 257;
                for (const auto& [j, a] : equation)
                    if (a != 0 && matrix.col_rows[sz(j)].size() < degree) {
                        pivot = j; degree = matrix.col_rows[sz(j)].size();
                    }
                if (pivot < 0) continue;
                const auto targets = matrix.col_rows[sz(pivot)].v;
                for (Index target : targets) {
                    if (target == source || !row_live[sz(target)]) continue;
                    const auto& old = matrix.rows[sz(target)];
                    const auto it = old.find(pivot);
                    if (it == old.end()) continue;
                    // Structural screen: every source entry absent from the
                    // target becomes a new nonzero (multiplier and coefficient
                    // are nonzero), and only shared entries can cancel, so a
                    // sparser row needs more shared entries than new ones.
                    {
                        std::size_t shared = 0;
                        auto o = old.begin();
                        for (const auto& [j, a] : equation) {
                            (void)a;
                            // Search a much longer target, walk a comparable one.
                            o = old.size() > 8 * equation.size()
                                ? std::lower_bound(o, old.end(), j,
                                      [](const auto& entry, Index k) { return entry.first < k; })
                                : std::find_if(o, old.end(),
                                      [j](const auto& entry) { return entry.first >= j; });
                            if (o != old.end() && o->first == j) ++shared;
                        }
                        if (shared <= equation.size() - shared) continue;
                    }
                    // The multiplier must be a binary64 number exactly. When
                    // the ratio is representable, IEEE division returns it;
                    // the exact product check confirms it.
                    const f64 pivot_coeff = equation.at(pivot);
                    const f64 multiplier = it->second / pivot_coeff;
                    if (!std::isfinite(multiplier)) continue;
                    {
                        model::DyadicSum check;
                        check.add_product(multiplier, pivot_coeff);
                        check.add(-it->second);
                        if (check.sign() != 0) continue;
                    }
                    // target - multiplier * source, entry by entry (both rows
                    // sorted by column). A first pass decides, without
                    // building anything, whether every new coefficient is a
                    // binary64 number and the row gets sparser; most targets
                    // fail and then cost no copy.
                    const auto combined = [&](f64 prior, f64 a, f64& value) {
                        model::DyadicSum t;
                        t.add(prior);
                        t.add_product(-multiplier, a);
                        return t.representable(value);
                    };
                    bool safe = true;
                    std::ptrdiff_t net = 0;   // entries added minus entries cancelled
                    {
                        auto o = old.begin();
                        for (const auto& [j, a] : equation) {
                            while (o != old.end() && o->first < j) ++o;
                            const bool present = o != old.end() && o->first == j;
                            f64 value;
                            if (!combined(present ? o->second : 0.0, a, value)) { safe = false; break; }
                            if (present && value == 0) --net;
                            else if (!present && value != 0) ++net;
                        }
                    }
                    if (!safe || net >= 0) continue;
                    f64 lower = matrix.row_lo[sz(target)], upper = matrix.row_hi[sz(target)];
                    const f64 source_side = matrix.row_lo[sz(source)];
                    const auto shifted = [&](f64& side) {
                        if (!std::isfinite(side)) return true;
                        model::DyadicSum t;
                        t.add(side);
                        t.add_product(-multiplier, source_side);
                        return t.representable(side);
                    };
                    if (!shifted(lower) || !shifted(upper)) continue;
                    // Only the source's columns can change: record the ones
                    // that cancel or appear, so the column incidence update
                    // does not rescan a long target row.
                    FlatMap rewritten;
                    rewritten.v.reserve(old.size());
                    std::vector<Index> cancelled, appeared;
                    {
                        auto o = old.begin();
                        for (const auto& [j, a] : equation) {
                            for (; o != old.end() && o->first < j; ++o) rewritten.v.push_back(*o);
                            const bool present = o != old.end() && o->first == j;
                            f64 value = 0;
                            combined(present ? o->second : 0.0, a, value);
                            if (value != 0) rewritten.v.emplace_back(j, value);
                            if (present && value == 0) cancelled.push_back(j);
                            if (!present && value != 0) appeared.push_back(j);
                            if (present) ++o;
                        }
                        for (; o != old.end(); ++o) rewritten.v.push_back(*o);
                    }
                    const auto removed = old.size()-rewritten.size();
                    for (const Index j : cancelled) matrix.col_rows[sz(j)].erase(target);
                    for (const Index j : appeared) matrix.col_rows[sz(j)].insert(target);
                    matrix.rows[sz(target)] = std::move(rewritten);
                    matrix.row_lo[sz(target)] = lower; matrix.row_hi[sz(target)] = upper;
                    DualRecoveryStep journal;
                    journal.kind = DualRecoveryKind::EquationSparsification;
                    journal.row = source; journal.other_rows = {target};
                    journal.other_row_coefficients = {multiplier};
                    map.recovery_steps.push_back(std::move(journal));
                    ++map.stats.equation_sparsifications;
                    map.stats.sparsification_nnz_removed += static_cast<Offset>(removed);
                    changed = true;
                    if (matrix.rows[sz(target)].empty() && lower <= 0 && upper >= 0) {
                        row_live[sz(target)] = 0; ++map.stats.rows_removed;
                    }
                }
            }
            if (!changed) break;
        }
    }
    if (options.coefficient_strengthening) {
        // For continuous LPs arbitrary integer coefficient strengthening is
        // invalid. Exact power-of-two row normalization preserves the polytope.
        for (Index i = 0; i < matrix.m; ++i) {
            check_deadline(options);
            if (!row_live[sz(i)] || matrix.rows[sz(i)].empty()) continue;
            f64 maximum = 0;
            for (const auto& [j, a] : matrix.rows[sz(i)]) {
                (void)j; maximum = std::max(maximum, std::fabs(a));
            }
            int exponent = 0; std::frexp(maximum, &exponent);
            if (exponent >= -8 && exponent <= 8) continue;
            const f64 scale = std::ldexp(1.0, std::clamp(-exponent, -512, 512));
            FlatMap row = matrix.rows[sz(i)]; bool safe = true;
            for (auto& [j, a] : row) {
                (void)j; f64 value;
                if (!representable(Rational(a)*Rational(scale), value)) { safe = false; break; }
                a = value;
            }
            f64 lower = matrix.row_lo[sz(i)], upper = matrix.row_hi[sz(i)];
            if (!safe || (std::isfinite(lower) && !representable(Rational(lower)*Rational(scale), lower)) ||
                (std::isfinite(upper) && !representable(Rational(upper)*Rational(scale), upper))) continue;
            matrix.rows[sz(i)] = std::move(row); matrix.row_lo[sz(i)] = lower; matrix.row_hi[sz(i)] = upper;
            DualRecoveryStep journal; journal.kind = DualRecoveryKind::RowScaling;
            journal.row = i; journal.coeff = scale;
            map.recovery_steps.push_back(std::move(journal)); ++map.stats.coefficient_strengthenings;
        }
    }
    if (options.domain_probing) {
        Index probes = 0;
        for (Index j = 0; j < matrix.n && probes < options.max_domain_probes; ++j) {
            check_deadline(options);
            if (!col_live[sz(j)] || !std::isfinite(lo[sz(j)]) || !std::isfinite(hi[sz(j)]) || lo[sz(j)] >= hi[sz(j)]) continue;
            const f64 midpoint = std::midpoint(lo[sz(j)], hi[sz(j)]);
            if (midpoint == lo[sz(j)] || midpoint == hi[sz(j)]) continue;
            const f64 old_lo = lo[sz(j)], old_hi = hi[sz(j)];
            for (int side = 0; side < 2; ++side) {
                auto trial_lo = lo, trial_hi = hi;
                if (side == 0) trial_hi[sz(j)] = midpoint; else trial_lo[sz(j)] = midpoint;
                ++map.stats.domain_probes;
                if (!propagate(matrix, row_live, trial_lo, trial_hi, 3)) {
                    if (side == 0) lo[sz(j)] = midpoint; else hi[sz(j)] = midpoint;
                }
            }
            ++probes;
            if (lo[sz(j)] != old_lo || hi[sz(j)] != old_hi) {
                map.bound_changes.push_back({j, -1, 0, old_lo, old_hi, lo[sz(j)], hi[sz(j)]});
                DualRecoveryStep journal;
                journal.kind = DualRecoveryKind::BoundTightening; journal.col = j;
                journal.old_lo = old_lo; journal.old_hi = old_hi;
                journal.new_lo = lo[sz(j)]; journal.new_hi = hi[sz(j)];
                map.recovery_steps.push_back(std::move(journal));
                ++map.stats.bounds_tightened;
            }
        }
    }
}
} // namespace sor::presolve::detail
