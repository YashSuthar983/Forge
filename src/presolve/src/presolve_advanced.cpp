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
        struct ModRow {
            std::vector<std::pair<Index, std::uint64_t>> entries;   // ascending columns
            std::uint64_t inverse_lead = 0;                           // 1 / entries.front()
            std::vector<Index> provenance;                           // original rows, sorted
            bool certifiable = true;
        };
        std::vector<std::uint64_t> accumulator(sz(matrix.n), 0);
        std::vector<char> in_heap(sz(matrix.n), 0);
        std::vector<Index> touched, heap;
        std::size_t live_entries = 0;
        constexpr std::size_t kProvenanceCap = 256;
        std::map<Index, ModRow> echelon;   // pivot column -> row whose first entry it is
        // Linear budget in the equality block: a complete echelon of a
        // filled-in block is O(m^2 n) (fit1p: 120 ms) and found no
        // dependency on any Netlib model; presolve's share of a small solve
        // must stay proportional to its input.
        std::uint64_t equality_nnz = 0;
        for (Index i = 0; i < matrix.m; ++i)
            if (row_live[sz(i)] && matrix.row_lo[sz(i)] == matrix.row_hi[sz(i)])
                equality_nnz += matrix.rows[sz(i)].size();
        std::uint64_t operations = 0;
        const std::uint64_t operation_cap = 32 * equality_nnz + 100'000;
        // Provenance of the trial row: original rows its reduction used, as
        // an unsorted list deduplicated by stamp (sorted when it is read).
        std::vector<std::uint32_t> provenance_stamp(sz(matrix.m), 0);
        std::uint32_t stamp = 0;
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
        for (Index i = 0; i < matrix.m && operations < operation_cap; ++i) {
            if (!row_live[sz(i)] || matrix.row_lo[sz(i)] != matrix.row_hi[sz(i)] ||
                !std::isfinite(matrix.row_lo[sz(i)]) || matrix.rows[sz(i)].size() >
                    static_cast<std::size_t>(options.max_aggregation_row_nnz)) continue;
            // Dense accumulator + min-heap of touched columns: the leading
            // column is the heap top (stale entries skipped lazily), and an
            // update is two word operations instead of a tree node.
            const auto clear_trial = [&] {
                for (const Index j : touched) { accumulator[sz(j)] = 0; in_heap[sz(j)] = 0; }
                touched.clear();
                heap.clear();
                live_entries = 0;
            };
            const auto set_entry = [&](Index j, std::uint64_t v) {
                if (accumulator[sz(j)] == 0 && v != 0) ++live_entries;
                else if (accumulator[sz(j)] != 0 && v == 0) --live_entries;
                accumulator[sz(j)] = v;
                if (v != 0 && !in_heap[sz(j)]) {
                    in_heap[sz(j)] = 1;
                    touched.push_back(j);
                    heap.push_back(j);
                    std::push_heap(heap.begin(), heap.end(), std::greater<Index>());
                }
            };
            const auto leading = [&]() -> Index {
                while (!heap.empty() && accumulator[sz(heap.front())] == 0) {
                    std::pop_heap(heap.begin(), heap.end(), std::greater<Index>());
                    in_heap[sz(heap.back())] = 0;
                    heap.pop_back();
                }
                return heap.empty() ? Index{-1} : heap.front();
            };
            clear_trial();
            for (const auto& [j, a] : matrix.rows[sz(i)]) set_entry(j, to_mod(a));
            std::vector<Index> provenance;
            ++stamp;
            bool certifiable = true, filled = false;
            const std::size_t fill_cap = matrix.rows[sz(i)].size() +
                static_cast<std::size_t>(std::max<Offset>(0, options.max_substitution_fill));
            for (Index lead = leading(); lead >= 0; lead = leading()) {
                const auto source = echelon.find(lead);
                if (source == echelon.end()) break;
                const auto& row = source->second;
                const std::uint64_t multiplier = mod_mul(accumulator[sz(lead)], row.inverse_lead);
                if (operations + row.entries.size() > operation_cap) { filled = true; break; }
                for (const auto& [j, a] : row.entries) {
                    ++operations;
                    set_entry(j, mod_add(accumulator[sz(j)], P - mod_mul(multiplier, a)));
                }
                certifiable = certifiable && row.certifiable;
                if (certifiable) {
                    for (const Index r : row.provenance)
                        if (provenance_stamp[sz(r)] != stamp) {
                            provenance_stamp[sz(r)] = stamp;
                            provenance.push_back(r);
                        }
                    if (provenance.size() > kProvenanceCap) { certifiable = false; provenance.clear(); }
                }
                if (live_entries > fill_cap) { filled = true; break; }
            }
            if (filled) continue;
            std::sort(provenance.begin(), provenance.end());
            if (live_entries == 0) {
                // Candidate: verify exactly over the rows the reduction used.
                if (!certifiable || !exactly_dependent(i, provenance)) continue;
                for (const auto& [j,a] : matrix.rows[sz(i)]) { (void)a; matrix.col_rows[sz(j)].erase(i); }
                matrix.rows[sz(i)].clear(); row_live[sz(i)] = 0;
                ++map.stats.rows_removed; ++map.stats.linear_dependencies_removed;
            } else {
                ModRow row;
                for (const Index j : touched) if (accumulator[sz(j)] != 0) row.entries.emplace_back(j, accumulator[sz(j)]);
                std::sort(row.entries.begin(), row.entries.end());
                row.inverse_lead = mod_inv(row.entries.front().second);
                row.certifiable = certifiable;
                if (certifiable) {
                    provenance.insert(std::upper_bound(provenance.begin(), provenance.end(), i), i);
                    row.provenance = std::move(provenance);
                }
                echelon.emplace(row.entries.front().first, std::move(row));
            }
        }
    }
    if (options.equation_sparsification) {
        for (int pass = 0; pass < options.sparsification_passes; ++pass) {
            bool changed = false;
            for (Index source = 0; source < matrix.m; ++source) {
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
                            while (o != old.end() && o->first < j) ++o;
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
