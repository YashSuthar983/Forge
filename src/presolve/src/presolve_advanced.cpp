#include "live_matrix.hpp"
#include "sor/model/exact.hpp"
#include <algorithm>
#include <cmath>
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
        // Exact sparse row echelon detects dependencies involving ANY number
        // of earlier equations, including non-dyadic elimination multipliers.
        // Only redundant equations are removed; no rounded row is installed.
        struct Equation { std::map<Index,Rational> entries; Rational rhs; };
        std::map<Index,Equation> echelon;
        std::uint64_t operations = 0;
        for (Index i = 0; i < matrix.m && operations < 200000; ++i) {
            if (!row_live[sz(i)] || matrix.row_lo[sz(i)] != matrix.row_hi[sz(i)] ||
                !std::isfinite(matrix.row_lo[sz(i)]) || matrix.rows[sz(i)].size() >
                    static_cast<std::size_t>(options.max_aggregation_row_nnz)) continue;
            Equation trial; trial.rhs = Rational(matrix.row_lo[sz(i)]);
            for (const auto& [j,a] : matrix.rows[sz(i)]) if (a != 0) trial.entries[j] = Rational(a);
            bool completed = true;
            while (!trial.entries.empty()) {
                const Index pivot = trial.entries.begin()->first;
                const auto source = echelon.find(pivot);
                if (source == echelon.end()) break;
                const Rational multiplier = trial.entries.begin()->second / source->second.entries.begin()->second;
                trial.rhs -= multiplier * source->second.rhs;
                for (const auto& [j,a] : source->second.entries) {
                    if (++operations > 200000) { completed = false; break; }
                    auto& value = trial.entries[j]; value -= multiplier*a;
                    if (value == 0) trial.entries.erase(j);
                    else {
                        const auto num = numerator(value);
                        if (boost::multiprecision::msb(denominator(value)) > 16384 ||
                            boost::multiprecision::msb(num < 0 ? -num : num) > 16384) { completed = false; break; }
                    }
                }
                if (!completed || trial.entries.size() > matrix.rows[sz(i)].size() +
                    static_cast<std::size_t>(std::max<Offset>(0,options.max_substitution_fill))) {
                    completed = false; break;
                }
            }
            if (!completed) continue;
            if (trial.entries.empty()) {
                if (trial.rhs != 0) continue; // A proof object is required before an infeasible report.
                for (const auto& [j,a] : matrix.rows[sz(i)]) { (void)a; matrix.col_rows[sz(j)].erase(i); }
                matrix.rows[sz(i)].clear(); row_live[sz(i)] = 0;
                ++map.stats.rows_removed; ++map.stats.linear_dependencies_removed;
            } else {
                const Index pivot = trial.entries.begin()->first;
                echelon.emplace(pivot, std::move(trial));
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
                    f64 multiplier;
                    const Rational ratio = Rational(it->second)/Rational(equation.at(pivot));
                    if (!representable(ratio, multiplier)) continue;
                    FlatMap rewritten = old;
                    bool safe = true;
                    for (const auto& [j, a] : equation) {
                        const auto old_entry = rewritten.find(j);
                        const Rational prior = old_entry == rewritten.end() ? Rational(0) : Rational(old_entry->second);
                        f64 value;
                        if (!representable(prior-ratio*Rational(a), value)) { safe = false; break; }
                        if (value == 0) rewritten.erase(j); else rewritten[j] = value;
                    }
                    if (!safe || rewritten.size() >= old.size()) continue;
                    f64 lower = matrix.row_lo[sz(target)], upper = matrix.row_hi[sz(target)];
                    const Rational shift = ratio*Rational(matrix.row_lo[sz(source)]);
                    if ((std::isfinite(lower) && !representable(Rational(lower)-shift, lower)) ||
                        (std::isfinite(upper) && !representable(Rational(upper)-shift, upper))) continue;
                    const auto removed = old.size()-rewritten.size();
                    for (const auto& [j, a] : old) {
                        (void)a;
                        if (rewritten.find(j) == rewritten.end()) matrix.col_rows[sz(j)].erase(target);
                    }
                    for (const auto& [j, a] : rewritten) {
                        (void)a;
                        if (old.find(j) == old.end()) matrix.col_rows[sz(j)].insert(target);
                    }
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
