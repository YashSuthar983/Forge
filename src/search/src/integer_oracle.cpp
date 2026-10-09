#include "sor/search/integer_oracle.hpp"
#include "sor/model/dyadic.hpp"
#include "sor/sparse/csc.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <set>

namespace sor::search {
namespace {
using Big = boost::multiprecision::cpp_int;
using core::Index;
using core::f64;
std::size_t sz(Index j) { return static_cast<std::size_t>(j); }
Big rounded_integer(const model::DyadicSum& sum, bool upward) {
    const auto value = sum.value();
    const Big num = numerator(value), den = denominator(value);
    Big integer = num / den;
    if (num % den != 0) {
        if (upward && num > 0) ++integer;
        if (!upward && num < 0) --integer;
    }
    return integer;
}
}
struct IntegerDomainOracle::Impl {
    model::LpProblem model;
    sparse::CscMatrix columns;
    Status status = Status::Ineligible;
    Result result;
    struct Path { std::shared_ptr<const Path> parent; Index column; f64 value; };
    struct Candidate { model::DyadicSum cost; std::shared_ptr<const Path> path; };
    using States = std::map<std::vector<Big>, Candidate>;
    States current, next;
    States::const_iterator cursor;
    std::vector<f64> lo, hi;
    std::vector<Big> remaining_min, remaining_max;
    std::vector<std::optional<Big>> row_lo, row_hi;
    std::vector<Index> last, live, after, old_slot;
    struct Transition { Index row, source; Big coefficient; bool retained; };
    std::vector<Transition> transitions;
    Index column = 0;
    f64 value = 0;
    bool prepared = false;

    bool possible(Index row, const Big& activity) const {
        return (!row_lo[sz(row)] || activity + remaining_max[sz(row)] >= *row_lo[sz(row)]) &&
               (!row_hi[sz(row)] || activity + remaining_min[sz(row)] <= *row_hi[sz(row)]);
    }
    void prepare_column() {
        const auto& cp = columns.pattern.col_ptr();
        const auto& ri = columns.pattern.row_idx();
        std::set<Index> affected(live.begin(), live.end());
        for (auto k = cp[sz(column)]; k < cp[sz(column) + 1]; ++k) {
            // Stored structural zeros cannot reopen a retired row. Its
            // activity has already been checked and removed from state keys.
            if (columns.vals[static_cast<std::size_t>(k)] == 0) continue;
            const Index row = ri[static_cast<std::size_t>(k)];
            const Big a(columns.vals[static_cast<std::size_t>(k)]);
            affected.insert(row);
            remaining_min[sz(row)] -= a * Big(a > 0 ? lo[sz(column)] : hi[sz(column)]);
            remaining_max[sz(row)] -= a * Big(a > 0 ? hi[sz(column)] : lo[sz(column)]);
        }
        for (std::size_t q = 0; q < live.size(); ++q) old_slot[sz(live[q])] = static_cast<Index>(q);
        transitions.clear(); after.clear();
        auto k = cp[sz(column)];
        for (Index row : affected) {
            while (k < cp[sz(column) + 1] && ri[static_cast<std::size_t>(k)] < row) ++k;
            Big a = k < cp[sz(column) + 1] && ri[static_cast<std::size_t>(k)] == row
                ? Big(columns.vals[static_cast<std::size_t>(k)]) : Big(0);
            const bool retained = last[sz(row)] > column;
            transitions.push_back({row, old_slot[sz(row)], std::move(a), retained});
            if (retained) after.push_back(row);
        }
        cursor = current.cbegin(); value = lo[sz(column)]; prepared = true;
    }
    void finish() {
        if (current.empty()) { status = Status::Infeasible; result.status = status; return; }
        // After all columns, no row remains live; exact state merging leaves
        // the globally cheapest feasible path, including the objective offset.
        const auto& best = current.begin()->second;
        result.point.resize(sz(model.n_cols()));
        for (auto path = best.path; path; path = path->parent) result.point[sz(path->column)] = path->value;
        result.lower_bound_min = best.cost.down();
        status = result.status = Status::Optimal;
    }
};
IntegerDomainOracle::IntegerDomainOracle(const model::LpProblem& lp, f64 tolerance)
    : impl_(std::make_unique<Impl>()) {
    auto& s = *impl_;
    if (!std::isfinite(tolerance) || tolerance < 0 || lp.is_integer.size() != sz(lp.n_cols())) return;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (!lp.is_integer[sz(j)] || !std::isfinite(lp.col_lo[sz(j)]) || !std::isfinite(lp.col_hi[sz(j)]) ||
            std::fabs(lp.col_lo[sz(j)]) >= 0x1p52 || std::fabs(lp.col_hi[sz(j)]) >= 0x1p52) return;
    }
    for (f64 a : lp.A.vals) if (!std::isfinite(a) || a != std::trunc(a)) return;
    lp.validate(true);
    s.model = lp; s.columns = sparse::to_csc(lp.A);
    const Index n = lp.n_cols(), m = lp.n_rows();
    s.remaining_min.resize(sz(m)); s.remaining_max.resize(sz(m));
    s.row_lo.resize(sz(m)); s.row_hi.resize(sz(m)); s.last.assign(sz(m), -1); s.old_slot.assign(sz(m), -1);
    for (Index j = 0; j < n; ++j) {
        s.lo.push_back(std::ceil(lp.col_lo[sz(j)])); s.hi.push_back(std::floor(lp.col_hi[sz(j)]));
        if (s.lo.back() > s.hi.back()) { s.status = s.result.status = Status::Infeasible; return; }
    }
    const auto& rp = lp.A.pattern.row_ptr(); const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < m; ++i) {
        if (std::isfinite(lp.row_lo[sz(i)])) {
            model::DyadicSum side; side.add(lp.row_lo[sz(i)]); side.add(-tolerance);
            s.row_lo[sz(i)] = rounded_integer(side, true);
        }
        if (std::isfinite(lp.row_hi[sz(i)])) {
            model::DyadicSum side; side.add(lp.row_hi[sz(i)]); side.add(tolerance);
            s.row_hi[sz(i)] = rounded_integer(side, false);
        }
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 a = lp.A.vals[static_cast<std::size_t>(k)]; if (a == 0) continue;
            const Index j = ci[static_cast<std::size_t>(k)]; const Big coefficient(a);
            s.last[sz(i)] = std::max(s.last[sz(i)], j);
            s.remaining_min[sz(i)] += coefficient * Big(a > 0 ? s.lo[sz(j)] : s.hi[sz(j)]);
            s.remaining_max[sz(i)] += coefficient * Big(a > 0 ? s.hi[sz(j)] : s.lo[sz(j)]);
        }
        if (!s.possible(i, 0)) { s.status = s.result.status = Status::Infeasible; return; }
    }
    Impl::Candidate zero; zero.cost.add(lp.maximize ? -lp.obj_offset : lp.obj_offset);
    s.current.emplace(std::vector<Big>{}, std::move(zero));
    s.status = s.result.status = Status::Pending;
    if (n == 0) s.finish();
}
IntegerDomainOracle::~IntegerDomainOracle() = default;
IntegerDomainOracle::Result IntegerDomainOracle::advance(std::uint64_t allowance, const core::CancelToken* cancel, double time_limit_s) {
    auto& s = *impl_;
    if (s.status != Status::Pending) return s.result;
    const auto started = core::SolverClock::now();
    std::uint64_t work = 0;
    while (s.column < s.model.n_cols() && work < allowance && !core::cancel_requested(cancel)) {
        if (time_limit_s > 0 && std::chrono::duration<double>(core::SolverClock::now() - started).count() >= time_limit_s) break;
        if (!s.prepared) s.prepare_column();
        if (s.cursor == s.current.cend()) {
            s.current = std::move(s.next); s.next.clear();
            s.live = std::move(s.after); s.prepared = false; ++s.column;
            if (s.current.empty() || s.column == s.model.n_cols()) { s.finish(); break; }
            continue;
        }
        std::vector<Big> activity; activity.reserve(s.after.size());
        bool feasible = true;
        for (const auto& transition : s.transitions) {
            Big v = transition.source >= 0 ? s.cursor->first[sz(transition.source)] : Big(0);
            v += transition.coefficient * Big(s.value);
            if (!s.possible(transition.row, v)) { feasible = false; break; }
            if (transition.retained) activity.push_back(std::move(v));
        }
        const auto charge = static_cast<std::uint64_t>(s.transitions.size()) + 1;
        work += charge; core::work_add(charge, core::WorkTag::ExactMp);
        if (feasible) {
            auto candidate = s.cursor->second;
            candidate.cost.add_product(s.model.maximize ? -s.model.c[sz(s.column)] : s.model.c[sz(s.column)], s.value);
            auto existing = s.next.find(activity);
            bool improves = existing == s.next.end();
            if (!improves) {
                auto difference = existing->second.cost; difference.negate(); difference.add_sum(candidate.cost);
                improves = difference.sign() < 0;
            }
            if (improves) {
                candidate.path = std::make_shared<const Impl::Path>(Impl::Path{candidate.path, s.column, s.value});
                s.next.insert_or_assign(std::move(activity), std::move(candidate));
            }
        }
        if (s.value < s.hi[sz(s.column)]) s.value += 1;
        else { ++s.cursor; s.value = s.lo[sz(s.column)]; }
    }
    return s.result;
}
} // namespace sor::search
