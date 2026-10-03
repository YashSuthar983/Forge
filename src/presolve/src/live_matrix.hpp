#pragma once

#include "sor/model/lp.hpp"
#include "sor/presolve/presolve.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <vector>

namespace sor::presolve::detail {

struct FlatMap {
    using value_type = std::pair<Index, f64>;
    std::vector<value_type> v;

    using iterator = std::vector<value_type>::iterator;
    using const_iterator = std::vector<value_type>::const_iterator;

    iterator begin() { return v.begin(); }
    iterator end() { return v.end(); }
    const_iterator begin() const { return v.begin(); }
    const_iterator end() const { return v.end(); }
    std::size_t size() const { return v.size(); }
    bool empty() const { return v.empty(); }
    void clear() { v.clear(); }

    iterator lower(Index k) {
        return std::lower_bound(v.begin(), v.end(), k,
                                [](const value_type& a, Index b) { return a.first < b; });
    }
    const_iterator lower(Index k) const {
        return std::lower_bound(v.begin(), v.end(), k,
                                [](const value_type& a, Index b) { return a.first < b; });
    }
    iterator find(Index k) {
        auto it = lower(k);
        return (it != v.end() && it->first == k) ? it : v.end();
    }
    const_iterator find(Index k) const {
        auto it = lower(k);
        return (it != v.end() && it->first == k) ? it : v.end();
    }
    f64& operator[](Index k) {
        auto it = lower(k);
        if (it != v.end() && it->first == k) return it->second;
        return v.insert(it, {k, 0.0})->second;
    }
    const f64& at(Index k) const {
        auto it = find(k);
        if (it == v.end()) throw std::out_of_range("FlatMap::at");
        return it->second;
    }
    void erase(Index k) {
        auto it = lower(k);
        if (it != v.end() && it->first == k) v.erase(it);
    }
};

struct FlatSet {
    std::vector<Index> v;

    using iterator = std::vector<Index>::iterator;
    using const_iterator = std::vector<Index>::const_iterator;

    iterator begin() { return v.begin(); }
    iterator end() { return v.end(); }
    const_iterator begin() const { return v.begin(); }
    const_iterator end() const { return v.end(); }
    std::size_t size() const { return v.size(); }
    bool empty() const { return v.empty(); }
    void clear() { v.clear(); }

    void insert(Index k) {
        auto it = std::lower_bound(v.begin(), v.end(), k);
        if (it == v.end() || *it != k) v.insert(it, k);
    }
    void erase(Index k) {
        auto it = std::lower_bound(v.begin(), v.end(), k);
        if (it != v.end() && *it == k) v.erase(it);
    }
    bool contains(Index k) const {
        auto it = std::lower_bound(v.begin(), v.end(), k);
        return it != v.end() && *it == k;
    }
};

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline void add_interval(f64& lo, f64& hi, f64 a, f64 l, f64 u) {
    if (a >= 0.0) {
        lo += a * l;
        hi += a * u;
    } else {
        lo += a * u;
        hi += a * l;
    }
}

// Mutable CSR + CSC with row/column activities, bound locks, and change queues.
struct LiveMatrix {
    using SparseRow = FlatMap;

    const model::LpProblem* in = nullptr;
    PresolveOptions options{};
    PresolveMap* out = nullptr;
    PresolveStatus* status = nullptr;
    Index* witness_row = nullptr;
    Index* witness_col = nullptr;
    std::string* reason = nullptr;

    Index m = 0;
    Index n = 0;
    std::vector<char> row_active;
    std::vector<char> col_active;
    std::vector<SparseRow> rows;
    std::vector<FlatSet> col_rows;
    std::vector<f64> row_lo;
    std::vector<f64> row_hi;
    std::vector<f64> col_lo;
    std::vector<f64> col_hi;
    std::vector<f64> cost;
    std::vector<f64> fixed;
    f64* obj_offset = nullptr;

    std::vector<f64> act_min;
    std::vector<f64> act_max;
    std::vector<Index> act_min_inf;
    std::vector<Index> act_max_inf;
    std::vector<char> down_lock;
    std::vector<char> up_lock;

    std::deque<Index> changed_rows;
    std::deque<Index> changed_cols;
    std::vector<char> row_queued;
    std::vector<char> col_queued;

    std::vector<std::vector<std::pair<Index, f64>>> original_column_entries;

    void build(const model::LpProblem& problem,
               const PresolveOptions& opts,
               PresolveMap& map_out,
               PresolveStatus& st,
               Index& wr,
               Index& wc,
               std::string& rs,
               std::vector<char>& row_live,
               std::vector<char>& col_live,
               std::vector<f64>& work_lo,
               std::vector<f64>& work_hi,
               std::vector<f64>& work_cost,
               std::vector<f64>& fixed_vals,
               f64& work_obj_offset);

    bool run_until_stable();

    void seed_all_queues();
    void queue_row(Index i);
    void queue_col(Index j);
    void recompute_row_activity(Index i);
    void recompute_col_locks(Index j);

    bool fix_column(Index j, f64 value, DualRecoveryKind kind, Index row = -1,
                    f64 coeff = 0.0, Index record = -1);
    bool remove_redundant_row(Index i);
    bool apply_implied_bounds_row(Index i);
    bool apply_dual_fixing_col(Index j);
    bool try_doubleton_equality(Index i);
    bool try_cost_tight_doubleton(Index j);
    bool try_dominated_columns();
    bool try_duplicate_rows();
    bool try_duplicate_columns();

    f64 stability_tol(f64 scale) const {
        return options.stability_tol_scale * (1.0 + std::fabs(scale));
    }

    void record_column_dual_state(DualRecoveryStep& step, Index row, Index col);
};

bool run_live_presolve_passes(
    const model::LpProblem& in,
    const PresolveOptions& options,
    PresolveMap& out,
    PresolveStatus& status,
    Index& witness_row,
    Index& witness_col,
    std::string& reason,
    std::vector<char>& row_live,
    std::vector<char>& col_live,
    std::vector<f64>& work_lo,
    std::vector<f64>& work_hi,
    std::vector<f64>& work_cost,
    std::vector<f64>& fixed,
    f64& work_obj_offset,
    LiveMatrix& lm);

void advanced_reductions(LiveMatrix& matrix, std::vector<char>& row_live,
    const std::vector<char>& col_live, std::vector<f64>& lo, std::vector<f64>& hi,
    PresolveMap& map, const PresolveOptions& options);

}  // namespace sor::presolve::detail
