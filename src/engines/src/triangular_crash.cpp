#include "triangular_crash.hpp"
#include <algorithm>
#include <cmath>
namespace sor::engines::detail {
namespace { std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
std::size_t sz(Offset i) { return static_cast<std::size_t>(i); }
}
TriangularCrashStats triangular_crash(const model::LpProblem& p, const sparse::CscMatrix& ac,
    std::span<const f64> lo, std::span<const f64> hi, std::span<const f64> ptol,
    std::vector<Index>& basis, std::vector<Index>& slot_of,
    std::vector<NonbasicStatus>& st, std::vector<f64>& value,
    const std::vector<f64>* eligible_costs) {
    const auto ns = p.n_cols(), m = p.n_rows();
    const auto& acp = ac.pattern.col_ptr(); const auto& ari = ac.pattern.row_idx();
    constexpr f64 kInf = model::kInf;
    TriangularCrashStats stats;
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    const auto& vv = p.A.vals;
    std::vector<f64> activity(sz(m), 0.0);
    std::vector<f64> row_max(sz(m), 0.0), col_max(sz(ns), 0.0);
    for (Index i = 0; i < m; ++i) {
        for (Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = vv[sz(k)];
            activity[sz(i)] += a * value[sz(j)];
            row_max[sz(i)] = std::max(row_max[sz(i)], std::fabs(a));
            col_max[sz(j)] = std::max(col_max[sz(j)], std::fabs(a));
        }
    }
    const auto row_violation = [&](Index i, f64 a) {

        const f64 l = lo[sz(ns + i)], u = hi[sz(ns + i)];
        const f64 t = ptol[sz(ns + i)];
        if (l > -kInf && a < l - t) return l - a;
        if (u <  kInf && a > u + t) return a - u;
        return 0.0;
    };
    const auto total_violation = [&]() {

        f64 total = 0.0;
        for (Index i = 0; i < m; ++i)
            if (st[sz(ns + i)] == NonbasicStatus::Basic)
                total += row_violation(i, activity[sz(i)]);
        return total;
    };

    stats.infeasibility_before = total_violation();
    // Snapshot the all-logical start so a crash that does not improve
    // total merit (or that later fails factorization) can be refused.
    const auto crash_basis_save = basis;
    const auto crash_slot_save = slot_of;
    const auto crash_st_save = st;
    const auto crash_value_save = value;
    const std::vector<f64> crash_activity_save = activity;
    std::vector<Index> order(sz(m));
    for (Index i = 0; i < m; ++i) order[sz(i)] = i;
    std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) {

        return row_violation(a, activity[sz(a)]) >
               row_violation(b, activity[sz(b)]);
    });
    std::vector<char> claimed_row(sz(m), 0);

    for (const Index i : order) {
        const f64 old_i = row_violation(i, activity[sz(i)]);
        if (!(old_i > 0.0)) continue;
        const f64 target = activity[sz(i)] < lo[sz(ns + i)]
                               ? lo[sz(ns + i)]
                               : hi[sz(ns + i)];
        Index best_j = -1;
        f64 best_delta = 0.0;
        f64 best_gain = 0.0;
        f64 best_score = 0.0;

        for (Offset rk = rp[sz(i)]; rk < rp[sz(i) + 1]; ++rk) {
            const Index j = ci[sz(rk)];
            const f64 aij = vv[sz(rk)];
            if ((eligible_costs && (*eligible_costs)[sz(j)] != 0) ||
                st[sz(j)] == NonbasicStatus::Basic ||
                lo[sz(j)] == hi[sz(j)])
                continue;
            const f64 scale = std::max(row_max[sz(i)], col_max[sz(j)]);
            if (!(std::fabs(aij) >= 1e-4 * scale)) continue;

            // Preserve the triangular construction exactly: numerical
            // smallness is insufficient because factorization sees the
            // stored coefficient, not our intent.
            bool touches_claimed = false;
            for (Offset ck = acp[sz(j)]; ck < acp[sz(j) + 1]; ++ck) {
                if (claimed_row[sz(ari[sz(ck)])]) {
                    touches_claimed = true;
                    break;
                }
            }
            if (touches_claimed) continue;

            const f64 delta = (target - activity[sz(i)]) / aij;
            const f64 next = value[sz(j)] + delta;
            if (!std::isfinite(next) ||
                (lo[sz(j)] > -kInf && next < lo[sz(j)] - ptol[sz(j)]) ||
                (hi[sz(j)] <  kInf && next > hi[sz(j)] + ptol[sz(j)]))
                continue;

            f64 gain = 0.0;
            for (Offset ck = acp[sz(j)]; ck < acp[sz(j) + 1]; ++ck) {
                const Index r = ari[sz(ck)];
                if (claimed_row[sz(r)]) continue;
                const f64 before = row_violation(r, activity[sz(r)]);
                const f64 after = row_violation(
                    r, activity[sz(r)] + ac.vals[sz(ck)] * delta);
                gain += before - after;
            }
            if (!(gain > 1e-12 * (1.0 + old_i))) continue;
            const f64 score = gain /
                (1.0 + 0.02 * static_cast<f64>(
                    acp[sz(j) + 1] - acp[sz(j)] - 1));
            if (score > best_score ||
                (score == best_score && (best_j < 0 || j < best_j))) {
                best_j = j;
                best_delta = delta;
                best_gain = gain;
                best_score = score;
            }
        }
        if (best_j < 0 || !(best_gain > 0.0)) continue;

        for (Offset ck = acp[sz(best_j)]; ck < acp[sz(best_j) + 1]; ++ck)
            activity[sz(ari[sz(ck)])] += ac.vals[sz(ck)] * best_delta;
        const Index logical = ns + i;
        basis[sz(i)] = best_j;
        slot_of[sz(best_j)] = i;
        st[sz(best_j)] = NonbasicStatus::Basic;
        slot_of[sz(logical)] = -1;
        if (target == lo[sz(logical)]) {
            st[sz(logical)] = NonbasicStatus::AtLower;
            value[sz(logical)] = lo[sz(logical)];
        } else {
            st[sz(logical)] = NonbasicStatus::AtUpper;
            value[sz(logical)] = hi[sz(logical)];
        }
        claimed_row[sz(i)] = 1;
        ++stats.columns;
    }
    stats.infeasibility_after = total_violation();
    // Refuse the whole crash if total primal infeasibility did not
    // strictly improve. Per-column gains can sum to a non-improvement
    // once claimed-row interactions are accounted for at the end.
    if (!(stats.infeasibility_after <
          stats.infeasibility_before -
              1e-12 * (1.0 + stats.infeasibility_before))) {
        basis = crash_basis_save;
        slot_of = crash_slot_save;
        st = crash_st_save;
        value = crash_value_save;
        activity = crash_activity_save;
        stats.columns = 0;
        stats.infeasibility_after =
            stats.infeasibility_before;
    }
    return stats;
}
}
