// Shared domain fixpoint driver (sor_search/src/domain_fixpoint.cpp), checked
// against a brute-force oracle: random systems of bound implications over small
// integer domains, propagated by several independent propagators. Stable must
// coincide with the unique greatest fixpoint (order independence) and be
// idempotent; Infeasible must coincide with that fixpoint being empty.
#include "sor/search/domain_fixpoint.hpp"

#include "test_helpers.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using sor::search::DomainFixpoint;
using sor::search::FixpointStatus;
using sor::search::FixpointStep;

namespace {

// Rule: if lo[a] >= t then lo[b] >= u ; if hi[a] <= t then hi[b] <= u ;
// and "cross": lo[a] >= t forces hi[b] <= u. All monotone narrowing rules.
struct Rule { int kind, a, b, t, u; };

struct Domain { std::vector<int> lo, hi; };

bool apply_rule(const Rule& r, Domain& d, bool& feasible) {
    bool changed = false;
    auto tighten_lo = [&](int v, int x) { if (x > d.lo[v]) { d.lo[v] = x; changed = true; } };
    auto tighten_hi = [&](int v, int x) { if (x < d.hi[v]) { d.hi[v] = x; changed = true; } };
    if (r.kind == 0 && d.lo[r.a] >= r.t) tighten_lo(r.b, r.u);
    if (r.kind == 1 && d.hi[r.a] <= r.t) tighten_hi(r.b, r.u);
    if (r.kind == 2 && d.lo[r.a] >= r.t) tighten_hi(r.b, r.u);
    for (std::size_t v = 0; v < d.lo.size(); ++v)
        if (d.lo[v] > d.hi[v]) feasible = false;
    return changed;
}

void test_against_oracle() {
    std::mt19937 rng(20260930);
    int stable = 0, infeasible = 0, pending = 0, resumed = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const int n = 3 + static_cast<int>(rng() % 5);
        const int nprop = 1 + static_cast<int>(rng() % 3);
        std::vector<std::vector<Rule>> props(nprop);
        const int nrules = 2 + static_cast<int>(rng() % 10);
        for (int i = 0; i < nrules; ++i) {
            Rule r{static_cast<int>(rng() % 3), static_cast<int>(rng() % n),
                   static_cast<int>(rng() % n), static_cast<int>(rng() % 6),
                   static_cast<int>(rng() % 6)};
            props[rng() % nprop].push_back(r);
        }
        Domain start{std::vector<int>(n, 0), std::vector<int>(n, 5)};
        for (int v = 0; v < n; ++v) {
            start.lo[v] = static_cast<int>(rng() % 3);
            start.hi[v] = 3 + static_cast<int>(rng() % 3);
        }

        // Oracle: apply every rule to a fixpoint in a fixed order.
        Domain ref = start;
        bool ref_feasible = true;
        for (bool moved = true; moved && ref_feasible;) {
            moved = false;
            for (const auto& ps : props)
                for (const auto& r : ps) moved = apply_rule(r, ref, ref_feasible) || moved;
        }

        // Subject: one propagator per rule list; each runs its rules ONCE per
        // call (not to its own fixpoint), so the driver has to re-run them.
        Domain d = start;
        DomainFixpoint fx;
        for (int p = 0; p < nprop; ++p)
            fx.add("p" + std::to_string(p), [&, p]() {
                FixpointStep s;
                for (const auto& r : props[p]) {
                    bool feas = true;
                    s.changed = apply_rule(r, d, feas) || s.changed;
                    if (!feas) { s.feasible = false; return s; }
                }
                s.converged = !s.changed;   // one pass with no change IS this rule set's fixpoint
                return s;
            });
        // A random tiny allowance sometimes: Pending must be resumable.
        const bool tiny = (rng() % 4) == 0;
        auto run = fx.run(tiny ? 2 : 0);
        int guard = 0;
        while (run.status == FixpointStatus::Pending && guard++ < 1000) {
            ++pending;
            ++resumed;
            run = fx.run(tiny ? 2 : 0, run.stale);
        }
        if (ref_feasible) {
            CHECK(run.status == FixpointStatus::Stable);
            CHECK(d.lo == ref.lo);
            CHECK(d.hi == ref.hi);
            // Idempotent: a further run from all-stale changes nothing.
            const Domain before = d;
            const auto again = fx.run(0);
            CHECK(again.status == FixpointStatus::Stable);
            CHECK(again.changing_steps == 0);
            CHECK(d.lo == before.lo && d.hi == before.hi);
            ++stable;
        } else {
            CHECK(run.status == FixpointStatus::Infeasible);
            CHECK(run.infeasible_id >= 0 && run.infeasible_id < nprop);
            ++infeasible;
        }
    }
    std::cout << "DOMAIN_FIXPOINT stable=" << stable << " infeasible=" << infeasible
              << " pending_resumes=" << pending << '\n';
    CHECK(stable > 500);
    CHECK(infeasible > 100);
    CHECK(pending > 20);
    (void)resumed;
}

void test_pending_reports_stale_work() {
    int calls = 0;
    DomainFixpoint fx;
    fx.add("always-changes", [&]() {
        ++calls;
        FixpointStep s;
        s.changed = true;
        s.converged = false;
        return s;
    });
    const auto r = fx.run(5);
    CHECK(r.status == FixpointStatus::Pending);
    CHECK(calls == 5);
    CHECK(r.stale.size() == 1 && r.stale[0] == 1);
}

// Contract: a propagator that reports "not converged" without changing anything
// is UNFINISHED, not finished -- the run is Pending, it is not re-run verbatim,
// and another propagator's change gives it a fresh look.
void test_unfinished_without_change_is_pending() {
    int calls = 0, other_calls = 0;
    bool domain_moved = false;
    DomainFixpoint fx;
    fx.add("stuck-until-moved", [&]() {
        ++calls;
        FixpointStep s;
        s.converged = domain_moved;   // can finish only once the domain has moved
        return s;
    });
    fx.add("mover", [&]() {
        ++other_calls;
        FixpointStep s;
        if (!domain_moved) { domain_moved = true; s.changed = true; }
        return s;
    });
    const auto r = fx.run(0);
    CHECK(r.status == FixpointStatus::Stable);   // the mover's change let it finish
    CHECK(calls == 2);

    int c2 = 0;
    DomainFixpoint g;
    g.add("stuck", [&]() { ++c2; FixpointStep s; s.converged = false; return s; });
    const auto r2 = g.run(0);
    CHECK(r2.status == FixpointStatus::Pending);   // unfinished, honestly reported
    CHECK(c2 == 1);                                // and not spun on
    CHECK(r2.stale.size() == 1 && r2.stale[0] == 1);
}

void test_event_log() {
    sor::search::DomainEventLog log;
    const std::vector<double> lo0 = {0, 0, 0}, hi0 = {5, 5, 5};
    std::vector<double> lo1 = lo0, hi1 = hi0;
    lo1[1] = 2;
    hi1[2] = 3;
    hi1[1] = 4;
    CHECK(log.record_diff(lo0, hi0, lo1, hi1, 3, 7) == 3);
    const auto first = log.size();
    lo1[0] = 1;
    CHECK(log.record_diff(lo0, hi0, lo1, hi1, 3, 8) == 4);   // diffs against the given base
    const auto v = log.vars_since(0);
    CHECK(v.size() == 3);
    const auto w = log.vars_since(first);
    CHECK(w.size() == 3 && w[0] == 0);   // column 0 was first in the second batch
    CHECK(log.events()[0].level == 3 && log.events()[0].source == 7);
}

}  // namespace

int main() {
    test_unfinished_without_change_is_pending();
    test_event_log();
    test_against_oracle();
    test_pending_reports_stale_work();
    return sor::test::finish("test_domain_fixpoint");
}
