// P11: the persistent conflict store. Unit checks pin the propagation rules
// and the refusal of clauses that are not exactly implied; the oracle stress
// solves random integer programs with the store learning from every
// infeasible node and requires the proven answer to equal exhaustive
// enumeration, with and without the store.
#include "sor/certify/finalize.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/conflict_store.hpp"
#include "sor/search/prop_trail.hpp"
#include "sor/search/propagate.hpp"
#include "sor/sparse/csr.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>

using namespace sor::search;
using sor::core::Status;
using sor::test::oracle::make_branching_ip;
using sor::test::oracle::make_random_milp;
using sor::test::oracle::solve_oracle;

namespace {

const std::vector<double> kRootLo(4, 0.0), kRootHi(4, 10.0);

void test_unit_propagation_and_conflict() {
    ConflictStore st(4, 1 << 20, 16);
    // x0 >= 3  or  x1 <= 1  or  x2 >= 5
    CHECK(st.add({{0, false, 3.0}, {1, true, 1.0}, {2, false, 5.0}}, kRootLo, kRootHi));
    auto lo = kRootLo, hi = kRootHi;
    // Nothing decided: nothing forced.
    CHECK(st.propagate(lo, hi, kRootLo, kRootHi));
    CHECK(lo == kRootLo && hi == kRootHi);
    // x0 <= 2 (literal 1 false) and x1 >= 2 (literal 2 false): x2 >= 5 forced.
    hi[0] = 2.0; lo[1] = 2.0;
    CHECK(st.propagate(lo, hi, kRootLo, kRootHi));
    CHECK(lo[2] == 5.0);
    CHECK(st.stats().propagations == 1 && st.stats().tightenings == 1);
    // All three false: the node is closed.
    auto lo2 = kRootLo, hi2 = kRootHi;
    hi2[0] = 2.0; lo2[1] = 2.0; hi2[2] = 4.0;
    CHECK(!st.propagate(lo2, hi2, kRootLo, kRootHi));
    CHECK(st.stats().conflicts == 1);
    // A satisfied clause forces nothing even when the rest is false.
    auto lo3 = kRootLo, hi3 = kRootHi;
    lo3[0] = 3.0; lo3[1] = 2.0; hi3[2] = 4.0;
    CHECK(st.propagate(lo3, hi3, kRootLo, kRootHi));
    CHECK(hi3[2] == 4.0 && lo3[0] == 3.0);
}

// Forcing can cascade through several clauses.
void test_chained_propagation() {
    ConflictStore st(4, 1 << 20, 16);
    CHECK(st.add({{0, true, 0.0}, {1, false, 1.0}}, kRootLo, kRootHi));   // x0<=0 or x1>=1
    CHECK(st.add({{1, true, 0.0}, {2, false, 4.0}}, kRootLo, kRootHi));   // x1<=0 or x2>=4
    auto lo = kRootLo, hi = kRootHi;
    lo[0] = 1.0;   // first literal false -> x1 >= 1 -> second clause -> x2 >= 4
    CHECK(st.propagate(lo, hi, kRootLo, kRootHi));
    CHECK(lo[1] == 1.0 && lo[2] == 4.0);
}

void test_exact_integer_complements_from_decisions() {
    PropTrail t;
    // decisions x0 <= 2 and x1 >= 4 (a branch on x0 twice keeps the tighter)
    t.push(0, BoundDir::Upper, 5.0, 10.0, ReasonKind::Branch, -1, 1);
    t.push(0, BoundDir::Upper, 2.0, 5.0, ReasonKind::Branch, -1, 2);
    t.push(1, BoundDir::Lower, 4.0, 0.0, ReasonKind::Branch, -1, 3);
    t.push(2, BoundDir::Lower, 1.0, 0.0, ReasonKind::Row, 7, 3);   // implied, ignored
    const std::vector<char> is_int{1, 1, 1, 1};
    const auto lits = negated_branch_decisions(t, is_int, kRootLo, kRootHi);
    CHECK(lits.size() == 2);
    ConflictStore st(4, 1 << 20, 16);
    CHECK(st.add(lits, kRootLo, kRootHi));
    // Under x0 <= 2 the clause forces x1 <= 3, its exact complement of x1 >= 4.
    auto lo = kRootLo, hi = kRootHi;
    hi[0] = 2.0;
    CHECK(st.propagate(lo, hi, kRootLo, kRootHi));
    CHECK(hi[1] == 3.0);
    // Under x1 >= 4 it forces x0 >= 3.
    auto lo2 = kRootLo, hi2 = kRootHi;
    lo2[1] = 4.0;
    CHECK(st.propagate(lo2, hi2, kRootLo, kRootHi));
    CHECK(lo2[0] == 3.0);
    // Both decisions: closed.
    auto lo3 = kRootLo, hi3 = kRootHi;
    hi3[0] = 2.0; lo3[1] = 4.0;
    CHECK(!st.propagate(lo3, hi3, kRootLo, kRootHi));
}

void test_invalid_or_incomplete_reasons_rejected() {
    const std::vector<char> is_int{1, 0, 1, 1};
    // A decision on a continuous column: the box is not exactly the decisions.
    PropTrail cont;
    cont.push(1, BoundDir::Upper, 2.0, 10.0, ReasonKind::Branch, -1, 1);
    CHECK(negated_branch_decisions(cont, is_int, kRootLo, kRootHi).empty());
    // A fractional decision bound.
    PropTrail frac;
    frac.push(0, BoundDir::Upper, 2.5, 10.0, ReasonKind::Branch, -1, 1);
    CHECK(negated_branch_decisions(frac, is_int, kRootLo, kRootHi).empty());
    // A vacuous clause (a literal the root box always satisfies) and an
    // over-long one are refused; a fractional literal is refused.
    ConflictStore st(4, 1 << 20, 2);
    CHECK(!st.add({{0, true, 10.0}, {1, false, 3.0}}, kRootLo, kRootHi));
    CHECK(!st.add({{0, false, 1.0}, {1, false, 3.0}, {2, false, 1.0}}, kRootLo, kRootHi));
    CHECK(!st.add({{0, false, 1.5}}, kRootLo, kRootHi));
    CHECK(!st.add({}, kRootLo, kRootHi));
    CHECK(st.size() == 0);
    // A literal the root already excludes is dropped; the clause that remains has
    // one live literal, i.e. it is a FACT (Unit): reported for the caller to apply
    // to the root box, not stored as a clause.
    std::vector<ConflictLiteral> norm;
    CHECK(st.add_typed({{0, true, -1.0}, {1, false, 3.0}}, kRootLo, kRootHi, &norm) ==
          ClauseResult::Unit);
    CHECK(norm.size() == 1 && norm[0].var == 1 && !norm[0].upper && norm[0].bound == 3.0);
    CHECK(st.size() == 0);
    // Duplicates are refused.
    CHECK(!st.add({{1, false, 3.0}}, kRootLo, kRootHi));
}

void test_eviction_under_budget() {
    ConflictStore st(4, 600, 8);
    std::size_t added = 0;
    for (int b = 1; b <= 9; ++b)
        added += st.add({{0, false, double(b)}, {1, true, double(b)}}, kRootLo, kRootHi);
    CHECK(added == 9);
    CHECK(st.stats().evicted > 0);
    CHECK(st.stats().bytes <= 600);
    CHECK(st.size() + st.stats().evicted == 9);
    ConflictStore tiny(4, 1, 16);
    const auto result = tiny.add_typed({{0, false, 1.0}, {1, true, 2.0}}, kRootLo, kRootHi);
    CHECK(result == ClauseResult::Rejected);
    CHECK(!clause_retained(result) && tiny.empty());
}

void test_deferred_scan_reports_pending() {
    ConflictStore st(4, 1 << 20, 16);
    CHECK(st.add({{0, true, 0.0}, {1, false, 1.0}}, kRootLo, kRootHi));
    for (int i = 0; i < 2000; ++i) {
        auto lo = kRootLo, hi = kRootHi;
        CHECK(st.propagate(lo, hi, kRootLo, kRootHi));
    }
    auto lo = kRootLo, hi = kRootHi;
    lo[0] = 1.0;
    bool converged = true;
    CHECK(st.propagate(lo, hi, kRootLo, kRootHi, &converged));
    CHECK(!converged && lo[1] == 0.0);
    // The scheduled complete scan must still enforce the clause.
    for (int i = 0; i < 16 && !converged; ++i)
        CHECK(st.propagate(lo, hi, kRootLo, kRootHi, &converged));
    CHECK(converged && lo[1] == 1.0);
}

// Conflict analysis shrinks the clause to the decisions that matter.
void test_analysis_keeps_only_relevant_decisions() {
    // r0: x0 + x1 >= 2   r1: x1 + x2 <= 1   (binaries x0..x3)
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(2, 4, {0, 0, 1, 1}, {0, 1, 1, 2}, {1.0, 1.0, 1.0, 1.0});
    lp.row_lo = {2.0, -1e30};
    lp.row_hi = {1e30, 1.0};
    lp.col_lo = {0, 0, 0, 0};
    lp.col_hi = {1, 1, 1, 1};
    lp.is_integer = {true, true, true, true};
    lp.c = {0, 0, 0, 0};
    const std::vector<double> rlo(4, 0.0), rhi(4, 1.0);
    const std::vector<char> is_int{1, 1, 1, 1};
    PropTrail t;
    auto lo = rlo, hi = rhi;
    // Decisions: x3 = 1 (irrelevant), x0 <= 0 (cause). Then propagate.
    t.push(3, BoundDir::Lower, 1.0, 0.0, ReasonKind::Branch, -1, 1);
    lo[3] = 1.0;
    t.push(0, BoundDir::Upper, 0.0, 1.0, ReasonKind::Branch, -1, 2);
    hi[0] = 0.0;
    const auto res = propagate_bounds_trail(lp, lo, hi, &t, 2);
    CHECK(!res.feasible);
    CHECK(res.conflict_var >= 0);
    const auto clause = explain_conflict_clause(t, lp, lp.n_rows(), is_int, rlo, rhi,
                                                lo, hi, res.conflict_var);
    // Only x0 <= 0 explains the conflict: the clause is x0 >= 1.
    CHECK(clause.size() == 1);
    if (clause.size() == 1) {
        CHECK(clause[0].var == 0 && !clause[0].upper && clause[0].bound == 1.0);
    }
    // The blunt fallback keeps both decisions.
    CHECK(negated_branch_decisions(t, is_int, rlo, rhi).size() == 2);
    // An untraced tightening on a column the explanation needs: give up.
    auto lo2 = lo, hi2 = hi;
    hi2[1] = 0.0;   // x1's box moved without a trail entry
    CHECK(explain_conflict_clause(t, lp, lp.n_rows(), is_int, rlo, rhi, lo2, hi2,
                                  res.conflict_var).empty());
    // A reason row beyond the global rows (a local cut): give up.
    CHECK(explain_conflict_clause(t, lp, 0, is_int, rlo, rhi, lo, hi,
                                  res.conflict_var).empty());
}

BabOptions store_options(bool store) {
    BabOptions o;
    o.para_bab.threads = 1;
    o.policy = MilpPolicy::Latest;
    o.conflict_cut.conflict_store = store;
    o.cuts_enabled = false;
    o.structural_presolve.enabled = false;
    o.mip_presolve = o.probing = o.symmetry = false;
    o.feasibility_jump = false;
    o.sub_mip_lns = false;
    o.rounding_heuristic = false;
    o.lp_rounding_repair = false;
    o.integer_dive = false;
    o.integer_neighborhood = false;
    o.balans.enabled = false;
    o.kernel_pump.enabled = false;
    o.mrens.enabled = false;
    o.hgtsm.enabled = false;
    o.time_limit_s = 20.0;
    o.max_nodes = 20000;
    o.gap_tol = o.abs_gap_tol = 1e-9;
    return o;
}

void test_restart_clause_outcomes_reach_the_root() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2,
        std::vector<sor::core::Index>{0, 0},
        std::vector<sor::core::Index>{0, 1}, std::vector<double>{1.0, 1.0});
    lp.c = {1.0, 2.0};
    lp.col_lo = {0.0, 0.0}; lp.col_hi = {1.0, 1.0};
    lp.row_lo = {1.5}; lp.row_hi = {sor::model::kInf};
    lp.is_integer = {true, true};
    // Both units are valid for the model's sole integer-feasible point (1,1).
    std::vector<std::vector<ConflictLiteral>> clauses{
        {{0, false, 1.0}}, {{1, false, 1.0}}};
    auto opts = store_options(true);
    opts.initial_clauses = &clauses;
    BabDiagnostics d;
    auto raw = solve_milp(lp, opts, d);
    CHECK(d.clause_units == 2);
    CHECK(d.conflict_store_root_units == 2);
    auto result = sor::certify::finalize_result(std::move(raw), milp_evidence(d, opts));
    CHECK(result.status == Status::Optimal);
    CHECK_NEAR(result.objective, solve_oracle(lp).objective, 1e-8);

    // x+y=1/2 has LP points but no integer point. Valid carried clauses
    // x>=1 and x<=0 contradict one another in the global root box.
    lp.row_lo = lp.row_hi = {0.5};
    CHECK(!solve_oracle(lp).feasible);
    clauses = {{{0, false, 1.0}}, {{0, true, 0.0}}};
    d = BabDiagnostics{};
    raw = solve_milp(lp, opts, d);
    CHECK(d.clause_contradictions == 1);
    CHECK(d.nodes == 0);
    CHECK(d.globally_proved);
    result = sor::certify::finalize_result(std::move(raw), milp_evidence(d, opts));
    CHECK(result.status == Status::Infeasible);
}

void oracle_stress() {
    std::uint64_t added = 0, forced = 0, tightened = 0, closed = 0, models = 0;
    std::uint64_t fixpoint_rounds = 0, root_units = 0;
    for (const bool ip : {true, false})
        for (std::uint32_t seed = 1; seed <= (ip ? 150u : 400u); ++seed) {
            const auto lp = ip ? make_branching_ip(seed) : make_random_milp(seed);
            const auto oracle = solve_oracle(lp);
            for (const bool store : {false, true}) {
                auto o = store_options(store);
                BabDiagnostics d;
                auto raw = solve_milp(lp, o, d);
                const auto res = sor::certify::finalize_result(
                    std::move(raw), milp_evidence(d, o));
                if (store) {
                    ++models;
                    added += d.conflict_store.added;
                    forced += d.conflict_store.propagations;
                    tightened += d.conflict_store.tightenings;
                    closed += d.conflict_store.conflicts;
                    fixpoint_rounds += d.prop_fixpoint_rounds;
                    root_units += d.conflict_store_root_units;
                } else {
                    CHECK(d.conflict_store.added == 0);
                }
                if (res.status == Status::Optimal) {
                    CHECK(oracle.feasible);
                    const bool same = !oracle.feasible ||
                        std::fabs(res.objective - oracle.objective) <=
                            1e-6 * (1.0 + std::fabs(oracle.objective));
                    if (!same)
                        std::cout << "WRONG seed=" << seed << " ip=" << ip << " store=" << store
                                  << " got " << res.objective << " oracle " << oracle.objective << '\n';
                    CHECK(same);
                } else if (res.status == Status::Infeasible) {
                    CHECK(!oracle.feasible);
                }
            }
        }
    std::cout << "CONFLICT_STORE models=" << models << " clauses=" << added << " forced=" << forced
              << " tightened=" << tightened << " closed=" << closed
              << " fixpoint_rounds=" << fixpoint_rounds << " root_units=" << root_units << '\n';
    CHECK(added + root_units > 0);
}

// Typed outcomes, and the length policy applies to the NORMALISED clause.
void test_typed_outcomes() {
    ConflictStore st(4, 1 << 20, 2);   // storage limit: 2 literals
    std::vector<ConflictLiteral> norm;
    // Input has 4 literals but two are false at the root (x0 >= 11 with hi 10;
    // x1 <= -1 with lo 0) and two merge on one column: normalises to 2 literals
    // and is stored although the RAW length exceeds the limit.
    auto r = st.add_typed({{0, false, 11.0}, {1, true, -1.0}, {2, true, 3.0}, {2, true, 5.0}},
                          kRootLo, kRootHi, &norm);
    CHECK(r == ClauseResult::Inserted || r == ClauseResult::Unit);
    CHECK(norm.size() == 1);   // {x2 <= 5}
    CHECK(r == ClauseResult::Unit);
    CHECK(norm[0].var == 2 && norm[0].upper && norm[0].bound == 5.0);
    // A genuinely long normalised clause is Rejected, not silently dropped as if learned.
    CHECK(st.add_typed({{0, true, 3.0}, {1, true, 3.0}, {2, true, 3.0}}, kRootLo, kRootHi) ==
          ClauseResult::Rejected);
    // Identical clause again -> AlreadyPresent (retained information).
    CHECK(st.add_typed({{0, true, 3.0}, {1, false, 2.0}}, kRootLo, kRootHi) == ClauseResult::Inserted);
    const auto again = st.add_typed({{1, false, 2.0}, {0, true, 3.0}}, kRootLo, kRootHi);
    CHECK(again == ClauseResult::AlreadyPresent);
    CHECK(clause_retained(again));
    // Vacuous (x0 <= 10 is always true at the root) -> Redundant; retains nothing.
    const auto vac = st.add_typed({{0, true, 10.0}, {1, false, 2.0}}, kRootLo, kRootHi);
    CHECK(vac == ClauseResult::Redundant);
    CHECK(!clause_retained(vac));
    // Every literal false at the root -> Contradiction.
    const auto con = st.add_typed({{0, false, 11.0}, {1, true, -1.0}}, kRootLo, kRootHi);
    CHECK(con == ClauseResult::Contradiction);
    CHECK(!clause_retained(con));
    // Malformed (fractional bound) -> Rejected.
    CHECK(st.add_typed({{0, true, 2.5}, {1, false, 2.0}}, kRootLo, kRootHi) == ClauseResult::Rejected);
}

}  // namespace

int main() {
    test_restart_clause_outcomes_reach_the_root();
    test_typed_outcomes();
    test_unit_propagation_and_conflict();
    test_chained_propagation();
    test_exact_integer_complements_from_decisions();
    test_invalid_or_incomplete_reasons_rejected();
    test_eviction_under_budget();
    test_deferred_scan_reports_pending();
    test_analysis_keeps_only_relevant_decisions();
    oracle_stress();
    return sor::test::finish("test_conflict_store");
}
