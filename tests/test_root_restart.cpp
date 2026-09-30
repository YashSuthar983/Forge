// Root restart (P7): reduced-cost fixing against an incumbent shrinks the
// root box, the wrapper presolves the tightened model again and re-solves
// with the old incumbent as a cutoff. Forced on tiny models (a restart at one
// fixed column) and compared with exhaustive enumeration: same optimum, same
// feasibility, valid claims, and a postsolved point that satisfies the
// ORIGINAL model even though the incumbent was excluded from the restarted
// box.
#include "sor/certify/finalize.hpp"
#include "sor/search/bab.hpp"
#include "sor/sparse/csr.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

using sor::core::Status;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::test::oracle::make_random_milp;
using sor::test::oracle::solve_oracle;

namespace {

void test_forced_restarts_match_oracle() {
    std::uint64_t restarted_runs = 0, restarts = 0, optimal = 0, infeasible = 0;
    std::uint64_t carried_cut_rows = 0, carried_clauses = 0, dropped_clauses = 0,
                  carried_incumbents = 0, rejected_incumbents = 0, emptied_boxes = 0;
    for (std::uint32_t seed = 1; seed <= 3000; ++seed) {
        auto lp = make_random_milp(seed);
        lp.obj_offset = static_cast<double>(static_cast<int>(seed % 5) - 2);
        const auto oracle = solve_oracle(lp);
        BabOptions o;
        o.para_bab.threads = 1;
        o.structural_presolve.enabled = true;
        o.root_restart = true;
        o.root_restart_max = 3;
        o.root_restart_min_fixed = 1;
        o.root_restart_min_fixed_frac = 0.0;
        o.gap_tol = o.abs_gap_tol = 1e-9;
        o.max_nodes = 5000;
        o.time_limit_s = 0.0;
        BabDiagnostics d;
        auto raw = sor::search::solve_milp(lp, o, d);
        const auto ev = sor::search::milp_evidence(d, o);
        const auto res = sor::certify::finalize_result(raw, ev);
        restarts += d.root_restarts;
        restarted_runs += d.root_restarts > 0;
        carried_cut_rows += d.restart_cut_rows_carried;
        carried_clauses += d.restart_clauses_carried;
        dropped_clauses += d.restart_clauses_dropped;
        carried_incumbents += d.restart_incumbent_carried;
        rejected_incumbents += d.restart_incumbent_rejected;
        emptied_boxes += d.restart_clauses_emptied;
        const double sense = lp.maximize ? -1.0 : 1.0;
        if (res.status == Status::Optimal) {
            ++optimal;
            CHECK(oracle.feasible);
            const bool close = oracle.feasible &&
                std::fabs(res.objective - oracle.objective) <=
                    1e-6 * (1.0 + std::fabs(oracle.objective));
            if (!close)
                std::cout << "WRONG OPTIMAL seed=" << seed << " got " << res.objective
                          << " oracle " << oracle.objective << " restarts="
                          << d.root_restarts << '\n';
            CHECK(close);
            CHECK(lp.max_row_violation(res.x) <= 1e-7);
            CHECK(lp.max_bound_violation(res.x) <= 1e-7);
        }
        if (res.status == Status::Infeasible) {
            ++infeasible;
            if (oracle.feasible)
                std::cout << "WRONG INFEASIBLE seed=" << seed << " restarts="
                          << d.root_restarts << " oracle=" << oracle.objective
                          << " reason=" << raw.termination_reason << '\n';
            CHECK(!oracle.feasible);
        }
        if (oracle.feasible) {
            // A finished solve must not lose feasible models.
            CHECK(res.status != Status::Infeasible);
            if (std::isfinite(d.dual_bound))
                CHECK(sense * d.dual_bound <=
                      sense * oracle.objective + 1e-7 * (1.0 + std::fabs(oracle.objective)));
            if (std::isfinite(raw.objective) && !raw.x.empty()) {
                CHECK(lp.max_row_violation(raw.x) <= 1e-7);
                CHECK(sense * raw.objective >= sense * oracle.objective - 1e-6);
            }
        }
    }
    std::cout << "ROOT_RESTART_STRESS runs_with_restart=" << restarted_runs
              << " restarts=" << restarts << " optimal=" << optimal
              << " infeasible=" << infeasible << " carried_cut_rows=" << carried_cut_rows
              << " carried_clauses=" << carried_clauses
              << " dropped_clauses=" << dropped_clauses
              << " carried_incumbents=" << carried_incumbents
              << " rejected_incumbents=" << rejected_incumbents
              << " emptied_boxes=" << emptied_boxes << '\n';
    CHECK(restarted_runs > 50);
    CHECK(carried_cut_rows > 0);   // the stage's cut rows reach the restarted model
    CHECK(optimal > 500);
}

}  // namespace

// Bound bookkeeping across stages: an unknown stage bound stays unknown (it
// never becomes the incumbent), a restarted stage certifies only
// min(stage bound, cutoff), and a stronger earlier certificate is kept.
void test_stage_bound_combination() {
    using sor::search::restarted_stage_bound;
    const double nan = std::nan("");
    const double inf = std::numeric_limits<double>::infinity();
    // Nothing certified before, stage bound unknown: still nothing.
    CHECK(std::isnan(restarted_stage_bound(false, nan, nan, 10.0)));
    CHECK(std::isnan(restarted_stage_bound(false, nan, -inf, 10.0)));
    // An earlier certificate survives an unknown stage.
    CHECK(restarted_stage_bound(false, 7.0, nan, 10.0) == 7.0);
    // Minimisation: the stage bound above its cutoff certifies only the cutoff.
    CHECK(restarted_stage_bound(false, nan, 12.0, 10.0) == 10.0);
    CHECK(restarted_stage_bound(false, nan, 8.0, 10.0) == 8.0);
    // The tighter of old and new wins.
    CHECK(restarted_stage_bound(false, 9.0, 8.0, 10.0) == 9.0);
    CHECK(restarted_stage_bound(false, 5.0, 8.0, 10.0) == 8.0);
    // Stage 0 has no cutoff.
    CHECK(restarted_stage_bound(false, nan, 8.0, nan) == 8.0);
    // Maximisation mirrors it.
    CHECK(restarted_stage_bound(true, nan, 8.0, 10.0) == 10.0);
    CHECK(restarted_stage_bound(true, nan, 12.0, 10.0) == 12.0);
    CHECK(restarted_stage_bound(true, 11.0, 12.0, 10.0) == 11.0);
    CHECK(std::isnan(restarted_stage_bound(true, nan, nan, 10.0)));
}

// map_restart_payload: merges no longer discard everything, and a clause whose
// every literal a presolve-fixed value falsifies proves the restarted box empty.
void test_payload_mapping() {
    using sor::search::ConflictLiteral;
    using sor::search::RestartPayload;
    sor::model::LpProblem stage;
    const int ns = 5;
    stage.A = sor::sparse::from_triplets(1, ns, {0, 0, 0, 0, 0}, {0, 1, 2, 3, 4},
                                         {1.0, 1.0, 1.0, 1.0, 1.0});
    stage.c.assign(ns, 0.0);
    stage.col_lo.assign(ns, 0.0);
    stage.col_hi.assign(ns, 1.0);
    stage.row_lo = {-sor::model::kInf};
    stage.row_hi = {2.0};
    stage.is_integer.assign(ns, true);

    RestartPayload in;
    in.pseudocosts.down_sum = {1, 2, 3, 4, 5};
    in.pseudocosts.up_sum = {1, 2, 3, 4, 5};
    in.pseudocosts.down_count = {1, 1, 1, 1, 1};
    in.pseudocosts.up_count = {1, 1, 1, 1, 1};
    // c0 names the merged column 1 -> dropped.  c1 names columns 0 and 2 -> kept.
    // c2: column 4 fixed at 0 falsifies x4 >= 1 and column 3 fixed at 1 falsifies
    // x3 <= 0 -> every literal falsified -> the restarted box is empty.
    in.clauses = {{{1, true, 0.0}}, {{0, false, 1.0}, {2, true, 0.0}},
                  {{4, false, 1.0}, {3, true, 0.0}}};

    sor::search::MilpPresolveResult pre2;
    pre2.reduced_col = {0, -1, 1, -1, -1};
    pre2.eliminated = {0, 1, 0, 1, 1};
    pre2.fixed_value = {0.0, 0.0, 0.0, 1.0, 0.0};
    pre2.reduced.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    pre2.reduced.c.assign(2, 0.0);
    pre2.reduced.col_lo.assign(2, 0.0);
    pre2.reduced.col_hi.assign(2, 1.0);
    pre2.reduced.row_lo = {-sor::model::kInf};
    pre2.reduced.row_hi = {2.0};
    pre2.reduced.is_integer.assign(2, true);
    sor::search::ColumnMerge mg;
    mg.members = {1};
    pre2.column_merges.push_back(mg);

    RestartPayload out;
    std::uint64_t dropped = 0;
    bool empty = false;
    sor::search::map_restart_payload(in, pre2, stage, out, dropped, empty);
    CHECK(dropped == 1);              // the clause naming the merged column
    CHECK(out.clauses.size() == 1);   // the unaffected one survived
    CHECK(out.clauses[0].size() == 2);
    CHECK(empty);                     // the all-falsified clause is detected
    CHECK(out.pseudocosts.down_sum[0] == 1.0);   // unaffected history travels
    CHECK(out.pseudocosts.down_sum[1] == 3.0);

    in.clauses.pop_back();            // without the falsified clause the box is not empty
    RestartPayload out2;
    dropped = 0;
    sor::search::map_restart_payload(in, pre2, stage, out2, dropped, empty);
    CHECK(!empty);
}

// An outside cutoff (a restart's kept point, another arm's incumbent) reaches
// every consumer that only needs a number to beat -- root/node LP objective
// limits, pruning, reduced-cost fixing -- even though the stage has no point of
// its own. Soundness against the exhaustive oracle: with the cutoff at the true
// optimum nothing better exists and the solve must not claim Infeasible or a
// point that beats the oracle; with a loose cutoff the optimum must still be
// found.
void test_external_cutoff_is_sound() {
    std::uint64_t foreign_prunes = 0, rc_fixed = 0, tight_runs = 0, loose_optimal = 0;
    for (std::uint32_t seed = 1; seed <= 1500; ++seed) {
        auto lp = make_random_milp(seed);
        lp.obj_offset = static_cast<double>(static_cast<int>(seed % 5) - 2);
        const auto oracle = solve_oracle(lp);
        if (!oracle.feasible) continue;
        const double opt = lp.objective(oracle.x);
        for (int loose = 0; loose < 2; ++loose) {
            BabOptions o;
            o.para_bab.threads = 1;
            o.structural_presolve.enabled = false;
            o.root_restart = false;
            o.gap_tol = o.abs_gap_tol = 1e-9;
            o.max_nodes = 5000;
            o.time_limit_s = 0.0;
            o.initial_cutoff = loose ? (lp.maximize ? opt - 1000.0 : opt + 1000.0) : opt;
            BabDiagnostics d;
            auto raw = sor::search::solve_milp(lp, o, d);
            foreign_prunes += d.foreign_cutoff_prunes;
            rc_fixed += d.rc_columns_fixed;
            if (!loose) {
                ++tight_runs;
                CHECK(raw.proposed_status != Status::Infeasible);   // foreign cutoff never proves that
                if (!raw.x.empty()) {
                    const double got = lp.objective(raw.x);
                    // no point strictly better than the true optimum can exist
                    CHECK(lp.maximize ? got <= opt + 1e-6 * (1.0 + std::fabs(opt))
                                      : got >= opt - 1e-6 * (1.0 + std::fabs(opt)));
                }
            } else {
                const auto ev = sor::search::milp_evidence(d, o);
                const auto res = sor::certify::finalize_result(raw, ev);
                if (res.status == Status::Optimal) {
                    ++loose_optimal;
                    CHECK(std::fabs(res.objective - oracle.objective) <=
                          1e-6 * (1.0 + std::fabs(oracle.objective)));
                }
            }
        }
    }
    std::cout << "EXTERNAL_CUTOFF tight_runs=" << tight_runs << " foreign_prunes=" << foreign_prunes
              << " rc_fixed=" << rc_fixed << " loose_optimal=" << loose_optimal << '\n';
    CHECK(foreign_prunes > 100);
    CHECK(loose_optimal == tight_runs);   // every loose-cutoff run still proves the optimum
}

int main() {
    test_external_cutoff_is_sound();
    test_payload_mapping();
    test_stage_bound_combination();
    test_forced_restarts_match_oracle();
    return sor::test::finish("test_root_restart");
}
