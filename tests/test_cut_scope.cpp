// Cut SCOPE: may a node-generated cut leave the subtree that produced it?
//
// The defect these cover: the promotion test checked branch-tightened BOUNDS
// only. It did not check whether the model the cut was derived from carried a
// subtree-local ROW, and it trusted a separator's derivation because of the
// prefix on its name. Each case below is one way a cut can be local while the
// old test called it global.
#include "sor/search/cuts.hpp"
#include "sor/search/tree_cuts.hpp"

#include "test_helpers.hpp"

#include <string>
#include <vector>

using sor::search::CutScopeConditions;
using sor::search::cut_family_derivation_trusted;
using sor::search::cut_may_leave_subtree;

namespace {

// A scope with every gate open: the case that SHOULD promote.
CutScopeConditions all_global() {
    CutScopeConditions s;
    s.bounds_are_root = true;
    s.rows_are_global = true;
    s.derivation_trusted = true;
    s.admits_incumbent = true;
    return s;
}

// 1. A LOCAL SOURCE ROW with unchanged root variable bounds.
//
// This is the n5-3 / enigma shape and the one the bounds-only test could not
// see: node_lp is global_lp plus the node's active local cuts, so a separator
// can consume a subtree-local row while every column still sits at its root
// bound. Bounds look pristine; the cut is still local.
void test_local_source_row_with_root_bounds() {
    CutScopeConditions s = all_global();
    s.rows_are_global = false;          // node carried a local cut row
    CHECK(s.bounds_are_root);           // ...and bounds are untouched
    CHECK(!cut_may_leave_subtree(s));   // must NOT promote

    // The bounds-only predicate the code used to apply would have promoted it.
    CHECK(s.bounds_are_root && s.derivation_trusted);
}

// 2. BOUND SUBSTITUTION that eliminates the locally fixed variable.
//
// A separator substitutes x onto a bound that branching fixed. The variable
// then vanishes from the support, so a support scan sees nothing wrong, but
// the branch decision has already been folded into the right-hand side.
// bounds_are_root is the flag that must carry this, and it must dominate an
// otherwise-clean scope.
void test_bound_substitution_hides_fixed_variable() {
    CutScopeConditions s = all_global();
    s.bounds_are_root = false;          // a branch-tightened bound was used
    CHECK(s.rows_are_global);           // support/rows look global
    CHECK(!cut_may_leave_subtree(s));
}

// 3. Cuts derived from OTHER CUTS.
//
// A second-generation cut inherits its parent's scope. Deriving from a local
// cut yields a local cut however clean the new derivation looks, so scope must
// be combined by AND across the inputs, never reset by the child.
void test_cut_derived_from_local_cut_stays_local() {
    CutScopeConditions parent = all_global();
    parent.rows_are_global = false;     // parent was subtree-local
    CutScopeConditions child = all_global();      // child's own derivation looks clean

    // Inheritance = conjunction of every input's scope.
    CutScopeConditions combined;
    combined.bounds_are_root = parent.bounds_are_root && child.bounds_are_root;
    combined.rows_are_global = parent.rows_are_global && child.rows_are_global;
    combined.derivation_trusted =
        parent.derivation_trusted && child.derivation_trusted;
    combined.admits_incumbent =
        parent.admits_incumbent && child.admits_incumbent;

    CHECK(cut_may_leave_subtree(child));      // judged alone: promotable
    CHECK(!cut_may_leave_subtree(combined));  // judged with its parent: not
}

// 4. SIBLING-SUBTREE REUSE.
//
// A cut valid in the down-branch is not valid in the up-branch. Only a cut
// that depends on neither subtree's decisions may cross between them, so
// "reusable by a sibling" is exactly cut_may_leave_subtree.
void test_sibling_reuse_requires_full_global_scope() {
    CutScopeConditions down = all_global();
    down.bounds_are_root = false;       // derived under the down-branch bound
    CHECK(!cut_may_leave_subtree(down));

    CutScopeConditions neutral = all_global();    // touched no branch decision
    CHECK(cut_may_leave_subtree(neutral));
}

// 5. POOL DEDUPLICATION and REINJECTION.
//
// Two cuts with identical content may arrive with different scopes. The pooled
// entry must keep the WEAKER one, or a later reinjection promotes the strict
// copy's twin on the loose copy's evidence.
void test_pool_dedup_keeps_weaker_scope() {
    CutScopeConditions strict = all_global();
    CutScopeConditions loose = all_global();
    loose.rows_are_global = false;

    CutScopeConditions merged;
    merged.bounds_are_root = strict.bounds_are_root && loose.bounds_are_root;
    merged.rows_are_global = strict.rows_are_global && loose.rows_are_global;
    merged.derivation_trusted =
        strict.derivation_trusted && loose.derivation_trusted;
    merged.admits_incumbent =
        strict.admits_incumbent && loose.admits_incumbent;

    CHECK(cut_may_leave_subtree(strict));
    CHECK(!cut_may_leave_subtree(merged));   // dedup must not launder scope
}

// 6. The family whitelist is a quarantine. The earlier n5-3 campaign is not
//    a derivation proof: one sweep deleted the detector, and later flags were
//    references outside the generating node.
void test_family_derivation_whitelist() {
    CHECK(cut_family_derivation_trusted("COV_3"));
    CHECK(cut_family_derivation_trusted("COVPC_0"));
    CHECK(cut_family_derivation_trusted("COVGNS_7"));

    CHECK(!cut_family_derivation_trusted("MIR_12"));
    CHECK(!cut_family_derivation_trusted("ZH_1"));
    CHECK(!cut_family_derivation_trusted("FC_0"));
    // Tableau GMI is node-local by construction.
    CHECK(!cut_family_derivation_trusted("GMI_4"));

    // A trusted family is still not promotable on its own.
    CutScopeConditions s;
    s.derivation_trusted = cut_family_derivation_trusted("COV_1");
    CHECK(s.derivation_trusted);
    CHECK(!cut_may_leave_subtree(s));   // bounds/rows still default to local
}

// Every single gate must be able to veto on its own.
void test_each_gate_vetoes_independently() {
    CHECK(cut_may_leave_subtree(all_global()));
    {
        CutScopeConditions s = all_global(); s.bounds_are_root = false;
        CHECK(!cut_may_leave_subtree(s));
    }
    {
        CutScopeConditions s = all_global(); s.rows_are_global = false;
        CHECK(!cut_may_leave_subtree(s));
    }
    {
        CutScopeConditions s = all_global(); s.derivation_trusted = false;
        CHECK(!cut_may_leave_subtree(s));
    }
    {
        CutScopeConditions s = all_global(); s.admits_incumbent = false;
        CHECK(!cut_may_leave_subtree(s));
    }
}

// Default-constructed scope is LOCAL. A separator that has not been taught to
// report provenance must not get promotion by forgetting to set a field.
void test_default_scope_is_local() {
    CutScopeConditions s;
    CHECK(!s.bounds_are_root);
    CHECK(!s.rows_are_global);
    CHECK(!s.derivation_trusted);
    CHECK(!cut_may_leave_subtree(s));
}

// The pool, not a hand-ANDed struct, must refuse a cut once any observation
// of the same content failed the global gate. select_global is the insertion
// path into the global model.
void test_gcs_pool_keeps_weaker_scope_and_selects_global() {
    sor::search::CutRow row;
    row.cols = {0};
    row.vals = {1.0};
    row.row_lo = 1.0;
    row.row_hi = sor::model::kInf;
    row.name = "COV_scope";
    sor::search::CutFeatureVec feats{};
    feats[0] = 1.0;
    const std::string id = sor::search::cut_content_id(row);

    auto observe = [&](sor::search::GcsPool& pool, bool global_flag) {
        pool.prefer_heuristic = true;
        pool.observe(row, feats, 1.0, true, global_flag, id);
    };

    sor::search::GcsPool local_then_global;
    observe(local_then_global, false);
    observe(local_then_global, true);
    CHECK(local_then_global.select_global(10, -1.0, false).empty());

    sor::search::GcsPool global_then_local;
    observe(global_then_local, true);
    observe(global_then_local, false);
    CHECK(global_then_local.select_global(10, -1.0, false).empty());

    sor::search::GcsPool only_global;
    observe(only_global, true);
    const auto promoted = only_global.select_global(10, -1.0, false);
    CHECK(promoted.size() == 1);
    CHECK(promoted[0].name == "COV_scope");
}

}  // namespace

int main() {
    test_local_source_row_with_root_bounds();
    test_bound_substitution_hides_fixed_variable();
    test_cut_derived_from_local_cut_stays_local();
    test_sibling_reuse_requires_full_global_scope();
    test_pool_dedup_keeps_weaker_scope();
    test_family_derivation_whitelist();
    test_each_gate_vetoes_independently();
    test_default_scope_is_local();
    test_gcs_pool_keeps_weaker_scope_and_selects_global();
    return sor::test::finish("test_cut_scope");
}
