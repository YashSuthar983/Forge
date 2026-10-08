// WP-A: propagation reason trail.
#include "forge/search/propagate.hpp"
#include "forge/search/prop_trail.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <vector>

using forge::core::f64;
using forge::core::Index;
using forge::model::kInf;
using forge::model::LpProblem;
using forge::sparse::from_triplets;

namespace {

void test_trail_records_row_reasons() {
    LpProblem lp;
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {10.0};
    lp.col_lo = {0.0, 3.0};
    lp.col_hi = {kInf, kInf};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    forge::search::PropTrail trail;
    const auto r = forge::search::propagate_bounds_trail(
        lp, lo, hi, &trail, 2, 1e-9, 10);
    CHECK(r.feasible);
    CHECK(trail.size() >= 1);
    bool saw_row0 = false;
    for (std::size_t k = 0; k < trail.size(); ++k) {
        const auto& e = trail.entries()[k];
        if (e.kind == forge::search::ReasonKind::Row && e.reason_id == 0 &&
            e.var == 0)
            saw_row0 = true;
    }
    CHECK(saw_row0);
    CHECK_NEAR(hi[0], 7.0, 1e-9);
}

void test_trail_replay_same_bounds() {
    LpProblem lp;
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {10.0};
    lp.col_lo = {0.0, 3.0};
    lp.col_hi = {kInf, kInf};
    lp.is_integer = {true, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    forge::search::PropTrail trail;
    const auto r = forge::search::propagate_bounds_trail(
        lp, lo, hi, &trail, 0, 1e-9, 10);
    CHECK(r.feasible);

    std::vector<f64> lo2 = lp.col_lo, hi2 = lp.col_hi;
    for (std::size_t k = 0; k < trail.size(); ++k) {
        const auto& e = trail.entries()[k];
        if (e.dir == forge::search::BoundDir::Lower)
            lo2[static_cast<std::size_t>(e.var)] = e.new_bound;
        else
            hi2[static_cast<std::size_t>(e.var)] = e.new_bound;
    }
    CHECK_NEAR(lo2[0], lo[0], 1e-9);
    CHECK_NEAR(hi2[0], hi[0], 1e-9);
}

void test_infeasible_sets_conflict_ids() {
    LpProblem lp;
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {1.0};
    lp.col_lo = {2.0, 2.0};
    lp.col_hi = {kInf, kInf};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    forge::search::PropTrail trail;
    const auto r = forge::search::propagate_bounds_trail(
        lp, lo, hi, &trail, 0, 1e-9, 10);
    CHECK(!r.feasible);
    CHECK(r.conflict_row == 0);
}

}  // namespace

// Global rows 1 and 3 of five are deleted (new indices 0,-,1,-,2). A reason
// naming a deleted row becomes Retired with the index retire() hands out, one
// per row however many entries name it; a local Cut reason is a position in
// its own LP's local block and does not move; other kinds are untouched.
void test_remap_row_reasons_after_row_deletion() {
    using forge::search::BoundDir;
    using forge::search::PropTrail;
    using forge::search::ReasonKind;
    PropTrail t;
    t.push(0, BoundDir::Upper, 1.0, 2.0, ReasonKind::Row, 0, 1);
    t.push(1, BoundDir::Lower, 1.0, 0.0, ReasonKind::Row, 1, 1);
    t.push(2, BoundDir::Upper, 0.0, 1.0, ReasonKind::Row, 4, 1);
    t.push(3, BoundDir::Upper, 0.0, 1.0, ReasonKind::Cut, 1, 1);      // local row 1
    t.push(4, BoundDir::Lower, 1.0, 0.0, ReasonKind::Branch, -1, 1);
    t.push(5, BoundDir::Lower, 1.0, 0.0, ReasonKind::Implication, 3, 1);
    t.push(6, BoundDir::Upper, 0.0, 1.0, ReasonKind::Row, 3, 1);
    t.push(7, BoundDir::Upper, 0.0, 1.0, ReasonKind::Row, 1, 1);
    std::vector<forge::search::Index> retired_from;
    std::vector<forge::search::Index> id_of(5, -1);
    t.remap_row_reasons({0, -1, 1, -1, 2}, [&](forge::search::Index r) {
        if (id_of[static_cast<std::size_t>(r)] < 0) {
            id_of[static_cast<std::size_t>(r)] = static_cast<forge::search::Index>(retired_from.size());
            retired_from.push_back(r);
        }
        return id_of[static_cast<std::size_t>(r)];
    }, 1);
    const auto& e = t.entries();
    CHECK(e[0].kind == ReasonKind::Row && e[0].reason_id == 0);
    CHECK(e[1].kind == ReasonKind::Retired && e[1].reason_id == 0);
    CHECK(e[2].kind == ReasonKind::Row && e[2].reason_id == 2);
    CHECK(e[3].kind == ReasonKind::Cut && e[3].reason_id == 1);
    CHECK(e[4].kind == ReasonKind::Branch && e[4].reason_id == -1);
    CHECK(e[5].kind == ReasonKind::Implication && e[5].reason_id == 3);   // a literal, not a row
    CHECK(e[6].kind == ReasonKind::Retired && e[6].reason_id == 1);
    CHECK(e[7].kind == ReasonKind::Retired && e[7].reason_id == 0);      // row 1 again: same id
    CHECK(retired_from.size() == 2 && retired_from[0] == 1 && retired_from[1] == 3);
}

// Trails share history through copies (PropTrail segments): a copy sees the
// source's entries, later pushes on either side stay private, and a remap
// pass rewrites a shared segment exactly once however many trails reach it.
void test_shared_trail_history() {
    using forge::search::BoundDir;
    using forge::search::PropTrail;
    using forge::search::PropTrailEntry;
    using forge::search::ReasonKind;
    PropTrail root;
    root.push(0, BoundDir::Upper, 0.0, 1.0, ReasonKind::Branch, -1, 1);
    root.push(1, BoundDir::Lower, 1.0, 0.0, ReasonKind::Row, 3, 1);
    PropTrail a = root;          // freezes root's two entries
    PropTrail b = root;
    a.push(2, BoundDir::Upper, 0.0, 1.0, ReasonKind::Row, 5, 2);
    b.push(3, BoundDir::Upper, 0.0, 1.0, ReasonKind::Row, 3, 2);
    root.push(4, BoundDir::Lower, 1.0, 0.0, ReasonKind::Branch, -1, 1);
    CHECK(root.size() == 3 && a.size() == 3 && b.size() == 3);
    CHECK(a.entries()[2].var == 2 && b.entries()[2].var == 3 && root.entries()[2].var == 4);
    CHECK(a.entries()[1].var == 1 && b.entries()[0].var == 0);
    std::size_t n_since = 0;
    for (const PropTrailEntry& e : a.since(2)) { CHECK(e.var == 2); ++n_since; }
    CHECK(n_since == 1);
    std::size_t visited = 0;
    b.for_each([&](const PropTrailEntry&) { ++visited; });
    CHECK(visited == 3);
    // Row 3 deleted, row 5 -> 4: the shared entry (var 1) is rewritten once.
    std::vector<forge::search::Index> new_row{0, 1, 2, -1, 3, 4};
    int retired = 0;
    const auto retire = [&](forge::search::Index) { return static_cast<forge::search::Index>(retired++); };
    a.remap_row_reasons(new_row, retire, 7);
    b.remap_row_reasons(new_row, retire, 7);
    root.remap_row_reasons(new_row, retire, 7);
    CHECK(retired == 2);   // the shared row-3 entry once, b's own row-3 entry once
    CHECK(a.entries()[1].kind == ReasonKind::Retired && b.entries()[1].kind == ReasonKind::Retired);
    CHECK(a.entries()[2].reason_id == 4);
    CHECK(b.entries()[2].kind == ReasonKind::Retired && b.entries()[2].reason_id == 1);
    // Truncate into the shared prefix: a private copy, the others untouched.
    a.truncate(1);
    CHECK(a.size() == 1 && a.entries()[0].var == 0);
    CHECK(b.size() == 3 && root.size() == 3);
}

int main() {
    test_shared_trail_history();
    test_remap_row_reasons_after_row_deletion();
    test_trail_records_row_reasons();
    test_trail_replay_same_bounds();
    test_infeasible_sets_conflict_ids();
    return forge::test::finish("test_conflict_trail");
}
