// Verified block symmetry and orbitopal fixing (search/orbitope.hpp).
#include "forge/certify/finalize.hpp"
#include "forge/search/bab.hpp"
#include "forge/search/orbitope.hpp"
#include "forge/search/symmetry.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using forge::core::Index;
using forge::model::LpProblem;
using namespace forge::search;

namespace {

// Bin packing with k identical bins: x[i][b] (item i in bin b), y[b] (bin b
// used, cost 1). Rows: sum_b x[i][b] = 1; sum_i w_i x[i][b] - C y_b <= 0.
// `odd_capacity` gives the last bin a different capacity (no symmetry with it).
LpProblem bin_packing(const std::vector<double>& w, int k, double cap, bool odd_capacity = false) {
    const Index r = static_cast<Index>(w.size());
    const Index n = r * k + k;
    LpProblem p;
    p.name = "binpack";
    p.c.assign(n, 0.0);
    p.col_lo.assign(n, 0.0);
    p.col_hi.assign(n, 1.0);
    p.is_integer.assign(n, true);
    const auto x = [&](Index i, Index b) { return i * k + b; };
    const auto y = [&](Index b) { return r * k + b; };
    for (Index b = 0; b < k; ++b) p.c[y(b)] = 1.0;
    std::vector<Index> ri, cj;
    std::vector<double> v;
    Index row = 0;
    for (Index i = 0; i < r; ++i, ++row) {
        for (Index b = 0; b < k; ++b) { ri.push_back(row); cj.push_back(x(i, b)); v.push_back(1.0); }
        p.row_lo.push_back(1.0);
        p.row_hi.push_back(1.0);
    }
    for (Index b = 0; b < k; ++b, ++row) {
        for (Index i = 0; i < r; ++i) { ri.push_back(row); cj.push_back(x(i, b)); v.push_back(w[i]); }
        ri.push_back(row); cj.push_back(y(b));
        v.push_back(-(odd_capacity && b == k - 1 ? cap + 1.0 : cap));
        p.row_lo.push_back(-forge::model::kInf);
        p.row_hi.push_back(0.0);
    }
    p.A = forge::sparse::from_triplets(row, n, ri, cj, v);
    p.validate();
    return p;
}

void test_detects_and_fixes_bin_packing_orbitope() {
    const std::vector<double> w{4, 3, 3, 2, 2};
    const LpProblem p = bin_packing(w, 4, 7.0);
    SymmetryOptions so;
    const auto classes = detect_permutation_orbits(p, so, nullptr);
    OrbitopeOptions oo;
    OrbitopeDiagnostics od;
    const auto orbs = detect_orbitopes(p, classes, oo, od);
    CHECK(orbs.size() == 1);
    if (orbs.size() != 1) return;
    const Orbitope& orb = orbs[0];
    CHECK(orb.partitioning);
    CHECK(orb.var.size() == w.size());          // the y row has no packing row: dropped
    CHECK(orb.var[0].size() == 4);
    CHECK(od.transpositions_verified == 3);
    // Root box: row i may only use blocks 0..i (staircase).
    std::vector<double> lo = p.col_lo, hi = p.col_hi;
    std::uint64_t fixed = 0;
    CHECK(orbitopal_fixing(orb, lo, hi, nullptr, 0, nullptr, fixed));
    for (std::size_t i = 0; i < orb.var.size(); ++i)
        for (std::size_t b = 0; b < 4; ++b)
            CHECK((hi[orb.var[i][b]] == 0.0) == (b > i));
    // Row 0 placed in block 0 and row 1 kept out of block 1: row 2 may not
    // open block 2 (only blocks 0 and 1 are reachable before it).
    hi[orb.var[1][1]] = 0.0;
    lo[orb.var[0][0]] = 1.0;
    fixed = 0;
    CHECK(orbitopal_fixing(orb, lo, hi, nullptr, 0, nullptr, fixed));
    CHECK(hi[orb.var[2][2]] == 0.0);
    CHECK(fixed >= 1);
}

void test_refutes_what_is_not_symmetric() {
    const LpProblem p = bin_packing({4, 3, 3, 2}, 3, 7.0, /*odd_capacity=*/true);
    SymmetryOptions so;
    const auto classes = detect_permutation_orbits(p, so, nullptr);
    OrbitopeOptions oo;
    OrbitopeDiagnostics od;
    const auto orbs = detect_orbitopes(p, classes, oo, od);
    // Color refinement separates the odd bin's columns, or verification
    // refutes a transposition through it: no orbitope spans all three bins.
    for (const Orbitope& o : orbs) CHECK(o.var.empty() || o.var[0].size() < 3);
}

// Optimum with and without orbitopal fixing on random bin packing models.
void test_optimum_survives() {
    std::uint32_t s = 7;
    const auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
    int compared = 0, fixed_somewhere = 0, detected = 0, with_nodes = 0;
    for (int trial = 0; trial < 40; ++trial) {
        // Large enough that the root does not close: the search branches.
        const int r = 7 + static_cast<int>(rnd() % 4);
        const int k = 4 + static_cast<int>(rnd() % 2);
        std::vector<double> w(static_cast<std::size_t>(r));
        for (auto& v : w) v = 3.0 + static_cast<double>(rnd() % 5);
        const LpProblem p = bin_packing(w, k, 10.0);
        double obj[2];
        forge::core::Status status[2];
        bool ok = true;
        for (int on = 0; on < 2; ++on) {
            BabOptions o;
            o.para_bab.threads = 1;
            o.gap_tol = o.abs_gap_tol = 1e-9;
            o.orbitope.enabled = on == 1;
            o.structural_presolve.enabled = false;
            o.cuts_enabled = false;
            BabDiagnostics d;
            auto raw = solve_milp(p, o, d);
            const auto res = forge::certify::finalize_result(std::move(raw), milp_evidence(d, o));
            status[on] = res.status;
            if (res.status != forge::core::Status::Optimal) ok = false;
            obj[on] = res.objective;
            if (on == 1 && d.orbitopal_fixings > 0) ++fixed_somewhere;
            if (on == 1) { detected += d.orbitope_diag.orbitopes > 0; with_nodes += d.nodes > 1; }
        }
        if (!ok) {
            // Infeasible models (more weight than bins) must agree too.
            CHECK(status[0] == status[1]);
            CHECK(status[0] == forge::core::Status::Infeasible);
            ++compared;
            continue;
        }
        ++compared;
        CHECK(std::fabs(obj[0] - obj[1]) <= 1e-9);
        if (std::fabs(obj[0] - obj[1]) > 1e-9)
            std::cout << "ORBITOPE mismatch trial=" << trial << " off=" << obj[0] << " on=" << obj[1] << '\n';
    }
    std::cout << "ORBITOPE compared=" << compared << " fixed_somewhere=" << fixed_somewhere
              << " detected=" << detected << " branched=" << with_nodes << '\n';
    CHECK(compared >= 30);
    CHECK(fixed_somewhere >= 15);
}

}  // namespace

int main() {
    test_detects_and_fixes_bin_packing_orbitope();
    test_refutes_what_is_not_symmetric();
    test_optimum_survives();
    return forge::test::finish("test_orbitope");
}
