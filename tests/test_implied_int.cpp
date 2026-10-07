// WP-F: TU / network / consecutive-ones implied integrality.
#include "forge/search/bab.hpp"
#include "forge/certify/finalize.hpp"
#include "forge/search/implied_int.hpp"
#include "forge/sparse/csr.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <limits>
#include <vector>

using forge::model::LpProblem;
using forge::model::kInf;
using forge::sparse::from_triplets;
namespace model = forge::model;
namespace core = forge::core;
namespace sparse = forge::sparse;
namespace search = forge::search;

namespace {

// Equality ±1: y continuous, x integer, x - y = 0 ⇒ y implied integer.
LpProblem eq_pm1_model() {
    LpProblem lp;
    lp.name = "eq_pm1";
    lp.c = {1.0, 1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {5.0, 5.0};
    lp.is_integer = {true, false};
    lp.row_lo = {0.0};
    lp.row_hi = {0.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
    return lp;
}

// Network incidence: two nodes, one arc flow f continuous, supply/demand.
//   +f = 1 (node A), -f = -1 (node B); f in [0,5] continuous ⇒ network TU.
LpProblem network_flow_model() {
    LpProblem lp;
    lp.name = "network";
    lp.c = {1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {5.0};
    lp.is_integer = {false};
    lp.row_lo = {1.0, -1.0};
    lp.row_hi = {1.0, -1.0};
    // row0: +f = 1, row1: -f = -1
    lp.A = from_triplets(2, 1, {0, 1}, {0, 0}, {1.0, -1.0});
    return lp;
}

// Pure C1 block: three eqs, two continuous with identity-like C1.
//   r0: x = 2, r1: x = 2  - single col consecutive ones of length 2.
LpProblem c1_model() {
    LpProblem lp;
    lp.name = "c1";
    lp.c = {1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {5.0};
    lp.is_integer = {false};
    lp.row_lo = {2.0, 2.0};
    lp.row_hi = {2.0, 2.0};
    lp.A = from_triplets(2, 1, {0, 1}, {0, 0}, {1.0, 1.0});
    return lp;
}

// Example 1.4 / Corollary 3.3 (paper §3.1): z implied by integer x,y via ≤ rows.
LpProblem dual_rational_model() {
    LpProblem lp;
    lp.name = "dual_rational";
    lp.c = {10.0, 10.0, 1.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {10.0, 10.0, 10.0};
    lp.is_integer = {true, true, false};
    lp.row_lo = {-kInf, -kInf, 0.0};
    lp.row_hi = {4.0, 3.0, kInf};
    // 3x + 2y + z <= 4; x + 3y - z <= 3; z >= 0
    lp.A = from_triplets(
        3, 3, {0, 0, 0, 1, 1, 1, 2}, {0, 1, 2, 0, 1, 2, 2},
        {3.0, 2.0, 1.0, 1.0, 3.0, -1.0, 1.0});
    return lp;
}

// Two parallel network arcs (continuous) in one ≤ row - Algorithm 1 column pass.
LpProblem tu_network_pair_model() {
    LpProblem lp;
    lp.name = "tu_network_pair";
    lp.c = {1.0, 1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {5.0, 5.0};
    lp.is_integer = {false, false};
    lp.row_lo = {-kInf};
    lp.row_hi = {2.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    return lp;
}

void test_equality_pm1() {
    auto lp = eq_pm1_model();
    forge::search::ImpliedIntOptions o;
    o.network = false;
    o.consecutive_ones = false;
    const auto d = forge::search::infer_implied_integers_ex(lp, o);
    CHECK(d.equality_pm1 >= 1);
    CHECK(lp.is_integer[1]);
}

void test_small_nonzero_coefficient_blocks_implied_integrality() {
    // x + 5e-10*y = 0, y integer in [0,1e9]. The feasible point
    // (x,y)=(-0.5,1e9) proves that x is NOT implied integer. Rounding the
    // small coefficient to zero would falsely mark it as integer.
    LpProblem lp;
    lp.name = "implied_int_small_coef";
    lp.c = {1.0, 0.0};
    lp.col_lo = {-1.0, 0.0};
    lp.col_hi = {0.0, 1e9};
    lp.is_integer = {false, true};
    lp.row_lo = {0.0};
    lp.row_hi = {0.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 5e-10});
    search::ImpliedIntOptions o;
    o.network = false;
    o.consecutive_ones = false;
    o.dual_rational = false;
    const auto d = search::infer_implied_integers_ex(lp, o);
    CHECK(d.total == 0);
    CHECK(!lp.is_integer[0]);
}

void test_network_marks_flow() {
    auto lp = network_flow_model();
    forge::search::ImpliedIntOptions o;
    o.equality_pm1 = true;
    o.network = true;
    o.consecutive_ones = false;
    const auto d = forge::search::infer_implied_integers_ex(lp, o);
    CHECK(d.total >= 1);
    CHECK(lp.is_integer[0]);
}

void test_c1_marks() {
    auto lp = c1_model();
    forge::search::ImpliedIntOptions o;
    o.equality_pm1 = false;
    o.network = false;
    o.consecutive_ones = true;
    const auto d = forge::search::infer_implied_integers_ex(lp, o);
    CHECK(d.consecutive_ones >= 1);
    CHECK(lp.is_integer[0]);
}

void test_dual_rational_marks_z() {
    auto lp = dual_rational_model();
    forge::search::ImpliedIntOptions o;
    o.equality_pm1 = false;
    o.network = false;
    o.consecutive_ones = false;
    o.tu_network_block = false;
    const auto d = forge::search::infer_implied_integers_ex(lp, o);
    CHECK(d.dual_rational >= 1);
    CHECK(lp.is_integer[2]);
}

void test_tu_network_block_pair() {
    auto lp = tu_network_pair_model();
    forge::search::ImpliedIntOptions o;
    o.equality_pm1 = false;
    o.network = false;
    o.consecutive_ones = false;
    o.dual_rational = false;
    // Explicit opt-in: the rule is correct on this fixture but UNSOUND on real
    // models (see the header), so it no longer defaults on. This test pins the
    // rule's own behaviour, not the shipped default.
    o.tu_network_block = true;
    const auto d = forge::search::infer_implied_integers_ex(lp, o);
    CHECK(d.tu_network_block >= 2);
    CHECK(lp.is_integer[0]);
    CHECK(lp.is_integer[1]);
}


// --- Oracle: a mark may restrict the feasible set, never the optimum --------
//
// Every rule promises that some optimum of the model stays feasible once the
// marked columns are integer and their bounds snapped. Checked by exhaustive
// enumeration (milp_oracle.hpp): the optimum of the model with the marks must
// equal the optimum without them. The generator leans towards the patterns the
// rules look for (+-1 continuous terms in equality rows, integer companions)
// and mixes in what must block them: fractional coefficients on integer
// columns, fractional continuous bounds, fractional sides, and a working box
// tighter than the model's own bounds (mip_presolve passes one).

struct IiCase {
    LpProblem lp;                       // rows + the box the problem lives in
    std::vector<double> lo, hi;         // working box (== lp bounds unless split)
    bool split_box = false;             // lp.col_lo/hi wider than the working box
};

IiCase make_ii_case(std::uint32_t seed) {
    using forge::test::oracle::next;
    std::uint32_t st = seed * 2654435761u + 977u;
    IiCase c;
    LpProblem& lp = c.lp;
    lp.name = "ii_" + std::to_string(seed);
    lp.maximize = (next(st) & 1u) != 0;
    const core::Index n = 3 + static_cast<core::Index>(next(st) % 3);
    const core::Index m = 1 + static_cast<core::Index>(next(st) % 3);
    lp.c.assign(n, 0.0);
    lp.col_lo.assign(n, 0.0);
    lp.col_hi.assign(n, 0.0);
    lp.is_integer.assign(n, false);
    for (core::Index j = 0; j < n; ++j) {
        lp.c[j] = static_cast<double>(static_cast<int>(next(st) % 7) - 3);
        if (next(st) % 2 == 0) {
            lp.is_integer[j] = true;
            lp.col_lo[j] = (next(st) % 3 == 0) ? -1.0 : 0.0;
            lp.col_hi[j] = lp.col_lo[j] + 1.0 + static_cast<double>(next(st) % 2);
        } else {
            static const double los[] = {0.0, 0.0, 0.5, -1.0};
            static const double spans[] = {1.0, 2.0, 1.5, 3.0, 2.5};
            lp.col_lo[j] = los[next(st) % 4];
            lp.col_hi[j] = lp.col_lo[j] + spans[next(st) % 5];
        }
    }
    static const double coefs[] = {1.0, 1.0, 1.0, -1.0, -1.0, 2.0, -2.0, 0.5, 3.0, 1.5};
    std::vector<core::Index> rows, cols;
    std::vector<double> vals;
    lp.row_lo.assign(m, 0.0);
    lp.row_hi.assign(m, 0.0);
    for (core::Index i = 0; i < m; ++i) {
        const core::Index len = 2 + static_cast<core::Index>(next(st) % 3);
        std::vector<char> used(n, 0);
        for (core::Index t = 0; t < len; ++t) {
            const core::Index j = static_cast<core::Index>(next(st) % static_cast<std::uint32_t>(n));
            if (used[j]) continue;
            used[j] = 1;
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(coefs[next(st) % 10]);
        }
        double rhs = static_cast<double>(static_cast<int>(next(st) % 7) - 2);
        if (next(st) % 7 == 0) rhs += 0.5;
        const std::uint32_t sense = next(st) % 4;
        if (sense <= 1) { lp.row_lo[i] = rhs; lp.row_hi[i] = rhs; }
        else if (sense == 2) { lp.row_lo[i] = -kInf; lp.row_hi[i] = rhs; }
        else { lp.row_lo[i] = rhs; lp.row_hi[i] = kInf; }
    }
    lp.A = from_triplets(m, n, rows, cols, vals);
    c.lo = lp.col_lo;
    c.hi = lp.col_hi;
    // Sometimes the problem being solved lives in a tighter, possibly
    // fractional, working box than the model's own integral bounds.
    if (next(st) % 3 == 0) {
        for (core::Index j = 0; j < n; ++j) {
            if (lp.is_integer[j] || next(st) % 2 != 0) continue;
            if (c.hi[j] - c.lo[j] > 1.0) {
                if (next(st) % 2 == 0) c.lo[j] += 0.5;
                else c.hi[j] -= 0.5;
                c.split_box = true;
            }
        }
    }
    return c;
}

double ii_scale(double v) { return 1.0 + std::fabs(v); }

// One rule set on one case: 1 if anything was marked (and checked).
int check_ii_case(const IiCase& c, const search::ImpliedIntOptions& o,
                  const char* label) {
    // The problem actually being solved: rows + working box.
    LpProblem truth = c.lp;
    truth.col_lo = c.lo;
    truth.col_hi = c.hi;
    LpProblem marked = c.lp;
    std::vector<double> lo = c.lo, hi = c.hi;
    const auto d = c.split_box
        ? search::infer_implied_integers_ex(marked, o, &lo, &hi)
        : search::infer_implied_integers_ex(marked, o);
    if (d.total == 0) return 0;
    LpProblem restricted = truth;
    restricted.is_integer = marked.is_integer;
    restricted.col_lo = c.split_box ? lo : marked.col_lo;
    restricted.col_hi = c.split_box ? hi : marked.col_hi;
    const auto want = forge::test::oracle::solve_oracle(truth);
    bool crossed = false;
    for (core::Index j = 0; j < restricted.n_cols(); ++j)
        if (restricted.col_lo[j] > restricted.col_hi[j]) crossed = true;
    if (crossed) {
        // An emptied box claims infeasibility: only right if it is.
        if (want.feasible)
            std::fprintf(stderr, "%s %s: snap emptied a box of a feasible model\n",
                         label, c.lp.name.c_str());
        CHECK(!want.feasible);
        return 1;
    }
    const auto got = forge::test::oracle::solve_oracle(restricted);
    const bool ok = want.feasible == got.feasible &&
        (!want.feasible ||
         std::fabs(want.objective - got.objective) <= 1e-7 * ii_scale(want.objective));
    if (!ok)
        std::fprintf(stderr, "%s %s: optimum %s%.9g -> %s%.9g after %llu marks\n",
                     label, c.lp.name.c_str(), want.feasible ? "" : "infeasible ",
                     want.objective, got.feasible ? "" : "infeasible ",
                     got.objective, static_cast<unsigned long long>(d.total));
    CHECK(ok);
    return 1;
}

void test_implied_int_oracle() {
    struct Config { const char* name; bool pm1, net, c1, dual; };
    const Config configs[] = {
        {"all", true, true, true, true},
        {"pm1", true, false, false, false},
        {"network", false, true, false, false},
        {"c1", false, false, true, false},
        {"dual", false, false, false, true},
    };
    int fired[5] = {0, 0, 0, 0, 0};
    for (std::uint32_t seed = 1; seed <= 1500; ++seed) {
        const IiCase c = make_ii_case(seed);
        for (int k = 0; k < 5; ++k) {
            search::ImpliedIntOptions o;
            o.equality_pm1 = configs[k].pm1;
            o.network = configs[k].net;
            o.consecutive_ones = configs[k].c1;
            o.dual_rational = configs[k].dual;
            o.tu_network_block = false;
            fired[k] += check_ii_case(c, o, configs[k].name);
        }
    }
    std::printf("implied-int oracle: models with marks all=%d pm1=%d network=%d c1=%d dual=%d\n",
                fired[0], fired[1], fired[2], fired[3], fired[4]);
    // The generator must actually exercise every rule.
    for (int k = 0; k < 5; ++k) CHECK(fired[k] >= 20);
}

// The three shapes that used to be marked (each would lose the optimum).
void test_implied_int_pinned_counterexamples() {
    search::ImpliedIntOptions o;
    o.tu_network_block = false;
    {
        // x + 0.5 y + z = 3, x cont [0,10], y bin, z int [0,5]; min x + z.
        // Optimum 2.5 needs y = 1 and x fractional.
        LpProblem lp;
        lp.c = {1.0, 0.0, 1.0};
        lp.col_lo = {0.0, 0.0, 0.0};
        lp.col_hi = {10.0, 1.0, 5.0};
        lp.is_integer = {false, true, true};
        lp.row_lo = {3.0}; lp.row_hi = {3.0};
        lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 0.5, 1.0});
        search::infer_implied_integers_ex(lp, o);
        CHECK(!lp.is_integer[0]);
    }
    {
        // x + w + z = 3, x cont [0,10], w cont [0,0.5], z int [0,5]; max w.
        LpProblem lp;
        lp.maximize = true;
        lp.c = {0.0, 1.0, 0.0};
        lp.col_lo = {0.0, 0.0, 0.0};
        lp.col_hi = {10.0, 0.5, 5.0};
        lp.is_integer = {false, false, true};
        lp.row_lo = {3.0}; lp.row_hi = {3.0};
        lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
        search::infer_implied_integers_ex(lp, o);
        CHECK(!lp.is_integer[0]);
        CHECK(!lp.is_integer[1]);
    }
    {
        // x - y >= 0, x cont [0.5,10], y int [0,5]; min x + 10 y. Optimum 0.5.
        LpProblem lp;
        lp.c = {1.0, 10.0};
        lp.col_lo = {0.5, 0.0};
        lp.col_hi = {10.0, 5.0};
        lp.is_integer = {false, true};
        lp.row_lo = {0.0}; lp.row_hi = {kInf};
        lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
        search::infer_implied_integers_ex(lp, o);
        CHECK(!lp.is_integer[0]);
        CHECK(lp.col_lo[0] == 0.5);
    }
    {
        // Same row with integral own bounds: marking is right (x = y or the
        // bound at the optimum).
        LpProblem lp;
        lp.c = {1.0, 10.0};
        lp.col_lo = {1.0, 0.0};
        lp.col_hi = {10.0, 5.0};
        lp.is_integer = {false, true};
        lp.row_lo = {0.0}; lp.row_hi = {kInf};
        lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
        const auto d = search::infer_implied_integers_ex(lp, o);
        CHECK(d.dual_rational == 1);
        CHECK(lp.is_integer[0]);
    }
    {
        // Working box tighter than the model's: [0.5, 10] in force while the
        // model says [0, 10]. The working box decides.
        LpProblem lp;
        lp.c = {1.0, 10.0};
        lp.col_lo = {0.0, 0.0};
        lp.col_hi = {10.0, 5.0};
        lp.is_integer = {false, true};
        lp.row_lo = {0.0}; lp.row_hi = {kInf};
        lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
        std::vector<double> lo = {0.5, 0.0}, hi = {10.0, 5.0};
        search::infer_implied_integers_ex(lp, o, &lo, &hi);
        CHECK(!lp.is_integer[0]);
        CHECK(lo[0] == 0.5);
    }
}
}  // namespace

void test_network_component_is_atomic() {
    for (bool reverse_second : {false, true}) {
        for (bool inequality : {false, true}) {
            LpProblem lp;
            lp.c = inequality ? std::vector<double>{0.0, 1.0} : std::vector<double>{1.0, 0.0, 0.0};
            lp.col_lo = inequality ? std::vector<double>{0.0, 0.0} : std::vector<double>{0.0, 0.0, 0.3};
            lp.col_hi = inequality ? std::vector<double>{1.0, 1.0} : std::vector<double>{1.0, 1.0, 0.7};
            lp.is_integer.assign(lp.c.size(), false);
            const double sign = reverse_second ? -1.0 : 1.0;
            if (inequality) {
                lp.A = from_triplets(2, 2, {0, 0, 1}, {0, 1, 1}, {1.0, 1.0, sign});
                lp.row_lo = {1.0, reverse_second ? -kInf : 0.3};
                lp.row_hi = {1.0, reverse_second ? -0.3 : kInf};
            } else {
                lp.A = from_triplets(2, 3, {0, 0, 1, 1}, {0, 1, 1, 2}, {1.0, 1.0, sign, sign});
                lp.row_lo = {1.0, sign}; lp.row_hi = lp.row_lo;
            }
            search::ImpliedIntOptions o;
            o.equality_pm1 = false; o.network = true; o.consecutive_ones = false;
            o.dual_rational = false; o.tu_network_block = false;
            auto diag = search::infer_implied_integers_ex(lp, o);
            CHECK(diag.total == 0);
            for (bool integer : lp.is_integer) CHECK(!integer);
            search::BabOptions opts; opts.para_bab.threads = 1;
            search::BabDiagnostics bd;
            auto raw = search::solve_milp(lp, opts, bd);
            auto result = forge::certify::finalize_result(std::move(raw), search::milp_evidence(bd, opts));
            CHECK(result.status == core::Status::Optimal);
            CHECK(std::fabs(result.objective - 0.3) <= 1e-6);
        }
    }
    LpProblem transport;
    transport.c = {1.0, 2.0, 2.0, 1.0};
    transport.col_lo.assign(4, 0.0); transport.col_hi.assign(4, 1.0);
    transport.is_integer.assign(4, false);
    transport.row_lo = {1.0, 1.0, -1.0, -1.0}; transport.row_hi = transport.row_lo;
    transport.A = from_triplets(4, 4, {0,0,1,1,2,2,3,3}, {0,1,2,3,0,2,1,3}, {1,1,1,1,-1,-1,-1,-1});
    search::ImpliedIntOptions o;
    o.equality_pm1 = false; o.network = true; o.consecutive_ones = false;
    o.dual_rational = false; o.tu_network_block = false;
    auto d = search::infer_implied_integers_ex(transport, o);
    CHECK(d.network == 4);
}


int main() {
    test_network_component_is_atomic();
    test_equality_pm1();
    test_small_nonzero_coefficient_blocks_implied_integrality();
    test_network_marks_flow();
    test_c1_marks();
    test_dual_rational_marks_z();
    test_tu_network_block_pair();
    test_implied_int_pinned_counterexamples();
    test_implied_int_oracle();
    
    // --- Regression: implied integrality must never fire on a pure LP ------
    //
    // 2026-09-19. Marking a continuous column integer RESTRICTS the feasible
    // set, and snap_integer_bounds() then rounds that column's bounds. On a
    // model the caller supplied with no integer columns there is no branching
    // to help, so the upside is zero and the downside is a wrong answer:
    //   netlib 80bau3b -> false Infeasible (true optimum 987224.19)
    //   netlib d2q06c  -> 122784.63 against a true LP optimum of 122784.21
    // Three separate rules were implicated (tu_network_block, dual_rational,
    // network), so this is guarded at the call site in bab.cpp rather than
    // rule by rule. This test pins the property the guard protects.
    {
        model::LpProblem lp;
        // min -x  s.t.  2x + 2y = 3,  0<=x,y<=3.  Optimum x=1.5 is FRACTIONAL,
        // so any integrality mark on x makes the true optimum unreachable.
        std::vector<core::Index> rows{0, 0};
        std::vector<core::Index> cols{0, 1};
        std::vector<core::f64> vals{2.0, 2.0};
        lp.A = sparse::from_triplets(1, 2, rows, cols, vals);
        lp.c = {-1.0, 0.0};
        lp.col_lo = {0.0, 0.0};
        lp.col_hi = {3.0, 3.0};
        lp.row_lo = {3.0};
        lp.row_hi = {3.0};
        lp.is_integer.assign(2, false);
        CHECK(lp.n_integer() == 0);

        // tu_network_block is unsound as implemented and must stay OFF.
        search::ImpliedIntOptions defaults;
        CHECK(!defaults.tu_network_block);

        // And the MILP entry point must answer a pure LP as an LP.
        search::BabOptions bopts;
        bopts.time_limit_s = 10.0;
        search::BabDiagnostics bdiag;
        const auto out = search::solve_milp(lp, bopts, bdiag);
        CHECK(out.proposed_status != core::Status::Infeasible);
        CHECK_NEAR(out.objective, -1.5, 1e-7);
    }

return forge::test::finish("test_implied_int");
}
