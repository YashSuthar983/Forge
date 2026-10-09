// Row activity cache and cached propagation (propagate.hpp: RowActivityCache,
// propagate_bounds_cached).
//   * consistency: after random sequences of bound changes (narrowing and
//     widening, finite and infinite), every cached activity is within its
//     error bound of the exact one (DyadicSum), and the infinite counts and
//     culprit sums match;
//   * soundness: on small integer models every feasible point of a random
//     box survives cached propagation, and an empty result has no point;
//   * strength: the cached fixpoint is inside plain propagation's box (its
//     widening is charged on an exact-to-ulps activity, never more loosely),
//     and a node seeded with its changes reaches the same box as a sweep.
#include "forge/model/dyadic.hpp"
#include "forge/search/propagate.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using forge::core::f64;
using forge::core::Index;
using forge::model::kInf;
using forge::model::LpProblem;
using forge::search::PropagationScratch;
using forge::search::RowActivityCache;

namespace {

std::size_t z(Index i) { return static_cast<std::size_t>(i); }

LpProblem random_model(std::mt19937& rng, Index n, Index m, bool wild) {
    LpProblem lp;
    lp.name = "act";
    std::vector<Index> r, c;
    std::vector<f64> v;
    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < n; ++j) {
            if (rng() % 2 != 0) continue;
            f64 a = static_cast<f64>(static_cast<int>(rng() % 9) - 4);
            if (a == 0.0) continue;
            if (wild && rng() % 4 == 0) a *= std::ldexp(1.0, static_cast<int>(rng() % 40) - 20);
            r.push_back(i); c.push_back(j); v.push_back(a);
        }
        const f64 side = static_cast<f64>(static_cast<int>(rng() % 9) - 2);
        const int kind = static_cast<int>(rng() % 3);
        lp.row_lo.push_back(kind == 0 ? -kInf : side - static_cast<f64>(rng() % 3));
        lp.row_hi.push_back(kind == 1 ? kInf : side + static_cast<f64>(rng() % 3));
    }
    lp.A = forge::sparse::from_triplets(m, n, r, c, v);
    lp.c.assign(z(n), 0.0);
    lp.col_lo.assign(z(n), 0.0);
    lp.col_hi.assign(z(n), 3.0);
    lp.is_integer.assign(z(n), true);
    return lp;
}

// Exact activities of row i over [lo, hi].
void exact_row(const LpProblem& lp, Index i, const std::vector<f64>& lo,
               const std::vector<f64>& hi, forge::model::DyadicSum& mn, int& mn_inf,
               long long& mn_cols, forge::model::DyadicSum& mx, int& mx_inf, long long& mx_cols) {
    mn_inf = mx_inf = 0;
    mn_cols = mx_cols = 0;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (auto k = rp[z(i)]; k < rp[z(i) + 1]; ++k) {
        const f64 a = lp.A.vals[z(k)];
        const Index j = ci[z(k)];
        const f64 bmin = a > 0 ? lo[z(j)] : hi[z(j)];
        const f64 bmax = a > 0 ? hi[z(j)] : lo[z(j)];
        if (std::isfinite(bmin)) mn.add_product(a, bmin); else { ++mn_inf; mn_cols += j; }
        if (std::isfinite(bmax)) mx.add_product(a, bmax); else { ++mx_inf; mx_cols += j; }
    }
}

bool within(const RowActivityCache::Side& s, const forge::model::DyadicSum& exact) {
    // |s.sum - exact| <= s.err, decided exactly: exact - (sum - err) >= 0 and
    // (sum + err) - exact >= 0, with sum +- err rounded outward to binary64.
    const auto lo_d = static_cast<f64>(s.sum - s.err);
    const auto hi_d = static_cast<f64>(s.sum + s.err);
    const f64 lo_out = std::nextafter(std::nextafter(lo_d, -kInf), -kInf);
    const f64 hi_out = std::nextafter(std::nextafter(hi_d, kInf), kInf);
    return exact.compare(lo_out) >= 0 && exact.compare(hi_out) <= 0;
}

void test_checkpoint_sync_restores_siblings() {
    std::mt19937 rng(87);
    auto lp = random_model(rng, 540, 9, true);
    auto lo = lp.col_lo, hi = lp.col_hi;
    const auto root = forge::search::DomainCheckpoint::capture(lo, hi, {});
    auto left = root, right = root;
    left.set_lower(7, 1); right.set_upper(270, 2);
    RowActivityCache cache; cache.build(lp, lo, hi);
    for (const auto* box : std::initializer_list<const forge::search::DomainCheckpoint*>{&root, &left, &right, &root, &right}) {
        cache.sync_checkpoint(*box);
        box->materialize(lo, hi);
        for (Index i = 0; i < lp.n_rows(); ++i) {
            forge::model::DyadicSum mn, mx; int ni, xi; long long nc, xc;
            exact_row(lp,i,lo,hi,mn,ni,nc,mx,xi,xc);
            CHECK(within(cache.min_side(i),mn)); CHECK(within(cache.max_side(i),mx));
        }
        // A consumer moves a bound without modifying the immutable anchor.
        // The next sibling must restore it, including an unchanged root page.
        cache.set_bounds(515, 2, 3);
    }
}

void test_structural_remap() {
    std::mt19937 rng(61);
    auto lp = random_model(rng, 5, 4, true);
    auto lo = lp.col_lo, hi = lp.col_hi;
    RowActivityCache cache;
    cache.build(lp, lo, hi);
    lo[1] = 1; hi[3] = 2;
    cache.sync(lo, hi);
    const auto saved = cache.min_side(3);
    const auto rebuilds = cache.rebuilds();
    auto edited = lp;
    std::vector<Index> rr, cc;
    std::vector<f64> vv;
    const std::vector<Index> map{3, -1, 0};
    const auto& rp = lp.A.pattern.row_ptr();
    for (Index i = 0; i < 3; ++i) {
        if (map[z(i)] < 0) { rr.push_back(i); cc.push_back(2); vv.push_back(-7); continue; }
        const Index old = map[z(i)];
        for (auto k = rp[z(old)]; k < rp[z(old)+1]; ++k) {
            rr.push_back(i); cc.push_back(lp.A.pattern.col_idx()[z(k)]); vv.push_back(lp.A.vals[z(k)]);
        }
    }
    edited.A = forge::sparse::from_triplets(3,5,rr,cc,vv);
    edited.row_lo = {-kInf,-kInf,-kInf}; edited.row_hi = {100,100,100};
    CHECK(cache.remap(edited,map));
    CHECK(cache.min_side(0).sum == saved.sum);
    CHECK(cache.min_side(0).err == saved.err);
    CHECK(cache.rebuilds() == rebuilds+1);
    hi[2] = 1;
    cache.sync(lo,hi);
    for (Index i = 0; i < 3; ++i) {
        forge::model::DyadicSum mn,mx; int ni,xi; long long nc,xc;
        exact_row(edited,i,lo,hi,mn,ni,nc,mx,xi,xc);
        CHECK(within(cache.min_side(i),mn)); CHECK(within(cache.max_side(i),mx));
    }
    auto foreign = edited; foreign.A.vals[0] = std::nextafter(foreign.A.vals[0], kInf);
    CHECK(!cache.remap(foreign,{0,1,2}));
}

void test_consistency() {
    std::mt19937 rng(5);
    int checked = 0;
    std::uint64_t rebuilds = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const Index n = 4 + static_cast<Index>(rng() % 12);
        const Index m = 2 + static_cast<Index>(rng() % 8);
        const auto lp = random_model(rng, n, m, true);
        std::vector<f64> lo(z(n)), hi(z(n));
        for (Index j = 0; j < n; ++j) { lo[z(j)] = -2.0; hi[z(j)] = 5.0; }
        RowActivityCache cache;
        cache.build(lp, lo, hi);
        for (int step = 0; step < 400; ++step) {
            const Index j = static_cast<Index>(rng() % static_cast<unsigned>(n));
            const int what = static_cast<int>(rng() % 6);
            f64 l = lo[z(j)], h = hi[z(j)];
            if (what == 0) l = -kInf;
            else if (what == 1) h = kInf;
            else {
                const f64 a = std::ldexp(static_cast<f64>(static_cast<int>(rng() % 2001) - 1000),
                                         static_cast<int>(rng() % 30) - 15);
                const f64 b = a + std::ldexp(static_cast<f64>(rng() % 1000), static_cast<int>(rng() % 20) - 10);
                l = a; h = b;
            }
            lo[z(j)] = l; hi[z(j)] = h;
            if (rng() % 3 == 0) cache.sync(lo, hi);
            else cache.set_bounds(j, l, h);
        }
        cache.sync(lo, hi);
        for (Index i = 0; i < m; ++i) {
            forge::model::DyadicSum emn, emx;
            int mni, mxi;
            long long mnc, mxc;
            exact_row(lp, i, lo, hi, emn, mni, mnc, emx, mxi, mxc);
            const auto& smn = cache.min_side(i);
            const auto& smx = cache.max_side(i);
            CHECK(smn.inf == mni);
            CHECK(smx.inf == mxi);
            CHECK(smn.inf_cols == mnc);
            CHECK(smx.inf_cols == mxc);
            CHECK(within(smn, emn));
            CHECK(within(smx, emx));
            ++checked;
        }
        rebuilds += cache.rebuilds();
    }
    std::cout << "ACTIVITY_CONSISTENCY rows=" << checked << " resums=" << rebuilds << '\n';
    CHECK(checked > 800);
}

std::vector<std::vector<f64>> points(const LpProblem& lp, const std::vector<f64>& lo,
                                     const std::vector<f64>& hi) {
    const Index n = lp.n_cols();
    std::vector<std::vector<f64>> out;
    std::vector<f64> x(z(n));
    for (Index j = 0; j < n; ++j) x[z(j)] = lo[z(j)];
    for (;;) {
        bool ok = true;
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        for (Index i = 0; i < lp.n_rows() && ok; ++i) {
            f64 act = 0.0;
            for (auto k = rp[z(i)]; k < rp[z(i) + 1]; ++k) act += lp.A.vals[z(k)] * x[z(ci[z(k)])];
            if (act < lp.row_lo[z(i)] - 1e-9 || act > lp.row_hi[z(i)] + 1e-9) ok = false;
        }
        if (ok) out.push_back(x);
        Index j = 0;
        while (j < n) {
            if (x[z(j)] + 1.0 <= hi[z(j)]) { x[z(j)] += 1.0; break; }
            x[z(j)] = lo[z(j)];
            ++j;
        }
        if (j == n) break;
    }
    return out;
}

void test_sound_and_strong() {
    std::mt19937 rng(9);
    int checked = 0, emptied = 0, tightened = 0, seeded_tried = 0, seeded_same = 0;
    for (int trial = 0; trial < 1500; ++trial) {
        const Index n = 3 + static_cast<Index>(rng() % 4);
        const Index m = 2 + static_cast<Index>(rng() % 4);
        auto lp = random_model(rng, n, m, false);
        std::vector<f64> lo(z(n)), hi(z(n));
        for (Index j = 0; j < n; ++j) {
            lo[z(j)] = static_cast<f64>(rng() % 2);
            hi[z(j)] = lo[z(j)] + static_cast<f64>(rng() % 3);
        }
        // Most rows hold at a random point of the box, tightly enough to
        // propagate; one in five keeps its random sides.
        {
            std::vector<f64> p(z(n));
            for (Index j = 0; j < n; ++j)
                p[z(j)] = lo[z(j)] + static_cast<f64>(rng() % static_cast<unsigned>(hi[z(j)] - lo[z(j)] + 1));
            const auto& rp = lp.A.pattern.row_ptr();
            const auto& ci = lp.A.pattern.col_idx();
            for (Index i = 0; i < m; ++i) {
                if (rng() % 5 == 0) continue;
                f64 act = 0.0;
                for (auto k = rp[z(i)]; k < rp[z(i) + 1]; ++k) act += lp.A.vals[z(k)] * p[z(ci[z(k)])];
                const int kind = static_cast<int>(rng() % 3);
                lp.row_lo[z(i)] = kind == 0 ? -kInf : act - static_cast<f64>(rng() % 2);
                lp.row_hi[z(i)] = kind == 1 ? kInf : act + static_cast<f64>(rng() % 2);
            }
        }
        const auto pts = points(lp, lo, hi);
        // The cache describes a wider box than the one propagated (a node
        // after a jump): sync must bring it there.
        std::vector<f64> wlo(z(n), 0.0), whi(z(n), 3.0);
        RowActivityCache cache;
        cache.build(lp, wlo, whi);
        PropagationScratch scratch;
        auto clo = lo, chi = hi;
        const auto cr = forge::search::propagate_bounds_cached(lp, cache, clo, chi, nullptr, nullptr, scratch,
                                                             nullptr, 0, 1e-9, 100000);
        auto plo = lo, phi = hi;
        const auto pr = forge::search::propagate_bounds(lp, plo, phi, 1e-9, 1000);
        if (!cr.feasible) {
            CHECK(pts.empty());
            ++emptied;
            ++checked;
            continue;
        }
        for (const auto& x : pts)
            for (Index j = 0; j < n; ++j) {
                CHECK(x[z(j)] >= clo[z(j)]);
                CHECK(x[z(j)] <= chi[z(j)]);
            }
        if (pr.feasible)
            for (Index j = 0; j < n; ++j) {
                CHECK(clo[z(j)] >= plo[z(j)]);
                CHECK(chi[z(j)] <= phi[z(j)]);
            }
        for (Index j = 0; j < n; ++j) tightened += clo[z(j)] > lo[z(j)] || chi[z(j)] < hi[z(j)];
        // A child: one more bound, propagated from its seed only.
        const Index b = static_cast<Index>(rng() % static_cast<unsigned>(n));
        if (chi[z(b)] > clo[z(b)]) {
            auto slo = clo, shi = chi;
            shi[z(b)] = slo[z(b)];
            auto flo = slo, fhi = shi;
            const std::vector<Index> seed{b};
            const auto sr = forge::search::propagate_bounds_cached(lp, cache, slo, shi, &seed, nullptr, scratch,
                                                                 nullptr, 0, 1e-9, 100000);
            RowActivityCache fresh;
            fresh.build(lp, flo, fhi);
            const auto fr = forge::search::propagate_bounds_cached(lp, fresh, flo, fhi, nullptr, nullptr, scratch,
                                                                 nullptr, 0, 1e-9, 100000);
            CHECK(sr.feasible == fr.feasible);
            if (sr.feasible && fr.feasible) {
                ++seeded_tried;
                if (slo == flo && shi == fhi) ++seeded_same;
            }
        }
        ++checked;
    }
    std::cout << "ACTIVITY_PROPAGATION checked=" << checked << " emptied=" << emptied
              << " tightened_cols=" << tightened << " seeded=" << seeded_tried << " same=" << seeded_same << '\n';
    CHECK(checked == 1500);
    CHECK(emptied > 20);
    CHECK(tightened > 500);
    CHECK(seeded_tried > 300);
    CHECK(seeded_same == seeded_tried);
}

// A local cut parallel to a model row (x + y <= 1 against x + y <= 2): what it
// alone implies is recorded as ReasonKind::Cut. Folded into the model row's
// side, the same deduction carried a global Row reason.
void test_local_rows_keep_their_scope() {
    LpProblem lp;
    lp.name = "scope";
    lp.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.row_lo = {-kInf};
    lp.row_hi = {2.0};
    lp.c = {0.0, 0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};
    std::vector<f64> lo{1.0, 0.0}, hi{1.0, 1.0};
    RowActivityCache cache;
    cache.build(lp, lp.col_lo, lp.col_hi);
    const std::vector<Index> cols{0, 1};
    const std::vector<f64> vals{1.0, 1.0};
    std::vector<forge::search::LocalRowView> local{{&cols, &vals, -kInf, 1.0}};
    PropagationScratch scratch;
    forge::search::PropTrail trail;
    const std::vector<Index> seed{0};
    const auto r = forge::search::propagate_bounds_cached(lp, cache, lo, hi, &seed, &local, scratch,
                                                        &trail, 1, 1e-9, 1000);
    CHECK(r.feasible);
    CHECK(hi[1] == 0.0);
    bool found = false;
    for (const auto& e : trail.entries())
        if (e.var == 1 && e.dir == forge::search::BoundDir::Upper) {
            found = true;
            CHECK(e.kind == forge::search::ReasonKind::Cut);
            CHECK(e.reason_id == 0);
        }
    CHECK(found);
}

}  // namespace

int main() {
    test_checkpoint_sync_restores_siblings();
    test_structural_remap();
    test_consistency();
    test_sound_and_strong();
    test_local_rows_keep_their_scope();
    return forge::test::finish("test_activity_cache");
}
