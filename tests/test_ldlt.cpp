// Sparse LDL' + AMD: solves must be right, orderings must be permutations,
// and AMD must actually reduce fill where fill is avoidable.
//
// Checked on: random sparse SPD; 2-D grid Laplacians (the textbook fill
// case, where the natural order fills badly and a minimum-degree order must
// do far better); quasi-definite KKT matrices [-(Q+D) A'; A D2] with the
// signs the IPM will pass; and a dense row, which AMD must order last.
#include "sor/core/parallel.hpp"
#include "sor/la/ldlt.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <vector>

using namespace sor;
using la::Index;
using la::Offset;
using la::SymCsc;
using core::f64;

namespace {

struct Lcg {
    std::uint64_t s;
    f64 next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<f64>(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

// Build SymCsc from a map of upper-triangle entries.
SymCsc from_map(Index n, const std::map<std::pair<Index, Index>, f64>& m) {
    SymCsc a;
    a.n = n;
    a.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto& [ij, v] : m) ++a.col_ptr[static_cast<std::size_t>(ij.second) + 1];
    for (Index j = 0; j < n; ++j) a.col_ptr[static_cast<std::size_t>(j) + 1] += a.col_ptr[static_cast<std::size_t>(j)];
    a.row_idx.resize(m.size());
    a.vals.resize(m.size());
    std::vector<Offset> cur(a.col_ptr.begin(), a.col_ptr.end() - 1);
    for (const auto& [ij, v] : m) {
        const auto slot = static_cast<std::size_t>(cur[static_cast<std::size_t>(ij.second)]++);
        a.row_idx[slot] = ij.first;
        a.vals[slot] = v;
    }
    return a;
}

std::vector<f64> matvec(const SymCsc& a, const std::vector<f64>& x) {
    std::vector<f64> y(x.size(), 0.0);
    for (Index j = 0; j < a.n; ++j)
        for (Offset t = a.col_ptr[static_cast<std::size_t>(j)]; t < a.col_ptr[static_cast<std::size_t>(j) + 1]; ++t) {
            const Index i = a.row_idx[static_cast<std::size_t>(t)];
            const f64 v = a.vals[static_cast<std::size_t>(t)];
            y[static_cast<std::size_t>(i)] += v * x[static_cast<std::size_t>(j)];
            if (i != j) y[static_cast<std::size_t>(j)] += v * x[static_cast<std::size_t>(i)];
        }
    return y;
}

f64 solve_residual(const SymCsc& a, la::Ldlt& f, std::uint64_t seed) {
    Lcg g{seed};
    std::vector<f64> x(static_cast<std::size_t>(a.n));
    for (auto& v : x) v = 2.0 * g.next() - 1.0;
    auto b = matvec(a, x);
    f.solve(b);
    f64 e = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) e = std::max(e, std::fabs(b[i] - x[i]));
    return e;
}

std::uint64_t hash_bits(const std::vector<f64>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (f64 x : v) {
        std::uint64_t u;
        std::memcpy(&u, &x, sizeof u);
        h ^= u;
        h *= 1099511628211ULL;
    }
    return h;
}

bool is_perm(const std::vector<Index>& p, Index n) {
    std::vector<Index> q = p;
    std::sort(q.begin(), q.end());
    for (Index i = 0; i < n; ++i) if (q[static_cast<std::size_t>(i)] != i) return false;
    return static_cast<Index>(p.size()) == n;
}

SymCsc grid_laplacian(Index k) {
    std::map<std::pair<Index, Index>, f64> m;
    auto id = [&](Index r, Index c) { return r * k + c; };
    for (Index r = 0; r < k; ++r)
        for (Index c = 0; c < k; ++c) {
            m[{id(r, c), id(r, c)}] = 4.0;
            if (c + 1 < k) m[{id(r, c), id(r, c + 1)}] = -1.0;
            if (r + 1 < k) m[{id(r, c), id(r + 1, c)}] = -1.0;
        }
    return from_map(k * k, m);
}

}  // namespace

int main() {
    // ---- grid Laplacian: correct, and AMD beats natural order on fill ----
    {
        const auto a = grid_laplacian(60);   // n = 3600
        std::vector<Index> natural(static_cast<std::size_t>(a.n));
        std::iota(natural.begin(), natural.end(), 0);
        la::Ldlt fn, fa;
        const std::vector<std::int8_t> pos(static_cast<std::size_t>(a.n), 1);
        fn.analyze(a, natural);
        CHECK(fn.factorize(a, pos, 1e-14));
        fa.analyze(a);
        CHECK(is_perm(fa.perm(), a.n));
        CHECK(fa.factorize(a, pos, 1e-14));
        CHECK(fa.regularized_pivots() == 0);
        CHECK(solve_residual(a, fn, 1) < 1e-10);
        CHECK(solve_residual(a, fa, 2) < 1e-10);
        std::printf("  grid 60x60: nnz(L) natural %lld, AMD %lld\n",
                    static_cast<long long>(fn.nnz_l()), static_cast<long long>(fa.nnz_l()));
        CHECK(fa.nnz_l() * 2 < fn.nnz_l());   // natural is banded (~k^3); AMD far less
    }

    // ---- random quasi-definite KKT: [-(Q + D1) A'; A D2] ----
    for (std::uint64_t seed = 1; seed <= 5; ++seed) {
        Lcg g{seed};
        const Index n = 300, m = 120, N = n + m;
        std::map<std::pair<Index, Index>, f64> e;
        for (Index j = 0; j < n; ++j) e[{j, j}] = -(1.0 + g.next());   // -(Q+D1) diagonal
        for (int t = 0; t < 600; ++t) {                                 // Q off-diagonal (PSD-ish via dominance)
            const Index i = static_cast<Index>(g.next() * n) % n, j = static_cast<Index>(g.next() * n) % n;
            if (i == j) continue;
            const f64 v = 0.1 * (2.0 * g.next() - 1.0);
            e[{std::min(i, j), std::max(i, j)}] -= v;
            e[{i, i}] -= std::fabs(v);
            e[{j, j}] -= std::fabs(v);
        }
        for (Index r = 0; r < m; ++r) {
            e[{n + r, n + r}] = 1e-2 + g.next();                          // D2
            for (int t = 0; t < 4; ++t) {
                const Index j = static_cast<Index>(g.next() * n) % n;
                e[{j, n + r}] += 2.0 * g.next() - 1.0;                     // A'
            }
        }
        const auto a = from_map(N, e);
        std::vector<std::int8_t> sign(static_cast<std::size_t>(N), 1);
        for (Index j = 0; j < n; ++j) sign[static_cast<std::size_t>(j)] = -1;
        la::Ldlt f;
        f.analyze(a);
        CHECK(is_perm(f.perm(), N));
        CHECK(f.factorize(a, sign, 1e-14));
        CHECK(f.regularized_pivots() == 0);   // quasi-definite: signs hold in any order
        CHECK(solve_residual(a, f, seed) < 1e-9);
    }

    // ---- a dense row goes last, and the factorization stays correct ----
    {
        const Index n = 2000;
        std::map<std::pair<Index, Index>, f64> e;
        for (Index j = 0; j < n; ++j) e[{j, j}] = 4.0 + static_cast<f64>(n);
        for (Index j = 0; j + 1 < n; ++j) e[{j, j + 1}] = -1.0;
        for (Index j = 1; j < n; ++j) e[{0, j}] = 0.5;               // row 0 dense
        const auto a = from_map(n, e);
        la::Ldlt f;
        f.analyze(a);
        CHECK(f.perm().back() == 0);
        CHECK(f.factorize(a, std::vector<std::int8_t>(static_cast<std::size_t>(n), 1), 1e-14));
        CHECK(solve_residual(a, f, 9) < 1e-10);
        CHECK(f.nnz_l() < 3 * n);
    }

    // ---- regularization: a zero pivot is replaced and counted ----
    {
        std::map<std::pair<Index, Index>, f64> e;
        e[{0, 0}] = 1.0; e[{1, 1}] = 0.0; e[{2, 2}] = 2.0;
        const auto a = from_map(3, e);
        la::Ldlt f;
        f.analyze(a);
        CHECK(f.factorize(a, {1, 1, 1}, 1e-8));
        CHECK(f.regularized_pivots() == 1);
    }

    // ---- the thread pool's own determinism contract ----
    {
        // deterministic_sum chunks a fixed way, so a sum whose value depends
        // on the association (0.1 repeated) must come out bit-identical at
        // every thread count -- and differ from the naive left-to-right sum,
        // which is what proves the test is not vacuous.
        const Offset n = 100000;
        const Offset chunk = 977;
        std::vector<f64> term(static_cast<std::size_t>(n));
        {
            Lcg g{31};
            for (Offset i = 0; i < n; ++i)   // magnitudes over ~2^60: the sum's
                term[static_cast<std::size_t>(i)] =      // value depends on the association
                    (2.0 * g.next() - 1.0) * std::ldexp(1.0, static_cast<int>(i % 60) - 30);
        }
        auto at = [&](Offset i) { return term[static_cast<std::size_t>(i)]; };
        // Independent reimplementation of the promised association: chunks of
        // `chunk` summed left to right, then the chunk sums left to right.
        f64 want = 0.0;
        for (Offset lo = 0; lo < n; lo += chunk) {
            f64 acc = 0.0;
            for (Offset i = lo; i < std::min(n, lo + chunk); ++i) acc += at(i);
            want += acc;
        }
        f64 naive = 0.0;
        for (Offset i = 0; i < n; ++i) naive += at(i);
        CHECK(want != naive);   // association matters here, so the test is not vacuous
        for (int p : {1, 2, 3, 5, 8}) {
            core::ThreadPool pool(p);
            const f64 v = core::deterministic_sum(pool, n, chunk, at);
            CHECK(hash_bits({v}) == hash_bits({want}));
        }
        // A fork-join job must run exactly once per worker.
        for (int p : {1, 2, 4, 8}) {
            core::ThreadPool pool(p);
            std::vector<int> hits(static_cast<std::size_t>(p), 0);
            pool.run(p, [&](int w) { ++hits[static_cast<std::size_t>(w)]; });
            for (int w = 0; w < p; ++w) CHECK(hits[static_cast<std::size_t>(w)] == 1);
        }
    }

    // ---- LDL' is BIT-IDENTICAL at 1 and N threads ----
    //
    // Big enough (n = 65,536) that the supernodal tree really is cut into
    // subtrees and the supernodes above the cut are factored cooperatively;
    // on a small matrix every thread count would take the serial path and
    // the check would pass vacuously.
    {
        const auto a = grid_laplacian(256);
        const std::vector<std::int8_t> pos(static_cast<std::size_t>(a.n), 1);
        la::Ldlt f;
        f.analyze(a);
        Lcg g{7};
        std::vector<f64> rhs(static_cast<std::size_t>(a.n));
        for (auto& v : rhs) v = 2.0 * g.next() - 1.0;
        std::uint64_t ref_fac = 0, ref_sol = 0;
        for (int p : {1, 2, 3, 5, 8}) {
            core::set_global_threads(p);
            CHECK(f.factorize(a, pos, 1e-14));
            auto b = rhs;
            f.solve(b);
            if (p == 1) {
                ref_fac = f.fingerprint();
                ref_sol = hash_bits(b);
            }
            CHECK(f.fingerprint() == ref_fac);
            CHECK(hash_bits(b) == ref_sol);
        }
        core::set_global_threads(0);
        std::printf("  grid 256x256: L/D and solve bit-identical at 1,2,3,5,8 threads\n");
    }

    // ---- relaxed amalgamation: same L, more stored zeros, same answer ----
    {
        const auto a = grid_laplacian(120);
        const std::vector<std::int8_t> pos(static_cast<std::size_t>(a.n), 1);
        la::LdltOptions off;
        off.amalgamate = false;
        la::Ldlt fa, fo;
        fo.set_options(off);
        fa.analyze(a);
        fo.analyze(a);
        // Merging changes neither the filled graph nor the true nnz(L) (any
        // topological order of the etree has the same fill, Liu 1990); it
        // only stores extra zeros and uses fewer, wider supernodes.
        CHECK(fa.nnz_l() == fo.nnz_l());
        CHECK(fa.stored_l() >= fo.stored_l());
        CHECK(fo.stored_l() == fo.nnz_l());
        CHECK(fa.supernodes() < fo.supernodes());
        CHECK(fa.factorize(a, pos, 1e-14));
        CHECK(fo.factorize(a, pos, 1e-14));
        CHECK(solve_residual(a, fa, 11) < 1e-10);
        CHECK(solve_residual(a, fo, 11) < 1e-10);
        std::printf("  grid 120x120: supernodes %d -> %d, stored %+.1f%%\n", fo.supernodes(),
                    fa.supernodes(),
                    100.0 * static_cast<double>(fa.stored_l() - fo.stored_l()) /
                        static_cast<double>(fo.stored_l()));
    }

    return sor::test::finish("test_ldlt");
}
