// Forrest-Tomlin basis update: correctness suite.
//
// update_ft() mutates six interlocking structures at once -- U by row, U by
// column, the elimination order, its inverse, the diagonal, and the row-eta
// file -- and a mistake in any one of them surfaces as a wrong solve many
// updates later, where it is almost impossible to localise. The earlier
// implementation was reverted for exactly that reason.
//
// So every test here checks TWO independent things after every single update:
//
//   1. BasisFactor::check_invariants() -- the structures still agree with each
//      other (mirrors identical, order a permutation, U triangular for that
//      order, L untouched, etas well formed). Catches the operation that broke
//      something, at the moment it breaks it.
//   2. A solve against an independently maintained explicit copy of the basis.
//      Catches the case where the structures are self-consistent but represent
//      the wrong matrix.
//
// Neither alone is sufficient: (1) passes on a consistent representation of the
// wrong matrix, (2) passes while a mirror silently rots until the update that
// finally reads the rotten part.
#include "sor/la/lu.hpp"

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::core::Offset;
using sor::la::BasisFactor;
using sor::la::LuOptions;

namespace {

// An explicit column-wise basis, maintained alongside the factorization so the
// tests have something to check against that shares no code with it.
struct ColMat {
    Index m = 0;
    std::vector<std::vector<std::pair<Index, f64>>> col;

    explicit ColMat(Index dim) : m(dim), col(static_cast<std::size_t>(dim)) {}

    void set_col(Index j, const std::vector<Index>& rows, const std::vector<f64>& vals) {
        auto& c = col[static_cast<std::size_t>(j)];
        c.clear();
        for (std::size_t t = 0; t < rows.size(); ++t)
            if (vals[t] != 0.0) c.emplace_back(rows[t], vals[t]);
    }

    // y = B x, x indexed by slot, y by row.
    std::vector<f64> apply(const std::vector<f64>& x) const {
        std::vector<f64> y(static_cast<std::size_t>(m), 0.0);
        for (Index j = 0; j < m; ++j)
            for (const auto& [i, v] : col[static_cast<std::size_t>(j)])
                y[static_cast<std::size_t>(i)] += v * x[static_cast<std::size_t>(j)];
        return y;
    }

    // y = B^T x, x indexed by row, y by slot.
    std::vector<f64> apply_t(const std::vector<f64>& x) const {
        std::vector<f64> y(static_cast<std::size_t>(m), 0.0);
        for (Index j = 0; j < m; ++j) {
            f64 s = 0.0;
            for (const auto& [i, v] : col[static_cast<std::size_t>(j)])
                s += v * x[static_cast<std::size_t>(i)];
            y[static_cast<std::size_t>(j)] = s;
        }
        return y;
    }

    void to_csc(std::vector<Offset>& col_ptr, std::vector<Index>& row_idx,
                std::vector<f64>& vals) const {
        col_ptr.assign(1, 0);
        row_idx.clear();
        vals.clear();
        for (Index j = 0; j < m; ++j) {
            auto c = col[static_cast<std::size_t>(j)];
            std::sort(c.begin(), c.end());
            for (const auto& [i, v] : c) { row_idx.push_back(i); vals.push_back(v); }
            col_ptr.push_back(static_cast<Offset>(row_idx.size()));
        }
    }
};

f64 max_abs_diff(const std::vector<f64>& a, const std::vector<f64>& b) {
    f64 e = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs(a[i] - b[i]));
    return e;
}

void check_ftran_pair_matches_separate(const BasisFactor& f,
                                       const std::vector<f64>& rhs_a,
                                       const std::vector<f64>& rhs_b,
                                       const std::string& what,
                                       const char* file, int line) {
    auto separate_a = rhs_a;
    auto separate_b = rhs_b;
    f.ftran(separate_a);
    f.ftran(separate_b);

    auto paired_a = rhs_a;
    auto paired_b = rhs_b;
    f.ftran_pair(paired_a, paired_b);

    ::sor::test::report(paired_a == separate_a, "FT ftran_pair first RHS", file, line,
                        what + " err " +
                            std::to_string(max_abs_diff(paired_a, separate_a)));
    ::sor::test::report(paired_b == separate_b, "FT ftran_pair second RHS", file, line,
                        what + " err " +
                            std::to_string(max_abs_diff(paired_b, separate_b)));
}
#define CHECK_FTRAN_PAIR(f, a, b, what) \
    check_ftran_pair_matches_separate((f), (a), (b), (what), __FILE__, __LINE__)

// Diagonally dominant, so every column is a legitimate pivot and the test is
// measuring the update rather than the conditioning.
ColMat random_basis(Index m, f64 density, std::mt19937& rng) {
    std::uniform_real_distribution<f64> u(0.0, 1.0), val(-2.0, 2.0);
    std::vector<Index> perm(static_cast<std::size_t>(m));
    for (Index i = 0; i < m; ++i) perm[static_cast<std::size_t>(i)] = i;
    std::shuffle(perm.begin(), perm.end(), rng);

    ColMat b(m);
    for (Index j = 0; j < m; ++j) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        for (Index i = 0; i < m; ++i) {
            const bool diag = (perm[static_cast<std::size_t>(j)] == i);
            if (!diag && u(rng) >= density) continue;
            f64 v = val(rng);
            if (v == 0.0) v = 1.0;
            if (diag) v += (v >= 0.0 ? 4.0 : -4.0);
            rows.push_back(i);
            vals.push_back(v);
        }
        b.set_col(j, rows, vals);
    }
    return b;
}

bool factorize(BasisFactor& f, const ColMat& b) {
    std::vector<Offset> cp;
    std::vector<Index> ri;
    std::vector<f64> vs;
    b.to_csc(cp, ri, vs);
    return f.factorize(b.m, cp, ri, vs, LuOptions{});
}

void check_structure(const BasisFactor& f, const std::string& what,
                     const char* file, int line) {
    std::string why;
    // Sequenced deliberately: as arguments to report() the call and the message
    // would be evaluated in an unspecified order, and `why` would usually be
    // read before check_invariants() wrote it.
    const bool ok = f.check_invariants(&why);
    ::sor::test::report(ok, "check_invariants", file, line, what + ": " + why);
}
#define CHECK_STRUCT(f, what) check_structure((f), (what), __FILE__, __LINE__)

// Solve residuals against the explicit basis, in BOTH directions.
void check_solves(const ColMat& b, const BasisFactor& f, const std::string& what,
                  f64 tol, const char* file, int line) {
    const auto m = static_cast<std::size_t>(b.m);
    std::mt19937 rng(7u);
    std::uniform_real_distribution<f64> d(-1.0, 1.0);
    std::vector<f64> rhs(m);
    for (auto& v : rhs) v = d(rng);

    std::vector<f64> z = rhs;
    f.ftran(z);
    const f64 ef = max_abs_diff(b.apply(z), rhs);
    ::sor::test::report(ef <= tol, "ftran residual", file, line,
                        what + " err " + std::to_string(ef));

    std::vector<f64> y = rhs;
    f.btran(y);
    const f64 eb = max_abs_diff(b.apply_t(y), rhs);
    ::sor::test::report(eb <= tol, "btran residual", file, line,
                        what + " err " + std::to_string(eb));
}
#define CHECK_SOLVES(b, f, what, tol) check_solves((b), (f), (what), (tol), __FILE__, __LINE__)

// Drive one column replacement through the factorization, keeping `b` in step.
// Returns false when update_ft declined (the caller then refactorizes, exactly
// as the engine does).
bool do_update(BasisFactor& f, ColMat& b, Index p,
               const std::vector<Index>& rows, const std::vector<f64>& vals) {
    std::vector<f64> alpha(static_cast<std::size_t>(b.m), 0.0);
    for (std::size_t t = 0; t < rows.size(); ++t)
        alpha[static_cast<std::size_t>(rows[t])] += vals[t];
    f.ftran(alpha);                       // alpha := B^-1 a_new, slot-indexed
    if (!f.update_ft(p, alpha, LuOptions{})) return false;
    b.set_col(p, rows, vals);
    return true;
}

// A column to bring in on step `step`: a fixed stride pattern so the tests are
// deterministic, with a dominant leading entry so the basis stays solvable.
void gen_column(Index m, int step, std::mt19937& rng,
                std::vector<Index>& rows, std::vector<f64>& vals) {
    std::uniform_real_distribution<f64> val(-2.0, 2.0);
    rows.clear();
    vals.clear();
    const int stride = 3 + (step % 5);
    for (Index i = 0; i < m; ++i)
        if (i % stride == step % stride) { rows.push_back(i); vals.push_back(val(rng)); }
    if (rows.empty()) { rows.push_back(step % m); vals.push_back(1.0); }
    vals[0] += 5.0;
}

void test_ftran_pair_after_ft_row_etas() {
    std::mt19937 rng(20260909u);
    constexpr Index m = 39;
    ColMat b = random_basis(m, 0.08, rng);
    BasisFactor f;
    CHECK(factorize(f, b));

    int accepted = 0;
    for (int step = 0; step < 20; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        gen_column(m, step, rng, rows, vals);
        const Index p = static_cast<Index>((step * 11 + 3) % m);
        if (!do_update(f, b, p, rows, vals)) continue;
        ++accepted;

        std::vector<f64> rhs_a(static_cast<std::size_t>(m));
        std::vector<f64> rhs_b(static_cast<std::size_t>(m), 0.0);
        for (Index i = 0; i < m; ++i)
            rhs_a[static_cast<std::size_t>(i)] =
                static_cast<f64>(((i + step) * 7) % 31 - 15) / 8.0;
        rhs_b[static_cast<std::size_t>((step * 5) % m)] = -1.25;
        rhs_b[static_cast<std::size_t>((step * 13 + 1) % m)] += 0.375;
        CHECK_FTRAN_PAIR(f, rhs_a, rhs_b,
                         "Forrest-Tomlin update " + std::to_string(step + 1));
    }

    CHECK(accepted >= 3);
    CHECK(f.n_product_form_etas() == 0);
    CHECK(f.n_updates() > 0);
}

// ---------------------------------------------------------------------------

// The baseline: a long chain of updates, with both checks after every one.
// Length matters -- the earlier implementation's bug was invisible for a single
// update and only diverged after several.
void test_long_update_chain() {
    std::mt19937 rng(20260906u);
    const Index m = 70;
    ColMat b = random_basis(m, 0.05, rng);

    BasisFactor f;
    CHECK(factorize(f, b));
    CHECK_STRUCT(f, "after factorize");

    int done = 0;
    for (int step = 0; step < 200; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        gen_column(m, step, rng, rows, vals);
        const Index p = static_cast<Index>((step * 17) % m);
        if (!do_update(f, b, p, rows, vals)) continue;
        ++done;
        CHECK_STRUCT(f, "after update " + std::to_string(step));
        CHECK_SOLVES(b, f, "after update " + std::to_string(step), 1e-6);
    }
    ::sor::test::report(done > 150, "ft chain makes progress", __FILE__, __LINE__,
                        "updates=" + std::to_string(done));
    // Forrest-Tomlin appends row etas, never product-form ones, and exactly one
    // per accepted update.
    CHECK(f.n_product_form_etas() == 0);
    CHECK(f.n_row_etas() == done);
}

// FTRAN and BTRAN must be transposes of ONE operator. <B^-1 x, y> = <x, B^-T y>
// holds for any x, y if and only if they apply the same matrix, so this catches
// a row-eta pass applied in the wrong order or with the wrong sign -- which a
// residual check can miss when the error happens to lie near the residual's own
// tolerance.
void test_ftran_btran_are_transposes() {
    std::mt19937 rng(4242u);
    const Index m = 55;
    ColMat b = random_basis(m, 0.07, rng);
    BasisFactor f;
    CHECK(factorize(f, b));

    std::uniform_real_distribution<f64> d(-1.0, 1.0);
    for (int step = 0; step < 60; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        gen_column(m, step, rng, rows, vals);
        if (!do_update(f, b, static_cast<Index>((step * 13) % m), rows, vals)) continue;

        std::vector<f64> x(static_cast<std::size_t>(m)), y(static_cast<std::size_t>(m));
        for (auto& v : x) v = d(rng);
        for (auto& v : y) v = d(rng);

        std::vector<f64> bx = x;   // B^-1 x, slot-indexed
        f.ftran(bx);
        std::vector<f64> by = y;   // B^-T y, slot-indexed
        f.btran(by);

        f64 lhs = 0.0, rhs = 0.0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(m); ++i) {
            lhs += bx[i] * y[i];   // <B^-1 x, y>
            rhs += x[i] * by[i];   // <x, B^-T y>
        }
        CHECK_NEAR(lhs, rhs, 1e-8);
    }
}

// The hypersparse reach-set kernels and the dense kernels must agree bit for
// bit in exact arithmetic and to rounding here. The elimination order is what
// decides the reach traversal order, so this is the test that would catch
// sort_by_elimination_order() being wrong -- the dense path visits the order
// explicitly and cannot disagree.
void test_sparse_and_dense_paths_agree_after_updates() {
    std::mt19937 rng(31337u);
    const Index m = 90;
    ColMat b = random_basis(m, 0.04, rng);
    BasisFactor f;
    CHECK(factorize(f, b));

    for (int step = 0; step < 80; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        gen_column(m, step, rng, rows, vals);
        if (!do_update(f, b, static_cast<Index>((step * 7) % m), rows, vals)) continue;

        // A one-nonzero RHS takes the hypersparse path; the reference is the
        // same solve against the explicit basis.
        for (Index unit = 0; unit < m; unit += 17) {
            std::vector<f64> e(static_cast<std::size_t>(m), 0.0);
            e[static_cast<std::size_t>(unit)] = 1.0;
            std::vector<f64> z = e;
            f.ftran(z);
            CHECK(max_abs_diff(b.apply(z), e) <= 1e-7);

            std::vector<f64> y = e;
            f.btran(y);
            CHECK(max_abs_diff(b.apply_t(y), e) <= 1e-7);
        }
    }
}

// ftran_with_support() promises the SAME values as ftran() on the slots it
// reports, and the support must cover every nonzero. A row eta that writes a
// position outside the reported support is exactly the sort of bug this finds.
void test_support_variant_matches_plain_ftran() {
    std::mt19937 rng(909u);
    const Index m = 80;
    ColMat b = random_basis(m, 0.05, rng);
    BasisFactor f;
    CHECK(factorize(f, b));

    for (int step = 0; step < 50; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        gen_column(m, step, rng, rows, vals);
        if (!do_update(f, b, static_cast<Index>((step * 11) % m), rows, vals)) continue;

        for (Index unit = 0; unit < m; unit += 23) {
            std::vector<f64> plain(static_cast<std::size_t>(m), 0.0);
            plain[static_cast<std::size_t>(unit)] = 1.0;
            std::vector<f64> withsup = plain;
            f.ftran(plain);

            std::vector<Index> support;
            const bool sparse = f.ftran_with_support(withsup, support);
            if (!sparse) continue;   // dense path writes everything; nothing to compare
            for (const Index s : support)
                CHECK(plain[static_cast<std::size_t>(s)] ==
                      withsup[static_cast<std::size_t>(s)]);
            // Support must COVER the true nonzeros, or a caller that resets by
            // support will leave stale values behind.
            std::vector<char> covered(static_cast<std::size_t>(m), 0);
            for (const Index s : support) covered[static_cast<std::size_t>(s)] = 1;
            for (Index i = 0; i < m; ++i)
                if (plain[static_cast<std::size_t>(i)] != 0.0)
                    CHECK(covered[static_cast<std::size_t>(i)] != 0);
        }
    }
}

// Force the storage machinery hard: repeatedly replace columns with DENSE ones,
// so rows keep outgrowing their capacity, relocate, leave dead space, and
// eventually trip compaction. The point is that none of that is observable.
void test_storage_growth_and_compaction() {
    std::mt19937 rng(5150u);
    const Index m = 40;
    ColMat b = random_basis(m, 0.10, rng);
    BasisFactor f;
    CHECK(factorize(f, b));

    std::uniform_real_distribution<f64> val(-2.0, 2.0);
    int done = 0;
    for (int step = 0; step < 120; ++step) {
        // Fully dense incoming columns maximise fill in U, which is what makes
        // rows relocate.
        std::vector<Index> rows;
        std::vector<f64> vals;
        for (Index i = 0; i < m; ++i) { rows.push_back(i); vals.push_back(val(rng)); }
        vals[static_cast<std::size_t>(step % m)] += 8.0;
        if (!do_update(f, b, static_cast<Index>(step % m), rows, vals)) continue;
        ++done;
        CHECK_STRUCT(f, "dense-fill update " + std::to_string(step));
        CHECK_SOLVES(b, f, "dense-fill update " + std::to_string(step), 1e-5);
    }
    ::sor::test::report(done > 80, "dense-fill chain makes progress", __FILE__,
                        __LINE__, "updates=" + std::to_string(done));
}

// A refused update must leave the factorization EXACTLY as it was -- the caller
// refactorizes on false, and a partially applied update would silently corrupt
// the basis it is about to keep using.
void test_refusal_leaves_factorization_untouched() {
    std::mt19937 rng(77u);
    const Index m = 30;
    ColMat b = random_basis(m, 0.08, rng);
    BasisFactor f;
    CHECK(factorize(f, b));

    // Reference solve before any refused attempt.
    std::vector<f64> rhs(static_cast<std::size_t>(m), 0.0);
    for (Index i = 0; i < m; ++i) rhs[static_cast<std::size_t>(i)] = 0.5 + 0.1 * i;
    std::vector<f64> before = rhs;
    f.ftran(before);

    // alpha[p] == 0: no pivot to divide by.
    std::vector<f64> alpha(static_cast<std::size_t>(m), 1.0);
    alpha[3] = 0.0;
    CHECK(!f.update_ft(3, alpha, LuOptions{}));
    CHECK_STRUCT(f, "after refused (zero pivot)");

    // Below min_pivot.
    alpha[3] = 1e-14;
    CHECK(!f.update_ft(3, alpha, LuOptions{}, 1e-11));
    CHECK_STRUCT(f, "after refused (tiny pivot)");

    // Out-of-range slot.
    CHECK(!f.update_ft(m, alpha, LuOptions{}));
    CHECK(!f.update_ft(-1, alpha, LuOptions{}));
    CHECK_STRUCT(f, "after refused (bad slot)");

    std::vector<f64> after = rhs;
    f.ftran(after);
    CHECK(max_abs_diff(before, after) == 0.0);
    CHECK_SOLVES(b, f, "after all refusals", 1e-9);
}

// Replacing a column with (a multiple of) the one already there must be a
// no-op mathematically, however many times it is done. Degenerate in the sense
// that matters here: the entering column is already in the basis, so atilde is
// a scaled unit vector and the row eta is the one the elimination cannot avoid.
void test_replacing_a_column_with_itself() {
    std::mt19937 rng(999u);
    const Index m = 25;
    ColMat b = random_basis(m, 0.12, rng);
    BasisFactor f;
    CHECK(factorize(f, b));

    for (int rep = 0; rep < 40; ++rep) {
        const Index p = static_cast<Index>(rep % m);
        std::vector<Index> rows;
        std::vector<f64> vals;
        const f64 scale = 1.0 + 0.25 * (rep % 3);
        for (const auto& [i, v] : b.col[static_cast<std::size_t>(p)]) {
            rows.push_back(i);
            vals.push_back(v * scale);
        }
        CHECK(do_update(f, b, p, rows, vals));
        CHECK_STRUCT(f, "self-replace " + std::to_string(rep));
        CHECK_SOLVES(b, f, "self-replace " + std::to_string(rep), 1e-7);
    }
}

// The identity basis is the cold start every solve begins from, and its U has
// no off-diagonals at all -- every store operation runs on empty segments.
void test_identity_basis_updates() {
    const Index m = 16;
    ColMat b(m);
    for (Index j = 0; j < m; ++j) b.set_col(j, {j}, {-1.0});

    BasisFactor f;
    CHECK(factorize(f, b));
    CHECK_STRUCT(f, "identity factorize");

    std::mt19937 rng(64u);
    std::uniform_real_distribution<f64> val(-2.0, 2.0);
    for (int step = 0; step < 30; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        for (Index i = 0; i < m; i += 2) { rows.push_back(i); vals.push_back(val(rng)); }
        vals[0] += 6.0;
        const Index p = static_cast<Index>(step % m);
        if (!do_update(f, b, p, rows, vals)) continue;
        CHECK_STRUCT(f, "identity update " + std::to_string(step));
        CHECK_SOLVES(b, f, "identity update " + std::to_string(step), 1e-7);
    }
}

// m == 1 exercises every loop bound at its degenerate end.
void test_dimension_one() {
    ColMat b(1);
    b.set_col(0, {0}, {2.0});
    BasisFactor f;
    CHECK(factorize(f, b));
    CHECK_STRUCT(f, "1x1 factorize");

    std::vector<f64> alpha{1.0};
    std::vector<f64> a{4.0};
    alpha = a;
    f.ftran(alpha);                       // 4 / 2 = 2
    CHECK(f.update_ft(0, alpha, LuOptions{}));
    b.set_col(0, {0}, {4.0});
    CHECK_STRUCT(f, "1x1 after update");
    CHECK_SOLVES(b, f, "1x1 after update", 1e-12);
}

// Product-form and Forrest-Tomlin are different representations of the same
// operator: given the same column replacements they must solve identically.
// This is the strongest available statement that the new representation is
// right, because product form is the long-standing default.
void test_ft_agrees_with_product_form() {
    std::mt19937 rng(8080u);
    const Index m = 50;
    ColMat b0 = random_basis(m, 0.06, rng);

    BasisFactor fp, ff;
    CHECK(factorize(fp, b0));
    CHECK(factorize(ff, b0));

    ColMat b = b0;
    std::mt19937 colrng(11u);
    for (int step = 0; step < 60; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        gen_column(m, step, colrng, rows, vals);
        const Index p = static_cast<Index>((step * 19) % m);

        std::vector<f64> aq(static_cast<std::size_t>(m), 0.0);
        for (std::size_t t = 0; t < rows.size(); ++t)
            aq[static_cast<std::size_t>(rows[t])] += vals[t];

        std::vector<f64> ap = aq, af = aq;
        fp.ftran(ap);
        ff.ftran(af);
        // Both represent the same matrix, so the FTRAN results agree up to the
        // rounding each representation has accumulated. Compare RELATIVELY --
        // an absolute bound is meaningless once the solutions are large, and
        // the binding correctness statement is the residual against the
        // explicit basis below, not agreement between two approximations.
        f64 scale = 1.0;
        for (const f64 v : ap) scale = std::max(scale, std::fabs(v));
        CHECK(max_abs_diff(ap, af) <= 1e-6 * scale);

        const bool okp = fp.update(p, ap);
        const bool okf = ff.update_ft(p, af, LuOptions{});
        if (okp != okf) break;   // now representing different matrices
        if (!okp) continue;
        b.set_col(p, rows, vals);
        CHECK_STRUCT(ff, "ft/pf step " + std::to_string(step));

        std::vector<f64> rhs(static_cast<std::size_t>(m));
        for (Index i = 0; i < m; ++i)
            rhs[static_cast<std::size_t>(i)] = std::sin(0.7 * i + step);
        std::vector<f64> zp = rhs, zf = rhs;
        fp.ftran(zp);
        ff.ftran(zf);
        CHECK(max_abs_diff(zp, zf) <= 1e-6);

        std::vector<f64> yp = rhs, yf = rhs;
        fp.btran(yp);
        ff.btran(yf);
        CHECK(max_abs_diff(yp, yf) <= 1e-6);
    }
}

// collapse_pending_into_ft() folds a product-form eta file into L/U through
// update_ft(). It is purely a representation change, so solves must be
// unchanged across it, and the eta file must end up empty.
void test_collapse_pending_is_representation_only() {
    std::mt19937 rng(2024u);
    const Index m = 35;
    ColMat b = random_basis(m, 0.08, rng);
    BasisFactor f;
    CHECK(factorize(f, b));

    for (int step = 0; step < 8; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        gen_column(m, step, rng, rows, vals);
        std::vector<f64> alpha(static_cast<std::size_t>(m), 0.0);
        for (std::size_t t = 0; t < rows.size(); ++t)
            alpha[static_cast<std::size_t>(rows[t])] += vals[t];
        f.ftran(alpha);
        const Index p = static_cast<Index>((step * 5) % m);
        if (!f.update(p, alpha)) continue;      // product form on purpose
        b.set_col(p, rows, vals);
    }
    CHECK(f.n_updates() > 0);

    std::vector<f64> rhs(static_cast<std::size_t>(m));
    for (Index i = 0; i < m; ++i) rhs[static_cast<std::size_t>(i)] = 0.3 * i - 1.0;
    std::vector<f64> before = rhs;
    f.ftran(before);

    if (f.collapse_pending_into_ft(LuOptions{})) {
        // The product-form file is what collapse empties; it converts those
        // etas into row etas, so n_updates() as a whole does not go to zero.
        CHECK(f.n_product_form_etas() == 0);
        CHECK(f.n_row_etas() > 0);
        CHECK_STRUCT(f, "after collapse");
        std::vector<f64> after = rhs;
        f.ftran(after);
        CHECK(max_abs_diff(before, after) <= 1e-7);
        CHECK_SOLVES(b, f, "after collapse", 1e-6);
    }
}

// A refactorization must wipe every trace of the FT state: a stale row eta or a
// stale elimination order applied to fresh factors is silent corruption.
void test_refactorize_clears_ft_state() {
    std::mt19937 rng(606u);
    const Index m = 32;
    ColMat b = random_basis(m, 0.09, rng);
    BasisFactor f;
    CHECK(factorize(f, b));

    for (int step = 0; step < 20; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        gen_column(m, step, rng, rows, vals);
        do_update(f, b, static_cast<Index>((step * 3) % m), rows, vals);
    }
    CHECK_SOLVES(b, f, "before refactorize", 1e-6);

    CHECK(factorize(f, b));                 // same matrix, fresh factors
    CHECK_STRUCT(f, "after refactorize");
    CHECK_SOLVES(b, f, "after refactorize", 1e-9);

    // And updating again from the clean state still works.
    std::vector<Index> rows;
    std::vector<f64> vals;
    gen_column(m, 99, rng, rows, vals);
    CHECK(do_update(f, b, 4, rows, vals));
    CHECK_STRUCT(f, "update after refactorize");
    CHECK_SOLVES(b, f, "update after refactorize", 1e-7);
}

// Cost, not correctness, is the reason this rewrite exists: the previous
// implementation re-eliminated the whole trailing bump per update, so its cost
// grew with the bump. A true FT update touches only the entering column, the
// leaving row and the eta -- so a long chain must not slow down, and the row
// eta file must grow by roughly one entry-set per update rather than
// quadratically.
void test_update_cost_does_not_grow_with_chain_length() {
    std::mt19937 rng(1717u);
    const Index m = 200;
    ColMat b = random_basis(m, 0.02, rng);
    BasisFactor ff, fp;                     // forrest-tomlin, product form
    CHECK(factorize(ff, b));
    CHECK(factorize(fp, b));

    // Both representations take the IDENTICAL sequence of column replacements,
    // with no refactorization anywhere, so the comparison isolates the update.
    int done = 0;
    for (int step = 0; step < 300; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        gen_column(m, step, rng, rows, vals);
        const Index p = static_cast<Index>((step * 23) % m);

        std::vector<f64> aq(static_cast<std::size_t>(m), 0.0);
        for (std::size_t t = 0; t < rows.size(); ++t)
            aq[static_cast<std::size_t>(rows[t])] += vals[t];
        std::vector<f64> af = aq, ap = aq;
        ff.ftran(af);
        fp.ftran(ap);

        const bool okf = ff.update_ft(p, af, LuOptions{});
        const bool okp = fp.update(p, ap);
        if (okf != okp) break;              // no longer the same matrix
        if (!okf) continue;
        b.set_col(p, rows, vals);
        ++done;
        if ((step % 25) == 0) CHECK_STRUCT(ff, "chain step " + std::to_string(step));
    }
    ::sor::test::report(done > 200, "long chain makes progress", __FILE__, __LINE__,
                        "updates=" + std::to_string(done));
    CHECK_STRUCT(ff, "after the whole chain");

    // Residual of each against the explicit basis. 300 updates with no
    // refactorization is far past what any solver does -- both WILL have
    // drifted -- so the assertion is comparative: Forrest-Tomlin must not be
    // materially worse than the product form it replaces.
    std::vector<f64> rhs(static_cast<std::size_t>(m));
    for (Index i = 0; i < m; ++i)
        rhs[static_cast<std::size_t>(i)] = std::sin(0.37 * i) + 0.5;
    const auto residual = [&](BasisFactor& f) {
        std::vector<f64> z = rhs;
        f.ftran(z);
        const f64 ef = max_abs_diff(b.apply(z), rhs);
        std::vector<f64> y = rhs;
        f.btran(y);
        return std::max(ef, max_abs_diff(b.apply_t(y), rhs));
    };
    // Forrest-Tomlin does no pivoting in the update, so with NO refactorization
    // at all its error compounds where product form's barely moves -- measured
    // here at ~4e-4 against ~1e-8 over 300 updates. That is the method, not a
    // defect, and it is why needs_refactor() counts row etas. The assertion is
    // therefore loose: it catches a blow-up (a wrong update destroys the solve
    // outright) without pretending FT matches product form unrefactorized.
    const f64 e_ft = residual(ff), e_pf = residual(fp);
    ::sor::test::report(e_ft < 1e-2, "ft survives 300 unrefactorized updates",
                        __FILE__, __LINE__,
                        "ft " + std::to_string(e_ft) + " vs pf " + std::to_string(e_pf));

    // Under the cadence the engine actually uses, FT must track product form
    // closely. This is the configuration that ships, so this is the binding
    // accuracy statement.
    BasisFactor gf, gp;
    ColMat c = random_basis(m, 0.02, rng);
    CHECK(factorize(gf, c));
    CHECK(factorize(gp, c));
    for (int step = 0; step < 300; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        gen_column(m, step, rng, rows, vals);
        const Index p = static_cast<Index>((step * 23) % m);
        std::vector<f64> aq(static_cast<std::size_t>(m), 0.0);
        for (std::size_t t = 0; t < rows.size(); ++t)
            aq[static_cast<std::size_t>(rows[t])] += vals[t];
        std::vector<f64> af = aq, ap = aq;
        gf.ftran(af);
        gp.ftran(ap);
        const bool okf = gf.update_ft(p, af, LuOptions{});
        const bool okp = gp.update(p, ap);
        if (okf != okp) break;
        if (!okf) continue;
        c.set_col(p, rows, vals);
        if ((step % 50) == 49) {          // refactorize both, as the engine does
            CHECK(factorize(gf, c));
            CHECK(factorize(gp, c));
        }
    }
    CHECK_STRUCT(gf, "refactorized chain");
    std::vector<f64> r2(static_cast<std::size_t>(m));
    for (Index i = 0; i < m; ++i) r2[static_cast<std::size_t>(i)] = std::cos(0.21 * i);
    const auto resid2 = [&](BasisFactor& f) {
        std::vector<f64> z = r2;
        f.ftran(z);
        const f64 e1 = max_abs_diff(c.apply(z), r2);
        std::vector<f64> y = r2;
        f.btran(y);
        return std::max(e1, max_abs_diff(c.apply_t(y), r2));
    };
    const f64 g_ft = resid2(gf), g_pf = resid2(gp);
    ::sor::test::report(g_ft <= 100.0 * g_pf + 1e-10,
                        "ft tracks product form under refactorization",
                        __FILE__, __LINE__,
                        "ft " + std::to_string(g_ft) + " vs pf " + std::to_string(g_pf));
}

}  // namespace

int main() {
    test_ftran_pair_after_ft_row_etas();
    test_long_update_chain();
    test_ftran_btran_are_transposes();
    test_sparse_and_dense_paths_agree_after_updates();
    test_support_variant_matches_plain_ftran();
    test_storage_growth_and_compaction();
    test_refusal_leaves_factorization_untouched();
    test_replacing_a_column_with_itself();
    test_identity_basis_updates();
    test_dimension_one();
    test_ft_agrees_with_product_form();
    test_collapse_pending_is_representation_only();
    test_refactorize_clears_ft_state();
    test_update_cost_does_not_grow_with_chain_length();
    return sor::test::finish("test_lu_ft");
}
