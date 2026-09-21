// Sparse LU: FTRAN, BTRAN, product-form update, singularity reporting.
//
// The LU is the one component whose bugs are silent -- a wrong factorization
// does not crash, it makes the simplex wander and then report a plausible wrong
// number. So every check here is against a matrix multiply the test does itself,
// never against the factorization's own view of things.
#include "sor/la/lu.hpp"

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::core::Offset;
using sor::la::BasisFactor;
using sor::la::LuOptions;

namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

class ScopedEnvironment {
public:
    ScopedEnvironment(const char* name, const char* value) : name_(name) {
        if (const char* old = std::getenv(name)) {
            had_old_ = true;
            old_ = old;
        }
        if (value) ::setenv(name, value, 1);
        else ::unsetenv(name);
    }

    ~ScopedEnvironment() {
        if (had_old_) ::setenv(name_.c_str(), old_.c_str(), 1);
        else ::unsetenv(name_.c_str());
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

private:
    std::string name_;
    std::string old_;
    bool had_old_ = false;
};

// A basis matrix held column-wise, exactly as BasisFactor::factorize takes it.
struct ColMat {
    Index m = 0;
    std::vector<Offset> col_ptr{0};
    std::vector<Index>  row_idx;
    std::vector<f64>    vals;

    // y (by row) = B x (by slot)
    std::vector<f64> apply(const std::vector<f64>& x) const {
        std::vector<f64> y(static_cast<std::size_t>(m), 0.0);
        for (Index j = 0; j < m; ++j)
            for (Offset k = col_ptr[static_cast<std::size_t>(j)];
                 k < col_ptr[static_cast<std::size_t>(j) + 1]; ++k)
                y[static_cast<std::size_t>(row_idx[static_cast<std::size_t>(k)])] +=
                    vals[static_cast<std::size_t>(k)] * x[static_cast<std::size_t>(j)];
        return y;
    }

    // d (by slot) = B' y (by row)
    std::vector<f64> apply_t(const std::vector<f64>& y) const {
        std::vector<f64> d(static_cast<std::size_t>(m), 0.0);
        for (Index j = 0; j < m; ++j) {
            f64 s = 0.0;
            for (Offset k = col_ptr[static_cast<std::size_t>(j)];
                 k < col_ptr[static_cast<std::size_t>(j) + 1]; ++k)
                s += vals[static_cast<std::size_t>(k)] *
                     y[static_cast<std::size_t>(row_idx[static_cast<std::size_t>(k)])];
            d[static_cast<std::size_t>(j)] = s;
        }
        return d;
    }

    // Replace column p. Rebuilds, which is fine for a test.
    void set_col(Index p, const std::vector<Index>& rows, const std::vector<f64>& v) {
        ColMat out;
        out.m = m;
        for (Index j = 0; j < m; ++j) {
            if (j == p) {
                for (std::size_t t = 0; t < rows.size(); ++t) {
                    out.row_idx.push_back(rows[t]);
                    out.vals.push_back(v[t]);
                }
            } else {
                for (Offset k = col_ptr[static_cast<std::size_t>(j)];
                     k < col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
                    out.row_idx.push_back(row_idx[static_cast<std::size_t>(k)]);
                    out.vals.push_back(vals[static_cast<std::size_t>(k)]);
                }
            }
            out.col_ptr.push_back(static_cast<Offset>(out.row_idx.size()));
        }
        *this = out;
    }
};

f64 max_abs_diff(const std::vector<f64>& a, const std::vector<f64>& b) {
    f64 d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
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

    ::sor::test::report(paired_a == separate_a, "ftran_pair first RHS", file, line,
                        what + " err " +
                            std::to_string(max_abs_diff(paired_a, separate_a)));
    ::sor::test::report(paired_b == separate_b, "ftran_pair second RHS", file, line,
                        what + " err " +
                            std::to_string(max_abs_diff(paired_b, separate_b)));
}
#define CHECK_FTRAN_PAIR(f, a, b, what) \
    check_ftran_pair_matches_separate((f), (a), (b), (what), __FILE__, __LINE__)

// Random sparse matrix with a guaranteed nonsingular structure: a permuted
// diagonal with a strong diagonal, plus off-diagonal clutter. Without the
// permutation the test would only ever exercise the singleton-peeling path.
ColMat random_basis(Index m, double density, std::mt19937& rng) {
    std::uniform_real_distribution<f64> val(-3.0, 3.0);
    std::uniform_real_distribution<f64> u(0.0, 1.0);

    std::vector<Index> perm(static_cast<std::size_t>(m));
    for (Index i = 0; i < m; ++i) perm[static_cast<std::size_t>(i)] = i;
    std::shuffle(perm.begin(), perm.end(), rng);

    // Build column-wise via per-column row lists so indices stay ascending.
    std::vector<std::vector<std::pair<Index, f64>>> cols(static_cast<std::size_t>(m));
    for (Index j = 0; j < m; ++j) {
        for (Index i = 0; i < m; ++i) {
            const bool diag = (perm[static_cast<std::size_t>(j)] == i);
            if (!diag && u(rng) >= density) continue;
            f64 v = val(rng);
            if (v == 0.0) v = 1.0;
            if (diag) v += (v >= 0.0 ? 4.0 : -4.0);   // dominant, so it is a valid pivot
            cols[static_cast<std::size_t>(j)].emplace_back(i, v);
        }
    }

    ColMat b;
    b.m = m;
    for (Index j = 0; j < m; ++j) {
        for (const auto& [i, v] : cols[static_cast<std::size_t>(j)]) {
            b.row_idx.push_back(i);
            b.vals.push_back(v);
        }
        b.col_ptr.push_back(static_cast<Offset>(b.row_idx.size()));
    }
    return b;
}

void check_solves(const ColMat& b, const BasisFactor& f, const char* what, f64 tol) {
    const auto m = static_cast<std::size_t>(b.m);
    std::mt19937 rng(7u);
    std::uniform_real_distribution<f64> d(-1.0, 1.0);

    std::vector<f64> rhs(m);
    for (auto& v : rhs) v = d(rng);

    // FTRAN: B (B^-1 rhs) == rhs
    std::vector<f64> z = rhs;
    f.ftran(z);
    const f64 e_f = max_abs_diff(b.apply(z), rhs);

    // BTRAN: B' (B^-T rhs) == rhs
    std::vector<f64> y = rhs;
    f.btran(y);
    const f64 e_b = max_abs_diff(b.apply_t(y), rhs);

    ::sor::test::report(e_f <= tol, "ftran residual", __FILE__, __LINE__,
                        std::string(what) + " ftran err " + std::to_string(e_f));
    ::sor::test::report(e_b <= tol, "btran residual", __FILE__, __LINE__,
                        std::string(what) + " btran err " + std::to_string(e_b));
}

// The all-logical starting basis: -I. Must factorize with zero fill, entirely
// by singleton peeling, because that is the basis every cold start begins from.
void test_identity_basis() {
    const Index m = 128;
    ColMat b;
    b.m = m;
    for (Index j = 0; j < m; ++j) {
        b.row_idx.push_back(j);
        b.vals.push_back(-1.0);
        b.col_ptr.push_back(static_cast<Offset>(b.row_idx.size()));
    }

    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));
    CHECK(f.is_valid());
    CHECK(f.stats().triangular_pivots == m);
    CHECK(f.stats().nucleus_pivots == 0);
    // nnz(L) + nnz(U) + m diagonal entries, with no fill, is exactly m.
    CHECK(f.stats().factor_nnz == m);
    check_solves(b, f, "identity", 1e-12);

    std::vector<f64> unit(static_cast<std::size_t>(m), 0.0);
    unit[17] = 1.0;
    std::vector<Index> support;
    CHECK(f.btran_with_support(unit, support));
    CHECK(support.size() == 1);
    CHECK(support[0] == 17);
    CHECK_NEAR(unit[17], -1.0, 1e-12);
}

void test_random_solves() {
    std::mt19937 rng(20260830u);
    for (const Index m : {1, 2, 5, 17, 64, 200}) {
        for (const double density : {0.0, 0.02, 0.1}) {
            const ColMat b = random_basis(m, density, rng);
            BasisFactor f;
            const bool ok = f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{});
            ::sor::test::report(ok, "factorize nonsingular", __FILE__, __LINE__,
                                "m=" + std::to_string(m));
            if (!ok) continue;
            check_solves(b, f, ("random m=" + std::to_string(m)).c_str(), 1e-8);
        }
    }
}

void test_ftran_pair_fresh_factorization_and_zero_rhs() {
    std::mt19937 rng(20260907u);
    constexpr Index m = 47;
    const ColMat b = random_basis(m, 0.08, rng);
    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));

    std::vector<f64> dense_a(static_cast<std::size_t>(m));
    std::vector<f64> sparse_b(static_cast<std::size_t>(m), 0.0);
    for (Index i = 0; i < m; ++i)
        dense_a[sz(i)] = static_cast<f64>((i * 17) % 23 - 11) / 7.0;
    sparse_b[3] = -2.5;
    sparse_b[31] = 0.75;
    CHECK_FTRAN_PAIR(f, dense_a, sparse_b, "fresh factorization");

    // Exercise both lanes with a zero RHS.  In addition to matching two
    // separate solves exactly, this catches accidental cross-lane reuse in a
    // paired triangular kernel.
    const std::vector<f64> zero(static_cast<std::size_t>(m), 0.0);
    CHECK_FTRAN_PAIR(f, zero, dense_a, "zero first RHS");
    CHECK_FTRAN_PAIR(f, sparse_b, zero, "zero second RHS");

    // The public contract explicitly permits aliasing and defines it as one
    // ordinary solve, rather than applying B^-1 twice to the same vector.
    auto alias_reference = dense_a;
    f.ftran(alias_reference);
    auto aliased = dense_a;
    f.ftran_pair(aliased, aliased);
    CHECK(aliased == alias_reference);
}

void test_ftran_pair_after_product_form_updates() {
    std::mt19937 rng(20260908u);
    constexpr Index m = 43;
    ColMat b = random_basis(m, 0.06, rng);
    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));
    std::uniform_real_distribution<f64> val(-2.0, 2.0);

    for (int step = 0; step < 16; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vals;
        const int stride = 4 + step % 5;
        for (Index i = 0; i < m; ++i) {
            if (i % stride != step % stride) continue;
            rows.push_back(i);
            vals.push_back(val(rng));
        }
        vals[0] += vals[0] >= 0.0 ? 6.0 : -6.0;

        std::vector<f64> alpha(static_cast<std::size_t>(m), 0.0);
        for (std::size_t t = 0; t < rows.size(); ++t)
            alpha[sz(rows[t])] = vals[t];
        f.ftran(alpha);

        Index p = 0;
        for (Index i = 1; i < m; ++i)
            if (std::fabs(alpha[sz(i)]) > std::fabs(alpha[sz(p)])) p = i;
        CHECK(f.update(p, alpha));
        b.set_col(p, rows, vals);

        std::vector<f64> rhs_a(static_cast<std::size_t>(m));
        std::vector<f64> rhs_b(static_cast<std::size_t>(m), 0.0);
        for (Index i = 0; i < m; ++i)
            rhs_a[sz(i)] = static_cast<f64>(((i + step) * 11) % 29 - 14) / 9.0;
        rhs_b[sz((step * 7) % m)] = 1.0 + 0.125 * step;
        rhs_b[sz((step * 13 + 5) % m)] -= 0.5;
        CHECK_FTRAN_PAIR(f, rhs_a, rhs_b,
                         "product-form update " + std::to_string(step + 1));
    }
    CHECK(f.n_product_form_etas() >= 16);
}

// The MPF / Forrest–Tomlin-style update must agree with factorizing the updated
// matrix from scratch. This is the check that catches a wrong T application
// order, which is otherwise invisible for a single update and only diverges
// after several.
void test_product_form_update() {
    std::mt19937 rng(99u);
    const Index m = 60;
    ColMat b = random_basis(m, 0.05, rng);

    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));

    std::uniform_int_distribution<int> slot(0, static_cast<int>(m) - 1);
    std::uniform_real_distribution<f64> val(-2.0, 2.0);

    for (int step = 0; step < 25; ++step) {
        // A new column with a handful of entries, one of them strong.
        std::vector<Index> rows;
        std::vector<f64> vs;
        for (Index i = 0; i < m; ++i)
            if (i % 7 == step % 7) { rows.push_back(i); vs.push_back(val(rng)); }
        if (rows.empty()) continue;
        vs[0] += 5.0;

        // alpha = B^-1 a_q under the CURRENT basis, which is what update() wants.
        std::vector<f64> alpha(static_cast<std::size_t>(m), 0.0);
        for (std::size_t t = 0; t < rows.size(); ++t)
            alpha[static_cast<std::size_t>(rows[t])] = vs[t];
        f.ftran(alpha);

        // Pick a leaving slot with a usable pivot, mirroring the ratio test.
        Index p = -1;
        f64 best = 0.0;
        for (Index i = 0; i < m; ++i)
            if (std::fabs(alpha[static_cast<std::size_t>(i)]) > best) {
                best = std::fabs(alpha[static_cast<std::size_t>(i)]);
                p = i;
            }
        CHECK(p >= 0);
        CHECK(f.update(p, alpha));

        b.set_col(p, rows, vs);
        check_solves(b, f, ("update step " + std::to_string(step)).c_str(), 1e-6);
    }
    CHECK(f.n_updates() > 0);
}

// REGRESSION. The Markowitz pivot search scans columns in count order and stops
// after a bounded number of candidates. An earlier version compacted the bucket
// it was scanning and then truncated it at the point it stopped, silently
// discarding every column it had not yet looked at. Those columns ended up in no
// bucket at all, so the search could never find them again and a perfectly
// well-conditioned basis was reported SINGULAR.
//
// Reproducing it needs columns that elimination never touches, because a fill or
// a cancellation re-files a column into its new bucket and hides the loss. A
// BLOCK-DIAGONAL matrix gives exactly that: eliminating inside one block cannot
// change any count in another. Every column and row here has four entries, so
// Phase A peels nothing and the whole matrix is nucleus, and each block is
// diagonally dominant, so the matrix is certainly nonsingular.
//
// Symptom in the solver before the fix: scsd8 reported NumericalFailure after
// 209 singular-basis repairs, grow15 needed 23165, and blend/grow15/agg3 all
// returned wrong objectives.
void test_bucket_not_truncated_by_early_exit() {
    constexpr Index kBlock = 4;
    for (const Index n_blocks : {2, 25, 120}) {
        const Index m = n_blocks * kBlock;
        ColMat b;
        b.m = m;
        for (Index j = 0; j < m; ++j) {
            const Index base = (j / kBlock) * kBlock;
            for (Index t = 0; t < kBlock; ++t) {
                const Index i = base + t;
                b.row_idx.push_back(i);
                b.vals.push_back(i == j ? 10.0 : (((i + j) % 2 == 0) ? 1.0 : -1.0));
            }
            b.col_ptr.push_back(static_cast<Offset>(b.row_idx.size()));
        }

        BasisFactor f;
        std::vector<Index> bad;
        const bool ok = f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}, &bad);
        ::sor::test::report(ok, "block-diagonal basis is nonsingular", __FILE__, __LINE__,
                            "m=" + std::to_string(m) + " singular_slots=" +
                                std::to_string(bad.size()));
        ::sor::test::report(f.stats().nucleus_pivots > 0,
                            "exercises Phase B", __FILE__, __LINE__,
                            "nucleus_pivots=" + std::to_string(f.stats().nucleus_pivots));
        if (ok) check_solves(b, f, ("blocks m=" + std::to_string(m)).c_str(), 1e-9);
    }
}

// A duplicated column is singular. The requirement is a report, not a throw and
// not a wrong answer: branch-and-bound is allowed to hand over a singular
// warm-start basis and the driver has to repair it.
void test_singular_reported() {
    const Index m = 6;
    ColMat b;
    b.m = m;
    for (Index j = 0; j < m; ++j) {
        const Index r = (j == 3) ? 2 : j;    // column 3 duplicates column 2
        b.row_idx.push_back(r);
        b.vals.push_back(2.0);
        b.col_ptr.push_back(static_cast<Offset>(b.row_idx.size()));
    }

    BasisFactor f;
    std::vector<Index> bad_slots, vacant;
    const bool ok = f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{},
                                &bad_slots, &vacant);
    CHECK(!ok);
    CHECK(!f.is_valid());
    CHECK(bad_slots.size() == 1);
    CHECK(vacant.size() == 1);
    CHECK(bad_slots.size() == vacant.size());
    // Row 5 is covered, so the uncovered row must be one of the two sharing row 2.
    CHECK(!vacant.empty() && vacant.front() >= 0 && vacant.front() < m);

    // Solves must still run rather than read past the end of a partial factor.
    std::vector<f64> v(static_cast<std::size_t>(m), 1.0);
    f.ftran(v);
    f.btran(v);
    CHECK(f.dimension() == m);
}

// An all-zero column has no pivot at any threshold. It must be reported, not
// turned into a huge multiplier that quietly poisons every later solve.
void test_zero_column() {
    const Index m = 4;
    ColMat b;
    b.m = m;
    for (Index j = 0; j < m; ++j) {
        if (j != 1) { b.row_idx.push_back(j); b.vals.push_back(1.0); }
        b.col_ptr.push_back(static_cast<Offset>(b.row_idx.size()));
    }
    BasisFactor f;
    std::vector<Index> bad_slots;
    CHECK(!f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}, &bad_slots));
    CHECK(bad_slots.size() == 1 && bad_slots.front() == 1);
}

// update() must refuse a pivot it cannot divide by, and must leave the
// factorization untouched when it refuses.
void test_update_rejects_tiny_pivot() {
    const Index m = 3;
    ColMat b;
    b.m = m;
    for (Index j = 0; j < m; ++j) {
        b.row_idx.push_back(j);
        b.vals.push_back(1.0);
        b.col_ptr.push_back(static_cast<Offset>(b.row_idx.size()));
    }
    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));

    std::vector<f64> alpha{1.0, 0.0, 1.0};
    CHECK(!f.update(1, alpha));          // alpha[1] == 0
    CHECK(f.n_updates() == 0);
    check_solves(b, f, "after refused update", 1e-12);
}

// Work-based refactor trigger (cuOpt PR #1043, 2026): work_since_factor()
// accumulates across ftran/btran calls, resets on factorize(), and
// needs_refactor()'s work_ratio_max fires once it crosses factor_nnz times
// the ratio -- and never fires when the ratio is 0 (disabled), regardless of
// how much work has accumulated.
void test_work_based_refactor_trigger() {
    std::mt19937 rng(777u);
    const Index m = 40;
    ColMat b = random_basis(m, 0.1, rng);
    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));
    CHECK(f.work_since_factor() == 0);

    std::vector<f64> rhs(static_cast<std::size_t>(m), 1.0);
    for (int i = 0; i < 5; ++i) {
        std::vector<f64> z = rhs;
        f.ftran(z);
    }
    const Offset work_after = f.work_since_factor();
    CHECK(work_after > 0);
    CHECK(!f.needs_refactor(0, 0.0, 0, 0.0));  // disabled: never fires

    const f64 tiny_ratio = 1e-9;  // guaranteed below by work_after > 0
    CHECK(f.needs_refactor(0, 0.0, 0, tiny_ratio));

    const f64 huge_ratio = 1e12;  // guaranteed not reached
    CHECK(!f.needs_refactor(0, 0.0, 0, huge_ratio));

    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));
    CHECK(f.work_since_factor() == 0);  // reset by factorize()
}

void test_empty_basis() {
    BasisFactor f;
    std::vector<Offset> cp{0};
    CHECK(f.factorize(0, cp, {}, {}, LuOptions{}));
    CHECK(f.is_valid());
    std::vector<f64> none;
    f.ftran(none);
    f.btran(none);
}

void test_hypersparse_unit_rhs() {
    const Index m = 80;
    std::mt19937 rng(42u);
    ColMat b = random_basis(m, 0.04, rng);
    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));

    // Unit RHS: hypersparse path must match dense.
    for (Index unit = 0; unit < m; unit += 17) {
        std::vector<f64> rhs(static_cast<std::size_t>(m), 0.0);
        rhs[static_cast<std::size_t>(unit)] = 1.0;
        std::vector<f64> z = rhs;
        f.ftran(z);
        const f64 e_f = max_abs_diff(b.apply(z), rhs);
        ::sor::test::report(e_f <= 1e-8, "hypersparse ftran unit", __FILE__, __LINE__,
                            "unit=" + std::to_string(unit));

        std::vector<f64> rho(static_cast<std::size_t>(m), 0.0);
        rho[static_cast<std::size_t>(unit)] = 1.0;
        std::vector<f64> w = rho;
        f.btran(w);
        const f64 e_b = max_abs_diff(b.apply_t(w), rho);
        ::sor::test::report(e_b <= 1e-8, "hypersparse btran unit", __FILE__, __LINE__,
                            "unit=" + std::to_string(unit));
    }

    // A few nonzeros (still sparse): mixed-density seeds.
    for (int trial = 0; trial < 8; ++trial) {
        std::vector<f64> rhs(static_cast<std::size_t>(m), 0.0);
        for (Index t = 0; t < 5; ++t)
            rhs[static_cast<std::size_t>((trial * 13 + t * 7) % m)] =
                (t % 2 ? 1.0 : -2.0);
        std::vector<f64> z = rhs;
        f.ftran(z);
        const f64 e_f = max_abs_diff(b.apply(z), rhs);
        ::sor::test::report(e_f <= 1e-8, "hypersparse ftran few-nnz", __FILE__, __LINE__,
                            "trial=" + std::to_string(trial));

        std::vector<f64> w = rhs;
        f.btran(w);
        const f64 e_b = max_abs_diff(b.apply_t(w), rhs);
        ::sor::test::report(e_b <= 1e-8, "hypersparse btran few-nnz", __FILE__, __LINE__,
                            "trial=" + std::to_string(trial));
    }

    // Through an update chain: each update consumes the FTRAN result of the
    // entering column (the product-form contract), so the flow mirrors the
    // engine exactly -- build a column, FTRAN it, update the slot with the
    // result, then verify every solve against the matrix whose column was
    // replaced.
    ColMat b2 = b;
    for (Index upd = 0; upd < 6; ++upd) {
        std::vector<f64> aq(static_cast<std::size_t>(m), 0.0);
        aq[static_cast<std::size_t>((upd * 11 + 3) % m)] = 2.0;
        aq[static_cast<std::size_t>((upd * 29 + 1) % m)] = 0.5;
        aq[static_cast<std::size_t>((upd * 7 + 5) % m)] = -1.5;
        std::vector<f64> alpha = aq;
        f.ftran(alpha);                      // alpha := B^-1 a_q

        // Leaving slot: the largest |alpha|, mirroring the ratio test.
        Index p = -1;
        f64 best = 0.0;
        for (Index i = 0; i < m; ++i)
            if (std::fabs(alpha[static_cast<std::size_t>(i)]) > best) {
                best = std::fabs(alpha[static_cast<std::size_t>(i)]);
                p = i;
            }
        CHECK(p >= 0);
        if (!f.update(p, alpha)) continue;   // unstable update: skip, not fail

        std::vector<Index> rows;
        std::vector<f64> vals;
        for (Index i = 0; i < m; ++i)
            if (aq[static_cast<std::size_t>(i)] != 0.0) {
                rows.push_back(i);
                vals.push_back(aq[static_cast<std::size_t>(i)]);
            }
        b2.set_col(p, rows, vals);

        for (Index unit = 0; unit < m; unit += 29) {
            std::vector<f64> rhs(static_cast<std::size_t>(m), 0.0);
            rhs[static_cast<std::size_t>(unit)] = 1.0;
            std::vector<f64> z = rhs;
            f.ftran(z);
            const f64 e_f = max_abs_diff(b2.apply(z), rhs);
            ::sor::test::report(e_f <= 1e-7, "hypersparse ftran etas", __FILE__,
                                __LINE__,
                                "upd=" + std::to_string(upd) +
                                " unit=" + std::to_string(unit));

            std::vector<f64> w = rhs;
            std::vector<Index> support;
            const bool sparse_support = f.btran_with_support(w, support);
            const f64 e_b = max_abs_diff(b2.apply_t(w), rhs);
            ::sor::test::report(e_b <= 1e-7, "hypersparse btran etas", __FILE__,
                                __LINE__,
                                "upd=" + std::to_string(upd) +
                                " unit=" + std::to_string(unit));
            if (sparse_support) {
                CHECK(std::is_sorted(support.begin(), support.end()));
                std::vector<char> present(static_cast<std::size_t>(m), 0);
                for (const Index i : support) present[static_cast<std::size_t>(i)] = 1;
                for (Index i = 0; i < m; ++i)
                    CHECK((w[static_cast<std::size_t>(i)] != 0.0) ==
                          static_cast<bool>(present[static_cast<std::size_t>(i)]));
            }
        }
    }
}

void test_moderately_dense_reach_keeps_exact_order_and_support() {
    // Twenty of 128 identity positions stay below the sparse solve's 25%
    // admission gate but above the adaptive ordering path's 12.5% cutoff.
    // This therefore covers the linear mark-scan ordering branch in all four
    // triangular orientations without relying on timing to prove it ran.
    constexpr Index m = 128;
    ColMat identity;
    identity.m = m;
    for (Index j = 0; j < m; ++j) {
        identity.row_idx.push_back(j);
        identity.vals.push_back(1.0);
        identity.col_ptr.push_back(static_cast<Offset>(identity.row_idx.size()));
    }
    std::vector<f64> rhs(sz(m), 0.0);
    std::vector<Index> expected;
    for (Index k = 0; k < 20; ++k) {
        const Index i = 3 + 5 * k;
        rhs[sz(i)] = (k & 1) ? -0.5 : 2.0;
        expected.push_back(i);
    }

    struct SparseSolves {
        std::vector<f64> ftran;
        std::vector<Index> ftran_support;
        std::vector<f64> btran;
        std::vector<Index> btran_support;
    };
    const auto solve_both = [&](const char* force_comparison) {
        // The flag is sampled by factorize(), so each factor owns one of the
        // two implementations even after this scope restores the process
        // environment.
        ScopedEnvironment flag("SOR_LU_COMPARISON_SORT", force_comparison);
        BasisFactor f;
        CHECK(f.factorize(m, identity.col_ptr, identity.row_idx, identity.vals,
                          LuOptions{}));
        SparseSolves out;
        out.ftran = rhs;
        CHECK(f.ftran_with_support(out.ftran, out.ftran_support));
        out.btran = rhs;
        CHECK(f.btran_with_support(out.btran, out.btran_support));
        return out;
    };

    const auto linear_scan = solve_both(nullptr);
    const auto comparison_sort = solve_both("1");
    CHECK(linear_scan.ftran == rhs);
    CHECK(linear_scan.btran == rhs);
    CHECK(linear_scan.ftran_support == expected);
    CHECK(linear_scan.btran_support == expected);
    CHECK(linear_scan.ftran == comparison_sort.ftran);
    CHECK(linear_scan.btran == comparison_sort.btran);
    CHECK(linear_scan.ftran_support == comparison_sort.ftran_support);
    CHECK(linear_scan.btran_support == comparison_sort.btran_support);
}

// Seeded BTRAN is a partial-write API: production reuses rho, clearing only
// the previous output support and the previous unit-input slot.  That contract
// is easy to satisfy for one call and easy to break on the next one, especially
// because product-form etas can write pivot slots that are neither the input
// seed nor part of the final output.  Exercise the exact reuse pattern across
// all eta implementations: no etas, the <8 plain reverse scan, and the >=8
// firing set.  A fresh full BTRAN is the independent oracle for every call.
void test_seeded_btran_reuse_across_eta_paths() {
    constexpr Index m = 128;  // sparse kernels are enabled only at m >= 64
    ColMat b;
    b.m = m;
    for (Index j = 0; j < m; ++j) {
        b.row_idx.push_back(j);
        b.vals.push_back(1.0);
        b.col_ptr.push_back(static_cast<Offset>(b.row_idx.size()));
    }

    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));

    std::vector<f64> reused(static_cast<std::size_t>(m), 0.0);
    std::vector<Index> previous_support;
    Index previous_seed = -1;
    bool previous_was_sparse = false;
    int sparse_calls = 0;

    const auto check_one = [&](Index seed, Index extra_zero,
                               const std::string& where) {
        // This is the engine's reset discipline.  A dense fallback overwrote
        // every entry, so it requires a full clear; a sparse call promises
        // that these two small sets contain every possibly-live old value.
        if (previous_was_sparse) {
            for (const Index i : previous_support) reused[sz(i)] = 0.0;
            if (previous_seed >= 0 && previous_seed != seed)
                reused[sz(previous_seed)] = 0.0;
        } else {
            std::fill(reused.begin(), reused.end(), 0.0);
        }
        reused[sz(seed)] = 1.0;

        std::vector<f64> reference(static_cast<std::size_t>(m), 0.0);
        reference[sz(seed)] = 1.0;
        f.btran(reference);

        // Duplicates and declared zeros are legal over-approximations and
        // exercise the seeded permutation's de-duplication path.
        const std::vector<Index> declared{seed, seed, extra_zero};
        std::vector<Index> support;
        const bool sparse =
            f.btran_seeded_with_support(reused, declared, support);
        sparse_calls += sparse ? 1 : 0;

        ::sor::test::report(max_abs_diff(reused, reference) <= 1e-13,
                            "seeded BTRAN equals full BTRAN", __FILE__,
                            __LINE__, where);
        if (sparse) {
            CHECK(std::is_sorted(support.begin(), support.end()));
            CHECK(std::adjacent_find(support.begin(), support.end()) ==
                  support.end());
            std::vector<char> present(static_cast<std::size_t>(m), 0);
            for (const Index i : support) present[sz(i)] = 1;
            for (Index i = 0; i < m; ++i)
                CHECK((reused[sz(i)] != 0.0) ==
                      static_cast<bool>(present[sz(i)]));
        }

        previous_support = support;
        previous_seed = seed;
        previous_was_sparse = sparse;
    };

    // Empty eta file, including immediate reuse of the same input slot.
    check_one(5, 97, "cold seed 5");
    check_one(5, 61, "cold repeated seed 5");
    check_one(41, 11, "cold changed seed 41");

    for (Index step = 0; step < 12; ++step) {
        const Index p = (7 + 11 * step) % m;
        Index q = (3 + 17 * step) % m;
        if (q == p) q = (q + 1) % m;

        // A sparse, well-conditioned elementary update.  q is read by the
        // eta and p is written, so a unit seed at q forces the eta path to do
        // real work rather than merely traverse an empty firing set.
        std::vector<f64> alpha(static_cast<std::size_t>(m), 0.0);
        alpha[sz(p)] = 1.0 + 0.01 * static_cast<f64>(step + 1);
        alpha[sz(q)] = (step & 1) ? -0.125 : 0.125;
        CHECK(f.update(p, alpha));
        CHECK(f.n_updates() == step + 1);

        const std::string stage = "after eta " + std::to_string(step + 1);
        check_one(q, (q + 53) % m, stage + " firing seed");
        check_one(q, (q + 79) % m, stage + " repeated seed");
        check_one((q + 31) % m, p, stage + " unrelated seed");
    }

    // The chosen identity factors and sparse etas must actually keep us in
    // the partial-write path; otherwise the stale-state assertions above
    // would only test the dense fallback.
    CHECK(sparse_calls == 39);
}

// Forrest-Tomlin update, exactly the differential methodology as
// test_product_form_update() above but through update_ft()'s dense bump
// re-triangularization instead of an eta -- dense reconstruction, checked
// after EVERY update. This is the check that caught the earlier FT attempt's
// bug (see lu.hpp's file header): an in-place elimination without proper
// cyclic permutation is invisible for a single update and only diverges
// after several, so a "looks right once" spot check would have passed it.
void test_ft_update() {
    std::mt19937 rng(1234u);
    const Index m = 60;
    ColMat b = random_basis(m, 0.05, rng);

    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));

    std::uniform_real_distribution<f64> val(-2.0, 2.0);
    int successes = 0;

    for (int step = 0; step < 40; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vs;
        for (Index i = 0; i < m; ++i)
            if (i % 7 == step % 7) { rows.push_back(i); vs.push_back(val(rng)); }
        if (rows.empty()) continue;
        vs[0] += 5.0;

        std::vector<f64> alpha(static_cast<std::size_t>(m), 0.0);
        for (std::size_t t = 0; t < rows.size(); ++t)
            alpha[static_cast<std::size_t>(rows[t])] = vs[t];
        f.ftran(alpha);

        Index p = -1;
        f64 best = 0.0;
        for (Index i = 0; i < m; ++i)
            if (std::fabs(alpha[static_cast<std::size_t>(i)]) > best) {
                best = std::fabs(alpha[static_cast<std::size_t>(i)]);
                p = i;
            }
        CHECK(p >= 0);
        // A fixed bump column order (no column pivoting) can rarely miss a
        // pivot ordering that a full refactor would find even though the
        // resulting matrix is genuinely nonsingular (alpha[p] != 0
        // guarantees that). That's an accepted, documented limitation, not
        // a bug -- test_ft_rejects_tiny_pivot_untouched() below checks the
        // factorization survives it. Just skip and keep going here.
        if (!f.update_ft(p, alpha, LuOptions{})) continue;

        ++successes;
        b.set_col(p, rows, vs);
        check_solves(b, f, ("ft update step " + std::to_string(step)).c_str(), 1e-6);
    }
    ::sor::test::report(successes > 20, "ft update makes progress", __FILE__, __LINE__,
                        "successes=" + std::to_string(successes));
}

// Differential: the SAME sequence of column replacements through update()
// (product form) and update_ft() must produce basis factors that solve
// identically, even though their internal representations (a growing eta
// file vs. a re-triangularized U/L) are completely different.
void test_ft_matches_product_form() {
    std::mt19937 rng(5678u);
    const Index m = 45;
    ColMat b0 = random_basis(m, 0.06, rng);

    BasisFactor fp, ff;  // product-form, forrest-tomlin
    CHECK(fp.factorize(m, b0.col_ptr, b0.row_idx, b0.vals, LuOptions{}));
    CHECK(ff.factorize(m, b0.col_ptr, b0.row_idx, b0.vals, LuOptions{}));

    ColMat b = b0;
    std::uniform_real_distribution<f64> val(-2.0, 2.0);
    for (int step = 0; step < 30; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vs;
        for (Index i = 0; i < m; ++i)
            if (i % 5 == step % 5) { rows.push_back(i); vs.push_back(val(rng)); }
        if (rows.empty()) continue;
        vs[0] += 4.0;

        std::vector<f64> aq(static_cast<std::size_t>(m), 0.0);
        for (std::size_t t = 0; t < rows.size(); ++t)
            aq[static_cast<std::size_t>(rows[t])] = vs[t];

        // Both factorizations currently represent the SAME matrix b, so
        // FTRAN of the same incoming column against either picks the same
        // leaving slot.
        std::vector<f64> alpha = aq;
        fp.ftran(alpha);
        Index p = -1;
        f64 best = 0.0;
        for (Index i = 0; i < m; ++i)
            if (std::fabs(alpha[static_cast<std::size_t>(i)]) > best) {
                best = std::fabs(alpha[static_cast<std::size_t>(i)]);
                p = i;
            }
        CHECK(p >= 0);

        std::vector<f64> alpha_ff = aq;
        ff.ftran(alpha_ff);

        const bool ok_p = fp.update(p, alpha);
        const bool ok_f = ff.update_ft(p, alpha_ff, LuOptions{});
        if (!ok_p || !ok_f) continue;  // nothing to compare if either declines

        b.set_col(p, rows, vs);
        check_solves(b, fp, ("diff product-form step " + std::to_string(step)).c_str(), 1e-6);
        check_solves(b, ff, ("diff forrest-tomlin step " + std::to_string(step)).c_str(), 1e-6);
    }
}

// Adversarial bump widths: leave the FIRST slot (bump = the whole matrix,
// the largest possible bump) and the LAST slot (bump width 1, the smallest).
void test_ft_bump_extremes() {
    std::mt19937 rng(999u);
    for (const Index m : {5, 40}) {
        ColMat b = random_basis(m, 0.1, rng);
        BasisFactor f;
        CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));

        std::uniform_real_distribution<f64> val(-2.0, 2.0);
        const std::vector<Index> leaving_slots = {0, m - 1};
        for (const Index p : leaving_slots) {
            std::vector<Index> rows;
            std::vector<f64> vs;
            for (Index i = 0; i < m; ++i) { rows.push_back(i); vs.push_back(val(rng)); }
            vs[static_cast<std::size_t>(p)] += 5.0;

            std::vector<f64> alpha(static_cast<std::size_t>(m), 0.0);
            for (std::size_t t = 0; t < rows.size(); ++t)
                alpha[static_cast<std::size_t>(rows[t])] = vs[t];
            f.ftran(alpha);
            if (std::fabs(alpha[static_cast<std::size_t>(p)]) < 1e-8) continue;  // unlucky draw

            if (!f.update_ft(p, alpha, LuOptions{})) continue;
            b.set_col(p, rows, vs);
            check_solves(b, f, ("bump extreme p=" + std::to_string(p)).c_str(), 1e-6);
            CHECK(f.current_bump_width() >= 1);
        }
    }
}

// Failure must leave the factorization exactly as usable as before: an
// intentionally zero alpha[p] is refused up front (same contract as
// update()), and the OLD matrix must still solve correctly afterwards.
void test_ft_rejects_tiny_pivot_untouched() {
    std::mt19937 rng(2024u);
    const Index m = 5;
    ColMat b = random_basis(m, 0.2, rng);
    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));

    std::vector<f64> alpha(static_cast<std::size_t>(m), 0.0);
    alpha[2] = 1.0;  // leaving slot 0, but alpha[0] == 0
    CHECK(!f.update_ft(0, alpha, LuOptions{}));
    check_solves(b, f, "after refused ft update", 1e-9);
}

// Collective FT:
// collapse_pending_into_ft() folds a BATCH of pending product-form etas into
// L/U via sequential update_ft() calls. Its entire contract is that this is
// a pure REPRESENTATION change -- ftran()/btran() must return IDENTICAL
// results before and after, for every unit vector, even though internally
// the eta file went from N pending entries to zero. That transparency, not
// just "still solves the current matrix", is what this checks: capture
// ftran(e_i) for every i against the eta-file representation, collapse, and
// compare against the SAME e_i again post-collapse.
void test_collective_ft_transparent() {
    std::mt19937 rng(42u);
    const Index m = 30;
    ColMat b = random_basis(m, 0.08, rng);
    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));

    std::uniform_real_distribution<f64> val(-2.0, 2.0);
    int pending_updates = 0;
    for (int step = 0; step < 8; ++step) {
        std::vector<Index> rows;
        std::vector<f64> vs;
        for (Index i = 0; i < m; ++i)
            if (i % 6 == step % 6) { rows.push_back(i); vs.push_back(val(rng)); }
        if (rows.empty()) continue;
        vs[0] += 4.0;

        std::vector<f64> alpha(sz(m), 0.0);
        for (std::size_t t = 0; t < rows.size(); ++t) alpha[sz(rows[t])] = vs[t];
        f.ftran(alpha);
        Index p = -1; f64 best = 0.0;
        for (Index i = 0; i < m; ++i)
            if (std::fabs(alpha[sz(i)]) > best) { best = std::fabs(alpha[sz(i)]); p = i; }
        CHECK(p >= 0);
        if (!f.update(p, alpha)) continue;  // product-form: pending eta added
        b.set_col(p, rows, vs);
        ++pending_updates;
    }
    ::sor::test::report(pending_updates >= 3, "batch has several pending etas",
                        __FILE__, __LINE__, "count=" + std::to_string(pending_updates));
    CHECK(f.n_updates() == pending_updates);

    // Snapshot ftran(e_i) for every i BEFORE collapsing.
    std::vector<std::vector<f64>> before(sz(m));
    for (Index i = 0; i < m; ++i) {
        std::vector<f64> e(sz(m), 0.0);
        e[sz(i)] = 1.0;
        f.ftran(e);
        before[sz(i)] = e;
    }

    const bool collapsed = f.collapse_pending_into_ft(LuOptions{});
    ::sor::test::report(collapsed, "collective collapse succeeded", __FILE__, __LINE__,
                        "n_updates_after=" + std::to_string(f.n_updates()));
    // What collapse empties is the PRODUCT-FORM file; it converts those etas
    // into Forrest-Tomlin row etas, which n_updates() also counts so that the
    // refactorization triggers can see them.
    if (collapsed) CHECK(f.n_product_form_etas() == 0);

    for (Index i = 0; i < m; ++i) {
        std::vector<f64> e(sz(m), 0.0);
        e[sz(i)] = 1.0;
        f.ftran(e);
        for (Index k = 0; k < m; ++k)
            CHECK_NEAR(e[sz(k)], before[sz(i)][sz(k)], 1e-8);
    }
    // And the collapsed factorization must still solve the ACTUAL current
    // matrix correctly, not just agree with its own pre-collapse self.
    check_solves(b, f, "after collective collapse", 1e-6);
}

// The same transparency check, but from a cold factorize() with NO pending
// etas at all -- collapse_pending_into_ft() must be a correct no-op rather
// than, say, misreading an empty eta file as one entry.
void test_collective_ft_noop_when_nothing_pending() {
    std::mt19937 rng(7u);
    const Index m = 10;
    ColMat b = random_basis(m, 0.2, rng);
    BasisFactor f;
    CHECK(f.factorize(m, b.col_ptr, b.row_idx, b.vals, LuOptions{}));
    CHECK(f.n_updates() == 0);
    CHECK(f.collapse_pending_into_ft(LuOptions{}));
    CHECK(f.n_updates() == 0);
    check_solves(b, f, "collapse with nothing pending", 1e-9);
}

}  // namespace

int main() {
    test_identity_basis();
    test_random_solves();
    test_ftran_pair_fresh_factorization_and_zero_rhs();
    test_ftran_pair_after_product_form_updates();
    test_product_form_update();
    test_hypersparse_unit_rhs();
    test_moderately_dense_reach_keeps_exact_order_and_support();
    test_seeded_btran_reuse_across_eta_paths();
    test_bucket_not_truncated_by_early_exit();
    test_singular_reported();
    test_zero_column();
    test_update_rejects_tiny_pivot();
    test_empty_basis();
    test_ft_update();
    test_ft_matches_product_form();
    test_ft_bump_extremes();
    test_ft_rejects_tiny_pivot_untouched();
    test_work_based_refactor_trigger();
    test_collective_ft_transparent();
    test_collective_ft_noop_when_nothing_pending();
    return sor::test::finish("test_lu");
}
