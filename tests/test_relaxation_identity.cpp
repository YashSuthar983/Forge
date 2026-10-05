// Relaxation identity (search/relaxation_identity.hpp): content, not proxies.
#include "sor/search/relaxation_identity.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

using sor::model::LpProblem;
using sor::search::relaxation_identity;

namespace {
LpProblem base() {
    LpProblem lp;
    lp.A = sor::sparse::from_triplets(2, 3, {0, 0, 1, 1}, {0, 1, 1, 2}, {1.0, 2.0, 3.0, 4.0});
    lp.c = {1.0, 1.0, 1.0};
    lp.col_lo = {0, 0, 0};
    lp.col_hi = {1, 1, 1};
    lp.row_lo = {-sor::model::kInf, -sor::model::kInf};
    lp.row_hi = {3.0, 5.0};
    lp.is_integer = {true, true, true};
    return lp;
}
}  // namespace

int main() {
    const auto a = base();
    const auto id = relaxation_identity(a, a.col_lo, a.col_hi);
    CHECK(id == relaxation_identity(base(), a.col_lo, a.col_hi));   // same content, same identity
    // Same row count, same box, same objective value scale -- a different cut system.
    auto b = base();
    b.A = sor::sparse::from_triplets(2, 3, {0, 0, 1, 1}, {0, 1, 1, 2}, {1.0, 2.0, 3.0, 5.0});
    CHECK(id != relaxation_identity(b, b.col_lo, b.col_hi));
    // Different row side only.
    auto c = base();
    c.row_hi[1] = 4.0;
    CHECK(id != relaxation_identity(c, c.col_lo, c.col_hi));
    // Different column box.
    auto lo = a.col_lo, hi = a.col_hi;
    hi[2] = 0.0;
    CHECK(id != relaxation_identity(a, lo, hi));
    // Different objective.
    auto d = base();
    d.c[1] = 2.0;
    CHECK(id != relaxation_identity(d, d.col_lo, d.col_hi));
    // Different sense.
    auto e = base();
    e.maximize = true;
    CHECK(id != relaxation_identity(e, e.col_lo, e.col_hi));
    // An extra row changes the row count AND the content.
    auto f = base();
    f.A = sor::sparse::from_triplets(3, 3, {0, 0, 1, 1, 2}, {0, 1, 1, 2, 0}, {1.0, 2.0, 3.0, 4.0, 1.0});
    f.row_lo.push_back(-sor::model::kInf);
    f.row_hi.push_back(1.0);
    CHECK(id != relaxation_identity(f, f.col_lo, f.col_hi));
    return sor::test::finish("test_relaxation_identity");
}
