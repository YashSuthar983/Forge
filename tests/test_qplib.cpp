// QPLIB reader + binary quadratic engine.
//
// The objective assertion here is an EXTERNAL-ORACLE test, and that is the
// point of it. An earlier version of this engine was self-consistent -- the
// solver and an independent recomputation agreed to 1e-9 -- and still wrong,
// because both shared the same misreading of QPLIB's objective convention.
// Only comparing against QPLIB's own published solution point caught it.
// Keep this assertion pinned to the published number.
#include "sor/io/qplib.hpp"
#include "sor/search/binquad.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace sor;

namespace {

std::string data_path(const char* name) {
    return std::string(SOR_TEST_DATA_DIR) + "/qplib/" + name;
}

// Recompute straight from the raw triplets, sharing no code with the engine.
double raw_objective(const io::QplibInstance& q,
                     const std::vector<std::uint8_t>& x) {
    double o = q.f_const;
    for (std::size_t t = 0; t < q.h_val.size(); ++t) {
        // 0.5 x'Hx with H stored as an upper triangle used AS-IS: each stored
        // entry contributes once, halved. Not mirrored.
        o += 0.5 * q.h_val[t] * x[q.h_row[t] - 1] * x[q.h_col[t] - 1];
    }
    for (int j = 0; j < q.n; ++j) o += q.g[j] * x[j];
    return o;
}

double raw_max_violation(const io::QplibInstance& q,
                         const std::vector<std::uint8_t>& x) {
    if (q.m == 0) return 0.0;
    std::vector<double> r(static_cast<std::size_t>(q.m), 0.0);
    for (std::size_t t = 0; t < q.a_val.size(); ++t)
        r[q.a_row[t] - 1] += q.a_val[t] * x[q.a_col[t] - 1];
    double worst = 0.0;
    for (int i = 0; i < q.m; ++i) {
        worst = std::max(worst, q.c_lo[i] - r[i]);
        worst = std::max(worst, r[i] - q.c_hi[i]);
    }
    return std::max(0.0, worst);
}

}  // namespace

int main() {
    // ---- reader: a constrained binary instance --------------------------
    {
        io::QplibReadReport rep;
        const auto q = io::read_qplib_file(data_path("QPLIB_3834.qplib"), rep);
        CHECK(q.name == "QPLIB_3834");
        CHECK(q.classification[0] == 'Q' && q.classification[1] == 'B' &&
              q.classification[2] == 'L');
        CHECK(!q.maximize);
        CHECK(q.n == 50);
        CHECK(q.m == 1);
        CHECK(rep.n_h_entries == 1225);
        CHECK(rep.n_a_entries == 50);
        CHECK(rep.h_has_off_diagonal);
        // Binary classification implies bounds and types; they are not in the file.
        CHECK(q.x_lo.size() == 50 && q.x_hi.size() == 50);
        CHECK(q.x_lo[0] == 0.0 && q.x_hi[0] == 1.0);
        CHECK(q.var_type[0] == io::QplibVarType::Binary);
        CHECK(q.is_discrete());
        // The single row is the cardinality constraint sum(x) == 10.
        CHECK(q.c_lo[0] == 10.0 && q.c_hi[0] == 10.0);
    }

    // ---- reader: bound-constrained, where m and A are ABSENT -------------
    // Reading an unconditional m here would consume nnz(H) and desync
    // everything after it. This case is why that bug is not possible now.
    {
        io::QplibReadReport rep;
        const auto q = io::read_qplib_file(data_path("QPLIB_3565.qplib"), rep);
        CHECK(q.classification[2] == 'B');
        CHECK(q.m == 0);
        CHECK(rep.n_a_entries == 0);
        CHECK(q.maximize);
        CHECK(q.n == 276);
        CHECK(rep.n_h_entries == 528);
    }

    // ---- engine: matches QPLIB's own published solution point ------------
    {
        io::QplibReadReport rep;
        const auto q = io::read_qplib_file(data_path("QPLIB_3834.qplib"), rep);
        search::BinQuadOptions o;
        o.time_limit_s = 5.0;
        o.seed = 42;
        search::BinQuadDiagnostics d;
        const auto res = search::solve_binquad(q, o, d);

        CHECK(res.feasible);
        CHECK(res.violation <= 1e-9);
        CHECK(raw_max_violation(q, res.x) <= 1e-9);

        // Exactly 10 variables set, per the cardinality row.
        int ones = 0;
        for (auto v : res.x) ones += v ? 1 : 0;
        CHECK(ones == 10);

        // The engine's own number must match a recomputation that shares no
        // code with it.
        CHECK(std::fabs(raw_objective(q, res.x) - res.objective) <=
              1e-9 * std::max(1.0, std::fabs(res.objective)));

        // And it must match QPLIB's published solution point value.
        constexpr double kPublished = 3760.715066;
        CHECK(std::fabs(res.objective - kPublished) <= 1e-4);
    }

    // ---- engine: refuses a non-binary instance ---------------------------
    {
        io::QplibReadReport rep;
        auto q = io::read_qplib_file(data_path("QPLIB_3834.qplib"), rep);
        q.var_type[0] = io::QplibVarType::Integer;
        search::BinQuadOptions o;
        search::BinQuadDiagnostics d;
        bool threw = false;
        try {
            (void)search::solve_binquad(q, o, d);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }

    return sor::test::finish("test_qplib");
}
