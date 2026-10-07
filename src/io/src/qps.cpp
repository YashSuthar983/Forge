#include "sor/io/qps.hpp"

#include "sor/model/dyadic.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace sor::io {

QpsProblem read_qps_file(const std::string& path, QpsReadReport& rep,
                         const MpsReadOptions& opt) {
    // One pass of the MPS reader reads the linear part and the quadratic
    // section together, gzip and fixed format included. (A second plain
    // ifstream scan used to read the quadratic section: on a .qps.gz it found
    // nothing and the QP was silently solved as an LP.)
    MpsReadReport mrep;
    QpsProblem qp;
    QuadraticTerms terms;
    MpsReadOptions linear_opt = opt;
    linear_opt.quadratic = &terms;
    qp.linear = read_mps_file_auto(path, mrep, linear_opt);
    static_cast<MpsReadReport&>(rep) = mrep;
    const auto n = qp.linear.n_cols();
    qp.q_diag.assign(static_cast<std::size_t>(n), 0.0);

    // Collect each (i, j) pair's entries; key by the unordered pair.
    struct Pair {
        model::DyadicSum lower, upper;  // entries written as (i>=j) / (i<j)
        std::size_t n_lower = 0, n_upper = 0, line = 0;
    };
    std::map<std::pair<core::Index, core::Index>, Pair> pairs;
    for (std::size_t k = 0; k < terms.vals.size(); ++k) {
        const core::Index i = terms.rows[k], j = terms.cols[k];
        auto& e = pairs[{std::max(i, j), std::min(i, j)}];
        if (e.n_lower + e.n_upper == 0) e.line = terms.lines[k];
        if (i >= j) { e.lower.add(terms.vals[k]); ++e.n_lower; }
        else        { e.upper.add(terms.vals[k]); ++e.n_upper; }
    }
    const auto where = [&](const Pair& e, core::Index i, core::Index j) {
        return " (line " + std::to_string(e.line) + ", columns '" +
               qp.linear.col_names[static_cast<std::size_t>(i)] + "' and '" +
               qp.linear.col_names[static_cast<std::size_t>(j)] + "')";
    };
    std::size_t repeated = 0;
    std::vector<core::Index> rows, cols;
    std::vector<core::f64> values;
    for (const auto& [key, e] : pairs) {
        const auto [i, j] = key;
        core::f64 value = 0.0;
        if (i == j) {
            value = e.lower.nearest();
            repeated += e.n_lower - 1;
        } else if (terms.full_matrix) {
            // QMATRIX: both triangles, and they must agree.
            if (e.n_lower == 0 || e.n_upper == 0 || e.lower.nearest() != e.upper.nearest())
                throw std::runtime_error(
                    "QMATRIX is not symmetric: the (i,j) and (j,i) entries differ" +
                    where(e, i, j));
            value = e.lower.nearest();
            repeated += e.n_lower - 1 + e.n_upper - 1;
        } else {
            // QUADOBJ: one triangle. Both orientations means the full matrix
            // was written in the wrong section; mirroring it would double
            // every off-diagonal, keeping one would guess.
            if (e.n_lower > 0 && e.n_upper > 0)
                throw std::runtime_error(
                    "QUADOBJ lists both (i,j) and (j,i): it takes one triangle of "
                    "Q, QMATRIX the full matrix" + where(e, i, j));
            const auto& sum = e.n_lower > 0 ? e.lower : e.upper;
            value = sum.nearest();
            repeated += (e.n_lower > 0 ? e.n_lower : e.n_upper) - 1;
        }
        if (value == 0.0) continue;
        ++rep.n_quad_entries;
        if (i == j) {
            qp.q_diag[static_cast<std::size_t>(i)] = value;
        } else {
            rep.has_off_diagonal = true;
        }
        rows.push_back(i);
        cols.push_back(j);
        values.push_back(value);
        if (i != j) {
            rows.push_back(j);
            cols.push_back(i);
            values.push_back(value);
        }
    }
    if (repeated > 0)
        rep.warnings.push_back(std::to_string(repeated) +
                               " quadratic entr" + (repeated == 1 ? "y" : "ies") +
                               " repeated for the same pair; summed exactly");
    if (rep.has_off_diagonal)
        qp.q_matrix = sparse::from_triplets(n, n, rows, cols, values);
    return qp;
}

}  // namespace sor::io
