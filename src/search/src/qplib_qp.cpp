// SOR — QPLIB -> QcqpProblem / QpProblem; see qplib_qp.hpp for the
// objective and constraint conventions.
#include "sor/search/qplib_qp.hpp"

#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <map>
#include <utility>

namespace sor::search {

void qplib_to_qcqp(const io::QplibInstance& q, engines::QcqpProblem& out) {
    const auto inf = [&](f64 v) {
        if (v >= q.inf_bound) return model::kInf;
        if (v <= -q.inf_bound) return -model::kInf;
        return v;
    };
    const Index n = q.n, m = q.m;
    const f64 sgn = q.maximize ? -1.0 : 1.0;

    out = engines::QcqpProblem{};
    out.objective_negated = q.maximize;
    auto& lp = out.qp.linear;
    lp.name = q.name;
    lp.maximize = false;
    for (const f64 g : q.g) lp.c.push_back(sgn * g);
    lp.obj_offset = sgn * q.f_const;

    std::vector<Index> ar, ac;
    ar.reserve(q.a_val.size());
    ac.reserve(q.a_val.size());
    for (std::size_t t = 0; t < q.a_val.size(); ++t) {
        ar.push_back(q.a_row[t] - 1);
        ac.push_back(q.a_col[t] - 1);
    }
    lp.A = sparse::from_triplets(m, n, ar, ac, q.a_val);
    for (Index i = 0; i < m; ++i) {
        lp.row_lo.push_back(inf(q.c_lo[static_cast<std::size_t>(i)]));
        lp.row_hi.push_back(inf(q.c_hi[static_cast<std::size_t>(i)]));
    }
    lp.is_integer.assign(static_cast<std::size_t>(n), false);
    for (Index j = 0; j < n; ++j) {
        const auto sj = static_cast<std::size_t>(j);
        f64 lo = inf(q.x_lo[sj]), hi = inf(q.x_hi[sj]);
        const auto t = q.var_type[sj];
        if (t == io::QplibVarType::Binary) {
            lo = std::max(lo, 0.0);
            hi = std::min(hi, 1.0);
        }
        lp.is_integer[sj] = t != io::QplibVarType::Continuous;
        lp.col_lo.push_back(lo);
        lp.col_hi.push_back(hi);
    }
    lp.col_names.resize(static_cast<std::size_t>(n));
    for (Index j = 0; j < n; ++j) {
        const auto sj = static_cast<std::size_t>(j);
        lp.col_names[sj] = sj < q.var_names.size() && !q.var_names[sj].empty()
                               ? q.var_names[sj] : io::qplib_default_var_name(q, j);
    }
    if (std::any_of(q.con_names.begin(), q.con_names.end(),
                    [](const std::string& s) { return !s.empty(); }))
        lp.row_names = q.con_names;

    std::vector<Index> qr, qc;
    std::vector<f64> qv;
    for (std::size_t t = 0; t < q.h_val.size(); ++t) {
        const Index r = q.h_row[t] - 1, c = q.h_col[t] - 1;
        const f64 v = sgn * q.h_val[t];
        if (r == c) {
            qr.push_back(r); qc.push_back(r); qv.push_back(v);
        } else {
            qr.push_back(r); qc.push_back(c); qv.push_back(0.5 * v);
            qr.push_back(c); qc.push_back(r); qv.push_back(0.5 * v);
        }
    }
    out.qp.q_matrix = sparse::from_triplets(n, n, qr, qc, qv);

    // Constraint Hessians, one QuadRow per row that has any entry.  The
    // objective's rule applied to the upper triangle: an off-diagonal file
    // entry v at (r,c) or (c,r) adds 0.5*v to Q_rc (r < c); a diagonal one
    // adds v to Q_rr.  Constraints are NOT negated for a maximisation.
    std::map<Index, std::map<std::pair<Index, Index>, f64>> rows;
    for (std::size_t t = 0; t < q.hc_val.size(); ++t) {
        Index r = q.hc_row[t] - 1, c = q.hc_col[t] - 1;
        if (r > c) std::swap(r, c);
        rows[q.hc_con[t] - 1][{r, c}] += (r == c ? 1.0 : 0.5) * q.hc_val[t];
    }
    for (const auto& [row, entries] : rows) {
        engines::QuadRow qr_row;
        qr_row.row = row;
        for (const auto& [rc, v] : entries) {
            if (v == 0.0) continue;   // cancelled duplicates: not a quadratic term
            qr_row.r.push_back(rc.first);
            qr_row.c.push_back(rc.second);
            qr_row.v.push_back(v);
        }
        if (!qr_row.v.empty()) out.quad.push_back(std::move(qr_row));
    }
}

bool qplib_to_qp(const io::QplibInstance& q, const QplibToQpOptions& opts,
                 engines::QpProblem& out, bool& negated, std::string& why) {
    if (q.has_quadratic_constraints()) {
        why = q.name + " has quadratic constraints; the QP engines take linear "
                       "constraints only";
        return false;
    }
    for (const auto t : q.var_type) {
        const bool ok = t == io::QplibVarType::Continuous || opts.keep_integer ||
                        (opts.relax_binary && t == io::QplibVarType::Binary);
        if (!ok) {
            why = q.name + (opts.relax_binary
                                ? " has general integer variables; only binaries relax"
                                : " has discrete variables; the convex QP engines "
                                  "are continuous only");
            return false;
        }
    }
    engines::QcqpProblem full;
    qplib_to_qcqp(q, full);
    negated = full.objective_negated;
    if (!engines::qcqp_to_qp(std::move(full), out, why)) return false;   // unreachable: checked above
    // The QP engines are continuous: binaries reach here only when the
    // caller asked to relax them (QCR), and then they must not stay typed.
    // The MIQP branch-and-bound asks to keep the integer typing instead.
    out.linear.is_integer.assign(static_cast<std::size_t>(q.n), false);
    if (opts.keep_integer)
        for (Index j = 0; j < q.n; ++j)
            out.linear.is_integer[static_cast<std::size_t>(j)] =
                q.var_type[static_cast<std::size_t>(j)] != io::QplibVarType::Continuous;
    return true;
}

}  // namespace sor::search
