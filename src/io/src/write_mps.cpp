#include "sor/io/write_mps.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace sor::io {
namespace {

using core::Index;
using core::f64;
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

std::string name_or(const std::vector<std::string>& names, Index i,
                    const char* prefix) {
    if (i >= 0 && sz(i) < names.size() && !names[sz(i)].empty())
        return names[sz(i)];
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%d", prefix, static_cast<int>(i + 1));
    return buf;
}

void write_body(std::ostream& out, const model::LpProblem& p) {
    const Index m = p.n_rows();
    const Index n = p.n_cols();
    out << "NAME          " << (p.name.empty() ? "SOR" : p.name) << "\n";
    if (p.maximize) out << "OBJSENSE\n MAX\n";
    out << "ROWS\n";
    out << " N  COST\n";
    for (Index i = 0; i < m; ++i) {
        const f64 lo = p.row_lo[sz(i)], hi = p.row_hi[sz(i)];
        char kind = 'L';
        if (lo == hi) kind = 'E';
        else if (lo > -model::kInf && hi >= model::kInf) kind = 'G';
        else if (lo <= -model::kInf && hi < model::kInf) kind = 'L';
        else kind = 'E';  // ranged: emit as E with mid - writer keeps lo/hi via RHS+RANGES later; use L/G prefer
        // Two-sided with finite lo and hi: emit as E only if equal; else L and rely on RANGES.
        if (lo > -model::kInf && hi < model::kInf && lo != hi) kind = 'L';
        out << " " << kind << "  " << name_or(p.row_names, i, "R") << "\n";
    }

    out << "COLUMNS\n";
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    const auto& av = p.A.vals;
    // Column-major emission: gather entries per column.
    std::vector<std::vector<std::pair<Index, f64>>> cols(sz(n));
    for (Index i = 0; i < m; ++i) {
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            cols[sz(ci[sz(k)])].push_back({i, av[sz(k)]});
    }

    bool in_int = false;
    for (Index j = 0; j < n; ++j) {
        const bool is_int = !p.is_integer.empty() && p.is_integer[sz(j)];
        if (is_int && !in_int) {
            out << "    MARK0000  'MARKER'                 'INTORG'\n";
            in_int = true;
        } else if (!is_int && in_int) {
            out << "    MARK0001  'MARKER'                 'INTEND'\n";
            in_int = false;
        }
        const std::string cj = name_or(p.col_names, j, "X");
        // Objective coefficient.
        if (p.c[sz(j)] != 0.0)
            out << "    " << cj << "  COST  " << p.c[sz(j)];
        int pair = (p.c[sz(j)] != 0.0) ? 1 : 0;
        for (const auto& [i, v] : cols[sz(j)]) {
            if (pair == 0) out << "    " << cj;
            out << "  " << name_or(p.row_names, i, "R") << "  " << v;
            ++pair;
            if (pair == 2) { out << "\n"; pair = 0; }
        }
        if (pair == 1) out << "\n";
        if (pair == 0 && cols[sz(j)].empty() && p.c[sz(j)] == 0.0)
            out << "    " << cj << "  COST  0\n";
    }
    if (in_int) out << "    MARK0001  'MARKER'                 'INTEND'\n";

    out << "RHS\n";
    bool any_rhs = (p.obj_offset != 0.0);
    if (p.obj_offset != 0.0)
        out << "    RHS       COST  " << (-p.obj_offset) << "\n";
    for (Index i = 0; i < m; ++i) {
        const f64 lo = p.row_lo[sz(i)], hi = p.row_hi[sz(i)];
        f64 rhs = 0.0;
        bool write = false;
        if (lo == hi) { rhs = lo; write = true; }
        else if (hi < model::kInf) { rhs = hi; write = true; }
        else if (lo > -model::kInf) { rhs = lo; write = true; }
        if (write && rhs != 0.0) {
            out << "    RHS       " << name_or(p.row_names, i, "R") << "  " << rhs << "\n";
            any_rhs = true;
        } else if (write && rhs == 0.0) {
            // zero RHS is default; skip
        }
        (void)any_rhs;
    }

    // RANGES for two-sided inequalities.
    bool any_range = false;
    for (Index i = 0; i < m; ++i) {
        const f64 lo = p.row_lo[sz(i)], hi = p.row_hi[sz(i)];
        if (lo > -model::kInf && hi < model::kInf && lo != hi) {
            if (!any_range) { out << "RANGES\n"; any_range = true; }
            // For L row with RHS=hi, range = hi - lo.
            out << "    RNG       " << name_or(p.row_names, i, "R") << "  "
                << (hi - lo) << "\n";
        }
    }

    out << "BOUNDS\n";
    for (Index j = 0; j < n; ++j) {
        const f64 lo = p.col_lo[sz(j)], hi = p.col_hi[sz(j)];
        const std::string cj = name_or(p.col_names, j, "X");
        const bool is_int = !p.is_integer.empty() && p.is_integer[sz(j)];
        if (lo == hi) {
            out << " FX BND       " << cj << "  " << lo << "\n";
        } else {
            if (lo <= -model::kInf && hi >= model::kInf) {
                out << " FR BND       " << cj << "\n";
            } else {
                if (lo != 0.0 && lo > -model::kInf)
                    out << (is_int ? " LI" : " LO") << " BND       " << cj
                        << "  " << lo << "\n";
                // Use UP for both (not UI): PuLP/CBC's MPS reader rejects UI.
                if (hi < model::kInf)
                    out << " UP BND       " << cj << "  " << hi << "\n";
                if (lo <= -model::kInf && hi < model::kInf && !is_int)
                    out << " MI BND       " << cj << "\n";
            }
        }
    }
}

}  // namespace

void write_mps(std::ostream& out, const model::LpProblem& p) {
    write_body(out, p);
    out << "ENDATA\n";
}

void write_mps_file(const std::string& path, const model::LpProblem& p) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    write_mps(out, p);
}

void write_qps(std::ostream& out, const model::LpProblem& p,
               const std::vector<f64>& q_diag) {
    if (static_cast<Index>(q_diag.size()) != p.n_cols())
        throw std::invalid_argument("write_qps: q_diag size != n_cols");
    write_body(out, p);
    out << "QUADOBJ\n";
    for (Index j = 0; j < p.n_cols(); ++j) {
        if (q_diag[sz(j)] == 0.0) continue;
        const std::string cj = name_or(p.col_names, j, "X");
        // Store Q_jj for obj = 1/2 x' Q x + c'x.
        out << "    " << cj << "  " << cj << "  " << q_diag[sz(j)] << "\n";
    }
    out << "ENDATA\n";
}

void write_qps_file(const std::string& path, const model::LpProblem& p,
                    const std::vector<f64>& q_diag) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    write_qps(out, p, q_diag);
}

}  // namespace sor::io
