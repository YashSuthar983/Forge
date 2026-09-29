#include "sor/io/solution_csv.hpp"

#include <cmath>
#include <ostream>
#include <string>

namespace sor::io {
namespace {

using core::f64;

// A name from an MPS file can legally contain a comma or a quote, and a
// planner's model often does (blend codes like "BLEND,A"). Quote every field
// and double any embedded quote -- RFC 4180, which is what a spreadsheet
// expects. Writing the name raw would silently shift every later column.
std::string csv(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (const char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// Empty cell for an infinite bound: a spreadsheet reads a blank as "no limit",
// while the text "inf" becomes a label that breaks any formula over the column.
void bound_cell(std::ostream& out, f64 v) {
    if (std::isinf(v)) return;
    out << v;
}

// Which bound the value is sitting on, in words, so the planner can see what
// is binding without comparing three columns by eye. The tolerance is the same
// 1e-9 the model's own violation helpers use for "at".
const char* at_bound(f64 v, f64 lo, f64 hi) {
    constexpr f64 tol = 1e-9;
    const bool at_lo = !std::isinf(lo) && std::fabs(v - lo) <= tol * (1.0 + std::fabs(lo));
    const bool at_hi = !std::isinf(hi) && std::fabs(v - hi) <= tol * (1.0 + std::fabs(hi));
    if (at_lo && at_hi) return "fixed";
    if (at_lo) return "lower";
    if (at_hi) return "upper";
    return "";
}

std::string name_or_index(const std::vector<std::string>& names, std::size_t i,
                          const char* prefix) {
    if (i < names.size() && !names[i].empty()) return names[i];
    return std::string(prefix) + std::to_string(i);
}

}  // namespace

void write_solution_csv(std::ostream& out, const model::LpProblem& p,
                        const core::SolveResult& r) {
    out.precision(17);
    out << "kind,name,value,lower,upper,at_bound\n";

    out << "summary," << csv("status") << ',' << ',' << ',' << ','
        << csv(std::string(core::to_string(r.status))) << '\n';
    out << "summary," << csv("proof_level") << ',' << ',' << ',' << ','
        << csv(std::string(core::to_string(r.proof))) << '\n';
    // The objective is reported in the FILE's own sense. p.objective() already
    // un-flips a maximize model, so a planner reading this never has to know
    // that the solver minimised internally.
    if (!r.x.empty() && r.x.size() == static_cast<std::size_t>(p.n_cols()))
        out << "summary," << csv("objective") << ',' << p.objective(r.x) << ",,,\n";
    else
        out << "summary," << csv("objective") << ',' << r.objective << ",,,\n";

    const auto n = static_cast<std::size_t>(p.n_cols());
    for (std::size_t j = 0; j < n && j < r.x.size(); ++j) {
        const f64 lo = j < p.col_lo.size() ? p.col_lo[j] : 0.0;
        const f64 hi = j < p.col_hi.size() ? p.col_hi[j] : model::kInf;
        out << (j < p.is_integer.size() && p.is_integer[j] ? "integer" : "variable")
            << ',' << csv(name_or_index(p.col_names, j, "c")) << ',' << r.x[j] << ',';
        bound_cell(out, lo);
        out << ',';
        bound_cell(out, hi);
        out << ',' << at_bound(r.x[j], lo, hi) << '\n';
    }

    // Row activity (Ax)_i, so the planner sees what each constraint actually
    // consumed rather than only whether it was satisfied. Computed here from
    // the model rather than taken from the engine: this file is meant to be
    // readable evidence, and evidence recomputed from the model is worth more
    // than evidence echoed back by the thing being checked.
    if (r.x.size() != static_cast<std::size_t>(p.n_cols())) return;
    const auto& A = p.A;
    const auto& rp = A.pattern.row_ptr();
    const auto& ci = A.pattern.col_idx();
    const auto m = static_cast<std::size_t>(p.n_rows());
    for (std::size_t i = 0; i < m; ++i) {
        f64 act = 0.0;
        for (auto k = rp[i]; k < rp[i + 1]; ++k)
            act += A.vals[static_cast<std::size_t>(k)] *
                   r.x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
        const f64 lo = i < p.row_lo.size() ? p.row_lo[i] : -model::kInf;
        const f64 hi = i < p.row_hi.size() ? p.row_hi[i] : model::kInf;
        out << "constraint," << csv(name_or_index(p.row_names, i, "r")) << ','
            << act << ',';
        bound_cell(out, lo);
        out << ',';
        bound_cell(out, hi);
        out << ',' << at_bound(act, lo, hi) << '\n';
    }
}

}  // namespace sor::io
