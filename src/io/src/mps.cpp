#include "sor/io/mps.hpp"

#include <chrono>

#include "sor/io/gzip.hpp"
#include "sor/io/lp_format.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace sor::io {
namespace {

using core::f64;
using core::Index;
using model::kInf;

enum class Section { None, Name, ObjSense, Rows, Columns, Rhs, Ranges, Bounds, Quadobj, End };
enum class RowKind { Objective, LessEqual, GreaterEqual, Equal };

std::string upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream is(line);
    std::string tok;
    while (is >> tok) out.push_back(tok);
    return out;
}

// Fixed-format MPS field columns (1-based in the spec):
//   f1 2-3, f2 5-12, f3 15-22, f4 25-36, f5 40-47, f6 50-61
//
// Empty fields are dropped so the resulting token list has the same semantics as
// the free-format splitter (element 0 = first meaningful token), while names
// containing SPACES are preserved. Netlib's `forplan` needs exactly this: it has
// row and column names like "DEDO3 1R" and "DEDO3 11" that free-format
// whitespace splitting shatters.
std::vector<std::string> split_fixed(const std::string& line) {
    static constexpr int beg[6] = {1, 4, 14, 24, 39, 49};
    static constexpr int len[6] = {2, 8, 8, 12, 8, 12};
    std::vector<std::string> out;
    for (int i = 0; i < 6; ++i) {
        if (static_cast<int>(line.size()) <= beg[i]) break;
        std::string f = line.substr(static_cast<std::size_t>(beg[i]),
                                    static_cast<std::size_t>(len[i]));
        const auto b = f.find_first_not_of(" \t");
        if (b == std::string::npos) continue;          // drop empty field
        const auto e = f.find_last_not_of(" \t");
        out.push_back(f.substr(b, e - b + 1));
    }
    return out;
}

f64 parse_num(const std::string& s, std::size_t line_no) {
    try {
        std::size_t used = 0;
        const f64 v = std::stod(s, &used);
        if (used != s.size() || !std::isfinite(v)) throw std::invalid_argument("nonfinite or trailing");
        return v;
    } catch (...) {
        throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                 ": cannot parse number '" + s + "'");
    }
}

struct Builder {
    std::vector<RowKind> row_kind;
    std::vector<std::string> row_names;
    std::unordered_map<std::string, Index> row_of;   // includes objective rows
    std::vector<Index> constraint_index;             // row_of index -> constraint row, -1 for obj
    Index obj_slot = -1;                             // slot of the FIRST N row

    std::vector<RowKind> con_kind;                   // per constraint row
    std::vector<std::string> con_names;

    std::vector<std::string> col_names;
    std::unordered_map<std::string, Index> col_of;

    std::vector<Index> tri_r, tri_c;
    std::vector<f64> tri_v;

    std::vector<f64> obj;                            // per column
    f64 obj_rhs = 0.0;                               // RHS on objective row

    std::vector<f64> rhs, ranges;
    std::vector<bool> has_range;
    std::vector<f64> col_lo, col_hi;
    std::vector<bool> lo_set, hi_set, integer_flag;

    Index ensure_col(const std::string& n) {
        auto it = col_of.find(n);
        if (it != col_of.end()) return it->second;
        const Index j = static_cast<Index>(col_names.size());
        col_of.emplace(n, j);
        col_names.push_back(n);
        obj.push_back(0.0);
        col_lo.push_back(0.0);
        col_hi.push_back(kInf);
        lo_set.push_back(false);
        hi_set.push_back(false);
        integer_flag.push_back(false);
        return j;
    }
};

}  // namespace

model::LpProblem read_mps(std::istream& in, MpsReadReport& rep,
                          const MpsReadOptions& opt) {
    // Phase timing. Reading was measured superlinear on large models --
    // 1 MB -> 0.6 s, 11 MB -> 12.7 s, 25 MB -> >90 s, with gzip ruled out --
    // and it sits entirely OUTSIDE the solver's time budget, so a big model
    // blows the limit before any phase that checks the clock even starts.
    // External profilers are unavailable here (perf_event_paranoid=4,
    // ptrace_scope), so the reader measures itself.
    using RClock = std::chrono::steady_clock;
    const auto t_start = RClock::now();
    double ms_parse = 0.0, ms_assemble = 0.0;
    Builder b;
    Section sec = Section::None;
    std::string name;
    bool maximize = false;
    bool in_integer_marker = false;
    std::size_t line_no = 0;
    std::string line;

    auto fields = [&](const std::string& l) {
        return opt.fixed_format ? split_fixed(l) : split_ws(l);
    };

    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '*') continue;                       // comment

        const bool is_header = !std::isspace(static_cast<unsigned char>(line[0]));
        if (is_header) {
            const auto f = split_ws(line);
            if (f.empty()) continue;
            const std::string key = upper(f[0]);
            if      (key == "NAME")     { sec = Section::Name; if (f.size() > 1) name = f[1]; }
            else if (key == "OBJSENSE") { sec = Section::ObjSense;
                                          if (f.size() > 1 && upper(f[1]).rfind("MAX", 0) == 0)
                                              maximize = true; }
            else if (key == "OBJSENS")  { sec = Section::ObjSense; }
            else if (key == "ROWS")     { sec = Section::Rows; }
            else if (key == "COLUMNS")  { sec = Section::Columns; }
            else if (key == "RHS")      { sec = Section::Rhs; }
            else if (key == "RANGES")   { sec = Section::Ranges; }
            else if (key == "BOUNDS")   { sec = Section::Bounds; }
            else if (key == "QUADOBJ" || key == "QMATRIX") {
                if (opt.strict)
                    throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                             ": QUADOBJ present; this reader is LP-only");
                rep.warnings.push_back("QUADOBJ section ignored (LP-only reader)");
                sec = Section::Quadobj;
            }
            else if (key == "ENDATA")   { sec = Section::End; break; }
            else {
                if (opt.strict)
                    throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                             ": unknown section '" + f[0] + "'");
                rep.warnings.push_back("ignored unknown section '" + f[0] + "'");
                sec = Section::None;
            }
            continue;
        }

        const auto f = fields(line);
        if (f.empty()) continue;

        switch (sec) {
            case Section::ObjSense: {
                if (upper(f[0]).rfind("MAX", 0) == 0) maximize = true;
                break;
            }
            case Section::Rows: {
                if (f.size() < 2)
                    throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                             ": ROWS entry needs kind and name");
                const std::string k = upper(f[0]);
                RowKind kind;
                if      (k == "N") kind = RowKind::Objective;
                else if (k == "L") kind = RowKind::LessEqual;
                else if (k == "G") kind = RowKind::GreaterEqual;
                else if (k == "E") kind = RowKind::Equal;
                else throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                              ": unknown row kind '" + f[0] + "'");
                if (b.row_of.count(f[1]))
                    throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                             ": duplicate row name '" + f[1] + "'");
                const Index slot = static_cast<Index>(b.row_kind.size());
                b.row_of.emplace(f[1], slot);
                b.row_kind.push_back(kind);
                b.row_names.push_back(f[1]);
                if (kind == RowKind::Objective) {
                    b.constraint_index.push_back(-1);
                    // Only the first N row is the objective; later N rows are
                    // free rows and their coefficients are discarded.
                    if (b.obj_slot < 0) b.obj_slot = slot;
                } else {
                    b.constraint_index.push_back(static_cast<Index>(b.rhs.size()));
                    b.con_kind.push_back(kind);
                    b.con_names.push_back(f[1]);
                    b.rhs.push_back(0.0);
                    b.ranges.push_back(0.0);
                    b.has_range.push_back(false);
                }
                break;
            }
            case Section::Columns: {
                // MARKER lines toggle the integer block.
                bool is_marker = false;
                for (const auto& t : f)
                    if (upper(t).find("MARKER") != std::string::npos) is_marker = true;
                if (is_marker) {
                    for (const auto& t : f) {
                        const std::string u = upper(t);
                        if (u.find("INTORG") != std::string::npos) in_integer_marker = true;
                        if (u.find("INTEND") != std::string::npos) in_integer_marker = false;
                    }
                    break;
                }
                if (f.size() < 3)
                    throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                             ": COLUMNS entry needs col, row, value");
                const Index j = b.ensure_col(f[0]);
                if (in_integer_marker) b.integer_flag[static_cast<std::size_t>(j)] = true;

                for (std::size_t k = 1; k + 1 < f.size(); k += 2) {
                    auto it = b.row_of.find(f[k]);
                    if (it == b.row_of.end())
                        throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                                 ": unknown row '" + f[k] + "'");
                    const f64 v = parse_num(f[k + 1], line_no);
                    const Index slot = it->second;
                    if (b.row_kind[static_cast<std::size_t>(slot)] == RowKind::Objective) {
                        if (slot == b.obj_slot) b.obj[static_cast<std::size_t>(j)] += v;
                        // else: an extra free N row -- discarded.
                    } else if (opt.small_matrix_value > 0.0 &&
                               std::fabs(v) <= opt.small_matrix_value) {
                        // Below the agreed threshold. The column still exists;
                        // only this coefficient is dropped.
                        ++rep.small_values_dropped;
                        rep.largest_small_value_dropped =
                            std::max(rep.largest_small_value_dropped, std::fabs(v));
                    } else {
                        b.tri_r.push_back(b.constraint_index[static_cast<std::size_t>(slot)]);
                        b.tri_c.push_back(j);
                        b.tri_v.push_back(v);
                    }
                }
                break;
            }
            case Section::Rhs: {
                // First token may be an RHS-set name, or may be omitted.
                std::size_t k = (f.size() % 2 == 1) ? 1 : 0;
                for (; k + 1 < f.size(); k += 2) {
                    auto it = b.row_of.find(f[k]);
                    if (it == b.row_of.end())
                        throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                                 ": unknown row '" + f[k] + "' in RHS");
                    const f64 v = parse_num(f[k + 1], line_no);
                    const Index slot = it->second;
                    if (b.row_kind[static_cast<std::size_t>(slot)] == RowKind::Objective) {
                        // Convention: RHS on the objective row is the NEGATIVE
                        // of the objective constant.
                        if (slot == b.obj_slot) b.obj_rhs = -v;
                    } else {
                        b.rhs[static_cast<std::size_t>(
                            b.constraint_index[static_cast<std::size_t>(slot)])] = v;
                    }
                }
                break;
            }
            case Section::Ranges: {
                std::size_t k = (f.size() % 2 == 1) ? 1 : 0;
                for (; k + 1 < f.size(); k += 2) {
                    auto it = b.row_of.find(f[k]);
                    if (it == b.row_of.end())
                        throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                                 ": unknown row '" + f[k] + "' in RANGES");
                    const Index slot = it->second;
                    if (b.row_kind[static_cast<std::size_t>(slot)] == RowKind::Objective)
                        continue;
                    const auto ci = static_cast<std::size_t>(
                        b.constraint_index[static_cast<std::size_t>(slot)]);
                    b.ranges[ci] = parse_num(f[k + 1], line_no);
                    b.has_range[ci] = true;
                    rep.had_ranges = true;
                }
                break;
            }
            case Section::Bounds: {
                if (f.size() < 3)
                    throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                             ": BOUNDS entry too short");
                const std::string type = upper(f[0]);
                // Layout is  TYPE  setname  colname  [value]  but the set name is
                // sometimes omitted. Disambiguate by looking for a known column.
                std::size_t ci = 2, vi = 3;
                if (!b.col_of.count(f[2]) && b.col_of.count(f[1])) { ci = 1; vi = 2; }
                const auto cit = b.col_of.find(f[ci]);
                if (cit == b.col_of.end())
                    throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                             ": unknown column '" + f[ci] + "' in BOUNDS");
                const auto j = static_cast<std::size_t>(cit->second);
                const bool needs_value =
                    (type == "UP" || type == "LO" || type == "FX" ||
                     type == "UI" || type == "LI");
                if (needs_value && vi >= f.size())
                    throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                             ": bound type " + type + " needs a value");
                const f64 v = needs_value ? parse_num(f[vi], line_no) : 0.0;

                if (type == "UP" || type == "UI") {
                    b.col_hi[j] = v;
                    b.hi_set[j] = true;
                    // Documented MPS quirk: a negative UP bound on a column whose
                    // lower bound is still the 0 default implies lower = -inf.
                    if (v < 0.0 && !b.lo_set[j]) b.col_lo[j] = -kInf;
                    if (type == "UI") b.integer_flag[j] = true;
                } else if (type == "LO" || type == "LI") {
                    b.col_lo[j] = v;
                    b.lo_set[j] = true;
                    if (type == "LI") b.integer_flag[j] = true;
                } else if (type == "FX") {
                    b.col_lo[j] = b.col_hi[j] = v;
                    b.lo_set[j] = b.hi_set[j] = true;
                } else if (type == "FR") {
                    b.col_lo[j] = -kInf; b.col_hi[j] = kInf;
                    b.lo_set[j] = b.hi_set[j] = true;
                } else if (type == "MI") {
                    b.col_lo[j] = -kInf; b.lo_set[j] = true;
                } else if (type == "PL") {
                    b.col_hi[j] = kInf; b.hi_set[j] = true;
                } else if (type == "BV") {
                    b.col_lo[j] = 0.0; b.col_hi[j] = 1.0;
                    b.lo_set[j] = b.hi_set[j] = true;
                    b.integer_flag[j] = true;
                } else {
                    throw std::runtime_error("MPS line " + std::to_string(line_no) +
                                             ": unknown bound type '" + f[0] + "'");
                }
                break;
            }
            case Section::Quadobj:
            case Section::Name:
            case Section::None:
            case Section::End:
                break;
        }
    }

    if (sec != Section::End)
        rep.warnings.push_back("no ENDATA record found");

    ms_parse = std::chrono::duration<double, std::milli>(
                   RClock::now() - t_start).count();
    const auto t_assemble = RClock::now();

    // Assemble.
    const Index n_cols = static_cast<Index>(b.col_names.size());
    const Index n_rows = static_cast<Index>(b.rhs.size());

    // An INTORG/INTEND column with no BOUNDS record defaults to binary in
    // MPS. An explicit bound record instead gives it general integer bounds.
    for (std::size_t j = 0; j < b.integer_flag.size(); ++j) {
        if (b.integer_flag[j] && !b.lo_set[j] && !b.hi_set[j])
            b.col_hi[j] = 1.0;
    }

    model::LpProblem p;
    p.name = name;
    p.maximize = maximize;
    p.A = sparse::from_triplets(n_rows, n_cols, b.tri_r, b.tri_c, b.tri_v);
    p.c = b.obj;
    p.obj_offset = b.obj_rhs;
    p.col_lo = b.col_lo;
    p.col_hi = b.col_hi;
    p.col_names = b.col_names;
    p.is_integer.assign(b.integer_flag.begin(), b.integer_flag.end());

    p.row_lo.resize(static_cast<std::size_t>(n_rows));
    p.row_hi.resize(static_cast<std::size_t>(n_rows));
    p.row_names = b.con_names;

    for (std::size_t i = 0; i < static_cast<std::size_t>(n_rows); ++i) {
        const RowKind kind = b.con_kind[i];
        const f64 r = b.rhs[i];
        if (!b.has_range[i]) {
            switch (kind) {
                case RowKind::LessEqual:    p.row_lo[i] = -kInf; p.row_hi[i] = r; break;
                case RowKind::GreaterEqual: p.row_lo[i] = r;     p.row_hi[i] = kInf; break;
                case RowKind::Equal:        p.row_lo[i] = r;     p.row_hi[i] = r; break;
                case RowKind::Objective:    break;
            }
        } else {
            // RANGES semantics from the MPS specification.
            const f64 R = b.ranges[i];
            const f64 a = std::fabs(R);
            switch (kind) {
                case RowKind::LessEqual:    p.row_lo[i] = r - a; p.row_hi[i] = r; break;
                case RowKind::GreaterEqual: p.row_lo[i] = r;     p.row_hi[i] = r + a; break;
                case RowKind::Equal:
                    if (R >= 0.0) { p.row_lo[i] = r;     p.row_hi[i] = r + R; }
                    else          { p.row_lo[i] = r + R; p.row_hi[i] = r; }
                    break;
                case RowKind::Objective: break;
            }
        }
    }

    // Integrality is discarded AFTER the model is fully assembled, so the
    // relaxation differs from the original in exactly one respect and the
    // report can state whether the file actually carried any.
    const std::size_t integer_columns = p.n_integer();
    if (opt.relax_integrality && integer_columns > 0) {
        p.is_integer.assign(p.is_integer.size(), false);
        rep.relaxed_integrality = true;
    }

    p.validate();

    if (rep.small_values_dropped > 0)
        rep.warnings.push_back(
            "dropped " + std::to_string(rep.small_values_dropped) +
            " matrix coefficient(s) at or below " +
            std::to_string(opt.small_matrix_value) + " (largest " +
            std::to_string(rep.largest_small_value_dropped) + ")");

    rep.n_rows = static_cast<std::size_t>(n_rows);
    rep.n_cols = static_cast<std::size_t>(n_cols);
    rep.nnz = p.nnz();
    // The count the FILE carried, so a relaxation still reports what it relaxed.
    rep.n_integer = integer_columns;
    rep.had_objsense_max = maximize;
    ms_assemble = std::chrono::duration<double, std::milli>(
                      RClock::now() - t_assemble).count();
    rep.ms_parse = ms_parse;
    rep.ms_assemble = ms_assemble;
    rep.lines_read = line_no;

    return p;
}

model::LpProblem read_mps_file(const std::string& path, MpsReadReport& rep,
                               const MpsReadOptions& opt) {
    // Gzip is detected from the CONTENT, not the extension: MIPLIB ships
    // `.mps.gz`, but a corpus unpacked in place keeps that name, and a file
    // named `.mps` is sometimes compressed.
    if (file_has_gzip_magic(path)) {
        std::istringstream in(read_maybe_gzip_file(path));
        rep.used_gzip = true;
        return read_mps(in, rep, opt);
    }
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open MPS file: " + path);
    return read_mps(in, rep, opt);
}

model::LpProblem read_mps_file_auto(const std::string& path, MpsReadReport& rep,
                                    const MpsReadOptions& opt) {
    // A .lp / .lp.gz file is CPLEX LP format, not MPS: route it to its own
    // reader so every loader that calls this function accepts both.
    if (has_lp_extension(path)) return read_lp_file(path, rep, opt);
    MpsReadOptions free_opt = opt;
    free_opt.fixed_format = false;
    try {
        auto p = read_mps_file(path, rep, free_opt);
        rep.used_fixed_format = false;
        return p;
    } catch (const std::exception& first_err) {
        MpsReadReport rep2;
        MpsReadOptions fixed_opt = opt;
        fixed_opt.fixed_format = true;
        try {
            auto p = read_mps_file(path, rep2, fixed_opt);
            rep = rep2;
            rep.used_fixed_format = true;
            rep.warnings.push_back(
                std::string("free-format parse failed (") + first_err.what() +
                "); succeeded in fixed format");
            return p;
        } catch (const std::exception& second_err) {
            // Report the FREE-format error: it is almost always the informative
            // one, since fixed format only helps with spaces in names.
            throw std::runtime_error(
                std::string("MPS parse failed in both formats. free: ") +
                first_err.what() + " | fixed: " + second_err.what());
        }
    }
}

model::LpProblem read_mps_file_auto(const std::string& path, MpsReadReport& rep,
                                    bool strict) {
    MpsReadOptions opt;
    opt.strict = strict;
    return read_mps_file_auto(path, rep, opt);
}

}  // namespace sor::io
