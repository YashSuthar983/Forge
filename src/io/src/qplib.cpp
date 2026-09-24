#include "sor/io/qplib.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <cerrno>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace sor::io {
namespace {

// Parse one real.  Overflow is NOT an error here: QPLIB's convex instances
// write their infinity sentinel as 1.79769313486232E+308, which is DBL_MAX
// rounded UP in the 15th digit and so lies just above it.  std::stod throws
// out_of_range and istream >> double sets failbit on that, and before this
// every one of the 19 convex continuous QPLIB instances failed to load.
// A magnitude past DBL_MAX can only mean "infinite" in this format, so it
// becomes +-inf and the ordinary inf_bound handling applies.
bool parse_real(const std::string& w, f64& out) {
    if (w.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(w.c_str(), &end);
    if (end == w.c_str() || *end != '\0') return false;
    if (errno == ERANGE && std::fabs(v) > 1.0) {   // overflow, not underflow
        out = v > 0 ? std::numeric_limits<f64>::infinity()
                    : -std::numeric_limits<f64>::infinity();
        return true;
    }
    out = v;   // underflow rounds toward zero, which is the right answer
    return true;
}


// A cursor over the significant lines of the file. QPLIB allows a free-form
// comment after the last value on a line, so every accessor reads only the
// leading whitespace-separated fields it needs and ignores the remainder.
class LineCursor {
public:
    LineCursor(std::vector<std::string> lines, std::string name)
        : lines_(std::move(lines)), name_(std::move(name)) {}

    const std::string& next() {
        if (pos_ >= lines_.size()) fail("unexpected end of file");
        return lines_[pos_++];
    }

    std::size_t consumed() const noexcept { return pos_; }
    std::size_t total() const noexcept { return lines_.size(); }
    void set_name(std::string n) { name_ = std::move(n); }

    [[noreturn]] void fail(const std::string& what) const {
        std::ostringstream os;
        os << "QPLIB" << (name_.empty() ? "" : " " + name_) << ": line "
           << pos_ << ": " << what;
        throw std::runtime_error(os.str());
    }

    // ---- typed field readers --------------------------------------------
    std::string word() {
        const std::string& l = next();
        std::istringstream is(l);
        std::string w;
        if (!(is >> w)) fail("expected a value, found a blank line");
        return w;
    }

    f64 real() {
        const std::string w = word();
        f64 v = 0.0;
        if (!parse_real(w, v)) fail("expected a number, found \"" + w + "\"");
        return v;
    }

    Index integer() {
        const std::string w = word();
        try {
            return static_cast<Index>(std::stoll(w));
        } catch (...) {
            fail("expected an integer, found \"" + w + "\"");
        }
    }

    // One (row, col, value) triplet.
    void triplet(Index& r, Index& c, f64& v) {
        const std::string& l = next();
        std::istringstream is(l);
        long long a = 0, b = 0;
        std::string xs;
        double x = 0.0;
        if (!(is >> a >> b >> xs) || !parse_real(xs, x)) fail("expected \"row col value\", found \"" + l + "\"");
        r = static_cast<Index>(a);
        c = static_cast<Index>(b);
        v = x;
    }

    // One (constraint, row, col, value) quadruple -- the per-constraint
    // Hessian block is the only section shaped like this.
    void quadruple(Index& ci, Index& r, Index& c, f64& v) {
        const std::string& l = next();
        std::istringstream is(l);
        long long a = 0, b = 0, d = 0;
        std::string xs;
        double x = 0.0;
        if (!(is >> a >> b >> d >> xs) || !parse_real(xs, x))
            fail("expected \"constraint row col value\", found \"" + l + "\"");
        ci = static_cast<Index>(a);
        r = static_cast<Index>(b);
        c = static_cast<Index>(d);
        v = x;
    }

    // One (index, name) pair; the name is the first token after the index.
    void name_pair(Index& i, std::string& nm) {
        const std::string& l = next();
        std::istringstream is(l);
        long long a = 0;
        if (!(is >> a >> nm)) fail("expected \"index name\", found \"" + l + "\"");
        i = static_cast<Index>(a);
    }

    // One (index, value) pair.
    void pair(Index& i, f64& v) {
        const std::string& l = next();
        std::istringstream is(l);
        long long a = 0;
        std::string xs;
        double x = 0.0;
        if (!(is >> a >> xs) || !parse_real(xs, x)) fail("expected \"index value\", found \"" + l + "\"");
        i = static_cast<Index>(a);
        v = x;
    }

private:
    std::vector<std::string> lines_;
    std::string name_;
    std::size_t pos_ = 0;
};

// default value, then a count, then that many (index, value) overrides.
// This triple is the format's universal idiom for a mostly-constant vector.
std::vector<f64> read_defaulted_vector(LineCursor& cur, Index len,
                                       const char* what) {
    const f64 def = cur.real();
    const Index count = cur.integer();
    std::vector<f64> v(static_cast<std::size_t>(len < 0 ? 0 : len), def);
    for (Index k = 0; k < count; ++k) {
        Index idx = 0;
        f64 val = 0.0;
        cur.pair(idx, val);
        if (idx < 1 || idx > len) {
            cur.fail(std::string(what) + ": index " + std::to_string(idx) +
                     " out of range 1.." + std::to_string(len));
        }
        v[static_cast<std::size_t>(idx - 1)] = val;
    }
    return v;
}

bool is_blank(const std::string& s) {
    return std::all_of(s.begin(), s.end(),
                       [](unsigned char c) { return std::isspace(c) != 0; });
}

}  // namespace

std::string qplib_default_var_name(const QplibInstance& q, Index j) {
    const auto sj = static_cast<std::size_t>(j);
    const auto t = q.var_type[sj];
    // GAMS prefixes by KIND, not by the file's type code: a binary is an
    // integer bounded [0, 1] and is written b<k>, any other integer i<k>.
    // The QPLIB type section has ONE code for every integer variable --
    // QPLIB_3562 carries code 1 on all 63 of its variables and QPLIB's own
    // instancedata.csv counts 7 of them binary and 56 integer -- so the
    // bounds are what separates the two.
    const bool binary = t == QplibVarType::Binary ||
                        (t == QplibVarType::Integer && q.x_lo[sj] == 0.0 && q.x_hi[sj] == 1.0);
    const char p = binary ? 'b' : t == QplibVarType::Continuous ? 'x' : 'i';
    // j + 2 because the GAMS model of MOST instances declares the objective
    // variable ("objvar") first and the instance's variable 1 second.  That
    // is not true of all of them -- see apply_qplib_varnames_file(), which
    // reads the per-instance mapping and should be preferred; this fallback
    // resolves 273 of the library's 453 instances.
    return p + std::to_string(j + 2);
}

bool apply_qplib_varnames_file(const std::string& instance_path, QplibInstance& q) {
    const std::string suffix = ".qplib";
    std::string side = instance_path;
    if (side.size() > suffix.size() &&
        side.compare(side.size() - suffix.size(), suffix.size(), suffix) == 0)
        side.erase(side.size() - suffix.size());
    side += ".varnames";
    std::ifstream in(side);
    if (!in) return false;
    std::vector<std::string> names;
    for (std::string l; std::getline(in, l);) {
        while (!l.empty() && std::isspace(static_cast<unsigned char>(l.back()))) l.pop_back();
        if (!l.empty()) names.push_back(l);
    }
    if (names.size() != static_cast<std::size_t>(q.n))
        throw std::runtime_error(side + ": " + std::to_string(names.size()) +
                                 " names for " + std::to_string(q.n) + " variables");
    q.var_names = std::move(names);
    return true;
}

f64 QplibPointEval::max_violation() const noexcept {
    return std::max({max_row_violation, max_bound_violation, max_integrality_violation});
}

QplibPointEval qplib_evaluate_point(const QplibInstance& q, const std::vector<f64>& x) {
    if (x.size() != static_cast<std::size_t>(q.n))
        throw std::invalid_argument("qplib_evaluate_point: point has the wrong length");
    const auto X = [&x](Index one_based) { return x[static_cast<std::size_t>(one_based - 1)]; };
    QplibPointEval e;
    // Objective and rows are accumulated in long double: the gate compares
    // against published values at 1e-6 relative, and some rows sum 1e5+
    // terms of mixed sign.
    long double obj = q.f_const;
    for (std::size_t t = 0; t < q.h_val.size(); ++t)
        obj += 0.5L * q.h_val[t] * X(q.h_row[t]) * X(q.h_col[t]);
    for (std::size_t j = 0; j < q.g.size(); ++j) obj += static_cast<long double>(q.g[j]) * x[j];
    e.objective = static_cast<f64>(obj);

    std::vector<long double> act(static_cast<std::size_t>(q.m), 0.0L);
    std::vector<bool> is_qc(static_cast<std::size_t>(q.m), false);
    for (std::size_t t = 0; t < q.a_val.size(); ++t)
        act[static_cast<std::size_t>(q.a_row[t] - 1)] +=
            static_cast<long double>(q.a_val[t]) * X(q.a_col[t]);
    for (std::size_t t = 0; t < q.hc_val.size(); ++t) {
        const auto i = static_cast<std::size_t>(q.hc_con[t] - 1);
        act[i] += 0.5L * q.hc_val[t] * X(q.hc_row[t]) * X(q.hc_col[t]);
        is_qc[i] = true;
    }
    for (std::size_t i = 0; i < act.size(); ++i) {
        const f64 a = static_cast<f64>(act[i]);
        f64 v = 0.0;
        if (a < q.c_lo[i]) v = q.c_lo[i] - a;           // c_lo = -inf never fires
        if (a > q.c_hi[i]) v = std::max(v, a - q.c_hi[i]);
        e.max_row_violation = std::max(e.max_row_violation, v);
        if (is_qc[i]) e.max_qc_violation = std::max(e.max_qc_violation, v);
    }
    for (std::size_t j = 0; j < x.size(); ++j) {
        // A binary is an integer on {0,1} by the format's definition, whatever
        // bounds the file lists for it.
        if (q.var_type[j] == QplibVarType::Binary) {
            if (x[j] < 0.0) e.max_bound_violation = std::max(e.max_bound_violation, -x[j]);
            if (x[j] > 1.0) e.max_bound_violation = std::max(e.max_bound_violation, x[j] - 1.0);
        }
        if (x[j] < q.x_lo[j]) e.max_bound_violation = std::max(e.max_bound_violation, q.x_lo[j] - x[j]);
        if (x[j] > q.x_hi[j]) e.max_bound_violation = std::max(e.max_bound_violation, x[j] - q.x_hi[j]);
        if (q.var_type[j] != QplibVarType::Continuous)
            e.max_integrality_violation =
                std::max(e.max_integrality_violation, std::fabs(x[j] - std::round(x[j])));
    }
    return e;
}

QplibSolution read_qplib_solution(const std::string& path, const QplibInstance& q) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("QPLIB solution: cannot open " + path);
    std::unordered_map<std::string, Index> index;
    index.reserve(static_cast<std::size_t>(q.n) * 2);
    for (Index j = 0; j < q.n; ++j) {
        const auto sj = static_cast<std::size_t>(j);
        const std::string nm = sj < q.var_names.size() && !q.var_names[sj].empty()
                                   ? q.var_names[sj] : qplib_default_var_name(q, j);
        index.emplace(nm, j);
    }
    QplibSolution sol;
    sol.x.assign(static_cast<std::size_t>(q.n), 0.0);
    std::vector<bool> seen(static_cast<std::size_t>(q.n), false);
    std::size_t lineno = 0;
    for (std::string l; std::getline(in, l);) {
        ++lineno;
        std::istringstream is(l);
        std::string nm, vs;
        if (!(is >> nm)) continue;   // blank
        f64 v = 0.0;
        if (!(is >> vs) || !parse_real(vs, v))
            throw std::runtime_error("QPLIB solution " + path + ": line " +
                                     std::to_string(lineno) + ": expected \"name value\"");
        if (nm == "objvar") {
            sol.has_objvar = true;
            sol.objvar = v;
            // In most instances "objvar" is a GAMS variable that the QPLIB
            // model does not have, and this line only states the objective.
            // In a few (QPLIB_10035/10036/10037/10039) the objective variable
            // IS variable 1 of the instance, and skipping the line would
            // leave it at 0 and report the point as infeasible by exactly the
            // published objective.  Fall through when the name resolves.
            if (index.find(nm) == index.end()) continue;
        }
        const auto it = index.find(nm);
        if (it == index.end())
            throw std::runtime_error("QPLIB solution " + path + ": unknown variable \"" + nm +
                                     "\" (name/type mapping disagrees with the instance)");
        const auto sj = static_cast<std::size_t>(it->second);
        if (seen[sj])
            throw std::runtime_error("QPLIB solution " + path + ": variable \"" + nm +
                                     "\" listed twice");
        seen[sj] = true;
        sol.x[sj] = v;
        ++sol.listed;
    }
    return sol;
}

bool QplibInstance::is_discrete() const noexcept {
    return std::all_of(var_type.begin(), var_type.end(), [](QplibVarType t) {
        return t != QplibVarType::Continuous;
    });
}

QplibInstance read_qplib_file(const std::string& path, QplibReadReport& rep) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("QPLIB: cannot open " + path);

    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);) {
        if (!is_blank(l)) lines.push_back(l);
    }
    LineCursor cur(std::move(lines), std::string{});

    QplibInstance q;
    q.name = cur.word();
    cur.set_name(q.name);

    const std::string cls = cur.word();
    if (cls.size() != 3) cur.fail("expected a 3-letter classification, got \"" + cls + "\"");
    q.classification = {cls[0], cls[1], cls[2]};

    // Scope gate, deliberately before anything else is parsed: a quadratic
    // constraint type means extra per-constraint Hessian sections this reader
    // does not decode. Refusing here keeps the line cursor honest.
    const char ctype = cls[2];
    // N none · B bounds only · L linear · D diagonal quadratic ·
    // C convex quadratic · Q general quadratic. All six are decoded now.
    if (ctype != 'N' && ctype != 'B' && ctype != 'L' &&
        ctype != 'D' && ctype != 'C' && ctype != 'Q') {
        cur.fail(std::string("unknown constraint type '") + ctype +
                 "' in classification " + cls);
    }
    const bool has_quad_constraints = (ctype == 'D' || ctype == 'C' || ctype == 'Q');
    // Variable-type codes: 0 continuous, 1 integer, 2 binary.  Anything else
    // (a semicontinuous code, say) used to be static_cast into the enum and
    // flow on as an out-of-range value; refuse it by name instead.
    auto var_type_code = [&cur](f64 v) {
        if (v == 0.0) return QplibVarType::Continuous;
        if (v == 1.0) return QplibVarType::Integer;
        if (v == 2.0) return QplibVarType::Binary;
        cur.fail("unknown variable type code " + std::to_string(v));
    };
    const char vtype = cls[1];
    if (vtype != 'B' && vtype != 'C' && vtype != 'I' && vtype != 'G' && vtype != 'M') {
        cur.fail(std::string("unknown variable type '") + vtype + "' in classification " + cls);
    }

    std::string sense = cur.word();
    std::transform(sense.begin(), sense.end(), sense.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    q.maximize = sense.rfind("max", 0) == 0;

    q.n = cur.integer();
    // The m line, the A section and the constraint-bound sections are ALL
    // ABSENT when the instance has no general linear constraints (constraint
    // type 'N' or 'B' -- verified on real QBB files, where the H count
    // follows n directly). Reading an unconditional m here silently consumes
    // nnz(H) instead and desyncs everything after it.
    // General constraints exist for L (linear) AND for the quadratic types
    // D/C/Q -- those instances still have an m, an A and row bounds, they
    // just carry per-constraint Hessians as well. Only N (none) and B
    // (bounds only) omit the whole block.
    const bool has_general_constraints =
        (ctype == 'L' || ctype == 'D' || ctype == 'C' || ctype == 'Q');
    q.m = has_general_constraints ? cur.integer() : 0;
    if (q.n < 0 || q.m < 0) cur.fail("negative n or m");

    // ---- H: upper triangle of the objective Hessian ----------------------
    // ABSENT when the objective is linear. On an LCQ file the g block follows
    // m directly; reading an unconditional nnz(H) here consumes the g default
    // instead and desyncs everything after it.
    const Index nnz_h = (cls[0] == 'L') ? 0 : cur.integer();
    q.h_row.resize(static_cast<std::size_t>(nnz_h));
    q.h_col.resize(static_cast<std::size_t>(nnz_h));
    q.h_val.resize(static_cast<std::size_t>(nnz_h));
    for (Index t = 0; t < nnz_h; ++t) {
        Index r = 0, c = 0;
        f64 v = 0.0;
        cur.triplet(r, c, v);
        if (r < 1 || r > q.n || c < 1 || c > q.n) cur.fail("H entry out of range");
        q.h_row[static_cast<std::size_t>(t)] = r;
        q.h_col[static_cast<std::size_t>(t)] = c;
        q.h_val[static_cast<std::size_t>(t)] = v;
        if (r != c) rep.h_has_off_diagonal = true;
    }
    rep.n_h_entries = static_cast<std::size_t>(nnz_h);

    q.g = read_defaulted_vector(cur, q.n, "objective gradient");
    q.f_const = cur.real();

    // ---- per-constraint Hessians, between f and A ------------------------
    // One count, then that many (constraint, row, col, value) QUADRUPLES --
    // note four fields, not the three of every other matrix block here.
    if (has_quad_constraints) {
        const Index nnz_hc = cur.integer();
        q.hc_con.resize(static_cast<std::size_t>(nnz_hc));
        q.hc_row.resize(static_cast<std::size_t>(nnz_hc));
        q.hc_col.resize(static_cast<std::size_t>(nnz_hc));
        q.hc_val.resize(static_cast<std::size_t>(nnz_hc));
        for (Index t = 0; t < nnz_hc; ++t) {
            Index ci = 0, r = 0, c = 0;
            f64 v = 0.0;
            cur.quadruple(ci, r, c, v);
            if (ci < 1 || ci > q.m) cur.fail("H_i constraint index out of range");
            if (r < 1 || r > q.n || c < 1 || c > q.n)
                cur.fail("H_i entry out of range");
            q.hc_con[static_cast<std::size_t>(t)] = ci;
            q.hc_row[static_cast<std::size_t>(t)] = r;
            q.hc_col[static_cast<std::size_t>(t)] = c;
            q.hc_val[static_cast<std::size_t>(t)] = v;
        }
        rep.n_hc_entries = static_cast<std::size_t>(nnz_hc);
    }

    // ---- A and the constraint bounds, only when constraints exist --------
    if (has_general_constraints) {
        const Index nnz_a = cur.integer();
        q.a_row.resize(static_cast<std::size_t>(nnz_a));
        q.a_col.resize(static_cast<std::size_t>(nnz_a));
        q.a_val.resize(static_cast<std::size_t>(nnz_a));
        for (Index t = 0; t < nnz_a; ++t) {
            Index r = 0, c = 0;
            f64 v = 0.0;
            cur.triplet(r, c, v);
            if (r < 1 || r > q.m || c < 1 || c > q.n) cur.fail("A entry out of range");
            q.a_row[static_cast<std::size_t>(t)] = r;
            q.a_col[static_cast<std::size_t>(t)] = c;
            q.a_val[static_cast<std::size_t>(t)] = v;
        }
        rep.n_a_entries = static_cast<std::size_t>(nnz_a);
    }

    q.inf_bound = cur.real();
    if (has_general_constraints) {
        q.c_lo = read_defaulted_vector(cur, q.m, "constraint lower bound");
        q.c_hi = read_defaulted_vector(cur, q.m, "constraint upper bound");
    }

    // ---- variable bounds and types, conditional on the classification ----
    if (vtype == 'B') {
        // Omitted entirely: implied [0,1] and binary for every variable.
        q.x_lo.assign(static_cast<std::size_t>(q.n), 0.0);
        q.x_hi.assign(static_cast<std::size_t>(q.n), 1.0);
        q.var_type.assign(static_cast<std::size_t>(q.n), QplibVarType::Binary);
    } else {
        q.x_lo = read_defaulted_vector(cur, q.n, "variable lower bound");
        q.x_hi = read_defaulted_vector(cur, q.n, "variable upper bound");
        if (vtype == 'G' || vtype == 'M') {
            const f64 def = cur.real();
            const Index count = cur.integer();
            q.var_type.assign(static_cast<std::size_t>(q.n), var_type_code(def));
            for (Index k = 0; k < count; ++k) {
                Index idx = 0;
                f64 val = 0.0;
                cur.pair(idx, val);
                if (idx < 1 || idx > q.n) cur.fail("variable type index out of range");
                q.var_type[static_cast<std::size_t>(idx - 1)] = var_type_code(val);
            }
        } else {
            q.var_type.assign(static_cast<std::size_t>(q.n),
                              vtype == 'I' ? QplibVarType::Integer
                                           : QplibVarType::Continuous);
        }
    }

    // The file's sentinel is its own; normalise to real infinities so callers
    // never have to know what value this particular instance chose.
    const f64 inf = q.inf_bound;
    auto denorm = [inf](std::vector<f64>& v) {
        for (f64& x : v) {
            if (x >= inf) x = core::kPosInf;
            else if (x <= -inf) x = -core::kPosInf;
        }
    };
    denorm(q.c_lo);
    denorm(q.c_hi);
    denorm(q.x_lo);
    denorm(q.x_hi);

    // ---- trailer: starting point, starting duals, names ------------------
    // Nothing here is needed to build the model, so a failure is recorded,
    // not thrown (see QplibReadReport::trailer_error).  Reading it at all is
    // the point: a file consumed exactly to its last line is strong evidence
    // that no conditional section above was mis-sized.
    q.var_names.assign(static_cast<std::size_t>(q.n), std::string{});
    q.con_names.assign(static_cast<std::size_t>(q.m), std::string{});
    try {
        (void)read_defaulted_vector(cur, q.n, "starting point");
        if (has_general_constraints)
            (void)read_defaulted_vector(cur, q.m, "starting constraint duals");
        (void)read_defaulted_vector(cur, q.n, "starting bound duals");
        auto names = [&cur](std::vector<std::string>& out, Index len, const char* what) {
            const Index count = cur.integer();
            for (Index k = 0; k < count; ++k) {
                Index idx = 0;
                std::string nm;
                cur.name_pair(idx, nm);
                if (idx < 1 || idx > len)
                    cur.fail(std::string(what) + " name index out of range");
                out[static_cast<std::size_t>(idx - 1)] = nm;
            }
        };
        names(q.var_names, q.n, "variable");
        names(q.con_names, q.m, "constraint");
    } catch (const std::runtime_error& e) {
        rep.trailer_error = e.what();
    }

    rep.lines_consumed = cur.consumed();
    rep.lines_total = cur.total();
    return q;
}

}  // namespace sor::io
