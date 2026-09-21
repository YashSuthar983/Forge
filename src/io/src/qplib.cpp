#include "sor/io/qplib.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace sor::io {
namespace {

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
        try {
            return std::stod(w);
        } catch (...) {
            fail("expected a number, found \"" + w + "\"");
        }
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
        double x = 0.0;
        if (!(is >> a >> b >> x)) fail("expected \"row col value\", found \"" + l + "\"");
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
        double x = 0.0;
        if (!(is >> a >> b >> d >> x))
            fail("expected \"constraint row col value\", found \"" + l + "\"");
        ci = static_cast<Index>(a);
        r = static_cast<Index>(b);
        c = static_cast<Index>(d);
        v = x;
    }

    // One (index, value) pair.
    void pair(Index& i, f64& v) {
        const std::string& l = next();
        std::istringstream is(l);
        long long a = 0;
        double x = 0.0;
        if (!(is >> a >> x)) fail("expected \"index value\", found \"" + l + "\"");
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
            const Index def = cur.integer();
            const Index count = cur.integer();
            q.var_type.assign(static_cast<std::size_t>(q.n),
                              static_cast<QplibVarType>(def));
            for (Index k = 0; k < count; ++k) {
                Index idx = 0;
                f64 val = 0.0;
                cur.pair(idx, val);
                if (idx < 1 || idx > q.n) cur.fail("variable type index out of range");
                q.var_type[static_cast<std::size_t>(idx - 1)] =
                    static_cast<QplibVarType>(static_cast<int>(val));
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

    rep.lines_consumed = cur.consumed();
    return q;
}

}  // namespace sor::io
