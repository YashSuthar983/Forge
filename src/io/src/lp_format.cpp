#include "sor/io/lp_format.hpp"

#include "sor/io/gzip.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace sor::io {

using core::f64;
using core::Index;
using model::kInf;

namespace {

using RClock = std::chrono::steady_clock;

enum class TokKind { Num, Name, Op, End };

struct Tok {
    TokKind kind = TokKind::End;
    std::string text;   // Name spelling, or Op in canonical form
    f64 num = 0.0;
    int line = 0;
    bool first_on_line = false;
};

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool name_start(char c) {
    if (std::isalpha(static_cast<unsigned char>(c))) return true;
    switch (c) {
        case '_': case '!': case '"': case '#': case '$': case '%': case '&':
        case '(': case ')': case ',': case ';': case '?': case '@':
        case '\'': case '{': case '}': case '~':
            return true;
        default:
            return false;
    }
}
bool name_char(char c) {
    return name_start(c) || std::isdigit(static_cast<unsigned char>(c)) || c == '.';
}

[[noreturn]] void fail(int line, const std::string& msg) {
    throw std::runtime_error("LP format line " + std::to_string(line) + ": " + msg);
}

std::vector<Tok> tokenize(const std::string& text, std::size_t& lines) {
    std::vector<Tok> out;
    int line = 1;
    bool line_start = true;
    std::size_t i = 0;
    const std::size_t n = text.size();
    auto push = [&](TokKind k, std::string t, f64 v) {
        Tok tok;
        tok.kind = k;
        tok.text = std::move(t);
        tok.num = v;
        tok.line = line;
        tok.first_on_line = line_start;
        line_start = false;
        out.push_back(std::move(tok));
    };
    while (i < n) {
        const char c = text[i];
        if (c == '\n') { ++line; line_start = true; ++i; continue; }
        if (c == '\r' || c == ' ' || c == '\t') { ++i; continue; }
        if (c == '\\') {                       // comment to end of line
            while (i < n && text[i] != '\n') ++i;
            continue;
        }
        const bool digit = std::isdigit(static_cast<unsigned char>(c)) != 0;
        const bool dot_num = c == '.' && i + 1 < n &&
                             std::isdigit(static_cast<unsigned char>(text[i + 1]));
        if (digit || dot_num) {
            std::size_t j = i;
            while (j < n && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
            if (j < n && text[j] == '.') {
                ++j;
                while (j < n && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
            }
            // An exponent only counts when digits follow it, so "3e" is the
            // number 3 followed by the name e.
            if (j < n && (text[j] == 'e' || text[j] == 'E')) {
                std::size_t k = j + 1;
                if (k < n && (text[k] == '+' || text[k] == '-')) ++k;
                if (k < n && std::isdigit(static_cast<unsigned char>(text[k]))) {
                    while (k < n && std::isdigit(static_cast<unsigned char>(text[k]))) ++k;
                    j = k;
                }
            }
            const std::string lex = text.substr(i, j - i);
            const f64 number = std::strtod(lex.c_str(), nullptr);
            if (!std::isfinite(number)) fail(line, "numeric literal overflows: " + lex);
            if (number == 0.0) {
                const auto exp = lex.find_first_of("eE");
                const auto mantissa = lex.substr(0, exp);
                if (mantissa.find_first_of("123456789") != std::string::npos)
                    fail(line, "numeric literal underflows to zero: " + lex);
            }
            push(TokKind::Num, lex, number);
            i = j;
            continue;
        }
        if (name_start(c)) {
            std::size_t j = i + 1;
            while (j < n && name_char(text[j])) ++j;
            push(TokKind::Name, text.substr(i, j - i), 0.0);
            i = j;
            continue;
        }
        // Operators.
        auto two = [&](const char* s) { return i + 1 < n && text[i] == s[0] && text[i + 1] == s[1]; };
        if (two("<=") || two("=<")) { push(TokKind::Op, "<=", 0); i += 2; continue; }
        if (two(">=") || two("=>")) { push(TokKind::Op, ">=", 0); i += 2; continue; }
        if (two("->")) fail(line, "indicator constraints are not supported");
        if (c == '<') { push(TokKind::Op, "<=", 0); ++i; continue; }
        if (c == '>') { push(TokKind::Op, ">=", 0); ++i; continue; }
        if (c == '=') { push(TokKind::Op, "=", 0); ++i; continue; }
        if (c == '+' || c == '-' || c == '*' || c == ':') {
            push(TokKind::Op, std::string(1, c), 0);
            ++i;
            continue;
        }
        if (c == '[' || c == ']' || c == '^')
            fail(line, "quadratic terms are not supported in .lp input "
                       "(use a .qps or .qplib file for a quadratic model)");
        fail(line, std::string("unexpected character '") + c + "'");
    }
    lines = static_cast<std::size_t>(line);
    Tok end;
    end.kind = TokKind::End;
    end.line = line;
    end.first_on_line = true;
    out.push_back(end);
    return out;
}

enum class Section { None, Objective, Constraints, Bounds, General, Binary, End };

struct Expr {
    std::vector<std::pair<int, f64>> terms;
    f64 constant = 0.0;
};

class Parser {
public:
    Parser(std::vector<Tok> toks, const MpsReadOptions& opt) : t_(std::move(toks)), opt_(opt) {}

    model::LpProblem run(MpsReadReport& rep, std::size_t lines);

private:
    std::vector<Tok> t_;
    std::size_t pos_ = 0;
    const MpsReadOptions& opt_;

    std::vector<std::string> col_names_;
    std::unordered_map<std::string, int> col_index_;
    std::vector<f64> lo_, hi_, obj_;
    std::vector<bool> integer_;
    std::vector<bool> binary_, explicit_lo_, explicit_hi_;

    std::vector<std::string> row_names_;
    std::vector<f64> row_lo_, row_hi_;
    std::vector<Index> tri_r_, tri_c_;
    std::vector<f64> tri_v_;
    f64 offset_ = 0.0;
    bool maximize_ = false;
    std::string obj_name_;
    std::size_t dropped_ = 0;
    f64 largest_dropped_ = 0.0;

    const Tok& cur() const { return t_[pos_]; }
    const Tok& peek(std::size_t k = 1) const { return t_[std::min(pos_ + k, t_.size() - 1)]; }
    void advance() { if (pos_ + 1 < t_.size()) ++pos_; }
    bool is_op(const Tok& tk, const char* s) const { return tk.kind == TokKind::Op && tk.text == s; }
    bool is_relation(const Tok& tk) const {
        return tk.kind == TokKind::Op && (tk.text == "<=" || tk.text == ">=" || tk.text == "=");
    }
    static bool is_infinity_name(const Tok& tk) {
        if (tk.kind != TokKind::Name) return false;
        const std::string s = lower(tk.text);
        return s == "inf" || s == "infinity";
    }

    // Section keywords count only as the first token of a line, so a variable
    // called "bound" in the middle of a row is not mistaken for a header.
    Section section_at(std::size_t i, std::size_t& ntok, bool& is_max) const;
    bool at_section() const { std::size_t n; bool m; return section_at(pos_, n, m) != Section::None; }

    int var(const std::string& name);
    Expr parse_expr(bool same_line_signs_only);
    void add_term(Expr& e, f64 sign, bool same_line_only);

    void parse_objective(bool is_max);
    void parse_constraint();
    void parse_bound();
    f64 read_value();
    void apply_bound(int j, const std::string& rel, f64 v);
};

Section Parser::section_at(std::size_t i, std::size_t& ntok, bool& is_max) const {
    ntok = 1;
    is_max = false;
    const Tok& tk = t_[i];
    if (tk.kind == TokKind::End) { ntok = 0; return Section::End; }
    if (tk.kind != TokKind::Name || !tk.first_on_line) return Section::None;
    const std::string s = lower(tk.text);
    const Tok& nx = t_[std::min(i + 1, t_.size() - 1)];
    // A row/objective label may happen to spell a section keyword.
    if (is_op(nx, ":")) return Section::None;
    const std::string n1 = nx.kind == TokKind::Name ? lower(nx.text) : std::string();
    if (s == "minimize" || s == "minimise" || s == "minimum" || s == "min") return Section::Objective;
    if (s == "maximize" || s == "maximise" || s == "maximum" || s == "max") { is_max = true; return Section::Objective; }
    if (s == "subject" && n1 == "to") { ntok = 2; return Section::Constraints; }
    if (s == "such" && n1 == "that") { ntok = 2; return Section::Constraints; }
    if (s == "st" || s == "s.t." || s == "s.t") return Section::Constraints;
    if (s == "bounds" || s == "bound") return Section::Bounds;
    if (s == "general" || s == "generals" || s == "gen" || s == "integer" || s == "integers")
        return Section::General;
    if (s == "binary" || s == "binaries" || s == "bin") return Section::Binary;
    if (s == "end") return Section::End;
    if (s == "semi" || s == "sc" || s == "sos" || s == "sos1" || s == "sos2" ||
        (s == "lazy" && n1 == "constraints") || (s == "user" && n1 == "cuts"))
        fail(tk.line, "section '" + tk.text + "' is not supported (semi-continuous, SOS, lazy "
                      "constraints and user cuts are rejected rather than ignored)");
    return Section::None;
}

int Parser::var(const std::string& name) {
    auto it = col_index_.find(name);
    if (it != col_index_.end()) return it->second;
    const int j = static_cast<int>(col_names_.size());
    col_index_.emplace(name, j);
    col_names_.push_back(name);
    lo_.push_back(0.0);
    hi_.push_back(kInf);
    obj_.push_back(0.0);
    integer_.push_back(false);
    binary_.push_back(false);
    explicit_lo_.push_back(false);
    explicit_hi_.push_back(false);
    return j;
}

// One term: [number [*]] name | number | name | inf. A coefficient binds to a
// name across line breaks on the lhs/objective. On the rhs, "... >= 2"
// followed by a name on the next line starts a new row instead of "2 name".
void Parser::add_term(Expr& e, f64 sign, bool same_line_only) {
    const Tok tk = cur();
    if (tk.kind == TokKind::Num) {
        advance();
        f64 coef = tk.num;
        if (is_op(cur(), "*") && cur().line == tk.line) {
            advance();
            if (cur().kind != TokKind::Name || is_infinity_name(cur()))
                fail(cur().line, "expected a variable after '*'");
        }
        if (cur().kind == TokKind::Name && !is_infinity_name(cur()) &&
            (cur().line == tk.line || (!same_line_only && !at_section() && !is_op(peek(), ":")))) {
            const int j = var(cur().text);
            advance();
            e.terms.emplace_back(j, sign * coef);
            if (is_op(cur(), "*")) fail(cur().line, "products of variables are not supported");
        } else {
            const f64 previous = e.constant;
            e.constant += sign * coef;
            if (std::isfinite(previous) && !std::isfinite(e.constant))
                fail(tk.line, "expression constant overflows");
        }
        return;
    }
    if (tk.kind == TokKind::Name) {
        advance();
        if (is_infinity_name(tk)) { e.constant += sign * kInf; return; }
        const int j = var(tk.text);
        e.terms.emplace_back(j, sign);
        if (is_op(cur(), "*")) fail(cur().line, "products of variables are not supported");
        return;
    }
    fail(tk.line, "expected a term, found '" + (tk.kind == TokKind::End ? std::string("end of file") : tk.text) + "'");
}

Expr Parser::parse_expr(bool same_line_signs_only) {
    Expr e;
    bool first = true;
    while (true) {
        f64 sign = 1.0;
        bool saw_sign = false;
        while (cur().kind == TokKind::Op && (cur().text == "+" || cur().text == "-")) {
            // In a right-hand side a sign that starts a new line begins the
            // next row, not another term.
            if (same_line_signs_only && !first && cur().first_on_line) break;
            if (cur().text == "-") sign = -sign;
            saw_sign = true;
            advance();
        }
        if (same_line_signs_only && !first && !saw_sign) break;
        if (!first && !saw_sign) break;
        if (cur().kind != TokKind::Num && cur().kind != TokKind::Name)
            fail(cur().line, "expected a number or variable");
        add_term(e, sign, same_line_signs_only);
        first = false;
        const bool more = cur().kind == TokKind::Op && (cur().text == "+" || cur().text == "-") &&
                          !(same_line_signs_only && cur().first_on_line);
        if (!more) break;
    }
    return e;
}

void Parser::parse_objective(bool is_max) {
    maximize_ = is_max;
    if (cur().kind == TokKind::Name && is_op(peek(), ":")) {
        obj_name_ = cur().text;
        advance();
        advance();
    }
    if (at_section()) return;                          // empty objective
    const Expr e = parse_expr(false);
    if (!std::isfinite(e.constant)) fail(cur().line, "objective constant must be finite");
    if (is_relation(cur())) fail(cur().line, "the objective must be a linear expression, not a relation");
    for (const auto& [j, c] : e.terms) {
        auto& coef = obj_[static_cast<std::size_t>(j)];
        coef += c;
        if (!std::isfinite(coef)) fail(cur().line, "objective coefficient overflows");
    }
    offset_ += e.constant;
}

void Parser::parse_constraint() {
    std::string name;
    if (cur().kind == TokKind::Name && is_op(peek(), ":")) {
        name = cur().text;
        advance();
        advance();
    }
    const int line = cur().line;
    const Expr e1 = parse_expr(false);
    if (!is_relation(cur())) fail(cur().line, "expected <=, >= or = in a constraint");
    const std::string s1 = cur().text;
    advance();
    const Expr e2 = parse_expr(true);

    f64 lo = -kInf, hi = kInf;
    std::map<int, f64> lin;

    if (is_relation(cur()) && !cur().first_on_line) {   // ranged: a <= expr <= b
        const std::string s2 = cur().text;
        advance();
        const Expr e3 = parse_expr(true);
        if (!e1.terms.empty() || !e3.terms.empty())
            fail(line, "a ranged constraint needs numbers on both outer sides");
        if (s1 == "=" || s2 == "=" || s1 != s2)
            fail(line, "a ranged constraint must use the same direction twice (a <= expr <= b)");
        for (const auto& [j, c] : e2.terms) lin[j] += c;
        if (!std::isfinite(e2.constant)) fail(line, "ranged expression constant must be finite");
        if (s1 == "<=") { lo = e1.constant - 0.0; hi = e3.constant; }
        else            { hi = e1.constant;       lo = e3.constant; }
        if (std::isfinite(lo)) {
            lo -= e2.constant;
            if (!std::isfinite(lo)) fail(line, "constraint side overflows");
        }
        if (std::isfinite(hi)) {
            hi -= e2.constant;
            if (!std::isfinite(hi)) fail(line, "constraint side overflows");
        }
    } else {
        for (const auto& [j, c] : e1.terms) lin[j] += c;
        for (const auto& [j, c] : e2.terms) lin[j] -= c;
        const f64 rhs = e2.constant - e1.constant;
        if (std::isfinite(e1.constant) && std::isfinite(e2.constant) && !std::isfinite(rhs))
            fail(line, "constraint side overflows");
        if (s1 == "<=")      { hi = rhs; }
        else if (s1 == ">=") { lo = rhs; }
        else                 { lo = hi = rhs; }
    }
    if (std::isnan(lo) || std::isnan(hi) || lo == kInf || hi == -kInf)
        fail(line, "invalid infinite constraint side");
    if (lo > hi) fail(line, "ranged constraint lower side exceeds upper side");

    const Index row = static_cast<Index>(row_names_.size());
    row_names_.push_back(name.empty() ? "R" + std::to_string(row + 1) : name);
    row_lo_.push_back(lo);
    row_hi_.push_back(hi);
    for (const auto& [j, c] : lin) {
        if (!std::isfinite(c)) fail(line, "constraint coefficient overflows");
        if (c == 0.0) continue;
        if (opt_.small_matrix_value > 0.0 && std::fabs(c) <= opt_.small_matrix_value) {
            ++dropped_;
            largest_dropped_ = std::max(largest_dropped_, std::fabs(c));
            continue;
        }
        tri_r_.push_back(row);
        tri_c_.push_back(static_cast<Index>(j));
        tri_v_.push_back(c);
    }
}

f64 Parser::read_value() {
    f64 sign = 1.0;
    bool signed_ = false;
    while (cur().kind == TokKind::Op && (cur().text == "+" || cur().text == "-")) {
        if (cur().text == "-") sign = -sign;
        signed_ = true;
        advance();
    }
    (void)signed_;
    f64 v;
    if (cur().kind == TokKind::Num) v = cur().num;
    else if (is_infinity_name(cur())) v = kInf;
    else fail(cur().line, "expected a number or inf in a bound");
    advance();
    v *= sign;
    return v;
}

void Parser::apply_bound(int j, const std::string& rel, f64 v) {
    const auto sj = static_cast<std::size_t>(j);
    if (rel != ">=") { hi_[sj] = v; explicit_hi_[sj] = true; }
    if (rel != "<=") { lo_[sj] = v; explicit_lo_[sj] = true; }
}

void Parser::parse_bound() {
    const Tok first = cur();
    if (first.kind == TokKind::Name && !is_infinity_name(first)) {
        const int j = var(first.text);
        advance();
        if (cur().kind == TokKind::Name && lower(cur().text) == "free" && !cur().first_on_line) {
            advance();
            lo_[static_cast<std::size_t>(j)] = -kInf;
            hi_[static_cast<std::size_t>(j)] = kInf;
            explicit_lo_[static_cast<std::size_t>(j)] = true;
            explicit_hi_[static_cast<std::size_t>(j)] = true;
            return;
        }
        if (!is_relation(cur())) fail(cur().line, "expected <=, >=, = or 'free' after '" + first.text + "'");
        const std::string rel = cur().text;
        advance();
        apply_bound(j, rel, read_value());
        return;
    }
    // number <= name [<= number]
    const f64 v1 = read_value();
    if (!is_relation(cur())) fail(cur().line, "expected <=, >= or = in a bound");
    const std::string r1 = cur().text;
    advance();
    if (cur().kind != TokKind::Name || is_infinity_name(cur()))
        fail(cur().line, "expected a variable name in a bound");
    const int j = var(cur().text);
    advance();
    // "v <= x" bounds x from below, "v >= x" from above.
    apply_bound(j, r1 == "<=" ? ">=" : r1 == ">=" ? "<=" : "=", v1);
    if (is_relation(cur()) && !cur().first_on_line) {
        const std::string r2 = cur().text;
        if (r1 == "=" || r2 != r1)
            fail(first.line, "a ranged bound must use the same inequality direction twice");
        advance();
        apply_bound(j, r2, read_value());
    }
}

model::LpProblem Parser::run(MpsReadReport& rep, std::size_t lines) {
    const auto t_parse = RClock::now();
    Section current = Section::None;
    bool seen_objective = false;
    bool seen_end = false;

    while (true) {
        std::size_t ntok = 0;
        bool is_max = false;
        const Section s = section_at(pos_, ntok, is_max);
        if (s == Section::End) {
            if (cur().kind != TokKind::End) {
                seen_end = true;
                advance();
                if (cur().kind != TokKind::End) fail(cur().line, "unexpected content after End");
            }
            break;
        }
        if (s != Section::None) {
            for (std::size_t k = 0; k < ntok; ++k) advance();
            current = s;
            if (s == Section::Objective) {
                if (seen_objective) fail(t_[pos_ - 1].line, "more than one objective section");
                seen_objective = true;
                parse_objective(is_max);
            }
            continue;
        }
        switch (current) {
            case Section::None:
                fail(cur().line, "expected Minimize or Maximize before '" + cur().text + "'");
            case Section::Objective:
                fail(cur().line, "unexpected '" + cur().text + "' after the objective (missing 'Subject To'?)");
            case Section::Constraints: parse_constraint(); break;
            case Section::Bounds:      parse_bound(); break;
            case Section::General:
            case Section::Binary: {
                if (cur().kind != TokKind::Name)
                    fail(cur().line, "expected a variable name in the integer section");
                const int j = var(cur().text);
                advance();
                integer_[static_cast<std::size_t>(j)] = true;
                if (current == Section::Binary) {
                    binary_[static_cast<std::size_t>(j)] = true;
                }
                break;
            }
            case Section::End: break;
        }
    }
    if (!seen_objective) fail(1, "no Minimize or Maximize section");
    if (!seen_end) fail(cur().line, "missing End section");

    const Index n_cols = static_cast<Index>(col_names_.size());
    const Index n_rows = static_cast<Index>(row_names_.size());
    for (std::size_t j = 0; j < col_names_.size(); ++j) {
        // Binary defaults apply only to sides not explicitly set by Bounds.
        // In particular, `b = 1` must remain fixed after `Binary b`.
        if (binary_[j]) {
            if (!explicit_lo_[j]) lo_[j] = 0.0;
            if (!explicit_hi_[j]) hi_[j] = 1.0;
            if (lo_[j] < 0.0 || hi_[j] > 1.0)
                rep.warnings.push_back("binary variable '" + col_names_[j] +
                                       "' has explicit bounds outside [0,1]");
        }
        if (lo_[j] == kInf || hi_[j] == -kInf)
            fail(1, "invalid infinite bound for variable '" + col_names_[j] + "'");
        if (lo_[j] > hi_[j])
            throw std::runtime_error(
                "LP format: variable '" + col_names_[j] + "' has lower bound " +
                std::to_string(lo_[j]) + " above upper bound " + std::to_string(hi_[j]) +
                " (a variable with only a negative upper bound keeps the default lower bound 0)");
    }
    const auto ms_parse = std::chrono::duration<double, std::milli>(RClock::now() - t_parse).count();
    const auto t_assemble = RClock::now();

    model::LpProblem p;
    p.name = obj_name_.empty() ? std::string("LP") : obj_name_;
    p.maximize = maximize_;
    p.A = sparse::from_triplets(n_rows, n_cols, tri_r_, tri_c_, tri_v_);
    p.c = obj_;
    p.obj_offset = offset_;
    p.col_lo = lo_;
    p.col_hi = hi_;
    p.col_names = col_names_;
    p.row_lo = row_lo_;
    p.row_hi = row_hi_;
    p.row_names = row_names_;
    p.is_integer.assign(integer_.begin(), integer_.end());

    const std::size_t integer_columns = p.n_integer();
    if (opt_.relax_integrality && integer_columns > 0) {
        p.is_integer.assign(p.is_integer.size(), false);
        rep.relaxed_integrality = true;
    }
    p.validate();

    rep.lines_read = lines;
    rep.n_rows = static_cast<std::size_t>(n_rows);
    rep.n_cols = static_cast<std::size_t>(n_cols);
    rep.nnz = p.nnz();
    rep.n_integer = integer_columns;
    rep.had_objsense_max = maximize_;
    rep.small_values_dropped = dropped_;
    rep.largest_small_value_dropped = largest_dropped_;
    if (dropped_ > 0)
        rep.warnings.push_back("dropped " + std::to_string(dropped_) +
                               " matrix coefficient(s) at or below " +
                               std::to_string(opt_.small_matrix_value));
    rep.ms_parse = ms_parse;
    rep.ms_assemble = std::chrono::duration<double, std::milli>(RClock::now() - t_assemble).count();
    return p;
}

}  // namespace

bool has_lp_extension(const std::string& path) {
    const std::string s = lower(path);
    auto ends = [&](const char* suf) {
        const std::string t(suf);
        return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
    };
    return ends(".lp") || ends(".lp.gz");
}

model::LpProblem read_lp(std::istream& in, MpsReadReport& rep, const MpsReadOptions& opt) {
    std::ostringstream ss;
    ss << in.rdbuf();
    std::size_t lines = 0;
    Parser parser(tokenize(ss.str(), lines), opt);
    return parser.run(rep, lines);
}

model::LpProblem read_lp_file(const std::string& path, MpsReadReport& rep, const MpsReadOptions& opt) {
    if (file_has_gzip_magic(path)) {
        std::istringstream in(read_maybe_gzip_file(path));
        rep.used_gzip = true;
        return read_lp(in, rep, opt);
    }
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open LP file: " + path);
    return read_lp(in, rep, opt);
}

}  // namespace sor::io
