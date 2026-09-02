#include "sor/io/qps.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace sor::io {
namespace {

std::string upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream in(line);
    std::string tok;
    while (in >> tok) out.push_back(tok);
    return out;
}

}  // namespace

QpsProblem read_qps_file(const std::string& path, QpsReadReport& rep,
                         const MpsReadOptions& opt) {
    // Linear part via the existing MPS reader (QUADOBJ is warned/ignored there).
    MpsReadReport mrep;
    QpsProblem qp;
    qp.linear = read_mps_file_auto(path, mrep, opt.strict);
    static_cast<MpsReadReport&>(rep) = mrep;
    // QUADOBJ is re-parsed below; drop the LP-only ignore warning.
    rep.warnings.erase(
        std::remove_if(rep.warnings.begin(), rep.warnings.end(),
                       [](const std::string& w) {
                           return w.find("QUADOBJ") != std::string::npos &&
                                  w.find("ignored") != std::string::npos;
                       }),
        rep.warnings.end());
    qp.q_diag.assign(static_cast<std::size_t>(qp.linear.n_cols()), 0.0);

    // Map column names → indices.
    std::unordered_map<std::string, core::Index> col_of;
    for (core::Index j = 0; j < qp.linear.n_cols(); ++j) {
        if (static_cast<std::size_t>(j) < qp.linear.col_names.size() &&
            !qp.linear.col_names[static_cast<std::size_t>(j)].empty())
            col_of[qp.linear.col_names[static_cast<std::size_t>(j)]] = j;
        else
            col_of["X" + std::to_string(j + 1)] = j;
    }

    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    bool in_quad = false;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) continue;
        // Strip CR.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        const bool header = !std::isspace(static_cast<unsigned char>(line[0]));
        if (header) {
            const auto f = split_ws(line);
            if (f.empty()) continue;
            const std::string key = upper(f[0]);
            if (key == "QUADOBJ" || key == "QMATRIX" || key == "QSECTION") {
                in_quad = true;
                continue;
            }
            if (key == "ENDATA") break;
            if (in_quad) in_quad = false;  // left the section
            continue;
        }
        if (!in_quad) continue;

        const auto f = split_ws(line);
        if (f.size() < 3) continue;
        // Forms:  col row value   OR   set col row value
        std::size_t k = (f.size() >= 4) ? 1 : 0;
        for (; k + 2 < f.size() || (k + 1 < f.size() && f.size() - k >= 3);) {
            if (k + 2 >= f.size()) break;
            const std::string& c1 = f[k];
            const std::string& c2 = f[k + 1];
            const double v = std::strtod(f[k + 2].c_str(), nullptr);
            auto i1 = col_of.find(c1);
            auto i2 = col_of.find(c2);
            if (i1 == col_of.end() || i2 == col_of.end()) {
                rep.warnings.push_back("QUADOBJ unknown column at line " +
                                       std::to_string(line_no));
                k += 3;
                continue;
            }
            ++rep.n_quad_entries;
            if (i1->second != i2->second) {
                rep.has_off_diagonal = true;
                rep.warnings.push_back(
                    "QUADOBJ off-diagonal entry ignored by diagonal QP engine");
            } else {
                qp.q_diag[static_cast<std::size_t>(i1->second)] += v;
            }
            k += 3;
            break;  // one triple per line is the common case
        }
    }
    return qp;
}

}  // namespace sor::io
