#include "sor/core/options.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace sor::core {
namespace {

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

// Print a real without trailing zero noise, so a listed default reads the way a
// user would type it back in.
std::string show_real(double v) {
    if (v == static_cast<double>(static_cast<long long>(v)) && std::fabs(v) < 1e15)
        return std::to_string(static_cast<long long>(v));
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

}  // namespace

bool parse_flag(const std::string& key, const std::string& value, bool& dst, std::string& err) {
    const std::string v = trim(value);
    if (v == "1" || v == "true" || v == "on" || v == "yes") { dst = true; return true; }
    if (v == "0" || v == "false" || v == "off" || v == "no") { dst = false; return true; }
    err = "option '" + key + "' takes 0/1 (or on/off), got '" + value + "'";
    return false;
}

bool parse_real(const std::string& key, const std::string& value, f64& dst, std::string& err) {
    const std::string v = trim(value);
    char* end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    if (v.empty() || end == nullptr || *end != '\0') {
        err = "option '" + key + "' takes a number, got '" + value + "'";
        return false;
    }
    dst = d;
    return true;
}

bool parse_int(const std::string& key, const std::string& value, long long& dst, std::string& err) {
    f64 d = 0.0;
    if (!parse_real(key, value, d, err)) return false;
    // Reject a fractional value rather than truncating it: a caller who wrote
    // 2.5 meant something, and silently reading 2 hides the mistake.
    if (d != std::floor(d)) {
        err = "option '" + key + "' takes a whole number, got '" + value + "'";
        return false;
    }
    dst = static_cast<long long>(d);
    return true;
}

OptionBinding bind_flag(std::string name, bool& dst, std::string help) {
    OptionBinding b;
    b.name = std::move(name); b.type = "bool"; b.help = std::move(help);
    const std::string key = b.name;
    b.set = [key, &dst](const std::string& v, std::string& e) { return parse_flag(key, v, dst, e); };
    b.show = [&dst]() { return dst ? std::string("on") : std::string("off"); };
    return b;
}

OptionBinding bind_real(std::string name, f64& dst, std::string help) {
    OptionBinding b;
    b.name = std::move(name); b.type = "real"; b.help = std::move(help);
    const std::string key = b.name;
    b.set = [key, &dst](const std::string& v, std::string& e) { return parse_real(key, v, dst, e); };
    b.show = [&dst]() { return show_real(static_cast<double>(dst)); };
    return b;
}

OptionBinding bind_double(std::string name, double& dst, std::string help) {
    OptionBinding b;
    b.name = std::move(name); b.type = "real"; b.help = std::move(help);
    const std::string key = b.name;
    b.set = [key, &dst](const std::string& v, std::string& e) {
        f64 d = 0.0;
        if (!parse_real(key, v, d, e)) return false;
        dst = static_cast<double>(d);
        return true;
    };
    b.show = [&dst]() { return show_real(dst); };
    return b;
}

namespace {
// One integral binder, reused for every width. The clamp is deliberate: an
// out-of-range value is refused, not wrapped, because a wrapped limit turns
// into a behaviour nobody asked for.
template <typename T>
OptionBinding bind_integral(std::string name, T& dst, std::string help, long long lo, long long hi) {
    OptionBinding b;
    b.name = std::move(name); b.type = "int"; b.help = std::move(help);
    const std::string key = b.name;
    b.set = [key, &dst, lo, hi](const std::string& v, std::string& e) {
        long long n = 0;
        if (!parse_int(key, v, n, e)) return false;
        if (n < lo || n > hi) {
            e = "option '" + key + "' is out of range";
            return false;
        }
        dst = static_cast<T>(n);
        return true;
    };
    b.show = [&dst]() { return std::to_string(static_cast<long long>(dst)); };
    return b;
}
}  // namespace

OptionBinding bind_int(std::string name, int& dst, std::string help) {
    return bind_integral(std::move(name), dst, std::move(help), -2147483648LL, 2147483647LL);
}
OptionBinding bind_i64(std::string name, std::int64_t& dst, std::string help) {
    return bind_integral(std::move(name), dst, std::move(help), -(1LL << 62), 1LL << 62);
}
OptionBinding bind_u64(std::string name, std::uint64_t& dst, std::string help) {
    return bind_integral(std::move(name), dst, std::move(help), 0LL, 1LL << 62);
}
OptionBinding bind_size(std::string name, std::size_t& dst, std::string help) {
    return bind_integral(std::move(name), dst, std::move(help), 0LL, 1LL << 62);
}

bool apply_option(const std::vector<OptionBinding>& table, const std::string& kv, std::string& err) {
    const auto eq = kv.find('=');
    if (eq == std::string::npos) {
        err = "expected key=value, got '" + kv + "'";
        return false;
    }
    const std::string key = trim(kv.substr(0, eq)), val = kv.substr(eq + 1);
    const auto it = std::find_if(table.begin(), table.end(),
                                 [&](const OptionBinding& b) { return b.name == key; });
    if (it == table.end()) {
        err = "unknown option '" + key + "'";
        return false;
    }
    return it->set(val, err);
}

std::string format_options(const std::vector<OptionBinding>& table) {
    std::size_t w = 0;
    for (const auto& b : table) w = std::max(w, b.name.size());
    std::ostringstream os;
    for (const auto& b : table) {
        os << "  " << b.name << std::string(w - b.name.size() + 2, ' ')
           << b.type << std::string(b.type.size() >= 4 ? 2 : 6 - b.type.size(), ' ')
           << "default " << b.show() << "\n"
           << "      " << b.help << "\n";
    }
    return os.str();
}

}  // namespace sor::core
