// SOR — one binding table per option struct, used for three jobs at once:
// setting an option from "key=value", listing what exists with its default, and
// generating the documentation. A single table means the three can never drift
// apart, which is the failure mode a hand-written flag list always reaches: the
// code grows an option, the help text does not, and the documented flag set
// stops being the real one.
#pragma once

#include "sor/core/result.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sor::core {

struct OptionBinding {
    std::string name;
    std::string type;    // "bool" | "int" | "real" — what the value must parse as
    std::string help;    // one line; what it does and, where it matters, why
    std::function<bool(const std::string& value, std::string& err)> set;
    std::function<std::string()> show;   // the value currently held
};

// Parsers shared by every binding, so "on"/"off"/"1" mean the same thing for
// every engine and a bad value reports the same way.
bool parse_flag(const std::string& key, const std::string& value, bool& dst, std::string& err);
bool parse_real(const std::string& key, const std::string& value, f64& dst, std::string& err);
bool parse_int(const std::string& key, const std::string& value, long long& dst, std::string& err);

// Bind one field. The lambdas capture by reference, so a binding table is only
// valid while its options object is.
OptionBinding bind_flag(std::string name, bool& dst, std::string help);
OptionBinding bind_real(std::string name, f64& dst, std::string help);
OptionBinding bind_double(std::string name, double& dst, std::string help);
OptionBinding bind_int(std::string name, int& dst, std::string help);
OptionBinding bind_i64(std::string name, std::int64_t& dst, std::string help);
OptionBinding bind_u64(std::string name, std::uint64_t& dst, std::string help);
OptionBinding bind_size(std::string name, std::size_t& dst, std::string help);

// Apply "key=value" against a table. False, with `err`, for an unknown key or a
// value of the wrong shape; an unknown key lists nothing, because a silently
// ignored option is worse than a refused one -- the caller believes it tuned
// something it did not.
bool apply_option(const std::vector<OptionBinding>& table, const std::string& kv, std::string& err);

// Every option, one per line: "  name   type   default   help".
std::string format_options(const std::vector<OptionBinding>& table);

}  // namespace sor::core
