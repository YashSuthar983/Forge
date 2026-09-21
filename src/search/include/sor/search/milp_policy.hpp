// SOR - MILP control policy (latest improved vs classical ablation).
//
// Default product path is Latest (2025-2026 algorithms). Classical is debug /
// ablation only: plain reliability branching, efficacy-only cut scoring,
// static LNS, graph-only conflict, root-only separation.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace sor::search {

enum class MilpPolicy : std::uint8_t {
    Latest = 0,
    Classical = 1,
};

inline constexpr MilpPolicy kDefaultMilpPolicy = MilpPolicy::Latest;

inline const char* milp_policy_name(MilpPolicy p) noexcept {
    switch (p) {
    case MilpPolicy::Latest: return "latest";
    case MilpPolicy::Classical: return "classical";
    }
    return "latest";
}

// Accepts "latest" / "classical" (case-insensitive). Returns false on unknown.
inline bool parse_milp_policy(std::string_view s, MilpPolicy& out) {
    auto lower = [](char c) {
        return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    };
    std::string t;
    t.reserve(s.size());
    for (char c : s) t.push_back(lower(c));
    if (t == "latest" || t == "default" || t == "improved") {
        out = MilpPolicy::Latest;
        return true;
    }
    if (t == "classical" || t == "classic" || t == "ablation") {
        out = MilpPolicy::Classical;
        return true;
    }
    return false;
}

inline bool milp_policy_is_latest(MilpPolicy p) noexcept {
    return p == MilpPolicy::Latest;
}

}  // namespace sor::search
