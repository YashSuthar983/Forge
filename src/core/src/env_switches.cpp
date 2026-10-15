#include "forge/core/env_switches.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

namespace forge::core {
namespace {

// Presence alone is not enough: FORGE_FOO=0 / false / off / no must disable.
// Unset or empty stays off. Any other non-empty spelling enables (including
// "1" and "yes").
bool flag(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return false;
    std::string s;
    s.reserve(std::strlen(v));
    for (const char* p = v; *p; ++p)
        s.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(*p))));
    if (s == "0" || s == "false" || s == "off" || s == "no" || s == "n")
        return false;
    return true;
}

std::string text(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

long long number(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(v, &end, 10);
    return end != v && *end == '\0' && parsed > 0 ? parsed : 0;
}

EnvSwitches read_environment() {
    EnvSwitches s;
    s.dual_choose_dse = flag("FORGE_DUAL_CHOOSE_DSE");
    // The Devex fallback and both BFRT guards below stay opt-in. Netlib-93 at
    // 1e-7 (2026-10-11), all three on vs off: pivot geomean 366 vs 347,
    // dfl001 Interrupted at 30 s (38,747 pivots) vs Optimal (18,058),
    // pilot4 2,193 vs 817, pilot.ja 7,597 vs 3,719; only pilot87 and
    // pilotnov gained (10,170 vs 11,700; 1,148 vs 1,437).
    s.dual_choose_devex_fallback = flag("FORGE_DUAL_CHOOSE_DEVEX_FALLBACK");
    s.dual_force_dense = flag("FORGE_DUAL_FORCE_DENSE");
    s.dual_separate_dse_ftran = flag("FORGE_DUAL_SEPARATE_DSE_FTRAN");
    s.dual_keep_basic_pivotal = flag("FORGE_DUAL_KEEP_BASIC_PIVOTAL");
    s.dual_accumulate_basic_pivotal = flag("FORGE_DUAL_ACCUMULATE_BASIC_PIVOTAL");
    s.dual_keep_fixed_pivotal = flag("FORGE_DUAL_KEEP_FIXED_PIVOTAL");
    s.dual_filter_active_pivotal = flag("FORGE_DUAL_FILTER_ACTIVE_PIVOTAL");
    s.dual_verify_chuzr_heap = flag("FORGE_DUAL_VERIFY_CHUZR_HEAP");
    s.dual_verify_chuzr = flag("FORGE_DUAL_VERIFY_CHUZR");
    s.dual_indexed_chuzr = flag("FORGE_DUAL_INDEXED_CHUZR");
    s.dual_fullscan_chuzr = flag("FORGE_DUAL_FULLSCAN_CHUZR");
    s.dual_no_cost_shift = flag("FORGE_DUAL_NO_COST_SHIFT");
    s.dual_bfrt_damage_guard = flag("FORGE_DUAL_BFRT_DAMAGE_GUARD");
    s.dual_bfrt_prefer_safe = flag("FORGE_DUAL_BFRT_PREFER_SAFE");
    s.dual_mid_solve_cost_audit = flag("FORGE_DUAL_MID_SOLVE_COST_AUDIT");
    s.dual_verbose_every = static_cast<std::uint64_t>(number("FORGE_DUAL_VERBOSE_EVERY"));
    s.dual_trace = text("FORGE_DUAL_TRACE");
    s.primal_dense_btran = flag("FORGE_PRIMAL_DENSE_BTRAN");
    s.primal_dense_ftran = flag("FORGE_PRIMAL_DENSE_FTRAN");
    s.primal_phase1_full_rebuild = flag("FORGE_PRIMAL_PHASE1_FULL_REBUILD");
    s.primal_verify_composite = flag("FORGE_PRIMAL_VERIFY_COMPOSITE");
    s.primal_fullscan = flag("FORGE_PRIMAL_FULLSCAN");
    s.primal_verify_heap = flag("FORGE_PRIMAL_VERIFY_HEAP");
    s.primal_trace = text("FORGE_PRIMAL_TRACE");
    s.lu_comparison_sort = flag("FORGE_LU_COMPARISON_SORT");
    s.lu_update_log = text("FORGE_LU_UPDATE_LOG");
    s.presolve_trace_recovery = flag("FORGE_PRESOLVE_TRACE_RECOVERY");
    s.presolve_no_retry = flag("FORGE_PRESOLVE_NO_RETRY");
    s.certificate_debug = flag("FORGE_CERTIFICATE_DEBUG");
    s.farkas_debug = flag("FORGE_FARKAS_DEBUG");
    s.ref_trace = flag("FORGE_REF_TRACE");
    s.fgmres_profile = flag("FORGE_FGMRES_PROFILE");
    s.dump_kkt = text("FORGE_DUMP_KKT");
    s.barrier_point_out = text("FORGE_BARRIER_POINT_OUT");
    s.barrier_point_in = text("FORGE_BARRIER_POINT_IN");
    s.stall_window = static_cast<int>(number("FORGE_STALL_WINDOW"));
    s.ipm_gondzio = number("FORGE_IPM_GONDZIO") > 0 ? static_cast<int>(number("FORGE_IPM_GONDZIO")) : -1;
    s.resto_ms = static_cast<long>(number("FORGE_RESTO_MS"));
    return s;
}

EnvSwitches& storage() {
    static EnvSwitches switches = read_environment();
    return switches;
}

}  // namespace

const EnvSwitches& env_switches() { return storage(); }

void reload_env_switches() { storage() = read_environment(); }

}  // namespace forge::core
