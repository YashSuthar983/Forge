// FORGE - developer switches read from FORGE_* environment variables.
//
// LAYER L0. These are A/B and debugging hooks, never defaults. They are read
// once, on first use, into one struct that solver code consults; nothing in
// the solver calls getenv. getenv scans the whole environment per call, and
// it was called on every cost shift of the dual and about fifteen times per
// LP solve, which a short MILP node LP notices. This is also the one place
// that lists them all.
//
// Boolean flags treat unset/empty/"0"/"false"/"off"/"no" as disabled; any
// other non-empty value enables. Default-off A/B validation must unset the
// variable (or set it to an explicit disable spelling), not export FOO=0
// and expect the old presence-only reader.
#pragma once

#include <cstdint>
#include <string>

namespace forge::core {

struct EnvSwitches {
    // Dual simplex.
    bool dual_choose_dse = false;                 // FORGE_DUAL_CHOOSE_DSE
    bool dual_choose_devex_fallback = false;      // FORGE_DUAL_CHOOSE_DEVEX_FALLBACK
    bool dual_force_dense = false;                // FORGE_DUAL_FORCE_DENSE
    bool dual_separate_dse_ftran = false;         // FORGE_DUAL_SEPARATE_DSE_FTRAN
    bool dual_keep_basic_pivotal = false;         // FORGE_DUAL_KEEP_BASIC_PIVOTAL
    bool dual_accumulate_basic_pivotal = false;   // FORGE_DUAL_ACCUMULATE_BASIC_PIVOTAL
    bool dual_keep_fixed_pivotal = false;         // FORGE_DUAL_KEEP_FIXED_PIVOTAL
    bool dual_filter_active_pivotal = false;      // FORGE_DUAL_FILTER_ACTIVE_PIVOTAL
    bool dual_verify_chuzr_heap = false;          // FORGE_DUAL_VERIFY_CHUZR_HEAP
    bool dual_verify_chuzr = false;               // FORGE_DUAL_VERIFY_CHUZR (active mode)
    bool dual_indexed_chuzr = false;              // FORGE_DUAL_INDEXED_CHUZR
    bool dual_fullscan_chuzr = false;             // FORGE_DUAL_FULLSCAN_CHUZR
    bool dual_no_cost_shift = false;              // FORGE_DUAL_NO_COST_SHIFT
    bool dual_bfrt_damage_guard = false;          // FORGE_DUAL_BFRT_DAMAGE_GUARD
    bool dual_bfrt_prefer_safe = false;           // FORGE_DUAL_BFRT_PREFER_SAFE
    bool dual_mid_solve_cost_audit = false;       // FORGE_DUAL_MID_SOLVE_COST_AUDIT
    std::uint64_t dual_verbose_every = 0;         // FORGE_DUAL_VERBOSE_EVERY (0: default)
    std::string dual_trace;                       // FORGE_DUAL_TRACE=<file>

    // Primal simplex.
    bool primal_dense_btran = false;              // FORGE_PRIMAL_DENSE_BTRAN
    bool primal_dense_ftran = false;              // FORGE_PRIMAL_DENSE_FTRAN
    bool primal_phase1_full_rebuild = false;      // FORGE_PRIMAL_PHASE1_FULL_REBUILD
    bool primal_verify_composite = false;         // FORGE_PRIMAL_VERIFY_COMPOSITE
    bool primal_fullscan = false;                 // FORGE_PRIMAL_FULLSCAN
    bool primal_verify_heap = false;              // FORGE_PRIMAL_VERIFY_HEAP
    std::string primal_trace;                     // FORGE_PRIMAL_TRACE=<file>

    // LU.
    bool lu_comparison_sort = false;              // FORGE_LU_COMPARISON_SORT
    bool lu_no_simd_solves = false;               // FORGE_NO_SIMD_SOLVES (force scalar)
    std::string lu_update_log;                    // FORGE_LU_UPDATE_LOG=<file>

    // Driver, presolve and certificates.
    bool presolve_trace_recovery = false;         // FORGE_PRESOLVE_TRACE_RECOVERY
    bool presolve_no_retry = false;               // FORGE_PRESOLVE_NO_RETRY
    bool certificate_debug = false;               // FORGE_CERTIFICATE_DEBUG
    bool farkas_debug = false;                    // FORGE_FARKAS_DEBUG

    // MILP search.
    bool ref_trace = false;                       // FORGE_REF_TRACE (with --verify-cuts)

    // Quadratic engines.
    bool fgmres_profile = false;                  // FORGE_FGMRES_PROFILE
    std::string dump_kkt;                         // FORGE_DUMP_KKT=<file>
    int stall_window = 0;                         // FORGE_STALL_WINDOW (0: default)
    long resto_ms = 0;                            // FORGE_RESTO_MS (0: default)
};

// The switches as the environment set them when first asked.
const EnvSwitches& env_switches();

// Read the environment again. Only for tests that change it at run time;
// not safe while another thread reads the switches.
void reload_env_switches();

}  // namespace forge::core
