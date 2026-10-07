// SOR - developer switches read from SOR_* environment variables.
//
// LAYER L0. These are A/B and debugging hooks, never defaults. They are read
// once, on first use, into one struct that solver code consults; nothing in
// the solver calls getenv. getenv scans the whole environment per call, and
// it was called on every cost shift of the dual and about fifteen times per
// LP solve, which a short MILP node LP notices. This is also the one place
// that lists them all.
#pragma once

#include <cstdint>
#include <string>

namespace sor::core {

struct EnvSwitches {
    // Dual simplex.
    bool dual_choose_dse = false;                 // SOR_DUAL_CHOOSE_DSE
    bool dual_choose_devex_fallback = false;      // SOR_DUAL_CHOOSE_DEVEX_FALLBACK
    bool dual_force_dense = false;                // SOR_DUAL_FORCE_DENSE
    bool dual_separate_dse_ftran = false;         // SOR_DUAL_SEPARATE_DSE_FTRAN
    bool dual_keep_basic_pivotal = false;         // SOR_DUAL_KEEP_BASIC_PIVOTAL
    bool dual_accumulate_basic_pivotal = false;   // SOR_DUAL_ACCUMULATE_BASIC_PIVOTAL
    bool dual_keep_fixed_pivotal = false;         // SOR_DUAL_KEEP_FIXED_PIVOTAL
    bool dual_filter_active_pivotal = false;      // SOR_DUAL_FILTER_ACTIVE_PIVOTAL
    bool dual_verify_chuzr_heap = false;          // SOR_DUAL_VERIFY_CHUZR_HEAP
    bool dual_verify_chuzr = false;               // SOR_DUAL_VERIFY_CHUZR (active mode)
    bool dual_indexed_chuzr = false;              // SOR_DUAL_INDEXED_CHUZR
    bool dual_fullscan_chuzr = false;             // SOR_DUAL_FULLSCAN_CHUZR
    bool dual_no_cost_shift = false;              // SOR_DUAL_NO_COST_SHIFT
    std::uint64_t dual_verbose_every = 0;         // SOR_DUAL_VERBOSE_EVERY (0: default)
    std::string dual_trace;                       // SOR_DUAL_TRACE=<file>

    // Primal simplex.
    bool primal_dense_btran = false;              // SOR_PRIMAL_DENSE_BTRAN
    bool primal_dense_ftran = false;              // SOR_PRIMAL_DENSE_FTRAN
    bool primal_phase1_full_rebuild = false;      // SOR_PRIMAL_PHASE1_FULL_REBUILD
    bool primal_verify_composite = false;         // SOR_PRIMAL_VERIFY_COMPOSITE
    bool primal_fullscan = false;                 // SOR_PRIMAL_FULLSCAN
    bool primal_verify_heap = false;              // SOR_PRIMAL_VERIFY_HEAP
    std::string primal_trace;                     // SOR_PRIMAL_TRACE=<file>

    // LU.
    bool lu_comparison_sort = false;              // SOR_LU_COMPARISON_SORT
    std::string lu_update_log;                    // SOR_LU_UPDATE_LOG=<file>

    // Driver, presolve and certificates.
    bool presolve_trace_recovery = false;         // SOR_PRESOLVE_TRACE_RECOVERY
    bool presolve_no_retry = false;               // SOR_PRESOLVE_NO_RETRY
    bool certificate_debug = false;               // SOR_CERTIFICATE_DEBUG
    bool farkas_debug = false;                    // SOR_FARKAS_DEBUG

    // Quadratic engines.
    bool fgmres_profile = false;                  // SOR_FGMRES_PROFILE
    std::string dump_kkt;                         // SOR_DUMP_KKT=<file>
    int stall_window = 0;                         // SOR_STALL_WINDOW (0: default)
    long resto_ms = 0;                            // SOR_RESTO_MS (0: default)
};

// The switches as the environment set them when first asked.
const EnvSwitches& env_switches();

// Read the environment again. Only for tests that change it at run time;
// not safe while another thread reads the switches.
void reload_env_switches();

}  // namespace sor::core
