// SOR - RouteDebug: process-wide structured tracing with a closed time ledger.
//
// LAYER L0 (core). Every engine emits through the same two primitives, so a
// single trace can follow one solve across io -> presolve -> LP routing ->
// engine -> crossover -> certify -> search, and a single ledger can say where
// the wall clock went.
//
// WHY A LEDGER AND NOT JUST TIMERS. Per-phase timers only cover the phases
// somebody thought to instrument, and they systematically miss early-exit
// paths. Measured on MIPLIB2017 app1-1: the MILP block reported "node LP" and
// a grand total, and 27 s of a 30 s solve was in neither -- it sat on the
// LP-infeasible prune path, which no happy-path timer touched. The ledger
// closes that hole by construction: buckets plus a FORCED residual,
//
//     residual_ms = wall_ms - sum(buckets)
//
// so unattributed time is a number you can read rather than an absence you
// have to notice.
//
// COST WHEN DISABLED. Emission is guarded by an inlined level check against a
// single relaxed atomic, so a disabled SOR_ROUTE is one predictable branch and
// no argument evaluation. That matters because these macros sit in pivot loops.
#pragma once

#include <chrono>
#include <cstddef>

namespace sor::core {

// Fixed ledger buckets. Deliberately few: a bucket per subsystem, plus Other,
// plus the residual that the formatter derives. Adding buckets is cheaper than
// leaving time unattributed, but every bucket that is not obviously distinct
// makes the ledger harder to read.
enum class RouteLedgerBucket {
    Io = 0,
    Presolve,
    ScaleFactor,
    Engine,
    Crossover,
    Certify,
    Search,
    NodeLp,
    Heur,
    Other,
    Count_
};

// --- configuration (CLI) ---------------------------------------------------
int  route_debug_level();
void route_debug_set_level(int level);            // 0..3, clamped
int  route_debug_pivot_every();                   // 0 = every event
void route_debug_set_pivot_every(int every);
void route_debug_set_file(const char* path);      // append JSONL; nullptr=stderr
// Comma-separated allowlists. Empty/unset = allow everything. When a PATH
// filter is set, events carrying no path tag are dropped -- include "ledger"
// to keep the ledger line.
void route_debug_set_comp_filter(const char* csv);
void route_debug_set_path_filter(const char* csv);

// --- emission --------------------------------------------------------------
// `fields` is a raw JSON fragment (no braces), e.g. "\"nodes\":42".
void route_debug_emit(int level, const char* comp, const char* path,
                      const char* event, const char* fields);
bool route_debug_want(int level, const char* comp, const char* path);

// --- ledger ----------------------------------------------------------------
void route_debug_ledger_reset();
void route_debug_ledger_add(RouteLedgerBucket bucket, double ms);
void route_debug_ledger_set_wall(double ms);
void route_debug_ledger_emit(const char* scope);
// Human-readable one-liner, including the forced residual.
void route_debug_ledger_format(char* out, std::size_t cap);

// RAII span: times a stage, bills it to a bucket, and emits enter/exit.
// Nested spans both bill, so a node LP inside a search span appears in both
// `node_lp` and `search`; the residual is still the authority on what is
// missing.
class RouteSpan {
public:
    RouteSpan(int level, const char* comp, const char* path, const char* event,
              const char* fields, RouteLedgerBucket bucket);
    ~RouteSpan();
    RouteSpan(const RouteSpan&) = delete;
    RouteSpan& operator=(const RouteSpan&) = delete;
    double elapsed_ms() const;

private:
    const char* comp_;
    const char* path_;
    const char* event_;
    RouteLedgerBucket bucket_;
    int level_;
    bool active_;
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace sor::core

// Three- and four-argument forms; the three-argument one carries no fields.
#define SOR_ROUTE_4(lvl, comp, event, fields)                                  \
    do {                                                                       \
        if (::sor::core::route_debug_level() >= (lvl))                         \
            ::sor::core::route_debug_emit((lvl), (comp), "", (event),          \
                                          (fields));                           \
    } while (0)
#define SOR_ROUTE_3(lvl, comp, event) SOR_ROUTE_4(lvl, comp, event, "")
#define SOR_ROUTE_PICK(_1, _2, _3, _4, NAME, ...) NAME
#define SOR_ROUTE(...)                                                         \
    SOR_ROUTE_PICK(__VA_ARGS__, SOR_ROUTE_4, SOR_ROUTE_3, _x, _y)(__VA_ARGS__)

// Four- and five-argument forms, as SOR_ROUTE above.
#define SOR_ROUTE_PATH_5(lvl, comp, path, event, fields)                       \
    do {                                                                       \
        if (::sor::core::route_debug_level() >= (lvl))                         \
            ::sor::core::route_debug_emit((lvl), (comp), (path), (event),      \
                                          (fields));                           \
    } while (0)
#define SOR_ROUTE_PATH_4(lvl, comp, path, event)                               \
    SOR_ROUTE_PATH_5(lvl, comp, path, event, "")
#define SOR_ROUTE_PATH_PICK(_1, _2, _3, _4, _5, NAME, ...) NAME
#define SOR_ROUTE_PATH(...)                                                    \
    SOR_ROUTE_PATH_PICK(__VA_ARGS__, SOR_ROUTE_PATH_5, SOR_ROUTE_PATH_4, _x,   \
                        _y, _z)(__VA_ARGS__)
