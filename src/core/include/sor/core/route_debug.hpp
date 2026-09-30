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
// COST WHEN DISABLED. A normal build does not define SOR_ROUTE_DEBUG, and
// every probe below is empty: SOR_ROUTE, SOR_ROUTE_PATH, SOR_FN, and
// RouteSpan compile to nothing. Pivot loops do not load an atomic or read
// a clock. Rebuild with -DSOR_ROUTE_FN=ON to compile them in.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

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

// Per-function probe. A normal build does not compile it: SOR_FN() is an
// empty statement, so pivot and LU loops do not load a flag or build a
// guard. Rebuild with -DSOR_ROUTE_FN=ON to compile the guards. Even then
// they stay dark until --debug-routes-fns or SOR_DEBUG_ROUTES_FNS=1.
// When that runtime flag is on, every instrumented function records its
// first enter immediately and a call-count / total-time summary at
// route_debug_fn_flush(). `--debug-routes-fn` is a comma-separated
// substring allowlist on the function name; empty means every function.
inline std::atomic<int>& route_debug_fns_flag() noexcept {
    static std::atomic<int> flag{0};
    return flag;
}
inline bool route_debug_fns_on() noexcept {
    return route_debug_fns_flag().load(std::memory_order_relaxed) != 0;
}
void route_debug_set_fns(bool on) noexcept;
void route_debug_fn_watch_main(const char* file, int line, const char* func);
void route_debug_set_fn_filter(const char* csv);
void route_debug_fn_flush();
std::uint64_t route_debug_fn_entered();

struct RouteFnRec;
RouteFnRec* route_fn_enter(const char* file, int line, const char* func);
void route_fn_note_enter(RouteFnRec* rec);
void route_fn_leave(RouteFnRec* rec, std::uint64_t ns);

// Stack guard. Constructed only when SOR_ROUTE_FN was on at compile time.
// Runtime-off is the flag load and a null check in the destructor. main is
// not tested here: that check would run on every call in the solve.
class RouteFn {
public:
    RouteFn(const char* file, int line, const char* func) noexcept
        : rec_(nullptr) {
        if (!route_debug_fns_on()) return;
        rec_ = route_fn_enter(file, line, func);
        if (rec_ == nullptr) return;
        route_fn_note_enter(rec_);
        t0_ = std::chrono::steady_clock::now();
    }
    ~RouteFn() {
        if (rec_ == nullptr) return;
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0_)
                            .count();
        route_fn_leave(rec_, static_cast<std::uint64_t>(ns < 0 ? 0 : ns));
    }
    RouteFn(const RouteFn&) = delete;
    RouteFn& operator=(const RouteFn&) = delete;

private:
    RouteFnRec* rec_;
    std::chrono::steady_clock::time_point t0_{};
};

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
// missing. Without SOR_ROUTE_DEBUG the type has no members and no clock.
#ifndef SOR_ROUTE_DEBUG
class RouteSpan {
public:
    RouteSpan(int, const char*, const char*, const char*, const char*,
              RouteLedgerBucket) noexcept {}
    ~RouteSpan() = default;
    RouteSpan(const RouteSpan&) = delete;
    RouteSpan& operator=(const RouteSpan&) = delete;
    double elapsed_ms() const noexcept { return 0.0; }
};
#else
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
#endif

}  // namespace sor::core

// Without SOR_ROUTE_DEBUG these are empty. Arguments are not evaluated
// only when they are written inside the macro; callers that format a
// buffer must sit behind a sample check that is itself inside ifndef.
#ifndef SOR_ROUTE_DEBUG
#define SOR_ROUTE(...)
#define SOR_ROUTE_PATH(...)
#define SOR_FN()
#else
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

// The extra macro layer is required because ## suppresses expansion of
// __LINE__, and a single paste gives every nested guard the same name.
#define SOR_FN_CAT_(a, b) a##b
#define SOR_FN_CAT(a, b) SOR_FN_CAT_(a, b)
#define SOR_FN()                                                               \
    ::sor::core::RouteFn SOR_FN_CAT(sor_fn_guard_, __LINE__)(                  \
        __FILE__, __LINE__, __func__)
#endif
