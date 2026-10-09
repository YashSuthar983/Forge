#pragma once
// Deterministic work accounting.
//
// Every internal budget of the MILP search (probing, heuristic shares, the
// cut loop's allowance, sub-MIP budgets, LP limits derived from them) used to
// be a wall-clock interval, so the same build on the same model took a
// different path on a busier or slower core: drayage-25-23 reached root bound
// 99.7k on one core and 11.7k on another because probing finished in time on
// one and not the other. Results could not be told from noise.
//
// Hot loops add abstract work units to a per-thread counter. With
// deterministic timing on, SolverClock reports the counter as time
// (units * ns_per_unit), so every budget measured with it depends only on the
// work done. The caller keeps one wall-clock stop for the overall limit.
//
// Safety floor: past a grace period, SolverClock never runs slower than
// kWallFloor of wall time, so a loop whose work is not counted still ends its
// budget. With the loops counted this floor does not bind; when it does, that
// phase is not deterministic, but it cannot hang.
#include <chrono>
#include <cstdint>

namespace forge::core {

extern thread_local std::uint64_t t_work_units;

// What a charge is for. Diagnostic only: the units are the same whatever the
// tag, and the ledger reports them per tag so the charges can be calibrated
// against wall time site by site.
enum class WorkTag : std::uint8_t {
    Untagged,
    SimplexIter,
    PrimalIter,
    LpSolve,
    LuFactor,
    LuSolve,
    Propagate,
    Fixprop,
    FeasJump,
    BabNode,
    HeurCall,
    LpPresolve,
    MilpPresolve,
    Scaling,
    Crash,
    ProbeEdge,
    ClauseScan,
    ExactDyadic,
    ExactMp,
    Gmi,
    Mir,
    Cover,
    ZeroHalf,
    Parallelism,
    Symmetry,
    ExactSolve,
    Interval,      // interval-bounded long double certificate sums (one per entry)
    Count
};
constexpr int kWorkTags = static_cast<int>(WorkTag::Count);
extern thread_local std::uint64_t t_work_tag_units[kWorkTags];
extern thread_local std::uint64_t t_work_frac;
const char* work_tag_name(WorkTag t) noexcept;

// Per-tag weight, in 1/256 of a unit: what one charged unit at that site
// costs relative to the clock's ns_per_work_unit(). Fitted by bounded least
// squares (scales in [0.05, 20], tags with little data pulled to 1) so that
// the weighted units predict wall time over 240 easy60 runs (10 s and 20 s,
// one thread, Ryzen 5 5600H): 222 of 240 within [0.5x, 2x], median 0.91.
// One weight per tag cannot follow per-model cache effects (an LU solve on a
// large factor costs several times more per entry), so about +-2x is the
// floor of this model. Raw units stay in t_work_tag_units for refits.
inline constexpr std::uint16_t kWorkTagWeight256[kWorkTags] = {
    256,  // Untagged
    226,  // SimplexIter   0.88
    256,  // PrimalIter
    256,  // LpSolve
    272,  // LuFactor      1.06
    50,   // LuSolve       0.20
    155,  // Propagate     0.61
    151,  // Fixprop       0.59
    178,  // FeasJump      0.70
    256,  // BabNode
    256,  // HeurCall
    256,  // LpPresolve
    256,  // MilpPresolve
    256,  // Scaling
    256,  // Crash
    256,  // ProbeEdge     (recharged after the add_edge fix; refit)
    14,   // ClauseScan    0.06
    253,  // ExactDyadic   0.99
    241,  // ExactMp       0.94
    256,  // Gmi
    256,  // Mir
    256,  // Cover
    240,  // ZeroHalf      0.94
    256,  // Parallelism
    256,  // Symmetry
    256,  // ExactSolve
    256,  // Interval      (new; refit)
};

inline void work_add(std::uint64_t units, WorkTag tag = WorkTag::Untagged) noexcept {
    const auto t = static_cast<int>(tag);
    t_work_tag_units[t] += units;
    // Fixed point with a carried remainder: exact and reproducible.
    t_work_frac += units * kWorkTagWeight256[t];
    t_work_units += t_work_frac >> 8;
    t_work_frac &= 0xFF;
}
inline std::uint64_t work_units() noexcept { return t_work_units; }

// Process-wide switch, set once before a solve (not mid-solve).
void set_deterministic_time(bool on) noexcept;
bool deterministic_time() noexcept;
// Nanoseconds one work unit stands for (calibrated on the reference machine).
double ns_per_work_unit() noexcept;

// Work attribution. Each thread charges the wall time and the work units
// spent while a category is innermost to that category, so one run shows,
// per kind of work, how far the work charges stand from the wall time they
// are meant to model (units * ns_per_work_unit() against wall). Categories
// nest through WorkScope; the outermost is Other. Units are counted in every
// timing mode, so a wall-mode run calibrates the deterministic clock.
enum class WorkCat : std::uint8_t {
    Other = 0, Simplex, Factor, Exact, Presolve, Probing, Propagation,
    Separation, Heuristic, Symmetry, Conflict, Count
};
constexpr int kWorkCats = static_cast<int>(WorkCat::Count);
const char* work_cat_name(WorkCat c) noexcept;
struct WorkLedger {
    std::int64_t wall_ns[kWorkCats] = {};
    std::uint64_t units[kWorkCats] = {};
};
// This thread's ledger, with the open category's time up to now included.
WorkLedger work_ledger() noexcept;
void work_ledger_reset() noexcept;

class WorkScope {
public:
    explicit WorkScope(WorkCat c) noexcept;
    ~WorkScope();
    WorkScope(const WorkScope&) = delete;
    WorkScope& operator=(const WorkScope&) = delete;
private:
    WorkCat prev_;
};

// How often, process-wide, the deterministic clock read the wall floor
// instead of the work count, and the largest lead the floor had (ns). Any
// binding means a budget measured across it depended on machine load.
std::uint64_t work_floor_binds() noexcept;
std::int64_t work_floor_max_lead_ns() noexcept;

struct SolverClock {
    using rep = std::int64_t;
    using period = std::nano;
    using duration = std::chrono::nanoseconds;
    using time_point = std::chrono::time_point<SolverClock, duration>;
    static constexpr bool is_steady = true;
    static time_point now() noexcept;
    static time_point raw_now() noexcept;
    static duration offset() noexcept;
    static void set_offset(duration) noexcept;
};

// A cooperatively scheduled solve keeps its logical clock while inactive.
// Run only at quiescent boundaries, on its owning thread. Nested contexts
// restore the enclosing clock; work counters and ledgers remain cumulative.
class SolverClockContext {
public:
    template<class Fn> void run(Fn&& fn) {
        const auto before = SolverClock::offset();
        const auto raw = SolverClock::raw_now();
        if (!started_) { offset_ = before; started_ = true; }
        else offset_ -= raw - paused_;
        SolverClock::set_offset(offset_);
        struct Restore {
            SolverClockContext& context;
            SolverClock::duration before;
            ~Restore() {
                context.paused_ = SolverClock::raw_now();
                SolverClock::set_offset(before);
            }
        } restore{*this, before};
        fn();
    }
private:
    SolverClock::duration offset_{};
    SolverClock::time_point paused_{};
    bool started_ = false;
};

}  // namespace forge::core
