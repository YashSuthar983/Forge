#include "forge/core/work_clock.hpp"

#include <algorithm>
#include <atomic>

namespace forge::core {

thread_local std::uint64_t t_work_units = 0;
thread_local std::uint64_t t_work_tag_units[kWorkTags] = {};
thread_local std::uint64_t t_work_frac = 0;

const char* work_tag_name(WorkTag t) noexcept {
    static const char* const kNames[kWorkTags] = {
        "Untagged",
        "SimplexIter",
        "PrimalIter",
        "LpSolve",
        "LuFactor",
        "LuSolve",
        "Propagate",
        "Fixprop",
        "FeasJump",
        "BabNode",
        "HeurCall",
        "LpPresolve",
        "MilpPresolve",
        "Scaling",
        "Crash",
        "ProbeEdge",
        "ClauseScan",
        "ExactDyadic",
        "ExactMp",
        "Gmi",
        "Mir",
        "Cover",
        "ZeroHalf",
        "Parallelism",
        "Symmetry",
        "ExactSolve",
        "Interval",
    };
    const int i = static_cast<int>(t);
    return i >= 0 && i < kWorkTags ? kNames[i] : "?";
}

namespace {
std::atomic<bool> g_deterministic{false};
// Calibrated so that work time tracks wall time on the reference machine
// (Ryzen 5 5600H, one thread): the median over ten easy60 models was
// ~15 ns per unit, with outliers from 7.6 (pg) to 56 (irp).
constexpr double kNsPerUnit = 15.0;
// The work clock never runs slower than this share of the wall time past a
// grace period. The floor is a stall guard for a loop whose work is not
// counted; it must not bind in ordinary runs, or every budget measured across
// the moment it binds inherits the wall clock. Without the grace it bound at
// startup (reading and setting up the model is not counted): the probe that
// timed its budget from there stopped after 2384, 2400 or 2416 binaries on
// three runs of neos-827175, and the runs went different ways.
constexpr double kWallFloor = 0.2;
constexpr double kFloorGraceNs = 5e9;

std::int64_t wall_ns() noexcept {
    // One epoch for the whole process, so wall and work readings share an
    // origin and the floor below compares like with like.
    static const auto origin = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - origin).count();
}
}  // namespace

void set_deterministic_time(bool on) noexcept { g_deterministic.store(on); }
bool deterministic_time() noexcept { return g_deterministic.load(std::memory_order_relaxed); }
double ns_per_work_unit() noexcept { return kNsPerUnit; }

namespace {
std::atomic<std::uint64_t> g_floor_binds{0};
std::atomic<std::int64_t> g_floor_max_lead{0};

thread_local WorkCat t_cat = WorkCat::Other;
thread_local std::int64_t t_cat_wall0 = -1;
thread_local std::uint64_t t_cat_units0 = 0;
thread_local WorkLedger t_ledger;

// Close the open category's span at `wall` and start a new one.
void ledger_flush(std::int64_t wall) noexcept {
    const auto c = static_cast<int>(t_cat);
    if (t_cat_wall0 >= 0) {
        t_ledger.wall_ns[c] += wall - t_cat_wall0;
        t_ledger.units[c] += t_work_units - t_cat_units0;
    }
    t_cat_wall0 = wall;
    t_cat_units0 = t_work_units;
}
}  // namespace

const char* work_cat_name(WorkCat c) noexcept {
    static const char* const kNames[kWorkCats] = {
        "other", "simplex", "factor", "exact", "presolve", "probing",
        "propagation", "separation", "heuristic", "symmetry", "conflict"};
    const int i = static_cast<int>(c);
    return i >= 0 && i < kWorkCats ? kNames[i] : "?";
}

WorkLedger work_ledger() noexcept {
    ledger_flush(wall_ns());
    return t_ledger;
}

void work_ledger_reset() noexcept {
    t_ledger = WorkLedger{};
    t_cat_wall0 = wall_ns();
    t_cat_units0 = t_work_units;
}

WorkScope::WorkScope(WorkCat c) noexcept : prev_(t_cat) {
    ledger_flush(wall_ns());
    t_cat = c;
}

WorkScope::~WorkScope() {
    ledger_flush(wall_ns());
    t_cat = prev_;
}

std::uint64_t work_floor_binds() noexcept { return g_floor_binds.load(std::memory_order_relaxed); }
std::int64_t work_floor_max_lead_ns() noexcept { return g_floor_max_lead.load(std::memory_order_relaxed); }

SolverClock::time_point SolverClock::now() noexcept {
    const std::int64_t wall = wall_ns();
    if (!deterministic_time()) return time_point(duration(wall));
    const auto work = static_cast<std::int64_t>(static_cast<double>(t_work_units) * kNsPerUnit);
    const auto floor_ns = static_cast<std::int64_t>(
        kWallFloor * std::max(0.0, static_cast<double>(wall) - kFloorGraceNs));
    if (floor_ns > work) {
        g_floor_binds.fetch_add(1, std::memory_order_relaxed);
        const std::int64_t lead = floor_ns - work;
        std::int64_t prev = g_floor_max_lead.load(std::memory_order_relaxed);
        while (lead > prev &&
               !g_floor_max_lead.compare_exchange_weak(prev, lead, std::memory_order_relaxed)) {}
        return time_point(duration(floor_ns));
    }
    return time_point(duration(work));
}

}  // namespace forge::core
