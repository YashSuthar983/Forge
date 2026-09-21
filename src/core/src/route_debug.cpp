// RouteDebug implementation - see sor/core/route_debug.hpp for the design and
// for why the ledger carries a forced residual.

#include "sor/core/route_debug.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace sor::core {
namespace {

// Level is read on every SOR_ROUTE in hot loops, so it is a plain relaxed
// atomic rather than anything guarded. Everything else is touched only when
// tracing is on.
std::atomic<int> g_level{0};
std::atomic<int> g_pivot_every{0};
std::atomic<unsigned long long> g_seq{0};

std::mutex& sink_mutex() {
    static std::mutex m;
    return m;
}
std::FILE*& sink_file() {
    static std::FILE* f = nullptr;
    return f;
}
std::vector<std::string>& comp_filter() {
    static std::vector<std::string> v;
    return v;
}
std::vector<std::string>& path_filter() {
    static std::vector<std::string> v;
    return v;
}

void split_csv(const char* csv, std::vector<std::string>& out) {
    out.clear();
    if (csv == nullptr) return;
    const char* p = csv;
    std::string cur;
    for (;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            if (*p == '\0') break;
        } else if (*p != ' ') {
            cur.push_back(*p);
        }
    }
}

bool allowed(const std::vector<std::string>& filter, const char* value) {
    if (filter.empty()) return true;
    if (value == nullptr) return false;
    for (const auto& f : filter)
        if (f == value) return true;
    return false;
}

struct Ledger {
    double bucket[static_cast<std::size_t>(RouteLedgerBucket::Count_)] = {};
    double wall_ms = 0.0;
};
Ledger& ledger() {
    static Ledger l;
    return l;
}

const char* bucket_name(RouteLedgerBucket b) {
    switch (b) {
        case RouteLedgerBucket::Io:          return "io";
        case RouteLedgerBucket::Presolve:    return "presolve";
        case RouteLedgerBucket::ScaleFactor: return "scale_factor";
        case RouteLedgerBucket::Engine:      return "engine";
        case RouteLedgerBucket::Crossover:   return "crossover";
        case RouteLedgerBucket::Certify:     return "certify";
        case RouteLedgerBucket::Search:      return "search";
        case RouteLedgerBucket::NodeLp:      return "node_lp";
        case RouteLedgerBucket::Heur:        return "heur";
        case RouteLedgerBucket::Other:       return "other";
        default:                             return "?";
    }
}

}  // namespace

int  route_debug_level() { return g_level.load(std::memory_order_relaxed); }
void route_debug_set_level(int level) {
    g_level.store(level < 0 ? 0 : (level > 3 ? 3 : level),
                  std::memory_order_relaxed);
}
int  route_debug_pivot_every() { return g_pivot_every.load(std::memory_order_relaxed); }
void route_debug_set_pivot_every(int every) {
    g_pivot_every.store(every < 0 ? 0 : every, std::memory_order_relaxed);
}

void route_debug_set_file(const char* path) {
    std::lock_guard<std::mutex> lock(sink_mutex());
    std::FILE*& f = sink_file();
    if (f != nullptr) { std::fclose(f); f = nullptr; }
    if (path == nullptr || *path == '\0') return;
    f = std::fopen(path, "a");
    if (f == nullptr)
        std::fprintf(stderr,
                     "warning: --debug-routes-file: cannot open %s; "
                     "falling back to stderr\n", path);
}

void route_debug_set_comp_filter(const char* csv) {
    std::lock_guard<std::mutex> lock(sink_mutex());
    split_csv(csv, comp_filter());
}
void route_debug_set_path_filter(const char* csv) {
    std::lock_guard<std::mutex> lock(sink_mutex());
    split_csv(csv, path_filter());
}

bool route_debug_want(int level, const char* comp, const char* path) {
    if (g_level.load(std::memory_order_relaxed) < level) return false;
    std::lock_guard<std::mutex> lock(sink_mutex());
    if (!allowed(comp_filter(), comp)) return false;
    // A PATH filter drops untagged events on purpose: the point of the filter
    // is to make a trace small, and legacy untagged sites would swamp it.
    if (!path_filter().empty() && (path == nullptr || *path == '\0'))
        return false;
    if (!path_filter().empty() && !allowed(path_filter(), path)) return false;
    return true;
}

void route_debug_emit(int level, const char* comp, const char* path,
                      const char* event, const char* fields) {
    if (!route_debug_want(level, comp, path)) return;
    const unsigned long long seq = g_seq.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(sink_mutex());
    std::FILE* out = sink_file() ? sink_file() : stderr;
    if (fields != nullptr && *fields != '\0')
        std::fprintf(out,
                     "{\"seq\":%llu,\"lvl\":%d,\"comp\":\"%s\",\"path\":\"%s\","
                     "\"event\":\"%s\",%s}\n",
                     seq, level, comp ? comp : "", path ? path : "",
                     event ? event : "", fields);
    else
        std::fprintf(out,
                     "{\"seq\":%llu,\"lvl\":%d,\"comp\":\"%s\",\"path\":\"%s\","
                     "\"event\":\"%s\"}\n",
                     seq, level, comp ? comp : "", path ? path : "",
                     event ? event : "");
}

void route_debug_ledger_reset() {
    std::lock_guard<std::mutex> lock(sink_mutex());
    ledger() = Ledger{};
}

void route_debug_ledger_add(RouteLedgerBucket bucket, double ms) {
    const auto i = static_cast<std::size_t>(bucket);
    if (i >= static_cast<std::size_t>(RouteLedgerBucket::Count_)) return;
    std::lock_guard<std::mutex> lock(sink_mutex());
    ledger().bucket[i] += ms;
}

void route_debug_ledger_set_wall(double ms) {
    std::lock_guard<std::mutex> lock(sink_mutex());
    ledger().wall_ms = ms;
}

void route_debug_ledger_format(char* out, std::size_t cap) {
    if (out == nullptr || cap == 0) return;
    std::lock_guard<std::mutex> lock(sink_mutex());
    const Ledger& l = ledger();
    double sum = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(RouteLedgerBucket::Count_); ++i)
        sum += l.bucket[i];
    // The residual is the point of the ledger: it is what no bucket claimed.
    // It can go slightly negative when nested spans double-bill; that is
    // information too, so it is reported rather than clamped.
    const double residual = l.wall_ms - sum;
    std::snprintf(out, cap,
                  "route ledger: wall=%.1f sum=%.1f residual=%.1f | io=%.1f "
                  "presolve=%.1f scale_factor=%.1f engine=%.1f crossover=%.1f "
                  "certify=%.1f search=%.1f node_lp=%.1f heur=%.1f other=%.1f",
                  l.wall_ms, sum, residual,
                  l.bucket[static_cast<std::size_t>(RouteLedgerBucket::Io)],
                  l.bucket[static_cast<std::size_t>(RouteLedgerBucket::Presolve)],
                  l.bucket[static_cast<std::size_t>(RouteLedgerBucket::ScaleFactor)],
                  l.bucket[static_cast<std::size_t>(RouteLedgerBucket::Engine)],
                  l.bucket[static_cast<std::size_t>(RouteLedgerBucket::Crossover)],
                  l.bucket[static_cast<std::size_t>(RouteLedgerBucket::Certify)],
                  l.bucket[static_cast<std::size_t>(RouteLedgerBucket::Search)],
                  l.bucket[static_cast<std::size_t>(RouteLedgerBucket::NodeLp)],
                  l.bucket[static_cast<std::size_t>(RouteLedgerBucket::Heur)],
                  l.bucket[static_cast<std::size_t>(RouteLedgerBucket::Other)]);
}

void route_debug_ledger_emit(const char* scope) {
    if (g_level.load(std::memory_order_relaxed) < 1) return;
    double sum = 0.0;
    double wall = 0.0;
    double b[static_cast<std::size_t>(RouteLedgerBucket::Count_)];
    {
        std::lock_guard<std::mutex> lock(sink_mutex());
        const Ledger& l = ledger();
        wall = l.wall_ms;
        for (std::size_t i = 0; i < static_cast<std::size_t>(RouteLedgerBucket::Count_); ++i) {
            b[i] = l.bucket[i];
            sum += b[i];
        }
    }
    char fields[768];
    int n = std::snprintf(fields, sizeof fields,
                          "\"scope\":\"%s\",\"wall_ms\":%.3f,\"sum_ms\":%.3f,"
                          "\"residual_ms\":%.3f", scope ? scope : "", wall, sum,
                          wall - sum);
    for (std::size_t i = 0; i < static_cast<std::size_t>(RouteLedgerBucket::Count_) &&
                            n > 0 && static_cast<std::size_t>(n) < sizeof fields; ++i) {
        n += std::snprintf(fields + n, sizeof(fields) - static_cast<std::size_t>(n),
                           ",\"%s_ms\":%.3f",
                           bucket_name(static_cast<RouteLedgerBucket>(i)), b[i]);
    }
    route_debug_emit(1, "ledger", "ledger", "ledger", fields);
}

RouteSpan::RouteSpan(int level, const char* comp, const char* path,
                     const char* event, const char* fields,
                     RouteLedgerBucket bucket)
    : comp_(comp), path_(path), event_(event), bucket_(bucket), level_(level),
      // The span must bill its bucket even when the event is filtered out,
      // otherwise a narrowed trace would silently move time into the residual
      // and make the ledger lie about where it went.
      active_(g_level.load(std::memory_order_relaxed) >= 1),
      t0_(std::chrono::steady_clock::now()) {
    if (route_debug_want(level, comp, path)) {
        char buf[512];
        std::snprintf(buf, sizeof buf, "\"phase\":\"enter\"%s%s",
                      (fields && *fields) ? "," : "",
                      (fields && *fields) ? fields : "");
        route_debug_emit(level, comp, path, event, buf);
    }
}

double RouteSpan::elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0_).count();
}

RouteSpan::~RouteSpan() {
    if (!active_) return;
    const double ms = elapsed_ms();
    route_debug_ledger_add(bucket_, ms);
    if (route_debug_want(level_, comp_, path_)) {
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "\"phase\":\"exit\",\"duration_ms\":%.3f", ms);
        route_debug_emit(level_, comp_, path_, event_, buf);
    }
}

}  // namespace sor::core
