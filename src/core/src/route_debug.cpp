// RouteDebug implementation - see sor/core/route_debug.hpp for the design and
// for why the ledger carries a forced residual.

#include "sor/core/route_debug.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sor::core {

struct RouteFnRec {
    std::string file;
    std::string func;
    std::string path;
    int line = 0;
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> total_ns{0};
};

namespace {

// Level is read on every SOR_ROUTE in hot loops, so it is a plain relaxed
// atomic rather than anything guarded. Everything else is touched only when
// tracing is on.
std::atomic<int> g_level{0};
std::atomic<int> g_pivot_every{0};
std::atomic<unsigned long long> g_seq{0};
std::atomic<unsigned long long> g_fn_entered{0};
std::atomic<int> g_fn_flushed{0};

std::mutex& fn_mutex() {
    static std::mutex m;
    return m;
}
std::vector<std::string>& fn_filter() {
    static std::vector<std::string> v;
    return v;
}
std::vector<std::unique_ptr<RouteFnRec>>& fn_recs() {
    static std::vector<std::unique_ptr<RouteFnRec>> v;
    return v;
}

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

std::string json_escape(std::string_view in) {
    std::string out;
    out.reserve(in.size());
    for (unsigned char c : in) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(static_cast<char>(c));
        } else if (c >= 0x20) {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

std::string shorten_file(std::string_view file) {
    // Header paths look like .../sor/src/sparse/include/sor/sparse/csr.hpp, so
    // the last "/sor/" is the include prefix, not the repository root.
    const auto root = file.find("/sor/");
    if (root != std::string_view::npos) {
        const auto rest = file.substr(root + 5);
        if (rest.starts_with("src/") || rest.starts_with("apps/") ||
            rest.starts_with("tests/"))
            return std::string(rest);
    }
    for (const char* prefix : {"src/", "apps/", "tests/"}) {
        const auto pos = file.find(prefix);
        if (pos != std::string_view::npos)
            return std::string(file.substr(pos));
    }
    return std::string(file);
}

std::string subsystem_of(std::string_view file) {
    const std::string short_file = shorten_file(file);
    std::string_view rest(short_file);
    if (rest.size() >= 4 && rest.substr(0, 4) == "src/") {
        rest.remove_prefix(4);
        const auto slash = rest.find('/');
        if (slash != std::string_view::npos && slash > 0)
            return std::string(rest.substr(0, slash));
    }
    if (rest.size() >= 5 && rest.substr(0, 5) == "apps/") return "apps";
    if (rest.size() >= 6 && rest.substr(0, 6) == "tests/") return "tests";
    return "other";
}

bool fn_allowed(const char* func) {
    if (fn_filter().empty()) return true;
    if (func == nullptr) return false;
    const std::string_view name(func);
    for (const auto& f : fn_filter())
        if (name.find(f) != std::string_view::npos) return true;
    return false;
}

// Lines written before --debug-routes-file is parsed. Flag order must not
// drop main: --debug-routes-fns is often the earlier argument.
std::vector<std::string>& early_lines() {
    static std::vector<std::string> v;
    return v;
}
bool g_sink_chosen = false;

void write_raw(std::FILE* out, const std::string& line) {
    std::fwrite(line.data(), 1, line.size(), out);
    std::fputc('\n', out);
}

void write_line_unlocked(const std::string& line) {
    if (std::FILE* out = sink_file()) {
        write_raw(out, line);
        return;
    }
    if (g_sink_chosen) {
        write_raw(stderr, line);
        return;
    }
    early_lines().push_back(line);
}

void drain_early_unlocked(std::FILE* out) {
    for (const auto& line : early_lines()) write_raw(out, line);
    early_lines().clear();
    g_sink_chosen = true;
}

const char* g_pending_main_file = nullptr;
int g_pending_main_line = 0;
const char* g_pending_main_func = nullptr;
RouteFnRec* g_pending_main_rec = nullptr;
std::chrono::steady_clock::time_point g_pending_main_t0{};

void route_debug_fn_commit_pending_main() {
    if (g_pending_main_func == nullptr) return;
    const char* file = g_pending_main_file;
    const int line = g_pending_main_line;
    const char* func = g_pending_main_func;
    g_pending_main_func = nullptr;
    RouteFnRec* rec = route_fn_enter(file, line, func);
    if (rec == nullptr) return;
    route_fn_note_enter(rec);
    g_pending_main_rec = rec;
}

void route_debug_fn_account_pending_main() {
    RouteFnRec* rec = g_pending_main_rec;
    if (rec == nullptr) return;
    g_pending_main_rec = nullptr;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - g_pending_main_t0)
                        .count();
    route_fn_leave(rec, static_cast<std::uint64_t>(ns < 0 ? 0 : ns));
}

}  // namespace

void route_debug_set_fns(bool on) noexcept {
    route_debug_fns_flag().store(on ? 1 : 0, std::memory_order_relaxed);
    if (on) route_debug_fn_commit_pending_main();
}

void route_debug_fn_watch_main(const char* file, int line, const char* func) {
    g_pending_main_file = file;
    g_pending_main_line = line;
    g_pending_main_func = func;
    g_pending_main_t0 = std::chrono::steady_clock::now();
}

// SOR_DEBUG_ROUTES_FNS=1 turns probes on before main, so test mains and every
// function they call are visible without a CLI flag.
struct FnTraceEnv {
    FnTraceEnv() {
        const char* env = std::getenv("SOR_DEBUG_ROUTES_FNS");
        if (env != nullptr && env[0] != '\0' && env[0] != '0')
            route_debug_set_fns(true);
    }
};
FnTraceEnv g_fn_trace_env;

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
    if (path == nullptr || *path == '\0') {
        drain_early_unlocked(stderr);
        return;
    }
    f = std::fopen(path, "w");
    if (f == nullptr) {
        std::fprintf(stderr,
                     "warning: --debug-routes-file: cannot open %s; "
                     "falling back to stderr\n", path);
        drain_early_unlocked(stderr);
        return;
    }
    drain_early_unlocked(f);
}

void route_debug_set_comp_filter(const char* csv) {
    std::lock_guard<std::mutex> lock(sink_mutex());
    split_csv(csv, comp_filter());
}
void route_debug_set_path_filter(const char* csv) {
    std::lock_guard<std::mutex> lock(sink_mutex());
    split_csv(csv, path_filter());
}

void route_debug_set_fn_filter(const char* csv) {
    std::lock_guard<std::mutex> lock(fn_mutex());
    split_csv(csv, fn_filter());
}

std::uint64_t route_debug_fn_entered() {
    return g_fn_entered.load(std::memory_order_relaxed);
}

RouteFnRec* route_fn_skip() noexcept {
    static RouteFnRec skip;
    return &skip;
}

RouteFnRec* route_fn_enter(const char* file, int line, const char* func) {
    struct Key {
        std::string file;
        int line = 0;
        bool operator==(const Key& o) const {
            return line == o.line && file == o.file;
        }
    };
    struct Hash {
        std::size_t operator()(const Key& k) const noexcept {
            return std::hash<std::string>{}(k.file) ^
                   (std::hash<int>{}(k.line) << 1);
        }
    };
    static std::unordered_map<Key, RouteFnRec*, Hash> index;
    std::lock_guard<std::mutex> lock(fn_mutex());
    Key key{file ? file : "", line};
    if (const auto it = index.find(key); it != index.end())
        return it->second == route_fn_skip() ? nullptr : it->second;
    if (!fn_allowed(func)) {
        index.emplace(std::move(key), route_fn_skip());
        return nullptr;
    }
    auto rec = std::make_unique<RouteFnRec>();
    rec->file = shorten_file(file ? file : "");
    rec->func = func ? func : "";
    rec->path = subsystem_of(file ? file : "");
    rec->line = line;
    RouteFnRec* p = rec.get();
    fn_recs().push_back(std::move(rec));
    index.emplace(std::move(key), p);
    return p;
}

void route_fn_note_enter(RouteFnRec* rec) {
    if (rec == nullptr || rec == route_fn_skip()) return;
    if (rec->calls.fetch_add(1, std::memory_order_relaxed) != 0) return;
    g_fn_entered.fetch_add(1, std::memory_order_relaxed);
    const unsigned long long seq = g_seq.fetch_add(1, std::memory_order_relaxed);
    const std::string line =
        std::string("{\"seq\":") + std::to_string(seq) +
        ",\"lvl\":1,\"comp\":\"fn\",\"path\":\"" + json_escape(rec->path) +
        "\",\"event\":\"" + json_escape(rec->func) +
        "\",\"phase\":\"enter\",\"file\":\"" + json_escape(rec->file) +
        "\",\"line\":" + std::to_string(rec->line) + "}";
    std::lock_guard<std::mutex> lock(sink_mutex());
    write_line_unlocked(line);
}

void route_fn_leave(RouteFnRec* rec, std::uint64_t ns) {
    if (rec == nullptr || rec == route_fn_skip()) return;
    rec->total_ns.fetch_add(ns, std::memory_order_relaxed);
}

void route_debug_fn_flush() {
    if (!route_debug_fns_on()) return;
    int expected = 0;
    if (!g_fn_flushed.compare_exchange_strong(expected, 1,
                                              std::memory_order_relaxed))
        return;
    // main's probe was built before the flag flipped. Fold its time in
    // before the summaries are written.
    route_debug_fn_account_pending_main();
    std::vector<RouteFnRec*> recs;
    {
        std::lock_guard<std::mutex> lock(fn_mutex());
        recs.reserve(fn_recs().size());
        for (const auto& r : fn_recs()) recs.push_back(r.get());
    }
    std::lock_guard<std::mutex> lock(sink_mutex());
    for (RouteFnRec* r : recs) {
        const unsigned long long seq =
            g_seq.fetch_add(1, std::memory_order_relaxed);
        const double ms =
            static_cast<double>(r->total_ns.load(std::memory_order_relaxed)) /
            1e6;
        const auto calls = r->calls.load(std::memory_order_relaxed);
        char tail[96];
        std::snprintf(tail, sizeof tail, ",\"calls\":%llu,\"total_ms\":%.3f}",
                      static_cast<unsigned long long>(calls), ms);
        const std::string line =
            std::string("{\"seq\":") + std::to_string(seq) +
            ",\"lvl\":1,\"comp\":\"fn\",\"path\":\"" + json_escape(r->path) +
            "\",\"event\":\"" + json_escape(r->func) +
            "\",\"phase\":\"summary\",\"file\":\"" + json_escape(r->file) +
            "\",\"line\":" + std::to_string(r->line) + tail;
        write_line_unlocked(line);
    }
    if (sink_file() == nullptr && !g_sink_chosen) drain_early_unlocked(stderr);
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
    std::string line =
        std::string("{\"seq\":") + std::to_string(seq) + ",\"lvl\":" +
        std::to_string(level) + ",\"comp\":\"" + (comp ? comp : "") +
        "\",\"path\":\"" + (path ? path : "") + "\",\"event\":\"" +
        (event ? event : "") + "\"";
    if (fields != nullptr && *fields != '\0') {
        line += ',';
        line += fields;
    }
    line += '}';
    std::lock_guard<std::mutex> lock(sink_mutex());
    write_line_unlocked(line);
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

#ifdef SOR_ROUTE_DEBUG
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
#endif

}  // namespace sor::core
