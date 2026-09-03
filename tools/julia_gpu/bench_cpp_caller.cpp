// Standalone C++ caller overhead bench for the Julia GPU sidecar.
//
// Measures, from the C++ side:
//   1. spawn → SORGPU_READY   (Julia boot + package load + optional warmup)
//   2. ping RTT               (IPC floor)
//   3. bench_overhead RPC     (Julia-reported first vs warm kernel)
//   4. upload + first/warm spmv wall times vs Julia kernel_ms (IPC delta)
//
// Build (no CMake required):
//   g++ -O2 -std=c++20 -o bench_julia_gpu_overhead \
//       tools/julia_gpu/bench_cpp_caller.cpp
//
// Run from the sor/ directory:
//   ./tools/julia_gpu/bench_julia_gpu_overhead [--no-warmup] [--device cpu]
//
// THIS IS A MEASUREMENT TOOL. It does not claim GPU acceleration.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::string env_or(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : std::string(fallback);
}

// Minimal key finder: first occurrence of "key": at any depth (string-aware).
std::optional<std::size_t> find_key(std::string_view s, std::string_view key) {
    bool in_str = false;
    for (std::size_t i = 0; i + key.size() + 3 < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (c == '\\') {
                ++i;
                continue;
            }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') {
            const std::size_t start = i + 1;
            if (start + key.size() < s.size() &&
                s.substr(start, key.size()) == key &&
                s[start + key.size()] == '"') {
                std::size_t k = start + key.size() + 1;
                while (k < s.size() &&
                       std::isspace(static_cast<unsigned char>(s[k])))
                    ++k;
                if (k < s.size() && s[k] == ':') {
                    ++k;
                    while (k < s.size() &&
                           std::isspace(static_cast<unsigned char>(s[k])))
                        ++k;
                    return k;
                }
            }
            // Skip to end of this string.
            bool esc = false;
            for (std::size_t j = start; j < s.size(); ++j) {
                if (esc) {
                    esc = false;
                    continue;
                }
                if (s[j] == '\\') {
                    esc = true;
                    continue;
                }
                if (s[j] == '"') {
                    i = j;
                    break;
                }
            }
            continue;
        }
    }
    return std::nullopt;
}

std::optional<double> get_number(std::string_view s, std::string_view key) {
    auto pos = find_key(s, key);
    if (!pos) return std::nullopt;
    char* end = nullptr;
    const double v = std::strtod(s.data() + *pos, &end);
    if (end == s.data() + *pos) return std::nullopt;
    return v;
}

std::optional<std::int64_t> get_int(std::string_view s, std::string_view key) {
    auto pos = find_key(s, key);
    if (!pos) return std::nullopt;
    char* end = nullptr;
    const long long v = std::strtoll(s.data() + *pos, &end, 10);
    if (end == s.data() + *pos) return std::nullopt;
    return static_cast<std::int64_t>(v);
}

bool get_bool(std::string_view s, std::string_view key, bool fallback) {
    auto pos = find_key(s, key);
    if (!pos) return fallback;
    if (s.compare(*pos, 4, "true") == 0) return true;
    if (s.compare(*pos, 5, "false") == 0) return false;
    return fallback;
}

std::optional<std::string> get_string(std::string_view s, std::string_view key) {
    auto pos = find_key(s, key);
    if (!pos || s[*pos] != '"') return std::nullopt;
    std::string out;
    for (std::size_t i = *pos + 1; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            out.push_back(s[i + 1]);
            ++i;
            continue;
        }
        if (s[i] == '"') return out;
        out.push_back(s[i]);
    }
    return std::nullopt;
}

struct Child {
    pid_t pid = -1;
    int to_julia = -1;    // parent write
    int from_julia = -1;  // parent read
    int from_err = -1;    // parent read stderr
    std::string err_acc;
};

void close_fd(int& fd) {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

void kill_child(Child& c) {
    if (c.pid > 0) {
        ::kill(c.pid, SIGTERM);
        int status = 0;
        ::waitpid(c.pid, &status, 0);
        c.pid = -1;
    }
    close_fd(c.to_julia);
    close_fd(c.from_julia);
    close_fd(c.from_err);
}

bool spawn_julia(Child& c, const std::string& julia, const std::string& project,
                 const std::string& device, bool warmup) {
    int in_pipe[2], out_pipe[2], err_pipe[2];
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0 || pipe(err_pipe) != 0)
        return false;

    const std::string proj_arg = "--project=" + project;
    const std::string script = project + "/server.jl";
    const std::string device_arg = "--device=" + device;

    std::vector<std::string> argv_s{julia, proj_arg, "--startup-file=no", script,
                                    device_arg};
    if (!warmup) argv_s.push_back("--no-warmup");

    std::vector<char*> argv;
    argv.reserve(argv_s.size() + 1);
    for (auto& s : argv_s) argv.push_back(s.data());
    argv.push_back(nullptr);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in_pipe[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&fa, out_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&fa, err_pipe[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&fa, in_pipe[1]);
    posix_spawn_file_actions_addclose(&fa, out_pipe[0]);
    posix_spawn_file_actions_addclose(&fa, err_pipe[0]);

    pid_t pid = 0;
    const int rc =
        ::posix_spawnp(&pid, julia.c_str(), &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);
    if (rc != 0) {
        ::close(in_pipe[1]);
        ::close(out_pipe[0]);
        ::close(err_pipe[0]);
        return false;
    }

    c.pid = pid;
    c.to_julia = in_pipe[1];
    c.from_julia = out_pipe[0];
    c.from_err = err_pipe[0];
    // Non-blocking stderr drain.
    const int flags = ::fcntl(c.from_err, F_GETFL, 0);
    ::fcntl(c.from_err, F_SETFL, flags | O_NONBLOCK);
    return true;
}

void drain_err(Child& c) {
    char buf[4096];
    for (;;) {
        const ssize_t n = ::read(c.from_err, buf, sizeof(buf));
        if (n <= 0) break;
        c.err_acc.append(buf, static_cast<std::size_t>(n));
    }
}

bool await_ready(Child& c, double timeout_ms, double& wall_ready_ms,
                 std::string& timing_json) {
    const auto t0 = Clock::now();
    while (ms_since(t0) < timeout_ms) {
        drain_err(c);
        const auto pos = c.err_acc.find("SORGPU_READY");
        if (pos != std::string::npos) {
            wall_ready_ms = ms_since(t0);
            const auto tpos = c.err_acc.find("SORGPU_TIMING ");
            if (tpos != std::string::npos) {
                const auto line_end = c.err_acc.find('\n', tpos);
                const auto json_start = tpos + std::strlen("SORGPU_TIMING ");
                timing_json = c.err_acc.substr(
                    json_start, (line_end == std::string::npos ? c.err_acc.size()
                                                               : line_end) -
                                    json_start);
            }
            return true;
        }
        if (c.pid > 0) {
            int status = 0;
            const pid_t r = ::waitpid(c.pid, &status, WNOHANG);
            if (r == c.pid) {
                c.pid = -1;
                drain_err(c);
                std::fprintf(stderr, "sidecar exited during startup:\n%s\n",
                             c.err_acc.c_str());
                return false;
            }
        }
        ::usleep(20 * 1000);
    }
    drain_err(c);
    std::fprintf(stderr, "handshake timed out:\n%s\n", c.err_acc.c_str());
    return false;
}

bool write_all(int fd, const char* p, std::size_t n) {
    while (n > 0) {
        const ssize_t w = ::write(fd, p, n);
        if (w < 0) return false;
        p += w;
        n -= static_cast<std::size_t>(w);
    }
    return true;
}

bool send_json(Child& c, const std::string& body) {
    const std::string header = std::to_string(body.size()) + "\n";
    return write_all(c.to_julia, header.data(), header.size()) &&
           write_all(c.to_julia, body.data(), body.size());
}

bool read_exact(int fd, char* p, std::size_t n) {
    while (n > 0) {
        const ssize_t r = ::read(fd, p, n);
        if (r <= 0) return false;
        p += r;
        n -= static_cast<std::size_t>(r);
    }
    return true;
}

std::optional<std::string> recv_json(Child& c) {
    std::string header;
    char ch = 0;
    while (true) {
        if (!read_exact(c.from_julia, &ch, 1)) return std::nullopt;
        if (ch == '\n') break;
        header.push_back(ch);
        if (header.size() > 64) return std::nullopt;
    }
    const int n = std::atoi(header.c_str());
    if (n < 0 || n > 256 * 1024 * 1024) return std::nullopt;
    std::string body(static_cast<std::size_t>(n), '\0');
    if (!read_exact(c.from_julia, body.data(), body.size())) return std::nullopt;
    return body;
}

std::string csv_row(const std::vector<double>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ',';
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6g", v[i]);
        s += buf;
    }
    s += ']';
    return s;
}

void print_kv(const char* k, double v, const char* unit = "ms") {
    std::printf("  %-28s %10.3f %s\n", k, v, unit);
}

}  // namespace

int main(int argc, char** argv) {
    bool warmup = false;  // default OFF: we want first-compile visible
    std::string device = env_or("SOR_JULIA_DEVICE", "cpu");
    int n_rows = 4096;
    int warm_rpc = 30;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--warmup") warmup = true;
        else if (a == "--no-warmup") warmup = false;
        else if (a == "--device" && i + 1 < argc) device = argv[++i];
        else if (a.rfind("--device=", 0) == 0) device = a.substr(8);
        else if (a == "--n" && i + 1 < argc) n_rows = std::atoi(argv[++i]);
        else if (a == "--warm-rpc" && i + 1 < argc)
            warm_rpc = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::puts(
                "Usage: bench_julia_gpu_overhead [--no-warmup|--warmup] "
                "[--device cpu|auto|cuda] [--n 4096] [--warm-rpc 30]");
            return 0;
        }
    }

    const std::string julia = env_or("SOR_JULIA", "julia");
    std::string project = env_or("SOR_JULIA_PROJECT", "");
    if (project.empty()) {
        // Prefer colocated tools/julia_gpu relative to this binary's CWD.
        project = "tools/julia_gpu";
        if (::access((project + "/server.jl").c_str(), R_OK) != 0)
            project = "sor/tools/julia_gpu";
    }
    if (::access((project + "/server.jl").c_str(), R_OK) != 0) {
        std::fprintf(stderr, "cannot read %s/server.jl (cwd + SOR_JULIA_PROJECT)\n",
                     project.c_str());
        return 1;
    }

    std::printf("=== Julia GPU caller overhead (C++ → sidecar) ===\n");
    std::printf("julia=%s  project=%s  device=%s  warmup=%s  n=%d\n\n",
                julia.c_str(), project.c_str(), device.c_str(),
                warmup ? "true" : "false", n_rows);

    Child child;
    const auto t_spawn = Clock::now();
    if (!spawn_julia(child, julia, project, device, warmup)) {
        std::fprintf(stderr, "posix_spawn failed\n");
        return 1;
    }

    double wall_ready_ms = 0.0;
    std::string timing_json;
    if (!await_ready(child, /*timeout_ms=*/300000.0, wall_ready_ms,
                     timing_json)) {
        kill_child(child);
        return 1;
    }
    const double spawn_to_ready_ms = ms_since(t_spawn);

    std::printf("-- process startup (C++ wall) --\n");
    print_kv("spawn_to_ready_wall", spawn_to_ready_ms);
    if (!timing_json.empty()) {
        if (auto v = get_number(timing_json, "select_device_ms"))
            print_kv("julia_select_device", *v);
        if (auto v = get_number(timing_json, "warmup_ms"))
            print_kv("julia_warmup_kernels", *v);
        if (auto v = get_number(timing_json, "ready_ms"))
            print_kv("julia_ready_internal", *v);
        std::printf("  %-28s %s\n", "accelerated",
                    get_bool(timing_json, "accelerated", false) ? "true"
                                                                : "false");
        if (auto d = get_string(timing_json, "device"))
            std::printf("  %-28s %s\n", "device", d->c_str());
    }
    std::printf("\n");

    // --- ping RTT floor ---
    std::vector<double> ping_ms;
    ping_ms.reserve(static_cast<std::size_t>(warm_rpc));
    for (int i = 0; i < warm_rpc; ++i) {
        const auto t0 = Clock::now();
        if (!send_json(child, "{\"op\":\"ping\",\"id\":" + std::to_string(i) +
                                  "}")) {
            std::fprintf(stderr, "ping send failed\n");
            kill_child(child);
            return 1;
        }
        auto resp = recv_json(child);
        if (!resp) {
            std::fprintf(stderr, "ping recv failed\n");
            kill_child(child);
            return 1;
        }
        ping_ms.push_back(ms_since(t0));
    }
    // First ping pays residual Julia specialize on the request path.
    const int ping_skip = std::min(5, std::max(0, static_cast<int>(ping_ms.size()) - 1));
    double ping_sum = 0.0, ping_min = ping_ms[static_cast<std::size_t>(ping_skip)];
    for (std::size_t i = static_cast<std::size_t>(ping_skip); i < ping_ms.size();
         ++i) {
        ping_sum += ping_ms[i];
        ping_min = std::min(ping_min, ping_ms[i]);
    }
    const double ping_mean =
        ping_sum / static_cast<double>(ping_ms.size() - ping_skip);
    std::printf("-- IPC floor (ping RTT, n=%d, skip_first=%d) --\n", warm_rpc,
                ping_skip);
    print_kv("ping_first", ping_ms.front());
    print_kv("ping_mean_steady", ping_mean);
    print_kv("ping_min", ping_min);
    std::printf("\n");

    // --- Julia-reported compile suite (one RPC; times are inside Julia) ---
    {
        const std::string req =
            "{\"op\":\"bench_overhead\",\"id\":100,\"n_rows\":" +
            std::to_string(n_rows) + ",\"n_cols\":" + std::to_string(n_rows) +
            ",\"nnz_per_row\":8,\"warm_iters\":20,\"ak_n\":1048576}";
        const auto t0 = Clock::now();
        if (!send_json(child, req)) {
            std::fprintf(stderr, "bench_overhead send failed\n");
            kill_child(child);
            return 1;
        }
        auto resp = recv_json(child);
        const double wall = ms_since(t0);
        if (!resp || !get_bool(*resp, "ok", false)) {
            std::fprintf(stderr, "bench_overhead failed: %s\n",
                         resp ? resp->c_str() : "(null)");
            kill_child(child);
            return 1;
        }
        std::printf("-- Julia suite via C++ RPC (wall includes IPC) --\n");
        print_kv("rpc_wall", wall);
        if (auto v = get_number(*resp, "ka_first_kernel_ms"))
            print_kv("ka_spmv_first_kernel", *v);
        if (auto v = get_number(*resp, "ka_warm_mean_ms"))
            print_kv("ka_spmv_warm_mean", *v);
        if (auto v = get_number(*resp, "ka_compile_proxy_ms"))
            print_kv("ka_spmv_compile_proxy", *v);
        if (get_bool(*resp, "ak_skipped", false)) {
            std::printf("  %-28s %s\n", "ak_foreachindex",
                        "skipped (not installed)");
        } else {
            if (auto v = get_number(*resp, "ak_first_kernel_ms"))
                print_kv("ak_first_kernel", *v);
            if (auto v = get_number(*resp, "ak_warm_mean_ms"))
                print_kv("ak_warm_mean", *v);
            if (auto v = get_number(*resp, "ak_compile_proxy_ms"))
                print_kv("ak_compile_proxy", *v);
        }
        std::printf("\n");
        std::printf("-- raw bench_overhead response (truncated) --\n");
        if (resp->size() > 1200)
            std::printf("%s...\n\n", resp->substr(0, 1200).c_str());
        else
            std::printf("%s\n\n", resp->c_str());
    }

    // --- Per-call SpMV from C++: wall vs kernel_ms ---
    // Identity-ish small pattern already compiled by bench_overhead if
    // --no-warmup; for a fair "warm RPC" we use the same n.
    {
        // Build a tiny CSR JSON for upload (n x n diagonal-ish, 1 nnz/row).
        std::string row_ptr = "[0";
        std::string col_idx = "[";
        std::string vals = "[";
        for (int r = 0; r < n_rows; ++r) {
            row_ptr += "," + std::to_string(r + 1);
            if (r) {
                col_idx += ",";
                vals += ",";
            }
            col_idx += std::to_string(r % n_rows);
            vals += "1.0";
        }
        row_ptr += "]";
        col_idx += "]";
        vals += "]";

        auto t0 = Clock::now();
        if (!send_json(child,
                       "{\"op\":\"upload_pattern\",\"id\":200,\"n_rows\":" +
                           std::to_string(n_rows) + ",\"n_cols\":" +
                           std::to_string(n_rows) + ",\"row_ptr\":" + row_ptr +
                           ",\"col_idx\":" + col_idx + "}")) {
            std::fprintf(stderr, "upload_pattern failed\n");
            kill_child(child);
            return 1;
        }
        auto r1 = recv_json(child);
        const double upload_pat_wall = ms_since(t0);
        if (!r1 || !get_bool(*r1, "ok", false)) {
            std::fprintf(stderr, "upload_pattern bad: %s\n",
                         r1 ? r1->c_str() : "(null)");
            kill_child(child);
            return 1;
        }
        const auto pid = get_int(*r1, "pattern_id").value_or(1);

        t0 = Clock::now();
        if (!send_json(child, "{\"op\":\"upload_vals\",\"id\":201,\"pattern_id\":" +
                                  std::to_string(pid) + ",\"vals\":" + vals +
                                  "}")) {
            kill_child(child);
            return 1;
        }
        auto r2 = recv_json(child);
        const double upload_vals_wall = ms_since(t0);
        const auto vid = get_int(*r2, "vals_id").value_or(1);

        std::string xjson = "[";
        for (int i = 0; i < n_rows; ++i) {
            if (i) xjson += ",";
            xjson += "1.0";
        }
        xjson += "]";

        const std::string spmv_req =
            "{\"op\":\"spmv\",\"id\":202,\"pattern_id\":" + std::to_string(pid) +
            ",\"vals_id\":" + std::to_string(vid) + ",\"x\":" + xjson + "}";

        // First RPC SpMV after suite (kernel already compiled if suite ran).
        t0 = Clock::now();
        send_json(child, spmv_req);
        auto r3 = recv_json(child);
        const double first_spmv_wall = ms_since(t0);
        const double first_kernel =
            get_number(*r3, "kernel_ms").value_or(std::nan(""));
        const double first_h2d = get_number(*r3, "h2d_ms").value_or(0.0);
        const double first_d2h = get_number(*r3, "d2h_ms").value_or(0.0);

        std::vector<double> walls, kernels, ipcs;
        for (int i = 0; i < warm_rpc; ++i) {
            t0 = Clock::now();
            send_json(child, spmv_req);
            auto r = recv_json(child);
            const double wall = ms_since(t0);
            const double k = get_number(*r, "kernel_ms").value_or(0.0);
            const double h2d = get_number(*r, "h2d_ms").value_or(0.0);
            const double d2h = get_number(*r, "d2h_ms").value_or(0.0);
            walls.push_back(wall);
            kernels.push_back(k);
            ipcs.push_back(std::max(0.0, wall - k - h2d - d2h));
        }
        auto mean = [](const std::vector<double>& v) {
            double s = 0;
            for (double x : v) s += x;
            return s / v.size();
        };
        auto vmin = [](const std::vector<double>& v) {
            double m = v[0];
            for (double x : v) m = std::min(m, x);
            return m;
        };

        std::printf("-- C++→Julia SpMV RPC (n=%d, diagonal CSR) --\n", n_rows);
        print_kv("upload_pattern_wall", upload_pat_wall);
        print_kv("upload_vals_wall", upload_vals_wall);
        print_kv("spmv_first_wall", first_spmv_wall);
        print_kv("spmv_first_kernel", first_kernel);
        print_kv("spmv_first_h2d", first_h2d);
        print_kv("spmv_first_d2h", first_d2h);
        print_kv("spmv_first_ipc_proxy",
                 std::max(0.0, first_spmv_wall - first_kernel - first_h2d -
                                   first_d2h));
        print_kv("spmv_warm_wall_mean", mean(walls));
        print_kv("spmv_warm_kernel_mean", mean(kernels));
        print_kv("spmv_warm_ipc_proxy_mean", mean(ipcs));
        print_kv("spmv_warm_wall_min", vmin(walls));
        print_kv("spmv_warm_ipc_proxy_min", vmin(ipcs));
        std::printf("\n");

        std::printf(
            "Interpretation:\n"
            "  spawn_to_ready_wall  ≈ Julia process + package load"
            "%s\n"
            "  ka_spmv_compile_proxy ≈ first_kernel - warm_mean (JIT/specialize)\n"
            "  ping_min / spmv_warm_ipc_proxy ≈ JSON-over-pipe floor per call\n"
            "  No SIH GPU claim rests on this path.\n",
            warmup ? " + kernel warmup" : " (compile deferred to first kernel)");
    }

    send_json(child, "{\"op\":\"shutdown\",\"id\":999}");
    (void)recv_json(child);
    kill_child(child);
    return 0;
}
