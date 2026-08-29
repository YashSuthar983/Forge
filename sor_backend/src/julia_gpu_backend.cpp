#include "sor/backend/julia_gpu_backend.hpp"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef SOR_JULIA_PROJECT_DIR
#define SOR_JULIA_PROJECT_DIR ""
#endif

namespace sor::backend {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Minimal JSON extraction.
//
// Deliberately not a general JSON library: the only producer is our own Julia
// server, and adding a JSON dependency to the numeric core is not worth it for a
// prototype. Keys are searched at brace depth 1 with string-awareness, so key
// order (which JSON3 does not guarantee) does not matter.
// ---------------------------------------------------------------------------
class JsonView {
public:
    explicit JsonView(const std::string& s) : s_(s) {}

    std::optional<std::size_t> find_key(std::string_view key) const {
        int depth = 0;
        bool in_str = false;
        for (std::size_t i = 0; i < s_.size(); ++i) {
            const char c = s_[i];
            if (in_str) {
                if (c == '\\') { ++i; continue; }
                if (c == '"') in_str = false;
                continue;
            }
            if (c == '"') {
                // Candidate key at depth 1.
                const std::size_t start = i + 1;
                std::size_t j = start;
                bool esc = false;
                for (; j < s_.size(); ++j) {
                    if (esc) { esc = false; continue; }
                    if (s_[j] == '\\') { esc = true; continue; }
                    if (s_[j] == '"') break;
                }
                if (depth == 1 && s_.compare(start, j - start, key) == 0) {
                    std::size_t k = j + 1;
                    while (k < s_.size() && (std::isspace(static_cast<unsigned char>(s_[k])))) ++k;
                    if (k < s_.size() && s_[k] == ':') {
                        ++k;
                        while (k < s_.size() &&
                               std::isspace(static_cast<unsigned char>(s_[k]))) ++k;
                        return k;
                    }
                }
                i = j;
                continue;
            }
            if (c == '{' || c == '[') ++depth;
            else if (c == '}' || c == ']') --depth;
        }
        return std::nullopt;
    }

    bool bool_at(std::size_t p) const { return s_.compare(p, 4, "true") == 0; }
    bool is_null_at(std::size_t p) const { return s_.compare(p, 4, "null") == 0; }

    double double_at(std::size_t p) const {
        return std::strtod(s_.c_str() + p, nullptr);
    }

    std::string string_at(std::size_t p) const {
        if (p >= s_.size() || s_[p] != '"') return {};
        std::string out;
        bool esc = false;
        for (std::size_t i = p + 1; i < s_.size(); ++i) {
            const char c = s_[i];
            if (esc) {
                switch (c) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    default:  out.push_back(c);    break;
                }
                esc = false;
                continue;
            }
            if (c == '\\') { esc = true; continue; }
            if (c == '"') break;
            out.push_back(c);
        }
        return out;
    }

    // Parses [a, b, c] of numbers. Accepts null/true/false as NaN-ish guards.
    bool double_array_at(std::size_t p, std::vector<double>& out) const {
        if (p >= s_.size() || s_[p] != '[') return false;
        out.clear();
        const char* cur = s_.c_str() + p + 1;
        const char* end = s_.c_str() + s_.size();
        while (cur < end) {
            while (cur < end && (std::isspace(static_cast<unsigned char>(*cur)) ||
                                 *cur == ',')) ++cur;
            if (cur >= end || *cur == ']') break;
            char* stop = nullptr;
            const double v = std::strtod(cur, &stop);
            if (stop == cur) return false;   // unparseable token
            out.push_back(v);
            cur = stop;
        }
        return true;
    }

    std::optional<double> stat(std::string_view field) const {
        // stats is a nested object, so depth-1 search will not find its members.
        const auto sp = find_key("stats");
        if (!sp) return std::nullopt;
        const std::string sub = s_.substr(*sp);
        JsonView inner(sub);
        // Inside `sub` the stats object itself is depth 1.
        const auto fp = inner.find_key(field);
        if (!fp) return std::nullopt;
        return inner.double_at(*fp);
    }

private:
    const std::string& s_;
};

// %.17g round-trips an IEEE double exactly.
void append_double(std::string& out, double v) {
    if (std::isnan(v))      { out += "null"; return; }
    if (std::isinf(v))      { out += (v > 0 ? "1e999" : "-1e999"); return; }
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    out += buf;
}

void append_double_array(std::string& out, const double* p, std::size_t n) {
    out.push_back('[');
    for (std::size_t i = 0; i < n; ++i) {
        if (i) out.push_back(',');
        append_double(out, p[i]);
    }
    out.push_back(']');
}

template <class T>
void append_int_array(std::string& out, const std::vector<T>& v) {
    out.push_back('[');
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out.push_back(',');
        out += std::to_string(static_cast<long long>(v[i]));
    }
    out.push_back(']');
}

// FNV-1a over raw bytes. Used to detect when a pattern or a value array has
// changed and must be re-uploaded.
std::uint64_t fnv1a(const void* data, std::size_t n, std::uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

std::string env_or(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

bool executable_exists(const std::string& name) {
    if (name.find('/') != std::string::npos)
        return ::access(name.c_str(), X_OK) == 0;
    const char* path = std::getenv("PATH");
    if (!path) return false;
    std::stringstream ss(path);
    std::string dir;
    while (std::getline(ss, dir, ':')) {
        if (dir.empty()) continue;
        if (::access((dir + "/" + name).c_str(), X_OK) == 0) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------
class JuliaGpuBackend final : public KernelBackend {
public:
    JuliaGpuBackend() = default;

    ~JuliaGpuBackend() override {
        if (pid_ > 0) {
            try { request(R"({"op":"shutdown","id":0})"); } catch (...) {}
            if (to_child_ >= 0) ::close(to_child_);
            if (from_child_ >= 0) ::close(from_child_);
            int status = 0;
            // Give it a moment, then insist.
            for (int i = 0; i < 50; ++i) {
                if (::waitpid(pid_, &status, WNOHANG) == pid_) { pid_ = -1; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (pid_ > 0) { ::kill(pid_, SIGTERM); ::waitpid(pid_, &status, 0); }
        }
        if (!stderr_path_.empty()) ::unlink(stderr_path_.c_str());
    }

    // Spawns the sidecar and completes the handshake. False on any failure.
    bool start() {
        const std::string julia = env_or("SOR_JULIA", "julia");
        project_ = env_or("SOR_JULIA_PROJECT", SOR_JULIA_PROJECT_DIR);
        const std::string device = env_or("SOR_JULIA_DEVICE", "auto");

        if (project_.empty()) return false;
        const std::string script = project_ + "/server.jl";
        if (::access(script.c_str(), R_OK) != 0) return false;
        if (!executable_exists(julia)) return false;

        int in_pipe[2], out_pipe[2];
        if (::pipe(in_pipe) != 0) return false;
        if (::pipe(out_pipe) != 0) {
            ::close(in_pipe[0]); ::close(in_pipe[1]);
            return false;
        }

        char tmpl[] = "/tmp/sor_julia_stderr_XXXXXX";
        const int errfd = ::mkstemp(tmpl);
        if (errfd < 0) {
            ::close(in_pipe[0]); ::close(in_pipe[1]);
            ::close(out_pipe[0]); ::close(out_pipe[1]);
            return false;
        }
        stderr_path_ = tmpl;

        const std::string proj_arg = "--project=" + project_;
        std::vector<std::string> argv_s{julia, proj_arg, "--startup-file=no",
                                        script, "--device", device};
        std::vector<char*> argv;
        for (auto& s : argv_s) argv.push_back(const_cast<char*>(s.c_str()));
        argv.push_back(nullptr);

        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, in_pipe[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&fa, out_pipe[1], STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&fa, errfd, STDERR_FILENO);
        posix_spawn_file_actions_addclose(&fa, in_pipe[1]);
        posix_spawn_file_actions_addclose(&fa, out_pipe[0]);

        pid_t pid = -1;
        const int rc = ::posix_spawnp(&pid, julia.c_str(), &fa, nullptr,
                                      argv.data(), environ);
        posix_spawn_file_actions_destroy(&fa);
        ::close(in_pipe[0]);
        ::close(out_pipe[1]);
        ::close(errfd);

        if (rc != 0) {
            ::close(in_pipe[1]);
            ::close(out_pipe[0]);
            return false;
        }
        pid_ = pid;
        to_child_ = in_pipe[1];
        from_child_ = out_pipe[0];

        if (!await_ready()) return false;

        // Handshake: ping must answer and tell us whether a GPU is really live.
        try {
            const std::string resp = request(R"({"op":"ping","id":1})");
            JsonView v(resp);
            const auto okp = v.find_key("ok");
            if (!okp || !v.bool_at(*okp)) return false;
            if (const auto d = v.find_key("device")) device_name_ = v.string_at(*d);
            if (const auto a = v.find_key("accelerated")) accelerated_ = v.bool_at(*a);
            return true;
        } catch (...) {
            return false;
        }
    }

    std::string_view name() const override { return "julia_gpu"; }
    bool is_accelerated() const override { return accelerated_; }
    std::string_view device_name() const { return device_name_; }

    // ---- kernels --------------------------------------------------------
    void spmv(const SparsePattern& p, const DeviceBuffer<f64>& vals,
              const DeviceBuffer<f64>& x, DeviceBuffer<f64>& y) override {
        do_spmv("spmv", p, vals, x, y,
                static_cast<std::size_t>(p.n_cols()),
                static_cast<std::size_t>(p.n_rows()));
    }

    void spmv_t(const SparsePattern& p, const DeviceBuffer<f64>& vals,
                const DeviceBuffer<f64>& x, DeviceBuffer<f64>& y) override {
        do_spmv("spmv_t", p, vals, x, y,
                static_cast<std::size_t>(p.n_rows()),
                static_cast<std::size_t>(p.n_cols()));
    }

    void project_box(DeviceBuffer<f64>& x, const DeviceBuffer<f64>& lo,
                     const DeviceBuffer<f64>& hi) override {
        if (x.size() != lo.size() || x.size() != hi.size())
            throw std::invalid_argument("project_box: mismatched sizes");
        std::string req = R"({"op":"project_box","id":)" + std::to_string(++msg_id_);
        req += R"(,"x":)";  append_double_array(req, x.host().data(), x.size());
        req += R"(,"lo":)"; append_double_array(req, lo.host().data(), lo.size());
        req += R"(,"hi":)"; append_double_array(req, hi.host().data(), hi.size());
        req += "}";
        const auto resp = call(req);
        JsonView v(resp);
        const auto xp = v.find_key("x");
        std::vector<double> out;
        if (!xp || !v.double_array_at(*xp, out))
            throw std::runtime_error("julia_gpu: project_box returned no x");
        if (out.size() != x.size())
            throw std::runtime_error("julia_gpu: project_box size mismatch");
        x.host() = std::move(out);
    }

    f64 dot(const DeviceBuffer<f64>& a, const DeviceBuffer<f64>& b) override {
        if (a.size() != b.size())
            throw std::invalid_argument("dot: mismatched sizes");
        std::string req = R"({"op":"dot","id":)" + std::to_string(++msg_id_);
        req += R"(,"a":)"; append_double_array(req, a.host().data(), a.size());
        req += R"(,"b":)"; append_double_array(req, b.host().data(), b.size());
        req += "}";
        const auto resp = call(req);
        JsonView v(resp);
        const auto vp = v.find_key("value");
        if (!vp) throw std::runtime_error("julia_gpu: dot returned no value");
        return v.double_at(*vp);
    }

    void spmv_batched(const SparsePattern& p, const DeviceBuffer<f64>& vals,
                      const BatchView<f64>& X, BatchView<f64>& Y) override {
        if (X.item_len() != static_cast<std::size_t>(p.n_cols()))
            throw std::invalid_argument("spmv_batched: X item_len != n_cols");
        if (Y.item_len() != static_cast<std::size_t>(p.n_rows()))
            throw std::invalid_argument("spmv_batched: Y item_len != n_rows");
        if (X.n_items() != Y.n_items())
            throw std::invalid_argument("spmv_batched: batch count mismatch");

        const auto ids = ensure_uploaded(p, vals);
        std::string req = R"({"op":"spmv_batched","id":)" + std::to_string(++msg_id_);
        req += R"(,"pattern_id":)" + std::to_string(ids.first);
        req += R"(,"vals_id":)" + std::to_string(ids.second);
        req += R"(,"n_items":)" + std::to_string(X.n_items());
        req += R"(,"X":)"; append_double_array(req, X.data(), X.total());
        req += "}";
        const auto resp = call(req);
        JsonView v(resp);
        std::vector<double> out;
        const auto yp = v.find_key("Y");
        if (!yp || !v.double_array_at(*yp, out))
            throw std::runtime_error("julia_gpu: spmv_batched returned no Y");
        if (out.size() != Y.total())
            throw std::runtime_error("julia_gpu: spmv_batched size mismatch");
        std::copy(out.begin(), out.end(), Y.data());
    }

    void project_box_batched(BatchView<f64>& X, const BatchView<f64>& LO,
                             const BatchView<f64>& HI) override {
        if (X.total() != LO.total() || X.total() != HI.total())
            throw std::invalid_argument("project_box_batched: mismatched shapes");
        std::string req = R"({"op":"project_box_batched","id":)" +
                          std::to_string(++msg_id_);
        req += R"(,"X":)";  append_double_array(req, X.data(), X.total());
        req += R"(,"LO":)"; append_double_array(req, LO.data(), LO.total());
        req += R"(,"HI":)"; append_double_array(req, HI.data(), HI.total());
        req += "}";
        const auto resp = call(req);
        JsonView v(resp);
        std::vector<double> out;
        const auto xp = v.find_key("X");
        if (!xp || !v.double_array_at(*xp, out))
            throw std::runtime_error("julia_gpu: project_box_batched returned no X");
        if (out.size() != X.total())
            throw std::runtime_error("julia_gpu: project_box_batched size mismatch");
        std::copy(out.begin(), out.end(), X.data());
    }

    TransferStats transfer_stats() const override { return stats_; }
    void reset_stats() override { stats_ = TransferStats{}; }

private:
    // ---- pattern / value handle caching ---------------------------------
    // The prompt's v0 protocol inlines the CSR on every call. PDHG does two
    // SpMVs per iteration for thousands of iterations, so that would make JSON
    // serialisation the entire measurement. Upload once, reference by handle.
    std::pair<int, int> ensure_uploaded(const SparsePattern& p,
                                       const DeviceBuffer<f64>& vals) {
        std::uint64_t ph = fnv1a(&p, 0);   // seed
        ph = fnv1a(p.row_ptr().data(),
                   p.row_ptr().size() * sizeof(core::Offset), ph);
        ph = fnv1a(p.col_idx().data(),
                   p.col_idx().size() * sizeof(core::Index), ph);
        const int nr = p.n_rows(), nc = p.n_cols();
        ph = fnv1a(&nr, sizeof(nr), ph);
        ph = fnv1a(&nc, sizeof(nc), ph);

        int pattern_id;
        auto pit = pattern_ids_.find(ph);
        if (pit != pattern_ids_.end()) {
            pattern_id = pit->second;
        } else {
            std::string req = R"({"op":"upload_pattern","id":)" +
                              std::to_string(++msg_id_);
            req += R"(,"n_rows":)" + std::to_string(p.n_rows());
            req += R"(,"n_cols":)" + std::to_string(p.n_cols());
            req += R"(,"row_ptr":)"; append_int_array(req, p.row_ptr());
            req += R"(,"col_idx":)"; append_int_array(req, p.col_idx());
            req += "}";
            const auto resp = call(req);
            JsonView v(resp);
            const auto idp = v.find_key("pattern_id");
            if (!idp) throw std::runtime_error("julia_gpu: no pattern_id returned");
            pattern_id = static_cast<int>(v.double_at(*idp));
            pattern_ids_[ph] = pattern_id;
        }

        const std::uint64_t vh =
            fnv1a(vals.host().data(), vals.size() * sizeof(f64), ph);
        int vals_id;
        auto vit = vals_ids_.find(vh);
        if (vit != vals_ids_.end()) {
            vals_id = vit->second;
        } else {
            std::string req = R"({"op":"upload_vals","id":)" +
                              std::to_string(++msg_id_);
            req += R"(,"pattern_id":)" + std::to_string(pattern_id);
            req += R"(,"vals":)";
            append_double_array(req, vals.host().data(), vals.size());
            req += "}";
            const auto resp = call(req);
            JsonView v(resp);
            const auto idp = v.find_key("vals_id");
            if (!idp) throw std::runtime_error("julia_gpu: no vals_id returned");
            vals_id = static_cast<int>(v.double_at(*idp));
            vals_ids_[vh] = vals_id;
        }
        return {pattern_id, vals_id};
    }

    void do_spmv(const char* op, const SparsePattern& p,
                 const DeviceBuffer<f64>& vals, const DeviceBuffer<f64>& x,
                 DeviceBuffer<f64>& y, std::size_t want_x, std::size_t want_y) {
        if (vals.size() != static_cast<std::size_t>(p.nnz()))
            throw std::invalid_argument("kernel: vals.size() != nnz");
        if (x.size() != want_x)
            throw std::invalid_argument(std::string(op) + ": bad x size");

        const auto ids = ensure_uploaded(p, vals);
        std::string req = std::string(R"({"op":")") + op + R"(","id":)" +
                          std::to_string(++msg_id_);
        req += R"(,"pattern_id":)" + std::to_string(ids.first);
        req += R"(,"vals_id":)" + std::to_string(ids.second);
        req += R"(,"x":)"; append_double_array(req, x.host().data(), x.size());
        req += "}";
        const auto resp = call(req);

        JsonView v(resp);
        std::vector<double> out;
        const auto yp = v.find_key("y");
        if (!yp || !v.double_array_at(*yp, out))
            throw std::runtime_error(std::string("julia_gpu: ") + op +
                                     " returned no y");
        if (out.size() != want_y)
            throw std::runtime_error(std::string("julia_gpu: ") + op +
                                     " size mismatch");
        y.host() = std::move(out);
    }

    // Sends a request, checks ok, folds the reported stats in.
    std::string call(const std::string& req) {
        const auto t0 = Clock::now();
        const std::string resp = request(req);
        const double round_trip = ms_since(t0);

        JsonView v(resp);
        const auto okp = v.find_key("ok");
        if (!okp || !v.bool_at(*okp)) {
            std::string msg = "julia_gpu: sidecar error";
            if (const auto ep = v.find_key("error")) {
                if (!v.is_null_at(*ep)) msg += ": " + v.string_at(*ep);
            }
            throw std::runtime_error(msg);
        }

        TransferStats s;
        if (const auto x = v.stat("h2d_bytes")) s.h2d_bytes = static_cast<std::uint64_t>(*x);
        if (const auto x = v.stat("d2h_bytes")) s.d2h_bytes = static_cast<std::uint64_t>(*x);
        if (const auto x = v.stat("h2d_ms"))    s.h2d_ms = *x;
        if (const auto x = v.stat("d2h_ms"))    s.d2h_ms = *x;
        if (const auto x = v.stat("kernel_ms")) s.kernel_ms = *x;
        // Everything the sidecar did NOT account for is IPC overhead: JSON
        // encode/decode plus pipe latency. Reported separately and never folded
        // into kernel time.
        s.ipc_ms = round_trip - (s.h2d_ms + s.d2h_ms + s.kernel_ms);
        if (s.ipc_ms < 0.0) s.ipc_ms = 0.0;
        s.calls = 1;
        stats_.add(s);
        return resp;
    }

    // Length-prefixed framing: "<n>\n<n bytes of JSON>".
    std::string request(const std::string& body) {
        const std::string header = std::to_string(body.size()) + "\n";
        write_all(header);
        write_all(body);

        const std::string len_line = read_line();
        if (len_line.empty()) throw std::runtime_error("julia_gpu: sidecar closed");
        const long n = std::strtol(len_line.c_str(), nullptr, 10);
        if (n < 0) throw std::runtime_error("julia_gpu: bad frame length");
        std::string out(static_cast<std::size_t>(n), '\0');
        std::size_t got = 0;
        while (got < static_cast<std::size_t>(n)) {
            const ssize_t r = ::read(from_child_, out.data() + got,
                                     static_cast<std::size_t>(n) - got);
            if (r < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("julia_gpu: read failed");
            }
            if (r == 0) throw std::runtime_error("julia_gpu: short frame (EOF)");
            got += static_cast<std::size_t>(r);
        }
        return out;
    }

    void write_all(const std::string& s) {
        std::size_t off = 0;
        while (off < s.size()) {
            const ssize_t w = ::write(to_child_, s.data() + off, s.size() - off);
            if (w < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("julia_gpu: write failed");
            }
            off += static_cast<std::size_t>(w);
        }
    }

    std::string read_line() {
        std::string line;
        char c;
        while (true) {
            const ssize_t r = ::read(from_child_, &c, 1);
            if (r < 0) {
                if (errno == EINTR) continue;
                return {};
            }
            if (r == 0) return line;          // EOF
            if (c == '\n') return line;
            line.push_back(c);
        }
    }

    // Julia's JIT warmup takes seconds. Wait for the READY marker on the
    // sidecar's stderr rather than racing it.
    bool await_ready() {
        const auto deadline = Clock::now() + std::chrono::seconds(180);
        while (Clock::now() < deadline) {
            std::ifstream f(stderr_path_);
            std::string all((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
            if (all.find("SORGPU_READY") != std::string::npos) return true;

            int status = 0;
            if (::waitpid(pid_, &status, WNOHANG) == pid_) {
                pid_ = -1;
                std::fprintf(stderr,
                             "julia_gpu: sidecar exited during startup:\n%s\n",
                             all.c_str());
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::fprintf(stderr, "julia_gpu: sidecar handshake timed out\n");
        return false;
    }

    pid_t pid_ = -1;
    int to_child_ = -1, from_child_ = -1;
    std::string project_, stderr_path_;
    std::string device_name_ = "unknown";
    bool accelerated_ = false;
    long msg_id_ = 1;

    std::unordered_map<std::uint64_t, int> pattern_ids_;
    std::unordered_map<std::uint64_t, int> vals_ids_;
    TransferStats stats_{};
};

}  // namespace

std::unique_ptr<KernelBackend> make_julia_gpu_backend() {
    auto be = std::make_unique<JuliaGpuBackend>();
    if (!be->start()) return nullptr;
    return be;
}

}  // namespace sor::backend
