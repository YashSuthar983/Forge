#include "bound_snapshots.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <unistd.h>

namespace sor::search::detail {
namespace {
using core::f64;
using core::Index;

std::atomic<std::uint64_t> milp_invocation_seq{0};

// Append-only identity for every solve_milp that runs while a bounds dump is
// requested. Each numeric snapshot has a run/invocation/phase-specific path;
// the log records its owner and the children that did not write a snapshot.
const char* milp_dump_base_path() {
    if (const char* p = std::getenv("SOR_DUMP_MIP_BOUNDS"))
        if (*p) return p;
    if (const char* p = std::getenv("SOR_DUMP_ROOT_BOUNDS0"))
        if (*p) return p;
    if (const char* p = std::getenv("SOR_DUMP_ROOT_BOUNDS"))
        if (*p) return p;
    return nullptr;
}

std::uint64_t fnv_mix(std::uint64_t fp, std::uint64_t x) {
    fp ^= x;
    fp *= 1099511628211ull;
    return fp;
}

std::uint64_t fnv_mix_f64(std::uint64_t fp, double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return fnv_mix(fp, bits);
}

std::uint64_t fnv_mix_string(std::uint64_t fp, const std::string& s) {
    fp = fnv_mix(fp, s.size());
    for (unsigned char c : s) fp = fnv_mix(fp, c);
    return fp;
}

// Full model identity: names, order, bounds, integrality, objective, and
// every coefficient. Dimensions alone collide for different restrictions.
std::uint64_t milp_content_fingerprint(const model::LpProblem& model) {
    std::uint64_t fp = 14695981039346656037ull;
    fp = fnv_mix_string(fp, model.name);
    fp = fnv_mix(fp, static_cast<std::uint64_t>(model.n_rows()));
    fp = fnv_mix(fp, static_cast<std::uint64_t>(model.n_cols()));
    fp = fnv_mix(fp, static_cast<std::uint64_t>(model.nnz()));
    fp = fnv_mix(fp, model.maximize ? 1ull : 0ull);
    fp = fnv_mix_f64(fp, model.obj_offset);
    for (std::size_t j = 0; j < model.c.size(); ++j) fp = fnv_mix_f64(fp, model.c[j]);
    for (double v : model.col_lo) fp = fnv_mix_f64(fp, v);
    for (double v : model.col_hi) fp = fnv_mix_f64(fp, v);
    for (double v : model.row_lo) fp = fnv_mix_f64(fp, v);
    for (double v : model.row_hi) fp = fnv_mix_f64(fp, v);
    for (std::size_t j = 0; j < model.is_integer.size(); ++j)
        fp = fnv_mix(fp, model.is_integer[j] ? 1ull : 0ull);
    for (const std::string& name : model.col_names) fp = fnv_mix_string(fp, name);
    for (const std::string& name : model.row_names) fp = fnv_mix_string(fp, name);
    const auto& rp = model.A.pattern.row_ptr();
    const auto& ci = model.A.pattern.col_idx();
    for (std::size_t i = 0; i + 1 < rp.size(); ++i) {
        for (core::Offset k = rp[i]; k < rp[i + 1]; ++k) {
            fp = fnv_mix(fp, static_cast<std::uint64_t>(i));
            fp = fnv_mix(fp, static_cast<std::uint64_t>(ci[static_cast<std::size_t>(k)]));
            fp = fnv_mix_f64(fp, model.A.vals[static_cast<std::size_t>(k)]);
        }
    }
    return fp;
}

// Ordered column identity. Snapshot row j is this column j.
std::uint64_t milp_column_order_fingerprint(const model::LpProblem& model) {
    std::uint64_t fp = 14695981039346656037ull;
    fp = fnv_mix(fp, static_cast<std::uint64_t>(model.n_cols()));
    for (Index j = 0; j < model.n_cols(); ++j) {
        fp = fnv_mix(fp, static_cast<std::uint64_t>(j));
        const bool integer = static_cast<std::size_t>(j) < model.is_integer.size() &&
                             model.is_integer[static_cast<std::size_t>(j)];
        fp = fnv_mix(fp, integer ? 1ull : 0ull);
        if (static_cast<std::size_t>(j) < model.col_names.size())
            fp = fnv_mix_string(fp, model.col_names[static_cast<std::size_t>(j)]);
    }
    return fp;
}

std::string milp_run_id() {
    static const std::string id = [] {
        std::random_device rd;
        std::uint64_t mixed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
        mixed ^= static_cast<std::uint64_t>(::getpid());
        char buf[17];
        std::snprintf(buf, sizeof buf, "%016llx",
                      static_cast<unsigned long long>(mixed));
        return std::string(buf);
    }();
    return id;
}

void sha256_hex(const std::string& bytes, char out[65]) {
    static constexpr std::uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    auto rotr = [](std::uint32_t x, std::uint32_t n) {
        return (x >> n) | (x << (32 - n));
    };
    std::uint32_t h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    std::uint32_t h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
    std::string msg = bytes;
    const std::uint64_t bit_len = static_cast<std::uint64_t>(bytes.size()) * 8ull;
    msg.push_back(static_cast<char>(0x80));
    while ((msg.size() % 64) != 56) msg.push_back(0);
    for (int i = 7; i >= 0; --i)
        msg.push_back(static_cast<char>((bit_len >> (8 * i)) & 0xff));
    for (std::size_t off = 0; off < msg.size(); off += 64) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const unsigned char* p =
                reinterpret_cast<const unsigned char*>(msg.data() + off + 4 * i);
            w[i] = (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) |
                   (std::uint32_t(p[2]) << 8) | std::uint32_t(p[3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t t1 = h + S1 + ch + K[i] + w[i];
            const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e; h5 += f; h6 += g; h7 += h;
    }
    const std::uint32_t hs[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for (int i = 0; i < 8; ++i)
        std::snprintf(out + 8 * i, 9, "%08x", hs[i]);
    out[64] = 0;
}

struct BoundSnapshot {
    bool wrote = false;
    std::string path;
    std::string sha256;
    std::string error;
};

// One lock covers the snapshot file and the log line that names it, so a
// reader never sees a hash for bytes that were not the bytes just written.
std::mutex& milp_dump_mu() {
    static std::mutex mu;
    return mu;
}

BoundSnapshot write_bound_snapshot(const char* base, std::uint64_t invocation,
                                   const char* phase, const std::vector<f64>& lo,
                                   const std::vector<f64>& hi) {
    BoundSnapshot out;
    if (!base || !*base || !phase) return out;
    out.path = std::string(base) + ".run" + milp_run_id() + ".inv" +
               std::to_string(invocation) + "." + phase;
    std::string body;
    body.reserve(lo.size() * 48 + 16);
    char line[128];
    std::snprintf(line, sizeof line, "%d\n", static_cast<int>(lo.size()));
    body += line;
    for (std::size_t j = 0; j < lo.size() && j < hi.size(); ++j) {
        std::snprintf(line, sizeof line, "%.17g %.17g\n", lo[j], hi[j]);
        body += line;
    }
    char hex[65];
    sha256_hex(body, hex);
    out.sha256 = hex;
    std::FILE* f = std::fopen(out.path.c_str(), "w");
    if (!f) {
        out.error = "open_failed";
        out.sha256.clear();
        return out;
    }
    const std::size_t written = std::fwrite(body.data(), 1, body.size(), f);
    const int closed = std::fclose(f);
    if (written != body.size() || closed != 0) {
        out.error = written != body.size() ? "write_failed" : "close_failed";
        out.sha256.clear();
        return out;
    }
    out.wrote = true;
    return out;
}

// Escape JSON string contents consistently, including filesystem paths.
std::string milp_json_escape(const std::string& value) {
    std::string out;
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out.push_back('\\');
        if (c >= 32) out.push_back(static_cast<char>(c));
        else {
            char escaped[7];
            std::snprintf(escaped, sizeof escaped, "\\u%04x", c);
            out += escaped;
        }
    }
    return out;
}

void append_milp_invocation(const model::LpProblem& model,
                            std::uint64_t invocation, std::uint64_t parent,
                            int sub_mip_depth, const char* phase,
                            const BoundSnapshot& snap) {
    const char* base = milp_dump_base_path();
    if (!base) return;
    const std::string path = std::string(base) + ".invocations.jsonl";
    std::FILE* f = std::fopen(path.c_str(), "a");
    if (!f) return;
    std::fprintf(
        f,
        "{\"run\":\"%s\",\"invocation\":%llu,\"parent\":%llu,\"sub_mip_depth\":%d,"
        "\"phase\":\"%s\",\"wrote_bounds\":%s,\"snapshot\":\"%s\","
        "\"snapshot_sha256\":\"%s\",\"snapshot_error\":\"%s\",\"content_fp\":\"%016llx\","
        "\"col_order_fp\":\"%016llx\",\"rows\":%d,\"cols\":%d,\"nnz\":%lld,"
        "\"maximize\":%s,\"name\":\"",
        milp_run_id().c_str(),
        static_cast<unsigned long long>(invocation),
        static_cast<unsigned long long>(parent),
        sub_mip_depth, milp_json_escape(phase ? phase : "").c_str(),
        snap.wrote ? "true" : "false", milp_json_escape(snap.path).c_str(),
        snap.sha256.c_str(), snap.error.c_str(),
        static_cast<unsigned long long>(milp_content_fingerprint(model)),
        static_cast<unsigned long long>(milp_column_order_fingerprint(model)),
        static_cast<int>(model.n_rows()), static_cast<int>(model.n_cols()),
        static_cast<long long>(model.nnz()),
        model.maximize ? "true" : "false");
    std::fputs(milp_json_escape(model.name).c_str(), f);
    // Snapshot row j is column j of col_order_fp. Submodels that drop or
    // reorder columns get a different content and column fingerprint.
    std::fprintf(f, "\",\"column_map\":\"identity\"}\n");
    std::fclose(f);
}

}  // namespace

std::uint64_t next_milp_invocation_id() {
    return milp_invocation_seq.fetch_add(1, std::memory_order_relaxed) + 1;
}

void record_milp_bounds(const model::LpProblem& model, std::uint64_t invocation,
                        std::uint64_t parent, int sub_mip_depth, const char* phase,
                        const char* snapshot_base, const std::vector<f64>& lo,
                        const std::vector<f64>& hi) {
    if (!milp_dump_base_path()) return;
    std::lock_guard<std::mutex> lock(milp_dump_mu());
    BoundSnapshot snap;
    if (snapshot_base && *snapshot_base)
        snap = write_bound_snapshot(snapshot_base, invocation, phase, lo, hi);
    append_milp_invocation(model, invocation, parent, sub_mip_depth, phase, snap);
}

}  // namespace sor::search::detail
