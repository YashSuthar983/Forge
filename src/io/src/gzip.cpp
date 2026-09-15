#include "sor/io/gzip.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

namespace sor::io {
namespace {

[[noreturn]] void fail(const char* what, std::size_t at) {
    throw std::runtime_error("gzip: " + std::string(what) + " at byte " +
                             std::to_string(at));
}

}  // namespace

bool has_gzip_magic(const char* data, std::size_t size) {
    return size >= 2 &&
           static_cast<unsigned char>(data[0]) == 0x1fu &&
           static_cast<unsigned char>(data[1]) == 0x8bu;
}

bool file_has_gzip_magic(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char head[2]{};
    in.read(head, 2);
    return in.gcount() == 2 && has_gzip_magic(head, 2);
}

std::uint32_t crc32(const char* data, std::size_t size, std::uint32_t seed) {
    // crc32_z accepts size_t; avoid colliding with the zlib crc32 symbol name
    // when this translation unit defines sor::io::crc32.
    return static_cast<std::uint32_t>(
        ::crc32_z(seed, reinterpret_cast<const Bytef*>(data), size));
}

std::string gunzip(const char* data, std::size_t size) {
    constexpr std::size_t kMinMember = 18;  // header + empty stored + trailer
    if (!has_gzip_magic(data, size)) fail("not a gzip stream", 0);
    if (size < kMinMember) fail("truncated gzip stream", size);

    std::string result;
    std::size_t at = 0;
    while (at < size) {
        if (!has_gzip_magic(data + at, size - at)) {
            if (at == 0) fail("not a gzip stream", 0);
            fail("trailing garbage after gzip member", at);
        }

        z_stream strm{};
        // windowBits = 15 + 16: decode a gzip wrapper (zlib manual).
        const int rc_init = inflateInit2(&strm, 15 + 16);
        if (rc_init != Z_OK)
            fail("inflateInit2 failed", at);

        strm.next_in =
            reinterpret_cast<Bytef*>(const_cast<char*>(data + at));
        strm.avail_in = static_cast<uInt>(size - at);

        std::string member;
        std::vector<char> chunk(1 << 16);
        int rc = Z_OK;
        while (rc != Z_STREAM_END) {
            strm.next_out = reinterpret_cast<Bytef*>(chunk.data());
            strm.avail_out = static_cast<uInt>(chunk.size());
            rc = inflate(&strm, Z_NO_FLUSH);
            if (rc != Z_OK && rc != Z_STREAM_END) {
                const std::string msg =
                    strm.msg ? strm.msg : "inflate failed";
                const std::size_t consumed = at + strm.total_in;
                inflateEnd(&strm);
                fail(msg.c_str(), consumed);
            }
            const std::size_t got = chunk.size() - strm.avail_out;
            member.append(chunk.data(), got);
        }
        const std::size_t consumed = strm.total_in;
        inflateEnd(&strm);
        if (consumed == 0) fail("truncated gzip stream", at);

        if (result.empty()) result = std::move(member);
        else                result += member;
        at += consumed;
    }
    return result;
}

std::string read_maybe_gzip_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    std::ostringstream buf;
    buf << in.rdbuf();
    std::string bytes = buf.str();
    if (!has_gzip_magic(bytes.data(), bytes.size())) return bytes;
    return gunzip(bytes.data(), bytes.size());
}

}  // namespace sor::io
