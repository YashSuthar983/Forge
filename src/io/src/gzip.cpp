#include "sor/io/gzip.hpp"

#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

namespace sor::io {
namespace {

[[noreturn]] void gzip_fail(const char* what, std::size_t at) {
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
    if (!has_gzip_magic(data, size)) gzip_fail("not a gzip stream", 0);
    if (size < kMinMember) gzip_fail("truncated gzip stream", size);

    std::string result;
    std::size_t at = 0;
    while (at < size) {
        if (!has_gzip_magic(data + at, size - at)) {
            if (at == 0) gzip_fail("not a gzip stream", 0);
            gzip_fail("trailing garbage after gzip member", at);
        }

        z_stream strm{};
        // windowBits = 15 + 16: decode a gzip wrapper (zlib manual).
        const int rc_init = inflateInit2(&strm, 15 + 16);
        if (rc_init != Z_OK)
            gzip_fail("inflateInit2 failed", at);

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
                gzip_fail(msg.c_str(), consumed);
            }
            const std::size_t got = chunk.size() - strm.avail_out;
            member.append(chunk.data(), got);
        }
        const std::size_t consumed = strm.total_in;
        inflateEnd(&strm);
        if (consumed == 0) gzip_fail("truncated gzip stream", at);

        if (result.empty()) result = std::move(member);
        else                result += member;
        at += consumed;
    }
    return result;
}

std::string read_maybe_gzip_file(const std::string& path) {
    if (file_has_gzip_magic(path)) {
        GzipFileStream in(path);
        return std::string(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

class GzipFileStream::Buffer : public std::streambuf {
public:
    explicit Buffer(const std::string& path)
        : file_(path, std::ios::binary), in_(kChunk), out_(kChunk) {
        if (!file_) throw std::runtime_error("cannot open file: " + path);
        start_member();
        setg(out_.data(), out_.data(), out_.data());
    }
    ~Buffer() override { if (open_) inflateEnd(&strm_); }

protected:
    int_type underflow() override {
        if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
        while (!done_) {
            if (strm_.avail_in == 0) refill();
            strm_.next_out = reinterpret_cast<Bytef*>(out_.data());
            strm_.avail_out = static_cast<uInt>(out_.size());
            const int rc = inflate(&strm_, Z_NO_FLUSH);
            const std::size_t got = out_.size() - strm_.avail_out;
            if (rc == Z_STREAM_END) {
                next_member();
            } else if (rc == Z_BUF_ERROR && strm_.avail_in == 0 && eof_) {
                gzip_fail("truncated gzip stream", offset());
            } else if (rc != Z_OK && rc != Z_BUF_ERROR) {
                gzip_fail(strm_.msg ? strm_.msg : "inflate failed", offset());
            }
            if (got > 0) {
                setg(out_.data(), out_.data(), out_.data() + got);
                return traits_type::to_int_type(*gptr());
            }
        }
        return traits_type::eof();
    }

private:
    static constexpr std::size_t kChunk = std::size_t{1} << 16;

    // Bytes of the file consumed by inflate so far.
    std::size_t offset() const { return read_ - strm_.avail_in; }

    void refill() {
        if (eof_) return;
        file_.read(in_.data(), static_cast<std::streamsize>(in_.size()));
        const auto n = static_cast<std::size_t>(file_.gcount());
        if (n < in_.size()) eof_ = true;
        read_ += n;
        strm_.next_in = reinterpret_cast<Bytef*>(in_.data());
        strm_.avail_in = static_cast<uInt>(n);
    }

    void start_member() {
        refill();
        if (strm_.avail_in < 2 ||
            !has_gzip_magic(reinterpret_cast<const char*>(strm_.next_in), strm_.avail_in))
            gzip_fail("not a gzip stream", 0);
        // windowBits = 15 + 16: decode a gzip wrapper, checking its CRC-32
        // and length trailer (zlib manual).
        if (inflateInit2(&strm_, 15 + 16) != Z_OK) gzip_fail("inflateInit2 failed", 0);
        open_ = true;
    }

    // After a member's trailer: another member, end of file, or garbage.
    void next_member() {
        if (strm_.avail_in == 0) refill();
        if (strm_.avail_in == 0) { done_ = true; return; }
        if (strm_.avail_in < 2 && !eof_) {
            // A magic split across reads: keep the byte and read more.
            in_[0] = static_cast<char>(*strm_.next_in);
            file_.read(in_.data() + 1, static_cast<std::streamsize>(in_.size() - 1));
            const auto n = static_cast<std::size_t>(file_.gcount());
            if (n < in_.size() - 1) eof_ = true;
            read_ += n;
            strm_.next_in = reinterpret_cast<Bytef*>(in_.data());
            strm_.avail_in = static_cast<uInt>(n + 1);
        }
        if (!has_gzip_magic(reinterpret_cast<const char*>(strm_.next_in), strm_.avail_in))
            gzip_fail("trailing garbage after gzip member", offset());
        if (inflateReset(&strm_) != Z_OK) gzip_fail("inflateReset failed", offset());
    }

    std::ifstream file_;
    std::vector<char> in_, out_;
    z_stream strm_{};
    std::size_t read_ = 0;
    bool open_ = false, eof_ = false, done_ = false;
};

GzipFileStream::GzipFileStream(const std::string& path)
    : std::istream(nullptr), buffer_(std::make_unique<Buffer>(path)) {
    rdbuf(buffer_.get());
    exceptions(std::ios::badbit);
}

GzipFileStream::~GzipFileStream() = default;

void GzipFileStream::finish() {
    while (buffer_->sbumpc() != std::char_traits<char>::eof()) {}
}

}  // namespace sor::io
