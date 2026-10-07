// SOR - gzip decompression via system zlib (libz).
//
// LAYER L2. MIPLIB and similar corpora ship `.mps.gz`. Compression is I/O
// plumbing, not a solver dependency; CMake links ZLIB::ZLIB.
#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <string>

namespace sor::io {

// True when `data` begins with the two-byte gzip magic (0x1f 0x8b). Callers
// sniff content rather than trusting the extension: MIPLIB ships `.mps.gz`,
// but a corpus that has been unpacked in place keeps the name.
bool has_gzip_magic(const char* data, std::size_t size);
bool file_has_gzip_magic(const std::string& path);

// Decompress every gzip member in `data`. Throws std::runtime_error naming the
// byte offset on malformed input, an unsupported compression method, or a
// CRC-32/length mismatch -- a silently truncated model is far worse than a
// failed read.
std::string gunzip(const char* data, std::size_t size);

// Read `path` whole, decompressing when it is gzip and returning the bytes
// verbatim when it is not.
std::string read_maybe_gzip_file(const std::string& path);

// An input stream over a gzip file, inflated as it is read: neither the
// compressed nor the decompressed file is ever held whole (64 KiB buffers).
// Every member of a multi-member file is read in turn. Malformed input
// (bad header, CRC-32 or length mismatch, truncation, trailing garbage)
// throws std::runtime_error naming the byte offset, from the read that meets
// it: the stream sets badbit exceptions so getline() rethrows it instead of
// reporting a quiet end of file.
class GzipFileStream : public std::istream {
public:
    explicit GzipFileStream(const std::string& path);
    ~GzipFileStream() override;
    GzipFileStream(const GzipFileStream&) = delete;
    GzipFileStream& operator=(const GzipFileStream&) = delete;

    // Read whatever the caller left (a reader stops at ENDATA) so every
    // member's CRC-32 and length are checked; throws like any other read.
    void finish();

private:
    class Buffer;
    std::unique_ptr<Buffer> buffer_;
};

// CRC-32 (same polynomial as gzip / zlib). Exposed for round-trip tests.
std::uint32_t crc32(const char* data, std::size_t size, std::uint32_t seed = 0);

}  // namespace sor::io
