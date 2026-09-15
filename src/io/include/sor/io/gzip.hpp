// SOR — gzip decompression via system zlib (libz).
//
// LAYER L2. MIPLIB and similar corpora ship `.mps.gz`. Compression is I/O
// plumbing, not a solver dependency: clean_room_policy.md allows system
// libraries; zlib is ledgered in docs/dependency_ledger.md.
#pragma once

#include <cstddef>
#include <cstdint>
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

// CRC-32 (same polynomial as gzip / zlib). Exposed for round-trip tests.
std::uint32_t crc32(const char* data, std::size_t size, std::uint32_t seed = 0);

}  // namespace sor::io
