// SOR — device abstraction for parallel binary-quadratic tabu search.
//
// LAYER L1.  P independent flip-tabu searches advance in lockstep on the
// device; the host (search/binquad_parallel.cpp) runs the global search on
// top -- elite pool, restarts -- and re-verifies every point it reports.
//
// The model is binquad.cpp's incremental form (see binquad.hpp): minimise
//   L'x + 0.5 x'Mx,  M = off-diagonal H with its sign already made "min",
// keeping h = Mx and r = Ax per search so a candidate's objective change is
// (1-2x_j)(L_j + h_j), O(1), and a flip costs one column of M and of A.
//
// One iteration of every search, four kernels:
//   gain       P x n   d_obj, d_viol and the penalised score of every flip,
//                      tabu-masked with the aspiration rule
//   select     P       argmin score (smallest index on ties); bookkeeping --
//                      tabu tenure, incumbent, penalty, stagnation
//   apply      P       flip the chosen variable; h, r by one column
//   save_best  P x n   copy x to best_x where the search just improved
//
// Differences from host-only binquad, by design: the swap neighbourhood is
// not on the device yet, and a stagnated search is flagged and restarted by
// the host at the next epoch boundary rather than kicked in place.
//
// cpu_binquad_device.cpp is the oracle: same arithmetic, same stateless
// hash and xorshift32 streams.  On instances whose data are integers every
// quantity is exact in double, so the two devices must agree bit for bit
// there -- tests/test_binquad_device.cpp holds them to that.
#pragma once

#include "sor/backend/device_buffer.hpp"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace sor::backend {

using core::f64;

struct BqData {
    std::int32_t n = 0, m = 0;
    std::vector<f64> lin;                         // L, min sense
    std::vector<std::int32_t> m_start, m_idx;     // M by rows (symmetric), min sense
    std::vector<f64> m_val;
    std::vector<std::int32_t> a_cstart, a_crow;   // A by columns
    std::vector<f64> a_cval;
    std::vector<std::int32_t> a_rstart, a_rcol;   // A by rows (rebuild of r)
    std::vector<f64> a_rval;
    std::vector<f64> c_lo, c_hi;                  // row bounds as in the file
};

struct BqParams {
    std::uint32_t searches = 64;                  // P
    std::uint32_t tenure_min = 10, tenure_span = 10;
    std::uint32_t stagnation_limit = 5000;
    f64 penalty0 = 1.0;                           // already times the scale
    f64 penalty_floor = 1e-6;                     // already times the scale
    std::uint32_t seed = 42;
};

// Per-search state, as read back at an epoch boundary.
struct BqSearch {
    f64 obj = 0.0, viol = 0.0, penalty = 0.0;
    f64 best_obj = 0.0, best_viol = 0.0;
    f64 iter = 0.0, since_improve = 0.0;
    bool have_feasible = false, stalled = false;
};

class BinQuadDevice {
public:
    virtual ~BinQuadDevice() = default;
    virtual std::string_view name() const = 0;
    virtual bool is_accelerated() const = 0;

    virtual void upload(const BqData&, const BqParams&) = 0;
    // Pool of elite points restarts draw from (each length n, 0/1).
    virtual void set_elite(const std::vector<std::vector<std::uint8_t>>& elite) = 0;
    // Restart search s from elite[base[s]] (or all-zeros when base[s] < 0)
    // with each bit flipped independently with probability flip_prob, drawn
    // from a stateless hash of (seed, epoch, s, j).  Searches with
    // restart[s] == false are untouched.  Resets tabu and stagnation; keeps
    // each search's own best.  The first call (all searches, all-zeros base,
    // flip_prob 0.5) is the random start.
    virtual void restart(const std::vector<std::uint8_t>& restart,
                         const std::vector<std::int32_t>& base, f64 flip_prob,
                         std::uint32_t epoch) = 0;
    virtual void run(std::uint32_t iterations) = 0;
    virtual void read_searches(std::vector<BqSearch>& out) = 0;
    virtual void read_best(std::uint32_t s, std::vector<std::uint8_t>& x) = 0;

    virtual TransferStats transfer_stats() const = 0;
};

std::unique_ptr<BinQuadDevice> make_cpu_binquad_device();
std::unique_ptr<BinQuadDevice> make_vulkan_binquad_device(int device = -1);
std::unique_ptr<BinQuadDevice> make_binquad_device(std::string_view name, int device = -1);

// The stateless hash and the per-search generator both devices use.  Public
// so the two implementations cannot drift apart.
inline std::uint32_t bq_hash(std::uint32_t a, std::uint32_t b, std::uint32_t c,
                             std::uint32_t d) {
    std::uint32_t h = a * 0x9E3779B1u ^ b * 0x85EBCA77u ^ c * 0xC2B2AE3Du ^ d * 0x27D4EB2Fu;
    h ^= h >> 15; h *= 0x2C1B3C6Du;
    h ^= h >> 12; h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}
inline std::uint32_t bq_xorshift(std::uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

}  // namespace sor::backend
