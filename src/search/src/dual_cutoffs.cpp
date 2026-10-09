#include "sor/search/domain_writer.hpp"
#include "sor/search/dual_cutoffs.hpp"
#include "sor/model/dyadic.hpp"

#include <algorithm>
#include <cmath>

namespace sor::search {
namespace {
bool anchored_minimum(const DualCutoffCert& cert, const std::vector<f64>& lo,
                      const std::vector<f64>& hi, std::uint64_t generation,
                      model::DyadicSum& minimum) {
    const auto n = lo.size();
    if (hi.size() != n || cert.rc_lo.size() != n || cert.box_lo.size() != n ||
        cert.generation > generation) return false;
    for (std::size_t j = 0; j < n; ++j)
        if (lo[j] < cert.box_lo[j] || hi[j] > cert.box_hi[j]) return false;
    minimum.add(cert.L);
    for (std::size_t j = 0; j < n; ++j) {
        const bool lower = cert.rc_lo[j] > 0;
        const f64 magnitude = lower ? cert.rc_lo[j] : cert.rc_hi[j] < 0 ? -cert.rc_hi[j] : 0;
        const f64 corner = lower ? lo[j] : hi[j];
        const f64 origin = lower ? cert.box_lo[j] : cert.box_hi[j];
        if (!(magnitude > 0) || !std::isfinite(magnitude) ||
            !std::isfinite(corner) || !std::isfinite(origin)) continue;
        minimum.add_product(lower ? magnitude : -magnitude, corner);
        minimum.add_product(lower ? -magnitude : magnitude, origin);
    }
    return true;
}
}

f64 DualCutoffStore::lower_bound(const std::vector<f64>& lo, const std::vector<f64>& hi,
                               std::uint64_t generation) const {
    f64 bound = -model::kInf;
    for (const auto& cert : certs_) {
        model::DyadicSum minimum;
        if (anchored_minimum(cert, lo, hi, generation, minimum)) bound = std::max(bound, minimum.down());
    }
    return bound;
}

bool DualCutoffStore::consider(const certify::RcEnclosure &enc,
                               const std::vector<f64> &box_lo,
                               const std::vector<f64> &box_hi,
                               std::uint64_t generation, bool root_scope,
                               DualCutoffDiagnostics *diag) {
  if (diag)
    ++diag->considered;
  if (!enc.finite || enc.lo.empty() || enc.lo.size() != enc.hi.size() ||
      !std::isfinite(enc.bound) || box_lo.size() != enc.lo.size() ||
      box_hi.size() != enc.lo.size() || !root_scope) {
    if (diag)
      ++diag->rejected_indefinite;
    return false;
  }
  bool useful = false;
  for (std::size_t j = 0; j < enc.lo.size(); ++j) {
    if (std::isnan(enc.lo[j]) || std::isnan(enc.hi[j]) ||
        enc.lo[j] > enc.hi[j] || std::isnan(box_lo[j]) ||
        std::isnan(box_hi[j]) || box_lo[j] > box_hi[j]) {
      if (diag)
        ++diag->rejected_indefinite;
      return false;
    }
    useful =
        useful || enc.lo[j] > opts_.sign_tol || enc.hi[j] < -opts_.sign_tol;
  }
  if (!useful) {
    if (diag)
      ++diag->rejected_indefinite;
    return false;
  }
  for (auto &cert : certs_) {
    if (cert.rc_lo == enc.lo && cert.rc_hi == enc.hi && cert.box_lo == box_lo &&
        cert.box_hi == box_hi) {
      // Same anchored inequality: the stronger certified constant
      // dominates exactly. No approximate normal comparison is used.
      if (enc.bound > cert.L) {
        cert.L = enc.bound;
        cert.generation = std::max(cert.generation, generation);
      }
      return true;
    }
  }
  DualCutoffCert cert;
  cert.rc_lo = enc.lo;
  cert.rc_hi = enc.hi;
  cert.L = enc.bound;
  cert.generation = generation;
  cert.root_scope = true;
  cert.box_lo = box_lo;
  cert.box_hi = box_hi;
  certs_.push_back(std::move(cert));
  // Eviction drops an acceleration witness, never a region obligation.
  if (opts_.max_certs > 0 &&
      certs_.size() > static_cast<std::size_t>(opts_.max_certs))
    certs_.erase(certs_.begin());
  if (diag)
    ++diag->retained;
  return true;
}

bool DualCutoffStore::propagate(std::vector<f64> &lo, std::vector<f64> &hi,
                                f64 cutoff_min,
                                const std::vector<char> &integer_mask,
                                std::uint64_t generation, PropTrail *trail,
                                int depth, DualCutoffDiagnostics *diag) const {
  if (diag)
    ++diag->propagate_calls;
  if (!std::isfinite(cutoff_min))
    return true;
  const std::size_t n = lo.size();
  if (hi.size() != n || integer_mask.size() < n)
    return true;
  for (const auto &cert : certs_) {
    if (cert.rc_lo.size() != n || cert.box_lo.size() != n)
      continue;
    // Content generations in one store only add valid global facts.
    // Older inequalities remain necessary after side tightening or cut
    // retirement. A different presolve/model context owns a new store.
    model::DyadicSum minimum;
    if (!anchored_minimum(cert, lo, hi, generation, minimum)) {
      if (diag)
        ++diag->rejected_stale;
      continue;
    }
    // Every feasible improving point obeys this anchored inequality:
    // f(x) >= L + sum positive_rc*(x-original_lo)
    //                 + sum negative_rc*(original_hi-x).
    // Charging the current corners recovers all paid-for tightening;
    // silently moving the anchors while keeping L loses that information.
    if (minimum.compare(cutoff_min) >= 0) {
      if (diag)
        ++diag->empties;
      return false; // no point below the actual improving cutoff
    }
    auto room = minimum;
    room.negate();
    room.add(cutoff_min);
    for (std::size_t j = 0; j < n; ++j) {
      const bool upper = cert.rc_lo[j] > 0.0;
      const f64 mag = upper                 ? cert.rc_lo[j]
                      : cert.rc_hi[j] < 0.0 ? -cert.rc_hi[j]
                                            : 0.0;
      if (!(mag > 0.0) || !std::isfinite(mag) || !(hi[j] > lo[j]))
        continue;
      const f64 corner = upper ? lo[j] : hi[j];
      const f64 origin = upper ? cert.box_lo[j] : cert.box_hi[j];
      if (!std::isfinite(corner) || !std::isfinite(origin))
        continue;
      auto endpoint = room;
      if (!upper)
        endpoint.negate();
      endpoint.add_product(mag, corner);
      f64 value =
          upper ? endpoint.quotient_up(mag) : endpoint.quotient_down(mag);
      if (integer_mask[j])
        value = upper ? std::floor(value) : std::ceil(value);
      f64 &bound = upper ? hi[j] : lo[j];
      if (upper ? !(value < bound) : !(value > bound))
        continue;
      write_domain_bound(lo, hi, static_cast<Index>(j),
                         upper ? BoundDir::Upper : BoundDir::Lower, value,
                         trail, {ReasonKind::Cutoff, -1, depth});
      if (diag)
        ++diag->tightenings;
      if (hi[j] < lo[j]) {
        if (diag)
          ++diag->empties;
        return false;
      }
    }
  }
  return true;
}
} // namespace sor::search
