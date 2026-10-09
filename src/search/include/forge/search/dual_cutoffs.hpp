// FORGE - cached dual / reduced-cost cutoff inequalities (audit rank 11).
//
// A certified dual enclosure yields objective(x) >= L + charged moves of the
// box corners. Improving points with objective < U must therefore obey the
// same reduced-cost inequalities the one-shot RC fixer derives. Keeping the
// enclosure alive lets later nodes tighten without rebuilding A'y, and keeps
// complementary certificates independent (never summed as bounds).
#pragma once

#include "forge/certify/finalize.hpp"
#include "forge/model/lp.hpp"
#include "forge/search/prop_trail.hpp"

#include <cstdint>
#include <vector>

namespace forge::search {

using core::f64;
using core::Index;

struct DualCutoffOptions {
    // Optional witness-cache capacity (0: retain all distinct root witnesses).
    // These are universal improving-region inequalities, so independently
    // valid witnesses compose by intersection, never by summing their bounds.
    int max_certs = 0;
    // Ignore columns whose reduced-cost enclosure straddles zero by more than
    // this (sign-indefinite; the interval fixer already skips them for L).
    f64 sign_tol = 0.0;
};

struct DualCutoffDiagnostics {
    std::uint64_t considered = 0;
    std::uint64_t retained = 0;
    std::uint64_t rejected_indefinite = 0;
    std::uint64_t rejected_stale = 0;
    std::uint64_t propagate_calls = 0;
    std::uint64_t tightenings = 0;
    std::uint64_t empties = 0;
};

// One enclosure of d = sense*c - A'y and its Lagrangian minimum L over the
// ORIGINAL box it was certified against. A store belongs to one solve/model
// context. Valid global row edits preserve older witnesses; propagation still
// requires exact containment in the certificate's original box.
struct DualCutoffCert {
    std::vector<f64> rc_lo;
    std::vector<f64> rc_hi;
    f64 L = 0.0;
    std::uint64_t generation = 0;
    bool root_scope = false;
    // The enclosure's original box anchors. Do not replace them with the
    // post-fixing box while retaining the old Lagrangian minimum.
    std::vector<f64> box_lo;
    std::vector<f64> box_hi;
};

class DualCutoffStore {
public:
    explicit DualCutoffStore(DualCutoffOptions opts = {}) : opts_(opts) {}

    // Retain a finite enclosure over `box_lo`/`box_hi`. Empty / indefinite
    // enclosures are ignored. `generation` is the caller's content generation
    // at derivation time.
    bool consider(const certify::RcEnclosure& enc, const std::vector<f64>& box_lo,
                  const std::vector<f64>& box_hi, std::uint64_t generation,
                  bool root_scope, DualCutoffDiagnostics* diag = nullptr);

    // Apply every still-valid certificate against [lo,hi] and cutoff_min
    // (minimisation sense), only when [lo,hi] nests inside the cert's box.
    // Returns false when a certificate proves the box empty under the cutoff
    // (caller treats as CertifiedCutoff, not original infeasibility).
    bool propagate(std::vector<f64>& lo, std::vector<f64>& hi, f64 cutoff_min,
                   const std::vector<char>& integer_mask, std::uint64_t generation,
                   PropTrail* trail, int depth, DualCutoffDiagnostics* diag = nullptr) const;

    // The strongest independent anchored lower bound for this contained box.
    // It remains useful without an incumbent, and does not combine witnesses.
    f64 lower_bound(const std::vector<f64>& lo, const std::vector<f64>& hi,
                    std::uint64_t generation) const;

    void clear() { certs_.clear(); }
    std::size_t size() const noexcept { return certs_.size(); }
    const DualCutoffOptions& options() const noexcept { return opts_; }

private:
    DualCutoffOptions opts_;
    std::vector<DualCutoffCert> certs_;
};

}  // namespace forge::search
