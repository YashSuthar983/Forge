// SOR — branch-and-bound for binary QP over batched QCR relaxations.
//
// LAYER L5.  The piece that turns Phase 2's root bound into a proof.  Best-
// first search over variable fixings; every node's bound is the QCR
// relaxation (qcr.hpp) under that node's fixings, solved K nodes at a time
// on a BatchedPdhcgDevice, then RE-DERIVED on the host by wolfe_bound() from
// the returned (x, y) with its PSD and floating-point charges.
//
// Why pruning on a first-order solve is sound here: the node bound is not
// "the relaxation's approximate optimum", which could sit above the true one.
// It is a weak-duality value that is valid at ANY (x, y) -- an inaccurate
// solve can only make it weaker, never invalid.  A node is pruned only when
// that certified value reaches the incumbent.  The fixing identity
// x_i^2 = x_i holds at every binary point of every node, and fixing bounds
// never touches Q, so the root's PSD certificate covers every node.
//
// A node inherits max(parent bound, own bound): both are valid for it.
// Leaves (every variable fixed) are evaluated exactly on the raw instance.
// The run is a proof (ProvedGlobalEpsilon) when the open list empties with
// the incumbent within gap_tol of the smallest open bound; anything less is
// reported as a gap.
#pragma once

#include "sor/backend/batched_pdhcg_device.hpp"
#include "sor/io/qplib.hpp"
#include "sor/search/binquad.hpp"
#include "sor/search/qcr.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

// Which engine solves a node's relaxation.  The bound never trusts the
// engine -- it is wolfe_bound() at whatever (x, y) comes back -- so this is a
// speed choice, not a soundness one.
//   Pdhcg  the batched first-order device, K nodes per pass (the GPU path).
//   Ipm    the in-repo interior point (qp_ipm.cpp), one node at a time on the
//          host.  The QCR relaxation is ill-conditioned BY DESIGN (the SDP
//          shift puts lambda_min(Q + 2 diag u) at ~0, which is exactly what
//          makes the bound tight), so PDHCG's inner CG saturates at every
//          outer iteration; the IPM does not care.  Measured on this box
//          (Ryzen 7700X, one core, load ~6) at the root: QPLIB_0633 n=75 IPM
//          3 ms vs PDHCG 5000 it 3.5 s, same bound; QPLIB_3307 n=256 34 ms vs
//          27.5 s; QPLIB_3709 n=600 0.32 s vs 88 s (PDHCG stopped at 2000 it);
//          QPLIB_10057 n=200 PDHCG never met KKT in 5000 it, its bound 500x
//          weaker than the IPM's.
//   Auto   Ipm when the device is not accelerated (CPU lanes), Pdhcg on a
//          GPU, where K lanes in one pass is what the device is for.
enum class BqpNodeSolver { Auto, Pdhcg, Ipm };

struct BqpBabOptions {
    BqpNodeSolver node_solver = BqpNodeSolver::Auto;
    std::size_t batch = 8;              // nodes bounded per device pass (K)
    double time_limit_s = 60.0;
    std::uint64_t max_nodes = 1000000;
    f64 gap_tol = 1e-6;                 // relative, on the incumbent's scale
    // Round each node's relaxation point and offer it as an incumbent.  Off
    // only in tests: without it incumbents come from leaves alone, which is
    // what makes an overstated bound observable as a wrong proof.
    bool rounding = true;
    // Strong / reliability branching (Applegate, Bixby, Chvatal & Cook 1995;
    // Achterberg, Koch & Martin 2005, Oper. Res. Lett. 33).  0 branches on
    // the variable closest to 1/2.  Otherwise the sb_candidates most
    // fractional free variables are PROBED -- both children's relaxations
    // solved with a truncated budget -- and the branching variable is the one
    // whose two probe gains are jointly largest (the product rule).  A
    // variable whose pseudocost has been observed `reliability` times on both
    // sides is trusted instead of probed, which is what makes this affordable
    // deep in the tree.  Probe bounds are certified exactly like node bounds,
    // so the children inherit them: the probing pays for itself twice.
    std::size_t sb_candidates = 0;
    std::uint64_t sb_iterations = 400;  // PDHCG iterations per probe
    int reliability = 4;
    // The root relaxation is solved once and every node inherits how good it
    // is, so it is worth far more iterations than a node.  0 = the same
    // budget as a node.
    std::uint64_t root_iterations = 0;
    bool verbose = false;               // one stderr line per device pass
    QcrOptions qcr;                     // shift choice + per-node PDHCG limits
};

struct BqpBabResult {
    bool have_incumbent = false;
    std::vector<std::uint8_t> x;
    f64 incumbent = 0.0;                // original sense, raw-data re-score
    f64 bound = 0.0;                    // original sense, certified
    bool bound_valid = false;
    f64 root_bound = 0.0;               // original sense, certified, root node only
    bool root_bound_valid = false;
    // QcrShift::Best only: the shift that LOST at the root, and its root
    // bound.  Published so the choice can be audited and so a sweep can show
    // what each shift was worth.
    std::string shift_alt;
    f64 root_bound_alt = 0.0;
    bool root_bound_alt_valid = false;
    bool proved = false;                // tree closed within gap_tol
    std::uint64_t nodes = 0, batches = 0, pruned = 0, leaves = 0;
    std::uint64_t probes = 0;           // strong-branching child relaxations solved
    std::string node_solver;            // "ipm" | "pdhcg": what bounded the nodes
    double total_ms = 0.0;
    std::string reason;
    QcrDiagnostics qcr;                 // the shift every node shares
};

// `warm` (may be null) seeds the incumbent, e.g. binquad's point; it is
// re-scored and re-checked here, not trusted.
BqpBabResult solve_bqp_bab(const io::QplibInstance& inst, const BinQuadResult* warm,
                           const BqpBabOptions& opts, backend::BatchedPdhcgDevice& device);

}  // namespace sor::search
