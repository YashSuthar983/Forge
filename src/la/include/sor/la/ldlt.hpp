// SOR — sparse symmetric LDL' factorization with a fill-reducing ordering.
//
// LAYER L1.  Depends on sor_core only.  Built for the interior-point QP
// engine, whose Newton systems are QUASI-DEFINITE:
//
//     K = [ -(Q + D1)   A' ]        D1, D2 diagonal and positive
//         [    A        D2 ]
//
// Vanderbei (1995) showed such a matrix has an LDL' factorization for EVERY
// symmetric permutation, with the sign of each pivot known in advance
// (negative on the first block, positive on the second).  That is what lets
// this factorization order purely for sparsity, with no pivoting for
// stability: the caller passes the expected signs, and a pivot that comes
// out with the wrong sign or too small is replaced by sign * reg and
// counted, so the caller can see how much regularization it got and repair
// the solve by iterative refinement.
//
// ORDERING: approximate minimum degree on the quotient graph (Amestoy, Davis
// & Duff, SIAM J. Matrix Anal. Appl. 17(4), 1996) -- elements and variables,
// the approximate external degree bound of the paper, element absorption,
// hash-based supervariable detection, and dense rows ordered last.
// Implemented from the paper's description; no ordering library consulted.
//
// FACTORIZATION: elimination tree (Liu 1986), postordered so that every
// subtree is a contiguous range of columns (same fill, Liu 1990); column
// counts by row-subtree traversal; then a SUPERNODAL left-looking numeric
// factorization.  Consecutive columns with nested structure -- fundamental
// supernodes (Liu, Ng & Peyton, SIAM J. Matrix Anal. Appl. 14(1), 1993) --
// are stored as one dense column-major panel, and each panel is updated by
// the panels below it through the linked-list scheme of Ng & Peyton (SIAM
// J. Sci. Comput. 14(5), 1993).  The work is then dense rank-k updates and a
// dense in-panel LDL', instead of one scalar row at a time: that is what
// made the up-looking row-by-row version (Davis, Algorithm 849, which this
// replaces) spend 9 s per factorization on QPLIB_8500's 500k KKT system.
// Implemented from those papers' descriptions; no factorization library
// consulted.
//
// RELAXED SUPERNODES (Ashcraft & Grimes, ACM TOMS 15(4), 1989; the node
// amalgamation of Duff & Reid, ACM TOMS 9(3), 1983): a fundamental partition
// of a KKT system is mostly width-1 supernodes, whose updates are scalar
// axpys with an index lookup per entry.  A child supernode is merged into its
// parent when the merged dense panel holds few explicit zeros (bounded per
// merged width, see LdltOptions); the columns are then re-postordered so
// every merged supernode is contiguous.  Any topological order of the
// elimination tree has the same filled graph (Liu 1990), so the only cost
// is the stored zeros, which the options bound.
//
// THREADS (Geist & Ng, Int. J. Parallel Programming 18(4), 1989, subtree-to-
// processor mapping; Pothen & Sun 1993): the supernodal elimination tree is
// cut into independent subtrees, factored concurrently one per task, and the
// supernodes above the cut are factored one at a time by all threads
// together (their descendant updates split by target column, their dense
// panel LDL' blocked and split by rows).  DETERMINISM: every entry of L is
// produced by the same sequence of floating-point operations whatever the
// thread count or schedule -- the updates into a supernode are applied in
// ascending order of the updating supernode (the linked lists are sorted),
// and the splitting is only ever of independent entries -- so the factor is
// bit-identical at 1 and N threads.  The triangular solves cut the tree at a
// partition fixed at analyze() (independent of the thread count), so they
// are bit-identical too.
#pragma once

#include "sor/core/result.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sor::la {

using core::f64;
using core::Index;
using Offset = std::int64_t;

// A symmetric matrix by its UPPER triangle, compressed by column: for column
// j, rows row_idx[col_ptr[j] .. col_ptr[j+1]) all satisfy row <= j.
// Duplicates are summed.
struct SymCsc {
    Index n = 0;
    std::vector<Offset> col_ptr;
    std::vector<Index> row_idx;
    std::vector<f64> vals;
};

// Fill-reducing permutation of the pattern: perm[new] = old.
std::vector<Index> amd_order(const SymCsc& pattern);

// Binary dump of a symmetric matrix and its pivot signs (developer tool: the
// IPM writes its KKT system with SOR_DUMP_KKT=<path>, apps/sor_ldlt_bench
// reads it).  Throws std::runtime_error on I/O failure.
void write_sym_csc(const char* path, const SymCsc& a, const std::vector<std::int8_t>& sign);
void read_sym_csc(const char* path, SymCsc& a, std::vector<std::int8_t>& sign);

struct LdltOptions {
    // Relaxed amalgamation (see the header comment).  A child is merged into
    // its parent when the merged supernode has at most relax_small columns,
    // or when its explicit zeros are at most relax_zeros of its stored
    // entries and it has at most relax_max_cols columns.
    bool amalgamate = true;
    Index relax_small = 16;
    f64 relax_zeros = 0.10;
    Index relax_max_cols = 256;
};

class Ldlt {
public:
    // Before analyze().
    void set_options(const LdltOptions& o) { opt_ = o; }
    const LdltOptions& options() const noexcept { return opt_; }

    // Ordering + elimination tree + column counts, once per sparsity pattern.
    // An empty perm means "compute AMD".
    void analyze(const SymCsc& a, std::vector<Index> perm = {});

    // Numeric factorization of a matrix with the analyzed pattern.  sign[i]
    // (ORIGINAL index) is the expected pivot sign, +1 or -1.  A pivot d with
    // sign*d < reg is replaced by sign*reg.  Returns false only on a
    // non-finite pivot (bad input), never on regularization.
    bool factorize(const SymCsc& a, const std::vector<std::int8_t>& sign, f64 reg);

    // Solves (L D L') x = b in place, b in ORIGINAL ordering.  Uses the
    // object's scratch space: one solve at a time per Ldlt object.
    void solve(std::vector<f64>& b) const;

    // Threads used by factorize()/solve(): the process-wide pool
    // (sor::core::global_pool()), at most this many of its workers; 0 = all.
    void set_max_threads(int t) { max_threads_ = t; }

    Index n() const noexcept { return n_; }
    Offset nnz_l() const noexcept { return nnz_l_; }
    // Strict-lower entries actually stored (nnz_l plus amalgamation zeros).
    Offset stored_l() const noexcept { return stored_l_; }
    // Multiply-adds the supernodal kernels actually perform, zeros included.
    f64 stored_flops() const noexcept { return stored_flops_; }
    Index supernodes() const noexcept { return ns_; }
    // Multiply-adds of one numeric factorization: sum over columns of count^2.
    f64 flops() const noexcept { return flops_; }
    Index regularized_pivots() const noexcept { return n_reg_; }
    const std::vector<Index>& perm() const noexcept { return perm_; }
    // Hash of the bits of L and D: equal fingerprints at 1 and N threads is
    // how the determinism contract is tested.
    std::uint64_t fingerprint() const;
    // Bytes held by the factorization's own arrays (symbolic + numeric).
    std::size_t memory_bytes() const;

private:
    // Per-worker scratch: row map of the supernode being worked on, the
    // dense update buffer, and the sorted list of updating supernodes.
    struct Scratch {
        std::vector<Index> rel;
        std::vector<f64> w;
        std::vector<Index> ds;
        std::vector<Offset> dp1, dp2;
        Index nreg = 0;
    };
    void ensure_scratch(int workers) const;
    void amalgamate(std::vector<Offset>& count);
    void build_structure(const std::vector<Offset>& count);
    void build_schedule(int workers);
    struct FCtx;                      // per-factorize() context (ldlt.cpp)
    void factor_node(Index s, int w, int nw, Scratch& sc, FCtx& cx);
    void split_tree(const std::vector<f64>& cost, Index parts, f64 ratio,
                    std::vector<Index>& roots, std::vector<Index>& top) const;
    void permute_upper(const SymCsc& a);   // a -> c_ (upper triangle of P A P')
    void etree();                          // parent_ from c_
    void symbolic();                       // supernodes and their row structure

    LdltOptions opt_;
    int max_threads_ = 0;
    Index n_ = 0;
    std::vector<Index> perm_, pinv_, parent_;
    Offset stored_l_ = 0;
    f64 stored_flops_ = 0.0;
    Offset nnz_l_ = 0;                // strict lower nonzeros of L (fill measure)
    f64 flops_ = 0.0;
    std::vector<f64> d_;
    Index n_reg_ = 0;

    // Supernode s covers columns [sfirst_[s], sfirst_[s+1]).  Its row
    // structure (own columns first, ascending) is sri_[srp_[s] .. srp_[s+1]),
    // and its panel is column-major at sx_[sxp_[s]] with leading dimension
    // equal to the number of rows.  The panel's diagonal entries hold 1.
    Index ns_ = 0;
    std::vector<Index> sfirst_, sup_of_;
    std::vector<Offset> srp_, sxp_;
    std::vector<Index> sri_;
    std::vector<f64> sx_;

    // Lower triangle of P A P' by column: for column j, rows lr_[lcp_[j] ..)
    // with values cx_[lsrc_[..]].
    std::vector<Offset> lcp_;
    std::vector<Index> lr_;
    std::vector<Offset> lsrc_;

    // permuted upper triangle, and where each input entry lands in it
    std::vector<Offset> cp_;
    std::vector<Index> ci_;
    std::vector<f64> cx_;
    std::vector<Offset> map_;         // input entry t -> slot in cx_

    // Supernodal elimination tree: sparent_[s] (-1 at a root), sub_lo_[s]
    // the first supernode of s's subtree (a postorder makes it the range
    // [sub_lo_[s], s]), and the dense work each subtree generates.
    std::vector<Index> sparent_, sub_lo_, schead_, snext_;
    std::vector<f64> scost_;          // own work of supernode s
    std::vector<f64> subcost_;        // work of s's whole subtree

    // Factorization schedule for sched_workers_ workers: subtree roots
    // (largest first) and the supernodes above them (ascending).
    int sched_workers_ = 0;
    std::vector<Index> froots_, ftop_;

    // Solve partition, FIXED at analyze() so the solve is bit-identical at
    // any thread count: subtree roots, the offset of each one's buffer for
    // updates leaving the subtree, and the supernodes above them.
    std::vector<Index> vroots_, vtop_;
    std::vector<Offset> vbufp_;
    mutable std::vector<f64> vbuf_, xs_;

    // Linked lists of the left-looking factorization (Ng & Peyton).
    std::vector<Index> head_, next_;
    std::vector<Offset> ptr_;
    mutable std::vector<Scratch> scratch_;
};

}  // namespace sor::la
