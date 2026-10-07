#include "sor/search/bab.hpp"
#include "bound_snapshots.hpp"
#include "sor/search/basis_map.hpp"
#include "sor/search/relaxation_identity.hpp"
#include "sor/search/branch_stats.hpp"
#include "sor/search/domain_fixpoint.hpp"
#include "sor/backend/binquad_device.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/engines/lp_batched.hpp"
#include "sor/search/binquad_milp_heuristic.hpp"
#include "sor/search/para_bab.hpp"
#include "sor/search/balans.hpp"
#include "sor/search/features.hpp"
#include "sor/search/fixprop.hpp"
#include "sor/search/kernel_pump.hpp"
#include "sor/search/lifted_branch.hpp"
#include "sor/search/mrens.hpp"
#include "sor/search/planbb.hpp"
#include "sor/search/propagate.hpp"
#include "sor/search/primal_polish.hpp"
#include "sor/search/sc_milp_branch.hpp"
#include "sor/search/sparse_sb.hpp"
#include "sor/search/conflict_cut.hpp"
#include "sor/search/implied_int.hpp"
#include "sor/search/tree_cuts.hpp"
#include "sor/search/cut_policy.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/core/route_debug.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <random>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

// A3: the single re-check every reported MILP point goes through, in the
// CALLER's original space (`p`), regardless of which of the incumbent-setting
// call sites produced it or what transformed/folded copy it was found valid
// against. Mirrors portfolio_point_is_feasible's row/bound checks.
//
// Returns the row/bound violation magnitude alone (for diag.final_primal_
// violation, which finalize_result compares against opts.primal_feas_tol --
// a much tighter tolerance, 1e-7 by default, than int_tol's 1e-6). Integrality
// uses its own tolerance and is NOT folded into that magnitude: a large
// integer variable (this model exercises y up to 2e9) can land a few times
// 1e-7 off an integer from ordinary double rounding while still being well
// inside int_tol, and comparing that gap against primal_feas_tol instead of
// int_tol would reject a numerically-fine heuristic point as if it were a row
// violation. An integrality violation beyond int_tol is instead an outright
// reject (+inf) since there is no partial credit for a fractional integer
// column.

// A2: guards both lazy creation of the shared device below AND every actual
// call into it -- BinQuadDevice is not documented safe for two threads to
// drive concurrently, and this is the only lock either concern has.
std::mutex& binquad_device_mutex() {
    static std::mutex m;
    return m;
}

// Caller must hold binquad_device_mutex(). Process-wide, created once: a
// Vulkan context and kernel compile is not a per-solve cost this heuristic
// should pay again on every eligible node/sub-MIP/retry in the process.
// make_binquad_device("vulkan") is caught, not just null-checked -- the
// try/catch inside it only wraps device construction, not vk::Context::
// create() itself, which is not documented as never throwing.
backend::BinQuadDevice* shared_binquad_device_locked() {
    static std::unique_ptr<backend::BinQuadDevice> device;
    static bool tried = false;
    if (tried) return device.get();
    tried = true;
    // Test-only hook (A2): forces the vulkan attempt to throw, so the catch
    // and cpu fallback are exercised without depending on a real Vulkan
    // failure. Read once, here, never outside a test.
    const bool force_throw = std::getenv("SOR_TEST_BINQUAD_THROW") != nullptr;
    try {
        if (force_throw)
            throw std::runtime_error("SOR_TEST_BINQUAD_THROW: forced");
        device = backend::make_binquad_device("vulkan");
    } catch (const std::exception&) {
        device.reset();
    }
    if (!device) device = backend::make_binquad_device("cpu");
    return device.get();
}

}  // namespace

f64 milp_point_max_violation(const model::LpProblem& p,
                             const std::vector<f64>& x, f64 int_tol) {
    if (static_cast<Index>(x.size()) != p.n_cols())
        return core::kPosInf;
    for (const f64 v : x)
        if (!std::isfinite(v)) return core::kPosInf;
    if (!p.is_integer.empty()) {
        for (Index j = 0; j < p.n_cols(); ++j) {
            if (!p.is_integer[sz(j)]) continue;
            const f64 v = x[sz(j)];
            if (std::fabs(v - std::round(v)) > int_tol) return core::kPosInf;
        }
    }
    return std::max(p.max_row_violation(x), p.max_bound_violation(x));
}

namespace {

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline bool route_sample_node(std::uint64_t nodes) noexcept {
#ifndef SOR_ROUTE_DEBUG
    (void)nodes;
    return false;
#else
    const int every = ::sor::core::route_debug_pivot_every();
    if (every <= 0) return true;
    return nodes % static_cast<std::uint64_t>(every) == 0;
#endif
}

// Objective granularity g: every feasible INTEGER point of `lp` has an
// objective that is an integer multiple of g. Returns 0 when no such g can be
// established.
//
// This is standard dual-bound strengthening and every commercial solver does
// it; SOR did not. If the objective can only take values ..., -g, 0, g, 2g,
// ..., then a dual bound b can be raised to the next multiple of g at or above
// b, because nothing strictly between multiples is attainable. On a
// minimisation with an incumbent already at a multiple of g, that closes any
// gap smaller than g -- which is exactly the situation a branch-and-bound run
// ends in when it has FOUND the optimum but cannot prove it.
//
// Measured need: vpm1 finishes with incumbent 20 and a dual bound of
// 19.8278 (cuts-heavy), i.e. 0.17 short, after 62858 nodes. Its objective is
// integral, so ceil(19.8278) = 20 closes it immediately.
//
// Conditions (all necessary):
//   * every column with a non-zero objective coefficient is an integer column
//     -- a continuous column with a cost makes the objective continuous;
//   * every such coefficient is exactly an integer in the input model.
// A near-integer coefficient is not enough: its error is multiplied by an
// integer variable's value, which can be large or unbounded. Likewise, a
// small nonzero coefficient cannot be dropped merely because it is below the
// integrality tolerance. If either occurs, no objective lattice is proved.
// Then g = gcd of those coefficients. Using the gcd rather than 1 is strictly
// stronger: an objective built from coefficients {4, 6} moves in steps of 2.
f64 objective_granularity(const model::LpProblem& lp) {
    if (lp.is_integer.size() < lp.c.size()) return 0.0;
    f64 g = 0.0;
    for (std::size_t j = 0; j < lp.c.size(); ++j) {
        const f64 cj = lp.c[j];
        if (!std::isfinite(cj)) return 0.0;
        if (cj == 0.0) continue;                     // no cost: irrelevant
        if (!lp.is_integer[j]) return 0.0;           // continuous cost column
        const f64 r = std::round(cj);
        if (cj != r || std::fabs(cj) > 0x1p52)
            return 0.0;                              // no exact integer proof
        const f64 a = std::fabs(r);
        // gcd over reals that are known integers.
        f64 x = g, y = a;
        while (y > 0.5) { const f64 t = std::fmod(x, y); x = y; y = t; }
        g = x;
        // Even once the gcd reaches one, every remaining objective term
        // must still be checked. A later fractional or continuous cost
        // invalidates the objective lattice entirely.
    }
    return g;
}

// Raise a minimisation dual bound to the next attainable objective value.
// `tol` guards against rounding a bound that is already (numerically) on a
// multiple up to the next one, which would be unsound.
// Attainable objective values are offset + g*Z, NOT g*Z: an objective offset
// shifts the whole lattice. Ignoring it would round a bound to a value the
// objective can never take, which is unsound in the dangerous direction.
f64 tighten_bound_to_granularity(f64 bound, f64 g, f64 offset, f64 tol) {
    if (g <= 0.0 || !std::isfinite(bound) || !std::isfinite(offset))
        return bound;
    const f64 k = std::ceil((bound - offset) / g - tol);
    const f64 tightened = offset + k * g;
    return tightened > bound ? tightened : bound;
}

// Share of the total budget any single ROOT SETUP phase may consume.
// Generous per phase but bounded: several phases at 25% still leave budget for
// the search, which is the only thing that proves anything. These phases ran
// unbudgeted and consumed 45.7 s of a 5 s limit on atlanta-ip.
constexpr f64 kRootSetupShare = 0.25;
// Ceiling on what ALL root setup phases may consume TOGETHER.
//
// Bounding each phase to a share of the remaining time stops any one of them
// running away, but it does not stop them collectively eating the budget: a
// sequence of phases each taking 25% of what is left converges on 100%. That
// is exactly what was measured -- after every individual phase was bounded,
// all six large instances honoured the time limit and STILL reported
// nodes = 0, having never branched at all.
//
// Root processing that leaves no time for search is worthless: the search is
// the only part that proves anything. Industrial solvers cap root work at a
// modest fraction for this reason. 20% here leaves 80% for the tree.
// (Now BabOptions::root_reduction_share, default 0.20.)
// Ceiling on the ROOT CUT LOOP, separately from the setup phases above.
//
// Cutting is the most valuable root work there is -- on gt2 it closes 97.7% of
// a 36% integrality gap while branching contributes nothing -- so it gets a
// far larger allowance than presolve. But its only bound used to be the GLOBAL
// time limit, so on a large model it simply took everything: atlanta-ip spent
// 21.6 s of a 30 s budget in the cut loop and branched ZERO nodes.
//
// A solver that never branches cannot prove anything that cuts alone do not
// close. 35% leaves roughly half the budget for the tree after setup. Note
// this cap does not bind on models where cutting is cheap -- gt2's whole loop
// is milliseconds -- so it costs nothing where cuts are winning.
// (Now BabOptions::root_cut_share, default 0.35.)
// Ceiling on ALL pre-search work TOGETHER -- setup phases AND the cut loop --
// measured from the start of the solve, not from the start of each phase.
//
// root_reduction_share caps the setup phases and root_cut_share the cut loop,
// but each was a share of the WHOLE limit measured from its own t0, so they
// composed to 20% + 35% = 55% before the root LP, which is itself unbudgeted.
// Measured on MIPLIB2017 benchmark instances at a 60 s limit, that produced a
// cut loop of ~21 s every time and a pre-search of 35-46 s:
//
//   lectsched-5-obj  setup 3.0 s -> pre-search 37.8 s   cut loop 21.1 s
//   savsched1        setup 3.1 s -> pre-search 41.6 s   cut loop 21.6 s
//   s250r10          setup 3.1 s -> pre-search 46.0 s   cut loop 21.6 s
//
// The search then got 14-25 s against node LPs costing 0.3-19 s each, so it
// branched 1-4 nodes. Across a 60-instance benchmark sample, 48% processed
// AT MOST ONE NODE, and the search is the only part that proves anything.
// (Now BabOptions::root_total_share, default 0.35.)
// A phase budget must be a share of the time REMAINING, never of the original
// limit. Sharing the original lets each phase overshoot independently, so a
// bigger limit buys a bigger overrun: atlanta-ip complied at a 5 s limit and
// took 32.6 s at a 10 s limit, because every phase simply helped itself to
// 25% of 10 s regardless of what earlier phases had already spent.
// Below this a phase cannot do anything useful -- and asking for "0" would be
// read as UNLIMITED by the 0-means-no-limit convention, which is exactly the
// footgun that made an earlier version of this fix a no-op. Skip instead.
constexpr f64 kMinPhaseBudget = 0.01;

inline bool is_integral(f64 v, f64 tol) {
    return std::fabs(v - std::round(v)) <= tol;
}

inline f64 frac_score(f64 v) {
    const f64 f = std::fabs(v - std::floor(v));
    return std::min(f, 1.0 - f);  // distance to nearest integer
}

// Positive means that increasing an integer column is generally useful for
// the relaxation/incumbent heuristics (coverage of lower/equality rows or a
// favorable objective coefficient); negative favors decreasing it.
// Direction bias of every column, in one pass over A: -sense*c_j plus a_ij
// for each >= row, -a_ij for each <= row and |a_ij| for each two-sided row.
// Positive favours the up branch. Accumulated per column in row order, so
// each value equals the former one-column scan exactly; that scan cost a
// full O(nnz) sweep of the matrix per call.
std::vector<f64> integer_up_bias_all(const model::LpProblem& lp) {
    const f64 sense = lp.maximize ? -1.0 : 1.0;
    std::vector<f64> bias(sz(lp.n_cols()));
    for (Index j = 0; j < lp.n_cols(); ++j) bias[sz(j)] = -sense * lp.c[sz(j)];
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const bool has_lo = std::isfinite(lp.row_lo[sz(i)]);
        const bool has_hi = std::isfinite(lp.row_hi[sz(i)]);
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            if (j < 0 || j >= lp.n_cols()) continue;
            const f64 a = lp.A.vals[sz(k)];
            if (has_lo && !has_hi) bias[sz(j)] += a;
            else if (has_hi && !has_lo) bias[sz(j)] -= a;
            else bias[sz(j)] += std::fabs(a);
        }
    }
    return bias;
}

enum class RoundMode : std::uint8_t {
    Objective,
    Nearest,
    Ceil,
};

// The LU factorization and dual steepest-edge weights at a node's final basis.
// Both children of a branched node start from exactly that basis, and the
// child that is not solved right after its parent used to refactorize it and
// rebuild (or reset) the weights from scratch: about 20-30% of node-LP time
// on the tree-heavy models (nu25, binkar, p200). One checkpoint is shared by
// both children.
struct LpCheckpoint {
    engines::FactorCarrier factor;      // has_factor; factor.basis = basis
    std::vector<core::Index> weights_basis;
    std::vector<f64> weights;
};

// Evictable holder: the payload is dropped when the cache runs over budget,
// and the node then simply solves from its basis alone, as before. `bytes`
// is what the cache charged for it (0 once evicted).
struct CheckpointHolder {
    std::unique_ptr<LpCheckpoint> data;
    std::uint64_t generation = 0;   // global_lp_generation it was captured at
    std::size_t bytes = 0;
};

// Memory-bounded cache of node checkpoints. When over budget the checkpoint
// with the HIGHEST node bound is evicted first: best-bound selection pops the
// lowest bound first, so the highest-bound one is needed last. (Evicting the
// oldest instead drops exactly the checkpoints about to be used.) Byte
// accounting is atomic and released by the holder's deleter, so a checkpoint
// freed because its last node was solved or discarded stops counting at once.
class CheckpointCache {
public:
    explicit CheckpointCache(std::size_t budget_bytes)
        : budget_(budget_bytes), live_(std::make_shared<std::atomic<std::size_t>>(0)) {}

    // `bound` is the bound of the branched node (minimisation sense; its
    // children inherit it). Returns null when the checkpoint alone exceeds a
    // quarter of the budget.
    std::shared_ptr<CheckpointHolder> add(std::unique_ptr<LpCheckpoint> data,
                                          std::uint64_t generation, f64 bound) {
        const std::size_t bytes = data->factor.factor.memory_bytes() +
            data->weights.capacity() * sizeof(f64) +
            (data->weights_basis.capacity() + data->factor.basis.capacity()) *
                sizeof(core::Index);
        if (bytes > budget_ / 4) {
            ++declined_;
            return nullptr;
        }
        auto* h = new CheckpointHolder;
        h->data = std::move(data);
        h->generation = generation;
        h->bytes = bytes;
        live_->fetch_add(bytes, std::memory_order_relaxed);
        std::shared_ptr<CheckpointHolder> holder(
            h, [live = live_](CheckpointHolder* p) {
                live->fetch_sub(p->bytes, std::memory_order_relaxed);
                delete p;
            });
        by_bound_.emplace(std::isnan(bound) ? 0.0 : bound, holder);
        ++created_;
        peak_ = std::max(peak_, live_->load(std::memory_order_relaxed));
        while (live_->load(std::memory_order_relaxed) > budget_ &&
               !by_bound_.empty()) {
            auto last = std::prev(by_bound_.end());
            if (auto victim = last->second.lock()) {
                if (victim.get() == holder.get() && by_bound_.size() == 1) break;
                if (victim->data) {
                    live_->fetch_sub(victim->bytes, std::memory_order_relaxed);
                    victim->bytes = 0;
                    victim->data.reset();
                    ++evicted_;
                }
            }
            by_bound_.erase(last);
        }
        // Sweep entries whose nodes are gone -- but only when the map has
        // doubled since the last sweep. Sweeping on every insert once the
        // map held 8192 live entries made each node O(frontier): 8.2 s of a
        // 20 s mas76 run (41%) went into it.
        if (by_bound_.size() > sweep_at_) {
            for (auto it = by_bound_.begin(); it != by_bound_.end();)
                it = it->second.expired() ? by_bound_.erase(it) : std::next(it);
            sweep_at_ = std::max<std::size_t>(8192, 2 * by_bound_.size());
        }
        return holder;
    }
    std::size_t live_bytes() const { return live_->load(std::memory_order_relaxed); }
    std::size_t peak_bytes() const { return peak_; }
    std::uint64_t created() const { return created_; }
    std::uint64_t evicted() const { return evicted_; }
    std::uint64_t declined() const { return declined_; }

private:
    std::size_t budget_;
    std::shared_ptr<std::atomic<std::size_t>> live_;
    std::multimap<f64, std::weak_ptr<CheckpointHolder>> by_bound_;
    std::size_t peak_ = 0;
    std::size_t sweep_at_ = 8192;
    std::uint64_t created_ = 0, evicted_ = 0, declined_ = 0;
};

struct Node {
    std::vector<f64> col_lo;
    std::vector<f64> col_hi;
    engines::SimplexBasis basis;
    // Parent's final factor/weights (see LpCheckpoint); null when none was
    // kept. Shared with the sibling; dropped once this node's LP is solved.
    std::shared_ptr<CheckpointHolder> checkpoint;
    bool has_basis = false;
    f64 bound = -std::numeric_limits<f64>::infinity();  // dual bound (min sense)
    int depth = 0;
    Index parent_branch_var = -1;
    int parent_branch_dir = 0;       // -1 = down, +1 = up
    f64 parent_bound = core::kNaN;
    f64 parent_branch_distance = 0.0;
    // The incoming branch's single ordinary observation has been recorded (a
    // requeued node re-solves but is the same sample). Reset per child.
    bool pc_consumed = false;
    f64 pc_recorded_unit = core::kNaN;   // unit gain this child contributed (for replacement)
    std::uint64_t parent_serial = 0;   // diag.nodes of the parent when this child was made
    // Length of the current unbroken plunge chain this node was created in
    // (0 if it entered via the best-bound queue). Only used to cap how long
    // a single dive can run before falling back to best-bound; see the
    // node-selection block below.
    int plunge_len = 0;
    // Local cuts valid in this node's subtree only (WP-C). Inherited from the
    // parent; new separations append for children and must not reach siblings.
    std::vector<ManagedCut> active_local;
    // global_lp_generation the stored basis was solved under. A node that
    // carries local rows keeps its warm basis only if no global row has been
    // added since: the rows are laid out as global rows, then local rows.
    std::uint64_t basis_generation = 0;
    // LP failures this node has already had (no usable point or certificate),
    // and the global generation at the last one. A node with failures is
    // solved cold by the primal route when it is retried.
    int lp_failures = 0;
    std::uint64_t defer_generation = 0;
    PropTrail prop_trail;
    // Event-driven propagation state (see the node loop). prop_valid: this
    // node's column bounds are a row-propagation fixpoint for the rows of
    // global_lp at generation prop_generation and for the root box at
    // version prop_root_version, EXCEPT for the columns in `dirty`. A child
    // gets its parent's validity plus the one column it branched on.
    bool prop_valid = false;
    std::uint64_t prop_generation = 0;
    std::uint64_t prop_root_version = 0;
    std::vector<Index> dirty;
};

// Best-bound first (minimize): smallest bound first.
struct NodeCmp {
    bool operator()(const Node& a, const Node& b) const {
        if (a.bound != b.bound) return a.bound > b.bound;  // min-heap via greater
        return a.depth < b.depth;
    }
};

// Round integer columns and repair row violations with local integer moves.
// This is a small feasibility-pump style heuristic: it is deliberately
// bounded and never used as a proof, but it is enough to turn the fractional
// schedule relaxation into an incumbent instead of discarding the root node.
bool try_round(const model::LpProblem& lp,
               const std::vector<f64>& x_lp,
               f64 int_tol,
               f64 feas_tol,
               std::vector<f64>& x_out,
               RoundMode mode = RoundMode::Objective) {
    const Index n = lp.n_cols();
    if (static_cast<Index>(x_lp.size()) != n) return false;
    x_out = x_lp;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        // Cost-aware rounding is the first step of objective diving. For a
        // minimization, positive reduced-cost directions prefer the floor and
        // negative-cost directions prefer the ceil (reversed for maximize).
        // Zero-cost structural variables keep nearest rounding so covering and
        // equality repairs retain the LP point's combinatorial signal.
        const f64 work_cost = (lp.maximize ? -1.0 : 1.0) * lp.c[sz(j)];
        f64 v = 0.0;
        if (mode == RoundMode::Ceil) {
            v = std::ceil(x_lp[sz(j)]);
        } else if (mode == RoundMode::Nearest) {
            v = std::round(x_lp[sz(j)]);
        } else {
            v = work_cost > int_tol ? std::floor(x_lp[sz(j)])
                                    : work_cost < -int_tol
                                          ? std::ceil(x_lp[sz(j)])
                                          : std::round(x_lp[sz(j)]);
        }
        v = std::min(std::max(v, lp.col_lo[sz(j)]), lp.col_hi[sz(j)]);
        if (!is_integral(v, int_tol)) {
            v = std::ceil(lp.col_lo[sz(j)] - int_tol);
            if (v > lp.col_hi[sz(j)] + int_tol) return false;
        }
        x_out[sz(j)] = v;
    }

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    const Index m = lp.n_rows();
    std::vector<f64> activity(sz(m), 0.0);
    std::vector<std::vector<std::pair<Index, f64>>> col_rows(sz(n));
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            activity[sz(i)] += av[sz(k)] * x_out[sz(j)];
            col_rows[sz(j)].emplace_back(i, av[sz(k)]);
        }
    const auto row_violation = [&](Index i, f64 act) {
        f64 v = 0.0;
        if (act < lp.row_lo[sz(i)] - feas_tol)
            v += lp.row_lo[sz(i)] - act;
        if (act > lp.row_hi[sz(i)] + feas_tol)
            v += act - lp.row_hi[sz(i)];
        return v;
    };
    // Incremental repair. Only a column in a currently violated row can
    // reduce total violation, and a move changes only its own rows, so the
    // violated-row set and the total are maintained per move instead of
    // rescanning all m rows and every integer column each pass (which cost
    // 0.25-0.5 s per call on piperout/nursesched). Same move rule as before:
    // the best score among moves with positive gain.
    std::vector<char> in_viol(sz(m), 0);
    std::vector<Index> viol_rows;
    std::vector<std::size_t> viol_pos(sz(m), 0);
    const auto set_violated = [&](Index i, bool v) {
        if (v == static_cast<bool>(in_viol[sz(i)])) return;
        if (v) {
            in_viol[sz(i)] = 1;
            viol_pos[sz(i)] = viol_rows.size();
            viol_rows.push_back(i);
        } else {
            in_viol[sz(i)] = 0;
            const std::size_t p = viol_pos[sz(i)];
            const Index last = viol_rows.back();
            viol_rows[p] = last;
            viol_pos[sz(last)] = p;
            viol_rows.pop_back();
        }
    };
    f64 total = 0.0;
    const auto resync = [&]() {
        total = 0.0;
        for (Index i = 0; i < m; ++i) {
            const f64 v = row_violation(i, activity[sz(i)]);
            total += v;
            set_violated(i, v > 0.0);
        }
    };
    resync();
    std::vector<std::uint32_t> seen(sz(n), 0);
    std::uint32_t pass_id = 0;
    const int max_passes = std::min(4096, std::max(64, 2 * static_cast<int>(n)));
    for (int pass = 0; pass < max_passes; ++pass) {
        if ((pass & 255) == 255) resync();  // bound floating drift in `total`
        if (viol_rows.empty() || total <= 0.0) break;
        const f64 before = total;
        Index best_j = -1;
        f64 best_next = 0.0;
        f64 best_gain = 0.0;
        f64 best_score = -std::numeric_limits<f64>::infinity();
        ++pass_id;
        for (const Index i : viol_rows) {
            for (core::Offset kk = rp[sz(i)]; kk < rp[sz(i) + 1]; ++kk) {
                const Index j = ci[sz(kk)];
                if (seen[sz(j)] == pass_id) continue;
                seen[sz(j)] = pass_id;
                if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
                const f64 xj = std::round(x_out[sz(j)]);
                for (int dir : {-1, 1}) {
                    f64 next = xj + static_cast<f64>(dir);
                    next = std::min(std::max(next, lp.col_lo[sz(j)]),
                                    lp.col_hi[sz(j)]);
                    if (std::fabs(next - xj) <= int_tol) continue;
                    const f64 delta = next - xj;
                    f64 change = 0.0;
                    for (const auto& [r, a] : col_rows[sz(j)])
                        change += row_violation(r, activity[sz(r)] + a * delta) -
                                  row_violation(r, activity[sz(r)]);
                    const f64 gain = -change;
                    if (gain <= 1e-12) continue;
                    const f64 obj_delta = (lp.maximize ? -1.0 : 1.0) *
                                          lp.c[sz(j)] * delta;
                    const f64 score = gain / (1.0 + std::max(0.0, obj_delta)) +
                                      1e-9 * std::fabs(delta);
                    if (score > best_score ||
                        (score == best_score && gain > best_gain)) {
                        best_score = score;
                        best_gain = gain;
                        best_j = j;
                        best_next = next;
                    }
                }
            }
        }
        // Never accept a move that makes global violation worse; a plateau is
        // left to the diversified construct/neighborhood heuristics.
        if (best_j < 0) break;
        const f64 delta = best_next - x_out[sz(best_j)];
        x_out[sz(best_j)] = best_next;
        for (const auto& [i, a] : col_rows[sz(best_j)]) {
            activity[sz(i)] += a * delta;
            set_violated(i, row_violation(i, activity[sz(i)]) > 0.0);
        }
        total = before - best_gain;
    }

    if (lp.max_row_violation(x_out) > feas_tol) return false;
    if (lp.max_bound_violation(x_out) > feas_tol) return false;
    return true;
}

// Fix the rounded integer columns and use the LP engine to repair continuous
// columns and equality rows. This is bounded and heuristic-only: its result is
// accepted only after checking the original MILP, and it never supplies a B&B
// bound or pruning certificate.
bool try_lp_rounding_repair(const model::LpProblem& lp,
                            const std::vector<f64>& x_lp,
                            f64 int_tol,
                            f64 feas_tol,
                            std::uint64_t max_iterations,
                            double time_limit_s,
                            std::vector<f64>& x_out) {
    if (static_cast<Index>(x_lp.size()) != lp.n_cols()) return false;
    model::LpProblem fixed = lp;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        f64 v = std::round(x_lp[sz(j)]);
        if (!std::isfinite(v)) return false;
        v = std::min(std::max(v, lp.col_lo[sz(j)]), lp.col_hi[sz(j)]);
        if (!std::isfinite(v) || !is_integral(v, int_tol)) return false;
        fixed.col_lo[sz(j)] = v;
        fixed.col_hi[sz(j)] = v;
    }

    engines::SimplexOptions repair_opts;
    repair_opts.method = engines::SimplexMethod::Auto;
    repair_opts.presolve = true;
    repair_opts.max_iterations = max_iterations;
    repair_opts.time_limit_s = time_limit_s;
    repair_opts.primal_feas_tol = feas_tol;
    repair_opts.dual_feas_tol = std::max(feas_tol, 1e-7);
    engines::SimplexDiagnostics repair_diag;
    const auto repaired = engines::solve_simplex(fixed, repair_opts,
                                                 repair_diag, nullptr);
    if (repaired.proposed_status != core::Status::Optimal &&
        repaired.proposed_status != core::Status::Feasible)
        return false;
    if (static_cast<Index>(repaired.x.size()) != lp.n_cols()) return false;
    if (lp.max_row_violation(repaired.x) > feas_tol ||
        lp.max_bound_violation(repaired.x) > feas_tol)
        return false;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (!lp.is_integer.empty() && lp.is_integer[sz(j)] &&
            !is_integral(repaired.x[sz(j)], int_tol))
            return false;
    }
    x_out = repaired.x;
    return true;
}

// Enumerate the small fractional integer core of an LP relaxation. This is a
// bounded exact neighborhood search: variables already integral in the LP are
// fixed to their nearest integer, while each remaining integer variable gets
// its floor/ceil alternatives. RENS/RINS-style neighborhoods of this form are
// particularly effective on degenerate low-row MIPs such as markshare.
bool try_fractional_enumeration(const model::LpProblem& lp,
                                const std::vector<f64>& x_lp,
                                f64 int_tol,
                                f64 feas_tol,
                                std::uint64_t max_combinations,
                                double time_limit_s,
                                std::uint64_t repair_iterations,
                                double repair_time_s,
                                std::vector<f64>& x_out) {
    if (static_cast<Index>(x_lp.size()) != lp.n_cols() || max_combinations == 0)
        return false;
    std::vector<Index> fractional;
    std::vector<f64> base = x_lp;
    std::vector<std::array<f64, 2>> choices;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        const f64 v = x_lp[sz(j)];
        if (!std::isfinite(v)) return false;
        const f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
        const f64 nearest = std::round(v);
        if (is_integral(v, int_tol)) {
            base[sz(j)] = std::min(std::max(nearest, lo), hi);
            continue;
        }
        const f64 down = std::min(std::max(std::floor(v), lo), hi);
        const f64 up = std::min(std::max(std::ceil(v), lo), hi);
        if (!is_integral(down, int_tol) || !is_integral(up, int_tol)) return false;
        fractional.push_back(j);
        choices.push_back({down, up});
    }
    if (fractional.empty() || fractional.size() >= 63) return false;
    const std::uint64_t combinations = 1ull << fractional.size();
    const std::uint64_t limit = std::min(combinations, max_combinations);
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    bool found = false;
    f64 best_obj = lp.maximize ? -std::numeric_limits<f64>::infinity()
                               : std::numeric_limits<f64>::infinity();
    std::vector<f64> best;
    for (std::uint64_t mask = 0; mask < limit && !over_budget(); ++mask) {
        std::vector<f64> candidate = base;
        for (std::size_t q = 0; q < fractional.size(); ++q)
            candidate[sz(fractional[q])] = choices[q][(mask >> q) & 1ull];
        std::vector<f64> repaired;
        if (!try_lp_rounding_repair(lp, candidate, int_tol, feas_tol,
                                    repair_iterations, repair_time_s,
                                    repaired)) continue;
        const f64 obj = lp.objective(repaired);
        if (!std::isfinite(obj)) continue;
        if (!found || (lp.maximize ? obj > best_obj : obj < best_obj)) {
            found = true;
            best_obj = obj;
            best = std::move(repaired);
        }
    }
    if (found) x_out = std::move(best);
    return found;
}

// Generate additional LP extreme points before rounding. Degenerate models
// such as markshare and pk1 have a zero-cost face containing many fractional
// optima; relying on one simplex basis repeatedly rounds the same poor binary
// pattern. Tiny deterministic objective perturbations select different
// vertices while leaving the original MILP objective untouched.
bool try_perturbed_rounding(const model::LpProblem& lp,
                            const engines::SimplexOptions& base_opts,
                            f64 int_tol,
                            f64 feas_tol,
                            int attempts,
                            double time_limit_s,
                            std::vector<f64>& x_out) {
    if (attempts <= 0 || lp.n_cols() == 0) return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    const f64 sense = lp.maximize ? -1.0 : 1.0;
    f64 scale = 1.0;
    for (const f64 c : lp.c) scale = std::max(scale, std::fabs(c));
    std::vector<f64> benefit(static_cast<std::size_t>(lp.n_cols()), 0.0);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const bool has_lo = std::isfinite(lp.row_lo[sz(i)]);
        const bool has_hi = std::isfinite(lp.row_hi[sz(i)]);
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = lp.A.vals[sz(k)];
            // Positive contribution toward a finite lower bound is useful;
            // positive contribution toward an upper bound is costly. For an
            // equality/slack row, maximizing absolute coverage is a useful
            // tie-break because the continuous slack then shrinks.
            if (has_lo && !has_hi) benefit[sz(j)] += a;
            else if (has_hi && !has_lo) benefit[sz(j)] -= a;
            else benefit[sz(j)] += std::fabs(a);
        }
    }

    bool found = false;
    f64 best_obj = lp.maximize ? -std::numeric_limits<f64>::infinity()
                               : std::numeric_limits<f64>::infinity();
    std::uint64_t state = 0xd1b54a32d192ed03ull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };
    for (int pass = 0; pass < attempts && !over_budget(); ++pass) {
        model::LpProblem probe = lp;
        const f64 eps = 1.0e-4 * scale;
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            const f64 noise = static_cast<f64>(
                static_cast<long long>(next_rand() % 2001) - 1000) / 1000.0;
            // Minimize in the working sense. The benefit term makes the LP
            // prefer useful coverage; noise breaks ties between equivalent
            // vertices and is intentionally much smaller than that signal.
            probe.c[sz(j)] -= sense * eps * (benefit[sz(j)] + 0.01 * noise);
        }
        engines::SimplexOptions solve_opts = base_opts;
        solve_opts.method = engines::SimplexMethod::Primal;
        solve_opts.presolve = true;
        solve_opts.max_iterations = std::max<std::uint64_t>(
            20000, base_opts.max_iterations == 0 ? 0 : base_opts.max_iterations);
        solve_opts.time_limit_s = std::min(
            0.08, time_limit_s > 0.0
                ? std::max(0.01, time_limit_s -
                    std::chrono::duration<double>(Clock::now() - start).count())
                : 0.08);
        engines::SimplexDiagnostics sd;
        const auto r = engines::solve_simplex(probe, solve_opts, sd, nullptr);
        if (r.proposed_status != core::Status::Optimal &&
            r.proposed_status != core::Status::Feasible)
            continue;
        for (RoundMode mode : {RoundMode::Objective, RoundMode::Nearest,
                               RoundMode::Ceil}) {
            std::vector<f64> candidate;
            if (!try_round(lp, r.x, int_tol, feas_tol, candidate, mode)) continue;
            const f64 obj = lp.objective(candidate);
            if (!std::isfinite(obj)) continue;
            if (!found || (lp.maximize ? obj > best_obj : obj < best_obj)) {
                found = true;
                best_obj = obj;
                x_out = std::move(candidate);
            }
        }
    }
    return found;
}

// Diversified constructive search for degenerate packing/equality MILPs. It
// builds integer assignments from row deficits without solving an LP for every
// failed trial, then sends only the most promising assignments through the
// exact continuous LP repair. This is a bounded incumbent heuristic.
bool try_randomized_construct(const model::LpProblem& lp,
                              f64 int_tol,
                              f64 feas_tol,
                              int restarts,
                              int lp_trials,
                              double time_limit_s,
                              std::uint64_t repair_iterations,
                              double repair_time_s,
                              std::vector<f64>& x_out) {
    const Index n = lp.n_cols();
    if (restarts <= 0 || lp_trials <= 0 || n == 0) return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    std::vector<Index> ints;
    for (Index j = 0; j < n; ++j)
        if (!lp.is_integer.empty() && lp.is_integer[sz(j)]) ints.push_back(j);
    if (ints.empty()) return false;

    struct Candidate {
        f64 score = std::numeric_limits<f64>::infinity();
        std::vector<f64> x;
    };
    std::vector<Candidate> pool;
    pool.reserve(static_cast<std::size_t>(lp_trials));
    std::uint64_t state = 0x243f6a8885a308d3ull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };
    const auto row_penalty = [&](const std::vector<f64>& act) {
        f64 p = 0.0;
        for (Index i = 0; i < lp.n_rows(); ++i) {
            const f64 lo = lp.row_lo[sz(i)], hi = lp.row_hi[sz(i)];
            if (act[sz(i)] < lo) p += lo - act[sz(i)];
            if (act[sz(i)] > hi) p += act[sz(i)] - hi;
        }
        return p;
    };
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<std::vector<std::pair<Index, f64>>> col_rows(sz(n));
    for (Index i = 0; i < lp.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            col_rows[sz(ci[sz(k)])].emplace_back(i, av[sz(k)]);

    for (int restart = 0; restart < restarts && !over_budget(); ++restart) {
        std::vector<f64> trial(static_cast<std::size_t>(n), 0.0);
        for (const Index j : ints) {
            f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
            if (!std::isfinite(lo)) lo = 0.0;
            f64 v = std::ceil(lo - int_tol);
            if (std::isfinite(hi) && hi > v &&
                (next_rand() & 3ull) == 0ull)
                v = std::floor(hi + int_tol);
            trial[sz(j)] = std::min(std::max(v, lp.col_lo[sz(j)]), hi);
        }
        std::vector<f64> act(static_cast<std::size_t>(lp.n_rows()), 0.0);
        for (Index i = 0; i < lp.n_rows(); ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                act[sz(i)] += av[sz(k)] * trial[sz(ci[sz(k)])];

        // Randomized greedy coordinate descent on row violation. Several
        // passes let a general integer move more than one unit; random noise
        // changes the order among equal-cost covering choices.
        const int passes = std::min<int>(32, std::max<int>(4, ints.size()));
        for (int pass = 0; pass < passes && !over_budget(); ++pass) {
            Index best_j = -1;
            f64 best_next = 0.0;
            f64 best_gain = 0.0;
            for (const Index j : ints) {
                const f64 v = std::round(trial[sz(j)]);
                for (int dir : {-1, 1}) {
                    f64 next = v + static_cast<f64>(dir);
                    next = std::min(std::max(next, lp.col_lo[sz(j)]),
                                    lp.col_hi[sz(j)]);
                    if (std::fabs(next - v) <= int_tol) continue;
                    const f64 delta = next - v;
                    f64 gain = 0.0;
                    for (const auto& [i, a] : col_rows[sz(j)]) {
                        const f64 old = act[sz(i)];
                        const f64 neu = old + a * delta;
                        const f64 lo = lp.row_lo[sz(i)], hi = lp.row_hi[sz(i)];
                        const f64 po = (old < lo ? lo - old : old > hi ? old - hi : 0.0);
                        const f64 pn = (neu < lo ? lo - neu : neu > hi ? neu - hi : 0.0);
                        gain += po - pn;
                    }
                    gain += 1.0e-6 *
                        static_cast<f64>(next_rand() % 1000) / 1000.0;
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_j = j;
                        best_next = next;
                    }
                }
            }
            if (best_j < 0 || best_gain <= 1.0e-7) break;
            const f64 delta = best_next - trial[sz(best_j)];
            trial[sz(best_j)] = best_next;
            for (const auto& [i, a] : col_rows[sz(best_j)])
                act[sz(i)] += a * delta;
        }
        Candidate c;
        c.score = row_penalty(act);
        c.x = std::move(trial);
        pool.push_back(std::move(c));
    }

    std::sort(pool.begin(), pool.end(), [](const Candidate& a, const Candidate& b) {
        return a.score < b.score;
    });
    if (static_cast<int>(pool.size()) > lp_trials)
        pool.resize(static_cast<std::size_t>(lp_trials));
    bool found = false;
    f64 best_obj = lp.maximize ? -std::numeric_limits<f64>::infinity()
                               : std::numeric_limits<f64>::infinity();
    int tested = 0;
    for (const Candidate& c : pool) {
        if (tested++ >= lp_trials || over_budget()) break;
        std::vector<f64> repaired;
        if (!try_lp_rounding_repair(lp, c.x, int_tol, feas_tol,
                                    repair_iterations, repair_time_s,
                                    repaired))
            continue;
        const f64 obj = lp.objective(repaired);
        if (!std::isfinite(obj)) continue;
        if (!found || (lp.maximize ? obj > best_obj : obj < best_obj)) {
            found = true;
            best_obj = obj;
            x_out = std::move(repaired);
        }
    }
    return found;
}

// Cost-aware constructor for generalized covering models.  A common MIPLIB
// pattern (gt2/transportation variants) has positive lower-bound demand rows,
// positive upper-bound availability rows, and each integer column linking one
// row of each type.  Greedy LP rounding is particularly weak there because it
// spreads fractional demand over expensive columns; this constructor builds
// integral units directly and then deletes redundant units.
bool try_covering_construct(const model::LpProblem& lp,
                            f64 int_tol,
                            f64 feas_tol,
                            int restarts,
                            double time_limit_s,
                            std::vector<f64>& x_out) {
    if (lp.maximize || restarts <= 0 || lp.n_cols() == 0) return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    const Index m = lp.n_rows(), n = lp.n_cols();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<Index> demand_rows, capacity_rows;
    for (Index i = 0; i < m; ++i) {
        if (std::isfinite(lp.row_lo[sz(i)]) &&
            !std::isfinite(lp.row_hi[sz(i)]) &&
            lp.row_lo[sz(i)] >= 0.0)
            demand_rows.push_back(i);
        if (!std::isfinite(lp.row_lo[sz(i)]) &&
            std::isfinite(lp.row_hi[sz(i)]) &&
            lp.row_hi[sz(i)] >= 0.0)
            capacity_rows.push_back(i);
    }
    if (demand_rows.empty() || capacity_rows.empty()) return false;
    std::vector<int> demand_pos(static_cast<std::size_t>(m), -1);
    std::vector<int> capacity_pos(static_cast<std::size_t>(m), -1);
    for (std::size_t q = 0; q < demand_rows.size(); ++q)
        demand_pos[sz(demand_rows[q])] = static_cast<int>(q);
    for (std::size_t q = 0; q < capacity_rows.size(); ++q)
        capacity_pos[sz(capacity_rows[q])] = static_cast<int>(q);
    struct Link { int d = -1, c = -1; f64 ad = 0.0, ac = 0.0; };
    std::vector<Link> links(static_cast<std::size_t>(n));
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) return false;
        Link link;
        for (Index i = 0; i < m; ++i) {
            f64 a = 0.0;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                if (ci[sz(k)] == j) a += av[sz(k)];
            if (std::fabs(a) <= 1e-12) continue;
            if (demand_pos[sz(i)] >= 0 && a > 0.0) {
                if (link.d >= 0) return false;
                link.d = demand_pos[sz(i)]; link.ad = a;
            } else if (capacity_pos[sz(i)] >= 0 && a > 0.0) {
                if (link.c >= 0) return false;
                link.c = capacity_pos[sz(i)]; link.ac = a;
            } else {
                return false;
            }
        }
        if (link.d < 0 || link.c < 0 || link.ad <= 0.0 || link.ac <= 0.0)
            return false;
        links[sz(j)] = link;
    }

    std::uint64_t state = 0x94d049bb133111ebull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };
    bool found = false;
    f64 best_obj = std::numeric_limits<f64>::infinity();
    std::vector<f64> best;
    for (int restart = 0; restart < restarts && !over_budget(); ++restart) {
        std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
        std::vector<f64> demand(demand_rows.size(), 0.0);
        std::vector<f64> capacity(capacity_rows.size(), 0.0);
        bool failed = false;
        for (std::size_t step = 0; step < static_cast<std::size_t>(n) * 8;
             ++step) {
            int chosen = -1;
            f64 chosen_score = std::numeric_limits<f64>::infinity();
            f64 chosen_noise = 0.0;
            for (Index j = 0; j < n; ++j) {
                const Link& link = links[sz(j)];
                if (x[sz(j)] >= lp.col_hi[sz(j)] - int_tol) continue;
                const f64 need = lp.row_lo[sz(demand_rows[link.d])] -
                                 demand[sz(link.d)];
                if (need <= feas_tol) continue;
                if (capacity[sz(link.c)] + link.ac >
                    lp.row_hi[sz(capacity_rows[link.c])] + feas_tol)
                    continue;
                const f64 gain = std::min(link.ad, need);
                if (gain <= 0.0) continue;
                const f64 cap_left = lp.row_hi[sz(capacity_rows[link.c])] -
                                     capacity[sz(link.c)];
                const f64 scarcity = 1.0 + 0.05 *
                    (cap_left > 0.0 ? link.ac / cap_left : 1.0e6);
                const f64 noise = static_cast<f64>(next_rand() % 10000u) /
                                  1.0e7;
                const f64 score = (lp.c[sz(j)] / gain) * scarcity + noise;
                if (score < chosen_score) {
                    chosen_score = score; chosen = static_cast<int>(j);
                    chosen_noise = noise;
                }
            }
            (void)chosen_noise;
            if (chosen < 0) break;
            const Link& link = links[sz(chosen)];
            const f64 need = lp.row_lo[sz(demand_rows[link.d])] -
                             demand[sz(link.d)];
            f64 units = std::ceil((need - feas_tol) / link.ad);
            units = std::max(1.0, units);
            units = std::min(units, lp.col_hi[sz(chosen)] - x[sz(chosen)]);
            const f64 cap_left = lp.row_hi[sz(capacity_rows[link.c])] -
                                 capacity[sz(link.c)];
            units = std::min(units, std::floor((cap_left + feas_tol) /
                                                link.ac + int_tol));
            if (units < 1.0 - int_tol) { failed = true; break; }
            units = std::floor(units + int_tol);
            x[sz(chosen)] += units;
            demand[sz(link.d)] += units * link.ad;
            capacity[sz(link.c)] += units * link.ac;
        }
        if (failed) continue;
        if (lp.max_row_violation(x) > feas_tol ||
            lp.max_bound_violation(x) > feas_tol)
            continue;

        // Delete units in descending cost order while preserving all demands.
        std::vector<Index> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](Index a, Index b) {
            return lp.c[sz(a)] > lp.c[sz(b)];
        });
        for (const Index j : order) {
            while (x[sz(j)] >= 1.0 - int_tol) {
                x[sz(j)] -= 1.0;
                if (lp.max_row_violation(x) > feas_tol) {
                    x[sz(j)] += 1.0;
                    break;
                }
            }
        }
        const f64 obj = lp.objective(x);
        if (std::isfinite(obj) && (!found || obj < best_obj)) {
            found = true; best_obj = obj; best = std::move(x);
        }
    }
    if (found) x_out = std::move(best);
    return found;
}

// Minimax discrepancy search for binary equalities with explicit positive and
// negative residual columns (the pk1 pattern).  Any binary assignment is made
// feasible by those residuals; the objective is the smallest possible maximum
// absolute discrepancy over the equality rows.  The search is deliberately
// independent of the LP relaxation and returns a fully represented model
// point, which the caller still checks against all original rows and bounds.
bool try_discrepancy_binary_search(const model::LpProblem& lp,
                                   f64 int_tol,
                                   f64 feas_tol,
                                   int restarts,
                                   int iterations,
                                   double time_limit_s,
                                   std::vector<f64>& x_out) {
    if (lp.maximize || restarts <= 0 || iterations <= 0) return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    const Index n = lp.n_cols();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<Index> bins;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (lp.col_lo[sz(j)] < -int_tol || lp.col_hi[sz(j)] > 1.0 + int_tol)
            return false;
        bins.push_back(j);
    }
    if (bins.size() < 8 || bins.size() > 120) return false;

    struct EqInfo { Index row = -1; f64 target = 0.0;
                    Index pos = -1, neg = -1; };
    std::vector<EqInfo> eqs;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        if (!std::isfinite(lp.row_lo[sz(i)]) ||
            !std::isfinite(lp.row_hi[sz(i)]) ||
            std::fabs(lp.row_lo[sz(i)] - lp.row_hi[sz(i)]) > 1e-9)
            continue;
        EqInfo eq;
        eq.row = i; eq.target = lp.row_hi[sz(i)];
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = av[sz(k)];
            if (lp.is_integer[sz(j)]) {
                if (a <= 0.0) { eq.pos = -2; break; }
                continue;
            }
            if (std::fabs(a - 1.0) <= 1e-9 && eq.pos < 0)
                eq.pos = j;
            else if (std::fabs(a + 1.0) <= 1e-9 && eq.neg < 0)
                eq.neg = j;
            else { eq.pos = -2; break; }
        }
        if (eq.pos >= 0 && eq.neg >= 0 &&
            std::fabs(eq.target - std::round(eq.target)) <= 1e-9)
            eqs.push_back(eq);
    }
    if (eqs.size() < 3) return false;

    std::vector<std::vector<std::pair<std::size_t, f64>>> bin_rows(
        static_cast<std::size_t>(n));
    for (std::size_t r = 0; r < eqs.size(); ++r) {
        for (core::Offset k = rp[sz(eqs[r].row)];
             k < rp[sz(eqs[r].row) + 1]; ++k) {
            const Index j = ci[sz(k)];
            if (lp.is_integer[sz(j)])
                bin_rows[sz(j)].emplace_back(r, av[sz(k)]);
        }
    }
    for (const Index j : bins)
        if (bin_rows[sz(j)].empty()) return false;

    const auto score = [&](const std::vector<f64>& act, f64& max_abs) {
        max_abs = 0.0; f64 sum_abs = 0.0;
        for (std::size_t r = 0; r < eqs.size(); ++r) {
            const f64 d = std::fabs(eqs[r].target - act[r]);
            max_abs = std::max(max_abs, d); sum_abs += d;
        }
        return max_abs + 1.0e-3 * sum_abs;
    };
    std::uint64_t state = 0x6a09e667f3bcc909ull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };
    f64 best_score = std::numeric_limits<f64>::infinity();
    f64 best_max = std::numeric_limits<f64>::infinity();
    std::vector<unsigned char> best_on(static_cast<std::size_t>(n), 0);

    for (int restart = 0; restart < restarts && !over_budget(); ++restart) {
        std::vector<unsigned char> on(static_cast<std::size_t>(n), 0);
        std::vector<f64> act(eqs.size(), 0.0);
        if ((restart & 1) == 0) {
            // Greedy seed: repeatedly select the column with the largest
            // reduction in current absolute discrepancy.
            for (std::size_t q = 0; q < bins.size(); ++q) {
                Index chosen = -1; f64 gain_best = 0.0;
                f64 before_max = 0.0; const f64 before = score(act, before_max);
                for (const Index j : bins) {
                    if (on[sz(j)]) continue;
                    for (const auto& [r, a] : bin_rows[sz(j)]) act[r] += a;
                    f64 after_max = 0.0; const f64 after = score(act, after_max);
                    for (const auto& [r, a] : bin_rows[sz(j)]) act[r] -= a;
                    const f64 gain = before - after +
                        1.0e-6 * static_cast<f64>(next_rand() & 0xffffu);
                    if (gain > gain_best) { gain_best = gain; chosen = j; }
                }
                if (chosen < 0 || gain_best <= 0.0) break;
                on[sz(chosen)] = 1;
                for (const auto& [r, a] : bin_rows[sz(chosen)]) act[r] += a;
            }
        } else {
            for (const Index j : bins) if (next_rand() & 1ull) {
                on[sz(j)] = 1;
                for (const auto& [r, a] : bin_rows[sz(j)]) act[r] += a;
            }
        }
        f64 current_max = 0.0;
        f64 current = score(act, current_max);
        f64 temp = std::max(2.0, current * 0.35);
        for (int it = 0; it < iterations && !over_budget(); ++it) {
            const Index a = bins[static_cast<std::size_t>(next_rand() % bins.size())];
            const bool a_on = on[sz(a)] != 0;
            const int da = a_on ? -1 : 1;
            std::array<Index, 8> move{a, -1, -1, -1, -1, -1, -1, -1};
            std::array<int, 8> delta{da, 0, 0, 0, 0, 0, 0, 0};
            int nmove = 1;
            const std::uint64_t move_code = next_rand() % 10u;
            if (move_code >= 6) nmove = 2;
            if (move_code == 8) nmove = 3;
            if (move_code == 9) nmove = 5;
            for (int q = 1; q < nmove; ++q) {
                Index j = bins[static_cast<std::size_t>(next_rand() % bins.size())];
                bool duplicate = false;
                for (int z = 0; z < q; ++z) if (move[z] == j) duplicate = true;
                if (duplicate) { --q; continue; }
                move[q] = j;
                delta[q] = on[sz(j)] ? -1 : 1;
            }
            for (int q = 0; q < nmove; ++q)
                for (const auto& [r, v] : bin_rows[sz(move[q])])
                    act[r] += delta[q] * v;
            f64 next_max = 0.0; const f64 next = score(act, next_max);
            const bool accept = next <= current ||
                (temp > 1e-9 && static_cast<f64>(next_rand() % 1000000u) /
                    1000000.0 < std::exp(std::min(0.0, (current - next) / temp)));
            if (accept) {
                for (int q = 0; q < nmove; ++q)
                    on[sz(move[q])] = static_cast<unsigned char>(
                        on[sz(move[q])] ? 0 : 1);
                current = next; current_max = next_max;
            } else {
                for (int q = 0; q < nmove; ++q)
                    for (const auto& [r, v] : bin_rows[sz(move[q])])
                        act[r] -= delta[q] * v;
            }
            temp *= 0.9997;
            if (current < best_score) {
                best_score = current; best_max = current_max; best_on = on;
            }
            if (best_max <= 0.0) break;
        }

        // Deterministic one/two-bit polishing from the annealed point.  Keep
        // the move only when it improves the lexicographic max/sum score;
        // this closes the common final discrepancy gaps without another LP.
        for (int polish = 0; polish < 3 && !over_budget(); ++polish) {
            bool improved = false;
            f64 polish_score = current;
            std::vector<unsigned char> polish_on = on;
            std::vector<f64> polish_act = act;
            for (const Index j : bins) {
                if (on[sz(j)]) continue;
                for (int d = 0; d < 2; ++d) {
                    const Index qj = d == 0 ? j : bins[static_cast<std::size_t>(
                        next_rand() % bins.size())];
                    if (qj == j) continue;
                    std::vector<unsigned char> trial_on = on;
                    std::vector<f64> trial_act = act;
                    trial_on[sz(j)] ^= 1;
                    for (const auto& [r, v] : bin_rows[sz(j)]) trial_act[r] += v;
                    if (d == 1 && !trial_on[sz(qj)]) {
                        trial_on[sz(qj)] = 1;
                        for (const auto& [r, v] : bin_rows[sz(qj)]) trial_act[r] += v;
                    }
                    f64 trial_max = 0.0;
                    const f64 trial_score = score(trial_act, trial_max);
                    if (trial_score + 1e-9 < polish_score) {
                        polish_score = trial_score;
                        polish_on = std::move(trial_on);
                        polish_act = std::move(trial_act);
                        improved = true;
                    }
                }
            }
            if (!improved) break;
            on = std::move(polish_on);
            act = std::move(polish_act);
            current = polish_score;
            score(act, current_max);
        }

        // Exhaustive one-for-one swaps are especially effective once the
        // annealer has selected roughly the right number of items.  Recompute
        // the residual score for each swap and apply the best improving move
        // for a few passes.
        for (int polish = 0; polish < 4 && !over_budget(); ++polish) {
            Index best_off = -1, best_on_var = -1;
            f64 best_move_score = current;
            for (const Index off : bins) {
                if (!on[sz(off)]) continue;
                for (const Index add : bins) {
                    if (on[sz(add)] || add == off) continue;
                    for (const auto& [r, v] : bin_rows[sz(off)]) act[r] -= v;
                    for (const auto& [r, v] : bin_rows[sz(add)]) act[r] += v;
                    f64 trial_max = 0.0;
                    const f64 trial_score = score(act, trial_max);
                    for (const auto& [r, v] : bin_rows[sz(add)]) act[r] -= v;
                    for (const auto& [r, v] : bin_rows[sz(off)]) act[r] += v;
                    if (trial_score + 1e-9 < best_move_score) {
                        best_move_score = trial_score;
                        best_off = off; best_on_var = add;
                    }
                }
            }
            if (best_off < 0) break;
            on[sz(best_off)] = 0; on[sz(best_on_var)] = 1;
            for (const auto& [r, v] : bin_rows[sz(best_off)]) act[r] -= v;
            for (const auto& [r, v] : bin_rows[sz(best_on_var)]) act[r] += v;
            current = best_move_score;
            score(act, current_max);
        }

        // Two-for-two exchanges close another common discrepancy plateau:
        // changing either selected item alone worsens the max residual, while
        // the coordinated replacement is improving.  Cap the pass count and
        // enumerate only the current selected/unselected sets, which is small
        // for the intended pk1-sized models.
        for (int polish = 0; polish < 2 && !over_budget(); ++polish) {
            std::vector<Index> selected, unselected;
            for (const Index j : bins)
                (on[sz(j)] ? selected : unselected).push_back(j);
            Index best_off1 = -1, best_off2 = -1;
            Index best_add1 = -1, best_add2 = -1;
            f64 best_move_score = current;
            for (std::size_t p = 0; p < selected.size(); ++p)
                for (std::size_t q = p + 1; q < selected.size(); ++q)
                    for (std::size_t a = 0; a < unselected.size(); ++a)
                        for (std::size_t b = a + 1; b < unselected.size(); ++b) {
                            const Index off1 = selected[p], off2 = selected[q];
                            const Index add1 = unselected[a], add2 = unselected[b];
                            for (const auto& [r, v] : bin_rows[sz(off1)]) act[r] -= v;
                            for (const auto& [r, v] : bin_rows[sz(off2)]) act[r] -= v;
                            for (const auto& [r, v] : bin_rows[sz(add1)]) act[r] += v;
                            for (const auto& [r, v] : bin_rows[sz(add2)]) act[r] += v;
                            f64 trial_max = 0.0;
                            const f64 trial_score = score(act, trial_max);
                            for (const auto& [r, v] : bin_rows[sz(add2)]) act[r] -= v;
                            for (const auto& [r, v] : bin_rows[sz(add1)]) act[r] -= v;
                            for (const auto& [r, v] : bin_rows[sz(off2)]) act[r] += v;
                            for (const auto& [r, v] : bin_rows[sz(off1)]) act[r] += v;
                            if (trial_score + 1e-9 < best_move_score) {
                                best_move_score = trial_score;
                                best_off1 = off1; best_off2 = off2;
                                best_add1 = add1; best_add2 = add2;
                            }
                        }
            if (best_off1 < 0) break;
            on[sz(best_off1)] = 0; on[sz(best_off2)] = 0;
            on[sz(best_add1)] = 1; on[sz(best_add2)] = 1;
            for (const auto& [r, v] : bin_rows[sz(best_off1)]) act[r] -= v;
            for (const auto& [r, v] : bin_rows[sz(best_off2)]) act[r] -= v;
            for (const auto& [r, v] : bin_rows[sz(best_add1)]) act[r] += v;
            for (const auto& [r, v] : bin_rows[sz(best_add2)]) act[r] += v;
            current = best_move_score;
            score(act, current_max);
        }
    }
    if (!std::isfinite(best_score)) return false;

    std::vector<f64> candidate(static_cast<std::size_t>(n), 0.0);
    for (const Index j : bins) candidate[sz(j)] = best_on[sz(j)] ? 1.0 : 0.0;
    f64 max_residual = 0.0;
    for (const auto& eq : eqs) {
        f64 act = 0.0;
        for (core::Offset k = rp[sz(eq.row)];
             k < rp[sz(eq.row) + 1]; ++k) {
            const Index j = ci[sz(k)];
            if (lp.is_integer[sz(j)]) act += av[sz(k)] * candidate[sz(j)];
        }
        const f64 d = eq.target - act;
        if (d >= 0.0) candidate[sz(eq.pos)] = d;
        else candidate[sz(eq.neg)] = -d;
        max_residual = std::max(max_residual, std::fabs(d));
    }
    for (Index j = 0; j < n; ++j) {
        if (!lp.is_integer[sz(j)] && std::fabs(lp.c[sz(j)]) > 1e-12 &&
            lp.col_hi[sz(j)] > 1.0e6) {
            candidate[sz(j)] = max_residual;
            break;
        }
    }
    if (lp.max_row_violation(candidate) > feas_tol ||
        lp.max_bound_violation(candidate) > feas_tol)
        return false;
    x_out = std::move(candidate);
    return true;
}

// Direct local search for binary multi-dimensional knapsack/set-partition
// models.  markshare and pk1 have continuous residual columns, so the
// integer assignment can be scored without solving an LP for every move.
// The final assignments still go through the original LP repair/check path.
bool try_structured_binary_search(const model::LpProblem& lp,
                                  f64 int_tol,
                                  f64 feas_tol,
                                  int restarts,
                                  int iterations,
                                  int repair_trials,
                                  double time_limit_s,
                                  std::uint64_t repair_iterations,
                                  double repair_time_s,
                                  std::vector<f64>& x_out) {
    const Index n = lp.n_cols();
    if (restarts <= 0 || iterations <= 0 || repair_trials <= 0 || n == 0)
        return false;
    const auto start = Clock::now();
    bool search_phase = true;
    const double search_limit_s = time_limit_s > 0.0
        ? std::max(0.0, time_limit_s - 0.6) : 0.0;
    const auto over_budget = [&]() {
        if (time_limit_s <= 0.0) return false;
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        return elapsed >= (search_phase ? search_limit_s : time_limit_s);
    };

    std::vector<Index> bins;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (lp.col_lo[sz(j)] < -int_tol || lp.col_hi[sz(j)] > 1.0 + int_tol)
            return false;
        if (lp.col_hi[sz(j)] - lp.col_lo[sz(j)] > int_tol)
            bins.push_back(j);
    }
    if (bins.size() < 8 || lp.n_rows() > 80) return false;

    struct RowInfo {
        Index row = -1;
        f64 target = 0.0;
        f64 slack_weight = 0.0;  // working objective coefficient
        std::vector<std::pair<Index, f64>> terms;
    };
    std::vector<RowInfo> rows;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    const f64 sense = lp.maximize ? -1.0 : 1.0;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        if (!std::isfinite(lp.row_lo[sz(i)]) ||
            !std::isfinite(lp.row_hi[sz(i)]) ||
            std::fabs(lp.row_lo[sz(i)] - lp.row_hi[sz(i)]) > 1e-9)
            continue;
        RowInfo info;
        info.row = i;
        info.target = lp.row_hi[sz(i)];
        int continuous_slacks = 0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = av[sz(k)];
            if (lp.is_integer[sz(j)]) {
                info.terms.emplace_back(j, a);
                continue;
            }
            // A nonnegative +1 residual column is the standard MPS form.
            if (std::fabs(a - 1.0) <= 1e-9 &&
                lp.col_lo[sz(j)] >= -feas_tol &&
                lp.col_hi[sz(j)] > lp.col_lo[sz(j)] + int_tol) {
                ++continuous_slacks;
                info.slack_weight += sense * lp.c[sz(j)];
            }
        }
        if (!info.terms.empty() && continuous_slacks == 1)
            rows.push_back(std::move(info));
    }
    if (rows.size() < 2) return false;

    std::vector<std::vector<std::pair<Index, f64>>> col_rows(sz(n));
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (const auto& [j, a] : rows[r].terms)
            col_rows[sz(j)].emplace_back(static_cast<Index>(r), a);
    for (const Index j : bins)
        if (col_rows[sz(j)].empty()) return false;

    // pk1 minimizes an objective column constrained above every residual;
    // identify that max-residual form when residual columns themselves have
    // zero cost.  markshare instead has positive-cost residual columns.
    bool has_residual_cost = false;
    for (const auto& row : rows)
        if (std::fabs(row.slack_weight) > 1e-12) has_residual_cost = true;
    bool max_residual_objective = false;
    f64 max_residual_weight = 0.0;
    if (!has_residual_cost) {
        for (Index j = 0; j < n; ++j) {
            if (lp.is_integer[sz(j)] || std::fabs(sense * lp.c[sz(j)]) <= 1e-12)
                continue;
            bool links_residual = false;
            for (Index i = 0; i < lp.n_rows() && !links_residual; ++i) {
                if (lp.row_lo[sz(i)] < -1e-9 ||
                    std::isfinite(lp.row_hi[sz(i)])) continue;
                bool has_minus_slack = false, has_plus_obj = false;
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    const Index q = ci[sz(k)];
                    if (q == j && std::fabs(av[sz(k)] - 1.0) <= 1e-9)
                        has_plus_obj = true;
                    if (!lp.is_integer[sz(q)] && q != j &&
                        std::fabs(av[sz(k)] + 1.0) <= 1e-9)
                        has_minus_slack = true;
                }
                if (has_plus_obj && has_minus_slack) links_residual = true;
            }
            if (links_residual) {
                max_residual_objective = true;
                max_residual_weight = sense * lp.c[sz(j)];
                break;
            }
        }
    }

    const auto score_state = [&](const std::vector<f64>& act,
                                 const std::vector<unsigned char>& on) {
        f64 over = 0.0;
        f64 under = 0.0;
        f64 residual_sum = 0.0;
        f64 residual_max = 0.0;
        for (const auto& row : rows) {
            const f64 residual = row.target - act[sz(row.row)];
            if (residual < -feas_tol) over += -residual;
            residual_sum += row.slack_weight * residual;
            residual_max = std::max(residual_max, residual);
            if (residual < 0.0) under += -residual;
        }
        f64 objective = has_residual_cost ? residual_sum
                          : (max_residual_objective
                                 ? max_residual_weight * residual_max
                                 : 0.0);
        if (!has_residual_cost && !max_residual_objective) {
            for (const Index j : bins)
                objective += sense * lp.c[sz(j)] * on[sz(j)];
        }
        // Upper violations dominate; a small under term keeps zero-cost
        // assignments moving toward the equality target before repair.
        return objective + 1.0e6 * over + 1.0e-3 * under;
    };

    // Seed a feasible assignment, then search the binary space with suffix
    // capacity bounds.  The bound is valid for this incumbent heuristic only:
    // it prunes assignments that cannot fill the remaining residual, while
    // the original B&B tree continues to use certified LP relaxations.
    std::vector<Index> order = bins;
    std::sort(order.begin(), order.end(), [&](Index a, Index b) {
        f64 sa = 0.0, sb = 0.0;
        for (const auto& [r, x] : col_rows[sz(a)]) sa += std::fabs(x) /
            std::max(1.0, rows[sz(r)].target);
        for (const auto& [r, x] : col_rows[sz(b)]) sb += std::fabs(x) /
            std::max(1.0, rows[sz(r)].target);
        return sa > sb;
    });
    const std::size_t k = order.size(), nr = rows.size();

    struct BeamState {
        std::array<f64, 80> act{};
        std::uint64_t mask = 0;
        f64 key = 0.0;
    };
    std::array<f64, 80> beam_weights{};
    for (std::size_t r = 0; r < nr; ++r) beam_weights[r] = 1.0;
    const auto beam_key = [&](const BeamState& s) {
        f64 residual_sum = 0.0, residual_max = 0.0;
        for (std::size_t r = 0; r < nr; ++r) {
            const f64 residual = rows[r].target - s.act[r];
            residual_sum += beam_weights[r] * residual;
            residual_max = std::max(residual_max, residual);
        }
        if (has_residual_cost) return residual_sum + 0.5 * residual_max;
        if (max_residual_objective) return max_residual_weight * residual_max +
                                            0.001 * residual_sum;
        return residual_sum;
    };
    std::vector<f64> beam_scores;
    std::vector<std::vector<f64>> beam_candidates;
    const std::size_t beam_width = bins.size() <= 55 ? 50000 : 40000;
    std::uint64_t beam_state = 0x243f6a8885a308d3ull;
    const auto beam_rand = [&]() {
        beam_state ^= beam_state << 7; beam_state ^= beam_state >> 9;
        beam_state ^= beam_state << 8; return beam_state;
    };
    for (int beam_pass = 0; beam_pass < 6 && !over_budget(); ++beam_pass) {
        for (std::size_t r = 0; r < nr; ++r) {
            if (beam_pass == 0) beam_weights[r] = 1.0;
            else if (beam_pass == 1) beam_weights[r] = (r == 0 ? 4.0 : 1.0);
            else beam_weights[r] = 0.5 +
                static_cast<f64>((beam_rand() % 500u)) / 100.0;
        }
        std::vector<Index> beam_order = order;
        if (beam_pass > 0) {
            for (std::size_t q = beam_order.size(); q > 1; --q) {
                const std::size_t r = static_cast<std::size_t>(beam_rand() % q);
                std::swap(beam_order[q - 1], beam_order[r]);
            }
        }
        std::vector<BeamState> beam(1), next;
        for (std::size_t p = 0; p < beam_order.size() && !over_budget(); ++p) {
            const Index j = beam_order[p];
            std::vector<std::pair<Index, f64>> terms = col_rows[sz(j)];
            next.clear();
            next.reserve(std::min<std::size_t>(beam.size() * 2, beam_width * 2));
            for (const BeamState& s : beam) {
                next.push_back(s);  // skip
                BeamState take = s;
                bool fits = true;
                for (const auto& [r, a] : terms) {
                    if (a < -feas_tol || take.act[sz(r)] + a >
                        rows[sz(r)].target + feas_tol) { fits = false; break; }
                    take.act[sz(r)] += a;
                }
                if (!fits) continue;
                take.mask ^= (1ull << static_cast<unsigned>(p));
                take.key = beam_key(take) +
                    1.0e-7 * static_cast<f64>(beam_rand() & 0xffffu);
                next.push_back(take);
            }
            for (BeamState& s : next) s.key = beam_key(s) +
                1.0e-7 * static_cast<f64>(beam_rand() & 0xffffu);
            if (next.size() > beam_width) {
                std::nth_element(next.begin(), next.begin() + beam_width,
                                 next.end(), [](const BeamState& a, const BeamState& b) {
                                     return a.key < b.key;
                                 });
                next.resize(beam_width);
            }
            beam.swap(next);
        }
        if (beam.empty()) continue;
        std::sort(beam.begin(), beam.end(), [&](const BeamState& a, const BeamState& b) {
            return beam_key(a) < beam_key(b);
        });
        const std::size_t keep = std::min<std::size_t>(32, beam.size());
        for (std::size_t bq = 0; bq < keep; ++bq) {
            const BeamState& state = beam[bq];
            std::vector<f64> candidate(static_cast<std::size_t>(n), 0.0);
            std::vector<unsigned char> candidate_on(static_cast<std::size_t>(n), 0);
            for (std::size_t p = 0; p < beam_order.size(); ++p)
                if ((state.mask >> static_cast<unsigned>(p)) & 1ull) {
                    candidate[sz(beam_order[p])] = 1.0;
                    candidate_on[sz(beam_order[p])] = 1;
                }
            std::vector<f64> candidate_act(static_cast<std::size_t>(lp.n_rows()), 0.0);
            for (const Index j : bins) if (candidate_on[sz(j)])
                for (const auto& [r, a] : col_rows[sz(j)])
                    candidate_act[sz(rows[sz(r)].row)] += a;
            beam_scores.push_back(score_state(candidate_act, candidate_on));
            beam_candidates.push_back(std::move(candidate));
        }
    }

    std::vector<f64> suffix((k + 1) * nr, 0.0);
    for (std::size_t p = k; p-- > 0;) {
        for (std::size_t r = 0; r < nr; ++r)
            suffix[p * nr + r] = suffix[(p + 1) * nr + r];
        for (const auto& [r, a] : col_rows[sz(order[p])])
            suffix[p * nr + static_cast<std::size_t>(r)] += std::max(0.0, a);
    }
    f64 best_feasible = std::numeric_limits<f64>::infinity();
    std::vector<unsigned char> best_on;

    // When every residual row has unit positive cost, a very small objective
    // is equivalent to an exact multidimensional subset sum.  Search the
    // exact residual patterns first (all rows filled, or one row short by a
    // single unit).  This turns markshare-style instances from a generic
    // packing heuristic into a bounded combinatorial feasibility problem while
    // retaining a strict node/time cap.
    bool unit_integer_residual = has_residual_cost && !max_residual_objective;
    if (unit_integer_residual) {
        for (const auto& row : rows) {
            if (std::fabs(row.slack_weight - 1.0) > 1e-9 ||
                std::fabs(row.target - std::round(row.target)) > 1e-9) {
                unit_integer_residual = false;
                break;
            }
            for (const auto& [j, a] : row.terms) {
                if (std::fabs(a - std::round(a)) > 1e-9 ||
                    std::fabs(a) <= 1e-12) {
                    unit_integer_residual = false;
                    break;
                }
            }
            if (!unit_integer_residual) break;
        }
    }
    if (unit_integer_residual && !over_budget()) {
        const std::uint64_t exact_cap = bins.size() <= 55
            ? 50000000ull : 10000000ull;
        std::uint64_t exact_nodes = 0;
        std::vector<unsigned char> exact_on(static_cast<std::size_t>(n), 0);
        std::vector<f64> exact_act(static_cast<std::size_t>(lp.n_rows()), 0.0);
        auto exact_search = [&](auto&& self, std::size_t p,
                               const std::vector<f64>& exact_target) -> bool {
            if (over_budget() || ++exact_nodes > exact_cap) return false;
            for (std::size_t r = 0; r < nr; ++r) {
                const f64 rem = exact_target[r] - exact_act[sz(rows[r].row)];
                if (rem < -feas_tol || rem > suffix[p * nr + r] + feas_tol)
                    return false;
            }
            if (p == k) {
                for (std::size_t r = 0; r < nr; ++r)
                    if (std::fabs(exact_target[r] -
                                  exact_act[sz(rows[r].row)]) > feas_tol)
                        return false;
                best_feasible = score_state(exact_act, exact_on);
                best_on = exact_on;
                return true;
            }
            const Index j = order[p];
            bool fits = true;
            for (const auto& [r, a] : col_rows[sz(j)]) {
                if (exact_act[sz(rows[sz(r)].row)] + a >
                    exact_target[sz(r)] + feas_tol) {
                    fits = false;
                    break;
                }
            }
            if (fits) {
                exact_on[sz(j)] = 1;
                for (const auto& [r, a] : col_rows[sz(j)])
                    exact_act[sz(rows[sz(r)].row)] += a;
                if (self(self, p + 1, exact_target)) return true;
                for (const auto& [r, a] : col_rows[sz(j)])
                    exact_act[sz(rows[sz(r)].row)] -= a;
                exact_on[sz(j)] = 0;
            }
            return self(self, p + 1, exact_target);
        };

        // Objective 0 and objective 1 residual patterns are enough for the
        // common exact-fill construction.  If they are not found, the normal
        // packing search below continues from its best incumbent.
        std::vector<f64> exact_target(nr, 0.0);
        for (std::size_t r = 0; r < nr; ++r) exact_target[r] = rows[r].target;
        exact_search(exact_search, 0, exact_target);
        if (!std::isfinite(best_feasible) || best_feasible > 1.0 + feas_tol) {
            for (std::size_t miss = 0; miss < nr && !over_budget(); ++miss) {
                exact_target.assign(nr, 0.0);
                for (std::size_t r = 0; r < nr; ++r)
                    exact_target[r] = rows[r].target - (r == miss ? 1.0 : 0.0);
                std::fill(exact_on.begin(), exact_on.end(), 0);
                std::fill(exact_act.begin(), exact_act.end(), 0.0);
                if (exact_search(exact_search, 0, exact_target)) break;
            }
        }
    }
    std::vector<unsigned char> seed_on(static_cast<std::size_t>(n), 0);
    std::vector<f64> seed_act(static_cast<std::size_t>(lp.n_rows()), 0.0);
    for (const Index j : order) {
        bool fits = true;
        for (const auto& [r, a] : col_rows[sz(j)])
            if (seed_act[sz(rows[sz(r)].row)] + a >
                rows[sz(r)].target + feas_tol) { fits = false; break; }
        if (!fits) continue;
        seed_on[sz(j)] = 1;
        for (const auto& [r, a] : col_rows[sz(j)])
            seed_act[sz(rows[sz(r)].row)] += a;
    }
    best_feasible = score_state(seed_act, seed_on);
    best_on = seed_on;
    std::uint64_t dfs_nodes = 0;
    const std::uint64_t dfs_cap = bins.size() <= 60 ? 20000000ull : 6000000ull;
    auto dfs = [&](auto&& self, std::size_t p,
                   std::vector<f64>& act,
                   std::vector<unsigned char>& on) -> void {
        if (over_budget() || ++dfs_nodes > dfs_cap) return;
        f64 lower = 0.0;
        f64 max_lower = 0.0;
        bool over = false;
        for (std::size_t r = 0; r < nr; ++r) {
            const auto& row = rows[r];
            const f64 current_res = row.target - act[sz(row.row)];
            if (current_res < -feas_tol) { over = true; break; }
            const f64 possible = suffix[p * nr + r] - current_res;
            const f64 min_res = std::max(0.0, row.target -
                                                   (act[sz(row.row)] +
                                                    suffix[p * nr + r]));
            if (has_residual_cost) lower += row.slack_weight * min_res;
            max_lower = std::max(max_lower, min_res);
            if (possible < -feas_tol) { over = true; break; }
        }
        if (over) return;
        if (!has_residual_cost && max_residual_objective)
            lower = max_residual_weight * max_lower;
        if (!has_residual_cost && !max_residual_objective) {
            for (std::size_t q = p; q < k; ++q)
                lower += std::min(0.0, sense * lp.c[sz(order[q])]);
        }
        if (lower + 1e-9 >= best_feasible) return;
        if (p == k) {
            const f64 value = score_state(act, on);
            bool feasible = true;
            for (const auto& row : rows)
                if (row.target - act[sz(row.row)] < -feas_tol) { feasible = false; break; }
            if (feasible && value + 1e-9 < best_feasible) {
                best_feasible = value;
                best_on = on;
            }
            return;
        }
        const Index j = order[p];
        bool fits = true;
        for (const auto& [r, a] : col_rows[sz(j)])
            if (act[sz(rows[sz(r)].row)] + a >
                rows[sz(r)].target + feas_tol) { fits = false; break; }
        if (fits) {
            on[sz(j)] = 1;
            for (const auto& [r, a] : col_rows[sz(j)])
                act[sz(rows[sz(r)].row)] += a;
            self(self, p + 1, act, on);
            for (const auto& [r, a] : col_rows[sz(j)])
                act[sz(rows[sz(r)].row)] -= a;
            on[sz(j)] = 0;
        }
        self(self, p + 1, act, on);
    };
    std::vector<unsigned char> dfs_on(static_cast<std::size_t>(n), 0);
    std::vector<f64> dfs_act(static_cast<std::size_t>(lp.n_rows()), 0.0);
    dfs(dfs, 0, dfs_act, dfs_on);

    // Large-neighborhood destroy/repair.  Exact-fill instances routinely
    // require removing several currently selected items before a different
    // combination can be inserted; one-for-one local exchanges cannot cross
    // that barrier.  Keep the assignment feasible throughout the rebuild and
    // use randomized regret among the best fitting columns to diversify the
    // basins.  This is an incumbent heuristic, never a relaxation bound.
    if (!over_budget() && !best_on.empty()) {
        std::uint64_t dr_state = 0xa4093822299f31d0ull;
        const auto dr_rand = [&]() {
            dr_state ^= dr_state << 7; dr_state ^= dr_state >> 9;
            dr_state ^= dr_state << 8; return dr_state;
        };
        std::vector<unsigned char> dr_on = best_on;
        std::vector<f64> dr_act(static_cast<std::size_t>(lp.n_rows()), 0.0);
        auto rebuild_activity = [&]() {
            std::fill(dr_act.begin(), dr_act.end(), 0.0);
            for (const Index j : bins) if (dr_on[sz(j)])
                for (const auto& [r, a] : col_rows[sz(j)])
                    dr_act[sz(rows[sz(r)].row)] += a;
        };
        rebuild_activity();
        const int destroy_passes = bins.size() <= 70 ? 50000 : 20000;
        for (int pass = 0; pass < destroy_passes && !over_budget(); ++pass) {
            // Alternate perturbing the incumbent with restarting from a
            // diversified beam candidate.  Larger destroys occur rarely so
            // most iterations cheaply polish a promising assignment.
            if (pass > 0 && (pass % 11) == 0 && !beam_candidates.empty()) {
                const std::size_t q = static_cast<std::size_t>(
                    dr_rand() % beam_candidates.size());
                std::fill(dr_on.begin(), dr_on.end(), 0);
                for (const Index j : bins)
                    dr_on[sz(j)] = beam_candidates[q][sz(j)] > 0.5;
            }
            rebuild_activity();
            std::vector<Index> selected;
            selected.reserve(bins.size());
            for (const Index j : bins) if (dr_on[sz(j)]) selected.push_back(j);
            if (!selected.empty()) {
                const std::size_t max_destroy = std::min<std::size_t>(
                    10, selected.size());
                const std::size_t destroy = 1 + static_cast<std::size_t>(
                    dr_rand() % max_destroy);
                for (std::size_t q = 0; q < destroy; ++q) {
                    const std::size_t at = static_cast<std::size_t>(
                        dr_rand() % selected.size());
                    const Index j = selected[at];
                    if (!dr_on[sz(j)]) continue;
                    dr_on[sz(j)] = 0;
                    for (const auto& [r, a] : col_rows[sz(j)])
                        dr_act[sz(rows[sz(r)].row)] -= a;
                    selected[at] = selected.back();
                    selected.pop_back();
                }
            }

            // Refill until no column fits.  Choosing from a short randomized
            // regret list avoids the deterministic greedy trap while keeping
            // each pass O(number of binaries * number of rows).
            for (std::size_t add = 0; add < bins.size(); ++add) {
                struct AddChoice { Index j = -1; f64 gain = 0.0; };
                std::array<AddChoice, 8> top{};
                std::size_t ntop = 0;
                for (const Index j : bins) {
                    if (dr_on[sz(j)]) continue;
                    bool fits = true;
                    f64 gain = 0.0;
                    f64 scarcity = 0.0;
                    for (const auto& [r, a] : col_rows[sz(j)]) {
                        const auto& row = rows[sz(r)];
                        if (a < -feas_tol || dr_act[sz(row.row)] + a >
                            row.target + feas_tol) { fits = false; break; }
                        const f64 residual = std::max(0.0,
                            row.target - dr_act[sz(row.row)]);
                        gain += std::max(0.0, row.slack_weight) * a;
                        // Columns that consume scarce residual capacity are
                        // useful only when they also close a large deficit.
                        scarcity += a / std::max(1.0, residual);
                    }
                    if (!fits) continue;
                    if (!has_residual_cost && max_residual_objective)
                        gain = 0.0;
                    gain += 1.0e-3 * scarcity;
                    AddChoice choice{j, gain};
                    std::size_t pos = ntop;
                    while (pos > 0 && top[pos - 1].gain < choice.gain &&
                           pos < top.size()) {
                        if (pos < top.size()) top[pos] = top[pos - 1];
                        --pos;
                    }
                    if (pos < top.size()) {
                        top[pos] = choice;
                        if (ntop < top.size()) ++ntop;
                    }
                }
                if (ntop == 0) break;
                const std::size_t pick = static_cast<std::size_t>(
                    dr_rand() % std::min<std::size_t>(ntop, 4));
                const Index j = top[pick].j;
                dr_on[sz(j)] = 1;
                for (const auto& [r, a] : col_rows[sz(j)])
                    dr_act[sz(rows[sz(r)].row)] += a;
            }
            const f64 value = score_state(dr_act, dr_on);
            if (value + 1e-9 < best_feasible) {
                best_feasible = value;
                best_on = dr_on;
            }
        }
    }
    search_phase = false;

    struct Candidate { f64 score = 0.0; std::vector<f64> x; };
    std::vector<Candidate> pool;
    for (std::size_t q = 0; q < beam_candidates.size(); ++q) {
        Candidate c;
        c.score = beam_scores[q];
        c.x = std::move(beam_candidates[q]);
        pool.push_back(std::move(c));
    }
    if (!best_on.empty() && std::isfinite(best_feasible)) {
        Candidate c;
        c.score = best_feasible;
        c.x.assign(static_cast<std::size_t>(n), 0.0);
        for (const Index j : bins) c.x[sz(j)] = best_on[sz(j)] ? 1.0 : 0.0;
        pool.push_back(std::move(c));
    }
    std::uint64_t state = 0x517cc1b727220a95ull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };

    for (int restart = 0; restart < restarts && !over_budget(); ++restart) {
        std::vector<unsigned char> on(static_cast<std::size_t>(n), 0);
        std::vector<f64> act(static_cast<std::size_t>(lp.n_rows()), 0.0);
        std::vector<Index> restart_order = bins;
        for (std::size_t q = restart_order.size(); q > 1; --q) {
            const std::size_t r = static_cast<std::size_t>(next_rand() % q);
            std::swap(restart_order[q - 1], restart_order[r]);
        }
        // Half the restarts begin at a random feasible packing; the other
        // half begin with a random dense assignment and remove violations.
        if ((restart & 1) == 0) {
            for (const Index j : restart_order) {
                bool fits = true;
                for (const auto& [r, a] : col_rows[sz(j)])
                    if (act[sz(rows[sz(r)].row)] + a >
                        rows[sz(r)].target + feas_tol) { fits = false; break; }
                if (fits && (next_rand() & 3ull) != 0ull) {
                    on[sz(j)] = 1;
                    for (const auto& [r, a] : col_rows[sz(j)])
                        act[sz(rows[sz(r)].row)] += a;
                }
            }
        } else {
            for (const Index j : bins) on[sz(j)] = (next_rand() & 1ull) != 0;
            for (const Index j : bins) if (on[sz(j)])
                for (const auto& [r, a] : col_rows[sz(j)])
                    act[sz(rows[sz(r)].row)] += a;
            // Remove random selected items until all equality upper bounds
            // are respected; this gives the descent a useful feasible basin.
            for (std::size_t pass = 0; pass < bins.size() * 2; ++pass) {
                bool valid = true;
                for (const auto& row : rows)
                    if (act[sz(row.row)] > row.target + feas_tol) { valid = false; break; }
                if (valid) break;
                const Index j = bins[static_cast<std::size_t>(next_rand() % bins.size())];
                if (!on[sz(j)]) continue;
                on[sz(j)] = 0;
                for (const auto& [r, a] : col_rows[sz(j)])
                    act[sz(rows[sz(r)].row)] -= a;
            }
        }
        // Rebuild activity for the dense path (and guard against stale values).
        std::fill(act.begin(), act.end(), 0.0);
        for (const Index j : bins) if (on[sz(j)])
            for (const auto& [r, a] : col_rows[sz(j)])
                act[sz(rows[sz(r)].row)] += a;

        f64 current = score_state(act, on);
        f64 restart_best_feasible = std::numeric_limits<f64>::infinity();
        std::vector<unsigned char> restart_best_on;
        for (int it = 0; it < iterations && !over_budget(); ++it) {
            f64 best_move = current;
            Index best_a = -1, best_b = -1, best_c = -1;
            int best_da = 0, best_db = 0, best_dc = 0;
            auto consider = [&](Index a, int da, Index b, int db) {
                if (a == b && b >= 0) return;
                const f64 old_a = on[sz(a)];
                const f64 old_b = b >= 0 ? on[sz(b)] : 0.0;
                if (b < 0 && ((da > 0 && old_a > 0.5) ||
                              (da < 0 && old_a < 0.5))) return;
                if (b >= 0 && ((db > 0 && old_b > 0.5) ||
                               (db < 0 && old_b < 0.5))) return;
                for (const auto& [r, coeff] : col_rows[sz(a)])
                    act[sz(rows[sz(r)].row)] += coeff * da;
                if (b >= 0) for (const auto& [r, coeff] : col_rows[sz(b)])
                    act[sz(rows[sz(r)].row)] += coeff * db;
                on[sz(a)] = static_cast<unsigned char>(old_a + da);
                if (b >= 0) on[sz(b)] = static_cast<unsigned char>(old_b + db);
                const f64 candidate = score_state(act, on);
                on[sz(a)] = static_cast<unsigned char>(old_a);
                if (b >= 0) on[sz(b)] = static_cast<unsigned char>(old_b);
                if (b >= 0) for (const auto& [r, coeff] : col_rows[sz(b)])
                    act[sz(rows[sz(r)].row)] -= coeff * db;
                for (const auto& [r, coeff] : col_rows[sz(a)])
                    act[sz(rows[sz(r)].row)] -= coeff * da;
                if (candidate + 1e-9 < best_move) {
                    best_move = candidate; best_a = a; best_b = b;
                    best_da = da; best_db = db;
                }
            };
            for (const Index j : bins) consider(j, on[sz(j)] ? -1 : 1, -1, 0);
            if (bins.size() <= 160) {
                for (std::size_t p = 0; p < bins.size(); ++p)
                    for (std::size_t q = p + 1; q < bins.size(); ++q) {
                        const Index a = bins[p], b = bins[q];
                        if (on[sz(a)] && !on[sz(b)]) consider(a, -1, b, 1);
                        else if (!on[sz(a)] && on[sz(b)]) consider(a, 1, b, -1);
                        else if (!on[sz(a)] && !on[sz(b)]) consider(a, 1, b, 1);
                    }
            }
            // Exact-fill instances often require a three-way exchange before
            // the residual objective improves. Enumerate triples only after
            // the cheaper one/two-bit neighborhood has reached a plateau.
            if (best_a < 0 && bins.size() <= 80) {
                for (std::size_t p = 0; p < bins.size() && best_a < 0; ++p)
                    for (std::size_t q = p + 1; q < bins.size() && best_a < 0; ++q)
                        for (std::size_t r = q + 1; r < bins.size(); ++r) {
                            const Index a = bins[p], b = bins[q], c = bins[r];
                            const int da = on[sz(a)] ? -1 : 1;
                            const int db = on[sz(b)] ? -1 : 1;
                            const int dc = on[sz(c)] ? -1 : 1;
                            for (const auto& [rr, coeff] : col_rows[sz(a)])
                                act[sz(rows[sz(rr)].row)] += coeff * da;
                            for (const auto& [rr, coeff] : col_rows[sz(b)])
                                act[sz(rows[sz(rr)].row)] += coeff * db;
                            for (const auto& [rr, coeff] : col_rows[sz(c)])
                                act[sz(rows[sz(rr)].row)] += coeff * dc;
                            on[sz(a)] = static_cast<unsigned char>(on[sz(a)] + da);
                            on[sz(b)] = static_cast<unsigned char>(on[sz(b)] + db);
                            on[sz(c)] = static_cast<unsigned char>(on[sz(c)] + dc);
                            const f64 candidate = score_state(act, on);
                            on[sz(a)] = static_cast<unsigned char>(on[sz(a)] - da);
                            on[sz(b)] = static_cast<unsigned char>(on[sz(b)] - db);
                            on[sz(c)] = static_cast<unsigned char>(on[sz(c)] - dc);
                            for (const auto& [rr, coeff] : col_rows[sz(c)])
                                act[sz(rows[sz(rr)].row)] -= coeff * dc;
                            for (const auto& [rr, coeff] : col_rows[sz(b)])
                                act[sz(rows[sz(rr)].row)] -= coeff * db;
                            for (const auto& [rr, coeff] : col_rows[sz(a)])
                                act[sz(rows[sz(rr)].row)] -= coeff * da;
                            if (candidate + 1e-9 < best_move) {
                                best_move = candidate;
                                best_a = a; best_b = b; best_da = da; best_db = db;
                                // Encode the third variable by applying the
                                // move immediately below through a temporary
                                // sentinel; its index is recovered from the
                                // current triple after the search.
                                best_c = c;
                                best_dc = dc;
                            }
                        }
            }
            if (best_a >= 0) {
                for (const auto& [r, coeff] : col_rows[sz(best_a)])
                    act[sz(rows[sz(r)].row)] += coeff * best_da;
                on[sz(best_a)] = static_cast<unsigned char>(on[sz(best_a)] + best_da);
                if (best_b >= 0) {
                    for (const auto& [r, coeff] : col_rows[sz(best_b)])
                        act[sz(rows[sz(r)].row)] += coeff * best_db;
                    on[sz(best_b)] = static_cast<unsigned char>(on[sz(best_b)] + best_db);
                }
                if (best_c >= 0) {
                    for (const auto& [r, coeff] : col_rows[sz(best_c)])
                        act[sz(rows[sz(r)].row)] += coeff * best_dc;
                    on[sz(best_c)] = static_cast<unsigned char>(on[sz(best_c)] + best_dc);
                }
                current = best_move;
            } else {
                // Random kick to cross a plateau, followed by the greedy
                // passes above on the next iterations.
                const std::size_t kick_count = 1 + static_cast<std::size_t>(next_rand() %
                    std::min<std::size_t>(5, bins.size()));
                std::vector<Index> changed;
                for (std::size_t z = 0; z < kick_count; ++z) {
                    const Index j = bins[static_cast<std::size_t>(next_rand() % bins.size())];
                    if (std::find(changed.begin(), changed.end(), j) != changed.end()) continue;
                    changed.push_back(j);
                    const int d = on[sz(j)] ? -1 : 1;
                    on[sz(j)] = static_cast<unsigned char>(on[sz(j)] + d);
                    for (const auto& [r, coeff] : col_rows[sz(j)])
                        act[sz(rows[sz(r)].row)] += coeff * d;
                }
                current = score_state(act, on);
            }
            bool feasible = true;
            for (const auto& row : rows)
                if (row.target - act[sz(row.row)] < -feas_tol) { feasible = false; break; }
            if (feasible && current < best_feasible) {
                best_feasible = current;
                restart_best_on = on;
            }
        }
        if (restart_best_on.empty()) continue;
        Candidate c;
        c.score = restart_best_feasible;
        c.x.assign(static_cast<std::size_t>(n), 0.0);
        for (const Index j : bins) c.x[sz(j)] = restart_best_on[sz(j)] ? 1.0 : 0.0;
        pool.push_back(std::move(c));
    }

    std::sort(pool.begin(), pool.end(), [](const Candidate& a, const Candidate& b) {
        return a.score < b.score;
    });
    if (static_cast<int>(pool.size()) > repair_trials)
        pool.resize(static_cast<std::size_t>(repair_trials));
    bool found = false;
    f64 best_obj = lp.maximize ? -std::numeric_limits<f64>::infinity()
                               : std::numeric_limits<f64>::infinity();
    for (const Candidate& c : pool) {
        if (over_budget()) break;
        std::vector<f64> repaired;
        if (!try_lp_rounding_repair(lp, c.x, int_tol, feas_tol,
                                    repair_iterations, repair_time_s,
                                    repaired)) {
            continue;
        }
        const f64 obj = lp.objective(repaired);
        if (!std::isfinite(obj)) continue;
        if (!found || (lp.maximize ? obj > best_obj : obj < best_obj)) {
            found = true;
            best_obj = obj;
            x_out = std::move(repaired);
        }
    }
    return found;
}

// Penalty local search for binary equality/knapsack blocks. Unlike the
// feasibility-only neighborhood, this deliberately crosses temporarily
// infeasible assignments, which is necessary for multidimensional subset-sum
// instances whose optimum is separated by an upper-bound violation.
bool try_binary_penalty_search(const model::LpProblem& lp,
                               f64 int_tol,
                               f64 feas_tol,
                               int restarts,
                               int iterations,
                               int lp_trials,
                               double time_limit_s,
                               std::uint64_t repair_iterations,
                               double repair_time_s,
                               std::vector<f64>& x_out) {
    const Index n = lp.n_cols();
    if (restarts <= 0 || iterations <= 0 || lp_trials <= 0) return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    std::vector<Index> bins;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (lp.col_lo[sz(j)] < -int_tol || lp.col_hi[sz(j)] > 1.0 + int_tol)
            continue;
        bins.push_back(j);
    }
    if (bins.size() < 8 || bins.size() * 2 < lp.n_integer()) return false;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<std::vector<std::pair<Index, f64>>> col_rows(sz(n));
    for (Index i = 0; i < lp.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            col_rows[sz(ci[sz(k)])].emplace_back(i, av[sz(k)]);
    const auto penalty = [&](const std::vector<f64>& act) {
        f64 p = 0.0;
        for (Index i = 0; i < lp.n_rows(); ++i) {
            const f64 lo = lp.row_lo[sz(i)], hi = lp.row_hi[sz(i)];
            const f64 under = act[sz(i)] < lo ? lo - act[sz(i)] : 0.0;
            const f64 over = act[sz(i)] > hi ? act[sz(i)] - hi : 0.0;
            p += under + 1000.0 * over;
        }
        return p;
    };
    struct Candidate { f64 score; std::vector<f64> x; };
    std::vector<Candidate> pool;
    pool.reserve(static_cast<std::size_t>(lp_trials));
    std::uint64_t state = 0x94d049bb133111ebull;
    const auto next_rand = [&]() {
        state ^= state << 7; state ^= state >> 9; state ^= state << 8;
        return state;
    };
    for (int restart = 0; restart < restarts && !over_budget(); ++restart) {
        std::vector<f64> trial(static_cast<std::size_t>(n), 0.0);
        if ((restart & 1) == 0) {
            // Start half the restarts from a feasible randomized packing. It
            // gives the flip search a useful incumbent basin instead of
            // spending its cooling schedule removing massive overshoots.
            std::vector<Index> order = bins;
            for (std::size_t q = order.size(); q > 1; --q) {
                const std::size_t r = static_cast<std::size_t>(next_rand() % q);
                std::swap(order[q - 1], order[r]);
            }
            for (const Index j : order) {
                bool fits = true;
                for (const auto& [i, a] : col_rows[sz(j)]) {
                    if (a > 0.0 &&
                        lp.row_hi[sz(i)] < a - feas_tol) { fits = false; break; }
                }
                if (!fits) continue;
                trial[sz(j)] = 1.0;
                bool upper_ok = true;
                for (Index i = 0; i < lp.n_rows(); ++i) {
                    f64 arow = 0.0;
                    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                        arow += av[sz(k)] * trial[sz(ci[sz(k)])];
                    if (arow > lp.row_hi[sz(i)] + feas_tol) {
                        upper_ok = false; break;
                    }
                }
                if (!upper_ok) trial[sz(j)] = 0.0;
            }
        } else {
            for (const Index j : bins)
                trial[sz(j)] = (next_rand() & 1ull) ? 1.0 : 0.0;
        }
        std::vector<f64> act(static_cast<std::size_t>(lp.n_rows()), 0.0);
        for (const Index j : bins)
            if (trial[sz(j)] != 0.0)
                for (const auto& [i, a] : col_rows[sz(j)]) act[sz(i)] += a;
        f64 score = penalty(act);
        f64 temp = std::max(1.0, score * 0.05);
        f64 best_feasible_score = std::numeric_limits<f64>::infinity();
        std::vector<f64> best_feasible;
        for (int it = 0; it < iterations && !over_budget(); ++it) {
            const Index j = bins[static_cast<std::size_t>(next_rand() % bins.size())];
            const f64 old = trial[sz(j)];
            const f64 neu = old > 0.5 ? 0.0 : 1.0;
            const f64 delta = neu - old;
            f64 next_score = score;
            for (const auto& [i, a] : col_rows[sz(j)]) {
                const f64 before = act[sz(i)];
                const f64 after = before + a * delta;
                const f64 lo = lp.row_lo[sz(i)], hi = lp.row_hi[sz(i)];
                const f64 pb = (before < lo ? lo - before : 0.0) +
                               1000.0 * (before > hi ? before - hi : 0.0);
                const f64 pa = (after < lo ? lo - after : 0.0) +
                               1000.0 * (after > hi ? after - hi : 0.0);
                next_score += pa - pb;
            }
            const bool accept = next_score <= score ||
                (temp > 1e-9 &&
                 static_cast<f64>(next_rand() % 1000000) / 1000000.0 <
                     std::exp(std::min(0.0, (score - next_score) / temp)));
            if (accept) {
                trial[sz(j)] = neu;
                for (const auto& [i, a] : col_rows[sz(j)]) act[sz(i)] += a * delta;
                score = next_score;
            }
            temp *= 0.9995;
            bool feasible_upper = true;
            for (Index i = 0; i < lp.n_rows(); ++i)
                if (act[sz(i)] > lp.row_hi[sz(i)] + feas_tol) {
                    feasible_upper = false;
                    break;
                }
            if (feasible_upper) {
                f64 deficit = 0.0;
                for (Index i = 0; i < lp.n_rows(); ++i)
                    if (act[sz(i)] < lp.row_lo[sz(i)])
                        deficit += lp.row_lo[sz(i)] - act[sz(i)];
                if (deficit < best_feasible_score) {
                    best_feasible_score = deficit;
                    best_feasible = trial;
                }
            }
        }
        if (!best_feasible.empty()) {
            Candidate c;
            c.score = best_feasible_score;
            c.x = std::move(best_feasible);
            pool.push_back(std::move(c));
        }
    }
    std::sort(pool.begin(), pool.end(), [](const Candidate& a, const Candidate& b) {
        return a.score < b.score;
    });
    if (static_cast<int>(pool.size()) > lp_trials)
        pool.resize(static_cast<std::size_t>(lp_trials));
    bool found = false;
    f64 best_obj = lp.maximize ? -std::numeric_limits<f64>::infinity()
                               : std::numeric_limits<f64>::infinity();
    for (const Candidate& c : pool) {
        if (over_budget()) break;
        std::vector<f64> repaired;
        if (!try_lp_rounding_repair(lp, c.x, int_tol, feas_tol,
                                    repair_iterations, repair_time_s,
                                    repaired))
            continue;
        const f64 obj = lp.objective(repaired);
        if (!std::isfinite(obj)) continue;
        if (!found || (lp.maximize ? obj > best_obj : obj < best_obj)) {
            found = true;
            best_obj = obj;
            x_out = std::move(repaired);
        }
    }
    return found;
}

// Re-optimize the continuous variables while exploring a bounded Hamming
// neighborhood of an integer incumbent. This is the small-instance form of
// RINS/local branching: fixing all integer variables makes every trial an LP,
// while one-variable moves and one-for-one swaps escape poor independent
// rounding (notably subset-sum and assignment models). It is incumbent-only;
// no trial is used for node bounds or pruning.
bool try_integer_neighborhood(const model::LpProblem& lp,
                              const std::vector<f64>& seed,
                              f64 int_tol,
                              f64 feas_tol,
                              std::uint64_t max_trials,
                              double time_limit_s,
                              std::uint64_t repair_iterations,
                              double repair_time_s,
                              std::vector<f64>& x_out) {
    if (static_cast<Index>(seed.size()) != lp.n_cols() || max_trials == 0)
        return false;
    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   time_limit_s;
    };
    std::vector<f64> best = seed;
    f64 best_obj = lp.objective(best);
    if (!std::isfinite(best_obj) || lp.max_row_violation(best) > feas_tol ||
        lp.max_bound_violation(best) > feas_tol)
        return false;

    bool improved = false;
    std::uint64_t trials = 0;
    const auto better = [&](f64 a, f64 b) {
        return lp.maximize ? a > b : a < b;
    };
    const bool all_integer =
        static_cast<Index>(std::count(lp.is_integer.begin(),
                                      lp.is_integer.end(), true)) ==
        lp.n_cols();

    // For a fully integral model, fixing a trial assignment and solving an LP
    // is redundant.  Use direct feasible integer exchanges instead.  Unequal
    // exchanges (one-for-two and two-for-one) are important for covering and
    // generalized-assignment models where a cheap item can replace several
    // expensive units.  This bounded local search is exact with respect to
    // every accepted incumbent but remains heuristic-only.
    if (all_integer && lp.n_cols() <= 400) {
        std::vector<Index> movable;
        for (Index j = 0; j < lp.n_cols(); ++j)
            if (lp.col_hi[sz(j)] - lp.col_lo[sz(j)] > int_tol)
                movable.push_back(j);
        const auto feasible = [&](const std::vector<f64>& x) {
            return lp.max_row_violation(x) <= feas_tol &&
                   lp.max_bound_violation(x) <= feas_tol;
        };
        if (feasible(best) && !movable.empty()) {
            for (int pass = 0; pass < 6 && trials < max_trials &&
                                !over_budget(); ++pass) {
                bool pass_improved = false;
                std::vector<f64> next_best = best;
                f64 next_obj = best_obj;
                auto evaluate = [&](const std::vector<f64>& trial) {
                    if (trials >= max_trials || over_budget()) return;
                    ++trials;
                    if (!feasible(trial)) return;
                    const f64 obj = lp.objective(trial);
                    if (std::isfinite(obj) && better(obj, next_obj)) {
                        next_obj = obj;
                        next_best = trial;
                        pass_improved = true;
                    }
                };
                for (const Index j : movable) {
                    const f64 v = std::round(best[sz(j)]);
                    for (int d : {-1, 1}) {
                        const f64 nv = std::min(std::max(v + d,
                            lp.col_lo[sz(j)]), lp.col_hi[sz(j)]);
                        if (std::fabs(nv - v) <= int_tol) continue;
                        std::vector<f64> trial = best;
                        trial[sz(j)] = nv;
                        evaluate(trial);
                    }
                }
                // One-for-two and two-for-one exchanges.  Limit the pair
                // enumeration for wide models, but cover all pairs on the
                // small structures this fast path is intended for.
                for (const Index jd : movable) {
                    const f64 vd = std::round(best[sz(jd)]);
                    if (vd <= lp.col_lo[sz(jd)] + int_tol) continue;
                    for (const Index ju : movable) {
                        if (jd == ju) continue;
                        const f64 vu = std::round(best[sz(ju)]);
                        if (vu >= lp.col_hi[sz(ju)] - int_tol) continue;
                        std::vector<f64> trial = best;
                        trial[sz(jd)] = vd - 1.0;
                        trial[sz(ju)] = std::min(lp.col_hi[sz(ju)], vu + 2.0);
                        evaluate(trial);
                        if (vu + 2.0 <= lp.col_hi[sz(ju)] + int_tol) {
                            trial[sz(jd)] = vd - 2.0;
                            trial[sz(ju)] = vu + 1.0;
                            if (vd - 2.0 >= lp.col_lo[sz(jd)] - int_tol)
                                evaluate(trial);
                        }
                        if (trials >= max_trials || over_budget()) break;
                    }
                    if (trials >= max_trials || over_budget()) break;
                }
                if (!pass_improved) break;
                best_obj = next_obj;
                best = std::move(next_best);
                improved = true;
            }
            // Randomized destroy/repair escapes multi-unit plateaus that are
            // invisible to the short exchange passes above.  Removed units
            // are restored by selecting the best feasible +1 move according
            // to reduction in total lower-row violation per objective cost.
            std::uint64_t dr_state = 0x2545f4914f6cdd1dull;
            const auto dr_rand = [&]() {
                dr_state ^= dr_state << 7; dr_state ^= dr_state >> 9;
                dr_state ^= dr_state << 8; return dr_state;
            };
            const auto activity_of = [&](const std::vector<f64>& x) {
                std::vector<f64> act(static_cast<std::size_t>(lp.n_rows()), 0.0);
                const auto& rp = lp.A.pattern.row_ptr();
                const auto& ci = lp.A.pattern.col_idx();
                for (Index i = 0; i < lp.n_rows(); ++i)
                    for (core::Offset k = rp[sz(i)];
                         k < rp[sz(i) + 1]; ++k)
                        act[sz(i)] += lp.A.vals[sz(k)] * x[sz(ci[sz(k)])];
                return act;
            };
            const int destroy_restarts = lp.n_cols() <= 220 ? 1200 : 300;
            for (int restart = 0; restart < destroy_restarts &&
                                  trials < max_trials && !over_budget(); ++restart) {
                std::vector<f64> trial = best;
                const int removes = 1 + static_cast<int>(dr_rand() % 8u);
                for (int q = 0; q < removes; ++q) {
                    const Index j = movable[static_cast<std::size_t>(
                        dr_rand() % movable.size())];
                    const f64 v = std::round(trial[sz(j)]);
                    if (v > lp.col_lo[sz(j)] + int_tol)
                        trial[sz(j)] = v - 1.0;
                }
                for (int repair = 0; repair < lp.n_cols() * 4; ++repair) {
                    const auto act = activity_of(trial);
                    f64 before = 0.0;
                    for (Index i = 0; i < lp.n_rows(); ++i)
                        if (act[sz(i)] < lp.row_lo[sz(i)])
                            before += lp.row_lo[sz(i)] - act[sz(i)];
                    if (before <= feas_tol) break;
                    Index best_j = -1;
                    f64 best_score = -std::numeric_limits<f64>::infinity();
                    for (const Index j : movable) {
                        const f64 v = std::round(trial[sz(j)]);
                        if (v >= lp.col_hi[sz(j)] - int_tol) continue;
                        bool upper_ok = true;
                        f64 after = before;
                        for (Index i = 0; i < lp.n_rows(); ++i) {
                            f64 delta = 0.0;
                            for (core::Offset k = lp.A.pattern.row_ptr()[sz(i)];
                                 k < lp.A.pattern.row_ptr()[sz(i) + 1]; ++k)
                                if (lp.A.pattern.col_idx()[sz(k)] == j)
                                    delta += lp.A.vals[sz(k)];
                            if (delta == 0.0) continue;
                            const f64 neu = act[sz(i)] + delta;
                            if (neu > lp.row_hi[sz(i)] + feas_tol) {
                                upper_ok = false; break;
                            }
                            const f64 old_v = std::max(0.0,
                                lp.row_lo[sz(i)] - act[sz(i)]);
                            const f64 new_v = std::max(0.0,
                                lp.row_lo[sz(i)] - neu);
                            after += new_v - old_v;
                        }
                        if (!upper_ok) continue;
                        const f64 gain = before - after;
                        if (gain <= 1e-12) continue;
                        const f64 cost = std::max(0.0, lp.c[sz(j)]);
                        const f64 score = gain / (1.0 + cost) +
                            1.0e-9 * static_cast<f64>(dr_rand() & 0xffffu);
                        if (score > best_score) {
                            best_score = score; best_j = j;
                        }
                    }
                    if (best_j < 0) break;
                    trial[sz(best_j)] = std::round(trial[sz(best_j)]) + 1.0;
                }
                ++trials;
                if (!feasible(trial)) continue;
                const f64 obj = lp.objective(trial);
                if (std::isfinite(obj) && better(obj, best_obj)) {
                    best_obj = obj;
                    best = std::move(trial);
                    improved = true;
                }
            }
            if (improved) x_out = best;
            return improved;
        }
    }
    const auto probe = [&](const std::vector<f64>& trial,
                           f64& obj_out,
                           std::vector<f64>& point_out) {
        if (trials >= max_trials || over_budget()) return false;
        ++trials;
        std::vector<f64> repaired;
        if (all_integer) {
            if (lp.max_row_violation(trial) > feas_tol ||
                lp.max_bound_violation(trial) > feas_tol)
                return false;
            repaired = trial;
        } else if (!try_lp_rounding_repair(lp, trial, int_tol, feas_tol,
                                           repair_iterations, repair_time_s,
                                           repaired)) {
            return false;
        }
        const f64 obj = lp.objective(repaired);
        if (!std::isfinite(obj)) return false;
        obj_out = obj;
        point_out = std::move(repaired);
        return true;
    };

    // Coordinate descent over the integer lattice. The former implementation
    // only moved variables once from the initial seed, which made a useful
    // exchange impossible after the first accepted move. Rebuild the move
    // lists after every improving pass so a binary can be flipped repeatedly
    // and general integers can walk several units toward a better assignment.
    const int max_passes = 8;
    for (int pass = 0; pass < max_passes && trials < max_trials &&
                        !over_budget(); ++pass) {
        std::vector<Index> movable;
        movable.reserve(static_cast<std::size_t>(lp.n_cols()));
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            const f64 v = std::round(best[sz(j)]);
            if (v > lp.col_lo[sz(j)] + int_tol ||
                v < lp.col_hi[sz(j)] - int_tol)
                movable.push_back(j);
        }

        bool pass_improved = false;
        std::vector<f64> pass_best;
        f64 pass_obj = best_obj;
        for (const Index j : movable) {
            const f64 v = std::round(best[sz(j)]);
            for (int dir : {-1, 1}) {
                if (trials >= max_trials || over_budget()) break;
                const f64 next = std::min(
                    std::max(v + static_cast<f64>(dir), lp.col_lo[sz(j)]),
                    lp.col_hi[sz(j)]);
                if (std::fabs(next - v) <= int_tol) continue;
                std::vector<f64> trial = best;
                trial[sz(j)] = next;
                f64 obj = 0.0;
                std::vector<f64> point;
                if (probe(trial, obj, point) && better(obj, pass_obj)) {
                    pass_obj = obj;
                    pass_best = std::move(point);
                    pass_improved = true;
                }
            }
        }
        if (pass_improved) {
            best_obj = pass_obj;
            best = std::move(pass_best);
            improved = true;
            continue;
        }

        // At a one-coordinate local minimum, test one-for-one exchanges. This
        // is the important step for exact-sum/assignment models: removing one
        // selected item may require adding another in the same move. Cap the
        // Cartesian product so large instances keep a predictable budget.
        std::vector<Index> down, up;
        for (const Index j : movable) {
            const f64 v = std::round(best[sz(j)]);
            if (v > lp.col_lo[sz(j)] + int_tol) down.push_back(j);
            if (v < lp.col_hi[sz(j)] - int_tol) up.push_back(j);
        }
        f64 swap_obj = best_obj;
        std::vector<f64> swap_best;
        const std::uint64_t swap_cap = std::min<std::uint64_t>(
            max_trials - trials, 4096);
        std::uint64_t swap_count = 0;
        for (const Index jd : down) {
            for (const Index ju : up) {
                if (jd == ju || swap_count >= swap_cap ||
                    trials >= max_trials || over_budget()) break;
                std::vector<f64> trial = best;
                const f64 vd = std::round(best[sz(jd)]);
                const f64 vu = std::round(best[sz(ju)]);
                trial[sz(jd)] = std::max(lp.col_lo[sz(jd)], vd - 1.0);
                trial[sz(ju)] = std::min(lp.col_hi[sz(ju)], vu + 1.0);
                if (std::fabs(trial[sz(jd)] - vd) <= int_tol ||
                    std::fabs(trial[sz(ju)] - vu) <= int_tol)
                    continue;
                ++swap_count;
                f64 obj = 0.0;
                std::vector<f64> point;
                if (probe(trial, obj, point) && better(obj, swap_obj)) {
                    swap_obj = obj;
                    swap_best = std::move(point);
                }
            }
            if (swap_count >= swap_cap || trials >= max_trials || over_budget())
                break;
        }
        if (!swap_best.empty()) {
            best_obj = swap_obj;
            best = std::move(swap_best);
            improved = true;
            continue;
        }
        break;
    }

    // A small deterministic multi-exchange tail catches plateaus where no
    // single move or pair swap improves, without spending the whole B&B
    // budget on thousands of LPs.
    if (trials < max_trials && !over_budget()) {
        std::uint64_t state = 0x9e3779b97f4a7c15ull;
        const auto next_rand = [&]() {
            state ^= state << 7; state ^= state >> 9; state ^= state << 8;
            return state;
        };
        for (std::uint64_t attempts = 0;
             attempts < 512 && trials < max_trials && !over_budget(); ++attempts) {
            std::vector<Index> down, up;
            for (Index j = 0; j < lp.n_cols(); ++j) {
                if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
                const f64 v = std::round(best[sz(j)]);
                if (v > lp.col_lo[sz(j)] + int_tol) down.push_back(j);
                if (v < lp.col_hi[sz(j)] - int_tol) up.push_back(j);
            }
            if (down.empty() || up.empty()) break;
            const std::size_t k = 1 +
                static_cast<std::size_t>(next_rand() %
                    std::min<std::size_t>(4, std::min(down.size(), up.size())));
            std::vector<f64> trial = best;
            std::vector<Index> used_d, used_u;
            for (std::size_t q = 0; q < k; ++q) {
                const Index jd = down[static_cast<std::size_t>(next_rand() % down.size())];
                const Index ju = up[static_cast<std::size_t>(next_rand() % up.size())];
                if (std::find(used_d.begin(), used_d.end(), jd) != used_d.end() ||
                    std::find(used_u.begin(), used_u.end(), ju) != used_u.end()) {
                    used_d.clear();
                    break;
                }
                used_d.push_back(jd); used_u.push_back(ju);
            }
            if (used_d.size() != k) continue;
            for (const Index j : used_d)
                trial[sz(j)] = std::round(best[sz(j)]) - 1.0;
            for (const Index j : used_u)
                trial[sz(j)] = std::round(best[sz(j)]) + 1.0;
            f64 obj = 0.0;
            std::vector<f64> point;
            if (probe(trial, obj, point) && better(obj, best_obj)) {
                best_obj = obj;
                best = std::move(point);
                improved = true;
            }
        }
    }

    // Perturb-and-polish restarts cross barriers that require a temporarily
    // worse feasible assignment before a two/three-variable exchange becomes
    // profitable. Keep the global incumbent separate from the restart point.
    if (trials < max_trials && !over_budget()) {
        std::uint64_t state = 0x6a09e667f3bcc909ull;
        const auto next_rand = [&]() {
            state ^= state << 7; state ^= state >> 9; state ^= state << 8;
            return state;
        };
        for (int restart = 0; restart < 256 && trials < max_trials &&
                              !over_budget(); ++restart) {
            std::vector<Index> movable;
            for (Index j = 0; j < lp.n_cols(); ++j) {
                if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
                const f64 v = std::round(best[sz(j)]);
                if (v > lp.col_lo[sz(j)] + int_tol ||
                    v < lp.col_hi[sz(j)] - int_tol)
                    movable.push_back(j);
            }
            if (movable.empty()) break;
            const std::size_t k = 1 + static_cast<std::size_t>(
                next_rand() % std::min<std::size_t>(6, movable.size()));
            std::vector<f64> trial = best;
            std::vector<Index> used;
            for (std::size_t q = 0; q < k; ++q) {
                const Index j = movable[static_cast<std::size_t>(
                    next_rand() % movable.size())];
                if (std::find(used.begin(), used.end(), j) != used.end()) {
                    used.clear();
                    break;
                }
                used.push_back(j);
                const f64 v = std::round(best[sz(j)]);
                if (v <= lp.col_lo[sz(j)] + int_tol)
                    trial[sz(j)] = std::min(lp.col_hi[sz(j)], v + 1.0);
                else if (v >= lp.col_hi[sz(j)] - int_tol)
                    trial[sz(j)] = std::max(lp.col_lo[sz(j)], v - 1.0);
                else
                    trial[sz(j)] = (next_rand() & 1ull) ? v + 1.0 : v - 1.0;
            }
            if (used.size() != k) continue;
            f64 obj = 0.0;
            std::vector<f64> point;
            if (!probe(trial, obj, point)) continue;
            if (better(obj, best_obj)) {
                best_obj = obj;
                best = point;
                improved = true;
            }
            // One cheap improving pass from the perturbed point.
            for (const Index j : movable) {
                if (trials >= max_trials || over_budget()) break;
                const f64 v = std::round(point[sz(j)]);
                for (int dir : {-1, 1}) {
                    const f64 next = std::min(
                        std::max(v + static_cast<f64>(dir), lp.col_lo[sz(j)]),
                        lp.col_hi[sz(j)]);
                    if (std::fabs(next - v) <= int_tol) continue;
                    std::vector<f64> one = point;
                    one[sz(j)] = next;
                    f64 one_obj = 0.0;
                    std::vector<f64> one_point;
                    if (probe(one, one_obj, one_point) &&
                        better(one_obj, best_obj)) {
                        best_obj = one_obj;
                        best = std::move(one_point);
                        improved = true;
                    }
                }
            }
        }
    }

    if (improved) x_out = std::move(best);
    return improved;
}

// Feasibility-pump style incumbent search. Alternate rounding the integer
// coordinates with an LP whose objective points toward the rounded target.
// This can repair equality-heavy models where fixing every rounded integer
// makes the continuous subproblem infeasible. It is deliberately bounded and
// is never used for node bounds or pruning.
bool try_feasibility_pump(const model::LpProblem& lp,
                          const std::vector<f64>& x_start,
                          f64 int_tol,
                          f64 feas_tol,
                          int max_rounds,
                          std::uint64_t max_iterations,
                          double time_limit_s,
                          std::vector<f64>& x_out) {
    if (static_cast<Index>(x_start.size()) != lp.n_cols()) return false;
    std::vector<f64> current = x_start;
    std::vector<f64> previous_target;
    for (int round = 0; round < max_rounds; ++round) {
        std::vector<f64> target = current;
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            f64 v = std::round(current[sz(j)]);
            if (!std::isfinite(v)) return false;
            target[sz(j)] = std::min(std::max(v, lp.col_lo[sz(j)]), lp.col_hi[sz(j)]);
            if (!is_integral(target[sz(j)], int_tol)) return false;
        }

        std::vector<f64> candidate;
        if (try_round(lp, current, int_tol, feas_tol, candidate)) {
            x_out = std::move(candidate);
            return true;
        }
        if (target == previous_target) break;
        previous_target = target;

        // Build the standard feasibility-pump L1 projection LP. For every
        // integer coordinate j, an auxiliary d_j is constrained by
        // d_j >= x_j-target_j and d_j >= target_j-x_j. Minimizing sum(d_j)
        // is materially different from a signed objective: it attracts the
        // whole LP point to the rounded target without forcing unrelated
        // variables to a bound.
        const Index n = lp.n_cols();
        const Index ni = static_cast<Index>(
            std::count(lp.is_integer.begin(), lp.is_integer.end(), true));
        model::LpProblem pump = lp;
        pump.name = lp.name + "_fp_projection";
        pump.maximize = false;
        pump.obj_offset = 0.0;
        pump.c.assign(static_cast<std::size_t>(n + ni), 0.0);
        pump.col_lo = lp.col_lo;
        pump.col_hi = lp.col_hi;
        pump.col_lo.resize(static_cast<std::size_t>(n + ni), 0.0);
        pump.col_hi.resize(static_cast<std::size_t>(n + ni), model::kInf);
        pump.is_integer.assign(static_cast<std::size_t>(n + ni), false);
        pump.row_lo = lp.row_lo;
        pump.row_hi = lp.row_hi;
        pump.row_names = lp.row_names;
        pump.row_names.resize(static_cast<std::size_t>(lp.n_rows() + 2 * ni));
        pump.col_names = lp.col_names;
        pump.col_names.resize(static_cast<std::size_t>(n + ni));
        std::vector<Index> rows, cols;
        std::vector<f64> vals;
        rows.reserve(static_cast<std::size_t>(lp.nnz() + 2 * ni));
        cols.reserve(rows.capacity());
        vals.reserve(rows.capacity());
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        for (Index i = 0; i < lp.n_rows(); ++i) {
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                rows.push_back(i);
                cols.push_back(ci[sz(k)]);
                vals.push_back(lp.A.vals[sz(k)]);
            }
        }
        Index d = 0;
        for (Index j = 0; j < n; ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            const Index dj = n + d;
            pump.c[sz(dj)] = 1.0;
            pump.col_names[sz(dj)] = "FP_D_" + std::to_string(j);
            const Index r1 = lp.n_rows() + 2 * d;
            const Index r2 = r1 + 1;
            pump.row_lo.push_back(-model::kInf);
            pump.row_hi.push_back(target[sz(j)]);
            pump.row_lo.push_back(-model::kInf);
            pump.row_hi.push_back(-target[sz(j)]);
            rows.push_back(r1); cols.push_back(j);  vals.push_back(1.0);
            rows.push_back(r1); cols.push_back(dj); vals.push_back(-1.0);
            rows.push_back(r2); cols.push_back(j);  vals.push_back(-1.0);
            rows.push_back(r2); cols.push_back(dj); vals.push_back(-1.0);
            ++d;
        }
        pump.A = sparse::from_triplets(lp.n_rows() + 2 * ni, n + ni,
                                       rows, cols, vals);
        pump.validate();
        engines::SimplexOptions pump_opts;
        pump_opts.method = engines::SimplexMethod::Primal;
        pump_opts.presolve = true;
        pump_opts.max_iterations = max_iterations;
        pump_opts.time_limit_s = time_limit_s;
        pump_opts.primal_feas_tol = feas_tol;
        pump_opts.dual_feas_tol = std::max(feas_tol, 1e-7);
        engines::SimplexDiagnostics pump_diag;
        const auto pumped = engines::solve_simplex(pump, pump_opts,
                                                   pump_diag, nullptr);
        if (pumped.proposed_status != core::Status::Optimal &&
            pumped.proposed_status != core::Status::Feasible)
            break;
        if (static_cast<Index>(pumped.x.size()) < lp.n_cols()) break;
        current.assign(pumped.x.begin(), pumped.x.begin() + n);
    }
    return false;
}

// Follow one LP relaxation down a bounded depth-first branch path. Unlike a
// normal B&B node, an unproved child is still useful here as a source of a
// feasible point; only the final original-model check can accept it. This is
// the standard LP-diving incumbent heuristic, kept separate from the proof
// search so an interrupted dive can never change global bounds.
bool try_integer_dive(const model::LpProblem& lp,
                      const std::vector<f64>& x_start,
                      const std::vector<f64>& root_lo,
                      const std::vector<f64>& root_hi,
                      const engines::SimplexBasis* root_basis,
                      f64 int_tol,
                      f64 feas_tol,
                      std::uint64_t max_nodes,
                      double total_time_s,
                      double lp_time_s,
                      f64 objective_sense,
                      const engines::SimplexOptions& base_opts,
                      std::vector<f64>& x_out,
                      std::uint64_t& lp_solves) {
    lp_solves = 0;
    if (static_cast<Index>(x_start.size()) != lp.n_cols() ||
        static_cast<Index>(root_lo.size()) != lp.n_cols() ||
        static_cast<Index>(root_hi.size()) != lp.n_cols() ||
        max_nodes == 0)
        return false;

    struct DiveNode {
        std::vector<f64> lo;
        std::vector<f64> hi;
        engines::SimplexBasis basis;
        bool has_basis = false;
        int depth = 0;
    };

    const auto start = Clock::now();
    const auto over_budget = [&]() {
        return total_time_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - start).count() >=
                   total_time_s;
    };
    const auto usable_point = [&](const core::RawResult& r,
                                  const engines::SimplexDiagnostics& d) {
        if (static_cast<Index>(r.x.size()) != lp.n_cols()) return false;
        for (const f64 v : r.x)
            if (!std::isfinite(v)) return false;
        // Interrupted/Feasible LP stages can still expose a valid primal
        // point. Residual checks are mandatory because status alone is not.
        return d.primal_residual <= feas_tol &&
               lp.max_row_violation(r.x) <= feas_tol &&
               lp.max_bound_violation(r.x) <= feas_tol;
    };
    const auto integer_point = [&](const std::vector<f64>& x) {
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (!lp.is_integer.empty() && lp.is_integer[sz(j)] &&
                !is_integral(x[sz(j)], int_tol))
                return false;
        }
        return true;
    };
    const auto branch_var = [&](const std::vector<f64>& x,
                                const std::vector<f64>& lo,
                                const std::vector<f64>& hi) {
        Index best = -1;
        f64 best_score = -1.0;
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
            if (lo[sz(j)] >= hi[sz(j)] - int_tol) continue;
            if (is_integral(x[sz(j)], int_tol)) continue;
            const f64 score = frac_score(x[sz(j)]);
            if (score > best_score) {
                best_score = score;
                best = j;
            }
        }
        return best;
    };

    std::vector<DiveNode> stack;
    std::vector<f64> up_bias;  // integer_up_bias_all(lp), on first branching
    DiveNode root;
    root.lo = root_lo;
    root.hi = root_hi;
    if (root_basis != nullptr && !root_basis->basic.empty()) {
        root.basis = *root_basis;
        root.has_basis = true;
    }
    stack.push_back(std::move(root));

    std::uint64_t visited = 0;
    bool found = false;
    f64 best_obj = objective_sense > 0.0
        ? std::numeric_limits<f64>::infinity()
        : -std::numeric_limits<f64>::infinity();
    std::vector<f64> best_point;
    while (!stack.empty() && visited < max_nodes && !over_budget()) {
        DiveNode node = std::move(stack.back());
        stack.pop_back();
        ++visited;

        model::LpProblem child = lp;
        child.col_lo = node.lo;
        child.col_hi = node.hi;

        engines::SimplexOptions solve_opts = base_opts;
        solve_opts.method = node.has_basis ? engines::SimplexMethod::Dual
                                            : engines::SimplexMethod::Primal;
        solve_opts.presolve = false;
        solve_opts.max_iterations = base_opts.max_iterations == 0
            ? 10000 : base_opts.max_iterations;
        solve_opts.time_limit_s = lp_time_s;
        solve_opts.primal_feas_tol = feas_tol;
        solve_opts.dual_feas_tol = std::max(feas_tol, 1e-7);

        engines::SimplexDiagnostics sd;
        engines::SimplexBasis basis;
        core::RawResult r;
        if (node.has_basis) {
            r = engines::solve_dual_simplex(child, solve_opts, sd, &basis,
                                            &node.basis);
        } else {
            r = engines::solve_simplex(child, solve_opts, sd, &basis);
        }
        ++lp_solves;

        if (r.proposed_status == core::Status::Infeasible ||
            r.proposed_status == core::Status::InfeasibleOrUnbounded ||
            !usable_point(r, sd))
            continue;

        if (integer_point(r.x)) {
            // Check the original bounds and rows, not a relaxed/dived copy.
            if (lp.max_row_violation(r.x) <= feas_tol &&
                lp.max_bound_violation(r.x) <= feas_tol) {
                const f64 obj = lp.objective(r.x);
                const bool better = !found ||
                    (objective_sense > 0.0 ? obj < best_obj : obj > best_obj);
                if (std::isfinite(obj) && better) {
                    found = true;
                    best_obj = obj;
                    best_point = std::move(r.x);
                }
            }
            continue;
        }

        const Index br = branch_var(r.x, node.lo, node.hi);
        if (br < 0) continue;
        const f64 xv = r.x[sz(br)];
        const f64 floor_v = std::floor(xv);
        const f64 ceil_v = std::ceil(xv);
        if (floor_v < node.lo[sz(br)] - int_tol &&
            ceil_v > node.hi[sz(br)] + int_tol)
            continue;

        DiveNode down;
        down.lo = node.lo;
        down.hi = node.hi;
        down.hi[sz(br)] = std::min(down.hi[sz(br)], floor_v);
        down.basis = basis;
        down.has_basis = !basis.basic.empty();
        down.depth = node.depth + 1;

        DiveNode up;
        up.lo = node.lo;
        up.hi = node.hi;
        up.lo[sz(br)] = std::max(up.lo[sz(br)], ceil_v);
        up.basis = basis;
        up.has_basis = !basis.basic.empty();
        up.depth = node.depth + 1;

        const f64 down_delta = objective_sense * lp.c[sz(br)] *
                               (floor_v - xv);
        const f64 up_delta = objective_sense * lp.c[sz(br)] *
                             (ceil_v - xv);
        // Prefer the direction that improves structural coverage when the
        // objective coefficient is zero (the common slack-variable pattern).
        // Otherwise retain the objective-directed ordering.
        if (up_bias.empty()) up_bias = integer_up_bias_all(lp);
        const f64 bias = up_bias[sz(br)];
        const bool down_first = std::fabs(bias) > 1.0e-10
            ? bias < 0.0 : down_delta <= up_delta;
        if (down_first) {
            if (up.lo[sz(br)] <= up.hi[sz(br)] + int_tol)
                stack.push_back(std::move(up));
            if (down.lo[sz(br)] <= down.hi[sz(br)] + int_tol)
                stack.push_back(std::move(down));
        } else {
            if (down.lo[sz(br)] <= down.hi[sz(br)] + int_tol)
                stack.push_back(std::move(down));
            if (up.lo[sz(br)] <= up.hi[sz(br)] + int_tol)
                stack.push_back(std::move(up));
        }
    }
    if (found) x_out = std::move(best_point);
    return found;
}

// Relaxation Enforced Neighborhood Search (RENS): fix integer variables whose
// LP value is already close to an integer and run a short dive on the smaller
// remaining subproblem. This is incumbent-only; the original model is still
// used for the final feasibility/objective check by the caller.
bool try_rens(const model::LpProblem& lp,
              const std::vector<f64>& x_lp,
              const engines::SimplexBasis* basis,
              f64 int_tol,
              f64 feas_tol,
              std::uint64_t max_nodes,
              double time_limit_s,
              double lp_time_s,
              const engines::SimplexOptions& base_opts,
              std::vector<f64>& x_out,
              std::uint64_t& lp_solves) {
    if (static_cast<Index>(x_lp.size()) != lp.n_cols()) return false;
    model::LpProblem sub = lp;
    int fixed = 0;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        const f64 v = x_lp[sz(j)];
        const f64 r = std::round(v);
        if (std::fabs(v - r) <= 0.15 &&
            r >= lp.col_lo[sz(j)] - int_tol &&
            r <= lp.col_hi[sz(j)] + int_tol) {
            sub.col_lo[sz(j)] = std::max(lp.col_lo[sz(j)], r);
            sub.col_hi[sz(j)] = std::min(lp.col_hi[sz(j)], r);
            ++fixed;
        }
    }
    if (fixed == 0 || fixed >= static_cast<int>(lp.n_integer())) return false;
    return try_integer_dive(sub, x_lp, sub.col_lo, sub.col_hi, basis,
                            int_tol, feas_tol, max_nodes, time_limit_s,
                            lp_time_s, lp.maximize ? -1.0 : 1.0,
                            base_opts, x_out, lp_solves);
}

Index pick_branch_var(const model::LpProblem& lp,
                      const std::vector<f64>& col_lo,
                      const std::vector<f64>& col_hi,
                      const std::vector<f64>& x,
                      f64 int_tol) {
    Index best = -1;
    f64 best_frac = 0.0;
    const Index n = lp.n_cols();
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (is_integral(x[sz(j)], int_tol)) continue;
        // Skip fixed integers.
        if (col_lo[sz(j)] == col_hi[sz(j)]) continue;
        const f64 sc = frac_score(x[sz(j)]);
        if (sc > best_frac) {
            best_frac = sc;
            best = j;
        }
    }
    return best;
}

std::vector<Index> branch_candidates(const model::LpProblem& lp,
                                     const std::vector<f64>& col_lo,
                                     const std::vector<f64>& col_hi,
                                     const std::vector<f64>& x,
                                     const std::vector<Index>& col_degree,
                                     f64 int_tol,
                                     int limit) {
    std::vector<std::pair<f64, Index>> ranked;
    const Index n = lp.n_cols();
    ranked.reserve(static_cast<std::size_t>(n));
    bool has_coupled = false;
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (is_integral(x[sz(j)], int_tol)) continue;
        if (col_lo[sz(j)] == col_hi[sz(j)]) continue;
        if (sz(j) < col_degree.size() && col_degree[sz(j)] > 1) {
            has_coupled = true;
            break;
        }
    }
    for (Index j = 0; j < n; ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (is_integral(x[sz(j)], int_tol)) continue;
        if (col_lo[sz(j)] == col_hi[sz(j)]) continue;
        // Keep singleton objective/slack variables eligible: equality rows can
        // make them implied integers, and branching on that slack often
        // exposes a strong incumbent (markshare/pk1). Pure zero-cost
        // singletons remain filtered when coupled variables exist.
        if (has_coupled && sz(j) < col_degree.size() && col_degree[sz(j)] <= 1 &&
            std::fabs(lp.c[sz(j)]) <= int_tol)
            continue;
        const f64 degree = (sz(j) < col_degree.size())
            ? static_cast<f64>(col_degree[sz(j)]) : 1.0;
        // Variables touching more rows have greater propagation impact. This
        // keeps a singleton auxiliary (for example a startup indicator) from
        // dominating a structurally coupled decision variable at cold start.
        const f64 score = frac_score(x[sz(j)]) *
                          (1.0 + 0.5 * std::min(degree, 8.0)) +
                          (std::fabs(lp.c[sz(j)]) > int_tol ? 0.25 : 0.0);
        ranked.emplace_back(score, j);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });
    if (limit > 0 && static_cast<int>(ranked.size()) > limit)
        ranked.resize(static_cast<std::size_t>(limit));
    std::vector<Index> out;
    out.reserve(ranked.size());
    for (const auto& p : ranked) out.push_back(p.second);
    return out;
}

model::LpProblem tighten_integral_rows(const model::LpProblem& in,
                                       std::uint64_t& tightened_bounds) {
    model::LpProblem out = in;
    const Index m = in.n_rows();
    const Index n = in.n_cols();
    const auto& rp = in.A.pattern.row_ptr();
    const auto& ci = in.A.pattern.col_idx();
    tightened_bounds = 0;

    for (Index i = 0; i < m; ++i) {
        bool integral_image = true;
        bool has_term = false;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = in.A.vals[sz(k)];
            if (std::fabs(a) <= 1e-12) continue;
            has_term = true;
            if (j < 0 || j >= n || in.is_integer.empty() ||
                !in.is_integer[sz(j)] ||
                std::fabs(a - std::round(a)) > 1e-12) {
                integral_image = false;
                break;
            }
        }
        if (!has_term || !integral_image) continue;

        const f64 tol_lo = 1e-9 * (1.0 + std::fabs(in.row_lo[sz(i)]));
        if (std::isfinite(in.row_lo[sz(i)])) {
            const f64 tightened = std::ceil(in.row_lo[sz(i)] - tol_lo);
            if (tightened > in.row_lo[sz(i)] + tol_lo) {
                out.row_lo[sz(i)] = tightened;
                ++tightened_bounds;
            }
        }
        const f64 tol_hi = 1e-9 * (1.0 + std::fabs(in.row_hi[sz(i)]));
        if (std::isfinite(in.row_hi[sz(i)])) {
            const f64 tightened = std::floor(in.row_hi[sz(i)] + tol_hi);
            if (tightened < in.row_hi[sz(i)] - tol_hi) {
                out.row_hi[sz(i)] = tightened;
                ++tightened_bounds;
            }
        }
    }
    return out;
}

// Add a bounded set of valid binary knapsack-cover cuts. Besides pure binary
// <= rows, an equality/<= row may contain nonnegative continuous residuals;
// dropping those residuals gives a valid binary capacity projection.
model::LpProblem add_binary_cover_cuts(const model::LpProblem& in,
                                       std::uint64_t& added_cuts) {
    model::LpProblem out = in;
    const Index m = in.n_rows(), n = in.n_cols();
    const auto& rp = in.A.pattern.row_ptr();
    const auto& ci = in.A.pattern.col_idx();
    const auto& av = in.A.vals;
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    rows.reserve(static_cast<std::size_t>(in.nnz() + 1024));
    cols.reserve(rows.capacity());
    vals.reserve(rows.capacity());
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            rows.push_back(i); cols.push_back(ci[sz(k)]); vals.push_back(av[sz(k)]);
        }
    added_cuts = 0;
    const std::uint64_t cut_limit = 256;
    for (Index i = 0; i < m && added_cuts < cut_limit; ++i) {
        if (!std::isfinite(in.row_hi[sz(i)]) || in.row_hi[sz(i)] < 0.0)
            continue;
        std::vector<std::pair<Index, f64>> terms;
        bool projectable = true;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = av[sz(k)];
            if (std::fabs(a) <= 1e-12) continue;
            if (!in.is_integer.empty() && in.is_integer[sz(j)] &&
                a > 0.0 && in.col_lo[sz(j)] >= -1e-9 &&
                in.col_hi[sz(j)] <= 1.0 + 1e-9) {
                terms.emplace_back(j, a);
                continue;
            }
            // A continuous nonnegative residual can be dropped from an upper
            // capacity row. Negative residuals or free-sign variables would
            // invalidate the projection and are rejected.
            if (a < 0.0 || in.col_lo[sz(j)] < -1e-9) {
                projectable = false;
                break;
            }
        }
        if (!projectable || terms.size() < 2) continue;
        std::sort(terms.begin(), terms.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        // Pair cover cuts catch the common conflict structure cheaply.
        for (std::size_t a = 0; a < terms.size() && added_cuts < cut_limit; ++a)
            for (std::size_t b = a + 1; b < terms.size() && added_cuts < cut_limit; ++b)
                if (terms[a].second + terms[b].second > in.row_hi[sz(i)] + 1e-9) {
                    const Index r = m + static_cast<Index>(added_cuts);
                    rows.push_back(r); cols.push_back(terms[a].first); vals.push_back(1.0);
                    rows.push_back(r); cols.push_back(terms[b].first); vals.push_back(1.0);
                    out.row_lo.push_back(-model::kInf);
                    out.row_hi.push_back(1.0);
                    ++added_cuts;
                }
        // One greedy cardinality cover per row adds a stronger, still-valid
        // inequality when the largest coefficients alone exceed capacity.
        if (added_cuts >= cut_limit) break;
        f64 accum = 0.0;
        std::vector<Index> cover;
        for (const auto& [j, a] : terms) {
            accum += a;
            cover.push_back(j);
            if (accum > in.row_hi[sz(i)] + 1e-9) {
                const Index r = m + static_cast<Index>(added_cuts);
                for (const Index q : cover) {
                    rows.push_back(r); cols.push_back(q); vals.push_back(1.0);
                }
                out.row_lo.push_back(-model::kInf);
                out.row_hi.push_back(static_cast<f64>(cover.size() - 1));
                ++added_cuts;
                break;
            }
        }
    }
    if (added_cuts == 0) return out;
    out.A = sparse::from_triplets(m + static_cast<Index>(added_cuts), n,
                                  rows, cols, vals);
    out.row_names.resize(static_cast<std::size_t>(m + added_cuts));
    for (std::uint64_t q = 0; q < added_cuts; ++q)
        out.row_names[static_cast<std::size_t>(m + q)] =
            "COVER_" + std::to_string(q);
    return out;
}

// Equality-row propagation: a continuous column with coefficient +/-1 in an
// The search needs two things from a relaxation: a primal point that satisfies
// the rows (the residual tests) and a valid lower bound. The bound it uses is
// node_lp_bound_min(), which takes the independently computed dual objective
// -- a Lagrangian that prices every reduced cost at its worst bound, valid for
// any multipliers -- whenever it is below the primal one. So the primal-dual
// gap does not decide validity, only how tight that bound is. Holding it to
// the simplex's own 1e-9 rejected sound LPs whose last few reduced costs are
// ~1e-9 dual infeasible against wide column bounds: sp150x300d's round-4 cut
// LP (gap 8.6e-7, residuals 1e-9) ended the root cut loop at 60.4 instead of
// 66.4, and its node LPs were re-solved for nothing.
constexpr f64 kRelaxationGapTol = 1e-6;

bool relaxation_proved(const core::RawResult& r,
                       const engines::SimplexDiagnostics& d,
                       const engines::SimplexOptions& opts) {
    return r.proposed_status == core::Status::Optimal &&
           d.primal_residual <= opts.primal_feas_tol &&
           d.dual_residual <= opts.dual_feas_tol &&
           d.dual_bound_finite &&
           d.gap_rel <= std::max(opts.gap_tol, kRelaxationGapTol);
}

// The primal objective of a minimization LP is an upper estimate of its
// optimum, even when the final simplex residuals pass their tolerances. Search
// may use it for pseudocost learning, but a subtree bound must use the
// independently computed dual objective. Taking the smaller of the two also
// guards against roundoff that reverses their expected ordering. This remains
// binary64 evidence, not an exact proof certificate.
f64 node_lp_bound_min(const core::RawResult& r, f64 sense) {
    const f64 primal = sense * r.objective;
    const f64 dual = sense * r.dual_bound;
    if (!std::isfinite(primal) || !std::isfinite(dual))
        return -std::numeric_limits<f64>::infinity();
    return std::min(primal, dual);
}

// A status alone cannot close a branch-and-bound subtree. In particular,
// simplex presolve can return Infeasible before constructing a Farkas ray.
// Check a candidate ray against THIS node's actual LP, including its current
// bounds and any local rows. An unverified status is a retry, never a prune.
bool node_lp_infeasibility_proved_impl(const model::LpProblem& node_lp,
                                      const core::RawResult& raw,
                                      f64 tolerance,
                                      bool root_relaxation_bounded) {
    if (!node_lp_status_proves_infeasible(raw.proposed_status,
                                          root_relaxation_bounded))
        return false;
    // A crossed bound in the node LP proves infeasibility by itself.
    if (node_lp.find_empty_domain().index >= 0) return true;
    const auto& ray = !raw.dual_farkas_ray.multipliers.empty()
                          ? raw.dual_farkas_ray.multipliers : raw.ray;
    return !ray.empty() &&
           certify::check_dual_farkas_ray(node_lp, ray, tolerance).certified;
}

// Para-B&B worker expand (paper-shaped slice): prop → node LP → prune/branch.
// No heuristics / Mexi / SB inside workers - those stay on the serial path /
// sync. Determinism comes from barrier merge order, not from this body.
struct ParaExpandOut {
    bool abandoned = false;
    bool infeasible = false;
    bool integer_hit = false;
    f64 incumbent = core::kNaN;
    std::vector<f64> x;
    f64 bound = -std::numeric_limits<f64>::infinity();
    f64 abandoned_bound = std::numeric_limits<f64>::infinity();
    f64 pruned_bound = std::numeric_limits<f64>::infinity();
    std::vector<Node> children;
    std::uint64_t lp_solves = 0;
    std::uint64_t lp_iterations = 0;
    double lp_ms = 0.0;
    // Stable key for deterministic merge (parent pop order).
    std::uint64_t order_key = 0;
};

ParaExpandOut para_expand_node(const model::LpProblem& global_lp,
                               Node node,
                               const BabOptions& opts,
                               f64 sense,
                               f64 incumbent_min,
                               bool have_incumbent,
                               f64 int_tol,
                               f64 feas_tol,
                               std::uint64_t order_key,
                               double lp_seconds_left,
                               f64 obj_granularity, f64 offset_min) {
    ParaExpandOut out;
    out.order_key = order_key;
    out.abandoned_bound = node.bound;

    model::LpProblem node_lp = global_lp;
    if (opts.domain_propagation) {
        auto prop = propagate_bounds(node_lp, node.col_lo, node.col_hi,
                                     feas_tol, opts.propagation_max_rounds);
        if (!prop.feasible) {
            out.infeasible = true;
            return out;
        }
    }
    node_lp.col_lo = node.col_lo;
    node_lp.col_hi = node.col_hi;

    engines::SimplexOptions lp_opts = opts.lp;
    lp_opts.verbose = false;
    // opts.lp carries the WHOLE solver time limit, not what is left of it.
    // Unclamped, a worker LP starting near the deadline could run for another
    // full limit: sct2 overran by a constant ~22 s at both a 30 s and a 60 s
    // limit (51.4 s and 83.0 s wall) because of this one line. The serial node
    // loop already clamps the same way.
    if (lp_seconds_left > 0.0)
        lp_opts.time_limit_s = lp_seconds_left;
    if (lp_opts.max_iterations == 0) {
        const std::uint64_t work_size =
            static_cast<std::uint64_t>(node_lp.n_rows()) +
            static_cast<std::uint64_t>(node_lp.n_cols()) +
            static_cast<std::uint64_t>(node_lp.n_rows());
        lp_opts.max_iterations =
            std::max<std::uint64_t>(100000, 200ull * work_size);
    }
    engines::SimplexDiagnostics sd;
    engines::SimplexBasis basis;
    const auto t_lp = Clock::now();
    core::RawResult lp_raw;
    if (!node.has_basis) {
        lp_raw = engines::solve_simplex(node_lp, lp_opts, sd, &basis);
    } else {
        lp_opts.presolve = false;
        if (lp_opts.method == engines::SimplexMethod::Auto)
            lp_opts.method = engines::SimplexMethod::Dual;
        // S3.1: see the identical comment at the serial loop's warm-start
        // call in solve_milp.
        lp_opts.warm_dse_reset = true;
        lp_raw = engines::solve_dual_simplex(node_lp, lp_opts, sd, &basis,
                                             &node.basis);
    }
    out.lp_solves = 1;
    out.lp_iterations = sd.iterations;
    out.lp_ms = ms_since(t_lp);

    // This worker does not carry the root-relaxation proof. An ambiguous LP
    // status cannot close a subtree, so hand the node back to the serial
    // path, which can retry and account for it in the global bound ledger.
    if (node_lp_infeasibility_proved(node_lp, lp_raw,
                                     opts.primal_feas_tol, false)) {
        out.infeasible = true;
        return out;
    }
    if (lp_raw.proposed_status == core::Status::InfeasibleOrUnbounded ||
        lp_raw.proposed_status == core::Status::Infeasible) {
        out.abandoned = true;
        return out;
    }
    if (lp_raw.proposed_status == core::Status::Interrupted) {
        out.abandoned = true;
        return out;
    }
    const bool proved = relaxation_proved(lp_raw, sd, lp_opts);
    if (!proved) {
        out.abandoned = true;
        return out;
    }
    const f64 lp_obj_min = sense * lp_raw.objective;
    // Never below the bound the node inherited (see node_bound_after_relaxation).
    out.bound = std::max(node.bound, node_lp_bound_min(lp_raw, sense));
    if (have_incumbent &&
        out.bound >= node_cutoff(incumbent_min, obj_granularity, offset_min,
                                 opts.gap_tol, opts.abs_gap_tol)) {
        if (out.bound < incumbent_min) out.pruned_bound = out.bound;
        return out;  // bound prune, no children
    }

    // Integer feasible?
    bool all_int = true;
    for (Index j = 0; j < node_lp.n_cols(); ++j) {
        if (node_lp.is_integer.empty() || !node_lp.is_integer[sz(j)]) continue;
        if (!is_integral(lp_raw.x[sz(j)], int_tol)) {
            all_int = false;
            break;
        }
    }
    if (all_int &&
        node_lp.max_row_violation(lp_raw.x) <= feas_tol &&
        node_lp.max_bound_violation(lp_raw.x) <= feas_tol) {
        out.integer_hit = true;
        out.incumbent = lp_raw.objective;
        out.x = lp_raw.x;
        return out;
    }

    const Index br = pick_branch_var(node_lp, node.col_lo, node.col_hi,
                                     lp_raw.x, int_tol);
    if (br < 0) return out;
    const f64 xv = lp_raw.x[sz(br)];
    const f64 floor_v = std::floor(xv);
    const f64 ceil_v = std::ceil(xv);

    auto make_child = [&](int dir) {
        Node c = node;
        c.prop_valid = false;   // worker path propagates from scratch
        c.dirty.clear();
        c.depth = node.depth + 1;
        c.parent_branch_var = br;
        c.parent_branch_dir = dir;
        c.parent_bound = lp_obj_min;
        c.pc_consumed = false;
        c.pc_recorded_unit = core::kNaN;
        c.parent_branch_distance =
            dir < 0 ? (xv - floor_v) : (ceil_v - xv);
        c.bound = out.bound;
        c.basis = basis;
        c.has_basis = true;
        c.plunge_len = 0;
        // Mirror the serial branch's prop_trail.push (see the down/up nodes
        // below the main tableau loop): a nogood learned from this child's
        // trail must include its own branching decision, or the clause is
        // missing a literal and over-generalizes, pruning sibling subtrees
        // that were never actually shown infeasible.
        if (dir < 0) {
            const f64 old_hi = c.col_hi[sz(br)];
            c.col_hi[sz(br)] = std::min(old_hi, floor_v);
            c.prop_trail.push(br, BoundDir::Upper, c.col_hi[sz(br)], old_hi,
                              ReasonKind::Branch, -1, c.depth);
        } else {
            const f64 old_lo = c.col_lo[sz(br)];
            c.col_lo[sz(br)] = std::max(old_lo, ceil_v);
            c.prop_trail.push(br, BoundDir::Lower, c.col_lo[sz(br)], old_lo,
                              ReasonKind::Branch, -1, c.depth);
        }
        return c;
    };
    out.children.push_back(make_child(-1));
    out.children.push_back(make_child(+1));
    return out;
}

}  // namespace

bool node_lp_infeasibility_proved(const model::LpProblem& node_lp,
                                 const core::RawResult& raw, f64 tolerance,
                                 bool root_relaxation_bounded) {
    return node_lp_infeasibility_proved_impl(
        node_lp, raw, tolerance, root_relaxation_bounded);
}

CertifiedRcStats certified_reduced_cost_bounds(
    const model::LpProblem& p, const std::vector<f64>& y, f64 incumbent,
    std::vector<f64>& lo, std::vector<f64>& hi, f64 tolerance) {
    CertifiedRcStats stats;
    const auto start = Clock::now();
    const Index n = p.n_cols();
    if (y.size() != sz(p.n_rows()) || lo.size() != sz(n) ||
        hi.size() != sz(n) || p.is_integer.size() != sz(n) ||
        !std::isfinite(incumbent)) return stats;
    const f64 sign = p.maximize ? -1.0 : 1.0;
    const f64 inc_min = sign * incumbent;
    const f64 margin = std::max(0.0, tolerance) * (1.0 + std::fabs(inc_min));
    std::vector<f64> ymin(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) ymin[i] = sign * y[i];
    const auto changes = certify::safe_reduced_cost_tightenings(
        p, ymin, lo, hi, inc_min + margin, nullptr, &stats.checked);
    // Turning an INFINITE bound into a finite one is only worth it when the
    // new bound is a sane number. A reduced cost of 1e-6 against a gap of 2e7
    // "tightens" x <= inf to x <= 2e13: useless as a bound, and every later
    // dual objective then charges roundoff-sized reduced costs (2.5e-14)
    // 2e13 each -- about 0.6 per column -- so LPs that are optimal stop
    // passing the primal-dual gap test (nu25-pr12: every node LP unproved,
    // 130 of 134 fell back to a cold primal solve).
    constexpr f64 kMaxNewFiniteBound = 1e6;
    for (const auto& c : changes) {
        const auto j = sz(c.col);
        const f64 old_bound = c.upper ? hi[j] : lo[j];
        if (!std::isfinite(old_bound) && std::fabs(c.value) > kMaxNewFiniteBound)
            continue;
        if (c.upper) {
            if (!(c.value < hi[j])) continue;
            hi[j] = c.value;
        } else {
            if (!(c.value > lo[j])) continue;
            lo[j] = c.value;
        }
        ++stats.tightened;
        if (lo[j] == hi[j]) ++stats.fixed;
    }
    stats.ms = ms_since(start);
    return stats;
}

// Maps a restart payload's pseudocosts and clauses through the presolve that
// re-reduced the restarted model (`stage` -> `pre2.reduced`). Cut rows are not
// mapped: they are added to the stage model BEFORE that presolve. Anything
// that has no exact image is dropped, and counted in `dropped`:
//  - with duplicate-column merging in the map every clause is dropped (a merged
//    column stands for a sum, so its literals mean something else);
//  - a fixed column: a literal it satisfies makes the clause redundant, a
//    literal it violates just leaves the clause;
//  - a binary-substituted column follows its representative, complemented if
//    the relation says so (0/1 literals only).
void map_restart_payload(const RestartPayload& in, const MilpPresolveResult& pre2,
                         const model::LpProblem& stage, RestartPayload& out,
                         std::uint64_t& dropped, bool& box_empty) {
    box_empty = false;
    const Index nn = pre2.reduced.n_cols();
    out.pseudocosts.down_sum.assign(sz(nn), 0.0);
    out.pseudocosts.up_sum.assign(sz(nn), 0.0);
    out.pseudocosts.down_count.assign(sz(nn), 0);
    out.pseudocosts.up_count.assign(sz(nn), 0);
    const Index ns = stage.n_cols();
    // A merged column stands for a SUM of its members: statistics and literals
    // about any member (or the representative) mean something else afterwards.
    // Only those columns lose their history; everything else keeps it.
    std::vector<char> affected(sz(ns), 0);
    for (const auto& mg : pre2.column_merges)
        for (const Index j : mg.members)
            if (j >= 0 && j < ns) affected[sz(j)] = 1;
    if (in.pseudocosts.down_sum.size() == sz(ns))
        for (Index j = 0; j < ns; ++j) {
            if (affected[sz(j)]) continue;
            const Index rj = pre2.reduced_col[sz(j)];
            if (rj < 0) continue;
            out.pseudocosts.down_sum[sz(rj)] = in.pseudocosts.down_sum[sz(j)];
            out.pseudocosts.up_sum[sz(rj)] = in.pseudocosts.up_sum[sz(j)];
            out.pseudocosts.down_count[sz(rj)] = in.pseudocosts.down_count[sz(j)];
            out.pseudocosts.up_count[sz(rj)] = in.pseudocosts.up_count[sz(j)];
        }
    std::vector<const BinaryRelation*> step_of(sz(ns), nullptr);
    for (const auto& st : pre2.binary_substitution_steps)
        if (st.first >= 0 && st.first < ns) step_of[sz(st.first)] = &st;
    for (const auto& clause : in.clauses) {
        std::vector<ConflictLiteral> lits;
        bool satisfied = false, ok = true;
        for (const auto& l : clause) {
            if (l.var < 0 || l.var >= ns || affected[sz(l.var)]) { ok = false; break; }
            const Index rj = pre2.reduced_col[sz(l.var)];
            if (rj >= 0) {
                lits.push_back({rj, l.upper, l.bound});
                continue;
            }
            if (step_of[sz(l.var)] != nullptr) {
                const BinaryRelation& st = *step_of[sz(l.var)];
                if (st.second < 0 || st.second >= ns || affected[sz(st.second)]) { ok = false; break; }
                const Index rr = pre2.reduced_col[sz(st.second)];
                const bool binary_literal = (l.upper && l.bound == 0.0) || (!l.upper && l.bound == 1.0);
                if (rr < 0 || !binary_literal) { ok = false; break; }
                // x_j <= 0  ==  x_r <= 0 (or x_r >= 1 if complemented); x_j >= 1 the reverse.
                const bool x_is_zero = l.upper;
                const bool r_is_zero = st.complement ? !x_is_zero : x_is_zero;
                lits.push_back({rr, r_is_zero, r_is_zero ? 0.0 : 1.0});
                continue;
            }
            // Fixed by the presolve.
            const f64 v = pre2.fixed_value[sz(l.var)];
            const bool holds = l.upper ? v <= l.bound + 1e-9 : v >= l.bound - 1e-9;
            if (holds) { satisfied = true; break; }
        }
        if (!ok) { ++dropped; continue; }
        if (satisfied) continue;
        // Every literal falsified by a fixed value: the clause is violated, so
        // no point of the restarted box beats the cutoff it was learned under.
        // (Silently skipping it, as an empty "redundant" clause, lost that.)
        if (lits.empty()) { if (!clause.empty()) box_empty = true; continue; }
        out.clauses.push_back(std::move(lits));
    }
}

// Independent-component solving. Columns are connected when they share a row;
// with two or more components that own rows the model is a sum of separate
// problems, so each is solved on its own (sequentially, each under a share of
// what is left of the deadline, smallest first) and the results are added.
// Optimal needs every component proved; an infeasible component proves the
// whole model infeasible; the combined bound is the sum of the component
// bounds. Columns in no row are settled directly from their cost. Returns
// false, with nothing done, when the model does not split or a case is not
// handled (foreign cutoff, an unbounded isolated column, an empty row that
// cannot hold), so the ordinary solve runs instead.
bool solve_milp_by_components(const model::LpProblem& problem,
                              const BabOptions& opts, BabDiagnostics& diag,
                              core::RawResult& out) {
    const auto tc0 = Clock::now();
    const Index n = problem.n_cols();
    const Index m = problem.n_rows();
    if (n < 2 || m < 2) return false;
    if (std::isfinite(opts.initial_cutoff)) return false;
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    const auto& av = problem.A.vals;

    std::vector<Index> parent(sz(n));
    for (Index j = 0; j < n; ++j) parent[sz(j)] = j;
    const auto find = [&](Index a) {
        while (parent[sz(a)] != a) { parent[sz(a)] = parent[sz(parent[sz(a)])]; a = parent[sz(a)]; }
        return a;
    };
    std::vector<char> in_row(sz(n), 0);
    for (Index i = 0; i < m; ++i) {
        Index first = -1;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (av[sz(k)] == 0.0) continue;
            const Index j = ci[sz(k)];
            in_row[sz(j)] = 1;
            if (first < 0) first = j;
            else parent[sz(find(j))] = find(first);
        }
        if (first < 0) {   // an empty row must hold on its own
            if (problem.row_lo[sz(i)] > 1e-9 || problem.row_hi[sz(i)] < -1e-9) return false;
        }
    }
    std::vector<Index> comp_of(sz(n), -1);
    std::vector<std::vector<Index>> cols;
    for (Index j = 0; j < n; ++j) {
        if (!in_row[sz(j)]) continue;
        const Index r = find(j);
        if (comp_of[sz(r)] < 0) { comp_of[sz(r)] = static_cast<Index>(cols.size()); cols.emplace_back(); }
        cols[sz(comp_of[sz(r)])].push_back(j);
    }
    if (cols.size() < 2) return false;
    std::vector<std::vector<Index>> rows(cols.size());
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            if (av[sz(k)] != 0.0) {
                rows[sz(comp_of[sz(find(ci[sz(k)]))])].push_back(i);
                break;
            }

    // Columns in no row.
    const f64 sense = problem.maximize ? -1.0 : 1.0;
    std::vector<f64> x(sz(n), 0.0);
    f64 obj = problem.obj_offset;
    Index isolated = 0;
    for (Index j = 0; j < n; ++j) {
        if (in_row[sz(j)]) continue;
        f64 lo = problem.col_lo[sz(j)], hi = problem.col_hi[sz(j)];
        const bool is_int = !problem.is_integer.empty() && problem.is_integer[sz(j)];
        if (is_int) { lo = std::ceil(lo - 1e-9); hi = std::floor(hi + 1e-9); }
        if (lo > hi) return false;
        const f64 cc = sense * problem.c[sz(j)];
        f64 v;
        if (cc > 0.0) v = lo;
        else if (cc < 0.0) v = hi;
        else v = std::min(std::max(0.0, lo), hi);
        if (!std::isfinite(v)) return false;   // unbounded: leave it to the real solve
        x[sz(j)] = v;
        obj += problem.c[sz(j)] * v;
        ++isolated;
    }

    // Order: smallest first, so a quick component frees time for the rest.
    std::vector<std::size_t> order(cols.size());
    for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return cols[a].size() < cols[b].size();
    });
    const auto weight = [&](std::size_t k) {
        f64 w = 1.0;
        for (const Index j : cols[k])
            if (!problem.is_integer.empty() && problem.is_integer[sz(j)]) w += 1.0;
        return w;
    };
    f64 weight_left = 0.0;
    for (const auto k : order) weight_left += weight(k);

    f64 bound = problem.obj_offset;
    for (Index j = 0; j < n; ++j)
        if (!in_row[sz(j)]) bound += problem.c[sz(j)] * x[sz(j)];
    bool all_optimal = true, all_points = true, bound_finite = true, infeasible = false;
    core::RawResult infeasible_raw;
    BabDiagnostics total;
    std::uint64_t proved_components = 0;
    for (const std::size_t k : order) {
        // Sub-model.
        model::LpProblem sub;
        sub.name = problem.name + "_comp";
        sub.maximize = problem.maximize;
        sub.obj_offset = 0.0;
        std::vector<Index> local(sz(n), -1);
        for (std::size_t q = 0; q < cols[k].size(); ++q) local[sz(cols[k][q])] = static_cast<Index>(q);
        const Index sn = static_cast<Index>(cols[k].size());
        const Index sm = static_cast<Index>(rows[k].size());
        sub.c.resize(sz(sn)); sub.col_lo.resize(sz(sn)); sub.col_hi.resize(sz(sn));
        sub.is_integer.assign(sz(sn), false);
        for (Index q = 0; q < sn; ++q) {
            const Index j = cols[k][sz(q)];
            sub.c[sz(q)] = problem.c[sz(j)];
            sub.col_lo[sz(q)] = problem.col_lo[sz(j)];
            sub.col_hi[sz(q)] = problem.col_hi[sz(j)];
            if (!problem.is_integer.empty()) sub.is_integer[sz(q)] = problem.is_integer[sz(j)];
        }
        std::vector<Index> tr, tc;
        std::vector<f64> tv;
        for (Index r = 0; r < sm; ++r) {
            const Index i = rows[k][sz(r)];
            sub.row_lo.push_back(problem.row_lo[sz(i)]);
            sub.row_hi.push_back(problem.row_hi[sz(i)]);
            for (core::Offset e = rp[sz(i)]; e < rp[sz(i) + 1]; ++e)
                if (av[sz(e)] != 0.0) { tr.push_back(r); tc.push_back(local[sz(ci[sz(e)])]); tv.push_back(av[sz(e)]); }
        }
        sub.A = sparse::from_triplets(sm, sn, tr, tc, tv);

        BabOptions so = opts;
        // The parent's tolerance has to cover the SUM of the children's gaps,
        // and a relative tolerance does not add over components (the parent's
        // objective can be far smaller than theirs once offsets and signs
        // cancel). Each child therefore gets 1/K of both allowances; the
        // parent re-checks the summed gap itself before claiming Optimal.
        const f64 share = 1.0 / static_cast<f64>(cols.size());
        so.gap_tol = opts.gap_tol * share;
        so.abs_gap_tol = opts.abs_gap_tol * share;
        so.component_child = true;
        so.component_solve = false;
        so.root_restart_allowed = false;
        so.pool = nullptr;
        so.cut_reference_point = nullptr;
        if (opts.time_limit_s > 0.0) {
            const double left = opts.time_limit_s - ms_since(tc0) / 1000.0;
            if (left <= 0.0) { all_optimal = false; all_points = false; bound_finite = false; continue; }
            so.time_limit_s = std::max(0.01, left * weight(k) / weight_left);
            if (so.lp.time_limit_s > 0.0) so.lp.time_limit_s = std::min(so.lp.time_limit_s, so.time_limit_s);
        }
        weight_left -= weight(k);
        BabDiagnostics sd;
        core::RawResult r = solve_milp(sub, so, sd);
        total.nodes += sd.nodes;
        total.lp_solves += sd.lp_solves;
        total.lp_iterations += sd.lp_iterations;
        total.lp_ms += sd.lp_ms;
        total.cut_rounds += sd.cut_rounds;
        total.invalid_cuts_detected += sd.invalid_cuts_detected;
        total.node_cuts_invalid_generated += sd.node_cuts_invalid_generated;
        total.node_cuts_invalid_rejected += sd.node_cuts_invalid_rejected;
        total.node_cuts_invalid_inserted += sd.node_cuts_invalid_inserted;
        total.node_cuts_ref_outside_node += sd.node_cuts_ref_outside_node;
        if (r.proposed_status == core::Status::Infeasible && sd.globally_proved) {
            infeasible = true;
            infeasible_raw = std::move(r);
            break;
        }
        const bool has_point = !r.x.empty() && static_cast<Index>(r.x.size()) == sn &&
                               std::isfinite(r.objective);
        if (r.proposed_status == core::Status::Optimal && sd.globally_proved && has_point) ++proved_components;
        else all_optimal = false;
        if (has_point) {
            for (Index q = 0; q < sn; ++q) x[sz(cols[k][sz(q)])] = r.x[sz(q)];
            obj += r.objective;
        } else {
            all_points = false;
        }
        f64 b = std::isfinite(sd.dual_bound) ? sd.dual_bound : r.dual_bound;
        if (!std::isfinite(b) && r.proposed_status == core::Status::Optimal &&
            sd.globally_proved && has_point) {
            // An exhausted tree pruned against incumbent minus the child's
            // tolerance: that, not the incumbent, is what it certified.
            const f64 allow = std::max(so.abs_gap_tol, so.gap_tol * (1.0 + std::fabs(r.objective)));
            b = problem.maximize ? r.objective + allow : r.objective - allow;
        }
        if (std::isfinite(b)) bound += b;
        else bound_finite = false;
    }

    diag = BabDiagnostics{};
    diag.policy_used = opts.policy;
    diag.nodes = total.nodes;
    diag.lp_solves = total.lp_solves;
    diag.lp_iterations = total.lp_iterations;
    diag.lp_ms = total.lp_ms;
    diag.cut_rounds = total.cut_rounds;
    diag.invalid_cuts_detected = total.invalid_cuts_detected;
    diag.node_cuts_invalid_generated = total.node_cuts_invalid_generated;
    diag.node_cuts_invalid_rejected = total.node_cuts_invalid_rejected;
    diag.node_cuts_invalid_inserted = total.node_cuts_invalid_inserted;
    diag.node_cuts_ref_outside_node = total.node_cuts_ref_outside_node;
    diag.component_count = cols.size();
    diag.component_isolated = static_cast<std::uint64_t>(isolated);
    diag.components_solved = proved_components;
    out = core::RawResult{};
    out.engine = "milp_bab";
    out.backend = "cpu";
    if (infeasible) {
        diag.globally_proved = true;
        diag.termination_reason = "a component is infeasible";
        out.proposed_status = core::Status::Infeasible;
        out.proposed_level = infeasible_raw.proposed_level;
        out.termination_reason = diag.termination_reason;
        diag.component_solve_ms = diag.total_ms = ms_since(tc0);
        return true;
    }
    if (all_points) {
        diag.final_primal_violation = milp_point_max_violation(problem, x, opts.int_tol);
        if (!(diag.final_primal_violation <= opts.primal_feas_tol)) all_points = false;
    }
    if (all_points) {
        out.x = x;
        out.objective = obj;
        diag.incumbent = obj;
    }
    const f64 sentinel = problem.maximize ? core::kPosInf : -core::kPosInf;
    out.dual_bound = bound_finite ? bound : sentinel;
    diag.dual_bound = out.dual_bound;
    if (all_points && bound_finite)
        diag.gap_rel = std::fabs(obj - bound) / (1.0 + std::fabs(obj));
    // Optimal only if every component is proved AND the summed certified gap
    // is within the PARENT's own tolerance, judged on the parent's objective.
    const f64 total_gap = all_points && bound_finite ? std::fabs(obj - bound) : core::kPosInf;
    const bool crosses = all_points && bound_finite &&
        (problem.maximize ? bound < obj - opts.gap_tol * (1.0 + std::fabs(obj))
                          : bound > obj + opts.gap_tol * (1.0 + std::fabs(obj)));
    const bool gap_ok = total_gap <= opts.abs_gap_tol ||
                        (all_points && bound_finite &&
                         diag.gap_rel <= opts.gap_tol);
    const bool proved = all_optimal && all_points && bound_finite && gap_ok && !crosses;
    diag.globally_proved = proved;
    if (proved) {
        out.proposed_status = core::Status::Optimal;
        out.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
        diag.termination_reason = "all independent components solved to optimality";
        // The certified bound stays what it is: the sum of the components'.
    } else if (all_points) {
        out.proposed_status = core::Status::Feasible;
        out.proposed_level = core::ProofLevel::FeasibleWithGap;
        diag.termination_reason = all_optimal
            ? "independent components: summed gap exceeds the parent tolerance"
            : "independent components: not all proved";
    } else {
        out.proposed_status = core::Status::Interrupted;
        out.proposed_level = core::ProofLevel::None;
        diag.termination_reason = "independent components: no point for every component";
    }
    out.termination_reason = diag.termination_reason;
    diag.component_solve_ms = diag.total_ms = ms_since(tc0);
    return true;
}

core::RawResult solve_milp(const model::LpProblem& problem,
                           const BabOptions& opts,
                           BabDiagnostics& diag) {
    // Structural presolve (Phase 8): a thin wrapper around the rest of this
    // function rather than a change threaded through it, so every existing
    // code path below is completely unaffected when it's off (the default).
    // Solve the REDUCED problem recursively (with presolve disabled on the
    // inner call, or this would recurse forever), then postsolve the point
    // back into the ORIGINAL column space and re-validate it against
    // `problem` -- not the reduced copy the inner call already checked
    // itself against -- before it is ever handed back to the caller. A
    // presolve/postsolve bug is exactly the kind of defect this second,
    // independent check is meant to catch: see milp_point_max_violation / A3.
    if (opts.structural_presolve.enabled) {
        const auto tpre0 = Clock::now();
        MilpPresolveStats pstats;
        // Presolve probing gets a bounded share of the budget.
        MilpPresolveOptions sp_opts = opts.structural_presolve;
        if (opts.time_limit_s > 0.0)
            sp_opts.probing_presolve_time_s =
                std::min(sp_opts.probing_presolve_time_s, 0.06 * opts.time_limit_s);
        if (!opts.probing) sp_opts.probing_presolve = false;
        MilpPresolveResult pre =
            run_structural_presolve(problem, sp_opts, pstats);
        if (pre.infeasible) {
            diag = BabDiagnostics{};
            diag.policy_used = opts.policy;
            diag.structural_presolve = pstats;
            diag.structural_presolve_applied = true;
            diag.globally_proved = true;
            diag.total_ms = ms_since(tpre0);
            core::RawResult raw;
            raw.engine = "milp";
            raw.backend = "cpu";
            raw.proposed_status = core::Status::Infeasible;
            raw.proposed_level = core::ProofLevel::BoundOnly;
            raw.termination_reason = "root structural presolve proved infeasible";
            diag.termination_reason = raw.termination_reason;
            return raw;
        }
        BabOptions inner_opts = opts;
        inner_opts.structural_presolve.enabled = false;
        // Shared pool points belong to the original model. Until the pool
        // supports postsolve, reduced solves must neither read nor publish it.
        inner_opts.pool = nullptr;
        // A parent's pseudocosts follow their columns into the reduced space.
        PseudocostSeed reduced_seed;
        inner_opts.pseudocost_seed = nullptr;
        if (opts.pseudocost_seed != nullptr &&
            opts.pseudocost_seed->down_sum.size() == sz(problem.n_cols()) &&
            opts.pseudocost_seed->up_sum.size() == sz(problem.n_cols()) &&
            opts.pseudocost_seed->down_count.size() == sz(problem.n_cols()) &&
            opts.pseudocost_seed->up_count.size() == sz(problem.n_cols())) {
            const auto nr = sz(pre.reduced.n_cols());
            reduced_seed.down_sum.assign(nr, 0.0);
            reduced_seed.up_sum.assign(nr, 0.0);
            reduced_seed.down_count.assign(nr, 0);
            reduced_seed.up_count.assign(nr, 0);
            for (Index j = 0; j < problem.n_cols(); ++j) {
                const Index rj = pre.reduced_col[sz(j)];
                if (rj < 0) continue;
                reduced_seed.down_sum[sz(rj)] = opts.pseudocost_seed->down_sum[sz(j)];
                reduced_seed.up_sum[sz(rj)] = opts.pseudocost_seed->up_sum[sz(j)];
                reduced_seed.down_count[sz(rj)] = opts.pseudocost_seed->down_count[sz(j)];
                reduced_seed.up_count[sz(rj)] = opts.pseudocost_seed->up_count[sz(j)];
            }
            inner_opts.pseudocost_seed = &reduced_seed;
        }
        std::vector<f64> reduced_reference;
        inner_opts.cut_reference_point = nullptr;
        // A merged column stands for a sum of members: a reference point has no
        // single value for it, so the debug check abstains.
        if (opts.cut_reference_point != nullptr && pre.column_merges.empty() &&
            static_cast<Index>(opts.cut_reference_point->size()) == problem.n_cols()) {
            reduced_reference.resize(sz(pre.reduced.n_cols()));
            for (Index j = 0; j < problem.n_cols(); ++j)
                if (pre.reduced_col[sz(j)] >= 0)
                    reduced_reference[sz(pre.reduced_col[sz(j)])] =
                        (*opts.cut_reference_point)[sz(j)];
            inner_opts.cut_reference_point = &reduced_reference;
        }
        const double presolve_s =
            std::chrono::duration<double>(Clock::now() - tpre0).count();
        const bool overall_expired =
            opts.time_limit_s > 0.0 && presolve_s >= opts.time_limit_s;
        const bool lp_expired =
            opts.lp.time_limit_s > 0.0 && presolve_s >= opts.lp.time_limit_s;
        if (overall_expired || lp_expired) {
            diag = BabDiagnostics{};
            diag.policy_used = opts.policy;
            diag.structural_presolve = pstats;
            diag.structural_presolve_applied = true;
            diag.total_ms = ms_since(tpre0);
            core::RawResult raw;
            raw.engine = "milp";
            raw.backend = "cpu";
            raw.proposed_status = core::Status::Interrupted;
            raw.termination_reason = overall_expired
                ? "time limit exhausted during structural presolve"
                : "LP time limit exhausted during structural presolve";
            diag.termination_reason = raw.termination_reason;
            return raw;
        }
        if (opts.time_limit_s > 0.0)
            inner_opts.time_limit_s = opts.time_limit_s - presolve_s;
        if (opts.lp.time_limit_s > 0.0)
            inner_opts.lp.time_limit_s = opts.lp.time_limit_s - presolve_s;
        if (inner_opts.time_limit_s > 0.0)
            inner_opts.lp.time_limit_s = inner_opts.lp.time_limit_s > 0.0
                ? std::min(inner_opts.lp.time_limit_s, inner_opts.time_limit_s)
                : inner_opts.time_limit_s;
        // Probing already spent part of the allowance in presolve: the inner
        // root probing gets what is left, not a second full share.
        if (pstats.probing_ms > 0.0)
            inner_opts.probe.probe_time_limit_s = std::max(
                0.3, opts.probe.probe_time_limit_s - pstats.probing_ms / 1000.0);
        inner_opts.root_restart_allowed =
            opts.root_restart && opts.root_restart_max > 0;
        inner_opts.probing_carry = opts.carry_probing && pre.probing_carry.fingerprint != 0
                                       ? &pre.probing_carry : nullptr;
        core::RawResult raw = solve_milp(pre.reduced, inner_opts, diag);

        // Root restarts. The inner solve asks for one after reduced-cost
        // fixing has shrunk the root box: presolve the tightened reduced
        // model again (physically smaller node LPs), and re-solve with the
        // best point so far as a cutoff. The best point is kept in ORIGINAL
        // space: reduced-cost fixing may exclude the incumbent itself, so it
        // need not be feasible in the restarted box.
        std::vector<f64> kept_x;
        f64 kept_obj = std::numeric_limits<f64>::quiet_NaN();
        const auto keep_if_better = [&](const std::vector<f64>& x_red) {
            if (x_red.size() != sz(pre.reduced.n_cols())) return;
            std::vector<f64> xo = postsolve_point(pre, x_red);
            if (!(milp_point_max_violation(problem, xo, opts.int_tol) <=
                  opts.primal_feas_tol))
                return;
            const f64 obj = problem.objective(xo);
            if (!std::isfinite(obj)) return;
            if (kept_x.empty() ||
                (problem.maximize ? obj > kept_obj : obj < kept_obj)) {
                kept_x = std::move(xo);
                kept_obj = obj;
            }
        };
        // The strongest bound certified so far, in ORIGINAL space (NaN: none).
        // Stage 0 has no cutoff, so its bound holds for every feasible point.
        // A restarted stage only searched the box reduced-cost fixing left
        // and pruned against a cutoff, so what it certifies is
        //     optimum >= min(stage bound, that cutoff)   (minimisation).
        // An unfinished stage with no finite bound certifies nothing, and a
        // bound never becomes the incumbent's value by default.
        f64 certified_bound = std::numeric_limits<f64>::quiet_NaN();
        const auto fold_stage_bound = [&](f64 stage_bound, f64 stage_cutoff) {
            certified_bound = restarted_stage_bound(problem.maximize, certified_bound,
                                                    stage_bound, stage_cutoff);
        };
        f64 stage_cutoff_used = std::numeric_limits<f64>::quiet_NaN();
        std::uint64_t restarts_done = 0;
        BabDiagnostics carried;  // work of the stages already finished
        while (diag.restart_requested &&
               restarts_done < static_cast<std::uint64_t>(opts.root_restart_max)) {
            keep_if_better(raw.x);
            if (kept_x.empty()) break;
            fold_stage_bound(diag.dual_bound, stage_cutoff_used);
            const Index cols_before = pre.reduced.n_cols();
            const Index rows_before = pre.reduced.n_rows();
            model::LpProblem next = pre.reduced;
            next.col_lo = diag.restart_col_lo;
            next.col_hi = diag.restart_col_hi;
            // The stage's state, taken now: `diag` is overwritten by the next
            // solve. Its global cut rows join the model before it is presolved
            // again (they only remove points no better than the cutoff, which
            // the restart excludes anyway).
            RestartPayload payload = std::move(diag.restart_payload);
            const model::LpProblem stage_model = next;
            if (opts.restart_carry_state && !payload.cuts.empty()) {
                apply_cuts_inplace(next, payload.cuts, opts.cut);
                carried.restart_cut_rows_carried += payload.cuts.size();
            }
            MilpPresolveStats p2;
            MilpPresolveResult pre2 =
                run_structural_presolve(next, [&] { auto o2 = opts.structural_presolve; o2.probing_presolve = false; return o2; }(), p2);
            ++restarts_done;
            carried.nodes += diag.nodes;
            carried.lp_solves += diag.lp_solves;
            carried.lp_iterations += diag.lp_iterations;
            carried.lp_ms += diag.lp_ms;
            carried.cut_rounds += diag.cut_rounds;
            carried.cut_loop_ms += diag.cut_loop_ms;
            carried.rc_columns_fixed += diag.rc_columns_fixed;
            carried.rc_bounds_tightened += diag.rc_bounds_tightened;
            carried.fpump_attempts += diag.fpump_attempts;
            carried.fpump_hits += diag.fpump_hits;
            carried.fpump_rounds += diag.fpump_rounds;
            carried.fpump_lp_solves += diag.fpump_lp_solves;
            carried.fpump_ms += diag.fpump_ms;
            const std::uint64_t fixed_at_request = diag.restart_columns_fixed;
            if (opts.verbose)
                std::printf("  [milp] root restart #%llu: %llu integer columns fixed; "
                            "re-presolve %d x %d -> %d x %d\n",
                            static_cast<unsigned long long>(restarts_done),
                            static_cast<unsigned long long>(fixed_at_request),
                            rows_before, cols_before,
                            pre2.infeasible ? 0 : pre2.reduced.n_rows(),
                            pre2.infeasible ? 0 : pre2.reduced.n_cols());
            RestartPayload mapped;
            bool mapped_box_empty = false;
            if (opts.restart_carry_state && !pre2.infeasible) {
                std::uint64_t dropped = 0;
                map_restart_payload(payload, pre2, stage_model, mapped, dropped,
                                    mapped_box_empty);
                carried.restart_clauses_carried += mapped.clauses.size();
                carried.restart_clauses_dropped += dropped;
                if (mapped_box_empty) ++carried.restart_clauses_emptied;
            }
            if (pre2.infeasible || mapped_box_empty) {
                // No point better than the kept one exists in the box that
                // reduced-cost fixing left (presolve emptied it, or a learned
                // clause has every literal falsified): the kept point is optimal.
                diag = BabDiagnostics{};
                diag.policy_used = opts.policy;
                diag.structural_presolve = pstats;
                diag.structural_presolve_applied = true;
                diag.root_restarts = restarts_done;
                diag.restart_columns_fixed = fixed_at_request;
                diag.nodes = carried.nodes;
                diag.lp_solves = carried.lp_solves;
                diag.lp_iterations = carried.lp_iterations;
                diag.incumbent = kept_obj;
                diag.dual_bound = kept_obj;
                diag.gap_rel = 0.0;
                diag.globally_proved = true;
                diag.final_primal_violation =
                    milp_point_max_violation(problem, kept_x, opts.int_tol);
                diag.total_ms = ms_since(tpre0);
                diag.termination_reason =
                    "root restart: no better point exists after reduced-cost fixing";
                core::RawResult done;
                done.engine = "milp_bab";
                done.backend = "cpu";
                done.x = kept_x;
                done.objective = kept_obj;
                done.dual_bound = kept_obj;
                done.termination_reason = diag.termination_reason;
                done.proposed_status = core::Status::Optimal;
                done.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
                return done;
            }
            const Index rows_removed = next.n_rows() - pre2.reduced.n_rows();
            const Index cols_removed = cols_before - pre2.reduced.n_cols();
            if (restarts_done > 1 && rows_removed <= 0 && cols_removed <= 0) {
                // A repeat restart must buy something: the last one removed
                // nothing, so the box it tightened is not worth a cold start.
                diag.restart_requested = false;
                --restarts_done;
                break;
            }
            BabOptions again = inner_opts;
            again.initial_cutoff = kept_obj;
            again.root_restart_allowed =
                restarts_done < static_cast<std::uint64_t>(opts.root_restart_max);
            const double used_s = ms_since(tpre0) / 1000.0;
            if (opts.time_limit_s > 0.0) {
                again.time_limit_s = opts.time_limit_s - used_s;
                if (!(again.time_limit_s > 0.0)) {
                    // Nothing new starts: `pre` and `raw` still describe the
                    // same (old) stage. The map is replaced only once the new
                    // stage is certain to run.
                    diag.restart_requested = false;
                    --restarts_done;
                    break;
                }
            }
            if (opts.restart_carry_state) {
                again.pseudocost_seed = &mapped.pseudocosts;
                again.initial_clauses = &mapped.clauses;
            }
            compose_presolve(pre, std::move(pre2));
            // The kept point, mapped into the restarted stage: it seeds the new
            // incumbent (so heuristics have a centre) when it survived the
            // reduced-cost box; solve_milp validates it against that stage.
            std::vector<f64> kept_reduced;
            if (opts.restart_carry_state && kept_x.size() == pre.reduced_col.size()) {
                // Forward map of the kept point. A merged column carries the SUM of
                // its members' values (merges are applied in the recorded order, so
                // a representative that was itself merged again sums its group);
                // the restarted stage validates the result, so a mapping that is
                // wrong for any reason costs nothing but the hint.
                std::vector<f64> val = kept_x;
                for (const auto& mg : pre.column_merges) {
                    Index rep = -1;
                    f64 total = 0.0;
                    for (const Index m : mg.members) {
                        if (m < 0 || m >= static_cast<Index>(val.size())) continue;
                        total += val[sz(m)];
                        if (rep < 0 && pre.reduced_col[sz(m)] >= 0) rep = m;
                    }
                    if (rep < 0) rep = mg.members.empty() ? -1 : mg.members.front();
                    if (rep >= 0 && rep < static_cast<Index>(val.size())) val[sz(rep)] = total;
                }
                kept_reduced.assign(sz(pre.reduced.n_cols()), 0.0);
                for (Index j = 0; j < static_cast<Index>(val.size()); ++j) {
                    const Index rj = pre.reduced_col[sz(j)];
                    if (rj >= 0) kept_reduced[sz(rj)] = val[sz(j)];
                }
                again.initial_solution = &kept_reduced;
            }
            stage_cutoff_used = kept_obj;
            if (opts.lp.time_limit_s > 0.0)
                again.lp.time_limit_s = std::min(
                    again.time_limit_s > 0.0 ? again.time_limit_s
                                             : opts.lp.time_limit_s,
                    opts.lp.time_limit_s - used_s);
            raw = solve_milp(pre.reduced, again, diag);
            diag.restart_rows_removed = static_cast<std::uint64_t>(rows_removed);
            diag.restart_cols_removed = static_cast<std::uint64_t>(cols_removed);
            diag.restart_columns_fixed = fixed_at_request;
        }
        diag.root_restarts = restarts_done;
        diag.nodes += carried.nodes;
        diag.lp_solves += carried.lp_solves;
        diag.lp_iterations += carried.lp_iterations;
        diag.lp_ms += carried.lp_ms;
        diag.cut_rounds += carried.cut_rounds;
        diag.cut_loop_ms += carried.cut_loop_ms;
        diag.rc_columns_fixed += carried.rc_columns_fixed;
        diag.rc_bounds_tightened += carried.rc_bounds_tightened;
        diag.fpump_attempts += carried.fpump_attempts;
        diag.fpump_hits += carried.fpump_hits;
        diag.fpump_rounds += carried.fpump_rounds;
        diag.fpump_lp_solves += carried.fpump_lp_solves;
        diag.fpump_ms += carried.fpump_ms;
        diag.restart_cut_rows_carried += carried.restart_cut_rows_carried;
        diag.restart_clauses_carried += carried.restart_clauses_carried;
        diag.restart_clauses_dropped += carried.restart_clauses_dropped;
        diag.restart_clauses_emptied += carried.restart_clauses_emptied;
        // A stage that ended still asking for a restart it may not take (the
        // limit, or no point kept) is an ordinary unfinished solve.
        diag.restart_requested = false;
        if (restarts_done > 0) keep_if_better(raw.x);

        diag.structural_presolve = pstats;
        diag.structural_presolve_applied =
            pstats.fixed_cols > 0 || pstats.singleton_rows > 0 ||
            pstats.binary_substitutions > 0 || pstats.merged_cols > 0 ||
            pstats.monotone_pairs_saturated > 0;
        const bool zero_column_solution = pre.reduced.n_cols() == 0 &&
            (raw.proposed_status == core::Status::Optimal ||
             raw.proposed_status == core::Status::Feasible);
        if (!raw.x.empty() || zero_column_solution) {
            raw.x = postsolve_point(pre, raw.x);
            diag.final_primal_violation =
                milp_point_max_violation(problem, raw.x, opts.int_tol);
            if (!(diag.final_primal_violation <= opts.primal_feas_tol)) {
                ++diag.structural_postsolve_failures;
                diag.globally_proved = false;
                if (raw.proposed_status == core::Status::Optimal)
                    raw.proposed_status = core::Status::Feasible;
                raw.proposed_level = core::ProofLevel::FeasibleWithGap;
                raw.termination_reason =
                    "structural presolve postsolve point failed re-validation "
                    "against the original model; claim downgraded";
                diag.termination_reason = raw.termination_reason;
            }
        }
        // A restart may have left the best point in `kept_x` (the restarted
        // search found nothing better, or ran out of time): report it, with
        // the honest combination of proof and bound.
        if (restarts_done > 0 && !kept_x.empty()) {
            const bool raw_has_point =
                !raw.x.empty() && std::isfinite(diag.final_primal_violation) &&
                diag.final_primal_violation <= opts.primal_feas_tol &&
                std::isfinite(raw.objective);
            const bool raw_better =
                raw_has_point && (problem.maximize ? raw.objective > kept_obj
                                                   : raw.objective < kept_obj);
            if (!raw_better) {
                const f64 stage_bound = diag.dual_bound;  // over the restarted box
                // The restarted stage settled its box: it exhausted against
                // the cutoff without a point, or proved its own (not better)
                // incumbent optimal. Either way nothing beats kept_obj.
                const bool exhausted = diag.proved_no_better_than_cutoff ||
                    diag.globally_proved ||
                    (!raw_has_point && raw.proposed_status == core::Status::Infeasible);
                raw.x = kept_x;
                raw.objective = kept_obj;
                diag.incumbent = kept_obj;
                diag.final_primal_violation =
                    milp_point_max_violation(problem, kept_x, opts.int_tol);
                // Every point better than kept_obj lies in the restarted box,
                // where stage_bound holds; kept_obj itself is the other
                // candidate. The optimum is at least the better of the two
                // lower bounds.
                fold_stage_bound(stage_bound, stage_cutoff_used);
                f64 bound = certified_bound;   // NaN: nothing certified
                if (exhausted) bound = kept_obj;
                diag.dual_bound = bound;
                diag.gap_rel = std::isfinite(bound)
                    ? std::fabs(kept_obj - bound) / (1.0 + std::fabs(kept_obj))
                    : core::kPosInf;
                raw.dual_bound = bound;
                const bool proved = exhausted && diag.final_primal_violation <=
                                                     opts.primal_feas_tol;
                diag.globally_proved = proved;
                if (proved) {
                    raw.proposed_status = core::Status::Optimal;
                    raw.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
                    raw.termination_reason =
                        "root restart: restarted search found nothing better";
                } else {
                    raw.proposed_status = core::Status::Feasible;
                    raw.proposed_level = core::ProofLevel::FeasibleWithGap;
                }
                diag.termination_reason = raw.termination_reason;
            }
        }
        // Include presolve, the inner solve and the original-space postsolve
        // validation in the caller's end-to-end total.
        diag.total_ms = ms_since(tpre0);
        return raw;
    }

    // Independent components (after presolve: this is the inner call).
    if (opts.component_solve && !opts.component_child && !opts.flow_cover_cuts) {
        core::RawResult by_components;
        if (solve_milp_by_components(problem, opts, diag, by_components))
            return by_components;
    }

    const auto t0 = Clock::now();
    std::optional<core::RouteSpan> root_span;
    root_span.emplace(1, "bab", "root", "root", "",
                      core::RouteLedgerBucket::Search);
    diag = BabDiagnostics{};
    diag.policy_used = opts.policy;
    if (opts.flow_cover_cuts) {
        core::RawResult rejected;
        rejected.engine = "milp";
        rejected.backend = "cpu";
        rejected.proposed_status = core::Status::Unsupported;
        rejected.termination_reason =
            "flow-cover cuts are disabled: separator has a known invalid-cut counterexample";
        diag.termination_reason = rejected.termination_reason;
        return rejected;
    }
    diag.invocation_id =
        detail::next_milp_invocation_id();
    diag.parent_invocation_id = opts.parent_invocation_id;
    detail::record_milp_bounds(problem, diag.invocation_id, diag.parent_invocation_id,
                       opts.sub_mip_depth, "enter", nullptr, {}, {});

    const int para_threads = resolve_para_bab_threads(
        opts.para_bab, milp_policy_is_latest(opts.policy));
    diag.para_bab.threads_used = para_threads;
    const int para_quota = std::max(
        1, opts.para_bab.nodes_per_worker_per_phase > 0
               ? opts.para_bab.nodes_per_worker_per_phase
               : 1);

    BabOptions cut_cfg = opts;
    if (cut_cfg.auto_cuts && milp_policy_is_latest(cut_cfg.policy))
        apply_auto_cuts_policy(cut_cfg);

    TreeCutOptions tree_cut_opts = opts.tree_cut;
    apply_policy_tree_cuts(opts.policy, tree_cut_opts);
    ConflictCutOptions conflict_cut_opts = opts.conflict_cut;
    apply_conflict_cut_policy(opts.policy, conflict_cut_opts);
    // Dense pure-binary (enigma-class): Mexi trail+analysis stalls node LPs
    // (Interrupted / unproved) even in SafeLimited. Keep default-on for mixed
    // MIPs (any continuous column); auto-disable only when every column is a
    // 0/1 integer unless --conflict-cut-paper / force_paper.
    // misc03 (159 binaries + 1 continuous) stays on; support+prop Verified
    // accepts small-support FUIP cuts. enigma (100 pure binary) stays off.
    if (conflict_cut_opts.enabled && !conflict_cut_opts.force_paper) {
        Index n_bin = 0, n_int = 0, n_cont = 0;
        for (Index j = 0; j < problem.n_cols(); ++j) {
            if (problem.is_integer.empty() || !problem.is_integer[sz(j)]) {
                ++n_cont;
                continue;
            }
            ++n_int;
            if (problem.col_lo[sz(j)] >= -1e-9 &&
                problem.col_hi[sz(j)] <= 1.0 + 1e-9)
                ++n_bin;
        }
        if (n_cont == 0 && n_bin >= 80 && n_bin == n_int)
            conflict_cut_opts.enabled = false;
    }
    int conflict_cut_cap = conflict_cut_opts.max_learned_cuts;
    DynSepOptions dynsep_opts = opts.dynsep;
    // Sync nested separator option structs / force-allow from BabOptions.
    dynsep_opts.zerohalf = cut_cfg.zerohalf;
    dynsep_opts.flowcover = cut_cfg.flowcover;
    if (cut_cfg.mir_cuts) dynsep_opts.allow_mir = true;
    if (cut_cfg.lifted_cover_cuts) dynsep_opts.allow_cover = true;
    if (cut_cfg.clique_cuts) dynsep_opts.allow_clique = true;
    if (cut_cfg.zerohalf_cuts) dynsep_opts.allow_zerohalf = true;
    if (cut_cfg.flow_cover_cuts) dynsep_opts.allow_flowcover = true;
    apply_dynsep_policy(opts.policy, dynsep_opts);
    // The small short-budget light-arm clamp lives AFTER the L2Sep block
    // below (it must override the prior's optional-arm enables).

    L2SepOptions l2sep_opts = opts.l2sep;
    apply_l2sep_policy(opts.policy, l2sep_opts);
    L2SepModel l2sep_model;
    if (l2sep_opts.enabled && !l2sep_opts.model_path.empty()) {
        if (!load_l2sep_model(l2sep_opts.model_path, l2sep_model))
            l2sep_model.clear();
    }
    L2SepDiagnostics l2sep_diag;

    HgtsmOptions hgtsm_opts = opts.hgtsm;
    apply_hgtsm_policy(opts.policy, hgtsm_opts);
    HgtsmModel hgtsm_model;
    HgtsmCollector hgtsm_collector;
    hgtsm_collector.max_samples = hgtsm_opts.collect_max_samples;
    if (hgtsm_opts.enabled && !hgtsm_opts.model_path.empty()) {
        if (!load_hgtsm_model(hgtsm_opts.model_path, hgtsm_model))
            hgtsm_model.clear();
    }
    HgtsmDiagnostics hgtsm_diag;
    HgtsmLpStateVec hgtsm_lp_state{};
    const bool hgtsm_use_graph =
        hgtsm_opts.enabled && hgtsm_prefer_graph(hgtsm_model, hgtsm_opts);

    GcsCollector gcs_collector;
    gcs_collector.max_samples = 50000;

    // L2Sep-v1: instance-aware DynSepOptions before the controller is built.
    if (l2sep_opts.enabled) {
        SeparatorState sep0;
        configure_dynsep_from_l2sep(l2sep_opts, l2sep_model, problem, sep0,
                                    nullptr, dynsep_opts, l2sep_diag,
                                    /*mid_tree=*/false);
        // Force-on user flags still win after L2Sep.
        if (cut_cfg.mir_cuts) dynsep_opts.allow_mir = true;
        if (cut_cfg.lifted_cover_cuts) dynsep_opts.allow_cover = true;
        if (cut_cfg.clique_cuts) dynsep_opts.allow_clique = true;
        if (cut_cfg.zerohalf_cuts) dynsep_opts.allow_zerohalf = true;
        if (cut_cfg.flow_cover_cuts) dynsep_opts.allow_flowcover = true;
    }
    // Small short-budget instances keep DynSep ON but under a structural
    // light-arm clamp (measured 2026-09-14, gt2/lseu: MIR/cover arms stalled
    // dual proofs). L2Sep still runs first and sets its own allow/budget
    // priors; this clamp then OVERRIDES the optional arms - it is a
    // structural rule keyed on (n_cols, time limit), not an L2Sep decision.
    // auto_cuts opts in explicitly and bypasses this clamp for MIR/cover.
    if (milp_policy_is_latest(opts.policy) && dynsep_opts.enabled &&
        !cut_cfg.auto_cuts && problem.n_cols() <= 400 && opts.time_limit_s > 0.0 &&
        opts.time_limit_s <= 60.0) {
        dynsep_opts.max_depth_optional = 0;
        dynsep_opts.max_optional_arms = 1;
        if (!cut_cfg.mir_cuts) dynsep_opts.allow_mir = false;
        if (!cut_cfg.lifted_cover_cuts) dynsep_opts.allow_cover = false;
    }

    DynSepController dynsep(dynsep_opts);
    f64 dynsep_last_gain = 0.0;
    std::uint64_t last_l2sep_node = 0;
    std::uint64_t last_gcs_reinject_node = 0;
    (void)last_l2sep_node;
    (void)last_gcs_reinject_node;

    core::RawResult raw;
    raw.engine = "milp_bab";
    raw.backend = "cpu";
    raw.proposed_status = core::Status::NoSolutionFound;
    raw.proposed_level = core::ProofLevel::None;

    // Root allowances in seconds (see BabOptions::root_*_share). A positive
    // cap bounds the share; 0 leaves it uncapped.
    const auto root_allowance_s = [&](double share, double cap_s) -> double {
        const double a = share * opts.time_limit_s;
        return cap_s > 0.0 ? std::min(a, cap_s) : a;
    };
    const double root_setup_allowance_s =
        root_allowance_s(opts.root_reduction_share, opts.root_reduction_cap_s);
    const double root_cut_allowance_s =
        root_allowance_s(opts.root_cut_share, opts.root_cut_max_s);
    const double root_total_allowance_s =
        opts.root_total_share * opts.time_limit_s;
    // Root allowance shared by every setup phase, so they cannot collectively
    // starve the search. Returns seconds remaining of the root budget.
    const auto root_budget_left = [&]() -> f64 {
        if (opts.time_limit_s <= 0.0) return 0.0;
        return std::max(0.0, root_setup_allowance_s - ms_since(t0) / 1000.0);
    };

    // Preserve simplex certificates on the LP-only exits. MILP tree evidence
    // does not describe an LP ray or an incumbent that bypassed the tree.
    const auto solve_lp_only = [&]() {
        engines::SimplexDiagnostics sd;
        raw = engines::solve_simplex(problem, opts.lp, sd, nullptr);
        diag.lp_solves = 1;
        diag.nodes = 1;
        diag.lp_only = true;
        diag.lp_only_evidence = certify::check_lp_result(
            problem, raw, engines::simplex_evidence(sd, opts.lp));
        auto checked = certify::finalize_result(raw, diag.lp_only_evidence);
        // LP presolve does not yet lift all terminal rays. Retry on the
        // original model only when its terminal claim lacks a certificate.
        if (opts.lp.presolve &&
            (raw.proposed_status == core::Status::Infeasible ||
             raw.proposed_status == core::Status::Unbounded) &&
            checked.status == core::Status::NoSolutionFound) {
            engines::SimplexOptions retry_opts = opts.lp;
            retry_opts.presolve = false;
            double budget = opts.lp.time_limit_s;
            if (opts.time_limit_s > 0.0)
                budget = budget > 0.0 ? std::min(budget, opts.time_limit_s) : opts.time_limit_s;
            const double remaining = budget - ms_since(t0) / 1000.0;
            if (budget <= 0.0 || remaining > 0.0) {
                if (budget > 0.0) retry_opts.time_limit_s = remaining;
                raw = engines::solve_simplex(problem, retry_opts, sd, nullptr);
                ++diag.lp_solves;
                diag.lp_only_evidence = certify::check_lp_result(
                    problem, raw, engines::simplex_evidence(sd, retry_opts));
            }
        }
        if (diag.lp_only_evidence.checker_passed) {
            diag.final_primal_violation = milp_point_max_violation(problem, raw.x, opts.int_tol);
            diag.lp_only_evidence.max_primal_violation = diag.final_primal_violation;
            diag.lp_only_evidence.checker_passed =
                diag.final_primal_violation <= opts.primal_feas_tol;
            if (diag.lp_only_evidence.checker_passed)
                diag.incumbent = problem.objective(raw.x);
        }
        checked = certify::finalize_result(raw, diag.lp_only_evidence);
        diag.globally_proved = checked.status == core::Status::Optimal ||
                               checked.status == core::Status::Infeasible ||
                               checked.status == core::Status::Unbounded;
        diag.dual_bound = raw.dual_bound;
        diag.gap_rel = sd.gap_rel;
        diag.total_ms = ms_since(t0);
        diag.termination_reason = "no integer columns; LP solve";
        return raw;
    };

    model::LpProblem mip = problem;

    // A model the caller supplied with NO integer columns is an LP, and must
    // be solved and answered as one. Implied integrality is a MIP presolve
    // technique -- its value is telling branch-and-bound that some continuous
    // variable will land integral anyway, so there is nothing to branch on.
    // On a pure LP it has zero upside and a catastrophic downside: every mark
    // it makes RESTRICTS the feasible set, and snap_integer_bounds() then
    // rounds the marked column's bounds.
    //
    // Measured 2026-09-19 on two integer-free Netlib LPs:
    //   80bau3b  -> false Infeasible   (simplex proves Optimal 9.8722419241e+05)
    //   d2q06c   -> objective 1.2278462651e+05 against a true LP optimum of
    //               1.2278421081e+05 -- a silently WORSE answer to a different
    //               problem than the one the user handed us.
    // Three separate rules were implicated (tu_network_block, dual_rational,
    // network), so this is a class of defect rather than one bad rule.
    if (problem.n_integer() == 0) {
        return solve_lp_only();
    }

    // Equality +-1 + TU/network/C1 implied integrality (WP-F).
    //
    // GATED, and the gate matters: this call used to be unconditional, so
    // --no-implied-int (which only reaches mip_pre.implied_integers inside
    // run_mip_presolve) could not switch it off. It also runs BEFORE the
    // `mip.n_integer() == 0` pure-LP fast path below, so marking even one
    // continuous column integer diverts a pure LP into the full MILP
    // machinery. On netlib 80bau3b and d2q06c -- both integer-free and both
    // Optimal under --engine simplex -- that produced a false Infeasible.
    ImpliedIntOptions implied_opts;
    implied_opts.enabled = opts.mip_pre.implied_integers;
    // Bounded like every other root phase: it runs BEFORE anything that checks
    // the clock, so unbudgeted it can consume the whole solver limit on a
    // large model (4586 ms on atlanta-ip, finding nothing).
    if (opts.time_limit_s > 0.0) {
        const f64 share = kRootSetupShare * root_budget_left();
        implied_opts.enabled = implied_opts.enabled && share > kMinPhaseBudget;
        implied_opts.time_limit_s = std::max(kMinPhaseBudget, share);
    }
    ImpliedIntDiagnostics implied_diag;
    if (implied_opts.enabled)
        implied_diag = infer_implied_integers_ex(mip, implied_opts);
    diag.root_implied_int_marked = implied_diag.total;

    // Latest branching learners (WP-H). Classical ignores them entirely.
    // Heuristic variable choice only - never writes dual bounds.
    SparseSbModel sparse_sb_model;
    SparseSbCollector sparse_sb_collector;
    sparse_sb_collector.max_samples = opts.sparse_sb.collect_max_samples;
    ScMilpModel sc_milp_model;
    ScMilpCollector sc_milp_collector;
    sc_milp_collector.max_samples = opts.sc_milp.collect_max_samples;
    LiftedBranchState lifted_state;
    lifted_state.buffer.max_samples = opts.sparse_sb.collect_max_samples;
    PlanBbPolicy planbb_policy;
    PlanBbModel planbb_model;
    PlanBbCollector planbb_collector;
    planbb_collector.max_samples = opts.planbb.collect_max_samples;
    PlanBbGraphPool planbb_graph = zero_graph_pool();

    const bool latest_branch = milp_policy_is_latest(opts.policy);
    const bool sparse_sb_want = latest_branch && opts.sparse_sb.enabled;
    auto try_default_model = [](std::string& path, const char* leaf) {
        if (!path.empty()) return;
        const char* cands[] = {
            "models/milp/",
            "./models/milp/",
            "../models/milp/",
            "share/sor/milp/",
        };
        for (const char* prefix : cands) {
            std::string p = std::string(prefix) + leaf;
            std::ifstream in(p);
            if (in) {
                path = std::move(p);
                return;
            }
        }
    };
    // Copy paths so we can fill defaults without mutating caller's options.
    SparseSbOptions sparse_sb_opts = opts.sparse_sb;
    ScMilpOptions sc_milp_opts = opts.sc_milp;
    try_default_model(sparse_sb_opts.model_path, "sparse_sb.model");
    try_default_model(sc_milp_opts.model_path, "sc_milp.model");
    if (sparse_sb_want && !sparse_sb_opts.model_path.empty()) {
        if (!load_sparse_sb_model(sparse_sb_opts.model_path, sparse_sb_model))
            sparse_sb_model.clear();
    }
    const bool sc_milp_want = latest_branch && sc_milp_opts.enabled;
    if (sc_milp_want && !sc_milp_opts.model_path.empty()) {
        if (!load_sc_milp_model(sc_milp_opts.model_path, sc_milp_model))
            sc_milp_model.clear();
    }
    const bool lifted_want = latest_branch && opts.lifted.enabled;
    if (lifted_want && !opts.lifted.expert_path.empty()) {
        if (load_lifted_sb_model(opts.lifted.expert_path, lifted_state.expert))
            lifted_state.seeded_from_file = lifted_state.expert.loaded;
        else
            lifted_state.expert.clear();
    }
    const bool planbb_want = latest_branch && opts.planbb.enabled;
    if (planbb_want && !opts.planbb.model_path.empty()) {
        if (!load_planbb_model(opts.planbb.model_path, planbb_model))
            planbb_model.clear();
    }
    if (planbb_want && !opts.planbb.policy_path.empty()) {
        if (!load_planbb_policy(opts.planbb.policy_path, planbb_policy))
            planbb_policy.clear();
    }
    const bool planbb_paper =
        planbb_want && planbb_use_paper_path(opts.planbb, &planbb_model);
    diag.planbb_paper = planbb_paper;
    const bool collect_planbb =
        planbb_want && opts.planbb.collect_labels;

    auto resolve_branch_strategy = [&]() -> BranchStrategy {
        if (!latest_branch) return BranchStrategy::Auto;
        if (opts.branch_strategy != BranchStrategy::Auto)
            return opts.branch_strategy;
        // Default under Auto is Achterberg's reliability/pseudocost
        // branching, not a learned scorer: SC-MILP reads a hand-typed linear
        // model off a relative path (`models/milp/sc_milp.model` and
        // friends), so which strategy ran depended on the process's working
        // directory, and with `use_heuristic_without_model` defaulted on it
        // fired even with no model file at all. opts.paper_reliability=false
        // opts back into that legacy auto-resolution for callers who still
        // want it.
        if (opts.paper_reliability) return BranchStrategy::Auto;
        // Legacy: SC-MILP when available (enigma-scale proofs). Sparse-SB
        // only on larger models - cold Sparse-SB hurt dense binaries. Tiny
        // models without SC fall through to reliability (BranchStrategy::Auto).
        if (sc_milp_want &&
            (sc_milp_model.loaded || sc_milp_opts.use_heuristic_without_model))
            return BranchStrategy::ScMilp;
        if (mip.n_cols() > 250 && sparse_sb_model.loaded)
            return BranchStrategy::SparseSb;
        if (mip.n_cols() > 250 && lifted_want) return BranchStrategy::Lifted;
        return BranchStrategy::Auto;  // reliability / pseudocost path
    };
    const BranchStrategy branch_strat = resolve_branch_strategy();
    diag.branch_strategy_resolved = branch_strat;

    const bool sparse_sb_model_ready =
        sparse_sb_want && sparse_sb_model.loaded &&
        (opts.branch_strategy == BranchStrategy::SparseSb ||
         mip.n_cols() > 250);
    const bool collect_sparse_sb =
        sparse_sb_want && opts.sparse_sb.collect_labels;
    const bool collect_sc_milp =
        sc_milp_want && opts.sc_milp.collect_labels;
    // Learned-branching feature vectors have a reader only when a learner
    // picks the branching variable or labels are being collected. Building
    // them anyway -- every candidate, every node -- used to be most of the
    // branching time under the default reliability selector, which never
    // looks at them (beasleyC3: 1343 of 1351 ms over 200 nodes).
    const bool learner_picks_variables = latest_branch &&
        (branch_strat == BranchStrategy::SparseSb ||
         branch_strat == BranchStrategy::ScMilp ||
         branch_strat == BranchStrategy::Lifted ||
         branch_strat == BranchStrategy::PlanBb);
    const bool lifted_features_used = lifted_want &&
        (branch_strat == BranchStrategy::Lifted ||
         opts.lifted_collect_out != nullptr);
    const bool branch_features_used = learner_picks_variables ||
        collect_sparse_sb || collect_sc_milp || collect_planbb ||
        lifted_features_used;
    // Per-model feature data, built on first use; mip never changes during
    // the search, so revision 0 describes it for the whole solve.
    BranchFeatureCache feature_cache;

    if (mip.n_integer() == 0) {
        // Pure LP - just call simplex. The raw result carries the LP's own
        // proposed status, but the MILP evidence machinery reads THIS diag,
        // so it must reflect what actually happened: a certified relaxation
        // of a tree with a single (root) node is exactly the "tree
        // exhausted, every LP proved" case, and without these fields
        // finalize_result would downgrade the LP's Optimal to
        // NoSolutionFound (observed on industrial blend_lp instances fed to
        // --engine milp). An uncertified LP stays honestly uncertified.
        return solve_lp_only();
    }

    const Index n = mip.n_cols();
    const f64 sense = mip.maximize ? -1.0 : 1.0;
    // Strong branching is highly effective on small MIPs but can dominate the
    // solve on wide models. Disable it there; every probe remains advisory and
    // certified node LPs still control correctness.
    const bool use_reliability = opts.reliability_branching &&
        n <= 2000 && mip.nnz() <= 10000;
    // Latest + small models: plunge toward integer for faster proofs (enigma).
    //
    // Plunging and Para-B&B phases compete for the frontier and cannot both be
    // on. This used to be settled statically by `para_threads <= 1`, which cost
    // enigma 4.75s -> 34.64s because enigma is incumbent-bound and finishes
    // before parallelism can amortise. The cost model (para_bab.hpp) now defers
    // that call to the live search, so plunging starts ON regardless of thread
    // count and is switched off only at the moment phases actually activate.

    const bool hybrid_nodes_initial =
        opts.hybrid_node_selection ||
        (milp_policy_is_latest(opts.policy) && n <= 500);
    bool hybrid_nodes = hybrid_nodes_initial;
    bool para_active = para_threads > 1 && !opts.para_bab_cost.adaptive;
    diag.para_bab.activated = para_active;
    std::uint64_t strong_branch_budget = use_reliability
        ? opts.strong_branch_nodes : 0;
    std::vector<Index> col_degree(sz(n), 0);
    for (const Index j : mip.A.pattern.col_idx()) {
        if (j >= 0 && j < n) ++col_degree[sz(j)];
    }
    // Child ordering bias per column (integer_up_bias_all), on first use.
    std::vector<f64> mip_up_bias;

    // Work in minimize sense for bounds: lower dual bound is valid.
    // Incumbent stored as original-sense objective.
    f64 best_incumbent = mip.maximize ? -std::numeric_limits<f64>::infinity()
                                          :  std::numeric_limits<f64>::infinity();
    std::vector<f64> best_x;
    bool have_incumbent = false;
    // Two different facts, kept apart:
    //  * have_incumbent / best_x: a point that is feasible for THIS stage's model.
    //    It decides what heuristics have to do (find a point vs improve one).
    //  * the cutoff: the objective of the best point known ANYWHERE that is valid
    //    for the problem (this stage's incumbent, a restart's kept point, another
    //    arm's point). It bounds everything that only needs a number to beat:
    //    LP objective limits, pruning, reduced-cost fixing, improvement targets.
    // A restarted stage has a cutoff and no local point; conflating the two used
    // to leave its LPs, probes and cut loop without the cutoff it was given.
    f64 external_cutoff = std::numeric_limits<f64>::quiet_NaN();
    if (std::isfinite(opts.initial_cutoff)) {
        // A point found before a restart: prune against it, but never claim
        // Infeasible from an exhausted tree (see used_foreign_cutoff).
        external_cutoff = opts.initial_cutoff;
        diag.used_foreign_cutoff = true;
    }
    const auto cutoff_known = [&]() {
        return have_incumbent || std::isfinite(external_cutoff);
    };
    // Best known objective, original sense (NaN: nothing known).
    const auto known_cutoff_orig = [&]() -> f64 {
        f64 c = std::numeric_limits<f64>::quiet_NaN();
        if (have_incumbent) c = best_incumbent;
        if (std::isfinite(external_cutoff) &&
            (!std::isfinite(c) || (mip.maximize ? external_cutoff > c : external_cutoff < c)))
            c = external_cutoff;
        return c;
    };
    // The same in the working (minimisation) sense; +inf when nothing is known.
    const auto known_cutoff_min = [&]() -> f64 {
        const f64 c = known_cutoff_orig();
        return std::isfinite(c) ? sense * c : std::numeric_limits<f64>::infinity();
    };

    // Directional pseudocosts store observed LP-bound improvement per unit of
    // integer movement. They are learned only from certified relaxations.
    BranchStats bstats;
    bstats.resize(n);
    auto& pc_down_sum = bstats.pc_sum[0];
    auto& pc_up_sum = bstats.pc_sum[1];
    auto& pc_down_count = bstats.pc_count[0];
    auto& pc_up_count = bstats.pc_count[1];
    // A heuristic child continues its parent's pseudocost observations.
    if (opts.pseudocost_seed != nullptr &&
        opts.pseudocost_seed->down_sum.size() == sz(n) &&
        opts.pseudocost_seed->up_sum.size() == sz(n) &&
        opts.pseudocost_seed->down_count.size() == sz(n) &&
        opts.pseudocost_seed->up_count.size() == sz(n)) {
        pc_down_sum = opts.pseudocost_seed->down_sum;
        pc_up_sum = opts.pseudocost_seed->up_sum;
        pc_down_count = opts.pseudocost_seed->down_count;
        pc_up_count = opts.pseudocost_seed->up_count;
        ++diag.sub_mip_context_seeded;
    }

    std::priority_queue<Node, std::vector<Node>, NodeCmp> open;
    // Hybrid node selection (item 16): a bounded depth-first "plunge" stack
    // that runs alongside the best-bound queue. Every node created goes into
    // exactly one of these two containers, so both the tree-exhausted check
    // and the final dual-bound drain below must account for both -- nothing
    // is dropped, this only changes visitation ORDER.
    std::vector<Node> plunge_stack;
    // Root basis seeding (set below): when the cut loop ran, its final
    // proved basis warm-starts the root node LP, which would otherwise
    // re-solve the identical relaxation cold -- measured as a full ~20-28s
    // duplicate solve on schedule_milp HUGE.
    engines::SimplexBasis cut_loop_basis;
    bool have_cut_loop_basis = false;
    // True once a root relaxation (cut loop or root node LP) has been
    // CERTIFIED optimal with a finite objective. Every node LP is a
    // restriction of the root box, so this implies no descendant can be
    // unbounded - which is what makes an InfeasibleOrUnbounded node status
    // trustworthy as "infeasible" for nogood learning below.
    bool root_relaxation_bounded = false;
    std::uint64_t tightened_row_bounds = 0;
    // Each invocation writes its own snapshot. A shared path is not used:
    // concurrent and recursive solves must not overwrite one another.
    detail::record_milp_bounds(mip, diag.invocation_id, diag.parent_invocation_id,
                       opts.sub_mip_depth, "received_box",
                       std::getenv("SOR_DUMP_MIP_BOUNDS"), mip.col_lo, mip.col_hi);
    model::LpProblem search_problem = mip;
    if (opts.integer_row_rounding)
        search_problem = tighten_integral_rows(mip, tightened_row_bounds);
    diag.integer_row_roundings = tightened_row_bounds;
    std::uint64_t cover_cuts = 0;
    search_problem = add_binary_cover_cuts(search_problem, cover_cuts);
    diag.binary_cover_cuts = cover_cuts;

    // Root probing / MIP-presolve / symmetry (WP-F/G). Hook sits once at MILP
    // entry before B&C: dual-fix⊕probing, clique probing, GF2, components,
    // TU implied-int, OBBT-lite, multi-round restart (conflict rebuild), then
    // color-refinement orbital fixing.
    //
    // Conflict-graph consumers (cuts, node prop) still read `conflict_graph`.
    // Dual fixing may drop suboptimal feasible points while retaining ≥1
    // optimum - that is intentional for the mip_presolve path only.
    ConflictGraph conflict_graph;
    // Resume probing where structural presolve stopped (same matrix, box inside
    // the probed one): its graph and the record of which binaries it covered.
    ProbingState probing_state;
    ProbingState* probing_state_ptr = nullptr;
    if (opts.probing_carry != nullptr && opts.carry_probing &&
        opts.probing_carry->usable_for(mip)) {   // `mip`: before cover rows were appended
        conflict_graph = opts.probing_carry->graph;
        probing_state = opts.probing_carry->state;
        probing_state_ptr = &probing_state;
        diag.probing_resumed = true;
        diag.probing_carried_probed = probing_state.probed_count();
    }
    std::vector<f64> root_lo = search_problem.col_lo,
                     root_hi = search_problem.col_hi;
    // Earliest possible snapshot of the root box, before probing, MIP
    // presolve, symmetry or reduced-cost fixing touch it. Comparing this with
    // SOR_DUMP_ROOT_BOUNDS localises which stage moved a bound.
    detail::record_milp_bounds(search_problem, diag.invocation_id,
                       diag.parent_invocation_id, opts.sub_mip_depth,
                       "pre_probing_box", std::getenv("SOR_DUMP_ROOT_BOUNDS0"),
                       root_lo, root_hi);
    const bool want_conflict_graph =
        opts.probing || cut_cfg.clique_cuts || opts.conflict_propagation ||
        (opts.mip_presolve && opts.mip_pre.clique_probing);

    auto mark_root_infeasible = [&](const char* reason) {
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.termination_reason = reason;
        diag.globally_proved = true;
        diag.total_ms = ms_since(t0);
        diag.termination_reason = raw.termination_reason;
        char rbuf[192];
        std::snprintf(rbuf, sizeof rbuf, "\"reason\":\"%s\"", reason);
        SOR_ROUTE(1, "bab", "root_infeasible", rbuf);
    };

    diag.ms_root_setup = ms_since(t0);
    if (opts.mip_presolve) {
        ProbingOptions probe_opts = opts.probe;
        probe_opts.enabled = opts.probing;
        probe_opts.dual_fix_in_probing = opts.mip_pre.dual_fix_in_probing;
        if (opts.time_limit_s > 0.0)
            probe_opts.probe_time_limit_s =
                std::min(probe_opts.probe_time_limit_s,
                         0.1 * opts.time_limit_s);
        MipPresolveOptions mpo = opts.mip_pre;
        mpo.enabled = true;
        // Root setup gets a BOUNDED share of the budget. Presolve and symmetry
        // are worth real time on a big model, but they are worth nothing at
        // all if they consume the whole budget and the search never runs --
        // which is exactly what happened: on atlanta-ip with a 1 s limit they
        // took 15.2 s and 25.3 s respectively, and the solve did 0 nodes.
        // Across a 20-instance MIPLIB2017 sample, HALF never searched a node.
        if (opts.time_limit_s > 0.0) {
            // Presolve is the most valuable root phase, so it may take the
            // whole remaining root allowance rather than a share of it.
            const f64 share = root_budget_left();
            mpo.enabled = mpo.enabled && share > kMinPhaseBudget;
            mpo.time_limit_s = std::max(kMinPhaseBudget, share);
        }
        diag.mip_presolve_diag = run_mip_presolve(
            search_problem, root_lo, root_hi, conflict_graph, mpo,
            /*run_probing=*/want_conflict_graph, probe_opts, &opts.lp, probing_state_ptr);
        diag.conflict = diag.mip_presolve_diag.conflict;
        diag.mip_restart_recommended =
            diag.mip_presolve_diag.restart_recommended;
        if (diag.mip_presolve_diag.infeasible) {
            mark_root_infeasible("root mip-presolve proved infeasible");
            return raw;
        }
        search_problem.col_lo = root_lo;
        search_problem.col_hi = root_hi;
    } else if (want_conflict_graph) {
        ProbingOptions probe_opts = opts.probe;
        probe_opts.enabled = opts.probing;
        probe_opts.dual_fix_in_probing = false;
        if (opts.time_limit_s > 0.0)
            probe_opts.probe_time_limit_s =
                std::min(probe_opts.probe_time_limit_s,
                         0.1 * opts.time_limit_s);
        diag.conflict = build_conflict_graph(search_problem, root_lo, root_hi,
                                             conflict_graph, probe_opts,
                                             probing_state_ptr);
        if (diag.conflict.infeasible) {
            mark_root_infeasible("root probing proved infeasible");
            return raw;
        }
        search_problem.col_lo = root_lo;
        search_problem.col_hi = root_hi;
    }

    if (opts.symmetry) {
        const ConflictGraph* cg_ptr =
            conflict_graph.empty() ? nullptr : &conflict_graph;
        SymmetryOptions sym = opts.sym;
        // Bounded share of what is LEFT, not of the original budget: presolve
        // has already run, so charging symmetry a share of the total would let
        // the two together overrun.
        //
        // NOTE the 0-means-unlimited convention: clamping a negative remainder
        // to 0 does NOT disable the phase, it UNCAPS it. That bug made this
        // whole fix a no-op on exactly the models it was written for, because
        // an overrunning presolve leaves a negative remainder. When nothing is
        // left, skip symmetry outright.
        bool run_symmetry = true;
        if (opts.time_limit_s > 0.0) {
            const f64 share = root_budget_left();
            if (share <= kMinPhaseBudget) {
                run_symmetry = false;
            } else {
                sym.time_limit_s = share;
            }
        }
        if (!run_symmetry) {
            diag.symmetry_diag.aborted_on_time = 1;
        } else
        diag.symmetry_diag = apply_symmetry(search_problem, cg_ptr, root_lo,
                                            root_hi, sym);
        search_problem.col_lo = root_lo;
        search_problem.col_hi = root_hi;
        // A folded representative now means a SUM, and its members are fixed
        // to 0 -- neither still matches what probing proved about the
        // original binary columns. Purge every fact naming any of them (as
        // either an implied-bound endpoint or a clique literal) before the
        // conflict graph is used again below (symmetry itself, cut
        // separation, node propagation). Facts among untouched columns are
        // unaffected: see ConflictGraph::forget_columns.
        if (!conflict_graph.empty() && diag.symmetry_diag.folding_applied) {
            std::vector<char> drop(sz(search_problem.n_cols()), 0);
            for (const FoldGroup& g : diag.symmetry_diag.folds)
                for (const Index m : g.members)
                    if (m >= 0 && sz(m) < drop.size()) drop[sz(m)] = 1;
            conflict_graph.forget_columns(drop);
        }
    }
    const auto mark_milestone = [&](const char* what, const std::string& detail) {
        if (opts.sub_mip_depth != 0 || diag.milestones.size() >= 64) return;
        char b[48];
        std::snprintf(b, sizeof b, "%8.3f s  ", ms_since(t0) / 1000.0);
        diag.milestones.push_back(std::string(b) + what + (detail.empty() ? "" : "  " + detail));
    };
    {
        char b[160];
        std::snprintf(b, sizeof b, "probing %s, %llu binaries probed of %zu",
                      diag.probing_resumed ? "resumed from presolve" : "fresh",
                      static_cast<unsigned long long>(probing_state_ptr ? probing_state.probed_count() : 0),
                      conflict_graph.binaries().size());
        mark_milestone("root setup done (presolve, probing, symmetry)", b);
    }
    // Folded representatives store an integer sum; lift before storing /
    // returning incumbents so x stays in the original binary space.
    auto lift_sym_x = [&](std::vector<f64>& x) {
        if (diag.symmetry_diag.folding_applied)
            lift_folded_solution(diag.symmetry_diag, x, opts.sym.tol);
    };

    auto accept_incumbent = [&](std::vector<f64> x, bool folded_space,
                                f64& obj_out) -> bool {
        obj_out = core::kNaN;
        if (folded_space) lift_sym_x(x);
        const f64 violation = milp_point_max_violation(problem, x, opts.int_tol);
        if (!std::isfinite(violation)) ++diag.incumbents_rejected_unsnappable;
        if (violation > opts.primal_feas_tol) return false;
        const f64 obj = problem.objective(x);
        if (!std::isfinite(obj)) return false;
        obj_out = obj;
        if (have_incumbent && !(problem.maximize ? obj > best_incumbent
                                                : obj < best_incumbent))
            return false;
        best_x = std::move(x);
        best_incumbent = obj;
        have_incumbent = true;
        {
            char b[64];
            std::snprintf(b, sizeof b, "obj=%.10g", obj);
            mark_milestone("incumbent accepted", b);
        }
        return true;
    };

    auto timed_out = [&]() {
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - t0).count() >
                   opts.time_limit_s;
    };
    auto seconds_left = [&]() {
        if (opts.time_limit_s <= 0.0) return std::numeric_limits<double>::max();
        return opts.time_limit_s -
               std::chrono::duration<double>(Clock::now() - t0).count();
    };

    // The heuristic-layer ceiling. Every heuristic below asks this before it
    // spends anything; once the layer has used its share of the budget the tree
    // gets the rest, which is the whole point of having a share at all.
    // Set in the node loop: true once the global dual bound has not improved
    // for opts.dual_stall_window nodes.
    bool dual_stalled = false;
    f64 last_dual_bound = -std::numeric_limits<f64>::infinity();
    std::uint64_t last_dual_node = 0;
    // Direct node rounding backs off geometrically after rounds that find
    // no improving point: the node LP point changes little between nearby
    // nodes, and a failed repair search costs up to ~80 ms on 10k-column
    // models (decomp2, drayage), which ate two thirds of a 60 s run.
    constexpr std::uint64_t kMaxDirectRoundInterval = 64;
    std::uint64_t direct_round_interval = 1;
    std::uint64_t next_direct_round_node = 0;

    auto heuristics_over_budget = [&](double intended_budget_ms = 0.0) {
        if (opts.time_limit_s <= 0.0) return false;
        // Two regimes, because the trade-off genuinely reverses. With no
        // incumbent the heuristics are the only thing that can produce one and
        // the tree has nothing to prune against, so they get most of the
        // budget. Once an incumbent exists the tree is what closes the gap, and
        // every further second in a heuristic is a second it does not get.
        // Third: small residual gap → proof mode (even tighter heuristic share).
        const bool proof_mode =
            have_incumbent && std::isfinite(diag.gap_rel) &&
            diag.gap_rel >= 0.0 &&
            diag.gap_rel < opts.heuristic_proof_gap;
        const double frac =
            !have_incumbent ? opts.heuristic_budget_frac_no_incumbent
            : dual_stalled  ? opts.heuristic_budget_frac_stalled
            : proof_mode   ? opts.heuristic_budget_frac_proof
                            : opts.heuristic_budget_frac;
        if (frac >= 1.0) return false;
        // The root primal passes (pump, SPP repair, objective face from the root
        // snapshot) have their own allowance (root_primal_total_frac). Charging
        // them here as well used the tree heuristics' budget up before they ran:
        // on pg the root passes took 3 s and Balans, RENS and the dives got
        // nothing, so the incumbent that p4 had by 20 s arrived at 40 s.
        const bool over = heuristic_spent_ms(diag) - diag.root_primal_ms >
                          frac * opts.time_limit_s * 1000.0;
        if (over) {
            ++diag.heuristic_budget_blocks;
            if (intended_budget_ms > 0.0)
                diag.heuristic_budget_blocked_ms += intended_budget_ms;
        }
        return over;
    };
    auto in_proof_mode = [&]() {
        return have_incumbent && std::isfinite(diag.gap_rel) &&
               diag.gap_rel >= 0.0 &&
               diag.gap_rel < opts.heuristic_proof_gap;
    };
    // Live gap for proof-mode gates. Without this, diag.gap_rel stays +inf
    // until teardown and heuristics_over_budget never enters proof_mode
    // (gt2 spent ~45% wall in Balans with incumbent already at opt).
    auto refresh_live_gap = [&](f64 dual_min_working) {
        if (!have_incumbent || !std::isfinite(best_incumbent)) return;
        if (!std::isfinite(dual_min_working)) return;
        const f64 dual_orig = sense * dual_min_working;
        diag.gap_rel = std::fabs(best_incumbent - dual_orig) /
                       (1.0 + std::fabs(best_incumbent));
    };


    // Feasibility Jump (Luteberget & Sandvik, MPC 2023; sor/search/feasjump.hpp).
    // Heuristic only, like everything else that can set best_x: the point it
    // returns is re-checked against `problem` -- the caller's model, not the
    // cut-augmented search copy -- before it is allowed to become the
    // incumbent, and it never supplies a bound or a pruning certificate.
    //
    // Runs against `mip` with the probing-tightened root box. Both are valid
    // for every integer-feasible point, so a point inside them is a point of
    // the original model; the explicit re-check below is what makes that a
    // verified statement rather than an argued one.
    auto try_feasjump = [&](const std::vector<f64>* seed, double budget,
                            f64 cutoff) -> bool {
        if (!opts.feasibility_jump || budget <= 0.01) return false;
        if (heuristics_over_budget(budget * 1000.0)) return false;
        FeasJumpOptions fo;
        fo.time_limit_s = budget;
        fo.feas_tol = opts.primal_feas_tol;
        fo.objective_cutoff = cutoff;
        // Distinct but deterministic per attempt: a repeated cold run with the
        // same seed would retrace the same walk and learn nothing.
        fo.seed = static_cast<std::uint32_t>(20260907u +
                                             diag.feasjump_attempts * 7919u);
        std::vector<f64> xf;
        FeasJumpDiagnostics fd;
        ++diag.feasjump_attempts;
        const bool ok =
            feasibility_jump(mip, root_lo, root_hi, seed, fo, xf, fd);
        diag.feasjump_moves += fd.moves;
        diag.feasjump_weight_updates += fd.weight_updates;
        diag.feasjump_restarts += fd.restarts;
        diag.feasjump_ms += fd.ms;
        if (diag.feasjump_attempts == 1 ||
            fd.best_violated_rows < diag.feasjump_best_violated_rows)
            diag.feasjump_best_violated_rows = fd.best_violated_rows;
        if (!ok) return false;
        f64 fobj;
        if (!accept_incumbent(std::move(xf), false, fobj)) return false;
        ++diag.feasjump_hits;
        ++diag.heuristic_hits;
        {
            f64 dual_min = last_dual_bound;
            if (!open.empty() && std::isfinite(open.top().bound))
                dual_min = open.top().bound;
            refresh_live_gap(dual_min);
        }
        if (opts.verbose)
            std::printf("  [milp] feasjump incumbent %.10e at node %llu\n",
                        best_incumbent,
                        static_cast<unsigned long long>(diag.nodes));
        return true;
    };

    // Fix-Propagate-Repair (Salvagnin, MPC 2024; sor/search/fixprop.hpp).
    // Constructive and LP-free, so it is a genuine complement to Feasibility
    // Jump rather than another variant of it: FJ walks COMPLETE assignments
    // and stalls a few violated rows short (measured: fhnw-binpack4-4 reaches
    // exactly 1 violated row and never closes it, at any budget), while this
    // extends a PARTIAL assignment that propagation keeps provably extensible.
    auto try_fixprop = [&](const std::vector<f64>* reference,
                           double budget) -> bool {
        if (!opts.fixprop || budget <= 0.01) return false;
        if (heuristics_over_budget(budget * 1000.0)) return false;
        FixPropOptions fpo;
        fpo.time_limit_s = budget;
        fpo.feas_tol = opts.primal_feas_tol;
        fpo.seed = 20260920u + diag.fixprop_attempts * 7919u;
        std::vector<f64> xf;
        FixPropDiagnostics fd;
        ++diag.fixprop_attempts;
        if (opts.verbose)
            std::printf("  [milp] fixprop start budget %.3f s at node %llu\n",
                        budget, static_cast<unsigned long long>(diag.nodes));
        const bool ok =
            fix_and_propagate(mip, root_lo, root_hi, reference, fpo, xf, fd);
        if (opts.verbose)
            std::printf("  [milp] fixprop end found=%d elapsed=%.3f s "
                        "dives=%llu propagation=%llu work=%llu\n",
                        ok ? 1 : 0, fd.ms / 1000.0,
                        static_cast<unsigned long long>(fd.dives),
                        static_cast<unsigned long long>(fd.propagations),
                        static_cast<unsigned long long>(fd.work));
        diag.fixprop_dives += fd.dives;
        diag.fixprop_fixings += fd.fixings;
        diag.fixprop_conflicts += fd.conflicts;
        diag.fixprop_backtracks += fd.backtracks;
        diag.fixprop_bottom_lps += fd.bottom_lps;
        diag.fixprop_best_depth_pct =
            std::max(diag.fixprop_best_depth_pct, fd.best_depth_pct);
        diag.fixprop_ms += fd.ms;
        if (!ok) return false;
        // fix_and_propagate already validated against `mip`; `problem` is the
        // caller's space, which is what an incumbent must satisfy.
        f64 fobj;
        if (!accept_incumbent(std::move(xf), false, fobj)) return false;
        ++diag.fixprop_hits;
        ++diag.heuristic_hits;
        {
            f64 dual_min = last_dual_bound;
            if (!open.empty() && std::isfinite(open.top().bound))
                dual_min = open.top().bound;
            refresh_live_gap(dual_min);
        }
        if (opts.verbose)
            std::printf("  [milp] fixprop incumbent %.10e at node %llu\n",
                        best_incumbent,
                        static_cast<unsigned long long>(diag.nodes));
        return true;
    };

    // Solves one LNS sub-problem: a restriction of this MILP (a tightened box,
    // possibly plus a local-branching distance row or a proximity cutoff row).
    // Every such sub-problem admits only points that are feasible for `mip`, so
    // a point it returns is a point of this MILP -- and it is re-validated
    // against `problem` before it can become the incumbent anyway.
    // Set for the root opportunity only: that child gets a real budget, so it
    // may strengthen its own root (a few cut rounds, probing) the way a
    // production sub-MIP does; interval-scheduled children stay bare.
    // The parent's active global cut rows, offered to heuristic children (set
    // once the root cut loop has settled; empty until then).
    // WHAT relaxation this is: column space, the exact matrix and row sides, the
    // column box, and the objective. Row count alone (or the optimal value) does
    // not identify it: two different cut systems can share both and still have
    // different points, bases and implications. The bound and the incumbent
    // target are attributes of a snapshot -- scheduling observations, not model
    // identity.
    using SnapshotIdentity = RelaxationIdentity;
    struct RootLpSnapshot {
        SnapshotIdentity id;
        bool valid = false;
        bool searched = false;                 // a root-primal pass has run on it
        f64 searched_target = std::numeric_limits<f64>::infinity();   // cutoff it ran under
        f64 bound = core::kNaN;
        f64 target = std::numeric_limits<f64>::infinity();
        std::vector<CutRow> context_rows;
        std::vector<Index> context_parent_rows;   // their row indices in the snapshot LP
        engines::SimplexBasis basis;              // proved basis of the snapshot LP
        std::vector<f64> x;                       // its point
        Index lp_rows = 0;
        model::LpProblem pump_model;
        bool has_pump_model = false;
    };
    RootLpSnapshot snapshot;
    std::vector<CutRow> sub_context_cuts;
    bool sub_mip_strengthen = false;
    // Objective-face child: feasibility heuristics stay on inside it even
    // when the parent has an incumbent (its objective row makes a point on
    // the face the whole goal).
    bool sub_mip_face = false;
    auto solve_sub_problem = [&](const model::LpProblem& sub, double budget,
                                 std::uint64_t node_budget,
                                 std::vector<f64>& x_sub,
                                 bool& exhausted) -> bool {
        exhausted = false;
        if (budget <= 0.05) return false;
        BabOptions so = opts;
        so.sub_mip_depth = opts.sub_mip_depth + 1;
        so.parent_invocation_id = diag.invocation_id;
        // The child solves a DIFFERENT, restricted problem -- its variable
        // space is not the caller's. It must never touch the portfolio pool:
        // a point that is feasible for the sub-MIP is validated against the
        // sub-MIP, so publishing it hands the parent a point that is not even
        // feasible for the real model. Measured on blend2: the portfolio
        // reported Optimal -33.0 against a true optimum of 7.598985, and it
        // survived BOTH the publish-side and consume-side validators because
        // both were checking against the sub-problem, not the original.
        //
        // Cancellation is dropped too: a child is a bounded heuristic, and a
        // rival arm winning the outer race must not abort it mid-repair.
        so.pool = nullptr;
        so.cancel = nullptr;
        // Neighborhood and objective-face restrictions may exclude the
        // parent's reference point. Such a point cannot validate cuts for
        // this child model. Retain the diagnostic only when it is feasible
        // in the child's unchanged column coordinates and restricted domain.
        if (so.cut_reference_point &&
            (sub.max_row_violation(*so.cut_reference_point) > opts.primal_feas_tol ||
             sub.max_bound_violation(*so.cut_reference_point) > opts.primal_feas_tol))
            so.cut_reference_point = nullptr;
        so.time_limit_s = budget;
        so.max_nodes = node_budget;
        so.verbose = false;
        // The child is a heuristic, so it drops everything whose value is in
        // PROVING rather than in finding: no cut loop, no reliability probes,
        // and above all no LNS of its own.
        so.sub_mip_lns = false;
        so.balans.enabled = false;
        so.kernel_pump.enabled = false;
        so.mrens.enabled = false;
        so.btbs.enabled = false;
        so.cl_tlns.enabled = false;
        so.cuts_enabled = sub_mip_strengthen;
        if (sub_mip_strengthen) so.cut.max_rounds = 5;
        // Explicit, not inherited: no separation inside the child's tree
        // either (it used to keep the parent's tree-cut settings by accident).
        so.tree_cut.enabled = false;
        so.reliability_branching = false;
        so.probing = sub_mip_strengthen;
        // A heuristic child must not pay for proof machinery. Measured on
        // easy60: sub-MIP children spent 5-16 s per run before their first
        // node (87% of child time on neos-860300 and mzzv42z, all of it on
        // dano3_3 and supportcase7) -- probing/clique-probing (10.8 s at
        // the top level of neos-860300), symmetry detection and implied-
        // integrality inference on a model that is mostly fixed. The
        // conflict graph was still built because conflict_propagation
        // requests it even with probing off. The root-strengthen and face
        // children keep these on: they get a real budget.
        if (!sub_mip_strengthen) {
            so.mip_presolve = false;
            so.symmetry = false;
            so.conflict_propagation = false;
            so.mip_pre.implied_integers = false;
        }
        so.clique_cuts = false;
        so.implied_bound_cuts = false;
        // The face child is a feasibility search on a tight objective face:
        // its LP point is near-optimal, so diving/rounding from it is the
        // natural way to land on the face.
        so.integer_dive = sub_mip_face;
        so.integer_neighborhood = false;
        // Feasibility Jump in the child ONLY while the parent has nothing.
        // With an incumbent in hand the LNS call is an improvement search, and
        // the child's own branch-and-bound is what does that -- measured, a
        // child FJ pass consumed most of a 0.6 s call and left it 5 nodes to
        // search with, which is why the sub-MIPs were doing nothing.
        so.feasibility_jump = !have_incumbent || sub_mip_face;
        so.feasibility_jump_time_s = std::min(0.2, budget * 0.2);
        so.feasibility_jump_seeded_time_s = std::min(0.2, budget * 0.2);
        so.feasibility_jump_improve_interval = 0;
        // The child is a heuristic; its own heuristic layer should not be
        // allowed to crowd out the little search it has room for.
        so.heuristic_budget_frac = 0.25;
        so.heuristic_budget_frac_no_incumbent = 0.5;
        so.lp.time_limit_s = budget;

        // Parent context: its global cut rows join the child model (valid for
        // every point of the original problem, so for any neighbourhood of it)
        // and its pseudocosts seed the child's branching.
        model::LpProblem sub_ctx_model;
        const model::LpProblem* sub_ptr = &sub;
        PseudocostSeed seed;
        engines::SimplexBasis child_basis;
        if (opts.sub_mip_context) {
            if (!sub_context_cuts.empty() && sub.n_cols() == n) {
                sub_ctx_model = sub;
                apply_cuts_inplace(sub_ctx_model, sub_context_cuts, cut_cfg.cut);
                sub_ptr = &sub_ctx_model;
                diag.sub_mip_context_cut_rows += sub_context_cuts.size();
            }
            // The parent's proved root basis, mapped onto the child's rows: the
            // model rows, any rows the neighbourhood added (their slacks basic),
            // then exactly the context rows that were appended.
            if (snapshot.valid && !snapshot.basis.basic.empty() && sub.n_cols() == n &&
                sub.n_rows() >= mip.n_rows() &&
                sub_ptr->n_rows() == sub.n_rows() + static_cast<Index>(
                    sub_ptr == &sub ? 0 : snapshot.context_parent_rows.size())) {
                const std::vector<Index> none;
                const auto mapped = map_basis_to_child(
                    snapshot.basis, n, snapshot.lp_rows, mip.n_rows(),
                    sub.n_rows() - mip.n_rows(),
                    sub_ptr == &sub ? none : snapshot.context_parent_rows, snapshot.x,
                    root_lo, root_hi);
                if (mapped) {
                    child_basis = *mapped;
                    so.initial_root_basis = &child_basis;
                }
            }
            seed.down_sum = pc_down_sum; seed.up_sum = pc_up_sum;
            seed.down_count = pc_down_count; seed.up_count = pc_up_count;
            so.pseudocost_seed = &seed;
            ++diag.sub_mip_context_seeded;
        }
        BabDiagnostics sd_sub;
        const core::RawResult sr = solve_milp(*sub_ptr, so, sd_sub);
        diag.sub_mip_nodes += sd_sub.nodes;
        diag.sub_mip_basis_carried += sd_sub.root_basis_carried;
        if (opts.sub_mip_depth == 0 && diag.sub_mip_calls < 6) {
            char b[128];
            std::snprintf(b, sizeof b, "setup %.0f ms, search %.0f ms, %llu nodes",
                          sd_sub.structural_presolve.ms + sd_sub.ms_before_search,
                          std::max(0.0, sd_sub.total_ms - sd_sub.structural_presolve.ms - sd_sub.ms_before_search),
                          static_cast<unsigned long long>(sd_sub.nodes));
            mark_milestone("child sub-MIP finished", b);
        }
        // Setup is everything before the child's first node pop, including
        // its structural presolve (which the child's own t0 excludes).
        ++diag.sub_mip_calls;
        const double child_setup_ms =
            sd_sub.structural_presolve.ms + sd_sub.ms_before_search;
        diag.sub_mip_child_setup_ms += child_setup_ms;
        diag.sub_mip_child_search_ms +=
            std::max(0.0, sd_sub.total_ms - child_setup_ms);
        if (sd_sub.total_ms - child_setup_ms >= 20.0) ++diag.sub_mip_children_searched;
        // "Exhausted" means the child settled the neighbourhood rather than
        // running out of budget -- a genuinely informative outcome, and one the
        // bandit scores above a plain miss.
        exhausted = sd_sub.globally_proved ||
                    sr.proposed_status == core::Status::Infeasible;
        if (sr.x.size() != sz(n)) return false;
        if (sr.proposed_status != core::Status::Optimal &&
            sr.proposed_status != core::Status::Feasible)
            return false;
        x_sub = sr.x;
        return true;
    };

    // Best solutions kept for the Crossover neighbourhood.
    const std::size_t pool_cap =
        milp_policy_is_latest(opts.policy) ? opts.balans.pool_size
                                           : opts.lns.pool_size;
    SolutionPool solution_pool(pool_cap);

    // Accepts a sub-MIP point as the incumbent if it checks out and improves.
    // Returns whether it became the new incumbent; a merely-feasible point is
    // still worth pooling, and the bandit scores that outcome differently.
    auto accept_sub_point = [&](std::vector<f64>& xs) -> bool {
        if (milp_point_max_violation(problem, xs, opts.int_tol) > opts.primal_feas_tol)
            return false;
        f64 o = problem.objective(xs);
        if (!std::isfinite(o)) return false;
        solution_pool.add(xs, o, problem.maximize);
        if (!accept_incumbent(xs, false, o)) return false;
        ++diag.heuristic_hits;
        {
            f64 dual_min = last_dual_bound;
            if (!open.empty() && std::isfinite(open.top().bound))
                dual_min = open.top().bound;
            refresh_live_gap(dual_min);
        }
        return true;
    };

    // Classical: AlnsScheduler over Neighborhood. Latest: Balans over
    // Neighborhood + KP/MRENS/FeasJump/BTBS/CL-TLNS. On small short-budget
    // MIPs Balans burns wall clock that Classical spends closing the dual
    // (gt2/lseu); keep Balans for larger models.
    const bool use_balans =
        milp_policy_is_latest(opts.policy) && opts.balans.enabled &&
        !(mip.n_cols() <= 400 && opts.time_limit_s > 0.0 &&
          opts.time_limit_s <= 60.0);
    AlnsScheduler alns(opts.lns);
    BalansScheduler balans(opts.balans);
    std::uint32_t lns_rng =
        use_balans ? opts.balans.seed : opts.lns.seed;
    std::uint64_t last_lns_node = 0;
    Clock::time_point last_lns_time = t0;
    std::uint64_t fpump_next_node = 1;
    std::uint64_t spp_next_node = 1;
    int face_attempts_done = 0;
    double face_next_start_s =
        opts.time_limit_s > 0.0 ? opts.objective_face_start_frac * opts.time_limit_s : 0.0;
    bool lns_root_done = false;
    std::uint64_t last_mrens_node = 0;
    // Up to three LP snapshots for MRENS (Lagromory stand-in).
    std::deque<std::vector<f64>> lp_snapshots;

    // Cold pre-tree run. This is the one that has to happen BEFORE the cut
    // loop: an instance whose root relaxation does not finish inside the whole
    // budget gets no other chance at a feasible point at all.
    // A restart's kept point, mapped into this stage. If it satisfies THIS model
    // (its reduced-cost box included) it is the incumbent. Usually it does not:
    // reduced-cost fixing excludes the incumbent's own neighbourhood by
    // construction. Then it is a HINT -- clamped into the box, it seeds the
    // pre-tree Feasibility Jump, which hunts a point better than the kept
    // objective from there instead of from scratch.
    std::vector<f64> restart_hint;
    if (opts.initial_solution != nullptr && opts.initial_solution->size() == sz(n) &&
        !have_incumbent) {
        std::vector<f64> x0 = *opts.initial_solution;
        f64 o0 = core::kNaN;
        if (accept_incumbent(x0, false, o0)) {
            ++diag.restart_incumbent_carried;
        } else {
            ++diag.restart_incumbent_rejected;
            for (Index j = 0; j < n; ++j) {
                f64 v = std::min(std::max(x0[sz(j)], root_lo[sz(j)]), root_hi[sz(j)]);
                if (!mip.is_integer.empty() && mip.is_integer[sz(j)]) {
                    v = std::round(v);
                    v = std::min(std::max(v, root_lo[sz(j)]), root_hi[sz(j)]);
                }
                x0[sz(j)] = v;
            }
            restart_hint = std::move(x0);
        }
    }
    if (opts.feasibility_jump) {
        double budget = opts.feasibility_jump_time_s;
        if (opts.time_limit_s > 0.0)
            budget = std::min({budget,
                               opts.feasibility_jump_root_frac * opts.time_limit_s,
                               std::max(0.0, seconds_left())});
        f64 hint_cutoff = core::kPosInf;
        if (!restart_hint.empty() && std::isfinite(opts.initial_cutoff)) {
            const f64 span = 1e-4 * (1.0 + std::fabs(opts.initial_cutoff));
            hint_cutoff = mip.maximize ? opts.initial_cutoff + span : opts.initial_cutoff - span;
        }
        if (try_feasjump(restart_hint.empty() ? nullptr : &restart_hint, budget, hint_cutoff) &&
            !restart_hint.empty())
            ++diag.restart_hint_hits;
    }

    // Same LP-free slot, before the root relaxation. No reference point exists
    // yet, which is deliberate: the rules that need one are late in the
    // portfolio and fall back to first-fail when it is absent.
    if (opts.fixprop && !have_incumbent) {
        double fp_budget = opts.fixprop_time_s;
        if (opts.time_limit_s > 0.0)
            fp_budget = std::min({fp_budget,
                                  opts.fixprop_root_frac * opts.time_limit_s,
                                  std::max(0.0, seconds_left())});
        try_fixprop(nullptr, fp_budget);
    }

    // GPU-track G1 (A2): a FALLBACK for a pure-binary model where the two
    // heuristics above found nothing, not a first attempt -- Feasibility
    // Jump alone finds an incumbent in well under 0.1s on most small
    // pure-binary models, so running this before it made every earlier
    // on/off timing comparison a placement artifact, not a result about the
    // heuristic. Eligibility (a presolve pass, no device) is checked before
    // any device is touched, so an ineligible model never pays for one.
    if (opts.gpu_binary_heuristic && !have_incumbent) {
        ++diag.gpu_bin_attempted;
        double budget = 2.0;
        if (opts.time_limit_s > 0.0)
            budget = std::min({budget, 0.1 * opts.time_limit_s,
                               std::max(0.0, seconds_left())});
        if (budget > 0.0 && binquad_milp_eligible(problem)) {
            ++diag.gpu_bin_eligible;
            std::lock_guard<std::mutex> gpu_lock(binquad_device_mutex());
            const auto gpu_init_t0 = Clock::now();
            backend::BinQuadDevice* device = shared_binquad_device_locked();
            diag.gpu_bin_init_ms += std::chrono::duration<double, std::milli>(
                Clock::now() - gpu_init_t0).count();
            if (device) {
                const auto gpu_bin_t0 = Clock::now();
                BinquadMilpHeuristicOptions bopts;
                bopts.bq.time_limit_s = budget;
                const BinquadMilpHeuristicResult r =
                    try_binquad_milp_heuristic(problem, bopts, *device);
                if (r.found) {
                    ++diag.gpu_bin_found;
                    f64 gpu_obj;
                    if (accept_incumbent(r.x, /*folded_space=*/false, gpu_obj) &&
                        opts.verbose)
                        std::printf("  [milp] gpu-binary incumbent %.10e at node %llu\n",
                                    best_incumbent,
                                    static_cast<unsigned long long>(diag.nodes));
                }
                diag.gpu_bin_ms += std::chrono::duration<double, std::milli>(
                    Clock::now() - gpu_bin_t0).count();
            }
        }
    }

    // Branch-and-Cut: a root-node cutting loop (Achterberg thesis 2007
    // Ch.3-4; Gomory Mixed-Integer separator, Ch.8.2-8.3 -- see
    // sor/search/cuts.hpp). Repeatedly solve the LP, separate cuts from the
    // optimal tableau, and reoptimize until integer-feasible, diminishing
    // returns, or the round cap. Under policy=latest, tree/local separation
    // continues inside the B&B loop (tree_cuts.hpp); local cuts stay on the
    // node path and never leak to siblings. Adding rows mid-tree invalidates
    // warm bases for that node (cold start), which is acceptable.
    //
    // GCS-v1 pool spans root + tree so multi-node violation stats accumulate
    // under content-stable fingerprints (not a full GNN).
    GcsPool gcs_pool;
    gcs_pool.pool_max = tree_cut_opts.gcs_pool_max;
    if (tree_cut_opts.gcs_enabled && !tree_cut_opts.gcs_model_path.empty()) {
        if (!load_gcs_model(tree_cut_opts.gcs_model_path, gcs_pool.model))
            gcs_pool.model.clear();
        else if (tree_cut_opts.gcs_prefer_heuristic)
            gcs_pool.prefer_heuristic = true;
        else
            gcs_pool.model.use_gnn =
                tree_cut_opts.gcs_use_gnn && gcs_pool.model.gnn.valid();
    }
    gcs_pool.prefer_heuristic =
        gcs_pool.prefer_heuristic || tree_cut_opts.gcs_prefer_heuristic;

    // Root bound dump for external re-derivation: the tightened root box is
    // what every cut at the root is derived from, so a cut can only be judged
    // against a point that survives it.
    detail::record_milp_bounds(search_problem, diag.invocation_id,
                       diag.parent_invocation_id, opts.sub_mip_depth,
                       "tightened_root_box", std::getenv("SOR_DUMP_ROOT_BOUNDS"),
                       root_lo, root_hi);
    // Freeze the protected prefix before root cut rounds or learned rows.
    cut_cfg.cut.original_rows = search_problem.n_rows();
    bool root_lp_outcome_logged = false;
    // The cut loop may already have proved the exact LP that the first tree
    // node would solve. Retain its point, dual evidence and basis together;
    // the root may consume them only if subsequent cuts, rollback, purge and
    // propagation left every LP coefficient and bound unchanged.
    struct ProvedRootLp {
        model::LpProblem problem;
        core::RawResult raw;
        engines::SimplexDiagnostics diag;
        engines::SimplexBasis basis;
        std::unique_ptr<engines::DualProbeSession> session;
    };
    std::optional<ProvedRootLp> proved_root_lp;
    struct UnfinishedRootLp {
        model::LpProblem problem;
        engines::SimplexBasis basis;
    };
    std::optional<UnfinishedRootLp> unfinished_root_lp;
    // Strongest global bound any root relaxation certified: a proved cut-loop
    // LP, the safe Lagrangian of one the loop could not finish, or a proved
    // depth-0 node LP. It seeds every root node below (the first root and a
    // restarted one), so an interrupted later relaxation of the same box can
    // never leave the root weaker than an earlier proof, and the pop-time
    // cutoff can prune a root the incumbent already matches.
    BoundCertificate root_cert;
    root_cert.scope = BoundCertificate::Scope::Global;
    // Wall time of the round-0 root LP: an upper-bound yardstick for what a
    // restricted sub-MIP of this model must spend just to solve its own LP.
    double root_lp_wall_s = 0.0;
    // Scheduling observes actual LP attempts, not imported result objects.
    // This ledger includes root/cut and tree relaxation work in the SAME
    // numerator, denominator and time budget. Reporting fields remain split.
    struct RelaxationWork {
        std::uint64_t solves = 0, iterations = 0;
        double ms = 0.0;
        void charge(const engines::SimplexDiagnostics& d) {
            ++solves;
            iterations += d.iterations;
        }
    } branching_lp_work;
    const f64 root_obj_granularity = objective_granularity(problem);
    const f64 root_offset_min = sense * problem.obj_offset;
    // Free integer columns of the root box before any reduced-cost fixing:
    // the base the restart trigger measures against.
    std::uint64_t root_free_ints_at_start = 0;
    for (Index j = 0; j < n; ++j)
        if (!mip.is_integer.empty() && mip.is_integer[sz(j)] &&
            root_hi[sz(j)] > root_lo[sz(j)])
            ++root_free_ints_at_start;
    // One feasibility-pump attempt from the LP point `x_lp` (no incumbent
    // assumed; the caller gates). Callable from the root LP snapshot as well
    // as from the node loop.
    // Set while a root primal pass runs: what the pass as a whole may still spend.
    std::optional<Clock::time_point> root_pass_deadline;
    const auto enclosing_left_s = [&]() {
        if (!root_pass_deadline) return std::numeric_limits<double>::infinity();
        return std::chrono::duration<double>(*root_pass_deadline - Clock::now()).count();
    };
    const auto try_fpump = [&](const std::vector<f64>& x_lp,
                               f64 cutoff_orig = std::numeric_limits<f64>::quiet_NaN(),
                               const model::LpProblem* pump_lp = nullptr) {
        const double budget = budget_allowance(
            opts.feasibility_pump_time_s,
            {opts.feasibility_pump_total_frac * opts.time_limit_s - diag.fpump_ms / 1000.0,
             0.3 * seconds_left(), enclosing_left_s()});
        if (budget > 0.2 && !heuristics_over_budget(budget * 1000.0)) {
            FeasPumpOptions fo;
            fo.time_limit_s = budget;
            fo.lp_time_limit_s = std::min(budget, std::max(0.05, budget / 3.0));
            fo.int_tol = opts.int_tol;
            fo.feas_tol = opts.primal_feas_tol;
            fo.seed = 20260930u + static_cast<std::uint32_t>(diag.fpump_attempts) * 7919u;
            fo.objective_cutoff = cutoff_orig;   // improvement mode when finite
            std::vector<f64> xp;
            FeasPumpDiagnostics fd;
            ++diag.fpump_attempts;
            if (pump_lp != nullptr) ++diag.fpump_on_snapshot;
            const bool ok = feasibility_pump(pump_lp != nullptr ? *pump_lp : mip, root_lo,
                                             root_hi, x_lp, fo, xp, fd);
            diag.fpump_rounds += static_cast<std::uint64_t>(fd.rounds);
            diag.fpump_lp_solves += fd.lp_solves;
            diag.fpump_flips += static_cast<std::uint64_t>(fd.flips);
            diag.fpump_restarts += static_cast<std::uint64_t>(fd.restarts);
            diag.fpump_ms += fd.ms;
            diag.heuristic_ms += fd.ms;
            if (ok) {
                f64 pobj;
                if (accept_incumbent(std::move(xp), false, pobj)) {
                    ++diag.fpump_hits;
                    ++diag.heuristic_hits;
                    if (opts.verbose)
                        std::printf("  [milp] feasibility pump incumbent %.10e "
                                    "(%d rounds, %llu LPs) at node %llu\n",
                                    best_incumbent, fd.rounds,
                                    static_cast<unsigned long long>(fd.lp_solves),
                                    static_cast<unsigned long long>(diag.nodes));
                }
            }
        }
    };
    // One SPP-repair attempt from `x_lp` (skipped unless unit-coefficient
    // binary rows are a tenth of the model).
    const auto try_spp_repair = [&](const std::vector<f64>& x_lp) {
        const auto st = detect_spp_structure(mip, root_lo, root_hi);
        const Index total_rows = mip.n_rows();
        if (st.rows() >= 5 && 10 * st.rows() >= total_rows) {
            const double budget = budget_allowance(
                opts.spp_repair_time_s,
                {opts.spp_repair_total_frac * opts.time_limit_s - diag.spp_ms / 1000.0,
                 0.2 * seconds_left(), enclosing_left_s()});
            if (budget > 0.05 && !heuristics_over_budget(budget * 1000.0)) {
                SppRepairOptions so;
                so.time_limit_s = budget;
                so.int_tol = opts.int_tol;
                so.feas_tol = opts.primal_feas_tol;
                so.seed = 20260930u + static_cast<std::uint32_t>(diag.spp_attempts) * 104729u;
                std::vector<f64> xr;
                SppRepairDiagnostics spp_d;
                ++diag.spp_attempts;
                const bool ok = spp_repair(mip, root_lo, root_hi, x_lp, so, xr, spp_d);
                diag.spp_moves += spp_d.moves;
                diag.spp_compound_moves += spp_d.compound_moves;
                diag.spp_ms += spp_d.ms;
                diag.heuristic_ms += spp_d.ms;
                if (ok) {
                    f64 sobj;
                    if (accept_incumbent(std::move(xr), false, sobj)) {
                        ++diag.spp_hits;
                        ++diag.heuristic_hits;
                        if (opts.verbose)
                            std::printf("  [milp] SPP repair incumbent %.10e (%llu moves) "
                                        "at node %llu\n", best_incumbent,
                                        static_cast<unsigned long long>(spp_d.moves),
                                        static_cast<unsigned long long>(diag.nodes));
                    }
                }
            }
        }
    };
    // One objective-face attempt against the certified bound `bound_now`
    // (minimisation sense). The caller decides whether now is the time; this
    // decides the target, the budget and whether the gap leaves anything to
    // find. Defined before the root cut loop so the root LP snapshot can feed it.
    const auto try_objective_face = [&](f64 bound_now) {
        const bool cut_known = cutoff_known();
        const f64 inc_min_now = known_cutoff_min();
        if (std::isfinite(bound_now) &&
            (!cut_known ||
             inc_min_now - bound_now > 1e-6 * (1.0 + std::fabs(inc_min_now)))) {
            const int k = face_attempts_done;
            static const double kWidth[] = {0.0, 0.02, 0.10, 0.30, 0.60, 0.90};
            const double width = kWidth[std::min(k, 5)];
            f64 target_min;
            if (k == 0 || !cut_known) {
                // At (just above) the proven bound; the next attainable
                // objective value when a lattice is known.
                target_min = tighten_bound_to_granularity(
                    bound_now, root_obj_granularity, root_offset_min, opts.int_tol);
                const f64 margin = root_obj_granularity > 0.0
                    ? 0.5 * root_obj_granularity
                    : 1e-6 * (1.0 + std::fabs(target_min));
                target_min += margin;
                if (!cut_known && k > 0)
                    target_min = bound_now + std::max(1.0, 0.02 * std::fabs(bound_now)) *
                                                 std::pow(3.0, k);
            } else {
                target_min = bound_now + width * (inc_min_now - bound_now);
            }
            if (!cut_known || target_min < inc_min_now) {
                const double budget = budget_allowance(
                    opts.objective_face_time_s * std::pow(2.0, k),
                    {0.25 * seconds_left(),
                     opts.objective_face_total_frac * opts.time_limit_s - diag.face_ms / 1000.0,
                     enclosing_left_s()});
                if (budget > 0.1 && !heuristics_over_budget(budget * 1000.0)) {
                    ++face_attempts_done;
                    ++diag.face_attempts;
                    NeighborhoodProblem np;
                    np.col_lo = root_lo;
                    np.col_hi = root_hi;
                    np.has_objective_cutoff = true;
                    np.objective_cutoff = sense * target_min;
                    const auto t_face = Clock::now();
                    const model::LpProblem sub = apply_neighborhood(mip, np);
                    std::vector<f64> xs;
                    bool exhausted = false;
                    sub_mip_strengthen = true;
                    sub_mip_face = true;
                    const bool got = solve_sub_problem(
                        sub, budget, /*node_budget=*/4000, xs, exhausted);
                    sub_mip_strengthen = false;
                    sub_mip_face = false;
                    const double spent = ms_since(t_face);
                    diag.face_ms += spent;
                    diag.sub_mip_ms += spent;
                    if (got && accept_sub_point(xs)) {
                        ++diag.face_hits;
                        if (opts.verbose)
                            std::printf("  [milp] objective-face incumbent %.10e "
                                        "(target %.6e, attempt %d)\n",
                                        best_incumbent, sense * target_min, k + 1);
                    } else if (exhausted) {
                        ++diag.face_exhausted;
                    }
                    face_next_start_s = ms_since(t0) / 1000.0 +
                                        std::max(1.0, 2.0 * spent / 1000.0);
                }
            }
        }

    };
    // Primal work from a root LP snapshot: the first proved root relaxation,
    // and later after a productive cut batch. It used to wait until the root
    // node had finished its own LP after the whole cut loop, and on models
    // where those LPs are expensive the heuristic budget was gone by then
    // (neos-827175: cut loop 21 s, node LP 35 s, zero pump / neighborhood /
    // face attempts, incumbent 122 against an optimum equal to the 112 bound).
    // While there is no incumbent: pump and set-partitioning repair. Whenever
    // the certified bound leaves a gap: an objective-face attempt.
    f64 root_primal_last_bound = core::kNaN;
    bool root_primal_last_had_incumbent = false;
    double root_primal_last_s = -1e9;
    // Early passes (while the cut loop is still improving the relaxation) may
    // spend at most half of the root-primal allowance; the other half is
    // reserved for one pass on the FINAL proved root relaxation, which is the
    // snapshot every later phase (child contexts, tree heuristics) works from.
    // ---- Root LP snapshot -------------------------------------------------
    // One object per proved root relaxation: the point, its certified bound, the
    // incumbent target it was published under, and the state its consumers need
    // -- the bounded dual-active cut rows offered to heuristic children, and the
    // model (original rows + those cut rows) the feasibility pump projects onto.
    // Published BEFORE the root-primal pass that uses it, so every consumer
    // works from the same relaxation. Identity hashes the columns, exact
    // matrix/row sides, root box and objective; bound and target are attributes.
    // Publishes `lp`/`raw` (a proved root relaxation whose row count and
    // multipliers agree) unless that identity is already the current snapshot.
    // A child is a restricted copy of the model, and handing it every cut row
    // (1500 on a 125-row model, pg) made each child LP heavier than the search it
    // served, so only the dual-active rows go: nonzero price, strongest first, at
    // most a quarter of the model's own row count (never fewer than 20).
    const auto publish_root_snapshot = [&](const model::LpProblem& lp,
                                           const core::RawResult& rlp, f64 bound_min,
                                           const engines::SimplexBasis* lp_basis = nullptr) -> bool {
        if (opts.sub_mip_depth != 0 || lp.n_cols() != n ||
            static_cast<Index>(rlp.x.size()) != n)
            return false;
        const SnapshotIdentity id = relaxation_identity(lp, root_lo, root_hi);
        if (snapshot.valid && snapshot.id == id) {
            // Same relaxation: the point, basis and bound are refreshed as
            // attributes; nothing derived from the model (context, pump model) is.
            snapshot.bound = bound_min;
            snapshot.target = known_cutoff_min();
            snapshot.x = rlp.x;
            if (lp_basis != nullptr && static_cast<Index>(lp_basis->basic.size()) == lp.n_rows())
                snapshot.basis = *lp_basis;
            return false;
        }
        const bool had_rows = !snapshot.context_rows.empty();
        snapshot = RootLpSnapshot{};
        snapshot.id = id;
        snapshot.bound = bound_min;
        snapshot.target = known_cutoff_min();
        snapshot.valid = true;
        snapshot.lp_rows = lp.n_rows();
        snapshot.x = rlp.x;
        if (lp_basis != nullptr && static_cast<Index>(lp_basis->basic.size()) == lp.n_rows())
            snapshot.basis = *lp_basis;
        ++diag.root_snapshots;
        if (opts.sub_mip_context && lp.n_rows() > mip.n_rows()) {
            const auto& crp = lp.A.pattern.row_ptr();
            const auto& cci = lp.A.pattern.col_idx();
            std::vector<std::pair<f64, Index>> ranked;   // (|dual|, row)
            const bool have_duals = rlp.y.size() == sz(lp.n_rows());
            for (Index r = mip.n_rows(); r < lp.n_rows(); ++r) {
                const f64 w = have_duals ? std::fabs(rlp.y[sz(r)]) : 0.0;
                if (w > 1e-9) ranked.emplace_back(w, r);
            }
            std::stable_sort(ranked.begin(), ranked.end(),
                             [](const auto& x, const auto& y) { return x.first > y.first; });
            const std::size_t cap = std::max<std::size_t>(20, sz(mip.n_rows()) / 4);
            if (ranked.size() > cap) ranked.resize(cap);
            std::size_t budget_nnz = 200000;
            for (const auto& [w, r] : ranked) {
                (void)w;
                const std::size_t len = sz(crp[sz(r) + 1] - crp[sz(r)]);
                if (len > budget_nnz) continue;
                budget_nnz -= len;
                CutRow row;
                for (core::Offset k = crp[sz(r)]; k < crp[sz(r) + 1]; ++k) {
                    row.cols.push_back(cci[sz(k)]);
                    row.vals.push_back(lp.A.vals[sz(k)]);
                }
                row.row_lo = lp.row_lo[sz(r)];
                row.row_hi = lp.row_hi[sz(r)];
                row.name = "ROOTCUT";
                snapshot.context_rows.push_back(std::move(row));
                snapshot.context_parent_rows.push_back(r);
            }
            if (!snapshot.context_rows.empty()) {
                snapshot.pump_model = mip;
                apply_cuts_inplace(snapshot.pump_model, snapshot.context_rows, cut_cfg.cut);
                snapshot.has_pump_model = true;
            }
        }
        // The child context follows the snapshot (replaced, or cleared when the
        // new snapshot has none).
        if (had_rows || !snapshot.context_rows.empty()) ++diag.snapshot_child_refreshes;
        sub_context_cuts = snapshot.context_rows;
        diag.snapshot_context_rows = snapshot.context_rows.size();
        return true;
    };
    const auto root_primal_pass = [&](const model::LpProblem& snap_lp,
                                      const core::RawResult& snap_raw, f64 bound_min,
                                      const engines::SimplexBasis* snap_basis,
                                      bool final_snapshot = false) {
        const std::vector<f64>& x_lp = snap_raw.x;
        // Context first, from the relaxation that supplies the point and bound.
        if (publish_root_snapshot(snap_lp, snap_raw, bound_min, snap_basis)) {
            char b[128];
            std::snprintf(b, sizeof b, "id=%llu bound=%.10g context_rows=%zu%s",
                          static_cast<unsigned long long>(diag.root_snapshots), bound_min,
                          snapshot.context_rows.size(), final_snapshot ? " (final)" : "");
            mark_milestone("root snapshot published", b);
        }
        if (!opts.root_primal_early || opts.sub_mip_depth != 0 || opts.time_limit_s <= 0.0 ||
            timed_out() || static_cast<Index>(x_lp.size()) != n)
            return;
        // One pass per (relaxation, cutoff): the same relaxation is searched again
        // only when the cutoff it was searched under has since improved.
        const f64 now_cut = known_cutoff_min();
        const bool cutoff_improved =
            std::isfinite(now_cut) &&
            (!std::isfinite(snapshot.searched_target) ||
             now_cut < snapshot.searched_target - 1e-9 * (1.0 + std::fabs(snapshot.searched_target)));
        if (snapshot.searched && !cutoff_improved) return;
        snapshot.searched = true;
        snapshot.searched_target = known_cutoff_min();
        const double share = final_snapshot ? 1.0 : 0.5;
        if (diag.root_primal_ms > share * opts.root_primal_total_frac * opts.time_limit_s * 1000.0)
            return;
        const auto t_pass = Clock::now();
        ++diag.root_primal_passes;
        root_pass_deadline = t_pass + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(std::max(
                0.0, opts.root_primal_total_frac * opts.time_limit_s - diag.root_primal_ms / 1000.0)));
        const bool had_incumbent = have_incumbent;
        // Nothing feasible for this stage yet: a feasibility pump, and when a
        // cutoff is known (a restart's kept point) it must beat that cutoff --
        // a merely feasible point would be pruned on arrival.
        if (!have_incumbent && opts.feasibility_pump_root &&
            diag.fpump_ms <= opts.feasibility_pump_total_frac * opts.time_limit_s * 1000.0) {
            fpump_next_node = std::max<std::uint64_t>(fpump_next_node, 100);
            f64 pump_cut = std::numeric_limits<f64>::quiet_NaN();
            if (cutoff_known()) {
                const f64 step = root_obj_granularity > 0.0 ? root_obj_granularity
                    : 1e-6 * (1.0 + std::fabs(known_cutoff_orig()));
                pump_cut = known_cutoff_orig() - sense * step;
            }
            try_fpump(x_lp, pump_cut,
                      snapshot.has_pump_model ? &snapshot.pump_model : nullptr);
        }
        // With an incumbent and an open gap: an improvement pump (objective row
        // at the incumbent, minus one lattice step or a hair).
        if (have_incumbent && opts.feasibility_pump_root && opts.feasibility_pump_improve &&
            diag.fpump_ms <= opts.feasibility_pump_total_frac * opts.time_limit_s * 1000.0 &&
            std::isfinite(bound_min) &&
            known_cutoff_min() - bound_min > 1e-4 * (1.0 + std::fabs(known_cutoff_orig()))) {
            const f64 step = root_obj_granularity > 0.0 ? root_obj_granularity
                                                        : 1e-6 * (1.0 + std::fabs(known_cutoff_orig()));
            try_fpump(x_lp, known_cutoff_orig() - sense * step,
                      snapshot.has_pump_model ? &snapshot.pump_model : nullptr);
        }
        if (!have_incumbent && opts.spp_repair &&
            diag.spp_ms <= opts.spp_repair_total_frac * opts.time_limit_s * 1000.0) {
            spp_next_node = std::max<std::uint64_t>(spp_next_node, 50);
            try_spp_repair(x_lp);
        }
        if (opts.objective_face && opts.sub_mip_lns &&
            face_attempts_done < opts.objective_face_max_attempts &&
            diag.face_ms <= opts.objective_face_total_frac * opts.time_limit_s * 1000.0 &&
            !timed_out())
            try_objective_face(bound_min);
        if (!had_incumbent && have_incumbent) ++diag.root_primal_hits;
        root_pass_deadline.reset();
        diag.root_primal_ms += ms_since(t_pass);
        root_primal_last_bound = bound_min;
        root_primal_last_had_incumbent = have_incumbent;
        root_primal_last_s = ms_since(t0) / 1000.0;
    };
    const Index mip_rows_before_cuts = search_problem.n_rows();
    if (opts.cuts_enabled) {
        const auto t_cutloop = Clock::now();
        CutDiagnostics cut_diag;
        CutPool cut_pool(cut_cfg.cut);
        cut_pool.set_scoring_context(search_problem);
        f64 prev_bound = core::kNaN;
        f64 mir_last_bound = core::kNaN;   // bound at the previous MIR separation (min sense)
        int stalled_rounds = 0;
        // Rows the model had before any cut round. Every cut row is a
        // contiguous suffix above this, which is what lets a scattered purge
        // be done as truncate-then-reappend.
        const Index rows_before_cuts = search_problem.n_rows();
        // The last root LP the loop actually PROVED, kept so a purge can be
        // judged against real multipliers. Rows appended after it were never
        // priced and are never purged on its evidence.
        std::vector<f64> last_solved_y, last_solved_x;
        Index last_solved_rows = rows_before_cuts;
        // Rounds appended since the last REALISED bound improvement, newest
        // last. Retraction is strict LIFO (each undo record describes the edit
        // made to the model as it stood at that moment), so this is a stack.
        struct PendingRound {
            CutUndo undo;
            std::vector<CutRow> cuts;
            int trace_index = -1;
        };
        std::vector<PendingRound> unpaid_rounds;
        // Cut-application transactions still awaiting an LP verdict: every round
        // applied since the last proved root LP, newest last. A round is
        // COMMITTED when the LP after it is proved; if the loop ends first (round
        // or time cap, an unproved/cycling LP) the batch is DEFERRED, and the
        // root-to-tree transition rolls it back instead of inheriting rows whose
        // relaxation nobody solved. Certified bounds already raised (including a
        // Lagrangian bound from the unproved LP) are independent evidence and stay.
        std::vector<CutUndo> pending_batches;
        // Filled at the top of each round so the round that APPLIED cuts is
        // the one credited with the gain they realised.
        int last_applied_trace = -1;
        // Which separator emitted a cut, read off the name the separator
        // stamped on it. Index 0 (GMI) is the base family and is never gated.
        enum CutFamily { kFamGmi = 0, kFamMir, kFamCover, kFamClique,
                         kFamVub, kFamZeroHalf, kFamCount };
        auto family_of = [](const CutRow& c) -> int {
            if (c.name.rfind("MIR_", 0) == 0) return kFamMir;
            if (c.name.rfind("COV_", 0) == 0 ||
                c.name.rfind("COVPC_", 0) == 0 ||
                c.name.rfind("COVGNS_", 0) == 0) return kFamCover;
            if (c.name.rfind("CLQ_", 0) == 0) return kFamClique;
            if (c.name.rfind("VUB_", 0) == 0) return kFamVub;
            if (c.name.rfind("ZH_", 0) == 0) return kFamZeroHalf;
            return kFamGmi;   // GMI and anything unrecognised
        };
        bool family_off[kFamCount] = {false, false, false, false, false, false};
        bool marginal_gate_done = false;
        // Warm continuation across cut rounds: cuts only ADD ROWS, so the
        // previous round's basis extends naturally (each new row's logical
        // basic in that row) and the dual re-optimizes from it in a handful
        // of pivots instead of a full cold solve. Measured on
        // schedule_milp HUGE: each cold round cost ~28s; the warm rounds
        // are near-free. A failed warm solve falls back to the cold path.
        engines::SimplexBasis& prior_basis = cut_loop_basis;
        bool& have_prior = have_cut_loop_basis;
        auto cut_loop_t0 = Clock::now();
        for (int round = 0; round < cut_cfg.cut.max_rounds; ++round) {
            if (timed_out()) break;
            // Round 0 is the root relaxation itself, not optional cut work:
            // the search must prove this LP before it can bound, fix by
            // reduced cost or prune anything, and a proved round-0 solve is
            // handed to the root node unchanged (root_lp_reuses). Charging it
            // to the cut allowance only interrupted it after setup phases had
            // spent that allowance (ex9: 512 pivots, physiciansched6-2: 1792,
            // neos-827175: 8819 of the 13027 a standalone dual needs), and
            // the node then re-solved from a stale hand-off until the time
            // limit. The caps below still bound every actual cut round.
            const bool root_relaxation_round = round == 0;
            if (!root_relaxation_round && opts.time_limit_s > 0.0 &&
                (std::chrono::duration<double>(Clock::now() - cut_loop_t0)
                         .count() > root_cut_allowance_s ||
                 std::chrono::duration<double>(Clock::now() - t0).count() >
                     root_total_allowance_s)) {
                ++diag.cut_loop_time_capped;
                break;
            }
            engines::SimplexOptions cut_lp_opts = opts.lp;
            // separate_gomory_mi() reconstructs the basis matrix from the
            // ORIGINAL row/col bounds and reads AtLower/AtUpper against
            // them. A presolved solve can lift a basis whose nonbasic
            // statuses reflect a presolve-tightened bound (e.g. a singleton
            // row folded into a column bound), which would silently
            // mismatch lp.col_lo/col_hi here. Solve unpresolved so the
            // returned SimplexBasis is exactly in this function's bound
            // space.
            cut_lp_opts.presolve = false;
            if (opts.time_limit_s > 0.0) {
                // Bound by the CUT loop's own allowance, not by the global
                // limit. Giving each round the whole remaining solve time is
                // why the round-boundary cap never fired: on atlanta-ip one
                // round's LP re-solve costs ~10 s, so two rounds consumed
                // 21.4 s of a 30 s budget without ever crossing the boundary
                // check, and the search branched zero nodes.
                const double global_left = opts.time_limit_s -
                    std::chrono::duration<double>(Clock::now() - t0).count();
                const double cut_left = root_cut_allowance_s -
                    std::chrono::duration<double>(Clock::now() - cut_loop_t0)
                        .count();
                // ...and by what the whole pre-search phase has left, so the
                // setup phases' spending actually subtracts from the cut
                // loop's allowance instead of sitting beside it.
                const double root_left = root_total_allowance_s -
                    std::chrono::duration<double>(Clock::now() - t0).count();
                const double left = root_relaxation_round
                    ? global_left
                    : std::min(global_left, std::min(cut_left, root_left));
                if (left <= 0.02) {
                    if (cut_left <= 0.02) ++diag.cut_loop_time_capped;
                    break;
                }
                cut_lp_opts.time_limit_s = left;
            }
            // Same incumbent cutoff as node LPs: a root LP whose multipliers
            // already certify the incumbent's value ends the whole search.
            if (opts.node_lp_cutoff && cutoff_known())
                cut_lp_opts.objective_limit =
                    node_cutoff(known_cutoff_min(), root_obj_granularity,
                                root_offset_min, opts.gap_tol, opts.abs_gap_tol);
            engines::SimplexDiagnostics cut_sd;
            engines::SimplexBasis cut_basis;
            std::unique_ptr<engines::DualProbeSession> cut_session;
            auto* session_out = &cut_session;
            core::RawResult cut_lp_raw;
            bool warm_used = false;
            bool skip_cold_retry = false;
            const auto t_round_lp = Clock::now();
            if (have_prior) {
                // Row extension: ns is unchanged and old indices are stable
                // (apply_cuts appends rows only).
                engines::SimplexBasis ext = prior_basis;
                const Index ns_ = search_problem.n_cols();
                const Index m_old =
                    static_cast<Index>(prior_basis.basic.size());
                const Index m_new = search_problem.n_rows();
                if (m_new >= m_old &&
                    static_cast<Index>(ext.status.size()) == ns_ + m_old) {
                    ext.status.resize(sz(ns_ + m_new));
                    for (Index i = m_old; i < m_new; ++i) {
                        ext.basic.push_back(ns_ + i);
                        ext.status[sz(ns_ + i)] = engines::NonbasicStatus::Basic;
                    }
                    ++diag.cut_lp_warm_attempts;
                    cut_lp_raw = engines::solve_dual_simplex(
                        search_problem, cut_lp_opts, cut_sd, &cut_basis, &ext,
                        nullptr, nullptr, session_out);
                    ++diag.lp_solves;
                    diag.cut_lp_iterations += cut_sd.iterations;
                    branching_lp_work.charge(cut_sd);
                    // A cutoff stop is final: re-solving the round cold would
                    // only repeat the pivots that reached the cutoff.
                    warm_used = relaxation_proved(cut_lp_raw, cut_sd, cut_lp_opts) ||
                                cut_lp_raw.termination_reason == "objective limit";
                    if (warm_used) ++diag.cut_lp_warm_hits;
                }
            }
            if (!warm_used) {
                if (have_prior) {
                    // The warm dual failed to prove; redo the round cold.
                    // (The wasted warm attempt is charged to lp_solves.) The
                    // cold retry gets what is LEFT of the round's deadline, not
                    // a second full allowance.
                    if (cut_lp_opts.time_limit_s > 0.0) {
                        const double rem =
                            cut_lp_opts.time_limit_s - ms_since(t_round_lp) / 1000.0;
                        if (rem <= 0.0) skip_cold_retry = true;   // nothing left: no new allowance
                        else cut_lp_opts.time_limit_s = rem;
                    }
                    if (!skip_cold_retry) {
                        cut_sd = engines::SimplexDiagnostics{};
                        cut_basis = engines::SimplexBasis{};
                    }
                }
                if (!skip_cold_retry) {
                    cut_lp_raw = engines::solve_simplex(
                        search_problem, cut_lp_opts, cut_sd, &cut_basis, session_out);
                    ++diag.lp_solves;
                    diag.cut_lp_iterations += cut_sd.iterations;
                    branching_lp_work.charge(cut_sd);
                }
            }
            const double round_lp_ms = ms_since(t_round_lp);
            diag.cut_lp_ms += round_lp_ms;
            branching_lp_work.ms += round_lp_ms;
            if (round == 0) root_lp_wall_s = ms_since(t_round_lp) / 1000.0;
            if (!relaxation_proved(cut_lp_raw, cut_sd, cut_lp_opts)) {
                char pbuf[280];
                std::snprintf(pbuf, sizeof pbuf,
                              "\"round\":%d,\"status\":%d,"
                              "\"iterations\":%llu,\"phase\":%d,"
                              "\"basis_rows\":%zu,\"primal_residual\":%.9g,"
                              "\"dual_residual\":%.9g,\"gap_rel\":%.9g",
                              round, static_cast<int>(cut_lp_raw.proposed_status),
                              static_cast<unsigned long long>(cut_sd.iterations),
                              cut_sd.final_phase, cut_basis.basic.size(),
                              cut_sd.primal_residual, cut_sd.dual_residual,
                              cut_sd.gap_rel);
                SOR_ROUTE(1, "bab", "root_cut_lp_unproved", pbuf);
                if (opts.verbose)
                    std::printf("  [milp] root cut round %d LP unproved (%s, "
                                "%llu iterations, warm %d, primal res %.3g, "
                                "dual res %.3g, gap %.3g, dual bound %s)\n",
                                round, cut_lp_raw.termination_reason.c_str(),
                                static_cast<unsigned long long>(cut_sd.iterations),
                                warm_used ? 1 : 0, cut_sd.primal_residual,
                                cut_sd.dual_residual, cut_sd.gap_rel,
                                cut_sd.dual_bound_finite ? "finite" : "none");
                if (cut_lp_raw.y.size() == sz(search_problem.n_rows())) {
                    std::vector<f64> y_min(cut_lp_raw.y.size());
                    for (std::size_t i = 0; i < y_min.size(); ++i)
                        y_min[i] = sense * cut_lp_raw.y[i];
                    const auto safe = certify::safe_lagrangian_lower_bound(
                        search_problem, y_min, search_problem.col_lo,
                        search_problem.col_hi);
                    ++diag.lagrangian_bound_checks;
                    if (safe.finite &&
                        root_cert.raise(safe.value,
                                        BoundCertificate::Source::Lagrangian))
                        ++diag.lagrangian_bound_raises;
                }
                // An interrupted phase-2 dual solve may still have made
                // thousands of useful pivots. Its basis is only a STARTING
                // HINT, never a node bound or infeasibility proof. The node
                // dual simplex rechecks feasibility and optimality itself.
                if (cut_lp_raw.proposed_status == core::Status::Interrupted &&
                    cut_sd.final_phase == 2 &&
                    static_cast<Index>(cut_basis.basic.size()) ==
                        search_problem.n_rows() &&
                    static_cast<Index>(cut_basis.status.size()) ==
                        search_problem.n_cols() + search_problem.n_rows()) {
                    unfinished_root_lp = UnfinishedRootLp{
                        search_problem, std::move(cut_basis)};
                }
                break;
            }
            proved_root_lp = ProvedRootLp{
                search_problem, cut_lp_raw, cut_sd, cut_basis, std::move(cut_session)};
            diag.cut_batches_committed += pending_batches.size();
            pending_batches.clear();
            // Every cut row is globally valid and the box is the root box, so
            // this proved bound holds for the whole problem -- and keeps
            // holding if a later round is interrupted or rolled back.
            if (root_cert.raise(node_lp_bound_min(cut_lp_raw, sense),
                                BoundCertificate::Source::LpOptimal))
                ++diag.root_bound_raises;
            // Hand this proved basis to the next round (and, after the loop,
            // to the root node). See CutOptions::warm_start_rounds: without
            // this assignment both warm paths are unreachable.
            if (cut_cfg.cut.warm_start_rounds && !cut_basis.basic.empty() &&
                static_cast<Index>(cut_basis.basic.size()) ==
                    search_problem.n_rows()) {
                prior_basis = cut_basis;
                have_prior = true;
            }
            // Reduced-cost fixing against the incumbent, every proved round.
            // Fixings shrink the box the next round's LP and separators work
            // in, which is what lets later rounds find stronger cuts; waiting
            // for the root node left the cut loop cutting over columns the
            // incumbent had already ruled out. Only points worse than the
            // incumbent are excluded, exactly as by the root RC pass below.
            if (opts.reduced_cost_strengthening && have_incumbent &&
                cut_lp_raw.y.size() == sz(search_problem.n_rows())) {
                const auto rc = certified_reduced_cost_bounds(
                    search_problem, cut_lp_raw.y, best_incumbent, root_lo,
                    root_hi, opts.primal_feas_tol);
                diag.rc_certificate_checks += rc.checked;
                diag.rc_strengthening_ms += rc.ms;
                if (rc.tightened > 0) {
                    ++diag.rc_strengthen_nodes;
                    diag.rc_bounds_tightened += rc.tightened;
                    diag.rc_columns_fixed += rc.fixed;
                    search_problem.col_lo = root_lo;
                    search_problem.col_hi = root_hi;
                }
            }
            // Certified finite root relaxation: all descendant LPs are bounded.
            root_relaxation_bounded = true;
            {
                const f64 bnow = node_lp_bound_min(cut_lp_raw, sense);
                const double now_s = ms_since(t0) / 1000.0;
                const bool productive =
                    std::isfinite(root_primal_last_bound) && std::isfinite(bnow) &&
                    bnow - root_primal_last_bound >=
                        0.05 * std::max(1.0, std::fabs(root_primal_last_bound)) &&
                    now_s - root_primal_last_s >= 2.0;
                if (round == 0 || productive) {
                    // Primal work is not cut work: the cut loop's own allowance
                    // must not be spent by it.
                    const auto t_pp = Clock::now();
                    root_primal_pass(search_problem, cut_lp_raw, bnow, &cut_basis);
                    cut_loop_t0 += Clock::now() - t_pp;
                }
            }
            if (round == 0) diag.root_bound_before_cuts = cut_lp_raw.objective;
            diag.root_bound_after_cuts = cut_lp_raw.objective;
            if (cut_cfg.cut.purge_nonbinding_cuts &&
                static_cast<Index>(cut_lp_raw.y.size()) ==
                    search_problem.n_rows()) {
                last_solved_y = cut_lp_raw.y;
                last_solved_x = cut_lp_raw.x;
                last_solved_rows = search_problem.n_rows();
            }

            bool integer_ok = true;
            for (Index j = 0; j < search_problem.n_cols(); ++j) {
                if (!search_problem.is_integer.empty() &&
                    search_problem.is_integer[sz(j)] &&
                    !is_integral(cut_lp_raw.x[sz(j)], opts.int_tol)) {
                    integer_ok = false;
                    break;
                }
            }
            if (integer_ok) {
                char rbuf[96];
                std::snprintf(rbuf, sizeof rbuf, "\"obj\":%.10g,\"integer\":true",
                              cut_lp_raw.objective);
                SOR_ROUTE(1, "bab", "root_lp_outcome", rbuf);
                root_lp_outcome_logged = true;
                break;
            }

            diag.cut_round_trace.push_back({});
            {
                auto& tr = diag.cut_round_trace.back();
                tr.round = round;
                tr.bound = cut_lp_raw.objective;
                tr.lp_iterations = cut_sd.iterations;
                tr.refactorizations = cut_sd.refactorizations;
                tr.lp_ms = cut_sd.total_ms;
            }
            if (round > 0 && std::isfinite(prev_bound)) {
                const f64 gain = std::fabs(cut_lp_raw.objective - prev_bound);
                const f64 scale = 1.0 + std::fabs(prev_bound);
                const f64 gain_rel = gain / scale;
                // Credit the gain to the round that ADDED the cuts, not to the
                // round that observed it. A round whose gain_rel stays NaN is
                // one the loop exited before ever re-solving -- not a round
                // that stalled, a round nobody measured -- and the rollback
                // below relies on being able to tell those apart.
                if (last_applied_trace >= 0 &&
                    last_applied_trace <
                        static_cast<int>(diag.cut_round_trace.size()))
                    diag.cut_round_trace[sz(last_applied_trace)].gain_rel =
                        gain_rel;
                if (gain_rel < cut_cfg.cut.min_progress_rel) {
                    // Tolerate a run of stalled rounds before giving up: a
                    // round that gains little often exposes structure the next
                    // one exploits. See CutOptions::min_progress_patience.
                    if (++stalled_rounds >= cut_cfg.cut.min_progress_patience)
                        break;
                } else {
                    stalled_rounds = 0;
                    // This gain justifies every round still on the stack: the
                    // dip rounds are what exposed the structure it exploits
                    // (the gt2 lesson in CutOptions::min_progress_patience).
                    unpaid_rounds.clear();
                }
            }
            prev_bound = cut_lp_raw.objective;

            cut_pool.start_round(cut_diag);
            // DynSep-v1: choose which generators run this round (Latest).
            // Classical / --no-dynsep: static flags only via decide().
            DynSepRoundInput dsin;
            dsin.round = round;
            dsin.depth = 0;
            dsin.at_root = true;
            dsin.last_bound_gain_rel = dynsep_last_gain;
            dsin.force_mir = cut_cfg.mir_cuts;
            dsin.force_cover = cut_cfg.lifted_cover_cuts;
            dsin.force_clique = cut_cfg.clique_cuts;
            dsin.force_ib = opts.implied_bound_cuts;
            dsin.force_zerohalf = cut_cfg.zerohalf_cuts;
            dsin.force_flowcover = cut_cfg.flow_cover_cuts;
            dsin.have_conflict_graph = !conflict_graph.empty();
            dsin.have_basis = !cut_basis.basic.empty();
            dsin.gap_rel = diag.gap_rel;
            {
                Index n_int = 0, n_frac = 0;
                f64 mean_frac = 0.0;
                for (Index j = 0; j < search_problem.n_cols(); ++j) {
                    if (search_problem.is_integer.empty() ||
                        !search_problem.is_integer[sz(j)])
                        continue;
                    ++n_int;
                    const f64 xv = cut_lp_raw.x[sz(j)];
                    const f64 f = xv - std::floor(xv);
                    const f64 frac = std::min(f, 1.0 - f);
                    if (frac > opts.int_tol) {
                        mean_frac += frac;
                        ++n_frac;
                    }
                }
                if (n_frac > 0) mean_frac /= static_cast<f64>(n_frac);
                dsin.frac_share =
                    n_int > 0 ? static_cast<f64>(n_frac) / static_cast<f64>(n_int)
                              : 0.0;
                dsin.mean_frac = mean_frac;
            }
            DynSepDecision dsd = dynsep.decide(dsin);
            // A family the marginal gate switched off is not merely filtered
            // out of the selection -- it is not run at all, so the separation
            // cost goes too. That saving is the point of gating over purging.
            if (family_off[kFamMir]) dsd.run_mir = false;
            if (family_off[kFamCover]) dsd.run_cover = false;
            if (family_off[kFamClique]) dsd.run_clique = false;
            if (family_off[kFamVub]) dsd.run_ib = false;
            if (family_off[kFamZeroHalf]) dsd.run_zerohalf = false;

            const auto t_separate = Clock::now();
            std::vector<CutRow> candidates;
            if (dsd.run_gmi) {
                CutOptions gmi_opts = cut_cfg.cut;
                gmi_opts.max_cuts_per_round = dsd.budget_gmi;
                candidates = separate_gomory_mi(
                    search_problem, cut_lp_raw.x, cut_basis, gmi_opts, cut_diag);
            }
            // Clique cuts join the same pool as the Gomory candidates, so
            // they compete on efficacy and go through the same duplicate,
            // dominance, and parallelism filters. They are separated from the
            // LP POINT alone -- no tableau, no basis -- which is why they can
            // be generated on the same rounds without any of the basis-space
            // caveats above.
            if (dsd.run_clique && !conflict_graph.empty()) {
                auto clique = separate_clique_cuts(conflict_graph,
                                                   cut_lp_raw.x, opts.probe);
                if (static_cast<int>(clique.size()) > dsd.budget_clique)
                    clique.resize(static_cast<std::size_t>(dsd.budget_clique));
                diag.clique_cut_candidates += clique.size();
                candidates.insert(candidates.end(),
                                  std::make_move_iterator(clique.begin()),
                                  std::make_move_iterator(clique.end()));
            }
            if (dsd.run_cover) {
                CoverOptions cov_opts = cut_cfg.cover;
                cov_opts.max_cuts = dsd.budget_cover;
                // DynSep cover arm: enable PC sequence-independent lifting
                // under Latest (Prasad / arXiv:2401.13773).
                if (milp_policy_is_latest(opts.policy))
                    cov_opts.pc_lift_hooks = true;
                auto cov = separate_lifted_covers(
                    search_problem, cut_lp_raw.x, root_lo, root_hi,
                    cov_opts, diag.cover);
                diag.lifted_cover_candidates += cov.size();
                candidates.insert(candidates.end(),
                                  std::make_move_iterator(cov.begin()),
                                  std::make_move_iterator(cov.end()));
            }
            if (dsd.run_mir) {
                MirOptions mir_opts = cut_cfg.mir;
                mir_opts.max_cuts = dsd.budget_mir;
                // Deep aggregation is expensive (swath1: cut loop 15 s -> 25 s, pg: cut
                // LP iterations tripled). Shallow and deep are explicit states: deep is
                // entered when the shallow rounds have stalled (progress below 3% of
                // the gap, or of the offset-free bound scale), and only while what
                // deep batches have cost so far -- their separation AND the LP solves
                // that followed them -- stays within 40% of the root cut allowance.
                // exp-1-500-5-5 needs 24 rows per base to leave 52k; its first rounds
                // gain 25% and 11% with 4.
                {
                    const f64 bnow = node_lp_bound_min(cut_lp_raw, sense);
                    const f64 inc_min = have_incumbent ? sense * best_incumbent
                                                       : std::numeric_limits<f64>::quiet_NaN();
                    const bool stalled = std::isfinite(mir_last_bound) &&
                        cut_round_progress(mir_last_bound, bnow, inc_min, root_offset_min) < 0.03;
                    mir_last_bound = bnow;
                    double deep_ms = 0.0;
                    const auto& tr = diag.cut_round_trace;
                    for (std::size_t r = 0; r + 1 < tr.size(); ++r)
                        if (tr[r].mir_depth > 4) deep_ms += tr[r].separate_ms + tr[r + 1].lp_ms;
                    const bool deep_affordable = opts.time_limit_s <= 0.0 ||
                        deep_ms <= 0.4 * root_cut_allowance_s * 1000.0;
                    if (round < 2 || !stalled || !deep_affordable)
                        mir_opts.max_aggregations = std::min(mir_opts.max_aggregations, 4);
                    else
                        ++diag.mir_deep_rounds;
                    diag.cut_round_trace.back().mir_depth = mir_opts.max_aggregations;
                }
                auto mc = separate_mir(search_problem, cut_lp_raw.x, root_lo,
                                       root_hi, mir_opts, diag.mir, nullptr,
                                       nullptr, &conflict_graph);
                diag.mir_candidates += mc.size();
                candidates.insert(candidates.end(),
                                  std::make_move_iterator(mc.begin()),
                                  std::make_move_iterator(mc.end()));
            }
            if (dsd.run_ib && !conflict_graph.empty()) {
                auto vub = separate_implied_bound_cuts(conflict_graph,
                                                       cut_lp_raw.x, opts.probe);
                if (static_cast<int>(vub.size()) > dsd.budget_ib)
                    vub.resize(static_cast<std::size_t>(dsd.budget_ib));
                diag.implied_bound_cut_candidates += vub.size();
                candidates.insert(candidates.end(),
                                  std::make_move_iterator(vub.begin()),
                                  std::make_move_iterator(vub.end()));
            }
            if (dsd.run_zerohalf) {
                ZeroHalfOptions zh_opts = dynsep_opts.zerohalf;
                zh_opts.max_cuts = dsd.budget_zerohalf;
                auto zh = separate_zerohalf(search_problem, cut_lp_raw.x,
                                            root_lo, root_hi, zh_opts,
                                            diag.zerohalf);
                diag.zerohalf_candidates += zh.size();
                candidates.insert(candidates.end(),
                                  std::make_move_iterator(zh.begin()),
                                  std::make_move_iterator(zh.end()));
            }
            // FLOW COVER IS GATED OFF BY DEFAULT: it generates INVALID cuts.
            //
            // Verified 2026-09-19 with --verify-cuts against blend2's true
            // optimum (7.5989850). Five FC_ cuts excluded the optimum, e.g.
            //   FC_0  activity 12040  not in [-inf, 312]
            // -- a factor of 38, so this is a derivation error, not numerical
            // noise. The effect is a FALSE Optimal (blend2 reported 16.554765).
            //
            // Two things made it hard to see. DynSep chooses separators on a
            // time budget, so FC cuts appeared even when the run asked only for
            // --clique-cuts --cover-cuts, and the wrong answer depended on the
            // TIME LIMIT (60s correct, 90s wrong). And each family is sound on
            // its own, so the fault only showed once a second family made
            // DynSep reach for flow cover -- 9 of 10 separator PAIRS produced a
            // false Optimal while every family alone was correct.
            //
            // Requires an explicit opt-in now. Fix flowcover.cpp, then re-check
            // with --verify-cuts before restoring the DynSep path.
            if (dsd.run_flowcover && opts.flow_cover_cuts) {
                FlowCoverOptions fc_opts = dynsep_opts.flowcover;
                fc_opts.max_cuts = dsd.budget_flowcover;
                auto fc = separate_flow_covers(search_problem, cut_lp_raw.x,
                                               root_lo, root_hi, fc_opts,
                                               diag.flowcover);
                diag.flowcover_candidates += fc.size();
                candidates.insert(candidates.end(),
                                  std::make_move_iterator(fc.begin()),
                                  std::make_move_iterator(fc.end()));
            }
            diag.cut_separate_ms += ms_since(t_separate);
            diag.cut_round_trace.back().separate_ms = ms_since(t_separate);
            diag.root_cuts_generated += candidates.size();
            const auto t_select = Clock::now();
            if (cut_cfg.auto_cuts && milp_policy_is_latest(cut_cfg.policy)) {
                CutFilterStats fst;
                const std::size_t before_filter = candidates.size();
                candidates = filter_cut_candidates_for_round(
                    std::move(candidates), search_problem, cut_lp_raw.x,
                    cut_cfg.cut, &fst);
                diag.root_prefilter.add(fst);
                diag.root_cuts_prefilter_rejected +=
                    before_filter - candidates.size();
            }
            cut_pool.add(candidates, cut_diag);
            if (hgtsm_opts.enabled) {
                Index n_int = 0, n_frac = 0;
                f64 mean_frac = 0.0;
                for (Index j = 0; j < search_problem.n_cols(); ++j) {
                    if (search_problem.is_integer.empty() ||
                        !search_problem.is_integer[sz(j)])
                        continue;
                    ++n_int;
                    const f64 xv = cut_lp_raw.x[sz(j)];
                    const f64 f = xv - std::floor(xv);
                    const f64 frac = std::min(f, 1.0 - f);
                    if (frac > opts.int_tol) {
                        mean_frac += frac;
                        ++n_frac;
                    }
                }
                if (n_frac > 0) mean_frac /= static_cast<f64>(n_frac);
                fill_hgtsm_lp_state(search_problem.n_cols(), n_int, n_frac,
                                    mean_frac, diag.gap_rel, 0,
                                    cut_pool.size(), dynsep_last_gain,
                                    hgtsm_lp_state);
                if (hgtsm_use_graph) {
                    cut_pool.set_external_batch_scorer(
                        [&](const std::vector<CutRow>& cuts,
                            const std::vector<f64>& xpt,
                            std::vector<f64>& scores) {
                            auto g = build_tripartite_snapshot(
                                search_problem, root_lo, root_hi, cuts, &xpt);
                            f64 ratio = 1.0;
                            hgtsm_score_sequence(hgtsm_model, g, scores,
                                                 &ratio);
                            hgtsm_diag.scored_cuts +=
                                static_cast<std::uint64_t>(cuts.size());
                            if (hgtsm_opts.collect_labels) {
                                std::vector<f64> labs = scores;
                                for (f64& lab : labs)
                                    lab = std::max(0.0, lab);
                                hgtsm_collector.add_round(std::move(g),
                                                          std::move(labs));
                            }
                            // Higher-level ratio: soft-cap how many we keep
                            // via score scaling (CutPool still owns filters).
                            if (ratio < 0.999 && !scores.empty()) {
                                const std::size_t keep = std::max<std::size_t>(
                                    1, static_cast<std::size_t>(std::ceil(
                                           ratio *
                                           static_cast<f64>(scores.size()))));
                                auto ord = hgtsm_sequence_order(
                                    scores, static_cast<int>(keep));
                                std::vector<char> keep_mask(scores.size(), 0);
                                for (std::size_t idx : ord)
                                    if (idx < keep_mask.size())
                                        keep_mask[idx] = 1;
                                for (std::size_t i = 0; i < scores.size();
                                     ++i)
                                    if (!keep_mask[i]) scores[i] = -1e100;
                            }
                        });
                    ++hgtsm_diag.graph_selects;
                    hgtsm_diag.used_graph = true;
                } else {
                    cut_pool.set_external_scorer(
                        [&](const CutRow& cut, const std::vector<f64>& xpt) {
                            CutFeatureVec cf{};
                            CutFeatureContext cctx{&search_problem, &xpt};
                            if (!fill_cut_features(cctx, cut, cf)) return 0.0;
                            ++hgtsm_diag.scored_cuts;
                            return hgtsm_score(hgtsm_model, cf, hgtsm_lp_state);
                        });
                    hgtsm_diag.used_graph = false;
                }
                ++hgtsm_diag.selects;
                hgtsm_diag.model_loaded = hgtsm_model.loaded;
                hgtsm_diag.used_builtin = !hgtsm_model.loaded;
            }
            const f64 bound_before_sel = cut_lp_raw.objective;
            const auto cuts = cut_pool.select_violated(cut_lp_raw.x, cut_diag);
            diag.cut_select_ms += ms_since(t_select);
            diag.root_cuts_selected += cuts.size();
            if (cuts.empty()) break;
            // GCS: observe root selections under content-stable ids.
            if (tree_cut_opts.gcs_enabled) {
                CutFeatureContext cctx{&search_problem, &cut_lp_raw.x};
                for (const auto& c : cuts) {
                    CutFeatureVec cf{};
                    if (!fill_cut_features(cctx, c, cf)) continue;
                    const bool globally_valid =
                        c.name.rfind("ZH_", 0) == 0 ||
                        c.name.rfind("FC_", 0) == 0 ||
                        c.name.rfind("COV_", 0) == 0 ||
                        c.name.rfind("COVPC_", 0) == 0 ||
                        c.name.rfind("COVGNS_", 0) == 0 ||
                        c.name.rfind("MIR_", 0) == 0 ||
                        c.name.rfind("CLQ_", 0) == 0 ||
                        c.name.rfind("VUB_", 0) == 0;
                    gcs_pool.observe(c, cf, cf[0], cf[4] > 0.0, globally_valid,
                                     cut_content_id(c), /*depth=*/0,
                                     diag.gap_rel, bound_before_sel);
                    if (hgtsm_opts.collect_labels)
                        hgtsm_collector.add(cf, hgtsm_lp_state, cf[0]);
                    if (opts.gcs_collect_out != nullptr ||
                        tree_cut_opts.gcs_enabled) {
                        GcsCutFeat gf{};
                        GcsCandidate tmp;
                        tmp.feats = cf;
                        tmp.efficacy_sum = cf[0];
                        tmp.seen_nodes = 1;
                        tmp.violation_nodes = cf[4] > 0.0 ? 1 : 0;
                        tmp.globally_valid = globally_valid;
                        fill_gcs_cut_feat(tmp, gf);
                        // Label = efficacy proxy; refined after bound update.
                        gcs_collector.add(gf, cf[0]);
                    }
                }
            }
            // Cut-validity diagnostic: name any cut that excludes the
            // reference point. Runs before the cuts enter the model, so the
            // report is about the cut as generated.
            if (opts.cut_reference_point != nullptr) {
                for (const auto& c : cuts) {
                    if (!cut_admits_point(c.cols, c.vals, c.row_lo, c.row_hi,
                                          *opts.cut_reference_point,
                                          opts.primal_feas_tol)) {
                        f64 act = 0.0;
                        for (std::size_t k = 0; k < c.cols.size(); ++k)
                            act += c.vals[k] *
                                   (*opts.cut_reference_point)[sz(c.cols[k])];
                        std::fprintf(stderr,
                            "INVALID CUT  %-18s activity %.9g not in [%.9g, %.9g]"
                            "  (nnz %zu)\n",
                            c.name.c_str(), act, c.row_lo, c.row_hi,
                            c.cols.size());
                        ++diag.invalid_cuts_detected;
                    }
                }
            }
            // MARGINAL GATE. Before this round's cuts go in, price each
            // optional family by what the root bound loses without it. See
            // CutOptions::marginal_gate.
            std::vector<CutRow> gated_cuts;
            const std::vector<CutRow>* cuts_to_apply = &cuts;
            if (cut_cfg.cut.marginal_gate && !marginal_gate_done &&
                round >= cut_cfg.cut.marginal_gate_round) {
                marginal_gate_done = true;
                const auto t_probe = Clock::now();
                auto& mg = diag.marginal_gate;
                mg.ran = true;
                for (const auto& c : cuts) ++mg.cuts_offered[family_of(c)];

                engines::SimplexOptions probe_opts = cut_lp_opts;
                auto probe_bound = [&](const std::vector<CutRow>& subset,
                                       f64& out) -> bool {
                    model::LpProblem trial = search_problem;
                    apply_cuts_inplace(trial, subset, cut_cfg.cut, nullptr);
                    engines::SimplexDiagnostics psd;
                    engines::SimplexBasis pb;
                    auto pricing_opts = probe_opts;
                    if (opts.time_limit_s > 0.0) {
                        const double left = std::min({
                            opts.time_limit_s - ms_since(t0) / 1000.0,
                            root_total_allowance_s - ms_since(t0) / 1000.0,
                            root_cut_allowance_s - ms_since(cut_loop_t0) / 1000.0});
                        if (!(left > 0.0)) return false;
                        pricing_opts.time_limit_s = pricing_opts.time_limit_s > 0.0
                            ? std::min(pricing_opts.time_limit_s, left) : left;
                    }
                    const auto t_pricing_lp = Clock::now();
                    core::RawResult pr =
                        engines::solve_simplex(trial, pricing_opts, psd, &pb);
                    branching_lp_work.charge(psd);
                    branching_lp_work.ms += ms_since(t_pricing_lp);
                    ++mg.probe_solves;
                    ++diag.lp_solves;
                    if (!relaxation_proved(pr, psd, pricing_opts)) return false;
                    out = pr.objective;
                    return true;
                };

                f64 b_all = 0.0;
                bool ok = probe_bound(cuts, b_all);
                for (int f = kFamMir; ok && f < kFamCount; ++f) {
                    if (mg.cuts_offered[f] == 0) continue;
                    std::vector<CutRow> without;
                    without.reserve(cuts.size());
                    for (const auto& c : cuts)
                        if (family_of(c) != f) without.push_back(c);
                    f64 b_wo = 0.0;
                    if (!probe_bound(without, b_wo)) { ok = false; break; }
                    const f64 rel = std::fabs(b_all - b_wo) /
                                    (1.0 + std::fabs(b_all));
                    mg.marginal_rel[f] = rel;
                    if (rel < cut_cfg.cut.marginal_gate_min_rel) {
                        family_off[f] = true;
                        mg.disabled[f] = true;
                    }
                }
                if (!ok) {
                    // A probe LP did not prove. Stand the gate down entirely
                    // rather than act on a partial reading: disabling a family
                    // on evidence we could not obtain is exactly the mistake
                    // this gate exists to stop someone making by eye.
                    mg.aborted = true;
                    for (int f = 0; f < kFamCount; ++f) {
                        family_off[f] = false;
                        mg.disabled[f] = false;
                    }
                } else {
                    bool any = false;
                    for (int f = 0; f < kFamCount; ++f) any |= family_off[f];
                    if (any) {
                        gated_cuts.reserve(cuts.size());
                        for (const auto& c : cuts)
                            if (!family_off[family_of(c)])
                                gated_cuts.push_back(c);
                        mg.cuts_dropped =
                            static_cast<int>(cuts.size() - gated_cuts.size());
                        cuts_to_apply = &gated_cuts;
                    }
                }
                mg.probe_ms = ms_since(t_probe);
            }

            diag.root_cuts_gate_dropped += cuts.size() - cuts_to_apply->size();
            CutUndo round_undo;
            const auto t_apply = Clock::now();
            apply_cuts_inplace(search_problem, *cuts_to_apply, cut_cfg.cut,
                               &round_undo);
            diag.cut_apply_ms += ms_since(t_apply);
            diag.root_cut_rows_appended +=
                static_cast<std::uint64_t>(round_undo.rows_added());
            diag.root_cut_rows_tightened +=
                static_cast<std::uint64_t>(round_undo.tightened_rows.size());
            // Everything below accounts for what was APPLIED. When the
            // marginal gate dropped a family, that differs from what the pool
            // selected, and reporting the selection would credit the model
            // with rows it does not carry.
            const std::vector<CutRow>& applied = *cuts_to_apply;
            {
                auto& tr = diag.cut_round_trace.back();
                tr.rows_added = round_undo.rows_added();
                tr.rows_tightened =
                    static_cast<int>(round_undo.tightened_rows.size());
                tr.cuts_selected = static_cast<int>(applied.size());
                for (const auto& c : applied) tr.added_nnz += static_cast<long>(c.cols.size());
                if (opts.trace_cut_batches) {
                    for (const auto& c : applied) {
                        f64 act = 0.0;
                        for (std::size_t k = 0; k < c.cols.size(); ++k)
                            act += c.vals[k] * cut_lp_raw.x[sz(c.cols[k])];
                        std::printf("  [cut r%02d] %-10s nnz %-4zu lo %-14.8g hi %-14.8g act %-14.8g id %s\n",
                                    round, c.name.c_str(), c.cols.size(), c.row_lo, c.row_hi, act,
                                    cut_content_id(c).c_str());
                    }
                }
                last_applied_trace =
                    static_cast<int>(diag.cut_round_trace.size()) - 1;
            }
            if (opts.cut_transaction) pending_batches.push_back(round_undo);
            if (cut_cfg.cut.rollback_stalled_rounds) {
                PendingRound pr;
                pr.undo = round_undo;
                pr.cuts = applied;
                pr.trace_index = last_applied_trace;
                unpaid_rounds.push_back(std::move(pr));
            }
            // The pool merges and filters both families, so attribute the
            // selected batch by the name the separator stamped on each row
            // rather than by proportion -- integer-dividing a candidate ratio
            // silently reported 0 clique cuts on rounds that did add some.
            std::vector<std::string> selected_names;
            selected_names.reserve(applied.size());
            f64 eff_sum = 0.0;
            for (const auto& c : applied) {
                selected_names.push_back(c.name);
                auto& trr = diag.cut_round_trace.back();
                if (c.name.rfind("CLQ_", 0) == 0) {
                    ++diag.clique_cuts_added; ++trr.clique;
                } else if (c.name.rfind("VUB_", 0) == 0) {
                    ++diag.implied_bound_cuts_added; ++trr.vub;
                } else if (c.name.rfind("COV_", 0) == 0 ||
                         c.name.rfind("COVPC_", 0) == 0 ||
                         c.name.rfind("COVGNS_", 0) == 0) {
                    ++diag.lifted_cover_cuts_added; ++trr.cover;
                } else if (c.name.rfind("MIR_", 0) == 0) {
                    ++diag.mir_cuts_added; ++trr.mir;
                } else if (c.name.rfind("ZH_", 0) == 0) {
                    ++diag.zerohalf_cuts_added; ++trr.zerohalf;
                } else if (c.name.rfind("FC_", 0) == 0) {
                    ++diag.flowcover_cuts_added;
                } else {
                    ++diag.gmi_cuts_added; ++trr.gmi;
                }
                // Approximate efficacy for DynSep feedback.
                f64 lhs = 0.0;
                for (std::size_t q = 0; q < c.cols.size(); ++q)
                    lhs += c.vals[q] * cut_lp_raw.x[sz(c.cols[q])];
                if (std::isfinite(c.row_hi))
                    eff_sum += std::max(0.0, lhs - c.row_hi);
                if (!opts.verbose) continue;
                // Per-cut shape, and how close the cut is to a row the model
                // already has. A cut that nearly duplicates an existing row
                // adds no information but does add near-linear dependence,
                // which is a plausible way for a "harmless" cut to wreck the
                // node LP's warm start -- so it is worth being able to see.
                f64 cn = 0.0, cmin = std::numeric_limits<f64>::infinity(),
                    cmax = 0.0;
                for (const f64 v : c.vals) {
                    cn += v * v;
                    cmin = std::min(cmin, std::fabs(v));
                    cmax = std::max(cmax, std::fabs(v));
                }
                cn = std::sqrt(cn);
                std::vector<f64> dense(sz(search_problem.n_cols()), 0.0);
                for (std::size_t q = 0; q < c.cols.size(); ++q)
                    dense[sz(c.cols[q])] = c.vals[q];
                const auto& rp2 = search_problem.A.pattern.row_ptr();
                const auto& ci2 = search_problem.A.pattern.col_idx();
                const auto& av2 = search_problem.A.vals;
                f64 best_cos = 0.0;
                Index best_row = -1;
                for (Index i = 0; i < search_problem.n_rows(); ++i) {
                    f64 dot = 0.0, rn = 0.0;
                    for (core::Offset k = rp2[sz(i)]; k < rp2[sz(i) + 1]; ++k) {
                        dot += av2[sz(k)] * dense[sz(ci2[sz(k)])];
                        rn += av2[sz(k)] * av2[sz(k)];
                    }
                    if (rn <= 0.0 || cn <= 0.0) continue;
                    const f64 cs = std::fabs(dot) / (cn * std::sqrt(rn));
                    if (cs > best_cos) { best_cos = cs; best_row = i; }
                }
                std::printf("  [cut] %-10s len %3zu  dyn %9.3g  rhs %12.5g  "
                            "max|cos| vs model rows %.5f (row %d)\n",
                            c.name.c_str(), c.cols.size(),
                            cmin > 0.0 ? cmax / cmin : 0.0, c.row_hi,
                            best_cos, static_cast<int>(best_row));
                for (f64& v : dense) v = 0.0;
            }
            const f64 mean_eff =
                applied.empty() ? 0.0
                                : eff_sum / static_cast<f64>(applied.size());
            dynsep.observe_selected_names(selected_names, mean_eff);
            dynsep.collect_round_labels(dsd);
            // Bound gain for next-round DynSep input (filled after reoptimize).
            dynsep_last_gain = mean_eff;
            ++diag.cut_rounds;
        }
        diag.cut_rounds_ms = ms_since(cut_loop_t0);
        diag.root_bound_before_rollback = diag.root_bound_after_cuts;
        const auto t_retract = Clock::now();
        // TRANSACTION OUTCOME for the batches no LP verdict covers. They are
        // retracted (newest first): the tree then starts from the last PROVED
        // relaxation -- whose point and basis root node 1 reuses -- rather than
        // from a model with rows whose LP was never solved. The unfinished-LP
        // hand-off described the deferred model, so it goes too.
        if (opts.cut_transaction && !pending_batches.empty()) {
            const std::size_t k = pending_batches.size();
            for (std::size_t q = k; q > 0; --q) {
                diag.cut_rows_retracted += static_cast<std::uint64_t>(pending_batches[q - 1].rows_added());
                retract_cuts_inplace(search_problem, pending_batches[q - 1]);
                ++diag.cut_batches_deferred;
                --diag.cut_rounds;
                const int ti = static_cast<int>(diag.cut_round_trace.size()) -
                               static_cast<int>(k - q) - 1;
                (void)ti;
            }
            // The stall-rollback list held these same rounds as its newest entries.
            for (std::size_t q = 0; q < k && !unpaid_rounds.empty(); ++q) unpaid_rounds.pop_back();
            pending_batches.clear();
            have_cut_loop_basis = false;
            unfinished_root_lp.reset();
            last_solved_rows = std::min(last_solved_rows, search_problem.n_rows());
        }
        // ROLLBACK. `unpaid_rounds` holds every round appended since the last
        // realised bound improvement, so by construction the loop's final
        // bound is the bound these rounds FAILED to move: removing them cannot
        // lower it, and there is nothing to re-solve to find that out.
        //
        // The last entry is special. If the loop exited without ever
        // re-solving after it (round cap, time cap, an unproved LP), its gain
        // was never measured -- it is not a round that stalled, it is a round
        // nobody looked at. Keeping the claim honest means retracting only
        // rounds the loop's own measure condemned, so that one is dropped from
        // the set unless a later solve did evaluate it.
        if (cut_cfg.cut.rollback_stalled_rounds && !unpaid_rounds.empty()) {
            std::size_t n_retract = unpaid_rounds.size();
            const int last_i = unpaid_rounds.back().trace_index;
            if (last_i >= 0 &&
                last_i < static_cast<int>(diag.cut_round_trace.size()) &&
                !std::isfinite(diag.cut_round_trace[sz(last_i)].gain_rel))
                --n_retract;
            for (std::size_t q = unpaid_rounds.size(); q > unpaid_rounds.size() - n_retract; --q) {
                PendingRound& pr = unpaid_rounds[q - 1];
                diag.cut_rows_retracted += static_cast<std::uint64_t>(pr.undo.rows_added());
                diag.cut_rows_retightened +=
                    static_cast<std::uint64_t>(pr.undo.tightened_rows.size());
                retract_cuts_inplace(search_problem, pr.undo);
                ++diag.cut_rounds_rolled_back;
                if (pr.trace_index >= 0 &&
                    pr.trace_index < static_cast<int>(diag.cut_round_trace.size()))
                    diag.cut_round_trace[sz(pr.trace_index)].rolled_back = true;
                --diag.cut_rounds;
            }
            unpaid_rounds.clear();
            // cut_loop_basis was built for the model WITH those rows: its
            // basic[] carries one logical per row that no longer exists, so
            // the block below that seeds the root node by EXTENDING it would
            // hand the root a basis of the wrong dimension. Drop it.
            //
            // This invalidation is load-bearing, not defensive. Do not delete
            // it on the belief that the basis is dead: have_cut_loop_basis is
            // unset only while CutOptions::warm_start_rounds is off, and that
            // flag assigns it (see the warm hand-off after the round solve).
            // The separate, always-live `cut_basis` -- the one the separators
            // read the tableau from -- is a different object and is not what
            // this touches.
            have_cut_loop_basis = false;
        }
        // PURGE. Drop the cut rows that carry no dual price at the last root
        // LP the loop proved. See CutOptions::purge_nonbinding_cuts for why
        // the bound survives this and why it cannot make an answer wrong.
        if (cut_cfg.cut.purge_nonbinding_cuts) {
            const Index m_now = search_problem.n_rows();
            Index priced = std::min(last_solved_rows, m_now);
            // A rollback above may have removed rows that `last_solved_y`
            // priced. The purge's argument -- drop row i and (x*, y* minus i)
            // is still an optimal pair -- needs y* to be dual feasible for the
            // model as it stands NOW, and dropping a retracted row that
            // carried a non-zero multiplier breaks exactly that. Degeneracy
            // makes this possible even though the retracted round moved no
            // bound, so check rather than assume: if any retracted row was
            // priced, these duals no longer describe this model and the purge
            // sits the round out.
            for (Index i = m_now; i < last_solved_rows &&
                     static_cast<Index>(last_solved_y.size()) >= last_solved_rows;
                 ++i) {
                if (std::fabs(last_solved_y[sz(i)]) > cut_cfg.cut.purge_dual_tol) {
                    priced = rows_before_cuts;
                    ++diag.cut_purge_skipped_stale_duals;
                    break;
                }
            }
            if (m_now > rows_before_cuts && priced > rows_before_cuts &&
                static_cast<Index>(last_solved_y.size()) >= priced) {
                const auto& rp = search_problem.A.pattern.row_ptr();
                const auto& ci = search_problem.A.pattern.col_idx();
                const auto& av = search_problem.A.vals;
                std::vector<Index> keep;
                keep.reserve(sz(m_now - rows_before_cuts));
                for (Index i = rows_before_cuts; i < m_now; ++i) {
                    if (i >= priced) { keep.push_back(i); continue; }
                    if (std::fabs(last_solved_y[sz(i)]) >
                        cut_cfg.cut.purge_dual_tol) {
                        keep.push_back(i);
                        continue;
                    }
                    // Zero multiplier. Keep it anyway when it is TIGHT: the
                    // duality argument allows dropping it, but a row sitting
                    // exactly on the optimal face is the one most likely to
                    // bind again a few branchings down, and dual degeneracy
                    // makes a zero price there uninformative.
                    f64 act = 0.0;
                    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                        act += av[sz(k)] * last_solved_x[sz(ci[sz(k)])];
                    const f64 slack_lo = std::isfinite(search_problem.row_lo[sz(i)])
                        ? act - search_problem.row_lo[sz(i)] : core::kPosInf;
                    const f64 slack_hi = std::isfinite(search_problem.row_hi[sz(i)])
                        ? search_problem.row_hi[sz(i)] - act : core::kPosInf;
                    if (std::min(slack_lo, slack_hi) > cut_cfg.cut.purge_slack_tol)
                        continue;   // priced at zero AND strictly slack: drop
                    keep.push_back(i);
                }
                if (static_cast<Index>(keep.size()) < m_now - rows_before_cuts) {
                    // Snapshot the survivors, drop every cut row, put them
                    // back. The kept rows already passed apply_cuts' duplicate
                    // and parallelism filters, so they are re-appended
                    // directly rather than re-run through it.
                    struct KeptRow {
                        std::vector<Index> cols;
                        std::vector<f64> vals;
                        f64 lo = 0.0, hi = 0.0;
                        std::string name;
                    };
                    std::vector<KeptRow> rows;
                    rows.reserve(keep.size());
                    for (const Index i : keep) {
                        KeptRow kr;
                        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                            kr.cols.push_back(ci[sz(k)]);
                            kr.vals.push_back(av[sz(k)]);
                        }
                        kr.lo = search_problem.row_lo[sz(i)];
                        kr.hi = search_problem.row_hi[sz(i)];
                        if (sz(i) < search_problem.row_names.size())
                            kr.name = search_problem.row_names[sz(i)];
                        rows.push_back(std::move(kr));
                    }
                    diag.cut_rows_purged +=
                        static_cast<std::uint64_t>(m_now - rows_before_cuts) -
                        static_cast<std::uint64_t>(rows.size());
                    diag.cut_rows_kept += static_cast<std::uint64_t>(rows.size());
                    search_problem.A.truncate_rows(rows_before_cuts);
                    search_problem.row_lo.resize(sz(rows_before_cuts));
                    search_problem.row_hi.resize(sz(rows_before_cuts));
                    if (!search_problem.row_names.empty())
                        search_problem.row_names.resize(sz(rows_before_cuts));
                    for (auto& kr : rows) {
                        search_problem.A.append_row(kr.cols, kr.vals);
                        search_problem.row_lo.push_back(kr.lo);
                        search_problem.row_hi.push_back(kr.hi);
                        if (!search_problem.row_names.empty())
                            search_problem.row_names.push_back(std::move(kr.name));
                    }
                    have_cut_loop_basis = false;
                } else {
                    diag.cut_rows_kept += static_cast<std::uint64_t>(keep.size());
                }
            }
        }
        diag.cut_retract_ms = ms_since(t_retract);
        diag.root_cut_rows_active =
            static_cast<std::uint64_t>(search_problem.n_rows() - rows_before_cuts);
        if (opts.verbose) {
            // Exact-duplicate row detection over the cut-augmented model.
            // Duplicated rows are linearly dependent, which is a direct way to
            // make a basis singular and to force the warm-started dual to work
            // far harder than the extra rows would suggest.
            const auto& rp3 = search_problem.A.pattern.row_ptr();
            const auto& ci3 = search_problem.A.pattern.col_idx();
            const auto& av3 = search_problem.A.vals;
            std::vector<std::string> keys;
            keys.reserve(sz(search_problem.n_rows()));
            for (Index i = 0; i < search_problem.n_rows(); ++i) {
                std::string k;
                for (core::Offset q = rp3[sz(i)]; q < rp3[sz(i) + 1]; ++q) {
                    char buf[48];
                    std::snprintf(buf, sizeof buf, "%d:%.12g,",
                                  static_cast<int>(ci3[sz(q)]), av3[sz(q)]);
                    k += buf;
                }
                keys.push_back(std::move(k));
            }
            std::vector<std::string> sorted = keys;
            std::sort(sorted.begin(), sorted.end());
            std::size_t dup = 0;
            for (std::size_t i = 1; i < sorted.size(); ++i)
                if (sorted[i] == sorted[i - 1]) ++dup;
            std::printf("  [cut] model now %d rows, %zu exact duplicate rows\n",
                        static_cast<int>(search_problem.n_rows()), dup);
        }
        diag.cut_loop_ms = ms_since(t_cutloop);
        diag.gmi_candidates_considered = cut_diag.candidates_considered;
        diag.gmi_integral_activity_rows = cut_diag.integral_activity_rows;
        diag.gmi_integer_activity_candidates = cut_diag.integer_activity_candidates;
        diag.gmi_integer_activity_terms = cut_diag.integer_activity_terms;
        diag.gmi_fractional_integer_bound_terms =
            cut_diag.fractional_integer_bound_terms;
        diag.gmi_missing_basis = cut_diag.gmi_missing_basis;
        diag.gmi_invalid_factor = cut_diag.gmi_invalid_factor;
        diag.gmi_empty_rows = cut_diag.gmi_empty_rows;
        diag.gmi_rejected_free = cut_diag.rejected_free_nonbasic;
        diag.gmi_rejected_dynamism = cut_diag.rejected_dynamism;
        diag.gmi_dynamism_repaired = cut_diag.dynamism_repaired;
        diag.tableau_cmir_cuts = cut_diag.tableau_cmir_cuts;
        diag.tableau_cmir_won = cut_diag.tableau_cmir_won;
        diag.tableau_cmir_only = cut_diag.tableau_cmir_only;
        diag.gmi_cmir_attempted = cut_diag.cmir_attempted;
        diag.gmi_cmir_recovered = cut_diag.cmir_recovered;
        diag.gmi_rejected_violation = cut_diag.rejected_violation;
        diag.cut_pool_inserted = cut_diag.pool_inserted;
        diag.cut_pool_duplicates = cut_diag.pool_duplicates;
        diag.cut_pool_dominated = cut_diag.pool_dominated;
        diag.cut_pool_parallel_rejections = cut_diag.pool_rejected_parallel;
        diag.cut_pool_aged_out = cut_diag.pool_aged_out;
        diag.cut_pool_evicted = cut_diag.pool_evicted;
        diag.dynsep = dynsep.diagnostics();
        diag.l2sep = l2sep_diag;
        diag.hgtsm = hgtsm_diag;
        if (!root_lp_outcome_logged &&
            std::isfinite(diag.root_bound_after_cuts)) {
            char rbuf[96];
            std::snprintf(rbuf, sizeof rbuf, "\"obj\":%.10g,\"integer\":false",
                          diag.root_bound_after_cuts);
            SOR_ROUTE(1, "bab", "root_lp_outcome", rbuf);
            root_lp_outcome_logged = true;
        }
    }

    {
        Node root;
        // root_lo/root_hi are mip's own bounds unless probing tightened them;
        // either way they are valid for every integer-feasible point.
        root.col_lo = root_lo;
        root.col_hi = root_hi;
        root.bound = root_cert.value;
        root.depth = 0;
        if (have_cut_loop_basis) {
            // Extend the cut loop's final basis to search_problem's current
            // dimensions (rows may have been added after the last solved
            // round) and seed the root with it.
            const Index ns_ = search_problem.n_cols();
            const Index m_old =
                static_cast<Index>(cut_loop_basis.basic.size());
            const Index m_new = search_problem.n_rows();
            if (static_cast<Index>(cut_loop_basis.status.size()) == ns_ + m_old &&
                m_new >= m_old) {
                cut_loop_basis.status.resize(sz(ns_ + m_new));
                for (Index i = m_old; i < m_new; ++i) {
                    cut_loop_basis.basic.push_back(ns_ + i);
                    cut_loop_basis.status[sz(ns_ + i)] =
                        engines::NonbasicStatus::Basic;
                }
                root.basis = std::move(cut_loop_basis);
                root.has_basis = true;
            }
        }
        else if (proved_root_lp &&
                 proved_root_lp->problem.n_cols() == search_problem.n_cols() &&
                 proved_root_lp->problem.n_rows() == search_problem.n_rows() &&
                 static_cast<Index>(proved_root_lp->basis.basic.size()) == search_problem.n_rows() &&
                 proved_root_lp->basis.status.size() ==
                     sz(search_problem.n_cols()) + sz(search_problem.n_rows())) {
            // The transition chose the last PROVED relaxation (deferred cut
            // batches were rolled back): start the root node from its basis, so a
            // changed root box (reduced-cost fixing) costs a warm dual, not a
            // cold solve of the whole LP.
            root.basis = proved_root_lp->basis;
            root.has_basis = true;
            ++diag.root_basis_from_proved;
        }
        else if (opts.initial_root_basis != nullptr &&
                 opts.initial_root_basis->n_struct == search_problem.n_cols() &&
                 static_cast<Index>(opts.initial_root_basis->basic.size()) <= search_problem.n_rows() &&
                 !opts.initial_root_basis->basic.empty() &&
                 opts.initial_root_basis->status.size() ==
                     sz(search_problem.n_cols()) + opts.initial_root_basis->basic.size()) {
            // The carried basis covers the model's rows; rows this solve appended
            // after them (binary cover rows) start with their slacks basic.
            root.basis = *opts.initial_root_basis;
            const Index ns_ = search_problem.n_cols();
            for (Index i = static_cast<Index>(root.basis.basic.size()); i < search_problem.n_rows(); ++i) {
                root.basis.basic.push_back(ns_ + i);
                root.basis.status.push_back(engines::NonbasicStatus::Basic);
            }
            root.has_basis = true;
            ++diag.root_basis_carried;
        }
        open.push(std::move(root));
    }

    // Final root-primal pass, on the last relaxation the cut loop proved. The
    // child context is published from THAT relaxation whether or not a pass
    // runs; the reserved allowance is spent once, on a snapshot that is new
    // (identity differs from every earlier one) and still has an open gap.
    if (opts.cuts_enabled && opts.sub_mip_depth == 0 && proved_root_lp &&
        proved_root_lp->raw.x.size() == sz(n)) {
        const f64 bfin = node_lp_bound_min(proved_root_lp->raw, sense);
        const bool gap_open =
            !cutoff_known() || !std::isfinite(bfin) ||
            known_cutoff_min() - bfin > 1e-6 * (1.0 + std::fabs(known_cutoff_orig()));
        if (opts.root_primal_final && gap_open)
            root_primal_pass(proved_root_lp->problem, proved_root_lp->raw, bfin,
                             &proved_root_lp->basis, true);
        else
            (void)publish_root_snapshot(proved_root_lp->problem, proved_root_lp->raw, bfin,
                                        &proved_root_lp->basis);
    }
    // The matrix and static model data never change during B&B. Reuse one
    // problem object and replace only its bound vectors for each node; copying
    // the complete CSR matrix at every node is pure overhead.
    model::LpProblem global_lp = std::move(search_problem);
    model::LpProblem node_lp = global_lp;
    struct RootRcInfo {
        model::LpProblem problem;
        std::vector<f64> x, y;
        f64 objective;
    };
    // Advances whenever the root box is tightened by reduced-cost fixing; a
    // node propagated before that has not seen the tighter bounds.
    std::optional<RootRcInfo> root_rc;
    f64 root_rc_inc_seen = std::numeric_limits<f64>::quiet_NaN();
    double root_rc_ms = 0.0;
    // Bumped whenever the root box gets tighter (reduced-cost fixing, a unit
    // clause); root_box_learned: a bound came from a clause rather than a certificate.
    std::uint64_t root_box_version = 0;
    bool root_box_learned = false;
    bool root_clause_contradiction = false;
    const auto apply_root_rc = [&]() {
        if (!opts.reduced_cost_strengthening || !root_rc || !cutoff_known() ||
            known_cutoff_orig() == root_rc_inc_seen || root_rc_ms >= 1000.0 || timed_out()) return;
        root_rc_inc_seen = known_cutoff_orig();
        const auto& rc = *root_rc;
        const auto result = certified_reduced_cost_bounds(
            rc.problem, rc.y, known_cutoff_orig(), root_lo, root_hi,
            opts.primal_feas_tol);
        root_rc_ms += result.ms;
        if (result.tightened > 0) ++root_box_version;
        diag.rc_certificate_checks += result.checked;
        if (result.tightened) ++diag.rc_strengthen_nodes;
        diag.rc_bounds_tightened += result.tightened;
        diag.rc_columns_fixed += result.fixed;
        diag.rc_strengthening_ms += result.ms;
    };
    const auto intersect_root_box = [&](Node& pending) {
        if (diag.rc_bounds_tightened == 0 && !root_box_learned) return true;
        for (Index j = 0; j < n; ++j) {
            pending.col_lo[sz(j)] = std::max(pending.col_lo[sz(j)], root_lo[sz(j)]);
            pending.col_hi[sz(j)] = std::min(pending.col_hi[sz(j)], root_hi[sz(j)]);
            if (pending.col_lo[sz(j)] > pending.col_hi[sz(j)]) return false;
        }
        return true;
    };
    // Bumped at every apply_cuts_inplace(global_lp, ...) call below, whether
    // it appended rows or only tightened an existing cut row's bounds in
    // place (CutUndo's own doc comment: "A round APPENDS rows, and it also
    // TIGHTENS pre-existing rows"). global_lp's row COUNT staying the same
    // does not mean its content is still what a cached copy last saw, so
    // this -- not n_rows() -- is what node_lp's copy-skip (S3.3) and the
    // node-LP carriers' invalidation (S3.2) are gated on.
    std::uint64_t global_lp_generation = 0;
    // S3.2: node LPs are all the SAME matrix (bound changes only), so the
    // parent's final DSE weights and LU factor are exact starting points for
    // a warm-started child. `&global_lp` is a stable token for as long as
    // this function runs.
    // One prepared LP (scaling, CSC, factor and weight carriers) serves every
    // global-shaped node: bound changes are O(n) via set_column_bounds, the
    // matrix is prepared once per global_lp generation instead of once per
    // node. Rebuilt whenever a global cut/nogood changes global_lp.
    // Event-driven node propagation: column->row incidence of global_lp for
    // the generation it was built at, reusable queue storage, and a version
    // that advances whenever the root box is tightened (a node propagated
    // before that has not seen the tighter root bounds).
    ColumnRowIndex prop_index;
    std::uint64_t prop_index_generation = ~0ull;
    PropagationScratch prop_scratch;
    std::unique_ptr<engines::DualProbeSession> node_session;
    // The local rows node_session was prepared over (empty: global rows only).
    // The hash key finds a candidate; THIS comparison decides a hit.
    std::vector<CutRow> node_session_local;
    // Prepared sessions of other row sets, kept so that switching between
    // subtrees does not re-prepare the LP (Ruiz scaling, column-wise copy). A
    // hit here is a prepared-matrix reuse only: whether the session's factor and
    // weights are usable for a given node is a separate, later check (the
    // `checkpoint_*` counters).
    struct CachedSession {
        std::uint64_t key = 0;
        std::vector<CutRow> local;
        std::unique_ptr<engines::DualProbeSession> session;
        std::uint64_t used = 0;
    };
    std::vector<CachedSession> session_cache;
    std::uint64_t session_cache_tick = 0;
    constexpr std::size_t kSessionCacheMax = 3;
    std::uint64_t node_session_generation = 0;
    // Factor/weight checkpoints of branched nodes (bounded; see LpCheckpoint).
    constexpr std::size_t kCheckpointBudgetBytes = 256ull << 20;
    CheckpointCache checkpoint_cache(kCheckpointBudgetBytes);
    // Did the most recent node LP run on node_session and leave its final
    // factor/weights there? Cleared by anything else that could replace them.
    bool session_state_is_last_node = false;
    // S3.3: skip node_lp = global_lp when node_lp already reflects this
    // generation and no local/tree cuts touched it since.
    std::uint64_t node_lp_generation = 0;
    bool node_lp_is_global_shaped = true;  // false once local cuts touch it
    const auto same_root_lp_structure = [](const model::LpProblem& old,
                                           const model::LpProblem& current) {
        return old.n_rows() == current.n_rows() &&
               old.n_cols() == current.n_cols() &&
               old.A.pattern.row_ptr() == current.A.pattern.row_ptr() &&
               old.A.pattern.col_idx() == current.A.pattern.col_idx() &&
               old.A.vals == current.A.vals &&
               old.row_lo == current.row_lo &&
               old.row_hi == current.row_hi &&
               old.c == current.c &&
               old.obj_offset == current.obj_offset &&
               old.maximize == current.maximize;
    };
    std::uint64_t local_cut_seq = 0;

    auto refresh_conflict_cap = [&]() {
        // Preserve the policy's historical total-failure signal while keeping
        // derivation and post-learning rejection diagnostics disjoint.
        const auto failures = diag.conflict_cut_diag.aborted +
                              diag.conflict_cut_diag.validation_rejected;
        if (failures >
            2u * (1u + diag.conflict_cuts_global)) {
            conflict_cut_cap = std::min(conflict_cut_cap,
                                       conflict_cut_opts.max_learned_cuts);
        } else if (diag.conflict_cuts_global >= 1 &&
                   failures <= diag.conflict_cut_diag.learned) {
            conflict_cut_cap =
                std::max(conflict_cut_cap, conflict_cut_opts.max_learned_cuts_hi);
        }
    };
    // P11: learned bound disjunctions, propagated at every node.
    std::vector<char> store_int(sz(n), 0);
    for (Index j = 0; j < n; ++j)
        store_int[sz(j)] = !global_lp.is_integer.empty() &&
                           global_lp.is_integer[sz(j)];
    ConflictStore conflict_store(n, conflict_cut_opts.store_bytes,
                                 conflict_cut_opts.store_max_len);
    auto try_apply_validated_global_cut = [&](const CutRow& learned,
                                              bool as_nogood) -> bool {
        auto reject_learned = [&]() {
            if (!as_nogood) {
                ++diag.conflict_cut_diag.validation_rejected;
            }
            return false;
        };
        if (conflict_cut_near_empty(learned, conflict_cut_opts.tol))
            return reject_learned();
        // Defense-in-depth gate, fail-closed on REFUTATION for both
        // families, Verified-only for the MEXI path:
        //   Refuted   -> a feasible point violating the cut was exhibited;
        //                reject outright (either family).
        //   Verified  -> the full integer box was enumerated and the cut
        //                holds; accept (either family).
        //   Unverified-> MEXI cuts: REJECT. Derivation trust produced false
        //                Optimal proofs on gen-ip002 (∞-bound coefficient
        //                tightening leaked +inf into the rhs; dual closed on
        //                a bad incumbent) and markshare1 (local-bound cMIR
        //                cuts, valid only in-subtree, applied globally;
        //                claimed Optimal 19 vs MIPLIB opt 1). The derivation
        //                bugs are now fixed at the source, but Verified-only
        //                stays: no enumeration, no global apply.
        //                NOGOODS: APPLY on derivation soundness. An
        //                assignment nogood is valid by induction: the node
        //                was pruned under global rows (valid) + local cuts
        //                (subtree-valid, and every point with the branch
        //                assignment lies in the subtree) + previously
        //                applied cuts (Verified-Mexi or nogood - all
        //                globally valid by the same induction) + sound
        //                propagation (row-implied bounds). build_nogood_
        //                from_branch_trail refuses trails that are not
        //                fully binary-representable, so the excluded set is
        //                exactly the node's box.
        const auto t_vb = Clock::now();
        const CutValidity vb = conflict_cut_check_binary(
            global_lp, learned, conflict_cut_opts.tol);
        diag.ms_check_binary += ms_since(t_vb);
        const auto t_vg = Clock::now();
        const CutValidity vg =
            (vb == CutValidity::Refuted)
                ? CutValidity::Refuted
                : conflict_cut_check_general(global_lp, learned,
                                             conflict_cut_opts.tol);
        diag.ms_check_general += ms_since(t_vg);
        if (vb == CutValidity::Refuted || vg == CutValidity::Refuted)
            return reject_learned();
        if (!as_nogood && vb != CutValidity::Verified &&
            vg != CutValidity::Verified)
            return reject_learned();
        // In-place append (or bound tighten): O(row nnz + shape index), not a
        // full global_lp rebuild. The rebuild path was 156-285 ms per nogood on
        // misc03 and 41-55% of that model's wall.
        const auto t_apply = Clock::now();
        apply_cuts_inplace(global_lp, {learned}, cut_cfg.cut);
        ++global_lp_generation;
        const double apply_ms = ms_since(t_apply);
        if (as_nogood) {
            diag.nogood_apply_ms += apply_ms;
            ++diag.nogood_cuts_global;
        } else {
            diag.conflict_analysis_ms += apply_ms;
            ++diag.conflict_cuts_global;
            refresh_conflict_cap();
        }
        return true;
    };
    // Files a learned clause: a unit is a global fact (applied to the root box,
    // which children intersect at creation), anything longer goes to the store.
    const auto store_learned_clause = [&](std::vector<ConflictLiteral> lits) -> ClauseResult {
        std::vector<ConflictLiteral> norm;
        const ClauseResult r = conflict_store.add_typed(std::move(lits), root_lo, root_hi, &norm);
        if (r == ClauseResult::Unit) {
            // A fact, not something to propagate only where a column already
            // differs from the root: apply it to the root box (children
            // intersect it at creation).
            const ConflictLiteral& u = norm.front();
            bool moved = false;
            if (u.upper && u.bound < root_hi[sz(u.var)] - 1e-9) { root_hi[sz(u.var)] = u.bound; moved = true; }
            if (!u.upper && u.bound > root_lo[sz(u.var)] + 1e-9) { root_lo[sz(u.var)] = u.bound; moved = true; }
            if (moved) {
                ++root_box_version;
                root_box_learned = true;
                ++diag.conflict_store_root_units;
            }
        }
        switch (r) {
            case ClauseResult::Inserted: ++diag.clause_inserted; break;
            case ClauseResult::Unit: ++diag.clause_units; break;
            case ClauseResult::AlreadyPresent: ++diag.clause_present; break;
            case ClauseResult::Redundant: ++diag.clause_redundant; break;
            case ClauseResult::Rejected: ++diag.clause_rejected; break;
            case ClauseResult::Contradiction:
                ++diag.clause_contradictions;
                root_clause_contradiction = true;
                break;
        }
        diag.conflict_store = conflict_store.stats();
        return r;
    };
    if (opts.initial_clauses != nullptr)
        for (const auto& clause : *opts.initial_clauses)
            (void)store_learned_clause(clause);
    // Ordinary nodes and strong-branch directions share the same independently
    // checked explanation. The LP must contain the actual tested box; a status
    // or a heuristic probe score alone never enters this path.
    const auto try_learn_farkas = [&](const model::LpProblem& tested_lp,
                                      const core::RawResult& tested_raw,
                                      bool rows_are_global, bool from_probe) {
        if (!opts.farkas_conflicts || !conflict_cut_opts.conflict_store ||
            !conflict_cut_opts.nogood_cuts || !rows_are_global ||
            tested_lp.n_rows() != global_lp.n_rows()) return false;
        const auto& ray = !tested_raw.dual_farkas_ray.multipliers.empty()
            ? tested_raw.dual_farkas_ray.multipliers : tested_raw.ray;
        std::size_t relaxed = 0;
        auto lits = farkas_conflict_clause(tested_lp, ray, opts.primal_feas_tol,
                                           store_int, root_lo, root_hi, &relaxed);
        if (lits.empty()) return false;
        ++diag.farkas_clauses;
        if (from_probe) ++diag.farkas_probe_clauses;
        diag.farkas_literals += lits.size();
        diag.farkas_relaxed_bounds += relaxed;
        const bool retained = clause_retained(store_learned_clause(std::move(lits)));
        if (!retained) ++diag.farkas_not_retained;
        return retained;
    };
    auto try_learn_nogood = [&](const PropTrail& trail_in, Index conflict_var = -1,
                                const std::vector<f64>* cur_lo = nullptr,
                                const std::vector<f64>* cur_hi = nullptr,
                                bool rows_are_global = false) {
        const auto t_nogood_total = Clock::now();
        struct NogoodTick {
            double& sink;
            Clock::time_point t;
            ~NogoodTick() { sink += ms_since(t); }
        } nogood_tick{diag.ms_nogood_total, t_nogood_total};
        const PropTrail& trail = trail_in;
        if (conflict_cut_opts.conflict_store && conflict_cut_opts.nogood_cuts) {
            // Persistent store: no LP row, no generation bump.
            std::vector<ConflictLiteral> lits;
            if (conflict_var >= 0 && cur_lo != nullptr && cur_hi != nullptr &&
                rows_are_global) {
                lits = explain_conflict_clause(trail, global_lp, global_lp.n_rows(),
                                               store_int, root_lo, root_hi, *cur_lo,
                                               *cur_hi, conflict_var);
                if (!lits.empty()) ++diag.conflict_store_explained;
            }
            if (lits.empty())
                lits = negated_branch_decisions(trail, store_int, root_lo, root_hi);
            store_learned_clause(std::move(lits));
            return;
        }
        // Separate switch from the Mexi path: --no-conflict-cut /
        // Classical turn both off via policy / CLI; the dense-binary
        // auto-off leaves nogoods on (no trail analysis involved). The cap
        // keeps global_lp growth bounded on infeasible-heavy trees.
        if (!conflict_cut_opts.nogood_cuts) return;
        if (diag.nogood_cuts_global >=
            static_cast<std::uint64_t>(conflict_cut_opts.max_nogood_cuts))
            return;
        const auto t_build = Clock::now();
        auto ng = build_nogood_from_branch_trail(trail, global_lp,
                                                 conflict_cut_opts.tol);
        diag.ms_nogood_build += ms_since(t_build);
        if (!ng) return;
        const auto t_val = Clock::now();
        (void)try_apply_validated_global_cut(*ng, /*as_nogood=*/true);
        diag.ms_nogood_validate += ms_since(t_val);
    };

    // Tree-restart bookkeeping (Latest).
    std::uint64_t nodes_since_incumbent_improve = 0;
    bool had_significant_incumbent_jump = false;
    f64 last_incumbent_for_restart = best_incumbent;
    int tree_restarts_done = 0;

    std::string reason = "node limit";
    bool stopped_early = false;
    // The bound of a node that was POPPED and then abandoned without pushing
    // its children. Such a node's subtree is represented in neither `open` nor
    // `plunge_stack`, so draining those two is not by itself a global bound --
    // this is what has to be folded in to keep it one. It stays at +inf while
    // no node has been abandoned, which makes the std::min below a no-op.
    f64 abandoned_bound = std::numeric_limits<f64>::infinity();
    f64 pruned_floor = std::numeric_limits<f64>::infinity();
    // Nodes whose LP gave neither a usable point nor a certificate. They keep
    // their certified bound (every bound site below includes them), leave the
    // frontier so the rest of the tree keeps being searched, and are retried
    // once by the cold primal route -- when the frontier runs dry or a global
    // row has changed the model since. A node that fails again is folded into
    // abandoned_bound and the search still carries on.
    std::vector<Node> deferred;
    f64 tree_sep_credit = 0.0;   // realized payoff of tree separation (see the gate)
    const auto deferred_floor = [&]() {
        f64 m = std::numeric_limits<f64>::infinity();
        for (const Node& d : deferred) m = std::min(m, d.bound);
        return m;
    };
    // After a Para-B&B phase that could not prove some LPs, force one serial
    // pass so those nodes are not permanently lost to abandoned_bound.
    diag.ms_before_search = ms_since(t0);
    root_span.reset();
    bool para_force_serial = false;
    // --- cost-model signals (para_bab.hpp) --------------------------------
    // Tracked by comparison rather than by stamping every incumbent write:
    // best_incumbent is assigned from many places in this function, and one
    // comparison per node is cheaper and far harder to get wrong than hooking
    // all of them.
    // Pruning bound contributed by ANOTHER portfolio arm. Never this arm's
    // answer -- only a licence to stop looking below it.
    // Objective granularity, computed once. `mip` is in the ORIGINAL objective
    // sense; the search works on sense*objective, and a positive scaling does
    // not change the step size, so the same g applies to the working bound.
    // From `problem`, NOT `mip`. mip carries implied-integrality marks, and
    // those assert only that a column is integral at SOME optimum -- not at
    // every feasible point, which is what granularity requires. Using them
    // here would be unsound, and implied integrality has already produced one
    // false Infeasible in this tree.
    const f64 obj_granularity = objective_granularity(problem);
    diag.objective_granularity = obj_granularity;
    const f64 offset_min = sense * problem.obj_offset;
    // Close a popped node whose subtree was NOT searched to the end -- an
    // integral point from an unproved LP (the LP optimum may be better and
    // fractional), or no branching candidate -- without losing the region:
    // below the cutoff its certified bound joins the global bound and the
    // tree can no longer count as exhausted; at or above it the region is
    // pruned exactly like any other node. This is what lets a proof skip the
    // old blanket requirement that every node LP be proved: every region
    // is either searched, pruned against a certified bound, or represented.
    const auto fold_unsearched_region = [&](const Node& region, const char* why) {
        if (have_incumbent) {
            const f64 inc_min = sense * best_incumbent;
            if (region.bound >= node_cutoff(inc_min, obj_granularity, offset_min,
                                            opts.gap_tol, opts.abs_gap_tol)) {
                if (region.bound < inc_min) {
                    ++diag.gap_prunes;
                    pruned_floor = std::min(pruned_floor, region.bound);
                }
                return;
            }
        }
        abandoned_bound = std::min(abandoned_bound, region.bound);
        stopped_early = true;
        reason = why;
        ++diag.unproved_regions_folded;
    };


    f64 cm_seen_incumbent = std::numeric_limits<f64>::quiet_NaN();
    std::uint64_t cm_last_improve_node = 0;
    while (root_clause_contradiction || !open.empty() || !plunge_stack.empty() || !deferred.empty()) {
        if (root_clause_contradiction) {
            // A valid global clause has no literal left in the root box.
            // It closes every outstanding region, including deferred or
            // previously abandoned LP work. The foreign-cutoff proof gate
            // below still distinguishes infeasibility from cutoff closure.
            while (!open.empty()) open.pop();
            plunge_stack.clear();
            deferred.clear();
            abandoned_bound = std::numeric_limits<f64>::infinity();
            stopped_early = false;
            break;
        }
        if (!deferred.empty()) {
            // Retry deferred nodes when nothing else is left, or as soon as a
            // global row has changed the model they failed on.
            const bool frontier_empty = open.empty() && plunge_stack.empty();
            std::vector<Node> keep;
            for (Node& d : deferred) {
                if (frontier_empty || d.defer_generation != global_lp_generation) {
                    ++diag.node_lp_retries;
                    open.push(std::move(d));
                } else {
                    keep.push_back(std::move(d));
                }
            }
            deferred = std::move(keep);
        }
        // The final proof gate accepts a closed gap even when the tree still
        // contains nodes. Apply the same test here so a proved incumbent does
        // not spend the remaining budget draining nodes that cannot improve
        // it. Include every outstanding region, not just open.top(): plunge
        // nodes and a previously abandoned region also constrain the bound.
        if (have_incumbent) {
            f64 remaining_bound = std::min({abandoned_bound, pruned_floor, deferred_floor()});
            if (!open.empty())
                remaining_bound = std::min(remaining_bound, open.top().bound);
            for (const Node& pending : plunge_stack)
                remaining_bound = std::min(remaining_bound, pending.bound);
            // The final proof gate strengthens a certified global bound to
            // the next attainable objective-lattice value. Do the same here
            // so a closed gap stops search before the wall-clock limit.
            remaining_bound = tighten_bound_to_granularity(
                remaining_bound, obj_granularity,
                sense * problem.obj_offset, opts.int_tol);
            const f64 inc_min = sense * best_incumbent;
            const f64 allowed_gap =
                opts.gap_tol * (1.0 + std::fabs(inc_min));
            if (std::isfinite(remaining_bound) &&
                remaining_bound >= inc_min - allowed_gap &&
                remaining_bound <= inc_min + allowed_gap) {
                reason = "global gap closed";
                break;
            }
        }
        // Whole-node-loop timer. The MILP timing block reported only "node LP"
        // and a grand total, so everything else the loop does was invisible --
        // on app1-1 that hid 27 s of a 40 s run.
        const auto t_loop_iter = Clock::now();
        struct LoopTick {
            double& sink;
            Clock::time_point t;
            ~LoopTick() { sink += ms_since(t); }
        } loop_tick{diag.ms_node_loop, t_loop_iter};
        ++diag.loop_iters_started;
        // --- portfolio: cancellation ------------------------------------
        // Polled per node, not per LP: a node is milliseconds, so this is
        // free, and a losing arm leaves within one node of a rival winning.
        if (core::cancel_requested(opts.cancel)) {
            reason = "cancelled (portfolio race lost)";
            stopped_early = true;
            SOR_ROUTE(1, "bab", "stop", "\"reason\":\"cancelled\"");
            break;
        }
        // --- portfolio: adopt a better incumbent from another arm ---------
        // Read as a CUTOFF only. Adopting another arm's objective prunes this
        // arm's tree without importing its solution vector: the portfolio
        // driver reports the pool's best point, so this worker never has to
        // claim a solution it did not construct.
        if (opts.pool != nullptr) {
            const f64 shared = opts.pool->best_objective();
            if (std::isfinite(shared) &&
                (!std::isfinite(external_cutoff) ||
                 (mip.maximize ? shared > external_cutoff
                               : shared < external_cutoff))) {
                // Kept STRICTLY separate from best_incumbent/best_x. Folding a
                // foreign objective into this arm's own incumbent made the arm
                // report an objective it had no point for -- measured on
                // blend2 as 8.1039 against a true optimum of 7.5990.
                external_cutoff = shared;
                diag.used_foreign_cutoff = true;
                ++diag.incumbents_adopted;
            }
        }
        if (have_incumbent && !(best_incumbent == cm_seen_incumbent)) {
            cm_seen_incumbent = best_incumbent;
            cm_last_improve_node = diag.nodes;
            {
                char ibuf[128];
                std::snprintf(ibuf, sizeof ibuf, "\"obj\":%.10g,\"node\":%llu",
                              best_incumbent,
                              static_cast<unsigned long long>(diag.nodes));
                SOR_ROUTE(1, "bab", "incumbent", ibuf);
            }
            // --- portfolio: publish, but only a pair we can VERIFY ---------
            // Five sites in this function assign best_incumbent/best_x, and
            // one of them (the foreign-cutoff adoption above) deliberately
            // does NOT update best_x. Hooking all five would eventually miss
            // one and publish an objective with somebody else's point, so
            // instead the pair is checked against the model before it is
            // shared. The recompute is O(nnz) but runs only on an improvement.
            // Validate in the CALLER'S space (`problem`), never this arm's
            // transformed `mip` -- see point_is_feasible_for().
            if (opts.pool != nullptr && !best_x.empty() &&
                portfolio_point_is_feasible(problem, best_x, opts.primal_feas_tol,
                                            opts.int_tol)) {
                const f64 verified = problem.objective(best_x);
                if (std::isfinite(verified) &&
                    opts.pool->publish_incumbent(verified, best_x))
                    ++diag.incumbents_published;
            }
        }
        apply_root_rc();
        // Root restart: reduced-cost fixing against the incumbent has shrunk
        // the root box a lot; hand it back to the wrapper to presolve again
        // (the tree carries every fixed column otherwise). Only while the
        // tree is still shallow (the good incumbent that enables the fixing
        // often arrives a few nodes in, from a neighborhood search; the
        // small tree discarded then costs little), only
        // in the original column space, and never inside a portfolio arm or
        // a sub-MIP.
        if (opts.root_restart && opts.root_restart_allowed &&
            diag.nodes <= opts.root_restart_max_nodes &&
            have_incumbent && best_x.size() == sz(n) && opts.pool == nullptr &&
            opts.sub_mip_depth == 0 && !diag.symmetry_diag.folding_applied &&
            root_free_ints_at_start > 0 &&
            (opts.time_limit_s <= 0.0 || seconds_left() > 0.2 * opts.time_limit_s)) {
            std::uint64_t free_now = 0;
            for (Index j = 0; j < n; ++j)
                if (!mip.is_integer.empty() && mip.is_integer[sz(j)] &&
                    root_hi[sz(j)] > root_lo[sz(j)])
                    ++free_now;
            const std::uint64_t fixed_now = root_free_ints_at_start - free_now;
            if (fixed_now >= opts.root_restart_min_fixed && fixed_now > 0 &&
                static_cast<f64>(fixed_now) >=
                    opts.root_restart_min_fixed_frac *
                        static_cast<f64>(root_free_ints_at_start)) {
                diag.restart_requested = true;
                diag.restart_col_lo = root_lo;
                diag.restart_col_hi = root_hi;
                diag.restart_columns_fixed = fixed_now;
                if (opts.restart_carry_state) {
                    RestartPayload& rp_out = diag.restart_payload;
                    if (mip_rows_before_cuts == mip.n_rows() &&
                        global_lp.n_rows() > mip.n_rows() &&
                        global_lp.n_cols() == mip.n_cols()) {
                        const auto& grp = global_lp.A.pattern.row_ptr();
                        const auto& gci = global_lp.A.pattern.col_idx();
                        std::size_t budget_nnz = 200000;
                        for (Index r = mip.n_rows(); r < global_lp.n_rows() &&
                                                       rp_out.cuts.size() < 400; ++r) {
                            const std::size_t len = sz(grp[sz(r) + 1] - grp[sz(r)]);
                            if (len > budget_nnz) break;
                            budget_nnz -= len;
                            CutRow row;
                            for (core::Offset k = grp[sz(r)]; k < grp[sz(r) + 1]; ++k) {
                                row.cols.push_back(gci[sz(k)]);
                                row.vals.push_back(global_lp.A.vals[sz(k)]);
                            }
                            row.row_lo = global_lp.row_lo[sz(r)];
                            row.row_hi = global_lp.row_hi[sz(r)];
                            row.name = "CARRIED";
                            rp_out.cuts.push_back(std::move(row));
                        }
                    }
                    rp_out.pseudocosts.down_sum = pc_down_sum;
                    rp_out.pseudocosts.up_sum = pc_up_sum;
                    rp_out.pseudocosts.down_count = pc_down_count;
                    rp_out.pseudocosts.up_count = pc_up_count;
                    rp_out.clauses = conflict_store.export_clauses();
                }
                reason = "root restart requested";
                stopped_early = true;
                break;
            }
        }
        // Latch the parallel switch once the search has shown it is tree-bound
        // rather than incumbent-bound. Never switches back: a frontier that
        // earned the workers does not stop deserving them, and flapping would
        // thrash the plunge stack.
        if (para_threads > 1 && !para_active) {
            ParaBabSearchState cm_state;
            cm_state.elapsed_s =
                std::chrono::duration<double>(Clock::now() - t0).count();
            cm_state.nodes_done = diag.nodes;
            cm_state.open_frontier = open.size();
            cm_state.nodes_since_incumbent = diag.nodes - cm_last_improve_node;
            cm_state.threads = para_threads;
            if (para_bab_should_activate(opts.para_bab_cost, cm_state)) {
                para_active = true;
                // Only if configured to: the phase gate already requires an
                // empty plunge stack, so by default dives and phases interleave
                // rather than one excluding the other.
                if (opts.para_bab_cost.disable_plunge_on_activate)
                    hybrid_nodes = false;
                diag.para_bab.activated = true;
                diag.para_bab.activated_at_s = cm_state.elapsed_s;
                diag.para_bab.activated_at_node = diag.nodes;
                diag.para_bab.serial_nodes = diag.nodes;
                char pbuf[128];
                std::snprintf(pbuf, sizeof pbuf,
                              "\"node\":%llu,\"threads\":%d",
                              static_cast<unsigned long long>(diag.nodes),
                              para_threads);
                SOR_ROUTE(1, "bab", "para_activated", pbuf);
            }
        }
        if (diag.nodes >= opts.max_nodes) {
            reason = "node limit (" + std::to_string(opts.max_nodes) + ")";
            SOR_ROUTE(1, "bab", "stop", "\"reason\":\"node_limit\"");
            break;
        }
        if (timed_out()) {
            reason = "time limit";
            SOR_ROUTE(1, "bab", "stop", "\"reason\":\"time_limit\"");
            break;
        }

        // ---- Para-B&B phase (arXiv:2604.09556): replicate + barrier ----
        // When enough open nodes exist and plunging is idle, expand a batch
        // of best-bound nodes in parallel. Merge is deterministic by pop
        // order. Workers use LP+branch only; heuristics stay serial.
        if (para_threads > 1 && para_active && !para_force_serial &&
            plunge_stack.empty() && open.size() >= 2 &&
            (opts.para_bab.max_phases == 0 ||
             diag.para_bab.phases < opts.para_bab.max_phases)) {
            const int workers =
                std::min(para_threads, static_cast<int>(open.size()));
            const int batch_n = std::min(
                workers * para_quota, static_cast<int>(open.size()));
            std::vector<Node> batch;
            batch.reserve(static_cast<std::size_t>(batch_n));
            for (int i = 0; i < batch_n; ++i) {
                Node pending = open.top();
                open.pop();
                if (intersect_root_box(pending)) batch.push_back(std::move(pending));
            }
            if (batch.empty()) continue;
            const f64 inc_min =
                have_incumbent ? sense * best_incumbent
                               : -std::numeric_limits<f64>::infinity();
            std::vector<ParaExpandOut> results(batch.size());
            // Captured once for the whole batch so every worker shares one
            // deadline (and the value stays deterministic across workers).
            const double para_lp_left =
                opts.time_limit_s > 0.0 ? seconds_left() : 0.0;
            if (opts.time_limit_s > 0.0 && !(para_lp_left > 0.0)) {
                for (auto& pending : batch) open.push(std::move(pending));
                reason = "time limit";
                SOR_ROUTE(1, "bab", "stop", "\"reason\":\"time_limit\"");
                break;
            }
            {
                std::vector<std::jthread> workers_jt;
                workers_jt.reserve(batch.size());
                for (std::size_t i = 0; i < batch.size(); ++i) {
                    workers_jt.emplace_back([&, i]() {
                        // Copy node: may requeue on unproved LP.
                        results[i] = para_expand_node(
                            global_lp, batch[i], opts, sense, inc_min,
                            have_incumbent, opts.int_tol, opts.primal_feas_tol,
                            static_cast<std::uint64_t>(i), para_lp_left,
                            obj_granularity, offset_min);
                    });
                }
            }  // jthreads join = barrier sync (Para-B&B phase end)
            ++diag.para_bab.phases;
            ++diag.para_bab.syncs;
            diag.para_bab.parallel_expansions += results.size();

            int para_infeas = 0, para_abandoned = 0, para_integer = 0,
                para_bound_prune = 0;
            for (const auto& r : results) {
                if (r.abandoned) ++para_abandoned;
                else if (r.infeasible) ++para_infeas;
                else if (r.integer_hit) ++para_integer;
                else if (r.children.empty() && !r.infeasible && !r.abandoned)
                    ++para_bound_prune;
            }
            {
                char pbuf[160];
                std::snprintf(
                    pbuf, sizeof pbuf,
                    "\"batch\":%d,\"phase\":%llu,\"infeas\":%d,"
                    "\"abandoned\":%d,\"integer\":%d,\"bound_prune\":%d",
                    batch_n, static_cast<unsigned long long>(diag.para_bab.phases),
                    para_infeas, para_abandoned, para_integer, para_bound_prune);
                SOR_ROUTE(1, "bab", "para_phase", pbuf);
            }

            // Deterministic merge in pop order.
            std::sort(results.begin(), results.end(),
                      [](const ParaExpandOut& a, const ParaExpandOut& b) {
                          return a.order_key < b.order_key;
                      });
            bool any_requeue = false;
            for (auto& r : results) {
                diag.lp_solves += r.lp_solves;
                diag.lp_iterations += r.lp_iterations;
                diag.lp_ms += r.lp_ms;
                branching_lp_work.solves += r.lp_solves;
                branching_lp_work.iterations += r.lp_iterations;
                branching_lp_work.ms += r.lp_ms;
                if (r.abandoned) {
                    // Do NOT fold into abandoned_bound: requeue for serial
                    // expansion (full node path) so proof stays sound.
                    open.push(std::move(batch[r.order_key]));
                    any_requeue = true;
                    continue;
                }
                ++diag.nodes;
                if (std::isfinite(r.pruned_bound)) {
                    ++diag.gap_prunes;
                    ++diag.para_gap_prunes;
                    pruned_floor = std::min(pruned_floor, r.pruned_bound);
                }
                if (r.infeasible) continue;
                if (r.integer_hit && std::isfinite(r.incumbent)) {
                    f64 obj;
                    if (accept_incumbent(std::move(r.x), true, obj)) {
                        ++diag.integer_feasible;
                    } else if (!std::isfinite(obj)) {
                        // Rejected point is not an infeasibility proof.
                        abandoned_bound = std::min(abandoned_bound, r.bound);
                        stopped_early = true;
                        reason = "parallel integer candidate failed original-model validation";
                    }
                }
                for (auto& c : r.children) {
                    if (have_incumbent) {
                        const f64 inc_m = sense * best_incumbent;
                        if (c.bound >= node_cutoff(inc_m, obj_granularity, offset_min,
                                                   opts.gap_tol, opts.abs_gap_tol)) {
                            if (c.bound < inc_m) {
                                ++diag.gap_prunes;
                                pruned_floor = std::min(pruned_floor, c.bound);
                            }
                            continue;
                        }
                    }
                    open.push(std::move(c));
                }
            }
            para_force_serial = any_requeue;
            if (any_requeue) {
                char pbuf[48];
                std::snprintf(pbuf, sizeof pbuf, "\"batch\":%d", batch_n);
                SOR_ROUTE(1, "bab", "para_requeue", pbuf);
            }
            continue;
        }
        para_force_serial = false;

        // Node selection: continue the current plunge if its inherited bound
        // isn't meaningfully worse than the best open node's, otherwise (or
        // once one side is empty) fall back to best-bound. Best-bound is the
        // ONLY thing that prunes or proves anything below -- this choice
        // only reorders which already-valid node gets expanded next.
        bool from_plunge = false;
        if (hybrid_nodes && !plunge_stack.empty()) {
            if (open.empty()) {
                from_plunge = true;
            } else {
                const f64 ob = open.top().bound;
                const f64 pb = plunge_stack.back().bound;
                from_plunge = !std::isfinite(ob) || !std::isfinite(pb) ||
                    pb <= ob + opts.plunge_bound_slack_rel * (1.0 + std::fabs(ob));
            }
        } else if (open.empty()) {
            from_plunge = true;  // hybrid disabled mid-run or open drained early
        }

        auto seg_last = Clock::now();
        const auto seg = [&](int k) {
            const auto now = Clock::now();
            diag.ms_seg[k] += std::chrono::duration<double, std::milli>(now - seg_last).count();
            seg_last = now;
        };
        seg(0);  // loop top -> pop
        const auto t_pop = Clock::now();
        diag.ms_node_loop_top += std::chrono::duration<double, std::milli>(
            t_pop - t_loop_iter).count();
        Node node;
        if (from_plunge) {
            node = std::move(plunge_stack.back());
            plunge_stack.pop_back();
            ++diag.plunge_nodes;
        } else {
            node = open.top();
            open.pop();
        }
        ++diag.nodes;
        const bool box_ok = intersect_root_box(node);
        diag.ms_node_pop += ms_since(t_pop);
        if (!box_ok) continue;
        seg(1);  // pop + restart check + gap bookkeeping -> bound prune

        // Incumbent-improve clock for tree restarts (Latest).
        if (have_incumbent && std::isfinite(best_incumbent)) {
            const bool better =
                !std::isfinite(last_incumbent_for_restart) ||
                (mip.maximize
                     ? best_incumbent > last_incumbent_for_restart + 1e-15
                     : best_incumbent < last_incumbent_for_restart - 1e-15);
            if (better) {
                if (!std::isfinite(last_incumbent_for_restart)) {
                    had_significant_incumbent_jump = true;
                } else {
                    const f64 scale =
                        1.0 + std::fabs(last_incumbent_for_restart);
                    const f64 rel =
                        std::fabs(best_incumbent - last_incumbent_for_restart) /
                        scale;
                    if (rel > opts.tree_restart_improve_rel)
                        had_significant_incumbent_jump = true;
                }
                nodes_since_incumbent_improve = 0;
                last_incumbent_for_restart = best_incumbent;
            } else {
                ++nodes_since_incumbent_improve;
            }
        } else {
            ++nodes_since_incumbent_improve;
        }

        // One-shot tree restart under Latest after a significant incumbent
        // jump followed by a long dry spell.
        if (milp_policy_is_latest(opts.policy) && opts.tree_restart &&
            tree_restarts_done < opts.tree_restart_max &&
            had_significant_incumbent_jump && have_incumbent &&
            nodes_since_incumbent_improve >= opts.tree_restart_node_gap) {
            // The frontier being discarded certified a bound for the whole
            // box: every point better than the incumbent lies in an open,
            // plunge or abandoned region, or in a region pruned with its
            // bound kept in pruned_floor. Seed the restarted root with it
            // (the stalled node just popped is still part of that frontier).
            // Not with a foreign cutoff: a region pruned against another
            // arm's objective left no floor behind.
            if (!diag.used_foreign_cutoff) {
                f64 frontier = std::min({abandoned_bound, pruned_floor,
                                         node.bound, sense * best_incumbent,
                                         deferred_floor()});
                for (const Node& pending : plunge_stack)
                    frontier = std::min(frontier, pending.bound);
                if (!open.empty()) frontier = std::min(frontier, open.top().bound);
                if (root_cert.raise(frontier, BoundCertificate::Source::Frontier))
                    ++diag.root_bound_raises;
            }
            while (!open.empty()) open.pop();
            plunge_stack.clear();
            deferred.clear();   // its floor is in `frontier` above

            // Reduced-cost fixing from a certified root LP (dual prices).
            {
                model::LpProblem root_lp = global_lp;
                root_lp.col_lo = root_lo;
                root_lp.col_hi = root_hi;
                engines::SimplexOptions ropts = opts.lp;
                ropts.verbose = false;
                engines::SimplexDiagnostics rsd;
                engines::SimplexBasis rbas;
                if (opts.time_limit_s > 0.0)
                    ropts.time_limit_s = std::max(std::numeric_limits<double>::min(),
                        opts.time_limit_s - ms_since(t0) / 1000.0);
                const auto t_restart_lp = Clock::now();
                auto rraw =
                    engines::solve_simplex(root_lp, ropts, rsd, &rbas);
                branching_lp_work.charge(rsd);
                branching_lp_work.ms += ms_since(t_restart_lp);
                ++diag.lp_solves;
                diag.lp_iterations += rsd.iterations;
                if (relaxation_proved(rraw, rsd, ropts) &&
                    static_cast<Index>(rraw.y.size()) == root_lp.n_rows() &&
                    static_cast<Index>(rraw.x.size()) == n) {
                    // The gap is measured from the certified bound, never
                    // the primal objective: a primal value above the LP
                    // optimum would shrink the gap and over-fix columns.
                    const f64 dual_min = node_lp_bound_min(rraw, sense);
                    if (root_cert.raise(dual_min,
                                        BoundCertificate::Source::LpOptimal))
                        ++diag.root_bound_raises;
                    const f64 inc_min = sense * best_incumbent;
                    const f64 gap = std::max(0.0, inc_min - dual_min);
                    std::vector<f64> rc(sz(n), 0.0);
                    for (Index j = 0; j < n; ++j)
                        rc[sz(j)] = global_lp.c[sz(j)];
                    const auto& rp = global_lp.A.pattern.row_ptr();
                    const auto& ci = global_lp.A.pattern.col_idx();
                    for (Index i = 0; i < global_lp.n_rows(); ++i) {
                        const f64 yi = rraw.y[sz(i)];
                        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1];
                             ++k)
                            rc[sz(ci[sz(k)])] -= yi * global_lp.A.vals[sz(k)];
                    }
                    if (global_lp.maximize)
                        for (f64& v : rc) v = -v;
                    const f64 rctol = opts.primal_feas_tol;
                    // Integer columns only. The test is a one-unit step, which
                    // is the next feasible integer when the occupied bound is
                    // integral. A continuous column can move a fraction of a
                    // unit and remain inside the gap, so it is left unchanged.
                    for (Index j = 0; j < n; ++j) {
                        if (sz(j) >= global_lp.is_integer.size() ||
                            !global_lp.is_integer[sz(j)])
                            continue;
                        double lo = root_lo[sz(j)];
                        double hi = root_hi[sz(j)];
                        if (!integer_reduced_cost_fix(lo, hi, rraw.x[sz(j)],
                                                      rc[sz(j)], gap, rctol))
                            continue;
                        root_lo[sz(j)] = lo;
                        root_hi[sz(j)] = hi;
                        ++diag.restart_rc_integer_fixed;
                    }
                    diag.restart_rc_col_lo = root_lo;
                    diag.restart_rc_col_hi = root_hi;
                    ++root_box_version;
                }
            }

            Node root;
            root.col_lo = root_lo;
            root.col_hi = root_hi;
            root.bound = root_cert.value;
            root.depth = 0;
            open.push(std::move(root));
            ++diag.tree_restarts;
            ++tree_restarts_done;
            nodes_since_incumbent_improve = 0;
            if (opts.verbose)
                std::printf("  [milp] tree restart #%d at node %llu\n",
                            tree_restarts_done,
                            static_cast<unsigned long long>(diag.nodes));
            continue;  // drop the stalled node; expand rebuilt root next
        }

        // Track the global dual bound. Under best-bound selection the popped
        // node's own bound IS the global bound, so its movement is exactly the
        // signal for whether the tree is still making progress. Plunge nodes
        // are not the global bound and must not be read as it.
        if (!from_plunge && std::isfinite(node.bound)) {
            const f64 scale = 1.0 + std::fabs(node.bound);
            if (!std::isfinite(last_dual_bound) ||
                node.bound > last_dual_bound + 1e-6 * scale) {
                last_dual_bound = node.bound;
                last_dual_node = diag.nodes;
            }
            refresh_live_gap(node.bound);
        } else if (have_incumbent) {
            f64 dual_min = last_dual_bound;
            if (!open.empty() && std::isfinite(open.top().bound))
                dual_min = open.top().bound;
            refresh_live_gap(dual_min);
        }
        // "Not converting nodes into proof" has two faces: the bound has
        // stopped moving, or it is moving but is still so far from the
        // incumbent that it will not arrive inside this budget.
        const bool bound_stalled =
            opts.dual_stall_window > 0 &&
            diag.nodes > last_dual_node + opts.dual_stall_window;
        bool gap_hopeless = false;
        if (have_incumbent && std::isfinite(last_dual_bound)) {
            const f64 inc_min = sense * best_incumbent;
            gap_hopeless = std::fabs(inc_min - last_dual_bound) /
                               (1.0 + std::fabs(inc_min)) >
                           opts.heuristic_focus_gap;
        }
        dual_stalled = bound_stalled || gap_hopeless;

        // Bound prune (minimize working objective = sense * original).
        {
            // Prune against whichever bound is tighter: this arm's own
            // incumbent, or a cutoff another arm published.
            const f64 cut = known_cutoff_orig();
            const bool cut_is_foreign = !have_incumbent || cut != best_incumbent;
            if (std::isfinite(cut)) {
                const f64 inc_min = sense * cut;
                if (node.bound >= node_cutoff(inc_min, obj_granularity, offset_min,
                                               opts.gap_tol, opts.abs_gap_tol)) {
                    if (node.bound < inc_min) {
                        ++diag.gap_prunes;
                        pruned_floor = std::min(pruned_floor, node.bound);
                    }
                    if (cut_is_foreign) ++diag.foreign_cutoff_prunes;
                    if (route_sample_node(diag.nodes)) {
                        char pbuf[96];
                        std::snprintf(pbuf, sizeof pbuf,
                                      "\"node\":%llu,\"kind\":\"cutoff\"",
                                      static_cast<unsigned long long>(diag.nodes));
                        SOR_ROUTE(3, "bab", "prune_bound", pbuf);
                    }
                    continue;
                }
            }
        }

        // Domain propagation (Achterberg thesis 2007, Ch.10.4): tighten this
        // node's bounds using the row structure BEFORE paying for an LP
        // solve. Rebuild from global_lp + this node's local cuts so local
        // inequalities participate without leaking to siblings.
        seg(2);  // bound prune -> node setup
        const auto t_node_setup = Clock::now();
        const std::uint64_t solve_generation = global_lp_generation;
        {
            core::RouteSpan setup_span(2, "bab", "node_setup", "node_setup", "",
                                       core::RouteLedgerBucket::Search);
            if (node.active_local.empty()) {
                // S3.3: node_lp already equals global_lp byte-for-byte when
                // it was last synced at THIS generation and no local cut has
                // touched it since (a sibling with local cuts may have run
                // in between) -- skip the O(nnz) copy and reuse it as is.
                // col_lo/col_hi get overwritten unconditionally below
                // regardless of which branch ran, so nothing else to do here.
                if (!node_lp_is_global_shaped ||
                    node_lp_generation != global_lp_generation) {
                    node_lp = global_lp;
                    node_lp_generation = global_lp_generation;
                    node_lp_is_global_shaped = true;
                }
            } else {
                const auto local_rows = managed_to_rows(node.active_local);
                diag.node_cuts_locally_applied +=
                    static_cast<std::uint64_t>(local_rows.size());
                node_lp = apply_cuts(global_lp, local_rows, cut_cfg.cut);
                ++diag.node_local_cut_rebuilds;
                node_lp_is_global_shaped = false;
                // The parent's basis, solved over global rows then these same
                // local rows, is the right warm start here. Any global row
                // added since (or a row count that no longer matches) makes
                // it describe a different LP: drop it.
                ++diag.node_local_cut_nodes;
                const bool basis_fits =
                    node.has_basis && node.basis_generation == solve_generation &&
                    static_cast<Index>(node.basis.basic.size()) == node_lp.n_rows() &&
                    node.basis.n_struct == node_lp.n_cols();
                if (basis_fits) {
                    ++diag.node_local_cut_basis_kept;
                } else {
                    if (node.has_basis) {
                        SOR_ROUTE_PATH(2, "bab", "node_setup", "basis_discard");
                    }
                    node.has_basis = false;
                    node.basis = engines::SimplexBasis{};
                }
            }
        }
        diag.ms_node_setup += ms_since(t_node_setup);
        seg(3);  // node setup (LP copy)
        // What the prepared LP session / factor checkpoints of this node are
        // valid for: the global rows at this generation AND, for a node under
        // subtree-local cuts, exactly this set of local rows. Descendants that
        // inherit the same rows share the key, so they share one session (built
        // once from global rows + local rows) instead of re-preparing an LP per
        // node -- which is what made inherited local rows too expensive.
        std::uint64_t session_key = global_lp_generation * 0x9e3779b97f4a7c15ull + 1;
        if (!node.active_local.empty()) {
            std::uint64_t h = 1469598103934665603ull;
            for (const ManagedCut& mc : node.active_local) {
                const CutRow& r = mc.row;
                for (const Index c : r.cols) h = (h ^ static_cast<std::uint64_t>(c + 1)) * 1099511628211ull;
                for (const f64 v : r.vals) {
                    std::uint64_t b;
                    std::memcpy(&b, &v, sizeof b);
                    h = (h ^ b) * 1099511628211ull;
                }
                std::uint64_t lo, hi;
                std::memcpy(&lo, &r.row_lo, sizeof lo);
                std::memcpy(&hi, &r.row_hi, sizeof hi);
                h = (h ^ lo) * 1099511628211ull;
                h = (h ^ hi) * 1099511628211ull;
            }
            session_key ^= h | 1ull;
            session_key = session_key * 0xff51afd7ed558ccdull + node.active_local.size();
        }
        const auto t_node_prop = Clock::now();
        const std::uint64_t inferences_before =
            diag.propagation_tightenings + diag.conflict_prop_tightenings +
            diag.conflict_store.tightenings;
        // Bound deductions propagation made in this node beyond its parent's
        // fixpoint (the branch's inference count).
        const auto child_inferences = [&]() {
            return static_cast<f64>(diag.propagation_tightenings +
                                    diag.conflict_prop_tightenings +
                                    diag.conflict_store.tightenings - inferences_before);
        };
        // A certified-empty child: one observation for its incoming branch.
        const auto observe_child_closed = [&](BranchOutcome why) {
            if (node.parent_branch_var < 0 || node.parent_branch_var >= n) return;
            (void)bstats.record_child(node.parent_branch_var, node.parent_branch_dir,
                                      node.parent_branch_distance, why, core::kNaN,
                                      child_inferences(), node.pc_consumed,
                                      node.pc_recorded_unit, opts.int_tol);
        };
        bool rows_at_fixpoint = false;
        const bool record_row_reasons = conflict_cut_opts.conflict_store &&
                                        conflict_cut_opts.nogood_cuts;
        if (opts.domain_propagation) {
            core::RouteSpan prop_span(2, "bab", "node_prop", "node_prop", "",
                                      core::RouteLedgerBucket::Search);
            PropagateResult prop;
            // Trail + Mexi analysis is expensive on dense binaries (enigma).
            // Cap learned cuts; after the budget, use plain propagation.
            const bool mexi_active =
                conflict_cut_opts.enabled &&
                static_cast<int>(diag.conflict_cuts_global) < conflict_cut_cap;
            const bool trail_rows = mexi_active || record_row_reasons;
            // A child of a propagated node differs from it by the bound it
            // branched on, so only the rows of that column need visiting --
            // provided the parent's fixpoint still holds for THIS model: the
            // same global rows (no nogood/cut since), the same root box (no
            // reduced-cost tightening since), no local rows.
            const bool event_mode =
                opts.event_propagation && node.prop_valid &&
                node.active_local.empty() &&
                node.prop_generation == global_lp_generation &&
                node.prop_root_version == root_box_version;
            bool converged = false;
            if (event_mode) {
                if (prop_index_generation != global_lp_generation) {
                    prop_index = build_column_row_index(global_lp);
                    prop_index_generation = global_lp_generation;
                }
                const std::uint64_t visit_cap = std::max<std::uint64_t>(
                    1000, 8ull * static_cast<std::uint64_t>(global_lp.n_rows()));
                prop = propagate_bounds_events(
                    node_lp, prop_index, node.col_lo, node.col_hi, node.dirty,
                    prop_scratch, trail_rows ? &node.prop_trail : nullptr,
                    node.depth, opts.primal_feas_tol, visit_cap);
                converged = prop.feasible &&
                            static_cast<std::uint64_t>(prop.rounds) < visit_cap;
                ++diag.prop_event_nodes;
                diag.prop_event_visits += static_cast<std::uint64_t>(prop.rounds);
            } else {
                if (trail_rows) {
                    prop = propagate_bounds_trail(
                        node_lp, node.col_lo, node.col_hi, &node.prop_trail,
                        node.depth, opts.primal_feas_tol, opts.propagation_max_rounds);
                } else {
                    prop = propagate_bounds(node_lp, node.col_lo, node.col_hi,
                                            opts.primal_feas_tol,
                                            opts.propagation_max_rounds);
                }
                // rounds is zero-based when a fixpoint was reached early and
                // equals the cap when the allowance ran out.
                converged = prop.feasible && prop.rounds < opts.propagation_max_rounds;
                ++diag.prop_full_nodes;
            }
            node.prop_valid = converged && node.active_local.empty();
            rows_at_fixpoint = converged;
            node.prop_generation = global_lp_generation;
            node.prop_root_version = root_box_version;
            node.dirty.clear();
            diag.propagation_tightenings += prop.tightened;
            if (!prop.feasible) {
                ++diag.propagation_prunes;
                observe_child_closed(BranchOutcome::CertifiedInfeasible);
                if (mexi_active) {
                    ConflictAnalysisContext cctx;
                    // Analyze against the global LP only. node_lp may contain
                    // local tree cuts (GMI under branched bounds); those must
                    // not seed a cut that is then applied to global_lp.
                    cctx.lp = &global_lp;
                    cctx.col_lo = &node.col_lo;
                    cctx.col_hi = &node.col_hi;
                    cctx.trail = &node.prop_trail;
                    cctx.conflict_var = prop.conflict_var;
                    cctx.conflict_row = prop.conflict_row;
                    cctx.n_global_rows = global_lp.n_rows();
                    {
                        const auto t_mexi = Clock::now();
                        auto learned = analyze_conflict_cuts(
                            cctx, conflict_cut_opts, diag.conflict_cut_diag);
                        diag.conflict_analysis_ms += ms_since(t_mexi);
                        if (learned) {
                            (void)try_apply_validated_global_cut(
                                *learned, /*as_nogood=*/false);
                        }
                    }
                }
                try_learn_nogood(node.prop_trail, prop.conflict_var, &node.col_lo,
                                 &node.col_hi, node.active_local.empty());
                if (route_sample_node(diag.nodes)) {
                    char pbuf[80];
                    std::snprintf(pbuf, sizeof pbuf, "\"node\":%llu",
                                  static_cast<unsigned long long>(diag.nodes));
                    SOR_ROUTE(3, "bab", "prune_prop_infeas", pbuf);
                }
                continue;
            }
        }

        // Graph and clause deductions, then back to the rows: the propagators
        // feed each other (a clique forces a binary, which is a new bound for
        // the rows it sits in, which can force a clause's last literal, ...),
        // so they are run to a fixpoint before the LP, not once in a fixed
        // order. Each round is bounded and a change by graph or store sends the
        // node back through row propagation over the whole node LP.
        //
        // Conflict propagation sees every binary the rows just fixed: each
        // fixed literal forces its conflict neighbours and its clique partners
        // false; that is reasoning across a whole propagation cascade (probing
        // implications) and across cardinality (cliques), neither of which a
        // row-at-a-time sweep can reach. Like row propagation it only narrows
        // node.col_lo/col_hi.
        {
            // One shared driver runs the row, graph and clause propagators to an
            // explicit state (Stable / Infeasible / Pending); see
            // domain_fixpoint.hpp. Rows start stale only if the row pass above
            // did not reach its own fixpoint.
            const bool use_graph = opts.conflict_propagation && !conflict_graph.empty();
            const bool use_store = conflict_cut_opts.conflict_store && !conflict_store.empty();
            if (use_graph || use_store) {
                DomainFixpoint fixpoint;
                bool rows_converged_now = rows_at_fixpoint;
                int id_rows = -1, id_graph = -1, id_store = -1;
                Index fixpoint_conflict_var = -1;
                // Every accepted bound update of this fixpoint, with its source.
                // The row propagator subscribes: once it has been at a fixpoint it
                // revisits only the rows of the columns other propagators have
                // moved since, instead of sweeping the model again.
                DomainEventLog elog;
                std::size_t rows_cursor = 0;
                bool rows_base_valid = rows_at_fixpoint;
                const bool incremental_ok = opts.event_propagation && node.active_local.empty();
                if (incremental_ok && prop_index_generation != global_lp_generation) {
                    prop_index = build_column_row_index(global_lp);
                    prop_index_generation = global_lp_generation;
                }
                if (opts.domain_propagation)
                    id_rows = fixpoint.add("rows", [&]() {
                        const std::vector<f64> lo0 = node.col_lo, hi0 = node.col_hi;
                        PropagateResult again;
                        FixpointStep st;
                        if (incremental_ok && rows_base_valid) {
                            const std::vector<Index> dirty = elog.vars_since(rows_cursor);
                            if (dirty.empty()) {
                                rows_cursor = elog.size();
                                return st;   // nothing moved since the last fixpoint
                            }
                            const std::uint64_t visit_cap = std::max<std::uint64_t>(
                                1000, 8ull * static_cast<std::uint64_t>(global_lp.n_rows()));
                            again = propagate_bounds_events(
                                node_lp, prop_index, node.col_lo, node.col_hi, dirty, prop_scratch,
                                record_row_reasons ? &node.prop_trail : nullptr,
                                node.depth, opts.primal_feas_tol, visit_cap);
                            st.converged = again.feasible &&
                                           static_cast<std::uint64_t>(again.rounds) < visit_cap;
                            ++diag.fixpoint_incremental_row_steps;
                        } else {
                            again = propagate_bounds_trail(
                                node_lp, node.col_lo, node.col_hi,
                                record_row_reasons ? &node.prop_trail : nullptr, node.depth,
                                opts.primal_feas_tol, opts.propagation_max_rounds);
                            st.converged = again.feasible && again.rounds < opts.propagation_max_rounds;
                            ++diag.fixpoint_full_row_steps;
                        }
                        diag.propagation_tightenings += again.tightened;
                        st.feasible = again.feasible;
                        if (!again.feasible) fixpoint_conflict_var = again.conflict_var;
                        st.changed = again.tightened > 0;
                        rows_converged_now = st.converged;
                        elog.record_diff(lo0, hi0, node.col_lo, node.col_hi, node.depth, id_rows);
                        rows_cursor = elog.size();
                        rows_base_valid = st.converged && st.feasible;
                        return st;
                    });
                if (use_graph)
                    id_graph = fixpoint.add("graph", [&]() {
                        const std::vector<f64> lo0 = node.col_lo, hi0 = node.col_hi;
                        std::uint64_t forced = 0;
                        const auto t_cprop = Clock::now();
                        const bool ok = propagate_conflicts(conflict_graph, node.col_lo,
                                                            node.col_hi, forced);
                        diag.ms_conflict_prop += ms_since(t_cprop);
                        diag.conflict_prop_tightenings += forced;
                        FixpointStep st;
                        st.feasible = ok;
                        st.changed = forced > 0;
                        if (st.changed)
                            elog.record_diff(lo0, hi0, node.col_lo, node.col_hi, node.depth, id_graph);
                        return st;
                    });
                if (use_store)
                    id_store = fixpoint.add("store", [&]() {
                        // Learned bound disjunctions: unit propagation over this
                        // node's box; a violated clause closes the node.
                        const std::vector<f64> lo0 = node.col_lo, hi0 = node.col_hi;
                        const std::uint64_t before = conflict_store.stats().tightenings;
                        bool converged = true;
                        const bool ok = conflict_store.propagate(node.col_lo, node.col_hi,
                                                                 root_lo, root_hi, &converged);
                        const std::uint64_t moved = conflict_store.stats().tightenings - before;
                        diag.conflict_store = conflict_store.stats();
                        FixpointStep st;
                        st.feasible = ok;
                        st.changed = moved > 0;
                        st.converged = converged;
                        if (st.changed)
                            elog.record_diff(lo0, hi0, node.col_lo, node.col_hi, node.depth, id_store);
                        return st;
                    });
                std::vector<char> stale(static_cast<std::size_t>(fixpoint.size()), 1);
                if (id_rows >= 0) stale[static_cast<std::size_t>(id_rows)] = rows_at_fixpoint ? 0 : 1;
                const FixpointRun fr = fixpoint.run(16, stale);
                diag.prop_fixpoint_rounds += static_cast<std::uint64_t>(fr.changing_steps);
                if (fr.status == FixpointStatus::Infeasible) {
                    ++diag.fixpoint_infeasible;
                    observe_child_closed(BranchOutcome::CertifiedInfeasible);
                    if (fr.infeasible_id == id_rows) {
                        ++diag.propagation_prunes;
                        // A row conflict reached through graph/store deductions
                        // is as useful as the initial row pass. The explainer
                        // rejects any missing reason chain and then falls back
                        // to the complete (sound) branch-decision nogood.
                        try_learn_nogood(node.prop_trail, fixpoint_conflict_var,
                                         &node.col_lo, &node.col_hi, node.active_local.empty());
                    } else if (fr.infeasible_id == id_graph) {
                        ++diag.conflict_prop_prunes;
                        try_learn_nogood(node.prop_trail);
                    } else if (fr.infeasible_id == id_store) {
                        ++diag.conflict_store_prunes;
                    }
                    continue;
                }
                if (fr.status == FixpointStatus::Pending) {
                    ++diag.fixpoint_pending;
                    // A deferred clause scan does not invalidate row work
                    // proved complete on this exact box. Only inherit rows
                    // when that propagator itself is neither stale nor unfinished.
                    node.prop_valid = id_rows >= 0 && rows_converged_now &&
                        !fr.stale[static_cast<std::size_t>(id_rows)] && node.active_local.empty();
                } else {
                    ++diag.fixpoint_stable;
                    // Every propagator has run on the final box, rows last after
                    // any change: the row fixpoint holds for this node.
                    node.prop_valid = id_rows >= 0 && rows_converged_now &&
                                      node.active_local.empty();
                }
            }
        }

        node_lp.col_lo = node.col_lo;
        node_lp.col_hi = node.col_hi;
        if (node.depth == 0 && unfinished_root_lp &&
            same_root_lp_structure(unfinished_root_lp->problem, node_lp)) {
            node.basis = std::move(unfinished_root_lp->basis);
            node.has_basis = true;
            unfinished_root_lp.reset();
            ++diag.root_lp_warm_handoffs;
            SOR_ROUTE(1, "bab", "root_lp_warm_handoff",
                      "\"source\":\"interrupted_cut_loop_phase2\"");
        }
        engines::SimplexOptions lp_opts = opts.lp;
        lp_opts.verbose = false;
        // The simplex default iteration cap is sized for a single standalone
        // LP.  A warm-started B&B child can need substantially more pivots
        // after a bound change; let the wall-clock budget, rather than that
        // small per-LP cap, be the primary limiter.
        if (lp_opts.max_iterations == 0) {
            const std::uint64_t work_size =
                static_cast<std::uint64_t>(node_lp.n_rows()) +
                static_cast<std::uint64_t>(node_lp.n_cols()) +
                static_cast<std::uint64_t>(node_lp.n_rows());
            lp_opts.max_iterations = std::max<std::uint64_t>(100000,
                                                               200ull * work_size);
        }
        if (opts.time_limit_s > 0.0) {
            const double elapsed =
                std::chrono::duration<double>(Clock::now() - t0).count();
            const double left = opts.time_limit_s - elapsed;
            if (!(left > 0.0)) {
                // The node was popped but never solved: put it back so its
                // bound stays in the final drain. Dropping it lost the
                // best-bound (minimum) region -- an optimistic global bound,
                // and a false "tree exhausted" if it was the last node.
                open.push(std::move(node));
                reason = "time limit";
                SOR_ROUTE(1, "bab", "stop", "\"reason\":\"time_limit\"");
                break;
            }
            lp_opts.time_limit_s = left;
        }
        engines::SimplexDiagnostics sd;
        engines::SimplexBasis node_basis;
        diag.ms_node_prop += ms_since(t_node_prop);
        seg(4);  // propagation + conflict propagation + lp options
        const auto t_lp = Clock::now();
        core::RawResult lp_raw;
        engines::SimplexDiagnostics lp_work;
        std::uint64_t lp_attempts = 0;
        const auto record_lp_attempt = [&](const engines::SimplexDiagnostics& attempt) {
            engines::accumulate_simplex_work(lp_work, attempt);
            branching_lp_work.charge(attempt);
            ++lp_attempts;
        };
        const auto try_node_polish = [&](core::RawResult& trial_raw,
                                         engines::SimplexDiagnostics& detail,
                                         const engines::SimplexOptions& options) {
            const double before = detail.primal_residual;
            Index corrected = 0;
            if (!polish_relaxation_primal(node_lp, options, trial_raw,
                                              detail, &corrected))
                return;
            char pbuf[160];
            std::snprintf(pbuf, sizeof pbuf,
                          "\"node\":%llu,\"corrected\":%d,"
                          "\"before\":%.9g,\"after\":%.9g",
                          static_cast<unsigned long long>(diag.nodes),
                          corrected, before, detail.primal_residual);
            SOR_ROUTE(1, "bab", "node_lp_polished", pbuf);
            if (opts.verbose)
                std::printf("  [milp] node LP polished %d column(s): "
                            "primal %.9g -> %.9g\n",
                            corrected, before, detail.primal_residual);
        };
        {
            core::RouteSpan nlp_span(1, "bab", "node_lp", "node_lp", "",
                                    core::RouteLedgerBucket::NodeLp);
        const auto same_proved_root_lp = [&]() {
            if (node.depth != 0 || !proved_root_lp) return false;
            const auto& old = proved_root_lp->problem;
            return same_root_lp_structure(old, node_lp) &&
                   old.col_lo == node_lp.col_lo &&
                   old.col_hi == node_lp.col_hi;
        };
        if (node.lp_failures > 0) {
            // Retry: a cold primal solve through the dispatcher, not the warm
            // dual route that just failed on this node.
            lp_opts.method = engines::SimplexMethod::Primal;
            node.has_basis = false;
        }
        bool root_workspace_transferred = false;
        // Numeric preparation belongs to the matrix/objective, while proof
        // belongs to the complete relaxation box. Propagation/RC fixing can
        // invalidate the old proof without changing B or its preparation.
        if (node.depth == 0 && proved_root_lp && proved_root_lp->session &&
            same_root_lp_structure(proved_root_lp->problem, node_lp)) {
            node_session = std::move(proved_root_lp->session);
            node_session_generation = session_key;
            node_session_local.clear();
            for (const auto& mc : node.active_local)
                node_session_local.push_back(mc.row);
            session_state_is_last_node = false;
            if (!node.has_basis) {
                node.basis = proved_root_lp->basis;
                node.has_basis = true;
            }
            root_workspace_transferred = true;
            ++diag.root_lp_session_handoffs;
            SOR_ROUTE(1, "bab", "root_lp_session_handoff",
                      "\"source\":\"proved_cut_loop_lp\"");
        }
        if (same_proved_root_lp()) {
            ++diag.root_lp_reuses;
            lp_raw = std::move(proved_root_lp->raw);
            sd = std::move(proved_root_lp->diag);
            node_basis = std::move(proved_root_lp->basis);
            session_state_is_last_node = root_workspace_transferred;
            proved_root_lp.reset();
            // The cut-loop proof was made on this same unpresolved LP.
            // Reusing it avoids a second cold root solve while retaining the
            // exact same status and dual evidence for all proof gates below.
            lp_opts.presolve = false;
            try_node_polish(lp_raw, sd, lp_opts);
            SOR_ROUTE(1, "bab", "root_lp_reused",
                      "\"source\":\"proved_cut_loop_lp\"");
        } else if (!node.has_basis ||
                   lp_opts.method == engines::SimplexMethod::Primal) {
            // The root has no compatible warm start. Let the normal simplex
            // dispatcher use presolve and its dual/primal fallback, then lift
            // the resulting basis back to the original model indices.
            if (node.has_basis)
                lp_opts.presolve = false;
            session_state_is_last_node = false;
            lp_raw = engines::solve_simplex(node_lp, lp_opts, sd, &node_basis);
            record_lp_attempt(sd);
            try_node_polish(lp_raw, sd, lp_opts);
            if (node_lp_status_proves_infeasible(lp_raw.proposed_status,
                                                  root_relaxation_bounded) &&
                !node_lp_infeasibility_proved(node_lp, lp_raw,
                                               opts.primal_feas_tol,
                                               root_relaxation_bounded) &&
                lp_opts.presolve && !timed_out()) {
                // A presolve terminal status has no reconstructed ray. Retry
                // the original node LP without presolve to seek one.
                SOR_ROUTE(1, "bab", "node_lp_infeasibility_retry",
                          "\"reason\":\"missing_or_invalid_ray\"");
                engines::SimplexOptions retry_opts = lp_opts;
                retry_opts.presolve = false;
                if (opts.time_limit_s > 0.0)
                    retry_opts.time_limit_s = std::max(
                        std::numeric_limits<double>::min(), opts.time_limit_s -
                        std::chrono::duration<double>(Clock::now() - t0).count());
                engines::SimplexDiagnostics retry_sd;
                engines::SimplexBasis retry_basis;
                auto retry_raw = engines::solve_simplex(
                    node_lp, retry_opts, retry_sd, &retry_basis);
                record_lp_attempt(retry_sd);
                ++diag.lp_fallbacks;
                lp_raw = std::move(retry_raw);
                sd = std::move(retry_sd);
                node_basis = std::move(retry_basis);
                try_node_polish(lp_raw, sd, retry_opts);
            }
        } else {
            // Bound changes preserve the row/column structure, so warm-start
            // child nodes with dual simplex and the parent's basis.
            // WP-J: product vs FT is opts.lp.update_method (passed via lp_opts).
            ++diag.warm_start_attempts;
            SOR_ROUTE_PATH(2, "bab", "warm_start", "warm_start");
            lp_opts.presolve = false;
            // EXTEND a basis recorded before rows were added.
            //
            // Every learned nogood (and every globally applied cut) appends
            // rows to the shared LP, which changes m. A stored node basis then
            // fails solve_dual_simplex's size check and the node re-solves
            // COLD. Measured on milo-v12-6-r2-40-1: warm starts 81/174, and
            // with --no-nogood-cuts 249/249 -- so half the tree was paying a
            // cold solve for rows it had never seen. Node throughput differed
            // by 2.2x (175 -> 393 nodes) and total LP work by 2.8x
            // (292,189 -> 105,574 iterations) for the identical objective.
            //
            // Appending each new row's LOGICAL as basic is always a valid
            // extension: it keeps the basis matrix nonsingular (the new column
            // is a unit vector on a new row) and starts the added rows
            // non-binding, which is the right initial state for a cut -- dual
            // simplex pivots them in only if they are violated.
            {
                const Index ns = node.basis.n_struct;
                const Index m_old =
                    static_cast<Index>(node.basis.basic.size());
                const Index m_now = node_lp.n_rows();
                if (ns == node_lp.n_cols() && m_now > m_old && m_old > 0) {
                    node.basis.status.resize(sz(ns + m_now),
                                             engines::NonbasicStatus::Basic);
                    node.basis.basic.reserve(sz(m_now));
                    for (Index r = m_old; r < m_now; ++r)
                        node.basis.basic.push_back(ns + r);
                    ++diag.warm_start_extended;
                }
            }
            if (lp_opts.method == engines::SimplexMethod::Auto)
                lp_opts.method = engines::SimplexMethod::Dual;
            // S3.1: this basis came from the parent node, not a cold logical
            // start, so reset_weights() would otherwise pay the exact m-BTRAN
            // rebuild on every single warm-started node.
            lp_opts.warm_dse_reset = true;
            // The prepared session is valid for one global_lp generation: a
            // global cut/nogood appends rows or tightens a row side, and
            // either makes the prepared copy stale. Only a node whose LP IS
            // global_lp (no local rows) may use it: a local-cut node_lp can
            // have the same row count with different rows.
            const auto same_local_rows = [&](const std::vector<CutRow>& a) {
                if (a.size() != node.active_local.size()) return false;
                for (std::size_t q = 0; q < a.size(); ++q) {
                    const CutRow& b = node.active_local[q].row;
                    if (a[q].cols != b.cols || a[q].vals != b.vals ||
                        a[q].row_lo != b.row_lo || a[q].row_hi != b.row_hi)
                        return false;
                }
                return true;
            };
            if (node_session && node_session_generation == session_key &&
                !same_local_rows(node_session_local)) {
                ++diag.session_key_collisions;   // hash matched, rows did not
                node_session.reset();
            }
            if (!node_session || node_session_generation != session_key) {
                // Stash the outgoing session (only if it is still current for the
                // global rows), then look for an exact match before building.
                if (node_session && global_lp.nnz() > 2000000) {
                    node_session.reset();   // too large to keep several prepared copies
                }
                if (node_session) {
                    session_cache.erase(
                        std::remove_if(session_cache.begin(), session_cache.end(),
                                       [&](const CachedSession& c) {
                                           return c.key == node_session_generation;
                                       }),
                        session_cache.end());
                    CachedSession c;
                    c.key = node_session_generation;
                    c.local = std::move(node_session_local);
                    c.session = std::move(node_session);
                    c.used = ++session_cache_tick;
                    session_cache.push_back(std::move(c));
                    while (session_cache.size() > kSessionCacheMax) {
                        auto oldest = session_cache.begin();
                        for (auto it = session_cache.begin(); it != session_cache.end(); ++it)
                            if (it->used < oldest->used) oldest = it;
                        session_cache.erase(oldest);
                        ++diag.session_cache_evictions;
                    }
                }
                node_session_local.clear();
                node_session.reset();
                for (auto it = session_cache.begin(); it != session_cache.end(); ++it) {
                    if (it->key == session_key && same_local_rows(it->local)) {
                        node_session = std::move(it->session);
                        node_session_local = std::move(it->local);
                        session_cache.erase(it);
                        ++diag.session_cache_hits;
                        break;
                    }
                }
                if (!node_session) {
                    node_session = std::make_unique<engines::DualProbeSession>(
                        node.active_local.empty() ? global_lp : node_lp, lp_opts);
                    for (const ManagedCut& mc : node.active_local)
                        node_session_local.push_back(mc.row);
                    if (!node.active_local.empty()) ++diag.local_session_builds;
                    ++diag.lp_session_builds_prepared;
                }
                node_session_generation = session_key;
                session_state_is_last_node = false;
                ++diag.lp_session_builds;
            }
            // Incumbent cutoff: stop the dual as soon as its multipliers prove
            // the node cannot beat the incumbent (engines/simplex.hpp
            // objective_limit). The early stop is only a claim; it is
            // certified below and, if the certificate falls short, the node
            // LP is finished normally.
            const f64 node_cutoff_min = cutoff_known()
                ? node_cutoff(known_cutoff_min(), obj_granularity, offset_min,
                              opts.gap_tol, opts.abs_gap_tol)
                : std::numeric_limits<f64>::infinity();
            if (opts.node_lp_cutoff && std::isfinite(node_cutoff_min))
                lp_opts.objective_limit = node_cutoff_min;
            {
                // Starting factor/weights for node.basis: this session's own
                // final factor whenever it matches this starting basis (no
                // copy), else the parent's checkpoint when it is
                // still cached and current, else none (basis alone).
                const engines::FactorCarrier* warm_factor = nullptr;
                const std::vector<f64>* warm_weights = nullptr;
                // Owns the checkpoint until the solve below has copied from
                // it (this node may hold the last reference).
                const std::shared_ptr<CheckpointHolder> checkpoint_in_use =
                    std::move(node.checkpoint);
                if (node_session->final_factor().has_factor &&
                    node_session->final_factor().basis == node.basis.basic) {
                    warm_factor = &node_session->final_factor();
                    if (node_session->final_weights_basis() == node.basis.basic)
                        warm_weights = &node_session->final_weights();
                    ++diag.checkpoint_immediate;
                } else if (checkpoint_in_use) {
                    const auto& h = *checkpoint_in_use;
                    if (h.data && h.generation == session_key &&
                        h.data->factor.basis == node.basis.basic) {
                        warm_factor = &h.data->factor;
                        if (h.data->weights_basis == node.basis.basic)
                            warm_weights = &h.data->weights;
                        ++diag.checkpoint_cached;
                    } else {
                        ++diag.checkpoint_unusable;  // evicted or stale
                    }
                }
                node_session->set_column_bounds(node_lp.col_lo, node_lp.col_hi);
                lp_raw = node_session->solve(lp_opts, sd, &node_basis, &node.basis,
                                             warm_weights, warm_factor);
                record_lp_attempt(sd);
                session_state_is_last_node = true;
                ++diag.lp_session_solves;
                if (!node.active_local.empty()) ++diag.local_session_solves;
            }
            if (lp_raw.termination_reason == "objective limit") {
                ++diag.node_lp_cutoff_exits;
                bool certified = lp_raw.y.size() == sz(node_lp.n_rows());
                if (certified) {
                    std::vector<f64> y_min(lp_raw.y.size());
                    for (std::size_t i = 0; i < y_min.size(); ++i)
                        y_min[i] = sense * lp_raw.y[i];
                    const auto t_safe = Clock::now();
                    const auto safe = certify::safe_lagrangian_lower_bound(
                        node_lp, y_min, node_lp.col_lo, node_lp.col_hi);
                    diag.node_lp_safe_bound_ms += ms_since(t_safe);
                    certified = safe.finite && safe.value >= node_cutoff_min;
                }
                if (!certified) {
                    engines::SimplexOptions finish = lp_opts;
                    finish.objective_limit = std::numeric_limits<f64>::infinity();
                    bool finish_has_time = true;
                    if (opts.time_limit_s > 0.0) {
                        const double left = opts.time_limit_s -
                            std::chrono::duration<double>(Clock::now() - t0).count();
                        finish_has_time = left > 0.0;
                        finish.time_limit_s = std::max(std::numeric_limits<double>::min(), left);
                    }
                    if (finish_has_time) {
                        ++diag.node_lp_cutoff_resolves;
                        const engines::SimplexBasis restart = node_basis;
                        if (node_session && session_state_is_last_node &&
                            node_session_generation == session_key) {
                            const auto* factor = &node_session->final_factor();
                            const auto* weights = &node_session->final_weights();
                            lp_raw = node_session->solve(finish, sd, &node_basis, &restart,
                                weights, factor);
                            ++diag.lp_session_solves;
                            if (!node.active_local.empty()) ++diag.local_session_solves;
                        } else {
                            lp_raw = engines::solve_dual_simplex(
                                node_lp, finish, sd, &node_basis, &restart);
                            session_state_is_last_node = false;
                        }
                        record_lp_attempt(sd);
                    }
                }
            }
            lp_opts.objective_limit = std::numeric_limits<f64>::infinity();
            diag.warm_start_hits += sd.warm_starts;
            try_node_polish(lp_raw, sd, lp_opts);

            // A node bound is usable only when the LP has passed all three
            // optimality checks. A feasible point or an unproved "Optimal" is
            // not a lower bound and must never drive pruning/branching.
            // A certified cutoff exit is pruned below; it needs no re-solve.
            const bool needs_fallback = !relaxation_proved(lp_raw, sd, lp_opts) &&
                lp_raw.termination_reason != "objective limit" &&
                !node_lp_infeasibility_proved(node_lp, lp_raw,
                                               opts.primal_feas_tol,
                                               root_relaxation_bounded) &&
                (!timed_out() || lp_raw.proposed_status != core::Status::Interrupted);
            if (needs_fallback) {
                if (opts.verbose)
                    std::printf("  [milp] node %llu warm dual unproved after %llu "
                                "iterations: %s (status %d, primal res %.3g, dual res "
                                "%.3g, gap %.3g, dual bound %s)\n",
                                static_cast<unsigned long long>(diag.nodes),
                                static_cast<unsigned long long>(sd.iterations),
                                lp_raw.termination_reason.c_str(),
                                static_cast<int>(lp_raw.proposed_status),
                                sd.primal_residual, sd.dual_residual, sd.gap_rel,
                                sd.dual_bound_finite ? "finite" : "none");
                engines::SimplexOptions fallback_opts = lp_opts;
                fallback_opts.method = engines::SimplexMethod::Primal;
                // A warm-started dual failure can be numerical rather than
                // structural. Retry the same bounded LP with a fresh primal
                // solve and the caller's presolve preference; this is still
                // accepted only after an independent certificate check.
                fallback_opts.presolve = opts.lp.presolve &&
                    lp_raw.proposed_status != core::Status::Infeasible &&
                    lp_raw.proposed_status != core::Status::InfeasibleOrUnbounded;
                bool fallback_has_time = true;
                if (opts.time_limit_s > 0.0) {
                    const double left = opts.time_limit_s -
                        std::chrono::duration<double>(Clock::now() - t0).count();
                    fallback_has_time = left > 0.0;
                    fallback_opts.time_limit_s = std::max(0.0, left);
                }
                if (fallback_has_time) {
                    engines::SimplexDiagnostics fallback_sd;
                    engines::SimplexBasis fallback_basis;
                    auto fallback_raw = engines::solve_simplex(
                        node_lp, fallback_opts, fallback_sd, &fallback_basis);
                    record_lp_attempt(fallback_sd);
                    ++diag.lp_fallbacks;
                    try_node_polish(fallback_raw, fallback_sd, fallback_opts);
                    if (!relaxation_proved(fallback_raw, fallback_sd,
                                           fallback_opts)) {
                        char pbuf[320];
                        std::snprintf(pbuf, sizeof pbuf,
                                      "\"node\":%llu,\"status\":%d,\"iterations\":%llu,"
                                      "\"primal_residual\":%.9g,\"dual_residual\":%.9g,"
                                      "\"gap_rel\":%.9g,\"gap_finite\":%s,"
                                      "\"dual_bound_finite\":%s,"
                                      "\"point_size\":%zu",
                                      static_cast<unsigned long long>(diag.nodes),
                                      static_cast<int>(fallback_raw.proposed_status),
                                      static_cast<unsigned long long>(fallback_sd.iterations),
                                      std::isfinite(fallback_sd.primal_residual) ? fallback_sd.primal_residual : 0.0,
                                      std::isfinite(fallback_sd.dual_residual) ? fallback_sd.dual_residual : 0.0,
                                      std::isfinite(fallback_sd.gap_rel) ? fallback_sd.gap_rel : 0.0,
                                      std::isfinite(fallback_sd.gap_rel) ? "true" : "false",
                                      fallback_sd.dual_bound_finite ? "true" : "false",
                                      fallback_raw.x.size());
                        SOR_ROUTE(1, "bab", "node_lp_fallback_unproved", pbuf);
                    }
                    if (relaxation_proved(fallback_raw, fallback_sd,
                                           fallback_opts) ||
                        node_lp_infeasibility_proved(node_lp, fallback_raw,
                                                     opts.primal_feas_tol,
                                                     root_relaxation_bounded)) {
                        lp_raw = std::move(fallback_raw);
                        sd = std::move(fallback_sd);
                        node_basis = std::move(fallback_basis);
                        session_state_is_last_node = false;
                    } else {
                        // Neither pass proved the node. Keep whichever left
                        // more usable evidence: a feasible point beats none,
                        // and otherwise the stronger certified (Lagrangian)
                        // bound. A fallback that only "failed to prove" used
                        // to be thrown away whole.
                        const auto has_point = [&](const core::RawResult& r) {
                            return static_cast<Index>(r.x.size()) == node_lp.n_cols() &&
                                   std::all_of(r.x.begin(), r.x.end(),
                                               [](f64 v) { return std::isfinite(v); }) &&
                                   node_lp.max_row_violation(r.x) <= opts.primal_feas_tol &&
                                   node_lp.max_bound_violation(r.x) <= opts.primal_feas_tol;
                        };
                        const auto safe_bound = [&](const core::RawResult& r) {
                            if (r.y.size() != sz(node_lp.n_rows())) return -std::numeric_limits<f64>::infinity();
                            std::vector<f64> ym(r.y.size());
                            for (std::size_t i = 0; i < ym.size(); ++i) ym[i] = sense * r.y[i];
                            const auto sb = certify::safe_lagrangian_lower_bound(
                                node_lp, ym, node_lp.col_lo, node_lp.col_hi);
                            return sb.finite && std::isfinite(sb.value)
                                       ? sb.value : -std::numeric_limits<f64>::infinity();
                        };
                        const bool orig_point = has_point(lp_raw);
                        const bool fb_point = has_point(fallback_raw);
                        bool take = false;
                        if (fb_point != orig_point) take = fb_point;
                        else take = safe_bound(fallback_raw) > safe_bound(lp_raw);
                        if (take) {
                            ++diag.lp_fallback_evidence_taken;
                            lp_raw = std::move(fallback_raw);
                            sd = std::move(fallback_sd);
                            node_basis = std::move(fallback_basis);
                            session_state_is_last_node = false;
                        }
                    }
                }
            }
        }
        }  // nlp_span
        if (lp_attempts > 0) {
            engines::install_simplex_work_totals(sd, lp_work);
            lp_raw.iterations = sd.iterations;
            diag.lp_solves += lp_attempts;
            diag.lp_iterations += sd.iterations;
            // S3.1/S3.2 measurement: these BabDiagnostics fields already
            // exist and are already printed ("node LP split:" / "node LP
            // non-loop:") but nothing summed the per-node SimplexDiagnostics
            // into them, so they always read 0 regardless of what actually
            // happened in the node loop.
            diag.node_lp_prep_ms += sd.preprocessing_ms;
            diag.node_lp_loop_ms += sd.loop_ms;
            diag.node_lp_simplex_ms += sd.total_ms;
            diag.node_lp_dse_rebuilds += sd.dse_weight_rebuilds;
            diag.node_lp_refactorizations += sd.refactorizations;
            diag.node_lp_first_factor_ms += sd.first_factor_ms;
            diag.node_lp_after_factor_ms += sd.after_first_factor_ms;
            diag.node_lp_dse_ms += sd.dse_rebuild_ms;
            diag.node_lp_post_ms += sd.post_solve_ms;
            diag.node_lp_dse_reuses += sd.dse_weight_reuses;
            diag.node_lp_factor_reuses += sd.factor_adoptions;
            diag.node_lp_factor_reuse_carrier_empty += sd.factor_reuse_carrier_empty;
            diag.node_lp_factor_reuse_matrix_null += sd.factor_reuse_matrix_null;
            diag.node_lp_factor_reuse_rows_mismatch += sd.factor_reuse_rows_mismatch;
            diag.node_lp_factor_reuse_preparation_mismatch += sd.factor_reuse_preparation_mismatch;
            diag.node_lp_factor_reuse_basis_mismatch += sd.factor_reuse_basis_mismatch;
            diag.node_lp_factor_reuse_skipped_refill_primal_cleanup +=
                sd.factor_reuse_skipped_refill_primal_cleanup;
        }
        const double node_lp_ms = ms_since(t_lp);
        diag.lp_ms += node_lp_ms;
        if (lp_attempts > 0) branching_lp_work.ms += node_lp_ms;
        seg(5);  // node LP incl. session, fallback, polish
        ++diag.loop_iters_past_lp;

        const bool node_infeasible_proved = node_lp_infeasibility_proved(
            node_lp, lp_raw, opts.primal_feas_tol,
            root_relaxation_bounded);
        if (!node_infeasible_proved &&
            node_lp_status_proves_infeasible(lp_raw.proposed_status,
                                              root_relaxation_bounded)) {
            SOR_ROUTE(1, "bab", "node_lp_unverified_infeasible",
                      "\"reason\":\"missing_or_invalid_ray\"");
        }
        if (node_infeasible_proved) {
            // Nogood trust: a plain Infeasible is the same certificate the
            // prune itself trusts. InfeasibleOrUnbounded is only safe to
            // turn into a GLOBAL cut when the certified root relaxation
            // rules out the unbounded arm (every descendant of a bounded
            // root box is bounded); without that, the node may merely be
            // unbounded and excluding its assignment globally could cut off
            // feasible points.
            const bool infeasible_trusted = node_infeasible_proved;
            observe_child_closed(BranchOutcome::CertifiedInfeasible);
            // LP-derived explanation first: the verified Farkas ray names the few
            // box bounds that made the node infeasible (relaxed as far as the
            // certificate's slack allows), a far shorter clause than "not all
            // branching decisions". Global rows only.
            const bool farkas_learned = try_learn_farkas(
                node_lp, lp_raw, node.active_local.empty(), false);
            if (infeasible_trusted && !farkas_learned) try_learn_nogood(node.prop_trail);
            if (route_sample_node(diag.nodes)) {
                char pbuf[80];
                std::snprintf(pbuf, sizeof pbuf, "\"node\":%llu",
                              static_cast<unsigned long long>(diag.nodes));
                SOR_ROUTE(3, "bab", "prune_lp_infeas", pbuf);
            }
            diag.ms_exit_prune += ms_since(t_loop_iter); ++diag.n_exit_prune;
            continue;  // prune
        }
        // Judge what this solve established, one fact at a time. An
        // interrupted/numerically unproved relaxation is not evidence that
        // the node is infeasible, and its objective is never a bound; but its
        // checked primal point can still drive branching and incumbents, and
        // its multipliers can still certify a Lagrangian bound. Only an LP
        // with no usable point forces an honest early stop.
        RelaxationOutcome rel;
        rel.infeasible_proved = node_infeasible_proved;
        rel.point_valid =
            static_cast<Index>(lp_raw.x.size()) == node_lp.n_cols() &&
            std::all_of(lp_raw.x.begin(), lp_raw.x.end(),
                        [](f64 v) { return std::isfinite(v); }) &&
            node_lp.max_row_violation(lp_raw.x) <= opts.primal_feas_tol &&
            node_lp.max_bound_violation(lp_raw.x) <= opts.primal_feas_tol;
        rel.lp_optimal = relaxation_proved(lp_raw, sd, lp_opts);
        rel.basis_usable =
            static_cast<Index>(node_basis.basic.size()) == node_lp.n_rows();
        if (rel.lp_optimal) rel.lp_bound = node_lp_bound_min(lp_raw, sense);
        const bool lp_point_feasible = rel.point_valid;
        // Certified bound from an UNPROVED relaxation. A dual simplex stopped
        // by a limit is dual feasible long before it is primal feasible, and
        // on degenerate models it often reaches the final objective thousands
        // of pivots before it can prove it (ex9: dual 81 against incumbent 81
        // while 0.35 of primal infeasibility was still being cleaned up).
        // Weak duality makes the Lagrangian of ANY multiplier vector a valid
        // lower bound over this node's box; safe_lagrangian_lower_bound
        // evaluates it rigorously. Such a node is pruned exactly like a proved
        // one when the certificate reaches the cutoff, and otherwise keeps the
        // larger of its inherited and certified bounds -- never an LP value.
        if (!rel.lp_optimal && lp_raw.y.size() == sz(node_lp.n_rows())) {
            std::vector<f64> y_min(lp_raw.y.size());
            for (std::size_t i = 0; i < y_min.size(); ++i)
                y_min[i] = sense * lp_raw.y[i];
            const auto t_safe = Clock::now();
            const auto safe = certify::safe_lagrangian_lower_bound(
                node_lp, y_min, node_lp.col_lo, node_lp.col_hi);
            diag.node_lp_safe_bound_ms += ms_since(t_safe);
            ++diag.lagrangian_bound_checks;
            if (safe.finite && std::isfinite(safe.value)) {
                rel.lagrangian_bound = safe.value;
                if (have_incumbent) {
                    const f64 inc_min = sense * best_incumbent;
                    if (safe.value >= node_cutoff(inc_min, obj_granularity,
                                                  offset_min, opts.gap_tol,
                                                  opts.abs_gap_tol)) {
                        ++diag.lagrangian_prunes;
                        const f64 certified = std::max(node.bound, safe.value);
                        if (certified < inc_min) {
                            ++diag.gap_prunes;
                            pruned_floor = std::min(pruned_floor, certified);
                        }
                        SOR_ROUTE(1, "bab", "prune_lagrangian_bound");
                        diag.ms_exit_b += ms_since(t_loop_iter); ++diag.n_exit_b;
                        continue;
                    }
                }
                if (safe.value > node.bound) ++diag.lagrangian_bound_raises;
            }
        }
        // The node keeps the stronger of its inherited bound and whatever
        // this solve certified. Before this, an unproved solve replaced the
        // bound with -inf further down, erasing both the parent's proof and a
        // Lagrangian raise, and every descendant inherited that -inf.
        if (!rel.lp_optimal && std::isfinite(node.bound)) ++diag.unproved_bounds_kept;
        node.bound = node_bound_after_relaxation(node.bound, rel);
        if (node.depth == 0 &&
            root_cert.raise(node.bound, rel.lp_optimal
                                            ? BoundCertificate::Source::LpOptimal
                                            : BoundCertificate::Source::Lagrangian))
            ++diag.root_bound_raises;
        // Give a node whose LP is unusable one more chance (cold primal route)
        // and keep searching the rest of the frontier. Only the clock ends the
        // search; a node that fails again is folded into the bound, not the
        // reason to stop.
        const auto defer_node = [&](const char* why) {
            if (node.lp_failures < opts.node_lp_max_retries) {
                ++node.lp_failures;
                node.defer_generation = global_lp_generation;
                node.has_basis = false;
                node.basis = engines::SimplexBasis{};
                node.checkpoint.reset();
                ++diag.node_lp_deferred;
                deferred.push_back(std::move(node));
            } else {
                abandoned_bound = std::min(abandoned_bound, node.bound);
                ++diag.abandoned_unproved_nodes;
                stopped_early = true;
                reason = why;
            }
            diag.ms_exit_prune += ms_since(t_loop_iter); ++diag.n_exit_prune;
        };
        if (lp_raw.proposed_status == core::Status::Interrupted &&
            (!lp_point_feasible || timed_out())) {
            if (!timed_out() && opts.node_lp_max_retries > 0) {
                SOR_ROUTE(1, "bab", "node_lp_defer", "\"reason\":\"interrupted\"");
                defer_node("node LP interrupted");
                continue;
            }
            reason = timed_out() ? "time limit" : "node LP interrupted";
            stopped_early = true;
            {
                char pbuf[128];
                std::snprintf(
                    pbuf, sizeof pbuf, "\"node\":%llu,\"reason\":\"%s\"",
                    static_cast<unsigned long long>(diag.nodes),
                    timed_out() ? "time_limit" : "interrupted");
                SOR_ROUTE(1, "bab", "node_lp_fail", pbuf);
            }
            // node.bound is the certified bound for this box (inherited or
            // raised above), a sound lower bound for the subtree being
            // walked away from.
            abandoned_bound = std::min(abandoned_bound, node.bound);
            ++diag.abandoned_unproved_nodes;
            break;
        }
        const bool node_lp_proved = rel.lp_optimal;
        if (node_lp_proved && node.depth == 0 && std::isfinite(lp_raw.objective))
            root_relaxation_bounded = true;
        if (!node_lp_proved) {
            if (!lp_point_feasible && !timed_out() && opts.node_lp_max_retries > 0) {
                SOR_ROUTE(1, "bab", "node_lp_defer", "\"reason\":\"unproved\"");
                defer_node("node LP unproved");
                continue;
            }
            if (!lp_point_feasible) {
                reason = "node LP unproved";
                stopped_early = true;
                {
                    const bool point_sized =
                        static_cast<Index>(lp_raw.x.size()) == node_lp.n_cols();
                    const double row_violation = point_sized
                        ? node_lp.max_row_violation(lp_raw.x) : core::kNaN;
                    const double bound_violation = point_sized
                        ? node_lp.max_bound_violation(lp_raw.x) : core::kNaN;
                    Index worst_row = -1;
                    long double worst_activity = 0.0L;
                    long double worst_rhs = 0.0L;
                    if (point_sized && std::isfinite(row_violation) &&
                        row_violation > 0.0) {
                        const auto& rp = node_lp.A.pattern.row_ptr();
                        const auto& ci = node_lp.A.pattern.col_idx();
                        long double largest = 0.0L;
                        for (Index r = 0; r < node_lp.n_rows(); ++r) {
                            long double activity = 0.0L;
                            for (core::Offset k = rp[sz(r)]; k < rp[sz(r) + 1]; ++k)
                                activity += static_cast<long double>(node_lp.A.vals[sz(k)]) *
                                            lp_raw.x[sz(ci[sz(k)])];
                            const long double lo = node_lp.row_lo[sz(r)];
                            const long double hi = node_lp.row_hi[sz(r)];
                            const long double below = std::max(0.0L, lo - activity);
                            const long double above = std::max(0.0L, activity - hi);
                            const long double violation = std::max(below, above);
                            if (violation > largest) {
                                largest = violation;
                                worst_row = r;
                                worst_activity = activity;
                                worst_rhs = below > above ? lo : hi;
                            }
                        }
                    }
                    Index continuous_singletons = 0;
                    Index free_singletons = 0;
                    if (worst_row >= 0) {
                        const auto& rp = node_lp.A.pattern.row_ptr();
                        const auto& ci = node_lp.A.pattern.col_idx();
                        std::vector<Index> incidence(sz(node_lp.n_cols()), 0);
                        for (Index j : ci) ++incidence[sz(j)];
                        for (core::Offset k = rp[sz(worst_row)];
                             k < rp[sz(worst_row) + 1]; ++k) {
                            const Index j = ci[sz(k)];
                            if (incidence[sz(j)] == 1 &&
                                (sz(j) >= node_lp.is_integer.size() ||
                                 !node_lp.is_integer[sz(j)])) {
                                ++continuous_singletons;
                                if (!std::isfinite(node_lp.col_lo[sz(j)]) &&
                                    !std::isfinite(node_lp.col_hi[sz(j)]))
                                    ++free_singletons;
                            }
                        }
                    }
                    char pbuf[600];
                    std::snprintf(pbuf, sizeof pbuf,
                                  "\"node\":%llu,\"reason\":\"unproved\","
                                  "\"status\":%d,\"iterations\":%llu,"
                                  "\"primal_residual\":%.9g,\"dual_residual\":%.9g,"
                                  "\"gap_rel\":%.9g,\"gap_finite\":%s,"
                                  "\"dual_bound_finite\":%s,"
                                  "\"point_size\":%zu,\"columns\":%d,"
                                  "\"row_violation\":%.9g,\"bound_violation\":%.9g,"
                                  "\"worst_row\":%d,\"worst_activity\":%.17g,"
                                  "\"worst_rhs\":%.17g,\"continuous_singletons\":%d,"
                                  "\"free_singletons\":%d",
                                  static_cast<unsigned long long>(diag.nodes),
                                  static_cast<int>(lp_raw.proposed_status),
                                  static_cast<unsigned long long>(sd.iterations),
                                  std::isfinite(sd.primal_residual) ? sd.primal_residual : 0.0,
                                  std::isfinite(sd.dual_residual) ? sd.dual_residual : 0.0,
                                  std::isfinite(sd.gap_rel) ? sd.gap_rel : 0.0,
                                  std::isfinite(sd.gap_rel) ? "true" : "false",
                                  sd.dual_bound_finite ? "true" : "false",
                                  lp_raw.x.size(), node_lp.n_cols(),
                                  std::isfinite(row_violation) ? row_violation : 0.0,
                                  std::isfinite(bound_violation) ? bound_violation : 0.0,
                                  worst_row, static_cast<double>(worst_activity),
                                  static_cast<double>(worst_rhs),
                                  continuous_singletons,
                                  free_singletons);
                    SOR_ROUTE(1, "bab", "node_lp_fail", pbuf);
                    if (opts.verbose)
                        std::printf("  [milp] node LP unproved: status=%d iterations=%llu "
                                    "primal=%.9g dual=%.9g gap=%.9g finite_bound=%d "
                                    "point=%zu/%d row=%.9g bound=%.9g "
                                    "worst_row=%d activity=%.17g rhs=%.17g "
                                    "continuous_singletons=%d free_singletons=%d "
                                    "row_name=%s reason=%s\n",
                                    static_cast<int>(lp_raw.proposed_status),
                                    static_cast<unsigned long long>(sd.iterations),
                                    sd.primal_residual, sd.dual_residual,
                                    sd.gap_rel, static_cast<int>(sd.dual_bound_finite),
                                    lp_raw.x.size(), node_lp.n_cols(),
                                    row_violation, bound_violation,
                                    worst_row, static_cast<double>(worst_activity),
                                    static_cast<double>(worst_rhs),
                                    continuous_singletons,
                                    free_singletons,
                                    worst_row >= 0 &&
                                            sz(worst_row) < node_lp.row_names.size()
                                        ? node_lp.row_names[sz(worst_row)].c_str()
                                        : "",
                                    lp_raw.termination_reason.c_str());
                }
                abandoned_bound = std::min(abandoned_bound, node.bound);
                ++diag.abandoned_unproved_nodes;
                break;
            }
        }

        if (!root_lp_outcome_logged && node.depth == 0 && node_lp_proved &&
            std::isfinite(lp_raw.objective)) {
            char rbuf[96];
            std::snprintf(rbuf, sizeof rbuf, "\"obj\":%.10g,\"integer\":false",
                          lp_raw.objective);
            SOR_ROUTE(1, "bab", "root_lp_outcome", rbuf);
            root_lp_outcome_logged = true;
        }

        seg(6);  // LP evidence: bounds, Lagrangian, fold, RC snapshots
        const auto t_post_lp = Clock::now();
        const f64 lp_obj = lp_raw.objective;          // original sense
        const f64 lp_obj_min = sense * lp_obj;        // minimize sense
        if (opts.reduced_cost_strengthening && node.depth == 0 && node_lp_proved &&
            node.active_local.empty() && lp_raw.y.size() == sz(node_lp.n_rows())) {
            root_rc = RootRcInfo{node_lp, lp_raw.x, lp_raw.y, lp_raw.objective};
            root_rc_inc_seen = std::numeric_limits<f64>::quiet_NaN();
        }
        // node.bound was settled above from certified evidence only; the
        // unproved LP objective (lp_obj) never enters it.
        if (opts.verbose && diag.nodes <= 12)
            std::printf("  [milp] node %llu lp_obj %.10e iters %llu depth %d frac_branch %d\n",
                        static_cast<unsigned long long>(diag.nodes), lp_obj,
                        static_cast<unsigned long long>(sd.iterations), node.depth,
                        static_cast<int>(pick_branch_var(
                            mip, node.col_lo, node.col_hi, lp_raw.x,
                            opts.int_tol)));

        // Ordinary child observation: only from a PROVED optimal relaxation
        // (an unproved objective is not a bound), and once per incoming branch.
        if (node.parent_branch_var >= 0 && node.parent_branch_var < n &&
            node_lp_proved && std::isfinite(node.parent_bound) &&
            std::isfinite(lp_obj_min)) {
            const f64 gain = std::max(0.0, lp_obj_min - node.parent_bound);
            const bool cut_off =
                cutoff_known() &&
                lp_obj_min >= node_cutoff(known_cutoff_min(), obj_granularity,
                                          offset_min, opts.gap_tol, opts.abs_gap_tol);
            if (bstats.record_child(node.parent_branch_var, node.parent_branch_dir,
                                    node.parent_branch_distance,
                                    cut_off ? BranchOutcome::CertifiedCutoff
                                            : BranchOutcome::ProvedOptimal,
                                    gain, child_inferences(), node.pc_consumed,
                                    node.pc_recorded_unit, opts.int_tol))
                ++diag.pseudocost_updates;
        }
        // LP snapshots for MRENS (multi-reference). Keep at most three.
        if (static_cast<Index>(lp_raw.x.size()) == n) {
            lp_snapshots.push_back(lp_raw.x);
            while (lp_snapshots.size() > 3) lp_snapshots.pop_front();
        }

        // Integer feasibility?
        bool integer_ok = true;
        for (Index j = 0; j < n; ++j) {
            if (!mip.is_integer.empty() && mip.is_integer[sz(j)] &&
                !is_integral(lp_raw.x[sz(j)], opts.int_tol)) {
                integer_ok = false;
                break;
            }
        }

        if (integer_ok) {
            f64 obj;
            if (accept_incumbent(lp_raw.x, true, obj)) {
                solution_pool.add(best_x, best_incumbent, problem.maximize);
                refresh_live_gap(std::isfinite(node.bound) ? node.bound : last_dual_bound);
                if (opts.verbose)
                    std::printf("  [milp] incumbent %.10e at node %llu\n",
                                best_incumbent,
                                static_cast<unsigned long long>(diag.nodes));
            }
            if (std::isfinite(obj)) {
                ++diag.integer_feasible;
                // An integral point closes the subtree only when it is the
                // subtree's LP OPTIMUM. From an unproved LP it is merely a
                // feasible point, and the rest of the box is kept by bound.
                if (!node_lp_proved)
                    fold_unsearched_region(
                        node, "integral point from an unproved node LP");
            } else {
                // Keep this region represented when its candidate fails the
                // original model check; dropping it could prove infeasibility.
                abandoned_bound = std::min(abandoned_bound, node.bound);
                stopped_early = true;
                reason = "integer candidate failed original-model validation";
            }
            diag.ms_exit_nobranch += ms_since(t_loop_iter); ++diag.n_exit_nobranch;
            continue;  // no branch
        }

        // Integral LP points must be validated before the approximate cutoff.
        if (have_incumbent && node_lp_proved) {
            const f64 inc_min = sense * best_incumbent;
            if (node.bound >= node_cutoff(inc_min, obj_granularity, offset_min,
                                           opts.gap_tol, opts.abs_gap_tol)) {
                if (node.bound < inc_min) {
                    ++diag.gap_prunes;
                    bool integral_point = static_cast<Index>(lp_raw.x.size()) == n;
                    for (Index j = 0; integral_point && j < n; ++j)
                        if (!mip.is_integer.empty() && mip.is_integer[sz(j)] &&
                            !is_integral(lp_raw.x[sz(j)], opts.int_tol))
                            integral_point = false;
                    diag.gap_pruned_integral_lp += integral_point;
                    pruned_floor = std::min(pruned_floor, node.bound);
                }
                if (route_sample_node(diag.nodes)) {
                    char pbuf[80];
                    std::snprintf(pbuf, sizeof pbuf, "\"node\":%llu",
                                  static_cast<unsigned long long>(diag.nodes));
                    SOR_ROUTE(3, "bab", "prune_bound", pbuf);
                }
                diag.ms_exit_b += ms_since(t_loop_iter); ++diag.n_exit_b;
                continue;
            }
        }

        diag.ms_node_post_lp += ms_since(t_post_lp);
        // Closed by the heuristics' own timers; everything between their end
        // and the branching step is timed as ms_node_pre_branch.
        // Rounding heuristic. Gated by the heuristic-layer ceiling like every
        // other heuristic: this block contains the rounding repair, the
        // feasibility pump and the structured searches, and on several
        // instances it was the single largest consumer of wall time.
        seg(7);  // post-LP bookkeeping -> heuristics
        // Objective feasibility pump while there is no incumbent: at the
        // root, then at nodes 100, 400, 1600, ... (a different seed each
        // time). Its time counts as heuristic time. It runs BEFORE the legacy rounding
        // constructions below, which can spend the whole no-incumbent heuristic
        // budget at the root (roll3000 and ns1830653 never reached it).
        if (opts.feasibility_pump_root && !have_incumbent && opts.sub_mip_depth == 0 &&
            diag.nodes >= fpump_next_node && opts.time_limit_s > 0.0 &&
            static_cast<Index>(lp_raw.x.size()) == n &&
            diag.fpump_ms <= opts.feasibility_pump_total_frac * opts.time_limit_s * 1000.0) {
            fpump_next_node = diag.nodes == 1 ? 100 : diag.nodes * 4;
            try_fpump(lp_raw.x);
        }
        // Set-partitioning / assignment repair from the LP point while there is
        // no incumbent (root, then nodes 50, 200, ...). Cheap when the model has
        // no unit-coefficient binary rows: the structure scan comes first.
        if (opts.spp_repair && !have_incumbent && opts.sub_mip_depth == 0 &&
            diag.nodes >= spp_next_node && opts.time_limit_s > 0.0 &&
            static_cast<Index>(lp_raw.x.size()) == n &&
            diag.spp_ms <= opts.spp_repair_total_frac * opts.time_limit_s * 1000.0) {
            spp_next_node = diag.nodes == 1 ? 50 : diag.nodes * 4;
            try_spp_repair(lp_raw.x);
        }
        const auto t_heur = Clock::now();
        std::optional<core::RouteSpan> heur_span;
        heur_span.emplace(2, "bab", "heur", "heur", "",
                          core::RouteLedgerBucket::Heur);
        if (opts.rounding_heuristic &&
            !heuristics_over_budget(opts.lp_rounding_repair_time_s * 1000.0)) {
            std::vector<f64> xh;
            bool rounded = false;
            const auto attempt_round = [&](const std::vector<f64>& point,
                                           std::vector<f64>& candidate,
                                           RoundMode mode) {
                const auto started = Clock::now();
                const bool ok = try_round(problem, point, opts.int_tol,
                                          opts.primal_feas_tol, candidate, mode);
                diag.rounding_ms += ms_since(started);
                ++diag.rounding_calls;
                return ok;
            };
            f64 rounded_obj = mip.maximize
                ? -std::numeric_limits<f64>::infinity()
                : std::numeric_limits<f64>::infinity();
            const auto consider_round = [&](RoundMode mode) {
                std::vector<f64> candidate;
                if (!attempt_round(lp_raw.x, candidate, mode))
                    return;
                const f64 obj = mip.objective(candidate);
                if (!std::isfinite(obj)) return;
                const bool better = !rounded ||
                    (mip.maximize ? obj > rounded_obj : obj < rounded_obj);
                if (better) {
                    rounded = true;
                    rounded_obj = obj;
                    xh = std::move(candidate);
                }
            };
            // Keep the objective-aware choice, but also test nearest and ceil
            // rounding. Lower-bound demand rows frequently need ceilings,
            // while signed-cost models can prefer the nearest point.
            if (diag.nodes >= next_direct_round_node) {
                ++diag.node_rounding_rounds;
                consider_round(RoundMode::Objective);
                consider_round(RoundMode::Nearest);
                consider_round(RoundMode::Ceil);
                if (diag.nodes == 1 && mip.n_cols() <= 1000 &&
                    mip.nnz() <= 10000) {
                    std::vector<f64> perturbed;
                    if (try_perturbed_rounding(problem, opts.lp, opts.int_tol,
                                               opts.primal_feas_tol, 12, 0.8,
                                               perturbed)) {
                        const f64 pobj = mip.objective(perturbed);
                        if (std::isfinite(pobj) &&
                            (!rounded || (mip.maximize ? pobj > rounded_obj
                                                            : pobj < rounded_obj))) {
                            rounded = true;
                            rounded_obj = pobj;
                            xh = std::move(perturbed);
                        }
                    }
                }
                if (diag.nodes == 1 && mip.n_cols() <= 3000 &&
                    mip.nnz() <= 20000) {
                    double enum_budget = 1.0;
                    if (opts.time_limit_s > 0.0)
                        enum_budget = std::min(enum_budget, std::max(0.0,
                            opts.time_limit_s -
                            std::chrono::duration<double>(Clock::now() - t0).count()));
                    std::vector<f64> enumerated;
                    if (enum_budget > 0.02 &&
                        try_fractional_enumeration(
                            problem, lp_raw.x, opts.int_tol,
                            opts.primal_feas_tol, 65536, enum_budget,
                            opts.lp_rounding_repair_max_iterations, 0.01,
                            enumerated)) {
                        const f64 eobj = problem.objective(enumerated);
                        if (std::isfinite(eobj) &&
                            (!rounded || (mip.maximize ? eobj > rounded_obj
                                                        : eobj < rounded_obj))) {
                            rounded = true;
                            rounded_obj = eobj;
                            xh = std::move(enumerated);
                        }
                    }
                }
                // LP points on highly symmetric 0/1 models can round into a
                // basin from which the local repair cannot escape. Retry from the
                // lower-bound corner; this is deterministic and often gives the
                // covering rows a much better starting point (p0201/markshare).
                if (!rounded) {
                    std::vector<f64> lower(mip.n_cols(), 0.0);
                    for (Index j = 0; j < mip.n_cols(); ++j) {
                        if (mip.col_lo[sz(j)] > -model::kInf)
                            lower[sz(j)] = mip.col_lo[sz(j)];
                    }
                    rounded = attempt_round(lower, xh, RoundMode::Objective);
                }
                bool improves = false;
                if (rounded) {
                    const f64 dobj = problem.objective(xh);
                    improves = std::isfinite(dobj) &&
                        (!have_incumbent ||
                         (mip.maximize ? dobj > best_incumbent
                                       : dobj < best_incumbent));
                }
                direct_round_interval = improves
                    ? 1
                    : std::min(kMaxDirectRoundInterval,
                               2 * direct_round_interval);
                next_direct_round_node = diag.nodes + direct_round_interval;
            }
            // Equality-heavy MILPs often need continuous columns to move after
            // integer rounding. Try a bounded LP repair at cold start and then
            // periodically while the tree has no incumbent.
            const bool repair_due = diag.nodes <= 8 || (diag.nodes % 256 == 0);
            if (!rounded && opts.lp_rounding_repair && repair_due) {
                ++diag.lp_repair_attempts;
                const auto t_rep = Clock::now();
                rounded = try_lp_rounding_repair(
                    problem, lp_raw.x, opts.int_tol,
                    opts.primal_feas_tol, opts.lp_rounding_repair_max_iterations,
                    opts.lp_rounding_repair_time_s, xh);
                diag.ms_lp_repair += ms_since(t_rep);
                if (rounded) ++diag.lp_repair_hits;
            }
            if (repair_due && mip.n_cols() <= 1000 &&
                mip.nnz() <= 10000) {
                ++diag.feasibility_pump_attempts;
                const auto t_pump = Clock::now();
                struct PumpTick { double& sink; Clock::time_point t;
                    ~PumpTick() { sink += ms_since(t); } }
                    pump_tick{diag.ms_feas_pump, t_pump};
                std::vector<f64> pumped;
                const bool pump_ok = try_feasibility_pump(
                    problem, lp_raw.x, opts.int_tol,
                    opts.primal_feas_tol, 12, 5000, 0.08, pumped);
                if (pump_ok) {
                    const f64 pobj = mip.objective(pumped);
                    if (std::isfinite(pobj) &&
                        (!rounded || (mip.maximize ? pobj > rounded_obj
                                                        : pobj < rounded_obj))) {
                        rounded = true;
                        rounded_obj = pobj;
                        xh = std::move(pumped);
                    }
                    ++diag.feasibility_pump_hits;
                }
            }
            if (diag.nodes == 1 && mip.n_cols() <= 3000 &&
                mip.nnz() <= 15000) {
                std::vector<f64> constructed;
                const int construct_restarts = mip.n_cols() > 1000 ? 48 : 96;
                const int construct_trials = mip.n_cols() > 1000 ? 24 : 48;
                if (try_randomized_construct(problem, opts.int_tol,
                                             opts.primal_feas_tol,
                                             construct_restarts, construct_trials,
                                             mip.n_cols() > 1000 ? 1.0 : 0.8,
                                             opts.lp_rounding_repair_max_iterations,
                                             opts.lp_rounding_repair_time_s,
                                             constructed)) {
                    const f64 cobj = mip.objective(constructed);
                    if (std::isfinite(cobj) &&
                        (!rounded || (mip.maximize ? cobj > rounded_obj
                                                        : cobj < rounded_obj))) {
                        rounded = true;
                        rounded_obj = cobj;
                        xh = std::move(constructed);
                    }
                }
                if (diag.nodes == 1 && problem.n_cols() <= 500 &&
                    problem.nnz() <= 5000) {
                    const double construct_budget = opts.time_limit_s > 0.0
                        ? std::min(0.8, std::max(0.0, opts.time_limit_s -
                            std::chrono::duration<double>(Clock::now() - t0).count()))
                        : 0.8;
                    std::vector<f64> covering;
                    if (construct_budget > 0.02 &&
                        try_covering_construct(problem, opts.int_tol,
                                               opts.primal_feas_tol, 32,
                                               construct_budget, covering)) {
                        const f64 cobj = problem.objective(covering);
                        if (opts.verbose)
                            std::printf("  [milp] covering heuristic %.10e\n", cobj);
                        if (std::isfinite(cobj) &&
                            (!rounded || (mip.maximize ? cobj > rounded_obj
                                                        : cobj < rounded_obj))) {
                            rounded = true;
                            rounded_obj = cobj;
                            xh = std::move(covering);
                        }
                    }
                }
                if (diag.nodes == 1 && problem.n_cols() <= 200 &&
                    problem.nnz() <= 5000) {
                    const double discrepancy_budget = opts.time_limit_s > 0.0
                        ? std::min(5.0, std::max(0.0, opts.time_limit_s -
                            std::chrono::duration<double>(Clock::now() - t0).count()))
                        : 1.8;
                    std::vector<f64> discrepancy;
                    if (discrepancy_budget > 0.02 &&
                        try_discrepancy_binary_search(
                            problem, opts.int_tol, opts.primal_feas_tol,
                            64, 100000, discrepancy_budget, discrepancy)) {
                        const f64 dobj = problem.objective(discrepancy);
                        if (opts.verbose)
                            std::printf("  [milp] discrepancy heuristic %.10e\n", dobj);
                        if (std::isfinite(dobj) &&
                            (!rounded || (mip.maximize ? dobj > rounded_obj
                                                        : dobj < rounded_obj))) {
                            rounded = true;
                            rounded_obj = dobj;
                            xh = std::move(discrepancy);
                        }
                    }
                }
                // Equality-heavy binary models (notably markshare and pk1)
                // need coordinated exchanges rather than independent flips.
                // Search their residual objective directly, then use the LP
                // repair only for the best few assignments.
                if (problem.n_rows() <= 80 && problem.n_cols() <= 200 &&
                    problem.nnz() <= 10000) {
                    // Small low-row binary models benefit disproportionately
                    // from spending time in the direct combinatorial search:
                    // its beam/DFS work is much cheaper than thousands of
                    // cold LP repairs, and it does not weaken proof logic.
                    double structured_budget =
                        (problem.n_cols() <= 100 && problem.n_rows() <= 10)
                            ? 6.0 : 1.5;
                    if (opts.time_limit_s > 0.0)
                        structured_budget = std::min(structured_budget,
                            std::max(0.0, opts.time_limit_s -
                                std::chrono::duration<double>(Clock::now() - t0).count()));
                    std::vector<f64> structured;
                    if (structured_budget > 0.02 &&
                        try_structured_binary_search(
                            problem, opts.int_tol, opts.primal_feas_tol,
                            4, 80, 32, structured_budget,
                            opts.lp_rounding_repair_max_iterations,
                            opts.lp_rounding_repair_time_s, structured)) {
                        const f64 sobj = problem.objective(structured);
                        if (std::isfinite(sobj) &&
                            (!rounded || (mip.maximize ? sobj > rounded_obj
                                                        : sobj < rounded_obj))) {
                            rounded = true;
                            rounded_obj = sobj;
                            xh = std::move(structured);
                        }
                    }
                }
                std::vector<f64> binary_search;
                if (problem.n_rows() <= 10 && try_binary_penalty_search(
                        problem, opts.int_tol, opts.primal_feas_tol,
                        24, 12000, 32, 1.2,
                        opts.lp_rounding_repair_max_iterations,
                        opts.lp_rounding_repair_time_s, binary_search)) {
                    const f64 bobj = problem.objective(binary_search);
                    if (std::isfinite(bobj) &&
                        (!rounded || (problem.maximize ? bobj > rounded_obj
                                                        : bobj < rounded_obj))) {
                        rounded = true;
                        rounded_obj = bobj;
                        xh = std::move(binary_search);
                    }
                }
            }
            // Independent rounding can select far too many structural
            // binaries. Explore a bounded Hamming neighborhood and re-solve
            // the continuous subproblem after each move (RINS/local search).
            // This is only an incumbent heuristic and is never used in a
            // node bound or pruning decision.
            if (rounded && opts.integer_neighborhood && diag.nodes == 1 &&
                mip.n_cols() <= 3000 && mip.nnz() <= 20000) {
                // This first root-neighborhood call used to take its full
                // three-second default even after the global deadline had
                // expired. Later calls already cap against seconds_left().
                const double left = std::min(
                    opts.integer_neighborhood_time_s,
                    std::max(0.0, seconds_left()));
                if (left > 0.0) {
                    ++diag.integer_neighborhood_attempts;
                    std::vector<f64> polished;
                    const bool improved = try_integer_neighborhood(
                        problem, xh, opts.int_tol, opts.primal_feas_tol,
                        opts.integer_neighborhood_max_trials, left,
                        opts.lp_rounding_repair_max_iterations,
                        opts.integer_neighborhood_lp_time_s, polished);
                    if (improved) {
                        xh = std::move(polished);
                        ++diag.integer_neighborhood_hits;
                    }
                    // The helper's trial count is bounded by the configured
                    // cap; expose that cap as a conservative diagnostic.
                    diag.integer_neighborhood_trials +=
                        opts.integer_neighborhood_max_trials;
                }
            }
                if (rounded) {
                    f64 hobj = problem.objective(xh);
                if (std::isfinite(hobj)) {
                    const bool better =
                        !have_incumbent ||
                        (mip.maximize ? (hobj > best_incumbent)
                                          : (hobj < best_incumbent));
                    if (better) {
                        // Run one bounded local-improvement pass as soon as
                        // the first incumbent appears.  Waiting for the root
                        // misses incumbents discovered later in the tree (for
                        // example gt2/assignment models).
                        if (opts.integer_neighborhood &&
                            diag.integer_neighborhood_attempts == 0 &&
                            mip.n_cols() <= 3000 && mip.nnz() <= 20000) {
                            ++diag.integer_neighborhood_attempts;
                            std::vector<f64> polished;
                            const bool all_integer_model =
                                static_cast<Index>(std::count(
                                    problem.is_integer.begin(),
                                    problem.is_integer.end(), true)) ==
                                problem.n_cols();
                            double left = all_integer_model
                                ? std::max(2.5, opts.integer_neighborhood_time_s)
                                : opts.integer_neighborhood_time_s;
                            if (opts.time_limit_s > 0.0) {
                                left = std::min(left, std::max(0.0,
                                    opts.time_limit_s -
                                    std::chrono::duration<double>(Clock::now() - t0).count()));
                            }
                            if (left > 0.0 && try_integer_neighborhood(
                                    problem, xh, opts.int_tol,
                                    opts.primal_feas_tol,
                                    all_integer_model
                                        ? std::max<std::uint64_t>(
                                            100000, opts.integer_neighborhood_max_trials)
                                        : opts.integer_neighborhood_max_trials,
                                    left,
                                    opts.lp_rounding_repair_max_iterations,
                                    opts.integer_neighborhood_lp_time_s,
                                    polished)) {
                                xh = std::move(polished);
                                ++diag.integer_neighborhood_hits;
                                hobj = problem.objective(xh);
                            }
                            diag.integer_neighborhood_trials +=
                                all_integer_model
                                    ? std::max<std::uint64_t>(
                                        100000, opts.integer_neighborhood_max_trials)
                                    : opts.integer_neighborhood_max_trials;
                        }
                        if (accept_incumbent(std::move(xh), false, hobj)) {
                            ++diag.heuristic_hits;
                            if (opts.verbose)
                                std::printf("  [milp] heuristic incumbent %.10e at node %llu\n",
                                            best_incumbent,
                                            static_cast<unsigned long long>(diag.nodes));
                        }
                    }
                }
            }
        }
        // Closed here, before the Feasibility Jump calls below: those keep their
        // own accumulator (diag.feasjump_ms), and the ceiling sums the two, so
        // the windows must not overlap or FJ time is charged twice and the
        // ceiling fires early.
        heur_span.reset();
        diag.heuristic_ms += ms_since(t_heur);
        // Time from here to the branching step that no heuristic timer
        // claims (ms_node_pre_branch), measured as elapsed minus the growth
        // of the heuristic totals.
        const auto t_pre_branch_start = Clock::now();
        const double heur_before_pre_branch = heuristic_spent_ms(diag);

        seg(8);  // in-tree rounding heuristics
        // Feasibility Jump, warm. The cold pre-tree run had no LP to look at;
        // this one starts from the root relaxation, which is a far better
        // starting assignment when the relaxation exists at all.
        if (opts.feasibility_jump && diag.nodes == 1)
            try_feasjump(&lp_raw.x,
                         std::min(opts.feasibility_jump_seeded_time_s,
                                  std::max(0.0, seconds_left())),
                         core::kPosInf);


        // Feasibility Jump, improving. With an incumbent in hand the objective
        // becomes a soft row just below it, so the same walk that hunts a
        // feasible point now hunts a better one. Bounded by interval and by
        // its own time budget so it cannot crowd out the tree.
        if (opts.feasibility_jump && have_incumbent &&
            opts.feasibility_jump_improve_interval > 0 &&
            diag.nodes % opts.feasibility_jump_improve_interval == 0) {
            const f64 span = 1e-4 * (1.0 + std::fabs(best_incumbent));
            const f64 cutoff = mip.maximize ? best_incumbent + span
                                            : best_incumbent - span;
            try_feasjump(&lp_raw.x,
                         std::min(opts.feasibility_jump_improve_time_s,
                                  std::max(0.0, seconds_left())),
                         cutoff);
        }
        // Adaptive LNS / Balans. Classical keeps AlnsScheduler over Neighborhood
        // only. Latest uses Balans as the primary primal controller (Neighborhood
        // + Kernel Pump / MRENS / FeasJump / BTBS-LNS / CL-TLNS meta-arms).
        {
            const std::uint64_t interval =
                use_balans ? opts.balans.min_interval : opts.lns.min_interval;
            // Due by any of three clocks: the node interval; a wall-clock
            // interval, because on slow-node models 150 nodes may never
            // arrive (nursesched-sprint02 and piperout-27 reached 123 and 112
            // nodes in a 60 s run and never got an attempt); and ONE root
            // opportunity after the first usable LP, whatever the interval.
            // The old blanket suppression in proof mode (small gap) is gone:
            // heuristics_over_budget() already caps the share of time such a
            // run may spend on heuristics, and a small gap with a poor
            // incumbent is exactly when a neighborhood search pays.
            const double lns_time_interval_s =
                opts.time_limit_s > 0.0 ? std::max(2.0, 0.05 * opts.time_limit_s)
                                        : 0.0;
            const bool root_opportunity = !lns_root_done;
            const bool interval_due =
                diag.nodes >= last_lns_node + interval ||
                (lns_time_interval_s > 0.0 &&
                 ms_since(last_lns_time) / 1000.0 >= lns_time_interval_s);
            const bool lns_on = opts.sub_mip_lns && opts.sub_mip_depth == 0 &&
                                (use_balans ? opts.balans.enabled
                                            : opts.lns.enabled) &&
                                !heuristics_over_budget() &&
                                (root_opportunity || interval_due);
            if (lns_on) {
                last_lns_node = diag.nodes;
                last_lns_time = Clock::now();
                const bool at_root_opportunity = root_opportunity;
                lns_root_done = true;
                sub_mip_strengthen = at_root_opportunity;
                if (at_root_opportunity) ++diag.lns_root_attempts;
                const bool has_inc = have_incumbent && best_x.size() == sz(n);

                // Without an incumbent nothing can be pruned, so the first
                // feasible point is worth far more than an improvement: give a
                // sub-MIP room to presolve, solve an LP no costlier than the
                // root's, and search -- bounded by a fifth of the time left.
                const auto first_solution_budget = [&](double base, bool have) {
                    if (have) return base;
                    return std::min(std::max(base, 4.0 * root_lp_wall_s + 0.5),
                                    std::max(0.0, 0.2 * seconds_left()));
                };
                auto run_neighborhood_arm =
                    [&](Neighborhood kind, f64 fixing, std::uint64_t nodes_budget,
                        double budget, LnsOutcome& outcome, double& spent) {
                        NeighborhoodProblem np;
                        const std::vector<f64>& inc_ref =
                            has_inc ? best_x : lp_raw.x;
                        // RENS orders its fixings by LP confidence: the
                        // reduced cost of each column at this node's LP.
                        std::vector<f64> rens_rc;
                        if (kind == Neighborhood::Rens &&
                            lp_raw.y.size() == sz(node_lp.n_rows()) &&
                            node_lp.n_cols() == n) {
                            rens_rc.assign(node_lp.c.begin(), node_lp.c.end());
                            const auto& rp = node_lp.A.pattern.row_ptr();
                            const auto& ci = node_lp.A.pattern.col_idx();
                            for (Index i = 0; i < node_lp.n_rows(); ++i)
                                for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                                    rens_rc[sz(ci[sz(k)])] -=
                                        lp_raw.y[sz(i)] * node_lp.A.vals[sz(k)];
                        }
                        const auto t_build = Clock::now();
                        const bool built = build_neighborhood(
                            kind, mip, root_lo, root_hi, lp_raw.x,
                            has_inc ? inc_ref : std::vector<f64>{},
                            solution_pool, fixing, opts.int_tol, lns_rng, np,
                            rens_rc.empty() ? nullptr : &rens_rc);
                        if (!built) {
                            diag.sub_mip_build_ms += ms_since(t_build);
                            outcome = LnsOutcome::NotBuilt;
                            return;
                        }
                        // The parent only wants a strictly better point, so
                        // say so: the child then prunes against the parent's
                        // incumbent from its first LP instead of finding it
                        // again. Crossover keeps feasible-but-not-better
                        // points, which feed the solution pool; Proximity
                        // already carries its own cutoff row.
                        if (has_inc && !np.has_objective_cutoff &&
                            kind != Neighborhood::Crossover) {
                            const f64 c_min = node_cutoff(
                                sense * best_incumbent, obj_granularity,
                                offset_min, opts.gap_tol, opts.abs_gap_tol);
                            if (std::isfinite(c_min)) {
                                np.has_objective_cutoff = true;
                                np.objective_cutoff = sense * c_min;
                                ++diag.lns_cutoff_rows;
                            }
                        }
                        const auto t_lns = Clock::now();
                        const model::LpProblem sub = apply_neighborhood(mip, np);
                        diag.sub_mip_build_ms += ms_since(t_build);
                        std::vector<f64> xs;
                        bool exhausted = false;
                        const bool got =
                            solve_sub_problem(sub, budget, nodes_budget, xs,
                                              exhausted);
                        spent = ms_since(t_lns) / 1000.0;
                        diag.sub_mip_ms += ms_since(t_lns);
                        if (got) {
                            outcome = accept_sub_point(xs) ? LnsOutcome::NewBest
                                                           : LnsOutcome::Feasible;
                        } else {
                            outcome = exhausted ? LnsOutcome::Exhausted
                                                : LnsOutcome::Nothing;
                        }
                    };

                if (!use_balans) {
                    std::vector<bool> usable(
                        static_cast<std::size_t>(alns.n_arms()), false);
                    for (int a = 0; a < alns.n_arms(); ++a) {
                        switch (alns.kind(a)) {
                            case Neighborhood::Rens:
                                usable[sz(a)] = true;
                                break;
                            case Neighborhood::Crossover:
                                usable[sz(a)] = solution_pool.size() >= 2;
                                break;
                            default: usable[sz(a)] = has_inc; break;
                        }
                        // Root opportunity: RENS without an incumbent, RINS
                        // with one -- the two that use the fresh LP point.
                        if (at_root_opportunity)
                            usable[sz(a)] = usable[sz(a)] &&
                                alns.kind(a) == (has_inc ? Neighborhood::Rins
                                                         : Neighborhood::Rens);
                    }
                    double budget = std::min(opts.lns.call_time_s,
                                             std::max(0.0, seconds_left() * 0.5));
                    budget = first_solution_budget(budget, has_inc);
                    const bool blocked =
                        heuristics_over_budget(budget * 1000.0) ||
                        (opts.time_limit_s > 0.0 &&
                         diag.sub_mip_ms >
                             opts.lns.budget_frac * opts.time_limit_s * 1000.0);
                    if (blocked) ++diag.lns.budget_blocks;
                    const int arm = blocked ? -1 : alns.select(usable);
                    if (arm >= 0 && budget > 0.05) {
                        ++diag.lns.attempts;
                        LnsOutcome outcome = LnsOutcome::NotBuilt;
                        double spent = 0.0;
                        run_neighborhood_arm(alns.kind(arm),
                                             alns.fixing_rate(arm),
                                             alns.node_budget(arm), budget,
                                             outcome, spent);
                        if (outcome != LnsOutcome::NotBuilt) ++diag.lns.built;
                        if (outcome == LnsOutcome::NewBest) {
                            ++diag.lns.hits;
                            if (opts.verbose)
                                std::printf("  [milp] LNS %s incumbent %.10e\n",
                                            to_string(alns.kind(arm)),
                                            best_incumbent);
                        }
                        alns.reward(arm, outcome, spent);
                    }
                } else {
                    std::vector<bool> usable(
                        static_cast<std::size_t>(balans.n_arms()), false);
                    for (int a = 0; a < balans.n_arms(); ++a) {
                        const BalansArm k = balans.kind(a);
                        switch (k) {
                            case BalansArm::Rens: usable[sz(a)] = true; break;
                            case BalansArm::Crossover:
                                usable[sz(a)] = solution_pool.size() >= 2;
                                break;
                            case BalansArm::KernelPump:
                                usable[sz(a)] = opts.kernel_pump.enabled;
                                break;
                            case BalansArm::Mrens:
                                usable[sz(a)] = opts.mrens.enabled;
                                break;
                            case BalansArm::FeasJump:
                                usable[sz(a)] = opts.feasibility_jump;
                                break;
                            case BalansArm::BtbsLns:
                                usable[sz(a)] = opts.btbs.enabled;
                                break;
                            case BalansArm::ClTlns:
                                usable[sz(a)] =
                                    opts.cl_tlns.enabled && has_inc;
                                break;
                            default: usable[sz(a)] = has_inc; break;
                        }
                        if (at_root_opportunity)
                            usable[sz(a)] = usable[sz(a)] &&
                                k == (has_inc ? BalansArm::Rins : BalansArm::Rens);
                    }
                    double budget =
                        std::min(opts.balans.call_time_s,
                                 std::max(0.0, seconds_left() * 0.5));
                    budget = first_solution_budget(budget, has_inc);
                    const bool blocked =
                        heuristics_over_budget(budget * 1000.0) ||
                        (opts.time_limit_s > 0.0 &&
                         diag.sub_mip_ms >
                             opts.balans.budget_frac * opts.time_limit_s *
                                 1000.0);
                    if (blocked) ++diag.balans.budget_blocks;
                    const int arm = blocked ? -1 : balans.select(usable);
                    if (arm >= 0 && budget > 0.05) {
                        ++diag.balans.attempts;
                        LnsOutcome outcome = LnsOutcome::NotBuilt;
                        double spent = 0.0;
                        const BalansArm kind = balans.kind(arm);
                        Neighborhood nk;
                        if (balans_to_neighborhood(kind, nk)) {
                            run_neighborhood_arm(nk, balans.fixing_rate(arm),
                                                 balans.node_budget(arm), budget,
                                                 outcome, spent);
                            if (outcome != LnsOutcome::NotBuilt)
                                ++diag.balans.built;
                            if (outcome == LnsOutcome::NewBest) {
                                ++diag.balans.hits;
                                if (opts.verbose)
                                    std::printf(
                                        "  [milp] Balans %s incumbent %.10e\n",
                                        to_string(kind), best_incumbent);
                            }
                        } else if (kind == BalansArm::KernelPump) {
                            const auto t_kp = Clock::now();
                            KernelPumpOptions ko = opts.kernel_pump;
                            ko.time_limit_s = std::min(ko.time_limit_s, budget);
                            ko.int_tol = opts.int_tol;
                            ko.feas_tol = opts.primal_feas_tol;
                            std::vector<f64> rc;
                            const std::vector<f64>* rc_ptr = nullptr;
                            if (static_cast<Index>(lp_raw.y.size()) ==
                                mip.n_rows()) {
                                // Minimization reduced costs c - A'y.
                                rc.assign(sz(n), 0.0);
                                for (Index j = 0; j < n; ++j)
                                    rc[sz(j)] = mip.c[sz(j)];
                                const auto& rp = mip.A.pattern.row_ptr();
                                const auto& ci = mip.A.pattern.col_idx();
                                for (Index i = 0; i < mip.n_rows(); ++i) {
                                    const f64 yi = lp_raw.y[sz(i)];
                                    for (core::Offset k = rp[sz(i)];
                                         k < rp[sz(i) + 1]; ++k)
                                        rc[sz(ci[sz(k)])] -=
                                            yi * mip.A.vals[sz(k)];
                                }
                                if (mip.maximize)
                                    for (f64& v : rc) v = -v;
                                rc_ptr = &rc;
                            }
                            std::vector<f64> xk;
                            KernelPumpDiagnostics kd;
                            const bool got = kernel_pump(mip, lp_raw.x, rc_ptr,
                                                         ko, xk, kd);
                            spent = ms_since(t_kp) / 1000.0;
                            diag.heuristic_ms += ms_since(t_kp);
                            diag.kernel_pump.pumps += kd.pumps;
                            diag.kernel_pump.lp_solves += kd.lp_solves;
                            diag.kernel_pump.lp_iterations += kd.lp_iterations;
                            diag.kernel_pump.ms += kd.ms;
                            diag.kernel_pump.kernel_size =
                                std::max(diag.kernel_pump.kernel_size,
                                         kd.kernel_size);
                            diag.kernel_pump.buckets =
                                std::max(diag.kernel_pump.buckets, kd.buckets);
                            ++diag.balans.built;
                            if (got && accept_sub_point(xk)) {
                                outcome = LnsOutcome::NewBest;
                                ++diag.balans.hits;
                                diag.kernel_pump.found = true;
                                if (opts.verbose)
                                    std::printf(
                                        "  [milp] KernelPump incumbent %.10e\n",
                                        best_incumbent);
                            } else if (got) {
                                outcome = LnsOutcome::Feasible;
                            } else {
                                outcome = LnsOutcome::Nothing;
                            }
                        } else if (kind == BalansArm::Mrens) {
                            const auto t_m = Clock::now();
                            ++diag.mrens.attempts;
                            std::vector<std::vector<f64>> refs;
                            for (const auto& s : lp_snapshots) refs.push_back(s);
                            if (refs.empty()) {
                                synthesize_mrens_refs(mip, lp_raw.x,
                                                      opts.mrens.max_refs,
                                                      opts.int_tol, lns_rng,
                                                      refs);
                            } else if (static_cast<int>(refs.size()) <
                                       opts.mrens.max_refs) {
                                std::vector<std::vector<f64>> extra;
                                synthesize_mrens_refs(mip, lp_raw.x,
                                                      opts.mrens.max_refs,
                                                      opts.int_tol, lns_rng,
                                                      extra);
                                for (auto& e : extra) {
                                    if (static_cast<int>(refs.size()) >=
                                        opts.mrens.max_refs)
                                        break;
                                    refs.push_back(std::move(e));
                                }
                            }
                            MrensNeighborhood nb;
                            MrensOptions mo = opts.mrens;
                            mo.int_tol = opts.int_tol;
                            const auto t_build = Clock::now();
                            const bool built = build_mrens_neighborhood(
                                mip, refs, root_lo, root_hi, mo, nb);
                            diag.mrens.refs_used = refs.size();
                            if (!built) diag.sub_mip_build_ms += ms_since(t_build);
                            if (built) {
                                ++diag.mrens.built;
                                ++diag.balans.built;
                                diag.mrens.fixed = nb.fixed;
                                diag.mrens.free_integer = nb.free_integer;
                                const model::LpProblem sub = apply_mrens(mip, nb);
                                diag.sub_mip_build_ms += ms_since(t_build);
                                std::vector<f64> xs;
                                bool exhausted = false;
                                const bool got = solve_sub_problem(
                                    sub, std::min(budget, mo.time_limit_s),
                                    std::min(balans.node_budget(arm),
                                             mo.max_nodes),
                                    xs, exhausted);
                                spent = ms_since(t_m) / 1000.0;
                                diag.sub_mip_ms += ms_since(t_m);
                                diag.mrens.seconds += spent;
                                if (got) {
                                    outcome = accept_sub_point(xs)
                                                  ? LnsOutcome::NewBest
                                                  : LnsOutcome::Feasible;
                                    if (outcome == LnsOutcome::NewBest) {
                                        ++diag.balans.hits;
                                        ++diag.mrens.hits;
                                        if (opts.verbose)
                                            std::printf(
                                                "  [milp] MRENS incumbent %.10e\n",
                                                best_incumbent);
                                    }
                                } else {
                                    outcome = exhausted ? LnsOutcome::Exhausted
                                                        : LnsOutcome::Nothing;
                                }
                            } else {
                                spent = ms_since(t_m) / 1000.0;
                                outcome = LnsOutcome::NotBuilt;
                            }
                        } else if (kind == BalansArm::FeasJump) {
                            const auto t_fj = Clock::now();
                            const bool hit = try_feasjump(
                                &lp_raw.x, std::min(budget, 0.4),
                                core::kPosInf);
                            spent = ms_since(t_fj) / 1000.0;
                            ++diag.balans.built;
                            outcome = hit ? LnsOutcome::NewBest
                                          : LnsOutcome::Nothing;
                            if (hit) ++diag.balans.hits;
                        } else if (kind == BalansArm::BtbsLns) {
                            const auto t_bt = Clock::now();
                            ++diag.btbs.attempts;
                            NeighborhoodProblem np;
                            std::vector<f64> importance(sz(n), 0.0);
                            for (Index j = 0; j < n; ++j) {
                                if (mip.is_integer.empty() ||
                                    !mip.is_integer[sz(j)])
                                    continue;
                                f64 sc = frac_score(lp_raw.x[sz(j)]);
                                if (pc_down_count[sz(j)] > 0)
                                    sc += 1e-3 * (pc_down_sum[sz(j)] /
                                                  pc_down_count[sz(j)]);
                                if (pc_up_count[sz(j)] > 0)
                                    sc += 1e-3 * (pc_up_sum[sz(j)] /
                                                  pc_up_count[sz(j)]);
                                if (!mip.c.empty())
                                    sc += 1e-6 * std::fabs(mip.c[sz(j)]);
                                if (has_inc)
                                    sc += std::fabs(best_x[sz(j)] -
                                                    lp_raw.x[sz(j)]);
                                importance[sz(j)] = sc;
                            }
                            BtbsOptions bo = opts.btbs;
                            bo.int_tol = opts.int_tol;
                            // Adaptive beam: higher fixing_rate → smaller beam.
                            bo.destroy_frac = std::min(
                                0.9, std::max(0.05, 1.0 - balans.fixing_rate(arm)));
                            const auto t_build = Clock::now();
                            const bool built = build_btbs_neighborhood(
                                mip, root_lo, root_hi,
                                has_inc ? best_x : lp_raw.x, lp_raw.x,
                                importance, bo, np);
                            if (!built) diag.sub_mip_build_ms += ms_since(t_build);
                            if (built) {
                                ++diag.btbs.built;
                                ++diag.balans.built;
                                diag.btbs.fixed = np.fixed;
                                diag.btbs.free_integer = np.free_integer;
                                const model::LpProblem sub =
                                    apply_neighborhood(mip, np);
                                diag.sub_mip_build_ms += ms_since(t_build);
                                std::vector<f64> xs;
                                bool exhausted = false;
                                const bool got = solve_sub_problem(
                                    sub,
                                    std::min(budget, bo.time_limit_s),
                                    std::min(balans.node_budget(arm),
                                             bo.max_nodes),
                                    xs, exhausted);
                                spent = ms_since(t_bt) / 1000.0;
                                diag.sub_mip_ms += ms_since(t_bt);
                                diag.btbs.seconds += spent;
                                if (got) {
                                    outcome = accept_sub_point(xs)
                                                  ? LnsOutcome::NewBest
                                                  : LnsOutcome::Feasible;
                                    if (outcome == LnsOutcome::NewBest) {
                                        ++diag.balans.hits;
                                        ++diag.btbs.hits;
                                        if (opts.verbose)
                                            std::printf(
                                                "  [milp] BTBS-LNS incumbent "
                                                "%.10e\n",
                                                best_incumbent);
                                    }
                                } else {
                                    outcome = exhausted ? LnsOutcome::Exhausted
                                                        : LnsOutcome::Nothing;
                                }
                            } else {
                                spent = ms_since(t_bt) / 1000.0;
                                outcome = LnsOutcome::NotBuilt;
                            }
                        } else if (kind == BalansArm::ClTlns) {
                            const auto t_cl = Clock::now();
                            ++diag.cl_tlns.attempts;
                            NeighborhoodProblem np;
                            ClTlnsOptions co = opts.cl_tlns;
                            co.int_tol = opts.int_tol;
                            co.disagree_tol = opts.int_tol;
                            co.destroy_frac = std::min(
                                0.9, std::max(0.05, 1.0 - balans.fixing_rate(arm)));
                            std::vector<f64> ref2;
                            if (!lp_snapshots.empty())
                                ref2 = lp_snapshots.front();
                            else if (solution_pool.size() >= 1)
                                ref2 = solution_pool.at(0);
                            const std::vector<f64>& inc =
                                has_inc ? best_x : lp_raw.x;
                            const auto t_build = Clock::now();
                            const bool built = build_cl_tlns_neighborhood(
                                mip, root_lo, root_hi, inc, lp_raw.x, ref2, co,
                                np);
                            if (!built) diag.sub_mip_build_ms += ms_since(t_build);
                            if (built) {
                                ++diag.cl_tlns.built;
                                ++diag.balans.built;
                                diag.cl_tlns.fixed = np.fixed;
                                diag.cl_tlns.free_integer = np.free_integer;
                                const model::LpProblem sub =
                                    apply_neighborhood(mip, np);
                                diag.sub_mip_build_ms += ms_since(t_build);
                                std::vector<f64> xs;
                                bool exhausted = false;
                                const bool got = solve_sub_problem(
                                    sub,
                                    std::min(budget, co.time_limit_s),
                                    std::min(balans.node_budget(arm),
                                             co.max_nodes),
                                    xs, exhausted);
                                spent = ms_since(t_cl) / 1000.0;
                                diag.sub_mip_ms += ms_since(t_cl);
                                diag.cl_tlns.seconds += spent;
                                if (got) {
                                    outcome = accept_sub_point(xs)
                                                  ? LnsOutcome::NewBest
                                                  : LnsOutcome::Feasible;
                                    if (outcome == LnsOutcome::NewBest) {
                                        ++diag.balans.hits;
                                        ++diag.cl_tlns.hits;
                                        if (opts.verbose)
                                            std::printf(
                                                "  [milp] CL-TLNS incumbent "
                                                "%.10e\n",
                                                best_incumbent);
                                    }
                                } else {
                                    outcome = exhausted ? LnsOutcome::Exhausted
                                                        : LnsOutcome::Nothing;
                                }
                            } else {
                                spent = ms_since(t_cl) / 1000.0;
                                outcome = LnsOutcome::NotBuilt;
                            }
                        }
                        balans.reward(arm, outcome, spent);
                        diag.balans.seconds += spent;
                    }
                }
            }
        }
        sub_mip_strengthen = false;  // only the root LNS call above uses it

        // ---- Objective-face search ------------------------------------
        // Many losses have a bound that is already the optimum (or within a
        // hair of it) and an incumbent far above it: decomp2 (-160 bound,
        // incumbent 3 -> -7), neos-827175, neos-1171448, neos-1582420,
        // piperout-08. Neighborhood searches around the incumbent cannot
        // reach a point that different. Here the whole model is solved with
        // the objective cut to  c'x <= T,  T just above the proven bound: a
        // feasibility search on the objective face, where propagation on the
        // cutoff row is strong and Feasibility Jump / fix-and-propagate work
        // on a tight target. Each failed attempt doubles the budget and
        // raises T toward the incumbent. A found point is validated like any
        // heuristic point; nothing is concluded from a failure (the child
        // runs on the implied-integer-marked model, so its infeasibility is
        // not evidence about the original).
        // It competes with neighborhood search for the heuristic budget: an
        // unrestricted version spent 14 s of piperout-08's 60 s in three
        // fruitless attempts and left Balans 5 attempts (17 before, one of
        // them a hit). So: only once the tree has visibly stalled (or run a
        // few hundred nodes), and at most 8% of the limit in total.
        if (opts.objective_face && opts.sub_mip_lns && opts.sub_mip_depth == 0 &&
            face_attempts_done < opts.objective_face_max_attempts &&
            diag.nodes >= 1 && opts.time_limit_s > 0.0 &&
            (dual_stalled || diag.nodes >= 400) &&
            diag.face_ms <= opts.objective_face_total_frac * opts.time_limit_s * 1000.0 &&
            ms_since(t0) / 1000.0 >= face_next_start_s && !timed_out()) {
            // Global lower bound now, minimisation sense: nothing open, in
            // the plunge stack or abandoned lies below it.
            f64 bound_now = open.empty() ? std::numeric_limits<f64>::infinity()
                                         : open.top().bound;
            for (const Node& pending : plunge_stack)
                bound_now = std::min(bound_now, pending.bound);
            bound_now = std::min(bound_now, node.bound);
            bound_now = std::min({bound_now, abandoned_bound, pruned_floor, deferred_floor()});
            if (!std::isfinite(bound_now)) bound_now = root_cert.value;
            try_objective_face(bound_now);
        }
        seg(9);  // FJ + LNS/Balans/objective face
        // Dedicated Kernel Pump at the root under Latest (incumbent hunt).
        // Skip on small short-budget MIPs: KP eats root wall that HiGHS spends
        // on cuts/branch; Balans can still call KP later if needed.
        const bool kp_root_ok =
            !(n <= 400 && opts.time_limit_s > 0.0 && opts.time_limit_s <= 45.0);
        if (use_balans && opts.kernel_pump.enabled && diag.nodes == 1 &&
            kp_root_ok && !heuristics_over_budget() &&
            static_cast<Index>(lp_raw.x.size()) == n) {
            const auto t_kp = Clock::now();
            KernelPumpOptions ko = opts.kernel_pump;
            ko.time_limit_s =
                std::min(ko.time_limit_s, std::max(0.0, seconds_left() * 0.2));
            ko.int_tol = opts.int_tol;
            ko.feas_tol = opts.primal_feas_tol;
            if (ko.time_limit_s > 0.05) {
                std::vector<f64> xk;
                KernelPumpDiagnostics kd;
                if (kernel_pump(mip, lp_raw.x, nullptr, ko, xk, kd)) {
                    if (accept_sub_point(xk) && opts.verbose)
                        std::printf("  [milp] KernelPump root incumbent %.10e\n",
                                    best_incumbent);
                    diag.kernel_pump.found = true;
                }
                diag.kernel_pump.pumps += kd.pumps;
                diag.kernel_pump.lp_solves += kd.lp_solves;
                diag.kernel_pump.lp_iterations += kd.lp_iterations;
                diag.kernel_pump.ms += kd.ms;
                diag.kernel_pump.kernel_size =
                    std::max(diag.kernel_pump.kernel_size, kd.kernel_size);
                diag.heuristic_ms += ms_since(t_kp);
            }
        }

        // Dedicated MRENS call on an interval under Latest.
        if (use_balans && opts.mrens.enabled && opts.sub_mip_depth == 0 &&
            diag.nodes >= last_mrens_node + opts.mrens.min_interval &&
            !heuristics_over_budget()) {
            last_mrens_node = diag.nodes;
            const auto t_m = Clock::now();
            ++diag.mrens.attempts;
            std::vector<std::vector<f64>> refs;
            for (const auto& s : lp_snapshots) refs.push_back(s);
            if (refs.empty() && static_cast<Index>(lp_raw.x.size()) == n) {
                synthesize_mrens_refs(mip, lp_raw.x, opts.mrens.max_refs,
                                      opts.int_tol, lns_rng, refs);
            }
            MrensNeighborhood nb;
            MrensOptions mo = opts.mrens;
            mo.int_tol = opts.int_tol;
            const auto t_build = Clock::now();
            const bool mrens_built =
                build_mrens_neighborhood(mip, refs, root_lo, root_hi, mo, nb);
            if (!mrens_built) diag.sub_mip_build_ms += ms_since(t_build);
            if (mrens_built) {
                ++diag.mrens.built;
                diag.mrens.fixed = nb.fixed;
                diag.mrens.free_integer = nb.free_integer;
                diag.mrens.refs_used = refs.size();
                const model::LpProblem sub = apply_mrens(mip, nb);
                diag.sub_mip_build_ms += ms_since(t_build);
                std::vector<f64> xs;
                bool exhausted = false;
                const double budget =
                    std::min(mo.time_limit_s, std::max(0.0, seconds_left() * 0.3));
                if (budget > 0.05 &&
                    solve_sub_problem(sub, budget, mo.max_nodes, xs,
                                      exhausted)) {
                    if (accept_sub_point(xs)) {
                        ++diag.mrens.hits;
                        if (opts.verbose)
                            std::printf("  [milp] MRENS incumbent %.10e\n",
                                        best_incumbent);
                    }
                }
            }
            diag.mrens.seconds += ms_since(t_m) / 1000.0;
            diag.sub_mip_ms += ms_since(t_m);
        }

        // A local repair can get trapped by coupled rows even when a nearby
        // integer branch path is easy. Give small/medium models one bounded
        // LP dive from the current relaxation before committing to the global
        // best-bound tree. The dive is incumbent-only and is skipped once a
        // valid incumbent already exists.
        // Always give the bounded dive a chance at the root. A rounded
        // incumbent can be far worse than a feasible point reached by
        // following the relaxation down a few branch decisions (the classic
        // weakness of independent rounding on markshare/assignment models).
        // Bound the dive by TIME, not by dimension.
        //
        // The old gate was `n_cols <= 3000 && nnz <= 15000`, which disables
        // the classic first-solution heuristic on essentially every real
        // MIPLIB instance: of the NOSOL instances whose tree actually
        // searches, 7 of 8 were blocked by it -- swath3 (6805 cols),
        // air05 (7195), wachplan (89361 nnz), rocI-4-11, ns1830653,
        // momentum1, kakapo -- all reporting "integer dive: 0 attempts"
        // while exploring hundreds or thousands of nodes and never finding a
        // single feasible point. A dimension cap is the wrong instrument when
        // the call already carries its own wall budget (integer_dive_time_s,
        // further clamped below for wide models and by the remaining solve
        // time). The generous ceiling that remains only stops the dive being
        // attempted on models where a single node LP cannot finish inside the
        // budget anyway.
        if (opts.integer_dive && diag.nodes == 1 && !heuristics_over_budget() &&
            problem.n_cols() <= 200000 && problem.nnz() <= 2000000) {
            const auto t_dive = Clock::now();
            ++diag.integer_dive_attempts;
            double dive_budget = opts.integer_dive_time_s;
            if (problem.n_cols() > 1000)
                dive_budget = std::min(dive_budget, 3.0);
            if (opts.time_limit_s > 0.0) {
                const double left = opts.time_limit_s -
                    std::chrono::duration<double>(Clock::now() - t0).count();
                dive_budget = std::min(dive_budget, std::max(0.0, left));
            }
            std::vector<f64> xd;
            std::uint64_t dive_solves = 0;
            const bool dived = dive_budget > 0.0 &&
                try_integer_dive(node_lp, lp_raw.x, node.col_lo, node.col_hi,
                                 &node_basis, opts.int_tol,
                                 opts.primal_feas_tol,
                                 problem.n_cols() > 1000
                                     ? std::min<std::uint64_t>(1024, opts.integer_dive_max_nodes)
                                     : opts.integer_dive_max_nodes,
                                 dive_budget,
                                 problem.n_cols() > 1000
                                     ? std::min(0.01, opts.integer_dive_lp_time_s)
                                     : opts.integer_dive_lp_time_s,
                                 sense, opts.lp, xd,
                                 dive_solves);
            diag.integer_dive_lp_solves += dive_solves;
            if (dived && problem.max_row_violation(xd) <= opts.primal_feas_tol &&
                problem.max_bound_violation(xd) <= opts.primal_feas_tol) {
                if (opts.integer_neighborhood &&
                    mip.n_cols() <= 3000 && mip.nnz() <= 20000) {
                    ++diag.integer_neighborhood_attempts;
                    std::vector<f64> polished;
                    double left = opts.integer_neighborhood_time_s;
                    if (opts.time_limit_s > 0.0) {
                        left = std::min(left, std::max(0.0, opts.time_limit_s -
                            std::chrono::duration<double>(Clock::now() - t0).count()));
                    }
                    if (left > 0.0 && try_integer_neighborhood(
                            problem, xd, opts.int_tol, opts.primal_feas_tol,
                            opts.integer_neighborhood_max_trials, left,
                            opts.lp_rounding_repair_max_iterations,
                            opts.integer_neighborhood_lp_time_s, polished)) {
                        xd = std::move(polished);
                        ++diag.integer_neighborhood_hits;
                    }
                    diag.integer_neighborhood_trials +=
                        opts.integer_neighborhood_max_trials;
                }
                f64 dobj;
                if (accept_incumbent(std::move(xd), false, dobj)) {
                    ++diag.integer_dive_hits;
                    ++diag.heuristic_hits;
                    if (opts.verbose)
                        std::printf("  [milp] dive incumbent %.10e at node %llu\n",
                                    best_incumbent,
                                    static_cast<unsigned long long>(diag.nodes));
                }
            }
            diag.heuristic_ms += ms_since(t_dive);
        }

        if (opts.integer_dive && diag.nodes == 1 && !heuristics_over_budget() &&
            problem.n_cols() <= 3000 && problem.nnz() <= 15000) {
            const auto t_rens = Clock::now();
            double rens_budget = 0.8;
            if (opts.time_limit_s > 0.0)
                rens_budget = std::min(rens_budget, std::max(0.0,
                    opts.time_limit_s - std::chrono::duration<double>(Clock::now() - t0).count()));
            if (rens_budget > 0.02) {
                ++diag.rens_attempts;
                std::vector<f64> xr;
                std::uint64_t rens_solves = 0;
                const bool found_rens = try_rens(
                    node_lp, lp_raw.x, &node_basis, opts.int_tol,
                    opts.primal_feas_tol,
                    problem.n_cols() > 1000 ? 256 : 512,
                    rens_budget,
                    problem.n_cols() > 1000 ? 0.01 : opts.integer_dive_lp_time_s,
                    opts.lp, xr, rens_solves);
                diag.rens_lp_solves += rens_solves;
                if (found_rens && problem.max_row_violation(xr) <= opts.primal_feas_tol &&
                    problem.max_bound_violation(xr) <= opts.primal_feas_tol) {
                    f64 robj;
                    if (accept_incumbent(std::move(xr), false, robj)) {
                        ++diag.rens_hits;
                        ++diag.heuristic_hits;
                    }
                }
            }
            diag.heuristic_ms += ms_since(t_rens);
        }

        // Branch. Under policy=latest, selectable learners (sparse-SB / SC-MILP /
        // Lifted / PlanB&B) choose VARIABLES only. Classical keeps RB. Probes
        // are advisory: only certified relaxations control bounds.
        Index br = -1;
        f64 best_branch_score = -std::numeric_limits<f64>::infinity();
        // ms_branching covers the whole step from here to the variable
        // choice (closed just before `if (br < 0)`); candidate collection and
        // feature construction are also timed exclusively below.
        seg(10);  // kernel pump, MRENS, dive, RENS
        const auto t_branch = Clock::now();
        bool rb_probed = false;   // reliability probes ran at this node
        diag.ms_node_pre_branch += std::max(0.0,
            std::chrono::duration<double, std::milli>(t_branch - t_pre_branch_start).count() -
            (heuristic_spent_ms(diag) - heur_before_pre_branch));
        const auto candidates = branch_candidates(
            mip, node.col_lo, node.col_hi, lp_raw.x, col_degree, opts.int_tol,
            use_reliability ? opts.strong_branch_candidates : 0);
        diag.ms_branch_candidates += ms_since(t_branch);

        BranchFeatureContext feat_ctx;
        feat_ctx.lp = &mip;
        feat_ctx.col_lo = &node.col_lo;
        feat_ctx.col_hi = &node.col_hi;
        feat_ctx.x = &lp_raw.x;
        feat_ctx.pc_down = &pc_down_sum;
        feat_ctx.pc_up = &pc_up_sum;
        feat_ctx.pc_down_count = &pc_down_count;
        feat_ctx.pc_up_count = &pc_up_count;
        feat_ctx.col_degree = &col_degree;
        feat_ctx.int_tol = opts.int_tol;
        feat_ctx.depth = node.depth;
        feat_ctx.dual_bound = node.bound;
        feat_ctx.incumbent = best_incumbent;
        feat_ctx.have_incumbent = have_incumbent;

        // Empty unless a reader exists (branch_features_used); every consumer
        // below checks the size against `candidates` before using them.
        std::vector<BranchFeatureVec> cand_feats;
        std::vector<LiftedFeatureVec> lifted_feats;
        if (branch_features_used && !candidates.empty()) {
            const auto t_features = Clock::now();
            if (!feature_cache.matches(mip, 0)) feature_cache.build(mip, 0);
            feature_cache.set_point(lp_raw.x);
            feat_ctx.cache = &feature_cache;
            cand_feats.reserve(candidates.size());
            if (lifted_features_used) lifted_feats.reserve(candidates.size());
            for (const Index j : candidates) {
                ++diag.branch_feature_vectors;
                BranchFeatureVec fv{};
                if (fill_branch_features(feat_ctx, j, fv))
                    cand_feats.push_back(fv);
                else
                    cand_feats.push_back(BranchFeatureVec{});
                if (lifted_features_used) {
                    LiftedFeatureVec lf{};
                    if (!fill_lifted_features(feat_ctx, j, lf)) {
                        for (int i = 0; i < kBranchFeatureDim; ++i)
                            lf[static_cast<std::size_t>(i)] =
                                cand_feats.back()[static_cast<std::size_t>(i)];
                    }
                    lifted_feats.push_back(lf);
                }
            }
            diag.ms_branch_features += ms_since(t_features);
        }

        if (planbb_want && (planbb_paper || collect_planbb) &&
            !candidates.empty()) {
            const auto snap = build_bipartite_snapshot(mip, node.col_lo,
                                                      node.col_hi, &lp_raw.x);
            pool_bipartite_for_planbb(snap, planbb_graph);
        }

        bool learner_chose = false;
        auto try_sparse_sb_pick = [&]() -> bool {
            if (!sparse_sb_model_ready || candidates.empty() ||
                cand_feats.size() != candidates.size())
                return false;
            const Index sb_br =
                pick_sparse_sb_branch(sparse_sb_model, candidates, cand_feats);
            if (sb_br < 0) return false;
            br = sb_br;
            ++diag.sparse_sb_picks;
            diag.last_branch_policy = "sparse-sb";
            return true;
        };
        auto try_sc_milp_pick = [&]() -> bool {
            if (!sc_milp_want || candidates.empty() ||
                cand_feats.size() != candidates.size())
                return false;
            const Index sc_br = pick_sc_milp_branch(
                sc_milp_model, candidates, cand_feats,
                sc_milp_opts.use_heuristic_without_model);
            if (sc_br < 0) return false;
            br = sc_br;
            ++diag.sc_milp_picks;
            diag.last_branch_policy = "sc-milp";
            return true;
        };
        auto try_lifted_pick = [&]() -> bool {
            if (!lifted_want || candidates.empty() ||
                lifted_feats.size() != candidates.size())
                return false;
            const Index lb =
                pick_lifted_branch(lifted_state, candidates, lifted_feats);
            if (lb < 0) return false;
            br = lb;
            ++diag.lifted_picks;
            diag.last_branch_policy = "lifted";
            return true;
        };

        if (latest_branch && !candidates.empty()) {
            switch (branch_strat) {
            case BranchStrategy::SparseSb:
                learner_chose = try_sparse_sb_pick();
                break;
            case BranchStrategy::ScMilp:
                learner_chose = try_sc_milp_pick();
                if (!learner_chose) learner_chose = try_sparse_sb_pick();
                if (!learner_chose) ++diag.sc_milp_fallbacks;
                break;
            case BranchStrategy::Lifted:
                learner_chose = try_lifted_pick();
                if (!learner_chose) learner_chose = try_sparse_sb_pick();
                if (!learner_chose) ++diag.lifted_fallbacks;
                break;
            case BranchStrategy::PlanBb: {
                // Prefer probe-backed MCTS / shallow lookahead when reliability
                // probes will run; otherwise pick from policy / paper model now.
                const bool want_est =
                    (opts.planbb.use_mcts || opts.planbb.shallow_lookahead) &&
                    use_reliability && !planbb_paper;
                // Paper path can pick without probes (learned dynamics).
                if (!want_est || planbb_paper) {
                    std::uint64_t mcts_sims = 0;
                    const Index pb = pick_planbb_branch(
                        planbb_paper ? &planbb_model : nullptr, planbb_policy,
                        opts.planbb, candidates, cand_feats, &planbb_graph,
                        nullptr, &mcts_sims);
                    diag.planbb_mcts_sims += mcts_sims;
                    if (pb >= 0) {
                        br = pb;
                        learner_chose = true;
                        ++diag.planbb_picks;
                        diag.last_branch_policy =
                            planbb_paper ? "planbb-paper" : "planbb";
                    } else {
                        learner_chose = try_sparse_sb_pick();
                        if (!learner_chose) ++diag.planbb_fallbacks;
                    }
                }
                break;
            }
            case BranchStrategy::Auto:
            case BranchStrategy::Reliability:
                break;
            }
        }

        const auto choose_score = [&](Index j, f64 xv) {
            const f64 down_dist = xv - std::floor(xv);
            const f64 up_dist = std::ceil(xv) - xv;
            const bool down_ready =
                pc_down_count[sz(j)] >= static_cast<std::uint32_t>(
                    std::max(1, opts.reliability_threshold));
            const bool up_ready =
                pc_up_count[sz(j)] >= static_cast<std::uint32_t>(
                    std::max(1, opts.reliability_threshold));
            const f64 down_est = down_ready
                ? (pc_down_sum[sz(j)] / pc_down_count[sz(j)]) * down_dist : 0.0;
            const f64 up_est = up_ready
                ? (pc_up_sum[sz(j)] / pc_up_count[sz(j)]) * up_dist : 0.0;
            if (down_ready && up_ready)
                return std::min(down_est, up_est) + 0.1 * std::max(down_est, up_est);
            // Fractionality remains the safe cold-start ordering. Existing
            // one-sided pseudocost information is only a deterministic tie
            // breaker until both directions are reliable.
            return frac_score(xv) + 1e-9 * (down_est + up_est);
        };

        if (!learner_chose) {
            if (latest_branch && branch_strat == BranchStrategy::SparseSb &&
                sparse_sb_want)
                ++diag.sparse_sb_fallbacks;
            for (const Index j : candidates) {
                const f64 xv = lp_raw.x[sz(j)];
                const f64 score = choose_score(j, xv);
                if (score > best_branch_score) {
                    best_branch_score = score;
                    br = j;
                }
            }
            if (br < 0)
                br = pick_branch_var(mip, node.col_lo, node.col_hi,
                                     lp_raw.x, opts.int_tol);
            if (!learner_chose)
                diag.last_branch_policy = "reliability";
        }
        // Strong probes: under classical / unlabeled latest, they override the
        // pseudocost pick. Under learners they still seed pseudocosts and can
        // collect imitation labels, but do not displace the model choice unless
        // the learner was unavailable (learner_chose == false). PlanB&B-lite
        // may re-pick using child estimates after probes.
        std::vector<PlanBbChildEstimate> planbb_ests;
        const bool planbb_want_est =
            latest_branch && branch_strat == BranchStrategy::PlanBb &&
            (opts.planbb.use_mcts || opts.planbb.shallow_lookahead ||
             planbb_paper || collect_planbb);
        if (planbb_want_est)
            planbb_ests.assign(candidates.size(), PlanBbChildEstimate{});


        // ---- Reliability branching on the node-LP session --------------
        // All fractional candidates are ranked by pseudocost score; only the
        // UNRELIABLE ones (fewer than reliability_threshold observations in
        // some direction) are strong-branched, most promising first, until
        // `lookahead` probed candidates in a row fail to improve the best
        // score or the work budget is spent. Probes reuse the node session's
        // prepared LP and factor (no per-probe preparation), are capped at
        // clamp(2 * average node-LP pivots, 10, 500) pivots, and stop at the
        // incumbent cutoff. A direction that is CERTIFIED dead -- Farkas
        // ray verified on the probe's LP, or a safe Lagrangian bound past
        // the cutoff -- tightens the node's own bound on that variable;
        // both directions dead close the node. The budget is a share of
        // node-LP time, replacing the old size cut-off (n > 2000 or
        // nnz > 10000 disabled probing outright) and the 128-probe cap.
        bool reliable_selected = false;
        bool node_requeued = false;
        // Branching-decision trace state (--trace-branching N).
        const bool tr_on = opts.trace_branching > 0 && opts.sub_mip_depth == 0 &&
                           static_cast<int>(diag.branching_traced) < opts.trace_branching;
        const char* tr_stop = "not-rb";
        // Why the reliability block was not entered (first failing condition).
        const char* tr_skip = !opts.reliability_branching ? "opt-off"
            : learner_chose ? "learner"
            : !(branch_strat == BranchStrategy::Auto || branch_strat == BranchStrategy::Reliability) ? "strategy"
            : !opts.paper_reliability ? "no-paper-rb"
            : !node_lp_proved ? "lp-unproved"
            : !node_session ? "no-session"
            : !session_state_is_last_node ? "session-not-last-node"
            : false ? "local-rows"
            : node_session_generation != session_key ? "session-generation"
            : node_basis.basic.empty() ? "no-basis"
            : diag.nodes < 1 ? "root" : "entered";
        int tr_rank = -1, tr_cands = 0, tr_probed = 0, tr_reliable_cands = 0;
        const auto tr_domain_hash = [&]() {
            std::uint64_t h = 1469598103934665603ull;
            for (Index j = 0; j < n; ++j) {
                const auto u = sz(j);
                if (node.col_lo[u] == root_lo[u] && node.col_hi[u] == root_hi[u]) continue;
                std::uint64_t a, b;
                std::memcpy(&a, &node.col_lo[u], 8);
                std::memcpy(&b, &node.col_hi[u], 8);
                h = (h ^ (a + 0x9e3779b97f4a7c15ull * (static_cast<std::uint64_t>(j) + 1))) * 1099511628211ull;
                h = (h ^ b) * 1099511628211ull;
            }
            return h;
        };
        if (opts.reliability_branching && !learner_chose &&
            (branch_strat == BranchStrategy::Auto ||
             branch_strat == BranchStrategy::Reliability) &&
            opts.paper_reliability && node_lp_proved && node_session &&
            session_state_is_last_node &&
            node_session_generation == session_key &&
            !node_basis.basic.empty() && diag.nodes >= 1) {
            const auto t_rb = Clock::now();
            const auto all_cands = branch_candidates(
                mip, node.col_lo, node.col_hi, lp_raw.x, col_degree,
                opts.int_tol, 0);
            if (all_cands.size() >= 2 || (all_cands.size() == 1 && false)) {
                // Average observed pseudocosts stand in for a direction
                // with no observations yet.
                f64 sum_d = 0.0, sum_u = 0.0;
                std::uint64_t cnt_d = 0, cnt_u = 0;
                for (Index j = 0; j < n; ++j) {
                    sum_d += pc_down_sum[sz(j)]; cnt_d += pc_down_count[sz(j)];
                    sum_u += pc_up_sum[sz(j)]; cnt_u += pc_up_count[sz(j)];
                }
                const f64 avg_d = cnt_d ? sum_d / static_cast<f64>(cnt_d) : 1.0;
                const f64 avg_u = cnt_u ? sum_u / static_cast<f64>(cnt_u) : 1.0;
                const std::uint32_t eta =
                    static_cast<std::uint32_t>(std::max(1, opts.reliability_threshold));
                struct Cand { Index j; f64 score; bool reliable; };
                std::vector<Cand> ranked;
                ranked.reserve(all_cands.size());
                for (const Index j : all_cands) {
                    const f64 xv = lp_raw.x[sz(j)];
                    const f64 dd = xv - std::floor(xv), du = std::ceil(xv) - xv;
                    const f64 pd = pc_down_count[sz(j)]
                        ? pc_down_sum[sz(j)] / pc_down_count[sz(j)] : avg_d;
                    const f64 pu = pc_up_count[sz(j)]
                        ? pc_up_sum[sz(j)] / pc_up_count[sz(j)] : avg_u;
                    const f64 sc = std::max(pd * dd, 1e-6) * std::max(pu * du, 1e-6);
                    ranked.push_back({j, sc,
                        pc_down_count[sz(j)] >= eta && pc_up_count[sz(j)] >= eta});
                }
                FeasibilityMeans fmeans = feasibility_means(bstats);
                const auto cand_key = [&](Index j, f64 score) {
                    return branch_key(bstats, j, score, fmeans);
                };
                // Objective score first (to numerical resolution), then observed
                // closure rate, then observed inference rate; the stable sort
                // keeps the fractionality/degree order for whatever still ties.
                std::stable_sort(ranked.begin(), ranked.end(),
                                 [&](const Cand& a, const Cand& b) {
                                     return key_better(cand_key(a.j, a.score),
                                                       cand_key(b.j, b.score));
                                 });
                const f64 lp_bound_here = lp_obj_min;
                const f64 cutoff_probe = cutoff_known()
                    ? node_cutoff(known_cutoff_min(), obj_granularity,
                                  offset_min, opts.gap_tol, opts.abs_gap_tol)
                    : std::numeric_limits<f64>::infinity();
                const std::uint64_t avg_iters = branching_lp_work.solves > 0
                    ? branching_lp_work.iterations / branching_lp_work.solves : 50;
                const std::uint64_t probe_cap =
                    std::clamp<std::uint64_t>(2 * avg_iters, 10, 500);
                const int max_probed = std::max(1, opts.rb_max_probed);
                const int lookahead = std::max(1, opts.rb_lookahead_candidates);
                int probed = 0, since_improve = 0;
                BranchKey best_key{std::numeric_limits<long long>::min(), -1.0, -1.0};
                Index best_j = ranked.front().j;
                bool dead_found = false, node_closed = false;
                Index dead_col = -1;
                // The batch's own running time counts before it is added to the
                // total, and is re-checked before every direction.
                const auto within_budget = [&]() {
                    return batch_has_allowance(diag.strong_branch_ms, ms_since(t_rb),
                                               opts.rb_lp_time_share, branching_lp_work.ms,
                                               opts.rb_startup_ms);
                };
                for (Cand& c : ranked) {
                    if (c.reliable) {
                        const BranchKey k = cand_key(c.j, c.score);
                        if (key_better(k, best_key)) { best_key = k; best_j = c.j; }
                        continue;
                    }
                    if (probed >= max_probed || since_improve >= lookahead ||
                        !within_budget() || timed_out()) {
                        tr_stop = probed >= max_probed ? "max-probed"
                                  : since_improve >= lookahead ? "lookahead"
                                  : timed_out() ? "time" : "budget";
                        break;
                    }
                    ++probed;
                    const Index j = c.j;
                    const f64 xv = lp_raw.x[sz(j)];
                    const f64 fl = std::floor(xv), ce = std::ceil(xv);
                    f64 gain[2] = {0.0, 0.0};
                    bool dead[2] = {false, false};
                    bool have[2] = {false, false};
                    for (int dir = 0; dir < 2; ++dir) {  // 0 = down, 1 = up
                        if (dir > 0 && (!within_budget() || timed_out())) break;
                        if (seconds_left() <= 0.0) break;
                        f64 plo = node.col_lo[sz(j)], phi = node.col_hi[sz(j)];
                        if (dir == 0) phi = std::min(phi, fl); else plo = std::max(plo, ce);
                        engines::SimplexOptions po = lp_opts;
                        po.presolve = false;
                        po.method = engines::SimplexMethod::Dual;
                        po.max_iterations = probe_cap;
                        po.time_limit_s = std::min(0.25, seconds_left());
                        po.objective_limit = cutoff_probe;
                        engines::SimplexDiagnostics pd;
                        const auto praw = node_session->probe(j, plo, phi, po, pd, node_basis);
                        ++diag.strong_branch_solves;
                        diag.strong_branch_iterations += pd.iterations;
                        diag.strong_branch_prep_ms += pd.preprocessing_ms;
                        diag.strong_branch_factor_ms += pd.first_factor_ms;
                        diag.strong_branch_loop_ms += pd.loop_ms;
                        diag.strong_branch_simplex_ms += pd.total_ms;
                        diag.strong_branch_dse_rebuilds += pd.dse_weight_rebuilds;
                        if (pd.factor_reused) ++diag.strong_branch_factor_reuses;
                        const bool proved = relaxation_proved(praw, pd, po);
                        // Certified dead direction?
                        bool is_dead = false;
                        bool infeasible_direction = false;
                        f64 direction_bound = core::kNaN;
                        if (!proved) {
                            std::vector<f64> save_lo = node_lp.col_lo, save_hi = node_lp.col_hi;
                            node_lp.col_lo[sz(j)] = plo;
                            node_lp.col_hi[sz(j)] = phi;
                            if (node_lp_infeasibility_proved(node_lp, praw, opts.primal_feas_tol,
                                                             root_relaxation_bounded)) {
                                is_dead = true;
                                infeasible_direction = true;
                                if (!try_learn_farkas(node_lp, praw, node.active_local.empty(), true)) {
                                    PropTrail probe_trail = node.prop_trail;
                                    probe_trail.push(j, dir == 0 ? BoundDir::Upper : BoundDir::Lower,
                                        dir == 0 ? phi : plo,
                                        dir == 0 ? node.col_hi[sz(j)] : node.col_lo[sz(j)],
                                        ReasonKind::Branch, -1, node.depth + 1);
                                    try_learn_nogood(probe_trail);
                                }
                            } else if (praw.termination_reason == "objective limit" &&
                                     praw.y.size() == sz(node_lp.n_rows())) {
                                std::vector<f64> y_min(praw.y.size());
                                for (std::size_t i = 0; i < y_min.size(); ++i)
                                    y_min[i] = sense * praw.y[i];
                                const auto safe = certify::safe_lagrangian_lower_bound(
                                    node_lp, y_min, node_lp.col_lo, node_lp.col_hi);
                                if (safe.finite && safe.value >= cutoff_probe) {
                                    is_dead = true;
                                    direction_bound = safe.value;
                                }
                            }
                            node_lp.col_lo = std::move(save_lo);
                            node_lp.col_hi = std::move(save_hi);
                        } else if (cutoff_known() &&
                                   node_lp_bound_min(praw, sense) >= cutoff_probe) {
                            is_dead = true;  // proved optimum already past the cutoff
                            direction_bound = node_lp_bound_min(praw, sense);
                        }
                        if (is_dead) {
                            dead[dir] = true;
                            // The discarded child has its own certificate.
                            // Its parent's weaker bound does not describe why
                            // this direction was closed. Preserve cutoff
                            // certificates even when the other side survives.
                            if (!infeasible_direction && std::isfinite(direction_bound)) {
                                pruned_floor = std::min(pruned_floor,
                                    std::max(node.bound, direction_bound));
                                ++diag.gap_prunes;
                            }
                            ++diag.strong_branch_infeasible;
                            const bool recorded = bstats.record(j, dir == 0 ? -1 : 1,
                                dir == 0 ? xv - fl : ce - xv,
                                infeasible_direction ? BranchOutcome::CertifiedInfeasible
                                                     : BranchOutcome::CertifiedCutoff,
                                proved ? std::max(0.0, node_lp_bound_min(praw, sense) - lp_bound_here)
                                       : core::kNaN,
                                -1.0, opts.int_tol);
                            if (recorded && proved) ++diag.pseudocost_updates;
                        } else if (proved) {
                            ++diag.strong_branch_proved;
                            const f64 dist = dir == 0 ? xv - fl : ce - xv;
                            const f64 g = std::max(0.0, node_lp_bound_min(praw, sense) -
                                                            lp_bound_here);
                            gain[dir] = g;
                            have[dir] = true;
                            if (bstats.record(j, dir == 0 ? -1 : 1, dist,
                                              BranchOutcome::ProvedOptimal, g, -1.0, opts.int_tol))
                                ++diag.pseudocost_updates;
                        } else {
                            ++diag.strong_branch_unproved;
                        }
                    }
                    if (dead[0] && dead[1]) { node_closed = true; break; }
                    if (dead[0] || dead[1]) {
                        // Only the surviving side can hold a better point.
                        if (dead[0]) node.col_lo[sz(j)] = std::max(node.col_lo[sz(j)], ce);
                        else node.col_hi[sz(j)] = std::min(node.col_hi[sz(j)], fl);
                        ++diag.sb_domain_reductions;
                        dead_found = true;
                        dead_col = j;
                        break;
                    }
                    if (have[0] && have[1]) {
                        const f64 sc = std::max(gain[0], 1e-6) * std::max(gain[1], 1e-6);
                        // The inference probe below compares leaders with the
                        // best PROBED score. Keeping the pre-probe estimate
                        // here excluded precisely those leaders whenever the
                        // observed gains differed from that estimate.
                        c.score = sc;
                        const BranchKey k = cand_key(j, sc);
                        if (key_better(k, best_key)) { best_key = k; best_j = j; since_improve = 0; }
                        else ++since_improve;
                    } else {
                        ++since_improve;
                    }
                }
                // Feasibility probe. When the leaders are tied on the objective
                // score and one of them has no measured propagation deductions,
                // give ONE such candidate a two-direction domain probe (bound
                // propagation to a fixpoint on a copy of the node box). A
                // direction that propagation empties is certified dead; a
                // surviving one contributes its inference count.
                if (!dead_found && !node_closed && opts.domain_propagation &&
                    within_budget() && !timed_out()) {
                    Index probe_j = -1;
                    for (const Cand& c : ranked) {
                        if (score_bucket(c.score) != best_key.bucket) continue;
                        if (inference_unobserved(bstats, c.j)) { probe_j = c.j; break; }
                    }
                    if (probe_j >= 0) {
                        ++diag.domain_probe_candidates;
                        const f64 xv = lp_raw.x[sz(probe_j)];
                        const f64 fl = std::floor(xv), ce = std::ceil(xv);
                        bool dir_dead[2] = {false, false};
                        for (int dir = 0; dir < 2; ++dir) {
                            if (!within_budget() || timed_out()) break;
                            std::vector<f64> lo = node.col_lo, hi = node.col_hi;
                            if (dir == 0) hi[sz(probe_j)] = std::min(hi[sz(probe_j)], fl);
                            else lo[sz(probe_j)] = std::max(lo[sz(probe_j)], ce);
                            DomainFixpoint dfx;
                            dfx.add("rows", [&]() {
                                const PropagateResult pr = propagate_bounds(
                                    node_lp, lo, hi, opts.primal_feas_tol,
                                    opts.propagation_max_rounds);
                                FixpointStep st;
                                st.feasible = pr.feasible;
                                st.changed = pr.tightened > 0;
                                st.converged = pr.feasible && pr.rounds < opts.propagation_max_rounds;
                                return st;
                            });
                            if (opts.conflict_propagation && !conflict_graph.empty())
                                dfx.add("graph", [&]() {
                                    std::uint64_t forced = 0;
                                    FixpointStep st;
                                    st.feasible = propagate_conflicts(conflict_graph, lo, hi, forced);
                                    st.changed = forced > 0;
                                    return st;
                                });
                            if (conflict_cut_opts.conflict_store && !conflict_store.empty())
                                dfx.add("store", [&]() {
                                    const std::uint64_t before = conflict_store.stats().tightenings;
                                    FixpointStep st;
                                    st.feasible = conflict_store.propagate(lo, hi, root_lo, root_hi,
                                                                          &st.converged);
                                    st.changed = conflict_store.stats().tightenings > before;
                                    // These deductions belong to the copied
                                    // probe box. Keep the diagnostics current
                                    // so the next ordinary child's delta does
                                    // not accidentally include this probe.
                                    diag.conflict_store = conflict_store.stats();
                                    return st;
                                });
                            const FixpointRun fr = dfx.run(16);
                            ++diag.domain_probe_runs;
                            f64 changed = 0.0;
                            for (Index k = 0; k < n; ++k)
                                changed += (lo[sz(k)] > node.col_lo[sz(k)]) +
                                           (hi[sz(k)] < node.col_hi[sz(k)]);
                            // The trial branch itself is not an inference.
                            changed = std::max(0.0, changed - 1.0);
                            dir_dead[dir] = fr.status == FixpointStatus::Infeasible;
                            if (dir_dead[dir]) {
                                // Complete decisions cover row/graph/clause
                                // cascades without a complete reason trail.
                                // Pending propagation never learns a nogood.
                                PropTrail probe_trail = node.prop_trail;
                                probe_trail.push(probe_j,
                                    dir == 0 ? BoundDir::Upper : BoundDir::Lower,
                                    dir == 0 ? fl : ce,
                                    dir == 0 ? node.col_hi[sz(probe_j)] : node.col_lo[sz(probe_j)],
                                    ReasonKind::Branch, -1, node.depth + 1);
                                try_learn_nogood(probe_trail);
                            }
                            const f64 dist = dir == 0 ? xv - fl : ce - xv;
                            (void)bstats.record(probe_j, dir == 0 ? -1 : 1, dist,
                                                dir_dead[dir] ? BranchOutcome::CertifiedInfeasible
                                                              : BranchOutcome::DomainOnly,
                                                core::kNaN, changed, opts.int_tol);
                        }
                        if (dir_dead[0] && dir_dead[1]) {
                            node_closed = true;
                            ++diag.domain_probe_closures;
                        } else if (dir_dead[0] || dir_dead[1]) {
                            if (dir_dead[0]) node.col_lo[sz(probe_j)] = std::max(node.col_lo[sz(probe_j)], ce);
                            else node.col_hi[sz(probe_j)] = std::min(node.col_hi[sz(probe_j)], fl);
                            ++diag.domain_probe_deductions;
                            ++diag.sb_domain_reductions;
                            dead_found = true;
                            dead_col = probe_j;
                        } else {
                            // Both survive: its new inference rate may now
                            // separate it from the tied leaders.
                            fmeans = feasibility_means(bstats);
                            for (const Cand& c : ranked)
                                if (c.j == best_j) { best_key = cand_key(best_j, c.score); break; }
                            const BranchKey k = cand_key(probe_j, [&]() {
                                for (const Cand& c : ranked) if (c.j == probe_j) return c.score;
                                return 1e-12; }());
                            if (key_better(k, best_key)) { best_key = k; best_j = probe_j; }
                        }
                    }
                }
                diag.strong_branch_ms += ms_since(t_rb);
                if (probed > 0) rb_probed = true;
                tr_probed = probed;
                tr_cands = static_cast<int>(ranked.size());
                for (const Cand& c : ranked) tr_reliable_cands += c.reliable ? 1 : 0;
                if (std::strcmp(tr_stop, "not-rb") == 0) tr_stop = "all-reliable-or-done";
                if (tr_on && (node_closed || dead_found))
                    std::printf("[brtrace] n=%llu parent=%llu depth=%d event=%s var=%d probed=%d\n",
                                static_cast<unsigned long long>(diag.nodes),
                                static_cast<unsigned long long>(node.parent_serial), node.depth,
                                node_closed ? "closed" : "requeue-deduction",
                                static_cast<int>(node_closed ? -1 : dead_col), probed);
                if (node_closed) {
                    ++diag.sb_nodes_closed;
                    // Every cutoff-closed direction retained its certificate
                    // above. Infeasible directions contain no feasible point.
                    diag.ms_exit_b += ms_since(t_loop_iter); ++diag.n_exit_b;
                    continue;
                }
                if (dead_found) {
                    // The node's LP point is no longer inside its box: solve
                    // it again with the tightened bound (warm start).
                    node.basis = node_basis;
                    node.has_basis = !node_basis.basic.empty();
                    // The tightened column is the only unpropagated change.
                    node.dirty.assign(1, dead_col);
                    open.push(std::move(node));
                    diag.ms_branching += ms_since(t_branch);
                    node_requeued = true;
                } else {
                    br = best_j;
                    for (std::size_t q = 0; q < ranked.size(); ++q)
                        if (ranked[q].j == br) { tr_rank = static_cast<int>(q); break; }
                    reliable_selected = true;
                    diag.last_branch_policy = "reliability";
                }
            }
        }
        if (node_requeued) continue;
        if (reliable_selected) ++diag.rb_reliable_nodes;

        bool strong_probes_ran = false;
        if (!reliable_selected && use_reliability && node_lp_proved &&
            diag.strong_branch_solves < strong_branch_budget &&
            !candidates.empty() && !node_basis.basic.empty()) {
            const auto t_strong = Clock::now();
            struct StrongTick {
                double& sink;
                Clock::time_point t;
                ~StrongTick() { sink += ms_since(t); }
            } strong_tick{diag.strong_branch_ms, t_strong};
            const std::uint64_t solves_before = diag.strong_branch_solves;
            f64 strong_best = -std::numeric_limits<f64>::infinity();
            Index strong_var = -1;
            const engines::SimplexBasis* warm = &node_basis;

            bool batch_ranked = false;
            if (opts.batch_lp_strong_branch) {
                std::vector<engines::BatchBoundProbe> probes;
                struct ProbeMeta {
                    std::size_t ci;
                    Index j;
                    int dir;
                    f64 dist;
                };
                std::vector<ProbeMeta> meta;
                const std::uint64_t remaining =
                    strong_branch_budget > diag.strong_branch_solves
                        ? strong_branch_budget - diag.strong_branch_solves
                        : 0;
                for (std::size_t ci = 0;
                     ci < candidates.size() && meta.size() < remaining; ++ci) {
                    const Index j = candidates[ci];
                    const f64 xv = lp_raw.x[sz(j)];
                    const f64 floor_v = std::floor(xv);
                    const f64 ceil_v = std::ceil(xv);
                    const f64 down_dist = xv - floor_v;
                    const f64 up_dist = ceil_v - xv;
                    if (down_dist <= opts.int_tol || up_dist <= opts.int_tol)
                        continue;
                    for (int dir : {-1, 1}) {
                        if (meta.size() >= remaining) break;
                        engines::BatchBoundProbe pr;
                        pr.col_lo = node.col_lo;
                        pr.col_hi = node.col_hi;
                        if (dir < 0)
                            pr.col_hi[sz(j)] =
                                std::min(pr.col_hi[sz(j)], floor_v);
                        else
                            pr.col_lo[sz(j)] =
                                std::max(pr.col_lo[sz(j)], ceil_v);
                        probes.push_back(std::move(pr));
                        meta.push_back(
                            {ci, j, dir, dir < 0 ? down_dist : up_dist});
                    }
                }
                if (probes.size() >= 2) {
                    try {
                        engines::BatchProbeOptions bopt;
                        bopt.hpr_steps = opts.batch_lp_sb_steps;
                        // Use node_lp matrix with current objective; bounds
                        // come from probes (node domains).
                        model::LpProblem batch_lp = node_lp;
                        batch_lp.col_lo = node.col_lo;
                        batch_lp.col_hi = node.col_hi;
                        const auto results = engines::batch_bound_probes_hpr(
                            batch_lp, probes, bopt);
                        ++diag.batch_lp_sb_batches;
                        diag.batch_lp_sb_probes += results.size();
                        diag.strong_branch_solves += results.size();

                        std::vector<f64> down_gain(
                            candidates.size(),
                            std::numeric_limits<f64>::quiet_NaN());
                        std::vector<f64> up_gain(
                            candidates.size(),
                            std::numeric_limits<f64>::quiet_NaN());
                        std::vector<char> down_ok(candidates.size(), 0);
                        std::vector<char> up_ok(candidates.size(), 0);

                        for (std::size_t pi = 0; pi < results.size(); ++pi) {
                            const auto& r = results[pi];
                            const auto& m = meta[pi];
                            if (!r.looks_feasible ||
                                !std::isfinite(r.primal_obj))
                                continue;
                            const f64 gain =
                                std::max(0.0, r.primal_obj - lp_obj_min);
                            const f64 unit =
                                gain / std::max(m.dist, opts.int_tol);
                            // Approximate (first-order) objectives rank candidates
                            // but are not certified bounds: no pseudocost sample.
                            (void)unit;
                            if (m.dir < 0) {
                                down_gain[m.ci] = gain;
                                down_ok[m.ci] = 1;
                            } else {
                                up_gain[m.ci] = gain;
                                up_ok[m.ci] = 1;
                            }
                        }

                        for (std::size_t ci = 0; ci < candidates.size(); ++ci) {
                            if (!down_ok[ci] || !up_ok[ci]) continue;
                            const f64 dg = down_gain[ci];
                            const f64 ug = up_gain[ci];
                            const f64 score =
                                std::min(dg, ug) + 0.1 * std::max(dg, ug);
                            if (score > strong_best) {
                                strong_best = score;
                                strong_var = candidates[ci];
                            }
                            if (ci < planbb_ests.size()) {
                                planbb_ests[ci].down_gain = dg;
                                planbb_ests[ci].up_gain = ug;
                                planbb_ests[ci].ok =
                                    std::isfinite(dg) && std::isfinite(ug);
                            }
                            if (std::isfinite(dg) && std::isfinite(ug) &&
                                ci < cand_feats.size()) {
                                const f64 prod = sb_product_score(dg, ug);
                                const int decision_id =
                                    static_cast<int>(diag.nodes);
                                if (collect_sparse_sb) {
                                    sparse_sb_collector.add(cand_feats[ci],
                                                            prod);
                                    ++diag.sparse_sb_samples;
                                }
                                if (collect_sc_milp) {
                                    sc_milp_collector.add(cand_feats[ci], prod,
                                                         decision_id);
                                    ++diag.sc_milp_samples;
                                }
                                if (lifted_want && ci < lifted_feats.size()) {
                                    lifted_state.observe(lifted_feats[ci],
                                                         prod);
                                    ++diag.lifted_samples;
                                    if (lifted_state.maybe_refit(opts.lifted))
                                        ++diag.lifted_refits;
                                }
                                if (collect_planbb) {
                                    planbb_collector.add(cand_feats[ci],
                                                         planbb_graph, dg, ug,
                                                         /*pruned=*/false);
                                    ++diag.planbb_samples;
                                }
                            }
                            batch_ranked = true;
                        }
                    } catch (...) {
                        batch_ranked = false;
                    }
                }
            }

            if (!batch_ranked)
            for (std::size_t ci = 0; ci < candidates.size(); ++ci) {
                const Index j = candidates[ci];
                const f64 xv = lp_raw.x[sz(j)];
                const f64 floor_v = std::floor(xv);
                const f64 ceil_v = std::ceil(xv);
                const f64 down_dist = xv - floor_v;
                const f64 up_dist = ceil_v - xv;
                if (down_dist <= opts.int_tol || up_dist <= opts.int_tol)
                    continue;

                f64 down_gain = std::numeric_limits<f64>::quiet_NaN();
                f64 up_gain = std::numeric_limits<f64>::quiet_NaN();
                bool down_ok = false, up_ok = false;
                for (int dir : {-1, 1}) {
                    if (diag.strong_branch_solves >= strong_branch_budget)
                        break;
                    node_lp.col_lo = node.col_lo;
                    node_lp.col_hi = node.col_hi;
                    if (dir < 0)
                        node_lp.col_hi[sz(j)] = std::min(node_lp.col_hi[sz(j)], floor_v);
                    else
                        node_lp.col_lo[sz(j)] = std::max(node_lp.col_lo[sz(j)], ceil_v);

                    engines::SimplexOptions probe_opts = lp_opts;
                    probe_opts.presolve = false;
                    probe_opts.method = engines::SimplexMethod::Dual;
                    probe_opts.max_iterations = std::min<std::uint64_t>(
                        probe_opts.max_iterations, 5000);
                    probe_opts.time_limit_s = opts.strong_branch_time_s;
                    if (opts.time_limit_s > 0.0) {
                        const double left = opts.time_limit_s -
                            std::chrono::duration<double>(Clock::now() - t0).count();
                        if (left <= 0.0) break;
                        probe_opts.time_limit_s = std::min(probe_opts.time_limit_s, left);
                    }
                    engines::SimplexDiagnostics probe_diag;
                    engines::SimplexBasis probe_basis;
                    const auto probe_raw = engines::solve_dual_simplex(
                        node_lp, probe_opts, probe_diag, &probe_basis, warm);
                    ++diag.strong_branch_solves;
                    diag.strong_branch_iterations += probe_diag.iterations;
                    diag.strong_branch_prep_ms += probe_diag.preprocessing_ms;
                    diag.strong_branch_factor_ms += probe_diag.first_factor_ms;
                    diag.strong_branch_loop_ms += probe_diag.loop_ms;
                    diag.strong_branch_simplex_ms += probe_diag.total_ms;
                    diag.strong_branch_dse_rebuilds +=
                        probe_diag.dse_weight_rebuilds;
                    if (probe_diag.factor_reused)
                        ++diag.strong_branch_factor_reuses;
                    const bool proved = relaxation_proved(probe_raw, probe_diag,
                                                          probe_opts);
                    const bool infeasible =
                        probe_raw.proposed_status == core::Status::Infeasible;
                    if (proved) ++diag.strong_branch_proved;
                    else if (infeasible) ++diag.strong_branch_infeasible;
                    else ++diag.strong_branch_unproved;
                    if (dir < 0) {
                        down_ok = proved || infeasible;
                        if (proved)
                            down_gain = std::max(0.0,
                                sense * probe_raw.objective - lp_obj_min);
                        else if (infeasible)
                            down_gain = std::numeric_limits<f64>::infinity();
                    } else {
                        up_ok = proved || infeasible;
                        if (proved)
                            up_gain = std::max(0.0,
                                sense * probe_raw.objective - lp_obj_min);
                        else if (infeasible)
                            up_gain = std::numeric_limits<f64>::infinity();
                    }
                    if (proved) {
                        const f64 unit_gain = std::max(0.0,
                            (dir < 0 ? down_gain : up_gain) /
                            (dir < 0 ? down_dist : up_dist));
                        (void)unit_gain;
                        if (bstats.record(j, dir, dir < 0 ? down_dist : up_dist,
                                          BranchOutcome::ProvedOptimal,
                                          dir < 0 ? down_gain : up_gain, -1.0, opts.int_tol))
                            ++diag.pseudocost_updates;
                    }
                }
                if (down_ok && up_ok) {
                    const f64 score = std::min(down_gain, up_gain) +
                                      0.1 * std::max(down_gain, up_gain);
                    if (score > strong_best) {
                        strong_best = score;
                        strong_var = j;
                    }
                    if (ci < planbb_ests.size()) {
                        planbb_ests[ci].down_gain = down_gain;
                        planbb_ests[ci].up_gain = up_gain;
                        planbb_ests[ci].ok =
                            std::isfinite(down_gain) && std::isfinite(up_gain);
                    }
                    if (std::isfinite(down_gain) && std::isfinite(up_gain) &&
                        ci < cand_feats.size()) {
                        const f64 prod = sb_product_score(down_gain, up_gain);
                        const int decision_id = static_cast<int>(diag.nodes);
                        if (collect_sparse_sb) {
                            sparse_sb_collector.add(cand_feats[ci], prod);
                            ++diag.sparse_sb_samples;
                        }
                        if (collect_sc_milp) {
                            sc_milp_collector.add(cand_feats[ci], prod,
                                                 decision_id);
                            ++diag.sc_milp_samples;
                        }
                        if (lifted_want && ci < lifted_feats.size()) {
                            lifted_state.observe(lifted_feats[ci], prod);
                            ++diag.lifted_samples;
                            if (lifted_state.maybe_refit(opts.lifted))
                                ++diag.lifted_refits;
                        }
                        if (collect_planbb) {
                            const bool pruned =
                                !std::isfinite(down_gain) ||
                                !std::isfinite(up_gain) ||
                                down_gain >= 1e30 || up_gain >= 1e30;
                            planbb_collector.add(cand_feats[ci], planbb_graph,
                                                 down_gain, up_gain, pruned);
                            ++diag.planbb_samples;
                        }
                    } else if (collect_planbb && ci < cand_feats.size() &&
                               (down_ok || up_ok)) {
                        // Infeasible child → prune label for dynamics.
                        const f64 dg = down_ok && std::isfinite(down_gain)
                                           ? down_gain
                                           : 0.0;
                        const f64 ug =
                            up_ok && std::isfinite(up_gain) ? up_gain : 0.0;
                        planbb_collector.add(cand_feats[ci], planbb_graph, dg,
                                             ug, /*pruned=*/true);
                        ++diag.planbb_samples;
                    }
                }
            }
            node_lp.col_lo = node.col_lo;
            node_lp.col_hi = node.col_hi;
            strong_probes_ran = diag.strong_branch_solves > solves_before;

            if (planbb_want_est && (!learner_chose || planbb_paper)) {
                bool any_ok = false;
                for (const auto& e : planbb_ests)
                    if (e.ok) { any_ok = true; break; }
                if (any_ok) ++diag.planbb_lookaheads;
                // Lite waits for probes. Paper may refine root transitions with
                // probe gains after the model-based first pick.
                const bool do_pick = !learner_chose || (planbb_paper && any_ok);
                if (do_pick) {
                    std::uint64_t mcts_sims = 0;
                    const Index pb = pick_planbb_branch(
                        planbb_paper ? &planbb_model : nullptr, planbb_policy,
                        opts.planbb, candidates, cand_feats, &planbb_graph,
                        any_ok ? &planbb_ests : nullptr, &mcts_sims);
                    diag.planbb_mcts_sims += mcts_sims;
                    if (pb >= 0) {
                        br = pb;
                        if (!learner_chose) ++diag.planbb_picks;
                        learner_chose = true;
                        diag.last_branch_policy =
                            planbb_paper
                                ? ((mcts_sims > 0) ? "planbb-paper-mcts"
                                                   : "planbb-paper")
                                : ((mcts_sims > 0) ? "planbb-mcts" : "planbb");
                    } else if (!learner_chose && try_sparse_sb_pick()) {
                        learner_chose = true;
                    } else if (!learner_chose) {
                        ++diag.planbb_fallbacks;
                        if (strong_var >= 0) br = strong_var;
                    }
                }
            } else if (!learner_chose && strong_var >= 0) {
                br = strong_var;
                diag.last_branch_policy =
                    batch_ranked ? "batch-sb" : "reliability";
            }
        }
        // Lite PlanB&B: if we deferred for probes but the probe block did not
        // run (unproved LP / empty basis), still pick from policy alone.
        if (latest_branch && branch_strat == BranchStrategy::PlanBb &&
            !learner_chose && !planbb_paper) {
            std::uint64_t mcts_sims = 0;
            const Index pb = pick_planbb_branch(
                nullptr, planbb_policy, opts.planbb, candidates, cand_feats,
                &planbb_graph, nullptr, &mcts_sims);
            diag.planbb_mcts_sims += mcts_sims;
            if (pb >= 0) {
                br = pb;
                learner_chose = true;
                ++diag.planbb_picks;
                diag.last_branch_policy =
                    (mcts_sims > 0) ? "planbb-mcts" : "planbb";
            } else {
                ++diag.planbb_fallbacks;
            }
        }

        diag.ms_branching += ms_since(t_branch);
        diag.bs_objective_samples = bstats.objective_samples;
        diag.bs_closure_samples = bstats.closure_samples;
        diag.bs_ignored_incomplete = bstats.ignored_incomplete;
        diag.bs_ignored_repeat = bstats.ignored_repeat;
        diag.bs_replaced = bstats.replaced;
        seg(11);  // branching
        if (br < 0) {
            // A fractional point with nothing to branch on used to drop its
            // whole subtree without a trace.
            fold_unsearched_region(node, "no branching candidate at a fractional node");
            continue;
        }
        if (tr_on && br >= 0) {
            ++diag.branching_traced;
            const auto j = sz(br);
            std::printf("[brtrace] n=%llu parent=%llu depth=%d dom=%016llx gen=%llu lp=%.9g "
                        "var=%d rank=%d/%d rel=%d/%d probed=%d stop=%s "
                        "pc_d=%.4g/%u pc_u=%.4g/%u closed_d=%u/%u closed_u=%u/%u "
                        "inf_d=%.3g inf_u=%.3g x=%.6g box=[%g,%g] skip=%s\n",
                        static_cast<unsigned long long>(diag.nodes),
                        static_cast<unsigned long long>(node.parent_serial), node.depth,
                        static_cast<unsigned long long>(tr_domain_hash()),
                        static_cast<unsigned long long>(global_lp_generation), lp_obj,
                        static_cast<int>(br), tr_rank, tr_cands, tr_reliable_cands, tr_cands,
                        tr_probed, tr_stop,
                        bstats.pc_count[0][j] ? bstats.pc_sum[0][j] / bstats.pc_count[0][j] : 0.0,
                        bstats.pc_count[0][j],
                        bstats.pc_count[1][j] ? bstats.pc_sum[1][j] / bstats.pc_count[1][j] : 0.0,
                        bstats.pc_count[1][j],
                        bstats.closed[0][j], bstats.observed[0][j],
                        bstats.closed[1][j], bstats.observed[1][j],
                        std::isnan(bstats.inference_rate(br, -1)) ? -1.0 : bstats.inference_rate(br, -1),
                        std::isnan(bstats.inference_rate(br, +1)) ? -1.0 : bstats.inference_rate(br, +1),
                        lp_raw.x[j], node.col_lo[j], node.col_hi[j], tr_skip);
        }
        if (!learner_chose) {
            ++diag.rb_nodes;
            // Once per branched node, whichever brancher probed.
            if (strong_probes_ran || rb_probed) ++diag.rb_nodes_with_sb;
        }

        apply_root_rc();
        if (!intersect_root_box(node)) continue;
        // L2Sep mid-tree reconfigure of DynSepOptions (Latest only).
        if (l2sep_opts.enabled && l2sep_opts.mid_tree_every_nodes > 0 &&
            diag.nodes >= last_l2sep_node + l2sep_opts.mid_tree_every_nodes) {
            SeparatorState sep_mid;
            sep_mid.last_round = static_cast<int>(diag.cut_rounds);
            sep_mid.last_bound_before = diag.root_bound_before_cuts;
            sep_mid.last_bound_after = diag.root_bound_after_cuts;
            configure_dynsep_from_l2sep(l2sep_opts, l2sep_model, global_lp,
                                        sep_mid, &lp_raw.x, dynsep_opts,
                                        l2sep_diag, /*mid_tree=*/true);
            if (cut_cfg.mir_cuts) dynsep_opts.allow_mir = true;
            if (cut_cfg.lifted_cover_cuts) dynsep_opts.allow_cover = true;
            if (cut_cfg.clique_cuts) dynsep_opts.allow_clique = true;
            if (cut_cfg.zerohalf_cuts) dynsep_opts.allow_zerohalf = true;
            if (cut_cfg.flow_cover_cuts) dynsep_opts.allow_flowcover = true;
            // Re-assert the structural light-arm clamp after the mid-tree
            // L2Sep re-apply (it would otherwise re-enable MIR/cover).
            if (milp_policy_is_latest(opts.policy) && dynsep_opts.enabled &&
                !cut_cfg.auto_cuts && problem.n_cols() <= 400 &&
                opts.time_limit_s > 0.0 && opts.time_limit_s <= 60.0) {
                dynsep_opts.max_depth_optional = 0;
                dynsep_opts.max_optional_arms = 1;
                if (!cut_cfg.mir_cuts) dynsep_opts.allow_mir = false;
                if (!cut_cfg.lifted_cover_cuts) dynsep_opts.allow_cover = false;
            }
            dynsep.update_options(dynsep_opts);
            last_l2sep_node = diag.nodes;
            diag.l2sep = l2sep_diag;
        }
        // Tree / local separation (WP-C + DynSep). Generators scheduled by
        // DynSep-v1; cuts from a node tableau with local bounds are LOCAL -
        // tagged, inherited by children, never visible to siblings.
        std::vector<ManagedCut> new_locals;
        bool node_closed_by_cuts = false;
        bool node_closed_infeasible = false;   // closed with no bound to keep
        std::optional<engines::SimplexBasis> child_basis_override;
        const std::uint64_t since_dual =
            (diag.nodes > last_dual_node) ? (diag.nodes - last_dual_node) : 0;
        // Tree separation must pay for itself. Its local cuts are still not
        // inserted into node LPs, so the only return is promotion of
        // globally valid cuts into global_lp; measured on easy60, air05 spent
        // 17.4 s of 60 s here with no promotion at all, neos-860300 16.5 s,
        // n2seq36q 14.1 s, decomp2 13.3 s. It may take 5% of the elapsed
        // time, 20% once it has promoted something.
        // The allowance is 5% of the elapsed time and grows only with what
        // separation has actually paid for: nodes it closed, proved
        // infeasible, gave an incumbent, or raised the bound by a meaningful
        // amount, per node it separated at (never with the mere count of
        // promoted cuts, which measured nothing on decomp2 or piperout-08).
        const f64 tree_credit_rate = diag.tree_cut_nodes > 0
            ? std::min(1.0, tree_sep_credit / static_cast<f64>(diag.tree_cut_nodes))
            : 0.0;
        const f64 tree_sep_frac = 0.05 + 0.15 * tree_credit_rate;
        diag.tree_sep_credit = tree_sep_credit;
        const f64 tree_sep_room_ms = tree_sep_frac * ms_since(t0) + 50.0 - diag.tree_sep_ms;
        const bool tree_sep_affordable = tree_sep_room_ms >= 0.0;
        if (!tree_sep_affordable && tree_cut_opts.enabled && node_lp_proved)
            ++diag.tree_sep_skipped_budget;
        if (tree_cut_opts.enabled && node_lp_proved &&
            tree_sep_affordable &&
            !node_basis.basic.empty() &&
            !in_proof_mode() &&
            should_separate_at_node(node.depth, since_dual, tree_cut_opts)) {
            ++diag.tree_cut_nodes;
            const auto t_tree_sep = Clock::now();
            struct TreeSepTick {
                double& sink;
                Clock::time_point t;
                ~TreeSepTick() { sink += ms_since(t); }
            } tree_sep_tick{diag.tree_sep_ms, t_tree_sep};
            DynSepRoundInput dsin;
            dsin.round = static_cast<int>(diag.cut_rounds);
            dsin.depth = node.depth;
            dsin.at_root = false;
            dsin.last_bound_gain_rel = dynsep_last_gain;
            dsin.force_mir = cut_cfg.mir_cuts;
            dsin.force_cover = cut_cfg.lifted_cover_cuts;
            dsin.force_clique = cut_cfg.clique_cuts;
            dsin.force_ib = opts.implied_bound_cuts;
            dsin.force_zerohalf = cut_cfg.zerohalf_cuts;
            dsin.force_flowcover = cut_cfg.flow_cover_cuts;
            dsin.have_conflict_graph = !conflict_graph.empty();
            dsin.have_basis = true;
            dsin.gap_rel = diag.gap_rel;
            const DynSepDecision dsd = dynsep.decide(dsin);

            CutDiagnostics tdiag;
            std::vector<CutRow> tcands;
            if (dsd.run_gmi) {
                CutOptions gmi_opts = cut_cfg.cut;
                gmi_opts.max_cuts_per_round =
                    std::min(dsd.budget_gmi, tree_cut_opts.max_cuts_per_node * 4);
                gmi_opts.time_limit_s = std::max(0.02, std::min(2.0, tree_sep_room_ms / 2000.0));
                tcands = separate_gomory_mi(node_lp, lp_raw.x, node_basis,
                                            gmi_opts, tdiag);
            }
            if (dsd.run_zerohalf) {
                ZeroHalfOptions zh_opts = dynsep_opts.zerohalf;
                zh_opts.max_cuts = dsd.budget_zerohalf;
                ZeroHalfDiagnostics zhd;
                auto zh = separate_zerohalf(node_lp, lp_raw.x, node.col_lo,
                                            node.col_hi, zh_opts, zhd);
                diag.zerohalf_candidates += zh.size();
                tcands.insert(tcands.end(),
                              std::make_move_iterator(zh.begin()),
                              std::make_move_iterator(zh.end()));
            }
            if (dsd.run_flowcover && opts.flow_cover_cuts) {
                FlowCoverOptions fc_opts = dynsep_opts.flowcover;
                fc_opts.max_cuts = dsd.budget_flowcover;
                FlowCoverDiagnostics fcd;
                // Same gate as the root site: flow cover emits invalid cuts.
                auto fc = separate_flow_covers(node_lp, lp_raw.x, node.col_lo,
                                               node.col_hi, fc_opts, fcd);
                diag.flowcover_candidates += fc.size();
                tcands.insert(tcands.end(),
                              std::make_move_iterator(fc.begin()),
                              std::make_move_iterator(fc.end()));
            }
            if (dsd.run_cover) {
                CoverOptions cov_opts = cut_cfg.cover;
                cov_opts.max_cuts = dsd.budget_cover;
                if (milp_policy_is_latest(opts.policy))
                    cov_opts.pc_lift_hooks = true;
                CoverDiagnostics cd;
                auto cov = separate_lifted_covers(node_lp, lp_raw.x, node.col_lo,
                                                  node.col_hi, cov_opts, cd);
                diag.cover.pc_sequence_independent += cd.pc_sequence_independent;
                diag.cover.pc_fallback_sequential += cd.pc_fallback_sequential;
                diag.cover.lifted_coefficients += cd.lifted_coefficients;
                tcands.insert(tcands.end(),
                              std::make_move_iterator(cov.begin()),
                              std::make_move_iterator(cov.end()));
            }
            if (dsd.run_mir) {
                MirOptions mir_opts = cut_cfg.mir;
                mir_opts.max_cuts = dsd.budget_mir;
                mir_opts.time_limit_s = std::max(0.02, std::min(2.0, tree_sep_room_ms / 2000.0));
                MirDiagnostics md;
                auto mc = separate_mir(node_lp, lp_raw.x, node.col_lo,
                                       node.col_hi, mir_opts, md,
                                       &root_lo, &root_hi, &conflict_graph);
                tcands.insert(tcands.end(),
                              std::make_move_iterator(mc.begin()),
                              std::make_move_iterator(mc.end()));
            }
            // Once per node, not per cut: does this node still sit exactly at
            // the root bounds? Only then can a cut derived here be global.
            const bool node_at_root_bounds =
                node.col_lo == root_lo && node.col_hi == root_hi;
            diag.tree_cuts_generated += tcands.size();
            if (cut_cfg.auto_cuts && milp_policy_is_latest(cut_cfg.policy) &&
                !tcands.empty()) {
                const std::size_t before_filter = tcands.size();
                tcands = filter_cut_candidates_for_round(
                    std::move(tcands), node_lp, lp_raw.x, cut_cfg.cut);
                diag.tree_cuts_prefilter_rejected +=
                    before_filter - tcands.size();
            }

            // GCS: persist violation stats across nodes for stored cuts.
            if (tree_cut_opts.gcs_enabled && tree_cut_opts.gcs_aggressive_observe)
                gcs_pool.touch_point(lp_raw.x, cut_cfg.cut.violation_min,
                                     node.depth, diag.gap_rel, node.bound);

            CutFeatureContext cctx{&node_lp, &lp_raw.x};
            Index n_int = 0, n_frac = 0;
            f64 mean_frac = 0.0;
            for (Index j = 0; j < node_lp.n_cols(); ++j) {
                if (node_lp.is_integer.empty() || !node_lp.is_integer[sz(j)])
                    continue;
                ++n_int;
                const f64 xvj = lp_raw.x[sz(j)];
                const f64 fj = xvj - std::floor(xvj);
                const f64 frac = std::min(fj, 1.0 - fj);
                if (frac > opts.int_tol) {
                    mean_frac += frac;
                    ++n_frac;
                }
            }
            if (n_frac > 0) mean_frac /= static_cast<f64>(n_frac);
            fill_hgtsm_lp_state(node_lp.n_cols(), n_int, n_frac, mean_frac,
                                diag.gap_rel, node.depth, tcands.size(),
                                dynsep_last_gain, hgtsm_lp_state);

            std::vector<f64> graph_scores;
            f64 graph_ratio = 1.0;
            if (hgtsm_opts.enabled && hgtsm_use_graph && !tcands.empty()) {
                auto g = build_tripartite_snapshot(node_lp, node.col_lo,
                                                   node.col_hi, tcands,
                                                   &lp_raw.x);
                hgtsm_score_sequence(hgtsm_model, g, graph_scores,
                                     &graph_ratio);
                hgtsm_diag.scored_cuts +=
                    static_cast<std::uint64_t>(tcands.size());
                ++hgtsm_diag.graph_selects;
                hgtsm_diag.used_graph = true;
                if (hgtsm_opts.collect_labels) {
                    std::vector<f64> labs = graph_scores;
                    hgtsm_collector.add_round(std::move(g), std::move(labs));
                }
            }

            std::vector<std::pair<f64, std::size_t>> ranked;
            ranked.reserve(tcands.size());
            // Does the reference point lie in THIS node's subtree? Only then
            // does "the cut excludes the reference" say anything about the
            // cut: a subtree-local cut is entitled to exclude a point that
            // lives outside its subtree. Both halves must hold -- the node's
            // BOUNDS and its LOCAL ROWS -- which is the same pair the scope
            // test is built from.
            bool ref_in_node = false;
            if (opts.cut_reference_point != nullptr) {
                const auto& rp = *opts.cut_reference_point;
                ref_in_node = static_cast<Index>(rp.size()) == node_lp.n_cols();
                for (Index j = 0; ref_in_node && j < node_lp.n_cols(); ++j) {
                    if (rp[sz(j)] < node.col_lo[sz(j)] - opts.primal_feas_tol ||
                        rp[sz(j)] > node.col_hi[sz(j)] + opts.primal_feas_tol) {
                        ref_in_node = false;
                        if (opts.cut_reference_debug)
                            std::fprintf(stderr,
                                "REF OUTSIDE NODE  sub_mip_depth=%d depth=%d col=%d ref=%.9g "
                                "not in [%.9g, %.9g]  root=[%.9g, %.9g]\n",
                                opts.sub_mip_depth, node.depth, static_cast<int>(j), rp[sz(j)],
                                node.col_lo[sz(j)], node.col_hi[sz(j)],
                                root_lo[sz(j)], root_hi[sz(j)]);
                    }
                }
                for (std::size_t q = 0; ref_in_node &&
                                        q < node.active_local.size(); ++q) {
                    const CutRow& lr = node.active_local[q].row;
                    if (!cut_admits_point(lr.cols, lr.vals, lr.row_lo, lr.row_hi,
                                          rp, opts.primal_feas_tol))
                        ref_in_node = false;
                }
            }

            // One source of truth for "may this cut leave its subtree",
            // recorded per candidate so the selection loop below labels and
            // stores cuts by the decision the gates actually made.
            std::vector<char> cand_global(tcands.size(), 0);
            for (std::size_t i = 0; i < tcands.size(); ++i) {
                CutFeatureVec cf{};
                if (!fill_cut_features(cctx, tcands[i], cf)) continue;
                f64 score = cf[0];
                if (hgtsm_opts.enabled) {
                    if (hgtsm_use_graph && i < graph_scores.size())
                        score = graph_scores[i];
                    else {
                        score = hgtsm_score(hgtsm_model, cf, hgtsm_lp_state);
                        ++hgtsm_diag.scored_cuts;
                    }
                    if (hgtsm_opts.collect_labels)
                        hgtsm_collector.add(cf, hgtsm_lp_state,
                                           /*label=*/cf[0] + dynsep_last_gain);
                }
                ranked.emplace_back(score, i);
                // Global validity is a property of DERIVATION, not of the
                // separator's name.
                //
                // These separators substitute variables onto a bound --
                // build_base_from() does x_j = bound +- s -- and at a node the
                // bounds handed in are node.col_lo/node.col_hi, i.e. TIGHTENED
                // BY BRANCHING. A cut derived that way holds only inside that
                // subtree. The old test asked only whether the name began
                // "MIR_"/"ZH_"/"FC_"/"COV_", so a locally-derived cut was
                // promoted into the pool and later reinjected into global_lp,
                // where it can cut off the true optimum. That is a false
                // Optimal (blend2: 8.1039210 against a true 7.5989850) or a
                // false Infeasible (enigma), and it needs only a couple of
                // nodes to appear.
                //
                // A node cut is globally valid only if every column it touches
                // still sits at its ROOT bounds, so no branching decision
                // entered the derivation. That keeps the common case -- a cut
                // whose support was never branched on -- while refusing the
                // unsound one.
                // Classification goes through ONE decision point so the
                // pool, the id and the promote/reinject paths cannot disagree.
                // See CutScope in tree_cuts.hpp for what each field means and
                // why the ZH_/MIR_/FC_ families are not on the trusted list.
                CutScopeConditions scope;
                scope.bounds_are_root = node_at_root_bounds;
                scope.rows_are_global = node.active_local.empty();
                scope.derivation_trusted =
                    cut_family_derivation_trusted(tcands[i].name);
                // A globally valid cut must admit the incumbent. Catches a bad
                // cut from ANY separator without knowing which is at fault.
                if (have_incumbent && !best_x.empty() &&
                    !cut_admits_point(tcands[i].cols, tcands[i].vals,
                                      tcands[i].row_lo, tcands[i].row_hi,
                                      best_x, opts.primal_feas_tol)) {
                    scope.admits_incumbent = false;
                    if (scope.bounds_are_root && scope.rows_are_global &&
                        scope.derivation_trusted)
                        ++diag.cuts_rejected_by_incumbent;
                }
                bool globally_valid = cut_may_leave_subtree(scope);
                if (!globally_valid && scope.derivation_trusted &&
                    scope.admits_incumbent) {
                    ++diag.local_cuts_kept_local;
                    if (scope.bounds_are_root && !scope.rows_are_global)
                        ++diag.cuts_kept_local_unknown_rows;
                }
                // Cut-validity diagnostic, restored and split into the three
                // populations of BabDiagnostics::node_cuts_invalid_*:
                // GENERATED (a separator emitted it, before any gate),
                // REJECTED (a gate then refused it -- contained), and the
                // count that actually matters, an invalid cut that a gate let
                // through. Judged BEFORE the gates, so "nothing invalid was
                // derived" and "something invalid was derived and contained"
                // can no longer be read as the same number.
                if (opts.cut_reference_point != nullptr) {
                    if (!ref_in_node) {
                        // A subtree-local cut is entitled to exclude a point
                        // outside its own subtree, so the check abstains here
                        // rather than crying wolf.
                        ++diag.node_cuts_ref_outside_node;
                    } else if (!cut_admits_point(
                                   tcands[i].cols, tcands[i].vals,
                                   tcands[i].row_lo, tcands[i].row_hi,
                                   *opts.cut_reference_point,
                                   opts.primal_feas_tol)) {
                        ++diag.node_cuts_invalid_generated;
                        if (!globally_valid) {
                            ++diag.node_cuts_invalid_rejected;
                        } else {
                            ++diag.invalid_cuts_detected;
                            f64 act = 0.0;
                            for (std::size_t k = 0; k < tcands[i].cols.size(); ++k)
                                act += tcands[i].vals[k] *
                                       (*opts.cut_reference_point)
                                           [sz(tcands[i].cols[k])];
                            std::fprintf(stderr,
                                "INVALID NODE CUT  %-18s activity %.9g not in "
                                "[%.9g, %.9g]  (nnz %zu)  depth=%d "
                                "local_rows=%zu scope=%c%c%c%c id=%s\n",
                                tcands[i].name.c_str(), act, tcands[i].row_lo,
                                tcands[i].row_hi, tcands[i].cols.size(),
                                node.depth, node.active_local.size(),
                                scope.bounds_are_root ? 'B' : '-',
                                scope.rows_are_global ? 'R' : '-',
                                scope.derivation_trusted ? 'D' : '-',
                                scope.admits_incumbent ? 'I' : '-',
                                cut_content_id(tcands[i]).c_str());
                        }
                    }
                }
                cand_global[i] = globally_valid ? 1 : 0;
                if (opts.verbose) {
                    // Stable content id at CLASSIFICATION, with the reason a
                    // cut was refused. The same id appears at generation (the
                    // pool observe below keys on it) and at actual global
                    // insertion, so a cut can be followed end to end.
                    std::printf("  [cut-class] %-14s id=%s scope=%c%c%c%c -> %s\n",
                                tcands[i].name.c_str(),
                                cut_content_id(tcands[i]).c_str(),
                                scope.bounds_are_root ? 'B' : '-',
                                scope.rows_are_global ? 'R' : '-',
                                scope.derivation_trusted ? 'D' : '-',
                                scope.admits_incumbent ? 'I' : '-',
                                globally_valid ? "GLOBAL" : "local");
                }
                // Content-stable id so the same cut accumulates multi-node stats.
                gcs_pool.observe(tcands[i], cf, cf[0], cf[4] > 0.0,
                                 globally_valid, cut_content_id(tcands[i]),
                                 node.depth, diag.gap_rel, node.bound);
            }
            if (hgtsm_opts.enabled) {
                ++hgtsm_diag.selects;
                hgtsm_diag.model_loaded = hgtsm_model.loaded;
                hgtsm_diag.used_builtin = !hgtsm_model.loaded;
            }
            // Apply higher-level ratio as a soft keep-cap on ranked scores.
            if (hgtsm_use_graph && graph_ratio < 0.999 && !ranked.empty()) {
                const int keep = std::max(
                    1, static_cast<int>(std::ceil(
                           graph_ratio * static_cast<f64>(ranked.size()))));
                std::sort(ranked.begin(), ranked.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first != b.first) return a.first > b.first;
                              return a.second < b.second;
                          });
                if (static_cast<int>(ranked.size()) > keep)
                    ranked.resize(static_cast<std::size_t>(keep));
            } else {
            std::sort(ranked.begin(), ranked.end(),
                      [](const auto& a, const auto& b) {
                          if (a.first != b.first) return a.first > b.first;
                          return a.second < b.second;
                      });
            }
            const int take = std::min(tree_cut_opts.max_cuts_per_node,
                                      static_cast<int>(ranked.size()));
            std::vector<std::string> sel_names;
            f64 eff_sum = 0.0;
            for (int t = 0; t < take; ++t) {
                ManagedCut mc;
                mc.row = tcands[ranked[static_cast<std::size_t>(t)].second];
                // Carry the decision the promotion gates already made for
                // THIS candidate. It used to be recomputed here as
                // "!tableau_gmi" -- global because the cut is merely not a
                // GMI -- which ignored bounds, rows and the incumbent check,
                // and disagreed with what was stored in the GCS pool. It gated
                // nothing (only the id prefix reads it), so it was not a
                // soundness hole, but it labelled subtree-local cuts "g-" and
                // any audit of those ids was reading fiction.
                const std::size_t ci = ranked[static_cast<std::size_t>(t)].second;
                mc.global = ci < cand_global.size() && cand_global[ci] != 0;
                mc.created_depth = node.depth;
                mc.created_node = diag.nodes;
                mc.id = (mc.global ? "g-" : "local-") +
                        std::to_string(++local_cut_seq);
                sel_names.push_back(mc.row.name);
                eff_sum += ranked[static_cast<std::size_t>(t)].first;
                // Collected for accounting only. new_locals is not written into
                // node.active_local: applying it would change the relaxation
                // without a test that the rows are valid for this node. Global
                // cuts still reach the model through the GCS promote path.
                new_locals.push_back(std::move(mc));
            }
            if (take > 0) {
                dynsep.observe_selected_names(
                    sel_names, eff_sum / static_cast<f64>(take));
                dynsep.collect_round_labels(dsd);
            }
            // Selected, not inserted: new_locals never reaches a node LP
            // (tree_local_cuts_inserted stays unchanged here).
            diag.tree_local_cuts_selected +=
                static_cast<std::uint64_t>(new_locals.size());
            // P9: put the selected cuts into THIS node's LP for one warm
            // re-solve. Families whose derivation over the node box is
            // covered by the oracle tests only; flow cover stays out.
            if (tree_cut_opts.resolve_with_local && node_lp_proved &&
                !new_locals.empty() && node_lp.n_rows() > 0 &&
                static_cast<Index>(node_basis.basic.size()) == node_lp.n_rows()) {
                std::vector<CutRow> lrows;
                for (const ManagedCut& mc : new_locals) {
                    const std::string& nm = mc.row.name;
                    if (nm.rfind("GMI_", 0) == 0 || nm.rfind("MIR_", 0) == 0 ||
                        nm.rfind("COV", 0) == 0)
                        lrows.push_back(mc.row);
                }
                if (!lrows.empty() && !timed_out()) {
                    const auto t_resolve = Clock::now();
                    ++diag.node_cut_resolves;
                    const model::LpProblem cut_node_lp =
                        apply_cuts(node_lp, lrows, cut_cfg.cut);
                    engines::SimplexBasis ext = node_basis;
                    const Index ns_r = node_lp.n_cols();
                    const Index m_old_r = node_lp.n_rows();
                    ext.status.resize(sz(ns_r + cut_node_lp.n_rows()),
                                      engines::NonbasicStatus::Basic);
                    for (Index r = m_old_r; r < cut_node_lp.n_rows(); ++r)
                        ext.basic.push_back(ns_r + r);
                    engines::SimplexOptions ro = lp_opts;
                    ro.presolve = false;
                    ro.method = engines::SimplexMethod::Dual;
                    ro.warm_dse_reset = true;
                    ro.max_iterations = static_cast<std::uint64_t>(
                        std::max(200.0, tree_cut_opts.resolve_iter_per_row *
                                            static_cast<f64>(cut_node_lp.n_rows())));
                    const f64 left_s = opts.time_limit_s > 0.0
                        ? std::max(0.0, opts.time_limit_s - ms_since(t0) / 1000.0)
                        : 1e9;
                    ro.time_limit_s = std::max(std::numeric_limits<double>::min(),
                                                std::min(left_s, 5.0));
                    if (cutoff_known())
                        ro.objective_limit =
                            node_cutoff(known_cutoff_min(), obj_granularity,
                                        offset_min, opts.gap_tol, opts.abs_gap_tol);
                    engines::SimplexDiagnostics rsd;
                    engines::SimplexBasis rbasis;
                    const auto t_resolve_lp = Clock::now();
                    const auto rraw = engines::solve_dual_simplex(
                        cut_node_lp, ro, rsd, &rbasis, &ext);
                    branching_lp_work.charge(rsd);
                    branching_lp_work.ms += ms_since(t_resolve_lp);
                    ++diag.lp_solves;
                    diag.lp_iterations += rsd.iterations;
                    diag.node_lp_simplex_ms += rsd.total_ms;
                    // Every outcome is judged as an ordinary node LP would be.
                    if (node_lp_infeasibility_proved(cut_node_lp, rraw, opts.primal_feas_tol,
                                                     root_relaxation_bounded)) {
                        // The cuts hold over this node's box: infeasible with
                        // them means infeasible here. Nothing to floor.
                        ++diag.node_cut_infeasible;
                        node_closed_by_cuts = true;
                        node_closed_infeasible = true;
                        tree_sep_credit += 1.0;
                    } else if (relaxation_proved(rraw, rsd, ro)) {
                        ++diag.node_cut_resolves_proved;
                        // An integral optimum of the cut LP is a feasible point
                        // of the model (the cuts only remove infeasible ones):
                        // take it, and the node is done.
                        bool integral_opt = static_cast<Index>(rraw.x.size()) == n;
                        for (Index j = 0; integral_opt && j < n; ++j)
                            if (!mip.is_integer.empty() && mip.is_integer[sz(j)] &&
                                !is_integral(rraw.x[sz(j)], opts.int_tol))
                                integral_opt = false;
                        if (integral_opt) {
                            f64 iobj = std::numeric_limits<f64>::quiet_NaN();
                            if (accept_incumbent(rraw.x, true, iobj)) {
                                solution_pool.add(best_x, best_incumbent, problem.maximize);
                                ++diag.node_cut_incumbents;
                                tree_sep_credit += 1.0;
                            }
                            if (std::isfinite(iobj)) {
                                node_closed_by_cuts = true;
                                node_closed_infeasible = true;   // subtree settled, no floor
                            }
                        }
                        const f64 nb = node_lp_bound_min(rraw, sense);
                        if (std::isfinite(nb) && nb > node.bound) {
                            if (!std::isfinite(node.bound) ||
                                nb - node.bound > 1e-4 * (1.0 + std::fabs(nb)))
                                tree_sep_credit += 0.5;
                            node.bound = nb;
                            ++diag.node_cut_bound_raises;
                            if (node.depth == 0)
                                root_cert.raise(nb, BoundCertificate::Source::LpOptimal);
                        }
                        if (cutoff_known() &&
                            node.bound >= node_cutoff(known_cutoff_min(),
                                                      obj_granularity, offset_min,
                                                      opts.gap_tol, opts.abs_gap_tol))
                        {
                            node_closed_by_cuts = true;
                            tree_sep_credit += 1.0;
                        }
                        // The cuts are now physically in a proved node LP:
                        // descendants inherit them, warm-started from this
                        // augmented basis. Bounded so a lineage cannot grow
                        // without limit.
                        if (!node_closed_by_cuts &&
                            static_cast<int>(node.active_local.size() + lrows.size()) <=
                                tree_cut_opts.max_local_rows &&
                            rbasis.basic.size() ==
                                static_cast<std::size_t>(cut_node_lp.n_rows())) {
                            for (const ManagedCut& mc : new_locals) {
                                const std::string& nm = mc.row.name;
                                if (nm.rfind("GMI_", 0) == 0 || nm.rfind("MIR_", 0) == 0 ||
                                    nm.rfind("COV", 0) == 0) {
                                    ManagedCut local = mc;
                                    local.global = false;   // scope: this subtree
                                    node.active_local.push_back(std::move(local));
                                }
                            }
                            child_basis_override = std::move(rbasis);
                            diag.node_cuts_inherited +=
                                static_cast<std::uint64_t>(lrows.size());
                        }
                    }
                    diag.node_cut_resolve_ms += ms_since(t_resolve);
                }
            }
            diag.dynsep = dynsep.diagnostics();
            diag.hgtsm = hgtsm_diag;

            if (tree_cut_opts.gcs_enabled && tree_cut_opts.gcs_promote_max > 0) {
                auto promo = gcs_pool.select_global(
                    tree_cut_opts.gcs_promote_max, tree_cut_opts.gcs_min_score,
                    /*skip_promoted=*/true);
                if (!promo.empty()) {
                    // Defensive: drop any row that is not globally_valid in pool.
                    std::vector<CutRow> safe;
                    safe.reserve(promo.size());
                    for (const CutRow& r : promo) {
                        const std::string id = cut_content_id(r);
                        bool ok = false;
                        for (const GcsCandidate& c : gcs_pool.cands) {
                            if (c.id == id && c.globally_valid) {
                                ok = true;
                                break;
                            }
                        }
                        if (ok) safe.push_back(r);
                    }
                    if (!safe.empty()) {
                        for (const auto& gc : safe) {
                            // Validity at the moment of INSERTION, not at
                            // classification. This is the count that can
                            // produce a wrong answer, so it is measured where
                            // the row actually enters the relaxation.
                            if (opts.cut_reference_point != nullptr &&
                                !cut_admits_point(gc.cols, gc.vals, gc.row_lo,
                                                  gc.row_hi,
                                                  *opts.cut_reference_point,
                                                  opts.primal_feas_tol)) {
                                ++diag.node_cuts_invalid_inserted;
                                std::fprintf(stderr,
                                    "INVALID CUT INSERTED  promote %-14s id=%s"
                                    " -> global_lp\n", gc.name.c_str(),
                                    cut_content_id(gc).c_str());
                            }
                            if (opts.verbose)
                                std::printf("  [cut-insert] promote %-12s id=%s"
                                            " -> global_lp\n",
                                            gc.name.c_str(),
                                            cut_content_id(gc).c_str());
                        }
                        apply_cuts_inplace(global_lp, safe, cut_cfg.cut);
                        ++global_lp_generation;
                        gcs_pool.mark_promoted(safe);
                        diag.gcs_promoted +=
                            static_cast<std::uint64_t>(safe.size());
                        gcs_pool.diag.promotes +=
                            static_cast<std::uint64_t>(safe.size());
                        node.has_basis = false;
                    }
                }
            }
        }
        if (node_closed_by_cuts) {
            ++diag.node_cut_prunes;
            const f64 inc_min = sense * best_incumbent;
            if (!node_closed_infeasible && node.bound < inc_min) {
                ++diag.gap_prunes;
                pruned_floor = std::min(pruned_floor, node.bound);
            }
            diag.ms_exit_b += ms_since(t_loop_iter); ++diag.n_exit_b;
            continue;
        }
        // GCS periodic re-injection of top multi-node cuts into global_lp.
        if (tree_cut_opts.gcs_enabled &&
            tree_cut_opts.gcs_reinject_every_nodes > 0 &&
            diag.nodes >=
                last_gcs_reinject_node + tree_cut_opts.gcs_reinject_every_nodes) {
            auto reinj = gcs_pool.select_global(tree_cut_opts.gcs_reinject_max,
                                                tree_cut_opts.gcs_min_score,
                                                /*skip_promoted=*/true);
            if (!reinj.empty()) {
                for (const auto& gc : reinj) {
                    if (opts.cut_reference_point != nullptr &&
                        !cut_admits_point(gc.cols, gc.vals, gc.row_lo,
                                          gc.row_hi,
                                          *opts.cut_reference_point,
                                          opts.primal_feas_tol)) {
                        ++diag.node_cuts_invalid_inserted;
                        std::fprintf(stderr,
                            "INVALID CUT INSERTED  reinject %-14s id=%s"
                            " -> global_lp\n", gc.name.c_str(),
                            cut_content_id(gc).c_str());
                    }
                    if (opts.verbose)
                        std::printf("  [cut-insert] reinject %-12s id=%s"
                                    " -> global_lp\n",
                                    gc.name.c_str(),
                                    cut_content_id(gc).c_str());
                }
                apply_cuts_inplace(global_lp, reinj, cut_cfg.cut);
                ++global_lp_generation;
                gcs_pool.mark_promoted(reinj);
                diag.gcs_reinjected +=
                    static_cast<std::uint64_t>(reinj.size());
                diag.gcs_promoted +=
                    static_cast<std::uint64_t>(reinj.size());
                gcs_pool.diag.reinjects +=
                    static_cast<std::uint64_t>(reinj.size());
                node.has_basis = false;
            }
            last_gcs_reinject_node = diag.nodes;
        }
        diag.gcs_pool_size = gcs_pool.cands.size();
        diag.gcs = gcs_pool.diag;

        const f64 xv = lp_raw.x[sz(br)];
        const f64 floor_v = std::floor(xv);
        const f64 ceil_v = std::ceil(xv);

        // Keep this node's final factor and weights for its children (shared
        // by both). Only while the session still describes this node's final
        // basis: a cutoff re-solve or fallback below the session replaced it.
        std::shared_ptr<CheckpointHolder> child_checkpoint;
        if (session_state_is_last_node && node_session &&
            node_session_generation == session_key) {
            const auto& ff = node_session->final_factor();
            if (ff.has_factor && ff.basis == node_basis.basic) {
                const auto t_ck = Clock::now();
                auto data = std::make_unique<LpCheckpoint>();
                data->factor = ff;
                if (node_session->final_weights_basis() == node_basis.basic) {
                    data->weights_basis = node_session->final_weights_basis();
                    data->weights = node_session->final_weights();
                }
                child_checkpoint =
                    checkpoint_cache.add(std::move(data), session_key,
                                         node.bound);
                diag.checkpoint_copy_ms += ms_since(t_ck);
            }
        }

        seg(12);  // strong-branch requeue, root rc, mid-tree separation, gcs
        const auto t_children = Clock::now();
        Node down = node;
        down.checkpoint = child_checkpoint;
        // The child's domain is this node's propagated domain plus the one
        // bound set below.
        down.dirty.assign(1, br);
        const f64 down_old_hi = down.col_hi[sz(br)];
        down.col_hi[sz(br)] = std::min(down.col_hi[sz(br)], floor_v);
        down.bound = node.bound;
        down.depth = node.depth + 1;
        // Always record Branch reasons for nogood / Mexi trails.
        down.prop_trail.push(br, BoundDir::Upper, floor_v, down_old_hi,
                             ReasonKind::Branch, -1, down.depth);
        // Local cuts are inherited (valid over this node's whole box, so over
        // the child's). Their rows sit after the global rows and the basis
        // handed down describes exactly that layout (tagged with the
        // generation it was solved under); a node that meets a different
        // global generation drops it at setup.
        down.basis = child_basis_override ? *child_basis_override : node_basis;
        down.basis_generation = solve_generation;
        down.has_basis = !down.basis.basic.empty();
        down.parent_branch_var = br;
        down.parent_branch_dir = -1;
        SOR_ROUTE_PATH(2, "bab", "branch", "branch");
        down.parent_bound = node_lp_proved && std::isfinite(lp_obj_min)
                                ? lp_obj_min : core::kNaN;
        down.parent_branch_distance = xv - floor_v;
        down.pc_consumed = false;
        down.pc_recorded_unit = core::kNaN;
        down.parent_serial = diag.nodes;

        Node up = node;
        up.checkpoint = child_checkpoint;
        up.dirty.assign(1, br);
        const f64 up_old_lo = up.col_lo[sz(br)];
        up.col_lo[sz(br)] = std::max(up.col_lo[sz(br)], ceil_v);
        up.bound = node.bound;
        up.depth = node.depth + 1;
        up.prop_trail.push(br, BoundDir::Lower, ceil_v, up_old_lo,
                           ReasonKind::Branch, -1, up.depth);
        up.basis = child_basis_override ? *child_basis_override : node_basis;
        up.basis_generation = solve_generation;
        up.has_basis = !up.basis.basic.empty();
        up.parent_branch_var = br;
        up.parent_branch_dir = +1;
        up.parent_bound = node_lp_proved && std::isfinite(lp_obj_min)
                              ? lp_obj_min : core::kNaN;
        up.parent_branch_distance = ceil_v - xv;
        up.pc_consumed = false;
        up.pc_recorded_unit = core::kNaN;
        up.parent_serial = diag.nodes;

        // Best-bound remains the proof ordering; the up-bias only picks
        // which child is the more promising integer direction on degenerate
        // zero-cost faces. Hybrid node selection (item 16) sends that
        // preferred child straight to the plunge stack (continuing a bounded
        // depth-first dive) and the other child to the best-bound queue as a
        // fallback -- both still get visited eventually either way.
        if (mip_up_bias.empty()) mip_up_bias = integer_up_bias_all(mip);
        const bool down_first = mip_up_bias[sz(br)] < 0.0;
        Node& preferred = down_first ? down : up;
        Node& fallback = down_first ? up : down;
        const bool preferred_range =
            preferred.col_lo[sz(br)] <= preferred.col_hi[sz(br)] + 1e-12;
        const bool fallback_range =
            fallback.col_lo[sz(br)] <= fallback.col_hi[sz(br)] + 1e-12;

        const bool plunge_this = hybrid_nodes &&
                                 node.plunge_len < opts.plunge_max_depth &&
                                 preferred_range;
        if (plunge_this) {
            preferred.plunge_len = node.plunge_len + 1;
            plunge_stack.push_back(std::move(preferred));
            if (fallback_range) {
                fallback.plunge_len = 0;
                open.push(std::move(fallback));
            }
        } else {
            if (preferred_range) {
                preferred.plunge_len = 0;
                open.push(std::move(preferred));
            }
            if (fallback_range) {
                fallback.plunge_len = 0;
                open.push(std::move(fallback));
            }
        }
        diag.ms_node_children += ms_since(t_children);
        seg(13);  // children
        ++diag.loop_iters_completed;
    }

    const bool tree_exhausted = !stopped_early && open.empty() && plunge_stack.empty() &&
                                deferred.empty();
    if (tree_exhausted && have_incumbent)
        reason = "tree exhausted";
    else if (tree_exhausted && !have_incumbent)
        reason = "tree exhausted with no integer feasible point";

    // Dual bound: for a complete tree with incumbent, dual = incumbent.
    // Otherwise, for minimize, take the min LP bound among remaining open nodes
    // (and consider proved if gap small).
    f64 dual_bound_min = std::numeric_limits<f64>::infinity();
    // Seeding with the incumbent is sound -- the explored part of the tree
    // cannot hold anything better than the best point found in it -- but ONLY
    // once every unexplored region is represented below. It is not, if a node
    // was popped and abandoned: that subtree is in neither container, and the
    // seed then survives every std::min and reports itself as the global
    // bound. With the containers empty, which the time limit expiring inside
    // the root node's own LP produces exactly, the gap comes out 0 and
    // gap_proved turns a timeout into ProvedGlobalEpsilon. Measured on `pg` at
    // a 60 s limit: incumbent 7250 certified optimal, against a published
    // optimum of -8674.34 and this solver's own feasible point at -8192.85.
    //
    // So the fix is not to drop the seed -- that would cost the legitimate
    // early exits the gap tolerance exists for -- but to fold in the one
    // region the drains cannot see.
    if (have_incumbent)
        dual_bound_min = std::min(dual_bound_min, sense * best_incumbent);
    dual_bound_min = std::min(dual_bound_min, abandoned_bound);
    dual_bound_min = std::min(dual_bound_min, pruned_floor);
    dual_bound_min = std::min(dual_bound_min, deferred_floor());
    // Drain open and the plunge stack for the actual global bound
    // (destructive OK at end). Nodes retain their parent's proved LP bound
    // until they are processed, and every node lives in exactly one of these
    // two containers, so both must be drained or the bound would be
    // unsoundly optimistic.
    while (!open.empty()) {
        dual_bound_min = std::min(dual_bound_min, open.top().bound);
        open.pop();
    }
    while (!plunge_stack.empty()) {
        dual_bound_min = std::min(dual_bound_min, plunge_stack.back().bound);
        plunge_stack.pop_back();
    }
    if (!have_incumbent && !std::isfinite(dual_bound_min))
        dual_bound_min = std::numeric_limits<f64>::quiet_NaN();

    // Strengthen the working (minimisation) bound to the next attainable
    // objective value before converting back. Nothing strictly between
    // multiples of g is reachable by a feasible integer point, so this is
    // valid, and it is what turns "found the optimum but 0.17 short of proving
    // it" into a proof.
    if (std::isfinite(dual_bound_min) && obj_granularity > 0.0) {
        // dual_bound_min is the working (minimisation) bound, i.e.
        // sense*objective, so the offset travels with the same sign.
        const f64 tightened = tighten_bound_to_granularity(
            dual_bound_min, obj_granularity, sense * problem.obj_offset,
            opts.int_tol);
        if (tightened > dual_bound_min) {
            ++diag.granularity_tightenings;
            dual_bound_min = tightened;
        }
    }

    const f64 dual_orig = std::isfinite(dual_bound_min) ? sense * dual_bound_min
                                                        : core::kNaN;

    diag.incumbent = have_incumbent ? best_incumbent : core::kNaN;
    diag.dual_bound = dual_orig;
    diag.gap_pruned_floor = std::isfinite(pruned_floor) ? sense * pruned_floor : core::kNaN;
    if (have_incumbent && std::isfinite(dual_orig)) {
        diag.gap_rel = std::fabs(best_incumbent - dual_orig) /
                       (1.0 + std::fabs(best_incumbent));
    }
    diag.total_ms = ms_since(t0);
    diag.termination_reason = reason;
    diag.checkpoints_created = checkpoint_cache.created();
    diag.checkpoints_evicted = checkpoint_cache.evicted();
    diag.checkpoints_declined = checkpoint_cache.declined();
    diag.checkpoint_peak_bytes = checkpoint_cache.peak_bytes();

    // A node's .bound is only ever certified evidence -- a proved LP bound,
    // a safe Lagrangian bound, or the certificate it inherited -- never an
    // unproved LP objective. Every region of the root box is searched,
    // pruned against such a bound (pruned_floor keeps the ones below the
    // incumbent), represented in open/plunge, or folded into abandoned_bound
    // (an abandoned node, an integral point from an unproved LP, a node with
    // nothing to branch on). So dual_bound_min is a sound global lower bound
    // whether or not the tree was exhausted, and a closed gap IS a proof.
    //
    // An unproved node LP therefore no longer vetoes the proof by itself (the
    // old all_lp_proven flag): the only unsound use of such an LP -- closing
    // its subtree on an integral point -- now keeps the subtree by bound, and
    // stopped_early keeps the tree from counting as exhausted.
    const bool gap_proved = have_incumbent && std::isfinite(dual_orig) &&
        (diag.gap_rel <= opts.gap_tol ||
         std::fabs(best_incumbent - dual_orig) <= opts.abs_gap_tol);
    // Correctness gate: never claim Optimal if the dual bound crosses the
    // incumbent (would require an invalid cut / bound). Keep Feasible.
    bool dual_crosses_incumbent = false;
    if (have_incumbent && std::isfinite(dual_orig)) {
        const f64 dual_min = sense * dual_orig;
        const f64 inc_min = sense * best_incumbent;
        if (dual_min >
            inc_min + opts.gap_tol * (1.0 + std::fabs(inc_min))) {
            dual_crosses_incumbent = true;
            if (reason.find("dual crossed") == std::string::npos)
                reason += "; dual crossed incumbent (refused Optimal)";
        }
    }
    // Empty queues do not erase regions removed by a cutoff or an earlier
    // gap tolerance. Their retained floor still limits the final proof.
    // For an incumbent, only the final bound can establish the requested gap.
    diag.globally_proved = !dual_crosses_incumbent &&
                           (have_incumbent ? gap_proved : tree_exhausted);
    if (tree_exhausted && have_incumbent && !gap_proved)
        reason += "; retained pruned-region bound leaves the requested gap open";
    diag.termination_reason = reason;
    diag.root_certified_bound =
        std::isfinite(root_cert.value) ? sense * root_cert.value : core::kNaN;

    diag.lns.arms = alns.arms();
    diag.balans.arms = balans.arms();

    if (opts.sparse_sb_collect_out != nullptr &&
        !sparse_sb_collector.samples.empty()) {
        auto& dst = opts.sparse_sb_collect_out->samples;
        const auto cap = opts.sparse_sb_collect_out->max_samples;
        for (const auto& s : sparse_sb_collector.samples) {
            if (dst.size() >= cap) break;
            dst.push_back(s);
        }
    }
    if (opts.sc_milp_collect_out != nullptr &&
        !sc_milp_collector.samples.empty()) {
        auto& dst = opts.sc_milp_collect_out->samples;
        const auto cap = opts.sc_milp_collect_out->max_samples;
        for (const auto& s : sc_milp_collector.samples) {
            if (dst.size() >= cap) break;
            dst.push_back(s);
        }
    }
    if (opts.lifted_collect_out != nullptr &&
        !lifted_state.buffer.samples.empty()) {
        auto& dst = opts.lifted_collect_out->samples;
        const auto cap = opts.lifted_collect_out->max_samples;
        for (const auto& s : lifted_state.buffer.samples) {
            if (dst.size() >= cap) break;
            dst.push_back(s);
        }
    }
    if (opts.dynsep_collect_out != nullptr &&
        !dynsep.collector().samples.empty()) {
        auto& dst = opts.dynsep_collect_out->samples;
        const auto cap = opts.dynsep_collect_out->max_samples;
        for (const auto& s : dynsep.collector().samples) {
            if (dst.size() >= cap) break;
            dst.push_back(s);
        }
    }
    diag.dynsep_samples =
        static_cast<std::uint64_t>(dynsep.collector().samples.size());
    if (opts.planbb_collect_out != nullptr &&
        !planbb_collector.samples.empty()) {
        auto& dst = opts.planbb_collect_out->samples;
        const auto cap = opts.planbb_collect_out->max_samples;
        for (const auto& s : planbb_collector.samples) {
            if (dst.size() >= cap) break;
            dst.push_back(s);
        }
    }
    if (opts.hgtsm_collect_out != nullptr) {
        auto& dst = *opts.hgtsm_collect_out;
        for (const auto& s : hgtsm_collector.samples) {
            if (dst.samples.size() >= dst.max_samples) break;
            dst.samples.push_back(s);
        }
        for (auto& r : hgtsm_collector.rounds) {
            if (dst.rounds.size() >= dst.max_rounds) break;
            dst.rounds.push_back(std::move(r));
        }
    }
    if (opts.gcs_collect_out != nullptr) {
        auto& dst = opts.gcs_collect_out->samples;
        const auto cap = opts.gcs_collect_out->max_samples;
        for (const auto& s : gcs_collector.samples) {
            if (dst.size() >= cap) break;
            dst.push_back(s);
        }
    }

    raw.iterations = diag.nodes;
    raw.termination_reason = reason;
    raw.dual_bound = dual_orig;

    if (have_incumbent) {
        raw.x = std::move(best_x);
        raw.objective = best_incumbent;
        diag.final_primal_violation =
            milp_point_max_violation(problem, raw.x, opts.int_tol);
        if (diag.globally_proved) {
            raw.proposed_status = core::Status::Optimal;
            raw.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
        } else {
            raw.proposed_status = core::Status::Feasible;
            raw.proposed_level = core::ProofLevel::FeasibleWithGap;
        }
    } else if (tree_exhausted && !diag.used_foreign_cutoff) {
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::BoundOnly;
    } else if (tree_exhausted) {
        // Exhausted with no incumbent of its own, but every prune was measured
        // against a cutoff another arm published. That is NOT an infeasibility
        // proof -- it proves "nothing BETTER than the cutoff lives anywhere",
        // not "nothing lives anywhere". Claiming Infeasible here would be the
        // same defect as the implied-integrality false Infeasible fixed
        // 2026-09-19.
        //
        // It IS, however, a real and valuable proof for the portfolio: this
        // arm searched the whole space and found nothing beating the cutoff,
        // so whichever arm achieves that cutoff holds a global optimum. The
        // flag lets the driver claim it against the pool's point; alone, this
        // arm can only say it found nothing.
        diag.proved_no_better_than_cutoff = true;
        raw.proposed_status = core::Status::NoSolutionFound;
        raw.proposed_level = core::ProofLevel::None;
        raw.termination_reason =
            "tree exhausted against a shared cutoff; optimality of that cutoff "
            "is proved, but this arm holds no point achieving it";
    } else {
        raw.proposed_status = core::Status::Interrupted;
        raw.proposed_level = core::ProofLevel::None;
    }

    {
        const auto st = core::to_string(raw.proposed_status);
        char tbuf[256];
        std::snprintf(
            tbuf, sizeof tbuf,
            "\"status\":\"%.*s\",\"nodes\":%llu,\"proved\":%s,\"reason\":\"%s\"",
            static_cast<int>(st.size()), st.data(),
            static_cast<unsigned long long>(diag.nodes),
            diag.globally_proved ? "true" : "false", reason.c_str());
        SOR_ROUTE(1, "bab", "terminal", tbuf);
    }

    return raw;
}

core::ProofEvidence milp_evidence(const BabDiagnostics& diag,
                                  const BabOptions& opts) {
    if (diag.lp_only) {
        auto ev = diag.lp_only_evidence;
        if (std::isfinite(diag.incumbent)) {
            ev.max_primal_violation = diag.final_primal_violation;
            ev.checker_passed = ev.max_primal_violation <= opts.primal_feas_tol;
        }
        return ev;
    }
    core::ProofEvidence ev;
    ev.has_basis = false;
    // A3: when an incumbent is being reported, use the real re-check computed
    // once in solve_milp() against the caller's original model (see
    // milp_point_max_violation) instead of an assumed 0. When there is no
    // incumbent at all -- a pure infeasibility proof -- there is no point for
    // "primal violation" to describe; residuals_within_tolerance() also
    // gates global_proof for the Infeasible claim below, via max_dual_violation,
    // so this must stay 0 rather than diag.final_primal_violation's unset +inf,
    // or a genuine infeasibility proof would be downgraded to NoSolutionFound.
    ev.max_primal_violation =
        std::isfinite(diag.incumbent) ? diag.final_primal_violation : 0.0;
    // Read diag.globally_proved directly (set once, in solve_milp()) rather
    // than re-deriving it here from termination_reason -- a STRING match on
    // "tree exhausted" duplicated the proof condition in two places and
    // silently required full tree exhaustion even when the gap had already
    // closed to within opts.gap_tol, which is a complete proof on its own.
    const bool globally_proved = diag.globally_proved;
    ev.max_dual_violation = globally_proved ? 0.0 : core::kPosInf;
    ev.gap_rel = diag.gap_rel;
    ev.primal_feas_tol = opts.primal_feas_tol;
    ev.dual_feas_tol = opts.primal_feas_tol;
    ev.gap_tol = opts.gap_tol;
    ev.checker_passed = std::isfinite(diag.incumbent) &&
                        std::isfinite(ev.max_primal_violation) &&
                        ev.max_primal_violation <= ev.primal_feas_tol;
    if (globally_proved)
        ev.claimed_level = core::ProofLevel::ProvedGlobalEpsilon;
    else if (std::isfinite(diag.incumbent))
        ev.claimed_level = core::ProofLevel::FeasibleWithGap;
    else
        ev.claimed_level = core::ProofLevel::None;
    return ev;
}

const std::vector<MilpCapability>& milp_capability_inventory() {
    using S = CapabilityStatus;
    static const std::vector<MilpCapability> inventory = {
        {"structural presolve", S::Partial,
         "fixed columns, singleton/empty rows, FBBT, binary row support and "
         "substitution, monotone binary-pair saturation, big-M coefficient "
         "strengthening; no side rounding, "
         "aggregation, parallel-row, dominated-column or dual-bound "
         "reductions"},
        {"root probing / MIP presolve / symmetry", S::Implemented,
         "once at MILP entry, bounded by root_reduction_share"},
        {"root cut loop", S::Implemented,
         "GMI, MIR, lifted cover, zero-half, implied bound; opt-in tableau "
         "c-MIR and small-term dynamism repair (net negative on easy60); "
         "clique opt-in; flow cover disabled (invalid cuts)"},
        {"tree / local cuts", S::Partial,
         "separated and ranked at nodes on a sparse depth schedule; the "
         "selected cuts are put into that node's LP for one warm re-solve "
         "(bound raise, prune); inheritance by descendants is implemented but "
         "off by default; sessions are cached by the exact local-row set; "
         "globally valid ones reach global_lp through GCS promotion"},
        {"reliability branching", S::Implemented,
         "unreliable candidates probed on the node-LP session under a "
         "node-LP-time share; --legacy-branching restores the capped path"},
        {"strong-branch domain deductions", S::Implemented,
         "certified dead probe directions tighten the node's bound; both "
         "dead close the node"},
        {"persistent conflict store (bound disjunctions)", S::Partial,
         "integer bound-literal clauses learned from infeasible nodes (decision "
         "nogoods, shrunk by implication-graph analysis where the trail is "
         "complete), plus independently checked Farkas explanations; propagated "
         "without touching the LP and remapped across restarts on unchanged columns"},
        {"objective feasibility pump", S::Implemented,
         "root (and nodes 100, 400, ...) while no incumbent exists; flips and "
         "randomised restarts break cycles; heuristic only"},
        {"objective-face search", S::Implemented,
         "child solve of the full model with c'x <= T just above the proven "
         "bound; heuristic only (a failed attempt proves nothing)"},
        {"independent component solving", S::Implemented,
         "after presolve, a model whose rows split into two or more independent "
         "components is solved component by component under shares of the "
         "deadline; Optimal needs every component proved, an infeasible "
         "component proves infeasibility, bounds add"},
        {"set-partitioning / assignment repair", S::Implemented,
         "weighted-violation local search with ejection-chain compound moves over "
         "unit-coefficient binary rows, run from the LP point while no incumbent exists; "
         "heuristic only"},
        {"estimate-driven node selection / bounded dives", S::Unavailable,
         "best-bound with hybrid plunging (n <= 500 or opt-in) only"},
        {"elapsed-work heuristic budget", S::Unavailable,
         "heuristics are capped by heuristic_budget_frac* of the time limit"},
        {"dive ranking portfolio", S::Unavailable,
         "root dive uses fractional ranking only"},
        {"root primal search (LNS / Balans)", S::Partial,
         "passes on proved root snapshots plus interval-gated node work; child "
         "receives selected parent cuts, a mapped basis, pseudocosts and cutoff; "
         "children still disable their own separation and probing"},
        {"reduced-cost fixing", S::Implemented,
         "certified linear-time Lagrangian pass after proved root cut rounds "
         "and in the node loop, using the known incumbent cutoff"},
        {"tree restart", S::Partial,
         "one frontier-clearing restart after a stall; separate root restart "
         "physically re-presolves after enough reduced-cost fixings"},
        {"root restart / re-presolve", S::Implemented,
         "after >= 10% of free integers fixed by reduced cost against an "
         "incumbent, tighten, presolve again and re-solve with a cutoff"},
        {"LP state reuse", S::Implemented,
         "prepared sessions in a bounded LRU with exact row-set checks; parent "
         "factor/weight checkpoints under a memory budget"},
        {"LP degeneracy recovery", S::Implemented,
         "dual cost perturbation and bounded cold cycling recovery; true-cost "
         "cleanup preserves a feasible basis including boxed variables; primal "
         "plateaus use temporary bound perturbation followed by original-bound restoration"},
    };
    return inventory;
}

double heuristic_spent_ms(const BabDiagnostics& d) {
    // The four timers cover disjoint windows (in-tree heuristic block, FJ,
    // fix-propagate-repair, sub-MIP searches), so their sum is not double
    // counted.
    return d.heuristic_ms + d.feasjump_ms + d.fixprop_ms + d.sub_mip_ms;
}

}  // namespace sor::search
