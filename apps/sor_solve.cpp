// sor_solve - CLI for LP / MILP / first-order engines.
//
// LAYER L8.
#include "sor/backend/kernel_backend.hpp"
#include "sor/core/route_debug.hpp"
#include "sor/backend/lp_device.hpp"
#include "sor/backend/pdhcg_device.hpp"
#include "sor/backend/qp_device.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/core/parallel.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/engines/lp.hpp"
#include "sor/engines/pdhg.hpp"
#include "sor/engines/hpr_qp.hpp"
#include "sor/engines/qcqp.hpp"
#include "sor/engines/qp.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/io/mps.hpp"
#include "sor/io/qplib.hpp"
#include "sor/io/qps.hpp"
#include "sor/io/solution.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/binquad.hpp"
#include "sor/search/bqp_bab.hpp"
#include "sor/search/qcr.hpp"
#include "sor/search/miqp_bb.hpp"
#include "sor/search/global_qp.hpp"
#include "sor/search/qplib_qp.hpp"
#include "sor/search/lattice_reform.hpp"
#include "sor/search/portfolio.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void usage() {
    std::fputs(
        "usage: sor_solve MODEL.{mps,lp,qps,qplib} [options]\n"
        "  --engine NAME    simplex (default) | auto | primal | dual | pdhg | hpr | barrier | milp | qp\n"
        "  --q-diag LIST    comma-separated diagonal of Q (if not using .qps)\n"
        "  --backend NAME   cpu (default) | vulkan | cuda\n"
        "  --engine hprqp   GPU-capable HPR-QP convex QP engine (honours --backend)\n"
        "  --engine qpauto  convex QP routed by outcome: interior point, then the\n"
        "                   first-order path with the remaining budget if unproved\n"
        "  --engine auto    (.qplib) routes on the instance's own 3-letter\n"
        "                   classification to whichever of qpauto/global/binquad/\n"
        "                   miqp/qpipm/qcqplocal is this solver's best answer for\n"
        "                   that shape today; prints \"auto engine: NAME\" then runs\n"
        "                   exactly that engine. Unroutable classes come back\n"
        "                   Unsupported with the reason, never a guess.\n"
        "  --engine binquad binary QP (.qplib): tabu incumbent + certified QCR bound;\n"
        "                   --backend vulkan runs both on the GPU\n"
        "  --eval-solution F  (.qplib) evaluate QPLIB solution file F against the\n"
        "                   model and the raw file; solves nothing (F=zero: x = 0)\n"
        "  --engine miqp    mixed-integer QP (.qplib, linear constraints; general\n"
        "                   integers): B&B over convex IPM node relaxations; a\n"
        "                   nonconvex Q is shifted onto bounded integer columns\n"
        "                   (--max-iter = node limit, --time-limit, --tol = gap)\n"
        "  --bq-searches P  parallel device searches for --engine binquad (default 256)\n"
        "  --bq-batch K     branch-and-bound nodes bounded per device pass (default 8)\n"
        "  --bq-search-only whole budget to the incumbent search; no bound, no proof\n"
        "  --engine global  nonconvex continuous QP (.qplib, linear constraints):\n"
        "                   spatial branch-and-bound, certified bound + checked point\n"
        "  --global-relax R auto | mccormick | shift | both   (--engine global)\n"
        "  --global-no-rlt | --global-no-obbt | --global-no-local\n"
        "  --global-opt K=V any other global option (psd_cuts, psd_max_dim,\n"
        "                   psd_rounds_root, psd_cuts_per_round, psd_max_cuts,\n"
        "                   psd_cut_tol, obbt_max_vars, local_every, ...)\n"
        "  --qcr-shift NAME best (default) | auto | sdp | eig | dd: which diagonal\n"
        "                   shift convexifies the QCR relaxation\n"
        "  --bq-strong C    reliability branching: probe the C most fractional\n"
        "                   variables per node (0 = off, default 4)\n"
        "  --bq-node NAME   auto (default) | ipm | pdhcg: engine for the B&B node\n"
        "                   relaxations (auto = ipm on the CPU, pdhcg on a GPU)\n"
        "  --method NAME    auto | primal | dual   (simplex and MILP node LPs)\n"
        "  --pricing NAME   choose | dantzig | devex | dse  (simplex pricing)\n"
        "  --basis-update NAME  product | ft       (simplex basis updates)\n"
        "  --collective-ft  fold pending product updates into L/U at cleanup\n"
        "  --refactor-interval N  maximum basis updates between refactors\n"
        "  --refactor-eta-ratio R refactor when eta nnz exceeds R*factor nnz\n"
        "  --refactor-work-ratio R refactor when solve work exceeds R*factor nnz\n"
        "  --refactor-u-nnz-ratio R  FT: refactor when nnz(U) exceeds R*factor nnz\n"
        "  --ft-update-limit N       FT: refactor after N row etas\n"
        "  --dual-resync-interval N rebuild dual/reduced costs every N pivots (0=off)\n"
        "  --dual-cost-perturbation M deterministic dual perturbation multiplier\n"
        "  --[no-]primal-crash  feasibility-reducing crash (simplex CLI default: on)\n"
        "  --qp-inner-epoch N  PDHCG-II sparse-Q inner loop: batch N iterations per\n"
        "                   device round trip (default 1 = today's per-iteration\n"
        "                   behaviour, exact); N > 1 defers the stop test and the\n"
        "                   BB step to every Nth iteration -- see QpOptions::inner_epoch\n"
        "  --max-iter N     iteration / node limit\n"
        "  --tol T          feasibility tolerance\n"
        "  --lp-gap-tol G   LP gap tolerance, overriding --tol for LP engines\n"
        "  --exact-proof    simplex/primal/dual: also close an exactly evaluated dual\n"
        "                   bound within the gap tolerance (off: stop at KKT tolerances)\n"
        "  --no-dual-perturbation  disable Koberstein dual cost perturbation (ablation)\n"
        "  --no-primal-bound-perturbation  disable primal plateau recovery (ablation)\n"
        "  --trace-lp        trace LP progress and degeneracy recovery\n"
        "  --conflict-store-max-len N  normalized clause length limit (default 64)\n"
        "  --dual-perturbation-at-start  also perturb up front when costs repeat\n"
        "  --dse-weight-floor F  lower bound on updated DSE weights (default 1e-4)\n"
        "  --mip-gap G      relative MILP proof gap (default 1e-4)\n"
        "  --time-limit S   wall-clock limit in seconds, presolve included\n"
        "                   (simplex default 900; 0 = none)\n"
        "  --no-scaling     skip Ruiz equilibration\n"
        "  --pow2-scaling   round Ruiz factors to powers of two (exact scaling)\n"
        "  --no-presolve    skip presolve (simplex/milp)\n"
        "  --relax-integrality  read integer columns as continuous\n"
        "  --small-matrix-value V  drop coefficients with |a| <= V\n"
        "  --[no-]fo-polish     enable/disable HPR feasibility polishing\n"
        "  --[no-]fo-certificates enable/disable HPR certificate detection\n"
        "  --[no-]fo-crossover  enable/disable Auto FO-to-simplex crossover\n"
        "  --auto-budget-split S  one of 60/25/15, 70/20/10, 80/15/5\n"
        "  --hpr-restart-off | --hpr-reflection-off | --hpr-weight-off\n"
        "  --hpr-weight-pid | --hpr-weight-smoothed  primal-weight controller (default smoothed)\n"
        "  --implied-slack  presolve: drop zero-cost singleton columns as slacks\n"
        "  --lattice-reform  opt-in AHL lattice reform for pure integer equalities\n"
        "  --no-probing     skip MILP root probing (conflict graph, implied bounds)\n"
        "  --no-mip-presolve  skip WP-F MIP root presolve (dual-fix, clique, GF2, ...)\n"
        "  --no-binary-row-support  skip short binary-row support reductions\n"
        "  --no-structural-fbbt  skip global propagation before structural elimination\n"
        "  --no-monotone-binary-pairs  skip objective-preserving activation-pair reductions\n"
        "  --structural-row-probing  enumerate joint binary supports before root setup\n"
        "  --structural-graph-relations  substitute mutually implied binaries before root LP\n"
        "  --structural-graph-support  propagate global conflicts inside joint row assignments\n"
        "  --structural-probe-time S total seconds for joint support cascades (default 1)\n"
        "  --diverse-row-probing  limit candidate binary-group overlap to one half\n"
        "  --gmi-tableau-trials N cap all tableau attempts per round (0=unlimited)\n"
        "  --rank-gmi          prefer fractional candidates away from integers\n"
        "  --no-symmetry    skip WP-G symmetry (orbits / orbital fixing)\n"
        "  --reflection     enable Reflection-complete (off by default; experimental)\n"
        "  --no-reflection  disable Reflection-complete (keep perm/fold)\n"
        "  --no-folding     disable Folding-complete (keep perm/reflection)\n"
        "  --no-dual-fix-probe  disable dual fixing inside probing\n"
        "  --no-clique-probe    disable clique probing strengthen\n"
        "  --no-gf2            disable GF(2)/XOR subsystem reductions\n"
        "  --no-components     disable disconnected-component tighten\n"
        "  --no-implied-int    disable TU/network implied integrality in presolve\n"
        "  --no-obbt           disable OBBT-lite / FBBT deepen\n"
        "  --mip-restarts N    max MIP-presolve restart rounds (default 3)\n"
        "  --no-feasjump    skip the Feasibility Jump primal heuristic\n"
        "  --no-sub-mip     skip the bandit-scheduled LNS portfolio\n"
        "  --no-balans      disable Balans (Latest uses classical ALNS instead)\n"
        "  --no-kernel-pump disable Kernel Pump (Latest primal)\n"
        "  --no-mrens       disable MRENS multi-reference RENS\n"
        "  --no-btbs        disable BTBS-LNS (Latest Balans arm)\n"
        "  --no-cl-tlns     disable CL-TLNS (Latest Balans arm)\n"
        "  --kernel-pump-time S  Kernel Pump wall-clock cap (seconds)\n"
        "  --mrens-time S   MRENS sub-MIP wall-clock cap (seconds)\n"
        "  --btbs-time S    BTBS-LNS sub-MIP wall-clock cap (seconds)\n"
        "  --cl-tlns-time S CL-TLNS sub-MIP wall-clock cap (seconds)\n"
        "  --heuristic-budget F  share of the time limit the heuristic layer may use\n"
        "  --clique-cuts    opt-in clique cut separation in the root cut loop\n"
        "  --no-vub-cuts    skip implied-bound (variable-bound) cut separation\n"
        "  --cover-cuts     opt-in lifted knapsack cover cut separation\n"
        "  --mir-cuts       opt-in mixed-integer rounding cut separation\n"
        "  --gmi-cmir-recovery  re-round high-dynamism GMI through c-MIR (opt-in)\n"
        "  --no-gmi-cmir-recovery  disable c-MIR recovery for ablation\n"
        "  --local-cut-rows N   let descendants inherit up to N node cuts (default 0)\n"
        "  --no-component-solve solve a model whose rows split into independent parts as one\n"
        "  --no-root-primal-early  primal work only after the root node LP (old order)\n"
        "  --no-sub-mip-context  heuristic children start from the bare model\n"
        "  --trace-cuts         print every selected root cut, batch by batch\n"
        "  --no-spp-repair      disable set-partitioning repair while no incumbent exists\n"
        "  --no-conflict-store  learn nogoods as LP rows instead of the propagated store\n"
        "  --no-node-cut-resolve  do not re-solve a node LP with its tree cuts\n"
        "  --tableau-cmir       also round each tableau row as a c-MIR base (opt-in: net negative on easy60)\n"
        "  --relax-small-terms  repair wide cuts by relaxing small terms instead of rejecting (opt-in)\n"
        "  --integer-slack-gmi  use exact-integer row activities in GMI (default)\n"
        "  --no-integer-slack-gmi  disable integer row-activity GMI for ablation\n"
        "  --integer-slack-gmi-basic  also separate from basic row activities\n"
        "  --no-aggregation skip MIR row aggregation (single-row bases only)\n"
        "  --no-mir-variable-bounds  MIR substitutes simple bounds only (ablation)\n"
        "  --mir-probe-bounds  use global probed variable bounds in MIR (experimental)\n"
        "  --mir-lifted-cover  also derive lifted mixed-binary covers on MIR bases\n"
        "  --no-milp-presolve  skip the structural MILP presolve (ablation)\n"
        "  --no-coef-strengthening  skip big-M coefficient strengthening (ablation)\n"
        "  --no-root-restart   never restart with a re-presolve after root reduced-cost fixing\n"
        "  --no-objective-face skip the objective-face feasibility search\n"
        "  --no-event-propagation  sweep every row at every node (ablation)\n"
        "  --no-fpump          skip the objective feasibility pump\n"
        "  --fpump-time S      wall-clock cap per feasibility pump call (default 6)\n"
        "  --plunge            best-bound with bounded plunging at every model size\n"
        "  --plunge-depth N    longest plunge chain (default 30)\n"
        "  --no-rc-strengthening  skip node reduced-cost bound tightening (ablation)\n"
        "  --no-node-lp-cutoff  solve node LPs to optimality past the incumbent cutoff\n"
        "  --events-out PATH      append JSONL search events (incumbents, bounds, end)\n"
        "  --root-reduction-cap S cap in seconds on the root reduction allowance\n"
        "                         (probing / MIP presolve / symmetry; default uncapped)\n"
        "  --root-reduction-share F  root reduction allowance as a share of the limit\n"
        "                         (default 0.20)\n"
        "  --root-cut-share F     root cut round time as a share of the limit (default 0.35)\n"
        "  --root-cut-max S       cap in seconds on the root cut round time (default uncapped)\n"
        "  --milp-capabilities    list what the MILP search implements, then exit\n"
        "  --legacy-branching  pre-2026-09-25 branching: capped strong branching,\n"
        "                      learned scorers under auto (ablation)\n"
        "  --cut-nnz-budget F   nonzeros added per cut round, as a multiple of n (0=off)\n"
        "  --cut-max-density F  reject cuts denser than this fraction of n (>1=off)\n"
        "  --cut-parallel-penalty F  score penalty for parallel cuts (0=off)\n"
        "  --cut-extra-scores F  weight of the sparsity and low-lock score terms\n"
        "  --milp-policy NAME latest (default) | classical (ablation only)\n"
        "  --no-fixprop     skip Fix-Propagate-Repair\n"
        "  --fixprop-time S      wall budget for Fix-Propagate-Repair\n"
        "  --feasjump-time S     wall budget for one Feasibility Jump run\n"
        "  --feasjump-root-frac F  cap FJ at F * time limit (default 0.10)\n"
        "  --batch-lp-sb / --no-batch-lp-sb    batched strong-branch LPs (default off)\n"
        "  --gpu-binary-heuristic / --no-gpu-binary-heuristic  BinQuad tabu "
        "search fallback on pure-binary models with no incumbent yet "
        "(default off)\n"
        "  --no-batch-lp-obbt  approximate OBBT probes are disabled for correctness\n"
        "  --bab-threads N  Para-B&B workers; 0=auto (min(8,cores)), 1=serial\n"
        "  --milp-portfolio N  race N diversified arms sharing incumbents\n"
        "                      (engine milp only; 0=one arm per hw thread)\n"
        "  --auto-cuts / --no-auto-cuts  select bounded cut families (default on)\n"
        "  --cut-max-rounds N   cap root cut rounds (default 100)\n"
        "  --cut-min-progress F  stop the cut loop below this relative gain\n"
        "  --cut-rollback       retract root cut rounds that bought no bound\n"
        "  --cut-patience N     consecutive stalled rounds tolerated (default 3)\n"
        "  --cut-purge          drop root cuts with no dual price at the root LP\n"
        "  --no-cut-warm-rounds re-solve each root cut round cold (default: warm\n"
        "                       start from the previous round's basis)\n"
        "  --cut-marginal-gate  probe each separator's marginal bound contribution\n"
        "                       at the root and stop running the ones worth nothing\n"
        "  --cut-marginal-min F  contribution below this (relative) disables a family\n"
        "  --verify-cuts PATH   abort on any cut that cuts off this .sol point\n"
        "  --branch-strategy NAME  auto|reliability|sparse-sb|sc-milp|lifted|planbb\n"
        "  --sparse-sb-model PATH  load sparse-SB branching model (policy=latest)\n"
        "  --sparse-sb-collect    record SB labels during strong-branch probes\n"
        "  --sc-milp-model PATH   load SC-MILP scoring model (policy=latest)\n"
        "  --sc-milp-collect     record SC-MILP preference labels (same probes)\n"
        "  --lifted-expert PATH   warm-start Lifted expert (sparse-SB format)\n"
        "  --planbb-policy PATH   load PlanB&B linear policy stub (lite)\n"
        "  --planbb-model PATH    load PlanB&B paper model (SOR_PLANBB_PAPER)\n"
        "  --planbb-paper         force paper MBRL path (full MCTS + dynamics)\n"
        "  --planbb-collect       record PlanB&B dynamics/SB labels during probes\n"
        "  --planbb-mcts-sims N   PlanB&B MCTS simulation cap (default 48)\n"
        "  --planbb-mcts-depth N  PlanB&B MCTS depth cap (default 3)\n"
        "  --no-planbb-mcts       use shallow lookahead only (no MCTS)\n"
        "  --no-conflict-prop  skip conflict-graph propagation at nodes\n"
        "  --conflict-cut      enable Mexi cut-based conflict (default on)\n"
        "  --no-conflict-cut   disable conflict learning (Mexi cuts + nogoods)\n"
        "  --conflict-cut-paper  force Paper Mexi even on dense pure-binary\n"
        "  --no-nogood-cuts    disable branch-trail nogood cuts only\n"
        "  --no-dynsep         disable DynSep separator adapter (Latest default on)\n"
        "  --dynsep-backend B  auto|gnn|ucb (default auto: GNN if model else UCB)\n"
        "  --dynsep-model PATH load DynSep GNN (SOR_DYNSEP); Latest prefers GNN\n"
        "  --dynsep-collect    collect (state→sep helped) labels during solve\n"
        "  --dynsep-ucb C      DynSep UCB1 exploration constant (default 1.25)\n"
        "  --dynsep-max-optional N  max optional separator arms per round\n"
        "  --no-l2sep          disable L2Sep instance-aware DynSep config (Latest default on)\n"
        "  --l2sep-model PATH  load L2Sep sparse logistic model (SOR_L2SEP)\n"
        "  --no-hgtsm          disable HGTSM cut sequence scoring (Latest default on)\n"
        "  --hgtsm-model PATH  load HGTSM cut scorer (SOR_HGTSM v1 linear / v2 graph)\n"
        "  --hgtsm-linear      prefer linear fallback even if graph payload loaded\n"
        "  --hgtsm-sequence NAME  transformer|gru  (default transformer)\n"
        "  --hgtsm-collect     record cut efficacy / bound-delta labels\n"
        "  --no-gcs            disable GCS global cut selection (Latest default on)\n"
        "  --gcs-model PATH    load GCS promote/reinject policy (SOR_GCS)\n"
        "  --gcs-heuristic     use multi-node heuristic score (ignore GNN)\n"
        "  --gcs-reinject N    GCS reinject top cuts every N nodes (0=off cadence)\n"
        "  --lp-concurrent N  race N independently checked simplex arms (auto route)\n"
        "  --lp-parallel-basis  evaluate paired basis solves with private worker factors\n"
        "  --lp-domain-probing  enable bounded continuous-domain probing in presolve\n"
        "  --no-lp-sparsification  disable exact equation sparsification\n"
        "  --[no-]dual-crash  zero-cost triangular dual cold start (default: on)\n"
        "  --threads N      worker threads for the sparse linear algebra\n"
        "  --debug-routes[=N]       JSONL route trace, N=0..3 (bare flag = 1)\n"
        "  --debug-routes-file=PATH append the JSONL (stderr if omitted)\n"
        "  --debug-routes-comp=LIST comma allowlist of comp values\n"
        "  --debug-routes-path=LIST comma allowlist of path tags\n"
        "  --debug-routes-fns       trace every function (enter + summary)\n"
        "  --debug-routes-fn=LIST   comma substrings; only those function names\n"
        "  --pivot-trace-every=N    emit 1 of every N level-3 pivot events\n"
        "  --verbose        iteration / node log\n"
        "  --hpr-vanilla | --hpr-full\n"
        "  --solution-out PATH   write a plain-text solution file for sor_check\n"
        "exit status: 0 Optimal; 1 error (unreadable model, ...); 2 bad option;\n"
        "  3 engine unavailable; 4 Interrupted (limit hit, no point);\n"
        "  5 NumericalFailure, NoSolutionFound or Unsupported; 6 Feasible (a\n"
        "  checked point without an optimality proof); 7 Infeasible;\n"
        "  8 Unbounded or InfeasibleOrUnbounded\n",
        stderr);
}

// ---------------------------------------------------------------------------
// Numeric option parsing.
//
// Every numeric flag used to go straight through strtod/strtoull with a null
// end pointer, which accepts far too much in silence: "abc" parses as 0,
// "-5" as a huge unsigned, and "nan"/"inf" as themselves. A mistyped flag
// therefore produced a DIFFERENT RUN instead of an error -- `--tol abc` ran at
// tolerance exactly 0, `--max-iter -5` ran with an effectively infinite cap,
// and both exited 0 as though nothing was wrong. These parse the WHOLE token,
// reject non-finite values, and enforce each option's own admissible range.
[[noreturn]] void bad_value(const char* what, const std::string& got,
                            const std::string& expected) {
    std::fprintf(stderr, "error: %s expects %s, got '%s'\n", what,
                 expected.c_str(), got.c_str());
    std::exit(2);
}

// std::to_string(0.0) is "0.000000", which reads badly in an error message.
std::string compact(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

std::string range_text(double lo, double hi, bool lo_open) {
    std::string s = "a finite number ";
    s += lo_open ? "greater than " : "at least ";
    s += compact(lo);
    if (hi < std::numeric_limits<double>::max()) s += " and at most " + compact(hi);
    return s;
}

double parse_real(const std::string& text, const char* what, double lo,
                  double hi = std::numeric_limits<double>::max(),
                  bool lo_open = false) {
    std::size_t used = 0;
    double v = 0.0;
    try {
        v = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != text.size() || text.empty() || !std::isfinite(v))
        bad_value(what, text, range_text(lo, hi, lo_open));
    if (lo_open ? !(v > lo) : !(v >= lo)) bad_value(what, text, range_text(lo, hi, lo_open));
    if (!(v <= hi)) bad_value(what, text, range_text(lo, hi, lo_open));
    return v;
}

// A leading '-' is rejected before stoull sees it: stoull WRAPS a negative
// value into a huge positive one rather than failing.
unsigned long long parse_uint(const std::string& text, const char* what,
                              unsigned long long lo, unsigned long long hi) {
    const std::string expected = "an integer from " + std::to_string(lo) +
                                 " to " + std::to_string(hi);
    if (text.empty() || text.find('-') != std::string::npos)
        bad_value(what, text, expected);
    std::size_t used = 0;
    unsigned long long v = 0;
    try {
        v = std::stoull(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != text.size() || v < lo || v > hi) bad_value(what, text, expected);
    return v;
}

// Set when an LP engine answers a model with integer columns: the claim is
// about the LP relaxation, and the solution file says so for sor_check.
bool answering_lp_relaxation = false;

// No-op when `path` is empty (the common case: --solution-out wasn't given).
void write_solution_out(const std::string& path, const sor::core::SolveResult& r) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "error: could not open '%s' for --solution-out; "
                             "the solution was NOT written\n",
                     path.c_str());
        return;
    }
    sor::io::write_solution(out, r, answering_lp_relaxation);
}

void print_result(const sor::core::SolveResult& r) {
    std::printf("\nstatus:            %s\n",
                std::string(sor::core::to_string(r.status)).c_str());
    std::printf("proof_level:       %s\n",
                std::string(sor::core::to_string(r.proof)).c_str());
    std::printf("                   %s\n",
                std::string(sor::core::human_line(r.status, r.proof)).c_str());
    if (!r.downgrade_reason.empty())
        std::printf("downgrade:         %s\n", r.downgrade_reason.c_str());
    std::printf("objective:         %.10e\n", r.objective);
}

// Owns the route ledger for one process. Constructed after flags are parsed
// so --debug-routes-fns is already on. Destructors of the solve run before
// this one, which is what makes the function summary complete.
struct RouteSession {
    bool on = false;
    std::chrono::steady_clock::time_point t0{};
    RouteSession() {
        if (sor::core::route_debug_fns_on() && sor::core::route_debug_level() < 1)
            sor::core::route_debug_set_level(1);
        on = sor::core::route_debug_level() > 0 || sor::core::route_debug_fns_on();
        if (!on) return;
        t0 = std::chrono::steady_clock::now();
        sor::core::route_debug_ledger_reset();
    }
    ~RouteSession() {
        if (!on) return;
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        sor::core::route_debug_ledger_set_wall(ms);
        sor::core::route_debug_fn_flush();
        sor::core::route_debug_ledger_emit("solve");
        char buf[1024];
        sor::core::route_debug_ledger_format(buf, sizeof buf);
        std::printf("%s\n", buf);
        if (sor::core::route_debug_fns_on()) {
            std::printf("fn trace: %llu functions entered\n",
                        static_cast<unsigned long long>(
                            sor::core::route_debug_fn_entered()));
        }
    }
};

// One code per kind of outcome, so a script can tell a proved answer from a
// failure without parsing stdout: a Feasible point (an LP optimum the checker
// downgraded, or a MILP incumbent at a limit) is not an optimum, and a proved
// Infeasible is not a numerical failure.
int exit_code_for(sor::core::Status s) {
    switch (s) {
        case sor::core::Status::Optimal:
            return 0;
        case sor::core::Status::Interrupted:
            return 4;
        case sor::core::Status::Feasible:
            return 6;
        case sor::core::Status::Infeasible:
            return 7;
        case sor::core::Status::Unbounded:
        case sor::core::Status::InfeasibleOrUnbounded:
            return 8;
        default:
            return 5;
    }
}

// A crossed bound (lower above upper) makes the model infeasible by its data
// alone, which finalize_result and sor_check re-derive from the model. The
// quadratic engines cannot represent it (one claimed Optimal), so their routes
// stop here. Returns the exit status, or -1 when every domain is nonempty.
int report_empty_domain(const sor::model::LpProblem& p,
                        const std::string& solution_out) {
    const auto empty = p.find_empty_domain();
    if (empty.index < 0) return -1;
    sor::core::RawResult raw;
    raw.proposed_status = sor::core::Status::Infeasible;
    raw.engine = "model";
    raw.termination_reason = p.describe(empty);
    sor::core::ProofEvidence ev;
    ev.empty_domain = true;
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    print_result(r);
    write_solution_out(solution_out, r);
    std::printf("termination:       %s\n", r.termination_reason.c_str());
    return exit_code_for(r.status);
}

void print_transfer(const sor::backend::TransferStats& s) {
    std::printf("  kernel           %10.3f\n", s.kernel_ms);
    std::printf("  host->device     %10.3f  (%llu bytes)\n", s.h2d_ms,
                static_cast<unsigned long long>(s.h2d_bytes));
    std::printf("  device->host     %10.3f  (%llu bytes)\n", s.d2h_ms,
                static_cast<unsigned long long>(s.d2h_bytes));
    std::printf("  ipc overhead     %10.3f\n", s.ipc_ms);
    std::printf("  kernel calls     %10llu\n",
                static_cast<unsigned long long>(s.calls));
}

}  // namespace

// QPLIB -> QpProblem for the convex QP engines: continuous variables and
// linear constraints only.  Discrete variables and quadratic constraints are
// refused rather than silently relaxed or dropped -- a relaxation answered
// as if it were the instance would be a wrong answer, not a bound.  The
// objective convention lives in search/qplib_qp.hpp, shared with QCR.
static bool load_qplib_convex(sor::engines::QpProblem& qp, const std::string& path) {
    sor::io::QplibReadReport rep;
    const auto q = sor::io::read_qplib_file(path, rep);
    bool negated = false;
    std::string why;
    if (!sor::search::qplib_to_qp(q, {}, qp, negated, why)) {
        std::fprintf(stderr, "error: %s\n", why.c_str());
        return false;
    }
    if (negated)
        std::fprintf(stderr, "note: %s is a maximisation; solved as min of the "
                             "negated objective\n", q.name.c_str());
    return true;
}

// Independent check of a binary point against the RAW QPLIB data: shares no
// code with binquad or with qplib_to_qp, so a convention error in either
// cannot hide here.  Objective in the instance's original sense.
static void qplib_binary_check(const sor::io::QplibInstance& q,
                               const std::vector<std::uint8_t>& x,
                               double& objective, double& violation) {
    // These checkers read LINEAR row activity only -- they never accumulate the
    // per-row quadratic Hessians (q.hc_*), so on an instance with quadratic
    // constraint rows the "violation" below is blind to exactly the rows that
    // decide feasibility.  That is safe today only because qplib_to_qp refuses
    // such instances (src/search/src/qplib_qp.cpp:104), so no live caller can
    // arrive here with one -- but --engine global DID reach this once and
    // silently dropped every quadratic row, which turned a feasible point
    // (viol 1e-11) into viol 3.5e+05.  The reverse direction is the dangerous
    // one: a point VIOLATING a quadratic row would look clean and could pass.
    // So refuse rather than leave the trap armed.  An infinite violation can
    // only sink a claim, never manufacture one.
    if (q.has_quadratic_constraints()) {
        objective = sor::core::kNaN;
        violation = sor::core::kPosInf;
        return;
    }
    objective = q.f_const;
    for (std::size_t t = 0; t < q.h_val.size(); ++t)
        objective += 0.5 * q.h_val[t] * x[static_cast<std::size_t>(q.h_row[t] - 1)] *
                     x[static_cast<std::size_t>(q.h_col[t] - 1)];
    for (std::size_t j = 0; j < q.g.size(); ++j) objective += q.g[j] * x[j];
    std::vector<double> ax(static_cast<std::size_t>(q.m), 0.0);
    for (std::size_t t = 0; t < q.a_val.size(); ++t)
        ax[static_cast<std::size_t>(q.a_row[t] - 1)] +=
            q.a_val[t] * x[static_cast<std::size_t>(q.a_col[t] - 1)];
    violation = 0.0;
    for (std::size_t i = 0; i < ax.size(); ++i) {
        if (q.c_lo[i] > -q.inf_bound && ax[i] < q.c_lo[i])
            violation = std::max(violation, q.c_lo[i] - ax[i]);
        if (q.c_hi[i] < q.inf_bound && ax[i] > q.c_hi[i])
            violation = std::max(violation, ax[i] - q.c_hi[i]);
    }
}

// Reads only the name and the 3-letter classification line (two significant
// lines) without building the model -- cheap enough to call once per
// instance from --engine auto without doubling the read of a 150 MB file.
// '?' marks a classification this couldn't make out (too few tokens on the
// second significant line, or an unreadable file); every caller must treat
// that as "not a recognised class", never as some specific letter.
static std::array<char, 3> qplib_read_classification(const std::string& path) {
    std::ifstream in(path);
    std::string line, cls;
    int seen = 0;
    while (seen < 2 && std::getline(in, line)) {
        std::istringstream is(line);
        std::string w;
        if (!(is >> w)) continue;
        if (++seen == 2) cls = w;
    }
    if (cls.size() != 3) return {'?', '?', '?'};
    return {cls[0], cls[1], cls[2]};
}

// True when the file's classification letter says quadratic constraints
// (D, C or Q).
static bool qplib_class_has_quadratic_rows(const std::string& path) {
    const auto cls = qplib_read_classification(path);
    return cls[2] == 'D' || cls[2] == 'C' || cls[2] == 'Q';
}

// --engine auto (.qplib): route QPLIB's own 3-letter classification to
// whichever concrete engine is the strongest one this solver has for that
// shape today.  [0] objective: L linear, D convex diagonal quadratic,
// C convex quadratic, Q possibly nonconvex quadratic.  [1] variables:
// C continuous, B all binary, M mixed binary+continuous, I mixed general-
// integer+continuous, G mixed binary+general-integer.  [2] constraints:
// N none, B box (bounds only, same as N for section layout), L general
// linear, D/C/Q quadratic rows (diagonal-convex / convex / possibly
// nonconvex, same "quadratic present" section layout).
//
// This table adds no solve path of its own -- it only picks which of the
// CLI's existing --engine branches to fall into, so a routed instance is
// held to exactly the claims that engine already makes for itself.  Kept
// pure (no I/O) so it is unit-testable without a file on disk; the CLI test
// (tests/test_qplib_auto_engine.cpp) drives it end to end through the
// binary instead, since that is what a mis-wired route table would break.
struct QplibAutoRoute {
    std::string engine;   // empty => unsupported; see `reason`
    std::string reason;
};

static QplibAutoRoute qplib_auto_route(const std::array<char, 3>& cls) {
    const char obj = cls[0], var = cls[1], con = cls[2];
    const std::string tag(cls.begin(), cls.end());
    const bool obj_ok = obj == 'L' || obj == 'D' || obj == 'C' || obj == 'Q';
    const bool var_ok = var == 'C' || var == 'B' || var == 'M' || var == 'I' || var == 'G';
    const bool con_ok = con == 'N' || con == 'B' || con == 'L' || con == 'D' ||
                        con == 'C' || con == 'Q';
    if (!obj_ok || !var_ok || !con_ok)
        return {"", "classification '" + tag + "' is not a recognised QPLIB "
                     "objective/variable/constraint letter combination"};

    const bool quad_rows = (con == 'D' || con == 'C' || con == 'Q');
    if (quad_rows) {
        // QCQP: quadratic (possibly nonconvex) rows are present.
        if (var == 'C') {
            // Continuous.  LCD is the one class QPLIB's own scheme
            // guarantees diagonal-convex rows under a linear objective; the
            // IPM's QCQP path (--engine qpipm) certifies convexity itself
            // and refuses anything it cannot prove, so this is a genuine
            // exact/proved route, not a guess.  Every other continuous QCQP
            // class -- nonconvex objective and/or rows -- goes to the local
            // method, which never claims more than Feasible.
            if (tag == "LCD") return {"qpipm", ""};
            return {"qcqplocal", ""};
        }
        // Binary / mixed-binary / mixed-integer with quadratic rows: the
        // QCQP branch-and-bound (miqcqp_bb). Its node relaxations must
        // certify convex themselves; a node that can't is refused there,
        // not here.
        return {"miqp", ""};
    }
    // No quadratic rows: linear / box / no general constraints.
    if (var == 'B')
        return {"binquad", ""};  // tabu incumbent + certified QCR bound
    if (var == 'C')
        return (obj == 'L' || obj == 'D' || obj == 'C')
            ? QplibAutoRoute{"qpauto", ""}
            : QplibAutoRoute{"global", ""};  // obj == 'Q': nonconvex continuous QP
    // Mixed binary/continuous, mixed integer/continuous, mixed binary/
    // integer, linearly constrained: the general MIQP B&B (a nonconvex Q is
    // shifted onto bounded integer columns).
    return {"miqp", ""};
}

// --eval-solution: a published QPLIB point, evaluated twice -- against the
// solver's MODEL (qplib_to_qcqp + evaluate_qcqp: what an engine would see)
// and against the RAW file data (io::qplib_evaluate_point, which shares no
// conversion code) -- plus the objective the .sol file itself states.  The
// parse-all sweep (scripts/qplib_parse_all.py) compares all three with
// QPLIB's published SOLOBJVALUE.  Exit 0 when model and raw agree on the
// objective to 1e-9 relative; 1 otherwise.  Feasibility is reported, not
// gated here: some published points are infeasible by QPLIB's own record.
static int eval_qplib_solution(const std::string& path, const std::string& sol_path,
                               double feas_tol) {
    sor::io::QplibReadReport rep;
    auto q = sor::io::read_qplib_file(path, rep);
    // The .sol names its variables and most .qplib files carry no names, so
    // take the mapping from the instance's own published GAMS source when the
    // sidecar is beside it (scripts/fetch_qplib_varnames.py) instead of
    // guessing it.  Reported on the EVAL line so a run cannot be read as
    // stronger evidence than it is.
    const bool named = sor::io::apply_qplib_varnames_file(path, q);
    // "zero" evaluates x = 0: a parse-and-convert check for the instances
    // QPLIB publishes no point for.
    sor::io::QplibSolution sol;
    if (sol_path == "zero") sol.x.assign(static_cast<std::size_t>(q.n), 0.0);
    else sol = sor::io::read_qplib_solution(sol_path, q);
    sor::engines::QcqpProblem model;
    sor::search::qplib_to_qcqp(q, model);
    model.validate();
    const auto me = sor::engines::evaluate_qcqp(model, sol.x);
    const auto re = sor::io::qplib_evaluate_point(q, sol.x);
    const double scale = std::max(1.0, std::fabs(re.objective));
    const bool agree = std::fabs(me.objective - re.objective) <= 1e-9 * scale;
    std::printf("model:             %s  (%c%c%c)  n %d  m %d  quadratic rows %zu\n",
                q.name.c_str(), q.classification[0], q.classification[1],
                q.classification[2], q.n, q.m, model.quad.size());
    std::printf("file consumed:     %s (%zu of %zu lines%s%s)\n",
                rep.fully_consumed() ? "yes" : "NO", rep.lines_consumed, rep.lines_total,
                rep.trailer_error.empty() ? "" : "; trailer: ",
                rep.trailer_error.c_str());
    std::printf("objective model:   %.17g\n", me.objective);
    std::printf("objective raw:     %.17g\n", re.objective);
    if (sol.has_objvar) std::printf("objective in .sol: %.17g\n", sol.objvar);
    std::printf("violation model:   rows %.3e  quadratic rows %.3e  bounds %.3e  integrality %.3e\n",
                me.max_row_violation, me.max_qc_violation, me.max_bound_violation,
                me.max_integrality_violation);
    std::printf("violation raw:     rows %.3e  quadratic rows %.3e  bounds %.3e  integrality %.3e\n",
                re.max_row_violation, re.max_qc_violation, re.max_bound_violation,
                re.max_integrality_violation);
    std::printf("feasible (%.0e):   %s\n", feas_tol, me.max_violation() <= feas_tol ? "yes" : "no");
    // One machine-readable line for the sweep.
    char objvar[64] = "none";
    if (sol.has_objvar) std::snprintf(objvar, sizeof objvar, "%.17g", sol.objvar);
    std::printf("EVAL name=%s class=%c%c%c n=%d m=%d nqc=%zu consumed=%d obj_model=%.17g "
                "obj_raw=%.17g objvar=%s viol_model=%.17g viol_raw=%.17g qcviol=%.17g "
                "named=%d agree=%d\n",
                q.name.c_str(), q.classification[0], q.classification[1], q.classification[2],
                q.n, q.m, model.quad.size(), rep.fully_consumed() ? 1 : 0, me.objective,
                re.objective, objvar,
                me.max_violation(), re.max_violation(), me.max_qc_violation,
                named ? 1 : 0, agree ? 1 : 0);
    return agree ? 0 : 1;
}

// Independent check of a mixed-integer point against the RAW QPLIB data, the
// general-integer sibling of qplib_binary_check: objective in the original
// sense from the stored triangle, rows, variable bounds, and integrality of
// Refuse an input that is not a readable, non-empty regular file, and say which
// of those it failed. Without this an unreadable path reaches the readers, which
// return an EMPTY model rather than an error -- and an empty LP is trivially
// optimal, so the solver reported "Optimal, ProvedOptimalFP, objective 0" for a
// file it never read. A confident answer to a question nobody asked is worse
// than any crash, so this is a hard refusal before any engine runs.
static bool input_is_usable(const std::string& path, std::string& err) {
    std::error_code ec;
    const std::filesystem::path p(path);
    if (!std::filesystem::exists(p, ec)) { err = "no such file: " + path; return false; }
    if (std::filesystem::is_directory(p, ec)) { err = "is a directory, not a model: " + path; return false; }
    if (!std::filesystem::is_regular_file(p, ec)) { err = "not a regular file: " + path; return false; }
    const auto sz = std::filesystem::file_size(p, ec);
    if (ec) { err = "cannot determine the size of " + path; return false; }
    if (sz == 0) { err = "file is empty: " + path; return false; }
    std::ifstream probe(path);
    if (!probe.good()) { err = "cannot open for reading: " + path; return false; }
    return true;
}

// Apply every "key=value" collected for one engine. A bad key or value is
// fatal, never a warning: a caller who mistyped an option believes it took
// effect, and silently ignoring it would make a tuning run a lie.
static bool apply_pending_options(const std::vector<sor::core::OptionBinding>& table,
                                 const std::vector<std::string>& kvs, const char* flag) {
    for (const auto& kv : kvs) {
        std::string err;
        if (!sor::core::apply_option(table, kv, err)) {
            std::fprintf(stderr, "error: %s %s\n", flag, err.c_str());
            return false;
        }
    }
    return true;
}

// --list-opts [engine]: every tunable option with its type, default and what it
// does, printed from the same binding tables the flags apply, so the listing
// cannot fall out of step with what the solver actually accepts.
static void print_engine_options(const std::string& which) {
    const bool all = which.empty() || which == "all";
    if (all || which == "qp" || which == "qpipm" || which == "qpauto" || which == "hprqp") {
        sor::engines::QpOptions o;
        std::printf("--qp-opt KEY=VALUE   (engines qp, qpipm, qpauto)\n%s\n",
                    sor::core::format_options(sor::engines::qp_option_bindings(o)).c_str());
    }
    if (all || which == "qcqplocal" || which == "qcqp") {
        sor::engines::QcqpLocalOptions o;
        std::printf("--qcqp-opt KEY=VALUE   (engine qcqplocal)\n%s\n",
                    sor::core::format_options(sor::engines::qcqp_local_option_bindings(o)).c_str());
    }
    if (all || which == "miqp") {
        sor::search::MiqpBbOptions o;
        std::printf("--miqp-opt KEY=VALUE   (engines miqp, miqcqp)\n%s\n",
                    sor::core::format_options(sor::search::miqp_bb_option_bindings(o)).c_str());
    }
    if (all || which == "global") {
        sor::search::GlobalQpOptions o;
        std::printf("--global-opt KEY=VALUE   (engine global)\n%s\n",
                    sor::core::format_options(sor::search::global_qp_option_bindings(o)).c_str());
    }
}

// every column the FILE types integer or binary.  Shares no code with
// qplib_to_qp or the MIQP search.
static void qplib_point_check(const sor::io::QplibInstance& q, const std::vector<double>& x,
                              double& objective, double& violation, double& int_violation) {
    // These checkers read LINEAR row activity only -- they never accumulate the
    // per-row quadratic Hessians (q.hc_*), so on an instance with quadratic
    // constraint rows the "violation" below is blind to exactly the rows that
    // decide feasibility.  That is safe today only because qplib_to_qp refuses
    // such instances (src/search/src/qplib_qp.cpp:104), so no live caller can
    // arrive here with one -- but --engine global DID reach this once and
    // silently dropped every quadratic row, which turned a feasible point
    // (viol 1e-11) into viol 3.5e+05.  The reverse direction is the dangerous
    // one: a point VIOLATING a quadratic row would look clean and could pass.
    // So refuse rather than leave the trap armed.  An infinite violation can
    // only sink a claim, never manufacture one.
    if (q.has_quadratic_constraints()) {
        objective = sor::core::kNaN;
        violation = sor::core::kPosInf;
        int_violation = sor::core::kPosInf;
        return;
    }
    objective = q.f_const;
    for (std::size_t t = 0; t < q.h_val.size(); ++t)
        objective += 0.5 * q.h_val[t] * x[static_cast<std::size_t>(q.h_row[t] - 1)] *
                     x[static_cast<std::size_t>(q.h_col[t] - 1)];
    for (std::size_t j = 0; j < q.g.size(); ++j) objective += q.g[j] * x[j];
    std::vector<long double> ax(static_cast<std::size_t>(q.m), 0.0L);
    for (std::size_t t = 0; t < q.a_val.size(); ++t)
        ax[static_cast<std::size_t>(q.a_row[t] - 1)] +=
            static_cast<long double>(q.a_val[t]) * x[static_cast<std::size_t>(q.a_col[t] - 1)];
    violation = 0.0;
    for (std::size_t i = 0; i < ax.size(); ++i) {
        const double a = static_cast<double>(ax[i]);
        if (q.c_lo[i] > -q.inf_bound && a < q.c_lo[i]) violation = std::max(violation, q.c_lo[i] - a);
        if (q.c_hi[i] < q.inf_bound && a > q.c_hi[i]) violation = std::max(violation, a - q.c_hi[i]);
    }
    int_violation = 0.0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (q.x_lo[j] > -q.inf_bound && x[j] < q.x_lo[j]) violation = std::max(violation, q.x_lo[j] - x[j]);
        if (q.x_hi[j] < q.inf_bound && x[j] > q.x_hi[j]) violation = std::max(violation, x[j] - q.x_hi[j]);
        if (q.var_type[j] != sor::io::QplibVarType::Continuous)
            int_violation = std::max(int_violation, std::fabs(x[j] - std::round(x[j])));
    }
}

// The same independent re-check for a continuous point (--engine global):
// objective from the raw triplets, and the worst violation of any row or
// variable bound, in the instance's original units and sense.
static void qplib_point_check(const sor::io::QplibInstance& q, const std::vector<double>& x,
                              double& objective, double& violation) {
    // These checkers read LINEAR row activity only -- they never accumulate the
    // per-row quadratic Hessians (q.hc_*), so on an instance with quadratic
    // constraint rows the "violation" below is blind to exactly the rows that
    // decide feasibility.  That is safe today only because qplib_to_qp refuses
    // such instances (src/search/src/qplib_qp.cpp:104), so no live caller can
    // arrive here with one -- but --engine global DID reach this once and
    // silently dropped every quadratic row, which turned a feasible point
    // (viol 1e-11) into viol 3.5e+05.  The reverse direction is the dangerous
    // one: a point VIOLATING a quadratic row would look clean and could pass.
    // So refuse rather than leave the trap armed.  An infinite violation can
    // only sink a claim, never manufacture one.
    if (q.has_quadratic_constraints()) {
        objective = sor::core::kNaN;
        violation = sor::core::kPosInf;
        return;
    }
    objective = q.f_const;
    for (std::size_t t = 0; t < q.h_val.size(); ++t)
        objective += 0.5 * q.h_val[t] * x[static_cast<std::size_t>(q.h_row[t] - 1)] *
                     x[static_cast<std::size_t>(q.h_col[t] - 1)];
    for (std::size_t j = 0; j < q.g.size(); ++j) objective += q.g[j] * x[j];
    std::vector<double> ax(static_cast<std::size_t>(q.m), 0.0);
    for (std::size_t t = 0; t < q.a_val.size(); ++t)
        ax[static_cast<std::size_t>(q.a_row[t] - 1)] +=
            q.a_val[t] * x[static_cast<std::size_t>(q.a_col[t] - 1)];
    violation = 0.0;
    for (std::size_t i = 0; i < ax.size(); ++i) {
        if (q.c_lo[i] > -q.inf_bound && ax[i] < q.c_lo[i])
            violation = std::max(violation, q.c_lo[i] - ax[i]);
        if (q.c_hi[i] < q.inf_bound && ax[i] > q.c_hi[i])
            violation = std::max(violation, ax[i] - q.c_hi[i]);
    }
    for (std::size_t j = 0; j < x.size() && j < q.x_lo.size(); ++j) {
        if (q.x_lo[j] > -q.inf_bound && x[j] < q.x_lo[j])
            violation = std::max(violation, q.x_lo[j] - x[j]);
        if (q.x_hi[j] < q.inf_bound && x[j] > q.x_hi[j])
            violation = std::max(violation, x[j] - q.x_hi[j]);
    }
}

// A heuristic/local result on a QPLIB instance: the point is re-evaluated on
// the RAW file data (rows incl. quadratic ones, bounds, integrality) and
// only then may finalize_result see a Feasible claim.  Objective reported in
// the instance's own sense.  Status is never better than Feasible: nothing
// here proves optimality.
static void print_local_result(const sor::io::QplibInstance& q,
                               const sor::engines::QcqpProblem& model,
                               sor::core::RawResult raw, const std::string& reason,
                               double feas_tol, const std::string& solution_out) {
    double chk_obj = sor::core::kNaN, chk_viol = sor::core::kPosInf;
    sor::core::ProofEvidence ev;
    ev.primal_feas_tol = feas_tol;
    ev.dual_feas_tol = feas_tol;
    ev.max_dual_violation = 0.0;
    ev.claimed_level = sor::core::ProofLevel::None;
    if (raw.x.size() == static_cast<std::size_t>(q.n)) {
        const auto re = sor::io::qplib_evaluate_point(q, raw.x);
        chk_obj = re.objective;
        chk_viol = re.max_violation();
        const double mine = model.objective_negated ? -raw.objective : raw.objective;
        const bool feasible = chk_viol <= feas_tol;
        const bool agrees = !std::isfinite(raw.objective) ||
                            std::fabs(chk_obj - mine) <= 1e-9 * (1.0 + std::fabs(chk_obj));
        ev.max_primal_violation = chk_viol;
        ev.checker_passed = feasible && agrees;
        if (ev.checker_passed) {
            raw.objective = chk_obj;   // the raw evaluator's value, original sense
            ev.claimed_level = sor::core::ProofLevel::FeasibleOnly;
            raw.proposed_status = sor::core::Status::Feasible;
            raw.proposed_level = sor::core::ProofLevel::FeasibleOnly;
        } else if (raw.proposed_status == sor::core::Status::Feasible) {
            raw.proposed_status = sor::core::Status::NoSolutionFound;
            raw.proposed_level = sor::core::ProofLevel::None;
        }
    }
    raw.dual_bound = sor::core::kNaN;
    if (!ev.checker_passed && raw.proposed_status == sor::core::Status::Optimal)
        raw.proposed_status = sor::core::Status::NoSolutionFound;
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    print_result(r);
    write_solution_out(solution_out, r);
    std::printf("raw re-check:      obj %.10e  viol %.3e\n", chk_obj, chk_viol);
    std::printf("termination:       %s\n", reason.c_str());
}

// Shared .qps / --q-diag loading for --engine qp and --engine hprqp. Returns
// false after printing the reason, so each caller can just propagate the exit
// code. Factored out when hprqp arrived rather than duplicated, because the two
// engines must accept exactly the same inputs or "--engine hprqp" silently
// means something different from "--engine qp" on the same file.
static bool load_qp_problem_impl(sor::engines::QpProblem& qp,
                                 const std::string& path,
                                 const std::string& q_diag_arg,
                                 bool path_is_qps,
                                 bool mps_format_forced,
                                 const sor::io::MpsReadOptions& mps_opts) {
    if (!q_diag_arg.empty()) {
        sor::io::MpsReadReport rep;
        qp.linear = mps_format_forced
            ? sor::io::read_mps_file(path, rep, mps_opts)
            : sor::io::read_mps_file_auto(path, rep, mps_opts);
        for (const auto& w : rep.warnings)
            std::fprintf(stderr, "warning: %s\n", w.c_str());
        qp.q_diag.clear();
        std::size_t start = 0;
        while (start <= q_diag_arg.size()) {
            const auto comma = q_diag_arg.find(',', start);
            const auto tok = q_diag_arg.substr(
                start, comma == std::string::npos ? std::string::npos
                                                  : comma - start);
            if (!tok.empty())
                qp.q_diag.push_back(parse_real(
                    tok, "--q-diag", -std::numeric_limits<double>::max()));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        return true;
    }
    if (path.size() >= 6 && path.compare(path.size() - 6, 6, ".qplib") == 0)
        return load_qplib_convex(qp, path);
    if (path_is_qps) {
        sor::io::QpsReadReport rep;
        auto loaded = sor::io::read_qps_file(path, rep, mps_opts);
        for (const auto& w : rep.warnings)
            std::fprintf(stderr, "warning: %s\n", w.c_str());
        qp.linear = std::move(loaded.linear);
        qp.q_diag = std::move(loaded.q_diag);
        qp.q_matrix = std::move(loaded.q_matrix);
        return true;
    }
    std::fprintf(stderr,
                 "error: a quadratic engine needs a .qps file or --q-diag\n");
    return false;
}

// Every QP route loads through here, and a model with no columns is refused for
// all of them in ONE place rather than at each call site.
//
// A zero-column problem is vacuously optimal -- no variables, nothing to
// violate, objective 0 -- so the engines are not wrong to converge on it. The
// danger is that this is indistinguishable from a file that did not parse, and
// what the caller sees is "Optimal / ProvedKKT" for input that carried no
// model. An earlier fix guarded only the LP path, so the four quadratic engines
// still claimed it; putting the check in the shared loader is what stops the
// next route from reintroducing it.
static bool load_qp_problem(sor::engines::QpProblem& qp,
                            const std::string& path,
                            const std::string& q_diag_arg,
                            bool path_is_qps,
                            bool mps_format_forced,
                            const sor::io::MpsReadOptions& mps_opts) {
    if (!load_qp_problem_impl(qp, path, q_diag_arg, path_is_qps, mps_format_forced, mps_opts))
        return false;
    if (qp.linear.n_cols() == 0) {
        std::fprintf(stderr,
                     "error: %s contains no variables -- nothing to solve. A model "
                     "with no columns is almost always a file that did not parse as "
                     "the format it was read as.\n",
                     path.c_str());
        return false;
    }
    return true;
}

// Every load path funnels through here before any engine sees the model. A
// column-less problem is vacuously optimal -- no variables, nothing to
// violate, objective 0 -- so an engine that converges on it is not wrong. The
// danger is that this is indistinguishable from input that never parsed (a
// missing file, a directory, an empty file, or a file the reader didn't
// recognize all come back as an empty model with a warning, not an error), so
// without this check the CLI reports "Optimal / ProvedOptimalFP" for a model
// nobody actually read.
bool refuse_if_no_columns(const std::string& path, int n_cols) {
    if (n_cols != 0) return false;
    std::fprintf(stderr,
                 "error: %s contains no variables -- nothing to solve. A model "
                 "with no columns is almost always a file that did not parse as "
                 "the format it was read as.\n",
                 path.c_str());
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }

    std::string path, backend_name = "cpu", engine_name = "simplex";
    std::string q_diag_arg;
    std::uint32_t bq_searches = 256;   // --engine binquad on a device
    std::uint32_t bq_batch = 8;        // --engine binquad: B&B nodes per pass
    bool bq_search_only = false;       // --engine binquad: incumbent search only
    std::string qcr_shift_arg;         // --engine binquad: which QCR shift (empty = default)
    std::uint32_t bq_strong = 4;       // --engine binquad: strong-branching candidates
    std::string bq_node_arg;           // --engine binquad: node relaxation engine
    std::string solution_out;
    std::string eval_solution;         // --eval-solution: QPLIB .sol to check
    sor::search::GlobalQpOptions global_opts;   // --engine global
    // --qp-opt / --qcqp-opt / --miqp-opt are collected here and applied where
    // each engine's options are built, AFTER that branch's own defaults, so an
    // explicitly given option always wins over a built-in choice.
    std::vector<std::string> qp_opt_args, qcqp_opt_args, miqp_opt_args;
    int lp_concurrent = 1;
    int n_threads = 0;                 // --threads: 0 = sor::core's default
    sor::engines::PdhgOptions pdhg_opts;
    sor::engines::HprOptions hpr_opts;
    sor::engines::SimplexOptions sx_opts;
    // Every route reports a result about the file, so the file must be read
    // completely: an unknown section, or a quadratic objective on a route
    // that cannot use it, is an error rather than a warning. (A QP solved
    // as an LP, then checked by sor_check reading the same way, used to be
    // reported VERIFIED.)
    sor::io::MpsReadOptions mps_opts;
    mps_opts.strict = true;
    bool mps_format_forced = false;
    bool tol_given = false;
    double lp_gap_tolerance = 0.0;
    bool lp_exact_proof = false;
    bool lp_gap_given = false;
    bool max_iter_given = false;
    int local_starts = 1;   // --starts: multi-start count of the local QCQP solver
    int qp_inner_epoch = 1;   // --qp-inner-epoch: PDHCG-II sparse-Q inner loop batching
    bool local_starts_given = false;
    bool primal_crash_given = false;
    bool lattice_reform = false;
    bool fo_polish = true;
    bool fo_certificates = true;
    bool fo_crossover = true;
    sor::core::LpAutoBudgetSplit auto_budget_split =
        sor::core::LpAutoBudgetSplit::Fo60Crossover25Simplex15;
    // MILP conflict-graph switches. Three independent consumers of one root
    // probing pass, kept separately switchable so a benchmark run can attribute
    // a change to the reduction, the cuts, or the node propagation.
    bool probing = true;
    bool mip_presolve = true;
    bool symmetry = true;
    bool reflection = false;  // experimental; enigma false-Infeasible if on
    bool folding = true;
    bool dual_fix_probe = true;
    bool clique_probe = true;
    bool gf2 = true;
    bool components = true;
    bool implied_int = true;
    bool obbt = true;
    int mip_restarts = -1;  // <0 keeps library default
    bool feasibility_jump = true;
    bool sub_mip_lns = true;
    bool balans = true;
    bool kernel_pump = true;
    bool mrens = true;
    bool btbs = true;
    bool cl_tlns = true;
    double kernel_pump_time = -1.0;
    double mrens_time = -1.0;
    double btbs_time = -1.0;
    double cl_tlns_time = -1.0;
    double heuristic_budget = -1.0;   // <0 keeps the library default
    bool clique_cuts = false;   // opt-in: see BabOptions::clique_cuts
    bool implied_bound_cuts = true;
    bool lifted_cover_cuts = false;   // opt-in: see BabOptions
    bool mir_cuts = false;
    bool mir_aggregate = true;
    bool mir_variable_bounds = true;
    bool mir_probe_bounds = false;
    bool mir_lifted_cover = false;
    bool node_lp_cutoff = true;
    bool legacy_branching = false;
    bool milp_presolve = true;
    bool coef_strengthening = true;
    bool root_restart = true;
    bool fpump = true;
    double fpump_time = -1.0;
    bool event_propagation = true;
    bool plunge_all = false;
    int plunge_depth = -1;
    bool objective_face = true;
    bool binary_row_support = true;
    bool structural_fbbt = true;
    bool monotone_binary_pairs = true;
    bool structural_row_probing = false;
    bool structural_graph_relations = false;
    bool structural_graph_support = false;
    double structural_probe_time = -1.0;
    bool diverse_row_probing = false;
    bool rc_strengthening = true;
    std::string events_out;
    double root_reduction_cap = -1.0;
    double root_reduction_share = -1.0;
    double root_cut_share = -1.0;
    double root_cut_max = -1.0;
    double cut_nnz_budget = -1.0;    // <0 keeps the library default
    double cut_max_density = -1.0;
    double cut_par_penalty = -1.0;
    double cut_extra_scores = -1.0;
    bool conflict_propagation = true;
    bool conflict_cut = true;  // default-on; auto-off on dense pure-binary
    bool conflict_cut_paper = false;
    bool nogood_cuts = true;
    bool dynsep = true;
    double dynsep_ucb = -1.0;
    int dynsep_max_optional = -1;
    std::string dynsep_backend;
    std::string dynsep_model;
    bool dynsep_collect = false;
    bool l2sep = true;
    std::string l2sep_model;
    bool hgtsm = true;
    std::string hgtsm_model;
    bool hgtsm_linear = false;
    std::string hgtsm_sequence = "transformer";
    bool hgtsm_collect = false;
    bool gcs = true;
    std::string gcs_model;
    bool gcs_heuristic = false;
    int gcs_reinject = -1;
    bool basis_update_explicit = false;
    std::string milp_policy = "latest";
    bool fixprop_enabled = true;
    double fixprop_time_s = 0.0;
    bool fixprop_time_given = false;
    double feasjump_time_s = 0.0;
    bool feasjump_time_given = false;
    double feasjump_root_frac = 0.0;
    bool feasjump_root_frac_given = false;
    // BatchLP strong branching / OBBT are ON in the library but OFF here: the
    // batched node LPs cost more per node than they save on the MIPLIB-easy
    // set (blend2 2.3x, mod008 1.75x), so the CLI is the opt-in gate.
    bool batch_lp_sb = false;
    bool batch_lp_obbt = false;
    // GPU-track G1: off by default, an unmeasured new code path (see
    // BabDiagnostics::gpu_bin_*).
    bool gpu_binary_heuristic = false;
    // Para-B&B worker count. 0 = auto (min(8, cores)) under the Latest policy,
    // 1 = serial tree. The library default is 1; the CLI hands 0 through so a
    // default run uses the parallel tree, as the pre-squash branch did.
    int bab_threads = 0;
    // Multi-arm portfolio (sor/search/portfolio.hpp): several diversified
    // BabOptions configurations raced against each other, sharing incumbents
    // through a validated pool. Orthogonal to --bab-threads (intra-search
    // node-level parallelism within ONE arm). 0 = flag not given (off);
    // otherwise the requested worker count (0 passed to PortfolioOptions
    // itself means "one arm per hardware thread", so this CLI's 0 and the
    // library's 0 are deliberately different things -- see the parse site).
    int milp_portfolio_workers = 0;
    bool milp_portfolio = false;
    double mip_gap_tol = 1e-4;
    double cut_min_progress = -1.0;  // <0 keeps the library default
    bool cut_rollback = false;
    bool cut_purge = false;
    bool cut_warm_rounds = true;
    bool cut_marginal_gate = false;
    double cut_marginal_min = -1.0;
    long long cut_patience = -1;     // <0 keeps the library default
    int cut_max_rounds = -1;         // <0 keeps the library default
    bool auto_cuts = true;
    bool gmi_cmir_recovery = false;
    bool relax_small_terms = false;
    bool tableau_cmir = false;
    bool node_cut_resolve = true;
    int local_cut_rows = 0;
    bool conflict_store = true;
    std::size_t conflict_store_max_len = 64;
    bool spp_repair_on = true;
    bool trace_cuts = false;
    bool sub_mip_context = true;
    bool root_primal_early = true;
    bool root_primal_final = true;
    bool carry_probing = true;
    bool cut_transaction = true;
    bool farkas_conflicts = true;
    int trace_branching = 0;
    int rb_threshold = -1, rb_max_probed = -1, rb_lookahead = -1;
    double rb_time_share = -1.0;
    bool root_cuts = true;
    bool component_solve = true;
    int gmi_tableau_trials = 0;
    bool rank_gmi = false;
    bool integer_slack_gmi = true;
    bool integer_slack_gmi_basic = false;
    std::string verify_cuts_path;    // reference .sol for the cut-validity check
    std::string branch_strategy = "auto";
    std::string sparse_sb_model;
    bool sparse_sb_collect = false;
    std::string sc_milp_model;
    bool sc_milp_collect = false;
    std::string lifted_expert;
    std::string planbb_policy;
    std::string planbb_model;
    bool planbb_paper = false;
    bool planbb_collect = false;
    int planbb_mcts_sims = -1;
    int planbb_mcts_depth = -1;
    bool planbb_mcts = true;
    double tol = 0.0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (a == "--backend")  backend_name = next("--backend");
        else if (a == "--engine")   engine_name  = next("--engine");
        else if (a == "--bq-search-only") bq_search_only = true;
        else if (a == "--qcr-shift") qcr_shift_arg = next("--qcr-shift");
        else if (a == "--bq-node") bq_node_arg = next("--bq-node");
        else if (a == "--bq-strong") {
            const double v = parse_real(next("--bq-strong"), "--bq-strong", 0.0);
            if (v < 0.0 || v > 256.0 || v != std::floor(v)) {
                std::fprintf(stderr, "error: --bq-strong must be an integer in [0, 256]\n");
                return 2;
            }
            bq_strong = static_cast<std::uint32_t>(v);
        }
        else if (a == "--bq-batch") {
            const double v = parse_real(next("--bq-batch"), "--bq-batch", 1.0);
            if (v < 1.0 || v > 4096.0 || v != std::floor(v)) {
                std::fprintf(stderr, "error: --bq-batch must be an integer in [1, 4096]\n");
                return 2;
            }
            bq_batch = static_cast<std::uint32_t>(v);
        }
        else if (a == "--bq-searches") {
            const double v = parse_real(next("--bq-searches"), "--bq-searches", 1.0);
            if (v < 1.0 || v > 65536.0 || v != std::floor(v)) {
                std::fprintf(stderr, "error: --bq-searches must be an integer in [1, 65536]\n");
                return 2;
            }
            bq_searches = static_cast<std::uint32_t>(v);
        }
        else if (a == "--q-diag")   q_diag_arg   = next("--q-diag");
        else if (a == "--eval-solution") eval_solution = next("--eval-solution");
        else if (a == "--global-relax") {
            const std::string r = next("--global-relax");
            if (r == "auto") global_opts.relaxation = sor::search::GlobalRelaxation::Auto;
            else if (r == "mccormick") global_opts.relaxation = sor::search::GlobalRelaxation::McCormick;
            else if (r == "shift") global_opts.relaxation = sor::search::GlobalRelaxation::Shift;
            else if (r == "both") global_opts.relaxation = sor::search::GlobalRelaxation::Both;
            else {
                std::fprintf(stderr, "error: --global-relax must be auto|mccormick|shift|both\n");
                return 2;
            }
        }
        else if (a == "--global-no-rlt") global_opts.rlt = false;
        else if (a == "--global-opt") {
            std::string gerr;
            if (!sor::search::set_global_option(global_opts, next("--global-opt"), gerr)) {
                std::fprintf(stderr, "error: %s\n", gerr.c_str());
                return 2;
            }
        }
        // Validated here against a default struct, then applied for real where
        // the engine's options are built. Checking twice is deliberate: a typo
        // should fail before the solver prints a banner and starts work, not
        // after.
        else if (a == "--qp-opt") {
            const std::string kv = next("--qp-opt");
            sor::engines::QpOptions probe;
            std::string e;
            if (!sor::core::apply_option(sor::engines::qp_option_bindings(probe), kv, e)) {
                std::fprintf(stderr, "error: --qp-opt %s\n", e.c_str());
                return 2;
            }
            qp_opt_args.push_back(kv);
        }
        else if (a == "--qcqp-opt") {
            const std::string kv = next("--qcqp-opt");
            sor::engines::QcqpLocalOptions probe;
            std::string e;
            if (!sor::core::apply_option(sor::engines::qcqp_local_option_bindings(probe), kv, e)) {
                std::fprintf(stderr, "error: --qcqp-opt %s\n", e.c_str());
                return 2;
            }
            qcqp_opt_args.push_back(kv);
        }
        else if (a == "--miqp-opt") {
            const std::string kv = next("--miqp-opt");
            sor::search::MiqpBbOptions probe;
            std::string e;
            if (!sor::core::apply_option(sor::search::miqp_bb_option_bindings(probe), kv, e)) {
                std::fprintf(stderr, "error: --miqp-opt %s\n", e.c_str());
                return 2;
            }
            miqp_opt_args.push_back(kv);
        }
        else if (a == "--list-opts") {
            print_engine_options(i + 1 < argc ? argv[i + 1] : "");
            return 0;
        }
        else if (a == "--global-no-obbt") global_opts.obbt_max_vars = 0;
        else if (a == "--global-no-local") global_opts.local_search = false;
        else if (a == "--max-iter") {
            const auto n = parse_uint(next("--max-iter"), "--max-iter", 1,
                                      std::numeric_limits<std::uint64_t>::max());
            pdhg_opts.max_iterations = n;
            hpr_opts.max_iterations  = n;
            sx_opts.max_iterations   = n;
            max_iter_given = true;
        }
        else if (a == "--starts") {
            local_starts = static_cast<int>(parse_real(next("--starts"), "--starts", 1.0, 1e6, false));
            local_starts_given = true;
        }
        else if (a == "--qp-inner-epoch") {
            qp_inner_epoch = static_cast<int>(parse_real(
                next("--qp-inner-epoch"), "--qp-inner-epoch", 1.0, 1e6, false));
        }
        else if (a == "--tol") {
            tol = parse_real(next("--tol"), "--tol", 0.0,
                             std::numeric_limits<double>::max(), true);
            tol_given = true;
        }
        else if (a == "--lp-gap-tol") {
            lp_gap_tolerance = parse_real(next("--lp-gap-tol"), "--lp-gap-tol", 0.0,
                std::numeric_limits<double>::max(), true);
            lp_gap_given = true;
        }
        else if (a == "--no-dual-perturbation") sx_opts.dual_perturbation = false;
        else if (a == "--no-primal-bound-perturbation") sx_opts.primal_bound_perturbation = false;
        else if (a == "--trace-lp") sx_opts.trace_degeneracy = true;
        else if (a == "--dse-weight-floor")
            sx_opts.dse_weight_floor = parse_real(next("--dse-weight-floor"),
                                                  "--dse-weight-floor", 0.0, 1.0);
        else if (a == "--dual-perturbation-at-start")
            sx_opts.dual_perturbation_at_start = true;
        else if (a == "--mip-gap")
            mip_gap_tol = parse_real(next("--mip-gap"), "--mip-gap", 0.0, 1.0);
        else if (a == "--time-limit") {
            const double t = parse_real(next("--time-limit"), "--time-limit", 0.0,
                                        std::numeric_limits<double>::max(), true);
            sx_opts.time_limit_s = t;
            hpr_opts.time_limit_s = t;
            pdhg_opts.time_limit_s = t;
            // PDHG/HPR's max_iterations is a fixed default (100000/200000),
            // not the "0 = auto-scale to problem size" convention the
            // simplex engines use -- so with a time budget also given, an
            // UNCHANGED default iteration cap was usually the thing that
            // actually stopped the loop (100000 PDHG iterations run in well
            // under a second on most Netlib instances), silently discarding
            // most of the requested time budget. When a time limit is
            // explicitly requested, let it be the real constraint.
            if (!max_iter_given) {
                pdhg_opts.max_iterations = std::numeric_limits<std::uint64_t>::max();
                hpr_opts.max_iterations = std::numeric_limits<std::uint64_t>::max();
            }
        }
        else if (a == "--method") {
            const std::string m = next("--method");
            if      (m == "auto")   sx_opts.method = sor::engines::SimplexMethod::Auto;
            else if (m == "primal") sx_opts.method = sor::engines::SimplexMethod::Primal;
            else if (m == "dual")   sx_opts.method = sor::engines::SimplexMethod::Dual;
            else { std::fprintf(stderr, "error: unknown method '%s'\n", m.c_str()); return 2; }
        }
        else if (a == "--pricing") {
            const std::string pricing = next("--pricing");
            if (pricing == "choose")
                sx_opts.pricing = sor::engines::SimplexPricing::Choose;
            else if (pricing == "dantzig")
                sx_opts.pricing = sor::engines::SimplexPricing::Dantzig;
            else if (pricing == "devex")
                sx_opts.pricing = sor::engines::SimplexPricing::Devex;
            else if (pricing == "dse")
                sx_opts.pricing = sor::engines::SimplexPricing::DSE;
            else {
                std::fprintf(stderr, "error: unknown simplex pricing '%s'\n",
                             pricing.c_str());
                return 2;
            }
        }
        else if (a == "--basis-update") {
            const std::string method = next("--basis-update");
            basis_update_explicit = true;
            if (method == "product")
                sx_opts.update_method = sor::la::UpdateMethod::ProductForm;
            else if (method == "ft")
                sx_opts.update_method = sor::la::UpdateMethod::ForrestTomlin;
            else {
                std::fprintf(stderr, "error: unknown basis update '%s'\n", method.c_str());
                return 2;
            }
        }
        else if (a == "--collective-ft") sx_opts.collective_ft = true;
        else if (a == "--refactor-interval")
            sx_opts.refactor_interval = static_cast<int>(parse_uint(
                next("--refactor-interval"), "--refactor-interval", 0,
                static_cast<unsigned long long>(std::numeric_limits<int>::max())));
        else if (a == "--refactor-eta-ratio")
            sx_opts.refactor_eta_ratio =
                parse_real(next("--refactor-eta-ratio"), "--refactor-eta-ratio", 0.0);
        else if (a == "--refactor-work-ratio")
            sx_opts.refactor_work_ratio =
                parse_real(next("--refactor-work-ratio"), "--refactor-work-ratio", 0.0);
        else if (a == "--refactor-u-nnz-ratio")
            sx_opts.refactor_u_nnz_ratio =
                parse_real(next("--refactor-u-nnz-ratio"),
                           "--refactor-u-nnz-ratio", 0.0);
        else if (a == "--ft-update-limit")
            sx_opts.ft_update_limit = static_cast<int>(parse_uint(
                next("--ft-update-limit"), "--ft-update-limit", 0,
                static_cast<unsigned long long>(std::numeric_limits<int>::max())));
        else if (a == "--dual-resync-interval")
            sx_opts.dual_resync_interval = static_cast<int>(parse_uint(
                next("--dual-resync-interval"), "--dual-resync-interval", 0,
                static_cast<unsigned long long>(std::numeric_limits<int>::max())));
        else if (a == "--exact-proof") lp_exact_proof = true;
        else if (a == "--cost-shifts") sx_opts.allow_cost_shifts = true;
        else if (a == "--no-cost-shifts") sx_opts.allow_cost_shifts = false;
        else if (a == "--dual-cost-perturbation")
            sx_opts.dual_cost_perturbation_multiplier = parse_real(
                next("--dual-cost-perturbation"), "--dual-cost-perturbation", 0.0);
        else if (a == "--primal-crash") {
            sx_opts.primal_crash = true;
            primal_crash_given = true;
        }
        else if (a == "--no-primal-crash") {
            sx_opts.primal_crash = false;
            primal_crash_given = true;
        }
        else if (a == "--pow2-scaling") sx_opts.ruiz_power_of_two = true;
        else if (a == "--dual-crash") sx_opts.dual_crash = true;
        else if (a == "--no-dual-crash") sx_opts.dual_crash = false;
        else if (a == "--lp-parallel-basis") sx_opts.parallel_basis_solves = true;
        else if (a == "--lp-domain-probing") sx_opts.presolve_domain_probing = true;
        else if (a == "--no-lp-sparsification") sx_opts.presolve_equation_sparsification = false;
        else if (a == "--no-scaling") {
            sx_opts.ruiz_iterations = 0;
            pdhg_opts.ruiz_iterations = 0;
            hpr_opts.ruiz_iterations = 0;
        }
        else if (a == "--no-presolve") sx_opts.presolve = false;
        else if (a == "--relax-integrality") mps_opts.relax_integrality = true;
        else if (a == "--small-matrix-value")
            mps_opts.small_matrix_value = parse_real(
                next("--small-matrix-value"), "--small-matrix-value", 0.0);
        else if (a == "--fo-polish") {
            fo_polish = true;
            hpr_opts.use_polishing = true;
        }
        else if (a == "--no-fo-polish") {
            fo_polish = false;
            hpr_opts.use_polishing = false;
        }
        else if (a == "--fo-certificates") {
            fo_certificates = true;
            hpr_opts.detect_certificates = true;
        }
        else if (a == "--no-fo-certificates") {
            fo_certificates = false;
            hpr_opts.detect_certificates = false;
        }
        else if (a == "--fo-crossover") fo_crossover = true;
        else if (a == "--no-fo-crossover") fo_crossover = false;
        else if (a == "--auto-budget-split") {
            const std::string split = next("--auto-budget-split");
            if (split == "60/25/15")
                auto_budget_split =
                    sor::core::LpAutoBudgetSplit::Fo60Crossover25Simplex15;
            else if (split == "70/20/10")
                auto_budget_split =
                    sor::core::LpAutoBudgetSplit::Fo70Crossover20Simplex10;
            else if (split == "80/15/5")
                auto_budget_split =
                    sor::core::LpAutoBudgetSplit::Fo80Crossover15Simplex05;
            else {
                std::fprintf(stderr,
                    "error: --auto-budget-split expects 60/25/15, "
                    "70/20/10, or 80/15/5, got '%s'\n", split.c_str());
                return 2;
            }
        }
        else if (a == "--implied-slack") sx_opts.presolve_implied_slack = true;
        else if (a == "--lattice-reform") lattice_reform = true;
        else if (a == "--no-probing") probing = false;
        else if (a == "--no-mip-presolve") mip_presolve = false;
        else if (a == "--no-binary-row-support") binary_row_support = false;
        else if (a == "--no-structural-fbbt") structural_fbbt = false;
        else if (a == "--no-monotone-binary-pairs") monotone_binary_pairs = false;
        else if (a == "--structural-row-probing") structural_row_probing = true;
        else if (a == "--structural-graph-relations") structural_graph_relations = true;
        else if (a == "--structural-graph-support") structural_graph_support = true;
        else if (a == "--diverse-row-probing") diverse_row_probing = true;
        else if (a == "--structural-probe-time")
            structural_probe_time = parse_real(next("--structural-probe-time"), "--structural-probe-time", 0.01, 60.0);
        else if (a == "--no-symmetry") symmetry = false;
        else if (a == "--reflection") reflection = true;
        else if (a == "--no-reflection") reflection = false;
        else if (a == "--no-folding") folding = false;
        else if (a == "--no-dual-fix-probe") dual_fix_probe = false;
        else if (a == "--no-clique-probe") clique_probe = false;
        else if (a == "--no-gf2") gf2 = false;
        else if (a == "--no-components") components = false;
        else if (a == "--no-implied-int") implied_int = false;
        else if (a == "--no-obbt") obbt = false;
        else if (a == "--mip-restarts")
            mip_restarts = static_cast<int>(
                parse_real(next("--mip-restarts"), "--mip-restarts", 0.0));
        else if (a == "--no-feasjump") feasibility_jump = false;
        else if (a == "--no-sub-mip") sub_mip_lns = false;
        else if (a == "--no-balans") balans = false;
        else if (a == "--no-kernel-pump") kernel_pump = false;
        else if (a == "--no-mrens") mrens = false;
        else if (a == "--no-btbs") btbs = false;
        else if (a == "--no-cl-tlns") cl_tlns = false;
        else if (a == "--kernel-pump-time")
            kernel_pump_time = parse_real(next("--kernel-pump-time"),
                                          "--kernel-pump-time", 0.0);
        else if (a == "--mrens-time")
            mrens_time = parse_real(next("--mrens-time"), "--mrens-time", 0.0);
        else if (a == "--btbs-time")
            btbs_time = parse_real(next("--btbs-time"), "--btbs-time", 0.0);
        else if (a == "--cl-tlns-time")
            cl_tlns_time =
                parse_real(next("--cl-tlns-time"), "--cl-tlns-time", 0.0);
        else if (a == "--heuristic-budget")
            heuristic_budget = parse_real(next("--heuristic-budget"),
                                         "--heuristic-budget", 0.0, 1.0);
        else if (a == "--clique-cuts") clique_cuts = true;
        else if (a == "--no-vub-cuts") implied_bound_cuts = false;
        else if (a == "--cover-cuts") lifted_cover_cuts = true;
        else if (a == "--mir-cuts") mir_cuts = true;
        else if (a == "--gmi-cmir-recovery") gmi_cmir_recovery = true;
        else if (a == "--no-gmi-cmir-recovery") gmi_cmir_recovery = false;
        else if (a == "--relax-small-terms") relax_small_terms = true;
        else if (a == "--no-relax-small-terms") relax_small_terms = false;
        else if (a == "--tableau-cmir") tableau_cmir = true;
        else if (a == "--no-tableau-cmir") tableau_cmir = false;
        else if (a == "--no-node-cut-resolve") node_cut_resolve = false;
        else if (a == "--no-conflict-store") conflict_store = false;
        else if (a == "--conflict-store-max-len")
            conflict_store_max_len = static_cast<std::size_t>(parse_uint(next(a.c_str()), a.c_str(), 1, 4096));
        else if (a == "--no-spp-repair") spp_repair_on = false;
        else if (a == "--trace-cuts") trace_cuts = true;
        else if (a == "--no-sub-mip-context") sub_mip_context = false;
        else if (a == "--no-root-primal-early") root_primal_early = false;
        else if (a == "--no-root-primal-final") root_primal_final = false;
        else if (a == "--no-carry-probing") carry_probing = false;
        else if (a == "--no-cut-transaction") cut_transaction = false;
        else if (a == "--no-farkas-conflicts") farkas_conflicts = false;
        else if (a == "--rb-threshold") rb_threshold = static_cast<int>(parse_uint(next(a.c_str()), a.c_str(), 0, 1000000));
        else if (a == "--rb-max-probed") rb_max_probed = static_cast<int>(parse_uint(next(a.c_str()), a.c_str(), 0, 1000000));
        else if (a == "--rb-lookahead") rb_lookahead = static_cast<int>(parse_uint(next(a.c_str()), a.c_str(), 0, 1000000));
        else if (a == "--rb-time-share" || a == "--rb-sb-share") rb_time_share = parse_real(next(a.c_str()), a.c_str(), 0.0, 1.0);
        else if (a == "--trace-branching") trace_branching = (i + 1 < argc && argv[i + 1][0] != '-') ? static_cast<int>(parse_uint(next(a.c_str()), a.c_str(), 0, 1000000)) : 256;
        else if (a == "--no-root-cuts") root_cuts = false;
        else if (a == "--no-component-solve") component_solve = false;
        else if (a == "--local-cut-rows") local_cut_rows = static_cast<int>(parse_uint(next(a.c_str()), a.c_str(), 0, 1000000));
        else if (a == "--gmi-tableau-trials")
            gmi_tableau_trials = static_cast<int>(parse_uint(next("--gmi-tableau-trials"), "--gmi-tableau-trials", 0, 1000000));
        else if (a == "--rank-gmi") rank_gmi = true;
        else if (a == "--integer-slack-gmi") integer_slack_gmi = true;
        else if (a == "--no-integer-slack-gmi") integer_slack_gmi = false;
        else if (a == "--integer-slack-gmi-basic") {
            integer_slack_gmi = true;
            integer_slack_gmi_basic = true;
        }
        else if (a == "--no-aggregation") mir_aggregate = false;
        else if (a == "--no-mir-variable-bounds") mir_variable_bounds = false;
        else if (a == "--mir-probe-bounds") mir_probe_bounds = true;
        else if (a == "--mir-lifted-cover") mir_lifted_cover = true;
        else if (a == "--legacy-branching") legacy_branching = true;
        else if (a == "--no-coef-strengthening") coef_strengthening = false;
        else if (a == "--no-milp-presolve") milp_presolve = false;
        else if (a == "--no-root-restart") root_restart = false;
        else if (a == "--no-objective-face") objective_face = false;
        else if (a == "--no-fpump") fpump = false;
        else if (a == "--fpump-time")
            fpump_time = parse_real(next("--fpump-time"), "--fpump-time", 0.0, 1e6);
        else if (a == "--no-event-propagation") event_propagation = false;
        else if (a == "--plunge") plunge_all = true;
        else if (a == "--plunge-depth")
            plunge_depth = static_cast<int>(parse_real(next("--plunge-depth"), "--plunge-depth", 1.0, 1000.0));
        else if (a == "--paper-node-selection") {
            std::fprintf(stderr, "error: --paper-node-selection is unavailable: "
                         "estimate-driven node selection is not implemented "
                         "(see --milp-capabilities)\n");
            return 2;
        }
        else if (a == "--no-rc-strengthening") rc_strengthening = false;
        else if (a == "--no-node-lp-cutoff") node_lp_cutoff = false;
        else if (a == "--legacy-heuristic-budget" ||
                 a == "--no-bounded-dives" || a == "--no-dive-rankings") {
            std::fprintf(stderr, "error: %s is unavailable (see --milp-capabilities)\n",
                         a.c_str());
            return 2;
        }
        else if (a == "--events-out") events_out = next("--events-out");
        else if (a == "--root-reduction-cap")
            root_reduction_cap = parse_real(next("--root-reduction-cap"), "--root-reduction-cap", 0.0, 1e9);
        else if (a == "--root-cut-share")
            root_cut_share = parse_real(next("--root-cut-share"), "--root-cut-share", 0.0, 1.0);
        else if (a == "--root-cut-max")
            root_cut_max = parse_real(next("--root-cut-max"), "--root-cut-max", 0.0, 1e9);
        else if (a == "--root-reduction-share")
            root_reduction_share = parse_real(next("--root-reduction-share"),
                                              "--root-reduction-share", 0.0, 1.0);
        else if (a == "--milp-capabilities") {
            for (const auto& c : sor::search::milp_capability_inventory())
                std::printf("%-48s %-12s %s\n", c.name,
                            sor::search::capability_status_name(c.status), c.note);
            return 0;
        }
        else if (a == "--cut-nnz-budget")
            cut_nnz_budget = parse_real(next("--cut-nnz-budget"), "--cut-nnz-budget", 0.0);
        else if (a == "--cut-max-density")
            cut_max_density = parse_real(next("--cut-max-density"), "--cut-max-density", 0.0);
        else if (a == "--cut-extra-scores")
            cut_extra_scores = parse_real(next("--cut-extra-scores"), "--cut-extra-scores", 0.0);
        else if (a == "--cut-parallel-penalty")
            cut_par_penalty = parse_real(next("--cut-parallel-penalty"), "--cut-parallel-penalty", 0.0);
        else if (a == "--no-conflict-prop") conflict_propagation = false;
        else if (a == "--conflict-cut") conflict_cut = true;
        else if (a == "--no-conflict-cut") conflict_cut = false;
        else if (a == "--conflict-cut-paper") conflict_cut_paper = true;
        else if (a == "--no-nogood-cuts") nogood_cuts = false;
        else if (a == "--no-dynsep") dynsep = false;
        else if (a == "--dynsep-backend")
            dynsep_backend = next("--dynsep-backend");
        else if (a == "--dynsep-model")
            dynsep_model = next("--dynsep-model");
        else if (a == "--dynsep-collect") dynsep_collect = true;
        else if (a == "--dynsep-ucb")
            dynsep_ucb = parse_real(next("--dynsep-ucb"), "--dynsep-ucb", 0.0);
        else if (a == "--dynsep-max-optional")
            dynsep_max_optional = static_cast<int>(
                parse_uint(next("--dynsep-max-optional"), "--dynsep-max-optional",
                           0, 16));
        else if (a == "--no-l2sep") l2sep = false;
        else if (a == "--l2sep-model") l2sep_model = next("--l2sep-model");
        else if (a == "--no-hgtsm") hgtsm = false;
        else if (a == "--hgtsm-model") hgtsm_model = next("--hgtsm-model");
        else if (a == "--hgtsm-linear") hgtsm_linear = true;
        else if (a == "--hgtsm-sequence")
            hgtsm_sequence = next("--hgtsm-sequence");
        else if (a == "--hgtsm-collect") hgtsm_collect = true;
        else if (a == "--no-gcs") gcs = false;
        else if (a == "--gcs-model") gcs_model = next("--gcs-model");
        else if (a == "--gcs-heuristic") gcs_heuristic = true;
        else if (a == "--gcs-reinject")
            gcs_reinject = static_cast<int>(
                parse_uint(next("--gcs-reinject"), "--gcs-reinject", 0,
                           1000000));
        else if (a == "--milp-policy") milp_policy = next("--milp-policy");
        else if (a == "--no-fixprop") fixprop_enabled = false;
        else if (a == "--fixprop-time") {
            fixprop_time_s = parse_real(next("--fixprop-time"),
                                        "--fixprop-time", 0.0);
            fixprop_time_given = true;
        }
        else if (a == "--feasjump-time") {
            feasjump_time_s = parse_real(next("--feasjump-time"),
                                         "--feasjump-time", 0.0);
            feasjump_time_given = true;
        }
        else if (a == "--feasjump-root-frac") {
            feasjump_root_frac = parse_real(next("--feasjump-root-frac"),
                                            "--feasjump-root-frac", 0.0, 1.0,
                                            true);
            feasjump_root_frac_given = true;
        }
        else if (a == "--bab-threads")
            bab_threads = static_cast<int>(
                parse_uint(next("--bab-threads"), "--bab-threads", 0, 1024));
        else if (a == "--milp-portfolio") {
            milp_portfolio = true;
            milp_portfolio_workers = static_cast<int>(parse_uint(
                next("--milp-portfolio"), "--milp-portfolio", 0, 1024));
        }
        else if (a == "--batch-lp-sb") batch_lp_sb = true;
        else if (a == "--no-batch-lp-sb") batch_lp_sb = false;
        else if (a == "--gpu-binary-heuristic") gpu_binary_heuristic = true;
        else if (a == "--no-gpu-binary-heuristic") gpu_binary_heuristic = false;
        else if (a == "--batch-lp-obbt") {
            std::fprintf(stderr,
                "error: --batch-lp-obbt is disabled: approximate primal "
                "objectives cannot certify bound tightening\n");
            return 2;
        }
        else if (a == "--no-batch-lp-obbt") batch_lp_obbt = false;
        else if (a == "--auto-cuts") auto_cuts = true;
        else if (a == "--no-auto-cuts") auto_cuts = false;
        else if (a == "--verify-cuts") verify_cuts_path = next("--verify-cuts");
        else if (a == "--cut-rollback") cut_rollback = true;
        else if (a == "--cut-purge") cut_purge = true;
        else if (a == "--cut-warm-rounds") cut_warm_rounds = true;
        else if (a == "--no-cut-warm-rounds") cut_warm_rounds = false;
        else if (a == "--cut-marginal-gate") cut_marginal_gate = true;
        // no separate warning path: both spellings warn at setup below.
        else if (a == "--cut-marginal-min") {
            cut_marginal_gate = true;
            cut_marginal_min = parse_real(next("--cut-marginal-min"),
                                          "--cut-marginal-min", 0.0, 1.0, true);
        }
        else if (a == "--cut-patience") {
            cut_patience = static_cast<long long>(parse_uint(
                next("--cut-patience"), "--cut-patience", 1, 1000));
        }
        else if (a == "--cut-min-progress") {
            cut_min_progress = parse_real(next("--cut-min-progress"),
                                          "--cut-min-progress", 0.0, 1.0, true);
        }
        else if (a == "--cut-max-rounds") {
            cut_max_rounds = static_cast<int>(
                parse_uint(next("--cut-max-rounds"), "--cut-max-rounds", 1,
                           1000000));
        }
        else if (a == "--branch-strategy")
            branch_strategy = next("--branch-strategy");
        else if (a == "--sparse-sb-model")
            sparse_sb_model = next("--sparse-sb-model");
        else if (a == "--sparse-sb-collect") sparse_sb_collect = true;
        else if (a == "--sc-milp-model")
            sc_milp_model = next("--sc-milp-model");
        else if (a == "--sc-milp-collect") sc_milp_collect = true;
        else if (a == "--lifted-expert")
            lifted_expert = next("--lifted-expert");
        else if (a == "--planbb-policy")
            planbb_policy = next("--planbb-policy");
        else if (a == "--planbb-model")
            planbb_model = next("--planbb-model");
        else if (a == "--planbb-paper") planbb_paper = true;
        else if (a == "--planbb-collect") planbb_collect = true;
        else if (a == "--planbb-mcts-sims")
            planbb_mcts_sims = static_cast<int>(
                parse_uint(next("--planbb-mcts-sims"), "--planbb-mcts-sims",
                           1, 100000));
        else if (a == "--planbb-mcts-depth")
            planbb_mcts_depth = static_cast<int>(
                parse_uint(next("--planbb-mcts-depth"), "--planbb-mcts-depth",
                           1, 32));
        else if (a == "--no-planbb-mcts") planbb_mcts = false;
        else if (a == "--fixed-mps") { mps_opts.fixed_format = true; mps_format_forced = true; }
        else if (a == "--free-mps")  { mps_opts.fixed_format = false; mps_format_forced = true; }
        else if (a == "--debug-routes" || a.rfind("--debug-routes=", 0) == 0) {
            std::string value = "1";
            if (a.rfind("--debug-routes=", 0) == 0) value = a.substr(std::strlen("--debug-routes="));
            else if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
                value = next("--debug-routes");
            sor::core::route_debug_set_level(static_cast<int>(parse_uint(
                value, "--debug-routes", 0, 3)));
        }
        else if (a == "--debug-routes-file" || a.rfind("--debug-routes-file=", 0) == 0) {
            const std::string route_file =
                a.rfind("--debug-routes-file=", 0) == 0
                    ? a.substr(std::strlen("--debug-routes-file="))
                    : next("--debug-routes-file");
            sor::core::route_debug_set_file(route_file.c_str());
        }
        else if (a == "--debug-routes-comp" || a.rfind("--debug-routes-comp=", 0) == 0) {
            const std::string comp_list =
                a.rfind("--debug-routes-comp=", 0) == 0
                    ? a.substr(std::strlen("--debug-routes-comp="))
                    : next("--debug-routes-comp");
            sor::core::route_debug_set_comp_filter(comp_list.c_str());
        }
        else if (a == "--debug-routes-path" || a.rfind("--debug-routes-path=", 0) == 0) {
            const std::string path_list =
                a.rfind("--debug-routes-path=", 0) == 0
                    ? a.substr(std::strlen("--debug-routes-path="))
                    : next("--debug-routes-path");
            sor::core::route_debug_set_path_filter(path_list.c_str());
        }
        else if (a == "--debug-routes-fns") {
            sor::core::route_debug_set_fns(true);
        }
        else if (a == "--debug-routes-fn" || a.rfind("--debug-routes-fn=", 0) == 0) {
            const std::string fn_list =
                a.rfind("--debug-routes-fn=", 0) == 0
                    ? a.substr(std::strlen("--debug-routes-fn="))
                    : next("--debug-routes-fn");
            sor::core::route_debug_set_fns(true);
            sor::core::route_debug_set_fn_filter(fn_list.c_str());
        }
        else if (a == "--pivot-trace-every" || a.rfind("--pivot-trace-every=", 0) == 0) {
            const std::string value =
                a.rfind("--pivot-trace-every=", 0) == 0
                    ? a.substr(std::strlen("--pivot-trace-every="))
                    : next("--pivot-trace-every");
            sor::core::route_debug_set_pivot_every(static_cast<int>(parse_uint(
                value, "--pivot-trace-every", 0,
                static_cast<unsigned long long>(std::numeric_limits<int>::max()))));
        }
        else if (a == "--verbose") {
            pdhg_opts.verbose = true;
            hpr_opts.verbose = true;
            sx_opts.verbose = true;
        }
        else if (a == "--hpr-vanilla") {
            hpr_opts.use_primal_weight = false;
            hpr_opts.use_restart = false;
            hpr_opts.use_halpern = false;
            hpr_opts.use_reflection = false;
            hpr_opts.use_adaptive_step = false;
        }
        else if (a == "--hpr-full") {
            hpr_opts.use_primal_weight = true;
            hpr_opts.use_restart = true;
            hpr_opts.use_halpern = true;
            hpr_opts.use_reflection = true;
            hpr_opts.use_adaptive_step = true;
        }
        else if (a == "--hpr-restart-off") hpr_opts.use_restart = false;
        else if (a == "--hpr-reflection-off") hpr_opts.use_reflection = false;
        else if (a == "--hpr-weight-off") hpr_opts.use_primal_weight = false;
        else if (a == "--hpr-weight-pid")
            hpr_opts.weight_policy = sor::engines::HprOptions::WeightPolicy::Pid;
        else if (a == "--hpr-weight-smoothed")
            hpr_opts.weight_policy = sor::engines::HprOptions::WeightPolicy::Smoothed;
        else if (a == "--lp-concurrent")
            lp_concurrent = static_cast<int>(parse_uint(next("--lp-concurrent"), "--lp-concurrent", 1, 16));
        else if (a == "--threads")
            n_threads = static_cast<int>(parse_uint(next("--threads"), "--threads", 1, 256));
        else if (a == "--solution-out") solution_out = next("--solution-out");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            return 2;
        }
        else path = a;
    }

    if (path.empty()) { usage(); return 2; }
    {
        std::string ierr;
        if (!input_is_usable(path, ierr)) {
            std::fprintf(stderr, "error: %s\n", ierr.c_str());
            return 2;
        }
    }
    // Check --solution-out is writable BEFORE solving. Discovering it after a
    // 60 s solve means the answer is computed and then thrown away. The probe
    // opens in append mode so it cannot truncate a file the caller still wants,
    // and removes what it created.
    if (!solution_out.empty()) {
        std::error_code ec;
        const std::filesystem::path outp(solution_out);
        const auto parent = outp.parent_path();
        if (!parent.empty() && !std::filesystem::exists(parent, ec)) {
            std::fprintf(stderr, "error: --solution-out directory does not exist: %s\n",
                         parent.string().c_str());
            return 2;
        }
        const bool existed = std::filesystem::exists(outp, ec);
        {
            std::ofstream probe(solution_out, std::ios::app);
            if (!probe.good()) {
                std::fprintf(stderr, "error: --solution-out is not writable: %s\n",
                             solution_out.c_str());
                return 2;
            }
        }
        if (!existed) std::filesystem::remove(outp, ec);
    }
#ifndef SOR_ROUTE_DEBUG
    if (sor::core::route_debug_level() > 0 || sor::core::route_debug_fns_on()) {
        std::fprintf(stderr, "error: route tracing is unavailable in this build; rebuild with -DSOR_ROUTE_FN=ON\n");
        return 2;
    }
#endif
    if (engine_name != "pdhg" && engine_name != "simplex" && engine_name != "auto" &&
        engine_name != "primal" && engine_name != "dual" && engine_name != "hpr" &&
        engine_name != "barrier" &&
        engine_name != "milp" && engine_name != "qp" &&
        engine_name != "hprqp" && engine_name != "binquad" && engine_name != "qpipm" &&
        engine_name != "qpauto" && engine_name != "miqp" &&
        engine_name != "global" && engine_name != "qcqplocal") {
        std::fprintf(stderr,
                     "error: engine '%s' not implemented "
                     "(have simplex|auto|primal|dual|pdhg|hpr|barrier|milp|qp|qpipm|qpauto|hprqp|binquad|miqp|global|qcqplocal)\n",
                     engine_name.c_str());
        return 3;
    }
    // Availability and algorithmic capabilities are checked by the selected
    // engine.  An unavailable non-CPU backend is reported as Unsupported; it
    // must never turn a requested claim run into an unlabelled CPU run.
    if (backend_name != "cpu" && backend_name != "vulkan" &&
        backend_name != "cuda") {
        std::fprintf(stderr,
                     "error: unknown backend '%s' (have cpu|vulkan|cuda)\n",
                     backend_name.c_str());
        return 2;
    }
    // The crash has an independent 93/93 Netlib proof gate and improves the
    // LP CLI's aggregate pivot count. Keep SimplexOptions' library default off:
    // MILP owns its node-LP policy separately and must not change merely
    // because the standalone LP default did.
    if ((engine_name == "simplex" || engine_name == "primal" ||
         engine_name == "dual") && !primal_crash_given)
        sx_opts.primal_crash = true;
    if (tol_given) {
        pdhg_opts.primal_tol = pdhg_opts.dual_tol = pdhg_opts.gap_tol = tol;
        hpr_opts.primal_tol = hpr_opts.dual_tol = hpr_opts.gap_tol = tol;
        sx_opts.primal_feas_tol = sx_opts.dual_feas_tol = tol;
        sx_opts.gap_tol = tol;
    }
    if (lp_gap_given) {
        sx_opts.gap_tol = pdhg_opts.gap_tol = hpr_opts.gap_tol = lp_gap_tolerance;
    }

    // The pool is process-wide and sized once, before any solve: sizing it
    // mid-solve would change how a reduction is chunked.  Printed with every
    // run because a timing that does not say how many threads produced it is
    // not a measurement.
    sor::core::set_global_threads(n_threads);
    sx_opts.pricing_threads = sor::core::global_threads();
    std::printf("threads:           %d\n", sor::core::global_threads());

    RouteSession route_session;
    try {
        const bool path_is_qps = path.size() >= 4 &&
            (path.compare(path.size() - 4, 4, ".qps") == 0 ||
             path.compare(path.size() - 4, 4, ".QPS") == 0);

        const bool path_is_qplib = path.size() >= 6 &&
            path.compare(path.size() - 6, 6, ".qplib") == 0;

        if (!eval_solution.empty()) {
            if (!path_is_qplib) {
                std::fprintf(stderr, "error: --eval-solution needs a .qplib model\n");
                return 2;
            }
            return eval_qplib_solution(path, eval_solution, tol_given ? tol : 1e-6);
        }

        // --engine auto on a .qplib: resolve to a concrete engine via
        // qplib_auto_route and fall through the SAME branches below that a
        // user naming that engine directly would hit -- auto is a router,
        // not a seventh solve path. An unroutable classification is refused
        // by status here, the same way an engine that can't read QPLIB is
        // refused further down.
        if (path_is_qplib && engine_name == "auto") {
            const auto cls = qplib_read_classification(path);
            const auto route = qplib_auto_route(cls);
            std::printf("model class:       %c%c%c\n", cls[0], cls[1], cls[2]);
            if (route.engine.empty()) {
                sor::core::RawResult raw;
                raw.proposed_status = sor::core::Status::Unsupported;
                raw.engine = "auto";
                raw.backend = backend_name;
                raw.termination_reason = route.reason;
                const auto r = sor::certify::finalize_result(std::move(raw),
                                                             sor::core::ProofEvidence{});
                print_result(r);
                std::printf("termination:       %s\n", r.termination_reason.c_str());
                return exit_code_for(r.status);
            }
            std::printf("auto engine:       %s\n", route.engine.c_str());
            engine_name = route.engine;
        }

        if (path_is_qplib && engine_name == "qcqplocal") {
            // Local solver for any continuous QCQP (nonconvex rows and
            // objective included).  A LOCAL method: the best it can report
            // is Feasible, after the point is re-evaluated on the RAW file
            // data (rows, bounds, integrality) by io::qplib_evaluate_point.
            sor::io::QplibReadReport qrep;
            const auto q = sor::io::read_qplib_file(path, qrep);
            sor::engines::QcqpProblem qcqp;
            sor::search::qplib_to_qcqp(q, qcqp);
            std::printf("model:             %s\n", q.name.c_str());
            std::printf("rows x cols:       %d x %d   (%c%c%c, %zu quadratic rows)\n", q.m, q.n,
                        q.classification[0], q.classification[1], q.classification[2],
                        qcqp.quad.size());
            std::printf("engine:            qcqp_local (interior point, local)\n");
            sor::engines::QcqpLocalOptions lo;
            lo.time_limit_s = sx_opts.time_limit_s;
            lo.verbose = sx_opts.verbose;
            // Default (no explicit --starts) used to mean exactly ONE start:
            // measured on QPLIB_2834 (LCQ, 156 vars), the single start's own
            // restart-on-stall machinery hit its max_iterations and returned
            // at 40.4 s of a 60 s budget -- 20 s (a third of the run) idle,
            // no second start ever attempted from a different point even
            // though solve_qcqp_local's own multi-start loop (qcqp_local.cpp)
            // already checks the wall-clock deadline before every start and
            // so cannot overrun it. When a real time limit is given, let
            // time -- not this count -- be the limiter: default to effectively
            // unbounded starts so idle budget goes to diversified restarts
            // instead of ending the run early. Kept at 1 when there is no
            // time limit (matches the pre-existing, deliberately-finite
            // default relied on by callers -- tests included -- that invoke
            // this path with no --time-limit and expect it to return).
            lo.starts = local_starts_given         ? local_starts
                        : sx_opts.time_limit_s > 0.0 ? 1000000
                                                      : 1;
            // seed the first start from a McCormick relaxation of
            // the instance's quadratic rows instead of the origin -- one
            // extra LP solve per CLI invocation, affordable here (see
            // QcqpLocalOptions::mccormick_start's comment for why this is
            // NOT the default for miqp_bb.cpp's per-node heuristic calls).
            lo.mccormick_start = true;
            if (tol_given) lo.feas_tol = tol;
            if (max_iter_given)
                lo.max_iterations = static_cast<int>(
                    std::min<std::uint64_t>(sx_opts.max_iterations, 100000000ULL));
            if (!apply_pending_options(sor::engines::qcqp_local_option_bindings(lo), qcqp_opt_args, "--qcqp-opt")) return 2;
            sor::engines::QcqpLocalDiagnostics ld;
            auto raw = sor::engines::solve_qcqp_local(qcqp, lo, ld);
            print_local_result(q, qcqp, raw, ld.reason, lo.feas_tol, solution_out);
            std::printf("starts:            %d run, best %d; %llu iterations\n", ld.starts_run,
                        ld.best_start, static_cast<unsigned long long>(ld.iterations));
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", ld.total_ms);
            return 0;
        }
        // Quadratic constraints: no engine here can represent them yet, and
        // every one of them would otherwise either mis-read the file or solve
        // it with the constraints silently missing (binquad scored rows
        // through A only).  Refuse by status, through finalize_result, BEFORE
        // any engine sees the model.  The header is peeked first so the
        // 150 MB linearly constrained instances are not read twice.
        if (path_is_qplib && qplib_class_has_quadratic_rows(path)) {
            sor::io::QplibReadReport qrep;
            const auto q = sor::io::read_qplib_file(path, qrep);
            sor::engines::QcqpProblem qcqp;
            sor::search::qplib_to_qcqp(q, qcqp);
            if (qcqp.has_quadratic_constraints() &&
                (engine_name == "qpipm" || engine_name == "qpauto")) {
                // Convex QCQP: the interior point with quadratic rows.  It
                // certifies every row convex itself (else Unsupported), and
                // its claim is the original-units KKT check with quadratic
                // rows; the point is then re-evaluated on the RAW file data
                // (io::qplib_evaluate_point, no shared conversion code)
                // before finalize_result sees it.
                std::printf("model:             %s\n", q.name.c_str());
                std::printf("rows x cols:       %d x %d   (%c%c%c, %zu quadratic rows)\n",
                            q.m, q.n, q.classification[0], q.classification[1],
                            q.classification[2], qcqp.quad.size());
                std::printf("engine:            qcqp_ipm\n");
                std::printf("backend:           cpu (interior point, quadratic rows)\n");
                sor::engines::QpOptions qopts;
                qopts.max_iterations = pdhg_opts.max_iterations;
                qopts.time_limit_s = sx_opts.time_limit_s;
                qopts.verbose = sx_opts.verbose;
                if (tol_given) qopts.feas_tol = qopts.stationarity_tol = qopts.gap_tol = tol;
                if (!apply_pending_options(sor::engines::qp_option_bindings(qopts), qp_opt_args, "--qp-opt")) return 2;
                sor::engines::QpDiagnostics diag;
                auto raw = sor::engines::solve_qcqp_ipm(qcqp, qopts, diag);
                auto ev = sor::engines::qp_evidence(diag, qopts);
                double chk_obj = sor::core::kNaN, chk_viol = sor::core::kPosInf;
                if (raw.x.size() == static_cast<std::size_t>(q.n)) {
                    const auto re = sor::io::qplib_evaluate_point(q, raw.x);
                    chk_obj = re.objective;
                    chk_viol = re.max_violation();
                    const double mine = qcqp.objective_negated ? -raw.objective : raw.objective;
                    // The KKT check passed its primal test net of rounding, so
                    // the raw evaluator may see that residual plus ulps -- not
                    // more.
                    const bool agrees =
                        std::fabs(chk_obj - mine) <= 1e-9 * (1.0 + std::fabs(chk_obj)) &&
                        chk_viol <= std::max(qopts.feas_tol, 1.01 * diag.primal_residual);
                    // Gate only: the evidence's violation stays the net one the
                    // claim is made on (the raw value is printed below).
                    ev.checker_passed = ev.checker_passed && agrees;
                }
                if (qcqp.objective_negated) {
                    raw.objective = -raw.objective;
                    raw.dual_bound = -raw.dual_bound;
                }
                const auto r = sor::certify::finalize_result(std::move(raw), ev);
                print_result(r);
                write_solution_out(solution_out, r);
                std::printf("stationarity:      %.3e  (rel %.3e)\n", diag.stationarity,
                            diag.stationarity_rel);
                std::printf("max primal viol:   %.3e  (rel %.3e)\n", diag.primal_residual,
                            diag.primal_residual_rel);
                std::printf("net of rounding:   primal %.3e  stationarity %.3e  gap %.3e\n",
                            diag.primal_net, diag.stationarity_net, diag.gap_net);
                if (diag.worst_solve_residual > 0.0)
                    std::printf("  %-18s %.3e\n", "solve residual:", diag.worst_solve_residual);
                std::printf("relative gap:      %.3e\n", diag.gap_rel);
                std::printf("raw re-check:      obj %.10e  viol %.3e\n", chk_obj, chk_viol);
                std::printf("iterations:        %llu\n",
                            static_cast<unsigned long long>(diag.iterations));
                std::printf("termination:       %s\n", r.termination_reason.c_str());
                std::printf("\ntiming (ms)\n");
                std::printf("  total            %10.3f\n", diag.total_ms);
                return exit_code_for(r.status);
            }
            if (qcqp.has_quadratic_constraints() && engine_name == "miqp") {
                // Mixed-integer convex QCQP: the MIQP branch-and-bound with
                // QCQP interior-point node relaxations.  Rows it cannot
                // certify convex are refused (Unsupported) by the solver
                // itself.  The incumbent is re-checked on the RAW file data
                // -- quadratic rows, bounds and integrality -- before
                // finalize_result sees the claim.
                std::printf("model:             %s\n", q.name.c_str());
                std::printf("rows x cols:       %d x %d   (%c%c%c, %zu quadratic rows)  integer cols %zu\n",
                            q.m, q.n, q.classification[0], q.classification[1],
                            q.classification[2], qcqp.quad.size(), qcqp.qp.linear.n_integer());
                std::printf("engine:            miqcqp_bb (QCQP IPM node relaxations)\n");
                sor::search::MiqpBbOptions mo;
                mo.time_limit_s = sx_opts.time_limit_s;
                mo.verbose = sx_opts.verbose;
                if (max_iter_given) mo.max_nodes = sx_opts.max_iterations;
                if (tol_given) mo.gap_rel = tol;
                if (!apply_pending_options(sor::search::miqp_bb_option_bindings(mo), miqp_opt_args, "--miqp-opt")) return 2;
                sor::search::MiqpBbDiagnostics md;
                auto raw = sor::search::solve_miqcqp_bb(qcqp, mo, md);
                auto ev = sor::search::miqcqp_bb_evidence(qcqp, mo, md, raw);
                double chk_obj = sor::core::kNaN, chk_viol = sor::core::kPosInf,
                       chk_int = sor::core::kPosInf;
                if (raw.x.size() == static_cast<std::size_t>(q.n)) {
                    const auto re = sor::io::qplib_evaluate_point(q, raw.x);
                    chk_obj = re.objective;
                    chk_viol = std::max(re.max_row_violation, re.max_bound_violation);
                    chk_int = re.max_integrality_violation;
                    const double mine = qcqp.objective_negated ? -raw.objective : raw.objective;
                    const bool agrees = chk_int == 0.0 && chk_viol <= mo.feas_tol &&
                        std::fabs(chk_obj - mine) <= 1e-9 * (1.0 + std::fabs(chk_obj));
                    ev.checker_passed = ev.checker_passed && agrees;
                    ev.max_primal_violation = std::max(ev.max_primal_violation, std::max(chk_viol, chk_int));
                }
                if (qcqp.objective_negated) {
                    raw.objective = -raw.objective;
                    raw.dual_bound = -raw.dual_bound;
                }
                const auto r = sor::certify::finalize_result(std::move(raw), ev);
                print_result(r);
                write_solution_out(solution_out, r);
                const bool neg = qcqp.objective_negated;
                std::printf("%s bound:       %.10e\n", neg ? "upper" : "lower",
                            neg ? -md.global_bound : md.global_bound);
                std::printf("root bound:        %.10e\n", neg ? -md.root_bound : md.root_bound);
                std::printf("gap (rel):         %.3e\n", md.gap_rel);
                std::printf("raw re-check:      obj %.10e  viol %.3e  int %.3e\n", chk_obj, chk_viol,
                            chk_int);
                std::printf("sigma (shift):     %.6e\n", md.sigma);
                std::printf("tree:              %llu nodes, %llu pruned by bound, %llu infeasible, "
                            "%llu integral, %llu unproved relaxations, %llu unresolved leaves, "
                            "%llu open, depth %d\n",
                            static_cast<unsigned long long>(md.nodes),
                            static_cast<unsigned long long>(md.pruned_bound),
                            static_cast<unsigned long long>(md.pruned_infeasible),
                            static_cast<unsigned long long>(md.integral_closed),
                            static_cast<unsigned long long>(md.unproved_nodes),
                            static_cast<unsigned long long>(md.unresolved_leaves),
                            static_cast<unsigned long long>(md.open_at_end), md.max_depth);
                std::printf("incumbents:        %llu (heuristic calls %llu)\n",
                            static_cast<unsigned long long>(md.incumbents),
                            static_cast<unsigned long long>(md.heuristic_calls));
                std::printf("termination:       %s\n", r.termination_reason.c_str());
                std::printf("\ntiming (ms)\n");
                std::printf("  total            %10.3f\n", md.total_ms);
                std::printf("  relaxations      %10.3f  (%llu IPM iterations)\n", md.relax_ms,
                            static_cast<unsigned long long>(md.ipm_iterations));
                std::printf("  heuristic        %10.3f\n", md.heuristic_ms);
                return exit_code_for(r.status);
            }
            if (qcqp.has_quadratic_constraints() && engine_name == "global") {
                // Nonconvex QCQP: spatial branch-and-bound over a McCormick/
                // RLT envelope of EVERY quadratic row, objective and
                // constraints alike (search/global_qp.hpp's
                // solve_global_qcqp).  The point is re-checked against the
                // RAW QPLIB data here, sharing no code with the engine;
                // Optimal needs the tree closed within the gap, the same
                // rule the linearly-constrained --engine global uses.
                std::printf("model:             %s\n", q.name.c_str());
                std::printf("rows x cols:       %d x %d   (%c%c%c, %zu quadratic rows)  integer cols %zu\n",
                            q.m, q.n, q.classification[0], q.classification[1], q.classification[2],
                            qcqp.quad.size(), qcqp.qp.linear.n_integer());
                std::printf("engine:            global (spatial branch-and-bound, quadratic constraints%s)\n",
                            qcqp.qp.linear.n_integer() > 0 ? " + integer branching" : "");
                global_opts.time_limit_s = sx_opts.time_limit_s > 0.0 ? sx_opts.time_limit_s : 60.0;
                if (max_iter_given) global_opts.max_nodes = sx_opts.max_iterations;
                if (tol_given) global_opts.gap_tol = tol;
                global_opts.verbose = sx_opts.verbose;
                // Seed the tree with a point from the LOCAL engine before
                // branching.  The two engines fail in opposite directions on
                // refinery pooling models: the spatial B&B proves optimality
                // on the small ones but returns NoSolutionFound on the larger
                // ones (benchmarks/refinery REF_large, 167 vars: 2 nodes, no
                // incumbent), while the local multi-start finds a good point
                // at every size and proves nothing.  An incumbent also gives
                // the tree something to prune against from the first node, so
                // it costs nothing when the B&B would have found one anyway.
                // The point is re-scored and re-checked by solve_global_qcqp,
                // so a bad seed can only waste its slice of the budget -- it
                // cannot enter the claim.
                std::vector<sor::core::f64> warm_x;
                {
                    const double budget =
                        global_opts.time_limit_s > 0.0
                            ? std::min(0.25 * global_opts.time_limit_s, 15.0)
                            : 5.0;
                    sor::engines::QcqpLocalOptions wo;
                    wo.time_limit_s = budget;
                    wo.starts = 1000000;          // time is the real limiter
                    wo.mccormick_start = true;
                    if (tol_given) wo.feas_tol = tol;
                    // This point is re-checked by solve_global_qcqp's own
                    // offer() against global_opts.feas_tol (default 1e-7),
                    // which is TIGHTER than QcqpLocalOptions::feas_tol's own
                    // default (1e-6, sor/engines/qcqp.hpp) -- the local
                    // multi-start loop stops improving a candidate once ITS
                    // OWN (looser) bar is cleared, so it can hand back a
                    // point offer() then silently discards.  Measured on
                    // REF_MP_large (432 vars, 252 quadratic rows,
                    // the measurement notes): best start's violation
                    // sat at 8.838e-07 -- inside 1e-6, outside 1e-7 -- for
                    // 1303 nodes of tree search with NO other feasibility
                    // source (solve_global_qcqp has no in-tree local search
                    // or face-polish for quadratic constraint rows), so
                    // every run reported NoSolutionFound despite a perfectly
                    // usable near-feasible point already in hand.  Only ever
                    // tightens (never loosens) the bar this search targets.
                    wo.feas_tol = std::min(wo.feas_tol, global_opts.feas_tol);
                    sor::engines::QcqpLocalDiagnostics wd;
                    auto wr = sor::engines::solve_qcqp_local(qcqp, wo, wd);
                    if (wd.feasible && !wr.x.empty()) {
                        warm_x = wr.x;
                        std::printf("warm start:        local engine, %.3g s, objective %.10e\n",
                                    budget, (qcqp.objective_negated ? -1.0 : 1.0) * wd.objective);
                    } else {
                        std::printf("warm start:        none (local engine found no feasible point in %.3g s)\n",
                                    budget);
                    }
                    global_opts.time_limit_s =
                        std::max(1.0, global_opts.time_limit_s - wd.total_ms / 1000.0);
                }
                const auto g = sor::search::solve_global_qcqp(
                    qcqp, global_opts, warm_x.empty() ? nullptr : &warm_x);
                const double sgn = qcqp.objective_negated ? -1.0 : 1.0;

                // NOT qplib_point_check: that helper predates quadratic
                // CONSTRAINT support and only ever accumulates q.a_val (the
                // linear matrix) into row activities -- it silently omits
                // every q.hc_val (per-row quadratic Hessian) term, so on an
                // instance with quadratic constraint rows it reports a
                // bogus violation for a genuinely feasible point (measured:
                // QPLIB_3089, an internally-feasible incumbent read back as
                // viol 3.5e5 through the wrong helper). qplib_evaluate_point
                // is the QCQP-aware one (already used by the qcqp_ipm branch
                // above): it adds hc_val into row activity exactly once,
                // upper-triangle, same convention as the objective's H.
                // qplib_evaluate_point's max_violation() already folds
                // max_integrality_violation in (see QplibPointEval), so
                // `feasible` below is never true for a fractional column on
                // the RAW data -- chk_int is pulled out only so the mixed-
                // integer case prints as legibly as the miqcqp_bb branch
                // above does, not because feasibility needs it separately.
                double chk_obj = sor::core::kNaN, chk_viol = sor::core::kPosInf,
                       chk_int = sor::core::kPosInf;
                if (g.have_incumbent) {
                    const auto re = sor::io::qplib_evaluate_point(q, g.x);
                    chk_obj = re.objective;
                    chk_viol = re.max_violation();
                    chk_int = re.max_integrality_violation;
                }
                const double feas_tol = tol_given ? tol : 1e-6;
                const double gap_tol = global_opts.gap_tol;
                const bool feasible = g.have_incumbent && chk_viol <= feas_tol;
                // The engine's min-form value and the raw re-score must agree.
                const bool objective_agrees =
                    feasible && std::fabs(chk_obj - sgn * g.incumbent) <=
                                    1e-9 * (1.0 + std::fabs(chk_obj));
                const bool bound_valid = g.bound_valid;
                const double bound = sgn * g.bound;
                double gap_rel = sor::core::kPosInf;
                bool inconsistent = false;
                if (feasible && bound_valid) {
                    const double diff = q.maximize ? bound - chk_obj : chk_obj - bound;
                    gap_rel = std::max(0.0, diff) / std::max(1.0, std::fabs(chk_obj));
                    inconsistent = diff < -feas_tol * std::max(1.0, std::fabs(chk_obj));
                }
                sor::core::RawResult raw;
                raw.engine = "global";
                raw.backend = "cpu";
                raw.objective = feasible ? chk_obj : sor::core::kNaN;
                raw.dual_bound = bound_valid ? bound : sor::core::kNaN;
                raw.x = g.x;
                raw.iterations = g.nodes;
                sor::core::ProofEvidence ev;
                ev.checker_passed = objective_agrees && !inconsistent;
                ev.max_primal_violation = feasible ? chk_viol : sor::core::kPosInf;
                // No dual residual in this proof: the bound is a weak-duality
                // value re-derived from the McCormick LP's multipliers with
                // rounding charged, same as the linear --engine global.
                ev.max_dual_violation = 0.0;
                ev.primal_feas_tol = feas_tol;
                ev.gap_rel = gap_rel;
                ev.gap_tol = gap_tol;
                if (!g.supported) {
                    raw.proposed_status = sor::core::Status::Unsupported;
                    ev.claimed_level = sor::core::ProofLevel::None;
                    raw.termination_reason = g.reason;
                } else if (inconsistent) {
                    raw.proposed_status = sor::core::Status::NumericalFailure;
                    raw.termination_reason = "bound and incumbent contradict each other";
                } else if (feasible && bound_valid && objective_agrees && g.proved &&
                           gap_rel <= gap_tol) {
                    raw.proposed_status = sor::core::Status::Optimal;
                    ev.claimed_level = sor::core::ProofLevel::ProvedGlobalEpsilon;
                    raw.termination_reason = g.reason;
                } else if (feasible) {
                    raw.proposed_status = sor::core::Status::Feasible;
                    ev.claimed_level = bound_valid ? sor::core::ProofLevel::FeasibleWithGap
                                                   : sor::core::ProofLevel::FeasibleOnly;
                    raw.termination_reason = g.reason;
                } else {
                    raw.proposed_status = sor::core::Status::NoSolutionFound;
                    raw.termination_reason = g.reason;
                }
                const auto r = sor::certify::finalize_result(std::move(raw), ev);
                print_result(r);
                write_solution_out(solution_out, r);
                std::printf("relaxation used:   %s\n", g.relaxation_used.c_str());
                std::printf("gap (rel):         %.3e\n", gap_rel);
                std::printf("root bound (LP):   %.10e\n", g.root_bound_lp);
                if (g.psd_cuts > 0) std::printf("root bound (PSD):  %.10e\n", g.root_bound_psd);
                if (qcqp.qp.linear.n_integer() > 0)
                    std::printf("raw re-check:      obj %.10e  viol %.3e  int %.3e\n", chk_obj,
                                chk_viol, chk_int);
                else
                    std::printf("raw re-check:      obj %.10e  viol %.3e\n", chk_obj, chk_viol);
                std::printf("tree:              %llu nodes, %llu pruned, %llu infeasible, "
                            "%llu branched, %llu lp solves, %llu local solves, %llu psd cuts, "
                            "%llu fbbt tightened, %llu obbt tightened\n",
                            static_cast<unsigned long long>(g.nodes),
                            static_cast<unsigned long long>(g.pruned),
                            static_cast<unsigned long long>(g.infeasible),
                            static_cast<unsigned long long>(g.branched),
                            static_cast<unsigned long long>(g.lp_solves),
                            static_cast<unsigned long long>(g.local_solves),
                            static_cast<unsigned long long>(g.psd_cuts),
                            static_cast<unsigned long long>(g.fbbt_tightened),
                            static_cast<unsigned long long>(g.obbt_tightened));
                std::printf("termination:       %s\n", r.termination_reason.c_str());
                std::printf("\ntiming (ms)\n");
                std::printf("  total            %10.3f\n", g.total_ms);
                return exit_code_for(r.status);
            }
            if (qcqp.has_quadratic_constraints()) {
                std::printf("model:             %s\n", q.name.c_str());
                std::printf("rows x cols:       %d x %d   (%c%c%c, %zu quadratic rows)\n",
                            q.m, q.n, q.classification[0], q.classification[1],
                            q.classification[2], qcqp.quad.size());
                std::printf("engine:            %s\n", engine_name.c_str());
                const auto r = sor::certify::finalize_result(
                    sor::engines::qcqp_unsupported(engine_name, backend_name, qcqp),
                    sor::core::ProofEvidence{});
                print_result(r);
                std::printf("termination:       %s\n", r.termination_reason.c_str());
                return exit_code_for(r.status);
            }
        }
        // The LP/MILP engines read MPS; handing them a .qplib used to die in
        // the MPS parser with a misleading message.  Say what is true.
        if (path_is_qplib && engine_name != "binquad" && engine_name != "qp" &&
            engine_name != "qpipm" && engine_name != "qpauto" &&
            engine_name != "hprqp" && engine_name != "miqp" &&
            engine_name != "global" && engine_name != "qcqplocal") {
            sor::core::RawResult raw;
            raw.proposed_status = sor::core::Status::Unsupported;
            raw.engine = engine_name;
            raw.backend = backend_name;
            raw.termination_reason = "engine '" + engine_name +
                "' does not read QPLIB files (use qp, qpipm, qpauto, hprqp, binquad, miqp or global)";
            const auto r = sor::certify::finalize_result(std::move(raw),
                                                         sor::core::ProofEvidence{});
            print_result(r);
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            return exit_code_for(r.status);
        }

        if (engine_name == "binquad") {
            // Binary quadratic (QPLIB QBL/QBN/QBB): binquad supplies the
            // incumbent, the QCR relaxation on --backend supplies a certified
            // bound, and the claim is whatever the two together prove.
            sor::io::QplibReadReport qrep;
            const auto q = sor::io::read_qplib_file(path, qrep);
            std::printf("model:             %s\n", q.name.c_str());
            std::printf("rows x cols:       %d x %d   (%c%c%c)\n", q.m, q.n,
                        q.classification[0], q.classification[1], q.classification[2]);
            std::printf("engine:            binquad + qcr\n");

            const double budget = sx_opts.time_limit_s > 0.0 ? sx_opts.time_limit_s : 60.0;
            const double search_share = bq_search_only ? 1.0 : 0.5;
            const auto t_engine = std::chrono::steady_clock::now();
            sor::search::BinQuadDiagnostics bd;
            sor::search::BinQuadResult inc;
            std::string search_backend = "cpu (host tabu search)";
            if (backend_name == "cpu") {
                sor::search::BinQuadOptions bo;
                bo.time_limit_s = search_share * budget;
                // BinQuadOptions::max_iterations defaults to 200000 as a
                // safety net for callers with no time limit at all; here
                // bo.time_limit_s is always positive (budget defaults to
                // 60s above), so the in-loop wall-clock check is already
                // the real stop condition. Left at the 200000 default, the
                // iteration cap binds FIRST on small/medium QBL/QBN
                // instances (O(n) per iteration): measured on QPLIB_10045
                // (n=150), the search returned after 411 ms of its 30000 ms
                // share, then the rest of the run's budget went entirely
                // to the QCR branch-and-bound's DUAL bound with no primal
                // improvement (incumbent identical bit-for-bit after 5450
                // node batches). Same idle-budget shape as the qcqplocal
                // --starts fix; raise the cap so time, not an iteration
                // count, is the limiter whenever a real time limit is set.
                bo.max_iterations = std::numeric_limits<std::uint64_t>::max();
                bo.verbose = sx_opts.verbose;
                inc = sor::search::solve_binquad(q, bo, bd);
            } else {
                // P local searches on the device, global search on the host.
                auto bdev = sor::backend::make_binquad_device(backend_name);
                if (!bdev) {
                    std::fprintf(stderr, "error: backend '%s' has no binquad device on "
                                         "this machine\n", backend_name.c_str());
                    return 2;
                }
                sor::search::BinQuadParallelOptions po;
                po.searches = bq_searches;
                po.time_limit_s = search_share * budget;
                inc = sor::search::solve_binquad_parallel(q, po, *bdev, bd);
                search_backend = std::string(bdev->name()) + " (" +
                                 std::to_string(bq_searches) + " parallel searches)";
            }
            std::printf("search backend:    %s\n", search_backend.c_str());

            // Bound: branch-and-bound over QCR relaxations, K nodes per device
            // pass, warm-started from the incumbent above.
            std::unique_ptr<sor::backend::BatchedPdhcgDevice> dev;
            if (backend_name == "cpu") {
                std::vector<std::unique_ptr<sor::backend::PdhcgDevice>> lanes;
                for (std::uint32_t l = 0; l < bq_batch; ++l)
                    lanes.push_back(sor::backend::make_cpu_pdhcg_device());
                dev = sor::backend::make_lanes_device(std::move(lanes));
            } else if (backend_name == "vulkan") {
                dev = sor::backend::make_vulkan_batched_pdhcg_device();
            }
            if (!dev) {
                std::fprintf(stderr, "error: backend '%s' has no batched PDHCG device on "
                                     "this machine\n", backend_name.c_str());
                return 2;
            }
            std::printf("bound backend:     %s (accelerated=%s, %u nodes per pass)\n",
                        std::string(dev->name()).c_str(),
                        dev->is_accelerated() ? "yes" : "no", bq_batch);
            sor::search::BqpBabOptions bb;
            bb.batch = bq_batch;
            bb.sb_candidates = bq_strong;
            bb.verbose = sx_opts.verbose;
            if (!bq_node_arg.empty()) {
                using NS = sor::search::BqpNodeSolver;
                if      (bq_node_arg == "auto")  bb.node_solver = NS::Auto;
                else if (bq_node_arg == "ipm")   bb.node_solver = NS::Ipm;
                else if (bq_node_arg == "pdhcg") bb.node_solver = NS::Pdhcg;
                else {
                    std::fprintf(stderr, "error: --bq-node must be auto, ipm or pdhcg\n");
                    return 2;
                }
            }
            // The search may return long before its share is up (the host
            // tabu search has its own iteration cap: 0.3 s of a 30 s share on
            // QPLIB_0633).  Whatever it left is the tree's, so the run still
            // ends at the budget, not before it.
            {
                const double used = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_engine).count();
                bb.time_limit_s = std::max(0.25 * budget, budget - used);
            }
            bb.qcr.qp.max_iterations = 5000;
            // The root is solved once and every node inherits its quality;
            // the time limit, not this number, is what actually stops it.
            bb.root_iterations = 200000;
            bb.qcr.sdp.verbose = sx_opts.verbose;
            bb.qcr.sdp.time_limit_s = 0.15 * budget;
            if (tol_given) bb.gap_tol = tol;
            if (!qcr_shift_arg.empty()) {
                using S = sor::search::QcrShift;
                if      (qcr_shift_arg == "auto") bb.qcr.shift = S::Auto;
                else if (qcr_shift_arg == "best") bb.qcr.shift = S::Best;
                else if (qcr_shift_arg == "sdp")  bb.qcr.shift = S::Sdp;
                else if (qcr_shift_arg == "eig")  bb.qcr.shift = S::MinEigenvalue;
                else if (qcr_shift_arg == "dd")   bb.qcr.shift = S::DiagonalDominance;
                else {
                    std::fprintf(stderr, "error: --qcr-shift must be one of "
                                         "auto|best|sdp|eig|dd\n");
                    return 2;
                }
            }
            sor::search::BqpBabResult tree;
            if (bq_search_only) {
                tree.reason = "skipped (--bq-search-only)";
                tree.qcr.reason = tree.reason;
            }
            else tree = sor::search::solve_bqp_bab(q, &inc, bb, *dev);
            const auto& qd = tree.qcr;
            // The tree may improve on the search's point; take the better.
            if (tree.have_incumbent) {
                inc.x = tree.x;
                inc.objective = tree.incumbent;
                inc.feasible = true;
            }
            struct { bool valid; double bound; } qb{tree.bound_valid, tree.bound};

            double chk_obj = 0.0, chk_viol = 0.0;
            if (!inc.x.empty()) qplib_binary_check(q, inc.x, chk_obj, chk_viol);
            const double feas_tol = tol_given ? tol : 1e-6;
            const double gap_tol = tol_given ? tol : 1e-6;
            const bool feasible = inc.feasible && !inc.x.empty() && chk_viol <= feas_tol;
            const bool objective_agrees =
                feasible && std::fabs(chk_obj - inc.objective) <=
                                1e-9 * (1.0 + std::fabs(chk_obj));
            // Sign of the gap: a minimisation's bound sits below the incumbent,
            // a maximisation's above.  A bound on the wrong side by more than
            // tolerance means something upstream is wrong, not that we won.
            double gap_rel = sor::core::kPosInf;
            bool inconsistent = false;
            if (feasible && qb.valid) {
                const double diff = q.maximize ? qb.bound - chk_obj : chk_obj - qb.bound;
                gap_rel = std::max(0.0, diff) / std::max(1.0, std::fabs(chk_obj));
                inconsistent = diff < -feas_tol * std::max(1.0, std::fabs(chk_obj));
            }

            sor::core::RawResult raw;
            raw.engine = "binquad+qcr";
            raw.backend = std::string(dev->name());
            raw.objective = feasible ? chk_obj : sor::core::kNaN;
            raw.dual_bound = qb.valid ? qb.bound : sor::core::kNaN;
            for (auto v : inc.x) raw.x.push_back(static_cast<double>(v));
            sor::core::ProofEvidence ev;
            ev.checker_passed = objective_agrees && !inconsistent;
            ev.max_primal_violation = feasible ? chk_viol : sor::core::kPosInf;
            // No dual residual in this proof: the bound is a weak-duality value
            // re-derived by wolfe_bound() with its error charges applied.
            ev.max_dual_violation = 0.0;
            ev.primal_feas_tol = feas_tol;
            ev.gap_rel = gap_rel;
            ev.gap_tol = gap_tol;
            if (inconsistent) {
                raw.proposed_status = sor::core::Status::NumericalFailure;
                raw.termination_reason = "bound and incumbent contradict each other";
            } else if (feasible && qb.valid && (tree.proved || gap_rel <= gap_tol)) {
                raw.proposed_status = sor::core::Status::Optimal;
                ev.claimed_level = sor::core::ProofLevel::ProvedGlobalEpsilon;
                raw.termination_reason = tree.proved
                    ? "branch-and-bound closed: every node pruned by a certified QCR bound"
                    : "incumbent meets the certified QCR bound";
            } else if (feasible) {
                raw.proposed_status = sor::core::Status::Feasible;
                ev.claimed_level = qb.valid ? sor::core::ProofLevel::FeasibleWithGap
                                            : sor::core::ProofLevel::FeasibleOnly;
                raw.termination_reason = qb.valid ? "gap open: " + tree.reason
                                                  : "no bound: " + qd.reason;
            } else {
                raw.proposed_status = sor::core::Status::NoSolutionFound;
                ev.claimed_level = qb.valid ? sor::core::ProofLevel::BoundOnly
                                            : sor::core::ProofLevel::None;
                raw.termination_reason = "no feasible binary point found";
            }
            raw.proposed_level = ev.claimed_level;
            const auto r = sor::certify::finalize_result(std::move(raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            std::printf("incumbent:         %.10e  (recomputed from raw QPLIB data)\n",
                        chk_obj);
            if (qb.valid)
                std::printf("qcr bound:         %.10e  (%s bound, certified)\n", qb.bound,
                            q.maximize ? "upper" : "lower");
            else
                std::printf("qcr bound:         none (%s)\n", qd.reason.c_str());
            std::printf("gap (rel):         %.3e\n", gap_rel);
            std::printf("qcr shift:         %s", qd.shift_used.c_str());
            if (qd.shift_used == "min-eigenvalue")
                std::printf("  lambda_min~%.6e  s=%.6e", qd.lambda_min_estimate,
                            qd.uniform_shift);
            std::printf("\nqcr charges:       psd %.3e  fp %.3e  form %.3e  (raw %.10e)\n",
                        qd.charge_psd, qd.charge_fp, qd.charge_form, qd.bound_raw);
            double root_gap_rel = sor::core::kPosInf;
            if (feasible && tree.root_bound_valid) {
                const double diff = q.maximize ? tree.root_bound - chk_obj
                                               : chk_obj - tree.root_bound;
                root_gap_rel = std::max(0.0, diff) / std::max(1.0, std::fabs(chk_obj));
            }
            if (tree.root_bound_valid)
                std::printf("root bound:        %.10e  (certified, root node only)\n",
                            tree.root_bound);
            std::printf("root gap (rel):    %.3e\n", root_gap_rel);
            if (!tree.shift_alt.empty())
                std::printf("shift runner-up:   %s root bound %.10e%s\n",
                            tree.shift_alt.c_str(), tree.root_bound_alt,
                            tree.root_bound_alt_valid ? "" : " (none)");
            if (qd.shift_used == "sdp")
                std::printf("sdp:               dual~%.10e primal~%.10e (min form, informative) "
                            "rank %d sweeps %lld rho %.3e products %zu  %.1f ms\n",
                            qd.sdp_dual, qd.sdp_primal, qd.sdp_rank, qd.sdp_sweeps,
                            qd.sdp_rho, qd.sdp_products, qd.sdp_ms);
            else if (!qd.sdp_reason.empty())
                std::printf("sdp:               not used (%s)\n", qd.sdp_reason.c_str());
            std::printf("node solver:       %s\n", tree.node_solver.c_str());
            std::printf("tree:              %llu nodes, %llu batches, %llu pruned, %llu leaves"
                        ", %llu probes (%s)\n",
                        static_cast<unsigned long long>(tree.nodes),
                        static_cast<unsigned long long>(tree.batches),
                        static_cast<unsigned long long>(tree.pruned),
                        static_cast<unsigned long long>(tree.leaves),
                        static_cast<unsigned long long>(tree.probes), tree.reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  binquad          %10.3f\n", bd.total_ms);
            std::printf("  branch-and-bound %10.3f\n", tree.total_ms);
            print_transfer(dev->transfer_stats());
            return exit_code_for(r.status);
        }

        if (engine_name == "global") {
            // Nonconvex continuous QP with linear constraints (QPLIB class 5,
            // Q C {N,B,L}): spatial branch-and-bound over McCormick/RLT LP and
            // alphaBB QP relaxations (search/global_qp.hpp).  The point is
            // re-checked against the RAW QPLIB data here, sharing no code with
            // the engine; Optimal needs the tree closed within the gap.
            sor::io::QplibReadReport qrep;
            const auto q = sor::io::read_qplib_file(path, qrep);
            std::printf("model:             %s\n", q.name.c_str());
            std::printf("rows x cols:       %d x %d   (%c%c%c)\n", q.m, q.n,
                        q.classification[0], q.classification[1], q.classification[2]);
            std::printf("engine:            global (spatial branch-and-bound)\n");
            sor::engines::QpProblem qp;
            bool negated = false;
            std::string why;
            if (!sor::search::qplib_to_qp(q, {}, qp, negated, why)) {
                sor::core::RawResult raw;
                raw.engine = "global";
                raw.proposed_status = sor::core::Status::Unsupported;
                raw.termination_reason = why;
                const auto r = sor::certify::finalize_result(std::move(raw), sor::core::ProofEvidence{});
                print_result(r);
                std::printf("reason:            %s\n", why.c_str());
                return exit_code_for(r.status);
            }
            global_opts.time_limit_s = sx_opts.time_limit_s > 0.0 ? sx_opts.time_limit_s : 60.0;
            if (max_iter_given) global_opts.max_nodes = sx_opts.max_iterations;
            if (tol_given) global_opts.gap_tol = tol;
            global_opts.verbose = sx_opts.verbose;
            const auto g = sor::search::solve_global_qp(qp, global_opts);
            const double sgn = negated ? -1.0 : 1.0;

            double chk_obj = sor::core::kNaN, chk_viol = sor::core::kPosInf;
            if (g.have_incumbent) qplib_point_check(q, g.x, chk_obj, chk_viol);
            const double feas_tol = tol_given ? tol : 1e-6;
            const double gap_tol = global_opts.gap_tol;
            const bool feasible = g.have_incumbent && chk_viol <= feas_tol;
            // The engine's min-form value and the raw re-score must agree.
            const bool objective_agrees =
                feasible && std::fabs(chk_obj - sgn * g.incumbent) <=
                                1e-9 * (1.0 + std::fabs(chk_obj));
            const bool bound_valid = g.bound_valid;
            const double bound = sgn * g.bound;
            double gap_rel = sor::core::kPosInf;
            bool inconsistent = false;
            if (feasible && bound_valid) {
                const double diff = q.maximize ? bound - chk_obj : chk_obj - bound;
                gap_rel = std::max(0.0, diff) / std::max(1.0, std::fabs(chk_obj));
                inconsistent = diff < -feas_tol * std::max(1.0, std::fabs(chk_obj));
            }
            sor::core::RawResult raw;
            raw.engine = "global";
            raw.backend = "cpu";
            raw.objective = feasible ? chk_obj : sor::core::kNaN;
            raw.dual_bound = bound_valid ? bound : sor::core::kNaN;
            raw.x = g.x;
            raw.iterations = g.nodes;
            sor::core::ProofEvidence ev;
            ev.checker_passed = objective_agrees && !inconsistent;
            ev.max_primal_violation = feasible ? chk_viol : sor::core::kPosInf;
            // No dual residual in this proof: the bound is a weak-duality value
            // re-derived from the relaxations' multipliers with rounding charged.
            ev.max_dual_violation = 0.0;
            ev.primal_feas_tol = feas_tol;
            ev.gap_rel = gap_rel;
            ev.gap_tol = gap_tol;
            if (!g.supported) {
                raw.proposed_status = sor::core::Status::Unsupported;
                ev.claimed_level = sor::core::ProofLevel::None;
                raw.termination_reason = g.reason;
            } else if (inconsistent) {
                raw.proposed_status = sor::core::Status::NumericalFailure;
                raw.termination_reason = "bound and incumbent contradict each other";
            } else if (feasible && bound_valid && objective_agrees && g.proved &&
                       gap_rel <= gap_tol) {
                raw.proposed_status = sor::core::Status::Optimal;
                ev.claimed_level = sor::core::ProofLevel::ProvedGlobalEpsilon;
                raw.termination_reason = g.reason;
            } else if (feasible) {
                raw.proposed_status = sor::core::Status::Feasible;
                ev.claimed_level = bound_valid ? sor::core::ProofLevel::FeasibleWithGap
                                               : sor::core::ProofLevel::FeasibleOnly;
                raw.termination_reason = "gap open: " + g.reason;
            } else {
                raw.proposed_status = sor::core::Status::NoSolutionFound;
                ev.claimed_level = bound_valid ? sor::core::ProofLevel::BoundOnly
                                               : sor::core::ProofLevel::None;
                raw.termination_reason = g.reason;
            }
            raw.proposed_level = ev.claimed_level;
            const auto r = sor::certify::finalize_result(std::move(raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            std::printf("reason:            %s\n", r.termination_reason.c_str());
            std::printf("incumbent:         %.10e  (recomputed from raw QPLIB data, viol %.2e)\n",
                        chk_obj, chk_viol);
            if (bound_valid)
                std::printf("bound:             %.10e  (%s bound, certified)\n", bound,
                            q.maximize ? "upper" : "lower");
            else
                std::printf("bound:             none\n");
            std::printf("gap (rel):         %.3e\n", gap_rel);
            std::printf("relaxation:        %s\n", g.relaxation_used.c_str());
            std::printf("root bounds:       lp %.10e  +psd %.10e  shift %.10e\n",
                        sgn * g.root_bound_lp, sgn * g.root_bound_psd,
                        sgn * g.root_bound_shift);
            std::printf("alphaBB shift:     d %.6e  psd slack %.3e  eq-penalty rho %.3e\n",
                        g.shift, g.shift_psd_slack, g.equality_rho);
            std::printf("tree:              %llu nodes, %llu branched, %llu pruned, %llu infeasible\n",
                        static_cast<unsigned long long>(g.nodes),
                        static_cast<unsigned long long>(g.branched),
                        static_cast<unsigned long long>(g.pruned),
                        static_cast<unsigned long long>(g.infeasible));
            std::printf("solves:            %llu LP, %llu QP, %llu local; fbbt %llu, obbt %llu tightenings\n",
                        static_cast<unsigned long long>(g.lp_solves),
                        static_cast<unsigned long long>(g.qp_solves),
                        static_cast<unsigned long long>(g.local_solves),
                        static_cast<unsigned long long>(g.fbbt_tightened),
                        static_cast<unsigned long long>(g.obbt_tightened));
            std::printf("psd cuts:          %llu\n",
                        static_cast<unsigned long long>(g.psd_cuts));
            std::printf("time (ms):         %.3f\n", g.total_ms);
            return exit_code_for(r.status);
        }

        if (engine_name == "hprqp") {
            // HPR-QP shares the .qps / --q-diag loading of --engine qp but
            // runs the dual Halpern Peaceman-Rachford method on a QpDevice,
            // so --backend selects where the iteration actually runs.
            sor::engines::QpProblem qp;
            if (!load_qp_problem(qp, path, q_diag_arg, path_is_qps,
                                 mps_format_forced, mps_opts))
                return 2;
            if (const int rc = report_empty_domain(qp.linear, solution_out); rc >= 0)
                return rc;

            std::printf("model:             %s\n",
                        qp.linear.name.empty() ? path.c_str()
                                               : qp.linear.name.c_str());
            std::printf("rows x cols:       %d x %d   nnz %lld\n",
                        qp.linear.n_rows(), qp.linear.n_cols(),
                        static_cast<long long>(qp.linear.nnz()));
            std::printf("engine:            hpr_qp\n");

            auto dev = sor::backend::make_qp_device(backend_name);
            if (!dev) {
                sor::core::RawResult unavailable;
                unavailable.proposed_status = sor::core::Status::Unsupported;
                unavailable.engine = "hpr_qp";
                unavailable.backend = backend_name;
                unavailable.termination_reason =
                    "requested QP device '" + backend_name + "' is unavailable";
                const auto r = sor::certify::finalize_result(
                    std::move(unavailable), sor::core::ProofEvidence{});
                print_result(r);
                std::printf("termination:       %s\n", r.termination_reason.c_str());
                return exit_code_for(r.status);
            }
            std::printf("backend:           %s (accelerated=%s)\n",
                        std::string(dev->name()).c_str(),
                        dev->is_accelerated() ? "yes" : "no");

            sor::engines::HprQpOptions hqopts;
            hqopts.max_iterations = pdhg_opts.max_iterations;
            hqopts.time_limit_s = sx_opts.time_limit_s;
            hqopts.verbose = sx_opts.verbose;
            if (tol_given) {
                hqopts.feas_tol = tol;
                hqopts.stationarity_tol = tol;
                hqopts.gap_tol = tol;
            }
            sor::engines::HprQpDiagnostics hqdiag;
            auto raw = sor::engines::solve_hpr_qp(qp, hqopts, *dev, hqdiag);
            const auto ev = sor::engines::hpr_qp_evidence(hqdiag, hqopts);
            const auto r = sor::certify::finalize_result(std::move(raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            std::printf("stationarity:      %.3e  (rel %.3e)\n",
                        hqdiag.stationarity, hqdiag.stationarity_rel);
            std::printf("max primal viol:   %.3e  (rel %.3e)\n",
                        hqdiag.primal_residual, hqdiag.primal_residual_rel);
            std::printf("relative gap:      %.3e\n", hqdiag.gap_rel);
            std::printf("lambda_Q/lambda_A: %.6e / %.6e\n",
                        hqdiag.lambda_q, hqdiag.lambda_a);
            std::printf("sigma (final):     %.6e\n", hqdiag.sigma_final);
            std::printf("restarts:          %llu (suff %llu / nec %llu / long %llu)\n",
                        static_cast<unsigned long long>(hqdiag.restarts),
                        static_cast<unsigned long long>(hqdiag.sufficient_restarts),
                        static_cast<unsigned long long>(hqdiag.necessary_restarts),
                        static_cast<unsigned long long>(hqdiag.long_loop_restarts));
            std::printf("iterations:        %llu\n",
                        static_cast<unsigned long long>(hqdiag.iterations));
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", hqdiag.total_ms);
            std::printf("  setup            %10.3f\n", hqdiag.setup_ms);
            std::printf("  loop             %10.3f\n", hqdiag.loop_ms);
            print_transfer(hqdiag.device_stats);
            return exit_code_for(r.status);
        }

        if (engine_name == "miqp") {
            // Mixed-integer QP from QPLIB with integer typing kept.  The claim
            // is re-checked on the RAW file data (qplib_point_check) before
            // finalize_result sees it, so a convention slip in qplib_to_qp
            // cannot turn into a wrong "optimal".
            sor::io::QplibReadReport qrep;
            const auto q = sor::io::read_qplib_file(path, qrep);
            sor::engines::QpProblem qp;
            bool negated = false;
            std::string why;
            sor::search::QplibToQpOptions qo;
            qo.keep_integer = true;
            if (!sor::search::qplib_to_qp(q, qo, qp, negated, why)) {
                std::fprintf(stderr, "error: %s\n", why.c_str());
                return 2;
            }
            std::printf("model:             %s\n", q.name.c_str());
            std::printf("rows x cols:       %d x %d   (%c%c%c)  integer cols %zu\n", q.m, q.n,
                        q.classification[0], q.classification[1], q.classification[2],
                        qp.linear.n_integer());
            std::printf("engine:            miqp_bb (IPM node relaxations)\n");
            sor::search::MiqpBbOptions mo;
            mo.time_limit_s = sx_opts.time_limit_s;
            mo.verbose = sx_opts.verbose;
            if (max_iter_given) mo.max_nodes = sx_opts.max_iterations;
            if (tol_given) mo.gap_rel = tol;
            if (!apply_pending_options(sor::search::miqp_bb_option_bindings(mo), miqp_opt_args, "--miqp-opt")) return 2;
            sor::search::MiqpBbDiagnostics md;
            auto raw = sor::search::solve_miqp_bb(qp, mo, md);
            auto ev = sor::search::miqp_bb_evidence(qp, mo, md, raw);
            double chk_obj = sor::core::kNaN, chk_viol = sor::core::kPosInf, chk_int = sor::core::kPosInf;
            if (!raw.x.empty()) {
                qplib_point_check(q, raw.x, chk_obj, chk_viol, chk_int);
                const double mine = negated ? -raw.objective : raw.objective;
                const bool agrees = chk_int == 0.0 && chk_viol <= mo.feas_tol &&
                                    std::fabs(chk_obj - mine) <= 1e-9 * (1.0 + std::fabs(chk_obj));
                ev.checker_passed = ev.checker_passed && agrees;
                ev.max_primal_violation = std::max(ev.max_primal_violation, std::max(chk_viol, chk_int));
            }
            // Report in the instance's own sense.
            if (negated) {
                raw.objective = -raw.objective;
                raw.dual_bound = -raw.dual_bound;
            }
            const auto r = sor::certify::finalize_result(std::move(raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            std::printf("%s bound:       %.10e\n", negated ? "upper" : "lower",
                        negated ? -md.global_bound : md.global_bound);
            std::printf("root bound:        %.10e\n", negated ? -md.root_bound : md.root_bound);
            std::printf("gap (rel):         %.3e\n", md.gap_rel);
            std::printf("raw re-check:      obj %.10e  viol %.3e  int %.3e\n", chk_obj, chk_viol,
                        chk_int);
            std::printf("sigma (shift):     %.6e\n", md.sigma);
            std::printf("tree:              %llu nodes, %llu pruned by bound, %llu infeasible, "
                        "%llu integral, %llu unproved relaxations, %llu unresolved leaves, "
                        "%llu open, depth %d\n",
                        static_cast<unsigned long long>(md.nodes),
                        static_cast<unsigned long long>(md.pruned_bound),
                        static_cast<unsigned long long>(md.pruned_infeasible),
                        static_cast<unsigned long long>(md.integral_closed),
                        static_cast<unsigned long long>(md.unproved_nodes),
                        static_cast<unsigned long long>(md.unresolved_leaves),
                        static_cast<unsigned long long>(md.open_at_end), md.max_depth);
            std::printf("incumbents:        %llu (heuristic calls %llu)\n",
                        static_cast<unsigned long long>(md.incumbents),
                        static_cast<unsigned long long>(md.heuristic_calls));
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", md.total_ms);
            std::printf("  relaxations      %10.3f  (%llu IPM iterations)\n", md.relax_ms,
                        static_cast<unsigned long long>(md.ipm_iterations));
            std::printf("  heuristic        %10.3f\n", md.heuristic_ms);
            return exit_code_for(r.status);
        }

        if (engine_name == "hprqp") {
            // HPR-QP shares the .qps / --q-diag loading of --engine qp but
            // runs the dual Halpern Peaceman-Rachford method on a QpDevice,
            // so --backend selects where the iteration actually runs.
            sor::engines::QpProblem qp;
            if (!load_qp_problem(qp, path, q_diag_arg, path_is_qps,
                                 mps_format_forced, mps_opts))
                return 2;
            if (const int rc = report_empty_domain(qp.linear, solution_out); rc >= 0)
                return rc;

            std::printf("model:             %s\n",
                        qp.linear.name.empty() ? path.c_str()
                                               : qp.linear.name.c_str());
            std::printf("rows x cols:       %d x %d   nnz %lld\n",
                        qp.linear.n_rows(), qp.linear.n_cols(),
                        static_cast<long long>(qp.linear.nnz()));
            std::printf("engine:            hpr_qp\n");

            auto dev = sor::backend::make_qp_device(backend_name);
            if (!dev) {
                sor::core::RawResult unavailable;
                unavailable.proposed_status = sor::core::Status::Unsupported;
                unavailable.engine = "hpr_qp";
                unavailable.backend = backend_name;
                unavailable.termination_reason =
                    "requested QP device '" + backend_name + "' is unavailable";
                const auto r = sor::certify::finalize_result(
                    std::move(unavailable), sor::core::ProofEvidence{});
                print_result(r);
                std::printf("termination:       %s\n", r.termination_reason.c_str());
                return exit_code_for(r.status);
            }
            std::printf("backend:           %s (accelerated=%s)\n",
                        std::string(dev->name()).c_str(),
                        dev->is_accelerated() ? "yes" : "no");

            sor::engines::HprQpOptions hqopts;
            hqopts.max_iterations = pdhg_opts.max_iterations;
            hqopts.time_limit_s = sx_opts.time_limit_s;
            hqopts.verbose = sx_opts.verbose;
            if (tol_given) {
                hqopts.feas_tol = tol;
                hqopts.stationarity_tol = tol;
                hqopts.gap_tol = tol;
            }
            sor::engines::HprQpDiagnostics hqdiag;
            auto raw = sor::engines::solve_hpr_qp(qp, hqopts, *dev, hqdiag);
            const auto ev = sor::engines::hpr_qp_evidence(hqdiag, hqopts);
            const auto r = sor::certify::finalize_result(std::move(raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            std::printf("stationarity:      %.3e  (rel %.3e)\n",
                        hqdiag.stationarity, hqdiag.stationarity_rel);
            std::printf("max primal viol:   %.3e  (rel %.3e)\n",
                        hqdiag.primal_residual, hqdiag.primal_residual_rel);
            std::printf("relative gap:      %.3e\n", hqdiag.gap_rel);
            std::printf("lambda_Q/lambda_A: %.6e / %.6e\n",
                        hqdiag.lambda_q, hqdiag.lambda_a);
            std::printf("sigma (final):     %.6e\n", hqdiag.sigma_final);
            std::printf("restarts:          %llu (suff %llu / nec %llu / long %llu)\n",
                        static_cast<unsigned long long>(hqdiag.restarts),
                        static_cast<unsigned long long>(hqdiag.sufficient_restarts),
                        static_cast<unsigned long long>(hqdiag.necessary_restarts),
                        static_cast<unsigned long long>(hqdiag.long_loop_restarts));
            std::printf("iterations:        %llu\n",
                        static_cast<unsigned long long>(hqdiag.iterations));
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", hqdiag.total_ms);
            std::printf("  setup            %10.3f\n", hqdiag.setup_ms);
            std::printf("  loop             %10.3f\n", hqdiag.loop_ms);
            print_transfer(hqdiag.device_stats);
            return exit_code_for(r.status);
        }

        if (engine_name == "qp" || engine_name == "qpipm" || engine_name == "qpauto") {
            sor::engines::QpProblem qp;
            if (!load_qp_problem(qp, path, q_diag_arg, path_is_qps,
                                 mps_format_forced, mps_opts))
                return 2;
            if (const int rc = report_empty_domain(qp.linear, solution_out); rc >= 0)
                return rc;

            std::printf("model:             %s\n",
                        qp.linear.name.empty() ? path.c_str()
                                               : qp.linear.name.c_str());
            std::printf("rows x cols:       %d x %d   nnz %lld\n",
                        qp.linear.n_rows(), qp.linear.n_cols(),
                        static_cast<long long>(qp.linear.nnz()));
            std::printf("engine:            qp\n");

            sor::engines::QpOptions qopts;
            qopts.max_iterations = pdhg_opts.max_iterations;
            qopts.time_limit_s = sx_opts.time_limit_s;
            qopts.verbose = sx_opts.verbose;
            qopts.inner_epoch = qp_inner_epoch;
            if (tol_given) {
                qopts.feas_tol = tol;
                qopts.stationarity_tol = tol;
                qopts.gap_tol = tol;
            }
            if (!apply_pending_options(sor::engines::qp_option_bindings(qopts), qp_opt_args, "--qp-opt")) return 2;
            sor::engines::QpDiagnostics diag;
            sor::core::RawResult raw;
            if (engine_name == "qpipm") {
                raw = sor::engines::solve_qp_ipm(qp, qopts, diag);
                std::printf("backend:           cpu (interior point)\n");
            } else if (engine_name == "qpauto") {
                raw = sor::engines::solve_qp_auto(qp, qopts, diag);
                std::printf("backend:           cpu (routed)\n");
            } else if (backend_name == "cpu") {
                // Unchanged default: the exact diagonal active-set path when
                // the model allows it, PDHCG-II on the CPU device otherwise.
                raw = sor::engines::solve_qp(qp, qopts, diag);
                std::printf("backend:           cpu\n");
            } else {
                // A device was asked for, so PDHCG-II runs on it -- the
                // active-set fast path is combinatorial and stays CPU-only.
                auto dev = sor::backend::make_pdhcg_device(backend_name);
                if (!dev) {
                    std::fprintf(stderr, "error: backend '%s' has no PDHCG device "
                                         "on this machine\n", backend_name.c_str());
                    return 2;
                }
                std::printf("backend:           %s (accelerated=%s)\n",
                            std::string(dev->name()).c_str(),
                            dev->is_accelerated() ? "yes" : "no");
                raw = sor::engines::solve_qp_pdhcg(qp, qopts, *dev, diag);
            }
            const auto ev = sor::engines::qp_evidence(diag, qopts);
            const auto r = sor::certify::finalize_result(std::move(raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            std::printf("stationarity:      %.3e  (rel %.3e)\n", diag.stationarity,
                        diag.stationarity_rel);
            std::printf("max primal viol:   %.3e  (rel %.3e)\n", diag.primal_residual,
                        diag.primal_residual_rel);
            if (diag.worst_solve_residual > 0.0)
                std::printf("  %-18s %.3e\n", "solve residual:", diag.worst_solve_residual);
            std::printf("relative gap:      %.3e\n", diag.gap_rel);
            std::printf("iterations:        %llu\n",
                        static_cast<unsigned long long>(diag.iterations));
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", diag.total_ms);
            if (diag.used_general_path) print_transfer(diag.device_stats);
            return exit_code_for(r.status);
        }

        sor::io::MpsReadReport rep;
        // An LP engine answers the LP relaxation of a model with integer
        // columns. Read it relaxed so presolve sees the model it solves
        // (integer flags block aggregation and other LP reductions), and mark
        // the claim so sor_check checks that model instead of rejecting the
        // point as a fractional incumbent.
        const bool lp_engine =
            engine_name == "auto" || engine_name == "simplex" ||
            engine_name == "primal" || engine_name == "dual" ||
            engine_name == "hpr" || engine_name == "pdhg" || engine_name == "barrier";
        if (lp_engine) mps_opts.relax_integrality = true;
        const auto problem = mps_format_forced
            ? sor::io::read_mps_file(path, rep, mps_opts)
            : sor::io::read_mps_file_auto(path, rep, mps_opts);
        answering_lp_relaxation = rep.relaxed_integrality;
        for (const auto& w : rep.warnings)
            std::fprintf(stderr, "warning: %s\n", w.c_str());
        if (refuse_if_no_columns(path, problem.n_cols())) return 2;

        std::printf("model:             %s\n",
                    problem.name.empty() ? path.c_str() : problem.name.c_str());
        std::printf("rows x cols:       %d x %d   nnz %lld\n",
                    problem.n_rows(), problem.n_cols(),
                    static_cast<long long>(problem.nnz()));
        // A model with no columns is trivially optimal at 0, and that is exactly
        // the problem: it is indistinguishable from a file that failed to parse.
        // Reporting "proved optimal" for input that carried no model is the
        // worst failure available to this program, so refuse instead.
        if (problem.n_cols() == 0) {
            std::fprintf(stderr,
                         "error: %s contains no variables -- nothing to solve. "
                         "A model with no columns is almost always a file that "
                         "did not parse as the format it was read as.\n",
                         path.c_str());
            return 2;
        }
        if (rep.n_integer > 0)
            std::printf("integer columns:   %zu\n", rep.n_integer);
        std::printf("engine:            %s\n", engine_name.c_str());

        if (engine_name == "milp") {
            if (rep.n_integer == 0) {
                std::fprintf(stderr,
                             "warning: no integer columns; milp reduces to LP\n");
            }
            sor::search::BabOptions bab;
            bab.lp = sx_opts;
            bab.gap_tol = mip_gap_tol;
            std::printf("mip gap tolerance: %.9g\n", bab.gap_tol);
            // Node LPs keep their own default representation unless the user
            // asked for one explicitly: the standalone-LP default is
            // Forrest-Tomlin, which is a large regression on short
            // warm-started node LPs. See search::default_node_lp_options().
            if (!basis_update_explicit)
                bab.lp.update_method =
                    sor::search::default_node_lp_options().update_method;
            bab.verbose = sx_opts.verbose;
            if (!sor::search::parse_milp_policy(milp_policy, bab.policy)) {
                std::fprintf(stderr,
                             "error: unknown --milp-policy '%s' "
                             "(want latest|classical)\n",
                             milp_policy.c_str());
                return 2;
            }
            if (!sor::search::parse_branch_strategy(branch_strategy,
                                                    bab.branch_strategy)) {
                std::fprintf(stderr,
                             "error: unknown --branch-strategy '%s' "
                             "(want auto|reliability|sparse-sb|sc-milp|lifted|planbb)\n",
                             branch_strategy.c_str());
                return 2;
            }
            bab.sparse_sb.model_path = sparse_sb_model;
            bab.sparse_sb.collect_labels = sparse_sb_collect;
            bab.sc_milp.model_path = sc_milp_model;
            bab.sc_milp.collect_labels = sc_milp_collect;
            bab.lifted.expert_path = lifted_expert;
            bab.planbb.policy_path = planbb_policy;
            bab.planbb.model_path = planbb_model;
            bab.planbb.paper_mode = planbb_paper;
            bab.planbb.collect_labels = planbb_collect;
            bab.planbb.use_mcts = planbb_mcts;
            if (planbb_mcts_sims > 0) bab.planbb.mcts_sims = planbb_mcts_sims;
            if (planbb_mcts_depth > 0) bab.planbb.mcts_depth = planbb_mcts_depth;
            std::printf("milp policy:       %s\n",
                        sor::search::milp_policy_name(bab.policy));
            std::printf("branch strategy:   %s\n",
                        sor::search::branch_strategy_name(bab.branch_strategy));
            bab.probing = probing;
            bab.mip_presolve = mip_presolve;
            // Held by pointer for the whole solve: every separator checks its
            // cut against this point and aborts on a violation.
            std::vector<sor::core::f64> cut_ref;
            if (!verify_cuts_path.empty()) {
                std::ifstream rf(verify_cuts_path);
                if (rf) {
                    const auto rs = sor::io::read_solution(rf);
                    cut_ref = rs.x;
                    if (static_cast<sor::core::Index>(cut_ref.size()) ==
                        problem.n_cols()) {
                        bab.cut_reference_point = &cut_ref;
                        bab.cut_reference_debug =
                            std::getenv("SOR_CUT_REF_DEBUG") != nullptr;
                        std::printf("cut verify:        reference point loaded "
                                    "(%zu cols)\n", cut_ref.size());
                    } else {
                        std::fprintf(stderr, "warning: --verify-cuts point has "
                                     "%zu values, model has %d cols; ignored\n",
                                     cut_ref.size(), problem.n_cols());
                    }
                } else {
                    std::fprintf(stderr, "warning: cannot open %s\n",
                                 verify_cuts_path.c_str());
                }
            }
            bab.mip_pre.dual_fix_in_probing = dual_fix_probe;
            bab.mip_pre.clique_probing = clique_probe;
            bab.mip_pre.gf2 = gf2;
            bab.mip_pre.components = components;
            bab.mip_pre.implied_integers = implied_int;
            bab.mip_pre.obbt_lite = obbt;
            bab.mip_pre.batch_lp_obbt = batch_lp_obbt && obbt;
            bab.batch_lp_strong_branch = batch_lp_sb;
            bab.gpu_binary_heuristic = gpu_binary_heuristic;
            bab.para_bab.threads = bab_threads;
            if (mip_restarts >= 0) bab.mip_pre.max_restarts = mip_restarts;
            bab.symmetry = symmetry;
            bab.sym.reflection = reflection;
            bab.sym.folding = folding;
            bab.feasibility_jump = feasibility_jump;
            bab.sub_mip_lns = sub_mip_lns;
            bab.balans.enabled = balans;
            bab.kernel_pump.enabled = kernel_pump;
            bab.mrens.enabled = mrens;
            bab.btbs.enabled = btbs;
            bab.cl_tlns.enabled = cl_tlns;
            bab.balans.include_btbs = btbs;
            bab.balans.include_cl_tlns = cl_tlns;
            if (kernel_pump_time >= 0.0)
                bab.kernel_pump.time_limit_s = kernel_pump_time;
            if (mrens_time >= 0.0) bab.mrens.time_limit_s = mrens_time;
            if (btbs_time >= 0.0) bab.btbs.time_limit_s = btbs_time;
            if (cl_tlns_time >= 0.0) bab.cl_tlns.time_limit_s = cl_tlns_time;
            if (heuristic_budget >= 0.0) {
                bab.heuristic_budget_frac = heuristic_budget;
                bab.heuristic_budget_frac_no_incumbent =
                    std::max(heuristic_budget, 0.80);
            }
            bab.clique_cuts = clique_cuts;
            bab.implied_bound_cuts = implied_bound_cuts;
            bab.lifted_cover_cuts = lifted_cover_cuts;
            bab.mir_cuts = mir_cuts;
            bab.auto_cuts = auto_cuts;
            bab.cut.gmi_cmir_recovery = gmi_cmir_recovery;
            bab.cut.relax_small_terms = relax_small_terms;
            bab.cut.tableau_cmir = tableau_cmir;
            bab.tree_cut.resolve_with_local = node_cut_resolve;
            bab.tree_cut.max_local_rows = local_cut_rows;
            bab.conflict_cut.conflict_store = conflict_store;
            bab.conflict_cut.store_max_len = conflict_store_max_len;
            bab.spp_repair = spp_repair_on;
            bab.trace_cut_batches = trace_cuts;
            bab.sub_mip_context = sub_mip_context;
            bab.root_primal_early = root_primal_early;
            bab.root_primal_final = root_primal_final;
            bab.carry_probing = carry_probing;
            bab.cut_transaction = cut_transaction;
            bab.farkas_conflicts = farkas_conflicts;
            bab.trace_branching = trace_branching;
            if (rb_threshold >= 0) bab.reliability_threshold = rb_threshold;
            if (rb_max_probed >= 0) bab.rb_max_probed = rb_max_probed;
            if (rb_lookahead >= 0) bab.rb_lookahead_candidates = rb_lookahead;
            if (rb_time_share >= 0.0) bab.rb_lp_time_share = rb_time_share;
            bab.cuts_enabled = root_cuts;
            bab.component_solve = component_solve;
            bab.mir.relax_small_terms = relax_small_terms;
            bab.cut.gmi_max_tableau_trials = gmi_tableau_trials;
            bab.cut.rank_gmi_candidates = rank_gmi;
            bab.cut.integer_slack_gmi = integer_slack_gmi;
            bab.cut.integer_activity_basic_gmi = integer_slack_gmi_basic;
            bab.mir.aggregate = mir_aggregate;
            bab.mir.variable_bounds = mir_variable_bounds;
            bab.mir.probe_bounds = mir_probe_bounds;
            bab.mir.lifted_cover = mir_lifted_cover;
            bab.paper_reliability = !legacy_branching;
            bab.structural_presolve.enabled = milp_presolve;
            bab.structural_presolve.coefficient_strengthening = coef_strengthening;
            bab.root_restart = root_restart;
            bab.objective_face = objective_face;
            bab.feasibility_pump_root = fpump;
            if (fpump_time >= 0.0) {
                bab.feasibility_pump_time_s = fpump_time;
                bab.feasibility_pump_total_frac = 1.0;
            }
            bab.event_propagation = event_propagation;
            if (plunge_all) bab.hybrid_node_selection = true;
            if (plunge_depth > 0) bab.plunge_max_depth = plunge_depth;
            bab.structural_presolve.binary_row_support = binary_row_support;
            bab.structural_presolve.structural_fbbt = structural_fbbt;
            bab.structural_presolve.monotone_binary_pairs = monotone_binary_pairs;
            bab.structural_presolve.row_probe.enabled = structural_row_probing;
            bab.structural_presolve.graph_relation_probe = structural_graph_relations;
            bab.structural_presolve.graph_support_propagation = structural_graph_support;
            if (diverse_row_probing) bab.structural_presolve.row_probe.max_row_overlap = 0.5;
            if (structural_probe_time > 0.0) {
                auto& pre = bab.structural_presolve;
                pre.row_probe_total_time_s = structural_probe_time;
                pre.row_probe.time_limit_s = std::min(0.5, structural_probe_time);
                pre.row_probe_max_passes = 64;
                pre.row_probe.max_total_row_visits = static_cast<std::uint64_t>(
                    std::ceil(structural_probe_time) * 2000000.0);
            }
            bab.reduced_cost_strengthening = rc_strengthening;
            bab.node_lp_cutoff = node_lp_cutoff;
            bab.events_path = events_out;
            if (root_reduction_cap >= 0.0) bab.root_reduction_cap_s = root_reduction_cap;
            if (root_reduction_share >= 0.0) bab.root_reduction_share = root_reduction_share;
            if (root_cut_share >= 0.0) bab.root_cut_share = root_cut_share;
            if (root_cut_max >= 0.0) bab.root_cut_max_s = root_cut_max;
            if (cut_min_progress >= 0.0) bab.cut.min_progress_rel = cut_min_progress;
            bab.cut.rollback_stalled_rounds = cut_rollback;
            bab.cut.purge_nonbinding_cuts = cut_purge;
            bab.cut.warm_start_rounds = cut_warm_rounds;
            bab.cut.marginal_gate = cut_marginal_gate;
            if (cut_marginal_gate)
                std::fprintf(stderr,
                    "warning: --cut-marginal-gate is MEASURED UNSAFE. It drove "
                    "tree MIR to promote cuts that exclude the known optimum "
                    "on blend2 (23 with GCS, 72 without). Diagnostic use "
                    "only -- see CutOptions::marginal_gate.\n");
            if (cut_marginal_min >= 0.0)
                bab.cut.marginal_gate_min_rel = cut_marginal_min;
            if (cut_patience > 0)
                bab.cut.min_progress_patience = static_cast<int>(cut_patience);
            if (cut_max_rounds > 0) bab.cut.max_rounds = cut_max_rounds;
            if (cut_nnz_budget >= 0.0) bab.cut.pool_nnz_budget_factor = cut_nnz_budget;
            if (cut_max_density >= 0.0) bab.cut.pool_max_density = cut_max_density;
            if (cut_par_penalty >= 0.0) bab.cut.pool_parallelism_penalty = cut_par_penalty;
            if (cut_extra_scores >= 0.0) {
                bab.cut.pool_weight_sparsity = cut_extra_scores;
                bab.cut.pool_weight_low_locks = cut_extra_scores;
            }
            bab.conflict_propagation = conflict_propagation;
            bab.fixprop = fixprop_enabled;
            if (fixprop_time_given) {
                bab.fixprop_time_s = fixprop_time_s;
                bab.fixprop_root_frac = 1.0;
            }
            if (feasjump_time_given) {
                bab.feasibility_jump_time_s = feasjump_time_s;
                bab.feasibility_jump_seeded_time_s = feasjump_time_s;
            }
            if (feasjump_root_frac_given)
                bab.feasibility_jump_root_frac = feasjump_root_frac;
            bab.conflict_cut.enabled = conflict_cut;
            // --no-conflict-cut is the conflict-LEARNING family switch:
            // both the Mexi path and branch-trail nogoods go off together.
            bab.conflict_cut.nogood_cuts = conflict_cut && nogood_cuts;
            bab.conflict_cut.force_paper = conflict_cut_paper;
            if (conflict_cut_paper)
                bab.conflict_cut.mode = sor::search::ConflictCutMode::Paper;
            bab.dynsep.enabled = dynsep;
            bab.dynsep.model_path = dynsep_model;
            bab.dynsep.collect_labels = dynsep_collect;
            if (!dynsep_backend.empty()) {
                if (dynsep_backend == "auto")
                    bab.dynsep.backend = sor::search::DynSepBackend::Auto;
                else if (dynsep_backend == "gnn")
                    bab.dynsep.backend = sor::search::DynSepBackend::Gnn;
                else if (dynsep_backend == "ucb")
                    bab.dynsep.backend = sor::search::DynSepBackend::Ucb;
                else {
                    std::fprintf(stderr,
                                 "error: unknown --dynsep-backend '%s' "
                                 "(want auto|gnn|ucb)\n",
                                 dynsep_backend.c_str());
                    return 2;
                }
            }
            if (dynsep_ucb >= 0.0) bab.dynsep.ucb_c = dynsep_ucb;
            if (dynsep_max_optional >= 0)
                bab.dynsep.max_optional_arms = dynsep_max_optional;
            bab.l2sep.enabled = l2sep;
            bab.l2sep.model_path = l2sep_model;
            bab.hgtsm.enabled = hgtsm;
            bab.hgtsm.model_path = hgtsm_model;
            bab.hgtsm.prefer_linear = hgtsm_linear;
            bab.hgtsm.collect_labels = hgtsm_collect;
            if (hgtsm_sequence == "gru" || hgtsm_sequence == "GRU")
                bab.hgtsm.sequence = sor::search::HgtsmSequenceKind::Gru;
            else
                bab.hgtsm.sequence =
                    sor::search::HgtsmSequenceKind::TransformerLite;
            bab.tree_cut.gcs_enabled = gcs;
            bab.tree_cut.gcs_model_path = gcs_model;
            bab.tree_cut.gcs_prefer_heuristic = gcs_heuristic;
            if (gcs_reinject >= 0)
                bab.tree_cut.gcs_reinject_every_nodes =
                    static_cast<std::uint64_t>(gcs_reinject);
            // --max-iter is the public MILP node limit.  A node LP must not
            // inherit that value as its own pivot cap: doing so interrupted
            // the schedule root relaxation at 5000 pivots and left B&B with
            // no incumbent.  The B&B driver applies the global wall-clock
            // budget to each relaxation instead.
            bab.lp.max_iterations = 0;
            if (sx_opts.max_iterations != 0)
                bab.max_nodes = sx_opts.max_iterations;
            if (sx_opts.time_limit_s > 0.0)
                bab.time_limit_s = sx_opts.time_limit_s;
            if (tol_given) {
                bab.primal_feas_tol = tol;
                bab.int_tol = std::max(tol, 1e-9);
            }
            sor::search::LatticeSolveOutcome out;
            sor::search::PortfolioDiagnostics pdiag;
            if (milp_portfolio) {
                sor::search::PortfolioOptions popts;
                popts.workers = milp_portfolio_workers;
                out.raw = sor::search::solve_milp_portfolio(problem, bab, popts,
                                                             out.diag, pdiag);
                std::printf("portfolio:         %d worker(s), winner '%s', "
                            "%llu exchange(s)\n",
                            pdiag.workers_used, pdiag.winner.c_str(),
                            static_cast<unsigned long long>(
                                pdiag.incumbent_exchanges));
            } else {
                out = sor::search::solve_milp_lattice(problem, bab, lattice_reform);
            }
            if (lattice_reform) {
                if (out.reform_applied) {
                    std::printf("lattice reform:    applied (%s), kernel dim %d, "
                                "forced-zero cont %zu, fixed %zu, %s\n",
                                out.note.c_str(), out.kernel_dim,
                                out.forced_zero_cols, out.fixed_cols,
                                out.exact_equivalence ? "exact equivalence"
                                                      : "restriction");
                    if (out.certified) {
                        std::printf("lattice reform:    certified optimal against "
                                    "original LP relaxation bound\n");
                    } else if (out.fell_back) {
                        std::printf("lattice reform:    restricted run not "
                                    "certifiable; original re-solved\n");
                    }
                } else {
                    std::printf("lattice reform:    skipped (%s)\n",
                                out.note.c_str());
                }
            }
            const auto& diag = out.diag;
            const auto ev = sor::search::milp_evidence(diag, bab);
            const auto r = sor::certify::finalize_result(std::move(out.raw), ev);
            print_result(r);
            write_solution_out(solution_out, r);
            if (std::isfinite(diag.dual_bound)) {
                std::printf("dual bound:        %.10e\n", diag.dual_bound);
                std::printf("mip gap:           %.3e\n", diag.gap_rel);
            }
            std::printf("nodes:             %llu\n",
                        static_cast<unsigned long long>(diag.nodes));
            std::printf("gap prunes:        %llu\n",
                        static_cast<unsigned long long>(diag.gap_prunes));
            std::printf("para gap prunes:   %llu\n",
                        static_cast<unsigned long long>(diag.para_gap_prunes));
            std::printf("gap pruned integral LP: %llu\n",
                        static_cast<unsigned long long>(diag.gap_pruned_integral_lp));
            std::printf("lp solves:         %llu\n",
                        static_cast<unsigned long long>(diag.lp_solves));
            std::printf("lp fallbacks:      %llu\n",
                        static_cast<unsigned long long>(diag.lp_fallbacks));
            std::printf("root LP reused:    %llu\n",
                        static_cast<unsigned long long>(diag.root_lp_reuses));
            std::printf("  root LP prepared handoffs: %llu\n",
                        static_cast<unsigned long long>(diag.root_lp_session_handoffs));
            std::printf("root LP handoffs:  %llu\n",
                        static_cast<unsigned long long>(diag.root_lp_warm_handoffs));
            std::printf("warm_start_hits:   %llu / %llu attempts "
                        "(node LP dual warm; --basis-update product|ft)\n",
                        static_cast<unsigned long long>(diag.warm_start_hits),
                        static_cast<unsigned long long>(diag.warm_start_attempts));
            std::printf("node LP:            %llu iterations in %.1f ms "
                        "(%.1f iters/node, %.3f ms/node)\n",
                        static_cast<unsigned long long>(diag.lp_iterations),
                        diag.lp_ms,
                        diag.nodes ? double(diag.lp_iterations) / double(diag.nodes) : 0.0,
                        diag.nodes ? diag.lp_ms / double(diag.nodes) : 0.0);
            if (std::isfinite(diag.root_bound_before_cuts))
                std::printf("root bound:        %.10e -> %.10e after cuts\n",
                            diag.root_bound_before_cuts, diag.root_bound_after_cuts);
            std::printf("integer row roundings: %llu\n",
                        static_cast<unsigned long long>(diag.integer_row_roundings));
            std::printf("binary cover cuts: %llu\n",
                        static_cast<unsigned long long>(diag.binary_cover_cuts));
            std::printf("GMI cuts:          %llu in %d rounds\n",
                        static_cast<unsigned long long>(diag.gmi_cuts_added),
                        diag.cut_rounds);
            std::printf("root cuts:         %llu generated, %llu prefiltered "
                        "(malformed %llu, dense %llu, efficacy %llu, parallel %llu, "
                        "budget %llu), %llu selected, %llu gate-dropped; rows %llu "
                        "appended, %llu tightened, %llu active after rollback/purge\n",
                        (unsigned long long)diag.root_cuts_generated,
                        (unsigned long long)diag.root_cuts_prefilter_rejected,
                        (unsigned long long)diag.root_prefilter.rejected_malformed,
                        (unsigned long long)diag.root_prefilter.rejected_dense,
                        (unsigned long long)diag.root_prefilter.rejected_efficacy,
                        (unsigned long long)diag.root_prefilter.rejected_parallel,
                        (unsigned long long)diag.root_prefilter.rejected_budget,
                        (unsigned long long)diag.root_cuts_selected,
                        (unsigned long long)diag.root_cuts_gate_dropped,
                        (unsigned long long)diag.root_cut_rows_appended,
                        (unsigned long long)diag.root_cut_rows_tightened,
                        (unsigned long long)diag.root_cut_rows_active);
            std::printf("root cut LPs:      %llu iterations, %.1f ms; warm %llu / %llu "
                        "attempts; separate %.1f ms, select %.1f ms, apply %.1f ms\n",
                        (unsigned long long)diag.cut_lp_iterations, diag.cut_lp_ms,
                        (unsigned long long)diag.cut_lp_warm_hits,
                        (unsigned long long)diag.cut_lp_warm_attempts,
                        diag.cut_separate_ms, diag.cut_select_ms, diag.cut_apply_ms);
            std::printf("tree sep credit:   %.1f\n", diag.tree_sep_credit);
            std::printf("node cut resolves: %llu (%llu proved, %llu bound raises, %llu prunes, %.1f ms)\n",
                        (unsigned long long)diag.node_cut_resolves,
                        (unsigned long long)diag.node_cut_resolves_proved,
                        (unsigned long long)diag.node_cut_bound_raises,
                        (unsigned long long)diag.node_cut_prunes,
                        diag.node_cut_resolve_ms);
            if (diag.component_count > 0)
                std::printf("components:        %llu independent, %llu proved, %llu isolated columns, %.1f ms\n",
                            (unsigned long long)diag.component_count,
                            (unsigned long long)diag.components_solved,
                            (unsigned long long)diag.component_isolated, diag.component_solve_ms);
            std::printf("node LP deferral: %llu deferred, %llu retried, %llu abandoned\n",
                        (unsigned long long)diag.node_lp_deferred, (unsigned long long)diag.node_lp_retries,
                        (unsigned long long)diag.abandoned_unproved_nodes);
            std::printf("sub-MIP context:   %llu parent cut rows handed to children, %llu children seeded\n",
                        (unsigned long long)diag.sub_mip_context_cut_rows,
                        (unsigned long long)diag.sub_mip_context_seeded);
            std::printf("root primal:       %llu passes, %llu found the first incumbent, %.1f ms\n",
                        (unsigned long long)diag.root_primal_passes,
                        (unsigned long long)diag.root_primal_hits, diag.root_primal_ms);
            std::printf("SPP repair:        %llu attempts, %llu hits, %llu moves (%llu compound), %.1f ms\n",
                        (unsigned long long)diag.spp_attempts, (unsigned long long)diag.spp_hits,
                        (unsigned long long)diag.spp_moves, (unsigned long long)diag.spp_compound_moves,
                        diag.spp_ms);
            std::printf("conflict store:    %llu clauses added (%llu live, %llu rejected, %llu evicted, %llu from analysis), "
                        "%llu forced, %llu bounds tightened, %llu nodes closed\n",
                        (unsigned long long)diag.conflict_store.added,
                        (unsigned long long)diag.conflict_store.live,
                        (unsigned long long)diag.conflict_store.rejected,
                        (unsigned long long)diag.conflict_store.evicted,
                        (unsigned long long)diag.conflict_store_explained,
                        (unsigned long long)diag.conflict_store.propagations,
                        (unsigned long long)diag.conflict_store.tightenings,
                        (unsigned long long)diag.conflict_store.conflicts);
            std::printf("local cuts:        %llu rows inherited, %llu nodes solved with them (%llu kept the warm basis)\n",
                        (unsigned long long)diag.node_cuts_inherited,
                        (unsigned long long)diag.node_local_cut_nodes,
                        (unsigned long long)diag.node_local_cut_basis_kept);
            std::printf("tree cuts:         %llu nodes, %llu generated, %llu prefiltered, "
                        "%llu selected, %llu inserted, %llu inherited rows applied "
                        "(%.1f ms)\n",
                        (unsigned long long)diag.tree_cut_nodes,
                        (unsigned long long)diag.tree_cuts_generated,
                        (unsigned long long)diag.tree_cuts_prefilter_rejected,
                        (unsigned long long)diag.tree_local_cuts_selected,
                        (unsigned long long)diag.tree_local_cuts_inserted,
                        (unsigned long long)diag.node_cuts_locally_applied,
                        diag.tree_sep_ms);
            std::printf("  tree separation skipped by cost gate at %llu nodes\n",
                        (unsigned long long)diag.tree_sep_skipped_budget);
            std::printf("GMI trace:         %llu fractional basic candidates, "
                        "%llu missing bases, %llu invalid factors, %llu empty rows, "
                        "%llu free, %llu dynamism, %llu unviolated rejects, "
                        "%llu integer terms at fractional bounds\n",
                        static_cast<unsigned long long>(diag.gmi_candidates_considered),
                        static_cast<unsigned long long>(diag.gmi_missing_basis),
                        static_cast<unsigned long long>(diag.gmi_invalid_factor),
                        static_cast<unsigned long long>(diag.gmi_empty_rows),
                        static_cast<unsigned long long>(diag.gmi_rejected_free),
                        static_cast<unsigned long long>(diag.gmi_rejected_dynamism),
                        static_cast<unsigned long long>(diag.gmi_rejected_violation),
                        static_cast<unsigned long long>(diag.gmi_fractional_integer_bound_terms));
            std::printf("GMI dynamism:      %llu wide cuts repaired by relaxing small terms\n",
                        static_cast<unsigned long long>(diag.gmi_dynamism_repaired));
            std::printf("GMI tableau c-MIR: %llu cuts (%llu beat the GMI, %llu with no GMI)\n",
                        static_cast<unsigned long long>(diag.tableau_cmir_cuts),
                        static_cast<unsigned long long>(diag.tableau_cmir_won),
                        static_cast<unsigned long long>(diag.tableau_cmir_only));
            std::printf("GMI c-MIR:         %llu recovered / %llu attempted\n",
                        static_cast<unsigned long long>(diag.gmi_cmir_recovered),
                        static_cast<unsigned long long>(diag.gmi_cmir_attempted));
            if (integer_slack_gmi)
                std::printf("GMI integer rows:  %llu eligible row-rounds, %llu fractional basic "
                            "row activities, %llu integral nonbasic row terms\n",
                            static_cast<unsigned long long>(diag.gmi_integral_activity_rows),
                            static_cast<unsigned long long>(diag.gmi_integer_activity_candidates),
                            static_cast<unsigned long long>(diag.gmi_integer_activity_terms));
            std::printf("LNS (bandit):      %llu hits / %llu built / %llu attempts, "
                        "%llu child nodes, %llu budget blocks (%.1f ms)\n",
                        static_cast<unsigned long long>(diag.lns.hits),
                        static_cast<unsigned long long>(diag.lns.built),
                        static_cast<unsigned long long>(diag.lns.attempts),
                        static_cast<unsigned long long>(diag.sub_mip_nodes),
                        static_cast<unsigned long long>(diag.lns.budget_blocks),
                        diag.sub_mip_ms);
            std::printf("feasibility pump:  %llu attempts, %llu hits, %llu rounds, %llu LPs, "
                        "%llu flips, %llu restarts (%.1f ms)\n",
                        (unsigned long long)diag.fpump_attempts,
                        (unsigned long long)diag.fpump_hits,
                        (unsigned long long)diag.fpump_rounds,
                        (unsigned long long)diag.fpump_lp_solves,
                        (unsigned long long)diag.fpump_flips,
                        (unsigned long long)diag.fpump_restarts, diag.fpump_ms);
            std::printf("objective face:    %llu attempts, %llu hits, %llu exhausted (%.1f ms)\n",
                        (unsigned long long)diag.face_attempts,
                        (unsigned long long)diag.face_hits,
                        (unsigned long long)diag.face_exhausted, diag.face_ms);
            std::printf("LNS scheduling:    %llu root attempts, %llu cutoff rows\n",
                        (unsigned long long)diag.lns_root_attempts,
                        (unsigned long long)diag.lns_cutoff_rows);
            std::printf("sub-MIP calls:     %llu; build %.1f ms, child setup %.1f ms, "
                        "child search %.1f ms\n",
                        (unsigned long long)diag.sub_mip_calls, diag.sub_mip_build_ms,
                        diag.sub_mip_child_setup_ms, diag.sub_mip_child_search_ms);
            for (const auto& a : diag.lns.arms) {
                if (a.calls == 0) continue;
                std::printf("  arm %-12s calls %-5llu hits %-4llu "
                            "mean reward %.3f  fixing rate %.2f  %.2fs\n",
                            sor::search::to_string(a.kind),
                            static_cast<unsigned long long>(a.calls),
                            static_cast<unsigned long long>(a.hits),
                            a.calls ? a.reward_sum / double(a.calls) : 0.0,
                            a.fixing_rate, a.seconds);
            }
            std::printf("Balans:            %llu hits / %llu built / %llu attempts, "
                        "%llu budget blocks (%.2fs)\n",
                        static_cast<unsigned long long>(diag.balans.hits),
                        static_cast<unsigned long long>(diag.balans.built),
                        static_cast<unsigned long long>(diag.balans.attempts),
                        static_cast<unsigned long long>(diag.balans.budget_blocks),
                        diag.balans.seconds);
            for (const auto& a : diag.balans.arms) {
                if (a.calls == 0) continue;
                std::printf("  balans %-12s calls %-5llu hits %-4llu "
                            "mean reward %.3f  fixing rate %.2f  %.2fs\n",
                            sor::search::to_string(a.kind),
                            static_cast<unsigned long long>(a.calls),
                            static_cast<unsigned long long>(a.hits),
                            a.calls ? a.reward_sum / double(a.calls) : 0.0,
                            a.fixing_rate, a.seconds);
            }
            std::printf("kernel pump:       found=%d  pumps %llu  lp %llu  "
                        "kernel %llu  buckets %llu  (%.1f ms)\n",
                        diag.kernel_pump.found ? 1 : 0,
                        static_cast<unsigned long long>(diag.kernel_pump.pumps),
                        static_cast<unsigned long long>(diag.kernel_pump.lp_solves),
                        static_cast<unsigned long long>(diag.kernel_pump.kernel_size),
                        static_cast<unsigned long long>(diag.kernel_pump.buckets),
                        diag.kernel_pump.ms);
            std::printf("MRENS:             %llu hits / %llu built / %llu attempts, "
                        "refs %llu, fixed %llu free %llu (%.2fs)\n",
                        static_cast<unsigned long long>(diag.mrens.hits),
                        static_cast<unsigned long long>(diag.mrens.built),
                        static_cast<unsigned long long>(diag.mrens.attempts),
                        static_cast<unsigned long long>(diag.mrens.refs_used),
                        static_cast<unsigned long long>(diag.mrens.fixed),
                        static_cast<unsigned long long>(diag.mrens.free_integer),
                        diag.mrens.seconds);
            std::printf("BTBS-LNS:          %llu hits / %llu built / %llu attempts, "
                        "fixed %llu free %llu (%.2fs)\n",
                        static_cast<unsigned long long>(diag.btbs.hits),
                        static_cast<unsigned long long>(diag.btbs.built),
                        static_cast<unsigned long long>(diag.btbs.attempts),
                        static_cast<unsigned long long>(diag.btbs.fixed),
                        static_cast<unsigned long long>(diag.btbs.free_integer),
                        diag.btbs.seconds);
            std::printf("CL-TLNS:           %llu hits / %llu built / %llu attempts, "
                        "fixed %llu free %llu (%.2fs)\n",
                        static_cast<unsigned long long>(diag.cl_tlns.hits),
                        static_cast<unsigned long long>(diag.cl_tlns.built),
                        static_cast<unsigned long long>(diag.cl_tlns.attempts),
                        static_cast<unsigned long long>(diag.cl_tlns.fixed),
                        static_cast<unsigned long long>(diag.cl_tlns.free_integer),
                        diag.cl_tlns.seconds);
            std::printf("heuristics:        %.1f ms total, %.1f ms blocked "
                        "(%llu denials)\n",
                        sor::search::heuristic_spent_ms(diag),
                        diag.heuristic_budget_blocked_ms,
                        static_cast<unsigned long long>(diag.heuristic_budget_blocks));
            std::printf("direct rounding:   %llu calls in %.1f ms (%llu node rounds)\n",
                        static_cast<unsigned long long>(diag.rounding_calls),
                        diag.rounding_ms,
                        static_cast<unsigned long long>(diag.node_rounding_rounds));
            std::printf("conflict learning: %llu mexi cuts global "
                        "(%llu aborted), %llu nogoods global\n",
                        static_cast<unsigned long long>(diag.conflict_cuts_global),
                        static_cast<unsigned long long>(diag.conflict_cut_diag.aborted),
                        static_cast<unsigned long long>(diag.nogood_cuts_global));
            if (diag.conflict_cut_diag.attempts > 0) {
                const auto& cd = diag.conflict_cut_diag;
                std::printf("  conflict derivation: %llu attempts, %llu learned, "
                            "%llu validation-rejected; aborts scope %llu, "
                            "seed %llu, trail %llu (missing bound %llu), reason %llu, "
                            "resolution %llu, final %llu\n",
                            static_cast<unsigned long long>(cd.attempts),
                            static_cast<unsigned long long>(cd.learned),
                            static_cast<unsigned long long>(cd.validation_rejected),
                            static_cast<unsigned long long>(cd.aborted_local_scope),
                            static_cast<unsigned long long>(cd.aborted_seed),
                            static_cast<unsigned long long>(cd.aborted_trail),
                            static_cast<unsigned long long>(cd.aborted_trail_missing_bound),
                            static_cast<unsigned long long>(cd.aborted_reason),
                            static_cast<unsigned long long>(cd.aborted_resolution),
                            static_cast<unsigned long long>(cd.aborted_final));
            }
            std::printf("feasibility jump:  %llu attempts, %llu hits, %llu moves, "
                        "%llu reweights, %llu restarts, best %zu violated rows "
                        "(%.1f ms)\n",
                        static_cast<unsigned long long>(diag.feasjump_attempts),
                        static_cast<unsigned long long>(diag.feasjump_hits),
                        static_cast<unsigned long long>(diag.feasjump_moves),
                        static_cast<unsigned long long>(diag.feasjump_weight_updates),
                        static_cast<unsigned long long>(diag.feasjump_restarts),
                        diag.feasjump_best_violated_rows,
                        diag.feasjump_ms);
            std::printf("gpu binary (G1):   %llu attempted, %llu eligible, %llu found "
                        "(init %.1f ms, search %.1f ms)\n",
                        static_cast<unsigned long long>(diag.gpu_bin_attempted),
                        static_cast<unsigned long long>(diag.gpu_bin_eligible),
                        static_cast<unsigned long long>(diag.gpu_bin_found),
                        diag.gpu_bin_init_ms, diag.gpu_bin_ms);
            std::printf("probing:           %llu probes, %llu fixings, "
                        "%llu implications, %llu bound tightenings%s (%.1f ms)\n",
                        static_cast<unsigned long long>(diag.conflict.probes),
                        static_cast<unsigned long long>(diag.conflict.probe_fixings),
                        static_cast<unsigned long long>(diag.conflict.probe_implications),
                        static_cast<unsigned long long>(diag.conflict.probe_tightenings),
                        diag.conflict.probing_truncated ? " [truncated]" : "",
                        diag.conflict.probe_ms);
            std::printf("conflict graph:    %llu row cliques, %llu edges\n",
                        static_cast<unsigned long long>(diag.conflict.row_cliques),
                        static_cast<unsigned long long>(diag.conflict.edges));
            {
                const auto& mp = diag.mip_presolve_diag;
                std::printf("mip-presolve ms:   dual-fix %.1f, probe/graph %.1f, "
                            "clique-probe %.1f, gf2 %.1f, comps %.1f, "
                            "implied-int %.1f, other %.1f\n",
                            mp.ms_dual_fix, mp.ms_conflict_graph, mp.ms_clique_probe,
                            mp.ms_gf2, mp.ms_components, mp.ms_implied_int,
                            mp.ms - (mp.ms_dual_fix + mp.ms_conflict_graph +
                                     mp.ms_clique_probe + mp.ms_gf2 +
                                     mp.ms_components + mp.ms_implied_int));
            }
            std::printf("mip-presolve:      dual-fix %llu, clique-probe %llu/%llu "
                        "fix/tight, gf2 %llu fix, comps %llu%s, implied-int %llu, "
                        "obbt lp %llu / fbbt %llu, reduced %.1f%%%s "
                        "restarts %llu (%.1f ms)\n",
                        static_cast<unsigned long long>(
                            diag.mip_presolve_diag.dual_fix.fixings),
                        static_cast<unsigned long long>(
                            diag.mip_presolve_diag.clique_probe.fixings),
                        static_cast<unsigned long long>(
                            diag.mip_presolve_diag.clique_probe.tightenings),
                        static_cast<unsigned long long>(
                            diag.mip_presolve_diag.gf2.fixings),
                        static_cast<unsigned long long>(
                            diag.mip_presolve_diag.components.n_components),
                        diag.mip_presolve_diag.components.disconnected
                            ? " [disc]"
                            : "",
                        static_cast<unsigned long long>(
                            diag.mip_presolve_diag.implied_int.total),
                        static_cast<unsigned long long>(
                            diag.mip_presolve_diag.obbt.lp_tightenings),
                        static_cast<unsigned long long>(
                            diag.mip_presolve_diag.obbt.fbbt_tightenings),
                        100.0 * diag.mip_presolve_diag.reduction_frac,
                        diag.mip_restart_recommended ? " [restart]" : "",
                        static_cast<unsigned long long>(
                            diag.mip_presolve_diag.restart_rounds),
                        diag.mip_presolve_diag.ms);
            if (diag.mip_presolve_diag.infeasible) {
                const auto& pd = diag.mip_presolve_diag;
                std::printf("  presolve infeasibility source: dual-fix %d, "
                            "probe %d, clique %d, gf2 %d, component %d "
                            "(dual-fix %llu, fbbt %llu, enumerated %llu)\n",
                            pd.dual_fix.infeasible, pd.conflict.infeasible,
                            pd.clique_probe.infeasible, pd.gf2.infeasible,
                            pd.components.infeasible,
                            static_cast<unsigned long long>(pd.components.dual_fixings),
                            static_cast<unsigned long long>(pd.components.fbbt_tightenings),
                            static_cast<unsigned long long>(pd.components.enum_components));
            }
            std::printf("symmetry:          %llu orbits (%llu binary), "
                        "%llu orbital fixings; reflection=%s folding=%s "
                        "(%.1f ms)\n",
                        static_cast<unsigned long long>(diag.symmetry_diag.n_orbits),
                        static_cast<unsigned long long>(
                            diag.symmetry_diag.n_binary_orbits),
                        static_cast<unsigned long long>(
                            diag.symmetry_diag.orbital_fixings),
                        diag.symmetry_diag.reflection_applied
                            ? "on"
                            : (reflection ? "off" : "disabled"),
                        diag.symmetry_diag.folding_applied
                            ? "on"
                            : (folding ? "off" : "disabled"),
                        diag.symmetry_diag.ms);
            std::printf("cut loop:          %.1f ms of the %.0f s budget\n",
                        diag.cut_loop_ms, bab.time_limit_s);
            // Node-cut validity, always printed when a reference is loaded,
            // as three separate populations. Printed unconditionally so the
            // absence of this line means "no reference", never "nothing
            // found" -- a deleted detector once reported a silent zero for a
            // whole campaign.
            if (!verify_cuts_path.empty()) {
                std::printf("node cut validity: generated %llu invalid, "
                            "rejected %llu, INSERTED %llu; "
                            "%llu abstained (ref outside node); "
                            "%llu local rows applied\n",
                            static_cast<unsigned long long>(
                                diag.node_cuts_invalid_generated),
                            static_cast<unsigned long long>(
                                diag.node_cuts_invalid_rejected),
                            static_cast<unsigned long long>(
                                diag.node_cuts_invalid_inserted),
                            static_cast<unsigned long long>(
                                diag.node_cuts_ref_outside_node),
                            static_cast<unsigned long long>(
                                diag.node_cuts_locally_applied));
            }
            std::printf("cut rollback:      %d rounds rolled back, %llu rows "
                        "retracted, %llu rows re-tightened\n",
                        diag.cut_rounds_rolled_back,
                        static_cast<unsigned long long>(diag.cut_rows_retracted),
                        static_cast<unsigned long long>(diag.cut_rows_retightened));
            {
                static const char* kFamName[6] = {"gmi", "mir", "cover",
                                                  "clique", "vub", "zerohalf"};
                const auto& mg = diag.marginal_gate;
                if (mg.ran) {
                    std::printf("marginal gate:     %d probe LPs in %.1f ms, "
                                "%d cuts dropped%s\n",
                                mg.probe_solves, mg.probe_ms, mg.cuts_dropped,
                                mg.aborted ? "  (ABORTED: a probe LP did not "
                                             "prove; no family disabled)" : "");
                    for (int f = 0; f < 6; ++f) {
                        if (mg.cuts_offered[f] == 0) continue;
                        std::printf("  [marg] %-9s offered %3d  contribution ",
                                    kFamName[f], mg.cuts_offered[f]);
                        if (std::isnan(mg.marginal_rel[f]))
                            std::printf("%-10s", "not probed");
                        else
                            std::printf("%-10.4g", mg.marginal_rel[f]);
                        std::printf("%s\n", mg.disabled[f] ? "  DISABLED" : "");
                    }
                }
            }
            std::printf("cut purge:         %llu rows purged, %llu kept, "
                        "%llu skipped stale (retract+purge pass %.3f ms)\n",
                        static_cast<unsigned long long>(diag.cut_rows_purged),
                        static_cast<unsigned long long>(diag.cut_rows_kept),
                        static_cast<unsigned long long>(
                            diag.cut_purge_skipped_stale_duals),
                        diag.cut_retract_ms);
            // Per-round ledger: what each root cut round cost and what the
            // next re-solve showed it bought. This is the measurement the
            // realised-gain gate acts on, so it is reported unconditionally.
            for (const auto& tr : diag.cut_round_trace) {
                std::printf("  [round %2d] bound %- 18.10g gain_rel %-10.3g "
                            "sel %3d rows +%-3d tight %-3d  "
                            "gmi %d mir %d cov %d clq %d vub %d zh %d%s"
                            "  | lp %llu it %llu refac %.0f ms, sep %.0f ms, +%ld nnz, mir depth %d\n",
                            tr.round, tr.bound, tr.gain_rel, tr.cuts_selected,
                            tr.rows_added, tr.rows_tightened, tr.gmi, tr.mir,
                            tr.cover, tr.clique, tr.vub, tr.zerohalf,
                            tr.rolled_back ? "  ROLLED BACK" : "",
                            static_cast<unsigned long long>(tr.lp_iterations),
                            static_cast<unsigned long long>(tr.refactorizations),
                            tr.lp_ms, tr.separate_ms, tr.added_nnz, tr.mir_depth);
            }
            std::printf("MIR cuts:          %llu of %llu candidates added; "
                        "%llu bases, rejected %llu frac / %llu dyn / %llu unviolated\n",
                        static_cast<unsigned long long>(diag.mir_cuts_added),
                        static_cast<unsigned long long>(diag.mir_candidates),
                        static_cast<unsigned long long>(diag.mir.bases_built),
                        static_cast<unsigned long long>(diag.mir.rejected_fractionality),
                        static_cast<unsigned long long>(diag.mir.rejected_dynamism),
                        static_cast<unsigned long long>(diag.mir.rejected_not_violated));
            std::printf("MIR dynamism:      %llu wide cuts repaired by relaxing small terms\n",
                        static_cast<unsigned long long>(diag.mir.dynamism_repaired));
            std::printf("MIR lifted covers: %llu bases with a cover, %llu kept over c-MIR\n",
                        static_cast<unsigned long long>(diag.mir.lifted_cover_bases),
                        static_cast<unsigned long long>(diag.mir.lifted_cover_cuts));
            if (mir_probe_bounds)
                std::printf("MIR probe bounds: %llu candidates, %llu substitutions\n",
                            static_cast<unsigned long long>(diag.mir.probe_bound_candidates),
                            static_cast<unsigned long long>(diag.mir.probe_bound_substitutions));
            std::printf("lifted covers:     %llu of %llu candidates added; "
                        "%llu covers from %llu knapsacks, %llu lifted coefs\n",
                        static_cast<unsigned long long>(diag.lifted_cover_cuts_added),
                        static_cast<unsigned long long>(diag.lifted_cover_candidates),
                        static_cast<unsigned long long>(diag.cover.covers_found),
                        static_cast<unsigned long long>(diag.cover.knapsacks_built),
                        static_cast<unsigned long long>(diag.cover.lifted_coefficients));
            std::printf("implied bounds:    %llu recorded; %llu of %llu VUB "
                        "cut candidates added\n",
                        static_cast<unsigned long long>(diag.conflict.implied_bounds),
                        static_cast<unsigned long long>(diag.implied_bound_cuts_added),
                        static_cast<unsigned long long>(diag.implied_bound_cut_candidates));
            std::printf("clique cuts:       %llu of %llu candidates added; "
                        "node propagation %llu tightenings, %llu prunes\n",
                        static_cast<unsigned long long>(diag.clique_cuts_added),
                        static_cast<unsigned long long>(diag.clique_cut_candidates),
                        static_cast<unsigned long long>(diag.conflict_prop_tightenings),
                        static_cast<unsigned long long>(diag.conflict_prop_prunes));
            std::printf("root snapshots:    %llu published; child context %llu rows (%llu refreshes); "
                        "pump ran on the snapshot model %llu times; "
                        "%llu children started from the parent basis\n",
                        static_cast<unsigned long long>(diag.root_snapshots),
                        static_cast<unsigned long long>(diag.snapshot_context_rows),
                        static_cast<unsigned long long>(diag.snapshot_child_refreshes),
                        static_cast<unsigned long long>(diag.fpump_on_snapshot),
                        static_cast<unsigned long long>(diag.sub_mip_basis_carried));
            for (const auto& m : diag.milestones) std::printf("milestone %s\n", m.c_str());
            std::printf("branching evidence: %llu objective samples, %llu closures; ignored "
                        "%llu incomplete, %llu repeat; %llu re-solved children replaced their sample; "
                        "domain probes %llu nodes (%llu runs, %llu closed, %llu deductions)\n",
                        static_cast<unsigned long long>(diag.bs_objective_samples),
                        static_cast<unsigned long long>(diag.bs_closure_samples),
                        static_cast<unsigned long long>(diag.bs_ignored_incomplete),
                        static_cast<unsigned long long>(diag.bs_ignored_repeat),
                        static_cast<unsigned long long>(diag.bs_replaced),
                        static_cast<unsigned long long>(diag.domain_probe_candidates),
                        static_cast<unsigned long long>(diag.domain_probe_runs),
                        static_cast<unsigned long long>(diag.domain_probe_closures),
                        static_cast<unsigned long long>(diag.domain_probe_deductions));
            if (diag.probing_resumed)
                std::printf("probing carry:     resumed from presolve; %llu binaries already probed\n",
                            static_cast<unsigned long long>(diag.probing_carried_probed));
            std::printf("fixpoint rows:    %llu incremental passes, %llu full passes\n",
                        static_cast<unsigned long long>(diag.fixpoint_incremental_row_steps),
                        static_cast<unsigned long long>(diag.fixpoint_full_row_steps));
            std::printf("domain fixpoint:  %llu stable, %llu pending, %llu infeasible; "
                        "%llu narrowing steps\n",
                        static_cast<unsigned long long>(diag.fixpoint_stable),
                        static_cast<unsigned long long>(diag.fixpoint_pending),
                        static_cast<unsigned long long>(diag.fixpoint_infeasible),
                        static_cast<unsigned long long>(diag.prop_fixpoint_rounds));
            std::printf("cut pool:          %llu inserted, %llu duplicate, "
                        "%llu dominated, %llu parallel-rejected, %llu aged, "
                        "%llu evicted\n",
                        static_cast<unsigned long long>(diag.cut_pool_inserted),
                        static_cast<unsigned long long>(diag.cut_pool_duplicates),
                        static_cast<unsigned long long>(diag.cut_pool_dominated),
                        static_cast<unsigned long long>(diag.cut_pool_parallel_rejections),
                        static_cast<unsigned long long>(diag.cut_pool_aged_out),
                        static_cast<unsigned long long>(diag.cut_pool_evicted));
            std::printf("strong branch LPs: %llu  (pseudocost updates %llu, "
                        "%llu iterations, %.1f ms; %llu proved, %llu infeasible, "
                        "%llu unproved)\n",
                        static_cast<unsigned long long>(diag.strong_branch_solves),
                        static_cast<unsigned long long>(diag.pseudocost_updates),
                        static_cast<unsigned long long>(diag.strong_branch_iterations),
                        diag.strong_branch_ms,
                        static_cast<unsigned long long>(diag.strong_branch_proved),
                        static_cast<unsigned long long>(diag.strong_branch_infeasible),
                        static_cast<unsigned long long>(diag.strong_branch_unproved));
            std::printf("  probe ms:        prep %.1f  factor %.1f  loop %.1f  "
                        "simplex total %.1f  (factor reuses %llu, dse rebuilds %llu)\n",
                        diag.strong_branch_prep_ms, diag.strong_branch_factor_ms,
                        diag.strong_branch_loop_ms, diag.strong_branch_simplex_ms,
                        (unsigned long long)diag.strong_branch_factor_reuses,
                        (unsigned long long)diag.strong_branch_dse_rebuilds);
            {
                const auto& ps = diag.structural_presolve;
                std::printf("milp presolve:     %s  rows %d -> %d, cols %d -> %d "
                            "(%.1f ms, %d rounds)\n",
                            diag.structural_presolve_applied ? "applied"
                                : (ps.infeasible ? "infeasible->original"
                                                 : "not applied"),
                            ps.rows_before, ps.rows_after, ps.cols_before,
                            ps.cols_after, ps.ms, ps.rounds);
                std::printf("  reductions:      fixed %llu singleton-rows %llu "
                            "redundant-rows %llu (by activity %llu) bounds %llu "
                            "coefs %llu postsolve-fail %llu\n",
                            (unsigned long long)ps.fixed_cols,
                            (unsigned long long)ps.singleton_rows,
                            (unsigned long long)ps.redundant_rows,
                            (unsigned long long)ps.redundant_by_activity,
                            (unsigned long long)ps.bounds_tightened,
                            (unsigned long long)ps.coefs_tightened,
                            (unsigned long long)diag.structural_postsolve_failures);
                std::printf("structural FBBT: %llu tightenings, %d sweeps, %llu terms\n",
                            (unsigned long long)ps.structural_fbbt_tightenings,
                            ps.structural_fbbt_rounds,
                            (unsigned long long)ps.structural_fbbt_work);
                std::printf("monotone binary pairs: %d saturated\n", ps.monotone_pairs_saturated);
                std::printf("row support probing: %llu rows, %llu assignments (%llu rejected), "
                            "%llu visits, %llu fixings, %d substitutions (%d numerical rejects), "
                            "%d passes%s (%.1f ms)\n",
                            (unsigned long long)ps.row_probe_rows,
                            (unsigned long long)ps.row_probe_assignments,
                            (unsigned long long)ps.row_probe_infeasible_assignments,
                            (unsigned long long)ps.row_probe_visits,
                            (unsigned long long)ps.row_probe_fixings, ps.binary_substitutions,
                            ps.binary_substitution_numerical_rejects, ps.row_probe_passes,
                            ps.row_probe_truncated ? " [truncated]" : "", ps.row_probe_ms);
                if (ps.row_probe_overlap_skipped)
                    std::printf("row support diversity: %llu overlapping groups skipped\n",
                        (unsigned long long)ps.row_probe_overlap_skipped);
                if (ps.row_probe_graph_ms > 0.0)
                    std::printf("row support graph: %llu edges, %llu visits (%.1f ms setup)\n",
                        (unsigned long long)ps.row_probe_graph_edges,
                        (unsigned long long)ps.row_probe_graph_visits,
                        ps.row_probe_graph_ms);
                if (ps.graph_probe_ms > 0.0)
                    std::printf("structural graph: %llu edges, %llu relations (%.1f ms)\n",
                        (unsigned long long)ps.graph_probe_edges,
                        (unsigned long long)ps.graph_probe_relations,
                        ps.graph_probe_ms);
                std::printf("binary row support: %llu rows, %llu assignments, %llu terms, %d fixings\n",
                            (unsigned long long)ps.binary_rows_checked,
                            (unsigned long long)ps.binary_assignments_checked,
                            (unsigned long long)ps.binary_row_work,
                            ps.binary_row_fixings);
            }
            std::printf("node LP split:     setup %.1f ms, simplex %.1f ms (prep %.1f, loop %.1f), "
                        "%llu DSE rebuilds, %llu refactorizations\n",
                        diag.ms_node_setup, diag.node_lp_simplex_ms, diag.node_lp_prep_ms,
                        diag.node_lp_loop_ms,
                        (unsigned long long)diag.node_lp_dse_rebuilds,
                        (unsigned long long)diag.node_lp_refactorizations);
            std::printf("node LP session:   %llu builds, %llu solves; start state: %llu from last "
                        "solve, %llu from checkpoint (%llu unusable); checkpoints %llu made, "
                        "%llu evicted, %llu too large, peak %.1f MiB, %.1f ms copying\n",
                        (unsigned long long)diag.lp_session_builds,
                        (unsigned long long)diag.lp_session_solves,
                        (unsigned long long)diag.checkpoint_immediate,
                        (unsigned long long)diag.checkpoint_cached,
                        (unsigned long long)diag.checkpoint_unusable,
                        (unsigned long long)diag.checkpoints_created,
                        (unsigned long long)diag.checkpoints_evicted,
                        (unsigned long long)diag.checkpoints_declined,
                        double(diag.checkpoint_peak_bytes) / (1024.0 * 1024.0),
                        diag.checkpoint_copy_ms);
            std::printf("root restarts:     %llu (last: %llu columns fixed; re-presolve removed "
                        "%llu rows, %llu columns)\n",
                        (unsigned long long)diag.root_restarts,
                        (unsigned long long)diag.restart_columns_fixed,
                        (unsigned long long)diag.restart_rows_removed,
                        (unsigned long long)diag.restart_cols_removed);
            std::printf("cut transactions:  %llu batches committed, %llu deferred and rolled back\n",
                        (unsigned long long)diag.cut_batches_committed,
                        (unsigned long long)diag.cut_batches_deferred);
            std::printf("local-row sessions: %llu built, %llu node LPs solved on them; prepared-LP "
                        "cache %llu hits, %llu evictions, %llu key collisions\n",
                        (unsigned long long)diag.local_session_builds,
                        (unsigned long long)diag.local_session_solves,
                        (unsigned long long)diag.session_cache_hits,
                        (unsigned long long)diag.session_cache_evictions,
                        (unsigned long long)diag.session_key_collisions);
            std::printf("clause outcomes:   %llu inserted, %llu unit, %llu present, %llu redundant, "
                        "%llu rejected, %llu contradictions\n",
                        (unsigned long long)diag.clause_inserted, (unsigned long long)diag.clause_units,
                        (unsigned long long)diag.clause_present, (unsigned long long)diag.clause_redundant,
                        (unsigned long long)diag.clause_rejected,
                        (unsigned long long)diag.clause_contradictions);
            std::printf("farkas conflicts:  %llu LP-infeasible explanations (%llu from probes), %llu literals, "
                        "%llu bounds relaxed to the root\n",
                        (unsigned long long)diag.farkas_clauses,
                        (unsigned long long)diag.farkas_probe_clauses,
                        (unsigned long long)diag.farkas_literals,
                        (unsigned long long)diag.farkas_relaxed_bounds);
            std::printf("restart carry:     %llu clauses carried, %llu dropped, %llu emptied a box; "
                        "%llu cut rows; incumbent re-accepted %llu times, %llu outside the box, %llu hint hits\n",
                        (unsigned long long)diag.restart_clauses_carried,
                        (unsigned long long)diag.restart_clauses_dropped,
                        (unsigned long long)diag.restart_clauses_emptied,
                        (unsigned long long)diag.restart_cut_rows_carried,
                        (unsigned long long)diag.restart_incumbent_carried,
                        (unsigned long long)diag.restart_incumbent_rejected,
                        (unsigned long long)diag.restart_hint_hits);
            std::printf("abandoned nodes:   %llu (LP neither proved nor usable)\n",
                        (unsigned long long)diag.abandoned_unproved_nodes);
            std::printf("bound evidence:    root certified %.10e (%llu raises); "
                        "%llu unproved node bounds kept, %llu unsearched regions "
                        "folded\n",
                        diag.root_certified_bound,
                        (unsigned long long)diag.root_bound_raises,
                        (unsigned long long)diag.unproved_bounds_kept,
                        (unsigned long long)diag.unproved_regions_folded);
            std::printf("node LP non-loop:  first factor %.1f ms, set-up %.1f ms (DSE rebuild %.1f), "
                        "post-solve %.1f ms, safe bound %.1f ms; DSE weights reused %llu; "
                        "factors adopted %llu\n",
                        diag.node_lp_first_factor_ms, diag.node_lp_after_factor_ms,
                        diag.node_lp_dse_ms, diag.node_lp_post_ms, diag.node_lp_safe_bound_ms,
                        (unsigned long long)diag.node_lp_dse_reuses,
                        (unsigned long long)diag.node_lp_factor_reuses);
            std::printf("factor reuse rejects: carrier empty %llu, matrix null %llu, "
                        "rows mismatch %llu, preparation mismatch %llu, basis mismatch %llu; skipped refill "
                        "(primal clean-up) %llu\n",
                        (unsigned long long)diag.node_lp_factor_reuse_carrier_empty,
                        (unsigned long long)diag.node_lp_factor_reuse_matrix_null,
                        (unsigned long long)diag.node_lp_factor_reuse_rows_mismatch,
                        (unsigned long long)diag.node_lp_factor_reuse_preparation_mismatch,
                        (unsigned long long)diag.node_lp_factor_reuse_basis_mismatch,
                        (unsigned long long)diag.node_lp_factor_reuse_skipped_refill_primal_cleanup);
            {
                int unavailable = 0, partial = 0;
                for (const auto& c : sor::search::milp_capability_inventory()) {
                    if (c.status == sor::search::CapabilityStatus::Unavailable) ++unavailable;
                    if (c.status == sor::search::CapabilityStatus::Partial) ++partial;
                }
                std::printf("capabilities:      %d partial, %d unavailable "
                            "(see --milp-capabilities)\n", partial, unavailable);
            }
            std::printf("lagrangian bounds: %llu unproved LPs checked, %llu pruned, %llu bounds raised\n",
                        static_cast<unsigned long long>(diag.lagrangian_bound_checks),
                        static_cast<unsigned long long>(diag.lagrangian_prunes),
                        static_cast<unsigned long long>(diag.lagrangian_bound_raises));
            std::printf("node LP cutoff:    %llu early stops, %llu re-solved\n",
                        static_cast<unsigned long long>(diag.node_lp_cutoff_exits),
                        static_cast<unsigned long long>(diag.node_lp_cutoff_resolves));
            std::printf("rc strengthening:  %llu passes, %llu bounds tightened, %llu fixed\n",
                        (unsigned long long)diag.rc_strengthen_nodes,
                        (unsigned long long)diag.rc_bounds_tightened,
                        (unsigned long long)diag.rc_columns_fixed);
            std::printf("certified root RC: %llu excluded boxes checked (%.1f ms)\n",
                        (unsigned long long)diag.rc_certificate_checks, diag.rc_strengthening_ms);
            std::printf("sb deductions:     %llu bounds tightened, %llu nodes closed; "
                        "%llu nodes branched by the session selector\n",
                        (unsigned long long)diag.sb_domain_reductions,
                        (unsigned long long)diag.sb_nodes_closed,
                        (unsigned long long)diag.rb_reliable_nodes);
            std::printf("reliability:       %llu nodes branched by the pseudocost/"
                        "strong-branch selector (%llu with probes)\n",
                        static_cast<unsigned long long>(diag.rb_nodes),
                        static_cast<unsigned long long>(diag.rb_nodes_with_sb));
            std::printf("bab threads:       %d\n", diag.para_bab.threads_used);
            if (diag.para_bab.threads_used > 1) {
                std::printf("Para-B&B:          %d threads, %llu phases, "
                            "%llu parallel expansions, %llu syncs\n",
                            diag.para_bab.threads_used,
                            static_cast<unsigned long long>(diag.para_bab.phases),
                            static_cast<unsigned long long>(
                                diag.para_bab.parallel_expansions),
                            static_cast<unsigned long long>(diag.para_bab.syncs));
                if (diag.para_bab.activated)
                    std::printf("  cost model:      activated at %.3fs / node "
                                "%llu (%llu serial nodes first)\n",
                                diag.para_bab.activated_at_s,
                                static_cast<unsigned long long>(
                                    diag.para_bab.activated_at_node),
                                static_cast<unsigned long long>(
                                    diag.para_bab.serial_nodes));
                else
                    std::printf("  cost model:      never activated -- stayed "
                                "serial with plunging\n");
            }
            std::printf("branch policy:     resolved=%s last=%s\n"
                        "  sparse-sb picks/fallbacks/samples: %llu / %llu / %llu\n"
                        "  sc-milp picks/fallbacks/samples:   %llu / %llu / %llu\n"
                        "  lifted picks/fallbacks/refits/samples: %llu / %llu / %llu / %llu\n"
                        "  planbb picks/fallbacks/lookaheads/mcts-sims/samples: "
                        "%llu / %llu / %llu / %llu / %llu%s\n",
                        sor::search::branch_strategy_name(diag.branch_strategy_resolved),
                        diag.last_branch_policy.empty() ? "-"
                                                        : diag.last_branch_policy.c_str(),
                        static_cast<unsigned long long>(diag.sparse_sb_picks),
                        static_cast<unsigned long long>(diag.sparse_sb_fallbacks),
                        static_cast<unsigned long long>(diag.sparse_sb_samples),
                        static_cast<unsigned long long>(diag.sc_milp_picks),
                        static_cast<unsigned long long>(diag.sc_milp_fallbacks),
                        static_cast<unsigned long long>(diag.sc_milp_samples),
                        static_cast<unsigned long long>(diag.lifted_picks),
                        static_cast<unsigned long long>(diag.lifted_fallbacks),
                        static_cast<unsigned long long>(diag.lifted_refits),
                        static_cast<unsigned long long>(diag.lifted_samples),
                        static_cast<unsigned long long>(diag.planbb_picks),
                        static_cast<unsigned long long>(diag.planbb_fallbacks),
                        static_cast<unsigned long long>(diag.planbb_lookaheads),
                        static_cast<unsigned long long>(diag.planbb_mcts_sims),
                        static_cast<unsigned long long>(diag.planbb_samples),
                        diag.planbb_paper ? " (paper)" : "");
            std::printf("integer feas:      %llu  (heuristic hits %llu)\n",
                        static_cast<unsigned long long>(diag.integer_feasible),
                        static_cast<unsigned long long>(diag.heuristic_hits));
            std::printf("  unsnappable:     %llu candidate incumbents refused\n",
                        static_cast<unsigned long long>(diag.incumbents_rejected_unsnappable));
            std::printf("LP repair:         %llu attempts, %llu hits\n",
                        static_cast<unsigned long long>(diag.lp_repair_attempts),
                        static_cast<unsigned long long>(diag.lp_repair_hits));
            std::printf("feasibility pump:  %llu attempts, %llu hits\n",
                        static_cast<unsigned long long>(diag.feasibility_pump_attempts),
                        static_cast<unsigned long long>(diag.feasibility_pump_hits));
            std::printf("integer dive:      %llu attempts, %llu LPs, %llu hits\n",
                        static_cast<unsigned long long>(diag.integer_dive_attempts),
                        static_cast<unsigned long long>(diag.integer_dive_lp_solves),
                        static_cast<unsigned long long>(diag.integer_dive_hits));
            std::printf("RENS:              %llu attempts, %llu LPs, %llu hits\n",
                        static_cast<unsigned long long>(diag.rens_attempts),
                        static_cast<unsigned long long>(diag.rens_lp_solves),
                        static_cast<unsigned long long>(diag.rens_hits));
            std::printf("integer neighborhood: %llu attempts, %llu trials, %llu hits\n",
                        static_cast<unsigned long long>(diag.integer_neighborhood_attempts),
                        static_cast<unsigned long long>(diag.integer_neighborhood_trials),
                        static_cast<unsigned long long>(diag.integer_neighborhood_hits));
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            if (!std::isfinite(diag.dual_bound))
                std::printf("dual bound:        none (no finite global bound)\n");
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", diag.total_ms);
            // Where the wall clock went. Phases overlap (e.g. the cut loop
            // contains root LP re-solves; heuristics run inside the node
            // loop), so these do not sum to total.
            std::printf("  structural pre   %10.3f\n", diag.structural_presolve.ms);
            std::printf("  root setup       %10.3f\n", diag.ms_root_setup);
            std::printf("  mip-presolve     %10.3f\n", diag.mip_presolve_diag.ms);
            std::printf("  before search    %10.3f\n", diag.ms_before_search);
            std::printf("  cut loop         %10.3f\n", diag.cut_loop_ms);
            std::printf("  cut rounds       %10.3f\n", diag.cut_rounds_ms);
            std::printf("  heuristics       %10.3f  (all heuristic work; parts below)\n",
                        sor::search::heuristic_spent_ms(diag));
            std::printf("  heur (in-tree)   %10.3f\n", diag.heuristic_ms);
            std::printf("  feasjump         %10.3f\n", diag.feasjump_ms);
            std::printf("  fixprop          %10.3f\n", diag.fixprop_ms);
            std::printf("  sub-mip          %10.3f\n", diag.sub_mip_ms);
            std::printf("  tree separation  %10.3f\n", diag.tree_sep_ms);
            std::printf("  node loop        %10.3f\n", diag.ms_node_loop);
            std::printf("  node LP          %10.3f\n", diag.lp_ms);
            std::printf("  node setup       %10.3f\n", diag.ms_node_setup);
            std::printf("  node prop        %10.3f  (%llu event nodes with %llu row visits, "
                        "%llu full sweeps)\n", diag.ms_node_prop,
                        (unsigned long long)diag.prop_event_nodes,
                        (unsigned long long)diag.prop_event_visits,
                        (unsigned long long)diag.prop_full_nodes);
            std::printf("  conflict prop    %10.3f\n", diag.ms_conflict_prop);
            std::printf("  nogood learning  %10.3f  (build %.1f, validate %.1f, apply %.1f; "
                        "check binary %.1f general %.1f)\n", diag.ms_nogood_total,
                        diag.ms_nogood_build, diag.ms_nogood_validate, diag.nogood_apply_ms,
                        diag.ms_check_binary, diag.ms_check_general);
            {
                static const char* kSeg[14] = {
                    "loop top->pop", "pop..gap bookkeeping", "bound prune", "node setup",
                    "propagation", "node LP", "LP evidence", "post-LP", "rounding heur",
                    "FJ/LNS/Balans/face", "KP/MRENS/dive/RENS", "branching",
                    "requeue/sep/gcs", "children"};
                std::printf("  node-loop slices (ms):");
                for (int k = 0; k < 14; ++k)
                    std::printf(" [%s %.0f]", kSeg[k], diag.ms_seg[k]);
                std::printf("\n");
            }
            std::printf("  node loop top    %10.3f\n", diag.ms_node_loop_top);
            std::printf("  pre-branch other %10.3f\n", diag.ms_node_pre_branch);
            std::printf("  node pop         %10.3f\n", diag.ms_node_pop);
            std::printf("  node post-LP     %10.3f\n", diag.ms_node_post_lp);
            std::printf("  node children    %10.3f\n", diag.ms_node_children);
            std::printf("  branching        %10.3f  (candidates %.3f, features %.3f "
                        "for %llu vectors)\n", diag.ms_branching,
                        diag.ms_branch_candidates, diag.ms_branch_features,
                        (unsigned long long)diag.branch_feature_vectors);
            std::printf("  strong branch    %10.3f\n", diag.strong_branch_ms);
            std::printf("  conflict         %10.3f\n", diag.conflict_analysis_ms);
            return exit_code_for(r.status);
        }

        if (engine_name == "auto") {
            if (answering_lp_relaxation) {
                std::printf("NOTE:              solving the LP RELAXATION "
                            "(use --engine milp for branch-and-bound)\n");
            }
            sor::core::LpOptions lp_opts;
            lp_opts.strategy = sor::core::LpStrategy::Auto;
            lp_opts.concurrent_solves = lp_concurrent;
            lp_opts.max_iterations = max_iter_given ? sx_opts.max_iterations : 0;
            lp_opts.time_limit_s = sx_opts.time_limit_s;
            lp_opts.primal_feas_tol = sx_opts.primal_feas_tol;
            lp_opts.dual_feas_tol = sx_opts.dual_feas_tol;
            lp_opts.gap_tol = sx_opts.gap_tol;
            lp_opts.presolve = sx_opts.presolve;
            lp_opts.presolve_implied_slack = sx_opts.presolve_implied_slack;
            lp_opts.fo_polish = fo_polish;
            lp_opts.fo_certificates = fo_certificates;
            lp_opts.fo_crossover = fo_crossover;
            lp_opts.auto_budget_split = auto_budget_split;
            lp_opts.backend = backend_name;
            sor::core::LpDiagnostics diag;
            sor::core::ProofEvidence ev;
            // As in the simplex route: the exact dual-bound proof only when
            // asked for (--exact-proof). The SimplexOptions default is on, which
            // the auto and first-order routes inherited silently.
            sx_opts.exact_proof = lp_exact_proof;
            auto raw = sor::engines::solve_lp(problem, lp_opts, diag, &ev, &sx_opts, &hpr_opts, &pdhg_opts);
            const auto r = sor::certify::finalize_result(
                sor::certify::check_lp_candidate(problem, std::move(raw), ev));
            print_result(r);
            write_solution_out(solution_out, r);
            std::printf("route:             %s (%s)\n",
                        std::string(sor::core::to_string(diag.routed_strategy)).c_str(),
                        diag.route_rationale.c_str());
            std::printf("max primal viol:   %.3e\n", r.max_primal_violation);
            std::printf("dual residual:     %.3e\n", r.max_dual_violation);
            std::printf("iterations:        %llu (FO %llu, crossover %llu, simplex %llu)\n",
                        static_cast<unsigned long long>(r.iterations),
                        static_cast<unsigned long long>(diag.fo_iterations),
                        static_cast<unsigned long long>(diag.crossover_iterations),
                        static_cast<unsigned long long>(diag.simplex_iterations));
            std::printf("stage time (ms):    FO %.3f, crossover %.3f, simplex %.3f\n",
                        diag.fo_elapsed_s * 1000.0,
                        diag.crossover_elapsed_s * 1000.0,
                        diag.simplex_elapsed_s * 1000.0);
            std::printf("crossover:         attempted %s, basis valid %s, cold fallback %s\n",
                        diag.crossover_attempted ? "yes" : "no",
                        diag.crossover_basis_valid ? "yes" : "no",
                        diag.crossover_cold_fallback ? "yes" : "no");
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", diag.elapsed_s * 1000.0);
            return exit_code_for(r.status);
        }

        if (engine_name == "simplex" || engine_name == "primal" ||
            engine_name == "dual") {
            if (engine_name == "primal")
                sx_opts.method = sor::engines::SimplexMethod::Primal;
            else if (engine_name == "dual")
                sx_opts.method = sor::engines::SimplexMethod::Dual;
            if (answering_lp_relaxation) {
                std::printf("NOTE:              solving the LP RELAXATION "
                            "(use --engine milp for branch-and-bound)\n");
            }
            // The LP engines stop at tolerance-level optimality unless the
            // exact dual-bound proof is requested.
            sx_opts.exact_proof = lp_exact_proof;
            sor::engines::SimplexDiagnostics diag;
            auto raw = sor::engines::solve_simplex(problem, sx_opts, diag, nullptr);
            auto ev = sor::engines::simplex_evidence(diag, sx_opts);
            const auto r = sor::certify::finalize_result(
                sor::certify::check_lp_candidate(problem, std::move(raw), ev));
            print_result(r);
            write_solution_out(solution_out, r);
            if (diag.dual_bound_finite) {
                std::printf("dual bound:        %.10e\n", r.dual_bound);
                std::printf("rel gap:           %.3e\n", r.gap_rel);
            }
            std::printf("max primal viol:   %.3e\n", r.max_primal_violation);
            std::printf("dual residual:     %.3e\n", r.max_dual_violation);
            std::printf("iterations:        %llu  (phase1 %llu, phase2 %llu)\n",
                        static_cast<unsigned long long>(diag.iterations),
                        static_cast<unsigned long long>(diag.phase1_iterations),
                        static_cast<unsigned long long>(diag.phase2_iterations));
            std::printf("certificate work:  %llu pivots, %llu stages, %llu preparations\n",
                        static_cast<unsigned long long>(diag.certificate_iterations),
                        static_cast<unsigned long long>(diag.certificate_stages),
                        static_cast<unsigned long long>(diag.certificate_preprocessing_builds));
            std::printf("termination:       %s\n", r.termination_reason.c_str());
            std::printf("\ntiming (ms)\n");
            std::printf("  total            %10.3f\n", diag.total_ms);
            std::printf("  scaling          %10.3f\n", diag.scaling_ms);
            std::printf("  CSR->CSC         %10.3f\n", diag.csc_ms);
            std::printf("  preparation      %10.3f\n", diag.preprocessing_ms);
            std::printf("  presolve         %10.3f\n", diag.presolve_ms);
            std::printf("    reductions     %10d rows / %d cols\n",
                        static_cast<int>(diag.presolve_rows_removed),
                        static_cast<int>(diag.presolve_cols_removed));
            std::printf("    singleton cols %10d\n",
                        static_cast<int>(
                            diag.presolve_singleton_columns_removed));
            std::printf("    forcing rows  %10d  (%d columns fixed)\n",
                        static_cast<int>(diag.presolve_forcing_rows_removed),
                        static_cast<int>(diag.presolve_forcing_columns_fixed));
            std::printf("    aggregations  %10d  (%lld positive fill)\n",
                        static_cast<int>(diag.presolve_equality_aggregations),
                        static_cast<long long>(diag.presolve_aggregation_fill));
            std::printf("    retries       %10llu\n",
                        static_cast<unsigned long long>(diag.presolve_retries));
            std::printf("  factorization    %10.3f\n", diag.factor_ms);
            std::printf("  pricing          %10.3f\n", diag.price_ms);
            std::printf("  triangular solve %10.3f\n", diag.solve_ms);
            std::printf("    FTRAN          %10.3f  (%llu calls)\n", diag.ftran_ms,
                        static_cast<unsigned long long>(diag.ftran_calls));
            std::printf("    BTRAN          %10.3f  (%llu calls)\n", diag.btran_ms,
                        static_cast<unsigned long long>(diag.btran_calls));
            std::printf("    pivotal rows   %10.3f\n", diag.pivotal_row_ms);
            std::printf("    ratio tests    %10.3f\n", diag.ratio_test_ms);
            std::printf("  basis updates    %10.3f  (%llu calls)\n", diag.basis_update_ms,
                        static_cast<unsigned long long>(diag.basis_update_calls));
            std::printf("  refactors/cFT    %10llu / %llu  (%llu guarded skips)\n",
                        static_cast<unsigned long long>(diag.refactorizations),
                        static_cast<unsigned long long>(diag.collective_ft_collapses),
                        static_cast<unsigned long long>(diag.collective_ft_skips));
            std::printf("  phase restarts   %10llu  (%llu dual resyncs)\n",
                        static_cast<unsigned long long>(diag.phase_restarts),
                        static_cast<unsigned long long>(diag.dual_resyncs));
            std::printf("  phase1 cost delta %9llu / %llu changed, max %llu\n",
                        static_cast<unsigned long long>(
                            diag.phase1_cost_change_iterations),
                        static_cast<unsigned long long>(diag.phase1_cost_changes),
                        static_cast<unsigned long long>(diag.phase1_cost_change_max));
            std::printf("  primal BTRAN     %10llu / %llu sparse, avg support %llu\n",
                        static_cast<unsigned long long>(diag.primal_btran_sparse),
                        static_cast<unsigned long long>(
                            diag.primal_btran_sparse + diag.primal_btran_dense),
                        static_cast<unsigned long long>(
                            diag.primal_btran_sparse
                                ? diag.primal_btran_support_entries /
                                      diag.primal_btran_sparse
                                : 0));
            std::printf("  phase1 composite %10llu  (%llu sparse / %llu dense / %llu fallback), avg support %llu\n",
                        static_cast<unsigned long long>(
                            diag.phase1_composite_updates),
                        static_cast<unsigned long long>(
                            diag.phase1_composite_sparse),
                        static_cast<unsigned long long>(
                            diag.phase1_composite_dense),
                        static_cast<unsigned long long>(
                            diag.phase1_composite_fallbacks),
                        static_cast<unsigned long long>(
                            diag.phase1_composite_sparse
                                ? diag.phase1_composite_support_entries /
                                      diag.phase1_composite_sparse
                                : 0));
            std::printf("  composite error %11.3g\n",
                        diag.phase1_composite_max_abs_error);
            std::printf("  Devex frameworks %10llu  (%llu exact checks)\n",
                        static_cast<unsigned long long>(diag.devex_frameworks),
                        static_cast<unsigned long long>(diag.devex_weight_checks));
            std::printf("  DSE weight checks %10llu  (%llu rejected rows)\n",
                        static_cast<unsigned long long>(diag.dse_weight_checks),
                        static_cast<unsigned long long>(diag.dse_weight_rejections));
            std::printf("  DSE rebuilds      %10llu  (%llu drift recovery)\n",
                        static_cast<unsigned long long>(diag.dse_weight_rebuilds),
                        static_cast<unsigned long long>(diag.dse_drift_rebuilds));
            std::printf("  DSE->Devex       %10llu  (%llu accuracy, %llu stability, %llu costly DSE iters)\n",
                        static_cast<unsigned long long>(diag.dse_to_devex_switches),
                        static_cast<unsigned long long>(diag.dse_accuracy_switches),
                        static_cast<unsigned long long>(diag.dse_stability_switches),
                        static_cast<unsigned long long>(diag.costly_dse_iterations));
            std::printf("  DSE log error    %10.3g\n", diag.dse_log_weight_error);
            std::printf("  paired FTRAN     %10llu\n",
                        static_cast<unsigned long long>(diag.dual_paired_ftrans));
            std::printf("  pivotal support  %10llu -> %llu entries\n",
                        static_cast<unsigned long long>(
                            diag.dual_pivotal_entries_full),
                        static_cast<unsigned long long>(
                            diag.dual_pivotal_entries_kept));
            std::printf("  dual pricing init D:%llu V:%llu S:%llu\n",
                        static_cast<unsigned long long>(diag.dual_dantzig_starts),
                        static_cast<unsigned long long>(diag.dual_devex_starts),
                        static_cast<unsigned long long>(diag.dual_dse_starts));
            std::printf("  cost perturb     %10llu  (%llu cleanups, %llu stall triggers)\n",
                        static_cast<unsigned long long>(diag.perturbed_costs),
                        static_cast<unsigned long long>(diag.perturbation_cleanups),
                        static_cast<unsigned long long>(diag.stall_perturbations));
            std::printf("  primal bounds    %10llu perturbations, %llu bounds, %llu restorations (%llu pivots)\n",
                        (unsigned long long)diag.primal_bound_perturbations,
                        (unsigned long long)diag.primal_perturbed_bounds,
                        (unsigned long long)diag.primal_bound_restorations,
                        (unsigned long long)diag.primal_bound_restore_iterations);
            std::printf("  cost shifts      %10llu  (%llu wrong-sign entering, max %.3e)\n",
                        static_cast<unsigned long long>(diag.cost_shifts),
                        static_cast<unsigned long long>(
                            diag.wrong_sign_entering_shifts),
                        diag.cost_shift_max);
            std::printf("  primal cleanup   %10llu  (%llu pivots, handed dual "
                        "infeas %.3e / primal infeas %.3e)\n",
                        static_cast<unsigned long long>(diag.primal_cleanups),
                        static_cast<unsigned long long>(
                            diag.primal_cleanup_iterations),
                        diag.cleanup_dual_infeasibility,
                        diag.cleanup_primal_infeasibility);
            std::printf("  trouble refactor %10llu  (%llu shifts refused)\n",
                        static_cast<unsigned long long>(
                            diag.numerical_trouble_refactors),
                        static_cast<unsigned long long>(
                            diag.refused_cost_shifts));
            std::printf("  residual refactors %8llu  (%llu refinement corrections)\n",
                        static_cast<unsigned long long>(diag.residual_refactors),
                        static_cast<unsigned long long>(diag.refinement_corrections));
            std::printf("  zero dual steps  %10llu\n",
                        static_cast<unsigned long long>(diag.numerical_zero_dual_steps));
            std::printf("  rho density      %llu sparse / %llu dense, avg support %llu\n",
                        static_cast<unsigned long long>(diag.rho_sparse_iters),
                        static_cast<unsigned long long>(diag.rho_dense_iters),
                        static_cast<unsigned long long>(
                            diag.rho_sparse_iters
                                ? diag.rho_support_entries / diag.rho_sparse_iters
                                : 0));
            std::printf("  ratio sorted     %10llu candidates\n",
                        static_cast<unsigned long long>(
                            diag.ratio_sorted_candidates));
            std::printf("  dual ratio test  %llu groups, %llu back-offs, %llu exhausted, %llu tiny excluded\n",
                        static_cast<unsigned long long>(diag.ratio_groups),
                        static_cast<unsigned long long>(diag.ratio_backoffs),
                        static_cast<unsigned long long>(diag.ratio_exhausted),
                        static_cast<unsigned long long>(
                            diag.ratio_small_pivot_exclusions));
            std::printf("  simplex loop     %10.3f\n", diag.loop_ms);
            std::printf("  flip batches     %10.3f  (%llu batches)\n", diag.flip_ms,
                        static_cast<unsigned long long>(diag.flip_batches));
            std::printf("  apply_pivot      %10.3f\n", diag.pivot_apply_ms);
            std::printf("  alpha sparse     %llu / %llu iters, avg support %llu\n",
                        static_cast<unsigned long long>(diag.alpha_sparse_iters),
                        static_cast<unsigned long long>(
                            diag.alpha_sparse_iters + diag.alpha_dense_iters),
                        static_cast<unsigned long long>(
                            diag.alpha_sparse_iters
                                ? diag.alpha_support_entries / diag.alpha_sparse_iters
                                : 0));
            std::printf("  primal FTRAN switches %6llu\n",
                        static_cast<unsigned long long>(
                            diag.primal_ftran_dense_switches));
            std::printf("  primal crash     %10llu  (infeas %.6g -> %.6g)\n",
                        static_cast<unsigned long long>(diag.primal_crash_columns),
                        diag.primal_crash_infeasibility_before,
                        diag.primal_crash_infeasibility_after);
            std::printf("  dual crash       %10llu\n",
                        static_cast<unsigned long long>(diag.dual_crash_columns));
            std::printf("  auto stages/builds %8llu / %llu\n",
                        static_cast<unsigned long long>(diag.stages),
                        static_cast<unsigned long long>(diag.preprocessing_builds));
            std::printf("  stage path       P:%llu D:%llu cold:%llu basis-restart:%llu\n",
                        static_cast<unsigned long long>(diag.primal_stages),
                        static_cast<unsigned long long>(diag.dual_stages),
                        static_cast<unsigned long long>(diag.cold_stages),
                        static_cast<unsigned long long>(diag.basis_restarts));
            return exit_code_for(r.status);
        }

        if (engine_name == "hpr" || engine_name == "pdhg" || engine_name == "barrier") {
            // Explicit and ablated FO engines share the same model preparation,
            // original-space recovery, and crossover policy.
            {
                if (answering_lp_relaxation) {
                    std::printf("NOTE:              solving the LP RELAXATION "
                                "(use --engine milp for branch-and-bound)\n");
                }
                sor::core::LpOptions lp_opts;
                lp_opts.strategy = engine_name == "hpr"
                    ? sor::core::LpStrategy::Hpr
                    : engine_name == "barrier" ? sor::core::LpStrategy::Barrier
                    : sor::core::LpStrategy::Pdhg;
                lp_opts.max_iterations = max_iter_given
                    ? (engine_name == "hpr" ? hpr_opts.max_iterations
                                            : pdhg_opts.max_iterations)
                    : 0;
                lp_opts.time_limit_s = engine_name == "hpr"
                    ? hpr_opts.time_limit_s : pdhg_opts.time_limit_s;
                lp_opts.primal_feas_tol = engine_name == "hpr"
                    ? hpr_opts.primal_tol : pdhg_opts.primal_tol;
                lp_opts.dual_feas_tol = engine_name == "hpr"
                    ? hpr_opts.dual_tol : pdhg_opts.dual_tol;
                lp_opts.gap_tol = engine_name == "hpr"
                    ? hpr_opts.gap_tol : pdhg_opts.gap_tol;
                lp_opts.presolve = sx_opts.presolve;
                lp_opts.presolve_implied_slack = sx_opts.presolve_implied_slack;
                lp_opts.fo_polish = fo_polish;
                lp_opts.fo_certificates = fo_certificates;
                lp_opts.fo_crossover = fo_crossover;
                lp_opts.backend = backend_name;
                sor::core::LpDiagnostics diag;
                sor::core::ProofEvidence ev;
                // As in the simplex route: the exact dual-bound proof only when
                // asked for (--exact-proof). The SimplexOptions default is on, which
                // the auto and first-order routes inherited silently.
                sx_opts.exact_proof = lp_exact_proof;
                auto raw = sor::engines::solve_lp(problem, lp_opts, diag, &ev, &sx_opts, &hpr_opts, &pdhg_opts);
                const auto r = sor::certify::finalize_result(
                    sor::certify::check_lp_candidate(problem, std::move(raw), ev));
                print_result(r);
                write_solution_out(solution_out, r);
                std::printf("backend:           %s\n", backend_name.c_str());
                if (!diag.presolve_status.empty())
                    std::printf("presolve:          %s (%s)\n",
                                diag.presolve_status.c_str(),
                                diag.presolve_reason.c_str());
                std::printf("max row violation: %.3e\n", r.max_primal_violation);
                std::printf("dual residual:     %.3e\n", r.max_dual_violation);
                std::printf("iterations:        %llu\n",
                            static_cast<unsigned long long>(r.iterations));
                std::printf("stage iterations:  FO %llu, crossover %llu, simplex %llu\n",
                            static_cast<unsigned long long>(diag.fo_iterations),
                            static_cast<unsigned long long>(diag.crossover_iterations),
                            static_cast<unsigned long long>(diag.simplex_iterations));
                std::printf("stage time (ms):    FO %.3f, crossover %.3f, simplex %.3f\n",
                            diag.fo_elapsed_s * 1000.0,
                            diag.crossover_elapsed_s * 1000.0,
                            diag.simplex_elapsed_s * 1000.0);
                std::printf("crossover:         attempted %s, basis valid %s, cold fallback %s\n",
                            diag.crossover_attempted ? "yes" : "no",
                            diag.crossover_basis_valid ? "yes" : "no",
                            diag.crossover_cold_fallback ? "yes" : "no");
                std::printf("termination:       %s\n",
                            r.termination_reason.c_str());
                std::printf("\ntiming (ms)\n");
                std::printf("  total            %10.3f\n",
                            diag.elapsed_s * 1000.0);
                return exit_code_for(r.status);
            }


        }

        std::fprintf(stderr, "error: unreachable engine dispatch\n");
        return 3;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
