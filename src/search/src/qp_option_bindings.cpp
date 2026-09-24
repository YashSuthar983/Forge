// Option binding tables for the mixed-integer and global QP search.
//
// Same contract as the engine tables (see sor/core/options.hpp): one table
// drives setting an option, listing it with its default, and the documented
// reference, so an option cannot exist in the struct and be missing from the
// help.
#include "sor/search/global_qp.hpp"
#include "sor/search/miqp_bb.hpp"

namespace sor::search {

std::vector<core::OptionBinding> miqp_bb_option_bindings(MiqpBbOptions& o) {
    std::vector<core::OptionBinding> t;
    t.push_back(core::bind_real("int_tol", o.int_tol,
        "How far from a whole number a variable may sit and still count integral."));
    t.push_back(core::bind_real("feas_tol", o.feas_tol,
        "Row and bound violation, original units, for an incumbent to be accepted."));
    t.push_back(core::bind_real("gap_rel", o.gap_rel,
        "Relative gap at which the tree is closed and optimality claimed."));
    t.push_back(core::bind_real("gap_abs", o.gap_abs,
        "Absolute gap at which the tree is closed."));
    t.push_back(core::bind_u64("max_nodes", o.max_nodes, "Node cap."));
    t.push_back(core::bind_real("time_limit_s", o.time_limit_s,
        "Wall-clock budget in seconds; 0 means no limit."));
    t.push_back(core::bind_u64("heuristic_every", o.heuristic_every,
        "Run the rounding heuristic every this many nodes; 0 restricts it to the root."));
    t.push_back(core::bind_flag("verbose", o.verbose, "Per-node logging."));
    return t;
}

bool set_miqp_bb_option(MiqpBbOptions& o, const std::string& kv, std::string& err) {
    auto t = miqp_bb_option_bindings(o);
    return core::apply_option(t, kv, err);
}

std::vector<core::OptionBinding> global_qp_option_bindings(GlobalQpOptions& o) {
    std::vector<core::OptionBinding> t;
    t.push_back(core::bind_double("time_limit_s", o.time_limit_s,
        "Wall-clock budget in seconds."));
    t.push_back(core::bind_u64("max_nodes", o.max_nodes, "Node cap."));
    t.push_back(core::bind_real("gap_tol", o.gap_tol,
        "Relative gap on max(1,|incumbent|) at which the tree is closed and "
        "global optimality claimed. Widening this weakens the claim -- it does "
        "not make the solver stronger."));
    t.push_back(core::bind_real("feas_tol", o.feas_tol,
        "Violation an incumbent must meet, original units. Note the local "
        "engine's own default is looser; a point good enough for it can be "
        "rejected here."));
    t.push_back(core::bind_real("int_tol", o.int_tol,
        "Integrality tolerance for branching on discrete columns."));
    t.push_back(core::bind_size("mccormick_max_terms", o.mccormick_max_terms,
        "Largest number of product columns for which the LP relaxation is chosen."));
    t.push_back(core::bind_flag("rlt", o.rlt,
        "Attempt RLT rows. Measured at the root and kept only if the bound "
        "improvement earns the LP it costs."));
    t.push_back(core::bind_size("rlt_max_new_terms", o.rlt_max_new_terms,
        "Cap on product columns RLT may introduce."));
    t.push_back(core::bind_flag("all_envelopes", o.all_envelopes,
        "All four McCormick inequalities per product, not only the side the "
        "objective sign needs; the extra side is what lets RLT rows bite."));
    t.push_back(core::bind_flag("equality_penalty", o.equality_penalty,
        "Add a penalty that vanishes on the equality rows before choosing the shift."));
    t.push_back(core::bind_flag("psd_cuts", o.psd_cuts,
        "Attempt PSD cuts, kept only if they move the root bound."));
    t.push_back(core::bind_int("psd_max_dim", o.psd_max_dim,
        "Largest quadratic dimension for which PSD cuts are separated."));
    t.push_back(core::bind_int("psd_rounds_root", o.psd_rounds_root,
        "Maximum PSD separation rounds at the root."));
    t.push_back(core::bind_int("psd_rounds_node", o.psd_rounds_node,
        "PSD separation rounds at an in-tree node."));
    t.push_back(core::bind_int("psd_node_depth_max", o.psd_node_depth_max,
        "Deepest node at which PSD cuts are separated."));
    t.push_back(core::bind_int("psd_cuts_per_round", o.psd_cuts_per_round,
        "Cuts generated per round."));
    t.push_back(core::bind_size("psd_max_cuts", o.psd_max_cuts,
        "Total PSD cut budget; a safety net, not a target."));
    t.push_back(core::bind_real("psd_cut_tol", o.psd_cut_tol,
        "How negative an eigenvalue must be before its eigenvector becomes a cut."));
    t.push_back(core::bind_flag("fbbt", o.fbbt,
        "Feasibility-based bound tightening by interval propagation."));
    t.push_back(core::bind_int("obbt_max_vars", o.obbt_max_vars,
        "Largest product-variable count for which OBBT runs (two LPs per variable)."));
    t.push_back(core::bind_int("obbt_node_depth_max", o.obbt_node_depth_max,
        "Deepest node at which OBBT is re-run; 0 keeps it at the root only."));
    t.push_back(core::bind_int("obbt_every", o.obbt_every,
        "Re-run OBBT every this many qualifying nodes."));
    t.push_back(core::bind_flag("face_polish", o.face_polish,
        "Exact stationarity solve on the free variables of a local point."));
    t.push_back(core::bind_int("polish_max_free", o.polish_max_free,
        "Free-variable cap for that polish; it is a dense cubic solve."));
    t.push_back(core::bind_flag("local_search", o.local_search,
        "Run local search from relaxation points, kept only if it improves the incumbent."));
    t.push_back(core::bind_int("local_every", o.local_every,
        "Run local search every this many nodes."));
    t.push_back(core::bind_int("local_max_iterations", o.local_max_iterations,
        "Iteration cap per local-search call."));
    t.push_back(core::bind_double("local_time_s", o.local_time_s,
        "Seconds per call to the full local solver on a quadratic-constraint instance."));
    t.push_back(core::bind_double("node_lp_time_s", o.node_lp_time_s,
        "Seconds allowed for one node relaxation."));
    t.push_back(core::bind_flag("verbose", o.verbose, "Per-node logging."));
    return t;
}

}  // namespace sor::search
