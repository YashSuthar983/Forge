// Finite-boundary AND/OR decomposition. A component message owns one exact
// boundary context and one disjoint objective piece. A point, a lower bound,
// and an exhausted optimum are different facts; only the first two enter the
// parent's upper/lower ledgers. Sparse models and row ownership compile once.
MilpSearchTask solve_milp_conditioned_task(const model::LpProblem &problem,
                            const BabOptions &opts, BabDiagnostics &diag,
                            bool& handled, SearchContinuationControl* control) {
  if (std::isfinite(opts.initial_cutoff))
    co_return core::RawResult{};
  const auto start = Clock::now();
  auto wall_start = WallClock::now();
  const double analysis_budget =
      opts.time_limit_s > 0
          ? std::min(opts.time_limit_s,
                     opts.root_reduction_share * opts.time_limit_s)
          : 0;
  // Zero share disables analysis; zero passed to an oracle means unlimited.
  if (opts.time_limit_s > 0 && !(analysis_budget > 0))
    co_return core::RawResult{};
  auto compiled =
      compile_conditional_components(problem, analysis_budget, opts.cancel);
  if (!compiled)
    co_return core::RawResult{};
  handled = true;
  core::RawResult out;
  auto &plan = *compiled;
  const f64 sense = problem.maximize ? -1.0 : 1.0;
  const auto &rp = problem.A.pattern.row_ptr();
  const auto &ci = problem.A.pattern.col_idx();
  f64 bound_min = core::kPosInf, best_min = core::kPosInf;
  bool all_infeasible = true;
  std::vector<f64> best;
  BabDiagnostics total;
  struct Message {
    bool infeasible = false, completed = false;
    f64 bound = -core::kPosInf;
    std::vector<f64> point;
    std::unique_ptr<ContinuedSearch> search;
  };
  std::vector<std::map<std::vector<f64>, Message>> messages(
      plan.components.size());
  std::vector<std::vector<f64>> prior_points(plan.components.size());
  // Master regions cover boundary boxes, not sampled assignments. Every
  // unresolved leaf remains in a separate lower-bound ledger even after its
  // component calls end. A message can only strengthen a region whose whole
  // component boundary matches the message's exact context.
  struct Region {
    std::vector<f64> lo, hi;
    f64 bound = -core::kPosInf;
    std::uint64_t serial = 0;
    bool entered = false;
    bool priced = false;
  };
  const auto compare = [](const Region &a, const Region &b) {
    return a.bound != b.bound ? a.bound > b.bound : a.serial < b.serial;
  };
  std::priority_queue<Region, std::vector<Region>, decltype(compare)> open(
      compare);
  std::vector<f64> block_box_bounds;
  for (const auto &block : plan.components) {
    const auto zero = certify::safe_lagrangian_lower_bound(
        block.model, std::vector<f64>(sz(block.model.n_rows()), 0),
        block.model.col_lo, block.model.col_hi);
    block_box_bounds.push_back(zero.finite ? zero.value : -core::kPosInf);
  }
  const auto seconds_left = [&] {
    return opts.time_limit_s > 0
               ? std::max(0.0, opts.time_limit_s - std::max(ms_since(start) / 1000,
                   std::chrono::duration<double>(WallClock::now() - wall_start).count() /
                       std::max(1.0, opts.wall_limit_factor)))
               : core::kPosInf;
  };
  auto master = compile_conditional_master(problem, plan.separator);
  RowActivityCache master_activity;
  master_activity.build(master, master.col_lo, master.col_hi);
  PropagationScratch master_scratch;
  // The decomposed LP master in original coordinates keeps the free block
  // variables explicit. This is the exact extended relaxation, rather than
  // rounded component value samples or an interpolated integer value hull.
  // Its duals are projected below onto only the linking boundary coordinates.
  model::LpProblem master_relaxation = problem;
  auto master_lp_options = tree_lp_options(opts.lp);
  master_lp_options.presolve = false;
  std::unique_ptr<engines::DualProbeSession> master_session;
  engines::SimplexBasis master_basis;
  DualCutoffStore master_witnesses;
  bool master_relaxation_bounded = false;
  const std::vector<char> boundary_integer(plan.separator.size(), true);
  const auto price_region = [&](Region& region) {
    const double left = seconds_left();
    if (!(left > 0) || core::cancel_requested(opts.cancel)) return true;
    const double slice_left = control
        ? control->seconds_left()
        : core::kPosInf;
    if (!(slice_left > 0)) return true;
    for (std::size_t q = 0; q < plan.separator.size(); ++q) {
      const Index j = plan.separator[q];
      master_relaxation.col_lo[sz(j)] = region.lo[q];
      master_relaxation.col_hi[sz(j)] = region.hi[q];
    }
    // Master and the AND node's unfinished components share the remaining
    // work. Every master solve continues its own basis/factor state.
    master_lp_options.time_limit_s = std::isfinite(left)
        ? left / static_cast<double>(1 + plan.components.size()) : 0;
    if (std::isfinite(slice_left)) master_lp_options.time_limit_s =
        master_lp_options.time_limit_s > 0 ? std::min(master_lp_options.time_limit_s, slice_left) : slice_left;
    if (!master_session)
      master_session = std::make_unique<engines::DualProbeSession>(master_relaxation, master_lp_options);
    master_session->set_column_bounds(master_relaxation.col_lo, master_relaxation.col_hi);
    engines::SimplexDiagnostics sd;
    const auto warm = master_basis;
    const auto result = master_session->solve(master_lp_options, sd, &master_basis,
        warm.basic.empty() ? nullptr : &warm);
    region.priced = true;
    ++total.lp_solves; total.lp_iterations += sd.iterations; total.lp_ms += sd.total_ms;
    if (node_lp_infeasibility_proved(master_relaxation, result, opts.primal_feas_tol, false)) return false;
    if (result.y.size() == sz(problem.n_rows())) {
      auto y_min = result.y;
      for (auto& y : y_min) y *= sense;
      const auto local_bound = certify::safe_lagrangian_lower_bound(problem, y_min,
          master_relaxation.col_lo, master_relaxation.col_hi);
      if (local_bound.finite) region.bound = std::max(region.bound, local_bound.value);
      const auto enclosure = certify::enclose_reduced_costs(problem, y_min, problem.col_lo, problem.col_hi);
      if (enclosure.finite) {
        master_relaxation_bounded = true;
        certify::RcEnclosure projected;
        projected.bound = enclosure.bound; projected.finite = true;
        for (const Index j : plan.separator) {
          projected.lo.push_back(enclosure.lo[sz(j)]);
          projected.hi.push_back(enclosure.hi[sz(j)]);
        }
        // Dropped free-column move terms are nonnegative over the original
        // box. Boundary anchors must also be the original real bounds, not
        // integer-rounded bounds that the constant did not charge.
        std::vector<f64> anchor_lo, anchor_hi;
        for (const Index j : plan.separator) {
          anchor_lo.push_back(problem.col_lo[sz(j)]); anchor_hi.push_back(problem.col_hi[sz(j)]);
        }
        master_witnesses.consider(projected, anchor_lo, anchor_hi, 0, true);
      }
    }
    return true;
  };
  std::uint64_t serial = 0;
  const auto bound_region = [&](Region &region) {
    const auto propagation = propagate_bounds_cached(master, master_activity,
        region.lo, region.hi, nullptr, nullptr, master_scratch, nullptr, 0,
        opts.primal_feas_tol, std::numeric_limits<std::uint64_t>::max(), true, opts.cancel);
    total.propagation_tightenings += propagation.tightened;
    if (!propagation.feasible) return false;
    model::DyadicSum bound;
    bound.add(sense * problem.obj_offset);
    for (Index j : plan.independent_cost_columns) {
      const auto pos =
          std::find(plan.separator.begin(), plan.separator.end(), j);
      const f64 c = sense * problem.c[sz(j)];
      f64 lo = problem.col_lo[sz(j)], hi = problem.col_hi[sz(j)];
      if (pos != plan.separator.end()) {
        const auto q = static_cast<std::size_t>(pos - plan.separator.begin());
        lo = region.lo[q];
        hi = region.hi[q];
      } else if (!problem.is_integer.empty() && problem.is_integer[sz(j)]) {
        lo = std::ceil(lo);
        hi = std::floor(hi);
      }
      if (lo > hi)
        return false;
      if (c == 0)
        continue;
      const f64 corner = c > 0 ? lo : hi;
      if (!std::isfinite(corner)) {
        region.bound = std::max(region.bound, master_witnesses.lower_bound(region.lo, region.hi, 0));
        return true;
      }
      bound.add_product(c, corner);
    }
    for (std::size_t b = 0; b < plan.components.size(); ++b) {
      std::vector<f64> key;
      bool fixed = true;
      for (const auto &[q, j] : plan.components[b].boundary) {
        (void)j;
        fixed = fixed && region.lo[q] == region.hi[q];
        key.push_back(region.lo[q]);
      }
      f64 value = block_box_bounds[b];
      if (fixed) {
        const auto it = messages[b].find(key);
        if (it != messages[b].end()) {
          if (it->second.infeasible)
            return false;
          value = std::max(value, it->second.bound);
        }
      }
      if (!std::isfinite(value)) {
        region.bound = std::max(region.bound, master_witnesses.lower_bound(region.lo, region.hi, 0));
        return true;
      }
      bound.add(value);
    }
    region.bound = std::max({region.bound, bound.down(),
        master_witnesses.lower_bound(region.lo, region.hi, 0)});
    return true;
  };
  Region root;
  for (Index j : plan.separator) {
    root.lo.push_back(std::ceil(problem.col_lo[sz(j)]));
    root.hi.push_back(std::floor(problem.col_hi[sz(j)]));
  }
  if (bound_region(root))
    open.push(std::move(root));
  std::vector<f64> assignment;
  f64 best_upper_min = core::kPosInf;
  // The master accepts an incumbent only in its own original column space.
  // It is also a useful bound before any component context has been priced.
  if (opts.initial_solution && seconds_left() > 0 &&
      !core::cancel_requested(opts.cancel)) {
    auto proposal = *opts.initial_solution;
    const double repair_allowance =
        std::min(kCanonicalRepairTime, seconds_left());
    if (canonicalize_milp_point(problem, proposal, opts.int_tol,
                                opts.primal_feas_tol, repair_allowance)) {
      f64 value = problem.objective(proposal);
      if (opts.point_validator)
        value = (*opts.point_validator)(proposal);
      if (std::isfinite(value)) {
        best_min = sense * value;
        model::DyadicSum exact_objective;
        exact_objective.add(sense * problem.obj_offset);
        for (Index j = 0; j < problem.n_cols(); ++j)
          exact_objective.add_product(sense * problem.c[sz(j)],
                                      proposal[sz(j)]);
        best_upper_min = exact_objective.up();
        best = std::move(proposal);
      }
    }
  }
  while (!open.empty()) {
    if (control && control->due(total.nodes) && !core::cancel_requested(opts.cancel)) {
      control->snapshot = core::RawResult{};
      auto& snapshot = control->snapshot;
      snapshot.engine = "milp_bab"; snapshot.backend = "cpu";
      snapshot.x = best;
      snapshot.objective = best.empty() ? core::kNaN : sense * best_min;
      snapshot.dual_bound = sense * std::min({bound_min, open.top().bound, best_upper_min});
      snapshot.proposed_status = best.empty() ? core::Status::Interrupted : core::Status::Feasible;
      snapshot.proposed_level = best.empty() ? core::ProofLevel::None : core::ProofLevel::FeasibleWithGap;
      snapshot.termination_reason = "conditional master suspended with retained child frontiers";
      diag = total;
      for (auto& component : messages) for (auto& [context, message] : component) {
        (void)context;
        if (message.search) {
          diag.nodes -= message.search->accounted_nodes;
          accumulate_bab_work(diag, message.search->diagnostics);
        }
      }
      diag.component_count = plan.components.size();
      diag.root_relaxation_bounded = master_relaxation_bounded;
      diag.unbounded_certified = false;
      diag.incumbent = snapshot.objective; diag.dual_bound = snapshot.dual_bound;
      diag.total_ms = ms_since(start); diag.globally_proved = false;
      co_await SearchBoundary{control, wall_start};
    }
    if (!(seconds_left() > 0) || total.nodes >= opts.max_nodes ||
        core::cancel_requested(opts.cancel)) {
      bound_min = std::min(bound_min, open.top().bound);
      all_infeasible = false;
      break;
    }
    Region region = open.top();
    open.pop();
    // A scheduling suspension does not create a new master node. Charging
    // this again starved every child under one-node continuation slices.
    if (!region.entered) {
      ++total.nodes;
      core::work_add(plan.separator.size() + plan.components.size(),
                     core::WorkTag::BabNode);
      region.entered = true;
    }
    if (!bound_region(region))
      continue;
    if (!region.priced) {
      if (!price_region(region)) continue;
    }
    if (std::isfinite(best_upper_min) && !master_witnesses.propagate(
          region.lo, region.hi, best_upper_min, boundary_integer, 0, nullptr, 0)) {
      bound_min = std::min(bound_min, best_upper_min);
      all_infeasible = false;
      continue;
    }
    if (region.bound >= best_upper_min) {
      bound_min = std::min(bound_min, region.bound);
      all_infeasible = false;
      continue;
    }
    std::size_t branch = plan.separator.size();
    for (std::size_t q = 0; q < plan.separator.size(); ++q)
      if (region.lo[q] < region.hi[q]) {
        branch = q;
        break;
      }
    if (branch != plan.separator.size()) {
      // Balanced integer-domain partition; both sides survive as obligations.
      const f64 middle = std::floor(
          region.lo[branch] + (region.hi[branch] - region.lo[branch]) / 2);
      Region lower = region, upper = std::move(region);
      lower.hi[branch] = middle;
      upper.lo[branch] = middle + 1;
      lower.serial = ++serial;
      upper.serial = ++serial;
      lower.entered = upper.entered = false;
      lower.priced = upper.priced = false;
      if (bound_region(lower))
        open.push(std::move(lower));
      if (bound_region(upper))
        open.push(std::move(upper));
      continue;
    }
    assignment = std::move(region.lo);
    std::vector<f64> point = problem.col_lo;
    for (std::size_t q = 0; q < plan.separator.size(); ++q)
      point[sz(plan.separator[q])] = assignment[q];
    bool assignment_infeasible = false, all_points = true, bound_finite = true, pending = false;
    model::DyadicSum slice_bound;
    slice_bound.add(sense * problem.obj_offset);
    for (Index j : plan.independent_cost_columns) {
      if (std::find(plan.separator.begin(), plan.separator.end(), j) ==
          plan.separator.end()) {
        f64 lo = problem.col_lo[sz(j)], hi = problem.col_hi[sz(j)];
        if (!problem.is_integer.empty() && problem.is_integer[sz(j)]) {
          lo = std::ceil(lo);
          hi = std::floor(hi);
        }
        if (lo > hi) {
          assignment_infeasible = true;
          break;
        }
        const f64 c = sense * problem.c[sz(j)];
        point[sz(j)] = c > 0   ? lo
                       : c < 0 ? hi
                               : std::min(std::max(0.0, lo), hi);
        // An infinite improving corner is a lower-bound fact, never a
        // feasible point. Retain a finite witness for the parent's recession
        // certificate while keeping this region's lower bound unbounded.
        if (!std::isfinite(point[sz(j)])) {
          bound_finite = false;
          point[sz(j)] = std::min(std::max(0.0, lo), hi);
        }
      }
      if (!std::isfinite(point[sz(j)])) {
        all_points = false;
        bound_finite = false;
      } else if (bound_finite)
        slice_bound.add_product(sense * problem.c[sz(j)], point[sz(j)]);
    }
    for (Index i : plan.constant_rows) {
      if (assignment_infeasible)
        break;
      model::DyadicSum activity;
      for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
        if (problem.A.vals[sz(k)] != 0)
          activity.add_product(problem.A.vals[sz(k)], point[sz(ci[sz(k)])]);
      auto lower = activity, upper = activity;
      lower.add(-opts.primal_feas_tol);
      upper.add(opts.primal_feas_tol);
      if ((std::isfinite(problem.row_lo[sz(i)]) &&
           upper.compare(problem.row_lo[sz(i)]) < 0) ||
          (std::isfinite(problem.row_hi[sz(i)]) &&
           lower.compare(problem.row_hi[sz(i)]) > 0))
        assignment_infeasible = true;
    }
    for (std::size_t b = 0;
         b < plan.components.size() && !assignment_infeasible; ++b) {
      auto &block = plan.components[b];
      std::vector<f64> key;
      for (const auto &[q, j] : block.boundary) {
        (void)j;
        key.push_back(assignment[q]);
      }
      auto it = messages[b].find(key);
      if (it == messages[b].end()) {
        for (const auto &[q, j] : block.boundary)
          block.model.col_lo[sz(j)] = block.model.col_hi[sz(j)] = assignment[q];
        Message message;
        const auto zero = certify::safe_lagrangian_lower_bound(
            block.model, std::vector<f64>(sz(block.model.n_rows()), 0),
            block.model.col_lo, block.model.col_hi);
        if (zero.finite) message.bound = zero.value;
        BabOptions child = opts;
        child.component_child = true;
        // Parent presolve already compiled these rows. The child owns this
        // exact column space and the ordinary solver's retained tree.
        child.component_solve = false;
        child.structural_presolve.enabled = false;
        child.sub_mip_depth = opts.sub_mip_depth + 1;
        child.point_validator = nullptr; child.pool = nullptr;
        child.root_restart_allowed = false; child.tree_restart = false;
        child.restart_carry_state = false;
        child.initial_root_basis = nullptr; child.initial_solution = nullptr;
        child.initial_clauses = nullptr; child.probing_carry = nullptr;
        child.pseudocost_seed = nullptr; child.cut_reference_point = nullptr;
        child.gap_tol = opts.gap_tol / static_cast<f64>(plan.components.size());
        child.abs_gap_tol = opts.abs_gap_tol / static_cast<f64>(plan.components.size());
        auto warm_point = prior_points[b];
        if (warm_point.size() == block.columns.size()) {
          for (const auto &[q, j] : block.boundary) warm_point[sz(j)] = assignment[q];
          if (milp_point_max_violation(block.model, warm_point, opts.int_tol) > opts.primal_feas_tol)
            warm_point.clear();
        } else warm_point.clear();
        message.search = std::make_unique<ContinuedSearch>(block.model, child, std::move(warm_point));
        it = messages[b].emplace(std::move(key), std::move(message)).first;
        ++total.conditional_contexts;
      } else ++total.conditional_cache_hits;
      auto& message_live = it->second;
      if (!message_live.completed && !message_live.infeasible &&
          !message_live.search->task.done() && seconds_left() > 0 &&
          total.nodes < opts.max_nodes && (!control || !control->due(total.nodes)) &&
          !core::cancel_requested(opts.cancel)) {
        auto& search = *message_live.search;
        const auto remaining_nodes = opts.max_nodes - total.nodes;
        search.options.max_nodes = remaining_nodes > std::numeric_limits<std::uint64_t>::max() - search.diagnostics.nodes
            ? std::numeric_limits<std::uint64_t>::max() : search.diagnostics.nodes + remaining_nodes;
        const double left = seconds_left();
        // This slice is a scheduling allowance, not a new proof problem.
        // The underlying hard allowance covers all active slices; suspended
        // time and other messages' work do not age the child's own clocks.
        search.options.time_limit_s = std::isfinite(left)
            ? search.diagnostics.total_ms / 1000 + left : 0;
        double allowance = std::isfinite(left)
            ? left / static_cast<double>(plan.components.size() - b) : core::kPosInf;
        if (control) allowance = std::min(allowance,
            control->seconds_left());
        const auto slice_nodes = control
            ? std::min(remaining_nodes, control->node_stop > total.nodes ? control->node_stop - total.nodes : 0)
            : remaining_nodes;
        auto result = search.advance(allowance, slice_nodes);
        total.nodes += search.diagnostics.nodes - search.accounted_nodes;
        search.accounted_nodes = search.diagnostics.nodes;
        const auto& sd = search.diagnostics;
        const bool terminal = search.task.done();
        message_live.infeasible = terminal && result.proposed_status == core::Status::Infeasible && sd.globally_proved;
        message_live.completed = terminal && result.proposed_status == core::Status::Optimal && sd.globally_proved;
        if (message_live.completed) ++total.components_solved;
        if (std::isfinite(result.dual_bound))
          message_live.bound = std::max(message_live.bound, sense * result.dual_bound);
        if (result.x.size() == block.columns.size() &&
            milp_point_max_violation(search.problem, result.x, opts.int_tol) <= opts.primal_feas_tol) {
          message_live.point = result.x; prior_points[b] = result.x;
        }
      }
      pending = pending || (!message_live.search->task.done() && !message_live.completed && !message_live.infeasible);
      const auto &message = it->second;
      if (message.infeasible) {
        assignment_infeasible = true;
        break;
      }
      if (std::isfinite(message.bound))
        slice_bound.add(message.bound);
      else
        bound_finite = false;
      if (message.point.empty())
        all_points = false;
      else
        for (std::size_t q = 0; q < block.owned; ++q)
          point[sz(block.columns[q])] = message.point[q];
    }
    if (!assignment_infeasible) {
      all_infeasible = false;
      const f64 slice_lower = std::max(region.bound, bound_finite ? slice_bound.down() : -core::kPosInf);
      if (!pending) bound_min = std::min(bound_min, slice_lower);
      if (all_points &&
          milp_point_max_violation(problem, point, opts.int_tol) <=
              opts.primal_feas_tol) {
        f64 objective = problem.objective(point);
        if (opts.point_validator)
          objective = (*opts.point_validator)(point);
        if (std::isfinite(objective) && sense * objective < best_min) {
          best_min = sense * objective;
          model::DyadicSum exact_objective;
          exact_objective.add(sense * problem.obj_offset);
          for (Index j = 0; j < problem.n_cols(); ++j)
            exact_objective.add_product(sense * problem.c[sz(j)], point[sz(j)]);
          best_upper_min = exact_objective.up();
          best = std::move(point);
        }
      }
      if (pending) {
        Region unfinished = std::move(region);
        unfinished.lo = unfinished.hi = assignment;
        unfinished.bound = slice_lower;
        unfinished.serial = ++serial;
        open.push(std::move(unfinished));
      }
    }
  }
  // Cumulative child diagnostics are merged exactly once. Node work was
  // already charged online to enforce the parent's actual shared allowance.
  for (auto& component : messages) for (auto& [context, message] : component) {
    (void)context;
    if (message.search) {
      total.nodes -= message.search->accounted_nodes;
      accumulate_bab_work(total, message.search->diagnostics);
    }
  }
  diag = std::move(total);
  // Child proof flags describe child models, not their Cartesian parent.
  // One bounded child does not rule out an improving ray in another piece.
  diag.root_relaxation_bounded = master_relaxation_bounded;
  diag.unbounded_certified = false;
  diag.policy_used = opts.policy;
  diag.component_count += plan.components.size();
  diag.total_ms = diag.component_solve_ms = ms_since(start);
  out = core::RawResult{};
  out.engine = "milp_bab";
  out.backend = "cpu";
  if (all_infeasible) {
    diag.globally_proved = true;
    out.proposed_status = core::Status::Infeasible;
    out.proposed_level = core::ProofLevel::BoundOnly;
    out.termination_reason = "every finite boundary assignment is infeasible";
  } else {
    out.x = std::move(best);
    out.objective = out.x.empty() ? core::kNaN : sense * best_min;
    out.dual_bound = diag.dual_bound = sense * bound_min;
    diag.incumbent = out.objective;
    const f64 gap = best_upper_min - bound_min;
    diag.gap_rel = std::isfinite(gap)
                       ? std::max(0.0, gap) / (1 + std::fabs(best_min))
                       : core::kPosInf;
    // Even an interrupted message can close the outer gap through its
    // independently valid bound. An observed feasible value cannot.
    const bool proved =
        !out.x.empty() && std::isfinite(bound_min) &&
        gap >= -opts.primal_feas_tol &&
        (gap <= opts.abs_gap_tol || diag.gap_rel <= opts.gap_tol);
    diag.globally_proved = proved;
    out.proposed_status = proved           ? core::Status::Optimal
                          : !out.x.empty() ? core::Status::Feasible
                                           : core::Status::Interrupted;
    out.proposed_level = proved           ? milp_global_proof(diag)
                         : !out.x.empty() ? core::ProofLevel::FeasibleWithGap
                                          : core::ProofLevel::None;
    out.termination_reason =
        proved
            ? "conditional component messages close the global gap"
            : "conditional decomposition retains unresolved boundary contexts";
  }
  diag.termination_reason = out.termination_reason;
  co_return out;
}
