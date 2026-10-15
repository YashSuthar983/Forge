#include "sor/certify/finalize.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/basis_map.hpp"
#include "sor/search/block_oracle.hpp"
#include "sor/search/certificate_rows.hpp"
#include "sor/search/conditional_cuts.hpp"
#include "sor/search/conditional_components.hpp"
#include "sor/search/domain_checkpoint.hpp"
#include "sor/search/domain_writer.hpp"
#include "sor/search/integer_oracle.hpp"
#include "sor/search/conflict_store.hpp"
#include "sor/search/matching.hpp"
#include "sor/search/milp_presolve.hpp"
#include "sor/search/objective_core.hpp"
#include "sor/search/recourse.hpp"
#include "sor/search/residue.hpp"
#include "sor/search/split.hpp"
#include "sor/search/structure.hpp"
#include "sor/search/surrogate.hpp"
#include "sor/search/tree_cuts.hpp"
#include "sor/sparse/csr.hpp"
#include "test_helpers.hpp"
#include <cmath>
#include <random>

using namespace sor::search;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::sparse::from_triplets;

namespace {
LpProblem binary_model(int n) {
  LpProblem lp;
  lp.A = from_triplets(0, n, {}, {}, {});
  lp.c.assign(n, 1);
  lp.col_lo.assign(n, 0);
  lp.col_hi.assign(n, 1);
  lp.is_integer.assign(n, true);
  return lp;
}
void admits(const CutRow &row, const std::vector<double> &x) {
  sor::model::DyadicSum activity;
  for (std::size_t k = 0; k < row.cols.size(); ++k)
    activity.add_product(row.vals[k], x[row.cols[k]]);
  if (std::isfinite(row.row_lo))
    CHECK(activity.compare(row.row_lo) >= 0);
  if (std::isfinite(row.row_hi))
    CHECK(activity.compare(row.row_hi) <= 0);
}
void test_exact_delivery_and_probe_geometry() {
  std::vector<sor::model::DyadicSum> g(2);
  g[0].add(1);
  g[0].add(0x1p-54);
  g[1].add(-1);
  sor::model::DyadicSum rhs;
  rhs.add(-1);
  CutRow row;
  CHECK(compile_geq_exact(g, rhs, {-2, 0}, {3, 1}, row));
  for (double x : {-2., -1., 0., 3.})
    for (double y : {0., 1.}) {
      sor::model::DyadicSum a;
      a.add_sum_product(g[0], x);
      a.add(-y);
      if (a.compare(-1) >= 0)
        admits(row, {x, y});
    }
  auto lp = binary_model(2);
  lp.c = {2, 3};
  lp.obj_offset = 100;
  CHECK(objective_probe_cut(lp, 0, 103, 102, row));
  auto continuous = lp;
  continuous.is_integer[0] = false;
  CHECK(!objective_probe_cut(continuous, 0, 103, 102, row));
  admits(row, {0, 1});
  admits(row, {1, 0});
  admits(row, {1, 1});
  CutOptions options;
  row.exact = false;
  row.compiled = false;
  CHECK(compile_cut(row, lp, options));
  const auto side = row.row_lo;
  CHECK(compile_cut(row, lp, options));
  CHECK(row.row_lo == side);
}
void test_modular_domain_lifetime() {
  auto lp = binary_model(3);
  lp.col_hi = {20, 1, 1};
  lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1, 3, 3});
  lp.row_lo = lp.row_hi = {6};
  ResidueSystem system;
  system.build(lp, {1, 1, 1}, 100000);
  ResidueState parent;
  auto lo = lp.col_lo, hi = lp.col_hi;
  std::uint64_t changes = 0;
  CHECK(system.propagate(lo, hi, changes, nullptr, 0, nullptr, &parent));
  CHECK(hi[0] == 6);
  auto child = parent;
  lo[0] = 1;
  hi[0] = 5;
  CHECK(system.propagate(lo, hi, changes, nullptr, 1, nullptr, &child));
  CHECK(lo[0] == 3 && hi[0] == 3);
  lo = lp.col_lo;
  hi = lp.col_hi;
  CHECK(system.propagate(lo, hi, changes, nullptr, 0, nullptr, &child));
  CHECK(lo[0] == 0 && hi[0] == 6);
  lo[0] = 1;
  hi[0] = 2;
  CHECK(!system.propagate(lo, hi, changes));
}
void test_matching_soundness_and_pending() {
  // Two jobs, two values: every column appears in an exact job row and
  // a value-capacity row. Exhaustively check every node box against the
  // original assignment constraints, independent of matching internals.
  auto lp = binary_model(4);
  lp.A = from_triplets(4, 4, {0, 0, 1, 1, 2, 2, 3, 3}, {0, 1, 2, 3, 0, 2, 1, 3},
                       {1, 1, 1, 1, 1, 1, 1, 1});
  lp.row_lo = {1, 1, -kInf, -kInf};
  lp.row_hi = {1, 1, 1, 1};
  MatchingService service;
  service.build(lp, {1, 1, 1, 1});
  auto lo = lp.col_lo, hi = lp.col_hi;
  std::uint64_t changes = 0;
  bool converged = true;
  CHECK(service.propagate(lo, hi, changes, 0, nullptr, 0, &converged));
  CHECK(!converged);
  for (int box = 0; box < 81; ++box) {
    int code = box;
    lo.assign(4, 0);
    hi.assign(4, 1);
    for (int j = 0; j < 4; ++j) {
      int state = code % 3;
      code /= 3;
      if (state == 1)
        hi[j] = 0;
      if (state == 2)
        lo[j] = 1;
    }
    const auto old_lo = lo, old_hi = hi;
    changes = 0;
    converged = false;
    const bool feasible =
        service.propagate(lo, hi, changes, 10000, nullptr, 0, &converged);
    CHECK(converged);
    bool witness = false;
    for (int bits = 0; bits < 16; ++bits) {
      std::vector<double> x(4);
      for (int j = 0; j < 4; ++j)
        x[j] = (bits >> j) & 1;
      bool inside = true;
      for (int j = 0; j < 4; ++j)
        inside = inside && x[j] >= old_lo[j] && x[j] <= old_hi[j];
      if (!inside || lp.max_row_violation(x) > 0)
        continue;
      witness = true;
      CHECK(feasible);
      for (int j = 0; j < 4; ++j)
        CHECK(x[j] >= lo[j] && x[j] <= hi[j]);
    }
    CHECK(feasible == witness);
  }
}
void test_overlapping_objective_cores() {
  std::mt19937 rng(912);
  for (int trial = 0; trial < 50; ++trial) {
    auto lp = binary_model(5);
    lp.obj_offset = 17;
    for (auto &c : lp.c)
      c = 1 + (rng() % 7);
    std::vector<std::vector<ConflictLiteral>> clauses;
    for (int i = 0; i < 7; ++i) {
      std::vector<ConflictLiteral> clause;
      for (int j = 0; j < 5; ++j)
        if (rng() % 2)
          clause.push_back({j, false, 1});
      if (!clause.empty())
        clauses.push_back(clause);
    }
    double optimum = kInf;
    for (int bits = 0; bits < 32; ++bits) {
      std::vector<double> x(5);
      for (int j = 0; j < 5; ++j)
        x[j] = (bits >> j) & 1;
      bool feasible = true;
      for (const auto &cl : clauses) {
        bool hit = false;
        for (const auto &l : cl)
          hit = hit || x[l.var] == 1;
        feasible = feasible && hit;
      }
      if (feasible)
        optimum = std::min(optimum, lp.objective(x));
    }
    for (std::uint64_t budget : {0u, 5u, 100000u}) {
      const auto r =
          price_objective_cores(lp, {1, 1, 1, 1, 1}, clauses, budget);
      CHECK(r.lower_bound <= optimum);
      if (budget == 100000 && !clauses.empty())
        CHECK(r.lower_bound == optimum);
    }
  }
}
void test_pricing_supports_original_feasible_points() {
  auto lp = binary_model(3);
  lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {2, 2, 2});
  lp.row_lo = {3};
  lp.row_hi = {kInf};
  const std::vector<double> x{.5, .5, .5};
  FiniteBlockOracle blocks;
  blocks.build(lp, {1, 1, 1}, 5, 100000);
  CHECK(!blocks.infeasible());
  auto cuts = blocks.separate(lp, x, 10, 1e-9, 100000);
  auto surrogate =
      separate_surrogate_knapsacks(lp, x, {1, 1, 1}, 10, 1e-9, 100000);
  CHECK(!cuts.empty());
  CHECK(!surrogate.empty());
  cuts.insert(cuts.end(), surrogate.begin(), surrogate.end());
  for (int bits = 0; bits < 8; ++bits) {
    std::vector<double> point(3);
    for (int j = 0; j < 3; ++j)
      point[j] = (bits >> j) & 1;
    if (lp.max_row_violation(point) > 0)
      continue;
    for (const auto &cut : cuts)
      admits(cut, point);
  }
  const auto split = sparse_integer_split(lp, x, {1, 1, 1}, 1e-9);
  CHECK(split.has_value());
  if (split)
    for (int bits = 0; bits < 8; ++bits) {
      double value = 0;
      for (std::size_t k = 0; k < split->direction.cols.size(); ++k)
        value +=
            split->direction.vals[k] * ((bits >> split->direction.cols[k]) & 1);
      CHECK(value <= split->floor || value >= split->floor + 1);
    }
}
void test_conditional_proof_publication() {
  auto lp = binary_model(3);
  lp.is_integer = {true, false, false};
  lp.col_hi = {1, 3, 3};
  lp.c = {0, -1, -1};
  lp.A = from_triplets(2, 3, {0, 0, 1, 1}, {0, 1, 0, 2}, {-1, 1, 1, 1});
  lp.row_lo = {-kInf, -kInf};
  lp.row_hi = {1, 2};
  BabOptions opts;
  opts.structural_presolve.enabled = false;
  opts.para_bab.threads = 1;
  BabDiagnostics diag;
  auto raw = solve_milp(lp, opts, diag);
  CHECK(diag.component_count > 0);
  const auto result =
      sor::certify::finalize_result(std::move(raw), milp_evidence(diag, opts));
  CHECK(result.status == sor::core::Status::Optimal);
  CHECK_NEAR(result.objective, -3, 1e-7);
  CHECK(diag.final_primal_violation <= opts.primal_feas_tol);
}
void test_block_decimal_row_contract() {
  auto lp = binary_model(3);
  lp.col_lo = {57, 0, 57};
  lp.col_hi = {72, 18, 75};
  lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {.9, 1, -1});
  lp.row_lo = {0};
  lp.row_hi = {0};
  const std::vector<double> point{60, 6, 60};
  CHECK(lp.max_row_violation(point) <= 1e-7);
  FiniteBlockOracle blocks;
  blocks.build(lp, {1, 1, 1}, 5, 100000, 1e-7);
  CHECK(!blocks.infeasible());
  for (const auto &cut : blocks.separate(lp, {60, 5, 60}, 10, 1e-9, 100000))
    admits(cut, point);
}
void test_recourse_original_space() {
  auto lp = binary_model(2);
  lp.is_integer = {true, false};
  lp.col_hi = {1, 4};
  lp.c = {0, 1};
  lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {2, 1});
  lp.row_lo = {3};
  lp.row_hi = {kInf};
  sor::engines::SimplexOptions options;
  options.presolve = false;
  RecourseService service;
  service.build(lp, {1, 0}, options);
  const auto cuts = service.separate({.5, 0}, options, 1e-9);
  CHECK(!cuts.empty());
  for (int x = 0; x <= 1; ++x)
    for (int y = 0; y <= 4; ++y)
      if (2 * x + y >= 3)
        for (const auto &row : cuts)
          admits(row, {static_cast<double>(x), static_cast<double>(y)});
}
void test_scoped_row_side_refresh() {
  auto global = binary_model(2);
  global.is_integer = {false, false};
  global.c = {-1, -1};
  global.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1, 1});
  global.row_lo = {-kInf};
  global.row_hi = {2};
  CutRow local;
  local.cols = {0};
  local.vals = {1};
  local.row_lo = -kInf;
  local.row_hi = .5;
  auto workspace = global;
  workspace.A.append_row(local.cols, local.vals);
  workspace.row_lo.push_back(local.row_lo);
  workspace.row_hi.push_back(local.row_hi);
  sor::engines::SimplexOptions options;
  options.presolve = false;
  sor::engines::DualProbeSession session(workspace, options);
  sor::engines::SimplexDiagnostics sd;
  sor::engines::SimplexBasis basis;
  const auto first = session.solve(options, sd, &basis, nullptr);
  CHECK_NEAR(first.objective, -1.5, 1e-7);
  global.row_hi[0] = 1;
  refresh_scoped_row_bounds(workspace, global, {local});
  CHECK(workspace.row_hi.size() == 2 && workspace.row_hi[1] == .5);
  session.set_row_bounds(workspace.row_lo, workspace.row_hi);
  sor::engines::SimplexDiagnostics after;
  const auto next = session.solve(options, after, nullptr, &basis);
  CHECK_NEAR(next.objective, -1, 1e-7);
  CHECK(workspace.max_row_violation(next.x) <= options.primal_feas_tol);
}
void test_integer_separator_and_constant_row_band() {
  auto lp = binary_model(3);
  lp.is_integer = {true, false, false};
  lp.col_hi = {3, 4, 4};
  lp.c = {0, -1, -1};
  lp.A = from_triplets(2, 3, {0, 0, 1, 1}, {0, 1, 0, 2}, {-1, 1, 1, 1});
  lp.row_lo = {-kInf, -kInf};
  lp.row_hi = {1, 4};
  CHECK(integer_separator(lp) == 0);
  BabOptions options;
  options.structural_presolve.enabled = false;
  options.para_bab.threads = 1;
  BabDiagnostics diag;
  auto raw = solve_milp(lp, options, diag);
  auto result = sor::certify::finalize_result(std::move(raw),
                                              milp_evidence(diag, options));
  CHECK(result.status == sor::core::Status::Optimal);
  CHECK_NEAR(result.objective, -5, 1e-7);
  // A fully fixed decimal row is consistent with the original primal band.
  auto fixed = binary_model(5);
  fixed.col_lo = {60, 6, 60, 0, 0};
  fixed.col_hi = {60, 6, 60, 1, 1};
  fixed.c = {0, 0, 0, -1, -1};
  fixed.A =
      from_triplets(3, 5, {0, 0, 0, 1, 2}, {0, 1, 2, 3, 4}, {.9, 1, -1, 1, 1});
  fixed.row_lo = {0, -kInf, -kInf};
  fixed.row_hi = {0, 1, 1};
  BabDiagnostics fd;
  auto fr = solve_milp(fixed, options, fd);
  auto final =
      sor::certify::finalize_result(std::move(fr), milp_evidence(fd, options));
  CHECK(final.status == sor::core::Status::Optimal);
  CHECK_NEAR(final.objective, -2, 1e-7);
  // An unsearched conditioning range must remain represented by its bound.
  options.max_nodes = 0;
  BabDiagnostics limited;
  const auto unfinished = solve_milp(lp, options, limited);
  CHECK(!limited.globally_proved);
  CHECK(unfinished.proposed_status != sor::core::Status::Infeasible);
  CHECK(limited.dual_bound <= -5);
}
void test_presolve_basis_rank() {
  MilpPresolveResult pre;
  pre.reduced = binary_model(2);
  pre.reduced.A = from_triplets(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {1, 1, 2, 2});
  pre.reduced.row_lo = {-kInf, -kInf};
  pre.reduced.row_hi = {1, 2};
  pre.reduced_col = {0, 1, -1};
  pre.reduced_row = {0, -1, 1};
  sor::engines::SimplexBasis parent;
  parent.n_struct = 3;
  parent.basic = {0, 1, 2};
  using S = sor::engines::NonbasicStatus;
  parent.status = {S::Basic,   S::Basic,   S::Basic,
                   S::AtUpper, S::AtUpper, S::AtUpper};
  auto mapped = map_basis_through_presolve(parent, pre);
  CHECK(mapped.has_value());
  if (mapped) {
    std::vector<Index> singular, vacant;
    basis_rank_selection(pre.reduced, 2, mapped->basic, singular, vacant);
    CHECK(singular.empty());
    int basics = 0;
    for (auto s : mapped->status)
      basics += s == S::Basic;
    CHECK(basics == 2);
  }
}
} // namespace
void test_boundary_projection_messages_and_cost_ownership() {
  auto lp = binary_model(5);
  // The two switches jointly couple x and y; z sees only the first switch.
  // No row shift is rounded, and the switches' cost is paid once by the parent.
  lp.c = {.25, .25, 1, 2, 3};
  lp.A = from_triplets(6, 5,
      {0,0,1,1,2,2,3,3,4,4,5,5},
      {0,1,0,2,1,2,0,3,1,3,0,4},
      {1,1,1,1,1,1,1,1,1,1,1,1});
  lp.row_lo.assign(6, 1); lp.row_hi.assign(6, kInf);
  const auto plan = compile_conditional_components(lp, 0);
  CHECK(plan.has_value());
  CHECK(plan->separator == std::vector<Index>({0,1}));
  CHECK(plan->components.size() == 3);
  for (const auto& block : plan->components)
    for (std::size_t q = block.owned; q < block.columns.size(); ++q)
      CHECK(block.model.c[q] == 0);
  BabOptions opts;
  opts.structural_presolve.enabled = false; opts.para_bab.threads = 1;
  BabDiagnostics diag;
  auto raw = solve_milp(lp, opts, diag);
  CHECK(diag.conditional_contexts > 0);
  CHECK(diag.conditional_cache_hits > 0);
  auto result = sor::certify::finalize_result(std::move(raw), milp_evidence(diag, opts));
  CHECK(result.status == sor::core::Status::Optimal);
  CHECK_NEAR(result.objective, .5, 1e-7);
  lp.maximize = true; for (auto& c : lp.c) c = -c;
  raw = solve_milp(lp, opts, diag);
  result = sor::certify::finalize_result(std::move(raw), milp_evidence(diag, opts));
  CHECK(result.status == sor::core::Status::Optimal);
  CHECK_NEAR(result.objective, -.5, 1e-7);
  opts.max_nodes = 0;
  raw = solve_milp(lp, opts, diag);
  CHECK(!diag.globally_proved);
  CHECK(raw.proposed_status != sor::core::Status::Infeasible);
  CHECK(raw.dual_bound >= -.5);
}

void test_domain_checkpoint_ownership() {
  constexpr std::size_t n = DomainCheckpoint::page_size * 3 + 7;
  std::vector<double> lo(n, 0), hi(n, 10);
  auto parent = DomainCheckpoint::capture(lo, hi, {});
  auto left = parent, right = parent;
  left.set_upper(1, 4);
  right.apply({{1, 6, false}, {n-1, 3, true}, {n-1, 2, true}});
  CHECK(parent.lower(1) == 0 && parent.upper(1) == 10);
  CHECK(left.lower(1) == 0 && left.upper(1) == 4);
  CHECK(right.lower(1) == 6 && right.upper(1) == 10);
  CHECK(right.upper(n-1) == 2);
  CHECK(!left.shares_page(right, 1));
  CHECK(left.shares_page(right, DomainCheckpoint::page_size));
  CHECK(parent.shares_page(left, n-1));
  left.materialize(lo, hi);
  hi[DomainCheckpoint::page_size + 3] = 5;
  auto continued = DomainCheckpoint::capture(lo, hi, left);
  CHECK(continued.shares_page(left, 1));
  CHECK(!continued.shares_page(left, DomainCheckpoint::page_size));
  CHECK(continued.shares_page(left, n-1));
  // Mutable worker arrays cannot mutate any captured node.
  lo[1] = 99; hi[1] = -99;
  CHECK(continued.lower(1) == 0 && continued.upper(1) == 4);
  CHECK(left.upper(DomainCheckpoint::page_size + 3) == 10);
}

void test_certified_cutoff_lattice_endpoint() {
  for (bool maximize : {false, true}) {
    auto lp = binary_model(1);
    lp.maximize = maximize;
    lp.c = {maximize ? 14.0 : -14.0};
    lp.obj_offset = .25;
    lp.A = from_triplets(1, 1, {0}, {0}, {1});
    lp.row_lo = {-kInf}; lp.row_hi = {0};
    const std::vector<double> incumbent{0};
    BabOptions options;
    options.structural_presolve.enabled = false;
    options.component_solve = false;
    options.mip_presolve = options.probing = options.symmetry = false;
    options.cuts_enabled = false;
    options.initial_solution = &incumbent;
    options.para_bab.threads = 1;
    options.gap_tol = options.abs_gap_tol = 1e-9;
    BabDiagnostics diag;
    auto raw = solve_milp(lp, options, diag);
    CHECK(diag.globally_proved);
    CHECK(raw.proposed_status == sor::core::Status::Optimal);
    CHECK_NEAR(raw.objective, .25, 1e-12);
    CHECK_NEAR(raw.dual_bound, .25, 1e-12);
  }
}

void test_failed_residue_probe_scope() {
    auto lp = binary_model(3); lp.col_hi = {6,1,1};
    lp.A = from_triplets(1,3,{0,0,0},{0,1,2},{1,3,3}); lp.row_lo = lp.row_hi = {6};
    ResidueSystem residues; residues.build(lp,{1,1,1},100000);
    ResidueState state; std::uint64_t changes = 0;
    auto lo = lp.col_lo, hi = lp.col_hi;
    CHECK(residues.propagate(lo,hi,changes,nullptr,0,nullptr,&state));
    lo[0] = hi[0] = 1;
    CHECK(!residues.propagate(lo,hi,changes,nullptr,1,nullptr,&state));
    lo = lp.col_lo; hi = lp.col_hi; lo[0] = hi[0] = 3;
    CHECK(residues.propagate(lo,hi,changes,nullptr,1,nullptr,&state));
}
void test_owned_domain_proof_and_conflict_seed() {
    auto lp = binary_model(2);
    lp.A = from_triplets(1,2,{0,0},{0,1},{1,1}); lp.row_lo = {-kInf}; lp.row_hi = {1};
    auto lo = lp.col_lo, hi = lp.col_hi;
    PropTrail trail; DomainEventLog events;
    write_domain_bound(lo,hi,0,BoundDir::Lower,1,&trail,{ReasonKind::Branch,-1,1},&events);
    const auto mark = trail.size();
    write_domain_bound(lo,hi,1,BoundDir::Lower,1,&trail,{ReasonKind::Branch,-1,2},&events);
    // A row contradiction is not an emptied coordinate domain. Both lower
    // bounds seed its proof, so the learned clause retains both premises.
    CHECK(explain_conflict_clause(trail,lp,1,{1,1},lp.col_lo,lp.col_hi,lo,hi,0).empty());
    std::vector<BoundRef> conflict{{0,false},{1,false}};
    const auto clause = explain_conflict_clause(trail,lp,1,{1,1},lp.col_lo,lp.col_hi,lo,hi,-1,false,nullptr,nullptr,&conflict);
    CHECK(clause.size() == 2);
    PropTrail scoped;
    auto scoped_lo = lp.col_lo, scoped_hi = lp.col_hi;
    write_domain_bound(scoped_lo,scoped_hi,0,BoundDir::Lower,1,&scoped,{ReasonKind::Cutoff,-1,0});
    write_domain_bound(scoped_lo,scoped_hi,1,BoundDir::Lower,1,&scoped,{ReasonKind::Branch,-1,1});
    const auto scoped_clause = explain_conflict_clause(scoped,lp,1,{1,1},lp.col_lo,lp.col_hi,
        scoped_lo,scoped_hi,-1,false,nullptr,nullptr,&conflict);
    CHECK(scoped_clause.size() == 2); // Cutoff premises survive root-level resolution.
    undo_domain(lo,hi,trail,mark,&events);
    CHECK(lo[1] == 0 && trail.size() == mark);
    CHECK(events.events().back().new_bound == 0);
    ConflictStore store(2,100000,20);
    CHECK(store.add({{0,true,0},{1,true,0}},lp.col_lo,lp.col_hi));
    std::vector<Index> seeds{0};
    CHECK(store.propagate_from(lo,hi,seeds,nullptr,&trail,1));
    CHECK(hi[1] == 0 && trail.entries().back().clause);
    store = ConflictStore(2,100000,20);
    write_domain_bound(lo,hi,1,BoundDir::Lower,1,&trail,{ReasonKind::Branch,-1,2});
    const auto pinned = explain_conflict_clause(trail,lp,1,{1,1},lp.col_lo,lp.col_hi,lo,hi,1,false,&store);
    CHECK(!pinned.empty());
    // Cached flattened histories observe a remap made through a sibling.
    PropTrail a; a.push(0,BoundDir::Lower,1,0,ReasonKind::Row,0,0);
    PropTrail b = a; CHECK(b.entries()[0].reason_id == 0);
    a.remap_row_reasons({2},[](Index) { return Index{0}; },7);
    CHECK(b.entries()[0].reason_id == 2);
}
void test_complete_integer_oracle() {
    auto lp = binary_model(3); lp.c = {0,0,1};
    lp.A = from_triplets(1,3,{0,0,0},{0,1,2},{1,2,3}); lp.row_lo = lp.row_hi = {3};
    IntegerDomainOracle oracle(lp,0);
    auto result = oracle.advance(1);
    CHECK(result.status == IntegerDomainOracle::Status::Pending);
    while (result.status == IntegerDomainOracle::Status::Pending) result = oracle.advance(1);
    CHECK(result.status == IntegerDomainOracle::Status::Optimal);
    CHECK(result.point == std::vector<double>({1,1,0}));
    CHECK(result.lower_bound_min == 0);
    auto zeros = binary_model(4); zeros.c = {0,0,1,0};
    zeros.A = from_triplets(2,4,{0,0,0,1,1},{0,1,3,2,3},{1,1,1,1,1});
    zeros.A.vals[2] = 0; zeros.row_lo = zeros.row_hi = {1,1};
    IntegerDomainOracle structural_zero(zeros,0);
    const auto zero_answer = structural_zero.advance(100000);
    CHECK(zero_answer.status == IntegerDomainOracle::Status::Optimal);
    CHECK(zero_answer.lower_bound_min == 0);
    for (unsigned seed = 1; seed <= 40; ++seed) {
        std::mt19937 rng(seed); auto p = binary_model(5);
        std::vector<Index> rows, cols; std::vector<double> values;
        for (Index i = 0; i < 3; ++i) for (Index j = 0; j < 5; ++j) {
            const double a = static_cast<int>(rng()%7)-3;
            if (a) { rows.push_back(i); cols.push_back(j); values.push_back(a); }
        }
        p.A = from_triplets(3,5,rows,cols,values);
        p.row_lo = {-2,-kInf,-1}; p.row_hi = {2,1,1}; p.maximize = seed%2;
        for (auto& c : p.c) c = (static_cast<int>(rng()%11)-5)*.25;
        double best = kInf;
        for (unsigned mask = 0; mask < 32; ++mask) {
            std::vector<double> x(5); for (unsigned j = 0; j < 5; ++j) x[j] = (mask>>j)&1;
            if (p.max_row_violation(x) == 0) best = std::min(best,(p.maximize?-1:1)*p.objective(x));
        }
        IntegerDomainOracle exact(p,0); auto answer = exact.advance(100000);
        if (best == kInf) CHECK(answer.status == IntegerDomainOracle::Status::Infeasible);
        else { CHECK(answer.status == IntegerDomainOracle::Status::Optimal); CHECK_NEAR(answer.lower_bound_min,best,0); CHECK(p.max_row_violation(answer.point) == 0); }
    }
}
void test_search_session_retains_frontier_and_clock() {
    auto lp = binary_model(4); lp.c = {.99,1.01,1.03,1.07};
    lp.A = from_triplets(1,4,{0,0,0,0},{0,1,2,3},{1,1,1,1}); lp.row_lo = {2.5}; lp.row_hi = {kInf};
    BabOptions options; options.policy = MilpPolicy::Classical;
    options.structural_presolve.binary_row_max_work = 0;
    options.structural_presolve.enabled = options.component_solve = options.mip_presolve = options.probing = options.symmetry = false;
    options.cuts_enabled = options.rounding_heuristic = options.lp_rounding_repair = options.integer_dive = false;
    options.fixprop = options.root_primal_early = options.root_primal_final = false;
    options.integer_row_rounding = false;
    options.reliability_branching = false; // Retain a frontier rather than closing it with strong probes.
    options.domain_propagation = options.objective_propagation = options.reduced_cost_strengthening = false;
    options.feasibility_jump = options.feasibility_pump_root = options.feasibility_pump_improve = false;
    options.integer_neighborhood = options.face_vertices = false;
    options.para_bab.threads = 1; options.time_limit_s = 20;
    MilpSearchSession session(lp,options);
    auto result = session.advance(kInf,1);
    CHECK(!session.finished());
    const auto nodes = session.diagnostics().nodes;
    CHECK(nodes == 1);
    const double first_bound = result.dual_bound;
    // Work belonging to a different scheduling context must not exhaust this
    // retained solve's own allowance or become one of its node transactions.
    sor::core::work_add(100000, sor::core::WorkTag::BabNode);
    while (!session.finished()) result = session.advance(kInf,1);
    CHECK(result.proposed_status == sor::core::Status::Optimal);
    CHECK(session.diagnostics().globally_proved);
    CHECK_NEAR(result.objective,3.03,1e-9);
    CHECK(first_bound <= result.objective);
    BabDiagnostics uninterrupted;
    const auto baseline = solve_milp(lp,options,uninterrupted);
    CHECK_NEAR(result.objective,baseline.objective,1e-9);
    CHECK(session.diagnostics().nodes == uninterrupted.nodes);
    CHECK(session.diagnostics().lp_solves == uninterrupted.lp_solves);
}

void test_search_session_compositional_resume() {
    auto lp = binary_model(3);
    lp.is_integer = {true,false,false}; lp.col_hi = {1,3,3}; lp.c = {0,-1,-1};
    lp.A = from_triplets(2,3,{0,0,1,1},{0,1,0,2},{-1,1,1,1});
    lp.row_lo = {-kInf,-kInf}; lp.row_hi = {1,2};
    BabOptions options; options.structural_presolve.enabled = false;
    options.para_bab.threads = 1; options.time_limit_s = 20;
    MilpSearchSession conditional(lp,options);
    auto result = conditional.advance(kInf,0);
    CHECK(!conditional.finished()); CHECK(conditional.diagnostics().component_count >= 2);
    CHECK(result.dual_bound <= -3);
    while (!conditional.finished()) result = conditional.advance(kInf,1);
    CHECK(result.proposed_status == sor::core::Status::Optimal);
    CHECK_NEAR(result.objective,-3,1e-7);
    CHECK(lp.max_row_violation(result.x) <= options.primal_feas_tol);

    // An eliminated fixed column must be restored on both suspended and
    // terminal publication; the reduced solver's frontier remains owned.
    auto reduced = binary_model(5); reduced.c = {2,.99,1.01,1.03,1.07};
    reduced.col_lo[0] = reduced.col_hi[0] = 1;
    reduced.A = from_triplets(1,5,{0,0,0,0},{1,2,3,4},{1,1,1,1});
    reduced.row_lo = {2.5}; reduced.row_hi = {kInf};
    options.policy = MilpPolicy::Classical;
    options.component_solve = options.mip_presolve = options.probing = options.symmetry = false;
    options.cuts_enabled = options.rounding_heuristic = options.lp_rounding_repair = options.integer_dive = false;
    options.fixprop = options.root_primal_early = options.root_primal_final = false;
    options.integer_row_rounding = options.root_restart = options.tree_restart = false;
    options.structural_presolve.enabled = true;
    auto& pre = options.structural_presolve;
    pre.structural_fbbt = pre.binary_row_support = pre.coefficient_strengthening = pre.side_rounding = false;
    pre.probing_presolve = pre.aggregation = pre.column_dominance = pre.monotone_binary_pairs = false;
    pre.row_probe.enabled = false;
    MilpSearchSession presolved(reduced,options);
    result = presolved.advance(kInf,0);
    CHECK(!presolved.finished()); CHECK(presolved.diagnostics().structural_presolve.cols_after == 4);
    while (!presolved.finished()) result = presolved.advance(kInf,1);
    CHECK(result.proposed_status == sor::core::Status::Optimal);
    CHECK(result.x.size() == 5); CHECK(result.x[0] == 1);
    CHECK_NEAR(result.objective,5.03,1e-7);

    // Disconnected blocks share the same retained master. An isolated
    // unbounded cost still supplies a finite feasible witness for its ray.
    auto unbounded = binary_model(3); unbounded.col_hi[2] = kInf;
    unbounded.c = {-1,-1,-1};
    unbounded.A = from_triplets(2,3,{0,1},{0,1},{1,1});
    unbounded.row_lo = {-kInf,-kInf}; unbounded.row_hi = {1,1};
    options.structural_presolve.enabled = false; options.component_solve = true;
    MilpSearchSession disconnected(unbounded,options);
    result = disconnected.advance(kInf,0); CHECK(!disconnected.finished());
    while (!disconnected.finished()) result = disconnected.advance(kInf,1);
    CHECK(result.proposed_status == sor::core::Status::Unbounded);
    CHECK(result.x.size() == 3); CHECK(unbounded.max_bound_violation(result.x) == 0);
    CHECK(disconnected.diagnostics().unbounded_certified);
}

int main() {
  test_failed_residue_probe_scope();
  test_owned_domain_proof_and_conflict_seed();
  test_complete_integer_oracle();
  test_search_session_retains_frontier_and_clock();
  test_search_session_compositional_resume();
  test_domain_checkpoint_ownership();
  test_certified_cutoff_lattice_endpoint();
  test_exact_delivery_and_probe_geometry();
  test_modular_domain_lifetime();
  test_matching_soundness_and_pending();
  test_overlapping_objective_cores();
  test_pricing_supports_original_feasible_points();
  test_boundary_projection_messages_and_cost_ownership();
  test_conditional_proof_publication();
  test_block_decimal_row_contract();
  test_recourse_original_space();
  test_scoped_row_side_refresh();
  test_integer_separator_and_constant_row_band();
  test_presolve_basis_rank();
  return sor::test::finish("test_milp_services");
}
