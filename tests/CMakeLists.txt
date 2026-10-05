function(forge_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE ${ARGN} forge_flags)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

forge_add_test(test_exact_model         forge_model)
forge_add_test(test_csr                 forge_sparse)
forge_add_test(test_spmv_plan           forge_sparse)
forge_add_test(test_csc                 forge_sparse)
forge_add_test(test_lu                  forge_la_cpu)
forge_add_test(test_basis_numerics       forge_la_cpu)
forge_add_test(test_quad_accumulator     forge_la_cpu)
forge_add_test(test_lu_ft               forge_la_cpu)
forge_add_test(test_backend_parity      forge_backend)
forge_add_test(test_pdhcg_device        forge_backend forge_engines forge_io)
forge_add_test(test_no_unproved_optimal forge_certify)
forge_add_test(test_lp_certificates      forge_certify forge_io forge_engines)
forge_add_test(test_mps                 forge_io)
forge_add_test(test_lp_format           forge_io)
forge_add_test(test_solution_io         forge_io)
forge_add_test(test_pdhg                forge_engines forge_certify forge_io)
forge_add_test(test_simplex             forge_engines forge_certify forge_io)
forge_add_test(test_primal_bound_perturbation forge_engines forge_certify)
forge_add_test(test_dispatch_regressions forge_engines forge_certify forge_io)
forge_add_test(test_dual_edge_weights   forge_engines)
forge_add_test(test_factor_reuse        forge_engines forge_certify forge_sparse)
forge_add_test(test_dual_ratio_test     forge_engines)
forge_add_test(test_dual_cost_perturbation forge_engines)
forge_add_test(test_presolve            forge_presolve)
forge_add_test(test_advanced_presolve   forge_presolve)
forge_add_test(test_hpr                 forge_engines forge_certify forge_io)
forge_add_test(test_lp_device           forge_backend forge_engines forge_io)
forge_add_test(test_binquad_device      forge_backend forge_search forge_io)
forge_add_test(test_binquad_milp_heuristic forge_backend forge_search forge_certify forge_io)
target_compile_definitions(test_binquad_device PRIVATE
  FORGE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
forge_add_test(test_lp_batched          forge_backend forge_engines forge_io)
forge_add_test(test_r2hpdg              forge_backend forge_engines)
forge_add_test(test_hpr_integration     forge_engines forge_certify forge_io)
forge_add_test(test_crossover           forge_engines forge_certify forge_io)
forge_add_test(test_lp_auto             forge_engines forge_certify forge_io)
forge_add_test(test_lp_barrier          forge_engines forge_certify forge_io)
forge_add_test(test_padic_solve         forge_certify)
target_compile_definitions(test_lp_barrier PRIVATE FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
forge_add_test(test_lp_concurrent       forge_engines forge_certify)
forge_add_test(test_milp                forge_search forge_certify forge_io)
add_test(NAME test_milp_root_handoff COMMAND test_milp --root-handoff)
forge_add_test(test_milp_diagnostics    forge_search forge_certify forge_io)
forge_add_test(test_milp_bound_evidence forge_search forge_certify forge_io)
forge_add_test(test_root_restart         forge_search forge_certify forge_io)
forge_add_test(test_reliability_branching forge_search forge_certify forge_io)
forge_add_test(test_event_propagation    forge_search forge_certify forge_io)
forge_add_test(test_fpump               forge_search forge_engines forge_certify forge_io)
forge_add_test(test_branch_features     forge_search forge_certify forge_io)
target_compile_definitions(test_branch_features PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(test_milp_bound_evidence PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(test_milp_diagnostics PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
forge_add_test(test_bab_gpu_binary_heuristic forge_search forge_certify forge_io)
forge_add_test(test_bab_node_lp_carriers forge_search forge_certify forge_io)
forge_add_test(test_milp_oracle         forge_search forge_certify forge_io)
forge_add_test(test_milp_presolve       forge_search forge_certify forge_io)
forge_add_test(test_presolve_coefficients forge_search forge_engines forge_certify forge_io)
forge_add_test(test_presolve_live_lift forge_engines forge_presolve forge_certify)
forge_add_test(test_milp_random_oracle  forge_search forge_certify forge_io)
forge_add_test(test_milp_portfolio      forge_search forge_certify forge_io)
forge_add_test(test_spp_repair  forge_search)
target_compile_definitions(test_milp PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
forge_add_test(test_milp_policy         forge_search forge_certify forge_io)
forge_add_test(test_sparse_sb           forge_search forge_certify forge_io)
forge_add_test(test_sc_milp_branch      forge_search forge_certify forge_io)
forge_add_test(test_lifted_branch       forge_search forge_certify forge_io)
forge_add_test(test_planbb              forge_search forge_certify forge_io)
forge_add_test(test_tree_cuts           forge_search forge_certify forge_io)
forge_add_test(test_cut_scope           forge_search forge_certify forge_io)
forge_add_test(test_conflict_trail      forge_search)
forge_add_test(test_conflict_cut        forge_search forge_certify forge_io)
forge_add_test(test_lattice_reform      forge_search forge_certify forge_io)
forge_add_test(test_cuts                forge_search forge_engines forge_certify forge_io)
forge_add_test(test_tableau_cmir       forge_search forge_engines forge_certify forge_io)
forge_add_test(test_node_cut_resolve    forge_search forge_engines forge_certify forge_io)
forge_add_test(test_conflict_store       forge_search forge_engines forge_certify forge_io)
forge_add_test(test_farkas_conflict     forge_search forge_engines forge_certify forge_io)
forge_add_test(test_component_solve      forge_search forge_engines forge_certify forge_io)
forge_add_test(test_column_merge         forge_search forge_engines forge_certify forge_io)
forge_add_test(test_cut_policy          forge_search forge_engines forge_certify forge_io)
forge_add_test(test_propagate           forge_search)
forge_add_test(test_domain_fixpoint     forge_search)
forge_add_test(test_conflict            forge_search)
forge_add_test(test_branch_stats        forge_search)
forge_add_test(test_basis_map           forge_search)
forge_add_test(test_relaxation_identity forge_search)
forge_add_test(test_probing_carry       forge_search)
forge_add_test(test_mip_presolve        forge_search forge_certify forge_io)
forge_add_test(test_gf2_presolve        forge_search)
forge_add_test(test_implied_int         forge_search)
forge_add_test(test_symmetry            forge_search forge_certify forge_io)
forge_add_test(test_feasjump            forge_search)
forge_add_test(test_fixprop             forge_search)
forge_add_test(test_primal_polish       forge_search)
forge_add_test(test_lns                 forge_search)
forge_add_test(test_balans              forge_search)
forge_add_test(test_kernel_pump         forge_search)
forge_add_test(test_mrens               forge_search)
forge_add_test(test_btbs_lns            forge_search)
forge_add_test(test_cl_tlns             forge_search)
forge_add_test(test_covers              forge_search)
forge_add_test(test_mir                 forge_search forge_engines forge_certify forge_io)
forge_add_test(test_zerohalf            forge_search)
forge_add_test(test_flowcover           forge_search)
forge_add_test(test_dynsep              forge_search forge_certify forge_io)
forge_add_test(test_l2sep              forge_search forge_certify forge_io)
forge_add_test(test_hgtsm              forge_search forge_certify forge_io)
forge_add_test(test_qp                  forge_engines forge_certify)
forge_add_test(test_qps                 forge_io forge_engines forge_certify)
forge_add_test(test_ps_must             forge_search forge_engines forge_certify forge_io
                                      forge_la_cpu)
forge_add_test(test_extensions          forge_search forge_engines forge_certify)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Python tests for the scripts/ tooling. They import the script under test
# directly and never build or run a solver, so they stay fast enough to belong
# in the default ctest run.
function(forge_add_python_test name)
  add_test(NAME ${name}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py)
  set_tests_properties(${name} PROPERTIES
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endfunction()

forge_add_python_test(test_compare)
forge_add_python_test(test_fetch_benchmarks)
forge_add_python_test(test_qplib_eval)
forge_add_python_test(test_python_api)
forge_add_python_test(test_bound_snapshots)
forge_add_python_test(test_forbidden_dependencies)
set_tests_properties(test_python_api test_bound_snapshots PROPERTIES
  ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1;FORGE_BINARY_DIR=${CMAKE_BINARY_DIR}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(test_forbidden_dependencies PROPERTIES
  ENVIRONMENT
    "PYTHONDONTWRITEBYTECODE=1;FORGE_SOURCE_DIR=${CMAKE_SOURCE_DIR};FORGE_BINARY_DIR=${CMAKE_BINARY_DIR}")

# CMAKE_SOURCE_DIR / CMAKE_BINARY_DIR (not CURRENT): this test shells out to the
# built CLIs and reads instances from the repo root.
target_compile_definitions(test_ps_must PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  FORGE_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_dependencies(test_ps_must forge_solve forge_check)

target_compile_definitions(test_dispatch_regressions PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_compile_definitions(test_hpr_integration PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Conflict-cut regressions use real MIPLIB fixtures. Resolve them from the
# source tree so CTest's build/tests directory cannot silently skip them.
target_compile_definitions(test_conflict_cut PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Argument validation runs the real CLIs and checks their exit codes, so it
# needs both binaries and the repo's models.
forge_add_test(test_cli_validation)
target_compile_definitions(test_cli_validation PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  FORGE_BINARY_DIR="${CMAKE_BINARY_DIR}"
  $<$<BOOL:${FORGE_ROUTE_FN}>:FORGE_ROUTE_DEBUG=1>)
add_dependencies(test_cli_validation forge_solve forge_check)

# Restored from origin/main
forge_add_test(test_options             forge_search forge_engines forge_core)
forge_add_test(test_qplib               forge_search forge_io)
target_compile_definitions(test_qplib PRIVATE
  FORGE_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/benchmarks")
forge_add_test(test_pooling             forge_search forge_io forge_engines)
target_compile_definitions(test_pooling PRIVATE
  FORGE_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/benchmarks")
forge_add_test(test_ldlt               forge_la_cpu)
forge_add_test(test_qp_device           forge_backend forge_sparse)
forge_add_test(test_batched_pdhcg       forge_engines forge_backend forge_sparse)
forge_add_test(test_miqp_bb             forge_search forge_engines forge_certify)

forge_add_test(test_qplib_auto_engine)
target_compile_definitions(test_qplib_auto_engine PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  FORGE_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_dependencies(test_qplib_auto_engine forge_solve)

forge_add_test(test_qcr                 forge_search forge_io)
target_compile_definitions(test_qcr PRIVATE
  FORGE_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/benchmarks")
forge_add_test(test_bqp_bab             forge_search forge_io)
forge_add_test(test_qp_ipm              forge_engines forge_certify)
forge_add_test(test_qcqp_ipm            forge_search forge_engines forge_certify)
forge_add_test(test_qcqp                forge_search forge_engines forge_certify forge_io)
target_compile_definitions(test_qcqp PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  FORGE_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_dependencies(test_qcqp forge_solve)
forge_add_test(test_global_qp          forge_search)

foreach(_t test_pdhcg_device test_batched_pdhcg test_lp_device test_qp_device)
  add_test(NAME ${_t}_profiled COMMAND ${_t})
  set_tests_properties(${_t}_profiled PROPERTIES ENVIRONMENT FORGE_VK_PROFILE=1)
endforeach()
