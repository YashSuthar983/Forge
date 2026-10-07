function(sor_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE ${ARGN} sor_flags)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

sor_add_test(test_exact_model         sor_model)
sor_add_test(test_csr                 sor_sparse)
sor_add_test(test_spmv_plan           sor_sparse)
sor_add_test(test_csc                 sor_sparse)
sor_add_test(test_lu                  sor_la_cpu)
sor_add_test(test_basis_numerics       sor_la_cpu)
sor_add_test(test_quad_accumulator     sor_la_cpu)
sor_add_test(test_lu_ft               sor_la_cpu)
sor_add_test(test_backend_parity      sor_backend)
sor_add_test(test_pdhcg_device        sor_backend sor_engines sor_io)
sor_add_test(test_no_unproved_optimal sor_certify)
sor_add_test(test_lp_certificates      sor_certify sor_io sor_engines)
sor_add_test(test_mps                 sor_io)
sor_add_test(test_lp_format           sor_io)
sor_add_test(test_solution_io         sor_io)
sor_add_test(test_pdhg                sor_engines sor_certify sor_io)
sor_add_test(test_simplex             sor_engines sor_certify sor_io)
sor_add_test(test_primal_bound_perturbation sor_engines sor_certify)
sor_add_test(test_dispatch_regressions sor_engines sor_certify sor_io)
sor_add_test(test_dual_edge_weights   sor_engines)
sor_add_test(test_factor_reuse        sor_engines sor_certify sor_sparse)
sor_add_test(test_dual_ratio_test     sor_engines)
sor_add_test(test_dual_cost_perturbation sor_engines)
sor_add_test(test_presolve            sor_presolve)
sor_add_test(test_advanced_presolve   sor_presolve)
sor_add_test(test_hpr                 sor_engines sor_certify sor_io)
sor_add_test(test_lp_device           sor_backend sor_engines sor_io)
sor_add_test(test_binquad_device      sor_backend sor_search sor_io)
sor_add_test(test_binquad_milp_heuristic sor_backend sor_search sor_certify sor_io)
target_compile_definitions(test_binquad_device PRIVATE
  SOR_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
sor_add_test(test_lp_batched          sor_backend sor_engines sor_io)
sor_add_test(test_r2hpdg              sor_backend sor_engines)
sor_add_test(test_hpr_integration     sor_engines sor_certify sor_io)
sor_add_test(test_crossover           sor_engines sor_certify sor_io)
sor_add_test(test_lp_auto             sor_engines sor_certify sor_io)
sor_add_test(test_lp_barrier          sor_engines sor_certify sor_io)
sor_add_test(test_padic_solve         sor_certify)
target_compile_definitions(test_lp_barrier PRIVATE SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(test_simplex PRIVATE SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
sor_add_test(test_lp_concurrent       sor_engines sor_certify)
sor_add_test(test_milp                sor_search sor_certify sor_io)
add_test(NAME test_milp_root_handoff COMMAND test_milp --root-handoff)
sor_add_test(test_milp_diagnostics    sor_search sor_certify sor_io)
sor_add_test(test_milp_bound_evidence sor_search sor_certify sor_io)
sor_add_test(test_root_restart         sor_search sor_certify sor_io)
sor_add_test(test_reliability_branching sor_search sor_certify sor_io)
sor_add_test(test_event_propagation    sor_search sor_certify sor_io)
sor_add_test(test_fpump               sor_search sor_engines sor_certify sor_io)
sor_add_test(test_branch_features     sor_search sor_certify sor_io)
target_compile_definitions(test_branch_features PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(test_milp_bound_evidence PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(test_milp_diagnostics PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
sor_add_test(test_bab_gpu_binary_heuristic sor_search sor_certify sor_io)
sor_add_test(test_bab_node_lp_carriers sor_search sor_certify sor_io)
sor_add_test(test_milp_oracle         sor_search sor_certify sor_io)
sor_add_test(test_milp_presolve       sor_search sor_certify sor_io)
sor_add_test(test_presolve_coefficients sor_search sor_engines sor_certify sor_io)
sor_add_test(test_presolve_live_lift sor_engines sor_presolve sor_certify)
sor_add_test(test_milp_random_oracle  sor_search sor_certify sor_io)
sor_add_test(test_milp_portfolio      sor_search sor_certify sor_io)
sor_add_test(test_spp_repair  sor_search)
target_compile_definitions(test_milp PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
sor_add_test(test_milp_policy         sor_search sor_certify sor_io)
sor_add_test(test_sparse_sb           sor_search sor_certify sor_io)
sor_add_test(test_sc_milp_branch      sor_search sor_certify sor_io)
sor_add_test(test_lifted_branch       sor_search sor_certify sor_io)
sor_add_test(test_planbb              sor_search sor_certify sor_io)
sor_add_test(test_tree_cuts           sor_search sor_certify sor_io)
sor_add_test(test_cut_scope           sor_search sor_certify sor_io)
sor_add_test(test_conflict_trail      sor_search)
sor_add_test(test_conflict_cut        sor_search sor_certify sor_io)
sor_add_test(test_lattice_reform      sor_search sor_certify sor_io)
sor_add_test(test_cuts                sor_search sor_engines sor_certify sor_io)
sor_add_test(test_tableau_cmir       sor_search sor_engines sor_certify sor_io)
sor_add_test(test_node_cut_resolve    sor_search sor_engines sor_certify sor_io)
sor_add_test(test_conflict_store       sor_search sor_engines sor_certify sor_io)
sor_add_test(test_farkas_conflict     sor_search sor_engines sor_certify sor_io)
sor_add_test(test_component_solve      sor_search sor_engines sor_certify sor_io)
sor_add_test(test_column_merge         sor_search sor_engines sor_certify sor_io)
sor_add_test(test_cut_policy          sor_search sor_engines sor_certify sor_io)
sor_add_test(test_propagate           sor_search)
sor_add_test(test_domain_fixpoint     sor_search)
sor_add_test(test_conflict            sor_search)
sor_add_test(test_branch_stats        sor_search)
sor_add_test(test_basis_map           sor_search)
sor_add_test(test_relaxation_identity sor_search)
sor_add_test(test_probing_carry       sor_search)
sor_add_test(test_mip_presolve        sor_search sor_certify sor_io)
sor_add_test(test_gf2_presolve        sor_search)
sor_add_test(test_implied_int         sor_search)
sor_add_test(test_symmetry            sor_search sor_certify sor_io)
sor_add_test(test_feasjump            sor_search)
sor_add_test(test_fixprop             sor_search)
sor_add_test(test_primal_polish       sor_search)
sor_add_test(test_lns                 sor_search)
sor_add_test(test_balans              sor_search)
sor_add_test(test_kernel_pump         sor_search)
sor_add_test(test_mrens               sor_search)
sor_add_test(test_btbs_lns            sor_search)
sor_add_test(test_cl_tlns             sor_search)
sor_add_test(test_covers              sor_search)
sor_add_test(test_mir                 sor_search sor_engines sor_certify sor_io)
sor_add_test(test_zerohalf            sor_search)
sor_add_test(test_flowcover           sor_search)
sor_add_test(test_dynsep              sor_search sor_certify sor_io)
sor_add_test(test_l2sep              sor_search sor_certify sor_io)
sor_add_test(test_hgtsm              sor_search sor_certify sor_io)
sor_add_test(test_qp                  sor_engines sor_certify)
sor_add_test(test_qps                 sor_io sor_engines sor_certify)
sor_add_test(test_ps_must             sor_search sor_engines sor_certify sor_io
                                      sor_la_cpu)
sor_add_test(test_extensions          sor_search sor_engines sor_certify)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Python tests for the scripts/ tooling. They import the script under test
# directly and never build or run a solver, so they stay fast enough to belong
# in the default ctest run.
function(sor_add_python_test name)
  add_test(NAME ${name}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py)
  set_tests_properties(${name} PROPERTIES
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endfunction()

sor_add_python_test(test_compare)
sor_add_python_test(test_fetch_benchmarks)
sor_add_python_test(test_qplib_eval)
sor_add_python_test(test_python_api)
sor_add_python_test(test_web_lp_text)
sor_add_python_test(test_bound_snapshots)
sor_add_python_test(test_forbidden_dependencies)
set_tests_properties(test_python_api test_bound_snapshots PROPERTIES
  ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1;SOR_BINARY_DIR=${CMAKE_BINARY_DIR}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(test_forbidden_dependencies PROPERTIES
  ENVIRONMENT
    "PYTHONDONTWRITEBYTECODE=1;SOR_SOURCE_DIR=${CMAKE_SOURCE_DIR};SOR_BINARY_DIR=${CMAKE_BINARY_DIR}")

# CMAKE_SOURCE_DIR / CMAKE_BINARY_DIR (not CURRENT): this test shells out to the
# built CLIs and reads instances from the repo root.
target_compile_definitions(test_ps_must PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  SOR_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_dependencies(test_ps_must sor_solve sor_check)

target_compile_definitions(test_dispatch_regressions PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_compile_definitions(test_hpr_integration PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Conflict-cut regressions use real MIPLIB fixtures. Resolve them from the
# source tree so CTest's build/tests directory cannot silently skip them.
target_compile_definitions(test_conflict_cut PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Argument validation runs the real CLIs and checks their exit codes, so it
# needs both binaries and the repo's models.
sor_add_test(test_cli_validation)
target_compile_definitions(test_cli_validation PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  SOR_BINARY_DIR="${CMAKE_BINARY_DIR}"
  $<$<BOOL:${SOR_ROUTE_FN}>:SOR_ROUTE_DEBUG=1>)
add_dependencies(test_cli_validation sor_solve sor_check)

# Restored from origin/main
sor_add_test(test_options             sor_search sor_engines sor_core)
sor_add_test(test_qplib               sor_search sor_io)
target_compile_definitions(test_qplib PRIVATE
  SOR_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/benchmarks")
sor_add_test(test_pooling             sor_search sor_io sor_engines)
target_compile_definitions(test_pooling PRIVATE
  SOR_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/benchmarks")
sor_add_test(test_ldlt               sor_la_cpu)
sor_add_test(test_qp_device           sor_backend sor_sparse)
sor_add_test(test_batched_pdhcg       sor_engines sor_backend sor_sparse)
sor_add_test(test_miqp_bb             sor_search sor_engines sor_certify)

sor_add_test(test_qplib_auto_engine)
target_compile_definitions(test_qplib_auto_engine PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  SOR_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_dependencies(test_qplib_auto_engine sor_solve)

sor_add_test(test_qcr                 sor_search sor_io)
target_compile_definitions(test_qcr PRIVATE
  SOR_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/benchmarks")
sor_add_test(test_bqp_bab             sor_search sor_io)
sor_add_test(test_qp_ipm              sor_engines sor_certify)
sor_add_test(test_qcqp_ipm            sor_search sor_engines sor_certify)
sor_add_test(test_qcqp                sor_search sor_engines sor_certify sor_io)
target_compile_definitions(test_qcqp PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  SOR_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_dependencies(test_qcqp sor_solve)
sor_add_test(test_global_qp          sor_search)

foreach(_t test_pdhcg_device test_batched_pdhcg test_lp_device test_qp_device)
  add_test(NAME ${_t}_profiled COMMAND ${_t})
  set_tests_properties(${_t}_profiled PROPERTIES ENVIRONMENT SOR_VK_PROFILE=1)
endforeach()
