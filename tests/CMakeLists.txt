function(forge_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE ${ARGN} forge_flags)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

forge_add_test(test_csr                 forge_sparse)
forge_add_test(test_csc                 forge_sparse)
forge_add_test(test_lu                  forge_la_cpu)
forge_add_test(test_lu_ft               forge_la_cpu)
forge_add_test(test_backend_parity      forge_backend)
forge_add_test(test_no_unproved_optimal forge_certify)
forge_add_test(test_lp_certificates      forge_certify forge_io)
forge_add_test(test_mps                 forge_io)
forge_add_test(test_solution_io         forge_io)
forge_add_test(test_pdhg                forge_engines forge_certify forge_io)
forge_add_test(test_simplex             forge_engines forge_certify forge_io)
forge_add_test(test_dispatch_regressions forge_engines forge_certify forge_io)
forge_add_test(test_dual_edge_weights   forge_engines)
forge_add_test(test_dual_ratio_test     forge_engines)
forge_add_test(test_dual_cost_perturbation forge_engines)
forge_add_test(test_presolve            forge_presolve)
forge_add_test(test_hpr                 forge_engines forge_certify forge_io)
forge_add_test(test_lp_device           forge_backend forge_engines forge_io)
forge_add_test(test_r2hpdg              forge_backend forge_engines)
forge_add_test(test_hpr_integration     forge_engines forge_certify forge_io)
forge_add_test(test_crossover           forge_engines forge_certify forge_io)
forge_add_test(test_lp_auto             forge_engines forge_certify forge_io)
forge_add_test(test_milp                forge_search forge_certify forge_io)
forge_add_test(test_lattice_reform      forge_search forge_certify forge_io)
forge_add_test(test_cuts                forge_search forge_engines forge_certify forge_io)
forge_add_test(test_propagate           forge_search)
forge_add_test(test_conflict            forge_search)
forge_add_test(test_feasjump            forge_search)
forge_add_test(test_lns                 forge_search)
forge_add_test(test_covers              forge_search)
forge_add_test(test_mir                 forge_search forge_engines forge_certify forge_io)
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
forge_add_python_test(test_compare_runs)
forge_add_python_test(test_gate)
forge_add_python_test(test_presolve_opportunities)
forge_add_python_test(test_repro_bench)
forge_add_python_test(test_verify_netlib_results)
forge_add_python_test(test_freeze_manifest)
forge_add_python_test(test_claim_run)
forge_add_python_test(test_lp_auto_train)
forge_add_python_test(test_forbidden_dependencies)
set_tests_properties(test_forbidden_dependencies PROPERTIES
  ENVIRONMENT
    "PYTHONDONTWRITEBYTECODE=1;FORGE_SOURCE_DIR=${CMAKE_SOURCE_DIR};FORGE_BINARY_DIR=${CMAKE_BINARY_DIR}")

# CMAKE_SOURCE_DIR / CMAKE_BINARY_DIR (not CURRENT): this test shells out to the
# built CLIs and reads instances from the repo root.
target_compile_definitions(test_ps_must PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  FORGE_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_dependencies(test_ps_must forge_solve forge_check forge_ext_demo)

target_compile_definitions(test_dispatch_regressions PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_compile_definitions(test_hpr_integration PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Argument validation runs the real CLIs and checks their exit codes, so it
# needs both binaries and the repo's models.
forge_add_test(test_cli_validation)
target_compile_definitions(test_cli_validation PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  FORGE_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_dependencies(test_cli_validation forge_solve forge_check)

if(FORGE_ENABLE_JULIA_GPU)
  forge_add_test(test_julia_gpu_parity forge_backend)
  set_tests_properties(test_julia_gpu_parity PROPERTIES SKIP_RETURN_CODE 77)
endif()
