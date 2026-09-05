function(forge_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE ${ARGN} forge_flags)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

forge_add_test(test_csr                 forge_sparse)
forge_add_test(test_csc                 forge_sparse)
forge_add_test(test_lu                  forge_la_cpu)
forge_add_test(test_backend_parity      forge_backend)
forge_add_test(test_no_unproved_optimal forge_certify)
forge_add_test(test_mps                 forge_io)
forge_add_test(test_solution_io         forge_io)
forge_add_test(test_pdhg                forge_engines forge_certify forge_io)
forge_add_test(test_simplex             forge_engines forge_certify forge_io)
forge_add_test(test_dual_edge_weights   forge_engines)
forge_add_test(test_presolve            forge_presolve)
forge_add_test(test_hpr                 forge_engines forge_certify forge_io)
forge_add_test(test_lp_device           forge_backend forge_engines forge_io)
forge_add_test(test_milp                forge_search forge_certify forge_io)
forge_add_test(test_lattice_reform      forge_search forge_certify forge_io)
forge_add_test(test_cuts                forge_search forge_engines forge_certify forge_io)
forge_add_test(test_propagate           forge_search)
forge_add_test(test_qp                  forge_engines forge_certify)
forge_add_test(test_qps                 forge_io forge_engines forge_certify)
forge_add_test(test_ps_must             forge_search forge_engines forge_certify forge_io
                                      forge_la_cpu)
forge_add_test(test_extensions          forge_search forge_engines forge_certify)

# CMAKE_SOURCE_DIR / CMAKE_BINARY_DIR (not CURRENT): this test shells out to the
# built CLIs and reads instances from the repo root.
target_compile_definitions(test_ps_must PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  FORGE_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_dependencies(test_ps_must forge_solve forge_check forge_ext_demo)

if(FORGE_ENABLE_JULIA_GPU)
  forge_add_test(test_julia_gpu_parity forge_backend)
  set_tests_properties(test_julia_gpu_parity PROPERTIES SKIP_RETURN_CODE 77)
endif()
