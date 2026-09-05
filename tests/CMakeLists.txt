function(sor_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE ${ARGN} sor_flags)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

sor_add_test(test_csr                 sor_sparse)
sor_add_test(test_csc                 sor_sparse)
sor_add_test(test_lu                  sor_la_cpu)
sor_add_test(test_backend_parity      sor_backend)
sor_add_test(test_no_unproved_optimal sor_certify)
sor_add_test(test_mps                 sor_io)
sor_add_test(test_solution_io         sor_io)
sor_add_test(test_pdhg                sor_engines sor_certify sor_io)
sor_add_test(test_simplex             sor_engines sor_certify sor_io)
sor_add_test(test_dual_edge_weights   sor_engines)
sor_add_test(test_presolve            sor_presolve)
sor_add_test(test_hpr                 sor_engines sor_certify sor_io)
sor_add_test(test_lp_device           sor_backend sor_engines sor_io)
sor_add_test(test_milp                sor_search sor_certify sor_io)
sor_add_test(test_lattice_reform      sor_search sor_certify sor_io)
sor_add_test(test_cuts                sor_search sor_engines sor_certify sor_io)
sor_add_test(test_propagate           sor_search)
sor_add_test(test_qp                  sor_engines sor_certify)
sor_add_test(test_qps                 sor_io sor_engines sor_certify)
sor_add_test(test_ps_must             sor_search sor_engines sor_certify sor_io
                                      sor_la_cpu)
sor_add_test(test_extensions          sor_search sor_engines sor_certify)

# CMAKE_SOURCE_DIR / CMAKE_BINARY_DIR (not CURRENT): this test shells out to the
# built CLIs and reads instances from the repo root.
target_compile_definitions(test_ps_must PRIVATE
  SOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  SOR_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_dependencies(test_ps_must sor_solve sor_check sor_ext_demo)

if(SOR_ENABLE_JULIA_GPU)
  sor_add_test(test_julia_gpu_parity sor_backend)
  set_tests_properties(test_julia_gpu_parity PROPERTIES SKIP_RETURN_CODE 77)
endif()
