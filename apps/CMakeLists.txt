add_executable(sor_solve sor_solve.cpp)
target_link_libraries(sor_solve PRIVATE
  sor_io sor_engines sor_search sor_certify sor_backend sor_flags)

add_executable(sor_gen sor_gen.cpp)
target_link_libraries(sor_gen PRIVATE sor_io sor_flags)

add_executable(sor_check sor_check.cpp)
target_link_libraries(sor_check PRIVATE sor_io sor_engines sor_certify sor_flags)

# Hardware characterisation of the GPU through SOR's own Vulkan plumbing.
# Uses the backend's internal vk_compute.hpp on purpose: it measures exactly
# the submit / dispatch / transfer path the QP devices pay.
if(SOR_ENABLE_VULKAN)
  add_executable(sor_gpu_bench sor_gpu_bench.cpp)
  target_include_directories(sor_gpu_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/src/vulkan)
  target_link_libraries(sor_gpu_bench PRIVATE sor_backend sor_flags)
endif()

# Developer tool: time / determinism-check the sparse LDL' on a KKT system
# dumped by the IPM (SOR_DUMP_KKT).  Not a solver front end.
add_executable(sor_ldlt_bench sor_ldlt_bench.cpp)
target_link_libraries(sor_ldlt_bench PRIVATE sor_la_cpu sor_flags)
