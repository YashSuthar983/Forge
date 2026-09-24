add_executable(forge_solve forge_solve.cpp)
target_link_libraries(forge_solve PRIVATE
  forge_io forge_engines forge_search forge_certify forge_backend forge_flags)

add_executable(forge_gen forge_gen.cpp)
target_link_libraries(forge_gen PRIVATE forge_io forge_flags)

add_executable(forge_check forge_check.cpp)
target_link_libraries(forge_check PRIVATE forge_io forge_engines forge_certify forge_flags)

# Hardware characterisation of the GPU through FORGE's own Vulkan plumbing.
# Uses the backend's internal vk_compute.hpp on purpose: it measures exactly
# the submit / dispatch / transfer path the QP devices pay.
if(FORGE_ENABLE_VULKAN)
  add_executable(forge_gpu_bench forge_gpu_bench.cpp)
  target_include_directories(forge_gpu_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/src/vulkan)
  target_link_libraries(forge_gpu_bench PRIVATE forge_backend forge_flags)
endif()

# Developer tool: time / determinism-check the sparse LDL' on a KKT system
# dumped by the IPM (FORGE_DUMP_KKT).  Not a solver front end.
add_executable(forge_ldlt_bench forge_ldlt_bench.cpp)
target_link_libraries(forge_ldlt_bench PRIVATE forge_la_cpu forge_flags)
