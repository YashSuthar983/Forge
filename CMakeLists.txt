cmake_minimum_required(VERSION 3.20)
project(sor LANGUAGES CXX VERSION 0.0.1)

# ---------------------------------------------------------------------------
# SOR — Sovereign Optimization Runtime
#
# Layering (docs/architecture.md §2) is enforced by target_link_libraries:
# a module may only link strictly lower layers. sor_verify may link nothing
# above L2. scripts/check_layering.py re-checks the include graph.
#
# Clean-room rule: no HiGHS / SCIP / CBC / GLPK / OR-Tools / cuOpt / PaPILO /
# Ipopt anywhere in this file or the link graph. See docs/clean_room_policy.md.
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(SOR_ENABLE_JULIA_GPU
       "Build the EXPERIMENTAL Julia GPU sidecar backend (prototype only)" OFF)
option(SOR_ENABLE_VULKAN
       "Build the Vulkan compute LpDevice (AMD/NVIDIA/Intel)" ON)
option(SOR_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

add_library(sor_flags INTERFACE)
# -O3 belongs to the build type, not here: hard-coding it made a Debug build
# silently optimized. -march=native is off by default because it makes published
# timings unreproducible on any other machine and unfair against a competitor
# built for a generic target -- opt in with -DSOR_NATIVE_ARCH=ON when profiling.
option(SOR_NATIVE_ARCH "Build for the host CPU (-march=native); not portable" OFF)
target_compile_options(sor_flags INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${SOR_NATIVE_ARCH}>>:-march=native>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${SOR_WARNINGS_AS_ERRORS}>>:-Werror>
)

# ------------------------------ L0: platform -------------------------------
add_library(sor_core sor_core/src/result.cpp)
target_include_directories(sor_core PUBLIC sor_core/include)
target_link_libraries(sor_core PRIVATE sor_flags)

# --------------------------- L1: linear algebra ----------------------------
add_library(sor_sparse sor_sparse/src/csr.cpp sor_sparse/src/csc.cpp)
target_include_directories(sor_sparse PUBLIC sor_sparse/include)
target_link_libraries(sor_sparse PUBLIC sor_core PRIVATE sor_flags)

add_library(sor_la_cpu sor_la_cpu/src/lu.cpp)
target_include_directories(sor_la_cpu PUBLIC sor_la_cpu/include)
target_link_libraries(sor_la_cpu PUBLIC sor_core PRIVATE sor_flags)

# ---- Vulkan SPIR-V shaders (optional) ------------------------------------
set(SOR_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sor_backend/vulkan/shaders)
set(SOR_SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
set(SOR_SHADER_NAMES
  spmv_csr spmv_csc primal_step dual_step halpern_mix avg_update)

if(SOR_ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
  file(MAKE_DIRECTORY ${SOR_SHADER_BINARY_DIR})
  set(SOR_SPV_OUTPUTS)
  foreach(s ${SOR_SHADER_NAMES})
    set(_src ${SOR_SHADER_SOURCE_DIR}/${s}.comp)
    set(_spv ${SOR_SHADER_BINARY_DIR}/${s}.spv)
    add_custom_command(
      OUTPUT ${_spv}
      COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.2 -o ${_spv} ${_src}
      DEPENDS ${_src}
      COMMENT "SPIR-V ${s}.comp"
      VERBATIM)
    list(APPEND SOR_SPV_OUTPUTS ${_spv})
  endforeach()
  add_custom_target(sor_shaders ALL DEPENDS ${SOR_SPV_OUTPUTS})
endif()

set(SOR_BACKEND_SOURCES
  sor_backend/src/cpu_backend.cpp
  sor_backend/src/backend_factory.cpp
  sor_backend/src/cpu_lp_device.cpp)
if(SOR_ENABLE_VULKAN)
  list(APPEND SOR_BACKEND_SOURCES
    sor_backend/src/vulkan/vk_context.cpp
    sor_backend/src/vulkan/vk_lp_device.cpp)
else()
  list(APPEND SOR_BACKEND_SOURCES sor_backend/src/vulkan_stub.cpp)
endif()
if(SOR_ENABLE_JULIA_GPU)
  list(APPEND SOR_BACKEND_SOURCES sor_backend/src/julia_gpu_backend.cpp)
endif()

add_library(sor_backend ${SOR_BACKEND_SOURCES})
target_include_directories(sor_backend PUBLIC sor_backend/include)
target_link_libraries(sor_backend PUBLIC sor_sparse PRIVATE sor_flags)
if(SOR_ENABLE_VULKAN)
  target_link_libraries(sor_backend PUBLIC Vulkan::Vulkan)
  target_compile_definitions(sor_backend PUBLIC SOR_ENABLE_VULKAN=1)
  target_compile_definitions(sor_backend PRIVATE
    SOR_SHADER_DIR=\"${SOR_SHADER_BINARY_DIR}\")
  add_dependencies(sor_backend sor_shaders)
endif()
if(SOR_ENABLE_JULIA_GPU)
  target_compile_definitions(sor_backend PUBLIC SOR_ENABLE_JULIA_GPU=1)
  target_compile_definitions(sor_backend PRIVATE
    SOR_JULIA_PROJECT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/tools/julia_gpu\")
endif()

# ------------------------------- L2: model ---------------------------------
add_library(sor_model sor_model/src/lp.cpp)
target_include_directories(sor_model PUBLIC sor_model/include)
target_link_libraries(sor_model PUBLIC sor_sparse PRIVATE sor_flags)

add_library(sor_io sor_io/src/mps.cpp sor_io/src/write_mps.cpp sor_io/src/qps.cpp
                   sor_io/src/solution.cpp)
target_include_directories(sor_io PUBLIC sor_io/include)
target_link_libraries(sor_io PUBLIC sor_model PRIVATE sor_flags)

# ------------------------------ L3: transform ------------------------------
add_library(sor_presolve sor_presolve/src/presolve.cpp)
target_include_directories(sor_presolve PUBLIC sor_presolve/include)
target_link_libraries(sor_presolve PUBLIC sor_model PRIVATE sor_flags)

# ------------------------------ L4: engines --------------------------------
add_library(sor_engines
  sor_engines/src/pdhg.cpp
  sor_engines/src/simplex.cpp
  sor_engines/src/dual_simplex.cpp
  sor_engines/src/dual_edge_weights.cpp
  sor_engines/src/dual_bfrt.cpp
  sor_engines/src/farkas.cpp
  sor_engines/src/hpr.cpp
  sor_engines/src/qp.cpp
  sor_engines/src/qp_pdhcg.cpp
  sor_engines/src/nlp.cpp)
target_include_directories(sor_engines PUBLIC sor_engines/include)
target_link_libraries(sor_engines PUBLIC sor_model sor_backend sor_la_cpu sor_presolve
                                  PRIVATE sor_flags)

# ------------------------------ L5: search ---------------------------------
add_library(sor_search sor_search/src/bab.cpp sor_search/src/cuts.cpp
                       sor_search/src/propagate.cpp sor_search/src/miqp.cpp
                       sor_search/src/minlp.cpp
                       sor_search/src/lattice_reform.cpp)
target_include_directories(sor_search PUBLIC sor_search/include)
target_link_libraries(sor_search PUBLIC sor_engines PRIVATE sor_flags)

# --------------------------- L7: verification ------------------------------
add_library(sor_certify sor_certify/src/finalize.cpp)
target_include_directories(sor_certify PUBLIC sor_certify/include)
target_link_libraries(sor_certify PUBLIC sor_core PRIVATE sor_flags)

# ----------------------------- L8: front ends ------------------------------
add_executable(sor_solve cli/sor_solve.cpp)
target_link_libraries(sor_solve PRIVATE
  sor_io sor_engines sor_search sor_certify sor_backend sor_flags)

add_executable(sor_gen cli/sor_gen.cpp)
target_link_libraries(sor_gen PRIVATE sor_io sor_flags)

add_executable(sor_check cli/sor_check.cpp)
target_link_libraries(sor_check PRIVATE sor_io sor_engines sor_flags)

add_executable(sor_ext_demo cli/sor_ext_demo.cpp)
target_link_libraries(sor_ext_demo PRIVATE sor_search sor_certify sor_flags)

# --------------------------------- tests -----------------------------------
enable_testing()

function(sor_add_test name)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE tests)
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
sor_add_test(test_presolve             sor_presolve)
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
target_compile_definitions(test_ps_must PRIVATE
  SOR_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  SOR_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(test_ps_must sor_solve sor_check sor_ext_demo)

if(SOR_ENABLE_JULIA_GPU)
  sor_add_test(test_julia_gpu_parity sor_backend)
  set_tests_properties(test_julia_gpu_parity PROPERTIES SKIP_RETURN_CODE 77)
endif()

message(STATUS "SOR ${PROJECT_VERSION}  build=${CMAKE_BUILD_TYPE}  "
               "julia_gpu=${SOR_ENABLE_JULIA_GPU}  vulkan=${SOR_ENABLE_VULKAN}")
