cmake_minimum_required(VERSION 3.20)
project(forge LANGUAGES CXX VERSION 0.0.1)

# ---------------------------------------------------------------------------
# FORGE — Forge Optimization Runtime
#
# Layering (docs/architecture.md §2) is enforced by target_link_libraries:
# a module may only link strictly lower layers. forge_verify may link nothing
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

option(FORGE_ENABLE_JULIA_GPU
       "Build the EXPERIMENTAL Julia GPU sidecar backend (prototype only)" OFF)
option(FORGE_ENABLE_VULKAN
       "Build the Vulkan compute LpDevice (AMD/NVIDIA/Intel)" ON)
option(FORGE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

add_library(forge_flags INTERFACE)
# -O3 belongs to the build type, not here: hard-coding it made a Debug build
# silently optimized. -march=native is off by default because it makes published
# timings unreproducible on any other machine and unfair against a competitor
# a generic target -- opt in with -DFORGE_NATIVE_ARCH=ON when profiling.
option(FORGE_NATIVE_ARCH "Build for the host CPU (-march=native); not portable" OFF)
target_compile_options(forge_flags INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${FORGE_NATIVE_ARCH}>>:-march=native>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${FORGE_WARNINGS_AS_ERRORS}>>:-Werror>
)

# ------------------------------ L0: platform -------------------------------
add_library(forge_core forge_core/src/result.cpp)
target_include_directories(forge_core PUBLIC forge_core/include)
target_link_libraries(forge_core PRIVATE forge_flags)

# --------------------------- L1: linear algebra ----------------------------
add_library(forge_sparse forge_sparse/src/csr.cpp forge_sparse/src/csc.cpp)
target_include_directories(forge_sparse PUBLIC forge_sparse/include)
target_link_libraries(forge_sparse PUBLIC forge_core PRIVATE forge_flags)

add_library(forge_la_cpu forge_la_cpu/src/lu.cpp)
target_include_directories(forge_la_cpu PUBLIC forge_la_cpu/include)
target_link_libraries(forge_la_cpu PUBLIC forge_core PRIVATE forge_flags)

# ---- Vulkan SPIR-V shaders (optional) ------------------------------------
set(FORGE_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/forge_backend/vulkan/shaders)
set(FORGE_SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
set(FORGE_SHADER_NAMES
  spmv_csr spmv_csc primal_step dual_step halpern_mix avg_update)

if(FORGE_ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
  file(MAKE_DIRECTORY ${FORGE_SHADER_BINARY_DIR})
  set(FORGE_SPV_OUTPUTS)
  foreach(s ${FORGE_SHADER_NAMES})
    set(_src ${FORGE_SHADER_SOURCE_DIR}/${s}.comp)
    set(_spv ${FORGE_SHADER_BINARY_DIR}/${s}.spv)
    add_custom_command(
      OUTPUT ${_spv}
      COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.2 -o ${_spv} ${_src}
      DEPENDS ${_src}
      COMMENT "SPIR-V ${s}.comp"
      VERBATIM)
    list(APPEND FORGE_SPV_OUTPUTS ${_spv})
  endforeach()
  add_custom_target(forge_shaders ALL DEPENDS ${FORGE_SPV_OUTPUTS})
endif()

set(FORGE_BACKEND_SOURCES
  forge_backend/src/cpu_backend.cpp
  forge_backend/src/backend_factory.cpp
  forge_backend/src/cpu_lp_device.cpp)
if(FORGE_ENABLE_VULKAN)
  list(APPEND FORGE_BACKEND_SOURCES
    forge_backend/src/vulkan/vk_context.cpp
    forge_backend/src/vulkan/vk_lp_device.cpp)
else()
  list(APPEND FORGE_BACKEND_SOURCES forge_backend/src/vulkan_stub.cpp)
endif()
if(FORGE_ENABLE_JULIA_GPU)
  list(APPEND FORGE_BACKEND_SOURCES forge_backend/src/julia_gpu_backend.cpp)
endif()

add_library(forge_backend ${FORGE_BACKEND_SOURCES})
target_include_directories(forge_backend PUBLIC forge_backend/include)
target_link_libraries(forge_backend PUBLIC forge_sparse PRIVATE forge_flags)
if(FORGE_ENABLE_VULKAN)
  target_link_libraries(forge_backend PUBLIC Vulkan::Vulkan)
  target_compile_definitions(forge_backend PUBLIC FORGE_ENABLE_VULKAN=1)
  target_compile_definitions(forge_backend PRIVATE
    FORGE_SHADER_DIR=\"${FORGE_SHADER_BINARY_DIR}\")
  add_dependencies(forge_backend forge_shaders)
endif()
if(FORGE_ENABLE_JULIA_GPU)
  target_compile_definitions(forge_backend PUBLIC FORGE_ENABLE_JULIA_GPU=1)
  target_compile_definitions(forge_backend PRIVATE
    FORGE_JULIA_PROJECT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/tools/julia_gpu\")
endif()

# ------------------------------- L2: model ---------------------------------
add_library(forge_model forge_model/src/lp.cpp)
target_include_directories(forge_model PUBLIC forge_model/include)
target_link_libraries(forge_model PUBLIC forge_sparse PRIVATE forge_flags)

add_library(forge_io forge_io/src/mps.cpp forge_io/src/write_mps.cpp forge_io/src/qps.cpp
                   forge_io/src/solution.cpp)
target_include_directories(forge_io PUBLIC forge_io/include)
target_link_libraries(forge_io PUBLIC forge_model PRIVATE forge_flags)

# ------------------------------ L3: transform ------------------------------
add_library(forge_presolve forge_presolve/src/presolve.cpp)
target_include_directories(forge_presolve PUBLIC forge_presolve/include)
target_link_libraries(forge_presolve PUBLIC forge_model PRIVATE forge_flags)

# ------------------------------ L4: engines --------------------------------
add_library(forge_engines
  forge_engines/src/pdhg.cpp
  forge_engines/src/simplex.cpp
  forge_engines/src/dual_simplex.cpp
  forge_engines/src/dual_edge_weights.cpp
  forge_engines/src/dual_bfrt.cpp
  forge_engines/src/farkas.cpp
  forge_engines/src/hpr.cpp
  forge_engines/src/qp.cpp
  forge_engines/src/qp_pdhcg.cpp
  forge_engines/src/nlp.cpp)
target_include_directories(forge_engines PUBLIC forge_engines/include)
target_link_libraries(forge_engines PUBLIC forge_model forge_backend forge_la_cpu forge_presolve
                                  PRIVATE forge_flags)

# ------------------------------ L5: search ---------------------------------
add_library(forge_search forge_search/src/bab.cpp forge_search/src/cuts.cpp
                       forge_search/src/propagate.cpp forge_search/src/miqp.cpp
                       forge_search/src/minlp.cpp
                       forge_search/src/lattice_reform.cpp)
target_include_directories(forge_search PUBLIC forge_search/include)
target_link_libraries(forge_search PUBLIC forge_engines PRIVATE forge_flags)

# --------------------------- L7: verification ------------------------------
add_library(forge_certify forge_certify/src/finalize.cpp)
target_include_directories(forge_certify PUBLIC forge_certify/include)
target_link_libraries(forge_certify PUBLIC forge_core PRIVATE forge_flags)

# ----------------------------- L8: front ends ------------------------------
add_executable(forge_solve cli/forge_solve.cpp)
target_link_libraries(forge_solve PRIVATE
  forge_io forge_engines forge_search forge_certify forge_backend forge_flags)

add_executable(forge_gen cli/forge_gen.cpp)
target_link_libraries(forge_gen PRIVATE forge_io forge_flags)

add_executable(forge_check cli/forge_check.cpp)
target_link_libraries(forge_check PRIVATE forge_io forge_engines forge_flags)

add_executable(forge_ext_demo cli/forge_ext_demo.cpp)
target_link_libraries(forge_ext_demo PRIVATE forge_search forge_certify forge_flags)

# --------------------------------- tests -----------------------------------
enable_testing()

function(forge_add_test name)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE tests)
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
forge_add_test(test_presolve             forge_presolve)
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
target_compile_definitions(test_ps_must PRIVATE
  FORGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  FORGE_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(test_ps_must forge_solve forge_check forge_ext_demo)

if(FORGE_ENABLE_JULIA_GPU)
  forge_add_test(test_julia_gpu_parity forge_backend)
  set_tests_properties(test_julia_gpu_parity PROPERTIES SKIP_RETURN_CODE 77)
endif()

message(STATUS "FORGE ${PROJECT_VERSION}  build=${CMAKE_BUILD_TYPE}  "
               "julia_gpu=${FORGE_ENABLE_JULIA_GPU}  vulkan=${FORGE_ENABLE_VULKAN}")
