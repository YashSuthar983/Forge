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
option(FORGE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

add_library(forge_flags INTERFACE)
target_compile_options(forge_flags INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow>
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

# Sparse LU of a simplex basis: FTRAN / BTRAN / product-form update.
# Deliberately depends on forge_core ONLY -- it takes the basis column-wise as
# plain arrays, so it needs no matrix type and stays usable by crossover and by
# any future barrier engine.
add_library(forge_la_cpu forge_la_cpu/src/lu.cpp)
target_include_directories(forge_la_cpu PUBLIC forge_la_cpu/include)
target_link_libraries(forge_la_cpu PUBLIC forge_core PRIVATE forge_flags)

set(FORGE_BACKEND_SOURCES forge_backend/src/cpu_backend.cpp
                        forge_backend/src/backend_factory.cpp)
if(FORGE_ENABLE_JULIA_GPU)
  list(APPEND FORGE_BACKEND_SOURCES forge_backend/src/julia_gpu_backend.cpp)
endif()

add_library(forge_backend ${FORGE_BACKEND_SOURCES})
target_include_directories(forge_backend PUBLIC forge_backend/include)
target_link_libraries(forge_backend PUBLIC forge_sparse PRIVATE forge_flags)
if(FORGE_ENABLE_JULIA_GPU)
  target_compile_definitions(forge_backend PUBLIC FORGE_ENABLE_JULIA_GPU=1)
  # Default sidecar location; overridable at runtime via FORGE_JULIA_PROJECT.
  target_compile_definitions(forge_backend PRIVATE
    FORGE_JULIA_PROJECT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/tools/julia_gpu\")
endif()

# ------------------------------- L2: model ---------------------------------
add_library(forge_model forge_model/src/lp.cpp)
target_include_directories(forge_model PUBLIC forge_model/include)
target_link_libraries(forge_model PUBLIC forge_sparse PRIVATE forge_flags)

add_library(forge_io forge_io/src/mps.cpp)
target_include_directories(forge_io PUBLIC forge_io/include)
target_link_libraries(forge_io PUBLIC forge_model PRIVATE forge_flags)

# ------------------------------ L4: engines --------------------------------
add_library(forge_engines forge_engines/src/pdhg.cpp forge_engines/src/simplex.cpp)
target_include_directories(forge_engines PUBLIC forge_engines/include)
target_link_libraries(forge_engines PUBLIC forge_model forge_backend forge_la_cpu
                                  PRIVATE forge_flags)

# --------------------------- L7: verification ------------------------------
# NOTE: forge_certify links ONLY L0. It must never gain an engine dependency.
add_library(forge_certify forge_certify/src/finalize.cpp)
target_include_directories(forge_certify PUBLIC forge_certify/include)
target_link_libraries(forge_certify PUBLIC forge_core PRIVATE forge_flags)

# ----------------------------- L8: front ends ------------------------------
add_executable(forge_solve cli/forge_solve.cpp)
target_link_libraries(forge_solve PRIVATE
  forge_io forge_engines forge_certify forge_backend forge_flags)

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
forge_add_test(test_pdhg                forge_engines forge_certify forge_io)
forge_add_test(test_simplex             forge_engines forge_certify forge_io)

if(FORGE_ENABLE_JULIA_GPU)
  forge_add_test(test_julia_gpu_parity forge_backend)
  # Missing julia or a non-functional GPU must SKIP, not fail.
  set_tests_properties(test_julia_gpu_parity PROPERTIES SKIP_RETURN_CODE 77)
endif()

message(STATUS "FORGE ${PROJECT_VERSION}  build=${CMAKE_BUILD_TYPE}  "
               "julia_gpu=${FORGE_ENABLE_JULIA_GPU}")
