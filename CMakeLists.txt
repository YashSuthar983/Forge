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
option(SOR_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

add_library(sor_flags INTERFACE)
target_compile_options(sor_flags INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow>
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

# Sparse LU of a simplex basis: FTRAN / BTRAN / product-form update.
# Deliberately depends on sor_core ONLY -- it takes the basis column-wise as
# plain arrays, so it needs no matrix type and stays usable by crossover and by
# any future barrier engine.
add_library(sor_la_cpu sor_la_cpu/src/lu.cpp)
target_include_directories(sor_la_cpu PUBLIC sor_la_cpu/include)
target_link_libraries(sor_la_cpu PUBLIC sor_core PRIVATE sor_flags)

set(SOR_BACKEND_SOURCES sor_backend/src/cpu_backend.cpp
                        sor_backend/src/backend_factory.cpp)
if(SOR_ENABLE_JULIA_GPU)
  list(APPEND SOR_BACKEND_SOURCES sor_backend/src/julia_gpu_backend.cpp)
endif()

add_library(sor_backend ${SOR_BACKEND_SOURCES})
target_include_directories(sor_backend PUBLIC sor_backend/include)
target_link_libraries(sor_backend PUBLIC sor_sparse PRIVATE sor_flags)
if(SOR_ENABLE_JULIA_GPU)
  target_compile_definitions(sor_backend PUBLIC SOR_ENABLE_JULIA_GPU=1)
  # Default sidecar location; overridable at runtime via SOR_JULIA_PROJECT.
  target_compile_definitions(sor_backend PRIVATE
    SOR_JULIA_PROJECT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/tools/julia_gpu\")
endif()

# ------------------------------- L2: model ---------------------------------
add_library(sor_model sor_model/src/lp.cpp)
target_include_directories(sor_model PUBLIC sor_model/include)
target_link_libraries(sor_model PUBLIC sor_sparse PRIVATE sor_flags)

add_library(sor_io sor_io/src/mps.cpp)
target_include_directories(sor_io PUBLIC sor_io/include)
target_link_libraries(sor_io PUBLIC sor_model PRIVATE sor_flags)

# ------------------------------ L4: engines --------------------------------
add_library(sor_engines sor_engines/src/pdhg.cpp sor_engines/src/simplex.cpp)
target_include_directories(sor_engines PUBLIC sor_engines/include)
target_link_libraries(sor_engines PUBLIC sor_model sor_backend sor_la_cpu
                                  PRIVATE sor_flags)

# --------------------------- L7: verification ------------------------------
# NOTE: sor_certify links ONLY L0. It must never gain an engine dependency.
add_library(sor_certify sor_certify/src/finalize.cpp)
target_include_directories(sor_certify PUBLIC sor_certify/include)
target_link_libraries(sor_certify PUBLIC sor_core PRIVATE sor_flags)

# ----------------------------- L8: front ends ------------------------------
add_executable(sor_solve cli/sor_solve.cpp)
target_link_libraries(sor_solve PRIVATE
  sor_io sor_engines sor_certify sor_backend sor_flags)

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
sor_add_test(test_pdhg                sor_engines sor_certify sor_io)
sor_add_test(test_simplex             sor_engines sor_certify sor_io)

if(SOR_ENABLE_JULIA_GPU)
  sor_add_test(test_julia_gpu_parity sor_backend)
  # Missing julia or a non-functional GPU must SKIP, not fail.
  set_tests_properties(test_julia_gpu_parity PROPERTIES SKIP_RETURN_CODE 77)
endif()

message(STATUS "SOR ${PROJECT_VERSION}  build=${CMAKE_BUILD_TYPE}  "
               "julia_gpu=${SOR_ENABLE_JULIA_GPU}")
