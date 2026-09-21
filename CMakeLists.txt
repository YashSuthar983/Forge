cmake_minimum_required(VERSION 3.20)
project(sor LANGUAGES CXX VERSION 0.0.1)

# ---------------------------------------------------------------------------
# SOR - Sovereign Optimization Runtime
#
# Layering (docs/architecture.md §2) is enforced by target_link_libraries: a
# module may only link strictly lower layers. The add_subdirectory() order
# below IS the layer order; each module owns its own CMakeLists.txt.
#
# Clean-room rule: no HiGHS / SCIP / CBC / GLPK / OR-Tools / cuOpt / PaPILO /
# Ipopt anywhere in this file or the link graph. See docs/architecture.md §8.
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(SOR_ENABLE_VULKAN
       "Build the Vulkan compute LpDevice (AMD/NVIDIA/Intel)" ON)
option(SOR_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
# -O3 belongs to the build type, not here: hard-coding it made a Debug build
# silently optimized. -march=native is off by default because it makes published
# timings unreproducible on any other machine and unfair against a competitor
# built for a generic target -- opt in with -DSOR_NATIVE_ARCH=ON when profiling.
option(SOR_NATIVE_ARCH "Build for the host CPU (-march=native); not portable" OFF)
# Floating-point contraction (a*b+c fused into one FMA) is a licence the
# compiler takes per expression, so the SAME source produces different
# roundings in two code paths that are algebraically identical. That is not a
# last-digit curiosity here: measured 2026-09-10 on Netlib-93 with
# -march=native, contraction changed the PIVOT COUNT on 56 of the 93 models
# (grow15 678 -> 928, pilot.ja 2,046 -> 1,832) because a different rounding
# picks a different ratio-test winner. It also breaks three tests outright --
# test_lu and test_lu_ft demand that the paired FTRAN kernel agree BITWISE with
# two separate solves, and test_dispatch_regressions pins which phase-1
# composite path runs.
#
# The measured cost of turning it off is a 0.4% geometric mean on that suite,
# i.e. nothing, and in exchange the pivot counts the merge gate compares
# against its committed baselines stop depending on the compiler's contraction
# choices. Reproducibility is worth more than 0.4%.
option(SOR_DETERMINISTIC_FP
       "Forbid FP contraction (-ffp-contract=off) so results are reproducible" ON)

# Keep every binary at the TOP of the build tree. With add_subdirectory(),
# CMake would otherwise put sor_solve in build/cli/ and libsor_core.a in
# build/src/core/ -- but scripts/*.py, the demo, and test_ps_must's
# SOR_BINARY_DIR all expect build/sor_solve. Pinning these keeps the split
# invisible to everything outside CMake.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(sor_flags INTERFACE)
target_compile_options(sor_flags INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${SOR_NATIVE_ARCH}>>:-march=native>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${SOR_DETERMINISTIC_FP}>>:-ffp-contract=off>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${SOR_WARNINGS_AS_ERRORS}>>:-Werror>
)


# ---------------------------------------------------------------------------
# Library modules, in layer order. A module may only link strictly lower
# layers; that invariant is what these comments document and what each
# module's own target_link_libraries() enforces.
# ---------------------------------------------------------------------------
add_subdirectory(src/core)        # L0  platform types
add_subdirectory(src/sparse)      # L1  sparse containers
add_subdirectory(src/la)          # L1  LU, FTRAN/BTRAN
add_subdirectory(src/backend)     # L1  CPU / Vulkan / Julia devices
add_subdirectory(src/model)       # L2  problem representation
add_subdirectory(src/io)          # L2  MPS / QPS / solution I/O
add_subdirectory(src/presolve)    # L3  problem transformation
add_subdirectory(src/engines)     # L4  simplex, PDHG, HPR, QP, NLP
add_subdirectory(src/search)      # L5  branch and cut
add_subdirectory(src/certify)     # L7  proof gate

# ----------------------------- front ends ----------------------------------
add_subdirectory(apps)            # L8  sor_solve, sor_gen, sor_check

# -------------------------------- tests ------------------------------------
enable_testing()
add_subdirectory(tests)

message(STATUS "SOR ${PROJECT_VERSION}  build=${CMAKE_BUILD_TYPE}  "
               "vulkan=${SOR_ENABLE_VULKAN}")
