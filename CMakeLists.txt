cmake_minimum_required(VERSION 3.20)
project(forge LANGUAGES CXX VERSION 0.0.1)

# ---------------------------------------------------------------------------
# FORGE — Forge Optimization Runtime
#
# Layering (docs/architecture.md §2) is enforced by target_link_libraries: a
# module may only link strictly lower layers. The add_subdirectory() order
# below IS the layer order; each module owns its own CMakeLists.txt.
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
# -O3 belongs to the build type, not here: hard-coding it made a Debug build
# silently optimized. -march=native is off by default because it makes published
# timings unreproducible on any other machine and unfair against a competitor
# a generic target -- opt in with -DFORGE_NATIVE_ARCH=ON when profiling.
option(FORGE_NATIVE_ARCH "Build for the host CPU (-march=native); not portable" OFF)

# Keep every binary at the TOP of the build tree. With add_subdirectory(),
# CMake would otherwise put forge_solve in build/cli/ and libforge_core.a in
# build/src/core/ -- but scripts/*.py, the demo, and test_ps_must's
# FORGE_BINARY_DIR all expect build/forge_solve. Pinning these keeps the split
# invisible to everything outside CMake.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(forge_flags INTERFACE)
target_compile_options(forge_flags INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${FORGE_NATIVE_ARCH}>>:-march=native>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${FORGE_WARNINGS_AS_ERRORS}>>:-Werror>
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
add_subdirectory(apps)            # L8  forge_solve, forge_gen, forge_check

# -------------------------------- tests ------------------------------------
enable_testing()
add_subdirectory(tests)

message(STATUS "FORGE ${PROJECT_VERSION}  build=${CMAKE_BUILD_TYPE}  "
               "julia_gpu=${FORGE_ENABLE_JULIA_GPU}  vulkan=${FORGE_ENABLE_VULKAN}")
