cmake_minimum_required(VERSION 3.20)
project(sor_highs_source_runner LANGUAGES CXX)

# This is deliberately a separate build.  SOR never links HiGHS; the binary is
# an external benchmark oracle whose build and runtime identity are recorded.
set(HIGHS_SOURCE_DIR "" CACHE PATH "Pinned HiGHS source checkout")
set(HIGHS_BUILD_DIR "" CACHE PATH "Matching built HiGHS tree")
set(HIGHS_EXPECTED_COMMIT "" CACHE STRING "Full pinned HiGHS git commit")

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "source runner requires -DCMAKE_BUILD_TYPE=Release")
endif()
if(NOT IS_DIRECTORY "${HIGHS_SOURCE_DIR}/.git")
  message(FATAL_ERROR "HIGHS_SOURCE_DIR must be a HiGHS git checkout")
endif()
if(NOT EXISTS "${HIGHS_BUILD_DIR}/CMakeCache.txt")
  message(FATAL_ERROR "HIGHS_BUILD_DIR must name the matching configured build")
endif()
string(LENGTH "${HIGHS_EXPECTED_COMMIT}" HIGHS_EXPECTED_COMMIT_LENGTH)
if(NOT HIGHS_EXPECTED_COMMIT_LENGTH EQUAL 40 OR
   NOT HIGHS_EXPECTED_COMMIT MATCHES "^[0-9a-fA-F]+$")
  message(FATAL_ERROR "HIGHS_EXPECTED_COMMIT must be the full pinned commit")
endif()

execute_process(
  COMMAND git -C "${HIGHS_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE HIGHS_SOURCE_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE HIGHS_GIT_RC)
execute_process(
  COMMAND git -C "${HIGHS_SOURCE_DIR}" status --porcelain
  OUTPUT_VARIABLE HIGHS_SOURCE_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE HIGHS_STATUS_RC)
if(NOT HIGHS_GIT_RC EQUAL 0 OR NOT HIGHS_STATUS_RC EQUAL 0)
  message(FATAL_ERROR "cannot inspect pinned HiGHS checkout")
endif()
if(NOT HIGHS_SOURCE_COMMIT STREQUAL HIGHS_EXPECTED_COMMIT)
  message(FATAL_ERROR "HiGHS checkout ${HIGHS_SOURCE_COMMIT} is not pinned commit ${HIGHS_EXPECTED_COMMIT}")
endif()
if(NOT HIGHS_SOURCE_STATUS STREQUAL "")
  message(FATAL_ERROR "pinned HiGHS checkout is dirty")
endif()

file(STRINGS "${HIGHS_BUILD_DIR}/CMakeCache.txt" HIGHS_CACHE_LINES)
function(cache_value KEY OUTPUT)
  set(VALUE "")
  foreach(LINE IN LISTS HIGHS_CACHE_LINES)
    if(LINE MATCHES "^${KEY}:[^=]*=(.*)$")
      set(VALUE "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  set(${OUTPUT} "${VALUE}" PARENT_SCOPE)
endfunction()
cache_value("CMAKE_BUILD_TYPE" HIGHS_BUILD_TYPE)
cache_value("CMAKE_CXX_COMPILER" HIGHS_CXX_COMPILER)
cache_value("CMAKE_CXX_FLAGS" HIGHS_CXX_FLAGS)
cache_value("CMAKE_CXX_FLAGS_RELEASE" HIGHS_RELEASE_FLAGS)
cache_value("CMAKE_HOME_DIRECTORY" HIGHS_CONFIGURED_SOURCE)
if(NOT HIGHS_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "pinned HiGHS build is not Release")
endif()
cmake_path(ABSOLUTE_PATH HIGHS_CXX_COMPILER NORMALIZE OUTPUT_VARIABLE HIGHS_CXX_REAL)
cmake_path(ABSOLUTE_PATH CMAKE_CXX_COMPILER NORMALIZE OUTPUT_VARIABLE RUNNER_CXX_REAL)
if(NOT HIGHS_CXX_REAL STREQUAL RUNNER_CXX_REAL)
  message(FATAL_ERROR "HiGHS and source runner compiler mismatch")
endif()
cmake_path(ABSOLUTE_PATH HIGHS_CONFIGURED_SOURCE NORMALIZE OUTPUT_VARIABLE HIGHS_HOME_REAL)
cmake_path(ABSOLUTE_PATH HIGHS_SOURCE_DIR NORMALIZE OUTPUT_VARIABLE HIGHS_SOURCE_REAL)
if(NOT HIGHS_HOME_REAL STREQUAL HIGHS_SOURCE_REAL)
  message(FATAL_ERROR "HIGHS_BUILD_DIR was configured from another source tree")
endif()
set(HIGHS_ALL_FLAGS "${HIGHS_CXX_FLAGS} ${HIGHS_RELEASE_FLAGS}")
foreach(REQUIRED_FLAG IN ITEMS "-O3" "NDEBUG" "-march=native")
  string(FIND "${HIGHS_ALL_FLAGS}" "${REQUIRED_FLAG}" FLAG_AT)
  if(FLAG_AT EQUAL -1)
    message(FATAL_ERROR "pinned HiGHS build flags lack ${REQUIRED_FLAG}")
  endif()
endforeach()

find_package(highs CONFIG REQUIRED NO_DEFAULT_PATH
  PATHS "${HIGHS_BUILD_DIR}" "${HIGHS_BUILD_DIR}/lib/cmake/highs")
if(TARGET highs::highs)
  set(HIGHS_TARGET highs::highs)
elseif(TARGET highs)
  set(HIGHS_TARGET highs)
else()
  message(FATAL_ERROR "HiGHS package exports neither highs::highs nor highs")
endif()

add_executable(sor_highs_source_runner highs_source_runner.cpp)
target_compile_features(sor_highs_source_runner PRIVATE cxx_std_17)
target_compile_options(sor_highs_source_runner PRIVATE -O3 -march=native)
target_include_directories(sor_highs_source_runner PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party")
target_compile_definitions(sor_highs_source_runner PRIVATE
  NDEBUG
  SOR_HIGHS_SOURCE_COMMIT="${HIGHS_SOURCE_COMMIT}"
  SOR_HIGHS_SOURCE_DIR="${HIGHS_SOURCE_REAL}"
  SOR_HIGHS_BUILD_DIR="${HIGHS_BUILD_DIR}"
  SOR_RUNNER_COMPILER="${RUNNER_CXX_REAL}"
  SOR_RUNNER_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  SOR_RUNNER_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}")
target_link_libraries(sor_highs_source_runner PRIVATE ${HIGHS_TARGET})
