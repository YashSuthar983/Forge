find_package(Threads REQUIRED)
add_library(forge_core src/result.cpp src/parallel.cpp src/route_debug.cpp src/options.cpp
  src/env_switches.cpp)
add_library(forge::core ALIAS forge_core)
target_include_directories(forge_core PUBLIC include)
target_link_libraries(forge_core PUBLIC Threads::Threads PRIVATE forge_flags)
target_compile_definitions(forge_core PUBLIC
  $<$<BOOL:${FORGE_ROUTE_FN}>:FORGE_ROUTE_DEBUG=1>)
