find_package(Threads REQUIRED)
add_library(sor_core src/result.cpp src/parallel.cpp src/route_debug.cpp src/options.cpp)
add_library(sor::core ALIAS sor_core)
target_include_directories(sor_core PUBLIC include)
target_link_libraries(sor_core PUBLIC Threads::Threads PRIVATE sor_flags)
target_compile_definitions(sor_core PUBLIC
  $<$<BOOL:${SOR_ROUTE_FN}>:SOR_ROUTE_DEBUG=1>)
