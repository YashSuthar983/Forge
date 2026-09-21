add_library(sor_core src/result.cpp src/route_debug.cpp)
add_library(sor::core ALIAS sor_core)
target_include_directories(sor_core PUBLIC include)
target_link_libraries(sor_core PRIVATE sor_flags)
