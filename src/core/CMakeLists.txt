add_library(forge_core src/result.cpp src/route_debug.cpp)
add_library(forge::core ALIAS forge_core)
target_include_directories(forge_core PUBLIC include)
target_link_libraries(forge_core PRIVATE forge_flags)
