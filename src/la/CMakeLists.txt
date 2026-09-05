add_library(forge_la_cpu src/lu.cpp)
add_library(forge::la ALIAS forge_la_cpu)
target_include_directories(forge_la_cpu PUBLIC include)
target_link_libraries(forge_la_cpu PUBLIC forge_core PRIVATE forge_flags)
