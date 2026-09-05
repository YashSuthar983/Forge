add_library(sor_la_cpu src/lu.cpp)
add_library(sor::la ALIAS sor_la_cpu)
target_include_directories(sor_la_cpu PUBLIC include)
target_link_libraries(sor_la_cpu PUBLIC sor_core PRIVATE sor_flags)
