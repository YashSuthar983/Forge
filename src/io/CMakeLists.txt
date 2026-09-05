add_library(sor_io src/mps.cpp src/write_mps.cpp src/qps.cpp src/solution.cpp)
add_library(sor::io ALIAS sor_io)
target_include_directories(sor_io PUBLIC include)
target_link_libraries(sor_io PUBLIC sor_model PRIVATE sor_flags)
