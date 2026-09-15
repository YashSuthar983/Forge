find_package(ZLIB REQUIRED)

add_library(forge_io
  src/mps.cpp src/write_mps.cpp src/qps.cpp src/solution.cpp src/gzip.cpp)
add_library(forge::io ALIAS forge_io)
target_include_directories(forge_io PUBLIC include)
target_link_libraries(forge_io PUBLIC forge_model PRIVATE forge_flags ZLIB::ZLIB)
