add_library(sor_sparse src/csr.cpp src/csc.cpp)
add_library(sor::sparse ALIAS sor_sparse)
target_include_directories(sor_sparse PUBLIC include)
target_link_libraries(sor_sparse PUBLIC sor_core PRIVATE sor_flags)
