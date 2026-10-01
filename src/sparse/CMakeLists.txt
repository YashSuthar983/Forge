add_library(forge_sparse src/csr.cpp src/spmv_plan.cpp src/csc.cpp)
add_library(forge::sparse ALIAS forge_sparse)
target_include_directories(forge_sparse PUBLIC include)
target_link_libraries(forge_sparse PUBLIC forge_core PRIVATE forge_flags)
