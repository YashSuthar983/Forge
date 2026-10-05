add_library(forge_la_cpu src/lu.cpp src/basis_numerics.cpp src/dense_lu.cpp src/amd.cpp src/ldlt.cpp)
add_library(forge::la ALIAS forge_la_cpu)
target_include_directories(forge_la_cpu PUBLIC include)
target_link_libraries(forge_la_cpu PUBLIC forge_core PRIVATE forge_flags)
# The LDL' kernels need a runtime alias check to vectorize, which GCC's -O2
# "very cheap" cost model refuses (measured: every inner loop of ldlt.cpp
# scalar).  Vectorizing over rows never reassociates, so results are
# unchanged; see the FORGE_KERNEL comment in ldlt.cpp.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(src/ldlt.cpp PROPERTIES COMPILE_OPTIONS "-fvect-cost-model=dynamic")
endif()
