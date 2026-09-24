add_library(sor_la_cpu src/lu.cpp src/amd.cpp src/ldlt.cpp)
add_library(sor::la ALIAS sor_la_cpu)
target_include_directories(sor_la_cpu PUBLIC include)
target_link_libraries(sor_la_cpu PUBLIC sor_core PRIVATE sor_flags)
# The LDL' kernels need a runtime alias check to vectorize, which GCC's -O2
# "very cheap" cost model refuses (measured: every inner loop of ldlt.cpp
# scalar).  Vectorizing over rows never reassociates, so results are
# unchanged; see the SOR_KERNEL comment in ldlt.cpp.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(src/ldlt.cpp PROPERTIES COMPILE_OPTIONS "-fvect-cost-model=dynamic")
endif()
