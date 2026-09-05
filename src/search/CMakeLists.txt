add_library(sor_search
  src/bab.cpp src/cuts.cpp src/propagate.cpp
  src/miqp.cpp src/minlp.cpp src/lattice_reform.cpp)
add_library(sor::search ALIAS sor_search)
target_include_directories(sor_search PUBLIC include)
target_link_libraries(sor_search PUBLIC sor_engines PRIVATE sor_flags)
