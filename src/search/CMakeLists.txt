add_library(forge_search
  src/bab.cpp src/cuts.cpp src/propagate.cpp
  src/miqp.cpp src/minlp.cpp src/lattice_reform.cpp)
add_library(forge::search ALIAS forge_search)
target_include_directories(forge_search PUBLIC include)
target_link_libraries(forge_search PUBLIC forge_engines PRIVATE forge_flags)
