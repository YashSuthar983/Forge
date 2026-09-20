add_library(sor_search
  src/binquad.cpp src/bab.cpp src/cuts.cpp src/propagate.cpp src/conflict.cpp
  src/feasjump.cpp src/lns.cpp src/covers.cpp src/mir.cpp
  src/miqp.cpp src/minlp.cpp src/lattice_reform.cpp
  src/features.cpp src/sparse_sb.cpp src/tree_cuts.cpp
  src/prop_trail.cpp src/conflict_cut.cpp
  src/dynsep.cpp src/zerohalf.cpp src/flowcover.cpp
  src/lifted_branch.cpp src/planbb.cpp src/sc_milp_branch.cpp
  src/balans.cpp src/kernel_pump.cpp src/mrens.cpp
  src/mip_presolve.cpp src/gf2_presolve.cpp src/component_presolve.cpp
  src/implied_int.cpp src/symmetry.cpp
  src/l2sep.cpp src/hgtsm.cpp)

add_library(sor::search ALIAS sor_search)
target_include_directories(sor_search PUBLIC include)
target_link_libraries(sor_search PUBLIC sor_engines sor_io PRIVATE sor_flags)
