add_library(forge_engines
  src/pdhg.cpp
  src/simplex.cpp
  src/dual_simplex.cpp
  src/dual_edge_weights.cpp
  src/dual_cost_perturbation.cpp
  src/dual_ratio_test.cpp
  src/farkas.cpp
  src/hpr.cpp
  src/qp.cpp
  src/qp_pdhcg.cpp
  src/nlp.cpp)
add_library(forge::engines ALIAS forge_engines)
target_include_directories(forge_engines PUBLIC include)
target_link_libraries(forge_engines
  PUBLIC  forge_model forge_backend forge_la_cpu forge_presolve
  PRIVATE forge_flags)
