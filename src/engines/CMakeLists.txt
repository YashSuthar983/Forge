add_library(sor_engines
  src/pdhg.cpp
  src/simplex.cpp
  src/dual_simplex.cpp
  src/dual_edge_weights.cpp
  src/dual_cost_perturbation.cpp
  src/dual_ratio_test.cpp
  src/farkas.cpp
  src/hpr.cpp
  src/hpr_qp.cpp
  src/crossover.cpp
  src/lp.cpp
  src/qp.cpp
  src/qp_pdhcg.cpp
  src/nlp.cpp)
add_library(sor::engines ALIAS sor_engines)
target_include_directories(sor_engines PUBLIC include)
target_link_libraries(sor_engines
  PUBLIC  sor_model sor_backend sor_la_cpu sor_presolve
  PRIVATE sor_certify sor_flags)
