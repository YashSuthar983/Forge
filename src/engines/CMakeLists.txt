add_library(forge_engines
  src/pdhg.cpp
  src/simplex.cpp
  src/triangular_crash.cpp
  src/dual_certificate_repair.cpp
  src/dual_simplex.cpp
  src/network_detect.cpp
  src/dual_edge_weights.cpp
  src/dual_cost_perturbation.cpp
  src/dual_ratio_test.cpp
  src/farkas.cpp
  src/hpr.cpp
  src/hpr_qp.cpp
  src/qp_option_bindings.cpp
  src/lp_batched.cpp
  src/crossover.cpp
  src/lp_concurrent.cpp
  src/lp.cpp
  src/qp.cpp src/qcqp.cpp src/qcqp_local.cpp
  src/qp_pdhcg.cpp src/qp_ipm.cpp src/qp_polish.cpp src/qp_krylov.cpp src/qp_route.cpp
  src/nlp.cpp)
add_library(forge::engines ALIAS forge_engines)
target_include_directories(forge_engines PUBLIC include)
target_link_libraries(forge_engines
  PUBLIC  forge_model forge_backend forge_la_cpu forge_presolve
  PRIVATE forge_certify forge_flags)
