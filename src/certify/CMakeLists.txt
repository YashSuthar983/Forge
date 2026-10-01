add_library(sor_certify src/finalize.cpp src/basis_certificate.cpp src/padic_solve.cpp)
add_library(sor::certify ALIAS sor_certify)
target_include_directories(sor_certify PUBLIC include)
target_link_libraries(sor_certify PUBLIC sor_core sor_model PRIVATE sor_la_cpu sor_flags)
