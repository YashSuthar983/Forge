add_executable(sor_solve sor_solve.cpp)
target_link_libraries(sor_solve PRIVATE
  sor_io sor_engines sor_search sor_certify sor_backend sor_flags)

add_executable(sor_gen sor_gen.cpp)
target_link_libraries(sor_gen PRIVATE sor_io sor_flags)

add_executable(sor_check sor_check.cpp)
target_link_libraries(sor_check PRIVATE sor_io sor_engines sor_certify sor_flags)
